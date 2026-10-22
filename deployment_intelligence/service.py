"""Artifact-backed inference and certification for the deployment demonstrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from ci3201_adversarial.smoothing import certify_feature
from shared.config import TrainingConfig
from shared.model import build_mlp, require_torch


DEFAULT_ARTIFACT_DIR = Path("outputs/shared-baseline")
CERTIFICATE_SPACE = "L2 radius in preprocessed feature space"


class ArtifactBackedNidsService:
    """Load the saved baseline once and expose schema-safe flow analysis."""

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.preprocessor = None
        self.model = None
        self.input_fields: list[str] = []
        self.lime_explainer = None

    def load(self) -> None:
        """Load the exact preprocessing and model artifacts used by Phase 1."""

        preprocessor_path = self.artifact_dir / "preprocessor.joblib"
        model_path = self.artifact_dir / "baseline_mlp.pt"
        missing = [path for path in (preprocessor_path, model_path) if not path.exists()]
        if missing:
            expected = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing deployment artifact(s): {expected}")

        self.preprocessor = joblib.load(preprocessor_path)
        self.input_fields = list(self.preprocessor.feature_names_in_)
        torch, _ = require_torch()
        self.model = build_mlp(
            input_dim=len(self.preprocessor.get_feature_names_out()),
            hidden_dims=TrainingConfig().hidden_dims,
            dropout=TrainingConfig().dropout,
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def metadata(self) -> dict[str, object]:
        self._require_loaded()
        return {
            "model_input_fields": self.input_fields,
            "excluded_fields": ["id", "label", "attack_cat"],
            "preprocessed_feature_count": len(self.preprocessor.get_feature_names_out()),
            "certificate_space": CERTIFICATE_SPACE,
        }

    def predict(self, raw_features: dict[str, Any]) -> dict[str, object]:
        """Validate a raw flow, preprocess it, and return the MLP probability."""

        transformed = self._transform(raw_features)
        torch, _ = require_torch()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(transformed))
            probability = float(torch.sigmoid(logits).item())
        return self._prediction_payload(probability)

    def predict_batch(self, raw_flows: list[dict[str, Any]]) -> list[dict[str, object]]:
        """Score validated raw flows together for the analyst alert queue."""

        self._require_loaded()
        if not raw_flows:
            return []

        frame = pd.DataFrame([self._ordered_features(flow) for flow in raw_flows])
        transformed = self.preprocessor.transform(frame).astype("float32")
        torch, _ = require_torch()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(transformed))
            probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return [self._prediction_payload(float(probability)) for probability in probabilities]

    def analyze(
        self,
        raw_features: dict[str, Any],
        sigma: float = 0.25,
        num_noise_samples: int = 64,
        alpha: float = 0.001,
        seed: int = 42,
    ) -> dict[str, object]:
        """Return a clean prediction and one flow-specific smoothing certificate."""

        transformed = self._transform(raw_features)
        prediction = self.predict(raw_features)
        certificate = certify_feature(
            model=self.model,
            feature=transformed[0],
            sigma=sigma,
            num_samples=num_noise_samples,
            batch_size=min(256, num_noise_samples),
            alpha=alpha,
            seed=seed,
        )
        predicted_label = None if certificate.abstained else self._label_name(certificate.prediction)
        return {
            **prediction,
            "certification": {
                "smoothed_prediction": predicted_label,
                "top_class_probability": certificate.top_class_probability,
                "lower_confidence_bound": certificate.lower_confidence_bound,
                "certified_radius": certificate.certified_radius,
                "abstained": certificate.abstained,
                "sigma": sigma,
                "num_noise_samples": num_noise_samples,
                "certificate_space": CERTIFICATE_SPACE,
            },
        }

    def explain(self, raw_features: dict[str, Any], num_features: int = 5) -> dict[str, object]:
        """Return a LIME explanation in the model's preprocessed feature space."""

        transformed = self._transform(raw_features)
        explanation = self._get_lime_explainer().explain_instance(
            data_row=transformed[0],
            predict_fn=self._predict_probabilities,
            labels=(1,),
            num_features=num_features,
            num_samples=1000,
        )
        feature_names = self.preprocessor.get_feature_names_out()
        contributions = [
            {
                "feature": str(feature_names[feature_index]),
                "contribution": float(weight),
                "direction": "toward_attack" if weight >= 0 else "toward_normal",
            }
            for feature_index, weight in explanation.local_exp[1]
        ]
        return {
            "method": "LIME",
            "explanation_space": "preprocessed feature space",
            "top_features": contributions,
        }

    def _transform(self, raw_features: dict[str, Any]) -> np.ndarray:
        self._require_loaded()
        frame = pd.DataFrame([self._ordered_features(raw_features)])
        return self.preprocessor.transform(frame).astype("float32")

    def _ordered_features(self, raw_features: dict[str, Any]) -> dict[str, Any]:
        """Validate one raw flow and put fields in the training-time order."""

        supplied = set(raw_features)
        expected = set(self.input_fields)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise ValueError("Invalid flow schema; " + "; ".join(details))

        return {field: raw_features[field] for field in self.input_fields}

    def _get_lime_explainer(self) -> LimeTabularExplainer:
        if self.lime_explainer is not None:
            return self.lime_explainer

        from shared.config import DatasetConfig

        training_frame = pd.read_csv(DatasetConfig().train_path, usecols=self.input_fields)
        background = training_frame.sample(n=min(500, len(training_frame)), random_state=42)
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.preprocessor.transform(background).astype("float32"),
            feature_names=list(self.preprocessor.get_feature_names_out()),
            class_names=["normal", "attack"],
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )
        return self.lime_explainer

    def _predict_probabilities(self, transformed_features: np.ndarray) -> np.ndarray:
        self._require_loaded()
        torch, _ = require_torch()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(transformed_features.astype("float32")))
            attack_probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - attack_probabilities, attack_probabilities])

    def _require_loaded(self) -> None:
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("The NIDS artifacts are not loaded. Call load() before inference.")

    @staticmethod
    def _label_name(label: int) -> str:
        return "attack" if label == 1 else "normal"

    def _prediction_payload(self, probability: float) -> dict[str, object]:
        return {
            "predicted_label": self._label_name(int(probability >= 0.5)),
            "attack_probability": probability,
            "decision_threshold": 0.5,
        }
