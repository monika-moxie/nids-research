"""Evaluate FGSM, PGD, and constrained PGD against the shared baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from ci3201_adversarial.attacks import (
    constrained_numeric_pgd_attack,
    fgsm_attack,
    numeric_feature_mask,
    pgd_attack,
)
from shared.config import DatasetConfig, TrainingConfig
from shared.data import load_unsw_nb15, split_features_and_target
from shared.model import build_mlp, require_torch
from shared.train_baseline import evaluate_binary, predict_probabilities


def load_trained_baseline(model_path: Path, input_dim: int):
    """Rebuild the baseline architecture and load its saved weights."""

    torch, _ = require_torch()
    model = build_mlp(
        input_dim=input_dim,
        hidden_dims=TrainingConfig().hidden_dims,
        dropout=TrainingConfig().dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_features(model, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Evaluate the baseline on clean or adversarial preprocessed features."""

    probabilities = predict_probabilities(model, features.astype("float32"))
    return evaluate_binary(labels.astype(int), probabilities)


def load_test_features(data_dir: Path, baseline_dir: Path):
    """Load raw test CSV rows and transform them with the saved preprocessor."""

    config = DatasetConfig(data_dir=data_dir)
    _, test_frame = load_unsw_nb15(config)
    x_test_raw, y_test = split_features_and_target(
        test_frame,
        config.target_column,
        config.leakage_columns,
    )

    preprocessor = joblib.load(baseline_dir / "preprocessor.joblib")
    features = preprocessor.transform(x_test_raw).astype("float32")
    labels = y_test.to_numpy().astype(int)
    feature_names = list(preprocessor.get_feature_names_out())
    return features, labels, feature_names


def maybe_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a deterministic subset for fast experiments when requested."""

    if sample_size <= 0 or sample_size >= len(labels):
        return features, labels

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(labels), size=sample_size, replace=False)
    return features[indices], labels[indices]


def run_attack_suite(
    data_dir: Path,
    baseline_dir: Path,
    output_path: Path,
    sample_size: int,
    epsilon: float,
    pgd_steps: int,
    pgd_step_size: float,
    random_state: int,
) -> dict[str, object]:
    """Run the Phase 2 attack suite and save metrics."""

    features, labels, feature_names = load_test_features(data_dir, baseline_dir)
    features, labels = maybe_subsample(features, labels, sample_size, random_state)

    model = load_trained_baseline(baseline_dir / "baseline_mlp.pt", features.shape[1])
    clean_metrics = evaluate_features(model, features, labels)

    fgsm_features = fgsm_attack(model, features, labels, epsilon)
    pgd_features = pgd_attack(model, features, labels, epsilon, pgd_step_size, pgd_steps)
    constrained_features = constrained_numeric_pgd_attack(
        model=model,
        features=features,
        labels=labels,
        numeric_mask=numeric_feature_mask(feature_names),
        epsilon=epsilon,
        step_size=pgd_step_size,
        steps=pgd_steps,
    )

    results = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "attack_space": "preprocessed feature space",
        "sample_size": int(len(labels)),
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "pgd_step_size": pgd_step_size,
        "metrics": {
            "clean": clean_metrics,
            "fgsm": evaluate_features(model, fgsm_features, labels),
            "pgd": evaluate_features(model, pgd_features, labels),
            "constrained_numeric_pgd": evaluate_features(model, constrained_features, labels),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/shared-baseline"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/ci3201-attacks/attack_metrics.json"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-step-size", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_attack_suite(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        pgd_step_size=args.pgd_step_size,
        random_state=args.random_state,
    )
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
