"""Run randomized smoothing certificates for the shared NIDS baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ci3201_adversarial.run_attacks import (
    load_test_features,
    load_trained_baseline,
    maybe_subsample,
)
from ci3201_adversarial.smoothing import (
    Certificate,
    certified_accuracy_by_radius,
    certify_feature,
)
from shared.config import DatasetConfig


def summarize_certificates(certificates: list[Certificate], labels: np.ndarray, radii: list[float]):
    """Build aggregate smoothing metrics."""

    predictions = np.asarray([certificate.prediction for certificate in certificates])
    abstained = np.asarray([certificate.abstained for certificate in certificates])
    radii_values = np.asarray([certificate.certified_radius for certificate in certificates])

    non_abstained = ~abstained
    smoothed_accuracy = float(np.mean(predictions == labels))
    coverage = float(np.mean(non_abstained))
    covered_accuracy = (
        float(np.mean(predictions[non_abstained] == labels[non_abstained]))
        if np.any(non_abstained)
        else 0.0
    )

    return {
        "smoothed_accuracy": smoothed_accuracy,
        "coverage": coverage,
        "covered_accuracy": covered_accuracy,
        "mean_certified_radius": float(np.mean(radii_values)),
        "median_certified_radius": float(np.median(radii_values)),
        "certified_accuracy_by_radius": certified_accuracy_by_radius(certificates, labels, radii),
    }


def run_smoothing(
    data_dir: Path,
    baseline_dir: Path,
    output_path: Path,
    sample_size: int,
    sigma: float,
    num_noise_samples: int,
    batch_size: int,
    alpha: float,
    radii: list[float],
    random_state: int,
) -> dict[str, object]:
    """Run randomized smoothing on a deterministic test subset."""

    features, labels, _ = load_test_features(data_dir, baseline_dir)
    features, labels = maybe_subsample(features, labels, sample_size, random_state)
    model = load_trained_baseline(baseline_dir / "baseline_mlp.pt", features.shape[1])

    certificates = []
    for index, feature in enumerate(features):
        certificates.append(
            certify_feature(
                model=model,
                feature=feature,
                sigma=sigma,
                num_samples=num_noise_samples,
                batch_size=batch_size,
                alpha=alpha,
                seed=random_state + index,
            )
        )

    summary = summarize_certificates(certificates, labels, radii)
    results = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "defense": "randomized smoothing",
        "certificate_space": "L2 radius in preprocessed feature space",
        "sample_size": int(len(labels)),
        "sigma": sigma,
        "num_noise_samples": num_noise_samples,
        "alpha": alpha,
        "radii": radii,
        "summary": summary,
        "first_10_certificates": [
            {
                "label": int(label),
                "prediction": certificate.prediction,
                "top_class_probability": certificate.top_class_probability,
                "lower_confidence_bound": certificate.lower_confidence_bound,
                "certified_radius": certificate.certified_radius,
                "abstained": certificate.abstained,
            }
            for certificate, label in list(zip(certificates, labels))[:10]
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/shared-baseline"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/ci3201-smoothing/smoothing_metrics.json"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--num-noise-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_smoothing(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        sigma=args.sigma,
        num_noise_samples=args.num_noise_samples,
        batch_size=args.batch_size,
        alpha=args.alpha,
        radii=args.radii,
        random_state=args.random_state,
    )
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
