"""Randomized smoothing utilities for certified NIDS robustness."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Certificate:
    """Certified prediction summary for one input."""

    prediction: int
    top_class_probability: float
    lower_confidence_bound: float
    certified_radius: float
    abstained: bool


def hoeffding_lower_bound(successes: int, trials: int, alpha: float) -> float:
    """Conservative lower bound for a Bernoulli probability."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    estimate = successes / trials
    margin = sqrt(log(1.0 / alpha) / (2.0 * trials))
    return max(0.0, estimate - margin)


def certified_radius_from_probability(probability_lower_bound: float, sigma: float) -> float:
    """Convert a binary-class probability lower bound into an L2 certificate."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if probability_lower_bound <= 0.5:
        return 0.0
    capped_probability = min(probability_lower_bound, 1.0 - 1e-12)
    return sigma * NormalDist().inv_cdf(capped_probability)


def predict_noisy_votes(
    model,
    feature: np.ndarray,
    sigma: float,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Return binary vote counts after Gaussian perturbations around one input."""

    import torch

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    votes = np.zeros(2, dtype=int)
    remaining = num_samples
    model.eval()

    while remaining > 0:
        active_batch = min(batch_size, remaining)
        noise = rng.normal(loc=0.0, scale=sigma, size=(active_batch, feature.shape[0]))
        noisy_features = feature.reshape(1, -1) + noise

        with torch.no_grad():
            logits = model(torch.as_tensor(noisy_features, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            predictions = (probabilities >= 0.5).astype(int)

        votes += np.bincount(predictions, minlength=2)
        remaining -= active_batch

    return votes


def certify_feature(
    model,
    feature: np.ndarray,
    sigma: float,
    num_samples: int,
    batch_size: int,
    alpha: float,
    seed: int,
) -> Certificate:
    """Certify one preprocessed feature vector using randomized smoothing."""

    votes = predict_noisy_votes(
        model=model,
        feature=feature,
        sigma=sigma,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
    )
    prediction = int(np.argmax(votes))
    top_votes = int(votes[prediction])
    probability_lower = hoeffding_lower_bound(top_votes, num_samples, alpha)
    radius = certified_radius_from_probability(probability_lower, sigma)
    abstained = probability_lower <= 0.5

    return Certificate(
        prediction=prediction,
        top_class_probability=top_votes / num_samples,
        lower_confidence_bound=probability_lower,
        certified_radius=radius,
        abstained=abstained,
    )


def certified_accuracy_by_radius(
    certificates: list[Certificate],
    labels: np.ndarray,
    radii: list[float],
) -> dict[str, float]:
    """Compute certified accuracy at each requested radius."""

    if len(certificates) != len(labels):
        raise ValueError("certificates and labels must have the same length")

    total = len(labels)
    results = {}
    for radius in radii:
        correct_and_certified = sum(
            (not certificate.abstained)
            and certificate.prediction == int(label)
            and certificate.certified_radius >= radius
            for certificate, label in zip(certificates, labels)
        )
        results[str(radius)] = correct_and_certified / total

    return results
