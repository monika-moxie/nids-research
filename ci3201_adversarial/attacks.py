"""Gradient-based adversarial attacks for preprocessed tabular NIDS features."""

from __future__ import annotations

import numpy as np


def _feature_tensor(torch, features: np.ndarray):
    tensor = torch.as_tensor(features, dtype=torch.float32).clone().detach()
    tensor.requires_grad_(True)
    return tensor


def _label_tensor(torch, labels: np.ndarray):
    return torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1)


def fgsm_attack(model, features: np.ndarray, labels: np.ndarray, epsilon: float) -> np.ndarray:
    """Create adversarial examples with the Fast Gradient Sign Method.

    FGSM asks: which direction in input space increases the model loss fastest?
    It then takes one bounded step in that direction.
    """

    import torch
    from torch import nn

    model.eval()
    x = _feature_tensor(torch, features)
    y = _label_tensor(torch, labels)
    loss = nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()

    adversarial = x + epsilon * x.grad.sign()
    return adversarial.detach().cpu().numpy()


def pgd_attack(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    step_size: float,
    steps: int,
) -> np.ndarray:
    """Create adversarial examples with projected gradient descent.

    PGD is a stronger iterative version of FGSM. After each step, it clips the
    total perturbation back into the allowed L-infinity epsilon budget.
    """

    import torch
    from torch import nn

    model.eval()
    original = torch.as_tensor(features, dtype=torch.float32)
    y = _label_tensor(torch, labels)
    adversarial = original.clone().detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = nn.BCEWithLogitsLoss()(model(adversarial), y)
        loss.backward()

        with torch.no_grad():
            adversarial = adversarial + step_size * adversarial.grad.sign()
            perturbation = torch.clamp(adversarial - original, min=-epsilon, max=epsilon)
            adversarial = original + perturbation

        adversarial = adversarial.detach()

    return adversarial.cpu().numpy()


def constrained_numeric_pgd_attack(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    numeric_mask: np.ndarray,
    epsilon: float,
    step_size: float,
    steps: int,
) -> np.ndarray:
    """Run PGD while changing only transformed numeric columns.

    The shared preprocessor produces numeric columns with names beginning
    `num__` and one-hot categorical columns with names beginning `cat__`.
    This attack leaves categorical one-hot indicators untouched.
    """

    import torch
    from torch import nn

    model.eval()
    original = torch.as_tensor(features, dtype=torch.float32)
    y = _label_tensor(torch, labels)
    mask = torch.as_tensor(numeric_mask.astype("float32")).reshape(1, -1)
    adversarial = original.clone().detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = nn.BCEWithLogitsLoss()(model(adversarial), y)
        loss.backward()

        with torch.no_grad():
            masked_step = adversarial.grad.sign() * mask
            adversarial = adversarial + step_size * masked_step
            perturbation = torch.clamp(adversarial - original, min=-epsilon, max=epsilon)
            adversarial = original + perturbation * mask

        adversarial = adversarial.detach()

    return adversarial.cpu().numpy()


def numeric_feature_mask(feature_names: list[str]) -> np.ndarray:
    """Mark transformed numeric features produced by the shared preprocessor."""

    return np.asarray([name.startswith("num__") for name in feature_names], dtype=bool)
