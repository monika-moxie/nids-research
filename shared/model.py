"""Deep tabular baseline model for NIDS binary classification."""

from __future__ import annotations


def require_torch():
    """Import PyTorch lazily so preprocessing tests can run without it installed."""

    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for the deep baseline. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torch, nn


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
    """Build a small multilayer perceptron for binary tabular classification."""

    _, nn = require_torch()
    layers = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ]
        )
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, 1))
    return nn.Sequential(*layers)
