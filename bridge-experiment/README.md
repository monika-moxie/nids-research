# Bridge Experiment

## Purpose

This folder will hold the integration experiment connecting certified robustness, federated learning, and local differential privacy.

## Why It Exists

The bridge question asks whether robustness guarantees from the adversarial track still survive when the model is trained in a federated setting and client updates are perturbed for privacy.

## Planned Work

- Compare centralized certified robustness against federated certified robustness
- Add local differential privacy noise and measure utility loss
- Report whether robustness, privacy, and accuracy can coexist under the chosen settings

## Dependency Boundary

This folder should orchestrate outputs from `shared`, `ci3201-adversarial`, and `ci3203-federated` rather than duplicating their internals.
