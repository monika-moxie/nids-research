# CI3201 Adversarial Robustness

## Purpose

This folder will hold adversarial machine learning experiments for the shared NIDS classifier.

## Why It Exists

NIDS models can be brittle when malicious traffic is slightly modified. This track studies how attacks can fool the classifier and how certified defenses can give measurable robustness guarantees.

## Planned Work

- FGSM attack
- PGD attack
- Realistic constrained attack for tabular NIDS features
- Randomized smoothing defense
- Certified accuracy at multiple perturbation radii

## Dependency Boundary

This folder should import the shared baseline pipeline instead of redefining dataset loading or preprocessing.
