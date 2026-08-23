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

## Phase 2 Attack Suite

The runnable Python package is named `ci3201_adversarial` because Python imports cannot use hyphens. This folder remains the course-facing documentation area for CI3201.

### What It Does

The attack suite evaluates the trained shared baseline under adversarial perturbations:

- FGSM: one-step gradient attack
- PGD: stronger multi-step gradient attack
- constrained numeric PGD: tabular-aware attack that leaves one-hot categorical features unchanged

### Why It Exists

Clean test metrics show normal model performance, but adversarial robustness asks whether small input changes can flip predictions. For NIDS, this matters because an attacker may slightly modify traffic behavior to evade detection.

### How It Works Technically

The trained model receives 194 preprocessed features. Each attack computes the gradient of binary cross-entropy loss with respect to the input features. The gradient says which feature direction would make the model more wrong.

FGSM takes one step:

```text
x_adv = x + epsilon * sign(gradient)
```

PGD repeats smaller steps and clips the final perturbation so every feature stays within the allowed epsilon budget.

Constrained numeric PGD applies the same PGD idea only to features whose preprocessor names begin with `num__`; categorical one-hot features beginning with `cat__` remain fixed.

### Run

```powershell
.\.venv\Scripts\python.exe -m ci3201_adversarial.run_attacks
```

Output:

```text
outputs/ci3201-attacks/attack_metrics.json
```

### Initial Results

Using the full 82,332-row UNSW-NB15 test set with epsilon `0.05`, PGD steps `10`, and PGD step size `0.01`:

| Setting | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| Clean | 0.8640 | 0.8886 | 0.9792 |
| FGSM | 0.1695 | 0.2223 | 0.0707 |
| PGD | 0.0771 | 0.1143 | 0.0162 |
| Constrained numeric PGD | 0.7614 | 0.8058 | 0.8651 |

Interpretation: the baseline is very brittle under unconstrained gradient attacks. The constrained numeric attack is less destructive, but it still reduces performance compared with clean evaluation.
