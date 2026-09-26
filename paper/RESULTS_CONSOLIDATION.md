# Results Consolidation

## Purpose

This document is the source of truth for the reported experimental results.
It keeps each number together with the condition that produced it so that the
paper does not compare unlike experiments or overstate preliminary findings.

## Reporting Rules

- Report the test split, sample size, seed or deterministic sampling rule, and
  relevant hyperparameters beside every result.
- Use the phrase "local-DP-style" for Phase 5 and Phase 6. No formal epsilon or
  delta privacy guarantee has been computed.
- State that all adversarial perturbations and smoothing radii are measured in
  the 194-dimensional preprocessed feature space. They are not guarantees of
  physically valid raw network-flow changes.
- Treat the bridge run as an initial integration experiment: its 10,000-row,
  two-round setting is smaller than the Phase 4/5 20,000-row, three-round runs.

## Table 1. Shared Centralized Baseline

Dataset: official UNSW-NB15 training and testing CSVs. Task: binary intrusion
detection. Model: MLP on 194 preprocessed features. Test size: 82,332 flows.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.8640 | 0.8093 | 0.9852 | 0.8886 | 0.9792 |

Confusion matrix (rows: true normal, true attack; columns: predicted normal,
predicted attack): `[[26475, 10525], [671, 44661]]`.

Interpretation: the detector catches 98.52% of attacks but generates false
alerts for 10,525 normal flows. This high-recall operating point is a useful
baseline for security experiments, but it is not a claim that false positives
are operationally negligible.

## Table 2. Adversarial Evaluation

Dataset: full official test set, 82,332 flows. Threat model: white-box attacks
on the preprocessed feature vector. Epsilon: 0.05. PGD: 10 steps, step size
0.01. Constrained numeric PGD changes only features named `num__` by the
preprocessor and leaves one-hot categorical features fixed.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.8640 | 0.8093 | 0.9852 | 0.8886 | 0.9792 |
| FGSM | 0.1695 | 0.2294 | 0.2155 | 0.2223 | 0.0707 |
| PGD | 0.0771 | 0.1212 | 0.1081 | 0.1143 | 0.0162 |
| Constrained numeric PGD | 0.7614 | 0.7300 | 0.8993 | 0.8058 | 0.8651 |

Interpretation: unconstrained gradient attacks severely break the MLP, while
the numeric-only constraint preserves substantially more utility. The
constrained attack is more credible for tabular NIDS because it does not create
fractional or contradictory one-hot category indicators. It remains an
approximation: perturbations occur after scaling, not through a raw-traffic
generation model.

## Table 3. Randomized Smoothing Certificate

Dataset: deterministic 500-flow test subset. Base model: centralized MLP.
Gaussian noise standard deviation: sigma = 0.25. Noisy samples per flow: 128.
Confidence level parameter: alpha = 0.001. Certificate: L2 radius in
preprocessed feature space using a conservative Hoeffding lower bound.

| Metric | Value |
| --- | ---: |
| Smoothed accuracy | 0.8400 |
| Coverage | 0.9720 |
| Covered accuracy | 0.8436 |
| Mean certified radius | 0.1360 |
| Median certified radius | 0.1446 |

| Certified radius threshold | Certified accuracy |
| ---: | ---: |
| 0.00 | 0.8200 |
| 0.05 | 0.7640 |
| 0.10 | 0.6520 |
| 0.20 | 0.1420 |

Interpretation: a larger required radius is a stronger condition, so certified
accuracy falls as the radius rises. The result is a local guarantee for the
smoothed classifier under the stated Gaussian-noise assumptions; it is not a
guarantee over all semantically valid network transformations.

## Table 4. FedAvg Utility Under Client Heterogeneity

Dataset: deterministic 20,000-row training subset and full official test set.
Configuration: 5 clients, 3 communication rounds, 1 local epoch per round.
IID clients use random shards; non-IID clients use a label-skew partition.

| Client distribution | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID | 0.8203 | 0.7680 | 0.9652 | 0.8554 | 0.9361 |
| Non-IID label skew | 0.8020 | 0.7667 | 0.9205 | 0.8366 | 0.8999 |

Interpretation: the non-IID split reduces F1 by 0.0188 and ROC-AUC by 0.0362
relative to IID training. This is consistent with client updates becoming less
aligned when clients have different class distributions.

## Table 5. Local-DP-Style Privacy-Utility Trade-off

Same federated configuration as Table 4. Before aggregation, every client
clips its model update to L2 norm 10.0 and adds Gaussian noise with standard
deviation `10.0 x noise_multiplier`. These are empirical utility results, not
a calibrated epsilon-delta DP guarantee.

| Noise multiplier | IID F1 | Non-IID F1 |
| ---: | ---: | ---: |
| 0.000 | 0.8554 | 0.8366 |
| 0.001 | 0.8520 | 0.8241 |
| 0.005 | 0.8138 | 0.7601 |
| 0.010 | 0.8036 | 0.7491 |

Interpretation: utility decreases with update noise, and the non-IID setting
is more sensitive. At noise multiplier 0.01, the F1 drop from the no-noise
case is 0.0518 for IID and 0.0875 for non-IID training.

## Table 6. Bridge: Robustness Under Federated and Private Training

Question: does the robustness profile change after non-IID FedAvg and
local-DP-style update noise? Bridge configuration: 10,000 training rows, 5
clients, 2 rounds, 1 local epoch, full 82,332-flow test set. Certification uses
a deterministic 200-flow subset, sigma = 0.25, 64 noisy samples per flow, and
alpha = 0.001. The private condition uses clip norm 10.0 and noise multiplier
0.005.

| Condition | Clean F1 | Smoothed accuracy | Coverage | Cert. acc. @ 0.05 | Cert. acc. @ 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized baseline | 0.8886 | 0.8450 | 0.9500 | 0.7200 | 0.4850 |
| FedAvg, non-IID | 0.7128 | 0.5550 | 1.0000 | 0.5550 | 0.5550 |
| FedAvg, non-IID + local-DP-style noise | 0.7151 | 0.5850 | 0.8100 | 0.4350 | 0.3500 |

Interpretation: the centralized model has substantially stronger clean F1. At
radius 0.10, the private federated condition has lower certified accuracy than
the non-private federated condition. The non-private federated value should not
be read as "better robustness than centralized" without qualification: its
clean F1 is much lower, its predictions are heavily attack-biased, and the
bridge training budget differs from the centralized baseline. The defensible
claim is that federated training and update noise alter the clean-utility and
certified-robustness profile.

## Figure Plan

1. Bar chart: F1 under clean, FGSM, PGD, and constrained numeric PGD.
2. Line chart: certified accuracy versus radius for centralized smoothing.
3. Two-line chart: IID and non-IID F1 versus noise multiplier.
4. Grouped bars: bridge clean F1 and certified accuracy at radius 0.10.

Each figure must cite the corresponding table and repeat the key experiment
condition in its caption.
