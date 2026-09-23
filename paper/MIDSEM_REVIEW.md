# Mid-Sem Review Packet

## Project Title

Robust and Privacy-Aware Network Intrusion Detection using Adversarial Evaluation, Certified Defense, Federated Learning, and Local Update Privacy

## One-Minute Summary

This project studies a binary Network Intrusion Detection System (NIDS) on the UNSW-NB15 dataset. It is organized as two related academic tracks:

- **CI3201 adversarial robustness:** evaluate whether a trained NIDS model can be fooled by adversarial perturbations, then apply randomized smoothing as a certified defense.
- **CI3203 federated learning and privacy:** simulate collaborative NIDS training with FedAvg, then add local-DP-style noise to client updates.

The bridge experiment, planned next, asks whether certified robustness survives when the model is trained under federated and privacy-preserving conditions.

## Why This Project Matters

NIDS models are commonly evaluated using clean test accuracy, but real security deployment is harder:

- Attackers may manipulate traffic features to evade detection.
- Network logs may be distributed across organizations or devices.
- Raw traffic data may be too sensitive to centralize.
- Federated model updates may still leak information.

This project therefore studies accuracy, adversarial robustness, certified robustness, federated learning, and local update privacy together rather than treating NIDS as a simple classification task.

## Dataset and Task

Dataset: **UNSW-NB15**

Task: binary intrusion detection

```text
0 = normal traffic
1 = attack traffic
```

Why UNSW-NB15:

- public and academically used
- has predefined train/test CSV files
- includes modern attack categories
- practical size for solo experimentation

Important preprocessing decisions:

- Dropped `id` because it is only a row identifier.
- Dropped `attack_cat` because it describes attack type and would leak label information.
- Numeric features: median imputation plus standard scaling.
- Categorical features: most-frequent imputation plus one-hot encoding.

## Phase 1: Shared Baseline

### What Was Done

Trained a multilayer perceptron (MLP) classifier on preprocessed UNSW-NB15 features.

### Why

All later experiments need the same baseline. Without a shared baseline, attack, defense, federated, and privacy results would not be comparable.

### How

The MLP receives 194 preprocessed features and predicts attack probability. A threshold of 0.5 converts probability into normal versus attack.

### Results

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8640 |
| Precision | 0.8093 |
| Recall | 0.9852 |
| F1 | 0.8886 |
| ROC-AUC | 0.9792 |

Confusion matrix:

```text
[[26475, 10525],
 [  671, 44661]]
```

Interpretation: the model catches most attacks, but creates some false positives. This is realistic for intrusion detection, where high recall is often prioritized.

## Phase 2: Adversarial Attack Suite

### What Was Done

Implemented and evaluated:

- FGSM
- PGD
- constrained numeric PGD

### Why

Clean accuracy does not prove security. We need to test whether small, adversarially chosen input changes can fool the detector.

### How

FGSM and PGD use gradients of binary cross-entropy loss with respect to the input feature vector.

FGSM:

```text
x_adv = x + epsilon * sign(gradient)
```

PGD repeats smaller gradient steps and projects the perturbation back into the allowed epsilon budget.

Constrained numeric PGD only changes transformed numeric features and keeps one-hot categorical features fixed.

### Results

| Setting | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| Clean | 0.8640 | 0.8886 | 0.9792 |
| FGSM | 0.1695 | 0.2223 | 0.0707 |
| PGD | 0.0771 | 0.1143 | 0.0162 |
| Constrained numeric PGD | 0.7614 | 0.8058 | 0.8651 |

Interpretation: the model is highly vulnerable to unconstrained gradient attacks. The constrained numeric attack is more realistic and less destructive, but still reduces performance.

## Phase 3: Certified Defense

### What Was Done

Implemented randomized smoothing and reported certified accuracy at multiple radii.

### Why

Attack testing is empirical. Randomized smoothing gives a measurable robustness certificate in local feature space.

### How

For each input `x`, we sample noisy copies:

```text
x + Gaussian noise
```

The model predicts all noisy copies. If one class wins strongly, a certified L2 radius is estimated:

```text
certified_radius = sigma * Phi^-1(p_lower)
```

where `p_lower` is a conservative lower confidence bound on the winning class probability.

### Results

Settings:

```text
sample size = 500 flows
sigma = 0.25
noise samples per flow = 128
alpha = 0.001
```

| Radius | Certified Accuracy |
| ---: | ---: |
| 0.00 | 0.8200 |
| 0.05 | 0.7640 |
| 0.10 | 0.6520 |
| 0.20 | 0.1420 |

Additional:

```text
Smoothed accuracy = 0.8400
Coverage = 0.9720
Mean certified radius = 0.1360
```

Important limitation: this is an L2 certificate in preprocessed feature space, not a guarantee over every possible raw network flow.

## Phase 4: Federated Learning with FedAvg

### What Was Done

Simulated federated training with:

- IID client partitioning
- non-IID label-skew partitioning
- weighted FedAvg aggregation

### Why

Real NIDS data may be distributed across clients that cannot share raw traffic logs.

### How

Each communication round:

1. Server sends global model to clients.
2. Clients train locally.
3. Clients return model weights.
4. Server averages weights:

```text
w_global = sum_k (n_k / total_examples) * w_client_k
```

### Results

Settings:

```text
5 clients
3 communication rounds
1 local epoch per round
20,000 training rows
```

| Setting | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| IID | 0.8203 | 0.8554 | 0.9361 |
| Non-IID label skew | 0.8020 | 0.8366 | 0.8999 |

Interpretation: FedAvg trains a useful detector. Non-IID label skew performs worse because local client updates are less aligned.

## Phase 5: Local-DP-Style Update Privacy

### What Was Done

Added clipping and Gaussian noise to client updates before server aggregation.

### Why

Federated learning keeps raw data local, but client updates can still leak information. Local update perturbation reduces this risk.

### How

Each client sends:

```text
update = local_weights - global_weights
clipped_update = clip_to_l2_norm(update, clip_norm)
private_update = clipped_update + Gaussian_noise
```

The server averages private updates.

### Results

Settings:

```text
5 clients
3 communication rounds
1 local epoch per round
20,000 training rows
clip norm = 10.0
```

| Noise Multiplier | IID F1 | Non-IID F1 |
| ---: | ---: | ---: |
| 0.000 | 0.8554 | 0.8366 |
| 0.001 | 0.8520 | 0.8241 |
| 0.005 | 0.8138 | 0.7601 |
| 0.010 | 0.8036 | 0.7491 |

Interpretation: increasing noise reduces model utility. Non-IID clients are more fragile under privacy noise.

Important limitation: this demonstrates local-DP-style update perturbation but does not compute a formal epsilon privacy budget.

## Current Research Gap Addressed

Many NIDS works evaluate only clean accuracy. Some study adversarial attacks. Some study federated learning. Fewer connect adversarial robustness, certified defense, federated learning, and privacy noise in one pipeline.

This project addresses that gap by building one shared NIDS baseline and studying:

- clean detection performance
- adversarial vulnerability
- certified robustness
- federated training under IID and non-IID data
- local update privacy and utility tradeoff
- planned bridge between certified robustness and FL plus privacy

## What Is Remaining

### Phase 6: Bridge Experiment

Question:

```text
Does certified robustness survive FedAvg and local-DP-style noise?
```

Planned method:

- compare centralized baseline smoothing
- compare FedAvg model smoothing
- compare FedAvg plus LDP model smoothing
- report clean utility and certified accuracy side by side

### Phase 7: Paper Skeleton and Result Consolidation

Planned contents:

- IEEE-style paper outline
- methods section
- results tables
- limitations
- conclusion

### Phase 8: Deployment Intelligence Layer

Optional practical demo, not core research.

## Strong Review Narrative

The project is currently beyond a basic classifier. We have built a shared NIDS baseline, shown that it is vulnerable to adversarial attacks, added a certified robustness method, simulated federated learning under IID and non-IID data, and added privacy noise to client updates. The remaining research step is to connect the two tracks and test whether certified robustness survives federated and privacy-preserving training.

## Likely Review Questions and Answers

### Why did you choose UNSW-NB15?

It is public, citable, has predefined train/test splits, and is more modern than older datasets like KDDCup99. It is also manageable for a solo project.

### Why did you drop `attack_cat`?

Because this is binary intrusion detection using `label`. `attack_cat` describes the attack family and would leak label-like information into the model.

### Why use an MLP?

The data is tabular flow data. An MLP is a simple deep baseline that can learn nonlinear feature interactions while remaining explainable enough for analysis.

### Why are FGSM and PGD used?

They are standard gradient-based adversarial attacks. FGSM is a fast one-step attack, while PGD is a stronger iterative attack.

### Why include constrained numeric PGD?

Unconstrained attacks can unrealistically modify one-hot categorical features. Numeric-only PGD is a more realistic tabular attack baseline.

### What does randomized smoothing certify?

It certifies local prediction stability within an L2 radius in preprocessed feature space.

### Why does non-IID FedAvg perform worse?

Because clients see biased local data distributions, so their updates are less aligned with the global objective.

### Why add privacy noise if FedAvg already keeps data local?

Because updates can still leak information about local data. Privacy noise reduces that leakage risk.

### What is the biggest limitation so far?

The privacy mechanism does not compute a formal epsilon, and randomized smoothing certificates are in preprocessed feature space rather than raw network-flow validity.

### What is the next most important experiment?

The bridge experiment: measuring whether certified robustness survives under FedAvg and local-DP-style update noise.

## Suggested Review Slide Flow

1. Problem: NIDS under attacks, distributed data, and privacy constraints
2. Dataset and baseline
3. CI3201: adversarial attacks
4. CI3201: certified defense
5. CI3203: FedAvg simulation
6. CI3203: local-DP privacy tradeoff
7. Key results table
8. Limitations
9. Next work: bridge experiment
