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

## Phase 6 Bridge Experiment

The runnable Python package is named `bridge_experiment` because Python imports cannot use hyphens. This folder remains the documentation area for the bridge experiment.

### What It Does

The bridge experiment compares certified robustness across three model conditions:

- centralized baseline
- non-IID FedAvg
- non-IID FedAvg with local-DP-style update noise

### Why It Exists

The earlier phases answer separate questions:

- CI3201: can the NIDS model resist or certify against adversarial perturbations?
- CI3203: can the NIDS model be trained with federated learning and privacy noise?

The bridge asks whether these goals coexist. A model may be accurate, federated, or private, but still lose certified robustness.

### How It Works Technically

The bridge runner:

1. Loads the centralized Phase 1 baseline.
2. Trains a non-IID FedAvg model.
3. Trains a non-IID FedAvg model with local-DP-style clipped/noised updates.
4. Evaluates clean metrics on the official test set.
5. Runs randomized smoothing on the same deterministic test subset for all three models.
6. Reports clean F1 and certified accuracy by radius.

The certificate remains an L2 certificate in preprocessed feature space.

### Run

```powershell
.\.venv\Scripts\python.exe -m bridge_experiment.run_bridge
```

Output:

```text
outputs/bridge-experiment/bridge_metrics.json
```

### Initial Results

Default CPU-friendly settings:

```text
train sample size = 10,000
certification sample size = 200
clients = 5
federated rounds = 2
local epochs = 1
sigma = 0.25
noise samples per flow = 64
LDP clip norm = 10.0
LDP noise multiplier = 0.005
```

| Condition | Clean F1 | Certified Acc @ 0.00 | Certified Acc @ 0.05 | Certified Acc @ 0.10 | Certified Acc @ 0.20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized baseline | 0.8886 | 0.8100 | 0.7200 | 0.4850 | 0.0000 |
| FedAvg non-IID | 0.7128 | 0.5550 | 0.5550 | 0.5550 | 0.0000 |
| FedAvg non-IID + LDP | 0.7151 | 0.4950 | 0.4350 | 0.3500 | 0.0000 |

Interpretation: the centralized baseline keeps the best clean performance. The bridge FedAvg models are weaker because this run is intentionally CPU-friendly, but the comparison still shows that federated/private training changes the certified robustness profile. FedAvg + LDP has lower certified accuracy than FedAvg at the tested radii, suggesting privacy noise can reduce local stability.

Important limitation: this is an initial bridge run. Larger training samples, more rounds, and larger certification subsets should be used before making final paper claims.
