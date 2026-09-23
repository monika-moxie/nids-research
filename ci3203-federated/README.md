# CI3203 Federated Learning and Privacy

## Purpose

This folder will hold federated learning and local differential privacy experiments for the shared NIDS classifier.

## Why It Exists

Real NIDS data may be distributed across organizations or devices that cannot directly share raw traffic logs. Federated learning studies collaborative training without centralizing data, while local differential privacy studies what happens when clients add privacy-preserving noise before sharing updates.

## Planned Work

- FedAvg simulation
- IID client partitioning
- Non-IID client partitioning
- Local differential privacy on client updates
- Privacy-utility tradeoff curve

## Dependency Boundary

This folder should reuse the shared preprocessing, model definition, and metrics so federated results are comparable to the central baseline.

## Phase 4 FedAvg Simulation

The runnable Python package is named `ci3203_federated` because Python imports cannot use hyphens. This folder remains the course-facing documentation area for CI3203.

### What It Does

The FedAvg simulation trains a shared NIDS detector across simulated clients:

- IID split: clients receive random, balanced-ish shards.
- non-IID label-skew split: clients receive shards sorted by label, so some clients are more normal-heavy and others more attack-heavy.
- FedAvg aggregation: client model weights are averaged by client dataset size.

### Why It Exists

Real NIDS data may be distributed across organizations, routers, edge devices, or departments. Federated learning tests whether those clients can collaborate without centralizing raw traffic records.

### How It Works Technically

Each communication round follows this loop:

1. The server sends the current global model to each client.
2. Each client trains locally on its own shard.
3. Clients return model weights, not raw data.
4. The server computes a weighted average:

```text
w_global = sum_k (n_k / total_examples) * w_client_k
```

where `n_k` is the number of examples owned by client `k`.

### Run

```powershell
.\.venv\Scripts\python.exe -m ci3203_federated.run_fedavg
```

Output:

```text
outputs/ci3203-fedavg/fedavg_metrics.json
```

### Initial Results

Using 5 clients, 3 communication rounds, 1 local epoch per round, and a deterministic 20,000-row training subset:

| Setting | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| IID | 0.8203 | 0.8554 | 0.9361 |
| Non-IID label skew | 0.8020 | 0.8366 | 0.8999 |

Interpretation: both settings learn useful detectors, but non-IID label skew performs worse because clients train on biased local views of the traffic distribution.

## Phase 5 Local Differential Privacy

Phase 5 adds privacy noise to client updates before the server sees them.

### What It Does

Each client still trains locally, but instead of sending the raw model update directly, it sends a clipped and noised update:

```text
update = local_weights - global_weights
clipped_update = clip_to_l2_norm(update, clip_norm)
private_update = clipped_update + Gaussian_noise
```

The server averages these private updates.

### Why It Exists

Federated learning keeps raw traffic records local, but model updates can still leak information about a client's data distribution. Local differential privacy reduces that risk because the update is perturbed before it leaves the client.

### How It Works Technically

Clipping limits how much any one client update can influence the global model. Noise then hides fine-grained information in the update. The experiment varies the `noise_multiplier` to show the privacy-utility tradeoff:

- low noise: better model utility, weaker privacy protection
- high noise: stronger privacy protection, worse model utility

This implementation demonstrates local-DP-style update perturbation but does not compute a formal privacy budget epsilon.

### Run

```powershell
.\.venv\Scripts\python.exe -m ci3203_federated.run_ldp
```

Output:

```text
outputs/ci3203-ldp/ldp_metrics.json
```

### Initial Privacy-Utility Results

Using 5 clients, 3 communication rounds, 1 local epoch per round, a 20,000-row training subset, and clip norm `10.0`:

| Noise multiplier | IID F1 | Non-IID F1 |
| ---: | ---: | ---: |
| 0.000 | 0.8554 | 0.8366 |
| 0.001 | 0.8520 | 0.8241 |
| 0.005 | 0.8138 | 0.7601 |
| 0.010 | 0.8036 | 0.7491 |

Interpretation: increasing noise generally reduces detection utility. The non-IID setting degrades more sharply because biased client updates are already harder to combine, and privacy noise makes the signal even less stable.
