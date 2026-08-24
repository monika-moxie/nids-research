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
