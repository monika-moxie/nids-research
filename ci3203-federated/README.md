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
