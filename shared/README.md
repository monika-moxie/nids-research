# Shared Pipeline

## Purpose

This folder will hold the common NIDS pipeline used by every experiment.

## Why It Exists

The adversarial robustness experiments and federated learning experiments must start from the same baseline classifier and preprocessing logic. Otherwise, differences in results could come from inconsistent data handling rather than the research variable being tested.

## What Belongs Here

- Dataset selection notes and download instructions
- Preprocessing and feature engineering
- Train, validation, and test splits
- Baseline deep classifier
- Baseline metrics and evaluation utilities

## What Does Not Belong Here

- Adversarial attack implementations
- Certified defense experiments
- Federated learning simulations
- Local differential privacy mechanisms
