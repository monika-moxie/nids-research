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

## Phase 1 Dataset

The shared baseline uses UNSW-NB15 for binary intrusion detection. The target is `label`, where `0` means normal traffic and `1` means attack traffic.

Why this matters: every later robustness, federated learning, privacy, and bridge experiment needs the same starting model. Otherwise, changes in accuracy could come from inconsistent preprocessing rather than from the method being studied.

Technical mechanism: `shared/data.py` fits preprocessing on the training CSV only, then applies the same fitted transformation to the test CSV. Numeric features are median-imputed and standardized. Categorical features are most-frequent-imputed and one-hot encoded. Leakage columns such as `id` and `attack_cat` are removed before training.

## Expected Data Layout

Download the official UNSW-NB15 training and testing CSV files and place them here:

```text
data/raw/UNSW-NB15/UNSW_NB15_training-set.csv
data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv
```

The `data/` folder is intentionally ignored by Git because datasets can be large.

## Run the Baseline

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train and evaluate:

```powershell
python -m shared.train_baseline --epochs 20
```

Expected outputs:

```text
outputs/shared-baseline/baseline_mlp.pt
outputs/shared-baseline/preprocessor.joblib
outputs/shared-baseline/metrics.json
```

## Verify the Preprocessing Contract

Run the smoke test:

```powershell
python -m unittest tests.test_shared_preprocessing
```
