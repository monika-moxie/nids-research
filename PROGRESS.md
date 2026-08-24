# Progress

## Current State

- Phase 1 is complete enough for downstream work: dataset choice, preprocessing code, baseline model code, training CLI, preprocessing smoke test, and real baseline metrics are in place.
- UNSW-NB15 is selected as the shared dataset for binary intrusion detection.
- `data/raw/UNSW-NB15/` contains the official training and testing CSV files.
- The active project copy is now `D:\SEM5_Project`, with a working `.venv`.
- Baseline metrics have been generated at `outputs/shared-baseline/metrics.json`: accuracy 0.8640, precision 0.8093, recall 0.9852, F1 0.8886, ROC-AUC 0.9792.
- Phase 2 attack code and full-test-set attack metrics are in place for FGSM, PGD, and constrained numeric PGD.
- Phase 3 randomized smoothing code and initial certified accuracy metrics are in place on a deterministic 500-flow sample.
- The project is organized as a two-part academic NIDS research project with a later bridge experiment.
- Next session should begin Phase 4: FedAvg simulation across simulated clients, unless Phase 3 needs a larger smoothing run for final tables.

## Phase Tracker

- [x] Phase 0 - Repo scaffolding
- [x] Phase 1 - Shared pipeline: dataset choice, preprocessing, baseline classifier, baseline metrics
- [x] Phase 2 - CI3201 attack suite: FGSM, PGD, realistic constrained attack
- [x] Phase 3 - CI3201 certified defense: randomized smoothing and certified accuracy
- [ ] Phase 4 - CI3203 FedAvg simulation across simulated clients
- [ ] Phase 5 - CI3203 local differential privacy and privacy-utility tradeoff
- [ ] Phase 6 - Bridge experiment: certified robustness under FL and LDP
- [ ] Phase 7 - Results consolidation and IEEE paper skeleton
- [ ] Phase 8 - Deployment Intelligence Layer: practical-applicability demo on top of FastAPI deployment, clearly separated from the core research contribution

## Session Notes

### 2026-08-20 - Phase 0

Initialized the scaffold for `nids-research` and created purpose READMEs for each major project area.

### 2026-08-20 - Phase 1

Selected UNSW-NB15 for the shared binary NIDS baseline. Added reusable preprocessing, a PyTorch MLP baseline, a training/evaluation CLI, dependency list, data documentation, and a preprocessing smoke test.

### 2026-08-20 - Phase 1 environment setup

Added tracked placeholder folders so `data/raw/UNSW-NB15/` is visible while real dataset CSVs remain ignored by Git.

Attempted to create and install dependencies into `.venv`; installation failed with `No space left on device`, so the empty failed `.venv` was removed. Recreate it after freeing disk space.

### 2026-08-23 - Phase 8 planning note

Recorded Phase 8 as a planned post-Phase-7 practical-applicability demo. It will add a clearly separated Deployment Intelligence Layer on top of the FastAPI deployment that can call the model, certified-robustness confidence, and SHAP/LIME explainer as tools; decide escalate versus suppress for flagged flows; convert explanation output into analyst-readable plain English; and draft a short incident report. Phase 8 is not started and is not part of the core research contribution.

### 2026-08-23 - Phase 1 baseline metrics

Confirmed the project has moved to `D:\SEM5_Project`, found the UNSW-NB15 CSVs and working `.venv`, then trained the shared MLP baseline for 20 epochs. Test metrics: accuracy 0.8640, precision 0.8093, recall 0.9852, F1 0.8886, ROC-AUC 0.9792, confusion matrix `[[26475, 10525], [671, 44661]]`.

### 2026-08-23 - Phase 2 attack suite

Added an importable `ci3201_adversarial` package with FGSM, PGD, and constrained numeric PGD attacks, plus smoke tests for perturbation-budget behavior. Ran the attack suite on the full 82,332-row UNSW-NB15 test set with epsilon 0.05, PGD steps 10, and PGD step size 0.01. Full-test F1 values: clean 0.8886, FGSM 0.2223, PGD 0.1143, constrained numeric PGD 0.8058.

### 2026-08-23 - Phase 3 randomized smoothing

Added randomized smoothing certification utilities and a smoothing runner. Ran an initial deterministic 500-flow certification experiment with sigma 0.25, 128 noisy samples per flow, alpha 0.001, and radii 0.0, 0.05, 0.10, and 0.20. Smoothed accuracy was 0.8400, coverage was 0.9720, mean certified radius was 0.1360, and certified accuracy by radius was `{0.0: 0.8200, 0.05: 0.7640, 0.10: 0.6520, 0.20: 0.1420}`.
