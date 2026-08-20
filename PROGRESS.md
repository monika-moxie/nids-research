# Progress

## Current State

- Phase 1 is partially complete: dataset choice, preprocessing code, baseline model code, training CLI, and preprocessing smoke test are in place.
- UNSW-NB15 is selected as the shared dataset for binary intrusion detection.
- `data/raw/UNSW-NB15/` is now visible for placing official CSV files.
- A project-local `.venv` is recommended, but the first dependency install failed with `No space left on device`; the empty failed `.venv` was removed and should be recreated after freeing disk space.
- Baseline metrics code exists, but real metrics have not been generated because dependencies and the official CSV files are not present yet.
- The project is organized as a two-part academic NIDS research project with a later bridge experiment.
- Next session should finish Phase 1 by installing dependencies, adding the official UNSW-NB15 CSVs, running training, and recording baseline metrics.

## Phase Tracker

- [x] Phase 0 - Repo scaffolding
- [ ] Phase 1 - Shared pipeline: dataset choice, preprocessing, baseline classifier, baseline metrics
- [ ] Phase 2 - CI3201 attack suite: FGSM, PGD, realistic constrained attack
- [ ] Phase 3 - CI3201 certified defense: randomized smoothing and certified accuracy
- [ ] Phase 4 - CI3203 FedAvg simulation across simulated clients
- [ ] Phase 5 - CI3203 local differential privacy and privacy-utility tradeoff
- [ ] Phase 6 - Bridge experiment: certified robustness under FL and LDP
- [ ] Phase 7 - Results consolidation and IEEE paper skeleton

## Session Notes

### 2026-08-20 - Phase 0

Initialized the scaffold for `nids-research` and created purpose READMEs for each major project area.

### 2026-08-20 - Phase 1

Selected UNSW-NB15 for the shared binary NIDS baseline. Added reusable preprocessing, a PyTorch MLP baseline, a training/evaluation CLI, dependency list, data documentation, and a preprocessing smoke test.

### 2026-08-20 - Phase 1 environment setup

Added tracked placeholder folders so `data/raw/UNSW-NB15/` is visible while real dataset CSVs remain ignored by Git.

Attempted to create and install dependencies into `.venv`; installation failed with `No space left on device`, so the empty failed `.venv` was removed. Recreate it after freeing disk space.
