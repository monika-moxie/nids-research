# Progress

## Current State

- Phase 1 is partially complete: dataset choice, preprocessing code, baseline model code, training CLI, and preprocessing smoke test are in place.
- UNSW-NB15 is selected as the shared dataset for binary intrusion detection.
- Baseline metrics code exists, but real metrics have not been generated because the official CSV files and PyTorch dependency are not installed locally yet.
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
