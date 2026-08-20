# Decisions

## Decision Log

### 2026-08-20 - Phase 0 scaffold layout

**Decision:** Use a modular top-level structure with separate directories for shared pipeline work, CI3201 adversarial robustness, CI3203 federated learning, the bridge experiment, and the paper.

**Why:** The research project has two course-facing parts plus one integration question. Keeping these boundaries visible makes the implementation easier to defend, test, and assign to presenters.

**Tradeoff:** This creates a few more folders up front, but it prevents later experiments from becoming tangled in one large script directory.

### 2026-08-20 - Phase 1 dataset choice

**Decision:** Use UNSW-NB15 as the shared baseline dataset for binary network intrusion detection.

**Why:** UNSW-NB15 is official, citable, tabular, and provides predefined train/test CSV files. It is large enough for meaningful ML experiments but manageable for a solo laptop workflow.

**Tradeoff:** CIC-IDS2017 and CICIoT2023 are also strong datasets, but they are larger and more operationally complex. UNSW-NB15 gives us a cleaner first baseline for adversarial, federated, and privacy extensions.

### 2026-08-20 - Phase 1 leakage control

**Decision:** Drop `id` and `attack_cat` from binary classifier inputs.

**Why:** `id` is an arbitrary row identifier, and `attack_cat` describes the attack class. Keeping either would make the classifier look better without learning real traffic behavior.

**Tradeoff:** Dropping `attack_cat` means the Phase 1 baseline is binary detection rather than multiclass attack classification. This is simpler and better aligned with later robustness and FL experiments.
