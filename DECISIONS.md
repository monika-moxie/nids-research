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

### 2026-08-20 - Phase 1 local environment

**Decision:** Use a project-local virtual environment named `.venv`.

**Why:** ML projects depend on specific library versions. A virtual environment keeps PyTorch, scikit-learn, pandas, and related packages isolated from system Python and from other coursework.

**Tradeoff:** The environment must be activated before running project commands, but this small habit prevents dependency conflicts later.

### 2026-08-20 - Phase 1 data folder tracking

**Decision:** Track empty data directories with `.gitkeep`, but ignore actual dataset CSV files.

**Why:** Git does not track empty folders, so placeholders make the expected dataset path visible. The real CSVs are large data artifacts and should stay local.

**Tradeoff:** Teammates must download or copy the dataset themselves, but the repository remains lightweight and GitHub-friendly.

### 2026-08-23 - Phase 1 baseline result

**Decision:** Treat the 20-epoch MLP trained on the official UNSW-NB15 train CSV as the first shared baseline for downstream attacks and defenses.

**Why:** The model has strong attack recall and ROC-AUC, giving Phase 2 a meaningful target. It is not perfect, which is useful: later robustness work should be evaluated on a realistic detector rather than a toy model.

**Tradeoff:** Precision is lower than recall, meaning the detector creates false positives. This is acceptable for the first baseline because NIDS systems often prefer catching attacks over suppressing alerts too aggressively.

### 2026-08-23 - Phase 2 attack space

**Decision:** Implement FGSM and PGD in the shared model's preprocessed 194-dimensional feature space.

**Why:** The trained MLP only sees preprocessed numeric tensors, not raw CSV rows. Attacking this feature space directly tests the actual decision boundary learned by the model.

**Tradeoff:** Fully unconstrained attacks can perturb one-hot categorical features, which may create unrealistic tabular records. To address this, Phase 2 also includes constrained numeric PGD.

### 2026-08-23 - Phase 2 constrained attack

**Decision:** Add a constrained PGD variant that only changes transformed numeric features whose preprocessor names begin with `num__`.

**Why:** Realistic tabular NIDS attacks should not freely turn categorical one-hot indicators into fractional or contradictory values. Restricting perturbations to numeric features makes the attack less powerful but more defensible.

**Tradeoff:** This is still an approximation because it operates after scaling rather than reconstructing valid raw network flows. It is a practical constrained baseline, not a perfect traffic generator.

### 2026-08-23 - Phase 2 evaluation size

**Decision:** Record Phase 2 attack metrics on the full official UNSW-NB15 test set.

**Why:** The earlier 5,000-flow sample was useful for fast development, but final attack claims should use the same full test set as the clean baseline.

**Tradeoff:** Full-test evaluation takes longer on CPU, but the resulting metrics are stronger for presentation and paper tables.
