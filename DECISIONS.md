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

### 2026-08-23 - Phase 3 smoothing certificate

**Decision:** Implement randomized smoothing certificates in the baseline model's preprocessed feature space using Gaussian noise and a conservative Hoeffding lower bound.

**Why:** The baseline model operates on preprocessed tensors, so certification must be computed in that same representation. Hoeffding's inequality gives a simple valid lower confidence bound without adding new dependencies.

**Tradeoff:** The certificate is conservative and applies to L2 perturbations in preprocessed feature space, not to raw network packet validity.

### 2026-08-23 - Phase 3 evaluation size

**Decision:** Record initial smoothing metrics on a deterministic 500-flow sample with 128 noisy samples per flow.

**Why:** Randomized smoothing is computationally expensive because every original flow requires many noisy forward passes. A deterministic sample gives a reproducible first certified-defense result on CPU.

**Tradeoff:** This is enough for a first Phase 3 result, but a larger run would be better for final paper-quality tables if time allows.

### 2026-08-24 - Phase 4 FedAvg simulation design

**Decision:** Simulate FedAvg locally using 5 clients, 3 communication rounds, 1 local epoch per round, and a deterministic 20,000-row training subset.

**Why:** This setup is small enough to run on CPU while still showing the central federated-learning mechanism: local client training followed by server-side weighted averaging.

**Tradeoff:** It is not a production-scale federation. The purpose is an explainable academic simulation that can later be scaled if needed.

### 2026-08-24 - Phase 4 IID and non-IID split

**Decision:** Compare random IID client partitioning with a simple label-skew non-IID partition.

**Why:** IID tests the ideal case where clients resemble the global distribution. Label-skew non-IID tests the more realistic case where some clients see mostly normal traffic while others see mostly attacks.

**Tradeoff:** Label-skew by sorting labels is intentionally simple and easy to defend, but it is only one kind of non-IID distribution.
