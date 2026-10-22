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

### 2026-10-21 - Phase 8 batch triage boundary

**Decision:** Add a CSV batch-screening queue to the Phase 8 demonstrator, but run randomized smoothing and LIME only after an analyst selects an individual flagged flow.

**Why:** Batch scoring reflects the operational workflow of screening many flows, while certificate sampling and local explanations are expensive and meaningful at an individual-flow level. Applying detailed explanations blindly to every uploaded row would make the interface slow and could imply that all flows have been equally reviewed.

**How:** `POST /batch-score` accepts a bounded CSV, validates its model-input schema, scores rows with the saved clean MLP in one vectorized batch, and returns rows at or above the fixed 0.50 attack threshold. The dashboard retains each flagged row's validated features and invokes the existing `/intelligence` endpoint only when an analyst selects that row.

**Tradeoff:** The queue does not certify or explain every row upfront. This is intentional: it separates fast screening from slower evidence generation and keeps the demonstrator responsive on a laptop.

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

### 2026-08-25 - Phase 5 local-DP-style mechanism

**Decision:** Implement client-side update clipping followed by Gaussian noise before server aggregation.

**Why:** Federated learning keeps raw data local, but model updates can still leak information. Clipping limits the maximum influence of one client update, and noise hides fine-grained update details before the server receives them.

**Tradeoff:** This demonstrates the privacy-utility tradeoff but does not compute a formal differential privacy epsilon. We should describe it as local-DP-style update perturbation unless a formal accountant is added later.

### 2026-08-25 - Phase 5 calibrated noise levels

**Decision:** Use clip norm 10.0 with noise multipliers 0.0, 0.001, 0.005, and 0.01 for the reported Phase 5 curve.

**Why:** Larger noise multipliers caused numerical instability in the small MLP, while aggressive clipping with clip norm 1.0 overly suppressed useful updates. The calibrated values produce a readable utility curve.

**Tradeoff:** The noise values are small, so the result is best framed as a practical tradeoff demonstration rather than a strong privacy claim.

### 2026-09-26 - Phase 7 reporting boundaries

**Decision:** Consolidate results into paper tables while preserving each experiment's evaluation scope and explicitly separating observations from claims.

**Why:** The experiments use different sample sizes and training budgets. A table that omits these details could imply unfair comparisons, particularly for the smaller bridge run. The project also needs to avoid overstating a preprocessed-space certificate as raw-flow robustness or noisy updates as a formal local-DP guarantee.

**Tradeoff:** The resulting paper language is more cautious, but it is scientifically stronger and easier to defend in a viva.

### 2026-09-26 - Phase 7 reproducible figures

**Decision:** Generate paper figures programmatically from saved JSON metrics rather than manually editing chart values.

**Why:** Reproducible figures preserve traceability from a reported visual back to the exact experiment output. The bridge figure deliberately places clean F1 beside certified accuracy so a certificate cannot be misread as a complete model-quality score.

**Tradeoff:** This adds Matplotlib as a dependency and a small maintenance script, but it makes chart regeneration and review substantially more reliable.

### 2026-09-26 - Phase 7 research positioning

**Decision:** Frame the work as a reproducible NIDS integration and trade-off study, not as a claim to invent adversarial attacks, randomized smoothing, FedAvg, or differential privacy.

**Why:** These are established methods with identifiable primary sources. The defensible contribution is the shared experimental pipeline, constrained tabular attack comparison, scoped certificates, federated/privacy utility analysis, and initial bridge evaluation.

**Tradeoff:** This narrows novelty language, but it prevents unsupported "first" claims and makes the final paper more credible.

### 2026-09-26 - Phase 7 repeated-seed reporting

**Decision:** Repeat the key FedAvg, local-update-noise, and bridge experiments across seeds 42, 1337, and 2026, and retain numerically failed high-noise runs in the reported evidence.

**Why:** A single seed cannot distinguish a stable effect from a random outcome. Excluding a failed run would systematically overstate the reliability of a privacy-noise setting.

**Tradeoff:** Three seeds are a CPU-feasible stability check rather than a large statistical study. The high-noise `0.01` result has only two successful runs and must be described as unstable.

### 2026-09-26 - Phase 8 artifact-backed API

**Decision:** Build the deployment API around the exact saved Phase 1 preprocessor and MLP artifacts, and reject missing or unknown raw feature fields.

**Why:** A practical demo must make the same prediction pipeline as the research baseline. Silent field mismatch would create results that are neither reproducible nor defensible.

**Tradeoff:** A client must supply all 42 required raw fields, which is stricter than a toy form, but the frontend can obtain this schema from the metadata endpoint and construct valid requests.

### 2026-10-15 - Phase 8 bounded intelligence workflow

**Decision:** Use LIME as a local explanation tool and a deterministic escalate/suppress policy rather than an unconstrained language-model agent.

**Why:** A practical analyst demonstration needs traceable recommendations. The policy can point directly to model probability and explanation evidence, while the incident report remains a human-review draft.

**Tradeoff:** The workflow is less open-ended than a conversational autonomous agent, but it is safer, reproducible, and defensible for a cybersecurity setting.

### 2026-10-16 - Phase 8 dashboard state behavior

**Decision:** Explicitly set the empty analysis state to `display: none` when it has the HTML `hidden` attribute.

**Why:** The component's grid display styling overrode the browser's default hidden behavior, causing the empty state to remain visible after a valid result was available.

**Tradeoff:** None for application behavior; this restores the intended one-state-at-a-time analyst workflow.
