# IEEE Paper Skeleton

## Working Title

Robust and Privacy-Aware Network Intrusion Detection: Adversarial Evaluation,
Randomized Smoothing, Federated Learning, and Local Update Perturbation

## Abstract Draft

Network intrusion detection systems must remain useful when traffic features
are manipulated, data is distributed across clients, and model updates require
privacy protection. This work studies binary intrusion detection on UNSW-NB15
through a shared multilayer-perceptron pipeline. We evaluate white-box FGSM,
PGD, and numeric-constrained PGD attacks; apply randomized smoothing to obtain
local L2 certificates in preprocessed feature space; simulate FedAvg under IID
and label-skewed non-IID client partitions; and perturb clipped client updates
with Gaussian noise before aggregation. The centralized baseline obtains F1
0.8886, while PGD reduces F1 to 0.1143 and constrained numeric PGD to 0.8058.
Under smoothing, certified accuracy is 0.6520 at radius 0.10 on a deterministic
500-flow subset. FedAvg F1 decreases from 0.8554 under IID partitions to 0.8366
under label skew, while increasing update noise further reduces utility. An
initial bridge experiment shows that federated training and local-DP-style
noise change the clean-utility and certified-robustness profile. The reported
privacy mechanism is not claimed to provide a formal epsilon guarantee, and
all certificates are limited to preprocessed feature space.

Keywords: network intrusion detection, adversarial machine learning,
randomized smoothing, federated learning, local differential privacy.

## I. Introduction

### Problem

Conventional NIDS evaluation emphasizes clean classification accuracy, but a
deployable detector also faces evasion attempts, decentralized traffic logs,
and confidentiality constraints. These pressures are often evaluated in
isolation, leaving uncertainty about their combined effect.

### Research Gap

This project builds one reproducible pipeline that connects: (1) adversarial
vulnerability, (2) certified local robustness, (3) FedAvg under client
heterogeneity, (4) local-DP-style update perturbation, and (5) an initial
integration experiment across these settings.

### Objectives

1. Establish a leakage-controlled centralized UNSW-NB15 NIDS baseline.
2. Measure vulnerability to unconstrained and numeric-constrained gradient
   attacks.
3. Estimate certified accuracy under randomized smoothing at several radii.
4. Compare IID and label-skewed non-IID FedAvg utility.
5. Measure the empirical utility cost of clipped, noised client updates.
6. Examine how federated and private training alter the certified robustness
   profile.

### Contributions

State these conservatively in the final paper:

1. A shared, modular UNSW-NB15 experimental pipeline for two related NIDS
   research tracks.
2. A side-by-side empirical comparison of unconstrained and numeric-only
   adversarial attacks against the same MLP detector.
3. A randomized-smoothing certificate analysis explicitly scoped to the
   preprocessed tabular feature representation.
4. An IID/non-IID FedAvg and local-update-noise utility study.
5. A bridge experiment that reports clean utility and certified performance
   together rather than treating them as interchangeable.

## II. Related Work

Use this section only after adding and verifying primary sources. Organize it
into four short subsections: adversarial attacks on tabular/NIDS models,
certified robustness and randomized smoothing, federated NIDS under non-IID
data, and differential privacy for federated learning. End the section by
stating precisely that this work integrates these concerns in one pipeline; do
not claim it is the first work to do so without a systematic literature search.

## III. Materials and Methods

### A. Dataset and Task

Use UNSW-NB15 predefined training and test CSV files. Define `label = 0` as
normal and `label = 1` as attack. Exclude `id` and `attack_cat` from model
inputs: the first is an arbitrary identifier and the second would leak
attack-family information into a binary target task.

### B. Preprocessing and Centralized MLP

Fit preprocessing on training data only. Numeric fields use median imputation
and standard scaling; categorical fields use most-frequent imputation and
one-hot encoding. The resulting 194-dimensional vector is passed to the MLP.
Report the architecture, loss, optimizer, epochs, seed, and 0.5 prediction
threshold from the saved baseline metrics.

### C. Adversarial Threat Model

For an input `x` and loss `L`, FGSM creates `x_adv = x + epsilon sign(grad_x
L)`. PGD repeats smaller gradient steps and projects the result into the
epsilon-bounded neighborhood. Numeric-constrained PGD applies the update only
to transformed numeric features. Clearly state epsilon 0.05, 10 PGD steps,
and step size 0.01.

### D. Randomized Smoothing

Define the smoothed classifier as majority voting over base-model predictions
for `x + N(0, sigma^2 I)`. Estimate a lower confidence bound for the winning
class probability with Hoeffding's inequality; for the binary case, a positive
certificate uses `R = sigma Phi^-1(p_lower)` when `p_lower > 0.5`. Explain
that `R` is an L2 radius in preprocessed feature space.

### E. Federated Learning and Local Update Perturbation

Each client trains a local model, then FedAvg computes `w = sum_k (n_k/N) w_k`.
For update perturbation, calculate `delta_k = w_k - w`, clip its overall L2
norm, and add Gaussian noise before aggregation. Describe this honestly as
local-DP-style update perturbation because no epsilon-delta accountant is
implemented.

### F. Bridge Protocol

Compare centralized baseline, non-IID FedAvg, and non-IID FedAvg with local
update noise. Report clean metrics and certificate metrics together. Keep the
bridge's 10,000-row/two-round budget visible because it differs from the main
FedAvg experiment.

## IV. Experimental Results

Use the six tables in `RESULTS_CONSOLIDATION.md`:

1. Centralized baseline utility.
2. Attack robustness.
3. Randomized smoothing certificates.
4. IID versus non-IID FedAvg.
5. Local-DP-style privacy-utility trade-off.
6. Bridge clean utility and certified robustness.

Suggested result narrative:

- The clean baseline is strong in recall and ROC-AUC but has meaningful false
  positives.
- PGD is the strongest tested unconstrained attack; numeric constraints reduce
  attack power and increase realism.
- Certification decreases with requested radius, as expected.
- Non-IID data and update noise both reduce federated utility.
- In the bridge, clean utility and certificate values must be interpreted
  jointly; certified accuracy alone cannot establish the best model.

## V. Discussion

Discuss the trade-offs rather than presenting every result as an improvement.
The core story is that a NIDS can be accurate on clean data yet vulnerable to
white-box perturbations; smoothing provides a local guarantee at a utility
cost; and distributed/privacy-aware training changes both utility and the
robustness profile. The constrained attack is important because it narrows the
gap between mathematical feature-space attacks and plausible tabular changes.

## VI. Limitations and Threats to Validity

1. UNSW-NB15 is one benchmark dataset; results may not transfer to live,
   evolving enterprise traffic.
2. The attack and certificate spaces are preprocessed features, not a full
   semantics-preserving raw-traffic manipulation space.
3. Smoothing uses 500 flows and 128 noisy samples; the bridge uses 200 flows
   and 64 noisy samples. Larger repeated runs are needed for final claims.
4. The FedAvg and bridge runs are CPU-friendly simulations, not deployment-
   scale federations.
5. Update noise has no formal privacy accountant, so it cannot be reported as
   a quantified local-DP guarantee.
6. Results currently use initial deterministic runs; repeated seeds and
   uncertainty intervals should be added before strong comparative claims.

## VII. Conclusion and Future Work

Conclude only what the current evidence supports: the shared NIDS baseline is
highly vulnerable to unconstrained gradient attacks; numeric constraints reduce
but do not remove vulnerability; smoothing provides local feature-space
certificates; and non-IID federation plus update noise expose a utility-
robustness trade-off. Future work should add repeated seeds, larger
certification samples, a formal privacy accountant, semantically valid traffic
constraints, and the separately scoped deployment intelligence layer.

## Appendix Checklist

- Hyperparameter table and hardware/runtime information.
- Data acquisition instructions; do not include dataset CSVs in Git.
- Exact command lines for every runner.
- Seed and deterministic sampling policy.
- Full confusion matrices where space permits.
- Code repository URL and commit hash chosen by the project owner.
