# Learning Log

## Notes

### 2026-08-20 - Phase 0: Why scaffolding matters

**Why we're doing it:**
A research codebase is not only a place to store code. It is also evidence of how the experiment was designed. A clear scaffold helps us explain what each component is responsible for and why comparisons are fair.

**How it works technically:**
The scaffold uses top-level directories as module boundaries. Future phases will add scripts, configs, notebooks, tests, and result artifacts inside the folder that matches the research question being answered.

### 2026-08-20 - Phase 1: Shared baseline pipeline

**Why we're doing it:**  
The baseline is the scientific control. Attacks, defenses, federated training, and privacy noise only mean something if they are applied to the same dataset split, preprocessing logic, and model family.

**How it works technically:**  
UNSW-NB15 is treated as a binary classification problem using the `label` column. The preprocessor fits only on training data to avoid test leakage. Numeric columns use median imputation plus standard scaling. Categorical columns use most-frequent imputation plus one-hot encoding with unknown test categories ignored. The baseline classifier is a multilayer perceptron trained with binary cross-entropy on logits.

**Viva defense point:**
Dropping `attack_cat` is necessary because it is not a network measurement; it is already a label-like explanation of the attack type. A model trained with it would be learning from the answer key.

### 2026-08-20 - Phase 1: Environment and data folders

**Why we're doing it:**  
The virtual environment makes results easier to reproduce because dependencies are installed for this project only. The visible data folder removes ambiguity about where the UNSW-NB15 CSVs must go.

**How it works technically:**  
`.venv` will contain an isolated Python interpreter and site-packages directory once recreated. `.gitignore` excludes `.venv`, generated outputs, and real dataset files. Tiny `.gitkeep` files are tracked so GitHub shows `data/raw/UNSW-NB15/` even before the CSVs are added locally.

**Practical note:**  
The first dependency install attempt failed with `No space left on device`. This is not a code bug; it means the laptop needs more free disk space before installing large ML packages such as PyTorch.

### 2026-08-23 - Phase 1: Reading the baseline metrics

**Why we're doing it:**  
Baseline metrics tell us whether the shared classifier is strong enough to attack and defend. If the clean model were weak, adversarial results would be less meaningful because attacks could be exploiting a bad classifier rather than a real robustness problem.

**How it works technically:**  
The model outputs an attack probability for each test flow. A threshold of 0.5 converts that probability into normal versus attack. Accuracy measures total correctness, precision measures how many alerts are truly attacks, recall measures how many attacks are caught, F1 balances precision and recall, and ROC-AUC measures ranking quality across thresholds.

**Result interpretation:**  
The baseline has high recall, 0.9852, so it catches most attacks. Precision is 0.8093, so some normal flows are falsely flagged. This is a realistic NIDS posture: sensitive detection with alert noise.

### 2026-08-23 - Phase 2: Adversarial attacks

**Why we're doing it:**  
Clean accuracy does not tell us whether a detector is robust. An attacker may make small changes to traffic features to push a malicious flow across the model's decision boundary. Phase 2 measures this brittleness directly.

**How FGSM works technically:**  
FGSM computes the gradient of the binary cross-entropy loss with respect to the input feature vector. The sign of that gradient tells us whether increasing or decreasing each feature would increase the model's mistake. The attack takes one step: `x_adv = x + epsilon * sign(gradient)`.

**How PGD works technically:**  
PGD repeats the same idea for multiple smaller steps. After each step, it clips the total perturbation so no feature moves more than epsilon from its original value. This usually makes PGD stronger than FGSM because it keeps rechecking the gradient after each move.

**How constrained numeric PGD works technically:**  
The preprocessor names numeric features with `num__` and categorical one-hot features with `cat__`. The constrained attack builds a Boolean mask and multiplies the gradient step by that mask. Numeric features can move; categorical one-hot features stay fixed.

**Result interpretation:**  
On the full 82,332-row test set, clean F1 was 0.8886. FGSM reduced F1 to 0.2223, and PGD reduced F1 to 0.1143, showing strong vulnerability under unconstrained gradient attacks. Constrained numeric PGD reduced F1 to 0.8058, showing a smaller but more realistic robustness drop.

### 2026-08-23 - Phase 3: Randomized smoothing

**Why we're doing it:**  
Attacks show that the model can fail, but a defense needs more than hope. Randomized smoothing gives a measurable statement about local stability: if noisy copies of an input mostly predict the same class, the smoothed classifier can be certified within a radius.

**How it works technically:**  
For each input vector `x`, the smoothed classifier samples many noisy versions `x + N(0, sigma^2 I)`. The base model predicts each noisy version. The majority class becomes the smoothed prediction. A lower confidence bound on the majority probability is computed with Hoeffding's inequality. If that lower bound is greater than 0.5, the binary certified radius is `sigma * Phi^-1(p_lower)`.

**Result interpretation:**  
With sigma 0.25 and 128 noisy samples on 500 flows, smoothed accuracy was 0.8400. Certified accuracy decreased as the radius grew: 0.8200 at radius 0.0, 0.7640 at 0.05, 0.6520 at 0.10, and 0.1420 at 0.20. This is expected because larger radii demand stronger guarantees.

**Viva defense point:**  
This is a certificate in L2 preprocessed feature space. It should not be oversold as a guarantee over every possible raw network flow transformation.

### 2026-08-24 - Phase 4: FedAvg simulation

**Why we're doing it:**  
Federated learning models the case where NIDS data is distributed across clients that cannot share raw traffic logs. This could represent organizations, departments, gateways, or edge devices. Phase 4 tests whether those clients can train a shared detector by exchanging model updates instead of raw data.

**How it works technically:**  
The server initializes a global MLP. In each communication round, every client receives the global model, trains locally on its own shard, and returns model weights. The server computes a weighted average of client weights: `w_global = sum_k (n_k / total) * w_k`, where `n_k` is the number of examples held by client `k`.

**IID versus non-IID:**  
The IID split randomly divides examples, so each client roughly resembles the full dataset. The non-IID split sorts by label before partitioning, creating clients with biased local distributions. In the run, one non-IID client had only normal flows and three clients had only attack flows.

**Result interpretation:**  
With 5 clients and 3 rounds on 20,000 training rows, IID FedAvg reached F1 0.8554 and ROC-AUC 0.9361. Non-IID label-skew FedAvg reached F1 0.8366 and ROC-AUC 0.8999. The non-IID drop is expected because local updates are less aligned when clients see different label distributions.

### 2026-08-25 - Phase 5: Local differential privacy on updates

**Why we're doing it:**  
FedAvg avoids sharing raw traffic logs, but client model updates can still reveal information about local data. Phase 5 adds client-side update perturbation to study how privacy protection affects model utility.

**How it works technically:**  
Each client trains locally and produces a local model. The client update is `local_weights - global_weights`. We clip the whole update to a maximum L2 norm, then add Gaussian noise with standard deviation `clip_norm * noise_multiplier`. The server averages these private updates instead of raw local weights.

**Privacy-utility tradeoff:**  
With clip norm 10.0, IID F1 moved from 0.8554 at no noise to 0.8520, 0.8138, and 0.8036 as noise multipliers increased to 0.001, 0.005, and 0.01. Non-IID F1 moved from 0.8366 to 0.8241, 0.7601, and 0.7491. This shows the expected pattern: stronger perturbation reduces utility, especially under non-IID data.

**Viva defense point:**  
This implementation is local-DP-style because noise is added before updates leave the client. However, we do not yet compute a formal epsilon privacy budget, so we should not overclaim formal DP guarantees.

### 2026-09-23 - Mid-sem review framing

**Why we're doing it:**  
A review is not only a code checkpoint. It tests whether the work has a coherent research story, whether the results answer the objectives, and whether limitations are understood honestly.

**How it works technically:**  
The review packet organizes the project into a baseline, adversarial track, certified defense, federated track, privacy extension, and bridge plan. It also maps each result table to the method that produced it, so the project can be defended without jumping through source files.

### 2026-09-23 - Phase 6: Bridge experiment

**Why we're doing it:**  
The adversarial and federated/privacy tracks answer different questions. The bridge experiment asks whether they can coexist: does a model still have certified robustness after non-IID federated training and local-DP-style update noise?

**How it works technically:**  
The bridge runner compares three model conditions on the same test distribution: centralized baseline, non-IID FedAvg, and non-IID FedAvg with clipped/noised updates. For each condition, it reports clean metrics on the official test set and randomized-smoothing certified accuracy on the same deterministic certification subset.

**Result interpretation:**  
The centralized baseline kept the strongest clean F1 at 0.8886. The small bridge FedAvg and FedAvg+LDP models had lower clean F1, around 0.7128 and 0.7151. At radius 0.10, certified accuracy was 0.4850 for centralized, 0.5550 for FedAvg, and 0.3500 for FedAvg+LDP. The result suggests privacy noise can reduce certified stability, but the FedAvg model also behaves differently because the bridge run is smaller than the full Phase 4 setup.

**Viva defense point:**  
This is an initial integration experiment, not the final word. The fair claim is that FL and privacy noise alter the robustness profile; larger runs should be used before making strong general conclusions.

### 2026-09-26 - Phase 7: Results consolidation and paper skeleton

**Why we're doing it:**  
Research code becomes a research contribution only when results can be traced to a method, compared fairly, and interpreted within their limits. Phase 7 turns separate scripts and JSON outputs into a coherent argument rather than a collection of metrics.

**How it works technically:**  
`paper/RESULTS_CONSOLIDATION.md` stores every reported table next to its data subset, model/training configuration, attack or noise setting, and interpretation. `paper/IEEE_PAPER_SKELETON.md` maps those tables into standard paper sections: problem, gap, methods, results, discussion, limitations, and conclusion.

**Viva defense point:**  
Certified accuracy and clean F1 measure different properties. A model with a larger certificate value but much lower clean utility is not automatically the better detector. We therefore report both values and preserve the different bridge training budget in the table caption and methods.

### 2026-09-26 - Phase 7: Reproducible figures

**Why we're doing it:**
Visuals make the central trade-offs immediately visible to a reviewer: vulnerability under attacks, declining certified accuracy as robustness demand increases, utility loss under update noise, and the need to read bridge certificate values together with clean performance.

**How it works technically:**
`paper/generate_figures.py` loads JSON artifacts from `outputs/`, extracts the published F1 and certified-accuracy values, and saves four PNGs in `paper/figures/`. It uses Matplotlib's non-interactive `Agg` backend, so the command works on a machine without a graphical Python window.

**Viva defense point:**
The charts do not generate new experimental evidence. They are a reproducible presentation of already saved metrics; rerunning the script after a result changes prevents transcription mistakes.

### 2026-09-26 - Phase 7: Related work and research gap

**Why we're doing it:**
The literature review explains why the project matters beyond a collection of scripts. It shows what previous researchers established and identifies the narrower question our pipeline investigates: how clean utility, adversarial vulnerability, certification, federation, and update noise interact in one NIDS setting.

**How it works technically:**
`paper/RELATED_WORK.md` uses numbered IEEE-style citations and maps them to structured records in `paper/references.bib`. The draft groups sources by dataset, adversarial robustness, certified robustness, federated learning, update privacy, and federated NIDS. This organization lets each paper section cite the source that supports its specific technical claim.

**Viva defense point:**
Our contribution is not a new theorem or a new federated algorithm. It is a controlled integration study with explicit limits: preprocessed feature-space attacks/certificates, a CPU-scale federation, and local-DP-style noise without a formal privacy accountant.
