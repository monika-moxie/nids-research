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
