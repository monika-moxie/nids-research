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
