# Handoff

## Handoff Packets

### 2026-08-20 - Phase 0: Repository Scaffold

**What it does in plain language:**  
Creates the project map. Each folder has a clear research responsibility so future code, results, and explanation packets have a stable home.

**Why it exists:**  
The project combines NIDS classification, adversarial ML, certified robustness, federated learning, local differential privacy, and a bridge experiment. Without a scaffold, the work can become hard to explain and harder to defend.

**Likely teacher questions and confident answers:**

1. **Why split `shared` from the course-specific folders?**  
   `shared` contains the common NIDS dataset, preprocessing, model, and baseline metrics. Both CI3201 and CI3203 depend on the same baseline, so separating it avoids duplicate pipelines and keeps comparisons fair.

2. **Why have a separate `bridge-experiment` folder?**  
   The bridge question is not only adversarial ML and not only federated learning. It asks whether certified robustness still holds after federated training and privacy noise, so it deserves its own experiment boundary.

3. **Why create documentation before experiments?**  
   The project must be explainable and resumable across sessions. Early documentation records the research intent before implementation details obscure it.

4. **Why include a `paper` folder now?**  
   The paper will later need methods, tables, plots, and an IEEE-style structure. Reserving that space from the start keeps results aligned with the final deliverable.

**Analogy:**  
This scaffold is like a lab notebook with labeled benches: one bench for the common dataset/model, one for attacks, one for federated training, one for combined stress testing, and one for writing the final report.

### 2026-08-20 - Phase 1: Shared Baseline Pipeline

**What it does in plain language:**  
Chooses UNSW-NB15 as the common NIDS dataset and adds the first reusable machine learning pipeline: clean the data, train a deep binary classifier, and report standard detection metrics.

**Why it exists:**  
Every later phase depends on a fair baseline. The adversarial attacks need a model to attack, randomized smoothing needs a model to certify, federated learning needs a model to distribute, and the bridge experiment needs the same baseline for comparison.

**Likely teacher questions and confident answers:**

1. **Why did you choose UNSW-NB15?**  
   It is official, citable, tabular, and has predefined training and testing CSV files. That makes the baseline reproducible and manageable for a solo project.

2. **Why drop `attack_cat`?**  
   Because the task is binary intrusion detection using `label`. `attack_cat` tells us the attack family, so using it as an input would leak label information and inflate performance.

3. **Why fit preprocessing only on the training data?**  
   The test set must simulate unseen future traffic. If we learn scaling or category information from the test set, the evaluation is no longer clean.

4. **Why use a multilayer perceptron?**  
   UNSW-NB15 is tabular flow data. A small MLP is a defensible deep baseline because it can learn nonlinear feature interactions while remaining simple enough to explain.

**Analogy:**  
The shared baseline is the calibrated measuring instrument for the whole project. Before testing attacks, defenses, or privacy noise, we first make sure everyone is measuring from the same ruler.

### 2026-08-20 - Phase 1: Environment and Data Placement

**What it does in plain language:**  
Creates a private Python environment for the project and makes the expected dataset folder visible.

**Why it exists:**  
The baseline cannot be trained until dependencies and CSVs are in the right place. The environment keeps dependencies controlled, and the data folder tells every teammate exactly where the dataset goes.

**Likely teacher questions and confident answers:**

1. **Why use a virtual environment?**  
   It isolates this project's packages, so results do not depend on unrelated Python packages installed elsewhere on the laptop.

2. **Why is the `data` folder in GitHub but not the CSV files?**  
   The folder path is part of the project contract, but datasets are large artifacts. We track placeholders and ignore real CSVs.

3. **Where exactly do the UNSW-NB15 files go?**  
   They go in `data/raw/UNSW-NB15/` with the official filenames `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.

4. **Why not commit the dataset for convenience?**  
   Large datasets bloat Git history and may have redistribution rules. Keeping them local is cleaner and more professional.

**Analogy:**  
The virtual environment is the project toolbox, and the data folder is the labeled shelf where the raw materials go.

**Current blocker:**  
The data folder exists, but dependencies are not installed. The first `pip install -r requirements.txt` attempt failed because the device ran out of space, and the empty failed `.venv` was removed.

### 2026-08-23 - Phase 1: Baseline Metrics

**What it does in plain language:**  
Trains the first real NIDS baseline on UNSW-NB15 and records how well it detects attacks on the official test set.

**Why it exists:**  
The attack and defense phases need a fixed baseline model. These metrics are the clean-performance reference point before adversarial perturbations, certified smoothing, federated training, or privacy noise are introduced.

**Likely teacher questions and confident answers:**

1. **What is the baseline performance?**  
   Accuracy is 0.8640, F1 is 0.8886, and ROC-AUC is 0.9792 on the official UNSW-NB15 test CSV.

2. **Why is recall higher than precision?**  
   The model catches nearly all attacks, but it also flags some normal traffic. For intrusion detection, high recall is often prioritized because missed attacks can be more costly than false alerts.

3. **What does the confusion matrix mean?**  
   `[[26475, 10525], [671, 44661]]` means 26,475 normal flows were correctly suppressed, 10,525 normal flows were false alarms, 671 attacks were missed, and 44,661 attacks were correctly detected.

4. **Why can this model now be used for Phase 2?**  
   It is accurate enough to be meaningful and imperfect enough to be realistic. Phase 2 can now measure how adversarial perturbations degrade a real trained detector.

**Analogy:**  
This baseline is the clean driving test before we add rain, fog, and rough roads. We first need to know how the car performs under normal conditions.
