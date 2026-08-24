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

### 2026-08-23 - Phase 2: Attack Suite

**What it does in plain language:**  
Tests whether the trained NIDS model can be fooled by small adversarial changes to its input features.

**Why it exists:**  
The CI3201 track is about adversarial robustness. A detector that performs well on clean traffic may still fail if an attacker deliberately changes features in the direction that maximizes model error.

**Likely teacher questions and confident answers:**

1. **What is FGSM?**  
   FGSM is a one-step gradient attack. It computes how each input feature should move to increase loss, then nudges every feature by a small epsilon in that direction.

2. **Why is PGD stronger than FGSM?**  
   PGD repeats multiple smaller gradient steps and clips the perturbation after each step. Because it updates the gradient repeatedly, it usually finds more damaging adversarial examples.

3. **Why include constrained numeric PGD?**  
   Unconstrained attacks can alter one-hot categorical features unrealistically. Constrained numeric PGD leaves categorical indicators fixed and only changes numeric features, making it more appropriate for tabular NIDS data.

4. **What did the attack results show?**  
   On the full 82,332-row test set, clean F1 was 0.8886. FGSM dropped F1 to 0.2223, PGD dropped it to 0.1143, and constrained numeric PGD dropped it to 0.8058. This shows the model is highly vulnerable to unconstrained gradient attacks and moderately vulnerable to a more realistic numeric-only attack.

**Analogy:**  
Clean testing asks whether the lock works with ordinary keys. Adversarial testing asks whether a skilled attacker can file the key slightly and still open the door.

### 2026-08-23 - Phase 3: Randomized Smoothing Defense

**What it does in plain language:**  
Wraps the trained NIDS model in noise-based voting and estimates how far an input can move before the smoothed prediction is no longer certified.

**Why it exists:**  
Phase 2 showed the model is vulnerable to attacks. Phase 3 adds the certified-defense part of CI3201 by reporting robustness guarantees at multiple radii.

**Likely teacher questions and confident answers:**

1. **What is randomized smoothing?**  
   It predicts many noisy copies of the same input and uses majority vote as the final prediction. If the vote is confident enough, we can certify local stability.

2. **What does certified radius mean?**  
   It is the L2 distance in preprocessed feature space within which the smoothed classifier's prediction is guaranteed to remain stable under the assumptions of randomized smoothing.

3. **Why does certified accuracy decrease as radius increases?**  
   A larger radius is a stronger requirement. Fewer examples can be proven correct and stable under larger perturbations.

4. **What were the Phase 3 results?**  
   On a deterministic 500-flow sample, smoothed accuracy was 0.8400. Certified accuracy was 0.8200 at radius 0.0, 0.7640 at 0.05, 0.6520 at 0.10, and 0.1420 at 0.20.

**Analogy:**  
Randomized smoothing is like asking a crowd of slightly noisy witnesses. If almost all witnesses agree, we trust the answer more and can tolerate more disturbance around the original input.

### 2026-08-24 - Phase 4: FedAvg Simulation

**What it does in plain language:**  
Simulates several clients training a shared NIDS model without pooling their raw traffic data in one place.

**Why it exists:**  
The CI3203 track studies federated learning and privacy. FedAvg is the baseline federated algorithm we need before adding local differential privacy in Phase 5.

**Likely teacher questions and confident answers:**

1. **What is FedAvg?**  
   FedAvg is federated averaging. Clients train local copies of the model, then the server averages their weights, usually weighted by how much data each client has.

2. **Why weight by client dataset size?**  
   A client with more examples provides a more statistically influential update. Weighting by `n_k / total` prevents tiny clients from affecting the global model as much as large clients.

3. **What is the difference between IID and non-IID here?**  
   IID clients get random shards that roughly match the global distribution. Non-IID clients get label-skewed shards, meaning some clients see mostly normal traffic and others mostly attacks.

4. **What were the results?**  
   With 5 clients and 3 rounds on 20,000 training rows, IID FedAvg reached F1 0.8554 and ROC-AUC 0.9361. Non-IID label-skew reached F1 0.8366 and ROC-AUC 0.8999.

**Analogy:**  
FedAvg is like five students solving the same problem set separately, then combining their answer sheets into one shared solution, giving more weight to students who saw more examples.
