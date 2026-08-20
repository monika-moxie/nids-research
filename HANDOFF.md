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
