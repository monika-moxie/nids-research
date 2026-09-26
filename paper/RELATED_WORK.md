# Related Work and Research Positioning

## Purpose

This is a literature-backed, paper-ready related-work draft. It distinguishes
established methods from this project's contribution, so the final paper does
not accidentally claim to invent FGSM, PGD, randomized smoothing, FedAvg, or
differential privacy.

## Draft for Section II: Related Work

### A. Benchmark NIDS Data

UNSW-NB15 was introduced by Moustafa and Slay as a network-intrusion dataset
for evaluating detection systems [1]. We use its predefined training and test
CSV files for binary intrusion detection. Our contribution is not a new
dataset split or a claim of state-of-the-art benchmark performance; it is a
controlled common baseline for robustness and federated-learning experiments.

### B. Adversarial Evaluation and Certified Robustness

Goodfellow, Shlens, and Szegedy introduced the fast gradient sign method
(FGSM), showing that small, loss-increasing input perturbations can cause
neural-network misclassification [2]. Madry et al. developed the robust-
optimization perspective and use projected gradient descent (PGD) as a strong
first-order adversary [3]. These methods motivate our white-box evaluation.
Because network-flow data is tabular, unrestricted post-preprocessing attacks
can make one-hot categorical indicators fractional or contradictory. We
therefore report both standard feature-space FGSM/PGD and a numeric-only PGD
baseline. This constraint is a practical evaluation choice, not a claim of a
fully semantics-preserving raw-traffic attack.

Empirical attacks alone cannot prove a defense. Cohen, Rosenfeld, and Kolter
showed that Gaussian randomized smoothing can turn a base classifier into one
with certified L2 robustness under stated probabilistic conditions [4]. Our
implementation follows that high-level principle: sample Gaussian-perturbed
preprocessed inputs, estimate the winning class probability conservatively,
and report certified accuracy at several radii. The certificate must be read
precisely: it concerns L2 movement in the 194-dimensional preprocessed feature
space, not every physically valid packet or flow transformation.

### C. Federated Learning and Update Privacy

FedAvg, introduced by McMahan et al., trains local client models and aggregates
them by data-size-weighted averaging without centralizing raw client examples
[5]. It makes the IID versus non-IID distinction essential because client data
distributions can differ substantially. We therefore compare a random IID
partition with a deliberately label-skewed non-IID partition.

Keeping raw data local does not itself establish a formal privacy guarantee.
Dwork formalized differential privacy as a privacy-risk bound [6], while Geyer,
Klein, and Nabi studied client-side differential privacy in federated learning
and the resulting privacy-utility trade-off [7]. Our Phase 5 mechanism clips a
client update and adds Gaussian noise before it leaves the client. Since this
project does not define a formal adjacency relation, calibrate noise to a
target epsilon and delta, or account for composition over rounds, it is
described throughout as **local-DP-style update perturbation**, not quantified
differential privacy.

### D. Federated Intrusion Detection and Project Gap

Federated intrusion detection is already an active field. Agrawal et al.
survey the associated concepts, challenges, and future directions [8], while
Li et al. present an efficient federated NIDS system using dynamic aggregation
[9]. These studies support the practical motivation for training across
distributed security data.

This project does **not** claim to be the first federated NIDS, the first
adversarial NIDS study, or the first privacy-aware NIDS. Its scoped
contribution is a modular and reproducible experimental pipeline that reports,
under explicit limitations:

1. clean NIDS utility and white-box adversarial vulnerability;
2. randomized-smoothing certificates in the same model representation;
3. IID/non-IID FedAvg and local-DP-style utility degradation; and
4. an initial bridge comparison that places clean F1 and certified accuracy
   side by side for centralized and federated/private conditions.

The bridge question is deliberately empirical: how does the robustness profile
change under the chosen federated and noised-update settings? It is not a
universal theorem about all federated or differentially private NIDS systems.

## Defensible Research Gap Statement

"Prior work establishes adversarial attacks, randomized smoothing, federated
optimization, and privacy mechanisms as important but often separately
evaluated concerns. This project provides a controlled UNSW-NB15 pipeline that
examines their interaction and reports both clean utility and local certified
robustness. The study is presented as an initial integration experiment, with
explicit constraints on the attack space, certificate space, federated scale,
and privacy accounting."

## Citation Discipline

- Use `[1]` through `[9]` in the final IEEE manuscript and map them to the
  BibTeX entries in `references.bib`.
- Read the source papers before copying any stronger claim into the final
  manuscript. This file is a verified starting set, not a substitute for a
  systematic review.
- Do not write "first" or "novel" without evidence from a broader,
  documented literature search.
- Cite [4] for randomized smoothing theory, but retain the distinction that
  this implementation uses a conservative Hoeffding bound and applies it in
  preprocessed feature space.

## References

[1] N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network
Intrusion Detection Systems," *2015 Military Communications and Information
Systems Conference (MilCIS)*, 2015, doi: 10.1109/MilCIS.2015.7348942.

[2] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and Harnessing
Adversarial Examples," *International Conference on Learning Representations
(ICLR)*, 2015, arXiv:1412.6572.

[3] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards Deep
Learning Models Resistant to Adversarial Attacks," *ICLR*, 2018,
arXiv:1706.06083.

[4] J. Cohen, E. Rosenfeld, and J. Z. Kolter, "Certified Adversarial
Robustness via Randomized Smoothing," *Proceedings of the 36th International
Conference on Machine Learning*, vol. 97, pp. 1310-1320, 2019.

[5] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
"Communication-Efficient Learning of Deep Networks from Decentralized Data,"
*Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics*, vol. 54, pp. 1273-1282, 2017.

[6] C. Dwork, "Differential Privacy," *International Colloquium on Automata,
Languages, and Programming (ICALP)*, pp. 1-12, 2006.

[7] R. C. Geyer, T. Klein, and M. Nabi, "Differentially Private Federated
Learning: A Client Level Perspective," arXiv:1712.07557, 2017.

[8] S. Agrawal et al., "Federated Learning for Intrusion Detection System:
Concepts, Challenges and Future Directions," *Computer Communications*, vol.
195, pp. 346-361, 2022, doi: 10.1016/j.comcom.2022.09.012.

[9] J. Li, X. Tong, J. Liu, and L. Cheng, "An Efficient Federated Learning
System for Network Intrusion Detection," *IEEE Systems Journal*, vol. 17, no.
2, pp. 2455-2464, 2023, doi: 10.1109/JSYST.2023.3236995.
