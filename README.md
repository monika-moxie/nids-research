# NIDS Research

This repository contains a two-part academic research project on Network Intrusion Detection Systems (NIDS).

## Research Map

- `shared/` builds the common dataset, preprocessing, baseline model, and baseline metrics used by both project tracks.
- `ci3201-adversarial/` studies adversarial attacks and certified defenses for the NIDS classifier.
- `ci3203-federated/` studies federated learning and local differential privacy for distributed NIDS training.
- `bridge-experiment/` tests whether certified robustness survives federated learning and privacy noise.
- `paper/` stores the final paper skeleton, tables, figures, and writing notes.

## Phase Discipline

Each session should work on one phase only. At the end of every session, update:

- `PROGRESS.md`
- `DECISIONS.md`
- `LEARNING_LOG.md`
- `HANDOFF.md`

## Local Environment

Use a project-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The official UNSW-NB15 CSV files belong in:

```text
data/raw/UNSW-NB15/
```

The folder is tracked, but the CSV files are ignored because datasets should not be committed to Git.
