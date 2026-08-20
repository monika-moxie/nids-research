# Dataset Choice: UNSW-NB15

## Decision

Phase 1 uses the UNSW-NB15 dataset for the shared NIDS baseline.

## Why This Dataset

UNSW-NB15 is a public academic network intrusion detection dataset from UNSW Canberra. It contains normal traffic and nine attack families, including Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms. The official page provides prebuilt training and testing CSV files: 175,341 training records and 82,332 testing records.

This makes it a good solo-project choice because it is large enough to be meaningful, small enough to handle on a laptop, and already structured as tabular ML data.

## Why Not CIC-IDS2017 for the Baseline

CIC-IDS2017 is also strong and widely used, but the official dataset is much larger because it includes PCAPs and multiple day-level CSV files. It is excellent for later discussion, but UNSW-NB15 is cleaner for building a first reproducible baseline under time and hardware constraints.

## Learning Framing

The baseline task is binary classification:

- `0`: normal traffic
- `1`: attack traffic

The `attack_cat` column is not used as an input feature for binary detection because it directly describes the attack class. Keeping it would create label leakage.

## Official Sources Checked

- UNSW-NB15 official dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CIC-IDS2017 official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- CICIoT2023 official dataset page: https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Expected Local Files

Place the official CSV files here:

```text
data/raw/UNSW-NB15/UNSW_NB15_training-set.csv
data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv
```

The folder is present in Git through `.gitkeep` placeholder files. The actual CSV files are intentionally ignored so the repository stays lightweight.
