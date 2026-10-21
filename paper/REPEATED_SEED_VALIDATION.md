# Repeated-Seed Validation

## Why This Validation Exists

The first reported federated and bridge results were deterministic single runs.
They establish feasibility, but a single random seed cannot show whether a
pattern is stable across model initialization, data subsampling, client
partitioning, Gaussian update noise, and smoothing samples. This validation
repeats the same CPU-friendly configurations for seeds `42`, `1337`, and
`2026`, then reports the arithmetic mean and sample standard deviation.

## How It Works

`paper/run_repeated_seed_validation.py` calls the existing experiment runners;
it does not reimplement FedAvg, privacy noise, or smoothing. Each raw result is
saved under `outputs/phase7-validation/per_seed/`. The summary at
`outputs/phase7-validation/summary.json` reports `mean +/- sample_std`, where
the sample standard deviation uses `n - 1` degrees of freedom.

Settings are unchanged from the original CPU-friendly experiments:

- FedAvg and update-noise study: 20,000 training rows, 5 clients, 3 rounds,
  1 local epoch, clip norm 10.0.
- Bridge: 10,000 training rows, 5 clients, 2 rounds, 1 local epoch, 200
  certification flows, sigma 0.25, and 64 noisy samples per flow.
- Local-update-noise levels: 0.0, 0.001, 0.005, and 0.01.

## Table 7. FedAvg Repeated-Seed Results

| Client distribution | F1 (mean +/- std) | ROC-AUC (mean +/- std) | Successful runs |
| --- | ---: | ---: | ---: |
| IID | 0.8460 +/- 0.0090 | 0.9359 +/- 0.0020 | 3/3 |
| Non-IID label skew | 0.8187 +/- 0.0241 | 0.8853 +/- 0.0128 | 3/3 |

Interpretation: the IID-to-non-IID F1 reduction remains across all three
seeds. The non-IID standard deviation is also larger, which is consistent with
greater sensitivity to the sampled label-skewed client partitions.

## Table 8. Local-DP-Style Noise Repeated-Seed F1

| Noise multiplier | IID F1 (mean +/- std) | Non-IID F1 (mean +/- std) | Successful runs |
| ---: | ---: | ---: | ---: |
| 0.000 | 0.8460 +/- 0.0090 | 0.8187 +/- 0.0241 | 3/3, 3/3 |
| 0.001 | 0.8426 +/- 0.0087 | 0.8070 +/- 0.0234 | 3/3, 3/3 |
| 0.005 | 0.8190 +/- 0.0053 | 0.7626 +/- 0.0172 | 3/3, 3/3 |
| 0.010 | 0.7952 +/- 0.0118 | 0.6188 +/- 0.1842 | 2/3, 2/3 |

Interpretation: the utility decrease persists at the stable noise levels.
At noise multiplier 0.01, one seed produced non-finite predictions in both
IID and non-IID conditions. Therefore the 0.01 scores summarize only two valid
runs and must be described as an unstable high-noise regime, not a dependable
privacy-utility operating point.

## Table 9. Bridge Repeated-Seed Results

| Condition | Clean F1 (mean +/- std) | Certified accuracy at radius 0.10 (mean +/- std) | Coverage (mean +/- std) |
| --- | ---: | ---: | ---: |
| Centralized baseline | 0.8886 +/- 0.0000 | 0.4683 +/- 0.0379 | 0.9317 +/- 0.0161 |
| FedAvg, non-IID | 0.7130 +/- 0.0008 | 0.5483 +/- 0.0257 | 0.9900 +/- 0.0100 |
| FedAvg, non-IID + local-DP-style noise | 0.7187 +/- 0.0031 | 0.4367 +/- 0.1172 | 0.8367 +/- 0.1173 |

Interpretation: the private bridge condition has lower mean certified accuracy
at radius 0.10 than the non-private FedAvg condition and substantially higher
variation. The FedAvg bridge condition has a higher certificate value than the
centralized model at radius 0.10, but it also has far lower clean F1. The
correct conclusion remains that federated training and update noise change the
utility-certification profile; these results do not establish one condition as
universally more robust.

## Viva Defense Points

1. **Why use three seeds?**  
   Three is a CPU-feasible first stability check. It is enough to expose clear
   instability, but it is not a substitute for a large statistical study.

2. **Why sample standard deviation instead of population standard deviation?**  
   These three runs are a sample of possible random seeds. Dividing by `n - 1`
   avoids underestimating uncertainty from that sample.

3. **Why retain failed high-noise runs?**  
   Omitting them would bias the result toward successful runs. Numerical
   instability is itself evidence that this noise level is not dependable under
   the current training configuration.

4. **Why does the centralized clean F1 have zero variation?**  
   The same saved centralized baseline is evaluated on the same full test set
   in every bridge run. Its smoothing metrics still vary because each seed
   selects a different certification subset and noise samples.

## Remaining Limitation

The validation confirms key qualitative patterns only for this CPU-scale
configuration and three seeds. Final work with more resources should add more
seeds, larger certification samples, confidence intervals, and a formal
privacy accountant before stronger general claims are made.
