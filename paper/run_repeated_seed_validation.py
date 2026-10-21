"""Repeat key federated experiments across seeds and summarize variability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from bridge_experiment.run_bridge import run_bridge
from ci3203_federated.run_fedavg import run_fedavg
from ci3203_federated.run_ldp import run_ldp_tradeoff
from shared.config import DatasetConfig


DEFAULT_SEEDS = [42, 1337, 2026]


def mean_and_sample_std(values: Iterable[float]) -> dict[str, float | int]:
    """Return a transparent summary; sample deviation is undefined for one run."""

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("At least one value is required for summary statistics.")
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "num_seeds": int(array.size),
    }


def summarize_metric_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize valid metrics without hiding failed or unstable experiment runs."""

    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    valid_records = [record for record in records if all(metric_name in record for metric_name in metric_names)]
    failed_records = [record for record in records if record not in valid_records]
    return {
        "num_attempted_runs": len(records),
        "num_successful_runs": len(valid_records),
        "num_failed_runs": len(failed_records),
        "failure_messages": [str(record.get("error", "missing metric values")) for record in failed_records],
        "metrics": {
            metric_name: mean_and_sample_std(record[metric_name] for record in valid_records)
            for metric_name in metric_names
        }
        if valid_records
        else {},
    }


def summarize_fedavg(results_by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for setting in ("iid", "non_iid_label_skew"):
        records = []
        for result in results_by_seed.values():
            matched = next(item for item in result["settings"] if item["setting"] == setting)
            records.append(matched["final_metrics"])
        summary[setting] = summarize_metric_records(records)
    return summary


def summarize_ldp(results_by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    first_result = next(iter(results_by_seed.values()))
    for reference in first_result["settings"]:
        setting = reference["setting"]
        noise_multiplier = reference["noise_multiplier"]
        label = f"{setting}_noise_{noise_multiplier:g}"
        records = []
        for result in results_by_seed.values():
            matched = next(
                item
                for item in result["settings"]
                if item["setting"] == setting and item["noise_multiplier"] == noise_multiplier
            )
            records.append(matched["final_metrics"])
        summary[label] = summarize_metric_records(records)
    return summary


def summarize_bridge(results_by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    first_result = next(iter(results_by_seed.values()))
    for reference in first_result["conditions"]:
        condition = reference["condition"]
        clean_metrics = []
        smoothing_records = []
        for result in results_by_seed.values():
            matched = next(item for item in result["conditions"] if item["condition"] == condition)
            clean_metrics.append(matched["clean_metrics"])
            smoothing_records.append(matched["smoothing"])

        radius_keys = reference["smoothing"]["certified_accuracy_by_radius"].keys()
        summary[condition] = {
            "clean_metrics": summarize_metric_records(clean_metrics),
            "smoothed_accuracy": mean_and_sample_std(item["smoothed_accuracy"] for item in smoothing_records),
            "coverage": mean_and_sample_std(item["coverage"] for item in smoothing_records),
            "mean_certified_radius": mean_and_sample_std(
                item["mean_certified_radius"] for item in smoothing_records
            ),
            "certified_accuracy_by_radius": {
                radius: mean_and_sample_std(
                    item["certified_accuracy_by_radius"][radius] for item in smoothing_records
                )
                for radius in radius_keys
            },
        }
    return summary


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    """Run selected experiments for every declared seed and summarize results."""

    output_dir = args.output_dir
    raw_dir = output_dir / "per_seed"
    fedavg_results: dict[int, dict[str, object]] = {}
    ldp_results: dict[int, dict[str, object]] = {}
    bridge_results: dict[int, dict[str, object]] = {}

    for seed in args.seeds:
        if args.summarize_only:
            if not args.skip_fedavg:
                fedavg_results[seed] = load_json(raw_dir / f"fedavg_seed_{seed}.json")
            if not args.skip_ldp:
                ldp_results[seed] = load_json(raw_dir / f"ldp_seed_{seed}.json")
            if not args.skip_bridge:
                bridge_results[seed] = load_json(raw_dir / f"bridge_seed_{seed}.json")
            continue
        if not args.skip_fedavg:
            fedavg_results[seed] = run_fedavg(
                data_dir=args.data_dir,
                output_path=raw_dir / f"fedavg_seed_{seed}.json",
                num_clients=args.num_clients,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_sample_size=args.train_sample_size,
                seed=seed,
            )
        if not args.skip_ldp:
            ldp_results[seed] = run_ldp_tradeoff(
                data_dir=args.data_dir,
                output_path=raw_dir / f"ldp_seed_{seed}.json",
                num_clients=args.num_clients,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_sample_size=args.train_sample_size,
                clip_norm=args.clip_norm,
                noise_multipliers=args.noise_multipliers,
                seed=seed,
            )
        if not args.skip_bridge:
            bridge_results[seed] = run_bridge(
                data_dir=args.data_dir,
                baseline_dir=args.baseline_dir,
                output_path=raw_dir / f"bridge_seed_{seed}.json",
                train_sample_size=args.bridge_train_sample_size,
                cert_sample_size=args.cert_sample_size,
                num_clients=args.num_clients,
                rounds=args.bridge_rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                clip_norm=args.clip_norm,
                noise_multiplier=args.bridge_noise_multiplier,
                sigma=args.sigma,
                num_noise_samples=args.num_noise_samples,
                smoothing_batch_size=args.smoothing_batch_size,
                alpha=args.alpha,
                radii=args.radii,
                seed=seed,
            )

    summary: dict[str, object] = {
        "purpose": "Repeated-seed validation of federated and bridge experiments",
        "seeds": args.seeds,
        "statistic_note": "sample_std uses n - 1 degrees of freedom; it is 0.0 for one seed.",
        "configuration": {
            "num_clients": args.num_clients,
            "fedavg_ldp_train_sample_size": args.train_sample_size,
            "fedavg_ldp_rounds": args.rounds,
            "bridge_train_sample_size": args.bridge_train_sample_size,
            "bridge_rounds": args.bridge_rounds,
            "cert_sample_size": args.cert_sample_size,
            "noise_multipliers": args.noise_multipliers,
            "bridge_noise_multiplier": args.bridge_noise_multiplier,
            "sigma": args.sigma,
            "num_noise_samples": args.num_noise_samples,
        },
    }
    if fedavg_results:
        summary["fedavg"] = summarize_fedavg(fedavg_results)
    if ldp_results:
        summary["ldp"] = summarize_ldp(ldp_results)
    if bridge_results:
        summary["bridge"] = summarize_bridge(bridge_results)

    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/shared-baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase7-validation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--bridge-rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-sample-size", type=int, default=20000)
    parser.add_argument("--bridge-train-sample-size", type=int, default=10000)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--noise-multipliers", type=float, nargs="+", default=[0.0, 0.001, 0.005, 0.01])
    parser.add_argument("--bridge-noise-multiplier", type=float, default=0.005)
    parser.add_argument("--cert-sample-size", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--num-noise-samples", type=int, default=64)
    parser.add_argument("--smoothing-batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--skip-fedavg", action="store_true")
    parser.add_argument("--skip-ldp", action="store_true")
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Read existing per-seed JSON files and write only the aggregate summary.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run_validation(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
