"""Generate reproducible paper figures from saved experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_attack_f1(metrics: dict, output_dir: Path) -> Path:
    labels = ["Clean", "FGSM", "PGD", "Constrained\nnumeric PGD"]
    keys = ["clean", "fgsm", "pgd", "constrained_numeric_pgd"]
    values = [metrics["metrics"][key]["f1"] for key in keys]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(labels, values, color=["#246a73", "#e07a5f", "#c8553d", "#6b8e23"])
    axis.set_ylabel("F1 score")
    axis.set_ylim(0, 1)
    axis.set_title("Adversarial attacks reduce NIDS detection performance")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    output_path = output_dir / "attack_f1.png"
    save_figure(figure, output_path)
    return output_path


def plot_certified_accuracy(metrics: dict, output_dir: Path) -> Path:
    certified = metrics["summary"]["certified_accuracy_by_radius"]
    radii = [float(radius) for radius in certified]
    values = [certified[str(radius)] for radius in radii]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(radii, values, marker="o", linewidth=2.2, color="#246a73")
    axis.set_xlabel("Certified L2 radius in preprocessed feature space")
    axis.set_ylabel("Certified accuracy")
    axis.set_ylim(0, 1)
    axis.set_xticks(radii)
    axis.set_title("Certified accuracy decreases as required radius increases")
    axis.grid(axis="y", alpha=0.25)
    output_path = output_dir / "certified_accuracy_by_radius.png"
    save_figure(figure, output_path)
    return output_path


def plot_privacy_utility(metrics: dict, output_dir: Path) -> Path:
    grouped: dict[str, list[tuple[float, float]]] = {"iid": [], "non_iid_label_skew": []}
    for experiment in metrics["settings"]:
        setting = experiment["setting"]
        if setting in grouped:
            grouped[setting].append((experiment["noise_multiplier"], experiment["final_metrics"]["f1"]))

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    labels = {"iid": "IID clients", "non_iid_label_skew": "Non-IID label skew"}
    colors = {"iid": "#246a73", "non_iid_label_skew": "#c8553d"}
    for setting, pairs in grouped.items():
        pairs.sort()
        noise, f1_scores = zip(*pairs)
        axis.plot(noise, f1_scores, marker="o", linewidth=2.2, label=labels[setting], color=colors[setting])

    axis.set_xlabel("Gaussian noise multiplier")
    axis.set_ylabel("Final test F1 score")
    axis.set_ylim(0, 1)
    axis.set_title("Local update noise produces a privacy-utility trade-off")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    output_path = output_dir / "privacy_utility_tradeoff.png"
    save_figure(figure, output_path)
    return output_path


def plot_bridge_tradeoff(metrics: dict, output_dir: Path) -> Path:
    conditions = metrics["conditions"]
    labels = ["Centralized", "FedAvg\nnon-IID", "FedAvg non-IID\n+ update noise"]
    clean_f1 = [condition["clean_metrics"]["f1"] for condition in conditions]
    certified_at_point_one = [
        condition["smoothing"]["certified_accuracy_by_radius"]["0.1"]
        for condition in conditions
    ]

    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    clean_bars = axis.bar([position - width / 2 for position in positions], clean_f1, width, label="Clean F1", color="#246a73")
    certificate_bars = axis.bar(
        [position + width / 2 for position in positions],
        certified_at_point_one,
        width,
        label="Certified accuracy at radius 0.10",
        color="#e07a5f",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_title("Bridge experiment: clean utility and certification must be read together")
    axis.legend(frameon=False)
    for bars in (clean_bars, certificate_bars):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{bar.get_height():.3f}", ha="center", fontsize=8)
    output_path = output_dir / "bridge_clean_and_certified.png"
    save_figure(figure, output_path)
    return output_path


def generate_figures(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attacks = load_json(PROJECT_ROOT / "outputs" / "ci3201-attacks" / "attack_metrics.json")
    smoothing = load_json(PROJECT_ROOT / "outputs" / "ci3201-smoothing" / "smoothing_metrics.json")
    ldp = load_json(PROJECT_ROOT / "outputs" / "ci3203-ldp" / "ldp_metrics.json")
    bridge = load_json(PROJECT_ROOT / "outputs" / "bridge-experiment" / "bridge_metrics.json")
    return [
        plot_attack_f1(attacks, output_dir),
        plot_certified_accuracy(smoothing, output_dir),
        plot_privacy_utility(ldp, output_dir),
        plot_bridge_tradeoff(bridge, output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for the NIDS research paper.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for generated_path in generate_figures(arguments.output_dir):
        print(generated_path)
