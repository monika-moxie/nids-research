"""Test whether certified robustness survives FedAvg and local-DP noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ci3201_adversarial.smoothing import (
    Certificate,
    certified_accuracy_by_radius,
    certify_feature,
)
from ci3203_federated.fedavg import (
    average_state_dicts,
    make_label_skew_partitions,
    train_one_client,
)
from ci3203_federated.privacy import aggregate_private_updates
from ci3203_federated.run_fedavg import evaluate_model, maybe_subsample_training
from shared.config import DatasetConfig, TrainingConfig
from shared.data import prepare_from_disk
from shared.model import build_mlp, require_torch


def load_centralized_model(model_path: Path, input_dim: int):
    """Load the Phase 1 centralized baseline model."""

    torch, _ = require_torch()
    model = build_mlp(
        input_dim=input_dim,
        hidden_dims=TrainingConfig().hidden_dims,
        dropout=TrainingConfig().dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def train_fedavg_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    """Train a non-IID label-skew FedAvg model for the bridge comparison."""

    torch, _ = require_torch()
    torch.manual_seed(seed)
    shards = make_label_skew_partitions(train_labels, num_clients, seed)
    model = build_mlp(
        input_dim=train_features.shape[1],
        hidden_dims=TrainingConfig().hidden_dims,
        dropout=TrainingConfig().dropout,
    )

    for _ in range(rounds):
        local_states = []
        client_sizes = []
        for shard in shards:
            local_states.append(
                train_one_client(
                    global_model=model,
                    features=train_features,
                    labels=train_labels,
                    shard=shard,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )
            client_sizes.append(shard.size)
        model.load_state_dict(average_state_dicts(local_states, client_sizes))

    model.eval()
    return model


def train_ldp_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
):
    """Train a non-IID FedAvg model with local-DP-style update perturbation."""

    torch, _ = require_torch()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    shards = make_label_skew_partitions(train_labels, num_clients, seed)
    model = build_mlp(
        input_dim=train_features.shape[1],
        hidden_dims=TrainingConfig().hidden_dims,
        dropout=TrainingConfig().dropout,
    )

    for _ in range(rounds):
        global_state = model.state_dict()
        local_states = []
        client_sizes = []
        for shard in shards:
            local_states.append(
                train_one_client(
                    global_model=model,
                    features=train_features,
                    labels=train_labels,
                    shard=shard,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )
            client_sizes.append(shard.size)
        model.load_state_dict(
            aggregate_private_updates(
                global_state=global_state,
                local_states=local_states,
                client_sizes=client_sizes,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                generator=generator,
            )
        )

    model.eval()
    return model


def select_eval_subset(features: np.ndarray, labels: np.ndarray, sample_size: int, seed: int):
    """Select the same deterministic certification subset for all compared models."""

    if sample_size <= 0 or sample_size >= len(labels):
        return features, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=sample_size, replace=False)
    return features[indices], labels[indices]


def smoothing_summary(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    num_noise_samples: int,
    batch_size: int,
    alpha: float,
    radii: list[float],
    seed: int,
) -> dict[str, object]:
    """Certify one model on a shared subset."""

    certificates: list[Certificate] = []
    for index, feature in enumerate(features):
        certificates.append(
            certify_feature(
                model=model,
                feature=feature,
                sigma=sigma,
                num_samples=num_noise_samples,
                batch_size=batch_size,
                alpha=alpha,
                seed=seed + index,
            )
        )

    predictions = np.asarray([certificate.prediction for certificate in certificates])
    abstained = np.asarray([certificate.abstained for certificate in certificates])
    radii_values = np.asarray([certificate.certified_radius for certificate in certificates])
    return {
        "smoothed_accuracy": float(np.mean(predictions == labels)),
        "coverage": float(np.mean(~abstained)),
        "mean_certified_radius": float(np.mean(radii_values)),
        "median_certified_radius": float(np.median(radii_values)),
        "certified_accuracy_by_radius": certified_accuracy_by_radius(certificates, labels, radii),
    }


def evaluate_condition(
    name: str,
    model,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    cert_features: np.ndarray,
    cert_labels: np.ndarray,
    sigma: float,
    num_noise_samples: int,
    smoothing_batch_size: int,
    alpha: float,
    radii: list[float],
    seed: int,
) -> dict[str, object]:
    """Return clean and certified metrics for one model condition."""

    return {
        "condition": name,
        "clean_metrics": evaluate_model(model, test_features, test_labels),
        "smoothing": smoothing_summary(
            model=model,
            features=cert_features,
            labels=cert_labels,
            sigma=sigma,
            num_noise_samples=num_noise_samples,
            batch_size=smoothing_batch_size,
            alpha=alpha,
            radii=radii,
            seed=seed,
        ),
    }


def run_bridge(
    data_dir: Path,
    baseline_dir: Path,
    output_path: Path,
    train_sample_size: int,
    cert_sample_size: int,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    clip_norm: float,
    noise_multiplier: float,
    sigma: float,
    num_noise_samples: int,
    smoothing_batch_size: int,
    alpha: float,
    radii: list[float],
    seed: int,
) -> dict[str, object]:
    """Run the Phase 6 bridge experiment."""

    prepared = prepare_from_disk(DatasetConfig(data_dir=data_dir))
    train_features, train_labels = maybe_subsample_training(
        prepared.x_train.astype("float32"),
        prepared.y_train.astype(int),
        train_sample_size,
        seed,
    )
    test_features = prepared.x_test.astype("float32")
    test_labels = prepared.y_test.astype(int)
    cert_features, cert_labels = select_eval_subset(test_features, test_labels, cert_sample_size, seed)

    centralized_model = load_centralized_model(
        baseline_dir / "baseline_mlp.pt",
        input_dim=test_features.shape[1],
    )
    fedavg_model = train_fedavg_model(
        train_features=train_features,
        train_labels=train_labels,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    ldp_model = train_ldp_model(
        train_features=train_features,
        train_labels=train_labels,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        seed=seed,
    )

    conditions = [
        evaluate_condition(
            "centralized_baseline",
            centralized_model,
            test_features,
            test_labels,
            cert_features,
            cert_labels,
            sigma,
            num_noise_samples,
            smoothing_batch_size,
            alpha,
            radii,
            seed,
        ),
        evaluate_condition(
            "fedavg_non_iid",
            fedavg_model,
            test_features,
            test_labels,
            cert_features,
            cert_labels,
            sigma,
            num_noise_samples,
            smoothing_batch_size,
            alpha,
            radii,
            seed,
        ),
        evaluate_condition(
            "fedavg_non_iid_ldp",
            ldp_model,
            test_features,
            test_labels,
            cert_features,
            cert_labels,
            sigma,
            num_noise_samples,
            smoothing_batch_size,
            alpha,
            radii,
            seed,
        ),
    ]

    results = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "bridge_question": "Does certified robustness survive FedAvg plus local-DP-style update noise?",
        "train_sample_size": int(len(train_labels)),
        "test_size": int(len(test_labels)),
        "cert_sample_size": int(len(cert_labels)),
        "federated_setting": "non_iid_label_skew",
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "smoothing": {
            "sigma": sigma,
            "num_noise_samples": num_noise_samples,
            "alpha": alpha,
            "radii": radii,
            "certificate_space": "L2 radius in preprocessed feature space",
        },
        "conditions": conditions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/shared-baseline"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/bridge-experiment/bridge_metrics.json"))
    parser.add_argument("--train-sample-size", type=int, default=10000)
    parser.add_argument("--cert-sample-size", type=int, default=200)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--num-noise-samples", type=int, default=64)
    parser.add_argument("--smoothing-batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_bridge(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        output_path=args.output_path,
        train_sample_size=args.train_sample_size,
        cert_sample_size=args.cert_sample_size,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        sigma=args.sigma,
        num_noise_samples=args.num_noise_samples,
        smoothing_batch_size=args.smoothing_batch_size,
        alpha=args.alpha,
        radii=args.radii,
        seed=args.seed,
    )
    compact = {
        item["condition"]: {
            "clean_f1": item["clean_metrics"]["f1"],
            "certified_accuracy_by_radius": item["smoothing"]["certified_accuracy_by_radius"],
        }
        for item in results["conditions"]
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
