"""Run FedAvg simulations for IID and non-IID UNSW-NB15 client splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ci3203_federated.fedavg import (
    average_state_dicts,
    label_distribution,
    make_iid_partitions,
    make_label_skew_partitions,
    train_one_client,
)
from shared.config import DatasetConfig, TrainingConfig
from shared.data import prepare_from_disk
from shared.model import build_mlp, require_torch
from shared.train_baseline import evaluate_binary, predict_probabilities


def evaluate_model(model, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    probabilities = predict_probabilities(model, features.astype("float32"))
    return evaluate_binary(labels.astype(int), probabilities)


def maybe_subsample_training(
    features: np.ndarray,
    labels: np.ndarray,
    train_sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally use a deterministic training subset for CPU-friendly simulation."""

    if train_sample_size <= 0 or train_sample_size >= len(labels):
        return features, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=train_sample_size, replace=False)
    return features[indices], labels[indices]


def run_one_setting(
    setting: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    """Run FedAvg for one partitioning setting."""

    torch, _ = require_torch()
    torch.manual_seed(seed)

    if setting == "iid":
        shards = make_iid_partitions(len(train_labels), num_clients, seed)
    elif setting == "non_iid_label_skew":
        shards = make_label_skew_partitions(train_labels, num_clients, seed)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    global_model = build_mlp(
        input_dim=train_features.shape[1],
        hidden_dims=TrainingConfig().hidden_dims,
        dropout=TrainingConfig().dropout,
    )

    history = []
    for round_index in range(1, rounds + 1):
        client_states = []
        client_sizes = []
        for shard in shards:
            client_states.append(
                train_one_client(
                    global_model=global_model,
                    features=train_features,
                    labels=train_labels,
                    shard=shard,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )
            client_sizes.append(shard.size)

        global_model.load_state_dict(average_state_dicts(client_states, client_sizes))
        round_metrics = evaluate_model(global_model, test_features, test_labels)
        history.append({"round": round_index, "test_metrics": round_metrics})

    return {
        "setting": setting,
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "client_label_distributions": [
            {
                "client_id": shard.client_id,
                "size": shard.size,
                **label_distribution(train_labels, shard),
            }
            for shard in shards
        ],
        "history": history,
        "final_metrics": history[-1]["test_metrics"],
    }


def run_fedavg(
    data_dir: Path,
    output_path: Path,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    train_sample_size: int,
    seed: int,
) -> dict[str, object]:
    """Run IID and non-IID FedAvg simulations."""

    prepared = prepare_from_disk(DatasetConfig(data_dir=data_dir))
    train_features, train_labels = maybe_subsample_training(
        prepared.x_train.astype("float32"),
        prepared.y_train.astype(int),
        train_sample_size,
        seed,
    )
    test_features = prepared.x_test.astype("float32")
    test_labels = prepared.y_test.astype(int)

    results = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "method": "FedAvg simulation",
        "train_sample_size": int(len(train_labels)),
        "test_size": int(len(test_labels)),
        "settings": [
            run_one_setting(
                setting="iid",
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features,
                test_labels=test_labels,
                num_clients=num_clients,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
            ),
            run_one_setting(
                setting="non_iid_label_skew",
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features,
                test_labels=test_labels,
                num_clients=num_clients,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
            ),
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--output-path", type=Path, default=Path("outputs/ci3203-fedavg/fedavg_metrics.json"))
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_fedavg(
        data_dir=args.data_dir,
        output_path=args.output_path,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_sample_size=args.train_sample_size,
        seed=args.seed,
    )
    print(json.dumps({item["setting"]: item["final_metrics"] for item in results["settings"]}, indent=2))


if __name__ == "__main__":
    main()
