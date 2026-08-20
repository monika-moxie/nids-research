"""Train and evaluate the shared UNSW-NB15 deep baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from shared.config import DatasetConfig, TrainingConfig
from shared.data import ensure_parent_dir, prepare_from_disk
from shared.model import build_mlp, require_torch


def evaluate_binary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Compute binary NIDS metrics from attack probabilities."""

    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def train_baseline(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    output_dir: Path,
) -> dict[str, object]:
    """Train the MLP baseline and write model, preprocessor, and metrics artifacts."""

    torch, nn = require_torch()
    prepared = prepare_from_disk(dataset_config)
    x_train, x_val, y_train, y_val = train_test_split(
        prepared.x_train.astype("float32"),
        prepared.y_train.astype("float32"),
        test_size=training_config.validation_fraction,
        random_state=training_config.random_state,
        stratify=prepared.y_train,
    )

    torch.manual_seed(training_config.random_state)
    model = build_mlp(
        input_dim=x_train.shape[1],
        hidden_dims=training_config.hidden_dims,
        dropout=training_config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    history = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_dataset)
        val_probabilities = predict_probabilities(model, x_val)
        val_metrics = evaluate_binary(y_val.astype(int), val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

    test_probabilities = predict_probabilities(model, prepared.x_test.astype("float32"))
    test_metrics = evaluate_binary(prepared.y_test.astype(int), test_probabilities)
    results = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "input_dim": int(prepared.x_train.shape[1]),
        "feature_count_after_preprocessing": len(prepared.feature_names),
        "training_config": training_config.__dict__,
        "history": history,
        "test_metrics": test_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "baseline_mlp.pt")
    joblib.dump(prepared.preprocessor, output_dir / "preprocessor.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def predict_probabilities(model, features: np.ndarray) -> np.ndarray:
    """Run the model and return probabilities for the attack class."""

    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().data_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shared-baseline"))
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(data_dir=args.data_dir)
    training_config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
    ensure_parent_dir(args.output_dir / "metrics.json")
    results = train_baseline(dataset_config, training_config, args.output_dir)
    print(json.dumps(results["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
