"""FedAvg simulation utilities for the shared NIDS baseline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientShard:
    """A local client dataset represented by row indices."""

    client_id: int
    indices: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.indices))


def make_iid_partitions(num_examples: int, num_clients: int, seed: int) -> list[ClientShard]:
    """Split examples randomly and evenly across clients."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_examples)
    splits = np.array_split(shuffled, num_clients)
    return [ClientShard(client_id=i, indices=split.astype(int)) for i, split in enumerate(splits)]


def make_label_skew_partitions(labels: np.ndarray, num_clients: int, seed: int) -> list[ClientShard]:
    """Create a simple non-IID split by sorting labels before assigning shards."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    rng = np.random.default_rng(seed)
    jitter = rng.random(len(labels)) * 1e-6
    sorted_indices = np.lexsort((jitter, labels))
    splits = np.array_split(sorted_indices, num_clients)
    return [ClientShard(client_id=i, indices=split.astype(int)) for i, split in enumerate(splits)]


def average_state_dicts(client_states: list[OrderedDict], client_sizes: list[int]):
    """Weighted FedAvg aggregation over PyTorch state dictionaries."""

    if len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes must have the same length")
    if not client_states:
        raise ValueError("at least one client state is required")

    total_size = float(sum(client_sizes))
    if total_size <= 0.0:
        raise ValueError("total client size must be positive")

    averaged = OrderedDict()
    for key in client_states[0].keys():
        averaged[key] = sum(
            state[key] * (client_size / total_size)
            for state, client_size in zip(client_states, client_sizes)
        )

    return averaged


def label_distribution(labels: np.ndarray, shard: ClientShard) -> dict[str, int]:
    """Count normal and attack labels inside one client shard."""

    shard_labels = labels[shard.indices]
    return {
        "normal": int(np.sum(shard_labels == 0)),
        "attack": int(np.sum(shard_labels == 1)),
    }


def train_one_client(
    global_model,
    features: np.ndarray,
    labels: np.ndarray,
    shard: ClientShard,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Train one client model from the current global model."""

    import copy
    import torch
    from torch import nn

    client_model = copy.deepcopy(global_model)
    client_model.train()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.as_tensor(features[shard.indices], dtype=torch.float32)
    y = torch.as_tensor(labels[shard.indices], dtype=torch.float32).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(local_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(client_model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return client_model.state_dict()
