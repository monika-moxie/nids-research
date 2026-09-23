"""Local differential privacy mechanisms for federated client updates."""

from __future__ import annotations

from collections import OrderedDict


def state_difference(local_state: OrderedDict, global_state: OrderedDict):
    """Compute the client update: local model weights minus global model weights."""

    updates = OrderedDict()
    for key, local_value in local_state.items():
        global_value = global_state[key]
        if local_value.is_floating_point():
            updates[key] = local_value - global_value
        else:
            updates[key] = local_value.new_zeros(local_value.shape)
    return updates


def update_l2_norm(update: OrderedDict) -> float:
    """Compute one L2 norm across all floating-point tensors in an update."""

    import torch

    squared_norms = [
        tensor.float().pow(2).sum()
        for tensor in update.values()
        if tensor.is_floating_point()
    ]
    if not squared_norms:
        return 0.0
    return float(torch.sqrt(sum(squared_norms)).item())


def clip_update(update: OrderedDict, max_norm: float):
    """Scale an update down if its L2 norm is larger than max_norm."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    norm = update_l2_norm(update)
    scale = min(1.0, max_norm / (norm + 1e-12))
    return OrderedDict(
        (key, tensor * scale if tensor.is_floating_point() else tensor)
        for key, tensor in update.items()
    )


def add_gaussian_noise(update: OrderedDict, noise_std: float, generator=None):
    """Add Gaussian noise to each floating-point tensor in the update."""

    import torch

    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    noisy = OrderedDict()
    for key, tensor in update.items():
        if tensor.is_floating_point() and noise_std > 0.0:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=tensor.shape,
                generator=generator,
                device=tensor.device,
            )
            noisy[key] = tensor + noise
        else:
            noisy[key] = tensor.clone()
    return noisy


def aggregate_private_updates(
    global_state: OrderedDict,
    local_states: list[OrderedDict],
    client_sizes: list[int],
    clip_norm: float,
    noise_multiplier: float,
    generator=None,
):
    """Clip, noise, and weighted-average local client updates."""

    if len(local_states) != len(client_sizes):
        raise ValueError("local_states and client_sizes must have the same length")
    if not local_states:
        raise ValueError("at least one local state is required")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier must be non-negative")

    total_size = float(sum(client_sizes))
    if total_size <= 0.0:
        raise ValueError("total client size must be positive")

    noise_std = clip_norm * noise_multiplier
    private_updates = []
    for local_state in local_states:
        update = state_difference(local_state, global_state)
        clipped = clip_update(update, clip_norm)
        private_updates.append(add_gaussian_noise(clipped, noise_std, generator))

    new_state = OrderedDict()
    for key, global_value in global_state.items():
        if global_value.is_floating_point():
            averaged_update = sum(
                update[key] * (client_size / total_size)
                for update, client_size in zip(private_updates, client_sizes)
            )
            new_state[key] = global_value + averaged_update
        else:
            new_state[key] = local_states[0][key].clone()

    return new_state
