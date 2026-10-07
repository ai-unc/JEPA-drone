"""Training stub. Wires the environment, dataset, encoders, predictor, loss, and baseline."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from skyjepa import baseline, losses
from skyjepa.config import Config
from skyjepa.data import dataset
from skyjepa.env import point_mass
from skyjepa.models import encoders, predictor
from skyjepa.windows import rollout_windows

_WINDOW_KEYS = (
    "state_history",
    "action_history",
    "future_states",
    "future_actions",
)


class WindowDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.tensors = {
            key: torch.from_numpy(np.ascontiguousarray(arrays[key], dtype=np.float32))
            for key in _WINDOW_KEYS
        }
        counts = {key: int(tensor.shape[0]) for key, tensor in self.tensors.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"window count mismatch: {counts}")
        self.length = counts["state_history"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: tensor[index] for key, tensor in self.tensors.items()}


def _require_shapes(arrays: dict[str, np.ndarray], config: Config) -> None:
    count = int(arrays["state_history"].shape[0])
    expected = {
        "state_history": (count, config.history_len, config.state_dim),
        "action_history": (count, config.history_len, config.action_dim),
        "future_states": (count, config.horizon, config.state_dim),
        "future_actions": (count, config.horizon, config.action_dim),
    }
    for key, shape in expected.items():
        actual = tuple(arrays[key].shape)
        if actual != shape:
            raise ValueError(f"{key} shape {actual} != {shape}")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode_sequence(encoder: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """windows (B, K, H, D) -> (B, K, latent_dim)."""

    batch, steps, history, dim = windows.shape
    encoded = encoder(windows.reshape(batch * steps, history, dim))
    return encoded.reshape(batch, steps, -1)


def _on_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _jepa_step(
    batch: dict[str, torch.Tensor],
    state_encoder: nn.Module,
    action_encoder: nn.Module,
    latent_predictor: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_windows, action_windows = rollout_windows(
        batch["state_history"],
        batch["action_history"],
        batch["future_states"],
        batch["future_actions"],
    )
    s_t = state_encoder(state_windows[:, 0])
    with torch.no_grad():
        targets = _encode_sequence(state_encoder, state_windows[:, 1:])
    z_seq = _encode_sequence(action_encoder, action_windows)
    predicted = latent_predictor.rollout(s_t, z_seq)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"rollout shape {tuple(predicted.shape)} != targets {tuple(targets.shape)}"
        )
    return losses.latent_prediction_loss(predicted, targets), s_t


def train_jepa(loader: DataLoader, config: Config) -> None:
    device = config.device
    state_encoder = encoders.StateHistoryEncoder(config).to(device)
    action_encoder = encoders.ActionHistoryEncoder(config).to(device)
    latent_predictor = predictor.LatentPredictor(config).to(device)
    params = [
        *state_encoder.parameters(),
        *action_encoder.parameters(),
        *latent_predictor.parameters(),
    ]
    optimizer = torch.optim.Adam(params, lr=config.lr)

    for epoch in range(config.epochs):
        total = 0.0
        steps = 0
        last_latent = None
        for batch in loader:
            batch = _on_device(batch, device)
            loss, s_t = _jepa_step(batch, state_encoder, action_encoder, latent_predictor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
            last_latent = s_t.detach()
        if last_latent is None:
            raise RuntimeError("no training batches; check the dataset windows")
        stats = losses.latent_stats(last_latent)
        average = total / max(steps, 1)
        print(
            f"epoch {epoch:02d}  L_pred={average:.6f}  "
            f"latent_mean={stats['mean']:.4f}  latent_std={stats['std']:.4f}"
        )
        if stats["std"] < 1e-3:
            print("latent std is near 0; embeddings may have collapsed")


def train_baseline(loader: DataLoader, config: Config) -> None:
    device = config.device
    model = baseline.DirectStatePredictor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for _ in range(config.epochs):
        for batch in loader:
            batch = _on_device(batch, device)
            predicted = model(
                batch["state_history"],
                batch["action_history"],
                batch["future_actions"],
            )
            loss = (predicted - batch["future_states"]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    curves = []
    with torch.no_grad():
        for batch in loader:
            batch = _on_device(batch, device)
            predicted = model(
                batch["state_history"],
                batch["action_history"],
                batch["future_actions"],
            )
            per_step = torch.linalg.vector_norm(
                predicted - batch["future_states"], dim=-1
            ).mean(dim=0)
            curves.append(per_step.cpu())
    errors = torch.stack(curves).mean(dim=0).numpy()
    path = Path("artifacts") / "prediction_error_vs_horizon.png"
    path.parent.mkdir(exist_ok=True)
    baseline.plot_prediction_error_vs_horizon(errors, str(path))


def main() -> None:
    config = Config()
    _seed(config.seed)
    trajectories = point_mass.generate_trajectories(
        n=config.n_trajectories,
        n_steps=config.n_steps,
        dt=config.dt,
        seed=config.seed,
    )
    arrays = dataset.build_dataset(
        trajectories,
        history_len=config.history_len,
        horizon=config.horizon,
    )
    _require_shapes(arrays, config)
    loader = DataLoader(
        WindowDataset(arrays),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_jepa(loader, config)
    train_baseline(loader, config)


if __name__ == "__main__":
    main()
