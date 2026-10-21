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

def _fit_normalizer(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """helper function that normalizes all the inputs for the eoncoder to be able to
    assess which feature has the biggest numbers. This helper computes one mean and one
    s.d. per feature."""


    states = torch.cat([tensors["state_history"], tensors["future_states"]], dim=1)
    actions = torch.cat([tensors["action_history"], tensors["future_actions"]], dim=1)
    states = states.reshape(-1, states.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])
    return {
        "state_mean": states.mean(0), "state_std": states.std(0) + 1e-6,
        "action_mean": actions.mean(0), "action_std": actions.std(0) + 1e-6,
    }

def _normalize(batch, norm):
    out = {}
    for key, value in batch.items():
        kind = "state" if "states" in key or key == "state_history" else "action"
        out[key] = (value - norm[f"{kind}_mean"]) / norm[f"{kind}_std"]
    return out

def _jepa_step(batch: dict[str, torch.Tensor], state_encoder: nn.Module, action_encoder: nn.Module, latent_predictor: nn.Module, config: Config,) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    # call rollout windows to cut these tensors into overlapping 10-step clips
    state_windows, action_windows = rollout_windows(
        batch["state_history"],
        batch["action_history"],
        batch["future_states"],
        batch["future_actions"],
    )
    s_t = state_encoder(state_windows[:, 0]) # turn the "now" clip into s_t
    targets = _encode_sequence(state_encoder, state_windows[:, 1:]) # turn the future clips into targets
    z_seq = _encode_sequence(action_encoder, action_windows) # turn action clips into z_seq
    predicted = latent_predictor.rollout(s_t, z_seq.transpose(0, 1)).transpose(0, 1) # guess the next 10 state summaries from s_t and z_seq
    if predicted.shape != targets.shape:
        raise ValueError(
            f"rollout shape {tuple(predicted.shape)} != targets {tuple(targets.shape)}"
        )
    # returns a weighted sum of the MSE (invariance), variance and covariance as a penalty
    loss, parts = losses.vicreg_loss(predicted, targets, s_t, lam=config.vicreg_lam, mu=config.vicreg_mu, nu=config.vicreg_nu,)
    return loss, parts, s_t

def train_jepa(loader: DataLoader, config: Config) -> None:
    device = config.device
    state_encoder = encoders.StateHistoryEncoder(config).to(device)
    action_encoder = encoders.ActionHistoryEncoder(config).to(device)
    latent_predictor = predictor.LatentPredictor(
        latent_dim=config.latent_dim,
        action_dim=config.latent_dim,  # the action encoder also outputs latent_dim
        hidden_dim=config.latent_dim
    ).to(device)

    params = [
        *state_encoder.parameters(),
        *action_encoder.parameters(),
        *latent_predictor.parameters(),
    ]
    
    optimizer = torch.optim.Adam(params, lr=config.lr) # pick the Adam optimizer
    norm = {k: v.to(device) for k, v in _fit_normalizer(loader.dataset.tensors).items()} # compute one mean and one s.d. per state and action feature 

    for epoch in range(config.epochs):
        totals = {"loss": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0}
        steps = 0
        last_latent = None
        for batch in loader:
            batch = _normalize(_on_device(batch, device), norm) # move every batch to the device and standardize
            loss, parts, s_t = _jepa_step(batch, state_encoder, action_encoder, latent_predictor, config)
            optimizer.zero_grad()
            loss.backward() # compute how to change every weight
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip) # caps the update from the line above
            optimizer.step() # apply the changes
            totals["loss"] += loss.item() # add the batch's total loss to the running total for the epoch
            for key, value in parts.items():
                totals[key] += value # do the same for the three separate terms in parts (pred error, variance penalty and covar penalty)
            steps += 1 
            last_latent = s_t.detach() 
        if last_latent is None:
            raise RuntimeError("no training batches; check the dataset windows")
        stats = losses.latent_stats(last_latent)
        avg = {key: value / max(steps, 1) for key, value in totals.items()}
        print(
            f"epoch {epoch:02d}  loss={avg['loss']:.4f}  pred={avg['pred']:.5f}  "
            f"var={avg['var']:.4f}  cov={avg['cov']:.4f}  "
            f"dim_std_min={stats['dim_std_min']:.4f}  dead={stats['dead_frac']:.0%}"
        )
        losses.check_collapse(stats, last_latent) # warns if the summaries have all bnecome the same vector (representation collapse)

# train a separate raw-state predictor on the same loader for comparison with the JEPA model
def train_baseline(loader: DataLoader, config: Config) -> None:
    model = baseline.DirectStatePredictor(config).to(config.device)
    model.fit_normalizer(loader.dataset.tensors)
    baseline.train_direct_predictor(model, loader, config)
    errors = baseline.prediction_error_vs_horizon(model, loader)
    path = Path("artifacts") / "prediction_error_vs_horizon.png"
    path.parent.mkdir(exist_ok=True)
    baseline.plot_prediction_error_vs_horizon(errors, str(path), dt=config.dt)

def main() -> None:
    config = Config() # start here
    _seed(config.seed) # seeds the random number generator
    
    # simulates 100 flights of a 2D point mass and returns NumPy arrays: states shaped (101, 4) as [x, y, vx, vy], and actions shaped (100, 2) as accelerations
    trajectories = point_mass.generate_trajectories(
        n=config.n_trajectories,
        n_steps=config.n_steps,
        dt=config.dt,
        seed=config.seed,
    )

    # slides a window of length 10 plus a future horizon of 10 along every flight producing four arrays: state_history, action_history, future_states, and future_actions
    arrays = dataset.build_dataset(
        trajectories,
        history_len=config.history_len,
        horizon=config.horizon,
    )
    _require_shapes(arrays, config) # check the shapes
    loader = DataLoader( # hands out shuffled batches of 32
        WindowDataset(arrays),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_jepa(loader, config) # train JEPA
    train_baseline(loader, config) # train the MLP that JEPA is to be compared against


if __name__ == "__main__":
    main()
