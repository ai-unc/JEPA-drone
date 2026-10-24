"""Training stub. Wires the environment, dataset, encoders, predictor, loss, and baseline."""

import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import argparse
import torch

from skyjepa.data import trajectory_loader
from skyjepa.data.dataset import Data
from skyjepa.logging.logger import _LOG, _ensure_logging
from skyjepa.models import encoders, predictor
from skyjepa import baseline, losses
from skyjepa.config import Config
from skyjepa.windows import sliding_windows



def _param_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())

class WindowDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.tensors = {
            key: torch.from_numpy(np.ascontiguousarray(arrays[key], dtype=np.float32))
            for key in Data.KEYS
        }
        counts = {key: int(tensor.shape[0]) for key, tensor in self.tensors.items()}
        if len(set(counts.values())) != 1:
            _LOG.error("window count mismatch: %s", counts)
            raise ValueError(f"window count mismatch: {counts}")
        self.length = counts["state_history"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: tensor[index] for key, tensor in self.tensors.items()}

def _require_shapes(arrays: dict[str, np.ndarray], config: Config) -> None:
    count = int(arrays["state_history"].shape[0])
    expected = {
        Data.STATE_HISTORY: (count, config.history_len, config.state_dim),
        Data.ACTIONS: (count, config.history_len + config.horizon - 1, config.action_dim),
        Data.FUTURE_STATES: (count, config.horizon, config.state_dim),
    }
    for key, shape in expected.items():
        actual = tuple(arrays[key].shape)
        if actual != shape:
            _LOG.error("%s shape %s != %s", key, actual, shape)
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


    states = torch.cat([tensors[Data.STATE_HISTORY], tensors[Data.FUTURE_STATES]], dim=1)
    actions = tensors[Data.ACTIONS]
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

    state_history = batch[Data.STATE_HISTORY]
    actions = batch[Data.ACTIONS]
    future_states = batch[Data.FUTURE_STATES]

    history_len = state_history.shape[1]

    action_windows = sliding_windows(actions, history_len)
    latent_actions = _encode_sequence(action_encoder, action_windows)

    states = torch.cat([state_history, future_states], dim=1)
    state_windows = sliding_windows(states, history_len)

    initial_state = state_windows[:, 0]
    targets = state_windows[:, 1:] # the next 10 windows of states are the targets for the predictor


    latent_initial_state = state_encoder(initial_state)
    latent_targets = _encode_sequence(state_encoder, targets)

    predicted_latent_state = latent_initial_state
    predictions = []

    #Important! This uses recursion to predict the next latent state based on the previous latent state and the next action. This is done for the entire horizon of 10 steps.
    #Important because the model must be trained to work over the entire horizon, not just one step.
    for i in range(config.horizon):
        latent_action = latent_actions[:, i]
        predicted_latent_state = latent_predictor(
            predicted_latent_state,
            latent_action
        )

        predictions.append(predicted_latent_state)

    predicted = torch.stack(predictions, dim=1)
    # returns a weighted sum of the MSE (invariance), variance and covariance as a penalty
    loss, parts = losses.vicreg_loss(predicted, latent_targets, latent_initial_state, lam=config.vicreg_lam, mu=config.vicreg_mu, nu=config.vicreg_nu,)
    return loss, parts, latent_initial_state

def train_jepa(loader: DataLoader, config: Config) -> None:
    _ensure_logging()
    device = config.device
    n_windows = len(loader.dataset)
    n_batches = len(loader)
    _LOG.info(
        "training JEPA on %s: %d windows, %d batches/epoch, %d epochs, batch_size=%d, lr=%g",
        device, n_windows, n_batches, config.epochs, config.batch_size, config.lr,
    )
    _LOG.info("building state encoder")
    state_encoder = encoders.StateHistoryEncoder(config).to(device)
    _LOG.info("building action encoder")
    action_encoder = encoders.ActionHistoryEncoder(config).to(device)
    _LOG.info("building latent predictor")
    latent_predictor = predictor.LatentPredictor(
        latent_dim=config.latent_dim,
        action_dim=config.latent_dim,  # the action encoder also outputs latent_dim
        hidden_dim=config.latent_dim
    ).to(device)
    _LOG.info(
        "model parameters: state=%d action=%d predictor=%d",
        _param_count(state_encoder),
        _param_count(action_encoder),
        _param_count(latent_predictor),
    )

    params = [
        *state_encoder.parameters(),
        *action_encoder.parameters(),
        *latent_predictor.parameters(),
    ]

    _LOG.info("building Adam optimizer")
    optimizer = torch.optim.Adam(params, lr=config.lr) # pick the Adam optimizer
    _LOG.info("fitting state and action normalizer")
    norm = {k: v.to(device) for k, v in _fit_normalizer(loader.dataset.tensors).items()} # compute one mean and one s.d. per state and action feature
    _LOG.info(
        "normalizer ready: state_std_min=%.3g action_std_min=%.3g",
        float(norm["state_std"].min()),
        float(norm["action_std"].min()),
    )

    for epoch in range(config.epochs):
        _LOG.info("epoch %02d/%02d", epoch + 1, config.epochs)
        totals = {"loss": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0}
        steps = 0
        last_latent = None
        for batch_index, batch in enumerate(loader, start=1):
            try:
                batch = _on_device(batch, device)
                batch = _normalize(batch, norm) # standardize each feature
                loss, parts, s_t = _jepa_step(
                    batch, state_encoder, action_encoder, latent_predictor, config,
                )
                optimizer.zero_grad()
                loss.backward() # compute how to change every weight
                grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip) # caps the update from the line above
                optimizer.step() # apply the changes
                loss_value = loss.item() # add the batch's total loss to the running total for the epoch
            except Exception:
                _LOG.exception(
                    "epoch %02d batch %04d/%04d failed",
                    epoch + 1, batch_index, n_batches,
                )
                raise
            totals["loss"] += loss_value
            for key, value in parts.items():
                totals[key] += value # do the same for the three separate terms in parts (pred error, variance penalty and covar penalty)
            steps += 1
            last_latent = s_t.detach()
            _LOG.info(
                "epoch %02d batch %04d/%04d  n=%d  loss=%.4f pred=%.5f var=%.4f cov=%.4f grad=%.3f",
                epoch + 1,
                batch_index,
                n_batches,
                batch["state_history"].shape[0],
                loss_value,
                parts["pred"],
                parts["var"],
                parts["cov"],
                float(grad_norm),
            )
        if last_latent is None:
            _LOG.error("epoch %02d produced no batches", epoch + 1)
            raise RuntimeError("no training batches; check the dataset windows")
        _LOG.info("computing latent stats")
        stats = losses.latent_stats(last_latent)
        avg = {key: value / max(steps, 1) for key, value in totals.items()}
        _LOG.info(
            "epoch %02d done  batches=%d  loss=%.4f  pred=%.5f  var=%.4f  cov=%.4f  dim_std_min=%.4f  dead=%.0f%%",
            epoch + 1,
            steps,
            avg["loss"],
            avg["pred"],
            avg["var"],
            avg["cov"],
            stats["dim_std_min"],
            100 * stats["dead_frac"],
        )
        _LOG.info("checking representation collapse")
        losses.check_collapse(stats, last_latent) # warns if the summaries have all bnecome the same vector (representation collapse)

    path = Path("artifacts") / "jepa.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_encoder": state_encoder.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "predictor": latent_predictor.state_dict(),
            "norm": {key: value.detach().cpu() for key, value in norm.items()},
            "config": {
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "latent_dim": config.latent_dim,
                "history_len": config.history_len,
                "horizon": config.horizon,
            },
        },
        path,
    )
    _LOG.info("wrote %s (%d bytes)", path, path.stat().st_size)

# train a separate raw-state predictor on the same loader for comparison with the JEPA model
def train_baseline(loader: DataLoader, config: Config) -> None:
    _ensure_logging()
    _LOG.info(
        "training baseline on %s: %d windows, %d batches/epoch, %d epochs, batch_size=%d",
        config.device, len(loader.dataset), len(loader), config.epochs, config.batch_size,
    )
    _LOG.info("building baseline model")
    model = baseline.DirectStatePredictor(config).to(config.device)
    _LOG.info("baseline parameters: %d", _param_count(model))
    _LOG.info("fitting baseline normalizer")
    model.fit_normalizer(loader.dataset.tensors)
    _LOG.info("running baseline training loop")
    history = baseline.train_direct_predictor(model, loader, config)
    if history:
        _LOG.info("baseline training done: final loss=%.6f over %d epochs", history[-1], len(history))
    else:
        _LOG.warning("baseline training finished with an empty loss history")
    _LOG.info("measuring baseline prediction error vs horizon")
    errors = baseline.prediction_error_vs_horizon(model, loader)
    _LOG.info("baseline horizon errors: %s", np.array2string(errors, precision=4))
    path = Path("artifacts") / "prediction_error_vs_horizon.png"
    _LOG.info("creating artifacts directory %s", path.parent)
    path.parent.mkdir(exist_ok=True)
    _LOG.info("plotting prediction error to %s", path)
    baseline.plot_prediction_error_vs_horizon(errors, str(path), dt=config.dt)
    _LOG.info("wrote %s (%d bytes)", path, path.stat().st_size)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("jepa", "baseline"))
    args = parser.parse_args()

    _ensure_logging()
    _LOG.info("training %s", args.model)
    _LOG.info("loading config")
    config = Config() # start here
    _LOG.info("config loaded: %s", config)
    _LOG.info("setting seed to %s", config.seed)
    _seed(config.seed) # seeds the random number generator

    # reads every CSV in drone/data_sequences/rnd_sky_training. Logs are 100 Hz and dt is 0.05 s, so every 5th state is kept and the 4 motor thrusts are averaged over those rows. Each flight is states (n + 1, 18) as [x, y, z, vx, vy, vz, rotation matrix, wx, wy, wz] and actions (n, 4) as [FL, FR, BL, BR]
    _LOG.info("loading trajectories at dt=%s", config.dt)
    stride = trajectory_loader.stride_for(config.dt)
    _LOG.info("trajectory stride=%d", stride)
    trajectories = trajectory_loader.load_trajectories(stride=stride)
    state_rows = sum(int(traj["states"].shape[0]) for traj in trajectories)
    _LOG.info("loaded %d trajectories, %d state rows", len(trajectories), state_rows)

    # slides a window of length 10 plus a future horizon of 10 along every flight producing four arrays: state_history, action_history, future_states, and future_actions
    _LOG.info(
        "building dataset windows (history=%d, horizon=%d)",
        config.history_len, config.horizon,
    )
    arrays = Data.build_dataset(trajectories, history_len=config.history_len, horizon=config.horizon)
    _LOG.info(
        "built %d windows %s",
        int(arrays["state_history"].shape[0]),
        {key: tuple(value.shape) for key, value in arrays.items()},
    )
    _LOG.info("checking window shapes")
    _require_shapes(arrays, config) # check the shapes
    _LOG.info("building window dataset")
    window_dataset = WindowDataset(arrays)
    _LOG.info("building dataloader (batch_size=%d, shuffle=True)", config.batch_size)
    loader = DataLoader( # hands out shuffled batches of 32
        window_dataset,
        batch_size=config.batch_size,
        shuffle=True)
    _LOG.info("dataloader ready: %d batches", len(loader))

    if args.model == "jepa":
        _LOG.info("starting JEPA training")
        train_jepa(loader, config)
    else:
        _LOG.info("starting baseline training")
        train_baseline(loader, config)
    _LOG.info("training finished")

if __name__ == "__main__":
    main()
