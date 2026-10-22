"""Training stub. Wires the environment, dataset, encoders, predictor, loss, and baseline."""

import random
import time
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import torch

from skyjepa.data import dataset, trajectory_loader
from skyjepa.logging.logger import _LOG, _ensure_logging, _ms, _sync, _timed
from skyjepa.models import encoders, predictor
from skyjepa.windows import rollout_windows
from skyjepa import baseline, losses
from skyjepa.config import Config

_WINDOW_KEYS = (
    "state_history",
    "action_history",
    "future_states",
    "future_actions",
)


def _param_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


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

def _jepa_step(batch: dict[str, torch.Tensor], state_encoder: nn.Module, action_encoder: nn.Module, latent_predictor: nn.Module, config: Config,) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, dict[str, float]]:
    # call rollout windows to cut these tensors into overlapping 10-step clips
    device = config.device
    phases: dict[str, float] = {}
    _sync(device)
    started = time.perf_counter()
    state_windows, action_windows = rollout_windows(
        batch["state_history"],
        batch["action_history"],
        batch["future_states"],
        batch["future_actions"],
    )
    _sync(device)
    phases["windows"] = time.perf_counter() - started
    started = time.perf_counter()
    s_t = state_encoder(state_windows[:, 0]) # turn the "now" clip into s_t
    _sync(device)
    phases["encode_state"] = time.perf_counter() - started
    started = time.perf_counter()
    targets = _encode_sequence(state_encoder, state_windows[:, 1:]) # turn the future clips into targets
    _sync(device)
    phases["encode_targets"] = time.perf_counter() - started
    started = time.perf_counter()
    z_seq = _encode_sequence(action_encoder, action_windows) # turn action clips into z_seq
    _sync(device)
    phases["encode_actions"] = time.perf_counter() - started
    started = time.perf_counter()
    predicted = latent_predictor.rollout(s_t, z_seq.transpose(0, 1)).transpose(0, 1) # guess the next 10 state summaries from s_t and z_seq
    _sync(device)
    phases["rollout"] = time.perf_counter() - started
    if predicted.shape != targets.shape:
        raise ValueError(
            f"rollout shape {tuple(predicted.shape)} != targets {tuple(targets.shape)}"
        )
    started = time.perf_counter()
    # returns a weighted sum of the MSE (invariance), variance and covariance as a penalty
    loss, parts = losses.vicreg_loss(predicted, targets, s_t, lam=config.vicreg_lam, mu=config.vicreg_mu, nu=config.vicreg_nu,)
    _sync(device)
    phases["vicreg"] = time.perf_counter() - started
    return loss, parts, s_t, phases

def train_jepa(loader: DataLoader, config: Config) -> None:
    _ensure_logging()
    device = config.device
    n_windows = len(loader.dataset)
    n_batches = len(loader)
    _LOG.info(
        "jepa  device=%s windows=%d batches/epoch=%d epochs=%d batch_size=%d lr=%g",
        device, n_windows, n_batches, config.epochs, config.batch_size, config.lr,
    )
    with _timed("jepa build state encoder"):
        state_encoder = encoders.StateHistoryEncoder(config).to(device)
    with _timed("jepa build action encoder"):
        action_encoder = encoders.ActionHistoryEncoder(config).to(device)
    with _timed("jepa build latent predictor"):
        latent_predictor = predictor.LatentPredictor(
            latent_dim=config.latent_dim,
            action_dim=config.latent_dim,  # the action encoder also outputs latent_dim
            hidden_dim=config.latent_dim
        ).to(device)
    _LOG.info(
        "jepa  params state=%d action=%d predictor=%d",
        _param_count(state_encoder),
        _param_count(action_encoder),
        _param_count(latent_predictor),
    )

    params = [
        *state_encoder.parameters(),
        *action_encoder.parameters(),
        *latent_predictor.parameters(),
    ]
    
    with _timed("jepa build Adam"):
        optimizer = torch.optim.Adam(params, lr=config.lr) # pick the Adam optimizer
    with _timed("jepa fit normalizer"):
        norm = {k: v.to(device) for k, v in _fit_normalizer(loader.dataset.tensors).items()} # compute one mean and one s.d. per state and action feature
    _LOG.info(
        "jepa  normalizer state_std_min=%.3g action_std_min=%.3g",
        float(norm["state_std"].min()),
        float(norm["action_std"].min()),
    )

    for epoch in range(config.epochs):
        _LOG.info("jepa  epoch %02d/%02d start", epoch, config.epochs)
        epoch_started = time.perf_counter()
        totals = {"loss": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0}
        steps = 0
        last_latent = None
        fetch_started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            _sync(device)
            fetch = time.perf_counter() - fetch_started
            try:
                started = time.perf_counter()
                batch = _on_device(batch, device)
                _sync(device)
                h2d = time.perf_counter() - started
                started = time.perf_counter()
                batch = _normalize(batch, norm) # standardize each feature
                _sync(device)
                norm_s = time.perf_counter() - started
                started = time.perf_counter()
                loss, parts, s_t, phases = _jepa_step(
                    batch, state_encoder, action_encoder, latent_predictor, config,
                )
                _sync(device)
                forward = time.perf_counter() - started
                started = time.perf_counter()
                optimizer.zero_grad()
                loss.backward() # compute how to change every weight
                _sync(device)
                backward = time.perf_counter() - started
                started = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip) # caps the update from the line above
                _sync(device)
                clip = time.perf_counter() - started
                started = time.perf_counter()
                optimizer.step() # apply the changes
                _sync(device)
                opt = time.perf_counter() - started
                started = time.perf_counter()
                loss_value = loss.item() # add the batch's total loss to the running total for the epoch
                _sync(device)
                item_s = time.perf_counter() - started
            except Exception:
                _LOG.exception(
                    "jepa  epoch %02d batch %04d/%04d failed",
                    epoch, batch_index, n_batches,
                )
                raise
            totals["loss"] += loss_value
            for key, value in parts.items():
                totals[key] += value # do the same for the three separate terms in parts (pred error, variance penalty and covar penalty)
            steps += 1
            last_latent = s_t.detach()
            total = fetch + h2d + norm_s + forward + backward + clip + opt + item_s
            _LOG.info(
                "jepa  epoch %02d batch %04d/%04d  n=%d  "
                "fetch=%s h2d=%s norm=%s windows=%s enc_s=%s enc_tgt=%s enc_a=%s "
                "rollout=%s vicreg=%s fwd=%s bwd=%s clip=%s opt=%s item=%s total=%s  "
                "loss=%.4f pred=%.5f var=%.4f cov=%.4f grad=%.3f",
                epoch,
                batch_index,
                n_batches,
                batch["state_history"].shape[0],
                _ms(fetch),
                _ms(h2d),
                _ms(norm_s),
                _ms(phases["windows"]),
                _ms(phases["encode_state"]),
                _ms(phases["encode_targets"]),
                _ms(phases["encode_actions"]),
                _ms(phases["rollout"]),
                _ms(phases["vicreg"]),
                _ms(forward),
                _ms(backward),
                _ms(clip),
                _ms(opt),
                _ms(item_s),
                _ms(total),
                loss_value,
                parts["pred"],
                parts["var"],
                parts["cov"],
                float(grad_norm),
            )
            fetch_started = time.perf_counter()
        if last_latent is None:
            _LOG.error("jepa  epoch %02d produced no batches", epoch)
            raise RuntimeError("no training batches; check the dataset windows")
        with _timed(f"jepa epoch {epoch:02d} latent stats"):
            stats = losses.latent_stats(last_latent)
        avg = {key: value / max(steps, 1) for key, value in totals.items()}
        _LOG.info(
            "jepa  epoch %02d done  %.3fs  batches=%d  loss=%.4f  pred=%.5f  "
            "var=%.4f  cov=%.4f  dim_std_min=%.4f  dead=%.0f%%",
            epoch,
            time.perf_counter() - epoch_started,
            steps,
            avg["loss"],
            avg["pred"],
            avg["var"],
            avg["cov"],
            stats["dim_std_min"],
            100 * stats["dead_frac"],
        )
        with _timed(f"jepa epoch {epoch:02d} collapse check"):
            losses.check_collapse(stats, last_latent) # warns if the summaries have all bnecome the same vector (representation collapse)

# train a separate raw-state predictor on the same loader for comparison with the JEPA model
def train_baseline(loader: DataLoader, config: Config) -> None:
    _ensure_logging()
    _LOG.info(
        "baseline  device=%s windows=%d batches=%d epochs=%d batch_size=%d",
        config.device, len(loader.dataset), len(loader), config.epochs, config.batch_size,
    )
    with _timed("baseline build model"):
        model = baseline.DirectStatePredictor(config).to(config.device)
    _LOG.info("baseline  params=%d", _param_count(model))
    with _timed("baseline fit normalizer"):
        model.fit_normalizer(loader.dataset.tensors)
    with _timed("baseline train"):
        history = baseline.train_direct_predictor(model, loader, config)
    if history:
        _LOG.info("baseline  final train loss=%.6f over %d epochs", history[-1], len(history))
    with _timed("baseline prediction error vs horizon"):
        errors = baseline.prediction_error_vs_horizon(model, loader)
    _LOG.info("baseline  horizon errors=%s", np.array2string(errors, precision=4))
    path = Path("artifacts") / "prediction_error_vs_horizon.png"
    with _timed("baseline make artifacts dir"):
        path.parent.mkdir(exist_ok=True)
    with _timed(f"baseline plot {path}"):
        baseline.plot_prediction_error_vs_horizon(errors, str(path), dt=config.dt)
    _LOG.info("baseline  wrote %s (%d bytes)", path, path.stat().st_size)

def main() -> None:
    _ensure_logging()
    run_started = time.perf_counter()
    _LOG.info("main  start")
    with _timed("load config"):
        config = Config() # start here
    _LOG.info("config  %s", config)
    with _timed(f"seed {config.seed}"):
        _seed(config.seed) # seeds the random number generator

    # reads every CSV in drone/data_sequences/rnd_sky_training. Logs are 100 Hz and dt is 0.05 s, so every 5th state is kept and the 4 motor thrusts are averaged over those rows. Each flight is states (n + 1, 18) as [x, y, z, vx, vy, vz, rotation matrix, wx, wy, wz] and actions (n, 4) as [FL, FR, BL, BR]
    with _timed("load trajectories"):
        stride = trajectory_loader.stride_for(config.dt)
        _LOG.info("trajectories  dt=%s stride=%d", config.dt, stride)
        trajectories = trajectory_loader.load_trajectories(stride=stride)
    state_rows = sum(int(traj["states"].shape[0]) for traj in trajectories)
    _LOG.info("trajectories  n=%d state_rows=%d", len(trajectories), state_rows)

    # slides a window of length 10 plus a future horizon of 10 along every flight producing four arrays: state_history, action_history, future_states, and future_actions
    with _timed("build dataset"):
        arrays = dataset.build_dataset(trajectories, history_len=config.history_len, horizon=config.horizon)
    _LOG.info(
        "windows  n=%d shapes=%s",
        int(arrays["state_history"].shape[0]),
        {key: tuple(value.shape) for key, value in arrays.items()},
    )
    with _timed("check window shapes"):
        _require_shapes(arrays, config) # check the shapes
    with _timed("build window dataset"):
        window_dataset = WindowDataset(arrays)
    with _timed("build dataloader"):
        loader = DataLoader( # hands out shuffled batches of 32
            window_dataset,
            batch_size=config.batch_size,
            shuffle=True)
    _LOG.info("loader  batches=%d batch_size=%d", len(loader), config.batch_size)

    with _timed("train jepa"):
        train_jepa(loader, config) # train JEPA
    with _timed("train baseline"):
        train_baseline(loader, config) # train the MLP that JEPA is to be compared against
    _LOG.info("main  done  %.3fs", time.perf_counter() - run_started)

if __name__ == "__main__":
    main()
