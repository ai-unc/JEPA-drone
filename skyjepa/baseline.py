"""Direct state-prediction baseline.

A conventional autoregressive world model: one network maps the recent state
and action history straight to the next physical state,

    x_{t+1} = f(x_{t-H+1} .. x_t, a_{t-H+1} .. a_t),

and a T-step forecast feeds each prediction back in as the newest state:

    x_t -> x̂_{t+1} -> x̂_{t+2} -> ... -> x̂_{t+T}.

It reads the same three window tensors as the JEPA (`build_dataset` output) and
uses the same history length H and horizon T from `Config`, so the two models
are compared on the same data over the same rollout.
"""

import time
from collections.abc import Mapping

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from skyjepa.config import Config
from skyjepa.logging.logger import _LOG, _ms, _sync

# Categorical slots 1-8 of the reference chart palette, in fixed order.
_SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"


class DirectStatePredictor(nn.Module):
    """One-step MLP on the state/action history, rolled out recursively.

    The MLP predicts the normalized change x_{t+1} - x_t. Normalization
    statistics are buffers set by `fit_normalizer` from training windows; until
    then they are the identity.
    """

    def __init__(self, config: Config, hidden_dim: int = 128) -> None:
        super().__init__()
        self.config = config
        history, state_dim, action_dim = (
            config.history_len,
            config.state_dim,
            config.action_dim,
        )
        self.net = nn.Sequential(
            nn.Linear(history * (state_dim + action_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        for name, dim in (("state", state_dim), ("action", action_dim), ("delta", state_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    @torch.no_grad()
    def fit_normalizer(self, windows: Mapping[str, np.ndarray | torch.Tensor]) -> None:
        """Set input and output statistics from training windows.

        `windows` holds `state_history`, `actions` and `future_states`
        as laid out in docs/tensor_shapes.md.
        """

        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(np.asarray(windows[key]), dtype=torch.float32)

        state_history = tensor("state_history")
        actions = tensor("actions")
        delta = tensor("future_states")[:, 0] - state_history[:, -1]
        for name, values in (
            ("state", state_history.reshape(-1, state_history.shape[-1])),
            ("action", actions.reshape(-1, actions.shape[-1])),
            ("delta", delta),
        ):
            getattr(self, f"{name}_mean").copy_(values.mean(dim=0))
            getattr(self, f"{name}_std").copy_(values.std(dim=0).clamp_min(1e-6))

    def step(self, state_window: torch.Tensor, action_window: torch.Tensor) -> torch.Tensor:
        """state_window (B, H, state_dim), action_window (B, H, action_dim) -> x_{t+1} (B, state_dim).

        Both windows end at time t; action_window[:, -1] is a_t.
        """

        states = (state_window - self.state_mean) / self.state_std
        actions = (action_window - self.action_mean) / self.action_std
        features = torch.cat([states.flatten(1), actions.flatten(1)], dim=1)
        delta = self.net(features) * self.delta_std + self.delta_mean
        return state_window[:, -1] + delta

    def rollout(
        self,
        state_history: torch.Tensor,
        actions: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Predict x̂_{t+1} .. x̂_{t+steps} by feeding each prediction back in.

        actions: (B, H + T - 1, action_dim), from a_{t-H+1} through a_{t+T-1}.
        Returns (B, steps, state_dim). `steps` defaults to the full horizon.
        """

        history_len = state_history.shape[1]
        horizon = actions.shape[1] - history_len + 1
        steps = horizon if steps is None else steps
        if not 1 <= steps <= horizon:
            raise ValueError(f"steps={steps} must be in [1, {horizon}]")

        state_window = state_history
        predictions = []
        for k in range(steps):
            next_state = self.step(state_window, actions[:, k:k + history_len])
            predictions.append(next_state)
            state_window = torch.cat([state_window[:, 1:], next_state.unsqueeze(1)], dim=1)
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        state_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Recursive rollout over the full horizon. Returns (B, T, state_dim)."""

        return self.rollout(state_history, actions)


def rollout_loss(
    model: DirectStatePredictor,
    batch: Mapping[str, torch.Tensor],
    steps: int,
) -> torch.Tensor:
    """Mean squared error of a `steps`-long rollout, in units of the one-step delta std.

    steps=1 is ordinary one-step (teacher-forced) training. steps=T trains
    through the full recursive rollout, like the JEPA's T-step latent loss.
    """

    predicted = model.rollout(
        batch["state_history"],
        batch["actions"],
        steps=steps,
    )
    error = (predicted - batch["future_states"][:, :steps]) / model.delta_std
    return error.pow(2).mean()


def train_direct_predictor(
    model: DirectStatePredictor,
    loader: DataLoader,
    config: Config,
    train_steps: int = 1,
) -> list[float]:
    """Adam with the JEPA's lr and epoch count. Returns the mean loss per epoch."""

    device = config.device
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    model.train()
    n_batches = len(loader)
    _LOG.info(
        "baseline train loop  epochs=%d batches/epoch=%d train_steps=%d device=%s",
        config.epochs, n_batches, train_steps, device,
    )
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        total = 0.0
        count = 0
        fetch_started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            _sync(device)
            fetch = time.perf_counter() - fetch_started
            try:
                started = time.perf_counter()
                batch = {key: value.to(device) for key, value in batch.items()}
                _sync(device)
                h2d = time.perf_counter() - started
                started = time.perf_counter()
                loss = rollout_loss(model, batch, train_steps)
                _sync(device)
                forward = time.perf_counter() - started
                started = time.perf_counter()
                optimizer.zero_grad()
                loss.backward()
                _sync(device)
                backward = time.perf_counter() - started
                started = time.perf_counter()
                optimizer.step()
                _sync(device)
                opt = time.perf_counter() - started
                started = time.perf_counter()
                loss_value = float(loss.item())
                size = batch["state_history"].shape[0]
                _sync(device)
                item_s = time.perf_counter() - started
            except Exception:
                _LOG.exception(
                    "baseline  epoch %02d batch %04d/%04d failed",
                    epoch, batch_index, n_batches,
                )
                raise
            total += loss_value * size
            count += size
            _LOG.info(
                "baseline  epoch %02d batch %04d/%04d  n=%d  "
                "fetch=%s h2d=%s fwd=%s bwd=%s opt=%s item=%s total=%s  loss=%.6f",
                epoch,
                batch_index,
                n_batches,
                size,
                _ms(fetch),
                _ms(h2d),
                _ms(forward),
                _ms(backward),
                _ms(opt),
                _ms(item_s),
                _ms(fetch + h2d + forward + backward + opt + item_s),
                loss_value,
            )
            fetch_started = time.perf_counter()
        history.append(total / max(count, 1))
        elapsed = time.perf_counter() - epoch_started
        print(f"baseline epoch {epoch:02d}  loss={history[-1]:.6f}")
        _LOG.info(
            "baseline  epoch %02d done  %.3fs  windows=%d  loss=%.6f",
            epoch, elapsed, count, history[-1],
        )
    model.eval()
    return history


@torch.no_grad()
def prediction_error_vs_horizon(
    model: DirectStatePredictor,
    loader: DataLoader,
    dims: slice | None = None,
) -> np.ndarray:
    """errors[k] = mean over windows of ||x̂_{t+1+k} - x_{t+1+k}||_2. Shape (T,).

    `dims` restricts the norm to some state components, e.g. slice(0, 3) for
    drone position.
    """

    device = model.config.device
    model.eval()
    total = None
    count = 0
    n_batches = len(loader)
    _LOG.info("baseline eval  batches=%d dims=%s", n_batches, dims)
    fetch_started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        _sync(device)
        fetch = time.perf_counter() - fetch_started
        try:
            started = time.perf_counter()
            batch = {key: value.to(device) for key, value in batch.items()}
            _sync(device)
            h2d = time.perf_counter() - started
            started = time.perf_counter()
            predicted = model(
                batch["state_history"],
                batch["actions"],
            )
            _sync(device)
            forward = time.perf_counter() - started
            started = time.perf_counter()
            error = predicted - batch["future_states"]
            if dims is not None:
                error = error[..., dims]
            per_step = torch.linalg.vector_norm(error, dim=-1).sum(dim=0)
            total = per_step if total is None else total + per_step
            size = batch["state_history"].shape[0]
            count += size
            _sync(device)
            reduce = time.perf_counter() - started
        except Exception:
            _LOG.exception("baseline eval  batch %04d/%04d failed", batch_index, n_batches)
            raise
        _LOG.info(
            "baseline eval  batch %04d/%04d  n=%d  fetch=%s h2d=%s fwd=%s reduce=%s total=%s",
            batch_index,
            n_batches,
            size,
            _ms(fetch),
            _ms(h2d),
            _ms(forward),
            _ms(reduce),
            _ms(fetch + h2d + forward + reduce),
        )
        fetch_started = time.perf_counter()
    if total is None:
        _LOG.error("baseline eval produced no batches")
        raise RuntimeError("no evaluation batches")
    started = time.perf_counter()
    errors = (total / count).cpu().numpy()
    _sync(device)
    _LOG.info(
        "baseline eval done  windows=%d  to_numpy=%s",
        count, _ms(time.perf_counter() - started),
    )
    return errors


def plot_prediction_error_vs_horizon(
    errors: np.ndarray | Mapping[str, np.ndarray],
    path: str,
    dt: float | None = None,
    ylabel: str = "mean L2 state error",
    title: str = "Prediction error vs rollout horizon",
) -> None:
    """Save error against horizon.

    `errors` has shape (T,), or maps a curve label to a (T,) array so other
    models (e.g. the JEPA through its prober) can share the axes.
    errors[k] is the mean L2 error of x_{t+1+k}. `dt`, if given, labels the
    step length on the x axis.
    """

    curves = dict(errors) if isinstance(errors, Mapping) else {"direct state prediction": errors}
    if len(curves) > len(_SERIES_COLORS):
        raise ValueError(f"at most {len(_SERIES_COLORS)} curves")

    figure = Figure(figsize=(6.4, 4.0), facecolor=_SURFACE)
    axes = figure.add_subplot()
    axes.set_facecolor(_SURFACE)
    horizon = 0
    for (label, curve), color in zip(curves.items(), _SERIES_COLORS):
        curve = np.asarray(curve, dtype=float).reshape(-1)
        steps = np.arange(1, curve.size + 1)
        horizon = max(horizon, curve.size)
        axes.plot(steps, curve, color=color, linewidth=2, marker="o", markersize=5, label=label)
        axes.annotate(
            f"{curve[-1]:.3g}",
            (steps[-1], curve[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT_PRIMARY,
        )

    xlabel = "rollout step k (prediction of $x_{t+k}$)"
    if dt is not None:
        xlabel += f", {dt * 1000:g} ms per step"
    axes.set_xlabel(xlabel, color=_TEXT_SECONDARY)
    axes.set_ylabel(ylabel, color=_TEXT_SECONDARY)
    axes.set_title(title, color=_TEXT_PRIMARY, loc="left", fontsize=11)
    axes.set_xticks(np.arange(1, horizon + 1))
    axes.set_xlim(0.5, horizon + 0.9)
    axes.set_ylim(bottom=0)
    axes.grid(axis="y", color=_GRID, linewidth=0.8)
    axes.tick_params(colors=_TEXT_SECONDARY)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_GRID)
    if len(curves) > 1:
        axes.legend(frameon=False, labelcolor=_TEXT_PRIMARY)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=_SURFACE)
