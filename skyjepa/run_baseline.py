"""Train the direct state-prediction baseline on drone logs and plot error vs horizon.

    uv run python -m skyjepa.run_baseline

Windows come from the shared `build_dataset` with the JEPA's history length and
horizon from `Config`. Whole trajectories are held out for evaluation. Two
variants are trained on the same windows: one-step (teacher-forced) loss, and
loss through the full T-step recursive rollout (the JEPA's training horizon).
"""

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from skyjepa import baseline
from skyjepa.config import Config
from skyjepa.data import dataset, drone
from skyjepa.train_jepa import WindowDataset


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=drone.DEFAULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = replace(Config(), epochs=args.epochs, seed=args.seed)
    _seed(config.seed)

    trajectories = drone.load_trajectories(args.data_dir, stride=drone.stride_for(config.dt))
    config = replace(
        config,
        state_dim=trajectories[0]["states"].shape[1],
        action_dim=trajectories[0]["actions"].shape[1],
    )
    train, test = drone.split_trajectories(trajectories, args.test_fraction, config.seed)
    windows = {
        name: dataset.build_dataset(
            split,
            history_len=config.history_len,
            horizon=config.horizon,
        )
        for name, split in (("train", train), ("test", test))
    }
    print(
        f"{len(train)} train / {len(test)} test trajectories, "
        f"{len(windows['train']['state_history'])} / {len(windows['test']['state_history'])} windows, "
        f"H={config.history_len} T={config.horizon} dt={config.dt}s"
    )
    train_loader = DataLoader(
        WindowDataset(windows["train"]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(WindowDataset(windows["test"]), batch_size=1024)

    variants = {
        "one-step loss": 1,
        f"{config.horizon}-step rollout loss": config.horizon,
    }
    state_errors = {}
    position_errors = {}
    for label, train_steps in variants.items():
        _seed(config.seed)
        model = baseline.DirectStatePredictor(config, hidden_dim=args.hidden_dim).to(config.device)
        model.fit_normalizer(windows["train"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"--- direct predictor, {label} ({n_params} parameters)")
        baseline.train_direct_predictor(model, train_loader, config, train_steps=train_steps)
        state_errors[label] = baseline.prediction_error_vs_horizon(model, test_loader)
        position_errors[label] = baseline.prediction_error_vs_horizon(
            model, test_loader, dims=slice(0, 3)
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.out_dir / "baseline_error_vs_horizon.png"
    baseline.plot_prediction_error_vs_horizon(
        {f"direct, {label}": curve for label, curve in state_errors.items()},
        str(plot_path),
        dt=config.dt,
        title="Direct state prediction: held-out error vs rollout horizon",
    )
    table_path = args.out_dir / "baseline_error_vs_horizon.csv"
    with open(table_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["step", "time_s"]
            + [f"state_l2[{label}]" for label in variants]
            + [f"position_m[{label}]" for label in variants]
        )
        for k in range(config.horizon):
            writer.writerow(
                [k + 1, round((k + 1) * config.dt, 6)]
                + [f"{state_errors[label][k]:.6f}" for label in variants]
                + [f"{position_errors[label][k]:.6f}" for label in variants]
            )

    print("step  " + "  ".join(f"{label:>24s}" for label in variants))
    for k in range(config.horizon):
        print(f"{k + 1:4d}  " + "  ".join(f"{state_errors[label][k]:24.4f}" for label in variants))
    print(f"saved {plot_path} and {table_path}")


if __name__ == "__main__":
    main()
