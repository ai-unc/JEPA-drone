"""Stage 2: train the physics-inspired prober on a frozen JEPA (paper Sec. IV-C).

    uv run python -m skyjepa.train_prober
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from skyjepa.data import trajectory_loader
from skyjepa.data.dataset import Data
from skyjepa.models.physics_prober import PhysicsProber
from skyjepa.run_jepa import load_jepa
from skyjepa.train import WindowDataset, _encode_sequence, _normalize, _seed
from skyjepa.windows import sliding_windows


def predict(batch, jepa, prober, norm, history_len):
    """Roll the frozen JEPA forward in latent space and decode with the prober. Returns (B, T, 18)."""

    state_encoder, action_encoder, latent_predictor = jepa
    with torch.no_grad():  # stop-gradient on the latents
        x = _normalize(batch, norm)
        z = _encode_sequence(action_encoder, sliding_windows(x[Data.ACTIONS], history_len))
        s = state_encoder(x[Data.STATE_HISTORY])
        latents = [s]
        for k in range(z.shape[1] - 1):
            s = latent_predictor(s, z[:, k])
            latents.append(s)
    # actions holds a_{t-H+1} .. a_{t+T-1}, so a_t is at index H-1. Step k uses s~_{t+k} and a_{t+k}.
    forces = batch[Data.ACTIONS][:, history_len - 1:]
    return prober(torch.stack(latents, 1), batch[Data.STATE_HISTORY][:, -1], forces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the physics prober on a frozen JEPA.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/jepa.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/prober.pt"))
    parser.add_argument("--data-dir", type=Path, default=trajectory_loader.DEFAULT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    *jepa, norm, config = load_jepa(args.checkpoint)
    _seed(config.seed)

    trajectories = trajectory_loader.load_trajectories(args.data_dir, trajectory_loader.stride_for(config.dt))
    train, test = trajectory_loader.split_trajectories(trajectories, args.test_fraction, config.seed)
    train_set, test_set = (
        WindowDataset(Data.build_dataset(split, config.history_len, config.horizon))
        for split in (train, test)
    )

    # Scale each state dimension by the std of its one-step change, so position, velocity,
    # rotation and angular velocity errors carry comparable weight in eq. 17.
    states = torch.cat(
        [train_set.tensors[Data.STATE_HISTORY][:, -1:], train_set.tensors[Data.FUTURE_STATES]], 1
    )
    scale = states.diff(dim=1).std(dim=(0, 1))

    prober = PhysicsProber(config.latent_dim, config.dt)
    optimizer = torch.optim.Adam(prober.parameters(), lr=config.lr)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            predicted = predict(batch, jepa, prober, norm, config.history_len)
            loss = ((predicted - batch[Data.FUTURE_STATES]) / scale).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch:02d}  loss={total / len(loader):.4f}")

    with torch.no_grad():
        batches = [
            (predict(batch, jepa, prober, norm, config.history_len), batch[Data.FUTURE_STATES])
            for batch in DataLoader(test_set, batch_size=1024)
        ]
    predicted, target = (torch.cat(t) for t in zip(*batches))
    position_rmse = (predicted[..., :3] - target[..., :3]).square().sum(-1).mean(0).sqrt()
    # attitude error is the geodesic angle between predicted and true rotation
    R_pred, R_true = (x[..., 6:15].unflatten(-1, (3, 3)) for x in (predicted, target))
    cos = ((R_pred.transpose(-1, -2) @ R_true).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    attitude_deg = torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).mean(0)
    print(f"held-out: {len(test)} trajectories, {len(test_set)} windows")
    print("step  time   pos RMSE (m)  attitude err (deg)")
    for k in range(config.horizon):
        print(f"{k + 1:4d}  {(k + 1) * config.dt:.2f}s  {position_rmse[k]:12.3f}  {attitude_deg[k]:18.2f}")

    torch.save({"prober": prober.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
