"""Load simulator flight logs as trajectories.

The CSVs in drone/data_sequences are written by the simulator at 100 Hz. Row i
holds the state x_i read before the step and the motor thrusts a_i sent at
that step, so a_i takes x_i to x_{i+1}. The last row's action has no recorded
result and is dropped.

Each trajectory uses the layout in docs/tensor_shapes.md:
  states  (n_steps + 1, 18)  [x, y, z, vx, vy, vz, R (row-major), wx, wy, wz]
  actions (n_steps, 4)       [FL, FR, BL, BR] thrusts in newtons
"""

from pathlib import Path

import numpy as np
import pandas as pd

LOG_HZ = 100
STATE_COLUMNS = (
    "x", "y", "z",
    "vx", "vy", "vz",
    "r1x", "r1y", "r1z",
    "r2x", "r2y", "r2z",
    "r3x", "r3y", "r3z",
    "wx", "wy", "wz",
)
ACTION_COLUMNS = ("FL", "FR", "BL", "BR")
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "drone" / "data_sequences" / "rnd_sky_training"


def stride_for(dt: float) -> int:
    """Log rows per model step for a model step of `dt` seconds."""

    stride = round(dt * LOG_HZ)
    if stride < 1 or abs(stride - dt * LOG_HZ) > 1e-6:
        raise ValueError(f"dt={dt} is not a whole number of {LOG_HZ} Hz log rows")
    return stride


def load_trajectory(path: str | Path, stride: int = 1) -> dict[str, np.ndarray]:
    """Read one log, keeping every `stride`-th state.

    The action for a coarse step is the mean thrust over the `stride` log rows
    it spans, so the total impulse between kept states is preserved.
    """

    frame = pd.read_csv(path, comment="#")
    states = frame[list(STATE_COLUMNS)].to_numpy(dtype=np.float32)
    actions = frame[list(ACTION_COLUMNS)].to_numpy(dtype=np.float32)
    n_steps = (len(frame) - 1) // stride
    return {
        "states": states[: n_steps * stride + 1 : stride],
        "actions": actions[: n_steps * stride].reshape(n_steps, stride, -1).mean(axis=1),
    }


def load_trajectories(folder: str | Path = DEFAULT_DIR, stride: int = 1) -> list[dict[str, np.ndarray]]:
    """Every *.csv in `folder`, in natural file order (rnd_sky2 before rnd_sky10)."""

    paths = sorted(
        Path(folder).glob("*.csv"),
        key=lambda path: (len(path.stem), path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"no CSV logs in {folder}")
    return [load_trajectory(path, stride) for path in paths]


def split_trajectories(
    trajectories: list[dict[str, np.ndarray]],
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """Hold out whole trajectories so no test window overlaps a training window."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(len(trajectories))
    n_test = max(1, round(len(trajectories) * test_fraction))
    test = [trajectories[i] for i in sorted(order[:n_test])]
    train = [trajectories[i] for i in sorted(order[n_test:])]
    return train, test
