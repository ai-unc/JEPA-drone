"""Planar point mass.

State is [x, y, vx, vy]. Action is [ax, ay].
Trajectory layout is fixed in docs/tensor_shapes.md.
"""

import numpy as np


class PointMass:
    def __init__(self, dt: float) -> None:
        self.dt = dt

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Return the initial state, shape (4,)."""

        raise NotImplementedError("PointMass.reset")

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply an action of shape (2,) and return the next state, shape (4,)."""

        raise NotImplementedError("PointMass.step")


def generate_trajectories(
    n: int,
    n_steps: int,
    dt: float,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Return `n` float32 trajectories.

    Each dict has:
      states  (n_steps + 1, 4)
      actions (n_steps, 2)
    actions[i] is applied to states[i] and produces states[i + 1].
    """

    raise NotImplementedError("generate_trajectories")
