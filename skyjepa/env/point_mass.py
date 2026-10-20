"""Planar point mass.

State is [x, y, vx, vy]. Action is [ax, ay].
Trajectory layout is fixed in docs/tensor_shapes.md.
"""

import numpy as np


class PointMass:
    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.state = np.zeros(4, dtype=np.float32)
        self._rng = np.random.default_rng()
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        position = self._rng.uniform(-1.0, 1.0, size=2)
        velocity = self._rng.normal(0.0, 0.1, size=2)
        self.state = np.concatenate([position, velocity]).astype(np.float32)
        return self.state.copy()
    def step(self, action: np.ndarray) -> np.ndarray:
        ax, ay = np.asarray(action, dtype=np.float32)
        x, y, vx, vy = self.state
        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        x = x + vx * self.dt
        y = y + vy * self.dt
        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        return self.state.copy()


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

    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(n):
        env = PointMass(dt)
        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        states = [state]
        actions = []
        for _step in range(n_steps):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            actions.append(action)
            states.append(env.step(action))
        trajectories.append({
            "states": np.stack(states),
            "actions": np.stack(actions)
        })
    return trajectories
