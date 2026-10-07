"""Shared sizes and training defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Hyperparameters every module should import.

    Tensor layouts that go with these numbers are in docs/tensor_shapes.md.
    """

    state_dim: int = 4
    action_dim: int = 2
    latent_dim: int = 24
    history_len: int = 10
    horizon: int = 10
    dt: float = 0.05
    n_trajectories: int = 100
    n_steps: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 10
    seed: int = 0
    device: str = "cpu"

    @property
    def H(self) -> int:
        return self.history_len

    @property
    def T(self) -> int:
        return self.horizon
