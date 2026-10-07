"""Latent dynamics.

Use a single nn.GRUCell. z_t is the input, s_t is the hidden state.
"""

import torch
from torch import nn

from skyjepa.config import Config


class LatentPredictor(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, s_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """One step. s_t (B, latent_dim), z_t (B, latent_dim) -> s_{t+1} (B, latent_dim)."""

        raise NotImplementedError("LatentPredictor.forward")

    def rollout(self, s_t: torch.Tensor, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq (B, T, latent_dim) -> s̃_{t+1:t+T} with shape (B, T, latent_dim)."""

        raise NotImplementedError("LatentPredictor.rollout")
