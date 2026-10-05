"""History encoders.

Two MLPs. Each maps a history window to a latent_dim vector.
"""

import torch
from torch import nn

from skyjepa.config import Config


class StateHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """state_history (B, H, state_dim) -> (B, latent_dim)."""

        raise NotImplementedError("StateHistoryEncoder")


class ActionHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """action_history (B, H, action_dim) -> (B, latent_dim)."""

        raise NotImplementedError("ActionHistoryEncoder")
