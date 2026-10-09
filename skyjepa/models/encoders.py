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

        self.tcn = nn.Sequential(
            nn.Conv1d(config.state_dim, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(16, config.latent_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """(B, H, state_dim) -> (B, latent_dim)."""
        x = state_history.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.projection(x)


class ActionHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.tcn = nn.Sequential(
            nn.Conv1d(config.action_dim, 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(8, config.latent_dim)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """(B, H, action_dim) -> (B, latent_dim)."""
        x = action_history.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.projection(x)




