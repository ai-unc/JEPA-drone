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


class ProprioceptiveEncoder(nn.Module):

    def __init__(self, state_dim: int = 18, feature_dim: int = 32, hidden_dim: int = 64, latent_dim: int = 32, history_length: int = 10,):
        super().__init__()

        self.state_dim = state_dim
        self.history_length = history_length

        # extract features from individual timesteps
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
            nn.GELU(),
        )

        # capture temporal movement dynamics
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # project into proprioceptive latent space
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:

        if states.ndim != 3 or states.shape[1:] != (self.history_length, self.state_dim):
            raise ValueError("Invalid state history shape")

        # express positions relative to the latest position
        relative_pos = states[:, :, :3] - states[:, -1:, :3]

        states = torch.cat([relative_pos, states[:, :, 3:]], dim=-1)

        # per-timestep feature extraction
        features = self.mlp(states)

        # temporal encoding
        _, hidden = self.gru(features)

        # project final hidden state
        latent = self.projection(hidden[-1])

        return latent


class ImageEncoder(nn.Module):
    """image encoder for AirSim only (not to be used for IRL testing)"""

    def __init__(self, latent_dim: int = 32):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GELU()
            )

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32 * 8 * 8, latent_dim)
            )

    def forward(self, images: torch.Tensor):

        if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
            raise ValueError("Expected shape (B, 1, 64, 64)")

        features = self.cnn(images)
        latent = self.projection(features)

        return latent