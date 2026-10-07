"""Direct state-prediction baseline."""

import numpy as np
import torch
from torch import nn

from skyjepa.config import Config


class DirectStatePredictor(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future states in one shot. Returns (B, T, state_dim)."""

        raise NotImplementedError("DirectStatePredictor")


def plot_prediction_error_vs_horizon(errors: np.ndarray, path: str) -> None:
    """Save error against horizon.

    `errors` has shape (T,). errors[k] is the mean L2 error of x_{t+1+k}.
    """

    raise NotImplementedError("plot_prediction_error_vs_horizon")
