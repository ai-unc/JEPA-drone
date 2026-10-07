"""JEPA latent prediction loss."""

import torch


def latent_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_pred for predicted and target shaped (B, T, latent_dim).

    Mean over the batch of (1/T) * sum_k ||predicted[:, k] - target[:, k]||_2^2.
    Returns a scalar.
    """

    raise NotImplementedError("latent_prediction_loss")


def latent_stats(latent: torch.Tensor) -> dict[str, float]:
    """Mean and std of every element in `latent`.

    Keys: "mean", "std". std near 0 means the representation collapsed.
    """

    raise NotImplementedError("latent_stats")
