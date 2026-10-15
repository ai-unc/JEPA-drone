"""JEPA latent prediction loss."""

import torch


def per_horizon_losses(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss at each future step: L1, L2, ..., LT. Returns shape (T,)."""
    squared_error = (predicted - target) ** 2       # (B, T, D)
    per_sample = squared_error.sum(dim=-1)          # (B, T)  add up the D numbers
    return per_sample.mean(dim=0)                   # (T,)    average over the batch


def latent_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_pred for predicted and target shaped (B, T, latent_dim).

    Mean over the batch of (1/T) * sum_k ||predicted[:, k] - target[:, k]||_2^2.
    Returns a scalar.
    """
    return per_horizon_losses(predicted, target).mean()


def latent_stats(latent: torch.Tensor) -> dict[str, float]:
    """Mean and std of every element in `latent`.

    Keys: "mean", "std". std near 0 means the representation collapsed.
    Extra keys look at each latent dimension separately:
      "dim_std_mean": average of the per-dimension stds
      "dim_std_min":  the smallest per-dimension std
      "dead_frac":    fraction of dimensions with std below 0.01
    """
    with torch.no_grad():
        flat = latent.reshape(-1, latent.shape[-1])  # (N, D) one row per summary
        dim_std = flat.std(dim=0)                    # (D,)  one std per dimension
        return {
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "dim_std_mean": dim_std.mean().item(),
            "dim_std_min": dim_std.min().item(),
            "dead_frac": (dim_std < 0.01).float().mean().item(),
        }


def check_collapse(stats: dict[str, float], threshold: float = 0.1) -> bool:
    """Print a warning and return True if the latents look collapsed."""
    collapsed = stats["dim_std_mean"] < threshold
    if collapsed:
        print(
            f"[WARN] possible representation collapse: "
            f"avg latent std={stats['dim_std_mean']:.4f} < {threshold}, "
            f"dead dims={stats['dead_frac']:.0%}"
        )
    return collapsed


def log_jepa_step(predicted, target, s_t, threshold: float = 0.1) -> dict[str, float]:
    """One call per batch: per-horizon losses L1..LT, latent stats, and collapse warning."""
    out = {f"L{k + 1}": v for k, v in enumerate(per_horizon_losses(predicted, target).tolist())}
    stats = latent_stats(s_t)
    out.update(stats)
    out["collapsed"] = float(check_collapse(stats, threshold))
    return out


if __name__ == "__main__":
    B, T, D = 64, 10, 24 # fake tensors shaped like real latents
    good = torch.randn(B, T, D) # fills the tensor with random normal noise (mean = 0, s.d. = 1)
    bad = torch.zeros(B, T, D) + 1e-4 * torch.randn(B, T, D)

    print("good latents (should NOT warn):")
    print(check_collapse(latent_stats(good)))

    print("collapsed latents (SHOULD warn):")
    print(check_collapse(latent_stats(bad)))

    print("loss on collapsed model (should be ~0):", latent_prediction_loss(bad, bad).item())
    print("loss when every number is off by 1 (should be 24.0):",
          latent_prediction_loss(good, good + 1).item())
    print("per-horizon shape (should be torch.Size([10])):",
          per_horizon_losses(good, good + 1).shape)
    print("log_jepa_step keys:", list(log_jepa_step(good, good + 1, good[:, 0]).keys()))