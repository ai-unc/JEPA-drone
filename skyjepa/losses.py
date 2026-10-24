"""JEPA latent prediction loss, given batch size D = 16 latent dimensions"""

from numpy._core import bool_
import torch
import torch.nn.functional as F

from skyjepa import config


def variance_loss(target: torch.Tensor, gamma: float = 1.0, e: float = 1e-4):
    """calculate the mean of each column, then calculate each column's standard deviation,
    then finally penalize dimensions with standard deviation below gamma"""

    target = target.reshape(-1, target.shape[-1]) # (..., D) -> (N, D)
    N = target.shape[0]

    target_means = torch.mean(target, dim=0)
    target_col_sd = torch.sqrt(((target - target_means) ** 2).sum(dim=0) / (N - 1) + e)

    return F.relu(gamma - target_col_sd).mean()

def covariance_loss(target: torch.Tensor) -> torch.Tensor:
    """
    first center each column before calculating the covariance matrix then square and sum every off-diagonal entry
    """

    Z = target.reshape(-1, target.shape[-1]) # (..., D) -> (N, D)
    N, D = Z.shape

    Z_c = Z - Z.mean(dim=0) # center each column, (N, D)

    C = (Z_c.T @ Z_c) / (N - 1) # covariance matrix, (D, D)

    off_diag = C - torch.diag(torch.diag(C)) # zero out the diagonal
    return (off_diag ** 2).sum() / D

def vicreg_loss(predicted: torch.Tensor, target: torch.Tensor, s_t, lam = 25.0, mu = 25.0, nu = 1.0, display=False) -> tuple[torch.Tensor, dict[str, float]]:
    """
    VICReg loss = lambda * L_inv + mu * L_var + nu * L_cov
    """
    #inv = F.mse_loss(predicted, target) # Invariance: computed once, comparing the two embeddings

    per_step_mse = (predicted - target).square().mean(dim=(0, 2))
    if display:
        print("Final step loss:", per_step_mse[-1].item())

    step = torch.arange(
        1, predicted.shape[1] + 1,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    weights = step
    inv = (weights * per_step_mse).sum() / weights.sum()

    embeddings = torch.cat([s_t.unsqueeze(1), target], dim=1)  # (B, T+1, D)

    # variance and covariance: computed independently for each embedding
    var = variance_loss(embeddings)
    cov = covariance_loss(embeddings)

    total = lam * inv + mu * var + nu * cov
    return total, {"pred": inv.item(), "var": var.item(), "cov": cov.item()}

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

def _dimensional_collapse(stats: dict[str, float], threshold: float = 0.1) -> bool:
    """Checks for dimensional collpase and returns True if detected"""
    if stats["dim_std_mean"] < threshold:
        print(
            f"[WARNING] possible representation collapse: "
            f"avg latent std={stats['dim_std_mean']:.4f} < {threshold}, "
            f"dead dims={stats['dead_frac']:.0%}"
        )
    return stats["dim_std_mean"] < threshold

def _rank_collapse(state_encoded: torch.Tensor, min_frac: float = 0.5) -> bool:
    with torch.no_grad():
        z = state_encoded.reshape(-1, state_encoded.shape[-1]).float() # (N, D)
        n, d = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(n - 1, 1) # (D, D)
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
        stable_rank = (eigvals.sum() / eigvals.max().clamp_min(1e-8)).item()
        max_rank = min(n - 1, d) # centering drops one direction
        collapsed = stable_rank < min_frac * max_rank
    if collapsed:
        print(f"[WARNING] rank collapse: stable rank {stable_rank:.1f}/{max_rank}")
    return collapsed

def check_collapse(stats: dict[str, float], latent: torch.Tensor, threshold: float = 0.1) -> bool:
    dim_collapsed = _dimensional_collapse(stats, threshold)
    rank_collapsed = _rank_collapse(latent)
    return dim_collapsed or rank_collapsed
    

if __name__ == "__main__":
    B, T, D = 64, 10, 24 # fake tensors shaped like real latents
    good = torch.randn(B, T, D) # fills the tensor with random normal noise (mean = 0, s.d. = 1)
    bad = torch.zeros(B, T, D) + 1e-4 * torch.randn(B, T, D) # almost all 0 with a tiny bit of noise added

    # print("good latents (should NOT warn):")
    # print(check_collapse(latent_stats(good)))

    # print("collapsed latents (SHOULD warn):")
    # print(check_collapse(latent_stats(bad)))

    # print("loss on collapsed model (should be ~0):", latent_prediction_loss(bad, bad).item())
    # print("loss when every number is off by 1 (should be 24.0):",
    #       latent_prediction_loss(good, good + 1).item())
    # print("per-horizon shape (should be torch.Size([10])):",
    #       per_horizon_losses(good, good + 1).shape)
    # print("log_jepa_step keys:", list(log_jepa_step(good, good + 1, good[:, 0]).keys()))