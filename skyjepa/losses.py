"""JEPA latent prediction loss, given batch size D = 16 latent dimensions"""

import torch
import torch.nn.functional as F


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

def _require_rollout(predicted: torch.Tensor, target: torch.Tensor) -> None:
    """Both tensors are the predictor rollout: (B, T, D)."""
    if predicted.ndim != 3 or predicted.shape != target.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} "
            "must both be (B, T, D) and match"
        )


def _require_state_latent(s_t: torch.Tensor, target: torch.Tensor) -> None:
    """s_t is the current latent (B, D) and lines up with target (B, T, D)."""
    expected = (target.shape[0], target.shape[-1])
    if tuple(s_t.shape) != expected:
        raise ValueError(
            f"s_t {tuple(s_t.shape)} must be (B, D)={expected} "
            f"to match target {tuple(target.shape)}"
        )


def vicreg_loss(predicted: torch.Tensor, target: torch.Tensor, s_t, lam = 25.0, mu = 25.0, nu = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """
    VICReg loss = lambda * L_inv + mu * L_var + nu * L_cov
    """
    _require_rollout(predicted, target)
    _require_state_latent(s_t, target)
    inv = F.mse_loss(predicted, target) # Invariance: computed once, comparing the two embeddings
    embeddings = torch.cat([s_t.unsqueeze(1), target], dim=1)  # (B, T+1, D)

    # variance and covariance: computed independently for each embedding
    var = variance_loss(embeddings)
    cov = covariance_loss(embeddings)

    total = lam * inv + mu * var + nu * cov
    return total, {"pred": inv.detach().item(), "var": var.detach().item(), "cov": cov.detach().item()}

def sigreg(
    latents: torch.Tensor,
    num_projections: int = 64,
    num_knots: int = 17,
    integration_limit: float = 5.0,
) -> torch.Tensor:
    """L_SIGReg for latents shaped ``[time, batch, latent_dim]``.

    SkyJEPA eqs. 10-12. Random 1-D projections are matched to N(0, 1) with an
    Epps-Pulley characteristic-function discrepancy. The integral uses a fixed
    symmetric grid; ``num_knots=17`` matches the paper.
    """
    if latents.ndim != 3:
        raise ValueError("latents must have shape [time, batch, latent_dim]")
    if num_projections < 1 or num_knots < 3:
        raise ValueError("num_projections must be positive and num_knots >= 3")
    _, _, latent_dim = latents.shape
    directions = torch.randn(
        num_projections,
        latent_dim,
        device=latents.device,
        dtype=latents.dtype,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    projected = torch.einsum("tbd,md->tbm", latents, directions)

    t = torch.linspace(
        -integration_limit,
        integration_limit,
        num_knots,
        device=latents.device,
        dtype=latents.dtype,
    )
    phase = projected.unsqueeze(-1) * t
    empirical_real = phase.cos().mean(dim=1)
    empirical_imag = phase.sin().mean(dim=1)
    target_real = torch.exp(-0.5 * t.square())
    discrepancy = (empirical_real - target_real).square() + empirical_imag.square()

    # Gaussian weighting gives an integrable Epps-Pulley statistic.
    weight = torch.exp(-0.5 * t.square())
    return torch.trapezoid(discrepancy * weight, t, dim=-1).mean()

def sigreg_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.02,
    num_projections: int = 64,
    num_knots: int = 17,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SkyJEPA objective, eq. 13: L_pred + λ_sig * L_SIGReg.

    ``predicted`` and ``target`` are ``(B, T, D)``. SIGReg is applied to the
    predicted rollout only, transposed to the paper's ``(T, B, D)`` layout.
    """
    _require_rollout(predicted, target)
    pred = latent_prediction_loss(predicted, target)
    # (B, T, D) -> (T, B, D). sigreg averages the characteristic function over batch.
    reg = sigreg(predicted.transpose(0, 1), num_projections=num_projections, num_knots=num_knots)
    total = pred + lam * reg
    return total, {"pred": pred.detach().item(), "sigreg": reg.detach().item()}

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