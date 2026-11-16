"""Physics-inspired prober, SkyJEPA Sec. IV-C (eqs. 14-16).

An MLP maps each latent to a translational acceleration residual dv (3) and an
angular acceleration map K (3x4). A differentiable kinematic integrator turns
them into physical states.

Conventions match the simulator logs (checked against rnd_configs data):
  world frame is NED, so gravity is +g along world z
  R is body-to-world and rotor thrust points along -z_body, i.e. -R[:, :, 2]
  omega is in the body frame, so R_{t+1} = R_t exp([omega]x dt)
  actions are rotor forces in newtons at the nominal thrust coefficient
"""

import torch
from torch import nn

MASS = 1.22  # SimulationConfigs.DEFAULT_TOTAL_MASS; the randomized per-flight mass is never used
GRAVITY = 9.80665  # physics_model.STANDARD_GRAVITY


def skew(w: torch.Tensor) -> torch.Tensor:
    """(B, 3) -> (B, 3, 3) with skew(w) @ u == cross(w, u)."""

    x, y, z = w.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).unflatten(-1, (3, 3))


def integrate(state, forces, dv, K, dt):
    """Euler-integrate eq. 16 from x_t.

    state (B, 18) is x_t; forces (B, T, 4), dv (B, T, 3) and K (B, T, 3, 4) drive
    step t+k -> t+k+1. Returns x_{t+1} .. x_{t+T} as (B, T, 18).
    """

    p, v, R, w = state[:, :3], state[:, 3:6], state[:, 6:15].unflatten(1, (3, 3)), state[:, 15:]
    g = state.new_tensor([0.0, 0.0, GRAVITY])
    states = []
    for k in range(forces.shape[1]):
        f = forces[:, k]
        acc = -f.sum(1, keepdim=True) / MASS * R[:, :, 2] + g + dv[:, k]  # eq. 15 in NED
        p, v, R, w = (
            p + v * dt,
            v + acc * dt,
            R @ torch.linalg.matrix_exp(skew(w) * dt),
            w + (K[:, k] @ f.unsqueeze(-1)).squeeze(-1) * dt,
        )
        states.append(torch.cat([p, v, R.flatten(1), w], 1))
    return torch.stack(states, 1)


class PhysicsProber(nn.Module):
    def __init__(self, latent_dim: int, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3 + 3 * 4),  # dv and K
        )
        # Start from nominal physics (dv = 0, K = 0) so training only learns corrections.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, latents, state, forces):
        """latents (B, T, latent_dim) are s~_{t+k} for k = 0..T-1 (eq. 14), state is x_t
        (B, 18), forces are a_t .. a_{t+T-1} (B, T, 4). Returns (B, T, 18)."""

        out = self.mlp(latents)
        return integrate(state, forces, out[..., :3], out[..., 3:].unflatten(-1, (3, 4)), self.dt)
