"""SkyJEPA latent dynamics predictor.

Paper references: Section IV-A (eq. 8) and Section V-C.

    one step :  s~_{t+1} = Pred(s_t, z_t)
    rollout  :  s~_{t+T} = Pred(... Pred(Pred(s_t, z_t), z_{t+1}) ..., z_{t+T-1})

s_t is the state latent from Enc_theta (state history encoder).
z_t is the action latent from Enc_phi (action history encoder).
"""

from typing import Optional

import torch
from torch import Tensor, nn


class LatentPredictor(nn.Module):
    """One-step latent predictor built on nn.GRUCell.

    The action latent z_t is the GRU input.
    The state latent s_t is the GRU hidden state.

    If latent_dim is not equal to hidden_dim, two Linear layers convert
    between the two sizes. If the sizes are equal, the Linear layers are
    not used and s_t is the hidden state directly.
    """

    def __init__(
        self,
        latent_dim: int = 16,   # last channel of the state TCN [8, 8, 16]
        action_dim: int = 8,    # last channel of the action TCN [4, 4, 8]
        hidden_dim: int = 24,   # GRU hidden size from the paper
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        same_size = latent_dim == hidden_dim
        self.in_proj = nn.Identity() if same_size else nn.Linear(latent_dim, hidden_dim)
        self.cell = nn.GRUCell(input_size=action_dim, hidden_size=hidden_dim)
        self.out_proj = nn.Identity() if same_size else nn.Linear(hidden_dim, latent_dim)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        """Predict one step.

        s: (B, latent_dim)   state latent at time t
        z: (B, action_dim)   action latent at time t
        returns: (B, latent_dim)   predicted state latent at time t+1
        """
        h = self.cell(z, self.in_proj(s))
        return self.out_proj(h)

    def rollout(self, s0: Tensor, z_seq: Tensor, T: Optional[int] = None) -> Tensor:
        """Predict T steps. Each step uses the prediction from the step before.

        s0:    (B, latent_dim) or (1, latent_dim)
               State latent at time t. Shape (1, latent_dim) is for MPPI:
               all samples start from the same state.
        z_seq: (T_max, B, action_dim)
               z_seq[k] is the action latent z_{t+k}.
        T:     number of steps. Default: all steps in z_seq.
        returns: (T, B, latent_dim)
               out[k] is the predicted latent s~_{t+k+1}.
        """
        steps = z_seq.shape[0] if T is None else T
        if steps > z_seq.shape[0]:
            raise ValueError(f"T={steps} is larger than the length of z_seq ({z_seq.shape[0]}).")

        batch = z_seq.shape[1]
        s = s0.expand(batch, -1) if s0.shape[0] != batch else s0

        preds = []
        for k in range(steps):
            # Do not detach s here. The loss must go back through all steps.
            s = self(s, z_seq[k])
            preds.append(s)
        return torch.stack(preds, dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, DZ = 32, 20, 16, 8
    pred = GRUPredictor(latent_dim=D, action_dim=DZ, hidden_dim=24)
    print("parameters:", sum(p.numel() for p in pred.parameters()))

    # 1. Shapes.
    s0 = torch.randn(B, D, requires_grad=True)
    z_seq = torch.randn(T, B, DZ)
    out = pred.rollout(s0, z_seq)
    assert out.shape == (T, B, D)
    assert pred.rollout(s0, z_seq, T=15).shape == (15, B, D)

    # 2. The rollout is the same as a manual loop of one-step calls.
    s = s0
    for k in range(T):
        s = pred(s, z_seq[k])
    assert torch.allclose(s, out[-1], atol=1e-6)

    # 3. The gradient of the last step gets to s0 (no detach in the loop).
    out[-1].sum().backward()
    assert s0.grad is not None and s0.grad.abs().sum() > 0

    # 4. MPPI use: one start state, 512 samples, no gradient.
    with torch.no_grad():
        mppi_out = pred.rollout(torch.randn(1, D), torch.randn(15, 512, DZ))
    assert mppi_out.shape == (15, 512, D)

    print("all checks passed")
