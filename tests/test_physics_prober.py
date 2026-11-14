"""Tests for the physics-inspired prober and its training path.

    uv run pytest tests/test_physics_prober.py
"""

import math

import numpy as np
import torch

from skyjepa.config import Config
from skyjepa.data.dataset import Data
from skyjepa.models import encoders, predictor
from skyjepa.models.physics_prober import GRAVITY, MASS, PhysicsProber, integrate
from skyjepa.train_prober import predict

DT = 0.05
B, T = 4, 10


def level_state(w=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Identity attitude at rest, shape (1, 18)."""

    return torch.cat([torch.zeros(6), torch.eye(3).flatten(), torch.tensor(w)]).unsqueeze(0)


def no_residual(steps: int):
    return torch.zeros(1, steps, 3), torch.zeros(1, steps, 3, 4)


def make_jepa(config: Config):
    torch.manual_seed(0)
    jepa = (
        encoders.StateHistoryEncoder(config),
        encoders.ActionHistoryEncoder(config),
        predictor.LatentPredictor(config.latent_dim, config.latent_dim, config.latent_dim),
    )
    norm = {
        "state_mean": torch.zeros(config.state_dim), "state_std": torch.ones(config.state_dim),
        "action_mean": torch.zeros(config.action_dim), "action_std": torch.ones(config.action_dim),
    }
    return jepa, norm


def test_output_shape():
    prober = PhysicsProber(24, DT)
    out = prober(torch.randn(B, T, 24), level_state().repeat(B, 1), torch.rand(B, T, 4))
    assert out.shape == (B, T, 18)


def test_free_fall_follows_gravity_down():
    # NED: with no thrust the drone accelerates toward +z.
    out = integrate(level_state(), torch.zeros(1, T, 4), *no_residual(T), DT)
    k = torch.arange(1, T + 1)
    assert torch.allclose(out[0, :, 5], GRAVITY * DT * k)
    assert torch.allclose(out[0, :, 2], GRAVITY * DT**2 * k * (k - 1) / 2)  # explicit Euler
    assert torch.allclose(out[0, :, [0, 1, 3, 4]], torch.zeros(T, 4))


def test_hover_thrust_cancels_gravity():
    forces = torch.full((1, T, 4), MASS * GRAVITY / 4)
    out = integrate(level_state(), forces, *no_residual(T), DT)
    assert torch.allclose(out[0, :, :6], torch.zeros(T, 6), atol=1e-5)


def test_thrust_points_along_minus_body_z():
    # Roll 90 deg about x: body z maps to world -y, so thrust (-z_body) pushes toward world +y.
    state = level_state()
    state[0, 6:15] = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]).flatten()
    out = integrate(state, torch.full((1, 1, 4), 2.0), *no_residual(1), DT)
    assert torch.allclose(out[0, 0, 3:6], torch.tensor([0.0, 8 / MASS * DT, GRAVITY * DT]))


def test_body_rate_rotation_matches_closed_form():
    wz = 2.0
    out = integrate(level_state((0.0, 0.0, wz)), torch.zeros(1, T, 4), *no_residual(T), DT)
    angle = wz * DT * T
    c, s = math.cos(angle), math.sin(angle)
    expected = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    assert torch.allclose(out[0, -1, 6:15].view(3, 3), expected, atol=1e-5)


def test_rotation_stays_valid():
    torch.manual_seed(0)
    steps = 50
    state = level_state((20.0, -15.0, 8.0)).repeat(B, 1)
    out = integrate(state, 4.18 * torch.rand(B, steps, 4), torch.randn(B, steps, 3),
                    5 * torch.randn(B, steps, 3, 4), DT)
    R = out[..., 6:15].unflatten(-1, (3, 3))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand_as(R), atol=1e-4)
    assert torch.allclose(torch.linalg.det(R), torch.ones(B, steps), atol=1e-4)


def test_angular_velocity_uses_K_times_forces():
    forces = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    K = torch.zeros(1, 1, 3, 4)
    K[0, 0, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    out = integrate(level_state(), forces, torch.zeros(1, 1, 3), K, DT)
    assert torch.allclose(out[0, 0, 15:], torch.tensor([-2.0 * DT, 0.0, 0.0]))


def test_gradients_reach_every_prober_parameter():
    prober = PhysicsProber(24, DT)
    torch.nn.init.normal_(prober.mlp[-1].weight, std=0.1)  # zero init would block the hidden layers
    out = prober(torch.randn(B, T, 24), level_state((1.0, 2.0, 3.0)).repeat(B, 1), torch.rand(B, T, 4))
    out.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in prober.parameters())


def test_multi_step_rollout_matches_chained_single_steps():
    torch.manual_seed(0)
    prober = PhysicsProber(24, DT)
    torch.nn.init.normal_(prober.mlp[-1].weight, std=0.1)
    latents, forces = torch.randn(B, T, 24), 4.18 * torch.rand(B, T, 4)
    state = level_state((1.0, -2.0, 0.5)).repeat(B, 1)
    full = prober(latents, state, forces)
    for k in range(T):
        state = prober(latents[:, k:k + 1], state, forces[:, k:k + 1])[:, 0]
        assert torch.allclose(full[:, k], state, atol=1e-5)


def synthetic_windows(config: Config):
    """Windows cut from a trajectory that nominal physics produces exactly from varying thrusts."""

    torch.manual_seed(1)
    n_steps = 40
    actions = 4.18 * torch.rand(1, n_steps, 4)
    start = level_state((0.3, -0.2, 0.5))
    states = torch.cat([start.unsqueeze(1), integrate(start, actions, *no_residual(n_steps), DT)], 1)
    trajectory = {"states": states[0].numpy(), "actions": actions[0].numpy()}
    arrays = Data.build_dataset([trajectory], config.history_len, config.horizon)
    return {key: torch.from_numpy(value) for key, value in arrays.items()}, trajectory


def test_windows_feed_x_t_and_a_t_onward():
    config = Config()
    batch, trajectory = synthetic_windows(config)
    jepa, norm = make_jepa(config)
    # The zero-initialized prober is pure nominal physics, so it must reproduce the
    # trajectory exactly. Any off-by-one in the state or force alignment breaks this.
    predicted = predict(batch, jepa, PhysicsProber(config.latent_dim, DT), norm, config.history_len)
    assert torch.allclose(predicted, batch[Data.FUTURE_STATES], atol=1e-4)
    first = config.history_len - 1
    assert np.allclose(batch[Data.ACTIONS][0, first:].numpy(), trajectory["actions"][first:first + config.horizon])


def test_training_step_leaves_jepa_untouched():
    config = Config()
    batch, _ = synthetic_windows(config)
    jepa, norm = make_jepa(config)
    before = [p.clone() for module in jepa for p in module.parameters()]
    prober = PhysicsProber(config.latent_dim, DT)
    optimizer = torch.optim.Adam(prober.parameters(), lr=1e-2)
    # +1 offset: the zero-init prober is already exact on this data, so the loss needs a gradient
    loss = (predict(batch, jepa, prober, norm, config.history_len) - batch[Data.FUTURE_STATES] + 1).square().mean()
    loss.backward()
    optimizer.step()
    after = [p for module in jepa for p in module.parameters()]
    assert all(p.grad is None for p in after)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert prober.mlp[-1].bias.abs().sum() > 0  # the prober itself did train
