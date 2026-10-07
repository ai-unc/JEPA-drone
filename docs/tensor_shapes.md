# Tensor shapes

Shared layout for the toy SkyJEPA stack. Sizes live in `skyjepa.config.Config`. Read them from there instead of hard-coding `10` or `24` in model and data code.

This repo uses history length 10, horizon 10, and MLP encoders. Latent size is 24, matching the paper's GRU width.

## Dimensions

| Symbol | Config field | Value | Meaning |
| --- | --- | --- | --- |
| H | `history_len` | 10 | state and action history length |
| T | `horizon` | 10 | prediction horizon |
| | `state_dim` | 4 | `[x, y, vx, vy]` |
| | `action_dim` | 2 | `[ax, ay]` |
| | `latent_dim` | 24 | encoder output and GRU hidden size |

The batch axis is first. Torch tensors are `float32`. NumPy arrays at the dataset boundary are `float32`.

`Config.H` and `Config.T` are aliases of `history_len` and `horizon`.

## Trajectory

`skyjepa.env.point_mass.generate_trajectories` returns a list of dicts:

| Key | Shape | Rows |
| --- | --- | --- |
| `states` | `(n_steps + 1, 4)` | `x_0 ... x_{n_steps}` |
| `actions` | `(n_steps, 2)` | `a_0 ... a_{n_steps - 1}` |

`a_i` is applied at `x_i` and produces `x_{i+1}`. Default `n_steps` is `Config.n_steps` (100). Default trajectory count is `Config.n_trajectories` (100).

## Dataset windows

`t` is the current time. `skyjepa.data.dataset.build_dataset` stacks one row per valid `t`:

| Key | Times | Shape |
| --- | --- | --- |
| `state_history` | `x_{t-H+1} ... x_t` | `(N, H, 4)` |
| `action_history` | `a_{t-H+1} ... a_t` | `(N, H, 2)` |
| `future_states` | `x_{t+1} ... x_{t+T}` | `(N, T, 4)` |
| `future_actions` | `a_t ... a_{t+T-1}` | `(N, T, 2)` |

`future_actions[:, 0]` is `a_t`, the same control as `action_history[:, -1]`. That action produces `future_states[:, 0]`. `future_actions[:, k]` is `a_{t+k}` and produces `future_states[:, k]`.

Slices on one trajectory:

```python
state_history = states[t - H + 1 : t + 1]
action_history = actions[t - H + 1 : t + 1]
future_states = states[t + 1 : t + 1 + T]
future_actions = actions[t : t + T]
```

Valid `t` for a trajectory with `n_steps` actions: `H - 1 <= t <= n_steps - T`.

Worked example, `H = 2`, `T = 2`, `t = 2`:

| Tensor | Values |
| --- | --- |
| `state_history` | `x_1, x_2` |
| `action_history` | `a_1, a_2` |
| `future_states` | `x_3, x_4` |
| `future_actions` | `a_2, a_3` |

## Latents

`skyjepa.windows.rollout_windows` turns a batch of the four arrays into encoder inputs.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| state windows | `(B, T+1, H, 4)` | index 0 ends at `t`; index `k` ends at `t+k` |
| action windows | `(B, T, H, 2)` | index `k` ends at `t+k` (`z_t ... z_{t+T-1}`) |
| `s_t` | `(B, 24)` | state encoder on state window 0 |
| targets | `(B, T, 24)` | state encoder on state windows `1:`, stop-gradient |
| `z_seq` | `(B, T, 24)` | action encoder on the action windows |
| rollout | `(B, T, 24)` | `s̃_{t+1} ... s̃_{t+T}` |

Both encoders are MLPs. Each takes `(B, H, dim)` and returns `(B, latent_dim)`.

The predictor is one `GRUCell`: input `z_t` `(B, 24)`, hidden state `s_t` `(B, 24)`, output `s_{t+1}` `(B, 24)`. `rollout(s_t, z_seq)` applies that step `T` times.

Action windows drop the duplicated `a_t` at `future_actions[:, 0]` before sliding, so each control appears once.

## Loss

```text
L_pred = (1/T) sum_{k=1}^{T} || s̃_{t+k} - s_{t+k} ||_2^2
```

Average that quantity over the batch. `latent_prediction_loss` takes `predicted` and `target` with shape `(B, T, 24)` and returns a scalar.

`latent_stats(latent)` returns `mean` and `std` over every element. `std` near 0 means the embeddings collapsed to a constant.

Target latents are encoded with `torch.no_grad()`. Gradients still reach both encoders through `s_t` and `z_seq`.

## Baseline

`DirectStatePredictor.forward(state_history, action_history, future_actions)` returns `(B, T, 4)`, the future physical states in one shot.

`plot_prediction_error_vs_horizon(errors, path)` draws one curve. `errors` has shape `(T,)`. `errors[k]` is the mean L2 error of the prediction of `x_{t+1+k}`.

## Modules

| File | Entry points |
| --- | --- |
| `skyjepa/env/point_mass.py` | `PointMass.reset`, `PointMass.step`, `generate_trajectories` |
| `skyjepa/data/dataset.py` | `build_dataset` |
| `skyjepa/models/encoders.py` | `StateHistoryEncoder`, `ActionHistoryEncoder` |
| `skyjepa/models/predictor.py` | `LatentPredictor.forward`, `LatentPredictor.rollout` |
| `skyjepa/losses.py` | `latent_prediction_loss`, `latent_stats` |
| `skyjepa/baseline.py` | `DirectStatePredictor`, `plot_prediction_error_vs_horizon` |
| `skyjepa/config.py` | `Config` |
| `skyjepa/windows.py` | `sliding_windows`, `rollout_windows` |
| `skyjepa/train_jepa.py` | `main` |
