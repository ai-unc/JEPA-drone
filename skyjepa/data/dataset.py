"""Window a list of trajectories.

For each trajectory and each valid t (H - 1 <= t <= n_steps - T):

    state_history  = states[t - H + 1 : t + 1]       # x_{t-H+1} .. x_t
    action_history = actions[t - H + 1 : t + 1]      # a_{t-H+1} .. a_t
    future_states  = states[t + 1 : t + 1 + T]       # x_{t+1} .. x_{t+T}
    future_actions = actions[t : t + T]              # a_t .. a_{t+T-1}

future_actions[0] is a_t, the same row as action_history[-1].
See docs/tensor_shapes.md.
"""

import numpy as np


def build_dataset(
    trajectories: list[dict[str, np.ndarray]],
    history_len: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Stack every valid window.

    Returns float32 arrays:
      state_history   (N, history_len, 4)
      action_history  (N, history_len, 2)
      future_states   (N, horizon, 4)
      future_actions  (N, horizon, 2)
    """

    raise NotImplementedError("build_dataset")
