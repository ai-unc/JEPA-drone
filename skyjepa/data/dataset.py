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
    if history_len < 1:
        raise ValueError("history_len must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not trajectories:
        raise ValueError("trajectories must not be empty")

    state_history = []
    action_history = []
    future_states = []
    future_actions = []

    for i, trajectory in enumerate(trajectories):
        states = np.asarray(trajectory["states"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)

        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"trajectory {i}: states and actions must be 2-D arrays"
            )

        if states.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                f"trajectory {i}: expected one more state than action, "
                f"got {states.shape[0]} states and {actions.shape[0]} actions"
            )

        n_steps = actions.shape[0]

        first_t = history_len - 1
        last_t = n_steps - horizon

        if first_t > last_t:
            raise ValueError(
                f"trajectory {i}: too short for "
                f"history_len={history_len} and horizon={horizon}"
            )

        for t in range(first_t, last_t + 1):
            state_history.append(
                states[t - history_len + 1 : t + 1]
            )

            action_history.append(
                actions[t - history_len + 1 : t + 1]
            )

            future_states.append(
                states[t + 1 : t + 1 + horizon]
            )

            future_actions.append(
                actions[t : t + horizon]
            )

    return {
        "state_history": np.stack(state_history).astype(
            np.float32, copy=False
        ),
        "action_history": np.stack(action_history).astype(
            np.float32, copy=False
        ),
        "future_states": np.stack(future_states).astype(
            np.float32, copy=False
        ),
        "future_actions": np.stack(future_actions).astype(
            np.float32, copy=False
        ),
    }