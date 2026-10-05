"""Slide history windows out of the four dataset tensors."""

import torch


def sliding_windows(sequence: torch.Tensor, history_len: int) -> torch.Tensor:
    """sequence (B, L, D) -> (B, L - history_len + 1, history_len, D).

    Window i is sequence[:, i : i + history_len]. The last index of that
    window is the window's current time.
    """

    if sequence.ndim != 3:
        raise ValueError(f"expected (batch, time, dim), got {tuple(sequence.shape)}")
    length = sequence.shape[1]
    if history_len < 1 or length < history_len:
        raise ValueError(f"history_len {history_len} does not fit length {length}")
    # unfold appends the window axis: (B, n_windows, D, H)
    unfolded = sequence.unfold(dimension=1, size=history_len, step=1)
    return unfolded.transpose(-1, -2)


def rollout_windows(
    state_history: torch.Tensor,
    action_history: torch.Tensor,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the encoder inputs for one JEPA step.

    state windows: (B, T+1, H, state_dim). Index 0 ends at time t.
    Index k ends at t+k and is the target window for s_{t+k}.

    action windows: (B, T, H, action_dim). Index k ends at t+k and is the
    input for z_{t+k}, which predicts s_{t+k+1}.

    future_actions[:, 0] must equal action_history[:, -1] (both are a_t).
    That duplicate step is dropped before the action windows are sliced.
    """

    history_len = state_history.shape[1]
    horizon = future_states.shape[1]
    if action_history.shape[1] != history_len:
        raise ValueError("action_history and state_history differ in length")
    if future_actions.shape[1] != horizon:
        raise ValueError("future_actions and future_states differ in length")
    if not torch.allclose(action_history[:, -1], future_actions[:, 0]):
        raise ValueError(
            "future_actions[:, 0] must be a_t, matching action_history[:, -1]. "
            "See docs/tensor_shapes.md."
        )

    states = torch.cat([state_history, future_states], dim=1)
    actions = torch.cat([action_history, future_actions[:, 1:]], dim=1)
    state_windows = sliding_windows(states, history_len)
    action_windows = sliding_windows(actions, history_len)
    if state_windows.shape[1] != horizon + 1:
        raise ValueError(
            f"expected {horizon + 1} state windows, got {state_windows.shape[1]}"
        )
    if action_windows.shape[1] != horizon:
        raise ValueError(
            f"expected {horizon} action windows, got {action_windows.shape[1]}"
        )
    return state_windows, action_windows
