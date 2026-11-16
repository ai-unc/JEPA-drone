import torch

def sliding_windows(actions: torch.Tensor, history_len: int) -> torch.Tensor:
    """sequence (B, L, D) -> (B, L - history_len + 1, history_len, D).

    Window i is sequence[:, i : i + history_len]. The last index of that
    window is the window's current time.
    """

    if actions.ndim != 3:
        raise ValueError(f"Action tensor dimensions are incorrect, receieved: {tuple(sequence.shape)}")
    length = actions.shape[1]
    if history_len < 1 or length < history_len:
        raise ValueError(f"history_len {history_len} does not fit length {length}")
    # unfold appends the window axis: (B, n_windows, D, H)
    unfolded = actions.unfold(dimension=1, size=history_len, step=1)
    return unfolded.transpose(-1, -2)