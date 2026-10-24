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