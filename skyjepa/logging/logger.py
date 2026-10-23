"""Training progress logger. Lines are tagged [INFO], [WARNING], or [ERROR]."""

import logging

import torch

_LOG = logging.getLogger("skyjepa.train")


def _ensure_logging() -> None:
    if _LOG.handlers:
        return
    _LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _LOG.addHandler(handler)
    _LOG.propagate = False


def _sync(device: str) -> None:
    """Finish pending GPU work so the next timer measures that work."""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:7.1f}ms"
