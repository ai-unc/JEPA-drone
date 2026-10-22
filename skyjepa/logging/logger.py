"""Step timers and the training progress logger"""

import logging
import time
from contextlib import contextmanager

import torch

_LOG = logging.getLogger("skyjepa.train")


def _ensure_logging() -> None:
    if _LOG.handlers:
        return
    _LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
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


@contextmanager
def _timed(step: str):
    _ensure_logging()
    _LOG.info("start  %s", step)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        _LOG.exception("fail   %s  after %.3fs", step, time.perf_counter() - started)
        raise
    else:
        _LOG.info("done   %s  %.3fs", step, time.perf_counter() - started)
