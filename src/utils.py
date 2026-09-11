"""
Small shared utilities used across the project. Kept deliberately minimal —
grows only as later stages actually need something reused in 2+ places.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Seed python's random, numpy, and torch (CPU + CUDA) for reproducibility.
    Torch is imported lazily so this module has no hard torch dependency
    for callers that only need the non-torch parts of src/.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where available; some ops don't have a
        # deterministic implementation, so we don't force-error on that.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
