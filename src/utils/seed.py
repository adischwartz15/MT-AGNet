"""Deterministic seeding helpers for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    Parameters
    ----------
    seed:
        Global seed value.
    deterministic_torch:
        If True and torch is available, enable deterministic cuDNN
        algorithms. This can slow down training but is important for
        reproducible ablation comparisons.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible multi-worker loading."""
    worker_seed = (torch_initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed() -> int:
    import torch

    return torch.initial_seed() % (2**32)
