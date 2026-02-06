"""
Reproducibility helpers: global seeding and RNG wrappers.
"""

import random
import numpy as np
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python stdlib and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Return an independent RandomState."""
    return np.random.RandomState(seed)


class SeededRandom:
    """Thin wrapper around numpy.random.RandomState for convenience."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed_value = seed

    def randint(self, low: int, high: int) -> int:
        return int(self.rng.randint(low, high))

    def random(self) -> float:
        return float(self.rng.random())

    def choice(self, a, size=None, replace=True, p=None):
        return self.rng.choice(a, size=size, replace=replace, p=p)

    def poisson(self, lam: float = 1.0) -> int:
        return int(self.rng.poisson(lam))

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)
        self.seed_value = seed
