"""
Lightweight space definitions – gymnasium-like, with no dependency.
"""

from typing import Tuple, Optional
import numpy as np


class DiscreteSpace:
    """A discrete space {0, 1, …, n-1}."""

    def __init__(self, n: int, seed: Optional[int] = None):
        self.n = n
        self._rng = np.random.RandomState(seed)

    def sample(self, rng=None) -> int:
        r = rng if rng is not None else self._rng
        return int(r.randint(self.n))

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

    def __repr__(self) -> str:
        return f"DiscreteSpace(n={self.n})"


class BoundedIntegerSpace:
    """Integers in [low, high]."""

    def __init__(self, low: int, high: int, seed: Optional[int] = None):
        self.low = low
        self.high = high
        self._rng = np.random.RandomState(seed)

    @property
    def n(self) -> int:
        return self.high - self.low + 1

    def sample(self) -> int:
        return int(self._rng.randint(self.low, self.high + 1))

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and self.low <= x <= self.high

    def __repr__(self) -> str:
        return f"BoundedIntegerSpace(low={self.low}, high={self.high})"


class TupleSpace:
    """Cartesian product of discrete spaces."""

    def __init__(self, spaces: Tuple):
        self.spaces = spaces

    def sample(self) -> tuple:
        return tuple(s.sample() for s in self.spaces)

    def contains(self, x: tuple) -> bool:
        return (
            isinstance(x, tuple)
            and len(x) == len(self.spaces)
            and all(s.contains(xi) for s, xi in zip(self.spaces, x))
        )

    def __repr__(self) -> str:
        return f"TupleSpace({self.spaces})"
