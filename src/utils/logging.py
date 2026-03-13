"""
Logging and metrics tracking utilities.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np


# ------------------------------------------------------------------ #
#  Per-episode metrics record                                        #
# ------------------------------------------------------------------ #

@dataclass
class EpisodeMetrics:
    """Stores statistics for one training episode."""
    episode: int = 0
    total_reward: float = 0.0
    total_sold: int = 0
    total_waste: int = 0
    total_stockout: int = 0
    total_ordered: int = 0
    steps: int = 0
    waste_rate: float = 0.0
    fill_rate: float = 1.0
    stockout_rate: float = 0.0
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode),
            "total_reward": float(self.total_reward),
            "total_sold": int(self.total_sold),
            "total_waste": int(self.total_waste),
            "total_stockout": int(self.total_stockout),
            "total_ordered": int(self.total_ordered),
            "steps": int(self.steps),
            "waste_rate": float(self.waste_rate),
            "fill_rate": float(self.fill_rate),
            "stockout_rate": float(self.stockout_rate),
            "elapsed": float(self.elapsed),
        }


# ------------------------------------------------------------------ #
#  Metrics tracker (across all episodes)                             #
# ------------------------------------------------------------------ #

class MetricsTracker:
    """Accumulates episode metrics during training."""

    def __init__(self):
        self.metrics: List[EpisodeMetrics] = []
        self._current: Optional[EpisodeMetrics] = None
        self._episode_start: float = 0.0

    # -- episode lifecycle ------------------------------------------------

    def start_episode(self, episode: int) -> None:
        self._current = EpisodeMetrics(episode=episode)
        self._episode_start = time.time()

    def record_step(
        self,
        reward: float = 0.0,
        sold: int = 0,
        waste: int = 0,
        stockout: int = 0,
        ordered: int = 0,
    ) -> None:
        if self._current is None:
            return
        self._current.total_reward += reward
        self._current.total_sold += sold
        self._current.total_waste += waste
        self._current.total_stockout += stockout
        self._current.total_ordered += ordered
        self._current.steps += 1

    def end_episode(self) -> EpisodeMetrics:
        m = self._current
        m.elapsed = time.time() - self._episode_start

        total_demand = m.total_sold + m.total_stockout
        if total_demand > 0:
            m.fill_rate = m.total_sold / total_demand
            m.stockout_rate = m.total_stockout / total_demand
        else:
            m.fill_rate = 1.0
            m.stockout_rate = 0.0

        # waste_rate = expired / ordered (proposal definition)
        m.waste_rate = m.total_waste / m.total_ordered if m.total_ordered > 0 else 0.0

        self.metrics.append(m)
        self._current = None
        return m

    # -- queries ----------------------------------------------------------

    def get_all_rewards(self) -> List[float]:
        return [m.total_reward for m in self.metrics]

    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        recent = self.metrics[-n:]
        if not recent:
            return {}
        rewards = [m.total_reward for m in recent]
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_waste_rate": float(np.mean([m.waste_rate for m in recent])),
            "mean_stockout_rate": float(np.mean([m.stockout_rate for m in recent])),
            "mean_fill_rate": float(np.mean([m.fill_rate for m in recent])),
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        tracker = cls()
        with open(path, "r") as f:
            data = json.load(f)
        for d in data:
            m = EpisodeMetrics(**d)
            tracker.metrics.append(m)
        return tracker


# ------------------------------------------------------------------ #
#  Simple console + file logger                                      #
# ------------------------------------------------------------------ #

class Logger:
    """Minimal logger with elapsed-time prefix."""

    def __init__(
        self,
        name: str = "RL",
        verbose: bool = True,
        log_file: Optional[str] = None,
    ):
        self.name = name
        self.verbose = verbose
        self._start = time.time()
        self._log_file = log_file

    def _elapsed(self) -> str:
        return f"[{time.time() - self._start:.1f}s]"

    def _write(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(msg + "\n")

    def info(self, msg: str) -> None:
        self._write(f"{self._elapsed()} [INFO] {msg}")

    def warning(self, msg: str) -> None:
        self._write(f"{self._elapsed()} [WARN] {msg}")

    def summary(self, stats: Dict[str, float]) -> None:
        self.info("=== Training Summary ===")
        for k, v in stats.items():
            self.info(f"  {k}: {v:.4f}")
