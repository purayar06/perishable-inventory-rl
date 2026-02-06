"""
Base classes for all agents.
"""

import pickle
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

import numpy as np


class BaseAgent(ABC):
    """Abstract base for every agent."""

    name: str = "BaseAgent"

    @abstractmethod
    def select_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, seed: Optional[int] = None) -> "BaseAgent":
        ...


class TabularAgent(BaseAgent):
    """
    Shared infrastructure for tabular RL agents:
      - Q-table as defaultdict(float)
      - epsilon-greedy action selection with multiplicative decay
      - pickle-based save / load
    """

    name: str = "TabularAgent"

    def __init__(
        self,
        num_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(seed)

        # Q(s, a) stored as {(state_tuple, action_int): float}
        self.q_table: Dict[Tuple, float] = defaultdict(float)

    # -- action selection -------------------------------------------------

    def select_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        """Epsilon-greedy when training=True, purely greedy otherwise."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.randint(self.num_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: Tuple[int, ...]) -> int:
        """Argmax_a Q(s, a) with random tie-breaking."""
        q_values = np.array([self.q_table[(state, a)] for a in range(self.num_actions)])
        max_q = q_values.max()
        best = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best))

    # -- epsilon decay ----------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": dict(self.q_table),
            "name": self.name,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, seed: Optional[int] = None) -> "TabularAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            num_actions=data["num_actions"],
            gamma=data["gamma"],
            alpha=data["alpha"],
            epsilon_start=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            seed=seed,
        )
        agent.epsilon = data["epsilon"]
        agent.q_table = defaultdict(float, data["q_table"])
        return agent
