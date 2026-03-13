"""
Linear Function Approximation agent (Semi-gradient SARSA / Q-Learning).

Instead of a Q-table, approximates Q(s, a) ≈ w_a · φ(s) using a weight
vector per action and a fixed feature extractor φ.

Update (semi-gradient):
    w_a ← w_a + α · δ · φ(s)
where δ = r + γ · Q(s', a') − Q(s, a)   [SARSA variant by default]
"""

import os
import pickle
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .base import BaseAgent
from ..envs.perishable_inventory import PerishableInventoryEnv
from ..features.linear_features import LinearFeatureExtractor


class LinearFAAgent(BaseAgent):
    """Semi-gradient TD agent with linear function approximation."""

    name = "Linear-FA"

    def __init__(
        self,
        num_actions: int,
        shelf_life: int,
        max_inventory: int,
        gamma: float = 0.99,
        alpha: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        td_method: str = "sarsa",
        seed: Optional[int] = None,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.td_method = td_method  # "sarsa" or "qlearning"
        self.rng = np.random.RandomState(seed)

        self.feature_extractor = LinearFeatureExtractor(shelf_life, max_inventory)
        self.num_features = self.feature_extractor.num_features

        # One weight vector per action: shape (num_actions, num_features)
        self.weights = np.zeros((num_actions, self.num_features))

    # -- Q-value computation ----------------------------------------------

    def q_value(self, state: Tuple[int, ...], action: int) -> float:
        phi = self.feature_extractor(state)
        return float(self.weights[action] @ phi)

    def q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        phi = self.feature_extractor(state)
        return self.weights @ phi

    # -- action selection --------------------------------------------------

    def select_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.randint(self.num_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: Tuple[int, ...]) -> int:
        qv = self.q_values(state)
        max_q = qv.max()
        best = np.where(qv == max_q)[0]
        return int(self.rng.choice(best))

    # -- epsilon decay -----------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- training ----------------------------------------------------------

    def train_episode(self, env: PerishableInventoryEnv) -> Dict[str, Any]:
        state, _ = env.reset()
        action = self.select_action(state, training=True)

        total_reward = 0.0
        total_sold = 0
        total_waste = 0
        total_stockout = 0
        total_ordered = 0

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            phi = self.feature_extractor(state)

            if self.td_method == "qlearning":
                # Off-policy: max over next actions
                next_q = 0.0 if done else float(self.q_values(next_state).max())
                next_action = self.select_action(next_state, training=True)
            else:
                # On-policy SARSA
                next_action = self.select_action(next_state, training=True)
                next_q = 0.0 if done else self.q_value(next_state, next_action)

            td_target = reward + self.gamma * next_q
            td_error = td_target - self.q_value(state, action)

            # Semi-gradient update
            self.weights[action] += self.alpha * td_error * phi

            total_reward += reward
            total_sold += info["sold"]
            total_waste += info["waste"]
            total_stockout += info["stockout"]
            total_ordered += info["order"]

            state = next_state
            action = next_action

            if done:
                break

        self.decay_epsilon()

        return {
            "total_reward": total_reward,
            "total_sold": total_sold,
            "total_waste": total_waste,
            "total_stockout": total_stockout,
            "total_ordered": total_ordered,
        }

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "num_actions": self.num_actions,
            "num_features": self.num_features,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "td_method": self.td_method,
            "weights": self.weights,
            "name": self.name,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, seed: Optional[int] = None) -> "LinearFAAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        # We need shelf_life & max_inventory to rebuild the feature extractor.
        # Infer from weights shape: num_features = D + 3 → D = num_features - 3
        nf = data["num_features"]
        shelf_life = nf - 3
        # max_inventory is not stored; default to 20 (will be overridden if used properly)
        agent = cls(
            num_actions=data["num_actions"],
            shelf_life=shelf_life,
            max_inventory=20,
            gamma=data["gamma"],
            alpha=data["alpha"],
            epsilon_start=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            td_method=data["td_method"],
            seed=seed,
        )
        agent.weights = data["weights"]
        agent.epsilon = data["epsilon"]
        return agent
