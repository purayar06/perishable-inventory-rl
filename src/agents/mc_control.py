"""
First-visit Monte Carlo Control with ε-greedy policy.

Collects full episode trajectory, then performs first-visit updates:
    Q(s, a) ← Q(s, a) + α [ G − Q(s, a) ]
where G is the discounted return from the first visit to (s, a).
"""

from typing import Dict, Any, List, Tuple
from .base import TabularAgent
from ..envs.perishable_inventory import PerishableInventoryEnv


class MonteCarloAgent(TabularAgent):
    """First-visit MC control with incremental mean update."""

    name = "Monte Carlo"

    def train_episode(self, env: PerishableInventoryEnv) -> Dict[str, Any]:
        # --- roll out full episode ---
        trajectory: List[Tuple[tuple, int, float]] = []
        state, _ = env.reset()

        total_reward = 0.0
        total_sold = 0
        total_waste = 0
        total_stockout = 0
        total_ordered = 0

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trajectory.append((state, action, reward))

            total_reward += reward
            total_sold += info["sold"]
            total_waste += info["waste"]
            total_stockout += info["stockout"]
            total_ordered += info["order"]

            state = next_state
            if done:
                break

        # --- first-visit MC update ---
        G = 0.0
        visited = set()
        for s, a, r in reversed(trajectory):
            G = r + self.gamma * G
            sa = (s, a)
            if sa not in visited:
                visited.add(sa)
                self.q_table[sa] += self.alpha * (G - self.q_table[sa])

        self.decay_epsilon()

        return {
            "total_reward": total_reward,
            "total_sold": total_sold,
            "total_waste": total_waste,
            "total_stockout": total_stockout,
            "total_ordered": total_ordered,
        }
