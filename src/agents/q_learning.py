"""
Q-Learning agent (off-policy TD control).

Update rule:
    Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
"""

from typing import Dict, Any
from .base import TabularAgent
from ..envs.perishable_inventory import PerishableInventoryEnv


class QLearningAgent(TabularAgent):
    """Off-policy tabular Q-Learning."""

    name = "Q-Learning"

    def train_episode(self, env: PerishableInventoryEnv) -> Dict[str, Any]:
        """Run one full episode and update Q-values after every step."""
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

            # Off-policy update: use max over next actions
            best_next = max(
                self.q_table[(next_state, a)] for a in range(self.num_actions)
            )
            td_target = reward + self.gamma * best_next * (1 - done)
            td_error = td_target - self.q_table[(state, action)]
            self.q_table[(state, action)] += self.alpha * td_error

            total_reward += reward
            total_sold += info["sold"]
            total_waste += info["waste"]
            total_stockout += info["stockout"]
            total_ordered += info["order"]

            state = next_state
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
