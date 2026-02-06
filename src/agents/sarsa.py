"""
SARSA agent (on-policy TD control).

Update rule:
    Q(s, a) ← Q(s, a) + α [ r + γ · Q(s', a') − Q(s, a) ]
where a' is chosen by the current ε-greedy policy.
"""

from typing import Dict, Any
from .base import TabularAgent
from ..envs.perishable_inventory import PerishableInventoryEnv


class SARSAAgent(TabularAgent):
    """On-policy tabular SARSA."""

    name = "SARSA"

    def train_episode(self, env: PerishableInventoryEnv) -> Dict[str, Any]:
        state, _ = env.reset()
        action = self.select_action(state, training=True)

        total_reward = 0.0
        total_sold = 0
        total_waste = 0
        total_stockout = 0

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action = self.select_action(next_state, training=True)

            # On-policy update
            td_target = reward + self.gamma * self.q_table[(next_state, next_action)] * (1 - done)
            td_error = td_target - self.q_table[(state, action)]
            self.q_table[(state, action)] += self.alpha * td_error

            total_reward += reward
            total_sold += info["sold"]
            total_waste += info["waste"]
            total_stockout += info["stockout"]

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
        }
