"""
Dynamic Programming – Value Iteration for the perishable inventory MDP.

Requires a model: the environment's simulate_step() and
get_demand_distribution() methods.
"""

import pickle
import os
import time
from typing import Dict, Tuple, Optional, Any

import numpy as np

from ..envs.perishable_inventory import PerishableInventoryEnv
from ..config import DPConfig


class DPValueIterationAgent:
    """Exact value iteration over the full state space."""

    name = "DP-ValueIteration"

    def __init__(
        self,
        env: PerishableInventoryEnv,
        config: Optional[DPConfig] = None,
        verbose: bool = True,
    ):
        self.env = env
        self.cfg = config or DPConfig()
        self.verbose = verbose

        self.D = env.D
        self.num_actions = env.num_actions
        self.num_states = env.num_states

        # Pre-compute
        self.all_states = env.get_all_states()
        demands, probs = env.get_demand_distribution()
        self.demand_dist = list(zip(demands.tolist(), probs.tolist()))

        # Value function & policy
        self.V = np.zeros(self.num_states)
        self.policy: Dict[Tuple[int, ...], int] = {}

    # ------------------------------------------------------------------ #
    #  Solve                                                              #
    # ------------------------------------------------------------------ #

    def solve(self) -> Dict[str, Any]:
        """Run value iteration until convergence."""
        t0 = time.time()
        theta = self.cfg.theta
        gamma = 0.99
        converged = False

        for iteration in range(1, self.cfg.max_iter + 1):
            delta = 0.0

            for idx, state in enumerate(self.all_states):
                old_v = self.V[idx]

                best_value = -np.inf
                for a in range(self.num_actions):
                    q_sa = 0.0
                    for demand, prob in self.demand_dist:
                        ns, reward, _, _, _ = self.env.simulate_step(state, a, int(demand))
                        ns_idx = self.env.get_state_index(ns)
                        q_sa += prob * (reward + gamma * self.V[ns_idx])
                    if q_sa > best_value:
                        best_value = q_sa

                self.V[idx] = best_value
                delta = max(delta, abs(old_v - best_value))

            if self.verbose and iteration % 10 == 0:
                print(f"  VI iter {iteration:4d}  |  delta = {delta:.8f}")

            if delta < theta:
                converged = True
                break

        # Extract greedy policy
        for idx, state in enumerate(self.all_states):
            best_a, best_val = 0, -np.inf
            for a in range(self.num_actions):
                q_sa = 0.0
                for demand, prob in self.demand_dist:
                    ns, reward, _, _, _ = self.env.simulate_step(state, a, int(demand))
                    ns_idx = self.env.get_state_index(ns)
                    q_sa += prob * (reward + gamma * self.V[ns_idx])
                if q_sa > best_val:
                    best_val = q_sa
                    best_a = a
            self.policy[state] = best_a

        elapsed = time.time() - t0
        stats = {
            "converged": converged,
            "iterations": iteration,
            "final_delta": float(delta),
            "elapsed_sec": elapsed,
        }
        if self.verbose:
            print(f"\nValue iteration {'converged' if converged else 'stopped'} "
                  f"in {iteration} iters ({elapsed:.1f}s), delta={delta:.2e}")
        return stats

    # ------------------------------------------------------------------ #
    #  Action selection                                                   #
    # ------------------------------------------------------------------ #

    def select_action(self, state: Tuple[int, ...]) -> int:
        return self.policy.get(state, 0)

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate_policy(
        self,
        env: PerishableInventoryEnv,
        num_episodes: int = 100,
        seed: int = 123,
    ) -> Dict[str, float]:
        rng = np.random.RandomState(seed)
        rewards, wastes, stockouts, fills = [], [], [], []

        for _ in range(num_episodes):
            state, _ = env.reset(seed=int(rng.randint(0, 2**31)))
            ep_reward = 0.0
            ep_sold = ep_waste = ep_stockout = 0

            while True:
                action = self.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_sold += info["sold"]
                ep_waste += info["waste"]
                ep_stockout += info["stockout"]
                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            total_demand = ep_sold + ep_stockout
            wastes.append(ep_waste)
            stockouts.append(ep_stockout)
            fills.append(ep_sold / total_demand if total_demand > 0 else 1.0)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_waste": float(np.mean(wastes)),
            "mean_stockout": float(np.mean(stockouts)),
            "mean_fill_rate": float(np.mean(fills)),
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        actions = list(self.policy.values())
        return {
            "num_states": len(self.policy),
            "mean_action": float(np.mean(actions)) if actions else 0.0,
            "max_action": int(max(actions)) if actions else 0,
        }

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"V": self.V, "policy": self.policy}, f)

    def load_from(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.V = data["V"]
        self.policy = data["policy"]
