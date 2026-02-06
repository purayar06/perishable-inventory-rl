"""
Evaluation utilities for trained agents.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List

import numpy as np

from ..config import Config, EnvConfig
from ..envs.perishable_inventory import PerishableInventoryEnv
from ..agents.base import TabularAgent


def evaluate_agent(
    agent,
    env: PerishableInventoryEnv,
    num_episodes: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Evaluate a trained agent over multiple episodes.

    Returns:
        Dictionary of evaluation statistics.
    """
    rng = np.random.RandomState(seed)
    rewards, wastes, stockouts, fills = [], [], [], []

    for _ in range(num_episodes):
        state = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        ep_sold = ep_waste = ep_stockout = 0

        while True:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_sold += info["sold"]
            ep_waste += info["waste"]
            ep_stockout += info["stockout"]
            if done:
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


def evaluate_random_policy(
    env: PerishableInventoryEnv,
    num_episodes: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate a uniformly random policy."""
    rng = np.random.RandomState(seed)
    rewards = []

    for _ in range(num_episodes):
        state = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        while True:
            action = int(rng.randint(env.num_actions))
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def evaluate_constant_policy(
    env: PerishableInventoryEnv,
    order_qty: int,
    num_episodes: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate a deterministic constant-order policy."""
    rng = np.random.RandomState(seed)
    rewards = []

    for _ in range(num_episodes):
        state = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        while True:
            state, reward, done, _ = env.step(order_qty)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def compare_agents(
    agent_paths: List[str],
    env_config: Optional[EnvConfig] = None,
    num_episodes: int = 100,
    seed: int = 123,
) -> Dict[str, Dict[str, Any]]:
    """Load multiple agents and compare evaluation metrics."""
    if env_config is None:
        env_config = EnvConfig()

    env = PerishableInventoryEnv(config=env_config, seed=seed)
    results = {}

    for path in agent_paths:
        agent = TabularAgent.load(path)
        name = os.path.basename(os.path.dirname(path))
        results[name] = evaluate_agent(agent, env, num_episodes, seed)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--agent-path", type=str, required=True,
                        help="Path to agent.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shelf-life", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--max-inventory", type=int, default=20)

    args = parser.parse_args()

    env_cfg = EnvConfig(
        shelf_life=args.shelf_life,
        max_order=args.max_order,
        max_inventory=args.max_inventory,
    )
    env = PerishableInventoryEnv(config=env_cfg, seed=args.seed)
    agent = TabularAgent.load(args.agent_path)

    stats = evaluate_agent(agent, env, args.episodes, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
