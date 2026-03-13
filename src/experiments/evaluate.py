"""
Evaluation utilities for trained agents.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List

import numpy as np

from ..config import EnvConfig, DPConfig
from ..envs.perishable_inventory import PerishableInventoryEnv
from ..agents.base import TabularAgent
from ..agents.q_learning import QLearningAgent
from ..agents.sarsa import SARSAAgent
from ..agents.mc_control import MonteCarloAgent
from ..agents.linear_fa import LinearFAAgent
from ..agents.dp_value_iteration import DPValueIterationAgent


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
    total_sold = total_waste = total_stockout = total_ordered = total_steps = 0

    for _ in range(num_episodes):
        state, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        ep_sold = ep_waste = ep_stockout = ep_ordered = 0

        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_sold += info["sold"]
            ep_waste += info["waste"]
            ep_stockout += info["stockout"]
            ep_ordered += info["order"]
            total_steps += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        total_demand = ep_sold + ep_stockout
        wastes.append(ep_waste / ep_ordered if ep_ordered > 0 else 0.0)
        stockouts.append(ep_stockout / total_demand if total_demand > 0 else 0.0)
        fills.append(ep_sold / total_demand if total_demand > 0 else 1.0)
        total_sold += ep_sold
        total_waste += ep_waste
        total_stockout += ep_stockout
        total_ordered += ep_ordered

    return {
        "num_episodes": num_episodes,
        "seed": seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_waste_rate": float(np.mean(wastes)),
        "mean_stockout_rate": float(np.mean(stockouts)),
        "mean_fill_rate": float(np.mean(fills)),
        "total_sold": int(total_sold),
        "total_waste": int(total_waste),
        "total_stockout": int(total_stockout),
        "total_ordered": int(total_ordered),
        "total_steps": int(total_steps),
    }


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_agent_for_run(run_dir: str, cfg: Dict[str, Any], env: PerishableInventoryEnv, seed: int):
    agent_type = cfg.get("agent_type")

    if agent_type == "qlearning":
        return QLearningAgent.load(os.path.join(run_dir, "agent.pkl"), seed=seed)
    if agent_type == "sarsa":
        return SARSAAgent.load(os.path.join(run_dir, "agent.pkl"), seed=seed)
    if agent_type == "mc":
        return MonteCarloAgent.load(os.path.join(run_dir, "agent.pkl"), seed=seed)
    if agent_type == "linear_fa":
        return LinearFAAgent.load(os.path.join(run_dir, "agent.pkl"), seed=seed)
    if agent_type == "dp":
        dp_cfg = DPConfig(
            theta=cfg.get("dp", {}).get("theta", 1e-6),
            max_iter=cfg.get("dp", {}).get("max_iter", 2000),
        )
        gamma = cfg.get("training", {}).get("gamma", 0.99)
        agent = DPValueIterationAgent(env, config=dp_cfg, gamma=gamma, verbose=False)
        agent.load_from(os.path.join(run_dir, "dp_agent.pkl"))
        return agent

    raise ValueError(f"Unsupported agent_type '{agent_type}' in {run_dir}")


def evaluate_all_runs(
    runs_dir: str,
    num_episodes: int = 100,
    seed: int = 123,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all run directories with a single fixed-seed protocol."""
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    results: Dict[str, Dict[str, Any]] = {}

    for entry in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            continue

        cfg = _load_json(cfg_path)
        env_cfg_dict = cfg.get("env", {})
        env_cfg = EnvConfig(
            shelf_life=env_cfg_dict.get("shelf_life", 5),
            max_order=env_cfg_dict.get("max_order", 10),
            max_inventory=env_cfg_dict.get("max_inventory", 20),
            horizon=env_cfg_dict.get("horizon", 60),
            demand_mean=env_cfg_dict.get("demand_mean", 6.0),
            selling_price=env_cfg_dict.get("selling_price", 10.0),
            ordering_cost=env_cfg_dict.get("ordering_cost", 4.0),
            waste_penalty=env_cfg_dict.get("waste_penalty", 6.0),
            stockout_penalty=env_cfg_dict.get("stockout_penalty", 8.0),
        )
        env = PerishableInventoryEnv(config=env_cfg, seed=seed)

        agent = _load_agent_for_run(run_dir, cfg, env, seed)
        stats = evaluate_agent(agent, env, num_episodes=num_episodes, seed=seed)

        results[entry] = {
            "agent_type": cfg.get("agent_type", entry),
            "evaluation": stats,
        }

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def evaluate_random_policy(
    env: PerishableInventoryEnv,
    num_episodes: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate a uniformly random policy."""
    rng = np.random.RandomState(seed)
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        while True:
            action = int(rng.randint(env.num_actions))
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
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
        state, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        ep_reward = 0.0
        while True:
            state, reward, terminated, truncated, info = env.step(order_qty)
            ep_reward += reward
            if terminated or truncated:
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
    parser.add_argument("--agent-path", type=str, default=None,
                        help="Path to a single saved agent file (agent.pkl or dp_agent.pkl)")
    parser.add_argument("--agent-type", type=str, default="qlearning",
                        choices=["qlearning", "sarsa", "mc", "linear_fa", "dp"],
                        help="Required when using --agent-path")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory containing run subfolders to evaluate uniformly")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Optional path to save JSON output")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shelf-life", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--max-inventory", type=int, default=20)

    args = parser.parse_args()

    if args.runs_dir:
        results = evaluate_all_runs(
            runs_dir=args.runs_dir,
            num_episodes=args.episodes,
            seed=args.seed,
            save_path=args.save_path,
        )
        print(json.dumps(results, indent=2))
        return

    if not args.agent_path:
        raise ValueError("Provide either --runs-dir or --agent-path")

    env_cfg = EnvConfig(
        shelf_life=args.shelf_life,
        max_order=args.max_order,
        max_inventory=args.max_inventory,
    )
    env = PerishableInventoryEnv(config=env_cfg, seed=args.seed)

    if args.agent_type == "qlearning":
        agent = QLearningAgent.load(args.agent_path, seed=args.seed)
    elif args.agent_type == "sarsa":
        agent = SARSAAgent.load(args.agent_path, seed=args.seed)
    elif args.agent_type == "mc":
        agent = MonteCarloAgent.load(args.agent_path, seed=args.seed)
    elif args.agent_type == "linear_fa":
        agent = LinearFAAgent.load(args.agent_path, seed=args.seed)
    else:
        agent = DPValueIterationAgent(env, config=DPConfig(), gamma=0.99, verbose=False)
        agent.load_from(args.agent_path)

    stats = evaluate_agent(agent, env, args.episodes, args.seed)
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
        with open(args.save_path, "w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
