"""
Hyper-parameter sweep utility.
"""

import os
import json
import itertools
from typing import Dict, Any, List, Optional

import numpy as np

from ..config import Config, EnvConfig, TrainingConfig
from .train import train_agent


def run_single_experiment(
    agent_type: str,
    config: Config,
    num_seeds: int = 3,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an agent with multiple seeds and return averaged results.
    """
    all_rewards = []
    for i in range(num_seeds):
        cfg = Config(
            env=config.env,
            training=config.training,
            seed=base_seed + i,
        )
        result = train_agent(agent_type, cfg, verbose=False)
        all_rewards.append(result["final_stats"]["mean_reward"])

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "individual_rewards": all_rewards,
    }


def run_sweep(
    agent_types: List[str] = None,
    alphas: List[float] = None,
    epsilon_decays: List[float] = None,
    demand_means: List[float] = None,
    episodes: int = 1000,
    num_seeds: int = 3,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Grid search over agent types × alphas × epsilon_decays × demand_means.
    """
    if agent_types is None:
        agent_types = ["qlearning", "sarsa", "mc"]
    if alphas is None:
        alphas = [0.05, 0.1, 0.2]
    if epsilon_decays is None:
        epsilon_decays = [0.99, 0.995, 0.999]
    if demand_means is None:
        demand_means = [4.0, 6.0, 8.0]

    results = []
    combos = list(itertools.product(agent_types, alphas, epsilon_decays, demand_means))
    total = len(combos)

    for idx, (agent, alpha, eps_decay, lam) in enumerate(combos, 1):
        print(f"[{idx}/{total}] agent={agent}  α={alpha}  ε_decay={eps_decay}  λ={lam}")

        cfg = Config(
            env=EnvConfig(demand_mean=lam),
            training=TrainingConfig(episodes=episodes, alpha=alpha, epsilon_decay=eps_decay),
        )
        exp_result = run_single_experiment(agent, cfg, num_seeds)

        entry = {
            "agent_type": agent,
            "alpha": alpha,
            "epsilon_decay": eps_decay,
            "demand_mean": lam,
            **exp_result,
        }
        results.append(entry)
        print(f"  → mean_reward = {exp_result['mean_reward']:.1f} ± {exp_result['std_reward']:.1f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results saved to {save_path}")

    return results
