"""
Sensitivity analysis: vary shelf-life D, demand rate λ, and penalty weights.

Trains each agent under every parameter combination and collects
evaluation metrics (mean reward, waste rate, stockout rate).
Results are saved to a JSON file for downstream plotting.
"""

import os
import json
import itertools
from typing import List, Optional, Dict, Any

import numpy as np

from ..config import Config, EnvConfig, TrainingConfig, DPConfig
from .train import train_agent, train_dp


def run_sensitivity_analysis(
    agent_types: Optional[List[str]] = None,
    shelf_lives: Optional[List[int]] = None,
    demand_means: Optional[List[float]] = None,
    waste_penalties: Optional[List[float]] = None,
    stockout_penalties: Optional[List[float]] = None,
    episodes: int = 1000,
    num_seeds: int = 3,
    base_seed: int = 42,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Vary shelf-life D, demand rate λ, and penalty weights (w, s) across agents.

    Args:
        agent_types: RL agents to evaluate (default: qlearning, sarsa, mc, linear_fa).
        shelf_lives: List of D values (default: [2, 3, 5]).
        demand_means: List of λ values (default: [3.0, 6.0, 9.0]).
        waste_penalties: List of w values (default: [2.0, 6.0, 10.0]).
        stockout_penalties: List of s values (default: [4.0, 8.0, 12.0]).
        episodes: Training episodes per run.
        num_seeds: Number of independent seeds.
        base_seed: Starting seed.
        save_path: JSON file to save results.

    Returns:
        List of result dictionaries.
    """
    if agent_types is None:
        agent_types = ["qlearning", "sarsa", "mc", "linear_fa", "dp"]
    if shelf_lives is None:
        shelf_lives = [2, 3, 5]
    if demand_means is None:
        demand_means = [3.0, 6.0, 9.0]
    if waste_penalties is None:
        waste_penalties = [2.0, 6.0, 10.0]
    if stockout_penalties is None:
        stockout_penalties = [4.0, 8.0, 12.0]

    results: List[Dict[str, Any]] = []
    combos = list(itertools.product(
        agent_types, shelf_lives, demand_means, waste_penalties, stockout_penalties
    ))
    total = len(combos)

    for idx, (agent, D, lam, w, s) in enumerate(combos, 1):
        print(f"[{idx}/{total}] agent={agent}  D={D}  λ={lam}  w={w}  s={s}")

        seed_rewards = []
        seed_waste = []
        seed_stockout = []

        for seed_i in range(num_seeds):
            cfg = Config(
                env=EnvConfig(
                    shelf_life=D,
                    demand_mean=lam,
                    waste_penalty=w,
                    stockout_penalty=s,
                ),
                training=TrainingConfig(episodes=episodes),
                dp=DPConfig(),
                seed=base_seed + seed_i,
            )
            
            if agent == "dp":
                if D > 2:
                    # DP is too slow/memory intensive for D > 2 in a sweep
                    stats = {"mean_reward": float('-inf'), "mean_waste_rate": 0.0, "mean_stockout_rate": 0.0}
                else:
                    # For DP, we only need to solve once (it's exact), but we can evaluate it
                    # over the seeds to maintain the same format, or just solve and evaluate once.
                    if seed_i == 0:
                        dp_res = train_dp(cfg, verbose=False)
                        dp_stats = dp_res["eval_stats"]
                    stats = dp_stats # Reuse exact same eval stats for the 3 'seeds' to match format
            else:
                result = train_agent(agent, cfg, verbose=False)
                stats = result["final_stats"]
                
            seed_rewards.append(stats["mean_reward"])
            seed_waste.append(stats.get("mean_waste_rate", 0.0))
            seed_stockout.append(stats.get("mean_stockout_rate", 0.0))

        entry = {
            "agent_type": agent,
            "shelf_life": D,
            "demand_mean": lam,
            "waste_penalty": w,
            "stockout_penalty": s,
            "mean_reward": float(np.mean(seed_rewards)),
            "std_reward": float(np.std(seed_rewards)),
            "mean_waste_rate": float(np.mean(seed_waste)),
            "mean_stockout_rate": float(np.mean(seed_stockout)),
        }
        results.append(entry)
        print(f"  → reward = {entry['mean_reward']:.1f} ± {entry['std_reward']:.1f}  "
              f"waste = {entry['mean_waste_rate']:.3f}  "
              f"stockout = {entry['mean_stockout_rate']:.3f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSensitivity results saved to {save_path}")

    return results
