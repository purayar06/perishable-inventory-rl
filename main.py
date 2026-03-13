#!/usr/bin/env python
"""
main.py — Single entry-point for training all agents and generating plots.

Usage:
    python main.py                       # Train all 3 agents + auto-generate plots
    python main.py --agents qlearning    # Train only Q-Learning
    python main.py --episodes 2000       # Override episode count
    python main.py --no-plot             # Train only, skip plotting
    python main.py --plot-only           # Skip training, just regenerate plots

All results are saved under outputs/runs/<agent>/ and all figures under
outputs/figures/.
"""

import argparse
import json
import os
import sys
import time

# ------------------------------------------------------------------ #
#  Make sure project root is on sys.path so imports work when called  #
#  as `python main.py` (not just `python -m ...`).                   #
# ------------------------------------------------------------------ #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, EnvConfig, TrainingConfig
from src.experiments.train import train_agent, train_dp, generate_post_training_plots
from src.experiments.evaluate import evaluate_all_runs
from src.plotting.make_plots import generate_all_plots, plot_dp_convergence


# ------------------------------------------------------------------ #
#  Defaults                                                           #
# ------------------------------------------------------------------ #
ALL_AGENTS = ["qlearning", "sarsa", "mc", "linear_fa", "dp"]

RUNS_DIR    = os.path.join("outputs", "runs")
FIGURES_DIR = os.path.join("outputs", "figures")


# ------------------------------------------------------------------ #
#  Training helper                                                    #
# ------------------------------------------------------------------ #

def train_all(
    agents: list,
    episodes: int,
    shelf_life: int,
    max_order: int,
    max_inventory: int,
    demand_mean: float,
    alpha: float,
    gamma: float,
    seed: int,
) -> dict:
    """Train the requested agents and return {name: results}."""
    results = {}

    for agent_type in agents:
        print(f"\n{'='*60}")
        print(f"  Training: {agent_type.upper()}")
        print(f"{'='*60}")

        config = Config(
            env=EnvConfig(
                shelf_life=shelf_life,
                max_order=max_order,
                max_inventory=max_inventory,
                demand_mean=demand_mean,
            ),
            training=TrainingConfig(
                episodes=episodes,
                gamma=gamma,
                alpha=alpha,
            ),
            seed=seed,
        )

        save_path = os.path.join(RUNS_DIR, agent_type)

        if agent_type == "dp":
            result = train_dp(
                config=config,
                verbose=True,
                save_path=save_path,
            )
            results[agent_type] = result
            print(f"  → DP eval mean reward: "
                  f"{result['eval_stats']['mean_reward']:.2f}")
        else:
            result = train_agent(
                agent_type=agent_type,
                config=config,
                verbose=True,
                save_path=save_path,
            )
            results[agent_type] = result
            print(f"  → Mean reward (last 100 eps): "
                  f"{result['final_stats']['mean_reward']:.2f}")

    return results


# ------------------------------------------------------------------ #
#  Plotting helper                                                    #
# ------------------------------------------------------------------ #

def plot_all(results: dict | None = None) -> None:
    """
    Generate every standard figure.

    If *results* is provided (just finished training), also produces
    per-agent metric panels.  Always runs the global comparison suite.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.make_plots import plot_metrics_over_training

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Per-agent metric panels (if we just trained)
    # Skip DP — it has no episodic learning trajectory.
    if results:
        for agent_type, res in results.items():
            if agent_type == "dp":
                # Generate convergence curve instead
                conv_file = os.path.join(
                    res.get("save_path", os.path.join(RUNS_DIR, "dp")),
                    "convergence.json",
                )
                if os.path.isfile(conv_file):
                    with open(conv_file, "r") as f:
                        deltas = json.load(f)
                    plot_dp_convergence(
                        deltas,
                        save_path=os.path.join(FIGURES_DIR, "dp_convergence.png"),
                    )
                    plt.close()
                continue
            metrics_file = os.path.join(
                res.get("save_path", os.path.join(RUNS_DIR, agent_type)),
                "metrics.json",
            )
            if os.path.isfile(metrics_file):
                panel_path = os.path.join(
                    FIGURES_DIR, f"training_metrics_{agent_type}.png"
                )
                plot_metrics_over_training(
                    metrics_file,
                    metrics_to_plot=["total_reward", "waste_rate", "fill_rate"],
                    save_path=panel_path,
                )
                plt.close()

    # Global comparison suite (learning curves, bar chart, scatter)
    print(f"\n{'='*60}")
    print("  Generating comparison plots")
    print(f"{'='*60}")
    generate_all_plots(RUNS_DIR, FIGURES_DIR)


def run_common_evaluation(eval_episodes: int = 100, eval_seed: int = 123) -> None:
    """Evaluate all available runs using one fixed-seed protocol."""
    print(f"\n{'='*60}")
    print("  Running common evaluation protocol")
    print(f"{'='*60}")
    save_path = os.path.join(RUNS_DIR, "evaluation_summary.json")
    evaluate_all_runs(
        runs_dir=RUNS_DIR,
        num_episodes=eval_episodes,
        seed=eval_seed,
        save_path=save_path,
    )
    print(f"  → Saved evaluation summary: {os.path.abspath(save_path)}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL agents and generate plots for the "
                    "Perishable Inventory project.",
    )

    # What to run
    p.add_argument(
        "--agents", nargs="+", default=ALL_AGENTS,
        choices=ALL_AGENTS,
        help="Which agents to train (default: all agents including DP baseline).",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Skip training; only regenerate plots from existing runs.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Train agents but skip plot generation.",
    )
    p.add_argument(
        "--no-eval", action="store_true",
        help="Skip common fixed-seed evaluation across all available runs.",
    )

    # Hyper-parameters
    p.add_argument("--episodes", "-e", type=int, default=3000,
                    help="Training episodes per agent (default 3000).")
    p.add_argument("--shelf-life", "-d", type=int, default=3,
                    help="Shelf life D (default 3).")
    p.add_argument("--max-order", type=int, default=5,
                    help="Maximum order quantity A_max (default 5).")
    p.add_argument("--max-inventory", type=int, default=10,
                    help="Maximum items per bucket N_max (default 10).")
    p.add_argument("--demand-mean", type=float, default=6.0,
                    help="Poisson demand mean λ (default 6.0).")
    p.add_argument("--alpha", type=float, default=0.1,
                    help="Learning rate (default 0.1).")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor (default 0.99).")
    p.add_argument("--seed", "-s", type=int, default=42,
                    help="Random seed (default 42).")
    p.add_argument("--eval-episodes", type=int, default=100,
                    help="Evaluation episodes for common protocol (default 100).")
    p.add_argument("--eval-seed", type=int, default=123,
                    help="Evaluation seed for common protocol (default 123).")

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    results = None

    # ---- Training phase ----
    if not args.plot_only:
        results = train_all(
            agents=args.agents,
            episodes=args.episodes,
            shelf_life=args.shelf_life,
            max_order=args.max_order,
            max_inventory=args.max_inventory,
            demand_mean=args.demand_mean,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
        )

    # ---- Plotting phase ----
    if not args.no_plot:
        plot_all(results)

    # ---- Common evaluation phase ----
    if not args.no_eval:
        run_common_evaluation(args.eval_episodes, args.eval_seed)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All done in {elapsed:.1f}s")
    print(f"  Runs   → {os.path.abspath(RUNS_DIR)}")
    print(f"  Figures → {os.path.abspath(FIGURES_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
