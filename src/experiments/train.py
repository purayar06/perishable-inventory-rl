"""
Training script for RL agents.

This module provides functions and CLI for training different agents
on the perishable inventory environment.
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from tqdm import tqdm

from ..config import Config, EnvConfig, TrainingConfig, DPConfig, get_config
from ..envs.perishable_inventory import PerishableInventoryEnv
from ..agents.q_learning import QLearningAgent
from ..agents.sarsa import SARSAAgent
from ..agents.mc_control import MonteCarloAgent
from ..agents.dp_value_iteration import DPValueIterationAgent
from ..utils.logging import Logger, MetricsTracker
from ..utils.seeding import set_global_seed
from ..plotting.make_plots import (
    plot_learning_curve,
    plot_metrics_over_training,
    generate_all_plots,
)


def create_agent(
    agent_type: str,
    num_actions: int,
    config: TrainingConfig,
    seed: Optional[int] = None,
):
    """
    Create an agent of the specified type.

    Args:
        agent_type: One of 'qlearning', 'sarsa', 'mc'
        num_actions: Number of actions in the environment
        config: Training configuration
        seed: Random seed

    Returns:
        Agent instance
    """
    agent_classes = {
        "qlearning": QLearningAgent,
        "sarsa": SARSAAgent,
        "mc": MonteCarloAgent,
    }

    if agent_type not in agent_classes:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Choose from {list(agent_classes.keys())}"
        )

    agent_class = agent_classes[agent_type]

    return agent_class(
        num_actions=num_actions,
        gamma=config.gamma,
        alpha=config.alpha,
        epsilon_start=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        seed=seed,
    )


def train_agent(
    agent_type: str,
    config: Optional[Config] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train an RL agent on the perishable inventory environment.

    Args:
        agent_type: Type of agent ('qlearning', 'sarsa', 'mc')
        config: Configuration object
        verbose: Whether to print progress
        save_path: Path to save trained agent and metrics

    Returns:
        Dictionary with training results
    """
    if config is None:
        config = Config()

    # Set global seed
    set_global_seed(config.seed)

    # Create environment
    env = PerishableInventoryEnv(config=config.env, seed=config.seed)

    # Create agent
    agent = create_agent(
        agent_type=agent_type,
        num_actions=env.num_actions,
        config=config.training,
        seed=config.seed,
    )

    # Setup logging
    logger = Logger(name=agent.name, verbose=verbose)
    tracker = MetricsTracker()

    # Training loop
    episodes = config.training.episodes
    log_interval = config.training.log_interval

    iterator = range(episodes)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {agent.name}")

    for episode in iterator:
        tracker.start_episode(episode)

        # Train one episode
        stats = agent.train_episode(env)

        # Record metrics
        tracker.record_step(
            reward=stats["total_reward"],
            sold=stats["total_sold"],
            waste=stats["total_waste"],
            stockout=stats["total_stockout"],
        )
        tracker.end_episode()

        # Log progress
        if verbose and (episode + 1) % log_interval == 0:
            recent_stats = tracker.get_recent_stats(log_interval)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    reward=f"{recent_stats['mean_reward']:.1f}",
                    eps=f"{agent.epsilon:.3f}",
                )

    # Final statistics
    final_stats = tracker.get_recent_stats(100)

    if verbose:
        logger.summary(final_stats)

    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Save agent
        agent.save(os.path.join(save_path, "agent.pkl"))

        # Save metrics
        tracker.save(os.path.join(save_path, "metrics.json"))

        # Save config
        config_dict = {
            "agent_type": agent_type,
            "env": {
                "shelf_life": config.env.shelf_life,
                "max_order": config.env.max_order,
                "max_inventory": config.env.max_inventory,
                "horizon": config.env.horizon,
                "demand_mean": config.env.demand_mean,
            },
            "training": {
                "episodes": config.training.episodes,
                "gamma": config.training.gamma,
                "alpha": config.training.alpha,
                "epsilon_start": config.training.epsilon_start,
                "epsilon_min": config.training.epsilon_min,
                "epsilon_decay": config.training.epsilon_decay,
            },
            "seed": config.seed,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        if verbose:
            logger.info(f"Results saved to {save_path}")

    return {
        "agent": agent,
        "tracker": tracker,
        "final_stats": final_stats,
        "all_rewards": tracker.get_all_rewards(),
        "save_path": save_path,
    }


def train_dp(
    config: Optional[Config] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Solve the MDP using Dynamic Programming (Value Iteration).

    Args:
        config: Configuration object
        verbose: Whether to print progress
        save_path: Path to save results

    Returns:
        Dictionary with DP solution and evaluation
    """
    if config is None:
        config = Config()

    set_global_seed(config.seed)

    env = PerishableInventoryEnv(config=config.env, seed=config.seed)

    dp_agent = DPValueIterationAgent(env, config.dp, verbose=verbose)
    solve_stats = dp_agent.solve()

    eval_stats = dp_agent.evaluate_policy(
        env,
        num_episodes=config.training.eval_episodes,
        seed=config.training.eval_seed,
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        dp_agent.save(os.path.join(save_path, "dp_agent.pkl"))

        results = {
            "solve_stats": solve_stats,
            "eval_stats": eval_stats,
            "policy_summary": dp_agent.get_policy_summary(),
        }
        with open(os.path.join(save_path, "dp_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            print(f"\nDP results saved to {save_path}")

    return {
        "agent": dp_agent,
        "solve_stats": solve_stats,
        "eval_stats": eval_stats,
    }


# ------------------------------------------------------------------ #
#  Auto-plot helper                                                    #
# ------------------------------------------------------------------ #

def generate_post_training_plots(
    save_path: str,
    agent_type: str,
    rewards: list,
    figures_dir: str = "outputs/figures",
    verbose: bool = True,
) -> None:
    """
    Auto-generate plots after a training run finishes.

    Produces:
    - A learning curve for this specific run.
    - A multi-metric panel (reward, waste-rate, fill-rate) from metrics.json.
    - If other runs exist in the same parent directory, also produces
      comparison plots (overlay, bar chart, waste-vs-stockout scatter).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for auto-saving
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)

    # 1. Individual learning curve
    lc_path = os.path.join(figures_dir, f"learning_curve_{agent_type}.png")
    plot_learning_curve(
        rewards,
        title=f"Learning Curve \u2013 {agent_type}",
        save_path=lc_path,
    )
    plt.close()

    # 2. Multi-metric panel (reward + waste + fill)
    metrics_file = os.path.join(save_path, "metrics.json")
    if os.path.isfile(metrics_file):
        panel_path = os.path.join(
            figures_dir, f"training_metrics_{agent_type}.png"
        )
        plot_metrics_over_training(
            metrics_file,
            metrics_to_plot=["total_reward", "waste_rate", "fill_rate"],
            save_path=panel_path,
        )
        plt.close()

    # 3. Cross-run comparison (if sibling runs exist)
    results_dir = os.path.dirname(save_path)  # e.g. outputs/runs
    if results_dir and os.path.isdir(results_dir):
        sibling_dirs = [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]
        if len(sibling_dirs) > 1:
            if verbose:
                print(
                    f"\nFound {len(sibling_dirs)} runs "
                    f"\u2013 generating comparison plots..."
                )
            generate_all_plots(results_dir, figures_dir)

    if verbose:
        print(f"Plots saved to {figures_dir}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on perishable inventory environment"
    )

    parser.add_argument(
        "--agent", "-a",
        type=str, default="qlearning",
        choices=["qlearning", "sarsa", "mc", "dp"],
        help="Agent type to train",
    )
    parser.add_argument("--episodes", "-e", type=int, default=3000,
                        help="Number of training episodes")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--shelf-life", "-d", type=int, default=5,
                        help="Shelf life (D)")
    parser.add_argument("--max-order", type=int, default=10,
                        help="Maximum order quantity (Amax)")
    parser.add_argument("--max-inventory", type=int, default=20,
                        help="Maximum inventory per bucket (Nmax)")
    parser.add_argument("--demand-mean", type=float, default=6.0,
                        help="Mean demand (lambda)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip automatic plot generation after training")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures",
                        help="Directory to save auto-generated plots")

    args = parser.parse_args()

    # Build configuration
    config = Config(
        env=EnvConfig(
            shelf_life=args.shelf_life,
            max_order=args.max_order,
            max_inventory=args.max_inventory,
            demand_mean=args.demand_mean,
        ),
        training=TrainingConfig(
            episodes=args.episodes,
            gamma=args.gamma,
            alpha=args.alpha,
        ),
        seed=args.seed,
    )

    # Determine output path
    if args.output:
        save_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("outputs", "runs", f"{args.agent}_{timestamp}")

    # Train
    if args.agent == "dp":
        results = train_dp(
            config=config,
            verbose=not args.quiet,
            save_path=save_path,
        )
        print(f"\nDP Evaluation: Mean reward = {results['eval_stats']['mean_reward']:.2f}")
    else:
        results = train_agent(
            agent_type=args.agent,
            config=config,
            verbose=not args.quiet,
            save_path=save_path,
        )
        print(f"\nFinal: Mean reward = {results['final_stats']['mean_reward']:.2f}")

        # ---- Auto-generate plots ----
        if not args.no_plot:
            generate_post_training_plots(
                save_path=save_path,
                agent_type=args.agent,
                rewards=results["all_rewards"],
                figures_dir=args.figures_dir,
                verbose=not args.quiet,
            )


if __name__ == "__main__":
    main()
