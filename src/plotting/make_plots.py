"""
Plotting utilities for RL training visualization.

This module provides functions for generating training curves,
comparison plots, and other visualizations.
"""

import argparse
import os
import json
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive by default
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def smooth_curve(data: List[float], window: int = 50) -> np.ndarray:
    """Smooth a curve using moving average."""
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    padded = np.concatenate([data[: window - 1], smoothed])
    return padded


# ------------------------------------------------------------------ #
#  Individual learning curve                                          #
# ------------------------------------------------------------------ #

def plot_learning_curve(
    rewards: List[float],
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Episode Reward",
    smooth_window: int = 50,
    show_raw: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    episodes = np.arange(1, len(rewards) + 1)

    if show_raw:
        ax.plot(episodes, rewards, alpha=0.3, color="blue", linewidth=0.5, label="Raw")

    smoothed = smooth_curve(rewards, smooth_window)
    ax.plot(episodes, smoothed, color="blue", linewidth=2,
            label=f"Smoothed (w={smooth_window})")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Multi-algorithm comparison overlay                                 #
# ------------------------------------------------------------------ #

def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Episode Reward",
    smooth_window: int = 50,
    figsize: Tuple[int, int] = (12, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, rewards), color in zip(results.items(), colors):
        episodes = np.arange(1, len(rewards) + 1)
        smoothed = smooth_curve(rewards, smooth_window)
        ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.5)
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=name)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Bar chart                                                           #
# ------------------------------------------------------------------ #

def plot_bar_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "mean_reward",
    title: str = "Algorithm Performance Comparison",
    ylabel: str = "Mean Reward",
    show_error: bool = True,
    error_metric: str = "std_reward",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    names = list(metrics.keys())
    values = [metrics[n].get(metric_name, 0) for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=plt.cm.tab10(np.linspace(0, 1, len(names))))

    if show_error:
        errors = [metrics[n].get(error_metric, 0) for n in names]
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=5)

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(values),
            f"{val:.1f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Waste vs Stockout scatter                                          #
# ------------------------------------------------------------------ #

def plot_waste_stockout_tradeoff(
    results: Dict[str, Dict[str, float]],
    title: str = "Waste vs Stockout Tradeoff",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, m), color in zip(results.items(), colors):
        waste = m.get("waste_rate", m.get("mean_waste_rate", 0))
        stock = m.get("stockout_rate", m.get("mean_stockout_rate", 0))
        ax.scatter(waste * 100, stock * 100, s=200, color=color,
                   label=name, alpha=0.8, edgecolors="black", linewidth=1)

    ax.set_xlabel("Waste Rate (%)", fontsize=12)
    ax.set_ylabel("Stockout Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.annotate("Ideal", xy=(0, 0), xytext=(5, 5), fontsize=10, color="green",
                alpha=0.7, arrowprops=dict(arrowstyle="->", color="green", alpha=0.7))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Policy heatmap (D=2 only)                                          #
# ------------------------------------------------------------------ #

def plot_policy_heatmap(
    policy: Dict[Tuple[int, ...], int],
    shelf_life: int = 2,
    max_inventory: int = 10,
    title: str = "Policy Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    if shelf_life != 2:
        print("Policy heatmap only supported for shelf_life=2")
        return None

    matrix = np.zeros((max_inventory + 1, max_inventory + 1))
    for state, action in policy.items():
        if len(state) == 2:
            n1, n2 = state
            if 0 <= n1 <= max_inventory and 0 <= n2 <= max_inventory:
                matrix[n1, n2] = action

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", origin="lower")
    ax.set_xlabel("Items with 2 days (n2)", fontsize=12)
    ax.set_ylabel("Items with 1 day (n1)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Order Quantity", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Multi-metric panel from a metrics.json file                        #
# ------------------------------------------------------------------ #

def plot_metrics_over_training(
    metrics_file: str,
    metrics_to_plot: List[str] = None,
    smooth_window: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    with open(metrics_file, "r") as f:
        all_metrics = json.load(f)

    if metrics_to_plot is None:
        metrics_to_plot = ["total_reward", "waste_rate", "fill_rate"]

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    episodes = [m["episode"] for m in all_metrics]
    for ax, metric_name in zip(axes, metrics_to_plot):
        values = [m.get(metric_name, 0) for m in all_metrics]
        smoothed = smooth_curve(values, smooth_window)
        ax.plot(episodes, values, alpha=0.3, color="blue", linewidth=0.5)
        ax.plot(episodes, smoothed, color="blue", linewidth=2)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode", fontsize=12)
    fig.suptitle("Training Metrics Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Sensitivity analysis plots                                          #
# ------------------------------------------------------------------ #

def plot_sensitivity_by_param(
    results: List[Dict[str, Any]],
    param_name: str,
    metric: str = "mean_reward",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a sensitivity metric vs one swept parameter, with one line per agent.

    Args:
        results: List of result dicts from run_sensitivity_analysis().
        param_name: Which parameter to put on the x-axis
                    (e.g. "shelf_life", "demand_mean", "waste_penalty", "stockout_penalty").
        metric: Which metric to plot ("mean_reward", "mean_waste_rate", "mean_stockout_rate").
        title: Plot title (auto-generated if None).
        ylabel: Y-axis label (defaults to metric name).
    """
    import pandas as pd

    df = pd.DataFrame(results)
    if param_name not in df.columns or metric not in df.columns:
        print(f"  [WARN] Column '{param_name}' or '{metric}' not in results — skipping.")
        return plt.figure()

    # Average over the other swept parameters so we get one curve per agent
    group_cols = ["agent_type", param_name]
    agg = df.groupby(group_cols, as_index=False)[metric].mean()

    fig, ax = plt.subplots(figsize=figsize)
    for agent, grp in agg.groupby("agent_type"):
        grp_sorted = grp.sort_values(param_name)
        ax.plot(grp_sorted[param_name], grp_sorted[metric], marker="o", label=agent)

    ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs {param_name.replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def generate_sensitivity_plots(
    results_path: str,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Generate all sensitivity analysis plots from a saved JSON file.

    Produces one figure per (parameter × metric) combination:
      - shelf_life, demand_mean, waste_penalty, stockout_penalty
      × mean_reward, mean_waste_rate, mean_stockout_rate
    """
    import pandas as pd

    if not os.path.isfile(results_path):
        print(f"Sensitivity results not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    if not results:
        print("Sensitivity results file is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    params = ["shelf_life", "demand_mean", "waste_penalty", "stockout_penalty"]
    metrics = ["mean_reward", "mean_waste_rate", "mean_stockout_rate"]

    df = pd.DataFrame(results)
    present_params = [p for p in params if p in df.columns]
    present_metrics = [m for m in metrics if m in df.columns]

    saved = 0
    for param in present_params:
        for metric in present_metrics:
            fname = f"sensitivity_{param}_{metric}.png"
            plot_sensitivity_by_param(
                results,
                param_name=param,
                metric=metric,
                save_path=os.path.join(output_dir, fname),
            )
            plt.close()
            saved += 1

    print(f"\nSensitivity analysis: saved {saved} plots to {output_dir}")


# ------------------------------------------------------------------ #
#  DP convergence curve                                                #
# ------------------------------------------------------------------ #

def plot_dp_convergence(
    deltas: List[float],
    title: str = "DP Value Iteration Convergence",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the Bellman error (delta) over VI iterations."""
    fig, ax = plt.subplots(figsize=figsize)
    iterations = np.arange(1, len(deltas) + 1)
    ax.semilogy(iterations, deltas, color="blue", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Bellman Error (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ------------------------------------------------------------------ #
#  Batch plot generation from outputs/runs                            #
# ------------------------------------------------------------------ #

def _load_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """Safely load a single training run."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.isfile(metrics_path):
        return None
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [WARN] Skipping {run_dir}: corrupted metrics.json ({exc})")
        return None
    if not metrics:
        return None

    label = os.path.basename(run_dir)
    agent_type = label
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            agent_type = cfg.get("agent_type", label)
        except (json.JSONDecodeError, ValueError):
            pass

    rewards = [m["total_reward"] for m in metrics]
    waste_rates = [m.get("waste_rate", 0.0) for m in metrics]
    stockout_rates = [m.get("stockout_rate", 0.0) for m in metrics]
    fill_rates = [m.get("fill_rate", 1.0) for m in metrics]

    tail = metrics[-min(100, len(metrics)):]
    return {
        "label": label,
        "agent_type": agent_type,
        "rewards": rewards,
        "waste_rates": waste_rates,
        "stockout_rates": stockout_rates,
        "fill_rates": fill_rates,
        "summary": {
            "mean_reward": float(np.mean([m["total_reward"] for m in tail])),
            "std_reward": float(np.std([m["total_reward"] for m in tail])),
            "mean_waste_rate": float(np.mean([m.get("waste_rate", 0) for m in tail])),
            "mean_stockout_rate": float(np.mean([m.get("stockout_rate", 0) for m in tail])),
            "mean_fill_rate": float(np.mean([m.get("fill_rate", 1) for m in tail])),
        },
    }


def generate_all_plots(
    results_dir: str,
    output_dir: str = "outputs/figures",
) -> None:
    """Generate all standard plots from training run directories."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating plots from {results_dir}")
    print(f"Saving to {output_dir}\n")

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    runs: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, entry)
        if not os.path.isdir(full):
            continue
        run = _load_run(full)
        if run is not None:
            runs.append(run)

    if not runs:
        print("No valid training runs found.")
        return

    saved = 0

    # Separate DP baseline (single-point, no learning trajectory)
    # from episodic RL runs that have actual learning curves.
    rl_runs = [r for r in runs if r["agent_type"] != "dp"]

    # 1. Individual curves (RL agents only)
    for r in rl_runs:
        plot_learning_curve(
            r["rewards"],
            title=f"Learning Curve \u2013 {r['agent_type']}",
            save_path=os.path.join(output_dir, f"learning_curve_{r['label']}.png"),
        )
        plt.close()
        saved += 1

    # 2. Comparison overlay (RL agents only)
    if len(rl_runs) > 1:
        plot_comparison(
            {r["label"]: r["rewards"] for r in rl_runs},
            title="Learning Curves \u2013 All Agents",
            save_path=os.path.join(output_dir, "learning_curves_comparison.png"),
        )
        plt.close()
        saved += 1

    # If available, use a common fixed-seed evaluation summary for
    # cross-agent comparison metrics (apples-to-apples protocol).
    eval_summary_path = os.path.join(results_dir, "evaluation_summary.json")
    comparison_source = {r["label"]: r["summary"] for r in runs}
    reward_title = "Mean Episode Reward (last 100 episodes)"
    if os.path.isfile(eval_summary_path):
        try:
            with open(eval_summary_path, "r") as f:
                eval_data = json.load(f)
            comparison_source = {
                k: {
                    "mean_reward": v.get("evaluation", {}).get("mean_reward", 0.0),
                    "std_reward": v.get("evaluation", {}).get("std_reward", 0.0),
                    "mean_waste_rate": v.get("evaluation", {}).get("mean_waste_rate", 0.0),
                    "mean_stockout_rate": v.get("evaluation", {}).get("mean_stockout_rate", 0.0),
                    "mean_fill_rate": v.get("evaluation", {}).get("mean_fill_rate", 1.0),
                }
                for k, v in eval_data.items()
            }
            print("Using evaluation_summary.json for comparison bar/scatter metrics")
            reward_title = "Mean Episode Reward (common evaluation protocol)"
        except (json.JSONDecodeError, ValueError, AttributeError):
            print("  [WARN] Invalid evaluation_summary.json, falling back to training summaries")

    # 3. Bar chart
    plot_bar_comparison(
        comparison_source,
        metric_name="mean_reward", error_metric="std_reward",
        title=reward_title,
        ylabel="Mean Reward",
        save_path=os.path.join(output_dir, "bar_reward_comparison.png"),
    )
    plt.close()
    saved += 1

    # 4. Tradeoff scatter
    plot_waste_stockout_tradeoff(
        comparison_source,
        title="Waste Rate vs Stockout Rate",
        save_path=os.path.join(output_dir, "waste_stockout_tradeoff.png"),
    )
    plt.close()
    saved += 1

    # 5. DP convergence curve (if convergence.json exists)
    dp_conv_path = os.path.join(results_dir, "dp", "convergence.json")
    if os.path.isfile(dp_conv_path):
        with open(dp_conv_path, "r") as f:
            deltas = json.load(f)
        plot_dp_convergence(
            deltas,
            save_path=os.path.join(output_dir, "dp_convergence.png"),
        )
        plt.close()
        saved += 1

    print(f"\nDone \u2013 saved {saved} plots to {output_dir}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument("--results-dir", type=str, default="outputs/runs")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--metrics-file", type=str, default=None)
    args = parser.parse_args()

    if args.metrics_file:
        plot_metrics_over_training(
            args.metrics_file,
            save_path=os.path.join(args.output_dir, "metrics_plot.png"),
        )
    else:
        generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
