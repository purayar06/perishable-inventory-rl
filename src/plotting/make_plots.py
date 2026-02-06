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

    # 1. Individual curves
    for r in runs:
        plot_learning_curve(
            r["rewards"],
            title=f"Learning Curve \u2013 {r['agent_type']}",
            save_path=os.path.join(output_dir, f"learning_curve_{r['label']}.png"),
        )
        plt.close()
        saved += 1

    # 2. Comparison overlay
    if len(runs) > 1:
        plot_comparison(
            {r["label"]: r["rewards"] for r in runs},
            title="Learning Curves \u2013 All Agents",
            save_path=os.path.join(output_dir, "learning_curves_comparison.png"),
        )
        plt.close()
        saved += 1

    # 3. Bar chart
    plot_bar_comparison(
        {r["label"]: r["summary"] for r in runs},
        metric_name="mean_reward", error_metric="std_reward",
        title="Mean Episode Reward (last 100 episodes)",
        ylabel="Mean Reward",
        save_path=os.path.join(output_dir, "bar_reward_comparison.png"),
    )
    plt.close()
    saved += 1

    # 4. Tradeoff scatter
    plot_waste_stockout_tradeoff(
        {r["label"]: r["summary"] for r in runs},
        title="Waste Rate vs Stockout Rate",
        save_path=os.path.join(output_dir, "waste_stockout_tradeoff.png"),
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
