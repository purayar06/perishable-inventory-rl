import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DPConfig, EnvConfig
from src.envs.perishable_inventory import PerishableInventoryEnv
from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.agents.mc_control import MonteCarloAgent
from src.agents.linear_fa import LinearFAAgent
from src.agents.dp_value_iteration import DPValueIterationAgent


RUNS_DIR = os.path.join(PROJECT_ROOT, "outputs", "runs")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
SUMMARY_PATH = os.path.join(RUNS_DIR, "evaluation_summary.json")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }

        .hero {
            background: linear-gradient(135deg, #0b3954 0%, #087e8b 45%, #bfd7ea 100%);
            border-radius: 20px;
            padding: 24px 28px;
            color: #ffffff;
            box-shadow: 0 8px 28px rgba(8, 32, 50, 0.18);
            margin-bottom: 12px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-family: 'IBM Plex Serif', serif;
            letter-spacing: 0.2px;
        }

        .hero p {
            margin: 8px 0 0 0;
            opacity: 0.96;
            line-height: 1.45;
            font-size: 1rem;
        }

        .chip {
            display: inline-block;
            background: #eff8ff;
            color: #0b3954;
            border: 1px solid #b6e0fe;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.85rem;
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_summary() -> Dict[str, Any]:
    if not os.path.isfile(SUMMARY_PATH):
        return {}
    return _read_json(SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def list_run_agents() -> List[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(
        [
            name
            for name in os.listdir(RUNS_DIR)
            if os.path.isdir(os.path.join(RUNS_DIR, name))
        ]
    )


def _agent_file_for(agent_name: str) -> str:
    if agent_name == "dp":
        return os.path.join(RUNS_DIR, agent_name, "dp_agent.pkl")
    return os.path.join(RUNS_DIR, agent_name, "agent.pkl")


def _load_agent(agent_name: str, env: PerishableInventoryEnv, cfg: Dict[str, Any], seed: int):
    path = _agent_file_for(agent_name)
    if agent_name == "qlearning":
        return QLearningAgent.load(path, seed=seed)
    if agent_name == "sarsa":
        return SARSAAgent.load(path, seed=seed)
    if agent_name == "mc":
        return MonteCarloAgent.load(path, seed=seed)
    if agent_name == "linear_fa":
        return LinearFAAgent.load(path, seed=seed)
    if agent_name == "dp":
        dp_cfg = DPConfig(
            theta=cfg.get("dp", {}).get("theta", 1e-6),
            max_iter=cfg.get("dp", {}).get("max_iter", 2000),
        )
        gamma = cfg.get("training", {}).get("gamma", 0.99)
        dp_agent = DPValueIterationAgent(env=env, config=dp_cfg, gamma=gamma, verbose=False)
        dp_agent.load_from(path)
        return dp_agent
    raise ValueError(f"Unknown agent: {agent_name}")


def _load_run_config(agent_name: str) -> Dict[str, Any]:
    cfg_path = os.path.join(RUNS_DIR, agent_name, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    return _read_json(cfg_path)


def summary_to_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, val in summary.items():
        ev = val.get("evaluation", {})
        rows.append(
            {
                "agent": key,
                "mean_reward": ev.get("mean_reward", 0.0),
                "std_reward": ev.get("std_reward", 0.0),
                "waste_rate": ev.get("mean_waste_rate", 0.0),
                "stockout_rate": ev.get("mean_stockout_rate", 0.0),
                "fill_rate": ev.get("mean_fill_rate", 0.0),
                "sold": ev.get("total_sold", 0),
                "waste": ev.get("total_waste", 0),
                "stockout": ev.get("total_stockout", 0),
                "ordered": ev.get("total_ordered", 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    return df


def run_day_by_day_simulation(
    agent_name: str,
    days: int,
    seed: int,
    demand_mean: float,
    waste_penalty: float,
    stockout_penalty: float,
    initial_inventory: List[int],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cfg = _load_run_config(agent_name)
    env_cfg_dict = cfg.get("env", {})

    env_cfg = EnvConfig(
        shelf_life=env_cfg_dict.get("shelf_life", 3),
        max_order=env_cfg_dict.get("max_order", 5),
        max_inventory=env_cfg_dict.get("max_inventory", 10),
        horizon=max(days, env_cfg_dict.get("horizon", 60)),
        demand_mean=demand_mean,
        waste_penalty=waste_penalty,
        stockout_penalty=stockout_penalty,
    )

    env = PerishableInventoryEnv(config=env_cfg, seed=seed)
    agent = _load_agent(agent_name, env, cfg, seed)

    state, _ = env.reset(seed=seed)
    if len(initial_inventory) == env_cfg.shelf_life:
        state = tuple(max(0, min(env_cfg.max_inventory, int(x))) for x in initial_inventory)
        env._state = state  # Controlled scenario injection for demonstration.

    rows = []
    total_reward = 0.0
    total_sold = 0
    total_waste = 0
    total_stockout = 0
    total_ordered = 0

    for day in range(1, days + 1):
        action = int(agent.select_action(state, training=False))
        next_state, reward, terminated, truncated, info = env.step(action)

        rows.append(
            {
                "day": day,
                "state_before": str(state),
                "order": action,
                "demand": int(info["demand"]),
                "sold": int(info["sold"]),
                "waste": int(info["waste"]),
                "stockout": int(info["stockout"]),
                "reward": float(reward),
                "state_after": str(next_state),
            }
        )

        total_reward += float(reward)
        total_sold += int(info["sold"])
        total_waste += int(info["waste"])
        total_stockout += int(info["stockout"])
        total_ordered += int(info["order"])

        state = next_state
        if terminated or truncated:
            break

    demand_total = total_sold + total_stockout
    metrics = {
        "total_reward": total_reward,
        "waste_rate": (total_waste / total_ordered) if total_ordered > 0 else 0.0,
        "stockout_rate": (total_stockout / demand_total) if demand_total > 0 else 0.0,
        "fill_rate": (total_sold / demand_total) if demand_total > 0 else 1.0,
        "total_ordered": total_ordered,
        "total_sold": total_sold,
        "total_waste": total_waste,
        "total_stockout": total_stockout,
    }

    return pd.DataFrame(rows), metrics


def build_initial_inventory(
    shelf_life: int,
    max_inventory: int,
    total_units: int,
    fresh_share_pct: int,
) -> List[int]:
    """Construct age-bucket inventory from simple business inputs.

    Buckets are ordered oldest -> freshest.
    """
    if shelf_life <= 0:
        return []

    capped_total = max(0, min(total_units, shelf_life * max_inventory))
    fresh_units = int(round(capped_total * (fresh_share_pct / 100.0)))
    fresh_units = min(fresh_units, max_inventory)

    remaining = capped_total - fresh_units
    buckets = [0] * shelf_life
    buckets[-1] = fresh_units

    # Fill older buckets from near-fresh to oldest for a realistic shelf profile.
    for idx in range(shelf_life - 2, -1, -1):
        put = min(max_inventory, remaining)
        buckets[idx] = put
        remaining -= put
        if remaining <= 0:
            break

    return buckets


def render_overview(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    if df.empty:
        c1.metric("Best Agent", "N/A")
        c2.metric("Best Reward", "N/A")
        c3.metric("Best Fill Rate", "N/A")
        c4.metric("Lowest Stockout", "N/A")
        return

    best = df.iloc[0]
    best_fill = df.sort_values("fill_rate", ascending=False).iloc[0]
    low_stock = df.sort_values("stockout_rate", ascending=True).iloc[0]

    c1.metric("Best Agent", str(best["agent"]))
    c2.metric("Best Reward", f"{best['mean_reward']:.2f}")
    c3.metric("Best Fill Rate", f"{best_fill['fill_rate']*100:.2f}%")
    c4.metric("Lowest Stockout", f"{low_stock['stockout_rate']*100:.2f}%")


def render_comparison(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No evaluation_summary.json found yet. Run main.py first.")
        return

    st.dataframe(
        df.assign(
            waste_rate_pct=(df["waste_rate"] * 100).round(2),
            stockout_rate_pct=(df["stockout_rate"] * 100).round(2),
            fill_rate_pct=(df["fill_rate"] * 100).round(2),
        )[[
            "agent",
            "mean_reward",
            "std_reward",
            "waste_rate_pct",
            "stockout_rate_pct",
            "fill_rate_pct",
            "ordered",
            "sold",
        ]],
        use_container_width=True,
    )

    fig_bar = px.bar(
        df,
        x="agent",
        y="mean_reward",
        error_y="std_reward",
        title="Mean Reward Under Common Evaluation Protocol",
        color="agent",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_scatter = px.scatter(
        df,
        x="waste_rate",
        y="stockout_rate",
        size="fill_rate",
        color="agent",
        hover_data=["mean_reward"],
        title="Waste vs Stockout Tradeoff",
    )
    fig_scatter.update_layout(
        xaxis_title="Waste Rate",
        yaxis_title="Stockout Rate",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def render_simulator() -> None:
    st.subheader("Real-Life Decision Simulator")
    st.caption("Pick a trained agent, adjust business settings, and simulate daily ordering decisions.")

    agents = [a for a in list_run_agents() if a != "evaluation_summary.json"]
    if not agents:
        st.warning("No trained runs found in outputs/runs.")
        return

    left, right = st.columns([1, 1])
    with left:
        selected_agent = st.selectbox("Agent", options=agents, index=0)
        days = st.slider("Simulation Days", min_value=7, max_value=60, value=30)
        sim_seed = st.number_input(
            "Scenario Replay Number",
            min_value=1,
            max_value=99999,
            value=123,
            help="Use the same number to replay the same random demand pattern.",
        )

    cfg = _load_run_config(selected_agent)
    env_cfg = cfg.get("env", {})
    D = int(env_cfg.get("shelf_life", 3))
    Nmax = int(env_cfg.get("max_inventory", 10))
    Amax = int(env_cfg.get("max_order", 5))
    default_demand_mean = float(env_cfg.get("demand_mean", 6.0))

    with right:
        avg_customers = st.slider(
            "Average Customers Per Day",
            min_value=1,
            max_value=50,
            value=max(1, int(round(default_demand_mean))),
            step=1,
            help="Typical number of customers expected daily.",
        )
        avg_units_per_customer = st.slider(
            "Average Units Per Customer",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="On average, how many units one customer buys.",
        )
        demand_mean = float(avg_customers * avg_units_per_customer)
        st.caption(f"Calculated demand level used by simulator: {demand_mean:.2f} units/day")
        cost_of_expiry = st.slider(
            "Cost Of Expiry (Per Unit)",
            min_value=1.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
            help="Higher means expired product is more painful financially.",
        )
        cost_of_lost_sale = st.slider(
            "Cost Of Lost Sale (Per Unit)",
            min_value=1.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            help="Higher means unmet demand is more painful financially.",
        )

    st.markdown("Starting stock (simplified):")
    stock_col1, stock_col2 = st.columns(2)
    with stock_col1:
        starting_total_stock = st.slider(
            "Total Starting Units",
            min_value=0,
            max_value=D * Nmax,
            value=min(D * Nmax, max(0, int(round(default_demand_mean * 1.2)))),
            step=1,
            help="Total units available at day 1 before ordering.",
        )
    with stock_col2:
        fresh_share_pct = st.slider(
            "Fresh Stock Share (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            help="How much of starting stock is fresh (higher shelf-life left).",
        )

    initial_inventory = build_initial_inventory(
        shelf_life=D,
        max_inventory=Nmax,
        total_units=starting_total_stock,
        fresh_share_pct=fresh_share_pct,
    )

    st.caption(
        "Simulator auto-converts business inputs into model parameters: "
        f"waste_penalty={cost_of_expiry:.1f}, stockout_penalty={cost_of_lost_sale:.1f}, "
        f"max_order={Amax}, age-buckets(oldest->freshest)={initial_inventory}"
    )

    if st.button("Run Simulation", type="primary"):
        sim_df, sim_metrics = run_day_by_day_simulation(
            agent_name=selected_agent,
            days=days,
            seed=int(sim_seed),
            demand_mean=float(demand_mean),
            waste_penalty=float(cost_of_expiry),
            stockout_penalty=float(cost_of_lost_sale),
            initial_inventory=initial_inventory,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reward", f"{sim_metrics['total_reward']:.2f}")
        m2.metric("Waste Rate", f"{sim_metrics['waste_rate']*100:.2f}%")
        m3.metric("Stockout Rate", f"{sim_metrics['stockout_rate']*100:.2f}%")
        m4.metric("Fill Rate", f"{sim_metrics['fill_rate']*100:.2f}%")

        fig_ops = px.line(
            sim_df,
            x="day",
            y=["order", "demand", "sold", "waste", "stockout"],
            title="Daily Operations Trace",
            markers=True,
        )
        st.plotly_chart(fig_ops, use_container_width=True)
        st.dataframe(sim_df, use_container_width=True)

        st.info(
            "Business takeaway: this simulation shows how a policy converts daily stock position into an order decision while balancing service and perishability costs."
        )


def render_figures() -> None:
    st.subheader("Generated Project Figures")
    st.caption("These are the same artifacts used in your main report workflow.")

    figure_files = [
        "bar_reward_comparison.png",
        "waste_stockout_tradeoff.png",
        "learning_curves_comparison.png",
        "dp_convergence.png",
        "training_metrics_qlearning.png",
        "training_metrics_sarsa.png",
        "training_metrics_mc.png",
        "training_metrics_linear_fa.png",
    ]

    existing = [f for f in figure_files if os.path.isfile(os.path.join(FIGURES_DIR, f))]
    if not existing:
        st.warning("No figure files found in outputs/figures yet.")
        return

    for f in existing:
        st.markdown(f"### {f}")
        st.image(os.path.join(FIGURES_DIR, f), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Perishable Inventory RL Demo",
        page_icon="📦",
        layout="wide",
    )
    _inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>Perishable Inventory RL Decision Dashboard</h1>
            <p>Academic demo that translates RL outputs into practical daily replenishment decisions under shelf-life constraints.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="chip">Common Evaluation Protocol</span>'
        '<span class="chip">Real-Life Simulator</span>'
        '<span class="chip">Agent Benchmarking</span>',
        unsafe_allow_html=True,
    )

    summary = load_summary()
    df = summary_to_df(summary)

    render_overview(df)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Executive Summary",
            "Agent Comparison",
            "Operations Simulator",
            "Figures Gallery",
        ]
    )

    with tab1:
        st.subheader("What This Demo Shows")
        st.write(
            "This dashboard explains how the trained policies can be used as a decision-support system for real operations such as grocery, pharmacy, blood-bank, or vaccine inventory management."
        )
        st.write(
            "Use the simulator tab to show day-by-day recommendations and the comparison tab to justify algorithm choice with KPI evidence."
        )

        if not df.empty:
            st.markdown("#### Current Ranking (Common Evaluation)")
            for idx, row in df.iterrows():
                st.write(
                    f"{idx+1}. {row['agent']} | reward={row['mean_reward']:.2f} | "
                    f"waste={row['waste_rate']*100:.2f}% | stockout={row['stockout_rate']*100:.2f}% | "
                    f"fill={row['fill_rate']*100:.2f}%"
                )

    with tab2:
        render_comparison(df)

    with tab3:
        render_simulator()

    with tab4:
        render_figures()


if __name__ == "__main__":
    main()
