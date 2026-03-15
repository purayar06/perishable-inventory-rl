# Streamlit Demo: Perishable Inventory RL

This folder contains a standalone Streamlit application for demonstrating how the project can be used in a real-life decision-support setting.

Important:
- This demo is fully isolated from the main project structure.
- You can remove this entire folder without impacting the core codebase.

## What the app does

1. Loads common evaluation results from `outputs/runs/evaluation_summary.json`.
2. Shows agent KPI comparison (reward, waste rate, stockout rate, fill rate).
3. Provides an operations simulator that runs day-by-day policy decisions.
4. Displays project figures from `outputs/figures`.

## Prerequisites

- Main project has already been run (`python main.py`) so outputs are available.
- Python 3.10+ recommended.

## Setup

From repository root:

1. Create and activate environment (optional but recommended).
2. Install demo dependencies:

   `pip install -r "streamlit demo/requirements.txt"`

## Run the app

From repository root:

`streamlit run "streamlit demo/app.py"`

## Suggested presentation flow

1. Executive Summary tab
- Explain business context and method ranking.

2. Agent Comparison tab
- Show common-protocol fairness and the waste-stockout tradeoff.

3. Operations Simulator tab
- Pick an agent, adjust demand/penalties, run simulation.
- Explain order recommendations and resulting service/waste outcomes.

4. Figures Gallery tab
- Connect dashboard story with report-ready figures.

## Notes

- If `evaluation_summary.json` is missing, run:
  `python -m src.experiments.evaluate --runs-dir outputs/runs --episodes 100 --seed 123 --save-path outputs/runs/evaluation_summary.json`

- If figures are missing, run:
  `python -m src.plotting.make_plots --results-dir outputs/runs --output-dir outputs/figures`
