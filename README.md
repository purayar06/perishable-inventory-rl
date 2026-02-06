# Perishable Inventory Control with Shelf-Life Constraints

A classical Reinforcement Learning project for managing perishable inventory with shelf-life constraints.

## Project Overview

This project implements a custom MDP (Markov Decision Process) for perishable inventory control and solves it using classical RL algorithms:

- **Dynamic Programming (Value Iteration)**: Baseline solution on bounded state space
- **Q-learning**: Tabular, epsilon-greedy exploration
- **SARSA**: Tabular, on-policy TD learning
- **Monte Carlo Control**: First-visit MC with epsilon-greedy policy

## Problem Description

### Environment
- **Shelf-life**: D days (default: 5)
- **State**: S_t = (n1, n2, ..., nD) where ni = number of items with remaining shelf-life i days
- **Action**: Order quantity A_t ∈ {0, 1, ..., Amax}
- **Demand**: Poisson(λ) distributed, truncated for DP
- **Selling rule**: FEFO (First-Expire-First-Out)

### Daily Step Order (Critical Logic)
1. Add ordered items to bucket D (freshest)
2. Sample demand from Poisson distribution
3. Sell inventory using FEFO (oldest first)
4. Items remaining in bucket 1 after sales become **waste**
5. Age inventory downward: n1←n2, ..., n(D-1)←nD, nD←0

### Reward Structure
```
R = p × sold - c × ordered - w × waste - s × stockout
```
Where:
- p = selling price (default: 10)
- c = ordering cost (default: 4)
- w = waste penalty (default: 6)
- s = stockout penalty (default: 8)

## Directory Structure

```
perishable-inventory-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                 # All hyperparameters and settings
│   ├── envs/
│   │   ├── __init__.py
│   │   └── perishable_inventory.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # Base classes for agents
│   │   ├── dp_value_iteration.py # DP baseline
│   │   ├── q_learning.py
│   │   ├── sarsa.py
│   │   └── mc_control.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── linear_features.py    # Placeholder for function approximation
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── train.py              # Training script
│   │   ├── evaluate.py           # Evaluation script
│   │   └── sweep.py              # Hyperparameter sweep
│   ├── plotting/
│   │   ├── __init__.py
│   │   └── make_plots.py
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py
│       ├── spaces.py
│       └── logging.py
├── outputs/
│   ├── runs/                     # Training outputs
│   └── figures/                  # Generated plots
└── tests/
    ├── test_env.py               # Environment unit tests
    └── test_agents_smoke.py      # Agent smoke tests
```

## Installation

```bash
# Navigate to the project directory
cd perishable-inventory-rl

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run environment tests only
python -m pytest tests/test_env.py -v

# Run agent smoke tests
python -m pytest tests/test_agents_smoke.py -v
```

### 2. Train Agents

```bash
# Train Q-learning agent (default parameters)
python -m src.experiments.train --agent qlearning --episodes 3000

# Train SARSA agent
python -m src.experiments.train --agent sarsa --episodes 3000

# Train Monte Carlo agent
python -m src.experiments.train --agent mc --episodes 3000

# Run Dynamic Programming (Value Iteration) baseline
# Note: Use smaller state space for tractability
python -m src.experiments.train --agent dp --shelf-life 3 --max-order 5 --max-inventory 10
```

### 3. Training with Custom Parameters

```bash
# Custom environment parameters
python -m src.experiments.train \
    --agent qlearning \
    --episodes 5000 \
    --shelf-life 3 \
    --max-order 8 \
    --max-inventory 15 \
    --demand-mean 5.0 \
    --alpha 0.15 \
    --gamma 0.99 \
    --seed 42 \
    --output outputs/runs/my_experiment
```

### 4. Hyperparameter Sweep

```bash
# Run a parameter sweep
python -m src.experiments.sweep \
    --agents qlearning sarsa \
    --alphas 0.05 0.1 0.2 \
    --epsilon-decays 0.99 0.995 0.999 \
    --episodes 1000 \
    --seeds 3
```

### 5. Evaluate Trained Agent

```bash
# Evaluate a trained agent
python -m src.experiments.evaluate \
    --agent-path outputs/runs/qlearning_*/agent.pkl \
    --agent-type qlearning \
    --episodes 100 \
    --seed 42
```

### 6. Generate Plots

```bash
# Generate plots from training results
python -m src.plotting.make_plots --results-dir outputs/runs --output-dir outputs/figures
```

## Usage as a Library

```python
from src.config import Config, EnvConfig, TrainingConfig
from src.envs import PerishableInventoryEnv
from src.agents import QLearningAgent, SARSAAgent, DPValueIterationAgent

# Create environment
config = EnvConfig(shelf_life=5, max_order=10, demand_mean=6.0, horizon=60)
env = PerishableInventoryEnv(config=config, seed=42)

# Create and train Q-learning agent
agent = QLearningAgent(
    num_actions=env.num_actions,
    gamma=0.99,
    alpha=0.1,
    seed=42
)

# Training loop
for episode in range(1000):
    stats = agent.train_episode(env)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Reward = {stats['total_reward']:.1f}")

# Evaluate
state, _ = env.reset()
total_reward = 0
while True:
    action = agent.select_action(state, training=False)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break
print(f"Evaluation reward: {total_reward:.1f}")
```

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| D | 5 | Shelf-life in days |
| Amax | 10 | Maximum order quantity |
| Nmax | 20 | Maximum items per bucket (DP) |
| T | 60 | Episode horizon |
| γ (gamma) | 0.99 | Discount factor |
| λ (lambda) | 6 | Poisson demand rate |
| p | 10 | Selling price |
| c | 4 | Ordering cost |
| w | 6 | Waste penalty |
| s | 8 | Stockout penalty |
| episodes | 3000 | Training episodes |
| α (alpha) | 0.1 | Learning rate |
| ε_start | 1.0 | Initial exploration rate |
| ε_min | 0.05 | Minimum exploration rate |
| ε_decay | 0.995 | Exploration decay rate |

## Algorithm Details

### Value Iteration (DP)
- Enumerates bounded state space: O((Nmax+1)^D) states
- Uses truncated Poisson demand (K ≈ 30)
- Provides optimal policy baseline
- Computationally expensive for large D or Nmax

### Q-Learning
- Off-policy TD control
- Update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
- Uses maximum Q-value of next state

### SARSA
- On-policy TD control  
- Update: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
- Uses Q-value of action actually taken

### Monte Carlo Control
- Learns from complete episodes
- First-visit updates
- No bootstrapping

## Authors

DSBA M2 - Reinforcement Learning Course Project

## License

This project is for educational purposes.
