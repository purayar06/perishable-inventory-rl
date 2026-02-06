# Perishable Inventory Control with Shelf-Life Constraints  
**Reinforcement Learning Project Proposal**  
**Team Members:** [Name 1], [Name 2], [Name 3]

---

## 1. Motivation

Inventory control is a classic sequential decision-making problem. The perishable setting (e.g., fresh food, blood banks) is uniquely challenging because inventory is not defined by a single quantity, but by an **age distribution**. Today's ordering decision affects tomorrow's feasibility through both **service** (meeting demand) and **spoilage** (waste).

This project proposes a reinforcement learning (RL) study to learn optimal ordering policies under stochastic demand with explicit shelf-life constraints, trading off the cost of waste against the cost of lost sales.

---

## 2. MDP Formulation (The Environment)

We formulate the daily ordering problem as a finite-horizon **Markov Decision Process (MDP)**.

- **State Space:**  
  \( S_t = (n_1, n_2, \dots, n_D) \), where \( n_i \) is the number of items with \( i \) days of remaining shelf-life.  
  *Note: The maximum inventory per age bucket will be bounded to keep the state space tractable.*

- **Action Space:**  
  \( A_t \in \{0, 1, \dots, A_{\max}\} \).  
  The discrete number of units ordered at the start of day \( t \).

- **Stochastic Demand:**  
  Daily demand \( U_t \sim \text{Poisson}(\lambda) \).

- **Transition Dynamics (Daily Sequence):**
  1. **Order Arrival:** Add \( A_t \) new items to the freshest bucket (\( n_D \)).
  2. **Demand Realization:** Sample \( U_t \).
  3. **Fulfillment (FEFO):** Meet demand using items with the lowest remaining shelf-life first.
  4. **Waste:** Any items remaining in bucket \( n_1 \) expire and are discarded.
  5. **Aging:** Shift all remaining items so that \( n_i \leftarrow n_{i+1} \).

- **Reward Function:**  
  A profit-based reward capturing operational trade-offs:
  - Revenue from items sold  
  - Ordering cost  
  - Penalty for wasted (expired) items  
  - Penalty for unmet demand (stockouts)

---

## 3. Proposed Methods

We will compare classical reinforcement learning approaches aligned with the course syllabus:

- **Dynamic Programming (Baseline):**  
  Implement **Value Iteration** on a small version of the problem (e.g., \( D = 2 \) or \( D = 3 \)) to compute the optimal policy for comparison.

- **Temporal-Difference Learning:**  
  Implement **Q-Learning** (off-policy) and **SARSA** (on-policy) with ε-greedy exploration (with decaying ε) to learn policies without explicit knowledge of the demand distribution.

- **Monte Carlo Methods:**  
  Implement **On-Policy Monte Carlo Control** to compare convergence behavior and variance against TD methods.

---

## 4. Evaluation Plan

Agents will be evaluated using a fixed-seed protocol over **N = 1000** test episodes.

- **Primary Metric:**  
  Average cumulative reward (profit).

- **Secondary Metrics:**  
  - Waste rate = expired inventory / ordered inventory  
  - Stockout rate = unmet demand / total demand

- **Hypothesis:**  
  Temporal-Difference methods are expected to converge faster than Monte Carlo methods due to lower variance in return estimation under stochastic demand.

- **Planned Visualizations:**  
  - Learning curves (cumulative reward vs. training episode) for each agent  
  - Bar chart comparing final average profit across all agents  
  - Waste-rate vs. stockout-rate scatter plot to visualize the trade-off frontier

- **Sensitivity Analysis:**  
  We will vary shelf-life length \( D \) and demand rate \( \lambda \) to study how problem parameters affect learned policy quality.

---

## 5. Reproducibility & Deliverables

The project will result in a modular and reproducible Python codebase, including:

- A custom environment implementing perishable inventory dynamics  
- Separate agent implementations for Dynamic Programming, Q-Learning, SARSA, and Monte Carlo methods  
- A command-line interface for training agents and generating plots

Final deliverables will include:
- A written report describing the environment, methods, and results  
- A scientific poster summarizing the main findings  
- A reproducible codebase accompanying the report

---
