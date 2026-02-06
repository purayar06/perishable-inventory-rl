# Perishable Inventory Control with Shelf-Life Constraints (Project Proposal)

**Team:** _Name 1_, _Name 2_, _Name 3_ (placeholders)

## B. Motivation
Inventory control is a classic sequential decision problem. The perishable setting is uniquely challenging because inventory is not only a quantity but an **age distribution**: today’s ordering decision changes tomorrow’s feasibility through both **service** (meeting demand) and **spoilage** (waste). This project proposes an RL study of ordering policies under stochastic demand with explicit shelf-life dynamics.

## C. RL Formulation / Environment (MDP)
We formulate daily ordering as an MDP.

- **State space:**
  \(s_t = (n_1,\dots,n_D)\), where \(n_i\) is the number of items with \(i\) days of remaining shelf-life (age-structured inventory vector).

- **Action space:**
  \(a_t\in\{0,1,\dots,A_{\max}\}\), the order quantity placed at the start of the day.

- **Demand model (stochastic):**
  Daily demand \(u_t\) is random (modeled as a Poisson process with rate \(\lambda\); other rates can be explored in sensitivity analysis).

- **Transition steps (explicit sequence):**
  1) Receive order: add \(a_t\) items to the freshest bucket.
  2) Realize demand \(u_t\).
  3) Sell using **FEFO** (oldest items sold first).
  4) Waste: any remaining items that expire today are discarded.
  5) Age inventory: shift all remaining items one day closer to expiry.

- **Reward definition:**
  Profit-like reward that trades off revenue, ordering cost, waste, and stockouts:
  $$
  r_t = p\,\text{sold}_t - c\,a_t - w\,\text{waste}_t - s\,\text{stockout}_t.
  $$
  Here \(p\) is selling price, \(c\) is ordering cost, \(w\) penalizes waste, and \(s\) penalizes unmet demand.

- **Episode setup:**
  Finite-horizon episodes of length \(T\) (days) with discount factor \(\gamma\in(0,1)\).

## D. Implemented Agents / Methods
We will compare classical control algorithms on this MDP:

- **DP Value Iteration baseline:**
  Solve a bounded version of the MDP via value iteration to produce an “optimal” baseline on a tractable state space (with truncated demand support).

- **Tabular Q-learning and SARSA:**
  Learn action-values via TD control with \(\epsilon\)-greedy exploration. Key hyperparameters are learning rate \(\alpha\), discount \(\gamma\), and exploration schedule \(\epsilon\).

- **Monte Carlo control:**
  Consider first-visit MC control as an additional baseline if time allows.

- **Function approximation:**
  Planned extension only (e.g., linear features) if tabular methods do not scale to larger \(D\) or inventory bounds.

## E. Evaluation Plan
- **Metrics:** average episode return (profit proxy), waste rate, stockout rate, and fill rate.
- **Protocol:** train each agent for a fixed number of episodes, then evaluate the learned greedy policy over held-out episodes.
- **Reproducibility:** run multiple random seeds and report mean \(\pm\) variability.
- **Sensitivity analysis:** vary demand rate \(\lambda\), shelf-life \(D\), and exploration/learning hyperparameters (sweep over \(\alpha\), \(\epsilon\)-decay).

## F. Expected Results & Poster Figures
We expect a clear tradeoff frontier between waste and stockouts:

- Learning curves showing convergence behavior across algorithms.
- Bar chart comparing mean return (with error bars across seeds).
- Waste-vs-stockout (or waste-vs-fill) scatter to visualize operational tradeoffs.

## G. Repo Reproducibility (brief)
Planned usage is via simple CLI scripts:

- Train: `python -m src.experiments.train --agent <qlearning|sarsa|dp|mc> --episodes <N>`
- Evaluate: `python -m src.experiments.evaluate --agent-path <path> --agent-type <type> --episodes <N>`
- Plot: `python -m src.plotting.make_plots --results-dir <runs_dir> --output-dir <figures_dir>`

## H. References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
- Course notes/readings on MDPs, dynamic programming, and inventory control.
