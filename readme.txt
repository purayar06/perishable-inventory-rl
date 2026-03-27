========================================================================
PERISHABLE INVENTORY CONTROL - REINFORCEMENT LEARNING DASHBOARD
========================================================================

1. WHAT IS IT?
--------------
This project tackles the difficult problem of managing perishable goods 
(goods with a finite shelf-life, like groceries or blood products) using 
Reinforcement Learning. 

The Streamlit App ("streamlit demo/app.py") is an interactive visual 
dashboard that allows you to observe our trained RL agents in real-time. 
Instead of staring at terminal logs, you can click through days to see 
exactly what the agent orders, what the stochastic demand is, and how 
inventory ages, sells, or spoils (waste) over time.

2. HOW TO RUN THE APP
---------------------
1. Ensure your Python environment is active and all dependencies are installed.
   (If you haven't yet, you can run: pip install -r requirements.txt)
   
2. Open your terminal and navigate to the project root directory.

3. Run the following command exactly as written:
   streamlit run "streamlit demo/app.py"

4. A new tab should automatically open in your default web browser 
   (typically at http://localhost:8501).

3. WHAT HAPPENS IN THE BACKEND
------------------------------
When you run the app, the backend initializes our custom Gymnasium-style 
Markov Decision Process (`PerishableInventoryEnv`). 
- It loads the pre-trained Q-tables and value functions from the `outputs/runs/` 
  directory for the agent you select (e.g., Value Iteration, SARSA, Q-Learning).
- At every "Step", the backend takes the current inventory age profile, queries 
  the RL agent for the optimal order quantity, applies a randomly generated 
  Poisson demand, and updates the inventory using FEFO (First-Expire-First-Out) 
  logic. 
- It computes the rewards/penalties and sends the new state back to the UI.

4. HOW TO USE THE STREAMLIT DASHBOARD
-------------------------------------
Once the browser tab opens, here is how you use the interface:

* SIDEBAR CONTROLS (Left side):
  - Agent Selection: Choose which RL algorithm's loaded brain to evaluate 
    (e.g., DP vs SARSA).
  - Environment Parameters: Slide the sliders to dynamically change the 
    Demand Mean, Waste Penalties, or Stockout Penalties to stress-test 
    how the agent adapts mathematically.
  - Action Buttons: 
    > "Step 1 Day": Advances the simulation by exactly one day.
    > "Step 10/30 Days": Fast-forwards the simulation to test long-term stability.
    > "Reset Environment": Wipes the inventory completely clean and starts 
      the episode from day zero.

* MAIN PANEL (Center):
  - Metrics: Quick glance at Cumulative Reward, total items Wasted, and total Stockouts.
  - Inventory Age Profile: A bar chart showing exactly how many units have 1 day, 
    2 days, or 3 days of shelf life remaining.
  - Event Log: A running historical text log showing exactly what happened today 
    (Amount Ordered -> Demand Received -> Units Sold -> Units Wasted).
  - Reward Trajectory: A line graph tracking to see if the agent is making a profit 
    over the 60-day horizon.