"""
Perishable Inventory Environment with Shelf-Life Constraints.

State:  (n_1, n_2, …, n_D)   where n_i = units with *i* days of life left
        Index 0 → oldest (1 day remaining), index D-1 → freshest.
Action: order quantity a ∈ {0, 1, …, A_max}

Daily sequence (within one step):
  1. Receive order  → fresh units added to bucket D-1
  2. Observe demand → Poisson(λ)
  3. Sell (FEFO)    → sell oldest first
  4. Waste          → units in bucket 0 with 0 remaining life are discarded
  5. Age            → each bucket shifts one position toward 0

Reward = p·sold − c·ordered − w·wasted − s·unmet_demand
"""

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from scipy.stats import poisson

from ..config import EnvConfig
from ..utils.seeding import SeededRandom
from ..utils.spaces import DiscreteSpace


# ------------------------------------------------------------------ #
#  Observation-space helper                                           #
# ------------------------------------------------------------------ #

class _ObservationSpace:
    """Describes (N_max+1)^D discrete observation tuples."""

    def __init__(self, dim: int, max_inventory: int):
        self.dim = dim
        self._max = max_inventory

    def contains(self, x: tuple) -> bool:
        return (
            isinstance(x, tuple)
            and len(x) == self.dim
            and all(isinstance(v, (int, np.integer)) and 0 <= v <= self._max for v in x)
        )


# ------------------------------------------------------------------ #
#  Environment                                                        #
# ------------------------------------------------------------------ #

class PerishableInventoryEnv:
    """Tabular MDP for perishable inventory with FEFO selling."""

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.cfg = config or EnvConfig()
        self.D = self.cfg.shelf_life
        self.A_max = self.cfg.max_order
        self.N_max = self.cfg.max_inventory
        self.T = self.cfg.horizon
        self.lam = self.cfg.demand_mean

        # cost / revenue
        self.p = self.cfg.selling_price
        self.c = self.cfg.ordering_cost
        self.w = self.cfg.waste_penalty
        self.s = self.cfg.stockout_penalty

        # derived sizes
        self.num_actions = self.A_max + 1
        self.num_states = (self.N_max + 1) ** self.D

        # spaces
        self.action_space = DiscreteSpace(self.num_actions, seed=seed)
        self.observation_space = _ObservationSpace(self.D, self.N_max)

        # RNG
        self._random = SeededRandom(seed)

        # episode bookkeeping
        self._state: Optional[Tuple[int, ...]] = None
        self._step_count: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------ #
    #  Reset / step  (gymnasium-style returns)                           #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, ...], dict]:
        """Reset to the empty-shelf initial state.  Returns (state, info)."""
        if seed is not None:
            self._random.seed(seed)
            self.action_space._rng = np.random.RandomState(seed)
        self._state = tuple([0] * self.D)
        self._step_count = 0
        self._done = False
        return self._state, {}

    def step(self, action: int):
        """
        Execute one day.

        Returns (next_state, reward, terminated, truncated, info).
        """
        assert not self._done, "Episode has ended; call reset()."
        assert 0 <= action <= self.A_max, f"Invalid action {action}"

        state = list(self._state)
        order_qty = action

        # 1. Receive order → freshest bucket
        state[self.D - 1] = min(state[self.D - 1] + order_qty, self.N_max)

        # 2. Demand
        demand = self._random.poisson(self.lam)

        # 3. Sell FEFO (oldest first)
        sold = 0
        remaining_demand = demand
        for i in range(self.D):
            sell = min(state[i], remaining_demand)
            state[i] -= sell
            sold += sell
            remaining_demand -= sell
            if remaining_demand == 0:
                break
        stockout = remaining_demand

        # 4. Waste – bucket 0 leftovers expire
        waste = state[0]
        state[0] = 0

        # 5. Age – shift toward index 0
        new_state = [0] * self.D
        for i in range(self.D - 1):
            new_state[i] = state[i + 1]
        new_state[self.D - 1] = 0
        new_state = tuple(min(n, self.N_max) for n in new_state)

        reward = self.compute_reward(order_qty, sold, waste, stockout)

        self._state = new_state
        self._step_count += 1
        terminated = self._step_count >= self.T
        self._done = terminated

        info = {
            "sold": sold,
            "waste": waste,
            "stockout": stockout,
            "demand": demand,
            "order": order_qty,
        }
        return self._state, reward, terminated, False, info

    # ------------------------------------------------------------------ #
    #  Reward helper                                                      #
    # ------------------------------------------------------------------ #

    def compute_reward(self, action: int, sold: int, waste: int, stockout: int) -> float:
        return (self.p * sold
                - self.c * action
                - self.w * waste
                - self.s * stockout)

    # ------------------------------------------------------------------ #
    #  Model-based helpers (DP)                                          #
    # ------------------------------------------------------------------ #

    def simulate_step(
        self, state: Tuple[int, ...], action: int, demand: int
    ) -> Tuple[Tuple[int, ...], float, int, int, int]:
        """
        Pure function version for DP.  Does NOT modify internal state.

        Returns (next_state, reward, sold, waste, stockout).
        """
        s = list(state)

        # 1. Receive
        s[self.D - 1] = min(s[self.D - 1] + action, self.N_max)

        # 2-3. Sell FEFO
        sold = 0
        remaining = demand
        for i in range(self.D):
            sell = min(s[i], remaining)
            s[i] -= sell
            sold += sell
            remaining -= sell
            if remaining == 0:
                break
        stockout = remaining

        # 4. Waste
        waste = s[0]
        s[0] = 0

        # 5. Age
        new_s = [0] * self.D
        for i in range(self.D - 1):
            new_s[i] = s[i + 1]
        new_s[self.D - 1] = 0
        new_s = tuple(min(n, self.N_max) for n in new_s)

        reward = self.compute_reward(action, sold, waste, stockout)
        return new_s, reward, sold, waste, stockout

    def get_demand_distribution(
        self, max_demand: Optional[int] = None, truncation: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Truncated Poisson PMF.

        Returns (demands, probs) as numpy arrays.
        """
        trunc = truncation or max_demand or (self.N_max * self.D + self.A_max)
        demands = np.arange(trunc + 1)
        probs = poisson.pmf(demands, self.lam)
        probs[-1] += 1.0 - probs.sum()
        return demands, probs

    # ------------------------------------------------------------------ #
    #  State ↔ index bijection                                           #
    # ------------------------------------------------------------------ #

    def get_state_index(self, state: Tuple[int, ...]) -> int:
        idx = 0
        base = 1
        for i in range(self.D):
            idx += state[i] * base
            base *= self.N_max + 1
        return idx

    def get_state_from_index(self, index: int) -> Tuple[int, ...]:
        state = []
        for _ in range(self.D):
            state.append(index % (self.N_max + 1))
            index //= (self.N_max + 1)
        return tuple(state)

    def get_all_states(self) -> List[Tuple[int, ...]]:
        return [self.get_state_from_index(i) for i in range(self.num_states)]
