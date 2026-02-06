"""
Configuration module for Perishable Inventory RL.

All hyperparameters are organised into dataclasses so that every
experiment is fully reproducible from a single Config object.
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment parameters."""
    shelf_life: int = 5            # D – number of age buckets
    max_order: int = 10            # A_max – maximum order quantity
    max_inventory: int = 20        # N_max – maximum items per age bucket
    horizon: int = 60              # T – episode length (days)
    demand_mean: float = 6.0       # λ – Poisson demand mean

    # Reward / cost coefficients
    selling_price: float = 10.0    # p – revenue per unit sold
    ordering_cost: float = 4.0     # c – cost per unit ordered
    waste_penalty: float = 6.0     # w – penalty per unit wasted
    stockout_penalty: float = 8.0  # s – penalty per unit of unmet demand


@dataclass
class TrainingConfig:
    """RL training hyperparameters."""
    episodes: int = 3000
    gamma: float = 0.99            # discount factor
    alpha: float = 0.1             # learning rate
    epsilon_start: float = 1.0     # initial exploration
    epsilon_min: float = 0.05      # minimum exploration
    epsilon_decay: float = 0.995   # multiplicative decay per episode
    log_interval: int = 100        # print every N episodes
    eval_episodes: int = 100       # episodes used for evaluation
    eval_seed: int = 123           # seed for deterministic evaluation


@dataclass
class DPConfig:
    """Dynamic-programming parameters."""
    theta: float = 1e-6            # convergence threshold
    max_iter: int = 1000           # maximum Bellman iterations


@dataclass
class Config:
    """Top-level configuration container."""
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    seed: int = 42


def get_config(**overrides) -> Config:
    """
    Factory: create a Config, optionally overriding top-level fields.

    Usage::

        cfg = get_config(seed=0)
        cfg = get_config(env=EnvConfig(shelf_life=3))
    """
    return Config(**overrides)
