"""
Unit tests for the Perishable Inventory Environment.

Tests cover:
- FEFO selling logic
- Aging and waste logic
- Reward calculation
- Episode termination
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import EnvConfig
from src.envs.perishable_inventory import PerishableInventoryEnv


class TestPerishableInventoryEnv:
    """Test suite for PerishableInventoryEnv."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return EnvConfig(
            shelf_life=3,  # D=3 days
            max_order=5,   # Amax=5
            max_inventory=10,
            horizon=10,
            demand_mean=3.0,
            selling_price=10.0,
            ordering_cost=4.0,
            waste_penalty=6.0,
            stockout_penalty=8.0,
        )
    
    @pytest.fixture
    def env(self, simple_config):
        """Create environment with simple config."""
        return PerishableInventoryEnv(config=simple_config, seed=42)
    
    def test_reset(self, env):
        """Test environment reset."""
        state, info = env.reset()
        
        # Initial state should be all zeros (empty inventory)
        assert state == (0, 0, 0), f"Expected (0,0,0), got {state}"
        assert env._step_count == 0
        assert env._done is False
    
    def test_action_space(self, env):
        """Test action space bounds."""
        assert env.action_space.n == 6  # {0, 1, 2, 3, 4, 5}
        assert env.action_space.contains(0)
        assert env.action_space.contains(5)
        assert not env.action_space.contains(6)
        assert not env.action_space.contains(-1)
    
    def test_state_space(self, env):
        """Test state space bounds."""
        assert env.observation_space.dim == 3  # D=3
        assert env.observation_space.contains((0, 0, 0))
        assert env.observation_space.contains((10, 10, 10))
        assert not env.observation_space.contains((11, 0, 0))
    
    def test_ordering_adds_to_freshest_bucket(self, simple_config):
        """Test that orders add to the freshest bucket (index D-1)."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        env.reset()
        
        # Manually test the internal step logic
        state = (0, 0, 0)  # Empty inventory
        action = 3
        demand = 0  # No demand, so we can see what's ordered
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        # After ordering 3 items with D=3:
        # - Items added to bucket 2 (freshest)
        # - No sales (demand=0)
        # - Bucket 0 was empty, so waste=0
        # - After aging: bucket[0] <- bucket[1]=0, bucket[1] <- bucket[2]=3, bucket[2] <- 0
        # So next_state should be (0, 3, 0)
        assert next_state == (0, 3, 0), f"Expected (0, 3, 0), got {next_state}"
        assert sold == 0
        assert waste == 0
        assert stockout == 0
    
    def test_fefo_selling_oldest_first(self, simple_config):
        """Test that items are sold oldest-first (FEFO)."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # State: (2, 3, 5) means 2 items with 1 day left, 3 with 2 days, 5 with 3 days
        state = (2, 3, 5)
        action = 0  # No new order
        demand = 4  # Demand for 4 items
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        # FEFO: sell 2 from bucket[0], then 2 from bucket[1]
        # After sales: (0, 1, 5)
        # Waste: bucket[0] after sales = 0
        # After aging: (1, 5, 0)
        assert sold == 4
        assert waste == 0
        assert stockout == 0
        assert next_state == (1, 5, 0), f"Expected (1, 5, 0), got {next_state}"
    
    def test_waste_calculation(self, simple_config):
        """Test that items expiring become waste."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # State: (5, 0, 0) - 5 items with only 1 day left
        state = (5, 0, 0)
        action = 0  # No order
        demand = 2  # Only sell 2
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        # Sell 2 from bucket[0], leaves 3
        # Waste = 3 (all remaining in bucket[0])
        # After aging: all buckets become 0
        assert sold == 2
        assert waste == 3
        assert stockout == 0
        assert next_state == (0, 0, 0), f"Expected (0, 0, 0), got {next_state}"
    
    def test_stockout_calculation(self, simple_config):
        """Test stockout when demand exceeds inventory."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # State: (1, 1, 1) - total 3 items
        state = (1, 1, 1)
        action = 0
        demand = 5  # Demand exceeds inventory
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        # Sell all 3 items, stockout = 5 - 3 = 2
        assert sold == 3
        assert waste == 0  # Everything sold
        assert stockout == 2
    
    def test_reward_calculation(self, simple_config):
        """Test reward computation with known values."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # Create scenario:
        # action=2 (order 2), sold=3, waste=1, stockout=0
        # R = 10*3 - 4*2 - 6*1 - 8*0 = 30 - 8 - 6 = 16
        reward = env.compute_reward(action=2, sold=3, waste=1, stockout=0)
        assert reward == 16.0, f"Expected 16.0, got {reward}"
        
        # Scenario with stockout:
        # action=0, sold=2, waste=0, stockout=3
        # R = 10*2 - 4*0 - 6*0 - 8*3 = 20 - 24 = -4
        reward = env.compute_reward(action=0, sold=2, waste=0, stockout=3)
        assert reward == -4.0, f"Expected -4.0, got {reward}"
    
    def test_aging_mechanics(self, simple_config):
        """Test that inventory ages correctly."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # State: (0, 0, 5) - 5 items with 3 days left (freshest)
        state = (0, 0, 5)
        action = 0
        demand = 0  # No sales
        
        next_state, _, _, _, _ = env.simulate_step(state, action, demand)
        # After aging: (0, 5, 0) - items now have 2 days left
        assert next_state == (0, 5, 0), f"Expected (0, 5, 0), got {next_state}"
        
        # Continue aging
        next_state, _, _, waste, _ = env.simulate_step(next_state, 0, 0)
        # After aging: (5, 0, 0) - items now have 1 day left
        assert next_state == (5, 0, 0), f"Expected (5, 0, 0), got {next_state}"
        assert waste == 0  # No waste yet
        
        # One more day - items expire
        next_state, _, _, waste, _ = env.simulate_step(next_state, 0, 0)
        assert next_state == (0, 0, 0), f"Expected (0, 0, 0), got {next_state}"
        assert waste == 5  # All items wasted
    
    def test_episode_termination(self, simple_config):
        """Test that episode terminates at horizon."""
        simple_config.horizon = 5
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        env.reset()
        
        for i in range(5):
            state, reward, terminated, truncated, info = env.step(0)
            if i < 4:
                assert not terminated, f"Episode terminated early at step {i+1}"
            else:
                assert terminated, "Episode should terminate at horizon"
    
    def test_step_order_complete(self, simple_config):
        """Test the complete step order is correct."""
        env = PerishableInventoryEnv(config=simple_config, seed=42)
        
        # Scenario: Start with (2, 3, 0), order 4, demand=5
        state = (2, 3, 0)
        action = 4
        demand = 5
        
        # Step 1: Add 4 to bucket[2] -> (2, 3, 4)
        # Step 2: Demand = 5
        # Step 3: Sell FEFO: 2 from [0], 3 from [1], 0 from [2] = 5 sold
        #         After sales: (0, 0, 4)
        # Step 4: Waste = bucket[0] = 0
        # Step 5: Age: (0, 4, 0)
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        assert sold == 5
        assert waste == 0
        assert stockout == 0
        assert next_state == (0, 4, 0), f"Expected (0, 4, 0), got {next_state}"
    
    def test_state_index_conversion(self, env):
        """Test state to index and back conversion."""
        test_states = [
            (0, 0, 0),
            (1, 2, 3),
            (10, 10, 10),
            (5, 0, 5),
        ]
        
        for state in test_states:
            index = env.get_state_index(state)
            recovered = env.get_state_from_index(index)
            assert recovered == state, f"State {state} -> index {index} -> {recovered}"
    
    def test_demand_distribution(self, env):
        """Test demand probability distribution."""
        demands, probs = env.get_demand_distribution(truncation=20)
        
        # Probabilities should sum to 1
        assert np.isclose(probs.sum(), 1.0), f"Probabilities sum to {probs.sum()}"
        
        # Should have correct number of values
        assert len(demands) == 21  # 0 to 20 inclusive
        assert demands[0] == 0
        assert demands[-1] == 20
    
    def test_reproducibility(self, simple_config):
        """Test that same seed gives same results."""
        env1 = PerishableInventoryEnv(config=simple_config, seed=123)
        env2 = PerishableInventoryEnv(config=simple_config, seed=123)
        
        env1.reset(seed=123)
        env2.reset(seed=123)
        
        for _ in range(10):
            action = 3
            s1, r1, d1, _, _ = env1.step(action)
            s2, r2, d2, _, _ = env2.step(action)
            
            assert s1 == s2, f"States differ: {s1} vs {s2}"
            assert r1 == r2, f"Rewards differ: {r1} vs {r2}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_demand(self):
        """Test behavior with zero demand."""
        config = EnvConfig(shelf_life=3, max_order=5, demand_mean=0.001, horizon=5)
        env = PerishableInventoryEnv(config=config, seed=42)
        
        state = (3, 2, 1)
        action = 0
        demand = 0
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        assert sold == 0
        assert waste == 3  # All items in bucket[0] become waste
        assert stockout == 0
    
    def test_max_inventory_cap(self):
        """Test that inventory is capped at max_inventory."""
        config = EnvConfig(shelf_life=2, max_order=10, max_inventory=5, horizon=5)
        env = PerishableInventoryEnv(config=config, seed=42)
        
        # Order more than max_inventory
        state = (0, 0)
        action = 10  # Order 10, but max is 5
        demand = 0
        
        next_state, _, _, _, _ = env.simulate_step(state, action, demand)
        
        # After aging: bucket[0] should be capped at 5
        assert next_state[0] <= config.max_inventory
    
    def test_high_demand(self):
        """Test behavior with demand higher than total inventory."""
        config = EnvConfig(shelf_life=3, max_order=5, horizon=5)
        env = PerishableInventoryEnv(config=config, seed=42)
        
        state = (1, 1, 1)  # Total = 3
        action = 2  # Order 2 more -> total = 5
        demand = 10  # Much higher than inventory
        
        next_state, reward, sold, waste, stockout = env.simulate_step(state, action, demand)
        
        assert sold == 5  # Sell everything
        assert waste == 0  # Nothing left to waste
        assert stockout == 5  # 10 - 5 = 5


class TestIntegration:
    """Integration tests running full episodes."""
    
    def test_full_episode(self):
        """Run a complete episode and verify basic properties."""
        config = EnvConfig(shelf_life=3, max_order=5, horizon=30)
        env = PerishableInventoryEnv(config=config, seed=42)
        
        state, _ = env.reset()
        total_reward = 0
        total_sold = 0
        total_waste = 0
        
        for step in range(30):
            action = env.action_space.sample(env._random.rng)
            next_state, reward, done, _, info = env.step(action)
            
            total_reward += reward
            total_sold += info["sold"]
            total_waste += info["waste"]
            
            state = next_state
            
            if done:
                break
        
        assert done, "Episode should be done after horizon"
        assert isinstance(total_reward, (int, float))
        assert total_sold >= 0
        assert total_waste >= 0
    
    def test_multiple_episodes(self):
        """Run multiple episodes and verify consistency."""
        config = EnvConfig(shelf_life=3, max_order=5, horizon=20)
        env = PerishableInventoryEnv(config=config, seed=42)
        
        rewards = []
        
        for episode in range(5):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(20):
                action = 3  # Fixed action
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        # All episodes should complete
        assert len(rewards) == 5
        
        # With fixed action and same seed progression, results should vary
        # (different demand realizations)
        assert not all(r == rewards[0] for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
