"""
Smoke tests for RL agents.

These tests verify that each agent can:
1. Be instantiated
2. Select actions
3. Train for multiple episodes without crashing
4. Return finite rewards
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import EnvConfig
from src.envs.perishable_inventory import PerishableInventoryEnv
from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.agents.mc_control import MonteCarloAgent


class TestQLearningSmoke:
    """Smoke tests for Q-Learning agent."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        config = EnvConfig(
            shelf_life=3,
            max_order=5,
            max_inventory=10,
            horizon=20,
            demand_mean=3.0,
        )
        return PerishableInventoryEnv(config=config, seed=42)
    
    @pytest.fixture
    def agent(self, simple_env):
        """Create Q-Learning agent."""
        return QLearningAgent(
            num_actions=simple_env.num_actions,
            gamma=0.99,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            seed=42,
        )
    
    def test_instantiation(self, agent):
        """Test agent can be instantiated."""
        assert agent is not None
        assert agent.name == "Q-Learning"
        assert agent.num_actions == 6
    
    def test_action_selection(self, agent, simple_env):
        """Test agent can select actions."""
        state, _ = simple_env.reset()
        
        for _ in range(10):
            action = agent.select_action(state, training=True)
            assert simple_env.action_space.contains(action)
            state, _, done, _, _ = simple_env.step(action)
            if done:
                break
    
    def test_training_no_crash(self, agent, simple_env):
        """Test agent can train for multiple episodes."""
        rewards = []
        
        for episode in range(50):
            stats = agent.train_episode(simple_env)
            rewards.append(stats["total_reward"])
        
        # Should complete all episodes
        assert len(rewards) == 50
        
        # Rewards should be finite
        assert all(np.isfinite(r) for r in rewards)
    
    def test_q_table_updates(self, agent, simple_env):
        """Test that Q-table gets updated during training."""
        initial_size = len(agent.q_table)
        
        for _ in range(10):
            agent.train_episode(simple_env)
        
        # Q-table should have entries now
        assert len(agent.q_table) > initial_size
    
    def test_epsilon_decay(self, agent, simple_env):
        """Test epsilon decays during training."""
        initial_epsilon = agent.epsilon
        
        for _ in range(10):
            agent.train_episode(simple_env)
        
        # Epsilon should have decayed
        assert agent.epsilon < initial_epsilon


class TestSARSASmoke:
    """Smoke tests for SARSA agent."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        config = EnvConfig(
            shelf_life=3,
            max_order=5,
            max_inventory=10,
            horizon=20,
            demand_mean=3.0,
        )
        return PerishableInventoryEnv(config=config, seed=42)
    
    @pytest.fixture
    def agent(self, simple_env):
        """Create SARSA agent."""
        return SARSAAgent(
            num_actions=simple_env.num_actions,
            gamma=0.99,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            seed=42,
        )
    
    def test_instantiation(self, agent):
        """Test agent can be instantiated."""
        assert agent is not None
        assert agent.name == "SARSA"
    
    def test_action_selection(self, agent, simple_env):
        """Test agent can select actions."""
        state, _ = simple_env.reset()
        
        for _ in range(10):
            action = agent.select_action(state, training=True)
            assert simple_env.action_space.contains(action)
            state, _, done, _, _ = simple_env.step(action)
            if done:
                break
    
    def test_training_no_crash(self, agent, simple_env):
        """Test agent can train for multiple episodes."""
        rewards = []
        
        for episode in range(50):
            stats = agent.train_episode(simple_env)
            rewards.append(stats["total_reward"])
        
        assert len(rewards) == 50
        assert all(np.isfinite(r) for r in rewards)
    
    def test_q_table_updates(self, agent, simple_env):
        """Test that Q-table gets updated during training."""
        initial_size = len(agent.q_table)
        
        for _ in range(10):
            agent.train_episode(simple_env)
        
        assert len(agent.q_table) > initial_size


class TestMonteCarloSmoke:
    """Smoke tests for Monte Carlo agent."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        config = EnvConfig(
            shelf_life=3,
            max_order=5,
            max_inventory=10,
            horizon=20,
            demand_mean=3.0,
        )
        return PerishableInventoryEnv(config=config, seed=42)
    
    @pytest.fixture
    def agent(self, simple_env):
        """Create Monte Carlo agent."""
        return MonteCarloAgent(
            num_actions=simple_env.num_actions,
            gamma=0.99,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            seed=42,
        )
    
    def test_instantiation(self, agent):
        """Test agent can be instantiated."""
        assert agent is not None
        assert agent.name == "Monte Carlo"
    
    def test_action_selection(self, agent, simple_env):
        """Test agent can select actions."""
        state, _ = simple_env.reset()
        
        for _ in range(10):
            action = agent.select_action(state, training=True)
            assert simple_env.action_space.contains(action)
            state, _, done, _, _ = simple_env.step(action)
            if done:
                break
    
    def test_training_no_crash(self, agent, simple_env):
        """Test agent can train for multiple episodes."""
        rewards = []
        
        for episode in range(50):
            stats = agent.train_episode(simple_env)
            rewards.append(stats["total_reward"])
        
        assert len(rewards) == 50
        assert all(np.isfinite(r) for r in rewards)
    
    def test_q_table_updates(self, agent, simple_env):
        """Test that Q-table gets updated during training."""
        initial_size = len(agent.q_table)
        
        for _ in range(10):
            agent.train_episode(simple_env)
        
        assert len(agent.q_table) > initial_size


class TestAgentComparison:
    """Tests comparing agent behaviors."""
    
    @pytest.fixture
    def env(self):
        """Create environment."""
        config = EnvConfig(
            shelf_life=3,
            max_order=5,
            max_inventory=10,
            horizon=30,
            demand_mean=3.0,
        )
        return PerishableInventoryEnv(config=config, seed=42)
    
    def test_all_agents_complete_training(self, env):
        """Test all agents can complete a training run."""
        agents = [
            QLearningAgent(num_actions=env.num_actions, seed=42),
            SARSAAgent(num_actions=env.num_actions, seed=42),
            MonteCarloAgent(num_actions=env.num_actions, seed=42),
        ]
        
        for agent in agents:
            rewards = []
            for _ in range(20):
                stats = agent.train_episode(env)
                rewards.append(stats["total_reward"])
            
            assert len(rewards) == 20, f"{agent.name} did not complete"
            assert all(np.isfinite(r) for r in rewards), f"{agent.name} has infinite rewards"
    
    def test_greedy_vs_exploratory(self, env):
        """Test greedy vs exploratory action selection."""
        agent = QLearningAgent(
            num_actions=env.num_actions,
            epsilon_start=0.5,
            seed=42,
        )
        
        # Train a bit
        for _ in range(10):
            agent.train_episode(env)
        
        state, _ = env.reset()
        
        # Sample many actions
        training_actions = [agent.select_action(state, training=True) for _ in range(100)]
        greedy_actions = [agent.select_action(state, training=False) for _ in range(100)]
        
        # Greedy actions should be less varied (mostly same action)
        greedy_unique = len(set(greedy_actions))
        training_unique = len(set(training_actions))
        
        # With epsilon=0.5, training should explore more
        # This might fail occasionally due to randomness, but generally holds
        assert greedy_unique <= training_unique or greedy_unique <= 2


class TestAgentSaveLoad:
    """Test agent save and load functionality."""
    
    def test_q_learning_save_load(self, tmp_path):
        """Test Q-Learning agent save and load."""
        agent = QLearningAgent(
            num_actions=5,
            gamma=0.95,
            alpha=0.2,
            seed=42,
        )
        
        # Add some Q-values
        agent.q_table[((0, 0), 0)] = 1.5
        agent.q_table[((1, 2), 3)] = -0.5
        agent.epsilon = 0.3
        
        # Save
        filepath = str(tmp_path / "agent.pkl")
        agent.save(filepath)
        
        # Load
        loaded = QLearningAgent.load(filepath, seed=123)
        
        assert loaded.gamma == agent.gamma
        assert loaded.alpha == agent.alpha
        assert loaded.epsilon == agent.epsilon
        assert loaded.q_table[((0, 0), 0)] == 1.5
        assert loaded.q_table[((1, 2), 3)] == -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
