"""
SAC (Soft Actor-Critic) Trading Agent.

Based on: Lapan, Maxim. Deep Reinforcement Learning Hands-On. Second Edition, MITP, 2019.
Original source: D:/Hive/Data/trading_repos/rl-trader/sac_stock_backend.py

Uses stable-baselines3 SAC implementation with custom trading environment.
SAC is an off-policy algorithm that learns a stochastic policy and
includes entropy regularization for better exploration.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import os
import json
from pathlib import Path

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    SAC = None
    BaseCallback = object

from .actor_critic import (
    Actions,
    Prices,
    State,
    State1D,
    StocksEnv,
    BaseActorCritic,
    load_relative,
    price_files,
)


@dataclass
class SACConfig:
    """Configuration for SAC trading agent."""
    # Policy architecture
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = None  # e.g., [64, 64]

    # SAC hyperparameters
    learning_rate: float = 3e-4
    buffer_size: int = int(1e6)  # Replay buffer size
    learning_starts: int = 100  # Steps before training starts
    batch_size: int = 64
    tau: float = 0.005  # Soft update coefficient
    gamma: float = 0.99  # Discount factor
    ent_coef: str = "auto"  # Auto-tuned entropy coefficient
    target_update_interval: int = 1  # Target network update frequency
    train_freq: int = 1  # Training frequency
    gradient_steps: int = 1  # Gradient steps per update
    target_entropy: str = "auto"

    # Action normalization
    clip_range: float = None  # If None, no clipping
    normalize_action: bool = False  # Normalize actions to [-1, 1]

    # Training
    total_timesteps: int = 100000
    verbose: int = 1

    # Environment
    bars_count: int = 30
    commission: float = 0.0
    reset_on_close: bool = True
    reward_on_close: bool = True
    state_1d: bool = False
    volumes: bool = False

    # Callback
    save_freq: int = 10000
    model_dir: str = "./models/sac"

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [64, 64]


class SACTradingCallback(BaseCallback if SAC else object):
    """Custom callback for monitoring and saving SAC training progress."""

    def __init__(self, config: SACConfig, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.save_freq = config.save_freq
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0:
            model_path = self.model_dir / f"sac_trader_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model to {model_path}")
        return True


class SACTrader(BaseActorCritic):
    """
    SAC-based trading agent.

    Implements Soft Actor-Critic for stock trading with:
    - Entropy regularization for exploration
    - Off-policy learning (sample efficient)
    - Stochastic policy
    - Twin Q-networks for better value estimation
    """

    def __init__(
        self,
        prices: Dict[str, Prices],
        config: Optional[SACConfig] = None,
        model: Optional[SAC] = None
    ):
        if SAC is None:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.config = config or SACConfig()
        self.prices = prices

        # Create environment
        self.env = StocksEnv(
            prices=prices,
            bars_count=self.config.bars_count,
            commission=self.config.commission,
            reset_on_close=self.config.reset_on_close,
            state_1d=self.config.state_1d,
            reward_on_close=self.config.reward_on_close,
            volumes=self.config.volumes
        )

        # Initialize base class
        super().__init__(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self._to_dict()
        )

        # Create or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model()

    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'initial_balance': 10000.0,
            'reward_scale': 1.0,
            'profit_taking_threshold': 0.02,
            'stop_loss_threshold': -0.01,
        }

    def _create_model(self) -> SAC:
        """Create SAC model with configured parameters."""
        policy_kwargs = {}
        if self.config.net_arch:
            policy_kwargs['net_arch'] = self.config.net_arch

        model = SAC(
            policy=self.config.policy_type,
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            target_update_interval=self.config.target_update_interval,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_entropy=self.config.target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=self.config.verbose
        )
        return model

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Get trading action for observation.

        Note: SAC uses stochastic policy by default. Set deterministic=True
        for inference without exploration.

        Args:
            observation: Current market state
            deterministic: If True, use deterministic policy (no exploration)

        Returns:
            Action index: 0=Skip, 1=Buy, 2=Close
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Optional[np.ndarray]]:
        """Stable-baselines3 compatible predict method."""
        return self.model.predict(observation, deterministic=deterministic)

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1
    ) -> None:
        """
        Train the SAC agent.

        Args:
            total_timesteps: Total training steps
            callback: Custom callback
            log_interval: Logging frequency
        """
        timesteps = total_timesteps or self.config.total_timesteps
        self.model.learn(
            total_timesteps=timesteps,
            callback=callback or SACTradingCallback(self.config),
            log_interval=log_interval
        )

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False
    ) -> "SACTrader":
        """
        Learn from environment (stable-baselines3 compatible interface).

        Args:
            total_timesteps: Total training timesteps
            callback: Custom callback
            log_interval: Logging interval
            reset_num_timesteps: Reset timestep counter
            progress_bar: Show progress bar

        Returns:
            Self
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
        return self

    def save(self, path: str) -> None:
        """Save model to path."""
        self.model.save(path)
        # Save config alongside
        config_path = path.replace('.zip', '_config.json')
        self._save_config(config_path)

    def _save_config(self, path: str) -> None:
        """Save configuration to JSON."""
        config_dict = {
            'bars_count': self.config.bars_count,
            'commission': self.config.commission,
            'reset_on_close': self.config.reset_on_close,
            'reward_on_close': self.config.reward_on_close,
            'state_1d': self.config.state_1d,
            'volumes': self.config.volumes,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'tau': self.config.tau,
            'gamma': self.config.gamma,
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str, prices: Optional[Dict[str, Prices]] = None) -> "SACTrader":
        """
        Load model from path.

        Args:
            path: Path to saved model
            prices: Price data for environment

        Returns:
            Loaded SACTrader instance
        """
        model = SAC.load(path)

        # Load config if available
        config_path = path.replace('.zip', '_config.json')
        config = SACConfig()
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                for key, value in saved_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

        # Create instance with or without prices
        if prices is not None:
            return cls(prices=prices, config=config, model=model)
        else:
            # Create minimal instance for inference only
            instance = cls.__new__(cls)
            instance.model = model
            instance.config = config
            instance.prices = {}
            instance.env = None
            return instance

    def run_backtest(
        self,
        prices: Optional[Dict[str, Prices]] = None,
        deterministic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run backtest on price data.

        Args:
            prices: Price data (uses self.prices if None)
            deterministic: Use deterministic policy (recommended for backtesting)

        Returns:
            List of step results with actions and rewards
        """
        test_prices = prices or self.prices
        if not test_prices:
            raise ValueError("No price data provided for backtest")

        env = StocksEnv(
            prices=test_prices,
            bars_count=self.config.bars_count,
            commission=self.config.commission,
            reset_on_close=self.config.reset_on_close,
            state_1d=self.config.state_1d,
            reward_on_close=self.config.reward_on_close,
            volumes=self.config.volumes
        )

        obs = env.reset()
        results = []
        total_reward = 0.0

        while True:
            # SAC typically uses stochastic policy, but for backtesting
            # deterministic gives consistent results
            action = self.get_action(obs, deterministic=deterministic)
            action_name = Actions(action).name

            obs, reward, done, info = env.step(action)
            total_reward += reward

            results.append({
                'action': action_name,
                'action_idx': action,
                'reward': reward,
                'cumulative_reward': total_reward,
                'info': info
            })

            if done:
                break

        return results

    def evaluate(
        self,
        prices: Optional[Dict[str, Prices]] = None,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance over multiple episodes.

        Args:
            prices: Price data
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        test_prices = prices or self.prices
        if not test_prices:
            raise ValueError("No price data for evaluation")

        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            env = StocksEnv(
                prices=test_prices,
                bars_count=self.config.bars_count,
                commission=self.config.commission,
                reset_on_close=self.config.reset_on_close,
                state_1d=self.config.state_1d,
                reward_on_close=self.config.reward_on_close,
                volumes=self.config.volumes
            )

            obs = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = self.get_action(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
        }

    @classmethod
    def from_dir(
        cls,
        data_dir: str,
        config: Optional[SACConfig] = None,
        **env_kwargs
    ) -> "SACTrader":
        """
        Create SACTrader from directory of CSV price files.

        Args:
            data_dir: Directory containing CSV files
            config: SAC configuration
            **env_kwargs: Additional environment kwargs

        Returns:
            SACTrader instance
        """
        prices = {
            file: load_relative(file)
            for file in price_files(data_dir)
        }

        if config:
            for key, value in env_kwargs.items():
                setattr(config, key, value)
        else:
            config = SACConfig(**env_kwargs)

        return cls(prices=prices, config=config)


def create_sac_trader(
    price_data_path: str,
    model_path: Optional[str] = None,
    **config_kwargs
) -> SACTrader:
    """
    Factory function to create SACTrader.

    Args:
        price_data_path: Path to CSV or directory of CSVs
        model_path: Optional path to saved model
        **config_kwargs: SACConfig parameters

    Returns:
        SACTrader instance
    """
    config = SACConfig(**config_kwargs)

    # Load price data
    if os.path.isfile(price_data_path):
        prices = {"data": load_relative(price_data_path)}
    elif os.path.isdir(price_data_path):
        prices = {file: load_relative(file) for file in price_files(price_data_path)}
    else:
        raise ValueError(f"Invalid path: {price_data_path}")

    if model_path and os.path.exists(model_path):
        return SACTrader.load(model_path, prices=prices)
    else:
        return SACTrader(prices=prices, config=config)
