"""
RL Trader Module - Reinforcement Learning Trading Agents.

Absorbed from: D:/Hive/Data/trading_repos/rl-trader
Adapted for stable-baselines3 integration.

Components:
- PPOTrader: Proximal Policy Optimization trading agent
- SACTrader: Soft Actor-Critic trading agent
- BaseActorCritic: Shared Actor-Critic base class
- StocksEnv: Gym-compatible trading environment
- Reward shaping strategies
- Multi-asset portfolio management

Usage:
    from rl_trader import PPOTrader, SACTrader, StocksEnv, load_relative

    # Create trader from CSV
    trader = PPOTrader.from_dir("./data")

    # Train
    trader.learn(total_timesteps=100000)

    # Run backtest
    results = trader.run_backtest()

    # Save/Load
    trader.save("./model.zip")
    loaded = PPOTrader.load("./model.zip")
"""

from .actor_critic import (
    Actions,
    Prices,
    State,
    State1D,
    StocksEnv,
    BaseActorCritic,
    read_csv,
    prices_to_relative,
    load_relative,
    price_files,
)

from .ppo_agent import (
    PPOTrader,
    PPOConfig,
    TradingCallback,
    create_ppo_trader,
)

from .sac_agent import (
    SACTrader,
    SACConfig,
    SACTradingCallback,
    create_sac_trader,
)

__all__ = [
    # Core
    'Actions',
    'Prices',
    'State',
    'State1D',
    'StocksEnv',
    'BaseActorCritic',

    # Utilities
    'read_csv',
    'prices_to_relative',
    'load_relative',
    'price_files',

    # PPO
    'PPOTrader',
    'PPOConfig',
    'TradingCallback',
    'create_ppo_trader',

    # SAC
    'SACTrader',
    'SACConfig',
    'SACTradingCallback',
    'create_sac_trader',
]

__version__ = '1.0.0'
