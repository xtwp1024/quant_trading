"""
Market-Making RL Module
Adapted from Market-Making-RL (D:/Hive/Data/trading_repos/Market-Making-RL/)

Key features:
- Avellaneda-Stoikov Heston model for reservation price
- Inventory risk penalty
- Continuous bid/ask spread control
- Variable-length trajectory support with masked tensors
- Stable-baselines3 PPO compatible Gymnasium environment
"""

from quant_trading.market_making.order_book import OrderBook
from quant_trading.market_making.market_env import MarketEnv, BaseMarket
from quant_trading.market_making.rewards import AvellanedaStoikovReward
from quant_trading.market_making.policies import (
    BasePolicy,
    GaussianPolicy,
    CategoricalPolicy,
    MaskedSequential,
    build_mlp,
)
from quant_trading.market_making.market_maker import (
    UniformMarketMaker,
    MaskedMarketMaker,
    MarketMaker,
)
from quant_trading.market_making.passivbot import PassivBotStrategy
from quant_trading.market_making.evolutionary_optimizer import EvolutionaryOptimizer
from quant_trading.market_making.exchange_adapters import ExchangeAdapter
from quant_trading.market_making.avellaneda_stoikov import (
    avellaneda_stoikov_sim,
    AvellanedaStoikov,
)
from quant_trading.market_making.lob_simulator import (
    LOBSimulator,
    LOBQuote,
    LOBTrade,
)
from quant_trading.market_making.market_maker_env import MarketMakerEnv
from quant_trading.market_making.optiver_mm import OptiverMarketMaker
from quant_trading.market_making.delta_hedge import (
    DeltaHedgeEngine,
    PerUnderlyingDeltaHedge,
    OptionPosition,
    FuturePosition,
    HedgeResult,
    HedgedPortfolio,
    calculate_time_to_expiry,
    calculate_current_time_to_expiry,
)
from quant_trading.market_making.as_optimizer import ASOptimizer
from quant_trading.market_making.inventory_model import InventoryModel
from quant_trading.market_making.simulator import MarketSimulator
from quant_trading.market_making.polymarket_mm import (
    PolymarketMarketMaker,
    Inventory,
    Quote,
)
from quant_trading.market_making.market_making_rl import (
    LOBSimulator,
    MarketMakingEnv,
    AvellanedaStoikovModel,
    OptimalSpreadCalculator,
    LOBQuote,
    LOBTrade,
    MarketState,
)
from quant_trading.market_making.market_making_rl_v2 import (
    ReplayBuffer,
    DQNAgent,
    DQNMarketMaker,
    A2CMarketMaker,
    MarketMakingTrainer,
    TrainingConfig,
    MarketMakingPolicy,
    SimpleMarketEnv,
    build_mlp_numpy,
    mlp_forward_numpy,
    serialize_params,
    deserialize_params,
)

__all__ = [
    # Order book
    "OrderBook",
    # Environment
    "MarketEnv",
    "BaseMarket",
    # Rewards
    "AvellanedaStoikovReward",
    # Policies
    "BasePolicy",
    "GaussianPolicy",
    "CategoricalPolicy",
    "MaskedSequential",
    "build_mlp",
    # Market makers
    "UniformMarketMaker",
    "MaskedMarketMaker",
    "MarketMaker",
    # Passivbot
    "PassivBotStrategy",
    "EvolutionaryOptimizer",
    "ExchangeAdapter",
    # Avellaneda-Stoikov
    "avellaneda_stoikov_sim",
    "AvellanedaStoikov",
    # LOB Simulator
    "LOBSimulator",
    "LOBQuote",
    "LOBTrade",
    # Market Maker Gymnasium Env
    "MarketMakerEnv",
    # Optiver
    "OptiverMarketMaker",
    # Delta Hedging
    "DeltaHedgeEngine",
    "PerUnderlyingDeltaHedge",
    "OptionPosition",
    "FuturePosition",
    "HedgeResult",
    "HedgedPortfolio",
    "calculate_time_to_expiry",
    "calculate_current_time_to_expiry",
    # Avellaneda-Stoikov Optimizer & Simulator
    "ASOptimizer",
    "InventoryModel",
    "MarketSimulator",
    # Polymarket Market Maker
    "PolymarketMarketMaker",
    "Inventory",
    "Quote",
    # CS234 Market Making RL (Stanford)
    "LOBSimulator",
    "MarketMakingEnv",
    "AvellanedaStoikovModel",
    "OptimalSpreadCalculator",
    "LOBQuote",
    "LOBTrade",
    "MarketState",
    # CS234 Market Making RL v2 (DQN + A2C)
    "ReplayBuffer",
    "DQNAgent",
    "DQNMarketMaker",
    "A2CMarketMaker",
    "MarketMakingTrainer",
    "TrainingConfig",
    "MarketMakingPolicy",
    "SimpleMarketEnv",
    "build_mlp_numpy",
    "mlp_forward_numpy",
    "serialize_params",
    "deserialize_params",
]
