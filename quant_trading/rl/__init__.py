"""
quant_trading.rl - Reinforcement Learning Trading Module.

Adapted from FinRL-DAPO-SR project for quant_trading package.

Components:
- StockTradingEnv: Gymnasium-compatible stock trading environment
- StockTradingEnv (LLM): Extended environment with LLM risk/sentiment signals
- DAPO Algorithm: Dual-clipping Asymmetric Policy Optimization
- Training & Backtest scripts

Usage:
    from quant_trading.rl import StockTradingEnv, DAPOBuffer, dapo

    # Create environment
    env = StockTradingEnv(df=price_data, stock_dim=5, ...)

    # Train with DAPO
    trained_model = dapo(lambda: env, ...)

Note:
    Requires: gymnasium, torch, numpy, pandas, scipy, stable-baselines3
    Optional: mpi4py (for distributed training)
"""

from .env_stocktrading import StockTradingEnv
from .env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnvLLM
from .dapo_algorithm import (
    dapo,
    DAPOBuffer,
    MLPActorCritic,
    combined_shape,
    discount_cumsum,
    mlp,
    count_vars,
)

# MultiStockRLTrading imports (adapted from D:/Hive/Data/trading_repos/MultiStockRLTrading/)
from .multi_stock_env import MultiStockTradingEnv
from .cross_attention_policy import (
    CrossAttentionActorCriticPolicy,
    MultiAssetCrossAttentionNetwork,
    AttentionPooling,
)
from .multi_stock_trainer import (
    train_ppo,
    make_env,
    load_market_data,
    add_features,
    run_inference,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_INITIAL_AMOUNT,
    DEFAULT_TRADE_COST,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TIMESTEPS,
    BASE_INDICATORS,
    POLICY_REGISTRY,
)

__all__ = [
    # Existing — Environments
    "StockTradingEnv",
    "StockTradingEnvLLM",
    # Existing — DAPO core
    "dapo",
    "DAPOBuffer",
    "MLPActorCritic",
    # Existing — Utilities
    "combined_shape",
    "discount_cumsum",
    "mlp",
    "count_vars",
    # MultiStockRLTrading — Environment
    "MultiStockTradingEnv",
    # MultiStockRLTrading — Policy
    "CrossAttentionActorCriticPolicy",
    "MultiAssetCrossAttentionNetwork",
    "AttentionPooling",
    # MultiStockRLTrading — Training utilities
    "train_ppo",
    "make_env",
    "load_market_data",
    "add_features",
    "run_inference",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_TRAIN_SPLIT",
    "DEFAULT_INITIAL_AMOUNT",
    "DEFAULT_TRADE_COST",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TIMESTEPS",
    "BASE_INDICATORS",
    "POLICY_REGISTRY",
    # Spread trading (Trading-Gym)
    "SpreadTrading",
    "DataGenerator",
    "CSVStreamer",
    "RandomWalk",
    "AR1",
    "RandomGenerator",
    "calc_spread",
]

# Spread trading (absorbed from Trading-Gym)
from .spread_trading_env import SpreadTrading
from .spread_data_generator import (
    DataGenerator,
    CSVStreamer,
    RandomWalk,
    AR1,
    RandomGenerator,
    calc_spread,
)

# DRQN imports (adapted from D:/Hive/Data/trading_repos/DRQN_Stock_Trading/)
# Key feature: action augmentation - all 3 actions evaluated per step
from .drqn_env import DRQNTradingEnv, TradingEnvironment
from .drqn_agent import DRQNAgent
from .replay_memory import ReplayMemory

__all__ = __all__ + [
    # DRQN — Environment
    "DRQNTradingEnv",
    "TradingEnvironment",
    # DRQN — Agent
    "DRQNAgent",
    # DRQN — Replay buffer
    "ReplayMemory",
]

__version__ = "1.0.0"

# Deep Q-Trading Agent (adapted from Jeong et al., 2019)
# Key feature: Share-sizing DQN - outputs both action AND share quantity
from .deep_q_networks import (
    # Architectures
    NumQModel,
    NumDRegModel,
    StonksNet,
    # Method constants
    NUMQ,
    NUMDREG_AD,
    NUMDREG_ID,
    # Mode constants
    ACT_MODE,
    NUM_MODE,
    FULL_MODE,
    # Action constants
    BUY,
    HOLD,
    SELL,
)
from .deep_q_agent import (
    # Agent
    DQN,
    NumQAgent,
    NumDRegAgent,
    # Environment
    FinanceEnvironment,
    ReplayMemory,
    # Training & evaluation
    train,
    evaluate,
    select_action,
    optimize_model,
    # Config
    DEFAULT_CONFIG,
)
from .stock_grouping import (
    # Autoencoder
    train_autoencoder,
    measure_correlation,
    measure_autoencoder_mse,
    create_groups,
    # Market regime detection
    ConfusedMarketDetector,
    detect_market_regime,
)

# Deep Q-Trading — Pure NumPy unified implementation (Jeong et al., 2019)
# Source: D:/Hive/Data/trading_repos/deep-q-trading-agent/
# Three architectures: NumQ (joint), NumDReg-AD (action-dependent), NumDReg-ID (action-independent)
# Key innovation: both action (BUY/HOLD/SELL) AND share quantity are predicted
from .deep_q_trading import (
    # Constants
    NUMQ, NUMDREG_AD, NUMDREG_ID,
    ACT_MODE, NUM_MODE, FULL_MODE,
    ACTION_BUY, ACTION_HOLD, ACTION_SELL,
    ACTION_SPACE, NUM_ACTIONS,
    DEFAULT_CONFIG,
    # Activation functions
    relu, sigmoid, tanh, softmax,
    relu_grad, sigmoid_grad, tanh_grad,
    smooth_l1_loss, mse_loss,
    # Layer
    DenseLayer,
    # Models
    NumQModel, NumDRegModel,
    # Stock grouping
    StockAutoencoder,
    # Environment & buffer
    FinanceEnvironment, ReplayBuffer,
    # Agents
    BaseAgent,
    NumQAgent,
    NumDRegADAgent,
    NumDRegIDAgent,
    # Transfer learning pipeline
    TransferLearningTrader,
    # Utilities
    compute_confidence, batch_compute_td_errors,
    GYMNASIUM_AVAILABLE,
)

__all__ = __all__ + [
    # Deep Q-Trading — Constants
    "NUMQ", "NUMDREG_AD", "NUMDREG_ID",
    "ACT_MODE", "NUM_MODE", "FULL_MODE",
    "ACTION_BUY", "ACTION_HOLD", "ACTION_SELL",
    "ACTION_SPACE", "NUM_ACTIONS",
    "DEFAULT_CONFIG",
    # Deep Q-Trading — Activation functions
    "relu", "sigmoid", "tanh", "softmax",
    "relu_grad", "sigmoid_grad", "tanh_grad",
    "smooth_l1_loss", "mse_loss",
    # Deep Q-Trading — Layer
    "DenseLayer",
    # Deep Q-Trading — Models
    "NumQModel", "NumDRegModel",
    # Deep Q-Trading — Stock grouping
    "StockAutoencoder",
    # Deep Q-Trading — Environment & buffer
    "FinanceEnvironment", "ReplayBuffer",
    # Deep Q-Trading — Agents
    "BaseAgent",
    "NumQAgent",
    "NumDRegADAgent",
    "NumDRegIDAgent",
    # Deep Q-Trading — Transfer learning
    "TransferLearningTrader",
    # Deep Q-Trading — Utilities
    "compute_confidence", "batch_compute_td_errors",
    "GYMNASIUM_AVAILABLE",
    # Aliases for compatibility with older separate-module imports
    "NumDRegAgent",   # -> maps to NumDRegADAgent
]

# DDPG imports (adapted from PyTorch-DDPG-Stock-Trading)
# Key feature: 3-timestep state window + Ornstein-Uhlenbeck exploration noise
from .ddpg_agent import DDPGAgent, train_ddpg
from .ddpg_networks import Actor, Critic, ActorNet, CriticNet
from .ou_noise import OUNoise
from .ddpg_market import DDPGMarketEnv, create_sample_market_data

__all__ = __all__ + [
    # DDPG — Agent
    "DDPGAgent",
    "train_ddpg",
    # DDPG — Networks
    "Actor",
    "Critic",
    "ActorNet",
    "CriticNet",
    # DDPG — Exploration
    "OUNoise",
    # DDPG — Environment
    "DDPGMarketEnv",
    "create_sample_market_data",
]

# DAPO-Agent (high-level DAPO wrapper — IEEE IDS 2025 Contest 2nd Place)
# Source: D:/Hive/Data/trading_repos/FinRL-DAPO-SR/
from .dapo_agent import DAPOAgent

# LLM Sentiment-to-RL pipeline (DeepSeek sentiment/risk signal integration)
from .llm_sentiment_rl import (
    # Pipeline
    SentimentPipeline,
    # Batch enrichment
    enrich_dataframe,
    # Environment helper
    prepare_env_kwargs,
    # Reward adjustment
    compute_portfolio_adjustment,
    # Constants
    LLM_SENTIMENT_NEUTRAL,
    LLM_RISK_NEUTRAL,
    SENTIMENT_LABELS,
    RISK_LABELS,
    # Dataset loader
    load_nasdaq_dataset,
)

# LLMSentimentFactor — lightweight REST-only LLM signal generator (no heavy SDK)
from .llm_sentiment import (
    LLMSentimentFactor,
    enrich_dataframe_with_sentiment,
)

__all__ = __all__ + [
    # DAPO-Agent
    "DAPOAgent",
    # LLM Sentiment-to-RL (high-level pipeline)
    "SentimentPipeline",
    "enrich_dataframe",
    "prepare_env_kwargs",
    # LLMSentimentFactor (standalone, REST-only)
    "LLMSentimentFactor",
    "enrich_dataframe_with_sentiment",
    "compute_portfolio_adjustment",
    "LLM_SENTIMENT_NEUTRAL",
    "LLM_RISK_NEUTRAL",
    "SENTIMENT_LABELS",
    "RISK_LABELS",
    "load_nasdaq_dataset",
]

# Crypto RL Trading (adapted from Vnadh/RL-Crypto-Trading-Bot)
# Source: D:/Hive/Data/trading_repos/RL-Crypto-Trading-Bot/
# Supports PPO / A2C / DQN — BTC/USDT trading with Sharpe up to 2.22
from .crypto_env import (
    CryptoTradingEnv,
    prepare_crypto_data,
    compute_metrics,
)
from .crypto_trading_agent import (
    CryptoTradingAgent,
    train_ppo,
    train_a2c,
    train_dqn,
    train_and_compare,
    DEFAULT_MODELS_CONFIG,
)

__all__ = __all__ + [
    # Environment
    "CryptoTradingEnv",
    "prepare_crypto_data",
    "compute_metrics",
    # Agent
    "CryptoTradingAgent",
    # Standalone trainers
    "train_ppo",
    "train_a2c",
    "train_dqn",
    "train_and_compare",
    # Config
    "DEFAULT_MODELS_CONFIG",
]

# FinRL Crypto Trading (adapted from CryptoBot-FinRL)
# Source: D:/Hive/Data/trading_repos/CryptoBot-FinRL/
# Pure NumPy DRL: DQN / PG / A2C / PPO — no stable-baselines3 required.
from .crypto_finrl import (
    # Core environment
    CryptoTradingEnv,
    # DRL agents (pure NumPy)
    FinRLAgent,
    # Portfolio management
    PortfolioAllocator,
    # Backtesting
    CryptoPortfolioBacktester,
    # Utilities
    data_split,
    add_technical_indicators,
    DEFAULT_CRYPTO_TICKERS,
    DEFAULT_TECH_INDICATORS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_START_TRADE_DATE,
)

__all__ = __all__ + [
    # FinRL Crypto — Environment
    "CryptoTradingEnv",
    # FinRL Crypto — Agents
    "FinRLAgent",
    # FinRL Crypto — Allocator
    "PortfolioAllocator",
    # FinRL Crypto — Backtester
    "CryptoPortfolioBacktester",
    # FinRL Crypto — Utilities
    "data_split",
    "add_technical_indicators",
    # Defaults
    "DEFAULT_CRYPTO_TICKERS",
    "DEFAULT_TECH_INDICATORS",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "DEFAULT_START_TRADE_DATE",
]

# JaxMARL-HFT — GPU-accelerated multi-agent RL for HFT (ICAIF 2025)
# Source: D:/Hive/Data/trading_repos/JaxMARL-HFT/
# Pure NumPy reimplementation, no JAX dependency.
from .jaxmarl_hft import (
    # LOBSTER data
    LOBSTERData,
    # Order-book state
    LOBState,
    EventType,
    Side,
    # Environments
    HFTMarketMakingEnv,
    MultiAgentHFT,
    # PPO agent
    PPOMarketMaker,
)

__all__ = __all__ + [
    # LOBSTER data
    "LOBSTERData",
    # Order-book state
    "LOBState",
    "EventType",
    "Side",
    # Environments
    "HFTMarketMakingEnv",
    "MultiAgentHFT",
    # PPO agent
    "PPOMarketMaker",
]

# HFT DRL (adapted from DRL_for_Active_High_Frequency_Trading — Briola et al., 2021)
# Pure NumPy + Gymnasium. No PyTorch/TensorFlow.
from .drl_hft import (
    # Core environment
    HFTLOBEnvironment,
    HFTLOBEnvConfig,
    # Feature extraction
    OrderBookFeatureExtractor,
    # Policy network
    HFTPolicy,
    # Agent (DQN / A2C / PPO)
    DeepHFTAgent,
    ReplayBuffer,
    A2CBuffer,
    # Latency model
    LatencyModel,
    # Convenience factory
    make_hft_env,
    # Enums & constants
    Position,
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    ACTION_STOP,
)

__all__ = __all__ + [
    # HFT — Environment
    "HFTLOBEnvironment",
    "HFTLOBEnvConfig",
    # HFT — Feature extraction
    "OrderBookFeatureExtractor",
    # HFT — Policy network
    "HFTPolicy",
    # HFT — Agent
    "DeepHFTAgent",
    "ReplayBuffer",
    "A2CBuffer",
    # HFT — Latency model
    "LatencyModel",
    # HFT — Factory
    "make_hft_env",
    # HFT — Enums & constants
    "Position",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
    "ACTION_STOP",
]

# MultiStockRL — Cross-attention multi-asset RL trading (absorbed from D:/Hive/Data/trading_repos/MultiStockRLTrading/)
# Pure NumPy + Gymnasium implementation. No PyTorch / stable-baselines3 dependency.
# Key components:
#   CrossAttentionActorCriticPolicy : multi-head cross-attention across assets
#   MultiStockTradingEnv            : Gymnasium-compatible multi-stock trading env
#   GATCapsulePolicy                : GAT + Capsule network combined policy
#   CustomPolicy                    : Simple MLP policy (flattened obs)
#   LSTMAttentionNetwork            : LSTM with temporal attention
#   GATCapsuleNetwork               : Core GAT+Capsule network (pure NumPy)
#   CapsuleLayer                    : Capsule layer with dynamic routing
#   GATLayer                        : Graph Attention Network layer
#   AttentionBlock / AttentionPooling : Attention primitives
from .multi_stock_rl import (
    # Core environment
    MultiStockTradingEnv,
    # Policies
    CrossAttentionActorCriticPolicy,
    GATCapsulePolicy,
    CustomPolicy,
    # Networks
    LSTMAttentionNetwork,
    GATCapsuleNetwork,
    CapsuleLayer,
    GATLayer,
    # Attention primitives
    AttentionBlock,
    AttentionPooling,
    # Utilities
    StandardScaler,
    NumPySeedMixin,
    FundamentalDataFeature,
    LLMAnalystConfig,
    LLMAnalyst,
    blend_actions,
    process_indicators,
    BASE_INDICATORS,
)

__all__ = __all__ + [
    # MultiStockRL — Environment
    "MultiStockTradingEnv",
    # MultiStockRL — Policies
    "CrossAttentionActorCriticPolicy",
    "GATCapsulePolicy",
    "CustomPolicy",
    # MultiStockRL — Networks
    "LSTMAttentionNetwork",
    "GATCapsuleNetwork",
    "CapsuleLayer",
    "GATLayer",
    # MultiStockRL — Attention
    "AttentionBlock",
    "AttentionPooling",
    # MultiStockRL — Utilities
    "StandardScaler",
    "NumPySeedMixin",
    "FundamentalDataFeature",
    "LLMAnalystConfig",
    "LLMAnalyst",
    "blend_actions",
    "process_indicators",
    "BASE_INDICATORS",
]

# Double DQN Trading (adapted from value-based-deep-reinforcement-learning-trading-model-in-pytorch)
# Source: D:/Hive/Data/trading_repos/value-based-deep-reinforcement-learning-trading-model-in-pytorch/
# Pure NumPy + Gymnasium. No PyTorch.
from .value_dqn_trading import (
    # Core classes
    DoubleDQNTradingAgent,
    StockTradingEnv,
    ValueNetwork,
    DQNPolicy,
    ReplayBuffer as DDQNReplayBuffer,
    # Alias for backward compatibility
    SimpleQNetwork,
    # Factory
    create_double_dqn_agent,
    # Constants
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    # Transition
    Transition,
)

__all__ = __all__ + [
    # Double DQN — Agent
    "DoubleDQNTradingAgent",
    # Double DQN — Replay buffer
    "DDQNReplayBuffer",
    # Double DQN — Environment
    "StockTradingEnv",
    # Double DQN — Network
    "ValueNetwork",
    "SimpleQNetwork",
    # Double DQN — Policy
    "DQNPolicy",
    # Double DQN — Factory
    "create_double_dqn_agent",
    # Double DQN — Constants
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
    # Double DQN — Transition
    "Transition",
]

# Portfolio Actor-Critic RL (absorbed from D:/Hive/Data/trading_repos/Portfolio-Management-ActorCriticRL/)
# Source: Portfolio-Management-ActorCriticRL/
# Pure NumPy + Gymnasium. No PyTorch.
# Key components:
#   A2CPortfolioAgent    : Advantage Actor-Critic for portfolio management
#   DDPGPortfolioAgent   : Deep Deterministic Policy Gradient for continuous actions
#   PPOPortfolioAgent    : Proximal Policy Optimization for portfolios
#   PortfolioRLBenchmark : Compare agents against Buy & Hold
from .portfolio_actor_critic import (
    # Neural network primitives
    NeuralNetwork,
    PolicyGradientLayer,
    xavier_init,
    he_init,
    # Experience replay
    ReplayBuffer,
    PPOMemory,
    # Utility
    OUActionNoise,
    # Agents
    A2CPortfolioAgent,
    DDPGPortfolioAgent,
    PPOPortfolioAgent,
    PortfolioActor,
    PortfolioCritic,
    MultiAssetEnv,
    # Benchmark
    PortfolioRLBenchmark,
)

__all__ = __all__ + [
    # Portfolio Actor-Critic — Neural network primitives
    "NeuralNetwork",
    "PolicyGradientLayer",
    "xavier_init",
    "he_init",
    # Portfolio Actor-Critic — Experience replay
    "ReplayBuffer",
    "PPOMemory",
    # Portfolio Actor-Critic — Utility
    "OUActionNoise",
    # Portfolio Actor-Critic — Agents
    "A2CPortfolioAgent",
    "DDPGPortfolioAgent",
    "PPOPortfolioAgent",
    "PortfolioActor",
    "PortfolioCritic",
    "MultiAssetEnv",
    # Portfolio Actor-Critic — Benchmark
    "PortfolioRLBenchmark",
]
