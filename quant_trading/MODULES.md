# Module Documentation

Detailed documentation for each module in the quant_trading system.

---

## agent/ - Multi-Agent Trading System

**Purpose:** Skill-first agent runtime providing 15+ trading agents with debate, risk, and research capabilities.

**Key Classes/Functions:**
```python
from quant_trading.agent import (
    # Skill framework
    Skill, SkillRouter,

    # Multi-LLM ensemble
    MultiLLMEnsemble,
    calibrate_probability,

    # Risk gate
    evaluate_risk_gate,
    DryRunSafetyGate,

    # Kelly sizing
    KellySizer,
    FractionalKellySizer,

    # Trading agents
    TradingAgentsCoordinator,  # 15-agent complete investment loop
    HedgeFundMultiAgent,      # Multi-agent hedge fund
    FinMemTradingAgent,       # Memory-augmented LLM trading

    # Bull/Bear debate
    BullAgent, BearAgent,
    DebateEngine,

    # Research team
    create_fundamentals_analyst,
    create_macro_analyst,
    ResearchPipeline,
)

# Usage example
coordinator = TradingAgentsCoordinator()
result = coordinator.run_cycle(market_data)
```

**Dependencies:** `agent.skill`, `agent.risk_gate`, `agent.kelly_sizer`, `agent.debate_engine`, `agent.research_team`

---

## arbitrage/ - Arbitrage Strategies

**Purpose:** Spot-futures arbitrage, triangular arbitrage, and pairs trading.

**Key Classes/Functions:**
```python
from quant_trading.arbitrage import (
    # Spot-Futures
    SpotFuturesArbitrageur,
    SpotPortfolioConstruction,
    FuturesPricingArbitrageIntervals,
    SpotFuturesRelationship,

    # Pairs trading
    ArbitragePredictor,

    # Triangular
    TriangularArbitrageEngine,
    TrianglePath,

    # Polymarket
    PolymarketArbitrageur,

    # IBKR pairs
    IBKRPairTrader,
)

# Usage example
arb = SpotFuturesArbitrageur(spot_df, futures_df)
interval = arb.calculate_arbitrage_interval()
signal = arb.generate_signal(predicted_spread, interval)
```

**Dependencies:** `pandas`, `numpy`, `sklearn`

---

## backtest/ - Simple Backtest Engine

**Purpose:** Basic backtesting engine with storage and analysis.

**Key Classes:**
```python
from quant_trading.backtest import (
    BacktestEngine,
    DataStorage,
    BacktestAnalyzer,
    BacktestMetrics,
    BacktestTrade,
)

# Usage example
engine = BacktestEngine(initial_capital=10000)
results = engine.run(data, strategy)
analyzer = BacktestAnalyzer(results)
metrics = analyzer.get_metrics()
```

---

## backtester/ - SuiteTrading v2 Engine

**Purpose:** Advanced backtesting with dual-runner engine, position FSM, anti-overfit, and Optuna optimization.

**Key Classes/Functions:**
```python
from quant_trading.backtester import (
    # Core engine
    SuiteEngine,
    BacktestDataset,
    StrategySignals,
    run_fsm_backtest,
    run_simple_backtest,

    # Position FSM (6-shift × 6-stop × 6-archetype)
    PositionStateMachine,
    FSMRiskConfig,

    # Anti-overfit (CSCV, DSR, Hansen SPA)
    AntiOverfitPipeline,
    deflated_sharpe_ratio,
    hansen_spa,

    # Optuna optimizer
    OptunaOptimizer,
    grid_search,

    # Event engine (Zipline-style)
    EventEngine,
    Event, BarEvent, OrderEvent, FillEvent,

    # Pipeline
    Pipeline,
    Factor, Rank, ZScore, Returns,
)

# Usage example
dataset = BacktestDataset(symbol="BTC/USDT", base_timeframe="1h", ohlcv=df)
signals = StrategySignals(entry_long=entries, exit_long=exits)
config = SuiteRiskConfig(initial_capital=4000.0)
engine = SuiteEngine(mode="fsm")
result = engine.run(dataset=dataset, signals=signals, risk_config=config)
```

**Dependencies:** `optuna`, `numba` (optional for `numba_engine`)

---

## config/ - Configuration

**Purpose:** Production and experimental configurations.

**Key Files:**
- `production.py` - Production settings
- `v36_config.py` - V36 strategy configuration

---

## connectors/ - Exchange Adapters

**Purpose:** Hummingbot-style exchange connectors with unified interface.

**Key Classes:**
```python
from quant_trading.connectors import (
    # Order types
    OrderType, TradeType, OrderState, PositionAction,

    # Binance connectors
    BinanceRESTConnector,
    BinanceWSConnector,
    BinanceOrderBookManager,

    # Multi-exchange adapters
    CEXAdapter,
    CCXTAdapter,        # 100+ exchanges
    UnicornBinanceREST,
    UnicornBinanceWebSocket,
)

# Usage example
connector = BinanceRESTConnector(api_key="...", api_secret="...")
ws = BinanceWSConnector()
orderbook = BinanceOrderBookManager()
```

**Dependencies:** `websockets`, `ccxt`

---

## core/ - Core Infrastructure

**Purpose:** Central brain, event bus, and self-healing engine (deferred imports).

**Key Classes:**
```python
from quant_trading.core import (
    TitanBrain,     # RAG knowledge system
    TitanCortex,   # Self-healing
    EventBus,
    TradingEngine,
    DatabaseManager,
)
```

---

## data/ - Data Layer

**Purpose:** Market data ingestion, normalization, and storage.

**Key Classes:**
```python
from quant_trading.data import (
    # Aitrados unified client
    AitradosClient,
    ExchangeDataFetcher,
    UnifiedDataFrame,

    # Providers
    AkshareMarketDataClient,
    MarketDataClient,
    IndicatorEngine,

    # MCP models
    KlineBar, KlineRequest, KlineResponse,
)
```

---

## decision/ - Decision Framework

**Purpose:** Adversarial multi-agent decision-making with bull/bear debates.

**Key Classes:**
```python
from quant_trading.decision import (
    DecisionCore,
    SignalWeight,
    VoteResult,
    OvertradingGuard,
    ReflectionAgent,
    BullAgent,
    BearAgent,
    MultiPeriodAgent,
    FourLayerStrategyFilter,  # Trend -> AI -> Setup -> Trigger
)

# Usage example
core = DecisionCore()
result = core.vote(signals, weights)
```

---

## execution/ - Order Execution

**Purpose:** Multi-broker order execution, commission models, and smart routing.

**Key Classes:**
```python
from quant_trading.execution import (
    # Core
    Executor,
    OrderEngine,

    # Commission
    MakerTakerCommission,
    TieredCommission,

    # Position tracking
    Position,
    PositionTracker,

    # Smart routing
    SmartOrderRouter,
    ExchangeQuote,

    # Broker adapters
    FreqtradeExchangeAdapter,
    HummingbotExchangeAdapter,
    LumibotBroker,
    OctoBotExecutor,
    PyCryptoBotExecutor,
)
```

---

## factors/ - Alpha & Signal Generation

**Purpose:** Formulaic alpha library (101 alphas), IC analysis, HFT factors, and ML predictors.

**Key Classes/Functions:**
```python
from quant_trading.factors import (
    # Alpha library
    Alpha101,         # All 101 formulaic alphas
    AlphaEvaluator,
    rank, ts_rank, correlation, delta, delay,

    # IC/IR analysis
    ICAnalyzer,
    information_coefficient,
    rank_ic,

    # High-frequency factors (A1-A39)
    HighFrequencyFactors,
    OrderFlowFactors,
    TradeFlowFactors,
    order_imbalance, vpin, spread_decomposition,

    # MyTT indicators
    MACD, KDJ, RSI, WR, CCI, ATR, BOLL,

    # ML predictors
    LSTMPredictor,
    XGBoostPredictor,

    # LOB features
    LOBFeatureEngine,
    HFTPredictor,

    # Kalman pairs trading
    KalmanFilter,
    KalmanNetNN,

    # GARCH
    GARCHModel,
)

# Usage example
alpha = Alpha101()
factors = alpha.compute(df)
ic_analyzer = ICAnalyzer()
ic_scores = ic_analyzer.compute_ic(factors, returns)
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `xgboost` (optional), `torch` (optional)

---

## five_force/ - Five Force Architecture

**Purpose:** Five Force cognitive architecture - the core differentiator (deferred imports).

**Note:** Module imports are deferred due to complex interdependencies.

---

## gene_lab/ - Genetic Algorithm Strategy Breeding

**Purpose:** Gene bank and genetic algorithm optimizer for strategy evolution.

**Key Classes/Functions:**
```python
from quant_trading.gene_lab import (
    GeneBank,           # JSON-backed gene vault
    GeneExtractor,      # AST parser for strategy files
    StrategyBreeder,    # Code-generation synthesizer
    Individual,
    Population,
    GeneticOptimizer,
    crossover, mutate, select_tournament,
    WalkForwardValidator,
)

# Usage example
bank = GeneBank()
extractor = GeneExtractor()
optimizer = GeneticOptimizer(population_size=100)
best = optimizer.optimize(fitness_fn)
```

---

## hft/ - High-Frequency Trading

**Purpose:** HFT infrastructure patterns (architectural reference).

**Key Classes:**
```python
from quant_trading.hft import (
    HFTEngine,
    HFTConfig,
    OrderBookManager,
    BPSSignal,
    CLOBClient,  # WebSocket client
    set_cpu_affinity,
    disable_gc,
    calculate_bps,
    LatencyTracker,
    PnLTracker,
)
```

**Note:** This is an architectural reference, not for live trading without exchange connection.

---

## indicators/ - Technical Indicators

**Purpose:** Pure NumPy technical indicators (zero Ta-Lib dependency).

**Key Functions:**
```python
from quant_trading.indicators import (
    # Level 0 helpers
    RD, RET, ABS, MAX, MIN, MA, REF, DIFF, STD, IF, SUM,
    HHV, LLV, EMA, SMA, WMA, DMA, AMA, AVEDEV, SLOPE,

    # Indicators
    MACD, KDJ, RSI, WR, CCI, TR, ATR,
    BOLL, BBI, BIAS, PSY, MTM, ROC,
    DMI, AROON, TRIX, VR, EMV, OBV, MFI,
)

# Usage example
ema_12 = EMA(close, 12)
rsi_val = RSI(close, 14)
macd, signal, hist = MACD(close)
```

---

## knowledge/ - Market Regime Knowledge

**Purpose:** Loss recovery, Obsidian integration, and regime knowledge graphs.

**Key Classes:**
```python
from quant_trading.knowledge import (
    LossRecoveryManager,
    ObsidianIntegrator,
    SentimentAnalyzer,
    MarketKnowledgeGraph,
    RegimeDetector,
)
```

---

## market_making/ - Market-Making Strategies

**Purpose:** Market-making models including Avellaneda-Stoikov, PassivBot, and Polymarket MM.

**Key Classes:**
```python
from quant_trading.market_making import (
    # Avellaneda-Stoikov
    AvellanedaStoikov,
    avellaneda_stoikov_sim,

    # PassivBot
    PassivBotStrategy,
    EvolutionaryOptimizer,

    # Policies
    GaussianPolicy,
    CategoricalPolicy,

    # Delta hedging
    DeltaHedgeEngine,
    PerUnderlyingDeltaHedge,

    # RL environments
    MarketMakerEnv,
    DQNMarketMaker,
    A2CMarketMaker,
)

# Usage example
market_maker = AvellanedaStoikov(
    sigma=0.05,
    gamma=0.1,
    k=1.5,
)
spread, reservation = market_maker.calculate_spread(
    time_to_expiry, inventory, risk_aversion
)
```

---

## memory/ - Layered Memory System

**Purpose:** FinMem-inspired layered memory for agents (perceptual, short-term, long-term, reflection).

**Key Classes:**
```python
from quant_trading.memory import (
    LayeredMemory,
    FinMemLayer,
    MemoryDatabase,
    ImportanceScorer,
    RecencyScorer,
    CompoundScorer,
    ConsolidationEngine,
    MemoryBank,
)
```

---

## ml/ - Machine Learning Models

**Purpose:** ML models for quant trading including DPML, neural nets, and feature engineering.

**Key Classes:**
```python
from quant_trading.ml import (
    # DPML (ECML PKDD 2022)
    DualProcessVolumePredictor,
    MetaLearner,
    System1Linear, System1LSTM, System2LSTM, System2Transformer,

    # Neural nets
    CNN1DModel,
    RNNModel,
    LSTMModel,

    # quant_ml
    XGBoostPredictor,
    LSTMPredictor,
    RandomForestClassifier,
    WalkForwardValidator,
    MLPipeline,
)
```

---

## multi_agent/ - PRISM Coordination

**Purpose:** Multi-agent coordination framework with parallel execution and result aggregation.

**Key Classes:**
```python
from quant_trading.multi_agent import (
    PrismCoordinator,
    Task,
    TaskResult,
    AgentConfig,
    AgentRole,
    MessageBus,
    ResourcePool,
    FinancialAnalysisAgent,
    MacroIntelligenceAgent,
    TradingDecisionAgent,
)
```

---

## optimization/ - Position & Strategy Optimization

**Purpose:** Kelly calculator, dynamic position sizing, and strategy selection.

**Key Classes:**
```python
from quant_trading.optimization import (
    KellyCalculator,
    PositionSizer,
    StrategySelector,
)
```

---

## options/ - Options Trading

**Purpose:** Black-Scholes pricing, Greeks calculation, and options strategies.

**Key Classes/Functions:**
```python
from quant_trading.options import (
    # Pricing
    BlackScholes,
    bs_price,
    bs_greeks,
    implied_volatility,
    implied_volatility_newton_raphson,

    # Greeks
    Greeks,
    calculate_greeks,
    calculate_portfolio_greeks,

    # Strategies
    StraddleStrategy,
    IronCondorStrategy,
    StrangleStrategy,
    RSIMomentumStrategy,

    # Risk
    OptionsRiskManager,
    RiskManager,
)

# Usage example
price = bs_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
```

---

## portfolio/ - Portfolio Optimization

**Purpose:** Portfolio optimization and risk metrics.

**Key Classes:**
```python
from quant_trading.portfolio import (
    # Optimizers
    MeanVarianceOptimizer,
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer,
    CVaROptimizer,
    optimize,  # Factory function

    # Risk metrics
    portfolio_risk_report,
    DrawdownInfo,
    value_at_risk,
    conditional_var,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
)

# Usage example
returns = {"AAPL": [0.01, -0.02, 0.03], "GOOG": [0.02, -0.01, 0.015]}
opt = MeanVarianceOptimizer(returns, target_return=0.12)
result = opt.solve()
print(result.weights)
```

---

## risk/ - Risk Management

**Purpose:** Comprehensive risk management with VaR, CVaR, Greeks, and unified risk controls.

**Key Classes/Functions:**
```python
from quant_trading.risk import (
    # Core
    RiskManager,
    UnifiedRiskManager,
    VolatilityGate,

    # VaR
    VaRCalculator,
    historical_var,
    parametric_var,
    cvar_var,

    # Metrics
    RiskMetricsCalculator,
    sharpe, sortino, calmar, max_drawdown,

    # Advanced (Matilda)
    lower_partial_moments,
    kappa_ratio,
    modigliani_ratio,
    roys_safety_first,

    # Portfolio
    HierarchicalRiskParity,
    EfficientFrontier,

    # GARCH
    GARCHModel,
    forecast_volatility,

    # Position sizing
    KellySizer,
    FractionalSizer,
    VolatilitySizer,

    # Dashboard
    RiskDashboardGenerator,
    RiskAlert,
)

# Usage example
risk_mgr = UnifiedRiskManager(config)
result = risk_mgr.check_signal(signal, portfolio)
```

---

## rl/ - Reinforcement Learning

**Purpose:** RL trading environments and agents (FinRL-DAPO, DRQN, DDPG, MultiStock RL).

**Key Classes:**
```python
from quant_trading.rl import (
    # Environments
    StockTradingEnv,
    MultiStockTradingEnv,
    CryptoTradingEnv,

    # DAPO (IEEE IDS 2025)
    DAPOAgent,
    DAPOBuffer,

    # Deep Q-Trading
    NumQAgent,
    NumDRegAgent,
    FinanceEnvironment,

    # MultiStock RL
    CrossAttentionActorCriticPolicy,
    MultiAssetCrossAttentionNetwork,
)
```

---

## rl_trader/ - RL Trader Agents

**Purpose:** PPO and SAC trading agents with stable-baselines3 integration.

**Key Classes:**
```python
from quant_trading.rl_trader import (
    PPOTrader,
    SACTrader,
    StocksEnv,
    BaseActorCritic,
    PPOConfig,
    SACConfig,
)
```

---

## sentiment/ - Sentiment Analysis

**Purpose:** Financial sentiment from news (FinBERT), Reddit, and congressional trading.

**Key Classes:**
```python
from quant_trading.sentiment import (
    FinBERTSentiment,
    VADERFallback,
    RedditSentimentAnalyzer,
    CryptoSentimentAnalyzer,
    SentimentRandomForest,
    RedditBERTEncoder,
    CongressDataCollector,
    get_congress_sentiment,
)
```

---

## strategy/ - Advanced Strategies

**Purpose:** Advanced technical analysis theories (缠论, Elliott Wave, Harmonic, Kalman).

**Key Classes:**
```python
from quant_trading.strategy import (
    # Chan Theory (缠论)
    ChanTheoryAnalyzer,
    Bi, XSegment, ZhongShu,

    # Elliott Wave
    ElliottWaveAnalyzer,
    Wave, WaveStructure, FibonacciLevel,

    # Harmonic patterns
    HarmonicPatternRecognizer,

    # Kalman pairs trading
    KalmanFilter,
    KalmanNet,
    PairsTradingStrategy,

    # HMM regime
    GaussianHMM,
    MarketRegimeDetector,

    # Copula pairs
    GaussianCopula,
    StudentTCopula,
    CopulaPairsStrategy,

    # US portfolio
    VolatilityRegimeDetector,
    RiskParityStrategy,
)
```

---

## strategies/ - Legacy Strategies (Deprecated)

**Purpose:** Legacy re-exports from `strategy/advanced/` for backward compatibility.

**Note:** All strategies have been migrated to `strategy/advanced/`. Use new paths.

---

## utils/ - Utilities

**Purpose:** Logging and helper utilities.

**Key Functions:**
```python
from quant_trading.utils import (
    setup_logger,
    get_logger,
)
```
