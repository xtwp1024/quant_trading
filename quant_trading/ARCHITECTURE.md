# System Architecture

> **量化之神 (God of Quant Trading)** - Unified Quant Trading System

## Layer Overview

```
Data Layer → Signal Layer → Strategy Layer → Execution Layer → Risk Layer
     ↓              ↓              ↓              ↓             ↓
  [connectors]  [factors]     [strategies]   [execution]    [risk]
  [cache]      [generators]  [selector]    [order_mgr]   [unified]
                [combiner]    [portfolio]   [bridge]
```

## System Layers

### 1. Data Layer (`data/`)
Exchange connectors and data providers feeding all upper layers.

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `connectors/` | Exchange adapter abstractions (Hummingbot-style) | `base_connector.py`, `binance_rest.py`, `binance_ws.py` |
| `data/providers/` | Market data clients (Akshare, etc.) | `market_data_client.py` |
| `data/storage/` | Order/position/trade persistence | `order_model.py`, `position_model.py` |
| `data/models/` | MCP tool schemas for data APIs | `mcp_tools/` |

**Key Classes:**
- `BinanceRESTConnector` - REST API connector with rate limiting
- `BinanceWSConnector` - WebSocket real-time data
- `CCXTAdapter` - Unified adapter for 100+ exchanges
- `AitradosClient` - Multi-exchange unified data client

### 2. Signal Layer (`factors/`, `indicators/`)
Alpha generation and signal computation.

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `factors/` | Formulaic alpha (101 alphas), IC analysis, HFT factors | `alpha_101.py`, `ic_analyzer.py`, `hft_predictor.py` |
| `indicators/` | Pure NumPy technical indicators (MyTT) | `mytt.py` |
| `sentiment/` | News/social sentiment analysis | `finbert_sentiment.py`, `reddit_sentiment.py` |
| `factors/lob_features/` | Order book microstructure | `lob_processor.py`, `LOBFeatureEngine.py` |

**Signal Naming Conflict:** The `signal/` folder does not exist - this was a planned module. All signal generation is handled by `factors/` and `indicators/`.

### 3. Strategy Layer (`strategy/`, `strategies/`, `portfolio/`)
Trading strategy implementations and portfolio management.

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `strategy/` | Advanced strategies (缠论, Elliott Wave, Harmonic) | `chan_theory.py`, `elliott_wave.py`, `kalman_bot.py` |
| `strategies/` | Legacy strategies (V36, GridHODL, Swing) - migrated to `strategy/advanced/` | `v36_strategy.py` |
| `portfolio/` | Portfolio optimization (Markowitz, HRP, Black-Litterman) | `optimizers.py`, `risk_metrics.py` |
| `arbitrage/` | Spot-futures, triangular, pairs arbitrage | `spot_futures.py`, `triangular.py` |
| `options/` | Options pricing (Black-Scholes), Greeks, strategies | `pricing/black_scholes.py`, `strategies.py` |

**Key Classes:**
- `V36Strategy` - Trend-following with dynamic support/resistance
- `KalmanFilter` / `KalmanNet` - Pairs trading with adaptive hedge ratio
- `GaussianHMM` - Market regime detection
- `MeanVarianceOptimizer`, `HRPOptimizer`, `BlackLittermanOptimizer`

### 4. Execution Layer (`execution/`, `connectors/`)
Order execution, commission models, and smart order routing.

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `execution/` | Multi-broker adapters (Freqtrade, Hummingbot, Lumibot, OctoBot) | `executor.py`, `order_engine.py` |
| `execution/commission/` | Commission models | `maker_taker.py`, `tiered.py` |
| `execution/router/` | Smart order routing | `router.py` |
| `connectors/` | Exchange connections (see Data Layer) | `binance_trading.py` |

**Key Classes:**
- `Executor` - Central order executor
- `SmartOrderRouter` - Best venue routing
- `FreqtradeExchangeAdapter`, `HummingbotExchangeAdapter` - Broker adapters

### 5. Risk Layer (`risk/`)
Comprehensive risk management and portfolio protection.

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `risk/manager.py` | Central RiskManager with VaR, CVaR | `risk_manager.py` |
| `risk/unified_risk.py` | UnifiedRiskManager with stop-loss, position limits | `unified_risk.py` |
| `risk/var_calculator.py` | Historical and parametric VaR | `var_calculator.py` |
| `risk/matilda_risk.py` | LPM/HPM, Kappa ratios, Modigliani | `matilda_risk.py` |
| `risk/portfolio_optimizer.py` | Efficient frontier, HRP, PMPT | `portfolio_optimizer.py` |
| `risk/garch_risk.py` | GARCH volatility forecasting | `garch_risk.py` |

**Key Classes:**
- `UnifiedRiskManager` - 15+ risk checks, dry-run safety gate
- `VaRCalculator` - Historical/parametric VaR
- `KellySizer` - Fractional Kelly position sizing
- `HierarchicalRiskParity` - HRP portfolio optimization

## Module Map (35+ Modules)

| Module | Purpose | Primary Exports |
|--------|---------|----------------|
| `agent/` | Multi-agent trading (15+ agents) | `TradingAgentsCoordinator`, `HedgeFundMultiAgent`, `FinMemTradingAgent` |
| `backtester/` | SuiteTrading v2 engine, anti-overfit | `SuiteEngine`, `PositionStateMachine`, `OptunaOptimizer` |
| `data/` | Market data ingestion | `AitradosClient`, `MarketDataClient` |
| `decision/` | Adversarial multi-agent decisions | `DecisionCore`, `BullAgent`, `BearAgent` |
| `execution/` | Order execution & routing | `Executor`, `SmartOrderRouter` |
| `factors/` | Alpha library (101 alphas) | `Alpha101`, `ICAnalyzer`, `HFTPredictor` |
| `five_force/` | Five Force cognitive architecture | (deferred imports) |
| `gene_lab/` | Genetic algorithm strategy breeding | `GeneBank`, `GeneticOptimizer`, `StrategyBreeder` |
| `hft/` | High-frequency trading infrastructure | `HFTEngine`, `OrderBookManager` |
| `indicators/` | Technical indicators (MyTT) | `MACD`, `RSI`, `KDJ`, `BOLL` |
| `knowledge/` | Market regime knowledge graphs | `MarketKnowledgeGraph`, `RegimeDetector` |
| `market_making/` | Market-making strategies | `AvellanedaStoikov`, `PassivBotStrategy` |
| `memory/` | Layered memory for agents | `LayeredMemory`, `FinMemLayer` |
| `ml/` | ML models (DPML, LSTM, XGBoost) | `DualProcessVolumePredictor`, `XGBoostPredictor` |
| `multi_agent/` | PRISM coordination framework | `PrismCoordinator`, `TaskAgent` |
| `optimization/` | Kelly calculator, position sizer | `KellyCalculator`, `PositionSizer` |
| `options/` | Options pricing & strategies | `BlackScholes`, `Greeks`, `IronCondorStrategy` |
| `portfolio/` | Portfolio optimization | `MeanVarianceOptimizer`, `HRPOptimizer` |
| `risk/` | Risk management | `UnifiedRiskManager`, `VaRCalculator` |
| `rl/` | Reinforcement learning trading | `StockTradingEnv`, `DAPOAgent`, `DQNAgent` |
| `rl_trader/` | RL trader agents (PPO, SAC) | `PPOTrader`, `SACTrader` |
| `sentiment/` | Sentiment analysis | `FinBERTSentiment`, `RedditSentimentAnalyzer` |
| `strategy/` | Advanced strategy theories | `ChanTheoryAnalyzer`, `KalmanBot` |
| `strategies/` | Legacy strategies (deprecated) | Re-exports from `strategy/advanced/` |
| `backtest/` | Simple backtest engine | `BacktestEngine`, `BacktestAnalyzer` |

## Data Flow

```
Exchange APIs (Binance, Coinbase, etc.)
    │
    ▼
┌─────────────────────┐
│   Connectors Layer   │  binance_ws.py, binance_rest.py, ccxt_adapter.py
│  (Data Ingestion)    │
└─────────────────────┘
    │ raw OHLCV, orderbook, trades
    ▼
┌─────────────────────┐
│     Data Layer       │  AitradosClient, MarketDataClient
│  (Normalization)     │
└─────────────────────┘
    │ unified DataFrame
    ▼
┌─────────────────────┐
│    Factors Layer     │  Alpha101, HFT factors, LOB features
│  (Signal Generation) │
└─────────────────────┘
    │ alpha signals, IC scores
    ▼
┌─────────────────────┐
│   Strategy Layer     │  V36Strategy, KalmanFilter, RegimeAwareStrategy
│  (Decision Making)   │
└─────────────────────┘
    │ trading signals (buy/sell/hold)
    ▼
┌─────────────────────┐
│   Execution Layer    │  Executor, SmartOrderRouter, Broker adapters
│  (Order Placement)   │
└─────────────────────┘
    │ fill events
    ▼
┌─────────────────────┐
│     Risk Layer       │  UnifiedRiskManager, VaR, KellySizer
│ (Risk Management)    │
└─────────────────────┘
    │ risk checks, position limits
    ▼
Broker / Exchange
```

## Key Interfaces

### ISignalGenerator
```python
class ISignalGenerator(Protocol):
    def generate(self, data: pd.DataFrame) -> pd.Series: ...
    def get_metadata(self) -> dict: ...
```

### IStrategy
```python
class IStrategy(Protocol):
    def get_signals(self, data: pd.DataFrame) -> dict: ...
    def reset(self) -> None: ...
```

### IExecutor
```python
class IExecutor(Protocol):
    def execute(self, signal: TradingSignal) -> OrderResult: ...
    def cancel(self, order_id: str) -> bool: ...
    def get_positions(self) -> list[Position]: ...
```

### IRiskManager
```python
class IRiskManager(Protocol):
    def check_signal(self, signal: TradingSignal, portfolio: Portfolio) -> RiskCheckResult: ...
    def check_position(self, position: Position) -> RiskCheckResult: ...
```

### IOptimizer
```python
class IOptimizer(Protocol):
    def optimize(self, returns: pd.DataFrame, **kwargs) -> OptimizationResult: ...
```

## Experiment Entry Points

### Quick Start
```bash
# Run all experiments
python experiments/run_all_experiments.py --all

# Run by category
python experiments/run_all_experiments.py --category v36
python experiments/run_all_experiments.py --category multi_strategy
python experiments/run_all_experiments.py --category agent
python experiments/run_all_experiments.py --category rl
python experiments/run_all_experiments.py --category survival
python experiments/run_all_experiments.py --category production
```

### Experiment Scripts

| Script | Purpose |
|--------|---------|
| `v36_winrate_improvement.py` | V36策略胜率改进回测 |
| `v36_multi_symbol_backtest.py` | V36多币种回测 |
| `multi_strategy_backtest.py` | 多策略回测 (V36 + Grid + Selector) |
| `multi_strategy_comparison.py` | 4种策略对比 (V36, DaveLandry, GridHODL, RSI) |
| `agent_backtest.py` | Agent系统回测 |
| `rl_backtest.py` | RL Agent回测 (PPO vs Buy&Hold) |
| `survival_test.py` | 30天生存测试 |
| `stress_tests.py` | 压力测试 (黑天鹅, 闪崩) |
| `monte_carlo_sim.py` | 蒙特卡洛模拟 |
| `production_backtest.py` | 生产回测 |
| `test_binance_connection.py` | Binance连接测试 |

## Known Issues

### signal/ Folder Naming Conflict
**Problem:** The module was planned as `signal/` but conflicts with Python's built-in `signal` module.

**Solution:** All signal generation functionality is implemented in:
- `factors/` - Alpha factors, signal generators
- `indicators/` - Technical indicator signals

To create a proper signal module, rename to `signal_modules/`:
```python
# Rename signal/ to signal_modules/
import quant_trading.signal_modules as signal_gen
```

## Phase 1 vs Phase 2 Components

### Phase 1: Core Infrastructure
- `connectors/` - Exchange adapters
- `execution/` - Order execution
- `risk/` - Risk management
- `backtester/` - SuiteTrading v2 backtest engine
- `factors/` - Alpha generation

### Phase 2: Advanced Intelligence
- `agent/` - Multi-agent coordination
- `decision/` - Adversarial bull/bear debates
- `gene_lab/` - Genetic algorithm strategy breeding
- `rl/` / `rl_trader/` - Reinforcement learning agents
- `five_force/` - Five Force cognitive architecture

## Architecture Sources

This system absorbs and unifies code from:
- **Hummingbot** - Connector architecture
- **Backtrader** - Event-driven backtesting
- **Zipline** - Pipeline backtesting
- **FinRL** - Deep reinforcement learning
- **PyBroker** - High-speed backtesting
- **GeneTrader** - Genetic algorithm optimization
- **FinBERT** - NLP sentiment analysis
- **KalmanBOT** - Adaptive pairs trading
- **Avellaneda-Stoikov** - Market-making models
