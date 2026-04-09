"""SuiteTrading v2 backtester — dual-runner engine, position FSM, anti-overfit, Optuna optimizer.

Absorbs SuiteTrading v2, pybroker, and Quantopian Zipline into ``quant_trading.backtester``:

- ``suite_engine`` — DoubleRunner (Simple + FSM), SuiteEngine orchestration
- ``position_fsm`` — PositionStateMachine with 6-shift × 6-stop × 6-archetype risk prototypes
- ``anti_overfit`` — CSCV (PBO), Deflated Sharpe Ratio, Hansen SPA
- ``optuna_optimizer`` — Optuna Bayesian optimisation with TPE/NSGA-II/CMA-ES
- ``numba_engine`` — Numba-accelerated vectorized backtest (63+ bt/sec), batch/stream modes
- ``walkforward`` — Walkforward rolling train/test analysis, PBO estimation
- ``bootstrap`` — BCa bootstrap confidence intervals for Sharpe/ProfitFactor metrics

Usage
-----
```python
from quant_trading.backtester import SuiteEngine, BacktestDataset, StrategySignals, SuiteRiskConfig

dataset = BacktestDataset(symbol="BTC/USDT", base_timeframe="1h", ohlcv=df)
signals = StrategySignals(entry_long=entries, exit_long=exits)
config = SuiteRiskConfig(initial_capital=4000.0)

engine = SuiteEngine(mode="fsm")
result = engine.run(dataset=dataset, signals=signals, risk_config=config)
```
"""

from quant_trading.backtester.suite_engine import (
    BacktestDataset,
    BacktestResult,
    PositionSnapshot,
    PositionState,
    SuiteEngine,
    SuiteRiskConfig,
    StrategySignals,
    SizingConfig,
    StopConfig,
    TrailingConfig,
    PartialTPConfig,
    BreakEvenConfig,
    PyramidConfig,
    TimeExitConfig,
    TradeRecord,
    run_fsm_backtest,
    run_simple_backtest,
)
from quant_trading.backtester.position_fsm import (
    RiskConfig,
)
from quant_trading.backtester.position_fsm import (
    BreakEvenConfig as FSMBreakEvenConfig,
    PartialTPConfig as FSMPartialTPConfig,
    PositionStateMachine,
    PositionState as FSMState,
    PositionSnapshot as FSMSnapshot,
    PyramidConfig as FSMPyramidConfig,
    RiskConfig as FSMRiskConfig,
    SizingConfig as FSMSizingConfig,
    StopConfig as FSMStopConfig,
    TimeExitConfig as FSMTimeExitConfig,
    TrailingConfig as FSMTrailingConfig,
    TransitionEvent,
    TransitionResult,
)
from quant_trading.backtester.anti_overfit import (
    AntiOverfitPipeline,
    AntiOverfitResult,
    CSCVResult,
    CSCVValidator,
    DSRResult,
    SPAResult,
    deflated_sharpe_ratio,
    hansen_spa,
)
from quant_trading.backtester.optuna_optimizer import (
    OptunaOptimizer,
    OptimizationResult,
    grid_search,
    suggest_categorical,
    suggest_float,
    suggest_int,
)
from quant_trading.backtester.event_engine import (
    Event,
    BarEvent,
    OrderEvent,
    FillEvent,
    PortfolioEvent,
    EventType,
    EventEngine,
    Context,
)
from quant_trading.backtester.pipeline import (
    Pipeline,
    Factor,
    Rank,
    ZScore,
    Returns,
    PercentChange,
    Filter,
    Screen,
    CustomFactor,
    PercentileFilter,
    TopCount,
)
from quant_trading.backtester.numba_engine import (
    NumbaEngine,
    BacktestMetrics,
    TradeRecord,
)
from quant_trading.backtester.walkforward import (
    WalkforwardAnalyzer,
    WalkforwardResult,
)
from quant_trading.backtester.bootstrap import (
    BootstrapEvaluator,
    BootstrapResult,
    normal_cdf,
    inverse_normal_cdf,
)
from quant_trading.backtester.zipline_lite import (
    # event-driven backtester core
    EventDrivenBacktester,
    # slippage models
    SlippageModel,
    VolumeShareSlippage,
    FixedSlippage,
    NoSlippage,
    # commission models
    CommissionModel,
    PerShareCommission,
    PerTradeCommission,
    PercentCommission,
    NoCommission,
    # portfolio & data
    Portfolio,
    Position,
    DataPortal,
    # pipeline
    PipelineEngine,
    PipelineLite,
    # strategy interface
    TradingAlgorithm,
    AlgoAPI,
    BarData,
    # events
    EventType,
    Event,
    BarEvent,
    OrderEvent,
    FillEvent,
    CancelEvent,
    DividendEvent,
    # built-in factors
    Returns,
    Rank,
    ZScore,
    # order
    Order,
    OrderStatus,
    LiquidityExceeded,
)
from quant_trading.backtester.suite_trading import (
    CSCVTest,
    CSCVResult,
    DSRTest,
    DSRResult,
    SPATest,
    SPAResult,
    WalkForwardEngine,
    WalkForwardResult,
    WalkForwardSplit,
    PositionFSM,
    BacktestMetrics,
    TradeRecord,
)
from quant_trading.backtester.backtrader_lite import (
    # Core engine
    CerebroLite,
    # Data feeds
    DataFeed,
    PandasDataFeed,
    CSVDataFeed,
    LineAccessor,
    # Broker
    BrokerSimulator,
    # Strategy
    Strategy,
    # Order and Position
    Order,
    OrderType,
    OrderStatus,
    Position,
    PositionSide,
    Bar,
    # Analyzers
    Analyzer,
    SharpeRatio,
    MaxDrawdown,
    CalmarRatio,
    AnnualReturn,
    TradeAnalyzer,
    # Indicators
    Indicator,
    SMA,
    EMA,
)
from quant_trading.backtester.backtrader_framework import (
    # Core engine
    Cerebro,
    # Data feeds
    DataFeedBase,
    PandasDataFeed as BTF_PandasDataFeed,
    CSVDataFeed as BTF_CSVDataFeed,
    # Broker
    Broker,
    # Slippage
    SlippageModel,
    FixedSlippage,
    VolumeShareSlippage,
    NoSlippage,
    # Commission
    CommissionScheme,
    PerShare,
    PerTrade,
    PercentCommission,
    # Strategy
    Strategy as BTFrameworkStrategy,
    ParamStore,
    # Sizer
    Sizer,
    FixedSize,
    PercentSize,
    # Analyzers
    Analyzer as BTAnalyzer,
    SharpeRatio as BTSharpeRatio,
    MaxDrawdown as BTMaxDrawdown,
    CalmarRatio as BTCalmarRatio,
    TradeAnalyzer as BTTradeAnalyzer,
    SQN,
    AnnualReturn as BTAnnualReturn,
    # Indicators
    Indicator as BTIndicator,
    SMA as BTSMA,
    EMA as BTEMA,
    MACD,
    RSI,
    BollingerBands,
    ATR,
    Stochastic,
    VWAP,
    ADX,
    CCI,
    ROC,
)

__all__ = [
    # suite_engine
    "BacktestDataset",
    "BacktestResult",
    "PositionSnapshot",
    "PositionState",
    "RiskConfig",
    "SuiteEngine",
    "SuiteRiskConfig",
    "StrategySignals",
    "SizingConfig",
    "StopConfig",
    "TrailingConfig",
    "PartialTPConfig",
    "BreakEvenConfig",
    "PyramidConfig",
    "TimeExitConfig",
    "TradeRecord",
    "run_fsm_backtest",
    "run_simple_backtest",
    # position_fsm
    "PositionStateMachine",
    "FSMState",
    "FSMSnapshot",
    "FSMRiskConfig",
    "FSMSizingConfig",
    "FSMStopConfig",
    "FSMTrailingConfig",
    "FSMPartialTPConfig",
    "FSMBreakEvenConfig",
    "FSMPyramidConfig",
    "FSMTimeExitConfig",
    "TransitionEvent",
    "TransitionResult",
    # anti_overfit
    "AntiOverfitPipeline",
    "AntiOverfitResult",
    "CSCVResult",
    "CSCVValidator",
    "DSRResult",
    "SPAResult",
    "deflated_sharpe_ratio",
    "hansen_spa",
    # optuna_optimizer
    "OptunaOptimizer",
    "OptimizationResult",
    "grid_search",
    "suggest_categorical",
    "suggest_float",
    "suggest_int",
    # event_engine (Zipline-style)
    "Event",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "PortfolioEvent",
    "EventType",
    "EventEngine",
    "Context",
    # pipeline
    "Pipeline",
    "Factor",
    "Rank",
    "ZScore",
    "Returns",
    "PercentChange",
    "Filter",
    "Screen",
    "CustomFactor",
    "PercentileFilter",
    "TopCount",
    # numba_engine (pybroker-style high-speed backtest)
    "NumbaEngine",
    "BacktestMetrics",
    # walkforward
    "WalkforwardAnalyzer",
    "WalkforwardResult",
    # bootstrap
    "BootstrapEvaluator",
    "BootstrapResult",
    "normal_cdf",
    "inverse_normal_cdf",
    # zipline_lite (Zipline-style event-driven backtester)
    "EventDrivenBacktester",
    "SlippageModel",
    "VolumeShareSlippage",
    "FixedSlippage",
    "NoSlippage",
    "CommissionModel",
    "PerShareCommission",
    "PerTradeCommission",
    "PercentCommission",
    "NoCommission",
    "Portfolio",
    "Position",
    "DataPortal",
    "PipelineEngine",
    "PipelineLite",
    "TradingAlgorithm",
    "AlgoAPI",
    "BarData",
    "EventType",
    "Event",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "CancelEvent",
    "DividendEvent",
    "Returns",
    "Rank",
    "ZScore",
    "Order",
    "OrderStatus",
    "LiquidityExceeded",
    # suite_trading (CSCV / DSR / SPA / WalkForward Engine)
    "CSCVTest",
    "CSCVResult",
    "DSRTest",
    "DSRResult",
    "SPATest",
    "SPAResult",
    "WalkForwardEngine",
    "WalkForwardResult",
    "WalkForwardSplit",
    "PositionFSM",
    "BacktestMetrics",
    # backtrader_lite (backtrader-style event-driven backtester)
    "CerebroLite",
    "DataFeed",
    "PandasDataFeed",
    "CSVDataFeed",
    "LineAccessor",
    "BrokerSimulator",
    "Strategy",
    "Order",
    "OrderType",
    "OrderStatus",
    "Position",
    "PositionSide",
    "Bar",
    "Analyzer",
    "SharpeRatio",
    "MaxDrawdown",
    "CalmarRatio",
    "AnnualReturn",
    "TradeAnalyzer",
    "Indicator",
    "SMA",
    "EMA",
    # backtrader_framework (full backtrader-pattern framework)
    "Cerebro",
    "DataFeedBase",
    "BTF_PandasDataFeed",
    "BTF_CSVDataFeed",
    "Broker",
    "SlippageModel",
    "FixedSlippage",
    "VolumeShareSlippage",
    "NoSlippage",
    "CommissionScheme",
    "PerShare",
    "PerTrade",
    "PercentCommission",
    "BTFrameworkStrategy",
    "ParamStore",
    "Sizer",
    "FixedSize",
    "PercentSize",
    "BTAnalyzer",
    "BTSharpeRatio",
    "BTMaxDrawdown",
    "BTCalmarRatio",
    "BTTradeAnalyzer",
    "SQN",
    "BTAnnualReturn",
    "BTIndicator",
    "BTSMA",
    "BTEMA",
    "MACD",
    "RSI",
    "BollingerBands",
    "ATR",
    "Stochastic",
    "VWAP",
    "ADX",
    "CCI",
    "ROC",
]
