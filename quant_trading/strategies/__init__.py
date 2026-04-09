"""
Quant God Trading Strategies
量化之神交易策略模块

从 AbuQuant 吸收的Chan理论策略 (D:/Hive/Data/trading_repos/abu):
- ChanAlgorithm: 均线交叉、突破、布林带三大策略
- ElliottWaveCounter: 5浪推动+3浪调整识别
- HarmonicPatternDetector: Gartley/Butterfly/Bat/Crab/Shark形态检测
- AbuQuantStrategy: Chan+Elliott+Harmonic 综合信号融合
- PatternScanner: 多标的形态扫描

从 kraken-infinity-grid 吸收的网格和定投策略:
- GridHODL: 网格+Holder结合策略
- SWING: 区间震荡高卖低买策略
- cDCA: 定时定额定投策略

从 jesse-strategies 吸收的7个Battle-Tested策略 (D:/Hive/Data/trading_repos/jesse-strategies/):
- DaveLandryStrategy: 20日通道均值回归
- DonchianStrategy: Donchian通道突破+ATR止损
- IFR2Strategy: RSI2均值回归+Ichimoku+Hilbert过滤
- MMMStrategy: 3/30日均线组合
- RSI4Strategy: Larry Connors RSI4均值回归
- SimpleBollingerStrategy: 布林带突破+Ichimoku过滤
- JesseStrategyBundle: 全部7策略bundle运行器

Grid & DCA strategies absorbed from kraken-infinity-grid:
- GridHODL: Grid + Holder combined strategy
- SWING: Range-bound swing trading strategy
- cDCA: Constant Dollar Cost Averaging strategy
"""

from .dynamic_grid import DynamicGridStrategy
from .grid_manager import GridManager

# 从 AbuQuant 吸收的策略 / Absorbed from AbuQuant
from .abu_quant import (
    ChanAlgorithm,
    ElliottWaveCounter,
    HarmonicPatternDetector,
    AbuQuantStrategy,
    PatternScanner,
    MA_CROSSOVER,
    BREAKOUT,
    BOLL_BAND,
    PatternType,
    WaveLabel,
)

# 从 kraken-infinity-grid 吸收的策略 / Absorbed from kraken-infinity-grid
from .grid_hodl import (
    GridHODLParams,
    GridHODLStrategy,
    GridOrder,
)
from .swing_strategy import (
    SWINGParams,
    SWINGStrategy,
)
from .cdca import (
    CDCAParams,
    CDCAStrategy,
)

# 从 jesse-strategies 吸收的7个Battle-Tested策略 / Absorbed from jesse-strategies
from .jesse_strategies import (
    DaveLandryStrategy,
    DonchianStrategy,
    IFR2Strategy,
    MMMStrategy,
    RSI4Strategy,
    SimpleBollingerStrategy,
    MongeYokohamaStrategy,
    JesseStrategyBundle,
    # Indicators / 指标
    sma,
    ema,
    atr,
    donchian,
    bollinger_bands,
    rsi,
    ichimoku_cloud,
    ht_trendmode,
)

# V36 A股趋势策略
from .v36_strategy import (
    V36Params,
    V36Strategy,
    V36StockPool,
    V36Backtester,
    V36SignalType,
    DEFAULT_STOCK_POOL,
    DEFAULT_SECTOR_MAP,
)

__all__ = [
    # Existing / 现有的
    "DynamicGridStrategy",
    "GridManager",
    # GridHODL (网格+Holder)
    "GridHODLParams",
    "GridHODLStrategy",
    "GridOrder",
    # SWING (区间震荡)
    "SWINGParams",
    "SWINGStrategy",
    # cDCA (定投)
    "CDCAParams",
    "CDCAStrategy",
    # AbuQuant (均线/突破/布林带 + 艾略特 + 谐波)
    "ChanAlgorithm",
    "ElliottWaveCounter",
    "HarmonicPatternDetector",
    "AbuQuantStrategy",
    "PatternScanner",
    "MA_CROSSOVER",
    "BREAKOUT",
    "BOLL_BAND",
    "PatternType",
    "WaveLabel",
    # Jesse strategies (7个Battle-Tested策略)
    "DaveLandryStrategy",
    "DonchianStrategy",
    "IFR2Strategy",
    "MMMStrategy",
    "RSI4Strategy",
    "SimpleBollingerStrategy",
    "MongeYokohamaStrategy",
    "JesseStrategyBundle",
    # Jesse indicators
    "sma",
    "ema",
    "atr",
    "donchian",
    "bollinger_bands",
    "rsi",
    "ichimoku_cloud",
    "ht_trendmode",
    # V36 A股趋势策略
    "V36Params",
    "V36Strategy",
    "V36StockPool",
    "V36Backtester",
    "V36SignalType",
    "DEFAULT_STOCK_POOL",
    "DEFAULT_SECTOR_MAP",
]
