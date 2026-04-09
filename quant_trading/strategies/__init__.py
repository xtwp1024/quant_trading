"""
Quant God Trading Strategies (Legacy)
=====================================
此模块已废弃，所有策略已迁移到 quant_trading.strategy.advanced
仅保留向后兼容重导出，请使用新路径导入。

迁移指南:
  旧路径                          -> 新路径
  quant_trading.strategies.v36_strategy  -> quant_trading.strategy.advanced.v36_strategy
  quant_trading.strategies.swing_strategy -> quant_trading.strategy.advanced.swing_strategy
  quant_trading.strategies.grid_hodl     -> quant_trading.strategy.advanced.grid_hodl
  quant_trading.strategies.cdca          -> quant_trading.strategy.advanced.cdca_strategy
"""

# 向后兼容重导出 (从 strategy/advanced/)
from quant_trading.strategy.advanced.v36_strategy import (
    V36Params,
    V36Strategy,
    V36StockPool,
    V36Backtester,
    V36SignalType,
    DEFAULT_STOCK_POOL,
    DEFAULT_SECTOR_MAP,
)
from quant_trading.strategy.advanced.swing_strategy import (
    SWINGParams,
    SWINGStrategy,
)
from quant_trading.strategy.advanced.grid_hodl import (
    GridHODLParams,
    GridHODLStrategy,
    GridOrder,
)
from quant_trading.strategy.advanced.cdca_strategy import (
    CDCAParams,
    CDCAStrategy,
)

__all__ = [
    # V36
    "V36Params",
    "V36Strategy",
    "V36StockPool",
    "V36Backtester",
    "V36SignalType",
    "DEFAULT_STOCK_POOL",
    "DEFAULT_SECTOR_MAP",
    # SWING
    "SWINGParams",
    "SWINGStrategy",
    # GridHODL
    "GridHODLParams",
    "GridHODLStrategy",
    "GridOrder",
    # cDCA
    "CDCAParams",
    "CDCAStrategy",
]
