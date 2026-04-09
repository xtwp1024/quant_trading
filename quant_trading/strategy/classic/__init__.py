"""经典策略模块"""

from quant_trading.strategy.classic.sma_cross import SMACrossStrategy
from quant_trading.strategy.classic.mean_reversion import MeanReversionStrategy
from quant_trading.strategy.classic.trend_following import TrendFollowingStrategy
from quant_trading.strategy.classic.grid_trading import GridTradingStrategy

__all__ = [
    "SMACrossStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "GridTradingStrategy",
]
