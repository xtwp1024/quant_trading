"""高级策略模块（论文级策略）"""

from quant_trading.strategy.advanced.dynamic_pmm import DynamicPMMStrategy, DynamicPMMParams
from quant_trading.strategy.advanced.hawkes_order_flow import HawkesOrderFlowStrategy
from quant_trading.strategy.advanced.info_theory import InfoTheoryStrategy
from quant_trading.strategy.advanced.pa_amm import PAAMMStrategy

# V36 趋势策略 (从strategies/迁移)
from quant_trading.strategy.advanced.v36_strategy import (
    V36Params,
    V36Strategy,
    V36StockPool,
    V36Backtester,
    V36SignalType,
)

# SWING 区间震荡策略
from quant_trading.strategy.advanced.swing_strategy import (
    SWINGParams,
    SWINGStrategy,
)

# Grid HODL 策略
from quant_trading.strategy.advanced.grid_hodl import (
    GridHODLParams,
    GridHODLStrategy,
    GridOrder,
)

# cDCA 定投策略
from quant_trading.strategy.advanced.cdca_strategy import (
    CDCAParams,
    CDCAStrategy,
)

__all__ = [
    # 原有策略
    "DynamicPMMStrategy",
    "DynamicPMMParams",
    "HawkesOrderFlowStrategy",
    "InfoTheoryStrategy",
    "PAAMMStrategy",
    # V36 趋势策略
    "V36Params",
    "V36Strategy",
    "V36StockPool",
    "V36Backtester",
    "V36SignalType",
    # SWING 区间震荡
    "SWINGParams",
    "SWINGStrategy",
    # Grid HODL
    "GridHODLParams",
    "GridHODLStrategy",
    "GridOrder",
    # cDCA 定投
    "CDCAParams",
    "CDCAStrategy",
]
