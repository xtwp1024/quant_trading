"""网格交易策略"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class GridTradingParams(StrategyParams):
    """网格交易参数"""
    grid_count: int = 10
    grid_range_percent: float = 0.1  # 网格范围（相对于当前价格）
    position_size_per_grid: float = 0.01  # 每格仓位
    grid_levels: List[float] = field(default_factory=list)


class GridTradingStrategy(BaseStrategy):
    """网格交易策略"""
    
    name = "grid_trading"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[GridTradingParams] = None,
    ) -> None:
        super().__init__(symbol, params or GridTradingParams())
        self._grids: Dict[float, str] = {}  # price -> 'buy'/'sell'
        self._initialized = False
    
    def _initialize_grids(self, current_price: float) -> None:
        """初始化网格"""
        if self._initialized:
            return
        
        grid_range = current_price * self.params.grid_range_percent
        upper_price = current_price + grid_range
        lower_price = current_price - grid_range
        
        if self.params.grid_levels:
            grid_prices = self.params.grid_levels
        else:
            grid_prices = np.linspace(
                lower_price,
                upper_price,
                self.params.grid_count + 1,
            )
        
        for i, price in enumerate(grid_prices):
            if price < current_price:
                self._grids[float(price)] = "buy"
            elif price > current_price:
                self._grids[float(price)] = "sell"
        
        self._initialized = True
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) == 0:
            return signals
        
        current = data.iloc[-1]
        current_price = float(current["close"])
        
        if not self._initialized:
            self._initialize_grids(current_price)
        
        for grid_price, side in sorted(self._grids.items()):
            if side == "buy" and current_price <= grid_price:
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current_price),
                        reason=f"Grid buy at {grid_price:.2f}",
                        metadata={"grid_price": float(grid_price)},
                    )
                )
                self._grids[grid_price] = "sell"
            
            elif side == "sell" and current_price >= grid_price:
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current_price),
                        reason=f"Grid sell at {grid_price:.2f}",
                        metadata={"grid_price": float(grid_price)},
                    )
                )
                self._grids[grid_price] = "buy"
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位"""
        position_value = context.portfolio_value * self.params.position_size_per_grid
        return position_value / signal.price
