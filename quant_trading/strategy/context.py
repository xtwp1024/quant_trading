"""策略上下文"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from quant_trading.data.storage.models import Order, Position


@dataclass
class StrategyContext:
    """策略上下文 - 策略执行时的环境信息"""
    
    symbol: str
    current_price: float
    timestamp: int
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    open_orders: List[Order] = field(default_factory=list)
    portfolio_value: float = 0.0
    
    @property
    def position(self) -> Optional[Position]:
        """当前交易对持仓"""
        return self.positions.get(self.symbol)
    
    @property
    def has_position(self) -> bool:
        """是否有持仓"""
        return self.symbol in self.positions and self.positions[self.symbol].amount > 0
    
    @property
    def position_value(self) -> float:
        """持仓价值"""
        pos = self.position
        if pos:
            return pos.amount * self.current_price
        return 0.0
    
    @property
    def available_cash(self) -> float:
        """可用资金"""
        return self.cash
    
    @property
    def exposure_ratio(self) -> float:
        """当前敞口比例"""
        if self.portfolio_value == 0:
            return 0.0
        total_position_value = sum(p.amount * p.current_price for p in self.positions.values())
        return total_position_value / self.portfolio_value
