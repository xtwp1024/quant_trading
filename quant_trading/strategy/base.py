"""策略基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal
from quant_trading.strategy.context import StrategyContext


@dataclass
class StrategyParams:
    """策略参数基类"""
    pass


class BaseStrategy(ABC):
    """策略基类"""
    
    name: str = "base_strategy"
    params: StrategyParams
    
    def __init__(self, symbol: str, params: Optional[StrategyParams] = None) -> None:
        self.symbol = symbol
        self.params = params or StrategyParams()
        self._data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位大小"""
        pass
    
    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """每个K线调用"""
        if self._data is None:
            return None
        
        signals = self.generate_signals(self._data)
        return signals[-1] if signals else None
    
    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """每个tick调用（高频策略）"""
        return None
    
    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass
    
    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        pass
    
    def update_data(self, new_data: pd.DataFrame) -> None:
        """更新数据"""
        if self._data is None:
            self._data = new_data
        else:
            self._data = pd.concat([self._data, new_data]).drop_duplicates(
                subset=["timestamp"]
            )
    
    def get_required_history(self) -> int:
        """获取所需历史数据长度"""
        return 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": self.params.__dict__,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseStrategy":
        """从字典创建"""
        raise NotImplementedError
