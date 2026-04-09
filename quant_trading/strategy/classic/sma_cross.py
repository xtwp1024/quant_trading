"""SMA交叉策略"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from quant_trading.signal.types import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class SMACrossParams(StrategyParams):
    """SMA交叉策略参数"""
    fast_period: int = 10
    slow_period: int = 30
    position_size: float = 0.1  # 仓位比例


class SMACrossStrategy(BaseStrategy):
    """SMA交叉策略"""
    
    name = "sma_cross"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[SMACrossParams] = None,
    ) -> None:
        super().__init__(symbol, params or SMACrossParams())
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.slow_period + 1:
            return signals
        
        df = data.copy()
        df["fast_sma"] = df["close"].rolling(self.params.fast_period).mean()
        df["slow_sma"] = df["close"].rolling(self.params.slow_period).mean()
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]
            
            if pd.isna(current["fast_sma"]) or pd.isna(current["slow_sma"]):
                continue
            
            if (
                previous["fast_sma"] <= previous["slow_sma"]
                and current["fast_sma"] > current["slow_sma"]
            ):
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        reason="SMA golden cross",
                    )
                )
            
            elif (
                previous["fast_sma"] >= previous["slow_sma"]
                and current["fast_sma"] < current["slow_sma"]
            ):
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        reason="SMA death cross",
                    )
                )
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位"""
        position_value = context.portfolio_value * self.params.position_size
        return position_value / signal.price
