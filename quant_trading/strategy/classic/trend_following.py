"""趋势跟踪策略"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from quant_trading.signal.types import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class TrendFollowingParams(StrategyParams):
    """趋势跟踪策略参数"""
    atr_period: int = 14
    atr_multiplier: float = 3.0
    position_size: float = 0.1


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略（基于ATR跟踪止损）"""
    
    name = "trend_following"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[TrendFollowingParams] = None,
    ) -> None:
        super().__init__(symbol, params or TrendFollowingParams())
        self._highest_price: Optional[float] = None
        self._lowest_price: Optional[float] = None
        self._trend: str = "none"  # none, up, down
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.atr_period + 1:
            return signals
        
        df = data.copy()
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(self.params.atr_period).mean()
        
        for i in range(self.params.atr_period, len(df)):
            current = df.iloc[i]
            
            if pd.isna(current["atr"]):
                continue
            
            price = current["close"]
            atr = current["atr"]
            
            if self._trend == "none":
                if self._highest_price is None or price > self._highest_price:
                    self._highest_price = price
                    stop_loss = price - self.params.atr_multiplier * atr
                    
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(price),
                            reason=f"Trend start: price={price:.2f}, stop={stop_loss:.2f}",
                            metadata={"stop_loss": float(stop_loss)},
                        )
                    )
                    self._trend = "up"
                    self._lowest_price = None
            
            elif self._trend == "up":
                if price > self._highest_price:
                    self._highest_price = price
                
                stop_loss = self._highest_price - self.params.atr_multiplier * atr
                
                if price < stop_loss:
                    signals.append(
                        Signal(
                            type=SignalType.EXIT_LONG,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(price),
                            reason=f"Trend exit: price={price:.2f} < stop={stop_loss:.2f}",
                        )
                    )
                    self._trend = "down"
                    self._highest_price = None
                    self._lowest_price = price
            
            elif self._trend == "down":
                if self._lowest_price is None or price < self._lowest_price:
                    self._lowest_price = price
                
                stop_loss = self._lowest_price + self.params.atr_multiplier * atr
                
                if price > stop_loss:
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(price),
                            reason=f"Trend reversal: price={price:.2f} > stop={stop_loss:.2f}",
                        )
                    )
                    self._trend = "up"
                    self._lowest_price = None
                    self._highest_price = price
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位"""
        position_value = context.portfolio_value * self.params.position_size
        return position_value / signal.price
