"""均值回归策略"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class MeanReversionParams(StrategyParams):
    """均值回归策略参数"""
    lookback_period: int = 20
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    position_size: float = 0.1


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    name = "mean_reversion"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[MeanReversionParams] = None,
    ) -> None:
        super().__init__(symbol, params or MeanReversionParams())
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback_period + 1:
            return signals
        
        df = data.copy()
        
        df["sma"] = df["close"].rolling(self.params.lookback_period).mean()
        df["std"] = df["close"].rolling(self.params.lookback_period).std()
        df["z_score"] = (df["close"] - df["sma"]) / df["std"]
        
        for i in range(self.params.lookback_period, len(df)):
            current = df.iloc[i]
            
            if pd.isna(current["z_score"]):
                continue
            
            z_score = current["z_score"]
            
            if z_score < -self.params.entry_z_score:
                strength = min(abs(z_score) / 3.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        strength=strength,
                        reason=f"Mean reversion buy: z_score={z_score:.2f}",
                    )
                )
            
            elif z_score > self.params.entry_z_score:
                strength = min(abs(z_score) / 3.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        strength=strength,
                        reason=f"Mean reversion sell: z_score={z_score:.2f}",
                    )
                )
            
            elif abs(z_score) < self.params.exit_z_score:
                signals.append(
                    Signal(
                        type=SignalType.CLOSE_ALL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        reason=f"Mean reversion exit: z_score={z_score:.2f}",
                    )
                )
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """根据信号强度计算仓位"""
        base_size = context.portfolio_value * self.params.position_size
        adjusted_size = base_size * signal.strength
        return adjusted_size / signal.price
