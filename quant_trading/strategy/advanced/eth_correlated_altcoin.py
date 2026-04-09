"""ETH 相关性山寨币策略"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class EthCorrelatedParams(StrategyParams):
    """ETH 相关性策略参数"""
    # ETH 趋势参数
    eth_fast_period: int = 20
    eth_slow_period: int = 50
    # 山寨币高抛低吸参数 (布林带 + RSI)
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    # 仓位控制 (30% 或 50%)
    base_position_size: float = 0.1  # 基础仓位 (占总资金比例)
    altcoin_allocation_ratio: float = 0.3  # 山寨币分配比例 (例如 30%)


class EthCorrelatedAltcoinStrategy(BaseStrategy):
    """ETH 相关性山寨币策略
    
    使用 ETH 的趋势作为大方向过滤器：
    - 当 ETH 处于上升趋势 (Fast MA > Slow MA) 时，寻找山寨币的超卖机会做多。
    - 当 ETH 处于下降趋势 (Fast MA < Slow MA) 时，寻找山寨币的超买机会做空。
    
    仓位管理：
    - 山寨币的实际下单仓位 = (总资金 * 基础仓位) * 分配比例
    """
    
    name = "eth_correlated_altcoin"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[EthCorrelatedParams] = None,
    ) -> None:
        super().__init__(symbol, params or EthCorrelatedParams())
        self.params: EthCorrelatedParams = self.params  # type hint
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """计算 RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号
        
        注意：要求输入的数据 data DataFrame 中除了山寨币本身的 OHLCV 数据外，
        必须包含以 "eth_" 为前缀的以太坊收盘价列，即 "eth_close"。
        """
        signals = []
        
        if len(data) < max(self.params.eth_slow_period, self.params.bb_period) + 1:
            return signals
            
        if "eth_close" not in data.columns:
            raise ValueError("Data must conatin 'eth_close' column for ETH trend filtering.")
        
        df = data.copy()
        
        # 1. 计算 ETH 趋势 (MA)
        df["eth_fast_ma"] = df["eth_close"].rolling(self.params.eth_fast_period).mean()
        df["eth_slow_ma"] = df["eth_close"].rolling(self.params.eth_slow_period).mean()
        df["eth_uptrend"] = df["eth_fast_ma"] > df["eth_slow_ma"]
        df["eth_downtrend"] = df["eth_fast_ma"] < df["eth_slow_ma"]
        
        # 2. 计算 山寨币 布林带 (BB)
        df["bb_middle"] = df["close"].rolling(self.params.bb_period).mean()
        df["bb_std_val"] = df["close"].rolling(self.params.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std_val"] * self.params.bb_std)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std_val"] * self.params.bb_std)
        
        # 3. 计算 山寨币 RSI
        df["rsi"] = self._calculate_rsi(df["close"], self.params.rsi_period)
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]
            
            # 确保指标都已计算出来
            if pd.isna(current["eth_slow_ma"]) or pd.isna(current["rsi"]) or pd.isna(current["bb_middle"]):
                continue
            # 过滤1：ETH 处于上升趋势 -> 寻找做多机会（低吸）
            if current["eth_uptrend"]:
                # 触发：价格低于布林带下轨，且 RSI 从超卖区回升
                if current["close"] < current["bb_lower"] and previous["rsi"] < self.params.rsi_oversold and current["rsi"] >= self.params.rsi_oversold:
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(current["close"]),
                            reason="ETH Uptrend + Altcoin BB Lower + RSI Oversold Recovery",
                        )
                    )
                # 平仓：价格触及布林带中轨或上轨
                elif current["close"] >= current["bb_middle"] and previous["close"] < previous["bb_middle"]:
                    signals.append(
                        Signal(
                            type=SignalType.EXIT_LONG,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(current["close"]),
                            reason="Mean Reversion (Exit Long to BB Middle)",
                        )
                    )
            
            # 过滤2：ETH 处于下降趋势 -> 寻找做空机会（高抛）
            elif current["eth_downtrend"]:
                # 触发：价格高于布林带上轨，且 RSI 从超买区回落
                if current["close"] > current["bb_upper"] and previous["rsi"] > self.params.rsi_overbought and current["rsi"] <= self.params.rsi_overbought:
                    signals.append(
                        Signal(
                            type=SignalType.SELL, # 做空
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(current["close"]),
                            reason="ETH Downtrend + Altcoin BB Upper + RSI Overbought Fall",
                        )
                    )
                # 平仓：价格触及布林带中轨或下轨
                elif current["close"] <= current["bb_middle"] and previous["close"] > previous["bb_middle"]:
                    signals.append(
                        Signal(
                            type=SignalType.EXIT_SHORT,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(current["close"]),
                            reason="Mean Reversion (Exit Short to BB Middle)",
                        )
                    )
                    
        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位大小
        """
        # 计算 ETH 的基准对标金额
        eth_base_value = context.portfolio_value * self.params.base_position_size
        
        # 计算具体 山寨币 的金额 (30% 或 50% 等)
        target_alt_value = eth_base_value * self.params.altcoin_allocation_ratio
        
        # 给个保底的仓位数量
        amount = target_alt_value / signal.price
        if amount <= 0:
            return 1.0 # default 1 token
        return amount

