"""高级交易策略 v0.2.30

迭代改进:
- v0.2.21: 统计套利策略
- v0.2.22: 配对交易策略
- v0.2.23: 波动率突破策略
- v0.2.24: 订单流策略
- v0.2.25: 趋势突破策略
- v0.2.26: 动量策略
- v0.2.27: 背离策略
- v0.2.28: 突破回踩策略
- v0.2.29: 多时间框架策略
- v0.2.30: 组合策略
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams


@dataclass
class StatArbParams(StrategyParams):
    """统计套利参数"""
    lookback_period: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    holding_period: int = 20


class StatisticalArbitrageStrategy(BaseStrategy):
    """统计套利策略 v0.2.21
    
    利用价格回归均值的特性:
    - Z-Score信号
    - 布林带回归
    - 协整关系
    """
    
    name = "stat_arb"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[StatArbParams] = None,
    ) -> None:
        super().__init__(symbol, params or StatArbParams())
        self._position_entry_time: Optional[int] = None
        self._z_score_history: deque = deque(maxlen=100)
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback_period:
            return signals
        
        df = data.copy()
        df["sma"] = df["close"].rolling(self.params.lookback_period).mean()
        df["std"] = df["close"].rolling(self.params.lookback_period).std()
        df["z_score"] = (df["close"] - df["sma"]) / df["std"]
        
        current = df.iloc[-1]
        z_score = current["z_score"]
        
        self._z_score_history.append(z_score)
        
        if self._position_entry_time is None:
            if z_score < -self.params.entry_threshold:
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=float(current["close"]),
                    strength=min(abs(z_score) / self.params.entry_threshold, 1.0),
                    reason=f"StatArb: z-score {z_score:.2f} below -threshold",
                    metadata={"z_score": z_score},
                ))
                self._position_entry_time = int(current["timestamp"])
            
            elif z_score > self.params.entry_threshold:
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=float(current["close"]),
                    strength=min(abs(z_score) / self.params.entry_threshold, 1.0),
                    reason=f"StatArb: z-score {z_score:.2f} above +threshold",
                    metadata={"z_score": z_score},
                ))
                self._position_entry_time = int(current["timestamp"])
        
        else:
            holding_time = int(current["timestamp"]) - self._position_entry_time
            time_bars = holding_time / (60 * 60 * 1000)
            
            if abs(z_score) < self.params.exit_threshold or time_bars > self.params.holding_period:
                signals.append(Signal(
                    type=SignalType.CLOSE_ALL,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=float(current["close"]),
                    reason=f"StatArb: z-score {z_score:.2f} near zero or timeout",
                    metadata={"z_score": z_score},
                ))
                self._position_entry_time = None
        
        return signals


@dataclass
class PairsTradingParams(StrategyParams):
    """配对交易参数"""
    lookback_period: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    symbol_a: str = ""
    symbol_b: str = ""


class PairsTradingStrategy(BaseStrategy):
    """配对交易策略 v0.2.22
    
    交易两个相关资产:
    - 协整检测
    - 价差交易
    - 双向持仓
    """
    
    name = "pairs_trading"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[PairsTradingParams] = None,
    ) -> None:
        super().__init__(symbol, params or PairsTradingParams())
        self._spread_history: deque = deque(maxlen=100)
        self._position: Optional[str] = None
    
    def calculate_spread(self, price_a: float, price_b: float) -> float:
        """计算价差"""
        return price_a / price_b
    
    def calculate_z_score(self, spread: float) -> float:
        """计算Z分数"""
        if len(self._spread_history) < 10:
            return 0.0
        
        mean = np.mean(self._spread_history)
        std = np.std(self._spread_history)
        
        if std == 0:
            return 0.0
        
        return (spread - mean) / std
    
    def generate_signals(self, data: pd.DataFrame, price_b: float = None) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback_period or price_b is None:
            return signals
        
        current = data.iloc[-1]
        price_a = float(current["close"])
        
        spread = self.calculate_spread(price_a, price_b)
        self._spread_history.append(spread)
        
        z_score = self.calculate_z_score(spread)
        
        if self._position is None:
            if z_score < -self.params.entry_threshold:
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=price_a,
                    reason=f"Pairs: spread z-score {z_score:.2f}",
                    metadata={"spread": spread, "z_score": z_score},
                ))
                self._position = "long_spread"
            
            elif z_score > self.params.entry_threshold:
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=price_a,
                    reason=f"Pairs: spread z-score {z_score:.2f}",
                    metadata={"spread": spread, "z_score": z_score},
                ))
                self._position = "short_spread"
        
        else:
            if abs(z_score) < self.params.exit_threshold:
                signals.append(Signal(
                    type=SignalType.CLOSE_ALL,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=price_a,
                    reason=f"Pairs: spread mean reversion",
                    metadata={"spread": spread, "z_score": z_score},
                ))
                self._position = None
        
        return signals


@dataclass
class VolatilityBreakoutParams(StrategyParams):
    """波动率突破参数"""
    atr_period: int = 14
    atr_multiplier: float = 2.0
    lookback: int = 20


class VolatilityBreakoutStrategy(BaseStrategy):
    """波动率突破策略 v0.2.23
    
    突破关键价位:
    - ATR-based止损
    - 波动率放大
    - 趋势确认
    """
    
    name = "volatility_breakout"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[VolatilityBreakoutParams] = None,
    ) -> None:
        super().__init__(symbol, params or VolatilityBreakoutParams())
        self._highest: float = 0
        self._lowest: float = float('inf')
    
    def calculate_atr(self, data: pd.DataFrame) -> float:
        """计算ATR"""
        if len(data) < self.params.atr_period + 1:
            return 0.0
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.params.atr_period).mean()
        
        return float(atr.iloc[-1])
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback:
            return signals
        
        current = data.iloc[-1]
        current_price = float(current["close"])
        
        if current_price > self._highest:
            self._highest = current_price
        
        if current_price < self._lowest:
            self._lowest = current_price
        
        atr = self.calculate_atr(data)
        breakout_level = self._highest + atr * self.params.atr_multiplier
        breakdown_level = self._lowest - atr * self.params.atr_multiplier
        
        if current_price >= breakout_level:
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=current_price,
                strength=min((current_price - self._highest + atr) / atr, 1.0) if atr > 0 else 0.5,
                reason=f"Volatility breakout: {current_price:.2f} > {breakout_level:.2f}",
                metadata={"breakout_level": breakout_level, "atr": atr},
            ))
            self._lowest = current_price
        
        elif current_price <= breakdown_level:
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=current_price,
                strength=min((self._lowest - current_price + atr) / atr, 1.0) if atr > 0 else 0.5,
                reason=f"Volatility breakdown: {current_price:.2f} < {breakdown_level:.2f}",
                metadata={"breakdown_level": breakdown_level, "atr": atr},
            ))
            self._highest = current_price
        
        return signals


@dataclass
class OrderFlowParams(StrategyParams):
    """订单流参数"""
    volume_ma_period: int = 20
    imbalance_threshold: float = 0.3
    cum_delta_threshold: float = 1000


class OrderFlowStrategy(BaseStrategy):
    """订单流策略 v0.2.24
    
    基于订单簿分析:
    - 订单不平衡
    - 累积增量
    - 交易强度
    """
    
    name = "order_flow"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[OrderFlowParams] = None,
    ) -> None:
        super().__init__(symbol, params or OrderFlowParams())
        self._volume_history: deque = deque(maxlen=50)
        self._cum_delta: float = 0
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.volume_ma_period:
            return signals
        
        df = data.copy()
        df["volume_ma"] = df["volume"].rolling(self.params.volume_ma_period).mean()
        
        current = df.iloc[-1]
        current_price = float(current["close"])
        volume = float(current["volume"])
        
        self._volume_history.append(volume)
        
        if "close" in df.columns and "open" in df.columns:
            if current["close"] > current["open"]:
                self._cum_delta += volume
            else:
                self._cum_delta -= volume
        
        avg_volume = np.mean(self._volume_history)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5 and self._cum_delta > self.params.cum_delta_threshold * volume_ratio:
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=current_price,
                strength=min(volume_ratio / 2.0, 1.0),
                reason=f"Order flow: buy pressure {self._cum_delta:.0f}",
                metadata={"cum_delta": self._cum_delta, "volume_ratio": volume_ratio},
            ))
            self._cum_delta = 0
        
        elif volume_ratio > 1.5 and self._cum_delta < -self.params.cum_delta_threshold * volume_ratio:
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=current_price,
                strength=min(volume_ratio / 2.0, 1.0),
                reason=f"Order flow: sell pressure {self._cum_delta:.0f}",
                metadata={"cum_delta": self._cum_delta, "volume_ratio": volume_ratio},
            ))
            self._cum_delta = 0
        
        return signals


@dataclass
class MomentumParams(StrategyParams):
    """动量参数"""
    fast_period: int = 5
    slow_period: int = 20
    threshold: float = 0.02


class MomentumStrategy(BaseStrategy):
    """动量策略 v0.2.26
    
    趋势动量交易:
    - ROC指标
    - 动量确认
    - 趋势持续性
    """
    
    name = "momentum"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[MomentumParams] = None,
    ) -> None:
        super().__init__(symbol, params or MomentumParams())
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.slow_period:
            return signals
        
        df = data.copy()
        
        df["fast_ma"] = df["close"].rolling(self.params.fast_period).mean()
        df["slow_ma"] = df["close"].rolling(self.params.slow_period).mean()
        df["momentum"] = (df["close"] - df["close"].shift(self.params.fast_period)) / df["close"].shift(self.params.fast_period)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_momentum = current["momentum"]
        prev_momentum = prev["momentum"]
        
        if prev_momentum < self.params.threshold and current_momentum >= self.params.threshold:
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=min(abs(current_momentum) / 0.1, 1.0),
                reason=f"Momentum bullish: {current_momentum:.2%}",
                metadata={"momentum": current_momentum},
            ))
        
        elif prev_momentum > -self.params.threshold and current_momentum <= -self.params.threshold:
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=min(abs(current_momentum) / 0.1, 1.0),
                reason=f"Momentum bearish: {current_momentum:.2%}",
                metadata={"momentum": current_momentum},
            ))
        
        return signals


@dataclass
class BreakoutRetestParams(StrategyParams):
    """突破回测参数"""
    lookback: int = 20
    retest_bars: int = 3
    confirmation_threshold: float = 0.5


class BreakoutRetestStrategy(BaseStrategy):
    """突破回踩策略 v0.2.28
    
    突破后回踩确认:
    - 突破关键位
    - 回踩测试
    - 确认入场
    """
    
    name = "breakout_retest"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[BreakoutRetestParams] = None,
    ) -> None:
        super().__init__(symbol, params or BreakoutRetestParams())
        self._breakout_level: Optional[float] = None
        self._retest_count: int = 0
        self._in_position: bool = False
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback:
            return signals
        
        df = data.copy()
        df["high"] = df["high"].rolling(self.params.lookback).max()
        df["low"] = df["low"].rolling(self.params.lookback).min()
        
        current = df.iloc[-1]
        current_price = float(current["close"])
        
        resistance = float(df.iloc[-1]["high"])
        support = float(df.iloc[-1]["low"])
        
        if not self._in_position:
            if current_price > resistance:
                self._breakout_level = resistance
                self._retest_count = 0
                self._in_position = True
                
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=int(current["timestamp"]),
                    price=current_price,
                    reason=f"Breakout: {current_price:.2f} > {resistance:.2f}",
                    metadata={"breakout_level": resistance},
                ))
        
        else:
            if current_price < self._breakout_level:
                self._retest_count += 1
            
            if self._retest_count >= self.params.retest_bars:
                if current_price > self._breakout_level * (1 - self.params.confirmation_threshold):
                    signals.append(Signal(
                        type=SignalType.CLOSE_ALL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=current_price,
                        reason=f"Retest confirmed: {self._retest_count} bars back",
                        metadata={"retest_count": self._retest_count},
                    ))
                    self._in_position = False
                    self._breakout_level = None
                    self._retest_count = 0
        
        return signals


class MultiTimeframeStrategy(BaseStrategy):
    """多时间框架策略 v0.2.29
    
    多个时间框架分析:
    - 日线趋势
    - 4小时确认
    - 1小时入场
    """
    
    name = "multi_timeframe"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[StrategyParams] = None,
    ) -> None:
        super().__init__(symbol, params or StrategyParams())
        self._trend_direction: Optional[str] = None
    
    def analyze_trend(self, data: pd.DataFrame, period: int) -> str:
        """分析趋势"""
        if len(data) < period:
            return "neutral"
        
        df = data.copy()
        df["ma"] = df["close"].rolling(period).mean()
        
        current_price = df.iloc[-1]["close"]
        ma = df.iloc[-1]["ma"]
        
        if current_price > ma * 1.02:
            return "bullish"
        elif current_price < ma * 0.98:
            return "bearish"
        
        return "neutral"
    
    def generate_signals(
        self,
        data_daily: pd.DataFrame,
        data_4h: pd.DataFrame,
        data_1h: pd.DataFrame,
    ) -> List[Signal]:
        """生成信号"""
        signals = []
        
        trend_daily = self.analyze_trend(data_daily, 20)
        trend_4h = self.analyze_trend(data_4h, 20)
        signal_1h = self.analyze_trend(data_1h, 10)
        
        current = data_1h.iloc[-1]
        
        if trend_daily == "bullish" and trend_4h == "bullish" and signal_1h == "bullish":
            signals.append(Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=1.0,
                reason="Multi-TF: All timeframes bullish",
                metadata={"daily": trend_daily, "4h": trend_4h, "1h": signal_1h},
            ))
        
        elif trend_daily == "bearish" and trend_4h == "bearish" and signal_1h == "bearish":
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=1.0,
                reason="Multi-TF: All timeframes bearish",
                metadata={"daily": trend_daily, "4h": trend_4h, "1h": signal_1h},
            ))
        
        return signals


class CompositeStrategy(BaseStrategy):
    """组合策略 v0.2.30
    
    组合多个策略:
    - 投票机制
    - 加权平均
    - 动态权重
    """
    
    name = "composite"
    
    def __init__(
        self,
        symbol: str,
        strategies: List[BaseStrategy],
        weights: List[float] = None,
    ) -> None:
        super().__init__(symbol, StrategyParams())
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        all_signals = []
        signal_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(data)
            
            for signal in signals:
                all_signals.append(signal)
                
                if signal.type == SignalType.BUY:
                    signal_scores["buy"] += weight * signal.strength
                elif signal.type == SignalType.SELL:
                    signal_scores["sell"] += weight * signal.strength
                else:
                    signal_scores["hold"] += weight
        
        if not all_signals:
            return []
        
        current = data.iloc[-1]
        
        if signal_scores["buy"] > signal_scores["sell"] and signal_scores["buy"] > 0.5:
            return [Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=signal_scores["buy"],
                reason="Composite: Buy signal",
                metadata={"scores": signal_scores},
            )]
        
        elif signal_scores["sell"] > signal_scores["buy"] and signal_scores["sell"] > 0.5:
            return [Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(current["timestamp"]),
                price=float(current["close"]),
                strength=signal_scores["sell"],
                reason="Composite: Sell signal",
                metadata={"scores": signal_scores},
            )]
        
        return []
