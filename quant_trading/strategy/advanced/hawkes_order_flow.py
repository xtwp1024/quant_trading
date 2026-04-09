"""
Hawkes订单流策略

基于论文: arXiv:2601.23172
"A unified theory of order flow, market impact, and volatility"

核心思想:
- 双层Hawkes过程建模订单流（核心流 + 反应流）
- 单参数H₀统一约束：成交量粗糙度、波动率粗糙度、冲击幂律指数
- H₀ ≈ 0.75 时：
  - H_vol = 0.25（粗糙成交量）
  - H_σ = 0（极粗糙波动率）
  - δ = 0.5（平方根冲击）
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class HawkesParams(StrategyParams):
    """Hawkes策略参数"""
    kappa: float = 1.5  # Hawkes衰减系数
    alpha: float = 0.6  # 自激励系数
    h0: float = 0.75  # 统一参数
    threshold: float = 2.0  # 信号阈值
    position_size: float = 0.1


class HawkesProcess:
    """Hawkes过程模拟器"""
    
    def __init__(self, kappa: float = 1.5, alpha: float = 0.6):
        self.kappa = kappa
        self.alpha = alpha
    
    def simulate(self, n_events: int, dt: float = 0.01) -> np.ndarray:
        """模拟Hawkes过程事件"""
        events = []
        t = 0.0
        intensity = 1.0
        
        while len(events) < n_events:
            intensity = 1.0 + self.alpha * sum(
                np.exp(-self.kappa * (t - ti)) for ti in events[-100:]
            )
            
            u = np.random.random()
            t += -np.log(u) / intensity
            
            events.append(t)
        
        return np.array(events)


class HawkesOrderFlowStrategy(BaseStrategy):
    """
    Hawkes订单流策略
    
    使用双层Hawkes过程分析订单流，预测价格冲击和波动率
    """
    
    name = "hawkes_order_flow"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[HawkesParams] = None,
    ) -> None:
        super().__init__(symbol, params or HawkesParams())
        
        self.core_process = HawkesProcess(
            kappa=self.params.kappa,
            alpha=self.params.alpha,
        )
        self.reaction_process = HawkesProcess(
            kappa=self.params.kappa * 2,
            alpha=self.params.alpha * 0.5,
        )
        
        self._buy_intensity: List[float] = []
        self._sell_intensity: List[float] = []
        self._price_impact: float = 0.0
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        df = data.copy()
        
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        
        df["buy_pressure"] = np.where(df["close"] > df["open"], df["volume"], 0)
        df["sell_pressure"] = np.where(df["close"] < df["open"], df["volume"], 0)
        
        h0 = self.params.h0
        h_vol = h0 - 0.5
        h_sigma = 2 * h0 - 1.5
        
        lookback = int(1 / (h_vol + 0.5) * 10) if h_vol > -0.5 else 20
        lookback = min(max(lookback, 10), 50)
        
        df["buy_intensity"] = df["buy_pressure"].rolling(lookback).sum()
        df["sell_intensity"] = df["sell_pressure"].rolling(lookback).sum()
        
        df["intensity_ratio"] = (
            df["buy_intensity"] / (df["sell_intensity"] + 1e-10)
        )
        
        delta = 2 - 2 * h0
        df["estimated_impact"] = (
            np.sign(df["returns"]) * 
            np.abs(df["volume"]) ** delta * 
            np.sign(df["buy_intensity"] - df["sell_intensity"])
        )
        
        df["signal"] = df["intensity_ratio"].rolling(10).mean()
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            signal_value = current["signal"]
            
            if signal_value > self.params.threshold:
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        strength=min((signal_value - 1) / self.params.threshold, 1.0),
                        reason=f"Strong buy pressure: ratio={signal_value:.2f}",
                        metadata={
                            "buy_intensity": float(current["buy_intensity"]),
                            "sell_intensity": float(current["sell_intensity"]),
                            "estimated_impact": float(current["estimated_impact"]),
                        },
                    )
                )
            
            elif signal_value < 1 / self.params.threshold:
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        strength=min((1 / signal_value - 1) / self.params.threshold, 1.0),
                        reason=f"Strong sell pressure: ratio={signal_value:.2f}",
                        metadata={
                            "buy_intensity": float(current["buy_intensity"]),
                            "sell_intensity": float(current["sell_intensity"]),
                            "estimated_impact": float(current["estimated_impact"]),
                        },
                    )
                )
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """根据信号强度和预估冲击计算仓位"""
        base_size = context.portfolio_value * self.params.position_size
        
        impact_factor = abs(signal.metadata.get("estimated_impact", 0)) if signal.metadata else 0
        impact_adjustment = 1 / (1 + impact_factor)
        
        adjusted_size = base_size * signal.strength * impact_adjustment
        
        return adjusted_size / signal.price
    
    def estimate_market_impact(self, volume: float, volatility: float) -> float:
        """
        估计市场冲击
        
        基于论文的冲击幂律: Impact ~ Volume^δ
        δ = 2 - 2*H₀
        """
        delta = 2 - 2 * self.params.h0
        
        impact = volume ** delta * volatility
        
        return impact


"""
参考文献:
-----------
[1] arXiv:2601.23172 - A unified theory of order flow, market impact, and volatility
    Authors: Youssef Ouazzani Chahdi, Johannes Muhle-Karbe, Mathieu Rosenbaum, Grégoire Szymanski

关键发现:
---------
1. 双层Hawkes模型可以同时解释:
   - 有符号订单流长期记忆性
   - 无符号成交量的粗糙性
   - 波动率的粗糙性
   - 市场冲击的平方根律

2. 单参数统一约束:
   - H_vol = H₀ - 0.5 (成交量粗糙度)
   - H_σ = 2*H₀ - 1.5 (波动率粗糙度)
   - δ = 2 - 2*H₀ (冲击幂律指数)

3. 实证估计 H₀ ≈ 0.75:
   - 粗糙成交量 (H_vol ≈ 0.25)
   - 极粗糙波动率 (H_σ ≈ 0)
   - 平方根冲击律 (δ ≈ 0.5)

实现注意事项:
-------------
- 本实现为概念验证版本
- 完整实现需要逐笔订单数据
- Hawkes参数需要根据具体市场校准
- 需要考虑交易成本和延迟
"""
