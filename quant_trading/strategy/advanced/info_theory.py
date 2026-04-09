"""
信息论建模策略

基于论文: arXiv:2602.14575
"Information-Theoretic Approach to Financial Market Modelling"

核心思想:
- 将金融市场视为信息通信系统
- Surprisal最小化 + KL散度最小化
- 市场时钟(Activity Time)建模
- 最小市场模型(MMM)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class InfoTheoryParams(StrategyParams):
    """信息论策略参数"""
    lookback_period: int = 100
    activity_threshold: float = 1.5
    kl_divergence_threshold: float = 0.1
    position_size: float = 0.1


class ActivityTimeCalculator:
    """
    活动时钟计算器
    
    将日历时间转换为市场活动时间
    τ(t) = ∫₀ᵗ activity(s) ds
    """
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def calculate(self, volume: pd.Series) -> pd.Series:
        """计算活动时钟"""
        normalized_volume = volume / volume.rolling(self.window).mean()
        activity_time = normalized_volume.cumsum()
        return activity_time


class SurprisalCalculator:
    """
    Surprisal（信息含量）计算器
    
    I(x) = -log P(x)
    """
    
    def __init__(self, bins: int = 50):
        self.bins = bins
    
    def calculate(self, series: pd.Series) -> pd.Series:
        """计算Surprisal"""
        hist, edges = np.histogram(series.dropna(), bins=self.bins, density=True)
        hist = hist + 1e-10  # 避免除零
        
        bin_indices = np.digitize(series, edges[:-1])
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        
        surprisal = -np.log(hist[bin_indices])
        
        return pd.Series(surprisal, index=series.index)


class KLDivergenceCalculator:
    """
    KL散度计算器
    
    D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    """
    
    def __init__(self, bins: int = 50):
        self.bins = bins
    
    def calculate(
        self,
        p_series: pd.Series,
        q_series: pd.Series,
    ) -> float:
        """计算KL散度"""
        p_hist, _ = np.histogram(p_series.dropna(), bins=self.bins, density=True)
        q_hist, _ = np.histogram(q_series.dropna(), bins=self.bins, density=True)
        
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        
        return kl_div


class InfoTheoryStrategy(BaseStrategy):
    """
    信息论建模策略
    
    使用信息论概念分析市场状态并生成交易信号
    """
    
    name = "info_theory"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[InfoTheoryParams] = None,
    ) -> None:
        super().__init__(symbol, params or InfoTheoryParams())
        
        self.activity_calculator = ActivityTimeCalculator()
        self.surprisal_calculator = SurprisalCalculator()
        self.kl_calculator = KLDivergenceCalculator()
        
        self._baseline_distribution: Optional[np.ndarray] = None
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < self.params.lookback_period:
            return signals
        
        df = data.copy()
        
        df["activity_time"] = self.activity_calculator.calculate(df["volume"])
        
        df["returns"] = df["close"].pct_change()
        df["surprisal"] = self.surprisal_calculator.calculate(df["returns"])
        
        df["activity_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        lookback = self.params.lookback_period
        
        for i in range(lookback, len(df)):
            current = df.iloc[i]
            historical = df.iloc[i-lookback:i]
            
            current_returns = df["returns"].iloc[i-20:i]
            baseline_returns = historical["returns"]
            
            try:
                kl_div = self.kl_calculator.calculate(current_returns, baseline_returns)
            except Exception:
                kl_div = 0.0
            
            avg_surprisal = historical["surprisal"].mean()
            current_surprisal = current["surprisal"]
            
            activity_signal = current["activity_ratio"]
            
            is_high_activity = activity_signal > self.params.activity_threshold
            is_high_surprisal = current_surprisal > avg_surprisal * 1.5
            is_distribution_shift = kl_div > self.params.kl_divergence_threshold
            
            if is_high_activity and is_high_surprisal:
                recent_returns = df["returns"].iloc[i-5:i].mean()
                
                if recent_returns > 0:
                    signal_type = SignalType.BUY
                    reason = f"High activity with positive returns (activity={activity_signal:.2f})"
                else:
                    signal_type = SignalType.SELL
                    reason = f"High activity with negative returns (activity={activity_signal:.2f})"
                
                strength = min(activity_signal / self.params.activity_threshold, 1.0)
                
                signals.append(
                    Signal(
                        type=signal_type,
                        symbol=self.symbol,
                        timestamp=int(current["timestamp"]),
                        price=float(current["close"]),
                        strength=strength,
                        reason=reason,
                        metadata={
                            "activity_ratio": float(activity_signal),
                            "surprisal": float(current_surprisal),
                            "kl_divergence": float(kl_div),
                        },
                    )
                )
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """根据信息论度量计算仓位"""
        base_size = context.portfolio_value * self.params.position_size
        
        surprisal_factor = signal.metadata.get("surprisal", 1.0) if signal.metadata else 1.0
        surprisal_adjustment = min(surprisal_factor / 2.0, 1.5)
        
        adjusted_size = base_size * signal.strength * surprisal_adjustment
        
        return adjusted_size / signal.price


"""
参考文献:
---------
[1] arXiv:2602.14575 - Information-Theoretic Approach to Financial Market Modelling
    Author: Eckhard Platen

关键概念:
---------
1. Surprisal最小化:
   - 市场动力学可以通过信息含量最小化来描述
   - 归一化因子过程退化到CIR结构

2. KL散度最小化:
   - 定价测度与真实世界测度的偏离
   - 基准中性定价的信息论正当性

3. 活动时钟(Activity Time):
   - 用交易活跃度替代日历时间
   - τ(t) = ∫ activity(s) ds

4. 最小市场模型(MMM):
   - 极简的市场模型结构
   - 因子在市场时钟下可解析

实现注意事项:
-------------
- 本实现为概念验证版本
- 完整实现需要估计市场时钟和因子权重
- KL散度估计对样本量敏感
- 需要考虑模型风险
"""
