# -*- coding: utf-8 -*-
"""
概率引擎信号集成器
Probability Engine Signal Integrator

将多种量化模型的信号与 V36 策略信号融合：
- Kelly Criterion: 凯利准则
- Omega Divergence: 欧米伽背离
- Hilbert Transform: 希尔伯特变换
- Bayesian Posterior: 贝叶斯后验
- Shannon Entropy: 香农熵
- Black-Scholes: 布莱克-舒尔斯

Usage:
    from quant_trading.signal.probability_engine import ProbabilityEngine

    engine = ProbabilityEngine(symbol='ETH/USDT')
    signal = engine.analyze(df, prices)
    print(signal)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from quant_trading.signal.gainsbot_generator import (
    GainsBotSignalGenerator,
    KellyCriterion,
    OmegaDivergence,
    HilbertTransform,
    BayesianPosterior,
    ShannonEntropy,
    BlackScholes,
)
from quant_trading.signal.types import Signal, SignalType, SignalDirection


class ProbabilitySignal(Enum):
    """概率引擎信号"""
    STRONG_BUY = "STRONG_BUY"    # 强烈买入
    BUY = "BUY"                  # 买入
    NEUTRAL = "NEUTRAL"         # 中性
    SELL = "SELL"               # 卖出
    STRONG_SELL = "STRONG_SELL" # 强烈卖出


@dataclass
class ProbabilityModelResult:
    """单个概率模型的结果"""
    name: str
    signal: str          # LONG/SHORT/NEUTRAL/BUY/SELL/HOLD/TREND/CHAOS
    confidence: float    # 0-1
    value: float         # 原始指标值
    weight: float        # 权重


@dataclass
class ProbabilityEngineResult:
    """概率引擎综合结果"""
    signal: ProbabilitySignal
    confidence: float
    models: List[ProbabilityModelResult]
    recommendation: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None


class ProbabilityEngine:
    """
    概率引擎 - 多模型信号融合

    将 6 个量化模型的信号进行加权融合，生成综合交易信号。
    """

    # 模型权重
    DEFAULT_WEIGHTS = {
        "kelly": 0.20,        # 凯利准则 - 仓位计算权威
        "omega": 0.15,        # 欧米伽 - 动量分析
        "hilbert": 0.15,      # 希尔伯特 - 周期分析
        "bayesian": 0.20,     # 贝叶斯 - 概率更新
        "entropy": 0.10,      # 香农熵 - 不确定性
        "black_scholes": 0.20 # BS模型 - 风险度量
    }

    # 信号映射表
    SIGNAL_MAP = {
        # Buy/Long signals
        "LONG": (ProbabilitySignal.BUY, 1),
        "BUY": (ProbabilitySignal.BUY, 1),
        "BULLISH": (ProbabilitySignal.BUY, 1),
        "TREND": (ProbabilitySignal.BUY, 0.5),
        # Sell/Short signals
        "SHORT": (ProbabilitySignal.SELL, 1),
        "SELL": (ProbabilitySignal.SELL, 1),
        "BEARISH": (ProbabilitySignal.SELL, 1),
        "CHAOS": (ProbabilitySignal.SELL, 0.5),
        # Neutral signals
        "NEUTRAL": (ProbabilitySignal.NEUTRAL, 0),
        "HOLD": (ProbabilitySignal.NEUTRAL, 0),
    }

    def __init__(
        self,
        symbol: str = "ETH/USDT",
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.6
    ):
        self.symbol = symbol
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.confidence_threshold = confidence_threshold
        self.generator = GainsBotSignalGenerator(symbol=symbol)

    def analyze(
        self,
        df: pd.DataFrame,
        prices: Optional[np.ndarray] = None
    ) -> ProbabilityEngineResult:
        """
        分析并生成综合信号

        Args:
            df: OHLCV 数据
            prices: 价格数组（可选，默认从 df 提取）

        Returns:
            ProbabilityEngineResult: 综合分析结果
        """
        if prices is None:
            prices = df['close'].values

        # 获取各模型信号
        model_results = self._get_model_signals(df, prices)

        # 计算加权信号
        weighted_signal = self._calculate_weighted_signal(model_results)

        # 确定最终信号
        final_signal = self._determine_signal(weighted_signal)

        # 生成建议
        recommendation = self._generate_recommendation(final_signal, model_results, prices)

        return ProbabilityEngineResult(
            signal=final_signal,
            confidence=weighted_signal['confidence'],
            models=model_results,
            recommendation=recommendation,
            entry_price=prices[-1] if len(prices) > 0 else None
        )

    def _get_model_signals(
        self,
        df: pd.DataFrame,
        prices: np.ndarray
    ) -> List[ProbabilityModelResult]:
        """获取各模型的信号"""
        results = []

        # 1. Kelly Criterion
        kelly = self._analyze_kelly(prices)
        results.append(kelly)

        # 2. Omega Divergence
        omega = self._analyze_omega(df)
        results.append(omega)

        # 3. Hilbert Transform
        hilbert = self._analyze_hilbert(prices)
        results.append(hilbert)

        # 4. Bayesian Posterior
        bayesian = self._analyze_bayesian(df)
        results.append(bayesian)

        # 5. Shannon Entropy
        entropy = self._analyze_entropy(df)
        results.append(entropy)

        # 6. Black-Scholes
        bs = self._analyze_black_scholes(df, prices)
        results.append(bs)

        return results

    def _analyze_kelly(self, prices: np.ndarray) -> ProbabilityModelResult:
        """分析凯利准则"""
        kelly_model = KellyCriterion()
        # 根据历史数据估算胜率和盈亏比
        if len(prices) >= 50:
            returns = np.diff(prices) / prices[:-1]
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.45
            avg_win = np.mean(wins) if len(wins) > 0 else 0.02
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

            kelly_model.win_rate = win_rate
            kelly_model.profit_loss_ratio = profit_loss_ratio

        signal, confidence = kelly_model.signal
        return ProbabilityModelResult(
            name="kelly",
            signal=signal,
            confidence=confidence,
            value=kelly_model.kelly_percent,
            weight=self.weights.get("kelly", 0.2)
        )

    def _analyze_omega(self, df: pd.DataFrame) -> ProbabilityModelResult:
        """分析欧米伽背离"""
        omega_model = OmegaDivergence()
        prices = df['close'].values

        if len(prices) >= 20:
            result = omega_model.calculate(prices)
            return ProbabilityModelResult(
                name="omega",
                signal=result.get("signal", "NEUTRAL"),
                confidence=result.get("confidence", 0.5),
                value=result.get("omega", 1.0),
                weight=self.weights.get("omega", 0.15)
            )

        return ProbabilityModelResult(
            name="omega", signal="NEUTRAL", confidence=0.3,
            value=1.0, weight=self.weights.get("omega", 0.15)
        )

    def _analyze_hilbert(self, prices: np.ndarray) -> ProbabilityModelResult:
        """分析希尔伯特变换"""
        hilbert_model = HilbertTransform()

        if len(prices) >= 40:
            result = hilbert_model.calculate(prices)
            return ProbabilityModelResult(
                name="hilbert",
                signal=result.get("signal", "HOLD"),
                confidence=result.get("confidence", 0.5),
                value=result.get("phase", 0.0),
                weight=self.weights.get("hilbert", 0.15)
            )

        return ProbabilityModelResult(
            name="hilbert", signal="HOLD", confidence=0.4,
            value=0.0, weight=self.weights.get("hilbert", 0.15)
        )

    def _analyze_bayesian(self, df: pd.DataFrame) -> ProbabilityModelResult:
        """分析贝叶斯后验"""
        bayesian_model = BayesianPosterior()
        prices = df['close'].values

        if len(prices) >= 20:
            result = bayesian_model.calculate(prices)
            return ProbabilityModelResult(
                name="bayesian",
                signal=result.get("signal", "HOLD"),
                confidence=result.get("confidence", 0.5),
                value=result.get("bull_probability", 0.5),
                weight=self.weights.get("bayesian", 0.2)
            )

        return ProbabilityModelResult(
            name="bayesian", signal="HOLD", confidence=0.4,
            value=0.5, weight=self.weights.get("bayesian", 0.2)
        )

    def _analyze_entropy(self, df: pd.DataFrame) -> ProbabilityModelResult:
        """分析香农熵"""
        entropy_model = ShannonEntropy()
        prices = df['close'].values

        if len(prices) >= 20:
            result = entropy_model.calculate(prices)
            return ProbabilityModelResult(
                name="entropy",
                signal=result.get("signal", "NEUTRAL"),
                confidence=result.get("confidence", 0.5),
                value=result.get("normalized_entropy", 0.5),
                weight=self.weights.get("entropy", 0.1)
            )

        return ProbabilityModelResult(
            name="entropy", signal="NEUTRAL", confidence=0.3,
            value=0.5, weight=self.weights.get("entropy", 0.1)
        )

    def _analyze_black_scholes(
        self,
        df: pd.DataFrame,
        prices: np.ndarray
    ) -> ProbabilityModelResult:
        """分析布莱克-舒尔斯"""
        bs_model = BlackScholes()

        if len(prices) >= 20:
            result = bs_model.calculate(prices)
            return ProbabilityModelResult(
                name="black_scholes",
                signal=result.get("signal", "HOLD"),
                confidence=result.get("confidence", 0.5),
                value=result.get("implied_volatility", 0.3),
                weight=self.weights.get("black_scholes", 0.2)
            )

        return ProbabilityModelResult(
            name="black_scholes", signal="HOLD", confidence=0.4,
            value=0.3, weight=self.weights.get("black_scholes", 0.2)
        )

    def _calculate_weighted_signal(
        self,
        model_results: List[ProbabilityModelResult]
    ) -> Dict[str, Any]:
        """计算加权信号"""
        total_weight = 0
        weighted_confidence = 0
        bullish_score = 0
        bearish_score = 0

        for result in model_results:
            weight = result.weight
            total_weight += weight
            weighted_confidence += result.confidence * weight

            signal_upper = result.signal.upper()
            mapped_signal = self.SIGNAL_MAP.get(signal_upper)
            if mapped_signal is None:
                continue

            probability_signal, direction_strength = mapped_signal
            directional_weight = weight * result.confidence * abs(direction_strength)

            if probability_signal == ProbabilitySignal.BUY:
                bullish_score += directional_weight
            elif probability_signal == ProbabilitySignal.SELL:
                bearish_score += directional_weight

        if total_weight > 0:
            weighted_confidence /= total_weight
            bullish_score /= total_weight
            bearish_score /= total_weight

        # 计算最终方向
        net_direction = bullish_score - bearish_score

        return {
            'confidence': weighted_confidence,
            'bullish': bullish_score,
            'bearish': bearish_score,
            'net_direction': net_direction
        }

    def _determine_signal(
        self,
        weighted_signal: Dict[str, Any]
    ) -> ProbabilitySignal:
        """确定最终信号"""
        net = weighted_signal['net_direction']
        conf = weighted_signal['confidence']

        # 调整阈值：考虑高权重高置信度模型的一致性
        if net >= 0.25 and conf >= 0.6:
            return ProbabilitySignal.STRONG_BUY
        elif net >= 0.15 and conf >= 0.5:
            return ProbabilitySignal.BUY
        elif net <= -0.25 and conf >= 0.6:
            return ProbabilitySignal.STRONG_SELL
        elif net <= -0.15 and conf >= 0.5:
            return ProbabilitySignal.SELL
        else:
            return ProbabilitySignal.NEUTRAL

    def _generate_recommendation(
        self,
        signal: ProbabilitySignal,
        model_results: List[ProbabilityModelResult],
        prices: np.ndarray
    ) -> str:
        """生成交易建议"""
        price = prices[-1] if len(prices) > 0 else 0

        signal_str = signal.value

        if signal == ProbabilitySignal.STRONG_BUY:
            sl = price * 0.97   # 3% 止损
            tp = price * 1.06   # 6% 止盈
            return f"强烈买入 | 现价 ${price:.2f} | 止损 ${sl:.2f} | 止盈 ${tp:.2f} | 风险回报比 1:2"
        elif signal == ProbabilitySignal.BUY:
            sl = price * 0.98
            tp = price * 1.04
            return f"买入 | 现价 ${price:.2f} | 止损 ${sl:.2f} | 止盈 ${tp:.2f}"
        elif signal == ProbabilitySignal.STRONG_SELL:
            sl = price * 1.03   # 3% 止损
            tp = price * 0.94   # 6% 止盈
            return f"强烈卖出 | 现价 ${price:.2f} | 止损 ${sl:.2f} | 止盈 ${tp:.2f} | 风险回报比 1:2"
        elif signal == ProbabilitySignal.SELL:
            sl = price * 1.02
            tp = price * 0.96
            return f"卖出 | 现价 ${price:.2f} | 止损 ${sl:.2f} | 止盈 ${tp:.2f}"
        else:
            return f"观望 | 现价 ${price:.2f} | 等待明确信号"

    def get_signal_for_trading(self, df: pd.DataFrame) -> Optional[Signal]:
        """获取可用于交易的 Signal 对象"""
        result = self.analyze(df)

        # 转换信号
        if result.signal == ProbabilitySignal.STRONG_BUY:
            signal_type = SignalType.BUY
        elif result.signal == ProbabilitySignal.BUY:
            signal_type = SignalType.BUY
        elif result.signal == ProbabilitySignal.STRONG_SELL:
            signal_type = SignalType.SELL
        elif result.signal == ProbabilitySignal.SELL:
            signal_type = SignalType.SELL
        else:
            return None

        # 获取价格
        price = float(df.iloc[-1]['close'])
        timestamp = int(df.index[-1].timestamp() * 1000)

        return Signal(
            type=signal_type,
            symbol=self.symbol,
            timestamp=timestamp,
            price=price,
            strength=result.confidence,
            reason=result.recommendation
        )


# 便捷函数
def get_probability_signal(
    symbol: str,
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> ProbabilityEngineResult:
    """获取概率引擎信号"""
    engine = ProbabilityEngine(symbol=symbol, weights=weights)
    return engine.analyze(df)


# 导出
__all__ = [
    'ProbabilityEngine',
    'ProbabilityEngineResult',
    'ProbabilitySignal',
    'ProbabilityModelResult',
    'get_probability_signal',
]
