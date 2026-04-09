#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Perception Layer for ETH Long Runner.
Computes technical indicators from market data.

复用: eth_analysis.py 中的技术指标计算逻辑
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("TechnicalPerception")


@dataclass
class TechnicalIndicators:
    """技术指标数据结构"""
    price: float
    ema12: float
    ema26: float
    sma20: float
    rsi: float
    atr: float
    macd_dif: float
    macd_dea: float
    macd_hist: float
    boll_upper: float
    boll_mid: float
    boll_lower: float
    support_levels: list = field(default_factory=list)
    resistance_levels: list = field(default_factory=list)
    trend: str = "neutral"
    trend_sign: str = "~"
    atr_ratio: float = 1.0
    rsi_state: str = "neutral"
    vol_state: str = "normal"


class TechnicalPerception:
    """
    技术指标感知层
    负责计算EMA/RSI/MACD/BOLL/ATR等指标
    """

    # ===================== 技术指标计算 =====================

    @staticmethod
    def EMA(series: np.ndarray, n: int) -> np.ndarray:
        """指数移动平均"""
        alpha = 2.0 / (n + 1)
        result = np.empty_like(series, dtype=float)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        return TechnicalPerception.EMA(tr, period)

    @staticmethod
    def RSI(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        delta = np.diff(close, prepend=close[0])
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = TechnicalPerception.EMA(gain, period)
        avg_loss = TechnicalPerception.EMA(loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def BOLL(close: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        mid = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            mid[i] = np.mean(close[i - period + 1:i + 1])
        std = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i - period + 1:i + 1])
        upper = mid + mult * std
        lower = mid - mult * std
        return upper, mid, lower

    @staticmethod
    def MACD(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        ema_fast = TechnicalPerception.EMA(close, fast)
        ema_slow = TechnicalPerception.EMA(close, slow)
        dif = ema_fast - ema_slow
        macd = TechnicalPerception.EMA(dif, signal)
        hist = dif - macd
        return dif, macd, hist

    # ===================== 支撑阻力位 =====================

    @staticmethod
    def find_support_resistance(ohlcv: np.ndarray, lookback: int = 50) -> Tuple[list, list]:
        """查找支撑位和阻力位"""
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        # 局部高点 (阻力位)
        resistance_levels = []
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                resistance_levels.append(recent_highs[i])

        # 局部低点 (支撑位)
        support_levels = []
        for i in range(2, len(recent_lows) - 2):
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                support_levels.append(recent_lows[i])

        # 聚类相似价位
        def cluster(levels: list, threshold: float = 0.005) -> list:
            if not levels:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]
            for level in levels[1:]:
                if abs(level - clusters[-1][0]) / clusters[-1][0] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            return [np.mean(c) for c in clusters]

        return cluster(support_levels), cluster(resistance_levels)

    # ===================== 趋势判断 =====================

    @staticmethod
    def analyze_trend(ema12: float, ema26: float, sma20: float) -> Tuple[str, str]:
        """判断中长期趋势"""
        if ema12 > ema26 and sma20 > ema26:
            return "强势上涨", "+"
        elif ema12 > ema26 and sma20 < ema26:
            return "短期上涨", "+"
        elif ema12 < ema26 and sma20 < ema26:
            return "弱势下跌", "-"
        elif ema12 < ema26 and sma20 > ema26:
            return "反弹迹象", "-"
        return "震荡", "~"

    # ===================== 波动率分析 =====================

    @staticmethod
    def volatility_analysis(atr_vals: np.ndarray, rsi: float, lookback: int = 20) -> Tuple[float, str, str]:
        """波动率状态分析"""
        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals[-lookback:])
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        if rsi > 70:
            rsi_state = "超买"
        elif rsi < 30:
            rsi_state = "超卖"
        elif rsi > 55:
            rsi_state = "偏强"
        elif rsi < 45:
            rsi_state = "偏弱"
        else:
            rsi_state = "中性"

        if atr_ratio > 1.5:
            vol_state = "高波动"
        elif atr_ratio < 0.7:
            vol_state = "低波动"
        else:
            vol_state = "正常波动"

        return atr_ratio, rsi_state, vol_state

    # ===================== 主计算函数 =====================

    def compute(self, ohlcv: np.ndarray) -> TechnicalIndicators:
        """
        从OHLCV数据计算所有技术指标

        Args:
            ohlcv: numpy array, shape (n, 5) = [open, high, low, close, volume]

        Returns:
            TechnicalIndicators 数据类
        """
        if len(ohlcv) < 30:
            logger.warning("数据长度不足30条，无法计算完整指标")
            return None

        close = ohlcv[:, 3]
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        vol = ohlcv[:, 4]

        current_price = close[-1]

        # 基础指标
        ema12 = self.EMA(close, 12)[-1]
        ema26 = self.EMA(close, 26)[-1]
        sma20 = np.mean(close[-20:])
        atr_vals = self.ATR(high, low, close, 14)
        atr = atr_vals[-1]
        rsi = self.RSI(close, 14)[-1]
        dif, macd, hist = self.MACD(close)
        upper_boll, mid_boll, lower_boll = self.BOLL(close)
        upper = upper_boll[-1]
        lower = lower_boll[-1]

        # 趋势判断
        trend, trend_sign = self.analyze_trend(ema12, ema26, sma20)

        # 支撑阻力位
        supports, resistances = self.find_support_resistance(ohlcv)

        # 波动率分析
        atr_ratio, rsi_state, vol_state = self.volatility_analysis(atr_vals, rsi)

        indicators = TechnicalIndicators(
            price=current_price,
            ema12=ema12,
            ema26=ema26,
            sma20=sma20,
            rsi=rsi,
            atr=atr,
            macd_dif=dif[-1],
            macd_dea=macd[-1],
            macd_hist=hist[-1],
            boll_upper=upper,
            boll_mid=mid_boll[-1],
            boll_lower=lower,
            support_levels=supports,
            resistance_levels=resistances,
            trend=trend,
            trend_sign=trend_sign,
            atr_ratio=atr_ratio,
            rsi_state=rsi_state,
            vol_state=vol_state
        )

        logger.info(f"[STATS] 技术指标计算完成: price={current_price:.2f}, RSI={rsi:.1f}, trend={trend}")
        return indicators

    def generate_signals(self, indicators: TechnicalIndicators) -> list:
        """
        从技术指标生成交易信号

        Returns:
            list of (signal_name, direction, weight) tuples
        """
        if indicators is None:
            return []

        signals = []
        rsi = indicators.rsi
        current_price = indicators.price
        upper = indicators.boll_upper
        lower = indicators.boll_lower

        # RSI信号
        if rsi < 30:
            signals.append(("RSI超卖", "BUY", 0.8))
        elif rsi > 70:
            signals.append(("RSI超买", "SELL", 0.8))
        elif rsi < 45:
            signals.append(("RSI偏弱", "BUY", 0.4))
        elif rsi > 55:
            signals.append(("RSI偏强", "SELL", 0.4))

        # MACD信号
        if indicators.macd_hist > 0:
            signals.append(("MACD多头", "BUY", 0.6))
        else:
            signals.append(("MACD空头", "SELL", 0.6))

        # 布林带信号
        if upper > lower:
            boll_pos = (current_price - lower) / (upper - lower) * 100
            if boll_pos < 20:
                signals.append(("价格接近下轨", "BUY", 0.7))
            elif boll_pos > 80:
                signals.append(("价格接近上轨", "SELL", 0.7))

        # EMA信号
        if indicators.ema12 > indicators.ema26:
            signals.append(("EMA多头排列", "BUY", 0.5))
        else:
            signals.append(("EMA空头排列", "SELL", 0.5))

        return signals

    def signal_to_direction(self, signals: list) -> Tuple[str, float]:
        """
        将信号列表转换为最终方向和强度

        Returns:
            (direction, strength) - direction: BUY/SELL/HOLD, strength: net weight
        """
        if not signals:
            return "HOLD", 0.0

        buy_weight = sum(s[2] for s in signals if s[1] == "BUY")
        sell_weight = sum(s[2] for s in signals if s[1] == "SELL")
        net = buy_weight - sell_weight

        if net > 1.0:
            return "BUY", net
        elif net > 0:
            return "BUY", net
        elif net < -1.0:
            return "SELL", abs(net)
        elif net < 0:
            return "SELL", abs(net)
        else:
            return "HOLD", 0.0
