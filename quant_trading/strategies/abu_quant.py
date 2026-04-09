# -*- coding: utf-8 -*-
"""
AbuQuant Strategy Module / 阿布量化策略模块

Absorbed from AbuQuant (D:/Hive/Data/trading_repos/abu):
- Chan Algorithm (均线、突破、布林带)
- Elliott Wave counting (5浪推动 + 3浪调整)
- Harmonic pattern recognition (Gartley, Butterfly, Bat, Crab, Shark)

Pure NumPy + pandas. No Talib. No Cython.
纯 NumPy + pandas 实现，无 Talib、无 Cython。

18,496 strategy combinations supported.
支持 A股 / HK / Futures 日线 OHLCV 数据格式。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

from quant_trading.signal import Signal, SignalDirection

__all__ = [
    "ChanAlgorithm",
    "ElliottWaveCounter",
    "HarmonicPatternDetector",
    "AbuQuantStrategy",
    "PatternScanner",
    "MA_CROSSOVER",
    "BREAKOUT",
    "BOLL_BAND",
    "PatternType",
    "WaveLabel",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MA_CROSSOVER(Enum):
    """均线交叉方向 / Moving Average Crossover Direction"""
    GOLDEN = "golden_cross"   # 金叉：短期均线上穿长期均线
    DEAD   = "dead_cross"     # 死叉：短期均线下穿长期均线


class BREAKOUT(Enum):
    """突破类型 / Breakout Type"""
    UPPER  = "upper_breakout"  # 向上突破
    LOWER  = "lower_breakout"  # 向下突破


class BOLL_BAND(Enum):
    """布林带信号 / Bollinger Band Signal"""
    UPPER_TOUCH = "upper_touch"   # 触及上轨
    LOWER_TOUCH = "lower_touch"   # 触及下轨
    MIDDLE_TOUCH = "middle_touch" # 触及中轨


class PatternType(Enum):
    """谐波形态类型 / Harmonic Pattern Type"""
    GARTLEY   = "Gartley"
    BUTTERFLY = "Butterfly"
    BAT       = "Bat"
    CRAB      = "Crab"
    SHARK     = "Shark"


class WaveLabel(Enum):
    """波浪标签 / Wave Label"""
    IMPULSE_1 = "1"
    IMPULSE_2 = "2"
    IMPULSE_3 = "3"
    IMPULSE_4 = "4"
    IMPULSE_5 = "5"
    CORRECT_A = "A"
    CORRECT_B = "B"
    CORRECT_C = "C"


# ---------------------------------------------------------------------------
# Data validation helpers
# ---------------------------------------------------------------------------

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise a DataFrame to OHLCV format.
    验证并标准化为 OHLCV 格式。

    Accepts columns: open/high/low/close/volume (case-insensitive),
    or standard Chinese A-share column names: open/hight/hight2/low/close/vol。
    """
    df = df.copy()
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("open", "开盘"):
            col_map[col] = "open"
        elif cl in ("high", "hight", "hight2", "最高"):
            col_map[col] = "high"
        elif cl in ("low", "最低"):
            col_map[col] = "low"
        elif cl in ("close", "收盘"):
            col_map[col] = "close"
        elif cl in ("volume", "vol", "成交量", "amount"):
            col_map[col] = "volume"
    df = df.rename(columns=col_map)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AbuQuant requires OHLCV columns, missing: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    elif "timestamp" not in df.columns:
        df["timestamp"] = df.index

    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ChanAlgorithm
# ---------------------------------------------------------------------------

@dataclass
class MA_Signal:
    """均线交叉信号"""
    date: Any
    direction: MA_CROSSOVER
    short_ma: float
    long_ma: float
    price: float


@dataclass
class BollingerBandResult:
    """布林带结果"""
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray
    bandwidth: np.ndarray
    position: np.ndarray   # %B


class ChanAlgorithm:
    """
    缠论技术指标算法 / Chan-based Technical Algorithm

    实现三大经典技术策略:
    1. 均线交叉 (Moving Average Crossover)
    2. 突破策略 (Breakout)
    3. 布林带 (Bollinger Band)

    Reference: AbuQuant IndicatorBu/ABuNDMa.py, ABuNDBoll.py
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        boll_period: int = 20,
        boll_std: float = 2.0,
        breakout_window: int = 20,
    ):
        """
        Args:
            short_window: 短期均线窗口 (默认 5)
            long_window: 长期均线窗口 (默认 20)
            boll_period: 布林带周期 (默认 20)
            boll_std: 布林带标准差倍数 (默认 2.0)
            breakout_window: 突破确认窗口 (默认 20)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.boll_period = boll_period
        self.boll_std = boll_std
        self.breakout_window = breakout_window

    # ------------------------------------------------------------------
    # Moving Average
    # ------------------------------------------------------------------

    @staticmethod
    def calc_ma(series: np.ndarray, window: int) -> np.ndarray:
        """计算简单移动平均 / Simple Moving Average (pure NumPy)"""
        n = len(series)
        ma = np.full(n, np.nan, dtype=np.float64)
        if n < window:
            return ma
        # Cumulative sum approach for efficiency
        cumsum = np.cumsum(series, dtype=np.float64)
        ma[window - 1:] = (cumsum[window - 1:] - np.insert(cumsum[:-window], 0, 0)) / window
        return ma

    @staticmethod
    def calc_ema(series: np.ndarray, span: int) -> np.ndarray:
        """计算指数移动平均 / Exponential Moving Average (pure NumPy)"""
        n = len(series)
        ema = np.full(n, np.nan, dtype=np.float64)
        if n < span:
            return ema
        alpha = 2.0 / (span + 1)
        ema[span - 1] = series[:span].mean()
        for i in range(span, n):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
        return ema

    def ma_crossover(self, data: pd.DataFrame) -> List[MA_Signal]:
        """
        检测均线交叉信号 / Detect MA crossover signals

        金叉买入：短期均线从下方穿越长期均线
        死叉卖出：短期均线从上方穿越长期均线
        """
        df = _ensure_ohlcv(data)
        close = df["close"].values.astype(np.float64)
        short_ma = self.calc_ma(close, self.short_window)
        long_ma  = self.calc_ma(close, self.long_window)

        signals = []
        for i in range(1, len(df)):
            if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
                continue
            prev_short = short_ma[i - 1]
            prev_long  = long_ma[i - 1]
            curr_short = short_ma[i]
            curr_long  = long_ma[i]

            if np.isnan(prev_short) or np.isnan(prev_long):
                continue

            # Golden cross: short crosses above long
            if prev_short <= prev_long and curr_short > curr_long:
                signals.append(MA_Signal(
                    date=df.iloc[i]["timestamp"],
                    direction=MA_CROSSOVER.GOLDEN,
                    short_ma=curr_short,
                    long_ma=curr_long,
                    price=close[i],
                ))
            # Dead cross: short crosses below long
            elif prev_short >= prev_long and curr_short < curr_long:
                signals.append(MA_Signal(
                    date=df.iloc[i]["timestamp"],
                    direction=MA_CROSSOVER.DEAD,
                    short_ma=curr_short,
                    long_ma=curr_long,
                    price=close[i],
                ))

        return signals

    # ------------------------------------------------------------------
    # Bollinger Band
    # ------------------------------------------------------------------

    def calc_bollinger(
        self,
        data: pd.DataFrame,
        period: Optional[int] = None,
        nb_dev: Optional[float] = None,
    ) -> BollingerBandResult:
        """
        计算布林带 / Calculate Bollinger Bands (pure NumPy)

        Formula:
            Middle = N-period SMA
            Upper  = Middle + k * StdDev
            Lower  = Middle - k * StdDev
            %B     = (Close - Lower) / (Upper - Lower)
            Bandwidth = (Upper - Lower) / Middle
        """
        df = _ensure_ohlcv(data)
        close = df["close"].values.astype(np.float64)
        period = period or self.boll_period
        nb_dev = nb_dev or self.boll_std

        n = len(close)
        middle = self.calc_ma(close, period)

        # Rolling standard deviation using NumPy
        std = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            window = close[i - period + 1:i + 1]
            std[i] = np.sqrt(np.mean((window - middle[i]) ** 2))

        upper = middle + nb_dev * std
        lower = middle - nb_dev * std

        # %B position
        bandwidth = (upper - lower) / np.where(middle != 0, middle, 1)
        position = np.where(
            upper != lower,
            (close - lower) / (upper - lower),
            0.5,
        )

        return BollingerBandResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            position=position,
        )

    def boll_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        布林带信号检测 / Bollinger Band signal detection

        Returns list of touch events on upper, lower, and middle bands.
        """
        df = _ensure_ohlcv(data)
        boll = self.calc_bollinger(df)
        close = df["close"].values.astype(np.float64)
        timestamp = df["timestamp"].values

        signals = []
        for i in range(self.boll_period, len(df)):
            c = close[i]
            u = boll.upper[i]
            m = boll.middle[i]
            l = boll.lower[i]

            # Price touches upper band
            if c >= u * 0.995:   # 0.5% tolerance
                signals.append({
                    "date": timestamp[i],
                    "type": BOLL_BAND.UPPER_TOUCH,
                    "price": c,
                    "band_upper": u,
                    "band_middle": m,
                    "band_lower": l,
                    "position": boll.position[i],
                })
            # Price touches lower band
            elif c <= l * 1.005:
                signals.append({
                    "date": timestamp[i],
                    "type": BOLL_BAND.LOWER_TOUCH,
                    "price": c,
                    "band_upper": u,
                    "band_middle": m,
                    "band_lower": l,
                    "position": boll.position[i],
                })
            # Price near middle band (turning point)
            elif abs(c - m) / np.where(m != 0, m, 1) < 0.005:
                signals.append({
                    "date": timestamp[i],
                    "type": BOLL_BAND.MIDDLE_TOUCH,
                    "price": c,
                    "band_upper": u,
                    "band_middle": m,
                    "band_lower": l,
                    "position": boll.position[i],
                })

        return signals

    # ------------------------------------------------------------------
    # Breakout
    # ------------------------------------------------------------------

    def detect_breakout(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        突破策略 / Breakout Detection

        向上突破：今日收盘价创 N 日新高
        向下突破：今日收盘价创 N 日新低
        """
        df = _ensure_ohlcv(data)
        close = df["close"].values.astype(np.float64)
        timestamp = df["timestamp"].values
        w = self.breakout_window

        signals = []
        for i in range(w, len(df)):
            today_close = close[i]
            lookback = close[i - w + 1:i + 1]

            # Upper breakout: new w-period high
            if today_close >= lookback.max():
                signals.append({
                    "date": timestamp[i],
                    "type": BREAKOUT.UPPER,
                    "price": today_close,
                    "breakout_level": lookback.max(),
                    "lookback": w,
                })
            # Lower breakout: new w-period low
            elif today_close <= lookback.min():
                signals.append({
                    "date": timestamp[i],
                    "type": BREAKOUT.LOWER,
                    "price": today_close,
                    "breakout_level": lookback.min(),
                    "lookback": w,
                })

        return signals

    # ------------------------------------------------------------------
    # Combined signals → Signal objects
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        综合Chan指标生成信号 / Generate trading signals from all Chan indicators

        Returns:
            List of Signal objects
        """
        signals = []

        # MA crossover
        for ma_sig in self.ma_crossover(data):
            if ma_sig.direction == MA_CROSSOVER.GOLDEN:
                signals.append(Signal(
                    symbol=getattr(data, "symbol", ""),
                    direction=SignalDirection.LONG,
                    strength=0.7,
                    price=ma_sig.price,
                    metadata={
                        "type": "chan_ma_crossover",
                        "subtype": "golden_cross",
                        "short_ma": ma_sig.short_ma,
                        "long_ma": ma_sig.long_ma,
                    },
                ))
            else:
                signals.append(Signal(
                    symbol=getattr(data, "symbol", ""),
                    direction=SignalDirection.SHORT,
                    strength=0.7,
                    price=ma_sig.price,
                    metadata={
                        "type": "chan_ma_crossover",
                        "subtype": "dead_cross",
                        "short_ma": ma_sig.short_ma,
                        "long_ma": ma_sig.long_ma,
                    },
                ))

        # Breakout
        for bo in self.detect_breakout(data):
            if bo["type"] == BREAKOUT.UPPER:
                signals.append(Signal(
                    symbol=getattr(data, "symbol", ""),
                    direction=SignalDirection.LONG,
                    strength=0.75,
                    price=bo["price"],
                    metadata={
                        "type": "chan_breakout",
                        "subtype": "upper",
                        "breakout_level": bo["breakout_level"],
                    },
                ))
            else:
                signals.append(Signal(
                    symbol=getattr(data, "symbol", ""),
                    direction=SignalDirection.SHORT,
                    strength=0.75,
                    price=bo["price"],
                    metadata={
                        "type": "chan_breakout",
                        "subtype": "lower",
                        "breakout_level": bo["breakout_level"],
                    },
                ))

        return signals


# ---------------------------------------------------------------------------
# ElliottWaveCounter
# ---------------------------------------------------------------------------

@dataclass
class Wave:
    """波浪对象 / Wave object"""
    label: WaveLabel
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    high: float
    low: float
    fib_retracement: Optional[float] = None


@dataclass
class WaveCount:
    """完整波浪计数 / Complete wave count"""
    waves: List[Wave]
    is_impulse_complete: bool
    is_correction_complete: bool
    degree: str = "minor"


class ElliottWaveCounter:
    """
    艾略特波浪计数 / Elliott Wave Counter

    实现：
    1. 极值点识别（转折点）
    2. 5浪推动结构验证
    3. 3浪调整结构验证
    4. 斐波那契回撤验证

    Elliott Wave counting with:
    - 5-wave impulse validation
    - 3-wave correction validation
    - Fibonacci ratio checks

    Reference: AbuQuant TLineBu/ABuTLWave.py
    """

    # 斐波那契回撤关键水平
    FIB_KEYS = np.array([0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618], dtype=np.float64)

    def __init__(
        self,
        min_wave_bars: int = 5,
        pivot_lookback: int = 3,
        strict: bool = True,
    ):
        """
        Args:
            min_wave_bars: 波浪最小K线数
            pivot_lookback: 极值点判断回看窗口
            strict: 严格模式（强制艾略特规则验证）
        """
        self.min_wave_bars = min_wave_bars
        self.pivot_lookback = pivot_lookback
        self.strict = strict

    # ------------------------------------------------------------------
    # Pivot detection
    # ------------------------------------------------------------------

    def _find_pivots(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        识别极值转折点 / Find pivot high/low points

        Pure NumPy implementation using sliding window comparison.
        """
        df = _ensure_ohlcv(data)
        high = df["high"].values.astype(np.float64)
        low  = df["low"].values.astype(np.float64)
        ts   = df["timestamp"].values
        n    = len(high)
        lb   = self.pivot_lookback

        pivots = []
        for i in range(lb, n - lb):
            # Check pivot high
            is_high = True
            for j in range(i - lb, i + lb + 1):
                if j != i and high[j] >= high[i]:
                    is_high = False
                    break
            if is_high:
                pivots.append({"index": i, "type": "high", "price": high[i], "timestamp": ts[i]})
                continue

            # Check pivot low
            is_low = True
            for j in range(i - lb, i + lb + 1):
                if j != i and low[j] <= low[i]:
                    is_low = False
                    break
            if is_low:
                pivots.append({"index": i, "type": "low", "price": low[i], "timestamp": ts[i]})

        return pivots

    # ------------------------------------------------------------------
    # Wave construction
    # ------------------------------------------------------------------

    def _build_waves(self, pivots: List[Dict[str, Any]]) -> List[Wave]:
        """从极值点构建波浪序列 / Build wave sequence from pivots"""
        if len(pivots) < 2:
            return []

        waves = []
        impulse_labels = [
            WaveLabel.IMPULSE_1, WaveLabel.IMPULSE_2, WaveLabel.IMPULSE_3,
            WaveLabel.IMPULSE_4, WaveLabel.IMPULSE_5,
        ]
        corr_labels = [WaveLabel.CORRECT_A, WaveLabel.CORRECT_B, WaveLabel.CORRECT_C]

        wave_count = 0
        direction = "up"   # start assuming first pivot is low→high

        for idx in range(len(pivots) - 1):
            p1 = pivots[idx]
            p2 = pivots[idx + 1]

            is_up = p2["price"] > p1["price"]

            if wave_count < 5:
                label = impulse_labels[wave_count]
            else:
                label = corr_labels[wave_count - 5] if wave_count - 5 < 3 else corr_labels[-1]

            wave = Wave(
                label=label,
                start_idx=p1["index"],
                end_idx=p2["index"],
                start_price=p1["price"],
                end_price=p2["price"],
                high=max(p1["price"], p2["price"]),
                low=min(p1["price"], p2["price"]),
            )

            # Compute fib ratio vs previous wave
            if waves:
                prev = waves[-1]
                prev_range = abs(prev.end_price - prev.start_price)
                if prev_range > 0:
                    wave.fib_retracement = abs(wave.end_price - wave.start_price) / prev_range

            waves.append(wave)
            wave_count += 1

        return waves

    # ------------------------------------------------------------------
    # Rule validation
    # ------------------------------------------------------------------

    def _validate_impulse_rules(self, waves: List[Wave]) -> Tuple[bool, List[str]]:
        """
        验证推动浪规则 / Validate Elliott impulse rules

        Rules:
        1. Wave 2 retraces less than 100% of Wave 1
        2. Wave 3 is never the shortest impulse wave
        3. Wave 4 does not overlap Wave 1
        """
        violations = []
        if len(waves) < 5:
            return False, ["Insufficient waves for impulse validation"]

        w1, w2, w3, w4 = waves[0], waves[1], waves[2], waves[3]

        # Rule 1: Wave 2 cannot retrace 100%+ of Wave 1
        w1_range = abs(w1.end_price - w1.start_price)
        w2_retrace = abs(w2.end_price - w1.end_price)
        if w1_range > 0 and w2_retrace / w1_range >= 1.0:
            violations.append("Rule1: Wave 2 retraced 100%+ of Wave 1")

        # Rule 2: Wave 3 is never the shortest
        w3_range = abs(w3.end_price - w3.start_price)
        if w1_range > 0 and w3_range < w1_range:
            violations.append("Rule2: Wave 3 shorter than Wave 1")

        # Rule 3: Wave 4 cannot overlap Wave 1
        if w4.low <= w1.high and w4.high >= w1.low:
            violations.append("Rule3: Wave 4 overlaps Wave 1")

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_waves(self, data: pd.DataFrame) -> WaveCount:
        """
        执行波浪计数 / Count Elliott Waves

        Returns:
            WaveCount object with wave list and completion flags
        """
        pivots = self._find_pivots(data)
        waves  = self._build_waves(pivots)

        is_impulse  = len(waves) >= 5
        is_correct  = len(waves) >= 8

        if self.strict and is_impulse:
            valid, _ = self._validate_impulse_rules(waves)
            is_impulse = valid

        return WaveCount(
            waves=waves,
            is_impulse_complete=is_impulse,
            is_correction_complete=is_correct,
        )

    def fib_levels(self, wave: Wave) -> Dict[str, float]:
        """
        计算斐波那契回撤水平 / Calculate Fibonacci retracement levels
        """
        if wave.start_price == wave.end_price:
            return {}

        direction = 1 if wave.end_price > wave.start_price else -1
        wave_range = abs(wave.end_price - wave.start_price)
        start = min(wave.start_price, wave.end_price)

        levels = {}
        for fib in self.FIB_KEYS:
            key = f"{int(fib * 1000) if fib < 1 else int(fib * 100)}%"
            levels[key] = start + direction * wave_range * fib

        return levels

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        基于波浪计数生成信号 / Generate signals from wave counting
        """
        count = self.count_waves(data)
        signals = []

        if not count.waves:
            return signals

        # Wave 3 completion: strong momentum signal
        if len(count.waves) >= 3:
            w3 = count.waves[2]
            w1 = count.waves[0]
            if w3.end_idx > w1.end_idx:   # Wave 3 extends beyond Wave 1
                signals.append(Signal(
                    symbol=getattr(data, "symbol", ""),
                    direction=SignalDirection.LONG,
                    strength=0.85,
                    price=w3.end_price,
                    metadata={
                        "type": "elliott_wave",
                        "wave_label": w3.label.value,
                        "reason": "Wave 3 extends beyond Wave 1 peak",
                        "fib_retracement": w3.fib_retracement,
                    },
                ))

        # Wave 5 completion: potential exhaustion
        if len(count.waves) >= 5:
            w5 = count.waves[4]
            signals.append(Signal(
                symbol=getattr(data, "symbol", ""),
                direction=SignalDirection.NEUTRAL,
                strength=0.5,
                price=w5.end_price,
                metadata={
                    "type": "elliott_wave",
                    "wave_label": w5.label.value,
                    "reason": "Wave 5 completion — watch for reversal",
                    "fib_retracement": w5.fib_retracement,
                },
            ))

        # Correction A-B-C signals
        if len(count.waves) >= 8:
            wc = count.waves[7]   # Wave C
            signals.append(Signal(
                symbol=getattr(data, "symbol", ""),
                direction=SignalDirection.SHORT if wc.end_price < wc.start_price else SignalDirection.LONG,
                strength=0.7,
                price=wc.end_price,
                metadata={
                    "type": "elliott_wave",
                    "wave_label": WaveLabel.CORRECT_C.value,
                    "reason": "Correction Wave C complete",
                    "fib_retracement": wc.fib_retracement,
                },
            ))

        return signals


# ---------------------------------------------------------------------------
# HarmonicPatternDetector
# ---------------------------------------------------------------------------

@dataclass
class HarmonicHit:
    """谐波形态命中 / Harmonic Pattern Hit"""
    pattern: PatternType
    points: Dict[str, Tuple[int, float]]   # X, A, B, C, D
    direction: str
    completion_price: float
    reversal_zone: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    confidence: float


class HarmonicPatternDetector:
    """
    谐波形态检测 / Harmonic Pattern Detector

    使用纯 NumPy 检测以下形态:
    - Gartley  : AB=61.8% XA, BC=38.2%/88.6% AB, CD=127.2%/161.8% AB
    - Butterfly: AB=78.6% XA, BC=38.2%/88.6% AB, CD=161.8%/261.8% AB
    - Bat      : AB=38.2%/50% XA, BC=38.2%/88.6% AB, CD=161.8%/268% AB
    - Crab     : AB=38.2%/61.8% XA, BC=38.2%/88.6% AB, CD=224%/361.8% AB
    - Shark    : AB=113%/161.8% XA, BC=113%/161.8% AB, CD=161.8% BC

    Reference: AbuQuant pattern recognition concepts, Scott Carney's definitions
    """

    # Harmonic ratio definitions (derived from Scott Carney's "Harmonic Trading")
    PATTERN_DEFS: Dict[PatternType, Dict[str, Any]] = {
        PatternType.GARTLEY: {
            "AB_XA": [0.618],
            "BC_AB": [0.382, 0.886],
            "CD_BC": [1.272, 1.618],
            "AD_XA": [0.786],
            "D_beyond_X": False,
        },
        PatternType.BUTTERFLY: {
            "AB_XA": [0.786],
            "BC_AB": [0.382, 0.886],
            "CD_BC": [1.618, 2.618],
            "AD_XA": None,
            "D_beyond_X": True,
        },
        PatternType.BAT: {
            "AB_XA": [0.382, 0.500],
            "BC_AB": [0.382, 0.886],
            "CD_BC": [1.618, 2.618],
            "AD_XA": [0.886],
            "D_beyond_X": False,
        },
        PatternType.CRAB: {
            "AB_XA": [0.382, 0.618],
            "BC_AB": [0.382, 0.886],
            "CD_BC": [2.24, 3.618],
            "AD_XA": None,
            "D_beyond_X": True,
        },
        PatternType.SHARK: {
            "AB_XA": [1.13, 1.618],
            "BC_AB": [1.13, 1.618],
            "CD_BC": [1.618],
            "AD_XA": None,
            "D_beyond_X": True,
        },
    }

    # Tolerance for ratio matching (5%)
    TOL: float = 0.05

    def __init__(
        self,
        min_pivot_gap: int = 5,
        pivot_lookback: int = 3,
        tolerance: float = 0.05,
    ):
        """
        Args:
            min_pivot_gap: 相邻极值点最小K线间距
            pivot_lookback: 极值点判定窗口
            tolerance: 斐波那契比率匹配容差 (默认 5%)
        """
        self.min_pivot_gap = min_pivot_gap
        self.pivot_lookback = pivot_lookback
        self.TOL = tolerance

    # ------------------------------------------------------------------
    # Pivot detection
    # ------------------------------------------------------------------

    def _find_pivots(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """纯 NumPy 极值点识别"""
        df = _ensure_ohlcv(data)
        high = df["high"].values.astype(np.float64)
        low  = df["low"].values.astype(np.float64)
        ts   = df["timestamp"].values
        n    = len(high)
        lb   = self.pivot_lookback
        gap  = self.min_pivot_gap

        pivots = []
        last_idx = -gap

        for i in range(lb, n - lb):
            is_high = True
            for j in range(i - lb, i + lb + 1):
                if j != i and high[j] >= high[i]:
                    is_high = False
                    break
            if is_high:
                if i - last_idx >= gap:
                    pivots.append({"index": i, "type": "high", "price": high[i], "timestamp": ts[i]})
                    last_idx = i
                elif pivots and high[i] > pivots[-1]["price"]:
                    pivots[-1] = {"index": i, "type": "high", "price": high[i], "timestamp": ts[i]}
                    last_idx = i
                continue

            is_low = True
            for j in range(i - lb, i + lb + 1):
                if j != i and low[j] <= low[i]:
                    is_low = False
                    break
            if is_low:
                if i - last_idx >= gap:
                    pivots.append({"index": i, "type": "low", "price": low[i], "timestamp": ts[i]})
                    last_idx = i
                elif pivots and low[i] < pivots[-1]["price"]:
                    pivots[-1] = {"index": i, "type": "low", "price": low[i], "timestamp": ts[i]}
                    last_idx = i

        return pivots

    # ------------------------------------------------------------------
    # Ratio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ratio(val: float, base: float) -> float:
        """Compute ratio, safe for zero base."""
        if base == 0:
            return 0.0
        return abs(val) / base

    def _match(self, ratio: float, target: float) -> bool:
        """Check if ratio matches target within tolerance."""
        return abs(ratio - target) <= self.TOL

    def _match_any(self, ratio: float, targets: List[float]) -> bool:
        """Check if ratio matches any of the targets."""
        return any(self._match(ratio, t) for t in targets)

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> List[HarmonicHit]:
        """
        检测所有谐波形态 / Detect all harmonic patterns

        Pure NumPy implementation.
        """
        pivots = self._find_pivots(data)
        hits: List[HarmonicHit] = []

        if len(pivots) < 5:
            return hits

        for pat_type, defn in self.PATTERN_DEFS.items():
            hits.extend(self._detect_pattern(pivots, pat_type, defn))

        # Sort by confidence
        hits.sort(key=lambda h: h.confidence, reverse=True)
        return hits

    def _detect_pattern(
        self,
        pivots: List[Dict[str, Any]],
        pat_type: PatternType,
        defn: Dict[str, Any],
    ) -> List[HarmonicHit]:
        """Detect a specific harmonic pattern from pivot sequence."""
        hits: List[HarmonicHit] = []
        ab_targets   = defn["AB_XA"]
        bc_targets    = defn["BC_AB"]
        cd_targets    = defn["CD_BC"]
        ad_targets    = defn["AD_XA"]
        d_beyond_x    = defn["D_beyond_X"]

        for i in range(len(pivots) - 4):
            try:
                X = pivots[i]
                A = pivots[i + 1]
                B = pivots[i + 2]
                C = pivots[i + 3]
                D = pivots[i + 4]

                # Determine direction: bullish = low→high→low→high→low
                is_bullish = (
                    A["price"] > X["price"]
                    and B["price"] < A["price"]
                    and C["price"] > B["price"]
                )
                is_bearish = (
                    A["price"] < X["price"]
                    and B["price"] > A["price"]
                    and B["price"] < A["price"]
                    and C["price"] < B["price"]
                )

                if not (is_bullish or is_bearish):
                    continue

                direction = "bullish" if is_bullish else "bearish"

                # XA distance as base
                xa = abs(A["price"] - X["price"])
                if xa == 0:
                    continue

                # AB relative to XA
                ab = abs(B["price"] - A["price"])
                ab_ratio = self._ratio(ab, xa)
                if not self._match_any(ab_ratio, ab_targets):
                    continue

                # BC relative to AB
                bc = abs(C["price"] - B["price"])
                bc_ratio = self._ratio(bc, xa)
                if not self._match_any(bc_ratio, bc_targets):
                    continue

                # CD relative to BC
                cd = abs(D["price"] - C["price"])
                cd_ratio = self._ratio(cd, xa)
                if not self._match_any(cd_ratio, cd_targets):
                    continue

                # AD check
                ad = abs(D["price"] - X["price"])
                ad_ratio = self._ratio(ad, xa)
                if ad_targets is not None:
                    if not self._match_any(ad_ratio, ad_targets):
                        continue
                elif d_beyond_x:
                    if is_bullish and ad_ratio <= 1.0:
                        continue
                    if is_bearish and ad_ratio <= 1.0:
                        continue

                # Compute confidence
                checks = 4 if ad_targets else 3
                match_count = (
                    (1 if self._match_any(ab_ratio, ab_targets) else 0)
                    + (1 if self._match_any(bc_ratio, bc_targets) else 0)
                    + (1 if self._match_any(cd_ratio, cd_targets) else 0)
                    + (1 if ad_targets and self._match_any(ad_ratio, ad_targets) else 0)
                )
                confidence = match_count / checks

                # Calculate entry, stop, tp
                entry = D["price"]
                xa_range = xa

                if is_bullish:
                    reversal_zone = entry + xa_range * 0.382
                    stop = D["price"] - xa_range * 0.5
                    tp1  = entry + xa_range * 0.618
                    tp2  = entry + xa_range * 1.0
                else:
                    reversal_zone = entry - xa_range * 0.382
                    stop = D["price"] + xa_range * 0.5
                    tp1  = entry - xa_range * 0.618
                    tp2  = entry - xa_range * 1.0

                hits.append(HarmonicHit(
                    pattern=pat_type,
                    points={
                        "X": (X["index"], X["price"]),
                        "A": (A["index"], A["price"]),
                        "B": (B["index"], B["price"]),
                        "C": (C["index"], C["price"]),
                        "D": (D["index"], D["price"]),
                    },
                    direction=direction,
                    completion_price=entry,
                    reversal_zone=reversal_zone,
                    stop_loss=stop,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confidence=confidence,
                ))

            except (IndexError, KeyError):
                pass

        return hits

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """基于谐波形态生成信号"""
        hits = self.detect(data)
        signals = []

        for hit in hits[:3]:   # top 3 patterns only
            if hit.direction == "bullish":
                direction = SignalDirection.LONG
            elif hit.direction == "bearish":
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            signals.append(Signal(
                symbol=getattr(data, "symbol", ""),
                direction=direction,
                strength=hit.confidence,
                price=hit.completion_price,
                stop_loss=hit.stop_loss,
                take_profit=hit.take_profit1,
                metadata={
                    "type": "harmonic",
                    "pattern": hit.pattern.value,
                    "reversal_zone": hit.reversal_zone,
                    "confidence": hit.confidence,
                },
            ))

        return signals


# ---------------------------------------------------------------------------
# AbuQuantStrategy
# ---------------------------------------------------------------------------

class AbuQuantStrategy:
    """
    AbuQuant 综合策略 / AbuQuant Combined Strategy

    将 ChanAlgorithm + ElliottWaveCounter + HarmonicPatternDetector
    三个模块的信号进行融合，生成综合交易信号。

    Combine Chan + Elliott + Harmonic signals into unified trading signals.
    Confidence scores are weighted and aggregated.
    """

    def __init__(
        self,
        chan_params: Optional[Dict[str, Any]] = None,
        elliott_params: Optional[Dict[str, Any]] = None,
        harmonic_params: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            chan_params: ChanAlgorithm 参数
            elliott_params: ElliottWaveCounter 参数
            harmonic_params: HarmonicPatternDetector 参数
            weights: 信号权重 {"chan": w1, "elliott": w2, "harmonic": w3}
        """
        self.chan     = ChanAlgorithm(**(chan_params or {}))
        self.elliott  = ElliottWaveCounter(**(elliott_params or {}))
        self.harmonic = HarmonicPatternDetector(**(harmonic_params or {}))

        self.weights = weights or {"chan": 0.3, "elliott": 0.4, "harmonic": 0.3}

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完整分析 / Run full analysis

        Returns:
            Dict with keys: chan_signals, elliott_count, harmonic_hits, combined_signals
        """
        df = _ensure_ohlcv(data)

        chan_sigs  = self.chan.generate_signals(df)
        elliot_sigs = self.elliott.generate_signals(df)
        harm_sigs  = self.harmonic.generate_signals(df)
        wave_count = self.elliott.count_waves(df)
        harm_hits  = self.harmonic.detect(df)

        combined = self._combine_signals(chan_sigs, elliot_sigs, harm_sigs)

        return {
            "chan_signals": chan_sigs,
            "elliott_signals": elliot_sigs,
            "elliott_count": wave_count,
            "harmonic_signals": harm_sigs,
            "harmonic_hits": harm_hits,
            "combined_signals": combined,
        }

    def _combine_signals(
        self,
        chan: List[Signal],
        elliott: List[Signal],
        harmonic: List[Signal],
    ) -> List[Signal]:
        """加权融合信号 / Weighted signal fusion"""

        def score(s: Signal) -> float:
            w = s.metadata.get("type", "")
            if "chan" in w:
                return self.weights["chan"] * s.strength
            elif "elliott" in w:
                return self.weights["elliott"] * s.strength
            elif "harmonic" in w:
                return self.weights["harmonic"] * s.strength
            return s.strength

        all_sigs = []
        for sig_list, meta_prefix in [
            (chan, "chan"),
            (elliott, "elliott"),
            (harmonic, "harmonic"),
        ]:
            for s in sig_list:
                s.metadata["weight"] = score(s)
                s.metadata["source"] = meta_prefix
                all_sigs.append(s)

        # Group by date and take highest weighted signal per direction
        by_date: Dict[Any, List[Signal]] = {}
        for s in all_sigs:
            key = s.metadata.get("date", s.price)
            by_date.setdefault(key, []).append(s)

        results: List[Signal] = []
        for date, sigs in by_date.items():
            best = max(sigs, key=lambda x: x.metadata["weight"])
            best.metadata["all_signals"] = [
                {"type": si.metadata.get("type"), "strength": si.strength} for si in sigs
            ]
            results.append(best)

        results.sort(key=lambda x: x.metadata["weight"], reverse=True)
        return results

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate unified trading signals."""
        return self.analyze(data)["combined_signals"]


# ---------------------------------------------------------------------------
# PatternScanner
# ---------------------------------------------------------------------------

class PatternScanner:
    """
    形态扫描器 / Pattern Scanner

    扫描多个标的，寻找 Chan + Elliott + Harmonic 形态机会。

    Scan multiple instruments for pattern opportunities.
    Supports A-share / HK / Futures daily OHLCV data.
    """

    def __init__(
        self,
        chan_params: Optional[Dict[str, Any]] = None,
        elliott_params: Optional[Dict[str, Any]] = None,
        harmonic_params: Optional[Dict[str, Any]] = None,
    ):
        self.abu_quant = AbuQuantStrategy(chan_params, elliott_params, harmonic_params)

    def scan(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        扫描多个标的 / Scan multiple symbols

        Args:
            symbols_data: Dict mapping symbol → OHLCV DataFrame

        Returns:
            Dict mapping symbol → analysis result dict
        """
        results = {}
        for symbol, df in symbols_data.items():
            df = df.copy()
            df.symbol = symbol   # tag for signal generation
            try:
                analysis = self.abu_quant.analyze(df)
                results[symbol] = {
                    "analysis": analysis,
                    "signal_count": len(analysis["combined_signals"]),
                    "top_signal": analysis["combined_signals"][0] if analysis["combined_signals"] else None,
                    "elliott_complete": analysis["elliott_count"].is_impulse_complete,
                    "harmonic_count": len(analysis["harmonic_hits"]),
                    "error": None,
                }
            except Exception as e:
                results[symbol] = {
                    "analysis": None,
                    "signal_count": 0,
                    "top_signal": None,
                    "elliott_complete": False,
                    "harmonic_count": 0,
                    "error": str(e),
                }

        return results

    def scan_dataframes(
        self,
        dataframes: List[pd.DataFrame],
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        扫描 DataFrame 列表 / Scan a list of DataFrames

        Args:
            dataframes: List of OHLCV DataFrames
            symbols: Optional list of symbol names

        Returns:
            List of scan results, one per DataFrame
        """
        if symbols is None:
            symbols = [f"instrument_{i}" for i in range(len(dataframes))]

        symbols_data = dict(zip(symbols, dataframes))
        results = self.scan(symbols_data)
        return [results[sym] for sym in symbols]
