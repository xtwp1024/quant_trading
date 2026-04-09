"""
Jesse Strategies — BaseStrategy Adapters
量化之神交易策略模块 — Jesse 7大策略 BaseStrategy 适配器

Absorbed from: D:/Hive/Data/trading_repos/jesse-strategies/
Original repo: https://github.com/gabrielweich/jesse-strategies

7 Battle-Tested 策略:
1. DaveLandry       — 20日通道均值回归
2. Donchian         — Donchian通道突破 + ATR止损
3. IFR2             — RSI2均值回归 + Ichimoku + Hilbert过滤
4. MMM              — MACD-MA组合 (3/30 SMA)
5. RSI4             — RSI4均值回归 (Larry Connors)
6. SimpleBollinger  — 布林带突破 + Ichimoku过滤
7. MongeYokohama   — 自定义EMA趋势跟踪

每个策略通过 BaseStrategy 适配器封装:
  - pd.DataFrame → np.ndarray 转换
  - SignalDict → List[Signal] 转换
  - 注册到 StrategyLoader (名称: dave_landry, donchian, ifr2, mmm, rsi4, simple_bollinger, monge_yokohama)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TypedDict

from quant_trading.signal import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


# =============================================================================
# Indicator Library — Pure NumPy (from Jesse, no Talib dependency)
# =============================================================================

def _sma(candles: np.ndarray, period: int, source_col: int = 3) -> np.ndarray:
    close = candles[:, source_col]
    out = np.full(close.shape, np.nan)
    for i in range(period - 1, len(close)):
        out[i] = close[i - period + 1:i + 1].mean()
    return out


def _ema(candles: np.ndarray, period: int, source_col: int = 3) -> np.ndarray:
    close = candles[:, source_col]
    out = np.full(close.shape, np.nan)
    k = 2.0 / (period + 1)
    if period <= len(close):
        out[period - 1] = close[:period].mean()
    for i in range(period, len(close)):
        out[i] = close[i] * k + out[i - 1] * (1 - k)
    return out


def _atr(candles: np.ndarray, period: int = 14) -> np.ndarray:
    high = candles[:, 1]
    low = candles[:, 2]
    close = candles[:, 3]
    tr = np.zeros(len(candles))
    tr[0] = high[0] - low[0]
    for i in range(1, len(candles)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    out = np.full(len(candles), np.nan)
    for i in range(period - 1, len(tr)):
        out[i] = tr[i - period + 1:i + 1].mean()
    return out


def _donchian(candles: np.ndarray, period: int = 20) -> tuple[np.ndarray, np.ndarray]:
    high = candles[:, 1]
    low = candles[:, 2]
    upper = np.full(len(candles), np.nan)
    lower = np.full(len(candles), np.nan)
    for i in range(period, len(candles)):
        upper[i] = high[i - period:i].max()
        lower[i] = low[i - period:i].min()
    return upper, lower


def _bollinger_bands(candles: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hl2 = (candles[:, 1] + candles[:, 2]) / 2.0
    mid_hl2 = np.full(hl2.shape, np.nan)
    for i in range(period - 1, len(hl2)):
        mid_hl2[i] = hl2[i - period + 1:i + 1].mean()
    std = np.full(hl2.shape, np.nan)
    for i in range(period - 1, len(hl2)):
        std[i] = hl2[i - period + 1:i + 1].std()
    upper = mid_hl2 + num_std * std
    lower = mid_hl2 - num_std * std
    return upper, mid_hl2, lower


def _rsi(candles: np.ndarray, period: int = 2) -> np.ndarray:
    close = candles[:, 3]
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(candles), np.nan)
    avg_loss = np.full(len(candles), np.nan)
    if period <= len(candles):
        avg_gain[period - 1] = gain[1:period + 1].mean()
        avg_loss[period - 1] = loss[1:period + 1].mean()
    for i in range(period, len(candles)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def _ichimoku_cloud(
    candles: np.ndarray,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, np.ndarray]:
    high = candles[:, 1]
    low = candles[:, 2]
    tenkan = np.full(len(candles), np.nan)
    for i in range(conversion_period - 1, len(candles)):
        tenkan[i] = (high[i - conversion_period + 1:i + 1].max() +
                     low[i - conversion_period + 1:i + 1].min()) / 2.0
    kijun = np.full(len(candles), np.nan)
    for i in range(base_period - 1, len(candles)):
        kijun[i] = (high[i - base_period + 1:i + 1].max() +
                    low[i - base_period + 1:i + 1].min()) / 2.0
    span_a = np.full(len(candles), np.nan)
    for i in range(base_period - 1 + displacement, len(candles)):
        past_tenkan = tenkan[i - displacement]
        past_kijun = kijun[i - displacement]
        if not (np.isnan(past_tenkan) or np.isnan(past_kijun)):
            span_a[i] = (past_tenkan + past_kijun) / 2.0
    span_b = np.full(len(candles), np.nan)
    for i in range(span_b_period - 1 + displacement, len(candles)):
        span_b[i] = (high[i - span_b_period - displacement + 1:i - displacement + 1].max() +
                     low[i - span_b_period - displacement + 1:i - displacement + 1].min()) / 2.0
    return {"span_a": span_a, "span_b": span_b, "tenkan_sen": tenkan, "kijun_sen": kijun}


def _ht_trendmode(candles: np.ndarray) -> np.ndarray:
    close = candles[:, 3]
    period = 20
    detrender = np.zeros(len(candles))
    Q1 = np.zeros(len(candles))
    I1 = np.zeros(len(candles))
    for i in range(period, len(candles) - 1):
        detrender[i] = (close[i] - close[i - 2]) / 2.0
        Q1[i] = (detrender[i] - detrender[i - 2]) / 2.0
        I1[i] = close[i - int(period / 4)] - close[i - 3 * int(period / 4)]
        if I1[i] == 0:
            I1[i] = 1e-10
    trend = np.zeros(len(candles))
    for i in range(period + 1, len(candles) - 1):
        if I1[i - 1] != 0 and Q1[i - 1] != 0:
            trend[i] = 1 if (I1[i] * I1[i - 1] + Q1[i] * Q1[i - 1]) > 0 else 0
        else:
            trend[i] = 0
    return trend


# =============================================================================
# Signal Dicts (internal Jesse format)
# =============================================================================

class _SignalDict(TypedDict):
    entries: np.ndarray
    exits: np.ndarray
    stoploss: np.ndarray
    position: np.ndarray


# =============================================================================
# BaseStrategy Adapters
# =============================================================================

def _df_to_candles(df: pd.DataFrame) -> np.ndarray:
    """Convert pd.DataFrame to np.ndarray (N, 5) for Jesse strategies."""
    cols = ["open", "high", "low", "close", "volume"]
    if all(c in df.columns for c in cols):
        arr = df[cols].values
    elif all(c in df.columns for c in ["open", "high", "low", "close"]):
        arr = df[["open", "high", "low", "close"]].values
        vol = df.get("volume", pd.Series(np.ones(len(df)))).values.reshape(-1, 1)
        arr = np.hstack([arr, vol])
    else:
        raise ValueError(f"DataFrame must have OHLCV columns, got: {df.columns.tolist()}")
    return arr


def _sigdict_to_signals(
    sigdict: _SignalDict,
    df: pd.DataFrame,
    strategy_name: str,
) -> List[Signal]:
    """Convert Jesse SignalDict to BaseStrategy List[Signal]."""
    signals = []
    timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
    prices = df["close"].values

    for i in range(len(sigdict["entries"])):
        if sigdict["entries"][i] == 1:
            sl = sigdict["stoploss"][i]
            signals.append(Signal(
                type=SignalType.ENTRY,
                symbol=df.index[i] if df.index[i] is not None else "",
                timestamp=int(timestamps[i]),
                price=float(prices[i]),
                strength=1.0,
                reason=strategy_name,
                stop_loss=float(sl) if sl > 0 else None,
                take_profit=None,
                metadata={},
                direction=SignalDirection.LONG,
            ))
        elif sigdict["exits"][i] in (1, 2):
            exit_type = SignalType.EXIT if sigdict["exits"][i] == 1 else SignalType.STOP_LOSS
            signals.append(Signal(
                type=exit_type,
                symbol=df.index[i] if df.index[i] is not None else "",
                timestamp=int(timestamps[i]),
                price=float(prices[i]),
                strength=1.0,
                reason=strategy_name,
                stop_loss=None,
                take_profit=None,
                metadata={},
                direction=SignalDirection.EXIT,
            ))
    return signals


# ---- Params dataclasses ----

@dataclass
class DaveLandryParams(StrategyParams):
    sma_period: int = 20


@dataclass
class DonchianParams(StrategyParams):
    donchian_period: int = 20
    atr_period: int = 14
    ma_period: int = 200


@dataclass
class IFR2Params(StrategyParams):
    rsi_period: int = 2
    ichimoku_conversion: int = 20
    ichimoku_base: int = 30
    ichimoku_lagging: int = 120
    ichimoku_displacement: int = 60


@dataclass
class MMMParams(StrategyParams):
    fast_period: int = 3
    slow_period: int = 30


@dataclass
class RSI4Params(StrategyParams):
    rsi_period: int = 4
    sma_period: int = 200
    rsi_os: float = 25.0
    rsi_exit: float = 55.0


@dataclass
class SimpleBollingerParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0


@dataclass
class MongeYokohamaParams(StrategyParams):
    ema_low_period: int = 5
    ema_high_period: int = 7
    entry_discount: float = 1.02


# =============================================================================
# Strategy 1: DaveLandry — 20-day channel mean reversion
# =============================================================================

class DaveLandryStrategy(BaseStrategy):
    name = "dave_landry"
    params: DaveLandryParams

    def __init__(self, symbol: str = "", params: Optional[DaveLandryParams] = None) -> None:
        p = params or DaveLandryParams()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]
        high = candles[:, 1]
        low = candles[:, 2]

        trend_ma = _sma(candles, self.params.sma_period)
        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False
        entry_price = 0.0

        for i in range(self.params.sma_period + 2, n):
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_pos:
                    exits[i] = 1
                    in_pos = False
                continue

            if not in_pos:
                if low[i] < close[i - 1] and low[i] < close[i - 2]:
                    entries[i] = 1
                    in_pos = True
                    entry_price = close[i]
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = entry_price - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] > high[i - 1] and close[i] > high[i - 2]:
                    exits[i] = 1
                    in_pos = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and low[i] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 2: Donchian — Donchian channel breakout + ATR stop
# =============================================================================

class DonchianStrategy(BaseStrategy):
    name = "donchian"
    params: DonchianParams

    def __init__(self, symbol: str = "", params: Optional[DonchianParams] = None) -> None:
        p = params or DonchianParams()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        upper, lower = _donchian(candles, self.params.donchian_period)
        trend_ma = _sma(candles, self.params.ma_period)
        atr_vals = _atr(candles, self.params.atr_period)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False
        entry_price = 0.0

        for i in range(max(self.params.donchian_period, self.params.ma_period) + 1, n):
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_pos:
                    exits[i] = 1
                    in_pos = False
                continue

            if not in_pos:
                if close[i] > upper[i]:
                    entries[i] = 1
                    in_pos = True
                    entry_price = close[i]
                    stoploss[i] = entry_price - 2 * atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            else:
                if close[i] < lower[i]:
                    exits[i] = 1
                    in_pos = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                else:
                    if not np.isnan(atr_vals[i]):
                        new_sl = close[i] - 2 * atr_vals[i]
                        stoploss[i] = max(stoploss[i - 1], new_sl) if stoploss[i - 1] > 0 else new_sl
                    else:
                        stoploss[i] = stoploss[i - 1]
                    if stoploss[i - 1] > 0 and low[i] < stoploss[i - 1]:
                        exits[i] = 2
                        in_pos = False
                        entry_price = 0.0
                        stoploss[i] = 0.0

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 3: IFR2 — RSI2 mean reversion + Ichimoku + Hilbert filter
# =============================================================================

class IFR2Strategy(BaseStrategy):
    name = "ifr2"
    params: IFR2Params

    def __init__(self, symbol: str = "", params: Optional[IFR2Params] = None) -> None:
        p = params or IFR2Params()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]
        high = candles[:, 1]

        rsi_vals = _rsi(candles, self.params.rsi_period)
        ichimoku = _ichimoku_cloud(
            candles,
            conversion_period=self.params.ichimoku_conversion,
            base_period=self.params.ichimoku_base,
            span_b_period=self.params.ichimoku_lagging,
            displacement=self.params.ichimoku_displacement,
        )
        span_a = ichimoku["span_a"]
        span_b = ichimoku["span_b"]
        ht_mode = _ht_trendmode(candles)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False

        warmup = max(self.params.ichimoku_conversion, self.params.ichimoku_base,
                     self.params.ichimoku_lagging) + self.params.ichimoku_displacement + 5

        for i in range(warmup, n):
            cloud_ok = (not np.isnan(span_a[i]) and not np.isnan(span_b[i]) and
                        close[i] > span_a[i] and close[i] > span_b[i])
            trend_ok = ht_mode[i] == 1

            if not in_pos:
                if cloud_ok and trend_ok and rsi_vals[i] < 10:
                    entries[i] = 1
                    in_pos = True
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] > high[i - 1] and close[i] > high[i - 2]:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 4: MMM — MACD-MA combination (3/30 SMA)
# =============================================================================

class MMMStrategy(BaseStrategy):
    name = "mmm"
    params: MMMParams

    def __init__(self, symbol: str = "", params: Optional[MMMParams] = None) -> None:
        p = params or MMMParams()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)

        ma_trend = _sma(candles, self.params.slow_period)
        ma_low = _sma(candles, self.params.fast_period, source_col=2)
        ma_high = _sma(candles, self.params.fast_period, source_col=1)

        close = candles[:, 3]
        low = candles[:, 2]

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False

        for i in range(self.params.slow_period + 1, n):
            if np.isnan(ma_trend[i]) or close[i] <= ma_trend[i]:
                if in_pos:
                    exits[i] = 1
                    in_pos = False
                continue

            if not in_pos:
                if close[i] < ma_low[i]:
                    entries[i] = 1
                    in_pos = True
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] > ma_high[i]:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 5: RSI4 — Larry Connors RSI4 mean reversion
# =============================================================================

class RSI4Strategy(BaseStrategy):
    name = "rsi4"
    params: RSI4Params

    def __init__(self, symbol: str = "", params: Optional[RSI4Params] = None) -> None:
        p = params or RSI4Params()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        rsi_vals = _rsi(candles, self.params.rsi_period)
        trend_ma = _sma(candles, self.params.sma_period)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False

        for i in range(self.params.sma_period + 1, n):
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_pos:
                    exits[i] = 1
                    in_pos = False
                continue

            if not in_pos:
                if rsi_vals[i] <= self.params.rsi_os:
                    entries[i] = 1
                    in_pos = True
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if rsi_vals[i] >= self.params.rsi_exit:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 6: SimpleBollinger — Bollinger breakout + Ichimoku filter
# =============================================================================

class SimpleBollingerStrategy(BaseStrategy):
    name = "simple_bollinger"
    params: SimpleBollingerParams

    def __init__(self, symbol: str = "", params: Optional[SimpleBollingerParams] = None) -> None:
        p = params or SimpleBollingerParams()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        bb_upper, bb_mid, bb_lower = _bollinger_bands(candles, self.params.bb_period, self.params.bb_std)
        ichimoku = _ichimoku_cloud(candles)
        span_a = ichimoku["span_a"]
        span_b = ichimoku["span_b"]

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False

        warmup = self.params.bb_period + 60

        for i in range(warmup, n):
            cloud_ok = (not np.isnan(span_a[i]) and not np.isnan(span_b[i]) and
                        close[i] > span_a[i] and close[i] > span_b[i])

            if not in_pos:
                if cloud_ok and close[i] > bb_upper[i]:
                    entries[i] = 1
                    in_pos = True
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] < bb_mid[i]:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Strategy 7: MongeYokohama — Custom EMA trend following
# =============================================================================

class MongeYokohamaStrategy(BaseStrategy):
    name = "monge_yokohama"
    params: MongeYokohamaParams

    def __init__(self, symbol: str = "", params: Optional[MongeYokohamaParams] = None) -> None:
        p = params or MongeYokohamaParams()
        super().__init__(symbol, p)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        candles = _df_to_candles(data)
        n = len(candles)
        close = candles[:, 3]

        ema_low = _ema(candles, self.params.ema_low_period, source_col=2)
        ema_high = _ema(candles, self.params.ema_high_period, source_col=1)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_pos = False
        entry_bar = -1

        warmup = max(self.params.ema_low_period, self.params.ema_high_period) + 2

        for i in range(warmup, n):
            if not in_pos:
                if close[i] < ema_low[i] / self.params.entry_discount:
                    entries[i] = 1
                    in_pos = True
                    entry_bar = i
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                bars_in = i - entry_bar
                if close[i] > ema_high[i]:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif bars_in >= 2 and close[i] > candles[entry_bar, 3]:
                    exits[i] = 1
                    in_pos = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_pos = False
                    stoploss[i] = 0.0
                else:
                    atr_val = _atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_pos else 0

        sigdict = _SignalDict(entries=entries, exits=exits, stoploss=stoploss, position=position)
        return _sigdict_to_signals(sigdict, data, self.name)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return 1.0


# =============================================================================
# Indicator re-exports (for direct use)
# =============================================================================

__all__ = [
    # Adapters
    "DaveLandryStrategy",
    "DonchianStrategy",
    "IFR2Strategy",
    "MMMStrategy",
    "RSI4Strategy",
    "SimpleBollingerStrategy",
    "MongeYokohamaStrategy",
    # Params
    "DaveLandryParams",
    "DonchianParams",
    "IFR2Params",
    "MMMParams",
    "RSI4Params",
    "SimpleBollingerParams",
    "MongeYokohamaParams",
    # Indicators
    "sma", "ema", "atr", "donchian", "bollinger_bands", "rsi",
    "ichimoku_cloud", "ht_trendmode",
]
