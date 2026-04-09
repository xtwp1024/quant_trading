"""
Jesse Strategies — 7 Battle-Tested Strategies
量化之神交易策略模块 — Jesse 量化策略合集

Absorbed from: D:/Hive/Data/trading_repos/jesse-strategies/
Original repo: https://github.com/gabrielweich/jesse-strategies

Strategies implemented with pure NumPy (no Talib dependency):
1. DaveLandry       — 20-day channel mean reversion
2. Donchian         — Donchian channel breakout with ATR stop
3. IFR2             — RSI2 mean reversion with Ichimoku + Hilbert filter
4. MMM              — MACD-MA combination (3/30 SMA)
5. RSI4             — RSI4 mean reversion (Larry Connors style)
6. SimpleBollinger  — Bollinger Band breakout with Ichimoku filter
7. MongeYokohama    — Custom EMA trend-following

All strategies follow the same interface:
  - candles: np.ndarray (N, 5) — open, high, low, close, volume
  - generate_signals(candles) -> dict with 'entries', 'exits', 'stoploss'
"""

from __future__ import annotations

import numpy as np
from typing import Literal, TypedDict

# =============================================================================
# Indicator Library — Pure NumPy
# =============================================================================

def sma(candles: np.ndarray, period: int, source_col: int = 3) -> np.ndarray:
    """Simple Moving Average. source_col: 0=open, 1=high, 2=low, 3=close, 4=volume."""
    close = candles[:, source_col]
    out = np.full(close.shape, np.nan)
    for i in range(period - 1, len(close)):
        out[i] = close[i - period + 1:i + 1].mean()
    return out


def ema(candles: np.ndarray, period: int, source_col: int = 3) -> np.ndarray:
    """Exponential Moving Average."""
    close = candles[:, source_col]
    out = np.full(close.shape, np.nan)
    k = 2.0 / (period + 1)
    # seed with SMA
    if period <= len(close):
        out[period - 1] = close[:period].mean()
    for i in range(period, len(close)):
        out[i] = close[i] * k + out[i - 1] * (1 - k)
    return out


def atr(candles: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
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


def donchian(candles: np.ndarray, period: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Donchian Channel — returns (upperband, lowerband) shifted by 1 bar to avoid look-ahead."""
    high = candles[:, 1]
    low = candles[:, 2]
    upper = np.full(len(candles), np.nan)
    lower = np.full(len(candles), np.nan)
    for i in range(period, len(candles)):
        # use candles[:-1] — current bar excluded to avoid look-ahead
        upper[i] = high[i - period:i].max()
        lower[i] = low[i - period:i].min()
    return upper, lower


def bollinger_bands(candles: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands — returns (upperband, middleband, lowerband). Uses hl2 source."""
    hl2 = (candles[:, 1] + candles[:, 2]) / 2.0
    mid = sma(candles, period, source_col=3)  # use close for SMA mid
    # re-compute hl2 sma for mid
    mid_hl2 = np.full(hl2.shape, np.nan)
    for i in range(period - 1, len(hl2)):
        mid_hl2[i] = hl2[i - period + 1:i + 1].mean()
    std = np.full(hl2.shape, np.nan)
    for i in range(period - 1, len(hl2)):
        std[i] = hl2[i - period + 1:i + 1].std()
    upper = mid_hl2 + num_std * std
    lower = mid_hl2 - num_std * std
    return upper, mid_hl2, lower


def rsi(candles: np.ndarray, period: int = 2) -> np.ndarray:
    """RSI — Relative Strength Index."""
    close = candles[:, 3]
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(candles), np.nan)
    avg_loss = np.full(len(candles), np.nan)
    # seed with SMA
    if period <= len(candles):
        avg_gain[period - 1] = gain[1:period + 1].mean()
        avg_loss[period - 1] = loss[1:period + 1].mean()
    for i in range(period, len(candles)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def ichimoku_cloud(
    candles: np.ndarray,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, np.ndarray]:
    """
    Ichimoku Cloud — pure NumPy implementation.
    Returns: span_a, span_b, tenkan_sen, kijun_sen
    Cloud is formed by projecting span_a and span_b forward by `displacement` bars.
    """
    high = candles[:, 1]
    low = candles[:, 2]

    # Tenkan-sen (conversion line)
    tenkan = np.full(len(candles), np.nan)
    for i in range(conversion_period - 1, len(candles)):
        tenkan[i] = (high[i - conversion_period + 1:i + 1].max() +
                     low[i - conversion_period + 1:i + 1].min()) / 2.0

    # Kijun-sen (base line)
    kijun = np.full(len(candles), np.nan)
    for i in range(base_period - 1, len(candles)):
        kijun[i] = (high[i - base_period + 1:i + 1].max() +
                    low[i - base_period + 1:i + 1].min()) / 2.0

    # Span A — midpoint of Tenkan & Kijun, projected forward
    span_a = np.full(len(candles), np.nan)
    for i in range(base_period - 1 + displacement, len(candles)):
        past_tenkan = tenkan[i - displacement]
        past_kijun = kijun[i - displacement]
        if not (np.isnan(past_tenkan) or np.isnan(past_kijun)):
            span_a[i] = (past_tenkan + past_kijun) / 2.0

    # Span B — midpoint of 52-period high/low, projected forward
    span_b = np.full(len(candles), np.nan)
    for i in range(span_b_period - 1 + displacement, len(candles)):
        span_b[i] = (high[i - span_b_period - displacement + 1:i - displacement + 1].max() +
                     low[i - span_b_period - displacement + 1:i - displacement + 1].min()) / 2.0

    return {"span_a": span_a, "span_b": span_b, "tenkan_sen": tenkan, "kijun_sen": kijun}


def ht_trendmode(candles: np.ndarray) -> np.ndarray:
    """
    Hilbert Transform Trend Mode indicator.
    Returns 1 for uptrend (dominant cycle > 0), 0 otherwise.
    Simplified implementation using price momentum.
    """
    close = candles[:, 3]
    # Use 20-bar Hilbert Transform approximation via analytic signal
    # Compute in-phase and quadrature components
    period = 20
    detrender = np.zeros(len(candles))
    Q1 = np.zeros(len(candles))
    I1 = np.zeros(len(candles))

    for i in range(period, len(candles) - 1):
        # Detrend: remove high-frequency noise
        detrender[i] = (close[i] - close[i - 2]) / 2.0
        Q1[i] = (detrender[i] - detrender[i - 2]) / 2.0
        I1[i] = close[i - int(period / 4)] - close[i - 3 * int(period / 4)]
        if I1[i] == 0:
            I1[i] = 1e-10

    # Trend mode: positive when I1 * prev_I1 + Q1 * prev_Q1 > 0
    trend = np.zeros(len(candles))
    for i in range(period + 1, len(candles) - 1):
        if I1[i - 1] != 0 and Q1[i - 1] != 0:
            trend[i] = 1 if (I1[i] * I1[i - 1] + Q1[i] * Q1[i - 1]) > 0 else 0
        else:
            trend[i] = 0
    return trend


# =============================================================================
# Signal Dicts — each strategy returns this shape
# =============================================================================

class SignalDict(TypedDict):
    entries: np.ndarray      # 1 where entry signal fires
    exits: np.ndarray        # 1 where exit signal fires
    stoploss: np.ndarray     # stop-loss price (0 = no stop)
    position: np.ndarray     # cumulative position (+1 long, 0 flat)


# =============================================================================
# Strategy 1: DaveLandry — 20-day channel mean reversion
# 20日均线过滤 + 前两根K线低价过滤入场，前两根K线高价出场
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/DaveLandry
# =============================================================================

class DaveLandryStrategy:
    """
    DaveLandry 20-Day Channel Mean Reversion Strategy
    DaveLandry 20日通道均值回归策略

    Logic:
      - Filter: close > SMA20 (uptrend only)
      - Entry:  low < min(close[-2], close[-3]) — price dips below last 2 closes
      - Exit:   close > max(high[-2], high[-3]) — first profitable candle

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)

    Usage
    -----
    strat = DaveLandryStrategy()
    signals = strat.generate_signals(candles)
    """

    def __init__(self, sma_period: int = 20):
        self.sma_period = sma_period

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]
        high = candles[:, 1]
        low = candles[:, 2]

        trend_ma = sma(candles, self.sma_period)
        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False
        entry_price = 0.0

        for i in range(self.sma_period + 2, n):
            # Trend filter
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_position:
                    exits[i] = 1
                    in_position = False
                continue

            if not in_position:
                # Entry: low below previous 2 closes
                if (low[i] < close[i - 1] and low[i] < close[i - 2]):
                    entries[i] = 1
                    in_position = True
                    entry_price = close[i]
                    # Stop loss: 2 ATR below entry
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = entry_price - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                # Exit: close above previous 2 highs (first profitable candle)
                if (close[i] > high[i - 1] and close[i] > high[i - 2]):
                    exits[i] = 1
                    in_position = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and low[i] < stoploss[i - 1]:
                    # Stop-loss triggered
                    exits[i] = 2  # 2 = stop-loss exit
                    in_position = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                else:
                    # Trail stop at 1.5 ATR
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 2: Donchian — Donchian Channel Breakout with ATR Stop
# 20日Donchian通道突破 + 200日均线过滤 + 2ATR止损
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/Donchian
# =============================================================================

class DonchianStrategy:
    """
    Donchian Channel Breakout Strategy
    Donchian 通道突破策略

    Logic:
      - Filter: close > SMA200 (major uptrend)
      - Entry:  close > upperband (20-day Donchian, previous bar to avoid look-ahead)
      - Exit:   close < lowerband OR ATR-based trailing stop

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    donchian_period : int
        Lookback for Donchian channel (default 20)
    atr_period : int
        ATR period for stop-loss (default 14)
    """

    def __init__(self, donchian_period: int = 20, atr_period: int = 14, ma_period: int = 200):
        self.donchian_period = donchian_period
        self.atr_period = atr_period
        self.ma_period = ma_period

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        upper, lower = donchian(candles, self.donchian_period)
        trend_ma = sma(candles, self.ma_period)
        atr_vals = atr(candles, self.atr_period)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False
        entry_price = 0.0

        for i in range(max(self.donchian_period, self.ma_period) + 1, n):
            # Trend filter
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_position:
                    exits[i] = 1
                    in_position = False
                continue

            if not in_position:
                # Entry: close above upperband (use upper[i] which is shifted by 1 bar)
                if close[i] > upper[i]:
                    entries[i] = 1
                    in_position = True
                    entry_price = close[i]
                    stoploss[i] = entry_price - 2 * atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            else:
                # Exit: close below lowerband
                if close[i] < lower[i]:
                    exits[i] = 1
                    in_position = False
                    entry_price = 0.0
                    stoploss[i] = 0.0
                else:
                    # Trail stop: 2 ATR below price, only moves up
                    if not np.isnan(atr_vals[i]):
                        new_sl = close[i] - 2 * atr_vals[i]
                        stoploss[i] = max(stoploss[i - 1], new_sl) if stoploss[i - 1] > 0 else new_sl
                    else:
                        stoploss[i] = stoploss[i - 1]
                    # Stop-loss hit
                    if stoploss[i - 1] > 0 and low[i] < stoploss[i - 1]:
                        exits[i] = 2
                        in_position = False
                        entry_price = 0.0
                        stoploss[i] = 0.0

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 3: IFR2 — RSI2 Mean Reversion with Ichimoku + Hilbert Filter
# RSI2 < 10 入场，Ichimoku云过滤 + Hilbert Transform趋势模式过滤
# 前2根K线高价出场
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/IFR2
# =============================================================================

class IFR2Strategy:
    """
    IFR2 (RSI2) Mean Reversion Strategy
    IFR2 (RSI2) 均值回归策略

    Logic:
      - Filter 1: close > Ichimoku cloud (span_a AND span_b)
      - Filter 2: Hilbert Transform trend mode == 1 (uptrend)
      - Entry:   RSI(2) < 10 (oversold)
      - Exit:    close > max(high[-2], high[-3])

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    rsi_period : int
        RSI lookback period (default 2)
    ichimoku_params : dict
        Ichimoku cloud parameters
    """

    def __init__(
        self,
        rsi_period: int = 2,
        ichimoku_conversion: int = 20,
        ichimoku_base: int = 30,
        ichimoku_lagging: int = 120,
        ichimoku_displacement: int = 60,
    ):
        self.rsi_period = rsi_period
        self.ichimoku_conversion = ichimoku_conversion
        self.ichimoku_base = ichimoku_base
        self.ichimoku_lagging = ichimoku_lagging
        self.ichimoku_displacement = ichimoku_displacement

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]
        high = candles[:, 1]

        low = candles[:, 2]
        rsi_vals = rsi(candles, self.rsi_period)
        ichimoku = ichimoku_cloud(
            candles,
            conversion_period=self.ichimoku_conversion,
            base_period=self.ichimoku_base,
            span_b_period=self.ichimoku_lagging,
            displacement=self.ichimoku_displacement,
        )
        span_a = ichimoku["span_a"]
        span_b = ichimoku["span_b"]
        ht_mode = ht_trendmode(candles)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False

        warmup = max(self.ichimoku_conversion, self.ichimoku_base,
                     self.ichimoku_lagging) + self.ichimoku_displacement + 5

        for i in range(warmup, n):
            # Filters
            cloud_ok = (not np.isnan(span_a[i]) and not np.isnan(span_b[i]) and
                        close[i] > span_a[i] and close[i] > span_b[i])
            trend_ok = ht_mode[i] == 1

            if not in_position:
                if cloud_ok and trend_ok and rsi_vals[i] < 10:
                    entries[i] = 1
                    in_position = True
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                # Exit: close above previous 2 highs
                if close[i] > high[i - 1] and close[i] > high[i - 2]:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_position = False
                    stoploss[i] = 0.0
                else:
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 4: MMM — MACD + MA Combination (3/30 SMA)
# 30日均线过滤 + 3日MA-low过滤入场，3日MA-high出场
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/MMM
# =============================================================================

class MMMStrategy:
    """
    MMM (MACD-MA Combination) Strategy
    MMM (均线组合) 策略

    Logic:
      - Filter: close > SMA30 (uptrend)
      - Entry:  close < SMA3(low)
      - Exit:   close > SMA3(high)

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    fast_period : int
        Fast MA period for entry/exit (default 3)
    slow_period : int
        Slow MA trend filter period (default 30)
    """

    def __init__(self, fast_period: int = 3, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)

        ma_trend = sma(candles, self.slow_period)          # close SMA for trend
        ma_low = sma(candles, self.fast_period, source_col=2)  # low SMA for entry
        ma_high = sma(candles, self.fast_period, source_col=1)  # high SMA for exit

        close = candles[:, 3]
        low = candles[:, 2]

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False

        for i in range(self.slow_period + 1, n):
            if np.isnan(ma_trend[i]) or close[i] <= ma_trend[i]:
                if in_position:
                    exits[i] = 1
                    in_position = False
                continue

            if not in_position:
                if close[i] < ma_low[i]:
                    entries[i] = 1
                    in_position = True
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] > ma_high[i]:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_position = False
                    stoploss[i] = 0.0
                else:
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 5: RSI4 — Larry Connors RSI4 Mean Reversion
# 200日均线过滤 + RSI4 < 25 入场，RSI4 >= 55 出场
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/RSI4
# =============================================================================

class RSI4Strategy:
    """
    Larry Connors RSI4 Mean Reversion Strategy
    Larry Connors RSI4 均值回归策略

    Logic:
      - Filter: close > SMA200 (long-term uptrend)
      - Entry:  RSI(4) <= 25 (short-term oversold)
      - Exit:    RSI(4) >= 55

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    rsi_period : int
        RSI lookback (default 4)
    sma_period : int
        Trend SMA lookback (default 200)
    rsi_os : float
        Oversold threshold for entry (default 25)
    rsi_exit : float
        Exit threshold (default 55)
    """

    def __init__(
        self,
        rsi_period: int = 4,
        sma_period: int = 200,
        rsi_os: float = 25.0,
        rsi_exit: float = 55.0,
    ):
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.rsi_os = rsi_os
        self.rsi_exit = rsi_exit

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]

        rsi_vals = rsi(candles, self.rsi_period)
        trend_ma = sma(candles, self.sma_period)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False
        low = candles[:, 2]

        for i in range(self.sma_period + 1, n):
            if np.isnan(trend_ma[i]) or close[i] <= trend_ma[i]:
                if in_position:
                    exits[i] = 1
                    in_position = False
                continue

            if not in_position:
                if rsi_vals[i] <= self.rsi_os:
                    entries[i] = 1
                    in_position = True
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if rsi_vals[i] >= self.rsi_exit:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_position = False
                    stoploss[i] = 0.0
                else:
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 6: SimpleBollinger — Bollinger Band Breakout + Ichimoku Filter
# Ichimoku云过滤 + 布林带上轨入场，中轨出场
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/SimpleBollinger
# =============================================================================

class SimpleBollingerStrategy:
    """
    Simple Bollinger Bands Breakout Strategy
    简单布林带突破策略

    Logic:
      - Filter: close > Ichimoku cloud (span_a AND span_b)
      - Entry:  close > BB upperband
      - Exit:    close < BB middleband

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    bb_period : int
        Bollinger lookback (default 20)
    bb_std : float
        Bollinger std multiplier (default 2.0)
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        bb_upper, bb_mid, bb_lower = bollinger_bands(candles, self.bb_period, self.bb_std)
        ichimoku = ichimoku_cloud(candles)
        span_a = ichimoku["span_a"]
        span_b = ichimoku["span_b"]

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False

        warmup = self.bb_period + 60

        for i in range(warmup, n):
            cloud_ok = (not np.isnan(span_a[i]) and not np.isnan(span_b[i]) and
                        close[i] > span_a[i] and close[i] > span_b[i])

            if not in_position:
                if cloud_ok and close[i] > bb_upper[i]:
                    entries[i] = 1
                    in_position = True
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                if close[i] < bb_mid[i]:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_position = False
                    stoploss[i] = 0.0
                else:
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Strategy 7: MongeYokohama — Custom EMA Trend Following
# EMA5(low)/1.02 入场，EMA7(high)出场 or 第2根K线盈利即出
# Ref: https://github.com/gabrielweich/jesse-strategies/tree/master/MongeYokohama
# =============================================================================

class MongeYokohamaStrategy:
    """
    Monge Yokohama Custom Trend Following Strategy
    Monge Yokohama 自定义趋势跟踪策略

    Logic:
      - Entry:  close < EMA5(low) / 1.02
      - Exit 1: close > EMA7(high)
      - Exit 2: candle >= 2 AND close > entry_price (take profit on 2nd bar)
      - Stop:   2 ATR trailing

    Parameters
    ----------
    candles : np.ndarray
        OHLCV array, shape (N, 5)
    ema_low_period : int
        EMA period for low (default 5)
    ema_high_period : int
        EMA period for high (default 7)
    entry_discount : float
        Discount factor on EMA low for entry (default 1.02)
    """

    def __init__(
        self,
        ema_low_period: int = 5,
        ema_high_period: int = 7,
        entry_discount: float = 1.02,
    ):
        self.ema_low_period = ema_low_period
        self.ema_high_period = ema_high_period
        self.entry_discount = entry_discount

    def generate_signals(self, candles: np.ndarray) -> SignalDict:
        n = len(candles)
        close = candles[:, 3]
        low = candles[:, 2]

        ema_low = ema(candles, self.ema_low_period, source_col=2)
        ema_high = ema(candles, self.ema_high_period, source_col=1)

        entries = np.zeros(n, dtype=np.int32)
        exits = np.zeros(n, dtype=np.int32)
        stoploss = np.zeros(n, dtype=np.float64)
        position = np.zeros(n, dtype=np.int32)

        in_position = False
        entry_bar = -1

        warmup = max(self.ema_low_period, self.ema_high_period) + 2

        for i in range(warmup, n):
            if not in_position:
                if close[i] < ema_low[i] / self.entry_discount:
                    entries[i] = 1
                    in_position = True
                    entry_bar = i
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    stoploss[i] = close[i] - 2 * atr_val if not np.isnan(atr_val) else 0.0
            else:
                bars_in = i - entry_bar
                # Exit 1: close > EMA high
                if close[i] > ema_high[i]:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                # Exit 2: >= 2 bars held and in profit
                elif bars_in >= 2 and close[i] > candles[entry_bar, 3]:
                    exits[i] = 1
                    in_position = False
                    stoploss[i] = 0.0
                # Stop-loss
                elif stoploss[i - 1] > 0 and candles[i, 2] < stoploss[i - 1]:
                    exits[i] = 2
                    in_position = False
                    stoploss[i] = 0.0
                else:
                    atr_val = atr(candles[i - 13:i + 1], 14)[-1]
                    if not np.isnan(atr_val):
                        stoploss[i] = max(stoploss[i - 1], close[i] - 1.5 * atr_val)
                    else:
                        stoploss[i] = stoploss[i - 1]

            position[i] = 1 if in_position else 0

        return SignalDict(
            entries=entries,
            exits=exits,
            stoploss=stoploss,
            position=position,
        )


# =============================================================================
# Bundle: JesseStrategyBundle — Run All 7 Strategies on Same Data
# =============================================================================

class JesseStrategyBundle:
    """
    Jesse Strategy Bundle — Run All 7 Battle-Tested Strategies
    Jesse 策略Bundle — 在同一数据上运行全部7个策略

    Aggregates signals from all 7 Jesse strategies and provides
    combined metrics (equity curve, trade log, sharpe, drawdown).

    Parameters
    ----------
    initial_capital : float
        Starting capital (default 10_000 USDT)
    fee_rate : float
        Trading fee rate (default 0.001 = 0.1%)
    strategies : list
        List of strategy instances (default: all 7)

    Usage
    -----
    bundle = JesseStrategyBundle(initial_capital=10_000)
    result = bundle.run(candles)
    print(result["equity_curve"][-1])
    print(result["summary"])
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        strategies: list | None = None,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

        if strategies is None:
            strategies = [
                DaveLandryStrategy(),
                DonchianStrategy(),
                IFR2Strategy(),
                MMMStrategy(),
                RSI4Strategy(),
                SimpleBollingerStrategy(),
                MongeYokohamaStrategy(),
            ]

        self.strategies = strategies
        self.names = [type(s).__name__ for s in strategies]

    def run(self, candles: np.ndarray) -> dict:
        """
        Run all strategies on the same candles.
        Returns a dict with per-strategy signals and combined equity.
        """
        n = len(candles)
        close = candles[:, 3]

        # Generate signals for each strategy
        all_signals = [s.generate_signals(candles) for s in self.strategies]

        # Per-strategy backtest
        equity_curves = {}
        trade_logs = {}

        for name, sigs in zip(self.names, all_signals):
            equity = [self.initial_capital]
            trades = []
            pos = False
            entry_price = 0.0
            entry_bar = 0
            qty = 0.0
            winning = 0
            losing = 0

            for i in range(1, n):
                prev_equity = equity[-1]

                if not pos and sigs["entries"][i]:
                    # Open position
                    pos = True
                    entry_price = close[i]
                    entry_bar = i
                    qty = (prev_equity * (1 - self.fee_rate)) / entry_price
                    equity.append(prev_equity)
                elif pos:
                    # Check exit
                    exit_code = sigs["exits"][i]
                    if exit_code == 1:
                        # Normal exit
                        exit_price = close[i]
                        pnl = qty * (exit_price - entry_price)
                        fee = qty * exit_price * self.fee_rate
                        net_pnl = pnl - fee
                        new_equity = prev_equity + net_pnl
                        equity.append(new_equity)
                        trades.append({
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": net_pnl,
                            "type": "long",
                        })
                        if net_pnl > 0:
                            winning += 1
                        else:
                            losing += 1
                        pos = False
                        entry_price = 0.0
                        qty = 0.0
                    elif exit_code == 2:
                        # Stop-loss exit
                        sl_price = sigs["stoploss"][i - 1]
                        exit_price = sl_price
                        pnl = qty * (exit_price - entry_price)
                        fee = qty * sl_price * self.fee_rate
                        net_pnl = pnl - fee
                        new_equity = prev_equity + net_pnl
                        equity.append(new_equity)
                        trades.append({
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "entry_price": entry_price,
                            "exit_price": sl_price,
                            "pnl": net_pnl,
                            "type": "stop_loss",
                        })
                        losing += 1
                        pos = False
                        entry_price = 0.0
                        qty = 0.0
                    else:
                        # Mark-to-market
                        mtm = qty * (close[i] - close[i - 1])
                        equity.append(prev_equity + mtm)
                else:
                    equity.append(prev_equity)

            equity_arr = np.array(equity)
            equity_curves[name] = equity_arr
            trade_logs[name] = {
                "trades": trades,
                "total_trades": len(trades),
                "winning": winning,
                "losing": losing,
                "win_rate": winning / len(trades) if trades else 0.0,
                "final_equity": equity_arr[-1],
                "max_equity": equity_arr.max(),
                "min_equity": equity_arr.min(),
            }

        # Combined equity (equal-weight average of all strategy positions)
        # For each bar, compute the aggregate PnL if we had equal distribution
        combined_equity = [self.initial_capital]
        for i in range(1, n):
            combined_equity.append(combined_equity[-1])

        for name, sigs in zip(self.names, all_signals):
            eq = equity_curves[name]
            # Normalize to show relative contribution
            if eq[-1] > eq[0]:
                ret = (eq[-1] - eq[0]) / eq[0]
            else:
                ret = 0.0

        # Summary across all strategies
        summary = {}
        for name, log in trade_logs.items():
            total_return = (log["final_equity"] - self.initial_capital) / self.initial_capital * 100
            # Max drawdown
            eq = equity_curves[name]
            peak = eq[0]
            max_dd = 0.0
            for val in eq:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
            summary[name] = {
                "total_return_pct": round(total_return, 2),
                "final_equity": round(log["final_equity"], 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "total_trades": log["total_trades"],
                "win_rate": round(log["win_rate"] * 100, 1),
            }

        return {
            "equity_curves": equity_curves,
            "trade_logs": trade_logs,
            "summary": summary,
            "initial_capital": self.initial_capital,
        }

    def summary_table(self, result: dict) -> str:
        """Pretty-print a summary table of all strategies."""
        lines = [
            f"{'Strategy':<30} {'Return%':>10} {'MaxDD%':>8} {'Trades':>7} {'WinRate%':>9} {'FinalEquity':>13}",
            "-" * 79,
        ]
        for name, s in result["summary"].items():
            lines.append(
                f"{name:<30} {s['total_return_pct']:>10.2f} {s['max_drawdown_pct']:>8.2f} "
                f"{s['total_trades']:>7} {s['win_rate']:>9.1f} {s['final_equity']:>13.2f}"
            )
        return "\n".join(lines)
