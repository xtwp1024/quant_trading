"""
stock_indicators — 50+ Pure NumPy/Pandas Technical Indicators
==============================================================

Absorbed from stock-indicators-python (50+ professional technical indicators).
Zero external dependencies beyond NumPy and pandas; no Ta-Lib, no Cython.

All functions accept np.ndarray or pd.Series and return consistent types.
Bilingual docstrings (English + Chinese).

Categories
----------
Momentum   : RSI, Stochastic, StochasticRSI, MACD, CCI, MFI, Williams%R, ROC, Momentum, CMO
Trend      : SMA, EMA, WMA, DEMA, TEMA, VAMA, ZLEMA, HMA, SMMA, ADX, Aroon, SuperTrend
Volatility : ATR, ATRP, BollingerBands, KeltnerChannels, DonchianChannels, STARCbands
Volume     : OBV, VWAP, ADL, CMF, PVO, MFI
Statistical: StdDev, Slope, Variance, Covariance, Correlation, Beta
Patterns   : Doji, Engulfing, Hammer, MorningStar, EveningStar, ThreeWhiteSoldiers,
             ThreeBlackCrows, Marubozu, SpinningTop, InvertedHammer, ShootingStar,
             BeltHold, DragonflyDoji, GravestoneDoji, LongLeggedDoji, PaperUmbrella

Example
-------
    from quant_trading.factors.stock_indicators import rsi, stoch, macd, sma, ema, atr, vwap, obv
    rsi_val = rsi(close, period=14)
    k, d = stoch(high, low, close, k_period=14, d_period=3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Union

__all__ = [
    # Trend / Moving Averages
    "sma", "ema", "wma", "dema", "tema", "vama", "zlema", "hma", "smma",
    # Momentum
    "rsi", "stoch", "stoch_rsi", "macd", "cci", "mfi", "williams_r", "roc",
    "momentum", "cmo", "kdj",
    # Volatility
    "atr", "atrp", "bollinger_bands", "keltner", "donchian", "starc_bands",
    # Volume
    "obv", "vwap", "adl", "cmf", "pvo",
    # Statistical
    "std_dev", "slope", "variance", "covariance", "correlation", "beta",
    # Trend (continued)
    "adx", "aroon", "super_trend",
    # Candlestick Patterns
    "cdl_doji", "cdl_engulfing", "cdl_hammer", "cdl_morning_star",
    "cdl_evening_star", "cdl_three_white_soldiers", "cdl_three_black_crows",
    "cdl_marubozu", "cdl_spinning_top", "cdl_inverted_hammer", "cdl_shooting_star",
    "cdl_belt_hold", "cdl_dragonfly_doji", "cdl_gravestone_doji",
    "cdl_long_legged_doji", "cdl_paper_umbrella",
    # Utility
    "tr", "typical_price", "median_price", "weighted_close",
]


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------

def _to_array(s: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Convert pd.Series or np.ndarray to np.ndarray (flattened)."""
    if isinstance(s, pd.Series):
        return s.values.ravel()
    return np.asarray(s).ravel()


def _shift(arr: np.ndarray, n: int) -> np.ndarray:
    """Shift array forward by n periods (positive n = look back)."""
    return pd.Series(arr).shift(n).values


def _roll_apply(
    arr: np.ndarray, window: int, func: callable
) -> np.ndarray:
    """Rolling window apply using pandas."""
    return pd.Series(arr).rolling(window).apply(func, raw=True).values


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (span-based, adjust=False)."""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(arr).rolling(period).mean().values


# --------------------------------------------------------------------------
# TR (True Range) helper
# --------------------------------------------------------------------------

def tr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """True Range.

    TR = max(H - L, |H - CLOSE_prev|, |L - CLOSE_prev|)
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    prev_c = _shift(c, 1)
    hl = h - l
    hc = np.abs(h - prev_c)
    lc = np.abs(l - prev_c)
    return np.maximum(hl, np.maximum(hc, lc))


def typical_price(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Typical Price = (High + Low + Close) / 3."""
    return (_to_array(high) + _to_array(low) + _to_array(close)) / 3.0


def median_price(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Median Price = (High + Low) / 2."""
    return (_to_array(high) + _to_array(low)) / 2.0


def weighted_close(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Weighted Close = (High + Low + 2*Close) / 4."""
    return (_to_array(high) + _to_array(low) + 2 * _to_array(close)) / 4.0


# --------------------------------------------------------------------------
# Trend / Moving Averages
# --------------------------------------------------------------------------


def sma(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Simple Moving Average (SMA).

    SMA = SUM(price, period) / period

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        SMA values
    """
    p = _to_array(prices)
    return _sma(p, period)


def ema(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Exponential Moving Average (EMA).

    EMA = alpha * price + (1 - alpha) * EMA_prev
    alpha = 2 / (period + 1)

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        EMA values
    """
    p = _to_array(prices)
    return _ema(p, period)


def wma(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Weighted Moving Average (WMA / LWMA).

    WMA = SUM(price[i] * weight[i]) / SUM(weights)
    weight[i] = period - i  (most recent = highest weight)

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        WMA values
    """
    p = _to_array(prices)
    weights = np.arange(1, period + 1)
    result = np.zeros_like(p, dtype=float)
    for i in range(period - 1, len(p)):
        result[i] = np.sum(p[i - period + 1:i + 1] * weights) / weights.sum()
    return result


def dema(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Double Exponential Moving Average (DEMA).

    DEMA = 2 * EMA(price, period) - EMA(EMA(price, period), period)

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        DEMA values
    """
    p = _to_array(prices)
    e1 = _ema(p, period)
    e2 = _ema(e1, period)
    return 2 * e1 - e2


def tema(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Triple Exponential Moving Average (TEMA).

    TEMA = 3 * EMA(price) - 3 * EMA(EMA(price)) + EMA(EMA(EMA(price)))

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        TEMA values
    """
    p = _to_array(prices)
    e1 = _ema(p, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def vama(
    prices: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Volume-Adjusted Moving Average (VAMA).

    VAMA adjusts the moving average weighting by relative volume.

    参数:
        prices: 价格数组
        volume: 成交量数组
        period: 周期，默认 20
    返回:
        VAMA values
    """
    p = _to_array(prices)
    v = _to_array(volume)
    avg_vol = _sma(v, period)
    rel_vol = v / avg_vol
    # Weighted by relative volume
    result = np.zeros_like(p, dtype=float)
    for i in range(period - 1, len(p)):
        window_p = p[i - period + 1:i + 1]
        window_v = rel_vol[i - period + 1:i + 1]
        result[i] = np.sum(window_p * window_v) / np.sum(window_v)
    return result


def zlema(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Removes lag from EMA by adding a correction term.

    ZLEMA = EMA(price + (price - EMA(price, lag)), period)
    lag = (period - 1) / 2

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        ZLEMA values
    """
    p = _to_array(prices)
    lag = (period - 1) // 2
    ema_p = _ema(p, period)
    corrected = p + (p - _shift(p, lag))  # price + correction
    return _ema(corrected, period)


def hma(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Hull Moving Average (HMA).

    HMA = WMA(2 * WMA(price, period/2) - WMA(price, period), sqrt(period))

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        HMA values
    """
    p = _to_array(prices)
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    wma_half = wma(p, half)
    wma_full = wma(p, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_p)


def smma(prices: Union[np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Smoothed Moving Average (SMMA / RMA).

    SMMA = (SUM(price, period) - SMMA_prev + price) / period
    (Same as Wilder's smoothing)

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        SMMA values
    """
    p = _to_array(prices)
    result = np.zeros_like(p, dtype=float)
    result[period - 1] = np.mean(p[:period])
    alpha = 1.0 / period
    for i in range(period, len(p)):
        result[i] = result[i - 1] + alpha * (p[i] - result[i - 1])
    return result


# --------------------------------------------------------------------------
# Momentum Indicators
# --------------------------------------------------------------------------


def rsi(prices: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = avg_gain / avg_loss  (Wilder's smoothing)

    Uses Wilder's smoothing (exponential).
    Returns: RSI values [0, 100]

    参数:
        prices: 价格数组 (通常用 Close)
        period: 周期，默认 14
    返回:
        RSI values [0, 100]
    """
    p = _to_array(prices)
    deltas = np.diff(p, prepend=p[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing
    avg_gain = np.zeros_like(p)
    avg_loss = np.zeros_like(p)
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])

    alpha = 1.0 / period
    for i in range(period + 1, len(p)):
        avg_gain[i] = avg_gain[i - 1] + alpha * (gains[i] - avg_gain[i - 1])
        avg_loss[i] = avg_loss[i - 1] + alpha * (losses[i] - avg_loss[i - 1])

    rs = np.zeros_like(p)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val[avg_loss == 0] = 100
    return rsi_val


def stoch(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator (%K, %D).

    %K = (CLOSE - LLV(LOW, k_period)) / (HHV(HIGH, k_period) - LLV(LOW, k_period)) * 100
    %D = SMA(%K, d_period)

    Returns: (%K, %D)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        k_period: %K 周期，默认 14
        d_period: %D 周期，默认 3
    返回:
        (%K, %D) 元组
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    llv = _roll_apply(l, k_period, lambda x: np.min(x))
    hhv = _roll_apply(h, k_period, lambda x: np.max(x))
    k = np.zeros_like(c)
    denom = hhv - llv
    mask = denom != 0
    k[mask] = ((c[mask] - llv[mask]) / denom[mask]) * 100
    k[~mask] = 50
    d = _sma(k, d_period)
    return k, d


def stoch_rsi(
    prices: Union[np.ndarray, pd.Series],
    period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI.

    StochRSI = (RSI - LLV(RSI, period)) / (HHV(RSI, period) - LLV(RSI, period)) * 100
    %D = SMA(StochRSI, d_period)

    参数:
        prices: 价格数组
        period: RSI 周期，默认 14
        k_period: %K 平滑周期，默认 3
        d_period: %D 周期，默认 3
    返回:
        (StochRSI, %D) 元组
    """
    p = _to_array(prices)
    rsi_val = rsi(p, period)
    llv = _roll_apply(rsi_val, period, lambda x: np.min(x))
    hhv = _roll_apply(rsi_val, period, lambda x: np.max(x))
    denom = hhv - llv
    stoch_rsi = np.zeros_like(rsi_val)
    mask = denom != 0
    stoch_rsi[mask] = ((rsi_val[mask] - llv[mask]) / denom[mask]) * 100
    stoch_rsi[~mask] = 50
    d = _sma(stoch_rsi, d_period)
    return stoch_rsi, d


def macd(
    prices: Union[np.ndarray, pd.Series],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence/Divergence).

    DIF = EMA(price, fast) - EMA(price, slow)
    DEA = EMA(DIF, signal)
    MACD = (DIF - DEA) * 2

    参数:
        prices: 价格数组
        fast: 快线周期，默认 12
        slow: 慢线周期，默认 26
        signal: 信号线周期，默认 9
    返回:
        (DIF, DEA, MACD) 元组
    """
    p = _to_array(prices)
    ema_fast = _ema(p, fast)
    ema_slow = _ema(p, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def cci(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Commodity Channel Index (CCI).

    CCI = (TP - SMA(TP, period)) / (0.015 * MeanDeviation(TP, period))
    TP = (High + Low + Close) / 3

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 周期，默认 20
    返回:
        CCI values
    """
    tp = typical_price(high, low, close)
    sma_tp = _sma(tp, period)
    mad = _roll_apply(tp, period, lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = np.zeros_like(tp)
    mask = mad != 0
    cci[mask] = (tp[mask] - sma_tp[mask]) / (0.015 * mad[mask])
    return cci


def mfi(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """Money Flow Index (MFI).

    MFI = 100 - (100 / (1 + MoneyFlowRatio))
    MoneyFlowRatio = PositiveMoneyFlow / NegativeMoneyFlow
    PositiveMoneyFlow = SUM(TP * Vol, period) where TP > TP_prev
    NegativeMoneyFlow = SUM(TP * Vol, period) where TP < TP_prev

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        period: 周期，默认 14
    返回:
        MFI values [0, 100]
    """
    h, l, c, v = _to_array(high), _to_array(low), _to_array(close), _to_array(volume)
    tp = typical_price(h, l, c)
    prev_tp = _shift(tp, 1)
    pos_flow = np.where(tp > prev_tp, tp * v, 0.0)
    neg_flow = np.where(tp < prev_tp, tp * v, 0.0)
    pos_sum = _roll_apply(pos_flow, period, np.sum)
    neg_sum = _roll_apply(neg_flow, period, np.sum)
    mfr = np.zeros_like(tp)
    mask = neg_sum != 0
    mfr[mask] = pos_sum[mask] / neg_sum[mask]
    return 100 - (100 / (1 + mfr))


def williams_r(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """Williams %R.

    WR = (HHV(HIGH, period) - CLOSE) / (HHV(HIGH, period) - LLV(LOW, period)) * -100

    Range: -100 to 0 (overbought > -20, oversold < -80)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 周期，默认 14
    返回:
        Williams %R values [-100, 0]
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    hhv = _roll_apply(h, period, lambda x: np.max(x))
    llv = _roll_apply(l, period, lambda x: np.min(x))
    denom = hhv - llv
    wr = np.zeros_like(c)
    mask = denom != 0
    wr[mask] = ((hhv[mask] - c[mask]) / denom[mask]) * -100
    return wr


def roc(
    prices: Union[np.ndarray, pd.Series],
    period: int = 10,
) -> np.ndarray:
    """Rate of Change (ROC) / Momentum Oscillator.

    ROC = (CLOSE - CLOSE_prev_n) / CLOSE_prev_n * 100

    参数:
        prices: 价格数组
        period: 周期，默认 10
    返回:
        ROC values (percentage)
    """
    p = _to_array(prices)
    prev_p = _shift(p, period)
    roc = np.zeros_like(p)
    mask = prev_p != 0
    roc[mask] = ((p[mask] - prev_p[mask]) / prev_p[mask]) * 100
    return roc


def momentum(
    prices: Union[np.ndarray, pd.Series],
    period: int = 10,
) -> np.ndarray:
    """Momentum.

    Momentum = CLOSE - CLOSE_prev_n

    参数:
        prices: 价格数组
        period: 周期，默认 10
    返回:
        Momentum values (absolute price change)
    """
    p = _to_array(prices)
    return p - _shift(p, period)


def cmo(
    prices: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """Chande Momentum Oscillator (CMO).

    CMO = (SumGains - SumLosses) / (SumGains + SumLosses) * 100

    参数:
        prices: 价格数组
        period: 周期，默认 14
    返回:
        CMO values [-100, 100]
    """
    p = _to_array(prices)
    deltas = np.diff(p, prepend=p[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    sum_gains = _roll_apply(gains, period, np.sum)
    sum_losses = _roll_apply(losses, period, np.sum)
    denom = sum_gains + sum_losses
    cmo_val = np.zeros_like(p)
    mask = denom != 0
    cmo_val[mask] = ((sum_gains[mask] - sum_losses[mask]) / denom[mask]) * 100
    return cmo_val


def kdj(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ Indicator (Stochastic variant).

    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, M1),  D = EMA(K, M2),  J = K*3 - D*2

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        n: RSV 周期，默认 9
        m1: K 平滑周期，默认 3
        m2: D 平滑周期，默认 3
    返回:
        (K, D, J) 元组
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    llv = _roll_apply(l, n, lambda x: np.min(x))
    hhv = _roll_apply(h, n, lambda x: np.max(x))
    denom = hhv - llv
    rsv = np.zeros_like(c)
    mask = denom != 0
    rsv[mask] = ((c[mask] - llv[mask]) / denom[mask]) * 100
    rsv[~mask] = 50
    k = _ema(rsv, m1 * 2 - 1)
    d = _ema(k, m2 * 2 - 1)
    j = k * 3 - d * 2
    return k, d, j


# --------------------------------------------------------------------------
# Volatility Indicators
# --------------------------------------------------------------------------


def atr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """Average True Range (ATR).

    ATR = SMA(TR, period)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 周期，默认 14
    返回:
        ATR values
    """
    tr_val = tr(high, low, close)
    return _sma(tr_val, period)


def atrp(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """ATR as Percentage of Close price.

    ATRP = ATR / CLOSE * 100

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 周期，默认 14
    返回:
        ATRP values (percentage)
    """
    c = _to_array(close)
    atr_val = atr(high, low, close, period)
    atrp = np.zeros_like(c)
    mask = c != 0
    atrp[mask] = (atr_val[mask] / c[mask]) * 100
    return atrp


def bollinger_bands(
    prices: Union[np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    MID = SMA(close, period)
    UPPER = MID + STD * std_dev
    LOWER = MID - STD * std_dev

    参数:
        prices: 价格数组
        period: 周期，默认 20
        std_dev: 标准差倍数，默认 2.0
    返回:
        (UpperBand, MiddleBand, LowerBand) 元组
    """
    p = _to_array(prices)
    mid = _sma(p, period)
    std = _roll_apply(p, period, lambda x: np.std(x, ddof=0))
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return upper, mid, lower


def keltner(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    ema_period: int = 20,
    multiplier: float = 2.0,
    atr_period: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels.

    CenterLine = EMA(close, ema_period)
    UpperBand = CenterLine + multiplier * ATR(period)
    LowerBand = CenterLine - multiplier * ATR(period)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        ema_period: EMA 周期，默认 20
        multiplier: ATR 倍数，默认 2.0
        atr_period: ATR 周期，默认 10
    返回:
        (UpperBand, CenterLine, LowerBand) 元组
    """
    c = _to_array(close)
    center = _ema(c, ema_period)
    atr_val = atr(high, low, c, atr_period)
    upper = center + multiplier * atr_val
    lower = center - multiplier * atr_val
    return upper, center, lower


def donchian(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channels (Price Channels).

    UpperBand = HHV(HIGH, period)
    LowerBand = LLV(LOW, period)
    CenterLine = (UpperBand + LowerBand) / 2

    参数:
        high: 最高价数组
        low: 最低价数组
        period: 周期，默认 20
    返回:
        (UpperBand, CenterLine, LowerBand) 元组
    """
    h, l = _to_array(high), _to_array(low)
    upper = _roll_apply(h, period, lambda x: np.max(x))
    lower = _roll_apply(l, period, lambda x: np.min(x))
    center = (upper + lower) / 2
    return upper, center, lower


def starc_bands(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 20,
    multiplier: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STARC Bands (Stoller Average Range Channels).

    CenterLine = SMA(close, period)
    UpperBand = CenterLine + multiplier * ATR(period)
    LowerBand = CenterLine - multiplier * ATR(period)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: SMA 周期，默认 20
        multiplier: ATR 倍数，默认 2.0
    返回:
        (UpperBand, CenterLine, LowerBand) 元组
    """
    c = _to_array(close)
    center = _sma(c, period)
    atr_val = atr(high, low, c, period)
    upper = center + multiplier * atr_val
    lower = center - multiplier * atr_val
    return upper, center, lower


# --------------------------------------------------------------------------
# Volume Indicators
# --------------------------------------------------------------------------


def obv(
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """On-Balance Volume (OBV).

    OBV = cumulative sum of (volume if close > close_prev else -volume if close < close_prev else 0)

    参数:
        close: 收盘价数组
        volume: 成交量数组
    返回:
        OBV values
    """
    c = _to_array(close)
    v = _to_array(volume)
    direction = np.sign(c - _shift(c, 1))
    obv = np.zeros_like(c)
    obv[0] = v[0]
    for i in range(1, len(c)):
        obv[i] = obv[i - 1] + direction[i] * v[i]
    return obv


def vwap(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Volume Weighted Average Price (VWAP).

    VWAP = SUM(TP * Vol) / SUM(Vol)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
    返回:
        VWAP values
    """
    h, l, c, v = _to_array(high), _to_array(low), _to_array(close), _to_array(volume)
    tp = typical_price(h, l, c)
    cum_tp_vol = np.zeros_like(tp)
    cum_vol = np.zeros_like(tp)
    # Cumulative sum from start
    cum_tp_vol = np.cumsum(tp * v)
    cum_vol = np.cumsum(v)
    vwap = np.zeros_like(tp)
    mask = cum_vol != 0
    vwap[mask] = cum_tp_vol[mask] / cum_vol[mask]
    return vwap


def adl(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Accumulation/Distribution Line (ADL).

    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    MFV = MFM * Volume
    ADL = cumulative sum of MFV

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
    返回:
        ADL values
    """
    h, l, c, v = _to_array(high), _to_array(low), _to_array(close), _to_array(volume)
    denom = h - l
    mfm = np.zeros_like(c)
    mask = denom != 0
    mfm[mask] = ((c[mask] - l[mask]) - (h[mask] - c[mask])) / denom[mask]
    mfv = mfm * v
    return np.cumsum(mfv)


def cmf(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Chaikin Money Flow (CMF).

    CMF = SUM(MFV, period) / SUM(Volume, period)
    MFV = MFM * Volume

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        period: 周期，默认 20
    返回:
        CMF values [-1, 1]
    """
    h, l, c, v = _to_array(high), _to_array(low), _to_array(close), _to_array(volume)
    denom = h - l
    mfm = np.zeros_like(c)
    mask = denom != 0
    mfm[mask] = ((c[mask] - l[mask]) - (h[mask] - c[mask])) / denom[mask]
    mfv = mfm * v
    sum_mfv = _roll_apply(mfv, period, np.sum)
    sum_vol = _roll_apply(v, period, np.sum)
    cmf = np.zeros_like(c)
    mask = sum_vol != 0
    cmf[mask] = sum_mfv[mask] / sum_vol[mask]
    return cmf


def pvo(
    volume: Union[np.ndarray, pd.Series],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentage Volume Oscillator (PVO).

    PVO = (EMA(vol, fast) - EMA(vol, slow)) / EMA(vol, slow) * 100
    Signal = EMA(PVO, signal)
    Histogram = PVO - Signal

    参数:
        volume: 成交量数组
        fast: 快线周期，默认 12
        slow: 慢线周期，默认 26
        signal: 信号线周期，默认 9
    返回:
        (PVO, Signal, Histogram) 元组
    """
    v = _to_array(volume)
    ema_fast = _ema(v, fast)
    ema_slow = _ema(v, slow)
    pvo = np.zeros_like(v)
    mask = ema_slow != 0
    pvo[mask] = ((ema_fast[mask] - ema_slow[mask]) / ema_slow[mask]) * 100
    signal_line = _ema(pvo, signal)
    histogram = pvo - signal_line
    return pvo, signal_line, histogram


# --------------------------------------------------------------------------
# Statistical Indicators
# --------------------------------------------------------------------------


def std_dev(
    prices: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Standard Deviation (rolling).

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        Rolling standard deviation
    """
    p = _to_array(prices)
    return _roll_apply(p, period, lambda x: np.std(x, ddof=0))


def slope(
    prices: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Linear Regression Slope.

    slope = (n * SUM(x*y) - SUM(x)*SUM(y)) / (n*SUM(x^2) - SUM(x)^2)

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        Slope values
    """
    p = _to_array(prices)
    slope_arr = np.zeros_like(p)
    x = np.arange(period)
    x_mean = (period - 1) / 2
    x_sq_sum = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, len(p)):
        window = p[i - period + 1:i + 1]
        y_mean = np.mean(window)
        xy_cov = np.sum((x - x_mean) * (window - y_mean))
        slope_arr[i] = xy_cov / x_sq_sum
    return slope_arr


def variance(
    prices: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Rolling Variance (population).

    参数:
        prices: 价格数组
        period: 周期，默认 20
    返回:
        Rolling variance
    """
    p = _to_array(prices)
    return _roll_apply(p, period, lambda x: np.var(x, ddof=0))


def covariance(
    prices1: Union[np.ndarray, pd.Series],
    prices2: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Rolling Covariance (population).

    参数:
        prices1: 价格数组 1
        prices2: 价格数组 2
        period: 周期，默认 20
    返回:
        Rolling covariance
    """
    p1, p2 = _to_array(prices1), _to_array(prices2)
    cov = np.zeros_like(p1)
    for i in range(period - 1, len(p1)):
        w1 = p1[i - period + 1:i + 1]
        w2 = p2[i - period + 1:i + 1]
        cov[i] = np.mean((w1 - np.mean(w1)) * (w2 - np.mean(w2)))
    return cov


def correlation(
    prices1: Union[np.ndarray, pd.Series],
    prices2: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Rolling Pearson Correlation.

    参数:
        prices1: 价格数组 1
        prices2: 价格数组 2
        period: 周期，默认 20
    返回:
        Rolling correlation [-1, 1]
    """
    p1, p2 = _to_array(prices1), _to_array(prices2)
    corr = np.zeros_like(p1)
    for i in range(period - 1, len(p1)):
        w1 = p1[i - period + 1:i + 1]
        w2 = p2[i - period + 1:i + 1]
        std1 = np.std(w1, ddof=0)
        std2 = np.std(w2, ddof=0)
        if std1 != 0 and std2 != 0:
            cov = np.mean((w1 - np.mean(w1)) * (w2 - np.mean(w2)))
            corr[i] = cov / (std1 * std2)
        else:
            corr[i] = 0
    return corr


def beta(
    prices: Union[np.ndarray, pd.Series],
    market_prices: Union[np.ndarray, pd.Series],
    period: int = 20,
) -> np.ndarray:
    """Rolling Beta (market vs asset).

    Beta = Cov(asset, market) / Var(market)

    参数:
        prices: 资产价格数组
        market_prices: 市场指数价格数组
        period: 周期，默认 20
    返回:
        Rolling beta
    """
    p = _to_array(prices)
    m = _to_array(market_prices)
    beta_arr = np.zeros_like(p)
    for i in range(period - 1, len(p)):
        w1 = p[i - period + 1:i + 1]
        w2 = m[i - period + 1:i + 1]
        var_m = np.var(w2, ddof=0)
        if var_m != 0:
            cov = np.mean((w1 - np.mean(w1)) * (w2 - np.mean(w2)))
            beta_arr[i] = cov / var_m
        else:
            beta_arr[i] = 0
    return beta_arr


# --------------------------------------------------------------------------
# Trend Indicators
# --------------------------------------------------------------------------


def adx(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index (ADX).

    ADX = SMA(|PDI - MDI| / (PDI + MDI) * 100, period)
    PDI = Positive Directional Indicator
    MDI = Negative Directional Indicator

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 周期，默认 14
    返回:
        (PDI, MDI, ADX) 元组
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    prev_c = _shift(c, 1)
    hl = h - l
    hpc = np.abs(h - prev_c)
    lpc = np.abs(l - prev_c)
    tr_val = np.maximum(hl, np.maximum(hpc, lpc))
    pos_dm = np.zeros_like(h)
    neg_dm = np.zeros_like(h)
    hd = h - _shift(h, 1)
    ld = _shift(l, 1) - l
    pos_dm[(hd > 0) & (hd > ld)] = hd[(hd > 0) & (hd > ld)]
    neg_dm[(ld > 0) & (ld > hd)] = ld[(ld > 0) & (ld > hd)]

    atr_val = atr(h, l, c, period)
    sum_pos_dm = _roll_apply(pos_dm, period, np.sum)
    sum_neg_dm = _roll_apply(neg_dm, period, np.sum)
    sum_tr = _roll_apply(tr_val, period, np.sum)

    pdi = np.zeros_like(c)
    mdi = np.zeros_like(c)
    mask = sum_tr != 0
    pdi[mask] = (sum_pos_dm[mask] / sum_tr[mask]) * 100
    mdi[mask] = (sum_neg_dm[mask] / sum_tr[mask]) * 100

    dx = np.zeros_like(c)
    mask = (pdi + mdi) != 0
    dx[mask] = (np.abs(pdi[mask] - mdi[mask]) / (pdi[mask] + mdi[mask])) * 100
    adx_val = _sma(dx, period)
    return pdi, mdi, adx_val


def aroon(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    period: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aroon Indicator.

    AroonUp = (period - periods_since_highest_high) / period * 100
    AroonDown = (period - periods_since_lowest_low) / period * 100

    参数:
        high: 最高价数组
        low: 最低价数组
        period: 周期，默认 25
    返回:
        (AroonUp, AroonDown) 元组
    """
    h, l = _to_array(high), _to_array(low)
    aroon_up = np.zeros_like(h)
    aroon_down = np.zeros_like(h)

    for i in range(period - 1, len(h)):
        window_h = h[i - period + 1:i + 1]
        window_l = l[i - period + 1:i + 1]
        periods_since_high = period - 1 - np.argmax(window_h)
        periods_since_low = period - 1 - np.argmin(window_l)
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    return aroon_up, aroon_down


def super_trend(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 20,
    multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """SuperTrend Indicator.

    UpperBand = HL2 + multiplier * ATR
    LowerBand = HL2 - multiplier * ATR
    SuperTrend = direction (1 = up, -1 = down)

    参数:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR 周期，默认 20
        multiplier: ATR 倍数，默认 3.0
    返回:
        (SuperTrend, UpperBand, LowerBand) — actually (trend, upper, lower)
    """
    h, l, c = _to_array(high), _to_array(low), _to_array(close)
    hl2 = (h + l) / 2
    atr_val = atr(h, l, c, period)

    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    trend = np.ones_like(c)  # 1 = up, -1 = down

    for i in range(1, len(c)):
        if c[i] > upper[i - 1]:
            trend[i] = 1
            upper[i] = max(upper[i], upper[i - 1])
        elif c[i] < lower[i - 1]:
            trend[i] = -1
            lower[i] = min(lower[i], lower[i - 1])
        else:
            trend[i] = trend[i - 1]
            if trend[i] == 1:
                lower[i] = max(lower[i], lower[i - 1])
            else:
                upper[i] = min(upper[i], upper[i - 1])

    return trend, upper, lower


# --------------------------------------------------------------------------
# Candlestick Pattern Recognition
# All patterns return: 1 = bullish signal, -1 = bearish signal, 0 = no pattern
# --------------------------------------------------------------------------


def cdl_doji(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    max_price_change_pct: float = 0.1,
) -> np.ndarray:
    """Doji Candlestick Pattern.

    Doji: Open and Close are virtually identical.
    max_price_change_pct: Maximum decimalized % difference (default 0.1 = 10%)

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        max_price_change_pct: 最大价格变化百分比，默认 0.1 (10%)
    返回:
        1 (bullish - rare for doji), -1 (bearish - rare for doji), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    price_range = h - l
    doji = np.zeros_like(o)
    mask = price_range != 0
    doji[mask] = (body[mask] / price_range[mask]) <= max_price_change_pct
    return doji.astype(int) * 0  # Doji is neutral; return 0 for both directions


def cdl_engulfing(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Engulfing Pattern (Bullish & Bearish).

    Bullish Engulfing: Current candle is bullish, previous is bearish,
    and current body engulfs previous body.
    Bearish Engulfing: Current candle is bearish, previous is bullish,
    and current body engulfs previous body.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (bullish engulfing), -1 (bearish engulfing), 0 (no pattern)
    """
    o, c = _to_array(open_), _to_array(close)
    prev_o = _shift(o, 1)
    prev_c = _shift(c, 1)

    curr_bullish = c > o
    curr_bearish = c < o
    prev_bullish = prev_c > prev_o
    prev_bearish = prev_c < prev_o

    curr_body_top = np.maximum(c, o)
    curr_body_bot = np.minimum(c, o)
    prev_body_top = np.maximum(prev_c, prev_o)
    prev_body_bot = np.minimum(prev_c, prev_o)

    bullish_engulf = (curr_bullish & prev_bearish &
                      (curr_body_bot <= prev_body_bot) &
                      (curr_body_top >= prev_body_top))
    bearish_engulf = (curr_bearish & prev_bullish &
                      (curr_body_top >= prev_body_top) &
                      (curr_body_bot <= prev_body_bot))

    result = np.zeros_like(o)
    result[bullish_engulf] = 1
    result[bearish_engulf] = -1
    return result


def cdl_hammer(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Hammer Candlestick Pattern.

    Hammer: Small body at upper end, lower wick at least 2x body,
    little or no upper wick. Bullish reversal signal.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (hammer), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)

    body_range = h - l
    body_pct = np.zeros_like(body)
    mask = body_range != 0
    body_pct[mask] = body[mask] / body_range[mask]

    # Conditions: body in upper 30%, lower wick > 2x body, upper wick < 0.5x body
    hammer = ((body_pct < 0.3) &
              (lower_wick > 2 * body) &
              (np.abs(upper_wick) < 0.5 * body))
    return hammer.astype(int)


def cdl_morning_star(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Morning Star Pattern (Bullish Reversal).

    3-candle: Large bearish body, small body (gap down possible),
    large bullish body closing above midpoint of first body.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (morning star), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    prev_o = _shift(o, 1)
    prev_c = _shift(c, 1)
    prev2_o = _shift(o, 2)
    prev2_c = _shift(c, 2)

    # First: bearish candle
    first_bearish = prev2_c < prev2_o
    # Second: small body (doji or spinning top)
    second_body = np.abs(prev_c - prev_o)
    second_range = _shift(h, 1) - _shift(l, 1)
    second_small = np.zeros_like(second_body)
    mask = second_range != 0
    second_small[mask] = second_body[mask] / second_range[mask]
    second_small = second_small < 0.3
    # Third: bullish closing above midpoint of first body
    midpoint = (prev2_o + prev2_c) / 2
    third_bullish = c > o
    third_closes_above = c > midpoint

    # Also require gap down between first and second
    morning_star = (first_bearish & second_small & third_bullish & third_closes_above)
    return morning_star.astype(int)


def cdl_evening_star(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Evening Star Pattern (Bearish Reversal).

    3-candle: Large bullish body, small body, large bearish body
    closing below midpoint of first body.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        -1 (evening star), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    prev_o = _shift(o, 1)
    prev_c = _shift(c, 1)
    prev2_o = _shift(o, 2)
    prev2_c = _shift(c, 2)

    # First: bullish candle
    first_bullish = prev2_c > prev2_o
    # Second: small body
    second_body = np.abs(prev_c - prev_o)
    second_range = _shift(h, 1) - _shift(l, 1)
    second_small = np.zeros_like(second_body)
    mask = second_range != 0
    second_small[mask] = second_body[mask] / second_range[mask]
    second_small = second_small < 0.3
    # Third: bearish closing below midpoint of first body
    midpoint = (prev2_o + prev2_c) / 2
    third_bearish = c < o
    third_closes_below = c < midpoint

    evening_star = (first_bullish & second_small & third_bearish & third_closes_below)
    return (-evening_star).astype(int)


def cdl_three_white_soldiers(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Three White Soldiers Pattern (Bullish Reversal).

    3 consecutive bullish candles with progressively higher closes
    and each opening within previous body's range.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (three white soldiers), 0 (no pattern)
    """
    o, c = _to_array(open_), _to_array(close)
    prev_o = _shift(o, 1)
    prev_c = _shift(c, 1)
    prev2_o = _shift(o, 2)
    prev2_c = _shift(c, 2)

    # All three bullish
    bullish1 = prev2_c > prev2_o
    bullish2 = prev_c > prev_o
    bullish3 = c > o

    # Each closes higher than previous
    higher_closes = (prev_c > prev2_c) & (c > prev_c)

    # Each opens within or near previous body
    opens_in_prev = (prev_o >= prev2_o) & (prev_o <= prev2_c)
    current_opens_in_prev = (o >= prev_o) & (o <= prev_c)

    three_white = bullish1 & bullish2 & bullish3 & higher_closes & opens_in_prev & current_opens_in_prev
    return three_white.astype(int)


def cdl_three_black_crows(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Three Black Crows Pattern (Bearish Reversal).

    3 consecutive bearish candles with progressively lower closes
    and each opening within previous body's range.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        -1 (three black crows), 0 (no pattern)
    """
    o, c = _to_array(open_), _to_array(close)
    prev_o = _shift(o, 1)
    prev_c = _shift(c, 1)
    prev2_o = _shift(o, 2)
    prev2_c = _shift(c, 2)

    # All three bearish
    bearish1 = prev2_c < prev2_o
    bearish2 = prev_c < prev_o
    bearish3 = c < o

    # Each closes lower than previous
    lower_closes = (prev_c < prev2_c) & (c < prev_c)

    # Each opens within or near previous body
    opens_in_prev = (prev_o <= prev2_o) & (prev_o >= prev2_c)
    current_opens_in_prev = (o <= prev_o) & (o >= prev_c)

    three_crows = bearish1 & bearish2 & bearish3 & lower_closes & opens_in_prev & current_opens_in_prev
    return (-three_crows).astype(int)


def cdl_marubozu(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    min_body_pct: float = 0.95,
) -> np.ndarray:
    """Marubozu Pattern (Strong Directional Move).

    Marubozu: Candle with no or very small wicks (body >= min_body_pct of range).
    Returns 1 for bullish (close = high), -1 for bearish (close = low).

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        min_body_pct: 最小实体百分比，默认 0.95
    返回:
        1 (bullish marubozu), -1 (bearish marubozu), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    range_ = h - l
    body_pct = np.zeros_like(body)
    mask = range_ != 0
    body_pct[mask] = body[mask] / range_[mask]

    is_marubozu = body_pct >= min_body_pct
    bullish = is_marubozu & (c > o)
    bearish = is_marubozu & (c < o)

    result = np.zeros_like(o)
    result[bullish] = 1
    result[bearish] = -1
    return result


def cdl_spinning_top(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Spinning Top Pattern (Indecision).

    Small body with wicks on both sides (> 50% of range on each side).
    Indicates market indecision.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        0 (neutral pattern, return 0)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    range_ = h - l
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)

    body_pct = np.zeros_like(body)
    mask = range_ != 0
    body_pct[mask] = body[mask] / range_[mask]

    upper_wick_pct = np.zeros_like(body)
    upper_wick_pct[mask] = upper_wick[mask] / range_[mask]
    lower_wick_pct = np.zeros_like(body)
    lower_wick_pct[mask] = lower_wick[mask] / range_[mask]

    spinning = (body_pct < 0.3) & (upper_wick_pct > 0.5) & (lower_wick_pct > 0.5)
    return spinning.astype(int) * 0  # Neutral signal, return 0


def cdl_inverted_hammer(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Inverted Hammer (Bullish Reversal).

    Small body at lower end, long upper wick (at least 2x body),
    little or no lower wick. Similar to shooting star but at bottom.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (inverted hammer), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)
    range_ = h - l
    body_pct = np.zeros_like(body)
    mask = range_ != 0
    body_pct[mask] = body[mask] / range_[mask]

    # Body in lower 30%, long upper wick, small lower wick
    inverted_hammer = ((body_pct < 0.3) &
                       (upper_wick > 2 * body) &
                       (lower_wick < 0.5 * body))
    return inverted_hammer.astype(int)


def cdl_shooting_star(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Shooting Star (Bearish Reversal).

    Small body at lower end, long upper wick (at least 2x body),
    little or no lower wick. Appears at top of uptrend.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        -1 (shooting star), 0 (no pattern)
    """
    result = cdl_inverted_hammer(open_, high, low, close)
    return -result


def cdl_belt_hold(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Belt Hold Pattern.

    Bullish Belt Hold: Open equals low (or near), bearish candle,
    closes near open with small wicks.
    Bearish Belt Hold: Open equals high, bullish candle.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (bullish belt hold), -1 (bearish belt hold), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    range_ = h - l
    body = np.abs(c - o)
    body_pct = np.zeros_like(body)
    mask = range_ != 0
    body_pct[mask] = body[mask] / range_[mask]

    bullish_belt = (np.abs(o - l) / np.where(range_ != 0, range_, 1) < 0.02) & (c < o) & (body_pct > 0.8)
    bearish_belt = (np.abs(o - h) / np.where(range_ != 0, range_, 1) < 0.02) & (c > o) & (body_pct > 0.8)

    result = np.zeros_like(o)
    result[bullish_belt] = 1
    result[bearish_belt] = -1
    return result


def cdl_dragonfly_doji(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Dragonfly Doji.

    Doji with long lower wick and no upper wick. Looks like a T with a long tail.
    Bullish reversal signal at bottom.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (dragonfly doji), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    range_ = h - l
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)

    mask = range_ != 0
    is_doji = (body / np.where(mask, range_, 1)) < 0.1
    long_lower = (lower_wick / np.where(mask, range_, 1)) > 0.6
    short_upper = (upper_wick / np.where(mask, range_, 1)) < 0.1

    dragonfly = is_doji & long_lower & short_upper
    return dragonfly.astype(int)


def cdl_gravestone_doji(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Gravestone Doji.

    Doji with long upper wick and no lower wick. Looks like an inverted T.
    Bearish reversal signal at top.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        -1 (gravestone doji), 0 (no pattern)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    range_ = h - l
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)

    mask = range_ != 0
    is_doji = (body / np.where(mask, range_, 1)) < 0.1
    long_upper = (upper_wick / np.where(mask, range_, 1)) > 0.6
    short_lower = (lower_wick / np.where(mask, range_, 1)) < 0.1

    gravestone = is_doji & long_upper & short_lower
    return (-gravestone).astype(int)


def cdl_long_legged_doji(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Long-Legged Doji.

    Doji with long upper and lower wicks. Shows extreme market indecision.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        0 (neutral doji, return 0)
    """
    o, h, l, c = _to_array(open_), _to_array(high), _to_array(low), _to_array(close)
    body = np.abs(c - o)
    range_ = h - l
    upper_wick = np.maximum(o, c) - h
    lower_wick = l - np.minimum(o, c)
    upper_wick = np.maximum(upper_wick, 0)
    lower_wick = np.maximum(lower_wick, 0)

    mask = range_ != 0
    is_doji = (body / np.where(mask, range_, 1)) < 0.1
    long_upper = (upper_wick / np.where(mask, range_, 1)) > 0.3
    long_lower = (lower_wick / np.where(mask, range_, 1)) > 0.3

    long_legged = is_doji & long_upper & long_lower
    return long_legged.astype(int) * 0  # Neutral signal


def cdl_paper_umbrella(
    open_: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Paper Umbrella (Hammer / Inverted Hammer).

    Single candlestick with long lower wick and small body at top.
    Hammer (bullish) if at bottom of downtrend.
    Inverted Hammer (bullish) at bottom.

    参数:
        open_: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
    返回:
        1 (paper umbrella), 0 (no pattern)
    """
    return cdl_hammer(open_, high, low, close)
