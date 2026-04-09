"""
High-Frequency Factor Library — 39+高频因子 (A股逐笔数据).

Pure NumPy/Pandas implementation; no Talib, no Cython.
All functions accept NumPy arrays and return NumPy arrays unless noted.

Factor List (__all__)
---------------------
Order-side (A1-A16):
    order_arrival_rate     — # orders arriving in last window
    cum_order_count        — cumulative order count up to t
    order_volume_sum       — total order volume in last window
    cum_order_volume       — cumulative order volume up to t
    buy_order_arrival_rate — # buy orders in last window
    sell_order_arrival_rate— # sell orders in last window
    buy_order_volume       — buy order volume in last window
    sell_order_volume      — sell order volume in last window
    fill_kill_count        — fill-and-kill order count (NaN for A-share)
    cancel_count           — # cancelled orders in last window
    cancel_volume          — cancelled order volume in last window
    buy_cancel_count       — # cancelled buy orders
    sell_cancel_count      — # cancelled sell orders
    buy_cancel_volume      — cancelled buy volume
    sell_cancel_volume     — cancelled sell volume
    cum_cancel_count       — cumulative cancelled order count

VWAP Cancellation (A17-A19):
    cancel_vwap            — VWAP of cancelled orders
    cancel_buy_vwap         — VWAP of cancelled buy orders
    cancel_sell_vwap        — VWAP of cancelled sell orders

Order Ratios (A20-A27):
    cancel_rate            — cancel count / arrival count ratio
    cancel_volume_rate     — cancel volume / arrival volume ratio
    buy_cancel_rate        — buy cancel count / buy arrival count
    sell_cancel_rate       — sell cancel count / sell arrival count
    avg_order_size         — mean order size (arrived)
    avg_buy_order_size     — mean buy order size
    avg_sell_order_size    — mean sell order size
    order_size_std        — std of order size

Trade-side (A28-A39):
    trade_arrival_rate     — # trades in last window
    cum_trade_count        — cumulative trade count
    trade_volume_sum       — total trade volume in last window
    cum_trade_volume       — cumulative trade volume
    buy_trade_volume       — buyer-initiated trade volume
    sell_trade_volume      — seller-initiated trade volume
    trade_imbalance        — (V_buy - V_sell) / (V_buy + V_sell)
    order_trade_ratio      — # orders / # trades ratio
    buy_trade_ratio        — buyer-initiated volume ratio
    trade_size_avg         — average trade size
    trade_size_skew        — skewness of trade size distribution
    trade_vwap             — volume-weighted average trade price

Core Microstructure Factors (独立因子):
    order_imbalance        — OIR = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    vpin                    — Volume-synchronized PIN
    flow_toxicity           — Llorca et al. flow toxicity
    spread_decomposition    — realized / effective / price impact
    order_flow_skew         — order flow skewness
    trade_direction        — Lee-Ready tick rule
    quote_velocity          — quote update velocity
    trade_activity_ratio   — trade volume / quoted volume
    price_impact           — Kyle's lambda
    midprice_ma_diff       — midprice vs MA spread
    volume_curve_slope     — volume curve shape
    order_arrival_intensity— Hawkes intensity lambda
    price_reversion        — short-term price reversion
    volatility_ratio       — realized vs implied vol
    trade_size_imbalance   — buy/sell size ratio
    quote_cluster_quality  — quote clustering quality
    liquidity_score         — composite liquidity score

Origin: D:/Hive/Data/trading_repos/high-frequency-factors/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

__all__ = [
    # Order-side A1-A16
    'order_arrival_rate', 'cum_order_count', 'order_volume_sum',
    'cum_order_volume', 'buy_order_arrival_rate', 'sell_order_arrival_rate',
    'buy_order_volume', 'sell_order_volume', 'fill_kill_count',
    'cancel_count', 'cancel_volume', 'buy_cancel_count', 'sell_cancel_count',
    'buy_cancel_volume', 'sell_cancel_volume', 'cum_cancel_count',
    # VWAP A17-A19
    'cancel_vwap', 'cancel_buy_vwap', 'cancel_sell_vwap',
    # Ratios A20-A27
    'cancel_rate', 'cancel_volume_rate', 'buy_cancel_rate', 'sell_cancel_rate',
    'avg_order_size', 'avg_buy_order_size', 'avg_sell_order_size', 'order_size_std',
    # Trade-side A28-A39
    'trade_arrival_rate', 'cum_trade_count', 'trade_volume_sum',
    'cum_trade_volume', 'buy_trade_volume', 'sell_trade_volume',
    'trade_imbalance', 'order_trade_ratio', 'buy_trade_ratio',
    'trade_size_avg', 'trade_size_skew', 'trade_vwap',
    # Core microstructure
    'order_imbalance', 'vpin', 'flow_toxicity', 'spread_decomposition',
    'order_flow_skew', 'trade_direction', 'quote_velocity',
    'trade_activity_ratio', 'price_impact', 'midprice_ma_diff',
    'volume_curve_slope', 'order_arrival_intensity', 'price_reversion',
    'volatility_ratio', 'trade_size_imbalance', 'quote_cluster_quality',
    'liquidity_score',
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum with left-closed window, pure numpy."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = arr[i - window:i].sum()
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with left-closed window, pure numpy."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = arr[i - window:i].mean()
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std with left-closed window, ddof=0, pure numpy."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = arr[i - window:i].std()
    return out


def _cumshift(arr: np.ndarray) -> np.ndarray:
    """Cumulative sum shifted by 1 (exclusive cumulative)."""
    return np.cumsum(arr) - arr


def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Element-wise divide with 0-guard."""
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.where(b != 0, a / b, fill)
    return c


# ---------------------------------------------------------------------------
# Order-side factors A1-A16
# ---------------------------------------------------------------------------

def order_arrival_rate(order_count_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of orders arriving in the last `window` seconds.

    A1 — Order Arrival Rate (60s window).
    """
    return _rolling_sum(order_count_per_sec.astype(float), window)


def cum_order_count(order_count_per_sec: np.ndarray) -> np.ndarray:
    """Cumulative count of arrived orders up to t (exclusive).

    A2 — Cumulative Order Count.
    """
    return _cumshift(order_count_per_sec.astype(float))


def order_volume_sum(order_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Total order volume arriving in the last `window` seconds.

    A3 — Order Volume Sum (60s window).
    """
    return _rolling_sum(order_vol_per_sec.astype(float), window)


def cum_order_volume(order_vol_per_sec: np.ndarray) -> np.ndarray:
    """Cumulative order volume up to t (exclusive).

    A4 — Cumulative Order Volume.
    """
    return _cumshift(order_vol_per_sec.astype(float))


def buy_order_arrival_rate(buy_order_count_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of buy orders arriving in the last `window` seconds.

    A5 — Buy Order Arrival Rate.
    """
    return _rolling_sum(buy_order_count_per_sec.astype(float), window)


def sell_order_arrival_rate(sell_order_count_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of sell orders arriving in the last `window` seconds.

    A6 — Sell Order Arrival Rate.
    """
    return _rolling_sum(sell_order_count_per_sec.astype(float), window)


def buy_order_volume(buy_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Buy order volume in the last `window` seconds.

    A7 — Buy Order Volume.
    """
    return _rolling_sum(buy_vol_per_sec.astype(float), window)


def sell_order_volume(sell_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Sell order volume in the last `window` seconds.

    A8 — Sell Order Volume.
    """
    return _rolling_sum(sell_vol_per_sec.astype(float), window)


def fill_kill_count(n_secs: int) -> np.ndarray:
    """Fill-and-kill order count (always NaN for A-share — no FAK data).

    A9 — Fill-and-Kill Order Count.
    """
    return np.full(n_secs, np.nan)


def cancel_count(cancel_count_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of cancelled orders in the last `window` seconds.

    A10 — Cancel Count.
    """
    return _rolling_sum(cancel_count_per_sec.astype(float), window)


def cancel_volume(cancel_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Cancelled order volume in the last `window` seconds.

    A11 — Cancel Volume.
    """
    return _rolling_sum(cancel_vol_per_sec.astype(float), window)


def buy_cancel_count(buy_cancel_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of cancelled buy orders in the last `window` seconds.

    A12 — Buy Cancel Count.
    """
    return _rolling_sum(buy_cancel_per_sec.astype(float), window)


def sell_cancel_count(sell_cancel_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of cancelled sell orders in the last `window` seconds.

    A13 — Sell Cancel Count.
    """
    return _rolling_sum(sell_cancel_per_sec.astype(float), window)


def buy_cancel_volume(buy_cancel_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Cancelled buy order volume in the last `window` seconds.

    A14 — Buy Cancel Volume.
    """
    return _rolling_sum(buy_cancel_vol_per_sec.astype(float), window)


def sell_cancel_volume(sell_cancel_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Cancelled sell order volume in the last `window` seconds.

    A15 — Sell Cancel Volume.
    """
    return _rolling_sum(sell_cancel_vol_per_sec.astype(float), window)


def cum_cancel_count(cancel_count_per_sec: np.ndarray) -> np.ndarray:
    """Cumulative cancelled order count up to t (exclusive).

    A16 — Cumulative Cancel Count.
    """
    return _cumshift(cancel_count_per_sec.astype(float))


# ---------------------------------------------------------------------------
# VWAP Cancellation A17-A19
# (require join with order book to get price; returns NaN if data unavailable)
# ---------------------------------------------------------------------------

def cancel_vwap(cancel_price: np.ndarray, cancel_vol: np.ndarray) -> np.ndarray:
    """Volume-weighted average price of cancelled orders up to t.

    A17 — Cancel VWAP (cumulative, forwarded).
    Requires cancel_price and cancel_vol arrays aligned to tick index.
    """
    amount = cancel_price * cancel_vol
    cum_amount = _cumshift(amount)
    cum_vol = _cumshift(cancel_vol)
    return _safe_divide(cum_amount, cum_vol, fill=np.nan)


def cancel_buy_vwap(cancel_buy_price: np.ndarray, cancel_buy_vol: np.ndarray) -> np.ndarray:
    """VWAP of cancelled buy orders up to t.

    A18 — Cancel Buy VWAP.
    """
    amount = cancel_buy_price * cancel_buy_vol
    cum_amount = _cumshift(amount)
    cum_vol = _cumshift(cancel_buy_vol)
    return _safe_divide(cum_amount, cum_vol, fill=np.nan)


def cancel_sell_vwap(cancel_sell_price: np.ndarray, cancel_sell_vol: np.ndarray) -> np.ndarray:
    """VWAP of cancelled sell orders up to t.

    A19 — Cancel Sell VWAP.
    """
    amount = cancel_sell_price * cancel_sell_vol
    cum_amount = _cumshift(amount)
    cum_vol = _cumshift(cancel_sell_vol)
    return _safe_divide(cum_amount, cum_vol, fill=np.nan)


# ---------------------------------------------------------------------------
# Order Ratios A20-A27
# ---------------------------------------------------------------------------

def cancel_rate(cancel_count_arr: np.ndarray, arrival_count_arr: np.ndarray,
                window: int = 60) -> np.ndarray:
    """Ratio: cancelled orders / arrived orders in last `window` seconds.

    A20 — Cancel Rate.
    """
    c = _rolling_sum(cancel_count_arr.astype(float), window)
    a = _rolling_sum(arrival_count_arr.astype(float), window)
    return _safe_divide(c, a, fill=np.nan)


def cancel_volume_rate(cancel_vol_arr: np.ndarray, arrival_vol_arr: np.ndarray,
                      window: int = 60) -> np.ndarray:
    """Ratio: cancelled volume / arrived volume in last `window` seconds.

    A21 — Cancel Volume Rate.
    """
    c = _rolling_sum(cancel_vol_arr.astype(float), window)
    a = _rolling_sum(arrival_vol_arr.astype(float), window)
    return _safe_divide(c, a, fill=np.nan)


def buy_cancel_rate(buy_cancel_arr: np.ndarray, buy_arrival_arr: np.ndarray,
                    window: int = 60) -> np.ndarray:
    """Buy cancel count / buy arrival count ratio.

    A22 — Buy Cancel Rate.
    """
    c = _rolling_sum(buy_cancel_arr.astype(float), window)
    a = _rolling_sum(buy_arrival_arr.astype(float), window)
    return _safe_divide(c, a, fill=np.nan)


def sell_cancel_rate(sell_cancel_arr: np.ndarray, sell_arrival_arr: np.ndarray,
                     window: int = 60) -> np.ndarray:
    """Sell cancel count / sell arrival count ratio.

    A23 — Sell Cancel Rate.
    """
    c = _rolling_sum(sell_cancel_arr.astype(float), window)
    a = _rolling_sum(sell_arrival_arr.astype(float), window)
    return _safe_divide(c, a, fill=np.nan)


def avg_order_size(order_vol_arr: np.ndarray, order_count_arr: np.ndarray,
                   window: int = 60) -> np.ndarray:
    """Mean order size (volume / count) in last `window` seconds.

    A24 — Average Order Size.
    """
    vol = _rolling_sum(order_vol_arr.astype(float), window)
    cnt = _rolling_sum(order_count_arr.astype(float), window)
    return _safe_divide(vol, cnt, fill=np.nan)


def avg_buy_order_size(buy_vol_arr: np.ndarray, buy_count_arr: np.ndarray,
                       window: int = 60) -> np.ndarray:
    """Mean buy order size.

    A25 — Average Buy Order Size.
    """
    vol = _rolling_sum(buy_vol_arr.astype(float), window)
    cnt = _rolling_sum(buy_count_arr.astype(float), window)
    return _safe_divide(vol, cnt, fill=np.nan)


def avg_sell_order_size(sell_vol_arr: np.ndarray, sell_count_arr: np.ndarray,
                        window: int = 60) -> np.ndarray:
    """Mean sell order size.

    A26 — Average Sell Order Size.
    """
    vol = _rolling_sum(sell_vol_arr.astype(float), window)
    cnt = _rolling_sum(sell_count_arr.astype(float), window)
    return _safe_divide(vol, cnt, fill=np.nan)


def order_size_std(order_vol_arr: np.ndarray, window: int = 60) -> np.ndarray:
    """Standard deviation of order size in last `window` seconds.

    A27 — Order Size Std.
    """
    return _rolling_std(order_vol_arr.astype(float), window)


# ---------------------------------------------------------------------------
# Trade-side factors A28-A39
# ---------------------------------------------------------------------------

def trade_arrival_rate(trade_count_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Number of trades in the last `window` seconds.

    A28 — Trade Arrival Rate.
    """
    return _rolling_sum(trade_count_per_sec.astype(float), window)


def cum_trade_count(trade_count_per_sec: np.ndarray) -> np.ndarray:
    """Cumulative trade count up to t (exclusive).

    A29 — Cumulative Trade Count.
    """
    return _cumshift(trade_count_per_sec.astype(float))


def trade_volume_sum(trade_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Total trade volume in the last `window` seconds.

    A30 — Trade Volume Sum.
    """
    return _rolling_sum(trade_vol_per_sec.astype(float), window)


def cum_trade_volume(trade_vol_per_sec: np.ndarray) -> np.ndarray:
    """Cumulative trade volume up to t (exclusive).

    A31 — Cumulative Trade Volume.
    """
    return _cumshift(trade_vol_per_sec.astype(float))


def buy_trade_volume(buy_trade_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Buyer-initiated trade volume in the last `window` seconds.

    A32 — Buy Trade Volume.
    """
    return _rolling_sum(buy_trade_vol_per_sec.astype(float), window)


def sell_trade_volume(sell_trade_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Seller-initiated trade volume in the last `window` seconds.

    A33 — Sell Trade Volume.
    """
    return _rolling_sum(sell_trade_vol_per_sec.astype(float), window)


def trade_imbalance(buy_trade_vol: np.ndarray, sell_trade_vol: np.ndarray,
                    window: int = 60) -> np.ndarray:
    """Trade Imbalance Ratio: (V_buy - V_sell) / (V_buy + V_sell).

    A34 — Trade Imbalance (60s window).
    Also available as standalone `order_imbalance()` for any window.
    """
    b = _rolling_sum(buy_trade_vol.astype(float), window)
    s = _rolling_sum(sell_trade_vol.astype(float), window)
    return _safe_divide(b - s, b + s, fill=0.0)


def order_trade_ratio(order_count_arr: np.ndarray, trade_count_arr: np.ndarray,
                      window: int = 60) -> np.ndarray:
    """Ratio: # orders / # trades in last `window` seconds.

    A35 — Order / Trade Ratio.
    """
    o = _rolling_sum(order_count_arr.astype(float), window)
    t = _rolling_sum(trade_count_arr.astype(float), window)
    return _safe_divide(o, t, fill=np.nan)


def buy_trade_ratio(buy_trade_vol: np.ndarray, trade_vol: np.ndarray,
                   window: int = 60) -> np.ndarray:
    """Buyer-initiated volume ratio: V_buy / V_total.

    A36 — Buy Trade Ratio.
    """
    b = _rolling_sum(buy_trade_vol.astype(float), window)
    t = _rolling_sum(trade_vol.astype(float), window)
    return _safe_divide(b, t, fill=np.nan)


def trade_size_avg(trade_vol_per_sec: np.ndarray, window: int = 60) -> np.ndarray:
    """Average trade size in last `window` seconds.

    A37 — Average Trade Size.
    """
    vol = _rolling_sum(trade_vol_per_sec.astype(float), window)
    cnt = _rolling_sum((trade_vol_per_sec > 0).astype(float), window)
    return _safe_divide(vol, cnt, fill=np.nan)


def trade_size_skew(trade_vol_arr: np.ndarray, window: int = 60) -> np.ndarray:
    """Skewness of trade size distribution in last `window` seconds.

    A38 — Trade Size Skewness.
    """
    n = len(trade_vol_arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = trade_vol_arr[i - window:i]
        m = seg.mean()
        s = seg.std()
        if s > 0:
            out[i] = ((seg - m) ** 3).mean() / (s ** 3)
    return out


def trade_vwap(trade_price_arr: np.ndarray, trade_vol_arr: np.ndarray) -> np.ndarray:
    """Cumulative VWAP of trades up to t (exclusive).

    A39 — Trade VWAP.
    """
    amount = trade_price_arr * trade_vol_arr
    cum_amount = _cumshift(amount)
    cum_vol = _cumshift(trade_vol_arr)
    return _safe_divide(cum_amount, cum_vol, fill=np.nan)


# ---------------------------------------------------------------------------
# Core microstructure factors (standalone, not from A1-A39)
# ---------------------------------------------------------------------------

def order_imbalance(buy_vol: np.ndarray, sell_vol: np.ndarray, window: int = 100) -> np.ndarray:
    """Order Imbalance Ratio (OIR).

    OIR = (buy_vol - sell_vol) / (buy_vol + sell_vol)

    Range: [-1, +1]. Positive values indicate buy pressure.

    Parameters
    ----------
    buy_vol, sell_vol : np.ndarray
        Buy/sell volume arrays (same length).
    window : int
        Rolling window size (default 100).

    Returns
    -------
    np.ndarray
        OIR series, NaN for first `window` elements.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> b = np.random.rand(500) * 1000
    >>> s = np.random.rand(500) * 1000
    >>> oir = order_imbalance(b, s, window=50)
    """
    b = _rolling_sum(buy_vol.astype(float), window)
    s = _rolling_sum(sell_vol.astype(float), window)
    return _safe_divide(b - s, b + s, fill=0.0)


def vpin(trade_vol: np.ndarray, buy_vol: np.ndarray, sell_vol: np.ndarray,
         n_bins: int = 50) -> np.ndarray:
    """Volume-synchronized PIN (VPIN).

    VPIN = |V_buy - V_sell| / V_total  (computed in volume buckets)

    Sensitive to informed trading; originally from Easley et al. (2012).
    Uses fixed-volume bucketing (n_bins trades per bucket) rather than time.

    Parameters
    ----------
    trade_vol : np.ndarray — per-trade volume
    buy_vol   : np.ndarray — buyer-initiated volume per trade
    sell_vol  : np.ndarray — seller-initiated volume per trade
    n_bins    : int        — trades per volume bucket (default 50)

    Returns
    -------
    np.ndarray
        VPIN series, NaN for first bucket.

    Reference
    ---------
    Easley, Lopez de Prado, O'Hara (2012). "Flow Toxicity and Liquidity."
    """
    n = len(trade_vol)
    if n < n_bins:
        return np.full(n, np.nan)

    # Compute per-trade direction indicator
    direction = np.where(buy_vol > sell_vol, 1, -1)
    vol_cum = np.cumsum(trade_vol)
    bucket_size = vol_cum[-1] / (n // n_bins)

    vpin_arr = np.full(n, np.nan)
    bucket_edges = np.arange(bucket_size, vol_cum[-1], bucket_size)

    start = 0
    for i, edge in enumerate(bucket_edges):
        if edge >= vol_cum[-1]:
            break
        end = np.searchsorted(vol_cum, edge, side='right')
        if end <= start:
            continue
        bucket_buy = buy_vol[start:end].sum()
        bucket_sell = sell_vol[start:end].sum()
        bucket_total = trade_vol[start:end].sum()
        if bucket_total > 0:
            vpin_arr[end - 1] = abs(bucket_buy - bucket_sell) / bucket_total
        start = end

    return vpin_arr


def flow_toxicity(midprice: np.ndarray, trade_direction: np.ndarray,
                  trade_size: np.ndarray, window: int = 100) -> np.ndarray:
    """Flow Toxicity — adverse selection cost for market makers.

    Flow toxicity measures how much the midprice moves against the
    trade direction, normalised by trade size.

    Toxicity = sign(direction) * (midprice_{t} - midprice_{t-1}) / tick_size

    Llorca et al. variant: rolling covariance of midprice return and
    signed trade size.

    Parameters
    ----------
    midprice        : np.ndarray — best bid-ask midprice
    trade_direction : np.ndarray — +1 buy-initiated, -1 sell-initiated
    trade_size      : np.ndarray — trade volume
    window          : int        — rolling window (default 100)

    Returns
    -------
    np.ndarray
        Flow toxicity series.

    Reference
    ---------
    Llorca et al., "Flow Toxicity and Liquidity in a High-Frequency World."
    """
    # Signed trade volume
    signed_vol = trade_direction * trade_size

    # Midprice returns
    returns = np.diff(midprice, prepend=midprice[0])
    returns[0] = 0.0

    n = len(midprice)
    out = np.full(n, np.nan)

    for i in range(window, n):
        r = returns[i - window:i]
        sv = signed_vol[i - window:i]
        # Rolling covariance (ddof=0)
        r_mean = r.mean()
        sv_mean = sv.mean()
        cov = ((r - r_mean) * (sv - sv_mean)).sum() / window
        out[i] = cov

    return out


def spread_decomposition(midprice: np.ndarray, trade_price: np.ndarray,
                         direction: np.ndarray, window: int = 50) -> dict:
    """Decompose bid-ask spread into components.

    Returns a dict with:
    - 'realized_spread'   : realized spread ( против direction × price impact)
    - 'effective_spread'  : 2 × |trade_price - midprice|
    - 'price_impact'      : Kyle's lambda × trade direction

    Parameters
    ----------
    midprice    : np.ndarray
    trade_price : np.ndarray
    direction   : np.ndarray — +1 buy, -1 sell
    window      : int        — rolling window (default 50)

    Returns
    -------
    dict of np.ndarray

    Reference
    ---------
    Hasbrouck (2007). "Academic Literature."
    """
    half_spread = np.abs(trade_price - midprice) * 2.0

    n = len(midprice)
    realized = np.full(n, np.nan)
    price_impact = np.full(n, np.nan)

    for i in range(window, n):
        # Realized spread: half-spread minus price impact
        # Price impact approximated by rolling regression slope
        sign = direction[i]
        # Simple price impact: mean return following trade in direction
        future_ret = (midprice[i + 1:min(i + 5, n)] - midprice[i]) / midprice[i]
        if len(future_ret) > 0:
            price_impact[i] = sign * future_ret.mean()
            realized[i] = half_spread[i] - abs(price_impact[i])

    effective = half_spread

    return {
        'realized_spread': realized,
        'effective_spread': effective,
        'price_impact': price_impact,
    }


def order_flow_skew(order_flow: np.ndarray, window: int = 100) -> np.ndarray:
    """Order Flow Skewness — skewness of signed order flow.

    skew = E[(OF - μ)^3] / σ^3

    Positive skew means more extreme buy-initiated bursts.

    Parameters
    ----------
    order_flow : np.ndarray — signed order flow (+buy, -sell)
    window     : int        — rolling window (default 100)

    Returns
    -------
    np.ndarray
    """
    n = len(order_flow)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = order_flow[i - window:i].astype(float)
        m = seg.mean()
        s = seg.std()
        if s > 0:
            out[i] = ((seg - m) ** 3).mean() / (s ** 3)
    return out


def trade_direction(trade_price: np.ndarray, midprice: np.ndarray,
                    prev_trade_price: Optional[np.ndarray] = None) -> np.ndarray:
    """Lee-Ready tick rule for trade direction classification.

    Classifies each trade as buyer-initiated (+) or seller-initiated (-)
    based on whether trade price is above or below the midprice.
    If trade == midprice, uses previous trade price as tiebreaker.

    Parameters
    ----------
    trade_price        : np.ndarray
    midprice           : np.ndarray
    prev_trade_price   : np.ndarray, optional — previous trade price array

    Returns
    -------
    np.ndarray
        +1 for buy-initiated, -1 for sell-initiated, 0 for unknown.
    """
    n = len(trade_price)
    direction = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if trade_price[i] > midprice[i]:
            direction[i] = 1.0
        elif trade_price[i] < midprice[i]:
            direction[i] = -1.0
        else:
            # Tiebreaker: use previous trade price
            if prev_trade_price is not None and i > 0:
                if trade_price[i] > prev_trade_price[i - 1]:
                    direction[i] = 1.0
                elif trade_price[i] < prev_trade_price[i - 1]:
                    direction[i] = -1.0
            # else remains 0

    return direction


def quote_velocity(quote_change_count: np.ndarray, window: int = 60) -> np.ndarray:
    """Quote Update Velocity — number of BBO quote updates per second.

    Parameters
    ----------
    quote_change_count : np.ndarray — 1 if BBO changed at tick, 0 otherwise
    window             : int        — rolling window in seconds

    Returns
    -------
    np.ndarray
    """
    return _rolling_sum(quote_change_count.astype(float), window)


def trade_activity_ratio(trade_vol: np.ndarray, quoted_vol: np.ndarray,
                         window: int = 60) -> np.ndarray:
    """Trade Activity Ratio — trade volume / quoted (available) volume.

    TAR = Σ trade_vol / Σ quoted_vol

    Parameters
    ----------
    trade_vol : np.ndarray — traded volume per tick
    quoted_vol: np.ndarray — best bid + best ask volume per tick
    window    : int        — rolling window

    Returns
    -------
    np.ndarray
    """
    t = _rolling_sum(trade_vol.astype(float), window)
    q = _rolling_sum(quoted_vol.astype(float), window)
    return _safe_divide(t, q, fill=np.nan)


def price_impact(midprice: np.ndarray, trade_direction: np.ndarray,
                 trade_size: np.ndarray, window: int = 100) -> np.ndarray:
    """Kyle's Lambda — price impact per unit of order flow.

    ΔP = λ × OF  →  λ = ΔP / OF

    Estimated via rolling regression of midprice return on signed volume.

    Parameters
    ----------
    midprice        : np.ndarray
    trade_direction : np.ndarray — +1 buy, -1 sell
    trade_size      : np.ndarray
    window          : int        — rolling window (default 100)

    Returns
    -------
    np.ndarray
        Kyle's lambda series.
    """
    returns = np.diff(midprice, prepend=midprice[0])
    returns[0] = 0.0
    signed_vol = trade_direction * trade_size

    n = len(midprice)
    out = np.full(n, np.nan)

    for i in range(window, n):
        r = returns[i - window:i]
        sv = signed_vol[i - window:i]
        sv_mean = sv.mean()
        r_mean = r.mean()
        cov = ((r - r_mean) * (sv - sv_mean)).sum() / window
        var = (sv ** 2).mean() - sv_mean ** 2
        if var > 0:
            out[i] = cov / var

    return out


def midprice_ma_diff(midprice: np.ndarray, window: int = 20) -> np.ndarray:
    """Midprice vs Moving Average Spread.

    diff = midprice - MA(midprice, window)

    Parameters
    ----------
    midprice : np.ndarray
    window   : int

    Returns
    -------
    np.ndarray
    """
    ma = _rolling_mean(midprice.astype(float), window)
    return midprice.astype(float) - ma


def volume_curve_slope(volume_arr: np.ndarray, window: int = 60) -> np.ndarray:
    """Volume Curve Slope — linear trend slope of volume over window.

    Fit a line through the volume series; return the slope.
    Positive slope = increasing volume; negative = decreasing.

    Parameters
    ----------
    volume_arr : np.ndarray
    window     : int

    Returns
    -------
    np.ndarray
    """
    n = len(volume_arr)
    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = volume_arr[i - window:i].astype(float)
        seg_mean = seg.mean()
        cov = ((t - t_mean) * (seg - seg_mean)).sum()
        out[i] = cov / t_var if t_var > 0 else 0.0

    return out


def order_arrival_intensity(order_count_arr: np.ndarray, window: int = 60) -> np.ndarray:
    """Order Arrival Intensity — lambda from Poisson model.

    λ_t = # orders in window / window_duration

    Also known as order flow intensity (Hawkes baseline).

    Parameters
    ----------
    order_count_arr : np.ndarray — order arrivals (0/1 per tick or count per sec)
    window          : int        — window in ticks or seconds

    Returns
    -------
    np.ndarray
    """
    return _rolling_mean(order_count_arr.astype(float), window)


def price_reversion(midprice: np.ndarray, window: int = 20) -> np.ndarray:
    """Short-Term Price Reversion.

    Measures mean-reversion: current return vs future return.
    Positive = price tends to revert (reversal strategy signal).

    rev = - Σ_{k=1}^{window} sign(ret_{t-k}) × ret_{t-k} / window

    Parameters
    ----------
    midprice : np.ndarray
    window   : int

    Returns
    -------
    np.ndarray
    """
    returns = np.diff(midprice, prepend=midprice[0])
    returns[0] = 0.0

    n = len(midprice)
    out = np.full(n, 0.0)

    for i in range(window, n):
        seg = returns[i - window:i]
        out[i] = -np.mean(np.sign(seg) * seg)

    return out


def volatility_ratio(realized_vol: np.ndarray, implied_vol: np.ndarray,
                      window: int = 60) -> np.ndarray:
    """Realized Volatility / Implied Volatility ratio.

    RV / IV < 1 suggests implied > realized (overpriced vol).
    RV / IV > 1 suggests realized > implied (underpriced vol).

    Parameters
    ----------
    realized_vol : np.ndarray — rolling realized vol (e.g., Parkinson, Garman-Klass)
    implied_vol  : np.ndarray — IV series (e.g., ATM option IV)
    window       : int        — not used if inputs are already rolling

    Returns
    -------
    np.ndarray
    """
    return _safe_divide(realized_vol.astype(float),
                        implied_vol.astype(float), fill=np.nan)


def trade_size_imbalance(buy_trade_size: np.ndarray, sell_trade_size: np.ndarray,
                         window: int = 60) -> np.ndarray:
    """Trade Size Imbalance — ratio of average buy size to average sell size.

    TSI = MA(buy_size, window) / MA(sell_size, window)

    Parameters
    ----------
    buy_trade_size, sell_trade_size : np.ndarray — per-trade sizes
    window : int

    Returns
    -------
    np.ndarray
    """
    b = _rolling_mean(buy_trade_size.astype(float), window)
    s = _rolling_mean(sell_trade_size.astype(float), window)
    return _safe_divide(b, s, fill=np.nan)


def quote_cluster_quality(bid_prices: np.ndarray, ask_prices: np.ndarray,
                           tick_size: float = 0.01) -> np.ndarray:
    """Quote Clustering Quality — how tightly quotes cluster around round prices.

    QC = 1 - (std of (price % tick_size) around round prices) / tick_size

    High QC (>0.8) means quotes are evenly distributed; low QC means clustering.

    Parameters
    ----------
    bid_prices, ask_prices : np.ndarray
    tick_size              : float

    Returns
    -------
    np.ndarray
    """
    prices = np.concatenate([bid_prices, ask_prices])
    # Deviation from nearest round tick
    dev = np.abs(prices % tick_size)
    dev = np.minimum(dev, tick_size - dev)  # distance to nearest round tick
    n = len(bid_prices)
    quality = np.full(n, np.nan)
    for i in range(n):
        window_slice = prices[max(0, i - 50):i + 50]
        if len(window_slice) > 1:
            quality[i] = 1.0 - (window_slice.std() / (tick_size / 2))
    return quality


def liquidity_score(bid_vol: np.ndarray, ask_vol: np.ndarray,
                    spread: np.ndarray, window: int = 100) -> np.ndarray:
    """Composite Liquidity Score.

    Combines order imbalance, spread, and depth into a single score.

    L = (1 - spread_ratio) × depth_ratio × (1 - |OIR|)

    Normalised to [0, 1]; higher = more liquid.

    Parameters
    ----------
    bid_vol, ask_vol : np.ndarray
    spread           : np.ndarray — (ask - bid) / midprice
    window           : int

    Returns
    -------
    np.ndarray
    """
    b = _rolling_mean(bid_vol.astype(float), window)
    s = _rolling_mean(ask_vol.astype(float), window)
    total_depth = b + s
    depth_ratio = _safe_divide(np.minimum(b, s), total_depth, fill=0.5)

    sp = _rolling_mean(spread.astype(float), window)
    spread_ratio = _safe_divide(sp, sp.mean() if sp.mean() > 0 else 1.0, fill=0.0)
    spread_ratio = np.clip(spread_ratio, 0, 1)

    oir = order_imbalance(bid_vol, ask_vol, window)

    score = (1.0 - spread_ratio) * depth_ratio * (1.0 - np.abs(oir))
    return np.clip(score, 0, 1)
