"""
101 Formulaic Alphas — Python / pandas implementation.

Based on the published paper:
    "101 Formulaic Alphas" (Kakushadze, 2016)
    https://doi.org/10.1016/j.trf.2016.06.009

Each alpha is a pure formula producing a cross-sectional signal.
All alphas work on a DataFrame with the standard OHLCV columns:
    open, high, low, close, volume, returns  (returns = close.pct_change())

For use in a trading system, compute alphas on a panel of stocks and rank
or demean the output before portfolio construction.

Helper operators
----------------
rank(x)          — cross-sectional percentile rank (0–1)
ts_rank(x, d)    — time-series rank over lookback d
correlation(x, y, d)   — rolling Pearson correlation, window d
covariance(x, y, d)    — rolling covariance, window d
decay_linear(x, d)     — linearly-weighted moving average over d periods
delta(x, d)      — d-period difference  (x_t - x_{t-d})
ts_argmax(x, d)  — index of max value in last d time steps
ts_argmin(x, d)  — index of min value in last d time steps
ts_max(x, d)     — rolling max over d
ts_min(x, d)     — rolling min over d
delay(x, d)      — value d periods ago
adv(d)           — d-day average volume
vwap             — volume-weighted average price
signedpower(x, a) — x ** sign(x) * |x| ** a
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

__all__ = ["Alpha101"]


# ----------------------------------------------------------------------
# Helper operators
# ----------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that many alphas depend on."""
    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()
    if "vwap" not in df.columns:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    if "cap" not in df.columns:
        df["cap"] = df["close"]  # proxy
    for d in (5, 10, 20, 30, 60, 120):
        col = f"adv{d}"
        if col not in df.columns:
            df[col] = df["volume"].rolling(window=d, min_periods=1).mean()
    return df


def rank(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional percentile rank (0–1)."""
    if isinstance(x, pd.DataFrame):
        return x.rank(axis=1, pct=True)
    return x.rank(pct=True)


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Time-series rank: rank of current value vs. last d values."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling Pearson correlation over d periods."""
    return x.rolling(window=d, min_periods=d).corr(y)


def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling covariance over d periods."""
    return x.rolling(window=d, min_periods=d).cov(y)


def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """
    Linearly-weighted moving average over d periods.
    Most recent observation gets weight d, oldest gets weight 1.
    """
    result = pd.Series(index=x.index, dtype=np.float64)
    for i in range(d - 1, len(x)):
        window = x.iloc[i - d + 1 : i + 1]
        weights = np.arange(1, d + 1, dtype=float)
        result.iloc[i] = np.dot(window.values, weights) / weights.sum()
    return result


def delta(x: Union[pd.Series, pd.DataFrame], d: int) -> Union[pd.Series, pd.DataFrame]:
    """d-period difference."""
    return x.diff(d)


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based within window) of max value in last d periods."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: np.argmax(arr), raw=True
    )


def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based within window) of min value in last d periods."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: np.argmin(arr), raw=True
    )


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """Rolling maximum over d periods."""
    return x.rolling(window=d, min_periods=d).max()


def ts_min(x: pd.Series, d: int) -> pd.Series:
    """Rolling minimum over d periods."""
    return x.rolling(window=d, min_periods=d).min()


def delay(x: pd.Series, d: int) -> pd.Series:
    """Value d periods ago."""
    return x.shift(d)


def signedpower(x: pd.Series, a: float) -> pd.Series:
    """x^{sign(x) * a}."""
    return np.sign(x) * (np.abs(x) ** a)


def _adv(d: int, df: pd.DataFrame) -> pd.Series:
    """Average volume over last d days."""
    return df[f"adv{d}"]


def _vwap(df: pd.DataFrame) -> pd.Series:
    return df["vwap"]


def _cap(df: pd.DataFrame) -> pd.Series:
    return df["cap"]


# ----------------------------------------------------------------------
# Alpha101 class — computes all 101 alphas
# ----------------------------------------------------------------------

class Alpha101:
    """
    Compute all 101 formulaic alphas from Kakushadze (2016).

    Usage
    -----
        alpha_engine = Alpha101()
        df = alpha_engine.compute(df)           # adds all alpha_001 … alpha_101 columns
        df = alpha_engine.compute(df, names=["alpha_001", "alpha_003"])  # subset
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Individual alpha formulas (top-level helpers accept df)
    # ------------------------------------------------------------------

    @staticmethod
    def alpha_001(df: pd.DataFrame) -> pd.Series:
        """
        rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
        """
        r = df["returns"]
        cond = r < 0
        signed_power = np.where(cond, r.rolling(20).std(), df["close"])
        signed_power = signedpower(pd.Series(signed_power, index=df.index), 2.0)
        return rank(ts_argmax(signed_power, 5)) - 0.5

    @staticmethod
    def alpha_002(df: pd.DataFrame) -> pd.Series:
        """
        -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
        """
        vol = df["volume"]
        rank_delta_vol = rank(delta(np.log(vol), 2))
        rank_ret = rank((df["close"] - df["open"]) / df["open"])
        return -correlation(rank_delta_vol, rank_ret, 6)

    @staticmethod
    def alpha_003(df: pd.DataFrame) -> pd.Series:
        """
        -1 * correlation(rank(open), rank(volume), 10)
        """
        return -correlation(rank(df["open"]), rank(df["volume"]), 10)

    @staticmethod
    def alpha_004(df: pd.DataFrame) -> pd.Series:
        """
        -1 * Ts_Rank(rank(open), 10)
        """
        return -ts_rank(rank(df["open"]), 10)

    @staticmethod
    def alpha_005(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(open - sum(vwap, 10)/10) * rank(-abs(close - vwap))
        """
        vwap10 = _vwap(df).rolling(10).mean()
        rank1 = rank(df["open"] - vwap10)
        rank2 = rank(-(df["close"] - _vwap(df)).abs())
        return -rank1 * rank2

    @staticmethod
    def alpha_006(df: pd.DataFrame) -> pd.Series:
        """
        -1 * correlation(open, volume, 10) * rank(-abs(close - open))
        """
        corr = correlation(df["open"], df["volume"], 10)
        return -corr * rank((df["close"] - df["open"]).abs())

    @staticmethod
    def alpha_007(df: pd.DataFrame) -> pd.Series:
        """
        (adv20 < volume) ? -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1
        """
        cond = _adv(20, df) < df["volume"]
        delta7 = delta(df["close"], 7)
        tsr = ts_rank(delta7.abs(), 60) * np.sign(delta7)
        return pd.Series(np.where(cond, -tsr, -1), index=df.index)

    @staticmethod
    def alpha_008(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
        """
        sum_open_5 = df["open"].rolling(5).sum()
        sum_ret_5 = df["returns"].rolling(5).sum()
        product = sum_open_5 * sum_ret_5
        return -rank(product - delay(product, 10))

    @staticmethod
    def alpha_009(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : (ts_max(delta(close, 1), 5) < 0) ? -delta(close, 1) : -delta(close, 1)
        """
        delta1 = delta(df["close"], 1)
        tsmin5 = ts_min(delta1, 5)
        tsmax5 = ts_max(delta1, 5)
        result = pd.Series(index=df.index, dtype=np.float64)
        result = np.where(tsmin5 > 0, delta1,
                          np.where(tsmax5 < 0, -delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_010(df: pd.DataFrame) -> pd.Series:
        """
        rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
               ((ts_max(delta(close, 1), 4) < 0) ? -delta(close, 1) : -delta(close, 1))))
        """
        delta1 = delta(df["close"], 1)
        tsmin4 = ts_min(delta1, 4)
        tsmax4 = ts_max(delta1, 4)
        result = np.where(tsmin4 > 0, delta1,
                          np.where(tsmax4 < 0, -delta1, -delta1))
        return rank(pd.Series(result, index=df.index))

    @staticmethod
    def alpha_011(df: pd.DataFrame) -> pd.Series:
        """
        (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))
        """
        diff = _vwap(df) - df["close"]
        return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(df["volume"], 3))

    @staticmethod
    def alpha_012(df: pd.DataFrame) -> pd.Series:
        """
        sign(delta(volume, 1)) * -1 * delta(close, 1)
        """
        return np.sign(delta(df["volume"], 1)) * (-delta(df["close"], 1))

    @staticmethod
    def alpha_013(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(covariance(rank(close), rank(volume), 5))
        """
        return -rank(covariance(rank(df["close"]), rank(df["volume"]), 5))

    @staticmethod
    def alpha_014(df: pd.DataFrame) -> pd.Series:
        """
        -1 * correlation(rank(high), rank(volume), 5).  Use min(0.5) floor.
        """
        corr = correlation(rank(df["high"]), rank(df["volume"]), 5)
        return -corr.clip(lower=0.5)

    @staticmethod
    def alpha_015(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ((0.5 < rank(sum.mean(delta(close, 1), 32) * (sum(returns, 32) / adv20)))) * -1
        """
        mean_delta = delta(df["close"], 1).rolling(32).mean()
        return_ratio = df["returns"].rolling(32).sum() / _adv(20, df)
        inner = mean_delta * return_ratio
        return -1 * (rank(inner) > 0.5).astype(float)

    @staticmethod
    def alpha_016(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((stddev(abs((close - open)), 20) + (close - open)) + correlation(close, open, 20)))
        """
        abs_diff = (df["close"] - df["open"]).abs()
        std20 = abs_diff.rolling(20).std()
        term = std20 + (df["close"] - df["open"])
        corr = correlation(df["close"], df["open"], 20)
        return -rank(term + corr)

    @staticmethod
    def alpha_017(df: pd.DataFrame) -> pd.Series:
        """
        ((-1 * rank((open - delay(high, 5)))) * rank((open - delay(close, 5)))) * rank((open - delay(low, 5))))
        """
        r1 = rank(df["open"] - delay(df["high"], 5))
        r2 = rank(df["open"] - delay(df["close"], 5))
        r3 = rank(df["open"] - delay(df["low"], 5))
        return -r1 * r2 * r3

    @staticmethod
    def alpha_018(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * rank((stddev(high, 10) + correlation(high, volume, 10)))) +
        (0.5 < rank((sum(high, 10) / 10))) ? -1 : -1
        """
        std_h = df["high"].rolling(10).std()
        corr_hv = correlation(df["high"], df["volume"], 10)
        term1 = -rank(std_h + corr_hv)
        avg_h = df["high"].rolling(10).mean()
        term2 = (rank(avg_h) > 0.5).astype(float)
        return term1 + term2

    @staticmethod
    def alpha_019(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * rank((close - delay(close, 5)))) * rank(correlation(adv20, low, 5)) * rank((close / open))
        """
        rank1 = rank(delta(df["close"], 5))
        corr_adv = correlation(_adv(20, df), df["low"], 5)
        close_open = df["close"] / df["open"]
        return -rank1 * corr_adv * rank(close_open)

    @staticmethod
    def alpha_020(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * correlation(rank(high), rank(volume), 5)) * rank(-abs(delta(close, 5)))
        """
        corr = correlation(rank(df["high"]), rank(df["volume"]), 5)
        return -corr * rank(-delta(df["close"], 5).abs())

    @staticmethod
    def alpha_021(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * ((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
               ((ts_max(delta(close, 1), 4) < 0) ? -delta(close, 1) : -delta(close, 1))))
        """
        delta1 = delta(df["close"], 1)
        tsmin4 = ts_min(delta1, 4)
        tsmax4 = ts_max(delta1, 4)
        result = np.where(tsmin4 > 0, delta1,
                         np.where(tsmax4 < 0, -delta1, -delta1))
        return pd.Series(-result, index=df.index)

    @staticmethod
    def alpha_022(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank((0 < ts_max((close - open)/open, 3))) *
            rank(correlation((close - open)/open, adv20, 10)) *
            rank((close - open)/open)
        """
        ret_osc = (df["close"] - df["open"]) / df["open"]
        ts_max_ret = ts_max(ret_osc, 3)
        cond = ts_max_ret > 0
        corr_ret_adv = correlation(ret_osc, _adv(20, df), 10)
        r1 = rank(np.where(cond, 1, 0))
        r2 = rank(corr_ret_adv)
        r3 = rank(ret_osc)
        return -r1 * r2 * r3

    @staticmethod
    def alpha_023(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 2)) ? -delta(close, 1) :
        ((0 < ts_max(delta(close, 1), 2)) ? delta(close, 1) : -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin2 = ts_min(delta1, 2)
        tsmax2 = ts_max(delta1, 2)
        result = np.where(tsmin2 > 0, -delta1,
                          np.where(tsmax2 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_024(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((sum(high, 20) / 20) - close)) * rank(correlation(high, volume, 5))
        """
        avg_h = df["high"].rolling(20).mean()
        rank1 = rank(avg_h - df["close"])
        corr = correlation(df["high"], df["volume"], 5)
        rank2 = rank(corr)
        return -rank1 * rank2

    @staticmethod
    def alpha_025(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 4)) ? -delta(close, 1) :
        (0 < ts_max(delta(close, 1), 4)) ? delta(close, 1) : -delta(close, 1)
        """
        delta1 = delta(df["close"], 1)
        tsmin4 = ts_min(delta1, 4)
        tsmax4 = ts_max(delta1, 4)
        result = np.where(tsmin4 > 0, -delta1,
                          np.where(tsmax4 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_026(df: pd.DataFrame) -> pd.Series:
        """
        -1 * delta((((close - low) - (high - close)) / (high - low)), 1) * rank((volume / adv20))
        """
        hl_diff = df["high"] - df["low"]
        numerator = (df["close"] - df["low"]) - hl_diff
        osc = numerator / (hl_diff + 1e-9)
        delta_osc = delta(osc, 1)
        vol_ratio = df["volume"] / (_adv(20, df) + 1e-9)
        return -delta_osc * rank(vol_ratio)

    @staticmethod
    def alpha_027(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ((-1 * low) * open ** 0.5 - vwap * (open ** 0.5) / low) ** 2
        """
        term = (-df["low"]) * np.sqrt(df["open"]) - _vwap(df) * np.sqrt(df["open"]) / (df["low"] + 1e-9)
        return -(term ** 2)

    @staticmethod
    def alpha_028(df: pd.DataFrame) -> pd.Series:
        """
        -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)), 6),
                          rank(volume), 6)
        """
        min_low = ts_min(df["low"], 12)
        max_high = ts_max(df["high"], 12)
        normalized = (df["close"] - min_low) / (max_high - min_low + 1e-9)
        rank_norm = rank(normalized)
        corr = correlation(rank_norm, rank(df["volume"]), 6)
        return -corr

    @staticmethod
    def alpha_029(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation((close - high), low, 15)) +
              rank((-1 * delta(((close + high) / 2), 20))))
        """
        corr = correlation(df["close"] - df["high"], df["low"], 15)
        delta_mid = delta((df["close"] + df["high"]) / 2.0, 20)
        return -(rank(corr) + rank(-delta_mid))

    @staticmethod
    def alpha_030(df: pd.DataFrame) -> pd.Series:
        """
        -1 * Ts_Rank(correlation(rank(low), rank(volume), 7), 3) * ts_rank(delta(delta(close, 1), 1), 6)
        """
        corr = correlation(rank(df["low"]), rank(df["volume"]), 7)
        tsr1 = ts_rank(corr, 3)
        delta2 = delta(delta(df["close"], 1), 1)
        tsr2 = ts_rank(delta2, 6)
        return -(tsr1 * tsr2)

    @staticmethod
    def alpha_031(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * rank(correlation(sum(((open * 0.85) + (vwap * 0.15)), 20), sum(adv20, 15), 6))) *
        rank(delta(((close - open) / open), 3))
        """
        blend = df["open"] * 0.85 + _vwap(df) * 0.15
        sum_blend = blend.rolling(20).sum()
        sum_adv = _adv(20, df).rolling(15).sum()
        corr = correlation(sum_blend, sum_adv, 6)
        delta_ret = delta((df["close"] - df["open"]) / (df["open"] + 1e-9), 3)
        return -rank(corr) * rank(delta_ret)

    @staticmethod
    def alpha_032(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ((rank((1 / close)) * rank(correlation(vwap, sum(adv20, 49), 15))) *
              rank((ts_min(delta(close, 1), 4) / delta(close, 1))))
        """
        rank1 = rank(1 / df["close"])
        corr = correlation(_vwap(df), _adv(20, df).rolling(49).sum(), 15)
        rank2 = rank(corr)
        delta1 = delta(df["close"], 1)
        ratio = ts_min(delta1, 4) / (delta1 + 1e-9)
        rank3 = rank(ratio)
        return -rank1 * rank2 * rank3

    @staticmethod
    def alpha_033(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(close, sum(adv30, 37), 15)) *
               rank(delta((close / vwap), 5)))
        """
        corr = correlation(df["close"], _adv(30, df).rolling(37).sum(), 15)
        delta_vwap = delta(df["close"] / (_vwap(df) + 1e-9), 5)
        return -rank(corr) * rank(delta_vwap)

    @staticmethod
    def alpha_034(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 2)) ?
            -delta(close, 1) :
            ((0 < ts_max(delta(close, 1), 2)) ?
                delta(close, 1) :
                -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin2 = ts_min(delta1, 2)
        tsmax2 = ts_max(delta1, 2)
        result = np.where(tsmin2 > 0, -delta1,
                          np.where(tsmax2 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_035(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((sum(close, 30) / 30) - close)) * rank(correlation((close * 0.5 + vwap * 0.5), adv30, 3))
        """
        avg_close = df["close"].rolling(30).mean()
        rank1 = rank(avg_close - df["close"])
        blend = 0.5 * df["close"] + 0.5 * _vwap(df)
        corr = correlation(blend, _adv(30, df), 3)
        rank2 = rank(corr)
        return -rank1 * rank2

    @staticmethod
    def alpha_036(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(((high * 0.9) + (close * 0.1)), adv30, 10)) *
               rank(correlation(ts_min(((low * 0.9) + (close * 0.1)), 2), low, 3))
        """
        blend_high = 0.9 * df["high"] + 0.1 * df["close"]
        blend_low = 0.9 * df["low"] + 0.1 * df["close"]
        corr1 = correlation(blend_high, _adv(30, df), 10)
        corr2 = correlation(ts_min(blend_low, 2), df["low"], 3)
        return -rank(corr1) * rank(corr2)

    @staticmethod
    def alpha_037(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(((close / open) - 1), 1), 5)) ?
            -delta(close, 1) :
            ((ts_max(delta(((close / open) - 1), 1), 5) < 0) ?
                delta(close, 1) :
                -delta(close, 1))
        """
        ret_osc = df["close"] / (df["open"] + 1e-9) - 1
        delta_osc = delta(ret_osc, 1)
        tsmin5 = ts_min(delta_osc, 5)
        tsmax5 = ts_max(delta_osc, 5)
        delta1 = delta(df["close"], 1)
        result = np.where(tsmin5 > 0, -delta1,
                          np.where(tsmax5 < 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_038(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ts_rank(correlation(rank(low), rank(adv10), 7), 3) *
              ts_rank(delta(delta(close, 1), 1), 10)
        """
        corr = correlation(rank(df["low"]), rank(_adv(10, df)), 7)
        tsr1 = ts_rank(corr, 3)
        delta2 = delta(delta(df["close"], 1), 1)
        tsr2 = ts_rank(delta2, 10)
        return -(tsr1 * tsr2)

    @staticmethod
    def alpha_039(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((vwap - close) / ts_max(((close - vwap) / vwap), 3))) *
              rank(correlation(vwap, adv180, 18))
        """
        diff = _vwap(df) - df["close"]
        max_rel = ts_max(((df["close"] - _vwap(df)) / (_vwap(df) + 1e-9)).abs(), 3)
        rank1 = rank(diff / (max_rel + 1e-9))
        corr = correlation(_vwap(df), _adv(180, df), 18)
        rank2 = rank(corr)
        return -rank1 * rank2

    @staticmethod
    def alpha_040(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 3)), 8)
        """
        vol_ratio = df["volume"] / (_adv(20, df) + 1e-9)
        tsr1 = ts_rank(vol_ratio, 20)
        delta3 = -delta(df["close"], 3)
        tsr2 = ts_rank(delta3, 8)
        return -(tsr1 * tsr2)

    @staticmethod
    def alpha_041(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(sum(((open * 0.7) + (low * 0.3)), 20), sum(adv60, 20), 6)) *
              rank((open - close))
        """
        blend = 0.7 * df["open"] + 0.3 * df["low"]
        sum_blend = blend.rolling(20).sum()
        sum_adv = _adv(60, df).rolling(20).sum()
        corr = correlation(sum_blend, sum_adv, 6)
        return -rank(corr) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_042(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank((close - ts_max(high, 15))) + rank((close - ts_min(low, 15)))) *
              rank(correlation(close, adv10, 5))
        """
        rank1 = rank(df["close"] - ts_max(df["high"], 15))
        rank2 = rank(df["close"] - ts_min(df["low"], 15))
        corr = correlation(df["close"], _adv(10, df), 5)
        return -(rank1 + rank2) * rank(corr)

    @staticmethod
    def alpha_043(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(((high * 0.9) + (close * 0.1)), sum(adv5, 15), 10)) *
              rank((ts_min(delta(close, 1), 3) / delta(close, 1)))
        """
        blend = 0.9 * df["high"] + 0.1 * df["close"]
        corr = correlation(blend, _adv(5, df).rolling(15).sum(), 10)
        delta1 = delta(df["close"], 1)
        ratio = ts_min(delta1, 3) / (delta1 + 1e-9)
        return -rank(corr) * rank(ratio)

    @staticmethod
    def alpha_044(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(sum(low, 5), sum(adv5, 20), 5)) *
              rank(-1 + (close / open)))
        """
        corr = correlation(df["low"].rolling(5).sum(), _adv(5, df).rolling(20).sum(), 5)
        close_open_ratio = -1 + df["close"] / (df["open"] + 1e-9)
        return -rank(corr) * rank(close_open_ratio)

    @staticmethod
    def alpha_045(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(rank(open), rank(volume), 10)) * rank(-abs(delta(close, 5)))
        """
        corr = correlation(rank(df["open"]), rank(df["volume"]), 10)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_046(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(sum(((high + low) / 2), 20), sum(adv40, 15), 7)) +
              rank(-abs(delta(close, 5))))
        """
        mid = (df["high"] + df["low"]) / 2.0
        corr = correlation(mid.rolling(20).sum(), _adv(40, df).rolling(15).sum(), 7)
        delta5 = -delta(df["close"], 5).abs()
        return -(rank(corr) + rank(delta5))

    @staticmethod
    def alpha_047(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 4)) ? -delta(close, 1) :
        (0 < ts_max(delta(close, 1), 4)) ? delta(close, 1) : -delta(close, 1)
        """
        delta1 = delta(df["close"], 1)
        tsmin4 = ts_min(delta1, 4)
        tsmax4 = ts_max(delta1, 4)
        result = np.where(tsmin4 > 0, -delta1,
                          np.where(tsmax4 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_048(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((stddev((close / open), 20) + (1 - (close / open))) +
                   correlation((close / open), adv30, 15)))
        """
        close_open = df["close"] / (df["open"] + 1e-9)
        std20 = (close_open).rolling(20).std()
        inner = std20 + (1 - close_open)
        corr = correlation(close_open, _adv(30, df), 15)
        return -rank(inner + corr)

    @staticmethod
    def alpha_049(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ((rank((1 - rank((close / open)))) + sign(delta(close, 1))) *
               (1 + rank((close / open))))
        """
        close_open = df["close"] / (df["open"] + 1e-9)
        rank1 = rank(1 - rank(close_open))
        sign_delta = np.sign(delta(df["close"], 1))
        term1 = rank1 + sign_delta
        term2 = 1 + rank(close_open)
        return -(term1 * term2)

    @staticmethod
    def alpha_050(df: pd.DataFrame) -> pd.Series:
        """
        ((-1 * rank(delta(((close / open) - 1), 3))) *
         correlation(rank(volume), rank(vwap), 15))
        """
        delta_ret = delta(df["close"] / (df["open"] + 1e-9) - 1, 3)
        corr = correlation(rank(df["volume"]), rank(_vwap(df)), 15)
        return -rank(delta_ret) * corr

    @staticmethod
    def alpha_051(df: pd.DataFrame) -> pd.Series:
        """
        ((-1 * rank(delta(vwap, 4))) *
         ts_rank(correlation(rank(close), rank(volume), 10), 10))
        """
        delta_vwap = -delta(_vwap(df), 4)
        corr = correlation(rank(df["close"]), rank(df["volume"]), 10)
        tsr = ts_rank(corr, 10)
        return -rank(delta_vwap) * tsr

    @staticmethod
    def alpha_052(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ts_max(rank(correlation(rank(low), rank(adv20), 8)), 3) *
              rank(-abs(delta(close, 5)))
        """
        corr = correlation(rank(df["low"]), rank(_adv(20, df)), 8)
        ts_max_rank = ts_max(rank(corr), 3)
        delta5 = -delta(df["close"], 5).abs()
        return -ts_max_rank * rank(delta5)

    @staticmethod
    def alpha_053(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(delta(((high + low) / 2), 4)) * correlation(rank(volume), rank((high + low)/2), 6)
        """
        mid = (df["high"] + df["low"]) / 2.0
        delta_mid = -delta(mid, 4)
        corr = correlation(rank(df["volume"]), rank(mid), 6)
        return -rank(delta_mid) * corr

    @staticmethod
    def alpha_054(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(sum(((low * 0.7) + (vwap * 0.3)), 20), sum(adv40, 15), 7)) *
              rank(correlation(rank(low), rank(volume), 6))
        """
        blend_low = 0.7 * df["low"] + 0.3 * _vwap(df)
        corr1 = correlation(blend_low.rolling(20).sum(), _adv(40, df).rolling(15).sum(), 7)
        corr2 = correlation(rank(df["low"]), rank(df["volume"]), 6)
        return -rank(corr1) * rank(corr2)

    @staticmethod
    def alpha_055(df: pd.DataFrame) -> pd.Series:
        """
        (0 - (2 * rank((((open - close) / open) - ts_min(((open - close) / open), 12)) *
                      (rank(rank((close / open)))))))) -
        (1 + rank(correlation(adv30, low, 5)))
        """
        close_open = df["close"] / (df["open"] + 1e-9)
        ret_minus_min = close_open - ts_min(close_open, 12)
        term1 = 2 * rank(ret_minus_min * rank(rank(close_open)))
        corr = correlation(_adv(30, df), df["low"], 5)
        term2 = 1 + rank(corr)
        return -(term1 - term2)

    @staticmethod
    def alpha_056(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(((high * 0.8) + (vwap * 0.2)), adv15, 8)) *
              rank(-abs(delta(close, 5))))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(15, df), 8)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_057(df: pd.DataFrame) -> pd.Series:
        """
        (0 - (1 * rank((open - ts_min(open, 12))))) *
        (0 - (1 * rank((ts_min(delta(close, 1), 3) / delta(close, 1))))) *
        (0 - (1 * rank(correlation(adv20, low, 5).rank())))) *
        (0 - (1 * rank(-delta(close, 1))))
        """
        rank1 = rank(open - ts_min(df["open"], 12))
        delta1 = delta(df["close"], 1)
        ratio = ts_min(delta1, 3) / (delta1 + 1e-9)
        rank2 = rank(ratio)
        corr = correlation(_adv(20, df), df["low"], 5)
        rank3 = rank(corr)
        rank4 = rank(-delta1)
        return -rank1 * -rank2 * -rank3 * -rank4

    @staticmethod
    def alpha_058(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (0 - (1 * rank((sum(close, 30) / 30 - close)))) *
        (1 + rank(correlation((sum(close, 20) / 20), sum(adv30, 20), 9)))
        """
        avg_close_30 = df["close"].rolling(30).mean()
        term1 = 1 * rank(avg_close_30 - df["close"])
        avg_close_20 = df["close"].rolling(20).mean()
        corr = correlation(avg_close_20, _adv(30, df).rolling(20).sum(), 9)
        term2 = 1 + rank(corr)
        return -term1 * term2

    @staticmethod
    def alpha_059(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta((((close / open) - 1) / open), 9)))) *
        (0 - (1 * rank(correlation(vwap, sum(adv10, 10), 10)))))
        """
        close_open = (df["close"] / (df["open"] + 1e-9) - 1) / (df["open"] + 1e-9)
        delta_osc = delta(close_open, 9)
        rank1 = rank(delta_osc)
        corr = correlation(_vwap(df), _adv(10, df).rolling(10).sum(), 10)
        rank2 = rank(corr)
        return -rank1 * -(rank2)

    @staticmethod
    def alpha_060(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(rank(((high * 0.8) + (vwap * 0.2))), rank(volume), 5)) *
              rank(delta(delta(close, 1), 1))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(rank(blend), rank(df["volume"]), 5)
        delta2 = delta(delta(df["close"], 1), 1)
        return -rank(corr) * rank(delta2)

    @staticmethod
    def alpha_061(df: pd.DataFrame) -> pd.Series:
        """
        ((0 < ts_min(delta(close, 1), 2)) ? -delta(close, 1) :
         ((0 < ts_max(delta(close, 1), 2)) ? delta(close, 1) : -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin2 = ts_min(delta1, 2)
        tsmax2 = ts_max(delta1, 2)
        result = np.where(tsmin2 > 0, -delta1,
                          np.where(tsmax2 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_062(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(((0.5 < rank((close - ts_min(low, 25)))) -
                    (0.5 < rank(ts_max(high, 25))))) *
                   (volume / adv20))
        """
        rank1 = rank(df["close"] - ts_min(df["low"], 25))
        rank2 = rank(ts_max(df["high"], 25))
        vol_ratio = df["volume"] / (_adv(20, df) + 1e-9)
        return -rank((rank1 - rank2)) * vol_ratio

    @staticmethod
    def alpha_063(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta((((close / open) - 1) / open), 3)))) *
        ts_rank(correlation(rank(volume), rank(vwap), 5), 3)
        """
        close_open = (df["close"] / (df["open"] + 1e-9) - 1) / (df["open"] + 1e-9)
        delta_osc = delta(close_open, 3)
        rank1 = rank(delta_osc)
        corr = correlation(rank(df["volume"]), rank(_vwap(df)), 5)
        tsr = ts_rank(corr, 3)
        return -rank1 * tsr

    @staticmethod
    def alpha_064(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank((close - ts_min(low, 6))) *
        rank(correlation((close / open), adv10, 5)) *
        rank((close - ts_min(low, 6)))
        """
        diff = df["close"] - ts_min(df["low"], 6)
        corr = correlation(df["close"] / (df["open"] + 1e-9), _adv(10, df), 5)
        return -rank(diff) * rank(corr) * rank(diff)

    @staticmethod
    def alpha_065(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(rank(((high * 0.8) + (vwap * 0.2))), rank(volume), 5)) *
              rank(-abs(delta(close, 5)))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(rank(blend), rank(df["volume"]), 5)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_066(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(sum(((low * 0.8) + (vwap * 0.2)), 20), sum(adv40, 20), 6))) *
        rank(-abs(delta(close, 5)))
        """
        blend = 0.8 * df["low"] + 0.2 * _vwap(df)
        corr = correlation(blend.rolling(20).sum(), _adv(40, df).rolling(20).sum(), 6)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_067(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(((low * 0.8) + (vwap * 0.2)), sum(adv30, 15), 20))) *
        rank((close - ts_min(low, 15)) / (close - ts_max(high, 15)))
        """
        blend = 0.8 * df["low"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(30, df).rolling(15).sum(), 20)
        numerator = df["close"] - ts_min(df["low"], 15)
        denominator = df["close"] - ts_max(df["high"], 15) + 1e-9
        ratio = numerator / denominator
        return -rank(corr) * rank(ratio)

    @staticmethod
    def alpha_068(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(rank(low), rank(adv30), 15))) *
        (close - ts_min(close, 15)) / (close - ts_max(close, 15))
        """
        corr = correlation(rank(df["low"]), rank(_adv(30, df)), 15)
        numerator = df["close"] - ts_min(df["close"], 15)
        denominator = df["close"] - ts_max(df["close"], 15) + 1e-9
        ratio = numerator / denominator
        return -rank(corr) * ratio

    @staticmethod
    def alpha_069(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta(vwap, 5))) * ts_rank(correlation(rank(close), rank(volume), 5), 3)
        """
        delta_vwap = -delta(_vwap(df), 5)
        corr = correlation(rank(df["close"]), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(delta_vwap) * tsr

    @staticmethod
    def alpha_070(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(ts_max(delta(((low * 0.8) + (vwap * 0.2)), 2), 4)) *
              rank(correlation(rank(vwap), rank(volume), 5))
        """
        blend = 0.8 * df["low"] + 0.2 * _vwap(df)
        ts_max_delta = ts_max(delta(blend, 2), 4)
        corr = correlation(rank(_vwap(df)), rank(df["volume"]), 5)
        return -rank(ts_max_delta) * rank(corr)

    @staticmethod
    def alpha_071(df: pd.DataFrame) -> pd.Series:
        """
        ((0 < ts_min(delta(close, 1), 4)) ? -delta(close, 1) :
         ((0 < ts_max(delta(close, 1), 4)) ? delta(close, 1) : -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin4 = ts_min(delta1, 4)
        tsmax4 = ts_max(delta1, 4)
        result = np.where(tsmin4 > 0, -delta1,
                          np.where(tsmax4 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_072(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank((ts_min(delta(close, 1), 4) / delta(close, 1))) *
              rank(correlation(adv30, low, 5))
        """
        delta1 = delta(df["close"], 1)
        ratio = ts_min(delta1, 4) / (delta1 + 1e-9)
        corr = correlation(_adv(30, df), df["low"], 5)
        return -rank(ratio) * rank(corr)

    @staticmethod
    def alpha_073(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 3)) ? -delta(close, 1) :
        ((0 < ts_max(delta(close, 1), 3)) ? delta(close, 1) : -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin3 = ts_min(delta1, 3)
        tsmax3 = ts_max(delta1, 3)
        result = np.where(tsmin3 > 0, -delta1,
                          np.where(tsmax3 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_074(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(delta((((high + low) / 2) * 0.8 + (vwap * 0.2)), 2)) *
              rank(correlation(rank(volume), rank(((high + low) / 2)), 6))
        """
        mid = (df["high"] + df["low"]) / 2.0
        blend = 0.8 * mid + 0.2 * _vwap(df)
        delta_blend = delta(blend, 2)
        corr = correlation(rank(df["volume"]), rank(mid), 6)
        return -rank(delta_blend) * rank(corr)

    @staticmethod
    def alpha_075(df: pd.DataFrame) -> pd.Series:
        """
        (0 < ts_min(delta(close, 1), 2)) ? -delta(close, 1) :
        ((0 < ts_max(delta(close, 1), 2)) ? delta(close, 1) : -delta(close, 1))
        """
        delta1 = delta(df["close"], 1)
        tsmin2 = ts_min(delta1, 2)
        tsmax2 = ts_max(delta1, 2)
        result = np.where(tsmin2 > 0, -delta1,
                          np.where(tsmax2 > 0, delta1, -delta1))
        return pd.Series(result, index=df.index)

    @staticmethod
    def alpha_076(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(delta((((high + low) / 2) / delay(close, 3) - 1) * 0.5 +
                   (close / delay(close, 3) - 1) * 0.5)) *
              ts_rank(correlation(rank(volume), rank(((high + low) / 2)), 5), 3)
        """
        mid = (df["high"] + df["low"]) / 2.0
        term1 = (mid / delay(df["close"], 3) - 1) * 0.5
        term2 = (df["close"] / delay(df["close"], 3) - 1) * 0.5
        delta_term = delta(term1 + term2, 1)
        corr = correlation(rank(df["volume"]), rank(mid), 5)
        tsr = ts_rank(corr, 3)
        return -rank(delta_term) * tsr

    @staticmethod
    def alpha_077(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(((high * 0.8) + (vwap * 0.2)), sum(adv10, 10), 5)))) *
        rank(-abs(delta(close, 5)))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(10, df).rolling(10).sum(), 5)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_078(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(sum(((low * 0.8) + (vwap * 0.2)), 20), sum(adv30, 20), 9)) *
              rank(-abs(delta(close, 5))))
        """
        blend = 0.8 * df["low"] + 0.2 * _vwap(df)
        corr = correlation(blend.rolling(20).sum(), _adv(30, df).rolling(20).sum(), 9)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_079(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(((high * 0.9) + (close * 0.1)), sum(adv5, 10), 10)))) *
        rank((close / open) - 1)
        """
        blend = 0.9 * df["high"] + 0.1 * df["close"]
        corr = correlation(blend, _adv(5, df).rolling(10).sum(), 10)
        close_open = df["close"] / (df["open"] + 1e-9) - 1
        return -rank(corr) * rank(close_open)

    @staticmethod
    def alpha_080(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank((close - ts_min(low, 15)))) *
        (0 - rank(correlation(((high + low + close) / 3), adv30, 5))) *
        (0 - rank(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5))))
        """
        rank1 = rank(df["close"] - ts_min(df["low"], 15))
        mid = (df["high"] + df["low"] + df["close"]) / 3.0
        corr1 = correlation(mid, _adv(30, df), 5)
        rank2 = rank(corr1)
        tsr_vol = ts_rank(df["volume"], 5)
        tsr_high = ts_rank(df["high"], 5)
        corr2 = correlation(tsr_vol, tsr_high, 5)
        rank3 = rank(corr2)
        return -rank1 * -rank2 * -rank3

    @staticmethod
    def alpha_081(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)))) *
        (0 - rank(correlation(low, volume, 5)))
        """
        mid = (df["high"] + df["low"]) / 2.0
        corr1 = correlation(mid.rolling(20).sum(), _adv(60, df).rolling(20).sum(), 9)
        corr2 = correlation(df["low"], df["volume"], 5)
        return -rank(corr1) * -rank(corr2)

    @staticmethod
    def alpha_082(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(((high * 0.8) + (vwap * 0.2)), sum(adv15, 15), 20)))) *
        rank(-abs(delta(close, 5)))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(15, df).rolling(15).sum(), 20)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_083(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(rank(((high * 0.8) + (vwap * 0.2))), rank(volume), 5)))) *
        ts_rank(correlation(close, open, 5), 3)
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr1 = correlation(rank(blend), rank(df["volume"]), 5)
        corr2 = correlation(df["close"], df["open"], 5)
        tsr = ts_rank(corr2, 3)
        return -rank(corr1) * tsr

    @staticmethod
    def alpha_084(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank((delay(close, 20) - ts_min(low, 20))) *
        (rank(rank(volume)) / rank(correlation(rank(high), rank(vwap), 11)))
        """
        diff = delay(df["close"], 20) - ts_min(df["low"], 20)
        rank1 = rank(diff)
        rank_vol = rank(rank(df["volume"]))
        corr = correlation(rank(df["high"]), rank(_vwap(df)), 11)
        rank2 = rank(corr)
        return -rank1 * (rank_vol / (rank2 + 1e-9))

    @staticmethod
    def alpha_085(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank((close - ts_min(low, 5)))) *
        ts_rank(correlation(rank(close), rank(volume), 5), 3)
        """
        diff = df["close"] - ts_min(df["low"], 5)
        corr = correlation(rank(df["close"]), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(diff) * tsr

    @staticmethod
    def alpha_086(df: pd.DataFrame) -> pd.Series:
        """
        -1 * ts_max(rank(correlation(rank(open), rank(volume), 10)), 3) *
              rank(-abs(delta(close, 5)))
        """
        corr = correlation(rank(df["open"]), rank(df["volume"]), 10)
        ts_max_rank = ts_max(rank(corr), 3)
        delta5 = -delta(df["close"], 5).abs()
        return -ts_max_rank * rank(delta5)

    @staticmethod
    def alpha_087(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(rank(high), rank(volume), 5)) *
              ts_rank(correlation(close, open, 5), 3)
        """
        corr1 = correlation(rank(df["high"]), rank(df["volume"]), 5)
        corr2 = correlation(df["close"], df["open"], 5)
        tsr = ts_rank(corr2, 3)
        return -rank(corr1) * tsr

    @staticmethod
    def alpha_088(df: pd.DataFrame) -> pd.Series:
        """
        (-1 * rank(rank(delta((((close / open) - 1) / open), 10))))) *
        (rank(rank((volume / adv20)))) *
        (rank(rank(delta(close, 3))))
        """
        close_open = (df["close"] / (df["open"] + 1e-9) - 1) / (df["open"] + 1e-9)
        delta_osc = delta(close_open, 10)
        rank1 = rank(rank(delta_osc))
        vol_ratio = rank(rank(df["volume"] / (_adv(20, df) + 1e-9)))
        delta_close = rank(rank(delta(df["close"], 3)))
        return -rank1 * vol_ratio * delta_close

    @staticmethod
    def alpha_089(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(ts_min(delta(close, 1), 3))) *
        (0 - rank(ts_rank(((vwap / (high - low)) - 1), 15))))
        """
        ts_min_delta = ts_min(delta(df["close"], 1), 3)
        rank1 = rank(ts_min_delta)
        vwap_osc = _vwap(df) / (df["high"] - df["low"] + 1e-9) - 1
        tsr = ts_rank(vwap_osc, 15)
        rank2 = rank(tsr)
        return -rank1 * -rank2

    @staticmethod
    def alpha_090(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta((((close / open) - 1) / open), 4))) *
        ts_rank(correlation(rank(volume), rank(((close / open) - 1)), 5), 3)
        """
        close_open = (df["close"] / (df["open"] + 1e-9) - 1) / (df["open"] + 1e-9)
        delta_osc = delta(close_open, 4)
        rank1 = rank(delta_osc)
        corr = correlation(rank(df["volume"]), rank(close_open), 5)
        tsr = ts_rank(corr, 3)
        return -rank1 * tsr

    @staticmethod
    def alpha_091(df: pd.DataFrame) -> pd.Series:
        """
        ((-1 * rank(ts_max(delta(((vwap / (high - low)) - 1), 3)))) *
         ts_rank(correlation(rank(vwap), rank(volume), 5), 3))
        """
        vwap_osc = _vwap(df) / (df["high"] - df["low"] + 1e-9) - 1
        ts_max_delta = ts_max(delta(vwap_osc, 3), 3)
        corr = correlation(rank(_vwap(df)), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(ts_max_delta) * tsr

    @staticmethod
    def alpha_092(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(delta(vwap, 4)) *
        ts_rank(correlation(rank(close), rank(volume), 5), 3)
        """
        delta_vwap = -delta(_vwap(df), 4)
        corr = correlation(rank(df["close"]), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(delta_vwap) * tsr

    @staticmethod
    def alpha_093(df: pd.DataFrame) -> pd.Series:
        """
        (0 - ts_max(rank(correlation(rank(open), rank(volume), 10)), 3))) *
        (-1 * delta(close, 3))
        """
        corr = correlation(rank(df["open"]), rank(df["volume"]), 10)
        ts_max_rank = ts_max(rank(corr), 3)
        delta3 = -delta(df["close"], 3)
        return -ts_max_rank * delta3

    @staticmethod
    def alpha_094(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta(((high + low + close) / 3), 4))) *
        ts_rank(correlation(rank((high + low + close) / 3), rank(volume), 5), 3)
        """
        mid = (df["high"] + df["low"] + df["close"]) / 3.0
        delta_mid = -delta(mid, 4)
        corr = correlation(rank(mid), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(delta_mid) * tsr

    @staticmethod
    def alpha_095(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(rank(low), rank(adv15), 8)))) *
        ts_rank(correlation(sum(((vwap * 0.8) + (open * 0.2)), 20), sum(adv5, 15), 6), 4)
        """
        corr1 = correlation(rank(df["low"]), rank(_adv(15, df)), 8)
        blend = 0.8 * _vwap(df) + 0.2 * df["open"]
        corr2 = correlation(blend.rolling(20).sum(), _adv(5, df).rolling(15).sum(), 6)
        tsr = ts_rank(corr2, 4)
        return -rank(corr1) * tsr

    @staticmethod
    def alpha_096(df: pd.DataFrame) -> pd.Series:
        """
        -1 * rank(correlation(sum(((low * 0.8) + (vwap * 0.2)), 15), sum(adv40, 15), 6)) *
              rank(-abs(delta(close, 5)))
        """
        blend = 0.8 * df["low"] + 0.2 * _vwap(df)
        corr = correlation(blend.rolling(15).sum(), _adv(40, df).rolling(15).sum(), 6)
        delta5 = -delta(df["close"], 5).abs()
        return -rank(corr) * rank(delta5)

    @staticmethod
    def alpha_097(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(((high * 0.8) + (vwap * 0.2)), sum(adv20, 15), 20)))) *
        rank((close - ts_min(low, 15)) / (close - ts_max(high, 15)))
        """
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(20, df).rolling(15).sum(), 20)
        numerator = df["close"] - ts_min(df["low"], 15)
        denominator = df["close"] - ts_max(df["high"], 15) + 1e-9
        ratio = numerator / denominator
        return -rank(corr) * rank(ratio)

    @staticmethod
    def alpha_098(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(correlation(rank(vwap), rank(volume), 5)))) *
        ts_rank(correlation(close, open, 5), 3)
        """
        corr1 = correlation(rank(_vwap(df)), rank(df["volume"]), 5)
        corr2 = correlation(df["close"], df["open"], 5)
        tsr = ts_rank(corr2, 3)
        return -rank(corr1) * tsr

    @staticmethod
    def alpha_099(df: pd.DataFrame) -> pd.Series:
        """
        (0 - rank(delta(vwap, 5))) *
        ts_rank(correlation(rank(close), rank(volume), 5), 3)
        """
        delta_vwap = -delta(_vwap(df), 5)
        corr = correlation(rank(df["close"]), rank(df["volume"]), 5)
        tsr = ts_rank(corr, 3)
        return -rank(delta_vwap) * tsr

    @staticmethod
    def alpha_100(df: pd.DataFrame) -> pd.Series:
        """
        -1 * (rank(correlation(sum(((high + low) / 2), 20), sum(adv30, 20), 6)) +
              rank(-abs(delta(close, 5))))
        """
        mid = (df["high"] + df["low"]) / 2.0
        corr = correlation(mid.rolling(20).sum(), _adv(30, df).rolling(20).sum(), 6)
        delta5 = -delta(df["close"], 5).abs()
        return -(rank(corr) + rank(delta5))

    @staticmethod
    def alpha_101(df: pd.DataFrame) -> pd.Series:
        """
        ((-1 * rank((close - ts_min(low, 12)))) *
         (rank(correlation(((high * 0.8) + (vwap * 0.2)), adv20, 6)) ** 2))
        """
        diff = df["close"] - ts_min(df["low"], 12)
        blend = 0.8 * df["high"] + 0.2 * _vwap(df)
        corr = correlation(blend, _adv(20, df), 6)
        return -rank(diff) * (rank(corr) ** 2)

    # ------------------------------------------------------------------
    # Compute all (or a subset of) alphas on a DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def _build_registry() -> dict:
        """Lazily build the alpha-name → method map after class is fully defined."""
        return {
            f"alpha_{i:03d}": getattr(Alpha101, f"alpha_{i:03d}")
            for i in range(1, 102)
        }

    # Populated after class definition below
    ALPHA_METHODS: dict = {}

    def compute(self, df: pd.DataFrame, names: list = None) -> pd.DataFrame:
        """
        Compute alpha columns and add them to a copy of df.

        Parameters
        ----------
        df        : DataFrame with OHLCV columns (open, high, low, close, volume)
        names     : list of alpha names to compute, or None for all 101

        Returns
        -------
        DataFrame with new alpha_* columns
        """
        df = _ensure_columns(df)
        methods = names or list(self.ALPHA_METHODS.keys())
        for name in methods:
            if name in self.ALPHA_METHODS:
                try:
                    df[name] = self.ALPHA_METHODS[name](df)
                except Exception:
                    df[name] = np.nan
        return df


# Populate the registry after the class is fully defined
Alpha101.ALPHA_METHODS = Alpha101._build_registry()
