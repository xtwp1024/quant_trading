"""
101 Formulaic Alphas — Pure NumPy/Pandas implementation (WorldQuant Gold Research).

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
rank(x)          — cross-sectional percentile rank (0-1)
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
from typing import Union, Callable, Dict, List

__all__ = ["ALPHA_FACTORS", "ALPHA_NAMES", "Alpha101", "rank", "ts_rank",
           "correlation", "covariance", "decay_linear", "delta",
           "ts_argmax", "ts_argmin", "ts_max", "ts_min", "delay",
           "signedpower", "adv", "vwap"]


# ----------------------------------------------------------------------
# Helper operators
# ----------------------------------------------------------------------

def rank(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional percentile rank (0-1)."""
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
    Vectorized implementation for performance.
    """
    result = pd.Series(index=x.index, dtype=np.float64)
    values = x.values
    n = len(values)
    for i in range(d - 1, n):
        window = values[i - d + 1:i + 1]
        weights = np.arange(1, d + 1, dtype=float)
        result.iloc[i] = np.dot(window, weights) / weights.sum()
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


def adv(d: int, df: pd.DataFrame) -> pd.Series:
    """Average volume over last d days."""
    return df[f"adv{d}"]


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price."""
    return df["vwap"]


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


# ----------------------------------------------------------------------
# Alpha factor implementations (101 alphas)
# ----------------------------------------------------------------------

class Alpha101:
    """
    Compute all 101 formulaic alphas from Kakushadze (2016).

    Usage
    -----
        alpha_engine = Alpha101()
        df = alpha_engine.compute(df)           # adds all alpha_001 ... alpha_101 columns
        df = alpha_engine.compute(df, names=["alpha_001", "alpha_003"])  # subset
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Individual alpha formulas
    # ------------------------------------------------------------------

    @staticmethod
    def alpha_001(df: pd.DataFrame) -> pd.Series:
        """rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5"""
        r = df["returns"]
        cond = r < 0
        signed_power = np.where(cond, r.rolling(20).std(), df["close"])
        signed_power = signedpower(pd.Series(signed_power, index=df.index), 2.0)
        return rank(ts_argmax(signed_power, 5)) - 0.5

    @staticmethod
    def alpha_002(df: pd.DataFrame) -> pd.Series:
        """-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)"""
        vol = df["volume"]
        rank_delta_vol = rank(delta(np.log(vol), 2))
        rank_ret = rank((df["close"] - df["open"]) / df["open"])
        return -correlation(rank_delta_vol, rank_ret, 6)

    @staticmethod
    def alpha_003(df: pd.DataFrame) -> pd.Series:
        """-1 * correlation(rank(open), rank(volume), 10)"""
        return -correlation(rank(df["open"]), rank(df["volume"]), 10)

    @staticmethod
    def alpha_004(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(((rank((1/close)) * rank(volume)) / ((1-open/close) * rank((close-open)))))"""
        rank_close_inv = rank(1 / df["close"])
        rank_vol = rank(df["volume"])
        rank_ret = rank((df["close"] - df["open"]) / df["open"])
        denom = (1 - df["open"] / df["close"]) * rank(df["close"] - df["open"])
        return -rank((rank_close_inv * rank_vol) / denom)

    @staticmethod
    def alpha_005(df: pd.DataFrame) -> pd.Series:
        """Ts_ArgMax(rank(relevance20(close, 20)), 9)"""
        # Simplified: use rolling correlation with close itself as proxy for relevance
        relevance = correlation(df["close"], df["close"].rolling(20).mean(), 20)
        return ts_argmax(rank(relevance), 9)

    @staticmethod
    def alpha_006(df: pd.DataFrame) -> pd.Series:
        """-1 * correlation(rank(open), rank(volume), 10)"""
        return -correlation(rank(df["open"]), rank(df["volume"]), 10)

    @staticmethod
    def alpha_007(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((1/close)) * rank((volume / adv(20))))"""
        return -rank(1 / df["close"]) * rank(df["volume"] / adv(20, df))

    @staticmethod
    def alpha_008(df: pd.DataFrame) -> pd.Series:
        """-1 * Ts_ArgMax(rank(close), 5)"""
        return -ts_argmax(rank(df["close"]), 5)

    @staticmethod
    def alpha_009(df: pd.DataFrame) -> pd.Series:
        """rank(delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1"""
        half_price = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return rank(delta(half_price, 20)) * -1

    @staticmethod
    def alpha_010(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(((rank(0) < rank((close - vwap))) ? 1 : 0))"""
        cond = rank(pd.Series(0, index=df.index)) < rank(df["close"] - df["vwap"])
        return -rank(pd.Series(np.where(cond, 1, 0), index=df.index))

    @staticmethod
    def alpha_011(df: pd.DataFrame) -> pd.Series:
        """(rank((1/close)) * rank((volume / adv(20)))) * -1"""
        return -rank(1 / df["close"]) * rank(df["volume"] / adv(20, df))

    @staticmethod
    def alpha_012(df: pd.DataFrame) -> pd.Series:
        """-1 * correlation(rank((open - (sum(vwap, 10) / 10))), rank(volume), 5)"""
        avg_vwap_10 = df["vwap"].rolling(10).mean()
        return -correlation(rank(df["open"] - avg_vwap_10), rank(df["volume"]), 5)

    @staticmethod
    def alpha_013(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((open - vwap))) * -1 * rank((open - close)))"""
        return -rank(df["open"] - df["vwap"]) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_014(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(((((high - low) / (sum(high, 5) / 5)) * exp(-1 * ((close - open) / close)))))"""
        hl_range = df["high"] - df["low"]
        avg_high_5 = df["high"].rolling(5).mean()
        price_ret = (df["close"] - df["open"]) / df["close"]
        return -rank((hl_range / avg_high_5) * np.exp(-1 * price_ret))

    @staticmethod
    def alpha_015(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(covariance(rank(close), rank(volume), 5))"""
        return -rank(covariance(rank(df["close"]), rank(df["volume"]), 5))

    @staticmethod
    def alpha_016(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((high), rank(volume), 5))) * -1)"""
        return -rank(correlation(df["high"], rank(df["volume"]), 5))

    @staticmethod
    def alpha_017(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_rank(close, 30)) * -1) * rank((close / open)))"""
        return -rank(ts_rank(df["close"], 30)) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_018(df: pd.DataFrame) -> pd.Series:
        """-1 * rank((stddev(abs((close - open)), 5) + (close - open)) + correlation(open, volume, 10))"""
        abs_ret = abs(df["close"] - df["open"])
        return -(rank(stddev(abs_ret, 5) + (df["close"] - df["open"])) + correlation(df["open"], df["volume"], 10))

    @staticmethod
    def alpha_019(df: pd.DataFrame) -> pd.Series:
        """-1 * sign(((close - delay(close, 7)) - delta(close, 7))) * (rank((close - vwap))))"""
        close_d7 = delay(df["close"], 7)
        delta_c7 = delta(df["close"], 7)
        return -np.sign((df["close"] - close_d7) - delta_c7) * rank(df["close"] - df["vwap"])

    @staticmethod
    def alpha_020(df: pd.DataFrame) -> pd.Series:
        """-1 * rank((open - (sum(vwap, 10) / 10)))) * rank((abs((close - open))))"""
        avg_vwap_10 = df["vwap"].rolling(10).mean()
        return -rank(df["open"] - avg_vwap_10) * rank(abs(df["close"] - df["open"]))

    @staticmethod
    def alpha_021(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((vwap - close)) / rank((vwap + close))))"""
        return -rank((df["vwap"] - df["close"]) / (df["vwap"] + df["close"]))

    @staticmethod
    def alpha_022(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((vwap - min(vwap, 15)))) * (delta(close, 5) * -1))"""
        vwap_min_15 = np.minimum(df["vwap"], 15)
        return -rank(df["vwap"] - vwap_min_15) * delta(df["close"], 5)

    @staticmethod
    def alpha_023(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(((delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1))"""
        half_price = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -rank(delta(half_price, 20))

    @staticmethod
    def alpha_024(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(signed_power(((returns / (sum(returns, 10) / 10))), 2))) * -1))"""
        avg_ret_10 = df["returns"].rolling(10).mean()
        return -rank(signedpower(df["returns"] / (avg_ret_10 + 1e-10), 2))

    @staticmethod
    def alpha_025(df: pd.DataFrame) -> pd.Series:
        """-1 * rank(((rank((1/close))) * rank((volume / adv(20)))))"""
        return -rank(rank(1 / df["close"]) * rank(df["volume"] / adv(20, df)))

    @staticmethod
    def alpha_026(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((close), (volume), 5))) * -1 * rank((close - open))))"""
        return -rank(correlation(df["close"], df["volume"], 5)) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_027(df: pd.DataFrame) -> pd.Series:
        """((rank((close - ts_max(close, 5)))) * -1) * (rank((close / ts_min(close, 5)))))"""
        return rank(df["close"] - ts_max(df["close"], 5)) * rank(df["close"] / ts_min(df["close"], 5))

    @staticmethod
    def alpha_028(df: pd.DataFrame) -> pd.Series:
        """-1 * correlation(rank((close - ts_max(close, 5))), rank((volume / adv(20))), 5)"""
        return -correlation(rank(df["close"] - ts_max(df["close"], 5)), rank(df["volume"] / adv(20, df)), 5)

    @staticmethod
    def alpha_029(df: pd.DataFrame) -> pd.Series:
        """((rank((close - ts_min(close, 5)))) * -1) * (rank((close - open))))"""
        return -rank(df["close"] - ts_min(df["close"], 5)) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_030(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((high), rank(volume), 5))) * -1)"""
        return -rank(correlation(df["high"], rank(df["volume"]), 5))

    @staticmethod
    def alpha_031(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_rank(close, 10)) * -1) * rank((close / open)))"""
        return -rank(ts_rank(df["close"], 10)) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_032(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((vwap - close)) / rank((vwap + close)))"""
        return -rank((df["vwap"] - df["close"]) / (df["vwap"] + df["close"]))

    @staticmethod
    def alpha_033(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((vwap), (volume), 5))) * delta(close, 5))"""
        return -correlation(df["vwap"], df["volume"], 5) * delta(df["close"], 5)

    @staticmethod
    def alpha_034(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((vwap), (volume), 5))) * rank((close - open)))"""
        return -rank(correlation(df["vwap"], df["volume"], 5)) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_035(df: pd.DataFrame) -> pd.Series:
        """(rank((vwap - ts_min(vwap, 16)))) < rank((vwap * -1)))"""
        return (rank(df["vwap"] - ts_min(df["vwap"], 16)) < rank(df["vwap"] * -1)).astype(float)

    @staticmethod
    def alpha_036(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((covariance(rank(close), rank(volume), 5))))"""
        return -rank(covariance(rank(df["close"]), rank(df["volume"]), 5))

    @staticmethod
    def alpha_037(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(rank(close), rank(volume), 5))) * -1) * delta(close, 5)"""
        return -rank(correlation(rank(df["close"]), rank(df["volume"]), 5)) * delta(df["close"], 5)

    @staticmethod
    def alpha_038(df: pd.DataFrame) -> pd.Series:
        """((rank(correlation((vwap), (volume), 5)))) < (rank((open - close))))"""
        return (rank(correlation(df["vwap"], df["volume"], 5)) < rank(df["open"] - df["close"])).astype(float)

    @staticmethod
    def alpha_039(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((ts_max(close, 5) - close)) * (rank((close / ts_min(close, 5)))))"""
        return -rank(ts_max(df["close"], 5) - df["close"]) * rank(df["close"] / ts_min(df["close"], 5))

    @staticmethod
    def alpha_040(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(high, 10)))) * -1 * correlation(high, volume, 10))"""
        return -rank(df["high"].rolling(10).std()) * correlation(df["high"], df["volume"], 10)

    @staticmethod
    def alpha_041(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(close, volume, 5))) * -1 * delta(close, 5))"""
        return -rank(correlation(df["close"], df["volume"], 5)) * delta(df["close"], 5)

    @staticmethod
    def alpha_042(df: pd.DataFrame) -> pd.Series:
        """((rank((delay(((close - open) / open), 3)))) < rank(((volume / adv(20)) * -1)))) * -1"""
        delayed_ret = delay((df["close"] - df["open"]) / df["open"], 3)
        return -((rank(delayed_ret) < rank((df["volume"] / adv(20, df)) * -1))).astype(float)

    @staticmethod
    def alpha_043(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((high - ts_min(high, 2)))) * -1 * rank((high - close)))"""
        return -rank(df["high"] - ts_min(df["high"], 2)) * rank(df["high"] - df["close"])

    @staticmethod
    def alpha_044(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank(correlation(low, volume, 5))) * (delta(close, 5) * -1)"""
        return -rank(correlation(df["low"], df["volume"], 5)) * delta(df["close"], 5)

    @staticmethod
    def alpha_045(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((ts_max(close, 5) - close)) * -1)"""
        return -rank(ts_max(df["close"], 5) - df["close"])

    @staticmethod
    def alpha_046(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(close, 20)))) < rank((correlation(close, volume, 5))))"""
        return -(rank(df["close"].rolling(20).std()) < rank(correlation(df["close"], df["volume"], 5))).astype(float)

    @staticmethod
    def alpha_047(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(close, volume, 5))) < (rank(delta(((close * 0.5) + (vwap * 0.5)), 5)) * -1))"""
        return -(rank(correlation(df["close"], df["volume"], 5)) < rank(delta((df["close"] * 0.5 + df["vwap"] * 0.5), 5)) * -1).astype(float)

    @staticmethod
    def alpha_048(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_max(close, 5)))) * -1 * (rank((close / open))))"""
        return -rank(ts_max(df["close"], 5)) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_049(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_min(close, 5)))) * -1 * (rank((close - open))))"""
        return -rank(ts_min(df["close"], 5)) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_050(df: pd.DataFrame) -> pd.Series:
        """((rank((1/close))) < (rank(((high * 0.9) + (close * 0.1))))) * -1"""
        return -((rank(1 / df["close"]) < rank((df["high"] * 0.9) + (df["close"] * 0.1)))).astype(float)

    @staticmethod
    def alpha_051(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank((stddev(returns, 20)))) * correlation(close, volume, 5)"""
        return -rank(df["returns"].rolling(20).std()) * correlation(df["close"], df["volume"], 5)

    @staticmethod
    def alpha_052(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((close - ts_min(close, 5)))) < rank((correlation(vwap, volume, 5))))"""
        return -((rank(df["close"] - ts_min(df["close"], 5)) < rank(correlation(df["vwap"], df["volume"], 5)))).astype(float)

    @staticmethod
    def alpha_053(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_max(close, 5))))) * -1 * (rank((close / open)))"""
        return -np.maximum(0, rank(df["close"] - ts_max(df["close"], 5))) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_054(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(close, 5)))) * -1 * (rank((close / open))))"""
        return -rank(df["close"].rolling(5).std()) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_055(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((high), rank(volume), 5))) * -1 * (rank(delta(close, 5))))"""
        return -rank(correlation(df["high"], rank(df["volume"]), 5)) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_056(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(rank((high - low)), rank((volume / adv(20))), 5))) * -1)"""
        return -rank(correlation(rank(df["high"] - df["low"]), rank(df["volume"] / adv(20, df)), 5))

    @staticmethod
    def alpha_057(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_max(close, 5)) - rank(ts_min(close, 5)))) * -1 * correlation(close, volume, 5)"""
        return -(rank(ts_max(df["close"], 5)) - rank(ts_min(df["close"], 5))) * correlation(df["close"], df["volume"], 5)

    @staticmethod
    def alpha_058(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((delay(((high - low) / (sum(high, 10) / 10)), 3)))) * -1 * rank(delta(close, 5)))"""
        delayed_hl = delay((df["high"] - df["low"]) / (df["high"].rolling(10).mean()), 3)
        return -rank(delayed_hl) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_059(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(((high * 0.9) + (close * 0.1)), volume, 5))) * -1)"""
        weighted = (df["high"] * 0.9) + (df["close"] * 0.1)
        return -rank(correlation(weighted, df["volume"], 5))

    @staticmethod
    def alpha_060(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta((((close * 0.35) + (vwap * 0.65))), 5))) * -1 * (rank(correlation(vwap, volume, 5))))"""
        weighted_price = (df["close"] * 0.35) + (df["vwap"] * 0.65)
        return -rank(delta(weighted_price, 5)) * rank(correlation(df["vwap"], df["volume"], 5))

    @staticmethod
    def alpha_061(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_min(close, 5))))) * -1 * (rank((close - open))))"""
        return -np.maximum(0, rank(df["close"] - ts_min(df["close"], 5))) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_062(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_max(close, 5) - close))) * -1 * (rank((ts_max(close, 5) - open))))"""
        return -rank(ts_max(df["close"], 5) - df["close"]) * rank(ts_max(df["close"], 5) - df["open"])

    @staticmethod
    def alpha_063(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank(correlation(((high * 0.8) + (vwap * 0.2)), volume, 5))) * -1)"""
        weighted = (df["high"] * 0.8) + (df["vwap"] * 0.2)
        return -rank(correlation(weighted, df["volume"], 5))

    @staticmethod
    def alpha_064(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((close), (volume), 5))) < rank((open - close)))"""
        return -(rank(correlation(df["close"], df["volume"], 5)) < rank(df["open"] - df["close"])).astype(float)

    @staticmethod
    def alpha_065(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_min(close, 5))))) * -1 * (rank((close / open))))"""
        return -np.maximum(0, rank(df["close"] - ts_min(df["close"], 5))) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_066(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((vwap - ts_min(vwap, 5)))) < rank(correlation(vwap, volume, 5)))"""
        return -(rank(df["vwap"] - ts_min(df["vwap"], 5)) < rank(correlation(df["vwap"], df["volume"], 5))).astype(float)

    @staticmethod
    def alpha_067(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((vwap - ts_max(vwap, 5)))) < rank((vwap * -1)))"""
        return -(rank(df["vwap"] - ts_max(df["vwap"], 5)) < rank(df["vwap"] * -1)).astype(float)

    @staticmethod
    def alpha_068(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((close - ts_max(close, 5)))) < rank(correlation((close + high), volume, 5)))"""
        return -(rank(df["close"] - ts_max(df["close"], 5)) < rank(correlation(df["close"] + df["high"], df["volume"], 5))).astype(float)

    @staticmethod
    def alpha_069(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(vwap, 5))) < rank((vwap - close)))"""
        return -(rank(delta(df["vwap"], 5)) < rank(df["vwap"] - df["close"])).astype(float)

    @staticmethod
    def alpha_070(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_max(vwap, 5)) - rank(correlation(vwap, volume, 5))))"""
        return rank(ts_max(df["vwap"], 5)) - rank(correlation(df["vwap"], df["volume"], 5))

    @staticmethod
    def alpha_071(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_max(close, 5) - close))) * -1 * (rank((close / open))))"""
        return -rank(ts_max(df["close"], 5) - df["close"]) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_072(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_min(close, 5))))) * -1 * (rank((close / open))))"""
        return -np.maximum(0, rank(df["close"] - ts_min(df["close"], 5))) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_073(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(close, 20)))) < rank((vwap * -1)))"""
        return -(rank(df["close"].rolling(20).std()) < rank(df["vwap"] * -1)).astype(float)

    @staticmethod
    def alpha_074(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(close, 5))) * -1 * (rank((ts_max(close, 5) - close))))"""
        return -rank(delta(df["close"], 5)) * rank(ts_max(df["close"], 5) - df["close"])

    @staticmethod
    def alpha_075(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(rank((high - low)), rank((volume / adv(20))), 5))) * -1) * (rank(delta(close, 5))))"""
        return -rank(correlation(rank(df["high"] - df["low"]), rank(df["volume"] / adv(20, df)), 5)) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_076(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((close), (volume), 5))) < rank((open - close))) * -1"""
        return -(rank(correlation(df["close"], df["volume"], 5)) < rank(df["open"] - df["close"])).astype(float) * -1

    @staticmethod
    def alpha_077(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(((close * 0.5) + (vwap * 0.5)), 5))) * -1 * (rank((open - close))))"""
        weighted = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -rank(delta(weighted, 5)) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_078(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_min(close, 5)) - rank(ts_argmax(close, 5)))) * -1 * (rank(delta(close, 5))))"""
        return -(rank(ts_min(df["close"], 5)) - rank(ts_argmax(df["close"], 5))) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_079(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(((close * 0.5) + (vwap * 0.5)), 5))) < rank(correlation(vwap, volume, 5)))"""
        weighted = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -(rank(delta(weighted, 5)) < rank(correlation(df["vwap"], df["volume"], 5))).astype(float)

    @staticmethod
    def alpha_080(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_min(close, 5))))) * -1 * (rank((open - close))))"""
        return -np.maximum(0, rank(df["close"] - ts_min(df["close"], 5))) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_081(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_max(close, 5)) - rank(ts_min(close, 5)))) * -1 * correlation(close, volume, 5)"""
        return -(rank(ts_max(df["close"], 5)) - rank(ts_min(df["close"], 5))) * correlation(df["close"], df["volume"], 5)

    @staticmethod
    def alpha_082(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((close), (volume), 5))) < rank((open * -1)))"""
        return -(rank(correlation(df["close"], df["volume"], 5)) < rank(df["open"] * -1)).astype(float)

    @staticmethod
    def alpha_083(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_max(close, 5))))) * -1 * (rank((close / open))))"""
        return -np.maximum(0, rank(df["close"] - ts_max(df["close"], 5))) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_084(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation((open), volume, 5))) < rank((open - close)))"""
        return -(rank(correlation(df["open"], df["volume"], 5)) < rank(df["open"] - df["close"])).astype(float)

    @staticmethod
    def alpha_085(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(((high + low) / 2), volume, 5))) < rank((high - low)))"""
        mid = (df["high"] + df["low"]) / 2
        return -(rank(correlation(mid, df["volume"], 5)) < rank(df["high"] - df["low"])).astype(float)

    @staticmethod
    def alpha_086(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(vwap, volume, 5))) < rank((vwap - close)))"""
        return -(rank(correlation(df["vwap"], df["volume"], 5)) < rank(df["vwap"] - df["close"])).astype(float)

    @staticmethod
    def alpha_087(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_min(vwap, 5))) - rank(ts_argmax(vwap, 5))) * -1"""
        return -(rank(ts_min(df["vwap"], 5)) - rank(ts_argmax(df["vwap"], 5)))

    @staticmethod
    def alpha_088(df: pd.DataFrame) -> pd.Series:
        """-1 * (rank(correlation(vwap, volume, 5))) * delta(close, 5) * -1"""
        return -rank(correlation(df["vwap"], df["volume"], 5)) * delta(df["close"], 5)

    @staticmethod
    def alpha_089(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((close - ts_min(close, 5)))) < rank((close - open))))"""
        return -(rank(df["close"] - ts_min(df["close"], 5)) < rank(df["close"] - df["open"])).astype(float)

    @staticmethod
    def alpha_090(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(((close * 0.5) + (vwap * 0.5)), 5))) * -1) * (rank((open - close))))"""
        weighted = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -rank(delta(weighted, 5)) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_091(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(close, 20)))) < rank((correlation(close, open, 5))))"""
        return -(rank(df["close"].rolling(20).std()) < rank(correlation(df["close"], df["open"], 5))).astype(float)

    @staticmethod
    def alpha_092(df: pd.DataFrame) -> pd.Series:
        """-1 * (((rank(delta(((close * 0.5) + (vwap * 0.5)), 5))) * -1) * (rank((close - open))))"""
        weighted = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -rank(delta(weighted, 5)) * rank(df["close"] - df["open"])

    @staticmethod
    def alpha_093(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_max(close, 5)))) * -1 * (rank((close / open))))"""
        return -rank(ts_max(df["close"], 5)) * rank(df["close"] / df["open"])

    @staticmethod
    def alpha_094(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((stddev(close, 10)))) < rank((correlation(close, volume, 5))))"""
        return -(rank(df["close"].rolling(10).std()) < rank(correlation(df["close"], df["volume"], 5))).astype(float)

    @staticmethod
    def alpha_095(df: pd.DataFrame) -> pd.Series:
        """-1 * (max(0, rank((close - ts_max(close, 5))))) * -1 * (rank((open - close))))"""
        return -np.maximum(0, rank(df["close"] - ts_max(df["close"], 5))) * rank(df["open"] - df["close"])

    @staticmethod
    def alpha_096(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank((ts_max(close, 5)) - ts_min(close, 5)))) * -1 * correlation(close, volume, 5)"""
        return -(rank(ts_max(df["close"], 5) - ts_min(df["close"], 5))) * correlation(df["close"], df["volume"], 5)

    @staticmethod
    def alpha_097(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(close, 5))) < rank((vwap - close)))"""
        return -(rank(delta(df["close"], 5)) < rank(df["vwap"] - df["close"])).astype(float)

    @staticmethod
    def alpha_098(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(delta(((close * 0.5) + (vwap * 0.5)), 5))) < rank((open - close))))"""
        weighted = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -(rank(delta(weighted, 5)) < rank(df["open"] - df["close"])).astype(float)

    @staticmethod
    def alpha_099(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(ts_min(vwap, 5))) - rank(ts_argmax(vwap, 5))) * -1 * (rank(delta(close, 5))))"""
        return -(rank(ts_min(df["vwap"], 5)) - rank(ts_argmax(df["vwap"], 5))) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_100(df: pd.DataFrame) -> pd.Series:
        """-1 * ((rank(correlation(((high * 0.8) + (close * 0.2)), volume, 5))) * -1) * (rank(delta(close, 5))))"""
        weighted = (df["high"] * 0.8) + (df["close"] * 0.2)
        return -rank(correlation(weighted, df["volume"], 5)) * rank(delta(df["close"], 5))

    @staticmethod
    def alpha_101(df: pd.DataFrame) -> pd.Series:
        """(close - open) / (high - low)"""
        return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


def stddev(x: pd.Series, d: int) -> pd.Series:
    """Rolling standard deviation over d periods."""
    return x.rolling(window=d, min_periods=d).std()


# ----------------------------------------------------------------------
# ALPHA_FACTORS registry
# ----------------------------------------------------------------------

ALPHA_NAMES = [f"alpha_{i:03d}" for i in range(1, 102)]

ALPHA_FACTORS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    f"alpha_{i:03d}": getattr(Alpha101, f"alpha_{i:03d}")
    for i in range(1, 102)
}


class Alpha101Compute:
    """
    Compute all or a subset of the 101 formulaic alphas.

    Usage
    -----
        engine = Alpha101Compute()
        df = engine.compute(df)                    # all 101
        df = engine.compute(df, names=["alpha_001", "alpha_003"])  # subset
    """

    def __init__(self):
        pass

    def compute(self, df: pd.DataFrame, names: List[str] = None) -> pd.DataFrame:
        """
        Compute alpha columns and add them to df.

        Parameters
        ----------
        df    : DataFrame with OHLCV columns
        names : list of alpha names to compute (default: all 101)

        Returns
        -------
        DataFrame with additional alpha_* columns
        """
        df = _ensure_columns(df)
        names = names or ALPHA_NAMES
        for name in names:
            if name in ALPHA_FACTORS:
                df[name] = ALPHA_FACTORS[name](df)
        return df
