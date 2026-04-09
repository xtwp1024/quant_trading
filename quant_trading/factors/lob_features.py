"""
Limit Order Book Microstructure Features.

Engineering module for LOB (Limit Order Book) tick-by-tick data, providing
the raw and derived features consumed by HFTPredictor.

Feature Groups
-------------
1. Price & Spread       : best bid/ask spread, mid-price, weighted spread
2. Quantity & Imbalance : bid/ask volumes, qty totals, imbalance ratios
3. Trade Price Position  : trade_price vs BBO, historical trade location
4. Diff / Delta          : first-order differences on all price/quantity columns
5. Direction Flags       : up/down classification based on price deltas
6. Lag Features          : 1-5 period lags on all non-flag columns
7. Rolling Statistics    : sum / mean / std / max / min over windows [5, 10, 20]
                           and time windows [1s, 3s, 5s, 10s]

This module is used internally by HFTPredictor but can also be used
independently for feature inspection or custom model pipelines.

Origin: D:/Hive/Data/trading_repos/HFT-price-prediction/
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mode

__all__ = ["LOBFeatureEngine", "LOBPreprocessor", "CORRELATION_FILTER_COLS"]


# ------------------------------------------------------------------
# Schema / dtype constants
# ------------------------------------------------------------------

RAW_COLS = [
    "timestamp", "bid_price", "bid_qty",
    "ask_price", "ask_qty",   "trade_price",
    "sum_trade_1s", "bid_advance_time", "ask_advance_time", "last_trade_time",
]

LABEL_COLS = ["_1s_side", "_3s_side", "_5s_side"]

CORRELATION_FILTER_COLS = ["ask_price"]  # removed in original pipeline


# ------------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------------

class LOBPreprocessor:
    """
    Clean and normalise raw LOB tick data.

    Handles
    ------
    - datetime parsing and sorting
    - dtype enforcement (float for numeric columns)
    - null-filling for sum_trade_1s and last_trade_time
    """

    _FLOAT_COLS = {
        "bid_price", "bid_qty", "ask_price", "ask_qty",
        "trade_price", "sum_trade_1s", "bid_advance_time",
        "ask_advance_time", "last_trade_time",
    }

    def __init__(self, fill_strategy: str = "zero_backfill"):
        """
        Parameters
        ----------
        fill_strategy : {'zero_backfill', 'ffill'}
            'zero_backfill' : sum_trade_1s null → 0; last_trade_time null →
                              previous + dt (clamped to 1s); any remaining → 0.
            'ffill'         : forward-fill all nulls (legacy behaviour).
        """
        if fill_strategy not in ("zero_backfill", "ffill"):
            raise ValueError("fill_strategy must be 'zero_backfill' or 'ffill'")
        self.fill_strategy = fill_strategy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        for c in self._FLOAT_COLS:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")

        # Sort by time
        data = data.sort_values("timestamp").reset_index(drop=True)

        # Fill nulls
        if self.fill_strategy == "zero_backfill":
            data = self._zero_backfill(data)
        else:
            data = data.ffill()

        return data

    @staticmethod
    def _zero_backfill(data: pd.DataFrame) -> pd.DataFrame:
        # sum_trade_1s: null → 0
        data["sum_trade_1s"] = data["sum_trade_1s"].fillna(0.0)

        # last_trade_time: backfill with dt increment
        class _Filler:
            prev_ts:  Optional[pd.Timestamp] = None
            prev_ltt: float = 0.0

            @classmethod
            def fill(cls, idx: int) -> float:
                row_ts  = data.loc[idx, "timestamp"]
                row_ltt = data.loc[idx, "last_trade_time"]
                if pd.isnull(row_ltt):
                    if cls.prev_ts is not None:
                        dt = (row_ts - cls.prev_ts).total_seconds()
                        row_ltt = cls.prev_ltt + dt if dt <= 1 else 0.0
                else:
                    cls.prev_ltt = float(row_ltt)
                cls.prev_ts = row_ts
                return row_ltt if not pd.isnull(row_ltt) else cls.prev_ltt

        data["last_trade_time"] = [_Filler.fill(i) for i in data.index]
        return data


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

class LOBFeatureEngine:
    """
    Full LOB microstructure feature builder.

    Mirrors the original ``feature_eng`` class but exposes individual
    feature groups as separate methods for selective use.

    Usage
    -----
    >>> engine = LOBFeatureEngine()
    >>> feats  = engine.build(raw_df)          # all groups
    >>> spreads = engine.spread_features(raw_df)  # only spread
    """

    MAX_LAG:    int = 5
    NUM_WINDOW: list[int] = [5, 10, 20]
    SEC_WINDOW: list[int] = [1, 3, 5, 10]

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature groups in sequence and return the full
        feature DataFrame (no label columns).
        """
        data = data.copy()
        self._ts = data["timestamp"]

        self._spread_features(data)
        self._qty_features(data)
        self._trade_price_features(data)
        self._diff_features(data)
        self._direction_features(data)

        data = data.drop(columns=["timestamp"], errors="ignore")

        self._lag_features(data)
        self._rolling_features(data)

        data.index = self._ts
        self._time_rolling_features(data)
        data.index = range(len(data))

        return data

    # ---- Group 1: spread & mid-price ----

    @staticmethod
    def _spread_features(data: pd.DataFrame) -> None:
        data["spread"] = data["ask_price"] - data["bid_price"]
        data["mid_price"] = (data["ask_price"] + data["bid_price"]) / 2.0

    # ---- Group 2: quantity & imbalance ----

    @staticmethod
    def _qty_features(data: pd.DataFrame) -> None:
        data["bid_ask_qty_total"] = data["ask_qty"] + data["bid_qty"]
        data["bid_ask_qty_diff"]  = data["ask_qty"] - data["bid_qty"]
        total = data["bid_ask_qty_total"].replace(0, np.nan)
        data["bid_ask_imbalance"] = data["bid_ask_qty_diff"] / total

    # ---- Group 3: trade price position ----

    def _trade_price_features(self, data: pd.DataFrame) -> None:
        # Current-bar position vs BBO
        data["trade_vs_bbo"] = 0
        data.loc[data["trade_price"] <= data["bid_price"], "trade_vs_bbo"] = -1
        data.loc[data["trade_price"] >= data["ask_price"], "trade_vs_bbo"] =  1

        # Historical-bar position via bisect (mirrors original)
        last_trade_ts = (
            data["timestamp"]
            - pd.to_timedelta(data["last_trade_time"], unit="s")
        )
        idx_list = [
            bisect_left(data["timestamp"].values, t)
            for t in last_trade_ts
        ]
        pos = []
        for i, idx in enumerate(idx_list):
            idx2 = min(idx + 1, data.shape[0] - 1)
            bp1, bp2 = data["bid_price"].iloc[idx],  data["bid_price"].iloc[idx2]
            ap1, ap2 = data["ask_price"].iloc[idx],  data["ask_price"].iloc[idx2]
            tp       = data["trade_price"].iloc[i]
            if (min(bp1, bp2) <= tp <= max(bp1, bp2)):
                pos.append(-1)
            elif (min(ap1, ap2) <= tp <= max(ap1, ap2)):
                pos.append(1)
            else:
                pos.append(0)
        data["trade_price_hist_pos"] = pos

    # ---- Group 4: first-order diffs ----

    @staticmethod
    def _diff_features(data: pd.DataFrame) -> None:
        for col in data.columns:
            if col == "timestamp":
                continue
            new = f"{col}_diff"
            data[new] = data[col] - data[col].shift(1)

    # ---- Group 5: direction flags ----

    @staticmethod
    def _direction_features(data: pd.DataFrame) -> None:
        data["up_down"] = 0
        data.loc[data["bid_price_diff"] < 0, "up_down"] = -1
        data.loc[data["ask_price_diff"] > 0, "up_down"] =  1

    # ---- Group 6: lag features ----

    def _lag_features(self, data: pd.DataFrame) -> None:
        exclude = {"trade_vs_bbo", "trade_price_hist_pos", "up_down"}
        rolling_cols = [c for c in data.columns if c not in exclude]
        for col in rolling_cols:
            for lag in range(1, self.MAX_LAG + 1):
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    # ---- Group 7a: count-based rolling ----

    def _rolling_features(self, data: pd.DataFrame) -> None:
        exclude = {"trade_vs_bbo", "trade_price_hist_pos"}
        rolling_cols = [c for c in data.columns if c not in exclude]

        sum_cols  = [c for c in rolling_cols if "diff"  in c or "up_down" in c]
        mean_cols = list(rolling_cols)
        max_cols  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        min_cols  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        std_cols  = list(rolling_cols)

        for col in rolling_cols:
            for w in self.NUM_WINDOW:
                self._apply_rolling(data, col, w, "sum",  sum_cols)
                self._apply_rolling(data, col, w, "mean", mean_cols)
                self._apply_rolling(data, col, w, "max",  max_cols)
                self._apply_rolling(data, col, w, "min",  min_cols)
                self._apply_rolling(data, col, w, "std",  std_cols)

    # ---- Group 7b: time-based rolling ----

    def _time_rolling_features(self, data: pd.DataFrame) -> None:
        exclude = {"trade_vs_bbo", "trade_price_hist_pos"}
        rolling_cols = [c for c in data.columns if c not in exclude]

        sum_cols  = [c for c in rolling_cols if "diff"  in c or "up_down" in c]
        mean_cols = list(rolling_cols)
        max_cols  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        min_cols  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        std_cols  = list(rolling_cols)

        mode_cols = ["up_down", "trade_vs_bbo", "trade_price_hist_pos"]

        for col in rolling_cols:
            for sw in self.SEC_WINDOW:
                w = f"{sw}s"
                self._apply_rolling(data, col, w, "sum",  sum_cols)
                self._apply_rolling(data, col, w, "mean", mean_cols)
                self._apply_rolling(data, col, w, "max",  max_cols)
                self._apply_rolling(data, col, w, "min",  min_cols)
                self._apply_rolling(data, col, w, "std",  std_cols)
                if col in mode_cols:
                    self._apply_rolling(data, col, w, "mode", [col])

    # ---- Rolling helper ----

    @staticmethod
    def _apply_rolling(
        data: pd.DataFrame, col: str, window: int | str,
        feature: str, col_list: list[str],
    ) -> None:
        if col not in col_list:
            return
        rolling = data[col].rolling(window=window)
        name = f"{col}_rolling_{feature}_{window}"
        if feature == "sum":
            data[name] = rolling.sum()
        elif feature == "mean":
            data[name] = rolling.mean()
        elif feature == "max":
            data[name] = rolling.max()
        elif feature == "min":
            data[name] = rolling.min()
        elif feature == "std":
            data[name] = rolling.std()
        elif feature == "mode":
            data[name] = rolling.apply(
                lambda x: mode(x, keepdims=True)[0][0], raw=False
            )
