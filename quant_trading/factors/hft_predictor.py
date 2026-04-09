"""
HFT Limit Order Book Price Movement Predictor.

LightGBM + Random Forest ensemble for high-frequency limit order book
price change classification (76-78% accuracy on 1s/3s/5s horizons).

Origin: D:/Hive/Data/trading_repos/HFT-price-prediction/
Labels: 0=no change, 1=bid down, 2=ask up.

Integration: used by AlphaAgent and PredictionAgent for short-horizon
price movement signals; complements HighFrequencyFactors (A1-A39).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from bisect import bisect_left
from scipy.stats import mode

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

__all__ = ["HFTPredictor", "HFTLabel"]


class HFTLabel:
    """HFT price movement label scheme."""
    NO_CHANGE = 0
    BID_DOWN   = 1   # bid price decreases
    ASK_UP     = 2   # ask price increases

    @classmethod
    def name(cls, v: int) -> str:
        return {0: "no_change", 1: "bid_down", 2: "ask_up"}.get(v, str(v))


# ------------------------------------------------------------------
# Internal feature-engineering helpers (mirrors original pipeline)
# ------------------------------------------------------------------

class _LOBFeatureEngine:
    """Limit-order-book microstructure feature builder.

    Mirrors the original ``feature_eng`` class from the HFT repository,
    producing the same derived columns (spread, qty imbalance, diffs,
    lags, rolling stats) that the GA+RF feature selector expects.
    """

    max_lag: int = 5
    num_window = [5, 10, 20]
    sec_window = [1, 3, 5, 10]

    @staticmethod
    def bid_ask_spread(data: pd.DataFrame) -> None:
        data["spread"] = data["ask_price"] - data["bid_price"]

    @staticmethod
    def bid_ask_qty_comb(data: pd.DataFrame) -> None:
        data["bid_ask_qty_total"] = data["ask_qty"] + data["bid_qty"]
        data["bid_ask_qty_diff"]  = data["ask_qty"] - data["bid_qty"]

    @staticmethod
    def trade_price_feature(data: pd.DataFrame) -> None:
        # current-bar trade price position
        data["trade_price_compare"] = 0
        data.loc[data["trade_price"] <= data["bid_price"], "trade_price_compare"] = -1
        data.loc[data["trade_price"] >= data["ask_price"], "trade_price_compare"] =  1

        # historical-bar position via bisect
        last_trade_ts = data["timestamp"] - pd.to_timedelta(data["last_trade_time"], unit="s")
        idx_list = [bisect_left(data["timestamp"].values, t) for t in last_trade_ts]
        pos = []
        for i, idx in enumerate(idx_list):
            idx1 = idx
            idx2 = min(idx1 + 1, data.shape[0] - 1)
            bp1, bp2 = data["bid_price"].iloc[idx1], data["bid_price"].iloc[idx2]
            ap1, ap2 = data["ask_price"].iloc[idx1], data["ask_price"].iloc[idx2]
            tp = data["trade_price"].iloc[i]
            if (min(bp1, bp2) <= tp <= max(bp1, bp2)):
                pos.append(-1)
            elif (min(ap1, ap2) <= tp <= max(ap1, ap2)):
                pos.append(1)
            else:
                pos.append(0)
        data["trade_price_pos"] = pos

    @staticmethod
    def diff_feature(data: pd.DataFrame) -> None:
        for col in data.columns:
            if col == "timestamp":
                continue
            data[f"{col}_diff"] = data[col] - data[col].shift(1)

    @staticmethod
    def up_or_down(data: pd.DataFrame) -> None:
        data["up_down"] = 0
        data.loc[data["bid_price_diff"] < 0, "up_down"] = -1
        data.loc[data["ask_price_diff"] > 0, "up_down"] =  1

    @staticmethod
    def lag_feature(data: pd.DataFrame, col: str, lag: int) -> None:
        data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    @staticmethod
    def rolling_feature(
        data: pd.DataFrame, col: str, window: int | str, feature: str
    ) -> None:
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
            data[name] = rolling.apply(lambda x: mode(x, keepdims=True)[0][0])

    @classmethod
    def build(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline: basic → lag/rolling → NA drop."""
        data = data.copy()
        ts = data["timestamp"]

        cls.bid_ask_spread(data)
        cls.bid_ask_qty_comb(data)
        cls.trade_price_feature(data)
        cls.diff_feature(data)
        cls.up_or_down(data)

        data = data.drop("timestamp", axis=1)

        rolling_cols = set(data.columns) - {"trade_price_compare", "trade_price_pos"}
        rolling_sum  = [c for c in rolling_cols if "diff" in c or "up_down" in c]
        rolling_mean = list(rolling_cols)
        rolling_max  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        rolling_min  = [c for c in rolling_cols if "bid_qty" in c or "ask_qty" in c]
        rolling_std  = list(rolling_cols)

        for col in rolling_cols:
            for lag in range(1, cls.max_lag + 1):
                cls.lag_feature(data, col, lag)

        for col in rolling_cols:
            for w in cls.num_window:
                for feat, col_list in [
                    ("sum",  rolling_sum),
                    ("mean", rolling_mean),
                    ("max",  rolling_max),
                    ("min",  rolling_min),
                    ("std",  rolling_std),
                ]:
                    if col in col_list:
                        cls.rolling_feature(data, col, w, feat)

        # Time-based rolling (strings like "5s")
        data.index = ts
        for col in rolling_cols:
            for sw in cls.sec_window:
                w = f"{sw}s"
                for feat, col_list in [
                    ("sum",  rolling_sum),
                    ("mean", rolling_mean),
                    ("max",  rolling_max),
                    ("min",  rolling_min),
                    ("std",  rolling_std),
                ]:
                    if col in col_list:
                        cls.rolling_feature(data, col, w, feat)
                if col in {"up_down", "trade_price_compare", "trade_price_pos"}:
                    cls.rolling_feature(data, col, w, "mode")
        data.index = range(len(data))

        return data


class _CorrelationFilter:
    """Remove highly correlated columns (|corr| >= threshold)."""

    remove_cols: list[str] = []

    @classmethod
    def filter(cls, x: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
        x = x.copy()
        idx2col = {i: c for i, c in enumerate(x.columns)}
        corr = np.array(x.corr())
        pairs = list(zip(*np.where(np.abs(corr) >= threshold)))
        to_del: list[set] = []
        for i, j in pairs:
            if i == j:
                continue
            has_int = False
            for k, s in enumerate(to_del):
                if {idx2col[i], idx2col[j]} & s:
                    has_int = True
                    to_del[k] = s | {idx2col[i], idx2col[j]}
                    break
            if not has_int:
                to_del.append({idx2col[i], idx2col[j]})

        for s in to_del:
            s_copy = s.copy()
            s_copy.pop()
            x = x.drop(list(s_copy), axis=1)
            cls.remove_cols += list(s_copy)

        return x


# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------

_FLOAT_COLS = [
    "bid_price", "bid_qty", "ask_price", "ask_qty",
    "trade_price", "sum_trade_1s", "bid_advance_time",
    "ask_advance_time", "last_trade_time",
]


def _preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    for c in _FLOAT_COLS:
        if c in data.columns:
            data[c] = data[c].astype(float)
    return data.sort_values("timestamp").reset_index(drop=True)


def _fill_null(data: pd.DataFrame) -> pd.DataFrame:
    """Fill sum_trade_1s nulls with 0; backfill last_trade_time."""
    data = data.copy()
    data.loc[data["sum_trade_1s"].isnull(), "sum_trade_1s"] = 0.0

    class Filler:
        prev_t: Optional[pd.Timestamp] = None
        prev_ltt: float = np.nan

        @classmethod
        def fill(cls, idx: int) -> float:
            row_ts  = data.loc[idx, "timestamp"]
            row_ltt = data.loc[idx, "last_trade_time"]
            if pd.isnull(row_ltt):
                if cls.prev_t is not None:
                    dt = (row_ts - cls.prev_t).total_seconds()
                    row_ltt = cls.prev_ltt + dt if dt <= 1 else np.nan
            cls.prev_t   = row_ts
            cls.prev_ltt = row_ltt if not pd.isnull(row_ltt) else cls.prev_ltt
            return row_ltt

    data["last_trade_time"] = [_Filler.fill(i) for i in data.index]
    return data


def _x_y_split(data: pd.DataFrame):
    label_cols = ["_1s_side", "_3s_side", "_5s_side"]
    feature_cols = list(set(data.columns) - set(label_cols))
    return data[feature_cols].copy(), data[label_cols].copy()


# ------------------------------------------------------------------
# Main predictor class
# ------------------------------------------------------------------

class HFTPredictor:
    """
    LightGBM + Random Forest ensemble for HFT limit-order-book
    price movement classification.

    Parameters
    ----------
    horizon : {1, 3, 5}
        Prediction horizon in seconds.
    model_dir : Path | str
        Directory containing ``lgbm.joblib`` and ``rf.joblib``.
    features_path : Path | str
        Path to ``features.json`` with ``keep_features`` and
        ``correlation_remove`` lists.

    Example
    -------
    >>> pred = HFTPredictor(horizon=5)
    >>> pred.load_models()
    >>> pred.predict(raw_lob_df)   # returns (np.array, pd.DataFrame)
    """

    LABEL_COLS = ["_1s_side", "_3s_side", "_5s_side"]

    def __init__(
        self,
        horizon: Literal[1, 3, 5] = 5,
        model_dir: Path | str | None = None,
        features_path: Path | str | None = None,
    ):
        if not _HAS_LGBM or not _HAS_SK:
            raise ImportError("HFTPredictor requires lightgbm and scikit-learn")

        self.horizon = horizon
        self.label_col = f"_{horizon}s_side"

        # Default to original repo artifacts
        if model_dir is None:
            model_dir = Path("D:/Hive/Data/trading_repos/HFT-price-prediction")
        if features_path is None:
            features_path = Path("D:/Hive/Data/trading_repos/HFT-price-prediction/features.txt")

        self.model_dir     = Path(model_dir)
        self.features_path = Path(features_path)

        self._lgbm: Optional[LGBMClassifier] = None
        self._rf:   Optional[RandomForestClassifier] = None
        self._keep_features:  list[str] = []
        self._corr_remove:    list[str] = []

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_models(self) -> "HFTPredictor":
        """Load pre-trained LightGBM and Random Forest models from disk."""
        import joblib
        lgbm_path = self.model_dir / "lgbm.joblib"
        rf_path   = self.model_dir / "rf.joblib"
        if not lgbm_path.exists() or not rf_path.exists():
            warnings.warn(f"Model files not found at {self.model_dir}; predict() will raise.")
        self._lgbm = joblib.load(lgbm_path)
        self._rf   = joblib.load(rf_path)
        return self

    def load_features(self) -> "HFTPredictor":
        """Load selected-feature list and correlation-remove list."""
        with open(self.features_path, "r") as f:
            feat_dict = json.load(f)
        self._keep_features = feat_dict.get("keep_features", [])
        self._corr_remove   = feat_dict.get("correlation_remove", [])
        return self

    def load(self) -> "HFTPredictor":
        """Convenience: load both models and features."""
        return self.load_models().load_features()

    def save_features(self, features: list[str], corr_remove: list[str]) -> None:
        """Save selected features to JSON."""
        self._keep_features = features
        self._corr_remove   = corr_remove
        out = {"keep_features": features, "correlation_remove": corr_remove}
        with open(self.features_path, "w") as f:
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Core predict
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame,
        return_labels: bool = True,
    ) -> tuple[np.ndarray, Optional[pd.DataFrame]]:
        """
        Run ensemble prediction on raw LOB data.

        Parameters
        ----------
        data : pd.DataFrame
            Raw limit-order-book DataFrame. Must contain:
            ``timestamp``, ``bid_price``, ``bid_qty``, ``ask_price``,
            ``ask_qty``, ``trade_price``, ``sum_trade_1s``,
            ``bid_advance_time``, ``ask_advance_time``, ``last_trade_time``,
            and optionally the label columns.
        return_labels : bool
            If True, return ground-truth labels alongside predictions.

        Returns
        -------
        predictions : np.ndarray
            Ensemble-predicted class labels (0=no change, 1=bid down, 2=ask up).
        labels : pd.DataFrame | None
            Ground-truth labels for the horizon, or None if ``return_labels=False``.
        """
        if self._lgbm is None or self._rf is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        data = data.copy()
        data = _preprocess(data)
        data = _fill_null(data)
        x, y = _x_y_split(data)

        x = _LOBFeatureEngine.build(x)
        x = x.drop(self._corr_remove, axis=1, errors="ignore")
        x, y = _drop_na(x, y)

        x = x[self._keep_features]

        lgbm_proba = self._lgbm.predict_proba(x)
        rf_proba   = self._rf.predict_proba(x)
        ensemble   = (lgbm_proba + rf_proba) / 2.0
        preds      = np.argmax(ensemble, axis=1)

        return preds, y[self.label_col] if return_labels else None

    # ------------------------------------------------------------------
    # Training entry-point (for re-training from scratch)
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        data: pd.DataFrame,
        horizon: Literal[1, 3, 5] = 5,
        model_dir: Path | str | None = None,
    ) -> "HFTPredictor":
        """
        Train the full pipeline: feature engineering → selection → ensemble.

        This is a high-level wrapper mirroring ``train_model()`` from the
        original repository.
        """
        import joblib

        model_dir = Path(model_dir) if model_dir else Path("D:/Hive/Data/trading_repos/HFT-price-prediction")
        label_col = f"_{horizon}s_side"

        data = _preprocess(data)
        data = _fill_null(data)
        x, y = _x_y_split(data)
        x = _LOBFeatureEngine.build(x)
        x = _CorrelationFilter.filter(x)
        x, y = _drop_na(x, y)
        y_h  = y[label_col]

        features = _hybrid_select(x, y_h)
        correlation_remove = _CorrelationFilter.remove_cols

        # LightGBM (grid search — original param grid is small enough)
        lgbm = _train_lgbm(x[features], y_h)
        rf   = _train_rf(x[features], y_h)

        joblib.dump(lgbm, model_dir / "lgbm.joblib")
        joblib.dump(rf,   model_dir / "rf.joblib")

        out = {"keep_features": features, "correlation_remove": correlation_remove}
        with open(model_dir / "features.json", "w") as f:
            json.dump(out, f, indent=2)

        inst = cls(horizon=horizon, model_dir=model_dir, features_path=model_dir / "features.json")
        inst._lgbm = lgbm
        inst._rf   = rf
        inst._keep_features = features
        inst._corr_remove   = correlation_remove
        return inst


# ------------------------------------------------------------------
# Internal training helpers
# ------------------------------------------------------------------

def _drop_na(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = x.dropna().reset_index(drop=True)
    return x, y.loc[x.index].reset_index(drop=True)


def _train_rf(x, y) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1)
    rf.fit(x, y)
    return rf


def _train_lgbm(x, y) -> LGBMClassifier:
    from itertools import product
    from sklearn.model_selection import GridSearchCV

    paramgrid = {
        "learning_rate":     np.arange(0.0005, 0.0015, 0.0001),
        "n_estimators":     range(800, 2000, 200),
        "max_depth":        [3, 4],
        "colsample_bytree": np.arange(0.2, 0.5, 0.1),
        "reg_alpha":        [1],
        "reg_lambda":       [1],
    }
    keys, vals = list(zip(*paramgrid.items()))
    combos = [dict(zip(keys, v)) for v in product(*vals)]

    if len(combos) > 1000:
        # GA search path (requires evolutionary_search — skip if unavailable)
        try:
            from evolutionary_search import EvolutionaryAlgorithmSearchCV
            tuner = EvolutionaryAlgorithmSearchCV(
                estimator=LGBMClassifier(),
                params=paramgrid,
                scoring="accuracy",
                cv=TimeSeriesSplit(n_splits=4),
                population_size=50,
                gene_mutation_prob=0.2,
                gene_crossover_prob=0.5,
                tournament_size=3,
                generations_number=20,
                n_jobs=-1,
            )
            tuner.fit(x, y)
            best = tuner.best_params_
        except Exception:
            warnings.warn("GA tuner unavailable, falling back to first param combo.")
            best = combos[0]
    else:
        tuner = GridSearchCV(
            LGBMClassifier(), paramgrid,
            scoring="accuracy", cv=TimeSeriesSplit(n_splits=4), n_jobs=-1,
        )
        tuner.fit(x, y)
        best = tuner.best_params_

    return LGBMClassifier(**best).fit(x, y)


def _hybrid_select(x: pd.DataFrame, y) -> list[str]:
    """Hybrid GA + RF-importance feature selection."""
    imp_features = _rf_importance_features(x, y)
    ga_features  = _ga_select_features(x, y)
    return list(set(imp_features) | set(ga_features))


def _rf_importance_features(x: pd.DataFrame, y, top_perc: float = 0.05) -> list[str]:
    """Top features by averaged RF importance across 3 bootstraps."""
    n = x.shape[1]
    imp = pd.DataFrame(np.zeros((n, 4)))
    imp.columns = ["feature", "i1", "i2", "i3"]
    imp["feature"] = list(x.columns)

    for i, col in enumerate(["i1", "i2", "i3"], 1):
        rf = RandomForestClassifier(n_estimators=10, max_depth=8, n_jobs=-1)
        rf.fit(x, y)
        imp[col] = [dict(zip(x.columns, rf.feature_importances_)).get(c, 0) for c in x.columns]

    imp["avg"] = imp[["i1", "i2", "i3"]].mean(axis=1)
    threshold  = np.percentile(imp["avg"], int((1 - top_perc) * 100))
    return list(imp.loc[imp["avg"] >= threshold, "feature"])


def _ga_select_features(x: pd.DataFrame, y) -> list[str]:
    """GA-selected features using GeneticSelectionCV."""
    try:
        from genetic_selection import GeneticSelectionCV
    except ImportError:
        warnings.warn("genetic_selection unavailable; returning empty list.")
        return []

    rf = RandomForestClassifier(max_depth=8, n_estimators=10, n_jobs=-1)
    selector = GeneticSelectionCV(
        rf,
        cv=TimeSeriesSplit(n_splits=4),
        verbose=0,
        scoring="accuracy",
        max_features=80,
        n_population=200,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=100,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=5,
        caching=True,
        n_jobs=-1,
    )
    selector.fit(x, y)
    return list(x.columns[selector.support_])
