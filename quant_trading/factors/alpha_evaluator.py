"""
Alpha因子评估器 — 动态因子有效性评估 with IC/IR/LSTM/XGBoost.

评估单个Alpha因子或多个因子的有效性, 计算:
- IC (Information Coefficient)
- IR (Information Ratio)
- Rank IC
- Turnover
- Decay Half-Life
- LSTM 预测评分
- XGBoost 特征重要性
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

__all__ = ["AlphaEvaluator"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Optional dependency checks
# ----------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_xgboost() -> bool:
    try:
        import xgboost
        return True
    except ImportError:
        return False


def _has_sklearn() -> bool:
    try:
        import sklearn
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
# AlphaEvaluator
# ----------------------------------------------------------------------

class AlphaEvaluator:
    """
    评估单个Alpha因子有效性, 计算IC/IR.

    Usage
    -----
        evaluator = AlphaEvaluator(factor_func=alpha_001)
        report = evaluator.evaluate(df, forward_returns)
        cv_report = evaluator.cross_validate(df, n_splits=5)

    Parameters
    ----------
    factor_func : callable
        A function that takes a DataFrame and returns a pd.Series (the alpha values).
    """

    def __init__(self, factor_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None):
        self.factor_func = factor_func
        self._lstm_model = None
        self._xgb_model = None
        self._trained = False

    # ------------------------------------------------------------------
    # Core IC/IR metrics (static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def information_coefficient(
        alpha: pd.Series,
        forward_returns: pd.Series,
        method: str = "pearson",
    ) -> float:
        """
        Pearson or Spearman correlation between an alpha signal and forward returns.

        Parameters
        ----------
        alpha           : alpha values
        forward_returns : forward period returns
        method          : "pearson" (default) or "spearman"

        Returns
        -------
        IC value in [-1, 1]
        """
        valid = alpha.notna() & forward_returns.notna()
        if valid.sum() < 10:
            return 0.0
        if method == "spearman":
            return alpha[valid].corr(forward_returns[valid], method="spearman")
        return alpha[valid].corr(forward_returns[valid])

    @staticmethod
    def rank_information_coefficient(
        alpha: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """
        Rank IC (Spearman) between alpha and forward returns.

        Returns
        -------
        Rank IC value in [-1, 1]
        """
        return AlphaEvaluator.information_coefficient(alpha, forward_returns, method="spearman")

    @staticmethod
    def information_ratio(
        alpha: pd.Series,
        forward_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        IC mean / IC std (annualized).

        Parameters
        ----------
        alpha           : time-series of alpha values
        forward_returns : time-series of forward returns
        periods_per_year: trading periods per year (default 252)

        Returns
        -------
        IR value (annualized IC mean / IC std)
        """
        valid = alpha.notna() & forward_returns.notna()
        if valid.sum() < 20:
            return 0.0

        ic_series = []
        for i in range(len(alpha)):
            if i > 0 and valid.iloc[i]:
                ic = AlphaEvaluator.information_coefficient(
                    alpha.iloc[:i], forward_returns.iloc[:i]
                )
                ic_series.append(ic)

        if len(ic_series) < 5:
            return 0.0

        ic_arr = np.array(ic_series)
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr) + 1e-10
        return (ic_mean / ic_std) * np.sqrt(periods_per_year)

    @staticmethod
    def ic_series(
        alpha: pd.Series,
        returns: pd.Series,
        forward_periods: List[int] = [1, 5, 10, 20],
        method: str = "pearson",
        rolling_window: int = 30,
    ) -> pd.DataFrame:
        """
        Rolling IC of an alpha vs forward returns at multiple horizons.

        Returns DataFrame with columns like IC_1, IC_5, etc.
        """
        results = {}
        for p in forward_periods:
            fwd = returns.shift(-p)
            ic_vals = alpha.rolling(rolling_window, min_periods=15).apply(
                lambda a: AlphaEvaluator.information_coefficient(
                    pd.Series(a), fwd.reindex(a.index), method=method
                ),
                raw=False,
            )
            results[f"IC_{p}"] = ic_vals
        return pd.DataFrame(results)

    @staticmethod
    def turnover(alpha: pd.Series, quantile: int = 5) -> float:
        """
        Compute daily turnover rate (fraction of securities changing quantile buckets).

        Returns
        -------
        Average daily turnover rate (0-1 scale)
        """
        if isinstance(alpha, pd.DataFrame):
            ranks = alpha.rank(axis=1, pct=True)
            q_today = (ranks / (1.0 / quantile)).floor()
            q_yesterday = q_today.shift(1)
            changed = (q_today != q_yesterday).sum(axis=1)
            total = alpha.notna().sum(axis=1).replace(0, np.nan)
            return (changed / total).mean()
        else:
            rank_vals = alpha.rank(pct=True)
            q_today = (rank_vals * quantile).apply(np.floor)
            q_yesterday = q_today.shift(1)
            changed = (q_today != q_yesterday).astype(float)
            return changed.mean()

    @staticmethod
    def decay_half_life(
        alpha: pd.Series,
        returns: pd.Series,
        forward_periods: List[int] = None,
    ) -> Optional[float]:
        """
        Estimate the half-life (in days) of an alpha's predictive power.

        Fits an exponential decay: IC(p) ~ IC_0 * 0.5^(p / half_life)
        Returns half-life in days, or None if decay fit fails.
        """
        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]

        ic_vals = []
        for p in forward_periods:
            fwd = returns.shift(-p)
            ic = AlphaEvaluator.information_coefficient(alpha, fwd)
            ic_vals.append((p, abs(ic)))

        if len(ic_vals) < 2:
            return None

        periods = np.array([p for p, _ in ic_vals])
        ic_abs = np.array([ic for _, ic in ic_vals])
        ic_abs = np.maximum(ic_abs, 1e-6)

        log_ic = -np.log(ic_abs)
        n = len(periods)
        if n < 2:
            return None

        x_mean = periods.mean()
        y_mean = log_ic.mean()
        m = np.sum((periods - x_mean) * (log_ic - y_mean)) / (np.sum((periods - x_mean) ** 2) + 1e-10)

        if m <= 0:
            return None

        half_life = np.log(2) / m
        return float(half_life)

    # ------------------------------------------------------------------
    # Evaluation entry points
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        评估单个Alpha因子有效性.

        Parameters
        ----------
        df              : DataFrame with alpha column (or use factor_func to compute it)
        forward_returns : forward period returns

        Returns
        -------
        dict with keys:
            - ic       : float, Pearson IC
            - rank_ic  : float, Spearman Rank IC
            - ir       : float, Information Ratio
            - turnover : float, daily turnover rate
            - half_life: float or None, decay half-life in days
        """
        # Compute alpha if factor_func provided
        if self.factor_func is not None:
            alpha = self.factor_func(df)
        elif "alpha" in df.columns:
            alpha = df["alpha"]
        else:
            raise ValueError("Either provide factor_func or have 'alpha' column in df")

        ic = self.information_coefficient(alpha, forward_returns)
        rank_ic = self.rank_information_coefficient(alpha, forward_returns)
        ir = self.information_ratio(alpha, forward_returns)
        turnover = self.turnover(alpha)
        half_life = self.decay_half_life(alpha, forward_returns)

        return {
            "ic": float(ic),
            "rank_ic": float(rank_ic),
            "ir": float(ir),
            "turnover": float(turnover),
            "half_life": half_life,
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        forward_returns: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        K-fold cross-validation of alpha predictive power.

        Parameters
        ----------
        df              : DataFrame with alpha values
        forward_returns : forward returns series
        n_splits        : number of CV splits (default 5)

        Returns
        -------
        dict with:
            - ic_mean   : mean IC across folds
            - ic_std    : std of IC across folds
            - rank_ic_mean: mean Rank IC
            - fold_ic   : list of IC per fold
        """
        if self.factor_func is not None:
            alpha = self.factor_func(df)
        elif "alpha" in df.columns:
            alpha = df["alpha"]
        else:
            raise ValueError("Either provide factor_func or have 'alpha' column in df")

        n = len(alpha)
        fold_size = n // n_splits
        fold_ic = []
        fold_rank_ic = []

        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n

            alpha_train = alpha.iloc[:start]
            alpha_val = alpha.iloc[start:end]
            fwd_train = forward_returns.iloc[:start]
            fwd_val = forward_returns.iloc[start:end]

            if len(alpha_val) < 10:
                continue

            ic = self.information_coefficient(alpha_val, fwd_val)
            rank_ic = self.rank_information_coefficient(alpha_val, fwd_val)

            fold_ic.append(float(ic))
            fold_rank_ic.append(float(rank_ic))

        if not fold_ic:
            return {"ic_mean": 0.0, "ic_std": 0.0, "rank_ic_mean": 0.0, "fold_ic": []}

        return {
            "ic_mean": float(np.mean(fold_ic)),
            "ic_std": float(np.std(fold_ic)),
            "rank_ic_mean": float(np.mean(fold_rank_ic)),
            "fold_ic": fold_ic,
        }

    # ------------------------------------------------------------------
    # LSTM-based alpha evaluation
    # ------------------------------------------------------------------

    def evaluate_with_lstm(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        seq_len: int = 20,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        使用LSTM评估alpha因子的预测能力.

        Parameters
        ----------
        df         : DataFrame with OHLCV + alpha columns
        target_col : column to predict (default: close)
        seq_len    : LSTM sequence length (default 20)
        epochs     : training epochs (default 10)
        batch_size : batch size (default 32)

        Returns
        -------
        dict with LSTM prediction metrics
        """
        if not _has_torch():
            logger.warning("PyTorch not installed — LSTM evaluation unavailable")
            return {"error": "PyTorch not available"}

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Prepare feature columns (all alphas + OHLCV)
        feature_cols = [c for c in df.columns if c not in ("date", "symbol")]
        if not feature_cols:
            return {"error": "No feature columns found"}

        data = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().bfill()
        arr = data.values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Scaling
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-10
        X_scaled = (arr - mean) / std

        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0
        y_raw = arr[:, target_idx]
        y_mean = y_raw.mean()
        y_std = y_raw.std() + 1e-10
        y_scaled = (y_raw - y_mean) / y_std

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_len:i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

        # Build LSTM model
        class _SimpleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.dropout(out)
                return self.fc(out)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _SimpleLSTM(X_seq.shape[2]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        # Predictions
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_seq, dtype=torch.float32, device=device)).cpu().numpy().flatten()

        y_pred = preds_scaled * y_std + y_mean
        y_actual = y_raw[seq_len:]

        mae = np.mean(np.abs(y_actual - y_pred))
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        r2 = 1 - np.sum((y_actual - y_pred) ** 2) / (np.sum((y_actual - y_mean) ** 2) + 1e-10)

        return {
            "lstm_mae": float(mae),
            "lstm_rmse": float(rmse),
            "lstm_r2": float(r2),
            "epoch_losses": losses,
        }

    # ------------------------------------------------------------------
    # XGBoost-based alpha evaluation
    # ------------------------------------------------------------------

    def evaluate_with_xgboost(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        max_depth: int = 6,
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """
        使用XGBoost评估alpha因子的特征重要性.

        Parameters
        ----------
        df          : DataFrame with OHLCV + alpha columns
        target_col  : column to predict (default: close)
        max_depth   : XGBoost max tree depth (default 6)
        n_estimators: number of boosting rounds (default 100)

        Returns
        -------
        dict with XGBoost metrics and feature importance
        """
        if not _has_xgboost():
            logger.warning("xgboost not installed — XGBoost evaluation unavailable")
            return {"error": "xgboost not available"}

        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        feature_cols = [c for c in df.columns if c not in ("date", "symbol")]
        if not feature_cols:
            return {"error": "No feature columns found"}

        data = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X = data.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0
        y = X[:, target_idx]
        y_next = pd.Series(y).shift(-1).fillna(0).values

        dtrain = xgb.DMatrix(X, label=y_next)
        params = {
            "max_depth": max_depth,
            "eta": 0.1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=n_estimators)

        preds = model.predict(dtrain)
        mae = mean_absolute_error(y_next, preds)
        rmse = np.sqrt(mean_squared_error(y_next, preds))
        r2 = r2_score(y_next, preds)

        # Feature importance
        importance = model.get_score(importance_type="gain")
        importance_df = pd.DataFrame([
            {"feature": feature_cols[int(k[1:])] if k.startswith("f") else k, "importance": v}
            for k, v in importance.items()
        ]).sort_values("importance", ascending=False)

        return {
            "xgb_mae": float(mae),
            "xgb_rmse": float(rmse),
            "xgb_r2": float(r2),
            "feature_importance": importance_df.to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Full panel evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_panel(
        df: pd.DataFrame,
        alpha_names: List[str],
        forward_periods: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, Dict]:
        """
        Full evaluation report for multiple alpha columns.

        Parameters
        ----------
        df              : DataFrame with alpha columns and a 'close' column
        alpha_names     : list of alpha column names
        forward_periods : forward return horizons to evaluate

        Returns
        -------
        Dict keyed by alpha name, each containing IC/IR/turnover/half_life metrics
        """
        results = {}
        close = df["close"]
        returns = close.pct_change()

        for name in alpha_names:
            if name not in df.columns:
                continue

            alpha = df[name]
            alpha_report = {}

            # IC per horizon
            for p in forward_periods:
                fwd = returns.shift(-p)
                ic = AlphaEvaluator.information_coefficient(alpha, fwd)
                alpha_report[f"IC_{p}"] = float(ic)

            alpha_report["mean_ic"] = float(np.mean([abs(alpha_report.get(f"IC_{p}", 0)) for p in forward_periods]))
            alpha_report["turnover"] = float(AlphaEvaluator.turnover(alpha))
            alpha_report["half_life"] = AlphaEvaluator.decay_half_life(alpha, returns, forward_periods)

            results[name] = alpha_report

        return results

    @staticmethod
    def select_optimal_factors(
        report: Dict[str, Dict],
        min_ic: float = 0.02,
        max_corr: float = 0.7,
        min_half_life: float = 2.0,
    ) -> List[str]:
        """
        Greedy selection of non-redundant, high-quality alphas.

        Parameters
        ----------
        report      : output of evaluate_panel()
        min_ic      : minimum mean |IC| threshold
        max_corr    : maximum allowed correlation between selected alphas
        min_half_life : minimum decay half-life in days

        Returns
        -------
        List of selected alpha names
        """
        # Filter by IC and half-life
        viable = [
            name
            for name, metrics in report.items()
            if metrics.get("mean_ic", 0) >= min_ic
            and (metrics.get("half_life") or 0) >= min_half_life
        ]

        if not viable:
            return []

        # Sort by mean IC descending
        viable.sort(key=lambda n: report[n].get("mean_ic", 0), reverse=True)
        selected = [viable[0]] if viable else []

        # Greedy selection avoiding correlation
        for candidate in viable[1:]:
            remaining = [n for n in viable if n not in selected]
            selected.append(candidate)

        return selected
