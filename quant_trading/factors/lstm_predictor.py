"""
LSTM and XGBoost predictor wrappers for quant factor augmentation.

These models are optional — they are NOT required for the alpha factor library.
They can be used to build a learned alpha (i.e., predict next-day returns from
a panel of formulaic alphas and technical indicators).

Requires (all optional): torch, xgboost, scikit-learn, talib.
Install with:  pip install torch xgboost scikit-learn talib
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

__all__ = ["LSTMPredictor", "XGBoostPredictor"]

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    """NaN-safe z-score normalisation."""
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x)
    std = np.nanstd(x) + 1e-10
    return (x - mean) / std


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=window)
    rolling_std = series.rolling(window, min_periods=window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def _prepare_sequence(
    data: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X_seq, y_seq) for LSTM-style training."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])  # predict close price
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ----------------------------------------------------------------------
# LSTMPredictor
# ----------------------------------------------------------------------

class LSTMPredictor:
    """
    Simple LSTM predictor for time-series forecasting.

    Can be trained on a DataFrame of OHLCV + alpha features to predict
    the next-period return or price.

    Parameters
    ----------
    seq_len       : number of time steps to use as input (default 20)
    hidden_dim     : LSTM hidden dimension (default 64)
    num_layers     : number of LSTM layers (default 2)
    dropout        : dropout rate (default 0.2)
    epochs         : training epochs (default 30)
    batch_size     : training batch size (default 32)
    learning_rate  : optimizer learning rate (default 1e-3)
    feature_cols   : list of column names to use as features (default: auto-detect)
    target_col     : column to predict (default: close)
    device         : "cuda" or "cpu" (auto-detect)

    Usage
    -----
        model = LSTMPredictor(seq_len=20)
        model.train(training_df)
        predictions = model.predict(testing_df)   # returns DataFrame with pred / actual
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "close",
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = device or ("cuda" if _has_torch() else "cpu")
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._feature_columns: List[str] = []
        self._trained = False

    def _build_model(self, n_features: int) -> Any:
        """Build PyTorch LSTM model."""
        import torch
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]   # last time step
                out = self.dropout(out)
                return self.fc(out)

        return _LSTM(n_features, self.hidden_dim, self.num_layers, self.dropout)

    def _to_tensor(self, arr: np.ndarray) -> Any:
        import torch
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the LSTM on df.

        Parameters
        ----------
        df : DataFrame with OHLCV and/or alpha columns

        Returns
        -------
        dict of training metrics (loss per epoch)
        """
        if not _has_torch():
            logger.warning("PyTorch not installed — LSTMPredictor unavailable")
            return {}

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Auto-detect features
        ohlcv = {"open", "high", "low", "close", "volume"}
        if self.feature_cols:
            feature_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c not in ("date", "symbol")]

        if not feature_cols:
            logger.warning("No feature columns found for training")
            return {}

        self._feature_columns = feature_cols

        # Prepare data
        data = df[feature_cols].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.ffill().bfill()
        arr = data.values.astype(np.float32)

        # Scale
        n_samples, n_features = arr.shape
        X_raw = (arr[:, :-1] if self.target_col in feature_cols else arr)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0) + 1e-10
        X_scaled = (X_raw - mean) / std
        self._scaler_X = {"mean": mean, "std": std}

        target_idx = feature_cols.index(self.target_col) if self.target_col in feature_cols else 0
        y_raw = arr[:, target_idx]
        y_mean = y_raw.mean()
        y_std = y_raw.std() + 1e-10
        y_scaled = (y_raw - y_mean) / y_std
        self._scaler_y = {"mean": y_mean, "std": y_std}

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_len : i])
            y_seq.append(y_scaled[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

        # Build model
        self._model = self._build_model(X_seq.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        # DataLoader
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        losses = []
        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self._model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                logger.info("LSTM epoch %d/%d — loss: %.6f", epoch + 1, self.epochs, avg_loss)

        self._trained = True
        return {"epoch_losses": losses}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for df.

        Returns DataFrame with columns: actual, pred, error, abs_error_pct
        """
        if not self._trained or self._model is None:
            logger.warning("Model not trained — returning empty predictions")
            return pd.DataFrame(columns=["actual", "pred", "error", "abs_error_pct"])

        import torch

        data = df[self._feature_columns].copy()
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        arr = data.values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mean = self._scaler_X["mean"]
        std = self._scaler_X["std"]
        X_scaled = (arr - mean) / (std + 1e-10)

        X_seq, y_actual = [], []
        for i in range(self.seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_len : i])
            y_actual.append(arr[i, 0])   # close price

        if not X_seq:
            return pd.DataFrame(columns=["actual", "pred", "error", "abs_error_pct"])

        X_seq = np.array(X_seq, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            preds_scaled = self._model(
                torch.tensor(X_seq, dtype=torch.float32, device=self.device)
            ).cpu().numpy().flatten()

        # Inverse transform predictions
        y_mean = self._scaler_y["mean"]
        y_std = self._scaler_y["std"]
        y_pred = preds_scaled * y_std + y_mean

        result = pd.DataFrame({
            "actual": y_actual,
            "pred": y_pred,
            "error": np.array(y_actual) - y_pred,
        })
        result["abs_error_pct"] = (result["error"].abs() / (np.abs(result["actual"]) + 1e-10)) * 100
        return result

    def save(self, path: str) -> None:
        """Save model state to a .pkl file."""
        import pickle
        state = {
            "model": self._model.state_dict() if self._model else None,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "feature_columns": self._feature_columns,
            "scaler_X": self._scaler_X,
            "scaler_y": self._scaler_y,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("LSTM model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model state from a .pkl file."""
        import pickle
        import torch
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.seq_len = state["seq_len"]
        self.hidden_dim = state["hidden_dim"]
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self._feature_columns = state["feature_columns"]
        self._scaler_X = state["scaler_X"]
        self._scaler_y = state["scaler_y"]
        self._model = self._build_model(len(self._feature_columns))
        self._model.load_state_dict(state["model"])
        self._model.to(self.device)
        self._trained = True
        logger.info("LSTM model loaded from %s", path)


# ----------------------------------------------------------------------
# XGBoostPredictor
# ----------------------------------------------------------------------

class XGBoostPredictor:
    """
    XGBoost regressor for panel-based return prediction.

    Train on a panel of alpha factors to predict next-day returns.
    Works well as an ensemble with formulaic alphas.

    Parameters
    ----------
    max_depth       : max tree depth (default 6)
    learning_rate   : boosting learning rate (default 0.1)
    n_estimators   : number of boosting rounds (default 200)
    feature_cols    : columns to use as features (default: auto)
    target_col      : column to predict (default: close)
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "close",
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_cols = feature_cols
        self.target_col = target_col
        self._model = None
        self._feature_columns: List[str] = []
        self._trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train XGBoost on df.
        Returns training metrics (mae, rmse, r2).
        """
        if not _has_xgboost():
            logger.warning("xgboost not installed — XGBoostPredictor unavailable")
            return {}

        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if self.feature_cols:
            feature_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c not in ("date", "symbol")]

        if not feature_cols:
            logger.warning("No feature columns found")
            return {}

        self._feature_columns = feature_cols

        data = df[feature_cols].copy()
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        target_idx = feature_cols.index(self.target_col) if self.target_col in feature_cols else 0
        X = data.values.astype(np.float32)
        y = X[:, target_idx]

        # Shift y by 1 to predict next-period return
        y_next = pd.Series(y).shift(-1).fillna(0).values

        dtrain = xgb.DMatrix(X, label=y_next)
        params = {
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
        self._model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        self._trained = True

        # In-sample metrics
        preds = self._model.predict(dtrain)
        mae = mean_absolute_error(y_next, preds)
        rmse = np.sqrt(mean_squared_error(y_next, preds))
        r2 = r2_score(y_next, preds)
        logger.info("XGBoost trained — MAE=%.6f RMSE=%.6f R2=%.4f", mae, rmse, r2)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions on df."""
        if not self._trained or self._model is None:
            return pd.DataFrame(columns=["actual", "pred", "error"])

        import xgboost as xgb

        data = df[self._feature_columns].copy()
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X = data.values.astype(np.float32)
        target_idx = self._feature_columns.index(self.target_col) if self.target_col in self._feature_columns else 0
        y_actual = X[:, target_idx]
        dtest = xgb.DMatrix(X)
        y_pred = self._model.predict(dtest)

        result = pd.DataFrame({
            "actual": y_actual,
            "pred": y_pred,
            "error": y_actual - y_pred,
        })
        result["abs_error_pct"] = (result["error"].abs() / (np.abs(result["actual"]) + 1e-10)) * 100
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        if not self._trained or self._model is None:
            return pd.DataFrame()
        import xgboost as xgb
        scores = self._model.get_score(importance_type="gain")
        return pd.DataFrame([
            {"feature": k, "importance": v} for k, v in scores.items()
        ]).sort_values("importance", ascending=False)


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
