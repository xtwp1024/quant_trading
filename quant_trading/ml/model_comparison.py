# -*- coding: utf-8 -*-
"""Multi-model comparison suite for time series forecasting.

多模型对比框架 — 评估神经网络模型 vs 传统统计模型 (ARMA).
支持: ARMA, CNN, TDNN, RNN, LSTM.

References:
    - TradingNeuralNetwork (D:/Hive/Data/trading_repos/TradingNeuralNetwork/)
    - ARMA analysis: TradingNeuralNetwork/analysis/arma.py
    - LSTM/Keras: TradingNeuralNetwork/lstm/lstm.py
    - SimpleRNN/Keras: TradingNeuralNetwork/RNN/Simple_RNN.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

from .nn_models import (
    BaseModel,
    ARMAModel,
    CNN1DModel,
    TDNNModel,
    RNNModel,
    LSTMModel,
)

__all__ = ["ModelComparisonSuite"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(_mse(y_true, y_pred))


def _sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio from a return series."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / returns.std()


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct directional predictions."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.diff(y_true)
    pred_dir = np.diff(y_pred)
    return np.mean((true_dir * pred_dir) > 0)


# --------------------------------------------------------------------------- #
# Model Comparison Suite
# --------------------------------------------------------------------------- #

class ModelComparisonSuite:
    """Multi-model comparison suite — evaluate NN vs traditional statistical models.

    多模型对比套件 — 评估 CNN/TDNN/RNN/LSTM vs ARMA 传统统计基线.
    支持训练、评估、对比可视化、最佳模型选择.

    模型列表:
        - ARMA(p, q): 传统统计时序模型
        - CNN: 1D卷积时序特征提取
        - TDNN: Time-Delay Neural Network
        - RNN: 基础循环网络
        - LSTM: 长短期记忆网络

    Args:
        seq_len: Input sequence length (lookback window) for NN models.
        output_horizon: Number of steps ahead to predict.
        test_ratio: Fraction of data to use for testing.
        random_seed: Random seed for reproducibility.

    Example:
        >>> suite = ModelComparisonSuite(seq_len=30, output_horizon=1)
        >>> suite.add_model("ARMA", ARMAModel(p=2, q=1))
        >>> suite.add_model("LSTM", LSTMModel(input_dim=1, hidden_size=64))
        >>> results = suite.train_all(train_data, val_data)
        >>> eval_df = suite.evaluate(test_data)
        >>> best = suite.get_best_model(metric='sharpe')
        >>> suite.plot_predictions()
    """

    def __init__(
        self,
        seq_len: int = 30,
        output_horizon: int = 1,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ):
        self.seq_len = seq_len
        self.output_horizon = output_horizon
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self._models: dict[str, BaseModel] = {}
        self._train_history: dict[str, list] = {}
        self._predictions: dict[str, np.ndarray] = {}
        self._ground_truth: np.ndarray | None = None
        self._test_indices: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def add_model(self, name: str, model: BaseModel) -> None:
        """Register a model to be evaluated.

        Args:
            name: Human-readable name for the model.
            model: An instance of a BaseModel subclass.
        """
        if not isinstance(model, BaseModel):
            raise TypeError(f"Model must be a BaseModel subclass, got {type(model)}")
        self._models[name] = model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_all(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train all registered models on the training data.

        Args:
            train_data: 1D or 2D training array. If 2D, each column is a feature.
            val_data: Optional validation data for early stopping / monitoring.

        Returns:
            Dictionary mapping model names to per-epoch loss history lists.
        """
        np.random.seed(self.random_seed)
        results = {}

        for name, model in self._models.items():
            print(f"\n{'='*50}")
            print(f"Training model: {name}")
            print(f"{'='*50}")

            # Inject seq_len and output_horizon if the model supports them
            if hasattr(model, "seq_len"):
                model.seq_len = self.seq_len
            if hasattr(model, "output_horizon"):
                model.output_horizon = self.output_horizon

            model.fit(train_data)

            # Capture a simple training metric (reconstruction error on train tail)
            if hasattr(model, "_model") and model._model is not None:
                # For PyTorch models — do a forward pass to get train loss
                try:
                    import torch
                    model._model.eval()
                    with torch.no_grad():
                        seq = torch.from_numpy(
                            train_data[-self.seq_len:].astype(np.float32)
                        ).float().to(model._model.training or True)
                        pred = model.predict(1)
                    train_loss = float(np.mean((train_data[-self.seq_len:, 0] - pred) ** 2))
                except Exception:
                    train_loss = None
            else:
                train_loss = None

            results[name] = {"train_loss": train_loss}
            self._train_history[name] = results[name]

        return results

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, test_data: np.ndarray) -> pd.DataFrame:
        """Evaluate all models on test data and return comparison metrics.

        Metrics computed:
            - MAE: Mean Absolute Error
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - Sharpe: Sharpe ratio of predicted returns
            - Directional Accuracy: % of correct direction predictions
            - MAPE: Mean Absolute Percentage Error

        Args:
            test_data: Held-out test dataset (same format as train data).

        Returns:
            DataFrame with one row per model and one column per metric.
        """
        if len(test_data) < self.seq_len + self.output_horizon:
            raise ValueError(
                f"test_data too short: need at least {self.seq_len + self.output_horizon} rows, "
                f"got {len(test_data)}"
            )

        self._ground_truth = test_data.copy()
        records = []

        for name, model in self._models.items():
            try:
                preds = model.predict(horizon=self.output_horizon)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                preds = np.zeros(self.output_horizon)

            self._predictions[name] = preds

            # Align ground truth
            y_true = test_data[self.seq_len:self.seq_len + self.output_horizon, 0]

            mae = _mae(y_true, preds)
            mse = _mse(y_true, preds)
            rmse = _rmse(y_true, preds)
            dir_acc = _directional_accuracy(y_true, preds)

            # Returns based on predicted vs actual price change
            true_returns = np.diff(y_true)
            pred_returns = np.diff(preds)
            sharpe = _sharpe_ratio(true_returns)

            # MAPE (avoid division by zero)
            mape = np.mean(np.abs((y_true - preds) / (np.abs(y_true) + 1e-8))) * 100

            records.append({
                "model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "Sharpe": sharpe,
                "Dir_Acc": dir_acc,
                "MAPE": mape,
            })

        self._metrics_df = pd.DataFrame(records)
        self._metrics_df = self._metrics_df.sort_values("Sharpe", ascending=False)
        return self._metrics_df

    # ------------------------------------------------------------------ #
    # Best model
    # ------------------------------------------------------------------ #

    def get_best_model(self, metric: str = "sharpe") -> str:
        """Return the name of the best model according to a given metric.

        Args:
            metric: One of 'MAE', 'MSE', 'RMSE', 'Sharpe', 'Dir_Acc', 'MAPE'.
                Lower-is-better for MAE/MSE/RMSE/MAPE; higher-is-better for Sharpe/Dir_Acc.

        Returns:
            Name of the best model.
        """
        if not hasattr(self, "_metrics_df") or self._metrics_df is None:
            raise ValueError("Must call evaluate() before get_best_model()")

        ascending = metric in ("MAE", "MSE", "RMSE", "MAPE")
        best_row = self._metrics_df.sort_values(metric, ascending=ascending).iloc[0]
        return best_row["model"]

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_predictions(self, ax=None) -> None:
        """Plot ground truth vs all model predictions on a single axes object.

        Args:
            ax: Optional matplotlib Axes. If None, a new figure is created.
        """
        if self._ground_truth is None:
            raise ValueError("Must call evaluate() before plot_predictions()")

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        y_true = self._ground_truth[:, 0]
        x = np.arange(len(y_true))

        ax.plot(x, y_true, label="Ground Truth", color="black", linewidth=2)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, (name, pred) in enumerate(self._predictions.items()):
            # Align prediction to the correct x positions
            pred_x = np.arange(self.seq_len, self.seq_len + len(pred))
            ax.plot(pred_x, pred, label=name, color=colors[i % len(colors)], linewidth=1.5)

        ax.set_xlabel("Time Index")
        ax.set_ylabel("Price / Value")
        ax.set_title("Model Comparison: Ground Truth vs Predictions")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_metrics(self, ax=None) -> None:
        """Plot a bar chart of evaluation metrics across all models.

        Args:
            ax: Optional matplotlib Axes.
        """
        if not hasattr(self, "_metrics_df") or self._metrics_df is None:
            raise ValueError("Must call evaluate() before plot_metrics()")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        df = self._metrics_df.copy()
        metric_cols = ["MAE", "RMSE", "Sharpe", "Dir_Acc"]
        x = np.arange(len(df))
        width = 0.2

        for i, col in enumerate(metric_cols):
            vals = df[col].values.astype(float)
            if col in ("MAE", "RMSE"):
                vals = vals / vals.max()  # Normalize for visual comparison
            ax.bar(x + i * width, vals, width, label=col)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df["model"].values, rotation=15)
        ax.set_ylabel("Score (normalized)")
        ax.set_title("Model Performance Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

    # ------------------------------------------------------------------ #
    # Convenience factory
    # ------------------------------------------------------------------ #

    @classmethod
    def default_suite(
        cls,
        input_dim: int = 1,
        seq_len: int = 30,
        output_horizon: int = 1,
        nn_epochs: int = 50,
    ) -> "ModelComparisonSuite":
        """Factory: create a suite pre-loaded with all default models.

        Args:
            input_dim: Number of input features.
            seq_len: Lookback window length.
            output_horizon: Prediction horizon.
            nn_epochs: Epochs for NN model training.

        Returns:
            Configured ModelComparisonSuite with all models registered.
        """
        suite = cls(seq_len=seq_len, output_horizon=output_horizon)

        suite.add_model("ARMA(2,1)", ARMAModel(p=2, q=1))
        suite.add_model(
            "CNN1D",
            CNN1DModel(
                input_dim=input_dim,
                filters=64,
                kernel_size=3,
                hidden_size=32,
                num_layers=2,
                dropout=0.2,
                lr=1e-3,
                epochs=nn_epochs,
            ),
        )
        suite.add_model(
            "TDNN",
            TDNNModel(
                input_dim=input_dim,
                hidden_sizes=[64, 32],
                output_horizon=output_horizon,
                seq_len=seq_len,
                dropout=0.2,
                lr=1e-3,
                epochs=nn_epochs,
            ),
        )
        suite.add_model(
            "RNN",
            RNNModel(
                input_dim=input_dim,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                output_horizon=output_horizon,
                seq_len=seq_len,
                lr=1e-3,
                epochs=nn_epochs,
            ),
        )
        suite.add_model(
            "LSTM",
            LSTMModel(
                input_dim=input_dim,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                output_horizon=output_horizon,
                seq_len=seq_len,
                lr=1e-3,
                epochs=nn_epochs,
            ),
        )
        return suite
