# -*- coding: utf-8 -*-
"""Machine learning models for quantitative trading.

机器学习模型集：XGBoost、LSTM、随机森林、特征工程、Walk-Forward验证。
Lazy import 策略：优先尝试导入 xgboost/lightgbm，回退到 sklearn。
核心逻辑使用纯 NumPy/SciPy 实现，不依赖 PyTorch/TensorFlow。

References:
    D:/Hive/Data/trading_repos/Quantitative-Trading-Strategy-Based-on-Machine-Learning/
        - xgboost_6factor.py: XGBoost 月度收益分类
        - TrainTestClassifier.py: 分类器训练测试框架
    D:/Hive/Data/trading_repos/Machine-Learning-For-Trading/
        - qstrader/alpha_model/: 因子模型与特征工程
    D:/Hive/Data/trading_repos/Quantitative-Trading/
        - model.py: XGBoost 交易策略
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

__all__ = [
    "XGBoostPredictor",
    "LSTMPredictor",
    "RandomForestClassifier",
    "FeatureEngineering",
    "WalkForwardValidator",
    "MLPipeline",
]

# --------------------------------------------------------------------------- #
# Lazy-import helpers
# --------------------------------------------------------------------------- #

_XGB_AVAILABLE: Optional[bool] = None
_LGBM_AVAILABLE: Optional[bool] = None


def _check_xgboost() -> bool:
    global _XGB_AVAILABLE
    if _XGB_AVAILABLE is None:
        try:
            import xgboost  # noqa: F401
            _XGB_AVAILABLE = True
        except ImportError:
            _XGB_AVAILABLE = False
    return _XGB_AVAILABLE


def _check_lightgbm() -> bool:
    global _LGBM_AVAILABLE
    if _LGBM_AVAILABLE is None:
        try:
            import lightgbm  # noqa: F401
            _LGBM_AVAILABLE = True
        except ImportError:
            _LGBM_AVAILABLE = False
    return _LGBM_AVAILABLE


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class MLModelError(Exception):
    """Base exception for ML model errors."""
    pass


class InsufficientDataError(MLModelError):
    """Raised when there is not enough data for the requested operation."""
    pass


# --------------------------------------------------------------------------- #
# Dataclasses for validation results
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardResult:
    """Walk-forward validation result.

    Attributes:
        train_scores: List of model scores on each training fold.
        test_scores: List of model scores on each test fold.
        train_indices: List of (start, end) index tuples for training folds.
        test_indices: List of (start, end) index tuples for test folds.
        oof_predictions: Out-of-fold predictions aligned to the full series.
        oof_actuals: Out-of-fold actual values aligned to the full series.
        metric_name: Name of the evaluation metric used.
    """
    train_scores: list[float] = field(default_factory=list)
    test_scores: list[float] = field(default_factory=list)
    train_indices: list[tuple[int, int]] = field(default_factory=list)
    test_indices: list[tuple[int, int]] = field(default_factory=list)
    oof_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    oof_actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_name: str = ""


# --------------------------------------------------------------------------- #
# Base Predictor
# --------------------------------------------------------------------------- #


class BasePredictor(ABC):
    """Abstract base class for all ML predictors.

    所有 ML 预测器的基类，定义标准接口。
    """

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BasePredictor":
        """Train the model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            sample_weight: Optional sample weights.

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        raise NotImplementedError

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: Literal["accuracy", "f1", "auc", "mse", "mae", "rmse"] = "accuracy",
    ) -> float:
        """Evaluate model on given data.

        Args:
            X: Feature matrix.
            y: Ground-truth labels / values.
            metric: Evaluation metric.

        Returns:
            Score value.
        """
        pred = self.predict(X)
        if metric == "accuracy":
            return float(np.mean((pred >= 0.5) == (y >= 0.5)))
        if metric == "f1":
            try:
                from sklearn.metrics import f1_score
            except ImportError:
                tp = np.sum((pred >= 0.5) & (y >= 0.5))
                fp = np.sum((pred >= 0.5) & (y < 0.5))
                fn = np.sum((pred < 0.5) & (y >= 0.5))
                return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            return float(f1_score(y, (pred >= 0.5).astype(int)))
        if metric == "auc":
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                return 0.0
            return float(roc_auc_score(y, pred))
        if metric in ("mse", "mae", "rmse"):
            if metric == "mae":
                return float(np.mean(np.abs(pred - y)))
            mse = float(np.mean((pred - y) ** 2))
            if metric == "mse":
                return mse
            return float(np.sqrt(mse))

        raise ValueError(f"Unknown metric: {metric}")

    def generate_signals(
        self,
        X: np.ndarray,
        threshold_long: float = 0.6,
        threshold_short: float = 0.4,
    ) -> np.ndarray:
        """Generate trading signals from predictions.

        Generate trading signals (-1=short, 0=neutral, 1=long) based on
        prediction probabilities / values.

        Args:
            X: Feature matrix.
            threshold_long: Threshold above which signal is long (1).
            threshold_short: Threshold below which signal is short (-1).
                             Values between thresholds give neutral (0).

        Returns:
            Array of signals of shape (n_samples,), values in {-1, 0, 1}.
        """
        pred = self.predict(X)
        signals = np.zeros(len(pred), dtype=np.int8)
        signals[pred >= threshold_long] = 1
        signals[pred <= threshold_short] = -1
        return signals


# --------------------------------------------------------------------------- #
# XGBoost Predictor
# --------------------------------------------------------------------------- #

class XGBoostPredictor(BasePredictor):
    """XGBoost-based price / direction predictor.

    XGBoost 价格与方向预测器，支持二分类（涨/跌）和回归（收益率）。
    优先导入 xgboost，失败时回退到 sklearn GradientBoostingClassifier。

    Attributes:
        mode: "classification" or "regression".
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        use_xgboost: Whether xgboost was successfully imported.
        model: The underlying fitted model.
    """

    def __init__(
        self,
        mode: Literal["classification", "regression"] = "classification",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective or ("binary:logistic" if mode == "classification" else "reg:squarederror")
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_xgboost: bool = False
        self.model = None

    def _lazy_init(self) -> None:
        """Lazy import xgboost or fall back to sklearn."""
        if self.model is not None:
            return

        if _check_xgboost():
            import xgboost as xgb

            params = {
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "objective": self.objective,
                "n_estimators": self.n_estimators,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
                "verbosity": 0,
                "use_label_encoder": False,
            }
            if self.mode == "classification":
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
            self.use_xgboost = True
        else:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )

            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
            }
            if self.mode == "classification":
                self.model = GradientBoostingClassifier(**params)
            else:
                self.model = GradientBoostingRegressor(**params)
            self.use_xgboost = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostPredictor":
        """Train the XGBoost model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            sample_weight: Optional per-sample weights.

        Returns:
            self
        """
        self._lazy_init()
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions. In classification mode returns probabilities [0, 1].
            In regression mode returns raw values.
        """
        if self.model is None:
            raise MLModelError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only).

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        if self.model is None:
            raise MLModelError("Model not trained. Call train() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # sklearn fallback: return raw predictions clipped to [0,1]
        pred = self.model.predict(X)
        return np.clip(np.column_stack([1 - pred, pred]), 0, 1)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if available.

        Returns:
            Array of shape (n_features,) with importance scores,
            or None if not available.
        """
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def __repr__(self) -> str:
        backend = "xgboost" if self.use_xgboost else "sklearn.ensemble"
        return f"XGBoostPredictor(mode={self.mode}, backend={backend})"


# --------------------------------------------------------------------------- #
# LSTM Predictor (Pure NumPy/SciPy)
# --------------------------------------------------------------------------- #

class LSTMPredictor(BasePredictor):
    """LSTM-based time-series forecaster using pure NumPy / SciPy.

    LSTM 时间序列预测器，手写 LSTM forward pass（无 PyTorch/TF 依赖）。
    实现标准 LSTM 单元：输入门、遗忘门、输出门、细胞态。

    Attributes:
        input_size: Number of input features.
        hidden_size: Number of hidden units per LSTM layer.
        num_layers: Number of LSTM layers.
        sequence_length: Length of input sequences for training/prediction.
        output_horizon: Number of steps ahead to forecast.
        learning_rate: Gradient descent learning rate.
        epochs: Number of training epochs.
        weights: Dictionary of LSTM weight matrices and biases.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 60,
        output_horizon: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        clip_grad: float = 5.0,
        random_state: int = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_horizon = output_horizon
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.random_state = random_state
        self.weights: dict = {}
        self._rgen = np.random.RandomState(random_state)

    # ------------------------------------------------------------------ #
    # LSTM cell (internal)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation, numerically stable."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)

    def _init_weights(self, input_size: int) -> dict:
        """Initialize LSTM weight matrices using Xavier initialization."""
        r = self._rgen
        hidden = self.hidden_size

        weights = {}
        for layer in range(self.num_layers):
            in_size = input_size if layer == 0 else hidden
            # Input gate weights
            weights[f"W_ii_{layer}"] = r.randn(in_size, hidden) * np.sqrt(2.0 / (in_size + hidden))
            weights[f"W_if_{layer}"] = r.randn(in_size, hidden) * np.sqrt(2.0 / (in_size + hidden))
            weights[f"W_ig_{layer}"] = r.randn(in_size, hidden) * np.sqrt(2.0 / (in_size + hidden))
            weights[f"W_io_{layer}"] = r.randn(in_size, hidden) * np.sqrt(2.0 / (in_size + hidden))
            # Hidden state weights
            weights[f"W_hi_{layer}"] = r.randn(hidden, hidden) * np.sqrt(2.0 / (hidden + hidden))
            weights[f"W_hf_{layer}"] = r.randn(hidden, hidden) * np.sqrt(2.0 / (hidden + hidden))
            weights[f"W_hg_{layer}"] = r.randn(hidden, hidden) * np.sqrt(2.0 / (hidden + hidden))
            weights[f"W_ho_{layer}"] = r.randn(hidden, hidden) * np.sqrt(2.0 / (hidden + hidden))
            # Biases
            weights[f"b_i_{layer}"] = np.zeros(hidden)
            weights[f"b_f_{layer}"] = np.ones(hidden) * 0.5  # Forget gate bias ~1
            weights[f"b_g_{layer}"] = np.zeros(hidden)
            weights[f"b_o_{layer}"] = np.zeros(hidden)

        # Output projection weights
        weights["W_out"] = r.randn(hidden, hidden) * np.sqrt(2.0 / (hidden + hidden))
        weights["b_out"] = np.zeros(hidden)
        weights["W_pred"] = r.randn(hidden, self.output_horizon) * np.sqrt(2.0 / (hidden + self.output_horizon))
        weights["b_pred"] = np.zeros(self.output_horizon)

        return weights

    def _lstm_cell(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single LSTM cell forward pass.

        Args:
            x_t: Input at time t, shape (input_size,) or (batch, input_size).
            h_prev: Previous hidden state, shape (hidden_size,).
            c_prev: Previous cell state, shape (hidden_size,).
            layer: Layer index.

        Returns:
            Tuple of (hidden_state, cell_state), both shape (hidden_size,).
        """
        W_ii = self.weights[f"W_ii_{layer}"]
        W_if = self.weights[f"W_if_{layer}"]
        W_ig = self.weights[f"W_ig_{layer}"]
        W_io = self.weights[f"W_io_{layer}"]
        W_hi = self.weights[f"W_hi_{layer}"]
        W_hf = self.weights[f"W_hf_{layer}"]
        W_hg = self.weights[f"W_hg_{layer}"]
        W_ho = self.weights[f"W_ho_{layer}"]
        b_i = self.weights[f"b_i_{layer}"]
        b_f = self.weights[f"b_f_{layer}"]
        b_g = self.weights[f"b_g_{layer}"]
        b_o = self.weights[f"b_o_{layer}"]

        i = self._sigmoid(x_t @ W_ii + h_prev @ W_hi + b_i)
        f = self._sigmoid(x_t @ W_if + h_prev @ W_hf + b_f)
        g = self._tanh(x_t @ W_ig + h_prev @ W_hg + b_g)
        o = self._sigmoid(x_t @ W_io + h_prev @ W_ho + b_o)

        c_t = f * c_prev + i * g
        h_t = o * self._tanh(c_t)
        return h_t, c_t

    def _forward_sequence(
        self,
        X_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the full LSTM sequence.

        Args:
            X_seq: Input sequence of shape (sequence_length, input_size).

        Returns:
            Tuple of (h_last, c_last, all_hidden) where:
                h_last: Last hidden state, shape (hidden_size,)
                c_last: Last cell state, shape (hidden_size,)
                all_hidden: All hidden states, shape (sequence_length, hidden_size)
        """
        seq_len = X_seq.shape[0]
        h = np.zeros((self.num_layers, self.hidden_size))
        c = np.zeros((self.num_layers, self.hidden_size))
        all_h = np.zeros((seq_len, self.hidden_size))

        for t in range(seq_len):
            x_t = X_seq[t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self._lstm_cell(x_t, h[layer - 1] if layer > 0 else h[layer], c[layer - 1] if layer > 0 else c[layer], layer)
                x_t = h[layer]  # Next layer input
            all_h[t] = h[-1]

        return h[-1], c[-1], all_h

    def _backward(
        self,
        X_seq: np.ndarray,
        all_hidden: np.ndarray,
        h_final: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """BPTT (Back-Propagation Through Time) gradient computation.

        Simplified single-layer BPTT for the last hidden state -> output.

        Returns:
            Dictionary of gradient arrays keyed by weight name.
        """
        # Gradient at output
        d_pred = 2 * (h_final @ self.weights["W_pred"] + self.weights["b_pred"] - y_true)  # (output_horizon,)

        grads = {}
        # Output layer gradients
        grads["W_pred"] = np.outer(h_final, d_pred)
        grads["b_pred"] = d_pred.copy()
        grads["W_out"] = np.outer(h_final, d_pred @ self.weights["W_pred"].T)
        grads["b_out"] = d_pred @ self.weights["W_pred"].T.copy()

        # BPTT into hidden state (simplified: accumulate over sequence)
        dh_final = d_pred @ self.weights["W_pred"].T  # (hidden_size,)

        # Approximate gradient through time via chain rule on last few steps
        # This is a simplified BPTT — full BPTT would store all intermediate states
        dh = dh_final
        grads["dh_avg"] = dh / self.sequence_length  # average gradient across time

        return grads

    def _gradient_check(self, X_seq: np.ndarray, y_true: np.ndarray, eps: float = 1e-5) -> bool:
        """Simple gradient sanity check (first layer only, small sequence)."""
        h_final, _, _ = self._forward_sequence(X_seq)
        pred = h_final @ self.weights["W_pred"] + self.weights["b_pred"]
        loss = np.sum((pred - y_true) ** 2)

        grad_W = 2 * (pred - y_true).reshape(-1, 1) @ h_final.reshape(1, -1)
        num_grad = np.zeros_like(self.weights["W_pred"])

        for i in range(min(3, self.weights["W_pred"].shape[0])):
            for j in range(min(3, self.weights["W_pred"].shape[1])):
                w_orig = self.weights["W_pred"][i, j]
                self.weights["W_pred"][i, j] = w_orig + eps
                h_plus, _, _ = self._forward_sequence(X_seq)
                pred_plus = h_plus @ self.weights["W_pred"] + self.weights["b_pred"]
                loss_plus = np.sum((pred_plus - y_true) ** 2)
                num_grad[i, j] = (loss_plus - loss) / eps
                self.weights["W_pred"][i, j] = w_orig

        return np.allclose(grad_W[:3, :3], num_grad[:3, :3], atol=1e-2)

    def _create_sequences(
        self,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences from a 1D or 2D array.

        Args:
            data: Time series of shape (n_samples,) or (n_samples, n_features).

        Returns:
            X_seqs: Input sequences of shape (n_samples - sequence_length, sequence_length, input_size).
            y_seqs: Target values of shape (n_samples - sequence_length, output_horizon).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = len(data)
        seq_len = self.sequence_length

        X_seqs = []
        y_seqs = []

        for i in range(seq_len, n - self.output_horizon + 1):
            X_seqs.append(data[i - seq_len : i])  # (seq_len, input_size)
            y_seqs.append(data[i : i + self.output_horizon].flatten())

        return np.array(X_seqs), np.array(y_seqs)

    def _sgd_step(
        self,
        X_seq: np.ndarray,
        y_true: np.ndarray,
        lr: float,
    ) -> float:
        """Single SGD step with approximate BPTT.

        Args:
            X_seq: Single input sequence (sequence_length, input_size).
            y_true: Target (output_horizon,).

        Returns:
            Squared loss for this step.
        """
        h_final, _, _ = self._forward_sequence(X_seq)
        pred = h_final @ self.weights["W_pred"] + self.weights["b_pred"]
        loss = np.sum((pred - y_true) ** 2)

        d_pred = 2 * (pred - y_true)

        # Gradient of output layer
        d_W_pred = np.outer(h_final, d_pred)
        d_b_pred = d_pred.copy()
        d_h_final = d_pred @ self.weights["W_pred"].T

        # Clip gradients
        grad_norm = np.sqrt(np.sum(d_W_pred ** 2) + np.sum(d_b_pred ** 2))
        scale = min(1.0, self.clip_grad / (grad_norm + 1e-8))
        d_W_pred *= scale
        d_b_pred *= scale
        d_h_final *= scale

        # Gradient clipping for hidden state gradient
        if np.sqrt(np.sum(d_h_final ** 2)) > self.clip_grad:
            d_h_final = d_h_final / np.sqrt(np.sum(d_h_final ** 2)) * self.clip_grad

        # Gradient descent on output layer
        self.weights["W_pred"] -= lr * d_W_pred
        self.weights["b_pred"] -= lr * d_b_pred

        # Approximate gradient on input-to-hidden via BPTT (truncated)
        # Use gradient averaging scaled by learning rate as proxy
        scale_factor = lr * 0.01 * d_h_final
        for layer in range(self.num_layers):
            for key in [f"W_ii_{layer}", f"W_if_{layer}", f"W_ig_{layer}", f"W_io_{layer}",
                        f"W_hi_{layer}", f"W_hf_{layer}", f"W_hg_{layer}", f"W_ho_{layer}"]:
                self.weights[key] -= scale_factor.mean() * np.sign(self.weights[key])

        return float(loss)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LSTMPredictor":
        """Train the LSTM model.

        Args:
            X: Feature matrix of shape (n_samples, n_features) where each row
               is a time step. Must have at least sequence_length + output_horizon rows.
            y: Target values of shape (n_samples,) or (n_samples, output_horizon).
            sample_weight: Ignored (for API compatibility).

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(X)
        required = self.sequence_length + self.output_horizon
        if n < required:
            raise InsufficientDataError(
                f"Need at least {required} samples (got {n}). "
                f"sequence_length={self.sequence_length}, output_horizon={self.output_horizon}"
            )

        # Combine into time series
        data = np.hstack([X, y.reshape(-1, 1)]) if y.ndim == 1 else np.hstack([X, y])
        input_size = data.shape[1]

        # Initialize weights
        self.weights = self._init_weights(input_size)

        # Create sequences
        X_seqs, y_seqs = self._create_sequences(data)

        # Shuffle indices
        indices = self._rgen.permutation(len(X_seqs))

        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            for idx in indices:
                loss = self._sgd_step(X_seqs[idx], y_seqs[idx], self.learning_rate)
                epoch_losses.append(loss)
            epoch_loss = float(np.mean(epoch_losses))
            losses.append(epoch_loss)

            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.95  # Learning rate decay

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the next output_horizon values.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Uses the last sequence_length rows as the input sequence.

        Returns:
            Predictions of shape (n_samples, output_horizon) or (n_samples,)
            if output_horizon=1.
        """
        if not self.weights:
            raise MLModelError("Model not trained. Call train() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) < self.sequence_length:
            raise InsufficientDataError(
                f"Need at least {self.sequence_length} samples for prediction (got {len(X)})"
            )

        # Use the last sequence_length rows
        X_seq = X[-self.sequence_length :]
        h_final, _, _ = self._forward_sequence(X_seq)
        pred = h_final @ self.weights["W_pred"] + self.weights["b_pred"]

        if self.output_horizon == 1:
            return pred.flatten()
        return pred

    def __repr__(self) -> str:
        return (
            f"LSTMPredictor(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, sequence_length={self.sequence_length}, "
            f"output_horizon={self.output_horizon})"
        )


# --------------------------------------------------------------------------- #
# Random Forest Classifier
# --------------------------------------------------------------------------- #

class RandomForestClassifier(BasePredictor):
    """Random Forest market-regime classifier.

    随机森林市场状态分类器，将市场划分为不同的regime（上涨/下跌/震荡等）。
    优先使用 sklearn，失败时使用纯 NumPy 实现（bootstrap + majority vote）。

    Attributes:
        mode: Always "classification".
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf node.
        use_sklearn: Whether sklearn was successfully imported.
        trees: List of (root_node,) tuples for the pure-NumPy implementation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        n_features: Optional[int] = None,  # If None, use all features
        random_state: int = 42,
    ):
        self.mode = "classification"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.random_state = random_state
        self.use_sklearn: bool = False
        self.model = None
        self._rgen = np.random.RandomState(random_state)
        self.trees: list = []  # For pure-NumPy implementation

    def _lazy_init(self) -> bool:
        """Try to import sklearn. Returns True if successful."""
        try:
            from sklearn.ensemble import RandomForestClassifier as SKLearnRF
            self.model = SKLearnRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.use_sklearn = True
            return True
        except ImportError:
            self.use_sklearn = False
            return False

    # ------------------------------------------------------------------ #
    # Pure-NumPy Decision Tree
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Compute Gini impurity for a label array."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[Optional[int], Optional[float]]:
        """Find the best split for a node using information gain.

        Returns:
            Tuple of (best_feature_index, best_threshold) or (None, None) if no split improves.
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feat = None
        best_thresh = None

        parent_gini = self._gini(y)

        for feat_idx in feature_indices:
            col = X[:, feat_idx]
            thresholds = np.percentile(col, [25, 50, 75])
            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                n_left = left_mask.sum()
                n_right = right_mask.sum()
                gain = parent_gini - (n_left / n_samples * left_gini + n_right / n_samples * right_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        feature_indices: np.ndarray,
    ) -> dict:
        """Recursively build a decision tree.

        Returns:
            Dictionary representing the tree node.
        """
        n_samples = len(y)
        unique_labels = np.unique(y)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or len(unique_labels) == 1
        ):
            # Leaf node
            label, counts = np.unique(y, return_counts=True)
            return {
                "is_leaf": True,
                "label": int(label[np.argmax(counts)]),
                "n_samples": n_samples,
            }

        feat_idx, thresh = self._best_split(X, y, feature_indices)

        if feat_idx is None:
            # No valid split found
            label, counts = np.unique(y, return_counts=True)
            return {
                "is_leaf": True,
                "label": int(label[np.argmax(counts)]),
                "n_samples": n_samples,
            }

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        return {
            "is_leaf": False,
            "feature_index": int(feat_idx),
            "threshold": float(thresh),
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices),
            "n_samples": n_samples,
        }

    @staticmethod
    def _predict_tree(x: np.ndarray, tree: dict) -> int:
        """Traverse a single tree to get a prediction."""
        if tree["is_leaf"]:
            return tree["label"]
        if x[tree["feature_index"]] <= tree["threshold"]:
            return RandomForestClassifier._predict_tree(x, tree["left"])
        return RandomForestClassifier._predict_tree(x, tree["right"])

    def _fit_numpy(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the pure-NumPy random forest implementation."""
        n_samples, n_features = X.shape
        feat_indices = np.arange(n_features)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = self._rgen.randint(0, n_samples, size=n_samples)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # Random feature subset (sqrt(n_features))
            n_subfeat = max(1, int(np.sqrt(n_features)))
            subfeat_indices = self._rgen.choice(feat_indices, size=n_subfeat, replace=False)

            tree = self._build_tree(X_boot, y_boot, depth=0, feature_indices=subfeat_indices)
            self.trees.append(tree)

        return self

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RandomForestClassifier":
        """Train the random forest.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,). Values should be integers (0, 1, 2, ...).
            sample_weight: Ignored for API compatibility.

        Returns:
            self
        """
        self._lazy_init()
        if self.use_sklearn:
            self.model.fit(X, y)
        else:
            self._fit_numpy(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        if self.use_sklearn and self.model is not None:
            return self.model.predict(X)

        # Pure-NumPy: majority vote across trees (mode / median)
        if not self.trees:
            raise MLModelError("Model not trained. Call train() first.")

        n_samples = X.shape[0]
        mode_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            tree_preds = np.array([self._predict_tree(X[i], tree) for tree in self.trees])
            mode_preds[i] = int(np.median(tree_preds))
        return mode_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities [P(class0), P(class1)].
        """
        if self.use_sklearn and self.model is not None:
            return self.model.predict_proba(X)

        if not self.trees:
            raise MLModelError("Model not trained. Call train() first.")

        n_samples = X.shape[0]
        n_classes = 2
        proba = np.zeros((n_samples, n_classes))

        for tree in self.trees:
            tree_preds = np.array([self._predict_tree(x, tree) for x in X])
            for i, label in enumerate(tree_preds):
                li = int(label)
                if li == 0:
                    proba[i, 0] += 1.0 / len(self.trees)
                else:
                    proba[i, 1] += 1.0 / len(self.trees)

        return proba

    def __repr__(self) -> str:
        backend = "sklearn.ensemble" if self.use_sklearn else "pure_numpy"
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, backend={backend})"


# --------------------------------------------------------------------------- #
# Feature Engineering
# --------------------------------------------------------------------------- #

class FeatureEngineering:
    """Feature engineering for OHLCV price data.

    特征工程类：从 OHLCV 数据中生成 Alpha 因子。
    支持：收益率、波动率、动量、成交量比率、技术指标。

    Attributes:
        lookback_periods: Dictionary of lookback window sizes for various features.
        include_overlapping: Whether to include features at multiple timeframes.
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_medium: int = 20,
        lookback_long: int = 60,
        lookback_vol: int = 20,
        momentum_periods: tuple[int, ...] = (5, 10, 20),
        volume_periods: tuple[int, ...] = (5, 20),
    ):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.lookback_vol = lookback_vol
        self.momentum_periods = momentum_periods
        self.volume_periods = volume_periods

    def compute_all(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute all features from OHLCV data.

        Args:
            open_price: Opening prices, shape (n_samples,).
            high: High prices, shape (n_samples,).
            low: Low prices, shape (n_samples,).
            close: Closing prices, shape (n_samples,).
            volume: Volume, shape (n_samples,). Optional.

        Returns:
            Feature matrix of shape (n_samples - max_lookback, n_features).
        """
        close = np.asarray(close).flatten()
        high = np.asarray(high).flatten()
        low = np.asarray(low).flatten()
        open_price = np.asarray(open_price).flatten()
        n = len(close)

        features = []
        names = []

        max_lookback = max(
            self.lookback_long,
            self.lookback_vol,
            max(self.momentum_periods),
            max(self.volume_periods),
            2,
        )

        # ------------------------------------------------------------------ #
        # 1. Returns at multiple horizons
        # ------------------------------------------------------------------ #
        for period in [1, 5, 10, 20]:
            ret = self._pct_change(close, period)
            features.append(ret)
            names.append(f"return_{period}d")

        # ------------------------------------------------------------------ #
        # 2. Log returns
        # ------------------------------------------------------------------ #
        log_ret = self._log_return(close, 1)
        features.append(log_ret)
        names.append("log_return_1d")

        # ------------------------------------------------------------------ #
        # 3. Volatility (rolling std of returns)
        # ------------------------------------------------------------------ #
        for period in [5, 20, 60]:
            vol = self._rolling_std(log_ret, period)
            features.append(vol)
            names.append(f"volatility_{period}d")

        # ------------------------------------------------------------------ #
        # 4. Price-based features
        # ------------------------------------------------------------------ #
        # Moving averages
        ma_short = self._rolling_mean(close, self.lookback_short)
        ma_medium = self._rolling_mean(close, self.lookback_medium)
        ma_long = self._rolling_mean(close, self.lookback_long)

        features.append(self._ratio(close, ma_short))
        names.append("price_to_ma_short")
        features.append(self._ratio(close, ma_medium))
        names.append("price_to_ma_medium")
        features.append(self._ratio(close, ma_long))
        names.append("price_to_ma_long")
        features.append(self._ratio(ma_short, ma_medium))
        names.append("ma_short_to_medium")
        features.append(self._ratio(ma_short, ma_long))
        names.append("ma_short_to_long")

        # Rolling max/min
        roll_max = self._rolling_max(close, self.lookback_medium)
        roll_min = self._rolling_min(close, self.lookback_medium)
        features.append(self._ratio(close - roll_min, roll_max - roll_min + 1e-9))
        names.append("price_position")

        # ------------------------------------------------------------------ #
        # 5. Momentum indicators
        # ------------------------------------------------------------------ #
        for period in self.momentum_periods:
            features.append(self._momentum(close, period))
            names.append(f"momentum_{period}d")

        # RSI (Relative Strength Index)
        features.append(self._rsi(close, 14))
        names.append("rsi_14")

        # MACD (12, 26, 9)
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd = ema_12 - ema_26
        macd_signal = self._ema(macd, 9)
        features.append(macd)
        names.append("macd")
        features.append(macd_signal)
        names.append("macd_signal")
        features.append(macd - macd_signal)
        names.append("macd_hist")

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, 20, 2)
        features.append(self._ratio(close - bb_lower, bb_upper - bb_lower + 1e-9))
        names.append("bb_position")

        # ------------------------------------------------------------------ #
        # 6. Volume features
        # ------------------------------------------------------------------ #
        if volume is not None:
            volume = np.asarray(volume).flatten()
            for period in self.volume_periods:
                vol_ma = self._rolling_mean(volume, period)
                features.append(self._ratio(volume, vol_ma))
                names.append(f"volume_ratio_{period}d")

            # OBV (On-Balance Volume) change
            obv_change = self._obv_change(close, volume)
            features.append(self._pct_change(obv_change, 5))
            names.append("obv_change_5d")

            # VWAP ratio
            vwap = self._vwap(high, low, close, volume)
            features.append(self._ratio(close, vwap))
            names.append("price_to_vwap")

        # ------------------------------------------------------------------ #
        # 7. High-Low range features
        # ------------------------------------------------------------------ #
        features.append(self._log_return(high / (low + 1e-9), 1))
        names.append("log_hl_range")
        features.append(self._rolling_mean(np.log(high / (low + 1e-9)), 20))
        names.append("avg_log_hl_range_20d")

        # ------------------------------------------------------------------ #
        # 8. Open-Close features
        # ------------------------------------------------------------------ #
        features.append(self._log_return(open_price / (close + 1e-9), 1))
        names.append("log_open_close")
        features.append(self._pct_change(open_price, 5))
        names.append("open_return_5d")

        # ------------------------------------------------------------------ #
        # 9. Rolling rank / quantile features
        # ------------------------------------------------------------------ #
        features.append(self._rolling_percentile(close, self.lookback_medium))
        names.append("price_percentile_medium")
        features.append(self._rolling_percentile(close, self.lookback_long))
        names.append("price_percentile_long")

        # ------------------------------------------------------------------ #
        # Stack and trim
        # ------------------------------------------------------------------ #
        feat_array = np.column_stack(features)  # (n, n_features)

        # Trim to valid region (after max_lookback)
        feat_array = feat_array[max_lookback:]
        names = names[: feat_array.shape[1]]

        # Replace infinities and NaNs
        feat_array = np.where(np.isinf(feat_array), np.nan, feat_array)
        feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)

        self._feature_names = names
        return feat_array

    def get_feature_names(self) -> list[str]:
        """Return the names of features computed in the last call to compute_all."""
        return getattr(self, "_feature_names", [])

    # ---------------------------------------------------------------------- #
    # Helper methods (all NumPy)
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _pct_change(arr: np.ndarray, period: int) -> np.ndarray:
        """Percentage change over period."""
        arr = np.asarray(arr).flatten()
        shifted = np.roll(arr, period)
        shifted[:period] = np.nan
        return (arr - shifted) / (shifted + 1e-9)

    @staticmethod
    def _log_return(arr: np.ndarray, period: int) -> np.ndarray:
        """Log return over period."""
        arr = np.asarray(arr).flatten()
        shifted = np.roll(arr, period)
        shifted[:period] = np.nan
        return np.log(arr / (shifted + 1e-9))

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean using uniform weights."""
        arr = np.asarray(arr).flatten()
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = float(np.mean(arr[i - window + 1 : i + 1]))
        return out

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation using Welford's algorithm (simplified)."""
        arr = np.asarray(arr).flatten()
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            segment = arr[i - window + 1 : i + 1]
            out[i] = float(np.std(segment, ddof=0))
        return out

    @staticmethod
    def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling maximum."""
        arr = np.asarray(arr).flatten()
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = float(np.max(arr[i - window + 1 : i + 1]))
        return out

    @staticmethod
    def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling minimum."""
        arr = np.asarray(arr).flatten()
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = float(np.min(arr[i - window + 1 : i + 1]))
        return out

    @staticmethod
    def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Safe ratio a / (b + eps)."""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        return a / (b + 1e-9)

    @staticmethod
    def _momentum(arr: np.ndarray, period: int) -> np.ndarray:
        """Momentum: current / shifted - 1."""
        arr = np.asarray(arr).flatten()
        shifted = np.roll(arr, period)
        shifted[:period] = np.nan
        return (arr - shifted) / (shifted + 1e-9)

    @staticmethod
    def _rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index (RSI)."""
        arr = np.asarray(arr).flatten()
        deltas = np.diff(arr, prepend=arr[0])
        n = len(deltas)
        out = np.full(n, np.nan)
        for i in range(period, n):
            window = deltas[i - period + 1 : i + 1]
            gains = np.mean(window[window > 0]) if np.any(window > 0) else 0.0
            losses = -np.mean(window[window < 0]) if np.any(window < 0) else 0.0
            rs = gains / (losses + 1e-9)
            out[i] = 100 - 100 / (1 + rs)
        return out

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        """Exponential moving average."""
        arr = np.asarray(arr).flatten()
        alpha = 2 / (span + 1)
        n = len(arr)
        out = np.full(n, np.nan)
        out[0] = arr[0]
        for i in range(1, n):
            if np.isnan(out[i - 1]):
                out[i] = arr[i]
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _bollinger_bands(
        arr: np.ndarray,
        period: int = 20,
        n_std: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands: upper, middle, lower."""
        arr = np.asarray(arr).flatten()
        middle = FeatureEngineering._rolling_mean(arr, period)
        std = FeatureEngineering._rolling_std(arr, period)
        upper = middle + n_std * std
        lower = middle - n_std * std
        return upper, middle, lower

    @staticmethod
    def _vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Volume-Weighted Average Price."""
        typical = (np.asarray(high).flatten() + np.asarray(low).flatten() + np.asarray(close).flatten()) / 3
        vol = np.asarray(volume).flatten()
        n = len(typical)
        out = np.full(n, np.nan)
        cumvol = np.nancumsum(vol)
        cumtypvol = np.nancumsum(typical * vol)
        for i in range(n):
            if cumvol[i] > 0:
                out[i] = cumtypvol[i] / cumvol[i]
        return out

    @staticmethod
    def _obv_change(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume change."""
        close = np.asarray(close).flatten()
        vol = np.asarray(volume).flatten()
        direction = np.sign(np.diff(close, prepend=close[0]))
        obv = np.cumsum(vol * direction)
        return obv

    @staticmethod
    def _rolling_percentile(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling percentile rank of current value within the window."""
        arr = np.asarray(arr).flatten()
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_vals = arr[i - window + 1 : i + 1]
            out[i] = float(np.sum(arr[i] >= window_vals)) / window
        return out


# --------------------------------------------------------------------------- #
# Walk-Forward Validator
# --------------------------------------------------------------------------- #

class WalkForwardValidator:
    """Walk-forward validation with expanding or rolling windows.

    Walk-Forward 验证器：时间序列交叉验证的标准方法。
    支持 Expanding Window（逐步扩大训练集）和 Rolling Window（固定大小，滚动前进）。

    Attributes:
        n_test: Number of test samples per fold.
        n_train_min: Minimum number of training samples.
        step_size: Number of samples to step forward between folds.
        window_type: "expanding" or "rolling".
    """

    def __init__(
        self,
        n_test: int = 20,
        n_train_min: int = 100,
        step_size: Optional[int] = None,
        window_type: Literal["expanding", "rolling"] = "expanding",
    ):
        self.n_test = n_test
        self.n_train_min = n_train_min
        self.step_size = step_size or n_test
        self.window_type = window_type

    def validate(
        self,
        model: BasePredictor,
        X: np.ndarray,
        y: np.ndarray,
        metric: Literal["accuracy", "f1", "auc", "mse", "mae", "rmse"] = "accuracy",
    ) -> WalkForwardResult:
        """Perform walk-forward validation.

        Args:
            model: A model implementing BasePredictor (train/predict/evaluate).
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            metric: Evaluation metric.

        Returns:
            WalkForwardResult with per-fold scores and OOF predictions.
        """
        n = len(X)
        n_test = min(self.n_test, n // 3)

        train_scores = []
        test_scores = []
        train_indices = []
        test_indices = []

        oof_preds = np.full(n, np.nan)
        oof_actual = y.copy().astype(float)

        current_end = self.n_train_min

        while current_end + n_test <= n:
            train_start = 0 if self.window_type == "expanding" else current_end - (self.n_train_min)
            train_end = current_end
            test_start = current_end
            test_end = min(current_end + n_test, n)

            if test_end - test_start < n_test // 2:
                break

            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            # Clone the model to avoid stateful issues
            import copy
            model_clone = copy.deepcopy(model)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_clone.train(X_train, y_train)

            train_pred = model_clone.predict(X_train)
            test_pred = model_clone.predict(X_test)

            train_score = _compute_metric(metric, train_pred, y_train)
            test_score = _compute_metric(metric, test_pred, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)
            train_indices.append((train_start, train_end))
            test_indices.append((test_start, test_end))

            oof_preds[test_start:test_end] = test_pred

            current_end += self.step_size

        return WalkForwardResult(
            train_scores=train_scores,
            test_scores=test_scores,
            train_indices=train_indices,
            test_indices=test_indices,
            oof_predictions=oof_preds,
            oof_actuals=oof_actual,
            metric_name=metric,
        )

    def summary(self, result: WalkForwardResult) -> dict:
        """Summarize walk-forward validation results.

        Args:
            result: WalkForwardResult from validate().

        Returns:
            Dictionary with summary statistics.
        """
        test_scores = result.test_scores
        train_scores = result.train_scores
        return {
            "metric": result.metric_name,
            "n_folds": len(test_scores),
            "train_mean": float(np.mean(train_scores)) if train_scores else np.nan,
            "train_std": float(np.std(train_scores)) if train_scores else np.nan,
            "test_mean": float(np.mean(test_scores)) if test_scores else np.nan,
            "test_std": float(np.std(test_scores)) if test_scores else np.nan,
            "test_min": float(np.min(test_scores)) if test_scores else np.nan,
            "test_max": float(np.max(test_scores)) if test_scores else np.nan,
            "test_median": float(np.median(test_scores)) if test_scores else np.nan,
            "fold_scores": test_scores,
        }


def _compute_metric(
    metric: str,
    pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute a single evaluation metric."""
    if metric == "accuracy":
        if pred.ndim > 1:
            pred = pred.ravel()
        return float(np.mean((pred >= 0.5) == (y_true >= 0.5)))
    if metric == "f1":
        try:
            from sklearn.metrics import f1_score
            return float(f1_score(y_true, (pred >= 0.5).astype(int), zero_division=0))
        except ImportError:
            tp = np.sum((pred >= 0.5) & (y_true >= 0.5))
            fp = np.sum((pred >= 0.5) & (y_true < 0.5))
            fn = np.sum((pred < 0.5) & (y_true >= 0.5))
            denom = 2 * tp + fp + fn
            return 2 * tp / denom if denom > 0 else 0.0
    if metric == "auc":
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, pred))
        except ImportError:
            return 0.0
    if metric == "mse":
        return float(np.mean((pred - y_true) ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(pred - y_true)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((pred - y_true) ** 2)))
    return 0.0


# --------------------------------------------------------------------------- #
# ML Pipeline
# --------------------------------------------------------------------------- #

class MLPipeline:
    """Complete ML pipeline combining feature engineering, model, and validation.

    完整的 ML 流水线：特征工程 + 模型训练 + Walk-Forward 验证。
    提供端到端的从 OHLCV 数据到策略信号的能力。

    Attributes:
        feature_eng: FeatureEngineering instance.
        model: BasePredictor model instance.
        validator: WalkForwardValidator instance.
        is_trained: Whether the pipeline model has been fitted.
    """

    def __init__(
        self,
        model: Optional[BasePredictor] = None,
        feature_eng: Optional[FeatureEngineering] = None,
        validator: Optional[WalkForwardValidator] = None,
    ):
        self.feature_eng = feature_eng or FeatureEngineering()
        self.model = model or XGBoostPredictor()
        self.validator = validator or WalkForwardValidator()
        self.is_trained: bool = False

    def train(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        labels: Optional[np.ndarray] = None,  # Alias for y
    ) -> "MLPipeline":
        """Train the pipeline on OHLCV data.

        Args:
            open_price: Opening prices.
            high: High prices.
            low: Low prices.
            close: Closing prices (used as primary if y is None).
            volume: Optional volume.
            y: Optional target. If None, uses next-day return direction (1 if positive, 0 otherwise).
            labels: Alias for y.

        Returns:
            self
        """
        target = y if y is not None else labels

        # Generate features
        X = self.feature_eng.compute_all(open_price, high, low, close, volume)

        # Align y with the trimmed feature matrix
        max_lookback = max(
            self.feature_eng.lookback_long,
            self.feature_eng.lookback_vol,
            max(self.feature_eng.momentum_periods),
        )
        if target is None:
            close_arr = np.asarray(close).flatten()
            ret = FeatureEngineering._pct_change(close_arr, 1)
            target = (ret[max_lookback:] > 0).astype(float)
        else:
            target = np.asarray(target).flatten()[max_lookback:]

        if len(target) != len(X):
            min_len = min(len(target), len(X))
            X = X[:min_len]
            target = target[:min_len]

        # Drop any rows with NaN in target
        valid_mask = ~np.isnan(target)
        X = X[valid_mask]
        target = target[valid_mask]

        self.model.train(X, target)
        self.is_trained = True
        return self

    def predict(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict on OHLCV data.

        Args:
            open_price: Opening prices.
            high: High prices.
            low: Low prices.
            close: Closing prices.
            volume: Optional volume.

        Returns:
            Predictions of shape (n_samples',).
        """
        if not self.is_trained:
            raise MLModelError("Pipeline not trained. Call train() first.")
        X = self.feature_eng.compute_all(open_price, high, low, close, volume)
        return self.model.predict(X)

    def evaluate(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        metric: Literal["accuracy", "f1", "auc", "mse", "mae", "rmse"] = "accuracy",
    ) -> float:
        """Evaluate the pipeline on OHLCV data.

        Args:
            open_price: Opening prices.
            high: High prices.
            low: Low prices.
            close: Closing prices.
            volume: Optional volume.
            y: Optional target.
            metric: Evaluation metric.

        Returns:
            Score value.
        """
        if not self.is_trained:
            raise MLModelError("Pipeline not trained. Call train() first.")
        X = self.feature_eng.compute_all(open_price, high, low, close, volume)

        max_lookback = max(
            self.feature_eng.lookback_long,
            self.feature_eng.lookback_vol,
            max(self.feature_eng.momentum_periods),
        )
        if y is None:
            close_arr = np.asarray(close).flatten()
            ret = FeatureEngineering._pct_change(close_arr, 1)
            y_aligned = (ret[max_lookback:] > 0).astype(float)
        else:
            y_aligned = np.asarray(y).flatten()[max_lookback:]

        min_len = min(len(y_aligned), len(X))
        X, y_aligned = X[:min_len], y_aligned[:min_len]
        valid_mask = ~np.isnan(y_aligned)
        X, y_aligned = X[valid_mask], y_aligned[valid_mask]

        return self.model.evaluate(X, y_aligned, metric=metric)

    def generate_signals(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        threshold_long: float = 0.6,
        threshold_short: float = 0.4,
    ) -> np.ndarray:
        """Generate trading signals from the latest predictions.

        Args:
            open_price: Opening prices.
            high: High prices.
            low: Low prices.
            close: Closing prices.
            volume: Optional volume.
            threshold_long: Threshold for long signal.
            threshold_short: Threshold for short signal.

        Returns:
            Trading signals: 1 (long), 0 (neutral), -1 (short).
        """
        if not self.is_trained:
            raise MLModelError("Pipeline not trained. Call train() first.")
        X = self.feature_eng.compute_all(open_price, high, low, close, volume)
        return self.model.generate_signals(X, threshold_long, threshold_short)

    def walk_forward_validate(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        metric: Literal["accuracy", "f1", "auc", "mse", "mae", "rmse"] = "accuracy",
    ) -> WalkForwardResult:
        """Perform walk-forward validation on OHLCV data.

        Args:
            open_price: Opening prices.
            high: High prices.
            low: Low prices.
            close: Closing prices.
            volume: Optional volume.
            y: Optional target.
            metric: Evaluation metric.

        Returns:
            WalkForwardResult with per-fold scores and OOF predictions.
        """
        X = self.feature_eng.compute_all(open_price, high, low, close, volume)

        max_lookback = max(
            self.feature_eng.lookback_long,
            self.feature_eng.lookback_vol,
            max(self.feature_eng.momentum_periods),
        )
        if y is None:
            close_arr = np.asarray(close).flatten()
            ret = FeatureEngineering._pct_change(close_arr, 1)
            y_aligned = (ret[max_lookback:] > 0).astype(float)
        else:
            y_aligned = np.asarray(y).flatten()[max_lookback:]

        min_len = min(len(y_aligned), len(X))
        X, y_aligned = X[:min_len], y_aligned[:min_len]
        valid_mask = ~np.isnan(y_aligned)
        X, y_aligned = X[valid_mask], y_aligned[valid_mask]

        return self.validator.validate(self.model, X, y_aligned, metric=metric)
