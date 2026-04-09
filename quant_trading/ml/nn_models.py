# -*- coding: utf-8 -*-
"""Neural network models for time series forecasting.

多模型价格预测: CNN, TDNN, RNN, LSTM 实现.
Pure PyTorch implementation.

References:
    - TradingNeuralNetwork (LSTM/cnn_classifier/Simple_RNN)
      D:/Hive/Data/trading_repos/TradingNeuralNetwork/
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "BaseModel",
    "ARMAModel",
    "CNN1DModel",
    "TDNNModel",
    "RNNModel",
    "LSTMModel",
]


# --------------------------------------------------------------------------- #
# Base Model
# --------------------------------------------------------------------------- #

class BaseModel(ABC):
    """Abstract base class for all time series models.

    模型基类 — 定义 fit / predict 接口.
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the model on the given time series data.

        Args:
            data: 1D or 2D numpy array. If 2D, each column is a separate series.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts for the given number of steps ahead.

        Args:
            horizon: Number of future time steps to predict.

        Returns:
            Array of shape (horizon,) or (horizon, n_features).
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# ARMA — Traditional Statistical Baseline
# --------------------------------------------------------------------------- #

class ARMAModel(BaseModel):
    """ARMA(p, q) time series model.

    传统统计时序模型 — 使用 statsmodels ARMA 实现，作为神经网络基线对比.

    Args:
        p: Autoregressive order.
        q: Moving average order.
        has_trend: Whether to include a constant trend term.

    Example:
        >>> model = ARMAModel(p=2, q=1)
        >>> model.fit(train_data)
        >>> preds = model.predict(horizon=5)
    """

    def __init__(self, p: int = 1, q: int = 1, has_trend: bool = True):
        self.p = p
        self.q = q
        self.has_trend = has_trend
        self._model = None
        self._arma_result = None
        self._last_values = None  # store last values for rolling prediction
        self._n_features = 1

    def fit(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._n_features = data.shape[1]
        self._last_values = data[-self.p:] if self.p > 0 else None

        try:
            import statsmodels.api as sm

            self._models = []
            self._results = []
            for i in range(self._n_features):
                series = data[:, i]
                # Build ARMA model
                arma_model = sm.tsa.ARMA(series, order=(self.p, self.q))
                arma_result = arma_model.fit(trend="c" if self.has_trend else "nc", disp=False)
                self._models.append(arma_model)
                self._results.append(arma_result)
        except ImportError:
            # Fallback: manual ARMA implementation
            self._fit_manual(data)

    def _fit_manual(self, data: np.ndarray) -> None:
        """Manual ARMA(p,q) fit using Yule-Walker for AR coefficients
        and OLS for MA, simple version for when statsmodels unavailable."""
        self._ar_coefs = []
        self._ma_coefs = []
        self._intercepts = []
        for i in range(self._n_features):
            series = data[:, i]
            n = len(series)
            # Simple AR(p) via OLS for fallback
            X, y = [], []
            for t in range(p, n):
                X.append([1] + [series[t - j] for j in range(1, self.p + 1)])
                y.append(series[t])
            X, y = np.array(X), np.array(y)
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                self._intercepts.append(coeffs[0])
                self._ar_coefs.append(coeffs[1:])
            except Exception:  # noqa: BLE001
                self._intercepts.append(0.0)
                self._ar_coefs.append(np.zeros(self.p))
            self._ma_coefs.append(np.zeros(self.q))

    def predict(self, horizon: int) -> np.ndarray:
        if self._results is not None and hasattr(self._results[0], "forecast"):
            # statsmodels path
            preds = np.zeros((horizon, self._n_features))
            for i, res in enumerate(self._results):
                f = res.forecast(horizon)
                preds[:, i] = f
            return preds
        # fallback manual
        preds = np.zeros((horizon, self._n_features))
        for i in range(self._n_features):
            cur = np.zeros(horizon)
            ar = self._ar_coefs[i] if i < len(self._ar_coefs) else np.zeros(self.p)
            for h in range(horizon):
                val = self._intercepts[i] if i < len(self._intercepts) else 0.0
                for j in range(min(h + 1, self.p)):
                    if h - j - 1 < len(cur):
                        val += ar[j] * cur[h - j - 1]
                cur[h] = val
            preds[:, i] = cur
        return preds


# --------------------------------------------------------------------------- #
# 1D CNN — Convolutional Feature Extraction
# --------------------------------------------------------------------------- #

class CNN1DModel(BaseModel):
    """1D CNN for time series feature extraction and forecasting.

    1维卷积时序模型 — 参考 TradingNeuralNetwork/cnn/cnn_classifier.py StockCNN
    的 2D 卷积思想，但使用 1D 卷积适配时序数据.

    Args:
        input_dim: Number of input features (channels).
        filters: Number of convolution filters.
        kernel_size: Size of the 1D convolution kernel.
        hidden_size: Size of the fully-connected output layer.
        num_layers: Number of Conv1D layers.
        dropout: Dropout probability.
        lr: Learning rate.
        epochs: Number of training epochs.

    Example:
        >>> model = CNN1DModel(input_dim=4, filters=64, kernel_size=3)
        >>> model.fit(train_data)
        >>> preds = model.predict(horizon=5)
    """

    def __init__(
        self,
        input_dim: int = 1,
        filters: int = 64,
        kernel_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._scaler = None

    def _build_model(self, seq_len: int) -> nn.Module:
        """Build the 1D CNN model architecture."""

        class _CNN1DNet(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                in_ch = self.input_dim
                for _ in range(self.num_layers):
                    layers.append(nn.Conv1d(in_ch, self.filters, self.kernel_size, padding=1))
                    layers.append(nn.ReLU())
                    layers.append(nn.MaxPool1d(2))
                    layers.append(nn.Dropout(self.dropout))
                    in_ch = self.filters
                self.conv = nn.Sequential(*layers)
                # Calculate output size after conv layers: seq_len // 2^num_layers
                conv_out_len = seq_len // (2 ** self.num_layers)
                self.fc = nn.Linear(self.filters * conv_out_len, self.hidden_size)
                self.out = nn.Linear(self.hidden_size, 1)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)
                x = self.conv(x)
                x = x.flatten(1)
                x = F.relu(self.fc(x))
                return self.out(x)

        return _CNN1DNet()

    def _prepare_data(
        self, data: np.ndarray, seq_len: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw time series into supervised (input, target) pairs."""
        X, y = [], []
        data = np.asarray(data, dtype=np.float32)
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i, 0] if data.ndim > 1 else data[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _to_tensor(self, X: np.ndarray, y: np.ndarray = None):
        x_t = torch.from_numpy(X).float().to(self._device)
        if y is not None:
            y_t = torch.from_numpy(y).float().to(self._device)
            return x_t, y_t
        return x_t

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Scale data
        self._scaler_min = data.min(axis=0)
        self._scaler_max = data.max(axis=0)
        scaled = (data - self._scaler_min) / (self._scaler_max - self._scaler_min + 1e-8)

        seq_len = min(30, len(data) // 4)
        seq_len = max(seq_len, 5)
        X, y = self._prepare_data(scaled, seq_len)

        self._seq_len = seq_len
        self._model = self._build_model(seq_len).to(self._device)

        dataset = TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).float()
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                out = self._model(batch_x).squeeze()
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"CNN1D Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(loader):.4f}")

    def predict(self, horizon: int) -> np.ndarray:
        self._model.eval()
        preds = []
        data = self._data_buffer.copy() if hasattr(self, "_data_buffer") else None

        if data is None or len(data) < self._seq_len:
            # Not enough history, return zeros
            return np.zeros(horizon)

        with torch.no_grad():
            for _ in range(horizon):
                seq = torch.from_numpy(data[-self._seq_len:]).float().to(self._device)
                seq = seq.unsqueeze(0)  # batch dim
                pred = self._model(seq).cpu().item()
                preds.append(pred)
                # Append prediction (scaled) for next step
                new_row = data[-1].copy()
                new_row[0] = pred
                data = np.vstack([data, new_row])

        preds = np.array(preds)
        # Inverse scale
        preds = preds * (self._scaler_max[0] - self._scaler_min[0]) + self._scaler_min[0]
        return preds


# --------------------------------------------------------------------------- #
# TDNN — Time-Delay Neural Network
# --------------------------------------------------------------------------- #

class TDNNModel(BaseModel):
    """Time-Delay Neural Network (TDNN) for time series forecasting.

    TDNN — 时延神经网络，将时序窗口内的滞后值作为独立特征输入全连接网络.
    对应传统的前馈神经网络，非循环结构.

    Args:
        input_dim: Number of input features.
        hidden_sizes: List of hidden layer sizes.
        output_horizon: Number of steps ahead to predict.
        dropout: Dropout probability.
        lr: Learning rate.
        epochs: Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_sizes: list = None,
        output_horizon: int = 1,
        seq_len: int = 30,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.output_horizon = output_horizon
        self.seq_len = seq_len
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._scaler_min = None
        self._scaler_max = None

    def _build_model(self) -> nn.Module:
        class _TDNNNet(nn.Module):
            def __init__(self):
                super().__init__()
                in_features = self.input_dim * self.seq_len
                layers = []
                h_sizes = [in_features] + self.hidden_sizes
                for i in range(len(h_sizes) - 1):
                    layers.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Linear(h_sizes[-1], self.output_horizon))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                x = x.flatten(1)  # (batch, seq_len * input_dim)
                return self.net(x)

        return _TDNNNet()

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._scaler_min = data.min(axis=0)
        self._scaler_max = data.max(axis=0)
        scaled = (data - self._scaler_min) / (self._scaler_max - self._scaler_min + 1e-8)

        # Build supervised dataset
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len:i])
            if self.output_horizon == 1:
                target = scaled[i, 0]
            else:
                future = min(self.output_horizon, len(scaled) - i)
                target = scaled[i:i + future, 0]
                if future < self.output_horizon:
                    target = np.pad(target, (0, self.output_horizon - future))
            y.append(target)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self._model = self._build_model().to(self._device)
        dataset = TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).float()
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self._model(batch_x.to(self._device))
                loss = criterion(out, batch_y.to(self._device))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"TDNN Epoch {epoch+1}/{self.epochs}")

    def predict(self, horizon: int) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            seq = torch.from_numpy(
                self._data_buffer[-self.seq_len:].astype(np.float32)
            ).float().to(self._device).unsqueeze(0)
            out = self._model(seq).cpu().numpy()
        out = out.flatten()
        # Inverse scale
        out = out * (self._scaler_max[0] - self._scaler_min[0]) + self._scaler_min[0]
        return out[:horizon]


# --------------------------------------------------------------------------- #
# RNN — Basic Recurrent Network
# --------------------------------------------------------------------------- #

class RNNModel(BaseModel):
    """Basic RNN for time series forecasting.

    基础循环神经网络 — 参考 TradingNeuralNetwork/RNN/Simple_RNN.py
    使用 PyTorch SimpleRNN 层实现.

    Args:
        input_dim: Number of input features.
        hidden_size: Hidden state size.
        num_layers: Number of RNN layers.
        dropout: Dropout between layers.
        output_horizon: Number of steps ahead to predict.
        seq_len: Input sequence length.
        lr: Learning rate.
        epochs: Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        seq_len: int = 30,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_horizon = output_horizon
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._scaler_min = None
        self._scaler_max = None

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.RNN(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                batch_first=True,
            ),
            nn.Flatten(),
            nn.Linear(self.hidden_size * self.seq_len, self.output_horizon),
        )

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._scaler_min = data.min(axis=0)
        self._scaler_max = data.max(axis=0)
        scaled = (data - self._scaler_min) / (self._scaler_max - self._scaler_min + 1e-8)

        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len:i])
            if self.output_horizon == 1:
                y.append(scaled[i, 0])
            else:
                future = min(self.output_horizon, len(scaled) - i)
                target = scaled[i:i + future, 0]
                if future < self.output_horizon:
                    target = np.pad(target, (0, self.output_horizon - future))
                y.append(target)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self._model = self._build_model().to(self._device)
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self._model(batch_x.to(self._device))
                loss = criterion(out, batch_y.to(self._device))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"RNN Epoch {epoch+1}/{self.epochs}")

    def predict(self, horizon: int) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            seq = torch.from_numpy(
                self._data_buffer[-self.seq_len:].astype(np.float32)
            ).float().to(self._device).unsqueeze(0)
            out = self._model(seq).cpu().numpy()
        out = out.flatten()
        out = out * (self._scaler_max[0] - self._scaler_min[0]) + self._scaler_min[0]
        return out[:horizon]


# --------------------------------------------------------------------------- #
# LSTM — Long Short-Term Memory Network
# --------------------------------------------------------------------------- #

class LSTMModel(BaseModel):
    """LSTM (Long Short-Term Memory) network for time series forecasting.

    LSTM 时序模型 — 参考 TradingNeuralNetwork/lstm/lstm.py 的多步预测结构,
    改为 Pure PyTorch 实现 (不依赖 Keras).

    Args:
        input_dim: Number of input features.
        hidden_size: LSTM hidden state size.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability between layers.
        output_horizon: Number of steps ahead to predict.
        seq_len: Input sequence length (lookback window).
        lr: Learning rate.
        epochs: Number of training epochs.

    Example:
        >>> model = LSTMModel(input_dim=4, hidden_size=64, num_layers=2)
        >>> model.fit(train_data)
        >>> preds = model.predict(horizon=5)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        seq_len: int = 30,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_horizon = output_horizon
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._scaler_min: np.ndarray | None = None
        self._scaler_max: np.ndarray | None = None

    def _build_model(self) -> nn.Module:
        """Build the LSTM architecture.

        Architecture:
            LSTM (return_sequences=False) -> FC hidden -> output
        """

        class _LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=self.input_dim,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc1 = nn.Linear(self.hidden_size, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, self.output_horizon)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                _, (h_n, _) = self.lstm(x)
                # Use last layer hidden state
                last_hidden = h_n[-1]  # (batch, hidden_size)
                x = self.relu(self.fc1(last_hidden))
                return self.fc2(x)

        return _LSTMNet()

    def fit(self, data: np.ndarray) -> None:
        """Train the LSTM model.

        Args:
            data: 2D array of shape (n_timesteps, n_features).
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Min-max scaling
        self._scaler_min = data.min(axis=0)
        self._scaler_max = data.max(axis=0)
        scaled = (data - self._scaler_min) / (self._scaler_max - self._scaler_min + 1e-8)

        # Store for prediction
        self._data_buffer = scaled.copy()

        # Build supervised dataset: lookback window -> next step(s)
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len:i])  # (seq_len, input_dim)
            if self.output_horizon == 1:
                # Single step: predict next value of first feature
                y.append(scaled[i, 0])
            else:
                # Multi-step: predict next `output_horizon` values
                future = min(self.output_horizon, len(scaled) - i)
                target = scaled[i:i + future, 0]
                if future < self.output_horizon:
                    target = np.pad(target, (0, self.output_horizon - future))
                y.append(target)

        X = np.array(X, dtype=np.float32)   # (n_samples, seq_len, input_dim)
        y = np.array(y, dtype=np.float32)  # (n_samples,) or (n_samples, horizon)

        self._model = self._build_model().to(self._device)
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                out = self._model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"LSTM Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(loader):.4f}")

    def predict(self, horizon: int) -> np.ndarray:
        """Generate predictions `horizon` steps ahead using rolling window.

        Args:
            horizon: Number of future steps to predict.

        Returns:
            Array of shape (horizon,).
        """
        self._model.eval()
        preds = []

        # Work on scaled data for rolling prediction
        scaled = self._data_buffer.copy()
        min_v, max_v = self._scaler_min[0], self._scaler_max[0]

        with torch.no_grad():
            for _ in range(horizon):
                seq = torch.from_numpy(
                    scaled[-self.seq_len:].astype(np.float32)
                ).float().to(self._device).unsqueeze(0)  # (1, seq_len, input_dim)
                out = self._model(seq).cpu().item()
                preds.append(out)
                # Append prediction back to buffer for next step
                new_row = scaled[-1].copy()
                new_row[0] = out
                scaled = np.vstack([scaled, new_row])

        preds = np.array(preds)
        # Inverse min-max scale
        preds = preds * (max_v - min_v) + min_v
        return preds
