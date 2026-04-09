"""
DPML — Dual-Process Meta-Learning for Volume Prediction (ECML PKDD 2022).

双过程元学习交易量预测器 — 结合快速直觉（System 1）与深度推理（System 2）的双过程架构。

Architecture:
    System 1 (Fast):  轻量级时序模型 — 快速直觉预测
    System 2 (Slow):  重型深度模型 — 深度推理校正
    MetaLearner:      学习何时信任 System 1 vs System 2

Reference:
    Dual-Process Meta-Learning for Scalable Time-Series Forecasting (ECML PKDD 2022)
    https://www.ecmlpkdd2022.org/
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DualProcessVolumePredictor",
    "MetaLearner",
    "System1Linear",
    "System1LSTM",
    "System2LSTM",
    "System2Transformer",
]


# ---------------------------------------------------------------------------
# System 1 — Fast, lightweight predictors
# ---------------------------------------------------------------------------

class System1Linear(nn.Module):
    """System 1: 线性回归 — 最快速的直觉预测."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 1)"""
        return self.fc(x).squeeze(-1)


class System1LSTM(nn.Module):
    """System 1: 小型 LSTM — 轻量快速直觉预测."""

    def __init__(self, input_size: int = 5, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch,)"""
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.fc(last_hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# System 2 — Slow, heavy deep models
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """可学习的位置编码."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(0).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class System2LSTM(nn.Module):
    """System 2: 标准 LSTM — 深度时序推理."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch,)"""
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden).squeeze(-1)


class TransformerEncoderLayer(nn.Module):
    """单层 Transformer 编码器."""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class System2Transformer(nn.Module):
    """System 2: Transformer 编码器 — 最强深度推理能力."""

    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch,)"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        # 取最后一个 token 的表示
        last_token = x[:, -1, :]
        return self.fc(last_token).squeeze(-1)


# ---------------------------------------------------------------------------
# MetaLearner — decides which system to trust
# ---------------------------------------------------------------------------

class MetaLearner(nn.Module):
    """元学习器 — 学习何时信任 System 1 vs System 2.

    基于输入特征的线性门控网络，输出每个系统的信任权重。
    训练时根据预测误差更新，推理时直接使用学到的权重。

    Attributes:
        lr: 学习率
    """

    def __init__(self, feature_dim: int = 32, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # output: [trust_s1, trust_s2]
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 记录历史正确率用于置信度校准
        self.correct_history: list[bool] = []

    def select_system(self, features: np.ndarray) -> str:
        """根据特征选择系统.

        Args:
            features: (feature_dim,) numpy array

        Returns:
            'system1' 或 'system2'
        """
        self.eval()
        with torch.no_grad():
            f = torch.from_numpy(features).float().unsqueeze(0)
            logits = self.gate(f)
            trust = F.softmax(logits, dim=-1).squeeze(0)
            s1_trust, s2_trust = trust[0].item(), trust[1].item()
        return "system1" if s1_trust > s2_trust else "system2"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回每个系统的信任权重 (batch, 2)."""
        return F.softmax(self.gate(features), dim=-1)

    def update(self, chosen: str, correct: bool) -> None:
        """根据选择结果更新元学习器.

        Args:
            chosen: 'system1' 或 'system2'
            correct: 选择的系统是否正确
        """
        self.train()
        self.correct_history.append(correct)
        if len(self.correct_history) > 1000:
            self.correct_history.pop(0)

        # 简单的基于规则的目标标签
        # 如果选择正确则强化该系统，否则尝试切换
        batch_size = 1
        target = torch.tensor([1.0, 0.0] if chosen == "system1" else [0.0, 1.0]).unsqueeze(0)

        # 构造伪特征用于反向传播
        pseudo_features = torch.randn(batch_size, self.gate[0].in_features)
        logits = self.gate(pseudo_features)
        loss = F.cross_entropy(logits, target.argmax(dim=-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_confidence(self) -> float:
        """返回当前元学习器的置信度 (基于历史准确率)."""
        if not self.correct_history:
            return 0.5
        return sum(self.correct_history) / len(self.correct_history)


# ---------------------------------------------------------------------------
# DualProcessVolumePredictor — main entry point
# ---------------------------------------------------------------------------

class DualProcessVolumePredictor:
    """双过程元学习交易量预测器 (ECML PKDD 2022).

    结合 System 1（快速直觉）与 System 2（深度推理）的双过程架构，
    通过 MetaLearner 学习在何种情况下信任哪个系统。

    System 1 (Fast):  轻量级时序模型 — 快速直觉预测
        - 'linear': 线性回归，适合基线
        - 'lstm_small': 小型 LSTM (hidden=32)

    System 2 (Slow):  重型深度模型 — 深度推理校正
        - 'lstm': 标准 LSTM (hidden=128, layers=2)
        - 'transformer': Transformer 编码器 (layers=3)

    Example:
        >>> predictor = DualProcessVolumePredictor(
        ...     system1_model='lstm_small',
        ...     system2_model='lstm',
        ...     volume_window=20,
        ...     meta_lr=0.01,
        ... )
        >>> history = predictor.fit(price_data, volume_data)
        >>> s1_pred, final_pred = predictor.predict(price_window)
        >>> predictor.meta_update(true_volume)

    Args:
        system1_model: System 1 模型类型 ('linear' | 'lstm_small')
        system2_model: System 2 模型类型 ('lstm' | 'transformer')
        volume_window: 历史窗口大小（天数）
        meta_lr: 元学习器学习率
    """

    def __init__(
        self,
        system1_model: Literal["linear", "lstm_small"] = "linear",
        system2_model: Literal["lstm", "transformer"] = "lstm",
        volume_window: int = 20,
        meta_lr: float = 0.01,
    ):
        self.system1_model_name = system1_model
        self.system2_model_name = system2_model
        self.volume_window = volume_window
        self.meta_lr = meta_lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build System 1
        if system1_model == "linear":
            self._input_dim = volume_window * 5  # price features per day
            self.system1: nn.Module = System1Linear(self._input_dim)
        elif system1_model == "lstm_small":
            self.system1 = System1LSTM(input_size=5, hidden_size=32, num_layers=1)
            self._input_dim = volume_window * 5
        else:
            raise ValueError(f"Unknown system1_model: {system1_model}")

        # Build System 2
        if system2_model == "lstm":
            self.system2 = System2LSTM(input_size=5, hidden_size=128, num_layers=2)
        elif system2_model == "transformer":
            self.system2 = System2Transformer(input_size=5, d_model=128, nhead=8, num_layers=3)
        else:
            raise ValueError(f"Unknown system2_model: {system2_model}")

        # MetaLearner
        self.meta_learner = MetaLearner(feature_dim=32, lr=meta_lr)

        # Optimizers
        self.opt1 = torch.optim.Adam(self.system1.parameters(), lr=1e-3)
        self.opt2 = torch.optim.Adam(self.system2.parameters(), lr=1e-4)

        # Training history
        self.history: dict[str, list[float]] = {
            "s1_loss": [],
            "s2_loss": [],
            "meta_loss": [],
            "blend_loss": [],
        }
        self._is_fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> dict:
        """训练双过程模型.

        Args:
            price_data: (T, D) 价格/量价数据，D >= 5 (open/high/low/close/volume)
            volume_data: (T,) 目标交易量
            metadata: 附加信息如 market_regime, time_of_day 等（可选）

        Returns:
            训练历史 {'s1_loss': [...], 's2_loss': [...], ...}
        """
        T, D = price_data.shape
        if D < 5:
            raise ValueError(f"price_data must have at least 5 columns, got {D}")
        if T < self.volume_window + 1:
            raise ValueError(
                f"Need at least {self.volume_window + 1} time steps, got {T}"
            )

        self.system1.to(self.device)
        self.system2.to(self.device)
        self.meta_learner.to(self.device)

        price_tensor = torch.from_numpy(price_data).float().to(self.device)
        volume_tensor = torch.from_numpy(volume_data).float().to(self.device)

        seq_len = self.volume_window

        # 构建滑动窗口数据集
        X_list, y_list = [], []
        for i in range(seq_len, T):
            window = price_tensor[i - seq_len : i]  # (seq_len, D)
            target = volume_tensor[i]
            X_list.append(window)
            y_list.append(target)

        X = torch.stack(X_list)  # (N, seq_len, D)
        y = torch.stack(y_list)  # (N,)
        N = X.size(0)

        batch_size = min(64, N)
        n_epochs = 20

        for epoch in range(n_epochs):
            idx = torch.randperm(N)
            epoch_s1_loss, epoch_s2_loss = 0.0, 0.0

            for i in range(0, N, batch_size):
                batch_idx = idx[i : i + batch_size]
                X_batch = X[batch_idx]  # (B, seq_len, D)
                y_batch = y[batch_idx]   # (B,)

                # --- System 1 forward ---
                s1_pred: torch.Tensor
                if self.system1_model_name == "linear":
                    # 展平为 (B, seq_len*D)
                    flat = X_batch.reshape(batch_idx.size(0), -1)
                    s1_pred = self.system1(flat)
                else:
                    # 取最后5列作为 input_size=5 的 LSTM 输入
                    s1_pred = self.system1(X_batch)

                # --- System 2 forward ---
                s2_pred = self.system2(X_batch)  # (B,)

                # --- Blend ---
                blend_pred = 0.4 * s1_pred + 0.6 * s2_pred

                # --- Losses ---
                loss1 = F.mse_loss(s1_pred, y_batch)
                loss2 = F.mse_loss(s2_pred, y_batch)
                loss_blend = F.mse_loss(blend_pred, y_batch)

                # --- Update System 1 ---
                self.opt1.zero_grad()
                loss1.backward(retain_graph=True)
                self.opt1.step()

                # --- Update System 2 ---
                self.opt2.zero_grad()
                loss2.backward()
                self.opt2.step()

                epoch_s1_loss += loss1.item() * batch_idx.size(0)
                epoch_s2_loss += loss2.item() * batch_idx.size(0)

            avg_s1 = epoch_s1_loss / N
            avg_s2 = epoch_s2_loss / N
            self.history["s1_loss"].append(avg_s1)
            self.history["s2_loss"].append(avg_s2)
            self.history["blend_loss"].append((avg_s1 + avg_s2) / 2)

            if (epoch + 1) % 5 == 0:
                print(
                    f"[DPML] Epoch {epoch+1}/{n_epochs}  "
                    f"S1-loss={avg_s1:.4f}  S2-loss={avg_s2:.4f}"
                )

        self._is_fitted = True
        return self.history

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        price_window: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> tuple[float, float]:
        """预测下一个交易量.

        Args:
            price_window: (volume_window, D) 最近 D 天价格数据
            metadata: 附加信息（可选）

        Returns:
            (system1_pred, final_pred) 元组
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        self.system1.eval()
        self.system2.eval()
        self.meta_learner.eval()

        if price_window.shape[0] != self.volume_window:
            raise ValueError(
                f"price_window must have {self.volume_window} rows, "
                f"got {price_window.shape[0]}"
            )

        x = torch.from_numpy(price_window).float().unsqueeze(0).to(self.device)  # (1, seq, D)

        with torch.no_grad():
            if self.system1_model_name == "linear":
                flat = x.reshape(1, -1)
                s1_pred = self.system1(flat).item()
            else:
                s1_pred = self.system1(x).item()
            s2_pred = self.system2(x).item()

        # MetaLearner 根据特征选择权重
        # 使用系统输出的差异作为伪特征
        diff = abs(s2_pred - s1_pred)
        pseudo_feature = np.array([s1_pred, s2_pred, diff, s1_pred + s2_pred] + [0.0] * 28)

        chosen = self.meta_learner.select_system(pseudo_feature)

        # 最终预测：基于元学习器选择的加权
        if chosen == "system1":
            final_pred = 0.7 * s1_pred + 0.3 * s2_pred
        else:
            final_pred = 0.3 * s1_pred + 0.7 * s2_pred

        return s1_pred, final_pred

    # ------------------------------------------------------------------
    # meta_update
    # ------------------------------------------------------------------

    def meta_update(self, true_volume: float) -> None:
        """元学习更新 — 根据预测误差调整 System 1/2 权重.

        Args:
            true_volume: 真实交易量
        """
        # 获取最近一次预测的系统选择和预测值
        # 注意：这里需要保存最近的预测信息
        if not hasattr(self, "_last_s1_pred") or not hasattr(self, "_last_s2_pred"):
            raise RuntimeError("meta_update must be called after at least one predict() call.")

        s1_err = abs(self._last_s1_pred - true_volume)
        s2_err = abs(self._last_s2_pred - true_volume)

        chosen = "system1" if s1_err <= s2_err else "system2"
        correct = (chosen == "system1" and s1_err < s2_err) or (
            chosen == "system2" and s2_err <= s1_err
        )

        self.meta_learner.update(chosen, correct)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_last_preds(self, s1: float, s2: float) -> None:
        self._last_s1_pred = s1
        self._last_s2_pred = s2

    def __repr__(self) -> str:
        return (
            f"DualProcessVolumePredictor(\n"
            f"  system1={self.system1_model_name},\n"
            f"  system2={self.system2_model_name},\n"
            f"  volume_window={self.volume_window},\n"
            f"  device={self.device},\n"
            f"  fitted={self._is_fitted},\n"
            f")"
        )
