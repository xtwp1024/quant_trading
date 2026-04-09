# -*- coding: utf-8 -*-
"""
HFT Simulator — LSTM + ODE-LSTM High-Frequency Trading Strategy.
高频交易模拟器 — LSTM + ODE-LSTM 策略

LSTM + ODE-LSTM hybrid model for high-frequency trading with real LOB (Limit Order Book) data backtesting.
采用常微分方程(ODE)建模LSTM隐藏状态动态，比标准LSTM更适合高频数据的时间连续性。

Author: HFTSimulator Project
Created: 2026-03-30
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

__all__ = [
    "LSTMODEModel",
    "HFTMarketSimulator",
]


class LSTMODEModel(nn.Module):
    """LSTM + ODE-LSTM Hybrid Model / LSTM + ODE-LSTM 混合模型.

    ODE-LSTM: Uses Ordinary Differential Equations to model hidden state dynamics.
    比标准LSTM更适合高频数据的时间连续性，因为ODE层能够更好地捕捉市场状态的平滑过渡。

    Args:
        input_dim (int): Input feature dimension / 输入特征维度
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
        n_layers (int, optional): Number of LSTM layers. Defaults to 2.
        ode_steps (int, optional): Number of ODE integration steps. Defaults to 10.

    Example:
        >>> model = LSTMODEModel(input_dim=20, hidden_dim=64, n_layers=2, ode_steps=10)
        >>> x = torch.randn(32, 10, 20)  # (batch, seq_len, input_dim)
        >>> dt = 0.001  # time step
        >>> output = model.forward(x, dt)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        ode_steps: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.ode_steps = ode_steps

        # Standard LSTM layers / 标准LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )

        # ODE function (dynamics) network / ODE函数(动力学)网络
        # dh/dt = f(h, x, t)
        self.ode_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

        # Output projection / 输出投影
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Trade direction head / 交易方向头
        self.fc_direction = nn.Linear(hidden_dim, 3)  # long, neutral, short

    def ode_step(
        self, h: torch.Tensor, x: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Single ODE integration step (Euler method).

        Args:
            h: Hidden state (batch, hidden_dim)
            x: Input features (batch, input_dim)
            dt: Time step

        Returns:
            Updated hidden state
        """
        # Concatenate hidden state and input for dynamics computation
        # 拼接隐藏状态和输入以计算动态
        h_x = torch.cat([h, x], dim=-1)
        # dh/dt = f(h, x, t)
        dh = self.ode_net(h_x)
        # Euler integration: h_new = h + dt * dh
        h_new = h + dt * dh
        return torch.tanh(h_new)

    def forward(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Forward propagation / 前向传播.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, input_dim)
            dt (float): Time step between observations

        Returns:
            torch.Tensor: Direction logits of shape (batch, seq_len, 3)
        """
        batch_size, seq_len, _ = x.shape

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Apply ODE refinement to LSTM outputs for temporal continuity
        # 使用ODE对LSTM输出进行时间连续性细化
        outputs = []
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            # LSTM output at time t
            h_lstm = lstm_out[:, t, :]

            # Refine with ODE (ode_steps integration steps)
            # 使用ODE细化
            h_ode = h_lstm.clone()
            step_dt = dt / self.ode_steps
            for _ in range(self.ode_steps):
                h_ode = self.ode_step(h_ode, x[:, t, :], step_dt)

            # Blend LSTM and ODE hidden states
            h = 0.5 * h_lstm + 0.5 * h_ode

            # Direction prediction
            direction = self.fc_direction(h)
            outputs.append(direction)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, 3)

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Predict next price movement / 预测下一个价格变动.

        Args:
            sequence (np.ndarray): Input sequence of shape (seq_len, input_dim)

        Returns:
            np.ndarray: Direction predictions (0=short, 1=neutral, 2=long)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq_len, input_dim)
            dt = 0.001  # default time step
            logits = self.forward(x, dt)
            predictions = torch.argmax(logits[:, -1, :], dim=-1).numpy()
        return predictions


class HFTMarketSimulator:
    """High-Frequency Trading Market Simulator / 高频市场模拟器.

    功能:
        1. 加载LOB (Limit Order Book) 数据
        2. 运行策略回测
        3. 计算HFT指标

    Args:
        model (LSTMODEModel): Trained LSTM-ODE model / 训练好的模型
        transaction_cost (float, optional): Transaction cost as fraction. Defaults to 0.0001 (1bp).

    Example:
        >>> model = LSTMODEModel(input_dim=20)
        >>> simulator = HFTMarketSimulator(model, transaction_cost=0.0001)
        >>> results = simulator.backtest(order_book_data, ground_truth)
    """

    def __init__(
        self,
        model: LSTMODEModel,
        transaction_cost: float = 0.0001,
    ):
        self.model = model
        self.transaction_cost = transaction_cost
        self._pnl_history = []
        self._equity_curve = []

    def backtest(
        self,
        order_book_data: np.ndarray,
        ground_truth: np.ndarray,
    ) -> dict:
        """Backtest HFT strategy / 回测HFT策略.

        Args:
            order_book_data (np.ndarray): LOB data of shape (n_ticks, n_levels, 4)
                Columns: [bid_price, bid_volume, ask_price, ask_volume]
            ground_truth (np.ndarray): Actual price changes of shape (n_ticks,)

        Returns:
            dict: Backtest results containing:
                - pnl: Total PnL
                - sharpe: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - trade_count: Number of trades
                - hit_ratio: Hit ratio (win rate)
        """
        n_ticks = order_book_data.shape[0]
        self._pnl_history = []
        self._equity_curve = []

        # Generate predictions
        predictions = []
        seq_len = 30  # lookback window
        self.model.eval()

        for t in range(seq_len, n_ticks):
            seq = order_book_data[t - seq_len : t]
            pred = self.model.predict(seq)
            predictions.append(pred[0])

        predictions = np.array(predictions)

        # Trading simulation
        position = 0  # -1: short, 0: neutral, 1: long
        prev_position = 0
        pnl = 0.0
        trade_count = 0
        wins = 0
        losses = 0

        equity = 1.0  # Starting equity
        peak_equity = equity
        max_drawdown = 0.0

        ground_truth = ground_truth[seq_len:]  # Align with predictions

        for i in range(len(predictions)):
            direction = predictions[i]
            price_change = ground_truth[i]

            # Update position
            prev_position = position
            position = direction - 1  # 0->-1, 1->0, 2->1

            # Calculate transaction cost on position change
            if position != prev_position:
                trade_count += 1
                pnl -= self.transaction_cost

            # Calculate PnL
            tick_pnl = position * price_change - self.transaction_cost
            pnl += tick_pnl

            # Update equity
            equity += tick_pnl
            self._equity_curve.append(equity)

            # Track drawdown
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

            # Track wins/losses for hit ratio
            if position != 0:
                if (position > 0 and price_change > 0) or (position < 0 and price_change < 0):
                    wins += 1
                else:
                    losses += 1

            self._pnl_history.append(pnl)

        # Calculate metrics
        total_return = equity - 1.0

        # Sharpe ratio (annualized, assuming 252 trading days, 390 minutes/day, ~1 tick/sec)
        if len(self._pnl_history) > 1:
            pnl_array = np.array(self._pnl_history)
            returns = np.diff(pnl_array)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 390 * 60)
        else:
            sharpe = 0.0

        hit_ratio = wins / (wins + losses + 1e-8)

        return {
            "pnl": pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": trade_count,
            "hit_ratio": hit_ratio,
            "total_return": total_return,
            "final_equity": equity,
            "equity_curve": np.array(self._equity_curve),
        }

    def compute_latency_penalty(self, execution_latency: float) -> float:
        """Compute latency penalty / 计算延迟惩罚.

        In HFT, execution latency directly impacts PnL due to adverse selection.
        高频交易中，执行延迟会因逆向选择直接影响PnL。

        Args:
            execution_latency (float): Execution latency in seconds

        Returns:
            float: Latency penalty (negative impact on PnL)
        """
        # Latency penalty model: quadratic relationship
        # 延迟惩罚模型：二次关系
        # Typical HFT penalty: ~1 tick per 100 microseconds
        latency_penalty_per_second = 0.0001  # 1bp per second
        penalty = latency_penalty_per_second * (execution_latency ** 2)
        return -penalty

    def evaluate_with_latency(
        self,
        order_book_data: np.ndarray,
        ground_truth: np.ndarray,
        avg_latency: float = 0.0001,
    ) -> dict:
        """Evaluate strategy with latency simulation / 带延迟模拟的策略评估.

        Args:
            order_book_data: LOB data
            ground_truth: Actual price changes
            avg_latency: Average execution latency in seconds (default 100 microseconds)

        Returns:
            dict: Backtest results with latency-adjusted PnL
        """
        results = self.backtest(order_book_data, ground_truth)

        # Apply latency penalty per trade
        latency_penalty = self.compute_latency_penalty(avg_latency)
        adjusted_pnl = results["pnl"] + latency_penalty * results["trade_count"]
        results["latency_adjusted_pnl"] = adjusted_pnl
        results["latency_penalty_per_trade"] = latency_penalty

        return results
