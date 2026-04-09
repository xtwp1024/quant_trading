"""
Multi-Stock RL Trading System / 多股票强化学习交易系统
================================================================

A pure NumPy + gymnasium implementation of multi-stock trading environment
and reinforcement learning policies with attention mechanisms.

Pure NumPy + gymnasium implementation.
All neural network components (LSTM, attention, GAT, Capsule) are implemented
in pure NumPy without PyTorch/TensorFlow.

纯 NumPy + gymnasium 实现。
所有神经网络组件（LSTM、注意力机制、GAT、Capsule）均使用纯 NumPy 实现，
不依赖 PyTorch/TensorFlow。

Author: Claude
Date: 2026-03-31
"""

from __future__ import annotations

import os
import sys
import json
import math
import cmath
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

# Lazy import for gymnasium with fallback
GYMNASIUM_AVAILABLE = False
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding
    GYMNASIUM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    seeding = None

# Lazy import for heavy dependencies
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None

# Constants / 常量
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

# Exports / 导出
__all__ = [
    "MultiStockTradingEnv",
    "CrossAttentionActorCriticPolicy",
    "GATCapsulePolicy",
    "CustomPolicy",
    "LSTMAttentionNetwork",
    "GATCapsuleNetwork",
    "CapsuleLayer",
    "AttentionBlock",
    "AttentionPooling",
    "StandardScaler",
    "NumPySeedMixin",
    "LLMAnalystConfig",
    "LLMAnalyst",
    "blend_actions",
    "process_indicators",
    "BASE_INDICATORS",
    "GATLayer",
    "FundamentalDataFeature",
]


# =============================================================================
# Utility Classes / 工具类
# =============================================================================

class NumPySeedMixin:
    """Mixin for numpy random seeding / NumPy随机种子混入类"""

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the random number generator / 设置随机种子"""
        if seeding is None:
            self.np_random = np.random.default_rng(seed)
            return [seed] if seed is not None else []
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class StandardScaler:
    """Pure NumPy StandardScaler implementation / 纯 NumPy 标准Scaler实现

    Maintains fit/transform state for scaling features.
    维护fit/transform状态用于特征缩放。
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """Compute mean and scale from training data / 从训练数据计算均值和缩放"""
        data = np.asarray(data, dtype=np.float64)
        self.mean_ = np.mean(data, axis=0)
        self.scale_ = np.std(data, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters / 使用拟合参数转换数据"""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        data = np.asarray(data, dtype=np.float64)
        return (data - self.mean_) / self.scale_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step / 一步完成fit和transform"""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform / 逆转换"""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        data = np.asarray(data, dtype=np.float64)
        return data * self.scale_ + self.mean_


# =============================================================================
# Attention Mechanisms (Pure NumPy) / 注意力机制（纯NumPy）
# =============================================================================

class AttentionBlock:
    """Multi-head attention block implemented in pure NumPy / 纯NumPy实现的多头注意力块

    Args:
        embed_dim: Embedding dimension / 嵌入维度
        num_heads: Number of attention heads / 注意力头数量
        dropout: Dropout rate (not used in pure numpy) / Dropout率（纯numpy中不使用）

    Attributes:
        query_weight, key_weight, value_weight: Weight matrices / 权重矩阵

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads: {embed_dim} % {num_heads} != 0")

        # Xavier/Glorot initialization / Xavier初始化
        scale = math.sqrt(2.0 / (embed_dim + self.head_dim))
        self.query_weight = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.key_weight = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.value_weight = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.output_weight = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention output / 计算注意力输出

        Args:
            query: Query tensor (batch, seq_len, embed_dim) / 查询张量
            key: Key tensor (batch, seq_len, embed_dim) / 键张量
            value: Value tensor (batch, seq_len, embed_dim) / 值张量

        Returns:
            output: Attention output (batch, seq_len, embed_dim) / 注意力输出
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len) / 注意力权重
        """
        batch_size, seq_len, _ = query.shape

        # Compute Q, K, V / 计算Q, K, V
        Q = np.einsum('bse,eh->bsh', query, self.query_weight)
        K = np.einsum('bse,eh->bsh', key, self.key_weight)
        V = np.einsum('bse,eh->bsh', value, self.value_weight)

        # Reshape for multi-head attention / 重塑为多头注意力格式
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention / 缩放点积注意力
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(self.head_dim)
        attention_weights = self._softmax(scores, axis=-1)

        # Apply attention to values / 应用注意力到值
        attended = np.einsum('bhqk,bhvd->bhqd', attention_weights, V)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection / 输出投影
        output = np.einsum('bse,eh->bsh', attended, self.output_weight)

        return output, attention_weights

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax / 数值稳定的softmax"""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class AttentionPooling:
    """Attention-based pooling layer / 基于注意力的池化层

    Args:
        embed_dim: Embedding dimension / 嵌入维度

    Pure NumPy implementation.
    纯NumPy实现。
    """

    def __init__(self, embed_dim: int) -> None:
        self.score_weight = np.random.randn(embed_dim, 1).astype(np.float64) * math.sqrt(2.0 / embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pool sequence using attention weights / 使用注意力权重池化序列

        Args:
            x: Input tensor (batch, seq_len, embed_dim) / 输入张量

        Returns:
            pooled: Pooled tensor (batch, embed_dim) / 池化后的张量
        """
        # Compute attention scores / 计算注意力分数
        scores = np.einsum('bse,ev->bsv', x, self.score_weight)
        weights = AttentionBlock._softmax(scores, axis=1)

        # Weighted sum / 加权求和
        pooled = np.einsum('bse,bsv->be', x, weights)
        return pooled


# =============================================================================
# LSTM Implementation (Pure NumPy) / LSTM实现（纯NumPy）
# =============================================================================

class LSTMAttentionNetwork:
    """LSTM with attention mechanism / 带注意力机制的LSTM

    Args:
        input_dim: Input feature dimension / 输入特征维度
        hidden_dim: Hidden state dimension / 隐藏状态维度
        num_assets: Number of assets / 资产数量
        window_size: Time window size / 时间窗口大小
        output_dim: Output dimension / 输出维度
        dropout: Dropout rate (not used in pure numpy) / Dropout率

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_assets: int,
        window_size: int,
        output_dim: int,
        dropout: float = 0.1
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.output_dim = output_dim

        # Initialize LSTM weights / 初始化LSTM权重
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        # Gates: input, forget, cell, output / 门：输入、遗忘、细胞、输出
        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(np.float64) * scale
        self.b_i = np.zeros(hidden_dim, dtype=np.float64)

        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(np.float64) * scale
        self.b_f = np.zeros(hidden_dim, dtype=np.float64)

        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(np.float64) * scale
        self.b_c = np.zeros(hidden_dim, dtype=np.float64)

        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(np.float64) * scale
        self.b_o = np.zeros(hidden_dim, dtype=np.float64)

        # Attention layer / 注意力层
        self.attention = AttentionBlock(hidden_dim, num_heads=4)

        # Output projection / 输出投影
        self.output_weight = np.random.randn(output_dim, hidden_dim).astype(np.float64) * math.sqrt(2.0 / hidden_dim)
        self.output_bias = np.zeros(output_dim, dtype=np.float64)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass / 前向传播

        Args:
            observations: Input tensor (batch, num_assets, window_size, input_dim)
                        / 输入张量 (batch, 资产数, 窗口大小, 输入维度)

        Returns:
            policy_output: Policy network output / 策略网络输出
            value_output: Value network output / 价值网络输出
        """
        batch_size, num_assets, window_size, input_dim = observations.shape
        assert num_assets == self.num_assets
        assert window_size == self.window_size

        # Process each asset through LSTM / 通过LSTM处理每个资产
        asset_embeddings = []
        for i in range(num_assets):
            obs_i = observations[:, i, :, :]  # (batch, window, input_dim)
            hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
            cell = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

            # LSTM forward / LSTM前向
            for t in range(window_size):
                x_t = obs_i[:, t, :]
                h_prev = hidden
                combined = np.concatenate([h_prev, x_t], axis=1)

                # Gates / 门
                i_t = self._sigmoid(np.einsum('hd,bd->bh', self.W_i, combined) + self.b_i)
                f_t = self._sigmoid(np.einsum('hd,bd->bh', self.W_f, combined) + self.b_f)
                c_tilde = np.tanh(np.einsum('hd,bd->bh', self.W_c, combined) + self.b_c)
                o_t = self._sigmoid(np.einsum('hd,bd->bh', self.W_o, combined) + self.b_o)

                cell = f_t * cell + i_t * c_tilde
                hidden = o_t * np.tanh(cell)

            asset_embeddings.append(hidden)

        # Stack asset embeddings / 堆叠资产嵌入
        asset_embeddings = np.stack(asset_embeddings, axis=1)  # (batch, num_assets, hidden_dim)

        # Apply attention across assets / 应用资产间注意力
        attended, _ = self.attention.forward(asset_embeddings, asset_embeddings, asset_embeddings)

        # Global pooling / 全局池化
        pooled = np.mean(attended, axis=1)  # (batch, hidden_dim)

        # Output projection / 输出投影
        policy_output = np.tanh(np.einsum('od,bh->bo', self.output_weight, pooled) + self.output_bias)
        value_output = np.tanh(np.einsum('od,bh->bo', self.output_weight, pooled) + self.output_bias)

        return policy_output, value_output

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid / 数值稳定的sigmoid"""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# =============================================================================
# Capsule Network (Pure NumPy) / Capsule网络（纯NumPy）
# =============================================================================

class CapsuleLayer:
    """Capsule layer with dynamic routing / 动态路由的Capsule层

    Args:
        in_capsules: Number of input capsules / 输入胶囊数量
        in_length: Dimension of input capsules / 输入胶囊维度
        out_capsules: Number of output capsules / 输出胶囊数量
        out_length: Dimension of output capsules / 输出胶囊维度
        num_iterations: Number of routing iterations / 路由迭代次数
        routing_type: Type of routing ('dynamic') / 路由类型

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        in_capsules: int,
        in_length: int,
        out_capsules: int,
        out_length: int,
        num_iterations: int = 3,
        routing_type: str = "dynamic"
    ) -> None:
        self.in_capsules = in_capsules
        self.in_length = in_length
        self.out_capsules = out_capsules
        self.out_length = out_length
        self.num_iterations = num_iterations
        self.routing_type = routing_type

        # Weight matrix for capsule transformation / 胶囊变换的权重矩阵
        scale = math.sqrt(2.0 / (in_capsules * in_length + out_capsules * out_length))
        self.W = np.random.randn(out_capsules, in_capsules, out_length, in_length).astype(np.float64) * scale

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with dynamic routing / 动态路由前向传播

        Args:
            inputs: Input capsules (batch, in_capsules, in_length)
                   / 输入胶囊 (batch, 输入胶囊数, 输入胶囊维度)

        Returns:
            output: Output capsules (batch, out_capsules, out_length)
                   / 输出胶囊 (batch, 输出胶囊数, 输出胶囊维度)
            routing_weights: Final routing weights / 最终路由权重
        """
        batch_size = inputs.shape[0]

        # Expand dimensions for batched matmul / 扩展维度以进行批量矩阵乘法
        inputs_expanded = inputs[:, :, np.newaxis, :]  # (batch, in_caps, 1, in_len)
        W_expanded = self.W[np.newaxis, :, :, :, :]  # (1, out_caps, in_caps, out_len, in_len)

        # Transform input capsules / 变换输入胶囊
        # u_hat: (batch, out_caps, in_caps, out_len)
        u_hat = np.einsum('bciv,vcivl->bcvl', inputs_expanded, W_expanded)

        # Initialize routing logits / 初始化路由logits
        routing_logits = np.zeros((batch_size, self.out_capsules, self.in_capsules), dtype=np.float64)

        # Dynamic routing iterations / 动态路由迭代
        for _ in range(self.num_iterations):
            # Compute routing weights via softmax / 通过softmax计算路由权重
            routing_weights = self._softmax(routing_logits, axis=1)
            # Expand for einsum / 扩展用于einsum
            routing_weights_exp = routing_weights[:, :, :, np.newaxis]  # (batch, out_caps, in_caps, 1)

            # Weighted sum of predicted outputs / 预测输出的加权和
            s = np.sum(routing_weights_exp * u_hat[:, :, np.newaxis, :], axis=2)

            # Squash to unit sphere / 压缩到单位球面
            v = self._squash(s)

            # Update routing logits / 更新路由logits
            if _ < self.num_iterations - 1:
                routing_logits = routing_logits + np.einsum('bcvl,bvl->bcv', u_hat, v)

        output = v
        return output, routing_weights

    @staticmethod
    def _squash(x: np.ndarray) -> np.ndarray:
        """Squash activation function / Squash激活函数"""
        norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        norm = np.sqrt(norm_sq + 1e-8)
        return (norm_sq / (norm + 1e-8)) * (x / norm)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax / 数值稳定的softmax"""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# GAT (Graph Attention Network) Layer / GAT（图注意力网络）层
# =============================================================================

class GATLayer:
    """Graph Attention Network layer / 图注意力网络层

    Args:
        in_features: Input feature dimension / 输入特征维度
        out_features: Output feature dimension / 输出特征维度
        num_heads: Number of attention heads / 注意力头数量
        dropout: Dropout rate (not used in pure numpy) / Dropout率

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        if self.head_dim * num_heads != out_features:
            raise ValueError(f"out_features must be divisible by num_heads")

        # LeakyReLU slope for negative inputs / LeakyReLU负输入斜率
        self.leaky_slope = 0.2

        # Initialize attention weights / 初始化注意力权重
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(num_heads, in_features, self.head_dim).astype(np.float64) * scale
        self.a = np.random.randn(num_heads, 2 * self.head_dim).astype(np.float64) * scale

    def forward(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Forward pass with graph attention / 图注意力前向传播

        Args:
            x: Node features (num_nodes, in_features) / 节点特征
            edge_index: Edge indices (2, num_edges) / 边索引

        Returns:
            output: Updated node features (num_nodes, out_features) / 更新后的节点特征
        """
        num_nodes = x.shape[0]

        # Linear transformation / 线性变换
        # h: (num_heads, num_nodes, head_dim)
        h = np.einsum('ni,hiv->nhv', x, self.W)

        # Compute attention coefficients / 计算注意力系数
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # Concatenate source and target features / 拼接源和目标特征
        h_source = h[:, source_nodes, :]  # (num_heads, num_edges, head_dim)
        h_target = h[:, target_nodes, :]  # (num_heads, num_edges, head_dim)

        h_concat = np.concatenate([h_source, h_target], axis=-1)  # (num_heads, num_edges, 2*head_dim)

        # Compute attention scores / 计算注意力分数
        e = np.einsum('hev,ha->hav', h_concat, self.a)  # (num_heads, num_edges, 1)
        e = self._leaky_relu(e.squeeze(-1))  # (num_heads, num_edges)

        # Create attention mask / 创建注意力掩码
        num_edges = edge_index.shape[1]
        attention_mask = np.zeros((self.num_heads, num_nodes, num_nodes), dtype=np.float64)
        attention_mask[:, source_nodes, target_nodes] = e

        # Mask non-edges / 掩码非边
        attention_mask = np.where(attention_mask > -1e10, attention_mask, -1e10)

        # Softmax over neighbors / 对邻居进行softmax
        attention_weights = self._softmax(attention_mask, axis=-1)

        # Apply attention to node features / 应用注意力到节点特征
        output = np.zeros((num_nodes, self.num_heads, self.head_dim), dtype=np.float64)
        np.scatter_add(output, target_nodes, attention_weights[:, :, target_nodes] * h_source.transpose(1, 2, 0))

        # Reshape and combine heads / 重塑和组合头
        output = output.reshape(num_nodes, self.out_features)
        return output

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        """Leaky ReLU activation / Leaky ReLU激活"""
        return np.where(x > 0, x, self.leaky_slope * x)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax / 数值稳定的softmax"""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# GAT-Capsule Network (Pure NumPy) / GAT-Capsule网络（纯NumPy）
# =============================================================================

class GATCapsuleNetwork:
    """Graph Attention Network with Capsule integration / 带Capsule集成的图注意力网络

    Args:
        input_dim: Input feature dimension / 输入特征维度
        hidden_dim: Hidden dimension / 隐藏维度
        num_assets: Number of assets (graph nodes) / 资产数量（图节点）
        window_size: Time window size / 时间窗口大小
        output_dim: Output dimension / 输出维度

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_assets: int,
        window_size: int,
        output_dim: int
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.output_dim = output_dim

        # Build complete graph edges (all asset pairs) / 构建完整图边（所有资产对）
        edges = []
        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    edges.append([i, j])
        self.edge_index = np.array(edges, dtype=np.int64).T  # (2, num_edges)

        # GAT layers / GAT层
        self.gat0 = GATLayer(input_dim, hidden_dim, num_heads=4)
        self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads=4)

        # Temporal attention / 时间注意力
        self.temporal_attention = AttentionBlock(hidden_dim, num_heads=4)

        # Capsule layer / Capsule层
        self.capsule = CapsuleLayer(
            in_capsules=num_assets,
            in_length=2 * hidden_dim,
            out_capsules=input_dim,
            out_length=hidden_dim,
            num_iterations=3
        )

        # Output projection / 输出投影
        self.output_weight = np.random.randn(output_dim, hidden_dim).astype(np.float64) * math.sqrt(2.0 / hidden_dim)
        self.output_bias = np.zeros(output_dim, dtype=np.float64)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass / 前向传播

        Args:
            observations: Input tensor (batch, num_assets, window_size, input_dim)
                        / 输入张量 (batch, 资产数, 窗口大小, 输入维度)

        Returns:
            policy_output: Policy network output / 策略网络输出
            value_output: Value network output / 价值网络输出
        """
        batch_size, num_assets, window_size, input_dim = observations.shape

        # Temporal attention across time steps / 时间步上的注意力
        obs_reshaped = observations.reshape(batch_size * num_assets, window_size, input_dim)
        temporal_attended, _ = self.temporal_attention.forward(obs_reshaped, obs_reshaped, obs_reshaped)

        # Pool across time / 时间池化
        temporal_pooled = np.mean(temporal_attended, axis=1)  # (batch * num_assets, input_dim)
        temporal_pooled = temporal_pooled.reshape(batch_size, num_assets, input_dim)

        # Apply GAT for asset interactions / 应用GAT处理资产交互
        asset_features = temporal_pooled  # (batch, num_assets, input_dim)

        # GAT layer 0 / GAT层0
        gat0_out = np.zeros((batch_size, num_assets, self.hidden_dim), dtype=np.float64)
        for b in range(batch_size):
            gat0_out[b] = self.gat0.forward(asset_features[b], self.edge_index)

        # GAT layer 1 / GAT层1
        gat1_out = np.zeros((batch_size, num_assets, self.hidden_dim), dtype=np.float64)
        for b in range(batch_size):
            gat1_out[b] = self.gat1.forward(gat0_out[b], self.edge_index)

        # Residual connection / 残差连接
        asset_features = np.tanh(gat0_out + gat1_out)

        # Capsule integration / Capsule集成
        # Reshape for capsule: (batch, num_assets, 1, 2*hidden_dim)
        caps_input = np.concatenate([asset_features, asset_features], axis=-1).reshape(
            batch_size, num_assets, 1, 2 * self.hidden_dim
        )

        # Expand to expected capsule format / 扩展到预期的capsule格式
        caps_input = np.tile(caps_input, (1, 1, self.input_dim, 1))  # (batch, num_assets, input_dim, 2*hidden_dim)
        caps_input = caps_input.reshape(batch_size, num_assets * self.input_dim, 2 * self.hidden_dim)

        caps_output, _ = self.capsule.forward(caps_input)
        caps_output = caps_output.reshape(batch_size, self.input_dim, self.hidden_dim)

        # Pool across capsules / 在capsule上池化
        pooled = np.mean(caps_output, axis=1)  # (batch, hidden_dim)

        # Output projection / 输出投影
        policy_output = np.tanh(np.einsum('od,bh->bo', self.output_weight, pooled) + self.output_bias)
        value_output = np.tanh(np.einsum('od,bh->bo', self.output_weight, pooled) + self.output_bias)

        return policy_output, value_output


# =============================================================================
# Cross-Attention Policy / 交叉注意力策略
# =============================================================================

class CrossAttentionActorCriticPolicy:
    """Cross-Attention Actor-Critic Policy for multi-asset trading
    / 多资产交易的交叉注意力Actor-Critic策略

    Args:
        observation_space_shape: Shape of observation space (num_assets, window_size, feature_dim)
                                / 观察空间形状 (资产数, 窗口大小, 特征维度)
        action_space_shape: Shape of action space (num_assets,) / 动作空间形状 (资产数,)
        embed_dim: Embedding dimension for attention / 注意力的嵌入维度
        num_heads: Number of attention heads / 注意力头数量
        hidden_dim: Hidden dimension for MLPs / MLP的隐藏维度
        dropout: Dropout rate (not used in pure numpy) / Dropout率

    This policy uses cross-attention to model relationships between different
    assets across time steps.

    此策略使用交叉注意力来建模不同资产之间跨时间步的关系。

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        observation_space_shape: Tuple[int, int, int],
        action_space_shape: Tuple[int, ...],
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ) -> None:
        self.num_assets = observation_space_shape[0]
        self.window_size = observation_space_shape[1]
        self.feature_dim = observation_space_shape[2]
        self.action_dim = action_space_shape[0]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Feature projection / 特征投影
        self.feature_projection_weight = np.random.randn(embed_dim, feature_dim).astype(np.float64) * math.sqrt(2.0 / feature_dim)
        self.feature_projection_bias = np.zeros(embed_dim, dtype=np.float64)

        # Temporal attention / 时间注意力
        self.temporal_attention = AttentionBlock(embed_dim, num_heads, dropout)
        self.temporal_pool = AttentionPooling(embed_dim)

        # Asset cross-attention / 资产交叉注意力
        self.asset_attention = AttentionBlock(embed_dim, num_heads, dropout)
        self.asset_mlp_weight1 = np.random.randn(embed_dim, embed_dim).astype(np.float64) * math.sqrt(2.0 / embed_dim)
        self.asset_mlp_bias1 = np.zeros(embed_dim, dtype=np.float64)
        self.asset_mlp_weight2 = np.random.randn(embed_dim, embed_dim).astype(np.float64) * math.sqrt(2.0 / embed_dim)
        self.asset_mlp_bias2 = np.zeros(embed_dim, dtype=np.float64)

        # Final layer norm / 最终层归一化
        self.final_layer_norm_scale = np.ones(embed_dim, dtype=np.float64)
        self.final_layer_norm_bias = np.zeros(embed_dim, dtype=np.float64)

        # Policy and value networks / 策略和价值网络
        flattened_dim = self.num_assets * embed_dim
        self.policy_weight = np.random.randn(action_dim, flattened_dim + embed_dim).astype(np.float64) * math.sqrt(2.0 / flattened_dim)
        self.policy_bias = np.zeros(action_dim, dtype=np.float64)
        self.value_weight = np.random.randn(action_dim, flattened_dim + embed_dim).astype(np.float64) * math.sqrt(2.0 / flattened_dim)
        self.value_bias = np.zeros(action_dim, dtype=np.float64)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass / 前向传播

        Args:
            observations: Input tensor (batch, num_assets, window_size, feature_dim)
                        / 输入张量 (batch, 资产数, 窗口大小, 特征维度)

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        batch_size = observations.shape[0]

        # Feature projection / 特征投影
        # Reshape: (batch * num_assets, window_size, feature_dim)
        obs_reshaped = observations.reshape(batch_size * self.num_assets, self.window_size, self.feature_dim)
        projected = np.einsum('bwi,ei->bwe', obs_reshaped, self.feature_projection_weight) + self.feature_projection_bias

        # Temporal attention / 时间注意力
        temporal_attended, _ = self.temporal_attention.forward(projected, projected, projected)
        temporal_pooled = self.temporal_pool.forward(temporal_attended)  # (batch * num_assets, embed_dim)
        temporal_pooled = temporal_pooled.reshape(batch_size, self.num_assets, self.embed_dim)

        # Asset cross-attention / 资产交叉注意力
        asset_attended, _ = self.asset_attention.forward(
            temporal_pooled, temporal_pooled, temporal_pooled
        )

        # MLP for asset features / 资产特征的MLP
        mlp_out = np.maximum(0, np.einsum('bae,ee->bae', asset_attended, self.asset_mlp_weight1) + self.asset_mlp_bias1)
        mlp_out = np.einsum('bae,ee->bae', mlp_out, self.asset_mlp_weight2) + self.asset_mlp_bias2

        # Final norm / 最终归一化
        encoded = np.tanh(asset_attended + mlp_out)
        encoded = encoded * self.final_layer_norm_scale + self.final_layer_norm_bias

        # Flatten and pool / 展平和池化
        flattened = encoded.reshape(batch_size, -1)  # (batch, num_assets * embed_dim)
        pooled = np.mean(encoded, axis=1)  # (batch, embed_dim)

        # Concatenate for output / 拼接用于输出
        combined = np.concatenate([flattened, pooled], axis=-1)

        # Policy and value outputs / 策略和价值输出
        policy_logits = np.tanh(np.einsum('ac,bc->ba', self.policy_weight, combined) + self.policy_bias)
        value = np.tanh(np.einsum('ac,bc->ba', self.value_weight, combined) + self.value_bias)

        return policy_logits, value

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict actions from observations / 从观察预测动作

        Args:
            observations: Input tensor / 输入张量

        Returns:
            actions: Predicted actions / 预测的动作
        """
        policy_logits, _ = self.forward(observations)
        return policy_logits

    def evaluate(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate policy and value / 评估策略和价值

        Args:
            observations: Input tensor / 输入张量

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        return self.forward(observations)


# =============================================================================
# GAT-Capsule Policy / GAT-Capsule策略
# =============================================================================

class GATCapsulePolicy:
    """GAT-Capsule Actor-Critic Policy / GAT-Capsule Actor-Critic策略

    Args:
        observation_space_shape: Shape of observation space (num_assets, window_size, feature_dim)
                                / 观察空间形状 (资产数, 窗口大小, 特征维度)
        action_space_shape: Shape of action space (num_assets,) / 动作空间形状 (资产数,)
        hidden_dim: Hidden dimension for networks / 网络的隐藏维度

    This policy combines Graph Attention Networks with Capsule networks
    to model complex asset relationships.

    此策略结合图注意力网络和Capsule网络来建模复杂的资产关系。

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        observation_space_shape: Tuple[int, int, int],
        action_space_shape: Tuple[int, ...],
        hidden_dim: int = 64
    ) -> None:
        self.num_assets = observation_space_shape[0]
        self.window_size = observation_space_shape[1]
        self.feature_dim = observation_space_shape[2]
        self.action_dim = action_space_shape[0]
        self.hidden_dim = hidden_dim

        # GAT-Capsule network / GAT-Capsule网络
        self.network = GATCapsuleNetwork(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_assets=num_assets,
            window_size=window_size,
            output_dim=action_dim
        )

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass / 前向传播

        Args:
            observations: Input tensor (batch, num_assets, window_size, feature_dim)
                        / 输入张量 (batch, 资产数, 窗口大小, 特征维度)

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        return self.network.forward(observations)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict actions from observations / 从观察预测动作

        Args:
            observations: Input tensor / 输入张量

        Returns:
            actions: Predicted actions / 预测的动作
        """
        policy_logits, _ = self.forward(observations)
        return policy_logits

    def evaluate(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate policy and value / 评估策略和价值

        Args:
            observations: Input tensor / 输入张量

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        return self.forward(observations)


# =============================================================================
# Custom Simple Policy / 自定义简单策略
# =============================================================================

class CustomPolicy:
    """Custom simple MLP policy / 自定义简单MLP策略

    Args:
        observation_space_shape: Shape of observation space (num_assets, window_size, feature_dim)
                                / 观察空间形状 (资产数, 窗口大小, 特征维度)
        action_space_shape: Shape of action space (num_assets,) / 动作空间形状 (资产数,)
        hidden_dim: Hidden dimension for networks / 网络的隐藏维度

    A simpler policy that flattens observations and passes through MLPs.

    简单的策略，展平观察后通过MLP。

    Pure NumPy implementation without PyTorch/TensorFlow.
    纯NumPy实现，不依赖PyTorch/TensorFlow。
    """

    def __init__(
        self,
        observation_space_shape: Tuple[int, int, int],
        action_space_shape: Tuple[int, ...],
        hidden_dim: int = 64
    ) -> None:
        self.num_assets = observation_space_shape[0]
        self.window_size = observation_space_shape[1]
        self.feature_dim = observation_space_shape[2]
        self.action_dim = action_space_shape[0]
        self.hidden_dim = hidden_dim

        # Flattened input dimension / 展平后的输入维度
        self.flattened_dim = self.num_assets * self.window_size * self.feature_dim

        # Policy network / 策略网络
        self.policy_weight1 = np.random.randn(hidden_dim * 4, self.flattened_dim).astype(np.float64) * math.sqrt(2.0 / self.flattened_dim)
        self.policy_bias1 = np.zeros(hidden_dim * 4, dtype=np.float64)
        self.policy_weight2 = np.random.randn(hidden_dim * 2, hidden_dim * 4).astype(np.float64) * math.sqrt(2.0 / (hidden_dim * 4))
        self.policy_bias2 = np.zeros(hidden_dim * 2, dtype=np.float64)
        self.policy_weight3 = np.random.randn(action_dim, hidden_dim * 2).astype(np.float64) * math.sqrt(2.0 / (hidden_dim * 2))
        self.policy_bias3 = np.zeros(action_dim, dtype=np.float64)

        # Value network / 价值网络
        self.value_weight1 = np.random.randn(hidden_dim * 4, self.flattened_dim).astype(np.float64) * math.sqrt(2.0 / self.flattened_dim)
        self.value_bias1 = np.zeros(hidden_dim * 4, dtype=np.float64)
        self.value_weight2 = np.random.randn(hidden_dim * 2, hidden_dim * 4).astype(np.float64) * math.sqrt(2.0 / (hidden_dim * 4))
        self.value_bias2 = np.zeros(hidden_dim * 2, dtype=np.float64)
        self.value_weight3 = np.random.randn(action_dim, hidden_dim * 2).astype(np.float64) * math.sqrt(2.0 / (hidden_dim * 2))
        self.value_bias3 = np.zeros(action_dim, dtype=np.float64)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass / 前向传播

        Args:
            observations: Input tensor (batch, num_assets, window_size, feature_dim)
                        / 输入张量 (batch, 资产数, 窗口大小, 特征维度)

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        batch_size = observations.shape[0]

        # Flatten / 展平
        x = observations.reshape(batch_size, -1)

        # Policy MLP / 策略MLP
        p = np.maximum(0, np.einsum('hi,bi->bh', self.policy_weight1, x) + self.policy_bias1)
        p = np.maximum(0, np.einsum('hi,bi->bh', self.policy_weight2, p) + self.policy_bias2)
        policy_logits = np.tanh(np.einsum('ai,bi->ba', self.policy_weight3, p) + self.policy_bias3)

        # Value MLP / 价值MLP
        v = np.maximum(0, np.einsum('hi,bi->bh', self.value_weight1, x) + self.value_bias1)
        v = np.maximum(0, np.einsum('hi,bi->bh', self.value_weight2, v) + self.value_bias2)
        value = np.tanh(np.einsum('ai,bi->ba', self.value_weight3, v) + self.value_bias3)

        return policy_logits, value

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict actions from observations / 从观察预测动作

        Args:
            observations: Input tensor / 输入张量

        Returns:
            actions: Predicted actions / 预测的动作
        """
        policy_logits, _ = self.forward(observations)
        return policy_logits

    def evaluate(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate policy and value / 评估策略和价值

        Args:
            observations: Input tensor / 输入张量

        Returns:
            policy_logits: Policy network output / 策略网络输出
            value: Value network output / 价值网络输出
        """
        return self.forward(observations)


# =============================================================================
# Multi-Stock Trading Environment / 多股票交易环境
# =============================================================================

class MultiStockTradingEnv:
    """Multi-Stock Trading Environment for reinforcement learning
    / 多股票交易强化学习环境

    A gymnasium-compatible trading environment that supports multiple stocks
    with portfolio-based actions and technical indicator features.

    支持多股票的gymnasium兼容交易环境，基于投资组合的动作和技术指标特征。

    Args:
        dfs: List of numpy arrays, each (time_steps, num_features) per stock
            / numpy数组列表，每个股票 (时间步数, 特征数)
        price_arrays: Dict mapping stock names to price arrays
                     / 映射股票名称到价格数组的字典
        initial_amount: Initial account balance / 初始账户余额
        trade_cost: Transaction cost rate / 交易成本率
        num_features: Number of features per stock / 每个股票的特征数
        num_stocks: Number of stocks / 股票数量
        window_size: Time window for observation / 观察的时间窗口
        frame_bound: Tuple of (start, end) time indices / (开始, 结束)时间索引元组
        scalers: Optional list of StandardScaler objects / 可选的标准Scaler对象列表
        tech_indicator_list: List of technical indicator names / 技术指标名称列表
        reward_scaling: Scaling factor for rewards / 奖励缩放因子
        suppression_rate: Rate for action suppression (0.0-1.0) / 动作抑制率
        representative: Name of representative stock index / 代表性股票指数名称

    Observation Space:
        Box space with shape (num_stocks, window_size, num_features)
        / 形状为 (num_stocks, window_size, num_features) 的Box空间

    Action Space:
        Box space with shape (num_stocks,) normalized to [-1, 1]
        / 归一化到 [-1, 1] 的形状为 (num_stocks,) 的Box空间

    Attributes:
        dfs: List of DataFrame arrays per stock / 每个股票的DataFrame数组列表
        prices: Current prices array / 当前价格数组
        signal_features: Processed technical features / 处理后的技术特征
        portfolio: Current portfolio positions / 当前投资组合持仓
        reserve: Cash reserve / 现金储备
        margin: Total margin / 总保证金

    Note:
        Requires gymnasium package. Install with: pip install gymnasium
        Requires gymnasium包。使用以下命令安装: pip install gymnasium
    """

    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "MultiStockTradingEnv requires gymnasium. "
            "Please install with: pip install gymnasium"
        )

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dfs: List[np.ndarray],
        price_arrays: Dict[str, np.ndarray],
        initial_amount: float = INITIAL_ACCOUNT_BALANCE,
        trade_cost: float = 0.0,
        num_features: int = 5,
        num_stocks: int = 5,
        window_size: int = 12,
        frame_bound: Optional[Tuple[int, int]] = None,
        scalers: Optional[List[StandardScaler]] = None,
        tech_indicator_list: Optional[List[str]] = None,
        reward_scaling: float = 1e-5,
        suppression_rate: float = 0.66,
        representative: Optional[str] = None
    ) -> None:
        if tech_indicator_list is None:
            tech_indicator_list = []

        if len(tech_indicator_list) != 0:
            num_features = len(tech_indicator_list)

        self.dfs = dfs
        self.price_arrays = price_arrays
        self.initial_amount = initial_amount
        self.margin = initial_amount
        self.portfolio = np.zeros(num_stocks, dtype=np.float64)
        self.PortfolioValue = 0.0
        self.reserve = initial_amount
        self.trade_cost = trade_cost
        self.state_space = num_features
        self.assets = num_stocks
        self.reward_scaling = reward_scaling
        self.tech_indicators = tech_indicator_list
        self.window_size = window_size
        self.frame_bound = frame_bound or (window_size, len(list(price_arrays.values())[0]) if price_arrays else 1000)

        # Spaces / 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_stocks, window_size, num_features),
            dtype=np.float32
        )

        # Episode state / 回合状态
        self._start_tick = self.window_size
        self._end_tick = len(list(price_arrays.values())[0]) - 1 if price_arrays else 0
        self._done = False
        self._current_tick = None
        self._last_trade_tick = None
        self._position = np.zeros(self.assets, dtype=np.float64)
        self._position_history: List[np.ndarray] = []
        self._total_reward = 0.0
        self._total_profit = 1.0
        self._first_rendering = True
        self.history: Dict[str, List[float]] = {}
        self.rewards: List[float] = []
        self.pvs: List[float] = []

        # Scalers / 缩放器
        if scalers is None:
            self.scalers: List[StandardScaler] = [StandardScaler() for _ in range(self.assets)]
        else:
            self.scalers = scalers

        self.representative = representative
        self.suppression_rate = float(np.clip(suppression_rate, 0.0, 1.0))

        # Internal state / 内部状态
        self.prices: Optional[np.ndarray] = None
        self.signal_features: Optional[np.ndarray] = None
        self.representative_prices: Optional[np.ndarray] = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed / 设置随机种子"""
        return NumPySeedMixin.seed(self, seed)

    def process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw data into features / 将原始数据处理为特征

        Returns:
            prices: Price array / 价格数组
            signal_features: Processed signal features / 处理后的信号特征
        """
        signal_features = []

        for i in range(self.assets):
            df = self.dfs[i]
            start = self.frame_bound[0] - self.window_size
            end = self.frame_bound[1]

            # Get the feature columns / 获取特征列
            if self.tech_indicators and len(self.tech_indicators) > 0:
                # Assume df has columns matching tech_indicators / 假设df有匹配tech_indicators的列
                feature_data = df[start:end] if df.ndim == 2 else df
            else:
                feature_data = df[start:end]

            # Fit scaler if not already fitted / 如果scaler未拟合则拟合
            if self.scalers[i].mean_ is None:
                signal_features_i = self.scalers[i].fit_transform(feature_data)
            else:
                signal_features_i = self.scalers[i].transform(feature_data)

            signal_features.append(signal_features_i)

        # Get prices / 获取价格
        if isinstance(list(self.price_arrays.values())[0], dict):
            # Price arrays is a dict of arrays / 价格数组是数组的字典
            first_key = list(self.price_arrays.keys())[0]
            self.prices = np.array([self.price_arrays[k] for k in self.price_arrays.keys()], dtype=np.float64).T
        else:
            self.prices = np.array(list(self.price_arrays.values()), dtype=np.float64).T

        self.prices = self.prices[start:end] if self.prices.ndim > 1 else self.prices[start:end]

        # Representative index / 代表性指数
        if self.representative and self.representative in self.price_arrays:
            self.representative_prices = np.array(self.price_arrays[self.representative], dtype=np.float64)[start:end]
        else:
            # Use first available as representative / 使用第一个可用作为代表
            first_key = list(self.price_arrays.keys())[0]
            self.representative_prices = np.array(self.price_arrays[first_key], dtype=np.float64)[start:end]

        self.signal_features = np.array(signal_features, dtype=np.float64)
        self._end_tick = len(self.prices) - 1

        return self.prices, self.signal_features

    def reset(self) -> np.ndarray:
        """Reset environment to initial state / 将环境重置为初始状态

        Returns:
            observation: Initial observation / 初始观察
        """
        self._done = False
        self._current_tick = self._start_tick
        self._end_tick = len(self.prices) - 1 if self.prices is not None else 0
        self._last_trade_tick = self._current_tick - 1
        self._position = np.zeros(self.assets, dtype=np.float64)
        self._position_history = [self._position.copy()]
        self.margin = self.initial_amount
        self.portfolio = np.zeros(self.assets, dtype=np.float64)
        self.PortfolioValue = 0.0
        self.reserve = self.initial_amount
        self._total_reward = 0.0
        self._total_profit = 1.0
        self._first_rendering = True
        self.history = {}
        self.rewards = []
        self.pvs = []

        return self._get_observation()

    def _update_profit(self) -> None:
        """Update total profit calculation / 更新总利润计算"""
        self._total_profit = (self.PortfolioValue + self.reserve) / self.initial_amount

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment / 在环境中执行一步

        Args:
            actions: Actions array (num_stocks,) normalized to [-1, 1]
                   / 动作数组 (num_stocks,) 归一化到 [-1, 1]

        Returns:
            observation: Current observation / 当前观察
            reward: Step reward / 当前步奖励
            done: Whether episode is done / 回合是否结束
            info: Additional information / 附加信息
        """
        self._done = False
        self._current_tick += 1

        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.assets,):
            raise ValueError(f"Expected action shape {(self.assets,)}, got {actions.shape}")

        if self._current_tick >= self._end_tick:
            self._done = True

        # Get current prices / 获取当前价格
        current_prices = self.prices[self._current_tick] if self.prices is not None else np.zeros(self.assets)

        # Handle NaN and zero prices / 处理NaN和零价格
        current_prices = np.nan_to_num(current_prices, nan=0.0, posinf=1.0, neginf=0.0)
        current_prices_for_division = current_prices.copy()
        current_prices_for_division[current_prices_for_division == 0] = 1e9

        # Action suppression / 动作抑制
        abs_portfolio_dist = np.abs(actions)
        num_to_suppress = int(np.floor(abs_portfolio_dist.size * self.suppression_rate))
        if num_to_suppress >= abs_portfolio_dist.size:
            num_to_suppress = abs_portfolio_dist.size - 1
        if num_to_suppress > 0:
            # Suppress smallest absolute actions / 抑制最小的绝对值动作
            threshold_idx = np.argpartition(abs_portfolio_dist, kth=num_to_suppress - 1)[num_to_suppress - 1]
            threshold_val = abs_portfolio_dist[threshold_idx]
            abs_portfolio_dist[abs_portfolio_dist < threshold_val] = 0

        # Calculate margin / 计算保证金
        self.margin = self.reserve + np.sum(self.portfolio * current_prices)

        # Normalize portfolio positions / 归一化投资组合持仓
        position_denominator = float(np.sum(abs_portfolio_dist))
        if position_denominator <= 0:
            norm_margin_pos = np.zeros_like(abs_portfolio_dist)
        else:
            norm_margin_pos = (abs_portfolio_dist / position_denominator) * self.margin

        # Calculate next positions / 计算下一持仓
        next_positions = np.sign(actions) * norm_margin_pos
        change_in_positions = next_positions - self._position
        actions_in_market = np.divide(change_in_positions, current_prices_for_division).astype(np.int64)

        new_portfolio = actions_in_market + self.portfolio
        new_pv = np.sum(new_portfolio * current_prices)
        new_reserve = self.margin - new_pv

        # Calculate profit and cost / 计算利润和成本
        profit = (new_pv + new_reserve) - (self.PortfolioValue + self.reserve)
        cost = self.trade_cost * np.sum(np.abs(np.sign(actions_in_market)))

        # Update state / 更新状态
        self._position = next_positions
        self.portfolio = new_portfolio
        self.PortfolioValue = new_pv
        self.reserve = new_reserve - cost

        # Calculate step reward / 计算步奖励
        step_reward = profit - cost
        self._total_reward += self.reward_scaling * step_reward

        self.rewards.append(self._total_reward)
        self.pvs.append(float(new_pv))

        self._update_profit()

        self._position_history.append(self._position.copy())

        observation = self._get_observation()
        info = {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
        }
        self._update_history(info)

        # Check for margin call / 检查保证金追缴
        if self.margin < 0:
            self._done = True

        return observation, float(step_reward), self._done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation / 获取当前观察

        Returns:
            observation: Current observation array / 当前观察数组
        """
        if self.signal_features is None:
            return np.zeros((self.assets, self.window_size, self.state_space), dtype=np.float32)

        start_idx = self._current_tick - self.window_size + 1
        end_idx = self._current_tick + 1

        obs = self.signal_features[:, start_idx:end_idx, :]
        return np.nan_to_num(obs).astype(np.float32)

    def _update_history(self, info: Dict[str, Any]) -> None:
        """Update history with info dict / 使用info字典更新历史"""
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode: str = "human") -> None:
        """Render the environment (placeholder) / 渲染环境（占位符）"""
        # Placeholder for rendering - actual rendering would need matplotlib
        # 渲染的占位符 - 实际渲染需要matplotlib
        pass

    def close(self) -> None:
        """Clean up environment resources / 清理环境资源"""
        pass


# =============================================================================
# Fundamental Data Feature Integration / 基本面数据特征集成
# =============================================================================

class FundamentalDataFeature:
    """Fundamental data feature integration / 基本面数据特征集成

    Loads and integrates fundamental data (earnings, ratios, etc.) with
    technical price data for enhanced trading decisions.

    加载并集成基本面数据（盈利、比率等）与技术价格数据，以增强交易决策。

    Args:
        repo_path: Path to fundamental data repository / 基本面数据仓库路径
        max_files: Maximum number of files to load / 最大加载文件数

    Note:
        Uses lazy imports for pandas. Falls back gracefully if not available.
        对pandas使用懒加载。如不可用，则优雅降级。

    Attributes:
        feature_names: List of fundamental feature names / 基本面特征名称列表
        is_enabled: Whether fundamental data is loaded / 基本面数据是否已加载
        repo_path: Path to data repository / 数据仓库路径
    """

    def __init__(self, repo_path: Optional[str] = None, max_files: int = 250) -> None:
        self.repo_path = self._resolve_repo_path(repo_path)
        self.max_files = max_files
        self.feature_names: List[str] = []
        self._fundamentals_by_symbol: Dict[str, np.ndarray] = {}
        self._dates_by_symbol: Dict[str, np.ndarray] = {}

        # Lazy import pandas / 懒加载pandas
        self._pandas_available = False
        self._pd = None
        try:
            import pandas as pd
            self._pd = pd
            self._pandas_available = True
        except ImportError:
            pass

        if self.repo_path is not None and self._pandas_available:
            self._load()

    @property
    def is_enabled(self) -> bool:
        """Whether fundamental data is enabled / 基本面数据是否启用"""
        return self.repo_path is not None and len(self.feature_names) > 0

    def merge_with_price_data(
        self,
        price_data: np.ndarray,
        symbol: str,
        dates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Merge fundamental data with price data / 将基本面数据与价格数据合并

        Args:
            price_data: Price data array (time_steps, features) / 价格数据数组
            symbol: Stock symbol / 股票代码
            dates: Optional date array for alignment / 用于对齐的可选日期数组

        Returns:
            merged: Merged data array / 合并后的数据数组
        """
        if not self.is_enabled:
            return price_data

        symbol_key = str(symbol).upper()
        fundamentals = self._fundamentals_by_symbol.get(symbol_key)

        if fundamentals is None:
            # Return zeros if no fundamental data / 如果没有基本面数据则返回零
            return np.concatenate([
                price_data,
                np.zeros((len(price_data), len(self.feature_names)), dtype=np.float64)
            ], axis=1)

        # Merge by forward-filling / 通过前向填充合并
        merged = np.zeros((len(price_data), price_data.shape[1] + len(self.feature_names)), dtype=np.float64)
        merged[:, :price_data.shape[1]] = price_data

        # Copy fundamental data / 复制基本面数据
        if len(fundamentals) >= len(price_data):
            merged[:, price_data.shape[1]:] = fundamentals[:len(price_data)]
        else:
            merged[:len(fundamentals), price_data.shape[1]:] = fundamentals
            # Forward fill / 前向填充
            merged[len(fundamentals):, price_data.shape[1]:] = fundamentals[-1]

        return merged

    def _resolve_repo_path(self, repo_path: Optional[str]) -> Optional[str]:
        """Resolve repository path / 解析仓库路径"""
        if repo_path:
            path = os.path.expanduser(repo_path)
            return path if os.path.exists(path) else None

        cwd = os.getcwd()
        candidates = [
            os.path.join(cwd, "indian_stock_market_data"),
            os.path.join(cwd, "indian-stock-market-data"),
            os.path.join(os.path.dirname(cwd), "indian_stock_market_data"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _load(self) -> None:
        """Load fundamental data from repository / 从仓库加载基本面数据"""
        if not self._pandas_available or self._pd is None:
            return

        frames = []
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            for f in filenames:
                if f.endswith('.csv') or f.endswith('.parquet'):
                    files.append(os.path.join(root, f))
                    if len(files) >= self.max_files:
                        break
            if len(files) >= self.max_files:
                break

        for path in files:
            parsed = self._parse_fundamental_file(path)
            if parsed is not None and not parsed.empty:
                frames.append(parsed)

        if not frames:
            return

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["symbol", "date"], how="outer")

        merged.sort_values(["symbol", "date"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        feature_names = [col for col in merged.columns if col not in ("symbol", "date")]
        self.feature_names = [f"fund_{col}" for col in feature_names]

        for symbol, group in merged.groupby("symbol"):
            symbol_df = group.set_index("date")[feature_names].sort_index()
            self._fundamentals_by_symbol[symbol] = symbol_df.values.astype(np.float64)
            self._dates_by_symbol[symbol] = symbol_df.index.values

    def _parse_fundamental_file(self, path: str):
        """Parse a single fundamental data file / 解析单个基本面数据文件"""
        if not self._pandas_available or self._pd is None:
            return None

        try:
            if path.endswith('.csv'):
                sample = self._pd.read_csv(path, nrows=50)
                if sample.empty:
                    return None
                symbol_col, date_col = self._detect_columns(sample)
                if symbol_col is None or date_col is None:
                    return None
                full_df = self._pd.read_csv(path)
            else:
                sample = self._pd.read_parquet(path)
                if sample.empty:
                    return None
                symbol_col, date_col = self._detect_columns(sample)
                if symbol_col is None or date_col is None:
                    return None
                full_df = sample
        except Exception:
            return None

        if full_df.empty:
            return None

        full_df[date_col] = self._pd.to_datetime(full_df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
        full_df[symbol_col] = full_df[symbol_col].astype(str).str.upper().str.strip()

        numeric_cols = full_df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in [symbol_col, date_col]]

        if not numeric_cols:
            return None

        cleaned = full_df[[symbol_col, date_col] + numeric_cols].dropna(subset=[symbol_col, date_col]).copy()
        if cleaned.empty:
            return None

        grouped = cleaned.groupby([symbol_col, date_col], as_index=False).mean(numeric_only=True)
        grouped.rename(columns={symbol_col: "symbol", date_col: "date"}, inplace=True)

        return grouped

    def _detect_columns(self, frame) -> Tuple[Optional[str], Optional[str]]:
        """Detect symbol and date columns / 检测符号和日期列"""
        SYMBOL_CANDIDATES = ["symbol", "ticker", "name", "stock", "company", "tradingsymbol", "security"]
        DATE_CANDIDATES = ["date", "datetime", "report_date", "fiscal_date", "as_of_date", "published_at"]

        columns = {str(column).lower(): column for column in frame.columns}

        symbol_col = None
        date_col = None

        for candidate in SYMBOL_CANDIDATES:
            if candidate in columns:
                symbol_col = columns[candidate]
                break

        for candidate in DATE_CANDIDATES:
            if candidate in columns:
                date_col = columns[candidate]
                break

        return symbol_col, date_col


# =============================================================================
# LLM Analyst / LLM分析师
# =============================================================================

@dataclass
class LLMAnalystConfig:
    """Configuration for LLM analyst / LLM分析师配置

    Attributes:
        enabled: Whether LLM analyst is enabled / LLM分析师是否启用
        endpoint: API endpoint URL / API端点URL
        api_key: API key for authentication / 认证API密钥
        model: Model name to use / 使用的模型名称
        timeout_seconds: Request timeout in seconds / 请求超时秒数
        blend_weight: Weight for blending LLM signals with model actions
                      / LLM信号与模型动作混合的权重
    """
    enabled: bool = False
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: int = 20
    blend_weight: float = 0.25

    @classmethod
    def from_env(cls) -> "LLMAnalystConfig":
        """Create config from environment variables / 从环境变量创建配置"""
        enabled = os.getenv("LLM_ANALYST_ENABLED", "0").lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            endpoint=os.getenv("LLM_ANALYST_ENDPOINT", "").strip(),
            api_key=os.getenv("LLM_ANALYST_API_KEY", "").strip(),
            model=os.getenv("LLM_ANALYST_MODEL", "").strip(),
            timeout_seconds=int(os.getenv("LLM_ANALYST_TIMEOUT", "20")),
            blend_weight=float(np.clip(float(os.getenv("LLM_ANALYST_BLEND_WEIGHT", "0.25")), 0.0, 1.0)),
        )


class LLMAnalyst:
    """LLM-based technical analyst / 基于LLM的技术分析师

    Provides an interface to use LLM APIs for generating trading signals
    based on market data analysis.

    提供使用LLM API生成基于市场数据分析的交易信号的接口。

    Args:
        config: LLMAnalystConfig object / LLMAnalystConfig对象

    Note:
        Uses urllib for HTTP requests. Falls back gracefully on failure.
        使用urllib进行HTTP请求。失败时优雅降级。

    Attributes:
        config: Configuration object / 配置对象
    """

    def __init__(self, config: LLMAnalystConfig) -> None:
        self.config = config

    def get_signal(
        self,
        symbols: Iterable[str],
        market_snapshot: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Get trading signal from LLM / 从LLM获取交易信号

        Args:
            symbols: List of stock symbols / 股票代码列表
            market_snapshot: Dict mapping symbols to market data dicts
                            / 映射符号到市场数据字典的字典

        Returns:
            signal: Dict mapping symbols to scores in [-1, 1]
                   / 映射符号到[-1, 1]分数的字典
        """
        symbols = list(symbols)

        if not self.config.enabled:
            return {symbol: 0.0 for symbol in symbols}

        if not self.config.endpoint or not self.config.model:
            raise ValueError("LLM analyst is enabled but endpoint/model are not configured.")

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a technical analyst. Return ONLY compact JSON mapping symbols to scores in [-1,1].",
                },
                {
                    "role": "user",
                    "content": json.dumps({"symbols": symbols, "snapshot": market_snapshot}, separators=(",", ":")),
                },
            ],
            "temperature": 0,
        }

        try:
            from urllib import request
        except ImportError:
            return {symbol: 0.0 for symbol in symbols}

        try:
            req = request.Request(
                self.config.endpoint,
                method="POST",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}",
                },
            )

            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")

            parsed = json.loads(raw)
            content = self._extract_content(parsed)
            scores = json.loads(content) if isinstance(content, str) else content

            signal = {symbol: 0.0 for symbol in symbols}
            for symbol in symbols:
                if symbol in scores:
                    signal[symbol] = float(np.clip(scores[symbol], -1.0, 1.0))
            return signal

        except Exception:
            return {symbol: 0.0 for symbol in symbols}

    @staticmethod
    def _extract_content(parsed_response: Dict) -> Any:
        """Extract content from LLM response / 从LLM响应中提取内容"""
        if isinstance(parsed_response, dict) and "choices" in parsed_response:
            choice = parsed_response["choices"][0]
            message = choice.get("message", {})
            return message.get("content", "{}")
        return parsed_response


def blend_actions(
    model_actions: np.ndarray,
    llm_scores: Dict[str, float],
    symbols: Iterable[str],
    weight: float
) -> np.ndarray:
    """Blend model actions with LLM scores / 将模型动作与LLM分数混合

    Args:
        model_actions: Actions from RL model / RL模型的动作
        llm_scores: Scores from LLM analyst / LLM分析师的分数
        symbols: List of stock symbols / 股票代码列表
        weight: Blend weight for LLM scores / LLM分数的混合权重

    Returns:
        blended_actions: Blended actions / 混合后的动作
    """
    symbols = list(symbols)
    model_actions = np.asarray(model_actions, dtype=np.float64)
    llm_vector = np.array([llm_scores.get(symbol, 0.0) for symbol in symbols], dtype=np.float64)
    weight = float(np.clip(weight, 0.0, 1.0))
    blended = (1 - weight) * model_actions + weight * llm_vector
    return np.clip(blended, -1.0, 1.0)


# =============================================================================
# Technical Indicators / 技术指标
# =============================================================================

def process_indicators(
    prices: np.ndarray,
    opens: Optional[np.ndarray] = None,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    volumes: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Process technical indicators from price data
    / 从价格数据处理技术指标

    Uses talib if available, otherwise uses pure NumPy implementations.

    如果talib可用则使用，否则使用纯NumPy实现。

    Args:
        prices: Close prices (time_steps,) / 收盘价 (时间步,)
        opens: Open prices (time_steps,) / 开盘价
        highs: High prices (time_steps,) / 最高价
        lows: Low prices (time_steps,) / 最低价
        volumes: Volumes (time_steps,) / 成交量

    Returns:
        indicators: Dict of indicator arrays / 指标数组字典
    """
    indicators = {}

    if TALIB_AVAILABLE and talib is not None:
        # Use talib for efficient computation / 使用talib高效计算
        close = np.asarray(prices, dtype=np.float64)
        indicators["sma"] = talib.SMA(close)
        indicators["5sma"] = talib.SMA(close, timeperiod=5)
        indicators["20sma"] = talib.SMA(close, timeperiod=20)

        if highs is not None and lows is not None:
            high = np.asarray(highs, dtype=np.float64)
            low = np.asarray(lows, dtype=np.float64)

            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, matype=talib.MA_Type.T3)
            indicators["bb_upper"] = bb_upper
            indicators["bb_middle"] = bb_middle
            indicators["bb_lower"] = bb_lower
            indicators["bb_squeeze"] = (bb_upper - bb_lower) / bb_middle

            indicators["mom"] = talib.MOM(close, timeperiod=10)
            indicators["adx"] = talib.ADX(high, low, close, timeperiod=10)
            indicators["mfi"] = talib.MFI(high, low, close, np.asarray(volumes, dtype=np.float64) if volumes is not None else close, timeperiod=10)
            indicators["rsi"] = talib.RSI(close, timeperiod=10)
            indicators["cci"] = talib.CCI(high, low, close, timeperiod=14)

            slowk, slowd = talib.STOCH(high, low, close)
            indicators["slowk"] = slowk
            indicators["slowd"] = slowd

            macd, macdsignal, macdhist = talib.MACD(close)
            indicators["macd"] = macd
            indicators["macdsignal"] = macdsignal
            indicators["macdhist"] = macdhist

    else:
        # Pure NumPy fallback / 纯NumPy备用
        close = np.asarray(prices, dtype=np.float64)

        # Simple Moving Average / 简单移动平均
        indicators["sma"] = _sma(close, 10)
        indicators["5sma"] = _sma(close, 5)
        indicators["20sma"] = _sma(close, 20)

        # Bollinger Bands / 布林带
        sma20 = indicators["20sma"]
        std20 = _rolling_std(close, 20)
        indicators["bb_upper"] = sma20 + 2 * std20
        indicators["bb_middle"] = sma20
        indicators["bb_lower"] = sma20 - 2 * std20
        indicators["bb_squeeze"] = (indicators["bb_upper"] - indicators["bb_lower"]) / (sma20 + 1e-8)

        # RSI / 相对强弱指标
        indicators["rsi"] = _rsi(close, 10)

        # MACD / 移动平均收敛散度
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        indicators["macd"] = ema12 - ema26
        indicators["macdsignal"] = _ema(indicators["macd"], 9)
        indicators["macdhist"] = indicators["macd"] - indicators["macdsignal"]

    return indicators


# Pure NumPy indicator implementations / 纯NumPy指标实现

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average / 简单移动平均"""
    result = np.zeros_like(data, dtype=np.float64)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average / 指数移动平均"""
    alpha = 2 / (period + 1)
    result = np.zeros_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling Standard Deviation / 滚动标准差"""
    result = np.zeros_like(data, dtype=np.float64)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def _rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index / 相对强弱指数"""
    deltas = np.diff(data, prepend=data[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = _sma(gains, period)
    avg_loss = _sma(losses, period)

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Base indicator names / 基本指标名称
BASE_INDICATORS = [
    "open", "high", "low", "close", "volume",
    "ret1min", "ret2min", "ret3min", "ret4min", "ret5min",
    "sma", "5sma", "20sma",
    "bb_upper", "bb_middle", "bb_lower", "bb_squeeze",
    "mom", "rsi", "macd", "macdsignal", "macdhist",
]


# =============================================================================
# Training Utilities / 训练工具
# =============================================================================

def create_multi_stock_env(
    price_data: Dict[str, np.ndarray],
    feature_arrays: List[np.ndarray],
    indicators: List[str],
    window_size: int = 12,
    train_end: Optional[int] = None,
    initial_amount: float = 1_000_000,
    trade_cost: float = 0.0
) -> MultiStockTradingEnv:
    """Create a multi-stock trading environment / 创建多股票交易环境

    Args:
        price_data: Dict mapping stock names to price arrays
                  / 映射股票名称到价格数组的字典
        feature_arrays: List of feature arrays per stock
                       / 每个股票的特征数组列表
        indicators: List of indicator names / 指标名称列表
        window_size: Time window size / 时间窗口大小
        train_end: End index for training / 训练结束索引
        initial_amount: Initial account balance / 初始账户余额
        trade_cost: Transaction cost rate / 交易成本率

    Returns:
        env: Configured MultiStockTradingEnv / 配置好的MultiStockTradingEnv
    """
    if train_end is None:
        first_prices = list(price_data.values())[0]
        train_end = len(first_prices) - 500  # Default holdout / 默认留出

    env = MultiStockTradingEnv(
        dfs=feature_arrays,
        price_arrays=price_data,
        initial_amount=initial_amount,
        trade_cost=trade_cost,
        num_features=len(indicators),
        num_stocks=len(feature_arrays),
        window_size=window_size,
        frame_bound=(window_size, train_end),
        tech_indicator_list=indicators,
    )
    env.process_data()
    return env


# =============================================================================
# Main Entry Point / 主入口
# =============================================================================

if __name__ == "__main__":
    # Example usage / 示例用法
    print("Multi-Stock RL Trading System / 多股票强化学习交易系统")
    print("=" * 60)
    print("Available classes / 可用类:")
    for name in __all__:
        print(f"  - {name}")
    print()
    print("Requirements / 要求:")
    print("  - numpy >= 1.20")
    print("  - gymnasium (optional, for trading env)")
    print("  - talib (optional, for technical indicators)")
