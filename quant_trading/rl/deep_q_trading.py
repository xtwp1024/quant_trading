"""
Deep Q-Trading Agent - Pure NumPy Implementation
深度Q交易智能体 - 纯NumPy实现

A pure NumPy (no PyTorch/TensorFlow) reimplementation of the deep-q-trading-agent
architectures from Jeong et al. (2019).

Three DQN architectures:
- NumQAgent:      Single-branch joint DQN (action Q-values + share ratios from shared trunk)
- NumDRegADAgent: Action-Dependent deep RL agent (share sizing branch depends on chosen action)
- NumDRegIDAgent: Action-Independent deep RL agent (single scalar share ratio)

Plus:
- StockAutoencoder: Autoencoder for grouping stocks by reconstruction error
- TransferLearningTrader: Pretrain-on-components → fine-tune-on-index pipeline

Key innovation from original paper: Networks output BOTH action (BUY/HOLD/SELL)
AND number of shares, enabling share-sizing decisions.

Features / 特性:
- Pure NumPy MLP with full backpropagation (chain rule gradient descent)
- Epsilon-greedy exploration with configurable decay
- Experience replay buffer (deque-based, batch sampling)
- Target network (hard and soft Polyak update)
- Confidence-based strategy fallback (confused-market detection)
- gymnasium.Env-compatible FinanceEnvironment

Author: Claude Code
Date: 2026-03-31
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from collections import deque

# Lazy gymnasium import with graceful degradation
# gymnasium延迟导入，支持优雅降级
GYMNASIUM_AVAILABLE = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None


# =============================================================================
# Constants / 常量
# =============================================================================

# Agent method types / 智能体方法类型
NUMQ = 0          # Joint NumQ network / 联合NumQ网络
NUMDREG_AD = 1    # Action-Dependent distribution regressor / 动作依赖分布回归
NUMDREG_ID = 2    # Action-Independent distribution regressor / 动作独立分布回归

# Training mode constants for NumDReg / NumDReg训练模式常量
ACT_MODE  = 0     # Train action branch only / 仅训练动作分支
NUM_MODE  = 1     # Train number branch only / 仅训练数量分支
FULL_MODE = 2     # Train both branches jointly / 联合训练两个分支

# Action indices / 动作索引
ACTION_BUY  = 0   # Buy signal / 买入信号
ACTION_HOLD = 1   # Hold signal / 持有信号
ACTION_SELL = 2   # Sell signal / 卖出信号

ACTION_SPACE = [1, 0, -1]  # Maps index → trading action value
NUM_ACTIONS  = 3           # BUY, HOLD, SELL

# Default hyperparameters (mirrors original config.yml)
DEFAULT_CONFIG = {
    "LOOKBACK": 200,
    "REWARD_WINDOW": 100,
    "SHARE_TRADE_LIMIT": 10,
    "THRESHOLD": 0.0002,
    "EPISODES": 33,
    "EPISODES_COMPONENT_STOCKS": 10,
    "BATCH_SIZE": 64,
    "GAMMA": 0.85,
    "MEMORY_CAPACITY": 256,
    "LR": 0.0001,
    "LR_NUMDREGAD": 0.0001,
    "LR_NUMDREGID": 0.0001,
    "EPSILON": 1.0,
    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.995,
    "UPDATE_TYPE": "SOFT",
    "TAU": 0.0003,
    "EPISODES_PER_TARGET_UPDATE": 1,
    "STEPS_PER_SOFT_UPDATE": 1,
    "LOSS": "SMOOTH_L1_LOSS",
}


# =============================================================================
# Activation Functions / 激活函数
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 if x > 0 else 0."""
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of tanh: 1 - tanh²(x)."""
    t = np.tanh(x)
    return 1.0 - t * t


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation (numerically stable).
    softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    """
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_grad(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Gradient of softmax along axis.
    d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)
    """
    s = softmax(x, axis=axis)  # (..., n)
    # For each sample: grad[i,j,k] = s[i,j] * (1[i==j] - s[i,k])
    if axis == -1 or axis == len(s.shape) - 1:
        grad = s[..., np.newaxis] * (np.eye(s.shape[-1])[np.newaxis, ...] - s[..., np.newaxis, :])
    else:
        raise NotImplementedError("softmax_grad for non-last axis")
    return grad


def smooth_l1_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Smooth L1 (Huber) loss: 0.5 * x² if |x| < 1 else |x| - 0.5
    """
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    abs_diff = np.abs(diff)
    loss = np.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
    return float(np.mean(loss))


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2))


# =============================================================================
# DenseLayer — Pure NumPy Fully-Connected Layer
# =============================================================================

class DenseLayer:
    """
    A dense (fully-connected) layer with Xavier initialization and
    full backpropagation support.

    Pure NumPy实现的全连接层，支持Xavier初始化和完整反向传播。

    Attributes:
        input_size:  Number of input features / 输入特征数
        output_size: Number of output features / 输出特征数
        weights:     Weight matrix (input_size × output_size)
        biases:      Bias vector (output_size,)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_scale: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize layer with Xavier/He-style initialization.

        Args:
            input_size:  Number of input features / 输入特征数
            output_size:  Number of output features / 输出特征数
            weight_scale: Extra scale multiplier / 额外缩放因子
            seed:         Random seed / 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        # Xavier-like initialization
        scale = weight_scale * np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float64) * scale
        self.biases  = np.zeros((1, output_size), dtype=np.float64)
        self.input_size  = input_size
        self.output_size = output_size

        # For gradient caching during forward pass
        self._cached_input: Optional[np.ndarray] = None
        self._cached_output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W + b.

        Args:
            x: Input array of shape (batch, input_size) or (input_size,)

        Returns:
            Output array of shape (batch, output_size) or (output_size,)
        """
        x = np.asarray(x, dtype=np.float64)
        self._cached_input = x.copy()
        out = np.dot(x, self.weights) + self.biases
        self._cached_output = out
        return out

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass: compute gradient w.r.t. inputs.

        Args:
            grad_output: Gradient of loss w.r.t. output (batch, output_size)

        Returns:
            Gradient w.r.t. input (batch, input_size)
        """
        grad_output = np.asarray(grad_output, dtype=np.float64)
        # grad_wrt_input = grad_output @ W.T
        grad_input = np.dot(grad_output, self.weights.T)
        # Store gradients for weight update
        self._grad_weights = np.dot(
            self._cached_input.reshape(-1, self.input_size).T,
            grad_output.reshape(-1, self.output_size)
        )
        self._grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input

    def update(self, lr: float) -> None:
        """Apply accumulated gradients with learning rate."""
        self.weights -= lr * self._grad_weights
        self.biases  -= lr * self._grad_biases

    def apply_gradient(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        """
        Convenience: forward + backward + update in one call.

        Returns gradient w.r.t. input for upstream layers.
        """
        return self.backward(grad_output)

    @property
    def grad_weights(self) -> np.ndarray:
        return getattr(self, '_grad_weights', np.zeros_like(self.weights))

    @property
    def grad_biases(self) -> np.ndarray:
        return getattr(self, '_grad_biases', np.zeros_like(self.biases))


# =============================================================================
# NumQModel — Joint Q-Network (Pure NumPy)
# =============================================================================

class NumQModel:
    """
    Joint NumQ Network: single shared trunk with two output heads.

    Architecture (mirrors Jeong et al. 2019):
        Input(200) → fc1(200→200) → ReLU
                   → fc2(200→100) → ReLU
                   → fc3(100→50)  → ReLU
                   ├──→ fc_q(50→3)   → Q-values  [BUY, HOLD, SELL]
                   └──→ softmax(fc_q(sigmoid(fc3))) → share ratios

    Returns:
        q: Q-values array of shape (batch, 3) or (3,)
        r: Share ratios (softmax) of shape (batch, 3) or (3,)

    网络结构（镜像Jeong等2019）：
        输入(200) → fc1(200→200) → ReLU
                   → fc2(200→100) → ReLU
                   → fc3(100→50)  → ReLU
                   ├──→ fc_q(50→3)   → Q值 [买入, 持有, 卖出]
                   └──→ softmax(fc_q(sigmoid(fc3))) → 股份比例
    """

    LOOKBACK = 200

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.input_size  = self.LOOKBACK
        self.output_size = NUM_ACTIONS

        # Shared trunk / 共享骨干网络
        self.fc1 = DenseLayer(200, 200, seed=seed)
        self.fc2 = DenseLayer(200, 100, seed=seed)
        self.fc3 = DenseLayer(100,  50, seed=seed)

        # Q-value head (maps shared features → Q per action)
        self.fc_q = DenseLayer(50, NUM_ACTIONS, seed=seed)

        # Cached activations for backprop
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass computing Q-values and share ratios.

        Args:
            x: State array of shape (batch, 200) or (200,)

        Returns:
            Tuple of (Q-values [batch, 3], share ratios [batch, 3])
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Shared trunk
        h1 = relu(self.fc1.forward(x))
        h2 = relu(self.fc2.forward(h1))
        h3 = relu(self.fc3.forward(h2))

        # Cache for backprop
        self._cache['h3'] = h3
        self._cache['h2'] = h2
        self._cache['h1'] = h1
        self._cache['x']  = x

        # Q-value head
        q = self.fc_q.forward(h3)

        # Share-ratio head: softmax(fc_q(sigmoid(h3)))
        r = softmax(self.fc_q.forward(sigmoid(h3)), axis=-1)

        # Return scalar if single sample
        if q.shape[0] == 1:
            q = q.squeeze(0)
            r = r.squeeze(0)
        return q, r

    def backward_q(
        self,
        action: int,
        td_error: np.ndarray
    ) -> None:
        """
        Backpropagate Q-branch gradient into all layers.

        Args:
            action:   Index of action taken (0=BUY, 1=HOLD, 2=SELL)
            td_error: TD error for the taken action (scalar or array)
        """
        td_error = np.asarray(td_error, dtype=np.float64)
        batch = td_error.shape[0] if td_error.ndim > 0 else 1

        # grad_loss_wrt_q = one_hot(action) * td_error
        grad_q = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
        if batch == 1:
            grad_q[0, action] = td_error
        else:
            grad_q[np.arange(batch), action] = td_error

        # Backprop through fc_q
        grad_h3 = self.fc_q.backward(grad_q)
        grad_act_h3 = relu_grad(self._cache['h3']) * grad_h3

        # Backprop through shared trunk
        grad_h2 = self.fc3.backward(grad_act_h3)
        grad_act_h2 = relu_grad(self._cache['h2']) * grad_h2
        grad_h1 = self.fc2.backward(grad_act_h2)
        grad_act_h1 = relu_grad(self._cache['h1']) * grad_h1
        self.fc1.backward(grad_act_h1)

    def backward_r(
        self,
        action: int,
        num_error: np.ndarray
    ) -> None:
        """
        Backpropagate number-branch gradient into all layers.

        Args:
            action:   Index of action taken
            num_error: Error signal for number/position-size prediction
        """
        num_error = np.asarray(num_error, dtype=np.float64)
        batch = num_error.shape[0] if num_error.ndim > 0 else 1

        # Grad w.r.t. softmax input (chain rule: softmax_grad ⊙ error)
        softmax_in = sigmoid(self._cache['h3'])
        s = softmax(self.fc_q.forward(softmax_in), axis=-1)

        # dL/dr * dr/d(softmax_in)
        # For the taken action: dL/d(softmax_in)[action] = (s[action] - 1) * num_error
        # For other actions:     dL/d(softmax_in)[j]     = -s[j] * num_error (if j == action)
        grad_softmax_in = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
        if batch == 1:
            grad_softmax_in[0, action] = (s[0, action] - 1.0) * num_error
            for j in range(NUM_ACTIONS):
                if j != action:
                    grad_softmax_in[0, j] = -s[0, j] * num_error
        else:
            for i in range(batch):
                grad_softmax_in[i, action] = (s[i, action] - 1.0) * num_error[i]
                for j in range(NUM_ACTIONS):
                    if j != action:
                        grad_softmax_in[i, j] = -s[i, j] * num_error[i]

        # grad through sigmoid: dL/d(h3) = grad_softmax_in * sigmoid_grad(h3) * W_q.T
        grad_h3_sigmoid = sigmoid_grad(self._cache['h3']) * np.dot(
            grad_softmax_in, self.fc_q.weights.T
        )

        grad_act_h3 = relu_grad(self._cache['h3']) * grad_h3_sigmoid
        grad_h2 = self.fc3.backward(grad_act_h3)
        grad_act_h2 = relu_grad(self._cache['h2']) * grad_h2
        grad_h1 = self.fc2.backward(grad_act_h2)
        grad_act_h1 = relu_grad(self._cache['h1']) * grad_h1
        self.fc1.backward(grad_act_h1)

    def update_weights(self, lr: float) -> None:
        """Apply gradient updates to all layers."""
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)
        self.fc_q.update(lr)

    def _get_layers(self) -> List[DenseLayer]:
        """Return all layers in order (for weight sync utilities)."""
        return [self.fc1, self.fc2, self.fc3, self.fc_q]


# =============================================================================
# NumDRegModel — Dual-Branch Distribution Regressor (Pure NumPy)
# =============================================================================

class NumDRegModel:
    """
    Dual-branch Distribution Regressor Network.

    Two variants (controlled by `method`):
        NUMDREG_AD: Action-Dependent — fc_r outputs 3 values (one per action)
        NUMDREG_ID: Action-Independent — fc_r outputs 1 value (shared)

    Architecture (mirrors Jeong et al. 2019):
        Input(200) → fc1(200→100) → ReLU
        ├──→ Action branch:
        │    fc2_act(100→50) → ReLU
        │    fc3_act(50→20)  → ReLU
        │    fc_q(20→3)      → Q-values  [BUY, HOLD, SELL]
        └──→ Number branch:
             fc2_num(100→50) → ReLU
             fc3_num(50→20)  → Sigmoid
             ├──→ fc_r(20→3) → share ratios per action (AD)
             └──→ fc_r(20→1) → single share ratio (ID)

    网络结构（镜像Jeong等2019）：
        输入(200) → fc1(200→100) → ReLU
        ├──→ 动作分支：
        │    fc2_act(100→50) → ReLU
        │    fc3_act(50→20)  → ReLU
        │    fc_q(20→3)      → Q值 [买入, 持有, 卖出]
        └──→ 数量分支：
             fc2_num(100→50) → ReLU
             fc3_num(50→20)  → Sigmoid
             ├──→ fc_r(20→3) → 每个动作的股份比例 (AD)
             └──→ fc_r(20→1) → 单一股份比例 (ID)

    Mode during training (`mode`):
        ACT_MODE:  Action-branch step — number from softmax(fc_q(sigmoid(x_act)))
        NUM_MODE:  Number-branch step — full number branch
        FULL_MODE: Both branches active
    """

    LOOKBACK = 200

    def __init__(
        self,
        method: int = NUMDREG_AD,
        mode: int = FULL_MODE,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
        self.method = method
        self.mode   = mode
        self.input_size = self.LOOKBACK
        self.num_outputs = NUM_ACTIONS if method == NUMDREG_AD else 1

        # Root shared layer
        self.fc1 = DenseLayer(200, 100, seed=seed)

        # Action branch
        self.fc2_act = DenseLayer(100, 50, seed=seed)
        self.fc3_act = DenseLayer(50,  20, seed=seed)
        self.fc_q    = DenseLayer(20, NUM_ACTIONS, seed=seed)

        # Number branch
        self.fc2_num = DenseLayer(100, 50, seed=seed)
        self.fc3_num = DenseLayer(50,  20, seed=seed)
        self.fc_r    = DenseLayer(20, self.num_outputs, seed=seed)

        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass computing Q-values and number/position-size values.

        Args:
            x: State array of shape (batch, 200) or (200,)

        Returns:
            Tuple of (Q-values [batch, 3], number values [batch, 3] or [batch, 1])
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Root
        h = relu(self.fc1.forward(x))
        self._cache['h'] = h
        self._cache['x'] = x

        # Action branch
        h_act = relu(self.fc2_act.forward(h))
        h_act = relu(self.fc3_act.forward(h_act))
        q = self.fc_q.forward(h_act)

        self._cache['h_act'] = h_act

        # Number branch
        if self.mode == ACT_MODE:
            # Derive shares from action branch (Q-based softmax)
            r = softmax(self.fc_q.forward(sigmoid(h_act)), axis=-1)
        else:
            h_num = relu(self.fc2_num.forward(h))
            h_num = sigmoid(self.fc3_num.forward(h_num))
            self._cache['h_num'] = h_num
            if self.method == NUMDREG_ID:
                r = sigmoid(self.fc_r.forward(h_num))
            else:
                r = softmax(self.fc_r.forward(h_num), axis=-1)

        if q.shape[0] == 1:
            q = q.squeeze(0)
            r = r.squeeze(0)
        return q, r

    def backward_q(self, action: int, td_error: np.ndarray) -> None:
        """Backpropagate Q-branch gradients into all action-branch layers."""
        td_error = np.asarray(td_error, dtype=np.float64)
        batch = td_error.shape[0] if td_error.ndim > 0 else 1

        grad_q = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
        if batch == 1:
            grad_q[0, action] = td_error
        else:
            grad_q[np.arange(batch), action] = td_error

        grad_h_act = self.fc_q.backward(grad_q)
        grad_act_h_act = relu_grad(self._cache['h_act']) * grad_h_act
        grad_h = self.fc3_act.backward(grad_act_h_act)
        grad_act_h = relu_grad(self._cache['h']) * grad_h
        self.fc2_act.backward(grad_act_h)
        self.fc1.backward(grad_act_h)

    def backward_r(
        self,
        action: int,
        num_error: np.ndarray
    ) -> None:
        """Backpropagate number-branch gradients into all number-branch layers."""
        num_error = np.asarray(num_error, dtype=np.float64)
        batch = num_error.shape[0] if num_error.ndim > 0 else 1

        if self.method == NUMDREG_ID:
            # Single scalar output; grad is just num_error
            grad_r = num_error.reshape(-1, 1) if batch > 1 else np.array([[num_error]])
            grad_h_num = self.fc_r.backward(grad_r)
            grad_sigmoid = sigmoid_grad(self._cache['h_num']) * grad_h_num
            grad_h = self.fc3_num.backward(grad_sigmoid)
            grad_act_h = relu_grad(self._cache['h']) * grad_h
            self.fc2_num.backward(grad_act_h)
            self.fc1.backward(grad_act_h)
        else:
            # AD: 3 outputs (one per action), only the taken action matters
            grad_r = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
            if batch == 1:
                grad_r[0, action] = num_error
            else:
                grad_r[np.arange(batch), action] = num_error

            grad_h_num = self.fc_r.backward(grad_r)
            grad_sigmoid = sigmoid_grad(self._cache['h_num']) * grad_h_num
            grad_h = self.fc3_num.backward(grad_sigmoid)
            grad_act_h = relu_grad(self._cache['h']) * grad_h
            self.fc2_num.backward(grad_act_h)
            self.fc1.backward(grad_act_h)

    def update_weights(self, lr: float) -> None:
        """Apply gradient updates to all layers."""
        self.fc1.update(lr)
        self.fc2_act.update(lr)
        self.fc3_act.update(lr)
        self.fc_q.update(lr)
        self.fc2_num.update(lr)
        self.fc3_num.update(lr)
        self.fc_r.update(lr)

    def _get_layers(self) -> List[DenseLayer]:
        """Return all layers in order (for weight sync utilities)."""
        return [self.fc1, self.fc2_act, self.fc3_act, self.fc_q,
                self.fc2_num, self.fc3_num, self.fc_r]

    def set_mode(self, mode: int) -> None:
        """Set training mode: ACT_MODE, NUM_MODE, or FULL_MODE."""
        self.mode = mode


# =============================================================================
# StockAutoencoder — Autoencoder for Stock Grouping
# =============================================================================

class StockAutoencoder:
    """
    Autoencoder for grouping stocks by reconstruction error.

    Architecture (mirrors Jeong et al. / StonksNet):
        Input (num_components,) → fc1(num_components → 5) → ReLU
                                → out(5 → num_components) → ReLU

    The encoder compresses each day's price vector (across all component stocks)
    into a 5-dimensional latent representation. The decoder reconstructs it.

    Stocks are grouped by MSE between input and reconstructed prices:
    - Low MSE → stocks well-represented by the index representation
    - High MSE → stocks with idiosyncratic behaviour

    This grouping is used to select component stocks for transfer-learning
    pretraining (Jeong et al. section 3.3).

    网络结构（镜像Jeong等 / StonksNet）：
        输入 (num_components,) → fc1(num_components → 5) → ReLU
                                → out(5 → num_components) → ReLU

    用法：
        >>> ae = StockAutoencoder(num_components=30)
        >>> ae.train(daily_price_matrix, epochs=20, lr=0.0001)
        >>> groups = ae.group_stocks(daily_price_matrix, symbols)
    """

    def __init__(
        self,
        num_components: int,
        latent_dim: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize autoencoder.

        Args:
            num_components: Number of stocks (input dimension) / 股票数量（输入维度）
            latent_dim:     Dimension of latent space (default 5) / 潜空间维度
            seed:           Random seed / 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        self.num_components = num_components
        self.latent_dim     = latent_dim

        self.encoder = DenseLayer(num_components, latent_dim, seed=seed)
        self.decoder = DenseLayer(latent_dim,      num_components, seed=seed)

        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode then decode a batch of daily price vectors.

        Args:
            x: Price vector of shape (batch, num_components) or (num_components,)

        Returns:
            Tuple of (latent representation, reconstructed prices)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        z = relu(self.encoder.forward(x))
        x_recon = relu(self.decoder.forward(z))

        self._cache['x'] = x
        self._cache['z'] = z
        return z, x_recon

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backpropagate reconstruction loss gradient.

        Args:
            loss_grad: Gradient of loss w.r.t. reconstruction (batch, num_components)
        """
        loss_grad = np.asarray(loss_grad, dtype=np.float64)
        grad_z = self.decoder.backward(loss_grad)
        grad_act_z = relu_grad(self._cache['z']) * grad_z
        self.encoder.backward(grad_act_z)

    def update_weights(self, lr: float) -> None:
        """Apply accumulated gradients."""
        self.encoder.update(lr)
        self.decoder.update(lr)

    def reconstruction_error(self, x: np.ndarray) -> float:
        """
        Compute mean squared reconstruction error.

        Args:
            x: Input price vector(s) of shape (batch, num_components)

        Returns:
            Mean squared error
        """
        _, recon = self.forward(x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(np.mean((x - recon) ** 2))

    def train(
        self,
        price_matrix: np.ndarray,
        epochs: int = 20,
        lr: float = 0.0001,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        Train autoencoder on daily price vectors (row-per-day).

        Args:
            price_matrix: Matrix of shape (num_days, num_components)
            epochs:       Number of training epochs / 训练轮数
            lr:           Learning rate / 学习率
            batch_size:   Mini-batch size / 小批量大小
            verbose:      Print epoch losses / 打印每轮损失

        Returns:
            List of average losses per epoch
        """
        price_matrix = np.asarray(price_matrix, dtype=np.float64)
        n_days = price_matrix.shape[0]
        losses = []

        for ep in range(epochs):
            epoch_losses = []
            indices = np.random.permutation(n_days)

            for start in range(0, n_days, batch_size):
                batch = price_matrix[indices[start:start + batch_size]]
                _, recon = self.forward(batch)
                loss_grad = 2.0 * (recon - batch) / (batch.shape[0] * self.num_components + 1e-8)
                self.backward(loss_grad)
                self.update_weights(lr)
                epoch_losses.append(np.mean((batch - recon) ** 2))

            avg_loss = float(np.mean(epoch_losses))
            losses.append(avg_loss)
            if verbose and (ep + 1) % 5 == 0:
                print(f"  [Autoencoder] Epoch {ep + 1}/{epochs} | MSE: {avg_loss:.6f}")

        return losses

    def group_stocks(
        self,
        price_matrix: np.ndarray,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Group stocks by reconstruction MSE (low → well-represented, high → idiosyncratic).

        Returns dict with keys: 'mse_high', 'mse_low', 'mse_highlow'
        Each maps to array of symbol indices sorted accordingly.

        Args:
            price_matrix: Matrix of shape (num_days, num_components)
            symbols:      Optional list of stock symbols (len = num_components)

        Returns:
            Dict with grouping arrays keyed by 'mse_high', 'mse_low', 'mse_highlow'
        """
        price_matrix = np.asarray(price_matrix, dtype=np.float64)
        n_days, n_components = price_matrix.shape

        # Compute per-stock MSE: average over all days
        per_stock_mse = np.zeros(n_components)
        for i in range(n_components):
            _, recon_i = self.forward(price_matrix[:, i])
            per_stock_mse[i] = float(np.mean((price_matrix[:, i] - recon_i[:, i if recon_i.shape[0] == 1 else slice(None)]) ** 2))

        # Recompute more cleanly: forward through all days at once
        _, recon_all = self.forward(price_matrix)  # (n_days, n_components)
        per_stock_mse = np.mean((price_matrix - recon_all) ** 2, axis=0)

        half = n_components // 2
        sorted_idx = np.argsort(per_stock_mse)

        result = {
            'mse_high':    sorted_idx[:half] if half > 0 else sorted_idx,
            'mse_low':     sorted_idx[-half:] if half > 0 else sorted_idx,
            'mse_highlow': np.concatenate([sorted_idx[-half:], sorted_idx[:half]]),
            'per_stock_mse': per_stock_mse,
        }

        if symbols is not None:
            result['symbols_by_mse'] = [symbols[i] for i in sorted_idx]

        return result


# =============================================================================
# ReplayBuffer — Experience Replay for DQN
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer (deque-based, fixed capacity).

    Stores (state, action, reward_all_actions, next_state) transitions
    compatible with the Jeong et al. style reward computation.

    经验回放缓冲区（基于deque，固定容量）。

    存储 (状态, 动作, 所有动作的奖励, 下一状态) 转换。
    """

    def __init__(self, capacity: int = 256):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of transitions / 最大转换数
        """
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        rewards_all_actions: np.ndarray,
        next_state: np.ndarray
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state:               Current state (lookback,) / 当前状态
            action:              Action taken (scalar) / 采取的动作
            rewards_all_actions: Rewards for all 3 actions [BUY, HOLD, SELL]
            next_state:          Next state (lookback,) / 下一状态
        """
        self.buffer.append((
            np.asarray(state, dtype=np.float64),
            int(action),
            np.asarray(rewards_all_actions, dtype=np.float64),
            np.asarray(next_state, dtype=np.float64)
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards_all, next_states)
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = list(zip(*np.random.choice(len(self.buffer), batch_size, replace=False)))
        return tuple(np.array(item) for item in batch)

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# FinanceEnvironment — Trading Environment (NumPy-only)
# =============================================================================

class FinanceEnvironment:
    """
    NumPy-based trading environment (no gymnasium required).

    Mirrors the FinanceEnvironment from pipelines/finance_environment.py
    with the key innovation: the state is a lookback window of daily price
    deltas, and the reward is computed for all 3 actions simultaneously
    to enable Q-learning over the joint action-space.

    State:   Price deltas over `lookback` days  (shape: [lookback])
    Actions: BUY(1), HOLD(0), SELL(-1)  encoded as indices 0,1,2
    Reward:  r = num * (1 + action * Δp/p) * prev_price / init_price

    不依赖gymnasium的NumPy交易环境。

    状态：过去`lookback`天的价格变化（形状：[lookback]）
    动作：买入(1)、持有(0)、卖出(-1)，编码为索引0,1,2
    奖励：r = num * (1 + action * Δp/p) * prev_price / init_price
    """

    def __init__(
        self,
        price_history: np.ndarray,
        date_history: Optional[np.ndarray] = None,
        lookback: int = 200,
        reward_window: int = 100,
        start_index: int = 0,
        end_index: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize environment.

        Args:
            price_history: Array of historical prices / 历史价格数组
            date_history:  Optional array of dates / 可选日期数组
            lookback:      Number of past prices for state / 状态回顾窗口
            reward_window: Window for computing profit baseline / 利润基准窗口
            start_index:   Starting row index / 起始行索引
            end_index:     Ending row index / 结束行索引
            seed:          Random seed / 随机种子
        """
        if seed is not None:
            np.random.seed(seed)

        self.price_history = np.asarray(price_history, dtype=np.float64).flatten()
        n = len(self.price_history)

        self.date_history = (
            np.asarray(date_history) if date_history is not None
            else np.arange(n)
        )
        if self.date_history.ndim != 1:
            self.date_history = self.date_history.flatten()

        self.lookback      = lookback
        self.reward_window = reward_window

        # Episode bounds adjusted for lookback padding
        self._raw_start = start_index
        self._raw_end   = end_index if end_index is not None else n - 1

        # Precompute price deltas (with lookback padding at the front)
        # Pad with the first price delta = 0 to create lookback warm-up values
        price_deltas = np.diff(self.price_history, prepend=self.price_history[0])
        price_deltas[0] = 0.0
        self.price_deltas_padded = price_deltas  # length = n

        # We will index into padded_deltas with offset = lookback
        self._offset = lookback

        # BUY=1, HOLD=0, SELL=-1
        self.action_space_vals = np.array([1.0, 0.0, -1.0], dtype=np.float64)

        # Replay memory
        self.replay_memory = ReplayBuffer(capacity=256)

        self._reset_internal()

    def _reset_internal(self) -> None:
        """Reset per-episode state."""
        self.time_step     = self._offset + self._raw_start
        self.end_time_step = self._offset + self._raw_end + 1

        self.current_price = self.price_history[self.time_step]
        self.prev_price    = self.current_price
        self.init_price    = self.price_history[
            max(self.time_step - self.reward_window, self._offset)
        ]

        self.episode_profit = 0.0
        self.episode_rewards: List[float] = []
        self.episode_losses:  List[float] = []
        self.current_action:  int = ACTION_HOLD
        self.current_num:     float = 0.0

    def start_episode(self, start_with_padding: bool = True) -> None:
        """
        Start a new episode.

        Args:
            start_with_padding: If True, include lookback warmup;
                               if False, skip the first lookback steps
        """
        self._reset_internal()
        if not start_with_padding:
            self.time_step += self.lookback

    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Advance environment by one timestep.

        Returns:
            Tuple of (state, done)
            - state: price deltas from (t-lookback) to (t-1), shape (lookback,)
            - done:  True if past end date
        """
        self.prev_price    = self.current_price
        self.current_price = self.price_history[self.time_step]

        # State: lookback price deltas ending at t-1
        start = self.time_step - self.lookback
        end   = self.time_step
        state = self.price_deltas_padded[start:end].copy()

        # Move to next timestep
        self.time_step += 1

        # Check if past the end
        done = self.time_step >= self.end_time_step

        if done:
            self.next_state = np.zeros(self.lookback, dtype=np.float64)
        else:
            ns_start = self.time_step - self.lookback
            ns_end   = self.time_step
            self.next_state = self.price_deltas_padded[ns_start:ns_end].copy()

        return state, done

    def compute_reward_all_actions(
        self,
        action_index: int,
        num: float
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute profit and reward for a given action and share count,
        and also compute rewards for ALL 3 actions (for Q-learning).

        Args:
            action_index: Index of chosen action (0=BUY, 1=HOLD, 2=SELL)
            num:          Number of shares to trade / 交易股数

        Returns:
            Tuple of (profit, reward, rewards_all_actions)
            - profit:            Profit from chosen action
            - reward:            Reward from chosen action
            - rewards_all_actions: Rewards for all 3 actions [BUY, HOLD, SELL]
        """
        action_val = self.action_space_vals[action_index]
        num = float(num)

        delta = (self.current_price - self.prev_price) / (self.prev_price + 1e-12)

        profit = num * action_val * delta
        reward = num * (1.0 + action_val * delta) * self.prev_price / (self.init_price + 1e-12)

        # Rewards for all 3 actions (used in replay buffer)
        rewards_all = np.array([
            num * (1.0 + av * delta) * self.prev_price / (self.init_price + 1e-12)
            for av in self.action_space_vals
        ], dtype=np.float64)

        self.episode_profit  += profit
        self.episode_rewards.append(float(reward))

        return profit, float(reward), rewards_all

    def compute_profit_and_reward(
        self,
        action_index: int,
        num: float
    ) -> Tuple[float, float]:
        """Compute profit and reward for a single (action, num) pair."""
        action_val = float(self.action_space_vals[action_index])
        num = float(num)
        delta = (self.current_price - self.prev_price) / (self.prev_price + 1e-12)
        profit = num * action_val * delta
        reward = num * (1.0 + action_val * delta) * self.prev_price / (self.init_price + 1e-12)
        self.episode_profit += profit
        self.episode_rewards.append(float(reward))
        return float(profit), float(reward)

    def update_replay_memory(self) -> None:
        """Push current transition into replay buffer."""
        if self.next_state.shape[0] == self.lookback:
            self.replay_memory.push(
                self.state_buffer,
                self.current_action,
                self.rewards_all_buffer,
                self.next_state
            )

    def add_loss(self, loss: float) -> None:
        """Record a loss value for this episode."""
        if loss is not None and not np.isnan(loss):
            self.episode_losses.append(float(loss))

    def on_episode_end(self) -> Tuple[float, float, float]:
        """
        Called at episode termination.

        Returns:
            Tuple of (avg_loss, avg_reward, total_profit)
        """
        avg_loss = float(np.mean(self.episode_losses)) if self.episode_losses else 0.0
        avg_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        return avg_loss, avg_reward, float(self.episode_profit)

    @staticmethod
    def load_prices_from_csv(
        file_path: str,
        date_col: int = 0,
        price_col: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load price history from a CSV file.

        Args:
            file_path: Path to CSV with date and price columns
            date_col:  Index of date column
            price_col: Index of price column

        Returns:
            Tuple of (price_history, date_history)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for FinanceEnvironment.load_prices_from_csv")

        df = pd.read_csv(file_path)
        dates  = pd.to_datetime(df.iloc[:, date_col]).values
        prices = df.iloc[:, price_col].astype(np.float64).values
        return prices, dates


# =============================================================================
# Agent Classes / 智能体类
# =============================================================================

class BaseAgent:
    """
    Base class for all DQN trading agents.
    所有DQN交易智能体的基类。

    Provides common infrastructure:
    - Epsilon-greedy exploration / epsilon贪婪探索
    - Target network sync (hard + soft Polyak) / 目标网络同步
    - Gradient update utilities / 梯度更新工具
    """

    BUY  = ACTION_BUY
    HOLD = ACTION_HOLD
    SELL = ACTION_SELL

    def __init__(
        self,
        state_size: int = 200,
        learning_rate: float = 0.0001,
        gamma: float = 0.85,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
        self.state_size  = state_size
        self.lr           = learning_rate
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._step_count  = 0

    # -------------------------------------------------------------------------
    # predict_action — core inference method (override in subclass)
    # -------------------------------------------------------------------------
    def predict_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Predict the best action index for a given state.

        Args:
            state:    Current state array of shape (state_size,)
            training: If True, apply epsilon-greedy exploration

        Returns:
            Action index: 0=BUY, 1=HOLD, 2=SELL
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def predict_shares(self, state: np.ndarray, action: int) -> float:
        """
        Predict number of shares to trade given state and action.

        Args:
            state:  Current state array
            action: Chosen action index (0=BUY, 1=HOLD, 2=SELL)

        Returns:
            Number of shares (scaled by SHARES_TRADE_LIMIT externally)
        """
        raise NotImplementedError

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all 3 actions. Must be implemented by subclass."""
        raise NotImplementedError

    def train(
        self,
        env: FinanceEnvironment,
        episodes: int = 33,
        batch_size: int = 64,
        share_limit: float = 10.0,
        target_update_freq: int = 1,
        update_type: str = "SOFT",
        tau: float = 0.0003,
        strategy: int = ACTION_HOLD,
        use_strategy: bool = False,
        threshold: float = 0.0002,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the agent on a given environment.

        Args:
            env:               FinanceEnvironment instance
            episodes:          Number of training episodes / 训练回合数
            batch_size:        Replay batch size / 回放批量大小
            share_limit:       Max shares to trade (scaling factor) / 最大交易股数
            target_update_freq: Episodes between hard target updates
            update_type:      "SOFT" (Polyak) or "HARD" / 目标网络更新类型
            tau:               Polyak interpolation factor / Polyak插值因子
            strategy:          Fallback action when confidence is low / 置信度低时回退动作
            use_strategy:     Use strategy fallback / 使用策略回退
            threshold:         Confidence threshold / 置信度阈值
            verbose:           Print progress / 打印进度

        Returns:
            Dict with keys: 'losses', 'rewards', 'profits', 'val_rewards', 'val_profits'
        """
        raise NotImplementedError

    def transfer_learn(
        self,
        source_weights: List[np.ndarray],
        freeze_trunk: bool = False
    ) -> None:
        """
        Load pretrained weights (transfer learning).

        Args:
            source_weights: List of weight arrays from a pretrained model
            freeze_trunk:   If True, freeze the shared layers / 如果为True则冻结共享层
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Common utilities / 通用工具
    # -------------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self, tau: float = 0.001) -> None:
        """Polyak soft update: θ_target = τ*θ_policy + (1-τ)*θ_target."""
        raise NotImplementedError

    def hard_update(self) -> None:
        """Hard update: copy policy net to target net."""
        raise NotImplementedError


# =============================================================================
# NumQAgent — Joint Q-Network Trading Agent
# =============================================================================

class NumQAgent(BaseAgent):
    """
    NumQ Agent: single-branch DQN predicting both action and share quantity.

    The network shares a trunk (fc1→fc2→fc3) and then branches into:
      - Q-head: outputs 3 Q-values (BUY/HOLD/SELL)
      - R-head: softmax(fc_q(sigmoid(h3))) → 3 share ratios

    For inference:
      action  = argmax Q(s, a)
      shares  = SHARES_LIMIT * r[action]

    联合Q网络智能体：单分支DQN，同时预测动作和股份数量。

    网络共享骨干（fc1→fc2→fc3），然后分叉为：
      - Q头：输出3个Q值（买入/持有/卖出）
      - R头：softmax(fc_q(sigmoid(h3))) → 3个股份比例

    Example:
        >>> agent = NumQAgent(state_size=200, learning_rate=0.0001, gamma=0.85)
        >>> agent.train(env, episodes=33)
        >>> action = agent.predict_action(state)
        >>> shares = agent.predict_shares(state, action)
    """

    def __init__(
        self,
        state_size: int = 200,
        learning_rate: float = 0.0001,
        gamma: float = 0.85,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1,
        loss_type: str = "SMOOTH_L1_LOSS",
        seed: Optional[int] = None
    ):
        """
        Initialize NumQ agent.

        Args:
            state_size:          Size of state vector (default 200) / 状态向量大小
            learning_rate:       Learning rate / 学习率
            gamma:               Discount factor / 折扣因子
            epsilon:             Initial exploration rate / 初始探索率
            epsilon_min:         Minimum exploration rate / 最小探索率
            epsilon_decay:       Epsilon decay rate / epsilon衰减率
            target_update_freq:  Steps between target network updates / 目标网络更新间隔
            loss_type:           "SMOOTH_L1_LOSS" or "MSE_LOSS"
            seed:                Random seed / 随机种子
        """
        super().__init__(
            state_size=state_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed
        )
        self.target_update_freq = target_update_freq
        self.loss_type          = loss_type

        self.policy_net  = NumQModel(seed=seed)
        self.target_net  = NumQModel(seed=seed)
        self._sync_target()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all 3 actions."""
        q, _ = self.policy_net.forward(state)
        return q.squeeze() if q.ndim > 1 else q

    def predict_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Predict best action using epsilon-greedy policy.

        Args:
            state:    State array of shape (200,)
            training: Apply epsilon-greedy exploration

        Returns:
            Action index (0=BUY, 1=HOLD, 2=SELL)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q, _ = self.policy_net.forward(state)
        return int(np.argmax(q.squeeze() if q.ndim > 1 else q))

    def predict_shares(self, state: np.ndarray, action: int) -> float:
        """
        Predict share ratio for a given state and action.

        The share ratio is the softmax probability from the R-head
        corresponding to the chosen action.

        Args:
            state:  State array of shape (200,)
            action: Action index (0, 1, or 2)

        Returns:
            Share ratio (0 to 1) — multiply by SHARES_TRADE_LIMIT for actual shares
        """
        _, r = self.policy_net.forward(state)
        r = r.squeeze() if r.ndim > 1 else r
        action = int(action)
        if action < len(r):
            return float(r[action])
        return float(r[0])

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        env: FinanceEnvironment,
        episodes: int = 33,
        batch_size: int = 64,
        share_limit: float = 10.0,
        target_update_freq: int = 1,
        update_type: str = "SOFT",
        tau: float = 0.0003,
        strategy: int = ACTION_HOLD,
        use_strategy: bool = False,
        threshold: float = 0.0002,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train NumQ agent on a FinanceEnvironment.

        Each step:
          1. Agent selects (action, num) based on current state
          2. Environment computes rewards for all 3 actions
          3. Transition stored in replay memory
          4. If memory ready: sample batch, compute TD error, backprop, update

        Returns:
            Dict with keys: 'losses', 'rewards', 'profits', 'val_rewards', 'val_profits'
        """
        self.target_update_freq = target_update_freq
        losses, rewards, profits = [], [], []
        val_rewards_list, val_profits_list = [], []

        optim_steps = 0

        for ep in range(episodes):
            env.start_episode(start_with_padding=True)
            ep_losses, ep_rewards = [], []

            for step_idx in range(10000):  # safety cap
                state, done = env.step()

                # Store state for replay buffer
                env.state_buffer = state.copy()

                # Select action and shares
                action_idx = self.predict_action(state, training=True)
                share_ratio = self.predict_shares(state, action_idx)
                num_shares  = share_limit * share_ratio

                # Compute reward for all 3 actions (needed for replay)
                profit, reward, rewards_all = env.compute_reward_all_actions(
                    action_idx, num_shares
                )
                env.rewards_all_buffer = rewards_all.copy()
                env.current_action     = action_idx
                env.current_num        = num_shares

                # Update replay memory
                env.update_replay_memory()

                # Training step if memory ready
                if len(env.replay_memory) < batch_size:
                    if done:
                        break
                    continue

                # Sample batch
                states_b, actions_b, rewards_all_b, next_states_b = \
                    env.replay_memory.sample(batch_size)

                # Flatten to (batch, lookback) if needed
                if states_b.ndim == 1:
                    states_b = states_b.reshape(1, -1)

                # Current Q values
                q_batch, _ = self.policy_net.forward(states_b)
                q_batch = q_batch.squeeze() if q_batch.ndim == 2 else q_batch

                # Next Q values from target net
                next_q_batch, _ = self.target_net.forward(next_states_b)
                next_q_batch = next_q_batch.squeeze() if next_q_batch.ndim == 2 else next_q_batch

                # Q-learning target
                next_max_q = np.max(next_q_batch, axis=-1)
                td_targets = rewards_all_b[:, action_idx] + self.gamma * next_max_q * (1 - 0.0)  # done handled implicitly

                # Compute loss
                if self.loss_type == "MSE_LOSS":
                    loss = mse_loss(q_batch[:, action_idx], td_targets)
                else:
                    loss = smooth_l1_loss(q_batch[:, action_idx], td_targets)

                ep_losses.append(loss)
                ep_rewards.append(float(reward))

                # Backprop through Q-branch
                self.policy_net.backward_q(action_idx, td_targets - q_batch[:, action_idx])
                self.policy_net.update_weights(self.lr)

                optim_steps += 1

                # Target network update
                if update_type == "SOFT" and optim_steps % 1 == 0:
                    self.soft_update(tau)
                elif update_type == "HARD" and optim_steps % target_update_freq == 0:
                    self.hard_update()

                # Confidence-based strategy fallback
                if use_strategy:
                    q_vals = self.get_q_values(state)
                    confidence = np.abs(q_vals[ACTION_BUY] - q_vals[ACTION_SELL]) / (np.sum(q_vals) + 1e-8)
                    if confidence < threshold:
                        action_idx = strategy

                if done:
                    break

            # Episode summary
            env.current_action = action_idx
            env.current_num = num_shares
            avg_loss, avg_reward, total_profit = env.on_episode_end()
            losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
            rewards.append(avg_reward)
            profits.append(total_profit)

            # Quick validation (HOLD strategy baseline)
            val_reward, val_profit = self._evaluate(env, share_limit, strategy=strategy)
            val_rewards_list.append(val_reward)
            val_profits_list.append(val_profit)

            self.decay_epsilon()

            if verbose and (ep + 1) % 5 == 0:
                print(f"  [NumQ] Ep {ep+1}/{episodes} | "
                      f"Loss={losses[-1]:.4f} | Reward={avg_reward:.4f} | "
                      f"Profit={total_profit:.4f} | ValReward={val_reward:.4f} | "
                      f"ε={self.epsilon:.4f}")

        return {
            'losses':      losses,
            'rewards':     rewards,
            'profits':     profits,
            'val_rewards': val_rewards_list,
            'val_profits': val_profits_list,
        }

    def _evaluate(
        self,
        env: FinanceEnvironment,
        share_limit: float,
        strategy: int = ACTION_HOLD
    ) -> Tuple[float, float]:
        """Quick validation pass (one episode, no exploration)."""
        env.start_episode(start_with_padding=True)
        total_reward, total_profit = 0.0, 0.0

        for _ in range(10000):
            state, done = env.step()
            action_idx  = self.predict_action(state, training=False)
            share_ratio = self.predict_shares(state, action_idx)
            num_shares  = share_limit * share_ratio
            _, reward   = env.compute_profit_and_reward(action_idx, num_shares)
            total_reward += reward
            if done:
                break

        return total_reward, env.episode_profit

    # -------------------------------------------------------------------------
    # Target network updates
    # -------------------------------------------------------------------------
    def hard_update(self) -> None:
        """Copy policy net weights to target net."""
        self._sync_target()

    def soft_update(self, tau: float = 0.001) -> None:
        """Polyak averaging: θ_target = τ*θ_policy + (1-τ)*θ_target."""
        self._polyak_update(tau)

    def _sync_target(self) -> None:
        """Hard sync: copy all weights from policy_net to target_net."""
        for p, t in zip(
            [self.policy_net.fc1, self.policy_net.fc2, self.policy_net.fc3, self.policy_net.fc_q],
            [self.target_net.fc1,  self.target_net.fc2,  self.target_net.fc3,  self.target_net.fc_q]
        ):
            t.weights = p.weights.copy()
            t.biases  = p.biases.copy()

    def _polyak_update(self, tau: float) -> None:
        """Polyak (soft) update."""
        for p, t in zip(
            [self.policy_net.fc1, self.policy_net.fc2, self.policy_net.fc3, self.policy_net.fc_q],
            [self.target_net.fc1,  self.target_net.fc2,  self.target_net.fc3,  self.target_net.fc_q]
        ):
            t.weights = tau * p.weights + (1.0 - tau) * t.weights
            t.biases  = tau * p.biases  + (1.0 - tau) * t.biases

    # -------------------------------------------------------------------------
    # Transfer learning
    # -------------------------------------------------------------------------
    def transfer_learn(
        self,
        source_weights: List[np.ndarray],
        freeze_trunk: bool = False
    ) -> None:
        """
        Load pretrained weights into policy net.

        Args:
            source_weights: Ordered list of weight arrays
                           [fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b, fc_q_W, fc_q_b]
            freeze_trunk:  If True, only update the Q head / 如果为True则只更新Q头
        """
        net = self.policy_net
        layer_pairs = (
            [(net.fc1, 0)] if freeze_trunk else
            [(net.fc1, 0), (net.fc2, 2), (net.fc3, 4), (net.fc_q, 6)]
        )

        for layer, idx in layer_pairs:
            if idx + 1 < len(source_weights):
                layer.weights = source_weights[idx].copy()
                layer.biases  = source_weights[idx + 1].copy()

        if freeze_trunk:
            # Also sync target net's trunk
            self._sync_target()


# =============================================================================
# NumDRegADAgent — Action-Dependent Deep RL Agent
# =============================================================================

class NumDRegADAgent(BaseAgent):
    """
    NumDReg-AD Agent: Action-Dependent distribution regressor.

    Architecture:
        - Shared root: fc1(200→100)
        - Action branch: fc2_act→fc3_act→fc_q (Q-values for 3 actions)
        - Number branch: fc2_num→fc3_num→fc_r (3 outputs: one share ratio per action)

    Mode controls which branch is trained:
        ACT_MODE:  Only the action branch is updated (number = softmax(Q))
        NUM_MODE:  Only the number branch is updated
        FULL_MODE: Both branches are updated

    动作依赖分布回归智能体。

    网络结构：
        - 共享根层：fc1(200→100)
        - 动作分支：fc2_act→fc3_act→fc_q (3个动作的Q值)
        - 数量分支：fc2_num→fc3_num→fc_r (3个输出：每个动作一个股份比例)

    Example:
        >>> agent = NumDRegADAgent(learning_rate=0.0001, gamma=0.85)
        >>> agent.train(env, episodes=33)
        >>> action = agent.predict_action(state)
        >>> shares = agent.predict_shares(state, action)
    """

    def __init__(
        self,
        state_size: int = 200,
        learning_rate: float = 0.0001,
        gamma: float = 0.85,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        mode: int = FULL_MODE,
        target_update_freq: int = 1,
        loss_type: str = "SMOOTH_L1_LOSS",
        seed: Optional[int] = None
    ):
        super().__init__(
            state_size=state_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed
        )
        self.mode              = mode
        self.target_update_freq = target_update_freq
        self.loss_type         = loss_type

        self.policy_net  = NumDRegModel(method=NUMDREG_AD, mode=mode, seed=seed)
        self.target_net  = NumDRegModel(method=NUMDREG_AD, mode=mode, seed=seed)
        self._sync_target()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        q, _ = self.policy_net.forward(state)
        return q.squeeze() if q.ndim > 1 else q

    def predict_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q, _ = self.policy_net.forward(state)
        return int(np.argmax(q.squeeze() if q.ndim > 1 else q))

    def predict_shares(self, state: np.ndarray, action: int) -> float:
        """
        Predict share ratio for given state and action.

        For NumDReg-AD, each action has its own share ratio (action-dependent).
        The network outputs 3 values [r_buy, r_hold, r_sell].
        We return r[action].
        """
        _, r = self.policy_net.forward(state)
        r = r.squeeze() if r.ndim > 1 else r
        action = int(action)
        return float(r[action]) if action < len(r) else float(r[0])

    def train(
        self,
        env: FinanceEnvironment,
        episodes: int = 33,
        batch_size: int = 64,
        share_limit: float = 10.0,
        target_update_freq: int = 1,
        update_type: str = "SOFT",
        tau: float = 0.0003,
        strategy: int = ACTION_HOLD,
        use_strategy: bool = False,
        threshold: float = 0.0002,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train NumDReg-AD agent (alternates between action and number branch)."""
        self.target_update_freq = target_update_freq
        losses, rewards, profits = [], [], []
        val_rewards_list, val_profits_list = [], []
        optim_steps = 0

        for ep in range(episodes):
            env.start_episode(start_with_padding=True)
            ep_losses, ep_rewards = [], []

            for step_idx in range(10000):
                state, done = env.step()
                env.state_buffer = state.copy()

                action_idx  = self.predict_action(state, training=True)
                share_ratio = self.predict_shares(state, action_idx)
                num_shares  = share_limit * share_ratio

                profit, reward, rewards_all = env.compute_reward_all_actions(
                    action_idx, num_shares
                )
                env.rewards_all_buffer = rewards_all.copy()
                env.current_action     = action_idx
                env.current_num        = num_shares
                env.update_replay_memory()

                if len(env.replay_memory) < batch_size:
                    if done:
                        break
                    continue

                states_b, actions_b, rewards_all_b, next_states_b = \
                    env.replay_memory.sample(batch_size)

                if states_b.ndim == 1:
                    states_b = states_b.reshape(1, -1)

                # ---- Action branch update ----
                self.policy_net.set_mode(ACT_MODE)
                q_batch, _ = self.policy_net.forward(states_b)
                q_batch = q_batch.squeeze() if q_batch.ndim == 2 else q_batch
                next_q_batch, _ = self.target_net.forward(next_states_b)
                next_q_batch = next_q_batch.squeeze() if next_q_batch.ndim == 2 else next_q_batch

                next_max_q  = np.max(next_q_batch, axis=-1)
                td_targets  = rewards_all_b[np.arange(len(actions_b)), actions_b] \
                              + self.gamma * next_max_q
                q_taken     = q_batch[np.arange(len(actions_b)), actions_b]

                if self.loss_type == "MSE_LOSS":
                    act_loss = float(np.mean((q_taken - td_targets) ** 2))
                else:
                    act_loss = smooth_l1_loss(q_taken, td_targets)

                self.policy_net.backward_q(actions_b[0] if len(actions_b) == 1 else actions_b,
                                           td_targets - q_taken)
                self.policy_net.update_weights(self.lr)

                # ---- Number branch update ----
                self.policy_net.set_mode(NUM_MODE)
                _, r_batch = self.policy_net.forward(states_b)
                r_batch = r_batch.squeeze() if r_batch.ndim == 2 else r_batch

                # Target for number: reward + gamma * max(next_r)
                next_r_batch, _ = self.target_net.forward(next_states_b)
                next_r_batch = next_r_batch.squeeze() if next_r_batch.ndim == 2 else next_r_batch
                if next_r_batch.ndim == 1:
                    next_r_batch = next_r_batch.reshape(-1, 1)

                next_max_r    = np.max(next_r_batch, axis=-1)
                num_targets   = rewards_all_b[np.arange(len(actions_b)), actions_b] \
                                + self.gamma * next_max_r
                r_taken       = r_batch[np.arange(len(actions_b)), actions_b] \
                                  if r_batch.ndim > 1 else r_batch

                if self.loss_type == "MSE_LOSS":
                    num_loss = float(np.mean((r_taken - num_targets) ** 2))
                else:
                    num_loss = smooth_l1_loss(r_taken, num_targets)

                self.policy_net.backward_r(actions_b[0] if len(actions_b) == 1 else actions_b,
                                              num_targets - r_taken)
                self.policy_net.update_weights(self.lr)

                total_loss = act_loss + num_loss
                ep_losses.append(total_loss)
                ep_rewards.append(float(reward))

                optim_steps += 1
                if update_type == "SOFT":
                    self.soft_update(tau)
                elif update_type == "HARD" and optim_steps % target_update_freq == 0:
                    self.hard_update()

                if done:
                    break

            env.current_action = action_idx
            env.current_num    = num_shares
            avg_loss, avg_reward, total_profit = env.on_episode_end()
            losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
            rewards.append(avg_reward)
            profits.append(total_profit)

            val_reward, val_profit = self._evaluate(env, share_limit)
            val_rewards_list.append(val_reward)
            val_profits_list.append(val_profit)

            self.decay_epsilon()

            if verbose and (ep + 1) % 5 == 0:
                print(f"  [NumDReg-AD] Ep {ep+1}/{episodes} | "
                      f"Loss={losses[-1]:.4f} | Reward={avg_reward:.4f} | "
                      f"ε={self.epsilon:.4f}")

        return {
            'losses':      losses,
            'rewards':     rewards,
            'profits':     profits,
            'val_rewards': val_rewards_list,
            'val_profits': val_profits_list,
        }

    def _evaluate(self, env: FinanceEnvironment, share_limit: float) -> Tuple[float, float]:
        env.start_episode(start_with_padding=True)
        total_reward = 0.0
        for _ in range(10000):
            state, done = env.step()
            action_idx  = self.predict_action(state, training=False)
            share_ratio = self.predict_shares(state, action_idx)
            num_shares  = share_limit * share_ratio
            _, reward   = env.compute_profit_and_reward(action_idx, num_shares)
            total_reward += reward
            if done:
                break
        return total_reward, env.episode_profit

    def hard_update(self) -> None:
        self._sync_target()

    def soft_update(self, tau: float = 0.001) -> None:
        self._polyak_update(tau)

    def _sync_target(self) -> None:
        for p, t in zip(self.policy_net._get_layers(), self.target_net._get_layers()):
            t.weights = p.weights.copy()
            t.biases  = p.biases.copy()

    def _polyak_update(self, tau: float) -> None:
        for p, t in zip(self.policy_net._get_layers(), self.target_net._get_layers()):
            t.weights = tau * p.weights + (1.0 - tau) * t.weights
            t.biases  = tau * p.biases  + (1.0 - tau) * t.biases

    def transfer_learn(
        self,
        source_weights: List[np.ndarray],
        freeze_trunk: bool = False
    ) -> None:
        """Load pretrained weights into policy net."""
        net = self.policy_net
        all_layers = [net.fc1, net.fc2_act, net.fc3_act, net.fc_q,
                      net.fc2_num, net.fc3_num, net.fc_r]
        if freeze_trunk:
            target_layers = [net.fc1]
        else:
            target_layers = all_layers

        for layer, idx in zip(target_layers, range(0, len(target_layers) * 2, 2)):
            if idx + 1 < len(source_weights):
                layer.weights = source_weights[idx].copy()
                layer.biases  = source_weights[idx + 1].copy()

        self._sync_target()


# =============================================================================
# NumDRegIDAgent — Action-Independent Deep RL Agent
# =============================================================================

class NumDRegIDAgent(BaseAgent):
    """
    NumDReg-ID Agent: Action-Independent distribution regressor.

    Same dual-branch structure as NumDReg-AD, but the number branch
    outputs a SINGLE scalar (sigmoid) representing position size/confidence
    independent of which action was taken.

    Architecture difference from AD:
        - fc_r(20→1) instead of fc_r(20→3)
        - Output is sigmoid (not softmax): represents confidence/position size

    The rationale: the number of shares should depend on the quality of the
    trading opportunity (confidence), not on which action is being taken.

    动作独立分布回归智能体。

    与NumDReg-AD相同的双分支结构，但数量分支输出单个标量（sigmoid），
    表示与所采取的动作无关的仓位大小/置信度。

    Example:
        >>> agent = NumDRegIDAgent(learning_rate=0.0001, gamma=0.85)
        >>> agent.train(env, episodes=33)
        >>> action = agent.predict_action(state)
        >>> shares = agent.predict_shares(state, action)
    """

    def __init__(
        self,
        state_size: int = 200,
        learning_rate: float = 0.0001,
        gamma: float = 0.85,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        mode: int = FULL_MODE,
        target_update_freq: int = 1,
        loss_type: str = "SMOOTH_L1_LOSS",
        seed: Optional[int] = None
    ):
        super().__init__(
            state_size=state_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed
        )
        self.mode              = mode
        self.target_update_freq = target_update_freq
        self.loss_type         = loss_type

        self.policy_net  = NumDRegModel(method=NUMDREG_ID, mode=mode, seed=seed)
        self.target_net  = NumDRegModel(method=NUMDREG_ID, mode=mode, seed=seed)
        self._sync_target()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        q, _ = self.policy_net.forward(state)
        return q.squeeze() if q.ndim > 1 else q

    def predict_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q, _ = self.policy_net.forward(state)
        return int(np.argmax(q.squeeze() if q.ndim > 1 else q))

    def predict_shares(self, state: np.ndarray, action: int) -> float:
        """
        Predict share ratio (scalar) for given state.

        For NumDReg-ID, the network outputs a single sigmoid value
        representing confidence/position size regardless of action.
        """
        _, r = self.policy_net.forward(state)
        r = r.squeeze() if r.ndim > 1 else r
        return float(r) if np.isscalar(r) or r.ndim == 0 else float(r[0])

    def train(
        self,
        env: FinanceEnvironment,
        episodes: int = 33,
        batch_size: int = 64,
        share_limit: float = 10.0,
        target_update_freq: int = 1,
        update_type: str = "SOFT",
        tau: float = 0.0003,
        strategy: int = ACTION_HOLD,
        use_strategy: bool = False,
        threshold: float = 0.0002,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train NumDReg-ID agent."""
        self.target_update_freq = target_update_freq
        losses, rewards, profits = [], [], []
        val_rewards_list, val_profits_list = [], []
        optim_steps = 0

        for ep in range(episodes):
            env.start_episode(start_with_padding=True)
            ep_losses, ep_rewards = [], []

            for step_idx in range(10000):
                state, done = env.step()
                env.state_buffer = state.copy()

                action_idx  = self.predict_action(state, training=True)
                share_ratio = self.predict_shares(state, action_idx)
                num_shares  = share_limit * share_ratio

                profit, reward, rewards_all = env.compute_reward_all_actions(
                    action_idx, num_shares
                )
                env.rewards_all_buffer = rewards_all.copy()
                env.current_action     = action_idx
                env.current_num        = num_shares
                env.update_replay_memory()

                if len(env.replay_memory) < batch_size:
                    if done:
                        break
                    continue

                states_b, actions_b, rewards_all_b, next_states_b = \
                    env.replay_memory.sample(batch_size)

                if states_b.ndim == 1:
                    states_b = states_b.reshape(1, -1)

                actions_b = np.asarray(actions_b, dtype=np.int64)

                # ---- Action branch update ----
                self.policy_net.set_mode(ACT_MODE)
                q_batch, _ = self.policy_net.forward(states_b)
                q_batch = q_batch.squeeze() if q_batch.ndim == 2 else q_batch

                next_q_batch, _ = self.target_net.forward(next_states_b)
                next_q_batch = next_q_batch.squeeze() if next_q_batch.ndim == 2 else next_q_batch

                next_max_q  = np.max(next_q_batch, axis=-1)
                td_targets  = rewards_all_b[np.arange(len(actions_b)), actions_b] \
                              + self.gamma * next_max_q
                q_taken     = q_batch[np.arange(len(actions_b)), actions_b]

                if self.loss_type == "MSE_LOSS":
                    act_loss = float(np.mean((q_taken - td_targets) ** 2))
                else:
                    act_loss = smooth_l1_loss(q_taken, td_targets)

                self.policy_net.backward_q(actions_b, td_targets - q_taken)
                self.policy_net.update_weights(self.lr)

                # ---- Number branch update (scalar output for ID) ----
                self.policy_net.set_mode(NUM_MODE)
                _, r_batch = self.policy_net.forward(states_b)
                r_batch = r_batch.squeeze() if r_batch.ndim == 2 else r_batch

                next_r_batch, _ = self.target_net.forward(next_states_b)
                next_r_batch = next_r_batch.squeeze() if next_r_batch.ndim == 2 else next_r_batch
                next_scalar_r  = float(np.mean(next_r_batch))

                num_targets = float(rewards_all_b[0, actions_b[0]]) + self.gamma * next_scalar_r
                if self.loss_type == "MSE_LOSS":
                    num_loss = float((r_batch - num_targets) ** 2)
                else:
                    num_loss = smooth_l1_loss(r_batch, num_targets)

                self.policy_net.backward_r(actions_b[0] if len(actions_b) == 1 else actions_b,
                                            num_targets - r_batch)
                self.policy_net.update_weights(self.lr)

                total_loss = act_loss + num_loss
                ep_losses.append(total_loss)
                ep_rewards.append(float(reward))

                optim_steps += 1
                if update_type == "SOFT":
                    self.soft_update(tau)
                elif update_type == "HARD" and optim_steps % target_update_freq == 0:
                    self.hard_update()

                if done:
                    break

            env.current_action = action_idx
            env.current_num    = num_shares
            avg_loss, avg_reward, total_profit = env.on_episode_end()
            losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
            rewards.append(avg_reward)
            profits.append(total_profit)

            val_reward, val_profit = self._evaluate(env, share_limit)
            val_rewards_list.append(val_reward)
            val_profits_list.append(val_profit)

            self.decay_epsilon()

            if verbose and (ep + 1) % 5 == 0:
                print(f"  [NumDReg-ID] Ep {ep+1}/{episodes} | "
                      f"Loss={losses[-1]:.4f} | Reward={avg_reward:.4f} | "
                      f"ε={self.epsilon:.4f}")

        return {
            'losses':      losses,
            'rewards':     rewards,
            'profits':     profits,
            'val_rewards': val_rewards_list,
            'val_profits': val_profits_list,
        }

    def _evaluate(self, env: FinanceEnvironment, share_limit: float) -> Tuple[float, float]:
        env.start_episode(start_with_padding=True)
        total_reward = 0.0
        for _ in range(10000):
            state, done = env.step()
            action_idx  = self.predict_action(state, training=False)
            share_ratio = self.predict_shares(state, action_idx)
            num_shares  = share_limit * share_ratio
            _, reward   = env.compute_profit_and_reward(action_idx, num_shares)
            total_reward += reward
            if done:
                break
        return total_reward, env.episode_profit

    def hard_update(self) -> None:
        self._sync_target()

    def soft_update(self, tau: float = 0.001) -> None:
        self._polyak_update(tau)

    def _sync_target(self) -> None:
        for p, t in zip(self.policy_net._get_layers(), self.target_net._get_layers()):
            t.weights = p.weights.copy()
            t.biases  = p.biases.copy()

    def _polyak_update(self, tau: float) -> None:
        for p, t in zip(self.policy_net._get_layers(), self.target_net._get_layers()):
            t.weights = tau * p.weights + (1.0 - tau) * t.weights
            t.biases  = tau * p.biases  + (1.0 - tau) * t.biases

    def transfer_learn(
        self,
        source_weights: List[np.ndarray],
        freeze_trunk: bool = False
    ) -> None:
        """Load pretrained weights into policy net."""
        net = self.policy_net
        all_layers = [net.fc1, net.fc2_act, net.fc3_act, net.fc_q,
                      net.fc2_num, net.fc3_num, net.fc_r]
        if freeze_trunk:
            target_layers = [net.fc1]
        else:
            target_layers = all_layers

        for layer, idx in zip(target_layers, range(0, len(target_layers) * 2, 2)):
            if idx + 1 < len(source_weights):
                layer.weights = source_weights[idx].copy()
                layer.biases  = source_weights[idx + 1].copy()

        self._sync_target()


# =============================================================================
# TransferLearningTrader — Transfer Learning Pipeline
# =============================================================================

class TransferLearningTrader:
    """
    Transfer learning pipeline for pretraining on component stocks
    and fine-tuning on index data.

    Based on Jeong et al. (2019) Section 3.3 Transfer Learning:
      1. Pretrain DQN on individual component stocks (easier learning task)
      2. Use the pretrained trunk (shared feature extractor) for the index agent
      3. Fine-tune on index data with lower learning rate

    The autoencoder is used to group component stocks by reconstruction error:
      - High-correlation + Low-MSE stocks → most relevant for index
      - Mixed groups used to verify generalization

    迁移学习管道：先在成分股上预训练，再在指数数据上微调。

    步骤：
      1. 在各成分股上预训练DQN（更简单的学习任务）
      2. 将预训练的共享特征提取器用于指数智能体
      3. 用较低学习率在指数数据上微调

    Example:
        >>> trader = TransferLearningTrader(agent_class=NumQAgent)
        >>> trader.pretrain(component_envs, episodes=10)
        >>> trader.finetune(index_env, episodes=33)
        >>> action = trader.predict_action(state)
    """

    def __init__(
        self,
        agent_class: type = NumQAgent,
        state_size: int = 200,
        gamma: float = 0.85,
        lr_pretrain: float = 0.0001,
        lr_finetune: float = 0.00003,
        seed: Optional[int] = None
    ):
        """
        Initialize transfer learning trader.

        Args:
            agent_class:      Agent class to use (NumQAgent, NumDRegADAgent, NumDRegIDAgent)
            state_size:       Size of state vector / 状态向量大小
            gamma:            Discount factor / 折扣因子
            lr_pretrain:      Learning rate for pretraining / 预训练学习率
            lr_finetune:      Learning rate for fine-tuning / 微调学习率
            seed:             Random seed / 随机种子
        """
        self.agent_class  = agent_class
        self.state_size   = state_size
        self.gamma        = gamma
        self.lr_pretrain  = lr_pretrain
        self.lr_finetune  = lr_finetune
        self.seed         = seed

        self.pretrained_agent: Optional[BaseAgent] = None
        self.finetuned_agent:  Optional[BaseAgent] = None
        self.autoencoder:      Optional[StockAutoencoder] = None
        self._is_pretrained    = False

    def pretrain(
        self,
        envs: Dict[str, FinanceEnvironment],
        episodes: int = 10,
        batch_size: int = 64,
        share_limit: float = 10.0,
        verbose: bool = True
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Pretrain on multiple component stock environments.

        Args:
            envs:     Dict mapping stock symbol → FinanceEnvironment
            episodes: Number of episodes per stock / 每只股票的回合数
            batch_size: Replay batch size / 回放批量大小
            share_limit: Max shares / 最大交易股数

        Returns:
            Dict mapping symbol → training history dict
        """
        if verbose:
            print(f"\n[TransferLearning] Starting pretraining on {len(envs)} component stocks | "
                  f"episodes={episodes}")

        results = {}
        for symbol, env in envs.items():
            if verbose:
                print(f"\n[TransferLearning] === Pretraining on {symbol} ===")

            agent = self.agent_class(
                state_size=self.state_size,
                learning_rate=self.lr_pretrain,
                gamma=self.gamma,
                seed=self.seed
            )

            history = agent.train(
                env,
                episodes=episodes,
                batch_size=batch_size,
                share_limit=share_limit,
                verbose=verbose
            )
            results[symbol] = history

            if verbose:
                final_reward = history['rewards'][-1] if history['rewards'] else 0.0
                final_profit = history['profits'][-1] if history['profits'] else 0.0
                print(f"[TransferLearning] {symbol} final | reward={final_reward:.4f} | profit={final_profit:.4f}")

        # Use the last trained agent as the pretrained model
        self.pretrained_agent = agent
        self._is_pretrained   = True

        if verbose:
            print(f"\n[TransferLearning] Pretraining complete.")

        return results

    def finetune(
        self,
        env: FinanceEnvironment,
        episodes: int = 33,
        batch_size: int = 64,
        share_limit: float = 10.0,
        freeze_trunk: bool = False,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the pretrained agent on index data.

        Args:
            env:           Index FinanceEnvironment
            episodes:      Number of fine-tuning episodes / 微调回合数
            batch_size:    Replay batch size / 回放批量大小
            share_limit:   Max shares / 最大交易股数
            freeze_trunk:  If True, only fine-tune the Q head / 如果为True则只微调Q头
            verbose:       Print progress / 打印进度

        Returns:
            Training history dict with keys: 'losses', 'rewards', 'profits'
        """
        if not self._is_pretrained:
            if verbose:
                print("[TransferLearning] No pretrained agent — training from scratch.")
            agent = self.agent_class(
                state_size=self.state_size,
                learning_rate=self.lr_finetune,
                gamma=self.gamma,
                seed=self.seed
            )
        else:
            if verbose:
                print(f"[TransferLearning] Fine-tuning pretrained agent | freeze_trunk={freeze_trunk}")

            # Create new agent and load pretrained weights
            agent = self.agent_class(
                state_size=self.state_size,
                learning_rate=self.lr_finetune,
                gamma=self.gamma,
                seed=self.seed
            )
            # Transfer weights from pretrained
            src_net = self.pretrained_agent.policy_net
            src_weights = []
            for layer in src_net._get_layers():
                src_weights.extend([layer.weights, layer.biases])
            agent.transfer_learn(src_weights, freeze_trunk=freeze_trunk)

        self.finetuned_agent = agent

        history = agent.train(
            env,
            episodes=episodes,
            batch_size=batch_size,
            share_limit=share_limit,
            verbose=verbose
        )

        if verbose:
            final_reward = history['rewards'][-1] if history['rewards'] else 0.0
            final_profit = history['profits'][-1] if history['profits'] else 0.0
            print(f"\n[TransferLearning] Fine-tuning complete | "
                  f"final_reward={final_reward:.4f} | final_profit={final_profit:.4f}")

        return history

    def predict_action(self, state: np.ndarray, training: bool = False) -> int:
        """Predict action using fine-tuned agent (or pretrained if not finetuned)."""
        agent = self.finetuned_agent or self.pretrained_agent
        if agent is None:
            raise RuntimeError("TransferLearningTrader: call pretrain() or finetune() first")
        return agent.predict_action(state, training=training)

    def predict_shares(self, state: np.ndarray, action: int) -> float:
        """Predict share ratio using fine-tuned agent (or pretrained if not finetuned)."""
        agent = self.finetuned_agent or self.pretrained_agent
        if agent is None:
            raise RuntimeError("TransferLearningTrader: call pretrain() or finetune() first")
        return agent.predict_shares(state, action)

    def train_autoencoder(
        self,
        price_matrix: np.ndarray,
        epochs: int = 20,
        lr: float = 0.0001,
        verbose: bool = True
    ) -> StockAutoencoder:
        """
        Train autoencoder on daily price vectors for stock grouping.

        Args:
            price_matrix: Matrix of shape (num_days, num_components)
            epochs:       Training epochs / 训练轮数
            lr:           Learning rate / 学习率
            verbose:      Print progress / 打印进度

        Returns:
            Trained StockAutoencoder instance
        """
        n_components = price_matrix.shape[1]
        self.autoencoder = StockAutoencoder(
            num_components=n_components,
            seed=self.seed
        )
        if verbose:
            print(f"\n[TransferLearning] Training autoencoder | "
                  f"days={price_matrix.shape[0]}, components={n_components}")
        losses = self.autoencoder.train(
            price_matrix,
            epochs=epochs,
            lr=lr,
            verbose=verbose
        )
        if verbose:
            print(f"[TransferLearning] Autoencoder final MSE: {losses[-1]:.6f}")
        return self.autoencoder

    def get_groups(
        self,
        price_matrix: np.ndarray,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get stock groups from autoencoder (must call train_autoencoder first).

        Returns dict with keys:
            'mse_high', 'mse_low', 'mse_highlow', 'per_stock_mse', 'symbols_by_mse'
        """
        if self.autoencoder is None:
            raise RuntimeError("Call train_autoencoder() first")
        return self.autoencoder.group_stocks(price_matrix, symbols)


# =============================================================================
# Utility Functions / 工具函数
# =============================================================================

def compute_confidence(q_values: np.ndarray) -> float:
    """
    Compute market-confidence score from Q-values.

    Confidence = |Q(BUY) - Q(SELL)| / sum(Q)

    High confidence → market trend is clear (BUY vs SELL diverging)
    Low confidence  → confused market (Q-values similar) → use strategy fallback

    Args:
        q_values: Q-values array of shape (3,)

    Returns:
        Confidence score between 0 and 1
    """
    q_values = np.asarray(q_values, dtype=np.float64).squeeze()
    return float(
        np.abs(q_values[ACTION_BUY] - q_values[ACTION_SELL]) /
        (np.sum(q_values) + 1e-12)
    )


def batch_compute_td_errors(
    q_values: np.ndarray,
    next_q_values: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.85
) -> np.ndarray:
    """
    Compute TD errors for a batch of transitions.

    Args:
        q_values:     Q-values   (batch_size, num_actions)
        next_q_values: Next Q-values (batch_size, num_actions)
        actions:      Taken actions (batch_size,)
        rewards:      Received rewards (batch_size,)
        dones:        Done flags (batch_size,)
        gamma:        Discount factor

    Returns:
        TD errors for each sample (batch_size,)
    """
    q_values     = np.asarray(q_values, dtype=np.float64)
    next_q_values = np.asarray(next_q_values, dtype=np.float64)
    actions      = np.asarray(actions, dtype=np.int64)
    rewards      = np.asarray(rewards, dtype=np.float64)
    dones        = np.asarray(dones, dtype=np.float64)

    next_max_q = np.max(next_q_values, axis=-1)
    td_target  = rewards + gamma * next_max_q * (1.0 - dones)

    batch_idx  = np.arange(len(actions))
    q_taken    = q_values[batch_idx, actions]

    return td_target - q_taken


# =============================================================================
# Exports / 导出
# =============================================================================

__all__ = [
    # Constants
    'NUMQ', 'NUMDREG_AD', 'NUMDREG_ID',
    'ACT_MODE', 'NUM_MODE', 'FULL_MODE',
    'ACTION_BUY', 'ACTION_HOLD', 'ACTION_SELL',
    'ACTION_SPACE', 'NUM_ACTIONS',
    'DEFAULT_CONFIG',

    # Activation functions
    'relu', 'sigmoid', 'tanh', 'softmax',
    'relu_grad', 'sigmoid_grad', 'tanh_grad',
    'smooth_l1_loss', 'mse_loss',

    # Layer
    'DenseLayer',

    # Models
    'NumQModel', 'NumDRegModel',

    # Stock grouping
    'StockAutoencoder',

    # Environment
    'FinanceEnvironment', 'ReplayBuffer',

    # Agents
    'BaseAgent',
    'NumQAgent',
    'NumDRegADAgent',
    'NumDRegIDAgent',

    # Transfer learning
    'TransferLearningTrader',

    # Utilities
    'compute_confidence', 'batch_compute_td_errors',

    # Availability
    'GYMNASIUM_AVAILABLE',
]
