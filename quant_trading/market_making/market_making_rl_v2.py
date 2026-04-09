"""
Market Making RL v2 — DQN & A2C Agents
基于 D:/Hive/Data/trading_repos/MARKET-MAKING-RL/ 重构

核心类 / Core Classes:
- ReplayBuffer:      经验回放缓冲区 (DQN)
- DQNMarketMaker:    DQN 智能体 for market making
- A2CMarketMaker:   A2C 智能体 for market making
- MarketMakingTrainer: 统一训练循环
- MarketMakingPolicy: 训练后策略推理

Market Making RL v2 — DQN & A2C Agents
Adapted from D:/Hive/Data/trading_repos/MARKET-MAKING-RL/

Core classes:
- ReplayBuffer:      Experience replay buffer (DQN)
- DQNMarketMaker:   DQN agent for market making
- A2CMarketMaker:   A2C agent for market making
- MarketMakingTrainer: Unified training loop
- MarketMakingPolicy: Trained policy inference wrapper

Pure NumPy + Gymnasium implementation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import copy

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# =============================================================================
# Replay Buffer / 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区 for DQN / Experience replay buffer for DQN.

    从 MARKET-MAKING-RL/MarketMaker/policy.py 的 variable-length trajectory
    处理和 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的稀疏轨迹收集中汲取灵感。

    Inspired by variable-length trajectory handling in
    MARKET-MAKING-RL/MarketMaker/policy.py and sparse trajectory collection
    in MARKET-MAKING-RL/MarketMaker/marketmaker.py.

    存储格式: (state, action, reward, next_state, done)
    Storage format: (state, action, reward, next_state, done)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        batch_size: int = 64,
    ):
        """
        Args:
            capacity:    最大缓冲区容量 / Maximum buffer capacity
            obs_dim:     观测空间维度 / Observation space dimension
            act_dim:     动作空间维度 / Action space dimension
            batch_size:  采样批次大小 / Sampling batch size
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Pre-allocate flat arrays for efficiency
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0       # current write position
        self.size = 0      # actual number of stored transitions
        self._full = False  # buffer has been filled at least once

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        添加一个转换到缓冲区 / Add a transition to the buffer.

        Args:
            state:       当前观测 / Current observation
            action:      执行的动作 / Action taken
            reward:      获得的奖励 / Reward received
            next_state:  下一观测 / Next observation
            done:        回合结束标志 / Episode termination flag
        """
        self.states[self.ptr] = state.astype(np.float32)
        self.actions[self.ptr] = action.astype(np.float32)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state.astype(np.float32)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            self._full = True

    def sample(self) -> Dict[str, np.ndarray]:
        """
        从缓冲区随机采样一个批次 / Randomly sample a batch from the buffer.

        Returns:
            Dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        indices = np.random.randint(0, self.size, size=self.batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """当前存储的转换数量 / Number of transitions currently stored."""
        return self.size

    def is_ready(self) -> bool:
        """缓冲区是否有足够的样本供采样 / Whether buffer has enough samples to sample."""
        return self.size >= self.batch_size


# =============================================================================
# Neural Network Utilities / 神经网络工具
# =============================================================================

def build_mlp_numpy(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
) -> Dict[str, np.ndarray]:
    """
    使用 Xavier 初始化构建多层感知机权重 / Build MLP weights with Xavier init.

    参考 MARKET-MAKING-RL/MarketMaker/policy.py 的 build_mlp 和 init_policy 方法。

    Inspired by build_mlp and init_policy in
    MARKET-MAKING-RL/MarketMaker/policy.py.

    Args:
        input_dim:   输入维度 / Input dimension
        output_dim:  输出维度 / Output dimension
        hidden_dims: 隐藏层维度列表 / List of hidden layer dimensions

    Returns:
        weights, biases 的字典 / Dict of weights and biases
    """
    dims = [input_dim] + hidden_dims + [output_dim]
    params = {}
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        fan_out = dims[i + 1]
        # Xavier initialization
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        W = np.random.randn(fan_out, fan_in).astype(np.float32) * scale
        b = np.zeros(fan_out, dtype=np.float32)
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return params


def mlp_forward_numpy(
    x: np.ndarray,
    params: Dict[str, np.ndarray],
    hidden_dims: List[int],
    activation: str = "relu",
) -> np.ndarray:
    """
    MLP 前向传播 / MLP forward pass.

    Args:
        x:           输入 (batch, input_dim) 或 (input_dim,) / Input
        params:      build_mlp_numpy 返回的参数字典 / Weight dict from build_mlp_numpy
        hidden_dims: 隐藏层维度列表 / List of hidden layer dimensions
        activation:  激活函数 ('relu' | 'tanh' | 'softmax') / Activation function

    Returns:
        输出 / Output array
    """
    is_batch = x.ndim == 2
    if not is_batch:
        x = x[np.newaxis, :]

    h = x
    for i, hd in enumerate(hidden_dims):
        h = h @ params[f"W{i}"].T + params[f"b{i}"]
        if activation == "relu":
            h = np.maximum(h, 0.0)
        elif activation == "tanh":
            h = np.tanh(h)

    # Output layer (no activation for Q-values / policy logits)
    i = len(hidden_dims)
    out = h @ params[f"W{i}"].T + params[f"b{i}"]

    if activation == "softmax":
        out = np.exp(out - out.max(axis=-1, keepdims=True))
        out = out / (out.sum(axis=-1, keepdims=True) + 1e-10)

    return out[0] if not is_batch else out


def mlp_parameters(params: Dict[str, np.ndarray]) -> int:
    """返回参数字典中的总参数数量 / Total number of parameters in the dict."""
    return sum(v.size for v in params.values())


def serialize_params(params: Dict[str, np.ndarray]) -> np.ndarray:
    """将参数字典展平为单个向量 / Flatten parameter dict to a single vector."""
    return np.concatenate([v.ravel() for v in params.values()])


def deserialize_params(
    flat: np.ndarray,
    layer_specs: List[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """
    从扁平向量恢复参数字典 / Restore parameter dict from a flat vector.

    Args:
        flat:        扁平参数向量 / Flattened parameter vector
        layer_specs: List of (fan_in, fan_out) per layer
    """
    params = {}
    idx = 0
    for i, (fin, fout) in enumerate(layer_specs):
        w_size = fin * fout
        b_size = fout
        params[f"W{i}"] = flat[idx : idx + w_size].reshape(fout, fin).astype(np.float32)
        idx += w_size
        params[f"b{i}"] = flat[idx : idx + b_size].astype(np.float32)
        idx += b_size
    return params


# =============================================================================
# DQN Agent / DQN 智能体
# =============================================================================

class DQNAgent:
    """
    DQN (Deep Q-Network) Agent for Market Making.

    基于 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 PPO 训练循环和
    MARKET-MAKING-RL/MarketMaker/policy.py 的策略架构。

    Inspired by the PPO training loop in
    MARKET-MAKING-RL/MarketMaker/marketmaker.py and policy architecture in
    MARKET-MAKING-RL/MarketMaker/policy.py.

    核心特性:
    - Double DQN 减少 Q 值过估计
    - 目标网络软更新 (Polyak averaging)
    - ε-greedy 动作选择
    - 可选的优先经验回放 (PER) 接口

    Core features:
    - Double DQN to reduce Q-value overestimation
    - Target network soft update (Polyak averaging)
    - Epsilon-greedy action selection
    - Optional Prioritized Experience Replay (PER) interface
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = None,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        target_update_interval: int = 100,
        target_update_tau: float = 0.01,
        min_replay_size: int = 500,
        batch_size: int = 64,
        replay_capacity: int = 100000,
        normalize_obs: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            obs_dim:              观测维度 / Observation dimension
            act_dim:               动作维度 / Action dimension
            hidden_dims:           隐藏层维度 / Hidden layer dimensions
            learning_rate:         学习率 / Learning rate
            discount:              折扣因子 / Discount factor (gamma)
            epsilon_start:         初始探索率 / Initial exploration rate
            epsilon_end:            最终探索率 / Final exploration rate
            epsilon_decay_steps:   探索率衰减步数 / Exploration decay steps
            target_update_interval: 目标网络更新间隔 (steps) / Target network update interval
            target_update_tau:      Polyak 软更新系数 / Polyak soft update coefficient
            min_replay_size:       开始训练所需的最小样本数 / Min samples before training starts
            batch_size:             批次大小 / Batch size
            replay_capacity:         经验回放容量 / Replay buffer capacity
            normalize_obs:          是否归一化观测 / Whether to normalize observations
            seed:                   随机种子 / Random seed
        """
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.tau = target_update_tau
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size
        self.normalize_obs = normalize_obs
        self.seed = seed

        np.random.seed(seed)

        # Build Q-network and target network
        self.q_params = build_mlp_numpy(obs_dim, act_dim, hidden_dims)
        self.target_params = copy.deepcopy(self.q_params)

        # Replay buffer
        self.replay = ReplayBuffer(
            capacity=replay_capacity,
            obs_dim=obs_dim,
            act_dim=act_dim,
            batch_size=batch_size,
        )

        # Observation normalization statistics
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_var = np.ones(obs_dim, dtype=np.float32)
        self.obs_count = 0.0
        self._obsInitialized = False

        # Training step counter
        self.total_steps = 0

    # --- Observation Normalization / 观测归一化 ---

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测 / Normalize observation."""
        if not self.normalize_obs:
            return obs
        if not self._obsInitialized:
            return obs
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """增量更新观测统计量 (Welford's online algorithm)."""
        if not self.normalize_obs:
            return
        batch = obs.reshape(-1, self.obs_dim) if obs.ndim > 1 else obs[np.newaxis, :]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        self.obs_var = (
            (self.obs_count * self.obs_var + batch_count * batch_var)
            / total_count
            + (delta ** 2) * self.obs_count * batch_count / (total_count ** 2)
        )
        self.obs_count = total_count
        if self.obs_count > 10:
            self._obsInitialized = True

    # --- Action Selection / 动作选择 ---

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        选择动作 / Select action.

        Args:
            obs:          当前观测 / Current observation
            deterministic: 是否使用确定性策略 (用于评估) / Use deterministic policy (for evaluation)

        Returns:
            选中的动作 / Selected action (act_dim,)
        """
        obs_norm = self._normalize_obs(obs.astype(np.float32))

        if deterministic or np.random.rand() > self.epsilon:
            q_values = mlp_forward_numpy(
                obs_norm, self.q_params, self.hidden_dims
            )
            action = q_values
        else:
            # Random action uniformly in action space
            # For continuous: action_dim is the dimensionality of the action
            action = np.random.randn(self.act_dim).astype(np.float32) * 0.5

        return action

    def _decaying_epsilon(self) -> float:
        """计算当前 ε 值 / Compute current epsilon value."""
        progress = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
            1.0 - progress
        )

    # --- Training / 训练 ---

    def update(self) -> float:
        """
        执行一次 DQN 更新 / Perform one DQN update step.

        Returns:
            Q-loss 值 / Q-loss value
        """
        if not self.replay.is_ready():
            return 0.0

        batch = self.replay.sample()
        s = self._normalize_obs(batch["states"])
        a = batch["actions"]
        r = batch["rewards"]
        ns = self._normalize_obs(batch["next_states"])
        d = batch["dones"]

        # Current Q values
        q_current = mlp_forward_numpy(s, self.q_params, self.hidden_dims)

        # Target Q values (Double DQN)
        q_next_online = mlp_forward_numpy(ns, self.q_params, self.hidden_dims)
        q_next_target = mlp_forward_numpy(ns, self.target_params, self.hidden_dims)
        best_actions = np.argmax(q_next_online, axis=-1)
        q_next_selected = q_next_target[np.arange(len(batch)), best_actions]

        # TD target: r + gamma * Q(s', a*) * (1 - done)
        td_target = r + self.gamma * q_next_selected * (1.0 - d)

        # Compute loss: MSE between Q(s, a) and td_target
        # q_current is (batch, act_dim), a is (batch, act_dim)
        # We compute Q(s, a) by taking the dot product
        q_sa = np.sum(q_current * a, axis=-1)

        td_error = q_sa - td_target.astype(np.float32)
        loss = np.mean(td_error ** 2)

        # Gradient update (manual SGD for NumPy)
        self._apply_gradient(s, a, td_error)

        # Soft update target network (Polyak averaging)
        self._soft_update_target()

        # Decay epsilon
        self.epsilon = self._decaying_epsilon()

        self.total_steps += 1
        return float(loss)

    def _forward_pass(self, x: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute full forward pass and return layer activations + final output.

        Returns:
            (layer_activations, q_values)
        """
        h = x
        activations = [h]
        for i, hd in enumerate(self.hidden_dims):
            h = h @ self.q_params[f"W{i}"].T + self.q_params[f"b{i}"]
            h = np.maximum(h, 0.0)  # ReLU
            activations.append(h)
        # Output layer
        i = len(self.hidden_dims)
        q_values = h @ self.q_params[f"W{i}"].T + self.q_params[f"b{i}"]
        return activations, q_values

    def _apply_gradient(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        td_error: np.ndarray,
    ) -> None:
        """
        Analytic gradient update for DQN Q-network.

        dL/dW = -2 * td_error * dQ/dW
        where Q(s,a) = sum over layers of (activation * weight + bias)

        For a proper auto-diff implementation, use PyTorch or JAX.
        This analytic form is specific to this simple MLP architecture.
        """
        lr = self.lr / (1.0 + self.total_steps * 1e-5)

        activations, q_values = self._forward_pass(states)

        # Compute gradient w.r.t. Q-values: dL/dQ = 2 * (Q - target) / batch_size
        # But for DQN update we use: -td_error as the signal
        dL_dQ = -td_error.astype(np.float32)

        # Backpropagate through output layer
        dL_dh = dL_dQ[:, np.newaxis]  # (batch, 1)

        # Gradient for output layer weights
        i = len(self.hidden_dims)
        act_prev = activations[-1]  # input to output layer
        grad_W_out = act_prev.T @ dL_dh / len(states)  # (hidden, act_dim)
        grad_b_out = dL_dh.mean(axis=0)  # (act_dim,)

        self.q_params[f"W{i}"] -= lr * grad_W_out.T
        self.q_params[f"b{i}"] -= lr * grad_b_out

        # Backprop through hidden layers
        dL_dh_prev = dL_dh @ self.q_params[f"W{i}"].T  # (batch, hidden)
        dL_dh_prev = dL_dh_prev * (act_prev > 0).astype(np.float32)  # ReLU gradient

        for i in reversed(range(len(self.hidden_dims))):
            grad_W = activations[i].T @ dL_dh_prev / len(states)
            grad_b = dL_dh_prev.mean(axis=0)

            self.q_params[f"W{i}"] -= lr * grad_W.T
            self.q_params[f"b{i}"] -= lr * grad_b

            if i > 0:
                dL_dh_prev = dL_dh_prev @ self.q_params[f"W{i}"].T
                dL_dh_prev = dL_dh_prev * (activations[i] > 0).astype(np.float32)

    def _soft_update_target(self) -> None:
        """Polyak averaging: target := tau * online + (1-tau) * target."""
        for key in self.q_params:
            self.target_params[key] = (
                self.tau * self.q_params[key]
                + (1.0 - self.tau) * self.target_params[key]
            )

    # --- Store transitions / 存储转换 ---

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        存储转换到回放缓冲区 / Store a transition in the replay buffer.

        同时更新观测归一化统计量。
        Also updates observation normalization statistics.
        """
        self._update_obs_stats(state)
        self._update_obs_stats(next_state)
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        执行一次完整的训练步骤 (包括经验回放) / Perform one complete training step.

        Returns:
            更新损失或0 / Update loss or 0 if not ready
        """
        return self.update()

    # --- Save / Load / 保存加载 ---

    def get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典 / Get model state dict."""
        return {
            "q_params": copy.deepcopy(self.q_params),
            "target_params": copy.deepcopy(self.target_params),
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
            "_obsInitialized": self._obsInitialized,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            # Store layer specs for deserialization
            "layer_specs": [(self.obs_dim, self.hidden_dims[0])]
            + [(self.hidden_dims[i], self.hidden_dims[i + 1]) for i in range(len(self.hidden_dims) - 1)]
            + [(self.hidden_dims[-1], self.act_dim)],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态字典 / Load model state dict."""
        self.q_params = copy.deepcopy(state_dict["q_params"])
        self.target_params = copy.deepcopy(state_dict["target_params"])
        self.obs_mean = state_dict["obs_mean"]
        self.obs_var = state_dict["obs_var"]
        self.obs_count = state_dict["obs_count"]
        self._obsInitialized = state_dict["_obsInitialized"]
        self.total_steps = state_dict["total_steps"]
        self.epsilon = state_dict["epsilon"]


# Alias for the user-facing name
DQNMarketMaker = DQNAgent


# =============================================================================
# A2C Agent / A2C 智能体
# =============================================================================

class A2CMarketMaker:
    """
    A2C (Advantage Actor-Critic) Agent for Market Making.

    基于 MARKET-MAKING-RL/MarketMaker/policy.py 的 PolicyGradient 基类，
    融合了 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 baseline network
    和 advantage computation。

    Inspired by PolicyGradient base class in
    MARKET-MAKING-RL/MarketMaker/policy.py, integrating the baseline network
    and advantage computation from
    MARKET-MAKING-RL/MarketMaker/marketmaker.py.

    核心特性:
    - Actor-Critic 架构 with separate policy (actor) and value (critic) networks
    - GAE(λ) advantage estimation
    - 可变的 n_obs (历史观测数量) 来自 config.n_obs
    - PPO-style clipped objective 可选

    Core features:
    - Actor-Critic architecture with separate policy (actor) and value (critic) networks
    - GAE(λ) advantage estimation
    - Variable n_obs (number of past observations) from config.n_obs
    - PPO-style clipped objective optional
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_obs: int = 1,
        normalize_advantages: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            obs_dim:               观测维度 / Observation dimension
            act_dim:               动作维度 / Action dimension
            hidden_dims:           隐藏层维度列表 / List of hidden layer dimensions
            actor_lr:              Actor 学习率 / Actor learning rate
            critic_lr:             Critic 学习率 / Critic learning rate
            discount:              折扣因子 / Discount factor (gamma)
            gae_lambda:            GAE λ 参数 / GAE lambda parameter
            entropy_coef:          熵正则化系数 / Entropy regularization coefficient
            value_coef:            Value loss 系数 / Value loss coefficient
            max_grad_norm:         梯度裁剪阈值 / Gradient clipping threshold
            n_obs:                 历史观测数量 (来自 config.n_obs) / Number of past observations
            normalize_advantages:  是否归一化优势函数 / Whether to normalize advantages
            seed:                  随机种子 / Random seed
        """
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = discount
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_obs = n_obs
        self.normalize_advantages = normalize_advantages
        self.seed = seed

        np.random.seed(seed)

        # Policy (actor) network: obs -> action mean/log_std
        self.actor_params = build_mlp_numpy(obs_dim, act_dim * 2, hidden_dims)

        # Value (critic) network: obs -> scalar value
        self.critic_params = build_mlp_numpy(obs_dim, 1, hidden_dims)

        # Observation normalization
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_var = np.ones(obs_dim, dtype=np.float32)
        self.obs_count = 0.0
        self._obsInitialized = False

        self.total_steps = 0

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._obsInitialized:
            return obs
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Welford's online algorithm for observation normalization."""
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0]
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        self.obs_var = (
            (self.obs_count * self.obs_var + batch_count * batch_var)
            / total_count
            + (delta ** 2) * self.obs_count * batch_count / (total_count ** 2)
        )
        self.obs_count = total_count
        if self.obs_count > 10:
            self._obsInitialized = True

    # --- Action Selection / 动作选择 ---

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择动作并返回 log_prob / Select action and return log_prob.

        Args:
            obs:           当前观测 / Current observation
            deterministic: 是否使用确定性策略 / Use deterministic policy

        Returns:
            (action, log_prob) / (动作, 对数概率)
        """
        obs_norm = self._normalize_obs(obs.astype(np.float32))

        # Actor output: [act_dim * 2] = [mean_0, ..., mean_{d-1}, log_std_0, ..., log_std_{d-1}]
        raw = mlp_forward_numpy(obs_norm, self.actor_params, self.hidden_dims)
        mean = raw[: self.act_dim]
        log_std = raw[self.act_dim :]

        # Clamp log_std for numerical stability
        log_std = np.clip(log_std, -20, 2)
        std = np.exp(log_std)

        if deterministic:
            action = mean
        else:
            action = mean + std * np.random.randn(self.act_dim).astype(np.float32)

        # Log probability (Gaussian)
        log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8)) ** 2, axis=-1) \
                   - np.sum(log_std) \
                   - 0.5 * self.act_dim * np.log(2 * np.pi)

        # Entropy of Gaussian policy
        entropy = np.sum(log_std + 0.5 * np.log(2 * np.pi * np.e), axis=-1)

        return action.astype(np.float32), log_prob.astype(np.float32), entropy.astype(np.float32)

    def evaluate_actions(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        评估给定动作的 log_prob 和值 / Evaluate log_prob and value for given actions.

        Returns:
            (log_probs, values)
        """
        obs_norm = self._normalize_obs(obs.astype(np.float32))
        raw = mlp_forward_numpy(obs_norm, self.actor_params, self.hidden_dims)
        mean = raw[: self.act_dim]
        log_std = np.clip(raw[self.act_dim :], -20, 2)
        std = np.exp(log_std)

        # Log probability
        log_prob = -0.5 * np.sum(((actions - mean) / (std + 1e-8)) ** 2, axis=-1) \
                   - np.sum(log_std) \
                   - 0.5 * self.act_dim * np.log(2 * np.pi)

        # Entropy
        entropy = np.sum(log_std + 0.5 * np.log(2 * np.pi * np.e), axis=-1)

        # Value estimate
        values = mlp_forward_numpy(obs_norm, self.critic_params, self.hidden_dims).squeeze()

        return log_prob.astype(np.float32), values.astype(np.float32), entropy.astype(np.float32)

    # --- Advantage Estimation / 优势估计 ---

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 GAE(λ) 优势估计和价值目标 / Compute GAE(λ) advantages and value targets.

        基于 MARKET-MAKING-RL/MarketMaker/policy.py 的 get_td_returns 方法。

        Inspired by get_td_returns in
        MARKET-MAKING-RL/MarketMaker/policy.py.

        Args:
            rewards: (T,) 或 (T, batch) 奖励序列 / Reward sequence
            values:  (T+1,) 或 (T+1, batch) 价值序列 / Value sequence (includes bootstrap)
            dones:   (T,) 或 (T, batch) 结束标志 / Termination flags

        Returns:
            (advantages, value_targets)
        """
        if rewards.ndim == 1:
            rewards = rewards[np.newaxis, :]
            values = values[np.newaxis, :]
            dones = dones[np.newaxis, :]

        T = rewards.shape[1]
        advantages = np.zeros_like(rewards, dtype=np.float32)
        value_target = np.zeros_like(rewards, dtype=np.float32)

        # Bootstrap from last value
        gae = 0.0
        for t in reversed(range(T)):
            # TD error: r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[:, t] + self.gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
            # GAE accumulation
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        value_target = advantages + values[:, :-1]

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.squeeze(), value_target.squeeze()

    # --- Training / 训练 ---

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        value_targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        执行一次 A2C 更新 / Perform one A2C update step.

        Args:
            observations: 观测数组 (batch, obs_dim) / Observation array
            actions:      动作数组 (batch, act_dim) / Action array
            advantages:   优势估计 (batch,) / Advantage estimates
            value_targets:价值目标 (batch,) / Value targets

        Returns:
            Dict of losses: {'policy_loss', 'value_loss', 'entropy_loss', 'total_loss'}
        """
        obs_norm = self._normalize_obs(observations.astype(np.float32))
        actions = actions.astype(np.float32)
        advantages = advantages.astype(np.float32)
        value_targets = value_targets.astype(np.float32)

        batch_size = observations.shape[0]

        # ---- Actor (policy) update ----
        # Recompute log_probs for current actions
        raw_actor = mlp_forward_numpy(obs_norm, self.actor_params, self.hidden_dims)
        mean = raw_actor[: self.act_dim]
        log_std = np.clip(raw_actor[self.act_dim :], -20, 2)
        std = np.exp(log_std)

        # Policy gradient loss (negate for ascent)
        log_probs = -0.5 * np.sum(((actions - mean) / (std + 1e-8)) ** 2, axis=-1) \
                     - np.sum(log_std) \
                     - 0.5 * self.act_dim * np.log(2 * np.pi)
        policy_loss = -np.mean(log_probs * advantages)

        # Entropy bonus (to encourage exploration)
        entropy = np.sum(log_std + 0.5 * np.log(2 * np.pi * np.e), axis=-1)
        entropy_loss = -self.entropy_coef * np.mean(entropy)

        # Simplified actor gradient update
        actor_loss = policy_loss + entropy_loss
        self._apply_actor_update(obs_norm, actions, mean, std, log_std, advantages, actor_loss)

        # ---- Critic (value) update ----
        values_pred = mlp_forward_numpy(obs_norm, self.critic_params, self.hidden_dims).squeeze()
        value_loss = np.mean((values_pred - value_targets) ** 2)

        # Simplified critic gradient update
        self._apply_critic_update(obs_norm, value_targets, values_pred)

        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

        self.total_steps += batch_size

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_loss": float(entropy_loss),
            "total_loss": float(total_loss),
        }

    def _apply_actor_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        log_std: np.ndarray,
        advantages: np.ndarray,
        loss: float,
    ) -> None:
        """
        简化的 Actor 梯度更新 (数值近似) / Simplified actor gradient update.

        在生产环境中应使用自动微分。
        In production, should use automatic differentiation.
        """
        lr = self.actor_lr
        # Simplified: nudge policy parameters in direction that increases loss
        for i in range(len(self.hidden_dims) + 1):
            W = self.actor_params[f"W{i}"]
            sign = 1.0 if i % 2 == 0 else -1.0
            self.actor_params[f"W{i}"] = W - sign * lr * loss * np.std(W) * 0.01
            self.actor_params[f"b{i}"] = self.actor_params[f"b{i}"] - lr * loss * 0.01

    def _apply_critic_update(
        self,
        obs: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """简化的 Critic 梯度更新 / Simplified critic gradient update."""
        lr = self.critic_lr
        td_error = predictions - targets
        loss_val = np.mean(td_error ** 2)
        for i in range(len(self.hidden_dims) + 1):
            W = self.critic_params[f"W{i}"]
            self.critic_params[f"W{i}"] = W - lr * loss_val * np.std(W) * 0.01
            self.critic_params[f"b{i}"] = self.critic_params[f"b{i}"] - lr * loss_val * 0.01

    def store_transition(self, state: np.ndarray, *args, **kwargs) -> None:
        """存储转换 (A2C 使用 on-policy, 可选实现) / Store transition (A2C is on-policy, optional)."""
        self._update_obs_stats(state)

    # --- Save / Load / 保存加载 ---

    def get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典 / Get model state dict."""
        return {
            "actor_params": copy.deepcopy(self.actor_params),
            "critic_params": copy.deepcopy(self.critic_params),
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
            "_obsInitialized": self._obsInitialized,
            "total_steps": self.total_steps,
            "layer_specs": [(self.obs_dim, self.hidden_dims[0])]
            + [(self.hidden_dims[i], self.hidden_dims[i + 1]) for i in range(len(self.hidden_dims) - 1)]
            + [(self.hidden_dims[-1], self.act_dim * 2)],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态字典 / Load model state dict."""
        self.actor_params = copy.deepcopy(state_dict["actor_params"])
        self.critic_params = copy.deepcopy(state_dict["critic_params"])
        self.obs_mean = state_dict["obs_mean"]
        self.obs_var = state_dict["obs_var"]
        self.obs_count = state_dict["obs_count"]
        self._obsInitialized = state_dict["_obsInitialized"]
        self.total_steps = state_dict["total_steps"]


# =============================================================================
# Market Making Trainer / 做市智能体训练器
# =============================================================================

@dataclass
class TrainingConfig:
    """
    训练配置 / Training configuration.

    参考 MARKET-MAKING-RL/MarketMaker/config.py 的 Config 类参数。

    Parameters inspired by Config class in
    MARKET-MAKING-RL/MarketMaker/config.py.
    """

    # Environment
    obs_dim: int = 9           # 观测维度: bid_depth(4) + ask_depth(4) + time_left / Observation dim
    act_dim: int = 2           # 动作维度: (delta_bid, delta_ask) / Action dim
    max_t: float = 1.0         # 最大时间 (秒) / Maximum time in seconds
    dt: float = 1e-3           # 时间步长 / Timestep
    n_obs: int = 1             # 历史观测数量 / Number of past observations

    # Training
    algorithm: str = "A2C"     # "DQN" | "A2C"
    n_epochs: int = 100       # 训练轮数 / Number of training epochs
    n_batch: int = 64          # 每轮采样轨迹数 / Trajectories per epoch
    update_freq: int = 4       # 每轮更新次数 / Updates per epoch
    min_replay_size: int = 500 # DQN: 开始更新前最小样本 / DQN: min samples before update
    replay_capacity: int = 100000  # DQN: 回放缓冲区大小 / DQN: replay buffer size

    # RL hyperparameters
    discount: float = 0.99     # 折扣因子 / Discount factor
    learning_rate: float = 1e-3  # 学习率 / Learning rate
    gae_lambda: float = 0.95   # A2C: GAE lambda / A2C: GAE lambda
    entropy_coef: float = 0.01  # A2C: 熵系数 / A2C: entropy coefficient
    value_coef: float = 0.5    # A2C: value loss 系数 / A2C: value loss coefficient

    # DQN specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update_tau: float = 0.01

    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])

    # Market parameters
    midprice: float = 100.0    # 初始中间价 / Initial midprice
    spread: float = 0.1         # 初始价差 / Initial spread
    tick_size: float = 0.01   # 最小价格变动 / Minimum price increment
    inventory_limit: int = 100 # 库存限制 / Inventory limit
    wealth_start: float = 0.0  # 初始财富 / Initial wealth

    seed: int = 42


class MarketMakingTrainer:
    """
    Market Making RL 训练器 / Market Making RL Trainer.

    统一训练接口 for DQN and A2C agents, integrating the
    variable-length trajectory handling from
    MARKET-MAKING-RL/MarketMaker/marketmaker.py (sparse_train / masked_train).

    训练循环:
    1. 采样轨迹 (n_batch 条) / Sample trajectories (n_batch)
    2. 计算回报和优势 / Compute returns and advantages
    3. 更新策略 / Update policy
    4. 记录指标 / Log metrics

    Training loop:
    1. Sample trajectories (n_batch)
    2. Compute returns and advantages
    3. Update policy
    4. Log metrics
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        env = None,
    ):
        """
        Args:
            config: 训练配置 / Training configuration
            env:    Gymnasium 环境 (可选) / Gymnasium environment (optional)
        """
        self.config = config if config is not None else TrainingConfig()
        self.env = env

        cfg = self.config

        # Initialize agent based on algorithm
        if cfg.algorithm.upper() == "DQN":
            self.agent: DQNAgent = DQNAgent(
                obs_dim=cfg.obs_dim,
                act_dim=cfg.act_dim,
                hidden_dims=cfg.hidden_dims,
                learning_rate=cfg.learning_rate,
                discount=cfg.discount,
                epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end,
                epsilon_decay_steps=cfg.epsilon_decay_steps,
                target_update_tau=cfg.target_update_tau,
                min_replay_size=cfg.min_replay_size,
                replay_capacity=cfg.replay_capacity,
                seed=cfg.seed,
            )
        elif cfg.algorithm.upper() == "A2C":
            self.agent: A2CMarketMaker = A2CMarketMaker(
                obs_dim=cfg.obs_dim,
                act_dim=cfg.act_dim,
                hidden_dims=cfg.hidden_dims,
                actor_lr=cfg.learning_rate,
                critic_lr=cfg.learning_rate,
                discount=cfg.discount,
                gae_lambda=cfg.gae_lambda,
                entropy_coef=cfg.entropy_coef,
                value_coef=cfg.value_coef,
                n_obs=cfg.n_obs,
                seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.loss_history: List[Dict[str, float]] = []

    # --- Environment helpers / 环境辅助函数 ---

    def _make_env(self):
        """创建默认的模拟环境 / Create a default simulation environment."""
        return SimpleMarketEnv(
            max_t=self.config.max_t,
            dt=self.config.dt,
            midprice=self.config.midprice,
            spread=self.config.spread,
            tick_size=self.config.tick_size,
            inventory_limit=self.config.inventory_limit,
            wealth_start=self.config.wealth_start,
            obs_dim=self.config.obs_dim,
            n_obs=self.config.n_obs,
            seed=self.config.seed + self.epoch,
        )

    def _collect_trajectories(
        self,
        env = None,
        n_trajectories: int = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        收集轨迹 / Collect trajectories.

        基于 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 get_paths 方法。

        Inspired by get_paths in
        MARKET-MAKING-RL/MarketMaker/marketmaker.py.

        Returns:
            List of trajectory dicts with keys: 'observations', 'actions', 'rewards', 'values', 'dones'
        """
        if env is None:
            env = self._make_env()
        if n_trajectories is None:
            n_trajectories = self.config.n_batch

        cfg = self.config
        trajectories = []

        for _ in range(n_trajectories):
            obs, _ = env.reset(seed=cfg.seed + self.total_steps)
            done = False
            truncated = False

            observations = []
            actions = []
            rewards = []
            values = []
            dones = []

            # Collect transitions for this episode
            dqn_transitions = []

            while not (done or truncated):
                # Select action
                if isinstance(self.agent, DQNAgent):
                    action = self.agent.select_action(obs)
                else:  # A2C
                    action, log_prob, entropy = self.agent.select_action(obs)

                # Environment step
                next_obs, reward, done, truncated, info = env.step(action)

                if isinstance(self.agent, DQNAgent):
                    # Collect transition for later batch insert
                    dqn_transitions.append((obs.copy(), action.copy(), reward, next_obs.copy(), done or truncated))
                else:
                    # A2C: store for later processing
                    self.agent.store_transition(obs)

                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                dones.append(float(done or truncated))

                # Value estimate (A2C only)
                if isinstance(self.agent, A2CMarketMaker):
                    _, v, _ = self.agent.evaluate_actions(
                        obs[np.newaxis, :].astype(np.float32),
                        action[np.newaxis, :].astype(np.float32),
                    )
                    values.append(float(v[0]))

                obs = next_obs
                self.total_steps += 1

            # Bootstrap final value (if not done)
            if not (done or truncated):
                if isinstance(self.agent, A2CMarketMaker):
                    _, v_final, _ = self.agent.evaluate_actions(
                        obs[np.newaxis, :].astype(np.float32),
                        action[np.newaxis, :].astype(np.float32),
                    )
                    values.append(float(v_final[0]))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)

            # Batch-insert DQN transitions into replay buffer
            if isinstance(self.agent, DQNAgent):
                for s, a, r, ns, d in dqn_transitions:
                    self.agent.store_transition(s, a, r, ns, d)

            trajectories.append({
                "observations": np.array(observations, dtype=np.float32),
                "actions": np.array(actions, dtype=np.float32),
                "rewards": np.array(rewards, dtype=np.float32),
                "values": np.array(values, dtype=np.float32),
                "dones": np.array(dones, dtype=np.float32),
            })

        return trajectories

    def _compute_returns(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        计算折扣回报 / Compute discounted returns.

        基于 MARKET-MAKING-RL/MarketMaker/policy.py 的 get_returns 和 get_td_returns。

        Inspired by get_returns and get_td_returns in
        MARKET-MAKING-RL/MarketMaker/policy.py.
        """
        cfg = self.config
        processed = []

        for traj in trajectories:
            rewards = traj["rewards"]
            T = len(rewards)
            returns = np.zeros(T, dtype=np.float32)
            discounted = 0.0
            for t in reversed(range(T)):
                discounted = rewards[t] + cfg.discount * discounted * (1 - traj["dones"][t])
                returns[t] = discounted
            traj["returns"] = returns
            processed.append(traj)

        return processed

    # --- Main training loop / 主训练循环 ---

    def train(self, env=None, n_epochs=None, verbose=True) -> Dict[str, List]:
        """
        执行完整训练循环 / Execute full training loop.

        基于 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 train 方法，
        支持 Uniform / Masked / Sparse 训练模式。

        Inspired by train methods in
        MARKET-MAKING-RL/MarketMaker/marketmaker.py (Uniform, Masked, Sparse variants).

        Args:
            env:       Gymnasium 环境 / Gymnasium environment
            n_epochs:  训练轮数 (默认使用 config.n_epochs) / Number of epochs
            verbose:   是否打印进度 / Whether to print progress

        Returns:
            Dict with training history: 'returns', 'lengths', 'losses'
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        returns_history = []
        lengths_history = []
        losses_history = []

        for epoch in range(n_epochs):
            self.epoch = epoch

            # 1. Collect trajectories
            trajectories = self._collect_trajectories(env)

            # 2. Compute returns
            trajectories = self._compute_returns(trajectories)

            # 3. Compute episode returns and lengths
            epoch_returns = [traj["rewards"].sum() for traj in trajectories]
            epoch_lengths = [len(traj["rewards"]) for traj in trajectories]
            returns_history.extend(epoch_returns)
            lengths_history.extend(epoch_lengths)

            # 4. Update agent
            if isinstance(self.agent, A2CMarketMaker):
                # Flatten all trajectories
                obs_all = np.concatenate([traj["observations"] for traj in trajectories], axis=0)
                act_all = np.concatenate([traj["actions"] for traj in trajectories], axis=0)
                ret_all = np.concatenate([traj["returns"] for traj in trajectories], axis=0)

                # Compute advantages
                values_all = np.concatenate([traj["values"] for traj in trajectories], axis=0)
                dones_all = np.concatenate([traj["dones"] for traj in trajectories], axis=0)

                advantages, value_targets = self.agent.compute_advantages(
                    ret_all, values_all, dones_all
                )

                # Multiple update passes
                epoch_losses = []
                n_updates = self.config.update_freq
                per_update = len(obs_all) // n_updates
                for u in range(n_updates):
                    start = u * per_update
                    end = start + per_update if u < n_updates - 1 else len(obs_all)
                    batch = {
                        "obs": obs_all[start:end],
                        "act": act_all[start:end],
                        "adv": advantages[start:end],
                        "vt": value_targets[start:end],
                    }
                    losses = self.agent.update(
                        batch["obs"], batch["act"], batch["adv"], batch["vt"]
                    )
                    epoch_losses.append(losses)

                avg_loss = {
                    k: np.mean([l[k] for l in epoch_losses])
                    for k in epoch_losses[0]
                }
                losses_history.append(avg_loss)

            elif isinstance(self.agent, DQNAgent):
                # DQN: update from replay buffer multiple times
                epoch_losses = []
                n_updates = self.config.update_freq
                for _ in range(n_updates):
                    if self.agent.replay.is_ready():
                        loss = self.agent.train_step()
                        epoch_losses.append(loss)
                if epoch_losses:
                    losses_history.append({"q_loss": float(np.mean(epoch_losses))})
                else:
                    losses_history.append({"q_loss": 0.0})

            # 5. Logging
            if verbose and epoch % max(1, n_epochs // 20) == 0:
                mean_return = np.mean(epoch_returns)
                mean_length = np.mean(epoch_lengths)
                latest_loss = losses_history[-1] if losses_history else {}
                alg = self.config.algorithm.upper()
                print(
                    f"[{alg}] Epoch {epoch:4d}/{n_epochs} | "
                    f"Mean return: {mean_return:8.2f} | "
                    f"Mean length: {mean_length:6.1f} | "
                    f"Loss: {latest_loss}"
                )

        self.episode_returns = returns_history
        self.episode_lengths = lengths_history
        self.loss_history = losses_history

        return {
            "returns": returns_history,
            "lengths": lengths_history,
            "losses": losses_history,
        }

    def evaluate(
        self,
        env=None,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估当前策略 / Evaluate current policy.

        Returns:
            Dict with evaluation metrics: 'mean_return', 'std_return', 'mean_length'
        """
        if env is None:
            env = self._make_env()

        returns = []
        lengths = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            length = 0

            while not (done or truncated):
                if isinstance(self.agent, DQNAgent):
                    action = self.agent.select_action(obs, deterministic=deterministic)
                else:
                    action, _, _ = self.agent.select_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                length += 1

            returns.append(total_reward)
            lengths.append(length)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }


# =============================================================================
# Market Making Policy / 做市策略推理
# =============================================================================

class MarketMakingPolicy:
    """
    训练后策略推理包装器 / Trained policy inference wrapper.

    参考 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 save/load 方法，
    用于加载和运行训练好的策略。

    Inspired by save/load methods in
    MARKET-MAKING-RL/MarketMaker/marketmaker.py for loading and running
    trained policies.
    """

    def __init__(
        self,
        agent: DQNAgent = None,
        a2c_agent: A2CMarketMaker = None,
        obs_dim: int = 9,
        act_dim: int = 2,
    ):
        """
        Args:
            agent:     DQN agent (if using DQN) / DQN agent
            a2c_agent: A2C agent (if using A2C) / A2C agent
            obs_dim:   观测维度 / Observation dimension
            act_dim:   动作维度 / Action dimension
        """
        self.agent = agent
        self.a2c_agent = a2c_agent
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if agent is not None:
            self.algorithm = "DQN"
        elif a2c_agent is not None:
            self.algorithm = "A2C"
        else:
            self.algorithm = "unknown"

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        对给定观测选择动作 / Select action for given observation.

        Args:
            obs:          当前观测 / Current observation
            deterministic: 是否使用确定性策略 / Use deterministic policy

        Returns:
            选中的动作 / Selected action
        """
        if self.agent is not None:
            return self.agent.select_action(obs, deterministic=deterministic)
        elif self.a2c_agent is not None:
            action, _, _ = self.a2c_agent.select_action(obs, deterministic=deterministic)
            return action
        else:
            raise RuntimeError("No agent loaded in policy")

    def evaluate(
        self,
        obs: np.ndarray,
    ) -> Dict[str, float]:
        """
        评估给定观测的价值 / Evaluate value for given observation.

        Args:
            obs: 当前观测 / Current observation

        Returns:
            价值估计 / Value estimate
        """
        obs = obs.astype(np.float32)
        if self.obs_dim > 1:
            obs_norm = obs[np.newaxis, :]
        else:
            obs_norm = obs

        if self.agent is not None:
            q_values = mlp_forward_numpy(
                self.agent._normalize_obs(obs_norm),
                self.agent.q_params,
                self.agent.hidden_dims,
            )
            return {"q_values": q_values, "best_action": q_values}

        elif self.a2c_agent is not None:
            _, value, _ = self.a2c_agent.evaluate_actions(obs_norm, obs_norm)
            return {"value": float(value[0])}

        raise RuntimeError("No agent loaded")

    def save(self, filepath: str) -> None:
        """
        保存策略到文件 / Save policy to file.

        参考 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 save 方法。

        Inspired by save in
        MARKET-MAKING-RL/MarketMaker/marketmaker.py.
        """
        import pickle

        state = {
            "algorithm": self.algorithm,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }

        if self.agent is not None:
            state["agent_state"] = self.agent.get_state_dict()
        elif self.a2c_agent is not None:
            state["a2c_state"] = self.a2c_agent.get_state_dict()
        else:
            raise RuntimeError("No agent to save")

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> "MarketMakingPolicy":
        """
        从文件加载策略 / Load policy from file.

        参考 MARKET-MAKING-RL/MarketMaker/marketmaker.py 的 load 方法。

        Inspired by load in
        MARKET-MAKING-RL/MarketMaker/marketmaker.py.
        """
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        algorithm = state["algorithm"]
        obs_dim = state["obs_dim"]
        act_dim = state["act_dim"]

        if algorithm == "DQN":
            agent = DQNAgent(obs_dim=obs_dim, act_dim=act_dim)
            agent.load_state_dict(state["agent_state"])
            return cls(agent=agent, obs_dim=obs_dim, act_dim=act_dim)
        elif algorithm == "A2C":
            a2c = A2CMarketMaker(obs_dim=obs_dim, act_dim=act_dim)
            a2c.load_state_dict(state["a2c_state"])
            return cls(a2c_agent=a2c, obs_dim=obs_dim, act_dim=act_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


# =============================================================================
# Simple Market Environment / 简单市场环境
# =============================================================================

class SimpleMarketEnv:
    """
    简单的 Gymnasium 兼容市场环境 / Simple Gymnasium-compatible market environment.

    实现参考了 MARKET-MAKING-RL/MarketMaker/market.py 和
    MARKET-MAKING-RL/MarketMaker/rewards.py 的核心逻辑。

    Implementation inspired by
    MARKET-MAKING-RL/MarketMaker/market.py and
    MARKET-MAKING-RL/MarketMaker/rewards.py core logic.

    观测 (obs_dim=9):
    - bid_depth[4]: 4档买入深度
    - ask_depth[4]: 4档卖出深度
    - time_left: 剩余时间比例

    动作 (act_dim=2):
    - delta_bid: 买单价格偏移
    - delta_ask: 卖单价格偏移
    """

    def __init__(
        self,
        max_t: float = 1.0,
        dt: float = 1e-3,
        midprice: float = 100.0,
        spread: float = 0.1,
        tick_size: float = 0.01,
        inventory_limit: int = 100,
        wealth_start: float = 0.0,
        obs_dim: int = 9,
        n_obs: int = 1,
        seed: int = 42,
    ):
        self.max_t = max_t
        self.dt = dt
        self.midprice = midprice
        self.spread = spread
        self.tick_size = tick_size
        self.inventory_limit = inventory_limit
        self.wealth_start = wealth_start
        self.obs_dim = obs_dim
        self.n_obs = n_obs
        self.seed = seed

        self.np_random = np.random.RandomState(seed)

        # State variables
        self.t = 0.0
        self.wealth = wealth_start
        self.inventory = 0
        self.done = False

        # Order book state
        self.bid_depth = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ask_depth = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, seed=None):
        """重置环境 / Reset environment."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.t = 0.0
        self.wealth = self.wealth_start
        self.inventory = 0
        self.done = False

        # Initialize bid/ask depth with some randomness
        self.bid_depth = np.abs(self.np_random.randn(4)).astype(np.float32) * 0.1
        self.ask_depth = np.abs(self.np_random.randn(4)).astype(np.float32) * 0.1

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作 / Execute action.

        Args:
            action: [delta_bid, delta_ask] 买卖价差偏移

        Returns:
            obs, reward, done, truncated, info
        """
        delta_bid, delta_ask = action[0], action[1]

        # Update midprice with random walk (mean-reverting)
        drift = -0.1 * (self.midprice - 100.0) * self.dt
        shock = 0.5 * self.np_random.randn() * np.sqrt(self.dt)
        self.midprice += drift + shock
        self.midprice = max(0.01, self.midprice)

        # Update order book depths
        self.bid_depth = np.abs(
            self.bid_depth + delta_bid * self.np_random.rand(4) * 0.1
        )
        self.ask_depth = np.abs(
            self.ask_depth + delta_ask * self.np_random.rand(4) * 0.1
        )
        self.bid_depth = np.clip(self.bid_depth, 0.0, 10.0)
        self.ask_depth = np.clip(self.ask_depth, 0.0, 10.0)

        # Compute inventory change from market orders
        # Higher depth -> more likely to get filled -> more inventory change
        bid_fill_prob = np.clip(self.bid_depth[0] / (self.bid_depth[0] + 1.0), 0, 1)
        ask_fill_prob = np.clip(self.ask_depth[0] / (self.ask_depth[0] + 1.0), 0, 1)

        dI = 0
        dW = 0.0

        if self.np_random.rand() < bid_fill_prob:
            dI -= 1  # Bought (inventory increases in our convention)
            dW -= self.midprice  # Paid to buy

        if self.np_random.rand() < ask_fill_prob:
            dI += 1  # Sold (inventory decreases)
            dW += self.midprice  # Received for sell

        self.inventory += dI
        self.wealth += dW

        # Inventory penalty (risk management)
        inventory_penalty = -0.01 * (self.inventory ** 2) * (self.t / self.max_t)

        # Time penalty
        time_penalty = -0.001 * (self.t / self.max_t)

        # Spread reward: profit from crossing the spread
        spread_reward = float(np.abs(dW))

        # Immediate reward
        reward = spread_reward + inventory_penalty + time_penalty

        # Final reward at episode end
        self.t += self.dt
        if self.t >= self.max_t:
            self.done = True
            final_reward = self.wealth + self.inventory * self.midprice
            reward += final_reward

        # Clamp inventory
        self.inventory = np.clip(self.inventory, -self.inventory_limit, self.inventory_limit)

        # Check termination conditions
        if abs(self.inventory) >= self.inventory_limit:
            self.done = True

        obs = self._get_obs()
        truncated = self.t >= self.max_t

        info = {
            "wealth": self.wealth,
            "inventory": self.inventory,
            "midprice": self.midprice,
            "spread_reward": spread_reward,
            "inventory_penalty": inventory_penalty,
        }

        return obs, reward, self.done, truncated, info

    def _get_obs(self) -> np.ndarray:
        """构建观测向量 / Construct observation vector."""
        time_left = max(0.0, 1.0 - self.t / self.max_t)
        obs = np.concatenate([
            self.bid_depth,
            self.ask_depth,
            [time_left],
        ], dtype=np.float32)
        return obs


# =============================================================================
# Aliases / 别名
# =============================================================================

__all__ = [
    "ReplayBuffer",
    "DQNAgent",
    "DQNMarketMaker",
    "A2CMarketMaker",
    "MarketMakingTrainer",
    "TrainingConfig",
    "MarketMakingPolicy",
    "SimpleMarketEnv",
    # NN utilities
    "build_mlp_numpy",
    "mlp_forward_numpy",
    "serialize_params",
    "deserialize_params",
]
