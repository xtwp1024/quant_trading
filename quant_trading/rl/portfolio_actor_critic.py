"""
Portfolio Actor-Critic RL Agents / 投资组合Actor-Critic强化学习智能体

基于 Portfolio-Management-ActorCriticRL 项目重构，纯 NumPy + Gymnasium 实现，无需 PyTorch。

Pure NumPy implementation of A2C (Advantage Actor-Critic) and DDPG (Deep Deterministic
Policy Gradient) for portfolio management. No autograd libraries required.

Classes / 类:
    - A2CPortfolioAgent: Advantage Actor-Critic portfolio agent
    - DDPGPortfolioAgent: Deep Deterministic Policy Gradient agent
    - PortfolioActor: Neural network actor for policy approximation
    - PortfolioCritic: Neural network critic for value estimation
    - MultiAssetEnv: Gymnasium environment for multi-asset trading

Usage / 用法:
    from quant_trading.rl.portfolio_actor_critic import (
        A2CPortfolioAgent, DDPGPortfolioAgent, PortfolioActor,
        PortfolioCritic, MultiAssetEnv, ReplayBuffer, OUActionNoise
    )

    # A2C Example
    agent = A2CPortfolioAgent(state_dim=30, action_dim=5, n_stocks=5)
    action = agent.select_action(state)
    agent.update(states, actions, rewards, done)

    # DDPG Example
    agent = DDPGPortfolioAgent(state_dim=30, action_dim=5)
    action = agent.select_action(state)
    agent.remember(state, action, reward, next_state, done)
    agent.learn()
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import copy

# Lazy import for gymnasium - graceful degradation
GYMNASIUM_AVAILABLE = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None


# ============================================================================
# Neural Network Base Classes / 神经网络基类 (Pure NumPy)
# ============================================================================

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot initialization / Xavier初始化"""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization for ReLU / He初始化"""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def zero_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Zero initialization for output layers / 零初始化"""
    return np.zeros((fan_in, fan_out))


class NeuralNetwork:
    """
    Pure NumPy MLP Neural Network / 纯NumPy多层感知机

    Implements forward pass and backward pass (gradient descent) from scratch.
    无需autograd库的手工前向传播和反向传播实现。
    """

    def __init__(
        self,
        layer_dims: List[int],
        activations: Optional[List[str]] = None,
        weight_init: str = 'xavier',
        learning_rate: float = 1e-3
    ):
        """
        Initialize neural network / 初始化神经网络

        Args:
            layer_dims: List of layer dimensions, e.g., [input, hidden1, hidden2, output]
            activations: List of activation functions for each layer
            weight_init: Weight initialization method ('xavier', 'he', 'zero')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        self.learning_rate = learning_rate

        # Default activations: ReLU for hidden layers, identity for output
        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['none']

        self.activations = activations

        # Initialize weights and biases / 初始化权重和偏置
        self.weights = []
        self.biases = []

        init_funcs = {
            'xavier': xavier_init,
            'he': he_init,
            'zero': zero_init
        }
        init_func = init_funcs.get(weight_init, xavier_init)

        for i in range(self.n_layers):
            w = init_func(layer_dims[i], layer_dims[i + 1])
            b = np.zeros(layer_dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def _activate(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function / 应用激活函数"""
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif activation == 'softplus':
            return np.log(1 + np.exp(np.clip(z, -500, 500)))
        elif activation == 'none' or activation == 'linear':
            return z
        else:
            return z

    def _activation_derivative(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function / 激活函数导数"""
        if activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif activation == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif activation == 'softplus':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif activation == 'none' or activation == 'linear':
            return np.ones_like(z)
        else:
            return np.ones_like(z)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass / 前向传播

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            output: Output array of shape (batch_size, output_dim)
            cache: List of (z, a) tuples for backprop
        """
        cache = []
        a = x

        for i in range(self.n_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z, self.activations[i])
            cache.append((z, a))

        return a, cache

    def backward(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        cache: List[Tuple[np.ndarray, np.ndarray]],
        loss_type: str = 'mse'
    ) -> None:
        """
        Backward pass with gradient descent / 反向传播与梯度下降

        Args:
            x: Input data
            y_true: True labels / targets
            cache: Cache from forward pass
            loss_type: 'mse' or 'cross_entropy'
        """
        m = x.shape[0]
        n = self.n_layers

        # Compute output gradient based on loss type
        y_pred = cache[-1][1]

        if loss_type == 'mse':
            # MSE loss: (1/2m) * sum((y_pred - y_true)^2)
            delta = y_pred - y_true
        elif loss_type == 'cross_entropy':
            # Cross-entropy loss with softmax
            # dL/dz = y_pred - y_true
            delta = y_pred - y_true
        else:
            delta = y_pred - y_true

        # Backprop through layers
        for i in range(n - 1, -1, -1):
            z, a = cache[i]
            dz = delta

            # Compute gradients
            if i == 0:
                da_prev = np.dot(dz, self.weights[i].T)
            else:
                da_prev = np.dot(delta, self.weights[i].T)

            # For first layer, use input x; otherwise use previous activation
            if i == 0:
                dw = np.dot(x.T, dz) / m
            else:
                dw = np.dot(cache[i - 1][1].T, dz) / m

            db = np.sum(dz, axis=0) / m

            # Update weights and biases
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

            # Propagate delta through activation derivative
            if i > 0:
                delta = da_prev * self._activation_derivative(z, self.activations[i])


# ============================================================================
# Policy Gradient Layer for A2C / A2C策略梯度层
# ============================================================================

class PolicyGradientLayer:
    """
    Policy Gradient computation for A2C / A2C策略梯度计算

    Computes advantage-weighted policy gradient loss and value-based critic loss.
    实现基于优势函数的策略梯度损失和价值Critic损失。
    """

    def __init__(self, actor_network: NeuralNetwork, critic_network: NeuralNetwork,
                 gamma: float = 0.99, entropy_coef: float = 0.01):
        """
        Initialize policy gradient layer / 初始化策略梯度层

        Args:
            actor_network: Neural network for policy (actor)
            critic_network: Neural network for value estimation
            gamma: Discount factor
            entropy_coef: Entropy bonus coefficient for exploration
        """
        self.actor = actor_network
        self.critic = critic_network
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def compute_advantage(
        self,
        rewards: List[float],
        values: List[float],
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantage using GAE (Generalized Advantage Estimation) / GAE优势估计

        A(t) = delta(t) + gamma * lambda * A(t+1)
        where delta(t) = r(t) + gamma * V(s(t+1)) - V(s(t))

        Args:
            rewards: List of rewards
            values: List of value estimates
            done: Whether episode is finished

        Returns:
            advantages: Computed advantage values
            returns: Computed return values (for critic training)
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        # Compute TD errors / 计算TD误差
        next_value = 0.0 if done else values[-1]
        cumulative = 0.0

        # GAE backwards computation
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = 0.0 if done else values[t]
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val - values[t]
            cumulative = delta + self.gamma * cumulative
            advantages[t] = cumulative

        # Returns = advantages + values (for semi-gradient TD)
        returns = advantages + np.array(values)

        return advantages, returns

    def policy_gradient_loss(
        self,
        actions: np.ndarray,
        log_probs: np.ndarray,
        advantages: np.ndarray
    ) -> float:
        """
        Compute policy gradient loss / 策略梯度损失

        L_policy = -E[log_prob(a|s) * A(s,a)]

        Args:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            advantages: Advantage estimates

        Returns:
            Policy loss value
        """
        # Policy loss is negative because we want to maximize
        # 策略损失取负因为我们要最大化
        loss = -np.mean(log_probs * advantages)
        return loss

    def entropy_bonus(self, probs: np.ndarray) -> float:
        """
        Compute entropy bonus for exploration / 熵奖励以鼓励探索

        H(pi) = -sum(pi * log(pi))
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
        return np.mean(entropy)


# ============================================================================
# Replay Buffer / 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer for DDPG / DDPG经验回放缓冲区

    Stores transitions (s, a, r, s', done) and samples batches for training.
    存储转移样本并支持批量采样训练。
    """

    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        """
        Initialize replay buffer / 初始化回放缓冲区

        Args:
            max_size: Maximum buffer capacity
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mem_cntr = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition / 存储转移样本"""
        index = self.mem_cntr % self.max_size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = 1.0 if done else 0.0

        self.mem_cntr += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch / 随机采样批量数据

        Args:
            batch_size: Size of batch to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        return (
            self.states[batch],
            self.actions[batch],
            self.rewards[batch],
            self.next_states[batch],
            self.dones[batch]
        )

    def clear(self) -> None:
        """Clear the buffer / 清空缓冲区"""
        self.mem_cntr = 0


# ============================================================================
# Ornstein-Uhlenbeck Noise / OU噪声
# ============================================================================

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for exploration / 探索用OU噪声

    Used in DDPG to add temporally correlated exploration noise.
    用于DDPG中添加时间相关的探索噪声。
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: float = 0.15,
        theta: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None
    ):
        """
        Initialize OU noise / 初始化OU噪声

        Args:
            mu: Mean of the noise process
            sigma: Volatility parameter
            theta: Mean reversion speed
            dt: Time step
            x0: Initial state
        """
        self.theta = theta
        self.mu = mu.astype(np.float32)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(mu)
        self.x_prev = self.x0.copy()

    def __call__(self) -> np.ndarray:
        """
        Sample from OU process / 采样OU过程

        dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0,1)
        """
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x.astype(np.float32)

    def reset(self) -> None:
        """Reset to initial state / 重置到初始状态"""
        self.x_prev = self.x0.copy()


# ============================================================================
# Portfolio Actor / 投资组合Actor网络
# ============================================================================

class PortfolioActor:
    """
    Portfolio Policy Network (Actor) / 投资组合策略网络

    Pure NumPy implementation of actor network for portfolio management.
    Outputs portfolio weights (action probabilities) given state.
    纯NumPy实现的策略网络，输入状态输出动作(组合权重)。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        entropy_coef: float = 1e-4
    ):
        """
        Initialize portfolio actor / 初始化投资组合Actor

        Args:
            state_dim: Dimension of state space
            action_dim: Number of assets + 1 (for cash)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            entropy_coef: Entropy coefficient for exploration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef

        # Build network architecture
        layer_dims = [state_dim] + hidden_dims + [action_dim]
        activations = ['relu'] * (len(layer_dims) - 2) + ['softmax']

        self.network = NeuralNetwork(
            layer_dims=layer_dims,
            activations=activations,
            weight_init='xavier',
            learning_rate=learning_rate
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute action probabilities / 前向传播计算动作概率

        Args:
            state: State array of shape (batch_size, state_dim)

        Returns:
            Action probabilities of shape (batch_size, action_dim)
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        probs, _ = self.network.forward(state)
        return probs

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action based on policy / 根据策略选择动作

        Args:
            state: Current state
            deterministic: If True, return argmax; else sample

        Returns:
            Selected action (portfolio weights)
        """
        probs = self.forward(state)

        if deterministic:
            action = probs[0]
        else:
            # Sample from categorical distribution
            action = np.random.choice(self.action_dim, p=probs[0])
            # Convert to one-hot then to weights
            one_hot = np.zeros(self.action_dim)
            one_hot[action] = 1.0
            action = one_hot

        return action

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """
        Update actor using policy gradient / 策略梯度更新Actor

        Args:
            states: Batch of states
            actions: Batch of actions taken
            advantages: Computed advantages

        Returns:
            Dictionary with loss values
        """
        probs, cache = self.network.forward(states)

        # Compute log probabilities of taken actions
        # For softmax, dL/dz = prob - one_hot(action)
        log_probs = np.log(probs + 1e-8)
        action_log_probs = np.sum(log_probs * actions, axis=1, keepdims=True)

        # Policy gradient loss
        policy_loss = -np.mean(action_log_probs * advantages.reshape(-1, 1))

        # Entropy bonus
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        entropy_loss = -self.entropy_coef * np.mean(entropy)

        # Total loss
        total_loss = policy_loss + entropy_loss

        # Backpropagate (simplified for policy gradient)
        # delta = -advantages * (1 - probs) for each action
        delta = -advantages.reshape(-1, 1) * (actions - probs)

        # Backward pass through network
        m = states.shape[0]
        for i in range(self.network.n_layers - 1, -1, -1):
            z, a = cache[i]

            if i == 0:
                da_prev = np.dot(delta, self.network.weights[i].T)
            else:
                da_prev = np.dot(delta, self.network.weights[i].T)

            dw = np.dot(states.T if i == 0 else cache[i - 1][1].T, delta) / m
            db = np.sum(delta, axis=0) / m

            # Compute delta for previous layer
            if i > 0:
                act_deriv = self.network._activation_derivative(z, self.network.activations[i])
                delta = da_prev * act_deriv

            self.network.weights[i] -= self.network.learning_rate * dw
            self.network.biases[i] -= self.network.learning_rate * db

        return {
            'policy_loss': float(policy_loss),
            'entropy_loss': float(entropy_loss),
            'total_loss': float(total_loss)
        }


# ============================================================================
# Portfolio Critic / 投资组合Critic网络
# ============================================================================

class PortfolioCritic:
    """
    Portfolio Value Network (Critic) / 投资组合价值网络

    Pure NumPy implementation of critic network for state-value estimation.
    Estimates V(s) - the expected return from a given state.
    纯NumPy实现的价值网络，估计状态价值V(s)。
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3
    ):
        """
        Initialize portfolio critic / 初始化投资组合Critic

        Args:
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        self.state_dim = state_dim

        # Build network architecture - outputs scalar value
        layer_dims = [state_dim] + hidden_dims + [1]
        activations = ['relu'] * (len(layer_dims) - 2) + ['none']

        self.network = NeuralNetwork(
            layer_dims=layer_dims,
            activations=activations,
            weight_init='xavier',
            learning_rate=learning_rate
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute state value / 前向传播计算状态价值

        Args:
            state: State array of shape (batch_size, state_dim)

        Returns:
            State values of shape (batch_size, 1)
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        values, _ = self.network.forward(state)
        return values

    def update(
        self,
        states: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Update critic using MSE loss / MSE损失更新Critic

        L_critic = E[(V(s) - R)^2]

        Args:
            states: Batch of states
            returns: Target return values

        Returns:
            Dictionary with loss value
        """
        values = self.forward(states)
        returns = returns.reshape(-1, 1)

        # MSE loss
        loss = np.mean((values - returns) ** 2)

        # Backward pass
        delta = 2 * (values - returns) / states.shape[0]

        cache = []
        a = states
        for i in range(self.network.n_layers):
            z = np.dot(a, self.network.weights[i]) + self.network.biases[i]
            a = self.network._activate(z, self.network.activations[i])
            cache.append((z, a))

        m = states.shape[0]
        for i in range(self.network.n_layers - 1, -1, -1):
            z, a = cache[i]

            if i == 0:
                da_prev = np.dot(delta, self.network.weights[i].T)
            else:
                da_prev = np.dot(delta, self.network.weights[i].T)

            dw = np.dot(states.T if i == 0 else cache[i - 1][1].T, delta) / m
            db = np.sum(delta, axis=0) / m

            if i > 0:
                act_deriv = self.network._activation_derivative(z, self.network.activations[i])
                delta = da_prev * act_deriv

            self.network.weights[i] -= self.network.learning_rate * dw
            self.network.biases[i] -= self.network.learning_rate * db

        return {'critic_loss': float(loss), 'value': float(np.mean(values))}


# ============================================================================
# A2C Portfolio Agent / A2C投资组合智能体
# ============================================================================

class A2CPortfolioAgent:
    """
    A2C (Advantage Actor-Critic) Portfolio Agent / A2C优势Actor-Critic投资组合智能体

    Combines policy gradient (actor) with value function approximation (critic).
    Uses synchronous updates with advantage estimation for stable training.
    结合策略梯度(Actor)和价值函数近似(Critic)，使用优势估计实现稳定训练。

    Attributes:
        actor: PortfolioActor network
        critic: PortfolioCritic network
        gamma: Discount factor
        entropy_coef: Entropy bonus coefficient
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_stocks: int = 5,
        hidden_dims: List[int] = [128, 128],
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 1e-4,
        gae_lambda: float = 0.95
    ):
        """
        Initialize A2C portfolio agent / 初始化A2C投资组合智能体

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (n_stocks + cash option)
            n_stocks: Number of stocks in portfolio
            hidden_dims: Hidden layer dimensions for networks
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            gae_lambda: GAE lambda parameter for advantage estimation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_stocks = n_stocks
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

        # Initialize actor and critic networks
        self.actor = PortfolioActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=actor_lr,
            entropy_coef=entropy_coef
        )

        self.critic = PortfolioCritic(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_lr
        )

        # Memory for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using current policy / 使用当前策略选择动作

        Args:
            state: Current state
            deterministic: If True, return argmax; else sample

        Returns:
            Selected action (portfolio weights)
        """
        return self.actor.select_action(state, deterministic)

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """
        Store transition in memory / 存储转移样本到记忆

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate from critic
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation / 广义优势估计

        Returns:
            advantages: GAE advantage estimates
            returns: Computed returns for critic training
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=bool)

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Update actor and critic networks / 更新Actor和Critic网络

        Performs one update step using collected trajectories.
        使用收集的轨迹执行一次更新。

        Returns:
            Dictionary with training metrics
        """
        if len(self.states) == 0:
            return {'total_loss': 0.0}

        # Convert to arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Normalize advantages (optional but often helps)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update critic
        critic_metrics = self.critic.update(states, returns)

        # Update actor
        actor_metrics = self.actor.update(states, actions, advantages)

        # Clear memory
        self.clear_memory()

        return {
            'actor_loss': actor_metrics.get('total_loss', 0.0),
            'critic_loss': critic_metrics.get('critic_loss', 0.0),
            'value': critic_metrics.get('value', 0.0),
            'entropy': actor_metrics.get('entropy_loss', 0.0)
        }

    def clear_memory(self) -> None:
        """Clear trajectory memory / 清空轨迹记忆"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []


# ============================================================================
# DDPG Portfolio Agent / DDPG投资组合智能体
# ============================================================================

class DDPGPortfolioAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) Portfolio Agent / DDPG深度确定性策略梯度智能体

    Actor-Critic algorithm for continuous action spaces with target networks.
    适用于连续动作空间的Actor-Critic算法，带有目标网络。

    Attributes:
        actor: Policy network (deterministic)
        critic: Q-value network
        target_actor: Target policy network
        target_critic: Target Q-value network
        noise: Ornstein-Uhlenbeck exploration noise
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,
        buffer_size: int = 100000,
        batch_size: int = 64
    ):
        """
        Initialize DDPG portfolio agent / 初始化DDPG投资组合智能体

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter for target networks
            buffer_size: Replay buffer size
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize actor and critic networks
        self.actor = PortfolioActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=actor_lr,
            entropy_coef=0.0  # DDPG doesn't use entropy
        )

        self.critic = PortfolioCritic(
            state_dim=state_dim + action_dim,  # Critic takes state+action
            hidden_dims=hidden_dims,
            learning_rate=critic_lr
        )

        # Target networks (copy of actor/critic)
        self.target_actor = PortfolioActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=actor_lr
        )
        self.target_critic = PortfolioCritic(
            state_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_lr
        )

        # Copy weights to target networks
        self._update_target_networks(tau=1.0)

        # OU noise for exploration
        self.noise = OUActionNoise(mu=np.zeros(action_dim))

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def _update_target_networks(self, tau: Optional[float] = None) -> None:
        """
        Soft update of target networks / 软更新目标网络

        theta_target = tau * theta_online + (1 - tau) * theta_target
        """
        if tau is None:
            tau = self.tau

        for i in range(len(self.actor.network.weights)):
            self.target_actor.network.weights[i] = (
                tau * self.actor.network.weights[i] +
                (1 - tau) * self.target_actor.network.weights[i]
            )
            self.target_actor.network.biases[i] = (
                tau * self.actor.network.biases[i] +
                (1 - tau) * self.target_actor.network.biases[i]
            )

        for i in range(len(self.critic.network.weights)):
            self.target_critic.network.weights[i] = (
                tau * self.critic.network.weights[i] +
                (1 - tau) * self.target_critic.network.weights[i]
            )
            self.target_critic.network.biases[i] = (
                tau * self.critic.network.biases[i] +
                (1 - tau) * self.target_critic.network.biases[i]
            )

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using deterministic policy / 确定策略选择动作

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            Selected action
        """
        action = self.actor.forward(state)[0]  # Take mean action

        if add_noise:
            action += self.noise()

        # Clip action to valid range [0, 1] for portfolio weights
        action = np.clip(action, 0, 1)

        # Normalize to sum to 1 (portfolio weights)
        if np.sum(action) > 0:
            action = action / np.sum(action)

        return action.astype(np.float32)

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in replay buffer / 存储转移样本到回放缓冲区
        """
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        """
        Update actor and critic networks / 更新Actor和Critic网络

        Samples from replay buffer and performs DDPG updates.
        从回放缓冲区采样并执行DDPG更新。

        Returns:
            Dictionary with training metrics
        """
        if self.replay_buffer.mem_cntr < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Get target action from target actor
        target_actions = self.target_actor.forward(next_states)

        # Compute target Q values
        state_action_next = np.hstack([next_states, target_actions])
        target_q = self.target_critic.forward(state_action_next)

        # Compute TD target: r + gamma * Q(s', a') * (1 - done)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        td_target = rewards + self.gamma * target_q * (1 - dones)

        # Update critic
        state_action = np.hstack([states, actions])
        current_q = self.critic.forward(state_action)

        # Critic loss (MSE)
        critic_loss = np.mean((current_q - td_target) ** 2)

        # Critic gradient update
        delta_q = 2 * (current_q - td_target) / self.batch_size

        # Simplified critic update
        state_action = np.hstack([states, actions])
        critic_pred, critic_cache = self.critic.network.forward(state_action)

        # Backprop through critic
        delta = 2 * (critic_pred - td_target) / self.batch_size
        m = states.shape[0]

        for i in range(self.critic.network.n_layers - 1, -1, -1):
            z, a = critic_cache[i]

            if i == 0:
                da_prev = np.dot(delta, self.critic.network.weights[i].T)
            else:
                da_prev = np.dot(delta, self.critic.network.weights[i].T)

            dw = np.dot(np.hstack([states, actions]).T if i == 0 else critic_cache[i-1][1].T, delta) / m
            db = np.sum(delta, axis=0) / m

            if i > 0:
                act_deriv = self.critic.network._activation_derivative(z, self.critic.network.activations[i])
                delta = da_prev * act_deriv

            self.critic.network.weights[i] -= self.critic.network.learning_rate * dw
            self.critic.network.biases[i] -= self.critic.network.learning_rate * db

        # Update actor (maximize Q(s, mu(s)))
        policy_actions = self.actor.forward(states)
        state_policy = np.hstack([states, policy_actions])
        q_actor = self.critic.forward(state_policy)

        # Actor loss (negative because we want to maximize Q)
        actor_loss = -np.mean(q_actor)

        # Actor gradient update (simplified)
        # dL/dmu = -dQ/da * da/dtheta
        # Approximate: -mean(Q) gradient through actor
        delta_actor = -np.ones_like(q_actor) * 1.0 / self.batch_size

        actor_pred, actor_cache = self.actor.network.forward(states)

        delta = delta_actor
        for i in range(self.actor.network.n_layers - 1, -1, -1):
            z, a = actor_cache[i]

            if i == 0:
                da_prev = np.dot(delta, self.actor.network.weights[i].T)
            else:
                da_prev = np.dot(delta, self.actor.network.weights[i].T)

            dw = np.dot(states.T if i == 0 else actor_cache[i-1][1].T, delta) / m
            db = np.sum(delta, axis=0) / m

            if i > 0:
                act_deriv = self.actor.network._activation_derivative(z, self.actor.network.activations[i])
                delta = da_prev * act_deriv

            self.actor.network.weights[i] -= self.actor.network.learning_rate * dw
            self.actor.network.biases[i] -= self.actor.network.learning_rate * db

        # Soft update target networks
        self._update_target_networks()

        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'q_value': float(np.mean(current_q))
        }


# ============================================================================
# PPO Portfolio Agent / PPO投资组合智能体
# ============================================================================

class PPOMemory:
    """PPO 经验存储器，用于存储轨迹数据"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.probs: List[float] = []
        self.vals: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def store(self, state: np.ndarray, action: np.ndarray, prob: float,
              val: float, reward: float, done: bool):
        """存储一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """清空存储器"""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self) -> Tuple[np.ndarray, ...]:
        """生成 shuffled batches"""
        n = len(self.states)
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []

        states = np.array(self.states)
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        indices = np.arange(n)
        np.random.shuffle(indices)
        batches = []
        for start in range(0, n, self.batch_size):
            batches.append(indices[start:start + self.batch_size])

        return states, actions, probs, vals, rewards, dones, batches


class PPOPortfolioAgent:
    """
    PPO (Proximal Policy Optimization) Portfolio Agent / PPO近端策略优化投资组合智能体

    Uses clipped surrogate objective to prevent destructively large policy updates.
    使用裁剪替代目标函数防止破坏性的策略更新。

    Attributes:
        actor: Policy network (actor)
        critic: Value network (critic)
        memory: PPOMemory for storing trajectories
        gamma: Discount factor
        policy_clip: PPO clip parameter epsilon
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        entropy_coef: float = 1e-2,
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        """
        初始化 PPO 投资组合智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            policy_clip: PPO 裁剪参数
            entropy_coef: 熵正则化系数
            n_epochs: 每次更新的 epoch 数
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Initialize actor and critic networks
        self.actor = PortfolioActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=actor_lr,
            entropy_coef=entropy_coef
        )

        self.critic = PortfolioCritic(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_lr
        )

        # Memory for trajectories
        self.memory = PPOMemory(batch_size)

        # Log standard deviation for action distribution
        self.log_std = np.zeros(action_dim, dtype=np.float32)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        选择动作并返回概率和价值估计

        Args:
            state: 当前状态

        Returns:
            action: 选择的动作
            prob: 动作的对数概率
            value: 价值估计
        """
        state = state.astype(np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, :]

        mu = self.actor.forward(state)[0]
        var = np.exp(self.log_std * 2)

        # Sample action
        action = mu + np.sqrt(var) * np.random.randn(self.action_dim)
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-10)

        # Log probability
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var + 1e-10))
        log_prob += -0.5 * np.sum((action - mu) ** 2 / (var + 1e-10))
        prob = float(log_prob)

        # Value estimate
        value = float(self.critic.forward(state)[0, 0])

        return action, prob, value

    def remember(self, state: np.ndarray, action: np.ndarray, prob: float,
                 value: float, reward: float, done: bool):
        """存储经验到记忆"""
        self.memory.store(state, action, prob, value, reward, done)

    def _compute_gae(self, rewards: np.ndarray, vals: np.ndarray,
                     dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GAE 优势估计"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = vals[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - int(dones[t])) - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = gae + vals[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        使用存储的轨迹更新 PPO 策略

        Returns:
            包含训练指标的字典
        """
        states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()

        if len(states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}

        # 计算 GAE
        advantages, returns = self._compute_gae(rewards, vals, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_policy_losses = []
        old_value_losses = []
        entropies = []

        for _ in range(self.n_epochs):
            for batch_indices in batches:
                batch_indices = np.array(batch_indices)
                if len(batch_indices) == 0:
                    continue

                # 获取当前策略
                mu = self.actor.forward(states[batch_indices])
                var = np.exp(self.log_std * 2)
                new_probs = np.array([
                    self._log_prob(actions[i], mu[j], var)
                    for j, i in enumerate(batch_indices)
                ])
                old_probs_batch = old_probs[batch_indices]

                # 计算概率比率
                prob_ratio = np.exp(new_probs - old_probs_batch)

                # PPO 裁剪损失
                weighted_probs = advantages[batch_indices] * prob_ratio
                weighted_clipped_probs = np.clip(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantages[batch_indices]

                policy_loss = -np.mean(np.minimum(weighted_probs, weighted_clipped_probs))

                # Value loss
                values_pred = self.critic.forward(states[batch_indices]).squeeze()
                value_loss = np.mean((values_pred - returns[batch_indices]) ** 2)

                # Entropy bonus
                entropy = 0.5 * np.sum(np.log(2 * np.pi * var + 1e-10))

                old_policy_losses.append(float(policy_loss))
                old_value_losses.append(float(value_loss))
                entropies.append(float(entropy))

        self.memory.clear()

        return {
            'policy_loss': float(np.mean(old_policy_losses)),
            'value_loss': float(np.mean(old_value_losses)),
            'entropy': float(np.mean(entropies))
        }

    def _log_prob(self, action: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
        """计算对数概率密度"""
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var + 1e-10))
        log_prob += -0.5 * np.sum((action - mu) ** 2 / (var + 1e-10))
        return log_prob


# ============================================================================
# Portfolio RL Benchmark / 投资组合强化学习基准比较器
# ============================================================================

class PortfolioRLBenchmark:
    """
    投资组合强化学习算法基准比较器

    支持 A2C、DDPG、PPO 与 Buy & Hold 策略的对比评估。
    计算多种性能指标：收益率、夏普比率、最大回撤等。

    Attributes:
        agents: Dict[str, Any] - 智能体字典
        agent_names: List[str] - 智能体名称列表

    Methods:
        add_agent(name, agent): 添加智能体
        evaluate(env, agent_names, episodes): 评估所有智能体
        compare(): 返回比较结果
        print_summary(): 打印评估结果摘要表格
    """

    def __init__(self):
        """初始化基准比较器"""
        self.agents: Dict[str, Any] = {}
        self.agent_names: List[str] = []
        self.results: Dict[str, Dict[str, float]] = {}

    def add_agent(self, name: str, agent: Any):
        """
        添加智能体到基准测试

        Args:
            name: 智能体名称
            agent: 智能体实例（可为 None 表示 Buy & Hold）
        """
        self.agents[name] = agent
        self.agent_names.append(name)
        self.results[name] = {}

    def evaluate(self, env, agent_names: Optional[List[str]] = None,
                 episodes: int = 1, verbose: bool = False) -> Dict[str, Dict[str, float]]:
        """
        评估所有添加的智能体

        Args:
            env: Gymnasium 兼容的投资组合交易环境
            agent_names: 要评估的智能体名称列表（None 表示全部）
            episodes: 每个智能体评估的回合数
            verbose: 是否打印详细信息

        Returns:
            results: 每个智能体的评估结果
        """
        if agent_names is None:
            agent_names = self.agent_names

        for name in agent_names:
            agent = self.agents.get(name)
            if agent is None:
                result = self._evaluate_buy_and_hold(env, episodes, verbose, name)
            else:
                result = self._evaluate_agent(env, agent, episodes, verbose, name)
            self.results[name] = result

        return self.results

    def _evaluate_agent(self, env, agent, episodes: int,
                         verbose: bool, name: str) -> Dict[str, float]:
        """评估单个智能体"""
        episode_returns = []
        episode_wealths = []

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_return = 0.0
            wealth_history = [1_000_000]

            while not done:
                action = agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_return += reward
                if hasattr(env.unwrapped, 'get_wealth'):
                    wealth_history.append(env.unwrapped.get_wealth())

            final_wealth = wealth_history[-1] if wealth_history else 1_000_000
            episode_returns.append(final_wealth - 1_000_000)
            episode_wealths.append(wealth_history)

            if verbose:
                print(f"{name} Episode {ep + 1}: Return = {total_return:,.0f}, "
                      f"Final Wealth = {final_wealth:,.0f}")

        return self._compute_stats(episode_returns, episode_wealths)

    def _evaluate_buy_and_hold(self, env, episodes: int,
                                verbose: bool, name: str) -> Dict[str, float]:
        """评估 Buy & Hold 基准策略"""
        episode_returns = []
        episode_wealths = []

        for ep in range(episodes):
            state, _ = env.reset()
            done = False

            n_assets = env.unwrapped.n_stocks if hasattr(env.unwrapped, 'n_stocks') else 1
            action = np.ones(n_assets + 1) / (n_assets + 1)

            wealth_history = [1_000_000]

            while not done:
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if hasattr(env.unwrapped, 'get_wealth'):
                    wealth_history.append(env.unwrapped.get_wealth())

            final_wealth = wealth_history[-1] if wealth_history else 1_000_000
            episode_returns.append(final_wealth - 1_000_000)
            episode_wealths.append(wealth_history)

            if verbose:
                print(f"{name} Episode {ep + 1}: Return = {episode_returns[-1]:,.0f}, "
                      f"Final Wealth = {final_wealth:,.0f}")

        return self._compute_stats(episode_returns, episode_wealths)

    def _compute_stats(self, episode_returns: List[float],
                       episode_wealths: List[List[float]]) -> Dict[str, float]:
        """计算统计数据"""
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        sharpe = mean_return / (std_return + 1e-10) * np.sqrt(252) if std_return > 0 else 0.0

        max_drawdowns = []
        for wealth in episode_wealths:
            wealth_arr = np.array(wealth)
            running_max = np.maximum.accumulate(wealth_arr)
            drawdown = (wealth_arr - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))

        return {
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'max_return': float(np.max(episode_returns)),
            'min_return': float(np.min(episode_returns)),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(np.mean(max_drawdowns)) if max_drawdowns else 0.0
        }

    def compare(self) -> Dict[str, Dict[str, float]]:
        """返回所有智能体的评估结果比较"""
        return self.results

    def print_summary(self):
        """打印评估结果摘要表格"""
        print("\n" + "=" * 80)
        print(f"{'Agent':<20} {'Mean Return':>15} {'Sharpe Ratio':>15} {'Max Drawdown':>15}")
        print("=" * 80)
        for name, result in self.results.items():
            print(f"{name:<20} {result.get('mean_return', 0):>15,.0f} "
                  f"{result.get('sharpe_ratio', 0):>15.3f} "
                  f"{result.get('max_drawdown', 0):>15.3%}")
        print("=" * 80)


# ============================================================================
# Multi-Asset Trading Environment / 多资产交易环境
# ============================================================================

class MultiAssetEnv:
    """
    Multi-Asset Portfolio Trading Environment / 多资产投资组合交易环境

    Gymnasium-compatible environment for portfolio management with multiple assets.
    Supports both discrete (portfolio weights) and continuous (asset allocation) actions.

    State space: Asset prices, technical indicators, portfolio value
    Action space: Portfolio weight allocation across assets

    兼容Gymnasium的多资产投资组合交易环境。
    支持离散（组合权重）和连续（资产配置）动作空间。
    """

    def __init__(
        self,
        prices: np.ndarray,
        initial_balance: float = 1000000.0,
        transaction_cost: float = 0.001,
        state_type: str = 'prices',
        enable_portfolio_value: bool = True
    ):
        """
        Initialize multi-asset environment / 初始化多资产环境

        Args:
            prices: Historical price data of shape (n_steps, n_assets)
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost ratio
            state_type: 'prices' for only prices, 'indicators' for full state
            enable_portfolio_value: Whether to include portfolio value in state
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for MultiAssetEnv. "
                "Install with: pip install gymnasium"
            )

        self.prices = prices.astype(np.float32)
        self.n_steps, self.n_assets = prices.shape
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.state_type = state_type
        self.enable_portfolio_value = enable_portfolio_value

        # Define spaces
        if state_type == 'prices':
            state_dim = n_assets + 1 if enable_portfolio_value else n_assets
        else:  # indicators
            state_dim = n_assets * 6 + 1 if enable_portfolio_value else n_assets * 6

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(n_assets,), dtype=np.float32
        )

        # Environment state
        self.reset()

    def reset(
        self,
        start_step: Optional[int] = None,
        initial_balance: Optional[float] = None
    ) -> np.ndarray:
        """
        Reset environment to initial state / 重置环境到初始状态

        Args:
            start_step: Starting time step (None for random)
            initial_balance: Initial cash balance (None for default)

        Returns:
            Initial observation
        """
        if start_step is None:
            # Leave some buffer for trading
            self.current_step = 0
        else:
            self.current_step = start_step

        if initial_balance is not None:
            self.initial_balance = initial_balance

        self.balance = self.initial_balance
        self.shares = np.zeros(self.n_assets, dtype=np.int32)
        self.wealth_history = [self.get_wealth()]

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation / 获取当前观测"""
        current_prices = self.prices[self.current_step]

        if self.state_type == 'prices':
            if self.enable_portfolio_value:
                return np.concatenate([
                    current_prices,
                    [self.get_wealth()]
                ]).astype(np.float32)
            else:
                return current_prices
        else:
            # Include additional indicators (simplified - just prices for now)
            if self.enable_portfolio_value:
                return np.concatenate([
                    current_prices,
                    [self.get_wealth()]
                ]).astype(np.float32)
            else:
                return current_prices

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step / 执行一步

        Args:
            action: Portfolio weights (will be normalized)

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended
            info: Additional information
        """
        # Normalize action to valid portfolio weights
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)

        # Current state
        current_wealth = self.get_wealth()
        current_prices = self.prices[self.current_step]

        # Compute target portfolio
        target_shares = np.floor(current_wealth * action[:-1] / current_prices)
        target_shares = target_shares.astype(np.int32)

        # Compute trades
        trades = target_shares - self.shares

        # Transaction costs
        trade_value = np.sum(np.abs(trades) * current_prices)
        cost = trade_value * self.transaction_cost

        # Execute trades
        self.shares = target_shares
        self.balance = current_wealth - np.dot(self.shares, current_prices) - cost

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step >= self.n_steps - 1

        # Get new state
        new_prices = self.prices[self.current_step]
        new_wealth = self.get_wealth()

        # Reward: change in wealth
        reward = new_wealth - current_wealth

        info = {
            'wealth': new_wealth,
            'balance': self.balance,
            'shares': self.shares.copy(),
            'return': reward / current_wealth if current_wealth > 0 else 0
        }

        self.wealth_history.append(new_wealth)

        return self._get_observation(), reward, done, False, info

    def get_wealth(self) -> float:
        """Get total wealth (cash + holdings) / 获取总财富"""
        return self.balance + np.dot(self.shares, self.prices[self.current_step])

    def render(self, mode: str = 'human') -> None:
        """Render environment state / 渲染环境状态"""
        print(f"Step: {self.current_step}/{self.n_steps}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares: {self.shares}")
        print(f"Wealth: {self.get_wealth():.2f}")

    def close(self) -> None:
        """Clean up environment / 清理环境"""
        pass


# ============================================================================
# Exports / 导出
# ============================================================================

__all__ = [
    # Core classes
    'A2CPortfolioAgent',
    'DDPGPortfolioAgent',
    'PPOPortfolioAgent',
    'PortfolioActor',
    'PortfolioCritic',
    'MultiAssetEnv',
    'PortfolioRLBenchmark',

    # Utility classes
    'ReplayBuffer',
    'OUActionNoise',
    'PPOMemory',
    'NeuralNetwork',
    'PolicyGradientLayer',

    # Utility functions
    'xavier_init',
    'he_init',

    # Version info
    '__version__'
]

__version__ = '1.0.0'
