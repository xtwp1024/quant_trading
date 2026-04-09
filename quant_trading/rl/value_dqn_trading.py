"""
Double DQN Trading Agent — Pure NumPy + Gymnasium Implementation.
==================================================================

Double DQN variant for single-stock trading.
Adapted from: D:/Hive/Data/trading_repos/value-based-deep-reinforcement-learning-trading-model-in-pytorch/

Original PyTorch source features:
- Dueling Q-Network (Conv1d backbone + dueling heads)
- Experience replay with fixed-size buffer
- Epsilon-greedy exploration
- DDQN mode: reduces Q-value overestimation

This NumPy implementation:
- Replaces Conv1d with dense layers (avoids PyTorch dependency)
- Gymnasium-compatible StockTradingEnv
- Standalone ReplayBuffer
- Online target network updates with soft sync

Bilingual docstrings: Chinese first, English second.

Usage:
    from quant_trading.rl import DoubleDQNTradingAgent, StockTradingEnv

    env = StockTradingEnv(data=price_series, history_length=180)
    agent = DoubleDQNTradingAgent(state_dim=env.state_dim, action_dim=3)

    for episode in range(100):
        agent.train_episode(env)

    returns = agent.backtest(env)
"""

# Lazy imports with graceful degradation
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM_AVAILABLE = True
except ImportError:
    _GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

from typing import Optional, Tuple, List, Dict, Any
from collections import deque, namedtuple


# ----------------------------------------------------------------------
# Action constants (compatible with the source repo)
# ----------------------------------------------------------------------
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


# ----------------------------------------------------------------------
# Named tuple for experience replay
# ----------------------------------------------------------------------
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ==============================================================================
# ReplayBuffer — Experience replay for trading
# ==============================================================================
class ReplayBuffer:
    """
    交易经验回放缓冲区 / Trading Replay Buffer.

    使用固定大小的循环缓冲区存储 (s, a, r, s', done) 元组.
    Uses a fixed-size circular buffer to store transition tuples.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10000):
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for ReplayBuffer")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._rng = np.random.default_rng()

    def push(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区 / Add a transition to the buffer.

        Args:
            state: Current state array.
            action: Integer action (0=hold, 1=buy, 2=sell).
            reward: Scalar reward.
            next_state: Next state array.
            done: Boolean indicating episode termination.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        随机采样一个批次 / Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition namedtuples.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        indices = self._rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ==============================================================================
# ValueNetwork — Pure NumPy dense Q-network (formerly SimpleQNetwork)
# ==============================================================================
class ValueNetwork:
    """
    价值网络（Q网络）/ Value Network (Q-Network).

    纯NumPy实现的全连接Q网络，替代PyTorch的Conv1d + Dueling架构.
    Pure NumPy fully-connected Q-network, replacing Conv1d + Dueling architecture.

    Architecture:
        fc1(input_dim -> 256) -> selu
        fc2(256 -> 128)       -> selu
        fc3(128 -> 64)        -> selu
        fc_q(64 -> action_dim)  -> linear Q-values

    Args:
        input_dim: Dimension of state vector.
        action_dim: Number of discrete actions (default 3: hold/buy/sell).
        seed: Random seed for reproducibility.
    """

    def __init__(self, input_dim: int, action_dim: int = 3, seed: int = 42):
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for ValueNetwork")
        self.input_dim = input_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._build_network()

    def _build_network(self):
        """Initialize network weights using He initialization."""
        self.W1 = self._he_init((self.input_dim, 256))
        self.b1 = np.zeros(256)
        self.W2 = self._he_init((256, 128))
        self.b2 = np.zeros(128)
        self.W3 = self._he_init((128, 64))
        self.b3 = np.zeros(64)
        self.Wq = self._he_init((64, self.action_dim))
        self.bq = np.zeros(self.action_dim)

    def _he_init(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return self._rng.normal(0, std, shape).astype(np.float64)

    def _selu(self, x: np.ndarray) -> np.ndarray:
        """SELU activation: scale * (x if x > 0 else alpha * (exp(x) - 1))."""
        alpha = 1.673263242354817284
        scale = 1.050700987355480493
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播计算Q值 / Forward pass computing Q-values.

        Args:
            x: State array of shape (input_dim,) or (batch, input_dim).

        Returns:
            Q-values of shape (action_dim,) or (batch, action_dim).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h1 = self._selu(x @ self.W1 + self.b1)
        h2 = self._selu(h1 @ self.W2 + self.b2)
        h3 = self._selu(h2 @ self.W3 + self.b3)
        q = h3 @ self.Wq + self.bq
        return q.squeeze(0) if q.shape[0] == 1 else q

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Alias for forward() returning Q-values."""
        return self.forward(state)

    def update_weights(self, other: "ValueNetwork", tau: float = 1.0):
        """
        软更新网络权重 / Soft update network weights.

        self = tau * other + (1 - tau) * self

        Args:
            other: Source network to copy from.
            tau: Interpolation factor (1.0 = hard copy).
        """
        for attr in ["W1", "b1", "W2", "b2", "W3", "b3", "Wq", "bq"]:
            getattr(self, attr).flat = (
                tau * getattr(other, attr).flat
                + (1 - tau) * getattr(self, attr).flat
            )


# Alias for backward compatibility
SimpleQNetwork = ValueNetwork


# ==============================================================================
# DQNPolicy — Epsilon-greedy policy wrapper
# ==============================================================================
class DQNPolicy:
    """
    DQN策略类 / DQN Policy.

    封装epsilon-greedy探索策略的策略类.
    Policy class encapsulating epsilon-greedy exploration.

    Args:
        q_network: ValueNetwork instance for Q-value computation.
        action_dim: Number of discrete actions.
        epsilon: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Epsilon reduction per step.
        seed: Random seed.
    """

    def __init__(
        self,
        q_network: ValueNetwork,
        action_dim: int = 3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-5,
        seed: int = 42,
    ):
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for DQNPolicy")
        self.q_network = q_network
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        基于epsilon-greedy策略选择动作 / Select action using epsilon-greedy policy.

        Args:
            state: Current state array.
            training: If True, apply epsilon-greedy; if False, greedy only.

        Returns:
            Selected action integer (0, 1, or 2).
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_dim))

        q_values = self.q_network.get_q_values(state)
        return int(np.argmax(q_values))

    def decay_epsilon(self):
        """衰减epsilon值 / Decay epsilon value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay


# ==============================================================================
# StockTradingEnv — Gymnasium-compatible single-stock environment
# ==============================================================================
class StockTradingEnv:
    """
    股票交易Gymnasium环境 / Gymnasium-compatible Single-Stock Trading Environment.

    改编自 source repo 的 Environment 类.
    Adapted from the source repo's Environment class.

    State space: [position_value] + history_close_prices
    Action space: Discrete(3) — 0=hold, 1=buy, 2=sell

    状态空间: [持仓价值] + 历史收盘价序列
    动作空间: Discrete(3) — 0=持有, 1=买入, 2=卖出

    Args:
        data: DataFrame or array-like with 'Close' column (price data).
        history_length: Number of past prices to include in state (default 90).
        initial_balance: Starting cash (default 10000).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data,  # pd.DataFrame or np.ndarray with 'Close' column
        history_length: int = 90,
        initial_balance: float = 10000.0,
    ):
        if not _GYMNASIUM_AVAILABLE:
            raise RuntimeError("gymnasium is required for StockTradingEnv")
        super().__init__()
        self.data = data
        self.history_length = history_length
        self.initial_balance = initial_balance

        # Support both DataFrame and numpy array
        if hasattr(data, "iloc"):
            self._close_prices = data["Close"].values.astype(np.float64)
        else:
            self._close_prices = np.asarray(data, dtype=np.float64).flatten()

        self.n_steps = len(self._close_prices)

        # State dim = 1 (position_value) + history_length (price history)
        self.state_dim = 1 + history_length
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(3)

        self._reset_hidden_state()

    def _reset_hidden_state(self):
        """Initialize/reset per-episode state variables."""
        self.t = 0
        self.balance = self.initial_balance
        self.profits = 0.0
        self.positions = []          # List of buy prices
        self.position_value = 0.0  # Unrealized P&L
        self.history = [
            self._close_prices[0] for _ in range(self.history_length)
        ]
        self.trades = []  # List of (action, price, pnl) for backtest

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境到初始状态 / Reset environment to initial state.

        Returns:
            observation: Initial state array.
            info: Empty dict (Gymnasium compatibility).
        """
        super().reset(seed=seed)
        self._reset_hidden_state()
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        """返回当前状态数组 / Return current state array."""
        return np.array([self.position_value] + self.history, dtype=np.float64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一个交易动作 / Execute one trading action.

        Args:
            action: 0=hold, 1=buy, 2=sell

        Returns:
            observation: Next state.
            reward: Scalar reward (-1, 0, 1 depending on profit).
            terminated: True if episode ends (no more data).
            truncated: False (no truncation in this env).
            info: Extra info dict.
        """
        reward = 0.0
        profit = 0.0
        done = False

        current_price = self._close_prices[self.t]

        if action == ACTION_BUY:
            # Buy one share at current price
            if self.balance >= current_price:
                self.positions.append(current_price)
                self.balance -= current_price

        elif action == ACTION_SELL:
            # Sell all held shares
            if len(self.positions) == 0:
                reward = -1.0  # Penalty for selling when no position
            else:
                for buy_price in self.positions:
                    profit += (current_price - buy_price) - 1  # -1 = trading cost
                # Reward signal
                avg_cost = sum(self.positions) / len(self.positions)
                reward_signal = profit / avg_cost if avg_cost > 0 else 0
                reward = 1.0 if reward_signal > 0 else -1.0

                self.profits += profit
                self.balance += len(self.positions) * current_price - profit
                self.trades.append(("SELL", current_price, profit))
                self.positions = []

        # Advance time
        self.t += 1
        if self.t >= self.n_steps - 1:
            done = True
            # Liquidate remaining positions at final price
            if len(self.positions) > 0:
                final_price = self._close_prices[-1]
                for buy_price in self.positions:
                    self.profits += (final_price - buy_price) - 1
                self.balance += len(self.positions) * final_price - self.profits
                self.positions = []

        # Update position value (unrealized P&L)
        self.position_value = 0.0
        for buy_price in self.positions:
            self.position_value += (current_price - buy_price) - 1

        # Update price history
        self.history.pop(0)
        self.history.append(current_price)

        observation = self._get_observation()
        return observation, reward, done, False, {"profit": self.profits}

    def render(self, mode: str = "human"):
        """渲染环境状态（保留接口）/ Render environment state (no-op)."""
        pass

    def backtest_report(self) -> Dict[str, Any]:
        """
        生成回测报告 / Generate backtest report.

        Returns:
            Dict with total_return, total_profit, num_trades, final_balance.
        """
        total_return = (self.balance + sum(self.positions) * self._close_prices[min(self.t, self.n_steps-1)]
                        - self.initial_balance) / self.initial_balance
        return {
            "total_return": total_return,
            "total_profit": self.profits,
            "num_trades": len(self.trades),
            "final_balance": self.balance,
        }


# ==============================================================================
# DoubleDQNTradingAgent — Double DQN for single-stock trading
# ==============================================================================
class DoubleDQNTradingAgent:
    """
    Double DQN交易智能体 / Double DQN Trading Agent.

    改编自 source repo 的 rl_agent_train 函数，使用 DDQN 更新规则:
    q_target = r + gamma * Q_target(s', argmax(Q_online(s')))

    DDQN reduces Q-value overestimation compared to vanilla DQN.

    核心方法:
    - select_action(): Epsilon-greedy action selection
    - update_target(): Soft sync of target network
    - train_episode(): Run one training episode
    - backtest(): Evaluate agent on environment without exploration

    Args:
        state_dim: Dimension of state vector.
        action_dim: Number of discrete actions (default 3).
        hidden_dim: Hidden layer dimension (default 256).
        learning_rate: Adam learning rate (default 1e-3).
        discount_rate: Gamma for future rewards (default 0.99).
        epsilon: Initial exploration rate (default 1.0).
        epsilon_min: Minimum exploration rate (default 0.05).
        epsilon_decay: Epsilon reduction per step (default 1e-5).
        target_update_freq: Steps between target network updates (default 500).
        replay_capacity: Experience replay buffer size (default 10000).
        batch_size: Training batch size (default 64).
        seed: Random seed.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        discount_rate: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-5,
        target_update_freq: int = 500,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        seed: int = 42,
    ):
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for DoubleDQNTradingAgent")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        # Build online and target networks
        self.q_online = ValueNetwork(state_dim, action_dim, seed=seed)
        self.q_target = ValueNetwork(state_dim, action_dim, seed=seed)
        # Hard copy initially
        self.q_target.update_weights(self.q_online, tau=1.0)

        # Policy wrapper
        self.policy = DQNPolicy(
            q_network=self.q_online,
            action_dim=action_dim,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )

        # Optimizer state (Adam-style per-layer moments)
        self.lr = learning_rate
        self._adam_state = {}
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._t = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Tracking
        self.global_step = 0
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        基于epsilon-greedy策略选择动作 / Select action using epsilon-greedy policy.

        Args:
            state: Current state array.
            training: If True, apply epsilon-greedy; if False, greedy only.

        Returns:
            Selected action integer (0, 1, or 2).
        """
        return self.policy.select_action(state, training=training)

    def _adam_update(self, gradients: Dict[str, np.ndarray]):
        """
        执行Adam优化器更新 / Perform Adam optimizer update.

        Args:
            gradients: Dict mapping parameter names to gradient arrays.
        """
        self._t += 1
        lr_t = self.lr * np.sqrt(1 - self._beta2**self._t) / (1 - self._beta1**self._t)

        for name, grad in gradients.items():
            if name not in self._adam_state:
                self._adam_state[name] = {
                    "m": np.zeros_like(grad),
                    "v": np.zeros_like(grad),
                }
            m = self._adam_state[name]["m"]
            v = self._adam_state[name]["v"]

            m[:] = self._beta1 * m + (1 - self._beta1) * grad
            v[:] = self._beta2 * v + (1 - self._beta2) * (grad * grad)

            param = getattr(self.q_online, name)
            param.flat = param.flat - lr_t * m / (np.sqrt(v) + self._eps)

    def _compute_gradients(self, batch: List[Transition]) -> Dict[str, np.ndarray]:
        """
        计算批次梯度 / Compute gradients for a batch.

        DDQN gradient:
            grad = 2 * (q_eval - q_target) * grad(q_eval)

        Returns:
            Dict mapping parameter names to gradient arrays.
        """
        states = np.array([t.state for t in batch], dtype=np.float64)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float64)
        next_states = np.array([t.next_state for t in batch], dtype=np.float64)
        dones = np.array([t.done for t in batch], dtype=np.float64)

        batch_size = len(batch)

        # ---- Forward pass on online network ----
        q_online_all = self.q_online.forward(states)          # (batch, action_dim)
        q_eval = q_online_all[np.arange(batch_size), actions]  # (batch,)

        # ---- Forward pass on target network for next states ----
        q_target_all = self.q_target.forward(next_states)      # (batch, action_dim)
        q_online_next = self.q_online.forward(next_states)    # (batch, action_dim)

        # ---- Double DQN: action selected by online, value from target ----
        best_actions = np.argmax(q_online_next, axis=1)       # (batch,)
        q_next = q_target_all[np.arange(batch_size), best_actions]  # (batch,)

        # ---- Compute targets ----
        not_done = 1.0 - dones
        q_target = rewards + not_done * self.gamma * q_next   # (batch,)

        # ---- Compute loss and gradients analytically (MSE) ----
        delta = q_eval - q_target                              # (batch,)
        loss = 0.5 * np.mean(delta ** 2)

        # ---- Backpropagate through one-hot action selection ----
        # d(q_eval)/d(W) = delta * grad of q with respect to W at selected actions
        # For one-hot action: grad = delta[:, None] * grad(q_network)/dW at action

        gradients = {}

        # Use numerical gradient for simplicity (finite differences)
        for name in ["W1", "b1", "W2", "b2", "W3", "b3", "Wq", "bq"]:
            grad = self._numerical_gradient(name, states, actions, q_target)
            gradients[name] = grad

        return gradients

    def _numerical_gradient(
        self,
        param_name: str,
        states: np.ndarray,
        actions: np.ndarray,
        q_target: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        计算数值梯度 (有限差分法) / Numerical gradient using finite differences.

        Args:
            param_name: Name of parameter to compute gradient for.
            states: Batch of states.
            actions: Batch of actions.
            q_target: Target Q-values.
            eps: Perturbation size.

        Returns:
            Gradient array of same shape as parameter.
        """
        param = getattr(self.q_online, param_name)
        grad = np.zeros_like(param)

        it = np.nditer(param.flat, flags=["multi_index"])
        while not it.finished:
            idx = it.index
            old_val = param.flat[idx]

            # f(theta + eps)
            param.flat[idx] = old_val + eps
            q_plus = self.q_online.forward(states)
            q_eval_plus = q_plus[np.arange(len(actions)), actions]
            loss_plus = 0.5 * np.mean((q_eval_plus - q_target) ** 2)

            # f(theta - eps)
            param.flat[idx] = old_val - eps
            q_minus = self.q_online.forward(states)
            q_eval_minus = q_minus[np.arange(len(actions)), actions]
            loss_minus = 0.5 * np.mean((q_eval_minus - q_target) ** 2)

            grad.flat[idx] = (loss_plus - loss_minus) / (2 * eps)
            param.flat[idx] = old_val
            it.iternext()

        return grad

    def update_target(self, tau: float = 0.3):
        """
        软更新目标网络 / Soft update target network.

        q_target = tau * q_online + (1 - tau) * q_target

        Args:
            tau: Interpolation factor (default 0.3, matching source repo).
        """
        self.q_target.update_weights(self.q_online, tau=tau)

    def train_episode(
        self,
        env,
        max_steps: Optional[int] = None,
        train_freq: int = 4,
    ) -> Dict[str, float]:
        """
        运行一个训练回合 / Run one training episode.

        Args:
            env: Gymnasium environment.
            max_steps: Maximum steps per episode (default env.n_steps).
            train_freq: How often to train (every N steps).

        Returns:
            Dict with episode_reward, episode_loss, episode_profit.
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float64)
        max_steps = max_steps or env.n_steps

        total_reward = 0.0
        total_loss = 0.0
        num_updates = 0

        for step in range(max_steps):
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float64)
            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)

            total_reward += reward

            # Train when buffer has enough samples
            if len(self.replay_buffer) >= self.batch_size:
                if step % train_freq == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    gradients = self._compute_gradients(batch)
                    self._adam_update(gradients)
                    num_updates += 1

                    # Compute loss for logging
                    states_b = np.array([t.state for t in batch])
                    actions_b = np.array([t.action for t in batch])
                    rewards_b = np.array([t.reward for t in batch])
                    next_states_b = np.array([t.next_state for t in batch])
                    dones_b = np.array([t.done for t in batch])

                    q_online_next = self.q_online.forward(next_states_b)
                    best_actions = np.argmax(q_online_next, axis=1)
                    q_target_next = self.q_target.forward(next_states_b)[
                        np.arange(len(batch)), best_actions
                    ]
                    q_target_vals = rewards_b + (1 - dones_b) * self.gamma * q_target_next
                    q_eval_vals = self.q_online.forward(states_b)[
                        np.arange(len(batch)), actions_b
                    ]
                    total_loss += 0.5 * np.mean((q_eval_vals - q_target_vals) ** 2)

                    # Update target network periodically
                    if self.global_step % self.target_update_freq == 0:
                        self.update_target(tau=0.3)

            # Decay epsilon
            self.policy.decay_epsilon()

            self.global_step += 1
            state = next_state

            if done:
                break

        return {
            "episode_reward": total_reward,
            "episode_loss": total_loss,
            "episode_profit": env.profits,
            "epsilon": self.epsilon,
            "num_updates": num_updates,
        }

    def backtest(
        self,
        env,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        在环境中运行回测（无探索） / Run backtest on environment (no exploration).

        Args:
            env: Gymnasium environment.
            max_steps: Maximum steps.

        Returns:
            Backtest report dict.
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float64)
        max_steps = max_steps or env.n_steps

        total_reward = 0.0
        actions_taken = []

        for step in range(max_steps):
            action = self.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float64)
            done = terminated or truncated

            actions_taken.append(action)
            total_reward += reward
            state = next_state

            if done:
                break

        report = env.backtest_report()
        report["total_reward"] = total_reward
        report["actions"] = actions_taken
        return report

    def save(self, path: str):
        """
        保存模型权重到 .npz 文件 / Save model weights to .npz file.

        Args:
            path: File path without extension (.npz appended).
        """
        arrays = {}
        for net_name, net in [("online", self.q_online), ("target", self.q_target)]:
            for attr in ["W1", "b1", "W2", "b2", "W3", "b3", "Wq", "bq"]:
                arrays[f"{net_name}_{attr}"] = getattr(net, attr)
        arrays["epsilon"] = np.array(self.epsilon)
        arrays["global_step"] = np.array(self.global_step)
        np.savez(path, **arrays)

    def load(self, path: str):
        """
        从 .npz 文件加载模型权重 / Load model weights from .npz file.

        Args:
            path: File path without extension (.npz appended).
        """
        data = np.load(path)
        for net_name, net in [("online", self.q_online), ("target", self.q_target)]:
            for attr in ["W1", "b1", "W2", "b2", "W3", "b3", "Wq", "bq"]:
                arr = data[f"{net_name}_{attr}"]
                getattr(net, attr)[:] = arr
        self.epsilon = float(data["epsilon"])
        self.global_step = int(data["global_step"])


# ==============================================================================
# Factory function for quick environment setup
# ==============================================================================
def create_double_dqn_agent(
    data,
    history_length: int = 90,
    **agent_kwargs,
) -> Tuple[DoubleDQNTradingAgent, StockTradingEnv]:
    """
    便捷工厂函数：创建环境和智能体 / Convenience factory: create env + agent.

    Args:
        data: Price data (DataFrame with 'Close' or array).
        history_length: Number of historical prices in state.
        **agent_kwargs: Passed to DoubleDQNTradingAgent constructor.

    Returns:
        Tuple of (DoubleDQNTradingAgent, StockTradingEnv).
    """
    env = StockTradingEnv(data=data, history_length=history_length)
    agent = DoubleDQNTradingAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        **agent_kwargs,
    )
    return agent, env


# ==============================================================================
# Exports
# ==============================================================================
__all__ = [
    # Core classes
    "DoubleDQNTradingAgent",
    "StockTradingEnv",
    "ValueNetwork",
    "DQNPolicy",
    "ReplayBuffer",
    # Alias for backward compatibility
    "SimpleQNetwork",
    # Factory
    "create_double_dqn_agent",
    # Constants
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
    # Transition namedtuple
    "Transition",
]


# ==============================================================================
# Example usage
# ==============================================================================
if __name__ == "__main__":
    # Generate synthetic price data for testing
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))

    env = StockTradingEnv(data=prices, history_length=90)
    agent = DoubleDQNTradingAgent(state_dim=env.state_dim, action_dim=3)

    print(f"State dim: {env.state_dim}, Action dim: {env.action_space.n}")

    # Quick training run
    result = agent.train_episode(env)
    print(f"Episode result: {result}")

    # Backtest
    bt_result = agent.backtest(env)
    print(f"Backtest result: {bt_result}")

    print("OK")
