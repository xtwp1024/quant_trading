"""
DRQNTradingEnv - Gymnasium-compatible DRQN trading environment.

Key features:
- Action augmentation: evaluates ALL 3 actions (short/hold/long) at each step,
  storing all transitions in the replay buffer. This is the novel action
  augmentation technique from the original DRQN_Stock_Trading project.
- Sinusoidal time encoding: minute, hour, day-of-week features.
- Per-stock Z-score normalization over 96-step clusters.
- Standalone: generates synthetic price data with no external dependencies.
- Soft target network updates handled by the DRQNAgent.

Based on the original TradingEnv from DRQN_Stock_Trading/code_server/trading_env.py.

Usage:
    from quant_trading.rl import DRQNTradingEnv

    env = DRQNTradingEnv(n_stocks=5, n_steps=10000)
    obs = env.reset()
    obs, rewards, terminated, truncated, info = env.step(action)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

try:
    import gymnasium as gym
    DiscreteSpace = gym.spaces.Discrete
    BoxSpace = gym.spaces.Box
    DictSpace = gym.spaces.Dict
except ImportError:
    import gym
    DiscreteSpace = gym.spaces.Discrete
    BoxSpace = gym.spaces.Box
    DictSpace = gym.spaces.Dict


class DRQNTradingEnv:
    """
    Gymnasium-compatible stock trading environment with DRQN features.

    Actions: -1 (short), 0 (hold), 1 (long)
    State: normalized price features + action history + sinusoidal time encoding.

    Key feature: action augmentation. At each step, all 3 actions are evaluated
    and all transitions are returned/stored. This provides 3x more training signal
    per environment step.

    Args:
        n_stocks: Number of stocks/assets in the portfolio.
        n_steps: Total number of trading steps (days/minutes).
        initial_value: Initial portfolio value.
        trade_size: Base trade size for commission calculation.
        spread: Bid-ask spread as fraction (default 0.005 = 0.5%%).
        T: Sequence length / cluster size for Z-score normalization (default 96).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_stocks: int = 5,
        n_steps: int = 10000,
        initial_value: float = 100000.0,
        trade_size: float = 10000.0,
        spread: float = 0.005,
        T: int = 96,
        seed: int = 2037,
    ):
        self.n_stocks = n_stocks
        self.n_steps = n_steps
        self.initial_value = initial_value
        self.portfolio_value = initial_value
        self.trade_size = trade_size
        self.spread = spread
        self.T = T
        self.seed = seed

        # Internal state
        self.current_step = 0
        self._rng = np.random.default_rng(seed)

        # Action encoding: -1 -> index 0, 0 -> index 1, 1 -> index 2
        self.actions = [-1, 0, 1]
        self._action_history: List[int] = []

        # Portfolio tracking
        self._portfolio_history: List[float] = [initial_value]
        self._prev_close: Optional[np.ndarray] = None

        # Generate synthetic data
        self._generate_synthetic_data()

        # Pre-compute Z-score normalization stats over T-step clusters
        self._compute_normalization_stats()

        # Sinusoidal time encoding for current step
        self._time_encoding = self._compute_time_encoding(self.current_step)

        # Action space: 3 discrete actions (short, hold, long)
        # State space dimension: n_stocks * n_features + 3 (action encoding) + 3 (time)
        # Features per stock: price, return, volume_proxy, volatility, momentum
        self.n_features_per_stock = 5
        self._state_dim = (
            self.n_stocks * self.n_features_per_stock + 3 + 3
        )
        self.observation_space = BoxSpace(
            low=-10.0, high=10.0, shape=(self._state_dim,), dtype=np.float32
        )
        self.action_space = DiscreteSpace(3)  # 0=-1, 1=0, 2=1

    # -------------------------------------------------------------------------
    # Synthetic data generation (standalone - no external dependencies)
    # -------------------------------------------------------------------------

    def _generate_synthetic_data(self) -> None:
        """
        Generate synthetic OHLCV-style price data using Geometric Brownian Motion.

        Each stock has: price, return, volume_proxy, volatility, momentum.
        """
        n_steps = self.n_steps + self.T + 10  # extra buffer for warmup

        self._prices = np.zeros((n_steps, self.n_stocks), dtype=np.float32)
        self._returns = np.zeros((n_steps, self.n_stocks), dtype=np.float32)
        self._volume_proxy = np.zeros((n_steps, self.n_stocks), dtype=np.float32)
        self._volatility = np.zeros((n_steps, self.n_stocks), dtype=np.float32)
        self._momentum = np.zeros((n_steps, self.n_stocks), dtype=np.float32)

        for i in range(self.n_stocks):
            # Drift and volatility per stock (vary by stock)
            drift = 0.00005 + 0.00002 * i
            volatility = 0.015 + 0.005 * i

            # Seed each stock slightly differently
            stock_seed = self.seed + i * 123
            rng = np.random.default_rng(stock_seed)

            log_returns = (
                drift
                + volatility * rng.standard_normal(n_steps)
            ).astype(np.float32)
            self._prices[:, i] = 100.0 * np.exp(np.cumsum(log_returns))
            self._returns[:, i] = np.exp(log_returns) - 1.0

        # Compute derived features
        for i in range(self.n_stocks):
            ret = self._returns[:, i]
            self._volume_proxy[:, i] = rng.uniform(0.5, 1.5, n_steps).astype(np.float32)
            self._volatility[:, i] = self._rolling_std(ret, window=20)
            self._momentum[:, i] = self._rolling_mean(ret, window=10)

        # Stack features: for each timestep, interleave stocks
        # Shape: (n_steps, n_stocks * 5)
        raw_features = np.zeros((n_steps, self.n_stocks * 5), dtype=np.float32)
        for i in range(self.n_stocks):
            offset = i * 5
            raw_features[:, offset + 0] = self._normalize_0_1(self._prices[:, i])
            raw_features[:, offset + 1] = self._returns[:, i] * 10  # scale returns
            raw_features[:, offset + 2] = self._normalize_0_1(self._volume_proxy[:, i])
            raw_features[:, offset + 3] = self._normalize_0_1(self._volatility[:, i])
            raw_features[:, offset + 4] = self._momentum[:, i] * 10

        self._raw_features = raw_features
        self._n_total_steps = n_steps

    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            if i < window:
                result[i] = np.std(x[: i + 1])
            else:
                result[i] = np.std(x[i - window + 1 : i + 1])
        return result

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            if i < window:
                result[i] = np.mean(x[: i + 1])
            else:
                result[i] = np.mean(x[i - window + 1 : i + 1])
        return result

    @staticmethod
    def _normalize_0_1(x: np.ndarray) -> np.ndarray:
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val < 1e-8:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # -------------------------------------------------------------------------
    # Z-score normalization over T-step clusters
    # -------------------------------------------------------------------------

    def _compute_normalization_stats(self) -> None:
        """
        Compute per-cluster Z-score normalization statistics.

        Divides the data into clusters of T steps and computes running
        mean/std. This matches the original DRQN_Stock_Trading approach
        of normalizing over 96-step windows.
        """
        n_clusters = self._n_total_steps // self.T
        all_means = []
        all_stds = []

        for c in range(n_clusters):
            start = c * self.T
            end = min(start + self.T, self._n_total_steps)
            cluster = self._raw_features[start:end]
            all_means.append(np.mean(cluster, axis=0))
            all_stds.append(np.std(cluster, axis=0) + 1e-8)

        # Average across clusters for stable statistics
        self._norm_mean = np.mean(all_means, axis=0).astype(np.float32)
        self._norm_std = np.mean(all_stds, axis=0).astype(np.float32)

    def _get_normalized_features(self, step: int) -> np.ndarray:
        """Return Z-score normalized features for a given step."""
        return (
            (self._raw_features[step] - self._norm_mean)
            / self._norm_std
        ).astype(np.float32)

    # -------------------------------------------------------------------------
    # Sinusoidal time encoding
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_time_encoding(step: int) -> np.ndarray:
        """
        Compute sinusoidal time encoding for minute/hour/day-of-week.

        Args:
            step: Current timestep (interpreted as minutes since start).

        Returns:
            Array of 3 sinusoidal features: [minute, hour, day_of_week].
        """
        minute_f = np.sin(2 * np.pi * (step % 60) / 60.0)
        hour_f = np.sin(2 * np.pi * (step % (60 * 24)) / (60.0 * 24.0))
        day_f = np.sin(2 * np.pi * ((step // (60 * 24)) % 7) / 7.0)
        return np.array([minute_f, hour_f, day_f], dtype=np.float32)

    # -------------------------------------------------------------------------
    # State construction
    # -------------------------------------------------------------------------

    def _hot_encode_action(self, action: int) -> np.ndarray:
        """One-hot encode action in [-1, 0, 1] space to 3-element vector."""
        vec = np.zeros(3, dtype=np.float32)
        vec[action + 1] = 1.0
        return vec

    def _build_state(
        self,
        normalized_features: np.ndarray,
        time_encoding: np.ndarray,
        prev_action: int,
    ) -> np.ndarray:
        """
        Build the full state vector.

        State = [normalized_features (n_stocks*5), action_encoding (3), time_encoding (3)]

        Args:
            normalized_features: Z-score normalized price features.
            time_encoding: Sinusoidal time features.
            prev_action: Previous action (-1, 0, or 1).

        Returns:
            Flat state vector of shape (state_dim,).
        """
        action_enc = self._hot_encode_action(prev_action)
        return np.concatenate([normalized_features, action_enc, time_encoding])

    # -------------------------------------------------------------------------
    # Gymnasium interface
    # -------------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed (unused, for Gymnasium API compatibility).

        Returns:
            Tuple of (initial observation, info dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = self.T  # Start after warmup period
        self._action_history = [0] * self.T  # Fill with "hold" (0)
        self.portfolio_value = self.initial_value
        self._portfolio_history = [self.initial_value]
        self._prev_close = self._prices[self.T - 1].copy()

        self._time_encoding = self._compute_time_encoding(self.current_step)

        normalized_features = self._get_normalized_features(self.current_step)
        state = self._build_state(normalized_features, self._time_encoding, 0)

        info: Dict[str, Any] = {
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
        }
        return state, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step with action augmentation.

        This is the key method that implements action augmentation:
        it evaluates ALL 3 actions (short/hold/long) at the current state,
        computing rewards and next states for each. This gives 3x more
        training signal per environment step.

        Args:
            action: Selected action (0=-1=short, 1=0=hold, 2=1=long).

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info).
            - next_state: State after taking the selected action.
            - reward: Reward for the selected action.
            - terminated: Whether the episode has ended (no more data).
            - truncated: Always False (no time limit).
            - info: Additional info dict with per-action rewards and states.
        """
        # Convert action index to action value
        action_value = action - 1  # 0->-1, 1->0, 2->1

        # Available actions for augmentation
        actions_list = [-1, 0, 1]

        # Current and next prices
        current_prices = self._prices[self.current_step]
        next_prices = self._prices[self.current_step + 1]
        price_change = (next_prices - current_prices) / (current_prices + 1e-8)
        avg_price_change = np.mean(price_change)

        prev_action = self._action_history[-1] if self._action_history else 0

        # Compute rewards for all 3 actions (action augmentation)
        # This is the NOVEL technique from the original DRQN_Stock_Trading:
        # evaluate all actions, store all transitions
        all_rewards = np.zeros(3, dtype=np.float32)
        all_next_normalized = []
        all_next_action_enc = []

        for i, a in enumerate(actions_list):
            # Position change: new_action - prev_action
            position_delta = a - prev_action

            # Commission based on position change
            commission = self.trade_size * abs(position_delta) * self.spread

            # Portfolio value change for this action
            pnl = (
                a * self.trade_size * avg_price_change
                - commission
            )
            new_portfolio = self.portfolio_value + pnl

            # Reward: log return if profitable, -1 if loss
            if new_portfolio * self.portfolio_value > 0 and self.portfolio_value != 0:
                reward = np.log(new_portfolio / self.portfolio_value)
            else:
                reward = -1.0

            all_rewards[i] = reward

            # Next state components for this action
            next_step = self.current_step + 1
            next_norm = self._get_normalized_features(next_step)
            next_time = self._compute_time_encoding(next_step)

            all_next_normalized.append(next_norm)
            all_next_action_enc.append(self._hot_encode_action(a))

        # Update portfolio with SELECTED action
        selected_idx = action  # 0, 1, or 2
        self.portfolio_value += (
            all_rewards[selected_idx] * self.portfolio_value
        )

        # Append selected action to history
        self._action_history.append(action_value)
        if len(self._action_history) > self.T:
            self._action_history.pop(0)

        # Advance step
        self.current_step += 1
        self._prev_close = next_prices.copy()
        self._portfolio_history.append(self.portfolio_value)
        self._time_encoding = self._compute_time_encoding(self.current_step)

        # Check termination
        terminated = bool(self.current_step >= self._n_total_steps - 2)

        # Build next state for SELECTED action
        next_norm_sel = all_next_normalized[selected_idx]
        next_state = self._build_state(
            next_norm_sel, self._time_encoding, action_value
        )

        # Info includes all action rewards/states for augmentation
        info: Dict[str, Any] = {
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
            "price_change": avg_price_change,
            "all_rewards": all_rewards,  # rewards for all 3 actions
            "all_next_states": [  # next state components for all 3 actions
                {
                    "normalized": all_next_normalized[i],
                    "action_enc": all_next_action_enc[i],
                }
                for i in range(3)
            ],
            "action_value": action_value,  # actual action value (-1,0,1)
        }

        reward = float(all_rewards[selected_idx])
        return next_state, reward, terminated, False, info

    def render(self, mode: str = "human") -> None:
        """Render the environment (no-op for now)."""
        pass

    @property
    def unwrapped(self) -> "DRQNTradingEnv":
        """Return the base environment for Gymnasium compatibility."""
        return self


# Alias for import convenience
TradingEnvironment = DRQNTradingEnv
