"""
MultiStockTradingEnv: Gymnasium environment for multi-asset RL trading.

Adapted from MultiStockRLTrading (D:/Hive/Data/trading_repos/MultiStockRLTrading/).

Observation space: (num_assets, window_size, num_features) tensor
Action space: Box(low=-1, high=1, shape=(num_assets,)) — continuous portfolio weights

Compatible with stable-baselines3 PPO / A2C.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils.seeding import np_random
from sklearn.preprocessing import StandardScaler


# Reasonable defaults (can be overridden via __init__)
DEFAULT_INITIAL_AMOUNT = 1_000_000
DEFAULT_TRADE_COST = 0.0
DEFAULT_REWARD_SCALING = 1e-5
DEFAULT_SUPPRESSION_RATE = 0.66


class MultiStockTradingEnv:
    """
    A Gymnasium-compatible multi-stock trading environment.

    Each step:
    1. Receives continuous portfolio weights (actions) in [-1, 1]^num_assets
    2. Allocates capital proportionally to the absolute magnitude of each weight
    3. Suppresses a fraction of the lowest-confidence positions (suppression_rate)
    4. Executes trades, deducts transaction costs, returns scaled profit as reward

    Observation:
        (num_assets, window_size, num_features) — per-asset feature windows over time.
        Features are typically technical indicators (RSI, MACD, Bollinger Bands, etc.)
        plus price/volume data, scaled per-asset with StandardScaler.

    Key attributes:
        action_space: Box(-1, 1, (num_assets,))  continuous
        observation_space: Box(-inf, inf, (num_assets, window_size, num_features))
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        price_df: pd.DataFrame,
        num_stocks: int,
        num_features: int,
        window_size: int,
        frame_bound: Tuple[int, int],
        initial_amount: float = DEFAULT_INITIAL_AMOUNT,
        trade_cost: float = DEFAULT_TRADE_COST,
        scalers: Optional[List[Optional[StandardScaler]]] = None,
        tech_indicator_list: Optional[List[str]] = None,
        reward_scaling: float = DEFAULT_REWARD_SCALING,
        suppression_rate: float = DEFAULT_SUPPRESSION_RATE,
        representative_col: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        dfs : List[pd.DataFrame]
            One DataFrame per asset, indexed by datetime, containing feature columns.
        price_df : pd.DataFrame
            Close prices for all assets (columns = asset names).
        num_stocks : int
            Number of traded assets.
        num_features : int
            Number of features per asset per timestep.
        window_size : int
            Number of historical timesteps in the observation window.
        frame_bound : Tuple[int, int]
            (start_idx, end_idx) of the price/feature arrays used for the episode.
        initial_amount : float
            Starting cash.
        trade_cost : float
            Fraction of traded notional deducted as cost per trade tick.
        scalers : List[StandardScaler], optional
            Per-asset StandardScaler instances (one per asset). If None, fitted on first reset.
        tech_indicator_list : List[str], optional
            Feature column names used for observation. Defaults to all numeric columns.
        reward_scaling : float
            Scalar applied to raw profit to produce the step reward.
        suppression_rate : float
            Fraction of lowest-confidence positions zeroed each step (0 = no suppression).
        representative_col : str, optional
            Column name in price_df used as market benchmark. Defaults to first column.
        """
        if tech_indicator_list is None:
            tech_indicator_list = []

        self._check_params(num_stocks, num_features, window_size, frame_bound)

        self.dfs = dfs
        self.price_df = price_df
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.initial_amount = float(initial_amount)
        self.trade_cost = float(trade_cost)
        self.reward_scaling = float(reward_scaling)
        self.tech_indicator_list = tech_indicator_list
        self.suppression_rate = float(np.clip(suppression_rate, 0.0, 1.0))
        self.representative_col = representative_col

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_stocks, window_size, num_features), dtype=np.float32
        )

        # Internal state (reset on env.reset())
        self._start_tick: int = window_size
        self._end_tick: int = len(price_df) - 1
        self._current_tick: int = self._start_tick
        self._last_trade_tick: int = self._start_tick - 1
        self._position: np.ndarray = np.zeros(num_stocks, dtype=np.float64)
        self._position_history: List[np.ndarray] = []
        self._total_reward: float = 0.0
        self._total_profit: float = 1.0
        self._done: bool = False
        self.margin: float = self.initial_amount
        self.reserve: float = self.initial_amount
        self.portfolio: List[int] = [0] * num_stocks
        self.PortfolioValue: float = 0.0
        self.history: Dict[str, List] = {}
        self.rewards: List[float] = []
        self.pvs: List[float] = []

        # Per-asset scalers
        if scalers is None:
            self.scalers: List[Optional[StandardScaler]] = [None] * num_stocks
        else:
            self.scalers = scalers

        # Pre-process data so reset() is fast
        self.prices: Optional[np.ndarray] = None
        self.signal_features: Optional[np.ndarray] = None
        self.representative: Optional[np.ndarray] = None
        self._processed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed."""
        self.np_random, seed = np_random(seed)
        return [seed]

    def process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features per asset, cache prices.
        Called automatically by reset() but can be called explicitly beforehand.
        """
        signal_features: List[np.ndarray] = []
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        for i in range(self.num_stocks):
            df = self.dfs[i]
            if self.scalers[i] is not None:
                current_scaler = self.scalers[i]
                feat_i = current_scaler.transform(df.loc[:, self.tech_indicator_list])[start:end]
            else:
                current_scaler = StandardScaler()
                feat_i = current_scaler.fit_transform(df.loc[:, self.tech_indicator_list])[start:end]
                self.scalers[i] = current_scaler
            signal_features.append(feat_i)

        self.prices = self.price_df.loc[:, :].to_numpy()[start:end]
        if self.representative_col:
            self.representative = self.price_df.loc[:, self.representative_col].to_numpy()[start:end]
        else:
            self.representative = self.prices[:, 0]  # fallback: first column

        self.signal_features = np.array(signal_features, dtype=np.float32)
        self._end_tick = len(self.prices) - 1
        self._processed = True
        return self.prices, self.signal_features

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state and return the first observation."""
        if seed is not None:
            self.np_random, _ = np_random(seed)

        if not self._processed:
            self.process_data()

        self._done = False
        self._current_tick = self._start_tick
        self._end_tick = len(self.prices) - 1
        self._last_trade_tick = self._current_tick - 1
        self._position = np.zeros(self.num_stocks, dtype=np.float64)
        self._position_history = [np.zeros(self.num_stocks, dtype=np.float64)] * (self.window_size - 1) + [self._position]
        self.margin = float(self.initial_amount)
        self.reserve = float(self.initial_amount)
        self.portfolio = [0] * self.num_stocks
        self.PortfolioValue = 0.0
        self._total_reward = 0.0
        self._total_profit = 1.0
        self.history = {}
        self.rewards = []
        self.pvs = []

        obs = self._get_observation()
        info: Dict = {}
        return obs, info

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Parameters
        ----------
        actions : array-like, shape (num_assets,)
            Continuous portfolio weights in [-1, 1].

        Returns
        -------
        observation : np.ndarray
            New observation.
        reward : float
            Scaled step profit.
        terminated : bool
            True if episode ended (end of data or margin < 0).
        truncated : bool
            Always False (no horizon truncation in this env).
        info : dict
            Diagnostic info (total_reward, total_profit).
        """
        self._done = False
        self._current_tick += 1

        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_stocks,):
            raise ValueError(f"Expected action shape {(self.num_stocks,)}, got {actions.shape}")

        if self._current_tick >= self._end_tick:
            self._done = True

        current_prices = self.prices[self._current_tick].copy()
        current_prices[np.isnan(current_prices)] = 0.0
        divisor = current_prices.copy()
        divisor[divisor == 0] = 1e9

        # Absolute portfolio distribution
        abs_portfolio_dist = np.abs(actions)

        # Suppression: zero out the lowest-confidence positions
        num_to_suppress = int(np.floor(abs_portfolio_dist.size * self.suppression_rate))
        num_to_suppress = min(num_to_suppress, abs_portfolio_dist.size - 1)
        if num_to_suppress > 0:
            threshold_idx = np.argpartition(abs_portfolio_dist, kth=num_to_suppress - 1)[num_to_suppress - 1]
            abs_portfolio_dist[abs_portfolio_dist <= threshold_idx] = 0

        # Available margin
        self.margin = self.reserve + float(np.sum(self.portfolio * current_prices))

        # Normalize positions
        pos_sum = float(np.sum(abs_portfolio_dist))
        if pos_sum <= 0:
            norm_margin_pos = np.zeros_like(abs_portfolio_dist)
        else:
            norm_margin_pos = (abs_portfolio_dist / pos_sum) * self.margin

        # Desired next positions (in currency units)
        next_positions = np.sign(actions) * norm_margin_pos
        change_in_positions = next_positions - self._position

        # Convert to share counts
        actions_in_market = np.divide(change_in_positions, divisor).astype(int)
        new_portfolio = np.asarray(self.portfolio, dtype=np.int64) + actions_in_market

        new_pv = float(np.sum(new_portfolio * current_prices))
        new_reserve = self.margin - new_pv

        # Transaction cost
        cost = self.trade_cost * float(np.sum(np.abs(np.sign(actions_in_market))))

        self._position = next_positions
        self.portfolio = new_portfolio.tolist()
        self.PortfolioValue = new_pv
        self.reserve = new_reserve - cost

        # Step reward
        step_reward = (new_pv + new_reserve) - (self.PortfolioValue + self.reserve - cost)
        step_reward = step_reward - cost
        self._total_reward += self.reward_scaling * step_reward
        self._total_profit = (self.PortfolioValue + self.reserve) / self.initial_amount

        self.rewards.append(self._total_reward)
        self.pvs.append(self.PortfolioValue)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
        }
        self._update_history(info)

        if self.margin < 0:
            self._done = True

        # gymnasium returns 5 values: obs, reward, terminated, truncated, info
        return observation, float(step_reward), self._done, False, info

    def render(self) -> None:
        """No-op placeholder — matplotlib plotting removed for clean dependency-free rendering."""
        pass

    def close(self) -> None:
        """No-op placeholder."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        t = self._current_tick
        w = self.window_size
        obs = self.signal_features[:, (t - w + 1) : t + 1, :]
        return np.nan_to_num(obs, nan=0.0, posinf=1.0).astype(np.float32)

    def _update_history(self, info: Dict) -> None:
        if not self.history:
            self.history = {key: [] for key in info}
        for key, value in info.items():
            self.history[key].append(value)

    def _check_params(
        self,
        num_stocks: int,
        num_features: int,
        window_size: int,
        frame_bound: Tuple[int, int],
    ) -> None:
        if num_stocks <= 0:
            raise ValueError(f"num_stocks must be positive, got {num_stocks}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if len(frame_bound) != 2:
            raise ValueError(f"frame_bound must be (start, end), got {frame_bound}")
        if frame_bound[0] >= frame_bound[1]:
            raise ValueError(f"frame_bound[0] ({frame_bound[0]}) must be < frame_bound[1] ({frame_bound[1]})")

    # ------------------------------------------------------------------
    # Not-implemented stubs (kept for interface compatibility)
    # ------------------------------------------------------------------

    def _process_data(self) -> None:
        raise NotImplementedError("process_data() is called automatically by reset()")

    def _calculate_reward(self, action) -> float:
        raise NotImplementedError()

    def max_possible_profit(self) -> float:
        raise NotImplementedError()
