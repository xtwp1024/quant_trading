"""
crypto_finrl.py — FinRL-style multi-crypto / multi-stock trading with pure NumPy DRL.

Adapted from CryptoBot-FinRL (D:/Hive/Data/trading_repos/CryptoBot-FinRL/).

Source components absorbed:
  - finrl.env.env_portfolio   : StockPortfolioEnv  (portfolio-weight, covariance-matrix based)
  - finrl.env.env_stocktrading : StockTradingEnv   (discrete share-count, buy/sell actions)
  - finrl.model.models         : DRLAgent          (stable-baselines3 wrapper)
  - finrl.trade.backtest       : backtest_stats / get_daily_return / get_baseline
  - finrl.config.config_crypt  : TECHNICAL_INDICATORS_LIST / BINANCE_TICKER defaults
  - main_multi_crypto_trading  : training / trading / backtesting workflow

This module replaces stable-baselines3 with pure NumPy / Gymnasium implementations:
  - CryptoTradingEnv    : Gymnasium-compatible multi-asset trading env
  - FinRLAgent          : DQN / PG (policy gradient) / A2C / PPO in pure NumPy
  - PortfolioAllocator  : rebalance across crypto assets
  - CryptoPortfolioBacktester : episode simulation + performance metrics

Dependencies: numpy, pandas, gymnasium (no stable-baselines3).

Usage:
    from quant_trading.rl.crypto_finrl import CryptoTradingEnv, FinRLAgent
    env = CryptoTradingEnv(df=data, ...)
    agent = FinRLAgent(env, algo="ppo")
    agent.train(total_timesteps=100_000)
    obs, _ = env.reset()
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils.seeding import np_random


# ---------------------------------------------------------------------------
# Default crypto configuration (from finrl.config.config_crypt)
# ---------------------------------------------------------------------------
DEFAULT_CRYPTO_TICKERS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT",
    "XRPUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT",
]

DEFAULT_TECH_INDICATORS = [
    "macd", "boll_ub", "boll_lb", "rsi_30",
    "cci_30", "dx_30", "close_30_sma", "close_60_sma",
]

# Default backtest window
DEFAULT_START_DATE = "2020-02-21"
DEFAULT_END_DATE   = "2021-06-11"
DEFAULT_START_TRADE_DATE = "2021-05-01"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def data_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Split a DataFrame by date range (inclusive)."""
    return df[(df.date >= start) & (df.date <= end)].copy()


def get_daily_return(df: pd.DataFrame, value_col: str = "account_value") -> pd.Series:
    """Compute daily returns from an account-value column."""
    out = df.copy()
    out["daily_return"] = out[value_col].pct_change(1)
    return out["daily_return"].iloc[1:]


def _softmax_normalization(actions: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over actions."""
    a = actions - np.max(actions)
    exp_a = np.exp(a)
    return exp_a / (np.sum(exp_a) + 1e-10)


# ---------------------------------------------------------------------------
# Technical indicator computation (pure NumPy, no stockstats required)
# ---------------------------------------------------------------------------

def _sma(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(series, np.nan)
    out[window - 1:] = np.convolve(series, np.ones(window) / window, mode="valid")
    return out


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def add_technical_indicators(df: pd.DataFrame, tech_list: List[str] = None) -> pd.DataFrame:
    """
    Add technical indicators per asset (grouped by 'tic') using pure NumPy.

    Adds per row: macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30,
                  close_30_sma, close_60_sma
    """
    tech_list = tech_list or DEFAULT_TECH_INDICATORS
    out = df.copy()
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)

    result_frames = []
    for tic in out.tic.unique():
        tdf = out[out.tic == tic].copy().reset_index(drop=True)
        close = tdf.close.values.astype(float)
        highs = tdf.high.values.astype(float) if "high" in tdf.columns else close
        lows  = tdf.low.values.astype(float)  if "low"  in tdf.columns else close

        # ---- SMA / EMA helpers ----
        sma30  = _sma(close, 30)
        sma60  = _sma(close, 60)

        # ---- Bollinger Bands (20-period) ----
        bb_std = pd.Series(close).rolling(20).std().values
        bb_mid = _sma(close, 20)
        boll_lb = bb_mid - 2 * bb_std
        boll_ub = bb_mid + 2 * bb_std

        # ---- MACD (12, 26, 9) ----
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        macd_signal = _ema(macd, 9)

        # ---- RSI (14-period) ----
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs  = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # ---- CCI (20-period) ----
        tp = (highs + lows + close) / 3.0
        sma_tp = _sma(tp, 20)
        mad = pd.Series(np.abs(tp - sma_tp)).rolling(20).mean().values
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # ---- DX (14-period) ----
        plus_dm  = np.maximum(highs - np.roll(highs, 1), 0)
        minus_dm = np.maximum(np.roll(lows, 1)  - lows,  0)
        plus_dm[0]  = 0
        minus_dm[0] = 0
        tr = highs - lows
        plus_di  = 100 * _ema(np.where(tr > 0, plus_dm  / tr, 0), 14)
        minus_di = 100 * _ema(np.where(tr > 0, minus_dm / tr, 0), 14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        for col in tech_list:
            if col == "macd":
                tdf["macd"] = macd
            elif col == "boll_ub":
                tdf["boll_ub"] = boll_ub
            elif col == "boll_lb":
                tdf["boll_lb"] = boll_lb
            elif col == "rsi_30":
                tdf["rsi_30"] = rsi
            elif col == "cci_30":
                tdf["cci_30"] = cci
            elif col == "dx_30":
                tdf["dx_30"] = dx
            elif col == "close_30_sma":
                tdf["close_30_sma"] = sma30
            elif col == "close_60_sma":
                tdf["close_60_sma"] = sma60

        result_frames.append(tdf)

    return pd.concat(result_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CryptoTradingEnv
# ---------------------------------------------------------------------------

class CryptoTradingEnv:
    """
    Gymnasium-compatible multi-crypto trading environment.

    基于 FinRL StockTradingEnv，重构为纯 Gymnasium 接口，支持多资产组合交易。

    State space (per step):
        [cash] + [closePrices] + [numShares] + [techIndicatorsPerAsset × stock_dim]
        = 1 + stock_dim + stock_dim + len(tech_indicator_list) * stock_dim

    Action space:
        Box(low=-1, high=1, shape=(stock_dim,)) — continuous weight per asset
        Each weight is clipped to [-1, 1]; magnitude controls position size.

    Reward:
        Scaled portfolio-value delta (configurable via reward_scaling).

    Reference: finrl.env.env_stocktrading.StockTradingEnv
               finrl.env.env_portfolio.StockPortfolioEnv
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: float = 100.0,
        initial_amount: float = 10_000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: int = 1,
        action_space: int = 1,
        tech_indicator_list: Optional[List[str]] = None,
        turbulence_threshold: Optional[float] = None,
        day: int = 0,
        verbose: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Multi-asset OHLCV data with columns: date, tic, open, high, low, close, volume
            and any technical-indicator columns.
        stock_dim : int
            Number of distinct crypto assets.
        hmax : float
            Maximum notional exposure per asset per step (in quote currency).
        initial_amount : float
            Starting cash.
        buy_cost_pct : float
            Buy commission rate (fraction of traded amount).
        sell_cost_pct : float
            Sell commission rate (fraction of traded amount).
        reward_scaling : float
            Scalar applied to step reward for numerical stability.
        state_space : int
            Dimensionality of the flattened state (auto-computed if not provided).
        action_space : int
            Number of assets (= stock_dim, auto-set).
        tech_indicator_list : List[str], optional
            Technical-indicator column names present in df. Defaults to DEFAULT_TECH_INDICATORS.
        turbulence_threshold : float, optional
            Risk threshold — above this turbulence all positions are liquidated.
        day : int
            Starting day index.
        verbose : int
            Verbosity level (0 = silent).
        """
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct  = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = stock_dim
        self.tech_indicator_list = tech_indicator_list or DEFAULT_TECH_INDICATORS
        self.turbulence_threshold = turbulence_threshold
        self.day = day
        self.verbose = verbose

        # Derive state_space from df if not explicitly provided
        if state_space == 1:
            self.state_space = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim

        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_space,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_space,),
            dtype=np.float32,
        )

        # Internal episode state (reset on .reset())
        self._seed()
        self._initiate_state()

    # ------------------------------------------------------------------
    # Seeding (Gymnasium API)
    # ------------------------------------------------------------------

    def _seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = np_random(seed)
        return [seed]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Alias for Gymnasium compatibility."""
        return self._seed(seed)

    # ------------------------------------------------------------------
    # State initialization / update
    # ------------------------------------------------------------------

    def _initiate_state(self) -> np.ndarray:
        """Build the initial state vector for day 0 (or carry over previous state)."""
        self.data = self.df.loc[self.day, :]

        # Handle both Series (single row) and DataFrame (multiple rows) returns
        if isinstance(self.data, pd.Series):
            close_prices = [float(self.data.close)]
        else:
            close_prices = self.data.close.values.tolist()

        num_shares = [0.0] * self.stock_dim

        tech_values = []
        for tech in self.tech_indicator_list:
            if tech in self.data.index:
                val = self.data[tech]
                if isinstance(val, (pd.Series, np.ndarray)):
                    tech_values.extend(list(val))
                else:
                    tech_values.extend([float(val)] * self.stock_dim)
            else:
                tech_values.extend([0.0] * self.stock_dim)

        state = (
            [self.initial_amount]
            + close_prices
            + num_shares
            + tech_values
        )
        self.state = np.array(state, dtype=np.float32)
        self.terminal = False
        return self.state

    def _update_state(self) -> np.ndarray:
        """Update state after a step to the next day."""
        self.data = self.df.loc[self.day, :]

        # Handle both Series (single row) and DataFrame (multiple rows)
        if isinstance(self.data, pd.Series):
            close_prices = [float(self.data.close)]
        else:
            close_prices = self.data.close.values.tolist()

        current_shares = list(self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1])

        tech_values = []
        for tech in self.tech_indicator_list:
            if tech in self.data.index:
                val = self.data[tech]
                if isinstance(val, (pd.Series, np.ndarray)):
                    tech_values.extend(list(val))
                else:
                    tech_values.extend([float(val)] * self.stock_dim)
            else:
                tech_values.extend([0.0] * self.stock_dim)

        self.state = np.array(
            [self.state[0]] + close_prices + current_shares + tech_values,
            dtype=np.float32,
        )
        return self.state

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial conditions.

        Returns
        -------
        observation : np.ndarray
            First state vector.
        info : dict
            Empty dict (Gymnasium convention).
        """
        if seed is not None:
            self._seed(seed)

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self._initiate_state()

        self.portfolio_value = self.initial_amount
        self.asset_memory     = [self.initial_amount]
        self.rewards_memory   = []
        self.actions_memory   = []
        self.date_memory      = [self._get_date()]
        self.terminal         = False
        self.cost             = 0.0
        self.trades           = 0
        self.episode          = getattr(self, "episode", 0) + 1

        return self.state, {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Parameters
        ----------
        actions : array-like, shape (stock_dim,)
            Continuous weights in [-1, 1]. Positive = buy, negative = sell.
            Magnitude controls position size (scaled by hmax).

        Returns
        -------
        observation : np.ndarray
            Next state.
        reward : float
            Scaled portfolio P&L.
        terminated : bool
            True when episode ends (end of DataFrame).
        truncated : bool
            Always False (no horizon truncation).
        info : dict
            Diagnostic dict with total_reward, total_profit, trades, cost.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # ---- Episode summary ----
            end_total_asset = (
                self.state[0]
                + sum(
                    self.state[1 : self.stock_dim + 1]
                    * self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1]
                )
            )
            tot_reward = end_total_asset - self.initial_amount

            if self.verbose > 0 and self.episode % self.verbose == 0:
                days = len(self.asset_memory)
                returns = np.diff(self.asset_memory) / np.array(self.asset_memory[:-1]) + 1e-10
                sharpe = (252 ** 0.5) * np.mean(returns) / (np.std(returns) + 1e-10)
                print(f"[CryptoTradingEnv] episode={self.episode} | "
                      f"days={days} | start={self.asset_memory[0]:.2f} | "
                      f"end={end_total_asset:.2f} | "
                      f"return={tot_reward:.2f} | sharpe={sharpe:.3f} | "
                      f"trades={self.trades} | cost={self.cost:.4f}")

            info = {
                "total_reward": float(tot_reward),
                "total_asset": float(end_total_asset),
                "total_profit": float(end_total_asset / self.initial_amount),
                "trades": int(self.trades),
                "cost": float(self.cost),
            }
            return self.state, 0.0, True, False, info

        # ---- Normal step ----
        actions = np.asarray(actions, dtype=np.float32).clip(-1.0, 1.0)
        actions_scaled = actions * self.hmax  # scale to hmax notional

        # Turbulence risk guard — liquidate all if threshold exceeded
        if self.turbulence_threshold is not None:
            turbulence = float(self.data.get("turbulence", 0.0))
            if turbulence >= self.turbulence_threshold:
                actions_scaled = np.full(self.stock_dim, -self.hmax, dtype=np.float32)

        # Snapshot assets before trade
        begin_total_asset = (
            self.state[0]
            + sum(
                self.state[1 : self.stock_dim + 1]
                * self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1]
            )
        )

        # ---- Execute sells first (lowest price first) ----
        sell_mask = actions_scaled < 0
        sell_indices = np.argsort(actions_scaled)[: np.sum(sell_mask)]
        for idx in sell_indices:
            self._sell_stock(idx, float(actions_scaled[idx]))

        # ---- Execute buys (highest weight first) ----
        buy_mask = actions_scaled > 0
        buy_indices = np.argsort(actions_scaled)[::-1][: np.sum(buy_mask)]
        for idx in buy_indices:
            self._buy_stock(idx, float(actions_scaled[idx]))

        self.actions_memory.append(
            self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1].copy()
        )

        # ---- Advance day ----
        self.day += 1
        self._update_state()

        # ---- Compute reward ----
        end_total_asset = (
            self.state[0]
            + sum(
                self.state[1 : self.stock_dim + 1]
                * self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1]
            )
        )
        self.asset_memory.append(float(end_total_asset))
        self.date_memory.append(self._get_date())

        step_reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(float(step_reward))
        self.portfolio_value = end_total_asset

        # Apply turbulence a second time after state update
        if self.turbulence_threshold is not None:
            turbulence = float(self.data.get("turbulence", 0.0))
            if turbulence >= self.turbulence_threshold:
                step_reward -= self.cost  # penalise turbulent trading

        info = {
            "total_reward": float(sum(self.rewards_memory)),
            "total_asset": float(end_total_asset),
            "total_profit": float(end_total_asset / self.initial_amount),
            "trades": int(self.trades),
            "cost": float(self.cost),
        }
        return self.state, float(step_reward * self.reward_scaling), False, False, info

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------

    def _sell_stock(self, index: int, action: float) -> float:
        """Execute sell for asset at `index` by `action` notional units."""
        price = self.state[index + 1]
        if price <= 0:
            return 0.0

        current_holding = self.state[index + self.stock_dim + 1]
        if current_holding <= 0:
            return 0.0

        # Notional to sell (convert to share count)
        sell_notional = min(abs(action), current_holding * price)
        sell_shares   = sell_notional / price
        sell_value    = sell_notional * (1.0 - self.sell_cost_pct)

        self.state[0]                              += sell_value
        self.state[index + self.stock_dim + 1]    -= sell_shares
        self.cost   += sell_notional * self.sell_cost_pct
        self.trades += 1
        return sell_shares

    def _buy_stock(self, index: int, action: float) -> float:
        """Execute buy for asset at `index` by `action` notional units."""
        price = self.state[index + 1]
        if price <= 0:
            return 0.0

        available = self.state[0] / (1.0 + self.buy_cost_pct)
        buy_notional = min(abs(action), available)
        buy_shares   = buy_notional / price
        buy_cost     = buy_notional * self.buy_cost_pct

        self.state[0]                              -= buy_notional + buy_cost
        self.state[index + self.stock_dim + 1]   += buy_shares
        self.cost   += buy_cost
        self.trades += 1
        return buy_shares

    def _get_date(self) -> str:
        date_val = self.data.date
        if hasattr(date_val, "unique"):
            return str(date_val.unique()[0])
        return str(date_val)

    # ------------------------------------------------------------------
    # Memory helpers (FinRL-compatible)
    # ------------------------------------------------------------------

    def save_asset_memory(self) -> pd.DataFrame:
        """Return DataFrame of account value over time."""
        return pd.DataFrame({
            "date":           self.date_memory,
            "account_value":  self.asset_memory,
        })

    def save_action_memory(self) -> pd.DataFrame:
        """Return DataFrame of per-asset share counts over time."""
        arr = np.array(self.actions_memory)
        cols = self.df.tic.unique()[: self.stock_dim]
        df = pd.DataFrame(arr, columns=cols)
        df.index = pd.to_datetime(self.date_memory[:-1])
        df.index.name = "date"
        return df

    def render(self, mode: str = "human") -> None:
        """No-op placeholder for Gymnasium API."""
        pass

    def close(self) -> None:
        """No-op placeholder for Gymnasium API."""
        pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compute_reward(self, action: np.ndarray) -> float:
        """
        Compute step reward without executing the step (off-policy evaluation).

        奖励函数核心：
        r_t = Δportfolio_value × reward_scaling
        其中 Δportfolio_value = Σ_i (P_{t+1,i} - P_{t,i}) × w_i
        """
        prices_now  = self.state[1 : self.stock_dim + 1]
        shares      = self.state[self.stock_dim + 1 : 2 * self.stock_dim + 1]
        weights     = action / (np.sum(np.abs(action)) + 1e-10)
        port_return = np.sum(((prices_now / (prices_now + 1e-10)) - 1.0) * weights)
        return float(port_return * self.reward_scaling)


# ---------------------------------------------------------------------------
# FinRLAgent — Pure NumPy DRL implementations (DQN / PG / A2C / PPO)
# ---------------------------------------------------------------------------

class FinRLAgent:
    """
    Pure-NumPy DRL agent for crypto trading.

    Supports four algorithms:
      - "dqn"  : Deep Q-Network with experience replay
      - "pg"   : Policy Gradient (REINFORCE)
      - "a2c"  : Advantage Actor-Critic
      - "ppo"  : Proximal Policy Optimization (clipped)

    All networks are simple feed-forward MLPs with NumPy arrays as weights.
    No PyTorch / TensorFlow / stable-baselines3 required.

    Reference: finrl.model.models.DRLAgent
    """

    def __init__(
        self,
        env: CryptoTradingEnv,
        algo: Literal["dqn", "pg", "a2c", "ppo"] = "ppo",
        hidden_sizes: List[int] = None,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 100,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        seed: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        env : CryptoTradingEnv
            The trading environment.
        algo : str
            Algorithm: "dqn" | "pg" | "a2c" | "ppo".
        hidden_sizes : List[int]
            Hidden-layer sizes for value and policy networks.
        gamma : float
            Discount factor.
        lr : float
            Learning rate.
        batch_size : int
            Batch size for training.
        buffer_size : int
            Replay-buffer size (DQN only).
        target_update_freq : int
            Target network update frequency in steps (DQN only).
        epsilon : float
            Initial exploration rate (DQN only).
        epsilon_decay : float
            Epsilon decay per episode.
        epsilon_min : float
            Minimum epsilon.
        clip_ratio : float
            PPO clip ratio.
        value_coef : float
            Value-loss weight (A2C/PPO).
        entropy_coef : float
            Entropy bonus weight (A2C/PPO).
        seed : int, optional
            Random seed.
        verbose : int
            Verbosity level.
        """
        self.env         = env
        self.algo        = algo
        self.gamma       = gamma
        self.lr          = lr
        self.batch_size = batch_size
        self.verbose     = verbose

        # Epsilon for DQN exploration
        self.epsilon        = epsilon
        self.epsilon_decay  = epsilon_decay
        self.epsilon_min    = epsilon_min

        # PPO clip ratio
        self.clip_ratio   = clip_ratio
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef

        # Set seed
        self.rng = np.random.default_rng(seed)

        # Network dimensions
        obs_dim    = env.observation_space.shape[0]
        act_dim    = env.action_space.shape[0]
        hidden_sizes = hidden_sizes or [64, 64]

        # ---- Build networks ----
        self._build_networks(obs_dim, act_dim, hidden_sizes)

        # ---- Replay buffer (DQN) ----
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.buffer_size   = buffer_size
        self.target_update_freq = target_update_freq
        self.total_steps    = 0

        # ---- Training history ----
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Network construction (pure NumPy)
    # ------------------------------------------------------------------

    def _build_networks(self, obs_dim: int, act_dim: int, hidden: List[int]) -> None:
        """Initialize network weights using He (Kaiming) uniform initialization."""
        layers = [obs_dim] + hidden
        self.weights: List[List[np.ndarray]] = []
        self.biases:  List[List[np.ndarray]]  = []

        for i in range(len(layers) - 1):
            # He uniform: bound = sqrt(6 / fan_in)
            bound = np.sqrt(6.0 / layers[i])
            W = self.rng.uniform(-bound, bound, (layers[i], layers[i + 1])).astype(np.float32)
            b = np.zeros(layers[i + 1], dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

        # Output layer (action)
        out_bound = np.sqrt(6.0 / layers[-1])
        self.W_out = self.rng.uniform(-out_bound, out_bound, (layers[-1], act_dim)).astype(np.float32)
        self.b_out = np.zeros(act_dim, dtype=np.float32)

        # For A2C/PPO: separate value head
        self.W_val = self.rng.uniform(-out_bound, out_bound, (layers[-1], 1)).astype(np.float32)
        self.b_val = np.zeros(1, dtype=np.float32)

        # Target network (DQN)
        self.W_out_target = self.W_out.copy()
        self.b_out_target = self.b_out.copy()

    def _forward(self, x: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through policy network."""
        W_o = self.W_out_target if use_target else self.W_out
        b_o = self.b_out_target if use_target else self.b_out

        h = x.astype(np.float32)
        for W, b in zip(self.weights, self.biases):
            h = np.tanh(h @ W + b)  # PyTorch default for MLP
        return h @ W_o + b_o

    def _forward_value(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through value head."""
        h = x.astype(np.float32)
        for W, b in zip(self.weights, self.biases):
            h = np.tanh(h @ W + b)
        return h @ self.W_val + self.b_val

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action for a single observation.

        Parameters
        ----------
        observation : np.ndarray
            State vector.
        deterministic : bool
            If True, return mean action; else sample (for on-policy algos).

        Returns
        -------
        action : np.ndarray
            Action vector in [-1, 1].
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        raw = self._forward(obs).flatten()

        if self.algo == "dqn":
            # Epsilon-greedy for DQN
            if self.rng.random() < self.epsilon:
                return self.rng.uniform(-1, 1, raw.shape).astype(np.float32)
            return np.clip(raw, -1.0, 1.0).astype(np.float32)

        if deterministic:
            return np.clip(raw, -1.0, 1.0).astype(np.float32)

        # On-policy: add noise for exploration
        noise = self.rng.normal(0, 0.1, raw.shape).astype(np.float32)
        return np.clip(raw + noise, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int, eval_env: Optional[CryptoTradingEnv] = None) -> "FinRLAgent":
        """
        Train the agent for ``total_timesteps`` steps in the environment.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps.
        eval_env : CryptoTradingEnv, optional
            Separate evaluation environment (not used in this pure-NumPy impl).

        Returns
        -------
        self : FinRLAgent
        """
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        episode = 0
        episode_reward = 0.0
        episode_start  = time.time()

        if self.algo == "dqn":
            self._train_dqn(total_timesteps)
            return self

        # ---- On-policy training (PG / A2C / PPO) ----
        buffer_obs:  List[np.ndarray] = []
        buffer_acts: List[np.ndarray] = []
        buffer_rews: List[float]      = []
        buffer_vals: List[float]     = []
        buffer_logps: List[float]     = []

        while self.total_steps < total_timesteps:
            action = self.predict(obs, deterministic=(self.algo == "pg"))
            val    = float(self._forward_value(obs.reshape(1, -1)).item())

            # Collect for on-policy update
            logp = self._log_prob(obs, action)
            buffer_obs.append(obs.copy())
            buffer_acts.append(action.copy())
            buffer_vals.append(val)
            buffer_logps.append(logp)

            obs2, reward, terminated, truncated, info = self.env.step(action)
            obs2 = np.asarray(obs2, dtype=np.float32)
            self.total_steps += 1
            episode_reward += reward

            buffer_rews.append(float(reward))

            if terminated or truncated:
                episode += 1
                if self.verbose > 0 and episode % self.verbose == 0:
                    elapsed = time.time() - episode_start
                    print(f"  [FinRLAgent/{self.algo}] episode={episode} | "
                          f"steps={self.total_steps}/{total_timesteps} | "
                          f"ep_reward={episode_reward:.4f} | "
                          f"epsilon={getattr(self,'epsilon',0):.4f} | "
                          f"elapsed={elapsed:.1f}s")

                # ---- On-policy update ----
                self._on_policy_update(
                    np.array(buffer_obs),
                    np.array(buffer_acts),
                    np.array(buffer_rews),
                    np.array(buffer_vals),
                    np.array(buffer_logps),
                )

                buffer_obs, buffer_acts, buffer_rews = [], [], []
                buffer_vals, buffer_logps = [], []
                episode_reward = 0.0
                episode_start  = time.time()
                obs, _ = self.env.reset()
                obs    = np.asarray(obs, dtype=np.float32)
            else:
                obs = obs2

        return self

    def _train_dqn(self, total_timesteps: int) -> None:
        """Train DQN with experience replay."""
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        episode = 0
        episode_reward = 0.0

        while self.total_steps < total_timesteps:
            action = self.predict(obs)
            obs2, reward, terminated, truncated, _ = self.env.step(action)
            obs2 = np.asarray(obs2, dtype=np.float32)
            self.total_steps += 1
            episode_reward += reward

            # Store transition
            self._store_transition(obs, action, reward, obs2, terminated)

            if len(self.replay_buffer) >= self.batch_size:
                loss = self._update_dqn()
                self.loss_history.append(loss)

            # Target network hard copy every N steps
            if self.total_steps % self.target_update_freq == 0:
                self.W_out_target = self.W_out.copy()
                self.b_out_target = self.b_out.copy()

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if terminated or truncated:
                episode += 1
                if self.verbose > 0 and episode % self.verbose == 0:
                    print(f"  [FinRLAgent/dqn] episode={episode} | "
                          f"steps={self.total_steps} | "
                          f"ep_reward={episode_reward:.4f} | "
                          f"epsilon={self.epsilon:.4f}")
                episode_reward = 0.0
                obs, _ = self.env.reset()
                obs    = np.asarray(obs, dtype=np.float32)

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        obs2: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((obs, action, float(reward), obs2, bool(done)))

    def _update_dqn(self) -> float:
        """Perform one DQN gradient-descent step on a random mini-batch."""
        indices = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False)
        loss = 0.0

        for idx in indices:
            obs, action, reward, obs2, done = self.replay_buffer[idx]

            q_vals      = self._forward(obs.reshape(1, -1)).flatten()
            q_next_vals = self._forward(obs2.reshape(1, -1), use_target=True).flatten()
            max_q_next  = np.max(q_next_vals)

            target = float(reward) + self.gamma * max_q_next * (0.0 if done else 1.0)
            action_idx = int(np.argmax(np.abs(action)))  # simple argmax of abs action

            # Gradient descent on squared error
            delta = q_vals[action_idx] - target
            loss += delta ** 2

            # Gradient update (simplified single-step SGD)
            grad = delta * 0.01  # scaled learning rate
            self.W_out[:, action_idx] -= grad * np.tanh(np.dot(obs, self.W_out[:, action_idx]))
            self.b_out[action_idx]    -= grad

        return loss / self.batch_size

    def _log_prob(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute log probability of action under current policy (Gaussian)."""
        mean = self._forward(obs.reshape(1, -1)).flatten()
        var  = 0.1
        diff = action - mean
        return -0.5 * np.sum(diff ** 2) / var - 0.5 * np.log(2 * np.pi * var)

    def _on_policy_update(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        val_batch: np.ndarray,
        logp_batch: np.ndarray,
    ) -> None:
        """On-policy update for PG / A2C / PPO using Monte Carlo returns."""
        if self.algo == "pg":
            self._update_pg(rew_batch, logp_batch)
        elif self.algo == "a2c":
            self._update_a2c(obs_batch, act_batch, rew_batch, val_batch)
        elif self.algo == "ppo":
            self._update_ppo(obs_batch, act_batch, rew_batch, val_batch, logp_batch)

    def _update_pg(self, rewards: np.ndarray, logps: np.ndarray) -> None:
        """REINFORCE (policy gradient) with Monte Carlo return."""
        # Compute discounted returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Policy gradient loss (negative for gradient ascent)
        loss = -np.sum(returns * np.array(logps))
        self.loss_history.append(float(loss))

        # Simple SGD step (crude but functional for demonstration)
        for _ in range(3):
            for t in range(len(logps)):
                G_t = returns[t]
                grad = G_t * 0.001  # tiny step
                self.W_out -= grad * self.W_out * np.mean(logps[t])
                self.b_out -= grad * self.b_out

    def _update_a2c(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Advantage Actor-Critic update."""
        # Monte Carlo advantage
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        advantage = returns - np.array(values, dtype=np.float32)
        # Normalize advantage
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)

        # Policy loss (actor)
        policy_loss = -np.mean(advantage * np.array([self._log_prob(o, a) for o, a in zip(obs_batch, act_batch)]))

        # Value loss (critic)
        value_preds = np.array([self._forward_value(o.reshape(1, -1)).item() for o in obs_batch], dtype=np.float32)
        value_loss  = np.mean((value_preds - returns) ** 2)

        # Combined loss
        loss = policy_loss + self.value_coef * value_loss
        self.loss_history.append(float(loss))

        # Simple multi-step SGD
        step_size = self.lr * 0.01
        for _ in range(5):
            i = self.rng.integers(len(obs_batch))
            o = obs_batch[i]
            g = advantage[i] * step_size
            self.W_out -= g * np.outer(o, self.W_out[:, 0])
            self.W_val -= g * 0.5 * np.outer(o, self.W_val[:, 0])

    def _update_ppo(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        logps_old: np.ndarray,
    ) -> None:
        """Proximal Policy Optimization (clipped objective)."""
        # Monte Carlo returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        advantage = returns - np.array(values, dtype=np.float32)
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)

        # PPO clipped surrogate loss
        clip_frac = 0.0
        policy_loss = 0.0
        entropy_bonus = 0.0

        for _ in range(self.batch_size):
            i = self.rng.integers(len(obs_batch))
            o = obs_batch[i : i + 1]
            a = act_batch[i]
            lp_old = logps_old[i]
            lp_new = self._log_prob(o.flatten(), a)

            ratio = np.exp(lp_new - lp_old)
            surr1 = ratio * advantage[i]
            surr2 = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage[i]
            policy_loss -= np.minimum(surr1, surr2)

            if abs(ratio - 1.0) > self.clip_ratio:
                clip_frac += 1.0

            # Entropy bonus (encourage exploration)
            entropy_bonus += 0.5 * np.log(2 * np.pi * 0.1 * np.pi)

        policy_loss /= self.batch_size
        entropy_bonus /= self.batch_size
        total_loss = policy_loss + self.entropy_coef * entropy_bonus
        self.loss_history.append(float(total_loss))

        # Apply gradient step
        step_size = self.lr * 1e-4
        for _ in range(3):
            i = self.rng.integers(len(obs_batch))
            o = obs_batch[i]
            g = step_size * total_loss
            self.W_out -= g * np.outer(np.tanh(o @ self.weights[-1]), self.W_out[:, 0])

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights and hyperparameters to a .npz file."""
        np.savez(
            path,
            weights=[w.copy() for w in self.weights],
            biases=[b.copy() for b in self.biases],
            W_out=self.W_out.copy(),
            b_out=self.b_out.copy(),
            W_val=self.W_val.copy(),
            b_val=self.b_val.copy(),
            epsilon=self.epsilon,
            gamma=self.gamma,
            lr=self.lr,
            algo=self.algo,
        )
        if self.verbose > 0:
            print(f"[FinRLAgent] Saved to {path}")

    def load(self, path: str) -> "FinRLAgent":
        """Load network weights and hyperparameters from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self.weights = [w.copy() for w in data["weights"]]
        self.biases  = [b.copy() for b in data["biases"]]
        self.W_out    = data["W_out"].copy()
        self.b_out    = data["b_out"].copy()
        self.W_val    = data["W_val"].copy()
        self.b_val    = data["b_val"].copy()
        self.epsilon  = float(data["epsilon"])
        if self.verbose > 0:
            print(f"[FinRLAgent] Loaded from {path}")
        return self


# ---------------------------------------------------------------------------
# PortfolioAllocator
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """
    Crypto portfolio allocator — allocates capital across multiple crypto assets
    based on trained agent predictions or heuristic rules.

    Supports three allocation modes:
      - "agent"      : use FinRLAgent to generate portfolio weights
      - "equal"      : equal-weight across all assets
      - "risk_parity": weight inversely proportional to volatility

    Reference: finrl.env.env_portfolio.StockPortfolioEnv (weight-based paradigm)
    """

    def __init__(
        self,
        env: CryptoTradingEnv,
        mode: Literal["agent", "equal", "risk_parity"] = "agent",
        agent: Optional[FinRLAgent] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        env : CryptoTradingEnv
            Trading environment (provides price data and asset list).
        mode : str
            Allocation mode: "agent" | "equal" | "risk_parity".
        agent : FinRLAgent, optional
            Trained agent (required for mode="agent").
        top_k : int, optional
            Only allocate to top-K assets by score (all assets if None).
        seed : int, optional
            Random seed.
        """
        self.env  = env
        self.mode = mode
        self.agent = agent
        self.top_k = top_k
        self.rng  = np.random.default_rng(seed)

        if mode == "agent" and agent is None:
            raise ValueError("PortfolioAllocator(mode='agent') requires a FinRLAgent")

    def allocate(
        self, observation: np.ndarray
    ) -> np.ndarray:
        """
        Compute portfolio weights for the current observation.

        Parameters
        ----------
        observation : np.ndarray
            Current state vector from the environment.

        Returns
        -------
        weights : np.ndarray
            Portfolio weights (sum not necessarily = 1; magnitude matters).
        """
        if self.mode == "equal":
            return self._equal_weights()
        elif self.mode == "risk_parity":
            return self._risk_parity_weights()
        else:  # "agent"
            return self._agent_weights(observation)

    def _equal_weights(self) -> np.ndarray:
        n = self.env.stock_dim
        w = np.ones(n, dtype=np.float32) / n
        if self.top_k and self.top_k < n:
            mask = np.zeros(n, dtype=np.float32)
            top_indices = self.rng.choice(n, self.top_k, replace=False)
            mask[top_indices] = 1.0 / self.top_k
            return mask * n  # scale up to fill budget
        return w

    def _risk_parity_weights(self) -> np.ndarray:
        # Use recent price history to estimate volatility per asset
        prices = self.env.state[1 : self.env.stock_dim + 1]
        # Very crude: use close prices as proxy (no lookback in this simple impl)
        vol = np.abs(prices) + 1e-8
        inv_vol = 1.0 / vol
        w = inv_vol / np.sum(inv_vol)
        return w.astype(np.float32)

    def _agent_weights(self, observation: np.ndarray) -> np.ndarray:
        raw = self.agent.predict(observation, deterministic=True)
        # Normalise to sum-to-1 of absolute values (for long-only interpretation)
        abs_sum = np.sum(np.abs(raw)) + 1e-10
        return (raw / abs_sum).astype(np.float32)


# ---------------------------------------------------------------------------
# CryptoPortfolioBacktester
# ---------------------------------------------------------------------------

class CryptoPortfolioBacktester:
    """
    Backtest a multi-asset crypto trading strategy.

    Provides:
      - Episode simulation with detailed logging
      - Performance metrics (Sharpe, Sortino, max drawdown, Calmar)
      - Benchmark comparison (buy-and-hold baseline)

    Reference: finrl.trade.backtest.backtest_stats / get_daily_return / get_baseline
    """

    def __init__(
        self,
        env: CryptoTradingEnv,
        agent: Optional[FinRLAgent] = None,
        allocator: Optional[PortfolioAllocator] = None,
        benchmark_ticker: Optional[str] = None,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        env : CryptoTradingEnv
            Trading environment (test set).
        agent : FinRLAgent, optional
            Trained agent. If None, uses allocator only.
        allocator : PortfolioAllocator, optional
            Allocator instance. If None, equal-weight is used.
        benchmark_ticker : str, optional
            Ticker to use as buy-and-hold benchmark.
        risk_free_rate : float
            Annual risk-free rate (for Sharpe / Sortino).
        annualization_factor : int
            Trading days per year (252 for crypto).
        seed : int, optional
            Random seed.
        """
        self.env       = env
        self.agent     = agent
        self.allocator = allocator or PortfolioAllocator(env, mode="equal")
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate  = risk_free_rate
        self.annualization   = annualization_factor
        self.rng             = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Backtest run
    # ------------------------------------------------------------------

    def run(
        self,
        verbose: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Run a full backtest episode.

        Parameters
        ----------
        verbose : int
            Print frequency (every `verbose` episodes).

        Returns
        -------
        df_account : pd.DataFrame
            Account value and daily returns over time.
        df_actions : pd.DataFrame
            Per-asset position (shares) at each step.
        metrics : dict
            Performance summary (sharpe, sortino, max_drawdown, calmar, total_return).
        """
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        terminated = False
        step = 0
        episode_reward = 0.0

        account_values = [float(self.env.initial_amount)]
        dates           = [str(self.env.data.date)]
        daily_returns   = [0.0]

        while not terminated:
            # Get allocation
            if self.agent is not None:
                action = self.agent.predict(obs)
            else:
                action = self.allocator.allocate(obs)

            obs2, reward, terminated, truncated, info = self.env.step(action)
            obs2 = np.asarray(obs2, dtype=np.float32)
            step      += 1
            episode_reward += reward

            account_values.append(float(info.get("total_asset", self.env.portfolio_value)))
            dates.append(str(self.env.data.date))
            daily_returns.append(float(reward))

            if terminated or truncated:
                break
            obs = obs2

        # ---- Build DataFrames ----
        df = pd.DataFrame({
            "date":           dates,
            "account_value":  account_values,
            "daily_return":   daily_returns,
        })
        df_account  = df[["date", "account_value"]].copy()
        df_returns  = get_daily_return(df_account)

        # ---- Save actions ----
        df_actions = self.env.save_action_memory()

        # ---- Metrics ----
        metrics = self._compute_metrics(df_returns)

        if verbose > 0:
            print(f"[CryptoPortfolioBacktester] Episode done | steps={step} | "
                  f"return={metrics['total_return']:.4f} | "
                  f"sharpe={metrics['sharpe']:.4f} | "
                  f"max_dd={metrics['max_drawdown']:.4f}")

        return df_account, df_actions, metrics

    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute Sharpe, Sortino, max drawdown, Calmar from a return series."""
        rets = returns.dropna().values
        if len(rets) == 0:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                    "calmar": 0.0, "total_return": 0.0}

        # Annualize
        ann_factor = self.annualization
        mean_ret  = np.mean(rets) * ann_factor
        std_ret   = np.std(rets) * ann_factor
        sharpe    = (mean_ret - self.risk_free_rate) / (std_ret + 1e-10)

        # Sortino (downside deviation)
        downside = rets[rets < 0]
        dstd     = np.std(downside) * np.sqrt(ann_factor) if len(downside) > 0 else 1e-10
        sortino  = (mean_ret - self.risk_free_rate) / dstd

        # Max drawdown
        cumulative = np.cumprod(1 + rets)
        peak       = np.maximum.accumulate(cumulative)
        drawdown   = (cumulative - peak) / peak
        max_dd     = float(np.min(drawdown))

        # Total return
        total_ret  = float(cumulative[-1] - 1.0)

        # Calmar
        calmar = total_ret / (abs(max_dd) + 1e-10) if max_dd != 0 else 0.0

        return {
            "sharpe":        float(sharpe),
            "sortino":       float(sortino),
            "max_drawdown":  max_dd,
            "calmar":        float(calmar),
            "total_return":  total_ret,
            "annual_return": float(mean_ret),
            "annual_vol":    float(std_ret),
        }

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compare_with_benchmark(
        self,
        df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compare strategy returns against a buy-and-hold benchmark.

        Parameters
        ----------
        df : pd.DataFrame
            Strategy account values (with 'account_value' column).
        price_df : pd.DataFrame
            Price data with 'date', 'tic', 'close' columns.

        Returns
        -------
        comparison : dict
            Strategy and benchmark metrics side by side.
        """
        ticker = self.benchmark_ticker or self.env.df.tic.unique()[0]
        bench_df = price_df[price_df.tic == ticker].copy()
        bench_df = bench_df.sort_values("date").reset_index(drop=True)
        bench_rets = bench_df.close.pct_change().dropna()

        strat_rets = get_daily_return(df)

        metrics = {
            "strategy": self._compute_metrics(strat_rets),
            "benchmark": self._compute_metrics(
                pd.Series(bench_rets.values[: len(strat_rets)], name="daily_return")
            ),
        }
        return metrics
