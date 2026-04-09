"""
MarketMakerEnv — Gymnasium market-making environment for PPO training.
PPO市场做市商智能体训练环境.

Integrates:
- LOBSimulator: event-driven limit order book
- AvellanedaStoikov: optimal quote computation
- CARA utility reward function

Supports reward types:
- 'pnl': wealth change + inventory liquidation at terminal
- 'cara': CARA (Constant Absolute Risk Aversion) utility

References:
    - Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
    - Stanford MARKET-MAKING-RL: D:/Hive/Data/trading_repos/MARKET-MAKING-RL/
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gymnasium
import numpy as np
from gymnasium import spaces

from quant_trading.market_making.avellaneda_stoikov import AvellanedaStoikov
from quant_trading.market_making.lob_simulator import LOBSimulator

__all__ = ["MarketMakerEnv"]


class MarketMakerEnv(gymnasium.Env):
    """Market-making Gymnasium environment — for PPO training.

    The agent posts limit bid/ask quotes and earns the spread when filled.
    Inventory risk is penalised via the Avellaneda-Stoikov framework.

    Action space:
        Box(low=-1, high=1, shape=(1,))
        The single action is a spread multiplier:
            0.0  = tightest possible spread (near zero)
            1.0  = optimal AS spread
            > 1  = wider spread (more conservative)

    Observation space (Dict):
        - inventory: Box(low=-max_inventory, high=max_inventory, shape=(1,))
        - spread:    Box(low=0, high=10*tick_size, shape=(1,))
        - midprice:  Box(low=0, high=inf, shape=(1,))
        - time_remaining: Box(low=0, high=T, shape=(1,))

    Reward types (set via reward_type):
        - 'pnl': ΔW + inventory_penalty (default)
        - 'cara': CARA utility U(W) = -exp(-γ·W)

    Example:
        >>> from quant_trading.market_making.lob_simulator import LOBSimulator
        >>> from quant_trading.market_making.market_maker_env import MarketMakerEnv
        >>> lob = LOBSimulator(tick_size=0.01)
        >>> env = MarketMakerEnv(lob=lob, sigma=0.01, T=1.0, reward_type='cara')
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lob: Optional[LOBSimulator] = None,
        initial_inventory: int = 0,
        max_inventory: int = 100,
        sigma: float = 0.01,
        T: float = 1.0,
        dt: float = 1 / 390,  # 1 trading day ≈ 390 seconds (for 6.5hr US equity day)
        reward_type: str = "pnl",  # 'pnl' | 'cara'
        tick_size: float = 0.01,
        initial_midprice: float = 100.0,
        gamma: float = 0.1,
        kappa: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # LOB instance
        self.lob = lob if lob is not None else LOBSimulator(tick_size=tick_size)
        self.lob.midprice = initial_midprice
        self.tick_size = tick_size
        self.initial_midprice = initial_midprice

        # AS parameters
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.max_inventory = max_inventory
        self.initial_inventory = initial_inventory

        # AS solver
        self._as = AvellanedaStoikov(
            gamma=gamma,
            kappa=kappa,
            sigma=sigma,
            T=T,
            spread=0.0,
            midprice=initial_midprice,
        )

        # Reward config
        valid_rewards = ("pnl", "cara")
        if reward_type not in valid_rewards:
            raise ValueError(f"reward_type must be one of {valid_rewards}")
        self.reward_type = reward_type

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "inventory": spaces.Box(
                low=-max_inventory, high=max_inventory, shape=(1,), dtype=np.float32
            ),
            "spread": spaces.Box(
                low=0.0, high=10.0 * tick_size, shape=(1,), dtype=np.float32
            ),
            "midprice": spaces.Box(
                low=0.0, high=1e6, shape=(1,), dtype=np.float32
            ),
            "time_remaining": spaces.Box(
                low=0.0, high=T, shape=(1,), dtype=np.float32
            ),
        })

        # Internal state
        self._wealth: float = 0.0
        self._inventory: int = initial_inventory
        self._time_elapsed: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = int(T / dt) if dt > 0 else 1000
        self._terminal: bool = False
        self._npty: Optional[np.random.Generator] = None

        if seed is not None:
            self._npty = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset the environment.

        Returns:
            tuple[dict, dict]: (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._npty = np.random.default_rng(seed)

        # Re-initialise LOB
        self.lob = LOBSimulator(tick_size=self.tick_size)
        self.lob.midprice = self.initial_midprice

        # Reset state
        self._wealth = 0.0
        self._inventory = self.initial_inventory
        self._time_elapsed = 0.0
        self._step_count = 0
        self._terminal = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: spread multiplier in [-1, 1]

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """
        if self._terminal:
            raise RuntimeError(
                "Cannot call step() after termination. Call reset() first."
            )

        # Parse action
        spread_mult = float(np.clip(action[0], -1.0, 1.0))

        # Evolve midprice (Brownian motion)
        self._evolve_midprice()

        # Compute AS quotes
        t = self._time_elapsed
        bid, ask = self._as.compute_bid_ask(t=t, inventory=self._inventory)

        # Apply spread multiplier
        current_spread = ask - bid
        adjusted_spread = current_spread * max(0.0, spread_mult)
        half_adj = adjusted_spread / 2.0
        mid = self.lob.get_midprice()
        adj_bid = mid - half_adj
        adj_ask = mid + half_adj

        # Submit limit orders
        self.lob.submit_limit_order(price=adj_bid, volume=1, side="buy")
        self.lob.submit_limit_order(price=adj_ask, volume=1, side="sell")

        # Simulate market orders (Poisson arrival)
        self._simulate_market_orders()

        # Advance time
        self._time_elapsed += self.dt
        self._step_count += 1

        # Check termination
        self._terminal = (
            self._step_count >= self._max_steps
            or abs(self._inventory) >= self.max_inventory
            or self.lob.get_spread() <= 0
        )

        # Compute reward
        reward = self._compute_reward()

        obs = self._get_obs()
        info = self._get_info()

        terminated = self._terminal
        truncated = False

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        self._terminal = True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _evolve_midprice(self) -> None:
        """Evolve midprice by Brownian motion with drift."""
        dt = self.dt
        dW = self._normal(0.0, math.sqrt(dt))
        drift = 0.0  # no drift by default
        self.lob.midprice += drift * dt + self.sigma * dW
        self.lob.timestamp = self._time_elapsed

    def _normal(self, loc: float, scale: float) -> float:
        """Sample from normal distribution."""
        if self._npty is not None:
            return self._npty.normal(loc, scale)
        return np.random.normal(loc, scale)  # type: ignore[assignment]

    def _simulate_market_orders(self) -> None:
        """Simulate random market order arrivals (Poisson process)."""
        # Rate parameters (simplified: symmetric)
        lambda_buy = 10.0
        lambda_sell = 10.0

        n_buy = np.random.poisson(lambda_buy * self.dt)
        n_sell = np.random.poisson(lambda_sell * self.dt)

        if n_buy > 0:
            trades = self.lob.submit_market_order(volume=n_buy, side="buy")
            for t in trades:
                self._wealth -= t.price * t.volume
                self._inventory += t.volume

        if n_sell > 0:
            trades = self.lob.submit_market_order(volume=n_sell, side="sell")
            for t in trades:
                self._wealth += t.price * t.volume
                self._inventory -= t.volume

    def _compute_reward(self) -> float:
        """Compute the step reward.

        Args:
            action: spread multiplier

        Returns:
            float: reward value
        """
        if self.reward_type == "pnl":
            # Immediate PnL reward (spread capture) - no explicit dW tracking
            # Instead use spread earned on any fills
            mid = self.lob.get_midprice()
            spread = self.lob.get_spread()

            # Inventory penalty (Avellaneda-Stoikov inspired)
            inv_penalty = self.gamma * (self.sigma**2) * (self._inventory**2) * self.dt
            reward = spread * self.dt - inv_penalty

            # Terminal liquidation bonus
            if self._terminal:
                mid = self.lob.get_midprice()
                reward += self._inventory * mid  # liquidate at midprice

            return float(reward)

        elif self.reward_type == "cara":
            # CARA utility: U(W) = -exp(-gamma * W)
            wealth = self._wealth + self._inventory * self.lob.get_midprice()
            return float(-np.exp(-self.gamma * wealth))

        raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _get_obs(self) -> dict:
        """Build observation dict."""
        return {
            "inventory": np.array([self._inventory], dtype=np.float32),
            "spread": np.array([self.lob.get_spread()], dtype=np.float32),
            "midprice": np.array([self.lob.get_midprice()], dtype=np.float32),
            "time_remaining": np.array(
                [max(0.0, self.T - self._time_elapsed)], dtype=np.float32
            ),
        }

    def _get_info(self) -> dict:
        """Build info dict."""
        return {
            "wealth": self._wealth,
            "inventory": self._inventory,
            "midprice": self.lob.get_midprice(),
            "spread": self.lob.get_spread(),
            "time_elapsed": self._time_elapsed,
            "time_remaining": max(0.0, self.T - self._time_elapsed),
        }

    # ------------------------------------------------------------------ #
    # Optional render (minimal)
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "human") -> None:
        """Render the environment (human mode only)."""
        if mode != "human":
            return
        print(
            f"[t={self._time_elapsed:.4f}/{self.T}] "
            f"midprice={self.lob.get_midprice():.4f}  "
            f"spread={self.lob.get_spread():.4f}  "
            f"inv={self._inventory}  "
            f"wealth={self._wealth:.4f}"
        )
