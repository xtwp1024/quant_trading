"""
Spread trading environment for pair / statistical arbitrage strategies.

Adapted from Trading-Gym (https://github.com/FinanceDataEngine/Trading-Gym).
Gymnasium-compatible discrete (buy/sell/hold) spread trading environment.

Key features:
- Pair and multi-product spread trading with long/short/flat positions
- Configurable ``spread_coefficients`` (positive=buy, negative=sell)
- ``DataGenerator`` abstraction for flexible data pipelines
- ``CSVStreamer`` for historical CSV data
- ``RandomGenerator`` for synthetic multi-product data
- ``history_length`` parameter for stacking historical states
- Discrete actions: hold (80%) / buy (10%) / sell (10%) default policy

Complements:
- ``StockTradingEnv``: single-asset directional trading
- ``MultiStockTradingEnv``: multi-asset portfolio management

Usage:
    from quant_trading.rl import SpreadTrading, CSVStreamer
    from quant_trading.rl.spread_data_generator import RandomGenerator, RandomWalk, AR1

    # CSV-based
    dg = CSVStreamer(filename="pair_data.csv")
    env = SpreadTrading(
        data_generator=dg,
        spread_coefficients=[1, -1],   # buy product 1, sell product 2
        episode_length=1000,
        history_length=2,
    )

    # Synthetic pair trading
    dg = RandomGenerator([
        RandomWalk(ba_spread=0.01),
        RandomWalk(ba_spread=0.01),
    ])
    env = SpreadTrading(
        data_generator=dg,
        spread_coefficients=[1, -1],
        episode_length=1000,
    )

    obs, info = env.reset()
    for step in range(100):
        action = env.action_space.sample()          # 0=hold, 1=buy, 2=sell
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .spread_data_generator import calc_spread


class SpreadTrading(gym.Env):
    """Gymnasium-compatible discrete spread trading environment.

    The agent can take one of three actions at each step:
    - 0 (hold): maintain current position
    - 1 (buy): enter long / close short
    - 2 (sell): enter short / close long

    Observations are the concatenation of:
    - ``history_length`` historical price rows (each row = ``[bid, ask, ...]``)
    - Current entry price (scalar)
    - Current position encoding: flat=[1,0,0], long=[0,1,0], short=[0,0,1]
    """

    metadata = {"render_modes": []}

    # Action encodings (one-hot)
    _actions = {
        0: np.array([1, 0, 0], dtype=np.float32),  # hold
        1: np.array([0, 1, 0], dtype=np.float32),  # buy
        2: np.array([0, 0, 1], dtype=np.float32),  # sell
    }

    # Position encodings (one-hot)
    _positions = {
        "flat":  np.array([1, 0, 0], dtype=np.float32),
        "long":  np.array([0, 1, 0], dtype=np.float32),
        "short": np.array([0, 0, 1], dtype=np.float32),
    }

    def __init__(
        self,
        data_generator,
        spread_coefficients,
        episode_length=1000,
        trading_fee=0.0,
        time_fee=0.0,
        history_length=2,
    ):
        """Initialise the spread trading environment.

        Args:
            data_generator (DataGenerator): DataGenerator yielding rows of
                bid/ask prices: ``[p1_bid, p1_ask, p2_bid, p2_ask, ...]``.
            spread_coefficients (list[int|float]): Signed coefficients defining
                the spread. Positive = buy this product, negative = sell.
            episode_length (int): Maximum number of steps per episode.
            trading_fee (float): Penalty incurred on every trade (buy or sell).
            time_fee (float): Penalty incurred on every step.
            history_length (int): Number of historical price rows to stack in
                the observation vector.
        """
        super().__init__()

        assert data_generator.n_products == len(spread_coefficients), (
            f"n_products={data_generator.n_products} != "
            f"len(spread_coefficients)={len(spread_coefficients)}"
        )
        assert history_length > 0

        self._data_generator = data_generator
        self._spread_coefficients = list(spread_coefficients)
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self._history_length = history_length

        self.n_actions = 3
        self._prices_history = []

        # Gymnasium spaces (set properly after first reset)
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._action = self._actions[0]

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options.

        Returns:
            tuple: (observation, info) — initial observation and info dict.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0.0
        self._total_pnl = 0.0
        self._position = self._positions["flat"]
        self._entry_price = 0.0
        self._exit_price = 0.0
        self._prices_history = []

        for _ in range(self._history_length):
            self._prices_history.append(self._data_generator.next())

        observation = self._get_observation()
        # Update observation space shape based on actual data
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9,
            shape=observation.shape,
            dtype=np.float32,
        )
        self._action = self._actions[0]

        info = {
            "total_pnl": 0.0,
            "position": "flat",
            "entry_price": 0.0,
        }
        return observation, info

    def step(self, action):
        """Execute one environment step.

        Args:
            action (int or array): Action index. 0=hold, 1=buy, 2=sell.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Accept both int and one-hot
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            self._action = self._actions[action_idx]
        else:
            action_idx = int(np.argmax(action))
            self._action = np.array(action, dtype=np.float32)

        self._iteration += 1
        terminated = False
        truncated = False
        instant_pnl = 0.0
        info = {
            "total_pnl": self._total_pnl,
            "position": ["flat", "long", "short"][
                list(self._position).index(1)
            ],
            "entry_price": self._entry_price,
        }

        reward = -self._time_fee

        # --- Buy action ------------------------------------------------- #
        if action_idx == 1:
            reward -= self._trading_fee
            if all(self._position == self._positions["flat"]):
                self._position = self._positions["long"]
                _, self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients
                )
            elif all(self._position == self._positions["short"]):
                _, self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients
                )
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions["flat"]
                self._entry_price = 0.0

        # --- Sell action ------------------------------------------------ #
        elif action_idx == 2:
            reward -= self._trading_fee
            if all(self._position == self._positions["flat"]):
                self._position = self._positions["short"]
                self._entry_price, _ = calc_spread(
                    self._prices_history[-1], self._spread_coefficients
                )
            elif all(self._position == self._positions["long"]):
                self._exit_price, _ = calc_spread(
                    self._prices_history[-1], self._spread_coefficients
                )
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions["flat"]
                self._entry_price = 0.0

        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward
        info["total_pnl"] = self._total_pnl

        # --- Termination checks ---------------------------------------- #
        try:
            self._prices_history.append(self._data_generator.next())
        except StopIteration:
            terminated = True
            info["status"] = "No more data."

        if self._iteration >= self._episode_length:
            truncated = True
            info["status"] = "Episode timeout."

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment.

        Note: Matplotlib rendering has been removed. This method is a no-op
        to satisfy the Gymnasium API. Override or wrap in a visualisation
        layer if graphical output is needed.
        """
        # No-op — matplotlib removed for standalone use
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_observation(self):
        """Build the observation vector.

        Returns:
            np.ndarray: Flattened observation of stacked price history,
            entry price, and position encoding.
        """
        recent = self._prices_history[-self._history_length :]
        obs = np.concatenate(
            [prices for prices in recent]
            + [np.array([self._entry_price], dtype=np.float32)]
            + [np.array(self._position, dtype=np.float32)],
            axis=0,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def random_action():
        """Random action for exploration.

        Default policy: 80% hold, 10% buy, 10% sell.

        Returns:
            int: Action index (0=hold, 1=buy, 2=sell).
        """
        return int(np.random.multinomial(1, [0.8, 0.1, 0.1]).argmax())

    @property
    def action_meanings(self):
        """Human-readable action labels."""
        return {0: "hold", 1: "buy", 2: "sell"}
