"""
Actor-Critic base architecture for reinforcement learning trading agents.

Based on: Lapan, Maxim. Deep Reinforcement Learning Hands-On. Second Edition, MITP, 2019.
Original source: D:/Hive/Data/trading_repos/rl-trader
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import collections


class Actions(Enum):
    """Trading actions available to the agent."""
    Skip = 0
    Buy = 1
    Close = 2


@dataclass
class Prices:
    """Price data container for OHLCV data."""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class State:
    """
    Flat state representation for stock trading.

    State shape: [high, low, close, volume] * bars_count + position_flag + relative_profit
    """

    def __init__(
        self,
        bars_count: int,
        commission_perc: float,
        reset_on_close: bool,
        reward_on_close: bool = True,
        volumes: bool = True
    ):
        assert isinstance(bars_count, int) and bars_count > 0
        assert isinstance(commission_perc, float) and commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices: Prices, offset: int) -> None:
        """Reset state with new price data."""
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self) -> Tuple[int]:
        """Return state shape."""
        if self.volumes:
            return (4 * self.bars_count + 1 + 1,)
        return (3 * self.bars_count + 1 + 1,)

    def encode(self) -> np.ndarray:
        """Encode current state as numpy array."""
        res = np.zeros(self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self) -> float:
        """Calculate real close price for current bar."""
        open_price = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open_price * (1.0 + rel_close)

    def step(self, action: Actions) -> Tuple[float, bool]:
        """
        Execute one step in the environment.

        Returns:
            Tuple of (reward, done)
        """
        reward = 0.0
        done = False
        close = self._cur_close()

        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0] - 1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done


class State1D(State):
    """
    1D convolutional state representation.

    State shape: (channels, bars_count) where channels = [high, low, close, volume, position_flag, rel_profit]
    """

    @property
    def shape(self) -> Tuple[int, int]:
        """Return 1D state shape."""
        if self.volumes:
            return (6, self.bars_count)
        return (5, self.bars_count)

    def encode(self) -> np.ndarray:
        """Encode current state as 2D numpy array for 1D convolution."""
        res = np.zeros(self.shape, dtype=np.float32)
        start = self._offset - (self.bars_count - 1)
        stop = self._offset + 1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst + 1] = self._cur_close() / self.open_price - 1.0
        return res


class StocksEnv(gym.Env):
    """
    Stock trading gym environment compatible with stable-baselines3.

    This environment simulates single-asset stock trading with discrete actions
    (Skip, Buy, Close) and supports both flat and 1D convolutional state representations.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        prices: Dict[str, Prices],
        bars_count: int = 10,
        commission: float = 0.0,
        reset_on_close: bool = True,
        state_1d: bool = False,
        random_ofs_on_reset: bool = True,
        reward_on_close: bool = False,
        volumes: bool = False
    ):
        assert isinstance(prices, dict)
        self._prices = prices

        if state_1d:
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes
            )
        else:
            self._state = State(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes
            )

        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32
        )
        self.random_ofs_on_reset = random_ofs_on_reset
        self._seed_val = None

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count

        if self.random_ofs_on_reset:
            offset = self.np_random.choice(
                abs(prices.high.shape[0] - bars * 10)
            ) + bars
        else:
            offset = bars

        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return (obs, reward, done, info)."""
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = self._np_random_init(seed)
        seed2 = self._hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @staticmethod
    def _np_random_init(seed):
        """Initialize random number generator."""
        if seed is None:
            seed = np.random.randint(0, 2 ** 31)
        return np.random.RandomState(seed), seed

    @staticmethod
    def _hash_seed(seed):
        """Hash seed for numpy random."""
        return (seed ^ 0x3b12d168) & 0xffffffff

    @classmethod
    def from_dir(cls, data_dir: str, **kwargs) -> 'StocksEnv':
        """Create environment from directory of CSV files."""
        prices = {
            file: load_relative(file)
            for file in price_files(data_dir)
        }
        return cls(prices, **kwargs)


# Utility functions for data loading
def read_csv(
    file_name: str,
    sep: str = ',',
    filter_data: bool = True,
    fix_open_price: bool = False
) -> Prices:
    """Read OHLCV data from CSV file."""
    import csv
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if 'open' not in h and sep == ',':
            return read_csv(file_name, ';')
        indices = [h.index(s) for s in ('open', 'high', 'low', 'close', 'volume')]
        o, h_l, l, c, v = [], [], [], [], []
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None

        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and all(map(lambda val: abs(val - vals[0]) < 1e-8, vals[:-1])):
                count_filter += 1
                continue

            po, ph, pl, pc, pv = vals

            if fix_open_price and prev_vals is not None:
                ppo, pph, ppl, ppc, ppv = prev_vals
                if abs(po - ppc) > 1e-8:
                    count_fixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)

            count_out += 1
            o.append(po)
            c.append(pc)
            h_l.append(ph)
            l.append(pl)
            v.append(pv)
            prev_vals = vals

    print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
        count_filter + count_out, count_filter, count_fixed))

    return Prices(
        open=np.array(o, dtype=np.float32),
        high=np.array(h_l, dtype=np.float32),
        low=np.array(l, dtype=np.float32),
        close=np.array(c, dtype=np.float32),
        volume=np.array(v, dtype=np.float32)
    )


def prices_to_relative(prices: Prices) -> Prices:
    """Convert absolute prices to relative (percentage change from open)."""
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)


def load_relative(csv_file: str) -> Prices:
    """Load and convert CSV prices to relative format."""
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name: str) -> List[str]:
    """Get list of CSV files in directory."""
    import glob
    import os
    return glob.glob(os.path.join(dir_name, "*.csv"))


class BaseActorCritic:
    """
    Base class for Actor-Critic trading agents.

    Provides common functionality for:
    - Portfolio state tracking
    - Position management
    - Reward shaping
    - Multi-asset support
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        config: Optional[Dict[str, Any]] = None
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}

        # Portfolio state
        self.portfolio_value = 1.0
        self.initial_balance = self.config.get('initial_balance', 10000.0)
        self.position = None  # Current position: None, 'long', 'short'
        self.entry_price = 0.0

        # Reward shaping parameters
        self.reward_scale = self.config.get('reward_scale', 1.0)
        self.profit_taking_threshold = self.config.get('profit_taking_threshold', 0.02)
        self.stop_loss_threshold = self.config.get('stop_loss_threshold', -0.01)

    def reset(self):
        """Reset agent state for new episode."""
        self.portfolio_value = 1.0
        self.position = None
        self.entry_price = 0.0

    def _shaping_reward(self, raw_reward: float, action: int) -> float:
        """
        Apply reward shaping strategies.

        Args:
            raw_reward: Raw reward from environment
            action: Action taken

        Returns:
            Shaped reward
        """
        reward = raw_reward * self.reward_scale

        # Additional shaping can be added here
        # e.g.,Drawdown penalty, volatility penalty, etc.

        return reward

    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Get action from observation.

        Must be implemented by subclasses.

        Args:
            observation: Current state observation
            deterministic: If True, return most likely action

        Returns:
            Action index
        """
        raise NotImplementedError("Subclasses must implement get_action()")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Stable-baselines3 compatible predict method.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, state)
        """
        return self.get_action(observation, deterministic), None

    def save(self, path: str) -> None:
        """Save agent model to path."""
        raise NotImplementedError("Subclasses must implement save()")

    def load(self, path: str) -> None:
        """Load agent model from path."""
        raise NotImplementedError("Subclasses must implement load()")
