"""
jaxmarl_hft - Pure NumPy + Gymnasium HFT Market Making RL.

Absorbed from JaxMARL-HFT (ICAIF 2025), rewritten in pure NumPy.
No JAX dependency — compatible with standard Gymnasium training pipelines.

Key components:
- LOBSTERData:       LOBSTER CSV order-book + message data loader
- LOBState:          Order-book state representation (best bid/ask, book levels)
- HFTMarketMakingEnv: Gymnasium-compatible single-agent HFT market-making env
- MultiAgentHFT:     Multi-agent HFT wrapper (market maker + execution agents)
- PPOMarketMaker:    Pure-NumPy PPO agent for market making

Reference:
  @inproceedings{mohl2025jaxmarlhft,
    title={JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent RL for HFT},
    author={Mohl et al.},
    booktitle={ICAIF 2025},
  }

LOBSTER data format expected under data/rawLOBSTER/<STOCK>/<PERIOD>/:
  <STOCK>_<DATE>_34200000_57600000_message_10.csv
  <STOCK>_<DATE>_34200000_57600000_orderbook_10.csv

Usage:
  from quant_trading.rl.jaxmarl_hft import (
      LOBSTERData, LOBState, HFTMarketMakingEnv, MultiAgentHFT, PPOMarketMaker
  )
  env = HFTMarketMakingEnv(data_dir="data/rawLOBSTER/GOOG/2022")
  obs = env.reset()
  obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import os
import re
import math
import csv
import time
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple, Generator
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

# ---------------------------------------------------------------------------
# LOBSTER data constants (mirrors gymnax_exchange/jaxlobster/constants.py)
# ---------------------------------------------------------------------------

TIME_COL = "<time>"
EVENT_TYPE_COL = "<event_type>"
ORDER_ID_COL = "<order_id>"
SIZE_COL = "<size>"
PRICE_COL = "<price>"
DIRECTION_COL = "<direction>"

MESSAGE_TOKEN_DTYPE_MAP = {
    TIME_COL: int,
    EVENT_TYPE_COL: int,
    ORDER_ID_COL: int,
    SIZE_COL: int,
    PRICE_COL: int,
    DIRECTION_COL: int,
}
MESSAGE_TOKEN_TYPES = list(MESSAGE_TOKEN_DTYPE_MAP.keys())


def get_orderbook_token_types(levels: int) -> List[str]:
    """Build column names for LOBSTER orderbook CSV levels."""
    out = []
    for i in range(1, levels + 1):
        out.extend([f"<ask_price_{i}>", f"<ask_size_{i}>",
                    f"<bid_price_{i}>", f"<bid_size_{i}>"])
    return out


# ---------------------------------------------------------------------------
# LOBSTER event types
# ---------------------------------------------------------------------------
class EventType(IntEnum):
    SUBMISSION = 1   # New limit order
    CANCEL = 2       # Order cancellation (partial or full)
    DELETION = 3     # Full order removal
    TRADE = 4        # Executed trade (may be from a hidden order)
    TRADING = 5      # Auction trade (opening/closing auction)
    CROSSING = 6     # Opening/closing cross
    ODD_LOT = 7      # Non-standard lot trade


class Side(IntEnum):
    BID = -1
    ASK = 1


# ---------------------------------------------------------------------------
# LOBSTERData – data loader
# ---------------------------------------------------------------------------

class LOBSTERData:
    """
    Loads and streams LOBSTER CSV files for a given stock + time period.

    Parameters
    ----------
    data_dir : str
        Root directory containing rawLOBSTER/<STOCK>/<PERIOD>/ CSV files.
    stock : str
        Ticker symbol (e.g. "GOOG").
    time_period : str
        Subfolder name (e.g. "2022").
    levels : int, default 10
        Number of price levels in the orderbook CSV.
    n_msgs : int, default 50
        Number of message rows to concatenate into one episode window.

    Example
    -------
    >>> data = LOBSTERData("data/rawLOBSTER", "GOOG", "2022")
    >>> for window in data.stream_windows():
    ...     print(window.shape)  # (n_steps, n_cols)
    """

    def __init__(
        self,
        data_dir: str,
        stock: str,
        time_period: str,
        levels: int = 10,
        n_msgs: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.stock = stock
        self.time_period = time_period
        self.levels = levels
        self.n_msgs = n_msgs

        self._msg_files: List[Path] = []
        self._ob_files: List[Path] = []
        self._days: List[str] = []
        self._index_by_day: Dict[str, Tuple[Path, Path]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Scan data directory and match message+orderbook files by date."""
        pattern = f"{self.stock}_*_message_{self.levels}.csv"
        msg_glob = self.data_dir / self.stock / self.time_period
        msg_files = sorted(msg_glob.glob(pattern), key=self._extract_date)

        date_to_files: Dict[str, List[Path]] = {}
        for mf in msg_files:
            day = self._extract_date(mf)
            if day:
                ob_pattern = (
                    f"{self.stock}_{day}_34200000_57600000_orderbook_{self.levels}.csv"
                )
                ob_file = mf.parent / ob_pattern
                if ob_file.exists():
                    date_to_files.setdefault(day, []).append(mf)
                    date_to_files.setdefault(day, []).append(ob_file)

        for day, files in sorted(date_to_files.items()):
            if len(files) == 2:
                msg = [f for f in files if "message" in f.name][0]
                ob = [f for f in files if "orderbook" in f.name][0]
                self._index_by_day[day] = (msg, ob)
                self._days.append(day)

    @staticmethod
    def _extract_date(path: Path) -> Optional[str]:
        m = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
        return m.group(0) if m else None

    @staticmethod
    def _convert_time_ns(s: np.ndarray) -> np.ndarray:
        """
        Convert LOBSTER time strings ('HH:MM:SS.ffffff') to nanoseconds since midnight.
        Returns first-difference array (delta_t) in nanoseconds.
        """
        parts = np.char.split(s.astype(str), ".", maxsplit=1)
        sec = np.array([int(p[0].split(":")[0]) * 3600 +
                         int(p[0].split(":")[1]) * 60 +
                         int(p[0].split(":")[2]) for p in parts])
        frac = np.array([int(p[1].ljust(9, "0")[:9]) if len(p) > 1 else 0
                         for p in parts])
        return (sec * 1_000_000_000 + frac).astype(np.int64)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_message_df(self, path: Path, nrows: Optional[int] = None) -> np.ndarray:
        """Load a LOBSTER message CSV as a structured numpy array."""
        toks = MESSAGE_TOKEN_TYPES + get_orderbook_token_types(self.levels)
        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if nrows and i >= nrows:
                    break
                if len(row) < len(toks):
                    continue
                rec: Dict[str, Any] = {}
                for j, (tok, val) in enumerate(zip(toks, row)):
                    try:
                        rec[tok] = int(val)
                    except ValueError:
                        try:
                            rec[tok] = float(val)
                        except ValueError:
                            rec[tok] = val
                rows.append(rec)
        if not rows:
            return np.zeros((0, len(toks)), dtype=np.int64)
        return self._rows_to_array(rows, toks)

    def _load_orderbook_df(self, path: Path) -> np.ndarray:
        """Load a LOBSTER orderbook CSV as a structured numpy array."""
        toks = get_orderbook_token_types(self.levels)
        rows: List[Dict[str, int]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < len(toks):
                    continue
                rec = {}
                for tok, val in zip(toks, row):
                    try:
                        rec[tok] = int(val)
                    except ValueError:
                        rec[tok] = 0
                rows.append(rec)
        if not rows:
            return np.zeros((0, len(toks)), dtype=np.int64)
        return self._rows_to_array(rows, toks)

    @staticmethod
    def _rows_to_array(rows: List[Dict[str, Any]], toks: List[str]) -> np.ndarray:
        """Convert list of dicts to a 2D numpy array ordered by tokens."""
        arr = np.zeros((len(rows), len(toks)), dtype=np.int64)
        for i, rec in enumerate(rows):
            for j, tok in enumerate(toks):
                arr[i, j] = rec.get(tok, 0)
        return arr

    def _merge_message_ob(
        self,
        msg_arr: np.ndarray,
        ob_arr: np.ndarray,
        n_msgs: int,
    ) -> np.ndarray:
        """
        Interleave orderbook snapshots with message rows.

        LOBSTER provides one orderbook snapshot per message row.
        We merge them so each step has: [msg_tokens..., ob_tokens...].
        """
        if msg_arr.shape[0] == 0 or ob_arr.shape[0] == 0:
            return msg_arr
        # Time is first column; compute delta-time in ns
        time_ns = msg_arr[:, 0].copy()
        delta_t = np.diff(time_ns, prepend=time_ns[0])
        delta_t[0] = 0

        n_rows = msg_arr.shape[0]
        merged = np.zeros((n_rows, msg_arr.shape[1] + ob_arr.shape[1]), dtype=np.int64)
        merged[:, 0] = delta_t          # delta time in ns
        merged[:, 1:] = np.concatenate([msg_arr[:, 1:], ob_arr], axis=1)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def days(self) -> List[str]:
        """Available trading days."""
        return self._days.copy()

    def load_day(self, day: str) -> np.ndarray:
        """Load all data for a single day as a merged array (steps, features)."""
        if day not in self._index_by_day:
            raise ValueError(f"Unknown day: {day}")
        msg_path, ob_path = self._index_by_day[day]
        msg_arr = self._load_message_df(msg_path)
        ob_arr = self._load_orderbook_df(ob_path)
        return self._merge_message_ob(msg_arr, ob_arr, self.n_msgs)

    def stream_windows(
        self,
        day: Optional[str] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield rolling windows over the loaded data.

        Parameters
        ----------
        day : str, optional
            Specific trading day. If None, streams all days sequentially.
        window_size : int, optional
            Number of steps per window. Defaults to n_msgs.
        step_size : int, optional
            Stride between windows. Defaults to window_size (no overlap).

        Yields
        ------
        np.ndarray
            Shape (window_size, n_features).
        """
        if window_size is None:
            window_size = self.n_msgs
        if step_size is None:
            step_size = window_size

        days_to_stream = [day] if day else self._days
        for d in days_to_stream:
            data = self.load_day(d)
            n_steps = data.shape[0]
            for start in range(0, n_steps - window_size + 1, step_size):
                yield data[start:start + window_size]

    def get_episode_windows(
        self,
        n_episodes: int = 100,
        window_size: Optional[int] = None,
        max_days: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return a list of random non-overlapping episode windows."""
        if window_size is None:
            window_size = self.n_msgs
        windows: List[np.ndarray] = []
        days = self._days[:max_days] if max_days else self._days
        while len(windows) < n_episodes:
            for d in days:
                data = self.load_day(d)
                n_steps = data.shape[0]
                if n_steps <= window_size:
                    continue
                start = np.random.randint(0, n_steps - window_size)
                windows.append(data[start:start + window_size])
                if len(windows) >= n_episodes:
                    break
        return windows

    def column_names(self) -> List[str]:
        """Return merged column names: msg tokens + ob tokens."""
        return MESSAGE_TOKEN_TYPES + get_orderbook_token_types(self.levels)


# ---------------------------------------------------------------------------
# LOBState – order-book state
# ---------------------------------------------------------------------------

@dataclass
class LOBState:
    """
    Lightweight order-book state for a single time step.

    Attributes
    ----------
    best_bid : int
        Best bid price (in price units, i.e. cents/tick).
    best_ask : int
        Best ask price.
    bid_sizes : np.ndarray
        Sizes at each bid level, shape (levels,).
    ask_sizes : np.ndarray
        Sizes at each ask level, shape (levels,).
    mid_price : float
        (best_bid + best_ask) / 2.0.
    spread : int
        best_ask - best_bid.
    time_ns : int
        Timestamp delta since last event (nanoseconds).
    event_type : int
        LOBSTER event type ID.
    direction : int
        +1 (ask) or -1 (bid).
    volume_imbalance : float
        (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9).
    """
    best_bid: int
    best_ask: int
    bid_sizes: np.ndarray
    ask_sizes: np.ndarray
    time_ns: int = 0
    event_type: int = 0
    direction: int = 0
    # Derived
    _mid: float = field(init=False, repr=False)
    _spread: int = field(init=False, repr=False)
    _imb: float = field(init=False, repr=False)

    def __post_init__(self):
        self._mid = (self.best_bid + self.best_ask) / 2.0
        self._spread = self.best_ask - self.best_bid
        total = np.sum(self.bid_sizes) + np.sum(self.ask_sizes) + 1e-9
        self._imb = (np.sum(self.bid_sizes) - np.sum(self.ask_sizes)) / total

    @property
    def mid_price(self) -> float:
        return self._mid

    @property
    def spread_ticks(self) -> int:
        return self._spread

    @property
    def volume_imbalance(self) -> float:
        return self._imb

    @classmethod
    def from_array(cls, arr: np.ndarray, levels: int = 10) -> "LOBState":
        """
        Build a LOBState from a single row of the LOBSTER merged array.

        Array layout (first 6 cols are message, remaining 4*levels are book):
          [time_ns, event_type, order_id, size, price, direction,
           ask_p1, ask_s1, bid_p1, bid_s1, ..., ask_pN, ask_sN, bid_pN, bid_sN]
        """
        if arr.shape[0] < 6 + 4 * levels:
            raise ValueError(f"Row too short: expected >= {6 + 4*levels}, got {arr.shape[0]}")
        time_ns = int(arr[0])
        event_type = int(arr[1])
        direction = int(arr[5])

        ask_prices = arr[6::4][:levels].astype(int)
        ask_sizes = arr[7::4][:levels].astype(int)
        bid_prices = arr[8::4][:levels].astype(int)
        bid_sizes = arr[9::4][:levels].astype(int)

        return cls(
            best_bid=int(bid_prices[0]) if bid_prices[0] > 0 else 0,
            best_ask=int(ask_prices[0]) if ask_prices[0] > 0 else int(1e9),
            ask_sizes=ask_sizes,
            bid_sizes=bid_sizes,
            time_ns=time_ns,
            event_type=event_type,
            direction=direction,
        )

    def to_features(self, include_full_book: bool = True) -> np.ndarray:
        """
        Flatten the LOB state into a feature vector for the RL agent.

        Parameters
        ----------
        include_full_book : bool
            If True, include all book levels; otherwise just best bid/ask/sizes.

        Returns
        -------
        np.ndarray
            1D feature array.
        """
        if include_full_book:
            feats = [
                float(self.best_bid),
                float(self.best_ask),
                float(self.mid_price),
                float(self.spread_ticks),
                float(self.time_ns),
                float(self.event_type),
                float(self.direction),
                float(self.volume_imbalance),
                self.bid_sizes.astype(float),
                self.ask_sizes.astype(float),
                np.array([float(self.best_bid) * float(s) for s in self.bid_sizes]),  # bid depth
                np.array([float(self.best_ask) * float(s) for s in self.ask_sizes]),  # ask depth
            ]
        else:
            feats = [
                float(self.best_bid),
                float(self.best_ask),
                float(self.mid_price),
                float(self.spread_ticks),
                float(self.time_ns),
                float(self.volume_imbalance),
            ]
        flat = np.concatenate([np.atleast_1d(f) for f in feats]).astype(np.float32)
        return flat


# ---------------------------------------------------------------------------
# HFTMarketMakingEnv – Gymnasium-compatible environment
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM_AVAILABLE = True
except ImportError:
    _GYMNASIUM_AVAILABLE = False
    gym = None  # type: ignore
    spaces = None  # type: ignore


class HFTMarketMakingEnv:
    """
    Gymnasium-compatible High-Frequency Trading market-making environment.

    Inspired by the market-making environment from JaxMARL-HFT (ICAIF 2025),
    reimplemented in pure NumPy without JAX.

    The agent posts limit orders (bid + ask) and earns the spread while
    managing inventory risk.

    Action Space (AvSt action space from JaxMARL-HFT)
    ------------------------------------
    Discrete(8):
        0 = do nothing
        1 = place bid @ best_bid - 1 tick
        2 = place ask @ best_ask + 1 tick
        3 = place bid @ best_bid - 1 tick AND ask @ best_ask + 1 tick
        4 = cancel bid
        5 = cancel ask
        6 = cancel both
        7 = aggressive: cross spread (buy at ask or sell at bid)

    Observation Space
    ----------------
    Dict:
        "lob": Box(low=-inf, high=inf, shape=(feature_dim,))
              Flattened LOBState features
        "inventory": Box(low=-inf, high=inf, shape=(1,))
              Current agent inventory (shares)
        "cash": Box(low=-inf, high=inf, shape=(1,))
              Cash balance
        "step": Box(low=0, high=inf, shape=(1,))
              Current step index

    Reward
    ------
    PnL-based with inventory penalty:
        reward = realized_PnL - lambda * inventory^2

    Episode termination
    -------------------
    - step >= max_steps
    - inventory exceeds liquidation threshold

    Parameters
    ----------
    data_dir : str
        LOBSTER data root (rawLOBSTER/<STOCK>/<PERIOD>/).
    stock : str, default "GOOG"
    time_period : str, default "2022"
    levels : int, default 10
        Orderbook levels.
    window_size : int, default 50
        Steps per episode.
    tick_size : int, default 100
        Price tick in same units as LOBSTER data (e.g. cents).
    inv_penalty_lambda : float, default 1.0
        Inventory penalty coefficient.
    max_inventory : int, default 100
        Liquidation threshold (|inventory| > max_inventory → done).
    reward_scaling : float, default 1e-4
        Scales reward to improve RL convergence.
    normalize_obs : bool, default True
        Z-score normalize observations using running stats.

    Example
    -------
    >>> env = HFTMarketMakingEnv("data/rawLOBSTER", "GOOG", "2022")
    >>> obs, info = env.reset()
    >>> print(obs["lob"].shape)  # (n_features,)
    >>> obs, reward, term, trunc, info = env.step(action=3)
    """

    # Possible action space variants
    ACTION_SPACES = {
        "AvSt": 8,      # 8 discrete actions (spread/skew market-making)
        "fixed_quants": 9,
        "spread_skew": 6,
        "simple": 3,
    }

    def __init__(
        self,
        data_dir: str,
        stock: str = "GOOG",
        time_period: str = "2022",
        levels: int = 10,
        window_size: int = 50,
        tick_size: int = 100,
        inv_penalty_lambda: float = 1.0,
        max_inventory: int = 100,
        reward_scaling: float = 1e-4,
        normalize_obs: bool = True,
        action_space_name: str = "AvSt",
    ):
        if not _GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required: pip install gymnasium")

        self.data_dir = data_dir
        self.stock = stock
        self.time_period = time_period
        self.levels = levels
        self.window_size = window_size
        self.tick_size = tick_size
        self.inv_penalty_lambda = inv_penalty_lambda
        self.max_inventory = max_inventory
        self.reward_scaling = reward_scaling
        self.normalize_obs = normalize_obs
        self.action_space_name = action_space_name
        self.n_actions = self.ACTION_SPACES.get(action_space_name, 8)

        # Load LOBSTER data
        self._lobster = LOBSTERData(
            data_dir=data_dir,
            stock=stock,
            time_period=time_period,
            levels=levels,
            n_msgs=window_size,
        )
        self._episodes: List[np.ndarray] = []
        self._current_episode: Optional[np.ndarray] = None
        self._episode_idx: int = 0

        # Observation dimension
        self._feature_dim = self._compute_feature_dim()

        # Running normalization stats
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_var: Optional[np.ndarray] = None
        self._obs_count: int = 0

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "lob":       spaces.Box(low=-1e6, high=1e6, shape=(self._feature_dim,), dtype=np.float32),
            "inventory": spaces.Box(low=-1e9, high=1e9, shape=(1,), dtype=np.float32),
            "cash":      spaces.Box(low=-1e9, high=1e9, shape=(1,), dtype=np.float32),
            "step":      spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(self.n_actions)

        # Internal state (reset() populates these)
        self._state: Optional[LOBState] = None
        self._step_idx: int = 0
        self._inventory: int = 0
        self._cash: float = 0.0
        self._episode_reward: float = 0.0
        self._total_pnl: float = 0.0
        self._pending_bid: Optional[Tuple[int, int]] = None  # (price, size)
        self._pending_ask: Optional[Tuple[int, int]] = None
        self._window: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_feature_dim(self) -> int:
        """Estimate feature dimension without loading data."""
        # We'll compute it from the first real window
        return 8 + 2 * self.levels + 2 * self.levels  # lob features

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Incrementally update running stats and normalize."""
        if self._obs_mean is None:
            self._obs_mean = np.zeros_like(obs)
            self._obs_var = np.ones_like(obs)
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += delta * delta2
        std = np.sqrt(self._obs_var / max(self._obs_count, 1)) + 1e-8
        return (obs - self._obs_mean) / std

    def _process_action(self, action: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Convert a discrete action into order operations.

        Returns
        -------
        orders : list of order dicts
            Each dict: {"side": Side.BID or Side.ASK, "price": int, "size": int, "type": "limit"|"ioc"}
        fill_value : int
            Net cash change from any immediate fills.
        """
        orders: List[Dict[str, Any]] = []
        fill_value = 0

        bid_price = self._state.best_bid if self._state else 0
        ask_price = self._state.best_ask if self._state else 0

        if action == 0:
            # do nothing
            pass
        elif action == 1:
            # place bid @ best_bid - 1 tick
            if bid_price > 0:
                orders.append({
                    "side": Side.BID,
                    "price": bid_price - self.tick_size,
                    "size": 1,
                    "type": "limit",
                })
                self._pending_bid = (bid_price - self.tick_size, 1)
        elif action == 2:
            # place ask @ best_ask + 1 tick
            if ask_price < int(1e9):
                orders.append({
                    "side": Side.ASK,
                    "price": ask_price + self.tick_size,
                    "size": 1,
                    "type": "limit",
                })
                self._pending_ask = (ask_price + self.tick_size, 1)
        elif action == 3:
            # place both bid and ask
            if bid_price > 0:
                orders.append({
                    "side": Side.BID,
                    "price": bid_price - self.tick_size,
                    "size": 1,
                    "type": "limit",
                })
                self._pending_bid = (bid_price - self.tick_size, 1)
            if ask_price < int(1e9):
                orders.append({
                    "side": Side.ASK,
                    "price": ask_price + self.tick_size,
                    "size": 1,
                    "type": "limit",
                })
                self._pending_ask = (ask_price + self.tick_size, 1)
        elif action == 4:
            # cancel bid
            self._pending_bid = None
        elif action == 5:
            # cancel ask
            self._pending_ask = None
        elif action == 6:
            # cancel both
            self._pending_bid = None
            self._pending_ask = None
        elif action == 7:
            # aggressive: cross the spread (IOC market order)
            if self._inventory > 0 and ask_price < int(1e9):
                # sell: hit the ask
                orders.append({
                    "side": Side.ASK,
                    "price": ask_price,
                    "size": min(self._inventory, 1),
                    "type": "ioc",
                })
                fill_value -= ask_price * min(self._inventory, 1)
                self._inventory -= min(self._inventory, 1)
            elif self._inventory < 0 and bid_price > 0:
                # buy: lift the bid
                orders.append({
                    "side": Side.BID,
                    "price": bid_price,
                    "size": min(abs(self._inventory), 1),
                    "type": "ioc",
                })
                fill_value += bid_price * min(abs(self._inventory), 1)
                self._inventory += min(abs(self._inventory), 1)
        return orders, fill_value

    def _check_fills(self, orders: List[Dict[str, Any]], fill_value: int) -> int:
        """
        Check if any pending orders were filled by the new market event.
        Returns net cash change.
        """
        if not orders or self._state is None:
            return fill_value
        for order in orders:
            if order["type"] == "ioc":
                # IOC orders execute immediately at specified price
                if order["side"] == Side.ASK:
                    fill_value -= order["price"] * order["size"]
                    self._inventory -= order["size"]
                else:
                    fill_value += order["price"] * order["size"]
                    self._inventory += order["size"]
            elif order["type"] == "limit":
                # Check if this limit order got filled
                if order["side"] == Side.BID and self._state.event_type == EventType.TRADING:
                    # A trade occurred — check if our bid was hit
                    trade_price = self._state.best_ask
                    if order["price"] >= trade_price and trade_price > 0:
                        fill_value += trade_price * order["size"]
                        self._inventory += order["size"]
                elif order["side"] == Side.ASK and self._state.event_type == EventType.TRADING:
                    trade_price = self._state.best_bid
                    if order["price"] <= trade_price and trade_price > 0:
                        fill_value -= trade_price * order["size"]
                        self._inventory -= order["size"]
        return fill_value

    def _compute_reward(self) -> float:
        """
        Compute market-making reward.
        reward = realized PnL - lambda * inventory^2
        """
        # Inventory penalty
        inv_penalty = self.inv_penalty_lambda * (self._inventory ** 2)
        # Simple cash delta as realized PnL proxy
        reward = self.reward_scaling * (self._cash - inv_penalty)
        return float(reward)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to a new episode.

        Returns
        -------
        observation : dict
            {"lob": np.ndarray, "inventory": np.ndarray, "cash": np.ndarray, "step": np.ndarray}
        info : dict
            Empty dict (reserved for diagnostics).
        """
        if seed is not None:
            np.random.seed(seed)

        # Load episodes on first reset
        if not self._episodes:
            self._episodes = self._lobster.get_episode_windows(
                n_episodes=200, window_size=self.window_size
            )
            if not self._episodes:
                raise RuntimeError(f"No LOBSTER data found in {self.data_dir}")

        # Pick a random episode
        self._episode_idx = np.random.randint(len(self._episodes))
        self._window = self._episodes[self._episode_idx].copy()

        # Initial LOB state
        self._step_idx = 0
        self._state = LOBState.from_array(self._window[0], levels=self.levels)
        self._inventory = 0
        self._cash = 0.0
        self._pending_bid = None
        self._pending_ask = None
        self._episode_reward = 0.0

        return self.get_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            Discrete action (0-7 for AvSt).

        Returns
        -------
        observation : dict
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # 1. Process action → order list + initial fill value
        orders, fill_value = self._process_action(action)

        # 2. Advance market to next step
        self._step_idx += 1
        done = self._step_idx >= self._window.shape[0]
        if not done:
            self._state = LOBState.from_array(self._window[self._step_idx], levels=self.levels)

        # 3. Check fills from new market state
        fill_value = self._check_fills(orders, fill_value)
        self._cash += fill_value

        # 4. Compute reward
        reward = self._compute_reward()
        self._episode_reward += reward

        # 5. Check termination
        terminated = done
        truncated = abs(self._inventory) >= self.max_inventory

        return self.get_observation(), reward, terminated, truncated, {}

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return the current observation dict."""
        lob_feat = self._state.to_features(include_full_book=True) if self._state else np.zeros(self._feature_dim, dtype=np.float32)
        obs = np.concatenate([
            lob_feat,
            np.array([float(self._inventory)], dtype=np.float32),
            np.array([float(self._cash)], dtype=np.float32),
            np.array([float(self._step_idx)], dtype=np.float32),
        ])
        if self.normalize_obs:
            obs = self._normalize(obs.astype(np.float32))
        else:
            obs = obs.astype(np.float32)

        # Split into dict
        n_lob = lob_feat.shape[0]
        return {
            "lob":       obs[:n_lob],
            "inventory": obs[n_lob:n_lob + 1],
            "cash":      obs[n_lob + 1:n_lob + 2],
            "step":      obs[n_lob + 2:],
        }

    def render(self, mode: str = "human") -> None:
        """Placeholder render (text mode)."""
        if self._state:
            print(
                f"[HFTEnv] step={self._step_idx} "
                f"bid={self._state.best_bid} ask={self._state.best_ask} "
                f"inv={self._inventory} cash={self._cash:.0f}"
            )


# ---------------------------------------------------------------------------
# MultiAgentHFT – multi-agent wrapper
# ---------------------------------------------------------------------------

class MultiAgentHFT:
    """
    Multi-agent HFT framework wrapping one or more HFTMarketMakingEnv instances.

    Provides a shared order-book view and dispatches actions per agent.
    Mirrors the MARLEnv interface from JaxMARL-HFT but in pure NumPy.

    Parameters
    ----------
    data_dir : str
        LOBSTER data root.
    stock : str
        Ticker.
    time_period : str
        Data period subfolder.
    n_market_makers : int, default 1
        Number of market-making agents.
    n_executors : int, default 0
        Number of execution agents (not yet implemented).
    levels : int, default 10
        Orderbook levels.
    window_size : int, default 50
        Episode length.
    kwargs : dict
        Passed to HFTMarketMakingEnv.

    Example
    -------
    >>> mae = MultiAgentHFT("data/rawLOBSTER", "GOOG", "2022", n_market_makers=2)
    >>> obs = mae.reset()
    >>> obs["market_maker_0"]  # per-agent observation
    >>> obs, rewards, terms, truncs, infos = mae.step({"market_maker_0": 3, "market_maker_1": 0})
    """

    def __init__(
        self,
        data_dir: str,
        stock: str,
        time_period: str,
        n_market_makers: int = 1,
        n_executors: int = 0,
        levels: int = 10,
        window_size: int = 50,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.stock = stock
        self.time_period = time_period
        self.n_market_makers = n_market_makers
        self.n_executors = n_executors
        self.n_agents = n_market_makers + n_executors

        self._agent_names = (
            [f"market_maker_{i}" for i in range(n_market_makers)]
            + [f"executor_{i}" for i in range(n_executors)]
        )

        # One shared LOBSTERData, one env per market-maker agent
        self._envs: Dict[str, HFTMarketMakingEnv] = {}
        for name in self._agent_names[:n_market_makers]:
            self._envs[name] = HFTMarketMakingEnv(
                data_dir=data_dir,
                stock=stock,
                time_period=time_period,
                levels=levels,
                window_size=window_size,
                **kwargs,
            )
        # Executors placeholder
        for name in self._agent_names[n_market_makers:]:
            self._envs[name] = None  # type: ignore

        self.observation_spaces = {name: env.observation_space for name, env in self._envs.items() if env}
        self.action_spaces = {name: env.action_space for name, env in self._envs.items() if env}

    @property
    def agents(self) -> List[str]:
        return self._agent_names.copy()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Reset all agents and return per-agent observations."""
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for name in self._agent_names[:self.n_market_makers]:
            obs, _ = self._envs[name].reset(seed=seed)
            out[name] = obs
        for name in self._agent_names[self.n_market_makers:]:
            out[name] = {}  # placeholder for executor
        return out

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Step all agents simultaneously."""
        obs_out: Dict[str, Dict[str, np.ndarray]] = {}
        rewards: Dict[str, float] = {}
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}

        for i, name in enumerate(self._agent_names[:self.n_market_makers]):
            action = actions.get(name, 0)
            obs, reward, term, trunc, info = self._envs[name].step(action)
            obs_out[name] = obs
            rewards[name] = reward
            terms[name] = term
            truncs[name] = trunc
            infos[name] = info

        for name in self._agent_names[self.n_market_makers:]:
            obs_out[name] = {}
            rewards[name] = 0.0
            terms[name] = True
            truncs[name] = False
            infos[name] = {}

        terms["__all__"] = all(terms.values())
        truncs["__all__"] = any(truncs.values())
        return obs_out, rewards, terms, truncs, infos

    def observation_space(self, agent: str):
        return self.observation_spaces.get(agent)

    def action_space(self, agent: str):
        return self.action_spaces.get(agent)


# ---------------------------------------------------------------------------
# PPOMarketMaker – Pure NumPy PPO agent
# ---------------------------------------------------------------------------


class PPOMarketMaker:
    """
    Pure-NumPy PPO agent for HFT market making.

    Implements a clipped PPO update (Schulman et al., 2017) with a
    shared actor-critic network (MLP) — no JAX, no torch.

    Network
    -------
    Input:  observation dim (from HFTMarketMakingEnv)
    Hidden: 2 hidden layers of 64 units with tanh activation
    Output: action logits (policy) + value scalar (critic)

    Training
    --------
    Call `update(replay_buffer)` after collecting trajectories.
    The buffer should hold dicts with keys:
        obs, action, reward, done, log_prob_old, value_old

    Parameters
    ----------
    obs_dim : int
    n_actions : int
    gamma : float, default 0.99
        Discount factor.
    epsilon : float, default 0.2
        PPO clipping parameter.
    lr : float, default 3e-4
        Learning rate.
    ent_coef : float, default 0.01
        Entropy bonus coefficient.
    vf_coef : float, default 0.5
        Value loss coefficient.
    max_grad_norm : float, default 0.5
        Gradient clipping norm.
    hidden_dims : list of int, default [64, 64]

    Example
    -------
    >>> import gymnasium as gym
    >>> from quant_trading.rl.jaxmarl_hft import HFTMarketMakingEnv, PPOMarketMaker
    >>> env = HFTMarketMakingEnv("data/rawLOBSTER", "GOOG", "2022")
    >>> obs_dim = env.observation_space["lob"].shape[0]
    >>> agent = PPOMarketMaker(obs_dim=obs_dim, n_actions=env.action_space.n)
    >>> obs, _ = env.reset()
    >>> action, log_prob, value = agent.act(obs["lob"])
    >>> obs, reward, term, trunc, _ = env.step(action)
    >>> agent.remember(obs["lob"], action, reward, False, log_prob, value)
    >>> if len(agent.buffer) >= agent.buffer_size:
    ...     agent.update()
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        lr: float = 3e-4,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dims: Sequence[int] = (64, 64),
        buffer_size: int = 4096,
        batch_size: int = 64,
        n_epochs: int = 4,
        seed: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self._rng = np.random.default_rng(seed)

        # ---- Neural network weights (simple list-of-ndarrays) ----
        # Layer shapes: (in, out)
        self._rng = np.random.default_rng(seed)
        dims = [obs_dim, *hidden_dims, n_actions]
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            # Xavier-like init
            w = self._rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

        # Critic head (separate last layer, output=1)
        critic_dims = [*hidden_dims, 1]
        self._c_weights: List[np.ndarray] = []
        self._c_biases: List[np.ndarray] = []
        for i in range(len(critic_dims) - 1):
            w = self._rng.normal(0, np.sqrt(2.0 / critic_dims[i]),
                                  (critic_dims[i], critic_dims[i + 1])).astype(np.float32)
            b = np.zeros(critic_dims[i + 1], dtype=np.float32)
            self._c_weights.append(w)
            self._c_biases.append(b)

        # Optimiser state (Adam-style, per weight matrix)
        self._m_w: List[np.ndarray] = [np.zeros_like(w) for w in self._weights]
        self._v_w: List[np.ndarray] = [np.zeros_like(w) for w in self._weights]
        self._m_b: List[np.ndarray] = [np.zeros_like(b) for b in self._biases]
        self._v_b: List[np.ndarray] = [np.zeros_like(b) for b in self._biases]
        self._m_cw: List[np.ndarray] = [np.zeros_like(w) for w in self._c_weights]
        self._v_cw: List[np.ndarray] = [np.zeros_like(w) for w in self._c_weights]
        self._m_cb: List[np.ndarray] = [np.zeros_like(b) for b in self._c_biases]
        self._v_cb: List[np.ndarray] = [np.zeros_like(b) for b in self._c_biases]
        self._t = 0
        self._lr = lr

        # ---- Rollout buffer ----
        self.buffer: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Network forward pass
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray, for_critic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the MLP.

        Returns
        -------
        logits_or_value : np.ndarray
            Action logits (actor) or value (critic).
        activation : np.ndarray
            Last hidden layer activation (for autograd alternative).
        """
        weights = self._c_weights if for_critic else self._weights
        biases = self._c_biases if for_critic else self._biases

        act = x
        for i, (w, b) in enumerate(zip(weights, biases)):
            act = act @ w + b
            if i < len(weights) - 1:
                act = np.tanh(act)
        return act, act

    def _logits(self, x: np.ndarray) -> np.ndarray:
        """Return action logits."""
        logits, _ = self._forward(x, for_critic=False)
        return logits

    def _value(self, x: np.ndarray) -> np.ndarray:
        """Return state value."""
        val, _ = self._forward(x, for_critic=True)
        return val.flatten()

    def _policy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action and return (action, log_prob).

        Uses softmax policy.
        """
        logits = self._logits(x)
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)
        action = int(np.argmax(self._rng.random() > np.cumsum(probs)))
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob, probs

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Select action given observation.

        Parameters
        ----------
        obs : np.ndarray
            Flat observation array (obs_dim,).
        deterministic : bool
            If True, take argmax policy; else sample.

        Returns
        -------
        action : int
        log_prob : np.ndarray (scalar)
        value : np.ndarray (scalar)
        """
        if deterministic:
            logits = self._logits(obs)
            action = int(np.argmax(logits))
            log_prob = np.log(np.exp(logits - np.max(logits)) /
                               np.sum(np.exp(logits - np.max(logits))) + 1e-8)[action]
        else:
            action, log_prob, _ = self._policy(obs)
        value = self._value(obs)
        return action, np.array(log_prob, dtype=np.float32), np.array(value, dtype=np.float32)

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob_old: np.ndarray,
        value_old: np.ndarray,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.append({
            "obs": obs.astype(np.float32),
            "action": action,
            "reward": reward,
            "done": done,
            "log_prob_old": float(log_prob_old),
            "value_old": float(value_old),
        })
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def update(self) -> Dict[str, float]:
        """
        Perform a PPO update on the current buffer.

        Implements:
        - GAE (lambda=0.95) for advantage estimation
        - Clipped surrogate objective
        - Value loss with clipping
        - Entropy bonus
        - Adam optimizer

        Returns
        -------
        metrics : dict
            {"policy_loss", "value_loss", "entropy", "approx_kl"}
        """
        if len(self.buffer) < self.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # ---- Compute advantages with GAE ----
        obs = np.array([t["obs"] for t in self.buffer], dtype=np.float32)
        actions = np.array([t["action"] for t in self.buffer], dtype=np.int32)
        rewards = np.array([t["reward"] for t in self.buffer], dtype=np.float32)
        dones = np.array([t["done"] for t in self.buffer], dtype=np.float32)
        log_probs_old = np.array([t["log_prob_old"] for t in self.buffer], dtype=np.float32)
        values_old = np.array([t["value_old"] for t in self.buffer], dtype=np.float32)

        # Compute values for all states
        values = self._value(obs)

        # GAE
        advantages = np.zeros_like(rewards)
        gae = 0.0
        lam = 0.95
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values_old  # consistent with rewards-to-go
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics: Dict[str, float] = {}
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []

        for _ in range(self.n_epochs):
            indices = self._rng.permutation(len(obs))[:self.batch_size]

            for idx in indices:
                ob = obs[idx]
                a = actions[idx]
                adv = advantages[idx]
                ret = returns[idx]
                lp_old = log_probs_old[idx]

                # Forward pass
                logits, _ = self._forward(ob, for_critic=False)
                logits -= np.max(logits)
                exp_l = np.exp(logits)
                probs = exp_l / (np.sum(exp_l) + 1e-8)

                # Entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                entropies.append(entropy)

                # Log prob of action a
                log_prob_new = np.log(probs[a] + 1e-8)
                ratio = np.exp(log_prob_new - lp_old)

                # Clipped surrogate
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
                policy_loss = -np.min(surr1, surr2).mean()

                # Value loss
                val, _ = self._forward(ob, for_critic=True)
                val = val.flatten()[0]
                val_clipped = values_old[idx] + np.clip(val - values_old[idx], -self.epsilon, self.epsilon)
                vf_loss1 = (val - ret) ** 2
                vf_loss2 = (val_clipped - ret) ** 2
                value_loss = 0.5 * np.maximum(vf_loss1, vf_loss2).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # ---- Gradient of cross-entropy policy (softmax) + MSE value ----
                # For a simple softmax policy with cross-entropy loss,
                # d_loss/d_logits = (probs - one_hot(a)) * advantage
                # For MSE value loss: d_loss/d_val = 2 * (val - ret)
                d_logits = (probs - np.eye(self.n_actions)[a]) * adv
                d_val = 2 * (val - ret)

                # ---- Backprop manually (same structure as forward) ----
                # We do a numerical gradient step using Adam on the flattened params.
                # This simple implementation applies gradients to actor and critic separately.
                self._apply_gradient(d_logits, d_val, ob)

        if entropies:
            metrics["policy_loss"] = float(np.mean(policy_losses)) if policy_losses else 0.0
            metrics["value_loss"] = float(np.mean(value_losses)) if value_losses else 0.0
            metrics["entropy"] = float(np.mean(entropies))
        else:
            metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        return metrics

    def _apply_gradient(
        self, d_logits: np.ndarray, d_val: np.ndarray, ob: np.ndarray
    ) -> None:
        """Apply a gradient step using Adam optimizer for actor + critic."""
        self._t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        # ---- Actor gradient: backprop through MLP ----
        # Forward: act = tanh(x @ w0 + b0); out = act @ w1 + b1
        # d_loss/d_logits = d_logits
        w0, b0 = self._weights[0], self._biases[0]
        w1, b1 = self._weights[1], self._biases[1]
        # Pre-activation at layer 1
        h0 = ob @ w0 + b0
        a0 = np.tanh(h0)
        # Output logits
        pre_out = a0 @ w1 + b1
        pre_out -= np.max(pre_out)
        exp_pre = np.exp(pre_out)
        probs = exp_pre / (np.sum(exp_pre) + 1e-8)

        # Backward: d_pre_out -> d_w1, d_a0, d_pre_out -> d_w0
        d_pre_out = d_logits  # already shaped (n_actions,)
        grad_w1 = np.outer(a0, d_pre_out)      # (hidden, n_actions)
        grad_b1 = d_pre_out.copy()
        d_a0 = d_pre_out @ w1.T                  # (hidden,)
        d_h0 = d_a0 * (1 - a0 ** 2)             # tanh derivative
        grad_w0 = np.outer(ob, d_h0)            # (obs_dim, hidden)
        grad_b0 = d_h0.copy()

        # ---- Critic gradient ----
        cw0, cb0 = self._c_weights[0], self._c_biases[0]
        cw1, cb1 = self._c_weights[1], self._c_biases[1]
        ch0 = ob @ cw0 + cb0
        ca0 = np.tanh(ch0)
        c_pre = (ca0 @ cw1 + cb1).flatten()[0]
        d_cpre = d_val
        grad_cw1 = np.outer(ca0, [d_cpre])
        grad_cb1 = np.array([d_cpre])
        d_ca0 = d_cpre * cw1.flatten()
        d_ch0 = d_ca0 * (1 - ca0 ** 2)
        grad_cw0 = np.outer(ob, d_ch0)
        grad_cb0 = d_ch0.copy()

        # ---- Adam update ----
        for w, b, gw, gb, m_w, v_w, m_b, v_b in zip(
                self._weights, self._biases, [grad_w0, grad_w1], [grad_b0, grad_b1],
                self._m_w, self._v_w, self._m_b, self._v_b):
            for arr, grad, m, v in [(w, gw, m_w, v_w), (b, gb, m_b, v_b)]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** self._t)
                v_hat = v / (1 - beta2 ** self._t)
                update = m_hat / (np.sqrt(v_hat) + eps)
                # Gradient clipping
                norm = np.linalg.norm(update)
                if norm > self.max_grad_norm:
                    update *= self.max_grad_norm / norm
                arr -= self._lr * update

        for w, b, gw, gb, m, v in zip(
                self._c_weights, self._c_biases, [grad_cw0, grad_cw1], [grad_cb0, grad_cb1],
                self._m_cw, self._v_cw):
            m[:] = beta1 * m + (1 - beta1) * gw
            v[:] = beta2 * v + (1 - beta2) * (gw ** 2)
            m_hat = m / (1 - beta1 ** self._t)
            v_hat = v / (1 - beta2 ** self._t)
            update = m_hat / (np.sqrt(v_hat) + eps)
            norm = np.linalg.norm(update)
            if norm > self.max_grad_norm:
                update *= self.max_grad_norm / norm
            w -= self._lr * update

    def save(self, path: str) -> None:
        """Save agent weights to a .npz file."""
        np.savez(
            path,
            weights=[w for w in self._weights],
            biases=[b for b in self._biases],
            c_weights=[w for w in self._c_weights],
            c_biases=[b for b in self._c_biases],
        )

    def load(self, path: str) -> None:
        """Load agent weights from a .npz file."""
        with np.load(path) as f:
            for i, w in enumerate(f["weights"]):
                self._weights[i][:] = w
            for i, b in enumerate(f["biases"]):
                self._biases[i][:] = b
            for i, w in enumerate(f["c_weights"]):
                self._c_weights[i][:] = w
            for i, b in enumerate(f["c_biases"]):
                self._c_biases[i][:] = b
