"""
drl_hft.py — Deep Reinforcement Learning for Active High-Frequency Trading (HFT).

Adapted from DRL_for_Active_High_Frequency_Trading (Briola et al., 2021, arXiv:2101.07107).
Ported from PyTorch/stable-baselines3 to pure NumPy + Gymnasium.

Key components:
- HFTLOBEnvironment  : Limit Order Book trading environment (Gymnasium-compatible)
- OrderBookFeatureExtractor : LOB feature engineering for RL state
- HFTPolicy          : Policy network (DQN / A2C / PPO, pure NumPy)
- DeepHFTAgent        : DRL agent with DQN / A2C / PPO implementations
- LatencyModel        : Models execution latency impact on fill/execution price

Bilingual docstrings: Chinese (primary) + English (secondary).

Usage:
    from quant_trading.rl.drl_hft import HFTLOBEnvironment, DeepHFTAgent

    env = HFTLOBEnvironment(lob_data=lob_df, last_n_ticks=10)
    agent = DeepHFTAgent(state_dim=env.observation_space.shape[0], action_dim=4, algo="dqn")
    # training loop ...
"""

from __future__ import annotations

import enum
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------- #
# Enums & Constants
# --------------------------------------------------------------------------- #

class Position(enum.Enum):
    """
    当前持仓方向。
    Position direction: neutral / long / short.
    """
    neutral = 0
    long    = 1
    short   = -1

    def __int__(self) -> int:
        return self.value


# Action mapping / 动作映射
# 0 = sell (close long / open short)  | 卖出（平多/开空）
# 1 = hold (do nothing)               | 持仓（无操作）
# 2 = buy  (close short / open long)   | 买入（平空/开多）
# 3 = stop-loss (close any open pos)   | 止损（平仓）

ACTION_HOLD     = 1
ACTION_BUY      = 2   # open / close short
ACTION_SELL     = 0   # open / close long
ACTION_STOP     = 3


# --------------------------------------------------------------------------- #
# Helper / 辅助函数
# --------------------------------------------------------------------------- #

def _weighted_mid_price(buy_tick: np.ndarray, sell_tick: np.ndarray) -> float:
    """
    Compute volume-weighted mid-price from LOB best bid/ask tick.
    根据最优买卖盘口计算成交量加权中间价。
    """
    return (buy_tick[0] * sell_tick[1] + sell_tick[0] * buy_tick[1]) / (buy_tick[1] + sell_tick[1] + 1e-12)


def _m2m(position: Position, price: float, buy_val: float, sell_val: float) -> float:
    """
    Mark-to-market unrealised PnL.
    持仓盯市未实现盈亏。
    """
    if position == Position.short:
        return price - buy_val
    if position == Position.long:
        return sell_val - price
    return 0.0


# --------------------------------------------------------------------------- #
# LatencyModel
# --------------------------------------------------------------------------- #

class LatencyModel:
    """
    延迟模型：估算订单执行延迟对成交价的影响。
    Latency model: estimates how order execution latency degrades the fill price.

    Uses a simple Gaussian latency jitter + linear price slippage model:
        effective_price = quoted_price + slippage(latency, spread)
    Pure NumPy; no external dependencies.

    Attributes:
        mean_latency_ms  : 平均延迟 (ms) / Mean network + exchange latency
        std_latency_ms   : 延迟标准差 (ms) / Latency standard deviation
        slippage_bps     : 每 ms 滑点 (bps) / Slippage per ms in basis points
        spread_pct       : 盘口价差 (小数 )/ Current spread as fraction of price
    """

    def __init__(
        self,
        mean_latency_ms: float = 0.5,
        std_latency_ms: float  = 0.3,
        slippage_bps: float   = 0.5,
    ) -> None:
        self.mean_latency_ms = mean_latency_ms
        self.std_latency_ms  = std_latency_ms
        self.slippage_bps    = slippage_bps

    def sample_latency(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        从延迟分布采样 (ms)。
        Sample a latency realization in milliseconds.
        """
        if rng is None:
            rng = np.random.default_rng()
        return max(0.0, rng.normal(self.mean_latency_ms, self.std_latency_ms))

    def slippage(
        self,
        latency_ms: float,
        spread_pct: float,
    ) -> float:
        """
        计算给定延迟下的价格滑点。
        Compute price slippage given latency and spread.

        Args:
            latency_ms : 延迟 (ms) / Latency in ms
            spread_pct : 当前买卖价差比例 / Spread as fraction of price

        Returns:
            滑点 (与报价价的比例) / Slippage as fraction of quoted price
        """
        return (latency_ms * self.slippage_bps / 10_000) + (spread_pct / 2)

    def effective_price(
        self,
        quoted_price: float,
        side: Literal["bid", "ask"],
        spread_pct: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        计算考虑延迟滑点后的实际成交价。
        Compute the actual fill price after latency slippage.

        Args:
            quoted_price : 报价价 / Quoted best bid/ask price
            side         : "bid" (买入) or "ask" (卖出)
            spread_pct   : 当前价差比例 / Current spread fraction

        Returns:
            实际成交价 / Effective fill price
        """
        lat_ms   = self.sample_latency(rng)
        slip     = self.slippage(lat_ms, spread_pct)
        direction = -1 if side == "bid" else 1   # bid = buy = price goes up; ask = sell = price goes down
        return quoted_price * (1 + direction * slip)


# --------------------------------------------------------------------------- #
# OrderBookFeatureExtractor
# --------------------------------------------------------------------------- #

class OrderBookFeatureExtractor:
    """
    从限价订单簿 (LOB) 原始数据中提取 RL 状态特征。
    Extract RL-friendly state features from raw Limit Order Book (LOB) data.

    Supports three feature sets inspired by Briola et al. (2021):
    1. vanilla    : 基础订单簿特征（价格 + 成交量）
    2. depth      : 增加各档位深度信息
    3. micro      : 增加价差/弹性/不平衡等微观结构特征

    Pure NumPy; no external DL framework needed.

    Attributes:
        feature_type : "vanilla" | "depth" | "micro"
        n_levels     : LOB 档位数 / Number of price levels to use
    """

    FEATURE_DIM = {
        "vanilla": 40,   # 20 bid + 20 ask volumes
        "depth":   80,   # 40 bid + 40 ask (2 levels each side)
        "micro":   55,   # 40 volumes + 5 spread + 5 imbalance + 5 mid-price
    }

    def __init__(
        self,
        feature_type: Literal["vanilla", "depth", "micro"] = "micro",
        n_levels: int = 20,
    ) -> None:
        if feature_type not in self.FEATURE_DIM:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        self.feature_type = feature_type
        self.n_levels      = n_levels

    @property
    def feature_dim(self) -> int:
        """特征向量维度。Feature vector dimensionality."""
        return self.FEATURE_DIM[self.feature_type]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(
        self,
        lob_snapshot: np.ndarray,
    ) -> np.ndarray:
        """
        从单帧 LOB snapshot 提取特征向量。
        Extract feature vector from a single LOB snapshot.

        Expected lob_snapshot shape: (2 * n_levels, 2) — [[bid_price, bid_vol], ...]

        Returns:
            feature vector of shape (feature_dim,)
        """
        if self.feature_type == "vanilla":
            return self._vanilla(lob_snapshot)
        elif self.feature_type == "depth":
            return self._depth(lob_snapshot)
        else:
            return self._micro(lob_snapshot)

    def extract_batch(
        self,
        lob_history: np.ndarray,
    ) -> np.ndarray:
        """
        从多帧 LOB 历史提取特征（用于窗口状态）。
        Extract features from a window of LOB history.

        Args:
            lob_history : (T, 2*n_levels, 2) array of T snapshots

        Returns:
            (T, feature_dim) array
        """
        return np.stack([self.extract(lob_history[i]) for i in range(lob_history.shape[0])])

    # ------------------------------------------------------------------ #
    # Internal feature builders
    # ------------------------------------------------------------------ #

    def _vanilla(self, lob: np.ndarray) -> np.ndarray:
        """仅用各档成交量 / Use only volumes at each level."""
        # lob shape: (2*n_levels, 2)  [bid, ask] interleaved or stacked
        # Convention used here: first half = bid side, second half = ask side
        n = self.n_levels
        bid_vols = lob[:n, 1]          # bid volumes
        ask_vols = lob[n:, 1]          # ask volumes
        return np.concatenate([bid_vols, ask_vols]).astype(np.float32)

    def _depth(self, lob: np.ndarray) -> np.ndarray:
        """各档价+量 / Price and volume at first two levels."""
        n = self.n_levels
        # Use first 2 levels on each side
        bid_prices = lob[:2, 0]
        bid_vols   = lob[:2, 1]
        ask_prices = lob[n:n+2, 0]
        ask_vols   = lob[n:n+2, 1]
        return np.concatenate([bid_prices, bid_vols, ask_prices, ask_vols]).astype(np.float32)

    def _micro(self, lob: np.ndarray) -> np.ndarray:
        """
        微观结构特征：
        成交量向量 + 价差特征 + 订单簿不平衡 + 中间价趋势
        Volume vector + spread + order-flow imbalance + mid-price trend.
        """
        n = self.n_levels

        # 1. 订单簿量向量 (40d)
        bid_vols = lob[:n, 1]
        ask_vols = lob[n:, 1]
        vol_vec  = np.concatenate([bid_vols, ask_vols])

        # 2. 价差特征 (5d)
        best_bid_price = lob[0, 0]
        best_ask_price = lob[n, 0]
        best_bid_vol   = lob[0, 1]
        best_ask_vol   = lob[n, 1]

        spread_abs  = best_ask_price - best_bid_price
        spread_pct  = spread_abs / ((best_bid_price + best_ask_price) / 2 + 1e-12)

        # 3. 订单簿不平衡 (5d)
        total_bid_vol = np.sum(bid_vols)
        total_ask_vol = np.sum(ask_vols)
        obi           = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-12)
        best_obi      = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol + 1e-12)

        # 4. 中间价 + 加权中间价 (5d)
        mid_price = (best_bid_price + best_ask_price) / 2
        wmid      = _weighted_mid_price(lob[0], lob[n])

        features = np.array([
            *vol_vec,
            spread_abs,
            spread_pct,
            obi,
            best_obi,
            mid_price,
            wmid,
            best_bid_vol,
            best_ask_vol,
        ], dtype=np.float32)

        return features


# --------------------------------------------------------------------------- #
# HFTLOBEnvironment
# --------------------------------------------------------------------------- #

@dataclass
class HFTLOBEnvConfig:
    """
    HFT LOB 环境配置。
    Configuration for HFTLOBEnvironment.
    """
    last_n_ticks:  int      = 10       # 历史快照窗口大小 / Number of past ticks in state
    use_m2m:       bool     = True     # 是否在状态中加入盯市价值 / Include mark-to-market in state
    transaction_cost_pct: float = 5e-4 # 固定交易成本比例 / Transaction cost as fraction
    stop_loss_pct: float    = -0.002   # 止损阈值 (负值) / Stop-loss threshold (negative = pct of price)
    max_position:  int      = 1        # 最大持仓手数 / Max position size (lots)
    seed:          Optional[int] = None # 随机种子 / Random seed

    def __post_init__(self) -> None:
        if self.last_n_ticks < 1:
            raise ValueError("last_n_ticks must be >= 1")


class HFTLOBEnvironment:
    """
    高频交易限价订单簿环境 — 基于 Briola et al. (2021) 的 Gymnasium-compatible 环境。
    Limit Order Book environment for high-frequency trading.

    Inspired by the LOB-based StockEnv from DRL_for_Active_High_Frequency_Trading.
    Uses raw LOB snapshots (price + volume at multiple levels) as the base data source.

    State space (Dict):
        volumes   : (last_n_ticks * 2 * n_levels,) flattened LOB volumes across window
        position  : (1,) current position (+1 long, 0 neutral, -1 short)
        m2m       : (1,) mark-to-market value (optional, if use_m2m=True)

    Action space (Discrete 4):
        0 = Sell / Close long / Open short
        1 = Hold
        2 = Buy  / Close short / Open long
        3 = Stop-loss (close any open position)

    Reward: Realised PnL from trade, or 0 on hold / invalid action.

    Key methods:
        get_state()       : 返回当前 RL 状态 / Return current RL state
        execute_action()  : 执行动作并返回 (next_state, reward, done, info)
        compute_pnl()     : 计算当前累计盈亏 / Compute cumulative PnL
        reset()           : 重置环境 / Reset environment
        step()            : Gymnasium-style step()

    Example:
        >>> env = HFTLOBEnvironment(lob_data=my_lob_df, n_levels=20)
        >>> state = env.reset()
        >>> state, reward, done, info = env.step(2)  # buy
    """

    # metadata / Gymnasium requirement
    metadata = {"render_modes": []}
    render_mode: Optional[str] = None

    def __init__(
        self,
        lob_data: npt.NDArray[np.float64],
        n_levels: int = 20,
        config: Optional[HFTLOBEnvConfig] = None,
    ) -> None:
        """
        Args:
            lob_data  : LOB 数据，shape (T, 2*n_levels, 2)，列为 [price, volume]
                        LOB data, shape (T, 2*n_levels, 2), columns = [price, volume]
            n_levels  : 订单簿档位数 / Number of price levels
            config    : 环境配置 / Environment configuration
        """
        self.n_levels = n_levels
        self.config   = config or HFTLOBEnvConfig()
        self._rng      = np.random.default_rng(self.config.seed)

        # Copy data / 深拷贝避免外部修改
        self._lob: np.ndarray = copy.deepcopy(lob_data.astype(np.float64))
        if self._lob.ndim != 3 or self._lob.shape[1] != 2 * n_levels or self._lob.shape[2] != 2:
            raise ValueError(
                f"lob_data must be shape (T, {2*n_levels}, 2) = (snapshots, bid+ask, price+vol), "
                f"got {self._lob.shape}"
            )
        self._T = self._lob.shape[0]

        # Precompute feature extractor / 预计算特征提取器
        self._feature_extractor = OrderBookFeatureExtractor(
            feature_type="micro",
            n_levels=n_levels,
        )

        # Observation & action spaces / 空间定义
        # volumes (flattened window) + position + m2m
        self.observation_space = gymnasium.spaces.Dict({
            "volumes": gymnasium.spaces.Box(
                low=0, high=1, shape=(self.config.last_n_ticks * 2 * n_levels,), dtype=np.float32
            ),
            "position": gymnasium.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            ),
        })
        if self.config.use_m2m:
            self.observation_space.spaces["m2m"] = gymnasium.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )

        self.action_space = gymnasium.spaces.Discrete(4)

        # Internal state / 内部状态
        self._reset_internals()

    # ------------------------------------------------------------------ #
    # Gymnasium interface
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境到初始状态。
        Reset environment to initial state.

        Returns:
            state : dict with keys volumes, position, (m2m if enabled)
            info  : empty dict (Gymnasium compatibility)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_internals()
        return self.get_state(), {}

    def step(
        self, action: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作。
        Execute one action.

        Returns:
            state      : next state dict
            reward     : float reward
            terminated : bool (episode ended)
            truncated  : bool (always False — no time limit here)
            info       : dict with diagnostic keys
        """
        self._pos += 1
        closed = False

        # Close on last tick / 到数据末尾自动平仓
        if self.config.last_n_ticks + self._pos >= self._T:
            self._done = True
            if self._position == Position.short:
                action = ACTION_SELL
            elif self._position == Position.long:
                action = ACTION_BUY
            else:
                action = ACTION_HOLD

        self._update_market_state()
        reward = 0.0

        # Action routing / 动作分发
        if action == ACTION_SELL:
            if self._position == Position.neutral:
                self._open_position(Position.short, self._sell_price, self._sell_price_scaled)
            elif self._position == Position.long:
                reward, closed = self._close_position(self._sell_price), True
            elif self._position == Position.short and not self._eval_mode:
                reward = -0.5 * (self._sell_price + self._buy_price)

        elif action == ACTION_BUY:
            if self._position == Position.neutral:
                self._open_position(Position.long, self._buy_price, self._buy_price_scaled)
            elif self._position == Position.short:
                reward, closed = self._close_position(self._buy_price), True
            elif self._position == Position.long and not self._eval_mode:
                reward = -0.5 * (self._sell_price + self._buy_price)

        elif action == ACTION_STOP:
            if not self._position == Position.neutral:
                unrealised = (
                    self._sell_price - self._price
                    if self._position == Position.long
                    else self._price - self._buy_price
                )
                if (self._day_pnl + unrealised < 0) and (unrealised < 0):
                    close_val = self._buy_price if self._position == Position.short else self._sell_price
                    reward, closed = self._close_position(close_val), True
                    self._done = True
                if not self._eval_mode:
                    reward = -0.5 * (self._sell_price + self._buy_price)

        next_state = self.get_state()
        info = {
            "closed":      closed,
            "open_pos":    self._opened,
            "closed_pos":  self._closed,
            "closed_pos_type": int(self._closed_pos_type),
            "action":      action,
            "position":    int(self._position),
            "day_pnl":     self._day_pnl,
        }
        return next_state, float(reward), self._done, False, info

    # Alias for non-Gymnasium usage / 非 Gymnasium 风格的简写
    def execute_action(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        执行动作的简写版本（无 Gymnasium 5-元组包装）。
        Shorthand for execute_action() without Gymnasium 5-tuple wrapping.
        """
        state, reward, done, trunc, info = self.step(action)
        return state, reward, done, info

    # ------------------------------------------------------------------ #
    # State access
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        返回当前 RL 状态。
        Return current RL state dict.

        Returns:
            dict with:
                volumes   : (last_n_ticks * 2 * n_levels,) flattened volumes
                position  : (1,) current position as float
                m2m       : (1,) mark-to-market (if use_m2m=True)
        """
        t = self._pos
        window = self.config.last_n_ticks
        n = self.n_levels

        # Extract volumes from the last `window` snapshots
        vol_list: List[np.ndarray] = []
        for i in range(max(0, t - window + 1), t + 1):
            snap = self._lob[i]
            bid_v = snap[:n, 1]   # bid volumes
            ask_v = snap[n:, 1]   # ask volumes
            vol_list.append(np.concatenate([bid_v, ask_v]))

        # Pad if window not fully covered yet
        while len(vol_list) < window:
            vol_list.insert(0, np.zeros(2 * n, dtype=np.float32))

        flat_vols = np.concatenate(vol_list).astype(np.float32)

        state: Dict[str, np.ndarray] = {
            "volumes":  flat_vols,
            "position": np.array([float(int(self._position))], dtype=np.float32),
        }
        if self.config.use_m2m:
            state["m2m"] = np.array([self._m2m_value], dtype=np.float32)

        return state

    def compute_pnl(self) -> float:
        """
        计算当日累计盈亏（单位：价格原始单位）。
        Compute cumulative day PnL in price units.
        """
        return self._day_pnl

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _reset_internals(self) -> None:
        """重置所有内部状态变量。Reset all internal state variables."""
        self._pos: int          = 0
        self._done: bool        = False
        self._position: Position = Position.neutral
        self._price: float       = 0.0
        self._price_scaled: float = 0.0
        self._buy_price: float   = 0.0
        self._sell_price: float  = 0.0
        self._buy_price_scaled: float = 0.0
        self._sell_price_scaled: float = 0.0
        self._m2m_value: float   = 0.0
        self._day_pnl: float     = 0.0
        self._opened: int        = 0
        self._closed: int        = 0
        self._closed_pos_type: Position = Position.neutral
        self._eval_mode: bool    = False

        # Scaled / unscaled data references
        self._scaled_snap:  np.ndarray = self._lob[:self.config.last_n_ticks]
        self._unscaled_snap: np.ndarray = self._lob[:self.config.last_n_ticks]

    def _update_market_state(self) -> None:
        """根据当前 tick 更新买卖报价及缩放值。Update best bid/ask from current tick."""
        n = self.n_levels
        t = self._pos
        if t >= self._T:
            return
        snap        = self._lob[t]
        self._buy_price        = snap[0, 0]       # best bid (for buying, i.e. you sell at bid)
        self._sell_price       = snap[n, 0]       # best ask (for selling, i.e. you buy at ask)
        self._buy_price_scaled  = self._buy_price  # placeholder — scale with running stats in production
        self._sell_price_scaled = self._sell_price

        # Mark-to-market
        if self._position != Position.neutral:
            self._m2m_value = _m2m(self._position, self._price, self._buy_price, self._sell_price)
        else:
            self._m2m_value = 0.0

    def _open_position(self, pos: Position, value: float, scaled_value: float) -> None:
        """开仓。Open a position."""
        self._position = pos
        self._opened    = self._pos
        self._price     = value
        self._price_scaled = scaled_value

    def _close_position(self, value: float) -> Tuple[float, bool]:
        """平仓，返回平仓盈亏。Close position; returns PnL and closed=True."""
        reward = (
            value - self._price
            if self._position == Position.long
            else self._price - value
        )
        # Apply transaction cost / 扣除交易成本
        cost = self.config.transaction_cost_pct * (value + self._price)
        reward -= cost

        self._closed            = self._pos
        self._closed_pos_type    = self._position
        self._day_pnl          += reward
        self._position          = Position.neutral
        self._price             = 0.0
        self._price_scaled      = 0.0
        self._m2m_value          = 0.0
        return reward, True


# --------------------------------------------------------------------------- #
# HFTPolicy — Pure NumPy neural network for HFT decisions
# --------------------------------------------------------------------------- #

class HFTPolicy:
    """
    纯 NumPy 实现的 HFT 策略网络（支持 DQN / A2C / PPO）。
    Pure NumPy policy network for HFT decisions.

    Implements:
    - Dense (Linear) layer
    - ReLU activation
    - Softmax (for policy head in A2C/PPO)
    - MSE loss helper (for value head)
    - Xavier weight initialisation

    This replaces PyTorch/tensorflow networks while providing identical接口。

    Attributes:
        input_dim      : 输入特征维度 / Input feature dimension
        hidden_dims    : 隐藏层维度列表 / List of hidden layer sizes
        output_dim     : 输出维度（动作数）/ Output dimension (num actions)
        weights        : list of (W, b) tuples
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim  = output_dim
        self._rng        = np.random.default_rng(seed)
        self.weights: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build_network()

    # ------------------------------------------------------------------ #
    # Network construction
    # ------------------------------------------------------------------ #

    def _xavier(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot 初始化。Xavier weight initialisation."""
        fan_in, fan_out = shape[1], shape[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return self._rng.normal(0, std, shape).astype(np.float32)

    def _build_network(self) -> None:
        """构建全连接网络：input → hidden → ... → output。Build fully-connected network."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.weights = []
        for i in range(len(dims) - 1):
            W = self._xavier((dims[i + 1], dims[i]))
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self.weights.append((W, b))

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(self, x: np.ndarray, output_activation: Literal["none", "softmax"] = "none") -> np.ndarray:
        """
        前向传播。
        Forward pass.

        Args:
            x                  : (batch, input_dim) or (input_dim,) input
            output_activation  : "none" (DQN value) or "softmax" (A2C/PPO policy)

        Returns:
            (batch, output_dim) or (output_dim,) output logits / probabilities
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        out = x
        for i, (W, b) in enumerate(self.weights):
            out = out @ W.T + b
            if i < len(self.weights) - 1:   # hidden layers → ReLU
                out = np.maximum(out, 0.0)

        if output_activation == "softmax":
            out = self._softmax(out)
        return out

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax。数值稳定的 softmax。"""
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------ #
    # Loss helpers (for training)
    # ------------------------------------------------------------------ #

    def mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方误差损失。Mean squared error loss."""
        return float(np.mean((y_true - y_pred) ** 2))

    def cross_entropy_loss(self, pi: np.ndarray, a: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        策略交叉熵损失（用于 A2C/PPO）。
        Policy cross-entropy loss (for A2C/PPO).

        Args:
            pi   : (batch, action_dim) action probabilities
            a    : (batch,) action indices
            weights : (batch,) importance sampling weights (optional)
        """
        if pi.ndim == 1:
            pi = pi[np.newaxis, :]
        actions = np.asarray(a, dtype=np.int32)
        log_probs = np.log(pi[np.arange(pi.shape[0]), actions] + 1e-12)
        if weights is not None:
            return float(-np.sum(weights * log_probs) / (np.sum(weights) + 1e-12))
        return float(-np.mean(log_probs))

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回所有 (W, b) 参数。Return all (W, b) parameters."""
        return list(self.weights)

    def set_params(self, weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """从外部设置参数（用于加载模型）。Set parameters from external source (for loading)."""
        self.weights = list(weights)

    def copy_from(self, other: "HFTPolicy") -> None:
        """从另一个 HFTPolicy 深拷贝参数。Deep-copy parameters from another HFTPolicy."""
        self.weights = [(W.copy(), b.copy()) for W, b in other.weights]


# --------------------------------------------------------------------------- #
# DeepHFTAgent — DQN / A2C / PPO implementations in pure NumPy
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    """
    经验回放缓冲区（用于 DQN）。Experience replay buffer for DQN.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._rng     = np.random.default_rng(seed)
        self._buffer: List[Dict[str, Any]] = []
        self._pos     = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加一条经验。Add one experience tuple."""
        entry = {
            "state":      state,
            "action":     action,
            "reward":     reward,
            "next_state": next_state,
            "done":       done,
        }
        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
        else:
            self._buffer[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """随机采样一个 batch。Randomly sample a batch."""
        indices = self._rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        batch = [self._buffer[i] for i in indices]
        return {
            "states":      np.stack([b["state"]      for b in batch]),
            "actions":     np.array([b["action"]     for b in batch], dtype=np.int32),
            "rewards":     np.array([b["reward"]     for b in batch], dtype=np.float32),
            "next_states": np.stack([b["next_state"] for b in batch]),
            "dones":       np.array([b["done"]       for b in batch], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)


class A2CBuffer:
    """
    A2C / PPO 的 on-policy rollout 缓冲区。
    On-policy rollout buffer for A2C / PPO.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam   = lam
        self._states:      List[np.ndarray] = []
        self._actions:      List[int]        = []
        self._rewards:      List[float]      = []
        self._dones:        List[bool]        = []
        self._log_probs:    List[float]       = []
        self._values:       List[float]       = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._log_probs.append(log_prob)
        self._values.append(value)

    def compute_returns_and_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 GAElambda returns and advantages。
        Compute GAE-lambda returns and advantages.
        """
        T = len(self._rewards)
        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            delta = self._rewards[t] + self.gamma * next_value * (1 - self._dones[t]) - self._values[t]
            gae = delta + self.gamma * self.lam * (1 - self._dones[t]) * gae
            advantages[t] = gae
            returns[t]    = gae + self._values[t]
            next_value    = self._values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def clear(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._log_probs.clear()
        self._values.clear()

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (states, actions, returns, advantages)。"""
        returns, advantages = self.compute_returns_and_advantages()
        return (
            np.stack(self._states),
            np.array(self._actions, dtype=np.int32),
            returns,
            advantages,
        )


class DeepHFTAgent:
    """
    深度强化学习 HFT 智能体，支持 DQN / A2C / PPO 三种算法，全部使用纯 NumPy 实现。
    Deep RL HFT agent supporting DQN, A2C, and PPO — pure NumPy, no PyTorch/TF.

    Based on Briola et al. (2021) framework adapted for:
    - Pure NumPy computation (no autograd)
    - Gymnasium-compatible HFTLOBEnvironment

    Attributes:
        algo             : "dqn" | "a2c" | "ppo"
        state_dim        : 状态特征维度 / State feature dimension
        action_dim       : 动作数量 (=4) / Number of actions
        hidden_dims      : 策略网络隐藏层 / Policy network hidden layers
        device           : "cpu" only (pure NumPy)

    Training example:
        >>> agent = DeepHFTAgent(state_dim=201, action_dim=4, algo="ppo")
        >>> for episode in range(100):
        ...     state = env.reset()[0]
        ...     done = False
        ...     while not done:
        ...         action, log_prob, value = agent.select_action(state)
        ...         next_state, reward, done, _, _ = env.step(action)
        ...         agent.store(state, action, reward, done, log_prob, value)
        ...         state = next_state
        ...     agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        algo: Literal["dqn", "a2c", "ppo"] = "dqn",
        seed: Optional[int] = None,
        # DQN specific
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        # A2C/PPO specific
        ppo_epochs: int = 4,
        ppo_clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        lr: float = 3e-4,
        # Replay buffer
        replay_capacity: int = 50_000,
        batch_size: int = 128,
    ) -> None:
        if algo not in ("dqn", "a2c", "ppo"):
            raise ValueError(f"Unsupported algo: {algo}")
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.algo      = algo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self._rng      = np.random.default_rng(seed)

        # Hyperparameters
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.target_update_freq = target_update_freq
        self.ppo_epochs       = ppo_epochs
        self.ppo_clip_eps     = ppo_clip_eps
        self.ent_coef         = ent_coef
        self.vf_coef          = vf_coef
        self.lr               = lr
        self.replay_capacity  = replay_capacity
        self.batch_size       = batch_size

        # Optimiser state (Adam-style, pure NumPy)
        self._m: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._t: int = 0

        # Build networks
        self._build_networks()

        # Buffers
        if algo == "dqn":
            self._replay = ReplayBuffer(replay_capacity, seed=seed)
        else:
            self._rollout = A2CBuffer(gamma=gamma)

        self._step_count: int = 0
        self._total_training_steps: int = 0

    # ------------------------------------------------------------------ #
    # Network setup
    # ------------------------------------------------------------------ #

    def _build_networks(self) -> None:
        """初始化策略网络 + 目标网络 (DQN) / 值网络。"""
        self.policy = HFTPolicy(
            input_dim=self.state_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.action_dim,
            seed=None,
        )

        if self.algo == "dqn":
            self.target_net = HFTPolicy(
                input_dim=self.state_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.action_dim,
                seed=None,
            )
            self.target_net.copy_from(self.policy)
            self.target_net.weights = [(W.copy(), b.copy()) for W, b in self.policy.weights]

        elif self.algo in ("a2c", "ppo"):
            self.value_net = HFTPolicy(
                input_dim=self.state_dim,
                hidden_dims=self.hidden_dims,
                output_dim=1,
                seed=None,
            )

        # Adam optimiser moments for each (W, b)
        self._m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.policy.weights]
        self._v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.policy.weights]

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        state: Dict[str, np.ndarray],
        training: bool = True,
    ) -> Tuple[int, Optional[float], Optional[float]]:
        """
        根据当前状态选择动作。
        Select action given current state.

        Args:
            state   : HFTLOBEnvironment.get_state() 返回的状态字典
            training: 是否在训练模式（用于 epsilon-greedy 探索）

        Returns:
            action    : 动作索引 (0-3)
            log_prob  : 动作 log 概率 (A2C/PPO) 或 None (DQN)
            value     : 状态价值估计 (A2C/PPO) 或 None (DQN)
        """
        x = self._flatten_state(state)
        logits = self.policy.forward(x, output_activation="none").ravel()

        if self.algo == "dqn":
            if training and self._rng.random() < self.epsilon:
                action = int(self._rng.integers(0, self.action_dim))
            else:
                action = int(np.argmax(logits))
            return action, None, None

        # A2C / PPO — policy + value
        pi = HFTPolicy._softmax(logits)
        value = float(self.value_net.forward(x, output_activation="none").ravel()[0])

        if training:
            action = int(self._rng.choice(self.action_dim, p=pi))
            log_prob = float(np.log(pi[action] + 1e-12))
        else:
            action = int(np.argmax(pi))
            log_prob = None

        return action, log_prob, value

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def store(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        """
        存储一步经验（on-policy 或 replay）。
        Store one step of experience.
        """
        flat_s = self._flatten_state(state)
        if self.algo == "dqn":
            self._replay.add(flat_s, action, reward, flat_s, done)   # next_state set later; here we use flat_s as placeholder
        else:
            self._rollout.add(
                flat_s, action, reward, done,
                log_prob=log_prob or 0.0,
                value=value or 0.0,
            )

    def update(self) -> Dict[str, float]:
        """
        执行一轮参数更新（DQN / A2C / PPO）。
        Perform one update step.

        Returns:
            dict of training metrics (loss, entropy, etc.)
        """
        if self.algo == "dqn":
            return self._update_dqn()
        elif self.algo == "a2c":
            return self._update_a2c()
        else:
            return self._update_ppo()

    def _update_dqn(self) -> Dict[str, float]:
        """DQN 更新（经验回放 + 目标网络）。"""
        if len(self._replay) < self.batch_size:
            return {"loss": 0.0}

        batch = self._replay.sample(self.batch_size)
        s     = batch["states"]
        a     = batch["actions"]
        r     = batch["rewards"]
        ns    = batch["next_states"]
        d     = batch["dones"]

        # Current Q values
        q_vals = self.policy.forward(s)                          # (B, action_dim)
        q_sa   = q_vals[np.arange(len(a)), a]                     # (B,)

        # Target Q values (double DQN)
        with np.errstate(divide='ignore', invalid='ignore'):
            next_q_online = self.policy.forward(ns)               # (B, action_dim)
            next_action   = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_net.forward(ns)           # (B, action_dim)
            next_q        = next_q_target[np.arange(len(a)), next_action]

        target_q = r + self.gamma * (1 - d) * next_q

        # MSE loss
        loss = float(np.mean((q_sa - target_q) ** 2))

        # Gradient descent step (Adam)
        grads = self._compute_dqn_gradients(s, a, target_q)
        self._apply_gradients(grads)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodic target network update
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.copy_from(self.policy)

        self._total_training_steps += 1
        return {"loss": loss, "epsilon": self.epsilon}

    def _update_a2c(self) -> Dict[str, float]:
        """A2C 更新（策略梯度 + 值函数估计）。"""
        if len(self._rollout._rewards) < self.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states, actions, returns, advantages = self._rollout.get_all()
        pi_old = self.policy.forward(states, output_activation="softmax")
        log_probs_old = np.log(pi_old[np.arange(len(actions)), actions] + 1e-12)

        policy_loss_total = 0.0
        value_loss_total  = 0.0
        entropy_total     = 0.0

        # Mini-batch updates
        n = len(states)
        indices = np.arange(n)
        self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            mb_idx = indices[start:start + self.batch_size]
            mb_states   = states[mb_idx]
            mb_actions  = actions[mb_idx]
            mb_returns  = returns[mb_idx]
            mb_advs     = advantages[mb_idx]

            pi_new = self.policy.forward(mb_states, output_activation="softmax")
            values = self.value_net.forward(mb_states).ravel()

            # Policy loss (REINFORCE with baseline = A2C)
            log_probs_new = np.log(pi_new[np.arange(len(mb_actions)), mb_actions] + 1e-12)
            ratio = np.exp(log_probs_new - log_probs_old[mb_idx])
            policy_loss = -(ratio * mb_advs).mean()

            # Entropy bonus
            ent = -np.sum(pi_new * np.log(pi_new + 1e-12), axis=-1).mean()
            entropy_total += float(ent)

            # Value loss
            value_loss = float(np.mean((values - mb_returns) ** 2))

            # Combined loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

            # Optimise
            grads = self._compute_policy_value_gradients(mb_states, mb_actions, mb_returns, mb_advs)
            self._apply_gradients(grads)

            policy_loss_total += float(policy_loss)
            value_loss_total  += value_loss

        self._rollout.clear()
        self._total_training_steps += 1
        return {
            "policy_loss": policy_loss_total,
            "value_loss":  value_loss_total,
            "entropy":     entropy_total,
        }

    def _update_ppo(self) -> Dict[str, float]:
        """PPO 更新（截断策略优化）。"""
        if len(self._rollout._rewards) < self.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0}

        states, actions, returns, advantages = self._rollout.get_all()
        n = len(states)

        clip_frac_total = 0.0
        policy_loss_total = 0.0
        value_loss_total  = 0.0

        for _ in range(self.ppo_epochs):
            indices = np.arange(n)
            self._rng.shuffle(indices)

            for start in range(0, n, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]
                mb_states  = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs    = advantages[mb_idx]

                pi_old = self.policy.forward(mb_states, output_activation="softmax")
                log_probs_old = np.log(pi_old[np.arange(len(mb_actions)), mb_actions] + 1e-12)

                # Policy update
                pi_new = self.policy.forward(mb_states, output_activation="softmax")
                log_probs_new = np.log(pi_new[np.arange(len(mb_actions)), mb_actions] + 1e-12)

                ratio = np.exp(log_probs_new - log_probs_old)
                clipped_ratio = np.clip(ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps)
                clip_frac = np.mean(np.abs(ratio - 1) > self.ppo_clip_eps)
                clip_frac_total += float(clip_frac)

                surr1 = ratio * mb_advs
                surr2 = clipped_ratio * mb_advs
                policy_loss = -np.min(surr1, surr2).mean()

                # Value loss
                values = self.value_net.forward(mb_states).ravel()
                value_loss = float(np.mean((values - mb_returns) ** 2))

                # Entropy
                ent = -np.sum(pi_new * np.log(pi_new + 1e-12), axis=-1).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

                grads = self._compute_policy_value_gradients(mb_states, mb_actions, mb_returns, mb_advs)
                self._apply_gradients(grads)

                policy_loss_total += float(policy_loss)
                value_loss_total  += value_loss

        self._rollout.clear()
        self._total_training_steps += 1
        return {
            "policy_loss":   policy_loss_total,
            "value_loss":    value_loss_total,
            "clip_fraction": clip_frac_total,
        }

    # ------------------------------------------------------------------ #
    # Gradient helpers (numerical, pure NumPy)
    # ------------------------------------------------------------------ #

    def _flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """将状态字典展平为特征向量。Flatten state dict to feature vector."""
        parts = [state["volumes"].flatten(), state["position"].flatten()]
        if "m2m" in state:
            parts.append(state["m2m"].flatten())
        return np.concatenate(parts).astype(np.float32)

    def _compute_dqn_gradients(
        self,
        s: np.ndarray,
        a: np.ndarray,
        target: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        计算 DQN 损失的解析梯度（单样本均方误差）。
        Compute analytical gradients for DQN MSE loss w.r.t. each (W, b).

        Uses chain rule:
            dL/dW = dL/dq_sa * dq_sa/dW
            where q_sa = sum_i W[last] @ h_last * one_hot(a)
        """
        # Forward pass to get hidden activations
        acts = self._forward_to_hidden(s)   # list of (batch, dim) activations
        logits = self.policy.forward(s)     # (batch, action_dim)
        q_sa   = logits[np.arange(len(a)), a]

        # dL/dq_sa = 2 * (q_sa - target) / N
        dL_dq = 2 * (q_sa - target)[:, np.newaxis] / len(a)   # (batch, 1)

        # Backprop through output layer (last weight)
        h = acts[-1]   # (batch, hidden_dim)
        dL_dW_out = dL_dq.T @ h          # (action_dim, hidden_dim)
        dL_db_out = dL_dq.sum(axis=0)    # (action_dim,)

        # Backprop through hidden layers
        grads = [(dL_dW_out, dL_db_out)]
        dL_dh = dL_dq @ self.policy.weights[-1][0]   # (batch, hidden_dim)

        for i in reversed(range(len(self.policy.weights) - 1)):
            W, b = self.policy.weights[i]
            # ReLU derivative: dL/dh = dL/dh_next * relu'(h)
            dL_dh = dL_dh * (acts[i] > 0).astype(float)
            dL_dW = dL_dh.T @ acts[i - 1] if i > 0 else dL_dh.T @ s
            dL_db = dL_dh.sum(axis=0)
            grads.insert(0, (dL_dW, dL_db))
            if i > 0:
                dL_dh = dL_dh @ W

        return grads

    def _compute_policy_value_gradients(
        self,
        s: np.ndarray,
        a: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        计算 A2C/PPO 联合损失 (policy + value + entropy) 的梯度。
        Compute gradients for A2C/PPO joint (policy + value + entropy) loss.
        """
        pi   = self.policy.forward(s, output_activation="softmax")   # (B, A)
        vals = self.value_net.forward(s).ravel()                    # (B,)

        # Policy gradient (REINFORCE baseline)
        log_pi = np.log(pi + 1e-12)
        d_log_pi = np.zeros_like(pi)
        d_log_pi[np.arange(len(a)), a] = 1.0 / (pi[np.arange(len(a)), a] + 1e-12)
        d_policy_loss = -np.outer(advantages, d_log_pi.mean(axis=0))   # approximate; flatten properly below

        # Flatten policy loss gradient
        acts_pol = self._forward_to_hidden(s)
        # dL_pol = advantages (B,) scaled by d_log_pi (B, A)
        dL_pol = (d_log_pi * advantages[:, np.newaxis]).mean(axis=0)   # (A,)
        # Backprop through policy head
        h_pol = acts_pol[-1]
        dL_dW_pol = np.outer(dL_pol, h_pol)
        dL_db_pol = dL_pol

        # Value gradient
        d_val = 2 * (vals - returns) / len(returns)
        acts_val = self._forward_to_hidden(s, use_value_net=True)
        h_val = acts_val[-1]
        dL_dW_val = np.outer(d_val, h_val)
        dL_db_val = d_val

        grads = [(dL_dW_pol, dL_db_pol), (dL_dW_val, dL_db_val)]
        return grads

    def _forward_to_hidden(
        self,
        x: np.ndarray,
        use_value_net: bool = False,
    ) -> List[np.ndarray]:
        """返回除输出层外的所有隐藏激活值。Return hidden activations (excl. output)."""
        net = self.value_net if use_value_net else self.policy
        acts = [x]
        for i, (W, b) in enumerate(net.weights):
            if i == len(net.weights) - 1:
                break
            h = np.maximum(acts[-1] @ W.T + b, 0.0)
            acts.append(h)
        return acts[1:]   # drop input

    def _apply_gradients(
        self,
        grads: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        使用 Adam 优化器将梯度应用到策略网络。
        Apply gradients to policy net using Adam.
        """
        net = self.policy
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for (W, b), (dW, db), (mW, mb), (vW, vb) in zip(
            net.weights, grads, self._m, self._v
        ):
            # Adam update
            mW_new = beta1 * mW + (1 - beta1) * dW
            vW_new = beta2 * vW + (1 - beta2) * dW ** 2
            mb_new = beta1 * mb + (1 - beta1) * db
            vb_new = beta2 * vb + (1 - beta2) * db ** 2

            mW_hat = mW_new / (1 - beta1 ** self._t)
            vW_hat = vW_new / (1 - beta2 ** self._t)
            mb_hat = mb_new / (1 - beta1 ** self._t)
            vb_hat = vb_new / (1 - beta2 ** self._t)

            W_new = W - self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b_new = b - self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            W[:] = W_new
            b[:] = b_new
            mW[:] = mW_new
            vW[:] = vW_new
            mb[:] = mb_new
            vb[:] = vb_new

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """
        保存模型参数到 .npz 文件。
        Save model parameters to .npz file.
        """
        arrays = {}
        for i, (W, b) in enumerate(self.policy.weights):
            arrays[f"W_{i}"] = W
            arrays[f"b_{i}"] = b
        if self.algo == "dqn":
            for i, (W, b) in enumerate(self.target_net.weights):
                arrays[f"tw_{i}"] = W
                arrays[f"tb_{i}"] = b
        elif self.algo in ("a2c", "ppo"):
            for i, (W, b) in enumerate(self.value_net.weights):
                arrays[f"vw_{i}"] = W
                arrays[f"vb_{i}"] = b
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """
        从 .npz 文件加载模型参数。
        Load model parameters from .npz file.
        """
        data = np.load(path)
        for i, (W, b) in enumerate(self.policy.weights):
            W[:] = data[f"W_{i}"]
            b[:] = data[f"b_{i}"]
        if self.algo == "dqn":
            for i, (W, b) in enumerate(self.target_net.weights):
                W[:] = data[f"tw_{i}"]
                b[:] = data[f"tb_{i}"]
        elif self.algo in ("a2c", "ppo"):
            for i, (W, b) in enumerate(self.value_net.weights):
                W[:] = data[f"vw_{i}"]
                b[:] = data[f"vb_{i}"]


# --------------------------------------------------------------------------- #
# Convenience factory / 便捷工厂函数
# --------------------------------------------------------------------------- #

def make_hft_env(
    lob_data: npt.NDArray[np.float64],
    n_levels: int = 20,
    last_n_ticks: int = 10,
    seed: Optional[int] = None,
) -> HFTLOBEnvironment:
    """
    创建 HFT LOB 环境的便捷函数。
    Convenience factory for HFTLOBEnvironment.

    Example:
        >>> lob = np.random.randn(1000, 40, 2).astype(np.float64)
        >>> env = make_hft_env(lob, n_levels=20, last_n_ticks=10)
    """
    config = HFTLOBEnvConfig(last_n_ticks=last_n_ticks, seed=seed)
    return HFTLOBEnvironment(lob_data=lob_data, n_levels=n_levels, config=config)


# --------------------------------------------------------------------------- #
# Stub for gymnasium import (delayed to avoid hard dependency at module top)
# --------------------------------------------------------------------------- #
try:
    import gymnasium
except ImportError:
    gymnasium = None   # type: ignore
