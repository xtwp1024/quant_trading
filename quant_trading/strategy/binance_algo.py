"""
Binance Algorithmic Trading - 综合量化交易模块
==============================================

本模块整合了来自 binance-algotrading 仓库的核心功能：
- Gymnasium 兼容的交易环境 (TradingGymEnv)
- 遗传算法优化器 (GeneticOptimizer)
- 移动平均线交叉策略 (MACrossoverStrategy)
- RSI 策略 (RSIStrategy)
- 布林带策略 (BollingerStrategy)
- 策略参数优化器 (StrategyOptimizer)

纯 NumPy + Gymnasium 实现，无 Cython 依赖。

This module integrates core functionalities from binance-algotrading repository:
- Gymnasium-compatible trading environment (TradingGymEnv)
- Genetic algorithm optimizer (GeneticOptimizer)
- Moving Average Crossover Strategy (MACrossoverStrategy)
- RSI Strategy (RSIStrategy)
- Bollinger Band Strategy (BollingerStrategy)
- Strategy parameter optimizer (StrategyOptimizer)

Pure NumPy + Gymnasium implementation, no Cython dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# =============================================================================
# 常量与类型定义 / Constants and Type Definitions
# =============================================================================

# 动作定义 / Action definitions
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

# 策略类型 / Strategy types
STRATEGY_MA_CROSSOVER = "ma_crossover"
STRATEGY_RSI = "rsi"
STRATEGY_BOLLINGER = "bollinger"

# 默认遗传算法参数 / Default GA parameters
DEFAULT_POP_SIZE = 50
DEFAULT_N_GEN = 100
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.7
DEFAULT_ELITE_RATIO = 0.1


# =============================================================================
# 技术指标计算 / Technical Indicators (Pure NumPy)
# =============================================================================

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    计算简单移动平均线 / Simple Moving Average

    Args:
        data: 价格数据数组 / Price data array
        period: 移动平均周期 / Moving average period

    Returns:
        SMA 数组，周期不足处为 NaN / SMA array, NaN where period insufficient
    """
    sma = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    计算指数移动平均线 / Exponential Moving Average

    Args:
        data: 价格数据数组 / Price data array
        period: EMA 周期 / EMA period

    Returns:
        EMA 数组 / EMA array
    """
    ema = np.full_like(data, np.nan)
    alpha = 2.0 / (period + 1)
    # Find first valid value
    first_valid = 0
    for i in range(len(data)):
        if not np.isnan(data[i]):
            first_valid = i
            ema[i] = data[i]
            break
    # Calculate EMA
    for i in range(first_valid + 1, len(data)):
        if not np.isnan(data[i]):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算相对强弱指数 / Relative Strength Index

    Args:
        prices: 价格数组 / Price array
        period: RSI 周期 / RSI period (default: 14)

    Returns:
        RSI 数组，值为 0-100 / RSI array, values 0-100
    """
    deltas = np.diff(prices, axis=0)
    # Handle the first element which is NaN after diff
    deltas = np.insert(deltas, 0, np.nan)

    rsi = np.full_like(prices, np.nan)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calculate first average
    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent values using smoothed average
    for i in range(period + 1, len(prices)):
        if np.isnan(prices[i]):
            continue
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算布林带 / Bollinger Bands

    Args:
        prices: 价格数组 / Price array
        period: 移动平均周期 / MA period (default: 20)
        num_std: 标准差倍数 / Number of standard deviations (default: 2.0)

    Returns:
        (上轨, 中轨, 下轨) 元组 / Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, period)

    # Calculate rolling standard deviation
    std_dev = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        if not np.isnan(middle[i]):
            std_dev[i] = np.std(prices[i - period + 1:i + 1])

    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    return upper, middle, lower


def calculate_macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 MACD / Moving Average Convergence Divergence

    Args:
        prices: 价格数组 / Price array
        fast_period: 快线周期 / Fast MA period (default: 12)
        slow_period: 慢线周期 / Slow MA period (default: 26)
        signal_period: 信号线周期 / Signal line period (default: 9)

    Returns:
        (MACD 线, 信号线, 柱状图) 元组 / Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    计算平均真实波幅 / Average True Range

    Args:
        high: 最高价数组 / High price array
        low: 最低价数组 / Low price array
        close: 收盘价数组 / Close price array
        period: ATR 周期 / ATR period (default: 14)

    Returns:
        ATR 数组 / ATR array
    """
    atr = np.full_like(high, np.nan)

    # Calculate True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # First ATR is simple average of TR
    atr[period - 1] = np.mean(tr[:period])

    # Subsequent ATRs use smoothed average
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# =============================================================================
# 数据类 / Data Classes
# =============================================================================

@dataclass
class TradingState:
    """
    交易状态数据类 / Trading state data class
    """
    balance: float = 1000.0          # 当前余额 / Current balance
    position: float = 0.0            # 持仓数量 / Position quantity
    entry_price: float = 0.0         # 入场价格 / Entry price
    total_trades: int = 0            # 总交易次数 / Total number of trades
    winning_trades: int = 0           # 盈利交易次数 / Winning trades
    losing_trades: int = 0           # 亏损交易次数 / Losing trades
    total_profit: float = 0.0        # 总利润 / Total profit
    max_drawdown: float = 0.0        # 最大回撤 / Maximum drawdown
    cumulative_fee: float = 0.0      # 累计手续费 / Cumulative fees


@dataclass
class BacktestResult:
    """
    回测结果数据类 / Backtest result data class
    """
    final_balance: float
    total_return: float              # 总收益率 / Total return
    sharpe_ratio: float              # 夏普比率 / Sharpe ratio
    max_drawdown: float              # 最大回撤 / Maximum drawdown
    num_trades: int                  # 交易次数 / Number of trades
    win_rate: float                  # 胜率 / Win rate
    avg_profit: float                # 平均盈利 / Average profit
    avg_loss: float                  # 平均亏损 / Average loss
    params: Dict[str, Any] = field(default_factory=dict)  # 策略参数 / Strategy parameters


# =============================================================================
# 交易环境 / Trading Environment
# =============================================================================

class TradingGymEnv(gym.Env):
    """
    Gymnasium 兼容的交易环境 / Gymnasium-compatible trading environment

    支持现货交易模拟，包含完整的订单执行、盈亏计算和风险管理功能。

    Features spot trading simulation with complete order execution,
    profit/loss calculation, and risk management.

    Observation Space:
        - 价格数据 (OHLCV) / Price data (OHLCV)
        - 技术指标值 / Technical indicator values
        - 账户状态 (余额、持仓) / Account state (balance, position)

    Action Space:
        - 0: 持有 / Hold
        - 1: 买入 / Buy
        - 2: 卖出 / Sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: np.ndarray,
        init_balance: float = 1000.0,
        fee: float = 0.0005,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_steps: int = 0,
        verbose: bool = False,
        **kwargs
    ):
        """
        初始化交易环境 / Initialize trading environment

        Args:
            df: 包含 OHLCV 数据的 numpy 数组，形状为 (n, 5) 或 (n, 6)
                第一列可选为时间戳，其余列为 O/H/L/C/V
                / numpy array with OHLCV data, shape (n, 5) or (n, 6)
                First column optional timestamp, rest are O/H/L/C/V
            init_balance: 初始资金 / Initial balance (default: 1000.0)
            fee: 交易手续费率 / Trading fee rate (default: 0.0005 = 0.05%)
            stop_loss: 止损百分比 (可选) / Stop loss percentage (optional)
            take_profit: 止盈百分比 (可选) / Take profit percentage (optional)
            max_steps: 最大交易步数，0 表示使用全部数据
                       / Maximum trading steps, 0 uses all data
            verbose: 是否打印详细信息 / Print detailed information
        """
        super().__init__()

        # 处理输入数据 / Process input data
        if df.shape[1] >= 5:
            # Assume first column is timestamp if string dates, otherwise use all 5 columns as OHLCV
            self.df = df[:, -5:].astype(np.float32)  # O, H, L, C, V
        else:
            self.df = df.astype(np.float32)

        self.n_steps = len(self.df)
        self.current_step = 0

        # 环境配置 / Environment configuration
        self.init_balance = init_balance
        self.fee = fee
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_steps = max_steps if max_steps > 0 else self.n_steps
        self.verbose = verbose

        # 账户状态 / Account state
        self.balance = init_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.save_balance = 0.0  # 盈利保存余额 / Profit saving balance

        # 统计信息 / Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_profits = []
        self.max_balance = init_balance
        self.max_drawdown = 0.0
        self.cumulative_fee = 0.0

        # 回合状态 / Episode state
        self.done = False

        # 定义空间 / Define spaces
        # Observation: OHLCV (5) + position_indicator (1) + balance_ratio (1) = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # 交易历史 / Trading history
        self.history = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态 / Reset environment to initial state

        Returns:
            (observation, info) 元组 / (observation, info) tuple
        """
        super().reset(seed=seed)

        self.current_step = random.randint(0, max(0, self.n_steps - self.max_steps - 1))
        self.balance = self.init_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.save_balance = 0.0

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_profits = []
        self.max_balance = self.init_balance
        self.max_drawdown = 0.0
        self.cumulative_fee = 0.0

        self.done = False
        self.history = []

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        """获取当前观察 / Get current observation"""
        obs = np.zeros(7, dtype=np.float32)

        # OHLCV 数据 / OHLCV data
        o, h, l, c, v = self.df[self.current_step]
        obs[0] = o / self.init_balance  # 归一化 / Normalized
        obs[1] = h / self.init_balance
        obs[2] = l / self.init_balance
        obs[3] = c / self.init_balance
        obs[4] = v / (np.max(self.df[:, 4]) + 1e-8)  # 归一化成交量 / Normalized volume

        # 持仓指示器 / Position indicator
        obs[5] = 1.0 if self.position > 0 else 0.0

        # 余额比例 / Balance ratio
        obs[6] = self.balance / self.init_balance

        return obs

    def _get_info(self) -> Dict:
        """获取额外信息 / Get additional info"""
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "total_trades": self.total_trades,
            "max_drawdown": self.max_drawdown,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行交易动作 / Execute trading action

        Args:
            action: 动作 (0=持有, 1=买入, 2=卖出) / Action (0=hold, 1=buy, 2=sell)

        Returns:
            (observation, reward, terminated, truncated, info) 元组
            / (observation, reward, terminated, truncated, info) tuple
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()

        current_price = float(self.df[self.current_step, 3])  # 收盘价 / Close price
        reward = 0.0

        # 执行动作 / Execute action
        if action == ACTION_BUY and self.position == 0:
            # 买入开多 / Buy to open long
            self._buy(current_price)
        elif action == ACTION_SELL and self.position > 0:
            # 卖出平多 / Sell to close long
            profit = self._sell(current_price)
            reward = profit / self.init_balance  # 归一化奖励 / Normalized reward

        # 更新最大回撤 / Update max drawdown
        current_total = self.balance + self.position * current_price + self.save_balance
        if current_total > self.max_balance:
            self.max_balance = current_total
        drawdown = (self.max_balance - current_total) / self.max_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # 检查止损/止盈 / Check stop loss/take profit
        if self.position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.stop_loss is not None and pnl_pct <= -self.stop_loss:
                self._sell(current_price, is_stop_loss=True)
                reward = -self.stop_loss
            elif self.take_profit is not None and pnl_pct >= self.take_profit:
                self._sell(current_price, is_take_profit=True)
                reward = self.take_profit

        # 推进步数 / Advance step
        self.current_step += 1

        # 检查结束条件 / Check termination conditions
        self.done = (
            self.current_step >= min(self.start_step + self.max_steps, self.n_steps - 1) or
            self.balance < self.init_balance * 0.1  # 资金不足 / Insufficient funds
        )

        if self.done:
            # 关闭持仓 / Close position
            if self.position > 0:
                self._sell(current_price)
            reward = (self.balance - self.init_balance) / self.init_balance

        return self._get_observation(), reward, self.done, False, self._get_info()

    def _buy(self, price: float) -> None:
        """执行买入 / Execute buy"""
        # 考虑手续费调整的价格 / Fee-adjusted price
        adj_price = price * (1.0 + self.fee)
        quantity = self.balance / adj_price

        if quantity > 0:
            cost = quantity * adj_price
            self.balance -= cost
            self.position = quantity
            self.entry_price = price
            self.cumulative_fee += cost * self.fee
            self.total_trades += 1

    def _sell(self, price: float, is_stop_loss: bool = False, is_take_profit: bool = False) -> float:
        """执行卖出 / Execute sell"""
        if self.position <= 0:
            return 0.0

        adj_price = price * (1.0 - self.fee)
        revenue = self.position * adj_price
        cost = self.position * self.entry_price * (1.0 + self.fee)

        profit = revenue - cost - self.cumulative_fee
        self.balance += revenue

        self.trade_profits.append(profit)
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.position = 0.0
        self.entry_price = 0.0

        return profit

    @property
    def start_step(self) -> int:
        """获取起始步数 / Get start step"""
        return self.current_step

    def render(self, mode: str = "human") -> None:
        """渲染环境状态 / Render environment state"""
        if self.verbose:
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
                  f"Position: {self.position:.6f}, Trades: {self.total_trades}")


# =============================================================================
# 策略类 / Strategy Classes
# =============================================================================

class BaseStrategy:
    """
    策略基类 / Base strategy class
    """

    name: str = "base_strategy"

    def __init__(self, symbol: str = "BTCUSDT", params: Optional[Dict] = None):
        """
        初始化策略 / Initialize strategy

        Args:
            symbol: 交易品种 / Trading symbol
            params: 策略参数 / Strategy parameters
        """
        self.symbol = symbol
        self.params = params or {}
        self.indicators = {}

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        生成交易信号 / Generate trading signals

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            信号数组 (-1=卖出, 0=持有, 1=买入) / Signal array (-1=sell, 0=hold, 1=buy)
        """
        raise NotImplementedError

    def calculate_indicators(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算技术指标 / Calculate technical indicators

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            指标字典 / Indicators dictionary
        """
        raise NotImplementedError


class MACrossoverStrategy(BaseStrategy):
    """
    移动平均线交叉策略 / Moving Average Crossover Strategy

    当短期均线上穿长期均线时买入（黄金交叉），当短期均线下穿时卖出（死亡交叉）。

    Buy when short MA crosses above long MA (golden cross),
    sell when short MA crosses below long MA (death cross).

    Parameters:
        fast_period: 快线周期 / Fast MA period (default: 10)
        slow_period: 慢线周期 / Slow MA period (default: 30)
    """

    name = "ma_crossover"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        params: Optional[Dict] = None
    ):
        default_params = {
            "fast_period": 10,
            "slow_period": 30,
        }
        default_params.update(params or {})
        super().__init__(symbol, default_params)

    def calculate_indicators(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算 MA 交叉指标 / Calculate MA crossover indicators

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            包含快线、慢线的字典 / Dictionary with fast and slow MA
        """
        close = data[:, 3]  # 收盘价 / Close price
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        fast_ma = calculate_sma(close, fast_period)
        slow_ma = calculate_sma(close, slow_period)

        self.indicators = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
        }
        return self.indicators

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        生成 MA 交叉信号 / Generate MA crossover signals

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            信号数组 / Signal array
        """
        indicators = self.calculate_indicators(data)
        fast_ma = indicators["fast_ma"]
        slow_ma = indicators["slow_ma"]

        n = len(data)
        signals = np.zeros(n, dtype=np.int32)

        for i in range(1, n):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                signals[i] = 0
            elif fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
                signals[i] = 1  # 买入信号 / Buy signal
            elif fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]:
                signals[i] = -1  # 卖出信号 / Sell signal
            else:
                signals[i] = 0  # 持有 / Hold

        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI 策略 / RSI Strategy

    当 RSI 低于超卖阈值时买入，当 RSI 高于超买阈值时卖出。

    Buy when RSI is below oversold threshold, sell when RSI is above overbought threshold.

    Parameters:
        rsi_period: RSI 计算周期 / RSI period (default: 14)
        oversold: 超卖阈值 / Oversold threshold (default: 30)
        overbought: 超买阈值 / Overbought threshold (default: 70)
    """

    name = "rsi"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        params: Optional[Dict] = None
    ):
        default_params = {
            "rsi_period": 14,
            "oversold": 30.0,
            "overbought": 70.0,
        }
        default_params.update(params or {})
        super().__init__(symbol, default_params)

    def calculate_indicators(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算 RSI 指标 / Calculate RSI indicator

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            包含 RSI 值的字典 / Dictionary with RSI values
        """
        close = data[:, 3]
        rsi_period = self.params["rsi_period"]

        rsi = calculate_rsi(close, rsi_period)

        self.indicators = {
            "rsi": rsi,
        }
        return self.indicators

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        生成 RSI 交易信号 / Generate RSI trading signals

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            信号数组 / Signal array
        """
        indicators = self.calculate_indicators(data)
        rsi = indicators["rsi"]

        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        n = len(data)
        signals = np.zeros(n, dtype=np.int32)
        in_position = False

        for i in range(n):
            if np.isnan(rsi[i]):
                signals[i] = 0
                continue

            if not in_position:
                if rsi[i] < oversold:
                    signals[i] = 1  # 买入 / Buy
                    in_position = True
                else:
                    signals[i] = 0
            else:
                if rsi[i] > overbought:
                    signals[i] = -1  # 卖出 / Sell
                    in_position = False
                else:
                    signals[i] = 0

        return signals


class BollingerStrategy(BaseStrategy):
    """
    布林带策略 / Bollinger Band Strategy

    当价格触及下轨时买入（超卖），当价格触及上轨时卖出（超买）。

    Buy when price touches lower band (oversold), sell when price touches upper band (overbought).

    Parameters:
        bb_period: 布林带周期 / Bollinger period (default: 20)
        bb_std: 标准差倍数 / Standard deviation multiplier (default: 2.0)
    """

    name = "bollinger"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        params: Optional[Dict] = None
    ):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
        }
        default_params.update(params or {})
        super().__init__(symbol, default_params)

    def calculate_indicators(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算布林带指标 / Calculate Bollinger Band indicators

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            包含布林带的字典 / Dictionary with Bollinger bands
        """
        close = data[:, 3]
        bb_period = self.params["bb_period"]
        bb_std = self.params["bb_std"]

        upper, middle, lower = calculate_bollinger_bands(close, bb_period, bb_std)

        self.indicators = {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }
        return self.indicators

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        生成布林带交易信号 / Generate Bollinger Band trading signals

        Args:
            data: OHLCV 数据 / OHLCV data

        Returns:
            信号数组 / Signal array
        """
        indicators = self.calculate_indicators(data)
        close = data[:, 3]
        lower = indicators["lower"]
        upper = indicators["upper"]

        n = len(data)
        signals = np.zeros(n, dtype=np.int32)
        in_position = False

        for i in range(n):
            if np.isnan(lower[i]) or np.isnan(upper[i]):
                signals[i] = 0
                continue

            if not in_position:
                # 买入条件：价格触及下轨 / Buy: price touches lower band
                if close[i] <= lower[i]:
                    signals[i] = 1
                    in_position = True
                else:
                    signals[i] = 0
            else:
                # 卖出条件：价格触及上轨 / Sell: price touches upper band
                if close[i] >= upper[i]:
                    signals[i] = -1
                    in_position = False
                else:
                    signals[i] = 0

        return signals


# =============================================================================
# 遗传算法优化器 / Genetic Algorithm Optimizer
# =============================================================================

class GeneticOptimizer:
    """
    遗传算法优化器 / Genetic Algorithm Optimizer

    实现完整的遗传算法流程：选择、交叉、变异、精英保留。

    Implements complete genetic algorithm: selection, crossover, mutation, elitism.

    Args:
        population_size: 种群大小 / Population size (default: 50)
        n_generations: 迭代代数 / Number of generations (default: 100)
        mutation_rate: 变异率 / Mutation rate (default: 0.1)
        crossover_rate: 交叉率 / Crossover rate (default: 0.7)
        elite_ratio: 精英比例 / Elite ratio (default: 0.1)
        random_state: 随机种子 / Random seed (optional)
    """

    def __init__(
        self,
        population_size: int = DEFAULT_POP_SIZE,
        n_generations: int = DEFAULT_N_GEN,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        crossover_rate: float = DEFAULT_CROSSOVER_RATE,
        elite_ratio: float = DEFAULT_ELITE_RATIO,
        random_state: Optional[int] = None,
    ):
        self.pop_size = population_size
        self.n_gen = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(population_size * elite_ratio))

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.population: List[Dict[str, Any]] = []
        self.fitness_history: List[float] = []
        self.best_individual: Optional[Dict[str, Any]] = None
        self.best_fitness: float = -np.inf

    def initialize_population(
        self,
        param_bounds: Dict[str, Tuple[Any, Any]]
    ) -> List[Dict[str, Any]]:
        """
        初始化种群 / Initialize population

        Args:
            param_bounds: 参数范围字典，格式为 {param_name: (min, max)}
                         / Parameter bounds dict, format {param_name: (min, max)}

        Returns:
            初始种群 / Initial population
        """
        self.population = []
        self.param_bounds = param_bounds

        for _ in range(self.pop_size):
            individual = {}
            for param_name, (low, high) in param_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    individual[param_name] = random.randint(low, high)
                else:
                    individual[param_name] = random.uniform(low, high)
            self.population.append(individual)

        return self.population

    def _evaluate_fitness(
        self,
        individual: Dict[str, Any],
        fitness_func: callable
    ) -> float:
        """
        评估个体适应度 / Evaluate individual fitness

        Args:
            individual: 个体参数 / Individual parameters
            fitness_func: 适应度函数 / Fitness function

        Returns:
            适应度值 / Fitness value
        """
        return fitness_func(individual)

    def _select_parents(
        self,
        population: List[Dict[str, Any]],
        fitnesses: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        锦标赛选择父代 / Tournament selection for parents

        Args:
            population: 当前种群 / Current population
            fitnesses: 适应度数组 / Fitness array

        Returns:
            选中的父代列表 / Selected parents list
        """
        tournament_size = max(2, self.pop_size // 10)
        parents = []

        for _ in range(2):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            tournament_fitness = fitnesses[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])

        return parents

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        均匀交叉 / Uniform crossover

        Args:
            parent1: 父代1 / Parent 1
            parent2: 父代2 / Parent 2

        Returns:
            (子代1, 子代2) 元组 / (child1, child2) tuple
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {}
        child2 = {}

        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _mutate(
        self,
        individual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        高斯变异 / Gaussian mutation

        Args:
            individual: 个体 / Individual

        Returns:
            变异后的个体 / Mutated individual
        """
        mutated = individual.copy()

        for key, (low, high) in self.param_bounds.items():
            if random.random() < self.mutation_rate:
                if isinstance(low, int) and isinstance(high, int):
                    # 整数参数：随机整数 / Integer parameter: random int
                    delta = int(np.random.normal(0, (high - low) * 0.1))
                    mutated[key] = max(low, min(high, mutated[key] + delta))
                else:
                    # 浮点参数：高斯变异 / Float parameter: gaussian mutation
                    delta = np.random.normal(0, (high - low) * 0.1)
                    mutated[key] = max(low, min(high, mutated[key] + delta))

        return mutated

    def evolve(
        self,
        fitness_func: callable,
        param_bounds: Dict[str, Tuple[Any, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行遗传算法优化 / Execute genetic algorithm optimization

        Args:
            fitness_func: 适应度函数，接收个体字典，返回适应度值
                         / Fitness function, takes individual dict, returns fitness
            param_bounds: 参数范围 / Parameter bounds
            verbose: 是否打印进度 / Print progress

        Returns:
            最优个体 / Best individual
        """
        # 初始化种群 / Initialize population
        if not self.population:
            self.initialize_population(param_bounds)

        self.fitness_history = []
        self.best_fitness = -np.inf

        for generation in range(self.n_gen):
            # 评估适应度 / Evaluate fitness
            fitnesses = np.array([
                self._evaluate_fitness(ind, fitness_func)
                for ind in self.population
            ])

            # 找到最优个体 / Find best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = self.population[best_idx].copy()

            self.fitness_history.append(self.best_fitness)

            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.4f}")

            # 精英保留 / Elitism
            sorted_idx = np.argsort(fitnesses)[::-1]
            elite = [self.population[i] for i in sorted_idx[:self.elite_size]]

            # 生成新种群 / Generate new population
            new_population = elite.copy()

            while len(new_population) < self.pop_size:
                # 选择父代 / Select parents
                parents = self._select_parents(self.population, fitnesses)

                # 交叉 / Crossover
                child1, child2 = self._crossover(parents[0], parents[1])

                # 变异 / Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            # 截断到种群大小 / Truncate to population size
            self.population = new_population[:self.pop_size]

        if verbose:
            print(f"Optimization complete. Best Fitness = {self.best_fitness:.4f}")
            print(f"Best Parameters: {self.best_individual}")

        return self.best_individual

    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最优参数 / Get best parameters

        Returns:
            最优参数字典 / Best parameters dictionary
        """
        return self.best_individual.copy() if self.best_individual else {}


# =============================================================================
# 策略优化器 / Strategy Optimizer
# =============================================================================

class StrategyOptimizer:
    """
    策略参数优化器 / Strategy parameter optimizer

    使用遗传算法优化策略参数，并通过回测评估性能。

    Uses genetic algorithm to optimize strategy parameters and evaluates
    performance through backtesting.

    Args:
        strategy_class: 策略类 / Strategy class
        data: OHLCV 数据 / OHLCV data
        param_bounds: 参数范围 / Parameter bounds
        population_size: 种群大小 / Population size (default: 50)
        n_generations: 迭代代数 / Number of generations (default: 100)
    """

    def __init__(
        self,
        strategy_class: type,
        data: np.ndarray,
        param_bounds: Dict[str, Tuple[Any, Any]],
        population_size: int = DEFAULT_POP_SIZE,
        n_generations: int = DEFAULT_N_GEN,
        init_balance: float = 1000.0,
        fee: float = 0.0005,
        random_state: Optional[int] = None,
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.param_bounds = param_bounds
        self.init_balance = init_balance
        self.fee = fee

        self.optimizer = GeneticOptimizer(
            population_size=population_size,
            n_generations=n_generations,
            random_state=random_state,
        )

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_result: Optional[BacktestResult] = None

    def _backtest(
        self,
        params: Dict[str, Any]
    ) -> BacktestResult:
        """
        回测策略 / Backtest strategy

        Args:
            params: 策略参数 / Strategy parameters

        Returns:
            回测结果 / Backtest result
        """
        # 创建策略和环境 / Create strategy and environment
        strategy = self.strategy_class(params=params)
        env = TradingGymEnv(
            df=self.data,
            init_balance=self.init_balance,
            fee=self.fee,
            verbose=False,
        )

        # 生成信号 / Generate signals
        signals = strategy.generate_signals(self.data)

        # 运行回测 / Run backtest
        obs, _ = env.reset()
        total_reward = 0.0
        rewards = []

        for step in range(len(self.data) - 1):
            action = signals[step] + 1  # Convert -1,0,1 to 0,1,2
            action = max(0, min(2, action))
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward
            if done:
                break

        # 计算回测指标 / Calculate backtest metrics
        final_balance = env.balance
        total_return = (final_balance - self.init_balance) / self.init_balance

        # 夏普比率 / Sharpe ratio
        if len(rewards) > 1 and np.std(rewards) > 0:
            sharpe = np.mean(rewards) / np.std(rewards) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 胜率 / Win rate
        trade_profits = env.trade_profits
        if len(trade_profits) > 0:
            win_rate = len([p for p in trade_profits if p > 0]) / len(trade_profits)
            avg_profit = np.mean([p for p in trade_profits if p > 0]) if any(p > 0 for p in trade_profits) else 0.0
            avg_loss = np.mean([p for p in trade_profits if p < 0]) if any(p < 0 for p in trade_profits) else 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0

        return BacktestResult(
            final_balance=final_balance,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=env.max_drawdown,
            num_trades=env.total_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            params=params,
        )

    def _fitness_function(self, params: Dict[str, Any]) -> float:
        """
        适应度函数 / Fitness function

        Args:
            params: 策略参数 / Strategy parameters

        Returns:
            适应度值（总收益率） / Fitness value (total return)
        """
        result = self._backtest(params)
        # 使用总收益率作为适应度 / Use total return as fitness
        return result.total_return

    def optimize(
        self,
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        执行优化 / Execute optimization

        Args:
            verbose: 是否打印进度 / Print progress

        Returns:
            (最优参数, 最佳回测结果) 元组 / (best_params, best_backtest_result) tuple
        """
        self.best_params = self.optimizer.evolve(
            fitness_func=self._fitness_function,
            param_bounds=self.param_bounds,
            verbose=verbose,
        )

        self.best_result = self._backtest(self.best_params)

        if verbose:
            print("\nOptimization Results:")
            print(f"Best Parameters: {self.best_params}")
            print(f"Total Return: {self.best_result.total_return * 100:.2f}%")
            print(f"Sharpe Ratio: {self.best_result.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {self.best_result.max_drawdown * 100:.2f}%")
            print(f"Number of Trades: {self.best_result.num_trades}")
            print(f"Win Rate: {self.best_result.win_rate * 100:.2f}%")

        return self.best_params, self.best_result

    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最优参数 / Get best parameters

        Returns:
            最优参数字典 / Best parameters dictionary
        """
        return self.best_params.copy() if self.best_params else {}

    def backtest(self, params: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """
        使用指定参数进行回测 / Backtest with specified parameters

        Args:
            params: 策略参数，如果为 None 则使用最优参数
                   / Strategy parameters, uses best params if None

        Returns:
            回测结果 / Backtest result
        """
        if params is None:
            params = self.best_params
        return self._backtest(params)


# =============================================================================
# 导出接口 / Exports
# =============================================================================

__all__ = [
    # 环境和基础 / Environment and base
    "TradingGymEnv",
    "TradingState",
    "BacktestResult",
    # 策略 / Strategies
    "BaseStrategy",
    "MACrossoverStrategy",
    "RSIStrategy",
    "BollingerStrategy",
    # 优化器 / Optimizers
    "GeneticOptimizer",
    "StrategyOptimizer",
    # 指标函数 / Indicator functions
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_macd",
    "calculate_atr",
    # 常量 / Constants
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
    "STRATEGY_MA_CROSSOVER",
    "STRATEGY_RSI",
    "STRATEGY_BOLLINGER",
]
