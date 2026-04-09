"""
PyCryptoBot - Quadency's crypto trading bot integration
========================================================

Adapted from Quadency's pycryptobot for use in the quant_trading system.

Supported strategies:
    - EMAStrategy: EMA crossover strategy (EMA12/EMA26)
    - THUMBStrategy: Technical analysis thumb strategy with multiple indicators
    - SMARoCStrategy: SMA Range of Candles strategy

Supported exchanges:
    - Binance (via BinanceAdapter)
    - Coinbase (via Coinbase Pro API)

参考自 Quadency 的 pycryptobot 量化交易机器人，适配到本系统中。

支持的策略:
    - EMAStrategy: EMA交叉策略 (EMA12/EMA26)
    - THUMBStrategy: 技术分析拇指策略 (多指标综合)
    - SMARoCStrategy: 蜡烛范围SMA策略

支持的交易所:
    - Binance (通过 BinanceAdapter)
    - Coinbase (通过 Coinbase Pro API)
"""

import asyncio
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

try:
    import pandas as pd
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from quant_trading.exchanges.binance import BinanceAdapter
from quant_trading.execution.executor import Order, OrderSide, OrderStatus, OrderType

# =============================================================================
# Enums and Constants
# =============================================================================


class Granularity(Enum):
    """时间周期枚举 / Time granularity enum"""
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    ONE_HOUR = 3600
    SIX_HOURS = 21600
    ONE_DAY = 86400


class Action(Enum):
    """交易动作枚举 / Trading action enum"""
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


# Binance API endpoints
BINANCE_API_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# Coinbase API endpoints
COINBASE_API_URL = "https://api.exchange.coinbase.com"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

# Default trading pairs
DEFAULT_MARKET_BINANCE = "BTCUSDT"
DEFAULT_MARKET_COINBASE = "BTC-USD"

# Trading fees (approximate)
DEFAULT_MARKET_FEE = 0.0015  # 0.15%
DEFAULT_TAKER_FEE = 0.0015
DEFAULT_MAKER_FEE = 0.0015


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PyCryptoBotConfig:
    """PyCryptoBot configuration / PyCryptoBot配置"""
    exchange: str = "binance"
    market: str = DEFAULT_MARKET_BINANCE
    granularity: Granularity = Granularity.ONE_HOUR
    api_key: str = ""
    api_secret: str = ""
    test_mode: bool = True

    # Buy/Sell thresholds
    buy_threshold: float = 0.0  # percentage below high to buy
    sell_threshold: float = 0.0  # percentage above low to sell

    # Trailing stop loss
    trailing_stop_loss: float = 0.0
    trailing_stop_loss_trigger: float = 0.0

    # Trading percentages
    buy_percent: float = 100.0
    sell_percent: float = 100.0

    # Trading constraints
    sell_at_loss: bool = True
    prevent_loss: bool = False
    prevent_loss_trigger: float = 1.0
    prevent_loss_margin: float = 0.1

    # Bull market only
    disable_bull_only: bool = False

    # Technical indicators enabled
    disable_buy_ema: bool = False
    disable_buy_macd: bool = False
    disable_buy_obv: bool = True
    disable_buy_elder_ray: bool = True
    disable_buy_bbands_s1: bool = True
    disable_buy_bbands_s2: bool = True

    # Sell constraints
    sell_upper_pcnt: Optional[float] = None
    sell_lower_pcnt: Optional[float] = None
    no_sell_min_pcnt: Optional[float] = None
    no_sell_max_pcnt: Optional[float] = None

    # Trailing buy
    trailing_buy_pcnt: float = 0.0
    trailing_immediate_buy: bool = False
    trailing_buy_immediate_pcnt: Optional[float] = None

    # Telegram
    telegram_token: Optional[str] = None
    telegram_client_id: Optional[str] = None
    disable_telegram: bool = True


@dataclass
class AppState:
    """Application state for tracking trading state / 应用状态"""
    action: Action = Action.WAIT
    last_action: Optional[Action] = None
    buy_high: float = 0.0
    sell_low: float = 0.0
    decimal_increase: float = 0.0
    margin: float = 0.0
    last_buy_price: float = 0.0
    last_buy_high: float = 0.0
    change_pcnt_high: float = 0.0
    tsl_trigger: float = 0.0
    tsl_pcnt: float = 0.0
    tsl_triggered: bool = False
    tsl_max: bool = False
    prevent_loss: bool = False
    trailing_buy: bool = False
    waiting_buy_price: float = 0.0
    trailing_sell: bool = False
    waiting_sell_price: Optional[float] = None
    trailing_buy_immediate: bool = False
    trailing_sell_immediate: bool = False
    fib_low: float = 0.0


@dataclass
class MarketData:
    """Market data candle / 市场数据蜡烛"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# =============================================================================
# Technical Analysis Functions
# =============================================================================


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """
    Calculate Exponential Moving Average
    计算指数移动平均线

    Args:
        prices: List of prices / 价格列表
        period: EMA period / 周期

    Returns:
        List of EMA values / EMA值列表
    """
    if len(prices) < period:
        return []

    ema = [sum(prices[:period]) / period]
    multiplier = 2 / (period + 1)

    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema


def calculate_sma(prices: List[float], period: int) -> List[float]:
    """
    Calculate Simple Moving Average
    计算简单移动平均线

    Args:
        prices: List of prices / 价格列表
        period: SMA period / 周期

    Returns:
        List of SMA values / SMA值列表
    """
    if len(prices) < period:
        return []

    sma = []
    for i in range(period - 1, len(prices)):
        sma.append(sum(prices[i - period + 1:i + 1]) / period)

    return sma


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD indicator
    计算MACD指标

    Args:
        prices: List of prices / 价格列表
        fast_period: Fast EMA period / 快线周期
        slow_period: Slow EMA period / 慢线周期
        signal_period: Signal line period / 信号线周期

    Returns:
        Tuple of (macd, signal, histogram) / (MACD线, 信号线, 柱状图)
    """
    if len(prices) < slow_period:
        return [], [], []

    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    # Align arrays
    offset = slow_period - fast_period
    macd_line = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = []

    # Calculate histogram
    signal_offset = signal_period - 1
    for i in range(len(signal_line)):
        macd_idx = i + signal_offset
        if macd_idx < len(macd_line):
            histogram.append(macd_line[macd_idx] - signal_line[i])

    return macd_line, signal_line, histogram


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index
    计算相对强弱指数

    Args:
        prices: List of prices / 价格列表
        period: RSI period / 周期

    Returns:
        List of RSI values / RSI值列表
    """
    if len(prices) < period + 1:
        return []

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [c if c > 0 else 0 for c in changes]
    losses = [-c if c < 0 else 0 for c in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi = [100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100]

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi.append(100 - (100 / (1 + rs)))

    return rsi


def calculate_bollinger_bands(
    prices: List[float],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands
    计算布林带

    Args:
        prices: List of prices / 价格列表
        period: Moving average period / 移动平均周期
        std_dev: Standard deviation multiplier / 标准差倍数

    Returns:
        Tuple of (upper, middle, lower) bands / (上轨, 中轨, 下轨)
    """
    if len(prices) < period:
        return [], [], []

    sma = calculate_sma(prices, period)
    upper = []
    middle = []
    lower = []

    for i in range(period - 1, len(prices)):
        slice_prices = prices[i - period + 1:i + 1]
        mean = sum(slice_prices) / period
        variance = sum((p - mean) ** 2 for p in slice_prices) / period
        std = math.sqrt(variance)

        middle.append(mean)
        upper.append(mean + std_dev * std)
        lower.append(mean - std_dev * std)

    return upper, middle, lower


def calculate_obv(prices: List[float], volumes: List[float]) -> List[float]:
    """
    Calculate On-Balance Volume
    计算能量潮指标

    Args:
        prices: List of prices / 价格列表
        volumes: List of volumes / 成交量列表

    Returns:
        List of OBV values / OBV值列表
    """
    if len(prices) != len(volumes) or len(prices) < 2:
        return []

    obv = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    return obv


def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> List[float]:
    """
    Calculate Average True Range
    计算平均真实波幅

    Args:
        highs: List of high prices / 最高价列表
        lows: List of low prices / 最低价列表
        closes: List of close prices / 收盘价列表
        period: ATR period / 周期

    Returns:
        List of ATR values / ATR值列表
    """
    if len(highs) < period + 1:
        return []

    tr = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))

    atr = [sum(tr[:period]) / period]
    for i in range(period, len(tr)):
        atr.append((atr[-1] * (period - 1) + tr[i]) / period)

    return atr


# =============================================================================
# Strategy Classes
# =============================================================================


class BaseStrategy:
    """Base class for PyCryptoBot strategies / 策略基类"""

    def __init__(self, config: PyCryptoBotConfig):
        """
        Initialize strategy with configuration
        使用配置初始化策略

        Args:
            config: PyCryptoBot configuration / PyCryptoBot配置
        """
        self.config = config
        self.state = AppState()

    def check_buy(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check if buy signal is present
        检查是否存在买入信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if buy signal / 是否买入信号
        """
        raise NotImplementedError

    def check_sell(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check if sell signal is present
        检查是否存在卖出信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if sell signal / 是否卖出信号
        """
        raise NotImplementedError

    def update_state(self, action: Action, price: float):
        """Update trading state / 更新交易状态"""
        self.state.last_action = self.state.action
        self.state.action = action
        if action == Action.BUY:
            self.state.last_buy_price = price
            self.state.last_buy_high = price
        elif action == Action.SELL:
            self.state.sell_low = price


class EMAStrategy(BaseStrategy):
    """
    EMA Crossover Strategy / EMA交叉策略

    Buy when:
        - EMA12 crosses above EMA26 (golden cross)
        - MACD is positive
        - RSI is not overbought

    Sell when:
        - EMA12 crosses below EMA26 (death cross)
        - MACD is negative
        - Or trailing stop loss triggered

    买入条件:
        - EMA12上穿EMA26 (金叉)
        - MACD为正
        - RSI未超买

    卖出条件:
        - EMA12下穿EMA26 (死叉)
        - MACD为负
        - 或追踪止损触发
    """

    def __init__(self, config: PyCryptoBotConfig):
        super().__init__(config)
        self.ema_fast_period = 12
        self.ema_slow_period = 26
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14

    def check_buy(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for EMA buy signal
        检查EMA买入信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if buy signal / 是否买入信号
        """
        if len(candles) < max(self.ema_slow_period, self.macd_slow, self.rsi_period) + 1:
            return False

        closes = [c.close for c in candles]

        # Calculate EMAs
        ema_fast = calculate_ema(closes, self.ema_fast_period)
        ema_slow = calculate_ema(closes, self.ema_slow_period)

        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return False

        # Check for golden cross (EMA12 crosses above EMA26)
        prev_fast = ema_fast[-2]
        curr_fast = ema_fast[-1]
        prev_slow = ema_slow[-2]
        curr_slow = ema_slow[-1]

        golden_cross = prev_fast <= prev_slow and curr_fast > curr_slow

        # Calculate MACD
        macd_line, signal_line, _ = calculate_macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )

        if len(macd_line) < 2 or len(signal_line) < 2:
            return False

        macd_bullish = macd_line[-1] > signal_line[-1] and macd_line[-1] > 0

        # Calculate RSI
        rsi_values = calculate_rsi(closes, self.rsi_period)
        if len(rsi_values) < 1:
            return False

        rsi_ok = 30 < rsi_values[-1] < 70  # Not overbought/oversold

        # Bull only mode check
        if not self.config.disable_bull_only:
            # Check if SMA50 > SMA200 for bull market
            sma_50 = calculate_sma(closes, 50)
            sma_200 = calculate_sma(closes, 200)
            if len(sma_50) < 1 or len(sma_200) < 1:
                return False
            if sma_50[-1] <= sma_200[-1]:
                return False

        # Not already holding
        if self.state.action == Action.BUY:
            return False

        return golden_cross and macd_bullish and rsi_ok

    def check_sell(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for EMA sell signal
        检查EMA卖出信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if sell signal / 是否卖出信号
        """
        if len(candles) < max(self.ema_slow_period, self.macd_slow, self.rsi_period) + 1:
            return False

        closes = [c.close for c in candles]

        # Calculate EMAs
        ema_fast = calculate_ema(closes, self.ema_fast_period)
        ema_slow = calculate_ema(closes, self.ema_slow_period)

        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return False

        # Check for death cross (EMA12 crosses below EMA26)
        prev_fast = ema_fast[-2]
        curr_fast = ema_fast[-1]
        prev_slow = ema_slow[-2]
        curr_slow = ema_slow[-1]

        death_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        # Calculate MACD
        macd_line, signal_line, _ = calculate_macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )

        if len(macd_line) < 2 or len(signal_line) < 2:
            return False

        macd_bearish = macd_line[-1] < signal_line[-1] and macd_line[-1] < 0

        # Check trailing stop loss
        trailing_triggered = False
        if self.config.trailing_stop_loss > 0 and self.state.action == Action.BUY:
            buy_price = self.state.last_buy_price
            current_price = closes[-1]
            margin = ((current_price - buy_price) / buy_price) * 100

            if margin > self.config.trailing_stop_loss_trigger:
                high_since_buy = max(
                    self.state.last_buy_high,
                    max(closes[-100:]) if len(closes) >= 100 else max(closes)
                )
                drawdown = ((high_since_buy - current_price) / high_since_buy) * 100

                if drawdown >= self.config.trailing_stop_loss:
                    trailing_triggered = True

        # Check sell at loss
        if not self.config.sell_at_loss and self.state.action == Action.BUY:
            buy_price = self.state.last_buy_price
            current_price = closes[-1]
            if current_price < buy_price:
                return False

        return death_cross or macd_bearish or trailing_triggered


class THUMBStrategy(BaseStrategy):
    """
    THUMB Strategy - Technical analysis with multiple indicators
    THUMB策略 - 多指标技术分析

    This strategy uses a point-based system with multiple technical indicators:
        - RSI and its moving average
        - ADX and DI+/DI-
        - MACD and signal line
        - OBV and its SMA
        - EMA/WMA crossover
        - Bollinger Bands

    Each indicator contributes points toward buy or sell signals.

    该策略使用多指标加权评分系统:
        - RSI及其移动平均线
        - ADX和DI+/DI-
        - MACD和信号线
        - OBV及其SMA
        - EMA/WMA交叉
        - 布林带

    每个指标为买入或卖出信号贡献分数。
    """

    def __init__(self, config: PyCryptoBotConfig):
        super().__init__(config)

        # Points configuration
        self.max_pts = 12
        self.sell_override_pts = 10
        self.pts_to_buy = 9
        self.pts_to_sell = 3
        self.immed_buy_pts = 11
        self.immed_sell_pts = 6
        self.sig_required_buy = 3
        self.sig_required_sell = 0

    def _calc_diff(self, first: float, second: float) -> float:
        """Calculate percentage difference between two values / 计算两值的百分比差"""
        if second == 0:
            return 0
        return round((first - second) / abs(second) * 100, 2)

    def _get_indicators(self, candles: List[MarketData]) -> Dict[str, Any]:
        """
        Calculate all technical indicators for THUMB strategy
        计算THUMB策略的所有技术指标

        Args:
            candles: Historical candles / 历史蜡烛数据

        Returns:
            Dictionary of indicators / 指标字典
        """
        if len(candles) < 200:
            return {}

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        result = {}

        # SMAs
        result["sma5"] = calculate_sma(closes, 5)
        result["sma10"] = calculate_sma(closes, 10)
        result["sma50"] = calculate_sma(closes, 50)
        result["sma100"] = calculate_sma(closes, 100)

        # EMAs
        result["ema5"] = calculate_ema(closes, 5)
        result["ema10"] = calculate_ema(closes, 10)
        result["ema12"] = calculate_ema(closes, 12)
        result["ema26"] = calculate_ema(closes, 26)

        # MACD
        macd_line, signal_line, histogram = calculate_macd(closes)
        result["macd"] = macd_line
        result["signal"] = signal_line
        result["macd_hist"] = histogram

        # RSI
        result["rsi14"] = calculate_rsi(closes, 14)

        # Calculate RSI MA
        rsi = result["rsi14"]
        if len(rsi) >= 14:
            result["rsima14"] = calculate_sma(rsi, 14)
        else:
            result["rsima14"] = []

        # OBV
        result["obv"] = calculate_obv(closes, volumes)
        if len(result["obv"]) >= 8:
            result["obvsm"] = calculate_sma(result["obv"], 8)
        else:
            result["obvsm"] = []

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower

        # ATR
        result["atr"] = calculate_atr(highs, lows, closes)

        # Percentage changes
        if len(closes) > 1:
            result["close_pc"] = [(closes[i] - closes[i - 1]) / closes[i - 1] * 100
                                   for i in range(1, len(closes))]
            result["ema5_pc"] = [0] + [(result["ema5"][i] - result["ema5"][i - 1]) /
                                        result["ema5"][i - 1] * 100
                                        for i in range(1, len(result["ema5"]))] if len(result["ema5"]) > 1 else [0]
            result["sma5_pc"] = [0] + [(result["sma5"][i] - result["sma5"][i - 1]) /
                                        result["sma5"][i - 1] * 100
                                        for i in range(1, len(result["sma5"]))] if len(result["sma5"]) > 1 else [0]
            result["sma10_pc"] = [0] + [(result["sma10"][i] - result["sma10"][i - 1]) /
                                         result["sma10"][i - 1] * 100
                                         for i in range(1, len(result["sma10"]))] if len(result["sma10"]) > 1 else [0]
            result["sma50_pc"] = [0] + [(result["sma50"][i] - result["sma50"][i - 1]) /
                                         result["sma50"][i - 1] * 100
                                         for i in range(1, len(result["sma50"]))] if len(result["sma50"]) > 1 else [0]
            result["sma100_pc"] = [0] + [(result["sma100"][i] - result["sma100"][i - 1]) /
                                          result["sma100"][i - 1] * 100
                                          for i in range(1, len(result["sma100"]))] if len(result["sma100"]) > 1 else [0]
        else:
            result["close_pc"] = [0]
            result["ema5_pc"] = [0]
            result["sma5_pc"] = [0]
            result["sma10_pc"] = [0]
            result["sma50_pc"] = [0]
            result["sma100_pc"] = [0]

        return result

    def check_buy(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for THUMB buy signal
        检查THUMB买入信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if buy signal / 是否买入信号
        """
        ind = self._get_indicators(candles)
        if not ind:
            return False

        buy_pts = 0
        pts_sig_required = 0

        # Get latest values
        idx = -1

        # RSI action
        rsi_ma_diff = self._calc_diff(ind["rsi14"][idx], ind["rsima14"][idx]) if len(ind["rsima14"]) > 0 else 0
        rsima_pc = ind["close_pc"][idx] if len(ind["close_pc"]) > idx + 1 else 0
        rsi_pc = ind["close_pc"][idx] if len(ind["close_pc"]) > idx + 1 else 0

        if rsi_ma_diff >= 3 and rsima_pc > 0 and rsi_pc > 0:
            pts_sig_required += 1
            if rsi_ma_diff > 10 or rsi_pc >= 3:
                buy_pts += 2
            else:
                buy_pts += 1

        # MACD action
        macd_sg_diff = self._calc_diff(ind["macd"][idx], ind["signal"][idx]) if len(ind["macd"]) > 0 and len(ind["signal"]) > 0 else 0
        macd_pc = ind["close_pc"][idx] if len(ind["close_pc"]) > idx + 1 else 0

        if len(ind["macd"]) > 0 and len(ind["signal"]) > 0:
            if macd_sg_diff > 15 and macd_pc > 0:
                pts_sig_required += 1
                if macd_sg_diff > 30 or macd_pc > 8:
                    buy_pts += 2
                else:
                    buy_pts += 1

        # OBV action
        if len(ind["obv"]) > 0 and len(ind["obvsm"]) > 0:
            obv_sm_diff = self._calc_diff(ind["obv"][idx], ind["obvsm"][idx])
            if len(ind["close_pc"]) > idx + 1:
                obv_sm_pc = ind["close_pc"][idx]
                if obv_sm_diff > 0.5 and obv_sm_pc > 0:
                    pts_sig_required += 1
                    buy_pts += 1

        # SMA conditions for market risk assessment
        if len(ind["sma5"]) > 0 and len(ind["sma10"]) > 0:
            sma5 = ind["sma5"][idx]
            sma10 = ind["sma10"][idx]
            sma5_pc = ind["sma5_pc"][idx] if len(ind["sma5_pc"]) > 0 else 0
            sma10_pc = ind["sma10_pc"][idx] if len(ind["sma10_pc"]) > 0 else 0

            # High risk condition
            if sma5 < sma10 and sma5_pc < 0 and sma10_pc < 0:
                return False  # High risk, no buying

        # Additional buy points based on trends
        if len(ind["sma5"]) > 0 and len(ind["sma10"]) > 0 and len(ind["sma50"]) > 0:
            if (ind["sma5"][idx] > ind["sma10"][idx] and
                ind["sma5_pc"][idx] > 0.1 and ind["sma10_pc"][idx] > 0.1):
                buy_pts += 1

            if (ind["sma10"][idx] > ind["sma50"][idx] and
                ind["sma50_pc"][idx] > 0):
                buy_pts += 1

        return buy_pts >= self.pts_to_buy and pts_sig_required >= self.sig_required_buy

    def check_sell(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for THUMB sell signal
        检查THUMB卖出信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if sell signal / 是否卖出信号
        """
        ind = self._get_indicators(candles)
        if not ind:
            return False

        sell_pts = 0
        idx = -1

        # RSI sell conditions
        rsi_ma_diff = self._calc_diff(ind["rsi14"][idx], ind["rsima14"][idx]) if len(ind["rsima14"]) > 0 else 0
        rsima_pc = ind["close_pc"][idx] if len(ind["close_pc"]) > idx + 1 else 0

        if len(ind["close_pc"]) > idx + 1 and ind["close_pc"][idx] < 0:
            if rsi_ma_diff < 0 or rsima_pc < 0:
                if rsi_ma_diff < -8 or rsima_pc < -3:
                    sell_pts += 2
                else:
                    sell_pts += 1

        # MACD sell conditions
        macd_pc = ind["close_pc"][idx] if len(ind["close_pc"]) > idx + 1 else 0
        if macd_pc < 0:
            sell_pts += 1
            if len(ind["macd"]) > 0 and len(ind["signal"]) > 0:
                macd_sg_diff = self._calc_diff(ind["macd"][idx], ind["signal"][idx])
                if macd_sg_diff < 0 or macd_pc < -8:
                    sell_pts += 1

        # OBV sell
        if len(ind["obv"]) > 0 and len(ind["obvsm"]) > 0:
            obv_sm_diff = self._calc_diff(ind["obv"][idx], ind["obvsm"][idx])
            if len(ind["close_pc"]) > idx + 1 and ind["close_pc"][idx] < 0:
                if obv_sm_diff < 0:
                    sell_pts += 1

        # High risk condition
        if len(ind["sma5"]) > 0 and len(ind["sma10"]) > 0:
            sma5 = ind["sma5"][idx]
            sma10 = ind["sma10"][idx]
            sma5_pc = ind["sma5_pc"][idx] if len(ind["sma5_pc"]) > 0 else 0
            sma10_pc = ind["sma10_pc"][idx] if len(ind["sma10_pc"]) > 0 else 0

            if sma5 < sma10 and sma5_pc < 0 and sma10_pc < 0:
                sell_pts += 3  # Strong sell signal in high risk

        return sell_pts >= self.pts_to_sell


class SMARoCStrategy(BaseStrategy):
    """
    SMA Range of Candles Strategy / 蜡烛范围SMA策略

    This strategy analyzes the range of candles over a SMA period to determine:
        - If the range is contracting or expanding
        - Trend strength based on candle ranges
        - Entry points based on range breakouts

    该策略分析SMA周期内蜡烛的范围来确定:
        - 范围是收缩还是扩张
        - 基于蜡烛范围的趋势强度
        - 基于范围突破的入场点
    """

    def __init__(self, config: PyCryptoBotConfig):
        super().__init__(config)
        self.sma_period = 20
        self.range_threshold = 0.02  # 2% range threshold

    def _calculate_candle_ranges(self, candles: List[MarketData]) -> List[float]:
        """
        Calculate range (high - low) for each candle
        计算每根蜡烛的范围

        Args:
            candles: Historical candles / 历史蜡烛数据

        Returns:
            List of candle ranges / 蜡烛范围列表
        """
        return [c.high - c.low for c in candles]

    def _calculate_sma_ranges(self, ranges: List[float]) -> List[float]:
        """
        Calculate SMA of candle ranges
        计算蜡烛范围的SMA

        Args:
            ranges: List of candle ranges / 蜡烛范围列表

        Returns:
            SMA of ranges / 范围的SMA
        """
        return calculate_sma(ranges, self.sma_period)

    def check_buy(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for SMARoC buy signal
        检查SMARoC买入信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if buy signal / 是否买入信号
        """
        if len(candles) < self.sma_period + 1:
            return False

        closes = [c.close for c in candles]
        ranges = self._calculate_candle_ranges(candles)
        sma_ranges = self._calculate_sma_ranges(ranges)

        if len(sma_ranges) < 2:
            return False

        idx = -1
        prev_idx = -2

        # Current range above SMA range
        current_range_above_sma = ranges[idx] > sma_ranges[idx]

        # Previous range below or equal to SMA range (contraction)
        prev_range_below_sma = ranges[prev_idx] <= sma_ranges[prev_idx]

        # Price above SMA
        sma = calculate_sma(closes, self.sma_period)
        if len(sma) < 1:
            return False

        price_above_sma = closes[idx] > sma[idx]

        # Range expansion confirmed
        range_expanding = ranges[idx] > ranges[prev_idx]

        # Check for golden cross (EMA5 > EMA10)
        ema5 = calculate_ema(closes, 5)
        ema10 = calculate_ema(closes, 10)

        if len(ema5) < 2 or len(ema10) < 2:
            return False

        golden_cross = ema5[prev_idx] <= ema10[prev_idx] and ema5[idx] > ema10[idx]

        # Not already holding
        if self.state.action == Action.BUY:
            return False

        return (current_range_above_sma and prev_range_below_sma and
                price_above_sma and range_expanding and golden_cross)

    def check_sell(self, candles: List[MarketData], indicators: Dict[str, Any]) -> bool:
        """
        Check for SMARoC sell signal
        检查SMARoC卖出信号

        Args:
            candles: Historical candles / 历史蜡烛数据
            indicators: Technical indicators / 技术指标

        Returns:
            True if sell signal / 是否卖出信号
        """
        if len(candles) < self.sma_period + 1:
            return False

        closes = [c.close for c in candles]
        ranges = self._calculate_candle_ranges(candles)
        sma_ranges = self._calculate_sma_ranges(ranges)

        if len(sma_ranges) < 2:
            return False

        idx = -1
        prev_idx = -2

        # Current range below SMA range (contraction)
        current_range_below_sma = ranges[idx] < sma_ranges[idx]

        # Previous range was above SMA (expansion then contraction)
        prev_range_above_sma = ranges[prev_idx] >= sma_ranges[prev_idx]

        # Price below SMA
        sma = calculate_sma(closes, self.sma_period)
        if len(sma) < 1:
            return False

        price_below_sma = closes[idx] < sma[idx]

        # Check for death cross (EMA5 < EMA10)
        ema5 = calculate_ema(closes, 5)
        ema10 = calculate_ema(closes, 10)

        if len(ema5) < 2 or len(ema10) < 2:
            return False

        death_cross = ema5[prev_idx] >= ema10[prev_idx] and ema5[idx] < ema10[idx]

        # Trailing stop loss check
        trailing_triggered = False
        if self.config.trailing_stop_loss > 0 and self.state.action == Action.BUY:
            buy_price = self.state.last_buy_price
            current_price = closes[idx]
            margin = ((current_price - buy_price) / buy_price) * 100

            if margin > self.config.trailing_stop_loss_trigger:
                high_since_buy = max(
                    self.state.last_buy_high,
                    max(closes[-100:]) if len(closes) >= 100 else max(closes)
                )
                drawdown = ((high_since_buy - current_price) / high_since_buy) * 100

                if drawdown >= self.config.trailing_stop_loss:
                    trailing_triggered = True

        # Sell at loss check
        if not self.config.sell_at_loss and self.state.action == Action.BUY:
            buy_price = self.state.last_buy_price
            if closes[idx] < buy_price:
                return False

        return ((current_range_below_sma and prev_range_above_sma and
                price_below_sma and death_cross) or trailing_triggered)


# =============================================================================
# Exchange Connectors
# =============================================================================


class QuadencyConnector:
    """
    Quadency connector for Binance and Coinbase
    Quadency连接器用于Binance和Coinbase

    This class provides a unified interface for both Binance and Coinbase APIs,
    adapting to the pycryptobot trading style.

    该类为Binance和Coinbase API提供统一接口,适配pycryptobot交易风格。
    """

    def __init__(self, config: PyCryptoBotConfig):
        """
        Initialize Quadency connector
        初始化Quadency连接器

        Args:
            config: PyCryptoBot configuration / PyCryptoBot配置
        """
        self.config = config
        self.exchange = config.exchange.lower()
        self.market = config.market
        self.granularity = config.granularity
        self.binance_adapter: Optional[BinanceAdapter] = None

        # Session for REST API
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    async def connect(self) -> None:
        """Connect to exchange / 连接到交易所"""
        if self.exchange == "binance":
            await self._connect_binance()
        elif self.exchange == "coinbase":
            self._connect_coinbase()
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")

    async def _connect_binance(self) -> None:
        """Connect to Binance / 连接到Binance"""
        self.binance_adapter = BinanceAdapter({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "testnet": self.config.test_mode
        })
        await self.binance_adapter.connect()

    def _connect_coinbase(self) -> None:
        """Connect to Coinbase / 连接到Coinbase"""
        # Coinbase uses HMAC authentication
        self.api_url = COINBASE_API_URL

    async def close(self) -> None:
        """Close connection / 关闭连接"""
        if self.binance_adapter:
            await self.binance_adapter.close()

    def _binance_timestamp(self) -> int:
        """Get current timestamp for Binance / 获取Binance当前时间戳"""
        return int(time.time() * 1000)

    def _coinbase_timestamp(self) -> str:
        """Get current timestamp for Coinbase / 获取Coinbase当前时间戳"""
        return datetime.utcnow().isoformat()

    def _sign_request(self, params: Dict[str, Any], secret: str) -> str:
        """
        Sign request for authentication
        签名请求用于认证

        Args:
            params: Request parameters / 请求参数
            secret: API secret / API密钥

        Returns:
            Signature / 签名
        """
        if self.exchange == "binance":
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(
                secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            return signature
        elif self.exchange == "coinbase":
            timestamp = str(int(time.time()))
            message = timestamp + "GET" + "/users/self/verify"
            signature = hmac.new(
                secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            return signature
        return ""

    async def fetch_candles(
        self,
        market: Optional[str] = None,
        granularity: Optional[Granularity] = None,
        limit: int = 300
    ) -> List[MarketData]:
        """
        Fetch historical candles / 获取历史蜡烛数据

        Args:
            market: Trading pair / 交易对
            granularity: Time granularity / 时间周期
            limit: Number of candles / 蜡烛数量

        Returns:
            List of market candles / 市场蜡烛列表
        """
        market = market or self.market
        granularity = granularity or self.granularity

        if self.exchange == "binance":
            return await self._fetch_binance_candles(market, granularity, limit)
        elif self.exchange == "coinbase":
            return self._fetch_coinbase_candles(market, granularity, limit)
        return []

    async def _fetch_binance_candles(
        self,
        market: str,
        granularity: Granularity,
        limit: int
    ) -> List[MarketData]:
        """Fetch candles from Binance / 从Binance获取蜡烛数据"""
        if not self.binance_adapter:
            await self._connect_binance()

        # Convert granularity to Binance interval
        interval_map = {
            Granularity.ONE_MINUTE: "1m",
            Granularity.FIVE_MINUTES: "5m",
            Granularity.FIFTEEN_MINUTES: "15m",
            Granularity.ONE_HOUR: "1h",
            Granularity.SIX_HOURS: "6h",
            Granularity.ONE_DAY: "1d",
        }
        interval = interval_map.get(granularity, "1h")

        ohlcv = await self.binance_adapter.fetch_ohlcv(market, interval, limit)

        candles = []
        for data in ohlcv:
            candles.append(MarketData(
                timestamp=datetime.fromtimestamp(data[0] / 1000),
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5])
            ))

        return candles

    def _fetch_coinbase_candles(
        self,
        market: str,
        granularity: Granularity,
        limit: int
    ) -> List[MarketData]:
        """Fetch candles from Coinbase / 从Coinbase获取蜡烛数据"""
        granularity_seconds = granularity.value
        end_time = self._coinbase_timestamp()

        url = f"{self.api_url}/products/{market}/candles"
        params = {
            "granularity": granularity_seconds,
            "end": end_time
        }

        response = self.session.get(url, params=params)
        if response.status_code != 200:
            return []

        data = response.json()

        candles = []
        for item in reversed(data[:limit]):
            candles.append(MarketData(
                timestamp=datetime.fromtimestamp(item[0]),
                low=float(item[1]),
                high=float(item[2]),
                open=float(item[3]),
                close=float(item[4]),
                volume=float(item[5])
            ))

        return candles

    async def fetch_ticker(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch current ticker / 获取当前行情

        Args:
            market: Trading pair / 交易对

        Returns:
            Ticker data / 行情数据
        """
        market = market or self.market

        if self.exchange == "binance":
            if not self.binance_adapter:
                await self._connect_binance()
            ticker = await self.binance_adapter.fetch_ticker(market)
            return {
                "price": ticker.get("last", 0),
                "bid": ticker.get("bid", 0),
                "ask": ticker.get("ask", 0),
                "volume": ticker.get("quoteVolume", 0),
                "timestamp": datetime.now()
            }
        elif self.exchange == "coinbase":
            url = f"{self.api_url}/products/{market}/ticker"
            response = self.session.get(url)
            if response.status_code != 200:
                return None
            data = response.json()
            return {
                "price": float(data.get("price", 0)),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "volume": float(data.get("volume", 0)),
                "timestamp": datetime.now()
            }
        return None

    async def fetch_balance(self) -> Dict[str, float]:
        """
        Fetch account balance / 获取账户余额

        Returns:
            Dictionary of currency balances / 货币余额字典
        """
        if self.exchange == "binance":
            if not self.binance_adapter:
                await self._connect_binance()
            balance = await self.binance_adapter.fetch_balance()
            return {k: float(v.get("free", 0)) for k, v in balance.items()}
        elif self.exchange == "coinbase":
            # Coinbase requires authenticated request
            timestamp = self._coinbase_timestamp()
            path = "/accounts"
            message = timestamp + "GET" + path
            signature = self._sign_request({}, self.config.api_secret)

            headers = {
                "CB-ACCESS-KEY": self.config.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": self.config.api_secret
            }

            response = self.session.get(f"{self.api_url}{path}", headers=headers)
            if response.status_code != 200:
                return {}

            data = response.json()
            return {acc.get("currency", ""): float(acc.get("available", 0))
                    for acc in data}

        return {}

    async def place_order(
        self,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        market: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place order / 下单

        Args:
            side: BUY or SELL / 买入或卖出
            quantity: Order quantity / 订单数量
            price: Limit price / 限价
            market: Trading pair / 交易对

        Returns:
            Order result / 订单结果
        """
        market = market or self.market

        if self.config.test_mode:
            return await self._simulate_order(side, quantity, price, market)

        if self.exchange == "binance":
            if not self.binance_adapter:
                await self._connect_binance()
            order_type = "limit" if price else "market"
            result = await self.binance_adapter.create_order(
                market, order_type, side.lower(), quantity, price
            )
            return {
                "id": str(result.get("orderId", "")),
                "side": side,
                "price": price,
                "quantity": quantity,
                "status": result.get("status", "NEW")
            }
        elif self.exchange == "coinbase":
            url = f"{self.api_url}/orders"
            data = {
                "product_id": market,
                "side": side.lower(),
                "type": "market" if price is None else "limit",
                "size": str(quantity)
            }
            if price:
                data["price"] = str(price)

            timestamp = self._coinbase_timestamp()
            message = timestamp + "POST" + "/orders"
            signature = self._sign_request(data, self.config.api_secret)

            headers = {
                "CB-ACCESS-KEY": self.config.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": self.config.api_secret
            }

            response = self.session.post(url, json=data, headers=headers)
            if response.status_code != 200:
                return None

            result = response.json()
            return {
                "id": result.get("id", ""),
                "side": result.get("side", ""),
                "price": float(result.get("price", 0)),
                "quantity": float(result.get("size", 0)),
                "status": result.get("status", "")
            }

        return None

    async def _simulate_order(
        self,
        side: str,
        quantity: float,
        price: Optional[float],
        market: str
    ) -> Dict[str, Any]:
        """Simulate order for backtesting / 模拟订单用于回测"""
        return {
            "id": f"sim_{int(time.time() * 1000)}",
            "side": side,
            "price": price or 0,
            "quantity": quantity,
            "status": "FILLED",
            "filled_quantity": quantity,
            "timestamp": datetime.now()
        }

    async def cancel_order(self, order_id: str, market: Optional[str] = None) -> bool:
        """
        Cancel order / 取消订单

        Args:
            order_id: Order ID / 订单ID
            market: Trading pair / 交易对

        Returns:
            Success / 是否成功
        """
        market = market or self.market

        if self.config.test_mode:
            return True

        if self.exchange == "binance":
            if not self.binance_adapter:
                await self._connect_binance()
            await self.binance_adapter.cancel_order(order_id, market)
            return True
        elif self.exchange == "coinbase":
            timestamp = self._coinbase_timestamp()
            path = f"/orders/{order_id}"
            message = timestamp + "DELETE" + path
            signature = self._sign_request({}, self.config.api_secret)

            headers = {
                "CB-ACCESS-KEY": self.config.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": self.config.api_secret
            }

            response = self.session.delete(f"{self.api_url}{path}", headers=headers)
            return response.status_code == 200

        return False


# =============================================================================
# PyCryptoBot Executor
# =============================================================================


class PyCryptoBotExecutor:
    """
    PyCryptoBot execution engine
    PyCryptoBot执行引擎

    This class orchestrates the trading strategy and exchange connector
    to execute trades based on technical analysis signals.

    该类协调交易策略和交易所连接器,基于技术分析信号执行交易。

    Example / 示例:
        >>> config = PyCryptoBotConfig(
        ...     exchange="binance",
        ...     market="BTCUSDT",
        ...     granularity=Granularity.ONE_HOUR,
        ...     test_mode=True
        ... )
        >>> executor = PyCryptoBotExecutor(config, EMAStrategy(config))
        >>> await executor.start()
    """

    def __init__(
        self,
        config: PyCryptoBotConfig,
        strategy: Optional[BaseStrategy] = None,
        connector: Optional[QuadencyConnector] = None
    ):
        """
        Initialize PyCryptoBot executor
        初始化PyCryptoBot执行器

        Args:
            config: PyCryptoBot configuration / PyCryptoBot配置
            strategy: Trading strategy / 交易策略
            connector: Exchange connector / 交易所连接器
        """
        self.config = config
        self.strategy = strategy or EMAStrategy(config)
        self.connector = connector or QuadencyConnector(config)
        self.state = AppState()
        self.running = False
        self.candles: List[MarketData] = []

        # Telegram
        self.telegram_enabled = not config.disable_telegram and bool(config.telegram_token)

    async def start(self) -> None:
        """Start the trading bot / 启动交易机器人"""
        await self.connector.connect()
        self.running = True

        try:
            while self.running:
                await self._trading_loop()
                await asyncio.sleep(self._get_sleep_interval())
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot / 停止交易机器人"""
        self.running = False
        await self.connector.close()

    async def _trading_loop(self) -> None:
        """Main trading loop / 主交易循环"""
        # Fetch latest candles
        self.candles = await self.connector.fetch_candles(limit=300)

        if len(self.candles) < 50:
            return

        # Get current ticker
        ticker = await self.connector.fetch_ticker()
        if not ticker:
            return

        current_price = ticker["price"]
        action = self._determine_action(current_price)

        # Execute action
        if action == Action.BUY:
            await self._execute_buy(current_price)
        elif action == Action.SELL:
            await self._execute_sell(current_price)

        # Update strategy state
        self.strategy.update_state(action, current_price)

    def _determine_action(self, current_price: float) -> Action:
        """
        Determine trading action based on strategy signals
        根据策略信号确定交易动作

        Args:
            current_price: Current price / 当前价格

        Returns:
            Action to take / 要执行的动作
        """
        indicators = self._calculate_indicators()

        buy_signal = self.strategy.check_buy(self.candles, indicators)
        sell_signal = self.strategy.check_sell(self.candles, indicators)

        if buy_signal and self.state.action != Action.BUY:
            return Action.BUY
        elif sell_signal and self.state.action == Action.BUY:
            return Action.SELL

        return Action.WAIT

    def _calculate_indicators(self) -> Dict[str, Any]:
        """
        Calculate technical indicators
        计算技术指标

        Returns:
            Dictionary of indicators / 指标字典
        """
        indicators = {}

        if len(self.candles) < 50:
            return indicators

        closes = [c.close for c in self.candles]
        highs = [c.high for c in self.candles]
        lows = [c.low for c in self.candles]
        volumes = [c.volume for c in self.candles]

        indicators["ema12"] = calculate_ema(closes, 12)
        indicators["ema26"] = calculate_ema(closes, 26)
        indicators["sma50"] = calculate_sma(closes, 50)
        indicators["sma200"] = calculate_sma(closes, 200)
        indicators["macd"], indicators["signal"], indicators["histogram"] = calculate_macd(closes)
        indicators["rsi"] = calculate_rsi(closes)
        indicators["bb_upper"], indicators["bb_middle"], indicators["bb_lower"] = calculate_bollinger_bands(closes)
        indicators["obv"] = calculate_obv(closes, volumes)

        return indicators

    async def _execute_buy(self, price: float) -> Optional[Dict[str, Any]]:
        """
        Execute buy order / 执行买入订单

        Args:
            price: Buy price / 买入价格

        Returns:
            Order result / 订单结果
        """
        if self.state.action == Action.BUY:
            return None

        # Get balance to determine buy amount
        balance = await self.connector.fetch_balance()

        # Determine quote currency
        if self.config.exchange == "binance":
            quote = self.config.market.replace("USDT", "").replace("BTC", "").replace("BNB", "")
            if quote in ["", self.config.market]:
                quote = "USDT"
        else:
            quote = self.config.market.split("-")[-1]

        available = balance.get(quote, 0)
        if available <= 0:
            return None

        # Calculate quantity
        quantity = (available * self.config.buy_percent / 100) / price

        # Place order
        result = await self.connector.place_order("BUY", quantity, None, self.config.market)

        if result:
            self.state.last_action = self.state.action
            self.state.action = Action.BUY
            self.state.last_buy_price = price
            self.state.last_buy_high = price
            self._send_telegram_message(f"BUY: Bought at {price}")

        return result

    async def _execute_sell(self, price: float) -> Optional[Dict[str, Any]]:
        """
        Execute sell order / 执行卖出订单

        Args:
            price: Sell price / 卖出价格

        Returns:
            Order result / 订单结果
        """
        if self.state.action != Action.BUY:
            return None

        # Calculate margin
        margin = ((price - self.state.last_buy_price) / self.state.last_buy_price) * 100

        # Check if selling at loss is allowed
        if not self.config.sell_at_loss and margin < 0:
            return None

        # Get balance to determine sell amount
        balance = await self.connector.fetch_balance()

        # Determine base currency
        if self.config.exchange == "binance":
            base = self.config.market.replace("USDT", "").replace("BUSD", "").replace("BTC", "")[:3]
            if base in ["", self.config.market]:
                base = self.config.market[:3]
        else:
            base = self.config.market.split("-")[0]

        available = balance.get(base, 0)
        if available <= 0:
            return None

        # Calculate quantity
        quantity = (available * self.config.sell_percent / 100)

        # Place order
        result = await self.connector.place_order("SELL", quantity, None, self.config.market)

        if result:
            self.state.last_action = self.state.action
            self.state.action = Action.SELL
            self.state.sell_low = price
            self._send_telegram_message(f"SELL: Sold at {price}, Margin: {margin:.2f}%")

        return result

    def _get_sleep_interval(self) -> float:
        """
        Get sleep interval between trading loop iterations
        获取交易循环迭代之间的睡眠间隔

        Returns:
            Sleep interval in seconds / 睡眠间隔(秒)
        """
        granularity_seconds = self.config.granularity.value
        return min(granularity_seconds / 2, 60)  # Max 60 seconds

    def _send_telegram_message(self, message: str) -> None:
        """
        Send Telegram message / 发送Telegram消息

        Args:
            message: Message to send / 要发送的消息
        """
        if not self.telegram_enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_client_id,
                "text": f"[PyCryptoBot] {message}"
            }
            self.connector.session.post(url, json=payload)
        except Exception:
            pass  # Silently fail for Telegram

    def get_state(self) -> Dict[str, Any]:
        """
        Get current bot state
        获取当前机器人状态

        Returns:
            State dictionary / 状态字典
        """
        return {
            "action": self.state.action.value if self.state.action else None,
            "last_action": self.state.last_action.value if self.state.last_action else None,
            "last_buy_price": self.state.last_buy_price,
            "last_buy_high": self.state.last_buy_high,
            "margin": self.state.margin,
            "candles_loaded": len(self.candles),
            "running": self.running
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration and State
    "PyCryptoBotConfig",
    "AppState",
    "MarketData",
    "Granularity",
    "Action",

    # Strategies
    "BaseStrategy",
    "EMAStrategy",
    "THUMBStrategy",
    "SMARoCStrategy",

    # Connector
    "QuadencyConnector",

    # Executor
    "PyCryptoBotExecutor",

    # Technical Analysis Functions
    "calculate_ema",
    "calculate_sma",
    "calculate_macd",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_obv",
    "calculate_atr",
]
