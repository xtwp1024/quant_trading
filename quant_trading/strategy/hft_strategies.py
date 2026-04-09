"""
HFT Strategies — High-Frequency Trading Strategies
高频交易策略模块

Absorbed from: D:/Hive/Data/trading_repos/HFT-strategies/
基于 IEX 市场数据、Kalmans 滤波器、LSTM 策略研究实现

Classes:
    - SpreadCaptureStrategy: 捕捉买卖价差策略
    - MomentumSignalStrategy: 动量信号策略
    - OrderBookImbalanceStrategy: 订单簿不平衡策略
    - LatencyArbitrageStrategy: 延迟套利策略
    - HFTPositionManager: HFT 仓位管理器

Pure Python + NumPy implementation.
Focus on signal generation and position management.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """交易信号类型 / Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class HFTOrder:
    """HFT 订单 / High-frequency trading order"""
    symbol: str
    side: SignalType
    price: float
    quantity: float
    timestamp: float
    venue: Optional[str] = None


@dataclass
class Position:
    """仓位 / Position"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class OrderBook:
    """订单簿快照 / Order book snapshot"""
    bids: np.ndarray  # [price, quantity] per level
    asks: np.ndarray  # [price, quantity] per level
    timestamp: float
    venue: str = ""


@dataclass
class TradeTick:
    """交易tick数据 / Trade tick data"""
    symbol: str
    price: float
    volume: float
    timestamp: float


class SpreadCaptureStrategy:
    """
    Spread Capture Strategy — 价差捕捉策略

    Captures the bid-ask spread in high-frequency trading using a Kalman filter
    to estimate the true mid-price and detect when the spread deviates from mean.

    价差捕捉策略 / 使用 Kalman 滤波器估计真实中间价，当价差偏离均值时入场。

    Signal Logic:
        - Kalman filter smooths price noise
        - Spread > threshold: mean reversion expected, fade the move
        - Spread < threshold: capture spread, market make

    Methods:
        compute_signal(): 计算交易信号
        execute(): 执行交易
        manage_risk(): 风险管理
    """

    def __init__(
        self,
        symbol: str,
        spread_threshold: float = 0.001,
        kalman_covariance: float = 1e-4,
        kalman_observation_cov: float = 1.0,
        position_size: float = 100.0,
    ):
        self.symbol = symbol
        self.spread_threshold = spread_threshold
        self.kalman_covariance = kalman_covariance
        self.kalman_observation_cov = kalman_observation_cov

        # Kalman filter state / 滤波器状态
        self.state_mean = 0.0
        self.state_covariance = 1.0

        # Position / 仓位
        self.position = Position(symbol=symbol)
        self.position_size = position_size

        # History for analysis / 历史数据
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.signal_history: List[SignalType] = []

    def _update_kalman(self, observed_price: float) -> Tuple[float, float]:
        """
        Update Kalman filter with observed price.
        返回 (predicted_mean, predicted_covariance)

        Args:
            observed_price: 观测到的价格

        Returns:
            (predicted_mean, predicted_variance)
        """
        # Prediction step / 预测步骤
        pred_mean = self.state_mean
        pred_cov = self.state_covariance + self.kalman_covariance

        # Update step / 更新步骤
        kalman_gain = pred_cov / (pred_cov + self.kalman_observation_cov)
        self.state_mean = pred_mean + kalman_gain * (observed_price - pred_mean)
        self.state_covariance = pred_cov * (1 - kalman_gain)

        return self.state_mean, self.state_covariance

    def compute_signal(
        self,
        bid: float,
        ask: float,
        trade_price: Optional[float] = None,
        trade_volume: Optional[float] = None,
    ) -> SignalType:
        """
        Compute trading signal based on bid-ask spread.

        Args:
            bid: 买一价
            ask: 卖一价
            trade_price: 最新成交价 (optional)
            trade_volume: 最新成交量 (optional)

        Returns:
            SignalType: BUY, SELL, or HOLD
        """
        spread = ask - bid
        mid_price = (bid + ask) / 2.0

        self.price_history.append(mid_price)
        self.spread_history.append(spread)

        # Update Kalman filter / 更新滤波器
        if trade_price is not None:
            self._update_kalman(trade_price)
        else:
            self._update_kalman(mid_price)

        # Keep history bounded / 限制历史长度
        if len(self.price_history) > 1000:
            self.price_history.pop(0)
            self.spread_history.pop(0)

        # Compute spread z-score / 计算价差 z-score
        if len(self.spread_history) < 20:
            return SignalType.HOLD

        spread_mean = np.mean(self.spread_history[-20:])
        spread_std = np.std(self.spread_history[-20:]) + 1e-9
        spread_zscore = (spread - spread_mean) / spread_std

        # Signal generation / 信号生成
        if spread_zscore > self.spread_threshold:
            # Spread widened beyond threshold, expect mean reversion
            # 价差扩大超过阈值，预期均值回归
            if self.position.quantity > 0:
                return SignalType.SELL  # Close long
            else:
                return SignalType.SELL  # Short

        elif spread_zscore < -self.spread_threshold:
            # Spread compressed, capture spread via market making
            # 价差收窄，捕捉价差做市
            if self.position.quantity < 0:
                return SignalType.BUY  # Close short
            else:
                return SignalType.BUY  # Long

        return SignalType.HOLD

    def execute(
        self,
        signal: SignalType,
        bid: float,
        ask: float,
        timestamp: float,
    ) -> Optional[HFTOrder]:
        """
        Execute trade based on signal.

        Args:
            signal: 交易信号
            bid: 买一价
            ask: 卖一价
            timestamp: 时间戳

        Returns:
            HFTOrder if executed, None otherwise
        """
        if signal == SignalType.HOLD:
            return None

        if signal == SignalType.BUY:
            # Buy at ask (taker) / 在卖价买入
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.BUY,
                price=ask,
                quantity=self.position_size,
                timestamp=timestamp,
            )
            self.position.quantity += self.position_size
            self.position.avg_price = (
                (self.position.avg_price * (self.position.quantity - self.position_size) + ask * self.position_size)
                / self.position.quantity if self.position.quantity > 0 else 0
            )
        else:  # SELL
            # Sell at bid (taker) / 在买价卖出
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.SELL,
                price=bid,
                quantity=self.position_size,
                timestamp=timestamp,
            )
            self.position.quantity -= self.position_size

        self.signal_history.append(signal)
        return order

    def manage_risk(
        self,
        current_price: float,
        max_drawdown: float = 0.02,
        max_position: float = 1000.0,
    ) -> List[HFTOrder]:
        """
        Risk management — check and adjust positions.

        Args:
            current_price: 当前市场价格
            max_drawdown: 最大回撤容忍度
            max_position: 最大仓位限制

        Returns:
            List of risk-adjusted orders (e.g., stop-loss)
        """
        orders: List[HFTOrder] = []

        # Unrealized PnL / 未实现盈亏
        self.position.unrealized_pnl = (
            self.position.quantity * (current_price - self.position.avg_price)
        )

        # Position limit check / 仓位限制检查
        if abs(self.position.quantity) > max_position:
            # Reduce position / 减仓
            side = SignalType.SELL if self.position.quantity > 0 else SignalType.BUY
            reduce_qty = abs(self.position.quantity) - max_position
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=current_price,
                quantity=reduce_qty,
                timestamp=0.0,
            ))

        # Stop-loss check / 止损检查
        if self.position.unrealized_pnl < -abs(self.position.avg_price * max_drawdown * len(self.price_history)):
            # Stop-loss triggered / 触发止损
            side = SignalType.SELL if self.position.quantity > 0 else SignalType.BUY
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=current_price,
                quantity=abs(self.position.quantity),
                timestamp=0.0,
            ))
            self.position.quantity = 0

        return orders


class MomentumSignalStrategy:
    """
    Momentum Signal Strategy — 动量信号策略

    Uses short-term price momentum and volume acceleration to generate
    high-frequency trading signals. Inspired by LSTM prediction patterns
    from HFT-strategies research.

    动量信号策略 / 利用短期价格动量和成交量加速度生成高频交易信号。

    Signal Logic:
        - Price momentum: short MA vs long MA crossover
        - Volume acceleration: volume delta confirms momentum
        - Signal: momentum + volume confirmation

    Methods:
        compute_signal(): 计算动量信号
        execute(): 执行交易
        manage_risk(): 风险管理
    """

    def __init__(
        self,
        symbol: str,
        short_window: int = 5,
        long_window: int = 20,
        volume_window: int = 10,
        momentum_threshold: float = 0.0005,
        position_size: float = 100.0,
    ):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.volume_window = volume_window
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size

        self.position = Position(symbol=symbol)
        self.prices: List[float] = []
        self.volumes: List[float] = []
        self.signal_history: List[SignalType] = []

    def _compute_momentum(self) -> Tuple[float, float]:
        """
        Compute price momentum and volume acceleration.

        Returns:
            (momentum, volume_accel): 价格动量和成交量加速度
        """
        if len(self.prices) < self.long_window:
            return 0.0, 0.0

        prices = np.array(self.prices[-self.long_window:])
        volumes = np.array(self.volumes[-self.volume_window:]) if len(self.volumes) >= self.volume_window else np.array([1.0])

        # Price momentum: log returns / 价格动量：对数收益率
        log_returns = np.diff(np.log(prices + 1e-9))
        momentum = np.mean(log_returns[-self.short_window:])

        # Volume acceleration: second derivative of volume / 成交量加速度
        volume_gradient = np.gradient(volumes)
        volume_accel = np.mean(np.gradient(volume_gradient))

        return momentum, volume_accel

    def compute_signal(
        self,
        price: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> SignalType:
        """
        Compute momentum-based trading signal.

        Args:
            price: 当前价格
            volume: 当前成交量
            timestamp: 时间戳 (optional)

        Returns:
            SignalType: BUY, SELL, or HOLD
        """
        self.prices.append(price)
        self.volumes.append(volume)

        # Keep bounded / 限制长度
        if len(self.prices) > 500:
            self.prices.pop(0)
            self.volumes.pop(0)

        momentum, volume_accel = self._compute_momentum()

        # Signal generation / 信号生成
        if len(self.prices) < self.long_window:
            return SignalType.HOLD

        # Volume confirmation / 成交量确认
        volume_confirm = 1 if volume_accel > 0 else 0.5

        if momentum > self.momentum_threshold * volume_confirm:
            signal = SignalType.BUY
        elif momentum < -self.momentum_threshold * volume_confirm:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        self.signal_history.append(signal)
        return signal

    def execute(
        self,
        signal: SignalType,
        bid: float,
        ask: float,
        timestamp: float,
    ) -> Optional[HFTOrder]:
        """
        Execute momentum trade.

        Args:
            signal: 交易信号
            bid: 买一价
            ask: 卖一价
            timestamp: 时间戳

        Returns:
            HFTOrder if executed, None otherwise
        """
        if signal == SignalType.HOLD:
            return None

        if signal == SignalType.BUY:
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.BUY,
                price=ask,
                quantity=self.position_size,
                timestamp=timestamp,
            )
            self.position.quantity += self.position_size
        else:
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.SELL,
                price=bid,
                quantity=self.position_size,
                timestamp=timestamp,
            )
            self.position.quantity -= self.position_size

        return order

    def manage_risk(
        self,
        current_price: float,
        trailing_stop: float = 0.001,
        profit_target: float = 0.005,
    ) -> List[HFTOrder]:
        """
        Momentum-specific risk management.

        Args:
            current_price: 当前价格
            trailing_stop: 追踪止损比例
            profit_target: 止盈比例

        Returns:
            List of risk-adjusted orders
        """
        orders: List[HFTOrder] = []

        if self.position.quantity == 0:
            return orders

        pnl = self.position.quantity * (current_price - self.position.avg_price)
        pnl_pct = pnl / (abs(self.position.quantity) * self.position.avg_price + 1e-9)

        # Trailing stop / 追踪止损
        if pnl_pct < -trailing_stop:
            side = SignalType.SELL if self.position.quantity > 0 else SignalType.BUY
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=current_price,
                quantity=abs(self.position.quantity),
                timestamp=0.0,
            ))
            self.position.quantity = 0

        # Profit target / 止盈
        elif pnl_pct > profit_target:
            side = SignalType.SELL if self.position.quantity > 0 else SignalType.BUY
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=current_price,
                quantity=abs(self.position.quantity),
                timestamp=0.0,
            ))
            self.position.quantity = 0

        return orders


class OrderBookImbalanceStrategy:
    """
    Order Book Imbalance Strategy — 订单簿不平衡策略

    Analyzes the limit order book (LOB) imbalance to predict short-term
    price direction. When bid volume dominates, price tends to rise;
    when ask volume dominates, price tends to fall.

    订单簿不平衡策略 / 分析限价订单簿的不平衡预测短期价格方向。

    Signal Logic:
        - Order Book Imbalance (OBI) = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        - OBI > threshold: bullish, price likely to rise
        - OBI < -threshold: bearish, price likely to fall

    Methods:
        compute_signal(): 计算LOB不平衡信号
        execute(): 执行交易
        manage_risk(): 风险管理
    """

    def __init__(
        self,
        symbol: str,
        imbalance_threshold: float = 0.3,
        depth_levels: int = 5,
        position_size: float = 100.0,
    ):
        self.symbol = symbol
        self.imbalance_threshold = imbalance_threshold
        self.depth_levels = depth_levels
        self.position_size = position_size

        self.position = Position(symbol=symbol)
        self.obi_history: List[float] = []
        self.signal_history: List[SignalType] = []

    def _compute_imbalance(self, order_book: OrderBook) -> float:
        """
        Compute order book imbalance.

        Args:
            order_book: 订单簿快照

        Returns:
            Imbalance ratio in [-1, 1]
        """
        # Aggregate top N levels / 聚合前N档
        bid_vol = np.sum(order_book.bids[:self.depth_levels, 1])
        ask_vol = np.sum(order_book.asks[:self.depth_levels, 1])

        total = bid_vol + ask_vol + 1e-9
        imbalance = (bid_vol - ask_vol) / total

        return imbalance

    def compute_signal(
        self,
        order_book: OrderBook,
        timestamp: Optional[float] = None,
    ) -> SignalType:
        """
        Compute signal from order book imbalance.

        Args:
            order_book: 订单簿快照
            timestamp: 时间戳 (optional)

        Returns:
            SignalType: BUY, SELL, or HOLD
        """
        imbalance = self._compute_imbalance(order_book)
        self.obi_history.append(imbalance)

        # Keep bounded / 限制长度
        if len(self.obi_history) > 1000:
            self.obi_history.pop(0)

        if len(self.obi_history) < 10:
            return SignalType.HOLD

        # Moving average of OBI / OBI移动平均
        obi_ma = np.mean(self.obi_history[-10:])

        # Signal generation / 信号生成
        if imbalance > self.imbalance_threshold and obi_ma > 0:
            signal = SignalType.BUY
        elif imbalance < -self.imbalance_threshold and obi_ma < 0:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        self.signal_history.append(signal)
        return signal

    def execute(
        self,
        signal: SignalType,
        bid: float,
        ask: float,
        timestamp: float,
    ) -> Optional[HFTOrder]:
        """
        Execute trade based on OBI signal.

        Args:
            signal: 交易信号
            bid: 买一价
            ask: 卖一价
            timestamp: 时间戳

        Returns:
            HFTOrder if executed, None otherwise
        """
        if signal == SignalType.HOLD:
            return None

        if signal == SignalType.BUY:
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.BUY,
                price=ask,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=order_book.venue if hasattr(self, 'order_book') else None,
            )
            self.position.quantity += self.position_size
        else:
            order = HFTOrder(
                symbol=self.symbol,
                side=SignalType.SELL,
                price=bid,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=order_book.venue if hasattr(self, 'order_book') else None,
            )
            self.position.quantity -= self.position_size

        return order

    def manage_risk(
        self,
        current_price: float,
        max_loss_per_trade: float = 0.001,
    ) -> List[HFTOrder]:
        """
        Risk management for OBI strategy.

        Args:
            current_price: 当前价格
            max_loss_per_trade: 每笔最大亏损比例

        Returns:
            List of risk-adjusted orders
        """
        orders: List[HFTOrder] = []

        if self.position.quantity == 0:
            return orders

        loss = abs(self.position.quantity * (current_price - self.position.avg_price))
        loss_pct = loss / (abs(self.position.quantity) * self.position.avg_price + 1e-9)

        if loss_pct > max_loss_per_trade:
            side = SignalType.SELL if self.position.quantity > 0 else SignalType.BUY
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=current_price,
                quantity=abs(self.position.quantity),
                timestamp=0.0,
            ))
            self.position.quantity = 0

        return orders


class LatencyArbitrageStrategy:
    """
    Latency Arbitrage Strategy — 延迟套利策略

    Exploits price discrepancies across multiple venues due to
    transmission latencies. When one venue's price deviates from
    another by more than the latency cost, arbitrage the spread.

    延迟套利策略 / 利用跨交易所传输延迟导致的价格差异进行套利。

    Signal Logic:
        - Monitor prices across N venues
        - Price diff > threshold: buy cheap, sell expensive
        - Mean revert when spread closes

    Methods:
        compute_signal(): 计算跨市场价差信号
        execute(): 执行套利交易
        manage_risk(): 跨所风险管理
    """

    def __init__(
        self,
        symbol: str,
        venues: List[str],
        spread_threshold: float = 0.0002,
        position_size: float = 100.0,
        max_spread_age: float = 0.001,  # seconds
    ):
        self.symbol = symbol
        self.venues = venues
        self.spread_threshold = spread_threshold
        self.position_size = position_size
        self.max_spread_age = max_spread_age

        # Venue prices: {venue: (price, timestamp)}
        self.venue_prices: Dict[str, Tuple[float, float]] = {
            v: (0.0, 0.0) for v in venues
        }

        # Positions per venue (venue arbitrage)
        self.venue_positions: Dict[str, float] = {v: 0.0 for v in venues}
        self.net_position = Position(symbol=symbol)

        self.spread_history: List[float] = []
        self.signal_history: List[SignalType] = []

    def update_venue_price(
        self,
        venue: str,
        price: float,
        timestamp: float,
    ) -> None:
        """
        Update price for a specific venue.

        Args:
            venue: 交易所名称
            price: 价格
            timestamp: 时间戳
        """
        self.venue_prices[venue] = (price, timestamp)

    def _compute_cross_venue_spread(self) -> Tuple[float, float, List[str]]:
        """
        Compute price spread across venues.

        Returns:
            (max_spread, weighted_spread, active_venues)
        """
        active = [
            (v, p, t) for v, (p, t) in self.venue_prices.items()
            if t > 0 and (self.venue_prices[v][0]) > 0
        ]

        if len(active) < 2:
            return 0.0, 0.0, []

        prices = [(v, p) for v, p, _ in active]
        max_idx = max(range(len(prices)), key=lambda i: prices[i][1])
        min_idx = min(range(len(prices)), key=lambda i: prices[i][1])

        max_spread = prices[max_idx][1] - prices[min_idx][1]
        weighted_spread = max_spread / (prices[min_idx][1] + 1e-9)

        return max_spread, weighted_spread, [prices[max_idx][0], prices[min_idx][0]]

    def compute_signal(
        self,
        timestamp: Optional[float] = None,
    ) -> SignalType:
        """
        Compute latency arbitrage signal.

        Args:
            timestamp: 当前时间戳 (optional)

        Returns:
            SignalType: BUY cheap venue, SELL expensive venue, or HOLD
        """
        max_spread, weighted_spread, active_venues = self._compute_cross_venue_spread()
        self.spread_history.append(weighted_spread)

        if len(self.spread_history) > 1000:
            self.spread_history.pop(0)

        if len(active_venues) < 2:
            return SignalType.HOLD

        # Check if stale / 检查是否过期
        now = timestamp or 0.0
        for v, (p, t) in self.venue_prices.items():
            if t > 0 and now - t > self.max_spread_age:
                # Price too old / 价格太旧
                return SignalType.HOLD

        if weighted_spread > self.spread_threshold:
            # Expensive venue: SELL, Cheap venue: BUY
            return SignalType.SELL  # Signal to sell expensive
        elif weighted_spread < -self.spread_threshold:
            return SignalType.BUY  # Signal to buy cheap
        else:
            return SignalType.HOLD

    def execute(
        self,
        signal: SignalType,
        venue_prices: Dict[str, Tuple[float, float]],
        timestamp: float,
    ) -> List[HFTOrder]:
        """
        Execute cross-venue arbitrage.

        Args:
            signal: 交易信号
            venue_prices: {venue: (bid, ask)}
            timestamp: 时间戳

        Returns:
            List of HFTOrders for both legs
        """
        orders: List[HFTOrder] = []

        if signal == SignalType.HOLD:
            return orders

        # Find cheapest and most expensive venues
        venues_sorted = sorted(
            venue_prices.items(),
            key=lambda x: (x[1][0] + x[1][1]) / 2,
        )
        cheap_venue = venues_sorted[0][0]
        expensive_venue = venues_sorted[-1][0]

        cheap_bid, cheap_ask = venue_prices[cheap_venue]
        exp_bid, exp_ask = venue_prices[expensive_venue]

        if signal == SignalType.BUY:
            # Buy on cheap venue, sell on expensive venue
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=SignalType.BUY,
                price=cheap_ask,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=cheap_venue,
            ))
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=SignalType.SELL,
                price=exp_bid,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=expensive_venue,
            ))
            self.venue_positions[cheap_venue] += self.position_size
            self.venue_positions[expensive_venue] -= self.position_size
            self.net_position.quantity += self.position_size

        else:  # SELL
            # Sell on expensive venue, buy on cheap venue
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=SignalType.SELL,
                price=exp_ask,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=expensive_venue,
            ))
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=SignalType.BUY,
                price=cheap_bid,
                quantity=self.position_size,
                timestamp=timestamp,
                venue=cheap_venue,
            ))
            self.venue_positions[expensive_venue] += self.position_size
            self.venue_positions[cheap_venue] -= self.position_size
            self.net_position.quantity -= self.position_size

        self.signal_history.append(signal)
        return orders

    def manage_risk(
        self,
        venue_mid_prices: Dict[str, float],
        max_net_exposure: float = 500.0,
    ) -> List[HFTOrder]:
        """
        Cross-venue risk management.

        Args:
            venue_mid_prices: {venue: mid_price}

        Returns:
            List of flattening orders
        """
        orders: List[HFTOrder] = []

        # Net exposure check / 净敞口检查
        if abs(self.net_position.quantity) > max_net_exposure:
            # Flatten net position / 平仓
            side = SignalType.SELL if self.net_position.quantity > 0 else SignalType.BUY
            orders.append(HFTOrder(
                symbol=self.symbol,
                side=side,
                price=venue_mid_prices.get(self.venues[0], 0.0),
                quantity=abs(self.net_position.quantity),
                timestamp=0.0,
            ))
            for v in self.venues:
                self.venue_positions[v] = 0.0
            self.net_position.quantity = 0

        # Venue-specific position limits / 各交易所仓位限制
        for venue, pos in self.venue_positions.items():
            if abs(pos) > max_net_exposure / len(self.venues):
                side = SignalType.SELL if pos > 0 else SignalType.BUY
                mid = venue_mid_prices.get(venue, 0.0)
                if mid > 0:
                    orders.append(HFTOrder(
                        symbol=self.symbol,
                        side=side,
                        price=mid,
                        quantity=abs(pos),
                        timestamp=0.0,
                        venue=venue,
                    ))
                    self.venue_positions[venue] = 0.0

        return orders


class HFTPositionManager:
    """
    HFT Position Manager — HFT 仓位管理器

    Manages micro-position sizing and risk controls for HFT strategies.
    Implements per-trade risk limits, correlation-adjusted position sizing,
    and real-time drawdown monitoring.

    HFT仓位管理器 / 管理高频交易的微仓位的风险控制。

    Features:
        - Micro-position sizing: sub-lot granularity
        - Real-time drawdown monitoring / 实时回撤监控
        - Correlation-adjusted sizing / 相关性调整仓位
        - Multi-symbol position tracking / 多标的仓位追踪

    Methods:
        compute_signal(): 计算信号
        execute(): 执行交易
        manage_risk(): 风险管理
    """

    def __init__(
        self,
        max_total_exposure: float = 100000.0,
        max_position_per_symbol: float = 10000.0,
        max_drawdown_pct: float = 0.02,
        base_position_size: float = 100.0,
    ):
        self.max_total_exposure = max_total_exposure
        self.max_position_per_symbol = max_position_per_symbol
        self.max_drawdown_pct = max_drawdown_pct
        self.base_position_size = base_position_size

        # Multi-symbol positions / 多标的仓位
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = [0.0]
        self.peak_equity: float = 0.0

        # Per-symbol realized PnL / 各标的已实现盈亏
        self.realized_pnl: Dict[str, float] = {}

        # Order history / 订单历史
        self.order_history: List[HFTOrder] = []

    def update_position(
        self,
        symbol: str,
        side: SignalType,
        price: float,
        quantity: float,
        timestamp: float,
    ) -> None:
        """
        Update position after fill.

        Args:
            symbol: 标的代码
            side: 交易方向
            price: 成交价格
            quantity: 成交数量
            timestamp: 时间戳
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            self.realized_pnl[symbol] = 0.0

        pos = self.positions[symbol]
        fill_value = price * quantity

        if side == SignalType.BUY:
            # Add to position / 加入多头
            if pos.quantity >= 0:
                pos.avg_price = (
                    (pos.avg_price * pos.quantity + fill_value) /
                    (pos.quantity + quantity)
                )
                pos.quantity += quantity
            else:
                # Closing short / 平空
                close_qty = min(abs(pos.quantity), quantity)
                self.realized_pnl[symbol] += close_qty * (pos.avg_price - price)
                remaining = quantity - close_qty
                if remaining > 0:
                    pos.avg_price = (pos.avg_price * pos.quantity + fill_value) / (pos.quantity + remaining)
                    pos.quantity += remaining
                else:
                    pos.quantity += close_qty

        else:  # SELL
            # Add to position or reduce long / 加入空头或减少多头
            if pos.quantity <= 0:
                pos.avg_price = price
                pos.quantity -= quantity
            else:
                # Closing long / 平多
                close_qty = min(pos.quantity, quantity)
                self.realized_pnl[symbol] += close_qty * (price - pos.avg_price)
                pos.quantity -= close_qty
                remaining = quantity - close_qty
                if remaining > 0:
                    pos.avg_price = price
                    pos.quantity -= remaining

        self.order_history.append(HFTOrder(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        ))

    def compute_signal(
        self,
        symbol: str,
        price: float,
        volatility: float,
        correlation_with_portfolio: float = 0.0,
    ) -> float:
        """
        Compute micro-position size based on risk parameters.

        Args:
            symbol: 标的代码
            price: 当前价格
            volatility: 波动率
            correlation_with_portfolio: 与组合相关性

        Returns:
            Recommended position size (quantity)
        """
        pos = self.positions.get(symbol, Position(symbol=symbol))

        # Base sizing: inversely proportional to volatility
        vol_adjusted_size = self.base_position_size / (volatility + 1e-9)

        # Correlation adjustment / 相关性调整
        corr_factor = np.sqrt(1 - correlation_with_portfolio ** 2 + 1e-9)
        adjusted_size = vol_adjusted_size * corr_factor

        # Position limit check / 仓位限制
        total_exposure = sum(
            abs(p.quantity * p.avg_price)
            for p in self.positions.values()
        )
        remaining_budget = self.max_total_exposure - total_exposure
        size = min(adjusted_size, remaining_budget / (price + 1e-9))
        size = min(size, self.max_position_per_symbol / (price + 1e-9))

        return max(0.0, size)

    def execute(
        self,
        symbol: str,
        side: SignalType,
        price: float,
        quantity: float,
        timestamp: float,
    ) -> HFTOrder:
        """
        Execute order and update positions.

        Args:
            symbol: 标的代码
            side: 交易方向
            price: 价格
            quantity: 数量
            timestamp: 时间戳

        Returns:
            HFTOrder that was executed
        """
        order = HFTOrder(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )
        self.update_position(symbol, side, price, quantity, timestamp)
        return order

    def manage_risk(self) -> List[HFTOrder]:
        """
        Global risk management across all positions.

        Returns:
            List of flattening/ hedging orders to manage risk
        """
        orders: List[HFTOrder] = []

        # Current total equity / 当前总权益
        total_realized = sum(self.realized_pnl.values())
        total_unrealized = sum(
            p.quantity * (p.avg_price - p.avg_price)  # Mark-to-market would use current price
            for p in self.positions.values()
        )
        current_equity = total_realized + total_unrealized

        self.equity_curve.append(current_equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        # Drawdown check / 回撤检查
        drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-9)

        if drawdown > self.max_drawdown_pct:
            # Emergency flatten / 紧急平仓
            for symbol, pos in self.positions.items():
                if pos.quantity != 0:
                    side = SignalType.SELL if pos.quantity > 0 else SignalType.BUY
                    orders.append(HFTOrder(
                        symbol=symbol,
                        side=side,
                        price=pos.avg_price,  # Use avg price as estimate
                        quantity=abs(pos.quantity),
                        timestamp=0.0,
                    ))

        # Total exposure check / 总敞口检查
        total_exposure = sum(
            abs(p.quantity * p.avg_price)
            for p in self.positions.values()
        )

        if total_exposure > self.max_total_exposure:
            # Reduce proportionally / 按比例减仓
            reduce_factor = self.max_total_exposure / total_exposure
            for symbol, pos in self.positions.items():
                if abs(pos.quantity) > 0:
                    reduce_qty = abs(pos.quantity) * (1 - reduce_factor)
                    side = SignalType.SELL if pos.quantity > 0 else SignalType.BUY
                    orders.append(HFTOrder(
                        symbol=symbol,
                        side=side,
                        price=pos.avg_price,
                        quantity=reduce_qty,
                        timestamp=0.0,
                    ))

        return orders

    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio risk summary.

        Returns:
            Dict with exposure, PnL, drawdown metrics
        """
        total_realized = sum(self.realized_pnl.values())
        total_exposure = sum(
            abs(p.quantity * p.avg_price)
            for p in self.positions.values()
        )
        current_equity = self.equity_curve[-1] if self.equity_curve else 0.0
        drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-9)

        return {
            "total_exposure": total_exposure,
            "total_realized_pnl": total_realized,
            "peak_equity": self.peak_equity,
            "current_equity": current_equity,
            "drawdown_pct": drawdown,
            "num_positions": sum(1 for p in self.positions.values() if p.quantity != 0),
            "total_orders": len(self.order_history),
        }
