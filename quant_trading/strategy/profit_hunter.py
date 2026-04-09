"""
Profit Hunter Strategy
价格异常检测策略

A momentum-based trading strategy that:
1. Monitors real-time price movements
2. Detects abnormal price swings (超过阈值)
3. Enters positions quickly on confirmed signals
4. Takes profit fast with strict stop-loss

Signals:
- Price spike detection (价格突破)
- Momentum confirmation (动量确认)
- Volume surge (成交量放大)

Example:
    strategy = ProfitHunterStrategy(
        price_change_threshold=0.02,
        momentum_threshold=0.015,
        profit_target=0.01,
        stop_loss=0.005
    )
    signals = strategy.scan(tickers)
    for sig in signals:
        if sig['action'] == 'buy':
            engine.execute_market(sig['symbol'], 'buy', sig['qty'])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

__all__ = ["ProfitHunterStrategy"]


class ProfitHunterStrategy:
    """Profit Hunter — 价格异常检测策略 / Price anomaly detection strategy.

    逻辑:
    1. 实时监控价格变动 / Real-time price monitoring
    2. 检测异常波动 (超过阈值) / Detect abnormal swings (exceeds threshold)
    3. 快速入场, 快速止盈 / Fast entry, fast profit-taking
    4. 严格止损 / Strict stop-loss

    Args:
        price_change_threshold: Price change % to trigger signal (default 0.02 = 2%)
        momentum_threshold: Volume/momentum ratio threshold (default 0.015 = 1.5%)
        profit_target: Take-profit percentage (default 0.01 = 1%)
        stop_loss: Stop-loss percentage (default 0.005 = 0.5%)
        lookback_periods: Number of periods for average calculation (default 5)
    """

    def __init__(
        self,
        price_change_threshold: float = 0.02,
        momentum_threshold: float = 0.015,
        profit_target: float = 0.01,
        stop_loss: float = 0.005,
        lookback_periods: int = 5,
    ):
        self.price_change_threshold = price_change_threshold
        self.momentum_threshold = momentum_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.lookback_periods = lookback_periods

        self.logger = logging.getLogger("ProfitHunterStrategy")
        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}

    def scan(self, tickers: List[dict]) -> List[dict]:
        """扫描所有ticker，返回异常信号 / Scan all tickers, return anomaly signals.

        Args:
            tickers: List of ticker dicts with price, volume, symbol, etc.

        Returns:
            List of signal dicts with action, symbol, quantity, entry price, etc.
        """
        signals = []

        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            if not symbol:
                continue

            current_price = ticker.get("price", 0)
            volume = ticker.get("volume", 0)
            price_change_pct = ticker.get("price_change_pct", 0)

            # Update history
            self._update_history(symbol, current_price, volume)

            # Compute signal
            signal = self.compute_signal(
                current_price=current_price,
                prev_price=self._get_prev_price(symbol),
                volume=volume,
                avg_volume=self._get_avg_volume(symbol),
            )

            if signal.get("action") in ("buy", "sell"):
                signals.append(signal)

        return signals

    def compute_signal(
        self,
        current_price: float,
        prev_price: float,
        volume: float,
        avg_volume: float,
    ) -> dict:
        """计算单个标的信号 / Compute signal for a single asset.

        Args:
            current_price: Current price
            prev_price: Previous price
            volume: Current volume
            avg_volume: Average volume over lookback period

        Returns:
            Signal dict with action, entry price, stop-loss, take-profit, etc.
        """
        if current_price <= 0 or prev_price <= 0:
            return {"action": "hold", "reason": "invalid_price"}

        # Calculate price change
        price_change = (current_price - prev_price) / prev_price
        price_change_abs = abs(price_change)

        # Calculate volume ratio
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Determine action
        action = "hold"
        reason = "no_signal"

        # Strong upward momentum
        if price_change >= self.price_change_threshold and volume_ratio >= 1.5:
            action = "buy"
            reason = "price_spike_up_volume_confirm"

        # Moderate upward momentum
        elif price_change >= self.momentum_threshold and volume_ratio >= 1.2:
            action = "buy"
            reason = "momentum_up_volume_confirm"

        # Strong downward momentum (short signal)
        elif price_change <= -self.price_change_threshold and volume_ratio >= 1.5:
            action = "sell"
            reason = "price_spike_down_volume_confirm"

        # Moderate downward momentum
        elif price_change <= -self.momentum_threshold and volume_ratio >= 1.2:
            action = "sell"
            reason = "momentum_down_volume_confirm"

        # Price drop with volume surge (potential reversal)
        elif price_change <= -self.price_change_threshold * 0.5 and volume_ratio >= 2.0:
            action = "buy"
            reason = "potential_reversal"

        if action == "hold":
            return {"action": "hold", "reason": reason}

        # Calculate entry, stop-loss, take-profit
        if action == "buy":
            entry_price = current_price
            stop_loss = current_price * (1 - self.stop_loss)
            take_profit = current_price * (1 + self.profit_target)
            quantity = self._calc_position_size(volume, avg_volume)
        else:  # sell (short)
            entry_price = current_price
            stop_loss = current_price * (1 + self.stop_loss)
            take_profit = current_price * (1 - self.profit_target)
            quantity = self._calc_position_size(volume, avg_volume)

        return {
            "action": action,
            "reason": reason,
            "symbol": "",  # Will be filled by caller
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": quantity,
            "price_change": price_change,
            "volume_ratio": volume_ratio,
            "confidence": self._calc_confidence(price_change, volume_ratio),
        }

    def _update_history(self, symbol: str, price: float, volume: float) -> None:
        """Update price and volume history for a symbol."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []

        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)

        # Keep only lookback periods
        if len(self._price_history[symbol]) > self.lookback_periods:
            self._price_history[symbol] = self._price_history[symbol][
                -self.lookback_periods :
            ]
        if len(self._volume_history[symbol]) > self.lookback_periods:
            self._volume_history[symbol] = self._volume_history[symbol][
                -self.lookback_periods :
            ]

    def _get_prev_price(self, symbol: str) -> float:
        """Get previous price from history."""
        history = self._price_history.get(symbol, [])
        return history[-1] if len(history) >= 2 else (history[-1] if history else 0)

    def _get_avg_volume(self, symbol: str) -> float:
        """Get average volume from history."""
        history = self._volume_history.get(symbol, [])
        if not history:
            return 0
        return sum(history) / len(history)

    def _calc_position_size(
        self, volume: float, avg_volume: float
    ) -> float:
        """Calculate position size based on volume.

        Args:
            volume: Current volume
            avg_volume: Average volume

        Returns:
            Suggested position size (as a ratio of notional)
        """
        # Base position on volume surge
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Higher volume = larger position (up to a cap)
        size_ratio = min(1.0, volume_ratio / 2.0)

        return size_ratio

    def _calc_confidence(self, price_change: float, volume_ratio: float) -> float:
        """Calculate signal confidence score.

        Args:
            price_change: Price change ratio
            volume_ratio: Volume / average volume

        Returns:
            Confidence score between 0 and 1
        """
        # Price change component (0-0.5)
        price_conf = min(0.5, abs(price_change) * 10)

        # Volume component (0-0.5)
        volume_conf = min(0.5, (volume_ratio - 1.0) / 3.0)

        confidence = price_conf + volume_conf

        # Ensure direction matches
        direction = 1 if price_change > 0 else -1

        return confidence * direction

    # -------------------------------------------------------------------------
    # Risk management helpers / 风险管理辅助
    # -------------------------------------------------------------------------

    def should_take_profit(
        self, current_price: float, entry_price: float, side: str
    ) -> bool:
        """检查是否应该止盈 / Check if take-profit should be triggered.

        Args:
            current_price: Current market price
            entry_price: Entry price of position
            side: 'buy' or 'sell'

        Returns:
            True if profit target reached
        """
        if entry_price == 0:
            return False

        pnl_pct = (current_price - entry_price) / entry_price
        if side == "sell":
            pnl_pct = -pnl_pct

        return pnl_pct >= self.profit_target

    def should_stop_loss(
        self, current_price: float, entry_price: float, side: str
    ) -> bool:
        """检查是否应该止损 / Check if stop-loss should be triggered.

        Args:
            current_price: Current market price
            entry_price: Entry price of position
            side: 'buy' or 'sell'

        Returns:
            True if stop-loss reached
        """
        if entry_price == 0:
            return False

        pnl_pct = (current_price - entry_price) / entry_price
        if side == "sell":
            pnl_pct = -pnl_pct

        return pnl_pct <= -self.stop_loss

    # -------------------------------------------------------------------------
    # Backtest helper / 回测辅助
    # -------------------------------------------------------------------------

    def generate_signals_from_klines(
        self, klines: List[dict]
    ) -> List[dict]:
        """从K线数据生成信号 / Generate signals from historical klines.

        Args:
            klines: List of kline dicts with open, high, low, close, volume

        Returns:
            List of signal dicts
        """
        signals = []

        for i, kline in enumerate(klines):
            if i < self.lookback_periods:
                continue

            current_price = kline.get("close", 0)
            volume = kline.get("volume", 0)

            # Get previous close prices
            prev_prices = [klines[j].get("close", 0) for j in range(i - self.lookback_periods, i)]
            prev_price = prev_prices[-1] if prev_prices else current_price

            # Get average volume
            prev_volumes = [klines[j].get("volume", 0) for j in range(i - self.lookback_periods, i)]
            avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else volume

            signal = self.compute_signal(current_price, prev_price, volume, avg_volume)
            signal["timestamp"] = kline.get("close_time", 0)
            signals.append(signal)

        return signals

    # -------------------------------------------------------------------------
    # State reset / 状态重置
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset strategy state / 重置策略状态."""
        self._price_history.clear()
        self._volume_history.clear()
        self.logger.info("Strategy state reset")
