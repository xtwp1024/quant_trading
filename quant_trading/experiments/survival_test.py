# -*- coding: utf-8 -*-
"""
30-Day Survival Test Framework
=============================

Simulates 30 days of live trading to test system survival.
Tracks survival (didn't blow up account), Sharpe ratio, win rate, and max drawdown.

Usage:
    python survival_test.py --days 30 --initial_capital 10000
    python survival_test.py --scenarios --Monte Carlo 1000

Success Criteria:
- Survival Rate: > 95% (didn't lose > 50% of capital)
- Sharpe Ratio: > 1.0
- Max Drawdown: < 30%
- Win Rate: > 45%
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from quant_trading.risk.manager import RiskManager, RiskConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("SurvivalTest")


# ===================== Data Models =====================

@dataclass
class Trade:
    """Trade record"""
    timestamp: int
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    duration_bars: int


@dataclass
class DailyMetrics:
    """Daily performance metrics"""
    date: str
    equity: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    drawdown_pct: float
    positions_open: int
    trades_count: int
    win_count: int
    loss_count: int


@dataclass
class SurvivalResult:
    """30-day survival test result"""
    test_days: int
    initial_capital: float
    final_capital: float
    survival_rate: float  # 1.0 = survived, 0.0 = blew up
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    daily_metrics: List[DailyMetrics]
    trades: List[Trade]
    survived: bool

    def to_dict(self) -> Dict:
        return {
            "test_days": self.test_days,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "survival_rate": self.survival_rate,
            "survival": self.survived,
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": f"{self.max_drawdown_pct * 100:.2f}%",
            "win_rate": f"{self.win_rate * 100:.2f}%",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 3),
            "criteria_met": {
                "survival_rate_95": self.survival_rate >= 0.95,
                "sharpe_1_0": self.sharpe_ratio >= 1.0,
                "max_dd_30": self.max_drawdown_pct <= 0.30,
                "win_rate_45": self.win_rate >= 0.45,
            }
        }


# ===================== Market Data Generator =====================

class MarketDataGenerator:
    """Generates realistic market data for simulation"""

    def __init__(
        self,
        initial_price: float = 50000.0,
        volatility: float = 0.02,
        drift: float = 0.0,
        trend: str = "sideways"
    ):
        self.initial_price = initial_price
        self.volatility = volatility  # Daily volatility
        self.drift = drift  # Daily drift
        self.trend = trend
        self.price = initial_price

    def reset(self):
        """Reset to initial price"""
        self.price = self.initial_price

    def generate_bar(self) -> Dict:
        """Generate a single bar (1-hour data)"""
        # Use random walk with drift
        daily_vol = self.volatility
        hourly_vol = daily_vol / np.sqrt(24)  # Scale to hourly

        # Random component
        random_return = np.random.normal(0, hourly_vol)

        # Trend component
        if self.trend == "bullish":
            trend_return = daily_vol * 0.3 / 24
        elif self.trend == "bearish":
            trend_return = -daily_vol * 0.3 / 24
        else:
            trend_return = 0

        # Total return
        total_return = random_return + trend_return + self.drift / 24

        # Update price
        self.price *= (1 + total_return)

        # Generate OHLC
        high_mult = 1 + abs(np.random.normal(0, hourly_vol * 0.5))
        low_mult = 1 - abs(np.random.normal(0, hourly_vol * 0.5))
        open_price = self.price * (1 + np.random.normal(0, hourly_vol * 0.3))

        bar = {
            "timestamp": int(time.time() * 1000),
            "open": open_price,
            "high": open_price * high_mult,
            "low": open_price * low_mult,
            "close": self.price,
            "volume": np.random.uniform(100, 1000),
        }
        return bar

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate a full day of hourly bars"""
        bars = []
        for _ in range(num_hours):
            bars.append(self.generate_bar())
        return bars


# ===================== Strategy Base =====================

class SurvivalStrategy:
    """Base strategy for survival testing"""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.position = None  # {"side": "LONG" or "SHORT", "entry_price": float, "size": float}
        self.entry_bar = 0

    def generate_signal(self, bars: List[Dict], bar_number: int) -> Optional[str]:
        """
        Generate trading signal based on bars

        Returns:
            "LONG" - open long position
            "SHORT" - open short position
            "CLOSE" - close any position
            None - no action
        """
        raise NotImplementedError

    def should_stop_loss(self, current_price: float, entry_price: float, side: str) -> bool:
        """Check if stop loss is triggered"""
        stop_loss_pct = self.params.get("stop_loss_pct", 0.02)
        if side == "LONG":
            return current_price < entry_price * (1 - stop_loss_pct)
        else:
            return current_price > entry_price * (1 + stop_loss_pct)

    def should_take_profit(self, current_price: float, entry_price: float, side: str) -> bool:
        """Check if take profit is triggered"""
        take_profit_pct = self.params.get("take_profit_pct", 0.05)
        if side == "LONG":
            return current_price > entry_price * (1 + take_profit_pct)
        else:
            return current_price < entry_price * (1 - take_profit_pct)


class MeanReversionStrategy(SurvivalStrategy):
    """Mean reversion strategy for survival testing"""

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 20)
        self.deviation_threshold = self.params.get("deviation_threshold", 2.0)
        self.prices = []

    def generate_signal(self, bars: List[Dict], bar_number: int) -> Optional[str]:
        """Mean reversion signal"""
        if len(bars) < self.lookback:
            return None

        # Get recent closes
        closes = [b["close"] for b in bars[-self.lookback:]]
        ma = np.mean(closes)
        std = np.std(closes)

        if std == 0:
            return None

        current_price = bars[-1]["close"]
        z_score = (current_price - ma) / std

        # Price too far below mean - expect bounce
        if z_score < -self.deviation_threshold:
            if self.position is None or self.position["side"] != "LONG":
                return "LONG"

        # Price too far above mean - expect drop
        elif z_score > self.deviation_threshold:
            if self.position is None or self.position["side"] != "SHORT":
                return "SHORT"

        # Near mean - close positions
        elif abs(z_score) < 0.5 and self.position is not None:
            return "CLOSE"

        return None


class TrendFollowingStrategy(SurvivalStrategy):
    """Trend following strategy for survival testing"""

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.fast_ma = self.params.get("fast_ma", 10)
        self.slow_ma = self.params.get("slow_ma", 30)
        self.prices = []

    def generate_signal(self, bars: List[Dict], bar_number: int) -> Optional[str]:
        """Trend following signal based on MA crossover"""
        if len(bars) < self.slow_ma:
            return None

        # Calculate MAs
        closes = [b["close"] for b in bars]
        fast_ma = np.mean(closes[-self.fast_ma:])
        slow_ma = np.mean(closes[-self.slow_ma:])

        # Previous MAs
        prev_fast_ma = np.mean(closes[-self.fast_ma-1:-1])
        prev_slow_ma = np.mean(closes[-self.slow_ma-1:-1])

        # Golden cross - buy signal
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            if self.position is None or self.position["side"] != "LONG":
                return "LONG"

        # Death cross - sell signal
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            if self.position is None or self.position["side"] != "SHORT":
                return "SHORT"

        return None


# ===================== Survival Test Engine =====================

class SurvivalTestEngine:
    """30-day survival test engine"""

    def __init__(
        self,
        strategy: SurvivalStrategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size_pct: float = 0.1,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct

        self.cash = initial_capital
        self.peak_cash = initial_capital
        self.position = None
        self.trades: List[Trade] = []
        self.daily_metrics: List[DailyMetrics] = []
        self.bars: List[Dict] = []
        self.bar_number = 0

        # Risk management
        risk_config = RiskConfig(
            max_position_size=initial_capital * position_size_pct * 2,
            max_single_loss=initial_capital * 0.02,
            max_daily_loss=initial_capital * 0.05,
        )
        self.risk_manager = RiskManager(risk_config)

    def reset(self):
        """Reset for new test run"""
        self.cash = self.initial_capital
        self.peak_cash = self.initial_capital
        self.position = None
        self.trades = []
        self.daily_metrics = []
        self.bars = []
        self.bar_number = 0
        self.strategy.position = None
        self.risk_manager.reset_daily()

    def execute_trade(self, side: str, price: float, timestamp: int):
        """Execute a trade"""
        # Close existing position first
        if self.position is not None:
            self._close_position(price, timestamp)

        # Open new position
        if side in ("LONG", "SHORT"):
            position_value = self.cash * self.position_size_pct
            size = position_value / price

            # Account for commission and slippage
            cost = position_value * (1 + self.commission + self.slippage)

            if cost > self.cash:
                logger.warning(f"Insufficient cash for trade: {cost} > {self.cash}")
                return

            self.cash -= cost * 0.1  # Margin requirement
            self.position = {
                "side": side,
                "entry_price": price,
                "size": size,
                "entry_time": timestamp,
                "entry_bar": self.bar_number,
            }

            logger.debug(f"Opened {side} position: price={price}, size={size}")

    def _close_position(self, price: float, timestamp: int):
        """Close current position"""
        if self.position is None:
            return

        entry_price = self.position["entry_price"]
        size = self.position["size"]
        side = self.position["side"]

        # Calculate PnL
        if side == "LONG":
            pnl = (price - entry_price) * size
        else:
            pnl = (entry_price - price) * size

        # Deduct commission
        commission = self.cash * self.commission
        net_pnl = pnl - commission

        # Update cash
        self.cash += net_pnl

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side=side,
            entry_price=entry_price,
            exit_price=price,
            size=size,
            pnl=net_pnl,
            pnl_pct=net_pnl / (entry_price * size) if entry_price * size > 0 else 0,
            commission=commission,
            duration_bars=self.bar_number - self.position["entry_bar"],
        )
        self.trades.append(trade)

        logger.debug(f"Closed {side} position: pnl={net_pnl:.2f}")

        self.position = None

    def check_stops(self, current_price: float, timestamp: int):
        """Check stop loss and take profit"""
        if self.position is None:
            return

        # Stop loss
        if self.strategy.should_stop_loss(
            current_price, self.position["entry_price"], self.position["side"]
        ):
            self._close_position(current_price, timestamp)
            return

        # Take profit
        if self.strategy.should_take_profit(
            current_price, self.position["entry_price"], self.position["side"]
        ):
            self._close_position(current_price, timestamp)

    def get_current_equity(self) -> float:
        """Get current equity including open position"""
        equity = self.cash
        if self.position is not None:
            if self.position["side"] == "LONG":
                unrealized = (self.bars[-1]["close"] - self.position["entry_price"]) * self.position["size"]
            else:
                unrealized = (self.position["entry_price"] - self.bars[-1]["close"]) * self.position["size"]
            equity += unrealized
        return equity

    def record_daily_metrics(self, date: str):
        """Record daily metrics"""
        equity = self.get_current_equity()
        daily_return = (equity - self.peak_cash) / self.peak_cash if self.peak_cash > 0 else 0

        if equity > self.peak_cash:
            self.peak_cash = equity

        drawdown = self.peak_cash - equity
        drawdown_pct = drawdown / self.peak_cash if self.peak_cash > 0 else 0

        # Count today's trades
        today_start = len([t for t in self.trades if t.timestamp >= (time.time() - 86400) * 1000])

        metrics = DailyMetrics(
            date=date,
            equity=equity,
            daily_return=daily_return,
            cumulative_return=(equity - self.initial_capital) / self.initial_capital,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
            positions_open=1 if self.position else 0,
            trades_count=0,
            win_count=len([t for t in self.trades if t.pnl > 0]),
            loss_count=len([t for t in self.trades if t.pnl <= 0]),
        )
        self.daily_metrics.append(metrics)

    def calculate_results(self) -> SurvivalResult:
        """Calculate survival test results"""
        equity = self.get_current_equity()

        # Survival check - didn't lose > 50% of capital
        survived = equity >= self.initial_capital * 0.5

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Avg win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Max drawdown - calculate from equity curve
        equity_values = [self.initial_capital] + [m.equity for m in self.daily_metrics]
        peak = self.initial_capital
        max_dd = 0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_dd_value = max(m.drawdown for m in self.daily_metrics) if self.daily_metrics else 0

        # Sharpe ratio (simplified)
        returns = [m.daily_return for m in self.daily_metrics if m.daily_return != 0]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        return SurvivalResult(
            test_days=len(self.daily_metrics),
            initial_capital=self.initial_capital,
            final_capital=equity,
            survival_rate=1.0 if survived else 0.0,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd_value,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            daily_metrics=self.daily_metrics,
            trades=self.trades,
            survived=survived,
        )

    def run_test(
        self,
        days: int = 30,
        market_generator: MarketDataGenerator = None,
        trend: str = "sideways",
    ) -> SurvivalResult:
        """
        Run 30-day survival test

        Args:
            days: Number of days to simulate
            market_generator: Optional pre-configured market generator
            trend: Market trend - "sideways", "bullish", "bearish"
        """
        logger.info(f"Starting {days}-day survival test with {trend} market")

        if market_generator is None:
            market_generator = MarketDataGenerator(
                initial_price=50000,
                volatility=0.02,
                trend=trend,
            )

        # Simulate each day
        for day in range(days):
            date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
            logger.info(f"Day {day + 1}/{days}: {date}")

            # Generate 24 hourly bars
            daily_bars = market_generator.generate_daily_bars(24)

            for bar in daily_bars:
                self.bar_number += 1
                self.bars.append(bar)

                # Check stops first
                self.check_stops(bar["close"], bar["timestamp"])

                # Generate signal
                signal = self.strategy.generate_signal(self.bars, self.bar_number)

                if signal == "CLOSE" and self.position is not None:
                    self._close_position(bar["close"], bar["timestamp"])
                elif signal in ("LONG", "SHORT"):
                    self.execute_trade(signal, bar["close"], bar["timestamp"])

            # Record daily metrics
            self.record_daily_metrics(date)

            # Check survival
            equity = self.get_current_equity()
            if equity < self.initial_capital * 0.5:
                logger.warning(f"SURVIVAL FAILURE: Equity {equity} below 50% threshold")
                break

        # Close any open position
        if self.position and self.bars:
            self._close_position(self.bars[-1]["close"], self.bars[-1]["timestamp"])

        result = self.calculate_results()
        logger.info(f"Survival test complete: survived={result.survived}, equity={result.final_capital:.2f}")

        return result


# ===================== Main =====================

def run_survival_tests(
    num_runs: int = 10,
    days: int = 30,
    initial_capital: float = 10000.0,
) -> List[SurvivalResult]:
    """Run multiple survival tests with different market conditions"""
    results = []

    strategies = [
        ("MeanReversion", MeanReversionStrategy()),
        ("TrendFollowing", TrendFollowingStrategy()),
    ]

    trends = ["sideways", "bullish", "bearish"]

    for trend in trends:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with {trend} market")
        logger.info(f"{'='*50}")

        for strategy_name, strategy in strategies:
            logger.info(f"\nStrategy: {strategy_name}")

            for run in range(num_runs):
                engine = SurvivalTestEngine(
                    strategy=strategy,
                    initial_capital=initial_capital,
                )

                result = engine.run_test(days=days, trend=trend)
                results.append(result)

                logger.info(
                    f"Run {run + 1}/{num_runs}: "
                    f"Survived={result.survived}, "
                    f"Sharpe={result.sharpe_ratio:.2f}, "
                    f"WinRate={result.win_rate:.2%}, "
                    f"MaxDD={result.max_drawdown_pct:.2%}"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="30-Day Survival Test Framework")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--runs", type=int, default=10, help="Number of test runs per scenario")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--trend", type=str, default="sideways", choices=["sideways", "bullish", "bearish"])
    parser.add_argument("--output-dir", type=str, default=None, help='Output directory (for compatibility with run_all_experiments)')

    args = parser.parse_args()

    # Run tests
    results = run_survival_tests(
        num_runs=args.runs,
        days=args.days,
        initial_capital=args.initial_capital,
    )

    # Aggregate statistics
    survival_count = sum(1 for r in results if r.survived)
    avg_survival_rate = np.mean([r.survival_rate for r in results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    avg_win_rate = np.mean([r.win_rate for r in results])
    avg_max_dd = np.mean([r.max_drawdown_pct for r in results])

    print("\n" + "="*60)
    print("SURVIVAL TEST SUMMARY")
    print("="*60)
    print(f"Total runs: {len(results)}")
    print(f"Survival rate: {survival_count}/{len(results)} ({survival_count/len(results)*100:.1f}%)")
    print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Average Win Rate: {avg_win_rate:.2%}")
    print(f"Average Max Drawdown: {avg_max_dd:.2%}")
    print(f"\nSuccess Criteria:")
    print(f"  Survival Rate > 95%: {'PASS' if avg_survival_rate >= 0.95 else 'FAIL'}")
    print(f"  Sharpe Ratio > 1.0: {'PASS' if avg_sharpe >= 1.0 else 'FAIL'}")
    print(f"  Max Drawdown < 30%: {'PASS' if avg_max_dd <= 0.30 else 'FAIL'}")
    print(f"  Win Rate > 45%: {'PASS' if avg_win_rate >= 0.45 else 'FAIL'}")
    print("="*60)

    # Save results
    if args.output:
        output_data = {
            "summary": {
                "total_runs": len(results),
                "survival_rate": avg_survival_rate,
                "avg_sharpe": avg_sharpe,
                "avg_win_rate": avg_win_rate,
                "avg_max_dd": avg_max_dd,
            },
            "individual_results": [r.to_dict() for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
