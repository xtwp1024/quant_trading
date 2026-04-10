# -*- coding: utf-8 -*-
"""
Stress Test Scenarios
====================

Black Thursday (2020-03-12): -12% in one day
Flash Crash: sudden spike and recovery
Long Bear Market: 70% decline over 6 months
High Volatility: 5x normal VIX
Liquidity Crisis: wide spreads

Usage:
    python stress_tests.py
    python stress_tests.py --scenario black_thursday
    python stress_tests.py --all --runs 100

Success Criteria under stress:
- Survival Rate: > 90% (didn't lose > 50% of capital)
- Sharpe Ratio: > 0.5 (still profitable)
- Max Drawdown: < 40% (higher tolerance in stress)
- Win Rate: > 40% (lower tolerance in stress)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
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
logger = logging.getLogger("StressTests")


# ===================== Stress Scenario Definitions =====================

class StressScenario(Enum):
    """Stress test scenario types"""
    BLACK_THURSDAY = "black_thursday"
    FLASH_CRASH = "flash_crash"
    LONG_BEAR = "long_bear"
    HIGH_VOLATILITY = "high_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    ALL = "all"


@dataclass
class StressTestResult:
    """Stress test result"""
    scenario: str
    initial_capital: float
    final_capital: float
    survival_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    peak_trough_ratio: float  # Maximum equity / minimum equity during test
    days_to_recover: int  # Days to recover to initial capital (or -1 if not recovered)
    bad_days: int  # Days with > 5% loss
    good_days: int  # Days with > 5% gain

    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "survival_rate": self.survival_rate,
            "max_drawdown_pct": f"{self.max_drawdown_pct * 100:.2f}%",
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "win_rate": f"{self.win_rate * 100:.2f}%",
            "total_trades": self.total_trades,
            "peak_trough_ratio": round(self.peak_trough_ratio, 3),
            "days_to_recover": self.days_to_recover,
            "bad_days": self.bad_days,
            "good_days": self.good_days,
        }


# ===================== Market Data Generators for Stress Scenarios =====================

class BlackThursdayGenerator:
    """
    Black Thursday (2020-03-12) simulation
    Real event: BTC dropped ~50% in 24 hours, from ~$7900 to ~$3900
    """

    def __init__(self, initial_price: float = 7900.0):
        self.initial_price = initial_price
        self.price = initial_price
        self.day = 0

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate bars with Black Thursday crash"""
        self.day += 1
        bars = []

        for hour in range(num_hours):
            if self.day == 1:
                # Day 1: Normal volatility
                vol = 0.01
                drift = 0
            elif self.day == 2:
                # Day 2: Flash crash
                if hour < 12:
                    # Gradual decline
                    vol = 0.02
                    drift = -0.03
                else:
                    # Sudden crash
                    vol = 0.05
                    drift = -0.08
            elif self.day == 3:
                # Day 3: Continued panic selling
                vol = 0.03
                drift = -0.02
            else:
                # Recovery attempt
                vol = 0.02
                drift = 0.01

            random_return = np.random.normal(drift / 24, vol / np.sqrt(24))
            self.price *= (1 + random_return)

            bar = {
                "timestamp": int(time.time() * 1000) + (self.day - 1) * 86400000 + hour * 3600000,
                "open": self.price * (1 + np.random.normal(0, 0.005)),
                "high": self.price * (1 + abs(np.random.normal(0, 0.01))),
                "low": self.price * (1 - abs(np.random.normal(0, 0.01))),
                "close": self.price,
                "volume": np.random.uniform(500, 2000),
            }
            bars.append(bar)

        return bars


class FlashCrashGenerator:
    """
    Flash Crash simulation
    Sudden spike down 20% and recovery within the same day
    """

    def __init__(self, initial_price: float = 50000.0):
        self.initial_price = initial_price
        self.price = initial_price
        self.day = 0

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate bars with flash crash"""
        self.day += 1
        bars = []

        for hour in range(num_hours):
            if self.day == 1:
                # Normal day
                vol = 0.01
                drift = 0
            elif self.day == 2:
                # Flash crash day
                if hour < 10:
                    # Normal trading
                    vol = 0.01
                    drift = 0
                elif hour == 10:
                    # Sudden crash
                    vol = 0.10
                    drift = -0.20
                elif hour < 16:
                    # Volatile recovery
                    vol = 0.05
                    drift = 0.03
                else:
                    # Return to normal
                    vol = 0.01
                    drift = 0
            else:
                # Post-crash normal
                vol = 0.01
                drift = 0

            random_return = np.random.normal(drift / 24, vol / np.sqrt(24))
            self.price *= (1 + random_return)

            bar = {
                "timestamp": int(time.time() * 1000) + (self.day - 1) * 86400000 + hour * 3600000,
                "open": self.price * (1 + np.random.normal(0, 0.005)),
                "high": self.price * (1 + abs(np.random.normal(0, 0.01))),
                "low": self.price * (1 - abs(np.random.normal(0, 0.01))),
                "close": self.price,
                "volume": np.random.uniform(500, 2000),
            }
            bars.append(bar)

        return bars


class LongBearMarketGenerator:
    """
    Long Bear Market simulation
    70% decline over 6 months (~180 days)
    """

    def __init__(self, initial_price: float = 60000.0, decline_pct: float = 0.70):
        self.initial_price = initial_price
        self.target_price = initial_price * (1 - decline_pct)
        self.price = initial_price
        self.day = 0
        self.decline_pct = decline_pct

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate bars for bear market"""
        self.day += 1

        # Calculate daily decline to reach target over 180 days
        daily_decline = (self.target_price / self.initial_price) ** (1 / 180)

        bars = []
        for hour in range(num_hours):
            # Volatile downward trend
            vol = 0.03  # High volatility
            daily_drift = (daily_decline - 1) + 0.005  # Additional drift

            random_return = np.random.normal(daily_drift / 24, vol / np.sqrt(24))
            self.price *= (1 + random_return)

            # Don't go below target
            self.price = max(self.price, self.target_price * 0.8)

            bar = {
                "timestamp": int(time.time() * 1000) + (self.day - 1) * 86400000 + hour * 3600000,
                "open": self.price * (1 + np.random.normal(0, 0.01)),
                "high": self.price * (1 + abs(np.random.normal(0, 0.02))),
                "low": self.price * (1 - abs(np.random.normal(0, 0.02))),
                "close": self.price,
                "volume": np.random.uniform(800, 1500),
            }
            bars.append(bar)

        return bars


class HighVolatilityGenerator:
    """
    High Volatility scenario
    5x normal VIX (volatility index)
    Normal VIX ~ 20, High VIX ~ 100
    """

    def __init__(self, initial_price: float = 50000.0, vol_multiplier: float = 5.0):
        self.initial_price = initial_price
        self.price = initial_price
        self.vol_multiplier = vol_multiplier
        self.day = 0

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate bars with extremely high volatility"""
        self.day += 1
        bars = []

        # 5x normal volatility
        vol = 0.02 * self.vol_multiplier / np.sqrt(24)

        for hour in range(num_hours):
            # High volatility random walk with no drift
            random_return = np.random.normal(0, vol)
            self.price *= (1 + random_return)

            bar = {
                "timestamp": int(time.time() * 1000) + (self.day - 1) * 86400000 + hour * 3600000,
                "open": self.price * (1 + np.random.normal(0, vol * 0.5)),
                "high": self.price * (1 + abs(np.random.normal(0, vol))),
                "low": self.price * (1 - abs(np.random.normal(0, vol))),
                "close": self.price,
                "volume": np.random.uniform(1000, 3000),
            }
            bars.append(bar)

        return bars


class LiquidityCrisisGenerator:
    """
    Liquidity Crisis simulation
    Wide spreads, high impact costs
    """

    def __init__(self, initial_price: float = 50000.0, spread_multiplier: float = 10.0):
        self.initial_price = initial_price
        self.price = initial_price
        self.spread_multiplier = spread_multiplier
        self.day = 0
        self.spread = 0.0001  # Normal spread

    def generate_daily_bars(self, num_hours: int = 24) -> List[Dict]:
        """Generate bars with liquidity crisis (wide spreads)"""
        self.day += 1
        bars = []

        for hour in range(num_hours):
            # Normal price movement
            vol = 0.01
            drift = 0

            random_return = np.random.normal(drift / 24, vol / np.sqrt(24))
            self.price *= (1 + random_return)

            # Wide spread simulation
            spread = self.spread * self.spread_multiplier

            bar = {
                "timestamp": int(time.time() * 1000) + (self.day - 1) * 86400000 + hour * 3600000,
                "open": self.price * (1 + np.random.normal(0, spread)),
                "high": self.price * (1 + abs(np.random.normal(0, spread * 2))),
                "low": self.price * (1 - abs(np.random.normal(0, spread * 2))),
                "close": self.price,
                "volume": np.random.uniform(200, 500),  # Low volume
                "spread": spread,
            }
            bars.append(bar)

        return bars


# ===================== Stress Test Engine =====================

class StressTestEngine:
    """Stress test engine"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        position_size_pct: float = 0.1,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct

        self.reset()

    def reset(self):
        """Reset state"""
        self.cash = self.initial_capital
        self.peak_cash = self.initial_capital
        self.position = None
        self.equity_curve = []
        self.trades = []

    def execute_strategy(
        self,
        market_generator,
        days: int,
        strategy_func: Callable,
    ) -> StressTestResult:
        """
        Execute stress test with given market generator

        Args:
            market_generator: Market data generator for the scenario
            days: Number of days to simulate
            strategy_func: Function that takes current bar and returns action

        Returns:
            StressTestResult with test metrics
        """
        self.reset()

        all_bars = []
        peak_equity = self.initial_capital
        trough_equity = self.initial_capital
        day_equities = []
        bad_days = 0
        good_days = 0

        for day in range(days):
            bars = market_generator.generate_daily_bars(24)
            all_bars.extend(bars)

            # Apply wider spreads if liquidity crisis
            spread_multiplier = getattr(market_generator, 'spread_multiplier', 1.0)

            for bar in bars:
                # Simple MA crossover strategy
                if len(all_bars) >= 30:
                    ma10 = np.mean([b["close"] for b in all_bars[-10:]])
                    ma30 = np.mean([b["close"] for b in all_bars[-30:]])
                    prev_ma10 = np.mean([b["close"] for b in all_bars[-11:-1]])
                    prev_ma30 = np.mean([b["close"] for b in all_bars[-31:-1]])

                    # Trading logic
                    if self.position is None:
                        if prev_ma10 <= prev_ma30 and ma10 > ma30:
                            # Golden cross - buy
                            price = bar["close"]
                            size = (self.cash * self.position_size_pct) / price
                            self.position = {
                                "side": "LONG",
                                "entry_price": price,
                                "size": size,
                            }
                            # Extra commission in liquidity crisis
                            self.cash -= price * size * self.commission * spread_multiplier
                    else:
                        # Check exit
                        should_exit = False

                        # Stop loss at 5%
                        if bar["close"] < self.position["entry_price"] * 0.95:
                            should_exit = True

                        # Take profit at 10%
                        elif bar["close"] > self.position["entry_price"] * 1.10:
                            should_exit = True

                        # Death cross exit
                        elif prev_ma10 >= prev_ma30 and ma10 < ma30:
                            should_exit = True

                        if should_exit:
                            price = bar["close"]
                            pnl = (price - self.position["entry_price"]) * self.position["size"]
                            commission = price * self.position["size"] * self.commission * spread_multiplier
                            self.cash += pnl - commission
                            self.trades.append({"pnl": pnl - commission})
                            self.position = None

                # Record equity
                equity = self.cash
                if self.position:
                    unrealized = (bar["close"] - self.position["entry_price"]) * self.position["size"]
                    equity += unrealized

                self.equity_curve.append(equity)
                peak_equity = max(peak_equity, equity)
                trough_equity = min(trough_equity, equity)

            # Calculate daily return
            day_start = self.equity_curve[-(len(bars) + 1)] if len(self.equity_curve) > len(bars) else self.initial_capital
            day_end = self.equity_curve[-1]
            daily_return = (day_end - day_start) / day_start if day_start > 0 else 0

            if daily_return < -0.05:
                bad_days += 1
            elif daily_return > 0.05:
                good_days += 1

            day_equities.append(day_end)

        # Close any open position
        if self.position and all_bars:
            last_bar = all_bars[-1]
            pnl = (last_bar["close"] - self.position["entry_price"]) * self.position["size"]
            self.cash += pnl
            self.position = None

        # Calculate metrics
        final_equity = self.cash
        surviving = final_equity >= self.initial_capital * 0.5

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Days to recover
        days_to_recover = -1
        if final_equity >= self.initial_capital:
            for i, eq in enumerate(self.equity_curve):
                if eq >= self.initial_capital:
                    days_to_recover = i // 24  # Convert bars to days
                    break

        # Win rate
        winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Sharpe (simplified)
        returns = []
        for i in range(24, len(self.equity_curve), 24):
            prev = self.equity_curve[i - 24]
            curr = self.equity_curve[i]
            if prev > 0:
                returns.append((curr - prev) / prev)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        return StressTestResult(
            scenario=market_generator.__class__.__name__.replace("Generator", ""),
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            survival_rate=1.0 if surviving else 0.0,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=len(self.trades),
            peak_trough_ratio=peak_equity / trough_equity if trough_equity > 0 else 0,
            days_to_recover=days_to_recover,
            bad_days=bad_days,
            good_days=good_days,
        )


# ===================== Scenario Runners =====================

def run_stress_scenario(
    scenario: StressScenario,
    days: int = 30,
    initial_capital: float = 10000.0,
    runs: int = 10,
) -> List[StressTestResult]:
    """Run stress test for a specific scenario"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running stress test: {scenario.value}")
    logger.info(f"{'='*60}")

    results = []

    for run in range(runs):
        # Create market generator based on scenario
        if scenario == StressScenario.BLACK_THURSDAY:
            generator = BlackThursdayGenerator()
            test_days = 10  # Short test for acute crisis
        elif scenario == StressScenario.FLASH_CRASH:
            generator = FlashCrashGenerator()
            test_days = 5  # Short test for flash events
        elif scenario == StressScenario.LONG_BEAR:
            generator = LongBearMarketGenerator()
            test_days = 180  # 6 months
        elif scenario == StressScenario.HIGH_VOLATILITY:
            generator = HighVolatilityGenerator()
            test_days = days
        elif scenario == StressScenario.LIQUIDITY_CRISIS:
            generator = LiquidityCrisisGenerator()
            test_days = days
        else:
            continue

        engine = StressTestEngine(initial_capital=initial_capital)
        result = engine.execute_strategy(generator, test_days, None)
        results.append(result)

        logger.info(
            f"Run {run + 1}/{runs}: "
            f"Survived={result.survival_rate > 0}, "
            f"Final=${result.final_capital:.2f}, "
            f"MaxDD={result.max_drawdown_pct:.2%}"
        )

    return results


def run_all_stress_scenarios(
    days: int = 30,
    initial_capital: float = 10000.0,
    runs: int = 10,
) -> Dict[str, List[StressTestResult]]:
    """Run all stress scenarios"""
    all_results = {}

    for scenario in StressScenario:
        if scenario == StressScenario.ALL:
            continue
        results = run_stress_scenario(
            scenario,
            days=days,
            initial_capital=initial_capital,
            runs=runs,
        )
        all_results[scenario.value] = results

    return all_results


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Stress Test Scenarios")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["black_thursday", "flash_crash", "long_bear",
                                "high_volatility", "liquidity_crisis", "all"])
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--runs", type=int, default=10, help="Number of test runs per scenario")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help='Output directory (for compatibility with run_all_experiments)')

    args = parser.parse_args()

    # Convert string to enum
    if args.scenario == "all":
        scenario = StressScenario.ALL
    else:
        scenario = StressScenario(args.scenario)

    # Run tests
    if scenario == StressScenario.ALL:
        all_results = run_all_stress_scenarios(
            days=args.days,
            initial_capital=args.initial_capital,
            runs=args.runs,
        )
    else:
        results = run_stress_scenario(
            scenario,
            days=args.days,
            initial_capital=args.initial_capital,
            runs=args.runs,
        )
        all_results = {scenario.value: results}

    # Print summary
    print("\n" + "="*70)
    print("STRESS TEST RESULTS SUMMARY")
    print("="*70)

    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print("-" * 50)

        survival_count = sum(1 for r in results if r.survival_rate > 0)
        avg_survival = np.mean([r.survival_rate for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        avg_max_dd = np.mean([r.max_drawdown_pct for r in results])
        avg_peak_trough = np.mean([r.peak_trough_ratio for r in results])

        print(f"  Survival Rate: {survival_count}/{len(results)} ({avg_survival*100:.1f}%)")
        print(f"  Avg Sharpe: {avg_sharpe:.3f}")
        print(f"  Avg Win Rate: {avg_win_rate:.2%}")
        print(f"  Avg Max Drawdown: {avg_max_dd:.2%}")
        print(f"  Avg Peak/Trough: {avg_peak_trough:.3f}")

    # Success criteria for stress tests (more lenient)
    print("\n" + "="*70)
    print("STRESS TEST SUCCESS CRITERIA (More Lenient)")
    print("="*70)
    print("  Survival Rate > 90%: ", end="")
    all_survival = [r.survival_rate for rs in all_results.values() for r in rs]
    print("PASS" if np.mean(all_survival) >= 0.90 else "FAIL")
    print("  Sharpe Ratio > 0.5: ", end="")
    all_sharpe = [r.sharpe_ratio for rs in all_results.values() for r in rs]
    print("PASS" if np.mean(all_sharpe) >= 0.5 else "FAIL")
    print("  Max Drawdown < 40%: ", end="")
    all_dd = [r.max_drawdown_pct for rs in all_results.values() for r in rs]
    print("PASS" if np.mean(all_dd) <= 0.40 else "FAIL")
    print("  Win Rate > 40%: ", end="")
    all_wr = [r.win_rate for rs in all_results.values() for r in rs]
    print("PASS" if np.mean(all_wr) >= 0.40 else "FAIL")
    print("="*70)

    # Save results
    if args.output:
        output_data = {
            scenario_name: [r.to_dict() for r in results]
            for scenario_name, results in all_results.items()
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
