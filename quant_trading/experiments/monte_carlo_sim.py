# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation
=====================

Simulates thousands of possible paths to calculate probability of ruin
and find safe trading parameters.

Usage:
    python monte_carlo_sim.py --paths 10000 --days 30
    python monte_carlo_sim.py --optimize
    python monte_carlo_sim.py --plot

Success Criteria:
- Probability of Ruin: < 5% (lost > 50% of capital)
- Expected Return: > 0
- Safe Parameter Space: identified
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from itertools import product

import numpy as np

# Add project path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MonteCarloSim")


# ===================== Data Models =====================

@dataclass
class SimulationPath:
    """Single Monte Carlo simulation path"""
    path_id: int
    final_equity: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    daily_returns: List[float]
    equity_curve: List[float]
    survived: bool
    ruin_day: Optional[int] = None  # Day when equity dropped below 50%

    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "final_equity": round(self.final_equity, 3),
            "max_drawdown": round(self.max_drawdown, 3),
            "max_drawdown_pct": f"{self.max_drawdown_pct * 100:.2f}%",
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "win_rate": f"{self.win_rate * 100:.2f}%",
            "total_trades": self.total_trades,
            "survived": self.survived,
            "ruin_day": self.ruin_day,
        }


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    num_paths: int
    initial_capital: float
    survival_rate: float
    probability_of_ruin: float
    expected_return: float
    expected_sharpe: float
    expected_max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    percentile_5_return: float
    percentile_95_return: float
    best_path: SimulationPath
    worst_path: SimulationPath
    paths: List[SimulationPath]

    def to_dict(self) -> Dict:
        return {
            "num_paths": self.num_paths,
            "initial_capital": self.initial_capital,
            "survival_rate": f"{self.survival_rate * 100:.2f}%",
            "probability_of_ruin": f"{self.probability_of_ruin * 100:.2f}%",
            "expected_return": f"{self.expected_return * 100:.2f}%",
            "expected_sharpe": round(self.expected_sharpe, 3),
            "expected_max_drawdown": f"{self.expected_max_drawdown * 100:.2f}%",
            "var_95": f"{self.var_95 * 100:.2f}%",
            "cvar_95": f"{self.cvar_95 * 100:.2f}%",
            "percentile_5_return": f"{self.percentile_5_return * 100:.2f}%",
            "percentile_95_return": f"{self.percentile_95_return * 100:.2f}%",
            "best_path": self.best_path.to_dict(),
            "worst_path": self.worst_path.to_dict(),
        }


@dataclass
class ParameterSet:
    """Trading parameter set for optimization"""
    position_size: float
    stop_loss: float
    take_profit: float
    win_rate: float
    avg_win: float
    avg_loss: float
    survival_rate: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "survival_rate": self.survival_rate,
            "expected_return": self.expected_return,
            "sharpe_ratio": self.sharpe_ratio,
        }


# ===================== Monte Carlo Engine =====================

class MonteCarloEngine:
    """
    Monte Carlo simulation engine for trading strategy survival analysis.

    Uses geometric Brownian motion to simulate price paths and
    evaluates strategy performance across thousands of scenarios.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        days: int = 30,
        trades_per_day: int = 2,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.days = days
        self.trades_per_day = trades_per_day
        self.commission = commission
        self.slippage = slippage

    def simulate_price_path(
        self,
        S0: float,
        mu: float,  # Drift (daily return)
        sigma: float,  # Volatility
        days: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Simulate price path using Geometric Brownian Motion

        Args:
            S0: Initial price
            mu: Daily drift
            sigma: Daily volatility
            days: Number of days

        Returns:
            Tuple of (prices, daily_returns)
        """
        dt = 1  # Daily steps
        num_steps = days

        # Generate random returns
        Z = np.random.standard_normal(num_steps)
        returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

        # Calculate prices
        prices = [S0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        # Calculate daily returns (excluding first)
        daily_returns = returns.tolist()

        return prices, daily_returns

    def simulate_trading_path(
        self,
        params: ParameterSet,
        prices: List[float],
        daily_returns: List[float],
    ) -> SimulationPath:
        """
        Simulate trading on a price path with given parameters

        Args:
            params: Trading parameters
            prices: Simulated price path
            daily_returns: Daily returns

        Returns:
            SimulationPath with results
        """
        equity = self.initial_capital
        peak_equity = equity
        position = None

        equity_curve = [equity]
        daily_pnl = []
        trades_count = 0
        wins = 0
        losses = 0

        ruin_day = None

        # Simulate each day
        for day, ret in enumerate(daily_returns):
            # Simple strategy: trade with probability based on win rate
            if position is None and np.random.random() < self.trades_per_day / 24:
                # Open position
                position = {
                    "entry_price": prices[day],
                    "side": np.random.choice(["LONG", "SHORT"], p=[0.5, 0.5]),
                }

            # Check position
            if position is not None:
                current_price = prices[day + 1]

                if position["side"] == "LONG":
                    pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                else:
                    pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]

                # Check stop loss
                if pnl_pct <= -params.stop_loss:
                    # Loss
                    pnl = equity * params.position_size * (-params.stop_loss)
                    equity += pnl - equity * self.commission
                    trades_count += 1
                    losses += 1
                    position = None

                # Check take profit
                elif pnl_pct >= params.take_profit:
                    # Win
                    pnl = equity * params.position_size * params.take_profit
                    equity += pnl - equity * self.commission
                    trades_count += 1
                    wins += 1
                    position = None

            # Update equity curve
            if position is not None:
                if position["side"] == "LONG":
                    unrealized = equity * params.position_size * ret
                else:
                    unrealized = -equity * params.position_size * ret
                current_equity = equity + unrealized
            else:
                current_equity = equity

            equity_curve.append(current_equity)

            # Track peak and ruin
            if current_equity > peak_equity:
                peak_equity = current_equity

            # Check for ruin (equity < 50% of initial)
            if current_equity < self.initial_capital * 0.5 and ruin_day is None:
                ruin_day = day

            # Daily PnL
            daily_pnl.append((current_equity - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0)

        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calculate Sharpe ratio
        returns = [equity_curve[i+1] / equity_curve[i] - 1 for i in range(len(equity_curve) - 1) if equity_curve[i] > 0]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        # Win rate
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        return SimulationPath(
            path_id=0,
            final_equity=equity,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=trades_count,
            daily_returns=daily_pnl,
            equity_curve=equity_curve,
            survived=equity >= self.initial_capital * 0.5,
            ruin_day=ruin_day,
        )

    def run_simulation(
        self,
        params: ParameterSet,
        num_paths: int = 1000,
        S0: float = 50000.0,
        mu: float = 0.0,
        sigma: float = 0.02,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation

        Args:
            params: Trading parameters
            num_paths: Number of simulation paths
            S0: Initial price
            mu: Expected daily return (drift)
            sigma: Daily volatility

        Returns:
            MonteCarloResult with aggregated statistics
        """
        logger.info(f"Running {num_paths} Monte Carlo paths...")

        paths = []
        all_returns = []
        all_sharpes = []
        all_max_dds = []

        for i in range(num_paths):
            # Simulate price path
            prices, daily_returns = self.simulate_price_path(S0, mu, sigma, self.days)

            # Simulate trading
            path = self.simulate_trading_path(params, prices, daily_returns)
            path.path_id = i
            paths.append(path)

            # Track statistics
            path_return = (path.final_equity - self.initial_capital) / self.initial_capital
            all_returns.append(path_return)
            all_sharpes.append(path.sharpe_ratio)
            all_max_dds.append(path.max_drawdown_pct)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Completed {i + 1}/{num_paths} paths...")

        # Aggregate results
        survival_count = sum(1 for p in paths if p.survived)
        survival_rate = survival_count / num_paths
        prob_ruin = 1 - survival_rate

        # Find best and worst paths
        sorted_paths = sorted(paths, key=lambda x: x.final_equity, reverse=True)
        best_path = sorted_paths[0]
        worst_path = sorted_paths[-1]

        # VaR and CVaR
        returns_sorted = sorted(all_returns)
        var_95_idx = int(num_paths * 0.05)
        var_95 = returns_sorted[var_95_idx]
        cvar_95 = np.mean(returns_sorted[:var_95_idx])

        # Percentiles
        percentile_5 = np.percentile(all_returns, 5)
        percentile_95 = np.percentile(all_returns, 95)

        result = MonteCarloResult(
            num_paths=num_paths,
            initial_capital=self.initial_capital,
            survival_rate=survival_rate,
            probability_of_ruin=prob_ruin,
            expected_return=np.mean(all_returns),
            expected_sharpe=np.mean(all_sharpes),
            expected_max_drawdown=np.mean(all_max_dds),
            var_95=var_95,
            cvar_95=cvar_95,
            percentile_5_return=percentile_5,
            percentile_95_return=percentile_95,
            best_path=best_path,
            worst_path=worst_path,
            paths=paths,
        )

        logger.info(
            f"Simulation complete: Survival={survival_rate*100:.1f}%, "
            f"PoR={prob_ruin*100:.1f}%, Sharpe={result.expected_sharpe:.3f}"
        )

        return result


# ===================== Parameter Optimization =====================

class ParameterOptimizer:
    """
    Find safe parameter space using Monte Carlo simulation
    """

    def __init__(
        self,
        engine: MonteCarloEngine,
        num_paths: int = 500,
    ):
        self.engine = engine
        self.num_paths = num_paths

    def grid_search(
        self,
        position_sizes: List[float],
        stop_losses: List[float],
        take_profits: List[float],
        win_rates: List[float],
        avg_wins: List[float],
        avg_losses: List[float],
    ) -> List[ParameterSet]:
        """
        Grid search over parameter space

        Returns:
            List of ParameterSet with results
        """
        results = []

        total_combinations = (
            len(position_sizes) * len(stop_losses) * len(take_profits) *
            len(win_rates) * len(avg_wins) * len(avg_losses)
        )

        logger.info(f"Grid search over {total_combinations} parameter combinations")
        logger.info(f"Using {self.num_paths} paths per combination")

        count = 0
        for ps, sl, tp, wr, aw, al in product(
            position_sizes, stop_losses, take_profits,
            win_rates, avg_wins, avg_losses
        ):
            count += 1

            params = ParameterSet(
                position_size=ps,
                stop_loss=sl,
                take_profit=tp,
                win_rate=wr,
                avg_win=aw,
                avg_loss=al,
            )

            # Run Monte Carlo
            result = self.engine.run_simulation(params, num_paths=self.num_paths)

            params.survival_rate = result.survival_rate
            params.expected_return = result.expected_return
            params.sharpe_ratio = result.expected_sharpe
            results.append(params)

            if count % 10 == 0:
                logger.info(f"  Progress: {count}/{total_combinations}")

        # Sort by survival rate and Sharpe
        results.sort(key=lambda x: (x.survival_rate, x.sharpe_ratio), reverse=True)

        return results

    def find_safe_parameters(
        self,
        min_survival_rate: float = 0.95,
        min_sharpe: float = 1.0,
    ) -> List[ParameterSet]:
        """
        Find parameters that meet safe trading criteria

        Returns:
            List of ParameterSet that meet criteria
        """
        # Define search space
        position_sizes = [0.05, 0.10, 0.15, 0.20]
        stop_losses = [0.02, 0.03, 0.05, 0.07]
        take_profits = [0.04, 0.06, 0.10, 0.15]
        win_rates = [0.40, 0.45, 0.50, 0.55]
        avg_wins = [0.02, 0.03, 0.05]
        avg_losses = [0.02, 0.03, 0.05]

        results = self.grid_search(
            position_sizes, stop_losses, take_profits,
            win_rates, avg_wins, avg_losses
        )

        # Filter safe parameters
        safe_params = [
            r for r in results
            if r.survival_rate >= min_survival_rate and r.sharpe_ratio >= min_sharpe
        ]

        logger.info(f"Found {len(safe_params)} safe parameter combinations out of {len(results)}")

        return safe_params


# ===================== Visualization =====================

def plot_results(result: MonteCarloResult, output_path: str = None):
    """
    Plot Monte Carlo simulation results

    Args:
        result: MonteCarloResult
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Monte Carlo Simulation Results ({result.num_paths} paths)", fontsize=14)

        # 1. Equity curve distribution
        ax1 = axes[0, 0]
        final_equities = [p.final_equity for p in result.paths]
        ax1.hist(final_equities, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(result.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.axvline(result.initial_capital * 0.5, color='orange', linestyle='--', label='Ruin Threshold')
        ax1.set_xlabel('Final Equity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Equity')
        ax1.legend()

        # 2. Sample equity curves
        ax2 = axes[0, 1]
        sample_size = min(100, len(result.paths))
        for path in result.paths[:sample_size]:
            ax2.plot(path.equity_curve, alpha=0.3, linewidth=0.5)
        ax2.axhline(result.initial_capital, color='r', linestyle='--', label='Initial')
        ax2.axhline(result.initial_capital * 0.5, color='orange', linestyle='--', label='Ruin')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Equity')
        ax2.set_title(f'Sample Equity Curves ({sample_size} paths)')
        ax2.legend()

        # 3. Max drawdown distribution
        ax3 = axes[1, 0]
        max_dds = [p.max_drawdown_pct for p in result.paths]
        ax3.hist(max_dds, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax3.axvline(np.mean(max_dds), color='r', linestyle='--', label=f'Mean: {np.mean(max_dds):.2%}')
        ax3.set_xlabel('Max Drawdown')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Max Drawdown')
        ax3.legend()

        # 4. Return distribution
        ax4 = axes[1, 1]
        returns = [(p.final_equity - result.initial_capital) / result.initial_capital for p in result.paths]
        ax4.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax4.axvline(np.mean(returns), color='r', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
        ax4.axvline(0, color='black', linestyle='-')
        ax4.axvline(result.var_95, color='orange', linestyle='--', label=f'VaR 95%: {result.var_95:.2%}')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Returns')
        ax4.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation")
    parser.add_argument("--paths", type=int, default=1000, help="Number of simulation paths")
    parser.add_argument("--days", type=int, default=30, help="Number of days per simulation")
    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--position_size", type=float, default=0.1, help="Position size (% of capital)")
    parser.add_argument("--stop_loss", type=float, default=0.05, help="Stop loss percentage")
    parser.add_argument("--take_profit", type=float, default=0.10, help="Take profit percentage")
    parser.add_argument("--win_rate", type=float, default=0.50, help="Strategy win rate")
    parser.add_argument("--avg_win", type=float, default=0.03, help="Average win percentage")
    parser.add_argument("--avg_loss", type=float, default=0.02, help="Average loss percentage")
    parser.add_argument("--output-dir", type=str, default=None, help='Output directory (for compatibility with run_all_experiments)')

    args = parser.parse_args()

    # Create engine
    engine = MonteCarloEngine(
        initial_capital=args.initial_capital,
        days=args.days,
    )

    if args.optimize:
        # Run parameter optimization
        logger.info("Running parameter optimization...")
        optimizer = ParameterOptimizer(engine, num_paths=200)

        safe_params = optimizer.find_safe_parameters(
            min_survival_rate=0.95,
            min_sharpe=1.0,
        )

        print("\n" + "="*70)
        print("SAFE PARAMETER SPACE")
        print("="*70)

        if safe_params:
            print(f"\nFound {len(safe_params)} safe parameter combinations:")
            for i, params in enumerate(safe_params[:10]):
                print(f"\n{i+1}. Position Size: {params.position_size:.0%}")
                print(f"   Stop Loss: {params.stop_loss:.0%}")
                print(f"   Take Profit: {params.take_profit:.0%}")
                print(f"   Win Rate: {params.win_rate:.0%}")
                print(f"   Survival Rate: {params.survival_rate:.2%}")
                print(f"   Sharpe Ratio: {params.sharpe_ratio:.3f}")
        else:
            print("No parameter combinations met the safe trading criteria.")

    else:
        # Run single Monte Carlo simulation
        params = ParameterSet(
            position_size=args.position_size,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            win_rate=args.win_rate,
            avg_win=args.avg_win,
            avg_loss=args.avg_loss,
        )

        result = engine.run_simulation(
            params,
            num_paths=args.paths,
        )

        # Print results
        print("\n" + "="*70)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*70)
        print(f"Number of Paths: {result.num_paths}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"\nSurvival Analysis:")
        print(f"  Survival Rate: {result.survival_rate * 100:.2f}%")
        print(f"  Probability of Ruin: {result.probability_of_ruin * 100:.2f}%")
        print(f"\nPerformance Metrics:")
        print(f"  Expected Return: {result.expected_return * 100:.2f}%")
        print(f"  Expected Sharpe: {result.expected_sharpe:.3f}")
        print(f"  Expected Max Drawdown: {result.expected_max_drawdown * 100:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%): {result.var_95 * 100:.2f}%")
        print(f"  CVaR (95%): {result.cvar_95 * 100:.2f}%")
        print(f"\nReturn Percentiles:")
        print(f"  5th Percentile: {result.percentile_5_return * 100:.2f}%")
        print(f"  95th Percentile: {result.percentile_95_return * 100:.2f}%")
        print(f"\nBest Path:")
        print(f"  Final Equity: ${result.best_path.final_equity:,.2f}")
        print(f"  Sharpe: {result.best_path.sharpe_ratio:.3f}")
        print(f"\nWorst Path:")
        print(f"  Final Equity: ${result.worst_path.final_equity:,.2f}")
        print(f"  Max Drawdown: {result.worst_path.max_drawdown_pct * 100:.2f}%")
        print("="*70)

        # Success criteria check
        print("\nSUCCESS CRITERIA:")
        print(f"  Probability of Ruin < 5%: {'PASS' if result.probability_of_ruin < 0.05 else 'FAIL'}")
        print(f"  Expected Return > 0: {'PASS' if result.expected_return > 0 else 'FAIL'}")
        print(f"  Expected Sharpe > 1.0: {'PASS' if result.expected_sharpe > 1.0 else 'FAIL'}")

        # Generate plot
        if args.plot:
            output_path = args.output.replace('.json', '_plot.png') if args.output else None
            plot_results(result, output_path)

    # Save results
    if args.output:
        output_data = {
            "params": {
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit,
                "win_rate": args.win_rate,
            },
            "summary": {
                "survival_rate": result.survival_rate,
                "probability_of_ruin": result.probability_of_ruin,
                "expected_return": result.expected_return,
                "expected_sharpe": result.expected_sharpe,
                "expected_max_drawdown": result.expected_max_drawdown,
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
