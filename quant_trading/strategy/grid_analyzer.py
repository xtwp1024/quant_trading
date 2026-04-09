"""
Grid Trading Analysis Utilities

Provides functions for analyzing grid trading performance, optimal spacing,
profit calculations, and exposure management.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class GridProfitAnalysis:
    """Analysis of profit across grid levels."""
    total_profit: float
    profit_per_buy_level: List[float]
    profit_per_sell_level: List[float]
    avg_profit_per_trade: float
    num_complete_cycles: int
    grid_efficiency: float  # 0-1, how many grids were traded


class GridExposureAnalyzer:
    """Analyze and manage exposure for grid trading."""

    def __init__(
        self,
        grid_prices: List[float],
        position_size: float,
        strategy_type: str = "simple_grid",
    ):
        self.grid_prices = sorted(grid_prices)
        self.position_size = position_size
        self.strategy_type = strategy_type
        self._calculate_exposure()

    def _calculate_exposure(self) -> None:
        """Calculate exposure at each grid level."""
        self.long_exposure: List[float] = []
        self.short_exposure: List[float] = []
        self.net_exposure: List[float] = []

        for i, price in enumerate(self.grid_prices):
            if self.strategy_type == "simple_grid":
                # In simple grid, exposure builds in one direction
                if i < len(self.grid_prices) // 2:
                    self.long_exposure.append(self.position_size)
                    self.short_exposure.append(0.0)
                else:
                    self.long_exposure.append(0.0)
                    self.short_exposure.append(self.position_size)
            elif self.strategy_type == "hedged_grid":
                # In hedged grid, both long and short exposure exist
                long_exp = self.position_size if i < len(self.grid_prices) - 1 else 0.0
                short_exp = self.position_size if i > 0 else 0.0
                self.long_exposure.append(long_exp)
                self.short_exposure.append(short_exp)

            self.net_exposure.append(self.long_exposure[-1] - self.short_exposure[-1])

    def get_max_long_exposure(self) -> float:
        """Get maximum long exposure across all levels."""
        return max(self.long_exposure) if self.long_exposure else 0.0

    def get_max_short_exposure(self) -> float:
        """Get maximum short exposure across all levels."""
        return max(self.short_exposure) if self.short_exposure else 0.0

    def get_total_exposure(self, current_price: float) -> Dict[str, float]:
        """Calculate total exposure at current price."""
        # Find which grid levels are in-the-money
        long_in_money = sum(1 for p in self.grid_prices if p <= current_price)
        short_in_money = sum(1 for p in self.grid_prices if p >= current_price)

        return {
            "long_exposure": long_in_money * self.position_size,
            "short_exposure": short_in_money * self.position_size,
            "net_exposure": (long_in_money - short_in_money) * self.position_size,
        }

    def get_exposure_at_price(self, price: float) -> Dict[str, float]:
        """Get exposure at a specific price level."""
        if price < min(self.grid_prices):
            return {"long": self.position_size, "short": 0.0, "net": self.position_size}
        elif price > max(self.grid_prices):
            return {"long": 0.0, "short": self.position_size, "net": -self.position_size}

        for i in range(len(self.grid_prices) - 1):
            if self.grid_prices[i] <= price < self.grid_prices[i + 1]:
                return {
                    "long": self.long_exposure[i],
                    "short": self.short_exposure[i],
                    "net": self.net_exposure[i],
                }
        return {"long": 0.0, "short": 0.0, "net": 0.0}


class GridSpacingOptimizer:
    """Analyze and optimize grid spacing for different market conditions."""

    @staticmethod
    def calculate_optimal_spacing(
        price_range: Tuple[float, float],
        num_grids: int,
        volatility: Optional[float] = None,
        spacing_type: str = "arithmetic",
    ) -> List[float]:
        """
        Calculate optimal grid spacing.

        Args:
            price_range: (min_price, max_price)
            num_grids: Number of grids
            volatility: Optional volatility measure (for adaptive spacing)
            spacing_type: 'arithmetic' or 'geometric'

        Returns:
            List of grid prices
        """
        bottom, top = price_range

        if spacing_type == "arithmetic":
            return np.linspace(bottom, top, num_grids).tolist()

        elif spacing_type == "geometric":
            ratio = (top / bottom) ** (1 / (num_grids - 1))
            grids = []
            current = bottom
            for _ in range(num_grids):
                grids.append(current)
                current *= ratio
            return grids

        return []

    @staticmethod
    def analyze_spacing_efficiency(
        grids: List[float],
        price_distribution: pd.Series,
    ) -> Dict[str, float]:
        """
        Analyze how efficiently grid spacing matches price distribution.

        Args:
            grids: List of grid prices
            price_distribution: Series of historical prices

        Returns:
            Dictionary with efficiency metrics
        """
        if len(grids) < 2:
            return {"efficiency": 0.0, "avg_grid_spacing": 0.0, "price_coverage": 0.0}

        # Calculate grid spacings
        spacings = np.diff(grids)

        if len(spacings) == 0:
            return {"efficiency": 0.0, "avg_grid_spacing": 0.0, "price_coverage": 0.0}

        # Calculate metrics
        avg_spacing = np.mean(spacings)
        spacing_variance = np.var(spacings)

        # How much of the price distribution falls within grid range
        min_price, max_price = price_distribution.min(), price_distribution.max()
        grid_min, grid_max = min(grids), max(grids)

        in_range = ((price_distribution >= grid_min) & (price_distribution <= grid_max)).mean()

        # Efficiency is higher when spacings are uniform and cover the distribution well
        spacing_uniformity = 1.0 / (1.0 + spacing_variance / (avg_spacing ** 2 + 1e-10))
        efficiency = spacing_uniformity * in_range

        return {
            "efficiency": efficiency,
            "avg_grid_spacing": avg_spacing,
            "spacing_variance": spacing_variance,
            "price_coverage": in_range,
            "grid_range": grid_max - grid_min,
        }


class GridProfitCalculator:
    """Calculate expected profit for grid trading strategies."""

    @staticmethod
    def calculate_grid_profit(
        grid_prices: List[float],
        position_size: float,
        trading_fee: float,
        side: str = "buy",
    ) -> Tuple[float, List[float]]:
        """
        Calculate profit for a round-trip trade across grid levels.

        Args:
            grid_prices: Sorted list of grid prices
            position_size: Size of each grid order
            trading_fee: Fee as decimal (e.g., 0.001 for 0.1%)
            side: 'buy' or 'sell' (direction of first trade)

        Returns:
            Tuple of (total_profit, profit_per_level)
        """
        profit_per_level = []
        total_profit = 0.0

        for i in range(len(grid_prices) - 1):
            buy_price = grid_prices[i]
            sell_price = grid_prices[i + 1]

            if side == "buy":
                buy_cost = buy_price * position_size
                sell_revenue = sell_price * position_size
            else:
                buy_cost = sell_price * position_size
                sell_revenue = buy_price * position_size

            gross_profit = sell_revenue - buy_cost
            fees = (buy_cost + sell_revenue) * trading_fee
            net_profit = gross_profit - fees

            profit_per_level.append(net_profit)
            total_profit += net_profit

        return total_profit, profit_per_level

    @staticmethod
    def calculate_hedged_grid_profit(
        grid_prices: List[float],
        position_size: float,
        trading_fee: float,
    ) -> Tuple[float, float]:
        """
        Calculate profit for hedged grid strategy.

        Args:
            grid_prices: List of grid prices
            position_size: Size of each grid order
            trading_fee: Fee as decimal

        Returns:
            Tuple of (total_profit, max_exposure)
        """
        if len(grid_prices) < 3:
            return 0.0, 0.0

        # In hedged grid, each level has both long and short exposure
        # Profit comes from the spread between adjacent levels
        total_profit = 0.0
        max_exposure = 0.0

        for i in range(len(grid_prices) - 1):
            price_diff = grid_prices[i + 1] - grid_prices[i]
            # Long position profit from price increase
            long_profit = price_diff * position_size
            # Short position profit from price decrease
            short_profit = price_diff * position_size

            # Net profit is sum minus fees
            fees = 2 * position_size * (grid_prices[i + 1] + grid_prices[i]) * trading_fee
            net_profit = long_profit + short_profit - fees

            total_profit += net_profit
            max_exposure = max(max_exposure, position_size * grid_prices[i + 1])

        return total_profit, max_exposure

    @staticmethod
    def estimate_max_drawdown(
        grid_prices: List[float],
        position_size: float,
        initial_balance: float,
        price_decline_percent: float,
    ) -> float:
        """
        Estimate maximum drawdown for a given price decline.

        Args:
            grid_prices: List of grid prices
            position_size: Size per grid
            initial_balance: Initial capital
            price_decline_percent: Maximum price decline to analyze

        Returns:
            Estimated max drawdown as decimal
        """
        if len(grid_prices) < 2:
            return 0.0

        # Find how many buy orders would be in profit vs loss
        # during a price decline
        price_range = grid_prices[-1] - grid_prices[0]
        decline_amount = price_range * price_decline_percent

        # Estimate based on grid spacing
        avg_spacing = np.mean(np.diff(grid_prices))
        grids_crossed = decline_amount / avg_spacing if avg_spacing > 0 else 0

        # Each crossed grid results in a losing sell (price went down)
        loss_per_grid = avg_spacing * position_size
        estimated_loss = grids_crossed * loss_per_grid

        return min(estimated_loss / initial_balance, 1.0)


class GridAnalyzer:
    """
    Comprehensive grid trading analyzer combining all analysis utilities.
    """

    def __init__(
        self,
        grid_prices: List[float],
        position_size: float,
        trading_fee: float,
        strategy_type: str = "simple_grid",
        spacing_type: str = "arithmetic",
    ):
        self.grid_prices = sorted(grid_prices)
        self.position_size = position_size
        self.trading_fee = trading_fee
        self.strategy_type = strategy_type
        self.spacing_type = spacing_type

        self.exposure_analyzer = GridExposureAnalyzer(
            grid_prices, position_size, strategy_type
        )
        self.spacing_optimizer = GridSpacingOptimizer()
        self.profit_calculator = GridProfitCalculator()

    def full_analysis(
        self,
        price_history: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive grid analysis.

        Args:
            price_history: Optional historical prices for distribution analysis

        Returns:
            Dictionary with all analysis results
        """
        # Calculate expected profit
        if self.strategy_type == "simple_grid":
            expected_profit, profit_per_level = self.profit_calculator.calculate_grid_profit(
                self.grid_prices, self.position_size, self.trading_fee
            )
        else:
            expected_profit, max_exposure = self.profit_calculator.calculate_hedged_grid_profit(
                self.grid_prices, self.position_size, self.trading_fee
            )
            profit_per_level = []

        # Analyze spacing efficiency
        spacing_analysis = {}
        if price_history is not None:
            spacing_analysis = self.spacing_optimizer.analyze_spacing_efficiency(
                self.grid_prices, price_history
            )

        # Get exposure summary
        max_long = self.exposure_analyzer.get_max_long_exposure()
        max_short = self.exposure_analyzer.get_max_short_exposure()

        return {
            "grid_info": {
                "num_grids": len(self.grid_prices),
                "price_range": (min(self.grid_prices), max(self.grid_prices)),
                "avg_spacing": float(np.mean(np.diff(self.grid_prices))) if len(self.grid_prices) > 1 else 0.0,
                "strategy_type": self.strategy_type,
                "spacing_type": self.spacing_type,
            },
            "expected_profit": expected_profit,
            "profit_per_level": profit_per_level,
            "exposure": {
                "max_long_exposure": max_long,
                "max_short_exposure": max_short,
                "max_net_exposure": abs(max_long - max_short),
            },
            "spacing_analysis": spacing_analysis,
        }

    def what_if_scenario(
        self,
        price_range: Tuple[float, float],
        num_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run what-if scenarios for different price ranges.

        Args:
            price_range: (min_price, max_price) to simulate
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with scenario analysis
        """
        bottom, top = price_range
        price_points = np.linspace(bottom, top, num_simulations)

        results = []
        for price in price_points:
            exposure = self.exposure_analyzer.get_exposure_at_price(price)
            results.append({
                "price": price,
                **exposure,
            })

        return {
            "scenario_count": num_simulations,
            "price_range": price_range,
            "results": results,
        }

    def compare_spacing_types(
        self,
        price_range: Tuple[float, float],
        num_grids: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare arithmetic vs geometric spacing.

        Returns:
            Comparison of both spacing types
        """
        arith_grids = self.spacing_optimizer.calculate_optimal_spacing(
            price_range, num_grids, spacing_type="arithmetic"
        )
        geo_grids = self.spacing_optimizer.calculate_optimal_spacing(
            price_range, num_grids, spacing_type="geometric"
        )

        arith_profit, _ = self.profit_calculator.calculate_grid_profit(
            arith_grids, self.position_size, self.trading_fee
        )
        geo_profit, _ = self.profit_calculator.calculate_grid_profit(
            geo_grids, self.position_size, self.trading_fee
        )

        return {
            "arithmetic": {
                "expected_profit": arith_profit,
                "avg_spacing": float(np.mean(np.diff(arith_grids))),
            },
            "geometric": {
                "expected_profit": geo_profit,
                "avg_spacing": float(np.mean(np.diff(geo_grids))),
            },
            "recommendation": "geometric" if geo_profit > arith_profit else "arithmetic",
        }


def analyze_backtest_results(
    data: pd.DataFrame,
    initial_balance: float,
) -> Dict[str, Any]:
    """
    Analyze grid trading backtest results.

    Args:
        data: DataFrame with 'account_value' column
        initial_balance: Starting balance

    Returns:
        Dictionary with performance metrics
    """
    if "account_value" not in data.columns:
        return {"error": "No account_value column in data"}

    account_values = data["account_value"].dropna()
    if len(account_values) == 0:
        return {"error": "No valid account values"}

    # Calculate returns
    returns = account_values.pct_change().dropna()

    # Basic metrics
    initial = account_values.iloc[0]
    final = account_values.iloc[-1]
    roi = (final - initial) / initial * 100

    # Drawdown analysis
    peak = account_values.expanding(min_periods=1).max()
    drawdown = (peak - account_values) / peak * 100
    max_drawdown = drawdown.max()
    max_drawdown_idx = drawdown.idxmax()

    # Runup analysis
    trough = account_values.expanding(min_periods=1).min()
    runup = (account_values - trough) / trough * 100
    max_runup = runup.max()

    # Time metrics
    time_in_profit = (account_values > initial).mean() * 100

    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

    # Sharpe-like ratio (simplified, no risk-free rate)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    return {
        "roi_percent": roi,
        "final_value": final,
        "max_drawdown_percent": max_drawdown,
        "max_drawdown_date": str(max_drawdown_idx),
        "max_runup_percent": max_runup,
        "time_in_profit_percent": time_in_profit,
        "annualized_volatility_percent": volatility,
        "sharpe_ratio": sharpe,
        "total_return": final - initial,
        "peak_value": peak.iloc[-1],
        "min_value": trough.iloc[-1],
    }
