"""
Market-Making Inventory Risk Management Model
==============================================

库存风险管理模型 for market makers.

This module provides tools for:
- Position limit calculation based on VaR
- Inventory holding cost computation
- Risk-adjusted PnL calculation
- Inventory drawdown monitoring

Usage:
    from quant_trading.market_making.inventory_model import InventoryModel

    model = InventoryModel(max_position=100, max_loss=10000.0)
    limit = model.compute_position_limit(price=100.0, var_threshold=0.05)
    cost = model.compute_inventory_cost(inventory, prices)
    adj_pnl = model.compute_risk_adjusted_pnl(pnl, inventory, var)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["InventoryModel"]


class InventoryModel:
    """Market-maker inventory risk management model.

    Provides risk management tools for market makers handling inventory risk:
    - Position sizing based on Value-at-Risk (VaR)
    - Holding cost calculation for overnight positions
    - Risk-adjusted PnL with VaR penalty

    Parameters
    ----------
    max_position : int, default 100
        Maximum absolute inventory position allowed.
    max_loss : float, default 10000.0
        Maximum dollar loss allowed per position or per day.
    risk_free_rate : float, default 0.0
        Risk-free rate for cost-of-carry calculations (annualized).
    """

    def __init__(
        self,
        max_position: int = 100,
        max_loss: float = 10000.0,
        risk_free_rate: float = 0.0,
    ):
        self.max_position = max_position
        self.max_loss = max_loss
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Position limits
    # ------------------------------------------------------------------

    def compute_position_limit(
        self,
        price: float,
        var_threshold: float = 0.05,
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> int:
        """Compute maximum position size based on Value-at-Risk (VaR).

        VaR at confidence level c for a long position over holding_period:
            VaR = position_size * price * z_c * sigma * sqrt(holding_period)

        Solving for position_size:
            position_size = max_loss / (price * z_c * sigma * sqrt(holding_period))

        where z_c is the standard normal quantile for (1 - var_threshold).

        Parameters
        ----------
        price : float
            Current asset price.
        var_threshold : float, default 0.05
            Probability threshold for VaR (5% -> 95% confidence).
            VaR will not be exceeded with probability (1 - var_threshold).
        confidence_level : float, default 0.95
            Confidence level for VaR calculation (0.95 = 95%).
        holding_period : int, default 1
            Holding period in days for VaR calculation.

        Returns
        -------
        int
            Maximum position size (in shares) allowed under VaR constraints.
        """
        # Standard normal quantile for (1 - var_threshold)
        # e.g., var_threshold=0.05 -> z = 1.645 for one-tailed 95% confidence
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - var_threshold)
        except ImportError:
            # Fallback: approximate using error function inverse
            # For 95% confidence (z=1.645), use simple approximation
            z = np.sqrt(2) * self._erfinv(1 - 2 * var_threshold)

        # Assume daily volatility of 1% if not provided
        daily_sigma = 0.01

        # VaR formula: max_loss = position * price * z * sigma * sqrt(T)
        # Solve for position:
        denominator = z * daily_sigma * np.sqrt(holding_period) * price
        if denominator <= 0:
            return self.max_position

        position_limit = self.max_loss / denominator

        # Return integer position, bounded by max_position
        return int(min(position_limit, self.max_position))

    def compute_var(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> np.ndarray:
        """Compute Value-at-Risk for given positions and prices.

        Parameters
        ----------
        positions : np.ndarray
            Position series (in shares).
        prices : np.ndarray
            Price series.
        confidence_level : float, default 0.95
            Confidence level (0.95 = 95%).
        holding_period : int, default 1
            Holding period in days.

        Returns
        -------
        np.ndarray
            VaR series (positive values representing potential loss).
        """
        # Compute returns
        returns = np.diff(prices) / prices[:-1]
        if len(returns) == 0:
            return np.array([0.0])

        # Position P&L changes
        position_values = positions[:-1] * prices[:-1]
        pnl_changes = position_values * returns

        # VaR using historical percentile
        var = np.percentile(pnl_changes, (1 - confidence_level) * 100)
        return np.abs(np.full(len(pnl_changes), var))

    # ------------------------------------------------------------------
    # Inventory costs
    # ------------------------------------------------------------------

    def compute_inventory_cost(
        self,
        inventory: np.ndarray,
        prices: np.ndarray,
        holding_cost_pct: float = 0.0001,
    ) -> float:
        """Compute total inventory holding cost.

        Inventory cost includes:
        - Capital cost: opportunity cost of margin tied up in inventory
        - Storage cost: holding cost as percentage of position value

        Total cost = sum over all periods of:
            inventory_t * price_t * holding_cost_pct * dt

        Parameters
        ----------
        inventory : np.ndarray
            Inventory position series (shares). Positive = long.
        prices : np.ndarray
            Price series (same length as inventory).
        holding_cost_pct : float, default 0.0001
            Holding cost as fraction of position value per period.
            Default 0.01% per period (e.g., per minute or per second).

        Returns
        -------
        float
            Total holding cost across all periods.
        """
        if len(inventory) != len(prices):
            raise ValueError("inventory and prices must have same length")

        # Position value = shares * price
        position_values = np.abs(inventory[:-1]) * prices[:-1]

        # Cost per period
        costs = position_values * holding_cost_pct

        return float(np.sum(costs))

    def compute_funding_cost(
        self,
        inventory: np.ndarray,
        prices: np.ndarray,
        borrow_rate: float = 0.03,
        dt: float = 1.0,
    ) -> float:
        """Compute cost of financing inventory (long/short borrowing cost).

        For short positions: need to borrow shares, pay borrow_rate
        For long positions: opportunity cost of capital at risk_free_rate

        Parameters
        ----------
        inventory : np.ndarray
            Inventory series (positive = long, negative = short).
        prices : np.ndarray
            Price series.
        borrow_rate : float, default 0.03
            Annualized borrow rate for short positions.
        dt : float, default 1.0
            Time step in years (1/252 for daily).

        Returns
        -------
        float
            Total funding cost.
        """
        position_values = inventory[:-1] * prices[:-1]

        # Long positions: opportunity cost at risk-free rate
        # Short positions: borrow cost at borrow_rate
        funding_rate = np.where(
            inventory[:-1] >= 0, self.risk_free_rate, borrow_rate
        )

        costs = position_values * funding_rate * dt
        return float(np.sum(costs))

    # ------------------------------------------------------------------
    # Risk-adjusted PnL
    # ------------------------------------------------------------------

    def compute_risk_adjusted_pnl(
        self,
        pnl: np.ndarray,
        inventory: np.ndarray,
        var: float,
        risk_lambda: float = 0.5,
    ) -> np.ndarray:
        """Compute risk-adjusted PnL with VaR penalty.

        Risk-adjusted PnL = PnL - lambda * VaR

        where lambda is the risk aversion coefficient. This penalizes
        returns that come with high VaR (potential large losses).

        Parameters
        ----------
        pnl : np.ndarray
            Raw PnL series.
        inventory : np.ndarray
            Inventory position series.
        var : float
            Value-at-Risk at the confidence level used for sizing.
        risk_lambda : float, default 0.5
            Risk penalty coefficient. Higher = more conservative.
            Typical values: 0.1 (aggressive) to 1.0 (conservative).

        Returns
        -------
        np.ndarray
            Risk-adjusted PnL series (length min(len(pnl), len(inventory)-1)).
        """
        # Align lengths: pnl and inventory should have same length
        # or pnl can be one shorter (if it represents per-step changes)
        min_len = min(len(pnl), len(inventory) - 1)
        pnl_adj = np.asarray(pnl)[:min_len]
        inv_adj = np.asarray(inventory)[:min_len]

        # VaR penalty per period
        var_penalty = risk_lambda * var

        # Inventory risk multiplier: larger positions get more penalty
        max_inv = max(1, np.max(np.abs(inv_adj)))
        inventory_risk = np.abs(inv_adj) / max_inv

        risk_adjusted = pnl_adj - var_penalty * (1 + inventory_risk)

        return risk_adjusted

    def compute_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Compute Sortino ratio (return / downside deviation).

        Sortino = (mean_return - risk_free_rate) / downside_deviation

        Unlike Sharpe, Sortino only penalizes downside volatility,
        treating upside volatility as beneficial.

        Parameters
        ----------
        returns : np.ndarray
            Return series.
        target_return : float, default 0.0
            Minimum acceptable return (MAR).
        risk_free_rate : float, default 0.0
            Risk-free rate for annualized calculation.

        Returns
        -------
        float
            Sortino ratio (annualized if dt < 1).
        """
        excess = returns - target_return
        downside = excess[excess < 0]

        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0

        downside_std = np.std(downside)
        mean_excess = np.mean(excess)

        return float((mean_excess - risk_free_rate) / downside_std)

    # ------------------------------------------------------------------
    # Inventory monitoring
    # ------------------------------------------------------------------

    def compute_inventory_drawdown(
        self, inventory: np.ndarray, prices: np.ndarray
    ) -> dict:
        """Compute inventory-related drawdowns.

        Parameters
        ----------
        inventory : np.ndarray
            Inventory position series.
        prices : np.ndarray
            Price series.

        Returns
        -------
        dict
            Dictionary containing:
            - 'max_drawdown': maximum drawdown in dollars
            - 'drawdown_duration': duration of longest drawdown
            - 'max_position': maximum absolute position
        """
        position_value = inventory * prices
        running_max = np.maximum.accumulate(position_value)
        drawdowns = running_max - position_value

        max_dd = float(np.max(drawdowns))

        # Find duration of longest drawdown
        in_drawdown = drawdowns > 0
        max_duration = 0
        current_duration = 0
        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return {
            "max_drawdown": max_dd,
            "drawdown_duration": max_duration,
            "max_position": int(np.max(np.abs(inventory))),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _erfinv(x: float) -> float:
        """Inverse error function approximation.

        For x in (0, 1), computes approximate inverse of erf.
        Used as fallback when scipy.stats.norm is unavailable.
        """
        # Approximation using Newton-Raphson
        # Initial guess
        if x <= 0:
            return -np.inf
        if x >= 1:
            return np.inf

        y = np.log(1.0 - x**2)
        y0 = 0.5 * np.log(np.pi) - 0.25 * y
        y = y0 + 0.5 * y
        return y  # rough approximation


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compute_var_normal(
    position: float,
    price: float,
    sigma: float,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """Compute parametric VaR using normal distribution.

    VaR = position * price * z_c * sigma * sqrt(holding_period)

    Parameters
    ----------
    position : float
        Position in shares.
    price : float
        Current price.
    sigma : float
        Volatility per period.
    confidence : float, default 0.95
        Confidence level.
    holding_period : int, default 1
        Holding period multiplier.

    Returns
    -------
    float
        VaR (potential loss).
    """
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
    except ImportError:
        z = np.sqrt(2) * np.arctanh(2 * confidence - 1)

    var = position * price * z * sigma * np.sqrt(holding_period)
    return float(abs(var))


def compute_inventory_sharpe(
    pnl: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: float = 252
) -> dict:
    """Compute Sharpe and related ratios for a PnL series.

    Parameters
    ----------
    pnl : np.ndarray
        PnL series (daily or per-period).
    risk_free_rate : float, default 0.0
        Risk-free rate (annualized).
    periods_per_year : float, default 252
        Number of periods per year (252 for daily data).

    Returns
    -------
    dict
        Dictionary with sharpe, sortino, max_drawdown, calmar.
    """
    returns = pnl  # assuming pnl is already returns or per-period P&L

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    # Sharpe (annualized)
    if std_return > 0:
        sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
        sharpe *= np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside))
        if downside_std > 0:
            sortino = (mean_return - risk_free_rate / periods_per_year) / downside_std
            sortino *= np.sqrt(periods_per_year)
        else:
            sortino = 0.0
    else:
        sortino = 0.0

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns))

    # Calmar: annualized return / max drawdown
    if max_dd > 0:
        calmar = mean_return * periods_per_year / max_dd
    else:
        calmar = 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "mean_return": mean_return,
        "std_return": std_return,
    }


if __name__ == "__main__":
    # Quick sanity check
    print("InventoryModel quick test")
    print("-" * 40)

    model = InventoryModel(max_position=100, max_loss=10000.0)

    # Test position limit
    limit = model.compute_position_limit(price=100.0, var_threshold=0.05)
    print(f"  Position limit @ price=100: {limit} shares")

    limit2 = model.compute_position_limit(price=50.0, var_threshold=0.05)
    print(f"  Position limit @ price=50: {limit2} shares")

    # Test VaR computation
    positions = np.array([0, 10, 20, 30, 20, 10, 0, -10, -20])
    prices = np.array([100, 101, 102, 101, 100, 99, 100, 101, 102])
    var = model.compute_var(positions, prices)
    print(f"  VaR @ 95% confidence: {var[0]:.2f}")

    # Test inventory cost
    inventory = np.array([0, 10, 20, 30, 20, 10, 0])
    prices_arr = np.array([100, 101, 102, 103, 102, 101, 100])
    cost = model.compute_inventory_cost(inventory, prices_arr, holding_cost_pct=0.0001)
    print(f"  Inventory holding cost: {cost:.2f}")

    # Test risk-adjusted PnL
    pnl = np.array([10, 20, 15, 25, 30, 20, 15, 10])
    inventory_arr = np.array([0, 10, 20, 30, 40, 30, 20, 10, 0])
    adj = model.compute_risk_adjusted_pnl(pnl, inventory_arr, var=10.0, risk_lambda=0.5)
    print(f"  Risk-adjusted PnL: {adj}")

    # Test Sharpe-like metrics
    returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.005])
    metrics = compute_inventory_sharpe(returns)
    print(f"  Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.4f}, Calmar: {metrics['calmar']:.2f}")
