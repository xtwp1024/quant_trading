"""
Avellaneda-Stoikov Market-Making Optimizer
===========================================

经典做市商理论实现 (Avellaneda & Stoikov, 2008)
High-frequency trading in a limit order book

核心最优化问题:
    maximize E[utility of terminal wealth]
    subject to: inventory dynamics, order arrival Poisson processes

关键公式:
    - reservation_price = S_t - q * gamma * sigma^2 * (T-t)
    - optimal_spread = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
    - bid = reservation_price - spread/2
    - ask = reservation_price + spread/2

参数说明:
    gamma: 风险厌恶系数 (risk aversion)
    sigma: 波动率 (volatility of midprice)
    kappa: 订单到达灵敏度 (order arrival sensitivity, k in paper)
    A: 基准订单到达率 (base order arrival rate at zero spread)

Usage:
    from quant_trading.market_making.as_optimizer import ASOptimizer

    # High-volatility market (e.g., crypto)
    opt = ASOptimizer(gamma=0.1, sigma=2.0, kappa=1.5, A=140.0, T=1.0, max_inventory=100)
    bid, ask = opt.optimal_quotes(S=100.0, q=0, t=0.0)
    result = opt.run_simulation(S0=100.0, n_steps=390, n_simulations=1000)

    # Low-volatility market (e.g., stablecoin)
    opt_low = ASOptimizer(gamma=0.1, sigma=1e-3, kappa=1.5, A=140.0, T=1.0, max_inventory=100)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["ASOptimizer"]


class ASOptimizer:
    """Avellaneda-Stoikov market-making optimal control.

    Optimization problem:
        maximize E[utility of terminal wealth]
        subject to: inventory dynamics, order arrival Poisson processes

    Key formulas (Avellaneda & Stoikov, 2008):
        - reservation_price = S_t - q * gamma * sigma^2 * (T-t)
        - optimal_spread = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
        - bid = reservation_price - spread/2
        - ask = reservation_price + spread/2

    Parameters
    ----------
    gamma : float, default 0.1
        Risk aversion parameter (gamma > 0). Higher values lead to tighter
        inventory control and narrower spreads when inventory is non-zero.
    sigma : float, default 1e-3
        Volatility of midprice. For dt in seconds, sigma should be per-second vol.
        For daily vol ~2%, sigma ≈ 2/sqrt(252*6.5*3600) ≈ 0.001.
    kappa : float, default 1.5
        Order arrival sensitivity (k in paper). Controls how fast order arrival
        rate decays with spread: lambda = A * exp(-kappa * spread).
        Typical values: 0.5 to 3.0.
    T : float, default 1.0
        Time horizon (in same units as t, e.g. fraction of trading day).
    max_inventory : int, default 100
        Maximum absolute inventory position allowed.
    A : float, default 140.0
        Base order arrival rate at zero spread. Higher A -> more trading.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 1e-3,
        kappa: float = 1e-3,
        T: float = 1.0,
        max_inventory: int = 100,
        A: float = 140.0,
    ):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if A <= 0:
            raise ValueError("A must be positive")

        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.max_inventory = max_inventory
        self.A = A  # Base order arrival rate at zero spread

    # ------------------------------------------------------------------
    # Core formulas
    # ------------------------------------------------------------------

    def reservation_price(self, S: float, q: int, t: float) -> float:
        """Compute reservation price at time t.

        The reservation price is the fair price at which the market maker
        would be indifferent to holding inventory q. It adjusts the mid-price
        downward when long (positive q) and upward when short (negative q).

            r(S, q, t) = S - q * gamma * sigma^2 * (T - t)

        Parameters
        ----------
        S : float
            Current mid-price of the asset.
        q : int
            Current inventory position (shares). Positive = long.
        t : float
            Current time (in same units as T, e.g. fraction of trading day).

        Returns
        -------
        float
            Reservation price.
        """
        tau = self.T - t
        if tau < 0:
            tau = 0.0
        return S - q * self.gamma * (self.sigma**2) * tau

    def optimal_spread(self, t: float, q: int = 0) -> float:
        """Compute optimal bid-ask spread at time t.

        The full round-trip spread from the Avellaneda-Stoikov paper (2008):

            full_spread = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

        The half-spread (per side) is:

            half_spread = gamma * sigma^2 * (T-t) / 2 + (1/gamma) * ln(1 + gamma/kappa)

        where:
        - gamma = risk aversion
        - sigma = volatility
        - kappa = order arrival sensitivity (k in the paper)
        - tau = T - t = time to expiry

        Parameters
        ----------
        t : float
            Current time.
        q : int, default 0
            Current inventory (affects the asymmetric part of the spread).

        Returns
        -------
        float
            Optimal full round-trip spread.
        """
        tau = self.T - t
        if tau < 0:
            tau = 0.0

        # Baseline spread: (2/gamma) * ln(1 + gamma/kappa)
        # This is the minimum spread due to risk aversion
        baseline = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)

        # Time-decaying spread component: gamma * sigma^2 * tau
        # This widens as we approach expiry (more inventory risk)
        time_component = self.gamma * (self.sigma**2) * tau

        full_spread = baseline + time_component
        return float(full_spread)

    def optimal_quotes(self, S: float, q: int, t: float) -> tuple[float, float]:
        """Compute optimal bid and ask quotes.

        The market maker posts:
            ask = reservation_price + spread/2
            bid = reservation_price - spread/2

        This results in asymmetric quotes when inventory is non-zero:
        - Long inventory -> lower bid (more aggressive buy), higher ask (less aggressive sell)
        - Short inventory -> higher bid (more aggressive buy), lower ask (less aggressive sell)

        Parameters
        ----------
        S : float
            Current mid-price.
        q : int
            Current inventory.
        t : float
            Current time.

        Returns
        -------
        tuple[float, float]
            (bid_price, ask_price) optimal quotes.
        """
        r = self.reservation_price(S, q, t)
        spread = self.optimal_spread(t, q)
        half_spread = spread / 2.0

        bid = r - half_spread
        ask = r + half_spread

        # Hard bounds: bid < ask sanity check
        if bid >= ask:
            mid = r
            bid = mid - 0.5 * spread
            ask = mid + 0.5 * spread

        return float(bid), float(ask)

    # ------------------------------------------------------------------
    # Order arrival model
    # ------------------------------------------------------------------

    def order_arrival_probability(
        self, spread: float, dt: float, side: str = "ask"
    ) -> float:
        """Compute probability of order arrival using exponential model.

        The order arrival rate follows a Poisson process with intensity:

            lambda(spread) = A * exp(-kappa * spread)

        where A is the base arrival rate at zero spread and kappa (k) is
        the spread sensitivity parameter. Higher spread -> lower arrival rate.

        Parameters
        ----------
        spread : float
            Distance from mid-price to quote (bid_spread or ask_spread).
        dt : float
            Time step size.
        side : str, default "ask"
            "ask" or "bid" (for documentation only).

        Returns
        -------
        float
            Probability of an order hitting the quote in time dt.
        """
        lam = self.A * np.exp(-self.kappa * spread)
        prob = 1.0 - np.exp(-lam * dt)
        return float(np.clip(prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        S0: float,
        n_steps: int = 390,
        n_simulations: int = 1000,
        initial_inventory: int = 0,
        seed: Optional[int] = None,
    ) -> dict:
        """Run Monte Carlo simulation to validate the strategy.

        Simulates n_simulations independent price paths using geometric
        Brownian motion and tracks P&L distribution.

        Parameters
        ----------
        S0 : float
            Initial mid-price.
        n_steps : int, default 390
            Number of time steps (390 = seconds per trading minute * minutes).
            Default 390 approximates a trading day with 1-step per minute.
        n_simulations : int, default 1000
            Number of Monte Carlo paths.
        initial_inventory : int, default 0
            Starting inventory position.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pnl': array of P&L for each simulation
            - 'final_inventory': array of final inventory positions
            - 'mean_pnl': mean P&L across simulations
            - 'std_pnl': standard deviation of P&L
            - 'sharpe_ratio': annualized Sharpe ratio (assuming 252 trading days)
            - 'max_drawdown': maximum drawdown across all paths
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.T / n_steps  # time step size
        sqrt_dt = np.sqrt(dt)

        # Storage for results
        pnls = np.empty(n_simulations)
        final_inventories = np.empty(n_simulations, dtype=int)
        max_drawdowns = np.empty(n_simulations)

        for sim in range(n_simulations):
            # Generate Brownian motion price path
            z = np.random.standard_normal(n_steps)
            price_path = S0 + self.sigma * sqrt_dt * np.cumsum(z)
            price_path = np.insert(price_path, 0, S0)  # shape (n_steps + 1,)

            inventory = initial_inventory
            cash_flow = 0.0
            equity_curve = np.empty(n_steps + 1)
            equity_curve[0] = 0.0

            for step in range(n_steps):
                S = price_path[step]
                t = step * dt

                # Get optimal quotes
                bid, ask = self.optimal_quotes(S, inventory, t)
                bid_spread = S - bid
                ask_spread = ask - S

                # Compute order arrival probabilities
                prob_bid = self.order_arrival_probability(bid_spread, dt, "bid")
                prob_ask = self.order_arrival_probability(ask_spread, dt, "ask")

                # Sample order arrivals using Poisson approximation
                # Using Bernoulli approximation for small probabilities
                bid_hit = np.random.random() < prob_bid
                ask_hit = np.random.random() < prob_ask

                # Inventory limits
                if inventory >= self.max_inventory:
                    bid_hit = False  # Can't buy more if at max long
                if inventory <= -self.max_inventory:
                    ask_hit = False  # Can't sell more if at max short

                # Execute trades
                if bid_hit:
                    inventory += 1
                    cash_flow -= bid  # Pay bid to buy (spend cash, gain inventory)

                if ask_hit:
                    inventory -= 1
                    cash_flow += ask  # Receive ask to sell (gain cash, lose inventory)

                # Mark-to-market equity at each step
                equity_curve[step + 1] = cash_flow + inventory * S

            # Terminal liquidation at final mid-price
            terminal_S = price_path[-1]
            final_pnl = cash_flow + inventory * terminal_S
            terminal_equity = final_pnl

            pnls[sim] = terminal_equity
            final_inventories[sim] = inventory

            # Compute max drawdown for this path
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = running_max - equity_curve
            max_drawdowns[sim] = np.max(drawdowns)

        # Compute statistics
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        sharpe = float(mean_pnl / std_pnl * np.sqrt(n_steps)) if std_pnl > 0 else 0.0

        return {
            "pnl": pnls,
            "final_inventory": final_inventories,
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(np.mean(max_drawdowns)),
            "pnl_percentile_5": float(np.percentile(pnls, 5)),
            "pnl_percentile_95": float(np.percentile(pnls, 95)),
        }

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        param_name: str,
        param_values: list[float],
        S0: float = 100.0,
        n_steps: int = 390,
        n_simulations: int = 500,
    ) -> dict:
        """Analyze strategy sensitivity to a parameter.

        Parameters
        ----------
        param_name : str
            Parameter to vary: 'gamma', 'sigma', 'kappa', 'max_inventory'.
        param_values : list[float]
            List of values to test.
        S0 : float, default 100.0
            Initial price.
        n_steps : int, default 390
            Number of time steps.
        n_simulations : int, default 500
            Simulations per parameter value.

        Returns
        -------
        dict
            Dictionary with 'param_values', 'mean_pnls', 'std_pnls', 'sharpes'.
        """
        original = getattr(self, param_name)
        results = {
            "param_values": param_values,
            "mean_pnls": [],
            "std_pnls": [],
            "sharpes": [],
        }

        try:
            for val in param_values:
                setattr(self, param_name, val)
                sim_result = self.run_simulation(
                    S0=S0, n_steps=n_steps, n_simulations=n_simulations
                )
                results["mean_pnls"].append(sim_result["mean_pnl"])
                results["std_pnls"].append(sim_result["std_pnl"])
                results["sharpes"].append(sim_result["sharpe_ratio"])
        finally:
            setattr(self, param_name, original)

        return results


# ---------------------------------------------------------------------------
# Standalone formula helpers
# ---------------------------------------------------------------------------


def reservation_price_formula(
    S: float, q: float, gamma: float, sigma: float, tau: float
) -> float:
    """Compute reservation price using the analytical formula.

        r(S, q, tau) = S - q * gamma * sigma^2 * tau

    Parameters
    ----------
    S : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.
    tau : float
        Time to expiry (T - t).

    Returns
    -------
    float
        Reservation price.
    """
    return S - q * gamma * (sigma**2) * tau


def optimal_spread_formula(
    gamma: float, sigma: float, kappa: float, tau: float
) -> float:
    """Compute optimal full spread using the paper formula.

    From Avellaneda-Stoikov (2008):

        full_spread = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)

    Parameters
    ----------
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.
    kappa : float
        Order arrival sensitivity (k in paper).
    tau : float
        Time to expiry (T - t).

    Returns
    -------
    float
        Optimal full spread.
    """
    if tau < 0:
        tau = 0.0
    term1 = gamma * (sigma**2) * tau
    term2 = (2.0 / gamma) * np.log(1.0 + gamma / kappa)
    return term1 + term2


if __name__ == "__main__":
    # Quick sanity check
    print("ASOptimizer quick test")
    print("-" * 40)

    # Default parameters (matching source implementations)
    opt = ASOptimizer(gamma=0.1, sigma=2.0, kappa=1.5, A=140.0, T=1.0, max_inventory=100)

    # Test formulas
    S, q, t = 100.0, 0, 0.0
    r = opt.reservation_price(S, q, t)
    spread = opt.optimal_spread(t, q)
    bid, ask = opt.optimal_quotes(S, q, t)
    print(f"  S={S}, q={q}, t={t}")
    print(f"  reservation_price = {r:.4f}")
    print(f"  optimal_spread    = {spread:.4f}")
    print(f"  bid={bid:.4f}, ask={ask:.4f}")

    # Test with non-zero inventory
    q = 50
    r = opt.reservation_price(S, q, t)
    spread = opt.optimal_spread(t, q)
    bid, ask = opt.optimal_quotes(S, q, t)
    print(f"\n  With q={q}:")
    print(f"  reservation_price = {r:.4f}")
    print(f"  bid={bid:.4f}, ask={ask:.4f}")

    # Run simulation
    print("\nRunning simulation (n=1000)...")
    result = opt.run_simulation(S0=100.0, n_steps=390, n_simulations=1000, seed=42)
    print(f"  mean_pnl      = {result['mean_pnl']:.4f}")
    print(f"  std_pnl       = {result['std_pnl']:.4f}")
    print(f"  sharpe_ratio  = {result['sharpe_ratio']:.4f}")
    print(f"  max_drawdown  = {result['max_drawdown']:.4f}")
    print(f"  pnl_5th pct    = {result['pnl_percentile_5']:.4f}")
    print(f"  pnl_95th pct   = {result['pnl_percentile_95']:.4f}")
