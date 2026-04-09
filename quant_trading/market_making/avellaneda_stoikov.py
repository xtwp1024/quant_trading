"""
Avellaneda-Stoikov Market Making Implementation
================================================

Implements the classic 2008 paper:
  Avellaneda, Marco, and Sasha Stoikov.
  "High-frequency trading in a limit order book."
  Quantitative Finance 8.3 (2008): 217-224.

Two strategies are provided:
  1. Inventory-based strategy  -- adjusts quotes based on current inventory
  2. Symmetric strategy        -- fixed symmetric spread around mid-price

Key Formulas (from the paper)
-----------------------------

Reservation price:
    r(S, t) = S - q * gamma * sigma^2 * (T - t)

Optimal spread (per side):
    spread = gamma * sigma^2 * (T - t) + (2 / gamma) * log(1 + gamma / k)

where:
    S      = current mid-price
    q      = current inventory (shares)
    T      = time to expiry (in same units as dt)
    sigma  = volatility of the underlying (per unit time)
    gamma  = inventory risk aversion parameter
    k      = order arrival rate sensitivity (1/kappa in the paper notation)
             kappa = 1/k  -- rate at which the spread is "consumed"

The full round-trip spread is twice this per-side spread.
The parameter A controls the base order arrival rate:
    P(hit ask) = A * exp(-k * ask_spread) * dt
    P(hit bid) = A * exp(-k * bid_spread) * dt

Usage
-----
    from quant_trading.market_making.avellaneda_stoikov import avellaneda_stoikov_sim

    results = avellaneda_stoikov_sim(
        S0=100,
        T=1,
        sigma=2,
        k=1.5,
        gamma=0.1,
        q0=0,
        A=140,
        dt=0.005,
        num_simulations=1000,
    )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AvellanedaResult:
    """Result container for a single simulation run."""
    pnl_inventory: float
    pnl_symmetric: float
    final_inventory_inv: float
    final_inventory_sym: float
    mid_prices: np.ndarray      # shape (num_steps + 1,)
    ask_prices_inv: np.ndarray  # shape (num_steps + 1,)
    bid_prices_inv: np.ndarray  # shape (num_steps + 1,)
    inventories_inv: np.ndarray # shape (num_simulations, num_steps + 1)
    inventories_sym: np.ndarray


@dataclass
class SummaryStats:
    """Aggregated statistics across many simulations."""
    pnl_inv_mean: float
    pnl_inv_std: float
    pnl_sym_mean: float
    pnl_sym_std: float
    final_inv_inv_mean: float
    final_inv_inv_std: float
    final_inv_sym_mean: float
    final_inv_sym_std: float


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def avellaneda_stoikov_sim(
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    k: float = 1.5,
    gamma: float = 0.1,
    q0: float = 0.0,
    A: float = 140.0,
    dt: float = 0.005,
    num_simulations: int = 1000,
    seed: int | None = 123,
    track_prices: bool = False,
) -> AvellanedaResult | SummaryStats:
    """
    Run the Avellaneda-Stoikov market-making simulation.

    Parameters
    ----------
    S0 : float
        Initial mid-price of the asset.
    T : float
        Total time horizon (must be in same time units as dt).
    sigma : float
        Volatility of the asset (per unit time, i.e. dW ~ N(0, sigma^2 * dt)).
    k : float
        Order arrival sensitivity. Higher k means orders arrive slower
        for a given spread. In the paper notation kappa = 1/k.
    gamma : float
        Inventory risk aversion. Higher gamma -> tighter inventory control
        -> narrower spreads when inventory is non-zero.
    q0 : float
        Initial inventory (number of shares). Usually 0.
    A : float
        Base order arrival rate (Poisson intensity at zero spread).
    dt : float
        Time step size.
    num_simulations : int
        Number of independent Monte Carlo paths to simulate.
    seed : int | None
        Random seed for reproducibility.
    track_prices : bool
        If True, also track and return mid-price and quote trajectories.
        Default False (faster for large num_simulations).

    Returns
    -------
    SummaryStats (default) or AvellanedaResult (if track_prices=True)
        Default: SummaryStats with mean/std of P&L and final inventory
                 for both strategies across all simulations.

    Notes
    -----
    The simulation models a market maker who posts limit orders on both
    sides of the order book. When an order is filled:
      - Ask fill : inventory decreases by 1, P&L += (S + ask_spread)
      - Bid fill : inventory increases by 1, P&L -= (S - bid_spread)
    At expiry the remaining inventory is liquidated at the mid-price S.

    Paper results (gamma=0.1, 1000 sims):
      Inventory strategy : profit ~65.0, std ~6.3
      Symmetric strategy  : profit ~69.0, std ~13.7
    """
    if seed is not None:
        np.random.seed(seed)

    num_steps = int(T / dt)

    # Pre-compute the average symmetric spread (fixed for all symmetric runs)
    #   sym_spread = gamma * sigma^2 * (T - t) + (2/gamma) * log(1 + gamma/k)
    #   averaged over all time steps
    sym_spread_total = 0.0
    for i in range(num_steps):
        tau = T - i * dt
        sym_spread_total += gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
    av_sym_spread = sym_spread_total / num_steps
    sym_trade_prob = A * np.exp(-k * av_sym_spread / 2.0) * dt

    # Storage
    pnls_inv = np.empty(num_simulations)
    pnls_sym = np.empty(num_simulations)
    final_inv_inv = np.empty(num_simulations)
    final_inv_sym = np.empty(num_simulations)

    if track_prices:
        mid_prices_arr = np.empty((num_simulations, num_steps + 1))
        ask_prices_inv_arr = np.empty((num_simulations, num_steps + 1))
        bid_prices_inv_arr = np.empty((num_simulations, num_steps + 1))
        inventories_inv_arr = np.empty((num_simulations, num_steps + 1))
        inventories_sym_arr = np.empty((num_simulations, num_steps + 1))
    else:
        mid_prices_arr = ask_prices_inv_arr = bid_prices_inv_arr = None
        inventories_inv_arr = inventories_sym_arr = None

    for sim in range(num_simulations):
        # ----- Brownian motion price path -----
        white_noise = sigma * np.sqrt(dt) * np.random.choice(
            [1, -1], size=num_steps
        )
        price_process = S0 + np.cumsum(white_noise)
        price_process = np.insert(price_process, 0, S0)   # shape (num_steps + 1,)

        # ----- Per-sim inventories and P&Ls -----
        q_inv = q0
        q_sym = q0
        pnl_inv = 0.0
        pnl_sym = 0.0

        inv_track_inv = np.empty(num_steps + 1)
        inv_track_sym = np.empty(num_steps + 1)
        inv_track_inv[0] = q_inv
        inv_track_sym[0] = q_sym

        if track_prices:
            mid_prices_arr[sim, 0] = S0
            ask_prices_inv_arr[sim, 0] = np.nan
            bid_prices_inv_arr[sim, 0] = np.nan

        # ----- Time steps -----
        for step, S in enumerate(price_process[:-1]):   # exclude last (terminal) step
            tau = T - step * dt

            # ---- Inventory-based strategy ----
            # Reservation price: r(S, t) = S - q * gamma * sigma^2 * (T - t)
            reservation_price = S - q_inv * gamma * sigma**2 * tau

            # Per-side spread
            per_side_spread = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
            per_side_spread *= 0.5

            # Asymmetric spread around reservation price
            if reservation_price >= S:
                ask_spread = per_side_spread + (reservation_price - S)
                bid_spread = per_side_spread - (reservation_price - S)
            else:
                ask_spread = per_side_spread - (S - reservation_price)
                bid_spread = per_side_spread + (S - reservation_price)

            ask_prob = A * np.exp(-k * ask_spread) * dt
            bid_prob = A * np.exp(-k * bid_spread) * dt
            ask_prob = float(np.clip(ask_prob, 0.0, 1.0))
            bid_prob = float(np.clip(bid_prob, 0.0, 1.0))

            ask_hit_inv = np.random.random() < ask_prob
            bid_hit_inv = np.random.random() < bid_prob

            if ask_hit_inv:
                q_inv -= 1.0
                pnl_inv += S + ask_spread
            if bid_hit_inv:
                q_inv += 1.0
                pnl_inv -= S - bid_spread

            inv_track_inv[step + 1] = q_inv

            if track_prices:
                ask_prices_inv_arr[sim, step + 1] = S + ask_spread
                bid_prices_inv_arr[sim, step + 1] = S - bid_spread
                mid_prices_arr[sim, step + 1] = S

            # ---- Symmetric strategy ----
            ask_hit_sym = np.random.random() < sym_trade_prob
            bid_hit_sym = np.random.random() < sym_trade_prob

            if ask_hit_sym:
                q_sym -= 1.0
                pnl_sym += S + av_sym_spread / 2.0
            if bid_hit_sym:
                q_sym += 1.0
                pnl_sym -= S - av_sym_spread / 2.0

            inv_track_sym[step + 1] = q_sym

        # ----- Terminal liquidation at mid-price -----
        terminal_S = price_process[-1]
        pnl_inv += q_inv * terminal_S
        pnl_sym += q_sym * terminal_S

        pnls_inv[sim] = pnl_inv
        pnls_sym[sim] = pnl_sym
        final_inv_inv[sim] = q_inv
        final_inv_sym[sim] = q_sym

        if track_prices:
            inventories_inv_arr[sim, :] = inv_track_inv
            inventories_sym_arr[sim, :] = inv_track_sym

    if track_prices:
        return AvellanedaResult(
            pnl_inventory=float(pnls_inv.mean()),
            pnl_symmetric=float(pnls_sym.mean()),
            final_inventory_inv=float(final_inv_inv.mean()),
            final_inventory_sym=float(final_inv_sym.mean()),
            mid_prices=mid_prices_arr,
            ask_prices_inv=ask_prices_inv_arr,
            bid_prices_inv=bid_prices_inv_arr,
            inventories_inv=inventories_inv_arr,
            inventories_sym=inventories_sym_arr,
        )

    return SummaryStats(
        pnl_inv_mean=float(pnls_inv.mean()),
        pnl_inv_std=float(pnls_inv.std()),
        pnl_sym_mean=float(pnls_sym.mean()),
        pnl_sym_std=float(pnls_sym.std()),
        final_inv_inv_mean=float(final_inv_inv.mean()),
        final_inv_inv_std=float(final_inv_inv.std()),
        final_inv_sym_mean=float(final_inv_sym.mean()),
        final_inv_sym_std=float(final_inv_sym.std()),
    )


# ---------------------------------------------------------------------------
# Convenience single-run result
# ---------------------------------------------------------------------------

def simulate_single_path(
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    k: float = 1.5,
    gamma: float = 0.1,
    q0: float = 0.0,
    A: float = 140.0,
    dt: float = 0.005,
    seed: int | None = None,
) -> dict:
    """
    Run a single simulation path and return detailed per-step results.

    Returns a dict with keys:
        'pnl_inv', 'pnl_sym', 'final_inv_inv', 'final_inv_sym',
        'mid_prices', 'ask_prices_inv', 'bid_prices_inv',
        'inventories_inv', 'inventories_sym'
    """
    if seed is not None:
        np.random.seed(seed)

    result = avellaneda_stoikov_sim(
        S0=S0, T=T, sigma=sigma, k=k, gamma=gamma,
        q0=q0, A=A, dt=dt,
        num_simulations=1,
        track_prices=True,
        seed=seed,
    )
    return {
        'pnl_inv':         result.pnl_inventory,
        'pnl_sym':         result.pnl_symmetric,
        'final_inv_inv':   result.final_inventory_inv,
        'final_inv_sym':   result.final_inventory_sym,
        'mid_prices':      result.mid_prices[0],
        'ask_prices_inv':  result.ask_prices_inv[0],
        'bid_prices_inv':  result.bid_prices_inv[0],
        'inventories_inv': result.inventories_inv[0],
        'inventories_sym': result.inventories_sym[0],
    }


# ---------------------------------------------------------------------------
# Formula helpers (standalone reference)
# ---------------------------------------------------------------------------

def reservation_price(S: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    """
    Compute the reservation price at mid-price S, inventory q,
    time-to-expiry tau = T - t.

        r(S, q, tau) = S - q * gamma * sigma^2 * tau
    """
    return S - q * gamma * sigma**2 * tau


def optimal_spread(
    gamma: float,
    sigma: float,
    tau: float,
    k: float,
) -> float:
    """
    Compute the optimal per-side (half) spread at time-to-expiry tau.

        spread(tau) = 0.5 * [gamma * sigma^2 * tau + (2/gamma) * log(1 + gamma/k)]

    The full round-trip spread is 2 * spread(tau).
    """
    return 0.5 * (gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k))


def order_arrival_probability(A: float, k: float, spread: float, dt: float) -> float:
    """
    Probability of an order arriving and hitting a quote with given spread
    over time dt:

        P(hit) = A * exp(-k * spread) * dt
    """
    return A * np.exp(-k * spread) * dt


# ---------------------------------------------------------------------------
# AvellanedaStoikov class (OOP interface per user spec)
# ---------------------------------------------------------------------------

class AvellanedaStoikov:
    """Classic Avellaneda-Stoikov market-making algorithm (2008).

    Paper: Avellaneda, M. & Stoikov, S. (2008)
        "A High-Frequency Trader's Perspective on Order Placement"

    This class provides an object-oriented interface to compute:
        - Reservation price (inventory-adjusted midprice)
        - Optimal spread
        - Bid/Ask quotes
        - Expected PnL

    Attributes:
        gamma (float): Risk aversion parameter. Default 0.1.
        kappa (float): Order arrival rate sensitivity. Default 1.0.
            Note: paper uses kappa = 1/k, here kappa = k directly.
        sigma (float): Asset volatility. Default 0.01.
        T (float): Time to maturity (same unit as t). Default 1.0.
        spread (float): Fixed spread offset added to optimal spread. Default 0.0.
        midprice (float): Reference midprice for quotes. Default 100.0.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.0,
        sigma: float = 0.01,
        T: float = 1.0,
        spread: float = 0.0,
        midprice: float = 100.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma (risk aversion) must be positive")
        if kappa <= 0:
            raise ValueError("kappa (arrival rate) must be positive")
        if sigma < 0:
            raise ValueError("sigma (volatility) must be non-negative")
        if T <= 0:
            raise ValueError("T (time to maturity) must be positive")

        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
        self.spread = spread
        self.midprice = midprice

        # Pre-computed constants
        self._eta = gamma * sigma**2  # γσ²
        # log(1 + 2γ/κ) / γ  (half-spread constant from Eq. 7)
        self._log_term = (
            np.log(1.0 + 2.0 * gamma / kappa) / gamma if gamma > 0 else 0.0
        )

    def compute_reservation_price(self, inventory: float, t: float) -> float:
        """Compute the inventory-adjusted reservation price.

        Formula (Avellaneda-Stoikov 2008, Eq. 3):
            r(t, q) = S(t) - q · γ · σ² · (T - t)

        Args:
            inventory (float): Current inventory (positive = long, negative = short)
            t (float): Current time (0 ≤ t ≤ T)

        Returns:
            float: Reservation price
        """
        return self.midprice - inventory * self._eta * (self.T - t)

    def compute_optimal_spread(self, t: float, inventory: float) -> float:
        """Compute the optimal full round-trip spread.

        Formula (Avellaneda-Stoikov 2008, Eq. 7):
            s*(t) = γσ²(T-t) + (2/γ)·ln(1 + 2γ/κ)

        Note: the explicit spread formula does NOT depend on q,
        only on time-to-maturity and model parameters.

        Args:
            t (float): Current time

        Returns:
            float: Full (round-trip) optimal spread
        """
        half_spread = self._eta * (self.T - t) / 2.0 + self._log_term
        return max(half_spread * 2.0 + self.spread, 0.0)

    def compute_bid_ask(self, t: float, inventory: float) -> tuple[float, float]:
        """Return (bid, ask) quotes around the reservation price.

        Args:
            t (float): Current time
            inventory (float): Current inventory

        Returns:
            tuple[float, float]: (bid_price, ask_price)
        """
        reservation = self.compute_reservation_price(inventory, t)
        half_spread = self.compute_optimal_spread(t, inventory) / 2.0
        bid = reservation - half_spread
        ask = reservation + half_spread
        return bid, ask

    def compute_expected_pnl(
        self,
        inventory_path: np.ndarray,
        prices: np.ndarray,
        initial_wealth: float = 0.0,
    ) -> float:
        """Compute expected terminal PnL given inventory and price paths.

        Under the Avellaneda-Stoikov model the expected terminal wealth is:
            E[W_T] = W_0 - γσ² · ∫₀ᵀ q(t)² dt + q_T · S_T

        Args:
            inventory_path (np.ndarray): Inventory trajectory, shape (N,)
            prices (np.ndarray): Midprice trajectory, shape (N,)
            initial_wealth (float): Starting wealth W_0. Default 0.0

        Returns:
            float: Expected terminal PnL (includes liquidation of remaining inventory)
        """
        if inventory_path.shape != prices.shape:
            raise ValueError("inventory_path and prices must have the same shape")

        n_steps = len(inventory_path)
        dt = self.T / n_steps if n_steps > 0 else 0.0

        # Inventory risk penalty: γσ² · Σ q² · Δt
        inventory_penalty = (
            self.gamma * self.sigma**2 * np.sum(inventory_path**2) * dt
        )

        terminal_inventory = inventory_path[-1]
        final_price = prices[-1] if len(prices) > 0 else self.midprice
        liquidation = terminal_inventory * final_price

        return initial_wealth - inventory_penalty + liquidation

    def compute_cara_utility(
        self,
        inventory_path: np.ndarray,
        prices: np.ndarray,
        initial_wealth: float = 0.0,
    ) -> float:
        """Compute CARA (Constant Absolute Risk Aversion) utility.

        CARA utility: U(W) = -exp(-γ · W)

        Under Gaussian terminal wealth assumption:
            E[U(W_T)] = -exp(-γ·E[W_T] + γ²·Var[W_T]/2)

        Args:
            inventory_path (np.ndarray): Inventory trajectory
            prices (np.ndarray): Price trajectory
            initial_wealth (float): Starting wealth

        Returns:
            float: Expected CARA utility (negative value)
        """
        expected_w = self.compute_expected_pnl(inventory_path, prices, initial_wealth)
        n_steps = len(inventory_path)
        dt = self.T / n_steps if n_steps > 0 else 0.0

        # Var[W_T] ≈ (γσ)² · ∫ q(t)² dt
        var_term = (
            (self.gamma * self.sigma) ** 2 * np.sum(inventory_path**2) * dt
        )

        utility = -np.exp(
            -self.gamma * expected_w + 0.5 * var_term
        )
        return utility


# ---------------------------------------------------------------------------
# Main / test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Quick sanity check reproducing paper results
    print("Running Avellaneda-Stoikov simulation (1000 paths, gamma=0.1)...")
    stats = avellaneda_stoikov_sim(
        S0=100, T=1, sigma=2, k=1.5, gamma=0.1,
        q0=0, A=140, dt=0.005, num_simulations=1000, seed=123
    )
    print(f"  Inventory strategy  -- profit: {stats.pnl_inv_mean:.1f}  std: {stats.pnl_inv_std:.1f}")
    print(f"  Symmetric strategy   -- profit: {stats.pnl_sym_mean:.1f}  std: {stats.pnl_sym_std:.1f}")
    print(f"  Final inv (inv str)  -- mean: {stats.final_inv_inv_mean:.3f}  std: {stats.final_inv_inv_std:.3f}")
    print(f"  Final inv (sym str)  -- mean: {stats.final_inv_sym_mean:.3f}  std: {stats.final_inv_sym_std:.3f}")

    # Test AvellanedaStoikov class
    print("\nTesting AvellanedaStoikov class...")
    as_cls = AvellanedaStoikov(gamma=0.1, kappa=1.0, sigma=0.01, T=1.0)
    bid, ask = as_cls.compute_bid_ask(t=0.5, inventory=10)
    print(f"  t=0.5, inventory=10 → bid={bid:.4f}, ask={ask:.4f}")
    r = as_cls.compute_reservation_price(inventory=10, t=0.5)
    print(f"  reservation_price={r:.4f}")
    s = as_cls.compute_optimal_spread(t=0.5, inventory=10)
    print(f"  optimal_spread={s:.4f}")
