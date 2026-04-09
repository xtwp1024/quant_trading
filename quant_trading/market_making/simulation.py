"""
Simulation Utilities for Market Making
=======================================

Provides:
  - Brownian motion price path generation
  - P&L tracking helpers
  - Result reporting / pretty-printing

This module is intentionally dependency-free (pure numpy) so it can
be imported without pulling in heavy frameworks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Price path generation
# ---------------------------------------------------------------------------

def brownian_motion(
    S0: float,
    sigma: float,
    T: float,
    dt: float,
    num_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate one or more Brownian motion price paths using binomial (random walk)
    approximation of geometric Brownian motion.

    Each step: S_{t+dt} = S_t * exp( -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z )
             approx as: S_t + sigma * sqrt(dt) * Z  (arithmetic form, used in the paper)

    Parameters
    ----------
    S0 : float
        Initial price level.
    sigma : float
        Per-step volatility.
    T : float
        Total time horizon.
    dt : float
        Time step size.
    num_paths : int
        Number of independent paths to generate.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    paths : np.ndarray
        Shape (num_paths, num_steps + 1) where num_steps = int(T / dt).
        paths[i, 0] == S0 for all i.
    """
    if seed is not None:
        np.random.seed(seed)

    num_steps = int(T / dt)
    # Standard normal draws
    Z = np.random.randn(num_paths, num_steps)
    # Arithmetic random walk: S_{t+1} = S_t + sigma * sqrt(dt) * Z
    increments = sigma * np.sqrt(dt) * Z
    # Build cumulative sum + initial value
    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 + np.cumsum(increments, axis=1)
    return paths


def geometric_brownian_motion(
    S0: float,
    sigma: float,
    T: float,
    dt: float,
    num_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate one or more geometric Brownian motion price paths.

    S_{t+dt} = S_t * exp( -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z )

    Parameters
    ----------
    S0, sigma, T, dt, num_paths, seed
        Same as for brownian_motion.

    Returns
    -------
    paths : np.ndarray
        Shape (num_paths, num_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    num_steps = int(T / dt)
    Z = np.random.randn(num_paths, num_steps)
    log_increments = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = np.exp(log_paths)
    return paths


# ---------------------------------------------------------------------------
# P&L tracking helpers
# ---------------------------------------------------------------------------

@dataclass
class PnLTracker:
    """
    Tracks P&L and inventory over a single simulation run.

    Usage
    -----
        tracker = PnLTracker(q0=0)
        for step, S in enumerate(mid_prices[:-1]):
            # ... decide ask_spread, bid_spread, hit probabilities ...
            if ask_hit: tracker.fill_ask(S, ask_price)
            if bid_hit: tracker.fill_bid(S, bid_price)
            tracker.record_inventory()
        tracker.liquidate(terminal_S)
        print(tracker.pnl)
    """
    initial_inventory: float
    inventory: float
    pnl: float
    inventory_history: list[float]
    pnl_history: list[float]

    def __init__(self, q0: float = 0.0) -> None:
        self.initial_inventory = q0
        self.inventory = q0
        self.pnl = 0.0
        self.inventory_history = [q0]
        self.pnl_history = [0.0]

    def fill_ask(self, mid_price: float, ask_price: float) -> None:
        """Record an ask (sell) fill at ask_price when mid is mid_price."""
        self.inventory -= 1.0
        self.pnl += ask_price

    def fill_bid(self, mid_price: float, bid_price: float) -> None:
        """Record a bid (buy) fill at bid_price when mid is mid_price."""
        self.inventory += 1.0
        self.pnl -= bid_price

    def record_inventory(self) -> None:
        """Snapshot current inventory (call once per time step)."""
        self.inventory_history.append(self.inventory)
        self.pnl_history.append(self.pnl)

    def liquidate(self, terminal_price: float) -> float:
        """
        Liquidate remaining inventory at terminal_price and add to P&L.
        Returns final P&L.
        """
        self.pnl += self.inventory * terminal_price
        self.inventory_history.append(self.inventory)
        self.pnl_history.append(self.pnl)
        return self.pnl


def compute_pnl_from_trades(
    trades: list[tuple[str, float, float]],
    terminal_price: float,
    q0: float = 0.0,
) -> float:
    """
    Compute P&L from a list of trades.

    Parameters
    ----------
    trades : list of (side, mid_price, quote_price)
        side is 'ask' or 'bid'.
        For ask fills: P&L += quote_price, inventory -= 1
        For bid fills: P&L -= quote_price, inventory += 1
    terminal_price : float
        Price at which remaining inventory is liquidated.
    q0 : float
        Initial inventory.

    Returns
    -------
    float : final P&L
    """
    inventory = q0
    pnl = 0.0
    for side, mid_price, quote_price in trades:
        if side == 'ask':
            inventory -= 1.0
            pnl += quote_price
        elif side == 'bid':
            inventory += 1.0
            pnl -= quote_price
        else:
            raise ValueError(f"Unknown side: {side!r}")
    pnl += inventory * terminal_price
    return pnl


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def print_simulation_report(
    name: str,
    pnl_arr: np.ndarray,
    final_inventory_arr: np.ndarray,
    num_sims: int | None = None,
) -> None:
    """
    Pretty-print a simulation report.

    Parameters
    ----------
    name : str
        Strategy / simulation name.
    pnl_arr : np.ndarray
        1-D array of final P&L values (one per simulation).
    final_inventory_arr : np.ndarray
        1-D array of final inventory values.
    num_sims : int | None
        Number of simulations (defaults to len(pnl_arr)).
    """
    if num_sims is None:
        num_sims = len(pnl_arr)
    pnl_arr = np.asarray(pnl_arr)
    final_inventory_arr = np.asarray(final_inventory_arr)

    print(f"\n{'='*55}")
    print(f"  Simulation Report: {name}  ({num_sims} paths)")
    print(f"{'='*55}")
    print(f"  P&L   mean : {pnl_arr.mean():>10.4f}")
    print(f"  P&L   std  : {pnl_arr.std():>10.4f}")
    print(f"  P&L   min  : {pnl_arr.min():>10.4f}")
    print(f"  P&L   max  : {pnl_arr.max():>10.4f}")
    print(f"  Inv   mean : {final_inventory_arr.mean():>10.4f}")
    print(f"  Inv   std  : {final_inventory_arr.std():>10.4f}")
    print(f"{'='*55}\n")


def compare_strategies(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Print a side-by-side comparison table of multiple strategies.

    Parameters
    ----------
    results : dict[str, tuple[pnl_arr, final_inv_arr]]
        Maps strategy name to (P&L array, final inventory array).
    """
    print(f"\n{'Strategy':<20} {'P&L mean':>12} {'P&L std':>10} {'Inv mean':>12} {'Inv std':>10}")
    print("-" * 68)
    for name, (pnl, inv) in results.items():
        print(f"  {name:<18} {pnl.mean():>12.4f} {pnl.std():>10.4f} {inv.mean():>12.4f} {inv.std():>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Histogram data helper
# ---------------------------------------------------------------------------

def pnl_histogram(
    pnl_arr: np.ndarray,
    bin_width: float = 4.0,
    x_range: tuple[float, float] = (-50.0, 150.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram data for P&L distribution.

    Parameters
    ----------
    pnl_arr : np.ndarray
        1-D array of P&L values.
    bin_width : float
        Width of each histogram bin.
    x_range : tuple (xmin, xmax)

    Returns
    -------
    counts : np.ndarray
    bin_edges : np.ndarray
    """
    bins = np.arange(x_range[0], x_range[1] + bin_width, bin_width)
    counts, bin_edges = np.histogram(pnl_arr, bins=bins)
    return counts, bin_edges
