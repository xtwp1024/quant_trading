"""
Market-Making Strategy Simulator
==================================

事件驱动蒙特卡洛仿真器 for market-making strategies.

Provides an event-driven backtesting framework for market-making
strategies with support for:
- Multiple price path models (geometric Brownian motion, jump diffusion)
- Configurable order arrival processes
- Inventory and PnL tracking
- Parameter sensitivity analysis
- Visualization of PnL distributions

Usage:
    from quant_trading.market_making.simulator import MarketSimulator
    from quant_trading.market_making.as_optimizer import ASOptimizer

    model = ASOptimizer(gamma=0.1, sigma=1e-3, kappa=1e-3)
    sim = MarketSimulator(model=model, midprice_model='geometric_brownian', n_simulations=1000)
    result = sim.run(S0=100.0, T=1.0, dt=1/390)
    sim.sensitivity_analysis('gamma', [0.05, 0.1, 0.2, 0.5])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Literal

__all__ = ["MarketSimulator"]


class MarketSimulator:
    """Market-making strategy event-driven Monte Carlo simulator.

    Event-driven backtesting framework for market-making strategies.
    Simulates price paths and tracks order fills, inventory, and PnL
    across multiple Monte Carlo paths.

    Parameters
    ----------
    model : ASOptimizer
        The market-making optimization model to simulate.
    midprice_model : str, default 'geometric_brownian'
        Price path model. Options:
        - 'geometric_brownian': Standard GBM (dS = sigma * S * dW)
        - 'brownian': Arithmetic Brownian motion (dS = sigma * dW)
        - 'mean_reverting': Ornstein-Uhlenbeck process
    n_simulations : int, default 1000
        Number of Monte Carlo paths to simulate.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model,
        midprice_model: str = "geometric_brownian",
        n_simulations: int = 1000,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.midprice_model = midprice_model
        self.n_simulations = n_simulations
        self.seed = seed

        # Valid models
        self._valid_models = {
            "geometric_brownian",
            "brownian",
            "mean_reverting",
        }
        if midprice_model not in self._valid_models:
            raise ValueError(
                f"Unknown midprice_model: {midprice_model}. "
                f"Valid options: {self._valid_models}"
            )

    # ------------------------------------------------------------------
    # Price path generation
    # ------------------------------------------------------------------

    def _generate_price_path(
        self,
        S0: float,
        T: float,
        dt: float,
        sigma: Optional[float] = None,
        mu: float = 0.0,
        theta: float = 0.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a single price path using the configured model.

        Parameters
        ----------
        S0 : float
            Initial price.
        T : float
            Time horizon.
        dt : float
            Time step.
        sigma : float, optional
            Volatility (uses model.sigma if not provided).
        mu : float, default 0.0
            Drift (for Brownian and geometric Brownian).
        theta : float, default 0.0
            Mean reversion speed (for mean_reverting model).

        Returns
        -------
        np.ndarray
            Price path of length n_steps + 1.
        """
        if sigma is None:
            sigma = self.model.sigma

        n_steps = int(T / dt)
        sqrt_dt = np.sqrt(dt)

        if seed is not None:
            np.random.seed(seed)

        if self.midprice_model == "geometric_brownian":
            # GBM: dS = mu * S * dt + sigma * S * dW
            # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            z = np.random.standard_normal(n_steps)
            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
            log_prices = np.log(S0) + np.cumsum(log_returns)
            prices = np.exp(log_prices)
            prices = np.insert(prices, 0, S0)

        elif self.midprice_model == "brownian":
            # Arithmetic Brownian: dS = mu * dt + sigma * dW
            z = np.random.standard_normal(n_steps)
            prices = S0 + np.cumsum(mu * dt + sigma * sqrt_dt * z)
            prices = np.insert(prices, 0, S0)

        elif self.midprice_model == "mean_reverting":
            # Ornstein-Uhlenbeck: dS = theta * (mu - S) * dt + sigma * dW
            prices = np.empty(n_steps + 1)
            prices[0] = S0
            S = S0
            for i in range(1, n_steps + 1):
                z = np.random.standard_normal(1)[0]
                S = S + theta * (mu - S) * dt + sigma * sqrt_dt * z
                prices[i] = S

        return prices

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    def run(
        self,
        S0: float,
        T: float = 1.0,
        dt: float = 1 / 390,
        initial_inventory: int = 0,
        track_paths: bool = False,
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation of the market-making strategy.

        Simulates multiple price paths and tracks PnL, inventory,
        and quote evolution for each path.

        Parameters
        ----------
        S0 : float
            Initial mid-price.
        T : float, default 1.0
            Time horizon (in trading days).
        dt : float, default 1/390
            Time step size (1/390 = 1 minute in a trading day).
        initial_inventory : int, default 0
            Starting inventory position.
        track_paths : bool, default False
            If True, track full path data for all simulations.
            Warning: memory intensive for large n_simulations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'simulation': simulation index
            - 'pnl': terminal PnL for each path
            - 'final_inventory': final inventory for each path
            - 'sharpe': per-path Sharpe ratio
            - 'max_drawdown': per-path maximum drawdown
            - 'mean_spread': average spread posted
            - 'fill_rate_bid': bid fill rate
            - 'fill_rate_ask': ask fill rate
            If track_paths=True, includes per-step data.
        """
        n_steps = int(T / dt)
        max_inv = self.model.max_inventory

        # Results storage
        results = {
            "simulation": [],
            "pnl": [],
            "final_inventory": [],
            "sharpe": [],
            "max_drawdown": [],
            "mean_spread": [],
            "fill_rate_bid": [],
            "fill_rate_ask": [],
            "total_bid_fills": [],
            "total_ask_fills": [],
        }

        # Path tracking (optional)
        if track_paths:
            path_data = {
                "simulation": [],
                "step": [],
                "midprice": [],
                "bid": [],
                "ask": [],
                "inventory": [],
                "cash": [],
                "equity": [],
            }

        rng = np.random.default_rng(self.seed)

        for sim_idx in range(self.n_simulations):
            # Generate price path
            if self.seed is not None:
                path_seed = self.seed + sim_idx
            else:
                path_seed = None

            prices = self._generate_price_path(S0, T, dt, seed=path_seed)

            # Initialize tracking variables
            inventory = initial_inventory
            cash = 0.0
            equity_curve = np.zeros(n_steps + 1)
            equity_curve[0] = 0.0

            spreads = []
            bid_fills = 0
            ask_fills = 0
            bid_hits_total = 0
            ask_hits_total = 0

            for step in range(n_steps):
                S = prices[step]
                t = step * dt

                # Get optimal quotes
                bid, ask = self.model.optimal_quotes(S, inventory, t)
                bid_spread = S - bid
                ask_spread = ask - S
                spreads.append(bid_spread + ask_spread)

                # Order arrival probabilities
                prob_bid = self.model.order_arrival_probability(bid_spread, dt, "bid")
                prob_ask = self.model.order_arrival_probability(ask_spread, dt, "ask")

                # Poisson arrivals: use np.random.poisson for arrival counts
                # For simplicity, use Bernoulli approximation
                bid_hit = rng.random() < prob_bid
                ask_hit = rng.random() < prob_ask

                # Inventory constraints
                if inventory >= max_inv:
                    bid_hit = False
                if inventory <= -max_inv:
                    ask_hit = False

                if bid_hit:
                    inventory += 1
                    cash -= bid
                    bid_fills += 1
                if ask_hit:
                    inventory -= 1
                    cash += ask
                    ask_fills += 1

                bid_hits_total += 1 if prob_bid > 0 else 0
                ask_hits_total += 1 if prob_ask > 0 else 0

                # Mark-to-market
                equity_curve[step + 1] = cash + inventory * prices[step + 1]

                # Path tracking
                if track_paths:
                    path_data["simulation"].append(sim_idx)
                    path_data["step"].append(step)
                    path_data["midprice"].append(S)
                    path_data["bid"].append(bid)
                    path_data["ask"].append(ask)
                    path_data["inventory"].append(inventory)
                    path_data["cash"].append(cash)
                    path_data["equity"].append(equity_curve[step + 1])

            # Terminal liquidation
            terminal_S = prices[-1]
            final_pnl = cash + inventory * terminal_S

            # Compute per-path metrics
            returns = np.diff(equity_curve) / (np.abs(equity_curve[:-1]) + 1e-10)
            sharpe = (
                np.mean(returns) / np.std((returns)) * np.sqrt(n_steps)
                if np.std(returns) > 0
                else 0.0
            )

            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = running_max - equity_curve
            max_dd = np.max(drawdowns)

            # Store results
            results["simulation"].append(sim_idx)
            results["pnl"].append(final_pnl)
            results["final_inventory"].append(inventory)
            results["sharpe"].append(sharpe)
            results["max_drawdown"].append(max_dd)
            results["mean_spread"].append(np.mean(spreads) if spreads else 0.0)
            results["fill_rate_bid"].append(
                bid_fills / max(bid_hits_total, 1)
            )
            results["fill_rate_ask"].append(
                ask_fills / max(ask_hits_total, 1)
            )
            results["total_bid_fills"].append(bid_fills)
            results["total_ask_fills"].append(ask_fills)

        df = pd.DataFrame(results)

        if track_paths:
            path_df = pd.DataFrame(path_data)
            df = {"summary": df, "paths": path_df}

        return df

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        param_name: str,
        param_range: list[float],
        S0: float = 100.0,
        T: float = 1.0,
        dt: float = 1 / 390,
        n_simulations: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run parameter sensitivity analysis.

        Varies a single parameter across a range of values and
        computes statistics for each value.

        Parameters
        ----------
        param_name : str
            Name of parameter to vary. Must be an attribute of the model.
        param_range : list[float]
            List of parameter values to test.
        S0 : float, default 100.0
            Initial price.
        T : float, default 1.0
            Time horizon.
        dt : float, default 1/390
            Time step.
        n_simulations : int, optional
            Simulations per parameter value.
            Defaults to self.n_simulations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'param_value': the tested parameter value
            - 'mean_pnl': mean terminal PnL
            - 'std_pnl': std of terminal PnL
            - 'mean_sharpe': mean Sharpe ratio
            - 'mean_max_drawdown': mean max drawdown
            - 'mean_inventory': mean final inventory
            - 'fill_rate': overall fill rate
        """
        if n_simulations is None:
            n_simulations = self.n_simulations

        original_value = getattr(self.model, param_name, None)
        if original_value is None:
            raise ValueError(f"Model has no attribute '{param_name}'")

        sensitivity_results = []

        try:
            for val in param_range:
                setattr(self.model, param_name, val)

                # Temporarily adjust simulator
                original_n_sim = self.n_simulations
                self.n_simulations = n_simulations

                results = self.run(S0=S0, T=T, dt=dt)

                if isinstance(results, dict):
                    df = results["summary"]
                else:
                    df = results

                sensitivity_results.append(
                    {
                        "param_value": val,
                        "mean_pnl": df["pnl"].mean(),
                        "std_pnl": df["pnl"].std(),
                        "mean_sharpe": df["sharpe"].mean(),
                        "mean_max_drawdown": df["max_drawdown"].mean(),
                        "mean_inventory": df["final_inventory"].mean(),
                        "fill_rate": (
                            df["total_bid_fills"].sum() + df["total_ask_fills"].sum()
                        )
                        / (2 * n_simulations * T / dt),
                    }
                )

                self.n_simulations = original_n_sim

        finally:
            setattr(self.model, param_name, original_value)

        return pd.DataFrame(sensitivity_results)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_pnl_distribution(
        self,
        ax=None,
        results: Optional[pd.DataFrame] = None,
        S0: float = 100.0,
        **kwargs,
    ) -> None:
        """Plot histogram of PnL distribution.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes.
        results : pd.DataFrame, optional
            Results DataFrame from run(). If None, runs simulation first.
        S0 : float, default 100.0
            Initial price (used if results is None).
        **kwargs
            Additional arguments passed to ax.hist().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if results is None:
            results = self.run(S0=S0)

        if isinstance(results, dict):
            df = results["summary"]
        else:
            df = results

        pnls = df["pnl"].values

        if ax is None:
            ax = plt.gca()

        defaults = {
            "bins": 50,
            "alpha": 0.7,
            "edgecolor": "black",
            "linewidth": 0.5,
        }
        defaults.update(kwargs)

        ax.hist(pnls, **defaults)

        # Add statistics annotations
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        ax.axvline(mean_pnl, color="red", linestyle="--", label=f"Mean: {mean_pnl:.2f}")
        ax.axvline(
            mean_pnl - std_pnl,
            color="orange",
            linestyle=":",
            label=f"±1 Std: {std_pnl:.2f}",
        )
        ax.axvline(mean_pnl + std_pnl, color="orange", linestyle=":")

        ax.set_xlabel("PnL")
        ax.set_ylabel("Frequency")
        ax.set_title(f"PnL Distribution (n={len(pnls)})")
        ax.legend()

    def plot_equity_curves(
        self,
        ax=None,
        results: Optional[pd.DataFrame] = None,
        S0: float = 100.0,
        n_show: int = 100,
        **kwargs,
    ) -> None:
        """Plot sample equity curves from simulation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        results : pd.DataFrame, optional
            Results with path data. If None, runs with track_paths=True.
        S0 : float, default 100.0
            Initial price.
        n_show : int, default 100
            Number of paths to show (randomly sampled).
        **kwargs
            Additional plot arguments.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if results is None:
            results = self.run(S0=S0, track_paths=True)

        if not isinstance(results, dict) or "paths" not in results:
            raise ValueError("Results must include path data. Run with track_paths=True.")

        paths_df = results["paths"]
        simulations = paths_df["simulation"].unique()
        n_show = min(n_show, len(simulations))

        if ax is None:
            ax = plt.gca()

        # Sample random simulations
        rng = np.random.default_rng(self.seed)
        show_sims = rng.choice(simulations, size=n_show, replace=False)

        for sim in show_sims:
            sim_data = paths_df[paths_df["simulation"] == sim]
            ax.plot(sim_data["step"], sim_data["equity"], alpha=0.3, **kwargs)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Equity")
        ax.set_title(f"Sample Equity Curves (n={n_show})")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(
        self, results: Optional[pd.DataFrame] = None, S0: float = 100.0
    ) -> dict:
        """Compute summary statistics from simulation results.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Results DataFrame from run().
        S0 : float, default 100.0
            Initial price (used if results is None).

        Returns
        -------
        dict
            Dictionary with comprehensive statistics.
        """
        if results is None:
            results = self.run(S0=S0)

        if isinstance(results, dict):
            df = results["summary"]
        else:
            df = results

        pnls = df["pnl"].values

        return {
            "n_simulations": len(pnls),
            "mean_pnl": float(np.mean(pnls)),
            "median_pnl": float(np.median(pnls)),
            "std_pnl": float(np.std(pnls)),
            "min_pnl": float(np.min(pnls)),
            "max_pnl": float(np.max(pnls)),
            "percentile_5": float(np.percentile(pnls, 5)),
            "percentile_25": float(np.percentile(pnls, 25)),
            "percentile_75": float(np.percentile(pnls, 75)),
            "percentile_95": float(np.percentile(pnls, 95)),
            "win_rate": float(np.mean(pnls > 0)),
            "mean_sharpe": float(df["sharpe"].mean()),
            "mean_max_drawdown": float(df["max_drawdown"].mean()),
            "mean_final_inventory": float(df["final_inventory"].mean()),
            "mean_fill_rate_bid": float(df["fill_rate_bid"].mean()),
            "mean_fill_rate_ask": float(df["fill_rate_ask"].mean()),
        }


if __name__ == "__main__":
    # Quick sanity check
    print("MarketSimulator quick test")
    print("-" * 40)

    from quant_trading.market_making.as_optimizer import ASOptimizer

    model = ASOptimizer(gamma=0.1, sigma=1e-3, kappa=1e-3, T=1.0, max_inventory=100)
    sim = MarketSimulator(model=model, midprice_model="geometric_brownian", n_simulations=500, seed=42)

    print("Running simulation (n=500)...")
    results = sim.run(S0=100.0, T=1.0, dt=1 / 390)

    if isinstance(results, dict):
        df = results["summary"]
    else:
        df = results

    stats = sim.summary_stats(results)
    print(f"\nSimulation Summary:")
    print(f"  n_simulations:     {stats['n_simulations']}")
    print(f"  mean_pnl:          {stats['mean_pnl']:.4f}")
    print(f"  std_pnl:           {stats['std_pnl']:.4f}")
    print(f"  min_pnl:           {stats['min_pnl']:.4f}")
    print(f"  max_pnl:           {stats['max_pnl']:.4f}")
    print(f"  win_rate:          {stats['win_rate']:.2%}")
    print(f"  mean_sharpe:       {stats['mean_sharpe']:.4f}")
    print(f"  mean_max_drawdown: {stats['mean_max_drawdown']:.4f}")

    # Sensitivity analysis
    print("\nRunning sensitivity analysis on gamma...")
    sens = sim.sensitivity_analysis("gamma", [0.05, 0.1, 0.2, 0.5], n_simulations=200)
    print(sens.to_string(index=False))
