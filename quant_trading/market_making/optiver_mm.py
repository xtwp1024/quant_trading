"""
Optiver Market Making Algorithm.

竞赛第4名方案核心逻辑:
1. 使用Black-Scholes定价期权
2. 动态调整价差 (根据库存/波动率/时间)
3. Delta对冲
4. 库存风险管理

Optiver-style market-making algorithm implementing:
- Black-Scholes option pricing
- Dynamic bid/ask spread (Avellaneda-Stoikov style)
- Delta hedging
- Inventory risk management

Pure NumPy + scipy.stats.norm implementation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["OptiverMarketMaker"]


_norm_cdf = stats.norm(0, 1).cdf
_norm_pdf = stats.norm(0, 1).pdf


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 parameter for Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 parameter for Black-Scholes."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


class OptiverMarketMaker:
    """
    Optiver风格做市商算法.

    竞赛第4名方案核心逻辑:
    1. 使用Black-Scholes定价期权
    2. 动态调整价差 (根据库存/波动率/时间)
    3. Delta对冲
    4. 库存管理

    Optiver-style market-making algorithm based on:
    - Black-Scholes theoretical option pricing
    - Avellaneda-Stoikov dynamic spread model
    - Delta-neutral hedging
    - Inventory risk management with mean-reversion

    Parameters
    ----------
    sigma : float
        Implied volatility for Black-Scholes (default 0.2).
    lambda_ass : float
        Order arrival rate (default 1e-6).
    gamma : float
        Inventory risk aversion coefficient (default 1e-7).
    kappa : float
        Mean reversion speed for inventory (default 2.5).

    Example
    -------
    >>> mm = OptiverMarketMaker(sigma=0.2, lambda_ass=1e-6, gamma=1e-7, kappa=2.5)
    >>> bid, ask = mm.compute_bid_ask(S=100.0, t=0.5, q=10, T=1.0)
    >>> delta = mm.compute_delta_hedge(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)
    >>> spread = mm.compute_spread(sigma=0.2, t=0.5, q=10)
    """

    def __init__(
        self,
        sigma: float = 0.2,
        lambda_ass: float = 1e-6,
        gamma: float = 1e-7,
        kappa: float = 2.5,
    ) -> None:
        """
        Initialize the Optiver market maker.

        Parameters
        ----------
        sigma : float
            Implied volatility (annualized).
        lambda_ass : float
            Order arrival rate (probability per unit time).
        gamma : float
            Inventory risk aversion (penalty coefficient).
        kappa : float
            Mean reversion speed for inventory.
        """
        self.sigma = sigma
        self.lambda_ass = lambda_ass
        self.gamma = gamma
        self.kappa = kappa

        # Internal inventory state / 库存状态
        self._inventory: float = 0.0
        self._inventory_history: list[float] = []

        # Risk-free rate / 无风险利率
        self.r: float = 0.03

    # ------------------------------------------------------------------
    # Black-Scholes helpers / Black-Scholes辅助函数
    # ------------------------------------------------------------------

    def _bs_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option value."""
        return S * _norm_cdf(_d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * _norm_cdf(
            _d2(S, K, T, r, sigma)
        )

    def _bs_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option value."""
        return np.exp(-r * T) * K * _norm_cdf(-_d2(S, K, T, r, sigma)) - S * _norm_cdf(
            -_d1(S, K, T, r, sigma)
        )

    def _bs_call_delta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call delta."""
        return _norm_cdf(_d1(S, K, T, r, sigma))

    def _bs_put_delta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put delta (same as call delta - 1)."""
        return self._bs_call_delta(S, K, T, r, sigma) - 1.0

    # ------------------------------------------------------------------
    # Core API (as specified)
    # ------------------------------------------------------------------

    def compute_bid_ask(
        self, S: float, t: float, q: float, T: float
    ) -> tuple[float, float]:
        """
        计算最优买卖报价 / Compute optimal bid/ask quotes.

        Uses the Avellaneda-Stoikov framework with inventory adjustments.
        The fair value is computed via Black-Scholes; the spread widens
        with volatility, time pressure, and inventory.

        Parameters
        ----------
        S : float
            Current underlying price.
        t : float
            Current time (fraction of day, e.g. 0.5 = halfway through day).
        q : float
            Current inventory position.
        T : float
            Time to expiry (in years).

        Returns
        -------
        tuple[float, float]
            (bid_price, ask_price).
        """
        fair = self._bs_call(S, S, T, self.r, self.sigma)

        spread = self.compute_spread(self.sigma, t, q)

        half = spread / 2.0
        bid = fair - half
        ask = fair + half

        return float(bid), float(ask)

    def compute_delta_hedge(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """
        计算Delta对冲数量 / Compute Delta hedge quantity.

        Returns the number of shares of the underlying needed to make
        the portfolio delta-neutral.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry (years).
        r : float
            Risk-free rate.
        sigma : float
            Volatility.

        Returns
        -------
        float
            Delta hedge quantity (shares to trade in underlying).
        """
        delta = self._bs_call_delta(S, K, T, r, sigma)
        return float(-delta * self._inventory)

    def compute_spread(self, sigma: float, t: float, q: float) -> float:
        """
        计算最优价差 / Compute optimal bid-ask spread.

        Spread formula (Avellaneda-Stoikov inspired):
            spread = 2 * gamma * q * sigma^2 * (T - t)
                   + 2 * log(1 + gamma / kappa)

        The first term captures inventory risk (widens when you are
        long/short or near expiry); the second term is the base spread
        that compensates for adverse selection.

        Parameters
        ----------
        sigma : float
            Volatility.
        t : float
            Current time within the trading day (0..1).
        q : float
            Current inventory.

        Returns
        -------
        float
            Optimal bid-ask spread.
        """
        term1 = 2.0 * self.gamma * q * sigma**2 * max(T_expired := 1.0 - t, 0.0)
        term2 = 2.0 * np.log(1.0 + self.gamma / self.kappa)
        spread = term1 + term2
        return float(max(spread, 1e-8))

    def update_inventory(self, trade_price: float, trade_side: str) -> None:
        """
        更新库存 / Update inventory after a trade.

        Parameters
        ----------
        trade_price : float
            Price at which the trade executed.
        trade_side : str
            'bid' (bought) or 'ask' (sold).
        """
        if trade_side == "bid":
            self._inventory += 1.0
        elif trade_side == "ask":
            self._inventory -= 1.0
        else:
            raise ValueError(f"trade_side must be 'bid' or 'ask', got {trade_side!r}")

        self._inventory_history.append(self._inventory)

    def run_simulation(self, price_paths: np.ndarray) -> dict:
        """
        蒙特卡洛模拟评估 / Monte Carlo simulation evaluation.

        Simulates the market-maker P&L over multiple price paths.

        Parameters
        ----------
        price_paths : np.ndarray
            2D array of shape (n_paths, n_steps) with underlying prices.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'pnl': total P&L across all paths
            - 'avg_pnl': average P&L per path
            - 'sharpe': Sharpe ratio (approx)
            - 'inventory': final inventory
            - 'max_inventory': peak absolute inventory
        """
        n_paths, n_steps = price_paths.shape
        pnls = np.zeros(n_paths)
        final_inventory = np.zeros(n_paths)
        max_inventory = np.zeros(n_paths)

        dt = 1.0 / n_steps  # time step (1 day = 1 unit)

        for p in range(n_paths):
            inv = 0.0
            peak = 0.0
            cash = 0.0

            for i in range(n_steps - 1):
                S = price_paths[p, i]
                t = i * dt

                # Compute bid/ask
                bid, ask = self.compute_bid_ask(S=S, t=t, q=inv, T=1.0)

                # Simulate trade with probability lambda_ass
                if np.random.random() < self.lambda_ass:
                    # Trade at bid (market sell) or ask (market buy)
                    if inv > 0:
                        cash -= ask   # bought back
                        inv -= 1.0
                    elif inv < 0:
                        cash += bid   # covered short
                        inv += 1.0

                # Mark-to-market: PnL from inventory
                mid = S
                mtm = inv * mid

                peak = max(peak, abs(inv))

            # Final liquidation
            final_S = price_paths[p, -1]
            final_pnl = cash + inv * final_S
            pnls[p] = final_pnl
            final_inventory[p] = inv
            max_inventory[p] = peak

        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0

        return {
            "pnl": float(np.sum(pnls)),
            "avg_pnl": avg_pnl,
            "sharpe": float(sharpe),
            "inventory": float(self._inventory),
            "max_inventory": float(np.max(max_inventory)),
        }
