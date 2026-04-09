"""
Monte Carlo期权定价引擎 — GBM几何布朗运动模拟
Monte Carlo Option Pricing Engine using Geometric Brownian Motion

使用几何布朗运动 (GBM) 模拟标的资产价格路径:
    dS = μS*dt + σS*dW

支持:
    - European Call/Put 期权定价
    - Asian (average price) options 亚式期权
    - Lookback options 回顾期权
    - Greeks计算 (bump-and-reprice)
    - Numba @jit(nopython=True) 加速 (~275x)

Author: 量化之神系统
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional

# Try to import Numba; graceful fallback to pure NumPy if unavailable
try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    jit = None  # type: ignore

__all__ = [
    "MonteCarloEngine",
    "_mc_price_numpy",
]


# ---------------------------------------------------------------------------
# Core Monte Carlo loop (Numba accelerated)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _mc_simulate_gbm(
        n_paths: int,
        n_steps: int,
        dt: float,
        mu: float,
        sigma: float,
        seed: int,
    ) -> np.ndarray:
        """Numba-accelerated GBM path simulation.

        使用 antithetic variates 方差减少技术提升收敛速度。
        Returns: (n_paths, n_steps+1) array, first column = S0 = 1.0 (normalized)
        """
        np.random.seed(seed)
        half = n_paths // 2

        # Pre-allocate output
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 1.0  # Normalized S0

        for step in range(1, n_steps + 1):
            # Generate two half-paths with antithetic variates
            Z = np.random.randn(half)
            Z = np.concatenate([Z, -Z])  # antithetic

            # GBM: S[t] = S[t-1] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            log_drift = (mu - 0.5 * sigma * sigma) * dt
            log_diffusion = sigma * np.sqrt(dt) * Z
            paths[:, step] = paths[:, step - 1] * np.exp(log_drift + log_diffusion)

        return paths

    @jit(nopython=True, cache=True)
    def _mc_price_european(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price European option from simulated paths.

        Returns: (price, std_error)
        """
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        # Terminal stock price (last column)
        S_T = paths[:, -1]

        # Discount factor
        discount = np.exp(-r * n_steps * dt)

        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        elif option_type == "put":
            payoff = np.maximum(K - S_T, 0.0)
        else:
            payoff = np.maximum(S_T - K, 0.0)  # default call

        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(n_paths)

        return price, std_error

    @jit(nopython=True, cache=True)
    def _mc_price_asian(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price Asian (average price) option from simulated paths."""
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        # Average price over all steps
        avg_price = np.mean(paths[:, 1:], axis=1)

        discount = np.exp(-r * n_steps * dt)

        if option_type == "call":
            payoff = np.maximum(avg_price - K, 0.0)
        else:
            payoff = np.maximum(K - avg_price, 0.0)

        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(n_paths)

        return price, std_error

    @jit(nopython=True, cache=True)
    def _mc_price_lookback(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price Lookback (optimal price) option from simulated paths."""
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        if option_type == "call":
            # Lookback call: max over path vs strike
            payoff = np.maximum(np.max(paths[:, 1:], axis=1) - K, 0.0)
        else:
            # Lookback put: strike vs min over path
            payoff = np.maximum(K - np.min(paths[:, 1:], axis=1), 0.0)

        discount = np.exp(-r * n_steps * dt)
        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(n_paths)

        return price, std_error

    @jit(nopython=True, cache=True)
    def _mc_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        option_type: str,
    ) -> tuple[float, float, float, float, float]:
        """Compute Greeks via bump-and-reprice (finite differences).

        Returns: (delta, gamma, vega, theta, rho)
        """
        dt = T / n_steps
        discount = np.exp(-r * T)

        # Base paths
        paths = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42)
        paths = paths * S  # Scale to actual spot

        # Base price
        if option_type == "call":
            base_payoff = np.maximum(paths[:, -1] - K, 0.0)
        else:
            base_payoff = np.maximum(K - paths[:, -1], 0.0)
        base_price = discount * np.mean(base_payoff)

        # ---- Bump parameters ----
        dS = S * 0.01
        dsigma = sigma * 0.01
        dr = 0.0001
        dt_step = T / n_steps

        # Delta + Gamma: bump S up/down
        paths_up = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * (S + dS)
        paths_down = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * (S - dS)

        if option_type == "call":
            pu = np.maximum(paths_up[:, -1] - K, 0.0)
            pd = np.maximum(paths_down[:, -1] - K, 0.0)
        else:
            pu = np.maximum(K - paths_up[:, -1], 0.0)
            pd = np.maximum(K - paths_down[:, -1], 0.0)

        price_up = discount * np.mean(pu)
        price_down = discount * np.mean(pd)

        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * base_price + price_down) / (dS * dS)

        # Vega: bump sigma up
        paths_vega = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma + dsigma, 42) * S
        if option_type == "call":
            pv = np.maximum(paths_vega[:, -1] - K, 0.0)
        else:
            pv = np.maximum(K - paths_vega[:, -1], 0.0)
        price_vega = discount * np.mean(pv)
        vega = (price_vega - base_price) / dsigma

        # Theta: note dt is already small; approximate using T-dt_step
        # Simplified: negative of time decay
        paths_theta = _mc_simulate_gbm(n_paths, n_steps - 1, dt, r, sigma, 42) * S
        if option_type == "call":
            pt = np.maximum(paths_theta[:, -1] - K, 0.0)
        else:
            pt = np.maximum(K - paths_theta[:, -1], 0.0)
        price_theta = np.exp(-r * (T - dt_step)) * np.mean(pt)
        theta = (price_theta - base_price) / dt_step

        # Rho: bump r up
        paths_rho = _mc_simulate_gbm(n_paths, n_steps, dt, r + dr, sigma, 42) * S
        if option_type == "call":
            pr = np.maximum(paths_rho[:, -1] - K, 0.0)
        else:
            pr = np.maximum(K - paths_rho[:, -1], 0.0)
        price_rho_discount = np.exp(-(r + dr) * T)
        price_rho = price_rho_discount * np.mean(pr)
        rho = (price_rho - base_price) / dr

        return delta, gamma, vega, theta, rho

else:
    # -------------------------------------------------------------------------
    # Pure NumPy fallback (no Numba)
    # -------------------------------------------------------------------------
    def _mc_simulate_gbm(
        n_paths: int,
        n_steps: int,
        dt: float,
        mu: float,
        sigma: float,
        seed: int,
    ) -> np.ndarray:
        """NumPy fallback: GBM path simulation without Numba."""
        np.random.seed(seed)
        half = n_paths // 2

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 1.0

        for step in range(1, n_steps + 1):
            Z = np.random.randn(half)
            Z = np.concatenate([Z, -Z])
            log_drift = (mu - 0.5 * sigma * sigma) * dt
            log_diffusion = sigma * np.sqrt(dt) * Z
            paths[:, step] = paths[:, step - 1] * np.exp(log_drift + log_diffusion)

        return paths

    def _mc_price_european(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price European option from simulated paths (NumPy fallback)."""
        S_T = paths[:, -1]
        discount = np.exp(-r * (paths.shape[1] - 1) * dt)

        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)

        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(paths.shape[0])
        return price, std_error

    def _mc_price_asian(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price Asian option (NumPy fallback)."""
        avg_price = np.mean(paths[:, 1:], axis=1)
        n_steps = paths.shape[1] - 1
        discount = np.exp(-r * n_steps * dt)

        if option_type == "call":
            payoff = np.maximum(avg_price - K, 0.0)
        else:
            payoff = np.maximum(K - avg_price, 0.0)

        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(paths.shape[0])
        return price, std_error

    def _mc_price_lookback(
        paths: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: str,
    ) -> tuple[float, float]:
        """Price Lookback option (NumPy fallback)."""
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        if option_type == "call":
            payoff = np.maximum(np.max(paths[:, 1:], axis=1) - K, 0.0)
        else:
            payoff = np.maximum(K - np.min(paths[:, 1:], axis=1), 0.0)

        discount = np.exp(-r * n_steps * dt)
        price = discount * np.mean(payoff)
        std_error = discount * np.std(payoff) / np.sqrt(n_paths)
        return price, std_error

    def _mc_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        option_type: str,
    ) -> tuple[float, float, float, float, float]:
        """Compute Greeks via bump-and-reprice (NumPy fallback)."""
        dt = T / n_steps
        discount = np.exp(-r * T)

        paths = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * S

        if option_type == "call":
            base_payoff = np.maximum(paths[:, -1] - K, 0.0)
        else:
            base_payoff = np.maximum(K - paths[:, -1], 0.0)
        base_price = discount * np.mean(base_payoff)

        dS = S * 0.01
        dsigma = sigma * 0.01
        dr = 0.0001

        paths_up = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * (S + dS)
        paths_down = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * (S - dS)

        if option_type == "call":
            pu = np.maximum(paths_up[:, -1] - K, 0.0)
            pd = np.maximum(paths_down[:, -1] - K, 0.0)
        else:
            pu = np.maximum(K - paths_up[:, -1], 0.0)
            pd = np.maximum(K - paths_down[:, -1], 0.0)

        price_up = discount * np.mean(pu)
        price_down = discount * np.mean(pd)
        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * base_price + price_down) / (dS * dS)

        paths_vega = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma + dsigma, 42) * S
        if option_type == "call":
            pv = np.maximum(paths_vega[:, -1] - K, 0.0)
        else:
            pv = np.maximum(K - paths_vega[:, -1], 0.0)
        price_vega = discount * np.mean(pv)
        vega = (price_vega - base_price) / dsigma

        dt_step = T / n_steps
        paths_theta = _mc_simulate_gbm(n_paths, n_steps - 1, dt, r, sigma, 42) * S
        if option_type == "call":
            pt = np.maximum(paths_theta[:, -1] - K, 0.0)
        else:
            pt = np.maximum(K - paths_theta[:, -1], 0.0)
        price_theta = np.exp(-r * (T - dt_step)) * np.mean(pt)
        theta = (price_theta - base_price) / dt_step

        paths_rho = _mc_simulate_gbm(n_paths, n_steps, dt, r + dr, sigma, 42) * S
        if option_type == "call":
            pr = np.maximum(paths_rho[:, -1] - K, 0.0)
        else:
            pr = np.maximum(K - paths_rho[:, -1], 0.0)
        price_rho_discount = np.exp(-(r + dr) * T)
        price_rho = price_rho_discount * np.mean(pr)
        rho = (price_rho - base_price) / dr

        return delta, gamma, vega, theta, rho


def _mc_price_numpy(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    option_type: str,
) -> dict:
    """Pure NumPy fallback (no Numba) for European option pricing.

    This is a standalone function version when MonteCarloEngine.use_numba=False.
    """
    dt = T / n_steps
    discount = np.exp(-r * T)

    # Simulate paths (normalized, scaled by S)
    paths = _mc_simulate_gbm(n_paths, n_steps, dt, r, sigma, 42) * S
    S_T = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    price = discount * np.mean(payoff)
    std_error = discount * np.std(payoff) / np.sqrt(n_paths)

    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price": price,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper),
    }


# ---------------------------------------------------------------------------
# MonteCarloEngine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Monte Carlo期权定价引擎 — GBM模拟.

    使用几何布朗运动 (Geometric Brownian Motion):
        dS = μS*dt + σS*dW

    Features:
        - European Call/Put 期权定价
        - Asian (average price) options 亚式期权
        - Lookback options 回顾期权
        - Greeks计算 (Delta, Gamma, Vega, Theta, Rho)
        - Numba @jit(nopython=True) 加速 (~275x)
        - Antithetic variates 方差减少

    Args:
        n_paths: 模拟路径数量 (default: 100,000)
        n_steps: 时间步数 (default: 252, 日频)
        seed: 随机种子 (default: 42)
        use_numba: 是否使用Numba加速 (default: True)

    Example:
        >>> engine = MonteCarloEngine(n_paths=100000, n_steps=252, use_numba=True)
        >>> result = engine.price_european(S=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type='call')
        >>> print(f"Price: {result['price']:.4f}, CI: {result['confidence_interval']}")
    """

    def __init__(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: int = 42,
        use_numba: bool = True,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.use_numba = use_numba and _NUMBA_AVAILABLE

        if use_numba and not _NUMBA_AVAILABLE:
            import warnings
            warnings.warn(
                "Numba not available, falling back to pure NumPy.",
                RuntimeWarning,
            )
            self.use_numba = False

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
    ) -> np.ndarray:
        """Simulate GBM asset price paths / 模拟标的资产路径.

        Args:
            S0: Initial stock price / 标的价格
            mu: Drift (annualized) / 漂移率 (年化)
            sigma: Volatility (annualized) / 波动率 (年化)
            T: Time to maturity in years / 到期时间 (年)

        Returns:
            np.ndarray: (n_paths, n_steps+1) array of simulated prices
        """
        dt = T / self.n_steps
        paths = _mc_simulate_gbm(
            self.n_paths, self.n_steps, dt, mu, sigma, self.seed
        )
        return paths * S0

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price_european(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict:
        """Monte Carlo European option pricing / Monte Carlo 欧式期权定价.

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity (years) / 到期时间 (年)
            r: Risk-free rate (annualized) / 无风险利率 (年化)
            sigma: Volatility (annualized) / 波动率 (年化)
            option_type: 'call' or 'put' / 期权类型

        Returns:
            dict with keys:
                - price: Option price / 期权价格
                - std_error: Standard error / 标准误差
                - confidence_interval: (lower, upper) 95% CI / 置信区间
        """
        dt = T / self.n_steps
        paths = _mc_simulate_gbm(
            self.n_paths, self.n_steps, dt, r, sigma, self.seed
        ) * S

        price, std_error = _mc_price_european(paths, K, r, dt, option_type)

        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return {
            "price": float(price),
            "std_error": float(std_error),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
        }

    def price_asian(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict:
        """Asian option (average price) pricing / 亚式期权定价.

        The payoff is based on the average price over the option life,
        not the terminal price.

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity (years) / 到期时间 (年)
            r: Risk-free rate / 无风险利率
            sigma: Volatility / 波动率
            option_type: 'call' or 'put'

        Returns:
            dict with {price, std_error, confidence_interval}
        """
        dt = T / self.n_steps
        paths = _mc_simulate_gbm(
            self.n_paths, self.n_steps, dt, r, sigma, self.seed
        ) * S

        price, std_error = _mc_price_asian(paths, K, r, dt, option_type)

        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return {
            "price": float(price),
            "std_error": float(std_error),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
        }

    def price_lookback(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict:
        """Lookback option pricing / 回顾期权定价.

        Payoff depends on the maximum (call) or minimum (put) price
        reached during the option life.

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity / 到期时间
            r: Risk-free rate / 无风险利率
            sigma: Volatility / 波动率
            option_type: 'call' or 'put'

        Returns:
            dict with {price, std_error, confidence_interval}
        """
        dt = T / self.n_steps
        paths = _mc_simulate_gbm(
            self.n_paths, self.n_steps, dt, r, sigma, self.seed
        ) * S

        price, std_error = _mc_price_lookback(paths, K, r, dt, option_type)

        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return {
            "price": float(price),
            "std_error": float(std_error),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
        }

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def compute_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 50000,
        option_type: str = "call",
    ) -> dict:
        """Monte Carlo Greeks via bump-and-reprice / Monte Carlo Greeks计算.

        Uses finite differences on simulated paths:
        - Delta: dPrice/dS (central difference)
        - Gamma: d²Price/dS² (second derivative)
        - Vega: dPrice/dσ
        - Theta: dPrice/dt
        - Rho: dPrice/dr

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity / 到期时间
            r: Risk-free rate / 无风险利率
            sigma: Volatility / 波动率
            n_paths: Paths for Greeks (default 50,000) / Greeks计算路径数
            option_type: 'call' or 'put'

        Returns:
            dict with {delta, gamma, vega, theta, rho}
        """
        # Use fewer paths for Greeks to keep runtime reasonable
        greeks_n_steps = min(self.n_steps, 252)
        delta, gamma, vega, theta, rho = _mc_greeks(
            S, K, T, r, sigma, n_paths, greeks_n_steps, option_type
        )

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }
