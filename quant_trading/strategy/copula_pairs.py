"""
Copula-based Pairs Trading Strategy
基于Copula的配对交易策略

Implements Gaussian Copula, Student-t Copula, and DCC-GARCH for
pairs trading with dynamic correlation modeling.
实现高斯Copula、Student-t Copula和DCC-GARCH，用于动态相关性的配对交易。

References:
- Christopher Krauss & Johannes Stübinger:
  "Nonlinear dependence modeling with bivariate copulas: Statistical arbitrage pairs trading on the S&P 100"
- Robert Engle (2002): "Dynamic Conditional Correlation - A Simple Class of Multivariate GARCH Models"
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import StrategyParams from base (avoid circular import via local import)
try:
    from .base import StrategyParams
except ImportError:
    from quant_trading.strategy.base import StrategyParams

# ----------------------------------------------------------------------
# Helper Functions / 辅助函数
# ----------------------------------------------------------------------


def to_uniform_margins(
    x: np.ndarray,
    y: np.ndarray,
    ecdf: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform raw returns to uniform [0,1] margins via empirical CDF.
    将原始收益率通过经验CDF转换为均匀[0,1]边缘分布。

    Uses rank-based probability integral transform: u_i = rank(x_i) / (n+1).
    使用基于排名的概率积分变换。

    Parameters
    ----------
    x : np.ndarray
        First asset returns (length n).
    y : np.ndarray
        Second asset returns (length n).
    ecdf : bool
        If True use empirical CDF; if False fit Gaussian margins.

    Returns
    -------
    u, v : tuple of np.ndarray
        Uniform marginals in [0, 1].
    """
    n = len(x)
    if ecdf:
        # Empirical CDF via ranks
        rank_x = np.argsort(np.argsort(x)) + 1.0
        rank_y = np.argsort(np.argsort(y)) + 1.0
        u = rank_x / (n + 1.0)
        v = rank_y / (n + 1.0)
    else:
        # Parametric Gaussian margins
        u = stats.norm.cdf(x, loc=np.mean(x), scale=np.std(x, ddof=1) + 1e-10)
        v = stats.norm.cdf(y, loc=np.mean(y), scale=np.std(y, ddof=1) + 1e-10)
    return u, v


def kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Kendall's tau rank correlation (O(n log n) via sorting).
    计算Kendall's tau秩相关（通过排序的O(n log n)算法）。

    tau = (concordant - discordant) / (n*(n-1)/2)
    Concordant: (u_i - u_j)*(v_i - v_j) > 0
    Discordant: (u_i - u_j)*(v_i - v_j) < 0

    Parameters
    ----------
    u, v : np.ndarray
        Uniform marginals.

    Returns
    -------
    float
        Kendall's tau in [-1, 1].
    """
    n = len(u)
    if n < 2:
        return 0.0
    # Sort by u and compute sign flips in v
    idx = np.argsort(u)
    v_sorted = v[idx]
    # Count discordant pairs: sign changes in v_sorted
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (u[idx[i]] - u[idx[j]]) * (v_sorted[i] - v_sorted[j]) < 0:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    concordant = total_pairs - discordant
    return (concordant - discordant) / total_pairs


def cornish_fisher_quantile(p: float, mu: float, sigma: float,
                             skew: float, kurt: float) -> float:
    """
    Cornish-Fisher expansion for quantile correction with higher moments.
    Cornish-Fisher展开，用于高阶矩的分位数校正。

    z = Φ⁻¹(p) + (S/6)*(Φ⁻¹(p)²-1) + (K-3)/24*(Φ⁻¹(p)³-3*Φ⁻¹(p)) - ...
    """
    z = stats.norm.ppf(p)
    if abs(sigma) < 1e-10:
        return mu
    x = (p - mu) / sigma
    # Skewness correction
    correction = skew / 6.0 * (z * z - 1.0)
    # Kurtosis correction
    correction += (kurt - 3.0) / 24.0 * (z * z * z - 3.0 * z)
    return mu + sigma * (x + correction)


# ----------------------------------------------------------------------
# Copula Base Classes / Copula基类
# ----------------------------------------------------------------------


class CopulaType(Enum):
    """Copula family enumeration."""
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


class BaseCopula:
    """
    Abstract base class for bivariate copula models.
    二元Copula模型的抽象基类。

    Subclasses must implement:
    - cdf(u, v, params): Joint CDF C(u, v)
    - pdf(u, v, params): Joint PDF c(u, v)
    - fit(u, v): Fit parameters via MLE
    - h_func_1(u, v, params): H(u|v) conditional CDF
    - h_func_2(u, v, params): H(v|u) conditional CDF
    """

    name: str = "base_copula"

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """
        Fit copula parameters to uniform margins via Maximum Likelihood.
        通过最大似然估计拟合Copula参数。

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] marginals.

        Returns
        -------
        dict
            Fitted parameters and diagnostics.
        """
        raise NotImplementedError

    def simulate(self, n: int, params: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate n samples from the fitted copula.
        从拟合的Copula中模拟n个样本。

        Parameters
        ----------
        n : int
            Number of samples.
        params : tuple
            Copula parameters.

        Returns
        -------
        u_sim, v_sim : tuple of np.ndarray
            Simulated uniform marginals.
        """
        raise NotImplementedError

    def compute_spread(
        self,
        u: np.ndarray,
        v: np.ndarray,
        method: str = "h-condition",
    ) -> np.ndarray:
        """
        Compute the copula-based spread (log-returns ratio or H-distance).
        计算基于Copula的价差（H条件概率或概率距离）。

        Parameters
        ----------
        u, v : np.ndarray
            Uniform marginals.
        method : str
            "h-condition": H(u|v) - 0.5 (centered distance)
            "probability": |C(u,v) - u*v| (probability distance)
            "log-ratio": log(u/(1-u)) / log(v/(1-v)) (odds ratio)

        Returns
        -------
        spread : np.ndarray
            Spread series.
        """
        raise NotImplementedError

    def generate_signals(
        self,
        u: np.ndarray,
        v: np.ndarray,
        entry_thresholds: Tuple[float, float] = (0.95, 0.05),
        exit_thresholds: Tuple[float, float] = (0.5, 0.5),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from conditional copula probabilities.
        基于条件Copula概率生成交易信号。

        Entry rules (based on Krauss & Stübinger):
          - H(u|v) >= upper_threshold AND H(v|u) <= lower_threshold → long u, short v
          - H(u|v) <= lower_threshold AND H(v|u) >= upper_threshold → short u, long v
        Exit rules:
          - Both cross mid-threshold → close positions

        Parameters
        ----------
        u, v : np.ndarray
            Uniform marginals of asset 1 and asset 2.
        entry_thresholds : tuple
            (upper, lower) thresholds for entry, e.g. (0.95, 0.05).
        exit_thresholds : tuple
            (upper, lower) thresholds for exit, e.g. (0.5, 0.5).

        Returns
        -------
        signal_u, signal_v : tuple of np.ndarray
            Signal arrays: 1=long, -1=short, 0=flat.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Gaussian Copula / 高斯Copula
# ----------------------------------------------------------------------


class GaussianCopula(BaseCopula):
    """
    Bivariate Gaussian (Normal) Copula.
    二元高斯（正态）Copula。

    CDF:  C(u,v;ρ) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
    PDF:  c(u,v;ρ) = exp(-q/(2(1-ρ²))) / √(1-ρ²)
          where q = z₁² - 2ρ·z₁·z₂ + z₂², z_i = Φ⁻¹(u_i)

    Properties:
    - Single parameter: correlation coefficient ρ ∈ (-1, 1)
    - Symmetric dependence, no tail dependence
    - Elliptical level sets

    Example
    -------
    >>> gc = GaussianCopula()
    >>> x = np.random.randn(1000)
    >>> y = 0.7*x + np.sqrt(1-0.7**2)*np.random.randn(1000)
    >>> u, v = to_uniform_margins(x, y)
    >>> result = gc.fit(u, v)
    >>> print(f"rho = {result['rho']:.4f}")
    >>> u_sim, v_sim = gc.simulate(100, result['params'])
    """

    name = "gaussian_copula"

    def __init__(self):
        self.rho: Optional[float] = None
        self.params: Optional[Tuple[float]] = None

    def _rho_to_theta(self, rho: float) -> float:
        """Map correlation ρ to canonical parameter θ for Gaussian copula."""
        # For Gaussian copula the canonical parameter is the correlation
        return rho

    def _theta_to_rho(self, theta: float) -> float:
        """Map canonical parameter θ to correlation ρ."""
        return np.clip(theta, -0.9999, 0.9999)

    def cdf(self, u: np.ndarray, v: np.ndarray,
            rho: Optional[float] = None) -> np.ndarray:
        """
        Bivariate Gaussian copula CDF.
        二元高斯Copula的累积分布函数。

        C(u,v;ρ) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float, optional
            Correlation coefficient. Uses self.rho if None.

        Returns
        -------
        c : np.ndarray
            CDF values in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        return stats.multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])

    def pdf(self, u: np.ndarray, v: np.ndarray,
            rho: Optional[float] = None) -> np.ndarray:
        """
        Bivariate Gaussian copula PDF.
        二元高斯Copula的概率密度函数。

        c(u,v;ρ) = exp(-(z₁² - 2ρ·z₁·z₂ + z₂²) / (2(1-ρ²))) / √(1-ρ²)

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float, optional
            Correlation coefficient.

        Returns
        -------
        density : np.ndarray
            PDF values.
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        rho = np.clip(rho, -0.9999, 0.9999)
        z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        q = z1**2 - 2 * rho * z1 * z2 + z2**2
        denom = 1 - rho**2
        density = np.exp(-q / (2 * denom)) / np.sqrt(denom)
        return density

    def h_func_1(self, u: np.ndarray, v: np.ndarray,
                 rho: Optional[float] = None) -> np.ndarray:
        """
        Conditional CDF H(u|v) = P(U ≤ u | V = v).
        条件累积分布函数 H(u|v) = P(U ≤ u | V = v)。

        H(u|v) = Φ((Φ⁻¹(u) - ρ·Φ⁻¹(v)) / √(1-ρ²))

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float, optional
            Correlation coefficient.

        Returns
        -------
        h1 : np.ndarray
            Conditional CDF values in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        rho = np.clip(rho, -0.9999, 0.9999)
        z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        denom = np.sqrt(1 - rho**2)
        h1 = stats.norm.cdf((z1 - rho * z2) / denom)
        return h1

    def h_func_2(self, u: np.ndarray, v: np.ndarray,
                 rho: Optional[float] = None) -> np.ndarray:
        """
        Conditional CDF H(v|u) = P(V ≤ v | U = u).
        条件累积分布函数 H(v|u) = P(V ≤ v | U = u)。

        H(v|u) = Φ((Φ⁻¹(v) - ρ·Φ⁻¹(u)) / √(1-ρ²))

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float, optional
            Correlation coefficient.

        Returns
        -------
        h2 : np.ndarray
            Conditional CDF values in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        rho = np.clip(rho, -0.9999, 0.9999)
        z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        denom = np.sqrt(1 - rho**2)
        h2 = stats.norm.cdf((z2 - rho * z1) / denom)
        return h2

    def h_func_1_pdf(self, u: np.ndarray, v: np.ndarray,
                      rho: Optional[float] = None) -> np.ndarray:
        """
        Conditional PDF h₁(u|v) = ∂H(u|v)/∂u.
        条件概率密度函数 h₁(u|v) = ∂H(u|v)/∂u。

        h₁(u|v) = φ((Φ⁻¹(u)-ρ·Φ⁻¹(v))/√(1-ρ²)) / φ(Φ⁻¹(u)) / √(1-ρ²)
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        rho = np.clip(rho, -0.9999, 0.9999)
        z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        denom = np.sqrt(1 - rho**2)
        numer = z1 - rho * z2
        h1 = stats.norm.cdf(numer / denom)
        h1_pdf = stats.norm.pdf(numer / denom) / (stats.norm.pdf(z1) * denom + 1e-10)
        return h1_pdf

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """
        Fit Gaussian copula via Maximum Likelihood Estimation (MLE).
        通过最大似然估计（MLE）拟合高斯Copula。

        The log-likelihood is:
          L(ρ) = Σ log c(u_i, v_i; ρ)
               = Σ [-q_i/(2(1-ρ²)) - 0.5*log(1-ρ²)]

        Kendall's tau initializes: ρ₀ = sin(π/2 * tau)

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] marginals from empirical CDF.

        Returns
        -------
        dict
            Fitted parameters:
              - rho: correlation coefficient
              - params: (rho,) tuple
              - log_likelihood: maximized log-likelihood
              - aic: Akaike Information Criterion
              - bic: Bayesian Information Criterion
              - kendall_tau: observed Kendall's tau
              - convergence: optimization success flag
        """
        n = len(u)
        if n < 10:
            raise ValueError(f"Need at least 10 observations, got {n}")

        # Remove any NaN or out-of-range values
        mask = np.isfinite(u) & np.isfinite(v) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
        u = np.asarray(u)[mask]
        v = np.asarray(v)[mask]

        # Initialize from Kendall's tau: rho = sin(pi/2 * tau)
        tau = kendall_tau(u, v)
        rho_init = np.clip(np.sin(np.pi / 2 * tau), -0.95, 0.95)

        # Negative log-likelihood
        def neg_ll(rho):
            rho = np.clip(rho, -0.9999, 0.9999)
            z1 = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            z2 = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
            q = z1**2 - 2 * rho * z1 * z2 + z2**2
            denom = 1 - rho**2
            ll = -0.5 * np.sum(np.log(denom + 1e-10) + q / (denom + 1e-10))
            return -ll

        # Bound-constrained optimization
        result = minimize(
            neg_ll,
            x0=[rho_init],
            bounds=[(-0.9999, 0.9999)],
            method="L-BFGS-B",
        )
        rho_opt = float(result.x[0])

        # Compute diagnostics
        ll_max = -result.fun
        aic = 2 - 2 * ll_max
        bic = np.log(n) - 2 * ll_max

        self.rho = rho_opt
        self.params = (rho_opt,)

        return {
            "rho": rho_opt,
            "params": (rho_opt,),
            "log_likelihood": ll_max,
            "aic": aic,
            "bic": bic,
            "kendall_tau": tau,
            "convergence": result.success,
            "n_observations": n,
        }

    def simulate(self, n: int,
                rho: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate from Gaussian copula.
        从高斯Copula中模拟样本。

        Algorithm: (Z1, Z2) ~ N₂(0, Σ) → U = Φ(Z1), V = Φ(Z2)

        Parameters
        ----------
        n : int
            Number of samples.
        rho : float, optional
            Correlation coefficient. Uses self.rho if None.

        Returns
        -------
        u_sim, v_sim : tuple of np.ndarray
            Simulated uniform marginals in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        if rho is None:
            raise ValueError("rho must be provided or model must be fitted first")
        rho = np.clip(rho, -0.9999, 0.9999)

        # Simulate bivariate normal
        mean = [0.0, 0.0]
        cov = [[1.0, rho], [rho, 1.0]]
        z1, z2 = np.random.multivariate_normal(mean, cov, n).T

        # Transform to uniform via CDF
        u_sim = stats.norm.cdf(z1)
        v_sim = stats.norm.cdf(z2)
        return u_sim, v_sim

    def compute_spread(self, u: np.ndarray, v: np.ndarray,
                       method: str = "h-condition",
                       rho: Optional[float] = None) -> np.ndarray:
        """
        Compute copula-based spread.

        Parameters
        ----------
        u, v : np.ndarray
            Uniform marginals.
        method : str
            "h-condition": H(u|v) - 0.5 (default, centered)
            "probability": C(u,v) - u*v (probability distance)
            "log-ratio": log(u/v) (log odds ratio)
        rho : float, optional
            Correlation. Uses self.rho if None.

        Returns
        -------
        spread : np.ndarray
        """
        rho = rho if rho is not None else self.rho
        if method == "h-condition":
            h1 = self.h_func_1(u, v, rho)
            return h1 - 0.5
        elif method == "probability":
            c = self.cdf(u, v, rho)
            return c - u * v
        elif method == "log-ratio":
            return np.log(u / (1 - u + 1e-10)) - np.log(v / (1 - v + 1e-10))
        else:
            raise ValueError(f"Unknown spread method: {method}")

    def generate_signals(self, u: np.ndarray, v: np.ndarray,
                         entry_thresholds: Tuple[float, float] = (0.95, 0.05),
                         exit_thresholds: Tuple[float, float] = (0.5, 0.5),
                         rho: Optional[float] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from conditional probabilities.

        Parameters
        ----------
        u, v : np.ndarray
            Uniform marginals.
        entry_thresholds : tuple
            (upper, lower) entry thresholds, default (0.95, 0.05).
        exit_thresholds : tuple
            (upper, lower) exit thresholds, default (0.5, 0.5).
        rho : float, optional
            Correlation. Uses self.rho if None.

        Returns
        -------
        signal_u, signal_v : tuple of np.ndarray
            Signals for asset 1 and asset 2: 1=long, -1=short, 0=flat.
        """
        rho = rho if rho is not None else self.rho
        upper_entry, lower_entry = entry_thresholds
        upper_exit, lower_exit = exit_thresholds

        h1 = self.h_func_1(u, v, rho)  # H(u|v)
        h2 = self.h_func_2(u, v, rho)  # H(v|u)

        signal_u = np.zeros(len(u), dtype=np.int32)
        signal_v = np.zeros(len(u), dtype=np.int32)

        pos_state = 0  # 0=flat, 1=long-u/short-v, -1=short-u/long-v

        for i in range(len(u)):
            h1_i = h1[i]
            h2_i = h2[i]

            if pos_state == 0:
                # Entry logic
                if h1_i >= upper_entry and h2_i <= lower_entry:
                    pos_state = 1  # Long u, short v
                elif h1_i <= lower_entry and h2_i >= upper_entry:
                    pos_state = -1  # Short u, long v

            elif pos_state == 1:
                # Long u, short v → exit when both cross 0.5
                if h1_i <= upper_exit and h2_i >= lower_exit:
                    pos_state = 0

            elif pos_state == -1:
                # Short u, long v → exit when both cross 0.5
                if h1_i >= lower_exit and h2_i <= upper_exit:
                    pos_state = 0

            signal_u[i] = pos_state
            signal_v[i] = -pos_state

        return signal_u, signal_v


# ----------------------------------------------------------------------
# Student-t Copula / Student-t Copula
# ----------------------------------------------------------------------


class StudentTCopula(BaseCopula):
    """
    Bivariate Student-t Copula.
    二元Student-t Copula。

    CDF:  C(u,v;ρ,ν) = T₂_ν(T_ν⁻¹(u), T_ν⁻¹(v); ρ)
    PDF:  c(u,v;ρ,ν) = Γ((ν+2)/2) / (Γ(ν/2)·π·ν·√(1-ρ²))
                · [1 + (z₁² - 2ρ·z₁·z₂ + z₂²)/(ν·(1-ρ²))]⁻⁽ν⁺²⁾/²
          where z_i = T_ν⁻¹(u_i)

    Properties:
    - Two parameters: correlation ρ ∈ (-1,1) and dof ν > 2
    - Symmetric dependence with tail dependence (stronger for smaller ν)
    - Fat tails captured via ν degrees of freedom

    Example
    -------
    >>> tc = StudentTCopula()
    >>> x = np.random.randn(1000)
    >>> y = 0.7*x + np.sqrt(1-0.7**2)*np.random.randn(1000)
    >>> u, v = to_uniform_margins(x, y)
    >>> result = tc.fit(u, v)
    >>> print(f"rho={result['rho']:.4f}, nu={result['nu']:.2f}")
    """

    name = "student_t_copula"

    def __init__(self):
        self.rho: Optional[float] = None
        self.nu: Optional[float] = None
        self.params: Optional[Tuple[float, float]] = None

    def _t_pdf_scalar(self, x: float, nu: float) -> float:
        """Student-t PDF at scalar x."""
        return stats.t.pdf(x, df=nu)

    def _t_cdf_scalar(self, x: float, nu: float) -> float:
        """Student-t CDF at scalar x."""
        return stats.t.cdf(x, df=nu)

    def cdf(self, u: np.ndarray, v: np.ndarray,
            rho: Optional[float] = None,
            nu: Optional[float] = None) -> np.ndarray:
        """
        Bivariate Student-t copula CDF.
        二元Student-t Copula的累积分布函数。

        C(u,v;ρ,ν) = T₂_ν(T_ν⁻¹(u), T_ν⁻¹(v); ρ)

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float
            Correlation coefficient.
        nu : float
            Degrees of freedom (ν > 2).

        Returns
        -------
        c : np.ndarray
            CDF values in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if rho is None or nu is None:
            raise ValueError("rho and nu must be provided or model must be fitted")

        rho = np.clip(rho, -0.9999, 0.9999)
        nu = max(nu, 2.01)

        z1 = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
        z2 = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)

        # Bivariate t-CDF using scipy (quadrature)
        c = np.array([
            stats.multivariate_t.cdf([zi1, zi2], loc=[0, 0],
                                      shape=[[1, rho], [rho, 1]], df=nu)
            for zi1, zi2 in zip(z1, z2)
        ])
        return c

    def pdf(self, u: np.ndarray, v: np.ndarray,
            rho: Optional[float] = None,
            nu: Optional[float] = None) -> np.ndarray:
        """
        Bivariate Student-t copula PDF.
        二元Student-t Copula的概率密度函数。

        c(u,v;ρ,ν) = Γ((ν+2)/2) / (Γ(ν/2)·π·ν·√(1-ρ²))
                     · [1 + (z₁² - 2ρ·z₁·z₂ + z₂²)/(ν·(1-ρ²))]⁻⁽ν⁺²⁾/²

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] values.
        rho : float
            Correlation coefficient.
        nu : float
            Degrees of freedom (ν > 2).

        Returns
        -------
        density : np.ndarray
            PDF values.
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if rho is None or nu is None:
            raise ValueError("rho and nu must be provided or model must be fitted")

        rho = np.clip(rho, -0.9999, 0.9999)
        nu = max(nu, 2.01)

        z1 = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
        z2 = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)

        q = z1**2 - 2 * rho * z1 * z2 + z2**2
        denom = 1 - rho**2
        term = 1 + q / (nu * denom)
        # Student-t copula PDF formula
        coef = np.exp(
            scipy.special.gammaln((nu + 2) / 2) - scipy.special.gammaln(nu / 2)
        ) / (np.pi * nu * np.sqrt(denom))
        density = coef * np.power(term, -(nu + 2) / 2)
        return density

    def h_func_1(self, u: np.ndarray, v: np.ndarray,
                 rho: Optional[float] = None,
                 nu: Optional[float] = None) -> np.ndarray:
        """
        Conditional CDF H(u|v) = P(U ≤ u | V = v).
        条件累积分布函数 H(u|v) = P(U ≤ u | V = v)。

        For Student-t copula, H(u|v) is computed via the ratio of
        bivariate t-CDF to marginal t-CDF (requires numerical integration).
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if rho is None or nu is None:
            raise ValueError("rho and nu must be provided or model must be fitted")

        rho = np.clip(rho, -0.9999, 0.9999)
        nu = max(nu, 2.01)

        z1 = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
        z2_scalar = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)

        # H(u|v) = t_{ν+1}( (T_ν⁻¹(u) - ρ·T_ν⁻¹(v)) / √((ν+(T_ν⁻¹(v))²)·(1-ρ²)/(ν+1)) )
        # Direct conditional formula for elliptical copulas
        denom_factor = np.sqrt((nu + z2_scalar**2) * (1 - rho**2) / (nu + 1))
        a = (z1 - rho * z2_scalar) / (denom_factor + 1e-10)
        h1 = stats.t.cdf(a, df=nu + 1)
        return h1

    def h_func_2(self, u: np.ndarray, v: np.ndarray,
                 rho: Optional[float] = None,
                 nu: Optional[float] = None) -> np.ndarray:
        """
        Conditional CDF H(v|u) = P(V ≤ v | U = u).
        条件累积分布函数 H(v|u) = P(V ≤ v | U = u)。
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if rho is None or nu is None:
            raise ValueError("rho and nu must be provided or model must be fitted")

        rho = np.clip(rho, -0.9999, 0.9999)
        nu = max(nu, 2.01)

        z2 = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)
        z1_scalar = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)

        denom_factor = np.sqrt((nu + z1_scalar**2) * (1 - rho**2) / (nu + 1))
        a = (z2 - rho * z1_scalar) / (denom_factor + 1e-10)
        h2 = stats.t.cdf(a, df=nu + 1)
        return h2

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """
        Fit Student-t copula via Maximum Likelihood Estimation (MLE).
        通过最大似然估计（MLE）拟合Student-t Copula。

        Two-parameter optimization: (ρ, ν) via L-BFGS-B.
        Initialization: ρ from Gaussian copula MLE, ν from profile likelihood.

        Parameters
        ----------
        u, v : np.ndarray
            Uniform [0,1] marginals from empirical CDF.

        Returns
        -------
        dict
            Fitted parameters with diagnostics.
        """
        import scipy.special  # for gamma functions

        n = len(u)
        if n < 10:
            raise ValueError(f"Need at least 10 observations, got {n}")

        mask = np.isfinite(u) & np.isfinite(v) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
        u = np.asarray(u)[mask]
        v = np.asarray(v)[mask]

        # Step 1: fit Gaussian copula for rho initialization
        gc = GaussianCopula()
        gc_result = gc.fit(u, v)
        rho_init = np.clip(gc_result["rho"], -0.9, 0.9)

        # Step 2: profile likelihood for nu initialization
        def neg_ll_profile(nu):
            nu = max(nu, 2.01)
            rho = rho_init
            rho = np.clip(rho, -0.9999, 0.9999)
            z1 = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
            z2 = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)
            q = z1**2 - 2 * rho * z1 * z2 + z2**2
            denom = 1 - rho**2
            term = 1 + q / (nu * denom)
            ll = np.sum(
                scipy.special.gammaln((nu + 2) / 2) - scipy.special.gammaln(nu / 2)
                - 0.5 * np.log(np.pi * nu * denom)
                -(nu + 2) / 2 * np.log(term + 1e-10)
            )
            return -ll

        # Find best nu between 3 and 100
        res_nu = minimize_scalar(neg_ll_profile, bounds=(3.0, 100.0), method="bounded")
        nu_init = res_nu.x

        def neg_ll_2d(params):
            rho, nu = params
            rho = np.clip(rho, -0.9999, 0.9999)
            nu = max(nu, 2.01)
            z1 = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
            z2 = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)
            q = z1**2 - 2 * rho * z1 * z2 + z2**2
            denom = 1 - rho**2
            term = 1 + q / (nu * denom)
            ll = np.sum(
                scipy.special.gammaln((nu + 2) / 2) - scipy.special.gammaln(nu / 2)
                - 0.5 * np.log(np.pi * nu * denom)
                - (nu + 2) / 2 * np.log(term + 1e-10)
            )
            return -ll if np.isfinite(ll) else 1e10

        result = minimize(
            neg_ll_2d,
            x0=[rho_init, nu_init],
            bounds=[(-0.9999, 0.9999), (2.01, 200.0)],
            method="L-BFGS-B",
        )
        rho_opt, nu_opt = result.x
        rho_opt = float(rho_opt)
        nu_opt = float(nu_opt)

        ll_max = -result.fun
        aic = 4 - 2 * ll_max  # 2 params
        bic = 2 * np.log(n) - 2 * ll_max

        self.rho = rho_opt
        self.nu = nu_opt
        self.params = (rho_opt, nu_opt)

        return {
            "rho": rho_opt,
            "nu": nu_opt,
            "params": (rho_opt, nu_opt),
            "log_likelihood": ll_max,
            "aic": aic,
            "bic": bic,
            "kendall_tau": kendall_tau(u, v),
            "convergence": result.success,
            "n_observations": n,
        }

    def simulate(self, n: int,
                 rho: Optional[float] = None,
                 nu: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate from Student-t copula.
        从Student-t Copula中模拟样本。

        Algorithm:
          1. Sample S ~ χ²_ν (chi-square with ν dof)
          2. Sample (Z1, Z2) ~ N₂(0, Σ) with Σ = [[1, ρ], [ρ, 1]]
          3. X_i = Z_i · √(ν/S)
          4. U_i = T_ν(X_i)

        Parameters
        ----------
        n : int
            Number of samples.
        rho : float, optional
            Correlation coefficient.
        nu : float, optional
            Degrees of freedom.

        Returns
        -------
        u_sim, v_sim : tuple of np.ndarray
            Simulated uniform marginals in [0, 1].
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if rho is None or nu is None:
            raise ValueError("rho and nu must be provided or model must be fitted")

        rho = np.clip(rho, -0.9999, 0.9999)
        nu = max(nu, 2.01)

        # Chi-square samples for the scaling
        s = np.random.chisquare(nu, n)
        s[s == 0] = 1e-10  # avoid division by zero

        # Bivariate normal
        mean = [0.0, 0.0]
        cov = [[1.0, rho], [rho, 1.0]]
        z1, z2 = np.random.multivariate_normal(mean, cov, n).T

        # Scale by √(ν/S)
        scale = np.sqrt(nu / s)
        x1 = z1 * scale
        x2 = z2 * scale

        u_sim = stats.t.cdf(x1, df=nu)
        v_sim = stats.t.cdf(x2, df=nu)
        return u_sim, v_sim

    def compute_spread(self, u: np.ndarray, v: np.ndarray,
                       method: str = "h-condition",
                       rho: Optional[float] = None,
                       nu: Optional[float] = None) -> np.ndarray:
        """
        Compute Student-t copula-based spread.
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        if method == "h-condition":
            h1 = self.h_func_1(u, v, rho, nu)
            return h1 - 0.5
        elif method == "probability":
            c = self.cdf(u, v, rho, nu)
            return c - u * v
        elif method == "log-ratio":
            return np.log(u / (1 - u + 1e-10)) - np.log(v / (1 - v + 1e-10))
        else:
            raise ValueError(f"Unknown spread method: {method}")

    def generate_signals(self, u: np.ndarray, v: np.ndarray,
                         entry_thresholds: Tuple[float, float] = (0.95, 0.05),
                         exit_thresholds: Tuple[float, float] = (0.5, 0.5),
                         rho: Optional[float] = None,
                         nu: Optional[float] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from conditional probabilities (Student-t).
        """
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        upper_entry, lower_entry = entry_thresholds
        upper_exit, lower_exit = exit_thresholds

        h1 = self.h_func_1(u, v, rho, nu)
        h2 = self.h_func_2(u, v, rho, nu)

        signal_u = np.zeros(len(u), dtype=np.int32)
        signal_v = np.zeros(len(u), dtype=np.int32)

        pos_state = 0
        for i in range(len(u)):
            h1_i = h1[i]
            h2_i = h2[i]
            if pos_state == 0:
                if h1_i >= upper_entry and h2_i <= lower_entry:
                    pos_state = 1
                elif h1_i <= lower_entry and h2_i >= upper_entry:
                    pos_state = -1
            elif pos_state == 1:
                if h1_i <= upper_exit and h2_i >= lower_exit:
                    pos_state = 0
            elif pos_state == -1:
                if h1_i >= lower_exit and h2_i <= upper_exit:
                    pos_state = 0
            signal_u[i] = pos_state
            signal_v[i] = -pos_state

        return signal_u, signal_v


# ----------------------------------------------------------------------
# DCC-GARCH Model / 动态条件相关GARCH模型
# ----------------------------------------------------------------------


class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH (DCC-GARCH) Model.
    动态条件相关GARCH模型。

    Implements Engle (2002) two-step DCC estimation:
      Step 1: Fit univariate GARCH(1,1) to each series → standardize residuals
      Step 2: Estimate DCC parameters (a, b) on standardized residuals

    Univariate GARCH(1,1):
      r_t = μ + σ_t · ε_t,   ε_t ~ N(0,1)
      σ²_t = ω + α · r_{t-1}² + β · σ²_{t-1}

    DCC dynamics:
      Q_t = (1-a-b)·Q + a·(e_{t-1}·e_{t-1}') + b·Q_{t-1}
      R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

    Parameters
    ----------
    a : float
        DCC alpha parameter (shock coefficient), a >= 0.
    b : float
        DCC beta parameter (lag coefficient), b >= 0, a+b < 1.

    Example
    -------
    >>> dcc = DCCGARCHModel()
    >>> dcc.fit(x, y)
    >>> correlations = dcc.get_correlations()  # time-varying correlations
    """

    name = "dcc_garch"

    def __init__(self):
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.rho_history: Optional[np.ndarray] = None

        # Univariate GARCH params: [omega, alpha, beta, mu]
        self.garch_params_x: Optional[np.ndarray] = None
        self.garch_params_y: Optional[np.ndarray] = None

    # --- Univariate GARCH(1,1) ---
    @staticmethod
    def _garch_log_likelihood(params: np.ndarray, r: np.ndarray) -> float:
        """Univariate GARCH(1,1) negative log-likelihood."""
        omega, alpha, beta, mu = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        T = len(r)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(r)

        ll = 0.0
        for t in range(1, T):
            e = r[t - 1] - mu
            sigma2[t] = (omega + alpha * e**2 + beta * sigma2[t - 1])
            if sigma2[t] <= 0:
                return 1e10
            ll += 0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + e**2 / sigma2[t])
        return -ll if np.isfinite(ll) else 1e10

    def _fit_garch(self, r: np.ndarray,
                   init_params: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        """Fit univariate GARCH(1,1) via MLE."""
        if init_params is None:
            var_r = np.var(r)
            init_params = np.array([var_r * 0.05, 0.05, 0.90, np.mean(r)])

        def objective(p):
            return self._garch_log_likelihood(p, r)

        result = minimize(
            objective,
            x0=init_params,
            bounds=[(1e-8, None), (1e-6, 0.999), (1e-6, 0.999), (None, None)],
            method="L-BFGS-B",
        )
        return result.x if result.success else init_params

    def _garch_forecast_sigma2(self, params: np.ndarray,
                               r: np.ndarray,
                               sigma2_last: float) -> np.ndarray:
        """Compute GARCH conditional variance series."""
        omega, alpha, beta, mu = params
        T = len(r)
        sigma2 = np.zeros(T)
        sigma2[0] = sigma2_last if sigma2_last > 0 else np.var(r)
        for t in range(1, T):
            e = r[t - 1] - mu
            sigma2[t] = omega + alpha * e**2 + beta * sigma2[t - 1]
        return sigma2

    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit DCC-GARCH model to two return series.
        拟合DCC-GARCH模型到两个收益率序列。

        Step 1: Fit univariate GARCH(1,1) to each series
        Step 2: Compute standardized residuals
        Step 3: Fit DCC parameters (a, b) to standardized residuals

        Parameters
        ----------
        x, y : np.ndarray
            Return series of asset 1 and asset 2 (same length).

        Returns
        -------
        dict
            Fitted parameters and diagnostics.
        """
        n = len(x)
        if n < 30:
            raise ValueError(f"Need at least 30 observations, got {n}")

        x = np.asarray(x)
        y = np.asarray(y)

        # ---- Step 1: Univariate GARCH ----
        self.garch_params_x = self._fit_garch(x)
        self.garch_params_y = self._fit_garch(y)

        # Compute conditional variances and standardized residuals
        omega_x, alpha_x, beta_x, mu_x = self.garch_params_x
        omega_y, alpha_y, beta_y, mu_y = self.garch_params_y

        sigma2_x = self._garch_forecast_sigma2(self.garch_params_x, x, np.var(x))
        sigma2_y = self._garch_forecast_sigma2(self.garch_params_y, y, np.var(y))

        # Standardized residuals (clipped to avoid division by zero)
        e_x = (x - mu_x) / np.sqrt(sigma2_x + 1e-10)
        e_y = (y - mu_y) / np.sqrt(sigma2_y + 1e-10)

        # ---- Step 2: DCC on standardized residuals ----
        # Initialize Q as identity matrix
        Q = np.eye(2)
        Q_uncor = np.eye(2)  # unconditional Q

        def dcc_neg_ll(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 1:
                return 1e10
            Q_t = Q.copy()
            ll = 0.0
            for t in range(1, n):
                e_vec = np.array([e_x[t - 1], e_y[t - 1]])
                Q_t = (1 - a - b) * Q_uncor + a * np.outer(e_vec, e_vec) + b * Q_t
                # Compute correlation from Q_t
                d_inv = 1.0 / np.sqrt(Q_t[0, 0] * Q_t[1, 1] + 1e-10)
                rho_t = Q_t[0, 1] * d_inv
                rho_t = np.clip(rho_t, -0.9999, 0.9999)
                # Negative log-likelihood for bivariate normal with correlation rho_t
                ll += -0.5 * np.log(1 - rho_t**2 + 1e-10) - 0.5 * (
                    e_x[t]**2 - 2 * rho_t * e_x[t] * e_y[t] + e_y[t]**2
                ) / (1 - rho_t**2 + 1e-10)
            return -ll if np.isfinite(ll) else 1e10

        # Initialize DCC params: a ≈ 0.05, b ≈ 0.93
        result = minimize(
            dcc_neg_ll,
            x0=[0.05, 0.93],
            bounds=[(1e-6, 0.999), (1e-6, 0.999)],
            method="L-BFGS-B",
        )
        self.a, self.b = float(result.x[0]), float(result.x[1])

        # ---- Compute time-varying correlations ----
        self.rho_history = np.zeros(n)
        Q_t = Q.copy()
        for t in range(n):
            if t > 0:
                e_vec = np.array([e_x[t - 1], e_y[t - 1]])
                Q_t = (1 - self.a - self.b) * Q_uncor + self.a * np.outer(e_vec, e_vec) + self.b * Q_t
            d_inv = 1.0 / np.sqrt(Q_t[0, 0] * Q_t[1, 1] + 1e-10)
            self.rho_history[t] = np.clip(Q_t[0, 1] * d_inv, -0.9999, 0.9999)

        ll_max = -result.fun
        return {
            "a": self.a,
            "b": self.b,
            "garch_x": self.garch_params_x.tolist(),
            "garch_y": self.garch_params_y.tolist(),
            "correlations": self.rho_history,
            "log_likelihood": ll_max,
            "convergence": result.success,
            "n_observations": n,
        }

    def get_correlations(self) -> np.ndarray:
        """
        Get the estimated time-varying correlation series.
        获取估计的时变相关序列。

        Returns
        -------
        rho_history : np.ndarray
            Correlation at each time step.
        """
        if self.rho_history is None:
            raise ValueError("Model must be fitted first. Call fit(x, y).")
        return self.rho_history.copy()

    def compute_spread(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "hedge-ratio",
    ) -> np.ndarray:
        """
        Compute the dynamic spread using DCC-GARCH correlations.
        使用DCC-GARCH动态相关计算价差。

        Parameters
        ----------
        x, y : np.ndarray
            Return series.
        method : str
            "hedge-ratio": spread = x - β·y where β = ρ·σ_x/σ_y (DCC-based hedge ratio)
            "residual": standardized residual from the mean-reversion equation

        Returns
        -------
        spread : np.ndarray
        """
        if self.rho_history is None:
            raise ValueError("Model must be fitted first. Call fit(x, y).")

        if method == "hedge-ratio":
            mu_x = self.garch_params_x[3]
            mu_y = self.garch_params_y[3]
            sigma2_x = self._garch_forecast_sigma2(self.garch_params_x, x, np.var(x))
            sigma2_y = self._garch_forecast_sigma2(self.garch_params_y, y, np.var(y))

            sigma_x = np.sqrt(sigma2_x + 1e-10)
            sigma_y = np.sqrt(sigma2_y + 1e-10)

            # DCC-based hedge ratio
            beta = self.rho_history * sigma_x / (sigma_y + 1e-10)
            spread = x - beta * y
            return spread
        else:
            raise ValueError(f"Unknown spread method: {method}")


# ----------------------------------------------------------------------
# Copula Pairs Strategy / Copula配对交易策略
# ----------------------------------------------------------------------


@dataclass
class CopulaPairsParams(StrategyParams):
    """
    Parameters for Copula-based Pairs Trading Strategy.
    基于Copula的配对交易策略参数。

    Attributes
    ----------
    copula_type : str
        "gaussian" or "student_t". Default: "gaussian".
    entry_upper : float
        Upper entry threshold for H(u|v). Default: 0.95.
    entry_lower : float
        Lower entry threshold for H(v|u). Default: 0.05.
    exit_upper : float
        Upper exit threshold. Default: 0.5.
    exit_lower : float
        Lower exit threshold. Default: 0.5.
    use_dcc : bool
        Whether to use DCC-GARCH for time-varying correlation. Default: False.
    min_periods : int
        Minimum training periods before signal generation. Default: 60.
    ecdf_margins : bool
        Use empirical CDF for margins. Default: True.
    """
    copula_type: str = "gaussian"
    entry_upper: float = 0.95
    entry_lower: float = 0.05
    exit_upper: float = 0.5
    exit_lower: float = 0.5
    use_dcc: bool = False
    min_periods: int = 60
    ecdf_margins: bool = True
    correlation_threshold: float = 0.5  # Minimum correlation for trading


class CopulaPairsStrategy:
    """
    Copula-based Pairs Trading Strategy.
    基于Copula的配对交易策略。

    Combines copula-based dependence modeling with dynamic correlation
    estimation (DCC-GARCH optional) for pairs trading on cointegrated assets.

    The strategy uses conditional copula probabilities (H-functions) to
    identify relative overvaluation/undervaluation between two assets:

      - H(u|v) >= entry_upper AND H(v|u) <= entry_lower → Long asset1, Short asset2
      - H(u|v) <= entry_lower AND H(v|u) >= entry_upper → Short asset1, Long asset2
      - Exit when both probabilities cross the mid-threshold (0.5)

    Trading Signal Enumeration (参考Krauss & Stübinger):
      - u相对于v被高估 → H(u|v)接近1时 → 做空u，做多v
      - u相对于v被低估 → H(u|v)接近0时 → 做多u，做空v

    Parameters
    ----------
    params : CopulaPairsParams
        Strategy configuration parameters.

    Attributes
    ----------
    copula : GaussianCopula or StudentTCopula
        Fitted copula model.
    dcc : DCCGARCHModel or None
        Fitted DCC-GARCH model (if use_dcc=True).

    Example
    -------
    >>> params = CopulaPairsParams(
    ...     copula_type="gaussian",
    ...     use_dcc=False,
    ...     entry_upper=0.95,
    ...     entry_lower=0.05,
    ... )
    >>> strategy = CopulaPairsStrategy(params=params)
    >>> # returns1, returns2 are numpy arrays of log returns
    >>> result = strategy.fit(returns1, returns2)
    >>> signals = strategy.generate_signals(returns1[-100:], returns2[-100:])
    >>> print(f"Signal counts: {np.sum(signals[0]!=0)}")
    """

    name: str = "copula_pairs"
    params: CopulaPairsParams

    def __init__(
        self,
        symbol1: str = "ASSET1",
        symbol2: str = "ASSET2",
        params: Optional[CopulaPairsParams] = None,
    ) -> None:
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.params = params or CopulaPairsParams()

        self.copula: Optional[Union[GaussianCopula, StudentTCopula]] = None
        self.dcc: Optional[DCCGARCHModel] = None
        self._fitted: bool = False
        self._copula_result: Optional[Dict[str, Any]] = None
        self._dcc_result: Optional[Dict[str, Any]] = None
        self._training_corr: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit the copula pairs strategy on training data.
        在训练数据上拟合Copula配对交易策略。

        Parameters
        ----------
        x, y : np.ndarray
            Training return series for asset 1 and asset 2 (same length).

        Returns
        -------
        dict
            Fit diagnostics including copula parameters and DCC parameters.
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < self.params.min_periods:
            raise ValueError(
                f"Need at least {self.params.min_periods} periods, got {len(x)}"
            )

        # ---- Transform to uniform margins ----
        u, v = to_uniform_margins(x, y, ecdf=self.params.ecdf_margins)

        # ---- Optionally compute training correlation ----
        self._training_corr = float(np.corrcoef(x, y)[0, 1])

        if self._training_corr < self.params.correlation_threshold:
            warnings.warn(
                f"Training correlation {self._training_corr:.3f} is below "
                f"threshold {self.params.correlation_threshold}. "
                "Signals may be unreliable."
            )

        # ---- Fit DCC-GARCH if requested ----
        if self.params.use_dcc:
            self.dcc = DCCGARCHModel()
            self._dcc_result = self.dcc.fit(x, y)

        # ---- Fit copula ----
        if self.params.copula_type == "gaussian":
            self.copula = GaussianCopula()
        elif self.params.copula_type == "student_t":
            self.copula = StudentTCopula()
        else:
            raise ValueError(
                f"Unknown copula type: {self.params.copula_type}. "
                "Use 'gaussian' or 'student_t'."
            )

        self._copula_result = self.copula.fit(u, v)
        self._fitted = True

        return {
            "copula_result": self._copula_result,
            "dcc_result": self._dcc_result,
            "training_correlation": self._training_corr,
            "n_training": len(x),
        }

    def compute_spread(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "h-condition",
    ) -> np.ndarray:
        """
        Compute the copula-based spread between the two assets.
        计算两个资产之间的基于Copula的价差。

        Parameters
        ----------
        x, y : np.ndarray
            Return series.
        method : str
            "h-condition": H(u|v) - 0.5 (default)
            "probability": C(u,v) - u*v
            "log-ratio": log(u/(1-u)) - log(v/(1-v))
            "hedge-ratio": x - β·y (requires DCC-GARCH, uses hedge ratio)

        Returns
        -------
        spread : np.ndarray
            Spread series (z-scored if method="h-condition").
        """
        if not self._fitted:
            raise ValueError("Strategy must be fitted before computing spread")

        if method == "hedge-ratio":
            if not self.params.use_dcc or self.dcc is None:
                raise ValueError("DCC-GARCH must be enabled for hedge-ratio spread")
            return self.dcc.compute_spread(x, y, method="hedge-ratio")

        u, v = to_uniform_margins(x, y, ecdf=self.params.ecdf_margins)

        if isinstance(self.copula, GaussianCopula):
            spread = self.copula.compute_spread(u, v, method=method, rho=self.copula.rho)
        else:
            spread = self.copula.compute_spread(
                u, v, method=method, rho=self.copula.rho, nu=self.copula.nu
            )

        return spread

    def generate_signals(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trading signals for both assets.
        为两个资产生成交易信号。

        Parameters
        ----------
        x, y : np.ndarray
            Return series for asset 1 and asset 2.

        Returns
        -------
        signal_x, signal_y, spread : tuple of np.ndarray
            signal_x: 1=long asset1, -1=short asset1, 0=flat
            signal_y: 1=long asset2, -1=short asset2, 0=flat
            spread: the computed spread series (z-scored)
        """
        if not self._fitted:
            raise ValueError("Strategy must be fitted before generating signals")

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        # Convert to uniform margins
        u, v = to_uniform_margins(x, y, ecdf=self.params.ecdf_margins)

        # Compute spread
        spread = self.compute_spread(x, y, method="h-condition")

        # Generate signals from conditional probabilities
        entry = (self.params.entry_upper, self.params.entry_lower)
        exit_ = (self.params.exit_upper, self.params.exit_lower)

        if isinstance(self.copula, GaussianCopula):
            sig_x, sig_y = self.copula.generate_signals(
                u, v,
                entry_thresholds=entry,
                exit_thresholds=exit_,
                rho=self.copula.rho,
            )
        else:
            sig_x, sig_y = self.copula.generate_signals(
                u, v,
                entry_thresholds=entry,
                exit_thresholds=exit_,
                rho=self.copula.rho,
                nu=self.copula.nu,
            )

        return sig_x, sig_y, spread

    def backtest(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fit on training data and generate signals + PnL on test data.
        在训练数据上拟合，在测试数据上生成信号和盈亏。

        Parameters
        ----------
        x_train, y_train : np.ndarray
            Training return series.
        x_test, y_test : np.ndarray
            Test return series.

        Returns
        -------
        dict
            Backtest results including signals, PnL, and performance metrics.
        """
        self.fit(x_train, y_train)

        sig_x_test, sig_y_test, spread_test = self.generate_signals(x_test, y_test)

        # Compute PnL
        pnl_x = sig_x_test[:-1] * x_test[1:]  # lag by 1 (signal at t → return at t+1)
        pnl_y = sig_y_test[:-1] * y_test[1:]

        total_pnl = pnl_x + pnl_y

        # Performance metrics
        sharpe = (
            np.mean(total_pnl) / (np.std(total_pnl) + 1e-10) * np.sqrt(252)
            if np.std(total_pnl) > 1e-10 else 0.0
        )
        cum_return = np.cumsum(total_pnl)
        max_dd = np.max(np.maximum.accumulate(cum_return) - cum_return)

        return {
            "signal_x": sig_x_test,
            "signal_y": sig_y_test,
            "spread": spread_test,
            "pnl": total_pnl,
            "cumulative_pnl": cum_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": np.sum(np.diff(sig_x_test) != 0) // 2,
            "copula_result": self._copula_result,
            "dcc_result": self._dcc_result,
            "training_correlation": self._training_corr,
        }

    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters and fit diagnostics."""
        return {
            "params": {
                "copula_type": self.params.copula_type,
                "entry_upper": self.params.entry_upper,
                "entry_lower": self.params.entry_lower,
                "exit_upper": self.params.exit_upper,
                "exit_lower": self.params.exit_lower,
                "use_dcc": self.params.use_dcc,
                "min_periods": self.params.min_periods,
                "ecdf_margins": self.params.ecdf_margins,
                "correlation_threshold": self.params.correlation_threshold,
            },
            "fitted": self._fitted,
            "copula_result": self._copula_result,
            "dcc_result": self._dcc_result,
            "training_correlation": self._training_corr,
        }

    def __repr__(self) -> str:
        return (
            f"CopulaPairsStrategy(copula={self.params.copula_type}, "
            f"use_dcc={self.params.use_dcc}, "
            f"fitted={self._fitted})"
        )


# ----------------------------------------------------------------------
# Utility: Select best copula via AIC/BIC
# ----------------------------------------------------------------------


def select_best_copula(
    u: np.ndarray,
    v: np.ndarray,
) -> Dict[str, Any]:
    """
    Select the best copula family (Gaussian vs Student-t) via AIC/BIC.
    通过AIC/BIC选择最佳Copula族（高斯 vs Student-t）。

    Parameters
    ----------
    u, v : np.ndarray
        Uniform marginals.

    Returns
    -------
    dict
        Selection result with both models' diagnostics.
    """
    gc = GaussianCopula()
    tc = StudentTCopula()

    result_gc = gc.fit(u, v)
    result_tc = tc.fit(u, v)

    best = "gaussian" if result_gc["aic"] <= result_tc["aic"] else "student_t"
    best_result = result_gc if best == "gaussian" else result_tc

    return {
        "gaussian_result": result_gc,
        "student_t_result": result_tc,
        "best_copula": best,
        "best_result": best_result,
        "aic_gaussian": result_gc["aic"],
        "aic_student_t": result_tc["aic"],
        "bic_gaussian": result_gc["bic"],
        "bic_student_t": result_tc["bic"],
    }
