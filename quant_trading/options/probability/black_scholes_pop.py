"""
Black-Scholes functions for Probability of Profit calculations.

These are simplified BS functions used specifically in Monte Carlo simulations
for POP calculations. They differ from the full BlackScholes class in pricing/.

Key differences from pricing/black_scholes.py:
- These are standalone functions (not a class)
- Designed for repeated calls within Monte Carlo loops
- Use math.erf directly for speed
- Handle edge cases at expiration
"""

from math import log, sqrt, exp, erf


def black_scholes_call(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Calculate Black-Scholes Call price.

    Args:
        s: Spot price (underlying price)
        k: Strike price
        r: Risk-free rate (decimal, e.g., 0.05 for 5%)
        t: Time to expiration (years)
        sigma: Volatility (decimal, e.g., 0.20 for 20%)

    Returns:
        Call option price
    """
    if t == 0:
        if s > k:
            return s - k
        elif s < k:
            return 0
        else:
            return 0

    d1 = (log(s / k) + (r + (1 / 2) * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-r * t) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)

    return c


def black_scholes_put(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Calculate Black-Scholes Put price.

    Args:
        s: Spot price (underlying price)
        k: Strike price
        r: Risk-free rate (decimal, e.g., 0.05 for 5%)
        t: Time to expiration (years)
        sigma: Volatility (decimal, e.g., 0.20 for 20%)

    Returns:
        Put option price
    """
    if t == 0:
        if s / k > 1:
            return 0
        elif s / k < 1:
            return k - s
        else:
            return 0

    d1 = (log(s / k) + (r + (1 / 2) * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-r * t) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)
    p = k * exp(-r * t) - s + c

    return p


__all__ = ["black_scholes_call", "black_scholes_put"]
