"""
Black-Scholes options pricing model with Greeks.

This module provides complete Black-Scholes pricing for European call/put options
along with all Greek letter sensitivities (delta, vega, gamma, theta, rho).

Based on the Optiver competition-winning market-making algorithm.
"""

from scipy import stats
import numpy as np


_norm_cdf = stats.norm(0, 1).cdf
_norm_pdf = stats.norm(0, 1).pdf


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 parameter for Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 parameter for Black-Scholes."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_value(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the fair value of a European call option using Black-Scholes.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Fair present value of the call option.
    """
    return S * _norm_cdf(_d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * _norm_cdf(
        _d2(S, K, T, r, sigma)
    )


def put_value(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the fair value of a European put option using Black-Scholes.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Fair present value of the put option.
    """
    return np.exp(-r * T) * K * _norm_cdf(-_d2(S, K, T, r, sigma)) - S * _norm_cdf(
        -_d1(S, K, T, r, sigma)
    )


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the delta of a European call option.

    Delta is the first derivative of the option value with respect to the
    underlying price. It represents the sensitivity of the option price
    to changes in the underlying stock price.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Delta of the call option (between 0 and 1).
    """
    return _norm_cdf(_d1(S, K, T, r, sigma))


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the delta of a European put option.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Delta of the put option (between -1 and 0).
    """
    return call_delta(S, K, T, r, sigma) - 1


def call_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the vega of a European call option.

    Vega is the derivative of the option value with respect to volatility.
    It measures sensitivity to changes in implied volatility.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Vega of the call option (positive, meaning call value increases with volatility).
    """
    return S * _norm_pdf(_d1(S, K, T, r, sigma)) * np.sqrt(T)


def put_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the vega of a European put option.

    Vega is the same for calls and puts (both positive).

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Vega of the put option (positive).
    """
    return call_vega(S, K, T, r, sigma)


def call_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the gamma of a European call option.

    Gamma is the second derivative of the option value with respect to the
    underlying price (or the first derivative of delta). It measures the
    rate of change of delta with respect to the underlying price.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Gamma of the option (same for calls and puts).
    """
    return _norm_pdf(_d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def put_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the gamma of a European put option.

    Gamma is the same for calls and puts.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Gamma of the option (same for calls and puts).
    """
    return call_gamma(S, K, T, r, sigma)


def call_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theta of a European call option.

    Theta is the derivative of the option value with respect to time.
    It measures the rate of time decay - how much value the option loses
    as time passes (all else being equal).

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Theta of the call option (typically negative, representing time decay).
    """
    term1 = -S * _norm_pdf(_d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * _norm_cdf(_d2(S, K, T, r, sigma))
    return term1 - term2


def put_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theta of a European put option.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Theta of the put option (typically negative).
    """
    term1 = -S * _norm_pdf(_d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * _norm_cdf(-_d2(S, K, T, r, sigma))
    return term1 + term2


def call_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the rho of a European call option.

    Rho is the derivative of the option value with respect to the interest rate.
    It measures sensitivity to changes in the risk-free rate.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Rho of the call option (positive for calls).
    """
    return K * T * np.exp(-r * T) * _norm_cdf(_d2(S, K, T, r, sigma))


def put_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the rho of a European put option.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.

    Returns
    -------
    float
        Rho of the put option (negative for puts).
    """
    return -K * T * np.exp(-r * T) * _norm_cdf(-_d2(S, K, T, r, sigma))


def get_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """
    Calculate all Greeks for a European option in a single call.

    Parameters
    ----------
    S : float
        Current price of the underlying stock.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        Dictionary containing 'price', 'delta', 'vega', 'gamma', 'theta', 'rho'.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if option_type == "call":
        price = call_value(S, K, T, r, sigma)
        delta = call_delta(S, K, T, r, sigma)
        vega = call_vega(S, K, T, r, sigma)
        gamma = call_gamma(S, K, T, r, sigma)
        theta = call_theta(S, K, T, r, sigma)
        rho = call_rho(S, K, T, r, sigma)
    else:
        price = put_value(S, K, T, r, sigma)
        delta = put_delta(S, K, T, r, sigma)
        vega = put_vega(S, K, T, r, sigma)
        gamma = put_gamma(S, K, T, r, sigma)
        theta = put_theta(S, K, T, r, sigma)
        rho = put_rho(S, K, T, r, sigma)

    return {
        "price": price,
        "delta": delta,
        "vega": vega,
        "gamma": gamma,
        "theta": theta,
        "rho": rho,
    }
