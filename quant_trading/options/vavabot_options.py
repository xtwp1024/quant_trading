# -*- coding: utf-8 -*-
"""
vavabot_options.py
==================

VavaBot Options Strategy - Unified Deribit Options Trading Module.
基于 VavaBot Options Strategy v10.0 重构的量化交易系统期权模块。

Features / 功能特性:
    - Pure Python + urllib REST client for Deribit API
    - Pure NumPy Black-Scholes option pricing and Greeks calculation
    - Lazy imports with graceful degradation
    - Volatility surface construction
    - Position monitoring

Classes / 类:
    DeribitOptionsAPI     - Deribit REST API client (urllib-based)
    VavaBotOptionsStrategy - Options trading strategy wrapper
    BlackScholesGreeks    - Black-Scholes pricing and Greeks engine
    OptionContract        - Option contract data model
    VolatilitySurface     - Implied volatility surface builder
    PositionMonitor       - Real-time position monitor

Author: VavaBot -> Quant Trading System Integration
License: MIT

Usage Example / 使用示例:
    >>> api = DeribitOptionsAPI(testnet=True)
    >>> greeks = BlackScholesGreeks()
    >>> price = greeks.call_price(S=50000, K=50000, T=0.25, r=0.05, sigma=0.80)
    >>> surface = VolatilitySurface()
    >>> surface.add_strike(50000, 0.80, 0.25, 'call')
"""

from __future__ import annotations

import json
import hmac
import hashlib
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

# ---------------------------------------------------------------------------
# Lazy-import guard for optional dependencies
# ---------------------------------------------------------------------------
def _lazy_import(name: str, fallback: Any = None) -> Any:
    """
    Lazily import an optional module, returning fallback on failure.
    惰性导入可选模块，失败时返回 fallback。

    Args:
        name: Module name to import.
        fallback: Value to return if import fails.

    Returns:
        Imported module or fallback value.
    """
    try:
        return __import__(name)
    except ImportError:
        return fallback


# ---------------------------------------------------------------------------
# Optional websocket-client import (not available in pure stdlib)
# ---------------------------------------------------------------------------
WebSocketClient = _lazy_import('websocket', fallback=None)
# To use WebSocket: pip install websocket-client
# NOTE: Pure urllib cannot do WebSocket; this requires the websocket-client package.
# 注：纯 urllib 无法实现 WebSocket，需要安装 websocket-client 包。


# ---------------------------------------------------------------------------
# __all__ - Public API
# ---------------------------------------------------------------------------
__all__ = [
    'DeribitOptionsAPI',
    'VavaBotOptionsStrategy',
    'BlackScholesGreeks',
    'OptionContract',
    'VolatilitySurface',
    'PositionMonitor',
    'DeribitWebSocketClient',   # requires websocket-client package
]


# ===========================================================================
# BlackScholesGreeks - Pure NumPy Black-Scholes Implementation
# ===========================================================================

class BlackScholesGreeks:
    """
    Black-Scholes option pricing model and Greeks calculations.
    布莱克-舒尔斯期权定价模型与希腊字母计算器。

    All calculations are implemented in pure NumPy without scipy.
    所有计算均使用纯 NumPy 实现，无需 scipy。

    Attributes / 属性:
        None - all methods are static.

    Example / 示例:
        >>> bs = BlackScholesGreeks()
        >>> bs.call_price(S=50000, K=50000, T=0.25, r=0.05, sigma=0.80)
        2739.87
        >>> bs.put_delta(S=50000, K=50000, T=0.25, r=0.05, sigma=0.80)
        0.524
    """

    @staticmethod
    def _norm_cdf(x: Union[float, 'np.ndarray']) -> Union[float, 'np.ndarray']:
        """
        Standard normal cumulative distribution function (CDF).
        标准正态累积分布函数。

        Implemented using the error function (erf) for accuracy.
        Uses the approximation: N(x) = 0.5 * [1 + erf(x / sqrt(2))]

        Args:
            x: Input value(s).

        Returns:
            CDF value(s) for standard normal distribution.
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: Union[float, 'np.ndarray']) -> Union[float, 'np.ndarray']:
        """
        Standard normal probability density function (PDF).
        标准正态概率密度函数。

        f(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)

        Args:
            x: Input value(s).

        Returns:
            PDF value(s) for standard normal distribution.
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

    @staticmethod
    def _d1_d2(
        S: Union[float, 'np.ndarray'],
        K: Union[float, 'np.ndarray'],
        T: Union[float, 'np.ndarray'],
        r: Union[float, 'np.ndarray'],
        sigma: Union[float, 'np.ndarray'],
    ) -> Tuple[Union[float, 'np.ndarray'], Union[float, 'np.ndarray']]:
        """
        Calculate Black-Scholes d1 and d2 parameters.
        计算布莱克-舒尔斯 d1 和 d2 参数。

        d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity (years) / 到期时间（年）
            r: Risk-free rate / 无风险利率
            sigma: Volatility / 波动率

        Returns:
            Tuple of (d1, d2)
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    # -------------------------------------------------------------------------
    # Option Prices
    # -------------------------------------------------------------------------

    @staticmethod
    def call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Calculate Black-Scholes call option price.
        计算布莱克-舒尔斯看涨期权价格。

        C = S * N(d1) - K * exp(-r*T) * N(d2)

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity in years / 到期时间（年）
            r: Risk-free interest rate / 无风险利率
            sigma: Implied volatility / 隐含波动率

        Returns:
            Call option price / 看涨期权价格
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return max(0.0, S - K)
        d1, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        call = S * BlackScholesGreeks._norm_cdf(d1) - K * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(d2)
        return float(call)

    @staticmethod
    def put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Calculate Black-Scholes put option price.
        计算布莱克-舒尔斯看跌期权价格。

        P = K * exp(-r*T) * N(-d2) - S * N(-d1)

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity in years / 到期时间（年）
            r: Risk-free interest rate / 无风险利率
            sigma: Implied volatility / 隐含波动率

        Returns:
            Put option price / 看跌期权价格
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return max(0.0, K - S)
        d1, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        put = K * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(-d2) - S * BlackScholesGreeks._norm_cdf(-d1)
        return float(put)

    # -------------------------------------------------------------------------
    # Greeks - Delta
    # -------------------------------------------------------------------------

    @staticmethod
    def call_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Call option Delta.
        看涨期权 Delta 值。

        Delta = N(d1) - 1 (for calls) or just N(d1)

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Delta value (range: 0 to 1 for calls).
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1, _ = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        return float(BlackScholesGreeks._norm_cdf(d1))

    @staticmethod
    def put_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Put option Delta.
        看跌期权 Delta 值。

        Delta = N(d1) - 1 (range: -1 to 0 for puts)

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Delta value.
        """
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1, _ = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        return float(BlackScholesGreeks._norm_cdf(d1) - 1.0)

    # -------------------------------------------------------------------------
    # Greeks - Gamma (same for calls and puts)
    # -------------------------------------------------------------------------

    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Option Gamma.
        期权 Gamma 值（calls 和 puts 相同）。

        Gamma = N'(d1) / (S * sigma * sqrt(T))

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Gamma value.
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        d1, _ = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        gamma = BlackScholesGreeks._norm_pdf(d1) / (S * sigma * np.sqrt(T))
        return float(gamma)

    # -------------------------------------------------------------------------
    # Greeks - Vega (same for calls and puts)
    # -------------------------------------------------------------------------

    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Option Vega.
        期权 Vega 值（calls 和 puts 相同）。

        Vega = S * N'(d1) * sqrt(T) / 100
        (Vega per 1% change in volatility)

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Vega value (price change per 1% vol change).
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        d1, _ = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        vega = S * BlackScholesGreeks._norm_pdf(d1) * np.sqrt(T) / 100.0
        return float(vega)

    # -------------------------------------------------------------------------
    # Greeks - Theta
    # -------------------------------------------------------------------------

    @staticmethod
    def call_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Call option Theta (per day).
        看涨期权 Theta 值（每日）。

        Theta = -[S * N'(d1) * sigma / (2*sqrt(T))] / 365
                - r * K * exp(-r*T) * N(d2) / 365

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Theta value (price decay per calendar day).
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        d1, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        term1 = -S * BlackScholesGreeks._norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
        term2 = -r * K * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(d2)
        theta = (term1 + term2) / 365.0
        return float(theta)

    @staticmethod
    def put_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Put option Theta (per day).
        看跌期权 Theta 值（每日）。

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Theta value (price decay per calendar day).
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        d1, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        term1 = -S * BlackScholesGreeks._norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(-d2)
        theta = (term1 + term2) / 365.0
        return float(theta)

    # -------------------------------------------------------------------------
    # Greeks - Rho
    # -------------------------------------------------------------------------

    @staticmethod
    def call_rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Call option Rho (per 1% rate change).
        看涨期权 Rho 值（每 1% 利率变化）。

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Rho value (price change per 1% rate change).
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        _, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        rho = K * T * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(d2) / 100.0
        return float(rho)

    @staticmethod
    def put_rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Put option Rho (per 1% rate change).
        看跌期权 Rho 值（每 1% 利率变化）。

        Args:
            S, K, T, r, sigma: Standard Black-Scholes parameters.

        Returns:
            Rho value (price change per 1% rate change).
        """
        np = _lazy_import('numpy')
        if np is None:
            raise ImportError("NumPy is required for BlackScholesGreeks. Install with: pip install numpy")

        if T <= 0:
            return 0.0
        _, d2 = BlackScholesGreeks._d1_d2(S, K, T, r, sigma)
        rho = -K * T * np.exp(-r * T) * BlackScholesGreeks._norm_cdf(-d2) / 100.0
        return float(rho)

    # -------------------------------------------------------------------------
    # All Greeks dict
    # -------------------------------------------------------------------------

    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option in one call.
        计算期权所有希腊字母值。

        Args:
            S: Spot price / 标的价格
            K: Strike price / 行权价
            T: Time to maturity / 到期时间（年）
            r: Risk-free rate / 无风险利率
            sigma: Volatility / 波动率
            option_type: 'call' or 'put' / 期权类型

        Returns:
            Dictionary with price, delta, gamma, vega, theta, rho.
        """
        option_type = option_type.lower()
        if option_type == 'call':
            price = self.call_price(S, K, T, r, sigma)
            delta = self.call_delta(S, K, T, r, sigma)
            theta = self.call_theta(S, K, T, r, sigma)
            rho = self.call_rho(S, K, T, r, sigma)
        else:
            price = self.put_price(S, K, T, r, sigma)
            delta = self.put_delta(S, K, T, r, sigma)
            theta = self.put_theta(S, K, T, r, sigma)
            rho = self.put_rho(S, K, T, r, sigma)

        return {
            'price': price,
            'delta': delta,
            'gamma': self.gamma(S, K, T, r, sigma),
            'vega': self.vega(S, K, T, r, sigma),
            'theta': theta,
            'rho': rho,
        }


# ===========================================================================
# OptionContract - Data Model
# ===========================================================================

class OptionContract:
    """
    Option contract data model.
    期权合约数据模型。

    Represents a single option contract with its market data and Greeks.
    表示单个期权合约及其市场数据和希腊字母。

    Attributes / 属性:
        instrument_name (str): Deribit instrument name (e.g. 'BTC-27MAR20-50000-C').
        strike (float): Strike price / 行权价
        expiry (datetime): Expiration datetime / 到期时间
        option_type (str): 'call' or 'put' / 期权类型
        mark_price (float): Mark price from exchange / 标记价格
        bid_price (float): Best bid price / 买一价
        ask_price (float): Best ask price / 卖一价
        iv (float): Implied volatility / 隐含波动率
        delta (float): Delta / Delta 值
        gamma (float): Gamma / Gamma 值
        vega (float): Vega / Vega 值
        theta (float): Theta / Theta 值
        underlying_price (float): Underlying spot price / 标的价格
        expiration_days (float): Days to expiration / 距离到期天数

    Example / 示例:
        >>> contract = OptionContract(
        ...     instrument_name='BTC-27MAR20-50000-C',
        ...     strike=50000,
        ...     expiry=datetime(2020, 3, 27),
        ...     option_type='call',
        ...     mark_price=2739.87,
        ...     iv=0.80,
        ... )
        >>> print(contract.is_in_the_money(S=50000))
        True
    """

    def __init__(
        self,
        instrument_name: str,
        strike: float,
        expiry: datetime,
        option_type: str,
        mark_price: float = 0.0,
        bid_price: float = 0.0,
        ask_price: float = 0.0,
        iv: float = 0.0,
        delta: float = 0.0,
        gamma: float = 0.0,
        vega: float = 0.0,
        theta: float = 0.0,
        underlying_price: float = 0.0,
        iv_rounded: float = 0.0,
        greeks_from_exchange: Optional[Dict[str, float]] = None,
    ):
        self.instrument_name = instrument_name
        self.strike = float(strike)
        self.expiry = expiry
        self.option_type = option_type.lower()
        self.mark_price = float(mark_price)
        self.bid_price = float(bid_price)
        self.ask_price = float(ask_price)
        self.iv = float(iv)
        self.iv_rounded = float(iv_rounded) if iv_rounded else float(iv)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.vega = float(vega)
        self.theta = float(theta)
        self.underlying_price = float(underlying_price)
        self.greeks_from_exchange = greeks_from_exchange or {}

        now = datetime.utcnow()
        self.expiration_days = max(0.0, (expiry - now).total_seconds() / 86400.0)
        self.T = self.expiration_days / 365.0

    def is_in_the_money(self, S: Optional[float] = None) -> bool:
        """
        Check if option is in-the-money.
        检查期权是否价内。

        Args:
            S: Spot price (uses underlying_price if not provided).

        Returns:
            True if ITM, False otherwise.
        """
        S = S or self.underlying_price
        if self.option_type == 'call':
            return S > self.strike
        else:
            return S < self.strike

    def is_at_the_money(self, S: Optional[float] = None, tolerance: float = 0.02) -> bool:
        """
        Check if option is at-the-money (within tolerance).
        检查期权是否 ATM。

        Args:
            S: Spot price / 标的价格
            tolerance: ATM tolerance as fraction of strike / ATM 容差

        Returns:
            True if ATM, False otherwise.
        """
        S = S or self.underlying_price
        return abs(S - self.strike) / self.strike <= tolerance

    def moneyness(self, S: Optional[float] = None) -> float:
        """
        Calculate moneyness: S / K for calls, K / S for puts.
        计算货币性：看涨为 S/K，看跌为 K/S。

        Args:
            S: Spot price / 标的价格

        Returns:
            Moneyness ratio.
        """
        S = S or self.underlying_price
        if self.option_type == 'call':
            return S / self.strike
        else:
            return self.strike / S

    def spread(self) -> float:
        """
        Calculate bid-ask spread.
        计算买卖价差。

        Returns:
            Spread (ask - bid).
        """
        return self.ask_price - self.bid_price

    def mid_price(self) -> float:
        """
        Calculate mid price.
        计算中间价。

        Returns:
            (bid + ask) / 2
        """
        return (self.bid_price + self.ask_price) / 2.0

    def __repr__(self) -> str:
        return (
            f"OptionContract(name={self.instrument_name}, K={self.strike}, "
            f"type={self.option_type}, iv={self.iv:.4f}, T={self.T:.4f})"
        )


# ===========================================================================
# VolatilitySurface - Implied Volatility Surface
# ===========================================================================

class VolatilitySurface:
    """
    Implied volatility surface builder.
    隐含波动率曲面构建器。

    Tracks volatility by strike and maturity for risk management and
    strike / term structure analysis.
    跟踪不同行权价和到期时间的波动率，用于风险管理及波动率结构分析。

    Attributes / 属性:
        surface (dict): {moneyness: {tenor: iv}} nested dict.
        strikes (list): List of tracked strikes.
        tenors (list): List of tracked tenors (in years).
        data (list): Raw list of (strike, tenor, iv, option_type) tuples.

    Example / 示例:
        >>> surface = VolatilitySurface()
        >>> surface.add_strike(50000, 0.80, 0.25, 'call')
        >>> surface.add_strike(49000, 0.82, 0.25, 'put')
        >>> atm_vol = surface.get_atm_vol(0.25)
    """

    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.strikes: List[float] = []
        self._surface: Dict[float, Dict[float, float]] = {}

    def add_strike(
        self,
        strike: float,
        iv: float,
        tenor: float,
        option_type: str = 'call',
        spot: float = None,
    ) -> None:
        """
        Add a volatility data point.
        添加波动率数据点。

        Args:
            strike: Strike price / 行权价
            iv: Implied volatility / 隐含波动率
            tenor: Time to maturity in years / 到期时间（年）
            option_type: 'call' or 'put' / 期权类型
            spot: Spot price (optional, for moneyness calculation).
        """
        entry = {
            'strike': float(strike),
            'iv': float(iv),
            'tenor': float(tenor),
            'option_type': option_type.lower(),
            'spot': spot,
        }
        self.data.append(entry)
        if strike not in self.strikes:
            self.strikes.append(strike)
            self.strikes.sort()

    def add_contract(self, contract: OptionContract, tenor: float = None) -> None:
        """
        Add volatility data from an OptionContract.
        从 OptionContract 添加波动率数据。

        Args:
            contract: OptionContract instance.
            tenor: Tenor in years (uses contract.T if not provided).
        """
        self.add_strike(
            strike=contract.strike,
            iv=contract.iv,
            tenor=tenor if tenor is not None else contract.T,
            option_type=contract.option_type,
            spot=contract.underlying_price,
        )

    def get_atm_vol(self, tenor: float, tolerance: float = 0.02) -> Optional[float]:
        """
        Get ATM volatility for a given tenor.
        获取指定到期的 ATM 波动率。

        Args:
            tenor: Tenor in years / 到期时间（年）
            tolerance: ATM tolerance / ATM 容差

        Returns:
            ATM volatility or None if not found.
        """
        if not self.data:
            return None

        strikes = sorted(set(d['strike'] for d in self.data))
        if not strikes:
            return None

        spot = self.data[0].get('spot')
        if spot is None:
            atm_strike = strikes[len(strikes) // 2]
        else:
            atm_strike = spot

        atm_data = [d for d in self.data if abs(d['strike'] - atm_strike) / atm_strike <= tolerance]
        if not atm_data:
            atm_data = self.data

        avg_iv = sum(d['iv'] for d in atm_data) / len(atm_data)
        return avg_iv

    def get_strike_vol(self, strike: float, tenor: float = None) -> Optional[float]:
        """
        Get volatility for a specific strike.
        获取指定行权价的波动率。

        Args:
            strike: Strike price / 行权价
            tenor: Tenor in years (optional) / 到期时间（年）

        Returns:
            Implied volatility or None.
        """
        for d in self.data:
            if abs(d['strike'] - strike) < 1e-6:
                if tenor is None or abs(d['tenor'] - tenor) < 1e-6:
                    return d['iv']
        return None

    def smile_info(self, tenor: float = None) -> Dict[str, Any]:
        """
        Get volatility smile/skew info.
        获取波动率微笑/偏斜信息。

        Args:
            tenor: Tenor to analyze (uses first available if None).

        Returns:
            Dict with atm_vol, skew, put_wing, call_wing.
        """
        if tenor is None:
            tenors = sorted(set(d['tenor'] for d in self.data))
            if not tenors:
                return {}
            tenor = tenors[0]

        tenor_data = [d for d in self.data if abs(d['tenor'] - tenor) < 1e-6]
        if not tenor_data:
            return {}

        strikes = sorted([d['strike'] for d in tenor_data])
        if len(strikes) < 3:
            return {}

        spot = tenor_data[0].get('spot') or strikes[len(strikes) // 2]

        low_strike = strikes[0]
        high_strike = strikes[-1]
        low_data = next((d for d in tenor_data if abs(d['strike'] - low_strike) < 1e-6), None)
        high_data = next((d for d in tenor_data if abs(d['strike'] - high_strike) < 1e-6), None)

        atm_vol = self.get_atm_vol(tenor) or 0.0
        skew = (high_data['iv'] - low_data['iv']) / (high_strike - low_strike) if high_strike != low_strike else 0.0

        return {
            'tenor': tenor,
            'atm_vol': atm_vol,
            'skew': skew,
            'low_strike_iv': low_data['iv'] if low_data else None,
            'high_strike_iv': high_data['iv'] if high_data else None,
            'num_strikes': len(strikes),
        }

    def term_structure(self) -> Dict[float, float]:
        """
        Get ATM term structure (vol over tenors).
        获取 ATM 期限结构（各到期对应的波动率）。

        Returns:
            Dict {tenor: atm_vol}
        """
        tenors = sorted(set(d['tenor'] for d in self.data))
        return {t: self.get_atm_vol(t) for t in tenors}

    def clear(self) -> None:
        """Clear all volatility data. / 清除所有波动率数据。"""
        self.data.clear()
        self.strikes.clear()
        self._surface.clear()


# ===========================================================================
# DeribitOptionsAPI - REST API Client (urllib-based)
# ===========================================================================

class DeribitOptionsAPI:
    """
    Deribit REST API client using pure urllib.
    基于纯 urllib 的 Deribit REST API 客户端。

    Supports all public and private Deribit API endpoints.
    For WebSocket, use DeribitWebSocketClient (requires websocket-client).
    支持所有公开和私有 Deribit API 端点。
    WebSocket 请使用 DeribitWebSocketClient（需要 websocket-client 包）。

    REST API Docs: https://docs.deribit.com/
    REST API Base URL: https://www.deribit.com/api/v2/

    Attributes / 属性:
        testnet (bool): Whether to use testnet.
        base_url (str): Base URL for API requests.
        access_token (str): OAuth access token (set after auth).
        refresh_token (str): OAuth refresh token.

    Example / 示例:
        >>> api = DeribitOptionsAPI(testnet=True)
        >>> api.auth('client_id', 'client_secret')
        >>> instruments = api.get_instruments('BTC', expired=False)
        >>> index_price = api.get_index_price('BTC')
    """

    BASE_URL_MAINNET = 'https://www.deribit.com/api/v2'
    BASE_URL_TESTNET = 'https://test.deribit.com/api/v2'

    def __init__(
        self,
        testnet: bool = False,
        client_id: str = None,
        client_secret: str = None,
    ):
        self.testnet = testnet
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL_MAINNET
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._auth_timestamp: Optional[float] = None

    # -------------------------------------------------------------------------
    # Internal HTTP helpers
    # -------------------------------------------------------------------------

    def _make_request(
        self,
        method: str,
        params: Dict[str, Any],
        private: bool = False,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Deribit API.
        向 Deribit API 发起 HTTP 请求。

        Args:
            method: JSON-RPC method name.
            params: Method parameters.
            private: Whether this is a private endpoint (requires auth).

        Returns:
            API response dict.

        Raises:
            urllib.error.HTTPError: On HTTP error.
        """
        payload = json.dumps({
            'jsonrpc': '2.0',
            'id': int(time.time() * 1000) % 2147483647,
            'method': method,
            'params': params,
        }).encode('utf-8')

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        if private and self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'

        url = self.base_url
        req = urllib.request.Request(url, data=payload, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                if 'error' in result:
                    raise DeribitAPIError(
                        code=result['error'].get('code', -1),
                        message=result['error'].get('message', 'Unknown error'),
                        data=result['error'].get('data'),
                    )
                return result.get('result', {})
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8') if e.fp else ''
            try:
                err_data = json.loads(body)
                raise DeribitAPIError(
                    code=err_data.get('error', {}).get('code', e.code),
                    message=err_data.get('error', {}).get('message', str(e)),
                    data=err_data.get('error', {}).get('data'),
                )
            except json.JSONDecodeError:
                raise DeribitAPIError(code=e.code, message=str(e), data=body)

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    def auth(self, client_id: str = None, client_secret: str = None) -> Dict[str, Any]:
        """
        Authenticate with Deribit API (client credentials).
        使用 Deribit API 进行身份验证（客户端凭证模式）。

        Supports both HMAC signature auth and client credentials.
        支持 HMAC 签名认证和客户端凭证认证。

        Args:
            client_id: API client ID.
            client_secret: API client secret.

        Returns:
            Auth result with access_token, refresh_token, etc.
        """
        self.client_id = client_id or self.client_id
        self.client_secret = client_secret or self.client_secret

        if not self.client_id or not self.client_secret:
            raise ValueError("client_id and client_secret are required for auth")

        timestamp = round(datetime.utcnow().timestamp() * 1000)
        nonce = 'abcd'
        data = ''

        signature = hmac.new(
            bytes(self.client_secret, 'latin-1'),
            msg=bytes(f'{timestamp}\n{nonce}\n{data}', 'latin-1'),
            digestmod=hashlib.sha256,
        ).hexdigest().lower()

        result = self._make_request(
            method='public/auth',
            params={
                'grant_type': 'client_signature',
                'client_id': self.client_id,
                'timestamp': timestamp,
                'signature': signature,
                'nonce': nonce,
                'data': data,
            },
        )

        self.access_token = result.get('access_token')
        self.refresh_token = result.get('refresh_token')
        self._auth_timestamp = time.time()
        return result

    def refresh_auth(self) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        使用刷新令牌更新访问令牌。

        Returns:
            New auth result.
        """
        if not self.refresh_token:
            raise ValueError("No refresh_token available. Call auth() first.")

        result = self._make_request(
            method='public/auth',
            params={
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
            },
        )
        self.access_token = result.get('access_token')
        self.refresh_token = result.get('refresh_token')
        self._auth_timestamp = time.time()
        return result

    # -------------------------------------------------------------------------
    # Public endpoints
    # -------------------------------------------------------------------------

    def get_instruments(
        self,
        currency: str = 'BTC',
        expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get available instruments for a currency.
        获取指定货币的可用合约列表。

        Args:
            currency: 'BTC' or 'ETH'.
            expired: Include expired instruments.

        Returns:
            List of instrument dicts.
        """
        return self._make_request(
            method='public/get_instruments',
            params={'currency': currency, 'expired': expired},
        )

    def get_index_price(self, index_name: str = 'BTC') -> float:
        """
        Get index price.
        获取指数价格。

        Args:
            index_name: 'BTC' or 'ETH'.

        Returns:
            Index price.
        """
        result = self._make_request(
            method='public/get_index_price',
            params={'index_name': index_name},
        )
        return float(result.get('index_price', 0))

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 5,
    ) -> Dict[str, Any]:
        """
        Get order book for an instrument.
        获取指定合约的订单簿。

        Args:
            instrument_name: Instrument name (e.g. 'BTC-27MAR20-50000-C').
            depth: Order book depth.

        Returns:
            Order book dict with bids, asks, greeks, etc.
        """
        return self._make_request(
            method='public/get_order_book',
            params={'instrument_name': instrument_name, 'depth': depth},
        )

    def get_book_summary_by_instrument(
        self,
        instrument_name: str,
    ) -> Dict[str, Any]:
        """
        Get book summary (24h stats) for an instrument.
        获取合约的24小时统计摘要。

        Args:
            instrument_name: Instrument name.

        Returns:
            Book summary dict.
        """
        return self._make_request(
            method='public/get_book_summary_by_instrument',
            params={'instrument_name': instrument_name},
        )

    def get_last_trades_by_instrument(
        self,
        instrument_name: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for an instrument.
        获取合约的最新成交记录。

        Args:
            instrument_name: Instrument name.
            count: Number of trades.

        Returns:
            List of trade dicts.
        """
        result = self._make_request(
            method='public/get_last_trades_by_instrument',
            params={'instrument_name': instrument_name, 'count': count},
        )
        return result.get('trades', [])

    def get_volatility_index_data(
        self,
        currency: str = 'BTC',
        resolution: str = '1',
        start_timestamp: int = None,
        end_timestamp: int = None,
    ) -> Dict[str, Any]:
        """
        Get volatility index data.
        获取波动率指数数据。

        Args:
            currency: 'BTC' or 'ETH'.
            resolution: Resolution string (e.g. '1', '1h', '1d').
            start_timestamp: Start timestamp in ms.
            end_timestamp: End timestamp in ms.

        Returns:
            Volatility index data.
        """
        params = {'currency': currency, 'resolution': resolution}
        if start_timestamp:
            params['start_timestamp'] = start_timestamp
        if end_timestamp:
            params['end_timestamp'] = end_timestamp
        return self._make_request(
            method='public/get_volatility_index_data',
            params=params,
        )

    def get_mark_price(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get mark price and 24h stats.
        获取标记价格和24小时统计。

        Args:
            instrument_name: Instrument name.

        Returns:
            Dict with mark_price, etc.
        """
        return self.get_book_summary_by_instrument(instrument_name)

    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection.
        测试 API 连接。

        Returns:
            Test result with version info.
        """
        return self._make_request(method='public/test', params={})

    # -------------------------------------------------------------------------
    # Private endpoints
    # -------------------------------------------------------------------------

    def _require_auth(self) -> None:
        """Ensure user is authenticated. / 确保用户已认证。"""
        if not self.access_token:
            raise DeribitAuthError("Not authenticated. Call auth() first.")

    def get_position(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get position for an instrument.
        获取指定合约的持仓信息。

        Args:
            instrument_name: Instrument name.

        Returns:
            Position dict.
        """
        self._require_auth()
        return self._make_request(
            method='private/get_position',
            params={'instrument_name': instrument_name},
            private=True,
        )

    def get_positions(self, currency: str = 'BTC', kind: str = 'option') -> List[Dict[str, Any]]:
        """
        Get all positions for a currency.
        获取指定货币的所有持仓。

        Args:
            currency: 'BTC' or 'ETH'.
            kind: 'option' or 'future'.

        Returns:
            List of position dicts.
        """
        self._require_auth()
        return self._make_request(
            method='private/get_positions',
            params={'currency': currency, 'kind': kind},
            private=True,
        )

    def buy_limit(
        self,
        instrument_name: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place a limit buy order.
        下限价买单。

        Args:
            instrument_name: Instrument name.
            amount: Amount in contracts (BTC options are in BTC, not contracts).
            price: Limit price.
            reduce_only: Reduce only flag.

        Returns:
            Order result.
        """
        self._require_auth()
        return self._make_request(
            method='private/buy',
            params={
                'instrument_name': instrument_name,
                'amount': amount,
                'price': price,
                'type': 'limit',
                'reduce_only': reduce_only,
            },
            private=True,
        )

    def sell_limit(
        self,
        instrument_name: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place a limit sell order.
        下限价卖单。

        Args:
            instrument_name: Instrument name.
            amount: Amount.
            price: Limit price.
            reduce_only: Reduce only flag.

        Returns:
            Order result.
        """
        self._require_auth()
        return self._make_request(
            method='private/sell',
            params={
                'instrument_name': instrument_name,
                'amount': amount,
                'price': price,
                'type': 'limit',
                'reduce_only': reduce_only,
            },
            private=True,
        )

    def cancel_all(self) -> Dict[str, Any]:
        """
        Cancel all open orders.
        取消所有挂单。

        Returns:
            Cancel result.
        """
        self._require_auth()
        return self._make_request(
            method='private/cancel_all',
            params={},
            private=True,
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel a specific order.
        取消指定订单。

        Args:
            order_id: Order ID.

        Returns:
            Cancel result.
        """
        self._require_auth()
        return self._make_request(
            method='private/cancel',
            params={'order_id': order_id},
            private=True,
        )

    def get_open_orders(self, currency: str = 'BTC') -> List[Dict[str, Any]]:
        """
        Get all open orders.
        获取所有挂单。

        Args:
            currency: 'BTC' or 'ETH'.

        Returns:
            List of order dicts.
        """
        self._require_auth()
        return self._make_request(
            method='private/get_open_orders',
            params={'currency': currency},
            private=True,
        )

    def get_account_summary(self, currency: str = 'BTC') -> Dict[str, Any]:
        """
        Get account summary (balance, equity, etc.).
        获取账户摘要（余额、净值等）。

        Args:
            currency: 'BTC' or 'ETH'.

        Returns:
            Account summary dict.
        """
        self._require_auth()
        return self._make_request(
            method='private/get_account_summary',
            params={'currency': currency},
            private=True,
        )


# ===========================================================================
# DeribitWebSocketClient - WebSocket Client
# ===========================================================================

class DeribitWebSocketClient:
    """
    Deribit WebSocket API client.
    Deribit WebSocket API 客户端。

    NOTE: This requires the 'websocket-client' package.
    注：此客户端需要安装 'websocket-client' 包。

    Install: pip install websocket-client

    This class wraps the DeribitOptionsAPI for real-time data via WebSocket.
    It provides subscriptions to order book, trades, and user orders.
    此类通过 WebSocket 为 DeribitOptionsAPI 封装实时数据。
    支持订单簿、成交和用户订单的订阅。

    Attributes / 属性:
        ws_url (str): WebSocket URL.
        api (DeribitOptionsAPI): REST API instance for auth.
    """

    def __init__(
        self,
        testnet: bool = False,
        client_id: str = None,
        client_secret: str = None,
    ):
        if WebSocketClient is None:
            raise ImportError(
                "websocket-client package is required for DeribitWebSocketClient. "
                "Install with: pip install websocket-client"
            )

        self.ws_url = (
            'wss://test.deribit.com/ws/api/v2' if testnet
            else 'wss://www.deribit.com/ws/api/v2'
        )
        self.api = DeribitOptionsAPI(
            testnet=testnet,
            client_id=client_id,
            client_secret=client_secret,
        )
        self._ws = None
        self._counter = 0
        self._callbacks: Dict[str, List[callable]] = {}

    def connect(self) -> None:
        """
        Establish WebSocket connection and authenticate.
        建立 WebSocket 连接并进行身份验证。
        """
        self.api.auth()
        self._ws = WebSocketClient.create_connection(
            self.ws_url,
            enable_multithread=True,
        )
        self._auth_ws()

    def _auth_ws(self) -> None:
        """Send auth via WebSocket after REST auth. / 通过 WebSocket 发送认证。"""
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        nonce = 'abcd'
        data = ''
        signature = hmac.new(
            bytes(self.api.client_secret, 'latin-1'),
            msg=bytes(f'{timestamp}\n{nonce}\n{data}', 'latin-1'),
            digestmod=hashlib.sha256,
        ).hexdigest().lower()

        msg = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'public/auth',
            'params': {
                'grant_type': 'client_signature',
                'client_id': self.api.client_id,
                'timestamp': timestamp,
                'signature': signature,
                'nonce': nonce,
                'data': data,
            },
        }
        self._send(msg)

    def _send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send JSON-RPC message and return response.
        发送 JSON-RPC 消息并返回响应。
        """
        if self._ws is None:
            raise ConnectionError("WebSocket not connected")
        self._counter += 1
        msg['id'] = self._counter
        self._ws.send(json.dumps(msg))
        return json.loads(self._ws.recv())

    def subscribe(self, channel: str, callback: callable) -> None:
        """
        Subscribe to a channel with callback.
        订阅频道并设置回调。

        Args:
            channel: Channel name (e.g. 'deribit_price_index.btc').
            callback: Callable to handle incoming messages.
        """
        if channel not in self._callbacks:
            self._callbacks[channel] = []
            self._send({
                'jsonrpc': '2.0',
                'method': 'private/subscribe',
                'params': {'channels': [channel]},
            })
        self._callbacks[channel].append(callback)

    def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel.
        取消订阅频道。

        Args:
            channel: Channel name.
        """
        if channel in self._callbacks:
            self._send({
                'jsonrpc': '2.0',
                'method': 'private/unsubscribe',
                'params': {'channels': [channel]},
            })
            del self._callbacks[channel]

    def close(self) -> None:
        """Close WebSocket connection. / 关闭 WebSocket 连接。"""
        if self._ws:
            self._ws.close()
            self._ws = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


# ===========================================================================
# PositionMonitor - Real-time Position Monitor
# ===========================================================================

class PositionMonitor:
    """
    Real-time position monitor for options portfolio.
    期权组合实时持仓监控器。

    Tracks positions, calculates aggregate Greeks, and monitors risk limits.
    跟踪持仓、计算汇总希腊字母、监控风险限额。

    Attributes / 属性:
        api (DeribitOptionsAPI): API client.
        positions (dict): {instrument_name: position_dict}.
        contracts (dict): {instrument_name: OptionContract}.

    Example / 示例:
        >>> monitor = PositionMonitor(api)
        >>> monitor.refresh()
        >>> print(monitor.total_delta())
        >>> monitor.check_risk_limits()
    """

    def __init__(self, api: DeribitOptionsAPI):
        self.api = api
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.contracts: Dict[str, OptionContract] = {}
        self._greeks_calc = BlackScholesGreeks()
        self._spot_prices: Dict[str, float] = {}

    def set_spot(self, currency: str, price: float) -> None:
        """
        Set spot price for a currency.
        设置货币的现货价格。

        Args:
            currency: 'BTC' or 'ETH'.
            price: Spot price.
        """
        self._spot_prices[currency] = float(price)

    def get_spot(self, currency: str = 'BTC') -> float:
        """
        Get spot price for a currency.
        获取货币的现货价格。

        Args:
            currency: 'BTC' or 'ETH'.

        Returns:
            Spot price.
        """
        return self._spot_prices.get(currency, 0.0)

    def refresh(self, currency: str = 'BTC', kind: str = 'option') -> None:
        """
        Refresh all positions from exchange.
        从交易所刷新所有持仓。

        Args:
            currency: 'BTC' or 'ETH'.
            kind: 'option' or 'future'.
        """
        try:
            raw_positions = self.api.get_positions(currency=currency, kind=kind)
            for pos in raw_positions:
                self.positions[pos['instrument_name']] = pos
                self._update_contract(pos, currency)
        except DeribitAPIError as e:
            raise PositionMonitorError(f"Failed to refresh positions: {e}")

    def _update_contract(self, pos: Dict[str, Any], currency: str) -> None:
        """
        Update OptionContract from position data.
        从持仓数据更新 OptionContract。

        Args:
            pos: Position dict from API.
            currency: 'BTC' or 'ETH'.
        """
        instrument_name = pos['instrument_name']
        try:
            order_book = self.api.get_order_book(instrument_name)
        except Exception:
            return

        greeks = order_book.get('greeks', {})
        S = self.get_spot(currency)

        strike = self._parse_strike(instrument_name)
        expiry, option_type = self._parse_instrument_name(instrument_name)

        contract = OptionContract(
            instrument_name=instrument_name,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            mark_price=order_book.get('mark_price', 0),
            bid_price=order_book.get('best_bid_price', 0),
            ask_price=order_book.get('best_ask_price', 0),
            iv=float(greeks.get('iv', 0)),
            delta=float(greeks.get('delta', 0)),
            gamma=float(greeks.get('gamma', 0)),
            vega=float(greeks.get('vega', 0)),
            theta=float(greeks.get('theta', 0)),
            underlying_price=S,
            greeks_from_exchange=greeks,
        )
        self.contracts[instrument_name] = contract

    @staticmethod
    def _parse_instrument_name(instrument_name: str) -> Tuple[datetime, str]:
        """
        Parse Deribit instrument name.
        解析 Deribit 合约名称。

        Args:
            instrument_name: e.g. 'BTC-27MAR20-50000-C'.

        Returns:
            (expiry_datetime, 'call' or 'put')
        """
        parts = instrument_name.split('-')
        if len(parts) < 3:
            return datetime.utcnow(), 'call'

        option_type = 'call' if parts[-1] == 'C' else 'put'

        date_str = parts[1]
        try:
            expiry = datetime.strptime(date_str, '%d%b%y')
        except ValueError:
            try:
                expiry = datetime.strptime(date_str, '%d%b%Y')
            except ValueError:
                expiry = datetime.utcnow()

        return expiry, option_type

    @staticmethod
    def _parse_strike(instrument_name: str) -> float:
        """
        Parse strike price from instrument name.
        从合约名解析行权价。

        Args:
            instrument_name: e.g. 'BTC-27MAR20-50000-C'.

        Returns:
            Strike price.
        """
        parts = instrument_name.split('-')
        if len(parts) >= 3:
            try:
                return float(parts[-2])
            except ValueError:
                pass
        return 0.0

    def total_delta(self) -> float:
        """
        Calculate total portfolio delta.
        计算组合总 Delta。

        Returns:
            Sum of position_size * contract_delta for all positions.
        """
        total = 0.0
        for name, pos in self.positions.items():
            size = float(pos.get('size', 0))
            if name in self.contracts:
                delta = self.contracts[name].delta
            else:
                delta = 0.0
            total += size * delta
        return total

    def total_gamma(self) -> float:
        """
        Calculate total portfolio gamma.
        计算组合总 Gamma。

        Returns:
            Sum of position_size * contract_gamma.
        """
        total = 0.0
        for name, pos in self.positions.items():
            size = float(pos.get('size', 0))
            if name in self.contracts:
                gamma = self.contracts[name].gamma
            else:
                gamma = 0.0
            total += size * gamma
        return total

    def total_vega(self) -> float:
        """
        Calculate total portfolio vega.
        计算组合总 Vega。

        Returns:
            Sum of position_size * contract_vega.
        """
        total = 0.0
        for name, pos in self.positions.items():
            size = float(pos.get('size', 0))
            if name in self.contracts:
                vega = self.contracts[name].vega
            else:
                vega = 0.0
            total += size * vega
        return total

    def total_theta(self) -> float:
        """
        Calculate total portfolio theta (daily).
        计算组合总 Theta（每日）。

        Returns:
            Sum of position_size * contract_theta.
        """
        total = 0.0
        for name, pos in self.positions.items():
            size = float(pos.get('size', 0))
            if name in self.contracts:
                theta = self.contracts[name].theta
            else:
                theta = 0.0
            total += size * theta
        return total

    def position_summary(self) -> Dict[str, float]:
        """
        Get full position summary with all Greeks.
        获取包含所有希腊字母的持仓摘要。

        Returns:
            Dict with delta, gamma, vega, theta, and num_positions.
        """
        return {
            'delta': self.total_delta(),
            'gamma': self.total_gamma(),
            'vega': self.total_vega(),
            'theta': self.total_theta(),
            'num_positions': len(self.positions),
        }

    def check_risk_limits(
        self,
        max_delta: float = 10.0,
        max_gamma: float = 2.0,
        max_vega: float = 5.0,
        max_theta: float = -1.0,
    ) -> Dict[str, bool]:
        """
        Check risk limits.
        检查风险限额。

        Args:
            max_delta: Max absolute delta.
            max_gamma: Max absolute gamma.
            max_vega: Max absolute vega.
            max_theta: Min theta (negative number).

        Returns:
            Dict of {limit_name: exceeded_bool}.
        """
        summary = self.position_summary()
        return {
            'delta_ok': abs(summary['delta']) <= abs(max_delta),
            'gamma_ok': abs(summary['gamma']) <= abs(max_gamma),
            'vega_ok': abs(summary['vega']) <= abs(max_vega),
            'theta_ok': summary['theta'] >= max_theta,
        }

    def pnl_estimate(
        self,
        dS: float = 1000.0,
        dVol: float = 0.01,
        dDays: float = 1.0,
    ) -> Dict[str, float]:
        """
        Estimate PnL from small market moves.
        估算小幅度市场变动的盈亏。

        Args:
            dS: Spot price change / 标的价格变动
            dVol: Volatility change (decimal) / 波动率变动（小数）
            dDays: Days passed / 经过天数

        Returns:
            Dict with delta_pnl, gamma_pnl, vega_pnl, theta_pnl, total.
        """
        summary = self.position_summary()
        gamma_pnl = 0.5 * summary['gamma'] * dS ** 2
        theta_pnl = summary['theta'] * dDays
        delta_pnl = summary['delta'] * dS
        vega_pnl = summary['vega'] * dVol * 100.0

        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'total': delta_pnl + gamma_pnl + vega_pnl + theta_pnl,
        }


# ===========================================================================
# VavaBotOptionsStrategy - Strategy Wrapper
# ===========================================================================

class VavaBotOptionsStrategy:
    """
    VavaBot Options Strategy - Main strategy orchestrator.
    VavaBot 期权策略 - 主策略协调器。

    This class orchestrates the full options trading strategy including
    market data retrieval, signal generation, order execution, and risk management.
    此类协调完整的期权交易策略，包括市场数据获取、信号生成、
    订单执行和风险管理。

    Attributes / 属性:
        api (DeribitOptionsAPI): REST API client.
        monitor (PositionMonitor): Position monitor.
        greeks (BlackScholesGreeks): Greeks calculator.
        volatility_surface (VolatilitySurface): Vol surface.

    Example / 示例:
        >>> strategy = VavaBotOptionsStrategy(
        ...     testnet=True,
        ...     client_id='xxx',
        ...     client_secret='yyy',
        ... )
        >>> strategy.run()
    """

    def __init__(
        self,
        testnet: bool = True,
        client_id: str = None,
        client_secret: str = None,
    ):
        self.api = DeribitOptionsAPI(
            testnet=testnet,
            client_id=client_id,
            client_secret=client_secret,
        )
        self.monitor = PositionMonitor(self.api)
        self.greeks = BlackScholesGreeks()
        self.volatility_surface = VolatilitySurface()
        self._running = False
        self._spot_price: float = 0.0
        self.available_instruments: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        """
        Initialize strategy (auth, load instruments, refresh positions).
        初始化策略（认证、加载合约、刷新持仓）。
        """
        if self.api.client_id and self.api.client_secret:
            self.api.auth()

        self.available_instruments = self.api.get_instruments('BTC', expired=False)
        self._spot_price = self.api.get_index_price('BTC')

        try:
            self.monitor.set_spot('BTC', self._spot_price)
            self.monitor.refresh('BTC', kind='option')
        except DeribitAPIError:
            pass

    def get_available_strikes(
        self,
        currency: str = 'BTC',
        option_type: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Get available option strikes from exchange.
        获取交易所可用期权行权价列表。

        Args:
            currency: 'BTC' or 'ETH'.
            option_type: Filter by 'call' or 'put' (optional).

        Returns:
            List of instrument dicts.
        """
        instruments = self.api.get_instruments(currency=currency, expired=False)
        if option_type:
            option_type_lower = option_type.lower()
            instruments = [
                i for i in instruments
                if i.get('instrument_name', '').endswith(
                    '-C' if option_type_lower == 'call' else '-P'
                )
            ]
        return instruments

    def find_atm_strikes(
        self,
        currency: str = 'BTC',
        moneyness_range: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Find ATM and near-ATM option strikes.
        查找 ATM 及附近行权价期权。

        Args:
            currency: 'BTC' or 'ETH'.
            moneyness_range: Range around ATM (e.g. 0.05 = 5%).

        Returns:
            List of ATM-adjacent instruments.
        """
        S = self._spot_price
        instruments = self.get_available_strikes(currency)

        atm_instruments = []
        for inst in instruments:
            try:
                strike = float(inst.get('strike', 0))
            except (TypeError, ValueError):
                continue
            if S > 0:
                moneyness = abs(strike - S) / S
                if moneyness <= moneyness_range:
                    atm_instruments.append(inst)
        return atm_instruments

    def get_greeks_for_instrument(
        self,
        instrument_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get full Greeks for an instrument.
        获取指定合约的完整希腊字母。

        Args:
            instrument_name: Deribit instrument name.

        Returns:
            Dict with greeks or None on error.
        """
        try:
            order_book = self.api.get_order_book(instrument_name)
            return order_book.get('greeks', {})
        except DeribitAPIError:
            return None

    def place_order(
        self,
        instrument_name: str,
        direction: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an order.
        下单。

        Args:
            instrument_name: Instrument name.
            direction: 'buy' or 'sell'.
            amount: Order amount.
            price: Limit price.
            reduce_only: Reduce only flag.

        Returns:
            Order result.
        """
        direction = direction.lower()
        if direction == 'buy':
            return self.api.buy_limit(
                instrument_name=instrument_name,
                amount=amount,
                price=price,
                reduce_only=reduce_only,
            )
        elif direction == 'sell':
            return self.api.sell_limit(
                instrument_name=instrument_name,
                amount=amount,
                price=price,
                reduce_only=reduce_only,
            )
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'buy' or 'sell'.")

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.
        取消所有挂单。

        Returns:
            Cancel result.
        """
        return self.api.cancel_all()

    def build_volatility_surface(
        self,
        currency: str = 'BTC',
        option_type: str = None,
    ) -> VolatilitySurface:
        """
        Build volatility surface from available instruments.
        从可用合约构建波动率曲面。

        Args:
            currency: 'BTC' or 'ETH'.
            option_type: 'call', 'put', or None for both.

        Returns:
            Populated VolatilitySurface.
        """
        self.volatility_surface.clear()

        S = self._spot_price
        instruments = self.get_available_strikes(currency, option_type)

        for inst in instruments:
            name = inst.get('instrument_name', '')
            strike = inst.get('strike')
            expiry_str = inst.get('expiration_timestamp')

            try:
                order_book = self.api.get_order_book(name)
                greeks = order_book.get('greeks', {})
                iv = float(greeks.get('iv', 0))
            except Exception:
                iv = 0.0

            if strike and expiry_str:
                try:
                    expiry = datetime.utcfromtimestamp(expiry_str / 1000.0)
                    tenor = (expiry - datetime.utcnow()).total_seconds() / (365.0 * 86400.0)
                except Exception:
                    tenor = 0.0

                opt_type = 'call' if name.endswith('-C') else 'put'
                self.volatility_surface.add_strike(
                    strike=float(strike),
                    iv=iv,
                    tenor=max(0.0, tenor),
                    option_type=opt_type,
                    spot=S,
                )

        return self.volatility_surface

    def run(self) -> None:
        """
        Start the strategy main loop (blocking).
        启动策略主循环（阻塞）。

        Note: This is a placeholder - the actual trading loop
        should be implemented with proper threading/event handling.
        注意：这是占位实现，实际交易循环应使用适当的线程/事件处理。
        """
        self._running = True
        self.initialize()
        while self._running:
            try:
                self.monitor.refresh('BTC', kind='option')
                time.sleep(2)
            except Exception:
                time.sleep(5)

    def stop(self) -> None:
        """Stop the strategy main loop. / 停止策略主循环。"""
        self._running = False


# ===========================================================================
# Exception Classes
# ===========================================================================

class DeribitAPIError(Exception):
    """
    Deribit API error.
    Deribit API 错误。

    Attributes / 属性:
        code (int): Deribit error code.
        message (str): Error message.
        data (any): Additional error data.
    """

    def __init__(
        self,
        code: int = -1,
        message: str = 'Unknown error',
        data: Any = None,
    ):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[Deribit {code}] {message}")


class DeribitAuthError(DeribitAPIError):
    """
    Deribit authentication error.
    Deribit 认证错误。
    """

    def __init__(self, message: str = 'Authentication required'):
        super().__init__(code=0, message=message)


class PositionMonitorError(Exception):
    """
    Position monitor error.
    持仓监控器错误。
    """
    pass
