"""
Black-Scholes 期权定价模型 + Implied Volatility Newton-Raphson 求解器
从 Options-Trading-Bot 仓库吸收：生产级 Black-Scholes 定价 + IV 求解 + Greeks
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


# ============================================================================
# 核心数学函数
# ============================================================================

def norm_cdf(x: float) -> float:
    """标准正态分布累计概率函数（scipy优先，fallback为erf近似）"""
    try:
        from scipy.stats import norm
        return norm.cdf(x)
    except ImportError:
        return norm_cdf_approx(x)


def norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数（scipy优先，fallback为近似）"""
    try:
        from scipy.stats import norm
        return norm.pdf(x)
    except ImportError:
        return norm_pdf_approx(x)


def norm_cdf_approx(x: float) -> float:
    """
    标准正态分布累计概率函数（使用erf近似）
    当 scipy.stats.norm 不可用时使用此近似
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def norm_pdf_approx(x: float) -> float:
    """标准正态分布概率密度函数（近似）"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """计算 d1 和 d2"""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


# 优先使用 scipy.stats.norm（更精确），否则回退到近似
try:
    _norm_cdf = norm.cdf
    _norm_pdf = norm.pdf
except Exception:
    _norm_cdf = norm_cdf_approx
    _norm_pdf = norm_pdf_approx


# ============================================================================
# Black-Scholes 定价器
# ============================================================================

@dataclass
class BlackScholes:
    """
    Black-Scholes 期权定价器

    Attributes:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
    """

    S: float  # 标的价格
    K: float  # 行权价
    T: float  # 到期时间（年）
    r: float  # 无风险利率
    sigma: float  # 波动率

    def __post_init__(self):
        """验证参数"""
        if self.S <= 0:
            raise ValueError("标的价格必须为正")
        if self.K <= 0:
            raise ValueError("行权价必须为正")
        if self.T < 0:
            raise ValueError("到期时间不能为负")
        if self.sigma <= 0:
            raise ValueError("波动率必须为正")

    @property
    def d1(self) -> float:
        """d1"""
        d1, _ = d1_d2(self.S, self.K, self.T, self.r, self.sigma)
        return d1

    @property
    def d2(self) -> float:
        """d2"""
        _, d2 = d1_d2(self.S, self.K, self.T, self.r, self.sigma)
        return d2

    def call_price(self) -> float:
        """计算 Call 价格"""
        if self.T <= 0:
            return max(0, self.S - self.K)
        d1, d2 = self.d1, self.d2
        return self.S * _norm_cdf(d1) - self.K * math.exp(-self.r * self.T) * _norm_cdf(d2)

    def put_price(self) -> float:
        """计算 Put 价格"""
        if self.T <= 0:
            return max(0, self.K - self.S)
        d1, d2 = self.d1, self.d2
        return self.K * math.exp(-self.r * self.T) * _norm_cdf(-d2) - self.S * _norm_cdf(-d1)

    def price(self, option_type: str = "call") -> float:
        """期权价格"""
        if option_type.lower() == "call":
            return self.call_price()
        return self.put_price()

    def call_delta(self) -> float:
        """Call Delta"""
        if self.T <= 0:
            return 1.0 if self.S > self.K else 0.0
        return _norm_cdf(self.d1)

    def put_delta(self) -> float:
        """Put Delta"""
        if self.T <= 0:
            return -1.0 if self.S < self.K else 0.0
        return _norm_cdf(self.d1) - 1

    def delta(self, option_type: str = "call") -> float:
        """Delta"""
        if option_type.lower() == "call":
            return self.call_delta()
        return self.put_delta()

    def gamma(self) -> float:
        """Gamma（Call 和 Put 相同）"""
        if self.T <= 0:
            return 0.0
        return _norm_pdf(self.d1) / (self.S * self.sigma * math.sqrt(self.T))

    def vega(self) -> float:
        """Vega（Call 和 Put 相同）"""
        if self.T <= 0:
            return 0.0
        return self.S * _norm_pdf(self.d1) * math.sqrt(self.T) / 100  # 每1%波动率

    def theta_call(self) -> float:
        """Call Theta（每日）"""
        if self.T <= 0:
            return 0.0
        term1 = -self.S * _norm_pdf(self.d1) * self.sigma / (2 * math.sqrt(self.T))
        term2 = self.r * self.K * math.exp(-self.r * self.T) * _norm_cdf(self.d2)
        return (term1 - term2) / 365

    def theta_put(self) -> float:
        """Put Theta（每日）"""
        if self.T <= 0:
            return 0.0
        term1 = -self.S * _norm_pdf(self.d1) * self.sigma / (2 * math.sqrt(self.T))
        term2 = self.r * self.K * math.exp(-self.r * self.T) * _norm_cdf(-self.d2)
        return (term1 + term2) / 365

    def theta(self, option_type: str = "call") -> float:
        """Theta（每日）"""
        if option_type.lower() == "call":
            return self.theta_call()
        return self.theta_put()

    def rho_call(self) -> float:
        """Call Rho"""
        if self.T <= 0:
            return 0.0
        return self.K * self.T * math.exp(-self.r * self.T) * _norm_cdf(self.d2) / 100

    def rho_put(self) -> float:
        """Put Rho"""
        if self.T <= 0:
            return 0.0
        return -self.K * self.T * math.exp(-self.r * self.T) * _norm_cdf(-self.d2) / 100

    def rho(self, option_type: str = "call") -> float:
        """Rho"""
        if option_type.lower() == "call":
            return self.rho_call()
        return self.rho_put()

    def all_greeks(self, option_type: str = "call") -> Dict:
        """计算所有 Greeks"""
        return {
            "price": self.call_price() if option_type.lower() == "call" else self.put_price(),
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(option_type),
            "rho": self.rho(option_type),
            "d1": self.d1,
            "d2": self.d2,
        }


# ============================================================================
# Implied Volatility Newton-Raphson 求解器
# ============================================================================

def implied_volatility_newton_raphson(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    使用 Newton-Raphson 方法计算隐含波动率

    Args:
        market_price: 市场价格
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        option_type: "call" 或 "put"
        initial_guess: 初始波动率猜测（默认0.5即50%）
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        隐含波动率（如果未收敛则返回 NaN）
    """
    if T <= 0 or market_price <= 0:
        return float('nan')

    # 边界检查：期权价格不能超过内在价值
    if option_type.lower() == "call":
        intrinsic = max(0, S - K * math.exp(-r * T))
    else:
        intrinsic = max(0, K * math.exp(-r * T) - S)

    if market_price < intrinsic * 0.99:  # 允许少量误差
        return float('nan')

    sigma = initial_guess

    for i in range(max_iter):
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
        price = bs.call_price() if option_type.lower() == "call" else bs.put_price()

        # Vega：价格对波动率的导数
        vega = bs.vega() * 100  # 转回原始值（vega 是按 1% 计算的）

        if abs(vega) < 1e-10:
            # Vega 太小时使用二分法
            sigma = sigma * 1.5 if price < market_price else sigma * 0.5
            continue

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Newton-Raphson 更新
        sigma = sigma - diff / vega

        # 边界约束
        sigma = max(0.01, min(sigma, 5.0))  # 1% 到 500%

    # 检查最终收敛状态
    final_bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    final_price = final_bs.call_price() if option_type.lower() == "call" else final_bs.put_price()
    final_diff = abs(final_price - market_price)

    return sigma if final_diff < tol * 10 else float('nan')


def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    使用二分法计算隐含波动率（更稳定但较慢）

    适用于 Newton-Raphson 不收敛的情况
    """
    if T <= 0:
        return float('nan')

    # 确定搜索范围
    vol_low = 0.01   # 1%
    vol_high = 5.0   # 500%

    bs_low = BlackScholes(S=S, K=K, T=T, r=r, sigma=vol_low)
    bs_high = BlackScholes(S=S, K=K, T=T, r=r, sigma=vol_high)

    if option_type.lower() == "call":
        price_low = bs_low.call_price()
        price_high = bs_high.call_price()
    else:
        price_low = bs_low.put_price()
        price_high = bs_high.put_price()

    # 检查边界
    if market_price <= price_low:
        return 0.01
    if market_price >= price_high:
        return 5.0

    for _ in range(max_iter):
        vol_mid = (vol_low + vol_high) / 2
        bs_mid = BlackScholes(S=S, K=K, T=T, r=r, sigma=vol_mid)
        price_mid = bs_mid.call_price() if option_type.lower() == "call" else bs_mid.put_price()

        diff = price_mid - market_price

        if abs(diff) < tol:
            return vol_mid

        if diff > 0:
            vol_low = vol_mid
        else:
            vol_high = vol_mid

    return (vol_low + vol_high) / 2


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    计算隐含波动率（自动选择方法）

    优先使用 Newton-Raphson，如果失败则使用二分法

    Returns:
        隐含波动率，如果无法计算则返回 NaN
    """
    iv = implied_volatility_newton_raphson(
        market_price=market_price,
        S=S, K=K, T=T, r=r,
        option_type=option_type,
        max_iter=max_iter,
        tol=tol,
    )

    if math.isnan(iv):
        # 回退到二分法
        iv = implied_volatility_bisection(
            market_price=market_price,
            S=S, K=K, T=T, r=r,
            option_type=option_type,
            max_iter=max_iter,
            tol=tol,
        )

    return iv


# ============================================================================
# 快捷函数
# ============================================================================

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    快捷函数：计算期权价格

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: "call" 或 "put"

    Returns:
        期权价格
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    return bs.call_price() if option_type.lower() == "call" else bs.put_price()


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict:
    """
    快捷函数：计算所有 Greeks

    Returns:
        包含 price, delta, gamma, vega, theta, rho 的字典
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    return bs.all_greeks(option_type)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict:
    """
    Black-Scholes 定价和 Greeks 计算（兼容旧接口）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'

    Returns:
        dict: 包含 price, delta, gamma, theta, vega, rho
    """
    if T <= 0 or sigma <= 0:
        # 处理过期或无效期权
        if option_type == 'call':
            price = max(0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0, K - S)
            delta = -1.0 if S < K else 0.0
        return {'price': price, 'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = -_norm_cdf(-d1)
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T) / 100
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) -
             r * K * math.exp(-r * T) * (_norm_cdf(d2) if option_type == 'call' else _norm_cdf(-d2))) / 365

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
    }


# ============================================================================
# 波动率表面工具
# ============================================================================

def calculate_iv_smile(
    market_prices: Dict[Tuple[float, str], float],
    S: float,
    T: float,
    r: float,
) -> Dict[Tuple[float, str], float]:
    """
    计算波动率微笑（IV Surface）

    Args:
        market_prices: {(strike, option_type): price} 字典
        S: 标的价格
        T: 到期时间（年）
        r: 无风险利率

    Returns:
        {(strike, option_type): iv} 字典
    """
    iv_surface = {}

    for (strike, opt_type), price in market_prices.items():
        iv = implied_volatility(
            market_price=price,
            S=S,
            K=strike,
            T=T,
            r=r,
            option_type=opt_type,
        )
        iv_surface[(strike, opt_type)] = iv

    return iv_surface


# ============================================================================
# 独立 Greeks 函数 (兼容 API)
# ============================================================================

def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = 'call') -> dict:
    """
    计算期权 Greeks（独立函数形式）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'

    Returns:
        dict: {
            delta, gamma, vega, theta, rho,
            d1, d2
        }
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    result = bs.all_greeks(option_type)
    # 添加 d1, d2
    result['d1'] = bs.d1
    result['d2'] = bs.d2
    return result


def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = 'call') -> float:
    """
    Delta = dV/dS（标的价格变化对期权价格的影响）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'

    Returns:
        Delta 值（call 在 ITM 时接近 1，put 接近 -1）
    """
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    return _norm_cdf(d1_d2(S, K, T, r, sigma)[0]) if option_type == 'call' else _norm_cdf(d1_d2(S, K, T, r, sigma)[0]) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Gamma = d²V/dS²（Delta 对标的价格的二阶导数）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率

    Returns:
        Gamma 值（Call 和 Put 相同）
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega = dV/dσ（波动率变化对期权价格的影响，每 1% vol 移动）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率

    Returns:
        Vega 值（每 1% 波动率变化的美元影响 / 100）
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T) / 100


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = 'call') -> float:
    """
    Theta = -dV/dt（时间衰减，每日）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'

    Returns:
        Theta 值（每日时间衰减的美元值）
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma)
    term1 = -S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
    if option_type == 'call':
        term2 = r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)
    return (term1 - term2) / 365


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = 'call') -> float:
    """
    Rho = dV/dr（利率变化对期权价格的影响，每 1% 利率移动）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'

    Returns:
        Rho 值（每 1% 利率变化的美元影响 / 100）
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
    else:
        return -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100


# ============================================================================
# 波动率微笑拟合 - SVI 模型
# ============================================================================

def svi_volatility(K: float, T: float, params: dict) -> float:
    """
    SVI (Stochastic Volatility Inspired) 波动率模型

    Args:
        K: 行权价
        T: 到期时间
        params: SVI 参数 {a, b, rho, m, sigma}

    Returns:
        波动率
    """
    a = params['a']
    b = params['b']
    rho = params['rho']
    m = params['m']
    sigma = params['sigma']

    # SVI 公式
    d = K - m
    sqrt_term = math.sqrt(d * d + sigma * sigma)
    return a + b * (rho * d + sqrt_term)


def svi_objective(params: list, strikes: list, market_ivs: list, T: float) -> float:
    """SVI 拟合目标函数（最小二乘）"""
    a, b, rho, m, sigma = params
    total_error = 0.0
    for K, market_iv in zip(strikes, market_ivs):
        if market_iv <= 0:
            continue
        try:
            model_iv = svi_volatility(K, T, {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma})
            total_error += (model_iv - market_iv) ** 2
        except (ValueError, RuntimeWarning):
            total_error += 1e6
    return total_error


def volatility_smile_fit(strikes: list[float],
                         market_ivs: list[float],
                         T: float,
                         model: str = 'svi') -> dict:
    """
    波动率微笑拟合

    Args:
        strikes: 行权价列表
        market_ivs: 市场隐含波动率列表（与 strikes 对应）
        T: 到期时间（年）
        model: 拟合模型（目前仅支持 'svi'）

    Returns:
        dict: SVI 参数 {a, b, rho, m, sigma}

    Note:
        SVI 模型公式:
        σ(K) = a + b * (ρ * (K - m) + sqrt((K - m)² + σ²))

        参数约束:
        - a >= 0 (水平参数)
        - b > 0 (开口参数)
        - |ρ| < 1 (相关性参数)
        - σ > 0 (混合参数)
    """
    if model != 'svi':
        raise ValueError(f"Unsupported model: {model}. Only 'svi' is supported.")

    if len(strikes) != len(market_ivs):
        raise ValueError("strikes and market_ivs must have the same length")

    if len(strikes) < 5:
        raise ValueError("At least 5 data points required for SVI fit")

    # 过滤有效数据
    valid_data = [(k, iv) for k, iv in zip(strikes, market_ivs) if iv > 0 and iv < 5]
    if len(valid_data) < 5:
        raise ValueError("At least 5 valid IV data points required")

    strikes_valid = [k for k, _ in valid_data]
    ivs_valid = [iv for _, iv in valid_data]

    # 初始猜测
    m0 = sum(strikes_valid) / len(strikes_valid)
    a0 = min(ivs_valid)
    b0 = (max(ivs_valid) - min(ivs_valid)) / (max(strikes_valid) - min(strikes_valid) + 1e-6)

    # 使用 scipy.optimize.minimize
    try:
        from scipy.optimize import minimize

        initial_params = [a0, max(b0, 0.01), 0.0, m0, 0.1]

        # 边界约束
        bounds = [
            (0.0, None),      # a
            (0.001, None),    # b
            (-0.999, 0.999),  # rho
            (min(strikes_valid), max(strikes_valid)),  # m
            (0.001, None),    # sigma
        ]

        result = minimize(
            svi_objective,
            initial_params,
            args=(strikes_valid, ivs_valid, T),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )

        if result.success:
            a, b, rho, m, sigma = result.x
        else:
            # 使用简单拟合作为 fallback
            a, b, rho, m, sigma = _simple_svi_fit(strikes_valid, ivs_valid)

    except ImportError:
        # scipy 不可用时使用简单拟合
        a, b, rho, m, sigma = _simple_svi_fit(strikes_valid, ivs_valid)

    return {
        'a': a,
        'b': b,
        'rho': rho,
        'm': m,
        'sigma': sigma,
        'model': 'svi'
    }


def _simple_svi_fit(strikes: list[float], ivs: list[float]) -> tuple:
    """简单的 SVI 拟合（不使用 scipy）"""
    import statistics
    a = min(ivs)
    m = statistics.median(strikes)
    b = (max(ivs) - min(ivs)) / (max(strikes) - min(strikes) + 1e-6)
    b = max(b, 0.01)
    rho = 0.0
    sigma = 0.1
    return a, b, rho, m, sigma


# ============================================================================
# __all__ 导出
# ============================================================================

__all__ = [
    # 核心类
    'BlackScholes',
    # 定价函数
    'bs_price',
    'black_scholes_greeks',
    # 独立 Greeks 函数
    'greeks',
    'delta',
    'gamma',
    'vega',
    'theta',
    'rho',
    # IV 求解器
    'implied_volatility',
    'implied_volatility_newton_raphson',
    'implied_volatility_bisection',
    # 波动率表面
    'calculate_iv_smile',
    'volatility_smile_fit',
    'svi_volatility',
    # 数学工具
    'd1_d2',
    'norm_cdf',
    'norm_pdf',
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试用例：ETH 期权
    S = 2177.0   # 标的价格
    K = 2200.0   # 行权价
    T = 30 / 365  # 30天
    r = 0.05     # 5% 无风险利率
    sigma = 0.80  # 80% 波动率

    print("=== Black-Scholes + IV Solver 测试 ===")
    print(f"标的: ETH ${S}")
    print(f"行权价: ${K}")
    print(f"到期: {T * 365:.0f} 天")
    print(f"波动率: {sigma * 100:.0f}%")
    print()

    # 测试定价和 Greeks
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)

    print("--- Call ---")
    greeks = bs.all_greeks("call")
    for k, v in greeks.items():
        if k in ("d1", "d2"):
            continue
        print(f"  {k}: {v:.6f}")

    print("\n--- Put ---")
    greeks = bs.all_greeks("put")
    for k, v in greeks.items():
        if k in ("d1", "d2"):
            continue
        print(f"  {k}: {v:.6f}")

    # 测试 IV 计算
    print("\n--- Implied Volatility Solver ---")
    market_call_price = bs.call_price()
    print(f"模拟市场价格: ${market_call_price:.2f}")

    iv_recovered = implied_volatility(
        market_price=market_call_price,
        S=S, K=K, T=T, r=r,
        option_type="call",
    )
    print(f"反推隐含波动率: {iv_recovered * 100:.2f}%")

    # 不同波动率对比
    print("\n--- 不同波动率对比 ---")
    for vol in [0.3, 0.5, 0.8, 1.0, 1.5]:
        bs_test = BlackScholes(S=S, K=K, T=T, r=r, sigma=vol)
        call = bs_test.call_price()
        put = bs_test.put_price()
        print(f"  {vol*100:4.0f}%: Call=${call:.2f} Put=${put:.2f}")
