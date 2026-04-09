"""
Greeks 计算工具
"""

import math
from dataclasses import dataclass
from typing import Dict

from .black_scholes import BlackScholes, norm_cdf, norm_pdf, d1_d2


@dataclass
class Greeks:
    """Greeks 容器"""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    # 扩展 Greeks
    vanna: float = 0.0  # dDelta/dVol
    charm: float = 0.0  # dDelta/dTime
    speed: float = 0.0  # dGamma/dSpot
    color: float = 0.0  # dGamma/dTime
    volga: float = 0.0  # dVega/dVol

    def to_dict(self) -> Dict:
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "vanna": self.vanna,
            "charm": self.charm,
            "speed": self.speed,
            "color": self.color,
            "volga": self.volga,
        }

    def __str__(self) -> str:
        return (
            f"Price: {self.price:.4f} | "
            f"Delta: {self.delta:.4f} | "
            f"Gamma: {self.gamma:.6f} | "
            f"Vega: {self.vega:.4f} | "
            f"Theta: {self.theta:.4f} | "
            f"Rho: {self.rho:.4f}"
        )


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """
    计算完整 Greeks（包含高阶 Greeks）

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: "call" 或 "put"

    Returns:
        Greeks 对象
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    d1, d2 = d1_d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        price = bs.call_price()
        delta = bs.call_delta()
        theta = bs.theta_call()
        rho = bs.rho_call()
    else:
        price = bs.put_price()
        delta = bs.put_delta()
        theta = bs.theta_put()
        rho = bs.rho_put()

    gamma = bs.gamma()
    vega = bs.vega()

    # 高阶 Greeks
    sqrt_t = math.sqrt(T) if T > 0 else 0

    # Vanna = dDelta/dVol = dVega/dSpot
    # Vanna ≈ norm_pdf(d1) * (d2 - d1) / (sigma * sqrt_t)
    if sqrt_t > 0 and sigma > 0:
        vanna = norm_pdf(d1) * (d2 - d1) / (sigma * sqrt_t)
    else:
        vanna = 0.0

    # Charm = dDelta/dTime
    # Charm ≈ -norm_pdf(d1) * (r * d2 - (r + 0.5 * sigma^2) * d1 / T)
    if T > 0:
        charm = -norm_pdf(d1) * (
            r * d2 - (r + 0.5 * sigma * sigma) * d1 / T
        ) / (sigma * sqrt_t) if sqrt_t > 0 else 0
    else:
        charm = 0.0

    # Speed = dGamma/dSpot
    # Speed ≈ -norm_pdf(d1) * (d1 + 1) / (S^2 * sigma * sqrt_t)
    if S > 0 and sqrt_t > 0 and sigma > 0:
        speed = -norm_pdf(d1) * (d1 + 1) / (S * S * sigma * sqrt_t)
    else:
        speed = 0.0

    # Color = dGamma/dTime
    if T > 0 and sqrt_t > 0:
        term = (r + 0.5 * sigma * sigma) * d1 / (sigma * sqrt_t) - d2 / (2 * T)
        color = -norm_pdf(d1) / (S * sigma * sqrt_t) * (term + 1 / (S * sqrt_t))
    else:
        color = 0.0

    # Volga = dVega/dVol
    if sqrt_t > 0:
        volga = norm_pdf(d1) * d1 * d2 / sigma
    else:
        volga = 0.0

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        charm=charm,
        speed=speed,
        color=color,
        volga=volga,
    )


def calculate_portfolio_greeks(positions: list) -> Greeks:
    """
    计算组合 Greeks

    Args:
        positions: 持仓列表，每项为 dict:
            {
                "option_type": "call" / "put",
                "size": 正数=买入, 负数=卖出,
                "S": 标的价格,
                "K": 行权价,
                "T": 到期时间,
                "r": 无风险利率,
                "sigma": 波动率,
            }

    Returns:
        组合 Greeks
    """
    total_price = 0.0
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for pos in positions:
        greeks = calculate_greeks(
            S=pos["S"],
            K=pos["K"],
            T=pos["T"],
            r=pos["r"],
            sigma=pos["sigma"],
            option_type=pos["option_type"],
        )
        size = pos.get("size", 1)

        total_price += greeks.price * size
        total_delta += greeks.delta * size
        total_gamma += greeks.gamma * size
        total_vega += greeks.vega * size
        total_theta += greeks.theta * size
        total_rho += greeks.rho * size

    return Greeks(
        price=total_price,
        delta=total_delta,
        gamma=total_gamma,
        vega=total_vega,
        theta=total_theta,
        rho=total_rho,
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    计算隐含波动率（使用牛顿法）

    Args:
        market_price: 市场价格
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        option_type: "call" 或 "put"
        tol: 收敛容差
        max_iter: 最大迭代次数

    Returns:
        隐含波动率
    """
    from .black_scholes import BlackScholes

    # 初始猜测
    sigma = 0.5

    for _ in range(max_iter):
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
        price = bs.call_price() if option_type.lower() == "call" else bs.put_price()

        # 边际误差
        vega = bs.vega()
        if abs(vega) < 1e-10:
            break

        # 牛顿法更新
        diff = market_price - price
        if abs(diff) < tol:
            break

        sigma += diff / (vega * 100)  # vega 是按1%波动率计算的

        # 边界检查
        sigma = max(0.01, min(sigma, 5.0))

    return sigma


if __name__ == "__main__":
    # 测试
    print("=== Greeks Calculator Test ===\n")

    S, K, T, r, sigma = 2177.0, 2200.0, 30 / 365, 0.05, 0.80

    print(f"标的: ETH ${S}, 行权价: ${K}, 到期: {T * 365:.0f}天, IV: {sigma * 100:.0f}%\n")

    for opt_type in ["call", "put"]:
        greeks = calculate_greeks(S, K, T, r, sigma, opt_type)
        print(f"--- {opt_type.upper()} ---")
        print(greeks)
        print()

    # 组合测试
    print("=== Portfolio Greeks ===")
    positions = [
        {"option_type": "call", "size": 1, "S": 2177, "K": 2200, "T": 30 / 365, "r": 0.05, "sigma": 0.80},
        {"option_type": "put", "size": -1, "S": 2177, "K": 2150, "T": 30 / 365, "r": 0.05, "sigma": 0.80},
    ]
    portfolio = calculate_portfolio_greeks(positions)
    print(portfolio)
