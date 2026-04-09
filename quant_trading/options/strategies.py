"""
期权策略模块 - 从 Options-Trading-Bot 仓库吸收
核心策略: Straddle, Iron Condor, Strangle, RSI Momentum
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .pricing.black_scholes import BlackScholes, bs_price, bs_greeks, implied_volatility


# ============================================================================
# 策略类型枚举
# ============================================================================

class StrategyType(Enum):
    """策略类型"""
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    RATIO_SPREAD = "ratio_spread"
    CALENDAR_SPREAD = "calendar_spread"
    JADE_LIZARD = "jade_lizard"
    STRIP_STRAP = "strip_strap"


class OptionPositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


# ============================================================================
# 策略信号
# ============================================================================

@dataclass
class StrategySignal:
    """策略信号"""
    action: str  # "OPEN" or "CLOSE"
    option_type: str  # "call" or "put"
    strike_price: float
    side: str  # "LONG" or "SHORT"
    size: float
    premium: float
    expiration_days: Optional[int] = None  # 到期天数（用于日历价差）


@dataclass
class StrategyResult:
    """策略执行结果"""
    signals: List[StrategySignal]
    max_profit: float
    max_loss: float
    breakeven: List[float]
    net_premium: float
    probability_profit: float  # 近似概率（基于 BS）
    strategy_type: str = ""


# ============================================================================
# 策略盈亏分析
# ============================================================================

def calculate_straddle_pnl(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call_premium: float,
    put_premium: float,
    side: str = "long",
) -> Dict:
    """
    计算 Straddle 策略盈亏

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        call_premium: Call 权利金
        put_premium: Put 权利金
        side: "long" 或 "short"

    Returns:
        包含各项分析数据的字典
    """
    direction = 1 if side == "long" else -1

    # 到期价值
    call_payoff = max(0, S - K) * direction
    put_payoff = max(0, K - S) * direction
    total_payoff = call_payoff + put_payoff

    # 净成本/收入
    net_premium = -(call_premium + put_premium) * direction

    # 盈亏平衡点（标的价格 = K +/- 总权利金）
    total_cost = call_premium + put_premium
    breakeven_upper = K + total_cost
    breakeven_lower = K - total_cost

    # 最大盈利/亏损
    if side == "long":
        max_profit = float('inf') if S == K else abs(total_payoff)
        max_loss = total_cost
        # Long Straddle: 盈利概率 = P(S > K + cost) + P(S < K - cost)
        prob_upper = 1 - BlackScholes(S=K + total_cost, K=K, T=T, r=r, sigma=sigma).put_delta()
        prob_lower = BlackScholes(S=K - total_cost, K=K, T=T, r=r, sigma=sigma).put_delta()
        prob_profit = prob_upper + prob_lower
    else:
        max_profit = total_cost
        max_loss = float('inf')
        # Short Straddle: 盈利概率
        prob_profit = 1 - (1 - BlackScholes(S=K + total_cost, K=K, T=T, r=r, sigma=sigma).put_delta() -
                           BlackScholes(S=K - total_cost, K=K, T=T, r=r, sigma=sigma).put_delta())

    return {
        "side": side,
        "net_premium": net_premium,
        "breakeven": [breakeven_lower, breakeven_upper],
        "max_profit": max_profit,
        "max_loss": max_loss,
        "prob_profit": min(1.0, max(0.0, prob_profit)),
        "at_expiry": {
            "call_payoff": call_payoff,
            "put_payoff": put_payoff,
            "total_payoff": total_payoff,
        },
    }


def calculate_iron_condor_pnl(
    S: float,
    put_sell_strike: float,
    put_buy_strike: float,
    call_sell_strike: float,
    call_buy_strike: float,
    put_sell_premium: float,
    put_buy_premium: float,
    call_sell_premium: float,
    call_buy_premium: float,
) -> Dict:
    """
    计算 Iron Condor 策略盈亏

    Args:
        S: 标的价格
        put_sell_strike: 卖出Put行权价（较高）
        put_buy_strike: 买入Put行权价（较低）
        call_sell_strike: 卖出Call行权价（较低）
        call_buy_strike: 买入Call行权价（较高）
        各权利金

    Returns:
        包含各项分析数据的字典
    """
    # 净收入（收到的权利金 - 付出的权利金）
    net_credit = (put_sell_premium + call_sell_premium -
                  put_buy_premium - call_buy_premium)

    # 最大盈利 = 净收入
    max_profit = net_credit

    # 最大亏损 = (Put价差 + Call价差) - 净收入
    put_spread = put_sell_strike - put_buy_strike
    call_spread = call_buy_strike - call_sell_strike
    max_loss = (put_spread + call_spread) - net_credit

    # 盈亏平衡点
    # Lower breakeven = put_sell_strike - net_credit
    # Upper breakeven = call_sell_strike + net_credit
    breakeven_lower = put_sell_strike - net_credit
    breakeven_upper = call_sell_strike + net_credit

    # 各腿到期价值
    put_sell_payoff = max(0, put_buy_strike - S) if S <= put_sell_strike else max(0, put_sell_strike - S)
    call_sell_payoff = max(0, S - call_sell_strike) if S >= call_sell_strike else max(0, call_sell_strike - S)

    # 更精确的到期价值计算
    def option_payoff(price, strike, opt_type):
        if opt_type == "put":
            return max(0, strike - price)
        else:
            return max(0, price - strike)

    # 简化：假设价格穿过多腿才会触发
    put_sell_payoff = 0 if S > put_sell_strike else max(0, put_sell_strike - S)
    put_buy_payoff = 0 if S > put_buy_strike else max(0, put_buy_strike - S)
    call_sell_payoff = 0 if S < call_sell_strike else max(0, S - call_sell_strike)
    call_buy_payoff = 0 if S < call_buy_strike else max(0, S - call_buy_strike)

    return {
        "net_credit": net_credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": [breakeven_lower, breakeven_upper],
        "prob_profit": 0.65,  # 简化估算
    }


def calculate_strangle_pnl(
    S: float,
    call_strike: float,
    put_strike: float,
    call_premium: float,
    put_premium: float,
    side: str = "long",
) -> Dict:
    """
    计算 Strangle 策略盈亏

    Args:
        S: 标的价格
        call_strike: Call 行权价
        put_strike: Put 行权价
        call_premium: Call 权利金
        put_premium: Put 权利金
        side: "long" 或 "short"

    Returns:
        策略分析字典
    """
    direction = 1 if side == "long" else -1

    # 到期价值
    call_payoff = max(0, S - call_strike) * direction
    put_payoff = max(0, put_strike - S) * direction
    total_payoff = call_payoff + put_payoff

    # 净成本
    total_cost = call_premium + put_premium
    net_premium = -total_cost * direction

    # 盈亏平衡点
    if side == "long":
        breakeven_lower = put_strike - total_cost
        breakeven_upper = call_strike + total_cost
    else:
        breakeven_lower = put_strike + total_cost
        breakeven_upper = call_strike - total_cost

    if side == "long":
        max_profit = float('inf')
        max_loss = total_cost
    else:
        max_profit = total_cost
        max_loss = float('inf')

    return {
        "side": side,
        "net_premium": net_premium,
        "breakeven": [breakeven_lower, breakeven_upper],
        "max_profit": max_profit,
        "max_loss": max_loss,
        "prob_profit": 0.4,  # 简化估算
    }


# ============================================================================
# 策略信号生成器
# ============================================================================

class StraddleStrategy:
    """
    Straddle 策略（跨式套利）

    Long Straddle: 买入相同行权价的 Call 和 Put
    预期大幅波动（突破）
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strike_pct = self.config.get("strike_pct", 0.0)  # ATM = 0
        self.side = self.config.get("side", "long")  # "long" 或 "short"
        self.size = self.config.get("size", 1)

    def generate_signals(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> List[StrategySignal]:
        """
        生成交易信号

        Args:
            underlying_price: 标的价格
            option_chain: {strike: {"call_price": x, "put_price": y, "iv": z}}

        Returns:
            信号列表
        """
        signals = []

        # 找 ATM 行权价
        strikes = sorted(option_chain.keys())
        if not strikes:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        call_price = option_chain.get(atm_strike, {}).get("call_price", 0)
        put_price = option_chain.get(atm_strike, {}).get("put_price", 0)

        if call_price > 0 and put_price > 0:
            direction = 1 if self.side == "long" else -1
            signals.append(StrategySignal(
                action="OPEN",
                option_type="call",
                strike_price=atm_strike,
                side=self.side.upper(),
                size=self.size * direction,
                premium=call_price,
            ))
            signals.append(StrategySignal(
                action="OPEN",
                option_type="put",
                strike_price=atm_strike,
                side=self.side.upper(),
                size=self.size * direction,
                premium=put_price,
            ))

        return signals

    def analyze(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> StrategyResult:
        """分析策略"""
        signals = self.generate_signals(underlying_price, option_chain)

        if not signals:
            return StrategyResult(
                signals=[],
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        # 提取信息
        call_sig = next((s for s in signals if s.option_type == "call"), None)
        put_sig = next((s for s in signals if s.option_type == "put"), None)

        if not call_sig or not put_sig:
            return StrategyResult(
                signals=signals,
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        # 计算 T（从配置或使用默认值）
        T = self.config.get("T", 30 / 365)
        r = self.config.get("r", 0.05)
        sigma = self.config.get("sigma", 0.8)

        analysis = calculate_straddle_pnl(
            S=underlying_price,
            K=call_sig.strike_price,
            T=T,
            r=r,
            sigma=sigma,
            call_premium=call_sig.premium,
            put_premium=put_sig.premium,
            side=self.side,
        )

        return StrategyResult(
            signals=signals,
            max_profit=analysis["max_profit"],
            max_loss=analysis["max_loss"],
            breakeven=analysis["breakeven"],
            net_premium=analysis["net_premium"],
            probability_profit=analysis["prob_profit"],
            strategy_type=f"{self.side}_straddle",
        )


class IronCondorStrategy:
    """
    Iron Condor 策略

    卖出 OTM Call 和 Put，买入更远 OTM Call 和 Put
    预期价格在一定范围内波动
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        # 距离 ATM 的百分比偏移
        self.put_sell_pct = self.config.get("put_sell_pct", -0.05)   # -5% OTM
        self.put_buy_pct = self.config.get("put_buy_pct", -0.10)     # -10% OTM
        self.call_sell_pct = self.config.get("call_sell_pct", 0.05)  # +5% OTM
        self.call_buy_pct = self.config.get("call_buy_pct", 0.10)    # +10% OTM
        self.size = self.config.get("size", 1)

    def generate_signals(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> List[StrategySignal]:
        """生成 Iron Condor 交易信号"""
        signals = []

        strikes = sorted(option_chain.keys())
        if len(strikes) < 4:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        # 计算目标行权价
        put_sell_strike = atm_strike * (1 + self.put_sell_pct)
        put_buy_strike = atm_strike * (1 + self.put_buy_pct)
        call_sell_strike = atm_strike * (1 + self.call_sell_pct)
        call_buy_strike = atm_strike * (1 + self.call_buy_pct)

        # 找最近的可用行权价
        def nearest_strike(target):
            return min(strikes, key=lambda x: abs(x - target))

        put_sell_strike = nearest_strike(put_sell_strike)
        put_buy_strike = nearest_strike(put_buy_strike)
        call_sell_strike = nearest_strike(call_sell_strike)
        call_buy_strike = nearest_strike(call_buy_strike)

        # 获取价格
        put_sell_price = option_chain.get(put_sell_strike, {}).get("put_price", 0)
        put_buy_price = option_chain.get(put_buy_strike, {}).get("put_price", 0)
        call_sell_price = option_chain.get(call_sell_strike, {}).get("call_price", 0)
        call_buy_price = option_chain.get(call_buy_strike, {}).get("call_price", 0)

        if all(p > 0 for p in [put_sell_price, put_buy_price, call_sell_price, call_buy_price]):
            # Iron Condor: 4条腿
            signals.append(StrategySignal(
                action="OPEN", option_type="put", strike_price=put_sell_strike,
                side="SHORT", size=self.size, premium=put_sell_price,
            ))
            signals.append(StrategySignal(
                action="OPEN", option_type="put", strike_price=put_buy_strike,
                side="LONG", size=self.size, premium=put_buy_price,
            ))
            signals.append(StrategySignal(
                action="OPEN", option_type="call", strike_price=call_sell_strike,
                side="SHORT", size=self.size, premium=call_sell_price,
            ))
            signals.append(StrategySignal(
                action="OPEN", option_type="call", strike_price=call_buy_strike,
                side="LONG", size=self.size, premium=call_buy_price,
            ))

        return signals

    def analyze(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> StrategyResult:
        """分析 Iron Condor"""
        signals = self.generate_signals(underlying_price, option_chain)

        if not signals:
            return StrategyResult(
                signals=[],
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        # 提取各腿
        def get_sig(opt_type, side):
            return next((s for s in signals
                        if s.option_type == opt_type and s.side == side), None)

        put_sell = get_sig("put", "SHORT")
        put_buy = get_sig("put", "LONG")
        call_sell = get_sig("call", "SHORT")
        call_buy = get_sig("call", "LONG")

        if not all([put_sell, put_buy, call_sell, call_buy]):
            return StrategyResult(
                signals=signals,
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        analysis = calculate_iron_condor_pnl(
            S=underlying_price,
            put_sell_strike=put_sell.strike_price,
            put_buy_strike=put_buy.strike_price,
            call_sell_strike=call_sell.strike_price,
            call_buy_strike=call_buy.strike_price,
            put_sell_premium=put_sell.premium,
            put_buy_premium=put_buy.premium,
            call_sell_premium=call_sell.premium,
            call_buy_premium=call_buy.premium,
        )

        return StrategyResult(
            signals=signals,
            max_profit=analysis["max_profit"],
            max_loss=analysis["max_loss"],
            breakeven=analysis["breakeven"],
            net_premium=analysis["net_credit"],
            probability_profit=analysis["prob_profit"],
            strategy_type="iron_condor",
        )


class StrangleStrategy:
    """
    Strangle 策略（异价跨式套利）

    Long Strangle: 买入不同行权价的 Call 和 Put（通常 OTM）
    预期大幅波动，但成本低于 Straddle
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.put_strike_pct = self.config.get("put_strike_pct", -0.05)  # OTM Put
        self.call_strike_pct = self.config.get("call_strike_pct", 0.05)  # OTM Call
        self.side = self.config.get("side", "long")
        self.size = self.config.get("size", 1)

    def generate_signals(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> List[StrategySignal]:
        """生成 Strangle 交易信号"""
        signals = []

        strikes = sorted(option_chain.keys())
        if len(strikes) < 2:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        # 目标行权价
        put_strike = atm_strike * (1 + self.put_strike_pct)
        call_strike = atm_strike * (1 + self.call_strike_pct)

        def nearest_strike(target):
            return min(strikes, key=lambda x: abs(x - target))

        put_strike = nearest_strike(put_strike)
        call_strike = nearest_strike(call_strike)

        put_price = option_chain.get(put_strike, {}).get("put_price", 0)
        call_price = option_chain.get(call_strike, {}).get("call_price", 0)

        if put_price > 0 and call_price > 0:
            direction = 1 if self.side == "long" else -1
            signals.append(StrategySignal(
                action="OPEN",
                option_type="put",
                strike_price=put_strike,
                side=self.side.upper(),
                size=self.size * direction,
                premium=put_price,
            ))
            signals.append(StrategySignal(
                action="OPEN",
                option_type="call",
                strike_price=call_strike,
                side=self.side.upper(),
                size=self.size * direction,
                premium=call_price,
            ))

        return signals

    def analyze(
        self,
        underlying_price: float,
        option_chain: Dict[float, Dict],
    ) -> StrategyResult:
        """分析 Strangle"""
        signals = self.generate_signals(underlying_price, option_chain)

        if not signals:
            return StrategyResult(
                signals=[],
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        call_sig = next((s for s in signals if s.option_type == "call"), None)
        put_sig = next((s for s in signals if s.option_type == "put"), None)

        if not call_sig or not put_sig:
            return StrategyResult(
                signals=signals,
                max_profit=0,
                max_loss=0,
                breakeven=[],
                net_premium=0,
                probability_profit=0,
            )

        T = self.config.get("T", 30 / 365)
        r = self.config.get("r", 0.05)
        sigma = self.config.get("sigma", 0.8)

        analysis = calculate_strangle_pnl(
            S=underlying_price,
            call_strike=call_sig.strike_price,
            put_strike=put_sig.strike_price,
            call_premium=call_sig.premium,
            put_premium=put_sig.premium,
            side=self.side,
        )

        return StrategyResult(
            signals=signals,
            max_profit=analysis["max_profit"],
            max_loss=analysis["max_loss"],
            breakeven=analysis["breakeven"],
            net_premium=analysis["net_premium"],
            probability_profit=analysis["prob_profit"],
            strategy_type=f"{self.side}_strangle",
        )


class RSIMomentumStrategy:
    """
    RSI 动量策略

    RSI < 30: 买入 Call（超卖）
    RSI > 70: 买入 Put（超买）
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.strike_pct = self.config.get("strike_pct", 0.0)  # ATM = 0
        self.size = self.config.get("size", 1)

    def generate_signals(
        self,
        underlying_price: float,
        rsi_value: float,
        option_chain: Dict[float, Dict],
    ) -> List[StrategySignal]:
        """生成 RSI 动量信号"""
        signals = []

        strikes = sorted(option_chain.keys())
        if not strikes:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        target_strike = atm_strike * (1 + self.strike_pct)
        strike = min(strikes, key=lambda x: abs(x - target_strike))

        if rsi_value < self.rsi_oversold:
            # 超卖 -> 买入 Call
            call_price = option_chain.get(strike, {}).get("call_price", 0)
            if call_price > 0:
                signals.append(StrategySignal(
                    action="OPEN",
                    option_type="call",
                    strike_price=strike,
                    side="LONG",
                    size=self.size,
                    premium=call_price,
                ))

        elif rsi_value > self.rsi_overbought:
            # 超买 -> 买入 Put
            put_price = option_chain.get(strike, {}).get("put_price", 0)
            if put_price > 0:
                signals.append(StrategySignal(
                    action="OPEN",
                    option_type="put",
                    strike_price=strike,
                    side="LONG",
                    size=self.size,
                    premium=put_price,
                ))

        return signals


# ============================================================================
# 策略注册表
# ============================================================================

STRATEGY_REGISTRY = {
    "long_straddle": lambda cfg: StraddleStrategy({**(cfg or {}), "side": "long"}),
    "short_straddle": lambda cfg: StraddleStrategy({**(cfg or {}), "side": "short"}),
    "long_strangle": lambda cfg: StrangleStrategy({**(cfg or {}), "side": "long"}),
    "short_strangle": lambda cfg: StrangleStrategy({**(cfg or {}), "side": "short"}),
    "iron_condor": IronCondorStrategy,
    "rsi_momentum": RSIMomentumStrategy,
}


def get_strategy(name: str, config: Dict = None):
    """获取策略实例"""
    if name in STRATEGY_REGISTRY:
        factory = STRATEGY_REGISTRY[name]
        if callable(factory) and not isinstance(factory, type):
            return factory(config or {})
        return factory(config or {})
    raise ValueError(f"Unknown strategy: {name}")


def list_strategies() -> List[str]:
    """列出所有可用策略"""
    return list(STRATEGY_REGISTRY.keys())


# ============================================================================
# 基础策略类（兼容旧接口）
# ============================================================================

class OptionStrategy:
    """基础策略类"""
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def generate_signals(self, timestamp, underlying_price, option_chain):
        return []


class LongStraddleStrategy(OptionStrategy):
    """Long Straddle"""
    pass


class IronCondorStrategy(OptionStrategy):
    """Iron Condor"""
    pass


class CoveredCallStrategy(OptionStrategy):
    """Covered Call"""
    pass


# Alias
OptionsStrategy = OptionStrategy


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=== Options Strategies Test ===\n")

    # 模拟期权链
    strikes = [2100, 2150, 2200, 2250, 2300]
    option_chain = {}
    S = 2177
    T = 30 / 365
    r = 0.05
    sigma = 0.80

    for k in strikes:
        call_p = bs_price(S=S, K=k, T=T, r=r, sigma=sigma, option_type="call")
        put_p = bs_price(S=S, K=k, T=T, r=r, sigma=sigma, option_type="put")
        option_chain[k] = {"call_price": call_p, "put_price": put_p}

    print(f"标的: ETH ${S}")
    print(f"期权链 (T={T * 365:.0f}天, IV={sigma * 100:.0f}%):")
    for k, v in sorted(option_chain.items()):
        print(f"  K={k}: Call=${v['call_price']:.2f} Put=${v['put_price']:.2f}")
    print()

    # 测试 Long Straddle
    print("--- Long Straddle ---")
    straddle = StraddleStrategy({"side": "long"})
    result = straddle.analyze(S, option_chain)
    print(f"  Signals: {len(result.signals)} legs")
    for sig in result.signals:
        print(f"    {sig.side} {sig.option_type} K={sig.strike_price} @ ${sig.premium:.2f}")
    print(f"  Net Premium: ${result.net_premium:.2f}")
    print(f"  Max Profit: {result.max_profit}")
    print(f"  Max Loss: ${result.max_loss:.2f}")
    print(f"  Breakeven: {result.breakeven}")
    print()

    # 测试 Iron Condor
    print("--- Iron Condor ---")
    condor = IronCondorStrategy({
        "put_sell_pct": -0.05,
        "put_buy_pct": -0.10,
        "call_sell_pct": 0.05,
        "call_buy_pct": 0.10,
    })
    result = condor.analyze(S, option_chain)
    print(f"  Signals: {len(result.signals)} legs")
    for sig in result.signals:
        print(f"    {sig.side} {sig.option_type} K={sig.strike_price} @ ${sig.premium:.2f}")
    print(f"  Net Credit: ${result.net_premium:.2f}")
    print(f"  Max Profit: ${result.max_profit:.2f}")
    print(f"  Max Loss: ${result.max_loss:.2f}")
    print(f"  Breakeven: {result.breakeven}")
    print()

    # 测试 Long Strangle
    print("--- Long Strangle ---")
    strangle = StrangleStrategy({"side": "long"})
    result = strangle.analyze(S, option_chain)
    print(f"  Signals: {len(result.signals)} legs")
    for sig in result.signals:
        print(f"    {sig.side} {sig.option_type} K={sig.strike_price} @ ${sig.premium:.2f}")
    print(f"  Net Premium: ${result.net_premium:.2f}")
    print(f"  Max Profit: {result.max_profit}")
    print(f"  Max Loss: ${result.max_loss:.2f}")
    print(f"  Breakeven: {result.breakeven}")
