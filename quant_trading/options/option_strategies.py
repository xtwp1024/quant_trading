"""
期权策略POP计算 — Probability of Profit Monte Carlo
Option Strategy POP Calculation using Monte Carlo

支持策略:
    - Call Credit Spread / Call信用价差
    - Put Credit Spread / Put信用价差
    - Long Strangle / 多头宽跨式
    - Iron Condor / 铁鹰式

核心方法: 在风险中性测度下模拟标的价格路径,
计算到期日期权组合的盈亏, 统计盈利概率 (POP).

Author: 量化之神系统
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .monte_carlo import MonteCarloEngine

__all__ = [
    "OptionStrategy",
    "CallCreditSpread",
    "PutCreditSpread",
    "LongStrangle",
    "IronCondor",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class OptionStrategy:
    """期权策略基类 — Base class for option strategies.

    提供统一的 POP / Max Profit / Max Loss / Breakeven 接口。
    子类只需实现 _payoff_at_expiry() 核心方法。

    Args:
        engine: MonteCarloEngine 实例 / MC引擎
        n_simulations: POP计算的模拟次数 (default: 100,000)
    """

    def __init__(
        self,
        engine: Optional[MonteCarloEngine] = None,
        n_simulations: int = 100000,
    ):
        self.engine = engine or MonteCarloEngine(
            n_paths=n_simulations,
            n_steps=252,
            seed=42,
            use_numba=True,
        )
        self.n_simulations = n_simulations

    def pop(self, **kwargs) -> float:
        """Probability of Profit / 盈利概率 (POP).

        通过Monte Carlo模拟计算策略盈利的概率。
        Returns: POP as a fraction (0.0 to 1.0)
        """
        S_paths = self.engine.simulate_paths(
            S0=kwargs.get("S0", kwargs.get("S", 100.0)),
            mu=kwargs.get("r", 0.05),  # drift = risk-free rate (risk-neutral)
            sigma=kwargs.get("sigma", 0.2),
            T=kwargs.get("T", 1.0),
        )
        # Shape: (n_paths, n_steps+1); last column = terminal price
        terminal_prices = S_paths[:, -1]

        # Compute payoff for each path
        payoffs = self._payoff_at_expiry(terminal_prices, **kwargs)

        # Net cash flow: payoff + net premium received
        net = payoffs + self._net_premium()
        profitable = np.sum(net > 0)
        return float(profitable / self.n_simulations)

    def max_profit(self) -> float:
        """理论最大盈利 (absolute value of best outcome)."""
        raise NotImplementedError("Subclass must implement max_profit()")

    def max_loss(self) -> float:
        """理论最大亏损 (absolute value of worst outcome)."""
        raise NotImplementedError("Subclass must implement max_loss()")

    def breakeven(self) -> list[float]:
        """盈亏平衡点列表."""
        raise NotImplementedError("Subclass must implement breakeven()")

    def _net_premium(self) -> float:
        """Net premium received (positive = credit, negative = debit)."""
        raise NotImplementedError("Subclass must implement _net_premium()")

    def _payoff_at_expiry(
        self, terminal_prices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """计算到期日盈亏矩阵.

        Args:
            terminal_prices: (n_paths,) array of terminal stock prices
            **kwargs: subclass-specific parameters

        Returns:
            (n_paths,) array of payoffs at expiry (before net premium)
        """
        raise NotImplementedError("Subclass must implement _payoff_at_expiry()")


# ---------------------------------------------------------------------------
# Credit Spreads
# ---------------------------------------------------------------------------

class CallCreditSpread(OptionStrategy):
    """Call Credit Spread — 卖出低行权价Call, 买入更高行权价Call.

    构造方式 (build-up):
        - Short Call @ short_strike (sell put to open)
        - Long Call @ long_strike  (buy call to hedge)
        - Net credit = short_premium - long_premium

    盈亏图:
        Profit:  S < short_strike           → net_credit
        Loss:    S > long_strike            → -(long_strike - short_strike - net_credit)
        Between: short_strike < S < long_strike → linear interpolation

    Example:
        >>> engine = MonteCarloEngine(n_paths=100000, use_numba=True)
        >>> spread = CallCreditSpread(
        ...     short_strike=105,
        ...     long_strike=110,
        ...     short_premium=3.0,
        ...     long_premium=1.5,
        ...     engine=engine,
        ... )
        >>> pop = spread.pop(S0=100, sigma=0.2, T=0.5)
        >>> print(f"POP: {pop:.2%}")
    """

    def __init__(
        self,
        short_strike: float,
        long_strike: float,
        short_premium: float,
        long_premium: float,
        engine: Optional[MonteCarloEngine] = None,
        n_simulations: int = 100000,
    ):
        super().__init__(engine=engine, n_simulations=n_simulations)
        self.short_strike = short_strike
        self.long_strike = long_strike
        self.short_premium = short_premium
        self.long_premium = long_premium

        # Validate
        if long_strike <= short_strike:
            raise ValueError("long_strike must be > short_strike")
        if short_premium <= long_premium:
            raise ValueError("short_premium should exceed long_premium for a credit spread")

    def _net_premium(self) -> float:
        """Net credit received."""
        return self.short_premium - self.long_premium

    def max_profit(self) -> float:
        """最大盈利 = net credit received."""
        return self._net_premium()

    def max_loss(self) -> float:
        """最大亏损 = spread width - net credit."""
        spread_width = self.long_strike - self.short_strike
        return -(spread_width - self._net_premium())

    def breakeven(self) -> list[float]:
        """盈亏平衡点 = short_strike + net_credit."""
        return [self.short_strike + self._net_premium()]

    def _payoff_at_expiry(
        self, terminal_prices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Call Credit Spread到期 payoff.

        Long call pays max(S_T - long_strike, 0)
        Short call pays max(S_T - short_strike, 0)
        Net = Long - Short
        """
        long_payoff = np.maximum(terminal_prices - self.long_strike, 0.0)
        short_payoff = np.maximum(terminal_prices - self.short_strike, 0.0)
        return long_payoff - short_payoff


class PutCreditSpread(OptionStrategy):
    """Put Credit Spread — 卖出高行权价Put, 买入更低行权价Put.

    构造方式:
        - Short Put @ short_strike (higher strike, sell)
        - Long Put @ long_strike   (lower strike, buy as hedge)
        - Net credit = short_premium - long_premium

    盈亏图:
        Profit:  S > short_strike           → net_credit
        Loss:    S < long_strike            → -(long_strike - short_strike - net_credit)
        Between: short_strike > S > long_strike → linear interpolation

    Example:
        >>> spread = PutCreditSpread(
        ...     short_strike=95,
        ...     long_strike=90,
        ...     short_premium=3.0,
        ...     long_premium=1.5,
        ... )
        >>> pop = spread.pop(S0=100, sigma=0.2, T=0.5)
    """

    def __init__(
        self,
        short_strike: float,
        long_strike: float,
        short_premium: float,
        long_premium: float,
        engine: Optional[MonteCarloEngine] = None,
        n_simulations: int = 100000,
    ):
        # long_strike < short_strike for put spread
        super().__init__(engine=engine, n_simulations=n_simulations)
        self.short_strike = short_strike
        self.long_strike = long_strike
        self.short_premium = short_premium
        self.long_premium = long_premium

        if long_strike >= short_strike:
            raise ValueError("long_strike must be < short_strike for put spread")
        if short_premium <= long_premium:
            raise ValueError("short_premium should exceed long_premium for a credit spread")

    def _net_premium(self) -> float:
        return self.short_premium - self.long_premium

    def max_profit(self) -> float:
        return self._net_premium()

    def max_loss(self) -> float:
        spread_width = self.short_strike - self.long_strike
        return -(spread_width - self._net_premium())

    def breakeven(self) -> list[float]:
        return [self.short_strike - self._net_premium()]

    def _payoff_at_expiry(
        self, terminal_prices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Put Credit Spread到期 payoff.

        Long put pays max(long_strike - S_T, 0)
        Short put pays max(short_strike - S_T, 0)
        Net = Long - Short
        """
        long_payoff = np.maximum(self.long_strike - terminal_prices, 0.0)
        short_payoff = np.maximum(self.short_strike - terminal_prices, 0.0)
        return long_payoff - short_payoff


# ---------------------------------------------------------------------------
# Long Strangle
# ---------------------------------------------------------------------------

class LongStrangle(OptionStrategy):
    """Long Strangle — 买入Call + 买入Put (strike不重合).

    构造:
        - Long Call @ call_strike (higher)
        - Long Put  @ put_strike  (lower)
        - Net debit = call_premium + put_premium

    盈亏图:
        Profit if S_T > call_strike + net_debit OR S_T < put_strike - net_debit
        Loss   if put_strike - net_debit < S_T < call_strike + net_debit

    Example:
        >>> strangle = LongStrangle(
        ...     call_strike=105,
        ...     put_strike=95,
        ...     call_premium=2.0,
        ...     put_premium=2.0,
        ... )
        >>> pop = strangle.pop(S0=100, sigma=0.25, T=0.5)
    """

    def __init__(
        self,
        call_strike: float,
        put_strike: float,
        call_premium: float,
        put_premium: float,
        engine: Optional[MonteCarloEngine] = None,
        n_simulations: int = 100000,
    ):
        super().__init__(engine=engine, n_simulations=n_simulations)
        self.call_strike = call_strike
        self.put_strike = put_strike
        self.call_premium = call_premium
        self.put_premium = put_premium

        if call_strike <= put_strike:
            raise ValueError("call_strike must be > put_strike for strangle")
        if call_premium <= 0 or put_premium <= 0:
            raise ValueError("Premiums must be positive")

    def _net_premium(self) -> float:
        """Net debit (negative, since we pay premiums)."""
        return -(self.call_premium + self.put_premium)

    def max_profit(self) -> float:
        """理论上无限 (call side) — 实际中是标的上涨幅度 - net_debit."""
        return float("inf")

    def max_loss(self) -> float:
        """最大亏损 = total premium paid."""
        return -(self.call_premium + self.put_premium)

    def breakeven(self) -> list[float]:
        """两个盈亏平衡点:
        - Upper: call_strike + total_premium
        - Lower: put_strike - total_premium
        """
        total_premium = self.call_premium + self.put_premium
        return [
            self.put_strike - total_premium,
            self.call_strike + total_premium,
        ]

    def _payoff_at_expiry(
        self, terminal_prices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Long Strangle到期 payoff."""
        call_payoff = np.maximum(terminal_prices - self.call_strike, 0.0)
        put_payoff = np.maximum(self.put_strike - terminal_prices, 0.0)
        return call_payoff + put_payoff


# ---------------------------------------------------------------------------
# Iron Condor
# ---------------------------------------------------------------------------

class IronCondor(OptionStrategy):
    """Iron Condor — 双重信用价差 (Call Spread + Put Spread).

    构造:
        - Put Spread:  Short Put @ put_short_strike, Long Put @ put_long_strike
        - Call Spread: Short Call @ call_short_strike, Long Call @ call_long_strike

    标准Iron Condor布局 (bearish on both sides):
        put_short_strike < put_long_strike < call_short_strike < call_long_strike

    盈利区间:
        - 若标的价格在 [put_long_strike, call_short_strike] 之间 → 收取全部信用
    亏损区间:
        - S_T <= put_long_strike: Put Spread亏损 (put_long_strike - put_short_strike - net_credit)
        - S_T >= call_short_strike: Call Spread亏损 (call_long_strike - call_short_strike - net_credit)

    Example:
        >>> engine = MonteCarloEngine(n_paths=100000, use_numba=True)
        >>> condor = IronCondor(
        ...     put_spread=(95, 90, 3.0, 1.5),    # (short_strike, long_strike, short_premium, long_premium)
        ...     call_spread=(105, 110, 3.0, 1.5), # (short_strike, long_strike, short_premium, long_premium)
        ...     engine=engine,
        ... )
        >>> pop = condor.pop(S0=100, sigma=0.2, T=0.5)
        >>> print(f"POP: {pop:.2%}, Max Loss: {condor.max_loss():.2f}")
    """

    def __init__(
        self,
        put_spread: tuple[float, float, float, float],
        call_spread: tuple[float, float, float, float],
        engine: Optional[MonteCarloEngine] = None,
        n_simulations: int = 100000,
    ):
        """Initialize Iron Condor.

        Args:
            put_spread: (short_strike, long_strike, short_premium, long_premium)
                       short_strike > long_strike (OTM put, e.g. 95/90)
            call_spread: (short_strike, long_strike, short_premium, long_premium)
                        short_strike < long_strike (OTM call, e.g. 105/110)
            engine: MonteCarloEngine instance
            n_simulations: Number of MC paths
        """
        super().__init__(engine=engine, n_simulations=n_simulations)

        put_short, put_long, put_short_p, put_long_p = put_spread
        call_short, call_long, call_short_p, call_long_p = call_spread

        # Validate put spread
        if put_long >= put_short:
            raise ValueError("Put spread: long_strike must be < short_strike")
        # Validate call spread
        if call_short >= call_long:
            raise ValueError("Call spread: short_strike must be < long_strike")

        # Store
        self.put_short_strike = put_short
        self.put_long_strike = put_long
        self.put_short_premium = put_short_p
        self.put_long_premium = put_long_p

        self.call_short_strike = call_short
        self.call_long_strike = call_long
        self.call_short_premium = call_short_p
        self.call_long_premium = call_long_p

        # Pre-build sub-strategies for convenience
        self._put_spread_obj = PutCreditSpread(
            short_strike=put_short,
            long_strike=put_long,
            short_premium=put_short_p,
            long_premium=put_long_p,
            engine=engine,
            n_simulations=n_simulations,
        )
        self._call_spread_obj = CallCreditSpread(
            short_strike=call_short,
            long_strike=call_long,
            short_premium=call_short_p,
            long_premium=call_long_p,
            engine=engine,
            n_simulations=n_simulations,
        )

    def _net_premium(self) -> float:
        """Total net credit received."""
        return (
            self._put_spread_obj._net_premium()
            + self._call_spread_obj._net_premium()
        )

    def max_profit(self) -> float:
        """最大盈利 = total net credit."""
        return self._net_premium()

    def max_loss(self) -> float:
        """最大亏损 = sum of (spread_width - credit) for each spread."""
        put_loss = abs(self._put_spread_obj.max_loss())
        call_loss = abs(self._call_spread_obj.max_loss())
        return -(put_loss + call_loss)

    def breakeven(self) -> list[float]:
        """两个盈亏平衡点:
        - Lower: put_short - net_credit_put_spread
        - Upper: call_short + net_credit_call_spread
        """
        put_breakeven = self._put_spread_obj.breakeven()[0]
        call_breakeven = self._call_spread_obj.breakeven()[0]
        return [put_breakeven, call_breakeven]

    def _payoff_at_expiry(
        self, terminal_prices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Iron Condor到期 payoff = Put Spread + Call Spread payoff.

        Put Spread payoff (bearish: profits when price stays above short strike):
            Long Put  @ put_long_strike  → max(put_long_strike - S_T, 0)
            Short Put @ put_short_strike → max(put_short_strike - S_T, 0)
            Net = Long - Short

        Call Spread payoff (bullish: profits when price stays below short strike):
            Long Call  @ call_long_strike  → max(S_T - call_long_strike, 0)
            Short Call @ call_short_strike → max(S_T - call_short_strike, 0)
            Net = Long - Short

        Combined:
            When S_T <= put_long_strike:    put_spread_loss + call_spread_max_profit
            When S_T >= call_short_strike:  call_spread_loss + put_spread_max_profit
            When put_long_strike < S_T < call_short_strike: both spreads profit max
        """
        # Put spread payoff
        put_long_payoff = np.maximum(self.put_long_strike - terminal_prices, 0.0)
        put_short_payoff = np.maximum(self.put_short_strike - terminal_prices, 0.0)
        put_spread_payoff = put_long_payoff - put_short_payoff

        # Call spread payoff
        call_long_payoff = np.maximum(terminal_prices - self.call_long_strike, 0.0)
        call_short_payoff = np.maximum(terminal_prices - self.call_short_strike, 0.0)
        call_spread_payoff = call_long_payoff - call_short_payoff

        return put_spread_payoff + call_spread_payoff
