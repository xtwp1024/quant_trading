"""
Probability of Profit (POP) Module

Monte Carlo-based probability calculations for options strategies.
Based on poptions library by Dev b.

Usage:
    from quant_trading.options.probability import (
        MonteCarlo, ProbabilityEngine,
        CallCreditSpread, PutCreditSpread, IronCondor,
        LongStrangle, CoveredCall, LongCall, LongPut
    )

Example:
    # Calculate POP for a call credit spread
    spread = CallCreditSpread(
        short_strike=105,
        short_price=2.00,
        long_strike=110,
        long_price=0.50,
    )
    result = spread.probability_of_profit(
        underlying=100,
        sigma=30,
        rate=5,
        days_to_expiration=30,
        mc_simulations=10000,
    )
    print(f"POP: {result['pop']}%")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Import Monte Carlo engine
from .monte_carlo import monte_carlo

# Import Black-Scholes functions for POP
from .black_scholes_pop import black_scholes_call, black_scholes_put

# Import strategies
from .strategies import (
    CallCreditSpread,
    PutCreditSpread,
    IronCondor,
    LongStrangle,
    CoveredCall,
    LongCall,
    LongPut,
)


@dataclass
class POPResult:
    """
    Probability of Profit Result.

    Attributes:
        pop: Probability of profit (percentage)
        pop_error: Error margin for POP (95% confidence interval)
        avg_dtc: Average days to close profitable trades
        avg_dtc_error: Error margin for average DTC
    """
    pop: float
    pop_error: float
    avg_dtc: float
    avg_dtc_error: float


class MonteCarlo:
    """
    Monte Carlo simulation engine for POP calculations.

    This is a convenience wrapper around the monte_carlo function.
    """

    @staticmethod
    def simulate(
        underlying: float,
        rate: float,
        sigma: float,
        days_to_expiration: int,
        closing_days_array: List[int],
        trials: int,
        initial_credit: float,
        min_profit: List[float],
        strikes: List[float],
        bsm_func: callable,
    ) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            underlying: Current underlying price
            rate: Risk-free interest rate (percentage)
            sigma: Volatility (percentage)
            days_to_expiration: Days until expiration
            closing_days_array: Days at which to check profit
            trials: Number of simulation trials
            initial_credit: Initial credit/debit from opening the trade
            min_profit: List of minimum profit targets
            strikes: List of strike prices
            bsm_func: Black-Scholes pricing function

        Returns:
            Dict with 'pop', 'pop_error', 'avg_dtc', 'avg_dtc_error'
        """
        import numpy as np

        pop, pop_error, avg_dtc, avg_dtc_error = monte_carlo(
            underlying=underlying,
            rate=rate,
            sigma=sigma,
            days_to_expiration=days_to_expiration,
            closing_days_array=np.array(closing_days_array),
            trials=trials,
            initial_credit=initial_credit,
            min_profit=np.array(min_profit),
            strikes=np.array(strikes),
            bsm_func=bsm_func,
        )

        return {
            "pop": pop,
            "pop_error": pop_error,
            "avg_dtc": avg_dtc,
            "avg_dtc_error": avg_dtc_error,
        }


class ProbabilityEngine:
    """
    High-level API for calculating Probability of Profit for options strategies.

    This class provides a unified interface for POP calculations across
    different strategy types.

    Example:
        engine = ProbabilityEngine()

        # Call credit spread
        result = engine.calculate_pop(
            strategy_type="call_credit_spread",
            underlying=100,
            sigma=30,
            rate=5,
            days_to_expiration=30,
            short_strike=105,
            short_price=2.00,
            long_strike=110,
            long_price=0.50,
        )

        # Iron condor
        result = engine.calculate_pop(
            strategy_type="iron_condor",
            underlying=100,
            sigma=30,
            rate=5,
            days_to_expiration=30,
            put_short_strike=95,
            put_short_price=1.50,
            put_long_strike=90,
            put_long_price=0.50,
            call_short_strike=105,
            call_short_price=2.00,
            call_long_strike=110,
            call_long_price=0.50,
        )
    """

    def __init__(self, default_simulations: int = 10000):
        """
        Initialize ProbabilityEngine.

        Args:
            default_simulations: Default number of Monte Carlo simulations
        """
        self.default_simulations = default_simulations

    def calculate_pop(
        self,
        strategy_type: str,
        underlying: float,
        sigma: float,
        rate: float,
        days_to_expiration: int,
        mc_simulations: Optional[int] = None,
        closing_days: Optional[int] = None,
        profit_target: float = 50,
        **strategy_params,
    ) -> POPResult:
        """
        Calculate probability of profit for a strategy.

        Args:
            strategy_type: Type of strategy ('call_credit_spread', 'put_credit_spread',
                          'iron_condor', 'long_strangle', 'covered_call', 'long_call', 'long_put')
            underlying: Current underlying price
            sigma: Implied volatility (percentage)
            rate: Risk-free interest rate (percentage)
            days_to_expiration: Days until expiration
            mc_simulations: Number of Monte Carlo simulations
            closing_days: Days at which to check profit (default: days_to_expiration)
            profit_target: Profit target as percentage or multiple (depending on strategy)
            **strategy_params: Strategy-specific parameters

        Returns:
            POPResult with pop, pop_error, avg_dtc, avg_dtc_error
        """
        if mc_simulations is None:
            mc_simulations = self.default_simulations
        if closing_days is None:
            closing_days = days_to_expiration

        strategy_type = strategy_type.lower().replace("-", "_").replace(" ", "_")

        if strategy_type == "call_credit_spread":
            strategy = CallCreditSpread(
                short_strike=strategy_params["short_strike"],
                short_price=strategy_params["short_price"],
                long_strike=strategy_params["long_strike"],
                long_price=strategy_params["long_price"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type == "put_credit_spread":
            strategy = PutCreditSpread(
                short_strike=strategy_params["short_strike"],
                short_price=strategy_params["short_price"],
                long_strike=strategy_params["long_strike"],
                long_price=strategy_params["long_price"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type == "iron_condor":
            strategy = IronCondor(
                put_short_strike=strategy_params["put_short_strike"],
                put_short_price=strategy_params["put_short_price"],
                put_long_strike=strategy_params["put_long_strike"],
                put_long_price=strategy_params["put_long_price"],
                call_short_strike=strategy_params["call_short_strike"],
                call_short_price=strategy_params["call_short_price"],
                call_long_strike=strategy_params["call_long_strike"],
                call_long_price=strategy_params["call_long_price"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type == "long_strangle":
            strategy = LongStrangle(
                call_strike=strategy_params["call_strike"],
                call_price=strategy_params["call_price"],
                put_strike=strategy_params["put_strike"],
                put_price=strategy_params["put_price"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type == "covered_call":
            strategy = CoveredCall(
                short_strike=strategy_params["short_strike"],
                short_price=strategy_params["short_price"],
                underlying_cost=strategy_params.get("underlying_cost", underlying),
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type in ("long_call", "call"):
            strategy = LongCall(
                strike=strategy_params["strike"],
                premium=strategy_params["premium"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        elif strategy_type in ("long_put", "put"):
            strategy = LongPut(
                strike=strategy_params["strike"],
                premium=strategy_params["premium"],
            )
            result = strategy.probability_of_profit(
                underlying=underlying,
                sigma=sigma,
                rate=rate,
                days_to_expiration=days_to_expiration,
                closing_days_array=[closing_days],
                profit_targets=[profit_target],
                mc_simulations=mc_simulations,
            )

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Extract first element from lists (since we pass single values)
        return POPResult(
            pop=result["pop"][0] if result["pop"] else 0,
            pop_error=result["pop_error"][0] if result["pop_error"] else 0,
            avg_dtc=result["avg_dtc"][0] if result["avg_dtc"] else 0,
            avg_dtc_error=result["avg_dtc_error"][0] if result["avg_dtc_error"] else 0,
        )


__all__ = [
    # Core Monte Carlo
    "monte_carlo",
    "MonteCarlo",
    "POPResult",
    # Black-Scholes for POP
    "black_scholes_call",
    "black_scholes_put",
    # Strategies
    "CallCreditSpread",
    "PutCreditSpread",
    "IronCondor",
    "LongStrangle",
    "CoveredCall",
    "LongCall",
    "LongPut",
    # Engine
    "ProbabilityEngine",
]
