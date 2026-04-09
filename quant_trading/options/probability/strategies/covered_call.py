"""
Covered Call Strategy for Probability of Profit calculations.
"""

import numpy as np
from typing import List, Dict, Optional

from ..monte_carlo import monte_carlo
from ..black_scholes_pop import black_scholes_call


def _bsm_covered_call(sim_price: np.ndarray, strikes: np.ndarray, rate: float,
                      time_fraction: float, sigma: float) -> np.ndarray:
    """BSM debit function for covered call."""
    P_short_calls = black_scholes_call(sim_price, strikes[0], rate, time_fraction, sigma)
    debit = P_short_calls
    credit = sim_price
    debit = debit - credit
    return debit


class CoveredCall:
    """
    Covered Call - Own stock, sell call against it.

    Profit occurs when stock stays below short call strike or if called away.
    Max profit = call premium + (short_strike - underlying)
    """

    def __init__(
        self,
        short_strike: float,
        short_price: float,
        underlying_cost: float,
    ):
        """
        Initialize Covered Call.

        Args:
            short_strike: Strike price of the short (sold) call
            short_price: Premium received for selling the call
            underlying_cost: Cost basis of the underlying shares
        """
        self.short_strike = short_strike
        self.short_price = short_price
        self.underlying_cost = underlying_cost
        # Credit received minus stock cost
        self.initial_credit = short_price - underlying_cost
        self.max_profit = short_price + (short_strike - underlying_cost)

    def probability_of_profit(
        self,
        underlying: float,
        sigma: float,
        rate: float,
        days_to_expiration: int,
        closing_days_array: Optional[List[int]] = None,
        profit_targets: Optional[List[float]] = None,
        mc_simulations: int = 10000,
    ) -> Dict:
        """
        Calculate probability of profit using Monte Carlo simulation.
        """
        if closing_days_array is None:
            closing_days_array = [days_to_expiration]
        if profit_targets is None:
            profit_targets = [50]  # 50% of max profit

        closing_days_array = np.array(closing_days_array)
        profit_targets = np.array([p / 100 for p in profit_targets])
        min_profit = np.array([self.max_profit * p for p in profit_targets])

        strikes = np.array([self.short_strike])

        for closing_days in closing_days_array:
            if closing_days > days_to_expiration:
                raise ValueError("Closing days cannot be beyond Days To Expiration.")

        if len(closing_days_array) != len(profit_targets):
            raise ValueError("closing_days_array and profit_targets sizes must be equal.")

        try:
            pop, pop_error, avg_dtc, avg_dtc_error = monte_carlo(
                underlying=underlying,
                rate=rate,
                sigma=sigma,
                days_to_expiration=days_to_expiration,
                closing_days_array=closing_days_array,
                trials=mc_simulations,
                initial_credit=self.initial_credit,
                min_profit=min_profit,
                strikes=strikes,
                bsm_func=_bsm_covered_call,
            )
        except RuntimeError as err:
            print(err.args)
            return {"pop": 0, "pop_error": 0, "avg_dtc": 0, "avg_dtc_error": 0}

        return {
            "pop": pop,
            "pop_error": pop_error,
            "avg_dtc": avg_dtc,
            "avg_dtc_error": avg_dtc_error,
        }


__all__ = ["CoveredCall"]
