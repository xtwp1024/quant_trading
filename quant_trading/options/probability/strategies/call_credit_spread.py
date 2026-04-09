"""
Call Credit Spread Strategy for Probability of Profit calculations.
"""

import numpy as np
from typing import List, Dict, Optional

from ..monte_carlo import monte_carlo
from ..black_scholes_pop import black_scholes_call


def _bsm_call_credit(sim_price: np.ndarray, strikes: np.ndarray, rate: float,
                     time_fraction: float, sigma: float) -> np.ndarray:
    """BSM debit function for call credit spread."""
    P_short_calls = black_scholes_call(sim_price, strikes[0], rate, time_fraction, sigma)
    P_long_calls = black_scholes_call(sim_price, strikes[1], rate, time_fraction, sigma)
    return P_short_calls - P_long_calls


class CallCreditSpread:
    """
    Call Credit Spread - Sell a higher strike call, buy a lower strike call.

    Profit occurs when the price stays below the short strike.
    Max profit = net credit received
    """

    def __init__(
        self,
        short_strike: float,
        short_price: float,
        long_strike: float,
        long_price: float,
    ):
        """
        Initialize Call Credit Spread.

        Args:
            short_strike: Strike price of the short (sold) call
            short_price: Premium received for selling the short call
            long_strike: Strike price of the long (bought) call
            long_price: Premium paid for buying the long call
        """
        if long_price >= short_price:
            raise ValueError("Long price cannot be greater than or equal to Short price")
        if short_strike >= long_strike:
            raise ValueError("Short strike cannot be greater than or equal to Long strike")

        self.short_strike = short_strike
        self.short_price = short_price
        self.long_strike = long_strike
        self.long_price = long_price
        self.initial_credit = short_price - long_price

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

        Args:
            underlying: Current underlying price
            sigma: Implied volatility (percentage, e.g., 30 for 30%)
            rate: Risk-free interest rate (percentage, e.g., 5 for 5%)
            days_to_expiration: Days until expiration
            closing_days_array: Days at which to check profit (default: [days_to_expiration])
            profit_targets: Profit targets as percentage of max profit (default: [50])
            mc_simulations: Number of Monte Carlo trials

        Returns:
            Dict with 'pop', 'pop_error', 'avg_dtc', 'avg_dtc_error'
        """
        if closing_days_array is None:
            closing_days_array = [days_to_expiration]
        if profit_targets is None:
            profit_targets = [50]

        closing_days_array = np.array(closing_days_array)
        profit_targets = np.array([p / 100 for p in profit_targets])
        min_profit = np.array([self.initial_credit * p for p in profit_targets])

        strikes = np.array([self.short_strike, self.long_strike])

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
                bsm_func=_bsm_call_credit,
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


__all__ = ["CallCreditSpread"]
