"""
Iron Condor Strategy for Probability of Profit calculations.
"""

import numpy as np
from typing import List, Dict, Optional

from ..monte_carlo import monte_carlo
from ..black_scholes_pop import black_scholes_call, black_scholes_put


def _bsm_iron_condor(sim_price: np.ndarray, strikes: np.ndarray, rate: float,
                     time_fraction: float, sigma: float) -> np.ndarray:
    """BSM debit function for iron condor."""
    P_short_calls = black_scholes_call(sim_price, strikes[0], rate, time_fraction, sigma)
    P_long_calls = black_scholes_call(sim_price, strikes[1], rate, time_fraction, sigma)
    P_short_puts = black_scholes_put(sim_price, strikes[2], rate, time_fraction, sigma)
    P_long_puts = black_scholes_put(sim_price, strikes[3], rate, time_fraction, sigma)

    debit = P_short_calls - P_long_calls + P_short_puts - P_long_puts
    return debit


class IronCondor:
    """
    Iron Condor - Sell OTM call spread and put spread.

    Profit occurs when the price stays between the short strikes.
    Max profit = net credit received
    """

    def __init__(
        self,
        put_short_strike: float,
        put_short_price: float,
        put_long_strike: float,
        put_long_price: float,
        call_short_strike: float,
        call_short_price: float,
        call_long_strike: float,
        call_long_price: float,
    ):
        """
        Initialize Iron Condor.

        Args:
            put_short_strike: Short put strike (higher strike)
            put_short_price: Premium received for short put
            put_long_strike: Long put strike (even higher)
            put_long_price: Premium paid for long put
            call_short_strike: Short call strike (lower strike)
            call_short_price: Premium received for short call
            call_long_strike: Long call strike (even lower)
            call_long_price: Premium paid for long call
        """
        if call_long_price >= call_short_price:
            raise ValueError("Long call price cannot be greater than or equal to Short call price")
        if call_short_strike >= call_long_strike:
            raise ValueError("Short call strike cannot be greater than or equal to Long call strike")
        if put_long_price >= put_short_price:
            raise ValueError("Long put price cannot be greater than or equal to Short put price")
        if put_short_strike <= put_long_strike:
            raise ValueError("Short put strike cannot be less than or equal to Long put strike")
        if call_short_strike < put_short_strike:
            raise ValueError("Short call strike cannot be less than Short put strike")

        self.put_short_strike = put_short_strike
        self.put_short_price = put_short_price
        self.put_long_strike = put_long_strike
        self.put_long_price = put_long_price
        self.call_short_strike = call_short_strike
        self.call_short_price = call_short_price
        self.call_long_strike = call_long_strike
        self.call_long_price = call_long_price
        self.initial_credit = (
            put_short_price - put_long_price + call_short_price - call_long_price
        )

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
            profit_targets = [50]

        closing_days_array = np.array(closing_days_array)
        profit_targets = np.array([p / 100 for p in profit_targets])
        min_profit = np.array([self.initial_credit * p for p in profit_targets])

        strikes = np.array([
            self.call_short_strike,
            self.call_long_strike,
            self.put_short_strike,
            self.put_long_strike,
        ])

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
                bsm_func=_bsm_iron_condor,
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


__all__ = ["IronCondor"]
