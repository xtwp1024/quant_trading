"""
Put Credit Spread Strategy for Probability of Profit calculations.
"""

import numpy as np
from typing import List, Dict, Optional

from ..monte_carlo import monte_carlo
from ..black_scholes_pop import black_scholes_put


def _bsm_put_credit(sim_price: np.ndarray, strikes: np.ndarray, rate: float,
                    time_fraction: float, sigma: float) -> np.ndarray:
    """BSM debit function for put credit spread."""
    P_short_puts = black_scholes_put(sim_price, strikes[0], rate, time_fraction, sigma)
    P_long_puts = black_scholes_put(sim_price, strikes[1], rate, time_fraction, sigma)
    return P_short_puts - P_long_puts


class PutCreditSpread:
    """
    Put Credit Spread - Sell a lower strike put, buy a higher strike put.

    Profit occurs when the price stays above the short strike.
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
        Initialize Put Credit Spread.

        Args:
            short_strike: Strike price of the short (sold) put
            short_price: Premium received for selling the short put
            long_strike: Strike price of the long (bought) put
            long_price: Premium paid for buying the long put
        """
        if long_price >= short_price:
            raise ValueError("Long price cannot be greater than or equal to Short price")
        if short_strike <= long_strike:
            raise ValueError("Short strike cannot be less than or equal to Long strike")

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
                bsm_func=_bsm_put_credit,
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


__all__ = ["PutCreditSpread"]
