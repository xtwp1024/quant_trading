"""
Long Put Strategy for Probability of Profit calculations.
"""

import numpy as np
from typing import List, Dict, Optional

from ..monte_carlo import monte_carlo
from ..black_scholes_pop import black_scholes_put


def _bsm_long_put(sim_price: np.ndarray, strikes: np.ndarray, rate: float,
                  time_fraction: float, sigma: float) -> np.ndarray:
    """BSM debit function for long put."""
    P_long_puts = black_scholes_put(sim_price, strikes[0], rate, time_fraction, sigma)
    credit = P_long_puts
    debit = -credit
    return debit


class LongPut:
    """
    Long Put - Buy a put option.

    Profit occurs when stock falls below strike - premium paid.
    Max loss = premium paid
    """

    def __init__(
        self,
        strike: float,
        premium: float,
    ):
        """
        Initialize Long Put.

        Args:
            strike: Strike price of the put
            premium: Premium paid for the put
        """
        self.strike = strike
        self.premium = premium
        self.initial_debit = premium
        self.initial_credit = -premium

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
            closing_days_array: Days at which to check profit
            profit_targets: Profit targets as multiple of premium (e.g., 2.0 for 2x)
            mc_simulations: Number of Monte Carlo trials

        Returns:
            Dict with 'pop', 'pop_error', 'avg_dtc', 'avg_dtc_error'
        """
        if closing_days_array is None:
            closing_days_array = [days_to_expiration]
        if profit_targets is None:
            profit_targets = [2.0]  # 2x profit target

        closing_days_array = np.array(closing_days_array)
        profit_targets = np.array(profit_targets)
        min_profit = np.array([self.initial_debit * p for p in profit_targets])

        strikes = np.array([self.strike])

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
                bsm_func=_bsm_long_put,
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


__all__ = ["LongPut"]
