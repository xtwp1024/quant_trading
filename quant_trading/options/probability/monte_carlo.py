"""
Monte Carlo Simulation for Options Probability of Profit (POP)

Based on poptions library by Dev b.
https://github.com/deltaonealpha/poptions

Key assumptions:
- Stock price volatility equals implied volatility and remains constant
- Geometric Brownian Motion models stock price
- Risk-free interest rates remain constant
- Black-Scholes Model prices options contracts
- No dividend yield considered
- No commissions or assignment risks considered
"""

import numpy as np

# Try to import numba for JIT compilation, fall back to pure Python
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator when numba is not available
    def jit(func):
        return func


def monte_carlo(
    underlying: float,
    rate: float,
    sigma: float,
    days_to_expiration: int,
    closing_days_array: np.ndarray,
    trials: int,
    initial_credit: float,
    min_profit: np.ndarray,
    strikes: np.ndarray,
    bsm_func: callable,
) -> tuple:
    """
    Run Monte Carlo simulation to calculate probability of profit.

    Args:
        underlying: Current underlying price
        rate: Risk-free interest rate (percentage)
        sigma: Volatility (percentage)
        days_to_expiration: Days until expiration
        closing_days_array: Array of days at which to check profit
        trials: Number of simulation trials
        initial_credit: Initial credit/debit from opening the trade
        min_profit: Array of minimum profit targets (same length as closing_days_array)
        strikes: Array of strike prices for the strategy
        bsm_func: Black-Scholes pricing function for the strategy

    Returns:
        Tuple of (pop_values, pop_errors, avg_dtc, avg_dtc_errors)
        - pop_values: Probability of profit for each closing day (percentage)
        - pop_errors: Error margin for each POP value
        - avg_dtc: Average days to close for profitable trades
        - avg_dtc_errors: Error margin for average DTC
    """
    dt = 1 / 365  # 365 calendar days in a year

    length = len(closing_days_array)
    max_closing_days = int(max(closing_days_array))

    sigma = sigma / 100
    rate = rate / 100

    counter1 = [0] * length
    dtc = [0] * length
    dtc_history = np.zeros((length, trials))

    indices = [0] * length

    for c in range(trials):

        epsilon_cum = 0
        t_cum = 0

        for i in range(length):
            indices[i] = 0

        # +1 added to account for first day. sim_prices[0,...] = underlying price.
        for r in range(max_closing_days + 1):

            # Brownian Motion
            W = (dt ** (1 / 2)) * epsilon_cum

            # Geometric Brownian Motion
            signal = (rate - 0.5 * (sigma ** 2)) * t_cum
            noise = sigma * W
            y = noise + signal
            stock_price = underlying * np.exp(y)  # Stock price on current day
            epsilon = np.random.randn()
            epsilon_cum += epsilon
            t_cum += dt

            # Prevents crashes
            if stock_price <= 0:
                stock_price = 0.001

            debit = bsm_func(stock_price, strikes, rate, dt * (days_to_expiration - r), sigma)

            profit = initial_credit - debit  # Profit if we were to close on current day
            sum_val = 0

            for i in range(length):
                if indices[i] == 1:  # Checks if combo has been evaluated
                    sum_val += 1
                    continue
                else:
                    if min_profit[i] <= profit:  # If target profit hit, combo has been evaluated
                        counter1[i] += 1
                        dtc[i] += r
                        dtc_history[i, c] = r

                        indices[i] = 1
                        sum_val += 1
                    elif r >= closing_days_array[i]:  # If closing days passed, combo has been evaluated
                        indices[i] = 1
                        sum_val += 1

            if sum_val == length:  # If all combos evaluated, break and start new trial
                break

    pop_counter1 = [c / trials * 100 for c in counter1]
    pop_counter1 = [round(x, 2) for x in pop_counter1]

    # Taken from Eq. 2.20 from Monte Carlo theory, methods and examples, by Art B. Owen
    pop_counter1_err = [2.58 * (x * (100 - x) / trials) ** (1 / 2) for x in pop_counter1]
    pop_counter1_err = [round(x, 2) for x in pop_counter1_err]

    avg_dtc = []  # Average days to close
    avg_dtc_error = []

    # Taken from Eq. 2.34 from Monte Carlo theory, methods and examples, by Art B. Owen, 2013
    for index in range(length):
        if counter1[index] > 0:
            avg_dtc.append(dtc[index] / counter1[index])

            n_a = counter1[index]
            mu_hat_a = dtc[index] / n_a
            summation = 0

            for value in dtc_history[index, :]:
                if value == 0:  # if 0 then it means that min_profit wasn't hit
                    continue

                summation = summation + (value - mu_hat_a) ** 2

            s_a_squared = (1 / n_a) * summation  # changed from n_a - 1 for simplicity
            std_dev = ((n_a - 1) * s_a_squared) ** (1 / 2) / n_a
            avg_dtc_error.append(2.58 * std_dev)

        else:
            avg_dtc.append(0)
            avg_dtc_error.append(0)

    avg_dtc = [round(x, 2) for x in avg_dtc]

    avg_dtc_error = [round(x, 2) for x in avg_dtc_error]

    return pop_counter1, pop_counter1_err, avg_dtc, avg_dtc_error


# Export the function
__all__ = ["monte_carlo"]
