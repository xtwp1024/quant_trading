"""
Avellaneda-Stoikov reward framework.
Adapted from Market-Making-RL/MarketMaker/rewards.py

The reward function implements the Avellaneda-Stoikov market-making model:

1. dW (wealth change): cash flow from filled market orders
2. Inventory penalty: risk-adjusted cost of holding inventory
3. Time penalty: encourage acting earlier vs later
4. Reservation price: indifference price given current inventory and time

Reservation price:
    r(S_t, q, T-t) = S_t - q * gamma * sigma^2 * (T-t)

Optimal spread:
    s*(T-t) = gamma * sigma^2 * (T-t) + 2/gamma * log(1 + 2*gamma / (alpha*(K_b+K_s)))
"""

import math
from typing import Optional

import numpy as np


class AvellanedaStoikovReward:
    """
    Avellaneda-Stoikov reward calculator.

    Supports:
    - Immediate reward: a * dW + inventory_penalty * sign(dI) * exp(-b * time_left) + c * (max_t - time_left)
    - Final reward: W_T + I_T * S_T (liquidate inventory at midprice)

    Args:
        gamma:      risk aversion parameter (avellaneda gamma)
        sigma:      volatility of midprice (midprice diffusion coefficient)
        max_t:      terminal time
        a_weight:   weight on wealth change in immediate reward
        inventory_penalty: coefficient on inventory risk term
        time_penalty:      coefficient on time (earlier = better)
        add_inventory:     include exponential inventory term
        add_time:           include linear time term
    """

    def __init__(
        self,
        gamma: float = 1.0,
        sigma: float = 1e-2,
        max_t: float = 1.0,
        a_weight: Optional[float] = None,
        inventory_penalty: float = 1.0,
        time_penalty: float = 0.0,
        add_inventory: bool = True,
        add_time: bool = False,
        midprice: float = 533.0,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.max_t = max_t
        self.midprice = midprice

        # Reward weights
        max_dW = midprice * 3000
        self.a = a_weight if a_weight is not None else 1.0 / max_dW
        self.b = 10.0 / max_t       # inventory decay rate
        self.c = 1.0 / max_t       # time weight

        self.inventory_penalty = inventory_penalty
        self.time_penalty = time_penalty
        self.add_inventory = add_inventory
        self.add_time = add_time

    def reward(
        self,
        dW: float,
        dI: int,
        time_left: float,
    ) -> float:
        """
        Compute the immediate reward for one step.

        Args:
            dW:       change in wealth (cash) this step
            dI:       change in inventory this step
            time_left: time remaining (seconds)

        Returns:
            float: immediate reward
        """
        rew = self.a * dW

        if self.add_inventory:
            # Exponential decay inventory penalty — encourages reducing inventory near end
            inv_term = math.exp(-self.b * time_left) * math.copysign(1.0, dI)
            rew -= self.inventory_penalty * inv_term

        if self.add_time:
            # Linear time penalty (discourage waiting)
            rew -= self.c * (self.max_t - time_left)

        return rew

    def final_reward(
        self,
        wealth: float,
        inventory: int,
        midprice: float,
    ) -> float:
        """
        Terminal reward: liquidate inventory at midprice.

        V_T = W_T + q_T * S_T
        """
        return wealth + inventory * midprice

    def wealth_reward(
        self,
        dW: float,
    ) -> float:
        """Pure wealth-change reward (dW weighted)."""
        return self.a * dW

    def reservation_price(
        self,
        midprice: float,
        inventory: int,
        t_left: float,
    ) -> float:
        """
        Reservation (indifference) price under Avellaneda-Stoikov.

        r(S_t, q, T-t) = S_t - q * gamma * sigma^2 * (T-t)

        The market maker is indifferent between having inventory q
        and cash when the asset is priced at the reservation price.
        """
        return midprice - inventory * self.gamma * self.sigma ** 2 * t_left

    def optimal_spread(
        self,
        t_left: float,
        alpha: float = 1.53,
        K_b: float = 1.0,
        K_s: float = 1.0,
    ) -> float:
        """
        Optimal symmetric spread per Avellaneda-Stoikov.

        s*(T-t) = gamma * sigma^2 * (T-t)
                + 2/gamma * log(1 + 2*gamma / (alpha*(K_b+K_s)))
        """
        return (
            self.gamma * self.sigma ** 2 * t_left
            + 2.0 / self.gamma * math.log(1 + 2.0 * self.gamma / (alpha * (K_b + K_s)))
        )

    def compute_reward_state(
        self,
        dW: float,
        dI: int,
        time_left: float,
    ) -> np.ndarray:
        """
        Build reward-state vector for logging/debugging.

        Returns:
            np.ndarray: [dW, dI, time_left, reservation_price, optimal_spread]
        """
        t_left = max(time_left, 0.0)
        res_price = self.reservation_price(self.midprice, dI, t_left)
        opt_spread = self.optimal_spread(t_left)
        return np.array([dW, dI, t_left, res_price, opt_spread], dtype=np.float32)


def make_reward_function(
    gamma: float = 1.0,
    sigma: float = 1e-2,
    max_t: float = 1.0,
    immediate_reward: bool = True,
    add_inventory: bool = True,
    add_time: bool = False,
    always_final: bool = False,
    inventory_penalty: float = 1.0,
    time_penalty: float = 0.0,
    midprice: float = 533.0,
) -> AvellanedaStoikovReward:
    """Factory to build an AvellanedaStoikovReward from simple flags."""
    return AvellanedaStoikovReward(
        gamma=gamma,
        sigma=sigma,
        max_t=max_t,
        add_inventory=add_inventory,
        add_time=add_time,
        inventory_penalty=inventory_penalty,
        time_penalty=time_penalty,
        midprice=midprice,
    )
