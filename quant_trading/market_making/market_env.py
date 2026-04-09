"""
Market making environment — Gymnasium-compatible wrapper.
Adapted from Market-Making-RL/MarketMaker/market.py

Provides:
- BaseMarket: order book simulation with inventory/wealth tracking
- MarketEnv:  Gymnasium.Env wrapper compatible with stable-baselines3 PPO
"""

import math
from typing import Optional

import gymnasium as gym
import numpy as np

from quant_trading.market_making.order_book import OrderBook


class BaseMarket:
    """
    Market environment with order-book simulation.

    Tracks:
    - Inventory (I): net position in the asset
    - Wealth   (W): cash balance
    - OrderBook: limit-order book state

    Action space (4D): (n_bid, delta_b, n_ask, delta_a)
        n_bid:   number of shares to bid
        delta_b: bid offset below midprice
        n_ask:   number of shares to ask
        delta_a: ask offset above midprice

    Observation space (4D): (n_bid, bid_price, n_ask, ask_price)
        n_bid:   volume at best bid
        bid_price: distance of best bid from midprice
        n_ask:   volume at best ask
        ask_price: distance of best ask from midprice
    """

    # Market-order intensity parameters (Avellaneda-Stoikov)
    alpha: float = 1.53        # order arrival sensitivity
    Lambda_b: float = 20.0     # avg buy volume per step
    Lambda_s: float = 20.0     # avg sell volume per step
    K_b: float = 1.0           # buy intensity decay
    K_s: float = 1.0           # sell intensity decay

    # Market dynamics coefficients:
    #   lambda_buy  = exp(betas[0] + betas[1]*log(1+delta) + betas[2]*log(1+delta)^2
    #                       + betas[3]*log(1+q)  + betas[4]*log(1+q)^2
    #                       + betas[5]*log(delta+1+q))
    betas: tuple = (7.2, -2.13, -0.8, -2.3, 0.167, -0.1)

    def __init__(
        self,
        inventory: int = 0,
        wealth: float = 0.0,
        *,
        midprice: float = 533.0,
        spread: float = 10.0,
        nstocks: int = 10000,
        make_bell: bool = True,
        nsteps: int = 1000,
        substeps: int = 1,
        max_t: float = 1.0,
        dt: float = 1e-3,
        discount: float = 0.9999,
        sigma: float = 1e-2,
        gamma: float = 1.0,
        # Avellaneda-Stoikov parameters
        gamma_as: float = 1.0,
        sigma_as: float = 1e-2,
    ):
        self.I: int = inventory          # current inventory
        self.W: float = wealth           # current wealth (cash)

        # Order-book configuration
        self.midprice: float = midprice
        self.spread: float = spread
        self.nstocks: int = nstocks
        self.make_bell: bool = make_bell
        self.nsteps: int = nsteps
        self.substeps: int = substeps
        self.betas = self.betas

        # Temporal parameters
        self.max_t: float = max_t
        self.dt: float = dt
        self.discount: float = discount

        # Avellaneda-Stoikov parameters
        self.gamma_as = gamma_as
        self.sigma_as = sigma_as

        # Build order book
        self.book: OrderBook = OrderBook(baseline=self.midprice)

    # ------------------------------------------------------------------
    # Market dynamics
    # ------------------------------------------------------------------

    def lambda_buy(self, delta: float, q: int) -> float:
        """Intensity of incoming buy (market) orders at offset delta with q shares on ask side."""
        if not q:
            return 0.0
        try:
            lam = math.exp(
                self.betas[0]
                + self.betas[1] * math.log(1 + delta)
                + self.betas[2] * math.log(1 + delta) ** 2
                + self.betas[3] * math.log(1 + q)
                + self.betas[4] * math.log(1 + q) ** 2
                + self.betas[5] * math.log(delta + 1 + q)
            )
            return max(0.0, lam)
        except (ValueError, OverflowError):
            return 0.0

    def lambda_sell(self, delta: float, q: int) -> float:
        """Intensity of incoming sell (market) orders at offset delta with q shares on bid side."""
        return self.lambda_buy(delta, q)

    def step(self, nsteps: int = 1) -> tuple[float, int, float]:
        """
        Evolve the market by `nsteps` steps.

        Returns:
            dW:   change in wealth (cash)
            dI:   change in inventory
            mid:  new midprice
        """
        dW = 0.0
        dI = 0

        for _ in range(nsteps):
            self.book.update_midprice(self.dt)

            try:
                lambda_buy = self.lambda_buy(self.book.delta_a, self.book.nlow_ask)
                lambda_sell = self.lambda_sell(self.book.delta_b, self.book.nhigh_bid)
                nbuy = np.random.poisson(lambda_buy)
                nsell = np.random.poisson(lambda_sell)
            except (ValueError, OverflowError):
                return 0.0, 0, self.book.midprice

            # Market buy: lift the ask (pay cash, gain inventory)
            n_ask_lift, bought = self.book.buy(nbuy)
            # Market sell: hit the bid (receive cash, lose inventory)
            n_bid_hit, sold = self.book.sell(nsell)

            dW += bought - sold
            dI += n_bid_hit - n_ask_lift

        self.I += dI
        self.W += dW

        return dW, dI, self.book.midprice

    # ------------------------------------------------------------------
    # State & action
    # ------------------------------------------------------------------

    def state(self) -> tuple[int, float, int, float]:
        """
        Return current order-book observation.

        Returns:
            (n_bid, bid_price, n_ask, ask_price)
        """
        return (
            int(self.book.nhigh_bid),
            float(self.book.delta_b),
            int(self.book.nlow_ask),
            float(self.book.delta_a),
        )

    def act(
        self,
        action: tuple | np.ndarray,
        policy=None,
    ) -> tuple:
        """
        Submit a limit-order action to the book.

        Action: (n_bid, delta_b, n_ask, delta_a)
            n_bid:   number of shares to bid (limit buy)
            delta_b: bid offset below midprice
            n_ask:   number of shares to ask (limit sell)
            delta_a: ask offset above midprice
        """
        if len(action) == 5:  # (n_bid, bid_price_abs, n_ask, ask_price_abs, time_left)
            n_bid, bid_price, n_ask, ask_price, time_left = action
        elif len(action) == 4:
            n_bid, delta_b, n_ask, delta_a = action
            bid_price = self.book.midprice - delta_b
            ask_price = self.book.midprice + delta_a
        else:
            raise ValueError(f"Invalid action length: {len(action)}")

        self.submit(n_bid, delta_b if len(action) == 4 else bid_price,
                    n_ask, delta_a if len(action) == 4 else ask_price)
        return action

    def submit(
        self,
        n_bid: float,
        delta_or_price_b: float,
        n_ask: float,
        delta_or_price_a: float,
    ) -> None:
        """
        Submit limit orders.

        Args:
            n_bid:   number of bid shares
            delta_or_price_b: either delta_b (if < midprice) or absolute bid price
            n_ask:   number of ask shares
            delta_or_price_a: either delta_a (if > midprice) or absolute ask price
        """
        delta_b = max(delta_or_price_b, 0)
        delta_a = max(delta_or_price_a, 0)

        n_bid = round(n_bid)
        n_ask = round(n_ask)

        # Clamp so bid < ask always
        bid_price = min(self.book.midprice - delta_b, self.book.low_ask - 0.01)
        ask_price = max(self.book.midprice + delta_a, self.book.high_bid + 0.01)

        self.book.bid(n_bid, bid_price)
        self.book.ask(n_ask, ask_price)

    def reset(
        self,
        *,
        mid: Optional[float] = None,
        spread: Optional[float] = None,
        nstocks: Optional[int] = None,
        nsteps: Optional[int] = None,
        substeps: Optional[int] = None,
        make_bell: Optional[bool] = None,
    ) -> None:
        """Reset the market to initial conditions."""
        mid = mid if mid is not None else self.midprice
        spread = spread if spread is not None else self.spread
        nstocks = nstocks if nstocks is not None else self.nstocks
        nsteps = nsteps if nsteps is not None else self.nsteps
        substeps = substeps if substeps is not None else self.substeps
        make_bell = make_bell if make_bell is not None else self.make_bell

        self.book = OrderBook(mid)
        self.I = 0
        self.W = 0.0

        if make_bell:
            tot_amount = 0
            while tot_amount < nstocks:
                delta_b = np.random.normal(spread / 2, spread / 4)
                delta_a = np.random.normal(spread / 2, spread / 4)
                nbid = np.random.poisson(2 * nstocks / nsteps)
                nask = np.random.poisson(2 * nstocks / nsteps)
                self.book.bid(nbid, mid - delta_b)
                self.book.ask(nask, mid + delta_a)
                tot_amount += nask + nbid
        else:
            self.book.bid(nstocks // 2, mid - spread / 2)
            self.book.ask(nstocks // 2, mid + spread / 2)

    def is_empty(self) -> bool:
        return self.book.is_empty()

    # ------------------------------------------------------------------
    # Avellaneda-Stoikov helpers
    # ------------------------------------------------------------------

    def reservation_price(self, midprice: float, inventory: int, t_left: float) -> float:
        """
        Compute the reservation (indifference) price.

        r = S_t - q * gamma * sigma^2 * T
        """
        return midprice - inventory * self.gamma_as * self.sigma_as ** 2 * t_left

    def optimal_spread(self, t_left: float) -> float:
        """
        Compute the optimal symmetric spread per Avellaneda-Stoikov.

        spread = gamma * sigma^2 * T + 2 * log(1 + 2*gamma/(alpha*(K_b+K_s))) / gamma
        """
        return (
            self.gamma_as * self.sigma_as ** 2 * t_left
            + 2 * math.log(1 + 2 * self.gamma_as / (self.alpha * (self.K_b + self.K_s))) / self.gamma_as
        )


# -------------------------------------------------------------------------- #
# Gymnasium Environment
# -------------------------------------------------------------------------- #


class MarketEnv(gym.Env):
    """
    Gymnasium wrapper for the market-making environment.

    Compatible with stable-baselines3 PPO and other off-the-shelf RL algorithms.

    Action space: Box(0, inf, (4,)) — (n_bid, delta_b, n_ask, delta_a)
    Observation:  Box(0, inf, (4,)) — (n_bid, delta_b, n_ask, delta_a)
        Plus: info dict with 'time_left', 'wealth', 'inventory', 'midprice'
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_t: float = 1.0,
        dt: float = 1e-3,
        discount: float = 0.9999,
        midprice: float = 533.0,
        spread: float = 10.0,
        nstocks: int = 10000,
        make_bell: bool = True,
        nsteps: int = 1000,
        substeps: int = 1,
        gamma_as: float = 1.0,
        sigma_as: float = 1e-2,
        inventory_penalty: float = 1.0,
        time_penalty: float = 0.0,
        immediate_reward: bool = True,
        always_final: bool = False,
    ):
        super().__init__()

        self.max_t = max_t
        self.dt = dt
        self.nt = math.ceil(max_t / dt)
        self.discount = discount
        self.immediate_reward = immediate_reward
        self.always_final = always_final
        self.inventory_penalty = inventory_penalty
        self.time_penalty = time_penalty

        # Market config
        self.midprice_cfg = midprice
        self.spread_cfg = spread
        self.nstocks_cfg = nstocks
        self.make_bell_cfg = make_bell
        self.nsteps_cfg = nsteps
        self.substeps_cfg = substeps
        self.gamma_as = gamma_as
        self.sigma_as = sigma_as

        # Derived
        self._max_inventory = nstocks

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([float(nstocks), spread * 5, float(nstocks), spread * 5]),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([float(nstocks), spread * 10, float(nstocks), spread * 10]),
            dtype=np.float32,
        )

        # Internal state
        self._market: Optional[BaseMarket] = None
        self._t: int = 0
        self._W0: float = 0.0
        self._I0: int = 0
        self._terminal: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment."""
        super().reset(seed=seed)

        self._market = BaseMarket(
            inventory=0,
            wealth=0.0,
            midprice=self.midprice_cfg,
            spread=self.spread_cfg,
            nstocks=self.nstocks_cfg,
            make_bell=self.make_bell_cfg,
            nsteps=self.nsteps_cfg,
            substeps=self.substeps_cfg,
            max_t=self.max_t,
            dt=self.dt,
            discount=self.discount,
            gamma_as=self.gamma_as,
            sigma_as=self.sigma_as,
        )
        self._market.reset()
        self._t = 0
        self._W0 = 0.0
        self._I0 = 0
        self._terminal = False

        obs = np.array(self._market.state(), dtype=np.float32)
        info = self._make_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ):
        """Execute one market-making step."""
        if self._market is None:
            raise RuntimeError("Call reset() before step()")

        # Clamp action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        n_bid, delta_b, n_ask, delta_a = action

        # Submit limit orders
        self._market.submit(n_bid, delta_b, n_ask, delta_a)

        # Market dynamics step
        dW, dI, mid = self._market.step(self.substeps_cfg)

        # Time tracking
        self._t += 1
        t_left = max(0.0, self.max_t - self._t * self.dt)
        time_frac = self._t / max(self.nt, 1)

        # Reward
        if self.immediate_reward:
            # dW weighted reward
            max_dW = self.midprice_cfg * 3000
            a_weight = 1.0 / max_dW

            rew = a_weight * dW

            # Inventory penalty: discourage large positions
            inv = self._market.I
            rew -= self.inventory_penalty * abs(inv) / self._max_inventory

            # Time penalty
            if self.time_penalty > 0:
                rew -= self.time_penalty * time_frac
        else:
            rew = 0.0

        # Termination
        self._terminal = (
            self._t >= self.nt
            or self._market.is_empty()
            or t_left <= 0
        )

        if self._terminal and (self._t >= self.nt or self.always_final):
            # Final reward: liquidate inventory at midprice
            final_val = self._market.W + self._market.I * mid
            rew += final_val - (self._W0 + self._I0 * self.midprice_cfg)

        obs = np.array(self._market.state(), dtype=np.float32)
        info = self._make_info(t_left=t_left)

        return obs, rew, self._terminal, False, info

    def _make_info(self, t_left: Optional[float] = None) -> dict:
        if t_left is None:
            t_left = max(0.0, self.max_t - self._t * self.dt)
        return {
            "time_left": t_left,
            "wealth": self._market.W if self._market else 0.0,
            "inventory": self._market.I if self._market else 0,
            "midprice": self._market.book.midprice if self._market else self.midprice_cfg,
            "dW": 0.0,  # filled by caller
            "dI": 0,    # filled by caller
        }

    def close(self) -> None:
        self._market = None
