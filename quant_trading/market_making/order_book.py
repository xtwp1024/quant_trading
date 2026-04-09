"""
Limit Order Book (LOB) implementation.
Adapted from Market-Making-RL/MarketMaker/book.py

A limit-order book with four operations:
- buy:  market-buy volume stocks against the lowest-priced asks
- sell: market-sell volume stocks against the highest-priced bids
- bid:  place a limit-buy order at a given price
- ask:  place a limit-sell order at a given price

Internally, bids are stored as (-price, volume) in a heap (max-heap by price).
Asks are stored as (price, volume) in a heap (min-heap by price).
"""

import heapq
import math
from typing import Optional

import numpy as np


class OrderBook:
    """
    Creates a limit-order book with four distinct actions:
    - buy:  market-buy # of stocks cheaper than some maximum price
    - sell: market-sell # of stocks more expensive than some minimum price
    - bid:  create a limit order to buy # stocks at some price
            stored as (-price, #) — highest bid tracked as (self.high_bid, self.nhigh_bid)
    - ask:  create a limit order to sell # stocks at some price
            stored as (price, #) — lowest ask tracked as (self.low_ask, self.nlow_ask)

    Also stores an evolving self.midprice, self.spread, self.delta_b, self.delta_a.
    """

    def __init__(self, baseline: float = 533.0, n_to_add: int = 100) -> None:
        # keep track of limit orders
        self.bids: list = []   # list of (-price, volume)
        self.asks: list = []   # list of (price, volume)

        # relevant pricing dynamics
        self.high_bid: float = 0.0
        self.nhigh_bid: int = 0
        self.low_ask: float = 0.0
        self.nlow_ask: int = 0

        self.midprice: float = 0.0
        self.spread: float = 0.0
        self.delta_b: float = 0.0   # distance from midprice to best bid
        self.delta_a: float = 0.0   # distance from midprice to best ask

        # Brownian motion parameters for midprice evolution
        self.drift: float = 3.59e-6
        self.scale: float = 2.4e-3
        self.max_t: float = 1.0
        self.baseline: float = baseline
        self.midprice = self.baseline

        self.n_to_add: int = n_to_add

        # Simple Brownian motion for midprice (no external dependency)
        self._brownian_t: float = 0.0

    def copy(self) -> "OrderBook":
        """Create a deep copy of the order book."""
        new_book = OrderBook(self.midprice, self.n_to_add)
        new_book.bids = [b.copy() for b in self.bids]
        new_book.asks = [a.copy() for a in self.asks]
        new_book.recalculate()
        return new_book

    def is_empty(self) -> bool:
        """Check if there are either no bids or no asks."""
        return not (len(self.bids) and len(self.asks))

    def _sample_brownian(self, dt: float) -> float:
        """Sample Brownian motion increment ~ N(0, dt)."""
        return np.random.normal(0.0, math.sqrt(dt))

    def update_midprice(self, dt: float = 1e-4) -> None:
        """Evolve the midprice by a Brownian motion step."""
        self._brownian_t += dt
        dW = self._sample_brownian(dt) * self.scale + self.drift * dt
        self.midprice += dW
        self.recalculate()

    def recalculate(self) -> None:
        """Recalculate spread, best bid/ask, and delta distances."""
        self.spread = 0.0
        self.delta_b = 0.0
        self.delta_a = 0.0
        self.low_ask = 0.0
        self.nlow_ask = 0
        self.high_bid = 0.0
        self.nhigh_bid = 0

        if len(self.asks):
            self.low_ask, self.nlow_ask = self.asks[0]
            self.delta_a = self.low_ask - self.midprice

            if len(self.bids):
                self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]

                # Prune bids above midprice
                while self.high_bid > self.midprice:
                    heapq.heappop(self.bids)
                    if len(self.bids) == 0:
                        self.bid(self.n_to_add, round(self.midprice, 2) - 0.01)
                    self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]

                # Prune asks below midprice
                while self.low_ask < self.midprice:
                    heapq.heappop(self.asks)
                    if len(self.asks) == 0:
                        self.ask(self.n_to_add, round(self.midprice, 2) + 0.01)
                    self.low_ask, self.nlow_ask = self.asks[0]

                self.spread = self.low_ask - self.high_bid
                self.delta_b = self.midprice - self.high_bid
                self.delta_a = self.low_ask - self.midprice

                if self.spread < 0:
                    # Unrealistic spread — log but don't crash
                    pass
        elif len(self.bids):
            self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]
            self.delta_b = self.midprice - self.high_bid

    def buy(self, volume: int, maxprice: float = 0.0) -> tuple[int, float]:
        """
        Market-buy up to `volume` stocks from the lowest-priced asks.
        Returns (n_bought, total_spent).
        """
        n_bought = 0
        total_spent = 0.0
        do_update = False

        while volume > 0 and len(self.asks):
            if maxprice > 0 and self.asks[0][0] > maxprice:
                break

            price, n = heapq.heappop(self.asks)
            n_bought_rn = min(n, volume)
            n_bought += n_bought_rn
            total_spent += price * n_bought_rn

            if volume < n:
                heapq.heappush(self.asks, (price, n - volume))
            else:
                do_update = True

            volume -= n

        if do_update:
            self.recalculate()

        return n_bought, total_spent

    def sell(self, volume: int, minprice: float = 0.0) -> tuple[int, float]:
        """
        Market-sell up to `volume` stocks to the highest-priced bids.
        Returns (n_sold, total_received).
        """
        n_sold = 0
        total_received = 0.0
        do_update = False

        while volume > 0 and len(self.bids):
            if minprice > 0 and -self.bids[0][0] < minprice:
                break

            neg_price, n = heapq.heappop(self.bids)
            price = -neg_price
            n_sold_rn = min(n, volume)
            n_sold += n_sold_rn
            total_received += price * n_sold_rn

            if volume < n:
                heapq.heappush(self.bids, (-price, n - volume))
            else:
                do_update = True

            volume -= n

        if do_update:
            self.recalculate()

        return n_sold, total_received

    def bid(self, volume: int, price: float) -> None:
        """
        Add a limit-buy order (bid) at `price` for `volume` shares.
        Sorted highest-to-lowest price in the heap.
        """
        price = round(price, 2)
        if volume == 0:
            return

        # Negative volume means widen spread by selling into the book
        if volume < 0:
            self.sell(-volume, minprice=price)
            self.recalculate()
            return

        # If bid price >= lowest ask, eat into the asks
        if len(self.asks):
            if price >= self.asks[0][0]:
                nbought, _ = self.buy(volume, maxprice=price)
                volume -= nbought
                if volume == 0:
                    return

        heapq.heappush(self.bids, (-price, volume))
        if len(self.bids) and price == -self.bids[0][0]:
            self.recalculate()

    def ask(self, volume: int, price: float) -> None:
        """
        Add a limit-sell order (ask) at `price` for `volume` shares.
        Sorted lowest-to-highest price in the heap.
        """
        price = round(price, 2)
        if volume == 0:
            return

        # Negative volume means widen spread by buying from the book
        if volume < 0:
            self.buy(-volume, maxprice=price)
            self.recalculate()
            return

        # If ask price <= highest bid, sell into the bids
        if len(self.bids):
            if price <= -self.bids[0][0]:
                nsold, _ = self.sell(volume, minprice=price)
                volume -= nsold
                if volume == 0:
                    return

        heapq.heappush(self.asks, (price, volume))
        if len(self.asks) and price == self.asks[0][0]:
            self.recalculate()

    def __str__(self) -> str:
        return (
            f"{self.nhigh_bid}, -${self._fmt(self.delta_b)} | "
            f"{self._fmt(self.midprice)} | "
            f"{self.nlow_ask}, +${self._fmt(self.delta_a)}"
        )

    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:.4f}"
