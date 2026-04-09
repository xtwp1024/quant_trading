"""
Spread trading data generators.

Adapted from Trading-Gym (https://github.com/FinanceDataEngine/Trading-Gym).
Gymnasium-compatible data pipeline for spread trading environments.

Classes:
- DataGenerator: Abstract base for data generators
- CSVStreamer: Load historical OHLC/price data from CSV files
- RandomWalk: Random walk generator for one product (bid/ask)
- AR1: AR(1) stationary process generator
- RandomGenerator: Multi-product synthetic data generator combining
  RandomWalk/AR1 per product with configurable bid-ask spreads

Usage:
    from quant_trading.rl.spread_data_generator import CSVStreamer, RandomGenerator

    # CSV: each row = [p1_bid, p1_ask, p2_bid, p2_ask, ...]
    dg = CSVStreamer(filename="data.csv")

    # Synthetic: one generator per product, combined into spread
    dg = RandomGenerator(generators=[RandomWalk(ba_spread=0.01),
                                      AR1(a=0.5, ba_spread=0.01)])
"""

import csv
import numpy as np


def calc_spread(prices, spread_coefficients):
    """Calculate the spread bid/ask based on spread_coefficients.

    Args:
        prices (np.ndarray): Array of (bid, ask) pairs per product,
            i.e. [p1_bid, p1_ask, p2_bid, p2_ask].
        spread_coefficients (list): Signed coefficients — positive means
            buy the product, negative means sell.

    Returns:
        tuple: (spread_bid, spread_ask)
    """
    spread_bid = sum(
        spread_coefficients[i] *
        prices[2 * i + int(spread_coefficients[i] < 0)]
        for i in range(len(spread_coefficients))
    )
    spread_ask = sum(
        spread_coefficients[i] *
        prices[2 * i + int(spread_coefficients[i] > 0)]
        for i in range(len(spread_coefficients))
    )
    return spread_bid, spread_ask


class DataGenerator:
    """Abstract base class for data generators.

    Subclasses must implement ``_generator()`` as a static method yielding
    rows of bid/ask prices (one row per product: ``[bid, ask, bid, ask, ...]``).

    The ``next()`` method manages iterator exhaustion and calls
    ``_iterator_end()`` when the generator is exhausted, allowing
    subclasses to implement rewind-on-end behaviour (see ``CSVStreamer``).
    """

    def __init__(self, **gen_kwargs):
        self._trainable = False
        self.gen_kwargs = gen_kwargs
        DataGenerator.rewind(self)
        self.n_products = len(self.next()) // 2
        DataGenerator.rewind(self)

    @staticmethod
    def _generator(**kwargs):
        """Generator function. Must be overridden by subclasses.

        Yields:
            np.ndarray: Row of bid/ask prices for all products.
        """
        raise NotImplementedError()

    def next(self):
        """Return the next row of price data.

        Returns:
            np.ndarray: Next row of prices.
        """
        try:
            return next(self.generator)
        except StopIteration:
            self._iterator_end()
            raise

    def rewind(self):
        """Rewind the generator to the beginning."""
        self.generator = self._generator(**self.gen_kwargs)

    def _iterator_end(self):
        """Called when the iterator is exhausted. Override for custom behaviour."""
        pass


class CSVStreamer(DataGenerator):
    """Data generator that streams rows from a CSV file.

    Each row should contain an even number of columns:
    ``[p1_bid, p1_ask, p2_bid, p2_ask, ...]``

    When the end of the file is reached the stream rewinds automatically.

    Args:
        filename (str): Path to the CSV file.
        header (bool): True if the file has a header row (default False).
    """

    @staticmethod
    def _generator(filename, header=False):
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            if header:
                next(reader, None)
            for row in reader:
                assert len(row) % 2 == 0
                yield np.array(row, dtype=np.float64)

    def _iterator_end(self):
        print("End of data reached, rewinding.")
        self.rewind()

    def rewind(self):
        """Only rewind after explicit call; not automatically on EOF."""
        self.generator = self._generator(**self.gen_kwargs)


class RandomWalk(DataGenerator):
    """Random walk generator for one product.

    Yields:
        tuple: (bid_price, ask_price) where ask = bid + ba_spread
    """

    @staticmethod
    def _generator(ba_spread=0.0):
        val = 0.0
        while True:
            yield val
            val += np.random.standard_normal()

    def __init__(self, ba_spread=0.0):
        self._ba_spread = ba_spread
        super().__init__(ba_spread=ba_spread)

    def next(self):
        raw = super().next()
        bid = raw
        ask = bid + self._ba_spread
        return np.array([bid, ask], dtype=np.float64)


class AR1(DataGenerator):
    """AR(1) stationary process generator for one product.

    The process is: ``x_t = a * x_{t-1} + epsilon_t`` with epsilon ~ N(0, sigma)
    where sigma is chosen so x is standardised (variance = 1).

    Args:
        a (float): AR1 coefficient, must satisfy |a| < 1.
        ba_spread (float): Bid-ask spread (default 0).
    """

    @staticmethod
    def _generator(a=0.0, ba_spread=0.0):
        assert abs(a) < 1
        sigma = np.sqrt(1 - a ** 2)
        val = np.random.normal(scale=sigma)
        while True:
            yield val
            val = (a - 1) * val + np.random.normal(scale=sigma)

    def __init__(self, a=0.0, ba_spread=0.0):
        self._ba_spread = ba_spread
        super().__init__(a=a, ba_spread=ba_spread)

    def next(self):
        raw = super().next()
        bid = raw
        ask = bid + self._ba_spread
        return np.array([bid, ask], dtype=np.float64)


class RandomGenerator(DataGenerator):
    """Multi-product synthetic data generator.

    Combines one sub-generator per product into a single flat array
    compatible with ``SpreadTrading``.

    Each sub-generator should be an instance of ``RandomWalk`` or ``AR1``.
    When ``next()`` is called all sub-generators are advanced and their
    bid/ask prices are concatenated.

    Args:
        generators (list): List of ``DataGenerator`` instances (one per product).
    """

    def __init__(self, generators):
        self._generators = list(generators)
        self._n_products = len(self._generators)
        # Initialise the base class without kwargs — we override next() entirely
        self._trainable = False
        self.gen_kwargs = {}
        DataGenerator.rewind(self)
        self.n_products = self._n_products
        DataGenerator.rewind(self)

    @staticmethod
    def _generator(**kwargs):
        """Dummy generator — ``next()`` is overridden in ``RandomGenerator``."""
        return np.array([], dtype=np.float64)

    def next(self):
        rows = [gen.next() for gen in self._generators]
        return np.concatenate(rows)

    def rewind(self):
        for gen in self._generators:
            gen.rewind()
        self.generator = iter([])  # dummy so base-class next() works
