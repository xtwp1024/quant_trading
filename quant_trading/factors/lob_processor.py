"""
LOBProcessor — Level2 订单簿数据处理器 (A股逐笔委托/成交).

Handles raw Level2 tick-by-tick data from Chinese A-share markets:
- Best Bid / Best Ask (BBO) snapshots
- Full order book depth (configurable depth levels)
- Trade tick classification (Lee-Ready tick rule)
- Order imbalance computation
- Resampling to second/minute frequencies

Designed for pure NumPy/Pandas; no Talib, no Cython.

Usage
-----
    proc = LOBProcessor(depth=10)
    tick = proc.process_tick(bid_p, bid_v, ask_p, ask_v)
    imbalance = proc.compute_imbalance()
    df = proc.resample(ticks, freq='1s')

References
----------
- Lee-Ready (1991): "Detecting the Lead-Lag Relationship in Stock Prices."
- Kyle (1985): "Continuous Auctions and Insider Trading."
-_obcf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union

__all__ = ["LOBProcessor"]


class LOBProcessor:
    """订单簿Level2数据处理器.

    Maintains a running best-bid-best-ask (BBO) snapshot and computes
    derived metrics on each tick: spread, midprice, imbalance, depth, etc.

    Parameters
    ----------
    depth : int
        Number of price levels to track on each side (default 10).
        Pass 1 for BBO-only mode.

    Attributes
    ----------
    bid_prices, bid_volumes : list[np.ndarray]
        Current order book state (depth levels).
    ask_prices, ask_volumes : list[np.ndarray]
        Current order book state.
    tick_count : int
        Number of ticks processed.

    Example
    -------
    >>> proc = LOBProcessor(depth=5)
    >>> tick = proc.process_tick(
    ...     bid_p=[10.01, 10.00, 9.99, 9.98, 9.97],
    ...     bid_v=[100, 200, 300, 400, 500],
    ...     ask_p=[10.03, 10.04, 10.05, 10.06, 10.07],
    ...     ask_v=[150, 250, 350, 450, 550],
    ... )
    >>> print(tick['midprice'], tick['spread'], tick['imbalance'])
    """

    def __init__(self, depth: int = 10) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialise or reset internal order-book state."""
        self.bid_prices: list[float] = [0.0] * self.depth
        self.bid_volumes: list[float] = [0.0] * self.depth
        self.ask_prices: list[float] = [float('inf')] * self.depth
        self.ask_volumes: list[float] = [0.0] * self.depth

        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.midprice: float = 0.0
        self.tick_count: int = 0

        # Rolling history for resampling
        self._tick_history: list[dict] = []

    # ------------------------------------------------------------------
    # Core tick processing
    # ------------------------------------------------------------------

    def process_tick(
        self,
        bid_p: Union[list, tuple, np.ndarray],
        bid_v: Union[list, tuple, np.ndarray],
        ask_p: Union[list, tuple, np.ndarray],
        ask_v: Union[list, tuple, np.ndarray],
    ) -> dict:
        """Process a single order-book tick; update internal state.

        Parameters
        ----------
        bid_p, ask_p : list[float] — price arrays (length >= depth)
        bid_v, ask_v : list[float] — volume arrays (length >= depth)

        Returns
        -------
        dict
            Snapshot with keys:
            - timestamp_us   : microseconds (set to tick_count if unknown)
            - best_bid      : float
            - best_ask      : float
            - midprice      : float
            - spread        : float (absolute)
            - spread_pct    : float (relative to midprice)
            - imbalance     : float (bid_vol / total_vol at best N levels)
            - depth         : float (sum of bid_v + ask_v at best N levels)
            - bid_depth     : float (sum of bid_v)
            - ask_depth     : float (sum of ask_v)
            - weighted_spread : float (2 × |weighted_mid - (bid+ask)/2|)
            - vwap_imbalance : float (vol-weighted mid vs simple mid)
        """
        n_bid = len(bid_p)
        n_ask = len(ask_p)

        # Update internal book — clip to self.depth
        self.bid_prices = list(bid_p[:self.depth]) + [0.0] * max(0, self.depth - n_bid)
        self.bid_volumes = list(bid_v[:self.depth]) + [0.0] * max(0, self.depth - n_bid)
        self.ask_prices = list(ask_p[:self.depth]) + [float('inf')] * max(0, self.depth - n_ask)
        self.ask_volumes = list(ask_v[:self.depth]) + [0.0] * max(0, self.depth - n_ask)

        self.best_bid = self.bid_prices[0]
        self.best_ask = self.ask_prices[0]

        if self.best_ask > 0 and self.best_bid > 0:
            self.midprice = (self.best_bid + self.best_ask) / 2.0
        else:
            self.midprice = 0.0

        self.tick_count += 1

        # Compute derived metrics
        imbalance = self.compute_imbalance()
        depth = self.compute_depth()
        bid_depth = sum(self.bid_volumes)
        ask_depth = sum(self.ask_volumes)

        spread = self.best_ask - self.best_bid
        spread_pct = (spread / self.midprice) if self.midprice > 0 else 0.0

        # Weighted midprice (depth-weighted)
        w_mid = (self.best_bid * ask_depth + self.best_ask * bid_depth) / (bid_depth + ask_depth + 1e-9)
        weighted_spread = 2.0 * abs(w_mid - self.midprice)

        tick_data = {
            'timestamp_us': self.tick_count,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'midprice': self.midprice,
            'spread': spread,
            'spread_pct': spread_pct,
            'imbalance': imbalance,
            'depth': depth,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'weighted_spread': weighted_spread,
            # Full book snapshot (for resampling)
            'bid_prices': np.array(self.bid_prices, dtype=np.float64),
            'bid_volumes': np.array(self.bid_volumes, dtype=np.float64),
            'ask_prices': np.array(self.ask_prices, dtype=np.float64),
            'ask_volumes': np.array(self.ask_volumes, dtype=np.float64),
        }

        self._tick_history.append(tick_data)
        return tick_data

    def compute_imbalance(self) -> float:
        """Compute volume-weighted order imbalance at current best N levels.

        OI = (Σ bid_vol - Σ ask_vol) / (Σ bid_vol + Σ ask_vol)

        Returns
        -------
        float
            Range [-1, +1]. Positive = bid-side pressure.
        """
        bid_total = sum(self.bid_volumes)
        ask_total = sum(self.ask_volumes)
        total = bid_total + ask_total
        if total <= 0:
            return 0.0
        return (bid_total - ask_total) / total

    def compute_depth(self, n_levels: Optional[int] = None) -> float:
        """Compute total quoted depth at best n_levels.

        Parameters
        ----------
        n_levels : int, optional
            Number of levels to sum. Defaults to self.depth.

        Returns
        -------
        float
            Sum of bid_volumes + ask_volumes at top n_levels.
        """
        if n_levels is None:
            n_levels = self.depth
        n_levels = min(n_levels, self.depth)

        bid_sum = sum(self.bid_volumes[:n_levels])
        ask_sum = sum(self.ask_volumes[:n_levels])
        return bid_sum + ask_sum

    def compute_vwap_imbalance(self) -> float:
        """Volume-weighted midprice vs simple midprice divergence.

        Returns
        -------
        float
        """
        bid_depth = sum(self.bid_volumes)
        ask_depth = sum(self.ask_volumes)
        total = bid_depth + ask_depth + 1e-9
        w_mid = (self.best_bid * ask_depth + self.best_ask * bid_depth) / total
        return w_mid - self.midprice

    # ------------------------------------------------------------------
    # Trade direction (Lee-Ready tick rule)
    # ------------------------------------------------------------------

    def classify_trade(self, trade_price: float,
                        prev_trade_price: Optional[float] = None) -> int:
        """Classify a trade as buyer-initiated (+1) or seller-initiated (-1).

        Lee-Ready tick rule: compare trade price to midprice.
        On tie (trade == mid), use previous trade price as tiebreaker.

        Parameters
        ----------
        trade_price        : float
        prev_trade_price   : float, optional

        Returns
        -------
        int
            +1 buy-initiated, -1 sell-initiated, 0 unknown.
        """
        if trade_price > self.midprice:
            return 1
        elif trade_price < self.midprice:
            return -1
        else:
            if prev_trade_price is not None:
                if trade_price > prev_trade_price:
                    return 1
                elif trade_price < prev_trade_price:
                    return -1
            return 0

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(self, ticks: list[dict], freq: str = '1s') -> pd.DataFrame:
        """Resample tick-level data to specified frequency.

        Parameters
        ----------
        ticks : list[dict]
            List of tick snapshots (each as returned by process_tick).
            Can also be the internal _tick_history.
        freq : str
            Pandas offset alias: '1s', '1min', '5s', '10s', etc.
            Default '1s'.

        Returns
        -------
        pd.DataFrame
            Resampled columns: midprice (last), spread (last), imbalance (mean),
            depth (mean), bid_depth (mean), ask_depth (mean), spread_pct (mean),
            weighted_spread (mean), trade_count (count of ticks).
        """
        if not ticks:
            return pd.DataFrame()

        df = pd.DataFrame(ticks)

        # Ensure timestamp is datetime-like index
        if 'timestamp_us' in df.columns:
            df = df.set_index('timestamp_us')

        # Numeric columns only
        scalar_cols = [
            'best_bid', 'best_ask', 'midprice', 'spread', 'spread_pct',
            'imbalance', 'depth', 'bid_depth', 'ask_depth', 'weighted_spread',
        ]

        available = [c for c in scalar_cols if c in df.columns]
        resampled = df[available].resample(freq, label='right', closed='right').agg({
            **{c: 'last' for c in ['midprice', 'spread', 'spread_pct', 'best_bid', 'best_ask']},
            **{c: 'mean' for c in ['imbalance', 'depth', 'bid_depth', 'ask_depth', 'weighted_spread']},
        })
        # Tick count per bar
        resampled['trade_count'] = df['midprice'].resample(freq, label='right', closed='right').count()

        return resampled.dropna(how='all')

    def resample_with_trades(
        self,
        ticks: list[dict],
        trade_prices: list[float],
        trade_vols: list[float],
        trade_dirs: list[int],
        freq: str = '1s',
    ) -> pd.DataFrame:
        """Resample combined order-book + trade data.

        Adds trade-side columns: buy_vol, sell_vol, trade_imbalance,
        vwap, price_impact (Kyle's lambda approximation).

        Parameters
        ----------
        ticks        : list[dict]
        trade_prices : list[float]
        trade_vols   : list[float]
        trade_dirs   : list[int] — +1 buy, -1 sell
        freq         : str

        Returns
        -------
        pd.DataFrame
        """
        if len(trade_prices) != len(trade_vols) or len(trade_prices) != len(trade_dirs):
            raise ValueError("trade_prices, trade_vols, trade_dirs must have same length")

        n = len(trade_prices)
        trade_df = pd.DataFrame({
            'timestamp_us': np.arange(n),
            'trade_price': trade_prices,
            'trade_vol': trade_vols,
            'trade_dir': trade_dirs,
        }).set_index('timestamp_us')

        lob_df = self.resample(ticks, freq=freq)

        # Align by index (simple approach: merge on floor(timestamp / bar_size))
        # For production, align using actual timestamps
        bar_trade = trade_df.resample(freq, label='right', closed='right').agg({
            'trade_price': lambda x: (x * trade_df.loc[x.index, 'trade_vol']).sum() / (trade_df.loc[x.index, 'trade_vol'].sum() + 1e-9) if len(x) > 0 else np.nan,
            'trade_vol': 'sum',
        })

        buy_vol = trade_df[trade_df['trade_dir'] > 0]['trade_vol'].resample(freq, label='right', closed='right').sum()
        sell_vol = trade_df[trade_df['trade_dir'] < 0]['trade_vol'].resample(freq, label='right', closed='right').sum()

        bar_trade.columns = ['vwap', 'total_vol']
        bar_trade['buy_vol'] = buy_vol
        bar_trade['sell_vol'] = sell_vol.fillna(0.0)
        bar_trade['trade_imbalance'] = (bar_trade['buy_vol'] - bar_trade['sell_vol']) / (bar_trade['buy_vol'] + bar_trade['sell_vol'] + 1e-9)

        result = lob_df.join(bar_trade, how='left')
        return result.dropna(how='all')

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset processor state (clear order book and history)."""
        self._reset_state()

    @property
    def state(self) -> dict:
        """Return current book state as dict (read-only copy)."""
        return {
            'depth': self.depth,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'midprice': self.midprice,
            'spread': self.best_ask - self.best_bid,
            'imbalance': self.compute_imbalance(),
            'depth_total': self.compute_depth(),
            'tick_count': self.tick_count,
        }
