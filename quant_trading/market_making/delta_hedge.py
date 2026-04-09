"""
Delta Hedging Engine for Options Market Making.

This module provides a standalone delta hedging engine that calculates and
executes delta-neutral strategies for multi-leg options portfolios.

Based on the Optiver competition-winning algorithm (4th highest profitability
among 23 teams in the LSE Optiver Algorithmic Trading Competition).

Features:
- Portfolio-level delta aggregation across options, futures, and stocks
- Threshold-based hedge execution with IOC orders
- Support for dual-listed stocks and multiple underlyings
- Position limit aware hedging

Usage:
    engine = DeltaHedgeEngine(volatility=3.0, interest_rate=0.03)
    delta = engine.calculate_portfolio_delta(options, positions, stock_value)
    hedge = engine.calculate_hedge(delta, stock_position, order_book)
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

from .optiver_black_scholes import call_delta, put_delta


@dataclass
class OptionPosition:
    """Option position with instrument details."""
    instrument_id: str
    expiry: dt.datetime
    strike: float
    option_kind: str  # 'call' or 'put'
    position: int  # current position quantity


@dataclass
class FuturePosition:
    """Future position details."""
    instrument_id: str
    expiry: dt.datetime
    position: int


@dataclass
class HedgeResult:
    """
    Result of delta hedge calculation.

    Attributes
    ----------
    aggregate_delta : float
        Total portfolio delta across all instruments.
    hedge_volume : int
        Volume to trade for the hedge.
    hedge_side : str or None
        'bid' (buy stock) or 'ask' (sell stock) or None (no hedge needed).
    hedge_price : float or None
        Price at which to execute the hedge.
    """
    aggregate_delta: float
    hedge_volume: int
    hedge_side: Optional[str]
    hedge_price: Optional[float] = None


@dataclass
class HedgedPortfolio:
    """
    Complete delta analysis for a portfolio.

    Attributes
    ----------
    total_delta : float
        Net delta of the entire portfolio.
    option_deltas : dict[str, float]
        Delta contribution from each option.
    future_delta : float
        Delta from futures (1:1 exposure).
    stock_delta : float
        Delta from stock positions.
    needs_hedge : bool
        Whether a hedge is recommended.
    hedge_result : HedgeResult
        Details of the recommended hedge.
    """
    total_delta: float
    option_deltas: dict[str, float]
    future_delta: float
    stock_delta: float
    needs_hedge: bool
    hedge_result: HedgeResult


class DeltaHedgeEngine:
    """
    Delta hedging engine for options market making.

    Calculates aggregate portfolio delta and determines optimal hedge trades
    to maintain delta-neutral exposure.

    The engine aggregates delta from:
    - Options (using Black-Scholes model)
    - Futures (1:1 exposure)
    - Stock positions (1:1 exposure)

    When aggregate delta exceeds the threshold, it recommends executing
    a hedge trade in the opposite direction in the underlying stock.

    Example
    -------
    >>> engine = DeltaHedgeEngine(volatility=2.5, interest_rate=0.03)
    >>> options = {
    ...     'NVDA_CALL_100': OptionPosition(
    ...         instrument_id='NVDA_CALL_100',
    ...         expiry=dt.datetime(2026, 6, 20),
    ...         strike=100.0,
    ...         option_kind='call',
    ...         position=10
    ...     )
    ... }
    >>> portfolio = engine.analyze_portfolio(
    ...     options=options,
    ...     option_positions={'NVDA_CALL_100': 10},
    ...     futures_positions={},
    ...     stock_position=5,
    ...     stock_dual_position=0,
    ...     stock_value=150.0
    ... )
    >>> print(f"Total delta: {portfolio.total_delta:.4f}")
    """

    # Default thresholds for hedge execution
    DEFAULT_HEDGE_THRESHOLD = 35.0
    DEFAULT_MAX_HEDGE_VOLUME = 100

    # Threshold multipliers for different stocks (SAN is more sensitive)
    SAN_THRESHOLD_MULTIPLIER = 1.0 / 35.0  # SAN threshold is 1.0

    def __init__(
        self,
        volatility: float = 3.0,
        interest_rate: float = 0.03,
        hedge_threshold: float = DEFAULT_HEDGE_THRESHOLD,
        max_hedge_volume: int = DEFAULT_MAX_HEDGE_VOLUME,
    ):
        """
        Initialize the delta hedge engine.

        Parameters
        ----------
        volatility : float
            Implied volatility for Black-Scholes delta calculations.
        interest_rate : float
            Risk-free interest rate (annualized).
        hedge_threshold : float
            Delta threshold that triggers hedging. Default 35.
        max_hedge_volume : int
            Maximum volume per hedge trade. Default 100.
        """
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.hedge_threshold = hedge_threshold
        self.max_hedge_volume = max_hedge_volume

    def calculate_time_to_expiry(self, expiry_date: dt.datetime) -> float:
        """
        Calculate time to expiry in years from current time.

        Parameters
        ----------
        expiry_date : dt.datetime
            Expiration datetime of the option.

        Returns
        -------
        float
            Time to expiry in years.
        """
        now = dt.datetime.now()
        return (expiry_date - now) / dt.timedelta(days=1) / 365

    def calculate_option_delta(
        self,
        strike: float,
        expiry: dt.datetime,
        option_kind: str,
        stock_value: float,
    ) -> float:
        """
        Calculate delta of a single option using Black-Scholes.

        Parameters
        ----------
        strike : float
            Strike price of the option.
        expiry : dt.datetime
            Expiration datetime of the option.
        option_kind : str
            'call' or 'put'.
        stock_value : float
            Current underlying stock price.

        Returns
        -------
        float
            Option delta.
        """
        time_to_expiry = self.calculate_time_to_expiry(expiry)

        if option_kind.lower() == 'call':
            return call_delta(
                S=stock_value,
                K=strike,
                T=time_to_expiry,
                r=self.interest_rate,
                sigma=self.volatility,
            )
        else:
            return put_delta(
                S=stock_value,
                K=strike,
                T=time_to_expiry,
                r=self.interest_rate,
                sigma=self.volatility,
            )

    def calculate_portfolio_delta(
        self,
        options: dict[str, OptionPosition],
        option_positions: dict[str, int],
        futures_positions: dict[str, FuturePosition],
        stock_position: int,
        stock_dual_position: int,
        stock_value: float,
    ) -> float:
        """
        Calculate aggregate delta of entire portfolio.

        Combines delta from all options, futures, and stock positions.
        For options, delta = option_delta * position_quantity.

        Parameters
        ----------
        options : dict[str, OptionPosition]
            Dictionary of option_id -> OptionPosition details.
        option_positions : dict[str, int]
            Dictionary of option_id -> position quantity.
        futures_positions : dict[str, FuturePosition]
            Dictionary of future_id -> FuturePosition details.
        stock_position : int
            Position in main underlying stock.
        stock_dual_position : int
            Position in dual-listed stock.
        stock_value : float
            Current underlying stock price.

        Returns
        -------
        float
            Aggregate portfolio delta.
        """
        aggregate_delta = 0.0

        # Calculate delta contribution from each option
        for option_id, option in options.items():
            position = option_positions.get(option_id, 0)
            delta = self.calculate_option_delta(
                strike=option.strike,
                expiry=option.expiry,
                option_kind=option.option_kind,
                stock_value=stock_value,
            )
            aggregate_delta += delta * position

        # Add futures delta (1:1 exposure, futures delta = position)
        for future_id, future in futures_positions.items():
            aggregate_delta += future.position

        # Add stock positions (1:1 exposure)
        aggregate_delta += stock_position
        aggregate_delta += stock_dual_position

        return aggregate_delta

    def calculate_hedge(
        self,
        portfolio_delta: float,
        stock_position: int,
        best_bid_price: Optional[float] = None,
        best_ask_price: Optional[float] = None,
        custom_threshold: Optional[float] = None,
    ) -> HedgeResult:
        """
        Calculate hedge trade required to neutralize portfolio delta.

        Determines whether a hedge is needed based on threshold and calculates
        the optimal hedge volume and side.

        Parameters
        ----------
        portfolio_delta : float
            Current aggregate delta of the portfolio.
        stock_position : int
            Current position in the underlying stock.
        best_bid_price : float, optional
            Best bid price for the underlying stock.
        best_ask_price : float, optional
            Best ask price for the underlying stock.
        custom_threshold : float, optional
            Override the default hedge threshold.

        Returns
        -------
        HedgeResult
            Delta, hedge volume, and side for the hedge trade.
        """
        threshold = custom_threshold if custom_threshold is not None else self.hedge_threshold

        # Calculate max volumes respecting position limits
        max_volume_to_buy = self.max_hedge_volume - stock_position
        max_volume_to_sell = self.max_hedge_volume + stock_position

        hedge_volume = min(
            abs(round(portfolio_delta)),
            max_volume_to_buy,
            max_volume_to_sell
        )

        hedge_side = None
        hedge_price = None

        if portfolio_delta > threshold and hedge_volume > 0:
            # Need to sell stock to neutralize positive delta
            hedge_side = 'ask'
            if best_bid_price is not None:
                hedge_price = best_bid_price
        elif portfolio_delta < -threshold and hedge_volume > 0:
            # Need to buy stock to neutralize negative delta
            hedge_side = 'bid'
            if best_ask_price is not None:
                hedge_price = best_ask_price

        return HedgeResult(
            aggregate_delta=portfolio_delta,
            hedge_volume=hedge_volume,
            hedge_side=hedge_side,
            hedge_price=hedge_price,
        )

    def analyze_portfolio(
        self,
        options: dict[str, OptionPosition],
        option_positions: dict[str, int],
        futures_positions: dict[str, FuturePosition],
        stock_position: int,
        stock_dual_position: int,
        stock_value: float,
        best_bid_price: Optional[float] = None,
        best_ask_price: Optional[float] = None,
    ) -> HedgedPortfolio:
        """
        Perform complete delta analysis of a portfolio.

        Calculates delta contributions from all instruments and determines
        if a hedge is recommended.

        Parameters
        ----------
        options : dict[str, OptionPosition]
            Dictionary of option_id -> OptionPosition details.
        option_positions : dict[str, int]
            Dictionary of option_id -> position quantity.
        futures_positions : dict[str, FuturePosition]
            Dictionary of future_id -> FuturePosition details.
        stock_position : int
            Position in main underlying stock.
        stock_dual_position : int
            Position in dual-listed stock.
        stock_value : float
            Current underlying stock price.
        best_bid_price : float, optional
            Best bid price for the underlying stock.
        best_ask_price : float, optional
            Best ask price for the underlying stock.

        Returns
        -------
        HedgedPortfolio
            Complete delta analysis including hedge recommendation.
        """
        # Calculate delta contribution from each option
        option_deltas = {}
        for option_id, option in options.items():
            position = option_positions.get(option_id, 0)
            delta = self.calculate_option_delta(
                strike=option.strike,
                expiry=option.expiry,
                option_kind=option.option_kind,
                stock_value=stock_value,
            )
            option_deltas[option_id] = delta * position

        # Calculate futures and stock delta
        future_delta = sum(f.position for f in futures_positions.values())
        stock_delta = stock_position + stock_dual_position

        # Calculate total
        total_delta = sum(option_deltas.values()) + future_delta + stock_delta

        # Calculate hedge
        hedge_result = self.calculate_hedge(
            portfolio_delta=total_delta,
            stock_position=stock_position,
            best_bid_price=best_bid_price,
            best_ask_price=best_ask_price,
        )

        needs_hedge = hedge_result.hedge_side is not None

        return HedgedPortfolio(
            total_delta=total_delta,
            option_deltas=option_deltas,
            future_delta=future_delta,
            stock_delta=stock_delta,
            needs_hedge=needs_hedge,
            hedge_result=hedge_result,
        )


class PerUnderlyingDeltaHedge:
    """
    Delta hedge manager for multiple underlyings.

    Manages delta hedging separately for each underlying instrument,
    handling multiple stocks, their options, futures, and dual listings.

    Example
    -------
    >>> manager = PerUnderlyingDeltaHedge()
    >>> manager.add_underlying('NVDA', volatility=3.0, interest_rate=0.03)
    >>> manager.add_underlying('SAN', volatility=2.5, interest_rate=0.03)
    >>>
    >>> # Analyze NVDA
    >>> nvda_result = manager.analyze_underlying(
    ...     underlying='NVDA',
    ...     options=nvda_options,
    ...     option_positions=nvda_option_positions,
    ...     futures_positions=nvda_futures,
    ...     stock_position=nvda_stock_pos,
    ...     stock_dual_position=nvda_dual_pos,
    ...     stock_value=150.0,
    ...     order_book=nvda_order_book,
    ... )
    """

    def __init__(self):
        """Initialize the per-underlying delta hedge manager."""
        self.engines: dict[str, DeltaHedgeEngine] = {}

    def add_underlying(
        self,
        underlying: str,
        volatility: float = 3.0,
        interest_rate: float = 0.03,
        hedge_threshold: float = DeltaHedgeEngine.DEFAULT_HEDGE_THRESHOLD,
        max_hedge_volume: int = DeltaHedgeEngine.DEFAULT_MAX_HEDGE_VOLUME,
    ) -> None:
        """
        Register an underlying for delta hedging.

        Parameters
        ----------
        underlying : str
            Underlying symbol (e.g., 'NVDA', 'SAN').
        volatility : float
            Implied volatility for this underlying.
        interest_rate : float
            Risk-free interest rate.
        hedge_threshold : float
            Delta threshold for this underlying.
        max_hedge_volume : int
            Maximum hedge volume for this underlying.
        """
        self.engines[underlying] = DeltaHedgeEngine(
            volatility=volatility,
            interest_rate=interest_rate,
            hedge_threshold=hedge_threshold,
            max_hedge_volume=max_hedge_volume,
        )

    def get_engine(self, underlying: str) -> Optional[DeltaHedgeEngine]:
        """
        Get the delta hedge engine for an underlying.

        Parameters
        ----------
        underlying : str
            Underlying symbol.

        Returns
        -------
        DeltaHedgeEngine or None
            Engine for the underlying, or None if not registered.
        """
        return self.engines.get(underlying)

    def analyze_underlying(
        self,
        underlying: str,
        options: dict[str, OptionPosition],
        option_positions: dict[str, int],
        futures_positions: dict[str, FuturePosition],
        stock_position: int,
        stock_dual_position: int,
        stock_value: float,
        order_book_bid: Optional[float] = None,
        order_book_ask: Optional[float] = None,
    ) -> Optional[HedgedPortfolio]:
        """
        Analyze and calculate hedge for a specific underlying.

        Parameters
        ----------
        underlying : str
            Underlying symbol.
        options : dict[str, OptionPosition]
            Dictionary of option_id -> OptionPosition details.
        option_positions : dict[str, int]
            Dictionary of option_id -> position quantity.
        futures_positions : dict[str, FuturePosition]
            Dictionary of future_id -> FuturePosition details.
        stock_position : int
            Position in main underlying stock.
        stock_dual_position : int
            Position in dual-listed stock.
        stock_value : float
            Current underlying stock price.
        order_book_bid : float, optional
            Best bid price for the underlying.
        order_book_ask : float, optional
            Best ask price for the underlying.

        Returns
        -------
        HedgedPortfolio or None
            Complete delta analysis, or None if underlying not registered.
        """
        engine = self.get_engine(underlying)
        if engine is None:
            return None

        return engine.analyze_portfolio(
            options=options,
            option_positions=option_positions,
            futures_positions=futures_positions,
            stock_position=stock_position,
            stock_dual_position=stock_dual_position,
            stock_value=stock_value,
            best_bid_price=order_book_bid,
            best_ask_price=order_book_ask,
        )


def calculate_time_to_expiry(expiry_date: dt.datetime, current_time: dt.datetime) -> float:
    """
    Calculate time to expiry in years.

    Parameters
    ----------
    expiry_date : dt.datetime
        Expiration datetime.
    current_time : dt.datetime
        Current datetime.

    Returns
    -------
    float
        Time to expiry in years.
    """
    return (expiry_date - current_time) / dt.timedelta(days=1) / 365


def calculate_current_time_to_expiry(expiry_date: dt.datetime) -> float:
    """
    Calculate time to expiry from current time.

    Parameters
    ----------
    expiry_date : dt.datetime
        Expiration datetime.

    Returns
    -------
    float
        Time to expiry in years.
    """
    return calculate_time_to_expiry(expiry_date, dt.datetime.now())
