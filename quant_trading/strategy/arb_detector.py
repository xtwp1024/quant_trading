"""Arbitrage opportunity detection using graph-based Bellman-Ford style algorithm"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in an order book"""
    price: Decimal
    volume: Decimal


@dataclass
class OrderBook:
    """Aggregated order book data"""
    symbol: str
    asks: List[OrderBookLevel]  # Sorted ascending by price
    bids: List[OrderBookLevel]  # Sorted descending by price

    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best (lowest) ask price"""
        return self.asks[0] if self.asks else None

    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best (highest) bid price"""
        return self.bids[0] if self.bids else None


@dataclass
class TradingPair:
    """Trading pair information"""
    symbol: str  # e.g., 'BTC/USDT'
    base: str    # e.g., 'BTC'
    quote: str   # e.g., 'USDT'
    tick_size: Decimal
    lot_size: Decimal
    maker_fee: Decimal
    taker_fee: Decimal


@dataclass
class ArbitrageOpportunity:
    """Triangular arbitrage opportunity details"""
    exchange: str
    first_symbol: str   # e.g., 'BTC/USDT'
    second_symbol: str  # e.g., 'ETH/BTC'
    third_symbol: str   # e.g., 'ETH/USDT'

    first_price: Decimal
    second_price: Decimal
    third_price: Decimal

    first_trade: Decimal
    second_trade: Decimal
    third_trade: Decimal

    gross_profit: Decimal
    gross_profit_pct: Decimal

    # After fees and slippage
    net_profit: Decimal
    net_profit_pct: Decimal

    first_price_impact: Optional[Decimal] = None
    second_price_impact: Optional[Decimal] = None
    third_price_impact: Optional[Decimal] = None

    is_executable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange": self.exchange,
            "first_symbol": self.first_symbol,
            "second_symbol": self.second_symbol,
            "third_symbol": self.third_symbol,
            "first_price": float(self.first_price),
            "second_price": float(self.second_price),
            "third_price": float(self.third_price),
            "first_trade": float(self.first_trade),
            "second_trade": float(self.second_trade),
            "third_trade": float(self.third_trade),
            "gross_profit": float(self.gross_profit),
            "gross_profit_pct": float(self.gross_profit_pct),
            "net_profit": float(self.net_profit),
            "net_profit_pct": float(self.net_profit_pct),
            "first_price_impact": float(self.first_price_impact) if self.first_price_impact else None,
            "second_price_impact": float(self.second_price_impact) if self.second_price_impact else None,
            "third_price_impact": float(self.third_price_impact) if self.third_price_impact else None,
            "is_executable": self.is_executable,
        }


class ArbitrageDetector:
    """
    Graph-based arbitrage detection using Bellman-Ford style negative cycle detection.

    In a currency graph:
    - Nodes = currencies (e.g., BTC, ETH, USDT)
    - Edges = trading pairs with exchange rates

    A profitable triangular arbitrage is a negative cycle in the log-price graph:
    rate(A->B) * rate(B->C) * rate(C->A) > 1
    equivalently: -log(rate(A->B)) - log(rate(B->C)) - log(rate(C->A)) < 0
    """

    def __init__(
        self,
        min_profit_pct: Decimal = Decimal("0.1"),
        min_profit_threshold: Decimal = Decimal("0.3"),
        initial_amount: Decimal = Decimal("100"),
        unavailable_pairs: Optional[set] = None,
    ):
        """
        Initialize arbitrage detector.

        Args:
            min_profit_pct: Minimum profit percentage after fees/slippage to execute
            min_profit_threshold: Gross profit threshold to trigger liquidity check
            initial_amount: Initial capital in quote currency (USDT)
            unavailable_pairs: Set of pairs to skip (e.g., {'YGG/BNB', 'RAD/BNB'})
        """
        self.min_profit_pct = min_profit_pct
        self.min_profit_threshold = min_profit_threshold
        self.initial_amount = initial_amount
        self.unavailable_pairs = unavailable_pairs or set()

    def find_triangular_opportunities(
        self,
        markets: Dict[str, TradingPair],
        tickers: Dict[str, Dict[str, Any]],
        order_books: Optional[Dict[str, OrderBook]] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Find all triangular arbitrage opportunities given current market data.

        Args:
            markets: Dict mapping symbol -> TradingPair info
            tickers: Dict mapping symbol -> {'ask': float, 'bid': float, ...}
            order_books: Optional dict of symbol -> OrderBook for slippage calculation

        Returns:
            List of ArbitrageOpportunity sorted by gross profit descending
        """
        opportunities = []

        # Group symbols by quote currency (USDT pairs)
        usdt_symbols = {s for s in markets.keys() if s.endswith("/USDT")}

        # Build symbols_by_base index
        symbols_by_base: Dict[str, set] = {}
        for symbol in markets.keys():
            base, quote = symbol.split("/")
            if base not in symbols_by_base:
                symbols_by_base[base] = set()
            symbols_by_base[base].add(symbol)

        # Iterate over all possible triangular paths: USDT -> X -> Y -> USDT
        for first_symbol in usdt_symbols:
            base1, quote1 = first_symbol.split("/")  # e.g., ('BTC', 'USDT')
            if base1 == "USDT":
                continue

            second_symbols = symbols_by_base.get(base1, set())

            for second_symbol in second_symbols:
                if second_symbol == first_symbol or second_symbol in self.unavailable_pairs:
                    continue

                base2, quote2 = second_symbol.split("/")

                # Determine third base currency
                if base2 == base1:
                    third_base = quote2  # e.g., second is ETH/BTC -> third_base = BTC
                else:
                    third_base = base2  # e.g., second is BTC/ETH -> third_base = ETH

                third_symbol = f"{third_base}/USDT"

                # Verify all pairs exist
                if third_symbol not in markets:
                    continue

                # Skip if any pair not in tickers
                if any(s not in tickers for s in [first_symbol, second_symbol, third_symbol]):
                    continue

                # Get tick sizes
                first_tick = markets[first_symbol].tick_size
                second_tick = markets[second_symbol].tick_size
                third_tick = markets[third_symbol].tick_size

                # Get fees
                first_fee = markets[first_symbol].taker_fee
                second_fee = markets[second_symbol].taker_fee
                third_fee = markets[third_symbol].taker_fee

                # Extract prices
                ticker1 = tickers.get(first_symbol, {})
                ticker2 = tickers.get(second_symbol, {})
                ticker3 = tickers.get(third_symbol, {})

                # For triangular arb: buy first (ask), sell second (bid), sell third (bid)
                first_ask = ticker1.get("ask")
                second_bid = ticker2.get("bid")
                third_bid = ticker3.get("bid")

                if not first_ask or not second_bid or not third_bid:
                    continue

                first_price = Decimal(str(first_ask)).quantize(
                    Decimal(str(first_tick)), rounding=ROUND_DOWN
                )
                second_price = Decimal(str(second_bid)).quantize(
                    Decimal(str(second_tick)), rounding=ROUND_DOWN
                )
                third_price = Decimal(str(third_bid)).quantize(
                    Decimal(str(third_tick)), rounding=ROUND_DOWN
                )

                # Check for zeros
                if first_price <= 0 or second_price <= 0 or third_price <= 0:
                    continue

                # Calculate triangular trade:
                # Step 1: Buy first pair with USDT (e.g., buy BTC/USDT)
                first_trade = self.initial_amount / first_price
                first_trade = first_trade.quantize(Decimal(str(first_tick)), rounding=ROUND_DOWN)

                # Step 2: Sell first trade amount in second pair (e.g., sell ETH/BTC)
                second_trade = first_trade * second_price
                second_trade = second_trade.quantize(Decimal(str(second_tick)), rounding=ROUND_DOWN)

                # Step 3: Sell second trade amount in third pair (e.g., sell ETH/USDT)
                third_trade = second_trade * third_price
                third_trade = third_trade.quantize(Decimal(str(third_tick)), rounding=ROUND_DOWN)

                if first_trade <= 0 or second_trade <= 0 or third_trade <= 0:
                    continue

                # Calculate gross profit
                gross_profit = third_trade - self.initial_amount
                gross_profit_pct = (gross_profit / self.initial_amount) * 100

                opp = ArbitrageOpportunity(
                    exchange="unknown",
                    first_symbol=first_symbol,
                    second_symbol=second_symbol,
                    third_symbol=third_symbol,
                    first_price=first_price,
                    second_price=second_price,
                    third_price=third_price,
                    first_trade=first_trade,
                    second_trade=second_trade,
                    third_trade=third_trade,
                    gross_profit=gross_profit,
                    gross_profit_pct=gross_profit_pct,
                    net_profit=gross_profit,  # Will be updated with fees
                    net_profit_pct=gross_profit_pct,
                )

                # Only add if above threshold to check liquidity
                if gross_profit_pct > self.min_profit_threshold:
                    opportunities.append(opp)

        # Sort by gross profit
        opportunities.sort(key=lambda x: -x.gross_profit_pct)
        return opportunities

    def calculate_price_impact(
        self,
        order_book: OrderBook,
        order_size: Decimal,
        side: str,
    ) -> Optional[Decimal]:
        """
        Calculate volume-weighted average price impact for an order.

        Args:
            order_book: OrderBook with asks (for buy) or bids (for sell)
            order_size: Amount to trade
            side: 'buy' or 'sell'

        Returns:
            Volume-weighted average price, or None if insufficient liquidity
        """
        levels = order_book.asks if side == "buy" else order_book.bids

        remaining = float(order_size)
        total_value = Decimal("0")
        total_volume = Decimal("0")

        for level in levels:
            if remaining <= 0:
                break
            volume_for_level = min(Decimal(str(level.volume)), Decimal(str(remaining)))
            total_value += volume_for_level * level.price
            total_volume += volume_for_level
            remaining -= float(volume_for_level)

        if total_volume <= 0:
            return None

        price_impact = total_value / total_volume
        return price_impact.quantize(order_book.asks[0].price if order_book.asks else order_book.bids[0].price)

    def apply_fees_and_slippage(
        self,
        opportunity: ArbitrageOpportunity,
        order_books: Dict[str, OrderBook],
        fees: Tuple[Decimal, Decimal, Decimal],
    ) -> ArbitrageOpportunity:
        """
        Recalculate profit after fees and order book slippage.

        Args:
            opportunity: The arbitrage opportunity to evaluate
            order_books: Dict of symbol -> OrderBook for slippage calculation
            fees: Tuple of (first_fee, second_fee, third_fee) as decimals

        Returns:
            Updated ArbitrageOpportunity with net profit after fees/slippage
        """
        first_fee, second_fee, third_fee = fees

        # Calculate price impacts from order books
        first_ob = order_books.get(opportunity.first_symbol)
        second_ob = order_books.get(opportunity.second_symbol)
        third_ob = order_books.get(opportunity.third_symbol)

        # Get tick sizes
        first_tick = Decimal("0.01")  # Would normally come from market data
        second_tick = Decimal("0.01")
        third_tick = Decimal("0.01")

        # Calculate price impact for first order (buy)
        if first_ob:
            first_impact = self.calculate_price_impact(first_ob, self.initial_amount, "buy")
            if first_impact:
                opportunity.first_price_impact = first_impact.quantize(first_tick, rounding=ROUND_UP)

        # Calculate price impact for second order (sell)
        if second_ob:
            second_impact = self.calculate_price_impact(second_ob, opportunity.first_trade, "sell")
            if second_impact:
                opportunity.second_price_impact = second_impact.quantize(second_tick, rounding=ROUND_DOWN)

        # Calculate price impact for third order (sell)
        if third_ob:
            third_impact = self.calculate_price_impact(third_ob, opportunity.second_trade, "sell")
            if third_impact:
                opportunity.third_price_impact = third_impact.quantize(third_tick, rounding=ROUND_DOWN)

        # Use impact prices if available, otherwise use original
        p1 = opportunity.first_price_impact or opportunity.first_price
        p2 = opportunity.second_price_impact or opportunity.second_price
        p3 = opportunity.third_price_impact or opportunity.third_price

        # Recalculate trades with impacted prices and fees
        first_trade_before = self.initial_amount / p1
        first_trade_after = first_trade_before * (Decimal("1") - first_fee)
        first_trade_amount = first_trade_after.quantize(first_tick, rounding=ROUND_DOWN)

        second_trade_before = first_trade_amount * p2
        second_trade_after = second_trade_before * (Decimal("1") - second_fee)
        second_trade_amount = second_trade_after.quantize(second_tick, rounding=ROUND_DOWN)

        third_trade_before = second_trade_amount * p3
        third_trade_after = third_trade_before * (Decimal("1") - third_fee)
        third_trade_amount = third_trade_after.quantize(third_tick, rounding=ROUND_DOWN)

        # Calculate net profit
        net_profit = third_trade_amount - self.initial_amount
        net_profit_pct = (net_profit / self.initial_amount) * 100

        opportunity.first_trade = first_trade_amount
        opportunity.second_trade = second_trade_amount
        opportunity.third_trade = third_trade_amount
        opportunity.net_profit = net_profit
        opportunity.net_profit_pct = net_profit_pct
        opportunity.is_executable = net_profit_pct > self.min_profit_pct

        return opportunity

    def detect_negative_cycles_bellman_ford(
        self,
        currencies: List[str],
        exchange_rates: Dict[Tuple[str, str], Decimal],
    ) -> List[List[str]]:
        """
        Bellman-Ford based arbitrage detection.

        Finds negative cycles in the log-exchange-rate graph where:
        -log(rate(A,B)) + -log(rate(B,C)) + -log(rate(C,A)) < 0

        This indicates a profitable arbitrage opportunity.

        Args:
            currencies: List of currency codes
            exchange_rates: Dict of (from_currency, to_currency) -> rate

        Returns:
            List of cycles (each cycle is a list of currency codes)
        """
        import math

        # Build log graph (negative log of exchange rate)
        # If rate(A->B) = 1.01, then -log(1.01) ≈ -0.01
        log_graph: Dict[str, Dict[str, float]] = {c: {} for c in currencies}

        for (cur1, cur2), rate in exchange_rates.items():
            if rate > 0:
                log_graph[cur1][cur2] = -math.log(float(rate))
            else:
                # Skip invalid zero or negative rates
                logger.warning(f"Skipping invalid rate {rate} for {cur1}->{cur2}")
                continue

        # Bellman-Ford: try to find negative cycles
        # Distance from source to itself should be negative if arbitrage exists
        cycles = []
        n = len(currencies)

        if n == 0:
            return cycles

        # Predecessor matrix for cycle reconstruction
        pred: Dict[str, Dict[str, Optional[str]]] = {c: {c2: None for c2 in currencies} for c in currencies}

        # Initialize distances
        dist: Dict[str, Dict[str, float]] = {c: {c2: 0.0 if c == c2 else float("inf") for c2 in currencies} for c in currencies}

        # Relax edges n-1 times
        for _ in range(n - 1):
            for u in currencies:
                for v in currencies:
                    if u == v:
                        continue
                    w = log_graph.get(u, {}).get(v)
                    if w is None:
                        continue
                    for c2 in currencies:
                        if dist[u][c2] + w < dist[v][c2]:
                            dist[v][c2] = dist[u][c2] + w
                            pred[v][c2] = u

        # Check for negative cycles (distance from source to itself < 0)
        for c in currencies:
            if dist[c][c] < -1e-10:
                # Reconstruct cycle with cycle detection to prevent infinite loops
                cycle = [c]
                current = c
                visited = set([c])
                for _ in range(n):
                    current = pred[c][current]
                    if current is None or current in visited:
                        break
                    cycle.append(current)
                    visited.add(current)
                if len(cycle) >= 3 and cycle[-1] == cycle[0]:
                    cycle = cycle[:-1]
                if len(cycle) >= 3:
                    cycles.append(cycle)

        return cycles


def create_order_book_from_array(symbol: str, asks_array: np.ndarray, bids_array: np.ndarray) -> OrderBook:
    """Helper to create OrderBook from numpy arrays."""
    asks = [OrderBookLevel(Decimal(str(p)), Decimal(str(v))) for p, v in asks_array]
    bids = [OrderBookLevel(Decimal(str(p)), Decimal(str(v))) for p, v in bids_array]
    return OrderBook(symbol=symbol, asks=asks, bids=bids)
