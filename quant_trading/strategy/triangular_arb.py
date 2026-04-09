"""Triangular Arbitrage Strategy for backtesting"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.arb_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    OrderBook,
    OrderBookLevel,
    TradingPair,
)

if TYPE_CHECKING:
    from quant_trading.strategy.context import StrategyContext


@dataclass
class TriangularArbParams(StrategyParams):
    """Triangular Arbitrage Strategy parameters"""

    # Trading parameters
    initial_amount: float = 100.0  # Initial capital in quote currency (USDT)
    min_profit_pct: float = 0.1  # Minimum profit % after fees/slippage to execute
    min_gross_profit_pct: float = 0.3  # Gross profit % threshold to check liquidity

    # Exchange fees (maker/taker)
    binance_fee: float = 0.001  # 0.1%
    kucoin_fee: float = 0.001
    okx_fee: float = 0.001
    huobi_fee: float = 0.002  # 0.2%

    # Supported exchanges
    exchanges: List[str] = field(default_factory=lambda: ["binance", "kucoin", "okx", "huobi"])

    # Symbol filters
    unavailable_pairs: List[str] = field(
        default_factory=lambda: ["YGG/BNB", "RAD/BNB", "VOXEL/BNB", "GLMR/BNB", "UNI/EUR"]
    )

    # Order book depth for slippage calculation
    max_order_book_depth: int = 20  # Number of price levels to consider

    # Execution settings
    check_liquidity: bool = True  # Whether to check order book depth before signal
    simulated_liquidity: float = 10000.0  # Simulated liquidity for backtesting


class TriangularArbitrageStrategy(BaseStrategy):
    """
    Triangular Arbitrage Strategy.

    Detects and signals triangular arbitrage opportunities across multiple exchanges
    (Binance, KuCoin, OKX, Huobi) using CCXT-style unified interface.

    The triangular arbitrage algorithm:
    1. Find triangular paths: USDT -> X -> Y -> USDT
       e.g., BTC -> ETH -> USDT -> BTC
    2. Calculate gross profit based on current prices
    3. If gross profit > threshold, check order book depth
    4. Apply fees and slippage
    5. If net profit > minimum, generate entry signal

    For backtesting, the strategy uses simulated/stubbed exchange data.
    """

    name: str = "triangular_arb"
    params: TriangularArbParams

    def __init__(self, symbol: str = "arb", params: Optional[TriangularArbParams] = None) -> None:
        """
        Initialize triangular arbitrage strategy.

        Args:
            symbol: Strategy identifier (not a single trading pair for this strategy)
            params: Strategy parameters
        """
        super().__init__(symbol=symbol, params=params)
        self.detector = ArbitrageDetector(
            min_profit_pct=Decimal(str(self.params.min_profit_pct)),
            min_profit_threshold=Decimal(str(self.params.min_gross_profit_pct)),
            initial_amount=Decimal(str(self.params.initial_amount)),
            unavailable_pairs=set(self.params.unavailable_pairs),
        )

        # Market data cache (for backtesting)
        self._markets: Dict[str, TradingPair] = {}
        self._tickers: Dict[str, Dict[str, Any]] = {}
        self._order_books: Dict[str, OrderBook] = {}

        # Detected opportunities
        self._current_opportunities: List[ArbitrageOpportunity] = []
        self._last_opportunity: Optional[ArbitrageOpportunity] = None

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on arbitrage opportunities.

        For triangular arb, data should contain columns:
        - timestamp
        - first_symbol, second_symbol, third_symbol (e.g., BTC/USDT, ETH/BTC, ETH/USDT)
        - first_ask, second_bid, third_bid (prices)
        - first_tick_size, second_tick_size, third_tick_size
        - exchange

        Or alternatively, tick data with columns:
        - timestamp, symbol, ask, bid, volume, tick_size
        """
        signals = []

        # Detect opportunities from market data
        opportunities = self._detect_from_dataframe(data)

        for opp in opportunities:
            self._last_opportunity = opp
            if opp.is_executable:
                # Generate entry signal with full opportunity details in metadata
                signal = Signal(
                    symbol=f"{opp.first_symbol}|{opp.second_symbol}|{opp.third_symbol}",
                    direction=SignalDirection.LONG,  # Arbitrage is always "long" in profit
                    strength=float(min(opp.net_profit_pct / 1.0, 1.0)),  # Normalize
                    price=float(opp.first_price),
                    metadata={
                        "strategy": "triangular_arb",
                        "exchange": opp.exchange,
                        "opportunity": opp.to_dict(),
                        "first_symbol": opp.first_symbol,
                        "second_symbol": opp.second_symbol,
                        "third_symbol": opp.third_symbol,
                        "gross_profit_pct": float(opp.gross_profit_pct),
                        "net_profit_pct": float(opp.net_profit_pct),
                        "initial_amount": float(self.params.initial_amount),
                    },
                )
                signals.append(signal)

        return signals

    def _detect_from_dataframe(self, data: pd.DataFrame) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage opportunities from market data DataFrame.

        DataFrame should have columns for price data of three pairs:
        - first_symbol_ask, first_symbol_bid
        - second_symbol_ask, second_symbol_bid
        - third_symbol_ask, third_symbol_bid
        - tick sizes for each
        """
        opportunities = []

        # Group by timestamp to process each time point
        if "timestamp" not in data.columns:
            data = data.copy()
            data["timestamp"] = range(len(data))

        grouped = data.groupby("timestamp")

        for ts, group in grouped:
            # Build tickers dict for this timestamp
            tickers: Dict[str, Dict[str, Any]] = {}

            for _, row in group.iterrows():
                # Try common column patterns
                for col in group.columns:
                    if col.endswith("_ask") or col.endswith("_bid"):
                        symbol_col = col.rsplit("_", 1)[0]
                        if symbol_col not in tickers:
                            tickers[symbol_col] = {}

                        if col.endswith("_ask"):
                            tickers[symbol_col]["ask"] = row[col]
                        elif col.endswith("_bid"):
                            tickers[symbol_col]["bid"] = row[col]

            # Build minimal markets dict
            markets: Dict[str, TradingPair] = {}
            for symbol in tickers.keys():
                parts = symbol.split("/")
                if len(parts) == 2:
                    markets[symbol] = TradingPair(
                        symbol=symbol,
                        base=parts[0],
                        quote=parts[1],
                        tick_size=Decimal("0.01"),
                        lot_size=Decimal("0.001"),
                        maker_fee=Decimal("0.001"),
                        taker_fee=Decimal("0.001"),
                    )

            if not markets:
                continue

            # Detect opportunities
            opps = self.detector.find_triangular_opportunities(markets, tickers)

            for opp in opps:
                opp.exchange = str(row.get("exchange", "binance"))

                # Apply simulated fees and slippage for backtesting
                fee = Decimal(str(self.params.huobi_fee if opp.exchange == "huobi" else self.params.binance_fee))
                opp = self.detector.apply_fees_and_slippage(
                    opp,
                    {},  # No real order books in backtesting
                    (fee, fee, fee),
                )
                opportunities.append(opp)

        return opportunities

    def calculate_position_size(self, signal: Signal, context: "StrategyContext") -> float:
        """
        Calculate position size for the arbitrage trade.

        For triangular arbitrage, position size is typically the initial amount
        invested across all three legs of the trade.
        """
        initial_amount = self.params.initial_amount

        # Scale position based on signal strength and available cash
        max_position = min(initial_amount * float(signal.strength), context.available_cash)

        return max_position

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """Process single bar/K-line data."""
        # Convert bar to DataFrame format expected by generate_signals
        df = pd.DataFrame([bar])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process single tick data.

        Expected tick format for triangular arb:
        {
            'symbol': 'BTC/USDT',
            'ask': 50000.0,
            'bid': 49999.0,
            'volume': 1.5,
            'timestamp': 1234567890
        }
        """
        symbol = tick.get("symbol")
        if not symbol:
            return None

        # Update tickers cache
        self._tickers[symbol] = {
            "ask": tick.get("ask"),
            "bid": tick.get("bid"),
            "volume": tick.get("volume", 0),
        }

        # Build market info if not exists
        if symbol not in self._markets:
            parts = symbol.split("/")
            if len(parts) == 2:
                self._markets[symbol] = TradingPair(
                    symbol=symbol,
                    base=parts[0],
                    quote=parts[1],
                    tick_size=Decimal(str(tick.get("tick_size", "0.01"))),
                    lot_size=Decimal(str(tick.get("lot_size", "0.001"))),
                    maker_fee=Decimal(str(self.params.binance_fee)),
                    taker_fee=Decimal(str(self.params.binance_fee)),
                )

        # Detect opportunities
        opps = self.detector.find_triangular_opportunities(self._markets, self._tickers)

        if not opps:
            return None

        # Get best opportunity
        opp = opps[0]
        opp.exchange = tick.get("exchange", "binance")

        fee = Decimal(str(self.params.huobi_fee if opp.exchange == "huobi" else self.params.binance_fee))
        opp = self.detector.apply_fees_and_slippage(opp, {}, (fee, fee, fee))

        self._last_opportunity = opp

        if not opp.is_executable:
            return None

        return Signal(
            symbol=f"{opp.first_symbol}|{opp.second_symbol}|{opp.third_symbol}",
            direction=SignalDirection.LONG,
            strength=float(min(opp.net_profit_pct / 1.0, 1.0)),
            price=float(opp.first_price),
            metadata={
                "strategy": "triangular_arb",
                "exchange": opp.exchange,
                "opportunity": opp.to_dict(),
                "net_profit_pct": float(opp.net_profit_pct),
            },
        )

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order fill callback."""
        logger = logging.getLogger(__name__)
        logger.info(f"Order filled: {order}")

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """Handle position change callback."""
        pass

    def update_data(self, new_data: pd.DataFrame) -> None:
        """Update strategy with new market data."""
        super().update_data(new_data)

    def get_required_history(self) -> int:
        """Return required historical data length."""
        return 10  # Triangular arb needs minimal history

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "initial_amount": self.params.initial_amount,
                "min_profit_pct": self.params.min_profit_pct,
                "min_gross_profit_pct": self.params.min_gross_profit_pct,
                "exchanges": self.params.exchanges,
                "unavailable_pairs": self.params.unavailable_pairs,
            },
            "last_opportunity": self._last_opportunity.to_dict() if self._last_opportunity else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriangularArbitrageStrategy":
        """Deserialize strategy from dict."""
        params = TriangularArbParams(**data.get("params", {}))
        strategy = cls(symbol=data.get("symbol", "arb"), params=params)
        return strategy

    def set_exchange_fee(self, exchange: str, fee: float) -> None:
        """Update fee for a specific exchange."""
        fee_dec = Decimal(str(fee))
        for market in self._markets.values():
            if exchange.lower() == "binance":
                self.params.binance_fee = fee
            elif exchange.lower() == "kucoin":
                self.params.kucoin_fee = fee
            elif exchange.lower() == "okx":
                self.params.okx_fee = fee
            elif exchange.lower() == "huobi":
                self.params.huobi_fee = fee

    def get_exchange_fee(self, exchange: str) -> float:
        """Get fee for a specific exchange."""
        exchange = exchange.lower()
        if exchange == "binance":
            return self.params.binance_fee
        elif exchange == "kucoin":
            return self.params.kucoin_fee
        elif exchange == "okx":
            return self.params.okx_fee
        elif exchange == "huobi":
            return self.params.huobi_fee
        return self.params.binance_fee  # Default

    def get_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current detected opportunities."""
        return self._current_opportunities

    def get_last_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Get the most recent arbitrage opportunity."""
        return self._last_opportunity


# ============================================================================
# CCXT Integration stubs for backtesting
# ============================================================================

class CCXTStub:
    """
    Stub implementation of CCXT exchange interface for backtesting.

    In production, this would be replaced with actual CCXT calls:
        import ccxt.async_support as ccxt
        exchange = ccxt.binance({'apiKey': ...})

    For backtesting, use this stub with set_market_data() to inject
    simulated ticker and order book data.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.id = exchange_id
        self._markets: Dict[str, TradingPair] = {}
        self._tickers: Dict[str, Dict[str, Any]] = {}
        self._order_books: Dict[str, Dict[str, Any]] = {}

    async def load_markets(self, reload: bool = False) -> Dict[str, TradingPair]:
        """Load market metadata (stub - returns cached)."""
        return self._markets

    async def fetch_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all tickers (stub)."""
        return self._tickers

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch single ticker (stub)."""
        return self._tickers.get(symbol, {})

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch order book (stub)."""
        return self._order_books.get(symbol, {"asks": [], "bids": []})

    def set_market_data(
        self,
        symbol: str,
        ask: float,
        bid: float,
        tick_size: float = 0.01,
        lot_size: float = 0.001,
    ) -> None:
        """Set simulated market data for backtesting."""
        base, quote = symbol.split("/")
        self._markets[symbol] = TradingPair(
            symbol=symbol,
            base=base,
            quote=quote,
            tick_size=Decimal(str(tick_size)),
            lot_size=Decimal(str(lot_size)),
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
        )
        self._tickers[symbol] = {"ask": ask, "bid": bid}

    def set_order_book(self, symbol: str, asks: List, bids: List) -> None:
        """Set simulated order book for backtesting."""
        self._order_books[symbol] = {"asks": asks, "bids": bids}

    @property
    def markets(self) -> Dict[str, TradingPair]:
        """Access markets dict."""
        return self._markets


def create_multi_exchange_stub(
    exchanges: List[str] = None,
) -> Dict[str, CCXTStub]:
    """
    Create stub CCXT exchanges for backtesting.

    Args:
        exchanges: List of exchange IDs (default: binance, kucoin, okx, huobi)

    Returns:
        Dict of exchange_id -> CCXTStub
    """
    if exchanges is None:
        exchanges = ["binance", "kucoin", "okx", "huobi"]

    return {ex_id: CCXTStub(ex_id) for ex_id in exchanges}
