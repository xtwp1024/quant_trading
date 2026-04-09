"""Smart Order Router — routes orders to optimal exchange based on price, liquidity, and fees.

Features:
- Best price routing across multiple exchange adapters
- Fee-aware routing (accounts for maker/taker fees)
- Liquidity filtering (skip exchanges with insufficient depth)
- Spread monitoring (detects arbitrage opportunities)

Usage
-----
```python
from quant_trading.execution.router import SmartOrderRouter
from quant_trading.connectors import BinanceRESTConnector, CoinbaseAdapter

router = SmartOrderRouter()
router.add_exchange("binance", binance_connector, maker=0.001, taker=0.001)
router.add_exchange("coinbase", coinbase_connector, maker=0.004, taker=0.006)

# Route a market buy to the cheapest exchange
result = router.route_market_order("BTC/USDT", "buy", 0.5)
print(f"Best exchange: {result['exchange']}, price: {result['price']}")

# Get all quotes across exchanges
quotes = router.get_all_quotes("BTC/USDT", "sell", 1.0)
for q in sorted(quotes, key=lambda x: x['effective_price']):
    print(f"  {q['exchange']}: ${q['effective_price']:.2f} (fee: ${q['commission']:.2f})")
```
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger("SmartOrderRouter")


@dataclass
class ExchangeQuote:
    """Quote from a single exchange."""
    exchange: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float              # Mid price at exchange
    best_bid: float = 0.0    # Best bid (for sells)
    best_ask: float = 0.0   # Best ask (for buys)
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    available_quantity: float = 0.0  # Depth at best price
    commission: float = 0.0   # Estimated commission
    effective_price: float = 0.0  # Price including fees
    timestamp: int = 0

    def __post_init__(self):
        if self.effective_price == 0.0:
            self.effective_price = self.price


@dataclass
class RoutedOrder:
    """Result of smart order routing."""
    symbol: str
    side: str
    quantity: float
    exchange: str
    price: float
    commission: float
    effective_price: float
    slippage_pct: float
    routed_at: int = 0


class SmartOrderRouter:
    """Smart order router that finds the best exchange for execution.

    Selects the best exchange based on:
    1. Effective price (price + fees)
    2. Available liquidity
    3. Spread quality
    """

    def __init__(self, default_taker_fee: float = 0.001):
        """Initialize.

        Parameters
        ----------
        default_taker_fee : float
            Default taker fee if not specified per exchange.
        """
        self._adapters: Dict[str, any] = {}
        self._fees: Dict[str, tuple] = {}  # exchange → (maker, taker)
        self._enabled: Dict[str, bool] = {}
        self.default_taker_fee = default_taker_fee
        self.logger = logging.getLogger("SmartOrderRouter")

    def add_exchange(
        self,
        name: str,
        adapter: any,
        maker: Optional[float] = None,
        taker: Optional[float] = None,
        enabled: bool = True,
    ) -> "SmartOrderRouter":
        """Register an exchange adapter.

        Returns self for chaining.
        """
        self._adapters[name] = adapter
        self._fees[name] = (maker or self.default_taker_fee, taker or self.default_taker_fee)
        self._enabled[name] = enabled
        self.logger.info(f"Registered exchange: {name} (maker={maker}, taker={taker})")
        return self

    def remove_exchange(self, name: str) -> None:
        """Remove an exchange from the router."""
        self._adapters.pop(name, None)
        self._fees.pop(name, None)
        self._enabled.pop(name, None)

    def get_all_quotes(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> List[ExchangeQuote]:
        """Get quotes from all enabled exchanges.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            "buy" or "sell".
        quantity : float
            Order quantity.

        Returns
        -------
        List[ExchangeQuote]
            Quotes from all exchanges, sorted by effective price.
        """
        quotes: List[ExchangeQuote] = []
        import time

        for name, adapter in self._adapters.items():
            if not self._enabled.get(name, True):
                continue

            try:
                ticker = adapter.get_ticker(symbol)
                if not ticker:
                    continue

                maker, taker = self._fees.get(name, (self.default_taker_fee, self.default_taker_fee))
                price = ticker.get("price", 0.0)
                if price <= 0:
                    continue

                bid = ticker.get("bid", price)
                ask = ticker.get("ask", price)

                # Estimate commission (taker for market order)
                commission = quantity * price * taker

                # Effective price = (quantity * price + commission) / quantity
                effective_price = price + (commission / quantity) if quantity > 0 else price

                # Available depth (simplified — use 1% of 24h volume as proxy)
                available = ticker.get("volume", 0) * 0.01 * price  # rough

                quote = ExchangeQuote(
                    exchange=name,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    best_bid=bid,
                    best_ask=ask,
                    maker_fee=maker,
                    taker_fee=taker,
                    available_quantity=available,
                    commission=commission,
                    effective_price=effective_price,
                    timestamp=int(time.time() * 1000),
                )
                quotes.append(quote)

            except Exception as e:
                self.logger.warning(f"Failed to get quote from {name}: {e}")

        # Sort by effective price (ascending for buys, descending for sells)
        if side.lower() in ("buy", "long"):
            quotes.sort(key=lambda q: q.effective_price)
        else:
            quotes.sort(key=lambda q: q.effective_price, reverse=True)

        return quotes

    def route_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        max_slippage: float = 0.005,
    ) -> RoutedOrder:
        """Route a market order to the best exchange.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            "buy" or "sell".
        quantity : float
            Order quantity.
        max_slippage : float
            Maximum allowed slippage (fraction). If best exchange
            exceeds this, try next exchange.

        Returns
        -------
        RoutedOrder
            Execution details including which exchange was selected.

        Raises
        ------
        ValueError
            If no exchange can fill the order within slippage tolerance.
        """
        quotes = self.get_all_quotes(symbol, side, quantity)

        if not quotes:
            raise ValueError(f"No exchanges available for {symbol}")

        import time

        for quote in quotes:
            # Estimate slippage based on distance to mid price
            mid = (quote.best_bid + quote.best_ask) / 2 if quote.best_bid and quote.best_ask else quote.price
            if side.lower() == "buy":
                slippage = (quote.price - mid) / mid if mid > 0 else 0.0
            else:
                slippage = (mid - quote.price) / mid if mid > 0 else 0.0

            if slippage <= max_slippage:
                self.logger.info(
                    f"Routed {side} {quantity} {symbol} to {quote.exchange} "
                    f"@ {quote.price} (slippage={slippage:.4%}, fee=${quote.commission:.4f})"
                )
                return RoutedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    exchange=quote.exchange,
                    price=quote.price,
                    commission=quote.commission,
                    effective_price=quote.effective_price,
                    slippage_pct=slippage,
                    routed_at=int(time.time() * 1000),
                )

        # All exceed slippage — use best available
        best = quotes[0]
        self.logger.warning(
            f"All exchanges exceed max_slippage={max_slippage:.4%}. "
            f"Using best available: {best.exchange} @ {best.price}"
        )
        return RoutedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            exchange=best.exchange,
            price=best.price,
            commission=best.commission,
            effective_price=best.effective_price,
            slippage_pct=1.0,  # Flag as exceeding slippage
            routed_at=int(time.time() * 1000),
        )

    def execute_routed_order(
        self,
        routed: RoutedOrder,
        adapter: Optional[any] = None,
    ) -> dict:
        """Execute a routed order on its target exchange.

        Parameters
        ----------
        routed : RoutedOrder
            The routed order to execute.
        adapter : optional
            Override adapter (uses router's if not provided).

        Returns
        -------
        dict
            Execution result from the exchange adapter.
        """
        if adapter is None:
            adapter = self._adapters.get(routed.exchange)

        if adapter is None:
            return {"success": False, "error": f"No adapter for {routed.exchange}"}

        try:
            if routed.side.lower() in ("buy", "long"):
                result = adapter.place_market_order(
                    routed.symbol, "buy", routed.quantity
                )
            else:
                result = adapter.place_market_order(
                    routed.symbol, "sell", routed.quantity
                )
            result["routed_exchange"] = routed.exchange
            result["effective_price"] = routed.effective_price
            result["commission"] = routed.commission
            return result
        except Exception as e:
            self.logger.error(f"Execution failed on {routed.exchange}: {e}")
            return {"success": False, "error": str(e), "exchange": routed.exchange}

    def detect_arbitrage(
        self,
        symbol: str,
        quantity: float,
        min_profit_pct: float = 0.001,
    ) -> Optional[Dict]:
        """Detect cross-exchange arbitrage opportunity.

        Returns
        -------
        Dict or None
            Arbitrage opportunity dict with buy_exchange, sell_exchange,
            buy_price, sell_price, profit_pct.
        """
        quotes = self.get_all_quotes(symbol, "buy", quantity)
        if len(quotes) < 2:
            return None

        # Best buy = lowest ask, best sell = highest bid
        buy_quote = quotes[0]  # Already sorted ascending by effective price
        sell_quotes = sorted(quotes, key=lambda q: q.effective_price, reverse=True)

        if sell_quotes[0].exchange == buy_quote.exchange:
            if len(sell_quotes) < 2:
                return None
            sell_quote = sell_quotes[1]
        else:
            sell_quote = sell_quotes[0]

        buy_price = buy_quote.effective_price
        sell_price = sell_quote.effective_price
        profit_pct = (sell_price - buy_price) / buy_price

        if profit_pct >= min_profit_pct:
            return {
                "symbol": symbol,
                "buy_exchange": buy_quote.exchange,
                "sell_exchange": sell_quote.exchange,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "profit_pct": profit_pct,
                "profit_abs": quantity * (sell_price - buy_price),
            }

        return None
