# Hummingbot Exchange Connector & Strategy Patterns

Analysis of Hummingbot v2.x market making framework for reusable patterns in 量化之神.

**Source:** `D:/Hive/Data/trading_repos/hummingbot/`

---

## 1. Connector Interface/Abstract Base Class Pattern

### Architecture Overview

```
hummingbot/connector/
    exchange_py_base.py    # Pure Python base (ExchangePyBase)
    exchange_base.pyx      # Cython base (ExchangeBase) - performance critical
    connector_base.pyx     # Base connector with common functionality
    perpetual_derivative_py_base.py  # Perpetual/futures base
    derivative_base.py     # Derivative abstract base
    exchange/              # Concrete exchange implementations
        binance/binance_exchange.py
        okx/okx_exchange.py
        ...
```

### Key Base Classes

#### ExchangePyBase (Pure Python)
Location: `hummingbot/connector/exchange_py_base.py`

Core abstract methods that each exchange must implement:

```python
class ExchangePyBase:
    # Properties
    @property
    def name(self) -> str:  # Exchange identifier
    @property
    def rate_limits_rules(self):  # Rate limit definitions
    @property
    def trading_rules_request_path(self):  # API endpoint for trading rules
    @property
    def trading_pairs_request_path(self):  # API endpoint for trading pairs

    # Abstract methods
    def _create_web_assistants_factory(self) -> WebAssistantsFactory
    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource
    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource
    def _get_fee(self, base, quote, order_type, order_side, amount, price, is_maker) -> TradeFeeBase
    async def _place_order(self, order_id, trading_pair, amount, trade_type, order_type, price, **kwargs)
    async def _place_cancel(self, order_id, tracked_order)
    async def _format_trading_rules(self, exchange_info_dict) -> List[TradingRule]
    async def _all_trade_updates_for_order(self, order) -> List[TradeUpdate]
    async def _request_order_status(self, tracked_order) -> OrderUpdate
    async def _update_balances(self)
    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info)
```

#### Order Tracking Pattern
```python
# InFlightOrder tracks individual orders through their lifecycle
class InFlightOrder:
    order_id: str              # Client order ID
    exchange_order_id: str     # Exchange's order ID
    trading_pair: str
    trade_type: TradeType     # BUY/SELL
    order_type: OrderType     # LIMIT, MARKET, etc.
    status: OrderState         # PENDING, OPEN, FILLED, CANCELLED, FAILED
    fill_timestamp: float
    _order_tracker: ClientOrderTracker  # References connector's tracker
```

### Connector Initialization Pattern
```python
class BinanceExchange(ExchangePyBase):
    def __init__(self,
                 binance_api_key: str,
                 binance_api_secret: str,
                 balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
                 rate_limits_share_pct: Decimal = Decimal("100"),
                 trading_pairs: Optional[List[str]] = None,
                 trading_required: bool = True,
                 domain: str = CONSTANTS.DEFAULT_DOMAIN):
        # Exchange-specific initialization
        self.api_key = binance_api_key
        self.secret_key = binance_api_secret
        self._domain = domain
        # Call parent init (includes throttler, order tracker, etc.)
        super().__init__(balance_asset_limit, rate_limits_share_pct)
```

---

## 2. Order Book / Market Data Subscription Pattern

### Order Book Tracker Pattern
```python
# hummingbot/core/data_type/order_book_tracker.py
class OrderBookTracker:
    def __init__(self, data_source: OrderBookTrackerDataSource):
        self._data_source = data_source
        self._order_book: Dict[str, OrderBook] = {}
        self._tracking_tasks: List[asyncio.Task] = []

    async def start(self):
        """Start tracking order books for all trading pairs"""
        for trading_pair in self._data_source.trading_pairs:
            self._tracking_tasks.append(asyncio.create_task(
                self._track_order_book(trading_pair)
            ))

    async def _track_order_book(self, trading_pair: str):
        """Continuously receive and process order book updates"""
        async for order_book_message in self._data_source.get_messages(trading_pair):
            # Process diff/snapshot messages
            self._order_book[trading_pair] = order_book_message
```

### Data Source Pattern
```python
# Binance example - binance_api_order_book_data_source.py
class BinanceAPIOrderBookDataSource(OrderBookTrackerDataSource):
    def __init__(self, trading_pairs: List[str], connector, domain: str, api_factory):
        self._api_factory = api_factory
        self._domain = domain

    async def _order_book_snapshot(self, trading_pair: str) -> OrderBookMessage:
        """Fetch snapshot via REST, then apply diffs via WebSocket"""
        # REST call to get initial snapshot
        snapshot = await self._api_get(path_url=CONSTANTS.SNAPSHOT_PATH_URL, params={...})
        # Return OrderBookMessage with snapshot data

    async def _parse_trade_message(self, message: Dict) -> OrderBookMessage:
        """Parse WebSocket trade event"""
        # Returns trade update message
```

### Candles Feed Pattern (Strategy V2)
```python
# hummingbot/data_feed/candles_feed/data_types.py
@dataclass
class CandlesConfig:
    connector: str                    # Exchange name
    trading_pair: str                 # Trading pair
    interval: str                     # "1m", "5m", "1h", etc.
    max_records: int = 500           # Max candles to cache

# Usage in controller
def get_candles_config(self) -> List[CandlesConfig]:
    return [
        CandlesConfig(
            connector="binance_perpetual",
            trading_pair="SOL-USDT",
            interval="1m",
            max_records=300
        )
    ]

# Fetching candles
candles_df = self.market_data_provider.get_candles_df(
    connector_name=self.config.connector_pair_dominant.connector_name,
    trading_pair=self.config.connector_pair_dominant.trading_pair,
    interval=self.config.interval,
    max_records=self.max_records
)
```

---

## 3. Order Placement / Cancellation Pattern

### Place Order Flow
```python
# Via connector (low-level)
async def _place_order(self, order_id, trading_pair, amount, trade_type, order_type, price, **kwargs):
    """Exchange-specific order placement"""
    symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
    api_params = {
        "symbol": symbol,
        "side": "BUY" if trade_type == TradeType.BUY else "SELL",
        "quantity": f"{amount:f}",
        "type": order_type.name.upper(),
        "newClientOrderId": order_id
    }
    if order_type == OrderType.LIMIT:
        api_params["price"] = f"{price:f}"
        api_params["timeInForce"] = "GTC"

    order_result = await self._api_post(
        path_url=CONSTANTS.ORDER_PATH_URL,
        data=api_params,
        is_auth_required=True
    )
    return str(order_result["orderId"]), order_result["transactTime"] * 1e-3

# Via Strategy Executor (high-level)
class PositionExecutor:
    def place_order(self, connector_name, trading_pair, order_type, side, amount, price):
        return self._strategy.buy(connector_name, trading_pair, amount, order_type, price)
```

### Cancel Order Flow
```python
async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
    """Cancel an order"""
    symbol = await self.exchange_symbol_associated_to_pair(trading_pair=tracked_order.trading_pair)
    cancel_result = await self._api_delete(
        path_url=CONSTANTS.ORDER_PATH_URL,
        params={"symbol": symbol, "origClientOrderId": order_id},
        is_auth_required=True
    )
    return cancel_result.get("status") == "CANCELED"
```

### Controller Order Actions (Strategy V2)
```python
# Controller creates executor actions
def determine_executor_actions(self) -> List[ExecutorAction]:
    actions: List[ExecutorAction] = []

    # Create new position executor
    if should_enter:
        config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name="binance_perpetual",
            trading_pair="SOL-USDT",
            side=TradeType.BUY,
            entry_price=price,
            amount=amount,
            triple_barrier_config=TripleBarrierConfig(take_profit=Decimal("0.01")),
            leverage=20
        )
        actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=config))

    # Stop existing executor
    for executor in executors_to_stop:
        actions.append(StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id,
            keep_position=False  # or True to keep position
        ))

    return actions
```

---

## 4. Balance / Position Tracking Pattern

### Balance Tracking
```python
# In ExchangePyBase
class ExchangePyBase:
    def __init__(self, ...):
        self._account_balances: Dict[str, Decimal] = {}         # Total balance
        self._account_available_balances: Dict[str, Decimal] = {}  # Available

    async def _update_balances(self):
        """Fetch and update balances from exchange"""
        account_info = await self._api_get(
            path_url=CONSTANTS.ACCOUNTS_PATH_URL,
            is_auth_required=True
        )
        for balance_entry in account_info["balances"]:
            asset = balance_entry["asset"]
            free = Decimal(balance_entry["free"])
            locked = Decimal(balance_entry["locked"])
            self._account_balances[asset] = free + locked
            self._account_available_balances[asset] = free

    def get_balance(self, asset: str) -> Decimal:
        return self._account_balances.get(asset, Decimal("0"))

    def get_available_balance(self, asset: str) -> Decimal:
        return self._account_available_balances.get(asset, Decimal("0"))
```

### Position Tracking (Strategy V2)
```python
# Position summary from executor
@dataclass
class PositionSummary:
    connector_name: str
    trading_pair: str
    side: TradeType              # LONG or SHORT
    amount: Decimal              # Position size
    entry_price: Decimal
    current_price: Decimal
    pnl_quote: Decimal           # Unrealized PnL in quote
    pnl_percentage: Decimal
    timestamp: float

# Controller tracks positions
class ControllerBase:
    def __init__(self, config, market_data_provider, actions_queue):
        self.positions_held: List[PositionSummary] = []

    async def update_processed_data(self):
        """Update positions from executors"""
        # Position updates come through executor events
```

---

## 5. Key Differences from quant_trading's CCXT-Based Adapters

### quant_trading Current Adapter Pattern
```python
# quant_trading/exchanges/adapter.py
class ExchangeAdapter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.exchange = ccxt.binance({...})  # Direct ccxt wrapper

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit)

    async def create_order(self, symbol, type, side, amount, price):
        return await self.exchange.create_order(symbol, type, side, amount, price)
```

### Hummingbot vs quant_trading Comparison

| Aspect | Hummingbot | quant_trading (CCXT) |
|--------|------------|---------------------|
| **Architecture** | Event-driven, stateful connector | Simple async wrapper |
| **Order Tracking** | Full `InFlightOrder` lifecycle management | Stateless, returns raw exchange response |
| **Rate Limiting** | Built-in `AsyncThrottler` per endpoint | ccxt built-in (enableRateLimit) |
| **Order Book** | Real-time WebSocket with diff/snapshot | `fetch_order_book` (snapshot only) |
| **Balance Updates** | Polling + WebSocket stream | `fetch_balance` polling only |
| **Position Tracking** | Per-executor, stored in `positions_held` | Manual via `fetch_positions` |
| **Strategy Integration** | Controller -> Executor -> Connector | Direct adapter calls |
| **Extensibility** | Exchange-specific base class per connector | Generic ccxt interface |

### What's Worth Absorbing

**High Value:**
1. **Executor Pattern** - Separates signal generation (Controller) from execution (Executor)
2. **Event-Driven Balance Updates** - WebSocket stream for real-time balance changes
3. **Candles Feed** - Unified OHLCV interface with caching
4. **Triple Barrier** - Take profit, stop loss, open order management
5. **Order Action Queue** - Async queue-based action processing

**Medium Value:**
1. **Rate Limiter Per Endpoint** - Fine-grained rate limit management
2. **Order State Machine** - `InFlightOrder` with proper state transitions
3. **Trading Rules Validation** - Automatic min order size, tick size enforcement

**Low Value (Keep as Reference):**
1. Cython components - Performance optimization not needed currently
2. Gateway/DEX integration - Different use case
3. Complex multi-connector orchestration

---

## 6. Recommendations for Integration

### Recommended: Controller-Executor Pattern

```python
# quant_trading/strategy/base.py (suggested extension)
class ControllerBase(ABC):
    """Abstract base for strategy controllers"""

    def __init__(self, config: StrategyConfig, market_data_provider):
        self.config = config
        self.market_data_provider = market_data_provider
        self.executors_info: List[ExecutorInfo] = []

    @abstractmethod
    async def update_processed_data(self):
        """Update market data and compute signals"""
        pass

    @abstractmethod
    def determine_executor_actions(self) -> List[ExecutorAction]:
        """Determine what executors to create/modify/stop"""
        pass


class ExecutorBase(ABC):
    """Abstract base for order executors"""

    def __init__(self, config: ExecutorConfig, connectors: Dict[str, ExchangeAdapter]):
        self.config = config
        self.connectors = connectors
        self.status = ExecutorStatus.PENDING

    @abstractmethod
    async def execute(self):
        """Execute the order"""
        pass

    @abstractmethod
    async def update(self):
        """Update executor state (fill, pnl, etc.)"""
        pass
```

### Recommended: Candles Feed Integration

```python
# quant_trading/data/candles.py (suggested)
class CandlesFeed:
    """Unified candles data feed"""

    def __init__(self, adapters: Dict[str, ExchangeAdapter]):
        self._adapters = adapters
        self._candles: Dict[str, deque] = {}  # symbol -> deque of candles

    async def subscribe(self, connector: str, symbol: str, timeframe: str):
        """Subscribe to real-time candle updates"""
        # Use exchange's WebSocket for real-time updates
        # Fall back to polling if WebSocket unavailable

    async def get_candles_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get candles as DataFrame"""
        return pd.DataFrame(list(self._candles[symbol])[-limit:])
```

### Reference Only: Order Book Real-Time

```python
# For future enhancement - keep as reference
# Hummingbot's OrderBookTracker with WebSocket is complex
# Consider: use exchange-specific WebSocket order book if available
# Most exchanges now support depth streams (e.g., binance: <symbol>@depth@100ms)
```

---

## 7. Stat Arb Pattern Reference

The statistical arbitrage controller (`controllers/generic/stat_arb.py`) provides:

### Key Concepts
- **Cointegration**: Two assets whose spread is mean-reverting
- **Z-Score**: (spread - mean) / std_dev of spread
- **Hedge Ratio**: Beta from linear regression of returns

### Signal Logic
```
if z_score > entry_threshold:     # Spread too high
    signal = 1                      # Long dominant, short hedge
elif z_score < -entry_threshold:    # Spread too low
    signal = -1                     # Short dominant, long hedge
else:
    signal = 0                      # No signal
```

### Position Management
- Uses theoretical position sizes based on `total_amount_quote`
- `pos_hedge_ratio` controls relative sizing
- Triple barrier for take profit / stop loss on pair PnL

### Files to Reference
- Original: `D:/Hive/Data/trading_repos/hummingbot/controllers/generic/stat_arb.py`
- Adapted: `D:/量化交易系统/量化之神/quant_trading/strategy/stat_arb_hummingbot.py`
