"""

Gate.io Exchange Adapter

Supports spot and futures trading

"""


import os
import math



from typing import Any, Dict, List, Optional





import ccxt.async_support as ccxt









from .logger import logger









class GateExchangeAdapter:


    """

    Gate.io Exchange Adapter


    Supported Features:


    - Spot trading

    - Perpetual contracts (USDT margin)


    - Market/Limit orders

    - Account queries

    """





    def __init__(self, config: Dict[str, Any]):


        self.config = config


        self.exchange: Optional[ccxt.Exchange] = None


        self.exchange_name = config.get('exchange', {}).get('name', 'gate')




        # Gate.io specific configuration


        gate_config = config.get('exchange', {}).get('gate', {})


        self.api_key = os.getenv('GATE_API_KEY', gate_config.get('api_key', ''))


        self.api_secret = os.getenv('GATE_API_SECRET', gate_config.get('secret', ''))


        self.uid = os.getenv('GATE_UID', gate_config.get('uid', ''))
        self._authentication_failed = False
        self._is_authenticated = False





    async def initialize(self) -> bool:


        """Initialize Gate.io connection"""


        try:


            # Gate.io public API endpoint


            # Mainnet API: https://api.gateio.ws


            # Backup: https://api.gate.io


            exchange_config = {


                'apiKey': self.api_key,


                'secret': self.api_secret,


                'enableRateLimit': True,


                'timeout': 30000,  # 30 second timeout


                'options': {


                    'defaultType': 'swap',  # Default to futures


                    'adjustForTimeDifference': True,  # Auto time calibration


                }


            }




            # Create Gate.io instance (CCXT will auto use public API)


            self.exchange = ccxt.gate(exchange_config)




            # Set UID if provided


            if self.uid:


                self.exchange.headers = {


                    'X-GATE-UID': self.uid


                }




            logger.info("Connecting to Gate.io public API: https://api.gateio.ws")




            # Load market data


            await self.exchange.load_markets()




            # Test connection


            if self.api_key and self.api_secret:


                try:


                    balance = await self.exchange.fetch_balance()


                    if balance:


                        logger.info("Gate.io API authentication successful")


                        logger.info(f"Account types: {list(balance.keys())}")


                    else:


                        logger.warning("Gate.io API auth: fetch_balance returned empty")


                    self._is_authenticated = True
                    self._authentication_failed = False
                    return True


                except Exception as auth_err:


                    self._is_authenticated = False
                    self._authentication_failed = True
                    logger.warning(f"Gate.io authentication failed: {auth_err}. Falling back to public data mode")


                    # Reinitialize in public mode


                    await self.exchange.close()


                    self.exchange = ccxt.gate({


                        'enableRateLimit': True,


                        'options': {'defaultType': 'swap'}


                    })


                    await self.exchange.load_markets()


                    return True


            else:


                self._is_authenticated = False
                self._authentication_failed = False
                logger.info("Gate.io running in public data mode (no API key)")


                return True




        except Exception as e:


            logger.error(f"Gate.io initialization failed: {e}")


            return False





    def _require_authenticated(self, operation: str):
        """Ensure private trading operations only run with authenticated credentials"""

        if not self.exchange:
            raise RuntimeError("Exchange not initialized")
        if not self.is_authenticated:
            raise RuntimeError(f"Gate authentication required for {operation}")


    async def get_ticker(self, symbol: str) -> Dict[str, Any]:


        """Fetch ticker data"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")




        # Gate.io uses different symbol format


        gate_symbol = self._convert_symbol_to_gate(symbol)




        try:


            ticker = await self.exchange.fetch_ticker(gate_symbol)


            if not ticker or not isinstance(ticker, dict):


                logger.warning(f"Failed to fetch ticker for {symbol}: returned non-dict type {type(ticker)}")


                return {}




            # Convert back to standard format


            return {


                'symbol': symbol,


                'last': ticker.get('last'),


                'bid': ticker.get('bid'),


                'ask': ticker.get('ask'),


                'volume': ticker.get('baseVolume'),


                'quoteVolume': ticker.get('quoteVolume'),


                'change': ticker.get('change'),


                'percentage': ticker.get('percentage'),


            }


        except Exception as e:


            logger.error(f"Failed to fetch ticker for {symbol}: {e}")


            return {}




    async def get_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 50) -> List[List]:


        """Fetch OHLCV kline data"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


        gate_symbol = self._convert_symbol_to_gate(symbol)




        try:


            ohlcv = await self.exchange.fetch_ohlcv(gate_symbol, timeframe=timeframe, limit=limit)


            return ohlcv if ohlcv else []


        except Exception as e:


            logger.error(f"Failed to fetch kline for {symbol}: {e}")


            return []




    async def get_balance(self) -> Dict[str, Any]:


        """Get account balance"""


        self._require_authenticated("fetch balance")


        try:


            balance = await self.exchange.fetch_balance()


            return balance if balance else {}


        except Exception as e:


            logger.error(f"Failed to fetch balance: {e}")


            return {}




    def get_contract_size(self, symbol: str) -> float:
        """Get contract multiplier"""

        if not self.exchange:
            return 1.0

        try:
            # Try direct fetch (CCXT Standard Symbol)
            market = self.exchange.market(symbol)
            return float(market.get('contractSize', 1.0))
        except Exception:  # noqa: BLE001
            try:
                # Try Gate Symbol
                gate_symbol = self._convert_symbol_to_gate(symbol)
                market = self.exchange.market(gate_symbol)
                return float(market.get('contractSize', 1.0))
            except Exception:  # noqa: BLE001
                return 1.0





    async def create_order(


        self,


        symbol: str,


        order_type: str,


        side: str,


        amount: float,


        price: Optional[float] = None,


        params: Optional[Dict] = None


    ) -> Dict[str, Any]:


        """


        Create an order



        Args:


            symbol: Trading pair


            order_type: Order type ('market' or 'limit')


            side: Direction ('buy' or 'sell')


            amount: Quantity


            price: Price (required for limit orders)


            params: Extra parameters



        Returns:


            Order information


        """


        self._require_authenticated("create order")


        gate_symbol = self._convert_symbol_to_gate(symbol)




        try:


            if order_type == 'market':


                order = await self.exchange.create_market_order(gate_symbol, side, amount, params or {})


            elif order_type == 'limit':


                if price is None:


                    raise ValueError("Limit orders require a price")


                order = await self.exchange.create_limit_order(gate_symbol, side, amount, price, params or {})


            else:


                raise ValueError(f"Unsupported order type: {order_type}")




            logger.info(f"Gate.io order created: {side.upper()} {amount} {gate_symbol} @ {order_type}")


            return order




        except Exception as e:


            logger.error(f"Gate.io order creation failed: {e}")


            raise




    async def create_trigger_order(
        self,
        symbol: str,
        trigger_price: float,
        rule: int,
        order_type: str,
        close_order_type: Optional[str] = None,
        auto_size: Optional[str] = None,
    ) -> Dict[str, Any]:


        """


        Create trigger order (take profit/stop loss)



        Args:


            symbol: Trading pair (e.g. 'AZTEC/USDT:USDT')


            trigger_price: Trigger price


            rule: 1 (<= trigger), 2 (>= trigger)


            order_type: 'stop_loss' or 'take_profit'


            close_order_type: Gate close position order type


            auto_size: Gate hedge-mode close target


        """


        self._require_authenticated("create trigger order")


        if trigger_price is None:


            raise ValueError("trigger_price is required")


        normalized_trigger_price = float(trigger_price)


        if not math.isfinite(normalized_trigger_price):


            raise ValueError("trigger_price must be finite")


        if normalized_trigger_price <= 0:


            raise ValueError("trigger_price must be positive")


        if close_order_type not in ('close-long-position', 'close-short-position'):
            raise ValueError(f"Unsupported close_order_type: {close_order_type}")

        if auto_size is not None and auto_size not in ('close_long', 'close_short'):
            raise ValueError(f"Unsupported auto_size: {auto_size}")

        expected_auto_size = {
            'close-long-position': 'close_long',
            'close-short-position': 'close_short',
        }[close_order_type]
        if auto_size is None:
            pass
        elif auto_size != expected_auto_size:
            raise ValueError(
                f"auto_size {auto_size} does not match close_order_type {close_order_type}"
            )

        gate_symbol = self._convert_symbol_to_gate(symbol)




        # Build Gate.io V4 API payload (fixed format)
        initial = {
            'contract': gate_symbol,
            'size': 0,
            'price': '0',
            'tif': 'ioc',
            'reduce_only': True,
        }
        if auto_size is None:
            initial['close'] = True
        else:
            initial['auto_size'] = auto_size

        params = {
            'settle': 'usdt',
            'order_type': close_order_type,
            'initial': initial,
            'trigger': {
                'strategy_type': 0,
                'price_type': 0,
                'price': str(normalized_trigger_price),
                'rule': rule,
                'expiration': 86400
            }
        }


        try:


            # HIGH: 禁止在日志中输出完整params，防止敏感信息泄露
            logger.debug(f"Gate.io trigger order params prepared for symbol: {symbol}")


            response = await self.exchange.private_futures_post_settle_price_orders(params)


            logger.info(
                f"Gate.io trigger order created: {symbol} @ {normalized_trigger_price} "
                f"(Rule={rule}, Type={order_type}, CloseType={close_order_type}, AutoSize={auto_size})"
            )


            return response


        except KeyError as e:
            logger.error(f"Gate.io API path error (KeyError: {e})")
            raise RuntimeError(f"Gate.io trigger order endpoint unavailable: {e}") from e
        except Exception as e:


            import traceback


            logger.error(f"Trigger order creation failed without retry: {e}")


            logger.error(traceback.format_exc())


            raise RuntimeError(f"Gate.io trigger order creation failed: {e}") from e




    async def fetch_trigger_orders(self, symbol: str) -> List[Dict[str, Any]]:


        """Fetch active trigger orders for a symbol"""


        self._require_authenticated("fetch trigger orders")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            req_params = {
                'settle': 'usdt',
                'status': 'active',
                'contract': gate_symbol
            }


            orders = await self.exchange.private_futures_get_settle_price_orders(req_params)
            if not isinstance(orders, list):
                raise RuntimeError(f"Unexpected trigger orders response type: {type(orders)}")
            return orders


        except Exception as e:


            logger.error(f"Failed to fetch trigger orders for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch trigger orders for {symbol}: {e}") from e


    async def cancel_trigger_order(self, order_id: str) -> bool:


        """Cancel a single trigger order"""


        self._require_authenticated("cancel trigger order")


        try:


            await self.exchange.private_futures_delete_settle_price_orders_order_id({
                'settle': 'usdt',
                'order_id': order_id,
            })
            logger.info(f"Trigger order cancelled: {order_id}")
            return True


        except Exception as e:


            logger.warning(f"Failed to cancel trigger order {order_id}: {e}")
            return False


    async def cancel_all_trigger_orders(self, symbol: str) -> bool:


        """Cancel all trigger orders (for given symbol)"""


        self._require_authenticated("cancel trigger orders")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            # 1. Fetch all trigger orders


            # GET /futures/usdt/price_orders


            req_params = {
                'settle': 'usdt',
                'status': 'active',
                'contract': gate_symbol
            }


            orders = await self.exchange.private_futures_get_settle_price_orders(req_params)




            for o in orders:


                oid = o.get('id')


                if oid:


                    await self.cancel_trigger_order(str(oid))




            logger.info(f"All trigger orders cancelled for {symbol}")


            return True


        except Exception as e:


            logger.warning(f"Failed to cancel trigger orders: {e}")


            return False




    async def get_position(self, symbol: Optional[str] = None) -> List[Dict]:


        """Fetch position information"""


        self._require_authenticated("fetch positions")


        try:


            # Gate.io futures positions - fetch all without param to avoid parsing errors


            positions = await self.exchange.fetch_positions()


            if not positions:


                return []




            active_positions = []
            for p in positions:
                contracts = float(p.get('contracts') or 0)
                raw_size = float(p.get('info', {}).get('size') or 0)
                if raw_size != 0 or contracts != 0:
                    active_positions.append(p)




            # DEBUG: Print all symbols with positions


            if active_positions:


                logger.info(f"Current positions: {[p.get('symbol') for p in active_positions]}")




            if symbol:


                # Try to match symbol (CCXT symbol may use /, Gate symbol uses _)


                # Simple contains match or exact match


                gate_symbol = self._convert_symbol_to_gate(symbol)


                # Check CCXT symbol info values


                filtered = []


                for p in active_positions:


                    # p['symbol'] might be 'AZTEC_USDT' or 'AZTEC/USDT:USDT'


                    # p['info']['contract'] might be 'AZTEC_USDT'


                    s = p.get('symbol', '')


                    c = p.get('info', {}).get('contract', '')




                    # Loose match: symbol string appears in any field


                    if symbol in s or gate_symbol == c or symbol in c:


                        filtered.append(p)


                    else:


                         # Log unmatched items for debugging


                         logger.debug(f"Skipping unmatched position Symbol={s}, Contract={c} vs Target={symbol}/{gate_symbol}")




                return filtered




            return active_positions




        except Exception as e:


            logger.error(f"Failed to fetch positions: {e}")


            return []





    async def set_leverage(self, leverage: int, symbol: str) -> bool:


        """Set leverage multiplier"""


        self._require_authenticated("set leverage")


        gate_symbol = self._convert_symbol_to_gate(symbol)




        try:


            await self.exchange.set_leverage(leverage, gate_symbol)


            logger.info(f"Leverage set for {gate_symbol} = {leverage}x")


            return True


        except Exception as e:


            logger.warning(f"Failed to set leverage for {gate_symbol}: {e}")


            return False




    async def close(self):


        """Close connection"""


        if self.exchange:


            await self.exchange.close()


            logger.info("Gate.io connection closed")




    def _convert_symbol_to_gate(self, symbol: str) -> str:


        """


        Convert trading pair format to Gate.io format




        Gate.io format:


        - Spot: BTC_USDT


        - Futures: BTC_USDT (perpetual)




        OKX format (input): BTC-USDT-SWAP


        """


        # Handle new CCXT Unified Symbol format: BASE/QUOTE:QUOTE (e.g. AZTEC/USDT:USDT)


        if ':' in symbol:


            symbol = symbol.split(':')[0]  # Take BASE/QUOTE, ignore settlement currency for symbol name




        if '/' in symbol:


            return symbol.replace('/', '_')




        # Remove -SWAP suffix and replace - with _


        if '-SWAP' in symbol:


            base = symbol.replace('-SWAP', '')


            return f"{base.replace('-', '_')}"


        else:


            return symbol.replace('-', '_')




    def _convert_symbol_from_gate(self, gate_symbol: str) -> str:


        """


        Convert from Gate.io format back to standard format


        """


        # BTC_USDT -> BTC-USDT-SWAP (assuming futures)


        parts = gate_symbol.split('_')


        if len(parts) == 2:


            return f"{parts[0]}-{parts[1]}-SWAP"


        return gate_symbol






    @property


    def is_initialized(self) -> bool:


        """Check if adapter is initialized"""


        return self.exchange is not None


    @property


    def is_authenticated(self) -> bool:


        """Check if adapter has authenticated trading credentials"""


        return self._is_authenticated


    @property


    def authentication_failed(self) -> bool:


        """Check if authentication was attempted but failed"""


        return self._authentication_failed






    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:


        """Query a single order"""


        self._require_authenticated("fetch order")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            order = await self.exchange.fetch_order(order_id, gate_symbol)


            return order


        except Exception as e:


            logger.error(f"Failed to query order {order_id}: {e}")


            return {}




    async def fetch_open_orders(self, symbol: str) -> List[Dict]:


        """Query current open orders"""


        self._require_authenticated("fetch open orders")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            orders = await self.exchange.fetch_open_orders(gate_symbol)


            return orders


        except Exception as e:


            logger.error(f"Failed to query open orders for {symbol}: {e}")


            return []




    async def cancel_order(self, order_id: str, symbol: str) -> bool:


        """Cancel a single order"""


        self._require_authenticated("cancel order")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            await self.exchange.cancel_order(order_id, gate_symbol)


            logger.info(f"Order cancelled: {order_id}")


            return True


        except Exception as e:


            logger.error(f"Failed to cancel order {order_id}: {e}")


            return False




    async def cancel_all_orders(self, symbol: str) -> bool:


        """Cancel all open orders"""


        self._require_authenticated("cancel all orders")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            await self.exchange.cancel_all_orders(gate_symbol)


            logger.info(f"All open orders cancelled: {symbol}")


            return True


        except Exception as e:


            logger.warning(f"Failed to cancel all orders (may have no open orders): {e}")


            return False





    async def fetch_my_trades(self, symbol: str, limit: int = 50) -> List[Dict]:


        """Query trade history"""


        self._require_authenticated("fetch trade history")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            # fetch_my_trades returns a list of trades


            trades = await self.exchange.fetch_my_trades(gate_symbol, limit=limit)


            return trades


        except Exception as e:


            logger.error(f"Failed to query trades for {symbol}: {e}")


            return []




def create_exchange_adapter(config: Dict[str, Any]) -> Optional[GateExchangeAdapter]:


    """Factory function: create exchange adapter"""


    exchange_name = config.get('exchange', {}).get('name', 'okx')




    if exchange_name.lower() == 'gate':


        adapter = GateExchangeAdapter(config)


        return adapter


    else:


        return None  # OKX or other exchanges use native CCXT method
