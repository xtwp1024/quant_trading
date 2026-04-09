"""

Gate.io Exchange Adapter

Supports spot and futures trading

"""


import os



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


                    return True


                except Exception as auth_err:


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


                logger.info("Gate.io running in public data mode (no API key)")


                return True




        except Exception as e:


            logger.error(f"Gate.io initialization failed: {e}")


            return False





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


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")




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


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


        gate_symbol = self._convert_symbol_to_gate(symbol)




        try:


            if order_type == 'market':


                order = await self.exchange.create_market_order(gate_symbol, side, amount)


            elif order_type == 'limit':


                if price is None:


                    raise ValueError("Limit orders require a price")


                order = await self.exchange.create_limit_order(gate_symbol, side, amount, price)


            else:


                raise ValueError(f"Unsupported order type: {order_type}")




            logger.info(f"Gate.io order created: {side.upper()} {amount} {gate_symbol} @ {order_type}")


            return order




        except Exception as e:


            logger.error(f"Gate.io order creation failed: {e}")


            raise




    async def create_trigger_order(self, symbol: str, trigger_price: float, rule: int, order_type: str) -> Dict[str, Any]:


        """


        Create trigger order (take profit/stop loss)



        Args:


            symbol: Trading pair (e.g. 'AZTEC/USDT:USDT')


            trigger_price: Trigger price


            rule: 1 (>=), 2 (<=)


            order_type: 'close-long-position' or 'close-short-position'


        """


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


        gate_symbol = self._convert_symbol_to_gate(symbol)




        # Build Gate.io V4 API payload (fixed format)


        # close=True means close position, size=0


        params = {


            'settle': 'usdt',


            'contract': gate_symbol,


            'size': 0,  # 0 = close position


            'price': '0',


            'close': True,  # Key: close flag


            'trigger': {


                'strategy_type': 0,


                'price_type': 0,


                'price': str(trigger_price),


                'rule': rule,


                'expiration': 86400


            }


        }




        # Exponential backoff retry mechanism (ref: EvoMap HTTP Retry Capsule)


        max_retries = 3


        base_delay = 1.0




        for attempt in range(max_retries):


            try:


                # HIGH: 禁止在日志中输出完整params，防止敏感信息泄露
                logger.debug(f"Gate.io trigger order params prepared for symbol: {symbol}")


                response = await self.exchange.private_futures_post_settle_price_orders(params)


                logger.info(f"Gate.io trigger order created: {symbol} @ {trigger_price} (Rule={rule}, Type={order_type})")


                return response


            except KeyError as e:
                logger.error(f"Gate.io API path error (KeyError: {e}), skipping this trigger order")
                return {}
            except Exception as e:


                if attempt < max_retries - 1:


                    import asyncio


                    delay = base_delay * (2 ** attempt)  # Exponential backoff


                    logger.warning(f"Trigger order failed (attempt {attempt+1}/{max_retries}), retrying in {delay}s: {e}")


                    await asyncio.sleep(delay)


                else:


                    import traceback


                    logger.error(f"Trigger order final failure: {e}")


                    logger.error(traceback.format_exc())


                    return {}




    async def cancel_all_trigger_orders(self, symbol: str) -> bool:


        """Cancel all trigger orders (for given symbol)"""


        if not self.exchange: return False




        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            # 1. Fetch all trigger orders


            # GET /futures/usdt/price_orders


            req_params = {


                'status': 'active',


                'contract': gate_symbol


            }




            orders = await self.exchange.private_futures_get_settle_price_orders(req_params)




            for o in orders:


                oid = o.get('id')


                if oid:


                    await self.exchange.private_futures_delete_settle_price_orders_order_id(oid)




            logger.info(f"All trigger orders cancelled for {symbol}")


            return True


        except Exception as e:


            logger.warning(f"Failed to cancel trigger orders: {e}")


            return False




    async def get_position(self, symbol: Optional[str] = None) -> List[Dict]:


        """Fetch position information"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")




        try:


            # Gate.io futures positions - fetch all without param to avoid parsing errors


            positions = await self.exchange.fetch_positions()


            if not positions:


                return []




            # Filter out zero positions


            active_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]




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


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


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






    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:


        """Query a single order"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            order = await self.exchange.fetch_order(order_id, gate_symbol)


            return order


        except Exception as e:


            logger.error(f"Failed to query order {order_id}: {e}")


            return {}




    async def fetch_open_orders(self, symbol: str) -> List[Dict]:


        """Query current open orders"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


        gate_symbol = self._convert_symbol_to_gate(symbol)


        try:


            orders = await self.exchange.fetch_open_orders(gate_symbol)


            return orders


        except Exception as e:


            logger.error(f"Failed to query open orders for {symbol}: {e}")


            return []




    async def cancel_order(self, order_id: str, symbol: str) -> bool:


        """Cancel a single order"""


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


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


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


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


        if not self.exchange:


            raise RuntimeError("Exchange not initialized")


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
