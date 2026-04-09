import asyncio
import logging
import aiohttp
import time
import json

logger = logging.getLogger("MarketFeed")

class MarketFeed:
    """
    Real-time Market Data Feed (OKX Public API).
    Fetches Tickers and Orderbooks.
    """
    def __init__(self, symbols=["ETH-USDT-SWAP"]):
        self.symbols = symbols
        self.running = False
        self.session = None
        self.data_callback = None

    async def start(self, callback):
        self.running = True
        self.data_callback = callback
        logger.info("📡 Connecting to OKX Public Feed...")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            while self.running:
                try:
                    tasks = [self._fetch_ticker(sym) for sym in self.symbols]
                    results = await asyncio.gather(*tasks)
                    
                    for res in results:
                        if res:
                            await self.data_callback(res)
                            
                    await asyncio.sleep(1) # Poll every 1s (Rate limit safe)
                except Exception as e:
                    logger.error(f"Feed Error: {e}")
                    await asyncio.sleep(5)

    async def _fetch_ticker(self, symbol):
        # OKX Ticker Endpoint
        # symbol format for OKX: ETH-USDT-SWAP
        try:
            url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['code'] == '0':
                        ticker = data['data'][0]
                        return {
                            'symbol': symbol,
                            'price': float(ticker['last']),
                            'bid': float(ticker['bidPx']),
                            'ask': float(ticker['askPx']),
                            'timestamp': int(time.time() * 1000)
                        }
        except Exception as e:
             # logger.warning(f"Ticker fetch failed: {e}") 
             pass
        return None

    def stop(self):
        self.running = False
