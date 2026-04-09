# -*- coding: utf-8 -*-
import asyncio
import logging
import random
from typing import Dict, Any, List
from dataclasses import dataclass
import uuid

logger = logging.getLogger("Sandbox")

@dataclass
class Scenario:
    name: str
    params: Dict[str, Any]

class SecureEnclave:
    """
    Simulates a Trusted Execution Environment (TEE) for testing purposes.
    WARNING: This is a MOCK implementation for sandbox testing only.
    Data is NOT actually encrypted - do NOT use in production.
    """
    def __init__(self):
        self._memory = {}
        self._locked = False

    def mock_encrypt_and_store(self, key: str, data: Any):
        """
        Mock encryption for sandbox testing.
        WARNING: This is NOT real encryption - only prefix is added.
        For production, use a proper TEE or encryption library.
        """
        if self._locked:
            raise PermissionError("Enclave is Locked")
        # MOCK: In production, use proper encryption (e.g., cryptography library)
        self._memory[key] = f"MOCK_ENC_{data}"

    def execute_in_isolation(self, func, *args):
        """Execute a function inside the 'black box'"""
        logger.info(f"🔒 [ENCLAVE] Executing {func.__name__} in secure memory...")
        return func(*args)

class MockExchange:
    """
    A simulated exchange that mimics CCXT but operates in a closed loop.
    Supports Scenario Injection (e.g., Flash Crash).
    """
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = {"USDT": initial_balance, "ETH": 0.0, "BTC": 0.0}
        self.orders = []
        self.market_price = {"ETH-USDT-SWAP": 2000.0, "BTC-USDT-SWAP": 60000.0}
        self.time_offset = 0
        
    async def fetch_ticker(self, symbol: str):
        price = self.market_price.get(symbol, 0)
        # Add some noise
        noise = random.uniform(-1, 1)
        return {
            "symbol": symbol,
            "last": price + noise,
            "ask": price + noise + 0.5,
            "bid": price + noise - 0.5
        }
        
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None):
        logger.info(f"📝 [SANDBOX] Order Placed: {side} {amount} {symbol} @ {price or 'Market'}")
        
        # Simulate Execution
        cost = amount * (price if price else self.market_price.get(symbol, 0))
        fee = cost * 0.001
        
        if side == 'buy':
            if self.balance["USDT"] >= cost + fee:
                self.balance["USDT"] -= (cost + fee)
                base = symbol.split('-')[0]
                self.balance[base] = self.balance.get(base, 0) + amount
                return {'id': str(uuid.uuid4()), 'status': 'closed', 'filled': amount}
            else:
                logger.warning("❌ [SANDBOX] Insufficient Funds")
                return {'id': None, 'status': 'rejected'}
                
        elif side == 'sell':
            # Simplified sell logic
             base = symbol.split('-')[0]
             if self.balance.get(base, 0) >= amount:
                 self.balance[base] -= amount
                 self.balance["USDT"] += (cost - fee)
                 return {'id': str(uuid.uuid4()), 'status': 'closed', 'filled': amount}
             
        return {'id': None, 'status': 'rejected'}

class SandboxEnvironment:
    """
    The War Room. Orchestrates the simulation.
    """
    def __init__(self, config):
        self.config = config
        self.enclave = SecureEnclave()
        self.exchange = MockExchange(config['sandbox']['sim_capital'])
        self.active_scenario = None
        
    def load_scenario(self, scenario_name: str):
        scenarios = self.config['sandbox']['scenarios']
        target = next((s for s in scenarios if s['name'] == scenario_name), None)
        if target:
            self.active_scenario = Scenario(target['name'], target)
            logger.info(f"🎭 [SANDBOX] Scenario Loaded: {scenario_name}")
            
    async def run_simulation(self, duration_sec: int = 10):
        logger.info(f"🚀 [SANDBOX] Simulation Step ({duration_sec}s)...")
        
        # Purely advance time/process background tasks
        # Price updates are now handled externally by the Adversarial Generator
        for i in range(duration_sec):
            await asyncio.sleep(0.1 if duration_sec > 1 else 0.01) # Accelerated time
            
        logger.info(f"🏁 [SANDBOX] Simulation Ended. Final Balance: {self.exchange.balance}")
        
if __name__ == "__main__":
    # Test Run
    import yaml
    cfg = {'sandbox': {'sim_capital': 100000, 'security': {'enclave_mode': 'TEST'}, 'scenarios': [{'name': 'Flash_Crash', 'drop_pct': 0.1}]}}
    
    sb = SandboxEnvironment(cfg)
    sb.load_scenario("Flash_Crash")
    asyncio.run(sb.run_simulation(5))
