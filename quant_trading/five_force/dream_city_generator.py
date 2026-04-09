import numpy as np
import pandas as pd
import logging
import random

logger = logging.getLogger("DreamCity")

class DreamCityGenerator:
    """
    Generates synthetic market data for 'Dream Training'.
    Capable of creating:
    1. Normal Market (GBM - Geometric Brownian Motion)
    2. Hell Mode (Flash Crashes, Extreme Volatility)
    3. Pump & Dump Scenarios
    """
    def __init__(self, start_price=1000.0, volatility=0.02):
        self.start_price = start_price
        self.volatility = volatility
        self.dt = 1/24/60 # 1 minute steps assuming 24h trading

    def generate_random_walk(self, length=1000):
        """
        Standard Geometric Brownian Motion.
        """
        prices = [self.start_price]
        for _ in range(length):
            change = np.random.normal(0, self.volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        return self._to_dataframe(prices)

    def generate_flash_crash(self, length=1000, crash_point=500, severity=0.3):
        """
        Simulates a market crash at a specific point.
        severity: Percentage drop (e.g. 0.3 = 30% drop)
        """
        prices = [self.start_price]
        for i in range(length):
            # Normal movement
            change = np.random.normal(0, self.volatility)
            
            # Crash Event
            if i == crash_point:
                change = -severity # Instant drop
            elif i > crash_point and i < crash_point + 50:
                 # High volatility aftershock
                 change = np.random.normal(0, self.volatility * 5)
                 # Slight rebound (Dead cat bounce)
                 change += 0.01 
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(0.01, new_price)) # Ensure no negative price
            
        return self._to_dataframe(prices)

    def generate_pump_and_dump(self, length=1000, start_point=200):
        """
        Simulates a rapid rise followed by a crash.
        """
        prices = [self.start_price]
        phase = "normal"
        
        for i in range(length):
            change = np.random.normal(0, self.volatility)
            
            if i > start_point and i < start_point + 100:
                phase = "pump"
                change += 0.005 # Consistent 0.5% upward drift per minute
            elif i >= start_point + 100 and i < start_point + 150:
                phase = "dump"
                change -= 0.015 # 1.5% drop per minute
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
        return self._to_dataframe(prices)

    def _to_dataframe(self, prices):
        # Create timestamps
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(prices), freq='1min')
        
        # Create OHLC from Close (Simulated)
        data = []
        for i, price in enumerate(prices):
            # Simulate slight OHLC variations
            noise_high = price * (1 + abs(np.random.normal(0, 0.002)))
            noise_low = price * (1 - abs(np.random.normal(0, 0.002)))
            
            open_p = prices[i-1] if i > 0 else price
            close_p = price
            high_p = max(open_p, close_p, noise_high)
            low_p = min(open_p, close_p, noise_low)
            volume = abs(np.random.normal(1000, 500))
            
            data.append({
                "timestamp": int(timestamps[i].timestamp() * 1000),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume
            })
            
        return pd.DataFrame(data)

if __name__ == "__main__":
    gen = DreamCityGenerator()
    df_crash = gen.generate_flash_crash()
    print("Generated Flash Crash Data:")
    print(df_crash.head())
    print(df_crash.iloc[495:505]) # Show around crash point
