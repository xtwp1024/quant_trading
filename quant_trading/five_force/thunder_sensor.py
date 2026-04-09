import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("ThunderSensor")

class ThunderSensor:
    """
    Real-Time Market Stream Processor.
    detects 'Regime Changes' (e.g. Calm -> Storm) in milliseconds.
    Inspired by: X (Twitter) Real-time Pipeline (Thunder).
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.prices = []
        self.returns = []
        
    def process_stream(self, candle):
        """
        Ingests a single candle dict {'close': 100, ...}
        Returns: { 'regime': 'CALM'|'STORM', 'entropy': float }
        """
        close = candle['close']
        self.prices.append(close)
        
        if len(self.prices) > 1:
            ret = np.log(self.prices[-1] / self.prices[-2])
            self.returns.append(ret)
            
        # Maintain window
        if len(self.returns) > self.window_size:
            self.returns.pop(0)
            self.prices.pop(0)
            
        return self._compute_state()
        
    def _compute_state(self):
        if len(self.returns) < self.window_size:
            return {'regime': 'BOOTSTRAP', 'entropy': 0.0}
            
        # 1. Volatility (Standard Deviation)
        vol = np.std(self.returns)
        
        # 2. Entropy (Shannon Limit of returns histogram)
        # We discretize returns into bins to calculate entropy
        hist, _ = np.histogram(self.returns, bins=5, density=True)
        # Avoid log(0)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        # Thresholds (Dynamic in full version, static for demo)
        if vol > 0.02: # 2% volatility in window
            regime = "CRISIS"
        elif entropy > 1.5:
            regime = "CHAOS" # Distributed randomness
        else:
            regime = "CALM"
            
        return {'regime': regime, 'entropy': float(entropy), 'vol': float(vol)}

if __name__ == "__main__":
    sensor = ThunderSensor(window_size=10)
    # Simulate a crash
    prices = [100, 101, 100, 102, 101, 100, 95, 90, 85, 80]
    for p in prices:
        state = sensor.process_stream({'close': p})
        print(f"Price: {p} -> {state}")
