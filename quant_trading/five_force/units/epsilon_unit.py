from ...modules.five_force.cognitive_unit import CognitiveUnit
from ...modules.five_force.time_warp import TimeWarpExecutor
import asyncio
import random
import time

class EpsilonUnit(CognitiveUnit):
    """
    FORMERLY: AI4_Monitor
    NEW IDENTITY: Epsilon Unit (ε) - The Time Crystal.
    Core Function: Non-Linear Time Perception, Anomaly Detection, Historical Alignment, Latency Management.
    """
    def __init__(self, kb, bus, config):
        super().__init__("EpsilonUnit(ε)", kb, bus)
        self.warper = TimeWarpExecutor()
        self.time_dilation = 1.0 # 1.0 = Realtime

    async def process(self, perception):
        # 1. Measure Latency
        start = time.time_ns()
        
        # 2. Historical Alignment (Attribution)
        # Mocking attribution of recent PnL to specific Factors
        attribution = {"Trend": 0.6, "MeanRev": 0.3, "Noise": 0.1}
        
        # 3. Time Warp Check
        # If latency is high, we might 'Slow Down' time (Dilation) -> Throttle system
        duration_ms = (time.time_ns() - start) / 1_000_000
        
        mode = "Nominal"
        if duration_ms > 50:
            self.time_dilation = 0.8 # Slow down
            mode = "Dilated"
        else:
            self.time_dilation = 1.0
            
        # 4. Active Time Warping (Random Execution Delay Simulation)
        # Epsilon occasionally injects a micro-delay to desynchronize order flow (Anti-Gaming)
        if perception.get("execute_warp_test", False):
            # Define async dummy
            async def mock_order(size):
                pass
                
            await self.warper.execute_warped_order(mock_order, 1000, splits=3, mode='Warped')
            mode = "Warping Active"
            
        msg = f"Time flow {mode} ({duration_ms:.0f}ms). Attribution Generated."
        self.logger.info(f"⏳ {self.name}: {msg}")
        return msg
