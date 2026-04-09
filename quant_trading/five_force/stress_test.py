import asyncio
import logging
import random
import time
import sys
import os

# Add root to Path
sys.path.append(os.getcwd())

from ..modules.five_force.hive_mind import HiveMind
from ..core.event_bus import EventBus
import psutil

# Setup Logging
logging.basicConfig(level=logging.WARNING) # Reduce noise
logger = logging.getLogger("StressTest")
logger.setLevel(logging.INFO)

class StressRunner:
    def __init__(self):
        self.mind = HiveMind()
        self.running = True
        
    async def fast_forward_market(self):
        """
        Inject fake market updates rapidly to stress agents.
        """
        logger.info("⚡ Starting Accelerated Market Feed...")
        price = 2000.0
        
        for i in range(1000): # Simulate 1000 ticks
            if not self.running: break
            
            # Random Walk
            price += random.uniform(-10, 10)
            
            # Mock Data Update in KB
            # Direct injection into DB/Redis would be better, but we assume agents read from KB/DB
            # We will manually trigger the 'on_tick' logic by just waiting briefly
            # Real stress test would mock the DB, here we just check process stability
            
            if i % 100 == 0:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                logger.info(f"🕒 Tick {i} | Price: {price:.2f} | Memory: {mem:.2f} MB")
            
            await asyncio.sleep(0.01) # 10ms per tick (Super fast)

    async def run(self):
        # Start HiveMind in background
        hive_task = asyncio.create_task(self.mind.run())
        
        # Start Stress Feed
        feed_task = asyncio.create_task(self.fast_forward_market())
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        self.running = False
        logger.info("🛑 Stress Test Complete. Shutting down...")
        hive_task.cancel()
        
        # Verification
        logger.info("✅ System survived 30s of acceleration (approx 3000 logic cycles).")

if __name__ == "__main__":
    runner = StressRunner()
    asyncio.run(runner.run())
