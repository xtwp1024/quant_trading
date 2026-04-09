import asyncio
import logging
import random
import sys
import os

# Add root to Path
sys.path.append(os.getcwd())

from ..modules.five_force.hive_mind import HiveMind

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Arena")

class ArenaRunner:
    """
    Iteration 10: Grand Arena.
    Simulates a competitive environment where strategies fight for allocation.
    """
    def __init__(self):
        self.mind = HiveMind()
        self.strategies = [
            "TrendStrategy", "GridStrategy", "MomentumStrategy", 
            "MacdCrossStrategy", "DualThrustStrategy"
        ]
        
    async def run_battle(self):
        logger.info("⚔️ OPENING THE GRAND ARENA ⚔️")
        logger.info(f"Combatants: {', '.join(self.strategies)}")
        
        # Simulate 30 days of battle in seconds
        days = 30
        scores = {s: 1000 for s in self.strategies} # Starting ELO/Capital
        
        for day in range(1, days+1):
            logger.info(f"📅 Day {day} Begins...")
            
            # 1. Simulate Daily Returns (Random)
            daily_returns = {}
            for s in self.strategies:
                ret = random.gauss(0.001, 0.02) # Mean 0.1%, Std 2%
                daily_returns[s] = ret
                
            # 2. AI4 Monitor Ranks them
            sorted_strats = sorted(daily_returns.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_strats[0]
            loser = sorted_strats[-1]
            
            # 3. Reallocate Capital (AI5 Arbiter Role Logic)
            scores[winner[0]] += 50
            scores[loser[0]] -= 50
            
            logger.info(f"   🏆 Winner: {winner[0]} (+{winner[1]*100:.2f}%) -> Score: {scores[winner[0]]}")
            logger.info(f"   💀 Loser:  {loser[0]} ({loser[1]*100:.2f}%) -> Score: {scores[loser[0]]}")
            
            await asyncio.sleep(0.5) # Fast forward
            
        logger.info("🏁 BATTLE ENDED. FINAL STANDINGS:")
        for s, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
             logger.info(f"   {s}: ${score}")

if __name__ == "__main__":
    arena = ArenaRunner()
    asyncio.run(arena.run_battle())
