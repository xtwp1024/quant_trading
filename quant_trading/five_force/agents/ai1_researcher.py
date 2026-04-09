from ...modules.five_force.agents.base_agent import BaseAgent
from ...modules.honey_badger import HoneyBadger
from ...modules.five_force.strategy_parser import StrategyGenomeParser
import asyncio
import random
import os

class AI1_Researcher(BaseAgent):
    """
    AI1: The Researcher (Hunter)
    Responsibility: Find strategies, validate factors, splice logic.
    """
    def __init__(self, kb, bus):
        super().__init__("AI1_Researcher", kb, bus)
        self.hunter = HoneyBadger()
        self.hunt_interval = 300 # 5 mins
        
        # Initialize Genome Parser
        strategies_path = os.path.join(os.getcwd(), "modules", "strategies")
        self.parser = StrategyGenomeParser(strategies_path)
        self.gene_bank = [] # Store extracted logic
        self.has_learned = False

    async def on_tick(self):
        """
        AI1 Periodically goes hunting and learns.
        """
        # 0. Learn from Local Strategies (Genome Extraction)
        if not self.has_learned:
             self.logger.info("🎓 AI1 Starting Deep Learning of Local Strategies...")
             genes = self.parser.harvest_genes()
             self.gene_bank = genes
             self.has_learned = True
             
             buy_genes = len([g for g in genes if g['type'] == 'BUY'])
             sell_genes = len([g for g in genes if g['type'] == 'SELL'])
             
             msg = f"🧬 Genome Learned! Extracted {len(genes)} Logic Fragments ({buy_genes} Entries, {sell_genes} Exits) from {len(set(g['strategy'] for g in genes))} Strategies."
             self.logger.info(msg)
             await self.kb.log_audit(self.agent_id, "GENOME_LEARN", msg)
             
             # Show a few examples
             if genes:
                 ex = genes[0]
                 self.logger.info(f"   🔬 Sample Gene ({ex['strategy']} {ex['type']}): {ex['condition']}")

        # 1. Check if we need more strategies (Ask KB)
        # For now, just infinite hunt schedule
        self.logger.info("🔭 Scanning horizon for Alpha...")
        
        # Simulate Hunt Step (Async wrapper for HoneyBadger)
        # HoneyBadger's methods are synchronous (requests), so we run in executor if needed.
        # For simplicity in this v1, we just pick a random keyword and search once per tick cycle (long sleep).
        
        keywords = ["binance grid python", "crypto arbitrage"]
        query = random.choice(keywords)
        
        # We don't want to block the event loop, so we'd ideally offload this.
        # But for prototype, we just log:
        self.logger.info(f"   Simulating Deep Hunt for '{query}'...")
        
        # TODO: Integrate actual hb.search_github(query) here safely
        
        # 3. Strategy Splicer (Iteration 9)
        if random.random() > 0.8:
            self.logger.info("🧬 AI1 Splicer: Combining 'MACD Entry' + 'BB Exit' -> New Strategy 'MacdBollinger'.")
            await self.kb.log_audit(self.agent_id, "STRAT_GEN", "Generated Hybrid Strategy: MacdBollinger")
        
        await asyncio.sleep(self.hunt_interval)

    async def check_factor_health(self):
        """
        Monitor decay of active strategies (Factors).
        If Win Rate < 40% or IC < 0.02, deprecate.
        """
        # Mocking retrieving Strategy Performance from KB
        # In real system, AI4 writes performance stats to Redis: 'stats:strategy_name'
        active_strategies = ["TrendStrategy", "GridStrategy", "MomentumStrategy"]
        
        for strat in active_strategies:
             # Mock Stats
             win_rate = random.uniform(0.3, 0.6) # Random win rate
             
             if win_rate < 0.35:
                 self.logger.warning(f"📉 Factor Decay Detected: {strat} (WinRate: {win_rate:.2f})")
                 # Action: Tag as 'DEPRECATED' in KB
                 await self.kb.log_audit(self.agent_id, "FACTOR_KILL", f"{strat} disabled due to Alpha Decay.")
                 # await self.kb.disable_strategy(strat)
             else:
                 self.logger.info(f"✨ Factor Valid: {strat} (WinRate: {win_rate:.2f})")
