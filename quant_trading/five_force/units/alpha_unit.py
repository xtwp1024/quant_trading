from ...modules.five_force.cognitive_unit import CognitiveUnit
from ...modules.honey_badger import HoneyBadger
from ...modules.five_force.strategy_parser import StrategyGenomeParser
import os
import random

class AlphaUnit(CognitiveUnit):
    """
    FORMERLY: AI1_Researcher
    NEW IDENTITY: Alpha Unit (α) - The Pattern Resonator.
    Core Function: Fractal Analysis, Cross-Timeframe Resonance, Gene Splicing.
    """
    def __init__(self, kb, bus):
        super().__init__("AlphaUnit(α)", kb, bus)
        self.hunter = HoneyBadger()
        
        # Genome Parser (Retained Skill)
        strategies_path = os.path.join(os.getcwd(), "modules", "strategies")
        self.parser = StrategyGenomeParser(strategies_path)
        self.gene_bank = []
        self.has_learned = False

    async def process(self, perception):
        """
        Alpha Unit's Cognitive Process.
        """
        thoughts = []
        
        # 1. Genome Learning (One-time awakening)
        if not self.has_learned:
             self.logger.info("🎓 Alpha Unit absorbing Strategy Genomes...")
             genes = self.parser.harvest_genes()
             self.gene_bank = genes
             self.has_learned = True
             thoughts.append(f"Learned {len(genes)} Genes")

        # 2. Fractal Scan (Mock)
        # In future, this will scan for Hurst Exponent > 0.5
        if random.random() > 0.95:
             thoughts.append("Resonance Detected: Market Fractal Dimension aligning.")
             
        # 3. Strategy Splicing (Retained)
        if random.random() > 0.98:
             self.logger.info("🧬 Alpha Splicing: Combining Genes to form 'ChimeraStrategy'...")
             # await self.kb.log_audit(...)
             
        return thoughts
