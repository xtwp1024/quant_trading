import random
import logging
from typing import Dict, Any

logger = logging.getLogger("GeneMutator")

class GeneMutator:
    """
    Phase 3: Dynamic Evolution.
    Mutates strategy parameters based on market regime and performance.
    """
    def __init__(self):
        self.mutation_rate = 0.1 # 10% change usually
        
    def mutate_config(self, config: Dict[str, Any], volatility: float) -> Dict[str, Any]:
        """
        Evolves the configuration dictionary.
        High volatility -> Shorter periods, wider stops.
        Low volatility -> Longer periods, tighter stops.
        """
        evolved = config.copy()
        
        logger.info(f"🧬 Mutating Genes (Vol: {volatility:.4f})...")
        
        for key, value in evolved.items():
            if isinstance(value, int) or isinstance(value, float):
                # Apply Mutation
                if random.random() < 0.3: # 30% chance to mutate a specific gene
                    original = value
                    
                    # Directional Drift based on Volatility
                    drift = 0
                    if "period" in key:
                         # High Vol -> Lower Period (React faster)
                         if volatility > 0.02: drift = -0.1
                         elif volatility < 0.005: drift = 0.1
                    elif "stop" in key:
                         # High Vol -> Wider Stop
                         if volatility > 0.02: drift = 0.2
                         
                    # Random Noise
                    noise = random.uniform(-self.mutation_rate, self.mutation_rate)
                    
                    # Calculate new value
                    new_val = value * (1.0 + drift + noise)
                    
                    # Type safety
                    if isinstance(value, int):
                        new_val = int(new_val)
                        
                    evolved[key] = new_val
                    logger.debug(f"    Gene '{key}': {original} -> {new_val}")
                    
        return evolved
