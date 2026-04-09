import random
import os
import logging
from datetime import datetime
from .gene_bank import GeneBank

logger = logging.getLogger("StrategyBreeder")

class StrategyBreeder:
    """
    Artificial Breeder for Quant Strategies.
    combines 'Entry Genes' and 'Exit Genes' to create new species.
    """
    def __init__(self, strategies_dir="modules/strategies"):
        self.bank = GeneBank()
        self.strategies_dir = strategies_dir
        
    def breed(self, count=1):
        """
        Creates 'count' new strategies.
        """
        created = []
        for _ in range(count):
            try:
                new_strat = self._synthesize_organism()
                if new_strat:
                    self._save_to_disk(new_strat)
                    created.append(new_strat['name'])
            except Exception as e:
                logger.error(f"Breeding failed: {e}")
                
        return created
        
    def _synthesize_organism(self):
        # 1. Select Genes
        entry_gene = self.bank.get_random_gene("BUY")
        exit_gene = self.bank.get_random_gene("SELL")
        
        if not entry_gene or not exit_gene:
            logger.warning("Not enough genes to breed.")
            return None
            
        # 2. Generate identity
        timestamp = int(datetime.now().timestamp())
        name = f"GeneticAlpha_{timestamp}_{random.randint(100,999)}"
        
        # 3. Resolve Dependencies
        imports = set()
        imports.update(entry_gene.get('dependencies', []))
        imports.update(exit_gene.get('dependencies', []))
        
        # Base imports
        imports.add("pandas as pd")
        imports.add("numpy as np")
        
        # 4. Construct Code
        code = f"from modules.strategies.base_strategy import BaseStrategy\n"
        
        # Dynamic Imports
        for imp in imports:
            if " " in imp: # e.g. "pandas as pd"
                code += f"import {imp}\n"
            else:
                code += f"import {imp}\n"
                
        # Class Definition
        code += f"\nclass {name}(BaseStrategy):\n"
        code += f"    \"\"\"\n"
        code += f"    Auto-Generated Strategy by The Legion.\n"
        code += f"    Parents: {entry_gene.get('strategy', 'Unknown')} & {exit_gene.get('strategy', 'Unknown')}\n"
        code += f"    \"\"\"\n"
        code += f"    def __init__(self, symbol, config=None):\n"
        code += f"        super().__init__(symbol, config)\n"
        code += f"        self.name = '{name}'\n"
        
        # Signal Generation Logic
        code += f"\n    def generate_signals(self, df):\n"
        code += f"        if len(df) < 100: return None\n"
        code += f"        last = df.iloc[-1]\n"
        
        # We need to ensure variables used in genes (like 'rsi', 'macd') are calculated.
        # This is the TRICKY part. The extracted gene `if rsi > 70` assumes `rsi` exists in local scope.
        # Solution: For V1, we only support genes that use 'last' or 'df'.
        # OR we inject a "Standard Feature Block" that calculates common indicators.
        # Let's inject a "Kitchen Sink" feature calculation block.
        
        code += self._get_feature_calculation_block()
        
        # Entry Logic
        code += f"\n        # Entry Gene (from {entry_gene.get('strategy')})\n"
        code += f"        try:\n"
        code += f"            if {entry_gene['condition']}:\n"
        code += f"                return {{'symbol': self.symbol, 'action': 'BUY', 'price': last['close']}}\n"
        code += f"        except Exception:\n"
        code += f"            pass\n"
        
        # Exit Logic
        code += f"\n        # Exit Gene (from {exit_gene.get('strategy')})\n"
        code += f"        try:\n"
        code += f"            if {exit_gene['condition']}:\n"
        code += f"                return {{'symbol': self.symbol, 'action': 'SELL', 'price': last['close']}}\n"
        code += f"        except Exception:\n"
        code += f"            pass\n"
        
        code += f"\n        return None\n"
        
        return {"name": name, "code": code}

    def _save_to_disk(self, strat_data):
        path = os.path.join(self.strategies_dir, f"{strat_data['name'].lower()}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(strat_data['code'])
        logger.info(f"🐣 Born: {strat_data['name']}")
        
    def _get_feature_calculation_block(self):
        """
        Injects a block of code that calculates common indicators so that
        random gene snippets (which rely on variables like 'rsi', 'ma') can resolve them.
        This is a heuristic 'Context Reconstruction'.
        """
        block = ""
        block += "        # --- Context Helpers ---\n"
        block += "        close = df['close']\n"
        block += "        # Attempt to provide common variables\n"
        
        # We can scan the genes to see what variables they need, but for now we dump common ones.
        # RSI
        block += "        try:\n"
        block += "            import pandas_ta as ta\n"
        block += "            rsi = ta.rsi(close, length=14).iloc[-1]\n"
        block += "            rsi_val = rsi # alias\n"
        block += "        except:\n"
        block += "            rsi = 50\n"

        # SMA/EMA
        block += "        try:\n"
        block += "            sma20 = close.rolling(20).mean().iloc[-1]\n"
        block += "            ema20 = close.ewm(span=20).mean().iloc[-1]\n"
        block += "        except: pass\n"
        
        # MACD
        block += "        try:\n"
        block += "            macd_df = ta.macd(close)\n"
        block += "            # macd line is usually first col\n"
        block += "            macd = macd_df.iloc[-1, 0]\n"
        block += "            signal = macd_df.iloc[-1, 2]\n"
        block += "        except: pass\n"
        
        # Previous candles locals
        block += "        try:\n"
        block += "            prev_close = df['close'].iloc[-2]\n"
        block += "            current_close = last['close']\n"
        block += "        except: pass\n"

        return block
