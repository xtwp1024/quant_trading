import ast
import os
import logging

logger = logging.getLogger("StrategyParser")

class StrategyGenomeParser:
    """
    Parses strategy source code to extract 'Genes' (Buy/Sell Logic).
    Uses AST (Abstract Syntax Tree) to safely analyze code structure.
    """
    def __init__(self, strategies_dir):
        self.strategies_dir = strategies_dir
        
    async def harvest_genes(self, fast_mode=True, allowed_sources=None):
        """
        Scans all strategies and extracts logic snippets.
        Recursively explores subdirectories.
        allowed_sources: list of trusted directory paths (default: only self.strategies_dir)
        """
        genes = []
        allowed_sources = allowed_sources or [self.strategies_dir]
        for root, dirs, files in os.walk(self.strategies_dir):
            if ".git" in root or "__pycache__" in root:
                continue
            # Security: Verify path is within allowed sources
            if not any(root.startswith(src) for src in allowed_sources):
                logger.warning(f"Skipping untrusted source path: {root}")
                continue
            for filename in files:
                if filename.endswith(".py") and filename != "__init__.py" and filename != "base_strategy.py":
                    path = os.path.join(root, filename)
                    try:
                        if fast_mode:
                            file_genes = self._heuristic_parse(path, filename)
                        else:
                            file_genes = self._parse_file(path, filename)

                        if file_genes:
                            genes.extend(file_genes)
                    except Exception:
                        pass

        return genes

    def _parse_file(self, filepath, filename):
        # Limit file size to 5MB for AST parsing to avoid memory issues with giant files like NFI
        if os.path.getsize(filepath) > 5 * 1024 * 1024:
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception:
            # Fallback to Regex for corrupted or non-UTF8 files or massive files that fail AST
            return self._heuristic_parse(filepath, filename)

        genes = []
        strategy_name = filename.replace(".py", "")
        
        # Heuristic: Detect methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Local Interface
                if node.name == "generate_signals":
                    genes.extend(self._extract_conditions(node, source, strategy_name))
                # Freqtrade Interface
                elif node.name in ["populate_buy_trend", "populate_sell_trend", "populate_entry_trend", "populate_exit_trend"]:
                    if "buy" in node.name or "entry" in node.name:
                        action = "BUY"
                    else:
                        action = "SELL"
                    genes.extend(self._extract_freqtrade_conditions(node, source, strategy_name, action))
                
        return genes

    def _extract_freqtrade_conditions(self, func_node, source, strategy_name, action):
        extracted = []
        # Freqtrade typically uses: dataframe.loc[(condition), 'buy/enter_long/...'] = 1
        signal_cols = ["'buy'", '"buy"', "'sell'", '"sell"', "'enter_long'", '"enter_long"', "'exit_long'", '"exit_long"', "'enter_short'", '"enter_short"', "'exit_short'", '"exit_short"']
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        target_src = ast.get_source_segment(source, target)
                        if target_src and (".loc" in target_src):
                            # Check if matches any signal column
                            is_signal = any(col in target_src for col in signal_cols)
                            if is_signal:
                                if isinstance(node.value, (ast.Constant, ast.Num)):
                                    val = getattr(node.value, 'value', getattr(node.value, 'n', None))
                                    if val in [1, True]:
                                        condition_src = ast.get_source_segment(source, target.slice)
                                        if condition_src:
                                            # Determine if this specific assignment is BUY or SELL based on col name
                                            local_action = action
                                            if any(x in target_src for x in ["sell", "exit"]):
                                                local_action = "SELL"
                                            elif any(x in target_src for x in ["buy", "enter"]):
                                                local_action = "BUY"

                                            gene = {
                                                "strategy": strategy_name,
                                                "type": local_action,
                                                "condition": condition_src.strip(),
                                                "file": strategy_name # Keep it simple for memory storage
                                            }
                                            extracted.append(gene)
        return extracted

    def _heuristic_parse(self, filepath, filename):
        """
        Regex-based fallback for very large or complex files.
        """
        import re
        genes = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Look for Freqtrade-style assignments: dataframe.loc[..., 'buy'] = 1
            # Using broader regex to catch multiple signal columns and variable names (df or dataframe)
            # Pattern: (df|dataframe).loc[condition, 'buy/sell/...'] = 1/True
            buy_matches = re.findall(r"(?:df|dataframe)\.loc\[(.*),\s*['\"](?:buy|enter_long|enter_short)[^'\"]*['\"]\]\s*=\s*[1True]", content)
            for cond in buy_matches:
                genes.append({"strategy": filename, "type": "BUY", "condition": cond.strip(), "file": filepath})
                
            sell_matches = re.findall(r"(?:df|dataframe)\.loc\[(.*),\s*['\"](?:sell|exit_long|exit_short)[^'\"]*['\"]\]\s*=\s*[1True]", content)
            for cond in sell_matches:
                genes.append({"strategy": filename, "type": "SELL", "condition": cond.strip(), "file": filepath})
        except Exception:
            pass
        return genes

    def _extract_conditions(self, func_node, source, strategy_name):
        extracted = []
        
        # Recursive walk through the function to find 'If' statements that append to signals
        # This is a heuristic: If we find `signals.append({... "action": "BUY" ...})` inside an If block,
        # we capture the If condition as the "Trigger".
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                # Check body for signal append
                action = self._detect_action_in_body(node)
                if action:
                    # Extract source code segment for the condition
                    condition_segment = ast.get_source_segment(source, node.test)
                    
                    # Store Gene
                    gene = {
                        "strategy": strategy_name,
                        "type": action, # BUY or SELL
                        "condition": condition_segment,
                        "file": strategy_name
                    }
                    extracted.append(gene)
                    
        return extracted

    def _detect_action_in_body(self, if_node):
        """
        Checks if the If-body appends a BUY or SELL signal.
        """
        for child in ast.walk(if_node):
            if isinstance(child, ast.Dict):
                # Iterate through key-value pairs
                # Note: child.keys can contain None (for **heuristics), but usually simple dicts don't
                for k, v in zip(child.keys, child.values):
                    if not k: continue 
                    
                    # Extract Key String
                    key_str = None
                    if isinstance(k, ast.Constant): # Python 3.8+
                        key_str = k.value
                    elif isinstance(k, ast.Str): # Python < 3.8
                        key_str = k.s
                        
                    if key_str == "action":
                        # Extract Value String
                        val_str = None
                        if isinstance(v, ast.Constant):
                            val_str = v.value
                        elif isinstance(v, ast.Str):
                            val_str = v.s
                            
                        if val_str in ["BUY", "SELL"]:
                            return val_str
        return None
