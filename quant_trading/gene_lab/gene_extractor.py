import ast
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("GeneExtractor")

# 常量定义
MAX_CONDITION_CODE_LENGTH = 500


class GeneExtractor:
    """
    Advanced AST Parser for Logic Extraction.
    Upgrade from StrategyParser: Identifies Imports and Parameters.
    """
    def __init__(self, strategies_dir: str) -> None:
        self.strategies_dir = strategies_dir

    def extract_all(self) -> List[Dict[str, Any]]:
        """提取所有策略文件中的基因"""
        genes: List[Dict[str, Any]] = []
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                path = os.path.join(self.strategies_dir, filename)
                try:
                    genes.extend(self._parse_file(path, filename))
                except Exception as e:
                    logger.error(f"Failed to parse {filename}: {e}")
        return genes

    def _parse_file(self, filepath: str, filename: str) -> List[Dict[str, Any]]:
        """解析单个策略文件"""
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        genes: List[Dict[str, Any]] = []

        # 1. Strategy Identity
        strategy_name = filename.replace(".py", "")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                strategy_name = node.name
                break

        # 2. Imports (To ensure gene viability)
        imports: List[Optional[str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        # 3. Logic Genes (in generate_signals)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
                genes.extend(self._extract_logic(node, source, strategy_name, imports))

        return genes

    def _extract_logic(
        self,
        func_node: ast.FunctionDef,
        source: str,
        strategy_name: str,
        imports: List[Optional[str]]
    ) -> List[Dict[str, Any]]:
        """提取逻辑基因"""
        extracted: List[Dict[str, Any]] = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                action = self._detect_signal_action(node)
                if action:
                    condition_code = ast.get_source_segment(source, node.test)

                    # Heuristic: Complexity Check
                    # If too short (True) or too long (> 500 chars), skip
                    if not condition_code or len(condition_code) > MAX_CONDITION_CODE_LENGTH:
                        continue

                    gene = {
                        "strategy": strategy_name,
                        "type": action,  # BUY/SELL
                        "condition": condition_code,
                        "dependencies": list(set(imports)),  # Dedup
                        "weight": 1.0
                    }
                    extracted.append(gene)

        return extracted

    def _detect_signal_action(self, if_node: ast.If) -> Optional[str]:
        """
        Detects if 'BUY' or 'SELL' is triggered in the If-block.
        Looks for: return {... 'action': 'BUY' ...} OR signals.append({...})
        """
        for child in ast.walk(if_node):
            # Case 1: Dict literal
            if isinstance(child, ast.Dict):
                # Safe iteration over paired keys/values
                for k, v in zip(child.keys, child.values):
                    # Key must be a string literal "action"
                    if not k:
                        continue  # Skip None keys (heuristics)

                    key_val: Optional[str] = None
                    if isinstance(k, (ast.Str, ast.Constant)):
                        key_val = getattr(k, "value", getattr(k, "s", None))

                    if key_val == "action":
                        # Value must be "BUY" or "SELL" literal
                        val_str: Optional[str] = None
                        if isinstance(v, (ast.Str, ast.Constant)):
                            val_str = getattr(v, "value", getattr(v, "s", None))

                        if val_str in ["BUY", "SELL"]:
                            return val_str

            # Case 2: String constant (sometimes simple assignment action="BUY")
            if isinstance(child, (ast.Str, ast.Constant)):
                val = getattr(child, "value", getattr(child, "s", None))
                if val in ["BUY", "SELL"]:
                    # Weak signal but often true in context
                    pass

        return None
