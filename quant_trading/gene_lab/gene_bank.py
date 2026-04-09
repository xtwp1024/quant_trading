import json
import os
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("GeneBank")

# 常量定义
DEFAULT_DB_PATH = "gene_pool.json"


class GeneBank:
    """
    The DNA Vault for Quant Strategies.
    Stores extracted logic snippets (Genes) categorized by:
    - Type (ENTRY / EXIT)
    - Origin (Strategy Name)
    - Complexity (Lines of Code)
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.genes: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """加载基因库数据"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load GeneBank: {e}")
                return []
        return []

    def save(self) -> None:
        """保存基因库数据"""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.genes, f, indent=4)
        logger.info(f"GeneBank saved with {len(self.genes)} genes.")

    def deposit_genes(self, new_genes: List[Dict[str, Any]]) -> int:
        """
        Adds new genes to the pool. Avoids duplicates based on 'condition' string.
        """
        count = 0
        existing_conditions = {g['condition'] for g in self.genes}

        for gene in new_genes:
            if gene['condition'] not in existing_conditions:
                # Add Metadata
                gene['id'] = f"GENE-{len(self.genes) + count + 1}"
                gene['discovered_at'] = datetime.now().isoformat()
                self.genes.append(gene)
                existing_conditions.add(gene['condition'])
                count += 1

        self.save()
        return count

    def get_random_gene(self, gene_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """随机获取一个基因"""
        candidates = self.genes
        if gene_type:
            candidates = [g for g in self.genes if g['type'] == gene_type]

        if not candidates:
            return None
        return random.choice(candidates)

    def get_stats(self) -> Dict[str, int]:
        """获取基因库统计信息"""
        return {
            "total_genes": len(self.genes),
            "entry_genes": len([g for g in self.genes if g['type'] == "BUY"]),
            "exit_genes": len([g for g in self.genes if g['type'] == "SELL"])
        }
