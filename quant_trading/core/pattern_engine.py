import chromadb
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - PATTERN - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternEngine")

# 常量定义
DEFAULT_DB_PATH = r"D:\Hive_Data\chroma_db"
DEFAULT_LOOKBACK = 50
DEFAULT_TOP_K = 5
MOCK_SIMILARITY_SCORE = 0.65


class PatternEngine:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, lookback: int = DEFAULT_LOOKBACK) -> None:
        self.db_path = db_path
        self.lookback = lookback
        self.client = None
        self.collection = None
        self.connected = False

        self._connect()

    def _connect(self) -> None:
        """建立数据库连接"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            # Try to get or create a collection for market patterns
            # We'll assume a collection named 'market_patterns_15m' exists or we create it
            self.collection = self.client.get_or_create_collection(name="market_patterns_15m")
            self.connected = True
            logger.info(f"Connected to ChromaDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.connected = False

    def find_similar_patterns(self, current_candles: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Vectorizes the current market state and searches for similar historical patterns.
        """
        if not self.connected or len(current_candles) < self.lookback:
            return {"score": 0.5, "matches": []}

        # 1. Vectorize: Normalize close prices to specific range (last 50 candles)
        # Simple normalization: (Price - Min) / (Max - Min)
        recent = current_candles.iloc[-self.lookback:].copy()
        closes = recent['close'].values.astype(float)

        min_p = np.min(closes)
        max_p = np.max(closes)

        if max_p - min_p == 0:
            normalized = np.zeros(len(closes))
        else:
            normalized = (closes - min_p) / (max_p - min_p)

        vector = normalized.tolist()

        # 2. Query DB
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k
            )

            # 3. Analyze Results (Mock analysis of returned metadata)
            # In a real system, metadata would contain 'future_return' or 'label'
            # Here we will simulate a "Win Rate" calculation based on mock distances

            # distances = results['distances'][0] # Lower is better
            # metadatas = results['metadatas'][0] # Should contain subsequent price action

            # MOCK LOGIC: Since we don't have populated data with 'outcome' in this exact collection likely
            # We will generate a synthetic score for now, but logged.

            avg_similarity = MOCK_SIMILARITY_SCORE  # Placeholder

            return {
                "score": avg_similarity,
                "matches": results['ids'][0] if results['ids'] else [],
                "note": f"Found {len(results['ids'][0]) if results['ids'] else 0} similar patterns."
            }

        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return {"score": 0.5, "matches": []}

    def save_pattern(self, candles: pd.DataFrame, outcome_label: float) -> None:
        """
        Saves a pattern to the DB (Self-Learning).
        outcome_label: 1.0 (win) or -1.0 (loss)
        """
        if not self.connected or len(candles) < self.lookback:
            return

        recent = candles.iloc[-self.lookback:].copy()
        closes = recent['close'].values.astype(float)

        min_p = np.min(closes)
        max_p = np.max(closes)

        if max_p - min_p == 0:
            return

        normalized = (closes - min_p) / (max_p - min_p)
        vector = normalized.tolist()

        pattern_id = f"pat_{int(time.time() * 1000)}"

        try:
            self.collection.add(
                embeddings=[vector],
                metadatas=[{"outcome": outcome_label, "timestamp": int(time.time())}],
                ids=[pattern_id]
            )
            logger.info(f"Saved new pattern {pattern_id} with outcome {outcome_label}")
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
