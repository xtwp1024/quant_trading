"""
FinMem-Layer: FinMem-inspired layered memory architecture for quantitative trading.

Absorbed from: D:/Hive/Data/trading_repos/FinMem-LLM-StockTrading/puppy/

Architecture (5 layers, matching original FinMem):
    PERCEPTUAL  -- Raw market stimuli: ticks, news, economic prints.
                   Fastest decay. Minimal retention. Filters sensory noise.
    SHORT_TERM  -- Recent trading memories: today's positions, recent signals.
                   High recency weight. Medium decay. Working memory.
    MID_TERM   -- Cross-day patterns: trends, sector rotations.
                   Balanced importance and recency. Bridge layer.
    LONG_TERM  -- Enduring knowledge: verified strategies, fundamental models.
                   High importance weight. Slowest decay. Core expertise.
    REFLECTION -- Synthesized insights from accumulated experience.
                   Highest-value distilled knowledge. Meta-cognitive layer.

Layer transitions (memory jump):
    Perceptual --> Short
    Short <--> Mid  (bidirectional: promote high-importance, demote low)
    Mid  <--> Long  (bidirectional)
    Long --> Reflection  (promotion only, when insight is synthesized)

Each layer is independently tunable via layer_params.
Scoring per entry: importance_score (layer-initialized, access-updated)
                 + recency_score (exponential decay)
                 + compound_score (importance/100 + recency)
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, date
from typing import List, Union, Dict, Any, Optional, Callable, Tuple

from quant_trading.memory.memory_db import MemoryDatabase
from quant_trading.memory.compound_scorer import CompoundScorer
from quant_trading.memory.exponential_decay import ExponentialDecay
from quant_trading.memory.importance_scorer import ImportanceScorerByLayer
from quant_trading.memory.recency_scorer import RecencyScorer


class FinMemLayer:
    """
    FinMem-style layered memory for quantitative trading agents.

    Coordinates five memory layers with bidirectional self-evolving transitions.
    All layers share the same symbol namespace (e.g., "AAPL", "BTC-USD").

    Example:
        >>> fm = FinMemLayer(symbols=["AAPL", "TSLA"])
        >>> fm.add_memory("AAPL", "Q4 earnings beat by 12%", layer="perceptual")
        >>> fm.add_memory("TSLA", "Short-term overbought RSI=78", layer="short")
        >>> results = fm.query("AAPL", top_k=3, layer="short")
        >>> fm.step()  # advance time: decay + cleanup + jumps
    """

    # Default cognitive span parameters per layer
    # Derived from original FinMem config files (tsla_gpt_config.toml etc.)
    DEFAULT_LAYER_PARAMS: Dict[str, Dict[str, Any]] = {
        "perceptual": {
            "jump_threshold_upper": 70.0,
            "jump_threshold_lower": -999999.0,
            "decay_recency_factor": 5.0,
            "decay_importance_factor": 0.95,
            "cleanup_recency_threshold": 0.01,
            "cleanup_importance_threshold": 5.0,
        },
        "short": {
            "jump_threshold_upper": 70.0,
            "jump_threshold_lower": -999999.0,
            "decay_recency_factor": 10.0,
            "decay_importance_factor": 0.988,
            "cleanup_recency_threshold": 0.01,
            "cleanup_importance_threshold": 5.0,
        },
        "mid": {
            "jump_threshold_upper": 80.0,
            "jump_threshold_lower": 20.0,
            "decay_recency_factor": 20.0,
            "decay_importance_factor": 0.99,
            "cleanup_recency_threshold": 0.005,
            "cleanup_importance_threshold": 15.0,
        },
        "long": {
            "jump_threshold_upper": 999999.0,
            "jump_threshold_lower": 10.0,
            "decay_recency_factor": 50.0,
            "decay_importance_factor": 0.995,
            "cleanup_recency_threshold": 0.001,
            "cleanup_importance_threshold": 25.0,
        },
        "reflection": {
            "jump_threshold_upper": 999999.0,
            "jump_threshold_lower": -999999.0,
            "decay_recency_factor": 100.0,
            "decay_importance_factor": 0.999,
            "cleanup_recency_threshold": 0.0001,
            "cleanup_importance_threshold": 30.0,
        },
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        layer_params: Optional[Dict[str, Dict[str, Any]]] = None,
        id_generator: Optional[Callable[[], int]] = None,
        logger: Optional[logging.Logger] = None,
        semantic_query_fns: Optional[Dict[str, Callable[[str, str, int], List[Dict]]]] = None,
    ):
        """
        Args:
            symbols:              List of asset tickers to initialize.
            layer_params:         Override default cognitive span parameters
                                  for any layer. Merged with defaults.
            id_generator:         Callable returning unique int IDs.
                                  Defaults to a simple counter.
            logger:               Optional logger.
            semantic_query_fns:   Optional semantic search functions per layer:
                                   {"short": fn, "mid": fn, ...}
        """
        self._logger = logger or logging.getLogger("FinMemLayer")
        self._id_gen = id_generator or self._default_id_generator()

        params = {**self.DEFAULT_LAYER_PARAMS}
        if layer_params:
            for layer, overrides in layer_params.items():
                if layer in params:
                    params[layer] = {**params[layer], **overrides}

        self._compound_scorer = CompoundScorer()

        self._layers: Dict[str, MemoryDatabase] = {}
        for layer in ["perceptual", "short", "mid", "long", "reflection"]:
            p = params[layer]
            self._layers[layer] = MemoryDatabase(
                db_name=f"finmem_{layer}",
                layer=layer,
                id_generator=self._id_gen,
                jump_threshold_upper=p["jump_threshold_upper"],
                jump_threshold_lower=p["jump_threshold_lower"],
                compound_scorer=self._compound_scorer,
                decay=ExponentialDecay(
                    recency_factor=p["decay_recency_factor"],
                    importance_factor=p["decay_importance_factor"],
                ),
                importance_scorer=ImportanceScorerByLayer(layer),
                recency_scorer=RecencyScorer(),
                cleanup_recency_threshold=p["cleanup_recency_threshold"],
                cleanup_importance_threshold=p["cleanup_importance_threshold"],
                logger=self._logger,
                semantic_query_fn=(
                    semantic_query_fns.get(layer) if semantic_query_fns else None
                ),
            )

        if symbols:
            for symbol in symbols:
                for layer_db in self._layers.values():
                    layer_db.add_symbol(symbol)

        self._logger.info(
            f"FinMemLayer initialized with symbols={symbols}, "
            f"layers={list(self._layers.keys())}"
        )

    @staticmethod
    def _default_id_generator() -> Callable[[], int]:
        counter = [0]
        def gen() -> int:
            counter[0] += 1
            return counter[0] - 1
        return gen

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_memory(
        self,
        symbol: str,
        text: str,
        layer: str = "perceptual",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add a memory entry to a specific layer.

        Args:
            symbol:    Asset ticker.
            text:      Memory content.
            layer:     Target layer: "perceptual", "short", "mid", "long", or "reflection".
            timestamp: Event time (default now).
        """
        if layer not in self._layers:
            raise ValueError(
                f"Invalid layer '{layer}'. Must be one of: {list(self._layers.keys())}"
            )
        self._layers[layer].add_memory(symbol, text, timestamp)
        self._logger.debug(f"add_memory symbol={symbol} layer={layer} text={text[:50]}...")

    def add_memories(
        self,
        symbol: str,
        texts: Union[str, List[str]],
        layer: str = "perceptual",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add multiple memory entries to a layer in one call."""
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            self.add_memory(symbol, text, layer, timestamp)

    def query(
        self,
        symbol: str,
        layer: str = "short",
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query a specific layer for top-k memories.

        Args:
            symbol:     Asset ticker.
            layer:      Memory layer to query.
            top_k:      Max results.
            query_text: Passed to the layer's semantic_query_fn if set.

        Returns:
            List of memory entry dicts sorted by compound_score.
        """
        if layer not in self._layers:
            raise ValueError(f"Invalid layer '{layer}'")
        return self._layers[layer].query(symbol, top_k, query_text)

    def query_all_layers(
        self,
        symbol: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query all layers and return results grouped by layer name.

        Useful for getting a comprehensive memory picture.
        """
        return {
            layer: self.query(symbol, layer, top_k, query_text)
            for layer in self._layers
        }

    def query_by_layer(
        self,
        symbol: str,
        layers: List[str],
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query a subset of layers."""
        return {
            layer: self.query(symbol, layer, top_k, query_text)
            for layer in layers
            if layer in self._layers
        }

    def update_access(
        self,
        symbol: str,
        entry_ids: List[int],
        feedback: int = 1,
        layer: str = "short",
    ) -> List[int]:
        """
        Apply feedback to accessed memories, updating their importance.

        Args:
            symbol:    Asset ticker.
            entry_ids: IDs of accessed entries.
            feedback:  1 = positive reinforcement, -1 = negative.
            layer:     Layer containing the entries.

        Returns:
            List of successfully updated entry IDs.
        """
        return self._layers[layer].update_access(symbol, entry_ids, feedback)

    def step(self) -> Dict[str, List[int]]:
        """
        Advance all layers by one time step.

        For each layer:
            1. Apply exponential decay to all entries
            2. Remove entries below cleanup thresholds
            3. Execute memory jumps (promotion/demotion) between adjacent layers

        Layer transition flow (two passes for bidirectional flow):
            Pass 1: short->mid (up), mid->long (up)
            Pass 2: mid->short (down), long->mid (down)
            Perceptual always flows into short (new stimuli)
            Long can promote to reflection (synthesized insights)

        Returns:
            Dict mapping layer name -> list of removed entry IDs.
        """
        removed: Dict[str, List[int]] = {}

        # Phase 1: Decay + cleanup + prepare_jump for all layers
        jump_dicts: Dict[str, Tuple] = {}
        for layer in ["perceptual", "short", "mid", "long", "reflection"]:
            removed[layer] = self._layers[layer].step()
            jump_dicts[layer] = self._layers[layer].prepare_jump()

        # Phase 2: Execute jumps

        # Perceptual -> Short (new stimuli always enter working memory)
        jump_up, jump_down, _ = jump_dicts["perceptual"]
        self._layers["short"].accept_jump(jump_up, direction="up")

        # Short <-> Mid
        jump_up, jump_down, _ = jump_dicts["short"]
        self._layers["mid"].accept_jump(jump_up, direction="up")
        self._layers["mid"].accept_jump(jump_down, direction="down")

        # Mid <-> Long
        jump_up, jump_down, _ = jump_dicts["mid"]
        self._layers["long"].accept_jump(jump_up, direction="up")
        self._layers["short"].accept_jump(jump_down, direction="down")

        # Long -> Reflection (promotion only, synthesis of high-value insights)
        jump_up, jump_down, _ = jump_dicts["long"]
        self._layers["reflection"].accept_jump(jump_up, direction="up")

        # Reflection can also demote to long if importance decays
        jump_up_ref, jump_down_ref, _ = jump_dicts["reflection"]
        self._layers["long"].accept_jump(jump_down_ref, direction="down")

        self._logger.debug(
            f"step() removed: perceptual={len(removed['perceptual'])}, "
            f"short={len(removed['short'])}, mid={len(removed['mid'])}, "
            f"long={len(removed['long'])}, reflection={len(removed['reflection'])}"
        )

        return removed

    def add_symbol(self, symbol: str) -> None:
        """Add a new asset ticker across all layers."""
        for layer_db in self._layers.values():
            layer_db.add_symbol(symbol)
        self._logger.info(f"Added symbol across all layers: {symbol}")

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-layer memory statistics."""
        return {layer: db.stats() for layer, db in self._layers.items()}

    def total_entries(self) -> int:
        """Total number of entries across all layers."""
        return sum(len(db) for db in self._layers.values())

    @property
    def layers(self) -> Dict[str, MemoryDatabase]:
        """Direct access to layer MemoryDatabase instances."""
        return self._layers

    def layer(self, name: str) -> MemoryDatabase:
        """Get a specific layer by name."""
        if name not in self._layers:
            raise ValueError(f"Invalid layer '{name}'. Must be one of: {list(self._layers.keys())}")
        return self._layers[name]

    def __repr__(self) -> str:
        total = self.total_entries()
        layer_counts = {layer: len(db) for layer, db in self._layers.items()}
        return f"<FinMemLayer total_entries={total} layer_counts={layer_counts}>"
