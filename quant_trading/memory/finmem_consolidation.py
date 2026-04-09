"""
FinMem Consolidation: Memory consolidation and forgetting logic.

Absorbed from: D:/Hive/Data/trading_repos/FinMem-LLM-StockTrading/puppy/

Implements the consolidation mechanism that drives memory self-evolution:
    1. Memory jump (promotion/demotion) between adjacent layers
    2. Forgetting: cleanup of decayed entries below thresholds
    3. Reflection synthesis: distillation of high-importance memories
       into the reflection layer

The consolidation cycle runs on each step() call. Each layer independently
evaluates its entries for promotion (high importance) or demotion
(low importance). The original FinMem uses a two-pass approach for
bidirectional flow between adjacent layers.

Key mechanisms:
    - Jump thresholds: upper (promote) and lower (demote) per layer
    - Recency refresh: promoted entries regain full recency (delta=0)
    - Importance decay: multiplicative decay per time step
    - Forgetting: entries removed when recency OR importance drops below
      their layer-specific cleanup thresholds
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple

from quant_trading.memory.memory_db import MemoryDatabase


# Re-export for convenience
from quant_trading.memory.memory_db import MemoryDatabase as MemoryBank


class ConsolidationEngine:
    """
    Orchestrates memory consolidation and forgetting across layers.

    Wraps a dict of MemoryBank instances (one per layer) and executes
    the full consolidation cycle: decay -> cleanup -> jump.

    This class encapsulates the "cognitive consolidation" process from
    FinMem, where memories self-evolve between layers based on importance
    signals. It also supports reflection synthesis via a user-provided
    synthesize_fn.

    Example:
        >>> engine = ConsolidationEngine(layers={"short": short_db, "mid": mid_db, ...})
        >>> removed = engine.step()  # returns dict of removed entry IDs per layer
        >>> # Or with reflection synthesis:
        >>> removed = engine.step(synthesize_fn=my_reflect_fn)
    """

    def __init__(
        self,
        layers: Dict[str, MemoryDatabase],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            layers:  Dict mapping layer name -> MemoryBank instance.
                     Expected keys: "perceptual", "short", "mid", "long", "reflection"
            logger:  Optional logger for consolidation events.
        """
        self._layers = layers
        self._logger = logger or logging.getLogger("ConsolidationEngine")

    def step(
        self,
        synthesize_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[int]]:
        """
        Execute one full consolidation cycle across all layers.

        Phase 1: Decay + cleanup for every layer (via MemoryBank.step())
        Phase 2: Jump (promotion/demotion) between adjacent layers
        Phase 3: Optional reflection synthesis

        Args:
            synthesize_fn: Optional callback for reflection synthesis.
                           Called as: synthesize_fn(symbol, high_importance_entries)
                           Should return a list of reflection entry dicts
                           (with text, importance_score, recency_score, etc.).
                           If provided, high-importance long-term entries will
                           be passed to this function and results stored in
                           the reflection layer.

        Returns:
            Dict mapping layer name -> list of removed entry IDs.
        """
        removed: Dict[str, List[int]] = {}

        # Phase 1: Decay + cleanup for all layers
        jump_dicts: Dict[str, Tuple] = {}
        for layer in ["perceptual", "short", "mid", "long", "reflection"]:
            if layer in self._layers:
                removed[layer] = self._layers[layer].step()
                jump_dicts[layer] = self._layers[layer].prepare_jump()

        # Phase 2: Jump (bidirectional between adjacent layers)

        # Perceptual -> Short
        if "perceptual" in self._layers and "short" in self._layers:
            jump_up, _, _ = jump_dicts.get("perceptual", ({}, {}, []))
            self._layers["short"].accept_jump(jump_up, direction="up")

        # Short <-> Mid
        if "short" in self._layers and "mid" in self._layers:
            jump_up, jump_down, _ = jump_dicts.get("short", ({}, {}, []))
            self._layers["mid"].accept_jump(jump_up, direction="up")
            self._layers["mid"].accept_jump(jump_down, direction="down")

        # Mid <-> Long
        if "mid" in self._layers and "long" in self._layers:
            jump_up, jump_down, _ = jump_dicts.get("mid", ({}, {}, []))
            self._layers["long"].accept_jump(jump_up, direction="up")
            self._layers["short"].accept_jump(jump_down, direction="down")

        # Long <-> Reflection
        if "long" in self._layers and "reflection" in self._layers:
            jump_up, jump_down, _ = jump_dicts.get("long", ({}, {}, []))
            self._layers["reflection"].accept_jump(jump_up, direction="up")
            self._layers["long"].accept_jump(jump_down, direction="down")

        # Phase 3: Reflection synthesis
        if synthesize_fn is not None and "reflection" in self._layers:
            self._synthesize(synthesize_fn)

        return removed

    def _synthesize(
        self,
        synthesize_fn: Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
    ) -> None:
        """
        Distill high-importance long-term memories into reflection entries.

        For each symbol with high-importance entries in long-term memory,
        call synthesize_fn to produce reflection entries and add them
        to the reflection layer.

        Args:
            synthesize_fn: fn(symbol, long_term_entries) -> List[reflection_entries]
        """
        if "long" not in self._layers or "reflection" not in self._layers:
            return

        long_db = self._layers["long"]
        reflection_db = self._layers["reflection"]

        for symbol in long_db.universe:
            entries = long_db.universe[symbol]["score_memory"]
            if not entries:
                continue

            # Select top entries by compound_score for synthesis
            top_entries = sorted(
                entries, key=lambda x: x["compound_score"], reverse=True
            )[:5]  # top 5 most important long-term memories

            try:
                reflection_entries = synthesize_fn(symbol, top_entries)
                for entry in reflection_entries:
                    reflection_db.add_memory(
                        symbol=symbol,
                        text=entry.get("text", ""),
                        timestamp=entry.get("date"),
                    )
                    # Boost importance for synthesized reflections
                    # (done via add_memory's layer-appropriate initialization)
                self._logger.debug(
                    f"Synthesized {len(reflection_entries)} reflection entries for {symbol}"
                )
            except Exception as e:
                self._logger.warning(f"Synthesis failed for {symbol}: {e}")

    def consolidate_symbol(
        self,
        symbol: str,
        synthesize_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[int]]:
        """
        Run consolidation for a single symbol across all layers.

        Useful for targeted consolidation without advancing the full memory age.

        Args:
            symbol:         Asset ticker to consolidate.
            synthesize_fn:  Optional synthesis callback.

        Returns:
            Dict of removed entry IDs per layer.
        """
        removed: Dict[str, List[int]] = {}

        for layer_name, layer_db in self._layers.items():
            if symbol not in layer_db.universe:
                continue
            removed[layer_name] = layer_db.step()
            jump_up, jump_down, _ = layer_db.prepare_jump()
            layer_db.accept_jump(jump_up, direction="up")
            layer_db.accept_jump(jump_down, direction="down")

        if synthesize_fn is not None:
            self._synthesize_for_symbol(symbol, synthesize_fn)

        return removed

    def _synthesize_for_symbol(
        self,
        symbol: str,
        synthesize_fn: Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
    ) -> None:
        """Run synthesis for a single symbol."""
        if "long" not in self._layers or "reflection" not in self._layers:
            return
        if symbol not in self._layers["long"].universe:
            return

        long_db = self._layers["long"]
        reflection_db = self._layers["reflection"]
        entries = long_db.universe[symbol]["score_memory"]

        top_entries = sorted(
            entries, key=lambda x: x["compound_score"], reverse=True
        )[:5]

        try:
            reflection_entries = synthesize_fn(symbol, top_entries)
            for entry in reflection_entries:
                reflection_db.add_memory(
                    symbol=symbol,
                    text=entry.get("text", ""),
                    timestamp=entry.get("date"),
                )
        except Exception as e:
            self._logger.warning(f"Synthesis failed for {symbol}: {e}")


class MemoryConsolidator:
    """
    Standalone consolidation utilities for fine-grained control.

    Provides static methods and a stateful interface for:
        - Evaluating entries for promotion/demotion
        - Computing consolidation decisions
        - Managing per-layer forgetting thresholds

    Example:
        >>> mc = MemoryConsolidator()
        >>> decisions = mc.evaluate_jump(entries, upper=70.0, lower=20.0)
        >>> print(decisions)  # {"promote": [...], "demote": [...], "retain": [...]}
    """

    def __init__(
        self,
        jump_threshold_upper: float = 70.0,
        jump_threshold_lower: float = 20.0,
        cleanup_recency_threshold: float = 0.01,
        cleanup_importance_threshold: float = 5.0,
    ):
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.cleanup_recency_threshold = cleanup_recency_threshold
        self.cleanup_importance_threshold = cleanup_importance_threshold

    def evaluate_jump(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify entries into promote / demote / retain based on importance thresholds.

        Args:
            entries: List of memory entry dicts with importance_score field.

        Returns:
            Dict with keys "promote", "demote", "retain" mapping to entry lists.
        """
        promote: List[Dict[str, Any]] = []
        demote: List[Dict[str, Any]] = []
        retain: List[Dict[str, Any]] = []

        for entry in entries:
            imp = entry.get("importance_score", 0)
            if imp >= self.jump_threshold_upper:
                promote.append(copy.deepcopy(entry))
            elif imp < self.jump_threshold_lower:
                demote.append(copy.deepcopy(entry))
            else:
                retain.append(entry)

        return {"promote": promote, "demote": demote, "retain": retain}

    def should_cleanup(self, entry: Dict[str, Any]) -> bool:
        """
        Determine if an entry should be forgotten (cleaned up).

        Returns True if either:
            - recency_score < cleanup_recency_threshold
            - importance_score < cleanup_importance_threshold
        """
        return (
            entry.get("recency_score", 0) < self.cleanup_recency_threshold
            or entry.get("importance_score", 0) < self.cleanup_importance_threshold
        )

    def filter_forgettable(
        self, entries: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split entries into memorable vs. forgettable.

        Returns:
            (retain, forget) lists
        """
        retain, forget = [], []
        for entry in entries:
            if self.should_cleanup(entry):
                forget.append(entry)
            else:
                retain.append(entry)
        return retain, forget
