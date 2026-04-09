"""
MemoryDatabase: stores, retrieves, and manages memory entries for a single layer.

Each MemoryDatabase instance is scoped to one memory layer (e.g., short-term,
mid-term, long-term). It manages:
    - Entry storage keyed by symbol (ticker/asset)
    - Per-entry scoring: importance, recency, delta, compound score
    - Access tracking and feedback-based importance updates
    - Exponential decay over time
    - Cleanup of low-scoring entries
    - Memory "jumps" between layers (promotion/demotion)

The database is NOT tied to any LLM or embedding model. It maintains entries
as plain Python dicts with numeric scores, making it suitable for any
quantitative trading agent.

Score computation per entry:
    {
        "id":           int,
        "symbol":       str,
        "text":         str,           # memory content
        "date":         datetime,
        "importance_score": float,    # [0, ~100]
        "recency_score":   float,     # [0, 1]
        "delta":           int,       # time steps since last access
        "compound_score":   float,    # recency + importance/100
        "access_counter":   int,      # number of times accessed
    }
"""

import copy
import logging
from datetime import datetime, date
from typing import List, Union, Dict, Any, Optional, Callable, Tuple
from sortedcontainers import SortedList

from quant_trading.memory.importance_scorer import (
    ImportanceScorer,
    ImportanceScorerByLayer,
)
from quant_trading.memory.recency_scorer import RecencyScorer
from quant_trading.memory.compound_scorer import CompoundScorer
from quant_trading.memory.exponential_decay import ExponentialDecay


class MemoryDatabase:
    """
    Single-layer in-memory database for trading agent memories.

    Supports:
        - Multi-symbol storage (each symbol has independent memory)
        - Score-based entry prioritization
        - Feedback-driven importance updates
        - Exponential decay + cleanup
        - Layer jump (promotion/demotion) for self-evolving memory

    Note:
        This replaces the FAISS-backed MemoryDB from FinMem's LLM pipeline.
        Vector/semantic search is not included; entries are ranked by
        compound_score (importance + recency). For semantic search, inject
        a custom query_fn at construction time.
    """

    def __init__(
        self,
        db_name: str,
        layer: str,
        id_generator: Callable[[], int],
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        compound_scorer: CompoundScorer,
        decay: ExponentialDecay,
        importance_scorer: Optional[ImportanceScorerByLayer] = None,
        recency_scorer: Optional[RecencyScorer] = None,
        cleanup_recency_threshold: float = 0.01,
        cleanup_importance_threshold: float = 5.0,
        logger: Optional[logging.Logger] = None,
        # Optional semantic search override: fn(symbol, query_text, top_k) -> List[Dict]
        semantic_query_fn: Optional[Callable[[str, str, int], List[Dict]]] = None,
    ):
        """
        Args:
            db_name:                 Human-readable name for this DB (e.g., "agent_short").
            layer:                   Layer name: "short", "mid", "long", "perceptual", "reflection".
            id_generator:            Callable returning a unique int ID on each call.
            jump_threshold_upper:     Min importance_score to qualify for promotion (jump up).
            jump_threshold_lower:     Max importance_score before demotion (jump down).
            compound_scorer:         CompoundScorer instance.
            decay:                   ExponentialDecay instance.
            importance_scorer:       ImportanceScorerByLayer (uses layer if None).
            recency_scorer:          RecencyScorer (defaults to RecencyScorer()).
            cleanup_recency_threshold:    Remove entries with recency_score < this.
            cleanup_importance_threshold:  Remove entries with importance_score < this.
            logger:                  Optional logger for memory events.
            semantic_query_fn:       Optional override for text-based memory query.
        """
        self.db_name = db_name
        self.layer = layer
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.compound_scorer = compound_scorer
        self.decay = decay
        self.importance_scorer = (
            importance_scorer or ImportanceScorerByLayer(layer)
        )
        self.recency_scorer = recency_scorer or RecencyScorer()
        self.cleanup_recency_threshold = cleanup_recency_threshold
        self.cleanup_importance_threshold = cleanup_importance_threshold
        self.logger = logger or logging.getLogger(db_name)

        # Per-symbol storage: symbol -> {"score_memory": SortedList, ...}
        self.universe: Dict[str, Dict[str, Any]] = {}

        # Optional semantic search override
        self.semantic_query_fn = semantic_query_fn

    # -------------------------------------------------------------------------
    # Symbol management
    # -------------------------------------------------------------------------

    def add_symbol(self, symbol: str) -> None:
        """Create an empty memory store for a new symbol."""
        if symbol in self.universe:
            return
        self.universe[symbol] = {
            "score_memory": SortedList(
                key=lambda x: x["compound_score"]
            ),
        }
        self.logger.info(f"[{self.db_name}] Added symbol: {symbol}")

    # -------------------------------------------------------------------------
    # Memory operations
    # -------------------------------------------------------------------------

    def add_memory(
        self,
        symbol: str,
        text: Union[str, List[str]],
        timestamp: Optional[Union[datetime, date]] = None,
    ) -> List[int]:
        """
        Add one or more memory entries for a symbol.

        Args:
            symbol:    Asset ticker (e.g., "AAPL", "BTC-USD").
            text:       Memory content string, or list of strings.
            timestamp:  Event time (defaults to now).

        Returns:
            List of assigned entry IDs.
        """
        if symbol not in self.universe:
            self.add_symbol(symbol)

        if isinstance(text, str):
            text = [text]

        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, date) and not isinstance(timestamp, datetime):
            timestamp = datetime.combine(timestamp, datetime.min.time())

        ids = []
        for content in text:
            entry_id = self.id_generator()
            importance = self.importance_scorer.initialize()
            recency = self.recency_scorer.initialize()
            compound = self.compound_scorer.recency_and_importance_score(
                recency_score=recency,
                importance_score=importance,
            )

            entry = {
                "id": entry_id,
                "symbol": symbol,
                "text": content,
                "date": timestamp,
                "importance_score": importance,
                "recency_score": recency,
                "delta": 0,
                "compound_score": compound,
                "access_counter": 0,
            }

            self.universe[symbol]["score_memory"].add(entry)
            ids.append(entry_id)

            self.logger.debug(
                f"[{self.db_name}] Added memory id={entry_id} symbol={symbol} "
                f"importance={importance:.2f} recency={recency:.2f}"
            )

        return ids

    def query(
        self,
        symbol: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k highest-compound-score memories for a symbol.

        If self.semantic_query_fn is provided, it is called with
        (symbol, query_text, top_k) and should return ranked entry dicts.
        Otherwise, returns top-k by compound_score alone.

        Args:
            symbol:     Asset ticker.
            top_k:      Maximum number of entries to return.
            query_text: Optional text to pass to semantic_query_fn.

        Returns:
            List of memory entry dicts (sorted by compound_score descending).
        """
        if (
            symbol not in self.universe
            or len(self.universe[symbol]["score_memory"]) == 0
            or top_k == 0
        ):
            return []

        # Use semantic search override if provided
        if self.semantic_query_fn is not None and query_text is not None:
            return self.semantic_query_fn(symbol, query_text, top_k)

        # Default: return top-k by compound_score
        all_entries = list(self.universe[symbol]["score_memory"])
        all_entries.sort(key=lambda x: x["compound_score"], reverse=True)
        return all_entries[:top_k]

    def query_by_importance(
        self, symbol: str, min_importance: float, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return memories with importance_score >= min_importance."""
        if symbol not in self.universe:
            return []
        entries = [
            e for e in self.universe[symbol]["score_memory"]
            if e["importance_score"] >= min_importance
        ]
        entries.sort(key=lambda x: x["importance_score"], reverse=True)
        if top_k is not None:
            entries = entries[:top_k]
        return entries

    def query_by_recency(
        self, symbol: str, min_recency: float, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return memories with recency_score >= min_recency."""
        if symbol not in self.universe:
            return []
        entries = [
            e for e in self.universe[symbol]["score_memory"]
            if e["recency_score"] >= min_recency
        ]
        entries.sort(key=lambda x: x["recency_score"], reverse=True)
        if top_k is not None:
            entries = entries[:top_k]
        return entries

    def get_entry(self, symbol: str, entry_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single entry by ID."""
        if symbol not in self.universe:
            return None
        for entry in self.universe[symbol]["score_memory"]:
            if entry["id"] == entry_id:
                return entry
        return None

    def update_access(
        self, symbol: str, entry_ids: List[int], feedback: int = 1
    ) -> List[int]:
        """
        Update access counters and recalculate importance for accessed entries.

        FinMem formula: importance += access_counter * 5

        Args:
            symbol:    Asset ticker.
            entry_ids: List of entry IDs that were accessed.
            feedback:  1 for positive feedback, -1 for negative.

        Returns:
            List of successfully updated entry IDs.
        """
        if symbol not in self.universe:
            return []

        success_ids = []
        score_memory = self.universe[symbol]["score_memory"]

        for entry_id in entry_ids:
            for entry in score_memory:
                if entry["id"] == entry_id:
                    entry["access_counter"] += feedback
                    entry["importance_score"] = self.importance_scorer.update(
                        importance_score=entry["importance_score"],
                        access_counter=entry["access_counter"],
                    )
                    entry["compound_score"] = (
                        self.compound_scorer.recency_and_importance_score(
                            recency_score=entry["recency_score"],
                            importance_score=entry["importance_score"],
                        )
                    )
                    success_ids.append(entry_id)
                    break

        return success_ids

    # -------------------------------------------------------------------------
    # Decay and cleanup
    # -------------------------------------------------------------------------

    def _decay(self) -> None:
        """
        Apply one step of exponential decay to all entries across all symbols.

        Updates recency_score, importance_score, delta, and compound_score.
        """
        for symbol in self.universe:
            score_memory = self.universe[symbol]["score_memory"]
            for entry in score_memory:
                recency, importance, delta = self.decay(
                    importance_score=entry["importance_score"],
                    delta=entry["delta"],
                )
                entry["recency_score"] = recency
                entry["importance_score"] = importance
                entry["delta"] = delta
                entry["compound_score"] = (
                    self.compound_scorer.recency_and_importance_score(
                        recency_score=recency,
                        importance_score=importance,
                    )
                )

    def _cleanup(self) -> List[int]:
        """
        Remove entries that have decayed below both recency and importance thresholds.

        Returns:
            List of removed entry IDs.
        """
        removed_ids: List[int] = []

        for symbol in self.universe:
            score_memory = self.universe[symbol]["score_memory"]
            to_remove = [
                entry["id"]
                for entry in score_memory
                if (
                    entry["recency_score"] < self.cleanup_recency_threshold
                    or entry["importance_score"] < self.cleanup_importance_threshold
                )
            ]

            if to_remove:
                new_memory = SortedList(
                    [e for e in score_memory if e["id"] not in to_remove],
                    key=lambda x: x["compound_score"],
                )
                self.universe[symbol]["score_memory"] = new_memory
                removed_ids.extend(to_remove)
                self.logger.debug(
                    f"[{self.db_name}] Cleanup removed {len(to_remove)} entries "
                    f"from {symbol}"
                )

        return removed_ids

    def step(self) -> List[int]:
        """
        Advance the memory layer by one time step.

        Applies decay then cleanup.

        Returns:
            List of removed entry IDs.
        """
        self._decay()
        return self._cleanup()

    # -------------------------------------------------------------------------
    # Memory jump (self-evolution between layers)
    # -------------------------------------------------------------------------

    def prepare_jump(
        self,
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], List[int]]:
        """
        Identify entries ready for promotion or demotion.

        Returns:
            (jump_up, jump_down, removed_ids)
            - jump_up:    Entries with importance >= jump_threshold_upper (promote)
            - jump_down:  Entries with importance < jump_threshold_lower (demote)
            - removed_ids: IDs removed from this layer's index
        """
        jump_up: Dict[str, List[Dict]] = {}
        jump_down: Dict[str, List[Dict]] = {}
        removed_ids: List[int] = []

        for symbol in self.universe:
            score_memory = self.universe[symbol]["score_memory"]
            up_entries: List[Dict] = []
            down_entries: List[Dict] = []
            surviving: List[Dict] = []

            for entry in score_memory:
                if entry["importance_score"] >= self.jump_threshold_upper:
                    up_entries.append(copy.deepcopy(entry))
                    removed_ids.append(entry["id"])
                elif entry["importance_score"] < self.jump_threshold_lower:
                    down_entries.append(copy.deepcopy(entry))
                    removed_ids.append(entry["id"])
                else:
                    surviving.append(entry)

            self.universe[symbol]["score_memory"] = SortedList(
                surviving, key=lambda x: x["compound_score"]
            )

            if up_entries:
                jump_up[symbol] = up_entries
            if down_entries:
                jump_down[symbol] = down_entries

        return jump_up, jump_down, removed_ids

    def accept_jump(
        self,
        jump_dict: Dict[str, List[Dict]],
        direction: str,
    ) -> None:
        """
        Absorb entries from another layer (promoted or demoted).

        Args:
            jump_dict:  {symbol: [entry_dicts]} from prepare_jump().
            direction:  "up" (promoted into this layer) or "down" (demoted).
        """
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")

        for symbol, entries in jump_dict.items():
            if symbol not in self.universe:
                self.add_symbol(symbol)

            for entry in entries:
                # Refresh recency on promotion (up); keep it on demotion (down)
                if direction == "up":
                    entry["recency_score"] = self.recency_scorer.reset()
                    entry["delta"] = 0

                # Recompute compound score in case delta changed
                entry["compound_score"] = (
                    self.compound_scorer.recency_and_importance_score(
                        recency_score=entry["recency_score"],
                        importance_score=entry["importance_score"],
                    )
                )

                self.universe[symbol]["score_memory"].add(entry)

            self.logger.debug(
                f"[{self.db_name}] Accepted {len(entries)} entries "
                f"direction={direction} symbol={symbol}"
            )

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a summary of memory usage per symbol."""
        return {
            symbol: {
                "count": len(self.universe[symbol]["score_memory"]),
                "avg_importance": (
                    sum(e["importance_score"] for e in self.universe[symbol]["score_memory"])
                    / len(self.universe[symbol]["score_memory"])
                    if self.universe[symbol]["score_memory"]
                    else 0.0
                ),
                "avg_recency": (
                    sum(e["recency_score"] for e in self.universe[symbol]["score_memory"])
                    / len(self.universe[symbol]["score_memory"])
                    if self.universe[symbol]["score_memory"]
                    else 0.0
                ),
            }
            for symbol in self.universe
        }

    def __len__(self) -> int:
        """Total number of entries across all symbols."""
        return sum(len(self.universe[s]["score_memory"]) for s in self.universe)

    def __repr__(self) -> str:
        return f"<MemoryDatabase {self.db_name} layer={self.layer} entries={len(self)}>"
