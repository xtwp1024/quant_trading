"""
FinMem-inspired Layered Memory System for Quantitative Trading Agents.

Architecture:
    - Perceptual layer: raw market stimuli (ticks, news, events)
    - Short-term layer: recent memories with high recency weight
    - Long-term layer: enduring knowledge with high importance weight
    - Reflection layer: synthesized insights from accumulated experience

Key mechanisms:
    - Importance scoring: initializes based on layer type, updates via access counter
    - Recency scoring: exponential decay over time (delta-based)
    - Compound scoring: combines importance + recency for prioritization
    - Memory jump: self-evolution between layers (short->mid->long and reverse)
    - Cognitive span tuning: adjustable retention beyond human perceptual limits
"""

from quant_trading.memory.layered_memory import LayeredMemory
from quant_trading.memory.memory_db import MemoryDatabase
from quant_trading.memory.importance_scorer import (
    ImportanceScorer,
    ImportanceScorerByLayer,
)
from quant_trading.memory.recency_scorer import RecencyScorer
from quant_trading.memory.compound_scorer import CompoundScorer
from quant_trading.memory.exponential_decay import ExponentialDecay

# FinMem absorption: layered architecture with reflection layer
from quant_trading.memory.finmem_layer import FinMemLayer
from quant_trading.memory.finmem_consolidation import (
    ConsolidationEngine,
    MemoryConsolidator,
    MemoryBank,
)

__all__ = [
    # Core layered memory
    "LayeredMemory",
    "MemoryDatabase",
    # FinMem enhanced layered memory (with reflection layer)
    "FinMemLayer",
    # Scoring components
    "ImportanceScorer",
    "ImportanceScorerByLayer",
    "RecencyScorer",
    "CompoundScorer",
    "ExponentialDecay",
    # Consolidation
    "ConsolidationEngine",
    "MemoryConsolidator",
    "MemoryBank",
]
