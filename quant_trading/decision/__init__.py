"""
Decision Module - Adversarial Multi-Agent Decision Framework
============================================================

Absorbed from LLM-TradeBot project with adaptations:
- Multi-agent weighted voting/aggregation
- Bull/Bear adversarial debate pattern
- Multi-timeframe alignment
- ReflectionAgent for learning from trade history
- Four-layer strategy filter pipeline

Architecture:
- DecisionCore: Central decision aggregation with weighted voting
- BullAgent/BearAgent: Adversarial perspective generation
- ReflectionAgent: Trade pattern learning from history
- MultiPeriodAgent: Multi-timeframe signal alignment
- FourLayerStrategyFilter: Trend -> AI Filter -> Setup -> Trigger pipeline
"""

from .decision_core import DecisionCore, SignalWeight, VoteResult, OvertradingGuard
from .reflection_agent import ReflectionAgent, ReflectionResult
from .bull_bear_agents import BullAgent, BearAgent
from .multi_period_agent import MultiPeriodAgent
from .layered_filter import FourLayerStrategyFilter

__all__ = [
    # Decision Core
    "DecisionCore",
    "SignalWeight",
    "VoteResult",
    "OvertradingGuard",
    # Agents
    "ReflectionAgent",
    "ReflectionResult",
    "BullAgent",
    "BearAgent",
    "MultiPeriodAgent",
    "FourLayerStrategyFilter",
]

__version__ = "1.0.0"
__author__ = "LLM-TradeBot (adapted for 量化之神)"
