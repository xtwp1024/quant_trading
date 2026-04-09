"""Market Regime Knowledge Graph — encodes market structure, regime transitions,
and regime-aware strategy selection.

Provides:
- MarketRegime              : Enum of market regime states
- RegimeTransition          : Transition probabilities between regimes
- MarketKnowledgeGraph      : Graph of market states, indicators, and relationships
- RegimeDetector            : Infers current regime from indicator state vectors
- StrategySelector          : Selects strategies based on detected regime
- RegimeHistory             : Tracks historical regime sequence

Usage
-----
```python
from quant_trading.knowledge.regime_graph import (
    MarketRegime, MarketKnowledgeGraph, RegimeDetector, StrategySelector
)

# Build knowledge graph
kg = MarketKnowledgeGraph()
kg.add_indicator("RSI", 0, 100)
kg.add_indicator("ATR_percent", 0, 10)
kg.add_state("bull_low_vol", [("RSI", 50, 70), ("ATR_percent", 0.5, 2)])
kg.add_state("bear_high_vol", [("RSI", 0, 40), ("ATR_percent", 3, 10)])
kg.add_transition("bull_low_vol", "bear_high_vol", prob=0.1)

# Detect current regime
detector = RegimeDetector(kg)
regime = detector.detect({"RSI": 35, "ATR_percent": 4.5})
print(f"Current regime: {regime}")

# Select strategies for this regime
selector = StrategySelector()
selector.assign("momentum", [MarketRegime.BULL_TRENDING])
selector.assign("mean_reversion", [MarketRegime.BEAR_RALLY])
print(selector.get_strategies(regime))
```
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger("MarketRegimeGraph")


# ---------------------------------------------------------------------------
# Market Regimes
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Market regime states.

    Each regime is defined by:
    - Trend direction: BULL, BEAR, SIDEWAYS
    - Volatility level: LOW, MEDIUM, HIGH
    - Combined modifiers: TRENDING, REVERSAL, BREAKOUT
    """
    # Base trend regimes
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"

    # Trend + volatility
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"

    # Special regimes
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    REGIME_UNCERTAIN = "uncertain"

    # Composite regimes
    BULL_BREAKOUT = "bull_breakout"
    BEAR_BREAKOUT = "bear_breakout"
    SIDEWAYS_SQUEEZE = "sideways_squeeze"

    @property
    def trend(self) -> str:
        """Return the trend component of the regime."""
        name = self.value.lower()
        if "bull" in name:
            return "bull"
        if "bear" in name:
            return "bear"
        return "sideways"

    @property
    def volatility(self) -> str:
        """Return the volatility component of the regime."""
        name = self.value.lower()
        if "high_vol" in name:
            return "high"
        if "low_vol" in name:
            return "low"
        return "medium"

    def is_bullish(self) -> bool:
        return self.trend == "bull"

    def is_bearish(self) -> bool:
        return self.trend == "bear"

    def is_high_vol(self) -> bool:
        return self.volatility == "high"

    def recommended_strategies(self) -> List[str]:
        """Return strategy types recommended for this regime."""
        strategies = {
            MarketRegime.BULL: ["momentum", "trend_following"],
            MarketRegime.BEAR: ["mean_reversion", "shorting"],
            MarketRegime.SIDEWAYS: ["range_bound", "mean_reversion"],
            MarketRegime.BULL_LOW_VOL: ["buy_and_hold", "momentum"],
            MarketRegime.BULL_HIGH_VOL: ["breakout", "momentum"],
            MarketRegime.BEAR_LOW_VOL: ["value", "mean_reversion"],
            MarketRegime.BEAR_HIGH_VOL: ["shorting", "volatility_strat"],
            MarketRegime.SIDEWAYS_LOW_VOL: ["grid_trading", "mean_reversion"],
            MarketRegime.SIDEWAYS_HIGH_VOL: ["volatility_strat", "breakout"],
            MarketRegime.BULL_TRENDING: ["momentum", "trend_following"],
            MarketRegime.BEAR_TRENDING: ["momentum", "shorting"],
            MarketRegime.HIGH_VOLATILITY: ["volatility_strat", "OptionsStrategy"],
            MarketRegime.LOW_VOLATILITY: ["buy_and_hold", "accumulation"],
            MarketRegime.BULL_BREAKOUT: ["breakout", "momentum"],
            MarketRegime.BEAR_BREAKOUT: ["mean_reversion", "shorting"],
            MarketRegime.SIDEWAYS_SQUEEZE: ["grid_trading", "straddle"],
            MarketRegime.REGIME_UNCERTAIN: ["wait", "hedged"],
        }
        return strategies.get(self, ["wait"])


# ---------------------------------------------------------------------------
# Regime Transition
# ---------------------------------------------------------------------------

@dataclass
class RegimeTransition:
    """A known transition between two market regimes."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    probability: float  # 0.0 - 1.0
    trigger_conditions: List[str] = field(default_factory=list)
    observed_count: int = 0

    def update_probability(self, new_prob: float) -> None:
        """Update transition probability from observed data."""
        # Running average
        self.probability = (
            self.probability * 0.9 + new_prob * 0.1
        )
        self.observed_count += 1


# ---------------------------------------------------------------------------
# Market Knowledge Graph
# ---------------------------------------------------------------------------

@dataclass
class IndicatorRange:
    """Defines a valid range for an indicator in a given regime."""
    indicator: str
    min_value: float
    max_value: float
    typical_value: Optional[float] = None


@dataclass
class MarketState:
    """A node in the knowledge graph representing a market regime."""
    regime: MarketRegime
    indicator_ranges: List[IndicatorRange] = field(default_factory=list)
    description: str = ""
    entry_indicators: List[str] = field(default_factory=list)  # Leading indicators
    confirmed_indicators: List[str] = field(default_factory=list)  # Lagging confirm


class MarketKnowledgeGraph:
    """Graph of market regimes and their relationships.

    Nodes are MarketState objects (regimes).
    Edges are RegimeTransition objects (known transitions).
    """

    def __init__(self):
        self._states: Dict[MarketRegime, MarketState] = {}
        self._transitions: Dict[Tuple[MarketRegime, MarketRegime], RegimeTransition] = {}
        self._indicators: Set[str] = set()
        self._indicator_ranges: Dict[str, Tuple[float, float]] = {}  # global min/max

    def add_indicator(
        self,
        name: str,
        global_min: float,
        global_max: float,
    ) -> None:
        """Register an indicator with its global range."""
        self._indicators.add(name)
        self._indicator_ranges[name] = (global_min, global_max)

    def add_state(
        self,
        regime: MarketRegime,
        indicator_ranges: List[Tuple[str, float, float]],
        description: str = "",
        entry_indicators: Optional[List[str]] = None,
        confirmed_indicators: Optional[List[str]] = None,
    ) -> None:
        """Add a market state (node) to the graph.

        Args:
            regime: The MarketRegime enum value
            indicator_ranges: List of (indicator_name, min, max) tuples
            description: Human-readable description
            entry_indicators: Indicators that lead this regime
            confirmed_indicators: Indicators that confirm this regime
        """
        ranges = [
            IndicatorRange(ind, mn, mx)
            for ind, mn, mx in indicator_ranges
        ]
        self._states[regime] = MarketState(
            regime=regime,
            indicator_ranges=ranges,
            description=description,
            entry_indicators=entry_indicators or [],
            confirmed_indicators=confirmed_indicators or [],
        )

    def add_transition(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        probability: float,
        trigger_conditions: Optional[List[str]] = None,
    ) -> None:
        """Add a known transition (edge) to the graph."""
        self._transitions[(from_regime, to_regime)] = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            probability=probability,
            trigger_conditions=trigger_conditions or [],
        )

    def get_state(self, regime: MarketRegime) -> Optional[MarketState]:
        return self._states.get(regime)

    def get_transition(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
    ) -> Optional[RegimeTransition]:
        return self._transitions.get((from_regime, to_regime))

    def get_next_regimes(
        self,
        current_regime: MarketRegime,
        min_probability: float = 0.0,
    ) -> List[Tuple[MarketRegime, float]]:
        """Get possible next regimes from current regime.

        Returns:
            List of (next_regime, probability) sorted by probability descending.
        """
        candidates = []
        for (frm, to), trans in self._transitions.items():
            if frm == current_regime and trans.probability >= min_probability:
                candidates.append((to, trans.probability))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def get_all_regimes(self) -> List[MarketRegime]:
        return list(self._states.keys())


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detects the current market regime from indicator values."""

    def __init__(self, graph: MarketKnowledgeGraph):
        self.graph = graph

    def detect(self, indicator_values: Dict[str, float]) -> MarketRegime:
        """Detect the current market regime from indicator readings.

        Uses the knowledge graph to find the best-matching regime.

        Args:
            indicator_values: Dict of {indicator_name: value}

        Returns:
            MarketRegime enum value
        """
        best_regime: Optional[MarketRegime] = None
        best_score = -1.0

        for regime, state in self.graph._states.items():
            score = self._score_regime(state, indicator_values)
            if score > best_score:
                best_score = score
                best_regime = regime

        if best_regime is None:
            return MarketRegime.REGIME_UNCERTAIN

        # If best score is too low, return uncertain
        if best_score < 0.3:
            return MarketRegime.REGIME_UNCERTAIN

        return best_regime

    def _score_regime(
        self,
        state: MarketState,
        values: Dict[str, float],
    ) -> float:
        """Score how well indicator values match a regime.

        Returns:
            Score 0.0 to 1.0
        """
        if not state.indicator_ranges:
            return 0.5  # Default

        scores = []
        for ind_range in state.indicator_ranges:
            if ind_range.indicator not in values:
                scores.append(0.5)  # Unknown indicator
                continue

            val = values[ind_range.indicator]
            mn, mx = ind_range.min_value, ind_range.max_value

            if mn <= val <= mx:
                # Within range — score based on proximity to typical
                if ind_range.typical_value is not None:
                    midpoint = (mn + mx) / 2
                    dist = abs(val - ind_range.typical_value)
                    range_size = mx - mn
                    score = 1.0 - min(dist / (range_size / 2), 1.0)
                else:
                    score = 1.0
            else:
                # Outside range — penalty based on distance
                if val < mn:
                    dist = mn - val
                else:
                    dist = val - mx
                range_size = mx - mn
                score = max(0.0, 1.0 - (dist / range_size))

            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5

    def detect_with_confidence(
        self, indicator_values: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """Detect regime with confidence score.

        Returns:
            Tuple of (regime, confidence)
        """
        best_score = -1.0
        best_regime = MarketRegime.REGIME_UNCERTAIN

        for regime, state in self.graph._states.items():
            score = self._score_regime(state, indicator_values)
            if score > best_score:
                best_score = score
                best_regime = regime

        return best_regime, max(0.0, best_score)


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

@dataclass
class RegimeStrategyAssignment:
    """Assignment of a strategy to specific regimes."""
    strategy_name: str
    regimes: List[MarketRegime]
    performance_history: Dict[MarketRegime, float] = field(default_factory=dict)
    weight: float = 1.0


class StrategySelector:
    """Selects and weights trading strategies based on detected regime.

    Supports:
    - Regime-specific strategy assignment
    - Performance-based reweighting
    - Fallback strategies for uncertain regimes
    """

    def __init__(self, default_strategies: Optional[List[str]] = None):
        """Initialize selector.

        Args:
            default_strategies: Strategies to use when regime is uncertain.
        """
        self._assignments: List[RegimeStrategyAssignment] = []
        self._fallback_strategies = default_strategies or ["wait"]
        self._regime_history: List[Tuple[int, MarketRegime]] = []  # (timestamp, regime)

    def assign(
        self,
        strategy_name: str,
        regimes: List[MarketRegime],
        weight: float = 1.0,
    ) -> None:
        """Assign a strategy to specific regimes.

        Args:
            strategy_name: Name of the strategy
            regimes: List of regimes this strategy works well in
            weight: Base weight for this strategy
        """
        self._assignments.append(
            RegimeStrategyAssignment(
                strategy_name=strategy_name,
                regimes=regimes,
                weight=weight,
            )
        )

    def update_performance(
        self,
        strategy_name: str,
        regime: MarketRegime,
        performance: float,
    ) -> None:
        """Update performance history for a strategy in a given regime."""
        for asn in self._assignments:
            if asn.strategy_name == strategy_name:
                asn.performance_history[regime] = performance

    def get_strategies(
        self,
        regime: MarketRegime,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Get top-k strategies for the given regime.

        Returns:
            List of (strategy_name, weight) sorted by weight descending.
        """
        candidates = []

        for asn in self._assignments:
            if regime in asn.regimes:
                # Combine base weight with performance history
                perf = asn.performance_history.get(regime, None)
                if perf is not None:
                    # Adjust weight by relative performance
                    adjusted_weight = asn.weight * (1.0 + perf)
                else:
                    adjusted_weight = asn.weight
                candidates.append((asn.strategy_name, adjusted_weight))

        if not candidates:
            # Use fallback strategies
            return [(s, 1.0) for s in self._fallback_strategies[:top_k]]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def record_regime(self, timestamp: int, regime: MarketRegime) -> None:
        """Record detected regime at a given timestamp."""
        self._regime_history.append((timestamp, regime))

    def get_regime_sequence(self) -> List[MarketRegime]:
        """Get the sequence of detected regimes."""
        return [r for _, r in self._regime_history]


# ---------------------------------------------------------------------------
# Regime History
# ---------------------------------------------------------------------------

@dataclass
class RegimeEpisode:
    """A continuous period in a particular regime."""
    regime: MarketRegime
    start_time: int  # Unix ms
    end_time: Optional[int] = None  # None = ongoing
    duration_bars: int = 0
    entry_indicators: Dict[str, float] = field(default_factory=dict)
    exit_indicators: Dict[str, float] = field(default_factory=dict)
    returns: List[float] = field(default_factory=list)

    @property
    def is_ongoing(self) -> bool:
        return self.end_time is None


class RegimeHistory:
    """Tracks historical regime episodes for analysis.

    Useful for:
    - Learning transition probabilities from data
    - Identifying regime persistence patterns
    - Backtesting regime-based strategies
    """

    def __init__(self):
        self._episodes: List[RegimeEpisode] = []
        self._current: Optional[RegimeEpisode] = None

    def enter_regime(
        self,
        regime: MarketRegime,
        timestamp: int,
        indicators: Optional[Dict[str, float]] = None,
    ) -> RegimeEpisode:
        """Record entry into a new regime."""
        # Close current episode
        if self._current is not None and self._current.is_ongoing:
            self._current.end_time = timestamp

        self._current = RegimeEpisode(
            regime=regime,
            start_time=timestamp,
            entry_indicators=indicators or {},
        )
        self._episodes.append(self._current)
        return self._current

    def update_indicators(
        self,
        timestamp: int,
        indicators: Dict[str, float],
    ) -> None:
        """Update indicators for the current (ongoing) regime."""
        if self._current and self._current.is_ongoing:
            self._current.exit_indicators = indicators

    def exit_regime(self, timestamp: int) -> None:
        """Mark current regime as exited."""
        if self._current and self._current.is_ongoing:
            self._current.end_time = timestamp

    def get_episodes(self, regime: Optional[MarketRegime] = None) -> List[RegimeEpisode]:
        """Get all episodes, optionally filtered by regime."""
        if regime is None:
            return list(self._episodes)
        return [e for e in self._episodes if e.regime == regime]

    def get_avg_duration(self, regime: MarketRegime) -> float:
        """Average duration in bars for a regime."""
        eps = self.get_episodes(regime)
        if not eps:
            return 0.0
        durations = [e.duration_bars for e in eps if not e.is_ongoing]
        return sum(durations) / len(durations) if durations else 0.0

    def get_transition_probability(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
    ) -> float:
        """Compute empirical transition probability from history."""
        from_eps = [e for e in self._episodes if e.regime == from_regime]
        if not from_eps:
            return 0.0

        transitions = 0
        for ep in from_eps:
            ep_idx = self._episodes.index(ep)
            if ep_idx + 1 < len(self._episodes):
                next_ep = self._episodes[ep_idx + 1]
                if next_ep.regime == to_regime:
                    transitions += 1

        return transitions / len(from_eps)

    def get_persistence(self, regime: MarketRegime) -> float:
        """Probability that a regime continues (doesn't transition)."""
        count = sum(1 for _ in self.get_episodes(regime))
        if count < 2:
            return 1.0
        same = sum(
            1 for i in range(len(self._episodes) - 1)
            if self._episodes[i].regime == regime
            and self._episodes[i + 1].regime == regime
        )
        return same / (count - 1)
