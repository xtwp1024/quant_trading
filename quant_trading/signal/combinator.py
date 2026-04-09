"""Signal combinator — fuses multiple signal streams into a consensus signal.

Provides several fusion strategies:
- WeightedAverage : Strength-weighted average, threshold-based firing
- Voting          : Majority vote across generators
- Ensemble        : All-or-nothing (all must agree)
- Priority        : First generator to fire wins
- Bayesian        : Probabilistic fusion assuming independent sources

Usage
-----
```python
from quant_trading.signal.combinator import WeightedAverageCombinator, VotingCombinator
from quant_trading.signal.generators import RSIGenerator, MACDGenerator

combo = WeightedAverageCombinator(threshold=0.6)
combo.add_generator(RSIGenerator(14), weight=0.5)
combo.add_generator(MACDGenerator(), weight=0.5)

signals = combo.generate(df)
```
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

class FusionStrategy(ABC):
    """Abstract base for signal fusion strategies."""

    @abstractmethod
    def combine(
        self,
        signal_groups: List[List[Signal]],
        weights: List[float],
        df: pd.DataFrame,
    ) -> List[Signal]:
        """Fuse multiple signal streams into a single stream.

        Parameters
        ----------
        signal_groups : List[List[Signal]]
            Each inner list is the signal stream from one generator.
        weights : List[float]
            Weight for each generator.
        df : pd.DataFrame
            OHLCV data (for price/timestamp lookups).

        Returns
        -------
        List[Signal]
            Fused signal stream.
        """
        ...


@dataclass
class WeightedAverageFusion(FusionStrategy):
    """Weighted average fusion with threshold.

    BUY  fires when weighted BUY strength  >= threshold
    SELL fires when weighted SELL strength >= threshold
    """
    threshold: float = 0.5
    min_interval_bars: int = 1  # Minimum bars between signals

    def combine(
        self,
        signal_groups: List[List[Signal]],
        weights: List[float],
        df: pd.DataFrame,
    ) -> List[Signal]:
        # Collect all unique timestamps
        timestamps = sorted(
            {s.timestamp for sigs in signal_groups for s in sigs}
        )

        result: List[Signal] = []
        last_ts: Optional[int] = None

        for ts in timestamps:
            buy_score = 0.0
            sell_score = 0.0
            price = 0.0
            symbol = "UNKNOWN"

            for sigs, w in zip(signal_groups, weights):
                for s in sigs:
                    if s.timestamp == ts:
                        if s.type == SignalType.BUY:
                            buy_score += w * s.strength
                        elif s.type == SignalType.SELL:
                            sell_score += w * s.strength
                        price = s.price
                        symbol = s.symbol

            total = buy_score + sell_score
            if total < 1e-10:
                continue

            buy_norm = buy_score / total
            sell_norm = sell_score / total

            # Determine signal direction
            if buy_norm >= self.threshold and buy_norm > sell_norm:
                sig_type = SignalType.BUY
                strength = float(buy_norm)
            elif sell_norm >= self.threshold and sell_norm > buy_norm:
                sig_type = SignalType.SELL
                strength = float(sell_norm)
            else:
                continue

            # Enforce min interval
            if last_ts is not None and ts - last_ts < self.min_interval_bars:
                continue

            result.append(
                Signal(
                    type=sig_type,
                    symbol=symbol,
                    timestamp=ts,
                    price=float(price),
                    strength=strength,
                    reason=f"WeightedFusion: buy={buy_norm:.2f} sell={sell_norm:.2f}",
                    metadata={"buy_score": float(buy_score), "sell_score": float(sell_score)},
                )
            )
            last_ts = ts

        return result


@dataclass
class VotingFusion(FusionStrategy):
    """Majority vote across generators.

    BUY  fires if more generators produce BUY than SELL
    SELL fires if more generators produce SELL than BUY
    """
    min_agreement: float = 0.5  # Fraction of generators that must agree

    def combine(
        self,
        signal_groups: List[List[Signal]],
        weights: List[float],
        df: pd.DataFrame,
    ) -> List[Signal]:
        timestamps = sorted(
            {s.timestamp for sigs in signal_groups for s in sigs}
        )
        n_generators = len(signal_groups)
        min_count = max(1, int(np.ceil(n_generators * self.min_agreement)))

        result: List[Signal] = []

        for ts in timestamps:
            buy_count = 0
            sell_count = 0
            total_strength = 0.0
            price = 0.0
            symbol = "UNKNOWN"

            for sigs in signal_groups:
                for s in sigs:
                    if s.timestamp == ts:
                        if s.type == SignalType.BUY:
                            buy_count += 1
                            total_strength += s.strength
                        elif s.type == SignalType.SELL:
                            sell_count += 1
                            total_strength += s.strength
                        price = s.price
                        symbol = s.symbol

            if buy_count > sell_count and buy_count >= min_count:
                result.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(total_strength / (buy_count + sell_count + 1e-10)),
                        reason=f"VotingFusion: BUY {buy_count}/{n_generators}",
                        metadata={"buy_count": buy_count, "sell_count": sell_count},
                    )
                )
            elif sell_count > buy_count and sell_count >= min_count:
                result.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(total_strength / (buy_count + sell_count + 1e-10)),
                        reason=f"VotingFusion: SELL {sell_count}/{n_generators}",
                        metadata={"buy_count": buy_count, "sell_count": sell_count},
                    )
                )

        return result


@dataclass
class EnsembleFusion(FusionStrategy):
    """All-or-nothing: all generators must agree for signal to fire.

    BUY  fires if ALL generators produce BUY at same timestamp
    SELL fires if ALL generators produce SELL at same timestamp
    """
    def combine(
        self,
        signal_groups: List[List[Signal]],
        weights: List[float],
        df: pd.DataFrame,
    ) -> List[Signal]:
        timestamps = sorted(
            {s.timestamp for sigs in signal_groups for s in sigs}
        )
        n = len(signal_groups)

        result: List[Signal] = []

        for ts in timestamps:
            types_at_ts: List[Optional[SignalType]] = []
            strengths: List[float] = []
            price = 0.0
            symbol = "UNKNOWN"

            for sigs in signal_groups:
                matched = [s for s in sigs if s.timestamp == ts]
                if matched:
                    types_at_ts.append(matched[0].type)
                    strengths.append(matched[0].strength)
                    price = matched[0].price
                    symbol = matched[0].symbol
                else:
                    types_at_ts.append(None)
                    strengths.append(0.0)

            if all(t == SignalType.BUY for t in types_at_ts if t is not None) and any(
                t == SignalType.BUY for t in types_at_ts
            ):
                result.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(np.mean(strengths)),
                        reason=f"EnsembleFusion: UNANIMOUS BUY",
                        metadata={"n_generators": n, "agreed": sum(1 for t in types_at_ts if t == SignalType.BUY)},
                    )
                )
            elif all(t == SignalType.SELL for t in types_at_ts if t is not None) and any(
                t == SignalType.SELL for t in types_at_ts
            ):
                result.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(np.mean(strengths)),
                        reason=f"EnsembleFusion: UNANIMOUS SELL",
                        metadata={"n_generators": n, "agreed": sum(1 for t in types_at_ts if t == SignalType.SELL)},
                    )
                )

        return result


@dataclass
class BayesianFusion(FusionStrategy):
    """Bayesian probabilistic fusion assuming independent signal sources.

    P(BUY|X) ∝ P(X|BUY) * P(BUY)
    Each generator's signal is treated as independent evidence.
    """
    prior_buy: float = 0.5
    prior_sell: float = 0.5
    threshold: float = 0.6

    def combine(
        self,
        signal_groups: List[List[Signal]],
        weights: List[float],
        df: pd.DataFrame,
    ) -> List[Signal]:
        timestamps = sorted(
            {s.timestamp for sigs in signal_groups for s in sigs}
        )

        result: List[Signal] = []

        for ts in timestamps:
            log_buy_ratio = 0.0
            log_sell_ratio = 0.0
            price = 0.0
            symbol = "UNKNOWN"

            for sigs, w in zip(signal_groups, weights):
                for s in sigs:
                    if s.timestamp == ts:
                        # Likelihood from signal strength (0.5 = no info)
                        p_buy_given_signal = 0.5 + w * (s.strength - 0.5)
                        p_sell_given_signal = 0.5 + w * (s.strength - 0.5)

                        if s.type == SignalType.BUY:
                            log_buy_ratio += np.log(p_buy_given_signal + 1e-10)
                            log_sell_ratio += np.log((1 - p_buy_given_signal) + 1e-10)
                        elif s.type == SignalType.SELL:
                            log_buy_ratio += np.log((1 - p_sell_given_signal) + 1e-10)
                            log_sell_ratio += np.log(p_sell_given_signal + 1e-10)

                        price = s.price
                        symbol = s.symbol

            # Posterior ratio (unnormalized)
            post_buy = np.exp(log_buy_ratio) * self.prior_buy
            post_sell = np.exp(log_sell_ratio) * self.prior_sell
            total = post_buy + post_sell

            if total < 1e-10:
                continue

            post_buy /= total
            post_sell /= total

            if post_buy >= self.threshold and post_buy > post_sell:
                result.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(post_buy),
                        reason=f"BayesianFusion: P(BUY)={post_buy:.3f}",
                        metadata={"p_buy": float(post_buy), "p_sell": float(post_sell)},
                    )
                )
            elif post_sell >= self.threshold and post_sell > post_buy:
                result.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=symbol,
                        timestamp=ts,
                        price=float(price),
                        strength=float(post_sell),
                        reason=f"BayesianFusion: P(SELL)={post_sell:.3f}",
                        metadata={"p_buy": float(post_buy), "p_sell": float(post_sell)},
                    )
                )

        return result


# ---------------------------------------------------------------------------
# SignalCombinator — user-facing wrapper
# ---------------------------------------------------------------------------

@dataclass
class SignalCombinator:
    """Combines multiple signal generators using a configurable fusion strategy.

    Usage:
        combinator = SignalCombinator(strategy="weighted", threshold=0.6)
        combinator.add(RSIGenerator(14), weight=0.4)
        combinator.add(MACDGenerator(), weight=0.3)
        combinator.add(BollingerGenerator(), weight=0.3)
        signals = combinator.generate(df)
    """
    strategy: str = "weighted"  # weighted | voting | ensemble | bayesian
    threshold: float = 0.5
    min_interval_bars: int = 1

    _generators: List[Any] = field(default_factory=list)
    _weights: List[float] = field(default_factory=list)

    def add(self, generator: Any, weight: float = 1.0) -> "SignalCombinator":
        """Add a signal generator with a weight.

        Returns self for chaining.
        """
        self._generators.append(generator)
        self._weights.append(weight)
        return self

    def _make_strategy(self) -> FusionStrategy:
        if self.strategy == "weighted":
            return WeightedAverageFusion(
                threshold=self.threshold,
                min_interval_bars=self.min_interval_bars,
            )
        elif self.strategy == "voting":
            return VotingFusion(min_agreement=self.threshold)
        elif self.strategy == "ensemble":
            return EnsembleFusion()
        elif self.strategy == "bayesian":
            return BayesianFusion(threshold=self.threshold)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        """Generate fused signals from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        List[Signal]
            Fused signal stream.
        """
        if not self._generators:
            return []

        signal_groups = [gen.generate(df) for gen in self._generators]
        strategy = self._make_strategy()
        return strategy.combine(signal_groups, self._weights, df)

    def get_signal_scores(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return per-bar BUY/SELL scores for analysis.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, close, buy_score, sell_score, net_score
        """
        if not self._generators:
            return pd.DataFrame()

        timestamps = sorted(
            {s.timestamp for gen in self._generators for s in gen.generate(df)}
        )

        rows = []
        for ts in timestamps:
            buy_score = 0.0
            sell_score = 0.0
            price = 0.0

            for gen, w in zip(self._generators, self._weights):
                for s in gen.generate(df):
                    if s.timestamp == ts:
                        if s.type == SignalType.BUY:
                            buy_score += w * s.strength
                        elif s.type == SignalType.SELL:
                            sell_score += w * s.strength
                        price = s.price

            rows.append(
                {
                    "timestamp": ts,
                    "close": price,
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "net_score": buy_score - sell_score,
                }
            )

        return pd.DataFrame(rows)
