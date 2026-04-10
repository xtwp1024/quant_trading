# -*- coding: utf-8 -*-
"""
Agent System Backtest
====================

Backtest comparing:
1. Agent System (MarketAnalysisOrchestrator with Bull/Bear debate, Decision Core, Reflection)
2. Baseline (Simple signal-based without agents)

Metrics compared:
- Win rate
- Sharpe ratio
- Max drawdown
- Total return
- Trade frequency

Usage:
    python agent_backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("AgentBacktest")


# ===================== Standalone Agent Components =====================


class Stance(Enum):
    """Agent stance types"""
    STRONGLY_BULLISH = "STRONGLY_BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    STRONGLY_BEARISH = "STRONGLY_BEARISH"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class BullResult:
    """Bull agent result"""
    stance: str
    bullish_reasons: str
    bull_confidence: float
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stance': self.stance,
            'bullish_reasons': self.bullish_reasons,
            'bull_confidence': self.bull_confidence,
        }


@dataclass
class BearResult:
    """Bear agent result"""
    stance: str
    bearish_reasons: str
    bear_confidence: float
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stance': self.stance,
            'bearish_reasons': self.bearish_reasons,
            'bear_confidence': self.bear_confidence,
        }


@dataclass
class VoteResult:
    """Decision vote result"""
    action: str
    confidence: float
    weighted_score: float
    vote_details: Dict[str, float]
    multi_period_aligned: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'weighted_score': self.weighted_score,
            'vote_details': self.vote_details,
            'multi_period_aligned': self.multi_period_aligned,
            'reason': self.reason,
        }


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    size_pct: float
    leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    kelly_fraction: float
    risk_level: str


@dataclass
class MarketDirectionConfidence:
    """Market direction confidence"""
    direction: Stance
    confidence: float
    bull_weight: float
    bear_weight: float
    key_factors: List[str]
    warnings: List[str]


@dataclass
class AgentRecommendation:
    """Final agent recommendation"""
    action: str
    confidence: float
    market_direction: MarketDirectionConfidence
    position_sizing: PositionSizing
    raw_vote_result: Dict[str, Any]


# ===================== Agent Implementations =====================


class SimpleBullAgent:
    """Simplified bull agent for backtesting"""

    def analyze(self, market_context: Dict[str, Any]) -> BullResult:
        """Rule-based bull analysis"""
        quant = market_context.get('quant_analysis', {})
        trend = quant.get('trend', {})
        osc = quant.get('oscillator', {})
        sentiment = quant.get('sentiment', {})

        reasons = []
        confidence = 50.0

        # Trend check
        t_1h = trend.get('trend_1h_score', 0)
        t_15m = trend.get('trend_15m_score', 0)

        if t_1h > 20:
            reasons.append(f"1h trend bullish ({t_1h:.0f})")
            confidence += 15

        if t_15m > 15:
            reasons.append(f"15m trend bullish ({t_15m:.0f})")
            confidence += 10

        # RSI check
        rsi = osc.get('rsi_1h', 50)
        if rsi < 40:
            reasons.append(f"RSI oversold ({rsi:.1f})")
            confidence += 15
        elif rsi < 50:
            confidence += 5

        # Sentiment
        sent = sentiment.get('total_sentiment_score', 0)
        if sent > 20:
            reasons.append(f"Positive sentiment ({sent:.0f})")
            confidence += 10

        confidence = min(95, confidence)

        if confidence >= 70:
            stance = Stance.STRONGLY_BULLISH
        elif confidence >= 55:
            stance = Stance.SLIGHTLY_BULLISH
        else:
            stance = Stance.NEUTRAL

        return BullResult(
            stance=stance.value,
            bullish_reasons="; ".join(reasons) if reasons else "No strong bullish signals",
            bull_confidence=confidence,
        )


class SimpleBearAgent:
    """Simplified bear agent for backtesting"""

    def analyze(self, market_context: Dict[str, Any]) -> BearResult:
        """Rule-based bear analysis"""
        quant = market_context.get('quant_analysis', {})
        trend = quant.get('trend', {})
        osc = quant.get('oscillator', {})
        sentiment = quant.get('sentiment', {})

        reasons = []
        confidence = 50.0

        # Trend check
        t_1h = trend.get('trend_1h_score', 0)
        t_15m = trend.get('trend_15m_score', 0)

        if t_1h < -20:
            reasons.append(f"1h trend bearish ({t_1h:.0f})")
            confidence += 15

        if t_15m < -15:
            reasons.append(f"15m trend bearish ({t_15m:.0f})")
            confidence += 10

        # RSI check
        rsi = osc.get('rsi_1h', 50)
        if rsi > 60:
            reasons.append(f"RSI overbought ({rsi:.1f})")
            confidence += 15
        elif rsi > 50:
            confidence += 5

        # Sentiment
        sent = sentiment.get('total_sentiment_score', 0)
        if sent < -20:
            reasons.append(f"Negative sentiment ({sent:.0f})")
            confidence += 10

        confidence = min(95, confidence)

        if confidence >= 70:
            stance = Stance.STRONGLY_BEARISH
        elif confidence >= 55:
            stance = Stance.SLIGHTLY_BEARISH
        else:
            stance = Stance.NEUTRAL

        return BearResult(
            stance=stance.value,
            bearish_reasons="; ".join(reasons) if reasons else "No strong bearish signals",
            bear_confidence=confidence,
        )


class SimpleDecisionCore:
    """Simplified decision core for backtesting"""

    def decide(
        self,
        quant_analysis: Dict[str, Any],
        bull_perspective: Optional[Dict[str, Any]] = None,
        bear_perspective: Optional[Dict[str, Any]] = None,
    ) -> VoteResult:
        """Simplified decision"""
        trend = quant_analysis.get('trend', {})
        t_1h = trend.get('trend_1h_score', 0)
        t_15m = trend.get('trend_15m_score', 0)
        t_5m = trend.get('trend_5m_score', 0)

        # Calculate weighted score
        score = t_1h * 0.5 + t_15m * 0.3 + t_5m * 0.2

        # Bull/Bear adjustment
        if bull_perspective and bear_perspective:
            bull_conf = bull_perspective.get('bull_confidence', 50)
            bear_conf = bear_perspective.get('bear_confidence', 50)
            if bull_conf > 60:
                score += 10
            elif bear_conf > 60:
                score -= 10

        # Check alignment
        signs = [
            1 if t_1h >= 25 else (-1 if t_1h <= -25 else 0),
            1 if t_15m >= 18 else (-1 if t_15m <= -18 else 0),
            1 if t_5m >= 12 else (-1 if t_5m <= -12 else 0),
        ]
        aligned = signs[0] == signs[1] == signs[2] and signs[0] != 0

        # Determine action
        if score > 20 and aligned:
            action = "open_long"
            confidence = min(95, 50 + score / 2)
        elif score < -20 and aligned:
            action = "open_short"
            confidence = min(95, 50 + abs(score) / 2)
        elif score > 15:
            action = "open_long"
            confidence = 55
        elif score < -15:
            action = "open_short"
            confidence = 55
        else:
            action = "wait"
            confidence = abs(score) / 2

        return VoteResult(
            action=action,
            confidence=confidence,
            weighted_score=score,
            vote_details={'trend_1h': t_1h, 'trend_15m': t_15m, 'trend_5m': t_5m},
            multi_period_aligned=aligned,
            reason=f"Score {score:.1f}, aligned={aligned}",
        )


class SimplePositionSizer:
    """Simplified position sizer"""

    def calculate(self, confidence: float, direction: str) -> PositionSizing:
        """Calculate position sizing"""
        size_pct = 10.0 + (confidence - 50) * 0.1
        size_pct = max(5, min(25, size_pct))

        leverage = 1.0 if size_pct < 15 else 1.5
        stop_loss_pct = 2.0
        take_profit_pct = 3.0 if direction == "long" else 4.0

        return PositionSizing(
            size_pct=round(size_pct, 2),
            leverage=leverage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            kelly_fraction=0.1,
            risk_level="moderate" if size_pct < 15 else "aggressive",
        )


class SimpleOrchestrator:
    """
    Simplified orchestrator for backtesting with:
    - Bull/Bear adversarial debate
    - Regime filtering
    - Confidence-based entry filtering
    """

    def __init__(self):
        self.bull_agent = SimpleBullAgent()
        self.bear_agent = SimpleBearAgent()
        self.decision_core = SimpleDecisionCore()
        self.position_sizer = SimplePositionSizer()

    def analyze_and_decide(self, market_context: Dict[str, Any]) -> AgentRecommendation:
        """Run simplified agent analysis with regime filtering"""
        # Check regime
        quant = market_context.get('quant_analysis', {})
        regime = quant.get('regime', {})
        regime_type = regime.get('regime', 'choppy') if regime else 'choppy'

        # Bull/Bear debate
        bull_result = self.bull_agent.analyze(market_context)
        bear_result = self.bear_agent.analyze(market_context)

        # Market direction
        bull_weight = bull_result.bull_confidence / 100
        bear_weight = bear_result.bear_confidence / 100
        total = bull_weight + bear_weight
        if total > 0:
            bull_weight /= total
            bear_weight /= total

        if bull_weight > 0.6:
            direction = Stance.SLIGHTLY_BULLISH if bull_weight < 0.75 else Stance.STRONGLY_BULLISH
        elif bear_weight > 0.6:
            direction = Stance.SLIGHTLY_BEARISH if bear_weight < 0.75 else Stance.STRONGLY_BEARISH
        else:
            direction = Stance.NEUTRAL

        market_direction = MarketDirectionConfidence(
            direction=direction,
            confidence=max(bull_result.bull_confidence, bear_result.bear_confidence),
            bull_weight=bull_weight,
            bear_weight=bear_weight,
            key_factors=[bull_result.bullish_reasons[:50], bear_result.bearish_reasons[:50]],
            warnings=[],
        )

        # Decision with regime filter
        vote_result = self.decision_core.decide(
            quant_analysis=quant,
            bull_perspective=bull_result.to_dict(),
            bear_perspective=bear_result.to_dict(),
        )

        # Apply regime filter - no trades in choppy regime
        if regime_type == 'choppy' and vote_result.action in ('open_long', 'open_short'):
            vote_result = VoteResult(
                action="wait",
                confidence=vote_result.confidence * 0.5,
                weighted_score=vote_result.weighted_score * 0.5,
                vote_details=vote_result.vote_details,
                multi_period_aligned=False,
                reason=vote_result.reason + " [filtered: choppy regime]",
            )

        # Apply confidence filter - only take high confidence trades
        if vote_result.action in ('open_long', 'open_short') and vote_result.confidence < 60:
            vote_result = VoteResult(
                action="wait",
                confidence=vote_result.confidence,
                weighted_score=vote_result.weighted_score,
                vote_details=vote_result.vote_details,
                multi_period_aligned=vote_result.multi_period_aligned,
                reason=vote_result.reason + " [filtered: low confidence]",
            )

        # Position sizing
        position_sizing = self.position_sizer.calculate(vote_result.confidence, vote_result.action)

        return AgentRecommendation(
            action=vote_result.action,
            confidence=vote_result.confidence,
            market_direction=market_direction,
            position_sizing=position_sizing,
            raw_vote_result=vote_result.to_dict(),
        )


# ===================== Data Models =====================


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    confidence: float
    exit_reason: str


@dataclass
class Position:
    """Current position"""
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    confidence: float


@dataclass
class BacktestResult:
    """Backtest result summary"""
    strategy_name: str
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_duration_hours: float
    trades: List[Trade]


@dataclass
class MarketBar:
    """Market data bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===================== Mock Data Generator =====================


class MockMarketDataGenerator:
    """Generates realistic mock market data for backtesting."""

    def __init__(self, symbol: str, start_date: datetime, end_date: datetime):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.current_price = 50000.0 if "BTC" in symbol else 3000.0
        self.trend = 0.0
        self.volatility = 0.02

    def generate_bars(self, interval_minutes: int = 60) -> List[MarketBar]:
        """Generate OHLCV bars"""
        bars = []
        current_time = self.start_date
        delta = timedelta(minutes=interval_minutes)

        while current_time < self.end_date:
            trend_pull = self.trend * 0.1
            change = np.random.normal(trend_pull, self.volatility)
            self.current_price *= (1 + change)

            self.trend *= 0.95
            if abs(change) > 2 * self.volatility:
                self.trend -= np.sign(change) * 0.1

            high_mult = 1 + abs(np.random.normal(0, 0.005))
            low_mult = 1 - abs(np.random.normal(0, 0.005))
            open_price = self.current_price * (1 + np.random.normal(0, 0.001))

            high = max(open_price, self.current_price) * high_mult
            low = min(open_price, self.current_price) * low_mult
            volume = np.random.uniform(100, 1000)

            bars.append(MarketBar(
                timestamp=current_time,
                open=open_price,
                high=high,
                low=low,
                close=self.current_price,
                volume=volume,
            ))

            current_time += delta

        return bars

    def generate_quant_analysis(self, bars: List[MarketBar]) -> Dict[str, Any]:
        """Generate mock quant analysis from bars"""
        if len(bars) < 60:
            return {}

        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])

        sma_20 = np.mean(closes[-20:])
        sma_60 = np.mean(closes[-60:])

        if closes[-1] > sma_20 > sma_60:
            trend_1h, trend_15m, trend_5m = 40.0, 35.0, 30.0
        elif closes[-1] < sma_20 < sma_60:
            trend_1h, trend_15m, trend_5m = -40.0, -35.0, -30.0
        else:
            trend_1h, trend_15m, trend_5m = 10.0, 5.0, 0.0

        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        recent_vol = volumes[-1]
        avg_vol = np.mean(volumes[-20:])
        vol_ratio = recent_vol / (avg_vol + 1e-10)

        return {
            'trend': {
                'trend_1h_score': trend_1h,
                'trend_15m_score': trend_15m,
                'trend_5m_score': trend_5m,
            },
            'oscillator': {
                'osc_1h_score': (rsi - 50) * 2,
                'osc_15m_score': (rsi - 50) * 1.5,
                'osc_5m_score': (rsi - 50),
                'rsi_1h': rsi,
            },
            'sentiment': {
                'total_sentiment_score': np.random.normal(0, 30),
            },
            'regime': {
                'regime': 'trending' if abs(trend_1h) > 25 else 'choppy',
                'adx': 25,
            },
            'traps': {
                'bull_trap_risk': random.random() < 0.1,
                'bear_trap_risk': random.random() < 0.1,
            },
            'symbol': self.symbol,
        }


# ===================== Baseline Strategy =====================


class BaselineStrategy:
    """Simple baseline strategy without agents."""

    def __init__(self):
        self.trend_threshold = 25  # Trend threshold for entry

    def should_long(self, quant_analysis: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if should enter long - simple trend following"""
        trend = quant_analysis.get('trend', {})

        trend_score = trend.get('trend_1h_score', 0)

        # Simple rule: Uptrend above threshold
        if trend_score > self.trend_threshold:
            confidence = min(75, 50 + trend_score / 2)
            return True, confidence

        return False, 0

    def should_short(self, quant_analysis: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if should enter short - simple trend following"""
        trend = quant_analysis.get('trend', {})

        trend_score = trend.get('trend_1h_score', 0)

        # Simple rule: Downtrend below threshold
        if trend_score < -self.trend_threshold:
            confidence = min(75, 50 + abs(trend_score) / 2)
            return True, confidence

        return False, 0


# ===================== Backtest Engine =====================


class AgentBacktestEngine:
    """Backtest engine comparing Agent System vs Baseline."""

    def __init__(self, symbol: str, bars: List[MarketBar], initial_capital: float = 10000.0):
        self.symbol = symbol
        self.bars = bars
        self.initial_capital = initial_capital

        self.baseline = BaselineStrategy()
        self.orchestrator = SimpleOrchestrator()

        self.agent_trades: List[Trade] = []
        self.baseline_trades: List[Trade] = []
        self.agent_balance = initial_capital
        self.baseline_balance = initial_capital
        self.agent_position: Optional[Position] = None
        self.baseline_position: Optional[Position] = None

    def run(self) -> Tuple[BacktestResult, BacktestResult]:
        """Run backtest for both strategies."""
        logger.info(f"Starting backtest: {self.symbol}, {len(self.bars)} bars")

        # Generate quant analysis for each bar
        quant_analyses = []
        for i in range(len(self.bars)):
            analysis_bars = self.bars[max(0, i-60):i+1]
            mock_gen = MockMarketDataGenerator(self.symbol, datetime.now(), datetime.now())
            mock_gen.current_price = self.bars[i].close
            quant_analyses.append(mock_gen.generate_quant_analysis(analysis_bars))

        # Process each bar
        for i, bar in enumerate(self.bars):
            if i < 60:
                continue

            quant = quant_analyses[i]
            quant['symbol'] = self.symbol

            market_context = {
                'quant_analysis': quant,
                'current_price': bar.close,
                'sentiment': quant.get('sentiment', {}),
                'position': {'position_pct': 50},
            }

            self._process_agent(market_context, bar)
            self._process_baseline(quant, bar)

        # Close open positions at end
        if self.agent_position:
            self._close_agent_position(self.bars[-1], "end_of_backtest")
        if self.baseline_position:
            self._close_baseline_position(self.bars[-1], "end_of_backtest")

        agent_result = self._calculate_result("Agent System", self.agent_trades)
        baseline_result = self._calculate_result("Baseline", self.baseline_trades)

        return agent_result, baseline_result

    def _process_agent(self, market_context: Dict[str, Any], bar: MarketBar) -> None:
        """Process agent strategy decisions"""
        if self.agent_position:
            should_close, reason = self._check_agent_exit(bar)
            if should_close:
                self._close_agent_position(bar, reason)
                return

        if not self.agent_position:
            try:
                recommendation = self.orchestrator.analyze_and_decide(market_context)

                if recommendation.action == "open_long":
                    self._open_agent_position(bar, recommendation, "long")
                elif recommendation.action == "open_short":
                    self._open_agent_position(bar, recommendation, "short")

            except Exception as e:
                logger.warning(f"Agent analysis error: {e}")
                should_long, confidence = self.baseline.should_long(market_context['quant_analysis'])
                if should_long:
                    self._open_agent_position(bar, None, "long")

    def _check_agent_exit(self, bar: MarketBar) -> Tuple[bool, str]:
        """Check if agent should exit position"""
        pos = self.agent_position

        if pos.side == "long":
            if bar.low < pos.stop_loss:
                return True, "stop_loss"
            if bar.high > pos.take_profit:
                return True, "take_profit"
        else:
            if bar.high > pos.stop_loss:
                return True, "stop_loss"
            if bar.low < pos.take_profit:
                return True, "take_profit"

        hours_held = (bar.timestamp - pos.entry_time).total_seconds() / 3600
        if hours_held > 48:
            return True, "time_stop"

        return False, ""

    def _open_agent_position(self, bar: MarketBar, recommendation, side: str) -> None:
        """Open a new position"""
        if recommendation and recommendation.position_sizing:
            size_pct = recommendation.position_sizing.size_pct
            stop_loss_pct = recommendation.position_sizing.stop_loss_pct
            take_profit_pct = recommendation.position_sizing.take_profit_pct
            confidence = recommendation.confidence
        else:
            size_pct, stop_loss_pct, take_profit_pct, confidence = 10.0, 2.0, 4.0, 50.0

        position_value = self.agent_balance * (size_pct / 100)
        quantity = position_value / bar.close

        if side == "long":
            stop_loss = bar.close * (1 - stop_loss_pct / 100)
            take_profit = bar.close * (1 + take_profit_pct / 100)
        else:
            stop_loss = bar.close * (1 + stop_loss_pct / 100)
            take_profit = bar.close * (1 - take_profit_pct / 100)

        self.agent_position = Position(
            side=side,
            entry_price=bar.close,
            quantity=quantity,
            entry_time=bar.timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
        )

    def _close_agent_position(self, bar: MarketBar, reason: str) -> None:
        """Close agent position"""
        pos = self.agent_position

        if pos.side == "long":
            exit_price = bar.close
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            exit_price = bar.close
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == "short":
            pnl_pct = -pnl_pct

        self.agent_balance += pnl

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            confidence=pos.confidence,
            exit_reason=reason,
        )
        self.agent_trades.append(trade)
        self.agent_position = None

    def _process_baseline(self, quant_analysis: Dict[str, Any], bar: MarketBar) -> None:
        """Process baseline strategy decisions"""
        if self.baseline_position:
            should_close, reason = self._check_baseline_exit(bar)
            if should_close:
                self._close_baseline_position(bar, reason)
                return

        if not self.baseline_position:
            should_long, confidence = self.baseline.should_long(quant_analysis)
            if should_long:
                self._open_baseline_position(bar, "long", confidence)
                return

            should_short, confidence = self.baseline.should_short(quant_analysis)
            if should_short:
                self._open_baseline_position(bar, "short", confidence)

    def _check_baseline_exit(self, bar: MarketBar) -> Tuple[bool, str]:
        """Check if baseline should exit"""
        pos = self.baseline_position

        if pos.side == "long":
            if bar.low < pos.stop_loss:
                return True, "stop_loss"
            if bar.high > pos.take_profit:
                return True, "take_profit"
        else:
            if bar.high > pos.stop_loss:
                return True, "stop_loss"
            if bar.low < pos.take_profit:
                return True, "take_profit"

        hours_held = (bar.timestamp - pos.entry_time).total_seconds() / 3600
        if hours_held > 48:
            return True, "time_stop"

        return False, ""

    def _open_baseline_position(self, bar: MarketBar, side: str, confidence: float) -> None:
        """Open baseline position"""
        size_pct = 10.0
        stop_loss_pct = 2.0
        take_profit_pct = 4.0

        position_value = self.baseline_balance * (size_pct / 100)
        quantity = position_value / bar.close

        if side == "long":
            stop_loss = bar.close * (1 - stop_loss_pct / 100)
            take_profit = bar.close * (1 + take_profit_pct / 100)
        else:
            stop_loss = bar.close * (1 + stop_loss_pct / 100)
            take_profit = bar.close * (1 - take_profit_pct / 100)

        self.baseline_position = Position(
            side=side,
            entry_price=bar.close,
            quantity=quantity,
            entry_time=bar.timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
        )

    def _close_baseline_position(self, bar: MarketBar, reason: str) -> None:
        """Close baseline position"""
        pos = self.baseline_position

        if pos.side == "long":
            exit_price = bar.close
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            exit_price = bar.close
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == "short":
            pnl_pct = -pnl_pct

        self.baseline_balance += pnl

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            confidence=pos.confidence,
            exit_reason=reason,
        )
        self.baseline_trades.append(trade)
        self.baseline_position = None

    def _calculate_result(self, name: str, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest result metrics"""
        if not trades:
            return BacktestResult(
                strategy_name=name,
                total_trades=0,
                win_rate=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_trade_duration_hours=0.0,
                trades=trades,
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0

        returns = [t.pnl_pct / 100 for t in trades]
        total_return_pct = sum(returns) * 100

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        cumulative = np.cumsum([1] + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            strategy_name=name,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            avg_trade_duration_hours=avg_duration,
            trades=trades,
        )


# ===================== Main =====================


def print_comparison(agent_result: BacktestResult, baseline_result: BacktestResult) -> None:
    """Print comparison of two strategies"""
    print("\n" + "=" * 70)
    print("AGENT SYSTEM BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Agent System':>20} {'Baseline':>20}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {agent_result.total_trades:>20} {baseline_result.total_trades:>20}")
    print(f"{'Win Rate':<25} {agent_result.win_rate*100:>19.1f}% {baseline_result.win_rate*100:>19.1f}%")
    print(f"{'Avg Win %':<25} {agent_result.avg_win_pct:>20.2f} {baseline_result.avg_win_pct:>20.2f}")
    print(f"{'Avg Loss %':<25} {agent_result.avg_loss_pct:>20.2f} {baseline_result.avg_loss_pct:>20.2f}")
    print(f"{'Total Return %':<25} {agent_result.total_return_pct:>20.2f} {baseline_result.total_return_pct:>20.2f}")
    print(f"{'Sharpe Ratio':<25} {agent_result.sharpe_ratio:>20.2f} {baseline_result.sharpe_ratio:>20.2f}")
    print(f"{'Max Drawdown %':<25} {agent_result.max_drawdown_pct:>20.2f} {baseline_result.max_drawdown_pct:>20.2f}")
    print(f"{'Avg Trade Duration (h)':<25} {agent_result.avg_trade_duration_hours:>20.1f} {baseline_result.avg_trade_duration_hours:>20.1f}")

    # Calculate improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    wr_improvement = (agent_result.win_rate - baseline_result.win_rate) * 100
    ret_improvement = agent_result.total_return_pct - baseline_result.total_return_pct
    dd_improvement = baseline_result.max_drawdown_pct - agent_result.max_drawdown_pct

    print(f"\nWin Rate Change:        {wr_improvement:+.1f} percentage points")
    print(f"Return Change:         {ret_improvement:+.2f} percentage points")
    print(f"Drawdown Improvement:  {dd_improvement:+.2f} percentage points (lower is better)")

    if agent_result.sharpe_ratio > baseline_result.sharpe_ratio:
        print(f"Sharpe Ratio:         +{agent_result.sharpe_ratio - baseline_result.sharpe_ratio:.2f} better risk-adjusted returns")
    else:
        print(f"Sharpe Ratio:         {agent_result.sharpe_ratio - baseline_result.sharpe_ratio:.2f} vs baseline")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    agent_wins = 0
    if agent_result.win_rate > baseline_result.win_rate:
        agent_wins += 1
    if agent_result.total_return_pct > baseline_result.total_return_pct:
        agent_wins += 1
    if agent_result.max_drawdown_pct < baseline_result.max_drawdown_pct:
        agent_wins += 1
    if agent_result.sharpe_ratio > baseline_result.sharpe_ratio:
        agent_wins += 1

    if agent_wins >= 3:
        print(f"\nAGENT SYSTEM WINS ({agent_wins}/4 metrics)")
        print("The multi-agent system outperforms the baseline strategy.")
    elif agent_wins == 2:
        print(f"\nTIE ({agent_wins}/4 metrics, depends on priorities)")
        print("Both strategies have similar performance.")
    else:
        print(f"\nBASELINE WINS ({4-agent_wins}/4 metrics)")
        print("The simple baseline outperforms the agent system.")

    print()


def main():
    parser = argparse.ArgumentParser(description='Agent System Backtest')
    parser.add_argument('--symbol', '-s', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (for compatibility with run_all_experiments)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    logger.info(f"Generating mock market data for {args.symbol}...")
    generator = MockMarketDataGenerator(args.symbol, start_date, end_date)
    bars = generator.generate_bars(interval_minutes=60)

    logger.info(f"Generated {len(bars)} bars from {start_date} to {end_date}")

    engine = AgentBacktestEngine(
        symbol=args.symbol,
        bars=bars,
        initial_capital=args.capital,
    )

    agent_result, baseline_result = engine.run()

    print_comparison(agent_result, baseline_result)

    output_file = f"agent_backtest_{args.symbol}_{args.start}_{args.end}.json"
    results = {
        'agent_system': {
            'total_trades': agent_result.total_trades,
            'win_rate': agent_result.win_rate,
            'avg_win_pct': agent_result.avg_win_pct,
            'avg_loss_pct': agent_result.avg_loss_pct,
            'total_return_pct': agent_result.total_return_pct,
            'sharpe_ratio': agent_result.sharpe_ratio,
            'max_drawdown_pct': agent_result.max_drawdown_pct,
            'avg_trade_duration_hours': agent_result.avg_trade_duration_hours,
        },
        'baseline': {
            'total_trades': baseline_result.total_trades,
            'win_rate': baseline_result.win_rate,
            'avg_win_pct': baseline_result.avg_win_pct,
            'avg_loss_pct': baseline_result.avg_loss_pct,
            'total_return_pct': baseline_result.total_return_pct,
            'sharpe_ratio': baseline_result.sharpe_ratio,
            'max_drawdown_pct': baseline_result.max_drawdown_pct,
            'avg_trade_duration_hours': baseline_result.avg_trade_duration_hours,
        },
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
