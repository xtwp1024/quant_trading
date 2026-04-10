# -*- coding: utf-8 -*-
"""
Market Analysis Orchestrator
============================

Multi-agent coordination system that orchestrates:
1. Bull/Bear Agents - Adversarial market perspective analysis
2. Decision Agent - Multi-signal fusion and trading recommendations
3. Reflection Agent - Learning from trade history

Architecture:
    MarketData -> BullAgent + BearAgent -> SentimentAggregator
                    -> TechnicalSynthesizer -> MarketDirectionConfidencer
                    -> DecisionAgent -> PositionSizer + RiskAssessor + EntryExitRecommender
                    -> ReflectionAgent (async on trade outcomes)

Usage:
    orchestrator = MarketAnalysisOrchestrator()
    result = await orchestrator.analyze_and_decide(market_context, trade_history)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json

# Import existing agents
from quant_trading.decision.bull_bear_agents import (
    BullAgent, BearAgent, BullResult, BearResult, Stance
)
from quant_trading.decision.decision_core import (
    DecisionCore, VoteResult, Action, SignalWeight
)
from quant_trading.decision.reflection_agent import (
    ReflectionAgent, ReflectionResult
)


class MarketDirection(Enum):
    """Market direction classification"""
    STRONGLY_BULLISH = "STRONGLY_BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    STRONGLY_BEARISH = "STRONGLY_BEARISH"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SentimentAggregation:
    """Aggregated sentiment from multiple sources"""
    total_score: float  # -100 to +100
    bull_score: float
    bear_score: float
    confidence: float  # 0-100
    sources: Dict[str, float]  # Breakdown by source
    trend: str  # "improving", "stable", "deteriorating"


@dataclass
class TechnicalSynthesis:
    """Synthesized technical analysis"""
    trend_alignment: str  # "aligned", "diverging", "mixed"
    momentum: str  # "strong", "moderate", "weak"
    volatility: str  # "high", "normal", "low"
    key_levels: Dict[str, float]  # Support/resistance levels
    oscillators: Dict[str, float]  # RSI, KDJ, etc.


@dataclass
class MarketDirectionConfidence:
    """Final market direction with confidence"""
    direction: MarketDirection
    confidence: float  # 0-100
    bull_weight: float  # 0-1
    bear_weight: float  # 0-1
    key_factors: List[str]
    warnings: List[str]


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    size_pct: float  # 0-100 of capital
    leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    kelly_fraction: float  # Kelly criterion fraction
    risk_level: str  # "conservative", "moderate", "aggressive"


@dataclass
class RiskAssessment:
    """Risk assessment for proposed trade"""
    risk_score: float  # 0-100 (higher = riskier)
    max_drawdown_pct: float
    volatility_adjustment: float
    trap_risk: Dict[str, bool]  # Various trap indicators
    regime_risk: str
    overall_rating: str  # "low", "medium", "high", "extreme"


@dataclass
class EntryExitRecommendation:
    """Entry and exit recommendations"""
    entry_price: float
    entry_range_low: float
    entry_range_high: float
    stop_loss: float
    take_profit: float
    time_horizon: str  # "scalp", "swing", "position"
    exit_conditions: List[str]


@dataclass
class AgentRecommendation:
    """Final recommendation from all agents"""
    action: str  # "open_long", "open_short", "wait", "hold"
    confidence: float  # 0-100
    market_direction: MarketDirectionConfidence
    position_sizing: PositionSizing
    risk_assessment: RiskAssessment
    entry_exit: EntryExitRecommendation
    debate_summary: str  # Summary of bull/bear debate
    reflection_insights: Optional[str]  # From reflection agent
    raw_vote_result: Dict[str, Any]


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources into a unified score.
    """

    def __init__(self):
        self.source_weights = {
            'social': 0.25,
            'news': 0.20,
            'onchain': 0.25,
            'funding': 0.15,
            'whale': 0.15,
        }

    def aggregate(
        self,
        social_sentiment: Optional[Dict[str, Any]] = None,
        news_sentiment: Optional[Dict[str, Any]] = None,
        onchain_data: Optional[Dict[str, Any]] = None,
        funding_rate: Optional[float] = None,
        whale_signals: Optional[Dict[str, Any]] = None,
    ) -> SentimentAggregation:
        """
        Aggregate multiple sentiment sources.

        Returns:
            SentimentAggregation with unified score
        """
        scores = {}
        total_weight = 0.0

        # Social sentiment
        if social_sentiment:
            score = social_sentiment.get('score', 0) * 100
            scores['social'] = score
            total_weight += self.source_weights['social']
        else:
            scores['social'] = 0.0

        # News sentiment
        if news_sentiment:
            score = news_sentiment.get('score', 0) * 100
            scores['news'] = score
            total_weight += self.source_weights['news']
        else:
            scores['news'] = 0.0

        # On-chain data
        if onchain_data:
            score = self._score_onchain(onchain_data)
            scores['onchain'] = score
            total_weight += self.source_weights['onchain']
        else:
            scores['onchain'] = 0.0

        # Funding rate
        if funding_rate is not None:
            # Funding rate: positive = bullish bias, negative = bearish
            score = max(-100, min(100, funding_rate * 1000))  # Scale to -100 to +100
            scores['funding'] = score
            total_weight += self.source_weights['funding']
        else:
            scores['funding'] = 0.0

        # Whale signals
        if whale_signals:
            score = self._score_whale(whale_signals)
            scores['whale'] = score
            total_weight += self.source_weights['whale']
        else:
            scores['whale'] = 0.0

        # Calculate weighted average
        if total_weight > 0:
            total_score = sum(scores[k] * self.source_weights[k] for k in scores) / total_weight
        else:
            total_score = 0.0

        # Determine trend
        trend = "stable"
        if abs(total_score) > 20:
            trend = "improving" if total_score > 0 else "deteriorating"

        # Calculate confidence based on source availability
        active_sources = sum(1 for k in scores if scores[k] != 0)
        confidence = min(100, active_sources * 25)

        # Bull/Bear breakdown
        bull_score = max(0, total_score)
        bear_score = max(0, -total_score)

        return SentimentAggregation(
            total_score=total_score,
            bull_score=bull_score,
            bear_score=bear_score,
            confidence=confidence,
            sources=scores,
            trend=trend,
        )

    def _score_onchain(self, onchain_data: Dict[str, Any]) -> float:
        """Score on-chain data"""
        score = 0.0
        count = 0

        # Exchange flows
        if 'exchange_flow' in onchain_data:
            flow = onchain_data['exchange_flow']
            if flow > 0:  # Inflow to exchange = bearish
                score -= min(50, flow * 10)
            else:  # Outflow = bullish
                score += min(50, abs(flow) * 10)
            count += 1

        # Active addresses growth
        if 'active_addresses_change' in onchain_data:
            change = onchain_data['active_addresses_change']
            score += max(-30, min(30, change * 5))
            count += 1

        # Gas prices (high gas = high usage = potential top signal)
        if 'gas_price' in onchain_data:
            gas = onchain_data['gas_price']
            if gas > 100:  # High gas could indicate overheating
                score -= 10
            count += 1

        return score / max(count, 1) if count > 0 else 0.0

    def _score_whale(self, whale_signals: Dict[str, Any]) -> float:
        """Score whale activity signals"""
        score = 0.0
        count = 0

        # Whale transactions
        if 'buy_pressure' in whale_signals:
            bp = whale_signals['buy_pressure']
            score += max(-50, min(50, bp * 50))
            count += 1

        # Large transaction ratio
        if 'large_tx_ratio' in whale_signals:
            ratio = whale_signals['large_tx_ratio']
            if ratio > 0.3:  # High ratio of large txs
                score -= 20  # Could indicate distribution
            count += 1

        # Whale accumulation indicator
        if 'accumulation_score' in whale_signals:
            acc = whale_signals['accumulation_score']
            score += max(-30, min(30, (acc - 0.5) * 100))
            count += 1

        return score / max(count, 1) if count > 0 else 0.0


class TechnicalSynthesizer:
    """
    Synthesizes technical analysis across multiple timeframes.
    """

    def __init__(self):
        self.oscillator_weights = {
            'rsi': 0.30,
            'kdj': 0.25,
            'macd': 0.25,
            'bb_position': 0.20,
        }

    def synthesize(
        self,
        trend_scores: Dict[str, float],
        oscillator_scores: Dict[str, float],
        price_data: Optional[Dict[str, Any]] = None,
    ) -> TechnicalSynthesis:
        """
        Synthesize technical analysis into unified view.

        Returns:
            TechnicalSynthesis with trend, momentum, volatility assessment
        """
        # Trend alignment
        t_1h = trend_scores.get('trend_1h', 0)
        t_15m = trend_scores.get('trend_15m', 0)
        t_5m = trend_scores.get('trend_5m', 0)

        signs = [1 if s > 20 else (-1 if s < -20 else 0) for s in [t_1h, t_15m, t_5m]]

        if signs[0] == signs[1] == signs[2] and signs[0] != 0:
            trend_alignment = "aligned"
        elif signs[0] == signs[1] and signs[0] != 0:
            trend_alignment = "partial"
        elif all(s == 0 for s in signs):
            trend_alignment = "mixed"
        else:
            trend_alignment = "diverging"

        # Momentum assessment
        osc_rsi = oscillator_scores.get('rsi_1h', 50)
        osc_kdj = oscillator_scores.get('kdj_j', 50)

        if 40 <= osc_rsi <= 60 and 40 <= osc_kdj <= 60:
            momentum = "moderate"
        elif (osc_rsi < 30 or osc_kdj < 20) or (osc_rsi > 70 or osc_kdj > 80):
            momentum = "strong"
        else:
            momentum = "weak"

        # Volatility assessment
        volatility = "normal"
        if price_data:
            if 'atr_pct' in price_data:
                atr = price_data['atr_pct']
                if atr > 3:
                    volatility = "high"
                elif atr < 1:
                    volatility = "low"

        # Key levels
        key_levels = {}
        if price_data:
            key_levels = {
                'resistance': price_data.get('resistance', 0),
                'support': price_data.get('support', 0),
                'pivot': price_data.get('pivot', 0),
            }

        return TechnicalSynthesis(
            trend_alignment=trend_alignment,
            momentum=momentum,
            volatility=volatility,
            key_levels=key_levels,
            oscillators={
                'rsi': osc_rsi,
                'kdj': osc_kdj,
                'macd': oscillator_scores.get('macd', 0),
            },
        )


class MarketDirectionCalculator:
    """
    Calculates final market direction confidence from bull/bear debate.
    """

    def __init__(self):
        self.weights = {
            'bull_confidence': 0.5,
            'bear_confidence': 0.5,
            'alignment_bonus': 0.1,
        }

    def calculate(
        self,
        bull_result: BullResult,
        bear_result: BearResult,
        sentiment: SentimentAggregation,
        technical: TechnicalSynthesis,
    ) -> MarketDirectionConfidence:
        """
        Calculate market direction from debate results.

        Returns:
            MarketDirectionConfidence with direction and confidence
        """
        # Base weights from bull/bear agents
        bull_weight = bull_result.bull_confidence / 100.0
        bear_weight = bear_result.bear_confidence / 100.0

        # Normalize
        total = bull_weight + bear_weight
        if total > 0:
            bull_weight /= total
            bear_weight /= total

        # Sentiment adjustment
        if sentiment.total_score > 30:
            bull_weight += 0.1
        elif sentiment.total_score < -30:
            bear_weight += 0.1

        # Technical alignment bonus
        if technical.trend_alignment == "aligned":
            if bull_weight > bear_weight:
                bull_weight += self.weights['alignment_bonus']
            else:
                bear_weight += self.weights['alignment_bonus']

        # Re-normalize
        total = bull_weight + bear_weight
        bull_weight /= total
        bear_weight /= total

        # Determine direction
        if bull_weight > 0.6:
            if bull_weight > 0.75:
                direction = MarketDirection.STRONGLY_BULLISH
            else:
                direction = MarketDirection.SLIGHTLY_BULLISH
        elif bear_weight > 0.6:
            if bear_weight > 0.75:
                direction = MarketDirection.STRONGLY_BEARISH
            else:
                direction = MarketDirection.SLIGHTLY_BEARISH
        else:
            direction = MarketDirection.NEUTRAL

        # Calculate confidence
        confidence = abs(bull_weight - bear_weight) * 100
        confidence *= (sentiment.confidence / 100)  # Scale by sentiment confidence
        confidence = max(10, min(95, confidence))

        # Key factors
        key_factors = []
        if bull_result.bullish_reasons and bull_result.bullish_reasons != "No strong bullish signals":
            key_factors.append(f"Bull: {bull_result.bullish_reasons[:100]}")
        if bear_result.bearish_reasons and bear_result.bearish_reasons != "No strong bearish signals":
            key_factors.append(f"Bear: {bear_result.bearish_reasons[:100]}")

        # Warnings
        warnings = []
        if technical.trend_alignment == "diverging":
            warnings.append("Timeframe divergence detected")
        if technical.volatility == "high":
            warnings.append("High volatility - widen stops")
        if abs(sentiment.total_score) > 70:
            warnings.append("Extreme sentiment - caution advised")

        return MarketDirectionConfidence(
            direction=direction,
            confidence=confidence,
            bull_weight=bull_weight,
            bear_weight=bear_weight,
            key_factors=key_factors,
            warnings=warnings,
        )


class PositionSizer:
    """
    Calculates position sizing based on Kelly criterion and risk parameters.
    """

    def __init__(self):
        self.base_size_pct = 10.0  # Base position size
        self.max_size_pct = 25.0  # Maximum position size
        self.base_leverage = 1.0
        self.max_leverage = 3.0

    def calculate(
        self,
        confidence: float,
        risk_assessment: RiskAssessment,
        market_direction: MarketDirectionConfidence,
        account_balance: float = 10000.0,
    ) -> PositionSizing:
        """
        Calculate position sizing.

        Returns:
            PositionSizing with size, leverage, stops
        """
        # Base Kelly calculation
        win_rate = confidence / 100.0
        avg_win = 2.0  # Assumed 2:1 reward/risk
        avg_loss = 1.0

        if avg_loss > 0 and win_rate > 0:
            kelly = (win_rate * avg_win - avg_loss) / avg_win
            kelly_fraction = max(0.05, min(0.25, kelly / 2))  # Half-Kelly for safety
        else:
            kelly_fraction = 0.10

        # Adjust based on confidence
        conf_mult = confidence / 70.0  # Normalize to ~1.0 at 70% confidence
        conf_mult = max(0.5, min(1.5, conf_mult))

        # Risk adjustment
        if risk_assessment.overall_rating == "high":
            conf_mult *= 0.7
        elif risk_assessment.overall_rating == "extreme":
            conf_mult *= 0.4

        # Direction adjustment
        if market_direction.direction in (MarketDirection.STRONGLY_BULLISH, MarketDirection.STRONGLY_BEARISH):
            conf_mult *= 1.2

        # Calculate size
        size_pct = self.base_size_pct * conf_mult * kelly_fraction / 0.1
        size_pct = max(2.0, min(self.max_size_pct, size_pct))

        # Calculate leverage
        leverage = self.base_leverage
        if risk_assessment.volatility_adjustment < 0.5:
            leverage = min(self.max_leverage, leverage * 1.5)
        elif risk_assessment.volatility_adjustment > 1.5:
            leverage = max(1.0, leverage * 0.7)

        # Calculate stops
        base_sl = 1.5  # 1.5% base stop
        base_tp = 3.0  # 3.0% base take profit

        sl_mult = 1.0 + (leverage - 1) * 0.2
        tp_mult = 1.0 + (leverage - 1) * 0.1

        stop_loss_pct = base_sl * sl_mult * (1 + risk_assessment.volatility_adjustment * 0.3)
        take_profit_pct = base_tp * tp_mult

        # Risk level
        if size_pct < 8 and leverage <= 1:
            risk_level = "conservative"
        elif size_pct < 15 and leverage <= 2:
            risk_level = "moderate"
        else:
            risk_level = "aggressive"

        return PositionSizing(
            size_pct=round(size_pct, 2),
            leverage=round(leverage, 1),
            stop_loss_pct=round(stop_loss_pct, 2),
            take_profit_pct=round(take_profit_pct, 2),
            kelly_fraction=round(kelly_fraction, 3),
            risk_level=risk_level,
        )


class RiskAssessor:
    """
    Assesses risk for proposed trades.
    """

    def __init__(self):
        self.trap_weights = {
            'bull_trap': 0.3,
            'bear_trap': 0.3,
            'whale_trap': 0.2,
            'fomo_trap': 0.2,
        }

    def assess(
        self,
        regime: Optional[Dict[str, Any]],
        traps: Optional[Dict[str, Any]],
        volatility: str,
        position_pct: float,
        sentiment: SentimentAggregation,
    ) -> RiskAssessment:
        """
        Assess risk for proposed trade.

        Returns:
            RiskAssessment with risk score and details
        """
        risk_score = 50.0  # Base risk
        trap_risk = {}
        warnings = []

        # Regime risk
        regime_risk = "medium"
        if regime:
            regime_type = regime.get('regime', 'unknown')
            if regime_type in ('choppy', 'volatile_directionless'):
                risk_score += 20
                regime_risk = "high"
            elif regime_type in ('trending_up', 'trending_down'):
                risk_score -= 10
                regime_risk = "low"

        # Trap risks
        if traps:
            if traps.get('bull_trap_risk'):
                trap_risk['bull_trap'] = True
                risk_score += 15
            if traps.get('bear_trap_risk'):
                trap_risk['bear_trap'] = True
                risk_score += 15
            if traps.get('whale_trap_risk'):
                trap_risk['whale_trap'] = True
                risk_score += 10
            if traps.get('fomo_top'):
                trap_risk['fomo_trap'] = True
                risk_score += 10
            if traps.get('panic_bottom'):
                trap_risk['panic_bottom'] = True
                risk_score -= 10

        # Position risk
        if position_pct > 80:
            risk_score += 15
        elif position_pct < 20:
            risk_score -= 10

        # Sentiment risk
        if abs(sentiment.total_score) > 70:
            risk_score += 15  # Extreme sentiment is risky

        # Volatility adjustment
        volatility_adjustment = 1.0
        if volatility == "high":
            volatility_adjustment = 1.5
            risk_score += 10
        elif volatility == "low":
            volatility_adjustment = 0.7
            risk_score -= 5

        # Max drawdown estimate
        max_drawdown_pct = volatility_adjustment * 5.0  # Base 5% estimate

        # Overall rating
        if risk_score >= 70:
            overall_rating = "extreme"
        elif risk_score >= 55:
            overall_rating = "high"
        elif risk_score >= 40:
            overall_rating = "medium"
        else:
            overall_rating = "low"

        return RiskAssessment(
            risk_score=min(100, max(0, risk_score)),
            max_drawdown_pct=max_drawdown_pct,
            volatility_adjustment=volatility_adjustment,
            trap_risk=trap_risk,
            regime_risk=regime_risk,
            overall_rating=overall_rating,
        )


class EntryExitRecommender:
    """
    Generates entry/exit recommendations based on analysis.
    """

    def calculate(
        self,
        current_price: float,
        market_direction: MarketDirectionConfidence,
        risk_assessment: RiskAssessment,
        technical: TechnicalSynthesis,
        time_horizon: str = "swing",
    ) -> EntryExitRecommendation:
        """
        Calculate entry and exit levels.

        Returns:
            EntryExitRecommendation with prices and conditions
        """
        # Entry range based on volatility
        vol_mult = risk_assessment.volatility_adjustment
        range_pct = 0.5 * vol_mult  # 0.5% base range

        entry_price = current_price
        entry_range_low = current_price * (1 - range_pct / 100)
        entry_range_high = current_price * (1 + range_pct / 100)

        # Adjust based on direction
        if market_direction.direction in (MarketDirection.STRONGLY_BULLISH, MarketDirection.SLIGHTLY_BULLISH):
            # Prefer buying on dips
            if technical.key_levels.get('support'):
                support = technical.key_levels['support']
                entry_range_low = min(entry_range_low, support)
                entry_price = (entry_range_low + current_price) / 2
        elif market_direction.direction in (MarketDirection.STRONGLY_BEARISH, MarketDirection.SLIGHTLY_BEARISH):
            # Prefer selling on rallies
            if technical.key_levels.get('resistance'):
                resistance = technical.key_levels['resistance']
                entry_range_high = max(entry_range_high, resistance)
                entry_price = (entry_range_high + current_price) / 2

        # Stop loss
        sl_pct = risk_assessment.volatility_adjustment * 1.5
        stop_loss = entry_price * (1 - sl_pct / 100)

        # Take profit
        tp_pct = sl_pct * 2.5  # 2.5:1 reward/risk
        take_profit = entry_price * (1 + tp_pct / 100)

        # Exit conditions
        exit_conditions = [
            f"Stop loss at {stop_loss:.4f} ({sl_pct:.1f}%)",
            f"Take profit at {take_profit:.4f} ({tp_pct:.1f}%)",
        ]

        if time_horizon == "scalp":
            exit_conditions.append("Intraday exit on momentum reversal")
        elif time_horizon == "swing":
            exit_conditions.append("Re-evaluate at next major resistance/support")
        else:  # position
            exit_conditions.append("Trail stop on 4h close below/below EMA")

        return EntryExitRecommendation(
            entry_price=round(entry_price, 4),
            entry_range_low=round(entry_range_low, 4),
            entry_range_high=round(entry_range_high, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            time_horizon=time_horizon,
            exit_conditions=exit_conditions,
        )


class MarketAnalysisOrchestrator:
    """
    Main orchestrator that coordinates all agents for market analysis and decision making.

    Flow:
    1. Receive market context
    2. Run Bull/Bear agents in parallel
    3. Aggregate sentiment
    4. Synthesize technical analysis
    5. Calculate market direction confidence
    6. Decision core vote
    7. Position sizing
    8. Risk assessment
    9. Entry/exit recommendation
    10. (Async) Reflection on trade outcomes
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            use_llm: Whether to use LLM for enhanced analysis
            llm_client: LLM client instance (if use_llm=True)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

        # Initialize sub-agents
        self.bull_agent = BullAgent(llm_client=llm_client if use_llm else None)
        self.bear_agent = BearAgent(llm_client=llm_client if use_llm else None)
        self.decision_core = DecisionCore()
        self.reflection_agent = ReflectionAgent(llm_client=llm_client if use_llm else None)

        # Initialize components
        self.sentiment_aggregator = SentimentAggregator()
        self.technical_synthesizer = TechnicalSynthesizer()
        self.direction_calculator = MarketDirectionCalculator()
        self.position_sizer = PositionSizer()
        self.risk_assessor = RiskAssessor()
        self.entry_exit_recommender = EntryExitRecommender()

        # Trade history for reflection
        self._trade_history: List[Dict] = []

    async def analyze_and_decide(
        self,
        market_context: Dict[str, Any],
        trade_history: Optional[List[Dict]] = None,
    ) -> AgentRecommendation:
        """
        Main entry point: Analyze market and generate recommendation.

        Args:
            market_context: Dict with:
                - quant_analysis: Technical analysis results
                - sentiment: Sentiment data
                - market_data: Raw OHLCV data
                - current_price: Current price
                - symbol: Trading symbol
            trade_history: Optional list of past trades for reflection

        Returns:
            AgentRecommendation with full analysis and trading decision
        """
        # 1. Run Bull/Bear agents
        bull_result, bear_result = self._run_bull_bear_debate(market_context)

        # 2. Aggregate sentiment
        sentiment = self._aggregate_sentiment(market_context)

        # 3. Synthesize technical analysis
        technical = self._synthesize_technical(market_context)

        # 4. Calculate market direction confidence
        market_direction = self.direction_calculator.calculate(
            bull_result, bear_result, sentiment, technical
        )

        # 5. Build context for decision core
        decision_context = self._build_decision_context(
            market_context, bull_result, bear_result, sentiment
        )

        # 6. Decision core vote
        vote_result = self.decision_core.decide(**decision_context)

        # 7. Risk assessment
        risk_assessment = self._assess_risk(market_context, sentiment, technical)

        # 8. Position sizing
        position_sizing = self.position_sizer.calculate(
            confidence=vote_result.confidence,
            risk_assessment=risk_assessment,
            market_direction=market_direction,
        )

        # 9. Entry/exit recommendation
        current_price = market_context.get('current_price', 0)
        entry_exit = self.entry_exit_recommender.calculate(
            current_price=current_price,
            market_direction=market_direction,
            risk_assessment=risk_assessment,
            technical=technical,
        )

        # 10. Debate summary
        debate_summary = self._summarize_debate(bull_result, bear_result)

        # 11. Reflection insights (if available)
        reflection_insights = None
        if trade_history:
            self._update_trade_history(trade_history)
            reflection_result = await self._generate_reflection()
            if reflection_result:
                reflection_insights = reflection_result.to_prompt_text()

        return AgentRecommendation(
            action=vote_result.action,
            confidence=vote_result.confidence,
            market_direction=market_direction,
            position_sizing=position_sizing,
            risk_assessment=risk_assessment,
            entry_exit=entry_exit,
            debate_summary=debate_summary,
            reflection_insights=reflection_insights,
            raw_vote_result=vote_result.to_dict(),
        )

    def _run_bull_bear_debate(
        self, market_context: Dict[str, Any]
    ) -> tuple[BullResult, BearResult]:
        """Run bull and bear agents"""
        bull_result = self.bull_agent.analyze(market_context)
        bear_result = self.bear_agent.analyze(market_context)
        return bull_result, bear_result

    def _aggregate_sentiment(self, market_context: Dict[str, Any]) -> SentimentAggregation:
        """Aggregate sentiment from sources"""
        quant = market_context.get('quant_analysis', {})
        sentiment_data = market_context.get('sentiment', {})

        return self.sentiment_aggregator.aggregate(
            social_sentiment=sentiment_data.get('social'),
            news_sentiment=sentiment_data.get('news'),
            onchain_data=sentiment_data.get('onchain'),
            funding_rate=sentiment_data.get('funding_rate'),
            whale_signals=sentiment_data.get('whale'),
        )

    def _synthesize_technical(self, market_context: Dict[str, Any]) -> TechnicalSynthesis:
        """Synthesize technical analysis"""
        quant = market_context.get('quant_analysis', {})

        trend_scores = quant.get('trend', {})
        oscillator_scores = quant.get('oscillator', {})
        price_data = market_context.get('price_data', {})

        return self.technical_synthesizer.synthesize(
            trend_scores=trend_scores,
            oscillator_scores=oscillator_scores,
            price_data=price_data,
        )

    def _build_decision_context(
        self,
        market_context: Dict[str, Any],
        bull_result: BullResult,
        bear_result: BearResult,
        sentiment: SentimentAggregation,
    ) -> Dict[str, Any]:
        """Build context for decision core"""
        quant = market_context.get('quant_analysis', {})

        return {
            'quant_analysis': quant,
            'market_data': market_context.get('market_data'),
            'predict_result': market_context.get('predict_result'),
            'bull_perspective': bull_result.to_dict(),
            'bear_perspective': bear_result.to_dict(),
            'reflection': None,
        }

    def _assess_risk(
        self,
        market_context: Dict[str, Any],
        sentiment: SentimentAggregation,
        technical: TechnicalSynthesis,
    ) -> RiskAssessment:
        """Assess risk for proposed trade"""
        quant = market_context.get('quant_analysis', {})
        regime = quant.get('regime')
        traps = quant.get('traps', {})
        position_data = market_context.get('position', {})
        position_pct = position_data.get('position_pct', 50)

        return self.risk_assessor.assess(
            regime=regime,
            traps=traps,
            volatility=technical.volatility,
            position_pct=position_pct,
            sentiment=sentiment,
        )

    def _summarize_debate(
        self, bull_result: BullResult, bear_result: BearResult
    ) -> str:
        """Summarize bull/bear debate"""
        parts = []

        if bull_result.stance != 'UNCERTAIN':
            parts.append(f"Bull ({bull_result.bull_confidence:.0f}%): {bull_result.bullish_reasons[:80]}")

        if bear_result.stance != 'UNCERTAIN':
            parts.append(f"Bear ({bear_result.bear_confidence:.0f}%): {bear_result.bearish_reasons[:80]}")

        if not parts:
            return "No strong conviction from either side"

        return " | ".join(parts)

    def _update_trade_history(self, trades: List[Dict]) -> None:
        """Update internal trade history"""
        self._trade_history = trades
        for trade in trades:
            self.reflection_agent.add_trade(trade)

    async def _generate_reflection(self) -> Optional[ReflectionResult]:
        """Generate reflection on trade history"""
        if len(self._trade_history) >= 3:
            return await self.reflection_agent.generate_reflection(self._trade_history)
        return None

    def record_trade_outcome(
        self,
        symbol: str,
        action: str,
        pnl: float,
        confidence: float,
    ) -> None:
        """Record trade outcome for future reflection"""
        trade = {
            'symbol': symbol,
            'action': action,
            'pnl': pnl,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
        }
        self._trade_history.append(trade)
        self.reflection_agent.add_trade(trade)

        # Record with decision core for overtrading guard
        self.decision_core.record_decision_outcome(
            symbol=symbol,
            action=action,
            pnl=pnl,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'decision_core': self.decision_core.get_statistics(),
            'reflection': self.reflection_agent.get_statistics(),
        }
