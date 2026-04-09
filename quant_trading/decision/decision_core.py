"""
Decision Core - The Critic
==========================

Central decision aggregation with weighted voting, multi-period alignment,
market regime detection, and adversarial intelligence.

Based on LLM-TradeBot's DecisionCoreAgent with key adaptations:
- Removes LLM provider implementations (keeps abstraction only)
- Rule-based fallback when no LLM available
- Preserves Bull/Bear adversarial debate pattern
- Multi-timeframe weighted voting

Architecture:
    Inputs: Quant analysis, ML predictions, Bull/Bear perspectives, reflection
    Process: Weighted voting + regime detection + trap filtering
    Output: Action (long/short/wait), confidence, reason
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json


class Action(Enum):
    """Trading actions"""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    WAIT = "wait"


@dataclass
class SignalWeight:
    """
    Signal weight configuration for weighted voting.

    Default configuration (weights should sum to ~1.0):
    - Trend signals: 0.45 (1h:0.30, 15m:0.12, 5m:0.03)
    - Oscillator signals: 0.20 (1h:0.10, 15m:0.07, 5m:0.03)
    - Prophet ML prediction: 0.05
    - Sentiment: 0.25 (dynamic)
    """
    # Trend signals (sum = 0.45)
    trend_5m: float = 0.03
    trend_15m: float = 0.12
    trend_1h: float = 0.30
    # Oscillator signals (sum = 0.20)
    oscillator_5m: float = 0.03
    oscillator_15m: float = 0.07
    oscillator_1h: float = 0.10
    # Prophet ML prediction
    prophet: float = 0.05
    # Sentiment (dynamic - only used when data available)
    sentiment: float = 0.25


@dataclass
class TradeRecord:
    """Trade record for overtrading guard"""
    symbol: str
    action: str
    timestamp: datetime
    pnl: float = 0.0


class OvertradingGuard:
    """
    Overtrading Guard - Prevents overtrading and consecutive losses

    Rules:
    - Same symbol: minimum 4 cycles between trades
    - Max 2 positions in 6 hours
    - After 2 consecutive losses: cooldown for 6 cycles
    """

    MIN_CYCLES_SAME_SYMBOL = 4
    MAX_POSITIONS_6H = 2
    LOSS_STREAK_COOLDOWN = 6
    CONSECUTIVE_LOSS_THRESHOLD = 2

    def __init__(self):
        self.trade_history: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.last_trade_cycle: Dict[str, int] = {}
        self.cooldown_until_cycle: int = 0

    def record_trade(self, symbol: str, action: str, pnl: float = 0.0, current_cycle: int = 0) -> None:
        """Record a trade"""
        self.trade_history.append(TradeRecord(
            symbol=symbol,
            action=action,
            timestamp=datetime.now(),
            pnl=pnl,
        ))
        self.last_trade_cycle[symbol] = current_cycle

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD:
                self.cooldown_until_cycle = current_cycle + self.LOSS_STREAK_COOLDOWN
        else:
            self.consecutive_losses = 0

    def can_open_position(self, symbol: str, current_cycle: int = 0) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check cooldown
        if current_cycle < self.cooldown_until_cycle:
            remaining = self.cooldown_until_cycle - current_cycle
            return False, f"Consecutive loss cooldown, {remaining} cycles remaining"

        # Check same symbol interval
        if symbol in self.last_trade_cycle:
            cycles_since = current_cycle - self.last_trade_cycle[symbol]
            if cycles_since < self.MIN_CYCLES_SAME_SYMBOL:
                return False, f"{symbol} trading interval not met, need {self.MIN_CYCLES_SAME_SYMBOL - cycles_since} more cycles"

        # Check 6-hour position limit
        six_hours_ago = datetime.now().timestamp() - 6 * 3600
        recent_opens = sum(
            1 for t in self.trade_history
            if t.timestamp.timestamp() > six_hours_ago and self._is_open_action(t.action)
        )
        if recent_opens >= self.MAX_POSITIONS_6H:
            return False, f"6-hour position limit reached ({self.MAX_POSITIONS_6H})"

        return True, "Allowed"

    def _is_open_action(self, action: str) -> bool:
        """Check if action is an open position action"""
        return action in ('open_long', 'open_short')

    def get_status(self) -> Dict[str, Any]:
        """Get current guard status"""
        return {
            'consecutive_losses': self.consecutive_losses,
            'cooldown_until': self.cooldown_until_cycle,
            'recent_trades': len(self.trade_history),
            'symbols_traded': list(self.last_trade_cycle.keys()),
        }


@dataclass
class VoteResult:
    """Result from decision voting"""
    action: str  # 'open_long', 'open_short', 'wait', 'hold'
    confidence: float  # 0-100
    weighted_score: float  # -100 to +100
    vote_details: Dict[str, float]  # Individual signal contributions
    multi_period_aligned: bool  # Whether timeframes are aligned
    reason: str  # Human-readable decision reason
    regime: Optional[Dict[str, Any]] = None  # Market regime info
    position: Optional[Dict[str, Any]] = None  # Price position info
    trade_params: Optional[Dict[str, Any]] = None  # Dynamic trade parameters
    traps: Optional[Dict[str, Any]] = None  # Market trap warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'weighted_score': self.weighted_score,
            'vote_details': self.vote_details,
            'multi_period_aligned': self.multi_period_aligned,
            'reason': self.reason,
            'regime': self.regime,
            'position': self.position,
            'trade_params': self.trade_params,
            'traps': self.traps,
        }


class DecisionCore:
    """
    The Critic - Decision Core Agent

    Core functionality:
    - Weighted voting: Integrates multiple signals with configurable weights
    - Multi-period alignment: Detects consistency across timeframes
    - Market awareness: Regime detection and position analysis
    - Trap detection: Identifies market manipulation patterns
    - Confidence calibration: Dynamic confidence adjustment
    """

    def __init__(self, weights: Optional[SignalWeight] = None):
        """
        Initialize DecisionCore.

        Args:
            weights: Custom signal weights (uses defaults if None)
        """
        self.weights = weights or SignalWeight()
        self.history: List[VoteResult] = []
        self.current_cycle = 0
        self.overtrading_guard = OvertradingGuard()

        # Performance tracking for adaptive weights
        self.performance_tracker: Dict[str, Dict[str, int]] = {
            'trend_5m': {'total': 0, 'correct': 0},
            'trend_15m': {'total': 0, 'correct': 0},
            'trend_1h': {'total': 0, 'correct': 0},
            'oscillator_5m': {'total': 0, 'correct': 0},
            'oscillator_15m': {'total': 0, 'correct': 0},
            'oscillator_1h': {'total': 0, 'correct': 0},
        }

    def decide(
        self,
        quant_analysis: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        predict_result: Optional[Dict[str, Any]] = None,
        bull_perspective: Optional[Dict[str, Any]] = None,
        bear_perspective: Optional[Dict[str, Any]] = None,
        reflection: Optional[str] = None,
    ) -> VoteResult:
        """
        Make a trading decision using weighted voting.

        Args:
            quant_analysis: Quant analysis with trend, oscillator, sentiment scores
            market_data: Optional raw market data (df_5m, df_15m, df_1h, current_price)
            predict_result: Optional ML prediction result
            bull_perspective: Optional bull agent perspective
            bear_perspective: Optional bear agent perspective
            reflection: Optional reflection text

        Returns:
            VoteResult with action, confidence, and reason
        """
        self.current_cycle += 1
        symbol = quant_analysis.get('symbol', 'UNKNOWN')

        # Check overtrading guard
        can_trade, trade_reason = self.overtrading_guard.can_open_position(symbol, self.current_cycle)

        # Extract signals
        trend_data = quant_analysis.get('trend', {})
        osc_data = quant_analysis.get('oscillator', {})
        sentiment_data = quant_analysis.get('sentiment', {})
        traps = quant_analysis.get('traps', {})

        scores = {
            'trend_5m': float(trend_data.get('trend_5m_score', 0) or 0),
            'trend_15m': float(trend_data.get('trend_15m_score', 0) or 0),
            'trend_1h': float(trend_data.get('trend_1h_score', 0) or 0),
            'oscillator_5m': float(osc_data.get('osc_5m_score', 0) or 0),
            'oscillator_15m': float(osc_data.get('osc_15m_score', 0) or 0),
            'oscillator_1h': float(osc_data.get('osc_1h_score', 0) or 0),
            'sentiment': float(sentiment_data.get('total_sentiment_score', 0) or 0),
        }

        # Integrate Prophet prediction
        if predict_result:
            prob = predict_result.get('probability_up', 0.5)
            scores['prophet'] = (prob - 0.5) * 200
        else:
            scores['prophet'] = 0.0

        # Calculate dynamic sentiment weight
        has_sentiment = scores.get('sentiment', 0) != 0
        w_sentiment = self.weights.sentiment if has_sentiment else 0.0
        w_others = 1.0 - w_sentiment

        # Calculate weighted score
        weighted_score = (
            (scores['trend_5m'] * self.weights.trend_5m +
             scores['trend_15m'] * self.weights.trend_15m +
             scores['trend_1h'] * self.weights.trend_1h +
             scores['oscillator_5m'] * self.weights.oscillator_5m +
             scores['oscillator_15m'] * self.weights.oscillator_15m +
             scores['oscillator_1h'] * self.weights.oscillator_1h +
             scores.get('prophet', 0) * self.weights.prophet) * w_others +
            (scores.get('sentiment', 0) * w_sentiment)
        )

        # Calculate vote details
        vote_details = {
            'trend_5m': scores['trend_5m'] * self.weights.trend_5m * w_others,
            'trend_15m': scores['trend_15m'] * self.weights.trend_15m * w_others,
            'trend_1h': scores['trend_1h'] * self.weights.trend_1h * w_others,
            'oscillator_5m': scores['oscillator_5m'] * self.weights.oscillator_5m * w_others,
            'oscillator_15m': scores['oscillator_15m'] * self.weights.oscillator_15m * w_others,
            'oscillator_1h': scores['oscillator_1h'] * self.weights.oscillator_1h * w_others,
            'prophet': scores.get('prophet', 0) * self.weights.prophet * w_others,
            'sentiment': scores.get('sentiment', 0) * w_sentiment,
        }

        # Check alignment
        aligned, alignment_reason = self._check_alignment(
            scores['trend_1h'],
            scores['trend_15m'],
            scores['trend_5m'],
        )

        # Get regime and position from market data
        regime = self._detect_regime(market_data) if market_data else None
        position = self._analyze_position(market_data) if market_data else None

        # Volume ratio check
        volume_ratio = self._get_volume_ratio(
            market_data.get('df_5m') if market_data else None
        ) if market_data else None

        # Choppy market handling
        is_choppy = regime and regime.get('regime') in ('choppy', 'volatile_directionless', 'ranging')
        if is_choppy and position:
            if position.get('location') == 'middle' and abs(weighted_score) < 30:
                result = VoteResult(
                    action='hold',
                    confidence=10.0,
                    weighted_score=0,
                    vote_details=vote_details,
                    multi_period_aligned=False,
                    reason=f"Choppy market + middle position, no entry",
                    regime=regime,
                    position=position,
                )
                self.history.append(result)
                return result

        # Score to action
        action, base_confidence = self._score_to_action(weighted_score, aligned, regime)

        # Low volume filter
        if self._is_open_action(action) and volume_ratio is not None and volume_ratio < 0.5:
            action = 'hold'
            base_confidence = 0.1
            alignment_reason = f"Low volume filter (RVOL {volume_ratio:.2f} < 0.5)"

        # Overtrading guard
        if self._is_open_action(action) and not can_trade:
            action = 'hold'
            base_confidence = 0.1
            alignment_reason = trade_reason

        # Trap filtering
        if self._is_open_action(action):
            action, base_confidence, alignment_reason = self._apply_trap_filter(
                action, traps, base_confidence, alignment_reason
            )

        # Bull/Bear adversarial adjustment
        if bull_perspective and bear_perspective:
            base_confidence = self._apply_bull_bear_adjustment(
                bull_perspective, bear_perspective, base_confidence, aligned
            )

        # Calculate final confidence
        final_confidence = base_confidence * 100

        # Regime and position adjustments
        if regime and position:
            final_confidence = self._calculate_comprehensive_confidence(
                final_confidence, regime, position, aligned
            )

        # Generate reason
        reason = self._generate_reason(
            weighted_score, aligned, alignment_reason, quant_analysis,
            regime=regime
        )

        # Calculate trade parameters
        trade_params = self._calculate_trade_params(
            regime, position, final_confidence, action
        )

        result = VoteResult(
            action=action,
            confidence=final_confidence,
            weighted_score=weighted_score,
            vote_details=vote_details,
            multi_period_aligned=aligned,
            reason=reason,
            regime=regime,
            position=position,
            trade_params=trade_params,
            traps=traps,
        )

        self.history.append(result)
        return result

    def _check_alignment(
        self,
        score_1h: float,
        score_15m: float,
        score_5m: float,
    ) -> Tuple[bool, str]:
        """
        Check multi-period alignment.

        Returns:
            (aligned: bool, reason: str)
        """
        signs = [
            1 if score_1h >= 25 else (-1 if score_1h <= -25 else 0),
            1 if score_15m >= 18 else (-1 if score_15m <= -18 else 0),
            1 if score_5m >= 12 else (-1 if score_5m <= -12 else 0),
        ]

        # All three aligned
        if signs[0] == signs[1] == signs[2] and signs[0] != 0:
            direction = 'bull' if signs[0] > 0 else 'bear'
            return True, f"Triple timeframe {direction} alignment"

        # 1h and 15m aligned
        if signs[0] == signs[1] and signs[0] != 0:
            direction = 'bull' if signs[0] > 0 else 'bear'
            return True, f"1h+15m {direction} alignment"

        return False, f"Misaligned (1h:{signs[0]}, 15m:{signs[1]}, 5m:{signs[2]}), waiting for 1h confirmation"

    def _detect_regime(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect market regime from market data"""
        df_5m = market_data.get('df_5m')
        if df_5m is None or len(df_5m) < 20:
            return None

        # Simple ADX estimation
        adx = self._estimate_adx(df_5m)

        if adx >= 25:
            regime_type = 'trending'
        elif adx >= 15:
            regime_type = 'volatile_directionless'
        else:
            regime_type = 'choppy'

        return {
            'regime': regime_type,
            'adx': adx,
            'confidence': 75.0,
            'reason': f"Regime: {regime_type} (ADX {adx:.1f})",
        }

    def _estimate_adx(self, df) -> float:
        """Estimate ADX from data"""
        if df is None or len(df) < 14:
            return 20.0

        try:
            # Simplified ADX estimation using momentum
            closes = df['close'].values if 'close' in df.columns else []
            if len(closes) < 14:
                return 20.0

            # Calculate average directional movement
            up_moves = []
            down_moves = []
            for i in range(1, min(15, len(closes))):
                diff = closes[i] - closes[i-1]
                if diff > 0:
                    up_moves.append(diff)
                    down_moves.append(0)
                else:
                    up_moves.append(0)
                    down_moves.append(abs(diff))

            avg_up = sum(up_moves) / len(up_moves) if up_moves else 0
            avg_down = sum(down_moves) / len(down_moves) if down_moves else 0

            if avg_down == 0:
                return 50.0

            dx = abs((avg_up - avg_down) / (avg_up + avg_down)) * 100
            return min(dx * 1.5, 100)  # Scale and cap
        except Exception:
            return 20.0

    def _analyze_position(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze price position in range"""
        df_5m = market_data.get('df_5m')
        current_price = market_data.get('current_price')

        if df_5m is None or current_price is None:
            return None

        try:
            # Calculate position in recent range
            recent_high = float(df_5m['high'].iloc[-20:].max())
            recent_low = float(df_5m['low'].iloc[-20:].min())
            range_size = recent_high - recent_low

            if range_size > 0:
                position_pct = (current_price - recent_low) / range_size * 100
            else:
                position_pct = 50.0

            if position_pct >= 80:
                location = 'high'
                quality = 'poor' if position_pct >= 90 else 'average'
            elif position_pct <= 20:
                location = 'low'
                quality = 'excellent' if position_pct <= 8 else 'average'
            else:
                location = 'middle'
                quality = 'average'

            return {
                'position_pct': position_pct,
                'location': location,
                'quality': quality,
                'recent_high': recent_high,
                'recent_low': recent_low,
            }
        except Exception:
            return None

    def _get_volume_ratio(self, df, window: int = 20) -> Optional[float]:
        """Calculate volume ratio"""
        if df is None or 'volume' not in df.columns or len(df) < window:
            return None

        try:
            recent_vol = float(df['volume'].iloc[-1])
            avg_vol = float(df['volume'].iloc[-window:].mean())
            return recent_vol / avg_vol if avg_vol > 0 else None
        except Exception:
            return None

    def _score_to_action(
        self,
        weighted_score: float,
        aligned: bool,
        regime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """Convert weighted score to action and confidence"""
        # Dynamic thresholds based on regime
        long_threshold = 20
        short_threshold = 18

        if regime:
            regime_type = regime.get('regime', '').lower()
            if regime_type == 'trending_down':
                short_threshold = 18
                long_threshold = 32
            elif regime_type == 'trending_up':
                long_threshold = 22
                short_threshold = 32
            elif regime_type in ('volatile_directionless', 'choppy', 'ranging'):
                long_threshold = 30
                short_threshold = 30

        # Aligned = lower threshold
        if aligned:
            long_threshold = max(12, long_threshold - 2)
            short_threshold = max(12, short_threshold - 2)

        # Strong signal
        if weighted_score > long_threshold + 15 and aligned:
            return 'open_long', 0.85
        if weighted_score < -(short_threshold + 15) and aligned:
            return 'open_short', 0.85

        # Medium signal
        if weighted_score > long_threshold:
            confidence = 0.55 + (weighted_score - long_threshold) * 0.01
            return 'open_long', min(confidence, 0.75)
        if weighted_score < -short_threshold:
            confidence = 0.55 + (abs(weighted_score) - short_threshold) * 0.01
            return 'open_short', min(confidence, 0.75)

        return 'wait', abs(weighted_score) / 100

    def _is_open_action(self, action: str) -> bool:
        """Check if action is an open position action"""
        return action in ('open_long', 'open_short')

    def _apply_trap_filter(
        self,
        action: str,
        traps: Dict[str, Any],
        base_confidence: float,
        alignment_reason: str,
    ) -> Tuple[str, float, str]:
        """Apply trap detection filters"""
        if not traps:
            return action, base_confidence, alignment_reason

        # Bull trap risk
        if traps.get('bull_trap_risk') and action == 'open_long':
            return 'hold', 0.1, "Bull trap risk detected, no long entry"

        # Weak rebound
        if traps.get('weak_rebound') and action == 'open_long':
            base_confidence *= 0.5
            alignment_reason += " | Weak rebound warning"
            if base_confidence < 0.6:
                return 'hold', base_confidence, "Weak rebound, insufficient confidence for long"

        # Volume divergence
        if traps.get('volume_divergence'):
            if action == 'open_long':
                base_confidence *= 0.7
                alignment_reason += " | Volume divergence warning (high price, low volume)"
            elif action == 'open_short':
                base_confidence = min(base_confidence * 1.2, 0.95)
                alignment_reason += " | Volume divergence confirmed for short"

        # Accumulation
        if traps.get('accumulation') and action == 'open_long':
            base_confidence = min(base_confidence * 1.2, 0.95)
            alignment_reason += " | Accumulation detected"

        # Panic bottom
        if traps.get('panic_bottom') and action == 'open_long':
            base_confidence = min(base_confidence * 1.3, 0.95)
            alignment_reason += " | Panic bottom opportunity"
        elif traps.get('panic_bottom') and action == 'open_short':
            return 'hold', 0.1, "Panic bottom detected, no short entry"

        # FOMO top
        if traps.get('fomo_top') and action == 'open_short':
            base_confidence = min(base_confidence * 1.3, 0.95)
            alignment_reason += " | FOMO top opportunity"
        elif traps.get('fomo_top') and action == 'open_long':
            return 'hold', 0.1, "FOMO top detected, no long entry"

        return action, base_confidence, alignment_reason

    def _apply_bull_bear_adjustment(
        self,
        bull_perspective: Dict[str, Any],
        bear_perspective: Dict[str, Any],
        base_confidence: float,
        aligned: bool,
    ) -> float:
        """Apply Bull/Bear adversarial adjustment"""
        bull_conf = bull_perspective.get('bull_confidence', 50)
        bear_conf = bear_perspective.get('bear_confidence', 50)

        # Strong resonance on one side
        if bull_conf > 60 or bear_conf > 60:
            base_confidence += 0.10

        # Conflicting perspectives
        elif 40 <= bull_conf <= 60 and 40 <= bear_conf <= 60:
            base_confidence -= 0.10

        return min(base_confidence, 1.0)

    def _calculate_comprehensive_confidence(
        self,
        base_conf: float,
        regime: Dict[str, Any],
        position: Dict[str, Any],
        aligned: bool,
    ) -> float:
        """Calculate comprehensive confidence with regime/position adjustments"""
        conf = base_conf

        # Additions
        if aligned:
            conf += 15
        if regime.get('regime') in ('trending_up', 'trending_down'):
            conf += 10
        if position.get('quality') == 'excellent':
            conf += 15

        # Subtractions
        if regime.get('regime') == 'choppy':
            conf -= 25
        if position.get('location') == 'middle':
            conf -= 30
        if regime.get('regime') == 'volatile_directionless':
            conf -= 20

        return max(5.0, min(100.0, conf))

    def _calculate_trade_params(
        self,
        regime: Optional[Dict[str, Any]],
        position: Optional[Dict[str, Any]],
        confidence: float,
        action: str,
    ) -> Dict[str, Any]:
        """Calculate dynamic trade parameters"""
        base_size = 100.0
        base_stop_loss = 1.5
        base_take_profit = 3.0

        size_mult = 1.0
        sl_mult = 1.0
        tp_mult = 1.0

        # Regime adjustments
        if regime:
            regime_type = regime.get('regime', '').lower()
            if 'volatile' in regime_type:
                size_mult *= 0.5
                sl_mult *= 1.5
                tp_mult *= 1.5
            elif regime_type in ('trending_up', 'trending_down'):
                size_mult *= 1.2
                tp_mult *= 1.5
            elif regime_type in ('choppy', 'ranging'):
                size_mult *= 0.7
                sl_mult *= 0.5
                tp_mult *= 0.4

        # Position adjustments
        if position:
            quality = position.get('quality', 'average')
            if quality == 'excellent':
                size_mult *= 1.3
            elif quality == 'poor':
                size_mult *= 0.5

        # Confidence adjustments
        if confidence > 70:
            size_mult *= min(confidence / 70, 1.5)
        elif confidence < 50:
            size_mult *= 0.7

        # Hold = no position
        if action == 'hold':
            size_mult = 0

        return {
            'position_size': round(base_size * size_mult, 2),
            'stop_loss_pct': round(base_stop_loss * sl_mult, 2),
            'take_profit_pct': round(base_take_profit * tp_mult, 2),
            'leverage_suggested': 1 if size_mult < 0.8 else (2 if size_mult > 1.2 else 1),
            'reason': f"size_mult={size_mult:.2f}, sl_mult={sl_mult:.2f}, tp_mult={tp_mult:.2f}",
        }

    def _generate_reason(
        self,
        weighted_score: float,
        aligned: bool,
        alignment_reason: str,
        quant_analysis: Dict[str, Any],
        regime: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate human-readable decision reason"""
        reasons = []

        # Regime
        if regime:
            regime_name = regime.get('regime', 'unknown').upper()
            reasons.append(f"[{regime_name}]")

        # Score
        reasons.append(f"Score: {weighted_score:.1f}")

        # Alignment
        reasons.append(f"Align: {alignment_reason}")

        # Top signals
        trend_data = quant_analysis.get('trend', {})
        osc_data = quant_analysis.get('oscillator', {})

        vote_details = {
            'trend_1h': trend_data.get('trend_1h_score', 0),
            'trend_15m': trend_data.get('trend_15m_score', 0),
            'oscillator_1h': osc_data.get('osc_1h_score', 0),
            'oscillator_15m': osc_data.get('osc_15m_score', 0),
        }

        sorted_signals = sorted(
            vote_details.items(),
            key=lambda x: abs(float(x[1]) if x[1] else 0),
            reverse=True,
        )[:2]

        for sig_name, sig_score in sorted_signals:
            score = float(sig_score) if sig_score else 0
            if abs(score) > 20:
                reasons.append(f"{sig_name}: {score:+.0f}")

        return " | ".join(reasons)

    def update_performance(self, signal_name: str, is_correct: bool) -> None:
        """Update signal performance for adaptive weights"""
        if signal_name in self.performance_tracker:
            self.performance_tracker[signal_name]['total'] += 1
            if is_correct:
                self.performance_tracker[signal_name]['correct'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics"""
        if not self.history:
            return {'total_decisions': 0}

        total = len(self.history)
        actions = [h.action for h in self.history]
        avg_confidence = sum(h.confidence for h in self.history) / total
        aligned_count = sum(1 for h in self.history if h.multi_period_aligned)

        return {
            'total_decisions': total,
            'action_distribution': {
                'open_long': actions.count('open_long'),
                'open_short': actions.count('open_short'),
                'wait': actions.count('wait'),
                'hold': actions.count('hold'),
            },
            'avg_confidence': avg_confidence,
            'alignment_rate': aligned_count / total,
            'performance_tracker': self.performance_tracker,
        }

    def record_decision_outcome(
        self,
        symbol: str,
        action: str,
        pnl: float,
        was_correct: Optional[bool] = None,
    ) -> None:
        """Record the outcome of a decision for performance tracking"""
        self.overtrading_guard.record_trade(
            symbol=symbol,
            action=action,
            pnl=pnl,
            current_cycle=self.current_cycle,
        )
