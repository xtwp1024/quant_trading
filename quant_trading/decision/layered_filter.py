"""
Four-Layer Strategy Filter
==========================

Centralizes the "Four-Layer Strategy" logic from LLM-TradeBot.

Pipeline:
    Layer 1: Trend + Fuel (1h EMA + Volume Proxy/OI)
    Layer 2: AI Filter (optional ML prediction alignment)
    Layer 3: Setup (15m KDJ + Bollinger Bands entry zone)
    Layer 4: Trigger (5m Pattern + RVOL volume confirmation)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class LayerStatus(Enum):
    """Layer pass/fail status"""
    PASS = "pass"
    FAIL = "fail"
    WAIT = "wait"


@dataclass
class LayerResult:
    """Result from a single layer evaluation"""
    layer_name: str
    status: LayerStatus
    blocking_reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class FourLayerResult:
    """Result from the four-layer strategy filter"""
    layer1_pass: bool
    layer2_pass: bool
    layer3_pass: bool
    layer4_pass: bool
    final_action: str  # 'long', 'short', 'wait'
    blocking_reason: Optional[str] = None
    confidence_boost: float = 0.0
    tp_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    adx: float = 20.0
    regime: str = "unknown"
    kdj_j: float = 50.0
    kdj_zone: str = "neutral"
    bb_position: str = "middle"
    setup_quality: str = "ACCEPTABLE"
    trigger_pattern: str = "None"
    trigger_rvol: float = 1.0
    atr_multiplier: float = 1.0
    atr_pct: float = 0.0
    funding_rate: float = 0.0
    oi_change: float = 0.0
    trend_1h: str = "neutral"
    ai_prediction_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'layer1_pass': self.layer1_pass,
            'layer2_pass': self.layer2_pass,
            'layer3_pass': self.layer3_pass,
            'layer4_pass': self.layer4_pass,
            'final_action': self.final_action,
            'blocking_reason': self.blocking_reason,
            'confidence_boost': self.confidence_boost,
            'tp_multiplier': self.tp_multiplier,
            'sl_multiplier': self.sl_multiplier,
            'adx': self.adx,
            'regime': self.regime,
            'kdj_j': self.kdj_j,
            'kdj_zone': self.kdj_zone,
            'bb_position': self.bb_position,
            'setup_quality': self.setup_quality,
            'trigger_pattern': self.trigger_pattern,
            'trigger_rvol': self.trigger_rvol,
            'atr_multiplier': self.atr_multiplier,
            'atr_pct': self.atr_pct,
            'funding_rate': self.funding_rate,
            'oi_change': self.oi_change,
            'trend_1h': self.trend_1h,
            'ai_prediction_note': self.ai_prediction_note,
        }


class FourLayerStrategyFilter:
    """
    Four-Layer Strategy Filter

    Evaluates market conditions across four layers before allowing a trade:
    1. Trend + Fuel: 1h EMA alignment + OI/volume confirmation
    2. AI Filter: Optional ML prediction alignment check
    3. Setup: 15m KDJ + Bollinger Bands zone
    4. Trigger: 5m pattern + RVOL confirmation
    """

    def __init__(self, use_ai_filter: bool = True):
        """
        Initialize the four-layer filter.

        Args:
            use_ai_filter: Whether to use AI/ML prediction filter (Layer 2)
        """
        self.use_ai_filter = use_ai_filter

    def evaluate(
        self,
        df_1h: Any,
        df_15m: Any,
        df_5m: Any,
        current_price: float,
        quant_analysis: Dict[str, Any],
        predict_result: Optional[Dict[str, Any]] = None,
    ) -> FourLayerResult:
        """
        Evaluate market conditions across all four layers.

        Args:
            df_1h: 1h dataframe with OHLCV + indicators
            df_15m: 15m dataframe with OHLCV + indicators
            df_5m: 5m dataframe with OHLCV + indicators
            current_price: Current market price
            quant_analysis: Quant analysis output containing trend, oscillator, sentiment
            predict_result: Optional ML prediction result for Layer 2

        Returns:
            FourLayerResult with layer pass/fail and adjustments
        """
        result = FourLayerResult(
            layer1_pass=False,
            layer2_pass=False,
            layer3_pass=False,
            layer4_pass=False,
            final_action="wait",
        )

        # Get sentiment data
        sentiment = quant_analysis.get('sentiment', {})
        oi_fuel = sentiment.get('oi_fuel', {})
        funding_rate = sentiment.get('details', {}).get('funding_rate', 0) or 0
        result.funding_rate = funding_rate

        # Get ADX from quant analysis or estimate
        adx_value = quant_analysis.get('regime', {}).get('adx', 20) if isinstance(quant_analysis.get('regime'), dict) else 20
        result.adx = adx_value

        # Get OI change
        oi_change = oi_fuel.get('oi_change_24h', 0) or 0
        result.oi_change = oi_change

        # ========== LAYER 1: Trend + Fuel ==========
        layer1_result = self._evaluate_layer1(df_1h, current_price, oi_change, adx_value, oi_fuel)
        result.layer1_pass = layer1_result.status == LayerStatus.PASS
        if layer1_result.blocking_reason:
            result.blocking_reason = layer1_result.blocking_reason
        result.trend_1h = layer1_result.details.get('trend_1h', 'neutral') if layer1_result.details else 'neutral'
        result.regime = layer1_result.details.get('regime', 'unknown') if layer1_result.details else 'unknown'

        if not result.layer1_pass:
            result.final_action = "wait"
            return result

        # ========== LAYER 2: AI Filter ==========
        if self.use_ai_filter and predict_result is not None:
            layer2_result = self._evaluate_layer2(result.trend_1h, predict_result, adx_value)
            result.layer2_pass = layer2_result.status == LayerStatus.PASS
            if layer2_result.blocking_reason:
                result.blocking_reason = layer2_result.blocking_reason
            result.confidence_boost = layer2_result.details.get('confidence_boost', 0) if layer2_result.details else 0
            result.ai_prediction_note = layer2_result.details.get('ai_prediction_note') if layer2_result.details else None
        else:
            # Skip AI filter if not available
            result.layer2_pass = True

        if not result.layer2_pass:
            result.final_action = "wait"
            return result

        # ========== LAYER 3: Setup (15m) ==========
        layer3_result = self._evaluate_layer3(df_15m, result.trend_1h)
        result.layer3_pass = layer3_result.status == LayerStatus.PASS
        if layer3_result.blocking_reason:
            result.blocking_reason = layer3_result.blocking_reason
        if layer3_result.details:
            result.kdj_j = layer3_result.details.get('kdj_j', 50)
            result.kdj_zone = layer3_result.details.get('kdj_zone', 'neutral')
            result.bb_position = layer3_result.details.get('bb_position', 'middle')
            result.setup_quality = layer3_result.details.get('setup_quality', 'ACCEPTABLE')

        if not result.layer3_pass:
            result.final_action = "wait"
            return result

        # ========== LAYER 4: Trigger (5m) ==========
        layer4_result = self._evaluate_layer4(df_5m, result.trend_1h)
        result.layer4_pass = layer4_result.status == LayerStatus.PASS
        if layer4_result.blocking_reason:
            result.blocking_reason = layer4_result.blocking_reason
        if layer4_result.details:
            result.trigger_pattern = layer4_result.details.get('pattern_type', 'None')
            result.trigger_rvol = layer4_result.details.get('rvol', 1.0)

        if not result.layer4_pass:
            result.final_action = "wait"
            return result

        # All layers passed - determine final action
        result.final_action = result.trend_1h

        # Apply sentiment adjustments
        sent_score = sentiment.get('total_sentiment_score', 0)
        if sent_score > 80:
            result.tp_multiplier = 0.8
        elif sent_score < -80:
            result.tp_multiplier = 1.5
            result.sl_multiplier = 0.8

        # Apply funding rate adjustments
        if result.trend_1h == 'long' and funding_rate > 0.05:
            result.tp_multiplier *= 0.7
        elif result.trend_1h == 'short' and funding_rate < -0.05:
            result.tp_multiplier *= 0.7

        # Apply ATR-based dynamic adjustment
        atr_result = self._calculate_atr_multiplier(df_1h)
        result.atr_multiplier = atr_result['multiplier']
        result.atr_pct = atr_result['atr_pct']
        result.tp_multiplier *= atr_result['multiplier']
        result.sl_multiplier *= atr_result['multiplier']

        # Minimum TP filter (1.5%)
        expected_tp_pct = 2.5 * result.tp_multiplier
        if expected_tp_pct < 1.5:
            result.layer4_pass = False
            result.final_action = "wait"
            result.blocking_reason = f"Expected TP {expected_tp_pct:.1f}% < minimum 1.5%"
            return result

        # Minimum risk-reward filter (2:1)
        expected_sl_pct = 1.0 * result.sl_multiplier
        if expected_sl_pct > 0 and (expected_tp_pct / expected_sl_pct) < 2.0:
            result.layer4_pass = False
            result.final_action = "wait"
            result.blocking_reason = f"Risk:Reward {expected_tp_pct/expected_sl_pct:.1f}:1 < 2:1"

        return result

    def _evaluate_layer1(
        self,
        df_1h: Any,
        current_price: float,
        oi_change: float,
        adx_value: float,
        oi_fuel: Dict[str, Any],
    ) -> LayerResult:
        """Layer 1: Trend + Fuel evaluation"""
        if df_1h is None or len(df_1h) < 20:
            close_1h = current_price
            ema20_1h = current_price
            ema60_1h = current_price
        else:
            close_1h = float(df_1h['close'].iloc[-1])
            ema20_1h = float(df_1h['ema_20'].iloc[-1]) if 'ema_20' in df_1h.columns else close_1h
            ema60_1h = float(df_1h['ema_60'].iloc[-1]) if 'ema_60' in df_1h.columns else close_1h

        # Determine trend direction
        trend_1h = self._determine_trend(df_1h, close_1h, ema20_1h, ema60_1h)
        regime = self._determine_regime(adx_value, trend_1h)

        # Layer 1 checks
        if trend_1h == 'neutral':
            return LayerResult(
                layer_name="Layer1_Trend",
                status=LayerStatus.FAIL,
                blocking_reason="No clear 1h trend (EMA 20/60)",
                details={'trend_1h': 'neutral', 'regime': regime},
            )

        if adx_value < 15:
            return LayerResult(
                layer_name="Layer1_Trend",
                status=LayerStatus.FAIL,
                blocking_reason=f"Weak Trend Strength (ADX {adx_value:.0f} < 15)",
                details={'trend_1h': trend_1h, 'regime': regime},
            )

        # OI divergence check (warning only, not blocking)
        if trend_1h == 'long' and oi_change < -5.0:
            pass  # Warning but allow

        if trend_1h == 'short' and oi_change > 5.0:
            pass  # Warning but allow

        # Whale trap detection
        if trend_1h == 'long' and oi_fuel.get('whale_trap_risk', False):
            return LayerResult(
                layer_name="Layer1_Trend",
                status=LayerStatus.FAIL,
                blocking_reason=f"Whale trap detected (OI {oi_change:.1f}%)",
                details={'trend_1h': trend_1h, 'regime': regime},
            )

        return LayerResult(
            layer_name="Layer1_Trend",
            status=LayerStatus.PASS,
            details={'trend_1h': trend_1h, 'regime': regime, 'adx': adx_value},
        )

    def _evaluate_layer2(
        self,
        trend_1h: str,
        predict_result: Dict[str, Any],
        adx_value: float,
    ) -> LayerResult:
        """Layer 2: AI/ML prediction alignment filter"""
        if predict_result is None:
            return LayerResult(layer_name="Layer2_AI", status=LayerStatus.PASS)

        # Check if ADX is too low for reliable prediction
        if adx_value < 5:
            return LayerResult(
                layer_name="Layer2_AI",
                status=LayerStatus.FAIL,
                blocking_reason="AI prediction invalidated: ADX<5",
                details={'ai_prediction_note': 'AI prediction invalidated: ADX<5'},
            )

        # Get prediction direction
        prob_up = predict_result.get('probability_up', 0.5)
        predicted_direction = 'long' if prob_up > 0.55 else ('short' if prob_up < 0.45 else 'neutral')

        # Check alignment with trend
        if trend_1h == 'neutral':
            return LayerResult(
                layer_name="Layer2_AI",
                status=LayerStatus.WAIT,
                blocking_reason="No clear trend for AI alignment check",
            )

        # Veto if strong disagreement
        if trend_1h == 'long' and prob_up < 0.4:
            return LayerResult(
                layer_name="Layer2_AI",
                status=LayerStatus.FAIL,
                blocking_reason="AI prediction veto: strongly bearish while trend is long",
                details={'confidence_boost': -10},
            )

        if trend_1h == 'short' and prob_up > 0.6:
            return LayerResult(
                layer_name="Layer2_AI",
                status=LayerStatus.FAIL,
                blocking_reason="AI prediction veto: strongly bullish while trend is short",
                details={'confidence_boost': -10},
            )

        # Boost confidence if aligned
        confidence_boost = 5 if predicted_direction == trend_1h else 0

        return LayerResult(
            layer_name="Layer2_AI",
            status=LayerStatus.PASS,
            details={'confidence_boost': confidence_boost},
        )

    def _evaluate_layer2_simple(self, trend_1h: str, quant_analysis: Dict[str, Any]) -> LayerResult:
        """Simplified Layer 2 evaluation without ML prediction"""
        # Skip AI filter if no prediction available
        return LayerResult(layer_name="Layer2_AI", status=LayerStatus.PASS)

    def _evaluate_layer3(self, df_15m: Any, trend_1h: str) -> LayerResult:
        """Layer 3: 15m Setup (KDJ + Bollinger Bands)"""
        if df_15m is None or len(df_15m) < 20:
            return LayerResult(
                layer_name="Layer3_Setup",
                status=LayerStatus.FAIL,
                blocking_reason="Insufficient 15m data",
            )

        close_15m = float(df_15m['close'].iloc[-1])
        bb_middle = float(df_15m['bb_middle'].iloc[-1]) if 'bb_middle' in df_15m.columns else close_15m
        bb_upper = float(df_15m['bb_upper'].iloc[-1]) if 'bb_upper' in df_15m.columns else close_15m * 1.02
        bb_lower = float(df_15m['bb_lower'].iloc[-1]) if 'bb_lower' in df_15m.columns else close_15m * 0.98
        kdj_j = float(df_15m['kdj_j'].iloc[-1]) if 'kdj_j' in df_15m.columns else 50

        # Determine KDJ zone
        kdj_zone = "neutral"
        if kdj_j > 80 or close_15m > bb_upper:
            kdj_zone = "overbought"
        elif kdj_j < 20 or close_15m < bb_lower:
            kdj_zone = "oversold"

        bb_position = "upper" if close_15m > bb_upper else "lower" if close_15m < bb_lower else "middle"

        details = {
            'kdj_j': kdj_j,
            'kdj_zone': kdj_zone,
            'bb_position': bb_position,
        }

        # Evaluate based on trend direction
        if trend_1h == 'long':
            if kdj_zone == 'overbought':
                return LayerResult(
                    layer_name="Layer3_Setup",
                    status=LayerStatus.FAIL,
                    blocking_reason=f"15m overbought (J={kdj_j:.0f}) - wait for pullback",
                    details={**details, 'setup_quality': 'REJECTED'},
                )
            # Setup ready
            setup_quality = 'IDEAL' if (close_15m < bb_middle or kdj_j < 50) else 'ACCEPTABLE'
            return LayerResult(
                layer_name="Layer3_Setup",
                status=LayerStatus.PASS,
                details={**details, 'setup_quality': setup_quality},
            )

        elif trend_1h == 'short':
            if kdj_zone == 'oversold':
                return LayerResult(
                    layer_name="Layer3_Setup",
                    status=LayerStatus.FAIL,
                    blocking_reason=f"15m oversold (J={kdj_j:.0f}) - wait for rally",
                    details={**details, 'setup_quality': 'REJECTED'},
                )
            # Setup ready
            setup_quality = 'IDEAL' if (close_15m > bb_middle or kdj_j > 50) else 'ACCEPTABLE'
            return LayerResult(
                layer_name="Layer3_Setup",
                status=LayerStatus.PASS,
                details={**details, 'setup_quality': setup_quality},
            )

        return LayerResult(
            layer_name="Layer3_Setup",
            status=LayerStatus.WAIT,
            blocking_reason="No clear trend for setup evaluation",
            details=details,
        )

    def _evaluate_layer4(self, df_5m: Any, trend_1h: str) -> LayerResult:
        """Layer 4: 5m Trigger (Pattern + RVOL)"""
        if df_5m is None or len(df_5m) < 10:
            return LayerResult(
                layer_name="Layer4_Trigger",
                status=LayerStatus.FAIL,
                blocking_reason="Insufficient 5m data",
            )

        # Calculate RVOL (Relative Volume)
        if 'volume' in df_5m.columns and len(df_5m) >= 20:
            recent_vol = float(df_5m['volume'].iloc[-1])
            avg_vol = float(df_5m['volume'].iloc[-20:].mean())
            rvol = recent_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            rvol = 1.0

        # Simple pattern detection (placeholder - could be enhanced)
        pattern_type = self._detect_pattern(df_5m, trend_1h)

        # Trigger requires RVOL > 1.0 (above average volume)
        if rvol < 1.0:
            return LayerResult(
                layer_name="Layer4_Trigger",
                status=LayerStatus.FAIL,
                blocking_reason=f"5min trigger not confirmed (RVOL={rvol:.1f}x < 1.0x)",
                details={'pattern_type': pattern_type, 'rvol': rvol},
            )

        return LayerResult(
            layer_name="Layer4_Trigger",
            status=LayerStatus.PASS,
            details={'pattern_type': pattern_type, 'rvol': rvol},
        )

    def _determine_trend(
        self,
        df_1h: Any,
        close: float,
        ema20: float,
        ema60: float,
    ) -> str:
        """Determine trend direction from EMA alignment"""
        # Primary: Strict EMA alignment
        if close > ema20 > ema60:
            return 'long'
        elif close < ema20 < ema60:
            return 'short'

        # Fallback: Check EMA20 slope
        if df_1h is not None and len(df_1h) >= 5 and 'ema_20' in df_1h.columns:
            ema20_3ago = float(df_1h['ema_20'].iloc[-4])
            ema_slope = (ema20 - ema20_3ago) / ema20_3ago * 100 if ema20_3ago > 0 else 0

            if ema_slope > 0.3 and close > ema20:
                return 'long'
            elif ema_slope < -0.3 and close < ema20:
                return 'short'

        return 'neutral'

    def _determine_regime(self, adx: float, trend: str) -> str:
        """Determine market regime from ADX and trend"""
        if adx >= 25:
            if trend == 'long':
                return 'trending_up'
            elif trend == 'short':
                return 'trending_down'
        elif adx >= 15:
            return 'volatile_directionless'
        else:
            return 'choppy'

    def _detect_pattern(self, df_5m: Any, trend: str) -> str:
        """Simple pattern detection for 5m timeframe"""
        if df_5m is None or len(df_5m) < 5:
            return 'None'

        # Simple momentum check
        closes = df_5m['close'].values if 'close' in df_5m.columns else []
        if len(closes) >= 5:
            recent_change = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] != 0 else 0
            if abs(recent_change) > 0.5:
                return f"momentum_{'up' if recent_change > 0 else 'down'}"

        return 'None'

    def _calculate_atr_multiplier(self, df_1h: Any) -> Dict[str, float]:
        """Calculate ATR-based multiplier for TP/SL adjustment"""
        if df_1h is None or len(df_1h) < 14:
            return {'multiplier': 1.0, 'atr_pct': 0.0, 'volatility': 'normal'}

        # Calculate ATR
        high = df_1h['high'].values if 'high' in df_1h.columns else df_1h['close'].values
        low = df_1h['low'].values if 'low' in df_1h.columns else df_1h['close'].values
        close = df_1h['close'].values if 'close' in df_1h.columns else df_1h['close'].values

        trs = []
        for i in range(1, min(15, len(high))):
            high_low = high[i] - low[i]
            high_close = abs(high[i] - close[i-1])
            low_close = abs(low[i] - close[i-1])
            trs.append(max(high_low, high_close, low_close))

        atr = sum(trs) / len(trs) if trs else 0
        current_price = close[-1]
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # Determine multiplier based on ATR percentage
        if atr_pct > 3:
            multiplier = 1.5
            volatility = 'high'
        elif atr_pct > 1.5:
            multiplier = 1.2
            volatility = 'elevated'
        elif atr_pct < 0.5:
            multiplier = 0.7
            volatility = 'low'
        else:
            multiplier = 1.0
            volatility = 'normal'

        return {'multiplier': multiplier, 'atr_pct': atr_pct, 'volatility': volatility}
