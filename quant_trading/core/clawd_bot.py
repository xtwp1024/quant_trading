import logging
import asyncio
import pandas as pd
import uuid
import pandas_ta as ta

from .pattern_engine import PatternEngine
from .sentiment import SentimentEngine
from .audit_log import AuditLog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLAWDBOT - %(levelname)s - %(message)s')
logger = logging.getLogger("ClawdBot")

class ClawdBot:
    """
    ClawdBot: The Multi-Dimensional AI Trading Agent.
    Integrates Technicals, Patterns, and Sentiment.
    """
    def __init__(self, symbol="ETH-USDT-SWAP"):
        self.symbol = symbol
        self.pattern_engine = PatternEngine()
        self.sentiment_engine = SentimentEngine()
        self.audit_log = AuditLog()
        
        self.active_trades = {} # trace_id -> trade_details
        
    async def analyze_market(self, candles: pd.DataFrame):
        """
        The Main Brain Loop.
        1. Technical Analysis (Standard)
        2. Pattern Recognition (History)
        3. Sentiment Analysis (Social)
        4. Synthesis & Decision
        """
        trace_id = str(uuid.uuid4())[:8]
        logger.info(f"🧠 [Trace: {trace_id}] Starting Analysis for {self.symbol}...")
        
        # 1. Technical Analysis
        ta_signal = self._analyze_technical(candles)
        
        # 2. Pattern Recognition
        pattern_res = self.pattern_engine.find_similar_patterns(candles)
        pattern_score = pattern_res.get('score', 0.5)
        
        # 3. Sentiment Analysis
        sentiment_res = self.sentiment_engine.analyze(self.symbol)
        sentiment_score = sentiment_res.get('score', 0.0)
        
        # 4. Aggregation (The "Vote")
        # Weights: TA (40%), Pattern (30%), Sentiment (30%)
        # Normalizing all to -1.0 (Sell) to 1.0 (Buy)
        
        # TA Signal is usually boolean/string in old bots, let's make it scored.
        ta_score = ta_signal.get('score', 0.0)
        
        weighted_score = (ta_score * 0.4) + ((pattern_score - 0.5) * 2 * 0.3) + (sentiment_score * 0.3)
        # Note: pattern_score is 0.0-1.0 (0.5 neutral). (X - 0.5)*2 maps it to -1 to 1.
        
        decision = "HOLD"
        if weighted_score > 0.6: decision = "BUY"
        elif weighted_score < -0.6: decision = "SELL"
        
        # Chain of Thought Generation
        reasoning = (
            f"Technical Score: {ta_score:.2f} ({ta_signal.get('reason', 'N/A')})\n"
            f"Pattern Score: {pattern_score:.2f} (Matches: {len(pattern_res['matches'])})\n"
            f"Sentiment Score: {sentiment_score:.2f} (Top Keywords: {sentiment_res['top_keywords']})\n"
            f"--> Weighted Aggregate: {weighted_score:.2f}\n"
            f"--> FINAL VERDICT: {decision}"
        )
        
        logger.info(f"🤖 DECISION: {decision} (Score: {weighted_score:.2f})")
        
        # Log to Diary
        self.audit_log.log_decision(
            trace_id=trace_id,
            symbol=self.symbol,
            dimensions={
                "technical": ta_score,
                "pattern": pattern_score,
                "sentiment": sentiment_score
            },
            final_decision=decision,
            reasoning=reasoning
        )
        
        return {
            "decision": decision,
            "trace_id": trace_id,
            "score": weighted_score
        }
        
    def _analyze_technical(self, df: pd.DataFrame):
        """
        Uses pandas_ta to generate a technical score.
        """
        # Simple Logic: EMA Cross + RSI
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.rsi(length=14, append=True)
        
        last = df.iloc[-1]
        
        ema_bullish = last['EMA_9'] > last['EMA_21']
        rsi_bullish = last['RSI_14'] < 70 and last['RSI_14'] > 40
        rsi_bearish = last['RSI_14'] > 30 and last['RSI_14'] < 60 # Weak logic just for demo
        
        score = 0.0
        reason = []
        
        if ema_bullish: 
            score += 0.5
            reason.append("EMA Golden Cross")
        else: 
            score -= 0.5
            reason.append("EMA Death Cross")
            
        if last['RSI_14'] < 30: 
            score += 0.5
            reason.append("RSI Oversold")
        elif last['RSI_14'] > 70: 
            score -= 0.5
            reason.append("RSI Overbought")
            
        # Clamp
        score = max(-1.0, min(1.0, score))
        
        return {"score": score, "reason": ", ".join(reason)}
