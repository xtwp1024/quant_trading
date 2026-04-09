from ...modules.five_force.cognitive_unit import CognitiveUnit
from ...modules.dynamic_risk import DynamicRiskManager
import pandas as pd

class BetaUnit(CognitiveUnit):
    """
    FORMERLY: AI2_Risk
    NEW IDENTITY: Beta Unit (β) - The Entropy Prophet.
    Core Function: Entropy Calculation, Phase Transition Prediction, VaR.
    """
    def __init__(self, kb, bus, config):
        super().__init__("BetaUnit(β)", kb, bus)
        self.risk_manager = DynamicRiskManager(config)
        self.symbol = "ETH-USDT-SWAP"

    async def process(self, perception):
        # 1. Perception Extraction
        # Expecting perception to contain 'market_snapshot' with 'close' prices list
        history = perception.get('history', [])
        
        # Default Safety
        volatility = 0.01 
        regime = "Normal"
        
        if len(history) > 20:
             # Calculate Realized Volatility (Std Dev of Returns)
             df = pd.DataFrame(history, columns=['close'])
             df['returns'] = df['close'].pct_change()
             volatility = df['returns'].std()
             
             # Detect Panic (Volatility Spike > 2x Avg OR Large Drop)
             # Fix: Smooth crash (linear drop) has low std dev. Check max single-candle drop.
             max_drop = df['returns'].min()
             
             if volatility > 0.01: 
                 regime = "High Volatility"
             
             # Trigger Panic if Volatility is Extreme OR if we see a massive candle drop (>2%)
             if volatility > 0.03 or max_drop < -0.02:
                 regime = "PANIC / CRASH"
                 
        # 2. Update Risk State in Knowledge Base
        await self.kb.update_risk_score(self.name, volatility, regime)
        
        # 3. VaR Calculation (Value at Risk)
        # Simple Parametric VaR (95%) = Portfolio * 1.65 * Vol
        portfolio_value = 10000 # Mock
        var_95 = portfolio_value * 1.65 * volatility
        
        log_msg = f"Regime: {regime} | Vol: {volatility:.4f} | VaR: ${var_95:.2f}"
        
        if regime == "PANIC / CRASH":
            self.logger.warning(f"🚨 {log_msg}")
            # Active Defense: Publish Risk Alert
            self.bus.publish("RISK_ALERT", {"level": "CRITICAL", "reason": "High Volatility"})
        else:
            self.logger.info(log_msg)
            
        return log_msg

    async def dream_simulation(self):
        from ...modules.five_force.dream_engine import DreamEngine
        self.logger.info("💤 BetaUnit entering REM Sleep (Dream Simulation)...")
        
        engine = DreamEngine(volatility=0.05) # Turbulent Dream
        dream_df = engine.weave_dream(duration_days=7)
        
        # Analyze Dream
        max_drawdown = (dream_df['close'].min() - dream_df['close'].iloc[0]) / dream_df['close'].iloc[0]
        self.logger.info(f"🛌 Dream Analysis: Encountered Synthetic Scenario with {max_drawdown:.2%} Drawdown.")
        
        if max_drawdown < -0.10:
             self.logger.warning("😱 Nightmare! Dream contained a Black Swan event.")
        else:
             self.logger.info("😌 Good Dream. Risk Contained.")
