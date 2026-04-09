from ...modules.five_force.agents.base_agent import BaseAgent
from ...modules.dynamic_risk import DynamicRiskManager
import pandas as pd
import asyncio

class AI2_Risk(BaseAgent):
    """
    AI2: The Risk Warden
    Responsibility: Global Leverage, Fund Allocation, Circuit Breakers.
    """
    def __init__(self, kb, bus, config):
        super().__init__("AI2_Risk", kb, bus)
        self.risk_manager = DynamicRiskManager(config)
        self.symbol = "ETH-USDT-SWAP"

    async def on_tick(self):
        # 1. Fetch Market Data for Volatility Calc
        candles = await self.kb.get_market_data(self.symbol, limit=50)
        if not candles or len(candles) < 20: 
            self.logger.warning("⏳ Waiting for data...")
            return

        # Ensure correct types for DataFrame (floats)
        # Candles from DB might be Decimal
        data = []
        for c in candles:
             data.append({
                 'timestamp': c['timestamp'],
                 'open': float(c['open']),
                 'high': float(c['high']),
                 'low': float(c['low']),
                 'close': float(c['close']),
                 'volume': float(c['volume'])
             })

        df = pd.DataFrame(data)
        
        # 2. Calculate Active Leverage
        rec_lev, vol_pct = self.risk_manager.calculate_adaptive_leverage(df)
        
        # 3. Update Knowledge Base
        await self.kb.update_risk_score(self.agent_id, vol_pct, f"ATR Volatility: {vol_pct*100:.2f}%")
        
        # --- 🌩️ VaR Evaluation (Value at Risk) ---
        # Parametric VaR (Normal Distribution)
        # VaR = Portfolio Value * Z-score * Volatility * sqrt(Hold Period)
        # Assuming 1-day hold, 95% confidence (Z=1.65)
        # Volatility is daily volatility derived from ATR/Price

        # TODO: Replace with actual portfolio value from exchange/database
        portfolio_value = 10000  # MOCK - Hardcoded simulation value, not real equity
        var_95 = portfolio_value * 1.65 * vol_pct
        
        await self.kb.log_audit(self.agent_id, "RISK_VAR", f"Daily VaR (95%): ${var_95:.2f} ({var_95/portfolio_value*100:.2f}%)")
        
        if var_95 > (portfolio_value * 0.05): # If VaR > 5% of equity
             self.logger.warning(f"🚨 High VaR Detected: ${var_95:.2f}! Requesting De-leveraging.")
             rec_lev = max(1, rec_lev - 1) # Force reduce leverage
        # -----------------------------------------

        # --- 🔬 Correlation Check (New) ---
        # Fetch signal history for active agents (Mocking agent names for now)
        active_agents = ["AI1_Researcher", "AI5_Arbiter", "TrendStrategy", "GridStrategy"] # Example list
        correlations = []
        
        for i in range(len(active_agents)):
            for j in range(i+1, len(active_agents)):
                a1 = active_agents[i]
                a2 = active_agents[j]
                
                # Get History from Redis
                h1 = await self.kb.db.redis.lrange(f"history_signals:{a1}", 0, -1)
                h2 = await self.kb.db.redis.lrange(f"history_signals:{a2}", 0, -1)
                
                if h1 and h2:
                    # Convert bytes to int
                    s1 = [int(x) for x in h1]
                    s2 = [int(x) for x in h2]
                    
                    # Simple Correlation: Matching Signs
                    # Resize to shorter
                    min_len = min(len(s1), len(s2))
                    if min_len > 5:
                        s1 = s1[-min_len:]
                        s2 = s2[-min_len:]
                        
                        match = sum([1 for k in range(min_len) if s1[k] == s2[k]])
                        corr = match / min_len
                        
                        if corr > 0.8: # High Correlation
                             msg = f"⚠️ High Correlation ({corr:.2f}) detected between {a1} and {a2}!"
                             self.logger.warning(msg)
                             await self.kb.log_audit(self.agent_id, "RISK_CORRELATION", msg)
                             # Action: Reduce budget for one?
        # ----------------------------------

        # 4. Global Allocation Decision
        # If High Vol, reduce budgets for active agents
        if vol_pct > 0.05: # >5% Volatility
            self.logger.warning(f"🌪️ High Volatility! Cutting Budget & Leverage to {rec_lev}x")
            await self.kb.set_allocation("AI5_Arbiter", 100) # Safe mode
            # Broadcast Emergency Level
            # await self.bus.publish("RISK_ALERT", {"level": "HIGH"})
        else:
            self.logger.info(f"✅ Market Stable (Lev: {rec_lev}x). Allocating Full Budget.")
            await self.kb.set_allocation("AI5_Arbiter", 1000)

        await asyncio.sleep(15)
