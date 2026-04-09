from ...modules.five_force.agents.base_agent import BaseAgent
import asyncio
import random

class AI4_Monitor(BaseAgent):
    """
    AI4: The Sentinel (Monitor)
    Responsibility: Audit, Attribution, Auto-Repair.
    """
    def __init__(self, kb, bus):
        super().__init__("AI4_Monitor", kb, bus)
        # Subscribe to all trade signals to archive them
        self.subscribe("TRADE_SIGNAL", self.archive_signal)

    async def archive_signal(self, event):
        """
        Capture signal from EventBus and store in KB for Risk/Audit.
        """
        signal = event.payload
        strategy_name = signal.get('strategy', 'UnknownStrategy')
        
        # Store in KB (This adds to signal history for Correlation Check)
        # We treat the strategy name as the Agent ID for history tracking
        await self.kb.store_signal(strategy_name, signal)
        
        # Log for audit
        # self.logger.info(f"💾 Archived Signal from {strategy_name}")

    async def on_tick(self):
        # 1. Health Check
        # Query active strategies state from KB (mocked for now)
        active_strats = self.kb.state.get("active_strategies", [])
        
        self.logger.info(f"👁️ Scanning System Health... (Strategies: {len(active_strats)})")
        
        # 2. Audit Log
        # In real impl, we verify if order flow matches signals
        await self.kb.log_audit(self.agent_id, "HEALTH_CHECK", "All Systems Green")
        
        # 3. Anomaly Detection (Statistical Z-Score)
        await self.check_anomalies()
        
        # 4. Attribution Report (Iteration 7)
        if len(active_strats) > 0 or True: # Force run for demo
            await self.generate_attribution()
        
        # 5. Latency Monitor (Iteration 18)
        # Mock Latency Check (Tick vs Trade Time)
        latency_ms = random.randint(10, 150)
        if latency_ms > 100:
             self.logger.warning(f"🐢 Slow Execution Detected: {latency_ms}ms")
            
        await asyncio.sleep(60) # Only check every minute

    async def generate_attribution(self):
        """
        Iteration 7: Daily PnL Attribution.
        Decompose PnL into Alpha Factors (Trend vs MeanRev).
        """
        # Mock PnL Data
        # In real system, query 'pnl_daily:strategy_name'
        report = {
            "Trend (Beta)": 1.2,
            "MeanReversion (Alpha)": -0.5,
            "Arbitrage (Risk-Free)": 0.3
        }
        
        # Log Report
        msg = f"📊 Daily Attribution: Trend={report['Trend (Beta)']}% | MeanRev={report['MeanReversion (Alpha)']}% | Arb={report['Arbitrage (Risk-Free)']}%"
        self.logger.info(msg)
        await self.kb.log_audit(self.agent_id, "ATTRIBUTION", msg)

    async def check_anomalies(self):
        """
        Detect 3-Sigma events in strategy Returns.
        """
        # Mocking returns stream for a strategy
        # In real usage, fetch 'returns_history:StrategyX' from Redis
        recent_returns = [0.01, -0.02, 0.05, 0.01, -0.01, 0.02, 0.03, -0.05, 0.01]
        
        # Inject an anomaly randomly
        import random
        if random.random() > 0.99: # 1% chance of mock anomaly
             recent_returns.append(-0.15) 
        
        # Calculate Z-Score
        import statistics
        mean = statistics.mean(recent_returns)
        stdev = statistics.stdev(recent_returns)
        
        if len(recent_returns) > 5:
            last_ret = recent_returns[-1]
            z_score = (last_ret - mean) / (stdev + 1e-6)
            
            if abs(z_score) > 3:
                msg = f"🚨 3-Sigma Anomaly Detected! Z-Score: {z_score:.2f} (Return: {last_ret*100:.2f}%)"
                self.logger.critical(msg)
                await self.kb.log_audit(self.agent_id, "ANOMALY", msg)
                # Action: Trigger Emergency Stop for this strategy?
                # await self.bus.publish("EMERGENCY_STOP", {"strategy": "Unknown"})
