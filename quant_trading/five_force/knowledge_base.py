import logging
import json
import asyncio
from ..core.database import DatabaseManager

logger = logging.getLogger("FiveForce.KB")

class KnowledgeBase:
    """
    The Central Nervous System for the Five Force Agents.
    Facilitates data sharing, signal broadcasting, and state management.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        # self.redis will be accessed via self.db.redis at runtime
        
        # In-memory critical state (synced with Redis)
        self.state = {
            "risk_level": "NORMAL", # NORMAL, CAUTION, EMERGENCY
            "global_allocation": {}, # {AgentID: Amount}
            "active_strategies": [],
            "market_regime": "NEUTRAL"
        }

    async def initialize(self):
        """Load initial state from DB/Redis"""
        logger.info("🧠 [KB] initializing Knowledge Base...")
        # Sync initial state if needed
        pass

    async def store_signal(self, agent_id, signal):
        """
        AI1/AI5 publishes a trade signal.
        """
        key = f"signal:{agent_id}:{signal['symbol']}"
        await self.db.redis.set(key, json.dumps(signal), ex=300) # 5 min expiry
        
        # Store History for Correlation (List)
        hist_key = f"history_signals:{agent_id}"
        # Store as simple integer: 1 (Buy), -1 (Sell), 0 (Neutral/Exit)
        val = 1 if signal['action'] == 'BUY' else -1
        # Push to list, trim to last 100
        await self.db.redis.rpush(hist_key, val)
        await self.db.redis.ltrim(hist_key, -100, -1)
        
        logger.info(f"🧠 [KB] Signal Stored from {agent_id}: {signal['action']} {signal['symbol']}")
        
    async def get_market_data(self, symbol, timeframe='15m', limit=100):
        """
        Standardized data access for all agents.
        """
        query = "SELECT * FROM market_candles WHERE symbol = $1 AND timeframe = $2 ORDER BY timestamp DESC LIMIT $3"
        rows = await self.db.execute_query(query, (symbol, timeframe, limit))
        if rows:
            return rows[::-1] # Chronological
        return []

    async def update_risk_score(self, agent_id, score, reason):
        """
        AI2 updates risk assessment.
        """
        key = f"risk_score:{agent_id}"
        await self.db.redis.set(key, score)
        logger.info(f"🛡️ [KB] Risk Update for {agent_id}: {score} ({reason})")

    async def log_audit(self, agent_id, action, details):
        """
        AI4 logs audit trail.
        """
        # In a real system, this goes to a dedicated audit table.
        logger.info(f"👁️ [Audit] {agent_id}: {action} | {details}")
        
    async def get_allocation(self, agent_id):
        """
        AI2 sets this, Agents read it.
        """
        val = await self.db.redis.get(f"allocation:{agent_id}")
        if val is None: return 0.0
        return float(val)

    async def set_allocation(self, agent_id, amount):
        await self.db.redis.set(f"allocation:{agent_id}", amount)
        logger.info(f"💰 [KB] Allocation Set: {agent_id} = ${amount}")
