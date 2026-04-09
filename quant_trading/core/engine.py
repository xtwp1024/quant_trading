import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .brain import TitanBrain
from .cortex import TitanCortex
from .database import DatabaseManager
from .event_bus import EventBus
from .logger import logger
from .knowledge_engine import TitanKnowledgeEngine
from .sentiment_observer import MarketSentimentObserver

# Legacy module imports - commented out, functionality moved to execution/
# from ..modules.executor import Executor
# from ..modules.market import MarketDataManager
# from ..modules.risk import RiskManager
# from ..modules.strategies.validated.Strategy_117_136_181 import Strategy_117_136_181
# from ..modules.strategies.triangular_arb import TriangularArbStrategy

class TradingEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.bus = EventBus()
        self.db = DatabaseManager(config)
        # HIGH: 追踪所有异步任务以便正确管理生命周期
        self._active_tasks: List[asyncio.Task] = []

        # Modules - legacy, commented out
        # self.market = MarketDataManager(self.bus, config)
        # self.risk = RiskManager(self.bus, self.db, config)
        # self.executor = Executor(self.bus, self.db, self.market, config)

        # Strategies - legacy, commented out
        # self.strat_trend = Strategy_117_136_181(self.bus, config)
        # self.strat_arb = TriangularArbStrategy(self.bus, config)

        # 进化: Titan 组件 (Evolution: Titan Components)
        self.knowledge = TitanKnowledgeEngine()
        self.brain = TitanBrain(self.knowledge, config)
        self.cortex = TitanCortex(config)
        self.sentiment = MarketSentimentObserver(self.bus, config)

        # HIGH: 禁止在日志中暴露API密钥的任何部分
        api_key = os.getenv('API_KEY')
        if api_key:
            logger.info("TitanCore API_KEY loaded: [REDACTED]")
        else:
            logger.info("TitanCore API_KEY not found in environment")

    async def start(self) -> None:
        logger.info("Starting Core Engine...")

        # 1. 数据库 (Database)
        await self.db.connect()

        # 2. 启动核心 (Start Core - Non-blocking)
        # HIGH: 保存task引用，防止任务被垃圾回收
        bus_task = asyncio.create_task(self.bus.start())
        self._active_tasks.append(bus_task)
        bus_task.add_done_callback(lambda t: self._active_tasks.remove(t) if t in self._active_tasks else None)

        # 3. 启动认知循环 (Start Cognitive Loop)
        cognitive_task = asyncio.create_task(self.cognitive_loop())
        self._active_tasks.append(cognitive_task)
        cognitive_task.add_done_callback(lambda t: self._active_tasks.remove(t) if t in self._active_tasks else None)

        logger.info("All Systems Operational. Engine Running.")

        # 保持活跃 (Keep Alive)
        while True:
            await asyncio.sleep(10)

    async def cognitive_loop(self) -> None:
        """
        思维循环：定期咨询 Hive Mind (The Thinking Loop: Periodically consult the Hive Mind).
        """
        logger.info("🧠 [TitanBrain] 认知循环已启动 (Cognitive Loop Started).")
        vibe_checks = ["Eth Bullish Market", "Liquidity Crisis", "High Volatility Scalping"]

        while True:
            # 获取当前市场情绪
            sentiment = self.sentiment.get_sentiment()
            current_trend = sentiment.get('trend', 'NEUTRAL')

            # 根据市场情绪和趋势选择上下文
            if 'BULLISH' in current_trend:
                current_vibe = vibe_checks[0]  # Eth Bullish Market
            elif sentiment.get('overall_score', 0) < -30:
                current_vibe = vibe_checks[1]  # Liquidity Crisis
            elif sentiment.get('volatility_index', 0) > 3:
                current_vibe = vibe_checks[2]  # High Volatility
            else:
                current_vibe = vibe_checks[int(asyncio.get_running_loop().time() % len(vibe_checks))]

            signal = await self.brain.analyze_market_context(current_vibe)

            # 元数据注入 Cortex (Metadata injection for Cortex)
            # 模拟 PnL 检查 (Simulate a PnL check)
            # self.cortex.update_health("TitanCore", pnl=-0.01 if "Crisis" in current_vibe else 0.005, slippage=0.001)

            # 导出状态到仪表板 (Export Status for Dashboard)
            self.export_status(current_vibe, sentiment)

            await asyncio.sleep(30) # 每 30 秒思考一次

    def export_status(self, vibe: str, sentiment: Dict[str, Any] = None) -> None:
        """导出系统状态到 JSON 文件以供仪表板使用 (Export system status to a JSON file)."""
        status = {
            "timestamp": time.time(),
            "vibe": vibe,
            "brain": self.brain.get_latest_insight(),
            "cortex": self.cortex.strategy_health,
            "sentiment": sentiment or {},
            "system_mode": self.config.get('system', {}).get('mode', 'unknown')
        }
        try:
            with open("titan_status.json", "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"导出状态失败 (Failed to export status): {e}")
