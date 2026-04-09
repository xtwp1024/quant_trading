import logging
import asyncio
from ...modules.five_force.knowledge_base import KnowledgeBase
from ..core.event_bus import EventBus

class BaseAgent:
    """
    Abstract Base Class for all Five Force Agents.
    """
    def __init__(self, agent_id: str, kb: KnowledgeBase, bus: EventBus):
        self.agent_id = agent_id
        self.kb = kb
        self.bus = bus
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        self.active = False

    async def start(self):
        """Start agent loop"""
        self.active = True
        self.logger.info(f"🟢 {self.agent_id} Started.")
        asyncio.create_task(self.run_loop())

    async def stop(self):
        """Stop agent loop"""
        self.active = False
        self.logger.info(f"🔴 {self.agent_id} Stopped.")

    async def run_loop(self):
        """Main Life Cycle - Override this"""
        while self.active:
            try:
                await self.on_tick()
            except Exception as e:
                self.logger.error(f"Error in loop: {e}")
            await asyncio.sleep(5) # Default Tick

    async def on_tick(self):
        """Per-tick logic"""
        pass

    async def emergency_stop(self, reason):
        """Called by AI2 (Risk) or AI4 (Monitor)"""
        self.logger.critical(f"🛑 EMERGENCY STOP TRIGGERED: {reason}")
        self.active = False
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        pass

    def subscribe(self, event_type: str, callback):
        """
        Subscribe to an event type via the EventBus.
        This provides a controlled interface for agents to subscribe to events.
        """
        self.bus.subscribe(event_type, callback)
