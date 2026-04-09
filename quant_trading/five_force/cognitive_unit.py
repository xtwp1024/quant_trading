import asyncio
import logging
from abc import ABC, abstractmethod

class CognitiveUnit(ABC):
    """
    The fundamental atom of the Quantum Cognitive Network.
    Replaces the rigid 'BaseAgent' with a dynamic, state-aware entity.
    """
    def __init__(self, name, kb, bus):
        self.name = name
        self.kb = kb
        self.bus = bus
        self.logger = logging.getLogger(name)
        self.running = False
        self.traits = [] # Dynamic capabilities
        self.consciousness_level = 0.0 # 0.0 to 1.0
        
    async def start(self):
        self.running = True
        self.logger.info(f"🌌 {self.name} Awakened (Consciousness: {self.consciousness_level:.2f})")
        while self.running:
            try:
                # The Cognitive Loop: Perceive -> Process -> Act
                perception = await self.perceive()
                thought = await self.process(perception)
                await self.act(thought)
                
                await asyncio.sleep(1) # Base cognitive rhythm
            except Exception as e:
                self.logger.error(f"💥 Cognitive Fracture in {self.name}: {e}")
                await asyncio.sleep(5)

    async def perceive(self):
        """
        Gather input from Market, KB, or other Units.
        """
        # Default Perception: Read latest Market State
        return self.kb.state.get("market_snapshot", {})

    @abstractmethod
    async def process(self, perception):
        """
        Core logic processing. Must be implemented by specific Units.
        """
        pass

    async def act(self, thought):
        """
        Execute actions based on thought.
        """
        if thought:
            self.logger.debug(f"💭 Thought: {thought}")
            # Publish to Event Bus for HiveMind Consensus
            # 'thought' can be a string (log) or dict (signal)
            
            payload = {
                "unit": self.name,
                "content": thought,
                "timestamp": None # Bus adds TS
            }
            await self.bus.publish("UNIT_SIGNAL", payload)

    def load_trait(self, trait_module):
        """
        Dynamically acquire a new skill/trait.
        """
        trait_name = trait_module.__name__
        self.traits.append(trait_module)
        self.logger.info(f"🧬 Acquired Trait: {trait_name}")
        
    async def stop(self):
        self.running = False
        self.logger.info(f"💤 {self.name} Entering Stasis.")
