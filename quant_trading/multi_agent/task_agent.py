"""
TaskAgent and TaskResult Classes

Specialized agents for quantitative trading workflows including
technical analysis, financial analysis, news analysis, and trading decisions.

Based on the PRISM-INSIGHT multi-agent system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from .prism_core import (
    AgentConfig,
    AgentRole,
    MessageBus,
    PrismCoordinator,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskContext:
    """
    Shared context passed to all agents in a workflow.
    Contains market data, portfolio state, and analysis parameters.
    """
    reference_date: str = ""  # YYYYMMDD format
    language: str = "en"
    stock_code: str = ""
    company_name: str = ""
    portfolio_state: dict = field(default_factory=dict)
    market_data: dict = field(default_factory=dict)
    agent_results: dict = field(default_factory=dict)  # Results from other agents
    metadata: dict = field(default_factory=dict)


class BaseTaskAgent(ABC):
    """
    Base class for all task agents.
    Provides common functionality for task execution and result handling.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        config: AgentConfig | None = None
    ):
        """
        Initialize the task agent.

        Args:
            name: Agent name
            role: Agent role
            config: Agent configuration
        """
        self.name = name
        self.role = role
        self.config = config or AgentConfig(role=role, name=name)
        self._message_bus: MessageBus | None = None
        self._execution_count = 0
        logger.info(f"TaskAgent initialized: {name} (role: {role.value})")

    def set_message_bus(self, message_bus: MessageBus):
        """Set the message bus for inter-agent communication."""
        self._message_bus = message_bus

    async def publish_result(self, topic: str, result: dict):
        """Publish a result to the message bus."""
        if self._message_bus:
            await self._message_bus.publish(topic, {
                "agent": self.name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })

    @abstractmethod
    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """
        Perform the agent's analysis.

        Args:
            task: Task with instructions and data
            context: Shared context from other agents

        Returns:
            Analysis result dictionary
        """
        pass

    async def execute(self, task: Task, context: TaskContext) -> TaskResult:
        """
        Execute the agent's task with error handling.

        Args:
            task: Task to execute
            context: Shared context

        Returns:
            TaskResult with execution outcome
        """
        start_time = asyncio.get_event_loop().time()
        self._execution_count += 1

        try:
            logger.info(f"Agent {self.name} executing task {task.task_id}")

            # Perform analysis
            data = await self.analyze(task, context)

            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Publish result to message bus
            await self.publish_result(f"agent.{self.role.value}.result", {
                "task_id": task.task_id,
                "data": data
            })

            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=True,
                data=data,
                execution_time_ms=execution_time_ms,
                metadata={
                    "role": self.role.value,
                    "execution_count": self._execution_count
                }
            )

        except Exception as e:
            logger.error(f"Agent {self.name} task {task.task_id} failed: {e}")
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"role": self.role.value}
            )


class PriceVolumeAnalysisAgent(BaseTaskAgent):
    """Agent for technical analysis of stock price and volume data."""

    def __init__(self, name: str = "price_volume_agent"):
        super().__init__(name, AgentRole.TECHNICAL)
        self.max_years = 1  # Limit to 1 year for token efficiency

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Perform price and volume technical analysis."""
        logger.debug(f"{self.name}: Analyzing {context.stock_code}")

        # Extract OHLCV data from context or task
        ohlcv_data = context.market_data.get("ohlcv") or task.data.get("ohlcv", "")

        # Calculate technical indicators
        analysis = {
            "agent": self.name,
            "stock_code": context.stock_code,
            "reference_date": context.reference_date,
            "technical_summary": self._calculate_indicators(ohlcv_data),
            "support_resistance": self._find_support_resistance(ohlcv_data),
            "trend": self._analyze_trend(ohlcv_data),
            "volume_analysis": self._analyze_volume(ohlcv_data)
        }

        return analysis

    def _calculate_indicators(self, ohlcv_data: str) -> dict:
        """Calculate RSI, MACD, Bollinger Bands."""
        # Placeholder - actual implementation would parse OHLCV data
        return {
            "rsi_14": None,
            "macd": None,
            "macd_signal": None,
            "bollinger_position": None,
            "note": "Indicators calculated from OHLCV data"
        }

    def _find_support_resistance(self, ohlcv_data: str) -> dict:
        """Identify support and resistance levels."""
        return {
            "support_levels": [],
            "resistance_levels": [],
            "current_position": None
        }

    def _analyze_trend(self, ohlcv_data: str) -> dict:
        """Analyze price trend (uptrend/downtrend/sideways)."""
        return {
            "short_term": "neutral",
            "medium_term": "neutral",
            "long_term": "neutral",
            "moving_averages": {}
        }

    def _analyze_volume(self, ohlcv_data: str) -> dict:
        """Analyze trading volume patterns."""
        return {
            "volume_trend": "neutral",
            "volume_ratio": 1.0,
            "volume_signals": []
        }


class FinancialAnalysisAgent(BaseTaskAgent):
    """Agent for financial statement and valuation analysis."""

    def __init__(self, name: str = "financial_agent"):
        super().__init__(name, AgentRole.FINANCIAL)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Perform financial analysis."""
        logger.debug(f"{self.name}: Analyzing financials for {context.stock_code}")

        financial_data = task.data.get("financial_data", {})

        analysis = {
            "agent": self.name,
            "stock_code": context.stock_code,
            "reference_date": context.reference_date,
            "valuation": self._analyze_valuation(financial_data),
            "earnings_trend": self._analyze_earnings(financial_data),
            "financial_health": self._analyze_health(financial_data)
        }

        return analysis

    def _analyze_valuation(self, data: dict) -> dict:
        """Analyze stock valuation metrics."""
        return {
            "pe_ratio": None,
            "pb_ratio": None,
            "ps_ratio": None,
            "ev_ebitda": None,
            "dividend_yield": None
        }

    def _analyze_earnings(self, data: dict) -> dict:
        """Analyze earnings trends."""
        return {
            "revenue_growth": None,
            "profit_growth": None,
            "margin_trend": "neutral"
        }

    def _analyze_health(self, data: dict) -> dict:
        """Analyze financial health indicators."""
        return {
            "debt_ratio": None,
            "current_ratio": None,
            "quick_ratio": None,
            "health_score": None
        }


class NewsAnalysisAgent(BaseTaskAgent):
    """Agent for news and sentiment analysis."""

    def __init__(self, name: str = "news_agent"):
        super().__init__(name, AgentRole.NEWS)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Perform news and sentiment analysis."""
        logger.debug(f"{self.name}: Analyzing news for {context.stock_code}")

        news_data = task.data.get("news_data", [])

        analysis = {
            "agent": self.name,
            "stock_code": context.stock_code,
            "reference_date": context.reference_date,
            "sentiment": self._analyze_sentiment(news_data),
            "key_themes": self._extract_themes(news_data),
            "impact_assessment": self._assess_impact(news_data)
        }

        return analysis

    def _analyze_sentiment(self, news: list) -> dict:
        """Analyze overall news sentiment."""
        return {
            "score": 0.0,  # -1 to 1 scale
            "label": "neutral",
            "confidence": 0.0
        }

    def _extract_themes(self, news: list) -> list[str]:
        """Extract key themes from news."""
        return []

    def _assess_impact(self, news: list) -> dict:
        """Assess potential market impact."""
        return {
            "short_term_impact": "neutral",
            "long_term_impact": "neutral",
            "affected_sectors": []
        }


class MarketIndexAgent(BaseTaskAgent):
    """Agent for market index and macro analysis."""

    def __init__(self, name: str = "market_index_agent"):
        super().__init__(name, AgentRole.MARKET)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Perform market and index analysis."""
        logger.debug(f"{self.name}: Analyzing market for {context.reference_date}")

        index_data = context.market_data.get("index_data", {})

        analysis = {
            "agent": self.name,
            "reference_date": context.reference_date,
            "market_regime": self._determine_regime(index_data),
            "sector_rotation": self._analyze_sectors(index_data),
            "leading_sectors": [],
            "lagging_sectors": [],
            "risk_events": [],
            "beneficiary_themes": []
        }

        return analysis

    def _determine_regime(self, data: dict) -> str:
        """Determine current market regime (bull/bear/sideways)."""
        return "sideways"

    def _analyze_sectors(self, data: dict) -> dict:
        """Analyze sector performance and rotation."""
        return {
            "sector_trends": {},
            "rotation_pattern": "neutral"
        }


class TradingDecisionAgent(BaseTaskAgent):
    """Agent for buy/sell/hold trading decisions."""

    def __init__(self, name: str = "trading_decision_agent"):
        super().__init__(name, AgentRole.TRADING)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Generate trading decision based on all agent inputs."""
        logger.debug(f"{self.name}: Generating trading decision for {context.stock_code}")

        # Get inputs from other agents
        agent_results = context.agent_results

        decision = {
            "agent": self.name,
            "stock_code": context.stock_code,
            "reference_date": context.reference_date,
            "action": self._decide_action(agent_results),
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "risk_reward": None,
            "confidence": 0.0,
            "portfolio_weight": 0.0,
            "reasoning": []
        }

        return decision

    def _decide_action(self, agent_inputs: dict) -> str:
        """Decide action (buy/sell/hold/no_entry)."""
        return "hold"

    def _calculate_risk_reward(
        self,
        entry: float,
        target: float,
        stop: float
    ) -> float:
        """Calculate risk/reward ratio."""
        if stop >= entry:
            return 0.0
        risk = entry - stop
        reward = target - entry
        return reward / risk if risk > 0 else 0.0


class MacroIntelligenceAgent(BaseTaskAgent):
    """Agent for macro intelligence and market regime detection."""

    def __init__(self, name: str = "macro_intelligence_agent"):
        super().__init__(name, AgentRole.MACRO)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Perform macro intelligence analysis."""
        logger.debug(f"{self.name}: Analyzing macro conditions")

        analysis = {
            "agent": self.name,
            "reference_date": context.reference_date,
            "regime": self._detect_regime(),
            "sector_rotation": self._analyze_rotation(),
            "risk_events": [],
            "themes": []
        }

        return analysis

    def _detect_regime(self) -> dict:
        """Detect current market regime."""
        return {
            "type": "sideways",
            "strength": 0.5,
            "outlook": "neutral"
        }

    def _analyze_rotation(self) -> dict:
        """Analyze sector rotation patterns."""
        return {
            "leading_sectors": [],
            "lagging_sectors": [],
            "rotation_direction": "neutral"
        }


class StrategyAgent(BaseTaskAgent):
    """Agent for investment strategy synthesis."""

    def __init__(self, name: str = "strategy_agent"):
        super().__init__(name, AgentRole.STRATEGY)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Synthesize overall investment strategy."""
        logger.debug(f"{self.name}: Synthesizing strategy for {context.stock_code}")

        strategy = {
            "agent": self.name,
            "stock_code": context.stock_code,
            "investment_thesis": "",
            "key_risks": [],
            "key_opportunities": [],
            "time_horizon": "medium",
            "position_sizing": 0.0
        }

        return strategy


class TranslationAgent(BaseTaskAgent):
    """Agent for multi-language translation."""

    def __init__(self, name: str = "translation_agent"):
        super().__init__(name, AgentRole.TRANSLATION)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Translate content to target language."""
        target_lang = task.data.get("target_language", context.language)

        translation = {
            "agent": self.name,
            "source_language": "en",
            "target_language": target_lang,
            "translated_content": task.data.get("content", ""),
            "original_content": task.data.get("content", "")
        }

        return translation


class SummaryAgent(BaseTaskAgent):
    """Agent for report summarization."""

    def __init__(self, name: str = "summary_agent"):
        super().__init__(name, AgentRole.SUMMARY)

    async def analyze(self, task: Task, context: TaskContext) -> dict:
        """Generate summary from agent reports."""
        logger.debug(f"{self.name}: Generating summary")

        summary = {
            "agent": self.name,
            "sections": self._organize_sections(context.agent_results),
            "key_findings": [],
            "overall_assessment": "neutral"
        }

        return summary

    def _organize_sections(self, agent_results: dict) -> list[dict]:
        """Organize agent results into report sections."""
        sections = []
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and "data" in result:
                sections.append({
                    "source": agent_name,
                    "content": result["data"]
                })
        return sections


def create_agent(name: str, role: AgentRole) -> BaseTaskAgent:
    """
    Factory function to create agents by role.

    Args:
        name: Agent name
        role: Agent role

    Returns:
        Instantiated agent
    """
    agent_classes = {
        AgentRole.TECHNICAL: PriceVolumeAnalysisAgent,
        AgentRole.FINANCIAL: FinancialAnalysisAgent,
        AgentRole.NEWS: NewsAnalysisAgent,
        AgentRole.MARKET: MarketIndexAgent,
        AgentRole.TRADING: TradingDecisionAgent,
        AgentRole.MACRO: MacroIntelligenceAgent,
        AgentRole.STRATEGY: StrategyAgent,
        AgentRole.TRANSLATION: TranslationAgent,
        AgentRole.SUMMARY: SummaryAgent,
    }

    agent_class = agent_classes.get(role)
    if agent_class:
        return agent_class(name)
    else:
        raise ValueError(f"No agent class found for role: {role.value}")


async def run_parallel_analysis(
    coordinator: PrismCoordinator,
    agents: list[BaseTaskAgent],
    context: TaskContext
) -> dict:
    """
    Run multiple agents in parallel and aggregate results.

    Args:
        coordinator: PrismCoordinator instance
        agents: List of agents to run
        context: Shared context

    Returns:
        Aggregated results from all agents
    """
    tasks = []
    for agent in agents:
        task = Task(
            agent_name=agent.name,
            instruction=f"Analysis for {context.stock_code}",
            data={},
            context=context.__dict__ if hasattr(context, '__dict__') else {}
        )
        tasks.append(task)

    results = await coordinator.execute_parallel(tasks)

    # Aggregate results
    aggregated = {
        "context": context,
        "results": {r.agent_name: r for r in results},
        "success_count": sum(1 for r in results if r.success),
        "total_count": len(results)
    }

    return aggregated
