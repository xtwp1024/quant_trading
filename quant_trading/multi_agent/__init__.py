"""
Prism Multi-Agent Module

A multi-agent coordination framework for quantitative trading,
providing parallel task execution, agent communication protocols,
resource allocation, and result aggregation.

Based on the PRISM-INSIGHT multi-agent system architecture.

Usage:
    from multi_agent import PrismCoordinator, TaskAgent, TaskContext

    # Create coordinator
    coordinator = PrismCoordinator(max_concurrent=10)

    # Register agents
    coordinator.register_agent(
        "analysis_agent",
        AgentRole.TECHNICAL,
        analysis_handler
    )

    # Submit and execute tasks
    task = Task(agent_name="analysis_agent", instruction="Analyze AAPL")
    await coordinator.submit_task(task)
    results = await coordinator.execute_parallel(tasks)
"""

from .prism_core import (
    AgentConfig,
    AgentRole,
    MessageBus,
    PrismCoordinator,
    ResourcePool,
    ResultAggregator,
    Task,
    TaskResult,
)

from .task_agent import (
    AgentRole as TaskAgentRole,
    BaseTaskAgent,
    FinancialAnalysisAgent,
    MacroIntelligenceAgent,
    MarketIndexAgent,
    NewsAnalysisAgent,
    PriceVolumeAnalysisAgent,
    StrategyAgent,
    SummaryAgent,
    TaskContext,
    TradingDecisionAgent,
    TranslationAgent,
    create_agent,
    run_parallel_analysis,
)

__all__ = [
    # Core coordination
    "PrismCoordinator",
    "Task",
    "TaskResult",
    "TaskContext",
    "AgentConfig",
    "AgentRole",
    "MessageBus",
    "ResourcePool",
    "ResultAggregator",

    # Agent types
    "BaseTaskAgent",
    "PriceVolumeAnalysisAgent",
    "FinancialAnalysisAgent",
    "NewsAnalysisAgent",
    "MarketIndexAgent",
    "TradingDecisionAgent",
    "MacroIntelligenceAgent",
    "StrategyAgent",
    "TranslationAgent",
    "SummaryAgent",
    "create_agent",
    "run_parallel_analysis",

    # Enums
    "AgentRole",
]
