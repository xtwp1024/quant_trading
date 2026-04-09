"""
Prism Multi-Agent Coordination Framework

Core framework for parallel task execution, agent communication,
and result aggregation across multiple specialized agents.

Based on the PRISM-INSIGHT multi-agent system architecture.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    MACRO = "macro"                    # Market regime, sector rotation, risk events
    TECHNICAL = "technical"           # Technical analysis (price/volume)
    FINANCIAL = "financial"            # Financial statement analysis
    INDUSTRY = "industry"              # Industry/sector analysis
    NEWS = "news"                      # News and sentiment analysis
    MARKET = "market"                   # Market index analysis
    STRATEGY = "strategy"              # Investment strategy synthesis
    SUMMARY = "summary"                # Report summarization
    TRANSLATION = "translation"        # Multi-language translation
    TRADING = "trading"                # Buy/sell decision agents
    JOURNAL = "journal"                # Trading journal feedback
    CONSULTATION = "consultation"      # User interaction


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    role: AgentRole
    name: str
    description: str = ""
    max_retries: int = 3
    timeout_seconds: int = 300
    priority: int = 1  # Higher priority agents execute first


@dataclass
class TaskResult(Generic[T]):
    """Result from a task execution."""
    task_id: str
    agent_name: str
    success: bool
    data: T | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """A task to be executed by an agent."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: AgentRole | None = None
    agent_name: str = ""
    instruction: str = ""
    data: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)  # Shared context across agents
    priority: int = 1
    dependencies: list[str] = field(default_factory=list)  # Task IDs this depends on
    created_at: datetime = field(default_factory=datetime.now)


class MessageBus:
    """
    Agent communication protocol - message bus for inter-agent messaging.
    Supports publish/subscribe pattern for decoupled agent communication.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[dict], Coroutine[Any, Any, None]]]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def subscribe(self, topic: str, handler: Callable[[dict], Coroutine[Any, Any, None]]):
        """Subscribe to a topic with an async handler."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)
        logger.debug(f"Agent subscribed to topic: {topic}")

    async def unsubscribe(self, topic: str, handler: Callable[[dict], Coroutine[Any, Any, None]]):
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic] = [h for h in self._subscribers[topic] if h != handler]

    async def publish(self, topic: str, message: dict):
        """Publish a message to all subscribers of a topic."""
        if topic in self._subscribers:
            for handler in self._subscribers[topic]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for topic {topic}: {e}")

    async def start(self):
        """Start the message bus."""
        self._running = True
        asyncio.create_task(self._process_messages())

    async def stop(self):
        """Stop the message bus."""
        self._running = False

    async def _process_messages(self):
        """Process queued messages."""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.publish(message.get("topic", ""), message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")


class ResourcePool:
    """
    Resource allocation among agents.
    Manages shared resources like API quotas, rate limits, and compute budget.
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._resource_usage: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, task_id: str, agent_name: str) -> bool:
        """Acquire resources for a task. Returns True if successful."""
        await self._semaphore.acquire()
        async with self._lock:
            self._active_tasks[task_id] = asyncio.current_task()
            self._resource_usage[agent_name] = self._resource_usage.get(agent_name, 0) + 1
        logger.debug(f"Resources acquired for task {task_id} by {agent_name}")
        return True

    async def release(self, task_id: str, agent_name: str):
        """Release resources after task completion."""
        async with self._lock:
            self._active_tasks.pop(task_id, None)
            if agent_name in self._resource_usage:
                self._resource_usage[agent_name] = max(0, self._resource_usage[agent_name] - 1)
        self._semaphore.release()
        logger.debug(f"Resources released for task {task_id} by {agent_name}")

    def get_active_count(self) -> int:
        """Get number of currently active tasks."""
        return len(self._active_tasks)

    def get_usage_by_agent(self, agent_name: str) -> int:
        """Get resource usage by agent."""
        return self._resource_usage.get(agent_name, 0)


class PrismCoordinator:
    """
    Core PrismCoordinator class for multi-agent coordination.

    Coordinates parallel task execution, manages agent lifecycle,
    handles resource allocation, and aggregates results.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: int = 300,
        enable_message_bus: bool = True
    ):
        """
        Initialize the PrismCoordinator.

        Args:
            max_concurrent: Maximum concurrent agent tasks
            default_timeout: Default timeout for agent tasks in seconds
            enable_message_bus: Enable inter-agent messaging
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._agents: dict[str, dict] = {}
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._resource_pool = ResourcePool(max_concurrent)
        self._message_bus = MessageBus() if enable_message_bus else None
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

        logger.info(f"PrismCoordinator initialized (max_concurrent={max_concurrent})")

    def register_agent(
        self,
        name: str,
        role: AgentRole,
        handler: Callable[[Task], Coroutine[Any, Any, TaskResult]],
        config: AgentConfig | None = None
    ):
        """
        Register an agent with the coordinator.

        Args:
            name: Unique agent name
            role: Agent role
            handler: Async function to handle tasks
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(role=role, name=name)

        self._agents[name] = {
            "name": name,
            "role": role,
            "handler": handler,
            "config": config
        }
        logger.info(f"Registered agent: {name} (role: {role.value})")

    def get_agent(self, name: str) -> dict | None:
        """Get agent by name."""
        return self._agents.get(name)

    def get_agents_by_role(self, role: AgentRole) -> list[dict]:
        """Get all agents with a specific role."""
        return [a for a in self._agents.values() if a["role"] == role]

    async def submit_task(self, task: Task) -> str:
        """
        Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        self._tasks[task.task_id] = task
        logger.debug(f"Task submitted: {task.task_id} for agent {task.agent_name}")
        return task.task_id

    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a single task with resource management.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution outcome
        """
        start_time = asyncio.get_event_loop().time()

        # Find appropriate agent
        agent_name = task.agent_name
        if not agent_name and task.agent_role:
            agents = self.get_agents_by_role(task.agent_role)
            if agents:
                agent_name = agents[0]["name"]

        agent = self._agents.get(agent_name)
        if not agent:
            return TaskResult(
                task_id=task.task_id,
                agent_name=agent_name or "unknown",
                success=False,
                error=f"No agent found: {agent_name or task.agent_role}"
            )

        # Acquire resources
        await self._resource_pool.acquire(task.task_id, agent["name"])

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent["handler"](task),
                timeout=agent["config"].timeout_seconds
            )
            result.execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._results[task.task_id] = result
            return result

        except asyncio.TimeoutError:
            result = TaskResult(
                task_id=task.task_id,
                agent_name=agent["name"],
                success=False,
                error=f"Task timed out after {agent['config'].timeout_seconds}s"
            )
            result.execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._results[task.task_id] = result
            return result

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            result = TaskResult(
                task_id=task.task_id,
                agent_name=agent["name"],
                success=False,
                error=str(e)
            )
            result.execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._results[task.task_id] = result
            return result

        finally:
            await self._resource_pool.release(task.task_id, agent["name"])

    async def execute_parallel(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of TaskResults in the same order as input tasks
        """
        logger.info(f"Executing {len(tasks)} tasks in parallel")

        # Create coroutines for all tasks
        coros = [self.execute_task(task) for task in tasks]

        # Execute all in parallel
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Convert exceptions to TaskResult
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=tasks[i].task_id,
                    agent_name=tasks[i].agent_name,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def execute_sequential(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute tasks sequentially with dependency ordering.

        Args:
            tasks: List of tasks to execute in order

        Returns:
            List of TaskResults
        """
        results = []
        for task in tasks:
            result = await self.execute_task(task)
            results.append(result)
            # Check if task failed and has dependencies
            if not result.success and task.dependencies:
                logger.warning(f"Task {task.task_id} failed, skipping dependent tasks")
                break
        return results

    def aggregate_results(self, results: list[TaskResult]) -> dict:
        """
        Aggregate results from multiple agents.

        Args:
            results: List of TaskResults

        Returns:
            Aggregated result dictionary
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_execution_time_ms": sum(r.execution_time_ms for r in results),
            "data": {r.task_id: r.data for r in successful if r.data is not None},
            "errors": {r.task_id: r.error for r in failed if r.error}
        }

    async def start(self):
        """Start the coordinator."""
        self._running = True
        if self._message_bus:
            await self._message_bus.start()
        logger.info("PrismCoordinator started")

    async def stop(self):
        """Stop the coordinator."""
        self._running = False
        if self._message_bus:
            await self._message_bus.stop()
        self._executor.shutdown(wait=False)
        logger.info("PrismCoordinator stopped")

    def get_status(self) -> dict:
        """Get coordinator status."""
        return {
            "running": self._running,
            "registered_agents": len(self._agents),
            "pending_tasks": len(self._tasks),
            "completed_tasks": len(self._results),
            "active_tasks": self._resource_pool.get_active_count(),
            "max_concurrent": self.max_concurrent
        }


class ResultAggregator:
    """
    Aggregates and synthesizes results from multiple agents.
    Provides methods for result fusion and consensus building.
    """

    @staticmethod
    def weighted_vote(results: list[dict], weight_key: str = "weight") -> dict:
        """
        Perform weighted voting on agent results.

        Args:
            results: List of result dictionaries with weights
            weight_key: Key containing the weight value

        Returns:
            Aggregated result with scores
        """
        if not results:
            return {"decision": None, "confidence": 0, "scores": {}}

        scores: dict[str, float] = {}
        total_weight = 0

        for result in results:
            decision = result.get("decision")
            weight = result.get(weight_key, 1.0)

            if decision:
                scores[decision] = scores.get(decision, 0) + weight
                total_weight += weight

        if not scores:
            return {"decision": None, "confidence": 0, "scores": {}}

        # Normalize scores
        normalized = {k: v / total_weight for k, v in scores.items()}
        best_decision = max(normalized, key=normalized.get)

        return {
            "decision": best_decision,
            "confidence": normalized[best_decision],
            "scores": normalized
        }

    @staticmethod
    def merge_reports(reports: list[str], format: str = "markdown") -> str:
        """
        Merge multiple agent reports into a single document.

        Args:
            reports: List of report strings
            format: Report format (markdown, html, text)

        Returns:
            Merged report string
        """
        if not reports:
            return ""

        if format == "markdown":
            merged = "# Merged Analysis Report\n\n"
            for i, report in enumerate(reports, 1):
                # Extract title if present
                lines = report.strip().split("\n")
                title = lines[0] if lines else f"Section {i}"
                merged += f"## Section {i}: {title}\n\n{report}\n\n---\n\n"
            return merged

        return "\n\n".join(reports)

    @staticmethod
    def consensus_score(results: list[dict], score_key: str = "score") -> float:
        """
        Calculate consensus score from multiple agent evaluations.

        Args:
            results: List of result dictionaries with scores
            score_key: Key containing the score value

        Returns:
            Consensus score (0-1)
        """
        if not results:
            return 0.0

        scores = [r.get(score_key, 0) for r in results if score_key in r]
        if not scores:
            return 0.0

        # Simple average consensus
        avg = sum(scores) / len(scores)

        # Calculate variance-based confidence
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Higher consensus = lower variance
        # Normalize to 0-1 range (assuming max std_dev of 50)
        confidence = max(0, 1 - (std_dev / 50))

        return avg * confidence
