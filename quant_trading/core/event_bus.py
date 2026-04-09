import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .logger import logger

@dataclass
class Event:
    type: str
    payload: Any = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.timestamp = time.time()

class EventBus:
    def __init__(self, max_queue_size: int = 1000) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        # HIGH: 追踪所有异步任务以便正确管理生命周期
        self._active_tasks: List[asyncio.Task] = []

    def subscribe(self, event_type: str, callback: Callable) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: Any = None) -> None:
        """把事件放入队列进行异步处理 (Put event into queue for async processing)"""
        event = Event(type=event_type, payload=payload)
        # HIGH: 队列满时记录溢出事件，防止事件丢失
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"事件队列已满 ({self._queue.maxsize})，事件 {event_type} 被丢弃")

    async def start(self) -> None:
        """主事件循环 (Main event loop)"""
        while True:
            event = await self._queue.get()
            if event.type in self._subscribers:
                for callback in self._subscribers[event.type]:
                    try:
                        # 支持同步和异步回调 (Support both async and sync callbacks)
                        if asyncio.iscoroutinefunction(callback):
                            # HIGH: 保存task引用，防止任务被垃圾回收
                            task = asyncio.create_task(callback(event))
                            self._active_tasks.append(task)
                            task.add_done_callback(lambda t: self._active_tasks.remove(t) if t in self._active_tasks else None)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(f"处理事件 {event.type} 时出错: {e}")
            self._queue.task_done()
