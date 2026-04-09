import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("AuditLog")


class AuditLog:
    """
    The 'Diary' of ClawdBot.
    Records Chain-of-Thought (CoT) and post-trade reflection.
    """
    def __init__(self, log_dir: str = "logs/clawdbot") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = os.path.join(self.log_dir, f"session_{int(time.time())}.jsonl")

    def log_decision(
        self,
        trace_id: str,
        symbol: str,
        dimensions: Dict[str, Any],
        final_decision: str,
        reasoning: str
    ) -> None:
        """
        Logs the decision making process (Reasoning Trace).
        """
        entry: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "type": "DECISION",
            "symbol": symbol,
            "dimensions": dimensions,  # Dict of scores/signals
            "decision": final_decision,  # BUY/SELL/HOLD
            "reasoning": reasoning  # Textual CoT
        }
        self._write(entry)

    def log_outcome(self, trace_id: str, pnl: float, learned_lesson: str) -> None:
        """
        Logs the result of a trade (Self-Reflection).
        """
        entry: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "type": "REFLECTION",
            "pnl": pnl,
            "learned_lesson": learned_lesson
        }
        self._write(entry)

    def _write(self, entry: Dict[str, Any]) -> None:
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
