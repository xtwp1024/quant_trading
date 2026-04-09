#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Bank - SQLite + BM25 Persistence for ETH Long Runner.
记忆银行 - 存储历史决策、检索相似情境
"""

import sqlite3
import logging
import json
import math
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("MemoryBank")


@dataclass
class DecisionRecord:
    """决策记录"""
    id: Optional[int]
    timestamp: str
    price: float
    signal: str  # BUY/SELL/HOLD
    strength: float
    score: float  # consensus score
    indicators: dict  # technical indicators snapshot
    research_summary: str
    debate_summary: str
    pnl: Optional[float] = None  # 实际盈亏 (复盘后填充)
    correct: Optional[bool] = None  # 决策是否正确


class MemoryBank:
    """
    记忆银行 - SQLite持久化 + BM25检索

    功能:
    1. 存储所有决策记录
    2. BM25相似度检索历史情境
    3. 观点演化追踪
    4. 复盘评分更新
    """

    def __init__(self, db_path: str = None):
        import os
        if db_path is None:
            _base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(_base, "data", "memory_bank.db")
        self.db_path = str(db_path)
        self._ensure_dir()
        self._init_db()
        self._bm25_index: List[Tuple[str, int]] = []  # (text, decision_id)
        self._load_bm25_index()

    def _ensure_dir(self):
        """确保目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 决策记录表
        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                signal TEXT NOT NULL,
                strength REAL NOT NULL,
                score REAL NOT NULL,
                indicators TEXT,  -- JSON
                research_summary TEXT,
                debate_summary TEXT,
                pnl REAL,
                correct BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 观点记录表 (用于追踪观点演化)
        c.execute("""
            CREATE TABLE IF NOT EXISTS view_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                view_type TEXT NOT NULL,  -- bull/bear/neutral
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                indicators TEXT,  -- JSON snapshot
                notes TEXT
            )
        """)

        # 信号统计表
        c.execute("""
            CREATE TABLE IF NOT EXISTS signal_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                buy_count INTEGER DEFAULT 0,
                sell_count INTEGER DEFAULT 0,
                hold_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                correct_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0
            )
        """)

        # 创建索引
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_signal ON decisions(signal)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_view_timestamp ON view_evolution(timestamp)")

        conn.commit()
        conn.close()
        logger.info(f"[DB] MemoryBank initialized: {self.db_path}")

    # ===================== BM25 检索 =====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.findall(r'\b\w+\b', text)

    def _load_bm25_index(self):
        """加载BM25索引"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, indicators, research_summary, debate_summary FROM decisions")
        rows = c.fetchall()
        conn.close()

        self._bm25_index = []
        for row in rows:
            decision_id, indicators_json, research, debate = row
            # 合并文本
            combined = ""
            if indicators_json:
                combined += indicators_json + " "
            if research:
                combined += research + " "
            if debate:
                combined += debate + " "
            self._bm25_index.append((combined, decision_id))

    def _bm25_score(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
        """计算BM25得分"""
        if not self._bm25_index:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 计算平均文档长度
        avg_doc_len = sum(len(self._tokenize(doc)) for doc, _ in self._bm25_index) / max(len(self._bm25_index), 1)

        doc_scores = []
        N = len(self._bm25_index)

        # 计算IDF
        doc_freq = {}
        for doc, _ in self._bm25_index:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        for doc_text, doc_id in self._bm25_index:
            doc_tokens = self._tokenize(doc_text)
            doc_len = len(doc_tokens)

            score = 0.0
            for q_token in query_tokens:
                if q_token in doc_freq:
                    idf = math.log((N - doc_freq[q_token] + 0.5) / (doc_freq[q_token] + 0.5) + 1)
                    tf = doc_tokens.count(q_token)
                    freq = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                    score += idf * freq

            doc_scores.append((doc_id, score))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def get_memories(self, situation: str, n_matches: int = 3) -> List[Dict[str, Any]]:
        """
        检索相似历史情境

        Args:
            situation: 当前情境描述
            n_matches: 返回数量

        Returns:
            相似历史决策列表
        """
        scores = self._bm25_score(situation, n_matches * 2)[:n_matches]

        if not scores:
            return []

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        results = []
        for doc_id, bm25_score in scores:
            c.execute("SELECT * FROM decisions WHERE id = ?", (doc_id,))
            row = c.fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "price": row[2],
                    "signal": row[3],
                    "strength": row[4],
                    "score": row[5],
                    "indicators": json.loads(row[6]) if row[6] else {},
                    "research_summary": row[7],
                    "debate_summary": row[8],
                    "pnl": row[9],
                    "correct": row[10],
                    "bm25_score": bm25_score
                })

        conn.close()
        return results

    # ===================== 决策存储 =====================

    def save_decision(self, decision: DecisionRecord) -> int:
        """
        保存决策记录

        Returns:
            decision id
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO decisions
            (timestamp, price, signal, strength, score, indicators, research_summary, debate_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp,
            decision.price,
            decision.signal,
            decision.strength,
            decision.score,
            json.dumps(decision.indicators),
            decision.research_summary,
            decision.debate_summary
        ))

        decision_id = c.lastrowid
        conn.commit()
        conn.close()

        # 重建BM25索引
        self._load_bm25_index()

        logger.info(f"[SAVE] Decision saved: id={decision_id}, signal={decision.signal}, price={decision.price}")
        return decision_id

    def update_decision_pnl(self, decision_id: int, pnl: float, correct: bool):
        """复盘后更新决策的盈亏"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            UPDATE decisions SET pnl = ?, correct = ? WHERE id = ?
        """, (pnl, correct, decision_id))
        conn.commit()
        conn.close()
        logger.info(f"[REVIEW] Decision {decision_id} updated: pnl={pnl:.4f}, correct={correct}")

    # ===================== 观点演化 =====================

    def save_view(self, view_type: str, confidence: float, price: float,
                  indicators: dict = None, notes: str = None):
        """保存观点记录"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO view_evolution
            (timestamp, view_type, confidence, price, indicators, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            view_type,
            confidence,
            price,
            json.dumps(indicators) if indicators else None,
            notes
        ))
        conn.commit()
        conn.close()

    def get_view_evolution(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取近期观点演化"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, view_type, confidence, price
            FROM view_evolution
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp
        """, (f"-{days} days",))

        rows = c.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "view_type": row[1],
                "confidence": row[2],
                "price": row[3]
            }
            for row in rows
        ]

    # ===================== 统计查询 =====================

    def get_signal_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取近期信号统计"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) as buy,
                SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END) as sell,
                SUM(CASE WHEN signal = 'HOLD' THEN 1 ELSE 0 END) as hold,
                AVG(score) as avg_score,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                COUNT(CASE WHEN correct IS NOT NULL THEN 1 END) as evaluated
            FROM decisions
            WHERE timestamp >= datetime('now', ?)
        """, (f"-{days} days",))

        row = c.fetchone()
        conn.close()

        if row and row[0] > 0:
            return {
                "total": row[0],
                "buy": row[1] or 0,
                "sell": row[2] or 0,
                "hold": row[3] or 0,
                "avg_score": row[4] or 0,
                "correct": row[5] or 0,
                "evaluated": row[6] or 0,
                "win_rate": row[5] / row[6] if row[6] > 0 else 0
            }
        return {
            "total": 0, "buy": 0, "sell": 0, "hold": 0,
            "avg_score": 0, "correct": 0, "evaluated": 0, "win_rate": 0
        }

    def get_recent_decisions(self, limit: int = 10) -> List[DecisionRecord]:
        """获取最近决策"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT id, timestamp, price, signal, strength, score,
                   indicators, research_summary, debate_summary, pnl, correct
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = c.fetchall()
        conn.close()

        return [
            DecisionRecord(
                id=row[0],
                timestamp=row[1],
                price=row[2],
                signal=row[3],
                strength=row[4],
                score=row[5],
                indicators=json.loads(row[6]) if row[6] else {},
                research_summary=row[7],
                debate_summary=row[8],
                pnl=row[9],
                correct=row[10]
            )
            for row in rows
        ]
