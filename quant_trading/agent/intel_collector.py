"""Intel collector with BM25 retrieval for skill-based information gathering.

Key features:
- BM25 ranking algorithm for relevance scoring
- Skill-based information gathering workflow
- Source collectors for RSS, websites, and APIs
- Workspace-scoped storage
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")


def _tokenize(text: str) -> list[str]:
    """Tokenize plain text for BM25 scoring."""
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IntelSource:
    """An intel source (RSS feed, website, API endpoint)."""

    id: int | None = None
    name: str = ""
    source_type: str = "rss"  # rss, website, api
    url: str = ""
    config_json: str = "{}"
    scope: str = "workspace"
    scope_key: str = ""
    is_active: bool = True
    is_deleted: bool = False


@dataclass
class IntelRawItem:
    """A raw intel item collected from a source."""

    id: int | None = None
    source_id: int = 0
    title: str = ""
    url: str = ""
    author: str = ""
    published_at: str | None = None
    collected_at: str = ""
    content_text: str = ""
    summary_text: str = ""
    dedup_key: str = ""
    metadata_json: str = "{}"


@dataclass
class IntelSearchHit:
    """A search result for collected intel."""

    item_id: int
    source_id: int
    source_name: str
    title: str
    url: str
    published_at: str | None
    collected_at: str | None
    summary_text: str
    content_preview: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON payload."""
        return {
            "itemId": self.item_id,
            "sourceId": self.source_id,
            "sourceName": self.source_name,
            "title": self.title,
            "url": self.url,
            "publishedAt": self.published_at,
            "collectedAt": self.collected_at,
            "summaryText": self.summary_text,
            "contentPreview": self.content_preview,
            "score": round(self.score, 6),
        }


# ---------------------------------------------------------------------------
# BM25 Ranker
# ---------------------------------------------------------------------------


class BM25Ranker:
    """
    Lightweight BM25 ranker for intel search.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It ranks documents based on query term frequency and document frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize BM25 ranker.

        Args:
            k1: Term frequency saturation parameter. Higher = less saturation.
                Typical values: 1.2-2.0
            b: Document length normalization. 0.0 = no normalization, 1.0 = full.
                Typical values: 0.5-0.75
        """
        self._k1 = k1
        self._b = b

    def rank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_fields: list[str] = ("title", "summary_text", "content_text"),
        limit: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Rank documents using BM25.

        Args:
            query: Search query text.
            documents: List of document dicts.
            text_fields: Fields to search within each document.
            limit: Maximum number of results.

        Returns:
            List of (score, document) tuples sorted by descending score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not documents:
            return []

        # Build corpus
        corpus: list[list[str]] = []
        term_doc_freq: Counter[str] = Counter()
        documents_tokenized: list[Counter[str]] = []

        for doc in documents:
            text_parts = [str(doc.get(field, "") or "") for field in text_fields]
            tokens = _tokenize(" ".join(text_parts))
            corpus.append(tokens)
            counts = Counter(tokens)
            documents_tokenized.append(counts)
            for term in counts:
                term_doc_freq[term] += 1

        total_docs = len(corpus)
        avg_doc_len = (sum(len(tokens) for tokens in corpus) / total_docs) if total_docs else 1.0

        # Score each document
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc, tokens, counts in zip(documents, corpus, documents_tokenized, strict=False):
            doc_len = max(1, len(tokens))
            score = 0.0

            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue

                df = term_doc_freq.get(term, 0)
                # IDF with Robertson-Sparck Jones formula
                idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
                # BM25 term frequency component
                denom = tf + self._k1 * (1.0 - self._b + self._b * (doc_len / avg_doc_len))
                score += idf * ((tf * (self._k1 + 1.0)) / denom)

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]


# ---------------------------------------------------------------------------
# Intel collector service
# ---------------------------------------------------------------------------


class IntelCollectorService:
    """
    Collect intel from various sources.

    Supports:
    - RSS feeds: Parse RSS/Atom feeds and extract items.
    - Websites: Fetch and extract content from web pages.
    - APIs: Custom API integrations.
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = Path(workspace) if workspace else Path(".")
        self._db_path = self._workspace / "data" / "intel.db"
        self._ranker = BM25Ranker()

    # -----------------------------------------------------------------------
    # Collection
    # -----------------------------------------------------------------------

    async def collect_source(self, source: IntelSource) -> list[IntelRawItem]:
        """
        Collect items from a single source.

        Args:
            source: The source to collect from.

        Returns:
            List of collected raw items.
        """
        if source.source_type == "rss":
            return await self._collect_rss(source)
        elif source.source_type == "website":
            return await self._collect_website(source)
        else:
            return []

    async def _collect_rss(self, source: IntelSource) -> list[IntelRawItem]:
        """Collect from an RSS or Atom feed."""
        import feedparser

        config = json.loads(source.config_json or "{}")
        url = str(config.get("url", "")).strip()
        if not url:
            raise ValueError("RSS source missing URL")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
        except Exception:
            return []

        # Parse feed (feedparser is sync, run in thread pool)
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, response.text)

        if getattr(parsed, "bozo", 0) and not getattr(parsed, "entries", []):
            return []

        now_iso = _utc_now_iso()
        items: list[IntelRawItem] = []

        for entry in parsed.entries[:50]:
            title = str(getattr(entry, "title", "") or "").strip()
            link = str(getattr(entry, "link", "") or "").strip()
            author = str(getattr(entry, "author", "") or "").strip()
            summary = str(getattr(entry, "summary", "") or "").strip()
            published_at = _coerce_published(
                str(getattr(entry, "published", "") or getattr(entry, "updated", "") or "")
            )

            items.append(
                IntelRawItem(
                    source_id=int(source.id or 0),
                    title=title,
                    url=link,
                    author=author,
                    published_at=published_at,
                    collected_at=now_iso,
                    content_text=summary,
                    summary_text=summary[:500],
                    dedup_key=_make_dedup_key(link, title, published_at),
                    metadata_json=json.dumps({"sourceType": source.source_type}, ensure_ascii=False),
                )
            )

        return items

    async def _collect_website(self, source: IntelSource) -> list[IntelRawItem]:
        """Collect content from a single webpage."""
        import httpx

        config = json.loads(source.config_json or "{}")
        url = str(config.get("url", "")).strip()
        if not url:
            raise ValueError("Website source missing URL")

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "QuantAgent/1.0"})
            response.raise_for_status()

        body = response.text[:20000]
        now_iso = _utc_now_iso()

        return [
            IntelRawItem(
                source_id=int(source.id or 0),
                title=config.get("title") or url,
                url=str(response.url),
                collected_at=now_iso,
                content_text=body,
                summary_text=body[:500],
                dedup_key=_make_dedup_key(str(response.url), url, None),
                metadata_json=json.dumps(
                    {"sourceType": source.source_type, "statusCode": response.status_code},
                    ensure_ascii=False,
                ),
            )
        ]

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 5,
        days: int = 30,
        scope: str = "workspace",
        scope_key: str = "",
    ) -> list[IntelSearchHit]:
        """
        Search collected intel with BM25 ranking.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            days: Only search items from the last N days.
            scope: Scope filter (workspace, user, etc.).
            scope_key: Scope key for filtering.

        Returns:
            Ranked search hits.
        """
        clean_query = str(query or "").strip()
        if not clean_query:
            return []

        # Fetch rows from storage
        rows = await self._fetch_rows(days=days, scope=scope, scope_key=scope_key)
        if not rows:
            return []

        # Rank with BM25
        scored = self._ranker.rank(clean_query, rows, limit=limit)

        hits: list[IntelSearchHit] = []
        for score, row in scored:
            content_text = str(row.get("content_text") or "")
            hits.append(
                IntelSearchHit(
                    item_id=int(row["id"]),
                    source_id=int(row["source_id"]),
                    source_name=str(row.get("source_name") or ""),
                    title=str(row.get("title") or ""),
                    url=str(row.get("url") or ""),
                    published_at=row.get("published_at"),
                    collected_at=row.get("collected_at"),
                    summary_text=str(row.get("summary_text") or ""),
                    content_preview=content_text[:240],
                    score=score,
                )
            )
        return hits

    async def _fetch_rows(
        self,
        days: int,
        scope: str,
        scope_key: str,
    ) -> list[dict[str, Any]]:
        """Fetch rows from the intel database."""
        import sqlite3

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            _init_intel_schema(conn)
            since_iso = (datetime.now(UTC) - timedelta(days=max(1, days))).isoformat().replace("+00:00", "Z")
            rows = conn.execute(
                """
                SELECT
                    ri.id,
                    ri.source_id,
                    ri.title,
                    ri.url,
                    ri.published_at,
                    ri.collected_at,
                    ri.summary_text,
                    ri.content_text,
                    s.name AS source_name
                FROM intel_raw_items ri
                JOIN intel_sources s ON s.id = ri.source_id
                WHERE s.scope = ?
                    AND s.scope_key = ?
                    AND s.is_deleted = 0
                    AND s.is_active = 1
                    AND COALESCE(ri.published_at, ri.collected_at) >= ?
                ORDER BY COALESCE(ri.published_at, ri.collected_at) DESC
                LIMIT 2000
                """,
                (scope, scope_key, since_iso),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------


def _init_intel_schema(conn: Any) -> None:
    """Initialize intel database schema."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS intel_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'rss',
            url TEXT NOT NULL DEFAULT '',
            config_json TEXT NOT NULL DEFAULT '{}',
            scope TEXT NOT NULL DEFAULT 'workspace',
            scope_key TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            is_deleted INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS intel_raw_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            collected_at TEXT NOT NULL,
            content_text TEXT NOT NULL DEFAULT '',
            summary_text TEXT NOT NULL DEFAULT '',
            dedup_key TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_id) REFERENCES intel_sources(id)
        );

        CREATE INDEX IF NOT EXISTS idx_items_source ON intel_raw_items(source_id);
        CREATE INDEX IF NOT EXISTS idx_items_published ON intel_raw_items(published_at);
        CREATE INDEX IF NOT EXISTS idx_items_dedup ON intel_raw_items(dedup_key);
        """
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _make_dedup_key(url: str, title: str, published_at: str | None) -> str:
    """Build a stable dedup key from item identity fields."""
    base = url.strip() or f"{title.strip()}|{published_at or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _coerce_published(value: str) -> str | None:
    """Convert RFC-style published timestamps to ISO when possible."""
    from email.utils import parsedate_to_datetime

    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except Exception:
        return text
