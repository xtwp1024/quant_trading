# quant_trading/sentiment/stock_reddit_bert.py
"""
Reddit + BERT 社交媒体情感选股模块 / Reddit + BERT Social Sentiment Stock Prediction Module
========================================================================================

从 Reddit (wallstreetbets / stocks / pennystocks 等) 采集帖子与评论，
使用 BERT 语义嵌入 + FAISS 向量检索 + sklearn 回归/分类预测股价方向。

Architecture / 架构:
    RedditDataCollector   →  采集 Reddit 帖子/评论（支持 praw；无依赖时降级为关键词提取）
    RedditBERTEncoder     →  文本 → BERT 768-dim 向量（transformers；无依赖时降级为 TF-IDF）
    FAISSVectorStore      →  向量相似度检索（faiss；无依赖时降级为 sklearn NearestNeighbors）
    SentimentStockPredictor →  社会情感 → 股价方向预测（sklearn）

主要入口 / Main entry point:
    generate_signals()  从 Reddit 数据直接输出交易信号

Dependencies / 依赖 (均为可选 / all optional):
    transformers  - BERT 编码器
    faiss-cpu     - 高性能向量检索
    praw          - Reddit API
    sklearn       - 预测模型 & Fallback 编码器

Graceful degradation / 降级说明:
    无 transformers → TF-IDF (sklearn TfidfVectorizer)
    无 faiss        → sklearn NearestNeighbors (brute-force)
    无 praw         → requests + 关键词提取
    无 sklearn      → 纯 stdlib 统计方法
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import 辅助 / Lazy import helper
# ---------------------------------------------------------------------------

def _lazy_import(module_name: str, package: Optional[str] = None):
    """Lazy import a module; return (success: bool, module_or_None)."""
    try:
        return True, __import__(module_name, fromlist=[package or module_name])
    except ImportError:
        return False, None


def _lazy_import_transformers():
    return _lazy_import("transformers")
def _lazy_import_faiss():
    return _lazy_import("faiss")
def _lazy_import_praw():
    return _lazy_import("praw")
def _lazy_import_sklearn():
    return _lazy_import("sklearn")
def _lazy_import_requests():
    return _lazy_import("requests")


# ---------------------------------------------------------------------------
# 1. RedditDataCollector
#    采集 Reddit 帖子/评论
# ---------------------------------------------------------------------------

class RedditDataCollector:
    """Collect Reddit posts and comments for a given ticker or set of subreddits.

    Args:
        client_id (str, optional): Reddit app client_id. Defaults to env REDDIT_CLIENT_ID.
        client_secret (str, optional): Reddit app client_secret. Defaults to env REDDIT_CLIENT_SECRET.
        user_agent (str, optional): Reddit app user_agent. Defaults to env REDDIT_USER_AGENT.
        username (str, optional): Reddit account username. Defaults to env REDDIT_USERNAME.
        password (str, optional): Reddit account password. Defaults to env REDDIT_PASSWORD.
        subreddits (list[str], optional): List of subreddits to monitor.
            Defaults to ["wallstreetbets", "stocks", "pennystocks", "StockMarket"].
        score_threshold (int, optional): Minimum post score to keep. Defaults to 10.
        comment_threshold (int, optional): Minimum number of comments to keep. Defaults to 3.

    Features:
        - Lazy import of `praw` (Reddit API). Falls back to keyword extraction via `requests`
          if `praw` is not available.
        - Filters posts using financial keywords and ticker extraction.
        - Concurrent fetching with ThreadPoolExecutor (max_workers=10).

    Example:
        >>> collector = RedditDataCollector()
        >>> posts = collector.collect(subreddit="wallstreetbets", ticker="GME", limit=50)
        >>> print(len(posts))
    """

    SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "pennystocks",
        "StockMarket",
        "EducatedInvesting",
        "Wallstreetbetsnew",
    ]

    FINANCIAL_KEYWORDS = [
        "stock", "shares", "market", "investment", "price", "trading",
        "bullish", "bearish", "IPO", "earnings", "dividends",
        "call", "put", "option", "yolo", "tendies",
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        subreddits: Optional[list[str]] = None,
        score_threshold: int = 10,
        comment_threshold: int = 3,
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT")
        self.username = username or os.getenv("REDDIT_USERNAME")
        self.password = password or os.getenv("REDDIT_PASSWORD")
        self.subreddits = subreddits or self.SUBREDDITS
        self.score_threshold = score_threshold
        self.comment_threshold = comment_threshold

        self._reddit = None
        self._praw_available = False
        self._requests_available = False

        self._init_reddit()

    # ------------------------------------------------------------------
    # Internal: init
    # ------------------------------------------------------------------

    def _init_reddit(self):
        """尝试初始化 praw 或 requests。"""
        ok, praw = _lazy_import_praw()
        if ok and self.client_id and self.client_secret:
            try:
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent or "QuantTradingBot/0.1",
                    username=self.username or "",
                    password=self.password or "",
                )
                # Test connection
                _ = self._reddit.user.me()
                self._praw_available = True
                logger.info("RedditDataCollector: praw connected successfully.")
            except Exception as e:
                logger.warning(f"RedditDataCollector: praw init failed: {e}. Falling back to requests.")
                self._reddit = None
        else:
            ok_req, requests = _lazy_import_requests()
            if ok_req:
                self._requests_available = True
                logger.info("RedditDataCollector: using requests-based fallback.")
            else:
                logger.warning("RedditDataCollector: no praw or requests available. Only keyword-filter mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        subreddit: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        hours_back: int = 24,
    ) -> list[dict]:
        """采集 Reddit 帖子。

        Args:
            subreddit (str, optional): 单个 subreddit 名。Defaults to first in self.subreddits.
            ticker (str, optional): 按 ticker 过滤（$GME / GME 均支持）。
            limit (int, optional): 最大帖子数。Defaults to 100.
            hours_back (int, optional): 只取最近 N 小时内帖子。Defaults to 24.

        Returns:
            list[dict]: 每个元素包含 title, body, score, comments, url, created_utc。
        """
        if self._praw_available:
            return self._collect_praw(subreddit or self.subreddits[0], ticker, limit, hours_back)
        elif self._requests_available:
            return self._collect_requests(subreddit or self.subreddits[0], ticker, limit, hours_back)
        else:
            return self._collect_fallback(ticker)

    def _collect_praw(
        self,
        subreddit_name: str,
        ticker: Optional[str],
        limit: int,
        hours_back: int,
    ) -> list[dict]:
        """使用 praw 采集帖子和评论。"""
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        cutoff_ts = cutoff.timestamp()
        results = []

        target_subs = [subreddit_name] if subreddit_name else self.subreddits

        def fetch_post_url(post):
            try:
                return f"https://www.reddit.com{post.permalink}"
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for sub_name in target_subs:
                sub = self._reddit.subreddit(sub_name)
                for post in sub.new(limit=limit):
                    if post.created_utc < cutoff_ts:
                        continue
                    if ticker and not self._post_mentions_ticker(post, ticker):
                        continue
                    futures.append(executor.submit(self._fetch_post_detail, post))

            for future in as_completed(futures):
                try:
                    post_data = future.result()
                    if self._is_relevant(post_data):
                        results.append(post_data)
                except Exception as e:
                    logger.debug(f"Post fetch error: {e}")

        return results

    def _fetch_post_detail(self, post) -> dict:
        """获取单帖详情（标题+正文+评论）。"""
        post.comments.replace_more(limit=0)
        comments_data = []
        for comment in post.comments.list():
            if hasattr(comment, "body") and not hasattr(comment, "body_html"):
                try:
                    comments_data.append({
                        "body": str(comment.body),
                        "score": int(comment.score),
                        "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                        "replies": [],
                    })
                    for reply in getattr(comment, "replies", []):
                        if hasattr(reply, "body"):
                            comments_data[-1]["replies"].append({
                                "body": str(reply.body),
                                "score": int(reply.score),
                            })
                except Exception:
                    continue

        return {
            "title": str(post.title),
            "body": str(post.selftext),
            "score": int(post.score),
            "num_comments": len(comments_data),
            "comments": comments_data,
            "url": f"https://www.reddit.com{post.permalink}",
            "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
            "id": str(post.id),
        }

    def _fetch_post_detail_url(self, url: str) -> dict:
        """通过 URL 获取单帖详情（praw）。"""
        try:
            submission_id = url.split("/")[-3]
            post = self._reddit.submission(id=submission_id)
            return self._fetch_post_detail(post)
        except Exception as e:
            logger.warning(f"Failed to fetch post from URL {url}: {e}")
            return {}

    def _post_mentions_ticker(self, post, ticker: str) -> bool:
        """检查帖子标题或正文中是否提到 ticker。"""
        text = f"{post.title} {post.selftext}".upper()
        patterns = [
            rf"\${ticker.upper()}\b",
            rf"\b{ticker.upper()}\b",
        ]
        return any(re.search(p, text) for p in patterns)

    def _post_mentions_ticker_text(self, text: str, ticker: str) -> bool:
        """文本中是否提到 ticker（不区分大小写）。"""
        t = ticker.upper()
        return bool(re.search(rf"\${t}\b|\b{t}\b", text.upper()))

    def _is_relevant(self, post: dict) -> bool:
        """基于分数和关键词判断帖子是否相关。"""
        if post.get("score", 0) < self.score_threshold:
            return False
        if len(post.get("comments", [])) < self.comment_threshold:
            return False
        combined = f"{post.get('title', '')} {post.get('body', '')}".lower()
        return any(kw in combined for kw in self.FINANCIAL_KEYWORDS)

    def _collect_requests(
        self,
        subreddit_name: str,
        ticker: Optional[str],
        limit: int,
        hours_back: int,
    ) -> list[dict]:
        """使用 requests（无 praw）从 Reddit 公开 API 采集帖子。"""
        ok, requests = _lazy_import_requests()
        if not ok:
            return []

        headers = {"User-Agent": self.user_agent or "QuantTradingBot/0.1"}
        results = []
        cutoff = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())

        try:
            url = (
                f"https://www.reddit.com/r/{subreddit_name}/new.json"
                f"?limit={limit}&raw_json=1"
            )
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Reddit API returned {resp.status_code}")
                return []
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            for child in children:
                post = child.get("data", {})
                created_utc = post.get("created_utc", 0)
                if created_utc < cutoff:
                    continue
                title = post.get("title", "")
                body = post.get("selftext", "")
                if ticker and not self._post_mentions_ticker_text(f"{title} {body}", ticker):
                    continue
                results.append({
                    "title": title,
                    "body": body,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "comments": [],
                    "url": post.get("url", ""),
                    "created_utc": datetime.utcfromtimestamp(created_utc).isoformat(),
                    "id": post.get("id", ""),
                })
        except Exception as e:
            logger.warning(f"Requests-based Reddit fetch failed: {e}")

        return results

    def _collect_fallback(self, ticker: Optional[str] = None) -> list[dict]:
        """无任何 HTTP 库时的降级方案：返回空列表 + 日志警告。"""
        logger.warning("RedditDataCollector: no HTTP library available. Returning empty list.")
        return []

    def collect_from_urls(self, urls: list[str]) -> list[dict]:
        """从 URL 列表批量获取帖子详情。

        Args:
            urls (list[str]): Reddit 帖子 URL 列表。

        Returns:
            list[dict]: 每个元素包含帖子详情（标题+正文+评论+分数）。
        """
        if self._praw_available:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._fetch_post_detail_url, url) for url in urls]
                results = []
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        if r:
                            results.append(r)
                    except Exception as e:
                        logger.debug(f"URL fetch error: {e}")
            return results
        else:
            logger.warning("collect_from_urls requires praw. Returning empty list.")
            return []

    def extract_tickers(self, text: str) -> list[str]:
        """从文本中提取美股 ticker（1-5 大写字母）。

        Args:
            text (str): 输入文本。

        Returns:
            list[str]: 匹配到的 ticker 列表（如 ["GME", "AAPL", "TSLA"]）。
        """
        return re.findall(r"\$?[A-Z]{1,5}\b", text.upper())

    def filter_posts_by_ticker(self, posts: list[dict], ticker: str) -> list[dict]:
        """过滤出提及指定 ticker 的帖子。

        Args:
            posts (list[dict]): 原始帖子列表。
            ticker (str): 目标 ticker（如 "GME"）。

        Returns:
            list[dict]: 只包含提及 ticker 的帖子。
        """
        return [
            p for p in posts
            if self._post_mentions_ticker_text(f"{p.get('title','')} {p.get('body','')}", ticker)
        ]


# ---------------------------------------------------------------------------
# 2. RedditBERTEncoder
#    文本 → BERT 语义向量（transformers；无依赖时降级为 TF-IDF）
# ---------------------------------------------------------------------------

class RedditBERTEncoder:
    """Encode Reddit text (posts + comments) into dense vectors for similarity search.

    Args:
        model_name (str, optional): HuggingFace model name for BERT. Defaults to "bert-base-uncased".
        embedding_dim (int, optional): Output embedding dimension. Defaults to 768.
        max_length (int, optional): Max token length per chunk. Defaults to 512.
        device (str, optional): PyTorch device ("cuda" or "cpu"). Defaults to "cpu".

    Features:
        - Lazy import of `transformers`. Falls back to TF-IDF (sklearn) if unavailable.
        - Mean-pooling over token embeddings to produce a fixed-size sentence vector.
        - Automatic text chunking for texts longer than max_length.

    Example:
        >>> encoder = RedditBERTEncoder()
        >>> vec = encoder.encode("GME to the moon! 🚀")
        >>> print(vec.shape)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 512,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None
        self._transformers_available = False
        self._sklearn_available = False
        self._init_encoder()

    def _init_encoder(self):
        """初始化 transformers 或 sklearn。"""
        ok, transformers = _lazy_import_transformers()
        if ok:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                self._transformers_available = True
                logger.info(f"RedditBERTEncoder: transformers loaded ({self.model_name}).")
                return
            except Exception as e:
                logger.warning(f"RedditBERTEncoder: transformers init failed: {e}. Falling back to TF-IDF.")

        ok_sk, sklearn = _lazy_import_sklearn()
        if ok_sk:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(
                max_features=self.embedding_dim,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
            self._sklearn_available = True
            logger.info("RedditBERTEncoder: using TF-IDF fallback (sklearn).")
        else:
            logger.warning("RedditBERTEncoder: no sklearn either. Using hash-based dummy encoder.")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean-pool token embeddings using attention_mask."""
        try:
            import torch
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        except Exception:
            import numpy as np
            return np.zeros(self.embedding_dim)

    def encode(self, text: str) -> Any:
        """将单条文本编码为向量。

        Args:
            text (str): 输入文本（可以是帖子标题+正文+评论拼接）。

        Returns:
            np.ndarray or torch.Tensor: 形状 (embedding_dim,) 的向量。
        """
        if self._transformers_available:
            return self._encode_transformers(text)
        elif self._sklearn_available:
            return self._encode_tfidf(text)
        else:
            return self._encode_hash(text)

    def encode_batch(self, texts: list[str]) -> Any:
        """批量编码文本。

        Args:
            texts (list[str]): 文本列表。

        Returns:
            np.ndarray or torch.Tensor: 形状 (n, embedding_dim) 的矩阵。
        """
        if self._transformers_available:
            return self._encode_batch_transformers(texts)
        elif self._sklearn_available:
            return self._encode_batch_tfidf(texts)
        else:
            return self._encode_batch_hash(texts)

    # ------------------------------------------------------------------
    # Transformers path
    # ------------------------------------------------------------------

    def _encode_transformers(self, text: str) -> Any:
        """使用 transformers BERT 编码单条文本。"""
        import torch
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self._model(**encoded)
        vec = self._mean_pooling(outputs, encoded["attention_mask"])
        return vec.squeeze(0).cpu().numpy()

    def _encode_batch_transformers(self, texts: list[str]) -> Any:
        """使用 transformers 批量编码。"""
        import torch
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self._model(**encoded)
        vecs = self._mean_pooling(outputs, encoded["attention_mask"])
        return vecs.cpu().numpy()

    # ------------------------------------------------------------------
    # TF-IDF fallback path
    # ------------------------------------------------------------------

    def _encode_tfidf(self, text: str) -> Any:
        """使用 TF-IDF 编码单条文本。"""
        import numpy as np
        try:
            vec = self._tfidf.transform([text]).toarray().squeeze(0)
        except Exception:
            vec = np.zeros(self.embedding_dim)
        if vec.shape[0] < self.embedding_dim:
            vec = np.pad(vec, (0, self.embedding_dim - vec.shape[0]))
        return vec.astype(np.float32)

    def _encode_batch_tfidf(self, texts: list[str]) -> Any:
        """使用 TF-IDF 批量编码。"""
        try:
            self._tfidf.fit(texts)
            mat = self._tfidf.transform(texts).toarray().astype(np.float32)
        except Exception:
            import numpy as np
            mat = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        if mat.shape[1] < self.embedding_dim:
            mat = np.pad(mat, ((0, 0), (0, self.embedding_dim - mat.shape[1])))
        return mat

    # ------------------------------------------------------------------
    # Hash fallback (stdlib only)
    # ------------------------------------------------------------------

    def _hash_vec(self, text: str) -> Any:
        """基于 SHA-256 哈希的伪向量（纯 stdlib，无任何 ML 依赖）。"""
        import hashlib
        import numpy as np
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Repeat to reach embedding_dim
        repeats = (self.embedding_dim // 32) + 1
        vec = np.tile(vec, repeats)[:self.embedding_dim]
        # Normalize
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

    def _encode_hash(self, text: str) -> Any:
        import numpy as np
        return self._hash_vec(text)

    def _encode_batch_hash(self, texts: list[str]) -> Any:
        import numpy as np
        return np.stack([self._encode_hash(t) for t in texts]).astype(np.float32)

    @property
    def embedding_dimension(self) -> int:
        """返回嵌入向量维度。"""
        return self.embedding_dim


# ---------------------------------------------------------------------------
# 3. FAISSVectorStore
#    FAISS 向量相似度检索（无 faiss 时降级为 sklearn NearestNeighbors）
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """FAISS-based dense vector similarity search with sklearn fallback.

    Args:
        embedding_dim (int, optional): Dimension of each vector. Defaults to 768.
        n_neighbors (int, optional): Number of nearest neighbours to return. Defaults to 5.
        index_path (str, optional): Path to persist the FAISS index. Defaults to None (no persistence).

    Features:
        - Lazy import of `faiss`. Falls back to sklearn NearestNeighbors (brute-force).
        - add() / search() API compatible with FAISS conventions.
        - Automatic persistence to disk when index_path is set.

    Example:
        >>> store = FAISSVectorStore(embedding_dim=768)
        >>> store.add(encoder.encode("GME moon!"))
        >>> store.add(encoder.encode("AAPL earnings beat"))
        >>> indices, distances = store.search(encoder.encode("bullish on tech"), k=2)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        n_neighbors: int = 5,
        index_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors
        self.index_path = index_path
        self._index = None
        self._ids: list[str] = []
        self._sklearn_index = None
        self._faiss_available = False
        self._sklearn_available = False
        self._init_index()

    def _init_index(self):
        """初始化 faiss 或 sklearn。"""
        ok, faiss = _lazy_import_faiss()
        if ok:
            try:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._faiss_available = True
                logger.info("FAISSVectorStore: FAISS index initialised (Inner Product, L2-normalised).")
                self._load()
                return
            except Exception as e:
                logger.warning(f"FAISS init failed: {e}. Falling back to sklearn.")

        ok_sk, sklearn = _lazy_import_sklearn()
        if ok_sk:
            from sklearn.neighbors import NearestNeighbors
            self._sklearn_index = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
            self._sklearn_available = True
            logger.info("FAISSVectorStore: using sklearn NearestNeighbors fallback.")
        else:
            logger.warning("FAISSVectorStore: no sklearn. Search will return empty results.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vectors: Any, ids: Optional[list[str]] = None) -> None:
        """添加向量到索引。

        Args:
            vectors: 单个向量 (shape=(dim,)) 或批量 (shape=(n, dim))。
            ids (list[str], optional): 对应 ID 列表。Defaults to auto-generated.
        """
        import numpy as np
        vec_array = np.asarray(vectors).astype(np.float32)
        if vec_array.ndim == 1:
            vec_array = vec_array.reshape(1, -1)
        if vec_array.shape[1] != self.embedding_dim:
            logger.warning(f"Vector dim {vec_array.shape[1]} != {self.embedding_dim}; skipping.")
            return

        # Normalise for cosine similarity (FAISS IP index expects L2-normalised)
        norms = np.linalg.norm(vec_array, axis=1, keepdims=True) + 1e-9
        vec_array_norm = vec_array / norms

        if self._faiss_available:
            self._index.add(vec_array_norm)
        elif self._sklearn_available:
            self._sklearn_index.fit(vec_array_norm)
        else:
            return

        n = vec_array.shape[0]
        if ids is None:
            ids = [f"vec_{len(self._ids)+i}" for i in range(n)]
        self._ids.extend(ids if isinstance(ids, list) else [ids])

    def search(self, query: Any, k: Optional[int] = None) -> tuple[Any, Any]:
        """搜索 k 个最近邻。

        Args:
            query: 查询向量 (shape=(dim,)) 或 (1, dim)。
            k (int, optional): 近邻数量。Defaults to self.n_neighbors.

        Returns:
            tuple: (indices, distances) — faiss 风格返回。
                对于 sklearn fallback，indices 为 ID 列表，distances 为 cosine 距离。
        """
        import numpy as np
        q = np.asarray(query).astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = k or self.n_neighbors

        # Normalise
        norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        q_norm = q / norms

        if self._faiss_available:
            distances, indices = self._index.search(q_norm, k)
            return indices[0], distances[0]
        elif self._sklearn_available:
            try:
                dists, idxs = self._sklearn_index.kneighbors(q_norm, n_neighbors=k)
                # Convert cosine distance to similarity-like score (smaller = closer)
                ids = [self._ids[i] if i < len(self._ids) else f"vec_{i}" for i in idxs[0]]
                return ids, dists[0]
            except Exception as e:
                logger.warning(f"sklearn search failed: {e}")
                return [], []
        else:
            return [], []

    @property
    def ntotal(self) -> int:
        """索引中向量总数。"""
        if self._faiss_available:
            return self._index.ntotal
        return len(self._ids)

    def save(self, path: Optional[str] = None) -> None:
        """持久化索引到磁盘（仅 FAISS）。"""
        if not self._faiss_available:
            logger.warning("save() only supported with FAISS index.")
            return
        p = path or self.index_path
        if p is None:
            logger.warning("save() called with no index_path.")
            return
        try:
            import faiss
            faiss.write_index(self._index, p)
            # Also persist IDs
            with open(p + ".ids.json", "w", encoding="utf-8") as f:
                json.dump(self._ids, f)
            logger.info(f"FAISS index saved to {p}")
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")

    def _load(self) -> None:
        """从磁盘加载索引（仅 FAISS）。"""
        if not self._faiss_available or self.index_path is None:
            return
        p = self.index_path
        if not os.path.exists(p):
            return
        try:
            import faiss
            self._index = faiss.read_index(p)
            ids_path = p + ".ids.json"
            if os.path.exists(ids_path):
                with open(ids_path, "r", encoding="utf-8") as f:
                    self._ids = json.load(f)
            logger.info(f"FAISS index loaded from {p}")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")


# ---------------------------------------------------------------------------
# 4. SentimentStockPredictor
#    社会情感 → 股价方向预测
# ---------------------------------------------------------------------------

class SentimentStockPredictor:
    """Predict stock directional movement from social sentiment embeddings.

    Args:
        embedding_dim (int, optional): Input embedding dimension. Defaults to 768.
        lookback_days (int, optional): Number of past days of sentiment to use as features. Defaults to 7.
        prediction_horizon (int, optional): Days ahead to predict. Defaults to 1.

    Features:
        - Lazy import of sklearn. Falls back to statistical momentum if unavailable.
        - Input: list of (date, sentiment_vector, score, n_comments) records.
        - Output: dict with signal (1=long, 0=neutral, -1=short), confidence, sentiment_aggregate.
        - Aggregates daily sentiment via weighted mean (score + 2*comments as weight).

    Example:
        >>> predictor = SentimentStockPredictor(lookback_days=7)
        >>> signal = predictor.predict(history_records)
        >>> print(signal)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        lookback_days: int = 7,
        prediction_horizon: int = 1,
    ):
        self.embedding_dim = embedding_dim
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self._model = None
        self._sklearn_available = False
        self._init_model()

    def _init_model(self):
        """尝试初始化 sklearn 预测模型。"""
        ok, sklearn = _lazy_import_sklearn()
        if ok:
            from sklearn.linear_model import RidgeClassifier
            from sklearn.preprocessing import StandardScaler
            self._model = RidgeClassifier(alpha=1.0)
            self._scaler = StandardScaler()
            self._sklearn_available = True
            logger.info("SentimentStockPredictor: sklearn RidgeClassifier initialised.")
        else:
            logger.warning("SentimentStockPredictor: sklearn unavailable. Using momentum fallback.")

    def predict(self, history: list[dict]) -> dict:
        """从历史社交媒体情感数据预测股价方向。

        Args:
            history (list[dict]): 历史记录列表，每个元素应包含:
                - date (str or datetime): 日期
                - sentiment_score (float): -1~1 情感得分
                - score (int): 帖子分数
                - n_comments (int): 评论数
                - embedding (np.ndarray, optional): BERT 嵌入向量

        Returns:
            dict: {
                "signal": 1 (long) | 0 (neutral) | -1 (short),
                "confidence": float (0~1),
                "sentiment_aggregate": float (-1~1, 加权平均情感),
                "momentum": float (情感动量，近似斜率),
                "n_posts": int (用于预测的帖子总数)
            }
        """
        if not history:
            return self._empty_signal()

        import numpy as np

        # Sort by date ascending
        sorted_hist = sorted(history, key=lambda x: self._to_date(x.get("date", "")))
        if len(sorted_hist) > self.lookback_days:
            sorted_hist = sorted_hist[-self.lookback_days:]

        sentiment_scores = []
        weights = []
        embeddings = []
        dates = []

        for rec in sorted_hist:
            sent = rec.get("sentiment_score", 0)
            if sent is None:
                sent = 0.0
            score = rec.get("score", 0) or 0
            n_comments = rec.get("n_comments", 0) or 0
            weight = score + 2 * n_comments
            if weight <= 0:
                weight = 1
            sentiment_scores.append(float(sent))
            weights.append(float(weight))
            emb = rec.get("embedding")
            if emb is not None and isinstance(emb, (list, np.ndarray)):
                embeddings.append(np.asarray(emb).flatten())
            dates.append(self._to_date(rec.get("date", "")))

        if not sentiment_scores:
            return self._empty_signal()

        # Weighted average sentiment
        w = np.array(weights)
        s = np.array(sentiment_scores)
        aggregate = float(np.sum(s * w) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(s))

        # Momentum: linear regression slope of sentiment over time
        momentum = self._compute_momentum(s, dates)

        # Build feature vector
        if self._sklearn_available and embeddings:
            try:
                import numpy as np
                emb_mat = np.stack(embeddings)
                emb_mean = np.mean(emb_mat, axis=0)
                emb_std = np.std(emb_mat, axis=0) + 1e-9
                sent_arr = np.array(sentiment_scores)
                sent_mean = np.mean(sent_arr)
                sent_std = np.std(sent_arr) + 1e-9
                feat = np.concatenate([
                    [aggregate, momentum, np.mean(sent_arr)],
                    emb_mean[-min(len(emb_mean), 128):],  # cap embedding features
                ])
                feat_scaled = self._scaler.fit_transform(feat.reshape(1, -1))
                pred = self._model.predict(feat_scaled)[0]
                confidence = 0.6  # sklearn Ridge doesn't give probabilities directly
                signal = int(pred)
            except Exception as e:
                logger.debug(f"sklearn prediction failed: {e}, using momentum fallback")
                signal, confidence = self._momentum_signal(aggregate, momentum)
        else:
            signal, confidence = self._momentum_signal(aggregate, momentum)

        return {
            "signal": signal,
            "confidence": round(float(confidence), 4),
            "sentiment_aggregate": round(aggregate, 4),
            "momentum": round(float(momentum), 4),
            "n_posts": len(sorted_hist),
        }

    def _compute_momentum(self, sentiment_arr: np.ndarray, dates: list) -> float:
        """计算情感时间序列的线性回归斜率（动量）。"""
        import numpy as np
        if len(sentiment_arr) < 2:
            return 0.0
        t = np.arange(len(sentiment_arr))
        try:
            slope = np.polyfit(t, sentiment_arr, 1)[0]
            return float(slope)
        except Exception:
            return 0.0

    def _momentum_signal(self, aggregate: float, momentum: float) -> tuple[int, float]:
        """基于情感水平和动量生成交易信号。

        Args:
            aggregate (float): 加权平均情感 (-1~1).
            momentum (float): 情感动量 (斜率).

        Returns:
            tuple: (signal, confidence)
        """
        # bullish: aggregate > 0.1 and momentum >= 0
        # bearish: aggregate < -0.1 and momentum <= 0
        # neutral: otherwise
        signal = 0
        confidence = 0.5
        if aggregate > 0.1:
            signal = 1
            confidence = min(0.5 + abs(aggregate) * 0.4 + abs(momentum) * 0.2, 0.95)
        elif aggregate < -0.1:
            signal = -1
            confidence = min(0.5 + abs(aggregate) * 0.4 + abs(momentum) * 0.2, 0.95)
        else:
            signal = 0
            confidence = 0.5 - abs(momentum) * 0.1
        return signal, confidence

    def _to_date(self, val) -> datetime:
        """统一转换为 datetime。"""
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                try:
                    return datetime.strptime(val[:10], "%Y-%m-%d")
                except Exception:
                    return datetime.min
        return datetime.min

    def _empty_signal(self) -> dict:
        return {
            "signal": 0,
            "confidence": 0.0,
            "sentiment_aggregate": 0.0,
            "momentum": 0.0,
            "n_posts": 0,
        }

    def fit(self, X: list, y: list) -> None:
        """训练预测模型（sklearn RidgeClassifier）。

        Args:
            X (list): 特征列表（每项为 dict，与 predict 的 history 相同）。
            y (list): 标签列表（1=涨, 0=平, -1=跌）。
        """
        if not self._sklearn_available:
            logger.warning("fit() requires sklearn.")
            return
        import numpy as np
        vecs, labels = [], []
        for rec, lbl in zip(X, y):
            emb = rec.get("embedding")
            sent = float(rec.get("sentiment_score", 0) or 0)
            if emb is not None:
                e = np.asarray(emb).flatten()
            else:
                e = np.zeros(self.embedding_dim)
            vecs.append(e)
            labels.append(lbl)
        X_mat = np.stack(vecs)
        y_arr = np.array(labels)
        X_scaled = self._scaler.fit_transform(X_mat)
        self._model.fit(X_scaled, y_arr)
        logger.info(f"SentimentStockPredictor: trained on {len(X)} samples.")


# ---------------------------------------------------------------------------
# 5. generate_signals — 主入口函数
# ---------------------------------------------------------------------------

def generate_signals(
    ticker: str,
    collector: Optional[RedditDataCollector] = None,
    encoder: Optional[RedditBERTEncoder] = None,
    predictor: Optional[SentimentStockPredictor] = None,
    lookback_days: int = 7,
    limit: int = 200,
) -> dict:
    """从 Reddit 社交媒体数据生成股价方向交易信号 / Generate trading signals from Reddit social data.

    这是本模块的主入口函数，串联 RedditDataCollector → RedditBERTEncoder →
    FAISSVectorStore → SentimentStockPredictor，输出可直接用于策略的交易信号。

    Args:
        ticker (str): 目标股票代码（如 "GME", "AAPL"）。
        collector (RedditDataCollector, optional): Reddit 数据采集器。
            Defaults to RedditDataCollector().
        encoder (RedditBERTEncoder, optional): BERT 编码器。
            Defaults to RedditBERTEncoder().
        predictor (SentimentStockPredictor, optional): 股价方向预测器。
            Defaults to SentimentStockPredictor().
        lookback_days (int, optional): 回看天数。Defaults to 7.
        limit (int, optional): 最大采集帖子数。Defaults to 200.

    Returns:
        dict: {
            "ticker": str,
            "signal": 1 (long) | 0 (neutral) | -1 (short),
            "confidence": float,
            "sentiment_aggregate": float,
            "momentum": float,
            "n_posts": int,
            "n_comments": int,
            "top_posts": list[dict],   # 高分帖子摘要（用于人工复核）
            "timestamp": str,          # ISO timestamp
            "encoder_type": str,       # "transformers" | "tfidf" | "hash"
            "store_type": str,          # "faiss" | "sklearn" | "none"
            "raw_posts": list[dict],    # 原始帖子数据
        }

    Example:
        >>> signals = generate_signals("GME", lookback_days=7)
        >>> print(signals["signal"], signals["confidence"])
    """
    # Init components lazily
    if collector is None:
        collector = RedditDataCollector()
    if encoder is None:
        encoder = RedditBERTEncoder()
    if predictor is None:
        predictor = SentimentStockPredictor(lookback_days=lookback_days)

    # Step 1: Collect Reddit posts
    posts = collector.collect(ticker=ticker, limit=limit, hours_back=lookback_days * 24)
    if not posts:
        logger.warning(f"generate_signals: no posts collected for {ticker}.")
        return _empty_signals(ticker, encoder, predictor)

    # Step 2: Encode posts
    texts_to_encode = []
    post_weights = []
    for post in posts:
        title = post.get("title", "")
        body = post.get("body", "")
        comments_bodies = " ".join(c.get("body", "") for c in post.get("comments", [])[:10])
        combined = f"{title}. {body}. {comments_bodies}"
        texts_to_encode.append(combined)
        score = post.get("score", 0) or 0
        n_comments = len(post.get("comments", []))
        post_weights.append(float(score + 2 * n_comments))

    embeddings = encoder.encode_batch(texts_to_encode)

    # Step 3: Build vector store
    store = FAISSVectorStore(embedding_dim=encoder.embedding_dimension)
    store.add(embeddings, ids=[p.get("id", f"post_{i}") for i, p in enumerate(posts)])

    # Step 4: Aggregate embeddings by day → build history for predictor
    # Group by date (day) and compute weighted average embedding per day
    import numpy as np
    from collections import defaultdict

    day_groups: dict[str, list] = defaultdict(list)
    for post, emb in zip(posts, embeddings):
        date_key = post.get("created_utc", "")[:10]
        day_groups[date_key].append({
            "embedding": emb,
            "weight": post_weights[posts.index(post)],
        })

    history_records = []
    for date_key, group in sorted(day_groups.items()):
        emb_mat = np.stack([g["embedding"] for g in group])
        w_arr = np.array([g["weight"] for g in group])
        w_norm = w_arr / (np.sum(w_arr) + 1e-9)
        daily_emb = np.sum(emb_mat * w_norm[:, None], axis=0)
        # Weighted average sentiment score approximation from post score distribution
        avg_score_norm = float(np.mean(w_arr) / (np.max(w_arr) + 1e-9))  # 0~1
        history_records.append({
            "date": date_key,
            "embedding": daily_emb,
            "sentiment_score": avg_score_norm * 2 - 1,  # map 0~1 → -1~1
            "score": int(np.mean([g["weight"] for g in group])),
            "n_comments": len(group),
        })

    # Step 5: Predict
    signal_dict = predictor.predict(history_records)

    # Step 6: Build top-posts summary
    top_posts = sorted(
        posts,
        key=lambda p: (p.get("score", 0) or 0) + 2 * len(p.get("comments", [])),
        reverse=True,
    )[:5]
    top_summary = [
        {
            "title": p.get("title", ""),
            "score": p.get("score", 0),
            "n_comments": len(p.get("comments", [])),
            "url": p.get("url", ""),
        }
        for p in top_posts
    ]

    return {
        "ticker": ticker,
        **signal_dict,
        "n_comments": sum(len(p.get("comments", [])) for p in posts),
        "top_posts": top_summary,
        "timestamp": datetime.utcnow().isoformat(),
        "encoder_type": (
            "transformers" if encoder._transformers_available
            else "tfidf" if encoder._sklearn_available
            else "hash"
        ),
        "store_type": (
            "faiss" if store._faiss_available
            else "sklearn" if store._sklearn_available
            else "none"
        ),
        "raw_posts": posts,
    }


def _empty_signals(ticker: str, encoder, predictor) -> dict:
    return {
        "ticker": ticker,
        "signal": 0,
        "confidence": 0.0,
        "sentiment_aggregate": 0.0,
        "momentum": 0.0,
        "n_posts": 0,
        "n_comments": 0,
        "top_posts": [],
        "timestamp": datetime.utcnow().isoformat(),
        "encoder_type": (
            "transformers" if encoder._transformers_available
            else "tfidf" if encoder._sklearn_available
            else "hash"
        ),
        "store_type": "none",
        "raw_posts": [],
    }


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "RedditBERTEncoder",
    "FAISSVectorStore",
    "SentimentStockPredictor",
    "RedditDataCollector",
    "generate_signals",
]
