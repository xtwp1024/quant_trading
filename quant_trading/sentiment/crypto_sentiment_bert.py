# quant_trading/sentiment/crypto_sentiment_bert.py
"""CryptoSentimentBertRfStrat — FinBERT + Memory-Augmented Random Forest.

整合自 D:/Hive/Data/trading_repos/CryptoSentimentBertRfStrat/，
用于加密货币新闻的情绪分析和交易信号生成。

核心组件:
    CryptoSentimentAnalyzer  — 顶层封装，支持懒加载 transformers
    FinBERTWrapper           — FinBERT / 关键词 回退情绪分析
    MemoryAugmentedFeatures  — 滚动窗口记忆特征 (10日/30日)
    SentimentRandomForest    — sklearn RandomForest 策略层

主要方法:
    generate_signals() 返回情绪方向 + 置信度

依赖:
    标准库 only (keyword fallback)
    可选: transformers, huggingface_hub, sklearn, pandas, numpy

Author: 整合自 CryptoSentimentBertRfStrat (RouRouf, Rao, Lund, Hsu)
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Keyword-based sentiment fallback (stdlib only)
# --------------------------------------------------------------------------- #

# 金融/加密货币情绪词典 — 扩展版
# 正值词 → +1,  负值词 → -1
_POSITIVE_WORDS: set[str] = {
    # General financial / crypto positive
    "bullish", "bull", "rally", "rallies", "rallied", "surging", "surge", "surged",
    "soaring", "soar", "soared", "gain", "gains", "gained", "profit", "profits",
    "profitable", "up", "ups", "higher", "high", "all-time high", "ath",
    "breakout", "break out", "moon", "mooning", "pump", "pumping",
    "green", "positive", "optimism", "optimistic", "recovery", "recover",
    "growth", "growing", "grow", "expansion", "expand",
    "adoption", "approve", "approval", "etf approval", "sec approval",
    "launch", "launching", "launched", "upgrade", "upgraded",
    "partnership", "partners", "invest", "investing", "investment",
    "buy", "buying", "long", "holding", "hold", "hodl",
    "rise", "rising", "rose", "climb", "climbing", "climbed",
    "beat", "exceed", "exceeded", "record", "records",
    "success", "successful", "win", "winning", "won",
    "happy", "excited", "exciting", "breakthrough",
    # DeFi / tech
    "upgrade", "innovation", "innovative", "secure", "security",
    "integrated", "integration", "listing", "listed",
    # Fear & Greed (greed side)
    "greed", "euphoria",
}

_NEGATIVE_WORDS: set[str] = {
    # General financial / crypto negative
    "bearish", "bear", "crash", "crashed", "crashing", "crashes",
    "plunge", "plunging", "plunged", "drop", "dropped", "drops",
    "fall", "fallen", "falls", "falling", "tumbled", "tumble", "tumbling",
    "loss", "losses", "lost", "lose", "losing",
    "down", "downs", "lower", "low", "all-time low", "atl",
    "breakdown", "break down", "dump", "dumping", "sell", "selling",
    "red", "negative", "pessimism", "pessimistic", "decline", "declining",
    "decline", "contraction", "contract",
    "ban", "banned", "ban", "regulation", "regulatory", "sec", "doj", "investigation",
    "hack", "hacked", "exploit", "exploited", "scam", "fraud", "fraudulent",
    "risk", "risky", "volatile", "volatility", "uncertainty", "uncertain",
    "liquidate", "liquidation", "liquidated", "margin call",
    "short", "shorting", "shorted",
    "fear", "fear&greed", "fear and greed", "panic", "panic selling",
    "failure", "failed", "fail", "reject", "rejected", "rejection",
    "concern", "concerns", "concerned", "warning", "warned",
    "delay", "delayed", "postpone", "postponed", "cancel", "cancelled",
    "unprofitable", "debt", "default", "defaulted",
    "outage", "downtime", "crisis", "collapse", "collapsing",
}

# Pattern to split text into tokens
_WORD_RE = re.compile(r"\b\w+\b", re.IGNORECASE)


def _keyword_sentiment(text: str) -> tuple[str, float, float]:
    """Compute sentiment from keyword matching.

    Pure stdlib — no external dependencies required.

    Returns:
        tuple: (label, score [-1,+1], confidence [0,1])
    """
    if not text:
        return "neutral", 0.0, 0.0

    tokens = _WORD_RE.findall(text.lower())
    pos_count = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return "neutral", 0.0, 0.0

    raw = (pos_count - neg_count) / total          # [-1, +1]
    confidence = min(1.0, total / 5.0)             # more matches → higher confidence

    if raw > 0.05:
        label = "positive"
    elif raw < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, float(raw), float(confidence)


# --------------------------------------------------------------------------- #
# FinBERTWrapper
# --------------------------------------------------------------------------- #

class FinBERTWrapper:
    """FinBERT 情绪分析包装器 — 支持 HuggingFace Inference API 回退到关键词.

    支持三种模式 (按优先级):
        1. ``'hf-api'`` — HuggingFace Inference API (需 HF_API_TOKEN 环境变量)
        2. ``'transformers'`` — 本地 transformers pipeline (需安装模型)
        3. ``'keyword'`` — 关键词情绪打分 (纯 stdlib，无依赖)

    Attributes:
        model_name: HuggingFace 模型 ID
        mode: 当前运行模式
        device: 推理设备 (-1=cpu, 0+=cuda)

    Example:
        >>> wrapper = FinBERTWrapper()
        >>> result = wrapper.analyze("Bitcoin ETF approval sparks rally")
        >>> print(result)
        {'sentiment': 'positive', 'score': 0.82, 'confidence': 0.95}
    """

    LABEL_MAP: dict[str, int] = {
        "positive": 1,
        "negative": -1,
        "neutral": 0,
    }

    def __init__(
        self,
        model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        mode: str = "auto",
        device: int = -1,
        hf_api_token: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace 模型 ID 或本地路径
            mode: 'auto' | 'hf-api' | 'transformers' | 'keyword'
            device: -1=cpu, 0+=cuda device index
            hf_api_token: HuggingFace API token (或从 HF_API_TOKEN env 读取)
        """
        self.model_name = model_name
        self.device = device
        self.hf_api_token = hf_api_token or _get_env("HF_API_TOKEN")

        # Resolve actual mode
        if mode == "auto":
            mode = self._detect_mode()
        self.mode: str = mode

        self._pipe = None
        self._tokenizer = None
        self._hf_client = None

        self._init_model()

    # ------------------------------------------------------------------ #
    # Mode detection
    # ------------------------------------------------------------------ #

    def _detect_mode(self) -> str:
        """Auto-detect best available mode."""
        if self.hf_api_token:
            return "hf-api"
        try:
            import transformers  # noqa: F401
            return "transformers"
        except ImportError:
            return "keyword"

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _init_model(self) -> None:
        if self.mode == "transformers":
            self._init_transformers()
        elif self.mode == "hf-api":
            self._init_hf_api()
        # else: keyword — no init needed

    def _init_transformers(self) -> None:
        """Load FinBERT via transformers pipeline."""
        try:
            from transformers import AutoTokenizer, pipeline  # noqa: F401

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=tokenizer,
                framework="pt",
                device=self.device,
                top_k=None,
            )
            self._tokenizer = tokenizer
            logger.info(
                "FinBERT (transformers) loaded: %s", self.model_name
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"transformers loading failed ({exc}). "
                "Falling back to keyword mode.",
                RuntimeWarning,
            )
            self.mode = "keyword"
            self._pipe = None
            self._tokenizer = None

    def _init_hf_api(self) -> None:
        """Init HuggingFace Inference API client."""
        try:
            from huggingface_hub import HfApi  # noqa: F401
        except ImportError:
            warnings.warn(
                "huggingface_hub not installed. Install with: pip install huggingface_hub",
                ImportWarning,
            )
            self.mode = "keyword"
            return

        self._hf_client = _HfApiClient(token=self.hf_api_token)
        logger.info("FinBERT (HF Inference API) initialized.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> dict[str, Any]:
        """分析单条文本情绪.

        Args:
            text: 输入金融/加密货币文本

        Returns:
            dict:
                - sentiment (str): 'positive' | 'negative' | 'neutral'
                - score (float): [-1, +1]
                - confidence (float): [0, 1]
                - mode (str): 使用的模型模式
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "mode": self.mode,
            }

        if self.mode == "transformers":
            return self._analyze_transformers(text)
        elif self.mode == "hf-api":
            return self._analyze_hf_api(text)
        else:
            label, score, confidence = _keyword_sentiment(text)
            return {
                "sentiment": label,
                "score": score,
                "confidence": confidence,
                "mode": self.mode,
            }

    def batch_analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量分析文本列表.

        Args:
            texts: 输入文本列表

        Returns:
            每条文本的分析结果列表
        """
        return [self.analyze(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Private — transformers
    # ------------------------------------------------------------------ #

    def _analyze_transformers(self, text: str) -> dict[str, Any]:
        """使用 transformers pipeline 分析文本."""
        try:
            # Truncate to 512 tokens
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            truncated = self._tokenizer.decode(
                encoded["input_ids"][0],
                skip_special_tokens=True,
            )
            raw = self._pipe(truncated)

            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    raw = raw[0]

            label_scores: dict[str, float] = {}
            for item in raw:
                label_scores[item["label"].lower()] = item["score"]

            sentiment = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
            raw_score = label_scores.get(sentiment, 0.0)

            if sentiment == "positive":
                score = raw_score
            elif sentiment == "negative":
                score = -raw_score
            else:
                score = 0.0

            return {
                "sentiment": sentiment,
                "score": float(score),
                "confidence": float(raw_score),
                "mode": self.mode,
            }

        except Exception as exc:
            logger.warning("transformers analysis failed: %s. Falling back to keyword.", exc)
            label, score, confidence = _keyword_sentiment(text)
            return {
                "sentiment": label,
                "score": score,
                "confidence": confidence,
                "mode": "keyword",
            }

    # ------------------------------------------------------------------ #
    # Private — HuggingFace Inference API
    # ------------------------------------------------------------------ #

    def _analyze_hf_api(self, text: str) -> dict[str, Any]:
        """使用 HuggingFace Inference API 分析文本."""
        if self._hf_client is None:
            label, score, confidence = _keyword_sentiment(text)
            return {
                "sentiment": label,
                "score": score,
                "confidence": confidence,
                "mode": "keyword",
            }

        try:
            raw = self._hf_client.query(self.model_name, text)
            # Response format: [{'label': 'positive', 'score': 0.x}, ...]
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    raw = raw[0]

            label_scores: dict[str, float] = {}
            for item in raw:
                label_scores[item["label"].lower()] = item["score"]

            sentiment = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
            raw_score = label_scores.get(sentiment, 0.0)

            if sentiment == "positive":
                score = raw_score
            elif sentiment == "negative":
                score = -raw_score
            else:
                score = 0.0

            return {
                "sentiment": sentiment,
                "score": float(score),
                "confidence": float(raw_score),
                "mode": self.mode,
            }

        except Exception as exc:
            logger.warning("HF API analysis failed: %s. Falling back to keyword.", exc)
            label, score, confidence = _keyword_sentiment(text)
            return {
                "sentiment": label,
                "score": score,
                "confidence": confidence,
                "mode": "keyword",
            }


# --------------------------------------------------------------------------- #
# HuggingFace Inference API Client (minimal, stdlib + optional requests)
# --------------------------------------------------------------------------- #

class _HfApiClient:
    """Minimal HuggingFace Inference API client — no transformers needed.

    Uses the Inference API endpoint directly with HTTP POST.
    """

    API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(self, token: str) -> None:
        self.token = token

    def query(self, model_id: str, text: str) -> list[dict[str, Any]]:
        """Call HF Inference API and return parsed JSON."""
        url = self.API_URL.format(model=model_id)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True},
        }

        # Use stdlib only — no requests dependency
        import json
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# MemoryAugmentedFeatures
# --------------------------------------------------------------------------- #

class MemoryAugmentedFeatures:
    """滚动窗口记忆特征 — 情绪 + 滚动统计.

    来自 CryptoSentimentBertRfStrat 的核心记忆增强特征：
    - Coinbase 新闻 10日滚动正面平均分
    - Coinbase 新闻 30日滚动正面平均分
    - BTC 新闻 30日滚动总分
    - Coinbase 中性新闻计数 10日滚动总和
    - Coinbase 新闻当日总分

    特征向量 (顺序固定，兼容 Random Forest 训练):
        0. coinbase_positive_avg_score_rolling_10
        1. coinbase_positive_avg_score_rolling_30
        2. btc_total_news_score_rolling_30
        3. coinbase_neutral_count_rolling_10
        4. coinbase_total_news_score          (当前值，非滚动)
        5. sentiment_latest                    (最新情绪分数)
        6. sentiment_momentum                  (情绪动量)
        7. sentiment_volatility                (情绪波动率)
        8. avg_sentiment_10                    (10日平均情绪)
        9. avg_sentiment_30                    (30日平均情绪)

    Attributes:
        rolling_windows: 滚动窗口配置 (默认 10 和 30 天)

    Example:
        >>> maf = MemoryAugmentedFeatures()
        >>> # Simulate adding daily sentiment scores
        >>> for score in [0.5, -0.2, 0.8, 0.3, -0.1]:
        ...     maf.add_day(score)
        >>> features = maf.get_feature_vector()
        >>> print(features)
        array([...])
    """

    FEATURE_NAMES: list[str] = [
        "coinbase_positive_avg_score_rolling_10",
        "coinbase_positive_avg_score_rolling_30",
        "btc_total_news_score_rolling_30",
        "coinbase_neutral_count_rolling_10",
        "coinbase_total_news_score",
        "sentiment_latest",
        "sentiment_momentum",
        "sentiment_volatility",
        "avg_sentiment_10",
        "avg_sentiment_30",
    ]

    def __init__(
        self,
        window_short: int = 10,
        window_long: int = 30,
    ) -> None:
        self.window_short = window_short
        self.window_long = window_long

        # Per-coin news rolling sentiment lists (updated by external news feed)
        self._coinbase_positive_scores: list[float] = []
        self._coinbase_neutral_counts: list[float] = []
        self._btc_total_news_scores: list[float] = []

        # Internal sentiment history (managed by add_day)
        self._sentiment_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_day(
        self,
        sentiment_score: float,
        coinbase_positive_score: float = 0.0,
        coinbase_neutral_count: float = 0.0,
        btc_total_news_score: float = 0.0,
    ) -> None:
        """记录一天的情绪数据.

        Args:
            sentiment_score: 当日综合情绪分数 [-1, +1]
            coinbase_positive_score: Coinbase 正面新闻平均分 (当前日)
            coinbase_neutral_count: Coinbase 中性新闻计数 (当前日)
            btc_total_news_score: BTC 新闻总分 (当前日)
        """
        self._sentiment_history.append(float(sentiment_score))
        self._coinbase_positive_scores.append(float(coinbase_positive_score))
        self._coinbase_neutral_counts.append(float(coinbase_neutral_count))
        self._btc_total_news_scores.append(float(btc_total_news_score))

        # Trim to window_long to prevent unbounded growth
        max_len = self.window_long
        if len(self._sentiment_history) > max_len:
            self._sentiment_history = self._sentiment_history[-max_len:]
        if len(self._coinbase_positive_scores) > max_len:
            self._coinbase_positive_scores = self._coinbase_positive_scores[-max_len:]
        if len(self._coinbase_neutral_counts) > max_len:
            self._coinbase_neutral_counts = self._coinbase_neutral_counts[-max_len:]
        if len(self._btc_total_news_scores) > max_len:
            self._btc_total_news_scores = self._btc_total_news_scores[-max_len:]

    def add_finbert_result(self, result: dict[str, Any]) -> None:
        """从 FinBERTWrapper.analyze() 结果直接更新记忆.

        Args:
            result: FinBERTWrapper.analyze() 返回的 dict
        """
        self.add_day(sentiment_score=result.get("score", 0.0))

    def get_feature_vector(self) -> "np.ndarray[Any, Any]":  # type: ignore[name-defined]
        """提取当前特征向量 (用于 RF 输入).

        Returns:
            numpy array of shape (10,) — 顺序见 FEATURE_NAMES
        """
        import numpy as np

        cb_pos_10 = float(np.sum(self._coinbase_positive_scores[-self.window_short :]))
        cb_pos_30 = float(np.sum(self._coinbase_positive_scores[-self.window_long :]))
        btc_total_30 = float(np.sum(self._btc_total_news_scores[-self.window_long :]))
        cb_neutral_10 = float(np.sum(self._coinbase_neutral_counts[-self.window_short :]))
        cb_total = (
            float(np.sum(self._coinbase_positive_scores))
            if self._coinbase_positive_scores
            else 0.0
        )

        sent = self._sentiment_history
        sent_latest = float(sent[-1]) if sent else 0.0
        sent_momentum = self._sentiment_momentum()
        sent_vol = self._sentiment_volatility()
        avg_10 = float(np.mean(sent[-self.window_short :])) if sent else 0.0
        avg_30 = float(np.mean(sent[-self.window_long :])) if sent else 0.0

        return np.array(
            [
                cb_pos_10,
                cb_pos_30,
                btc_total_30,
                cb_neutral_10,
                cb_total,
                sent_latest,
                sent_momentum,
                sent_vol,
                avg_10,
                avg_30,
            ],
            dtype=np.float64,
        )

    def get_feature_names(self) -> list[str]:
        """返回特征名称列表 (与 get_feature_vector 顺序对应)."""
        return list(self.FEATURE_NAMES)

    def reset(self) -> None:
        """重置所有滚动窗口和历史."""
        self._sentiment_history.clear()
        self._coinbase_positive_scores.clear()
        self._coinbase_neutral_counts.clear()
        self._btc_total_news_scores.clear()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _sentiment_momentum(self) -> float:
        """情绪动量: 最新 - 滚动均值 (10日)."""
        import numpy as np

        if len(self._sentiment_history) < 2:
            return 0.0
        latest = self._sentiment_history[-1]
        hist = self._sentiment_history[-self.window_short :]
        return float(latest - np.mean(hist))

    def _sentiment_volatility(self) -> float:
        """情绪波动率: 10日滚动标准差."""
        import numpy as np

        if len(self._sentiment_history) < 2:
            return 0.0
        hist = self._sentiment_history[-self.window_short :]
        return float(np.std(hist))


# --------------------------------------------------------------------------- #
# SentimentRandomForest
# --------------------------------------------------------------------------- #

class SentimentRandomForest:
    """Sklearn RandomForest 策略层 — 基于情绪 + 技术指标特征.

    来自 CryptoSentimentBertRfStrat 的随机森林策略层，
    使用以下特征组:
    - TA-Lib 技术指标 (30+ 个)
    - 记忆增强情绪特征 (MemoryAugmentedFeatures)

    Attributes:
        model: sklearn RandomForestRegressor 或 None (未训练)
        n_estimators: 树的数量
        feature_names: 特征名称列表 (与 predict 输入对应)
        is_fitted: bool — 模型是否已训练/加载

    Example:
        >>> rf = SentimentRandomForest(n_estimators=100)
        >>> # feature_vector from MemoryAugmentedFeatures + tech indicators
        >>> pred = rf.predict([feature_vector])
        >>> print(pred)
        [0.023]
    """

    def __init__(
        self,
        n_estimators: int = 100,
        model_path: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_estimators: RandomForest 树的数量
            model_path: 已训练模型 pickle 路径 (可选)
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.model_path = model_path
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_fitted: bool = False

        # Full feature names matching the original CryptoSentimentBertRfStrat
        self.feature_names: list[str] = [
            # Technical indicators (TA-Lib order from final_strategy.py)
            "AROONOSC", "ROC_rolling_10", "MOM_rolling_10", "ROC", "MFI",
            "WILLR", "RSI", "CMO", "MOM", "NATR", "WILLR_rolling_10",
            "MACDHIST", "STOCH_FASTD_rolling_10", "PLUS_DI", "PLUS_DI_rolling_10",
            "STOCH_FASTK_rolling_10", "CCI", "STOCH_FASTK", "MACDHIST_rolling_10",
            "btc_total_news_score_rolling_30", "ULTOSC",
            "coinbase_positive_avg_score_rolling_10", "STOCH_FASTD",
            "HT_PHASOR_quadrature_rolling_10", "coinbase_total_news_score",
            "coinbase_neutral_count_rolling_10", "MINUS_DI", "TRANGE_rolling_10",
            "coinbase_positive_avg_score_rolling_30", "HT_DCPHASE",
        ]

        # Fallback: memory-only feature names (when no TA-Lib data)
        self._memory_feature_names: list[str] = [
            "coinbase_positive_avg_score_rolling_10",
            "coinbase_positive_avg_score_rolling_30",
            "btc_total_news_score_rolling_30",
            "coinbase_neutral_count_rolling_10",
            "coinbase_total_news_score",
            "sentiment_latest",
            "sentiment_momentum",
            "sentiment_volatility",
            "avg_sentiment_10",
            "avg_sentiment_30",
        ]

        self._init_model(model_path)

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _init_model(self, model_path: Optional[str]) -> None:
        if model_path:
            self.load(model_path)
        else:
            self._init_sklearn()

    def _init_sklearn(self) -> None:
        """初始化 sklearn RandomForestRegressor (lazy import)."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            logger.info("RandomForestRegressor initialized (n_estimators=%d).", self.n_estimators)
        except ImportError as exc:
            warnings.warn(
                f"sklearn not available: {exc}. "
                "SentimentRandomForest will operate in prediction-only mode.",
                ImportWarning,
            )
            self.model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
        y: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
    ) -> "SentimentRandomForest":
        """训练 RandomForest 模型.

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,) — 通常是 15日移动平均收益

        Returns:
            self (链式调用)
        """
        if self.model is None:
            raise RuntimeError("sklearn not installed; cannot train model.")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("RandomForest trained on %d samples, %d features.", X.shape[0], X.shape[1])
        return self

    def predict(
        self,
        X: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
    ) -> "np.ndarray[Any, Any]":  # type: ignore[name-defined]
        """使用训练好的模型预测.

        Args:
            X: 特征矩阵 (n_samples, n_features) 或单个特征向量 (n_features,)

        Returns:
            预测值数组 (n_samples,)
        """
        import numpy as np

        # Handle single vector
        if isinstance(X, list):
            X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.model is not None and self.is_fitted:
            return self.model.predict(X)  # type: ignore[union-attr]

        # Fallback: return simple moving average of last column
        if X.shape[0] == 1:
            return np.array([np.mean(X[0, -4:])])
        return np.mean(X, axis=1)

    def predict_proba(
        self,
        X: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
    ) -> "np.ndarray[Any, Any]":  # type: ignore[name-defined]
        """预测方向概率 (分类模式).

        将回归输出转换为三分类概率:
            - positive (signal > 0.01)
            - neutral  (-0.005 <= signal <= 0.01)
            - negative (signal < -0.005)

        Args:
            X: 特征矩阵

        Returns:
            概率数组 (n_samples, 3) — [P(neg), P(neu), P(pos)]
        """
        import numpy as np

        raw = self.predict(X)
        probs = np.zeros((len(raw), 3), dtype=np.float64)

        for i, val in enumerate(raw):
            if val < -0.005:
                # Negative: high prob for negative class
                probs[i] = [0.7, 0.2, 0.1]
            elif val > 0.01:
                # Positive
                probs[i] = [0.1, 0.2, 0.7]
            else:
                # Neutral
                probs[i] = [0.2, 0.6, 0.2]

        return probs

    def apply_importance_function(self, x: float) -> float:
        """应用 CryptoSentimentBertRfStrat 的重要性函数.

        基于分段线性函数将 15DAR 预测值转换为 Kelly 系数权重。

        Args:
            x: 原始预测值 (15日移动平均收益)

        Returns:
            重要性分数 (clamped to [0.51, 1.0])
        """
        if x < -0.05:
            return -1.0
        elif -0.05 <= x < -0.004:
            return 16.433 * x - 0.1777
        elif -0.004 <= x < 0:
            return 60.0 * x
        elif 0 <= x < 0.01:
            return 40.0 * x
        elif 0.01 <= x <= 0.05:
            return 14.975 * x + 0.25
        else:
            return 1.0

    def load(self, path: str) -> "SentimentRandomForest":
        """从 pickle 文件加载已训练的模型.

        Args:
            path: 模型文件路径

        Returns:
            self
        """
        try:
            import joblib
            self.model = joblib.load(path)
            self.is_fitted = True
            logger.info("Loaded RandomForest model from %s", path)
        except Exception as exc:
            warnings.warn(f"Failed to load model from {path}: {exc}", RuntimeWarning)
            self.model = None
        return self

    def save(self, path: str) -> None:
        """保存模型到 pickle 文件.

        Args:
            path: 输出文件路径
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        try:
            import joblib
            joblib.dump(self.model, path)
            logger.info("Saved RandomForest model to %s", path)
        except Exception as exc:
            warnings.warn(f"Failed to save model to {path}: {exc}", RuntimeWarning)


# --------------------------------------------------------------------------- #
# CryptoSentimentAnalyzer — Top-level orchestrator
# --------------------------------------------------------------------------- #

class CryptoSentimentAnalyzer:
    """顶层情绪分析器 — 整合 FinBERT + Memory + RF 策略.

    CryptoSentimentBertRfStrat 项目的完整情绪分析流水线封装：

        文本输入
            │
            ▼
        FinBERTWrapper (transformers / HF API / keyword)
            │
            ▼ 情绪分数
        MemoryAugmentedFeatures (滚动窗口统计)
            │
            ▼ 特征向量
        SentimentRandomForest (技术指标 + 情绪特征)
            │
            ▼ 预测信号
        generate_signals() → direction + confidence + Kelly weight

    Attributes:
        finbert: FinBERTWrapper 实例
        memory: MemoryAugmentedFeatures 实例
        rf: SentimentRandomForest 实例

    Example:
        >>> analyzer = CryptoSentimentAnalyzer()
        >>> signals = analyzer.generate_signals(
        ...     news_texts=["BTC breaks $100k on ETF approval news"],
        ...     technical_features=None,   # TA-Lib features if available
        ... )
        >>> print(signals)
        {'direction': 'positive', 'confidence': 0.82, 'kelly_fraction': 0.65,
         'raw_pred': 0.023, 'sentiment_score': 0.71, 'mode': 'keyword'}
    """

    def __init__(
        self,
        finbert_model: Optional[str] = None,
        rf_model_path: Optional[str] = None,
        use_rf: bool = True,
        hf_api_token: Optional[str] = None,
    ) -> None:
        """
        Args:
            finbert_model: FinBERT 模型名或本地路径 (默认: distilroberta FinBERT)
            rf_model_path: 已训练 RF 模型路径 (可选)
            use_rf: 是否启用 RandomForest 预测
            hf_api_token: HuggingFace API token
        """
        self.finbert_model_name = (
            finbert_model
            or "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        self.rf_model_path = rf_model_path
        self.use_rf = use_rf

        # Lazy — initialized on first use
        self._finbert: Optional[FinBERTWrapper] = None
        self._memory: Optional[MemoryAugmentedFeatures] = None
        self._rf: Optional[SentimentRandomForest] = None

        # Store HF token for deferred init
        self._hf_token = hf_api_token

    # ------------------------------------------------------------------ #
    # Lazy accessors
    # ------------------------------------------------------------------ #

    @property
    def finbert(self) -> FinBERTWrapper:
        if self._finbert is None:
            self._finbert = FinBERTWrapper(
                model_name=self.finbert_model_name,
                hf_api_token=self._hf_token,
            )
        return self._finbert

    @property
    def memory(self) -> MemoryAugmentedFeatures:
        if self._memory is None:
            self._memory = MemoryAugmentedFeatures()
        return self._memory

    @property
    def rf(self) -> SentimentRandomForest:
        if self._rf is None:
            self._rf = SentimentRandomForest(model_path=self.rf_model_path)
        return self._rf

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_signals(
        self,
        news_texts: list[str],
        technical_features: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """生成综合交易信号.

        Args:
            news_texts: 当前周期的新闻标题/摘要列表
            technical_features: TA-Lib 技术指标字典 (可选, key 为指标名)

        Returns:
            dict:
                - direction (str): 'positive' | 'neutral' | 'negative'
                - confidence (float): 置信度 [0, 1]
                - kelly_fraction (float): Kelly 赌注比例 [0.51, 0.99]
                - raw_pred (float | None): RF 预测值 (若 RF 可用)
                - sentiment_score (float): 情绪分数 [-1, +1]
                - mode (str): FinBERT 运行模式
                - feature_vector (np.ndarray): 完整特征向量
        """
        import numpy as np

        # 1. Analyze news sentiment
        if not news_texts:
            sentiment_score = 0.0
            avg_confidence = 0.0
            mode = "keyword"
        else:
            results = self.finbert.batch_analyze(news_texts)
            # Weighted average sentiment score
            scores = [r["score"] for r in results]
            confidences = [r["confidence"] for r in results]
            total_conf = sum(confidences)
            if total_conf > 0:
                sentiment_score = sum(s * c for s, c in zip(scores, confidences)) / total_conf
            else:
                sentiment_score = sum(scores) / len(scores) if scores else 0.0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            mode = results[0].get("mode", "keyword") if results else "keyword"

        # 2. Update memory with latest sentiment
        self.memory.add_day(
            sentiment_score=sentiment_score,
            coinbase_positive_score=sentiment_score if sentiment_score > 0 else 0.0,
            coinbase_neutral_count=1.0 if abs(sentiment_score) < 0.05 else 0.0,
            btc_total_news_score=sentiment_score,
        )

        # 3. Get memory-augmented feature vector
        mem_features = self.memory.get_feature_vector()  # shape (10,)

        # 4. Merge with technical features if provided
        if technical_features and self.rf.is_fitted:
            # Build full 31-feature vector (matching RF training order)
            full_vector = self._merge_features(mem_features, technical_features)
        else:
            # Fall back to memory-only features
            full_vector = mem_features
            if self.rf.is_fitted:
                logger.debug("No TA-Lib features provided; using memory-only features.")

        # 5. RF prediction (if available and enabled)
        raw_pred: Optional[float] = None
        if self.use_rf and self.rf.is_fitted:
            raw_pred = float(self.rf.predict(full_vector)[0])

        # 6. Determine direction
        if raw_pred is not None:
            threshold_pos = 0.01
            threshold_neg = -0.005
            if raw_pred > threshold_pos:
                direction = "positive"
            elif raw_pred < threshold_neg:
                direction = "negative"
            else:
                direction = "neutral"
            importance = abs(self.rf.apply_importance_function(raw_pred))
        else:
            # No RF: use raw sentiment score
            if sentiment_score > 0.05:
                direction = "positive"
            elif sentiment_score < -0.05:
                direction = "negative"
            else:
                direction = "neutral"
            importance = abs(sentiment_score)

        # 7. Kelly fraction (clamped [0.51, 0.99])
        importance = max(0.51, min(0.99, importance + 0.4))
        kelly_fraction = (1 * abs(importance - (1 - importance))) / 1
        kelly_fraction = max(0.0, min(0.8, kelly_fraction))

        return {
            "direction": direction,
            "confidence": float(avg_confidence),
            "kelly_fraction": float(kelly_fraction),
            "raw_pred": raw_pred,
            "sentiment_score": float(sentiment_score),
            "mode": mode,
            "feature_vector": full_vector,
        }

    def analyze_text(self, text: str) -> dict[str, Any]:
        """分析单条文本 (FinBERTWrapper 封装).

        Args:
            text: 输入文本

        Returns:
            FinBERTWrapper.analyze() 结果
        """
        return self.finbert.analyze(text)

    def batch_analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量分析文本 (FinBERTWrapper 封装).

        Args:
            texts: 文本列表

        Returns:
            FinBERTWrapper.batch_analyze() 结果
        """
        return self.finbert.batch_analyze(texts)

    def reset(self) -> None:
        """重置记忆状态."""
        self.memory.reset()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _merge_features(
        self,
        mem_features: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
        tech_features: dict[str, float],
    ) -> "np.ndarray[Any, Any]":  # type: ignore[name-defined]
        """Merge memory features with TA-Lib technical indicators.

        Memory features map to the last 10 slots of the 31-feature RF vector.
        TA-Lib features fill the first 21 slots per final_strategy.py order.
        """
        import numpy as np

        # Build 21-dim tech feature vector (ordered per self.rf.feature_names)
        tech_vector = np.zeros(21, dtype=np.float64)
        for i, name in enumerate(self.rf.feature_names[:21]):
            tech_vector[i] = float(tech_features.get(name, 0.0))

        # Concatenate: [tech(21), memory(10)] = 31
        return np.concatenate([tech_vector, mem_features])


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """从环境变量读取值 (cross-platform safe)."""
    import os
    return os.environ.get(key, default)
