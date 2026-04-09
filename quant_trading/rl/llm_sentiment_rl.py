"""
llm_sentiment_rl.py — End-to-end LLM sentiment / risk signal pipeline for RL trading.

Adapted from FinRL-DAPO-SR (IEEE IDS 2025 Contest 2nd Place Solution).

Purpose
-------
This module provides the complete data-processing and runtime pipeline that
connects raw textual market data (news headlines, filings, transcripts) to
RL-augmented training and inference in the DAPO agent.

The pipeline has three stages:

1. **Signal Generation** — call DeepSeek (or a compatible LLM) to translate
   free-text into discrete sentiment (1-5) and risk (1-5) scores.

2. **State Augmentation** — merge those scores into the per-ticker time-series
   DataFrame so that :class:`~quant_trading.rl.StockTradingEnvLLM` can read
   them as part of the RL state.

3. **Reward Adjustment** — at *train time* inside the DAPO loop, apply the
   portfolio-weighted adjustment formula::

       r' = r × (S_f^alpha) / (R_f^beta + eps)

   to modulate the raw financial reward with the LLM intelligence.

For **batch pre-processing** (producing training CSVs) use
:func:`enrich_dataframe`.  For **live / real-time inference** use
:class:`SentimentPipeline`.

Integration with existing RL stack
----------------------------------
The pipeline is intentionally orthogonal to the policy algorithm.  It works
with any RL environment that exposes the standard Gymnasium interface, but is
designed to slot into :mod:`quant_trading.rl.dapo_agent.DAPOAgent` via the
``adjustment_type`` / ``alpha`` / ``beta`` constructor arguments.

Example — Batch Pre-processing
------------------------------
>>> from quant_trading.rl.llm_sentiment_rl import enrich_dataframe
>>> enriched = enrich_dataframe(
...     df=price_df,
...     text_column="headline",
...     api_key=os.environ["DEEPSEEK_API_KEY"],
... )
>>> enriched.to_csv("train_data_deepseek_2013_2018.csv", index=False)

Example — Real-time Pipeline
-----------------------------
>>> from quant_trading.rl.llm_sentiment_rl import SentimentPipeline
>>> pipeline = SentimentPipeline(api_key=os.environ["DEEPSEEK_API_KEY"])
>>> sentiment, risk = pipeline.score(text="Fed signals rate cuts amid slowing inflation")
>>> print(f"Sentiment={sentiment} (risk={risk})")

Example — DAPO Training with LLM signals
----------------------------------------
>>> from quant_trading.rl import StockTradingEnvLLM, DAPOAgent
>>> from quant_trading.rl.llm_sentiment_rl import prepare_env_kwargs
>>>
>>> env_cfg = prepare_env_kwargs(
...     df=enriched_df,
...     stock_dim=5,
...     hmax=100,
...     initial_amount=1_000_000,
... )
>>> agent = DAPOAgent(
...     env_fn=lambda: StockimentLLMEnv(**env_cfg),
...     adjustment_type="both",
...     alpha=1.0,
...     beta=1.0,
... )
>>> agent.train(epochs=50)

References
----------
- FinRL-DAPO-SR: D:/Hive/Data/trading_repos/FinRL-DAPO-SR/
- train_dapo_llm_risk.py — original training script
- dapo_llm_features.py — on-policy reward adjustment helpers
- sentiment_signal.py — DeepSeek API wrapper
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SentimentPipeline",
    "enrich_dataframe",
    "prepare_env_kwargs",
    "compute_portfolio_adjustment",
    "LLM_SENTIMENT_NEUTRAL",
    "LLM_RISK_NEUTRAL",
    "SENTIMENT_LABELS",
    "RISK_LABELS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_SENTIMENT_NEUTRAL = 3
LLM_RISK_NEUTRAL = 3

SENTIMENT_LABELS = {
    1: "strong_sell",
    2: "moderate_sell",
    3: "neutral",
    4: "moderate_buy",
    5: "strong_buy",
}

RISK_LABELS = {
    1: "very_safe",
    2: "moderately_safe",
    3: "neutral_risk",
    4: "moderately_risky",
    5: "very_risky",
}

# Score -> multiplicative weight (used in the DAPO reward formula)
SENTIMENT_WEIGHT_MAP = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
RISK_WEIGHT_MAP = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}

# Default technical indicators (consistent with FinRL-DAPO-SR)
DEFAULT_INDICATORS = ["macd", "rsi", "cci", "dx", "boll", "closebb", "volbb"]


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def label_sentiment(score: int) -> str:
    """Return human-readable label for a sentiment score."""
    return SENTIMENT_LABELS.get(score, LLM_SENTIMENT_NEUTRAL)


def label_risk(score: int) -> str:
    """Return human-readable label for a risk score."""
    return RISK_LABELS.get(score, LLM_RISK_NEUTRAL)


# ---------------------------------------------------------------------------
# DeepSeek API wrapper (lazy import)
# ---------------------------------------------------------------------------

def _get_deepseek_client(api_key: Optional[str] = None):
    """Return an authenticated DeepSeek / OpenAI-compatible client."""
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DeepSeek API key not found. Set DEEPSEEK_API_KEY in your environment "
            "or pass api_key=<your_key> to SentimentPipeline."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required for DeepSeek API. "
            "Install with: pip install openai"
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=15.0,
    )


def _build_sentiment_prompt(text: str) -> str:
    return (
        "You are a quantitative finance analyst. "
        "Classify the following market-related text into one of five sentiment categories:\n"
        "1 = strong sell signal\n"
        "2 = moderate sell signal\n"
        "3 = neutral / hold\n"
        "4 = moderate buy signal\n"
        "5 = strong buy signal\n\n"
        "Respond with ONLY the integer category (1-5). No explanation. "
        f"Text: {text[:2000]}"
    )


def _build_risk_prompt(text: str) -> str:
    return (
        "You are a quantitative risk analyst. "
        "Classify the following market-related text into one of five risk categories:\n"
        "1 = very safe / low risk\n"
        "2 = moderately safe\n"
        "3 = neutral risk\n"
        "4 = moderately risky\n"
        "5 = very risky / high risk\n\n"
        "Respond with ONLY the integer category (1-5). No explanation. "
        f"Text: {text[:2000]}"
    )


def _parse_score(raw: str) -> int:
    """Parse and clamp a [1-5] integer score from LLM output."""
    raw = raw.strip().rstrip(".").rstrip("!")
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        import re
        digits = re.findall(r"\d", raw)
        if digits:
            return max(1, min(5, int(digits[0])))
        raise ValueError(f"Cannot parse score from: {raw!r}")


# ---------------------------------------------------------------------------
# SentimentPipeline — real-time scoring object
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """
    Stateful object for real-time LLM sentiment/risk scoring.

    Wraps the DeepSeek API and exposes a simple ``score(text)`` interface.
    Results are cached in-memory to avoid redundant API calls during a
    trading session.

    Parameters
    ----------
    api_key : str, optional
        DeepSeek API key.  If omitted reads from ``DEEPSEEK_API_KEY``.
    model : str
        DeepSeek model name (default: ``"deepseek-chat"``).
    rate_limit_delay : float
        Seconds to sleep between API calls to avoid 429 errors
        (default: 0.1).
    cache_size : int
        Maximum number of entries in the LRU cache (default: 10 000).

    Example
    -------
    >>> pipeline = SentimentPipeline()
    >>> sent, risk = pipeline.score("Fed signals rate cuts amid slowing inflation")
    >>> print(f"Sentiment: {SENTIMENT_LABELS[sent]}, Risk: {RISK_LABELS[risk]}")
    Sentiment: moderate_buy, Risk: moderately_safe
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        rate_limit_delay: float = 0.1,
        cache_size: int = 10_000,
    ):
        self._api_key = api_key
        self._model = model
        self._delay = rate_limit_delay
        self._cache: dict[str, tuple[int, int]] = {}
        self._cache_size = cache_size
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            self._client = _get_deepseek_client(self._api_key)
        return self._client

    def _cache_key(self, text: str) -> str:
        # Simple deterministic key: first 128 chars + length
        return f"{text[:128]}::{len(text)}"

    def score(
        self,
        text: str,
        sentiment_only: bool = False,
    ) -> Union[tuple[int, int], int]:
        """
        Score a single piece of text.

        Parameters
        ----------
        text : str
            Market-related text (headline, headline, macro commentary).
        sentiment_only : bool
            If True, return only the sentiment score (int) instead of the
            (sentiment, risk) tuple (default: False).

        Returns
        -------
        int  or  (int, int)
            Sentiment score [1-5] or (sentiment, risk) tuple.
        """
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0] if sentiment_only else cached

        try:
            # Generate sentiment
            sentiment_response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _build_sentiment_prompt(text)}],
                temperature=0.0,
                max_tokens=4,
            )
            sentiment = _parse_score(
                sentiment_response.choices[0].message.content.strip()
            )

            # Generate risk
            risk_response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _build_risk_prompt(text)}],
                temperature=0.0,
                max_tokens=4,
            )
            risk = _parse_score(risk_response.choices[0].message.content.strip())

        except Exception as exc:
            logger.warning(f"SentimentPipeline API call failed: {exc}. "
                           "Returning neutral scores (3, 3).")
            sentiment = LLM_SENTIMENT_NEUTRAL
            risk = LLM_RISK_NEUTRAL

        result = (sentiment, risk)
        self._cache[key] = result

        # Simple size-gated eviction
        if len(self._cache) > self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        return sentiment if sentiment_only else result

    def clear_cache(self) -> None:
        """Clear the in-memory score cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Batch enrichment helpers
# ---------------------------------------------------------------------------

def enrich_dataframe(
    df: pd.DataFrame,
    text_column: str,
    api_key: Optional[str] = None,
    sentiment_col: str = "llm_sentiment",
    risk_col: str = "llm_risk",
    sentiment_only: bool = False,
    skip_existing: bool = True,
    delay: float = 0.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enrich a price/feature DataFrame with LLM-derived sentiment and risk scores.

    This is a **batch pre-processing** function intended for offline use
    (producing training CSVs).  It iterates over every row that has non-null
    ``text_column`` and calls DeepSeek for both sentiment and risk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a ``text_column`` and a ``date`` column.
    text_column : str
        Column containing the text to analyse (e.g. "headline", "news").
    api_key : str, optional
        DeepSeek API key.
    sentiment_col, risk_col : str
        Names for the output columns (default: ``"llm_sentiment"``,
        ``"llm_risk"``).
    sentiment_only : bool
        If True, only generate sentiment scores (skip risk calls).
    skip_existing : bool
        If True (default), rows that already have a non-null value in both
        ``sentiment_col`` and ``risk_col`` are not re-processed.  This makes
        the function idempotent and suitable for resuming interrupted runs.
    delay : float
        Seconds to sleep between API calls (rate-limit guard, default: 0.1).
    verbose : bool
        If True, log progress every 50 rows (default: True).

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two new columns.

    Example
    -------
    >>> price_df = pd.read_csv("daily_prices.csv")
    >>> news_df = pd.read_csv("daily_news.csv")
    >>> merged = price_df.merge(news_df, on=["date", "tic"])
    >>> enriched = enrich_dataframe(
    ...     merged,
    ...     text_column="headline",
    ...     api_key=os.environ["DEEPSEEK_API_KEY"],
    ... )
    >>> enriched.to_csv("train_data_deepseek_2013_2018.csv", index=False)
    """
    pipeline = SentimentPipeline(api_key=api_key)

    df = df.copy()

    if sentiment_col not in df.columns:
        df[sentiment_col] = LLM_SENTIMENT_NEUTRAL
    if risk_col not in df.columns:
        df[risk_col] = LLM_RISK_NEUTRAL

    mask = df[text_column].notna()
    if skip_existing:
        mask = mask & (
            df[sentiment_col].isna() | (df[sentiment_col] == LLM_SENTIMENT_NEUTRAL)
        ) | (
            df[risk_col].isna() | (df[risk_col] == LLM_RISK_NEUTRAL)
        )

    total = mask.sum()
    if total == 0:
        logger.info("enrich_dataframe: no rows to process.")
        return df

    if verbose:
        logger.info(f"enrich_dataframe: processing {total} rows...")

    for idx, (_, row) in enumerate(df[mask].iterrows()):
        if verbose and (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{total} rows...")

        text = str(row[text_column])[:2000]

        if sentiment_only:
            sentiment = pipeline.score(text, sentiment_only=True)
            df.at[idx, sentiment_col] = sentiment
        else:
            sentiment, risk = pipeline.score(text)
            df.at[idx, sentiment_col] = sentiment
            df.at[idx, risk_col] = risk

        time.sleep(delay)

    if verbose:
        logger.info(f"enrich_dataframe: done. Processed {total} rows.")

    return df


# ---------------------------------------------------------------------------
# Environment configuration helper
# ---------------------------------------------------------------------------

def prepare_env_kwargs(
    df: pd.DataFrame,
    stock_dim: int,
    hmax: int = 100,
    initial_amount: int = 1_000_000,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    reward_scaling: float = 1e-4,
    tech_indicator_list: Optional[list[str]] = None,
    turbulence_threshold: Optional[float] = None,
    llm_sentiment_col: str = "llm_sentiment",
    llm_risk_col: str = "llm_risk",
) -> dict:
    """
    Build the ``env_kwargs`` dictionary for
    :class:`~quant_trading.rl.StockTradingEnvLLM`.

    Parameters
    ----------
    df : pd.DataFrame
        Merged training / trade DataFrame that already contains
        ``llm_sentiment`` and ``llm_risk`` columns.
    stock_dim : int
        Number of distinct tickers.
    hmax, initial_amount, buy_cost_pct, sell_cost_pct, reward_scaling
        Standard :class:`StockTradingEnvLLM` parameters.
    tech_indicator_list : list[str], optional
        List of technical-indicator column names.  Defaults to
        ``DEFAULT_INDICATORS``.
    turbulence_threshold : float, optional
        turbulence cutoff for risk-sensitive trading.
    llm_sentiment_col, llm_risk_col : str
        Column names for the LLM signals in ``df``.

    Returns
    -------
    dict
        ``env_kwargs`` dict ready to unpack into ``StockTradingEnvLLM(**kwargs)``.

    Example
    -------
    >>> from quant_trading.rl import StockTradingEnvLLM, DAPOAgent
    >>> from quant_trading.rl.llm_sentiment_rl import prepare_env_kwargs
    >>> env_cfg = prepare_env_kwargs(df=enriched_df, stock_dim=5)
    >>> agent = DAPOAgent(
    ...     env_fn=lambda: StockTradingEnvLLM(**env_cfg),
    ...     adjustment_type="both",
    ... )
    >>> agent.train(epochs=50)
    """
    if tech_indicator_list is None:
        tech_indicator_list = DEFAULT_INDICATORS

    state_space = 1 + 2 * stock_dim + (2 + len(tech_indicator_list)) * stock_dim

    return {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [buy_cost_pct] * stock_dim,
        "sell_cost_pct": [sell_cost_pct] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dim,
        "reward_scaling": reward_scaling,
        "turbulence_threshold": turbulence_threshold,
        "llm_sentiment_col": llm_sentiment_col,
        "llm_risk_col": llm_risk_col,
    }


# ---------------------------------------------------------------------------
# Portfolio-level reward adjustment (used inside the DAPO training loop)
# ---------------------------------------------------------------------------

def compute_portfolio_adjustment(
    position_values: np.ndarray,
    llm_sentiments: np.ndarray,
    llm_risks: np.ndarray,
    adjustment_type: str = "both",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute the multiplicative LLM reward adjustment factor for a portfolio.

    Implements the DAPO reward formula::

        r' = r × (S_f^alpha) / (R_f^beta + eps)

    where S_f and R_f are portfolio-weighted aggregates of sentiment and
    risk scores mapped through ``SENTIMENT_WEIGHT_MAP`` and ``RISK_WEIGHT_MAP``.

    Parameters
    ----------
    position_values : np.ndarray (stock_dim,)
        Value of each stock held (price × shares). Zero entries are allowed.
    llm_sentiments : np.ndarray (stock_dim,)
        LLM sentiment scores in [1, 5] per stock.
    llm_risks : np.ndarray (stock_dim,)
        LLM risk scores in [1, 5] per stock.
    adjustment_type : {"both", "sentiment", "risk", "none"}
        Which adjustment to apply.
    alpha, beta : float
        Exponents for sentiment and risk (default: 1.0 each).

    Returns
    -------
    float
        Multiplicative factor.  Multiply the raw reward by this factor to
        obtain the LLM-adjusted reward.

    Example
    -------
    >>> positions = np.array([10000.0, 5000.0, 0.0])  # $15k total
    >>> sentiments = np.array([4, 2, 3])              # buy, sell, neutral
    >>> risks = np.array([3, 3, 3])                    # all neutral risk
    >>> factor = compute_portfolio_adjustment(positions, sentiments, risks)
    >>> adjusted_reward = raw_reward * factor
    """
    total_value = np.sum(position_values)

    if total_value == 0:
        return 1.0  # no position → no adjustment

    stock_weights = position_values / total_value

    sentiments_w = np.vectorize(SENTIMENT_WEIGHT_MAP.get)(llm_sentiments)
    risks_w = np.vectorize(RISK_WEIGHT_MAP.get)(llm_risks)

    aggregated_sentiment = np.dot(stock_weights, sentiments_w)
    aggregated_risk = np.dot(stock_weights, risks_w)

    eps = 1e-8

    if adjustment_type == "both":
        factor = (aggregated_sentiment ** alpha) / ((aggregated_risk ** beta) + eps)
    elif adjustment_type == "sentiment":
        factor = aggregated_sentiment ** alpha
    elif adjustment_type == "risk":
        factor = 1.0 / ((aggregated_risk ** beta) + eps)
    else:  # "none"
        factor = 1.0

    return float(factor)


# ---------------------------------------------------------------------------
# Dataset loading utilities (from FinRL-DAPO-SR train script)
# ---------------------------------------------------------------------------

def load_nasdaq_dataset(
    risk_file: str = "train_data_deepseek_risk_2013_2018.csv",
    sentiment_file: str = "train_data_deepseek_sentiment_2013_2018.csv",
    dataset_dir: str = "./dataset",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and merge the NASDAQ 2013-2023 LLM-enriched dataset.

    The dataset is downloaded from HuggingFace ``benstaf/nasdaq_2013_2023``
    if not found locally.

    Parameters
    ----------
    risk_file, sentiment_file : str
        Local filenames within ``dataset_dir``.
    dataset_dir : str
        Directory for storing the downloaded CSV files.
    cache_dir : str, optional
        Alias for ``dataset_dir`` (for compatibility).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``llm_sentiment`` and ``llm_risk`` columns,
        indexed by a sequential ``new_idx`` derived from unique dates.
    """
    if cache_dir is not None:
        dataset_dir = cache_dir

    os.makedirs(dataset_dir, exist_ok=True)

    risk_path = os.path.join(dataset_dir, risk_file)
    sentiment_path = os.path.join(dataset_dir, sentiment_file)

    # Download risk data if missing
    if not os.path.exists(risk_path):
        logger.info(f"Downloading risk dataset to {dataset_dir}...")
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "benstaf/nasdaq_2013_2023",
                data_files="train_data_deepseek_risk_2013_2018.csv",
            )
            dataset["train"].to_csv(risk_path, index=False)
            logger.info(f"Risk dataset saved to {risk_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download risk dataset. "
                f"Please download manually from HuggingFace: "
                f"https://huggingface.co/datasets/benstaf/nasdaq_2013_2023"
            ) from exc

    # Download sentiment data if missing
    if not os.path.exists(sentiment_path):
        logger.info(f"Downloading sentiment dataset to {dataset_dir}...")
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "benstaf/nasdaq_2013_2023",
                data_files="train_data_deepseek_sentiment_2013_2018.csv",
            )
            dataset["train"].to_csv(sentiment_path, index=False)
            logger.info(f"Sentiment dataset saved to {sentiment_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download sentiment dataset. "
                f"Please download manually from HuggingFace: "
                f"https://huggingface.co/datasets/benstaf/nasdaq_2013_2023"
            ) from exc

    # Load and merge
    train_risk = pd.read_csv(risk_path)
    train_sentiment = pd.read_csv(sentiment_path)
    train = pd.merge(
        train_risk,
        train_sentiment,
        on=["date", "tic"],
        suffixes=("", "_sentiment"),
    )

    # Re-index by unique dates
    unique_dates = train["date"].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    # Fill neutral defaults for any missing LLM signals
    train["llm_sentiment"] = train["llm_sentiment"].fillna(LLM_SENTIMENT_NEUTRAL)
    train["llm_risk"] = train["llm_risk"].fillna(LLM_RISK_NEUTRAL)

    logger.info(
        f"Loaded NASDAQ dataset: {len(train)} rows, "
        f"{train['tic'].nunique()} tickers, "
        f"date range: {train['date'].min()} – {train['date'].max()}"
    )
    return train
