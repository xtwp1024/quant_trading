"""
dapo_llm_features.py — LLM-augmented feature generation for DAPO RL trading.

Adapted from FinRL-DAPO-SR (D:/Hive/Data/trading_repos/FinRL-DAPO-SR/).

Purpose
-------
Generate sentiment and risk signals that are appended to the RL state space.
These features augment the DAPO state with LLM-derived market intelligence,
enabling the policy to condition actions on qualitative market sentiment
and risk assessments.

Key idea: LLM-generated sentiment and risk are NOT trading signals directly.
They are STATE FEATURES that the RL policy learns to interpret in context.
The DAPO algorithm uses them to adjust rewards (not actions directly).

State-space integration
-----------------------
The augmented state space (per stock) is:
  [cash, price, shares, tech_indicators..., llm_sentiment, llm_risk]

Sentiment scores: 1-5 (1=strong sell, 5=strong buy, 3=neutral)
Risk scores:     1-5 (1=very safe, 5=very risky, 3=neutral)

Reward adjustment formula (from DAPO paper):
  r' = r × (S_f^alpha) / (R_f^beta + 1e-8)

Where S_f and R_f are portfolio-weighted aggregates of sentiment and risk.

References
---------
- FinRL-DAPO-SR: D:/Hive/Data/trading_repos/FinRL-DAPO-SR/
- train_dapo_llm_risk.py: dataset loading + state space construction
- dapo_algorithm.py: reward adjustment via extract_llm_features()
"""

from __future__ import annotations

from typing import Optional, Union
import os
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentiment and risk score mappings (from FinRL-DAPO-SR)
# Score -> multiplicative weight applied to portfolio reward
SENTIMENT_WEIGHT_MAP = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
RISK_WEIGHT_MAP = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}

# Neutral default score (no adjustment)
NEUTRAL_SENTIMENT = 3
NEUTRAL_RISK = 3

# ---------------------------------------------------------------------------
# Core feature generators (stateless — operate on a single ticker/date row)
# ---------------------------------------------------------------------------


def generate_sentiment_signal(
    text: str,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    sentiment_only: bool = True,
) -> Union[int, tuple[int, str]]:
    """
    Generate a sentiment score in [1, 5] from free-text market news or filings.

    Score semantics (from FinRL-DAPO-SR):
        1 = strong sell signal
        2 = moderate sell signal
        3 = neutral / hold
        4 = moderate buy signal
        5 = strong buy signal

    Parameters
    ----------
    text : str
        Raw text content (e.g. news headline, SEC filing abstract).
    model : str
        LLM model name passed to the client.
    api_key : str, optional
        DeepSeek API key. If None the function returns a stub (neutral score).
    sentiment_only : bool
        If True returns only the integer score.
        If False returns (score, reasoning_string).

    Returns
    -------
    int  or  (int, str)
        Sentiment score in [1, 5].  When sentiment_only=False, also returns
        the model's short reasoning.

    Notes
    -----
    - LLM calls are NOT made during training batches — only during data
      pre-processing or live inference when api_key is present.
    - When no api_key is set, a UserWarning is logged once and neutral (3)
      is returned.

    Example
    -------
    >>> score = generate_sentiment_signal(
    ...     "Fed signals rate cuts may be delayed amid inflation concerns",
    ...     api_key=os.environ.get("DEEPSEEK_API_KEY"),
    ... )
    >>> print(score)  # likely 1 or 2
    """
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None:
        # Stub path — no API key available
        logger.warning(
            "generate_sentiment_signal: no DEEPSEEK_API_KEY found. "
            "Returning neutral score (3)."
        )
        return (NEUTRAL_SENTIMENT, "no API key — stub") if not sentiment_only else NEUTRAL_SENTIMENT

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI client not installed. Install with: pip install openai")
        return NEUTRAL_SENTIMENT

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=15.0,
    )

    sentiment_prompt = (
        "You are a quantitative finance analyst. "
        "Classify the following market-related text into one of five sentiment categories:\n"
        "1 = strong sell signal\n"
        "2 = moderate sell signal\n"
        "3 = neutral / hold\n"
        "4 = moderate buy signal\n"
        "5 = strong buy signal\n\n"
        "Rules:\n"
        "- Respond with ONLY the integer category (1-5). No explanation.\n"
        "- For ambiguous or neutral text, prefer category 3.\n"
        f"Text: {text[:2000]}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": sentiment_prompt}],
        temperature=0.0,
        max_tokens=4,
    )

    raw = response.choices[0].message.content.strip()

    try:
        score = int(raw)
        score = max(1, min(5, score))  # clamp to [1, 5]
    except ValueError:
        logger.warning(f"LLM returned non-integer sentiment: {raw!r}. Using neutral (3).")
        score = NEUTRAL_SENTIMENT

    reasoning = getattr(response, "reasoning", "") if hasattr(response, "reasoning") else ""

    logger.debug(f"Sentiment score: {score} | text: {text[:80]!r}...")
    return (score, reasoning) if not sentiment_only else score


def generate_risk_signal(
    text: str,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    risk_only: bool = True,
) -> Union[int, tuple[int, str]]:
    """
    Generate a risk score in [1, 5] from free-text market content.

    Score semantics (from FinRL-DAPO-SR):
        1 = very safe / low risk
        2 = moderately safe
        3 = neutral risk
        4 = moderately risky
        5 = very risky / high risk

    Parameters
    ----------
    text : str
        Raw text content (e.g. earnings call transcript, macro news).
    model : str
        LLM model name passed to the client.
    api_key : str, optional
        DeepSeek API key. If None the function returns a stub (neutral score).
    risk_only : bool
        If True returns only the integer score.
        If False returns (score, reasoning_string).

    Returns
    -------
    int  or  (int, str)
        Risk score in [1, 5].  When risk_only=False, also returns reasoning.

    Notes
    -----
    - The risk score represents MARKET-WIDE risk perception for the asset,
      not idiosyncratic financial risk (volatility, VaR, etc.).
    - Higher risk score REDUCES the adjusted reward in the DAPO formula
      (reward divided by risk^beta), reflecting risk-off sentiment.

    Example
    -------
    >>> score = generate_risk_signal(
    ...     "Interest rates spike sharply; credit spreads widen across sectors",
    ...     api_key=os.environ.get("DEEPSEEK_API_KEY"),
    ... )
    >>> print(score)  # likely 4 or 5
    """
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None:
        logger.warning(
            "generate_risk_signal: no DEEPSEEK_API_KEY found. "
            "Returning neutral score (3)."
        )
        return (NEUTRAL_RISK, "no API key — stub") if not risk_only else NEUTRAL_RISK

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI client not installed. Install with: pip install openai")
        return NEUTRAL_RISK

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=15.0,
    )

    risk_prompt = (
        "You are a quantitative risk analyst. "
        "Classify the following market-related text into one of five risk categories:\n"
        "1 = very safe / low risk\n"
        "2 = moderately safe\n"
        "3 = neutral risk\n"
        "4 = moderately risky\n"
        "5 = very risky / high risk\n\n"
        "Rules:\n"
        "- Respond with ONLY the integer category (1-5). No explanation.\n"
        "- For ambiguous or neutral text, prefer category 3.\n"
        f"Text: {text[:2000]}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": risk_prompt}],
        temperature=0.0,
        max_tokens=4,
    )

    raw = response.choices[0].message.content.strip()

    try:
        score = int(raw)
        score = max(1, min(5, score))
    except ValueError:
        logger.warning(f"LLM returned non-integer risk: {raw!r}. Using neutral (3).")
        score = NEUTRAL_RISK

    reasoning = getattr(response, "reasoning", "") if hasattr(response, "reasoning") else ""

    logger.debug(f"Risk score: {score} | text: {text[:80]!r}...")
    return (score, reasoning) if not risk_only else score


# ---------------------------------------------------------------------------
# Batch utilities — pre-process a DataFrame to add llm_sentiment / llm_risk
# ---------------------------------------------------------------------------


def add_llm_features_to_dataframe(
    df: pd.DataFrame,
    text_column: str,
    api_key: Optional[str] = None,
    sentiment_col: str = "llm_sentiment",
    risk_col: str = "llm_risk",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Enrich a price/feature DataFrame with LLM-derived sentiment and risk columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a `text_column` and a `date` column.
    text_column : str
        Name of the column containing text to analyse (e.g. "news_headline").
    api_key : str, optional
        DeepSeek API key.
    sentiment_col, risk_col : str
        Names for the output columns.
    skip_existing : bool
        If True, rows that already have a non-null sentiment/risk value are
        not re-processed (useful for resuming interrupted runs).

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with two new columns.

    Notes
    -----
    - This is a BATCH pre-processing function — call it once on historical data
      to produce training CSVs, NOT on every training step.
    - DeepSeek API calls are rate-limited; a small sleep is introduced between
      calls to avoid 429 errors.

    Example
    -------
    >>> price_df = pd.read_csv("daily_prices.csv")
    >>> news_df = pd.read_csv("daily_news.csv")
    >>> merged = price_df.merge(news_df, on=["date", "tic"])
    >>> enriched = add_llm_features_to_dataframe(
    ...     merged,
    ...     text_column="headline",
    ...     api_key=os.environ["DEEPSEEK_API_KEY"],
    ... )
    >>> enriched.to_csv("train_data_deepseek_risk_2013_2018.csv", index=False)
    """
    import time

    df = df.copy()

    if sentiment_col not in df.columns:
        df[sentiment_col] = NEUTRAL_SENTIMENT
    if risk_col not in df.columns:
        df[risk_col] = NEUTRAL_RISK

    mask = df[text_column].notna() & (
        df[sentiment_col].isna() | df[risk_col].isna()
    )
    if not skip_existing:
        mask = df[text_column].notna()

    rows_to_process = df.loc[mask].iterrows()
    total = mask.sum()
    logger.info(f"add_llm_features_to_dataframe: {total} rows to process.")

    for idx, (_, row) in enumerate(rows_to_process):
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{total} rows...")

        text = str(row[text_column])[:2000]

        sentiment = generate_sentiment_signal(text, api_key=api_key)
        risk = generate_risk_signal(text, api_key=api_key)

        df.at[idx, sentiment_col] = sentiment
        df.at[idx, risk_col] = risk

        time.sleep(0.1)  # rate-limit guard

    return df


# ---------------------------------------------------------------------------
# Portfolio-weighted reward adjustment (used inside DAPO training loop)
# ---------------------------------------------------------------------------


def compute_llm_reward_adjustment(
    position_values: np.ndarray,
    llm_sentiments: np.ndarray,
    llm_risks: np.ndarray,
    adjustment_type: str = "both",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute the LLM-based reward adjustment factor for a portfolio.

    This function implements the DAPO reward adjustment formula::

        r' = r × (S_f^alpha) / (R_f^beta + eps)

    where S_f and R_f are portfolio-weighted aggregates.

    Parameters
    ----------
    position_values : np.ndarray (stock_dim,)
        Value of each stock held (price × shares). Zero entries are allowed.
    llm_sentiments : np.ndarray (stock_dim,)
        LLM sentiment scores in [1, 5] for each stock.
    llm_risks : np.ndarray (stock_dim,)
        LLM risk scores in [1, 5] for each stock.
    adjustment_type : {"both", "sentiment", "risk", "none"}
        Which adjustment to apply.
    alpha, beta : float
        Exponents for sentiment and risk, respectively.

    Returns
    -------
    float
        Multiplicative adjustment factor. Multiply the raw reward by this
        factor to get the LLM-adjusted reward.

    Example
    -------
    >>> positions = np.array([10000.0, 5000.0, 0.0])   # $15k total, 2 stocks
    >>> sentiments = np.array([4, 2, 3])              # buy, sell, neutral
    >>> risks = np.array([3, 3, 3])                    # all neutral risk
    >>> factor = compute_llm_reward_adjustment(positions, sentiments, risks)
    >>> adjusted_reward = raw_reward * factor
    """
    total_value = np.sum(position_values)

    if total_value == 0:
        return 1.0  # no positions → no adjustment

    # Portfolio weight of each stock
    stock_weights = position_values / total_value

    # Map scores to multiplicative weights
    sentiments_w = np.vectorize(SENTIMENT_WEIGHT_MAP.get)(llm_sentiments)
    risks_w = np.vectorize(RISK_WEIGHT_MAP.get)(llm_risks)

    # Portfolio-weighted aggregates
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


def extract_llm_features_from_state(
    state: np.ndarray,
    stock_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract LLM sentiment and risk sub-arrays from a flattened RL state vector.

    State layout (from FinRL-DAPO-SR)::
        [cash, prices..., shares..., tech_indicators..., llm_sentiments..., llm_risks...]

    Parameters
    ----------
    state : np.ndarray (state_space,)
        Single-row state as returned by the environment.
    stock_dim : int
        Number of stocks in the portfolio.

    Returns
    -------
    (sentiments, risks) : (np.ndarray, np.ndarray)
        Both arrays have shape (stock_dim,) with values in [1, 5].

    Example
    -------
    >>> sentiments, risks = extract_llm_features_from_state(state, stock_dim=5)
    >>> print(sentiments)  # array([3, 4, 2, 3, 5])
    """
    # sentiment block is the second-to-last block of size stock_dim
    sentiment_start = -2 * stock_dim
    risk_start = -stock_dim

    sentiments = state[sentiment_start:risk_start]
    risks = state[risk_start:]

    return sentiments.astype(np.float32), risks.astype(np.float32)
