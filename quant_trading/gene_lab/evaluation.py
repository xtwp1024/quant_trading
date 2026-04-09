"""Evaluation — absorbed from GeneTrader.

Backtest result parsing and anti-overfitting fitness calculation.

Fitness function is designed to PREVENT OVERFITTING by:
1. Requiring minimum trade counts for statistical significance
2. Disqualifying excessive drawdown (>35%)
3. Requiring minimum profit factor (>=1.0) and win rate (>=30%)
4. Penalizing too many parameters (complexity penalty)
5. Balancing profit with risk-adjusted metrics (Sharpe/Sortino/Profit Factor)
"""
import os
import re
import math
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional

logger = logging.getLogger("GeneLab.Evaluation")

_PATTERNS = {
    "absolute_profit": re.compile(
        r"Absolute profit\s*│\s*([-\d.]+)\s*USDT", re.IGNORECASE
    ),
    "total_profit_percent": re.compile(
        r"Total profit %\s*│\s*([\d.-]+)%", re.IGNORECASE
    ),
    "max_drawdown": re.compile(
        r"Max % of account underwater\s*│\s*([\d.]+)%", re.IGNORECASE
    ),
    "sharpe_ratio": re.compile(r"Sharpe\s*│\s*([\d.]+)", re.IGNORECASE),
    "sortino_ratio": re.compile(r"Sortino\s*│\s*([\d.]+)", re.IGNORECASE),
    "profit_factor": re.compile(r"Profit factor\s*│\s*([\d.]+)", re.IGNORECASE),
    "avg_profit": re.compile(r"│\s*TOTAL\s*│.*?│\s*([\d.-]+)\s*│", re.DOTALL | re.IGNORECASE),
    "total_trades": re.compile(
        r"Total/Daily Avg Trades\s*│\s*(\d+)\s*/", re.IGNORECASE
    ),
    "daily_avg_trades": re.compile(
        r"Total/Daily Avg Trades\s*│\s*\d+\s*/\s*([\d.]+)", re.IGNORECASE
    ),
    "avg_duration_winners": re.compile(
        r"Avg\. Duration Winners\s*│\s*(.*?)\s*│", re.DOTALL | re.IGNORECASE
    ),
}


def extract_win_rate(content: str) -> float:
    """Extract win rate from Freqtrade backtest output."""
    for line in content.split("\n"):
        if "TOTAL" in line:
            parts = [p.strip() for p in line.split("│")]
            try:
                return float(parts[-2].split()[3]) / 100
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def _extract_value_from_pattern(
    pattern: re.Pattern,
    content: str,
    default: Union[float, str] = 0,
    is_string: bool = False,
) -> Union[float, str]:
    """Extract a value using a pre-compiled regex pattern."""
    match = pattern.search(content)
    if match:
        value = match.group(1).strip()
        if is_string:
            return value
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _parse_duration(duration_str: str) -> int:
    """Parse duration string to total minutes.

    Args:
        duration_str: Duration like "1 day, 2:30:00" or "2:30:00"

    Returns:
        Total duration in minutes
    """
    if not duration_str or duration_str == "0:00:00":
        return 0

    parts = duration_str.split(", ")
    total_minutes = 0

    try:
        for part in parts:
            if "day" in part:
                total_minutes += int(part.split()[0]) * 24 * 60
            else:
                time_parts = part.split(":")
                if len(time_parts) >= 2:
                    total_minutes += int(time_parts[0]) * 60 + int(time_parts[1])
    except (ValueError, IndexError):
        return 0

    return total_minutes


def _empty_results() -> Dict[str, Any]:
    return {
        "total_profit_usdt": 0,
        "total_profit_percent": 0,
        "win_rate": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "profit_factor": 0,
        "avg_profit": 0,
        "total_trades": 0,
        "daily_avg_trades": 0,
        "avg_trade_duration": 0,
    }


def parse_backtest_results(file_path: str) -> Dict[str, Any]:
    """Parse backtest results from a Freqtrade output file.

    Args:
        file_path: Path to the backtest results file

    Returns:
        Dictionary containing parsed metrics
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Backtest results file not found: {file_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading backtest results file {file_path}: {e}")
        raise

    if "SUMMARY METRICS" not in content:
        logger.warning(
            f"{file_path} does not contain summary metrics. No trades were executed."
        )
        return _empty_results()

    duration_str = _extract_value_from_pattern(
        _PATTERNS["avg_duration_winners"],
        content,
        default="0:00:00",
        is_string=True,
    )

    parsed_result = {
        "total_profit_usdt": _extract_value_from_pattern(
            _PATTERNS["absolute_profit"], content
        ),
        "total_profit_percent": _extract_value_from_pattern(
            _PATTERNS["total_profit_percent"], content
        )
        / 100,
        "win_rate": extract_win_rate(content),
        "max_drawdown": _extract_value_from_pattern(_PATTERNS["max_drawdown"], content)
        / 100,
        "sharpe_ratio": _extract_value_from_pattern(_PATTERNS["sharpe_ratio"], content),
        "sortino_ratio": _extract_value_from_pattern(_PATTERNS["sortino_ratio"], content),
        "profit_factor": _extract_value_from_pattern(_PATTERNS["profit_factor"], content),
        "avg_profit": _extract_value_from_pattern(_PATTERNS["avg_profit"], content),
        "total_trades": _extract_value_from_pattern(_PATTERNS["total_trades"], content),
        "daily_avg_trades": _extract_value_from_pattern(
            _PATTERNS["daily_avg_trades"], content
        ),
        "avg_trade_duration": _parse_duration(duration_str),
    }

    return parsed_result


def fitness_function(
    parsed_result: Dict[str, Any],
    generation: int,
    strategy_name: str,
    timeframe: str,
    num_parameters: int = 0,
    backtest_weeks: int = 30,
) -> float:
    """Calculate anti-overfitting fitness score for a trading strategy.

    Disqualifications (negative return):
        -1.0  Insufficient trades (< backtest_weeks/2, min 15)
        -2.0  Excessive drawdown (>35%)
        -3.0  Unprofitable (profit_factor < 1.0)
        -4.0  Win rate < 30%

    Components (balanced weights):
        profit_score        25%  — smooth tanh of total profit %
        win_rate_score      10%  — Gaussian peaked at 55% win rate
        risk_adjusted_score 25%  — Sharpe(40%) + Sortino(40%) + PF(20%)
        drawdown_penalty    15%  — exponential decay
        trade_frequency     10%  — optimal ~2 trades/day
        duration_score       5%  — optimal ~12h trades
        trade_confidence    10%  — statistical significance bonus
        complexity_penalty (applied as multiplier)

    Returns:
        Fitness score (higher is better, negative = disqualified)
    """
    total_profit_percent = parsed_result["total_profit_percent"]
    win_rate = parsed_result["win_rate"]
    max_drawdown = parsed_result["max_drawdown"]
    sharpe_ratio = parsed_result["sharpe_ratio"]
    sortino_ratio = parsed_result["sortino_ratio"]
    profit_factor = parsed_result["profit_factor"]
    daily_avg_trades = parsed_result["daily_avg_trades"]
    avg_trade_duration = parsed_result["avg_trade_duration"]
    total_trades = parsed_result["total_trades"]

    # --- DISQUALIFICATION CHECKS ---
    min_trades = max(backtest_weeks // 2, 15)
    if total_trades < min_trades:
        logger.warning(
            f"Strategy {strategy_name}: Insufficient trades ({total_trades} < {min_trades})"
        )
        return -1.0

    if max_drawdown > 0.35:
        logger.warning(
            f"Strategy {strategy_name}: Excessive drawdown ({max_drawdown:.1%} > 35%)"
        )
        return -2.0

    if profit_factor < 1.0:
        logger.warning(
            f"Strategy {strategy_name}: Unprofitable (PF={profit_factor:.2f} < 1.0)"
        )
        return -3.0

    if win_rate < 0.30:
        logger.warning(
            f"Strategy {strategy_name}: Win rate too low ({win_rate:.1%} < 30%)"
        )
        return -4.0

    # --- COMPONENT SCORES ---
    profit_score = math.tanh(total_profit_percent / 2.0)

    win_rate_target = 0.55
    win_rate_score = math.exp(-((win_rate - win_rate_target) ** 2) / 0.08)

    sharpe_component = math.tanh(sharpe_ratio / 2) if sharpe_ratio > 0 else -0.5
    sortino_component = math.tanh(sortino_ratio / 2) if sortino_ratio > 0 else -0.5
    pf_component = (
        math.tanh((profit_factor - 1) / 2) if profit_factor > 1 else -0.5
    )
    risk_adjusted_score = sharpe_component * 0.4 + sortino_component * 0.4 + pf_component * 0.2

    drawdown_penalty = math.exp(-3 * max_drawdown)

    optimal_trades = 2.0
    trade_frequency_score = math.exp(
        -((daily_avg_trades - optimal_trades) ** 2) / 8
    )

    optimal_duration = 720  # 12 hours in minutes
    duration_score = math.exp(
        -((avg_trade_duration - optimal_duration) ** 2) / (2 * optimal_duration**2)
    )

    if num_parameters > 0:
        complexity_penalty = math.exp(-0.1 * max(0, num_parameters - 5))
    else:
        complexity_penalty = 1.0

    trade_confidence = min(
        1.0, 0.5 + 0.5 * (total_trades - min_trades) / max(1, 100 - min_trades)
    )

    # --- COMBINED FITNESS ---
    fitness = (
        profit_score * 0.25
        + win_rate_score * 0.10
        + risk_adjusted_score * 0.25
        + drawdown_penalty * 0.15
        + trade_frequency_score * 0.10
        + duration_score * 0.05
        + trade_confidence * 0.10
    ) * complexity_penalty

    logger.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Strategy: {strategy_name}, Gen: {generation}, "
        f"Profit: {total_profit_percent:.4f}, WinRate: {win_rate:.2%}, "
        f"Sharpe: {sharpe_ratio:.2f}, PF: {profit_factor:.2f}, "
        f"MaxDD: {max_drawdown:.2%}, Trades: {int(total_trades)}, "
        f"Fitness: {fitness:.4f}"
    )

    return fitness


# --- LambdaRank IC/NDCG Evaluation (absorbed from Liumon) ---
# Learning-to-Rank evaluation: rank-aware metrics for cross-sectional signal quality.
# LambdaRank framework: https://github.com/20070316lbw-netizen/Liumon
# Key insight: IC measures rank correlation; NDCG@k measures top-K ranking quality.

import numpy as np
from scipy.stats import spearmanr
from typing import List, Optional


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Information Coefficient: Spearman rank correlation between predictions and realized returns.

    IC is the primary reliability metric for alpha signals — high IC means the model's
    ranking of assets aligns with their actual ranking by forward returns.

    Args:
        predictions: Model predicted scores/rankings per asset
        actuals: Realized forward returns per asset

    Returns:
        IC value (range: -1 to 1). Positive IC indicates predictive skill.
    """
    if len(predictions) < 10:
        return 0.0
    ic, _ = spearmanr(predictions, actuals)
    return float(ic) if not np.isnan(ic) else 0.0


def compute_ndcg_at_k(
    predictions: np.ndarray,
    actuals: np.ndarray,
    k: int = 10,
    relevance_gain: bool = True,
) -> float:
    """Normalized Discounted Cumulative Gain at K (NDCG@k).

    Measures the quality of top-K ranking: how well the model places
    high-reward assets at the top of the prediction list.

    Args:
        predictions: Model predicted scores
        actuals: Realized returns (used as relevance labels)
        k: Cutoff position for top-K evaluation
        relevance_gain: If True, use exp(actuals) - 1 as gains; else use actuals directly

    Returns:
        NDCG@k value (0 to 1). Higher is better ranking quality.
    """
    if len(predictions) < k:
        k = len(predictions)

    # Build relevance vector from actuals
    if relevance_gain:
        gains = np.exp(np.clip(actuals, -1, 1)) - 1  # clip for stability
    else:
        gains = actuals

    # Sort by predictions (descending) to get ideal ranking
    sorted_indices = np.argsort(-predictions)
    sorted_gains = gains[sorted_indices]

    # DCG@k = sum(g_i / log2(i + 2)) for i in [0, k-1]
    positions = np.arange(k) + 2  # +2 because log2(1) = 0, we start at position 2
    dcg = np.sum(sorted_gains[:k] / np.log2(positions))

    # Ideal DCG: sort by actual gains descending
    ideal_sorted = np.sort(gains)[::-1]
    idcg = np.sum(ideal_sorted[:k] / np.log2(np.arange(k) + 2))

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_ranking_metrics(
    df,
    pred_col: str,
    return_col: str,
    group_col: str = "date",
    k: int = 10,
    min_group_size: int = 20,
) -> dict:
    """Evaluate ranking quality across time periods (cross-sectional IC + NDCG).

    Adapted from Liumon's calculate_metrics() approach:
    - IC per period: rank correlation between prediction and forward return
    - NDCG@k per period: top-K ranking quality
    - Returns composite with t-statistic for IC significance

    Args:
        df: DataFrame with predictions, returns, and group identifiers
        pred_col: Column name for predicted scores
        return_col: Column name for realized forward returns
        group_col: Column for cross-sectional grouping (default: "date")
        k: NDCG cutoff (default: 10)
        min_group_size: Minimum samples per group for IC computation

    Returns:
        dict with mean_ic, std_ic, t_stat_ic, mean_ndcg, ic_series, ndcg_series
    """
    from sklearn.metrics import ndcg_score as _ndcg_score

    ic_list: List[float] = []
    ndcg_list: List[float] = []

    for period, grp in df.groupby(group_col):
        if len(grp) < min_group_size:
            continue

        preds = grp[pred_col].values.astype(float)
        rets = grp[return_col].values.astype(float)

        # Filter NaNs
        valid_mask = ~(np.isnan(preds) | np.isnan(rets))
        if valid_mask.sum() < min_group_size:
            continue
        preds = preds[valid_mask]
        rets = rets[valid_mask]

        ic = compute_ic(preds, rets)
        ic_list.append(ic)

        # NDCG@k (scikit-learn interface: shape [n_samples, n_scores])
        try:
            y_true = rets.reshape(1, -1)
            y_score = preds.reshape(1, -1)
            ndcg = _ndcg_score(y_true, y_score, k=k)
        except Exception:
            ndcg = compute_ndcg_at_k(preds, rets, k=k)
        ndcg_list.append(ndcg)

    if not ic_list:
        return {
            "mean_ic": 0.0, "std_ic": 0.0, "t_stat_ic": 0.0,
            "mean_ndcg": 0.0, "ic_series": [], "ndcg_series": [],
        }

    ic_arr = np.array(ic_list)
    ndcg_arr = np.array(ndcg_list)

    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr))
    n = len(ic_arr)
    t_stat_ic = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0

    logger.info(
        f"[RankingEvaluation] IC={mean_ic:.4f} (t={t_stat_ic:.2f}) | "
        f"NDCG@{k}={np.mean(ndcg_arr):.4f} | periods={n}"
    )

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat_ic": t_stat_ic,
        "mean_ndcg": float(np.mean(ndcg_arr)),
        "ic_series": ic_list,
        "ndcg_series": ndcg_list,
    }


def calculate_ic_decay(
    df,
    return_cols: List[str],
    pred_col: str = "pred",
    group_col: str = "date",
    min_group_size: int = 20,
) -> dict:
    """IC decay analysis across multiple forward return horizons.

    Computes IC for each return horizon to measure signal decay rate.
    Short-horizon IC >> long-horizon IC indicates rapid alpha decay.

    Args:
        df: DataFrame with predictions and return columns
        return_cols: List of forward return columns (e.g., ["label_5d", "label_20d"])
        pred_col: Prediction column name
        group_col: Grouping column

    Returns:
        dict mapping return_col -> IC metrics dict
    """
    results = {}
    for col in return_cols:
        if col not in df.columns:
            continue
        metrics = evaluate_ranking_metrics(
            df, pred_col, col, group_col, min_group_size=min_group_size
        )
        results[col] = metrics
        logger.info(
            f"[IC Decay] {col}: IC={metrics['mean_ic']:.4f} "
            f"(t={metrics['t_stat_ic']:.2f})"
        )
    return results


def ranking_fitness_composite(
    ic_metrics: dict,
    ndcg_weight: float = 0.3,
    min_ic_t_stat: float = 1.5,
) -> float:
    """Composite fitness from LambdaRank-style IC + NDCG evaluation.

    Combines IC significance and NDCG top-K quality into a single score.
    Disqualifies if IC t-statistic is below significance threshold.

    Args:
        ic_metrics: Output of evaluate_ranking_metrics()
        ndcg_weight: Weight for NDCG in composite (IC weight = 1 - ndcg_weight)
        min_ic_t_stat: Minimum IC t-stat for non-disqualification (default: 1.5)

    Returns:
        Composite fitness score. Negative = disqualified.
    """
    mean_ic = ic_metrics["mean_ic"]
    t_stat = ic_metrics["t_stat_ic"]
    mean_ndcg = ic_metrics["mean_ndcg"]

    if t_stat < min_ic_t_stat:
        logger.warning(
            f"[RankingFitness] IC t-stat {t_stat:.2f} < {min_ic_t_stat} threshold — disqualified"
        )
        return -1.0

    # Normalize IC to [0, 1] via tanh (positive IC means skill)
    ic_score = (math.tanh(mean_ic * 10) + 1) / 2  # scale IC to 0-1

    # NDCG is already roughly 0-1
    composite = ic_score * (1 - ndcg_weight) + mean_ndcg * ndcg_weight

    logger.info(
        f"[RankingFitness] IC_score={ic_score:.4f} | NDCG={mean_ndcg:.4f} "
        f"| Composite={composite:.4f}"
    )
    return composite
