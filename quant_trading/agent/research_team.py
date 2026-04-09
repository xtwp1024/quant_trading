"""Multi-agent research team: 6 analysts -> bull/bear researchers -> research manager.

Workflow:
  1. Six analysts run in parallel, each producing a dimension-specific report:
     - FundamentalsAnalyst    (基本面: 财报、资产负债表、现金流、利润表)
     - MacroAnalyst           (宏观: 行业周期、宏观政策、利率)
     - MarketAnalyst          (市场技术面: K线、均线、RSI、MACD、布林带)
     - NewsAnalyst            (新闻: 财经新闻、公告、事件)
     - SmartMoneyAnalyst      (主力资金: 大单、机构持仓、资金流向)
     - SocialMediaAnalyst     (社交媒体: 论坛、舆情、情绪)
  2. BullResearcher and BearResearcher debate using structured claims.
  3. ResearchManager裁决，决定是否形成投资计划。

The memory system uses BM25 to retrieve similar historical situations.
Streaming token output is supported via an optional tracker callback.
"""

from __future__ import annotations

import asyncio
import contextvars
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

# ContextVar for passing progress tracker into async analyst nodes
current_tracker_var: contextvars.ContextVar = contextvars.ContextVar(
    "current_tracker", default=None
)


# ---------------------------------------------------------------------------
# Memory (BM25-based, no API calls required)
# ---------------------------------------------------------------------------

class FinancialSituationMemory:
    """BM25-based memory for retrieving similar historical situations and advice."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.documents: list[str] = []
        self.recommendations: list[str] = []
        self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_index(self):
        if self.documents:
            from rank_bm25 import BM25Okapi
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def add_situations(self, situations_and_advice: list[tuple[str, str]]) -> None:
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 2) -> list[dict[str, Any]]:
        if not self.documents or self.bm25 is None:
            return []
        query_tokens = self._tokenize(current_situation)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]
        max_score = max(scores) if max(scores) > 0 else 1
        results = []
        for idx in top_indices:
            results.append({
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": scores[idx] / max_score,
            })
        return results

    def clear(self) -> None:
        self.documents = []
        self.recommendations = []
        self.bm25 = None


# ---------------------------------------------------------------------------
# Horizon context builder (mirrors TradingAgents intent_parser logic)
# ---------------------------------------------------------------------------

def build_horizon_context(horizon: str, focus_areas: list[str], specific_questions: list[str], agent_type: str) -> str:
    """Build horizon-specific context string for analyst prompts."""
    horizon_map = {
        "short": {"days": 14, "label": "短期", "trading": "短线交易", "view": "近期走势"},
        "medium": {"days": 90, "label": "中期", "trading": "趋势跟踪", "view": "中期趋势"},
        "long": {"days": 180, "label": "长期", "trading": "价值投资", "view": "长期价值"},
    }
    info = horizon_map.get(horizon, horizon_map["medium"])
    parts = [
        f"[分析视角] {info['view']} | 交易类型: {info['trading']} | 视野: {info['label']}",
    ]
    if focus_areas:
        parts.append(f"[重点领域] {'; '.join(focus_areas)}")
    if specific_questions:
        parts.append(f"[核心问题] {'; '.join(specific_questions)}")
    parts.append(f"[角色定位] {agent_type} analyst ({agent_type.title()})")
    return "\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Agent progress tracker protocol
# ---------------------------------------------------------------------------

class AgentProgressTracker:
    """Optional callback sink for streaming token and debate events."""

    def _emit_token(self, agent: str, field: str, token: str) -> None:
        pass

    def emit_debate_token(self, debate: str, agent: str, round_num: int, token: str) -> None:
        pass

    def emit_debate_message(self, debate: str, agent: str, round_num: int, content: str, is_verdict: bool = False) -> None:
        pass


# ---------------------------------------------------------------------------
# Base analyst node (shared streaming logic)
# ---------------------------------------------------------------------------

async def _stream_llm_response(
    llm: Any,
    prompt: Any,
    tracker: Optional[AgentProgressTracker],
    agent_name: str,
    output_field: str,
) -> str:
    """Helper: stream LLM response, optionally emit tokens to tracker."""
    full_content = ""
    async for chunk in llm.astream(prompt):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        full_content += content
        if tracker:
            tracker._emit_token(agent_name, output_field, content)
    return full_content


# ---------------------------------------------------------------------------
# Analyst creators
# ---------------------------------------------------------------------------

def create_fundamentals_analyst(
    llm: Any,
    memory: Optional[FinancialSituationMemory] = None,
    data_collector: Optional[Any] = None,
):
    """Fundamentals analyst: evaluates financial health from 4-dim financial data."""
    async def node(state: dict) -> dict:
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "fundamentals")

        system_msg = (
            horizon_ctx
            + "你是一位专业的A股基本面分析师。你的职责是根据财务报表评估公司价值。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。\n\n"
            + "VERDICT格式: <!-- VERDICT: {\"direction\": \"看多/中性/看空\", \"confidence\": \"高/中/低\"} -->"
        )

        pool = data_collector.get(ticker, current_date) if data_collector else None
        if pool is not None:
            outputs = {k: pool.get(k, "无数据") for k in
                       ["fundamentals", "balance_sheet", "cashflow", "income_statement"]}
        else:
            # Fallback: direct tool calls
            from quant_trading.agent.intel_collector import (
                get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
            )
            async def _safe(tool, payload):
                try:
                    return await asyncio.to_thread(tool.invoke, payload)
                except Exception as exc:
                    return f"调用失败：{exc}"

            tasks = {
                "fundamentals": _safe(get_fundamentals, {"ticker": ticker, "curr_date": current_date}),
                "balance_sheet": _safe(get_balance_sheet, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
                "cashflow": _safe(get_cashflow, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
                "income_statement": _safe(get_income_statement, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
            }
            keys = list(tasks.keys())
            results = await asyncio.gather(*[tasks[k] for k in keys])
            outputs = dict(zip(keys, results))

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"以下是 {ticker} 在 {current_date} 的基本面资料。\n\n"
                f"【get_fundamentals】\n{outputs.get('fundamentals', '无数据')}\n\n"
                f"【get_balance_sheet】\n{outputs.get('balance_sheet', '无数据')}\n\n"
                f"【get_cashflow】\n{outputs.get('cashflow', '无数据')}\n\n"
                f"【get_income_statement】\n{outputs.get('income_statement', '无数据')}"
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "Fundamentals Analyst", "fundamentals_report")

        return {
            "fundamentals_report": full_content,
            "analyst_traces": [{
                "agent": "fundamentals_analyst",
                "horizon": horizon,
                "data_window": "财报周期",
                "key_finding": "基本面分析完成",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


def create_macro_analyst(llm: Any):
    """Macro analyst: evaluates sector/macro economic conditions."""
    async def node(state: dict) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "macro")

        system_msg = (
            horizon_ctx
            + "你是一位专业的宏观经济与行业周期分析师。你的职责是评估宏观政策和行业趋势。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。"
        )

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"分析 {ticker} 在 {current_date} 的宏观与行业背景。\n"
                "请提供：\n1. 所属行业周期位置\n2. 宏观政策影响\n3. 利率与流动性环境\n4. 行业竞争格局\n5. 风险因素"
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "Macro Analyst", "macro_report")

        return {
            "macro_report": full_content,
            "analyst_traces": [{
                "agent": "macro_analyst",
                "horizon": horizon,
                "data_window": "宏观周期",
                "key_finding": "宏观分析完成",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


def create_market_analyst(
    llm: Any,
    data_collector: Optional[Any] = None,
):
    """Market analyst: evaluates technical indicators and price action."""
    MARKET_INDICATORS = [
        "close_50_sma", "close_200_sma", "close_10_ema",
        "rsi", "macd", "boll", "boll_ub", "boll_lb", "atr", "vwma",
    ]

    async def node(state: dict) -> dict:
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        horizon = state.get("horizon", "short")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "market")

        system_msg = (
            horizon_ctx
            + "你是一位专业的技术面分析师。你的职责是根据K线与指标判断价格趋势。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。"
        )

        days = 14 if horizon == "short" else 90
        end_dt = datetime.strptime(current_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days)

        if data_collector is not None:
            pool = data_collector.get(ticker, current_date)
            if pool is not None:
                windowed = data_collector.get_window(pool, horizon, current_date)
                stock_data = windowed.get("stock_data", "无数据")
                indicators = windowed.get("indicators", {})
                data_window = windowed.get("_data_window", f"{days}天")
            else:
                stock_data, indicators, data_window = await _fetch_direct(ticker, current_date, days, horizon)
        else:
            stock_data, indicators, data_window = await _fetch_direct(ticker, current_date, days, horizon)

        indicator_blocks = [
            f"【{ind}】\n{indicators.get(ind, '无数据')}"
            for ind in MARKET_INDICATORS
        ]

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"以下是 {ticker} 在 {current_date} 的 K 线数据与指标（数据窗口：{data_window}）。\n\n"
                f"【get_stock_data】\n{stock_data}\n\n"
                + "\n\n".join(indicator_blocks)
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "Market Analyst", "market_report")

        return {
            "market_report": full_content,
            "analyst_traces": [{
                "agent": "market_analyst",
                "horizon": horizon,
                "data_window": data_window,
                "key_finding": f"技术面结论：{_extract_verdict(full_content)[0]}",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


async def _fetch_direct(ticker: str, current_date: str, days: int, horizon: str):
    """Direct data fetch when no data collector is available."""
    from quant_trading.agent.intel_collector import (
        get_stock_data, get_indicators,
    )
    async def _safe(tool, payload):
        try:
            return await asyncio.to_thread(tool.invoke, payload)
        except Exception as exc:
            return f"调用失败：{exc}"

    end_dt = datetime.strptime(current_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days)

    MARKET_INDICATORS = [
        "close_50_sma", "close_200_sma", "close_10_ema",
        "rsi", "macd", "boll", "boll_ub", "boll_lb", "atr", "vwma",
    ]

    tasks = {
        "stock_data": _safe(get_stock_data, {
            "symbol": ticker, "start_date": start_dt.strftime("%Y-%m-%d"), "end_date": current_date,
        })
    }
    for ind in MARKET_INDICATORS:
        tasks[ind] = _safe(get_indicators, {
            "symbol": ticker, "indicator": ind, "curr_date": current_date, "look_back_days": days,
        })

    keys = list(tasks.keys())
    results = await asyncio.gather(*[tasks[k] for k in keys])
    res_map = dict(zip(keys, results))
    stock_data = res_map.pop("stock_data")
    return stock_data, res_map, f"{days}天"


def create_news_analyst(llm: Any):
    """News analyst: evaluates news, announcements and events."""
    async def node(state: dict) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "news")

        system_msg = (
            horizon_ctx
            + "你是一位专业的财经新闻分析师。你的职责是评估新闻与事件对股价的影响。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。"
        )

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"分析 {ticker} 在 {current_date} 附近的重大新闻、公告与事件。\n"
                "请提供：\n1. 重大新闻列表\n2. 公告要点\n3. 事件驱动因素\n4. 市场预期影响\n5. 短期情绪判断"
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "News Analyst", "news_report")

        return {
            "news_report": full_content,
            "analyst_traces": [{
                "agent": "news_analyst",
                "horizon": horizon,
                "data_window": "近期新闻",
                "key_finding": "新闻分析完成",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


def create_smart_money_analyst(llm: Any):
    """Smart money analyst: evaluates institutional activity and capital flow."""
    async def node(state: dict) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "smart_money")

        system_msg = (
            horizon_ctx
            + "你是一位专业的主力资金分析师。你的职责是追踪机构动向与大单资金流。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。"
        )

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"分析 {ticker} 在 {current_date} 的主力资金与机构持仓情况。\n"
                "请提供：\n1. 大单净流入/出\n2. 机构持仓变化\n3. 资金流向（5日/10日/20日）\n4. 主力成本区间\n5. 机构行为判断"
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "Smart Money Analyst", "smart_money_report")

        return {
            "smart_money_report": full_content,
            "analyst_traces": [{
                "agent": "smart_money_analyst",
                "horizon": horizon,
                "data_window": "资金流周期",
                "key_finding": "主力资金分析完成",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


def create_social_media_analyst(llm: Any):
    """Social media analyst: evaluates retail sentiment from forums and social platforms."""
    async def node(state: dict) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "social_media")

        system_msg = (
            horizon_ctx
            + "你是一位专业的社交媒体情绪分析师。你的职责是评估散户情绪与舆情走向。\n"
            + "请全程使用中文，并在分析结尾标注VERDICT标签。"
        )

        from langchain_core.messages import HumanMessage
        messages = [
            {"role": "system", "content": system_msg},
            HumanMessage(content=(
                f"分析 {ticker} 在 {current_date} 的社交媒体舆情与散户情绪。\n"
                "请提供：\n1. 主要论坛/平台情绪概况\n2. 热门讨论主题\n3. KOL观点汇总\n4. 情绪分歧点\n5. 舆情风险提示"
            )),
        ]

        tracker = current_tracker_var.get()
        full_content = await _stream_llm_response(llm, messages, tracker, "Social Media Analyst", "sentiment_report")

        return {
            "sentiment_report": full_content,
            "analyst_traces": [{
                "agent": "social_media_analyst",
                "horizon": horizon,
                "data_window": "舆情周期",
                "key_finding": "情绪分析完成",
                "verdict": _extract_verdict(full_content)[0],
                "confidence": _extract_verdict(full_content)[1],
            }],
        }

    return node


# ---------------------------------------------------------------------------
# Bull / Bear researchers
# ---------------------------------------------------------------------------

def create_bull_researcher(llm: Any, memory: Optional[FinancialSituationMemory] = None):
    """Bull researcher: builds bullish investment thesis with structured claims."""

    async def node(state: dict) -> dict:
        from quant_trading.agent.debate_engine import (
            format_claim_subset_for_prompt,
            format_claims_for_prompt,
            update_debate_state_with_payload,
        )

        invest_state = state["investment_debate_state"]
        history = invest_state.get("history", "")
        current_response = invest_state.get("current_response", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        claims = invest_state.get("claims", [])
        focus_claim_ids = invest_state.get("focus_claim_ids", [])
        unresolved_claim_ids = invest_state.get("unresolved_claim_ids", [])
        round_summary = invest_state.get("round_summary", "")
        round_goal = invest_state.get("round_goal", "")

        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "bull")

        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = (memory.get_memories(curr_situation, n_matches=2) if memory else [])
        past_memory_str = "\n\n".join(rec["recommendation"] for rec in past_memories) or "无历史记忆"

        prompt = horizon_ctx + (
            f"你是一位专业的多方分析师。你的职责是基于证据建立最强有力的看多论点。\n\n"
            f"【研究报告】\n市场技术面:\n{market_report}\n\n"
            f"情绪面:\n{sentiment_report}\n\n"
            f"新闻面:\n{news_report}\n\n"
            f"基本面:\n{fundamentals_report}\n\n"
            f"【历史记忆】\n{past_memory_str}\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【当前对方观点】\n{current_response}\n\n"
            f"【已登记Claims】\n{format_claims_for_prompt(claims)}\n\n"
            f"【待攻击Claims】\n{format_claim_subset_for_prompt(claims, focus_claim_ids)}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(claims, unresolved_claim_ids)}\n\n"
            f"【轮次摘要】\n{round_summary}\n\n"
            f"【本轮目标】\n{round_goal}\n\n"
            "请提出1-2条新的多方claim（有证据支撑），并在DEBATE_STATE标签中返回结构化状态。"
        )

        tracker = current_tracker_var.get()
        try:
            debate_round = int(invest_state.get("count", 0) or 0) // 2 + 1
        except (ValueError, TypeError):
            debate_round = 1

        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker._emit_token("Bull Researcher", "investment_debate_state", content)
                tracker.emit_debate_token(debate="research", agent="Bull Researcher", round_num=debate_round, token=content)

        if tracker:
            tracker.emit_debate_message(debate="research", agent="Bull Researcher", round_num=debate_round, content=full_content)

        new_invest_state = update_debate_state_with_payload(
            state=invest_state,
            raw_response=full_content,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        return {"investment_debate_state": new_invest_state}

    return node


def create_bear_researcher(llm: Any, memory: Optional[FinancialSituationMemory] = None):
    """Bear researcher: challenges bullish thesis with structured claims."""

    async def node(state: dict) -> dict:
        from quant_trading.agent.debate_engine import (
            format_claim_subset_for_prompt,
            format_claims_for_prompt,
            update_debate_state_with_payload,
        )

        invest_state = state["investment_debate_state"]
        history = invest_state.get("history", "")
        current_response = invest_state.get("current_response", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        claims = invest_state.get("claims", [])
        focus_claim_ids = invest_state.get("focus_claim_ids", [])
        unresolved_claim_ids = invest_state.get("unresolved_claim_ids", [])
        round_summary = invest_state.get("round_summary", "")
        round_goal = invest_state.get("round_goal", "")

        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, "bear")

        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = (memory.get_memories(curr_situation, n_matches=2) if memory else [])
        past_memory_str = "\n\n".join(rec["recommendation"] for rec in past_memories) or "无历史记忆"

        prompt = horizon_ctx + (
            f"你是一位专业的空方分析师。你的职责是挑战多方论点，找出最脆弱的假设。\n\n"
            f"【研究报告】\n市场技术面:\n{market_report}\n\n"
            f"情绪面:\n{sentiment_report}\n\n"
            f"新闻面:\n{news_report}\n\n"
            f"基本面:\n{fundamentals_report}\n\n"
            f"【历史记忆】\n{past_memory_str}\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【当前对方观点】\n{current_response}\n\n"
            f"【已登记Claims】\n{format_claims_for_prompt(claims)}\n\n"
            f"【待攻击Claims】\n{format_claim_subset_for_prompt(claims, focus_claim_ids)}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(claims, unresolved_claim_ids)}\n\n"
            f"【轮次摘要】\n{round_summary}\n\n"
            f"【本轮目标】\n{round_goal}\n\n"
            "请提出1-2条新的空方claim（有证据支撑），并在DEBATE_STATE标签中返回结构化状态。"
        )

        tracker = current_tracker_var.get()
        try:
            debate_round = int(invest_state.get("count", 0) or 0) // 2 + 1
        except (ValueError, TypeError):
            debate_round = 1

        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker._emit_token("Bear Researcher", "investment_debate_state", content)
                tracker.emit_debate_token(debate="research", agent="Bear Researcher", round_num=debate_round, token=content)

        if tracker:
            tracker.emit_debate_message(debate="research", agent="Bear Researcher", round_num=debate_round, content=full_content)

        new_invest_state = update_debate_state_with_payload(
            state=invest_state,
            raw_response=full_content,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        return {"investment_debate_state": new_invest_state}

    return node


# ---------------------------------------------------------------------------
# Research manager (judge)
# ---------------------------------------------------------------------------

def create_research_manager(llm: Any, memory: Optional[FinancialSituationMemory] = None):
    """Research manager: synthesizes bull/bear debate and produces investment plan."""

    async def node(state: dict) -> dict:
        from quant_trading.agent.debate_engine import (
            format_claim_subset_for_prompt,
            format_claims_for_prompt,
            summarize_game_theory_signals,
        )

        history = state["investment_debate_state"].get("history", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        game_theory_report = state.get("game_theory_report", "")
        game_theory_signals = state.get("game_theory_signals", {})

        invest_state = state["investment_debate_state"]
        claims = invest_state.get("claims", [])
        unresolved_claim_ids = invest_state.get("unresolved_claim_ids", [])
        round_summary = invest_state.get("round_summary", "")

        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = (memory.get_memories(curr_situation, n_matches=2) if memory else [])
        past_memory_str = "\n\n".join(rec["recommendation"] for rec in past_memories) or "无历史记忆"

        prompt = (
            f"你是一位专业的投资研究总监。你的职责是综合多空双方观点，形成最终投资计划。\n\n"
            f"【研究报告摘要】\n市场技术面:\n{market_report[:500]}\n...\n"
            f"情绪面:\n{sentiment_report[:300]}\n...\n"
            f"新闻面:\n{news_report[:300]}\n...\n"
            f"基本面:\n{fundamentals_report[:300]}\n...\n\n"
            f"【历史记忆】\n{past_memory_str}\n\n"
            f"【完整辩论历史】\n{history}\n\n"
            f"【博弈分析】\n{ game_theory_report}\n"
            f"【博弈信号摘要】\n{summarize_game_theory_signals(game_theory_signals)}\n\n"
            f"【Claims汇总】\n{format_claims_for_prompt(claims)}\n\n"
            f"【未解决争议】\n{format_claim_subset_for_prompt(claims, unresolved_claim_ids)}\n\n"
            f"【轮次摘要】\n{round_summary}\n\n"
            "请综合以上分析，形成最终投资计划，包括：\n"
            "1. 核心投资逻辑（1-3条）\n"
            "2. 买入/卖出建议及价格区间\n"
            "3. 关键风险因素\n"
            "4. 持仓周期建议\n"
            "5. 止损建议"
        )

        tracker = current_tracker_var.get()
        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker._emit_token("Research Manager", "investment_plan", content)
                tracker.emit_debate_token(debate="research", agent="Research Manager", round_num=-1, token=content)

        if tracker:
            tracker.emit_debate_message(debate="research", agent="Research Manager", round_num=-1, content=full_content, is_verdict=True)

        new_invest_state = {
            "judge_decision": full_content,
            "history": invest_state.get("history", ""),
            "bull_history": invest_state.get("bull_history", ""),
            "bear_history": invest_state.get("bear_history", ""),
            "current_speaker": invest_state.get("current_speaker", ""),
            "current_response": full_content,
            "count": invest_state["count"],
            "claims": claims,
            "focus_claim_ids": invest_state.get("focus_claim_ids", []),
            "open_claim_ids": invest_state.get("open_claim_ids", []),
            "resolved_claim_ids": invest_state.get("resolved_claim_ids", []),
            "unresolved_claim_ids": unresolved_claim_ids,
            "round_summary": round_summary,
            "round_goal": invest_state.get("round_goal", ""),
            "claim_counter": invest_state.get("claim_counter", 0),
        }
        return {"investment_debate_state": new_invest_state, "investment_plan": full_content}

    return node


# ---------------------------------------------------------------------------
# Game theory manager
# ---------------------------------------------------------------------------

def create_game_theory_manager(llm: Any):
    """Game theory manager: analyzes strategic interactions from smart money + sentiment data."""

    def node(state: dict) -> dict:
        from quant_trading.agent.debate_engine import extract_tagged_json, strip_tagged_json

        smart_money_report = state.get("smart_money_report", "无主力资金数据")
        sentiment_report = state.get("sentiment_report", "无情绪数据")

        prompt = (
            "你是一位专业的博弈论分析师。你的职责是基于主力资金和情绪数据，"
            "判断市场参与者的战略互动格局。\n\n"
            f"【主力资金报告】\n{smart_money_report}\n\n"
            f"【情绪报告】\n{sentiment_report}\n\n"
            "请分析博弈格局，并在GAME_THEORY标签中返回结构化信号：\n"
            "<!-- GAME_THEORY: {\"board\": \"...\", \"players\": [...], "
            "\"dominant_strategy\": \"...\", \"fragile_equilibrium\": \"...\", "
            "\"likely_actions\": {...}, \"counter_consensus_signal\": \"...\", "
            "\"confidence\": \"...\"} -->"
        )

        result = llm.invoke(prompt)
        structured_signals = extract_tagged_json(result.content, "GAME_THEORY")
        cleaned_report = strip_tagged_json(result.content, "GAME_THEORY")

        return {
            "game_theory_report": cleaned_report,
            "game_theory_signals": structured_signals,
        }

    return node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_verdict(text: str) -> tuple[str, str]:
    """Extract direction and confidence from a VERDICT tag."""
    import re, json
    m = re.search(r"<!--\s*VERDICT:\s*(\{.*?\})\s*-->", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("direction", "中性"), d.get("confidence", "中")
        except Exception:
            pass
    return "中性", "中"


# ---------------------------------------------------------------------------
# Debator factories (stub implementations for TradingAgents compatibility)
# ---------------------------------------------------------------------------

def create_aggressive_debator(llm: Any) -> Any:
    """Create an aggressive debator agent (stub)."""
    return None


def create_conservative_debator(llm: Any) -> Any:
    """Create a conservative debator agent (stub)."""
    return None


def create_neutral_debator(llm: Any) -> Any:
    """Create a neutral debator agent (stub)."""
    return None


def create_trader(llm: Any) -> Any:
    """Create a trader agent (stub)."""
    return None
