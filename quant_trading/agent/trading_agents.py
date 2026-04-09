"""TradingAgentsCoordinator — top-level orchestrator for the dual-debate quant agent system.

Architecture (15 agents, complete investment research loop):
  Stage 1 — Parallel Analysis (6 analysts)
    FundamentalsAnalyst · MacroAnalyst · MarketAnalyst
    NewsAnalyst · SmartMoneyAnalyst · SocialMediaAnalyst

  Stage 2 — Bull vs Bear Research Debate (2 researchers)
    BullResearcher  (多方)  vs  BearResearcher (空方)
    Structured claim-driven confrontation with evidence tracking

  Stage 3 — Investment Decision (ResearchManager judge)
    Synthesizes bull/bear debate → investment_plan

  Stage 4 — Risk Debate (3 risk debaters + RiskManager)
    AggressiveDebater · ConservativeDebter · NeutralDebater
    → RiskManager judge → final_trade_decision

  Stage 5 — Trader
    Produces trader_investment_plan with position sizing, entry/exit

Dual-state design:
  - investment_debate_state: Bull vs Bear structured claims
  - risk_debate_state: 3-way risk debate (Aggressive / Conservative / Neutral)

Supports multi-model LLM backends (OpenAI / Claude / Gemini / DeepSeek)
via a configurable LLM client interface. Streaming token output via tracker.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional

from quant_trading.agent.debate_engine import (
    build_empty_invest_debate_state,
    build_empty_risk_debate_state,
    build_empty_risk_feedback_state,
    extract_risk_judge_result,
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
    safe_int,
    strip_tagged_json,
    summarize_game_theory_signals,
    summarize_risk_feedback,
    update_debate_state_with_payload,
)
from quant_trading.agent.research_team import (
    AgentProgressTracker,
    FinancialSituationMemory,
    build_horizon_context,
    current_tracker_var,
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_game_theory_manager,
    create_macro_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_research_manager,
    create_smart_money_analyst,
    create_social_media_analyst,
    create_trader,
)


# ---------------------------------------------------------------------------
# LLM client factory (multi-provider)
# ---------------------------------------------------------------------------

def create_llm_client(provider: str = "openai", **kwargs) -> Any:
    """
    Create an async LLM client by provider name.

    Supported providers:
      - openai       : OpenAI GPT models (uses langchain-openai)
      - anthropic    : Anthropic Claude models (uses langchain-anthropic)
      - google       : Google Gemini models (uses langchain-google-genai)
      - deepseek     : DeepSeek models (uses langchain-deepseek)

    Returns an async chat model that implements .astream() and .invoke().
    """
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=kwargs.get("model", os.getenv("TA_LLM_QUICK", "gpt-4o-mini")),
            base_url=kwargs.get("base_url") or os.getenv("TA_BASE_URL", "https://api.openai.com/v1"),
            api_key=kwargs.get("api_key") or os.getenv("TA_API_KEY", ""),
            temperature=kwargs.get("temperature", 0.3),
            streaming=True,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=kwargs.get("model", "claude-3-5-haiku-20241107"),
            anthropic_api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=kwargs.get("temperature", 0.3),
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model", "gemini-2.0-flash"),
            google_api_key=kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY", ""),
            temperature=kwargs.get("temperature", 0.3),
        )
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=kwargs.get("model", "deepseek-chat"),
            deepseek_api_key=kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=kwargs.get("base_url") or "https://api.deepseek.com/v1",
            temperature=kwargs.get("temperature", 0.3),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Default config (mirrors TradingAgents-AShare default_config.py)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "llm_provider": os.getenv("TA_LLM_PROVIDER", "openai"),
    "deep_think_llm": os.getenv("TA_LLM_DEEP", "gpt-4o"),
    "quick_think_llm": os.getenv("TA_LLM_QUICK", "gpt-4o-mini"),
    "backend_url": os.getenv("TA_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.getenv("TA_API_KEY", ""),
    "max_debate_rounds": int(os.getenv("TA_MAX_DEBATE", "2")),
    "max_risk_discuss_rounds": int(os.getenv("TA_MAX_RISK", "1")),
    "max_recur_limit": 100,
    "prompt_language": os.getenv("TA_LANGUAGE", "zh"),
    "provider_trace": os.getenv("TA_TRACE", "1").lower() in ("1", "true", "yes", "on"),
}


# ---------------------------------------------------------------------------
# Trader (investment plan -> specific trading instructions)
# ---------------------------------------------------------------------------

def create_trader_node(llm, memory: Optional[FinancialSituationMemory] = None):
    """Trader node: converts investment plan into specific trading instructions."""
    async def node(state: dict) -> dict:
        from langchain_core.messages import AIMessage

        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        previous_plan = state.get("trader_investment_plan", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        risk_feedback = state.get("risk_feedback_state", {})

        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = (memory.get_memories(curr_situation, n_matches=2) if memory else [])
        past_memory_str = "\n\n".join(rec["recommendation"] for rec in past_memories) or "无历史记忆"

        risk_feedback_summary = summarize_risk_feedback(risk_feedback)

        system_prompt = (
            "你是一位专业的A股交易员。你的职责是将投资计划转化为具体的交易指令。\n"
            "请全程使用中文。\n\n"
            f"【市场环境】\n{past_memory_str}\n\n"
            f"【风控反馈】\n{risk_feedback_summary}"
        )

        user_prompt = (
            f"【公司】{company_name}\n\n"
            f"【投资计划】\n{investment_plan}\n\n"
            f"【上一版交易计划（如有）】\n{previous_plan or '无'}\n\n"
            "请将上述投资计划转化为具体的交易指令，包括：\n"
            "1. 操作方向（买入/卖出/观望）\n"
            "2. 入场价格区间\n"
            "3. 仓位建议（占总资金比例）\n"
            "4. 止损价格\n"
            "5. 止盈目标\n"
            "6. 持有周期\n"
            "7. 风险收益比"
        )

        tracker = current_tracker_var.get()
        full_content = ""
        async for chunk in llm.astream([{"role": "system", "content": system_prompt},
                                          {"role": "user", "content": user_prompt}]):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker._emit_token("Trader", "trader_investment_plan", content)

        result = AIMessage(content=full_content)
        updated_feedback = dict(risk_feedback)
        if updated_feedback.get("revision_required"):
            updated_feedback["revision_required"] = False

        response_state = {
            "messages": [result],
            "trader_investment_plan": full_content,
            "sender": "Trader",
        }
        if risk_feedback.get("latest_risk_verdict") == "revise":
            response_state["risk_debate_state"] = build_empty_risk_debate_state()
            response_state["risk_feedback_state"] = updated_feedback

        return response_state

    return node


# ---------------------------------------------------------------------------
# Risk debaters
# ---------------------------------------------------------------------------

def create_aggressive_debater_node(llm):
    """Aggressive risk debater: identifies aggressive execution risks."""
    async def node(state: dict) -> dict:
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        aggressive_history = risk_state.get("aggressive_history", "")
        current_conservative = risk_state.get("current_conservative_response", "")
        current_neutral = risk_state.get("current_neutral_response", "")
        claims = risk_state.get("claims", [])
        focus_ids = risk_state.get("focus_claim_ids", [])
        unresolved_ids = risk_state.get("unresolved_claim_ids", [])
        round_summary = risk_state.get("round_summary", "")
        round_goal = risk_state.get("round_goal", "")
        trader_plan = state.get("trader_investment_plan", "")

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = (
            "你是一位激进派风控分析师。你的职责是识别交易计划中最激进、最极端的风险。\n"
            "请全程使用中文，并使用RISK_STATE标签返回结构化状态。\n\n"
            f"【交易计划】\n{trader_plan}\n\n"
            f"【市场技术面】\n{market_report[:300]}...\n\n"
            f"【情绪面】\n{sentiment_report[:200]}...\n\n"
            f"【新闻面】\n{news_report[:200]}...\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【保守派观点】\n{current_conservative}\n\n"
            f"【中性派观点】\n{current_neutral}\n\n"
            f"【已登记风险Claims】\n{format_claims_for_prompt(claims, empty_message='当前没有已登记风险claim，本轮请先提出最关键的执行风险。')}\n\n"
            f"【待攻击Claims】\n{format_claim_subset_for_prompt(claims, focus_ids)}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(claims, unresolved_ids)}\n\n"
            f"【轮次摘要】\n{round_summary or '暂无风险轮次摘要，请先建立核心风险claim。'}\n\n"
            f"【本轮目标】\n{round_goal}"
        )

        tracker = current_tracker_var.get()
        try:
            debate_round = int(risk_state.get("count", 0) or 0) // 3 + 1
        except (ValueError, TypeError):
            debate_round = 1

        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker.emit_debate_token(debate="risk", agent="Aggressive Analyst", round_num=debate_round, token=content)

        clean_response = strip_tagged_json(full_content, "RISK_STATE")
        new_risk_state = update_debate_state_with_payload(
            state=risk_state,
            raw_response=full_content,
            speaker_label="Aggressive Analyst",
            speaker_key="Aggressive",
            stance="aggressive",
            history_key="aggressive_history",
            marker="RISK_STATE",
            claim_prefix="RISK",
            domain="risk",
            speaker_field="latest_speaker",
            store_current_response=False,
        )
        if tracker:
            tracker.emit_debate_message(debate="risk", agent="Aggressive Analyst", round_num=debate_round, content=clean_response)

        new_risk_state["current_aggressive_response"] = f"Aggressive Analyst: {clean_response}"
        new_risk_state["current_conservative_response"] = risk_state.get("current_conservative_response", "")
        new_risk_state["current_neutral_response"] = risk_state.get("current_neutral_response", "")

        return {"risk_debate_state": new_risk_state}

    return node


def create_conservative_debater_node(llm):
    """Conservative risk debater: identifies defensive/protective risks."""
    async def node(state: dict) -> dict:
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        conservative_history = risk_state.get("conservative_history", "")
        current_aggressive = risk_state.get("current_aggressive_response", "")
        current_neutral = risk_state.get("current_neutral_response", "")
        claims = risk_state.get("claims", [])
        focus_ids = risk_state.get("focus_claim_ids", [])
        unresolved_ids = risk_state.get("unresolved_claim_ids", [])
        round_summary = risk_state.get("round_summary", "")
        round_goal = risk_state.get("round_goal", "")
        trader_plan = state.get("trader_investment_plan", "")

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = (
            "你是一位保守派风控分析师。你的职责是识别交易计划中最关键的防守型风险。\n"
            "请全程使用中文，并使用RISK_STATE标签返回结构化状态。\n\n"
            f"【交易计划】\n{trader_plan}\n\n"
            f"【市场技术面】\n{market_report[:300]}...\n\n"
            f"【情绪面】\n{sentiment_report[:200]}...\n\n"
            f"【新闻面】\n{news_report[:200]}...\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【激进派观点】\n{current_aggressive}\n\n"
            f"【中性派观点】\n{current_neutral}\n\n"
            f"【已登记风险Claims】\n{format_claims_for_prompt(claims, empty_message='当前没有已登记风险claim，本轮请先提出最关键的防守风险。')}\n\n"
            f"【待攻击Claims】\n{format_claim_subset_for_prompt(claims, focus_ids)}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(claims, unresolved_ids)}\n\n"
            f"【轮次摘要】\n{round_summary or '暂无风险轮次摘要，请先攻击最脆弱的进攻型风险假设。'}\n\n"
            f"【本轮目标】\n{round_goal}"
        )

        tracker = current_tracker_var.get()
        try:
            debate_round = int(risk_state.get("count", 0) or 0) // 3 + 1
        except (ValueError, TypeError):
            debate_round = 1

        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker.emit_debate_token(debate="risk", agent="Conservative Analyst", round_num=debate_round, token=content)

        clean_response = strip_tagged_json(full_content, "RISK_STATE")
        new_risk_state = update_debate_state_with_payload(
            state=risk_state,
            raw_response=full_content,
            speaker_label="Conservative Analyst",
            speaker_key="Conservative",
            stance="conservative",
            history_key="conservative_history",
            marker="RISK_STATE",
            claim_prefix="RISK",
            domain="risk",
            speaker_field="latest_speaker",
            store_current_response=False,
        )
        if tracker:
            tracker.emit_debate_message(debate="risk", agent="Conservative Analyst", round_num=debate_round, content=clean_response)

        new_risk_state["current_aggressive_response"] = risk_state.get("current_aggressive_response", "")
        new_risk_state["current_conservative_response"] = f"Conservative Analyst: {clean_response}"
        new_risk_state["current_neutral_response"] = risk_state.get("current_neutral_response", "")

        return {"risk_debate_state": new_risk_state}

    return node


def create_neutral_debater_node(llm):
    """Neutral risk debater: mediates between aggressive and conservative positions."""
    async def node(state: dict) -> dict:
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        neutral_history = risk_state.get("neutral_history", "")
        current_aggressive = risk_state.get("current_aggressive_response", "")
        current_conservative = risk_state.get("current_conservative_response", "")
        claims = risk_state.get("claims", [])
        focus_ids = risk_state.get("focus_claim_ids", [])
        unresolved_ids = risk_state.get("unresolved_claim_ids", [])
        round_summary = risk_state.get("round_summary", "")
        round_goal = risk_state.get("round_goal", "")
        trader_plan = state.get("trader_investment_plan", "")

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = (
            "你是一位中立派风控分析师。你的职责是识别最关键的风险矛盾并提出平衡方案。\n"
            "请全程使用中文，并使用RISK_STATE标签返回结构化状态。\n\n"
            f"【交易计划】\n{trader_plan}\n\n"
            f"【市场技术面】\n{market_report[:300]}...\n\n"
            f"【情绪面】\n{sentiment_report[:200]}...\n\n"
            f"【新闻面】\n{news_report[:200]}...\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【激进派观点】\n{current_aggressive}\n\n"
            f"【保守派观点】\n{current_conservative}\n\n"
            f"【已登记风险Claims】\n{format_claims_for_prompt(claims, empty_message='当前没有已登记风险claim，本轮请先识别最关键的执行矛盾。')}\n\n"
            f"【待攻击Claims】\n{format_claim_subset_for_prompt(claims, focus_ids)}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(claims, unresolved_ids)}\n\n"
            f"【轮次摘要】\n{round_summary or '暂无风险轮次摘要，请先识别真正有信息增量的风险分歧。'}\n\n"
            f"【本轮目标】\n{round_goal}"
        )

        tracker = current_tracker_var.get()
        try:
            debate_round = int(risk_state.get("count", 0) or 0) // 3 + 1
        except (ValueError, TypeError):
            debate_round = 1

        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker.emit_debate_token(debate="risk", agent="Neutral Analyst", round_num=debate_round, token=content)

        clean_response = strip_tagged_json(full_content, "RISK_STATE")
        new_risk_state = update_debate_state_with_payload(
            state=risk_state,
            raw_response=full_content,
            speaker_label="Neutral Analyst",
            speaker_key="Neutral",
            stance="neutral",
            history_key="neutral_history",
            marker="RISK_STATE",
            claim_prefix="RISK",
            domain="risk",
            speaker_field="latest_speaker",
            store_current_response=False,
        )
        if tracker:
            tracker.emit_debate_message(debate="risk", agent="Neutral Analyst", round_num=debate_round, content=clean_response)

        new_risk_state["current_aggressive_response"] = risk_state.get("current_aggressive_response", "")
        new_risk_state["current_conservative_response"] = risk_state.get("current_conservative_response", "")
        new_risk_state["current_neutral_response"] = f"Neutral Analyst: {clean_response}"

        return {"risk_debate_state": new_risk_state}

    return node


# ---------------------------------------------------------------------------
# Risk manager (judge)
# ---------------------------------------------------------------------------

def create_risk_manager_node(llm, memory: Optional[FinancialSituationMemory] = None):
    """Risk manager: judges the 3-way risk debate and issues pass/revise/reject verdict."""
    async def node(state: dict) -> dict:
        company_name = state["company_of_interest"]
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        trader_plan = state.get("trader_investment_plan", "")
        risk_feedback = state.get("risk_feedback_state", {})

        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = (memory.get_memories(curr_situation, n_matches=2) if memory else [])
        past_memory_str = "\n\n".join(rec["recommendation"] for rec in past_memories) or "无历史记忆"

        prompt = (
            "你是一位专业的风控总监。你的职责是对交易计划进行最终风控审核。\n"
            "请全程使用中文，并在RISK_JUDGE标签中返回裁决结果。\n\n"
            f"【公司】{company_name}\n\n"
            f"【交易计划】\n{trader_plan}\n\n"
            f"【历史记忆】\n{past_memory_str}\n\n"
            f"【辩论历史】\n{history}\n\n"
            f"【已登记Claims】\n{format_claims_for_prompt(risk_state.get('claims', []), empty_message='当前没有已登记风控claim。')}\n\n"
            f"【未解决Claims】\n{format_claim_subset_for_prompt(risk_state.get('claims', []), risk_state.get('unresolved_claim_ids', []))}\n\n"
            f"【轮次摘要】\n{risk_state.get('round_summary', '暂无风险轮次摘要。')}\n\n"
            "裁决标准：\n"
            "  - pass    : 计划可执行，无重大风险\n"
            "  - revise  : 需要按风控意见修改后重提\n"
            "  - reject  : 风险过高，拒绝执行\n\n"
            "RISK_JUDGE格式: <!-- RISK_JUDGE: {\"verdict\": \"pass|revise|reject\", "
            "\"hard_constraints\": [...], \"soft_constraints\": [...], "
            "\"execution_preconditions\": [...], \"de_risk_triggers\": [...], "
            "\"revision_reason\": \"...\"} -->"
        )

        tracker = current_tracker_var.get()
        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker.emit_debate_token(debate="risk", agent="Risk Manager", round_num=-1, token=content)

        judge_result = extract_risk_judge_result(full_content)
        cleaned_response = judge_result["cleaned_response"]
        verdict = judge_result["verdict"]
        hard_constraints = judge_result["hard_constraints"]
        soft_constraints = judge_result["soft_constraints"]
        execution_preconditions = judge_result["execution_preconditions"]
        de_risk_triggers = judge_result["de_risk_triggers"]
        revision_reason = judge_result["revision_reason"]

        if tracker:
            tracker.emit_debate_message(debate="risk", agent="Risk Manager", round_num=-1, content=cleaned_response, is_verdict=True)

        new_risk_state = {
            "judge_decision": cleaned_response,
            "history": risk_state["history"],
            "aggressive_history": risk_state["aggressive_history"],
            "conservative_history": risk_state["conservative_history"],
            "neutral_history": risk_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_state.get("current_neutral_response", ""),
            "count": risk_state["count"],
            "claims": risk_state.get("claims", []),
            "focus_claim_ids": risk_state.get("focus_claim_ids", []),
            "open_claim_ids": risk_state.get("open_claim_ids", []),
            "resolved_claim_ids": risk_state.get("resolved_claim_ids", []),
            "unresolved_claim_ids": risk_state.get("unresolved_claim_ids", []),
            "round_summary": risk_state.get("round_summary", ""),
            "round_goal": risk_state.get("round_goal", ""),
            "claim_counter": risk_state.get("claim_counter", 0),
        }

        new_risk_feedback = {
            "retry_count": safe_int(risk_feedback.get("retry_count", 0), 0) + (1 if verdict == "revise" else 0),
            "max_retries": safe_int(risk_feedback.get("max_retries", 1), 1),
            "revision_required": verdict == "revise",
            "latest_risk_verdict": verdict,
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints,
            "execution_preconditions": execution_preconditions,
            "de_risk_triggers": de_risk_triggers,
            "revision_reason": revision_reason or ("风控要求交易员按硬约束重写方案" if verdict == "revise" else ""),
        }

        return {
            "risk_debate_state": new_risk_state,
            "risk_feedback_state": new_risk_feedback,
            "final_trade_decision": cleaned_response,
        }

    return node


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def safe_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# TradingAgentsCoordinator
# ---------------------------------------------------------------------------

class TradingAgentsCoordinator:
    """
    Top-level coordinator for the complete 15-agent quant research loop.

    Supports two execution modes:
      - run_async(): fully async pipeline, all stages as async generators
      - run_sync():  synchronous wrapper for sync contexts

    Usage:
        coordinator = TradingAgentsCoordinator(
            llm_provider="openai",
            config={"max_debate_rounds": 2, "max_risk_discuss_rounds": 1},
        )
        result = await coordinator.run_async(
            company_of_interest="000001.SZ",
            trade_date="2026-03-28",
            horizon="medium",
        )
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        config: Optional[dict] = None,
        memory: Optional[FinancialSituationMemory] = None,
        data_collector: Optional[Any] = None,
        tracker: Optional[AgentProgressTracker] = None,
    ):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.memory = memory or FinancialSituationMemory()
        self.data_collector = data_collector
        self.tracker = tracker

        # Create main LLM client (quick thinker for parallel analysts)
        self.llm = create_llm_client(
            provider=llm_provider,
            model=self.config.get("quick_think_llm"),
            base_url=self.config.get("backend_url"),
            api_key=self.config.get("api_key"),
        )

        # Create node instances
        self._analyst_nodes = self._build_analyst_nodes()
        self._researcher_nodes = self._build_researcher_nodes()
        self._manager_nodes = self._build_manager_nodes()
        self._risk_nodes = self._build_risk_nodes()
        self._trader_node = create_trader_node(self.llm, self.memory)

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _build_analyst_nodes(self) -> dict[str, Callable]:
        return {
            "fundamentals": create_fundamentals_analyst(self.llm, self.memory, self.data_collector),
            "macro": create_macro_analyst(self.llm),
            "market": create_market_analyst(self.llm, self.data_collector),
            "news": create_news_analyst(self.llm),
            "smart_money": create_smart_money_analyst(self.llm),
            "social_media": create_social_media_analyst(self.llm),
        }

    def _build_researcher_nodes(self) -> dict[str, Callable]:
        return {
            "bull": create_bull_researcher(self.llm, self.memory),
            "bear": create_bear_researcher(self.llm, self.memory),
        }

    def _build_manager_nodes(self) -> dict[str, Callable]:
        return {
            "research": create_research_manager(self.llm, self.memory),
            "game_theory": create_game_theory_manager(self.llm),
        }

    def _build_risk_nodes(self) -> dict[str, Callable]:
        return {
            "aggressive": create_aggressive_debater_node(self.llm),
            "conservative": create_conservative_debater_node(self.llm),
            "neutral": create_neutral_debater_node(self.llm),
            "risk_manager": create_risk_manager_node(self.llm, self.memory),
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _run_stage1_analysts(self, state: dict) -> dict:
        """Stage 1: Run 6 analysts in parallel."""
        if self.tracker:
            current_tracker_var.set(self.tracker)

        tasks = {name: node(state) for name, node in self._analyst_nodes.items()}
        results = await asyncio.gather(*tasks.values())
        updates = dict(zip(tasks.keys(), results))

        # Merge analyst traces
        all_traces = []
        for updates_dict in updates.values():
            if "analyst_traces" in updates_dict:
                all_traces.extend(updates_dict["analyst_traces"])

        state.update(updates)
        if all_traces:
            state["analyst_traces"] = all_traces
        return state

    async def _run_stage2_research_debate(self, state: dict) -> dict:
        """Stage 2: Run bull/bear structured debate for max_debate_rounds."""
        max_rounds = self.config.get("max_debate_rounds", 2)

        # Ensure investment debate state exists
        if "investment_debate_state" not in state:
            state["investment_debate_state"] = build_empty_invest_debate_state()

        for round_num in range(max_rounds):
            # Parallel bull and bear research
            bull_task = self._researcher_nodes["bull"](state)
            bear_task = self._researcher_nodes["bear"](state)
            bull_result, bear_result = await asyncio.gather(bull_task, bear_task)

            state.update(bull_result)
            state.update(bear_result)

        # Research manager synthesizes
        research_result = await self._manager_nodes["research"](state)
        state.update(research_result)

        # Game theory analysis
        gt_result = self._manager_nodes["game_theory"](state)
        if isinstance(gt_result, dict):
            state.update(gt_result)

        return state

    async def _run_stage3_trader(self, state: dict) -> dict:
        """Stage 3: Trader converts investment plan to trading instructions."""
        if self.tracker:
            current_tracker_var.set(self.tracker)

        trader_result = await self._trader_node(state)
        state.update(trader_result)
        return state

    async def _run_stage4_risk_debate(self, state: dict) -> dict:
        """Stage 4: Run 3-way risk debate + risk manager judgment."""
        if self.tracker:
            current_tracker_var.set(self.tracker)

        max_rounds = self.config.get("max_risk_discuss_rounds", 1)

        # Ensure risk debate state exists
        if "risk_debate_state" not in state:
            state["risk_debate_state"] = build_empty_risk_debate_state()
        if "risk_feedback_state" not in state:
            state["risk_feedback_state"] = build_empty_risk_feedback_state()

        for round_num in range(max_rounds):
            # Parallel 3-way risk debate
            agg_task = self._risk_nodes["aggressive"](state)
            cons_task = self._risk_nodes["conservative"](state)
            neut_task = self._risk_nodes["neutral"](state)
            agg_result, cons_result, neut_result = await asyncio.gather(agg_task, cons_task, neut_task)

            state.update(agg_result)
            state.update(cons_result)
            state.update(neut_result)

        # Risk manager issues verdict
        risk_result = await self._risk_nodes["risk_manager"](state)
        state.update(risk_result)

        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        company_of_interest: str,
        trade_date: str,
        horizon: str = "medium",
        user_context: Optional[dict] = None,
        selected_analysts: Optional[list[str]] = None,
        **extra_state,
    ) -> dict:
        """
        Run the complete research loop asynchronously.

        Args:
            company_of_interest: Stock ticker symbol
            trade_date: Analysis date (YYYY-MM-DD)
            horizon: "short", "medium", or "long"
            user_context: Optional user constraints (cash_available, risk_profile, etc.)
            selected_analysts: Optional list of analyst keys to run (default: all 6)
            **extra_state: Additional state fields

        Returns:
            Final state dict containing all reports, debate states, and trade decision
        """
        if self.tracker:
            current_tracker_var.set(self.tracker)

        # Build initial state
        state: dict[str, Any] = {
            "company_of_interest": company_of_interest,
            "trade_date": trade_date,
            "horizon": horizon,
            "user_intent": {
                "raw_query": f"分析 {company_of_interest} 在 {trade_date} 的投资价值",
                "ticker": company_of_interest,
                "horizons": [horizon],
                "focus_areas": [],
                "specific_questions": [],
                "user_context": user_context or {},
            },
            "investment_debate_state": build_empty_invest_debate_state(),
            "risk_debate_state": build_empty_risk_debate_state(),
            "risk_feedback_state": build_empty_risk_feedback_state(),
            "analyst_traces": [],
            "messages": [],
            **extra_state,
        }

        # Stage 1: Parallel analysts
        state = await self._run_stage1_analysts(state)

        # Stage 2: Bull vs Bear research debate
        state = await self._run_stage2_research_debate(state)

        # Stage 3: Trader
        state = await self._run_stage3_trader(state)

        # Stage 4: Risk debate (loop if revise)
        revision_count = 0
        max_total_revisions = state["risk_feedback_state"].get("max_retries", 1) * 2
        while True:
            state = await self._run_stage4_risk_debate(state)
            risk_fb = state.get("risk_feedback_state", {})
            if not risk_fb.get("revision_required"):
                break
            revision_count += 1
            if revision_count >= max_total_revisions:
                break
            # Trader revises plan
            state = await self._run_stage3_trader(state)

        return state

    def run_sync(
        self,
        company_of_interest: str,
        trade_date: str,
        horizon: str = "medium",
        user_context: Optional[dict] = None,
        **extra_state,
    ) -> dict:
        """Synchronous wrapper around run_async."""
        import asyncio
        return asyncio.run(self.run_async(
            company_of_interest=company_of_interest,
            trade_date=trade_date,
            horizon=horizon,
            user_context=user_context,
            **extra_state,
        ))
