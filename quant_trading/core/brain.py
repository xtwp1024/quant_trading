# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import logger
from .knowledge_engine import TitanKnowledgeEngine
from .llm import TitanLLM
import json

class TitanBrain:
    """
    TitanBrain: Titan V13 的认知层 (The Cognitive Layer for Titan V13).
    利用内部知识库生成语义交易信号 (Uses internal knowledge engine to generate semantic trading signals).
    """
    def __init__(self, knowledge_engine: TitanKnowledgeEngine, config: Dict[str, Any]) -> None:
        self.knowledge = knowledge_engine
        self.config = config
        self.llm = TitanLLM(config)
        self.last_signal: Optional[Dict[str, Any]] = None
        logger.info("🧠 [TitanBrain] 认知皮层已初始化 (Cognitive Cortex Initialized).")

    async def analyze_market_context(self, current_vibe: str) -> Dict[str, Any]:
        """
        从 Hive Mind 中回忆事实并生成“Alpha 偏见” (Recall facts from the Hive Mind and generate an 'Alpha Bias').
        """
        logger.info(f"🧠 [TitanBrain] 正在分析上下文: '{current_vibe}'...")
        
        # 1. 回忆相关的技术/理论 DNA (Recall related technical/theoretical DNA)
        memories = self.knowledge.recall(current_vibe, top_k=5)
        
        if not memories:
            logger.warning("🧠 [TitanBrain] 未找到此氛围的相关记忆 (No relevant memories found for this vibe).")
            return {"bias": "neutral", "confidence": 0.0}

        if self.llm.enabled:
            # LLM Analysis
            try:
                memory_text = "\n".join([f"- {m.get('content', '')[:200]}..." for m in memories])
                prompt = f"""
                Analyze the following market context and memories to generate a trading signal.
                Current Market Vibe: {current_vibe}
                
                Relevant Memories/Knowledge:
                {memory_text}
                
                Respond in STRICT JSON format: {{"bias": "long"|"short"|"neutral", "confidence": <float 0.0-1.0>, "reasoning": "<short explanation>"}}
                """
                
                response = self.llm.query(prompt, system_prompt="You are a veteran hedge fund analyst. Output strictly valid JSON.")

                # Clean response locally (handling potential markdown fences)
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].strip()

                # HIGH: 显式捕获JSON解析错误，提供更精确的错误处理
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ [TitanBrain] LLM返回了无效的JSON: {e}, 原始响应: {response[:100]}...")
                    return {"bias": "neutral", "confidence": 0.0, "error": "invalid_json"}
                
                signal = {
                    "bias": data.get("bias", "neutral"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "supporting_docs": [m.get('filename', 'unknown') for m in memories],
                    "reasoning": data.get("reasoning", "LLM Analysis")
                }
                logger.info(f"🧠 [TitanBrain] LLM 信号: {signal['bias']} (Docs: {len(memories)})")
                
                self.last_signal = signal
                return signal
            except Exception as e:
                logger.error(f"❌ [TitanBrain] LLM Analysis Failed: {e}. Falling back to legacy logic.")
        
        # Legacy / Mock Logic (Fallback)
        # 2. 提取特定的 Alpha 基因 (逻辑片段) (Extract specific alpha genes)
        # 在真正的进化中，这将输入给 LLM/IntelBee (In a real evolution, this would be fed to an LLM/IntelBee)
        # 对于 'Titan V13' 集成，我们计算共识得分 (For 'Titan V13', we calculate a consensus score)
        # FIX: 使用 .get('score', 0.5) 避免 KeyError
        avg_score = sum([m.get('score', 0.5) for m in memories]) / len(memories)
        
        # 模拟逻辑: 基于顶部记忆情感的偏见 (Mock logic: Bias based on top memory sentiment)
        signal = {
            "bias": "long" if "bull" in current_vibe.lower() else "short" if "bear" in current_vibe.lower() else "neutral",
            "confidence": avg_score,
            "supporting_docs": [m.get('filename', 'unknown') for m in memories], 
            "reasoning": "Legacy Vibe Analysis"
        }
        
        self.last_signal = signal
        logger.info(f"🧠 [TitanBrain] 信号已生成: {signal['bias']} (置信度: {signal['confidence']:.2f})")
        return signal

    def get_latest_insight(self) -> Optional[Dict[str, Any]]:
        return self.last_signal
