# -*- coding: utf-8 -*-
"""
Titan LLM Interface
Provides a unified way to interact with Large Language Models (OpenAI compatible).
"""

import logging
import os
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger("Titan.LLM")

class TitanLLM:
    """
    Lightweight client for OpenAI-compatible LLM APIs.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('llm', {})
        self.enabled = self.config.get('enabled', False)
        # HIGH: API密钥必须从环境变量获取，禁止从配置文件读取
        self.api_key = os.getenv('LLM_API_KEY')
        self.base_url = os.getenv('LLM_BASE_URL', self.config.get('base_url', 'https://api.openai.com/v1'))
        self.model = self.config.get('model', 'gpt-3.5-turbo')

        if self.enabled and not self.api_key:
            logger.warning("⚠️ LLM enabled but no API key provided. Falling back to mock mode.")
            self.enabled = False

    def query(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Send a query to the LLM.
        """
        if not self.enabled:
            return "MOCK_RESPONSE: LLM disabled or not configured."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        try:
            logger.info(f"🤖 sending query to {self.model}...")
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            return content.strip()
            
        except Exception as e:
            logger.error(f"❌ LLM Request Failed: {e}")
            return f"ERROR: {e}"
