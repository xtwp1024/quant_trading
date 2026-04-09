"""
AI Sentiment Analyzer - AI情绪分析模块
支持多种 AI API：GLM-4、OpenAI 兼容 API（如本地 Gemini）
"""

import json
import time
import logging
import os
import httpx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("SentimentAnalyzer")

# 默认API配置
DEFAULT_API_CONFIG: Dict[str, Any] = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30000  # 30秒
}

# 常量定义
MAX_TEXT_LENGTH = 10000
MIN_SCORE = -1.0
MAX_SCORE = 1.0


class SentimentAnalyzer:
    """AI情绪分析器"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        data_dir: str = "./data",
        model: str = "glm-4"
    ) -> None:
        """
        初始化情绪分析器

        Args:
            api_key: API密钥
            api_url: API地址
            data_dir: 数据目录
            model: 模型名称
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.data_dir = data_dir
        self.config = DEFAULT_API_CONFIG.copy()

        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)

    def _validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        验证输入文本

        Returns:
            (is_valid, error_message)
        """
        if not text or not isinstance(text, str):
            return False, "无效的输入文本"

        if not text.strip():
            return False, "输入文本为空"

        if len(text) > MAX_TEXT_LENGTH:
            return False, f"输入文本过长（最大{MAX_TEXT_LENGTH}字符）"

        return True, None

    async def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """
        调用 AI API 进行情绪分析

        Args:
            text: 待分析的文本内容

        Returns:
            分析结果，失败返回None
        """
        is_valid, error = self._validate_input(text)
        if not is_valid:
            logger.warning(f"输入验证失败: {error}")
            return None

        try:
            prompt = self._build_prompt(text)

            async with httpx.AsyncClient(timeout=self.config["timeout"] / 1000) as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": self.config["temperature"],
                        "max_tokens": self.config["max_tokens"]
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )

                response.raise_for_status()
                data = response.json()

                # 解析返回的JSON
                content = data["choices"][0]["message"]["content"]

                # 清理可能的 markdown 代码块标记
                content = content.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()

                result = json.loads(content)

                # 验证返回的分数是否有效
                if not self._is_valid_score(result.get("score")):
                    logger.warning(f"API返回的情绪分数无效: {result.get('score')}")
                    return None

                # 添加时间戳
                result["timestamp"] = int(time.time() * 1000)
                result["datetime"] = datetime.now().isoformat()

                return result

        except httpx.TimeoutException:
            logger.warning("API请求超时")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"API请求失败: {e.response.status_code}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"情绪分析失败: {e}")
            return None

    def _build_prompt(self, text: str) -> str:
        """构建分析Prompt"""
        return f"""角色：资深量化交易员及心理分析专家。

任务：对提供的文字内容进行金融情绪评分。

打分规则：
- 范围 -1 到 1
- -1 代表极度恐慌/看空（割肉、崩盘、离场）
- 0 代表中性（观望、平稳）
- 1 代表极度贪婪/看涨（梭哈、翻倍、牛回速归）

待分析文本：
{text}

输出要求：仅输出JSON格式，如：
{{"score": 0.8, "keywords": ["上涨", "支撑位"], "reason": "市场普遍看好减半行情"}}
输出示例（严格按照此格式）:
{{"score": 0.5, "keywords": ["突破", "看涨"], "reason": "关键阻力位被突破"}}
"""

    def _is_valid_score(self, score: Any) -> bool:
        """验证分数是否有效"""
        if not isinstance(score, (int, float)):
            return False
        return MIN_SCORE <= score <= MAX_SCORE

    async def analyze_text(self, source: str, text: str) -> Optional[Dict[str, Any]]:
        """
        分析单个文本源

        Args:
            source: 数据源名称（如"某B站UP主视频"）
            text: 文本内容

        Returns:
            分析结果
        """
        logger.info(f"正在分析: {source}")
        result = await self.analyze_sentiment(text)

        if result:
            result["source"] = source
            self._format_sentiment_output(result)
            self._save_to_csv(result)

        return result

    async def analyze_batch(self, texts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量分析多个文本源

        Args:
            texts: 文本数组，格式: [{{source: '名称', text: '内容'}}]

        Returns:
            分析结果数组
        """
        results = []
        for item in texts:
            result = await self.analyze_text(item.get("source", "Unknown"), item.get("text", ""))
            if result:
                results.append(result)
        return results

    def _get_sentiment_emoji(self, score: float) -> str:
        """获取情绪表情"""
        if score >= 0.6:
            return "🟢"
        elif score >= 0.2:
            return "🟡"
        elif score >= -0.2:
            return "⚪"
        elif score >= -0.6:
            return "🟠"
        else:
            return "🔴"

    def _get_sentiment_level(self, score: float) -> str:
        """获取情绪等级"""
        if score >= 0.6:
            return "极度贪婪"
        elif score >= 0.2:
            return "轻度贪婪"
        elif score >= -0.2:
            return "中性"
        elif score >= -0.6:
            return "轻度恐慌"
        else:
            return "极度恐慌"

    def _format_sentiment_output(self, result: Dict[str, Any]) -> None:
        """格式化输出情绪分析结果"""
        emoji = self._get_sentiment_emoji(result.get("score", 0))
        level = self._get_sentiment_level(result.get("score", 0))
        keywords = ", ".join(result.get("keywords", []))

        logger.info("=" * 50)
        logger.info(f"市场情绪分析报告: 来源={result.get('source')}")
        logger.info(f"情绪评分: {emoji} {result.get('score', 0):.2f} ({level})")
        logger.info(f"关键词: {keywords}")
        logger.info(f"分析理由: {result.get('reason', 'N/A')}")
        logger.info(f"分析时间: {result.get('datetime', 'N/A')}")
        logger.info("=" * 50)

    def _get_csv_path(self) -> str:
        """获取CSV文件路径"""
        import os
        return os.path.join(self.data_dir, "情绪分析记录.csv")

    def _get_csv_header(self) -> str:
        """获取CSV表头"""
        return "timestamp,datetime,source,score,keywords,reason\n"

    def _save_to_csv(self, result: Dict[str, Any]) -> None:
        """保存情绪分析结果到CSV文件"""
        csv_path = self._get_csv_path()
        keywords_str = "|".join(result.get("keywords", []))
        row = f"{result.get('timestamp')},{result.get('datetime')},\"{result.get('source')}\",{result.get('score')},\"{keywords_str}\",\"{result.get('reason', '')}\"\n"

        # 检查文件是否存在
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(self._get_csv_header())

        # 追加数据行
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(row)

        logger.info("情绪分析数据已保存到CSV")

    def get_sentiment_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取最近N小时的情感摘要

        Args:
            hours: 小时数

        Returns:
            摘要统计
        """
        import csv

        csv_path = self._get_csv_path()
        if not os.path.exists(csv_path):
            return {"count": 0, "avgScore": 0, "maxScore": 0, "minScore": 0}

        cutoff_time = int((time.time() - hours * 3600) * 1000)
        scores: List[float] = []
        sources: List[str] = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = int(row.get('timestamp', 0))
                    if timestamp >= cutoff_time:
                        try:
                            score = float(row.get('score', 0))
                            scores.append(score)
                            sources.append(row.get('source', ''))
                        except ValueError:
                            continue

        except IOError as e:
            logger.warning(f"读取CSV失败: {e}")

        if not scores:
            return {"count": 0, "avgScore": 0, "maxScore": 0, "minScore": 0}

        return {
            "count": len(scores),
            "avgScore": sum(scores) / len(scores),
            "maxScore": max(scores),
            "minScore": min(scores),
            "sources": list(set(sources))
        }


# 导出主要类
__all__ = ["SentimentAnalyzer"]
