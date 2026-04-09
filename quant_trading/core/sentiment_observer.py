"""
市场情绪观测器 (Market Sentiment Observer)
整合多种数据源监测市场情绪状态
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal
import numpy as np

from .logger import logger
from .event_bus import EventBus, Event


class MarketSentimentObserver:
    """
    多维度市场情绪观测器

    观测维度:
    1. 恐慌贪婪指数 (Fear & Greed)
    2. 资金费率 (Funding Rate)
    3. 多空比 (Long/Short Ratio)
    4. 价格动量 (Momentum)
    5. 波动率 (Volatility)
    6. 成交量 (Volume Analysis)
    """

    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        self.bus = event_bus
        self.config = config
        self.market = None  # 将在启动时注入

        # 情绪数据存储
        self.sentiment_data: Dict[str, Any] = {
            "overall_score": 0.0,  # -100 (极度恐慌) 到 100 (极度贪婪)
            "fear_greed_index": 50,
            "funding_rate_sentiment": 0,
            "long_short_ratio": 1.0,
            "momentum_score": 0,
            "volatility_index": 0,
            "volume_surprise": 0,
            "dominance": {},  # BTC/ETH dominance
            "trend": "NEUTRAL",
            "last_update": None
        }

        # 历史数据 (用于计算趋势)
        self.history: List[Dict] = []
        self.max_history = 100

        # 订阅事件
        self.bus.subscribe('MARKET_TICKER', self.on_ticker)
        self.bus.subscribe('MARKET_CANDLE', self.on_candle)

        logger.info("📊 [SentimentObserver] 市场情绪观测器已初始化")

    async def start(self, market_data_manager=None):
        """启动情绪观测器"""
        if market_data_manager:
            self.market = market_data_manager

        # 启动定期分析任务
        asyncio.create_task(self._periodic_analysis())
        logger.info("📊 [SentimentObserver] 市场情绪观测已启动")

    async def on_ticker(self, event: Event):
        """处理实时价格数据"""
        ticker = event.payload
        # 记录关键数据用于情绪分析
        if ticker.get('symbol') == 'BTC-USDT-SWAP':
            self.sentiment_data['btc_price'] = ticker.get('last', 0)
        elif ticker.get('symbol') == 'ETH-USDT-SWAP':
            self.sentiment_data['eth_price'] = ticker.get('last', 0)

    async def on_candle(self, event: Event):
        """处理K线数据用于技术分析"""
        candle = event.payload
        # 可以在这里计算更复杂的技术指标
        pass

    async def _periodic_analysis(self):
        """定期执行综合情绪分析"""
        while True:
            try:
                await self._analyze_fear_greed()
                await self._analyze_funding_rate()
                await self._analyze_momentum()
                await self._analyze_volatility()
                await self._calculate_overall_sentiment()

                # 发布情绪更新事件
                await self.bus.publish('SENTIMENT_UPDATE', {
                    'sentiment': self.sentiment_data,
                    'timestamp': datetime.now().isoformat()
                })

                # 记录历史
                self._record_history()

            except Exception as e:
                logger.error(f"📊 [SentimentObserver] 分析失败: {e}")

            await asyncio.sleep(60)  # 每分钟更新

    async def _analyze_fear_greed(self):
        """
        分析恐慌贪婪指数

        基于多个因素:
        - 价格动量
        - 市场波动率
        - 成交量
        - 社交媒体情绪 (如果可用)
        """
        # 简化版: 基于价格变化计算
        btc_price = self.sentiment_data.get('btc_price', 0)
        eth_price = self.sentiment_data.get('eth_price', 0)

        if btc_price == 0 or len(self.history) < 2:
            return

        # 计算价格变化率
        prev_btc = self.history[-1].get('btc_price', btc_price)
        price_change_pct = ((btc_price - prev_btc) / prev_btc) * 100 if prev_btc > 0 else 0

        # 基于价格变化计算恐慌贪婪 (简化版)
        # 大涨 = 贪婪, 大跌 = 恐慌
        fg_index = 50 + (price_change_pct * 5)  # 1%变化 = 5点
        fg_index = max(0, min(100, fg_index))  # 限制在0-100

        self.sentiment_data['fear_greed_index'] = int(fg_index)
        self.sentiment_data['price_change_pct'] = price_change_pct

    async def _analyze_funding_rate(self):
        """
        分析资金费率情绪

        正费率高 = 多头过度 = 潜在回调
        负费率高 = 空头过度 = 潜在反弹
        """
        # 这里需要从交易所获取实际资金费率
        # 模拟数据
        funding_rate = 0.01  # 0.01% 正费率

        if funding_rate > 0.05:
            sentiment = "EXTREME_GREED"  # 多头拥挤
        elif funding_rate > 0.02:
            sentiment = "GREED"
        elif funding_rate < -0.05:
            sentiment = "EXTREME_FEAR"  # 空头拥挤
        elif funding_rate < -0.02:
            sentiment = "FEAR"
        else:
            sentiment = "NEUTRAL"

        self.sentiment_data['funding_rate_sentiment'] = sentiment
        self.sentiment_data['funding_rate'] = funding_rate

    async def _analyze_momentum(self):
        """
        分析价格动量
        """
        if len(self.history) < 10:
            return

        # 获取最近的价格数据
        recent_prices = [h.get('btc_price', 0) for h in self.history[-10:] if h.get('btc_price', 0) > 0]

        if len(recent_prices) < 5:
            return

        # 计算RSI (简化版)
        gains = []
        losses = []

        for i in range(1, len(recent_prices)):
            change = recent_prices[i] - recent_prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 动量得分 (-50 到 +50)
        momentum_score = rsi - 50
        self.sentiment_data['momentum_score'] = momentum_score
        self.sentiment_data['rsi'] = rsi

    async def _analyze_volatility(self):
        """
        分析市场波动率
        """
        if len(self.history) < 20:
            return

        recent_prices = [h.get('btc_price', 0) for h in self.history[-20:] if h.get('btc_price', 0) > 0]

        if len(recent_prices) < 10:
            return

        # 计算标准差
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)

        if returns:
            volatility = np.std(returns) * 100  # 转换为百分比
            self.sentiment_data['volatility_index'] = volatility

    async def _calculate_overall_sentiment(self):
        """
        计算综合情绪指数

        整合所有维度得出最终得分
        """
        # 恐慌贪婪 (0-100 -> -50到+50)
        fg = self.sentiment_data.get('fear_greed_index', 50) - 50

        # 动量 (-50到+50)
        mom = self.sentiment_data.get('momentum_score', 0)

        # 资金费率情绪转换为分数
        fr_sentiment = self.sentiment_data.get('funding_rate_sentiment', 'NEUTRAL')
        fr_map = {
            'EXTREME_FEAR': -40,
            'FEAR': -20,
            'NEUTRAL': 0,
            'GREED': 20,
            'EXTREME_GREED': 40
        }
        fr = fr_map.get(fr_sentiment, 0)

        # 波动率 (低波动=稳定, 高波动=不确定)
        vol = self.sentiment_data.get('volatility_index', 0)
        vol_score = -min(20, vol)  # 高波动略微降低得分

        # 综合得分 (加权平均)
        overall = (fg * 0.4 + mom * 0.3 + fr * 0.2 + vol_score * 0.1)
        overall = max(-100, min(100, overall))

        self.sentiment_data['overall_score'] = round(overall, 2)

        # 确定趋势标签
        if overall >= 60:
            trend = "STRONGLY_BULLISH"
        elif overall >= 30:
            trend = "BULLISH"
        elif overall >= 10:
            trend = "MILDLY_BULLISH"
        elif overall <= -60:
            trend = "STRONGLY_BEARISH"
        elif overall <= -30:
            trend = "BEARISH"
        elif overall <= -10:
            trend = "MILDLY_BEARISH"
        else:
            trend = "NEUTRAL"

        self.sentiment_data['trend'] = trend
        self.sentiment_data['last_update'] = datetime.now().isoformat()

        logger.info(f"📊 市场情绪: {trend} (得分: {overall})")

    def _record_history(self):
        """记录历史数据"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': self.sentiment_data['overall_score'],
            'fear_greed_index': self.sentiment_data['fear_greed_index'],
            'momentum_score': self.sentiment_data.get('momentum_score', 0),
            'trend': self.sentiment_data['trend'],
            'btc_price': self.sentiment_data.get('btc_price', 0),
            'eth_price': self.sentiment_data.get('eth_price', 0)
        }

        self.history.append(record)

        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_sentiment(self) -> Dict[str, Any]:
        """获取当前情绪数据"""
        return self.sentiment_data.copy()

    def get_sentiment_summary(self) -> str:
        """获取情绪摘要文本"""
        score = self.sentiment_data.get('overall_score', 0)
        trend = self.sentiment_data.get('trend', 'UNKNOWN')

        emoji_map = {
            'STRONGLY_BULLISH': '🚀🚀🚀',
            'BULLISH': '🚀🚀',
            'MILDLY_BULLISH': '🚀',
            'NEUTRAL': '😐',
            'MILDLY_BEARISH': '📉',
            'BEARISH': '📉📉',
            'STRONGLY_BEARISH': '📉📉📉'
        }

        emoji = emoji_map.get(trend, '❓')

        return f"{emoji} {trend} (得分: {score:+.1f})"

    def get_history(self, limit: int = 50) -> List[Dict]:
        """获取历史数据"""
        return self.history[-limit:]
