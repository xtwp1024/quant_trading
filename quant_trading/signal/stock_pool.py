#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Pool Classifier - 股票池分类器
将交易标的分为三类:
1. 做T类 - 高波动、日内交易机会多
2. 趋势类 - 趋势明确、适合趋势跟踪
3. 弱势类 - 震荡偏弱、等待止跌企稳信号

支持真实A股数据接入 via akshare
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("StockPool")


class AStockDataProvider:
    """
    A股数据提供者 - 从akshare获取真实数据

    支持:
    - 单个股票历史K线数据
    - 批量股票数据获取
    - 量比数据
    """

    def __init__(self, adjust: str = "qfq"):
        """
        初始化A股数据提供者

        Args:
            adjust: 复权类型 "qfq"(前复权) / "hfq"(后复权) / ""(不复权)
        """
        self.adjust = adjust
        self._client = None

    def _get_client(self):
        """延迟初始化akshare客户端"""
        if self._client is None:
            from quant_trading.data.providers import AkshareMarketDataClient
            self._client = AkshareMarketDataClient(adjust=self.adjust)
        return self._client

    def fetch_ohlcv(
        self,
        symbol: str,
        days: int = 250,
        period_type: str = "1d"
    ) -> Tuple[np.ndarray, float]:
        """
        获取单个股票的OHLCV数据

        Args:
            symbol: 股票代码，如 "000001" (平安银行)
            days: 获取历史数据天数
            period_type: K线周期 "1d"/"1w"/"1m"

        Returns:
            (ohlcv, volume_ratio): numpy数组 (n,5) 和 量比
        """
        client = self._get_client()
        end_date = datetime.now()

        try:
            df = client.fetch(symbol, start=None, end=end_date, period_type=period_type)
        except Exception as exc:
            logger.warning(f"获取{symbol}数据失败: {exc}")
            return np.array([]), 1.0

        if df.empty or len(df) < 20:
            logger.warning(f"{symbol}数据不足: {len(df) if not df.empty else 0}条")
            return np.array([]), 1.0

        ohlcv = self._df_to_ohlcv(df)
        volume_ratio = self._calculate_volume_ratio(df)

        if len(ohlcv) > days:
            ohlcv = ohlcv[-days:]

        return ohlcv, volume_ratio

    def _df_to_ohlcv(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame转换为numpy OHLCV数组 (n, 5) = [open, high, low, close, volume]"""
        # 列名映射 (akshare使用中文列名)
        col_map = {
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
        }
        rows = []
        for _, row in df.iterrows():
            o = self._get_float(row, "open", "开盘")
            h = self._get_float(row, "high", "最高")
            l = self._get_float(row, "low", "最低")
            c = self._get_float(row, "close", "收盘")
            v = self._get_float(row, "volume", "成交量")
            rows.append([o, h, l, c, v])

        if not rows:
            return np.array([])

        arr = np.array(rows)
        valid_mask = ~(np.isnan(arr).any(axis=1) | (arr[:, 4] == 0))
        return arr[valid_mask]

    @staticmethod
    def _get_float(row: pd.Series, *keys) -> float:
        """尝试从多个可能的键中获取浮点数"""
        for key in keys:
            try:
                val = row.get(key)
                if val is not None and not pd.isna(val):
                    return float(val)
            except (ValueError, TypeError):
                continue
        return 0.0

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """计算量比 (今日成交量/5日平均成交量)"""
        vol_col = None
        for col in ["volume", "成交量"]:
            if col in df.columns:
                vol_col = col
                break

        if len(df) < 6 or vol_col is None:
            return 1.0

        volumes = df[vol_col].dropna().values
        if len(volumes) < 6:
            return 1.0

        today_vol = volumes[-1]
        avg_5day = np.mean(volumes[-6:-1])
        if avg_5day <= 0:
            return 1.0

        return today_vol / avg_5day

    def fetch_batch(
        self,
        symbols: List[str],
        days: int = 250
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        批量获取多只股票的OHLCV数据

        Args:
            symbols: 股票代码列表
            days: 历史数据天数

        Returns:
            (symbols_ohlcv, volume_ratios)
        """
        symbols_ohlcv = {}
        volume_ratios = {}

        for symbol in symbols:
            try:
                ohlcv, vol_ratio = self.fetch_ohlcv(symbol, days)
                if len(ohlcv) >= 20:
                    symbols_ohlcv[symbol] = ohlcv
                    volume_ratios[symbol] = vol_ratio
                    logger.debug(f"{symbol}: 获取{len(ohlcv)}条数据, 量比={vol_ratio:.2f}")
            except Exception as exc:
                logger.warning(f"获取{symbol}数据异常: {exc}")
                continue

        return symbols_ohlcv, volume_ratios

    def get_stock_list_by_market(
        self,
        market: str = "沪深A股"
    ) -> List[Dict[str, str]]:
        """
        获取市场股票列表

        Args:
            market: 市场类型 "沪深A股" / "科创板" / "创业板" 等

        Returns:
            [{"symbol": "000001", "name": "平安银行"}, ...]
        """
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            return [
                {"symbol": row["code"], "name": row["name"]}
                for _, row in df.iterrows()
            ]
        except Exception as exc:
            logger.error(f"获取股票列表失败: {exc}")
            return []

    # 预设热门股票池 (沪深成交额排名靠前的股票)
    _POPULAR_STOCKS = [
        {"symbol": "600519", "name": "贵州茅台"},
        {"symbol": "000858", "name": "五粮液"},
        {"symbol": "601318", "name": "中国平安"},
        {"symbol": "600036", "name": "招商银行"},
        {"symbol": "300750", "name": "宁德时代"},
        {"symbol": "000001", "name": "平安银行"},
        {"symbol": "600276", "name": "恒瑞医药"},
        {"symbol": "000002", "name": "万科A"},
        {"symbol": "601166", "name": "兴业银行"},
        {"symbol": "600900", "name": "长江电力"},
        {"symbol": "300059", "name": "东方财富"},
        {"symbol": "600030", "name": "中信证券"},
        {"symbol": "601888", "name": "中国中免"},
        {"symbol": "600887", "name": "伊利股份"},
        {"symbol": "002594", "name": "比亚迪"},
        {"symbol": "601012", "name": "隆基绿能"},
        {"symbol": "600028", "name": "中国石化"},
        {"symbol": "601398", "name": "工商银行"},
        {"symbol": "601939", "name": "建设银行"},
        {"symbol": "600000", "name": "浦发银行"},
        {"symbol": "000333", "name": "美的集团"},
        {"symbol": "002415", "name": "海康威视"},
        {"symbol": "601328", "name": "交通银行"},
        {"symbol": "601088", "name": "中国神华"},
        {"symbol": "601601", "name": "中国太保"},
    ]

    def get_top_stocks_by_amount(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取成交额排名前列的股票 (用于热门股票池)

        由于实时行情接口不稳定,优先使用预设热门股票池

        Args:
            limit: 返回数量

        Returns:
            [{"symbol": "000001", "name": "平安银行", "amount": 123456}, ...]
        """
        # 优先使用预设热门股票池
        result = self._POPULAR_STOCKS[:limit]
        for stock in result:
            stock["amount"] = 1_000_000_000  # 预设成交额
        return result


class PoolType(str, Enum):
    """股票池类型"""
    SCALPING = "做T"
    TREND = "趋势"
    WATCH = "弱势"


@dataclass
class PoolClassification:
    """分类结果"""
    symbol: str
    pool_type: PoolType
    confidence: float
    score: float
    characteristics: Dict[str, Any]
    reason: str
    trend_signal: Optional[str] = None
    stabilization_signal: Optional[str] = None


@dataclass
class MarketCharacteristics:
    """市场特征"""
    volatility: float
    trend_strength: float
    momentum: float
    rsi: float
    volume_ratio: float
    price_position: float
    amplitude: float


class StockPoolClassifier:
    """
    股票池分类器

    分类逻辑:
    - 做T类: 高波动(ATR>3%),振幅大,成交量活跃
    - 趋势类: 趋势强(ADX>25),动量明确,RSI在40-60区间外
    - 弱势类: 低波动,趋势弱,RSI偏弱,价格位置偏低
    """

    THRESHOLDS = {
        "做T": {
            "volatility_min": 3.0,
            "amplitude_min": 2.0,
            "volume_min": 1.2,
        },
        "趋势": {
            "adx_min": 25,
            "momentum_min": 1.0,
            "rsi_strong": (30, 70),
        },
        "弱势": {
            "volatility_max": 2.0,
            "adx_max": 20,
            "rsi_weak_max": 45,
            "price_position_max": 40,
        }
    }

    def __init__(self):
        self.classifications: Dict[str, PoolClassification] = {}

    def analyze(self, symbol: str, ohlcv: np.ndarray, volume_ratio: float = 1.0) -> PoolClassification:
        """
        分析并分类单个标的

        Args:
            symbol: 标的代码
            ohlcv: OHLCV数据 (n, 5) = [open, high, low, close, volume]
            volume_ratio: 量比
        """
        if len(ohlcv) < 20:
            return self._default_classification(symbol)

        close = ohlcv[:, 3]
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]

        chars = self._calculate_characteristics(ohlcv, volume_ratio)
        scores = self._calculate_pool_scores(chars)

        best_pool = max(scores, key=scores.get)
        confidence = scores[best_pool] / sum(scores.values()) if sum(scores.values()) > 0 else 0.5

        reason = self._generate_reason(best_pool, chars)
        trend_signal, stabilization_signal = self._generate_signals(best_pool, chars)

        classification = PoolClassification(
            symbol=symbol,
            pool_type=PoolType(best_pool),
            confidence=confidence,
            score=scores[best_pool],
            characteristics={
                "volatility": chars.volatility,
                "trend_strength": chars.trend_strength,
                "momentum": chars.momentum,
                "rsi": chars.rsi,
                "volume_ratio": chars.volume_ratio,
                "price_position": chars.price_position,
                "amplitude": chars.amplitude,
            },
            reason=reason,
            trend_signal=trend_signal,
            stabilization_signal=stabilization_signal
        )

        self.classifications[symbol] = classification
        return classification

    def _calculate_characteristics(self, ohlcv: np.ndarray, volume_ratio: float) -> MarketCharacteristics:
        """计算市场特征"""
        close = ohlcv[:, 3]
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]

        atr = self._calculate_atr(ohlcv, 14)
        current_price = close[-1]
        volatility = (atr / current_price) * 100 if current_price > 0 else 0

        adx = self._calculate_adx(ohlcv, 14)
        roc = ((close[-1] - close[-14]) / close[-14] * 100) if len(close) >= 14 and close[-14] > 0 else 0
        rsi = self._calculate_rsi(close, 14)

        sma20 = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
        std20 = np.std(close[-20:]) if len(close) >= 20 else 0
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        price_position = ((current_price - lower) / (upper - lower) * 100) if upper > lower else 50

        amplitude = ((np.max(high) - np.min(low)) / current_price * 100) if current_price > 0 else 0

        return MarketCharacteristics(
            volatility=volatility,
            trend_strength=adx,
            momentum=roc,
            rsi=rsi,
            volume_ratio=volume_ratio,
            price_position=price_position,
            amplitude=amplitude
        )

    def _calculate_atr(self, ohlcv: np.ndarray, period: int) -> float:
        """计算ATR"""
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        close = ohlcv[:, 3]

        tr = np.zeros(len(ohlcv))
        tr[0] = high[0] - low[0]
        for i in range(1, len(ohlcv)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        return np.mean(tr[-period:])

    def _calculate_adx(self, ohlcv: np.ndarray, period: int) -> float:
        """计算简化版ADX"""
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        close = ohlcv[:, 3]

        plus_dm = np.zeros(len(ohlcv))
        minus_dm = np.zeros(len(ohlcv))

        for i in range(1, len(ohlcv)):
            high_diff = high[i] - high[i - 1]
            low_diff = low[i - 1] - low[i]

            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        atr = self._calculate_atr(ohlcv, period)
        plus_di = np.mean(plus_dm[-period:]) / atr * 100 if atr > 0 else 0
        minus_di = np.mean(minus_dm[-period:]) / atr * 100 if atr > 0 else 0

        adx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        return adx

    def _calculate_rsi(self, close: np.ndarray, period: int) -> float:
        """计算RSI"""
        if len(close) < period + 1:
            return 50

        deltas = np.diff(close, prepend=close[0])
        deltas = np.insert(deltas, 0, 0)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_pool_scores(self, chars: MarketCharacteristics) -> Dict[str, float]:
        """计算各分类得分"""
        scores = {"做T": 0.0, "趋势": 0.0, "弱势": 0.0}

        # 做T评分
        t_score = 0
        if chars.volatility > self.THRESHOLDS["做T"]["volatility_min"]:
            t_score += 2
        elif chars.volatility > self.THRESHOLDS["做T"]["volatility_min"] / 2:
            t_score += 1

        if chars.amplitude > self.THRESHOLDS["做T"]["amplitude_min"]:
            t_score += 2
        elif chars.amplitude > self.THRESHOLDS["做T"]["amplitude_min"] / 2:
            t_score += 1

        if chars.volume_ratio > self.THRESHOLDS["做T"]["volume_min"]:
            t_score += 1

        scores["做T"] = t_score / 5.0

        # 趋势评分
        trend_score = 0
        if chars.trend_strength > self.THRESHOLDS["趋势"]["adx_min"]:
            trend_score += 3
        elif chars.trend_strength > self.THRESHOLDS["趋势"]["adx_min"] / 2:
            trend_score += 1.5

        if abs(chars.momentum) > self.THRESHOLDS["趋势"]["momentum_min"]:
            trend_score += 2

        rsi_min, rsi_max = self.THRESHOLDS["趋势"]["rsi_strong"]
        if chars.rsi < rsi_min or chars.rsi > rsi_max:
            trend_score += 1

        scores["趋势"] = trend_score / 6.0

        # 弱势评分
        weak_score = 0
        if chars.volatility < self.THRESHOLDS["弱势"]["volatility_max"]:
            weak_score += 2

        if chars.trend_strength < self.THRESHOLDS["弱势"]["adx_max"]:
            weak_score += 2

        if chars.rsi < self.THRESHOLDS["弱势"]["rsi_weak_max"]:
            weak_score += 2

        if chars.price_position < self.THRESHOLDS["弱势"]["price_position_max"]:
            weak_score += 1

        scores["弱势"] = weak_score / 7.0

        return scores

    def _generate_reason(self, pool_type: str, chars: MarketCharacteristics) -> str:
        """生成分类原因"""
        if pool_type == "做T":
            return f"波动率{chars.volatility:.1f}%,振幅{chars.amplitude:.1f}%,量比{chars.volume_ratio:.1f} -> 高波动环境,适合日内交易"
        elif pool_type == "趋势":
            return f"ADX={chars.trend_strength:.1f},ROC={chars.momentum:+.1f}%,RSI={chars.rsi:.1f} -> 趋势明确,适合趋势跟踪"
        else:
            return f"波动率{chars.volatility:.1f}%,ADX={chars.trend_strength:.1f},RSI={chars.rsi:.1f},价格位置{chars.price_position:.0f}% -> 弱势震荡,观望等待"

    def _generate_signals(self, pool_type: str, chars: MarketCharacteristics) -> Tuple[Optional[str], Optional[str]]:
        """生成信号"""
        trend_signal = None
        stabilization_signal = None

        if pool_type == "趋势":
            if chars.momentum > 2:
                trend_signal = "强势上涨趋势"
            elif chars.momentum > 0.5:
                trend_signal = "温和上涨趋势"
            elif chars.momentum < -2:
                trend_signal = "强势下跌趋势"
            elif chars.momentum < -0.5:
                trend_signal = "温和下跌趋势"
            else:
                trend_signal = "趋势整理"

        elif pool_type == "弱势":
            if chars.rsi > 40 and chars.rsi < 60:
                if chars.price_position > 30:
                    stabilization_signal = "初步企稳"
                else:
                    stabilization_signal = "低位企稳观察"

            if chars.volatility < 1.5 and chars.trend_strength < 15:
                stabilization_signal = "极度缩量企稳"
            elif chars.rsi < 30:
                stabilization_signal = "超卖等待反弹"
            elif chars.price_position < 20:
                stabilization_signal = "接近支撑区域"

        return trend_signal, stabilization_signal

    def _default_classification(self, symbol: str) -> PoolClassification:
        """默认分类"""
        return PoolClassification(
            symbol=symbol,
            pool_type=PoolType.WATCH,
            confidence=0.5,
            score=0.0,
            characteristics={},
            reason="数据不足，默认归类为弱势观察",
            stabilization_signal="等待数据积累"
        )


class StockPoolManager:
    """
    股票池管理器

    根据分类结果指导交易:
    - 做T类: 使用网格/马丁策略,高频交易
    - 趋势类: 使用趋势跟踪策略,突破买入
    - 弱势类: 观望为主,等待企稳信号后轻仓介入
    """

    def __init__(self):
        self.classifier = StockPoolClassifier()
        self.pool_stocks: Dict[PoolType, List[str]] = {
            PoolType.SCALPING: [],
            PoolType.TREND: [],
            PoolType.WATCH: []
        }

    def update_pool(self, symbols_ohlcv: Dict[str, np.ndarray], volume_ratios: Dict[str, float] = None):
        """更新股票池"""
        volume_ratios = volume_ratios or {}
        self.pool_stocks = {PoolType.SCALPING: [], PoolType.TREND: [], PoolType.WATCH: []}

        for symbol, ohlcv in symbols_ohlcv.items():
            vol_ratio = volume_ratios.get(symbol, 1.0)
            classification = self.classifier.analyze(symbol, ohlcv, vol_ratio)
            self.pool_stocks[classification.pool_type].append(symbol)

    def get_pool_summary(self) -> Dict[str, Any]:
        """获取股票池汇总"""
        result = {}
        for pool_type in [PoolType.SCALPING, PoolType.TREND, PoolType.WATCH]:
            symbols = self.pool_stocks[pool_type]
            result[pool_type.value] = {
                "count": len(symbols),
                "symbols": symbols,
                "details": [
                    {
                        "symbol": sym,
                        "reason": self.classifier.classifications[sym].reason,
                        "signal": (
                            self.classifier.classifications[sym].trend_signal or
                            self.classifier.classifications[sym].stabilization_signal
                        )
                    }
                    for sym in symbols if sym in self.classifier.classifications
                ]
            }
        return result

    def get_trading_strategy(self, symbol: str) -> str:
        """根据股票池类型返回推荐策略"""
        if symbol in self.pool_stocks[PoolType.SCALPING]:
            return "网格/马丁策略,高抛低吸,止损略宽"
        elif symbol in self.pool_stocks[PoolType.TREND]:
            return "趋势跟踪策略,突破买入/杀跌,止损略紧"
        elif symbol in self.pool_stocks[PoolType.WATCH]:
            return "观望策略,等待企稳信号,轻仓试探"
        return "默认策略"

    def get_stabilization_candidates(self) -> List[Dict[str, Any]]:
        """
        获取弱势类中显示企稳信号的标的

        Returns:
            [{symbol, signal, reason}, ...]
        """
        candidates = []
        for symbol in self.pool_stocks[PoolType.WATCH]:
            cls = self.classifier.classifications.get(symbol)
            if cls and cls.stabilization_signal:
                candidates.append({
                    "symbol": symbol,
                    "signal": cls.stabilization_signal,
                    "reason": cls.reason,
                    "characteristics": cls.characteristics
                })
        return candidates

    def get_trend_stocks(self) -> List[Dict[str, Any]]:
        """获取趋势类标的及其信号"""
        stocks = []
        for symbol in self.pool_stocks[PoolType.TREND]:
            cls = self.classifier.classifications.get(symbol)
            if cls:
                stocks.append({
                    "symbol": symbol,
                    "signal": cls.trend_signal,
                    "reason": cls.reason,
                    "momentum": cls.characteristics.get("momentum", 0),
                    "adx": cls.characteristics.get("trend_strength", 0)
                })
        return sorted(stocks, key=lambda x: abs(x["momentum"]), reverse=True)
