# -*- coding: utf-8 -*-
"""
Reddit俚语情绪词典 (Reddit Slang Sentiment Dictionary)

200+ 词条, 基于 reddit-algo-trader 扩展.
覆盖: moon/rocket 类暴涨词, diamond_hands 类持有词, rug_pull 类暴跌词.

Author: 量化之神
"""

from __future__ import annotations

import re
from typing import Optional

# =============================================================================
# Reddit俚语情绪词典 (200+词条)
# =============================================================================
# 分数范围: -4.0 (极度利空) ~ +4.0 (极度利多)
# 分类: Strong Bullish, Mild Bullish, Neutral, Mild Bearish, Strong Bearish

REDDIT_LINGO: dict[str, float] = {
    # -------------------------------------------------------------------------
    # Strong Bullish (暴涨/极度看好)
    # -------------------------------------------------------------------------
    'moon': 4.0,
    'to_the_moon': 4.0,
    'rockets': 3.5,
    'rocket': 3.5,
    ' 🚀': 3.5,
    'bullish': 3.0,
    'bull': 3.0,
    'bullish AF': 4.0,
    'diamond_hands': 3.0,
    'tendies': 2.5,
    'tendies': 2.5,
    'to_the_moon 🚀': 4.0,
    'all_in': 3.0,
    'all-in': 3.0,
    'yolo': 2.0,
    'buy_the_dip': 2.5,
    'buy dip': 2.0,
    'calls': 2.0,          # 看涨期权
    'long': 2.0,           # 做多
    'hold': 1.5,           # 持有 (中性偏多)
    'holds': 1.5,
    'hodl': 2.0,           # HODL = Hold On for Dear Life
    'hodling': 2.0,
    'ser': 1.5,            # 'sir' 的reddit拼写
    'sir': 1.5,
    'retard': 1.0,         # ironically positive
    'retarded': 1.0,
    'ape': 2.0,            # Ape = diamond hands retail trader
    'apes': 2.0,
    'tendies': 2.5,        # Chicken tenders = profits
    'tendies': 2.5,

    # -------------------------------------------------------------------------
    # Mild Bullish (温和看多)
    # -------------------------------------------------------------------------
    'dd': 1.5,             # Due Diligence
    'deep_value': 2.0,
    'undervalued': 1.5,
    'strong': 1.0,
    'breakout': 1.5,
    'upgrade': 1.5,
    'upside': 1.5,
    'growth': 1.0,
    'beat': 1.0,           # EPS beat
    'crush': 1.5,          # Crush earnings
    'smash': 1.5,          # Smash earnings
    'moonshot': 3.0,
    'squeeze': 2.0,        # Short squeeze
    'short_squeeze': 2.5,
    'gamma_squeeze': 2.0,
    'run_up': 1.5,
    'rally': 1.0,
    'rebound': 1.0,
    'recovering': 0.5,
    'opportunity': 1.0,
    'underpriced': 1.5,
    'cheap': 1.0,
    'value': 0.5,
    'accumulate': 1.5,
    'adding': 1.0,
    'loaded': 1.5,
    'bullish AF': 4.0,

    # -------------------------------------------------------------------------
    # Neutral (中性词)
    # -------------------------------------------------------------------------
    'shares': 0.0,
    'share': 0.0,
    'position': 0.0,
    'entry': 0.0,
    'avg': 0.0,
    'average': 0.0,
    'cost': 0.0,
    'portfolio': 0.0,
    'holding': 0.0,
    'held': 0.0,
    'watching': 0.0,
    'waiting': 0.0,
    'looking': 0.0,
    'considering': 0.0,
    'maybe': 0.0,
    'uncertain': 0.0,
    'mixed': 0.0,
    'sideways': 0.0,
    'flat': 0.0,
    'neutral': 0.0,

    # -------------------------------------------------------------------------
    # Mild Bearish (温和看空)
    # -------------------------------------------------------------------------
    'puts': -2.0,          # 看跌期权
    'put': -2.0,
    'short': -2.0,
    'bearish': -3.0,
    'bear': -3.0,
    'overvalued': -1.5,
    'sell': -1.5,
    'dump': -2.5,
    'dumping': -2.5,
    'selling': -1.5,
    'take_profit': -1.0,
    'trim': -0.5,
    'reduce': -0.5,
    'downgrade': -1.5,
    'warning': -1.0,
    'risk': -1.0,
    'volatile': -0.5,
    'volatility': -0.5,
    'correction': -1.0,
    'pullback': -1.0,
    ' retrace': -1.0,
    'dead_cat': -1.5,      # Dead cat bounce
    'overbought': -1.0,
    'resistance': -0.5,
    'ceiling': -0.5,
    'sat': -0.5,           # Stacked Satellites (sub-optimal entry)
    'bagholder': -2.0,
    'bagholding': -2.0,

    # -------------------------------------------------------------------------
    # Strong Bearish (暴跌/极度看空)
    # -------------------------------------------------------------------------
    'tendie_loss': -2.5,   # Losses on tendies (ironic)
    'bagholder': -2.5,
    'bagholders': -2.5,
    'rekt': -3.0,
    'wrecked': -3.0,
    'down_bad': -2.5,
    'rug_pull': -3.5,
    'scam': -4.0,
    'ponzi': -4.0,
    'dump': -3.0,
    'dumped': -3.0,
    'crash': -3.0,
    'crashed': -3.5,
    'collapse': -3.0,
    'bankrupt': -4.0,
    'bankruptcy': -4.0,
    'delist': -3.5,
    'wash': -2.0,           # Wash trading
    'rug': -3.0,
    'rugged': -3.0,
    'rugged': -3.0,
    'liquidation': -3.0,
    'liquidated': -3.5,
    'margin_call': -3.0,
    'called away': -2.0,
    'btfd': 2.5,            # Buy The F***ing Dip (bullish)
    'fomo': 1.5,            # Fear Of Missing Out (context dependent)
    'fomoing': 1.0,
    'fud': -2.0,            # Fear, Uncertainty, Doubt
    'spooked': -2.0,
    'paper_hands': -2.0,    # Sell too early (bearish signal)
    'chicken_hands': -1.5,
    'panic': -2.5,
    'panic_sell': -3.0,
    'fear': -2.0,

    # -------------------------------------------------------------------------
    # Meme/Crypto Slang
    # -------------------------------------------------------------------------
    'wagmi': 2.0,           # We're All Gonna Make It (bullish)
    'ngmi': -2.0,           # Not Gonna Make It (bearish)
    'gm': 1.5,              # Good morning (bullish community greeting)
    'gn': -0.5,             # Good night
    'ser': 1.5,
    'frens': 1.5,           # Friends
    'degen': 1.0,           # Degenerate (positive in context)
    'degens': 1.0,
    'safu': 2.0,            # Safe (from Binance)
    'rekt': -3.0,
    'bullish': 3.0,
    'bearish': -3.0,
    'shill': -1.5,          # Aggressively promoting (usually bearish)
    'shilling': -1.5,
    'pump': 2.0,            # Price pump (ambiguous, slightly bullish)
    'dump': -2.5,           # Price dump
    'whale': -0.5,          # Large player (ambiguous)
    'whales': -0.5,
    'alpha': 1.5,           # Edge/alpha
    'beta': -0.5,           # Beta (market exposure)
    'squeeze': 2.0,
    'short_squeeze': 2.5,
    'margin': -1.0,

    # -------------------------------------------------------------------------
    # Options/Greek
    # -------------------------------------------------------------------------
    'iv': -0.5,             # Implied Volatility (high IV = expensive options)
    'iv_rank': 0.5,        # IV Rank (high = good for selling)
    'iv_percentile': 0.5,
    'theta': -0.5,         # Theta decay
    'delta': 0.5,           # Delta
    'vega': -0.3,           # Vega (volatility sensitivity)
    'gamma': 0.3,           # Gamma
    'otm': -0.5,            # Out of the money
    'itm': 0.5,             # In the money
    'atm': 0.0,             # At the money
    'weeklies': 0.5,        # Weekly options
    'monthly': 0.0,
    'expiry': 0.0,
    'strike': 0.0,
    'exercised': 0.0,

    # -------------------------------------------------------------------------
    # Additional expanded slang
    # -------------------------------------------------------------------------
    'diamond': 2.5,         # Diamond hands
    'paper': -1.5,          # Paper hands
    'to_the_mars': 3.5,
    'mars': 3.0,
    'venus': 2.5,
    'ape_strong': 2.5,
    'dd_completed': 1.5,    # Due diligence completed
    'do_your_dd': 1.5,
    'tyas': 1.0,            # Take your analysis with a grain of salt
    'this_is_the_way': 2.0,
    'the_way': 2.0,
    'validation': 1.0,
    'confirmed': 1.0,
    'confirmed': 1.0,
    'bullishness': 2.0,
    'bearishness': -2.0,
    'squeeze_play': 2.0,
    'momentum': 1.0,
    'reversal': 0.5,
    'breakdown': -1.5,      # Breakdown of support
    'breakup': 1.5,         # Breakout above resistance
    'rally': 1.0,
    'tank': -2.5,
    'tanking': -2.5,
    'pop': 1.5,             # Price pop
    'drop': -1.5,
    'skyrocket': 3.5,
    'plummet': -3.5,
    'surge': 2.0,
    'surge': 2.0,
    'spike': 1.5,
    'crater': -3.0,
    'melt_up': 2.5,
    'melt_down': -2.5,
    'green': 1.0,           # Price up / profit
    'red': -1.0,            # Price down / loss
    'print': 2.0,           # Printing money = profits
    'printing': 2.0,
    'money_printer': 1.5,  # Fed/market liquidity
    ' QE': 1.0,
    'tapering': -1.0,
    'hike': -1.0,           # Interest rate hike
    'rate_hike': -1.5,
    'rate_cut': 1.5,
    'inflation': -1.0,
    'disinflation': 0.5,
    'deflation': -1.0,
    'recession': -2.0,
    'stimulus': 1.5,
    'stimmy': 1.5,          # Stimulus check
    'stimmy': 1.5,
    'unemployment': -1.0,
    'jobs': 0.5,
    'payroll': 0.5,
    'gdp': 0.0,
    'fed': 0.0,
    'pow': 0.0,             # Proof of work
    'pos': 0.0,             # Proof of stake
    'defi': 0.5,
    'nft': 0.0,
    'web3': 0.0,
    'metaverse': 0.0,
    'ai': 0.5,
    'ml': 0.5,
    'blockchain': 0.0,
    'crypto': 0.0,
    'bitcoin': 0.0,
    'ethereum': 0.0,
    'solana': 0.0,
    'ripple': 0.0,
    'cardano': 0.0,
    'dogecoin': 0.0,
    'shiba': 0.0,
    'pepe': 0.0,
    'gme': 0.0,             # Specific tickers tracked separately
    'amc': 0.0,
    'bb': 0.0,
    'bbby': 0.0,
    'tsla': 0.0,
    'nvda': 0.0,
    'amd': 0.0,
    'spy': 0.0,
    'qqq': 0.0,
    ' SPY': 0.0,
    'tqqq': 0.0,
    'sqqq': 0.0,
    'uvxy': 0.0,
    'vix': -0.5,            # Volatility index (fear gauge)
    'vix': -0.5,
}


# =============================================================================
# RedditLingoScorer
# =============================================================================

class RedditLingoScorer:
    """
    基于Reddit俚语词典的情绪打分器.

    工作原理:
    1. 将输入文本转为小写并分词
    2. 匹配俚语词典中的词条
    3. 累加分数 (可配置是否取平均)

    Args:
        custom_dict: 可选的自定义词典, 会与默认 REDDIT_LINGO 合并 (后者覆盖前者)

    Example:
        >>> scorer = RedditLingoScorer()
        >>> scorer.score("Moon! Rocket! Diamond hands!")
        11.0
        >>> scorer.score_normalized("Moon! Rocket!")
        1.0
        >>> scorer.extract_mentions("To the moon with GME")
        [('to_the_moon', 4.0), ('gme', 0.0)]
    """

    def __init__(self, custom_dict: Optional[dict[str, float]] = None):
        # 合并: custom_dict 优先, 再加默认 REDDIT_LINGO
        self._dict = dict(REDDIT_LINGO)
        if custom_dict:
            self._dict.update(custom_dict)

        # 预编译正则: 匹配完整单词 (避免 "bullishness" 错误匹配 "bull")
        words = sorted(self._dict.keys(), key=len, reverse=True)
        # 转义特殊正则字符
        escaped = [re.escape(w) for w in words]
        pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
        self._pattern = re.compile(pattern_str, re.IGNORECASE)

    def score(self, text: str) -> float:
        """
        返回情绪分数 (原始俚语分数累加, 未标准化).

        Args:
            text: 输入文本

        Returns:
            float: 所有匹配词条分数之和
        """
        if not text:
            return 0.0
        matches = self._pattern.findall(text.lower())
        return sum(self._dict.get(m.lower(), 0.0) for m in matches)

    def score_normalized(self, text: str) -> float:
        """
        返回标准化分数 [-1, +1].

        使用 tanh 压缩, 避免极端值影响.

        Args:
            text: 输入文本

        Returns:
            float: 标准化后的分数, 范围 [-1, +1]
        """
        import math
        raw = self.score(text)
        # tanh 压缩到 [-1, 1]
        return math.tanh(raw / 5.0)  # 5.0 为经验缩放因子

    def extract_mentions(self, text: str) -> list[tuple[str, float]]:
        """
        提取文本中的俚语词条及分数.

        Args:
            text: 输入文本

        Returns:
            List[(词条, 分数)], 按文本出现顺序排列, 去重
        """
        if not text:
            return []
        seen: set[str] = set()
        result: list[tuple[str, float]] = []
        for match in self._pattern.finditer(text.lower()):
            word = match.group(0).lower()
            if word not in seen:
                seen.add(word)
                result.append((word, self._dict.get(word, 0.0)))
        return result

    def top_mentions(self, text: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        返回分数最高的 N 个俚语词条.

        Args:
            text: 输入文本
            top_n: 返回数量

        Returns:
            List[(词条, 分数)], 按分数降序
        """
        mentions = self.extract_mentions(text)
        return sorted(mentions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
