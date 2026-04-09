# -*- coding: utf-8 -*-
"""
3Commas Cyber Bots - Unified Strategy Module
3Commas 赛博机器人 - 统一策略模块

该模块集成了来自 cyberjunky/3commas-cyber-bots 的核心策略功能：
- DCABotStrategy: 美元成本平均定投机器人 (Dollar-Cost Averaging Bot)
- TrailingStopLoss: 追踪止损 & 追踪止盈 (Trailing Stop Loss & Take Profit)
- CompoundStrategy: 利润复利策略 (Profit Compounding Strategy)
- AltRankStrategy: AltRank 排名选币策略 (AltRank Score Based Pair Selection)
- GalaxyScoreStrategy: GalaxyScore 评分选币策略 (GalaxyScore Based Pair Selection)
- ThreeCommasAPI: 3Commas REST API 客户端 (urllib-based, 无外部依赖)
- MarketCollector: 市场数据采集器 (Market Data Collector)
- DealCluster: 交易聚类管理 (Deal Clustering Manager)

All API calls use pure urllib (no py3cw dependency).
所有 API 调用使用纯 urllib (无外部依赖)。

Author: Claude Code
Date: 2026-03-31
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple, Union

# Lazy import for NumPy - graceful degradation
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False


# =============================================================================
# Constants / 常量
# =============================================================================

THREECOMMAS_API_BASE = "https://api.3commas.io"
THREECOMMAS_API_V1 = f"{THREECOMMAS_API_BASE}/v1"
THREECOMMAS_API_V2 = f"{THREECOMMAS_API_BASE}/v2"

LUNARCRUSH_API_BASE = "https://lunarcrush.com/api"
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINMARKETCAP_API_BASE = "https://pro-api.coinmarketcap.com/v1"

# =============================================================================
# Exceptions / 异常类
# =============================================================================

class ThreeCommasAPIError(Exception):
    """3Commas API 错误 / 3Commas API Error"""
    def __init__(self, message: str, status_code: Optional[int] = None, error_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_data = error_data or {}


class APIRateLimitError(ThreeCommasAPIError):
    """API 速率限制错误 / API Rate Limit Error"""
    pass


class APIDataError(ThreeCommasAPIError):
    """API 数据错误 / API Data Error"""
    pass


# =============================================================================
# Utility Functions / 工具函数
# =============================================================================

def _sign_request(secret: str, params: Dict[str, Any]) -> str:
    """
    Generate HMAC SHA256 signature for 3Commas API.
    为 3Commas API 生成 HMAC SHA256 签名。

    Args:
        secret: API secret key / API 密钥
        params: Request parameters / 请求参数

    Returns:
        Hex-encoded signature / 十六进制编码的签名
    """
    if isinstance(params, dict):
        # Sort parameters by key for consistent signing
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    else:
        param_str = str(params)
    return hmac.new(
        secret.encode("utf-8"),
        param_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def _make_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Make HTTP request using urllib.
    使用 urllib 发起 HTTP 请求。

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH) / HTTP 方法
        url: Full URL / 完整 URL
        headers: Request headers / 请求头
        data: Request body data / 请求体数据
        timeout: Request timeout in seconds / 请求超时（秒）

    Returns:
        (data, error) tuple / (数据, 错误) 元组
    """
    headers = headers or {}
    body = None

    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result, None
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else "{}"
        try:
            error_data = json.loads(error_body)
        except json.JSONDecodeError:
            error_data = {"error": error_body}
        return None, error_data
    except urllib.error.URLError as e:
        return None, {"error": str(e.reason)}
    except Exception as e:
        return None, {"error": str(e)}


def _numpy_score_normalize(values, inverse: bool = False):
    """
    Normalize values to 0-1 range using NumPy.
    使用 NumPy 将值标准化到 0-1 范围。

    Args:
        values: Input array / 输入数组
        inverse: If True, lower is better (invert the normalization) / 如果为 True，越低越好（反转标准化）

    Returns:
        Normalized array / 标准化后的数组
    """
    if not _NUMPY_AVAILABLE:
        # Fallback without numpy - simple min-max normalization
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        return [1.0 - n for n in normalized] if inverse else normalized

    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.ones_like(values) * 0.5
    normalized = (values - min_val) / (max_val - min_val)
    return 1.0 - normalized if inverse else normalized


def _round_decimals_up(value: float, decimals: int) -> float:
    """
    Round up to specified decimal places (avoids floating point issues).
    向上舍入到指定小数位（避免浮点问题）。

    Args:
        value: Value to round / 要舍入的值
        decimals: Number of decimal places / 小数位数

    Returns:
        Rounded value / 舍入后的值
    """
    factor = 10 ** decimals
    return math.ceil(value * factor) / factor


def _calculate_deal_funds(
    base_order: float,
    safety_order: float,
    max_safety_orders: int,
    martingale_volume_coef: float
) -> List[float]:
    """
    Calculate total funds needed per deal.
    计算每笔交易所需的总资金。

    Args:
        base_order: Base order volume / 基础订单量
        safety_order: Safety order volume / 安全订单量
        max_safety_orders: Maximum number of safety orders / 最大安全订单数
        martingale_volume_coef: Martingale volume coefficient / 马丁格尔音量系数

    Returns:
        [total_per_deal, base_order_contribution, safety_order_contribution]
    """
    total_so = 0.0
    so_volume = safety_order
    for _ in range(max_safety_orders):
        total_so += so_volume
        so_volume *= martingale_volume_coef

    total_per_deal = base_order + total_so
    return [total_per_deal, base_order, total_so]


# =============================================================================
# ThreeCommasAPI - Pure urllib REST Client
# 3CommasAPI - 纯 urllib REST 客户端
# =============================================================================

class ThreeCommasAPI:
    """
    3Commas REST API Client (Pure Python + urllib).
    3Commas REST API 客户端（纯 Python + urllib）。

    无外部依赖，支持 HMAC 签名认证和 RSA 认证。

    Attributes:
        api_key: API key / API 密钥
        api_secret: API secret (for HMAC signing) / API 密钥（HMAC 签名用）
        rsa_key: RSA private key (optional) / RSA 私钥（可选）
        timeout: Request timeout in seconds / 请求超时（秒）
        retry_count: Number of retries on failure / 失败重试次数

    Example / 示例:
        >>> api = ThreeCommasAPI(api_key="your_key", api_secret="your_secret")
        >>> error, data = api.request("GET", "bots", action="show", action_id="12345")
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        rsa_key: Optional[str] = None,
        timeout: int = 30,
        retry_count: int = 3
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rsa_key = rsa_key
        self.timeout = timeout
        self.retry_count = retry_count

        # RSA key loading is lazy - graceful degradation
        self._rsa_key_loaded = False
        self._rsa_sign_func = None

    def _load_rsa_key(self) -> bool:
        """
        Lazily load RSA key for signing.
        懒加载 RSA 密钥用于签名。

        Returns:
            True if RSA key was loaded successfully / 是否成功加载 RSA 密钥
        """
        if self.rsa_key and not self._rsa_key_loaded:
            try:
                # Try to import Crypto for RSA support
                from Crypto.PublicKey import RSA
                from Crypto.Signature import pkcs1_15
                from Crypto.Hash import SHA256
                self._rsa_key_obj = RSA.import_key(self.rsa_key)
                self._rsa_sign_func = lambda data: pkcs1_15.new(self._rsa_key_obj).sign(SHA256.new(data.encode()))
                self._rsa_key_loaded = True
                return True
            except ImportError:
                # pycryptodome not available, will use HMAC instead
                self._rsa_key_loaded = True
                return False
        return bool(self.rsa_key)

    def _sign(self, params: Dict[str, Any]) -> str:
        """
        Sign request parameters.
        签名请求参数。

        Args:
            params: Request parameters / 请求参数

        Returns:
            Signature string / 签名字符串
        """
        if self.rsa_key and self._load_rsa_key() and self._rsa_sign_func:
            # Use RSA signature
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            signature = self._rsa_sign_func(param_str)
            return signature.hex()
        else:
            # Fallback to HMAC
            return _sign_request(self.api_secret, params)

    def request(
        self,
        method: str,
        entity: str,
        action: str = "",
        action_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Make API request to 3Commas.
        向 3Commas 发起 API 请求。

        Args:
            method: HTTP method / HTTP 方法
            entity: API entity (bots, deals, accounts, etc.) / API 实体
            action: API action (show, update, create, etc.) / API 操作
            action_id: Optional ID for the action / 可选的操作 ID
            payload: Request payload / 请求数据
            additional_headers: Extra headers / 额外请求头

        Returns:
            (data, error) tuple - data is None if error occurred
            (数据, 错误) 元组 - 如果出错 data 为 None
        """
        # Build URL
        if entity == "bots" and action == "pairs_black_list":
            url = f"{THREECOMMAS_API_V1}/bots/{action}"
        elif action_id:
            url = f"{THREECOMMAS_API_V1}/{entity}/{action_id}/{action}" if action else f"{THREECOMMAS_API_V1}/{entity}/{action_id}"
        elif action:
            url = f"{THREECOMMAS_API_V1}/{entity}/{action}"
        else:
            url = f"{THREECOMMAS_API_V1}/{entity}"

        # Build query parameters
        params = payload or {}
        if action_id and "bot_id" not in params and "id" not in params:
            params["bot_id"] = action_id

        # Build headers
        headers = {
            "APIKEY": self.api_key,
            "Signature": self._sign(params),
            "Content-Type": "application/json"
        }
        if additional_headers:
            headers.update(additional_headers)

        # Make request with retries
        last_error = None
        for attempt in range(self.retry_count):
            data, error = _make_request(method, url, headers, payload, self.timeout)

            if error:
                status_code = error.get("status_code")
                if status_code in [429, 502]:  # Rate limit or bad gateway
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    last_error = error
                    continue
                return None, error

            return data, None

        return None, last_error or {"error": "Max retries exceeded"}

    # -------------------------------------------------------------------------
    # Bot Operations / 机器人操作
    # -------------------------------------------------------------------------

    def get_bot(self, bot_id: Union[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """获取机器人信息 / Get bot information"""
        return self.request("GET", "bots", action="show", action_id=str(bot_id))

    def get_bots(self, account_id: Optional[int] = None) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """获取机器人列表 / Get list of bots"""
        payload = {"account_id": account_id} if account_id else None
        return self.request("GET", "bots", payload=payload)

    def update_bot(
        self,
        bot_id: int,
        name: str,
        pairs: List[str],
        base_order_volume: float,
        take_profit: float,
        safety_order_volume: float,
        martingale_volume_coefficient: float,
        martingale_step_coefficient: float,
        max_safety_orders: int,
        max_active_deals: int,
        active_safety_orders_count: int,
        safety_order_step_percentage: float,
        take_profit_type: str,
        strategy_list: List[str],
        leverage_type: str,
        leverage_custom_value: float,
        **kwargs
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """更新机器人配置 / Update bot configuration"""
        payload = {
            "bot_id": bot_id,
            "name": name,
            "pairs": pairs,
            "base_order_volume": base_order_volume,
            "take_profit": take_profit,
            "safety_order_volume": safety_order_volume,
            "martingale_volume_coefficient": martingale_volume_coefficient,
            "martingale_step_coefficient": martingale_step_coefficient,
            "max_safety_orders": max_safety_orders,
            "max_active_deals": max_active_deals,
            "active_safety_orders_count": active_safety_orders_count,
            "safety_order_step_percentage": safety_order_step_percentage,
            "take_profit_type": take_profit_type,
            "strategy_list": strategy_list,
            "leverage_type": leverage_type,
            "leverage_custom_value": leverage_custom_value,
        }
        payload.update(kwargs)
        return self.request("POST", "bots", action="update", action_id=str(bot_id), payload=payload)

    def start_new_deal(
        self,
        bot_id: int,
        pair: str,
        skip_signal_checks: bool = False
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """触发机器人开始新交易 / Trigger bot to start new deal"""
        payload = {"pair": pair, "skip_signal_checks": skip_signal_checks, "bot_id": bot_id}
        return self.request("POST", "bots", action="start_new_deal", action_id=str(bot_id), payload=payload)

    def enable_bot(self, bot_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """启用机器人 / Enable bot"""
        return self.request("POST", "bots", action="enable", action_id=str(bot_id))

    def disable_bot(self, bot_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """禁用机器人 / Disable bot"""
        return self.request("POST", "bots", action="disable", action_id=str(bot_id))

    def get_pairs_blacklist(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """获取交易对黑名单 / Get pairs blacklist"""
        return self.request("GET", "bots", action="pairs_black_list")

    # -------------------------------------------------------------------------
    # Deal Operations / 交易操作
    # -------------------------------------------------------------------------

    def get_deals(
        self,
        bot_id: Optional[int] = None,
        scope: str = "finished",
        limit: int = 100
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """
        获取交易列表 / Get deals list.

        Args:
            bot_id: Filter by bot ID / 按机器人 ID 过滤
            scope: "active" or "finished" / "活跃" 或 "已完成"
            limit: Maximum number of results / 最大结果数
        """
        payload = {
            "scope": scope,
            "limit": limit,
        }
        if bot_id:
            payload["bot_id"] = str(bot_id)
        return self.request("GET", "deals", payload=payload)

    def panic_sell(self, deal_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """紧急平仓 / Panic sell deal"""
        return self.request("POST", "deals", action="panic_sell", action_id=str(deal_id))

    def add_funds(
        self,
        deal_id: int,
        quantity: float,
        rate: float,
        is_market: bool = False
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """向交易添加资金 / Add funds to deal"""
        payload = {
            "deal_id": deal_id,
            "quantity": quantity,
            "rate": rate,
            "is_market": is_market
        }
        return self.request("POST", "deals", action="add_funds", action_id=str(deal_id), payload=payload)

    def cancel_order(
        self,
        deal_id: int,
        order_id: int
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """取消订单 / Cancel order"""
        payload = {"deal_id": deal_id, "order_id": order_id}
        return self.request("POST", "deals", action="cancel_order", action_id=str(deal_id), payload=payload)

    def get_market_orders(
        self,
        deal_id: int
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """获取市价单列表 / Get market orders"""
        return self.request("GET", "deals", action="market_orders", action_id=str(deal_id))

    def get_data_for_adding_funds(
        self,
        deal_id: int
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """获取添加资金所需数据 / Get data for adding funds"""
        return self.request("GET", "deals", action="data_for_adding_funds", action_id=str(deal_id))

    # -------------------------------------------------------------------------
    # Account Operations / 账户操作
    # -------------------------------------------------------------------------

    def get_accounts(self, forced_mode: str = "real") -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """获取账户列表 / Get accounts list"""
        headers = {"Forced-Mode": forced_mode} if forced_mode else {}
        return self.request("GET", "accounts", additional_headers=headers)

    def get_account_info(self, account_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """获取账户详情 / Get account information"""
        return self.request("GET", "accounts", action="account_info", action_id=str(account_id))

    def get_account_marketcode(self, account_id: int) -> Tuple[Optional[str], Optional[Dict]]:
        """获取账户的市场代码 / Get account market code"""
        data, error = self.request("GET", "accounts", action="account_info", action_id=str(account_id))
        if data:
            return data.get("market_code"), None
        return None, error

    def get_market_pairs(self, market_code: str) -> Tuple[Optional[List[str]], Optional[Dict]]:
        """获取市场交易对列表 / Get market pairs"""
        payload = {"market_code": market_code}
        return self.request("GET", "accounts", action="market_pairs", payload=payload)

    def get_currency_rate(
        self,
        market_code: str,
        pair: str
    ) -> Tuple[Optional[float], Optional[Dict]]:
        """获取货币汇率 / Get currency rate"""
        payload = {"market_code": market_code, "pair": pair}
        data, error = self.request("GET", "accounts", action="currency_rates", payload=payload)
        if data:
            return data.get("last"), None
        return None, error

    def get_btc_usd_price(self) -> float:
        """获取 BTC/USD 价格 / Get BTC/USD price"""
        price, _ = self.get_currency_rate("binance", "USDT_BTC")
        return price if price else 20000.0


# =============================================================================
# TrailingStopLoss - Trailing Stop Loss & Take Profit
# 追踪止损 & 追踪止盈
# =============================================================================

class TrailingStopLoss:
    """
    追踪止损和追踪止盈计算器。
    Trailing Stop Loss and Take Profit Calculator.

    基于 3Commas 的追踪止损/止盈逻辑，支持根据激活价差动态调整。

    Attributes:
        initial_stoploss_percentage: 初始止损百分比 / Initial stop loss percentage
        sl_increment_factor: 止损递增因子 / Stop loss increment factor
        tp_increment_factor: 止盈递增因子 / Take profit increment factor

    Example / 示例:
        >>> tsl = TrailingStopLoss(initial_stoploss_percentage=2.0, sl_increment_factor=0.5)
        >>> sl_pct, tp_pct = tsl.calculate(deal_data, config, activation_diff=1.5)
    """

    def __init__(
        self,
        initial_stoploss_percentage: float = 0.0,
        sl_increment_factor: float = 0.0,
        tp_increment_factor: float = 0.0
    ):
        self.initial_stoploss_percentage = initial_stoploss_percentage
        self.sl_increment_factor = sl_increment_factor
        self.tp_increment_factor = tp_increment_factor

    def calculate_sl_price(
        self,
        deal_data: Dict[str, Any],
        activation_diff: float
    ) -> Tuple[float, float, float]:
        """
        计算追踪止损价格和百分比。
        Calculate trailing stop loss price and percentage.

        Args:
            deal_data: Deal data from 3Commas / 3Commas 交易数据
            activation_diff: Activation difference percentage / 激活价差百分比

        Returns:
            (current_sl_percentage, base_price_sl_percentage, understandable_sl_percentage)
            (当前止损百分比, 基于价格的止损百分比, 易理解的止损百分比)
        """
        current_sl_pct = float(deal_data.get("stop_loss_percentage") or 0.0)

        if self.initial_stoploss_percentage == 0.0:
            return current_sl_pct, 0.0, 0.0

        # Calculate average price
        strategy = deal_data.get("strategy", "long")
        if strategy == "short":
            average_price = float(deal_data.get("sold_average_price") or 0.0)
        else:
            average_price = float(deal_data.get("bought_average_price") or 0.0)

        if average_price == 0.0:
            return current_sl_pct, 0.0, 0.0

        # Calculate SL price based on average price and activation diff
        percentage_price = average_price * (
            (self.initial_stoploss_percentage / 100.0) +
            ((activation_diff / 100.0) * self.sl_increment_factor)
        )

        if strategy == "short":
            sl_price = average_price - percentage_price
        else:
            sl_price = average_price + percentage_price

        # Calculate percentage from base order price
        base_price = float(deal_data.get("base_order_average_price") or 0.0)
        if base_price == 0.0:
            return current_sl_pct, 0.0, 0.0

        if strategy == "short":
            base_price_sl_pct = ((sl_price / base_price) * 100.0) - 100.0
            understandable_sl_pct = 100.0 - ((sl_price / average_price) * 100.0)
        else:
            base_price_sl_pct = 100.0 - ((sl_price / base_price) * 100.0)
            understandable_sl_pct = ((sl_price / average_price) * 100.0) - 100.0

        return current_sl_pct, round(base_price_sl_pct, 2), round(understandable_sl_pct, 2)

    def calculate_tp_percentage(
        self,
        deal_data: Dict[str, Any],
        activation_diff: float,
        last_profit_percentage: float = 0.0
    ) -> Tuple[float, float]:
        """
        计算追踪止盈百分比。
        Calculate trailing take profit percentage.

        Args:
            deal_data: Deal data from 3Commas / 3Commas 交易数据
            activation_diff: Activation difference percentage / 激活价差百分比
            last_profit_percentage: Last profit percentage / 上次盈利百分比

        Returns:
            (current_tp_percentage, new_tp_percentage)
            (当前止盈百分比, 新止盈百分比)
        """
        current_tp_pct = float(deal_data.get("take_profit") or 0.0)

        # If close strategy is set, 3Commas manages TP
        close_strategy = deal_data.get("close_strategy_list", [])
        if len(close_strategy) > 0:
            min_profit_pct = float(deal_data.get("min_profit_percentage") or 0.0)
            return min_profit_pct, min_profit_pct

        if self.tp_increment_factor <= 0.0:
            return current_tp_pct, current_tp_pct

        new_tp_pct = current_tp_pct
        actual_profit = float(deal_data.get("actual_profit_percentage") or 0.0)

        if last_profit_percentage > 0.0:
            # Update based on profit change
            new_tp_pct = round(
                current_tp_pct + (
                    (actual_profit - last_profit_percentage) * self.tp_increment_factor
                ), 2
            )
        else:
            # Initial calculation
            new_tp_pct = round(current_tp_pct + (activation_diff * self.tp_increment_factor), 2)

        return current_tp_pct, new_tp_pct

    def calculate_safety_order(
        self,
        bot_data: Dict[str, Any],
        deal_data: Dict[str, Any],
        filled_so_count: int,
        current_profit: float
    ) -> Tuple[int, float, float, float, float]:
        """
        计算下一个安全订单。
        Calculate next safety order.

        Args:
            bot_data: Bot configuration / 机器人配置
            deal_data: Current deal data / 当前交易数据
            filled_so_count: Number of filled safety orders / 已成交安全订单数
            current_profit: Current profit percentage / 当前盈利百分比

        Returns:
            (so_buy_count, so_buy_volume, so_buy_price, total_drop_pct, next_so_drop_pct)
            (需买入计数, 需买入量, 买入价格, 总跌幅百分比, 下一安全订单跌幅百分比)
        """
        so_buy_count = 0
        so_buy_volume = 0.0
        so_buy_price = 0.0
        so_next_drop_pct = 0.0

        so_volume = 0.0
        total_volume = 0.0
        so_percentage_drop = 0.0
        total_drop_percentage = 0.0

        max_safety_orders = int(deal_data.get("max_safety_orders", 0))
        base_order_avg_price = float(deal_data.get("base_order_average_price") or 0.0)

        for counter in range(max_safety_orders):
            if counter == 0:
                next_so_volume = float(bot_data.get("safety_order_volume", 0))
                next_so_pct_drop = float(bot_data.get("safety_order_step_percentage", 0))
            else:
                next_so_volume = so_volume * float(bot_data.get("martingale_volume_coefficient", 1))
                next_so_pct_drop = so_percentage_drop * float(bot_data.get("martingale_step_coefficient", 1))

            next_total_volume = total_volume + next_so_volume
            next_total_drop_pct = total_drop_percentage + next_so_pct_drop
            next_so_buy_price = base_order_avg_price * ((100.0 - next_total_drop_pct) / 100.0)

            if counter < filled_so_count:
                # Already filled
                so_volume = next_so_volume
                total_volume = next_total_volume
                so_percentage_drop = next_so_pct_drop
                total_drop_percentage = next_total_drop_pct
                so_buy_price = next_so_buy_price
            elif next_total_drop_pct <= current_profit:
                # Not filled but required based on (negative) profit
                so_volume = next_so_volume
                total_volume = next_total_volume
                so_percentage_drop = next_so_pct_drop
                total_drop_percentage = next_total_drop_pct
                so_buy_price = next_so_buy_price
                so_buy_count += 1
                so_buy_volume += next_so_volume
            else:
                # Not filled and not required
                so_next_drop_pct = next_total_drop_pct
                break

        return so_buy_count, so_buy_volume, so_buy_price, total_drop_percentage, so_next_drop_pct


# =============================================================================
# CompoundStrategy - Profit Compounding Strategy
# 利润复利策略
# =============================================================================

class CompoundStrategy:
    """
    利润复利策略 - 将交易利润复利到基础订单和安全订单。
    Profit Compounding Strategy - Compounds deal profits into base and safety orders.

    支持两种模式：
    - "boso": 复利到基础订单和安全订单 (Base Order + Safety Order)
    - "deals": 复利到最大交易数 (Max Deals)

    Supports two modes:
    - "boso": Compound into Base Order + Safety Order volume
    - "deals": Compound into max number of active deals

    Attributes:
        profit_to_compound: 利润复利比例 (0.0-1.0) / Profit compounding ratio
        compound_mode: 复利模式 / Compounding mode

    Example / 示例:
        >>> cs = CompoundStrategy(profit_to_compound=0.5, compound_mode="boso")
        >>> new_bo, new_so = cs.calculate_compound(deal, bot_config)
    """

    # Compound modes / 复利模式
    MODE_BOSO = "boso"  # Base Order + Safety Order / 基础订单 + 安全订单
    MODE_DEALS = "deals"  # Max deals / 最大交易数
    MODE_SAFETY_ORDERS = "safetyorders"  # Safety orders only / 仅安全订单

    def __init__(
        self,
        profit_to_compound: float = 1.0,
        compound_mode: str = "boso"
    ):
        """
        Initialize CompoundStrategy.
        初始化复利策略。

        Args:
            profit_to_compound: 利润复利比例 (0.0-1.0) / Profit compounding ratio
            compound_mode: 复利模式 / Compounding mode
        """
        self.profit_to_compound = max(0.0, min(1.0, profit_to_compound))
        self.compound_mode = compound_mode

    def calculate_compound_bo_so(
        self,
        deal_data: Dict[str, Any],
        bot_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        计算复利后的基础订单和安全订单量。
        Calculate compounded base order and safety order volumes.

        Args:
            deal_data: Completed deal data / 已完成交易数据
            bot_data: Bot configuration / 机器人配置

        Returns:
            (new_base_order_volume, new_safety_order_volume)
            (新基础订单量, 新安全订单量)
        """
        final_profit = float(deal_data.get("final_profit") or 0.0)
        if final_profit <= 0:
            return 0.0, 0.0

        profit_to_use = final_profit * self.profit_to_compound

        base_order_volume = float(bot_data.get("base_order_volume", 0))
        safety_order_volume = float(bot_data.get("safety_order_volume", 0))
        max_active_deals = int(bot_data.get("max_active_deals", 1))
        max_safety_orders = int(bot_data.get("max_safety_orders", 0))
        martingale_volume_coef = float(bot_data.get("martingale_volume_coefficient", 1))
        leverage_type = bot_data.get("leverage_type", "not_specified")

        leverage = 1.0
        if leverage_type != "not_specified":
            leverage = float(bot_data.get("leverage_custom_value", 1))

        # Calculate total SO funds needed
        funds_so_needed = safety_order_volume
        total_so_funds = safety_order_volume
        if max_safety_orders > 1:
            for _ in range(1, max_safety_orders):
                funds_so_needed *= martingale_volume_coef
                total_so_funds += funds_so_needed

        # Calculate BO/SO ratio
        total_funds = base_order_volume + total_so_funds
        if total_funds == 0:
            return 0.0, 0.0

        bo_percentage = (100 * base_order_volume) / total_funds
        so_percentage = (100 * total_so_funds) / total_funds

        # Calculate compound values
        bo_profit = ((profit_to_use * bo_percentage) / 100) / max_active_deals * leverage
        bo_profit = float(_round_decimals_up(bo_profit, 8))

        if max_safety_orders >= 1:
            so_profit = bo_profit * (safety_order_volume / base_order_volume)
            so_profit = float(_round_decimals_up(so_profit, 8))
        else:
            so_profit = 0.0

        return bo_profit, so_profit

    def calculate_max_deals_increase(
        self,
        profit_sum: float,
        bot_data: Dict[str, Any]
    ) -> int:
        """
        计算可增加的最大交易数。
        Calculate maximum deals increase possible.

        Args:
            profit_sum: Total profit to compound / 用于复利的总利润
            bot_data: Bot configuration / 机器人配置

        Returns:
            Number of additional deals possible / 可增加的交易数
        """
        start_bo = float(bot_data.get("start_bo", 0))
        start_so = float(bot_data.get("start_so", 0))
        max_safety_orders = int(bot_data.get("max_safety_orders", 0))
        martingale_volume_coef = float(bot_data.get("martingale_volume_coefficient", 1))

        total_per_deal = _calculate_deal_funds(start_bo, start_so, max_safety_orders, martingale_volume_coef)

        if total_per_deal[0] <= 0:
            return 0

        profit_to_use = profit_sum * self.profit_to_compound
        additional_deals = int(profit_to_use // total_per_deal[0])

        return additional_deals


# =============================================================================
# AltRankStrategy - AltRank Score Based Pair Selection
# AltRank 排名选币策略
# =============================================================================

class AltRankStrategy:
    """
    AltRank 排名选币策略 - 基于 LunarCrush AltRank 评分选择交易对。
    AltRank Score Based Pair Selection - Selects trading pairs based on LunarCrush AltRank.

    AltRank 是 LunarCrush 提供的综合排名指标，数值越低表示排名越靠前。

    Attributes:
        max_alt_rank: 最大 AltRank 分数阈值 / Maximum AltRank score threshold
        min_volume_btc: 最小 24 小时 BTC 交易量 / Minimum 24h BTC volume

    Example / 示例:
        >>> ars = AltRankStrategy(max_alt_rank=1500, min_volume_btc=1.0)
        >>> pairs = ars.select_pairs(lunarcrush_data, ticker_list, blacklist)
    """

    def __init__(
        self,
        max_alt_rank: int = 1500,
        min_volume_btc: float = 0.0
    ):
        """
        Initialize AltRankStrategy.
        初始化 AltRank 策略。

        Args:
            max_alt_rank: 最大 AltRank 分数阈值 (数值越低排名越靠前) / Maximum AltRank score
            min_volume_btc: 最小 24 小时 BTC 交易量 / Minimum 24h BTC volume
        """
        self.max_alt_rank = max_alt_rank
        self.min_volume_btc = min_volume_btc

    def select_pairs(
        self,
        lunarcrush_data: List[Dict[str, Any]],
        ticker_list: List[str],
        blacklist: Optional[List[str]] = None,
        max_pairs: int = 10
    ) -> List[str]:
        """
        根据 AltRank 选择交易对。
        Select trading pairs based on AltRank.

        Args:
            lunarcrush_data: LunarCrush API 返回的数据 / LunarCrush API data
            ticker_list: 交易所支持的交易对列表 / Supported trading pairs on exchange
            blacklist: 黑名单交易对列表 / Blacklisted pairs
            max_pairs: 最大选择数量 / Maximum number of pairs to select

        Returns:
            选中的交易对列表 / List of selected trading pairs
        """
        blacklist = blacklist or []
        new_pairs = []
        bad_pairs = []
        black_pairs = []

        for entry in lunarcrush_data:
            if len(new_pairs) >= max_pairs:
                break

            try:
                coin = entry.get("s", "")
                acr_score = float(entry.get("acr", float("inf")))
                vol_btc = float(entry.get("volbtc", 0) or 0)

                # Check AltRank threshold
                if acr_score > self.max_alt_rank:
                    continue

                # Check volume threshold
                if vol_btc < self.min_volume_btc:
                    continue

                # Format pair (assuming BTC base)
                pair = f"BTC_{coin}"

                # Check if pair is valid
                if pair in blacklist:
                    black_pairs.append(pair)
                    continue

                if pair not in ticker_list:
                    bad_pairs.append(pair)
                    continue

                new_pairs.append(pair)

            except (KeyError, ValueError):
                continue

        return new_pairs

    def score_coins(self, lunarcrush_data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        对 coins 进行 AltRank 评分。
        Score coins by AltRank.

        Args:
            lunarcrush_data: LunarCrush API 返回的数据 / LunarCrush API data

        Returns:
            [(coin, score), ...] 按评分排序 / Sorted by score
        """
        scored = []
        for entry in lunarcrush_data:
            try:
                coin = entry.get("s", "")
                acr_score = float(entry.get("acr", float("inf")))
                scored.append((coin, acr_score))
            except (KeyError, ValueError):
                continue

        return sorted(scored, key=lambda x: x[1])


# =============================================================================
# GalaxyScoreStrategy - GalaxyScore Based Pair Selection
# GalaxyScore 评分选币策略
# =============================================================================

class GalaxyScoreStrategy:
    """
    GalaxyScore 评分选币策略 - 基于 LunarCrush GalaxyScore 评分选择交易对。
    GalaxyScore Based Pair Selection - Selects pairs based on LunarCrush GalaxyScore.

    GalaxyScore 是 LunarCrush 提供的综合评分指标，数值越高表示项目质量越好。
    数据已按 GalaxyScore 降序排列，因此遇到低于阈值的项目时可以停止处理。

    Attributes:
        min_galaxy_score: 最小 GalaxyScore 阈值 / Minimum GalaxyScore threshold
        max_alt_rank: 最大 AltRank 分数阈值 / Maximum AltRank score threshold
        min_volume_btc: 最小 24 小时 BTC 交易量 / Minimum 24h BTC volume

    Example / 示例:
        >>> gss = GalaxyScoreStrategy(min_galaxy_score=5.0, max_alt_rank=1500)
        >>> pairs = gss.select_pairs(lunarcrush_data, ticker_list, blacklist)
    """

    def __init__(
        self,
        min_galaxy_score: float = 0.0,
        max_alt_rank: int = 1500,
        min_volume_btc: float = 0.0
    ):
        """
        Initialize GalaxyScoreStrategy.
        初始化 GalaxyScore 策略。

        Args:
            min_galaxy_score: 最小 GalaxyScore 阈值 / Minimum GalaxyScore threshold
            max_alt_rank: 最大 AltRank 分数阈值 / Maximum AltRank score threshold
            min_volume_btc: 最小 24 小时 BTC 交易量 / Minimum 24h BTC volume
        """
        self.min_galaxy_score = min_galaxy_score
        self.max_alt_rank = max_alt_rank
        self.min_volume_btc = min_volume_btc

    def select_pairs(
        self,
        lunarcrush_data: List[Dict[str, Any]],
        ticker_list: List[str],
        blacklist: Optional[List[str]] = None,
        max_pairs: int = 10
    ) -> List[str]:
        """
        根据 GalaxyScore 选择交易对。
        Select trading pairs based on GalaxyScore.

        Note: LunarCrush data is sorted by GalaxyScore descending.
        注意：LunarCrush 数据按 GalaxyScore 降序排列。

        Args:
            lunarcrush_data: LunarCrush API 返回的数据 / LunarCrush API data
            ticker_list: 交易所支持的交易对列表 / Supported trading pairs on exchange
            blacklist: 黑名单交易对列表 / Blacklisted pairs
            max_pairs: 最大选择数量 / Maximum number of pairs to select

        Returns:
            选中的交易对列表 / List of selected trading pairs
        """
        blacklist = blacklist or []
        new_pairs = []
        bad_pairs = []
        black_pairs = []

        for entry in lunarcrush_data:
            # Data is sorted, so if below min score, stop processing
            try:
                galaxy_score = float(entry.get("gs", 0))
                if galaxy_score < self.min_galaxy_score:
                    # Sorted list - next coins will also be below min
                    break

                coin = entry.get("s", "")
                acr_score = float(entry.get("acr", float("inf")))
                vol_btc = float(entry.get("volbtc", 0) or 0)

                # Check AltRank threshold
                if acr_score > self.max_alt_rank:
                    continue

                # Check volume threshold
                if vol_btc < self.min_volume_btc:
                    continue

                # Format pair (assuming BTC base)
                pair = f"BTC_{coin}"

                # Check if pair is valid
                if pair in blacklist:
                    black_pairs.append(pair)
                    continue

                if pair not in ticker_list:
                    bad_pairs.append(pair)
                    continue

                new_pairs.append(pair)

                if len(new_pairs) >= max_pairs:
                    break

            except (KeyError, ValueError):
                continue

        return new_pairs

    def score_coins(self, lunarcrush_data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        对 coins 进行 GalaxyScore 评分。
        Score coins by GalaxyScore.

        Args:
            lunarcrush_data: LunarCrush API 返回的数据 / LunarCrush API data

        Returns:
            [(coin, score), ...] 按评分降序排序 / Sorted by score descending
        """
        scored = []
        for entry in lunarcrush_data:
            try:
                coin = entry.get("s", "")
                gs_score = float(entry.get("gs", 0))
                scored.append((coin, gs_score))
            except (KeyError, ValueError):
                continue

        return sorted(scored, key=lambda x: x[1], reverse=True)


# =============================================================================
# DCABotStrategy - Dollar-Cost Averaging Bot Strategy
# DCABotStrategy - 美元成本平均定投机器人策略
# =============================================================================

class DCABotStrategy:
    """
    美元成本平均 (DCA) 定投机器人策略。
    Dollar-Cost Averaging (DCA) Bot Strategy.

    该策略结合了:
    - 追踪止损/止盈 (TrailingStopLoss)
    - 利润复利 (CompoundStrategy)
    - AltRank/GalaxyScore 选币 (AltRankStrategy/GalaxyScoreStrategy)

    Attributes:
        trailing_sl: 追踪止损配置 / Trailing stop loss configuration
        compound: 复利策略配置 / Compound strategy configuration
        pair_selector: 交易对选择器 (AltRankStrategy 或 GalaxyScoreStrategy)

    Example / 示例:
        >>> dca = DCABotStrategy(
        ...     trailing_sl=TrailingStopLoss(initial_stoploss_percentage=2.0),
        ...     compound=CompoundStrategy(profit_to_compound=0.5),
        ...     pair_selector=AltRankStrategy(max_alt_rank=1500)
        ... )
        >>> new_pairs = dca.update_pairs(api, bot_id, lunarcrush_data)
    """

    def __init__(
        self,
        trailing_sl: Optional[TrailingStopLoss] = None,
        compound: Optional[CompoundStrategy] = None,
        pair_selector: Optional[Union[AltRankStrategy, GalaxyScoreStrategy]] = None
    ):
        """
        Initialize DCABotStrategy.
        初始化 DCA 机器人策略。

        Args:
            trailing_sl: 追踪止损配置 / Trailing stop loss configuration
            compound: 复利策略配置 / Compound strategy configuration
            pair_selector: 交易对选择器 / Pair selector
        """
        self.trailing_sl = trailing_sl or TrailingStopLoss()
        self.compound = compound or CompoundStrategy()
        self.pair_selector = pair_selector or AltRankStrategy()

    def get_bot_data(
        self,
        api: ThreeCommasAPI,
        bot_id: Union[int, str]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        获取机器人数据。
        Get bot data from 3Commas.

        Args:
            api: 3Commas API 客户端 / 3Commas API client
            bot_id: 机器人 ID / Bot ID

        Returns:
            (bot_data, error) / (机器人数据, 错误)
        """
        return api.get_bot(bot_id)

    def get_active_deals(
        self,
        api: ThreeCommasAPI,
        bot_id: Optional[int] = None
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """
        获取活跃交易列表。
        Get active deals.

        Args:
            api: 3Commas API 客户端 / 3Commas API client
            bot_id: 可选的机器人 ID 过滤 / Optional bot ID filter

        Returns:
            (deals_list, error) / (交易列表, 错误)
        """
        return api.get_deals(bot_id=bot_id, scope="active")

    def get_finished_deals(
        self,
        api: ThreeCommasAPI,
        bot_id: Optional[int] = None,
        limit: int = 100
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """
        获取已完成交易列表。
        Get finished deals.

        Args:
            api: 3Commas API 客户端 / 3Commas API client
            bot_id: 可选的机器人 ID 过滤 / Optional bot ID filter
            limit: 最大结果数 / Maximum results

        Returns:
            (deals_list, error) / (交易列表, 错误)
        """
        return api.get_deals(bot_id=bot_id, scope="finished", limit=limit)

    def update_pairs(
        self,
        api: ThreeCommasAPI,
        bot_id: Union[int, str],
        lunarcrush_data: List[Dict[str, Any]],
        ticker_list: List[str],
        blacklist: Optional[List[str]] = None,
        max_deals_adjustment: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        更新机器人的交易对。
        Update bot's trading pairs.

        Args:
            api: 3Commas API 客户端 / 3Commas API client
            bot_id: 机器人 ID / Bot ID
            lunarcrush_data: LunarCrush 数据 / LunarCrush data
            ticker_list: 交易所支持的交易对 / Supported trading pairs
            blacklist: 黑名单 / Blacklist
            max_deals_adjustment: 是否调整最大交易数 / Whether to adjust max deals

        Returns:
            (success, new_pairs) / (成功标志, 新交易对列表)
        """
        # Get bot data
        bot_data, error = api.get_bot(bot_id)
        if error or not bot_data:
            return False, []

        # Use pair selector to get new pairs
        new_pairs = self.pair_selector.select_pairs(
            lunarcrush_data,
            ticker_list,
            blacklist,
            max_pairs=int(bot_data.get("max_active_deals", 10))
        )

        if not new_pairs:
            return False, []

        # Sort pairs for consistent updates
        sorted_pairs = sorted(new_pairs)

        # Calculate new max deals if needed
        new_max_deals = None
        if max_deals_adjustment and len(new_pairs) < bot_data.get("max_active_deals", 10):
            new_max_deals = len(new_pairs)

        # Update bot
        payload = {
            "name": str(bot_data["name"]),
            "pairs": sorted_pairs,
            "base_order_volume": float(bot_data["base_order_volume"]),
            "take_profit": float(bot_data["take_profit"]),
            "safety_order_volume": float(bot_data["safety_order_volume"]),
            "martingale_volume_coefficient": float(bot_data["martingale_volume_coefficient"]),
            "martingale_step_coefficient": float(bot_data["martingale_step_coefficient"]),
            "max_safety_orders": int(bot_data["max_safety_orders"]),
            "max_active_deals": new_max_deals if new_max_deals else int(bot_data["max_active_deals"]),
            "active_safety_orders_count": int(bot_data["active_safety_orders_count"]),
            "safety_order_step_percentage": float(bot_data["safety_order_step_percentage"]),
            "take_profit_type": bot_data["take_profit_type"],
            "strategy_list": bot_data["strategy_list"],
            "leverage_type": bot_data["leverage_type"],
            "leverage_custom_value": float(bot_data["leverage_custom_value"]),
            "bot_id": int(bot_data["id"]),
        }

        data, error = api.request(
            "POST", "bots", action="update", action_id=str(bot_id), payload=payload
        )

        return error is None, sorted_pairs


# =============================================================================
# DealCluster - Deal Clustering Manager
# 交易聚类管理
# =============================================================================

class DealCluster:
    """
    交易聚类管理器 - 将多个机器人的交易聚合管理。
    Deal Clustering Manager - Aggregates and manages deals across multiple bots.

    主要功能：
    - 聚合多个机器人的交易
    - 识别同一币种在多个机器人上的活跃交易
    - 生成排除文件防止过度交易

    Features:
    - Aggregate deals across multiple bots
    - Identify same coin active across multiple bots
    - Generate exclude files to prevent over-trading

    Attributes:
        cluster_id: 聚类 ID / Cluster identifier
        bot_ids: 聚类中的机器人 ID 列表 / List of bot IDs in cluster
        max_same_deals: 同一币种最大活跃交易数 / Max same deals per coin

    Example / 示例:
        >>> cluster = DealCluster("cluster_1", bot_ids=[12345, 67890], max_same_deals=2)
        >>> cluster.process_deals(api)
    """

    def __init__(
        self,
        cluster_id: str,
        bot_ids: List[Union[int, str]],
        max_same_deals: int = 1
    ):
        """
        Initialize DealCluster.
        初始化交易聚类。

        Args:
            cluster_id: 聚类唯一标识符 / Unique cluster identifier
            bot_ids: 属于此聚类的机器人 ID 列表 / List of bot IDs in this cluster
            max_same_deals: 每个币种最大活跃交易数 / Maximum active deals per coin
        """
        self.cluster_id = cluster_id
        self.bot_ids = [int(bid) for bid in bot_ids]
        self.max_same_deals = max_same_deals
        self._deals_cache: Dict[int, List[Dict]] = {}

    def process_deals(
        self,
        api: ThreeCommasAPI
    ) -> Dict[str, List[str]]:
        """
        处理聚类中所有机器人的交易。
        Process deals for all bots in cluster.

        Args:
            api: 3Commas API 客户端 / 3Commas API client

        Returns:
            {
                "active_coins": [coin, ...],  # Currently active coins
                "disabled_coins": [coin, ...],  # Coins to disable
                "enabled_coins": [coin, ...],   # Coins to enable
            }
        """
        all_deals = []
        coin_active_count: Dict[str, int] = {}

        # Collect deals from all bots
        for bot_id in self.bot_ids:
            bot_data, error = api.get_bot(bot_id)
            if error or not bot_data:
                continue

            # Get active deals for this bot
            deals, _ = api.get_deals(bot_id=bot_id, scope="active")
            if deals:
                all_deals.extend(deals)

            # Cache bot data
            self._deals_cache[bot_id] = bot_data

        # Count active deals per coin
        for deal in all_deals:
            pair = deal.get("pair", "")
            if "_" in pair:
                coin = pair.split("_")[1]
                coin_active_count[coin] = coin_active_count.get(coin, 0) + 1

        # Determine which coins to disable
        disabled_coins = [
            coin for coin, count in coin_active_count.items()
            if count >= self.max_same_deals
        ]

        enabled_coins = [
            coin for coin in coin_active_count.keys()
            if coin not in disabled_coins
        ]

        return {
            "active_coins": list(coin_active_count.keys()),
            "disabled_coins": disabled_coins,
            "enabled_coins": enabled_coins,
            "coin_counts": coin_active_count
        }

    def get_excluded_pairs(self) -> List[str]:
        """
        获取应该被排除的交易对列表。
        Get list of pairs that should be excluded.

        Returns:
            需要排除的交易对列表 / List of pairs to exclude
        """
        result = self.process_deals(ThreeCommasAPI.__new__(ThreeCommasAPI))  # Placeholder
        return result.get("disabled_coins", [])

    def calculate_cluster_score(self) -> float:
        """
        计算聚类健康度评分。
        Calculate cluster health score.

        Returns:
            0.0-1.0 的评分 / Score between 0.0 and 1.0
        """
        if not _NUMPY_AVAILABLE:
            return 0.5

        total_deals = sum(len(deals) for deals in self._deals_cache.values())
        unique_coins = len(set(
            deal.get("pair", "").split("_")[1]
            for deals in self._deals_cache.values()
            for deal in deals
            if "_" in deal.get("pair", "")
        ))

        if total_deals == 0:
            return 1.0

        # Lower concentration = higher score
        concentration = unique_coins / total_deals if total_deals > 0 else 1.0
        return min(1.0, concentration)


# =============================================================================
# MarketCollector - Market Data Collector
# 市场数据采集器
# =============================================================================

class MarketCollector:
    """
    市场数据采集器 - 从多个数据源采集市场数据。
    Market Data Collector - Collects market data from multiple sources.

    支持的数据源：
    - LunarCrush (AltRank, GalaxyScore)
    - CoinGecko (价格, 交易量, 变化率)
    - CoinMarketCap (价格, 排名)

    Supported data sources:
    - LunarCrush (AltRank, GalaxyScore)
    - CoinGecko (price, volume, change rates)
    - CoinMarketCap (price, rank)

    Attributes:
        api_keys: 各数据源的 API 密钥 / API keys for data sources
        timeout: 请求超时 / Request timeout

    Example / 示例:
        >>> mc = MarketCollector(lunarcrush_key="your_key")
        >>> data = mc.fetch_lunarcrush(limit=150)
    """

    def __init__(
        self,
        lunarcrush_key: Optional[str] = None,
        coingecko_key: Optional[str] = None,
        coinmarketcap_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize MarketCollector.
        初始化市场数据采集器。

        Args:
            lunarcrush_key: LunarCrush API 密钥 / LunarCrush API key
            coingecko_key: CoinGecko API 密钥 / CoinGecko API key
            coinmarketcap_key: CoinMarketCap API 密钥 / CoinMarketCap API key
            timeout: 请求超时（秒）/ Request timeout in seconds
        """
        self.api_keys = {
            "lunarcrush": lunarcrush_key,
            "coingecko": coingecko_key,
            "coinmarketcap": coinmarketcap_key
        }
        self.timeout = timeout
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def _get_cached(self, key: str, max_age: float = 300) -> Optional[Any]:
        """从缓存获取数据（如果未过期）/ Get cached data if not expired"""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < max_age:
                return data
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """设置缓存 / Set cache"""
        self._cache[key] = (time.time(), data)

    def fetch_lunarcrush(
        self,
        list_type: str = "altrank",
        limit: int = 150,
        symbol: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        从 LunarCrush 获取数据。
        Fetch data from LunarCrush.

        Args:
            list_type: 列表类型 ("altrank" 或 "galaxyscore") / List type
            limit: 返回结果数量限制 / Result limit
            symbol: 可选的单一币种符号过滤 / Optional single symbol filter

        Returns:
            LunarCrush 数据列表或 None / List of LunarCrush data or None
        """
        cache_key = f"lunarcrush_{list_type}_{limit}_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        api_key = self.api_keys.get("lunarcrush", "")
        if not api_key:
            return None

        params = {"key": api_key, "limit": limit}
        if symbol:
            params["symbol"] = symbol

        url = f"{LUNARCRUSH_API_BASE}/{'galaxyscore' if list_type == 'galaxyscore' else 'altrank'}"
        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        headers = {"Content-Type": "application/json"}
        data, error = _make_request("GET", full_url, headers, timeout=self.timeout)

        if error:
            return None

        # Parse response - LunarCrush returns data in 'data' field
        result = data.get("data", []) if isinstance(data, dict) else []

        self._set_cache(cache_key, result)
        return result

    def fetch_coingecko(
        self,
        start: int = 1,
        end: int = 200,
        vs_currency: str = "btc"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        从 CoinGecko 获取数据。
        Fetch data from CoinGecko.

        Args:
            start: 起始排名 / Starting rank
            end: 结束排名 / Ending rank
            vs_currency: 比较货币 / Comparison currency

        Returns:
            CoinGecko 数据列表或 None / List of CoinGecko data or None
        """
        cache_key = f"coingecko_{start}_{end}_{vs_currency}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        headers = {}
        api_key = self.api_keys.get("coingecko", "")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        # CoinGecko /coins/markets endpoint
        params = {
            "vs_currency": vs_currency.lower(),
            "order": "market_cap_desc",
            "per_page": end - start + 1,
            "page": (start // 250) + 1,
            "sparkline": "false"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{COINGECKO_API_BASE}/coins/markets?{query_string}"

        data, error = _make_request("GET", url, headers, timeout=self.timeout)

        if error:
            return None

        result = data if isinstance(data, list) else []
        self._set_cache(cache_key, result)
        return result

    def fetch_coinmarketcap(
        self,
        start: int = 1,
        limit: int = 200,
        convert: str = "BTC"
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict]]:
        """
        从 CoinMarketCap 获取数据。
        Fetch data from CoinMarketCap.

        Args:
            start: 起始排名 / Starting rank
            limit: 返回数量 / Number of results
            convert: 转换货币 / Conversion currency

        Returns:
            (data, error) / (数据, 错误)
        """
        api_key = self.api_keys.get("coinmarketcap", "")
        if not api_key:
            return None, {"error": "No API key"}

        headers = {
            "Accept": "application/json",
            "X-CMC_PRO_API_KEY": api_key
        }

        params = {
            "start": start,
            "limit": limit,
            "convert": convert.upper()
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{COINMARKETCAP_API_BASE}/cryptocurrency/listings/latest?{query_string}"

        data, error = _make_request("GET", url, headers, timeout=self.timeout)

        if error:
            return None, error

        if isinstance(data, dict) and data.get("status", {}).get("error_code"):
            return None, data.get("status", {})

        result = data.get("data", []) if isinstance(data, dict) else []
        return result, None

    def aggregate_rankings(
        self,
        rankings_data: Dict[str, List[Dict[str, Any]]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        聚合多个排名数据源，计算综合评分。
        Aggregate multiple ranking data sources with weighted scoring.

        Args:
            rankings_data: 各数据源的排名数据 / Rankings from different sources
                {
                    "altrank": [{"s": "BTC", "acr": 1}, ...],
                    "galaxyscore": [{"s": "BTC", "gs": 9.5}, ...],
                    "coinmarketcap": [{"symbol": "BTC", "cmc_rank": 1}, ...]
                }
            weights: 各数据源权重 / Weights for each source

        Returns:
            [(coin, aggregated_score), ...] 按评分降序 / Sorted by score descending
        """
        weights = weights or {
            "altrank": 0.4,
            "galaxyscore": 0.4,
            "coinmarketcap": 0.2
        }

        # Collect all coins
        all_coins: Dict[str, Dict[str, float]] = {}

        # Process AltRank
        if "altrank" in rankings_data and _NUMPY_AVAILABLE:
            altrank_scores = []
            coins = []
            for entry in rankings_data["altrank"]:
                coin = entry.get("s", "")
                score = float(entry.get("acr", float("inf")))
                coins.append(coin)
                altrank_scores.append(score)

            if altrank_scores:
                normalized = _numpy_score_normalize(altrank_scores, inverse=True)
                for i, coin in enumerate(coins):
                    if coin not in all_coins:
                        all_coins[coin] = {}
                    all_coins[coin]["altrank"] = normalized[i]

        # Process GalaxyScore
        if "galaxyscore" in rankings_data and _NUMPY_AVAILABLE:
            gs_scores = []
            coins = []
            for entry in rankings_data["galaxyscore"]:
                coin = entry.get("s", "")
                score = float(entry.get("gs", 0))
                coins.append(coin)
                gs_scores.append(score)

            if gs_scores:
                normalized = _numpy_score_normalize(gs_scores, inverse=False)
                for i, coin in enumerate(coins):
                    if coin not in all_coins:
                        all_coins[coin] = {}
                    all_coins[coin]["galaxyscore"] = normalized[i]

        # Calculate aggregated scores
        results = []
        for coin, scores in all_coins.items():
            total_score = 0.0
            total_weight = 0.0

            for source, weight in weights.items():
                if source in scores:
                    total_score += scores[source] * weight
                    total_weight += weight

            if total_weight > 0:
                final_score = total_score / total_weight
                results.append((coin, final_score))

        return sorted(results, key=lambda x: x[1], reverse=True)


# =============================================================================
# Exports / 导出
# =============================================================================

__all__ = [
    # Core classes / 核心类
    "ThreeCommasAPI",
    "TrailingStopLoss",
    "CompoundStrategy",
    "AltRankStrategy",
    "GalaxyScoreStrategy",
    "DCABotStrategy",
    "DealCluster",
    "MarketCollector",

    # Exceptions / 异常类
    "ThreeCommasAPIError",
    "APIRateLimitError",
    "APIDataError",

    # Utility functions / 工具函数
    "_sign_request",
    "_round_decimals_up",
    "_calculate_deal_funds",
    "_numpy_score_normalize",
]

# =============================================================================
# Module Metadata / 模块元信息
# =============================================================================

__version__ = "1.0.0"
__author__ = "Claude Code"
__source__ = "cyberjunky/3commas-cyber-bots"
__license__ = "MIT"
