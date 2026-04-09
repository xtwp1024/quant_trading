# -*- coding: utf-8 -*-
"""
配置验证模块 - Configuration Validator

确保所有必需的配置项存在且有效，在系统启动时验证。
"""

import os
import sys
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """配置验证器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors = []
        self.warnings = []

    def validate_all(self) -> ValidationResult:
        """执行所有验证"""
        self._validate_required_keys()
        self._validate_trading_config()
        self._validate_risk_config()
        self._validate_environment_variables()
        self._validate_data_types()

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings
        )

    def _validate_required_keys(self) -> None:
        """验证必需的配置键"""
        required_keys = ['symbol', 'strategy', 'system']

        for key in required_keys:
            if key not in self.config:
                self.errors.append(f"缺少必需配置项: '{key}'")
            elif self.config[key] is None:
                self.errors.append(f"配置项 '{key}' 值为空")

    def _validate_trading_config(self) -> None:
        """验证交易配置"""
        if 'strategy' not in self.config:
            return

        strategy = self.config['strategy']

        # 验证交易对
        if 'symbol' not in self.config:
            self.errors.append("缺少交易对配置: 'symbol'")
        else:
            symbol = self.config['symbol']
            if '/' not in symbol:
                self.errors.append(f"交易对格式错误: '{symbol}'，应为 'BASE/QUOTE'")

        # 验证杠杆配置
        if 'risk' not in self.config:
            self.errors.append("缺少风险配置: 'risk'")
        else:
            risk = self.config['risk']
            self._validate_leverage(risk)

        # 验证网格配置
        if 'grid_levels' in strategy:
            grid_levels = strategy['grid_levels']
            if not isinstance(grid_levels, int) or grid_levels < 2:
                self.errors.append(f"网格层数必须 ≥ 2: {grid_levels}")
            if grid_levels > 20:
                self.warnings.append(f"网格层数过多({grid_levels})，可能影响性能")

        # 验证止盈止损
        for key in ['take_profit', 'stop_loss']:
            if key in strategy:
                value = strategy[key]
                try:
                    pct = Decimal(str(value))
                    if pct < 0 or pct > Decimal('0.5'):
                        self.errors.append(f"{key} 必须在 0-50% 之间: {value}")
                except (ValueError, InvalidOperation):
                    self.errors.append(f"{key} 格式错误: {value}")

    def _validate_leverage(self, risk: Dict) -> None:
        """验证杠杆配置"""
        if 'max_leverage' not in risk:
            return

        max_lev = risk['max_leverage']
        if not isinstance(max_lev, (int, float)) or max_lev < 1:
            self.errors.append(f"最大杠杆必须是正数: {max_lev}")
        if max_lev > 125:
            self.errors.append(f"最大杠杆超出安全范围(125): {max_lev}")

        # 警告高杠杆
        if max_lev > 20:
            self.warnings.append(f"高杠杆警告({max_lev}x)，建议 ≤ 20x")

    def _validate_risk_config(self) -> None:
        """验证风险控制配置"""
        if 'risk' not in self.config:
            return

        risk = self.config['risk']

        # 验证最大日亏损
        if 'max_daily_loss' in risk:
            value = risk['max_daily_loss']
            try:
                pct = Decimal(str(value))
                if pct < 0 or pct > Decimal('0.2'):
                    self.errors.append(f"最大日亏损必须在 0-20% 之间: {value}")
            except (ValueError, InvalidOperation):
                self.errors.append(f"最大日亏损格式错误: {value}")

        # 验证最大仓位
        if 'max_position_size' in risk:
            value = risk['max_position_size']
            if not isinstance(value, (int, float)) or value <= 0:
                self.errors.append(f"最大仓位必须是正数: {value}")

    def _validate_environment_variables(self) -> None:
        """验证环境变量"""
        required_env_vars = ['API_KEY', 'API_SECRET', 'API_PASSWORD']

        # 检查交易所 API 密钥
        exchange_keys = ['API_KEY', 'API_SECRET', 'GATE_API_KEY', 'GATE_API_SECRET']
        has_exchange_key = any(os.getenv(k) for k in exchange_keys)
        if not has_exchange_key:
            self.warnings.append("未设置交易所 API 密钥，可能使用模拟模式")

        # 检查数据库密码
        if not os.getenv('DB_PASS'):
            self.warnings.append("未设置 DB_PASS，使用默认值")

    def _validate_data_types(self) -> None:
        """验证数据类型"""
        # 验证 symbol 类型
        if 'symbol' in self.config:
            symbol = self.config['symbol']
            if not isinstance(symbol, str):
                self.errors.append(f"交易对必须是字符串: {type(symbol)}")

        # 验证 timeframe 类型
        if 'timeframe' in self.config:
            timeframe = self.config['timeframe']
            valid_timeframes = ['1m', '3m', '5m', '15m', '1h', '4h', '1d']
            if timeframe not in valid_timeframes:
                self.errors.append(f"无效的时间周期: {timeframe}")

    def print_report(self, result: ValidationResult) -> None:
        """打印验证报告"""
        print("\n" + "="*60)
        print("🔍 配置验证报告")
        print("="*60)

        if result.is_valid:
            print("✅ 所有配置项验证通过")
        else:
            print("❌ 发现配置错误:")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")

        if result.warnings:
            print("\n⚠️  配置警告:")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")

        print("="*60)

        if not result.is_valid:
            print("\n❌ 配置验证失败，程序退出")
            sys.exit(1)


def validate_and_start(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证配置并启动系统

    Args:
        config: 配置字典

    Returns:
        验证通过的配置字典

    Raises:
        SystemExit: 如果配置验证失败
    """
    validator = ConfigValidator(config)
    result = validator.validate_all()
    validator.print_report(result)

    if not result.is_valid:
        sys.exit(1)

    # 如果有警告，记录到日志
    if result.warnings:
        from core.logger import logger
        logger.warning(f"配置验证通过，但有 {len(result.warnings)} 个警告")
        logger.info("系统启动配置验证完成")

    return config
