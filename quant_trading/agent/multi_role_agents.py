"""Multi-Role Agent Factory.

多角色Agent工厂 — 创建6分析师 + 2研究员 + 3风控 + 4交易员 架构.

根据角色类型创建专业的辩论智能体:
- 6分析师: 基本面、技术面、情绪、量化、宏观、行业
- 2研究员: 策略研究员、另类数据研究员
- 3风控: 风险经理、仓位风控、流动性风控
- 4交易员: 日内、趋势、套利、做市

Architecture:
    Role enum -> MultiRoleAgentFactory.create() -> DebateAgent subclass
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Callable

from .debate_engine import DebateAgent, BullAgent, BearAgent, Stance

__all__ = [
    "Role",
    "MultiRoleAgentFactory",
]


class Role(str, Enum):
    """Agent角色枚举 / Agent role enumeration.

    6+2+3+4 架构:
    - 6分析师: 基本面、技术面、情绪、量化、宏观、行业
    - 2研究员: 策略研究员、另类数据研究员
    - 3风控: 风险经理、仓位风控、流动性风控
    - 4交易员: 日内、趋势、套利、做市
    """

    # 分析师 (6)
    FUNDAMENTAL_ANALYST = "fundamental_analyst"  # 基本面分析师
    TECHNICAL_ANALYST = "technical_analyst"  # 技术分析师
    SENTIMENT_ANALYST = "sentiment_analyst"  # 情绪分析师
    QUANTITATIVE_ANALYST = "quantitative_analyst"  # 量化分析师
    MACRO_ANALYST = "macro_analyst"  # 宏观分析师
    INDUSTRY_ANALYST = "industry_analyst"  # 行业分析师

    # 研究员 (2)
    STRATEGY_RESEARCHER = "strategy_researcher"  # 策略研究员
    ALTERNATIVE_DATA_RESEARCHER = "alternative_data_researcher"  # 另类数据研究员

    # 风控 (3)
    RISK_MANAGER = "risk_manager"  # 风险经理
    POSITION_RISK_MANAGER = "position_risk_manager"  # 仓位风控
    LIQUIDITY_RISK_MANAGER = "liquidity_risk_manager"  # 流动性风控

    # 交易员 (4)
    INTRADAY_TRADER = "intraday_trader"  # 日内交易员
    TREND_TRADER = "trend_trader"  # 趋势交易员
    ARBITRAGE_TRADER = "arbitrage_trader"  # 套利交易员
    MARKET_MAKER = "market_maker"  # 做市商

    # 扩展角色
    NEWS_ANALYST = "news_analyst"  # 新闻分析师
    OPTIONS_ANALYST = "options_analyst"  # 期权分析师


class MultiRoleAgentFactory:
    """多角色Agent工厂 / Multi-role agent factory.

    根据角色创建专业Agent，支持bull/bear双立场.

    Example:
        >>> factory = MultiRoleAgentFactory()
        >>> analyst = factory.create(Role.FUNDAMENTAL_ANALYST, stance=Stance.BULL)
        >>> print(analyst.agent_id, analyst.role, analyst.stance)
    """

    _ROLE_PROMPTS = {
        Role.FUNDAMENTAL_ANALYST: "基本面分析师 — 评估公司财务、估值、盈利预期",
        Role.TECHNICAL_ANALYST: "技术分析师 — 评估价格形态、成交量、技术指标",
        Role.SENTIMENT_ANALYST: "情绪分析师 — 评估市场情绪、投资者行为、资金流向",
        Role.QUANTITATIVE_ANALYST: "量化分析师 — 评估数学模型、统计信号、算法策略",
        Role.MACRO_ANALYST: "宏观分析师 — 评估经济周期、政策走向、全球宏观因素",
        Role.INDUSTRY_ANALYST: "行业分析师 — 评估行业景气度、竞争格局、产业链",
        Role.STRATEGY_RESEARCHER: "策略研究员 — 研究alpha来源、因子暴露、策略逻辑",
        Role.ALTERNATIVE_DATA_RESEARCHER: "另类数据研究员 — 研究卫星、信用卡、社交媒体等数据",
        Role.RISK_MANAGER: "风险经理 — 评估整体风险敞口、风险预算、风险收益比",
        Role.POSITION_RISK_MANAGER: "仓位风控 — 评估仓位大小、集中度、杠杆率",
        Role.LIQUIDITY_RISK_MANAGER: "流动性风控 — 评估市场深度、冲击成本、流动性风险",
        Role.INTRADAY_TRADER: "日内交易员 — 专注短期价格波动、日内买卖点",
        Role.TREND_TRADER: "趋势交易员 — 专注中短期趋势、动量策略",
        Role.ARBITRAGE_TRADER: "套利交易员 — 专注价差交易、跨市场套利",
        Role.MARKET_MAKER: "做市商 — 专注买卖价差、库存管理、订单簿",
        Role.NEWS_ANALYST: "新闻分析师 — 分析突发新闻、事件驱动机会",
        Role.OPTIONS_ANALYST: "期权分析师 — 分析波动率、希腊字母、期权结构",
    }

    @staticmethod
    def create(
        role: Role,
        llm: Optional[Callable[..., str]] = None,
        stance: str = Stance.NEUTRAL,
    ) -> DebateAgent:
        """根据角色创建专业Agent / Create professional agent by role.

        Args:
            role: 角色枚举
            llm: LLM调用函数 (optional)
            stance: 立场 — bull/bear/neutral (default neutral)

        Returns:
            DebateAgent: 配置好的辩论智能体
        """
        agent_id = f"{role.value}-{uuid_short()}"
        agent = _RoleDebateAgent(
            agent_id=agent_id,
            role=role.value,
            stance=stance,
            role_description=MultiRoleAgentFactory._ROLE_PROMPTS.get(
                role, role.value
            ),
        )
        if llm:
            agent.set_llm(llm)
        return agent

    @staticmethod
    def create_research_team() -> list[DebateAgent]:
        """创建6分析师团队 / Create 6-analyst research team.

        Returns:
            list[DebateAgent]: 包含6种分析师的团队
        """
        roles = [
            Role.FUNDAMENTAL_ANALYST,
            Role.TECHNICAL_ANALYST,
            Role.SENTIMENT_ANALYST,
            Role.QUANTITATIVE_ANALYST,
            Role.MACRO_ANALYST,
            Role.INDUSTRY_ANALYST,
        ]
        return [MultiRoleAgentFactory.create(role) for role in roles]

    @staticmethod
    def create_research_team_with_researchers() -> list[DebateAgent]:
        """创建6分析师 + 2研究员 团队 / Create 6-analyst + 2-researcher team.

        Returns:
            list[DebateAgent]: 8人研究团队
        """
        team = MultiRoleAgentFactory.create_research_team()
        team.append(MultiRoleAgentFactory.create(Role.STRATEGY_RESEARCHER))
        team.append(MultiRoleAgentFactory.create(Role.ALTERNATIVE_DATA_RESEARCHER))
        return team

    @staticmethod
    def create_risk_team() -> list[DebateAgent]:
        """创建3风控团队 / Create 3-risk-manager team.

        Returns:
            list[DebateAgent]: 包含3种风控角色的团队
        """
        roles = [
            Role.RISK_MANAGER,
            Role.POSITION_RISK_MANAGER,
            Role.LIQUIDITY_RISK_MANAGER,
        ]
        return [MultiRoleAgentFactory.create(role) for role in roles]

    @staticmethod
    def create_trading_team() -> list[DebateAgent]:
        """创建4交易员团队 / Create 4-trader team.

        Returns:
            list[DebateAgent]: 包含4种交易员的团队
        """
        roles = [
            Role.INTRADAY_TRADER,
            Role.TREND_TRADER,
            Role.ARBITRAGE_TRADER,
            Role.MARKET_MAKER,
        ]
        return [MultiRoleAgentFactory.create(role) for role in roles]

    @staticmethod
    def create_full_team() -> list[DebateAgent]:
        """创建完整15人团队 (6分析师 + 2研究员 + 3风控 + 4交易员) / Create full 15-agent team.

        Returns:
            list[DebateAgent]: 15人完整团队
        """
        team: list[DebateAgent] = []
        team.extend(MultiRoleAgentFactory.create_research_team())
        team.append(MultiRoleAgentFactory.create(Role.STRATEGY_RESEARCHER))
        team.append(MultiRoleAgentFactory.create(Role.ALTERNATIVE_DATA_RESEARCHER))
        team.extend(MultiRoleAgentFactory.create_risk_team())
        team.extend(MultiRoleAgentFactory.create_trading_team())
        return team

    @staticmethod
    def create_bull_team(roles: list[Role]) -> list[DebateAgent]:
        """创建多头阵营 / Create bull team.

        Args:
            roles: 角色列表

        Returns:
            list[DebateAgent]: 均为bull立场的Agent列表
        """
        return [
            MultiRoleAgentFactory.create(role, stance=Stance.BULL)
            for role in roles
        ]

    @staticmethod
    def create_bear_team(roles: list[Role]) -> list[DebateAgent]:
        """创建空头阵营 / Create bear team.

        Args:
            roles: 角色列表

        Returns:
            list[DebateAgent]: 均为bear立场的Agent列表
        """
        return [
            MultiRoleAgentFactory.create(role, stance=Stance.BEAR)
            for role in roles
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def uuid_short() -> str:
    """Generate short UUID for agent IDs."""
    import uuid
    return uuid.uuid4().hex[:8]


class _RoleDebateAgent(DebateAgent):
    """Role-specific debate agent with extended prompt context.

    带有角色特定提示词上下文的辩论智能体.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        stance: str,
        role_description: str = "",
    ):
        super().__init__(agent_id=agent_id, role=role, stance=stance)
        self.role_description = role_description

    def _build_analysis_prompt(self, topic: str, context: str) -> str:
        return f"""You are a {self.role_description} with a {self.stance} stance.

角色: {self.role}
立场: {self.stance}

辩论主题: {topic}

背景数据:
{context}

请以结构化方式提出你的核心论点，格式如下:
---
content: <你的核心论点，简洁有力，聚焦于{self.role}>
evidence: [<证据1>, <证据2>, ...]
confidence: <0到1之间的置信度>
---

只返回格式化的论点，不要额外解释。"""

    def _build_rebuttal_prompt(self, opposing_text: str) -> str:
        return f"""You are a {self.role_description} with a {self.stance} stance.

角色: {self.role}
立场: {self.stance}

对方的论点:
{opposing_text}

请针对对方论点提出有力的反驳:
---
content: <你的反驳论点>
evidence: [<反驳证据1>, <反驳证据2>, ...]
confidence: <0到1之间的置信度>
---

只返回格式化的论点，不要额外解释。"""
