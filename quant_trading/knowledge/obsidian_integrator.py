"""
Obsidian Integrator - Obsidian集成模块
负责将数据同步到Obsidian Vault
"""

import os
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ObsidianIntegrator")


class ObsidianIntegrator:
    """Obsidian集成器"""

    def __init__(self, vault_path: str, config: Optional[Dict[str, str]] = None) -> None:
        """
        初始化Obsidian集成器

        Args:
            vault_path: Obsidian保险库路径
            config: 配置选项
        """
        self.vault_path = vault_path
        self.config = config or {}

        # 默认配置
        self.data_folder = self.config.get("dataFolder", "30_资源/量化数据")
        self.logs_folder = self.config.get("logsFolder", "30_资源/量化日志")
        self.reports_folder = self.config.get("reportsFolder", "10_项目/12_量化系统/监控报告")
        self.monitor_file = self.config.get("monitorFile", "策略监控中心.md")

        # 构建完整路径
        self.paths: Dict[str, str] = {
            "data": os.path.join(vault_path, self.data_folder),
            "logs": os.path.join(vault_path, self.logs_folder),
            "reports": os.path.join(vault_path, self.reports_folder),
            "monitor": os.path.join(vault_path, self.monitor_file)
        }

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """确保所有必要的目录存在"""
        for dir_path in [self.paths["data"], self.paths["logs"], self.paths["reports"]]:
            os.makedirs(dir_path, exist_ok=True)

        # 确保监控文件的父目录存在
        monitor_dir = os.path.dirname(self.paths["monitor"])
        os.makedirs(monitor_dir, exist_ok=True)

        logger.info("Obsidian目录结构已准备就绪")

    def sync_csv_to_obsidian(self, source_path: str, filename: str) -> bool:
        """
        复制CSV文件到Obsidian

        Args:
            source_path: 源文件路径
            filename: 目标文件名

        Returns:
            是否成功
        """
        try:
            dest_path = os.path.join(self.paths["data"], filename)
            shutil.copy2(source_path, dest_path)
            logger.info(f"已同步到Obsidian: {filename}")
            return True
        except IOError as e:
            logger.error(f"同步CSV失败: {filename}, 错误: {e}")
            return False

    def update_monitor_panel(self, market_data: Dict[str, Any], sentiment_data: List[Dict[str, Any]]) -> None:
        """
        更新Obsidian监控面板

        Args:
            market_data: 市场数据
            sentiment_data: 情绪分析数据
        """
        try:
            content = self._generate_monitor_content(market_data, sentiment_data)
            with open(self.paths["monitor"], 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"监控面板已更新: {self.paths['monitor']}")
        except IOError as e:
            logger.error(f"更新监控面板失败: {e}")

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

    def _generate_monitor_content(self, market_data: Dict[str, Any], sentiment_data: List[Dict]) -> str:
        """生成监控面板内容"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        content = f"# 📈 量化交易实时监控\n\n"
        content += f"> **最后更新**: {date_str} {time_str}\n"
        content += "> **系统状态**: 🟢 正常运行\n\n"

        # 1. 市场行情部分
        content += "---\n\n## 📊 核心行情监控\n\n"

        if market_data and len(market_data) > 0:
            for symbol, data in market_data.items():
                content += f"### {symbol}\n\n"
                content += "| 指标 | 数值 |\n"
                content += "|:---|:---|\n"
                content += f"| 💰 当前价格 | {data.get('last', 'N/A')} |\n"
                content += f"| 📈 24h最高 | {data.get('high', 'N/A')} |\n"
                content += f"| 📉 24h最低 | {data.get('low', 'N/A')} |\n"
                percentage = data.get('percentage', 0)
                content += f"| 📊 24h涨跌幅 | {percentage:.2f}% |\n"
                volume = data.get('volume', 0)
                content += f"| 📦 24h成交量 | {volume:.2f} |\n"
                content += f"| ⏰ 更新时间 | {data.get('datetime', 'N/A')} |\n\n"

        # 2. 情绪分析部分
        content += "---\n\n## 🧠 市场情绪雷达\n\n"

        if sentiment_data and len(sentiment_data) > 0:
            content += "| 来源 | 关键词 | 情绪评分 | 等级 | 分析理由 | 时间 |\n"
            content += "|:---|:---|:---:|:---:|:---|:---|\n"

            for item in sentiment_data:
                emoji = self._get_sentiment_emoji(item.get('score', 0))
                level = self._get_sentiment_level(item.get('score', 0))
                keywords = ','.join(item.get('keywords', []))
                reason = item.get('reason', '')
                short_reason = reason[:20] + '...' if len(reason) > 20 else reason

                content += f"| {item.get('source', 'N/A')} | {keywords} | {emoji} {item.get('score', 0):.2f} | {level} | {short_reason} | {item.get('datetime', 'N/A')} |\n"
        else:
            content += "> 暂无情绪分析数据\n\n"

        # 3. 数据文件链接
        content += "---\n\n## 📁 数据文件\n\n"
        content += "### 历史行情数据\n\n"
        content += "- [[BTC_USDT_行情]]\n"
        content += "- [[ETH_USDT_行情]]\n"
        content += "- [[情绪分析记录]]\n\n"

        # 4. 快速操作
        content += "---\n\n## 🚀 快速操作\n\n"
        content += "### 监控系统命令\n\n"
        content += "```bash\n"
        content += "# 启动监控\n"
        content += "cd D:\\量化交易\n"
        content += "npm start\n\n"
        content += "# 停止监控\n"
        content += "# 按 Ctrl+C\n"
        content += "```\n\n"

        # 5. 待办事项
        content += "---\n\n## 📋 今日待办\n\n"
        content += "- [ ] 检查市场情绪变化\n"
        content += "- [ ] 查看价格预警\n"
        content += "- [ ] 更新交易策略\n"
        content += "- [ ] 回顾交易记录\n\n"

        # 6. 系统统计
        content += "---\n\n## 📈 系统统计\n\n"
        content += f"- **今日抓取次数**: {{today_fetch_count}}\n"
        content += f"- **情绪分析次数**: {{today_analysis_count}}\n"
        content += f"- **运行时长**: {{uptime}}\n\n"

        # 7. 相关资源
        content += "---\n\n## 🔗 相关资源\n\n"
        content += "- [[情绪分析Prompt]]\n"
        content += "- [[交易策略文档]]\n"
        content += "- [[风险管理规则]]\n\n"

        content += f"---\n\n*本页面由量化监控系统自动生成 | {date_str} {time_str}*\n"

        return content

    def generate_daily_report(self, summary: Dict[str, Any]) -> Optional[str]:
        """
        生成每日报告

        Args:
            summary: 当日数据汇总

        Returns:
            文件路径，失败返回None
        """
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}_监控报告.md"
            filepath = os.path.join(self.paths["reports"], filename)

            content = f"# 📊 {date_str} 量化交易日报\n\n"

            # BTC
            btc = summary.get("btc", {})
            content += "## 📈 行情回顾\n\n"
            content += "### BTC/USDT\n"
            content += f"- 开盘: {btc.get('open', 'N/A')}\n"
            content += f"- 收盘: {btc.get('close', 'N/A')}\n"
            content += f"- 最高: {btc.get('high', 'N/A')}\n"
            content += f"- 最低: {btc.get('low', 'N/A')}\n"
            content += f"- 涨跌幅: {btc.get('change', 'N/A')}%\n\n"

            # ETH
            eth = summary.get("eth", {})
            content += "### ETH/USDT\n"
            content += f"- 开盘: {eth.get('open', 'N/A')}\n"
            content += f"- 收盘: {eth.get('close', 'N/A')}\n"
            content += f"- 最高: {eth.get('high', 'N/A')}\n"
            content += f"- 最低: {eth.get('low', 'N/A')}\n"
            content += f"- 涨跌幅: {eth.get('change', 'N/A')}%\n\n"

            # 情绪分析
            content += "## 🧠 情绪分析\n\n"
            content += f"- 平均情绪分: {summary.get('avgSentiment', 'N/A')}\n"
            keywords = summary.get('topKeywords', [])
            content += f"- 主要关键词: {', '.join(keywords)}\n\n"

            # 交易统计
            content += "## 📊 交易统计\n\n"
            content += f"- 监控次数: {summary.get('fetchCount', 0)}\n"
            content += f"- 分析次数: {summary.get('analysisCount', 0)}\n\n"

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"每日报告已生成: {filename}")
            return filepath

        except IOError as e:
            logger.error(f"生成每日报告失败: {e}")
            return None

    def sync_log_to_obsidian(self, log_content: str) -> None:
        """
        同步日志文件到Obsidian

        Args:
            log_content: 日志内容
        """
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"system_{date_str}.md"
            filepath = os.path.join(self.paths["logs"], filename)

            content = f"# 📋 系统日志 - {date_str}\n\n"
            content += "```\n"
            content += log_content
            content += "\n```\n"

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"日志已同步: {filename}")

        except IOError as e:
            logger.error(f"同步日志失败: {e}")

    def get_file_list(self, folder: str = "") -> List[str]:
        """
        获取Obsidian中的文件列表

        Args:
            folder: 文件夹路径（相对于vault）

        Returns:
            文件列表
        """
        target_path = os.path.join(self.vault_path, folder) if folder else self.vault_path

        if not os.path.exists(target_path):
            return []

        try:
            files = os.listdir(target_path)
            return [f for f in files if os.path.isfile(os.path.join(target_path, f))]
        except OSError:
            return []

    def create_attachment(self, content: str, filename: str, folder: str = "") -> Optional[str]:
        """
        在Obsidian中创建附件

        Args:
            content: 文件内容
            filename: 文件名
            folder: 目标文件夹

        Returns:
            文件路径，失败返回None
        """
        try:
            if folder:
                target_path = os.path.join(self.vault_path, folder, filename)
                os.makedirs(os.path.join(self.vault_path, folder), exist_ok=True)
            else:
                target_path = os.path.join(self.vault_path, filename)

            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"附件已创建: {filename}")
            return target_path

        except IOError as e:
            logger.error(f"创建附件失败: {filename}, 错误: {e}")
            return None


# 导出主要类
__all__ = ["ObsidianIntegrator"]
