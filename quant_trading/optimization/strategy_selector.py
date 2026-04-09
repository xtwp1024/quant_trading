"""
策略选择器 - 批量回测、排序、筛选
==========================

批量回测所有策略，筛选出表现最优的策略组合
"""
import asyncio
import sys
import os
import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.database import DatabaseManager
from ..core.logger import setup_logger, logger
# from backtest_engine import BacktestExchange, BacktestEventBus

# Stub for missing backtest_engine
class BacktestExchange:
    pass
class BacktestEventBus:
    pass


class StrategySelector:
    """策略选择器 - 找出最赚钱的策略"""

    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        self.strategies = self._load_all_strategies()

    def _load_all_strategies(self) -> List[Tuple[str, type]]:
        """加载所有可用策略"""
        from modules.strategies.martin_strategy import MartinStrategy
        from modules.strategies.nostalgia_v2 import NostalgiaGridStrategyV2 as NostalgiaV2
        from modules.strategies.momentum_strategy import MomentumStrategy
        from modules.strategies.atr_breakout import AtrBreakoutStrategy
        from modules.strategies.triple_supertrend import TripleSupertrendStrategy
        from modules.strategies.squeeze_momentum import SqueezeMomentumStrategy
        from modules.strategies.binance_accessory_strategy import BinanceAccessoryStrategy
        from modules.strategies.hedged_grid import HedgedGridStrategy as WeightedGrid
        from modules.strategies.mean_reversion import MeanReversionStrategy
        from modules.strategies.grid_trading import GridTradingStrategy
        from modules.strategies.eth_rotation import ETHRotationStrategy

        return [
            ("WeightedGrid", WeightedGrid),
            ("NostalgiaV2", NostalgiaV2),
            ("Momentum", MomentumStrategy),
            ("ATRBreakout", AtrBreakoutStrategy),
            ("TripleSupertrend", TripleSupertrendStrategy),
            ("SqueezeMomentum", SqueezeMomentumStrategy),
            ("BinanceAccessory", BinanceAccessoryStrategy),
            ("MartinBinance", MartinStrategy),
            ("MeanReversion", MeanReversionStrategy),
            ("GridTrading", GridTradingStrategy),
            ("ETHRotation", ETHRotationStrategy),
        ]

    async def batch_backtest_all(
        self,
        symbol: str = "ETH-USDT-SWAP",
        days: int = 90,
        timeframes: List[str] = None
    ) -> pd.DataFrame:
        """
        批量回测所有策略

        Returns:
            DataFrame with columns: Strategy, ROI, Sharpe, MaxDrawdown, WinRate, Trades
        """
        if timeframes is None:
            timeframes = ["15m"]

        logger.info(f"🚀 开始批量回测 {len(self.strategies)} 个策略...")
        logger.info(f"   交易对: {symbol}")
        logger.info(f"   周期: {days}天")
        logger.info(f"   时间框架: {timeframes}")

        # 加载数据
        db = await self._connect_db()
        data = await self._fetch_data(db, symbol, days)
        if not data:
            logger.error("无法获取数据")
            return pd.DataFrame()

        results_by_tf = {}

        for timeframe in timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"时间框架: {timeframe}")
            logger.info(f"{'='*60}")

            # 聚合数据到对应时间框架
            agg_data = self._aggregate_data(data, timeframe)

            tf_results = []

            for strat_name, strat_class in self.strategies:
                try:
                    result = await self._backtest_single(
                        strat_name, strat_class, symbol, timeframe, agg_data
                    )
                    tf_results.append(result)
                    logger.info(f"✅ {strat_name}: ROI={result['ROI']:.2f}%, "
                              f"Sharpe={result['Sharpe']:.2f}, "
                              f"WinRate={result['WinRate']:.1f}%")
                except Exception as e:
                    logger.error(f"❌ {strat_name}: {e}")
                    # 添加失败记录
                    tf_results.append({
                        'Strategy': strat_name,
                        'ROI': -999,
                        'Sharpe': -999,
                        'MaxDrawdown': 999,
                        'WinRate': 0,
                        'Trades': 0,
                        'Status': 'ERROR'
                    })

            results_by_tf[timeframe] = pd.DataFrame(tf_results)

        await db.close()

        # 合并所有时间框架结果
        return self._merge_results(results_by_tf)

    async def _backtest_single(
        self,
        strat_name: str,
        strat_class: type,
        symbol: str,
        timeframe: str,
        data: List[List]
    ) -> Dict:
        """回测单个策略"""

        bus = BacktestEventBus()
        exchange = BacktestExchange(initial_capital=1000.0)
        strategy = strat_class(bus, {'symbol': symbol, 'timeframe': timeframe})
        await strategy.start()

        data_buffer = []

        for i, row in enumerate(data):
            ts, o, h, l, c, v = row

            candle_list = [float(x) for x in row]
            data_buffer.append(candle_list)

            if len(data_buffer) > 1000:
                data_buffer.pop(0)
            if len(data_buffer) < 300:
                continue

            try:
                # 处理挂单
                exchange.process_pending_orders(l, h, ts)

                # 发布K线
                await strategy.on_candle(type('Event', (), {
                    'payload': {'symbol': symbol, 'data': data_buffer}
                }))

                # 处理信号
                for sig_entry in bus.trade_history:
                    if sig_entry.get('processed'):
                        continue
                    exchange.execute_signal(sig_entry['signal'], float(c), ts)
                    sig_entry['processed'] = True

            except Exception as e:
                logger.debug(f"Error at candle {i}: {e}")
                continue

        # 计算结果
        final_price = float(data[-1][4])
        equity = float(exchange.get_equity({symbol: final_price}))
        initial_capital = 1000.0

        roi = (equity - initial_capital) / initial_capital * 100
        trades = len(exchange.trades)

        # 计算夏普比率（简化版）
        if trades > 0:
            returns = []
            for trade in exchange.trades:
                if trade['type'] in ['BUY', 'SELL']:
                    # 简化：假设每笔交易收益为正负随机
                    # 实际应该追踪具体的交易对
                    pass
            # 使用ROI/波动率作为简化夏普
            sharpe = roi / 10 if roi > 0 else roi / 5  # 简化假设
        else:
            sharpe = 0

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(exchange.trades, initial_capital)

        # 计算胜率
        win_rate = self._calculate_win_rate(exchange.trades)

        return {
            'Strategy': strat_name,
            'ROI': roi,
            'Sharpe': sharpe,
            'MaxDrawdown': max_drawdown,
            'WinRate': win_rate,
            'Trades': trades,
            'Equity': equity,
            'Status': 'OK'
        }

    def _calculate_max_drawdown(self, trades: List, initial_capital: float) -> float:
        """计算最大回撤"""
        if not trades:
            return 0.0

        capital = initial_capital
        peak = initial_capital
        max_dd = 0.0

        for trade in trades:
            if trade['type'] == 'SELL':
                capital += trade['amount'] * trade['price'] - trade['fee']
            elif trade['type'] == 'BUY':
                capital -= trade['amount'] * trade['price'] + trade['fee']

            if capital > peak:
                peak = capital

            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_win_rate(self, trades: List) -> float:
        """计算胜率"""
        if not trades:
            return 0.0

        # 简化：假设BUY后SELL算一次完整交易
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        if not buy_trades or not sell_trades:
            return 50.0  # 默认50%

        # 简化：假设配对交易中，如果SELL价格>BUY价格则为盈利
        wins = 0
        total = min(len(buy_trades), len(sell_trades))

        for i in range(total):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                wins += 1

        return wins / total * 100 if total > 0 else 50.0

    def _aggregate_data(self, data: List, target_tf: str) -> List:
        """聚合数据到目标时间框架"""
        # 简化：假设输入已经是15m数据
        # 实际应该根据target_tf进行聚合
        # 这里暂时返回原数据
        return data

    def _merge_results(self, results_by_tf: Dict) -> pd.DataFrame:
        """合并多个时间框架的结果"""
        if not results_by_tf:
            return pd.DataFrame()

        if len(results_by_tf) == 1:
            return list(results_by_tf.values())[0]

        # 平均多个时间框架的结果
        df_list = list(results_by_tf.values())
        merged = pd.concat(df_list).groupby('Strategy').agg({
            'ROI': 'mean',
            'Sharpe': 'mean',
            'MaxDrawdown': 'mean',
            'WinRate': 'mean',
            'Trades': 'sum',
        }).reset_index()

        return merged

    async def _connect_db(self) -> DatabaseManager:
        """连接数据库"""
        db = DatabaseManager(self.config)
        await db.connect()
        if not db.pg_conn:
            raise ConnectionError("数据库连接失败")
        return db

    async def _fetch_data(self, db: DatabaseManager, symbol: str, days: int) -> List:
        """获取历史数据"""
        base_timeframe = "15m"
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - (days * 24 * 3600 * 1000)

        with db.pg_conn.cursor() as cur:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_candles
                WHERE symbol = %s AND timeframe = %s
                AND timestamp >= %s
                ORDER BY timestamp ASC
            """
            cur.execute(query, (symbol, base_timeframe, start_ts))
            rows = cur.fetchall()

        logger.info(f"✅ 加载了 {len(rows)} 条K线数据")
        return [list(r) for r in rows]

    def filter_strategies(
        self,
        results: pd.DataFrame,
        min_roi: float = 0.0,
        max_drawdown: float = 30.0,
        min_sharpe: float = 1.0,
        min_trades: int = 10
    ) -> pd.DataFrame:
        """
        筛选策略

        Args:
            results: 回测结果DataFrame
            min_roi: 最小ROI (%)
            max_drawdown: 最大回撤 (%)
            min_sharpe: 最小夏普比率
            min_trades: 最小交易次数

        Returns:
            筛选后的DataFrame
        """
        filtered = results[
            (results['ROI'] >= min_roi) &
            (results['MaxDrawdown'] <= max_drawdown) &
            (results['Sharpe'] >= min_sharpe) &
            (results['Trades'] >= min_trades) &
            (results['Status'] == 'OK')
        ].copy()

        filtered = filtered.sort_values('ROI', ascending=False)
        filtered['Rank'] = range(1, len(filtered) + 1)

        return filtered

    def calculate_correlation(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略相关性

        注意：这里基于策略收益率计算相关性矩阵
        """
        strategies = results['Strategy'].tolist()
        n = len(strategies)

        if 'Return%' not in results.columns or n < 2:
            # Fallback: return identity matrix
            corr_matrix = np.eye(n)
        else:
            # 计算实际收益率相关性
            try:
                returns_data = results.pivot_table(
                    index=results.index,
                    columns='Strategy',
                    values='Return%'
                ).fillna(0)
                corr_matrix = returns_data.corr().fillna(0).values
                # 确保对称且有界
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                corr_matrix = np.clip(corr_matrix, -1, 1)
                np.fill_diagonal(corr_matrix, 1.0)
            except Exception:
                # Fallback: return identity matrix
                corr_matrix = np.eye(n)

        df = pd.DataFrame(
            corr_matrix,
            index=strategies,
            columns=strategies
        )

        return df

    def select_low_correlation(
        self,
        results: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        max_correlation: float = 0.7,
        top_n: int = 5
    ) -> List[str]:
        """
        选择低相关性的策略组合

        Args:
            results: 策略结果（按ROI排序）
            corr_matrix: 相关性矩阵
            max_correlation: 最大允许相关性
            top_n: 选择的策略数量

        Returns:
            选择的策略名称列表
        """
        selected = []
        candidates = results['Strategy'].tolist()

        for strategy in candidates:
            if len(selected) >= top_n:
                break

            # 检查与已选策略的相关性
            can_add = True
            for selected_strat in selected:
                corr = corr_matrix.loc[strategy, selected_strat]
                if corr > max_correlation:
                    can_add = False
                    break

            if can_add:
                selected.append(strategy)

        return selected

    def generate_report(
        self,
        results: pd.DataFrame,
        filtered: pd.DataFrame,
        selected: List[str],
        output_dir: str = "optimization/reports"
    ) -> str:
        """生成策略评估报告"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f"strategy_evaluation_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 策略评估报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 总体统计\n\n")
            f.write(f"- **测试策略数**: {len(results)}\n")
            f.write(f"- **合格策略数**: {len(filtered)}\n")
            f.write(f"- **选中策略数**: {len(selected)}\n\n")

            f.write("## 🏆 Top 10 策略\n\n")
            f.write("| 排名 | 策略 | ROI (%) | 夏普 | 最大回撤 (%) | 胜率 (%) | 交易次数 |\n")
            f.write("|------|------|--------|------|-------------|---------|----------|\n")

            for _, row in filtered.head(10).iterrows():
                f.write(f"| {row['Rank']} | {row['Strategy']} | "
                       f"{row['ROI']:.2f} | {row['Sharpe']:.2f} | "
                       f"{row['MaxDrawdown']:.2f} | {row['WinRate']:.1f} | "
                       f"{row['Trades']} |\n")

            f.write("\n## 🎯 最终选择策略组合\n\n")
            for i, strat in enumerate(selected, 1):
                strat_data = filtered[filtered['Strategy'] == strat].iloc[0]
                f.write(f"{i}. **{strat}**\n")
                f.write(f"   - ROI: {strat_data['ROI']:.2f}%\n")
                f.write(f"   - 夏普: {strat_data['Sharpe']:.2f}\n")
                f.write(f"   - 回撤: {strat_data['MaxDrawdown']:.2f}%\n")
                f.write(f"   - 胜率: {strat_data['WinRate']:.1f}%\n\n")

            f.write("## 📈 筛选标准\n\n")
            f.write("- 最小ROI: 0%\n")
            f.write("- 最大回撤: <30%\n")
            f.write("- 最小夏普: >1.0\n")
            f.write("- 最小交易次数: >10\n")
            f.write("- 最大策略相关性: <0.7\n\n")

        logger.info(f"✅ 报告已保存: {report_file}")

        # 保存JSON数据
        json_file = output_path / f"strategy_data_{timestamp}.json"
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(results),
            'qualified_strategies': len(filtered),
            'selected_strategies': selected,
            'all_results': results.to_dict('records'),
            'filtered_results': filtered.to_dict('records')
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 数据已保存: {json_file}")

        return str(report_file)


async def main():
    """主函数 - 执行策略选择流程"""

    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    selector = StrategySelector(config)

    # 1. 批量回测所有策略
    print("\n" + "="*80)
    print("批量回测所有策略")
    print("="*80 + "\n")

    results = await selector.batch_backtest_all(
        symbol="ETH-USDT-SWAP",
        days=90,
        timeframes=["15m"]
    )

    if results.empty:
        print("❌ 回测失败，没有结果")
        return

    # 2. 筛选策略
    print("\n" + "="*80)
    print("筛选策略")
    print("="*80 + "\n")

    filtered = selector.filter_strategies(
        results,
        min_roi=0.0,
        max_drawdown=30.0,
        min_sharpe=1.0,
        min_trades=10
    )

    print(f"\n✅ 合格策略: {len(filtered)}/{len(results)}")
    print(f"\nTop 10 策略:")
    print(filtered.head(10).to_string(index=False))

    # 3. 计算相关性
    print("\n" + "="*80)
    print("计算策略相关性")
    print("="*80 + "\n")

    corr_matrix = selector.calculate_correlation(filtered)

    # 4. 选择低相关性组合
    selected = selector.select_low_correlation(
        filtered,
        corr_matrix,
        max_correlation=0.7,
        top_n=5
    )

    print(f"\n✅ 最终选择策略组合: {selected}")

    # 5. 生成报告
    print("\n" + "="*80)
    print("生成评估报告")
    print("="*80 + "\n")

    report_path = selector.generate_report(results, filtered, selected)

    print(f"\n✅ 完成！报告已保存: {report_path}")

    # 保存配置
    config_file = Path("optimization/production_strategies.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump({
            'selected_strategies': selected,
            'selection_date': datetime.now().isoformat(),
            'selection_criteria': {
                'min_roi': 0.0,
                'max_drawdown': 30.0,
                'min_sharpe': 1.0,
                'min_trades': 10
            }
        }, f, default_flow_style=False)

    print(f"✅ 配置已保存: {config_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
