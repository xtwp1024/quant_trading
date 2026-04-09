#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import A-Share Full Data - A股全量数据导入

功能:
- 获取全量A股列表 (约5000+只)
- 批量获取历史K线数据
- 支持并发加速
- 保存到本地数据库
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, project_root)

import akshare as ak

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ImportAShare")


class AShareImporter:
    """A股全量数据导入器"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or os.path.join(project_root, "data", "ashare"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._stock_list: List[Dict[str, str]] = []
        self._cache_file = self.data_dir / "stock_list.json"
        self._data_cache_file = self.data_dir / "data_cache.json"

    def get_stock_list(self, force_update: bool = False) -> List[Dict[str, str]]:
        """
        获取全量A股列表

        Args:
            force_update: 是否强制更新缓存

        Returns:
            [{"symbol": "000001", "name": "平安银行"}, ...]
        """
        # 尝试从缓存加载
        if not force_update and self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cached_count = len(data)
                    logger.info(f"从缓存加载股票列表: {cached_count}只")
                    self._stock_list = data
                    return data
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")

        # 从akshare获取
        logger.info("正在从akshare获取全量A股列表...")
        try:
            df = ak.stock_info_a_code_name()
            self._stock_list = [
                {"symbol": str(row["code"]).zfill(6), "name": str(row["name"])}
                for _, row in df.iterrows()
            ]

            # 过滤ST股票和异常股票
            self._stock_list = [
                s for s in self._stock_list
                if not s["name"].startswith("*ST")
                and not s["name"].startswith("ST")
                and not s["name"].endswith("退")
                and len(s["symbol"]) == 6
            ]

            logger.info(f"获取到 {len(self._stock_list)} 只A股 (已过滤ST/退市)")

            # 保存到缓存
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._stock_list, f, ensure_ascii=False)

            return self._stock_list

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return self._stock_list if self._stock_list else []

    def fetch_single_stock(
        self,
        symbol: str,
        days: int = 250,
        adjust: str = "qfq"
    ) -> Optional[Dict[str, Any]]:
        """
        获取单只股票的历史K线数据

        Returns:
            {"symbol": "000001", "ohlcv": [...], "volume_ratio": 1.5} or None
        """
        try:
            # 使用已有的AkshareMarketDataClient
            from quant_trading.data.providers import AkshareMarketDataClient

            client = AkshareMarketDataClient(adjust=adjust)
            df = client.fetch(symbol, period_type="1d")

            if df is None or df.empty or len(df) < 20:
                return None

            # 转换为OHLCV数组
            rows = []
            for _, row in df.iterrows():
                try:
                    o = float(row.get("开盘", row.get("open", 0)))
                    h = float(row.get("最高", row.get("high", 0)))
                    l = float(row.get("最低", row.get("low", 0)))
                    c = float(row.get("收盘", row.get("close", 0)))
                    v = float(row.get("成交量", row.get("volume", 0)))
                    if o > 0 and h > 0 and l > 0 and c > 0 and v >= 0:
                        rows.append([o, h, l, c, v])
                except (ValueError, TypeError):
                    continue

            if len(rows) < 20:
                return None

            ohlcv = np.array(rows)

            # 计算量比
            volumes = [r[4] for r in rows]
            if len(volumes) >= 6:
                avg_5day = sum(volumes[-6:-1]) / 5
                today_vol = volumes[-1]
                volume_ratio = today_vol / avg_5day if avg_5day > 0 else 1.0
            else:
                volume_ratio = 1.0

            return {
                "symbol": symbol,
                "ohlcv": ohlcv.tolist(),
                "volume_ratio": volume_ratio,
                "count": len(rows),
                "latest_price": rows[-1][3] if rows else 0
            }

        except Exception as e:
            logger.debug(f"获取 {symbol} 失败: {e}")
            return None

    def fetch_batch(
        self,
        symbols: List[str],
        days: int = 250,
        max_workers: int = 10,
        progress_callback=None
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多只股票数据 (并发)

        Args:
            symbols: 股票代码列表
            days: 历史数据天数
            max_workers: 并发数
            progress_callback: 进度回调 (current, total)

        Returns:
            {symbol: {"ohlcv": [...], "volume_ratio": 1.5}, ...}
        """
        results = {}
        total = len(symbols)
        completed = 0
        failed = 0

        logger.info(f"开始批量获取 {total} 只股票数据 (并发数={max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_single_stock, symbol, days): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    result = future.result()
                    if result:
                        results[symbol] = result
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

                if completed % 100 == 0 or completed == total:
                    logger.info(f"进度: {completed}/{total} ({failed}只失败)")
                    if progress_callback:
                        progress_callback(completed, total)

        logger.info(f"批量获取完成: 成功{len(results)}只, 失败{failed}只")
        return results

    def import_all(
        self,
        max_stocks: int = None,
        max_workers: int = 10,
        save_interval: int = 500
    ) -> Dict[str, Any]:
        """
        全量导入所有A股数据

        Args:
            max_stocks: 最大导入股票数 (None=全部)
            max_workers: 并发数
            save_interval: 每多少只保存一次中间结果

        Returns:
            导入统计
        """
        # 获取股票列表
        stock_list = self.get_stock_list()
        if not stock_list:
            logger.error("股票列表为空")
            return {"success": 0, "failed": 0}

        if max_stocks:
            stock_list = stock_list[:max_stocks]

        symbols = [s["symbol"] for s in stock_list]
        symbols_with_name = {s["symbol"]: s["name"] for s in stock_list}

        logger.info(f"准备导入 {len(symbols)} 只股票...")

        # 分批导入
        all_results = {}
        batch_num = 0

        for i in range(0, len(symbols), save_interval):
            batch = symbols[i:i + save_interval]
            batch_num += 1

            logger.info(f"处理第 {batch_num} 批 ({len(batch)} 只)")
            batch_results = self.fetch_batch(
                batch,
                max_workers=max_workers
            )

            # 合并结果
            for symbol, data in batch_results.items():
                data["name"] = symbols_with_name.get(symbol, symbol)
                all_results[symbol] = data

            # 保存中间结果
            cache_file = self.data_dir / f"cache_batch_{batch_num}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)

            logger.info(f"第 {batch_num} 批完成, 累计成功 {len(all_results)} 只")

            # 避免请求过快
            if i + save_interval < len(symbols):
                time.sleep(1)

        # 保存最终结果
        final_cache = self.data_dir / "all_stocks_data.json"
        logger.info(f"保存最终结果到 {final_cache}...")

        with open(final_cache, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)

        # 保存索引文件
        index_data = {
            "update_time": datetime.now().isoformat(),
            "total_count": len(all_results),
            "stocks": [
                {"symbol": s["symbol"], "name": s["name"]}
                for s in stock_list if s["symbol"] in all_results
            ]
        }
        index_file = self.data_dir / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        logger.info(f"全量导入完成: 成功 {len(all_results)}/{len(symbols)} 只")

        return {
            "success": len(all_results),
            "failed": len(symbols) - len(all_results),
            "save_file": str(final_cache),
            "index_file": str(index_file)
        }

    def load_cached_data(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存的数据"""
        cache_file = self.data_dir / "all_stocks_data.json"
        if not cache_file.exists():
            logger.warning(f"缓存文件不存在: {cache_file}")
            return {}

        logger.info(f"从缓存加载数据: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"加载了 {len(data)} 只股票的数据")
        return data


def main():
    """入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="A股全量数据导入")
    parser.add_argument("--max-stocks", type=int, help="最大导入股票数 (默认全部)")
    parser.add_argument("--workers", type=int, default=10, help="并发数 (默认10)")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--load-cache", action="store_true", help="仅加载缓存数据")
    parser.add_argument("--update-list", action="store_true", help="更新股票列表缓存")

    args = parser.parse_args()

    importer = AShareImporter(data_dir=args.data_dir)

    if args.load_cache:
        # 仅加载缓存
        data = importer.load_cached_data()
        print(f"加载了 {len(data)} 只股票的数据")
        return

    if args.update_list:
        # 仅更新股票列表
        stocks = importer.get_stock_list(force_update=True)
        print(f"更新了 {len(stocks)} 只股票的列表")
        return

    # 全量导入
    logger.info("=" * 60)
    logger.info("A股全量数据导入")
    logger.info("=" * 60)

    result = importer.import_all(
        max_stocks=args.max_stocks,
        max_workers=args.workers
    )

    print()
    print("=" * 60)
    print("导入完成")
    print(f"成功: {result['success']} 只")
    print(f"失败: {result['failed']} 只")
    print(f"数据文件: {result['save_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
