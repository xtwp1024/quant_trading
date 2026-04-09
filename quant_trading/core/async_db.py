# -*- coding: utf-8 -*-
"""
异步数据库连接池 - Async Database Connection Pool

使用 asyncpg 创建连接池，解决单连接的性能瓶颈。
"""

import asyncio
import asyncpg
import os
from typing import Optional
from .logger import logger


class AsyncDatabasePool:
    """异步数据库连接池"""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        """
        初始化数据库连接池

        Args:
            dsn: 数据库连接字符串
            min_size: 最小连接数
            max_size: 最大连接数
        """
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """初始化连接池"""
        try:
            logger.info("🔄 初始化数据库连接池...")

            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
                max_queries=50000  # 最大并发查询数
            )

            # 测试连接
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')

            logger.info(f"✅ 数据库连接池初始化成功 (min={self.min_size}, max={self.max_size})")

        except Exception as e:
            logger.error(f"❌ 数据库连接池初始化失败: {e}")
            raise

    async def close(self) -> None:
        """关闭所有连接"""
        if self.pool:
            await self.pool.close()
            logger.info("数据库连接池已关闭")

    async def execute_query(self, query: str, args: tuple = ()) -> list:
        """
        执行查询（使用连接池中的连接）

        Args:
            query: SQL 查询语句 (使用 $1, $2 等占位符)
            args: 查询参数元组

        Returns:
            查询结果列表
        """
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        async with self.pool.acquire() as conn:
            try:
                # HIGH: 使用参数化查询防止SQL注入，asyncpg自动处理转义
                rows = await conn.fetch(query, *args)
                return rows
            except Exception as e:
                logger.error(f"SQL执行失败: {query} Error: {e}")
                raise

    async def execute_many(self, query: str, args_list: list) -> None:
        """
        批量执行查询（使用事务）

        Args:
            query: SQL 查询语句
            args_list: 参数列表
        """
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(query, args_list)

    async def get_connection(self):
        """
        获取原始连接（用于兼容旧代码）

        DEPRECATED: 建议使用 execute_query
        """
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        return await self.pool.acquire()

    @property
    def is_initialized(self) -> bool:
        """检查连接池是否已初始化"""
        return self.pool is not None

    @property
    def size(self) -> int:
        """当前连接数"""
        if not self.pool:
            return 0
        return self.pool.get_size()


# 全局连接池实例
_global_pool: Optional[AsyncDatabasePool] = None


def get_global_pool() -> AsyncDatabasePool:
    """
    获取全局连接池实例（单例模式）
    """
    global _global_pool
    if _global_pool is None:
        dsn = os.getenv('POSTGRES_URL')
        if not dsn:
            raise RuntimeError("POSTGRES_URL 环境变量未设置")

        # HIGH: 连接池大小从环境变量读取，禁止硬编码
        min_size = int(os.getenv('DB_POOL_MIN_SIZE', '2'))
        max_size = int(os.getenv('DB_POOL_MAX_SIZE', '10'))
        _global_pool = AsyncDatabasePool(dsn, min_size=min_size, max_size=max_size)
        logger.info(f"📊 创建全局数据库连接池 (min={min_size}, max={max_size})")

    return _global_pool


async def close_global_pool() -> None:
    """关闭全局连接池"""
    global _global_pool
    if _global_pool:
        await _global_pool.close()
        _global_pool = None
        logger.info("全局数据库连接池已关闭")
