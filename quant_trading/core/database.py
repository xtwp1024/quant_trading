# -*- coding: utf-8 -*-
"""
数据库管理器 - Database Manager with Async Connection Pool

使用异步连接池提升数据库性能，解决单连接瓶颈问题。
"""

import asyncio
import redis.asyncio as redis
from typing import Optional, List, Dict, Any
from decimal import Decimal
from .async_db import AsyncDatabasePool
from .logger import logger


class DatabaseManager:
    """
    异步数据库管理器

    Features:
    - 使用 asyncpg 连接池提升性能
    - PostgreSQL + Redis 双存储
    - 自动初始化表结构
    - 事务保护
    """

    def __init__(self, config):
        """
        初始化数据库管理器

        Args:
            config: 配置字典，需包含 database.postgres_url 和 database.redis_url
        """
        self.pg_url = config['database']['postgres_url']
        self.redis_url = config['database']['redis_url']
        self.redis = None
        self.db_pool: Optional[AsyncDatabasePool] = None

        # 从环境变量读取连接池配置（可选）
        import os
        self.pool_min_size = int(os.getenv('DB_POOL_MIN', '2'))
        self.pool_max_size = int(os.getenv('DB_POOL_MAX', '10'))

    async def connect(self):
        """
        连接数据库（PostgreSQL 连接池 + Redis）

        Raises:
            RuntimeError: 如果连接池初始化失败
        """
        try:
            # 1. 初始化 PostgreSQL 连接池
            logger.info("🔄 初始化 PostgreSQL 连接池...")
            self.db_pool = AsyncDatabasePool(
                dsn=self.pg_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size
            )
            await self.db_pool.initialize()
            logger.info(f"✅ PostgreSQL 连接池已创建 (min={self.pool_min_size}, max={self.pool_max_size})")

            # 2. 连接 Redis
            try:
                logger.info("🔄 连接 Redis...")
                self.redis = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=2.0)
                await self.redis.ping()
                logger.info("✅ Redis 已连接")
            except Exception as e:
                logger.warning(f"⚠️ Redis 连接失败: {e} (已降级运行，缓存功能不可用)")
                self.redis = None

            # 3. 初始化表结构
            await self._init_tables()

        except Exception as e:
            logger.critical(f"❌ 数据库连接失败: {e}")
            raise RuntimeError(f"数据库连接失败: {e}")

    async def close(self):
        """关闭所有数据库连接"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("✅ PostgreSQL 连接池已关闭")

        if self.redis:
            await self.redis.close()
            logger.info("✅ Redis 已关闭")

    async def _init_tables(self):
        """初始化数据库表结构"""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(50),
                symbol VARCHAR(20),
                side VARCHAR(4),
                price DECIMAL,
                amount DECIMAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS fees (
                id SERIAL PRIMARY KEY,
                trade_id INTEGER,
                fee_amount DECIMAL,
                fee_asset VARCHAR(10),
                fee_type VARCHAR(10),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS leverage_logs (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                old_leverage INTEGER,
                new_leverage INTEGER,
                trigger VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS risks (
                id SERIAL PRIMARY KEY,
                type VARCHAR(50),
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS market_candles (
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(5) NOT NULL,
                timestamp BIGINT NOT NULL,
                open DECIMAL NOT NULL,
                high DECIMAL NOT NULL,
                low DECIMAL NOT NULL,
                close DECIMAL NOT NULL,
                volume DECIMAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS market_trades (
                id VARCHAR(50) PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL NOT NULL,
                amount DECIMAL NOT NULL,
                side VARCHAR(4) NOT NULL,
                timestamp BIGINT NOT NULL
            );
            """
        ]

        for query in queries:
            await self.db_pool.execute_query(query)

        logger.info("✅ 数据库表结构已初始化")

    async def record_trade(self, strategy: str, symbol: str, side: str,
                       price: Decimal, amount: Decimal,
                       fee_amount: Optional[Decimal] = None,
                       fee_asset: Optional[str] = None,
                       fee_type: Optional[str] = None) -> int:
        """
        记录交易订单（含手续费，事务保护）

        Args:
            strategy: 策略名称
            symbol: 交易对
            side: BUY/SELL
            price: 成交价格
            amount: 成交数量
            fee_amount: 手续费金额（可选）
            fee_asset: 手续费币种（可选）
            fee_type: 手续费类型 MAKER/TAKER（可选）

        Returns:
            int: 交易记录 ID

        Raises:
            Exception: 数据库操作失败时回滚事务
        """
        # 使用 Decimal 转 str 保证精度
        price_str = str(price)
        amount_str = str(amount)

        query = """
            INSERT INTO trades (strategy, symbol, side, price, amount)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """

        try:
            # 使用连接池执行查询（asyncpg 使用 $1, $2 参数占位符）
            result = await self.db_pool.execute_query(query, strategy, symbol, side, price_str, amount_str)

            if result and len(result) > 0:
                trade_id = result[0][0]  # asyncpg 返回 (id,)
            else:
                raise Exception("未能获取交易记录 ID")

            # 记录手续费
            if fee_amount is not None and fee_asset is not None:
                fee_query = """
                    INSERT INTO fees (trade_id, fee_amount, fee_asset, fee_type)
                    VALUES ($1, $2, $3, $4)
                """
                await self.db_pool.execute_query(
                    fee_query,
                    trade_id,
                    str(fee_amount),
                    fee_asset,
                    fee_type or 'MAKER'
                )

            logger.info(f"✅ 交易已记录: ID={trade_id}, {strategy} {symbol} {side} @{price}")
            return trade_id

        except Exception as e:
            logger.error(f"❌ 记录交易失败: {e}")
            raise

    async def save_candle(self, symbol: str, timeframe: str, timestamp: int,
                       open_p: Decimal, high_p: Decimal, low_p: Decimal,
                       close_p: Decimal, volume: Decimal) -> None:
        """
        保存 K 线数据（UPSERT）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            timestamp: Unix 时间戳（毫秒）
            open_p, high_p, low_p, close_p: OHLC 价格
            volume: 成交量
        """
        query = """
            INSERT INTO market_candles (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        await self.db_pool.execute_query(
            query,
            symbol,
            timeframe,
            timestamp,
            str(open_p),
            str(high_p),
            str(low_p),
            str(close_p),
            str(volume)
        )

    async def save_trade(self, trade_id: str, symbol: str, price: Decimal,
                     amount: Decimal, side: str, timestamp: int) -> None:
        """
        保存公共交易数据（交易所实时成交）

        Args:
            trade_id: 交易所成交单 ID
            symbol: 交易对
            price: 成交价格
            amount: 成交数量
            side: 买卖方向
            timestamp: Unix 时间戳（毫秒）
        """
        query = """
            INSERT INTO market_trades (id, symbol, price, amount, side, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO NOTHING
        """

        await self.db_pool.execute_query(
            query,
            str(trade_id),
            symbol,
            str(price),
            str(amount),
            side,
            timestamp
        )

    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置 Redis 缓存"""
        if self.redis:
            try:
                await self.redis.set(key, value, ex=ttl)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

    async def get_cache(self, key: str) -> Optional[str]:
        """获取 Redis 缓存"""
        if self.redis:
            try:
                return await self.redis.get(key)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        return None

    async def delete_cache(self, key: str) -> None:
        """删除 Redis 缓存"""
        if self.redis:
            try:
                await self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

    async def evolve_schema(self, sql: str) -> bool:
        """
        执行数据库结构演进 SQL（如 ALTER TABLE）

        Args:
            sql: 要执行的 SQL 语句

        Returns:
            bool: 是否执行成功

        Note:
            为防止 SQL 注入，只允许执行以下 DDL 语句：
            - CREATE TABLE/INDEX/VIEW
            - ALTER TABLE ADD/DROP/ALTER COLUMN
            - DROP TABLE/INDEX
            禁止：SELECT、INSERT、UPDATE、DELETE、TRUNCATE、多语句执行
        """
        # SQL 注入防护：白名单验证
        sql_normalized = sql.strip()

        # 1. 禁止多语句执行（防止注入多条SQL）
        if ';' in sql_normalized:
            logger.error(f"❌ SQL 注入防护：禁止多语句执行")
            return False

        # 2. 转换为大写进行关键字检测
        sql_upper = sql_normalized.upper()

        # 3. 禁止数据查询和操作语句
        forbidden_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE',
            'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL',
            'COPY', 'pg_read_file', 'pg_write_file'
        ]
        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                logger.error(f"❌ SQL 注入防护：禁止使用 {keyword}")
                return False

        # 4. 只允许 DDL 语句（CREATE、ALTER、DROP）
        allowed_prefixes = ('CREATE', 'ALTER', 'DROP')
        if not sql_upper.startswith(allowed_prefixes):
            logger.error(f"❌ SQL 注入防护：只允许 CREATE/ALTER/DROP 语句")
            return False

        # 5. DROP 语句额外验证：只允许 DROP INDEX/ COLUMN，不允许 DROP TABLE（防误删）
        if sql_upper.startswith('DROP'):
            drop_forbidden = ['DROP TABLE', 'DROP DATABASE', 'DROP SCHEMA']
            for forbid in drop_forbidden:
                if sql_upper.startswith(forbid):
                    logger.error(f"❌ SQL 注入防护：禁止 {forbid}")
                    return False

        try:
            await self.db_pool.execute_query(sql)
            logger.info(f"🧬 数据库结构演进成功: {sql_normalized}")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库结构演进失败: {e}")
            return False

    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """
        执行只读查询

        Args:
            query: SQL 查询语句
            params: 查询参数（使用 $1, $2 占位符）

        Returns:
            查询结果列表
        """
        if params:
            return await self.db_pool.execute_query(query, *params)
        else:
            return await self.db_pool.execute_query(query)

    async def get_connection_info(self) -> Dict[str, Any]:
        """
        获取连接池状态信息

        Returns:
            包含连接池状态的字典
        """
        return {
            'pool_initialized': self.db_pool.is_initialized if self.db_pool else False,
            'pool_size': self.db_pool.size if self.db_pool else 0,
            'pool_min_size': self.pool_min_size,
            'pool_max_size': self.pool_max_size,
            'redis_connected': self.redis is not None
        }
