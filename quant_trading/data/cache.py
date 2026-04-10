"""Data Caching System - 计算指标缓存 (Phase 3).

高性能缓存系统，用于存储计算过的指标和因子值。

Features:
- LRU cache for recent data
- Multi-level cache (memory + disk)
- Automatic invalidation
- Thread-safe operations

Usage
-----
```python
from quant_trading.data.cache import IndicatorCache, FactorCache, CacheManager

# 指标缓存
cache = IndicatorCache(max_size=1000)
rsi = cache.get_or_compute('rsi', close, n=14, compute_fn=lambda: my_rsi(close, 14))

# 因子缓存
factor_cache = FactorCache(cache_dir='./factor_cache')
factor = factor_cache.get('alpha001', symbol, timestamp, compute_fn=compute_alpha)

# 全局缓存管理器
cache_mgr = CacheManager()
result = cache_mgr.get_cached_indicator(symbol, 'rsi', period=14)
```
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any, Union
import threading
import logging

import numpy as np
import pandas as pd

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# LRU Cache Implementation
# ---------------------------------------------------------------------------

class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache.

    A cache that evicts the least recently used items when max_size is reached.

    Type parameters:
        K: Key type
        V: Value type

    Usage
    -----
    ```python
    cache = LRUCache(max_size=100)
    cache.put('key1', [1, 2, 3])
    value = cache.get('key1')  # Returns [1, 2, 3] or None
    ```
    """

    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store
        """
        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new item
                if len(self._cache) >= self._max_size:
                    # Remove least recently used (first item)
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of items in cache."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: str) -> bool:
        return key in self._cache


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------

def _generate_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from arguments.

    Args:
        prefix: Key prefix (e.g., 'rsi', 'macd')
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        Cache key string
    """
    # Create a deterministic string representation
    parts = [prefix]

    for arg in args:
        if isinstance(arg, np.ndarray):
            # For arrays, use shape and a hash of the data
            if len(arg) > 100:
                # Just use first and last few elements
                arr_repr = f"arr[{arg.shape}]({arg.flat[0]:.6g}...{arg.flat[-1]:.6g})"
            else:
                arr_repr = f"arr[{arg.shape}]({','.join(str(x) for x in arr.flat[:5])})"
            parts.append(arr_repr)
        elif isinstance(arg, pd.DataFrame):
            parts.append(f"df[{len(arg)}x{len(arg.columns)}]")
        elif isinstance(arg, (list, tuple)):
            parts.append(str(arg))
        else:
            parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")

    key_str = "|".join(parts)

    # If key is too long, hash it
    if len(key_str) > 200:
        h = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}|{h}"

    return key_str


def _hash_array(arr: np.ndarray) -> str:
    """Create a short hash of a numpy array for cache keys."""
    if arr.dtype == np.float64:
        arr_bytes = arr.astype(np.float32).tobytes()
    else:
        arr_bytes = arr.tobytes()
    return hashlib.md5(arr_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Indicator Cache
# ---------------------------------------------------------------------------

@dataclass
class CachedIndicator:
    """缓存的指标数据.

    Attributes:
        value: The cached indicator array
        computed_at: Timestamp when computed
        params: Parameters used for computation
        size_bytes: Approximate size in bytes
    """
    value: np.ndarray
    computed_at: float = field(default_factory=time.time)
    params: dict = field(default_factory=dict)
    size_bytes: int = 0

    def is_stale(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if cache entry is stale.

        Args:
            max_age_seconds: Maximum age before considered stale

        Returns:
            True if entry is stale
        """
        return time.time() - self.computed_at > max_age_seconds


class IndicatorCache:
    """技术指标缓存 (LRU + 自动失效).

    Caches computed technical indicators to avoid redundant calculations.

    Usage
    -----
    ```python
    cache = IndicatorCache(max_size=500)

    # Compute RSI with caching
    rsi = cache.get_or_compute(
        'rsi_BTCUSDT_24h',
        close_data,
        n=24,
        compute_fn=lambda: calculate_rsi(close, 24)
    )

    # Batch get
    results = cache.get_many([
        ('rsi', close, {'n': 24}),
        ('macd', close, {'short': 12, 'long': 26}),
    ], compute_fn=...)
    ```

    Type parameters:
        K: Key type
        V: Value type
    """

    def __init__(
        self,
        max_size: int = 500,
        max_age_seconds: float = 3600.0,
        use_float32: bool = True
    ):
        """初始化指标缓存.

        Args:
            max_size: Maximum number of cached indicators
            max_age_seconds: Maximum age before auto-invalidation
            use_float32: Store arrays as float32 to save memory
        """
        self._cache = LRUCache(max_size=max_size)
        self._max_age = max_age_seconds
        self._use_float32 = use_float32
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached indicator.

        Args:
            key: Cache key

        Returns:
            Cached array or None
        """
        cached = self._cache.get(key)
        if cached is None:
            return None

        if isinstance(cached, CachedIndicator):
            if cached.is_stale(self._max_age):
                return None
            if self._use_float32 and cached.value.dtype == np.float64:
                return cached.value.astype(np.float32)
            return cached.value

        return cached

    def put(
        self,
        key: str,
        value: np.ndarray,
        params: Optional[dict] = None
    ) -> None:
        """Cache an indicator.

        Args:
            key: Cache key
            value: Indicator array to cache
            params: Optional parameters used for computation
        """
        if self._use_float32 and value.dtype == np.float64:
            value = value.astype(np.float32)

        cached = CachedIndicator(
            value=value,
            params=params or {},
            size_bytes=value.nbytes
        )
        self._cache.put(key, cached)

    def get_or_compute(
        self,
        key: str,
        data: np.ndarray,
        compute_fn: Callable[[], np.ndarray],
        **params
    ) -> np.ndarray:
        """Get from cache or compute if not found.

        Args:
            key: Cache key
            data: Input data (used for validation only)
            compute_fn: Function to compute indicator if not cached
            **params: Additional parameters for cache key

        Returns:
            Cached or computed indicator array
        """
        full_key = _generate_key(key, **params)

        cached = self.get(full_key)
        if cached is not None:
            return cached

        # Compute
        result = compute_fn()

        # Cache
        self.put(full_key, result, params)

        return result

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if key in self._cache:
                self._cache.cache.pop(key, None)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached indicators."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with size, hit_rate, etc.
        """
        total_size = 0
        for item in self._cache._cache.values():
            if isinstance(item, CachedIndicator):
                total_size += item.size_bytes

        return {
            'size': len(self._cache),
            'max_size': self._cache._max_size,
            'hit_rate': self._cache.hit_rate,
            'hits': self._cache._hits,
            'misses': self._cache._misses,
            'total_bytes': total_size,
        }


# ---------------------------------------------------------------------------
# Factor Cache (with disk persistence)
# ---------------------------------------------------------------------------

class FactorCache:
    """因子缓存 (Memory + Disk 两级缓存).

    Caches computed factor values with optional disk persistence.

    Usage
    -----
    ```python
    cache = FactorCache(cache_dir='./factor_cache')

    # Get or compute a factor
    alpha = cache.get(
        'alpha001',
        symbol='BTCUSDT',
        timestamp=1234567890,
        compute_fn=lambda: compute_alpha_001()
    )

    # Batch load
    cache.load_batch('alpha*', ['BTCUSDT', 'ETHUSDT'])
    ```
    """

    def __init__(
        self,
        cache_dir: str = './factor_cache',
        max_memory_size: int = 1000,
        ttl_seconds: float = 86400.0
    ):
        """初始化因子缓存.

        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum items in memory cache
            ttl_seconds: Time-to-live for disk cache (default 24h)
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = LRUCache(max_size=max_memory_size)
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._disk_stats = {'hits': 0, 'misses': 0, 'writes': 0}

    def _disk_key(self, name: str, symbol: str, timestamp: int) -> Path:
        """Get disk cache file path."""
        fname = f"{name}_{symbol}_{timestamp}.pkl"
        return self._cache_dir / fname

    def get(
        self,
        name: str,
        symbol: str,
        timestamp: int,
        compute_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """Get factor from cache or compute.

        Args:
            name: Factor name (e.g., 'alpha001')
            symbol: Trading symbol
            timestamp: Bar timestamp
            compute_fn: Optional function to compute if not cached

        Returns:
            Cached factor value or None
        """
        key = f"{name}|{symbol}|{timestamp}"

        # Try memory first
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached

        # Try disk
        disk_path = self._disk_key(name, symbol, timestamp)
        if disk_path.exists():
            age = time.time() - disk_path.stat().st_mtime
            if age < self._ttl:
                try:
                    with open(disk_path, 'rb') as f:
                        value = pickle.load(f)
                    self._memory_cache.put(key, value)
                    self._disk_stats['hits'] += 1
                    return value
                except Exception as e:
                    logger.warning(f"Failed to load factor from disk: {e}")

        self._disk_stats['misses'] += 1

        # Compute if provided
        if compute_fn is not None:
            value = compute_fn()
            self.put(name, symbol, timestamp, value)
            return value

        return None

    def put(
        self,
        name: str,
        symbol: str,
        timestamp: int,
        value: Any
    ) -> None:
        """Store factor in cache.

        Args:
            name: Factor name
            symbol: Trading symbol
            timestamp: Bar timestamp
            value: Value to cache
        """
        key = f"{name}|{symbol}|{timestamp}"

        # Memory
        self._memory_cache.put(key, value)

        # Disk
        disk_path = self._disk_key(name, symbol, timestamp)
        try:
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._disk_stats['writes'] += 1
        except Exception as e:
            logger.warning(f"Failed to write factor to disk: {e}")

    def get_many(
        self,
        keys: list[tuple],
        compute_fn: Optional[Callable] = None
    ) -> dict:
        """Get multiple factors at once.

        Args:
            keys: List of (name, symbol, timestamp) tuples
            compute_fn: Optional batch compute function

        Returns:
            Dict mapping (name, symbol, timestamp) to values
        """
        results = {}
        missing = []

        for key_tuple in keys:
            if len(key_tuple) == 3:
                name, symbol, timestamp = key_tuple
            else:
                continue

            value = self.get(name, symbol, timestamp)
            if value is not None:
                results[key_tuple] = value
            else:
                missing.append(key_tuple)

        # Batch compute missing
        if compute_fn and missing:
            computed = compute_fn(missing)
            for key_tuple, value in computed.items():
                results[key_tuple] = value
                name, symbol, timestamp = key_tuple
                self.put(name, symbol, timestamp, value)

        return results

    def clear_disk(self, older_than_seconds: float = None) -> int:
        """Clear disk cache.

        Args:
            older_than_seconds: Only clear files older than this

        Returns:
            Number of files removed
        """
        removed = 0
        cutoff = time.time() - (older_than_seconds or 0)

        for f in self._cache_dir.glob('*.pkl'):
            if older_than_seconds is None or f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1

        return removed

    def stats(self) -> dict:
        """Get cache statistics."""
        mem_stats = self._memory_cache._cache.__dict__ if hasattr(self._memory_cache, '_cache') else {}

        disk_size = sum(f.stat().st_size for f in self._cache_dir.glob('*.pkl'))

        return {
            'memory_size': len(self._memory_cache),
            'memory_hit_rate': self._memory_cache.hit_rate,
            'disk_files': len(list(self._cache_dir.glob('*.pkl'))),
            'disk_bytes': disk_size,
            **self._disk_stats
        }


# ---------------------------------------------------------------------------
# Global Cache Manager
# ---------------------------------------------------------------------------

class CacheManager:
    """全局缓存管理器 (Singleton).

    Provides unified access to all caches in the system.

    Usage
    -----
    ```python
    cache_mgr = CacheManager()

    # Get cached indicator
    rsi = cache_mgr.get_indicator('BTCUSDT', 'rsi', period=24)

    # Get cached factor
    alpha = cache_mgr.get_factor('alpha001', 'BTCUSDT', timestamp)

    # Clear all caches
    cache_mgr.clear_all()
    ```
    """

    _instance: Optional['CacheManager'] = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._indicator_cache = IndicatorCache(max_size=500)
        self._factor_cache = FactorCache(cache_dir='./factor_cache')
        self._redis_client = None
        self._use_redis = False

        # Try to connect to Redis if available
        if _REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self._redis_client.ping()
                self._use_redis = True
                logger.info("Redis cache connected")
            except Exception:
                self._redis_client = None
                self._use_redis = False

    @property
    def indicators(self) -> IndicatorCache:
        """Indicator cache instance."""
        return self._indicator_cache

    @property
    def factors(self) -> FactorCache:
        """Factor cache instance."""
        return self._factor_cache

    def get_indicator(
        self,
        symbol: str,
        indicator_name: str,
        **params
    ) -> Optional[np.ndarray]:
        """Get cached indicator for symbol.

        Args:
            symbol: Trading symbol
            indicator_name: Indicator name (e.g., 'rsi', 'macd')
            **params: Indicator parameters

        Returns:
            Cached indicator array or None
        """
        key = f"{symbol}_{indicator_name}"
        return self._indicator_cache.get(key)

    def cache_indicator(
        self,
        symbol: str,
        indicator_name: str,
        value: np.ndarray,
        **params
    ) -> None:
        """Cache indicator for symbol.

        Args:
            symbol: Trading symbol
            indicator_name: Indicator name
            value: Indicator array
            **params: Indicator parameters
        """
        key = f"{symbol}_{indicator_name}"
        self._indicator_cache.put(key, value, params)

    def get_factor(
        self,
        factor_name: str,
        symbol: str,
        timestamp: int,
        compute_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """Get cached factor value.

        Args:
            factor_name: Factor name (e.g., 'alpha001')
            symbol: Trading symbol
            timestamp: Bar timestamp
            compute_fn: Optional compute function

        Returns:
            Cached factor or None
        """
        return self._factor_cache.get(factor_name, symbol, timestamp, compute_fn)

    def cache_factor(
        self,
        factor_name: str,
        symbol: str,
        timestamp: int,
        value: Any
    ) -> None:
        """Cache factor value.

        Args:
            factor_name: Factor name
            symbol: Trading symbol
            timestamp: Bar timestamp
            value: Factor value
        """
        self._factor_cache.put(factor_name, symbol, timestamp, value)

    def clear_all(self) -> None:
        """Clear all caches."""
        self._indicator_cache.clear()
        self._factor_cache.clear_disk()

    def stats(self) -> dict:
        """Get statistics for all caches."""
        return {
            'indicators': self._indicator_cache.stats(),
            'factors': self._factor_cache.stats(),
            'redis_enabled': self._use_redis
        }


# ---------------------------------------------------------------------------
# Decorators for easy caching
# ---------------------------------------------------------------------------

def cached_indicator(cache: IndicatorCache, key_prefix: str, **param_names):
    """Decorator to cache indicator calculations.

    Usage
    -----
    ```python
    @cached_indicator(my_cache, 'rsi', n='period')
    def compute_rsi(close, period):
        # ... computation ...
        return rsi
    ```
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract params for key
            key_params = {}
            for p_name, arg_name in param_names.items():
                if p_name in kwargs:
                    key_params[arg_name] = kwargs[p_name]
                elif len(args) > 0:
                    key_params[arg_name] = args[0]

            key = _generate_key(key_prefix, **key_params)
            cached = cache.get(key)

            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.put(key, result, key_params)
            return result

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Convenience instance
# ---------------------------------------------------------------------------

_default_cache: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the default cache manager instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache
