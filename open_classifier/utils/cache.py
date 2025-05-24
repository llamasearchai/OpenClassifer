"""
High-performance caching utilities for OpenClassifier.
Provides LRU cache, Redis cache, and hybrid caching strategies.
"""

import time
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, Union, Callable
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import redis
import asyncio
from functools import wraps

from open_classifier.core.config import settings
from open_classifier.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_requests = 0


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    Provides efficient caching with automatic eviction of oldest items.
    """

    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = RLock()
        self.stats = CacheStats()

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl is None:
            return False
        
        timestamp = self.timestamps.get(key)
        if timestamp is None:
            return True
        
        return time.time() - timestamp > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            self.stats.total_requests += 1
            
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            if self._is_expired(key):
                del self.cache[key]
                del self.timestamps[key]
                self.stats.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats.hits += 1
            return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing item
                self.cache[key] = value
                self.timestamps[key] = current_time
                self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                    self.stats.evictions += 1
                
                self.cache[key] = value
                self.timestamps[key] = current_time

    def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.stats.reset()

    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class RedisCache:
    """
    Redis-based cache implementation for distributed caching.
    Provides persistence and shared cache across multiple instances.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "openclassifier:"
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis_client:
            self.stats.errors += 1
            return None

        try:
            self.stats.total_requests += 1
            cache_key = self._make_key(key)
            
            data = self.redis_client.get(cache_key)
            if data is None:
                self.stats.misses += 1
                return None
            
            self.stats.hits += 1
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self.stats.errors += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set an item in the Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            self.stats.errors += 1
            return False

        try:
            cache_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            data = pickle.dumps(value)
            result = self.redis_client.setex(cache_key, ttl, data)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            self.stats.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """
        Delete an item from the Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key)
            result = self.redis_client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            self.stats.errors += 1
            return False

    def clear(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            pattern = f"{self.key_prefix}{pattern}"
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self.stats.errors += 1
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key)
            return bool(self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class HybridCache:
    """
    Hybrid cache combining local LRU cache with Redis for optimal performance.
    Uses local cache for hot data and Redis for persistent/shared storage.
    """

    def __init__(
        self,
        local_maxsize: int = 1000,
        local_ttl: Optional[float] = 300,  # 5 minutes
        redis_ttl: int = 3600,  # 1 hour
        redis_url: Optional[str] = None
    ):
        """
        Initialize hybrid cache.
        
        Args:
            local_maxsize: Maximum size of local cache
            local_ttl: Local cache TTL in seconds
            redis_ttl: Redis cache TTL in seconds
            redis_url: Redis connection URL
        """
        self.local_cache = LRUCache(maxsize=local_maxsize, ttl=local_ttl)
        self.redis_cache = RedisCache(redis_url=redis_url, default_ttl=redis_ttl)
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get item from hybrid cache (local first, then Redis)."""
        self.stats.total_requests += 1
        
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try Redis cache
        value = self.redis_cache.get(key)
        if value is not None:
            # Store in local cache for faster access
            self.local_cache.set(key, value)
            self.stats.hits += 1
            return value
        
        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in both local and Redis caches."""
        self.local_cache.set(key, value)
        self.redis_cache.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete item from both caches."""
        local_deleted = self.local_cache.delete(key)
        redis_deleted = self.redis_cache.delete(key)
        return local_deleted or redis_deleted

    def clear(self) -> None:
        """Clear both caches."""
        self.local_cache.clear()
        self.redis_cache.clear()
        self.stats.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hybrid": self.stats,
            "local": self.local_cache.get_stats(),
            "redis": self.redis_cache.get_stats()
        }


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Generated cache key
    """
    # Create a string representation of arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use their string representation
            key_parts.append(str(arg))
        else:
            key_parts.append(repr(arg))
    
    # Add keyword arguments (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={repr(v)}")
    
    # Create a hash of the combined key parts
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    cache: Optional[Union[LRUCache, RedisCache, HybridCache]] = None,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance to use (defaults to global cache)
        ttl: Time-to-live for cached results
        key_func: Function to generate cache keys
        
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        nonlocal cache
        
        if cache is None:
            # Use global cache instance
            cache = _global_cache
        
        if key_func is None:
            def default_key_func(*args, **kwargs):
                return f"{func.__module__}.{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            key_generator = default_key_func
        else:
            key_generator = key_func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_generator(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            if isinstance(cache, RedisCache):
                cache.set(cache_key, result, ttl)
            else:
                cache.set(cache_key, result)
            
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_generator(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Execute async function and cache result
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            if isinstance(cache, RedisCache):
                cache.set(cache_key, result, ttl)
            else:
                cache.set(cache_key, result)
            
            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


# Global cache instance
_global_cache = HybridCache()


def get_cache() -> HybridCache:
    """Get the global cache instance."""
    return _global_cache


def clear_all_caches() -> None:
    """Clear all caches."""
    _global_cache.clear()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return _global_cache.get_stats()


# Specialized cache decorators for common use cases
def cache_embeddings(ttl: int = 86400):  # 24 hours
    """Cache decorator specifically for embeddings."""
    return cached(ttl=ttl, key_func=lambda text, model=None: f"embedding:{hashlib.md5(text.encode()).hexdigest()}:{model}")


def cache_classification(ttl: int = 3600):  # 1 hour
    """Cache decorator specifically for classification results."""
    return cached(ttl=ttl, key_func=lambda text, labels, **kwargs: f"classify:{hashlib.md5(text.encode()).hexdigest()}:{hashlib.md5(str(sorted(labels)).encode()).hexdigest()}")


def cache_similarity(ttl: int = 1800):  # 30 minutes
    """Cache decorator specifically for similarity computations."""
    return cached(ttl=ttl, key_func=lambda query, candidates, **kwargs: f"similarity:{hashlib.md5(query.encode()).hexdigest()}:{hashlib.md5(str(candidates).encode()).hexdigest()}") 