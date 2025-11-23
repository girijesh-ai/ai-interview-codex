"""
Redis Cache Layer for Session Memory (Tier 2)

Demonstrates:
- Repository pattern for cache abstraction
- Strategy pattern for different cache backends
- Facade pattern for simplified interface
- Decorator pattern for cache operations
- Builder pattern for cache key construction
- Singleton pattern for Redis client
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import pickle
from enum import Enum
import logging

# Redis client
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)


# ============================================================================
# VALUE OBJECTS & ENUMS
# ============================================================================

class CacheNamespace(str, Enum):
    """Cache namespaces for organization."""
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    METRICS = "metrics"
    COORDINATION = "coordination"
    TEMPORARY = "temp"


class SerializationFormat(str, Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


@dataclass(frozen=True)
class CacheKey:
    """Value object for cache keys.

    Demonstrates:
    - Value object pattern
    - Immutability
    - Composite key construction
    """
    namespace: CacheNamespace
    identifier: str
    suffix: Optional[str] = None

    def to_string(self) -> str:
        """Convert to Redis key format."""
        parts = [self.namespace.value, self.identifier]
        if self.suffix:
            parts.append(self.suffix)
        return ":".join(parts)

    @classmethod
    def from_string(cls, key: str) -> "CacheKey":
        """Parse Redis key."""
        parts = key.split(":")
        namespace = CacheNamespace(parts[0])
        identifier = parts[1]
        suffix = parts[2] if len(parts) > 2 else None
        return cls(namespace, identifier, suffix)


@dataclass
class CacheEntry:
    """Cache entry with metadata.

    Demonstrates:
    - Entity pattern (has identity via key)
    - Metadata tracking
    """
    key: CacheKey
    value: Any
    ttl: Optional[int] = None  # Seconds
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_keys: int = 0
    memory_used: int = 0  # Bytes

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============================================================================
# CACHE REPOSITORY INTERFACE
# ============================================================================

class CacheRepository(ABC):
    """Abstract cache repository.

    Demonstrates:
    - Repository pattern
    - Interface segregation principle (ISP)
    - Dependency inversion principle (DIP)
    """

    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve value from cache."""
        pass

    @abstractmethod
    async def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def get_many(self, keys: List[CacheKey]) -> Dict[str, Any]:
        """Retrieve multiple values."""
        pass

    @abstractmethod
    async def set_many(self, items: Dict[CacheKey, Any], ttl: Optional[int] = None) -> bool:
        """Store multiple values."""
        pass

    @abstractmethod
    async def delete_many(self, keys: List[CacheKey]) -> int:
        """Delete multiple keys."""
        pass

    @abstractmethod
    async def increment(self, key: CacheKey, amount: int = 1) -> int:
        """Increment counter."""
        pass

    @abstractmethod
    async def expire(self, key: CacheKey, ttl: int) -> bool:
        """Set expiration."""
        pass

    @abstractmethod
    async def get_ttl(self, key: CacheKey) -> Optional[int]:
        """Get remaining TTL."""
        pass

    @abstractmethod
    async def keys_by_pattern(self, pattern: str) -> List[str]:
        """Find keys matching pattern."""
        pass

    @abstractmethod
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Clear all keys in namespace."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


# ============================================================================
# SERIALIZATION STRATEGY
# ============================================================================

class SerializationStrategy(ABC):
    """Abstract serialization strategy.

    Demonstrates:
    - Strategy pattern
    """

    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        pass


class JsonSerializationStrategy(SerializationStrategy):
    """JSON serialization strategy."""

    def serialize(self, value: Any) -> bytes:
        """Serialize to JSON."""
        return json.dumps(value, default=str).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        """Deserialize from JSON."""
        return json.loads(data.decode('utf-8'))


class PickleSerializationStrategy(SerializationStrategy):
    """Pickle serialization strategy."""

    def serialize(self, value: Any) -> bytes:
        """Serialize with pickle."""
        return pickle.dumps(value)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize with pickle."""
        return pickle.loads(data)


class StringSerializationStrategy(SerializationStrategy):
    """String serialization strategy."""

    def serialize(self, value: Any) -> bytes:
        """Serialize to string."""
        return str(value).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        """Deserialize from string."""
        return data.decode('utf-8')


class SerializationStrategyFactory:
    """Factory for creating serialization strategies.

    Demonstrates:
    - Factory pattern
    """

    @staticmethod
    def create(format: SerializationFormat) -> SerializationStrategy:
        """Create serialization strategy."""
        if format == SerializationFormat.JSON:
            return JsonSerializationStrategy()
        elif format == SerializationFormat.PICKLE:
            return PickleSerializationStrategy()
        elif format == SerializationFormat.STRING:
            return StringSerializationStrategy()
        else:
            raise ValueError(f"Unsupported serialization format: {format}")


# ============================================================================
# REDIS CACHE IMPLEMENTATION
# ============================================================================

class RedisCache(CacheRepository):
    """Redis cache implementation.

    Demonstrates:
    - Repository pattern implementation
    - Strategy pattern for serialization
    - Connection pooling
    - Error handling
    - Async operations
    """

    def __init__(
        self,
        redis_client: Redis,
        serialization_format: SerializationFormat = SerializationFormat.JSON,
        default_ttl: Optional[int] = 3600  # 1 hour
    ):
        """Initialize Redis cache.

        Args:
            redis_client: Redis client instance
            serialization_format: Default serialization format
            default_ttl: Default TTL in seconds
        """
        self.client = redis_client
        self.serializer = SerializationStrategyFactory.create(serialization_format)
        self.default_ttl = default_ttl
        self._stats = CacheStats()

    async def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            redis_key = key.to_string()
            data = await self.client.get(redis_key)

            if data is None:
                self._stats.misses += 1
                logger.debug(f"Cache miss: {redis_key}")
                return None

            self._stats.hits += 1
            logger.debug(f"Cache hit: {redis_key}")

            # Deserialize
            value = self.serializer.deserialize(data)
            return value

        except RedisError as e:
            logger.error(f"Redis error on get: {e}")
            self._stats.misses += 1
            return None

    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        try:
            redis_key = key.to_string()

            # Serialize
            data = self.serializer.serialize(value)

            # Set with TTL
            ttl = ttl or self.default_ttl
            if ttl:
                await self.client.setex(redis_key, ttl, data)
            else:
                await self.client.set(redis_key, data)

            self._stats.sets += 1
            logger.debug(f"Cache set: {redis_key} (ttl={ttl})")
            return True

        except RedisError as e:
            logger.error(f"Redis error on set: {e}")
            return False

    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        try:
            redis_key = key.to_string()
            result = await self.client.delete(redis_key)

            if result > 0:
                self._stats.deletes += 1
                logger.debug(f"Cache delete: {redis_key}")
                return True
            return False

        except RedisError as e:
            logger.error(f"Redis error on delete: {e}")
            return False

    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists.

        Args:
            key: Cache key

        Returns:
            Existence status
        """
        try:
            redis_key = key.to_string()
            return await self.client.exists(redis_key) > 0
        except RedisError as e:
            logger.error(f"Redis error on exists: {e}")
            return False

    async def get_many(self, keys: List[CacheKey]) -> Dict[str, Any]:
        """Retrieve multiple values.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value
        """
        try:
            redis_keys = [k.to_string() for k in keys]
            values = await self.client.mget(redis_keys)

            result = {}
            for redis_key, data in zip(redis_keys, values):
                if data is not None:
                    result[redis_key] = self.serializer.deserialize(data)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1

            return result

        except RedisError as e:
            logger.error(f"Redis error on get_many: {e}")
            return {}

    async def set_many(
        self,
        items: Dict[CacheKey, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store multiple values.

        Args:
            items: Dictionary of key -> value
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        try:
            pipeline = self.client.pipeline()

            for key, value in items.items():
                redis_key = key.to_string()
                data = self.serializer.serialize(value)

                ttl = ttl or self.default_ttl
                if ttl:
                    pipeline.setex(redis_key, ttl, data)
                else:
                    pipeline.set(redis_key, data)

            await pipeline.execute()
            self._stats.sets += len(items)
            return True

        except RedisError as e:
            logger.error(f"Redis error on set_many: {e}")
            return False

    async def delete_many(self, keys: List[CacheKey]) -> int:
        """Delete multiple keys.

        Args:
            keys: List of cache keys

        Returns:
            Number of keys deleted
        """
        try:
            redis_keys = [k.to_string() for k in keys]
            count = await self.client.delete(*redis_keys)
            self._stats.deletes += count
            return count

        except RedisError as e:
            logger.error(f"Redis error on delete_many: {e}")
            return 0

    async def increment(self, key: CacheKey, amount: int = 1) -> int:
        """Increment counter.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            New value
        """
        try:
            redis_key = key.to_string()
            return await self.client.incrby(redis_key, amount)
        except RedisError as e:
            logger.error(f"Redis error on increment: {e}")
            return 0

    async def expire(self, key: CacheKey, ttl: int) -> bool:
        """Set expiration.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        try:
            redis_key = key.to_string()
            return await self.client.expire(redis_key, ttl)
        except RedisError as e:
            logger.error(f"Redis error on expire: {e}")
            return False

    async def get_ttl(self, key: CacheKey) -> Optional[int]:
        """Get remaining TTL.

        Args:
            key: Cache key

        Returns:
            TTL in seconds or None
        """
        try:
            redis_key = key.to_string()
            ttl = await self.client.ttl(redis_key)
            return ttl if ttl > 0 else None
        except RedisError as e:
            logger.error(f"Redis error on get_ttl: {e}")
            return None

    async def keys_by_pattern(self, pattern: str) -> List[str]:
        """Find keys matching pattern.

        Args:
            pattern: Pattern (e.g., "session:*")

        Returns:
            List of matching keys
        """
        try:
            keys = await self.client.keys(pattern)
            return [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]
        except RedisError as e:
            logger.error(f"Redis error on keys_by_pattern: {e}")
            return []

    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Clear all keys in namespace.

        Args:
            namespace: Cache namespace

        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{namespace.value}:*"
            keys = await self.keys_by_pattern(pattern)

            if keys:
                count = await self.client.delete(*keys)
                self._stats.evictions += count
                logger.info(f"Cleared namespace {namespace.value}: {count} keys")
                return count
            return 0

        except RedisError as e:
            logger.error(f"Redis error on clear_namespace: {e}")
            return 0

    async def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        try:
            info = await self.client.info('stats')

            self._stats.total_keys = await self.client.dbsize()
            self._stats.memory_used = int(info.get('used_memory', 0))

            return self._stats

        except RedisError as e:
            logger.error(f"Redis error on get_stats: {e}")
            return self._stats

    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            Connection status
        """
        try:
            return await self.client.ping()
        except RedisError:
            return False


# ============================================================================
# SESSION CACHE FACADE
# ============================================================================

class SessionCache:
    """Facade for session-specific caching.

    Demonstrates:
    - Facade pattern
    - Single responsibility principle (SRP)
    - Domain-specific interface
    """

    def __init__(self, cache: CacheRepository, default_ttl: int = 1800):
        """Initialize session cache.

        Args:
            cache: Cache repository
            default_ttl: Default TTL (30 minutes)
        """
        self.cache = cache
        self.default_ttl = default_ttl

    async def store_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store session data.

        Args:
            session_id: Session identifier
            data: Session data
            ttl: Time to live

        Returns:
            Success status
        """
        key = CacheKey(CacheNamespace.SESSION, session_id)
        return await self.cache.set(key, data, ttl or self.default_ttl)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        key = CacheKey(CacheNamespace.SESSION, session_id)
        return await self.cache.get(key)

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data.

        Args:
            session_id: Session identifier
            updates: Fields to update

        Returns:
            Success status
        """
        # Get existing session
        existing = await self.get_session(session_id)
        if existing is None:
            existing = {}

        # Merge updates
        existing.update(updates)

        # Store back
        return await self.store_session(session_id, existing)

    async def delete_session(self, session_id: str) -> bool:
        """Delete session.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        key = CacheKey(CacheNamespace.SESSION, session_id)
        return await self.cache.delete(key)

    async def extend_session(self, session_id: str, ttl: int) -> bool:
        """Extend session TTL.

        Args:
            session_id: Session identifier
            ttl: New TTL in seconds

        Returns:
            Success status
        """
        key = CacheKey(CacheNamespace.SESSION, session_id)
        return await self.cache.expire(key, ttl)


# ============================================================================
# AGENT COORDINATION CACHE
# ============================================================================

class AgentCoordinationCache:
    """Cache for agent coordination.

    Demonstrates:
    - Facade pattern
    - Domain-specific operations
    """

    def __init__(self, cache: CacheRepository):
        """Initialize coordination cache.

        Args:
            cache: Cache repository
        """
        self.cache = cache

    async def acquire_lock(
        self,
        resource: str,
        agent_id: str,
        ttl: int = 60
    ) -> bool:
        """Acquire distributed lock.

        Args:
            resource: Resource identifier
            agent_id: Agent acquiring lock
            ttl: Lock timeout

        Returns:
            Lock acquired status
        """
        key = CacheKey(CacheNamespace.COORDINATION, resource, "lock")

        # Try to set if not exists
        try:
            redis_key = key.to_string()
            result = await self.cache.client.set(
                redis_key,
                agent_id,
                nx=True,  # Only if not exists
                ex=ttl
            )
            return result is not None
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False

    async def release_lock(self, resource: str, agent_id: str) -> bool:
        """Release distributed lock.

        Args:
            resource: Resource identifier
            agent_id: Agent releasing lock

        Returns:
            Lock released status
        """
        key = CacheKey(CacheNamespace.COORDINATION, resource, "lock")

        # Only delete if owned by this agent
        try:
            redis_key = key.to_string()
            current_owner = await self.cache.get(key)

            if current_owner == agent_id:
                return await self.cache.delete(key)
            return False
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            return False

    async def publish_event(self, channel: str, event: Dict[str, Any]) -> int:
        """Publish event to channel.

        Args:
            channel: Channel name
            event: Event data

        Returns:
            Number of subscribers
        """
        try:
            data = json.dumps(event)
            return await self.cache.client.publish(channel, data)
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return 0

    async def increment_counter(self, counter_name: str) -> int:
        """Increment coordination counter.

        Args:
            counter_name: Counter name

        Returns:
            New counter value
        """
        key = CacheKey(CacheNamespace.COORDINATION, counter_name)
        return await self.cache.increment(key)


# ============================================================================
# CACHE FACTORY
# ============================================================================

class RedisCacheFactory:
    """Factory for creating Redis cache instances.

    Demonstrates:
    - Factory pattern
    - Connection pool management
    - Configuration abstraction
    """

    @staticmethod
    async def create(
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        decode_responses: bool = False,
        serialization_format: SerializationFormat = SerializationFormat.JSON,
        default_ttl: Optional[int] = 3600
    ) -> RedisCache:
        """Create Redis cache instance.

        Args:
            host: Redis host
            port: Redis port
            db: Database number
            password: Password
            max_connections: Max connections in pool
            decode_responses: Decode responses to strings
            serialization_format: Serialization format
            default_ttl: Default TTL

        Returns:
            Redis cache instance
        """
        # Create connection pool
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=decode_responses
        )

        # Create client
        client = Redis(connection_pool=pool)

        # Test connection
        try:
            await client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Create cache
        return RedisCache(client, serialization_format, default_ttl)

    @staticmethod
    async def create_from_url(
        url: str,
        serialization_format: SerializationFormat = SerializationFormat.JSON,
        default_ttl: Optional[int] = 3600
    ) -> RedisCache:
        """Create Redis cache from URL.

        Args:
            url: Redis URL (redis://...)
            serialization_format: Serialization format
            default_ttl: Default TTL

        Returns:
            Redis cache instance
        """
        client = await redis.from_url(url)

        # Test connection
        try:
            await client.ping()
            logger.info(f"Connected to Redis at {url}")
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        return RedisCache(client, serialization_format, default_ttl)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Create Redis cache
        cache = await RedisCacheFactory.create(
            host="localhost",
            port=6379,
            serialization_format=SerializationFormat.JSON
        )

        # Test basic operations
        print("\n=== Testing Basic Operations ===")

        # Set value
        key = CacheKey(CacheNamespace.SESSION, "user-123")
        await cache.set(key, {"name": "John", "email": "john@example.com"}, ttl=300)
        print(f"Set session data for user-123")

        # Get value
        data = await cache.get(key)
        print(f"Retrieved: {data}")

        # Check existence
        exists = await cache.exists(key)
        print(f"Exists: {exists}")

        # Get TTL
        ttl = await cache.get_ttl(key)
        print(f"TTL: {ttl} seconds")

        # Test session facade
        print("\n=== Testing Session Facade ===")

        session_cache = SessionCache(cache)

        # Store session
        await session_cache.store_session(
            "session-456",
            {
                "user_id": "user-789",
                "authenticated": True,
                "last_activity": datetime.now().isoformat()
            }
        )
        print("Stored session data")

        # Retrieve session
        session = await session_cache.get_session("session-456")
        print(f"Retrieved session: {session}")

        # Update session
        await session_cache.update_session(
            "session-456",
            {"page_views": 5}
        )
        print("Updated session")

        # Test coordination cache
        print("\n=== Testing Agent Coordination ===")

        coord_cache = AgentCoordinationCache(cache)

        # Acquire lock
        locked = await coord_cache.acquire_lock("resource-1", "agent-A", ttl=30)
        print(f"Lock acquired: {locked}")

        # Try to acquire same lock (should fail)
        locked2 = await coord_cache.acquire_lock("resource-1", "agent-B", ttl=30)
        print(f"Second lock acquired: {locked2}")

        # Release lock
        released = await coord_cache.release_lock("resource-1", "agent-A")
        print(f"Lock released: {released}")

        # Get stats
        print("\n=== Cache Statistics ===")
        stats = await cache.get_stats()
        print(f"Hits: {stats.hits}")
        print(f"Misses: {stats.misses}")
        print(f"Hit rate: {stats.hit_rate:.2%}")
        print(f"Total keys: {stats.total_keys}")
        print(f"Memory used: {stats.memory_used / 1024:.2f} KB")

        # Cleanup
        await cache.clear_namespace(CacheNamespace.SESSION)
        print("\nCleared session namespace")

        # Close connection
        await cache.client.close()
        print("Connection closed")

    # Run
    asyncio.run(main())
