"""
Redis Patterns Example
======================
Demonstrates:
- Async Redis client setup
- LLM response caching with TTL
- Session management
- Pub/Sub for real-time streaming
- Rate limiting (sliding window, token bucket)

Note: Uses mock Redis for demo without server.
In production, install: redis>=4.5.0

Run with: python a04_redis_patterns.py
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import hashlib
import json
import asyncio
import time


# ==============================================================================
# MOCK REDIS CLIENT
# ==============================================================================

class MockRedis:
    """
    In-memory mock of Redis for demo purposes.
    
    Implements core Redis commands used in AI applications.
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, datetime] = {}
        self.sorted_sets: Dict[str, Dict[str, float]] = {}
        self.pubsub_channels: Dict[str, List[asyncio.Queue]] = {}
    
    def _check_expiry(self, key: str) -> bool:
        """Check if key has expired."""
        if key in self.expiry:
            if datetime.now() > self.expiry[key]:
                self.data.pop(key, None)
                self.expiry.pop(key, None)
                return True
        return False
    
    # String operations
    async def get(self, key: str) -> Optional[str]:
        self._check_expiry(key)
        return self.data.get(key)
    
    async def set(self, key: str, value: str):
        self.data[key] = value
    
    async def setex(self, key: str, ttl: timedelta, value: str):
        self.data[key] = value
        self.expiry[key] = datetime.now() + ttl
    
    async def incr(self, key: str) -> int:
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + 1)
        return current + 1
    
    async def incrby(self, key: str, amount: int) -> int:
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + amount)
        return current + amount
    
    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count
    
    async def expire(self, key: str, seconds: int):
        if key in self.data:
            self.expiry[key] = datetime.now() + timedelta(seconds=seconds)
    
    # Sorted set operations (for rate limiting)
    async def zadd(self, key: str, mapping: Dict[str, float]):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(mapping)
    
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        if key not in self.sorted_sets:
            return 0
        
        to_remove = [
            k for k, v in self.sorted_sets[key].items()
            if min_score <= v <= max_score
        ]
        for k in to_remove:
            del self.sorted_sets[key][k]
        return len(to_remove)
    
    async def zcard(self, key: str) -> int:
        return len(self.sorted_sets.get(key, {}))
    
    async def zrange(self, key: str, start: int, end: int, withscores: bool = False):
        if key not in self.sorted_sets:
            return []
        
        items = sorted(self.sorted_sets[key].items(), key=lambda x: x[1])
        if end == -1:
            end = len(items)
        else:
            end += 1
        
        sliced = items[start:end]
        if withscores:
            return sliced
        return [k for k, v in sliced]
    
    async def zrem(self, key: str, *members: str) -> int:
        if key not in self.sorted_sets:
            return 0
        
        count = 0
        for member in members:
            if member in self.sorted_sets[key]:
                del self.sorted_sets[key][member]
                count += 1
        return count
    
    # Pub/Sub
    async def publish(self, channel: str, message: str) -> int:
        if channel not in self.pubsub_channels:
            return 0
        
        for queue in self.pubsub_channels[channel]:
            await queue.put({"type": "message", "channel": channel, "data": message})
        
        return len(self.pubsub_channels[channel])
    
    def pubsub(self):
        return MockPubSub(self)
    
    def pipeline(self):
        return MockPipeline(self)


class MockPubSub:
    """Mock Pub/Sub handler."""
    
    def __init__(self, redis: MockRedis):
        self.redis = redis
        self.queue = asyncio.Queue()
        self.subscriptions: List[str] = []
    
    async def subscribe(self, channel: str):
        if channel not in self.redis.pubsub_channels:
            self.redis.pubsub_channels[channel] = []
        self.redis.pubsub_channels[channel].append(self.queue)
        self.subscriptions.append(channel)
    
    async def unsubscribe(self, channel: str):
        if channel in self.redis.pubsub_channels:
            if self.queue in self.redis.pubsub_channels[channel]:
                self.redis.pubsub_channels[channel].remove(self.queue)
        if channel in self.subscriptions:
            self.subscriptions.remove(channel)
    
    async def listen(self):
        while True:
            try:
                msg = await asyncio.wait_for(self.queue.get(), timeout=30)
                yield msg
            except asyncio.TimeoutError:
                break
    
    async def close(self):
        for channel in self.subscriptions:
            await self.unsubscribe(channel)


class MockPipeline:
    """Mock pipeline for atomic operations."""
    
    def __init__(self, redis: MockRedis):
        self.redis = redis
        self.commands: List[tuple] = []
    
    def zadd(self, key: str, mapping: Dict[str, float]):
        self.commands.append(("zadd", key, mapping))
        return self
    
    def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        self.commands.append(("zremrangebyscore", key, min_score, max_score))
        return self
    
    def zcard(self, key: str):
        self.commands.append(("zcard", key))
        return self
    
    def expire(self, key: str, seconds: int):
        self.commands.append(("expire", key, seconds))
        return self
    
    def incrby(self, key: str, amount: int):
        self.commands.append(("incrby", key, amount))
        return self
    
    async def execute(self):
        results = []
        for cmd in self.commands:
            if cmd[0] == "zadd":
                await self.redis.zadd(cmd[1], cmd[2])
                results.append(None)
            elif cmd[0] == "zremrangebyscore":
                r = await self.redis.zremrangebyscore(cmd[1], cmd[2], cmd[3])
                results.append(r)
            elif cmd[0] == "zcard":
                r = await self.redis.zcard(cmd[1])
                results.append(r)
            elif cmd[0] == "expire":
                await self.redis.expire(cmd[1], cmd[2])
                results.append(None)
            elif cmd[0] == "incrby":
                r = await self.redis.incrby(cmd[1], cmd[2])
                results.append(r)
        
        self.commands = []
        return results


# ==============================================================================
# LLM CACHE
# ==============================================================================

class LLMCache:
    """
    Cache LLM responses to avoid duplicate API calls.
    
    Theory:
    - Hash request (model + messages) for cache key
    - Only cache deterministic responses (temperature=0)
    - TTL prevents stale responses
    - Saves $$$ on API costs
    """
    
    def __init__(
        self,
        redis: MockRedis,
        prefix: str = "llm:",
        ttl_hours: int = 24
    ):
        self.redis = redis
        self.prefix = prefix
        self.ttl = timedelta(hours=ttl_hours)
    
    def _make_key(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> str:
        """Create deterministic cache key."""
        if temperature > 0:
            return ""  # Don't cache non-deterministic
        
        content = json.dumps({
            "model": model,
            "messages": messages,
        }, sort_keys=True)
        
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:24]
        return f"{self.prefix}{model}:{hash_val}"
    
    async def get(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> Optional[str]:
        """Get cached response."""
        key = self._make_key(model, messages, temperature)
        if not key:
            return None
        
        result = await self.redis.get(key)
        if result:
            await self.redis.incr(f"{self.prefix}hits")
        else:
            await self.redis.incr(f"{self.prefix}misses")
        
        return result
    
    async def set(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        temperature: float = 0.0
    ):
        """Cache a response."""
        key = self._make_key(model, messages, temperature)
        if key:
            await self.redis.setex(key, self.ttl, response)
    
    async def get_or_generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        generate_fn: Callable,
        temperature: float = 0.0
    ) -> tuple:
        """Get from cache or generate and cache."""
        cached = await self.get(model, messages, temperature)
        if cached:
            return cached, True
        
        response = await generate_fn()
        await self.set(model, messages, response, temperature)
        return response, False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hits = await self.redis.get(f"{self.prefix}hits") or "0"
        misses = await self.redis.get(f"{self.prefix}misses") or "0"
        
        total = int(hits) + int(misses)
        hit_rate = int(hits) / total if total > 0 else 0
        
        return {
            "hits": int(hits),
            "misses": int(misses),
            "hit_rate": round(hit_rate, 3)
        }


# ==============================================================================
# SESSION STORE
# ==============================================================================

@dataclass
class SessionMessage:
    """Message in a session."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """Conversation session."""
    session_id: str
    user_id: str
    messages: List[SessionMessage] = field(default_factory=list)
    model: str = "gpt-4"
    created_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    """
    Store conversation sessions in Redis.
    
    Theory:
    - Fast access for real-time chat
    - Auto-expire inactive sessions
    - Serialize to JSON for flexibility
    """
    
    def __init__(
        self,
        redis: MockRedis,
        prefix: str = "session:",
        ttl_hours: int = 24
    ):
        self.redis = redis
        self.prefix = prefix
        self.ttl = timedelta(hours=ttl_hours)
    
    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"
    
    def _serialize(self, session: Session) -> str:
        return json.dumps({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in session.messages
            ]
        })
    
    def _deserialize(self, data: str) -> Session:
        d = json.loads(data)
        return Session(
            session_id=d["session_id"],
            user_id=d["user_id"],
            model=d["model"],
            created_at=datetime.fromisoformat(d["created_at"]),
            messages=[
                SessionMessage(
                    role=m["role"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"])
                )
                for m in d["messages"]
            ]
        )
    
    async def create(self, session_id: str, user_id: str, model: str = "gpt-4") -> Session:
        """Create new session."""
        session = Session(session_id=session_id, user_id=user_id, model=model)
        await self.redis.setex(self._key(session_id), self.ttl, self._serialize(session))
        return session
    
    async def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        data = await self.redis.get(self._key(session_id))
        if data:
            return self._deserialize(data)
        return None
    
    async def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """Add message to session."""
        session = await self.get(session_id)
        if not session:
            return None
        
        session.messages.append(SessionMessage(role=role, content=content))
        await self.redis.setex(self._key(session_id), self.ttl, self._serialize(session))
        return session
    
    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        result = await self.redis.delete(self._key(session_id))
        return result > 0


# ==============================================================================
# RATE LIMITING
# ==============================================================================

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    
    Theory:
    - Uses sorted set with timestamps as scores
    - Remove entries older than window
    - Count remaining entries
    - More accurate than fixed window (no burst at edges)
    """
    
    def __init__(
        self,
        redis: MockRedis,
        prefix: str = "ratelimit:",
        requests_per_minute: int = 60
    ):
        self.redis = redis
        self.prefix = prefix
        self.limit = requests_per_minute
        self.window = 60  # seconds
    
    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"
    
    async def check(self, user_id: str) -> tuple:
        """
        Check if request is allowed.
        
        Returns: (allowed, remaining, reset_seconds)
        """
        key = self._key(user_id)
        now = time.time()
        window_start = now - self.window
        
        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current
        current = await self.redis.zcard(key)
        
        if current >= self.limit:
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            reset = int(oldest[0][1] + self.window - now) if oldest else self.window
            return False, 0, reset
        
        # Add this request
        await self.redis.zadd(key, {str(now): now})
        await self.redis.expire(key, self.window)
        
        return True, self.limit - current - 1, self.window


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for LLM quotas.
    
    Theory:
    - Better for metering usage (tokens) vs requests
    - Bucket refills at constant rate
    - Allows bursts up to bucket size
    """
    
    def __init__(
        self,
        redis: MockRedis,
        prefix: str = "tokenbucket:",
        tokens_per_minute: int = 100000
    ):
        self.redis = redis
        self.prefix = prefix
        self.limit = tokens_per_minute
        self.window = 60
    
    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"
    
    async def consume(self, user_id: str, tokens: int) -> tuple:
        """
        Try to consume tokens.
        
        Returns: (success, remaining)
        """
        key = self._key(user_id)
        
        current = await self.redis.get(key)
        used = int(current) if current else 0
        
        remaining = self.limit - used
        
        if tokens > remaining:
            return False, remaining
        
        await self.redis.incrby(key, tokens)
        await self.redis.expire(key, self.window)
        
        return True, remaining - tokens


# ==============================================================================
# PUB/SUB STREAMING
# ==============================================================================

class TokenPublisher:
    """
    Publish LLM tokens via Pub/Sub.
    
    Use case: Stream tokens to web clients in real-time
    """
    
    def __init__(self, redis: MockRedis, prefix: str = "stream:"):
        self.redis = redis
        self.prefix = prefix
    
    def _channel(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"
    
    async def publish_token(self, session_id: str, token: str):
        await self.redis.publish(
            self._channel(session_id),
            json.dumps({"type": "token", "data": token})
        )
    
    async def publish_done(self, session_id: str):
        await self.redis.publish(
            self._channel(session_id),
            json.dumps({"type": "done"})
        )


class TokenSubscriber:
    """Subscribe to token streams."""
    
    def __init__(self, redis: MockRedis, prefix: str = "stream:"):
        self.redis = redis
        self.prefix = prefix
    
    def _channel(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"
    
    async def subscribe(self, session_id: str) -> AsyncIterator[dict]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._channel(session_id))
        
        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    data = json.loads(msg["data"])
                    yield data
                    if data["type"] in ("done", "error"):
                        break
        finally:
            await pubsub.close()


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate Redis patterns."""
    
    print("=" * 60)
    print("Redis Patterns Demo")
    print("=" * 60)
    
    redis = MockRedis()
    
    # ========== LLM CACHE ==========
    print("\n--- LLM Response Cache ---")
    
    cache = LLMCache(redis)
    messages = [{"role": "user", "content": "What is Python?"}]
    
    async def generate():
        await asyncio.sleep(0.1)  # Simulate API latency
        return "Python is a high-level programming language."
    
    # First call - cache miss
    response1, from_cache1 = await cache.get_or_generate("gpt-4", messages, generate)
    print(f"Call 1: '{response1[:40]}...' (from_cache: {from_cache1})")
    
    # Second call - cache hit
    response2, from_cache2 = await cache.get_or_generate("gpt-4", messages, generate)
    print(f"Call 2: '{response2[:40]}...' (from_cache: {from_cache2})")
    
    stats = await cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # ========== SESSION STORE ==========
    print("\n--- Session Management ---")
    
    sessions = SessionStore(redis)
    
    session = await sessions.create("sess_123", "user_456", "gpt-4")
    print(f"Created session: {session.session_id}")
    
    await sessions.add_message("sess_123", "user", "Hello!")
    await sessions.add_message("sess_123", "assistant", "Hi there!")
    
    loaded = await sessions.get("sess_123")
    print(f"Session has {len(loaded.messages)} messages")
    for msg in loaded.messages:
        print(f"  [{msg.role}]: {msg.content}")
    
    # ========== RATE LIMITING ==========
    print("\n--- Rate Limiting ---")
    
    # Sliding window
    rate_limiter = SlidingWindowRateLimiter(redis, requests_per_minute=5)
    
    print("Sliding window (5 req/min):")
    for i in range(7):
        allowed, remaining, reset = await rate_limiter.check("user_123")
        print(f"  Request {i+1}: allowed={allowed}, remaining={remaining}")
    
    # Token bucket
    token_limiter = TokenBucketRateLimiter(redis, tokens_per_minute=1000)
    
    print("\nToken bucket (1000 tokens/min):")
    for tokens in [300, 400, 500]:
        success, remaining = await token_limiter.consume("user_456", tokens)
        print(f"  Consume {tokens}: success={success}, remaining={remaining}")
    
    # ========== PUB/SUB ==========
    print("\n--- Pub/Sub Streaming ---")
    
    publisher = TokenPublisher(redis)
    subscriber = TokenSubscriber(redis)
    
    async def simulate_stream():
        """Simulate LLM streaming tokens."""
        await asyncio.sleep(0.1)
        for word in ["Hello", "from", "the", "LLM!"]:
            await publisher.publish_token("stream_789", word)
            await asyncio.sleep(0.05)
        await publisher.publish_done("stream_789")
    
    # Start publishing in background
    publish_task = asyncio.create_task(simulate_stream())
    
    # Subscribe and receive
    print("Received tokens: ", end="")
    async for event in subscriber.subscribe("stream_789"):
        if event["type"] == "token":
            print(event["data"], end=" ")
        elif event["type"] == "done":
            print("\n[DONE]")
            break
    
    await publish_task
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- LLM Cache: Hash-based keys, TTL, hit/miss stats
- Sessions: JSON serialization, auto-expire
- Rate Limiting: Sliding window, token bucket
- Pub/Sub: Real-time token streaming
""")


if __name__ == "__main__":
    asyncio.run(demo())
