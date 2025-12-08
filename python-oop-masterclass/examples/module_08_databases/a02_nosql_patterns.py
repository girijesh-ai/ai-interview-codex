"""
MongoDB + Weaviate + Redis Example
==================================
Demonstrates NoSQL patterns for AI applications:
- MongoDB for document storage (using PyMongo Async)
- Weaviate for vector search (simulated)
- Redis for caching

Note: This example uses mock implementations for demo purposes.
In production, install: pymongo, weaviate-client, redis

Run with: python a02_nosql_patterns.py
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import hashlib
import asyncio


# ==============================================================================
# MOCK CLIENTS (for demo without actual databases)
# ==============================================================================

class MockMongoCollection:
    """Mock MongoDB collection for demo."""
    
    def __init__(self):
        self.data: Dict[str, dict] = {}
        self._counter = 0
    
    async def insert_one(self, doc: dict) -> object:
        self._counter += 1
        doc_id = f"doc_{self._counter}"
        doc["_id"] = doc_id
        self.data[doc_id] = doc.copy()
        return type("Result", (), {"inserted_id": doc_id})()
    
    async def find_one(self, filter: dict) -> Optional[dict]:
        if "_id" in filter:
            return self.data.get(filter["_id"])
        for doc in self.data.values():
            if all(doc.get(k) == v for k, v in filter.items()):
                return doc
        return None
    
    async def update_one(self, filter: dict, update: dict):
        doc = await self.find_one(filter)
        if doc and "$set" in update:
            doc.update(update["$set"])
        if doc and "$push" in update:
            for key, val in update["$push"].items():
                if key not in doc:
                    doc[key] = []
                doc[key].append(val)
    
    def find(self, filter: dict):
        return MockMongoCursor([
            doc for doc in self.data.values()
            if all(doc.get(k) == v for k, v in filter.items())
        ])


class MockMongoCursor:
    """Mock cursor for iteration."""
    
    def __init__(self, docs: List[dict]):
        self.docs = docs
    
    def sort(self, field: str, direction: int):
        self.docs.sort(
            key=lambda x: x.get(field, ""),
            reverse=(direction == -1)
        )
        return self
    
    def limit(self, n: int):
        self.docs = self.docs[:n]
        return self
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.docs:
            raise StopAsyncIteration
        return self.docs.pop(0)


class MockRedis:
    """Mock Redis for demo."""
    
    def __init__(self):
        self.data: Dict[str, str] = {}
        self.expiry: Dict[str, datetime] = {}
    
    async def get(self, key: str) -> Optional[str]:
        if key in self.expiry and datetime.now() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return None
        return self.data.get(key)
    
    async def setex(self, key: str, ttl: timedelta, value: str):
        self.data[key] = value
        self.expiry[key] = datetime.now() + ttl
    
    async def delete(self, key: str):
        self.data.pop(key, None)
        self.expiry.pop(key, None)


# ==============================================================================
# DOCUMENT MODELS
# ==============================================================================

@dataclass
class Message:
    """Message in a conversation."""
    role: str
    content: str
    tokens: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Conversation:
    """Conversation document."""
    user_id: str
    title: Optional[str] = None
    model: str = "gpt-4"
    messages: List[Message] = None
    total_tokens: int = 0
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "title": self.title,
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "total_tokens": self.total_tokens,
        }


# ==============================================================================
# MONGODB REPOSITORY
# ==============================================================================

class ConversationRepository:
    """Repository for MongoDB conversation operations."""
    
    def __init__(self, collection: MockMongoCollection):
        self.collection = collection
    
    async def create(self, conv: Conversation) -> str:
        """Create a new conversation."""
        result = await self.collection.insert_one(conv.to_dict())
        conv.id = result.inserted_id
        return conv.id
    
    async def get_by_id(self, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        doc = await self.collection.find_one({"_id": conv_id})
        if doc:
            return Conversation(
                id=doc["_id"],
                user_id=doc["user_id"],
                title=doc.get("title"),
                model=doc.get("model", "gpt-4"),
                messages=[
                    Message(**m) if isinstance(m, dict) else m
                    for m in doc.get("messages", [])
                ],
                total_tokens=doc.get("total_tokens", 0)
            )
        return None
    
    async def add_message(self, conv_id: str, message: Message):
        """Add message to conversation."""
        await self.collection.update_one(
            {"_id": conv_id},
            {
                "$push": {"messages": message.to_dict()},
                "$set": {"total_tokens": message.tokens}  # Simplified
            }
        )
    
    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get user's conversations."""
        cursor = self.collection.find({"user_id": user_id}).limit(10)
        result = []
        async for doc in cursor:
            result.append(Conversation(
                id=doc["_id"],
                user_id=doc["user_id"],
                title=doc.get("title"),
                model=doc.get("model", "gpt-4"),
            ))
        return result


# ==============================================================================
# VECTOR STORE (Simulated Weaviate)
# ==============================================================================

@dataclass
class DocumentChunk:
    """Document chunk for vector storage."""
    content: str
    source: str
    embedding: List[float] = None


@dataclass
class SearchResult:
    """Vector search result."""
    content: str
    source: str
    score: float


class VectorStore:
    """Simulated vector store (Weaviate-like)."""
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add documents to store."""
        for chunk in chunks:
            # Simulate embedding generation
            chunk.embedding = self._fake_embed(chunk.content)
        self.documents.extend(chunks)
        print(f"Added {len(chunks)} chunks to vector store")
    
    def _fake_embed(self, text: str) -> List[float]:
        """Fake embedding for demo."""
        # In production: use OpenAI, Cohere, or local model
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h[:8]]
    
    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0
    
    def search(self, query: str, limit: int = 3) -> List[SearchResult]:
        """Semantic search (simulated)."""
        query_embedding = self._fake_embed(query)
        
        results = []
        for doc in self.documents:
            score = self._cosine_sim(query_embedding, doc.embedding)
            results.append(SearchResult(
                content=doc.content,
                source=doc.source,
                score=score
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


# ==============================================================================
# REDIS CACHE
# ==============================================================================

class LLMCache:
    """LLM response cache using Redis."""
    
    def __init__(self, redis: MockRedis, ttl_hours: int = 24):
        self.redis = redis
        self.ttl = timedelta(hours=ttl_hours)
    
    def _make_key(self, model: str, messages: List[dict]) -> str:
        """Create cache key."""
        content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    async def get(self, model: str, messages: List[dict]) -> Optional[str]:
        """Get cached response."""
        return await self.redis.get(self._make_key(model, messages))
    
    async def set(self, model: str, messages: List[dict], response: str):
        """Cache response."""
        await self.redis.setex(self._make_key(model, messages), self.ttl, response)
    
    async def get_or_generate(
        self,
        model: str,
        messages: List[dict],
        generate_fn
    ) -> tuple:
        """Get from cache or generate."""
        cached = await self.get(model, messages)
        if cached:
            return cached, True
        
        response = await generate_fn()
        await self.set(model, messages, response)
        return response, False


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate NoSQL patterns."""
    
    print("=" * 60)
    print("MongoDB + Weaviate + Redis Demo")
    print("=" * 60)
    
    # ========== MongoDB ==========
    print("\n--- MongoDB (Conversation Storage) ---")
    
    collection = MockMongoCollection()
    repo = ConversationRepository(collection)
    
    # Create conversation
    conv = Conversation(
        user_id="user_123",
        title="Python Help",
        model="gpt-4"
    )
    conv_id = await repo.create(conv)
    print(f"Created conversation: {conv_id}")
    
    # Add messages
    await repo.add_message(conv_id, Message(role="user", content="What is FastAPI?", tokens=5))
    await repo.add_message(conv_id, Message(role="assistant", content="FastAPI is a modern web framework...", tokens=20))
    
    # Retrieve
    retrieved = await repo.get_by_id(conv_id)
    print(f"Retrieved: {retrieved.title} with {len(retrieved.messages)} messages")
    
    # ========== Weaviate (Vector Search) ==========
    print("\n--- Weaviate (Vector Search) ---")
    
    vector_store = VectorStore()
    
    # Add documents
    vector_store.add_documents([
        DocumentChunk(content="FastAPI is a modern Python web framework", source="docs/fastapi.md"),
        DocumentChunk(content="SQLAlchemy is an ORM for Python", source="docs/sqlalchemy.md"),
        DocumentChunk(content="Redis is an in-memory data store", source="docs/redis.md"),
    ])
    
    # Search
    results = vector_store.search("web framework for Python")
    print(f"Search results:")
    for r in results:
        print(f"  - {r.source}: {r.content[:40]}... (score: {r.score:.3f})")
    
    # ========== Redis (Caching) ==========
    print("\n--- Redis (LLM Caching) ---")
    
    redis = MockRedis()
    cache = LLMCache(redis)
    
    messages = [{"role": "user", "content": "Hello!"}]
    
    async def generate():
        await asyncio.sleep(0.1)  # Simulate LLM delay
        return "Hello! How can I help you?"
    
    # First call: cache miss
    response1, from_cache1 = await cache.get_or_generate("gpt-4", messages, generate)
    print(f"First call: '{response1}' (from_cache: {from_cache1})")
    
    # Second call: cache hit
    response2, from_cache2 = await cache.get_or_generate("gpt-4", messages, generate)
    print(f"Second call: '{response2}' (from_cache: {from_cache2})")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- MongoDB: Document storage with async operations
- Weaviate: Vector search for RAG (simulated)
- Redis: LLM response caching with TTL
""")


if __name__ == "__main__":
    asyncio.run(demo())
