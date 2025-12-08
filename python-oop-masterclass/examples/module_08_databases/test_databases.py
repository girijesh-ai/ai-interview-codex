"""
Tests for Module 08 Database Examples
=====================================
Tests for MongoDB, Weaviate, and Redis patterns.

Run with: pytest test_databases.py -v
"""

import pytest
import asyncio
import sys
import os

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a02_nosql_patterns import (
    MockMongoCollection, MockRedis,
    ConversationRepository as MongoConvRepo,
    Conversation as MongoConversation,
    Message as MongoMessage,
    VectorStore, DocumentChunk,
    LLMCache,
)
from a03_weaviate_rag import (
    FixedSizeChunker, SemanticChunker, MarkdownChunker,
    VectorStore as WeaviateStore, SearchResult,
    RAGRetriever, DocumentChunk as WeaviateChunk,
)
from a04_redis_patterns import (
    MockRedis as RedisClient,
    LLMCache as RedisCachePattern,
    SessionStore, Session,
    SlidingWindowRateLimiter, TokenBucketRateLimiter,
)


# ==============================================================================
# MONGODB TESTS (08b)
# ==============================================================================

class TestMongoDBPatterns:
    """Tests for MongoDB document patterns."""
    
    def test_create_conversation(self):
        """Test creating a MongoDB conversation."""
        async def run():
            collection = MockMongoCollection()
            repo = MongoConvRepo(collection)
            
            conv = MongoConversation(user_id="user_123", title="Test Chat")
            conv_id = await repo.create(conv)
            
            assert conv_id is not None
            return conv_id
        
        result = asyncio.run(run())
        assert result is not None
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        async def run():
            collection = MockMongoCollection()
            repo = MongoConvRepo(collection)
            
            conv = MongoConversation(user_id="user_123")
            conv_id = await repo.create(conv)
            
            msg = MongoMessage(role="user", content="Hello!", tokens=5)
            await repo.add_message(conv_id, msg)
            
            retrieved = await repo.get_by_id(conv_id)
            return len(retrieved.messages)
        
        count = asyncio.run(run())
        assert count == 1
    
    def test_get_user_conversations(self):
        """Test retrieving user's conversations."""
        async def run():
            collection = MockMongoCollection()
            repo = MongoConvRepo(collection)
            
            for i in range(3):
                conv = MongoConversation(user_id="user_123", title=f"Chat {i}")
                await repo.create(conv)
            
            convs = await repo.get_user_conversations("user_123")
            return len(convs)
        
        count = asyncio.run(run())
        assert count == 3


# ==============================================================================
# WEAVIATE/VECTOR SEARCH TESTS (08c)
# ==============================================================================

class TestChunkingStrategies:
    """Tests for text chunking strategies."""
    
    @pytest.fixture
    def sample_text(self):
        return """This is the first paragraph with some content.

This is the second paragraph with different content.

This is the third paragraph for testing purposes."""
    
    def test_fixed_size_chunker(self, sample_text):
        """Test fixed size chunking."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(sample_text, "test.md")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, WeaviateChunk) for c in chunks)
    
    def test_semantic_chunker(self, sample_text):
        """Test semantic chunking by paragraphs."""
        chunker = SemanticChunker(min_size=10, max_size=200)
        chunks = chunker.chunk(sample_text, "test.md")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source == "test.md"
    
    def test_markdown_chunker(self):
        """Test markdown header-based chunking."""
        md_text = """# Header 1

Content for header 1.

## Header 2

Content for header 2."""
        
        chunker = MarkdownChunker(max_size=500)
        chunks = chunker.chunk(md_text, "doc.md")
        
        assert len(chunks) == 2
        assert "Header 1" in chunks[0].content
        assert "Header 2" in chunks[1].content


class TestVectorStore:
    """Tests for vector store operations."""
    
    @pytest.fixture
    def store(self):
        store = WeaviateStore()
        chunker = SemanticChunker(min_size=10, max_size=500)
        
        docs = [
            "Python is a programming language for web development.",
            "JavaScript runs in web browsers for interactive content.",
            "Redis is an in-memory database for caching.",
        ]
        
        for i, doc in enumerate(docs):
            chunks = chunker.chunk(doc, f"doc{i}.md")
            store.add_documents(chunks)
        
        return store
    
    def test_semantic_search(self, store):
        """Test vector similarity search."""
        results = store.semantic_search("programming language", limit=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)
    
    def test_keyword_search(self, store):
        """Test BM25 keyword search."""
        results = store.keyword_search("Python", limit=3)
        
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)
    
    def test_hybrid_search(self, store):
        """Test hybrid search combining vector and keyword."""
        results = store.hybrid_search("web development", limit=3, alpha=0.5)
        
        assert len(results) >= 1


class TestRAGRetriever:
    """Tests for RAG retrieval patterns."""
    
    def test_retrieve_context(self):
        """Test context retrieval."""
        store = WeaviateStore()
        chunker = SemanticChunker(min_size=10, max_size=500)
        
        chunks = chunker.chunk("FastAPI is a modern Python web framework.", "fastapi.md")
        store.add_documents(chunks)
        
        retriever = RAGRetriever(store)
        context = retriever.retrieve_context("web framework", max_tokens=500)
        
        assert len(context) > 0
        assert "FastAPI" in context
    
    def test_build_rag_messages(self):
        """Test RAG message building."""
        store = WeaviateStore()
        retriever = RAGRetriever(store)
        
        context = "FastAPI is a web framework."
        messages = retriever.build_rag_messages("What is FastAPI?", context)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ==============================================================================
# REDIS TESTS (08d)
# ==============================================================================

class TestLLMCache:
    """Tests for LLM response caching."""
    
    def test_cache_miss_then_hit(self):
        """Test cache miss followed by cache hit."""
        async def run():
            redis = RedisClient()
            cache = RedisCachePattern(redis)
            messages = [{"role": "user", "content": "Hello"}]
            
            async def generate():
                return "Hi there!"
            
            # First call - miss
            response1, from_cache1 = await cache.get_or_generate(
                "gpt-4", messages, generate
            )
            
            # Second call - hit
            response2, from_cache2 = await cache.get_or_generate(
                "gpt-4", messages, generate
            )
            
            return from_cache1, from_cache2, response1 == response2
        
        from_cache1, from_cache2, same = asyncio.run(run())
        assert from_cache1 is False
        assert from_cache2 is True
        assert same is True
    
    def test_no_cache_for_non_deterministic(self):
        """Test that temperature>0 responses are not cached."""
        async def run():
            redis = RedisClient()
            cache = RedisCachePattern(redis)
            messages = [{"role": "user", "content": "Hello"}]
            
            return await cache.get("gpt-4", messages, temperature=0.7)
        
        result = asyncio.run(run())
        assert result is None


class TestSessionStore:
    """Tests for session management."""
    
    def test_create_session(self):
        """Test session creation."""
        async def run():
            redis = RedisClient()
            store = SessionStore(redis)
            session = await store.create("sess_123", "user_456", "gpt-4")
            return session
        
        session = asyncio.run(run())
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.model == "gpt-4"
    
    def test_add_messages(self):
        """Test adding messages to session."""
        async def run():
            redis = RedisClient()
            store = SessionStore(redis)
            await store.create("sess_123", "user_456")
            
            await store.add_message("sess_123", "user", "Hello!")
            await store.add_message("sess_123", "assistant", "Hi!")
            
            session = await store.get("sess_123")
            return len(session.messages)
        
        count = asyncio.run(run())
        assert count == 2
    
    def test_delete_session(self):
        """Test session deletion."""
        async def run():
            redis = RedisClient()
            store = SessionStore(redis)
            await store.create("sess_123", "user_456")
            
            result = await store.delete("sess_123")
            session = await store.get("sess_123")
            return result, session
        
        deleted, session = asyncio.run(run())
        assert deleted is True
        assert session is None


class TestRateLimiting:
    """Tests for rate limiting patterns."""
    
    def test_sliding_window(self):
        """Test sliding window rate limiter."""
        async def run():
            redis = RedisClient()
            limiter = SlidingWindowRateLimiter(redis, requests_per_minute=3)
            
            results = []
            for _ in range(4):
                allowed, _, _ = await limiter.check("user_123")
                results.append(allowed)
            
            return results
        
        results = asyncio.run(run())
        assert results[:3] == [True, True, True]
        assert results[3] is False
    
    def test_token_bucket(self):
        """Test token bucket rate limiter."""
        async def run():
            redis = RedisClient()
            limiter = TokenBucketRateLimiter(redis, tokens_per_minute=1000)
            
            success1, remaining1 = await limiter.consume("user_123", 500)
            success2, remaining2 = await limiter.consume("user_123", 600)
            
            return success1, remaining1, success2, remaining2
        
        s1, r1, s2, r2 = asyncio.run(run())
        assert s1 is True
        assert r1 == 500
        assert s2 is False


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
