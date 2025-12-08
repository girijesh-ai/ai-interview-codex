"""
Tests for RAG System
=====================
Run with: pytest test_rag_system.py -v
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a01_rag_system import (
    Document, Chunk, SearchResult,
    MockEmbedder, InMemoryVectorStore, FixedSizeChunker,
    IngestionService, RAGService,
    create_services,
)


# ==============================================================================
# DOMAIN MODEL TESTS
# ==============================================================================

class TestDomainModels:
    """Test domain models."""
    
    def test_document_creation(self):
        doc = Document(id="test", content="Hello world")
        assert doc.id == "test"
        assert doc.content == "Hello world"
    
    def test_chunk_creation(self):
        chunk = Chunk(id="c1", content="text", document_id="d1")
        assert chunk.document_id == "d1"


# ==============================================================================
# EMBEDDER TESTS
# ==============================================================================

class TestMockEmbedder:
    """Test embedder interface."""
    
    def test_embed_returns_vector(self):
        async def run():
            embedder = MockEmbedder()
            vec = await embedder.embed("Hello")
            return vec
        
        vec = asyncio.run(run())
        assert len(vec) > 0  # Check non-empty
        assert all(0 <= v <= 1 for v in vec)
    
    def test_embed_batch(self):
        async def run():
            embedder = MockEmbedder()
            vecs = await embedder.embed_batch(["a", "b", "c"])
            return vecs
        
        vecs = asyncio.run(run())
        assert len(vecs) == 3


# ==============================================================================
# VECTOR STORE TESTS
# ==============================================================================

class TestInMemoryVectorStore:
    """Test vector store."""
    
    def test_add_and_count(self):
        async def run():
            store = InMemoryVectorStore()
            chunks = [
                Chunk(id="1", content="a", document_id="d1", embedding=[1.0, 0.0]),
                Chunk(id="2", content="b", document_id="d1", embedding=[0.0, 1.0]),
            ]
            await store.add(chunks)
            return await store.count()
        
        count = asyncio.run(run())
        assert count == 2
    
    def test_search_returns_sorted(self):
        async def run():
            store = InMemoryVectorStore()
            chunks = [
                Chunk(id="1", content="match", document_id="d1", embedding=[1.0, 0.0, 0.0]),
                Chunk(id="2", content="other", document_id="d1", embedding=[0.0, 1.0, 0.0]),
            ]
            await store.add(chunks)
            results = await store.search([1.0, 0.0, 0.0], limit=1)
            return results
        
        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0].chunk.content == "match"


# ==============================================================================
# CHUNKER TESTS
# ==============================================================================

class TestFixedSizeChunker:
    """Test chunking strategy."""
    
    def test_chunks_document(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        doc = Document(id="test", content="A" * 100)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 1
        assert all(c.document_id == "test" for c in chunks)
    
    def test_small_document_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        doc = Document(id="test", content="Hello world")
        chunks = chunker.chunk(doc)
        
        assert len(chunks) == 1


# ==============================================================================
# SERVICE TESTS
# ==============================================================================

class TestIngestionService:
    """Test ingestion pipeline."""
    
    def test_ingest_creates_chunks(self):
        async def run():
            ingestion, _, store = create_services()
            doc = Document(id="test", content="Some content here " * 20)
            count = await ingestion.ingest(doc)
            return count, await store.count()
        
        chunks_created, total = asyncio.run(run())
        assert chunks_created > 0
        assert total == chunks_created


class TestRAGService:
    """Test RAG query."""
    
    def test_query_returns_answer(self):
        async def run():
            ingestion, rag, _ = create_services()
            
            # Ingest a document
            doc = Document(id="python_doc", content="Python is a programming language.")
            await ingestion.ingest(doc)
            
            # Query
            result = await rag.query("What is Python?")
            return result
        
        result = asyncio.run(run())
        assert "question" in result
        assert "answer" in result
        assert "sources" in result


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
