"""
RAG System Example (Capstone)
==============================
Demonstrates integration of all previous patterns:
- Strategy: Chunker, Embedder
- Repository: VectorStore
- Factory: create_services()
- Dependency Injection: Service composition

Run demo: python a01_rag_system.py
Run tests: pytest test_rag_system.py -v
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import uuid
import hashlib
import math
import asyncio


# ==============================================================================
# DOMAIN MODELS (Pure Python, no dependencies)
# ==============================================================================

@dataclass
class Document:
    """Source document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Document chunk with embedding."""
    id: str
    content: str
    document_id: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Vector search result."""
    chunk: Chunk
    score: float


# ==============================================================================
# INTERFACES (ABC - Strategy Pattern)
# ==============================================================================

class Embedder(ABC):
    """Interface for embedding text."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class VectorStore(ABC):
    """Interface for vector storage (Repository Pattern)."""
    
    @abstractmethod
    async def add(self, chunks: List[Chunk]) -> None:
        pass
    
    @abstractmethod
    async def search(self, embedding: List[float], limit: int = 5) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def count(self) -> int:
        pass


class ChunkingStrategy(ABC):
    """Interface for chunking (Strategy Pattern)."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass


# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================

class MockEmbedder(Embedder):
    """Mock embedder for testing (no API calls)."""
    
    async def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:64]]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]


class InMemoryVectorStore(VectorStore):
    """In-memory vector store."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
    
    async def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
    
    async def search(self, embedding: List[float], limit: int = 5) -> List[SearchResult]:
        scored = []
        for chunk in self.chunks:
            score = self._cosine_similarity(embedding, chunk.embedding)
            scored.append(SearchResult(chunk=chunk, score=score))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]
    
    async def count(self) -> int:
        return len(self.chunks)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4())[:8],
                    content=chunk_text,
                    document_id=document.id,
                    metadata={"start": start, "end": end},
                ))
            
            start = end - self.overlap
        
        return chunks


# ==============================================================================
# SERVICES (Business Logic)
# ==============================================================================

class IngestionService:
    """
    Document ingestion pipeline.
    
    Dependency Injection: All deps passed in constructor.
    """
    
    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedder: Embedder,
        store: VectorStore,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
    
    async def ingest(self, document: Document) -> int:
        """Ingest document: chunk → embed → store."""
        # 1. Chunk
        chunks = self.chunker.chunk(document)
        
        # 2. Embed
        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # 3. Store
        await self.store.add(chunks)
        
        return len(chunks)


class RAGService:
    """
    Retrieval-Augmented Generation service.
    
    Combines retrieval and generation.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm_call: Callable[[str], str],
    ):
        self.embedder = embedder
        self.store = store
        self.llm_call = llm_call
    
    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer question using RAG."""
        # 1. Embed question
        query_embedding = await self.embedder.embed(question)
        
        # 2. Search
        results = await self.store.search(query_embedding, limit=top_k)
        
        # 3. Build context
        context = "\n\n".join([
            f"[{r.chunk.document_id}]: {r.chunk.content}"
            for r in results
        ])
        
        # 4. Generate
        prompt = f"""Answer based on context. If not found, say so.

Context:
{context}

Question: {question}
Answer:"""
        
        answer = await self.llm_call(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [r.chunk.document_id for r in results],
            "chunks_used": len(results),
        }


# ==============================================================================
# FACTORY (Create configured services)
# ==============================================================================

def create_services():
    """
    Factory function to create all services.
    
    Factory Pattern: Centralized creation with configuration.
    """
    # Create components
    chunker = FixedSizeChunker(chunk_size=200, overlap=30)
    embedder = MockEmbedder()
    store = InMemoryVectorStore()
    
    # Mock LLM
    async def mock_llm(prompt: str) -> str:
        if "Python" in prompt:
            return "Python is a programming language known for readability."
        elif "FastAPI" in prompt:
            return "FastAPI is a modern web framework for Python."
        return "Based on the context, I cannot find specific information."
    
    # Create services
    ingestion = IngestionService(chunker, embedder, store)
    rag = RAGService(embedder, store, mock_llm)
    
    return ingestion, rag, store


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate RAG system."""
    
    print("=" * 60)
    print("RAG System Capstone Demo")
    print("=" * 60)
    
    # Create services
    ingestion, rag, store = create_services()
    
    # ========== INGEST DOCUMENTS ==========
    print("\n--- Document Ingestion ---")
    
    documents = [
        Document(
            id="python_basics",
            content="""Python is a high-level programming language created by Guido van Rossum.
It emphasizes code readability and uses significant indentation. Python supports
multiple programming paradigms including procedural, object-oriented, and functional
programming. It has a large standard library and active community.""",
        ),
        Document(
            id="fastapi_intro",
            content="""FastAPI is a modern, fast web framework for building APIs with Python 3.7+.
It uses standard Python type hints for automatic validation and documentation.
Built on Starlette and Pydantic, it provides high performance comparable to
NodeJS and Go. FastAPI supports async/await for concurrent request handling.""",
        ),
        Document(
            id="oop_patterns",
            content="""Design patterns are reusable solutions to common software problems.
The Strategy pattern defines a family of algorithms and makes them interchangeable.
The Factory pattern creates objects without specifying the exact class.
The Repository pattern abstracts data access from business logic.""",
        ),
    ]
    
    for doc in documents:
        chunks_created = await ingestion.ingest(doc)
        print(f"  Ingested '{doc.id}': {chunks_created} chunks")
    
    total = await store.count()
    print(f"  Total chunks in store: {total}")
    
    # ========== RAG QUERIES ==========
    print("\n--- RAG Queries ---")
    
    questions = [
        "What is Python?",
        "How does FastAPI work?",
        "What is the Strategy pattern?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = await rag.query(question, top_k=2)
        print(f"A: {result['answer']}")
        print(f"   Sources: {result['sources']}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Patterns Demonstrated:
- Strategy: ChunkingStrategy, Embedder interfaces
- Repository: VectorStore abstraction
- Factory: create_services() for wiring
- DI: Services receive all dependencies

This is production-ready architecture without over-engineering.
""")


if __name__ == "__main__":
    asyncio.run(demo())
