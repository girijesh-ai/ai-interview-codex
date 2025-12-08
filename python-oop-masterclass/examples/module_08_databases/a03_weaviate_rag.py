"""
Weaviate Vector Search & RAG Example
=====================================
Demonstrates:
- Weaviate-like vector search patterns
- Chunking strategies for RAG
- Semantic, keyword, and hybrid search
- RAG retrieval and context building

Note: Uses mock implementations for demo without Weaviate server.
In production, install: weaviate-client>=4.7.0

Run with: python a03_weaviate_rag.py
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from abc import ABC, abstractmethod
import hashlib
import re
import math
from datetime import datetime


# ==============================================================================
# DOCUMENT MODELS
# ==============================================================================

@dataclass
class DocumentChunk:
    """A chunk of text from a document."""
    content: str
    source: str
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.embedding:
            self.embedding = self._generate_embedding()
    
    def _generate_embedding(self) -> List[float]:
        """Generate a simple hash-based embedding for demo."""
        h = hashlib.sha256(self.content.encode()).digest()
        return [b / 255.0 for b in h[:16]]


@dataclass
class SearchResult:
    """Result from a vector search."""
    content: str
    source: str
    score: float
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# CHUNKING STRATEGIES
# ==============================================================================

class ChunkingStrategy(ABC):
    """Base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Split text by fixed character count with overlap.
    
    Good for: General purpose, predictable chunk sizes
    
    Theory:
    - Overlap prevents information loss at boundaries
    - Try to break at sentence boundaries when possible
    - Typical sizes: 500-1500 chars depending on embedding model
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                for marker in ['. ', '! ', '? ', '\n\n']:
                    last_break = text.rfind(marker, start, end)
                    if last_break > start + self.chunk_size // 2:
                        end = last_break + len(marker)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_index=index,
                    char_start=start,
                    char_end=end,
                ))
                index += 1
            
            start = end - self.overlap
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Split text by semantic boundaries (paragraphs).
    
    Good for: Preserving context, coherent passages
    
    Theory:
    - Paragraphs are natural semantic units
    - Merge small paragraphs, split large ones
    - Better retrieval quality for Q&A
    """
    
    def __init__(self, min_size: int = 200, max_size: int = 1500):
        self.min_size = min_size
        self.max_size = max_size
    
    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current = ""
        index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Would this exceed max?
            if len(current) + len(para) > self.max_size and current:
                if len(current) >= self.min_size:
                    chunks.append(DocumentChunk(
                        content=current.strip(),
                        source=source,
                        chunk_index=index,
                    ))
                    index += 1
                current = ""
            
            current += para + "\n\n"
        
        # Don't forget last chunk
        if current.strip() and len(current) >= self.min_size:
            chunks.append(DocumentChunk(
                content=current.strip(),
                source=source,
                chunk_index=index,
            ))
        
        return chunks


class MarkdownChunker(ChunkingStrategy):
    """
    Split markdown by headers.
    
    Good for: Technical docs, preserving structure
    
    Theory:
    - Headers indicate topic changes
    - Keep header with its content
    - Enables filtering by section
    """
    
    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
    
    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        # Split by headers
        sections = re.split(r'(^#{1,4}\s+.+$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_header = ""
        current_content = ""
        index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if re.match(r'^#{1,4}\s+', section):
                # Save previous
                if current_content:
                    chunks.append(DocumentChunk(
                        content=f"{current_header}\n{current_content}".strip(),
                        source=source,
                        chunk_index=index,
                        metadata={"header": current_header}
                    ))
                    index += 1
                
                current_header = section
                current_content = ""
            else:
                current_content += section + "\n"
        
        # Last chunk
        if current_content:
            chunks.append(DocumentChunk(
                content=f"{current_header}\n{current_content}".strip(),
                source=source,
                chunk_index=index,
                metadata={"header": current_header}
            ))
        
        return chunks


# ==============================================================================
# VECTOR STORE (Simulated Weaviate)
# ==============================================================================

class VectorStore:
    """
    In-memory vector store simulating Weaviate.
    
    Implements:
    - Cosine similarity search (semantic)
    - BM25 keyword search
    - Hybrid search (combines both)
    """
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the store."""
        self.documents.extend(chunks)
        print(f"Added {len(chunks)} chunks (total: {len(self.documents)})")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0
    
    def _bm25_score(self, query: str, document: str) -> float:
        """
        Simple BM25-like scoring.
        
        BM25 is the standard for keyword search.
        This is a simplified version for demo.
        """
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        doc_term_freq = {}
        
        for term in doc_terms:
            doc_term_freq[term] = doc_term_freq.get(term, 0) + 1
        
        score = 0
        k1 = 1.2
        b = 0.75
        avg_len = 500  # Approximate average
        
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                idf = 1.0  # Simplified
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * len(doc_terms) / avg_len)
                score += idf * numerator / denominator
        
        return score
    
    # ==========================================================================
    # SEARCH METHODS
    # ==========================================================================
    
    def semantic_search(
        self,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Pure vector search by semantic similarity.
        
        How it works:
        1. Embed the query
        2. Find nearest neighbors by cosine similarity
        3. Return top-k results
        """
        # Generate query embedding
        query_embedding = DocumentChunk(
            content=query, source="query", chunk_index=0
        ).embedding
        
        # Score all documents
        scored = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                content=doc.content,
                source=doc.source,
                score=score,
                chunk_index=doc.chunk_index,
            )
            for doc, score in scored[:limit]
        ]
    
    def keyword_search(
        self,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        BM25 keyword search.
        
        How it works:
        1. Tokenize query and documents
        2. Score by term frequency and inverse document frequency
        3. Return top-k results
        """
        scored = []
        for doc in self.documents:
            score = self._bm25_score(query, doc.content)
            if score > 0:
                scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                content=doc.content,
                source=doc.source,
                score=score,
                chunk_index=doc.chunk_index,
            )
            for doc, score in scored[:limit]
        ]
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and keyword.
        
        alpha: Weight between vector (1.0) and keyword (0.0)
        
        Theory:
        - Vector search finds semantically similar content
        - Keyword search finds exact term matches
        - Hybrid gets best of both worlds
        - Typical alpha: 0.5-0.7 for balanced results
        """
        # Get both search results
        semantic = {r.content: r.score for r in self.semantic_search(query, limit * 2)}
        keyword = {r.content: r.score for r in self.keyword_search(query, limit * 2)}
        
        # Normalize scores
        max_sem = max(semantic.values()) if semantic else 1
        max_kw = max(keyword.values()) if keyword else 1
        
        # Combine scores
        all_contents = set(semantic.keys()) | set(keyword.keys())
        combined = []
        
        for content in all_contents:
            sem_score = semantic.get(content, 0) / max_sem
            kw_score = keyword.get(content, 0) / max_kw
            final_score = alpha * sem_score + (1 - alpha) * kw_score
            
            # Find the original document
            doc = next((d for d in self.documents if d.content == content), None)
            if doc:
                combined.append(SearchResult(
                    content=content,
                    source=doc.source,
                    score=final_score,
                    chunk_index=doc.chunk_index,
                ))
        
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:limit]


# ==============================================================================
# RAG RETRIEVER
# ==============================================================================

class RAGRetriever:
    """
    Retrieval-Augmented Generation helper.
    
    RAG Pattern:
    1. User asks a question
    2. Retrieve relevant documents
    3. Build context from documents
    4. Send context + question to LLM
    5. LLM generates answer using context
    """
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    def retrieve_context(
        self,
        query: str,
        max_tokens: int = 3000,
        strategy: str = "hybrid"
    ) -> str:
        """
        Retrieve and format context for LLM.
        
        Returns formatted string with source citations.
        """
        # Choose search strategy
        if strategy == "hybrid":
            results = self.store.hybrid_search(query, limit=10, alpha=0.6)
        elif strategy == "semantic":
            results = self.store.semantic_search(query, limit=10)
        else:
            results = self.store.keyword_search(query, limit=10)
        
        # Build context within token limit (approx 4 chars/token)
        context_parts = []
        char_limit = max_tokens * 4
        total_chars = 0
        
        for result in results:
            if total_chars + len(result.content) > char_limit:
                break
            
            context_parts.append(
                f"[Source: {result.source}]\n{result.content}"
            )
            total_chars += len(result.content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_rag_messages(
        self,
        query: str,
        context: str,
        system_prompt: str = ""
    ) -> List[Dict[str, str]]:
        """
        Build messages for LLM in OpenAI format.
        
        This is the final step before calling the LLM.
        """
        if not system_prompt:
            system_prompt = """You are a helpful assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so.
Always cite sources when possible using [Source: filename] format."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer based on the context above:"""}
        ]


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    """Demonstrate Weaviate-like vector search and RAG."""
    
    print("=" * 60)
    print("Weaviate Vector Search & RAG Demo")
    print("=" * 60)
    
    # Sample documents
    documents = [
        {
            "content": """FastAPI is a modern, high-performance web framework for building APIs 
with Python. It's based on standard Python type hints and provides automatic 
interactive documentation. FastAPI is built on Starlette for web handling and 
Pydantic for data validation. It supports async/await for high concurrency.""",
            "source": "docs/fastapi.md"
        },
        {
            "content": """SQLAlchemy 2.0 introduces native async support through AsyncSession. 
The new declarative syntax uses Mapped[T] for type-safe column definitions. 
The repository pattern helps abstract database operations. Unit of Work 
manages transactions across multiple repositories.""",
            "source": "docs/sqlalchemy.md"
        },
        {
            "content": """Redis is an in-memory data store used for caching, sessions, and 
pub/sub messaging. The redis-py library provides async support through 
redis.asyncio. Common patterns include LLM response caching, rate limiting 
with sliding windows, and real-time token streaming via pub/sub.""",
            "source": "docs/redis.md"
        },
        {
            "content": """Weaviate is a vector database designed for AI applications. It supports 
semantic search using embeddings, hybrid search combining vectors and keywords, 
and built-in RAG capabilities. The v4 Python client uses a collection-based 
API with gRPC for performance.""",
            "source": "docs/weaviate.md"
        },
    ]
    
    # ========== CHUNKING ==========
    print("\n--- Chunking Strategies ---")
    
    long_doc = """
# FastAPI Guide

## Introduction

FastAPI is a modern web framework for building APIs with Python 3.7+.
It offers automatic documentation, type hints, and async support.

## Key Features

FastAPI provides automatic request validation using Pydantic models.
It generates OpenAPI documentation automatically from your code.
Performance is comparable to NodeJS and Go thanks to Starlette and async.

## Getting Started

Install with pip install fastapi uvicorn.
Create a main.py with your endpoints.
Run with uvicorn main:app --reload.
"""
    
    chunkers = [
        ("Fixed Size (500 chars)", FixedSizeChunker(chunk_size=500, overlap=50)),
        ("Semantic (paragraphs)", SemanticChunker(min_size=100, max_size=500)),
        ("Markdown (headers)", MarkdownChunker(max_size=1000)),
    ]
    
    for name, chunker in chunkers:
        chunks = chunker.chunk(long_doc, "fastapi_guide.md")
        print(f"\n{name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            preview = chunk.content[:50].replace('\n', ' ')
            print(f"  [{i}] {preview}...")
    
    # ========== VECTOR STORE ==========
    print("\n--- Vector Store ---")
    
    store = VectorStore()
    chunker = SemanticChunker(min_size=50, max_size=500)
    
    for doc in documents:
        chunks = chunker.chunk(doc["content"], doc["source"])
        store.add_documents(chunks)
    
    # ========== SEARCH ==========
    print("\n--- Search Comparison ---")
    
    query = "web framework for Python APIs"
    
    print(f"\nQuery: '{query}'")
    
    print("\nSemantic Search (vector similarity):")
    for r in store.semantic_search(query, limit=3):
        print(f"  [{r.score:.3f}] {r.source}: {r.content[:60]}...")
    
    print("\nKeyword Search (BM25):")
    for r in store.keyword_search(query, limit=3):
        print(f"  [{r.score:.3f}] {r.source}: {r.content[:60]}...")
    
    print("\nHybrid Search (alpha=0.6):")
    for r in store.hybrid_search(query, limit=3, alpha=0.6):
        print(f"  [{r.score:.3f}] {r.source}: {r.content[:60]}...")
    
    # ========== RAG ==========
    print("\n--- RAG Retrieval ---")
    
    rag = RAGRetriever(store)
    
    question = "How do I cache LLM responses?"
    context = rag.retrieve_context(question, max_tokens=500)
    messages = rag.build_rag_messages(question, context)
    
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved context ({len(context)} chars):")
    print(context[:300] + "...")
    
    print("\nRAG Messages for LLM:")
    print(f"  System: {messages[0]['content'][:80]}...")
    print(f"  User: {messages[1]['content'][:100]}...")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- Chunking: Fixed, Semantic, Markdown strategies
- Vector Store: Embedding + cosine similarity
- Search: Semantic, Keyword (BM25), Hybrid
- RAG: Context retrieval + prompt building
""")


if __name__ == "__main__":
    demo()
