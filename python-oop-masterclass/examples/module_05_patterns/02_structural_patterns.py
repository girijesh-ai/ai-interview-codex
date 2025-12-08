"""
Module 05, Example 02: Structural Patterns for AI Systems

Covers:
- Adapter: Unify different LLM APIs
- Decorator: Add logging, caching, retry to LLM calls
- Facade: Simple interface for complex RAG system
- Proxy: Lazy loading, caching, access control
- Composite: Nested agent orchestration
- Bridge: Separate LLM abstraction from implementation

Run this file:
    python 02_structural_patterns.py

Follow along with: 05-design-patterns-complete.md
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import functools
import time


# =============================================================================
# PATTERN 1: ADAPTER - Unify Different LLM APIs
# =============================================================================

print("=== Pattern 1: Adapter - Unify LLM APIs ===")


# Target interface our application expects
class LLMClient(ABC):
    """Standard interface for LLM clients."""
    
    @abstractmethod
    def complete(self, messages: List[Dict], **kwargs) -> str:
        pass


# Adaptee 1: OpenAI-style API
class OpenAIAPI:
    """OpenAI's actual API (different interface)."""
    
    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.7
    ) -> Dict:
        return {
            "choices": [{"message": {"content": f"[OpenAI] Response"}}],
            "model": model
        }


# Adaptee 2: Anthropic-style API
class AnthropicAPI:
    """Anthropic's actual API (different interface)."""
    
    def messages_create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict]
    ) -> Dict:
        return {
            "content": [{"text": f"[Claude] Response"}],
            "model": model
        }


# Adapters translate to common interface
class OpenAIAdapter(LLMClient):
    """Adapter for OpenAI API.
    
    Adapter Pattern:
    - Converts interface of a class to another
    - Allows incompatible interfaces to work together
    - Useful for integrating third-party APIs
    """
    
    def __init__(self, api: OpenAIAPI, model: str = "gpt-4"):
        self._api = api
        self._model = model
    
    def complete(self, messages: List[Dict], **kwargs) -> str:
        response = self._api.create_chat_completion(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7)
        )
        return response["choices"][0]["message"]["content"]


class AnthropicAdapter(LLMClient):
    """Adapter for Anthropic API."""
    
    def __init__(self, api: AnthropicAPI, model: str = "claude-3-opus"):
        self._api = api
        self._model = model
    
    def complete(self, messages: List[Dict], **kwargs) -> str:
        response = self._api.messages_create(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages
        )
        return response["content"][0]["text"]


# Usage - same interface for all providers
def chat(client: LLMClient, user_message: str) -> str:
    return client.complete([{"role": "user", "content": user_message}])

openai_client = OpenAIAdapter(OpenAIAPI())
anthropic_client = AnthropicAdapter(AnthropicAPI())

print(f"OpenAI: {chat(openai_client, 'Hello')}")
print(f"Anthropic: {chat(anthropic_client, 'Hello')}")


# =============================================================================
# PATTERN 2: DECORATOR - Enhance LLM Calls
# =============================================================================

print("\n=== Pattern 2: Decorator - Enhance LLM Calls ===")


class LLMService(ABC):
    """Base LLM service interface."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass


class BasicLLMService(LLMService):
    """Basic LLM service."""
    
    def complete(self, prompt: str) -> str:
        return f"Response to: {prompt[:30]}..."


class LLMDecorator(LLMService):
    """Base decorator for LLM services.
    
    Decorator Pattern:
    - Add behavior without modifying original class
    - Stack multiple decorators
    - Single responsibility - each decorator does one thing
    """
    
    def __init__(self, wrapped: LLMService):
        self._wrapped = wrapped
    
    def complete(self, prompt: str) -> str:
        return self._wrapped.complete(prompt)


class LoggingDecorator(LLMDecorator):
    """Add logging to LLM calls."""
    
    def complete(self, prompt: str) -> str:
        print(f"  [LOG] Calling LLM with: {prompt[:20]}...")
        result = self._wrapped.complete(prompt)
        print(f"  [LOG] Got response: {len(result)} chars")
        return result


class CachingDecorator(LLMDecorator):
    """Add caching to LLM calls."""
    
    def __init__(self, wrapped: LLMService):
        super().__init__(wrapped)
        self._cache: Dict[str, str] = {}
    
    def complete(self, prompt: str) -> str:
        if prompt in self._cache:
            print(f"  [CACHE] Hit!")
            return self._cache[prompt]
        
        print(f"  [CACHE] Miss, calling LLM...")
        result = self._wrapped.complete(prompt)
        self._cache[prompt] = result
        return result


class RetryDecorator(LLMDecorator):
    """Add retry logic to LLM calls."""
    
    def __init__(self, wrapped: LLMService, max_retries: int = 3):
        super().__init__(wrapped)
        self._max_retries = max_retries
    
    def complete(self, prompt: str) -> str:
        for attempt in range(self._max_retries):
            try:
                return self._wrapped.complete(prompt)
            except Exception as e:
                print(f"  [RETRY] Attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries - 1:
                    raise
        return ""


# Stack decorators
service = BasicLLMService()
service = CachingDecorator(service)
service = LoggingDecorator(service)
# service = RetryDecorator(service)  # Can add more

print("First call:")
result1 = service.complete("What is Python?")

print("\nSecond call (cached):")
result2 = service.complete("What is Python?")


# =============================================================================
# PATTERN 3: FACADE - Simple RAG Interface
# =============================================================================

print("\n=== Pattern 3: Facade - Simple RAG Interface ===")


# Complex subsystems
class DocumentLoader:
    def load(self, path: str) -> str:
        return f"Content from {path}"


class TextSplitter:
    def split(self, text: str, chunk_size: int = 500) -> List[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


class EmbeddingModel:
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class VectorStore:
    def __init__(self):
        self._data: List[tuple] = []
    
    def add(self, texts: List[str], embeddings: List[List[float]]) -> None:
        for t, e in zip(texts, embeddings):
            self._data.append((t, e))
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[str]:
        return [t for t, e in self._data[:k]]


class LLMForRAG:
    def generate(self, prompt: str, context: List[str]) -> str:
        return f"Based on {len(context)} documents: Answer to query"


class RAGFacade:
    """Facade providing simple interface for complex RAG system.
    
    Facade Pattern:
    - Provides simple interface to complex subsystem
    - Reduces dependencies on internal classes
    - Makes the system easier to use
    """
    
    def __init__(self):
        self._loader = DocumentLoader()
        self._splitter = TextSplitter()
        self._embedder = EmbeddingModel()
        self._vector_store = VectorStore()
        self._llm = LLMForRAG()
        self._indexed = False
    
    def index_document(self, path: str) -> int:
        """Simple method hiding complex pipeline."""
        # Load
        content = self._loader.load(path)
        
        # Split
        chunks = self._splitter.split(content)
        
        # Embed
        embeddings = self._embedder.embed(chunks)
        
        # Store
        self._vector_store.add(chunks, embeddings)
        self._indexed = True
        
        return len(chunks)
    
    def query(self, question: str) -> str:
        """Simple query method."""
        if not self._indexed:
            raise RuntimeError("No documents indexed")
        
        # Embed query
        query_embedding = self._embedder.embed([question])[0]
        
        # Search
        relevant_docs = self._vector_store.search(query_embedding)
        
        # Generate
        return self._llm.generate(question, relevant_docs)


# Simple usage - complex internals hidden
rag = RAGFacade()
chunks = rag.index_document("data/python_docs.pdf")
print(f"Indexed {chunks} chunks")

answer = rag.query("What is Python?")
print(f"Answer: {answer}")


# =============================================================================
# PATTERN 4: PROXY - Control Access to LLM
# =============================================================================

print("\n=== Pattern 4: Proxy - Control LLM Access ===")


class ExpensiveLLMService:
    """Expensive LLM service (like GPT-4)."""
    
    def __init__(self):
        print("  [LLM] Initializing expensive model...")
        self._model = "gpt-4-turbo"
    
    def complete(self, prompt: str) -> str:
        return f"[{self._model}] {prompt[:20]}..."


class LLMProxy:
    """Proxy for expensive LLM service.
    
    Proxy Pattern uses:
    - Virtual Proxy: Lazy initialization
    - Protection Proxy: Access control
    - Caching Proxy: Cache results
    - Logging Proxy: Track usage
    """
    
    def __init__(self, max_requests: int = 10):
        self._llm: Optional[ExpensiveLLMService] = None
        self._request_count = 0
        self._max_requests = max_requests
        self._cache: Dict[str, str] = {}
    
    def _get_llm(self) -> ExpensiveLLMService:
        """Lazy initialization - only create when needed."""
        if self._llm is None:
            print("  [PROXY] First request - creating LLM...")
            self._llm = ExpensiveLLMService()
        return self._llm
    
    def complete(self, prompt: str) -> str:
        # Protection: Rate limiting
        if self._request_count >= self._max_requests:
            raise RuntimeError("Rate limit exceeded")
        
        # Caching
        if prompt in self._cache:
            print("  [PROXY] Cache hit")
            return self._cache[prompt]
        
        # Delegate to real service
        self._request_count += 1
        result = self._get_llm().complete(prompt)
        
        # Cache result
        self._cache[prompt] = result
        
        return result
    
    @property
    def requests_remaining(self) -> int:
        return self._max_requests - self._request_count


# Usage
proxy = LLMProxy(max_requests=5)

print(f"LLM created yet? {proxy._llm is not None}")  # False

result = proxy.complete("Hello world")  # Now creates LLM
print(f"Result: {result}")
print(f"Requests remaining: {proxy.requests_remaining}")


# =============================================================================
# PATTERN 5: COMPOSITE - Nested Agent Orchestration
# =============================================================================

print("\n=== Pattern 5: Composite - Agent Orchestration ===")


class Agent(ABC):
    """Base agent interface.
    
    Composite Pattern:
    - Treat individual objects and compositions uniformly
    - Build tree structures of agents
    - Recursive processing
    """
    
    @abstractmethod
    def execute(self, task: str) -> str:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class LeafAgent(Agent):
    """Single agent (leaf node)."""
    
    def __init__(self, name: str, specialty: str):
        self._name = name
        self._specialty = specialty
    
    def execute(self, task: str) -> str:
        return f"[{self._name}] Executed {self._specialty} task: {task[:20]}"
    
    def get_name(self) -> str:
        return self._name


class AgentTeam(Agent):
    """Team of agents (composite node)."""
    
    def __init__(self, name: str):
        self._name = name
        self._agents: List[Agent] = []
    
    def add(self, agent: Agent) -> None:
        self._agents.append(agent)
    
    def execute(self, task: str) -> str:
        results = []
        print(f"  [{self._name}] Delegating to {len(self._agents)} agents...")
        for agent in self._agents:
            results.append(agent.execute(task))
        return f"[{self._name}] Combined: {len(results)} results"
    
    def get_name(self) -> str:
        return self._name


# Build agent hierarchy
research_agent = LeafAgent("Researcher", "research")
writer_agent = LeafAgent("Writer", "writing")
reviewer_agent = LeafAgent("Reviewer", "review")

content_team = AgentTeam("Content Team")
content_team.add(research_agent)
content_team.add(writer_agent)

full_team = AgentTeam("Full Production Team")
full_team.add(content_team)  # Nested team
full_team.add(reviewer_agent)

# Execute - works uniformly for single agent or team
print("Single agent:")
print(research_agent.execute("Find Python info"))

print("\nFull team (nested):")
print(full_team.execute("Create Python tutorial"))


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("STRUCTURAL PATTERNS FOR AI SYSTEMS")
print("=" * 60)
print("""
┌────────────────────────────────────────────────────────────┐
│ ADAPTER:                                                    │
│   Use for: Make incompatible interfaces work together       │
│   AI Use: Unify OpenAI/Anthropic/Cohere APIs                │
├────────────────────────────────────────────────────────────┤
│ DECORATOR:                                                  │
│   Use for: Add behavior dynamically                         │
│   AI Use: Logging, caching, retry for LLM calls             │
├────────────────────────────────────────────────────────────┤
│ FACADE:                                                     │
│   Use for: Simple interface to complex subsystem            │
│   AI Use: RAGFacade, AgentFacade                            │
├────────────────────────────────────────────────────────────┤
│ PROXY:                                                      │
│   Use for: Control access, lazy init, caching               │
│   AI Use: Rate limiting, model lazy loading                 │
├────────────────────────────────────────────────────────────┤
│ COMPOSITE:                                                  │
│   Use for: Tree structures, recursive processing            │
│   AI Use: Agent teams, nested orchestration                 │
└────────────────────────────────────────────────────────────┘
""")
