"""
Module 05, Example 01: Creational Patterns for AI Systems

Covers:
- Singleton: LLM Client connection pool
- Factory Method: Multi-provider LLM factory
- Abstract Factory: Complete AI stack factory
- Builder: Complex prompt/agent configuration
- Prototype: Clone agent configurations

Run this file:
    python 01_creational_patterns.py

Follow along with: 05-design-patterns-complete.md
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from copy import deepcopy
import threading


# =============================================================================
# PATTERN 1: SINGLETON - LLM Connection Pool
# =============================================================================

print("=== Pattern 1: Singleton - LLM Connection Pool ===")


class LLMConnectionPool:
    """Singleton for managing LLM API connections.
    
    Why Singleton here?
    - Only one connection pool needed per application
    - Prevents creating too many API connections
    - Centralizes rate limiting and API key management
    
    Thread-safe implementation using double-checked locking.
    """
    
    _instance: Optional["LLMConnectionPool"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "LLMConnectionPool":
        # Double-checked locking for thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        # Only initialize once
        if self._initialized:
            return
        
        self._connections: Dict[str, Any] = {}
        self._request_count = 0
        self._initialized = True
        print("  LLMConnectionPool initialized (only once)")
    
    def get_connection(self, provider: str) -> str:
        """Get or create connection for provider."""
        if provider not in self._connections:
            self._connections[provider] = f"Connection to {provider}"
            print(f"  Created new connection: {provider}")
        return self._connections[provider]
    
    def make_request(self) -> int:
        """Track request across all connections."""
        self._request_count += 1
        return self._request_count


# Test Singleton
pool1 = LLMConnectionPool()
pool2 = LLMConnectionPool()

print(f"Same instance: {pool1 is pool2}")  # True
pool1.get_connection("openai")
pool2.get_connection("anthropic")  # Uses same pool
print(f"Total connections: {len(pool1._connections)}")


# =============================================================================
# PATTERN 2: FACTORY METHOD - LLM Provider Factory
# =============================================================================

print("\n=== Pattern 2: Factory Method - LLM Provider Factory ===")


class LLMProvider(ABC):
    """Abstract LLM provider."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate completion."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI implementation."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        return f"[OpenAI/{self.model}] Response to: {prompt[:30]}..."
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic implementation."""
    
    def __init__(self, model: str = "claude-3-opus"):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        return f"[Claude/{self.model}] Response to: {prompt[:30]}..."
    
    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class LLMFactory(ABC):
    """Abstract factory for creating LLM providers.
    
    Factory Method Pattern:
    - Defines interface for creating objects
    - Subclasses decide which class to instantiate
    - Allows swapping providers without changing client code
    """
    
    @abstractmethod
    def create_provider(self) -> LLMProvider:
        """Factory method - subclasses implement this."""
        pass
    
    def generate(self, prompt: str) -> str:
        """Template method using the factory."""
        provider = self.create_provider()
        return provider.complete(prompt)


class OpenAIFactory(LLMFactory):
    def create_provider(self) -> LLMProvider:
        return OpenAIProvider()


class AnthropicFactory(LLMFactory):
    def create_provider(self) -> LLMProvider:
        return AnthropicProvider()


# Usage - client code doesn't know concrete class
def process_with_llm(factory: LLMFactory, prompt: str) -> str:
    return factory.generate(prompt)

print(process_with_llm(OpenAIFactory(), "Hello"))
print(process_with_llm(AnthropicFactory(), "Hello"))


# =============================================================================
# PATTERN 3: ABSTRACT FACTORY - Complete AI Stack
# =============================================================================

print("\n=== Pattern 3: Abstract Factory - AI Stack ===")


class VectorStore(ABC):
    """Abstract vector store."""
    
    @abstractmethod
    def store(self, text: str, embedding: List[float]) -> None:
        pass
    
    @abstractmethod
    def search(self, embedding: List[float], k: int) -> List[str]:
        pass


class Embedder(ABC):
    """Abstract embedding model."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


# OpenAI Stack
class OpenAIEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]  # Simulated


class PineconeVectorStore(VectorStore):
    def store(self, text: str, embedding: List[float]) -> None:
        print(f"    Pinecone: Stored '{text[:20]}...'")
    
    def search(self, embedding: List[float], k: int) -> List[str]:
        return [f"Result {i}" for i in range(k)]


# Open Source Stack
class HuggingFaceEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.4, 0.5, 0.6]  # Simulated


class ChromaVectorStore(VectorStore):
    def store(self, text: str, embedding: List[float]) -> None:
        print(f"    Chroma: Stored '{text[:20]}...'")
    
    def search(self, embedding: List[float], k: int) -> List[str]:
        return [f"Local result {i}" for i in range(k)]


class AIStackFactory(ABC):
    """Abstract Factory for complete AI stack.
    
    Creates families of related objects:
    - LLM Provider
    - Embedding Model
    - Vector Store
    
    Ensures components work together.
    """
    
    @abstractmethod
    def create_llm(self) -> LLMProvider:
        pass
    
    @abstractmethod
    def create_embedder(self) -> Embedder:
        pass
    
    @abstractmethod
    def create_vector_store(self) -> VectorStore:
        pass


class OpenAIStackFactory(AIStackFactory):
    """Production stack with OpenAI + Pinecone."""
    
    def create_llm(self) -> LLMProvider:
        return OpenAIProvider("gpt-4")
    
    def create_embedder(self) -> Embedder:
        return OpenAIEmbedder()
    
    def create_vector_store(self) -> VectorStore:
        return PineconeVectorStore()


class OpenSourceStackFactory(AIStackFactory):
    """Local stack with HuggingFace + Chroma."""
    
    def create_llm(self) -> LLMProvider:
        return AnthropicProvider("claude-3-haiku")
    
    def create_embedder(self) -> Embedder:
        return HuggingFaceEmbedder()
    
    def create_vector_store(self) -> VectorStore:
        return ChromaVectorStore()


def build_rag_system(factory: AIStackFactory):
    """Build RAG system from factory - works with any stack."""
    llm = factory.create_llm()
    embedder = factory.create_embedder()
    vector_store = factory.create_vector_store()
    
    print(f"  LLM: {llm.get_model_name()}")
    
    # Simulate RAG
    embedding = embedder.embed("What is Python?")
    vector_store.store("Python is a programming language", embedding)
    results = vector_store.search(embedding, k=3)
    
    return results

print("Production stack:")
build_rag_system(OpenAIStackFactory())

print("\nOpen source stack:")
build_rag_system(OpenSourceStackFactory())


# =============================================================================
# PATTERN 4: BUILDER - Agent Configuration
# =============================================================================

print("\n=== Pattern 4: Builder - Agent Configuration ===")


@dataclass
class AgentConfig:
    """Complex agent configuration."""
    name: str
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = False
    streaming: bool = False
    retry_count: int = 3


class AgentBuilder:
    """Builder for complex agent configuration.
    
    Builder Pattern benefits:
    - Fluent interface for readable configuration
    - Validates configuration before building
    - Separates construction from representation
    """
    
    def __init__(self, name: str):
        self._config = AgentConfig(name=name)
    
    def with_model(self, model: str) -> "AgentBuilder":
        self._config.model = model
        return self
    
    def with_temperature(self, temp: float) -> "AgentBuilder":
        if not 0 <= temp <= 2:
            raise ValueError("Temperature must be 0-2")
        self._config.temperature = temp
        return self
    
    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        self._config.system_prompt = prompt
        return self
    
    def with_tools(self, *tools: str) -> "AgentBuilder":
        self._config.tools = list(tools)
        return self
    
    def with_memory(self) -> "AgentBuilder":
        self._config.memory_enabled = True
        return self
    
    def with_streaming(self) -> "AgentBuilder":
        self._config.streaming = True
        return self
    
    def build(self) -> AgentConfig:
        """Build and validate configuration."""
        if not self._config.system_prompt:
            self._config.system_prompt = f"You are {self._config.name}."
        return self._config


# Fluent interface - reads like English
agent_config = (
    AgentBuilder("Python Tutor")
    .with_model("gpt-4")
    .with_temperature(0.3)
    .with_system_prompt("You are an expert Python tutor.")
    .with_tools("search", "code_execute", "file_read")
    .with_memory()
    .with_streaming()
    .build()
)

print(f"Agent: {agent_config.name}")
print(f"Model: {agent_config.model}")
print(f"Tools: {agent_config.tools}")
print(f"Memory: {agent_config.memory_enabled}")


# =============================================================================
# PATTERN 5: PROTOTYPE - Clone Agent Configs
# =============================================================================

print("\n=== Pattern 5: Prototype - Clone Configurations ===")


@dataclass
class AgentTemplate:
    """Agent template that can be cloned.
    
    Prototype Pattern:
    - Clone existing objects instead of creating from scratch
    - Useful for expensive-to-create configurations
    - Preserves state while allowing customization
    """
    
    name: str
    model: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> "AgentTemplate":
        """Create deep copy of this template."""
        return deepcopy(self)
    
    def with_name(self, name: str) -> "AgentTemplate":
        """Clone with different name."""
        cloned = self.clone()
        cloned.name = name
        return cloned


# Create base template
base_agent = AgentTemplate(
    name="Base Agent",
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
    tools=["search", "calculate"],
    settings={"temperature": 0.7, "max_tokens": 4096}
)

# Clone and customize
research_agent = base_agent.with_name("Research Agent")
research_agent.tools.append("web_scrape")
research_agent.system_prompt = "You are a research assistant."

coding_agent = base_agent.with_name("Coding Agent")
coding_agent.tools = ["code_execute", "file_write", "file_read"]
coding_agent.system_prompt = "You are a coding assistant."

print(f"Base tools: {base_agent.tools}")
print(f"Research tools: {research_agent.tools}")
print(f"Coding tools: {coding_agent.tools}")

# Original unchanged
print(f"Base still has {len(base_agent.tools)} tools")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("CREATIONAL PATTERNS FOR AI SYSTEMS")
print("=" * 60)
print("""
┌────────────────────────────────────────────────────────────┐
│ SINGLETON:                                                  │
│   Use for: Connection pools, API clients, config managers   │
│   AI Use: LLMConnectionPool, EmbeddingCache                 │
├────────────────────────────────────────────────────────────┤
│ FACTORY METHOD:                                             │
│   Use for: Defer instantiation to subclasses                │
│   AI Use: LLMProviderFactory, ToolFactory                   │
├────────────────────────────────────────────────────────────┤
│ ABSTRACT FACTORY:                                           │
│   Use for: Families of related objects                      │
│   AI Use: AIStackFactory (LLM + Embedder + VectorStore)     │
├────────────────────────────────────────────────────────────┤
│ BUILDER:                                                    │
│   Use for: Complex object construction                      │
│   AI Use: AgentBuilder, PromptBuilder                       │
├────────────────────────────────────────────────────────────┤
│ PROTOTYPE:                                                  │
│   Use for: Clone expensive objects                          │
│   AI Use: AgentTemplate.clone(), PromptTemplate.clone()     │
└────────────────────────────────────────────────────────────┘
""")
