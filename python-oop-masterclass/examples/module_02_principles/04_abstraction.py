"""
Module 02, Example 04: Abstraction - ABCs and Interfaces

This example demonstrates:
- Abstract Base Classes (ABC)
- Abstract methods and properties
- Defining contracts/interfaces
- Concrete implementations

Run this file:
    python 04_abstraction.py

Follow along with: 02-oop-principles-multi-provider.md
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


# =============================================================================
# PART 1: Defining Abstract Interfaces
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class defining the LLM provider contract.
    
    This is abstraction in action:
    - Defines WHAT providers must do (the interface)
    - Hides HOW they do it (implementation details)
    - All providers MUST implement abstract methods
    - Cannot instantiate this class directly
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name. (Abstract property)"""
        ...
    
    @abstractmethod
    def complete(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> str:
        """Generate a completion. (Abstract method)
        
        All providers must implement this with their own logic.
        """
        ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text. (Abstract method)"""
        ...
    
    # Non-abstract method with default implementation
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost (optional to override)."""
        return 0.0  # Default: unknown cost


class VectorStore(ABC):
    """Abstract base class for vector stores.
    
    Used for semantic search in RAG systems.
    """
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents to the store."""
        ...
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        ...
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        ...


print("=== Part 1: Abstract Interfaces ===")

# Cannot instantiate abstract class
try:
    provider = LLMProvider()
except TypeError as e:
    print(f"Cannot instantiate ABC: {e}")


# =============================================================================
# PART 2: Concrete Implementations
# =============================================================================

class OpenAIProvider(LLMProvider):
    """Concrete OpenAI implementation.
    
    Must implement ALL abstract methods/properties from LLMProvider.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self._api_key = api_key
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Implement abstract property."""
        return self._model
    
    def complete(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> str:
        """Implement abstract method."""
        # Real implementation would call OpenAI API
        last_msg = messages[-1].content if messages else ""
        return f"[OpenAI {self._model}] Response to: {last_msg[:20]}..."
    
    def count_tokens(self, text: str) -> int:
        """Implement abstract method."""
        # Rough estimate (real impl would use tiktoken)
        return len(text) // 4
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Override default with real pricing."""
        # GPT-4 pricing
        return (input_tokens * 0.03 + output_tokens * 0.06) / 1000


class AnthropicProvider(LLMProvider):
    """Concrete Anthropic implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        self._api_key = api_key
        self._model = model
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def complete(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> str:
        last_msg = messages[-1].content if messages else ""
        return f"[Claude {self._model}] Response to: {last_msg[:20]}..."
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        # Claude pricing
        return (input_tokens * 0.015 + output_tokens * 0.075) / 1000


print("\n=== Part 2: Concrete Implementations ===")

openai = OpenAIProvider("sk-key", "gpt-4")
claude = AnthropicProvider("sk-key", "claude-3")

messages = [ChatMessage("user", "What is Python?")]

print(f"OpenAI model: {openai.model_name}")
print(f"OpenAI complete: {openai.complete(messages)}")
print(f"OpenAI cost estimate: ${openai.estimate_cost(1000, 500):.4f}")

print(f"\nClaude model: {claude.model_name}")
print(f"Claude complete: {claude.complete(messages)}")


# =============================================================================
# PART 3: Incomplete Implementation (Error Demo)
# =============================================================================

print("\n=== Part 3: Incomplete Implementation ===")

# This would raise TypeError because not all abstract methods are implemented
# Uncomment to see the error:

# class IncompleteProvider(LLMProvider):
#     """Missing some abstract methods."""
#     
#     @property
#     def model_name(self) -> str:
#         return "incomplete"
#     
#     # Missing: complete(), count_tokens()
# 
# provider = IncompleteProvider()  # TypeError!

print("If a class doesn't implement all abstract methods,")
print("Python raises TypeError when you try to instantiate it.")


# =============================================================================
# PART 4: Using Abstractions
# =============================================================================

def run_agent_with_provider(
    provider: LLMProvider,
    task: str
) -> str:
    """Run an agent with any LLM provider.
    
    This function works with ANY LLMProvider implementation:
    - OpenAI, Anthropic, Ollama, or custom
    - Doesn't need to know internal details
    - Just uses the abstract interface
    """
    messages = [
        ChatMessage("system", "You are a helpful assistant."),
        ChatMessage("user", task)
    ]
    
    # Use abstract methods
    input_tokens = sum(provider.count_tokens(m.content) for m in messages)
    response = provider.complete(messages)
    output_tokens = provider.count_tokens(response)
    cost = provider.estimate_cost(input_tokens, output_tokens)
    
    return f"{response}\n[Tokens: {input_tokens}+{output_tokens}, Cost: ${cost:.4f}]"


print("\n=== Part 4: Using Abstractions ===")

result_openai = run_agent_with_provider(openai, "Explain OOP")
result_claude = run_agent_with_provider(claude, "Explain OOP")

print("OpenAI result:")
print(result_openai)
print("\nClaude result:")
print(result_claude)


# =============================================================================
# PART 5: Why Abstraction Matters
# =============================================================================

print("\n=== Part 5: Why Abstraction Matters ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                                                               │
│   ┌─────────┐     ┌──────────────────┐     ┌───────────┐    │
│   │  Agent  │────▶│ LLMProvider ABC  │────▶│ RAGSystem │    │
│   └─────────┘     └──────────────────┘     └───────────┘    │
│                            │                                  │
│              ┌─────────────┼─────────────┐                   │
│              ▼             ▼             ▼                   │
│        ┌─────────┐   ┌───────────┐  ┌──────────┐            │
│        │ OpenAI  │   │ Anthropic │  │  Ollama  │            │
│        └─────────┘   └───────────┘  └──────────┘            │
│        (hidden)       (hidden)       (hidden)                │
└─────────────────────────────────────────────────────────────┘

Benefits of Abstraction:
✓ Hide complexity: Users don't need to know API details
✓ Enforce contracts: All providers must implement required methods
✓ Swap implementations: Change providers without changing app code
✓ Test easily: Create mock providers for testing
""")
