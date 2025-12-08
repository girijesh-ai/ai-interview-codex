"""
Module 02, Example 02: Inheritance - LLM Provider Hierarchy

This example demonstrates:
- Single inheritance
- super() for calling parent methods
- Method overriding
- Extending parent functionality

Run this file:
    python 02_inheritance.py

Follow along with: 02-oop-principles-multi-provider.md
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ChatMessage:
    """Simple chat message."""
    role: str
    content: str


# =============================================================================
# PART 1: Base Class
# =============================================================================

class BaseLLM:
    """Base class for all LLM providers.
    
    This class contains common functionality that ALL providers need:
    - Configuration storage (model, temperature, max_tokens)
    - Conversation management
    - Basic header generation
    
    Child classes inherit all of this and add provider-specific features.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> None:
        """Initialize base LLM."""
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._conversation: List[ChatMessage] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation."""
        self._conversation.append(ChatMessage(role, content))
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation = []
    
    def get_headers(self) -> Dict[str, str]:
        """Get base headers for API requests."""
        return {"Content-Type": "application/json"}
    
    @property
    def conversation_length(self) -> int:
        """Get number of messages."""
        return len(self._conversation)


print("=== Part 1: Base Class ===")
base = BaseLLM("sk-key", "base-model")
base.add_message("user", "Hello!")
print(f"Base model: {base.model}")
print(f"Base headers: {base.get_headers()}")


# =============================================================================
# PART 2: Single Inheritance - Provider Implementations
# =============================================================================

class OpenAILLM(BaseLLM):
    """OpenAI-specific LLM implementation.
    
    Inherits from BaseLLM and adds:
    - OpenAI-specific URL and headers
    - Organization ID support
    - OpenAI-specific complete() method
    """
    
    BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        organization: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize OpenAI LLM."""
        # Call parent's __init__ first
        super().__init__(api_key, model, **kwargs)
        # Add OpenAI-specific attribute
        self.organization = organization
    
    def get_headers(self) -> Dict[str, str]:
        """Get OpenAI-specific headers.
        
        Uses super() to get base headers, then adds OpenAI-specific ones.
        """
        headers = super().get_headers()  # Get parent's headers
        headers["Authorization"] = f"Bearer {self._api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers
    
    def complete(self, prompt: str) -> str:
        """Send completion request to OpenAI."""
        self.add_message("user", prompt)
        # Simulated response
        response = f"[OpenAI {self.model}] Response to: {prompt[:30]}..."
        self.add_message("assistant", response)
        return response


class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation."""
    
    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2024-01-01"
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        **kwargs
    ) -> None:
        super().__init__(api_key, model, **kwargs)
    
    def get_headers(self) -> Dict[str, str]:
        """Anthropic uses different auth headers."""
        headers = super().get_headers()
        headers["x-api-key"] = self._api_key  # Different from OpenAI!
        headers["anthropic-version"] = self.API_VERSION
        return headers
    
    def complete(self, prompt: str) -> str:
        """Send completion request to Anthropic."""
        self.add_message("user", prompt)
        response = f"[Claude {self.model}] Response to: {prompt[:30]}..."
        self.add_message("assistant", response)
        return response


class OllamaLLM(BaseLLM):
    """Local Ollama implementation (no API key needed)."""
    
    def __init__(
        self,
        model: str = "llama2",
        host: str = "http://localhost:11434",
        **kwargs
    ) -> None:
        # Ollama doesn't need an API key
        super().__init__(api_key="", model=model, **kwargs)
        self.host = host
    
    def get_headers(self) -> Dict[str, str]:
        """Ollama needs no auth headers."""
        return super().get_headers()
    
    def complete(self, prompt: str) -> str:
        """Send completion request to local Ollama."""
        self.add_message("user", prompt)
        response = f"[Ollama {self.model}] Response to: {prompt[:30]}..."
        self.add_message("assistant", response)
        return response


print("\n=== Part 2: Provider Implementations ===")

# Create different providers
openai = OpenAILLM("sk-openai-key", organization="org-123")
claude = AnthropicLLM("sk-anthropic-key")
local = OllamaLLM("llama2")

# Each has provider-specific headers
print(f"OpenAI headers: {openai.get_headers()}")
print(f"Anthropic headers: {claude.get_headers()}")
print(f"Ollama headers: {local.get_headers()}")

# Each has same interface
print(f"\nOpenAI: {openai.complete('What is Python?')}")
print(f"Claude: {claude.complete('What is Python?')}")
print(f"Ollama: {local.complete('What is Python?')}")


# =============================================================================
# PART 3: Understanding super()
# =============================================================================

print("\n=== Part 3: How super() Works ===")

class Parent:
    def __init__(self, name: str):
        self.name = name
        print(f"Parent.__init__ called with name={name}")
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"


class Child(Parent):
    def __init__(self, name: str, age: int):
        # MUST call parent's __init__ to initialize parent attributes
        super().__init__(name)  # Calls Parent.__init__(self, name)
        self.age = age
        print(f"Child.__init__ called with age={age}")
    
    def greet(self) -> str:
        # Extend parent's behavior
        parent_greeting = super().greet()  # Get parent's greeting
        return f"{parent_greeting} and I'm {self.age} years old"


child = Child("Alice", 25)
print(child.greet())


# =============================================================================
# PART 4: Inheritance Hierarchy
# =============================================================================

print("\n=== Part 4: Inheritance Hierarchy ===")
print("""
                    BaseLLM
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   OpenAILLM     AnthropicLLM    OllamaLLM
        │
   GPT4TurboLLM (could extend further)

Each child class:
✓ Inherits all parent attributes and methods
✓ Can override methods (like get_headers, complete)
✓ Can add new attributes (like organization, host)
✓ Uses super() to access parent functionality
""")

# Check inheritance
print(f"OpenAILLM inherits from BaseLLM: {issubclass(OpenAILLM, BaseLLM)}")
print(f"openai is instance of BaseLLM: {isinstance(openai, BaseLLM)}")
print(f"openai is instance of OpenAILLM: {isinstance(openai, OpenAILLM)}")

# All share base functionality
print(f"\nAll have conversation_length: {openai.conversation_length}")
