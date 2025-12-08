"""
Module 01, Example 02: LLM Client with Methods

This example demonstrates:
- Instance methods (operate on self)
- Class methods (@classmethod - alternative constructors)
- Static methods (@staticmethod - utility functions)
- Class vs instance attributes

Run this file:
    python 02_llm_client.py

Follow along with: 01-oop-fundamentals-llm-clients.md
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime


# Import shared types
import sys
sys.path.append(str(__file__).rsplit('/', 2)[0])  # Add examples/ to path
from shared.types import ChatMessage


# =============================================================================
# PART 1: Instance vs Class Attributes
# =============================================================================

class LLMClient:
    """LLM Client demonstrating instance vs class attributes.
    
    Class attributes (defined at class level):
    - Shared by ALL instances
    - Good for: constants, counters, shared configuration
    
    Instance attributes (defined in __init__):
    - Unique to each instance
    - Good for: per-object state, configuration
    """
    
    # ===== CLASS ATTRIBUTES (shared by all instances) =====
    SUPPORTED_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
    DEFAULT_MAX_TOKENS = 4096
    _total_api_calls = 0  # Track calls across ALL clients
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7
    ) -> None:
        """Initialize an LLM client.
        
        Args:
            api_key: API key for authentication
            model: Model identifier
            temperature: Sampling temperature (0.0 to 2.0)
        """
        # ===== INSTANCE ATTRIBUTES (unique to each instance) =====
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.conversation: List[ChatMessage] = []
        self._total_tokens = 0


print("=== Part 1: Instance vs Class Attributes ===")

client1 = LLMClient("key-1", "gpt-4")
client2 = LLMClient("key-2", "gpt-3.5-turbo")

# Instance attributes are unique
print(f"Client 1 model: {client1.model}")  # gpt-4
print(f"Client 2 model: {client2.model}")  # gpt-3.5-turbo

# Class attributes are shared
print(f"Supported models (from client1): {client1.SUPPORTED_MODELS}")
print(f"Supported models (from class): {LLMClient.SUPPORTED_MODELS}")


# =============================================================================
# PART 2: Instance Methods
# =============================================================================

class LLMClientWithMethods(LLMClient):
    """LLM Client with instance methods.
    
    Instance methods:
    - First parameter is 'self' (the instance)
    - Can read and modify instance attributes
    - Can access class attributes via self or class name
    """
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        This is an instance method - it operates on self.conversation
        which is unique to this instance.
        """
        message = ChatMessage(role=role, content=content)
        self.conversation.append(message)
    
    def get_conversation_length(self) -> int:
        """Get number of messages in conversation."""
        return len(self.conversation)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation = []
        self._total_tokens = 0
    
    def send_message(self, content: str) -> str:
        """Send a message and get a response.
        
        In a real implementation, this would call the LLM API.
        """
        # Add user message
        self.add_message("user", content)
        
        # Increment class-level counter
        LLMClientWithMethods._total_api_calls += 1
        
        # Simulate response (real impl would call API)
        response = f"[{self.model}] Response to: {content[:30]}..."
        self.add_message("assistant", response)
        
        return response


print("\n=== Part 2: Instance Methods ===")

client = LLMClientWithMethods("sk-test-key")
client.add_message("system", "You are a helpful assistant.")
response = client.send_message("What is Python?")

print(f"Response: {response}")
print(f"Conversation length: {client.get_conversation_length()}")


# =============================================================================
# PART 3: Class Methods (@classmethod)
# =============================================================================

class LLMClientWithClassMethods(LLMClientWithMethods):
    """LLM Client with class methods.
    
    Class methods:
    - Use @classmethod decorator
    - First parameter is 'cls' (the class itself)
    - Often used for alternative constructors
    - Can access class attributes, but NOT instance attributes
    """
    
    @classmethod
    def from_env(cls, env_var: str = "OPENAI_API_KEY") -> "LLMClientWithClassMethods":
        """Create client from environment variable.
        
        This is a class method acting as an alternative constructor.
        Instead of: client = LLMClient(os.environ["OPENAI_API_KEY"])
        You can do: client = LLMClient.from_env()
        
        Args:
            env_var: Name of environment variable
            
        Returns:
            New LLMClient instance
        """
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls(api_key)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClientWithClassMethods":
        """Create client from configuration dictionary.
        
        Args:
            config: Dict with 'api_key', optional 'model', 'temperature'
        """
        return cls(
            api_key=config["api_key"],
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", 0.7)
        )
    
    @classmethod
    def get_total_api_calls(cls) -> int:
        """Get total API calls across all instances."""
        return cls._total_api_calls


print("\n=== Part 3: Class Methods ===")

# Alternative constructor from config
config = {
    "api_key": "sk-from-config",
    "model": "gpt-4-turbo",
    "temperature": 0.5
}
client = LLMClientWithClassMethods.from_config(config)
print(f"Created from config: model={client.model}, temp={client.temperature}")

# Class method to get class-level data
print(f"Total API calls: {LLMClientWithClassMethods.get_total_api_calls()}")


# =============================================================================
# PART 4: Static Methods (@staticmethod)
# =============================================================================

class LLMClientComplete(LLMClientWithClassMethods):
    """Complete LLM Client with static methods.
    
    Static methods:
    - Use @staticmethod decorator
    - NO 'self' or 'cls' parameter
    - Cannot access instance OR class data
    - Used for utility functions that logically belong to the class
    """
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Estimate token count for text.
        
        This is a static method because:
        - It doesn't need any instance data
        - It doesn't need any class data
        - It's a utility function that logically belongs to LLMClient
        
        Args:
            text: Text to count tokens for
            model: Model (different models have different tokenizers)
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English
        # In production, use tiktoken library
        return len(text) // 4
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format
        """
        if not api_key:
            return False
        # OpenAI keys start with 'sk-'
        return api_key.startswith("sk-") and len(api_key) > 20
    
    @staticmethod
    def estimate_cost(
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4"
    ) -> float:
        """Estimate API call cost.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model being used
            
        Returns:
            Estimated cost in USD
        """
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        rates = pricing.get(model, pricing["gpt-4"])
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        
        return round(input_cost + output_cost, 6)


print("\n=== Part 4: Static Methods ===")

# Static methods can be called without an instance
text = "What is object-oriented programming in Python?"
tokens = LLMClientComplete.count_tokens(text)
print(f"Text: '{text}'")
print(f"Estimated tokens: {tokens}")

# Validate API key
print(f"Valid key 'sk-abc123...xyz': {LLMClientComplete.validate_api_key('sk-abc123456789012345xyz')}")
print(f"Valid key 'invalid': {LLMClientComplete.validate_api_key('invalid')}")

# Estimate cost
cost = LLMClientComplete.estimate_cost(1000, 500, "gpt-4")
print(f"Estimated cost (1000 in, 500 out, gpt-4): ${cost}")


# =============================================================================
# SUMMARY: When to Use Each Method Type
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: WHEN TO USE EACH METHOD TYPE")
print("=" * 60)
print("""
┌─────────────────┬──────────────────────────────────────────┐
│ Method Type     │ When to Use                              │
├─────────────────┼──────────────────────────────────────────┤
│ INSTANCE        │ Needs to read/modify instance data       │
│ def method(self)│ Example: send_message(), add_message()   │
├─────────────────┼──────────────────────────────────────────┤
│ CLASS           │ Alternative constructors                  │
│ @classmethod    │ Operations on class-level data           │
│ def method(cls) │ Example: from_env(), from_config()       │
├─────────────────┼──────────────────────────────────────────┤
│ STATIC          │ Utility functions that don't need        │
│ @staticmethod   │ instance or class data                   │
│ def method()    │ Example: count_tokens(), validate_key()  │
└─────────────────┴──────────────────────────────────────────┘
""")
