"""
Module 02, Example 03: Polymorphism - Unified Provider Interface

This example demonstrates:
- Method overriding polymorphism
- Duck typing
- Protocol-based structural subtyping
- Writing flexible functions

Run this file:
    python 03_polymorphism.py

Follow along with: 02-oop-principles-multi-provider.md
"""

from typing import List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


# =============================================================================
# PART 1: Base Classes
# =============================================================================

class BaseLLM:
    """Base LLM for inheritance-based polymorphism."""
    
    def __init__(self, model: str):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement complete()")


class OpenAILLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return f"[OpenAI {self.model}] {prompt[:30]}..."


class AnthropicLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return f"[Claude {self.model}] {prompt[:30]}..."


class OllamaLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return f"[Ollama {self.model}] {prompt[:30]}..."


# =============================================================================
# PART 2: Method Overriding Polymorphism
# =============================================================================

def process_with_any_llm(llm: BaseLLM, prompts: List[str]) -> List[str]:
    """Process prompts with ANY LLM provider.
    
    This function demonstrates polymorphism:
    - It accepts any BaseLLM subclass
    - It calls llm.complete() without knowing which provider
    - Each provider's complete() behaves differently
    
    This is the Open/Closed Principle in action:
    - Open for extension (add new providers)
    - Closed for modification (this function never changes)
    """
    responses = []
    for prompt in prompts:
        response = llm.complete(prompt)  # Same method, different behavior!
        responses.append(response)
    return responses


print("=== Part 2: Method Overriding Polymorphism ===")

prompts = ["What is Python?", "Explain OOP"]

# Same function works with all providers
openai_responses = process_with_any_llm(OpenAILLM("gpt-4"), prompts)
claude_responses = process_with_any_llm(AnthropicLLM("claude-3"), prompts)
local_responses = process_with_any_llm(OllamaLLM("llama2"), prompts)

print("OpenAI:", openai_responses[0])
print("Claude:", claude_responses[0])
print("Ollama:", local_responses[0])


# =============================================================================
# PART 3: Duck Typing Polymorphism
# =============================================================================

class CustomLLM:
    """Custom LLM that doesn't inherit from BaseLLM.
    
    In Python, if it has a complete() method, it works!
    "If it walks like a duck and quacks like a duck, it's a duck."
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = "custom"
    
    def complete(self, prompt: str) -> str:
        return f"[Custom {self.name}] {prompt[:30]}..."


print("\n=== Part 3: Duck Typing ===")

# CustomLLM doesn't inherit from BaseLLM, but it works!
custom = CustomLLM("MyModel")
custom_responses = process_with_any_llm(custom, prompts)  # Works!
print("Custom:", custom_responses[0])

print("\nDuck typing means: If it has complete(), it works!")
print(f"CustomLLM inherits from BaseLLM: {issubclass(CustomLLM, BaseLLM)}")


# =============================================================================
# PART 4: Protocol-Based Polymorphism (Structural Subtyping)
# =============================================================================

@runtime_checkable
class Completable(Protocol):
    """Protocol defining what it means to be completable.
    
    Protocols enable structural subtyping:
    - Any class with matching methods satisfies the protocol
    - No inheritance required
    - Can check at runtime with isinstance()
    """
    
    model: str
    
    def complete(self, prompt: str) -> str:
        """Complete a prompt."""
        ...


def process_completable(llm: Completable, prompt: str) -> str:
    """Process with any Completable object.
    
    Type checkers understand this works with ANY class
    that has a 'model' attribute and 'complete()' method.
    """
    return llm.complete(prompt)


print("\n=== Part 4: Protocols ===")

# All these satisfy Completable protocol
providers = [
    OpenAILLM("gpt-4"),
    AnthropicLLM("claude-3"),
    OllamaLLM("llama2"),
    CustomLLM("custom"),
]

for provider in providers:
    is_completable = isinstance(provider, Completable)
    print(f"{type(provider).__name__} is Completable: {is_completable}")


# =============================================================================
# PART 5: Why Polymorphism Matters
# =============================================================================

print("\n=== Part 5: Why Polymorphism Matters ===")
print("""
WITHOUT Polymorphism:
─────────────────────
def process_prompts(llm, prompts):
    if isinstance(llm, OpenAILLM):
        responses = [call_openai(llm, p) for p in prompts]
    elif isinstance(llm, AnthropicLLM):
        responses = [call_anthropic(llm, p) for p in prompts]
    elif isinstance(llm, OllamaLLM):
        responses = [call_ollama(llm, p) for p in prompts]
    # Adding new provider = modify this function!
    return responses

WITH Polymorphism:
──────────────────
def process_prompts(llm, prompts):
    return [llm.complete(p) for p in prompts]
    # Adding new provider = no changes needed!

Benefits:
✓ Open/Closed Principle: Extend without modifying
✓ Single Responsibility: Each class handles itself
✓ Testability: Easy to mock any provider
✓ Flexibility: Swap providers at runtime
""")


# =============================================================================
# PART 6: Operator Polymorphism (Bonus)
# =============================================================================

class ChatMessagePoly:
    """Message with operator overloading polymorphism."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def __add__(self, other: "ChatMessagePoly") -> "ChatMessagePoly":
        """Combine messages with + operator."""
        return ChatMessagePoly(
            role="combined",
            content=f"{self.content}\n{other.content}"
        )
    
    def __str__(self) -> str:
        return f"[{self.role}]: {self.content}"


print("\n=== Part 6: Operator Polymorphism ===")

msg1 = ChatMessagePoly("user", "Hello!")
msg2 = ChatMessagePoly("assistant", "Hi there!")
combined = msg1 + msg2  # Uses __add__

print(f"msg1: {msg1}")
print(f"msg2: {msg2}")
print(f"combined: {combined}")
