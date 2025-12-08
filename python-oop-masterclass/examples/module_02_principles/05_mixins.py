"""
Module 02, Example 05: Multiple Inheritance & Mixins

This example demonstrates:
- Multiple inheritance
- Mixins for adding capabilities
- Method Resolution Order (MRO)
- The diamond problem

Run this file:
    python 05_mixins.py

Follow along with: 02-oop-principles-multi-provider.md
"""

from typing import List, Dict, Any, AsyncIterator, Iterator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


# =============================================================================
# PART 1: Base LLM Class
# =============================================================================

class BaseLLM:
    """Base class for all LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self._conversation: List[ChatMessage] = []
    
    def complete(self, prompt: str) -> str:
        """Basic completion."""
        return f"[{self.model}] Response to: {prompt[:30]}..."


# =============================================================================
# PART 2: Capability Mixins
# =============================================================================

class StreamingMixin:
    """Mixin that adds streaming capability.
    
    A mixin is a class that:
    - Provides specific functionality
    - Is meant to be combined with other classes
    - Usually doesn't work standalone
    """
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream response token by token."""
        response = f"[Streaming {getattr(self, 'model', 'unknown')}] {prompt[:30]}..."
        for word in response.split():
            yield word + " "


class FunctionCallingMixin:
    """Mixin that adds function/tool calling capability."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools: Dict[str, Any] = {}
    
    def register_tool(self, name: str, func: Any) -> None:
        """Register a tool for the LLM to call."""
        self._tools[name] = func
    
    def complete_with_tools(self, prompt: str) -> Dict[str, Any]:
        """Complete with potential tool calls."""
        return {
            "response": f"Analyzing: {prompt[:30]}...",
            "tool_calls": [{"name": "search", "args": {"query": prompt}}],
            "available_tools": list(self._tools.keys())
        }


class VisionMixin:
    """Mixin that adds image understanding capability."""
    
    def analyze_image(self, image_url: str, prompt: str) -> str:
        """Analyze an image with a prompt."""
        return f"[Vision] Analyzing {image_url}: {prompt[:20]}..."


# =============================================================================
# PART 3: Combining Mixins with Multiple Inheritance
# =============================================================================

class OpenAILLM(StreamingMixin, FunctionCallingMixin, BaseLLM):
    """OpenAI LLM with streaming and function calling.
    
    Inherits from:
    - StreamingMixin: Adds stream() method
    - FunctionCallingMixin: Adds tool registration and calling
    - BaseLLM: Provides base complete() method
    """
    
    def __init__(self, model: str = "gpt-4", **kwargs):
        super().__init__(model=model, **kwargs)


class ClaudeLLM(StreamingMixin, BaseLLM):
    """Claude LLM with streaming only."""
    
    def __init__(self, model: str = "claude-3-opus", **kwargs):
        super().__init__(model=model, **kwargs)


class GPT4VisionLLM(VisionMixin, StreamingMixin, FunctionCallingMixin, BaseLLM):
    """GPT-4 Vision with all capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(model="gpt-4-vision", **kwargs)


print("=== Part 3: Combined Mixins ===")

openai = OpenAILLM()
claude = ClaudeLLM()
gpt4v = GPT4VisionLLM()

# OpenAI has streaming and function calling
print("OpenAI streaming:", end=" ")
for chunk in openai.stream("Hello!"):
    print(chunk, end="")
print()

openai.register_tool("search", lambda q: f"Results for {q}")
print(f"OpenAI tools: {openai.complete_with_tools('Search for Python')}")

# Claude has streaming but NOT function calling
print("\nClaude streaming:", end=" ")
for chunk in claude.stream("Hello!"):
    print(chunk, end="")
print()

# Claude doesn't have register_tool
print(f"Claude has register_tool: {hasattr(claude, 'register_tool')}")

# GPT-4V has everything including vision
print(f"\nGPT-4V vision: {gpt4v.analyze_image('image.jpg', 'Describe this')}")


# =============================================================================
# PART 4: Method Resolution Order (MRO)
# =============================================================================

print("\n=== Part 4: Method Resolution Order ===")

print(f"OpenAILLM MRO:")
for i, cls in enumerate(OpenAILLM.__mro__):
    print(f"  {i}. {cls.__name__}")

print(f"\nGPT4VisionLLM MRO:")
for i, cls in enumerate(GPT4VisionLLM.__mro__):
    print(f"  {i}. {cls.__name__}")

print("""
MRO determines method lookup order:
1. When you call a method, Python searches classes in MRO order
2. First match wins
3. C3 linearization ensures consistent, predictable order
""")


# =============================================================================
# PART 5: The Diamond Problem
# =============================================================================

print("\n=== Part 5: The Diamond Problem ===")


class Base:
    def __init__(self):
        print("Base.__init__")
        self.value = "base"
    
    def method(self):
        return "Base.method"


class Left(Base):
    def __init__(self):
        print("Left.__init__")
        super().__init__()
        self.left_value = "left"
    
    def method(self):
        return "Left.method -> " + super().method()


class Right(Base):
    def __init__(self):
        print("Right.__init__")
        super().__init__()
        self.right_value = "right"
    
    def method(self):
        return "Right.method -> " + super().method()


class Diamond(Left, Right):
    """Diamond inheritance pattern.
    
         Base
        /    \
     Left    Right
        \    /
        Diamond
    
    The "problem": Which path to Base?
    Python's solution: MRO ensures Base is called only once.
    """
    
    def __init__(self):
        print("Diamond.__init__")
        super().__init__()


print("Creating Diamond instance:")
d = Diamond()

print(f"\nMethod resolution:")
print(f"  {d.method()}")

print(f"\nDiamond MRO: {[c.__name__ for c in Diamond.__mro__]}")

print("""
Key insight: super() follows MRO, not just parent class.
In Diamond, super() in Left calls Right (not Base directly).
This ensures Base.__init__ is called only once.
""")


# =============================================================================
# PART 6: Mixin Best Practices
# =============================================================================

print("\n=== Part 6: Mixin Best Practices ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                    MIXIN GUIDELINES                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Single Responsibility: Each mixin adds ONE capability    │
│    ✓ StreamingMixin: Only streaming                         │
│    ✓ VisionMixin: Only image analysis                       │
│                                                              │
│ 2. Use super().__init__: Ensures proper MRO initialization  │
│                                                              │
│ 3. Mixins go BEFORE base class in inheritance:              │
│    ✓ class MyLLM(StreamingMixin, FunctionMixin, BaseLLM)    │
│    ✗ class MyLLM(BaseLLM, StreamingMixin, FunctionMixin)    │
│                                                              │
│ 4. Don't make mixins too dependent on specific bases        │
│    Use getattr() for safe attribute access                   │
│                                                              │
│ 5. Document what mixins expect from the class               │
│    "This mixin expects self.model to be defined"            │
└─────────────────────────────────────────────────────────────┘
""")
