"""
Module 03, Example 06: Generators - Streaming Responses

This example demonstrates:
- Generator basics (yield)
- Generator protocol (__iter__, __next__)
- Streaming LLM responses
- Async generators for real-time streaming

Run this file:
    python 06_generators.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

from typing import Generator, Iterator, List, AsyncIterator
import time


# =============================================================================
# PART 1: Generator Basics
# =============================================================================

print("=== Part 1: Generator Basics ===")


def simple_generator():
    """A simple generator function.
    
    Generators use 'yield' instead of 'return'.
    Each yield pauses the function and returns a value.
    The function resumes from where it left off on next().
    """
    print("  Starting generator...")
    yield "First"
    print("  After first yield...")
    yield "Second"
    print("  After second yield...")
    yield "Third"
    print("  Generator exhausted!")


print("Creating generator (nothing runs yet):")
gen = simple_generator()
print(f"Type: {type(gen)}")

print("\nIterating with next():")
print(f"  next(): {next(gen)}")
print(f"  next(): {next(gen)}")
print(f"  next(): {next(gen)}")

# Trying to get another value raises StopIteration
try:
    next(gen)
except StopIteration:
    print("  StopIteration raised (generator exhausted)")


# =============================================================================
# PART 2: Streaming LLM Response
# =============================================================================

print("\n=== Part 2: Streaming LLM Response ===")


def stream_llm_response(prompt: str) -> Generator[str, None, None]:
    """Stream response token by token.
    
    In real implementation, this would yield chunks from the API.
    """
    # Simulate a response
    response = f"Here is my response to your question about {prompt[:20]}. "
    response += "Python is a versatile programming language used in AI, web development, and more."
    
    # Yield word by word (simulating streaming)
    words = response.split()
    for i, word in enumerate(words):
        time.sleep(0.05)  # Simulate network delay
        yield word + " "


print("Streaming response:")
for chunk in stream_llm_response("Python basics"):
    print(chunk, end="", flush=True)
print("\n")


# =============================================================================
# PART 3: Generator Class (Iterator Protocol)
# =============================================================================

print("=== Part 3: Generator Class ===")


class TokenStream:
    """Custom iterator that streams tokens.
    
    Implements the iterator protocol:
    - __iter__(): Returns the iterator (self)
    - __next__(): Returns next value or raises StopIteration
    """
    
    def __init__(self, text: str, delay: float = 0.01):
        self.tokens = text.split()
        self.delay = delay
        self.index = 0
    
    def __iter__(self) -> Iterator[str]:
        """Return iterator (self)."""
        return self
    
    def __next__(self) -> str:
        """Return next token or raise StopIteration."""
        if self.index >= len(self.tokens):
            raise StopIteration
        
        token = self.tokens[self.index]
        self.index += 1
        time.sleep(self.delay)
        return token + " "
    
    def __len__(self) -> int:
        """Return total number of tokens."""
        return len(self.tokens)


# Usage
stream = TokenStream("This is a streaming response from the LLM.")
print(f"Total tokens: {len(stream)}")
print("Streaming: ", end="")
for token in stream:
    print(token, end="", flush=True)
print()


# =============================================================================
# PART 4: Generator Expression
# =============================================================================

print("\n=== Part 4: Generator Expressions ===")

# List comprehension (creates list in memory)
word_list = [word.upper() for word in "hello world from ai".split()]
print(f"List: {word_list}")
print(f"Type: {type(word_list)}")

# Generator expression (lazy, memory efficient)
word_gen = (word.upper() for word in "hello world from ai".split())
print(f"Generator: {word_gen}")
print(f"Type: {type(word_gen)}")

# Generators are lazy - computed on demand
print("Values:", list(word_gen))


# =============================================================================
# PART 5: Combining Generators
# =============================================================================

print("\n=== Part 5: Combining Generators ===")


def generate_messages(conversation: List[dict]) -> Generator[str, None, None]:
    """Generate formatted messages."""
    for msg in conversation:
        yield f"[{msg['role'].upper()}]: {msg['content']}"


def stream_with_thinking(prompt: str) -> Generator[str, None, None]:
    """Stream response with visible thinking."""
    # First, yield thinking tokens
    yield "[Thinking] "
    for word in "Let me analyze this question...".split():
        yield word + " "
        time.sleep(0.02)
    
    yield "\n[Response] "
    
    # Then yield actual response
    for word in f"Based on my analysis of '{prompt[:15]}...', here's what I found.".split():
        yield word + " "
        time.sleep(0.02)


print("Streaming with thinking:")
for chunk in stream_with_thinking("What is machine learning?"):
    print(chunk, end="", flush=True)
print("\n")


# =============================================================================
# PART 6: Async Generator (for real streaming)
# =============================================================================

print("=== Part 6: Async Generators ===")


async def async_stream_response(prompt: str) -> AsyncIterator[str]:
    """Async generator for streaming.
    
    In production, this would yield chunks from async API calls.
    """
    import asyncio
    
    response = f"Async response to: {prompt[:20]}. This streams asynchronously."
    
    for word in response.split():
        await asyncio.sleep(0.02)
        yield word + " "


# Usage pattern (shown as string since we need asyncio.run())
print("""
Async generator usage:

async def main():
    async for chunk in async_stream_response("Hello"):
        print(chunk, end="", flush=True)

asyncio.run(main())
""")


# =============================================================================
# PART 7: Real-World Streaming Pattern
# =============================================================================

print("=== Part 7: Production Streaming Pattern ===")


class StreamingLLMClient:
    """LLM client with streaming support."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> Generator[dict, None, None]:
        """Stream response with metadata.
        
        Yields dictionaries with:
        - delta: The new text chunk
        - type: "token", "usage", or "done"
        - usage: Token counts (on final message)
        """
        # Simulate streaming response
        response = f"Response to {prompt[:20]}: This is a streaming example."
        
        tokens = response.split()
        for i, token in enumerate(tokens[:max_tokens]):
            yield {
                "delta": token + " ",
                "type": "token",
                "index": i
            }
            time.sleep(0.02)
        
        # Final message with usage
        yield {
            "delta": "",
            "type": "done",
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) // 4 + len(tokens)
            }
        }


# Usage
client = StreamingLLMClient()
full_response = ""

print("Streaming with metadata:")
for chunk in client.stream("Explain generators"):
    if chunk["type"] == "token":
        print(chunk["delta"], end="", flush=True)
        full_response += chunk["delta"]
    elif chunk["type"] == "done":
        print(f"\n\n📊 Usage: {chunk['usage']}")


# =============================================================================
# PART 8: Summary
# =============================================================================

print("\n=== Part 8: Summary ===")
print("""
┌────────────────────────────────────────────────────────────┐
│              GENERATOR PATTERNS                             │
├────────────────────────────────────────────────────────────┤
│ FUNCTION GENERATOR:                                         │
│   def gen(): yield value                                    │
│                                                             │
│ GENERATOR EXPRESSION:                                       │
│   (x for x in iterable)                                     │
│                                                             │
│ CLASS ITERATOR:                                             │
│   __iter__(self) -> Iterator                                │
│   __next__(self) -> value                                   │
│                                                             │
│ ASYNC GENERATOR:                                            │
│   async def gen(): yield value                              │
│   async for item in gen():                                  │
├────────────────────────────────────────────────────────────┤
│ AI USE CASES:                                               │
│   ✓ Streaming LLM responses                                 │
│   ✓ Processing large datasets                               │
│   ✓ Real-time token display                                 │
│   ✓ Memory-efficient batch processing                       │
│   ✓ Progressive UI updates                                  │
└────────────────────────────────────────────────────────────┘
""")
