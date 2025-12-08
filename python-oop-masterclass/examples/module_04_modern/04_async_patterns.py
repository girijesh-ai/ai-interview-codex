"""
Module 04, Example 04: Async Patterns for AI

This example demonstrates:
- Async/await basics
- Concurrent LLM calls
- Async context managers
- Async iterators for streaming

Run this file:
    python 04_async_patterns.py

Follow along with: 04-modern-python-ai-features.md
"""

import asyncio
from typing import List, AsyncIterator, Optional
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# PART 1: Why Async for AI?
# =============================================================================

print("=== Part 1: Why Async? ===")
print("""
Synchronous (one at a time):
    [API Call 1: 500ms] → [API Call 2: 500ms] → [API Call 3: 500ms]
    Total: 1500ms

Asynchronous (concurrent):
    [API Call 1: 500ms]
    [API Call 2: 500ms]  (running at same time)
    [API Call 3: 500ms]  (running at same time)
    Total: ~500ms

For I/O-bound work (API calls, DB queries), async is 3x+ faster!
""")


# =============================================================================
# PART 2: Basic Async
# =============================================================================

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class LLMResponse:
    content: str
    model: str
    latency_ms: float = 0

async def call_llm_async(
    prompt: str,
    model: str = "gpt-4",
    delay: float = 0.5
) -> LLMResponse:
    """Simulate async LLM API call."""
    start = datetime.now()
    
    # This is where we'd await the actual API call
    await asyncio.sleep(delay)
    
    latency = (datetime.now() - start).total_seconds() * 1000
    return LLMResponse(
        content=f"Response to: {prompt[:30]}...",
        model=model,
        latency_ms=latency
    )

async def demo_basic_async():
    """Demonstrate basic async/await."""
    print("\n=== Part 2: Basic Async ===")
    
    # Single async call
    response = await call_llm_async("What is Python?")
    print(f"Single call: {response.latency_ms:.0f}ms")
    
    return response

# Run it
result = asyncio.run(demo_basic_async())


# =============================================================================
# PART 3: Concurrent Calls
# =============================================================================

async def demo_concurrent():
    """Demonstrate concurrent async calls."""
    print("\n=== Part 3: Concurrent Calls ===")
    
    prompts = [
        "What is Python?",
        "What is JavaScript?", 
        "What is Rust?",
        "What is Go?",
    ]
    
    # Sequential (slow)
    start = datetime.now()
    sequential_results = []
    for prompt in prompts:
        r = await call_llm_async(prompt, delay=0.2)
        sequential_results.append(r)
    seq_time = (datetime.now() - start).total_seconds() * 1000
    print(f"Sequential (4 calls): {seq_time:.0f}ms")
    
    # Concurrent (fast)
    start = datetime.now()
    tasks = [call_llm_async(p, delay=0.2) for p in prompts]
    concurrent_results = await asyncio.gather(*tasks)
    conc_time = (datetime.now() - start).total_seconds() * 1000
    print(f"Concurrent (4 calls): {conc_time:.0f}ms")
    print(f"Speedup: {seq_time/conc_time:.1f}x faster!")
    
    return concurrent_results

asyncio.run(demo_concurrent())


# =============================================================================
# PART 4: Async LLM Client Class
# =============================================================================

class AsyncLLMClient:
    """Async LLM client with batch processing."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self._request_count = 0
    
    async def complete(self, messages: List[ChatMessage]) -> str:
        """Single completion request."""
        self._request_count += 1
        await asyncio.sleep(0.1)  # Simulate API call
        return f"[{self.model}] Response to: {messages[-1].content[:20]}..."
    
    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Stream response tokens."""
        response = await self.complete(messages)
        for word in response.split():
            await asyncio.sleep(0.02)
            yield word + " "
    
    async def batch_complete(
        self,
        message_lists: List[List[ChatMessage]]
    ) -> List[str]:
        """Process multiple conversations concurrently."""
        tasks = [self.complete(msgs) for msgs in message_lists]
        return await asyncio.gather(*tasks)

async def demo_async_client():
    """Demonstrate async LLM client."""
    print("\n=== Part 4: Async LLM Client ===")
    
    client = AsyncLLMClient("gpt-4")
    
    # Single request
    response = await client.complete([
        ChatMessage("user", "Hello!")
    ])
    print(f"Single: {response}")
    
    # Streaming
    print("Streaming: ", end="")
    async for chunk in client.stream([ChatMessage("user", "Hi there")]):
        print(chunk, end="", flush=True)
    print()
    
    # Batch (concurrent)
    responses = await client.batch_complete([
        [ChatMessage("user", "Question 1")],
        [ChatMessage("user", "Question 2")],
        [ChatMessage("user", "Question 3")],
    ])
    print(f"Batch: {len(responses)} responses")

asyncio.run(demo_async_client())


# =============================================================================
# PART 5: Async Context Manager
# =============================================================================

class AsyncConversation:
    """Async context manager for conversations."""
    
    def __init__(self, client: AsyncLLMClient, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.messages: List[ChatMessage] = []
        self.start_time: Optional[datetime] = None
    
    async def __aenter__(self) -> "AsyncConversation":
        """Async enter - setup."""
        self.start_time = datetime.now()
        self.messages = [ChatMessage("system", self.system_prompt)]
        print(f"  [Conversation started]")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async exit - cleanup."""
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"  [Conversation ended: {len(self.messages)} messages, {duration:.2f}s]")
        # Could save to database, log, etc.
        return False
    
    async def send(self, content: str) -> str:
        """Send message and get response."""
        self.messages.append(ChatMessage("user", content))
        response = await self.client.complete(self.messages)
        self.messages.append(ChatMessage("assistant", response))
        return response

async def demo_async_context():
    """Demonstrate async context manager."""
    print("\n=== Part 5: Async Context Manager ===")
    
    client = AsyncLLMClient()
    
    async with AsyncConversation(client, "You are helpful.") as conv:
        r1 = await conv.send("Hello!")
        print(f"  Response 1: {r1}")
        
        r2 = await conv.send("What is Python?")
        print(f"  Response 2: {r2}")

asyncio.run(demo_async_context())


# =============================================================================
# PART 6: Async Iterator
# =============================================================================

class AsyncTokenStream:
    """Async iterator for token streaming."""
    
    def __init__(self, text: str, delay: float = 0.02):
        self.tokens = text.split()
        self.delay = delay
        self.index = 0
    
    def __aiter__(self) -> "AsyncTokenStream":
        return self
    
    async def __anext__(self) -> str:
        if self.index >= len(self.tokens):
            raise StopAsyncIteration
        
        token = self.tokens[self.index]
        self.index += 1
        await asyncio.sleep(self.delay)
        return token + " "

async def demo_async_iterator():
    """Demonstrate async iterator."""
    print("\n=== Part 6: Async Iterator ===")
    
    stream = AsyncTokenStream("This is a streaming response from the LLM")
    
    print("Async stream: ", end="")
    async for token in stream:
        print(token, end="", flush=True)
    print()

asyncio.run(demo_async_iterator())


# =============================================================================
# PART 7: Summary
# =============================================================================

print("\n=== Part 7: Summary ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                  ASYNC PATTERNS FOR AI                       │
├─────────────────────────────────────────────────────────────┤
│ BASIC ASYNC:                                                 │
│   async def func(): ...                                      │
│   result = await func()                                      │
├─────────────────────────────────────────────────────────────┤
│ CONCURRENT CALLS:                                            │
│   tasks = [func(x) for x in items]                           │
│   results = await asyncio.gather(*tasks)                     │
├─────────────────────────────────────────────────────────────┤
│ ASYNC CONTEXT MANAGER:                                       │
│   async def __aenter__(self): ...                            │
│   async def __aexit__(self, ...): ...                        │
│   async with Manager() as m: ...                             │
├─────────────────────────────────────────────────────────────┤
│ ASYNC ITERATOR:                                              │
│   def __aiter__(self): return self                           │
│   async def __anext__(self): ...                             │
│   async for item in iterator: ...                            │
├─────────────────────────────────────────────────────────────┤
│ AI USE CASES:                                                │
│   ✓ Parallel LLM calls (3-10x faster)                        │
│   ✓ Token streaming                                          │
│   ✓ Concurrent tool execution                                │
│   ✓ Batch processing                                         │
└─────────────────────────────────────────────────────────────┘
""")
