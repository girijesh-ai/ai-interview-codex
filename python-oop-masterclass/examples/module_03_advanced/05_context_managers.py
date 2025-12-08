"""
Module 03, Example 05: Context Managers - Conversation Sessions

This example demonstrates:
- Sync context managers (__enter__, __exit__)
- Async context managers (__aenter__, __aexit__)
- The @contextmanager decorator
- Resource management patterns

Run this file:
    python 05_context_managers.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

import uuid
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager


# =============================================================================
# PART 1: Why Context Managers?
# =============================================================================

print("=== Part 1: Why Context Managers? ===")
print("""
Context managers ensure proper setup and cleanup:

WITHOUT context manager:
    file = open("data.txt")
    try:
        data = file.read()
    finally:
        file.close()  # Must remember to close!

WITH context manager:
    with open("data.txt") as file:
        data = file.read()
    # Automatically closed!

For AI applications:
- Manage conversation sessions
- Handle API connections
- Track token usage
- Save conversation history
""")


# =============================================================================
# PART 2: Class-Based Context Manager
# =============================================================================

class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def __repr__(self):
        return f"[{self.role}]: {self.content[:30]}..."


class ConversationSession:
    """Context manager for conversation sessions.
    
    Automatically:
    - Creates session ID on enter
    - Sets up system prompt
    - Saves conversation on exit
    - Handles errors gracefully
    """
    
    def __init__(
        self,
        system_prompt: str,
        save_on_exit: bool = True
    ) -> None:
        self.system_prompt = system_prompt
        self.save_on_exit = save_on_exit
        self.messages: List[ChatMessage] = []
        self._session_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
    
    def __enter__(self) -> "ConversationSession":
        """Called when entering 'with' block.
        
        Returns:
            Self, to be used as the context variable
        """
        self._session_id = str(uuid.uuid4())[:8]
        self._start_time = datetime.now()
        
        # Add system prompt
        self.messages.append(ChatMessage("system", self.system_prompt))
        
        print(f"📝 Started session: {self._session_id}")
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object]
    ) -> bool:
        """Called when exiting 'with' block.
        
        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value
            exc_tb: Traceback
            
        Returns:
            False to propagate exceptions, True to suppress
        """
        duration = (datetime.now() - self._start_time).total_seconds()
        
        if exc_type is not None:
            print(f"❌ Session {self._session_id} failed after {duration:.2f}s")
            print(f"   Error: {exc_val}")
            # Could log error, save partial state, etc.
        else:
            print(f"✅ Session {self._session_id} completed in {duration:.2f}s")
            if self.save_on_exit:
                self._save_conversation()
        
        # Cleanup
        self.messages = []
        
        return False  # Don't suppress exceptions
    
    def send(self, message: str) -> str:
        """Send a message in this session."""
        # Add user message
        self.messages.append(ChatMessage("user", message))
        
        # Simulate LLM response
        response = f"Response to: {message[:20]}..."
        self.messages.append(ChatMessage("assistant", response))
        
        return response
    
    def _save_conversation(self) -> None:
        """Save conversation to storage."""
        print(f"💾 Saved {len(self.messages)} messages to session_{self._session_id}.json")


print("\n=== Part 2: Class-Based Context Manager ===")

# Normal usage
with ConversationSession("You are helpful.") as session:
    response1 = session.send("Hello!")
    response2 = session.send("What is Python?")
    print(f"Got {len(session.messages)} messages")

# Error handling
print("\nWith error:")
try:
    with ConversationSession("System prompt", save_on_exit=False) as session:
        session.send("This works")
        raise ValueError("Something went wrong!")
except ValueError:
    print("(Error was propagated as expected)")


# =============================================================================
# PART 3: Generator-Based Context Manager
# =============================================================================

print("\n=== Part 3: Generator-Based Context Manager ===")


@contextmanager
def token_budget(max_tokens: int):
    """Context manager to track token usage.
    
    Using @contextmanager is simpler than writing a class
    for basic setup/teardown patterns.
    """
    # SETUP (before yield)
    usage = {"input": 0, "output": 0, "total": 0}
    print(f"🎯 Token budget started: {max_tokens} tokens")
    
    try:
        yield usage  # This is what 'as x' receives
        
    except Exception as e:
        # CLEANUP on error
        print(f"❌ Error occurred: {e}")
        raise
        
    finally:
        # CLEANUP (always runs)
        print(f"📊 Token usage: {usage['total']}/{max_tokens}")
        if usage['total'] > max_tokens:
            print(f"⚠️  Budget exceeded by {usage['total'] - max_tokens} tokens!")


# Usage
with token_budget(100) as usage:
    # Simulate some API calls
    usage["input"] += 25
    usage["output"] += 30
    usage["total"] = usage["input"] + usage["output"]
    print(f"After call 1: {usage['total']} tokens")
    
    usage["input"] += 20
    usage["output"] += 40
    usage["total"] = usage["input"] + usage["output"]
    print(f"After call 2: {usage['total']} tokens")


# =============================================================================
# PART 4: Nested Context Managers
# =============================================================================

print("\n=== Part 4: Nested Context Managers ===")


@contextmanager
def api_connection(provider: str):
    """Simulate API connection."""
    print(f"  🔌 Connecting to {provider}...")
    connection = {"provider": provider, "connected": True}
    try:
        yield connection
    finally:
        print(f"  🔌 Disconnecting from {provider}")
        connection["connected"] = False


# Nested contexts
with ConversationSession("Agent assistant") as session:
    with api_connection("OpenAI") as conn:
        with token_budget(200) as budget:
            response = session.send("Complex query")
            budget["total"] += 50


# =============================================================================
# PART 5: Async Context Manager
# =============================================================================

print("\n=== Part 5: Async Context Manager ===")


class AsyncConversationSession:
    """Async context manager for streaming conversations."""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages: List[ChatMessage] = []
    
    async def __aenter__(self) -> "AsyncConversationSession":
        """Async enter."""
        print("🚀 Starting async session...")
        # Could do async setup here (connect to DB, etc.)
        self.messages.append(ChatMessage("system", self.system_prompt))
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object]
    ) -> bool:
        """Async exit."""
        print(f"🏁 Ending async session ({len(self.messages)} messages)")
        # Could do async cleanup (save to DB, etc.)
        return False


@asynccontextmanager
async def async_token_budget(max_tokens: int):
    """Async version of token budget."""
    usage = {"total": 0}
    print(f"🎯 Async budget started: {max_tokens}")
    try:
        yield usage
    finally:
        print(f"📊 Async usage: {usage['total']}/{max_tokens}")


# Demo async (would need asyncio.run() in real code)
print("""
Async context managers work the same way, but with async/await:

async with AsyncConversationSession("System") as session:
    async with async_token_budget(100) as budget:
        response = await session.stream("Query")
        async for chunk in response:
            budget["total"] += 1
""")


# =============================================================================
# PART 6: Summary
# =============================================================================

print("\n=== Part 6: Summary ===")
print("""
┌────────────────────────────────────────────────────────────┐
│              CONTEXT MANAGER PATTERNS                       │
├────────────────────────────────────────────────────────────┤
│ CLASS-BASED:                                                │
│   class Manager:                                            │
│       def __enter__(self) -> self:                          │
│       def __exit__(self, exc_type, exc_val, exc_tb) -> bool │
├────────────────────────────────────────────────────────────┤
│ GENERATOR-BASED (simpler):                                  │
│   @contextmanager                                           │
│   def manager():                                            │
│       setup()                                               │
│       try:                                                  │
│           yield resource                                    │
│       finally:                                              │
│           cleanup()                                         │
├────────────────────────────────────────────────────────────┤
│ AI USE CASES:                                               │
│   ✓ Conversation sessions                                   │
│   ✓ Token budget tracking                                   │
│   ✓ API connections                                         │
│   ✓ Database transactions                                   │
│   ✓ Streaming responses                                     │
└────────────────────────────────────────────────────────────┘
""")
