"""
Module 03, Example 04: __slots__ - Memory Optimization

This example demonstrates:
- How __dict__ works (default)
- How __slots__ saves memory
- When to use __slots__
- Trade-offs and limitations

Run this file:
    python 04_slots.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


# =============================================================================
# PART 1: Default Behavior (__dict__)
# =============================================================================

print("=== Part 1: Default __dict__ Behavior ===")


class MessageNormal:
    """Regular class - uses __dict__ for attributes."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


msg = MessageNormal("user", "Hello!")

# Every instance has its own __dict__
print(f"Instance __dict__: {msg.__dict__}")
print(f"Size of __dict__: {sys.getsizeof(msg.__dict__)} bytes")

# You can add attributes dynamically
msg.extra_field = "I can add anything!"
print(f"After adding field: {msg.__dict__}")


# =============================================================================
# PART 2: Using __slots__
# =============================================================================

print("\n=== Part 2: Using __slots__ ===")


class MessageSlots:
    """Optimized class - uses __slots__ for attributes."""
    
    __slots__ = ("role", "content")
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


msg_slotted = MessageSlots("user", "Hello!")

# No __dict__!
try:
    print(msg_slotted.__dict__)
except AttributeError as e:
    print(f"No __dict__: {e}")

# Cannot add dynamic attributes
try:
    msg_slotted.extra_field = "Will this work?"
except AttributeError as e:
    print(f"Cannot add attributes: {e}")


# =============================================================================
# PART 3: Memory Comparison
# =============================================================================

print("\n=== Part 3: Memory Comparison ===")

# Memory tracking
try:
    import tracemalloc
    
    # Test normal class
    tracemalloc.start()
    messages_normal = [MessageNormal("user", f"Message {i}") for i in range(10000)]
    normal_current, normal_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Test slots class
    tracemalloc.start()
    messages_slotted = [MessageSlots("user", f"Message {i}") for i in range(10000)]
    slots_current, slots_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Normal class: {normal_current / 1024:.1f} KB")
    print(f"Slots class:  {slots_current / 1024:.1f} KB")
    print(f"Memory saved: {(1 - slots_current / normal_current) * 100:.1f}%")
except Exception as e:
    print(f"Memory test skipped: {e}")
    print("(tracemalloc may not be available)")


# =============================================================================
# PART 4: Production ChatMessage with __slots__
# =============================================================================

print("\n=== Part 4: Production ChatMessage ===")


class ChatMessage:
    """Memory-efficient chat message for production.
    
    Uses __slots__ for reduced memory footprint when
    handling millions of messages.
    """
    
    __slots__ = (
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_call_id",
        "created_at",
        "_token_count"  # For cached computation
    )
    
    def __init__(
        self,
        role: str,
        content: str,
        name: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None
    ) -> None:
        self.role = role
        self.content = content
        self.name = name
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.created_at = datetime.now()
        self._token_count: Optional[int] = None
    
    @property
    def token_count(self) -> int:
        """Lazy token count with caching."""
        if self._token_count is None:
            self._token_count = len(self.content) // 4
        return self._token_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    def __repr__(self) -> str:
        return f"ChatMessage(role={self.role!r}, content={self.content[:30]!r}...)"


# Usage
msg = ChatMessage("assistant", "Here is the response to your question...")
print(f"Message: {msg}")
print(f"Token count: {msg.token_count}")
print(f"API format: {msg.to_dict()}")


# =============================================================================
# PART 5: Dataclass with slots (Python 3.10+)
# =============================================================================

print("\n=== Part 5: Dataclass with slots ===")

import sys
if sys.version_info >= (3, 10):
    # Only define this class on Python 3.10+
    exec('''
@dataclass(slots=True)
class MessageDataclass:
    """Dataclass with slots - best of both worlds."""
    role: str
    content: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

dc_msg = MessageDataclass("user", "Hello from dataclass!")
print(f"Dataclass message: {dc_msg}")

# Verify no __dict__
try:
    print(dc_msg.__dict__)
except AttributeError:
    print("Dataclass with slots=True has no __dict__ ✓")
''')
else:
    print(f"Dataclass slots=True requires Python 3.10+")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
    print("Showing equivalent manual implementation instead:")
    
    # Show the equivalent manual approach
    class ManualSlotsDataclass:
        """Manual slots dataclass for Python < 3.10"""
        __slots__ = ("role", "content", "created_at")
        
        def __init__(self, role: str, content: str, created_at=None):
            self.role = role
            self.content = content
            self.created_at = created_at or datetime.now()
        
        def __repr__(self):
            return f"ManualSlotsDataclass(role={self.role!r}, content={self.content!r})"
    
    manual_msg = ManualSlotsDataclass("user", "Hello!")
    print(f"Manual slots dataclass: {manual_msg}")


# =============================================================================
# PART 6: __slots__ Inheritance
# =============================================================================

print("\n=== Part 6: Inheritance with __slots__ ===")


class BaseMessage:
    __slots__ = ("role", "content")
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class ExtendedMessage(BaseMessage):
    """Child class extending parent slots."""
    
    __slots__ = ("metadata",)  # Add new slots (don't repeat parent's)
    
    def __init__(self, role: str, content: str, metadata: dict = None):
        super().__init__(role, content)
        self.metadata = metadata or {}


extended = ExtendedMessage("user", "Hello", {"source": "api"})
print(f"Extended has: role={extended.role}, metadata={extended.metadata}")

# Still no __dict__
try:
    print(extended.__dict__)
except AttributeError:
    print("Extended class also has no __dict__ ✓")


# =============================================================================
# PART 7: Trade-offs Summary
# =============================================================================

print("\n=== Part 7: Trade-offs Summary ===")
print("""
┌────────────────────────────────────────────────────────────┐
│                    __slots__ TRADE-OFFS                     │
├────────────────────────────────────────────────────────────┤
│ BENEFITS:                                                   │
│ ✓ ~40% memory reduction per instance                       │
│ ✓ Slightly faster attribute access                         │
│ ✓ Prevents accidental attribute creation                   │
├────────────────────────────────────────────────────────────┤
│ LIMITATIONS:                                                │
│ ✗ No dynamic attribute addition                            │
│ ✗ No __dict__ (can add if needed in slots)                 │
│ ✗ No weak references (add __weakref__ to slots if needed)  │
│ ✗ Multiple inheritance requires care                       │
├────────────────────────────────────────────────────────────┤
│ WHEN TO USE:                                                │
│ ✓ Creating millions of small objects                       │
│ ✓ Memory-constrained environments                          │
│ ✓ Objects with known, fixed attributes                     │
│ ✓ Chat messages, embeddings, tokens in AI applications     │
└────────────────────────────────────────────────────────────┘
""")
