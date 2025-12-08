"""
Module 01, Example 04: Special Methods (Dunder Methods)

This example demonstrates:
- __str__ and __repr__ for string representation
- __eq__ and __hash__ for equality and hashing
- __len__ for length
- __bool__ for truthiness
- __add__ for concatenation
- __iter__ for iteration

Run this file:
    python 04_dunder_methods.py

Follow along with: 01-oop-fundamentals-llm-clients.md
"""

from datetime import datetime
from typing import List, Iterator, Optional


# =============================================================================
# PART 1: String Representation (__str__ and __repr__)
# =============================================================================

class ChatMessage:
    """Chat message with proper string representations.
    
    __str__:  User-friendly output (print(), str())
    __repr__: Developer-friendly output (debugging, REPL)
    
    Rule of thumb:
    - __str__ should be readable
    - __repr__ should be unambiguous (ideally, evaluatable)
    """
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.created_at = datetime.now()
    
    def __str__(self) -> str:
        """User-friendly string for display.
        
        Called by: print(msg), str(msg), f"{msg}"
        """
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role.upper()}]: {preview}"
    
    def __repr__(self) -> str:
        """Developer-friendly string for debugging.
        
        Called by: repr(msg), displaying in REPL, in lists
        
        Goal: If possible, this should be evaluatable Python code
        that could recreate the object.
        """
        return f"ChatMessage(role={self.role!r}, content={self.content!r})"


print("=== Part 1: String Representation ===")

msg = ChatMessage("user", "What is Python?")

# __str__ is for end users
print(f"str(): {str(msg)}")
print(f"print(): {msg}")

# __repr__ is for developers
print(f"repr(): {repr(msg)}")

# In lists, __repr__ is used
messages = [
    ChatMessage("user", "Hello"),
    ChatMessage("assistant", "Hi there!")
]
print(f"In list: {messages}")


# =============================================================================
# PART 2: Equality and Hashing (__eq__ and __hash__)
# =============================================================================

class ChatMessageEquatable:
    """Chat message with equality support.
    
    __eq__:   Define when two objects are "equal"
    __hash__: Required if you want to use in sets/dicts
    
    Key rule: If two objects are equal, they MUST have the same hash.
    """
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on role and content.
        
        Called by: msg1 == msg2, msg1 != msg2 (negated)
        """
        if not isinstance(other, ChatMessageEquatable):
            return NotImplemented  # Let Python handle it
        return self.role == other.role and self.content == other.content
    
    def __hash__(self) -> int:
        """Generate hash for use in sets/dicts.
        
        Called by: hash(msg), when used as dict key or in set
        
        Must be consistent with __eq__:
        If a == b, then hash(a) == hash(b)
        """
        return hash((self.role, self.content))
    
    def __repr__(self) -> str:
        return f"ChatMessage({self.role!r}, {self.content!r})"


print("\n=== Part 2: Equality ===")

msg1 = ChatMessageEquatable("user", "Hello")
msg2 = ChatMessageEquatable("user", "Hello")
msg3 = ChatMessageEquatable("user", "Goodbye")

# Equality
print(f"msg1 == msg2: {msg1 == msg2}")  # True (same content)
print(f"msg1 == msg3: {msg1 == msg3}")  # False (different content)
print(f"msg1 is msg2: {msg1 is msg2}")  # False (different objects)

# Usable in sets and dicts
unique_messages = {msg1, msg2, msg3}  # msg1 and msg2 deduplicated
print(f"Unique messages: {len(unique_messages)}")  # 2


# =============================================================================
# PART 3: Length and Boolean (__len__ and __bool__)
# =============================================================================

class Conversation:
    """Conversation with length and boolean support.
    
    __len__:  Define what len() returns
    __bool__: Define when object is truthy
    
    Note: If __bool__ is not defined, Python uses __len__ > 0
    """
    
    def __init__(self, messages: Optional[List[ChatMessageEquatable]] = None):
        self.messages = messages or []
    
    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessageEquatable(role, content))
    
    def __len__(self) -> int:
        """Return number of messages.
        
        Called by: len(conversation)
        """
        return len(self.messages)
    
    def __bool__(self) -> bool:
        """Return True if conversation has messages.
        
        Called by: if conversation:, bool(conversation)
        
        If not defined, Python would use len(self) > 0
        """
        return len(self.messages) > 0
    
    def __repr__(self) -> str:
        return f"Conversation({len(self)} messages)"


print("\n=== Part 3: Length and Boolean ===")

empty_conv = Conversation()
conv = Conversation()
conv.add("system", "You are helpful.")
conv.add("user", "Hello!")

# __len__
print(f"len(empty): {len(empty_conv)}")
print(f"len(conv): {len(conv)}")

# __bool__
print(f"bool(empty): {bool(empty_conv)}")
print(f"bool(conv): {bool(conv)}")

# Use in if statements
if conv:
    print("Conversation has messages!")
if not empty_conv:
    print("Empty conversation is falsy")


# =============================================================================
# PART 4: Iteration (__iter__ and __next__)
# =============================================================================

class IterableConversation:
    """Conversation that can be iterated.
    
    __iter__: Return an iterator object
    
    This enables: for msg in conversation:
    """
    
    def __init__(self):
        self.messages: List[ChatMessageEquatable] = []
    
    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessageEquatable(role, content))
    
    def __iter__(self) -> Iterator[ChatMessageEquatable]:
        """Return iterator over messages.
        
        Called by: for msg in conversation:, list(conversation)
        """
        return iter(self.messages)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"Conversation({len(self)} messages)"


print("\n=== Part 4: Iteration ===")

conv = IterableConversation()
conv.add("system", "You are helpful.")
conv.add("user", "What is Python?")
conv.add("assistant", "Python is a programming language.")

# Iterate with for loop
print("Messages:")
for msg in conv:
    print(f"  {msg}")

# Can use list() to convert
all_msgs = list(conv)
print(f"As list: {len(all_msgs)} messages")


# =============================================================================
# PART 5: Operators (__add__, __contains__)
# =============================================================================

class CombinableConversation:
    """Conversation with operator support.
    
    __add__:      Enable + operator
    __contains__: Enable 'in' operator
    """
    
    def __init__(self, messages: Optional[List[ChatMessageEquatable]] = None):
        self.messages = list(messages) if messages else []
    
    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessageEquatable(role, content))
    
    def __add__(self, other: "CombinableConversation") -> "CombinableConversation":
        """Combine two conversations.
        
        Called by: conv1 + conv2
        """
        if not isinstance(other, CombinableConversation):
            return NotImplemented
        return CombinableConversation(self.messages + other.messages)
    
    def __contains__(self, role: str) -> bool:
        """Check if conversation contains a role.
        
        Called by: "user" in conversation
        """
        return any(msg.role == role for msg in self.messages)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"Conversation({len(self)} messages)"


print("\n=== Part 5: Operators ===")

conv1 = CombinableConversation()
conv1.add("system", "You are helpful.")

conv2 = CombinableConversation()
conv2.add("user", "Hello!")
conv2.add("assistant", "Hi there!")

# __add__: Combine with +
combined = conv1 + conv2
print(f"conv1: {conv1}")
print(f"conv2: {conv2}")
print(f"combined: {combined}")

# __contains__: Check with 'in'
print(f"'user' in combined: {'user' in combined}")
print(f"'tool' in combined: {'tool' in combined}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("COMMON DUNDER METHODS")
print("=" * 60)
print("""
┌──────────────────┬──────────────────────────────────────────┐
│ Method           │ Called By                                │
├──────────────────┼──────────────────────────────────────────┤
│ __init__(self)   │ Constructor: MyClass()                   │
│ __str__(self)    │ str(), print(), f"{}"                    │
│ __repr__(self)   │ repr(), debugging, in lists              │
│ __eq__(self, o)  │ ==, !=                                   │
│ __hash__(self)   │ hash(), dict keys, sets                  │
│ __len__(self)    │ len()                                    │
│ __bool__(self)   │ if obj:, bool()                          │
│ __iter__(self)   │ for x in obj:, list()                    │
│ __add__(self, o) │ +                                        │
│ __contains__(o)  │ x in obj                                 │
│ __getitem__(k)   │ obj[key]                                 │
│ __setitem__(k,v) │ obj[key] = value                         │
│ __call__(*args)  │ obj()  (make object callable)            │
└──────────────────┴──────────────────────────────────────────┘
""")
