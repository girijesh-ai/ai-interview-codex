"""
Module 01, Example 01: ChatMessage Class Basics

This example demonstrates fundamental OOP concepts:
- Defining classes
- The __init__ constructor
- Instance attributes
- Creating objects (instances)

Run this file:
    python 01_chat_message.py

Follow along with: 01-oop-fundamentals-llm-clients.md
"""

from datetime import datetime
from typing import Optional


# =============================================================================
# PART 1: Your First Class
# =============================================================================

class ChatMessageBasic:
    """A basic chat message - the simplest possible class.
    
    This demonstrates:
    - Class definition syntax
    - pass keyword for empty class
    """
    pass


# Creating an instance of an empty class
basic_message = ChatMessageBasic()
print("=== Part 1: Basic Class ===")
print(f"Type: {type(basic_message)}")
print(f"Is instance: {isinstance(basic_message, ChatMessageBasic)}")


# =============================================================================
# PART 2: Adding the __init__ Constructor
# =============================================================================

class ChatMessageWithInit:
    """Chat message with constructor.
    
    The __init__ method is called automatically when you create an instance.
    It's where you initialize the object's attributes.
    
    Think of __init__ as the "setup" method that runs once when the object
    is created, giving it its initial state.
    """
    
    def __init__(self, role: str, content: str) -> None:
        """Initialize a new ChatMessage.
        
        Args:
            role: The role of the message sender ('system', 'user', 'assistant')
            content: The text content of the message
        """
        # Instance attributes - unique to each object
        self.role = role
        self.content = content
        self.created_at = datetime.now()


print("\n=== Part 2: With __init__ ===")
msg1 = ChatMessageWithInit("user", "Hello, AI!")
msg2 = ChatMessageWithInit("assistant", "Hello! How can I help you today?")

print(f"Message 1 - Role: {msg1.role}, Content: {msg1.content}")
print(f"Message 2 - Role: {msg2.role}, Content: {msg2.content}")
print(f"Created at: {msg1.created_at}")


# =============================================================================
# PART 3: Understanding 'self'
# =============================================================================

class ChatMessageExplained:
    """Chat message that demonstrates what 'self' really is.
    
    'self' is a reference to the instance being created or operated on.
    It's how methods know which object's data to work with.
    
    When you call: msg.get_role()
    Python translates it to: ChatMessageExplained.get_role(msg)
    """
    
    def __init__(self, role: str, content: str) -> None:
        # 'self' here refers to the new instance being created
        self.role = role
        self.content = content
    
    def get_role(self) -> str:
        # 'self' here refers to the instance the method is called on
        return self.role
    
    def get_content(self) -> str:
        return self.content


print("\n=== Part 3: Understanding self ===")
msg = ChatMessageExplained("user", "What is Python?")

# These two calls are equivalent:
print(f"Using method: {msg.get_role()}")
print(f"Explicit self: {ChatMessageExplained.get_role(msg)}")


# =============================================================================
# PART 4: Optional Parameters and Defaults
# =============================================================================

class ChatMessage:
    """Production-quality chat message with optional parameters.
    
    This version includes:
    - Required parameters (role, content)
    - Optional parameters with defaults (name)
    - Auto-generated values (created_at)
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        name: Optional[str] = None
    ) -> None:
        """Initialize a ChatMessage.
        
        Args:
            role: Message role ('system', 'user', 'assistant', 'tool')
            content: Text content of the message
            name: Optional sender name (useful for tool messages)
        """
        self.role = role
        self.content = content
        self.name = name
        self.created_at = datetime.now()
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role.upper()}]: {preview}"


print("\n=== Part 4: Optional Parameters ===")

# Without optional parameter
user_msg = ChatMessage("user", "Hello!")
print(user_msg)

# With optional parameter
tool_msg = ChatMessage("tool", "Search results: Python tutorials...", name="search")
print(tool_msg)
print(f"Tool name: {tool_msg.name}")


# =============================================================================
# PART 5: Building a Conversation
# =============================================================================

print("\n=== Part 5: Building a Conversation ===")

conversation = [
    ChatMessage("system", "You are a helpful Python tutor."),
    ChatMessage("user", "What is a class in Python?"),
    ChatMessage(
        "assistant",
        "A class is a blueprint for creating objects. It defines the attributes "
        "(data) and methods (functions) that objects of that class will have."
    ),
    ChatMessage("user", "Can you give me an example?"),
]

print("Conversation:")
for i, msg in enumerate(conversation, 1):
    print(f"  {i}. {msg}")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. A CLASS is a blueprint for creating objects
2. __init__ is the constructor - it initializes new instances
3. 'self' refers to the current instance
4. Instance attributes are unique to each object
5. Type hints (: str, -> None) document expected types
6. Optional[str] = None makes parameters optional
""")
