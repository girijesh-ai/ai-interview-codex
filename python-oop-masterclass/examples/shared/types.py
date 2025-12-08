"""
Common types used across all examples.

This module provides the core data structures that are used throughout
the OOP masterclass examples.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class Role(Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """Represents a message in an LLM conversation.
    
    This is the fundamental building block of all LLM interactions.
    Every conversation is essentially a list of ChatMessage objects.
    
    Attributes:
        role: The role of the message sender (system/user/assistant/tool)
        content: The text content of the message
        name: Optional name for the sender (used for tool messages)
        tool_calls: Optional list of tool calls (for assistant messages)
        tool_call_id: Optional ID linking to a tool call (for tool messages)
        created_at: Timestamp when the message was created
    
    Example:
        >>> msg = ChatMessage(role="user", content="Hello!")
        >>> print(msg)
        [USER]: Hello!
    """
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role.upper()}]: {preview}"
    
    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"ChatMessage(role={self.role!r}, content={self.content[:30]!r}...)"
    
    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        return len(self.content) // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class LLMResponse:
    """Response from an LLM provider.
    
    Attributes:
        content: The generated text content
        model: The model that generated the response
        usage: Token usage statistics
        finish_reason: Why the generation stopped
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


if __name__ == "__main__":
    # Demo the types
    print("=== ChatMessage Demo ===\n")
    
    # Create messages
    system_msg = ChatMessage(
        role="system",
        content="You are a helpful AI assistant specialized in Python."
    )
    
    user_msg = ChatMessage(
        role="user",
        content="What is object-oriented programming?"
    )
    
    assistant_msg = ChatMessage(
        role="assistant",
        content="Object-oriented programming (OOP) is a programming paradigm..."
    )
    
    # Display messages
    conversation = [system_msg, user_msg, assistant_msg]
    for msg in conversation:
        print(msg)
    
    print(f"\nTotal tokens: {sum(m.token_count for m in conversation)}")
    
    # Show dict conversion
    print("\n=== API Format ===")
    print(user_msg.to_dict())
