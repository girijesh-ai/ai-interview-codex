"""Shared test classes for examples."""

class ChatMessage:
    """Basic ChatMessage for test imports."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def __str__(self):
        return f"[{self.role}]: {self.content}"
