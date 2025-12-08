"""
Clean Architecture for AI Chat Systems
=======================================
Demonstrates layered architecture with proper dependency direction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


# ==============================================================================
# LAYER 1: ENTITIES (Core Domain)
# ==============================================================================
# No external dependencies - pure business logic

@dataclass
class Message:
    """Core domain entity - represents a chat message."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Conversation:
    """Aggregate root for a conversation."""
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> Message:
        """Domain logic: add message to conversation."""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def get_context(self, limit: int = 10) -> List[Message]:
        """Domain logic: get recent context."""
        return self.messages[-limit:]
    
    def get_last_user_message(self) -> Optional[str]:
        """Domain logic: find last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


# ==============================================================================
# LAYER 2: USE CASES (Application Business Rules)
# ==============================================================================
# Orchestrates domain entities, depends only on Layer 1 + ports

class ChatUseCase:
    """
    Use case for chat interactions.
    
    Contains business rules but no framework dependencies.
    Uses ports (abstractions) for external systems.
    """
    
    def __init__(
        self,
        llm_gateway: "LLMGateway",
        conversation_repo: "ConversationRepository"
    ):
        self.llm_gateway = llm_gateway
        self.conversation_repo = conversation_repo
    
    def send_message(self, conversation_id: str, user_message: str) -> str:
        """
        Business logic for sending a message.
        
        1. Get or create conversation
        2. Add user message
        3. Generate response
        4. Add assistant message
        5. Save conversation
        """
        # Get or create conversation
        conversation = self.conversation_repo.get(conversation_id)
        if not conversation:
            conversation = Conversation(id=conversation_id)
        
        # Business rule: validate message
        if not user_message.strip():
            raise ValueError("Message cannot be empty")
        
        # Add user message
        conversation.add_message("user", user_message)
        
        # Generate response via gateway (abstraction)
        context = conversation.get_context()
        response = self.llm_gateway.generate(context)
        
        # Add assistant message
        conversation.add_message("assistant", response)
        
        # Persist (through abstraction)
        self.conversation_repo.save(conversation)
        
        return response
    
    def get_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history."""
        conversation = self.conversation_repo.get(conversation_id)
        if not conversation:
            return []
        return [msg.to_dict() for msg in conversation.messages]


class SummarizeUseCase:
    """Use case for summarizing conversations."""
    
    def __init__(self, llm_gateway: "LLMGateway"):
        self.llm_gateway = llm_gateway
    
    def summarize(self, conversation: Conversation) -> str:
        """Summarize a conversation."""
        # Business rule: need at least 3 messages to summarize
        if len(conversation.messages) < 3:
            raise ValueError("Need at least 3 messages to summarize")
        
        # Format for summarization
        text = "\n".join(
            f"{msg.role}: {msg.content}" 
            for msg in conversation.messages
        )
        
        summary_prompt = Message(
            role="user",
            content=f"Summarize this conversation:\n\n{text}"
        )
        
        return self.llm_gateway.generate([summary_prompt])


# ==============================================================================
# LAYER 3: INTERFACE ADAPTERS (Ports/Gateways)
# ==============================================================================
# Abstract interfaces that outer layers implement

class LLMGateway(ABC):
    """Port for LLM interaction."""
    
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate response from messages."""
        pass


class ConversationRepository(ABC):
    """Port for conversation persistence."""
    
    @abstractmethod
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve conversation by ID."""
        pass
    
    @abstractmethod
    def save(self, conversation: Conversation) -> None:
        """Save conversation."""
        pass
    
    @abstractmethod
    def delete(self, conversation_id: str) -> bool:
        """Delete conversation."""
        pass


# ==============================================================================
# LAYER 4: FRAMEWORKS & DRIVERS (External)
# ==============================================================================
# Concrete implementations with external dependencies

class OpenAIGateway(LLMGateway):
    """OpenAI implementation of LLM gateway."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def generate(self, messages: List[Message]) -> str:
        """
        In production, this would call:
        openai.chat.completions.create(...)
        """
        # Simulated response
        last_msg = messages[-1].content if messages else ""
        return f"[{self.model}] Response to: {last_msg[:50]}"


class MockLLMGateway(LLMGateway):
    """Mock gateway for testing."""
    
    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.calls: List[List[Message]] = []
    
    def generate(self, messages: List[Message]) -> str:
        self.calls.append(messages)
        return self.response


class InMemoryConversationRepository(ConversationRepository):
    """In-memory repository for development/testing."""
    
    def __init__(self):
        self._storage: Dict[str, Conversation] = {}
    
    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._storage.get(conversation_id)
    
    def save(self, conversation: Conversation) -> None:
        self._storage[conversation.id] = conversation
    
    def delete(self, conversation_id: str) -> bool:
        if conversation_id in self._storage:
            del self._storage[conversation_id]
            return True
        return False


class RedisConversationRepository(ConversationRepository):
    """
    Redis implementation for production.
    
    In production, this would use redis-py.
    """
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis_url = redis_url
        # In production: self.redis = redis.from_url(redis_url)
        self._mock_storage: Dict[str, str] = {}
    
    def get(self, conversation_id: str) -> Optional[Conversation]:
        data = self._mock_storage.get(f"conv:{conversation_id}")
        if not data:
            return None
        # In production: deserialize from Redis
        parsed = json.loads(data)
        conv = Conversation(id=parsed["id"])
        for msg_data in parsed["messages"]:
            conv.add_message(msg_data["role"], msg_data["content"])
        return conv
    
    def save(self, conversation: Conversation) -> None:
        data = {
            "id": conversation.id,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in conversation.messages
            ]
        }
        self._mock_storage[f"conv:{conversation.id}"] = json.dumps(data)
    
    def delete(self, conversation_id: str) -> bool:
        key = f"conv:{conversation_id}"
        if key in self._mock_storage:
            del self._mock_storage[key]
            return True
        return False


# ==============================================================================
# COMPOSITION ROOT (Dependency Injection)
# ==============================================================================

class Container:
    """Dependency injection container."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
    
    def get_llm_gateway(self) -> LLMGateway:
        if self.environment == "production":
            return OpenAIGateway(model="gpt-4")
        elif self.environment == "test":
            return MockLLMGateway()
        else:
            return OpenAIGateway(model="gpt-3.5-turbo")
    
    def get_conversation_repo(self) -> ConversationRepository:
        if self.environment == "production":
            return RedisConversationRepository()
        else:
            return InMemoryConversationRepository()
    
    def get_chat_use_case(self) -> ChatUseCase:
        return ChatUseCase(
            llm_gateway=self.get_llm_gateway(),
            conversation_repo=self.get_conversation_repo()
        )


# ==============================================================================
# DEMO
# ==============================================================================

def main():
    print("=" * 60)
    print("Clean Architecture Demo")
    print("=" * 60)
    
    # Development environment
    container = Container(environment="development")
    chat_use_case = container.get_chat_use_case()
    
    # Use the system
    print("\n1. Sending first message...")
    response1 = chat_use_case.send_message("conv-123", "Hello!")
    print(f"   Response: {response1}")
    
    print("\n2. Sending second message...")
    response2 = chat_use_case.send_message("conv-123", "How are you?")
    print(f"   Response: {response2}")
    
    print("\n3. Getting history...")
    history = chat_use_case.get_history("conv-123")
    for msg in history:
        print(f"   {msg['role']}: {msg['content'][:40]}...")
    
    # Test environment - uses mocks
    print("\n4. Testing with mocks...")
    test_container = Container(environment="test")
    test_use_case = test_container.get_chat_use_case()
    
    test_response = test_use_case.send_message("test-1", "Test message")
    print(f"   Test response: {test_response}")
    
    print("\n✅ Clean Architecture demo complete!")
    print("""
Architecture Layers:
  Layer 1 (Entities):    Message, Conversation
  Layer 2 (Use Cases):   ChatUseCase, SummarizeUseCase
  Layer 3 (Adapters):    LLMGateway, ConversationRepository
  Layer 4 (Frameworks):  OpenAIGateway, RedisConversationRepository
  
Key Benefits:
  - Business logic (Layer 2) is framework-independent
  - Easy to test with MockLLMGateway
  - Can swap Redis for PostgreSQL without changing use cases
  - Can swap OpenAI for Anthropic without changing use cases
""")


if __name__ == "__main__":
    main()
