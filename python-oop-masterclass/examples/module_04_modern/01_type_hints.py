"""
Module 04, Example 01: Advanced Type Hints for AI

This example demonstrates:
- TypeVar and Generics
- Bounded TypeVars
- ParamSpec for decorators
- Protocols for structural typing

Run this file:
    python 01_type_hints.py

Follow along with: 04-modern-python-ai-features.md
"""

from typing import (
    TypeVar, Generic, List, Dict, Optional, Any,
    Callable, Protocol, runtime_checkable
)
from abc import ABC, abstractmethod
import functools

# Note: ParamSpec requires Python 3.10+ or typing_extensions
import sys
PARAMSPEC_AVAILABLE = False

if sys.version_info >= (3, 10):
    from typing import ParamSpec
    PARAMSPEC_AVAILABLE = True
else:
    try:
        from typing_extensions import ParamSpec
        PARAMSPEC_AVAILABLE = True
    except ImportError:
        # Will use simpler decorator without ParamSpec
        pass


# =============================================================================
# PART 1: TypeVar and Generics
# =============================================================================

print("=== Part 1: TypeVar and Generics ===")

# T is a placeholder for any type
T = TypeVar('T')

class Repository(Generic[T]):
    """Generic repository - works with any entity type.
    
    The same code works for ChatMessage, User, Agent, etc.
    Type checker knows the exact types you're working with.
    """
    
    def __init__(self) -> None:
        self._items: Dict[str, T] = {}
    
    def add(self, id: str, item: T) -> None:
        """Add item to repository."""
        self._items[id] = item
        print(f"  Added {type(item).__name__} with id={id}")
    
    def get(self, id: str) -> Optional[T]:
        """Get item by id."""
        return self._items.get(id)
    
    def list_all(self) -> List[T]:
        """List all items."""
        return list(self._items.values())


# Usage with different types
from dataclasses import dataclass

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass 
class User:
    name: str
    email: str

# Type-safe repositories
message_repo: Repository[ChatMessage] = Repository()
message_repo.add("1", ChatMessage("user", "Hello"))
message_repo.add("2", ChatMessage("assistant", "Hi there!"))

user_repo: Repository[User] = Repository()
user_repo.add("u1", User("Alice", "alice@example.com"))

# Type checker knows these types!
msg = message_repo.get("1")  # Optional[ChatMessage]
if msg:
    print(f"  Message content: {msg.content}")

user = user_repo.get("u1")  # Optional[User]
if user:
    print(f"  User email: {user.email}")


# =============================================================================
# PART 2: Bounded TypeVar
# =============================================================================

print("\n=== Part 2: Bounded TypeVar ===")

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate completion."""
        ...

class OpenAIProvider(LLMProvider):
    def complete(self, prompt: str) -> str:
        return f"[OpenAI] {prompt[:20]}..."

class ClaudeProvider(LLMProvider):
    def complete(self, prompt: str) -> str:
        return f"[Claude] {prompt[:20]}..."

# T must be a subclass of LLMProvider
ProviderT = TypeVar('ProviderT', bound=LLMProvider)

class ProviderPool(Generic[ProviderT]):
    """Pool of LLM providers with round-robin selection.
    
    Only accepts LLMProvider subclasses!
    """
    
    def __init__(self, providers: List[ProviderT]) -> None:
        self._providers = providers
        self._index = 0
    
    def get_next(self) -> ProviderT:
        """Get next provider (round-robin)."""
        provider = self._providers[self._index]
        self._index = (self._index + 1) % len(self._providers)
        return provider

# Create pool with OpenAI providers
openai_pool: ProviderPool[OpenAIProvider] = ProviderPool([
    OpenAIProvider(),
    OpenAIProvider()
])

# Get providers (type is OpenAIProvider, not just LLMProvider)
p1 = openai_pool.get_next()
p2 = openai_pool.get_next()
print(f"  Provider 1: {p1.complete('Hello')}")
print(f"  Provider 2: {p2.complete('World')}")


# =============================================================================
# PART 3: ParamSpec for Type-Safe Decorators
# =============================================================================

print("\n=== Part 3: ParamSpec for Decorators ===")

if PARAMSPEC_AVAILABLE:
    P = ParamSpec('P')  # Captures all parameters
    R = TypeVar('R')    # Captures return type

    def retry(max_attempts: int = 3):
        """Type-safe retry decorator.
        
        ParamSpec preserves the original function's signature!
        Without ParamSpec, type checker loses parameter info.
        """
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        print(f"    Attempt {attempt} failed: {e}")
                        last_error = e
                raise last_error  # type: ignore
            return wrapper
        return decorator

    attempt_count = 0

    @retry(max_attempts=3)
    def call_llm(prompt: str, temperature: float = 0.7) -> str:
        """Original function - signature is preserved!"""
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("API timeout")
        return f"Response to '{prompt}' at temp={temperature}"

    result: str = call_llm("Hello", temperature=0.5)
    print(f"  Result: {result}")

else:
    print("  ParamSpec requires Python 3.10+ or typing_extensions")
    print("  Showing concept with simpler decorator:")
    
    # Fallback without ParamSpec
    R = TypeVar('R')
    
    def retry_simple(max_attempts: int = 3):
        """Simple retry decorator (without ParamSpec)."""
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        print(f"    Attempt {attempt} failed: {e}")
                        last_error = e
                raise last_error  # type: ignore
            return wrapper
        return decorator
    
    attempt_count = 0
    
    @retry_simple(max_attempts=3)
    def call_llm(prompt: str, temperature: float = 0.7) -> str:
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("API timeout")
        return f"Response to '{prompt}' at temp={temperature}"
    
    result = call_llm("Hello", temperature=0.5)
    print(f"  Result: {result}")

# Type checker knows:
# - call_llm takes (prompt: str, temperature: float = 0.7)
# - call_llm returns str


# =============================================================================
# PART 4: Protocols for Structural Typing
# =============================================================================

print("\n=== Part 4: Protocols (Structural Typing) ===")

@runtime_checkable
class Embeddable(Protocol):
    """Anything that can be converted to text for embedding.
    
    Classes don't need to inherit from this!
    They just need to have the right method.
    """
    
    def to_embedding_text(self) -> str:
        """Return text for embedding."""
        ...

# These classes don't inherit from Embeddable
class Document:
    def __init__(self, content: str):
        self.content = content
    
    def to_embedding_text(self) -> str:
        return self.content

class WebPage:
    def __init__(self, url: str, html: str):
        self.url = url
        self.html = html
    
    def to_embedding_text(self) -> str:
        return f"URL: {self.url}\n{self.html[:100]}"

class CodeFile:
    def __init__(self, path: str, code: str):
        self.path = path
        self.code = code
    
    def to_embedding_text(self) -> str:
        return f"# {self.path}\n{self.code}"

# Function works with anything Embeddable
def embed(item: Embeddable) -> List[float]:
    """Create embedding for any embeddable item."""
    text = item.to_embedding_text()
    # Simulate embedding (real impl would call API)
    return [hash(text) % 100 / 100.0 for _ in range(4)]

# All work because they have to_embedding_text()
doc = Document("Hello world")
page = WebPage("https://python.org", "<html>Python</html>")
code = CodeFile("main.py", "print('hello')")

print(f"  Document is Embeddable: {isinstance(doc, Embeddable)}")
print(f"  WebPage is Embeddable: {isinstance(page, Embeddable)}")
print(f"  CodeFile is Embeddable: {isinstance(code, Embeddable)}")

print(f"  Document embedding: {embed(doc)[:2]}...")
print(f"  WebPage embedding: {embed(page)[:2]}...")


# =============================================================================
# PART 5: Summary
# =============================================================================

print("\n=== Part 5: Summary ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│               ADVANCED TYPE HINTS FOR AI                     │
├─────────────────────────────────────────────────────────────┤
│ TypeVar + Generic[T]:                                        │
│   Write reusable code that works with any type               │
│   Repository[ChatMessage], Cache[K, V], Pool[T]              │
├─────────────────────────────────────────────────────────────┤
│ Bounded TypeVar (bound=...):                                 │
│   Constrain T to specific base types                         │
│   ProviderT = TypeVar('ProviderT', bound=LLMProvider)        │
├─────────────────────────────────────────────────────────────┤
│ ParamSpec:                                                   │
│   Preserve function signatures in decorators                 │
│   @retry, @cache, @timeout work with any function           │
├─────────────────────────────────────────────────────────────┤
│ Protocol + @runtime_checkable:                               │
│   Structural typing - no inheritance needed                  │
│   "If it quacks like a duck..."                              │
└─────────────────────────────────────────────────────────────┘
""")
