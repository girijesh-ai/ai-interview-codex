"""
Module 03, Example 02: __new__ vs __init__ - Singleton Pattern

This example demonstrates:
- The difference between __new__ and __init__
- Implementing Singleton pattern with __new__
- Object caching/flyweight pattern
- When to use __new__

Run this file:
    python 02_new_vs_init.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

from typing import Dict, Optional


# =============================================================================
# PART 1: Understanding __new__ vs __init__
# =============================================================================

print("=== Part 1: __new__ vs __init__ ===")
print("""
Object creation is a TWO-PHASE process:

1. __new__(cls, ...) → CREATES the instance
   - Called first
   - Receives the CLASS (cls), not instance
   - Must RETURN the new instance

2. __init__(self, ...) → INITIALIZES the instance  
   - Called second (only if __new__ returns correct type)
   - Receives the INSTANCE (self)
   - Returns Nothing (None)

Most of the time, you only need __init__.
Use __new__ when you need to control WHETHER/HOW an object is created.
""")


class DemoClass:
    """Demonstration of __new__ and __init__ order."""
    
    def __new__(cls, value):
        print(f"  1. __new__ called with cls={cls.__name__}, value={value}")
        instance = super().__new__(cls)
        print(f"  2. __new__ created instance: {id(instance)}")
        return instance
    
    def __init__(self, value):
        print(f"  3. __init__ called with self={id(self)}, value={value}")
        self.value = value
        print(f"  4. __init__ set self.value = {value}")


print("\nCreating DemoClass('hello'):")
obj = DemoClass("hello")
print(f"Final object: {obj}, value: {obj.value}")


# =============================================================================
# PART 2: Singleton Pattern with __new__
# =============================================================================

print("\n=== Part 2: Singleton LLM Client ===")


class SingletonLLM:
    """Singleton LLM client using __new__.
    
    Use case: Expensive model loading should only happen once.
    Multiple calls to SingletonLLM() return the SAME instance.
    """
    
    _instance: Optional["SingletonLLM"] = None
    
    def __new__(cls, api_key: str = "default-key") -> "SingletonLLM":
        """Create or return the singleton instance."""
        if cls._instance is None:
            print(f"Creating NEW singleton instance...")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        else:
            print(f"Returning EXISTING singleton instance...")
        return cls._instance
    
    def __init__(self, api_key: str = "default-key") -> None:
        """Initialize (only runs fully once)."""
        # Prevent re-initialization
        if self._initialized:
            return
        
        print(f"Initializing with api_key={api_key[:10]}...")
        self._api_key = api_key
        self._model_loaded = True
        self._initialized = True
    
    def complete(self, prompt: str) -> str:
        return f"[Singleton] Response to: {prompt[:30]}..."


# Test singleton behavior
client1 = SingletonLLM("sk-first-key")
client2 = SingletonLLM("sk-second-key")  # Returns same instance!
client3 = SingletonLLM()

print(f"\nclient1 is client2: {client1 is client2}")  # True
print(f"client1 is client3: {client1 is client3}")  # True
print(f"All same object ID: {id(client1)}")


# =============================================================================
# PART 3: Per-Model Singleton (Multiton Pattern)
# =============================================================================

print("\n=== Part 3: Per-Model Singleton ===")


class PerModelLLM:
    """One singleton instance per model.
    
    Use case: Each model has expensive state, but we want
    to share instances for the same model.
    """
    
    _instances: Dict[str, "PerModelLLM"] = {}
    
    def __new__(cls, model: str = "gpt-4") -> "PerModelLLM":
        """Create or return instance for this model."""
        if model not in cls._instances:
            print(f"Creating new instance for model: {model}")
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model] = instance
        else:
            print(f"Reusing existing instance for model: {model}")
        return cls._instances[model]
    
    def __init__(self, model: str = "gpt-4") -> None:
        if self._initialized:
            return
        
        self.model = model
        self._api_key: Optional[str] = None
        self._initialized = True
        print(f"Initialized instance for: {model}")
    
    def configure(self, api_key: str) -> None:
        self._api_key = api_key


# Test per-model singleton
gpt4_a = PerModelLLM("gpt-4")
gpt4_b = PerModelLLM("gpt-4")    # Same instance
claude = PerModelLLM("claude-3")  # Different instance

print(f"\ngpt4_a is gpt4_b: {gpt4_a is gpt4_b}")  # True
print(f"gpt4_a is claude: {gpt4_a is claude}")   # False

# Configuration is shared for same model
gpt4_a.configure("sk-shared-key")
print(f"gpt4_b._api_key: {gpt4_b._api_key}")  # sk-shared-key (shared!)


# =============================================================================
# PART 4: Object Caching (Flyweight)
# =============================================================================

print("\n=== Part 4: Object Caching ===")


class CachedMessage:
    """Message with object caching.
    
    For frequently used messages (like system prompts),
    return cached instances instead of creating new ones.
    """
    
    _cache: Dict[tuple, "CachedMessage"] = {}
    
    def __new__(cls, role: str, content: str) -> "CachedMessage":
        """Return cached instance if exists."""
        key = (role, content)
        
        if key in cls._cache:
            print(f"Cache HIT for: {role}:{content[:20]}...")
            return cls._cache[key]
        
        print(f"Cache MISS for: {role}:{content[:20]}...")
        instance = super().__new__(cls)
        cls._cache[key] = instance
        return instance
    
    def __init__(self, role: str, content: str) -> None:
        # Only initialize if not already done
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.role = role
        self.content = content
        self._initialized = True


# Common system prompt - should be cached
sys_prompt = "You are a helpful AI assistant."

msg1 = CachedMessage("system", sys_prompt)
msg2 = CachedMessage("system", sys_prompt)  # Cache hit!
msg3 = CachedMessage("user", "Hello")       # Cache miss (different)

print(f"\nmsg1 is msg2: {msg1 is msg2}")  # True (same object)
print(f"msg1 is msg3: {msg1 is msg3}")  # False (different)


# =============================================================================
# PART 5: When to Use __new__
# =============================================================================

print("\n=== Part 5: When to Use __new__ ===")
print("""
┌────────────────────────────────────────────────────────────┐
│              USE __new__ WHEN:                              │
├────────────────────────────────────────────────────────────┤
│ ✓ Implementing Singleton pattern                           │
│ ✓ Caching/Flyweight pattern (reuse objects)               │
│ ✓ Subclassing immutable types (str, int, tuple)           │
│ ✓ Object pooling (reuse expensive objects)                │
│ ✓ Controlling WHETHER an object is created                 │
├────────────────────────────────────────────────────────────┤
│              USE __init__ WHEN:                             │
├────────────────────────────────────────────────────────────┤
│ ✓ Normal object initialization                             │
│ ✓ Setting attributes                                       │
│ ✓ Validating arguments                                     │
│ ✓ 99% of the time!                                         │
└────────────────────────────────────────────────────────────┘

Key difference:
    __new__  → Controls WHETHER/WHICH object is returned
    __init__ → Controls HOW the object is initialized
""")
