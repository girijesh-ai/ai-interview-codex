"""
Module 03, Example 03: Descriptors - Validated Prompts

This example demonstrates:
- What descriptors are
- The descriptor protocol (__get__, __set__, __delete__)
- __set_name__ for automatic naming
- Building reusable validators

Run this file:
    python 03_descriptors.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

from typing import Any, Optional


# =============================================================================
# PART 1: What is a Descriptor?
# =============================================================================

print("=== Part 1: What is a Descriptor? ===")
print("""
A descriptor is any object that implements at least one of:
- __get__(self, obj, type): Controls attribute READ
- __set__(self, obj, value): Controls attribute WRITE  
- __delete__(self, obj): Controls attribute DELETE

When you access an attribute that is a descriptor,
Python calls the descriptor's methods instead of just
returning/setting the value.

Descriptors power: @property, @classmethod, @staticmethod
""")


# =============================================================================
# PART 2: Simple Descriptor Example
# =============================================================================

class SimpleDescriptor:
    """A minimal descriptor to understand the concept."""
    
    def __get__(self, obj, objtype=None):
        print(f"  __get__ called with obj={obj}, objtype={objtype}")
        if obj is None:  # Class-level access
            return self
        return getattr(obj, '_simple_value', None)
    
    def __set__(self, obj, value):
        print(f"  __set__ called with obj={obj}, value={value}")
        obj._simple_value = value


class Demo:
    attr = SimpleDescriptor()  # The descriptor is a CLASS attribute


print("\n=== Part 2: Descriptor in Action ===")
d = Demo()
print("Setting d.attr = 'hello':")
d.attr = "hello"
print("Getting d.attr:")
result = d.attr
print(f"Result: {result}")


# =============================================================================
# PART 3: Validated String Descriptor
# =============================================================================

class ValidatedString:
    """Descriptor that validates string length.
    
    Reusable across multiple classes and attributes.
    """
    
    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        required: bool = True
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.required = required
        self.name = ""  # Will be set by __set_name__
    
    def __set_name__(self, owner: type, name: str) -> None:
        """Called automatically when descriptor is assigned to class.
        
        Python 3.6+ feature that tells the descriptor its name.
        """
        self.name = name
        self.private_name = f"_desc_{name}"
    
    def __get__(self, obj: Optional[object], objtype: type = None) -> Any:
        """Get the value."""
        if obj is None:
            return self  # Class-level access
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj: object, value: Any) -> None:
        """Set with validation."""
        if value is None and self.required:
            raise ValueError(f"{self.name} is required")
        
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(f"{self.name} must be a string, got {type(value)}")
            
            if len(value) < self.min_length:
                raise ValueError(
                    f"{self.name} must be at least {self.min_length} characters"
                )
            
            if len(value) > self.max_length:
                raise ValueError(
                    f"{self.name} must be at most {self.max_length} characters"
                )
        
        setattr(obj, self.private_name, value)


print("\n=== Part 3: Validated String ===")


class Prompt:
    # Reusable descriptors as class attributes
    system_prompt = ValidatedString(min_length=10, max_length=2000)
    user_prompt = ValidatedString(min_length=1, max_length=5000)
    
    def __init__(self, system: str, user: str):
        self.system_prompt = system  # Triggers ValidatedString.__set__
        self.user_prompt = user


# Valid prompt
try:
    prompt = Prompt(
        system="You are a helpful AI assistant.",  # >= 10 chars ✓
        user="What is Python?"
    )
    print(f"Created prompt with system: {prompt.system_prompt[:30]}...")
except ValueError as e:
    print(f"Validation error: {e}")

# Invalid prompt (too short)
try:
    bad_prompt = Prompt(system="Short", user="Hi")  # < 10 chars
except ValueError as e:
    print(f"Validation error: {e}")


# =============================================================================
# PART 4: Token Limit Descriptor
# =============================================================================

class TokenLimit:
    """Descriptor that validates token count."""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.name = ""
    
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = f"_desc_{name}"
    
    def __get__(self, obj: Optional[object], objtype: type = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, self.private_name, "")
    
    def __set__(self, obj: object, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        
        # Rough token estimate (1 token ≈ 4 chars)
        token_count = len(value) // 4
        
        if token_count > self.max_tokens:
            raise ValueError(
                f"{self.name} exceeds token limit: "
                f"{token_count} tokens > {self.max_tokens} max"
            )
        
        setattr(obj, self.private_name, value)


print("\n=== Part 4: Token Limit ===")


class LargePrompt:
    content = TokenLimit(max_tokens=100)  # Max 100 tokens
    
    def __init__(self, content: str):
        self.content = content


# Valid (short content)
try:
    short = LargePrompt("Hello, this is a short prompt.")
    print(f"Created prompt with ~{len(short.content)//4} tokens")
except ValueError as e:
    print(f"Error: {e}")

# Invalid (too long)
try:
    long_text = "This is a very long prompt. " * 50  # ~500 tokens
    long_prompt = LargePrompt(long_text)
except ValueError as e:
    print(f"Error: {e}")


# =============================================================================
# PART 5: Comparing Descriptor vs Property
# =============================================================================

print("\n=== Part 5: Descriptor vs Property ===")
print("""
@property:
    - Defined INSIDE the class
    - Works for ONE attribute in ONE class
    - Simpler syntax
    
Descriptor:
    - Defined as separate class
    - REUSABLE across multiple classes and attributes
    - More code but more flexible

Use @property for: Simple one-off getters/setters
Use Descriptor for: Reusable validation logic
""")

# Property example (for comparison)
class WithProperty:
    def __init__(self, value: str):
        self._value = value
    
    @property
    def value(self) -> str:
        return self._value
    
    @value.setter
    def value(self, v: str) -> None:
        if len(v) < 5:
            raise ValueError("Too short!")
        self._value = v


# Descriptor is REUSED in Prompt class for multiple attributes
# Property would need to be written twice (once per attribute)


# =============================================================================
# PART 6: Builtin Descriptors
# =============================================================================

print("\n=== Part 6: Built-in Descriptors ===")
print("""
Python uses descriptors internally for:

┌─────────────────────┬──────────────────────────────────────┐
│ @property           │ Data descriptor with __get__/__set__ │
│ @classmethod        │ Non-data descriptor with __get__     │
│ @staticmethod       │ Non-data descriptor with __get__     │
│ Regular methods     │ Functions are descriptors!           │
│ __slots__           │ Creates descriptors for attributes   │
└─────────────────────┴──────────────────────────────────────┘

When you write:
    obj.method()

Python actually does:
    type(obj).__dict__['method'].__get__(obj, type(obj))()

Methods are descriptors that bind 'self' automatically!
""")

# Proof that functions are descriptors
def my_function(self):
    return "hello"

print(f"Functions have __get__: {hasattr(my_function, '__get__')}")
