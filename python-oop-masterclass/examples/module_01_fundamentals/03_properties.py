"""
Module 01, Example 03: Properties and Encapsulation

This example demonstrates:
- Protected attributes (_single_underscore convention)
- Private attributes (__double_underscore name mangling)
- @property decorator for getters
- @property.setter for setters
- Computed properties
- Read-only properties

Run this file:
    python 03_properties.py

Follow along with: 01-oop-fundamentals-llm-clients.md
"""

from typing import List, Optional


# =============================================================================
# PART 1: The Problem Without Encapsulation
# =============================================================================

class UnsafeLLMClient:
    """An unsafe client with exposed internal state.
    
    PROBLEM: Anyone can access and modify sensitive data directly.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key  # Exposed!
        self.rate_limit = 100   # Can be modified!
        self.token_budget = 10000


print("=== Part 1: The Problem ===")
unsafe_client = UnsafeLLMClient("sk-secret-key-12345")

# Anyone can see the API key!
print(f"Exposed API key: {unsafe_client.api_key}")

# Anyone can bypass rate limiting!
unsafe_client.rate_limit = 999999
print(f"Rate limit (bypassed!): {unsafe_client.rate_limit}")


# =============================================================================
# PART 2: Convention-Based Protection
# =============================================================================

class ProtectedLLMClient:
    """Client using Python naming conventions for protection.
    
    Python uses naming conventions (not enforcement) for encapsulation:
    - _single_underscore: "Protected" - use within class/subclasses
    - __double_underscore: "Private" - name mangled to prevent accidents
    """
    
    def __init__(self, api_key: str):
        self._api_key = api_key       # Protected (convention)
        self.__rate_limiter = 100     # Private (name mangled)
    
    def get_masked_key(self) -> str:
        """Get masked API key for display."""
        return f"{self._api_key[:7]}...{self._api_key[-4:]}"


print("\n=== Part 2: Naming Conventions ===")
client = ProtectedLLMClient("sk-secret-key-12345")

# Protected attributes are accessible (but convention says don't)
print(f"Protected (accessible but shouldn't use): {client._api_key}")

# Private attributes are name-mangled
try:
    print(client.__rate_limiter)
except AttributeError as e:
    print(f"Private access failed: {e}")

# But can still be accessed with mangled name (not truly private)
print(f"Name mangled: {client._ProtectedLLMClient__rate_limiter}")


# =============================================================================
# PART 3: Properties - Controlled Access
# =============================================================================

class SecureLLMClient:
    """Secure client using properties for controlled access.
    
    Properties let you:
    - Control read access (getter)
    - Control write access (setter)
    - Add validation when setting values
    - Create computed values
    - Make attributes read-only
    """
    
    SUPPORTED_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._conversation: List[str] = []
    
    # ===== READ-ONLY PROPERTY (no setter) =====
    
    @property
    def api_key_masked(self) -> str:
        """Get masked API key for safe display.
        
        This is a read-only property - there's no setter,
        so you can only read, not write.
        """
        if len(self._api_key) < 10:
            return "***"
        return f"{self._api_key[:7]}...{self._api_key[-4:]}"
    
    # ===== PROPERTY WITH VALIDATION SETTER =====
    
    @property
    def model(self) -> str:
        """Get current model."""
        return self._model
    
    @model.setter
    def model(self, value: str) -> None:
        """Set model with validation.
        
        This setter validates the model before accepting it.
        If invalid, it raises ValueError.
        """
        if value not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{value}' not supported. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        self._model = value
    
    @property
    def temperature(self) -> float:
        """Get temperature."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature with range validation."""
        if not 0.0 <= value <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self._temperature = value
    
    # ===== COMPUTED PROPERTY =====
    
    @property
    def token_count(self) -> int:
        """Compute total tokens in conversation.
        
        This is a computed property - it calculates the value
        each time it's accessed. No need to store it.
        """
        total = 0
        for msg in self._conversation:
            total += len(msg) // 4  # Rough token estimate
        return total
    
    @property
    def estimated_cost(self) -> float:
        """Compute estimated cost based on token count."""
        tokens = self.token_count
        # Simplified pricing
        rate = 0.00003 if self._model == "gpt-4" else 0.000002
        return tokens * rate
    
    # Helper method
    def add_message(self, content: str) -> None:
        """Add a message to the conversation."""
        self._conversation.append(content)


print("\n=== Part 3: Properties in Action ===")

client = SecureLLMClient("sk-abc123secretkey456xyz")

# Read-only property
print(f"Masked key: {client.api_key_masked}")

# Property with validation
print(f"Current model: {client.model}")
client.model = "gpt-3.5-turbo"
print(f"Changed model: {client.model}")

# Validation error
try:
    client.model = "invalid-model"
except ValueError as e:
    print(f"Validation error: {e}")

# Temperature validation
try:
    client.temperature = 3.0  # Too high!
except ValueError as e:
    print(f"Temperature error: {e}")

# Computed properties
client.add_message("Hello, how are you?")
client.add_message("I'm doing great, thanks for asking!")
print(f"Token count: {client.token_count}")
print(f"Estimated cost: ${client.estimated_cost:.6f}")


# =============================================================================
# PART 4: Properties with Deleter
# =============================================================================

class CacheableLLMClient:
    """Client demonstrating property deleter."""
    
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._cached_response: Optional[str] = None
    
    @property
    def cached_response(self) -> Optional[str]:
        """Get cached response."""
        return self._cached_response
    
    @cached_response.setter
    def cached_response(self, value: str) -> None:
        """Set cached response."""
        self._cached_response = value
    
    @cached_response.deleter
    def cached_response(self) -> None:
        """Clear cached response."""
        print("Clearing cached response...")
        self._cached_response = None


print("\n=== Part 4: Property Deleter ===")

client = CacheableLLMClient("sk-key")
client.cached_response = "This is a cached response"
print(f"Cached: {client.cached_response}")

del client.cached_response  # Calls the deleter
print(f"After delete: {client.cached_response}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: ENCAPSULATION & PROPERTIES")
print("=" * 60)
print("""
┌────────────────────────┬────────────────────────────────────┐
│ Concept                │ Usage                               │
├────────────────────────┼────────────────────────────────────┤
│ _protected             │ Internal use, subclasses OK        │
│ __private              │ Truly internal, name mangled       │
│ @property              │ Read-only or computed values       │
│ @prop.setter           │ Validation on assignment           │
│ @prop.deleter          │ Custom cleanup on deletion         │
└────────────────────────┴────────────────────────────────────┘

WHY USE PROPERTIES?
1. Control access to internal state
2. Add validation without changing API
3. Compute values on-demand
4. Make refactoring easier (change internal storage freely)
""")
