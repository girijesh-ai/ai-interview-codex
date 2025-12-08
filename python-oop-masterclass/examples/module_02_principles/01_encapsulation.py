"""
Module 02, Example 01: Encapsulation - Secure LLM Client

This example demonstrates:
- Protected attributes (_single_underscore)
- Private attributes (__double_underscore)
- Internal rate limiting
- Controlled access to sensitive data

Run this file:
    python 01_encapsulation.py

Follow along with: 02-oop-principles-multi-provider.md
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import threading


# =============================================================================
# PART 1: The Problem Without Encapsulation
# =============================================================================

class UnsafeLLMClient:
    """BAD: Everything is exposed."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key  # Anyone can see this!
        self.rate_limit_remaining = 100  # Anyone can modify!
        self.token_budget = 10000


print("=== Part 1: The Problem ===")
unsafe = UnsafeLLMClient("sk-secret-key-12345")
print(f"Exposed API key: {unsafe.api_key}")
unsafe.rate_limit_remaining = 999999  # Bypass rate limiting!
print(f"Bypassed rate limit: {unsafe.rate_limit_remaining}")


# =============================================================================
# PART 2: Proper Encapsulation
# =============================================================================

class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass


class SecureLLMClient:
    """LLM Client with properly encapsulated sensitive data.
    
    Demonstrates:
    - Protected attributes (_single_underscore): Internal use, subclass OK
    - Private attributes (__double_underscore): Truly internal, name mangled
    - Property-based access control
    - Internal rate limiting that can't be bypassed
    """
    
    # Class-level configuration
    _DEFAULT_RATE_LIMIT = 60  # requests per minute
    
    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        rate_limit: int = 60
    ) -> None:
        """Initialize secure client."""
        # Protected: use within class and subclasses
        self._api_key = api_key
        self._org_id = org_id
        self._rate_limit = rate_limit
        
        # Private: only this class should access (name mangled)
        self.__request_timestamps: list = []
        self.__token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.__lock = threading.Lock()
    
    # ========== Controlled Access via Properties ==========
    
    @property
    def api_key_preview(self) -> str:
        """Get masked API key for logging/display.
        
        The full key is NEVER exposed outside the class.
        """
        if len(self._api_key) < 10:
            return "***"
        return f"{self._api_key[:7]}...{self._api_key[-4:]}"
    
    @property
    def token_usage(self) -> Dict[str, int]:
        """Get token usage (read-only copy)."""
        return self.__token_usage.copy()
    
    @property
    def rate_limit_remaining(self) -> int:
        """Get remaining requests in current window."""
        self.__cleanup_old_requests()
        return max(0, self._rate_limit - len(self.__request_timestamps))
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return self.rate_limit_remaining <= 0
    
    # ========== Internal Rate Limiting ==========
    
    def __cleanup_old_requests(self) -> None:
        """Remove request timestamps older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        with self.__lock:
            self.__request_timestamps = [
                ts for ts in self.__request_timestamps
                if ts > cutoff
            ]
    
    def __record_request(self) -> None:
        """Record a request for rate limiting."""
        with self.__lock:
            self.__request_timestamps.append(datetime.now())
    
    def __check_rate_limit(self) -> None:
        """Check rate limit before request."""
        if self.is_rate_limited:
            raise RateLimitError("Rate limit exceeded. Please wait.")
    
    # ========== Internal Token Tracking ==========
    
    def _update_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> None:
        """Update token usage (for internal/subclass use)."""
        with self.__lock:
            self.__token_usage["prompt_tokens"] += prompt_tokens
            self.__token_usage["completion_tokens"] += completion_tokens
            self.__token_usage["total_tokens"] += (
                prompt_tokens + completion_tokens
            )
    
    # ========== Public API ==========
    
    def complete(self, prompt: str) -> str:
        """Send completion request."""
        # Internal checks (encapsulated)
        self.__check_rate_limit()
        self.__record_request()
        
        # Simulate API call
        response = f"Response to: {prompt[:30]}..."
        
        # Update internal tracking
        self._update_token_usage(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(response) // 4
        )
        
        return response


print("\n=== Part 2: Secure Client ===")
client = SecureLLMClient("sk-very-secret-key-12345", rate_limit=5)

# Safe access via property
print(f"Masked key: {client.api_key_preview}")

# Cannot access private attributes directly
try:
    print(client.__request_timestamps)
except AttributeError as e:
    print(f"Private access blocked: {e}")

# Rate limiting is automatic and can't be bypassed
print(f"Initial rate limit: {client.rate_limit_remaining}")

for i in range(3):
    response = client.complete(f"Message {i}")
    print(f"Request {i+1}: Remaining = {client.rate_limit_remaining}")

# Token usage is tracked internally
print(f"Token usage: {client.token_usage}")


# =============================================================================
# PART 3: Visibility Diagram
# =============================================================================

print("\n=== Part 3: Access Levels ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                     SecureLLMClient                         │
├─────────────────────────────────────────────────────────────┤
│ PUBLIC (anyone can use)                                     │
│   • api_key_preview (property)                              │
│   • token_usage (property)                                  │
│   • rate_limit_remaining (property)                         │
│   • complete(prompt)                                        │
├─────────────────────────────────────────────────────────────┤
│ PROTECTED (internal + subclasses)                           │
│   • _api_key                                                │
│   • _rate_limit                                             │
│   • _update_token_usage()                                   │
├─────────────────────────────────────────────────────────────┤
│ PRIVATE (this class only)                                   │
│   • __request_timestamps                                    │
│   • __token_usage                                           │
│   • __check_rate_limit()                                    │
│   • __record_request()                                      │
└─────────────────────────────────────────────────────────────┘
""")
