"""
Testing & Production Example
=============================
Demonstrates:
- Mocking LLM responses for tests
- FastAPI test patterns
- Structured JSON logging
- Health checks

Run tests: pytest test_production.py -v
Run demo: python a01_production_patterns.py
"""

import logging
import json
import sys
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
from functools import wraps
from unittest.mock import Mock, AsyncMock
import asyncio


# ==============================================================================
# STRUCTURED LOGGING
# ==============================================================================

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key in ["request_id", "model", "duration_ms", "tokens"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        return json.dumps(log_data)


def setup_logging(json_format: bool = True, level: str = "INFO"):
    """Configure application logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
    
    root.addHandler(handler)


# ==============================================================================
# LLM CALL LOGGING DECORATOR
# ==============================================================================

logger = logging.getLogger("llm")


def log_llm_call(func: Callable) -> Callable:
    """Decorator to log LLM API calls with timing and tokens."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        model = kwargs.get("model", "unknown")
        
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            
            logger.info(
                "LLM call completed",
                extra={
                    "model": model,
                    "duration_ms": round(duration, 2),
                    "tokens": result.get("usage", {}).get("total_tokens", 0),
                }
            )
            return result
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"LLM call failed: {e}", extra={
                "model": model,
                "duration_ms": round(duration, 2),
            })
            raise
    
    return wrapper


# ==============================================================================
# MOCK LLM FOR TESTING
# ==============================================================================

@dataclass
class MockLLMResponse:
    """Mock response matching OpenAI format."""
    content: str
    model: str = "gpt-4o-mini"
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
        }


class MockLLMClient:
    """
    Mock LLM client for testing.
    
    Use this instead of real API calls in tests.
    Configurable responses for different scenarios.
    """
    
    def __init__(self):
        self.responses: List[str] = []
        self.call_count = 0
        self.calls: List[Dict] = []
    
    def add_response(self, content: str):
        """Queue a response."""
        self.responses.append(content)
    
    async def complete(self, messages: List[Dict], model: str = "gpt-4o-mini") -> Dict:
        """Mock completion call."""
        self.call_count += 1
        self.calls.append({"messages": messages, "model": model})
        
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = f"Mock response #{self.call_count}"
        
        return MockLLMResponse(content=content, model=model).to_dict()
    
    async def stream(self, messages: List[Dict], model: str = "gpt-4o-mini"):
        """Mock streaming."""
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = "Streaming mock response"
        
        for word in content.split():
            await asyncio.sleep(0.01)
            yield word + " "


# ==============================================================================
# HEALTH CHECK PATTERN
# ==============================================================================

class HealthChecker:
    """
    Check health of dependencies.
    
    Use in /health endpoint to verify system status.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def add_check(self, name: str, check_fn: Callable):
        """Register a health check."""
        self.checks[name] = check_fn
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_ok = True
        
        for name, check_fn in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    ok = await check_fn()
                else:
                    ok = check_fn()
                results[name] = "ok" if ok else "error"
                if not ok:
                    all_ok = False
            except Exception as e:
                results[name] = f"error: {str(e)}"
                all_ok = False
        
        return {
            "status": "healthy" if all_ok else "degraded",
            "checks": results,
        }


# ==============================================================================
# REQUEST ID TRACKING
# ==============================================================================

class RequestContext:
    """Track request ID through async code."""
    
    _current_id: str = ""
    
    @classmethod
    def set_id(cls, request_id: str):
        cls._current_id = request_id
    
    @classmethod
    def get_id(cls) -> str:
        return cls._current_id or "no-request"
    
    @classmethod
    def generate_id(cls) -> str:
        new_id = str(uuid.uuid4())[:8]
        cls.set_id(new_id)
        return new_id


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    """Demonstrate production patterns."""
    
    print("=" * 60)
    print("Testing & Production Patterns Demo")
    print("=" * 60)
    
    # ========== LOGGING ==========
    print("\n--- JSON Logging ---")
    
    setup_logging(json_format=False)  # Human-readable for demo
    logger = logging.getLogger("demo")
    
    logger.info("Application starting")
    logger.warning("This is a warning")
    
    # With extra fields
    logger.info("Request processed", extra={
        "request_id": "abc123",
        "duration_ms": 45.2,
    })
    
    # ========== MOCK LLM ==========
    print("\n--- Mock LLM Client ---")
    
    async def test_mock():
        mock = MockLLMClient()
        mock.add_response("Hello from mock!")
        mock.add_response("Second response")
        
        # First call
        result1 = await mock.complete([{"role": "user", "content": "Hi"}])
        print(f"Response 1: {result1['content']}")
        
        # Second call
        result2 = await mock.complete([{"role": "user", "content": "Hello again"}])
        print(f"Response 2: {result2['content']}")
        
        print(f"Total calls: {mock.call_count}")
    
    asyncio.run(test_mock())
    
    # ========== STREAMING MOCK ==========
    print("\n--- Mock Streaming ---")
    
    async def test_stream():
        mock = MockLLMClient()
        mock.add_response("This is a streaming response")
        
        print("Streaming: ", end="", flush=True)
        async for token in mock.stream([{"role": "user", "content": "Stream please"}]):
            print(token, end="", flush=True)
        print()
    
    asyncio.run(test_stream())
    
    # ========== HEALTH CHECKS ==========
    print("\n--- Health Checks ---")
    
    async def test_health():
        checker = HealthChecker()
        
        # Add checks
        checker.add_check("database", lambda: True)
        checker.add_check("redis", lambda: True)
        checker.add_check("api_key", lambda: True)
        
        result = await checker.run_checks()
        print(f"Status: {result['status']}")
        print(f"Checks: {result['checks']}")
        
        # Simulate failure
        checker.add_check("external_api", lambda: False)
        result = await checker.run_checks()
        print(f"With failure - Status: {result['status']}")
    
    asyncio.run(test_health())
    
    # ========== REQUEST CONTEXT ==========
    print("\n--- Request ID Tracking ---")
    
    request_id = RequestContext.generate_id()
    print(f"Generated request ID: {request_id}")
    print(f"Current ID: {RequestContext.get_id()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- JSON Logging: Structured logs for aggregation
- Mock LLM: Test without API calls
- Health Checks: Verify dependencies
- Request IDs: Track requests through system

No over-engineering - practical production patterns.
""")


if __name__ == "__main__":
    demo()
