"""
Module 03, Example 07: Decorators - Production Patterns

This example demonstrates:
- Function decorators
- Decorators with arguments
- Class decorators
- Production patterns: @retry, @cache, @tool

Run this file:
    python 07_decorators.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

import functools
import time
from typing import Callable, Any, Dict, TypeVar, ParamSpec
from dataclasses import dataclass


# Type variables for proper typing
P = ParamSpec('P')
R = TypeVar('R')


# =============================================================================
# PART 1: Basic Decorator
# =============================================================================

print("=== Part 1: Basic Decorator ===")


def simple_logger(func: Callable) -> Callable:
    """A simple decorator that logs function calls.
    
    A decorator is a function that:
    1. Takes a function as input
    2. Returns a new function (the wrapper)
    3. The wrapper adds behavior before/after the original
    """
    @functools.wraps(func)  # Preserve original function's metadata
    def wrapper(*args, **kwargs):
        print(f"📞 Calling {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"✅ {func.__name__} returned: {result}")
        return result
    return wrapper


@simple_logger
def greet(name: str) -> str:
    return f"Hello, {name}!"


# When you write @simple_logger, Python does:
# greet = simple_logger(greet)

result = greet("AI")


# =============================================================================
# PART 2: Decorator with Arguments
# =============================================================================

print("\n=== Part 2: Decorator with Arguments ===")


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Retry decorator for flaky operations.
    
    When a decorator needs arguments, we need THREE levels:
    1. Outer function: receives decorator arguments
    2. Middle function: receives the function to decorate
    3. Inner function: the actual wrapper
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"  ⚠️  Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# Simulating a flaky API
call_count = 0

@retry(max_attempts=3, delay=0.05)
def flaky_api_call():
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"API failed (attempt {call_count})")
    return "Success!"


print("Calling flaky API:")
try:
    result = flaky_api_call()
    print(f"Result: {result}")
except Exception as e:
    print(f"All retries failed: {e}")


# =============================================================================
# PART 3: Caching Decorator
# =============================================================================

print("\n=== Part 3: Caching Decorator ===")


def cache(maxsize: int = 100):
    """Simple cache decorator for expensive computations."""
    def decorator(func: Callable) -> Callable:
        cache_dict: Dict[tuple, Any] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable key
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in cache_dict:
                print(f"  💾 Cache HIT for {func.__name__}")
                return cache_dict[key]
            
            print(f"  🔄 Cache MISS - computing {func.__name__}")
            result = func(*args, **kwargs)
            
            # Evict oldest if full (simple LRU)
            if len(cache_dict) >= maxsize:
                oldest = next(iter(cache_dict))
                del cache_dict[oldest]
            
            cache_dict[key] = result
            return result
        
        # Add method to inspect cache
        wrapper.cache_info = lambda: {"size": len(cache_dict), "maxsize": maxsize}
        return wrapper
    return decorator


@cache(maxsize=10)
def expensive_embedding(text: str) -> list:
    """Simulate expensive embedding computation."""
    time.sleep(0.1)  # Simulate delay
    return [0.1] * 10  # Fake embedding


print("First call (cache miss):")
emb1 = expensive_embedding("Hello world")

print("\nSecond call same input (cache hit):")
emb2 = expensive_embedding("Hello world")

print("\nThird call different input (cache miss):")
emb3 = expensive_embedding("Different text")

print(f"\nCache info: {expensive_embedding.cache_info()}")


# =============================================================================
# PART 4: Tool Decorator for Agents
# =============================================================================

print("\n=== Part 4: @tool Decorator ===")

# Tool registry
_tool_registry: Dict[str, Any] = {}


@dataclass
class ToolInfo:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]


def tool(
    name: str = None,
    description: str = None
):
    """Decorator to register functions as agent tools.
    
    Similar to how LangChain's @tool works.
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "No description"
        
        # Extract parameter info from type hints
        hints = func.__annotations__
        params = {k: str(v) for k, v in hints.items() if k != "return"}
        
        # Register the tool
        _tool_registry[tool_name] = ToolInfo(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters=params
        )
        
        print(f"🔧 Registered tool: {tool_name}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"  🔨 Executing tool: {tool_name}")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@tool(name="search", description="Search the web for information")
def web_search(query: str) -> str:
    return f"Search results for: {query}"


@tool(name="calculator", description="Perform mathematical calculations")
def calculate(expression: str) -> float:
    return eval(expression)  # Don't do this in production!


@tool()  # Uses function name and docstring
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": 22, "unit": unit}


print("\nRegistered tools:")
for name, info in _tool_registry.items():
    print(f"  - {name}: {info.description}")
    print(f"    Parameters: {info.parameters}")


# =============================================================================
# PART 5: Class Decorator
# =============================================================================

print("\n=== Part 5: Class Decorator ===")


def register_agent(registry: dict):
    """Class decorator that registers agent classes."""
    def decorator(cls):
        agent_name = getattr(cls, "name", cls.__name__)
        registry[agent_name] = cls
        print(f"📋 Registered agent: {agent_name}")
        return cls
    return decorator


agent_registry = {}


@register_agent(agent_registry)
class ReActAgent:
    name = "react"
    
    def run(self, task: str) -> str:
        return f"[ReAct] Processing: {task}"


@register_agent(agent_registry)
class ChainOfThoughtAgent:
    name = "cot"
    
    def run(self, task: str) -> str:
        return f"[CoT] Thinking: {task}"


print(f"\nAgent registry: {list(agent_registry.keys())}")


# =============================================================================
# PART 6: Timing Decorator
# =============================================================================

print("\n=== Part 6: Timing Decorator ===")


def timed(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️  {func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper


@timed
def slow_operation():
    time.sleep(0.1)
    return "Done!"


slow_operation()


# =============================================================================
# PART 7: Combining Decorators
# =============================================================================

print("\n=== Part 7: Stacking Decorators ===")


@simple_logger
@timed
@retry(max_attempts=2)
def complex_operation(x: int) -> int:
    """Multiple decorators applied."""
    if x < 0:
        raise ValueError("x must be positive")
    return x * 2


# Decorators apply bottom-up:
# 1. @retry wraps the function
# 2. @timed wraps that
# 3. @simple_logger wraps that

result = complex_operation(5)


# =============================================================================
# PART 8: Summary
# =============================================================================

print("\n=== Part 8: Summary ===")
print("""
┌────────────────────────────────────────────────────────────┐
│              DECORATOR PATTERNS                             │
├────────────────────────────────────────────────────────────┤
│ BASIC:                                                      │
│   @decorator                                                │
│   def func(): ...                                           │
│   # Same as: func = decorator(func)                         │
├────────────────────────────────────────────────────────────┤
│ WITH ARGUMENTS:                                             │
│   @decorator(arg)                                           │
│   def func(): ...                                           │
│   # Same as: func = decorator(arg)(func)                    │
├────────────────────────────────────────────────────────────┤
│ PRODUCTION PATTERNS:                                        │
│   @retry(attempts=3) - Handle transient failures            │
│   @cache(maxsize=100) - Memoize expensive operations        │
│   @tool(name="x") - Register agent tools                    │
│   @timed - Measure execution time                           │
│   @validate - Validate inputs                               │
├────────────────────────────────────────────────────────────┤
│ BEST PRACTICES:                                             │
│   ✓ Use @functools.wraps to preserve metadata               │
│   ✓ Keep decorators focused (single responsibility)         │
│   ✓ Document what the decorator does                        │
│   ✓ Consider order when stacking (bottom-up)                │
└────────────────────────────────────────────────────────────┘
""")
