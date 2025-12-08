"""
Module 04, Example 02: Dataclasses for AI Data

This example demonstrates:
- Basic dataclasses
- Dataclass options (frozen, order, slots, kw_only)
- field() options
- __post_init__ validation

Run this file:
    python 02_dataclasses.py

Follow along with: 04-modern-python-ai-features.md
"""

from dataclasses import dataclass, field, asdict, astuple
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid


# =============================================================================
# PART 1: Basic Dataclass
# =============================================================================

print("=== Part 1: Basic Dataclass ===")

# WITHOUT dataclass - lots of boilerplate
class ChatMessageOld:
    def __init__(self, role: str, content: str, name: str = None):
        self.role = role
        self.content = content
        self.name = name
    
    def __repr__(self):
        return f"ChatMessageOld(role={self.role!r}, content={self.content!r})"
    
    def __eq__(self, other):
        return (self.role == other.role and 
                self.content == other.content and
                self.name == other.name)

# WITH dataclass - automatic __init__, __repr__, __eq__!
@dataclass
class ChatMessage:
    role: str
    content: str
    name: Optional[str] = None

msg1 = ChatMessage("user", "Hello!")
msg2 = ChatMessage("user", "Hello!")
msg3 = ChatMessage("assistant", "Hi there!")

print(f"Message: {msg1}")
print(f"msg1 == msg2: {msg1 == msg2}")  # True - auto __eq__
print(f"msg1 == msg3: {msg1 == msg3}")  # False


# =============================================================================
# PART 2: Dataclass Options
# =============================================================================

print("\n=== Part 2: Dataclass Options ===")

# frozen=True - Immutable (can be used in sets/dicts)
@dataclass(frozen=True)
class ImmutableMessage:
    role: str
    content: str

immutable = ImmutableMessage("user", "Can't change me")
try:
    immutable.role = "assistant"  # Error!
except Exception as e:
    print(f"frozen=True prevents modification: {type(e).__name__}")

# Can use in sets because it's hashable
message_set = {ImmutableMessage("user", "Hi"), ImmutableMessage("user", "Hi")}
print(f"Set has {len(message_set)} unique messages")

# order=True - Auto-generate comparison methods
@dataclass(order=True)
class PrioritizedTask:
    priority: int
    name: str = field(compare=False)  # Don't include in ordering

tasks = [
    PrioritizedTask(3, "Low priority"),
    PrioritizedTask(1, "High priority"),
    PrioritizedTask(2, "Medium priority"),
]
sorted_tasks = sorted(tasks)
print("Sorted tasks by priority:")
for task in sorted_tasks:
    print(f"  {task.priority}: {task.name}")


# =============================================================================
# PART 3: Using field()
# =============================================================================

print("\n=== Part 3: field() Options ===")

@dataclass
class ToolCall:
    """Tool call from LLM with auto-generated ID."""
    
    # Auto-generate unique ID
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Regular fields
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    # Not included in repr (for sensitive data)
    api_key: Optional[str] = field(default=None, repr=False)
    
    # Not included in comparisons
    timestamp: datetime = field(
        default_factory=datetime.now,
        compare=False
    )

tool1 = ToolCall(name="search", arguments={"query": "python"})
tool2 = ToolCall(name="search", arguments={"query": "python"})

print(f"Tool: {tool1}")  # api_key not shown
print(f"ID auto-generated: {tool1.id}")
print(f"Same tool (ignoring timestamp): {tool1 == tool2}")


# =============================================================================
# PART 4: Complex AI Data Structure
# =============================================================================

print("\n=== Part 4: Production LLMResponse ===")

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass
class LLMResponse:
    """Complete LLM API response."""
    
    content: str
    model: str
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"
    
    # Complex fields with defaults
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Raw response not in repr
    raw: Optional[Dict] = field(default=None, repr=False)
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

response = LLMResponse(
    content="Here's the search result...",
    model="gpt-4",
    usage=Usage(prompt_tokens=50, completion_tokens=100),
    tool_calls=[
        ToolCall(name="search", arguments={"query": "python"})
    ]
)

print(f"Response: {response}")
print(f"Has tool calls: {response.has_tool_calls}")
print(f"Total tokens: {response.usage.total_tokens}")


# =============================================================================
# PART 5: __post_init__ Validation
# =============================================================================

print("\n=== Part 5: __post_init__ Validation ===")

@dataclass
class AgentStep:
    """Single step in ReAct agent execution."""
    
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Optional[str] = None
    
    VALID_ACTIONS = frozenset(["search", "calculate", "lookup", "finish"])
    
    def __post_init__(self):
        """Validate after __init__ completes."""
        
        # Validate thought
        if not self.thought.strip():
            raise ValueError("Thought cannot be empty")
        
        # Validate action
        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {self.action}. "
                f"Must be one of: {self.VALID_ACTIONS}"
            )
        
        # Normalize thought
        self.thought = self.thought.strip()

# Valid step
step = AgentStep(
    thought="I need to search for Python info",
    action="search",
    action_input={"query": "Python programming"}
)
print(f"Valid step: {step.action}")

# Invalid step
try:
    bad_step = AgentStep(
        thought="Let me think...",
        action="invalid_action",  # Not in VALID_ACTIONS
        action_input={}
    )
except ValueError as e:
    print(f"Validation error: {e}")


# =============================================================================
# PART 6: Conversion Utilities
# =============================================================================

print("\n=== Part 6: Conversion Utilities ===")

@dataclass
class SimpleMessage:
    role: str
    content: str

msg = SimpleMessage("user", "Hello!")

# Convert to dict
msg_dict = asdict(msg)
print(f"As dict: {msg_dict}")

# Convert to tuple
msg_tuple = astuple(msg)
print(f"As tuple: {msg_tuple}")

# Reconstruct from dict
msg_copy = SimpleMessage(**msg_dict)
print(f"Reconstructed: {msg_copy}")


# =============================================================================
# PART 7: Summary
# =============================================================================

print("\n=== Part 7: Summary ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                    DATACLASS PATTERNS                        │
├─────────────────────────────────────────────────────────────┤
│ BASIC:                                                       │
│   @dataclass                                                 │
│   class Message:                                             │
│       role: str                                              │
│       content: str                                           │
├─────────────────────────────────────────────────────────────┤
│ OPTIONS:                                                     │
│   frozen=True    → Immutable, hashable                       │
│   order=True     → Sortable (__lt__, __le__, etc.)           │
│   slots=True     → Memory efficient (Python 3.10+)           │
│   kw_only=True   → All fields keyword-only (Python 3.10+)    │
├─────────────────────────────────────────────────────────────┤
│ field() OPTIONS:                                             │
│   default_factory=list  → Mutable defaults                   │
│   repr=False            → Hide from __repr__                 │
│   compare=False         → Exclude from equality              │
├─────────────────────────────────────────────────────────────┤
│ UTILITIES:                                                   │
│   asdict(obj)           → Convert to dictionary              │
│   astuple(obj)          → Convert to tuple                   │
│   __post_init__()       → Validation after init              │
└─────────────────────────────────────────────────────────────┘
""")
