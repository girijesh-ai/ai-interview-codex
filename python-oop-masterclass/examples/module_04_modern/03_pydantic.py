"""
Module 04, Example 03: Pydantic v2 for Production AI

This example demonstrates:
- Basic Pydantic models
- Field validation
- Nested models
- Settings management
- JSON schema generation

Run this file:
    python 03_pydantic.py

Follow along with: 04-modern-python-ai-features.md

NOTE: Requires pydantic >= 2.0
    pip install pydantic pydantic-settings
"""

# Try to import pydantic, provide fallback message if not installed
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("=" * 60)
    print("Pydantic is not installed. Install with:")
    print("    pip install pydantic pydantic-settings")
    print("=" * 60)
    print("\nShowing Pydantic concepts with dataclass fallback...\n")

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# PART 1: Basic Pydantic Model vs Dataclass
# =============================================================================

print("=== Part 1: Basic Validation ===")

if PYDANTIC_AVAILABLE:
    class ChatMessage(BaseModel):
        """Pydantic model with automatic validation."""
        
        role: Literal["system", "user", "assistant", "tool"]
        content: str = Field(..., min_length=1, max_length=100000)
        name: Optional[str] = Field(None, max_length=64)
        
        @field_validator('content')
        @classmethod
        def content_not_blank(cls, v: str) -> str:
            if not v.strip():
                raise ValueError('Content cannot be just whitespace')
            return v.strip()
    
    # Valid message
    msg = ChatMessage(role="user", content="Hello!")
    print(f"Valid message: {msg}")
    
    # Validation happens automatically!
    try:
        bad_msg = ChatMessage(
            role="invalid",  # Not in Literal
            content="Hi"
        )
    except Exception as e:
        print(f"Validation error: {type(e).__name__}")
    
    try:
        empty_msg = ChatMessage(role="user", content="   ")  # Just whitespace
    except Exception as e:
        print(f"Whitespace validation: {type(e).__name__}")

else:
    # Dataclass fallback (no automatic validation)
    @dataclass
    class ChatMessage:
        role: str
        content: str
        name: Optional[str] = None
    
    msg = ChatMessage(role="user", content="Hello!")
    print(f"Message (no validation): {msg}")


# =============================================================================
# PART 2: Complex Validation
# =============================================================================

print("\n=== Part 2: Complex Validation ===")

if PYDANTIC_AVAILABLE:
    class CompletionRequest(BaseModel):
        """API request with comprehensive validation."""
        
        model: str = Field(..., pattern=r'^(gpt-4|gpt-3\.5|claude)')
        messages: List[ChatMessage] = Field(..., min_length=1)
        temperature: float = Field(0.7, ge=0, le=2)
        max_tokens: int = Field(4096, ge=1, le=128000)
        stream: bool = False
        
        @model_validator(mode='after')
        def validate_messages(self):
            """Validate message list has correct structure."""
            # First message should be system or user
            if self.messages and self.messages[0].role not in ["system", "user"]:
                raise ValueError("First message must be system or user")
            return self
        
        model_config = ConfigDict(
            json_schema_extra={
                "examples": [{
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                }]
            }
        )
    
    # Valid request
    request = CompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="Hi!")],
        temperature=0.5
    )
    print(f"Valid request: model={request.model}, temp={request.temperature}")
    
    # Invalid temperature
    try:
        bad_request = CompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi!")],
            temperature=3.0  # > 2
        )
    except Exception as e:
        print(f"Temperature validation: {type(e).__name__}")

else:
    print("Skipping (pydantic not installed)")


# =============================================================================
# PART 3: Nested Models
# =============================================================================

print("\n=== Part 3: Nested Models ===")

if PYDANTIC_AVAILABLE:
    class ToolCall(BaseModel):
        """Tool call from LLM."""
        id: str
        name: str
        arguments: Dict[str, Any]
    
    class Usage(BaseModel):
        """Token usage."""
        prompt_tokens: int = Field(ge=0)
        completion_tokens: int = Field(ge=0)
        
        @property
        def total_tokens(self) -> int:
            return self.prompt_tokens + self.completion_tokens
    
    class LLMResponse(BaseModel):
        """Complete LLM response."""
        content: str
        model: str
        finish_reason: Literal["stop", "length", "tool_calls"] = "stop"
        tool_calls: List[ToolCall] = Field(default_factory=list)
        usage: Usage
        created_at: datetime = Field(default_factory=datetime.now)
        
        model_config = ConfigDict(
            # Validate on assignment (not just creation)
            validate_assignment=True,
        )
    
    response = LLMResponse(
        content="Here is your answer...",
        model="gpt-4",
        usage=Usage(prompt_tokens=50, completion_tokens=100),
        tool_calls=[
            ToolCall(id="1", name="search", arguments={"query": "python"})
        ]
    )
    
    print(f"Response model: {response.model}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Tool calls: {len(response.tool_calls)}")

else:
    print("Skipping (pydantic not installed)")


# =============================================================================
# PART 4: JSON Serialization
# =============================================================================

print("\n=== Part 4: JSON Serialization ===")

if PYDANTIC_AVAILABLE:
    # To JSON
    json_str = response.model_dump_json(indent=2)
    print(f"As JSON:\n{json_str[:200]}...")
    
    # From JSON
    parsed = LLMResponse.model_validate_json(json_str)
    print(f"Parsed back: {parsed.model}")
    
    # JSON schema (for OpenAI function calling)
    schema = LLMResponse.model_json_schema()
    print(f"Schema keys: {list(schema.keys())}")

else:
    print("Skipping (pydantic not installed)")


# =============================================================================
# PART 5: Settings from Environment
# =============================================================================

print("\n=== Part 5: Settings Management ===")

try:
    from pydantic_settings import BaseSettings
    from pydantic import SecretStr
    
    class AISettings(BaseSettings):
        """Load settings from environment variables."""
        
        # These read from OPENAI_API_KEY, DEFAULT_MODEL, etc.
        openai_api_key: SecretStr = Field(default="sk-test-key")
        anthropic_api_key: Optional[SecretStr] = None
        default_model: str = Field(default="gpt-4")
        temperature: float = Field(default=0.7, ge=0, le=2)
        max_retries: int = Field(default=3, ge=1, le=10)
        
        model_config = ConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
        )
    
    # Create settings (would normally load from .env)
    settings = AISettings()
    
    print(f"Model: {settings.default_model}")
    print(f"Temperature: {settings.temperature}")
    # SecretStr hides the value in logs
    print(f"API Key (hidden): {settings.openai_api_key}")
    # Get actual value when needed
    # actual_key = settings.openai_api_key.get_secret_value()

except ImportError:
    print("pydantic-settings not installed")
    print("Install with: pip install pydantic-settings")


# =============================================================================
# PART 6: Structured Output Parsing
# =============================================================================

print("\n=== Part 6: Structured Output Parsing ===")

if PYDANTIC_AVAILABLE:
    class ExtractedEntity(BaseModel):
        """Entity extracted from text."""
        name: str
        entity_type: Literal["person", "org", "location", "date"]
        confidence: float = Field(ge=0, le=1)
    
    class DocumentAnalysis(BaseModel):
        """Structured document analysis - LLM output schema."""
        summary: str = Field(..., max_length=500)
        key_points: List[str] = Field(..., max_length=10)
        entities: List[ExtractedEntity]
        sentiment: Literal["positive", "negative", "neutral"]
    
    # Simulate parsing LLM JSON output
    llm_output = '''
    {
        "summary": "Python is a versatile programming language.",
        "key_points": ["Easy to learn", "Wide ecosystem", "Used in AI"],
        "entities": [
            {"name": "Python", "entity_type": "org", "confidence": 0.95}
        ],
        "sentiment": "positive"
    }
    '''
    
    analysis = DocumentAnalysis.model_validate_json(llm_output)
    print(f"Summary: {analysis.summary}")
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Entities: {[e.name for e in analysis.entities]}")

else:
    print("Skipping (pydantic not installed)")


# =============================================================================
# PART 7: Summary
# =============================================================================

print("\n=== Part 7: Summary ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│                    PYDANTIC V2 FOR AI                        │
├─────────────────────────────────────────────────────────────┤
│ BASIC MODEL:                                                 │
│   class MyModel(BaseModel):                                  │
│       field: str = Field(..., min_length=1)                  │
├─────────────────────────────────────────────────────────────┤
│ VALIDATORS:                                                  │
│   @field_validator('field')                                  │
│   @model_validator(mode='after')                             │
├─────────────────────────────────────────────────────────────┤
│ SERIALIZATION:                                               │
│   model.model_dump()          → dict                         │
│   model.model_dump_json()     → JSON string                  │
│   Model.model_validate_json() → Parse JSON                   │
├─────────────────────────────────────────────────────────────┤
│ AI USE CASES:                                                │
│   ✓ Validate LLM structured outputs                          │
│   ✓ Generate JSON schemas for function calling               │
│   ✓ API request/response validation                          │
│   ✓ Settings from environment (.env)                         │
│   ✓ SecretStr for API keys                                   │
└─────────────────────────────────────────────────────────────┘
""")
