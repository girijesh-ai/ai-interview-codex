"""
Module 04, Example 05: Pattern Matching for AI

This example demonstrates:
- Basic pattern matching (Python 3.10+)
- Matching dataclasses
- Guard clauses
- Routing tool calls
- Parsing LLM responses

Run this file:
    python 05_pattern_matching.py

Follow along with: 04-modern-python-ai-features.md

NOTE: Requires Python 3.10+
"""

import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Check Python version
if sys.version_info < (3, 10):
    print("=" * 60)
    print(f"Pattern matching requires Python 3.10+")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
    print("=" * 60)
    print("\nShowing pattern matching concepts as pseudo-code...\n")
    PATTERN_MATCHING_AVAILABLE = False
else:
    PATTERN_MATCHING_AVAILABLE = True


# =============================================================================
# Data Structures for Examples
# =============================================================================

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

@dataclass
class AgentAction:
    action: str
    thought: str
    input: str

@dataclass 
class ChatMessage:
    role: str
    content: str


# =============================================================================
# PART 1: Basic Pattern Matching
# =============================================================================

print("=== Part 1: Basic Pattern Matching ===")

if PATTERN_MATCHING_AVAILABLE:
    def get_llm_model_info(model_name: str) -> str:
        """Basic string pattern matching."""
        match model_name:
            case "gpt-4":
                return "OpenAI GPT-4: 128K context, $0.03/1K"
            case "gpt-3.5-turbo":
                return "OpenAI GPT-3.5: 16K context, $0.001/1K"  
            case "claude-3-opus":
                return "Anthropic Opus: 200K context, $0.015/1K"
            case name if name.startswith("claude"):
                return f"Anthropic model: {name}"
            case name if name.startswith("gpt"):
                return f"OpenAI model: {name}"
            case _:
                return f"Unknown model: {model_name}"
    
    print(f"gpt-4: {get_llm_model_info('gpt-4')}")
    print(f"claude-3-opus: {get_llm_model_info('claude-3-opus')}")
    print(f"other: {get_llm_model_info('llama-3')}")
else:
    print("Pattern matching syntax (requires Python 3.10+):")
    print("""
    match model_name:
        case "gpt-4":
            return "OpenAI GPT-4"
        case "claude-3-opus":
            return "Anthropic Opus"
        case _:
            return "Unknown"
    """)


# =============================================================================
# PART 2: Matching Dataclasses
# =============================================================================

print("\n=== Part 2: Matching Dataclasses ===")

if PATTERN_MATCHING_AVAILABLE:
    def execute_tool(tool_call: ToolCall) -> str:
        """Route tool calls using pattern matching."""
        
        match tool_call:
            # Match specific tool with extracted args
            case ToolCall(name="search", arguments={"query": query}):
                return f"🔍 Searching for: {query}"
            
            # Match with multiple args
            case ToolCall(name="calculator", arguments={"expression": expr}):
                try:
                    result = eval(expr)  # Don't do in production!
                    return f"🔢 {expr} = {result}"
                except:
                    return f"❌ Invalid expression: {expr}"
            
            # Match with optional args (guard clause)
            case ToolCall(name="weather", arguments=args) if "city" in args:
                city = args["city"]
                unit = args.get("unit", "celsius")
                return f"🌤️ Weather in {city}: 22°{unit[0].upper()}"
            
            # Match any tool by name only
            case ToolCall(name=name, arguments=args):
                return f"❓ Unknown tool: {name} with args: {args}"
    
    # Test different tool calls
    tools = [
        ToolCall("search", {"query": "Python tutorials"}),
        ToolCall("calculator", {"expression": "2 + 3 * 4"}),
        ToolCall("weather", {"city": "London"}),
        ToolCall("unknown", {"data": "test"}),
    ]
    
    for tool in tools:
        result = execute_tool(tool)
        print(f"  {result}")

else:
    print("Tool routing with pattern matching:")
    print("""
    match tool_call:
        case ToolCall(name="search", arguments={"query": query}):
            return f"Searching: {query}"
        case ToolCall(name=name):
            return f"Unknown: {name}"
    """)


# =============================================================================
# PART 3: Parsing LLM Responses
# =============================================================================

print("\n=== Part 3: Parsing LLM Responses ===")

if PATTERN_MATCHING_AVAILABLE:
    def parse_llm_response(response: dict) -> str:
        """Parse various LLM response formats."""
        
        match response:
            # Standard completion
            case {"content": str(content), "finish_reason": "stop"}:
                return f"✅ Complete: {content[:50]}..."
            
            # Tool calls present
            case {"content": content, "tool_calls": [first, *rest]}:
                tool_name = first.get("function", {}).get("name", "unknown")
                return f"🔧 Tool call: {tool_name} (+{len(rest)} more)"
            
            # Length limit reached
            case {"content": content, "finish_reason": "length"}:
                return f"⚠️ Truncated: {content[:30]}..."
            
            # Streaming delta
            case {"delta": {"content": str(chunk)}}:
                return f"📨 Chunk: {chunk}"
            
            # Error response
            case {"error": {"message": msg, "code": code}}:
                return f"❌ Error {code}: {msg}"
            
            # Empty/null response
            case {"content": None} | {"content": ""}:
                return "⚪ Empty response"
            
            case _:
                return f"❓ Unknown format: {list(response.keys())}"
    
    # Test different response formats
    responses = [
        {"content": "Here is the answer to your question...", "finish_reason": "stop"},
        {"content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
        {"content": "Long response that got cut off...", "finish_reason": "length"},
        {"delta": {"content": "streaming"}},
        {"error": {"message": "Rate limited", "code": 429}},
        {"content": None},
    ]
    
    for resp in responses:
        result = parse_llm_response(resp)
        print(f"  {result}")

else:
    print("Response parsing with pattern matching:")
    print("""
    match response:
        case {"content": str(content), "finish_reason": "stop"}:
            return "Complete"
        case {"error": {"message": msg}}:
            return f"Error: {msg}"
    """)


# =============================================================================
# PART 4: Agent Action Processing
# =============================================================================

print("\n=== Part 4: Agent Actions ===")

if PATTERN_MATCHING_AVAILABLE:
    def process_agent_action(action: AgentAction) -> str:
        """Process ReAct agent actions."""
        
        match action:
            # Final answer
            case AgentAction(action="Final Answer", input=answer):
                return f"✅ DONE: {answer}"
            
            # Search with query length validation
            case AgentAction(action="Search", input=query) if len(query) > 100:
                return f"❌ Query too long ({len(query)} chars)"
            
            case AgentAction(action="Search", input=query):
                return f"🔍 Searching: {query}"
            
            # Calculate
            case AgentAction(action="Calculate", input=expr):
                return f"🔢 Calculating: {expr}"
            
            # Lookup with thought inspection
            case AgentAction(action="Lookup", thought=thought, input=term) if "verify" in thought.lower():
                return f"✓ Verification lookup: {term}"
            
            case AgentAction(action="Lookup", input=term):
                return f"📖 Looking up: {term}"
            
            # Unknown action
            case AgentAction(action=unknown):
                return f"❓ Unknown action: {unknown}"
    
    # Test different actions
    actions = [
        AgentAction("Search", "I need to find information", "Python OOP"),
        AgentAction("Calculate", "Let me compute this", "100 * 0.05"),
        AgentAction("Lookup", "I should verify this fact", "Python creator"),
        AgentAction("Final Answer", "I have the answer", "Python was created by Guido"),
        AgentAction("Unknown", "Testing", "data"),
    ]
    
    for action in actions:
        result = process_agent_action(action)
        print(f"  {result}")

else:
    print("Agent action processing:")
    print("""
    match action:
        case AgentAction(action="Final Answer", input=answer):
            return f"Done: {answer}"
        case AgentAction(action="Search", input=query):
            return f"Searching: {query}"
    """)


# =============================================================================
# PART 5: Matching Sequences
# =============================================================================

print("\n=== Part 5: Matching Sequences ===")

if PATTERN_MATCHING_AVAILABLE:
    def analyze_conversation(messages: List[ChatMessage]) -> str:
        """Analyze conversation structure."""
        
        match messages:
            # Empty
            case []:
                return "Empty conversation"
            
            # Single message
            case [ChatMessage(role="user", content=content)]:
                return f"Single user message: {content[:30]}..."
            
            # System + user (common start)
            case [ChatMessage(role="system"), ChatMessage(role="user", content=q)]:
                return f"New conversation with: {q[:30]}..."
            
            # Any messages ending with assistant response
            case [*_, ChatMessage(role="assistant", content=last)]:
                return f"Last response: {last[:30]}..."
            
            # Pattern with variable middle
            case [first, *middle, last]:
                return f"Conversation: {len(middle)+2} messages"
            
            case _:
                return "Unknown pattern"
    
    conversations = [
        [],
        [ChatMessage("user", "Hello!")],
        [ChatMessage("system", "You are helpful."), ChatMessage("user", "Hi")],
        [
            ChatMessage("system", "Be helpful"),
            ChatMessage("user", "Question"),
            ChatMessage("assistant", "Here is my answer...")
        ],
    ]
    
    for conv in conversations:
        result = analyze_conversation(conv)
        print(f"  {result}")

else:
    print("Sequence matching:")
    print("""
    match messages:
        case []:
            return "Empty"
        case [first, *rest]:
            return f"Has {len(rest)+1} messages"
    """)


# =============================================================================
# PART 6: Summary
# =============================================================================

print("\n=== Part 6: Summary ===")
print("""
┌─────────────────────────────────────────────────────────────┐
│              PATTERN MATCHING FOR AI (3.10+)                 │
├─────────────────────────────────────────────────────────────┤
│ BASIC:                                                       │
│   match value:                                               │
│       case "literal": ...                                    │
│       case _: ...  # default                                 │
├─────────────────────────────────────────────────────────────┤
│ DATACLASS MATCHING:                                          │
│   case ToolCall(name="search", arguments=args):              │
│       # Extract and use args                                 │
├─────────────────────────────────────────────────────────────┤
│ GUARD CLAUSES:                                               │
│   case ToolCall(name=n) if len(n) > 10:                      │
│       # Only matches if condition is true                    │
├─────────────────────────────────────────────────────────────┤
│ DICTIONARY MATCHING:                                         │
│   case {"key": value, **rest}:                               │
│       # Extract key, capture rest                            │
├─────────────────────────────────────────────────────────────┤
│ SEQUENCE MATCHING:                                           │
│   case [first, *middle, last]:                               │
│       # Unpack sequences                                     │
├─────────────────────────────────────────────────────────────┤
│ AI USE CASES:                                                │
│   ✓ Route tool calls by name/args                            │
│   ✓ Parse LLM response formats                               │
│   ✓ Process agent actions                                    │
│   ✓ Analyze conversation structure                           │
└─────────────────────────────────────────────────────────────┘
""")
