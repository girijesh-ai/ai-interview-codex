# Module 04: Human-in-the-Loop Patterns

**Level:** Intermediate to Advanced  
**Duration:** 6-8 hours  
**Prerequisites:** Modules 01-03 completed  
**Updated:** December 2025 - Includes InjectedState and tool-based Command patterns

## Table of Contents
1. [Introduction](#introduction)
2. [Breakpoints and Interrupts](#breakpoints-and-interrupts)
3. [Tools with State Access (InjectedState)](#tools-with-state-access-inject edstate)
4. [Tools Returning Command](#tools-returning-command)
5. [Approval Workflows](#approval-workflows)
6. [Streaming and Interactive Execution](#streaming-and-interactive-execution)
7. [State Persistence for HITL](#state-persistence-for-hitl)
8. [Production HITL Patterns](#production-hitl-patterns)
9. [Best Practices](#best-practices)
10. [Summary & Next Steps](#summary--next-steps)

---

## Introduction

**Human-in-the-Loop (HITL)** patterns are essential for production AI systems where human oversight, approval, or interaction is required. This module covers both traditional HITL patterns and **2024-2025 innovations** like tools accessing state with `InjectedState` and returning `Command` objects.

> **🆕 What's New in 2024-2025**
> - `InjectedState` - Tools can access graph state invisibly to LLM
> - Tools returning `Command` - Tools can update state AND route
> - Enhanced breakpoint mechanisms
> - Production-grade approval workflows

### Learning Objectives

By the end of this module, you will:
- ✅ Implement breakpoints and interrupts for human review
- ✅ Create tools that access state with `InjectedState`
- ✅ Build tools that return `Command` for state updates and routing
- ✅ Design approval workflows for production
- ✅ Implement streaming for real-time interaction
- ✅ Persist state across human interactions

---

## Breakpoints and Interrupts

### Traditional Breakpoints

Pause execution before/after specific nodes for human review:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class ApprovalState(TypedDict):
    request: str
    analysis: str
    decision: str
    approved: bool

def analyze_request(state: ApprovalState):
    \"\"\"Analyze the request - runs automatically.\"\"\"
    analysis = perform_analysis(state['request'])
    return {"analysis": analysis}

def make_decision(state: ApprovalState):
    \"\"\"Make decision based on analysis - runs after approval.\"\"\"
    decision = decide(state['analysis'])
    return {"decision": decision, "approved": True}

# Build graph with breakpoint
workflow = StateGraph(ApprovalState)
workflow.add_node("analyze", analyze_request)
workflow.add_node("decide", make_decision)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "decide")
workflow.add_edge("decide", END)

# Compile with interrupt BEFORE decision node
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["decide"]  # Pause before decision for human review
)

# First run - stops at breakpoint
config = {"configurable": {"thread_id": "1"}}
result = app.invoke({"request": "Deploy to production"}, config)
print(f"Analysis: {result['analysis']}")
print("Waiting for human approval...")

# Human reviews, then continues
result = app.invoke(None, config)  # Resume from checkpoint
print(f"Decision: {result['decision']}")
```

```mermaid
graph LR
    START([START]) --> A[Analyze Request]
    A --> BP{{BREAKPOINT}}
    BP -.Human<br/>Review.-> D[Make Decision]
    D --> END([END])
    
    style START fill:#90EE90
    style BP fill:#FFD700
    style END fill:#FFB6C1
```

### Interrupt After Pattern

Review results after processing:

```python
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["analyze"]  # Review analysis before proceeding
)

# Run stops after analysis
result = app.invoke({"request": "Critical update"}, config)
# Human reviews analysis
# If approved, continue:
result = app.invoke(None, config)
```

---

## Tools with State Access (InjectedState)

### The InjectedState Pattern (December 2024)

Tools can access graph state without the LLM seeing it as a parameter. This is crucial for:
- Tracking usage statistics
- Accessing session data
- Implementing rate limiting
- Maintaining conversation context

### Basic InjectedState Example

```python
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated, TypedDict

class ConversationState(TypedDict):
    messages: list
    user_id: str
    usage_count: int
    session_data: dict

@tool
def search_with_tracking(
    query: str,
    state: Annotated[dict, InjectedState]
) -> str:
    \"\"\"
    Search tool that accesses state to track usage.
    
    The LLM only sees 'query' parameter.
    'state' is injected automatically by LangGraph.
    \"\"\"
    # Access state data
    user_id = state.get('user_id', 'unknown')
    usage_count = state.get('usage_count', 0)
    
    # Check rate limits
    if usage_count >= 10:
        return f"Rate limit reached for user {user_id}"
    
    # Perform search
    results = perform_search(query)
    
    # Return results (state update happens separately)
    return f"Found {len(results)} results: {results}"

# Tool sees full state, but LLM schema only has 'query'
print(search_with_tracking.args_schema.schema())
# Output: {'query': {'type': 'string'}}
# 'state' parameter is hidden from LLM!
```

### InjectedState with State Updates via Command

Tools can both access state AND update it by returning `Command`:

```python
from langgraph.types import Command

@tool
def smart_search(
    query: str,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"
    Search tool that updates usage tracking in state.
    \"\"\"
    # Access current state
    user_id = state.get('user_id')
    usage_count = state.get('usage_count', 0)
    
    # Perform search
    results = perform_search(query)
    
    # Return Command to update state
    return Command(
        # Update usage tracking
        update={
            "usage_count": usage_count + 1,
            "last_query": query,
            "search_results": results
        },
        # Optionally route to specific node
        # goto="process_results"  # Can add routing if needed
    )
```

### Real-World Example: Session-Aware Assistant

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain.tools import tool

class SessionState(TypedDict):
    messages: list
    user_id: str
    session_id: str
    user_preferences: dict
    api_calls_count: int
    last_activity: str

@tool
def get_user_data(
    data_type: str,
    state: Annotated[dict, InjectedState]
) -> str:
    \"\"\"
    Get user data based on session context.
    LLM only sees data_type parameter.
    \"\"\"
    user_id = state.get('user_id')
    preferences = state.get('user_preferences', {})
    
    if data_type == "preferences":
        return f"User preferences: {preferences}"
    elif data_type == "history":
        # Access user history from database using user_id
        history = fetch_user_history(user_id)
        return f"Recent history: {history}"
    
    return "Unknown data type"

@tool
def update_preferences(
    preference_key: str,
    preference_value: str,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"
    Update user preferences and track API usage.
    \"\"\"
    current_prefs = state.get('user_preferences', {})
    api_count = state.get('api_calls_count', 0)
    
    # Update preferences
    updated_prefs = {**current_prefs, preference_key: preference_value}
    
    # Save to database
    save_preferences(state['user_id'], updated_prefs)
    
    # Return Command with state updates
    return Command(
        update={
            "user_preferences": updated_prefs,
            "api_calls_count": api_count + 1,
            "last_activity": "preference_update"
        }
    )

# Use in agent graph
from langgraph.prebuilt import create_react_agent

tools = [get_user_data, update_preferences]
llm = ChatOpenAI(model="gpt-4")

agent = create_react_agent(
    llm,
    tools,
    state_schema=SessionState,
    checkpointer=MemorySaver()
)
```

**Key Benefits:**
1. **Privacy**: Sensitive state data hidden from LLM
2. **Efficiency**: Reduces token usage and latency
3. **Flexibility**: Tools can access any state field
4. **Tracking**: Easy to implement usage monitoring

---

## Tools Returning Command

### Pattern: Tool-Driven State Updates and Routing

Tools can update state and control graph flow by returning `Command`:

```python
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

@tool
def approval_required_action(
    action_type: str,
    action_data: dict,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"
    Perform action that requires approval.
    Routes to approval node after updating state.
    \"\"\"
    # Prepare action for approval
    action_details = {
        "type": action_type,
        "data": action_data,
        "requested_by": state.get('user_id'),
        "timestamp": datetime.now().isoformat()
    }
    
    # Check if auto-approval allowed
    if state.get('auto_approve_enabled') and is_low_risk(action_type):
        # Execute and complete
        result = execute_action(action_details)
        return Command(
            update={"action_result": result, "status": "completed"},
            goto=END
        )
    else:
        # Requires human approval
        return Command(
            update={
                "pending_action": action_details,
                "status": "awaiting_approval"
            },
            goto="human_approval_node"  # Route to approval workflow
        )

@tool
def search_and_analyze(
    query: str,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"
    Search and decide if deep analysis needed.
    \"\"\"
    results = search(query)
    
    # Determine if complex analysis needed
    if needs_deep_analysis(results):
        return Command(
            update={
                "search_results": results,
                "analysis_required": True
            },
            goto="deep_analysis_node"  # Dynamic routing
        )
    else:
        return Command(
            update={
                "search_results": results,
                "summary": quick_summary(results)
            },
            goto="respond_node"  # Skip analysis
        )
```

### Complete Example: Multi-Tool Workflow with Command Routing

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, ToolNode
from langchain.tools import tool

class WorkflowState(TypedDict):
    messages: list
    current_step: str
    results: dict
    requires_approval: bool
    approved: bool

@tool
def risk_assessment(
    proposal: str,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"Assess risk and route accordingly.\"\"\"
    risk_level = assess_risk(proposal)
    
    if risk_level == "high":
        return Command(
            update={
                "results": {"risk": "high", "requires_review": True},
                "requires_approval": True
            },
            goto="approval_node"
        )
    else:
        return Command(
            update={
                "results": {"risk": risk_level},
                "requires_approval": False
            },
            goto="execute_node"
        )

@tool
def execute_proposal(
    proposal: str,
    state: Annotated[dict, InjectedState]
) -> Command:
    \"\"\"Execute if approved or low risk.\"\"\"
    if state.get('requires_approval') and not state.get('approved'):
        return Command(
            update={"error": "Not approved"},
            goto="error_handler"
        )
    
    result = execute(proposal)
    return Command(
        update={"results": {**state['results'], "execution": result}},
        goto=END
    )

# Build workflow
workflow = StateGraph(WorkflowState)
tools = [risk_assessment, execute_proposal]

workflow.add_node("agent", create_react_agent(llm, tools))
workflow.add_node("approval_node", human_approval_node)
workflow.add_node("execute_node", ToolNode([execute_proposal]))
workflow.add_node("error_handler", error_handler_node)

workflow.add_edge(START, "agent")
app = workflow.compile(checkpointer=MemorySaver())
```

---

## Approval Workflows

### Pattern 1: Simple Approval Gate

```python
class ApprovalState(TypedDict):
    request: str
    analysis: dict
    approved: bool
    approver: str
    comments: str

def approval_gate(state: ApprovalState):
    \"\"\"Checkpoint for manual approval.\"\"\"
    # This node just waits - human updates state externally
    return {}  # No automatic updates

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_gate"]
)

# Initial run
config = {"configurable": {"thread_id": "approval-123"}}
app.invoke({"request": "Deploy model v2"}, config)

# Human reviews and approves
state = app.get_state(config)
state.values['approved'] = True
state.values['approver'] = "manager@company.com"
app.update_state(config, state.values)

# Continue execution
app.invoke(None, config)
```

### Pattern 2: Multi-Stage Approval

```python
class MultiApprovalState(TypedDict):
    request: str
    technical_approved: bool
    manager_approved: bool
    security_approved: bool
    final_decision: str

def technical_review(state):
    return {"technical_approved": False}  # Waits for approval

def manager_review(state):
    return {"manager_approved": False}

def security_review(state):
    return {"security_approved": False}

def final_decision(state):
    if all([
        state.get('technical_approved'),
        state.get('manager_approved'),
        state.get('security_approved')
    ]):
        return {"final_decision": "approved"}
    return {"final_decision": "rejected"}

workflow = StateGraph(MultiApprovalState)
workflow.add_node("technical", technical_review)
workflow.add_node("manager", manager_review)
workflow.add_node("security", security_review)
workflow.add_node("final", final_decision)

workflow.add_edge(START, "technical")
workflow.add_edge("technical", "manager")
workflow.add_edge("manager", "security")
workflow.add_edge("security", "final")
workflow.add_edge("final", END)

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["technical", "manager", "security"]
)
```

---

## Streaming and Interactive Execution

### Streaming Node Outputs

```python
# Stream updates as they occur
for chunk in app.stream({"request": "Process data"}):
    print(f"Update: {chunk}")
    # Shows each node's output in real-time

# Stream with specific config
config = {"configurable": {"thread_id": "stream-1"}}
for chunk in app.stream(initial_state, config):
    node_name = list(chunk.keys())[0]
    node_output = chunk[node_name]
    print(f"{node_name}: {node_output}")
```

### Interactive Chat Pattern

```python
class ChatState(TypedDict):
    messages: list
    awaiting_input: bool
    user_response: str

def ask_user_node(state: ChatState):
    \"\"\"Node that requests user input.\"\"\"
    question = "What would you like to do next?"
    return {
        "messages": state['messages'] + [AIMessage(content=question)],
        "awaiting_input": True
    }

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["ask_user"]
)

# Run until user input needed
result = app.invoke(initial_state, config)

# Get user input
user_input = input("Your response: ")

# Continue with user input
result = app.invoke(
    {"user_response": user_input, "awaiting_input": False},
    config
)
```

---

## State Persistence for HITL

### Using Checkpointers

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory (development)
checkpointer = MemorySaver()

# PostgreSQL (production)
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)

app = workflow.compile(checkpointer=checkpointer)

# Each thread maintains separate state
config1 = {"configurable": {"thread_id": "user-1"}}
config2 = {"configurable": {"thread_id": "user-2"}}

app.invoke(state1, config1)  # Independent from config2
app.invoke(state2, config2)
```

### State Inspection and Modification

```python
# Get current state
current_state = app.get_state(config)
print(current_state.values)
print(current_state.next)  # Next nodes to execute

# Modify state
current_state.values['approved'] = True
app.update_state(config, current_state.values)

# Replay from specific checkpoint
history = app.get_state_history(config)
for state in history:
    print(f"Step {state.step}: {state.values}")
```

---

## Production HITL Patterns

### Pattern: Approval with Timeout

```python
from datetime import datetime, timedelta

class TimedApprovalState(TypedDict):
    request: str
    approval_deadline: str
    approved: bool
    auto_approved: bool

def check_approval_timeout(state: TimedApprovalState):
    deadline = datetime.fromisoformat(state['approval_deadline'])
    
    if datetime.now() > deadline:
        # Auto-approve low-risk or escalate
        if is_low_risk(state['request']):
            return {"approved": True, "auto_approved": True}
        else:
            escalate_to_senior(state['request'])
            return {"approved": False}
    
    return {}  # Wait for manual approval
```

### Pattern: Collaborative Review

```python
class CollaborativeState(TypedDict):
    document: str
    reviewers: list
    reviews: dict  # {reviewer_id: review}
    all_approved: bool

def aggregate_reviews(state: CollaborativeState):
    \"\"\"Check if all reviewers approved.\"\"\"
    required_reviewers = set(state['reviewers'])
    completed_reviews = set(state['reviews'].keys())
    
    if required_reviewers == completed_reviews:
        all_approved = all(
            review['approved']
            for review in state['reviews'].values()
        )
        return {"all_approved": all_approved}
    
    return {}  # Wait for more reviews
```

---

## Best Practices

### 1. Always Use Checkpointers for HITL

```python
# ✅ GOOD: Persist state for resumption
app = workflow.compile(checkpointer=MemorySaver())

# ❌ BAD: No checkpoint means can't resume
app = workflow.compile()  # State lost after pause
```

### 2. Use InjectedState for Sensitive Data

```python
# ✅ GOOD: Hide sensitive data from LLM
@tool
def access_db(query: str, state: Annotated[dict, InjectedState]):
    api_key = state['api_key']  # Not visible to LLM
    return db.query(query, api_key)

# ❌ BAD: Exposing secrets to LLM
@tool
def bad_access_db(query: str, api_key: str):  # LLM might log/leak this!
    return db.query(query, api_key)
```

### 3. Provide Clear Context at Breakpoints

```python
def review_node(state):
    \"\"\"Clear documentation for human reviewer.\"\"\"
    return {
        "review_instructions": "Check analysis for accuracy",
        "review_checklist": ["Data quality", "Logic soundness", "Risk assessment"],
        "estimated_review_time": "5 minutes"
    }
```

### 4. Handle Resume Failures Gracefully

```python
try:
    result = app.invoke(None, config)  # Resume
except Exception as e:
    # State may have changed incompatibly
    logger.error(f"Resume failed: {e}")
    # Provide fallback or notify user
```

---

## Summary & Next Steps

### What You Learned

✅ **Breakpoints and Interrupts**: Pause execution for human review  
✅ **InjectedState**: Tools accessing state invisibly to LLM  
✅ **Tools with Command**: State updates and routing from tools  
✅ **Approval Workflows**: Multi-stage, production-grade patterns  
✅ **Streaming**: Real-time interaction and updates  
✅ **State Persistence**: Checkpoint management for resumable workflows  

### Key Takeaways

1. **InjectedState is Essential** for production tools needing context
2. **Command from Tools** enables sophisticated tool-driven workflows
3. **Checkpointers are Required** for any pause/resume functionality
4. **Design for Resumption** - assume execution can stop anytime
5. **Clear Documentation** at breakpoints helps human reviewers

### Practice Now

Continue to **[module-04-practice.ipynb](./module-04-practice.ipynb)** to practice:
- Building approval workflows
- Creating tools with InjectedState
- Implementing tools that return Command
- Streaming and interactive patterns

### Next Module

**Module 05: Persistence & Memory** will cover:
- Advanced checkpointer patterns
- Cross-session memory management
- Time travel debugging
- Memory optimization strategies

---

## Additional Resources

- [LangGraph HITL Documentation](https://python.langchain.com/docs/langgraph/)
- [InjectedState API Reference](https://python.langchain.com/api/langgraph/)
- [Checkpointers Guide](https://python.langchain.com/docs/langgraph/)
- [Command Tool Documentation](https://python.langchain.com/docs/langgraph/)

---

**Ready to build production-grade interactive systems?** Let's practice! 🚀
