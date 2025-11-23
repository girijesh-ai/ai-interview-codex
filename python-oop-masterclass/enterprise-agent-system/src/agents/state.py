"""
LangGraph State Management with Advanced Features

Demonstrates:
- Custom state with typed channels
- Multiple reducer types
- Short-term memory via state
- State persistence patterns
- Message management
"""

from __future__ import annotations
import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, List, Dict, Any, Optional, Set
from typing_extensions import TypedDict
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from ..domain.models import (
    Priority,
    RequestStatus,
    RequestCategory,
    AgentType,
    DecisionType,
    Confidence
)


# ============================================================================
# CUSTOM REDUCERS - How state updates are combined
# ============================================================================

def add_decision(
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Custom reducer for decisions - deduplicate and maintain order.

    Args:
        existing: Existing decisions
        new: New decisions to add

    Returns:
        Combined decisions with no duplicates
    """
    # Deduplicate by decision_id
    all_decisions = existing + new
    seen_ids = set()
    unique_decisions = []

    for decision in all_decisions:
        decision_id = decision.get("decision_id")
        if decision_id not in seen_ids:
            seen_ids.add(decision_id)
            unique_decisions.append(decision)

    # Sort by timestamp
    unique_decisions.sort(key=lambda d: d.get("timestamp", 0))

    return unique_decisions


def merge_metrics(
    existing: Dict[str, float],
    new: Dict[str, float]
) -> Dict[str, float]:
    """Custom reducer for metrics - intelligent aggregation.

    Args:
        existing: Existing metrics
        new: New metrics to merge

    Returns:
        Merged metrics with proper aggregation
    """
    result = existing.copy()

    for key, value in new.items():
        if key.endswith('_count'):
            # Sum counts
            result[key] = result.get(key, 0) + value
        elif key.endswith('_avg'):
            # Weighted average
            count_key = key.replace('_avg', '_count')
            old_count = existing.get(count_key, 0)
            new_count = new.get(count_key, 1)
            total_count = old_count + new_count

            if total_count > 0:
                result[key] = (
                    (existing.get(key, 0) * old_count + value * new_count)
                    / total_count
                )
        elif key.endswith('_max'):
            # Keep maximum
            result[key] = max(result.get(key, 0), value)
        elif key.endswith('_min'):
            # Keep minimum
            result[key] = min(result.get(key, float('inf')), value)
        else:
            # Replace for other keys
            result[key] = value

    return result


def union_sets(
    existing: Set[str],
    new: Set[str]
) -> Set[str]:
    """Reducer for sets - union operation.

    Args:
        existing: Existing set
        new: New set

    Returns:
        Union of both sets
    """
    return existing | new


def merge_context(
    existing: Dict[str, Any],
    new: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer for context dictionaries - deep merge.

    Args:
        existing: Existing context
        new: New context to merge

    Returns:
        Deeply merged context
    """
    result = existing.copy()

    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_context(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # Append to lists
            result[key] = result[key] + value
        else:
            # Replace for other types
            result[key] = value

    return result


# ============================================================================
# AGENT STATE - Main state schema with all channels
# ============================================================================

class AgentState(TypedDict):
    """Main state schema for the agent system.

    Demonstrates:
    - Multiple channel types
    - Custom reducers for complex state
    - Short-term memory via state
    - Type safety with TypedDict
    """

    # ========================================================================
    # CONVERSATION HISTORY - Using built-in add_messages reducer
    # ========================================================================

    # Messages are automatically appended
    # Supports: HumanMessage, AIMessage, SystemMessage, ToolMessage
    messages: Annotated[List[BaseMessage], add_messages]

    # ========================================================================
    # REQUEST CONTEXT - Simple replacement (no reducer)
    # ========================================================================

    # Core identifiers
    request_id: str
    customer_id: str
    thread_id: str  # For LangGraph threading

    # Request metadata
    priority: int  # Priority.value
    category: Optional[str]  # RequestCategory.value
    status: str  # RequestStatus.value

    # ========================================================================
    # AGENT COORDINATION - Track workflow
    # ========================================================================

    # Current agent handling request
    current_agent: str  # AgentType.value

    # Workflow stage tracking
    workflow_stage: str

    # Agent execution history (list of agent names)
    agent_history: Annotated[List[str], operator.add]

    # ========================================================================
    # DECISION TRACKING - Custom reducer for deduplication
    # ========================================================================

    # List of agent decisions with custom reducer
    decisions: Annotated[List[Dict[str, Any]], add_decision]

    # ========================================================================
    # HUMAN-IN-THE-LOOP - Approval workflow
    # ========================================================================

    # Whether human approval is required
    requires_approval: bool

    # Approval status: None, "pending", "approved", "rejected"
    approval_status: Optional[str]

    # Approver information
    approved_by: Optional[str]
    approval_notes: Optional[str]

    # ========================================================================
    # MEMORY REFERENCES - Short-term memory via state
    # ========================================================================

    # IDs of relevant context from vector DB
    relevant_context_ids: Annotated[Set[str], union_sets]

    # Recently retrieved documents/chunks
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]

    # Session data from Redis (merged)
    session_data: Annotated[Dict[str, Any], merge_context]

    # Conversation summary (for long conversations)
    conversation_summary: Optional[str]

    # ========================================================================
    # METRICS & MONITORING - Custom aggregation
    # ========================================================================

    # Performance metrics with intelligent aggregation
    metrics: Annotated[Dict[str, float], merge_metrics]

    # ========================================================================
    # CONFIGURATION & FLAGS
    # ========================================================================

    # Retry count for error handling
    retry_count: int

    # Maximum retries allowed
    max_retries: int

    # Confidence threshold for auto-approval
    confidence_threshold: float

    # Enable/disable features
    enable_research: bool
    enable_quality_check: bool

    # ========================================================================
    # TIMESTAMPS - Simple replacement
    # ========================================================================

    started_at: str  # ISO format
    updated_at: str  # ISO format

    # ========================================================================
    # PROPOSED SOLUTION - Draft response
    # ========================================================================

    proposed_solution: Optional[str]
    solution_confidence: Optional[float]

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    last_error: Optional[str]
    error_count: int


# ============================================================================
# STATE BUILDER - For creating initial state
# ============================================================================

class StateBuilder:
    """Builder for creating initial agent state.

    Demonstrates:
    - Builder pattern
    - Fluent interface
    - Default value management
    """

    def __init__(self):
        """Initialize builder with defaults."""
        self._state: Dict[str, Any] = {
            # Messages
            "messages": [],

            # Identifiers
            "request_id": "",
            "customer_id": "",
            "thread_id": "",

            # Request metadata
            "priority": Priority.MEDIUM.value,
            "category": None,
            "status": RequestStatus.PENDING.value,

            # Agent coordination
            "current_agent": AgentType.SUPERVISOR.value,
            "workflow_stage": "initialized",
            "agent_history": [],

            # Decisions
            "decisions": [],

            # HITL
            "requires_approval": False,
            "approval_status": None,
            "approved_by": None,
            "approval_notes": None,

            # Memory
            "relevant_context_ids": set(),
            "retrieved_documents": [],
            "session_data": {},
            "conversation_summary": None,

            # Metrics
            "metrics": {},

            # Configuration
            "retry_count": 0,
            "max_retries": 3,
            "confidence_threshold": 0.8,
            "enable_research": True,
            "enable_quality_check": True,

            # Timestamps
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),

            # Solution
            "proposed_solution": None,
            "solution_confidence": None,

            # Errors
            "last_error": None,
            "error_count": 0
        }

    def with_request_id(self, request_id: str) -> StateBuilder:
        """Set request ID."""
        self._state["request_id"] = request_id
        return self

    def with_customer_id(self, customer_id: str) -> StateBuilder:
        """Set customer ID."""
        self._state["customer_id"] = customer_id
        return self

    def with_thread_id(self, thread_id: str) -> StateBuilder:
        """Set thread ID."""
        self._state["thread_id"] = thread_id
        return self

    def with_initial_message(self, content: str) -> StateBuilder:
        """Set initial message."""
        self._state["messages"] = [HumanMessage(content=content)]
        return self

    def with_priority(self, priority: Priority) -> StateBuilder:
        """Set priority."""
        self._state["priority"] = priority.value
        return self

    def with_category(self, category: RequestCategory) -> StateBuilder:
        """Set category."""
        self._state["category"] = category.value
        return self

    def with_session_data(self, data: Dict[str, Any]) -> StateBuilder:
        """Set session data."""
        self._state["session_data"] = data
        return self

    def build(self) -> AgentState:
        """Build and return state."""
        return self._state.copy()


# ============================================================================
# STATE UTILITIES - Helper functions for state manipulation
# ============================================================================

class StateUtils:
    """Utility functions for working with state.

    Demonstrates:
    - Utility pattern
    - Static methods
    - State query methods
    """

    @staticmethod
    def get_last_user_message(state: AgentState) -> Optional[str]:
        """Get last user message content.

        Args:
            state: Agent state

        Returns:
            Last user message content or None
        """
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message.content
        return None

    @staticmethod
    def get_conversation_length(state: AgentState) -> int:
        """Get number of messages in conversation.

        Args:
            state: Agent state

        Returns:
            Number of messages
        """
        return len(state["messages"])

    @staticmethod
    def add_system_message(state: AgentState, content: str) -> AgentState:
        """Add system message to state.

        Args:
            state: Agent state
            content: Message content

        Returns:
            Updated state
        """
        state["messages"].append(SystemMessage(content=content))
        state["updated_at"] = datetime.now().isoformat()
        return state

    @staticmethod
    def record_decision(
        state: AgentState,
        agent_type: AgentType,
        decision_type: DecisionType,
        confidence: float,
        reasoning: str
    ) -> AgentState:
        """Record agent decision in state.

        Args:
            state: Agent state
            agent_type: Agent making decision
            decision_type: Type of decision
            confidence: Confidence score
            reasoning: Decision reasoning

        Returns:
            Updated state
        """
        decision = {
            "decision_id": f"{agent_type.value}_{datetime.now().timestamp()}",
            "agent": agent_type.value,
            "type": decision_type.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

        # Use reducer (will be applied by LangGraph)
        state["decisions"] = state["decisions"] + [decision]
        state["updated_at"] = datetime.now().isoformat()

        return state

    @staticmethod
    def update_metrics(
        state: AgentState,
        metrics: Dict[str, float]
    ) -> AgentState:
        """Update state metrics.

        Args:
            state: Agent state
            metrics: New metrics to add/update

        Returns:
            Updated state
        """
        # Use custom reducer
        state["metrics"] = merge_metrics(state["metrics"], metrics)
        state["updated_at"] = datetime.now().isoformat()
        return state

    @staticmethod
    def should_retry(state: AgentState) -> bool:
        """Check if operation should retry.

        Args:
            state: Agent state

        Returns:
            True if should retry
        """
        return state["retry_count"] < state["max_retries"]

    @staticmethod
    def increment_retry(state: AgentState) -> AgentState:
        """Increment retry count.

        Args:
            state: Agent state

        Returns:
            Updated state
        """
        state["retry_count"] += 1
        state["updated_at"] = datetime.now().isoformat()
        return state

    @staticmethod
    def requires_human_approval(state: AgentState) -> bool:
        """Check if state requires human approval.

        Args:
            state: Agent state

        Returns:
            True if approval required and not yet given
        """
        return (
            state["requires_approval"] and
            state["approval_status"] in [None, "pending"]
        )

    @staticmethod
    def is_high_confidence(state: AgentState) -> bool:
        """Check if solution confidence is high enough.

        Args:
            state: Agent state

        Returns:
            True if confidence exceeds threshold
        """
        confidence = state.get("solution_confidence")
        threshold = state.get("confidence_threshold", 0.8)

        return confidence is not None and confidence >= threshold

    @staticmethod
    def get_execution_time(state: AgentState) -> float:
        """Calculate execution time in seconds.

        Args:
            state: Agent state

        Returns:
            Execution time in seconds
        """
        started = datetime.fromisoformat(state["started_at"])
        updated = datetime.fromisoformat(state["updated_at"])
        return (updated - started).total_seconds()


# ============================================================================
# STATE SNAPSHOT - For checkpointing
# ============================================================================

@dataclass
class StateSnapshot:
    """Snapshot of state for checkpointing.

    Demonstrates:
    - Dataclass usage
    - Serialization pattern
    - Checkpoint management
    """

    request_id: str
    thread_id: str
    checkpoint_id: str
    state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation
        """
        # Convert messages to serializable format
        serializable_state = self.state.copy()

        if "messages" in serializable_state:
            serializable_state["messages"] = [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, "additional_kwargs", {})
                }
                for msg in serializable_state["messages"]
            ]

        # Convert sets to lists
        if "relevant_context_ids" in serializable_state:
            serializable_state["relevant_context_ids"] = list(
                serializable_state["relevant_context_ids"]
            )

        return {
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "state": serializable_state,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSnapshot:
        """Create snapshot from dictionary.

        Args:
            data: Dictionary data

        Returns:
            StateSnapshot instance
        """
        state = data["state"].copy()

        # Restore messages
        if "messages" in state:
            message_classes = {
                "HumanMessage": HumanMessage,
                "AIMessage": AIMessage,
                "SystemMessage": SystemMessage
            }

            state["messages"] = [
                message_classes[msg["type"]](
                    content=msg["content"],
                    additional_kwargs=msg.get("additional_kwargs", {})
                )
                for msg in state["messages"]
            ]

        # Restore sets
        if "relevant_context_ids" in state:
            state["relevant_context_ids"] = set(state["relevant_context_ids"])

        return cls(
            request_id=data["request_id"],
            thread_id=data["thread_id"],
            checkpoint_id=data["checkpoint_id"],
            state=state,
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Build initial state
    initial_state = (
        StateBuilder()
        .with_request_id("req_12345")
        .with_customer_id("cust_789")
        .with_thread_id("thread_abc")
        .with_initial_message("I need help with my order")
        .with_priority(Priority.HIGH)
        .with_category(RequestCategory.TECHNICAL)
        .with_session_data({"last_seen": "2025-01-01"})
        .build()
    )

    print("Initial state created")
    print(f"Request ID: {initial_state['request_id']}")
    print(f"Messages: {len(initial_state['messages'])}")

    # Example: Record decision
    state_with_decision = StateUtils.record_decision(
        initial_state,
        AgentType.TRIAGE,
        DecisionType.ROUTE,
        0.95,
        "High priority technical issue"
    )

    print(f"Decisions: {len(state_with_decision['decisions'])}")

    # Example: Update metrics
    state_with_metrics = StateUtils.update_metrics(
        state_with_decision,
        {
            "triage_duration_avg": 1.5,
            "triage_duration_count": 1,
            "confidence_max": 0.95
        }
    )

    print(f"Metrics: {state_with_metrics['metrics']}")
