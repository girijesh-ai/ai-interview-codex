"""
Event Schemas and Types for Kafka Event Streaming

Demonstrates:
- Value object pattern for events
- Event sourcing architecture
- Type safety with dataclasses
- Domain events pattern
- Serialization/deserialization
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from uuid import uuid4
import json


# ============================================================================
# EVENT TYPES
# ============================================================================

class EventType(str, Enum):
    """Event types in the system."""
    # Request lifecycle
    REQUEST_RECEIVED = "request.received"
    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Agent actions
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_DECISION = "agent.decision"

    # Triage events
    REQUEST_TRIAGED = "request.triaged"
    PRIORITY_CHANGED = "priority.changed"

    # Research events
    RESEARCH_STARTED = "research.started"
    RESEARCH_COMPLETED = "research.completed"
    DOCUMENTS_RETRIEVED = "documents.retrieved"

    # Solution events
    SOLUTION_GENERATED = "solution.generated"
    SOLUTION_REVISED = "solution.revised"

    # Quality events
    QUALITY_CHECK_PASSED = "quality.check.passed"
    QUALITY_CHECK_FAILED = "quality.check.failed"

    # Escalation events
    REQUEST_ESCALATED = "request.escalated"
    ESCALATION_RESOLVED = "escalation.resolved"

    # Human-in-the-loop events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    HUMAN_DECISION_NEEDED = "human.decision.needed"
    HUMAN_DECISION_RECEIVED = "human.decision.received"

    # Memory events
    CONTEXT_RETRIEVED = "context.retrieved"
    CONVERSATION_STORED = "conversation.stored"
    CACHE_UPDATED = "cache.updated"

    # System events
    SYSTEM_ERROR = "system.error"
    PERFORMANCE_METRIC = "performance.metric"
    HEALTH_CHECK = "health.check"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# BASE EVENT
# ============================================================================

@dataclass
class BaseEvent:
    """Base class for all events.

    Demonstrates:
    - Value object pattern
    - Immutability via frozen
    - Consistent event structure
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = EventType.SYSTEM_ERROR
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: EventPriority = EventPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        data["event_type"] = self.event_type.value
        data["priority"] = self.priority.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEvent":
        """Create from dictionary."""
        # Convert string enums back
        if "event_type" in data:
            data["event_type"] = EventType(data["event_type"])
        if "priority" in data:
            data["priority"] = EventPriority(data["priority"])
        return cls(**data)


# ============================================================================
# REQUEST EVENTS
# ============================================================================

@dataclass
class RequestReceivedEvent(BaseEvent):
    """Event when request is received."""
    request_id: str = ""
    customer_id: str = ""
    channel: str = ""  # email, chat, phone, etc.
    initial_message: str = ""

    def __post_init__(self):
        self.event_type = EventType.REQUEST_RECEIVED


@dataclass
class RequestStartedEvent(BaseEvent):
    """Event when request processing starts."""
    request_id: str = ""
    customer_id: str = ""
    assigned_to: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.REQUEST_STARTED


@dataclass
class RequestCompletedEvent(BaseEvent):
    """Event when request is completed."""
    request_id: str = ""
    customer_id: str = ""
    resolution: str = ""
    duration_seconds: float = 0.0
    satisfaction_score: Optional[float] = None

    def __post_init__(self):
        self.event_type = EventType.REQUEST_COMPLETED
        self.priority = EventPriority.HIGH


@dataclass
class RequestFailedEvent(BaseEvent):
    """Event when request processing fails."""
    request_id: str = ""
    customer_id: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.REQUEST_FAILED
        self.priority = EventPriority.CRITICAL


# ============================================================================
# AGENT EVENTS
# ============================================================================

@dataclass
class AgentStartedEvent(BaseEvent):
    """Event when agent starts processing."""
    request_id: str = ""
    agent_type: str = ""
    agent_id: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.AGENT_STARTED


@dataclass
class AgentCompletedEvent(BaseEvent):
    """Event when agent completes processing."""
    request_id: str = ""
    agent_type: str = ""
    agent_id: str = ""
    duration_ms: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.AGENT_COMPLETED


@dataclass
class AgentDecisionEvent(BaseEvent):
    """Event when agent makes a decision."""
    request_id: str = ""
    agent_type: str = ""
    decision_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type = EventType.AGENT_DECISION
        # High priority for low confidence decisions
        if self.confidence < 0.5:
            self.priority = EventPriority.HIGH


# ============================================================================
# TRIAGE EVENTS
# ============================================================================

@dataclass
class RequestTriagedEvent(BaseEvent):
    """Event when request is triaged."""
    request_id: str = ""
    category: str = ""
    priority: int = 2
    urgency: str = ""
    complexity: str = ""
    estimated_resolution_time: Optional[int] = None  # Minutes

    def __post_init__(self):
        self.event_type = EventType.REQUEST_TRIAGED


@dataclass
class PriorityChangedEvent(BaseEvent):
    """Event when priority changes."""
    request_id: str = ""
    old_priority: int = 2
    new_priority: int = 2
    reason: str = ""
    changed_by: str = ""

    def __post_init__(self):
        self.event_type = EventType.PRIORITY_CHANGED


# ============================================================================
# RESEARCH EVENTS
# ============================================================================

@dataclass
class ResearchStartedEvent(BaseEvent):
    """Event when research starts."""
    request_id: str = ""
    query: str = ""
    search_strategy: str = ""

    def __post_init__(self):
        self.event_type = EventType.RESEARCH_STARTED


@dataclass
class DocumentsRetrievedEvent(BaseEvent):
    """Event when documents are retrieved."""
    request_id: str = ""
    query: str = ""
    document_count: int = 0
    sources: List[str] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.event_type = EventType.DOCUMENTS_RETRIEVED


# ============================================================================
# SOLUTION EVENTS
# ============================================================================

@dataclass
class SolutionGeneratedEvent(BaseEvent):
    """Event when solution is generated."""
    request_id: str = ""
    solution_type: str = ""
    confidence: float = 0.0
    template_used: Optional[str] = None
    tokens_used: int = 0

    def __post_init__(self):
        self.event_type = EventType.SOLUTION_GENERATED


@dataclass
class SolutionRevisedEvent(BaseEvent):
    """Event when solution is revised."""
    request_id: str = ""
    revision_number: int = 1
    reason: str = ""
    changes_made: str = ""

    def __post_init__(self):
        self.event_type = EventType.SOLUTION_REVISED


# ============================================================================
# QUALITY EVENTS
# ============================================================================

@dataclass
class QualityCheckPassedEvent(BaseEvent):
    """Event when quality check passes."""
    request_id: str = ""
    checks_passed: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    reviewer: str = ""

    def __post_init__(self):
        self.event_type = EventType.QUALITY_CHECK_PASSED


@dataclass
class QualityCheckFailedEvent(BaseEvent):
    """Event when quality check fails."""
    request_id: str = ""
    checks_failed: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    reviewer: str = ""

    def __post_init__(self):
        self.event_type = EventType.QUALITY_CHECK_FAILED
        self.priority = EventPriority.HIGH


# ============================================================================
# ESCALATION EVENTS
# ============================================================================

@dataclass
class RequestEscalatedEvent(BaseEvent):
    """Event when request is escalated."""
    request_id: str = ""
    escalation_reason: str = ""
    escalation_level: int = 1
    escalated_to: str = ""
    escalated_by: str = ""

    def __post_init__(self):
        self.event_type = EventType.REQUEST_ESCALATED
        self.priority = EventPriority.HIGH


@dataclass
class EscalationResolvedEvent(BaseEvent):
    """Event when escalation is resolved."""
    request_id: str = ""
    resolution: str = ""
    resolved_by: str = ""
    resolution_time_minutes: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.ESCALATION_RESOLVED


# ============================================================================
# HUMAN-IN-THE-LOOP EVENTS
# ============================================================================

@dataclass
class ApprovalRequestedEvent(BaseEvent):
    """Event when approval is requested."""
    request_id: str = ""
    approval_type: str = ""
    requested_by: str = ""
    requires_approval_from: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.APPROVAL_REQUESTED
        self.priority = EventPriority.HIGH


@dataclass
class ApprovalGrantedEvent(BaseEvent):
    """Event when approval is granted."""
    request_id: str = ""
    approval_type: str = ""
    approved_by: str = ""
    approval_notes: str = ""

    def __post_init__(self):
        self.event_type = EventType.APPROVAL_GRANTED
        self.priority = EventPriority.HIGH


@dataclass
class ApprovalRejectedEvent(BaseEvent):
    """Event when approval is rejected."""
    request_id: str = ""
    approval_type: str = ""
    rejected_by: str = ""
    rejection_reason: str = ""

    def __post_init__(self):
        self.event_type = EventType.APPROVAL_REJECTED
        self.priority = EventPriority.HIGH


@dataclass
class HumanDecisionNeededEvent(BaseEvent):
    """Event when human decision is needed."""
    request_id: str = ""
    decision_type: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.event_type = EventType.HUMAN_DECISION_NEEDED
        self.priority = EventPriority.CRITICAL


@dataclass
class HumanDecisionReceivedEvent(BaseEvent):
    """Event when human decision is received."""
    request_id: str = ""
    decision_type: str = ""
    decision: str = ""
    decided_by: str = ""
    decision_notes: str = ""

    def __post_init__(self):
        self.event_type = EventType.HUMAN_DECISION_RECEIVED
        self.priority = EventPriority.HIGH


# ============================================================================
# MEMORY EVENTS
# ============================================================================

@dataclass
class ContextRetrievedEvent(BaseEvent):
    """Event when context is retrieved."""
    request_id: str = ""
    retrieval_strategy: str = ""
    sources: List[str] = field(default_factory=list)
    item_count: int = 0
    duration_ms: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.CONTEXT_RETRIEVED


@dataclass
class ConversationStoredEvent(BaseEvent):
    """Event when conversation is stored."""
    request_id: str = ""
    message_count: int = 0
    storage_tier: str = ""  # cache, vector, etc.
    ttl_seconds: Optional[int] = None

    def __post_init__(self):
        self.event_type = EventType.CONVERSATION_STORED


# ============================================================================
# SYSTEM EVENTS
# ============================================================================

@dataclass
class SystemErrorEvent(BaseEvent):
    """Event when system error occurs."""
    error_type: str = ""
    error_message: str = ""
    component: str = ""
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False

    def __post_init__(self):
        self.event_type = EventType.SYSTEM_ERROR
        self.priority = EventPriority.CRITICAL


@dataclass
class PerformanceMetricEvent(BaseEvent):
    """Event for performance metrics."""
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    component: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.PERFORMANCE_METRIC
        self.priority = EventPriority.LOW


@dataclass
class HealthCheckEvent(BaseEvent):
    """Event for health checks."""
    component: str = ""
    status: str = ""  # healthy, degraded, unhealthy
    checks: Dict[str, bool] = field(default_factory=dict)
    response_time_ms: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.HEALTH_CHECK
        self.priority = EventPriority.LOW


# ============================================================================
# EVENT FACTORY
# ============================================================================

class EventFactory:
    """Factory for creating events.

    Demonstrates:
    - Factory pattern
    - Type mapping
    """

    _event_classes = {
        EventType.REQUEST_RECEIVED: RequestReceivedEvent,
        EventType.REQUEST_STARTED: RequestStartedEvent,
        EventType.REQUEST_COMPLETED: RequestCompletedEvent,
        EventType.REQUEST_FAILED: RequestFailedEvent,
        EventType.AGENT_STARTED: AgentStartedEvent,
        EventType.AGENT_COMPLETED: AgentCompletedEvent,
        EventType.AGENT_DECISION: AgentDecisionEvent,
        EventType.REQUEST_TRIAGED: RequestTriagedEvent,
        EventType.PRIORITY_CHANGED: PriorityChangedEvent,
        EventType.RESEARCH_STARTED: ResearchStartedEvent,
        EventType.DOCUMENTS_RETRIEVED: DocumentsRetrievedEvent,
        EventType.SOLUTION_GENERATED: SolutionGeneratedEvent,
        EventType.SOLUTION_REVISED: SolutionRevisedEvent,
        EventType.QUALITY_CHECK_PASSED: QualityCheckPassedEvent,
        EventType.QUALITY_CHECK_FAILED: QualityCheckFailedEvent,
        EventType.REQUEST_ESCALATED: RequestEscalatedEvent,
        EventType.ESCALATION_RESOLVED: EscalationResolvedEvent,
        EventType.APPROVAL_REQUESTED: ApprovalRequestedEvent,
        EventType.APPROVAL_GRANTED: ApprovalGrantedEvent,
        EventType.APPROVAL_REJECTED: ApprovalRejectedEvent,
        EventType.HUMAN_DECISION_NEEDED: HumanDecisionNeededEvent,
        EventType.HUMAN_DECISION_RECEIVED: HumanDecisionReceivedEvent,
        EventType.CONTEXT_RETRIEVED: ContextRetrievedEvent,
        EventType.CONVERSATION_STORED: ConversationStoredEvent,
        EventType.SYSTEM_ERROR: SystemErrorEvent,
        EventType.PERFORMANCE_METRIC: PerformanceMetricEvent,
        EventType.HEALTH_CHECK: HealthCheckEvent,
    }

    @classmethod
    def create_from_type(cls, event_type: EventType, **kwargs) -> BaseEvent:
        """Create event from type.

        Args:
            event_type: Event type
            **kwargs: Event data

        Returns:
            Event instance
        """
        event_class = cls._event_classes.get(event_type, BaseEvent)
        return event_class(**kwargs)

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> BaseEvent:
        """Create event from dictionary.

        Args:
            data: Event data

        Returns:
            Event instance
        """
        event_type = EventType(data.get("event_type", EventType.SYSTEM_ERROR.value))
        event_class = cls._event_classes.get(event_type, BaseEvent)
        return event_class.from_dict(data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create request received event
    event = RequestReceivedEvent(
        request_id="req-123",
        customer_id="cust-456",
        channel="chat",
        initial_message="I need help with my account"
    )

    print("Event created:")
    print(f"  ID: {event.event_id}")
    print(f"  Type: {event.event_type.value}")
    print(f"  Priority: {event.priority.value}")
    print(f"  Timestamp: {event.timestamp}")
    print(f"  Request: {event.request_id}")

    # Convert to JSON
    json_data = event.to_json()
    print(f"\nJSON: {json_data}")

    # Convert back from JSON
    data = json.loads(json_data)
    restored_event = EventFactory.create_from_dict(data)
    print(f"\nRestored event type: {restored_event.event_type.value}")

    # Create agent decision event
    decision = AgentDecisionEvent(
        request_id="req-123",
        agent_type="triage",
        decision_type="classify",
        confidence=0.85,
        reasoning="Customer mentioned account issue",
        alternatives=["billing", "technical"]
    )
    print(f"\nDecision event: {decision.event_type.value}")
    print(f"  Confidence: {decision.confidence}")
    print(f"  Priority: {decision.priority.value}")
