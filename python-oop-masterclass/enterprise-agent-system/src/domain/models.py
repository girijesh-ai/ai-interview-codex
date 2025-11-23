"""
Domain Models for Enterprise Agent System

Demonstrates all OOP principles from the masterclass:
- Value Objects (immutable)
- Entities (with identity)
- Aggregate Roots
- SOLID principles
- Design patterns (Factory, Builder, Repository, etc.)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Protocol
from uuid import UUID, uuid4


# ============================================================================
# ENUMS - Type Safety
# ============================================================================

class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: Priority) -> bool:
        """Compare priorities."""
        return self.value < other.value


class RequestCategory(Enum):
    """Request categories for routing."""
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    PRODUCT = "product"
    REFUND = "refund"
    GENERAL = "general"


class RequestStatus(Enum):
    """Request lifecycle status."""
    PENDING = "pending"
    TRIAGED = "triaged"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    QUALITY_CHECK = "quality_check"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


class AgentType(Enum):
    """Types of agents in the system."""
    SUPERVISOR = "supervisor"
    TRIAGE = "triage"
    RESEARCH = "research"
    SOLUTION = "solution"
    ESCALATION = "escalation"
    QUALITY = "quality"


class DecisionType(Enum):
    """Types of agent decisions."""
    ROUTE = "route"
    ESCALATE = "escalate"
    APPROVE = "approve"
    REJECT = "reject"
    RETRY = "retry"
    COMPLETE = "complete"


# ============================================================================
# VALUE OBJECTS - Immutable, no identity
# ============================================================================

@dataclass(frozen=True)
class Money:
    """Value object for monetary amounts.

    Following DDD principles: immutable, self-validating.
    """
    amount: Decimal
    currency: str = "USD"

    def __post_init__(self):
        """Validate money value."""
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter code")

    def add(self, other: Money) -> Money:
        """Add money amounts.

        Args:
            other: Another Money instance

        Returns:
            New Money instance with sum

        Raises:
            ValueError: If currencies don't match
        """
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def multiply(self, factor: int) -> Money:
        """Multiply money by factor."""
        return Money(self.amount * factor, self.currency)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.currency} {self.amount:.2f}"


@dataclass(frozen=True)
class ContactInfo:
    """Value object for contact information."""
    email: str
    phone: Optional[str] = None
    preferred_channel: str = "email"

    def __post_init__(self):
        """Validate contact info."""
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email address")

    def is_valid_for_notification(self, channel: str) -> bool:
        """Check if contact info supports channel."""
        if channel == "email":
            return bool(self.email)
        elif channel == "phone":
            return bool(self.phone)
        return False


@dataclass(frozen=True)
class TimeRange:
    """Value object for time range."""
    start: datetime
    end: datetime

    def __post_init__(self):
        """Validate time range."""
        if self.end < self.start:
            raise ValueError("End time must be after start time")

    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        return (self.end - self.start).total_seconds()

    def overlaps(self, other: TimeRange) -> bool:
        """Check if time ranges overlap."""
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Confidence:
    """Value object for confidence scores."""
    score: float
    explanation: str = ""

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    def is_high(self, threshold: float = 0.8) -> bool:
        """Check if confidence is high."""
        return self.score >= threshold

    def is_low(self, threshold: float = 0.5) -> bool:
        """Check if confidence is low."""
        return self.score < threshold


# ============================================================================
# ENTITIES - Have identity, mutable
# ============================================================================

@dataclass
class Customer:
    """Customer entity with identity.

    Demonstrates:
    - Entity pattern (has ID)
    - Encapsulation (private attributes)
    - Properties for controlled access
    """
    id: UUID
    name: str
    contact: ContactInfo
    tier: str = "standard"
    created_at: datetime = field(default_factory=datetime.now)
    _lifetime_value: Decimal = field(default=Decimal("0.00"), init=False)

    def __eq__(self, other: object) -> bool:
        """Entities are equal if IDs match."""
        if not isinstance(other, Customer):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)

    @property
    def lifetime_value(self) -> Decimal:
        """Get lifetime value."""
        return self._lifetime_value

    def add_transaction(self, amount: Money) -> None:
        """Add transaction to lifetime value."""
        if amount.currency == "USD":
            self._lifetime_value += amount.amount

    def is_premium(self) -> bool:
        """Check if customer is premium."""
        return self.tier in ["premium", "enterprise"]

    def get_priority_boost(self) -> int:
        """Get priority boost based on tier."""
        boosts = {
            "standard": 0,
            "premium": 1,
            "enterprise": 2
        }
        return boosts.get(self.tier, 0)


@dataclass
class AgentDecision:
    """Represents a decision made by an agent.

    Demonstrates:
    - Entity with identity
    - Immutable after creation (no setters)
    - Rich domain model
    """
    id: UUID
    agent_type: AgentType
    decision_type: DecisionType
    confidence: Confidence
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate decision."""
        if not self.reasoning:
            raise ValueError("Decision must have reasoning")

    def is_confident(self) -> bool:
        """Check if decision is confident."""
        return self.confidence.is_high()

    def requires_human_review(self) -> bool:
        """Check if decision requires human review."""
        return (
            self.confidence.is_low() or
            self.decision_type in [DecisionType.ESCALATE, DecisionType.REJECT]
        )


@dataclass
class Message:
    """Message entity for conversation tracking."""
    id: UUID
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_user_message(self) -> bool:
        """Check if message is from user."""
        return self.role == "user"

    def contains_keywords(self, keywords: List[str]) -> bool:
        """Check if message contains any keywords."""
        content_lower = self.content.lower()
        return any(kw.lower() in content_lower for kw in keywords)


# ============================================================================
# AGGREGATE ROOT - CustomerRequest
# ============================================================================

class CustomerRequest:
    """Aggregate root for customer support request.

    Demonstrates:
    - Aggregate root pattern
    - Encapsulation of business logic
    - Transaction boundary
    - Invariant enforcement
    - Domain events (not shown but mentioned)
    """

    def __init__(
        self,
        request_id: UUID,
        customer: Customer,
        initial_message: str,
        priority: Optional[Priority] = None
    ):
        """Initialize customer request.

        Args:
            request_id: Unique request identifier
            customer: Customer entity
            initial_message: Initial message text
            priority: Optional priority override
        """
        self.id = request_id
        self.customer = customer
        self._messages: List[Message] = []
        self._decisions: List[AgentDecision] = []
        self._status = RequestStatus.PENDING
        self._category: Optional[RequestCategory] = None
        self._priority = priority or Priority.MEDIUM
        self._assigned_agent: Optional[AgentType] = None
        self._requires_approval = False
        self._approved_by: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self._resolution: Optional[str] = None

        # Add initial message
        self.add_message("user", initial_message)

    # ========================================================================
    # Public Interface - Following interface segregation principle
    # ========================================================================

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Raises:
            ValueError: If request is completed
        """
        if self._status == RequestStatus.COMPLETED:
            raise ValueError("Cannot add message to completed request")

        message = Message(
            id=uuid4(),
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self._messages.append(message)
        self.updated_at = datetime.now()

    def add_decision(
        self,
        agent_type: AgentType,
        decision_type: DecisionType,
        confidence: Confidence,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """Record agent decision.

        Args:
            agent_type: Type of agent making decision
            decision_type: Type of decision
            confidence: Confidence in decision
            reasoning: Explanation of decision
            metadata: Additional metadata

        Returns:
            Created AgentDecision
        """
        decision = AgentDecision(
            id=uuid4(),
            agent_type=agent_type,
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self._decisions.append(decision)
        self.updated_at = datetime.now()

        # Update state based on decision
        if decision.requires_human_review():
            self._requires_approval = True

        return decision

    def triage(
        self,
        category: RequestCategory,
        priority: Priority,
        agent: AgentType
    ) -> None:
        """Triage the request.

        Args:
            category: Request category
            priority: Priority level
            agent: Agent performing triage
        """
        if self._status != RequestStatus.PENDING:
            raise ValueError("Can only triage pending requests")

        self._category = category
        self._priority = priority
        self._assigned_agent = agent
        self._status = RequestStatus.TRIAGED
        self.updated_at = datetime.now()

    def escalate(self, reason: str) -> None:
        """Escalate request to human.

        Args:
            reason: Reason for escalation
        """
        self._status = RequestStatus.ESCALATED
        self._requires_approval = True
        self.updated_at = datetime.now()

        # Record escalation decision
        self.add_decision(
            agent_type=AgentType.ESCALATION,
            decision_type=DecisionType.ESCALATE,
            confidence=Confidence(1.0),
            reasoning=reason
        )

    def approve(self, approver: str) -> None:
        """Approve request.

        Args:
            approver: Identifier of approver

        Raises:
            ValueError: If approval not required
        """
        if not self._requires_approval:
            raise ValueError("Request does not require approval")

        self._approved_by = approver
        self._status = RequestStatus.APPROVED
        self._requires_approval = False
        self.updated_at = datetime.now()

    def complete(self, resolution: str) -> None:
        """Complete the request.

        Args:
            resolution: Final resolution text
        """
        if self._requires_approval:
            raise ValueError("Cannot complete request requiring approval")

        self._resolution = resolution
        self._status = RequestStatus.COMPLETED
        self.updated_at = datetime.now()

    # ========================================================================
    # Query Methods - Read-only access
    # ========================================================================

    @property
    def status(self) -> RequestStatus:
        """Get current status."""
        return self._status

    @property
    def category(self) -> Optional[RequestCategory]:
        """Get request category."""
        return self._category

    @property
    def priority(self) -> Priority:
        """Get effective priority (with customer boost)."""
        base_priority = self._priority.value
        boost = self.customer.get_priority_boost()
        adjusted = min(base_priority + boost, Priority.CRITICAL.value)
        return Priority(adjusted)

    @property
    def messages(self) -> List[Message]:
        """Get all messages (read-only copy)."""
        return self._messages.copy()

    @property
    def decisions(self) -> List[AgentDecision]:
        """Get all decisions (read-only copy)."""
        return self._decisions.copy()

    @property
    def requires_approval(self) -> bool:
        """Check if approval is required."""
        return self._requires_approval

    @property
    def resolution(self) -> Optional[str]:
        """Get resolution text."""
        return self._resolution

    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        lines = []
        for msg in self._messages:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{timestamp}] {msg.role}: {msg.content}")
        return "\n".join(lines)

    def get_latest_message(self) -> Optional[Message]:
        """Get latest message."""
        return self._messages[-1] if self._messages else None

    def is_high_priority(self) -> bool:
        """Check if request is high priority."""
        return self.priority >= Priority.HIGH

    def time_since_creation(self) -> float:
        """Get time since creation in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def has_been_triaged(self) -> bool:
        """Check if request has been triaged."""
        return self._category is not None

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __eq__(self, other: object) -> bool:
        """Aggregate roots compared by ID."""
        if not isinstance(other, CustomerRequest):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"CustomerRequest(id={self.id}, "
            f"customer={self.customer.id}, "
            f"status={self._status.value}, "
            f"priority={self.priority.value})"
        )

    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"Request #{self.id} - {self._status.value} "
            f"(Priority: {self.priority.value})"
        )


# ============================================================================
# DOMAIN SERVICES - Operations that don't belong to entities
# ============================================================================

class PriorityCalculator:
    """Domain service for calculating request priority.

    Demonstrates:
    - Domain service pattern
    - Single Responsibility Principle
    - Stateless service
    """

    @staticmethod
    def calculate_priority(
        customer: Customer,
        category: RequestCategory,
        keywords: List[str],
        message: str
    ) -> Priority:
        """Calculate priority based on multiple factors.

        Args:
            customer: Customer entity
            category: Request category
            keywords: Urgent keywords to check
            message: Request message

        Returns:
            Calculated priority
        """
        # Start with base priority
        priority_score = 2  # MEDIUM

        # Customer tier boost
        priority_score += customer.get_priority_boost()

        # Category boost
        category_boosts = {
            RequestCategory.BILLING: 1,
            RequestCategory.REFUND: 1,
            RequestCategory.TECHNICAL: 0
        }
        priority_score += category_boosts.get(category, 0)

        # Keyword urgency
        message_lower = message.lower()
        urgent_keywords = ["urgent", "emergency", "critical", "asap"]
        if any(kw in message_lower for kw in urgent_keywords):
            priority_score += 2

        # Cap at CRITICAL
        priority_score = min(priority_score, Priority.CRITICAL.value)

        return Priority(max(priority_score, Priority.LOW.value))


class ConfidenceCalculator:
    """Domain service for calculating confidence scores."""

    @staticmethod
    def calculate_solution_confidence(
        category: RequestCategory,
        similar_cases_count: int,
        knowledge_base_match: float
    ) -> Confidence:
        """Calculate confidence in proposed solution.

        Args:
            category: Request category
            similar_cases_count: Number of similar historical cases
            knowledge_base_match: KB similarity score (0-1)

        Returns:
            Confidence score
        """
        # Base confidence from KB match
        score = knowledge_base_match * 0.6

        # Boost from similar cases
        if similar_cases_count > 10:
            score += 0.3
        elif similar_cases_count > 5:
            score += 0.2
        elif similar_cases_count > 0:
            score += 0.1

        # Category-specific adjustments
        if category in [RequestCategory.ACCOUNT, RequestCategory.GENERAL]:
            score += 0.1  # Usually simpler

        score = min(score, 1.0)

        explanation = (
            f"Based on {similar_cases_count} similar cases "
            f"and {knowledge_base_match:.2%} KB match"
        )

        return Confidence(score, explanation)


# ============================================================================
# REPOSITORY INTERFACES - Following Dependency Inversion Principle
# ============================================================================

class CustomerRequestRepository(ABC):
    """Abstract repository for customer requests.

    Demonstrates:
    - Repository pattern
    - Dependency Inversion Principle
    - Interface Segregation Principle
    """

    @abstractmethod
    async def save(self, request: CustomerRequest) -> None:
        """Save customer request.

        Args:
            request: Request to save
        """
        pass

    @abstractmethod
    async def find_by_id(self, request_id: UUID) -> Optional[CustomerRequest]:
        """Find request by ID.

        Args:
            request_id: Request identifier

        Returns:
            Request if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_customer(
        self,
        customer_id: UUID,
        limit: int = 10
    ) -> List[CustomerRequest]:
        """Find requests by customer.

        Args:
            customer_id: Customer identifier
            limit: Maximum number of requests

        Returns:
            List of requests
        """
        pass

    @abstractmethod
    async def find_pending(self, limit: int = 100) -> List[CustomerRequest]:
        """Find pending requests.

        Args:
            limit: Maximum number of requests

        Returns:
            List of pending requests
        """
        pass


class CustomerRepository(ABC):
    """Abstract repository for customers."""

    @abstractmethod
    async def find_by_id(self, customer_id: UUID) -> Optional[Customer]:
        """Find customer by ID."""
        pass

    @abstractmethod
    async def save(self, customer: Customer) -> None:
        """Save customer."""
        pass


# ============================================================================
# FACTORY PATTERN - Object creation
# ============================================================================

class CustomerRequestFactory:
    """Factory for creating customer requests.

    Demonstrates:
    - Factory pattern
    - Builder pattern (fluent interface)
    - Encapsulation of creation logic
    """

    def __init__(
        self,
        priority_calculator: PriorityCalculator,
        customer_repository: CustomerRepository
    ):
        """Initialize factory.

        Args:
            priority_calculator: Service for calculating priority
            customer_repository: Repository for loading customers
        """
        self.priority_calculator = priority_calculator
        self.customer_repository = customer_repository

    async def create_request(
        self,
        customer_id: UUID,
        message: str,
        category: Optional[RequestCategory] = None
    ) -> CustomerRequest:
        """Create new customer request with intelligent defaults.

        Args:
            customer_id: Customer identifier
            message: Initial message
            category: Optional category override

        Returns:
            Created CustomerRequest

        Raises:
            ValueError: If customer not found
        """
        # Load customer
        customer = await self.customer_repository.find_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        # Auto-detect category if not provided
        if category is None:
            category = self._detect_category(message)

        # Calculate priority
        priority = self.priority_calculator.calculate_priority(
            customer=customer,
            category=category,
            keywords=[],
            message=message
        )

        # Create request
        request = CustomerRequest(
            request_id=uuid4(),
            customer=customer,
            initial_message=message,
            priority=priority
        )

        return request

    def _detect_category(self, message: str) -> RequestCategory:
        """Auto-detect category from message.

        Args:
            message: Message text

        Returns:
            Detected category
        """
        message_lower = message.lower()

        # Simple keyword matching (in production, use ML)
        if any(kw in message_lower for kw in ["password", "login", "account"]):
            return RequestCategory.ACCOUNT
        elif any(kw in message_lower for kw in ["bill", "charge", "payment"]):
            return RequestCategory.BILLING
        elif any(kw in message_lower for kw in ["refund", "return", "money back"]):
            return RequestCategory.REFUND
        elif any(kw in message_lower for kw in ["bug", "error", "broken"]):
            return RequestCategory.TECHNICAL
        elif any(kw in message_lower for kw in ["product", "feature"]):
            return RequestCategory.PRODUCT
        else:
            return RequestCategory.GENERAL


# ============================================================================
# SPECIFICATION PATTERN - Business rules as objects
# ============================================================================

class Specification(ABC):
    """Abstract specification for business rules.

    Demonstrates:
    - Specification pattern
    - Composite pattern (and, or, not)
    """

    @abstractmethod
    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if request satisfies specification."""
        pass

    def and_(self, other: Specification) -> Specification:
        """Combine specifications with AND."""
        return AndSpecification(self, other)

    def or_(self, other: Specification) -> Specification:
        """Combine specifications with OR."""
        return OrSpecification(self, other)

    def not_(self) -> Specification:
        """Negate specification."""
        return NotSpecification(self)


class HighPrioritySpecification(Specification):
    """Specification for high priority requests."""

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if request is high priority."""
        return request.is_high_priority()


class RequiresApprovalSpecification(Specification):
    """Specification for requests requiring approval."""

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if request requires approval."""
        return request.requires_approval


class PremiumCustomerSpecification(Specification):
    """Specification for premium customers."""

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if customer is premium."""
        return request.customer.is_premium()


class AndSpecification(Specification):
    """AND combination of specifications."""

    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if both specifications are satisfied."""
        return (
            self.left.is_satisfied_by(request) and
            self.right.is_satisfied_by(request)
        )


class OrSpecification(Specification):
    """OR combination of specifications."""

    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if either specification is satisfied."""
        return (
            self.left.is_satisfied_by(request) or
            self.right.is_satisfied_by(request)
        )


class NotSpecification(Specification):
    """NOT negation of specification."""

    def __init__(self, spec: Specification):
        self.spec = spec

    def is_satisfied_by(self, request: CustomerRequest) -> bool:
        """Check if specification is not satisfied."""
        return not self.spec.is_satisfied_by(request)


# Usage example of specifications:
# premium_urgent = PremiumCustomerSpecification().and_(HighPrioritySpecification())
# if premium_urgent.is_satisfied_by(request):
#     # Handle premium urgent request
