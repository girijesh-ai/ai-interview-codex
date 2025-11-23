# Enterprise Agent System - Refactoring to A+ Quality
## Google-Level Engineering Standards Implementation Plan

**Author**: Senior Staff Engineer
**Date**: November 22, 2025
**Current Grade**: B+
**Target Grade**: A+
**Timeline**: 6-8 weeks
**Effort**: ~320 engineering hours

---

## Executive Summary

This document outlines a comprehensive refactoring plan to elevate the Enterprise Agent System from B+ to A+ quality, following Google's engineering standards. The plan addresses architectural improvements, code quality enhancements, security hardening, performance optimization, and operational excellence.

**Key Principles Applied**:
- Google's Code Review Standards
- Site Reliability Engineering (SRE) Best Practices
- Defense in Depth Security
- Observability-First Design
- API Design Guidelines
- Clean Architecture Principles

---

## Table of Contents

1. [Architecture Refactoring](#1-architecture-refactoring)
2. [Code Quality Improvements](#2-code-quality-improvements)
3. [Security Hardening](#3-security-hardening)
4. [Performance Optimization](#4-performance-optimization)
5. [Testing Strategy](#5-testing-strategy)
6. [Observability & Monitoring](#6-observability--monitoring)
7. [API Design](#7-api-design)
8. [Error Handling](#8-error-handling)
9. [Documentation](#9-documentation)
10. [Infrastructure as Code](#10-infrastructure-as-code)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Architecture Refactoring

### 1.1 Hexagonal Architecture (Ports & Adapters)

**Current Issue**: Tight coupling between layers, infrastructure leaking into domain.

**Google Standard**: Clean separation of concerns with dependency inversion.

**Refactoring**:

```
enterprise-agent-system/
├── src/
│   ├── core/                          # Core domain (no dependencies)
│   │   ├── domain/                    # Pure domain logic
│   │   │   ├── entities/
│   │   │   │   ├── customer.py
│   │   │   │   ├── request.py
│   │   │   │   └── decision.py
│   │   │   ├── value_objects/
│   │   │   │   ├── money.py
│   │   │   │   ├── confidence.py
│   │   │   │   └── contact_info.py
│   │   │   ├── aggregates/
│   │   │   │   └── request_aggregate.py
│   │   │   ├── events/                # Domain events
│   │   │   │   ├── request_created.py
│   │   │   │   └── request_completed.py
│   │   │   └── exceptions/            # Domain exceptions
│   │   │       ├── validation_error.py
│   │   │       └── business_rule_violation.py
│   │   │
│   │   ├── application/               # Application services (use cases)
│   │   │   ├── use_cases/
│   │   │   │   ├── create_request.py
│   │   │   │   ├── process_request.py
│   │   │   │   └── approve_request.py
│   │   │   ├── commands/              # CQRS Commands
│   │   │   │   └── create_request_command.py
│   │   │   ├── queries/               # CQRS Queries
│   │   │   │   └── get_request_query.py
│   │   │   └── ports/                 # Interface definitions
│   │   │       ├── repositories/
│   │   │       │   ├── request_repository.py
│   │   │       │   └── customer_repository.py
│   │   │       ├── services/
│   │   │       │   ├── memory_service.py
│   │   │       │   ├── event_service.py
│   │   │       │   └── llm_service.py
│   │   │       └── notifications/
│   │   │           └── notification_service.py
│   │   │
│   │   └── agents/                    # Agent orchestration
│   │       ├── workflows/
│   │       │   └── request_workflow.py
│   │       ├── strategies/
│   │       │   ├── triage_strategy.py
│   │       │   └── research_strategy.py
│   │       └── policies/
│   │           └── escalation_policy.py
│   │
│   ├── adapters/                      # Infrastructure implementations
│   │   ├── inbound/                   # Driving adapters
│   │   │   ├── api/
│   │   │   │   ├── rest/
│   │   │   │   │   ├── v1/
│   │   │   │   │   │   ├── handlers/
│   │   │   │   │   │   ├── middleware/
│   │   │   │   │   │   └── validators/
│   │   │   │   │   └── v2/
│   │   │   │   ├── graphql/
│   │   │   │   └── grpc/
│   │   │   ├── cli/
│   │   │   └── events/
│   │   │       └── kafka_consumer.py
│   │   │
│   │   └── outbound/                  # Driven adapters
│   │       ├── persistence/
│   │       │   ├── postgres/
│   │       │   │   ├── repositories/
│   │       │   │   ├── migrations/
│   │       │   │   └── models/
│   │       │   └── redis/
│   │       │       └── cache_impl.py
│   │       ├── messaging/
│   │       │   └── kafka/
│   │       │       ├── producer_impl.py
│   │       │       └── consumer_impl.py
│   │       ├── external/
│   │       │   ├── openai/
│   │       │   │   └── llm_service_impl.py
│   │       │   └── weaviate/
│   │       │       └── vector_store_impl.py
│   │       └── notifications/
│   │           ├── email/
│   │           └── sms/
│   │
│   ├── shared/                        # Shared kernel
│   │   ├── types/
│   │   ├── utils/
│   │   ├── validation/
│   │   └── monitoring/
│   │
│   └── config/                        # Configuration
│       ├── settings.py
│       ├── dependencies.py
│       └── logging.py
│
├── tests/
│   ├── unit/
│   │   ├── core/
│   │   └── adapters/
│   ├── integration/
│   │   ├── api/
│   │   └── persistence/
│   ├── e2e/
│   ├── performance/
│   └── contract/                      # Contract testing
│
└── tools/
    ├── scripts/
    ├── generators/
    └── analyzers/
```

**Benefits**:
- Domain logic completely independent of infrastructure
- Easy to test domain without external dependencies
- Can swap implementations without changing core
- Clear dependency flow (inward only)

---

### 1.2 CQRS (Command Query Responsibility Segregation)

**Implementation**:

```python
# src/core/application/commands/create_request_command.py
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass(frozen=True)
class CreateRequestCommand:
    """Command to create a new request.

    Immutable command object following CQRS pattern.
    """
    request_id: str
    customer_id: str
    message: str
    priority: int
    category: Optional[str] = None
    metadata: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def validate(self) -> None:
        """Validate command invariants."""
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.customer_id:
            raise ValueError("customer_id is required")
        if len(self.message) == 0:
            raise ValueError("message cannot be empty")
        if len(self.message) > 10000:
            raise ValueError("message too long (max 10000 chars)")


# src/core/application/commands/command_handler.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TCommand = TypeVar('TCommand')
TResult = TypeVar('TResult')

class CommandHandler(ABC, Generic[TCommand, TResult]):
    """Abstract command handler."""

    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        """Handle command and return result."""
        pass


# src/core/application/commands/handlers/create_request_handler.py
from typing import Optional
from ..command_handler import CommandHandler
from ..create_request_command import CreateRequestCommand
from ...ports.repositories.request_repository import RequestRepository
from ...ports.services.event_service import EventService
from ....domain.aggregates.request_aggregate import RequestAggregate

class CreateRequestCommandHandler(CommandHandler[CreateRequestCommand, RequestAggregate]):
    """Handler for CreateRequestCommand."""

    def __init__(
        self,
        request_repository: RequestRepository,
        event_service: EventService
    ):
        self._repository = request_repository
        self._events = event_service

    async def handle(self, command: CreateRequestCommand) -> RequestAggregate:
        """Create and persist a new request."""
        # Validate command
        command.validate()

        # Create aggregate
        request = RequestAggregate.create(
            request_id=command.request_id,
            customer_id=command.customer_id,
            message=command.message,
            priority=command.priority,
            category=command.category
        )

        # Persist
        await self._repository.save(request)

        # Publish domain events
        for event in request.domain_events:
            await self._events.publish(event)

        request.clear_events()

        return request


# src/core/application/queries/get_request_query.py
@dataclass(frozen=True)
class GetRequestQuery:
    """Query to get request by ID."""
    request_id: str
    include_history: bool = False
    include_decisions: bool = True


# src/core/application/queries/query_handler.py
class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Abstract query handler."""

    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        """Handle query and return result."""
        pass
```

**Benefits**:
- Clear separation between reads and writes
- Optimized query models
- Better scalability
- Audit trail built-in

---

### 1.3 Domain Events

**Implementation**:

```python
# src/core/domain/events/domain_event.py
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

@dataclass(frozen=True)
class DomainEvent(ABC):
    """Base class for all domain events."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: str = field(default="")
    version: int = field(default=1)

    def to_dict(self) -> dict:
        """Serialize event to dictionary."""
        return {
            'event_id': str(self.event_id),
            'event_type': self.__class__.__name__,
            'occurred_at': self.occurred_at.isoformat(),
            'aggregate_id': self.aggregate_id,
            'version': self.version,
            'data': self._get_event_data()
        }

    @abstractmethod
    def _get_event_data(self) -> dict:
        """Get event-specific data."""
        pass


# src/core/domain/events/request_events.py
@dataclass(frozen=True)
class RequestCreatedEvent(DomainEvent):
    """Event raised when request is created."""
    request_id: str
    customer_id: str
    priority: int
    category: Optional[str]

    def _get_event_data(self) -> dict:
        return {
            'request_id': self.request_id,
            'customer_id': self.customer_id,
            'priority': self.priority,
            'category': self.category
        }


@dataclass(frozen=True)
class RequestTriagedEvent(DomainEvent):
    """Event raised when request is triaged."""
    request_id: str
    assigned_category: str
    assigned_priority: int
    confidence: float

    def _get_event_data(self) -> dict:
        return {
            'request_id': self.request_id,
            'assigned_category': self.assigned_category,
            'assigned_priority': self.assigned_priority,
            'confidence': self.confidence
        }


# src/core/domain/aggregates/request_aggregate.py
class RequestAggregate:
    """Request aggregate root with domain events."""

    def __init__(self, request_id: str, customer_id: str):
        self._request_id = request_id
        self._customer_id = customer_id
        self._domain_events: List[DomainEvent] = []

    @classmethod
    def create(
        cls,
        request_id: str,
        customer_id: str,
        message: str,
        priority: int,
        category: Optional[str] = None
    ) -> 'RequestAggregate':
        """Factory method to create new request."""
        request = cls(request_id, customer_id)
        request._message = message
        request._priority = priority
        request._category = category
        request._status = RequestStatus.PENDING
        request._created_at = datetime.utcnow()

        # Raise domain event
        event = RequestCreatedEvent(
            aggregate_id=request_id,
            request_id=request_id,
            customer_id=customer_id,
            priority=priority,
            category=category
        )
        request._domain_events.append(event)

        return request

    def triage(self, category: str, priority: int, confidence: float) -> None:
        """Triage the request."""
        if self._status != RequestStatus.PENDING:
            raise BusinessRuleViolation("Can only triage pending requests")

        self._category = category
        self._priority = priority
        self._status = RequestStatus.TRIAGED

        # Raise domain event
        event = RequestTriagedEvent(
            aggregate_id=self._request_id,
            request_id=self._request_id,
            assigned_category=category,
            assigned_priority=priority,
            confidence=confidence
        )
        self._domain_events.append(event)

    @property
    def domain_events(self) -> List[DomainEvent]:
        """Get domain events."""
        return self._domain_events.copy()

    def clear_events(self) -> None:
        """Clear domain events after publishing."""
        self._domain_events.clear()
```

---

## 2. Code Quality Improvements

### 2.1 Immutability by Default

**Current Issue**: Mutable objects causing bugs (Customer hashable but mutable).

**Google Standard**: Prefer immutable data structures.

**Refactoring**:

```python
# src/core/domain/value_objects/money.py
from decimal import Decimal
from typing import Union
import operator

class Money:
    """Immutable money value object.

    Following Google's immutability guidelines and DDD principles.
    """

    __slots__ = ('_amount', '_currency')

    def __init__(self, amount: Union[int, float, Decimal, str], currency: str = "USD"):
        """Initialize money.

        Args:
            amount: Monetary amount (will be converted to Decimal)
            currency: 3-letter ISO currency code

        Raises:
            ValueError: If amount is negative or currency invalid
        """
        # Validate and normalize
        amount_decimal = self._to_decimal(amount)
        if amount_decimal < 0:
            raise ValueError(f"Amount cannot be negative: {amount_decimal}")

        currency_upper = currency.upper()
        if len(currency_upper) != 3:
            raise ValueError(f"Currency must be 3-letter code: {currency}")

        # Use object.__setattr__ to set on frozen object
        object.__setattr__(self, '_amount', amount_decimal)
        object.__setattr__(self, '_currency', currency_upper)

    @staticmethod
    def _to_decimal(value: Union[int, float, Decimal, str]) -> Decimal:
        """Convert value to Decimal with proper precision."""
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, str)):
            return Decimal(value)
        if isinstance(value, float):
            # Handle float precision issues
            return Decimal(str(value))
        raise TypeError(f"Cannot convert {type(value)} to Decimal")

    @property
    def amount(self) -> Decimal:
        """Get amount."""
        return self._amount

    @property
    def currency(self) -> str:
        """Get currency."""
        return self._currency

    def __setattr__(self, name, value):
        """Prevent mutation."""
        raise AttributeError(f"Money is immutable. Cannot set {name}")

    def __delattr__(self, name):
        """Prevent deletion."""
        raise AttributeError(f"Money is immutable. Cannot delete {name}")

    def __add__(self, other: 'Money') -> 'Money':
        """Add two money amounts."""
        self._check_currency_match(other)
        return Money(self._amount + other._amount, self._currency)

    def __sub__(self, other: 'Money') -> 'Money':
        """Subtract two money amounts."""
        self._check_currency_match(other)
        return Money(self._amount - other._amount, self._currency)

    def __mul__(self, factor: Union[int, float, Decimal]) -> 'Money':
        """Multiply money by factor."""
        return Money(self._amount * self._to_decimal(factor), self._currency)

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> 'Money':
        """Divide money by divisor."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return Money(self._amount / self._to_decimal(divisor), self._currency)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Money):
            return NotImplemented
        return (self._amount == other._amount and
                self._currency == other._currency)

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self._amount, self._currency))

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Money(amount={self._amount}, currency='{self._currency}')"

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"{self._currency} {self._amount:,.2f}"

    def _check_currency_match(self, other: 'Money') -> None:
        """Ensure currencies match."""
        if self._currency != other._currency:
            raise ValueError(
                f"Currency mismatch: cannot operate on "
                f"{self._currency} and {other._currency}"
            )

    def allocate(self, ratios: list[int]) -> list['Money']:
        """Allocate money according to ratios.

        Handles rounding properly to ensure sum equals original.

        Args:
            ratios: List of integer ratios

        Returns:
            List of Money objects that sum to original

        Example:
            >>> money = Money("100.00", "USD")
            >>> money.allocate([1, 1, 1])
            [Money("33.34", "USD"), Money("33.33", "USD"), Money("33.33", "USD")]
        """
        if not ratios:
            raise ValueError("Ratios cannot be empty")
        if any(r < 0 for r in ratios):
            raise ValueError("Ratios must be non-negative")

        total_ratio = sum(ratios)
        if total_ratio == 0:
            raise ValueError("Total ratio cannot be zero")

        # Calculate allocations
        remainder = self._amount
        results = []

        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                # Last allocation gets the remainder
                results.append(Money(remainder, self._currency))
            else:
                allocation = (self._amount * ratio) / total_ratio
                # Quantize to 2 decimal places
                allocation = allocation.quantize(Decimal('0.01'))
                results.append(Money(allocation, self._currency))
                remainder -= allocation

        return results
```

---

### 2.2 Result Type Pattern (No Exceptions for Control Flow)

**Google Standard**: Don't use exceptions for expected failures.

**Implementation**:

```python
# src/shared/types/result.py
from typing import TypeVar, Generic, Callable, Union
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')

@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result."""
    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Get value or raise."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or default."""
        return self.value

    def map(self, func: Callable[[T], 'U']) -> 'Result[U, E]':
        """Map success value."""
        return Success(func(self.value))


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Failed result."""
    error: E

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def unwrap(self) -> None:
        """Get value or raise."""
        raise ValueError(f"Called unwrap on Failure: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Get value or default."""
        return default

    def map(self, func: Callable) -> 'Result[T, E]':
        """Map does nothing on failure."""
        return self


# Type alias for Result
Result = Union[Success[T], Failure[E]]


# Usage example
# src/core/domain/value_objects/email.py
from dataclasses import dataclass
import re
from typing import ClassVar
from ....shared.types.result import Result, Success, Failure

@dataclass(frozen=True)
class EmailValidationError:
    """Email validation error."""
    message: str
    code: str

@dataclass(frozen=True)
class Email:
    """Email value object with validation."""
    address: str

    EMAIL_REGEX: ClassVar[re.Pattern] = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    @classmethod
    def create(cls, address: str) -> Result['Email', EmailValidationError]:
        """Create email with validation.

        Returns Result instead of raising exceptions.
        """
        # Normalize
        normalized = address.strip().lower()

        # Validate length
        if len(normalized) == 0:
            return Failure(EmailValidationError(
                message="Email cannot be empty",
                code="EMPTY_EMAIL"
            ))

        if len(normalized) > 254:
            return Failure(EmailValidationError(
                message="Email too long (max 254 chars)",
                code="EMAIL_TOO_LONG"
            ))

        # Validate format
        if not cls.EMAIL_REGEX.match(normalized):
            return Failure(EmailValidationError(
                message=f"Invalid email format: {normalized}",
                code="INVALID_FORMAT"
            ))

        # Validate domain
        local, domain = normalized.rsplit('@', 1)
        if domain in ['example.com', 'test.com', 'localhost']:
            return Failure(EmailValidationError(
                message=f"Invalid domain: {domain}",
                code="INVALID_DOMAIN"
            ))

        return Success(cls(address=normalized))


# Usage in application layer
result = Email.create("user@example.com")
if result.is_success():
    email = result.unwrap()
    # Use email
else:
    error = result.error
    # Handle error without exceptions
    logger.warning(f"Email validation failed: {error.message}")
```

---

### 2.3 Specification Pattern for Complex Business Rules

**Implementation**:

```python
# src/core/domain/specifications/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class Specification(ABC, Generic[T]):
    """Base specification for business rules."""

    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if candidate satisfies specification."""
        pass

    def and_(self, other: 'Specification[T]') -> 'Specification[T]':
        """AND combinator."""
        return AndSpecification(self, other)

    def or_(self, other: 'Specification[T]') -> 'Specification[T]':
        """OR combinator."""
        return OrSpecification(self, other)

    def not_(self) -> 'Specification[T]':
        """NOT combinator."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND specification."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        return (self.left.is_satisfied_by(candidate) and
                self.right.is_satisfied_by(candidate))


class OrSpecification(Specification[T]):
    """OR specification."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        return (self.left.is_satisfied_by(candidate) or
                self.right.is_satisfied_by(candidate))


class NotSpecification(Specification[T]):
    """NOT specification."""

    def __init__(self, spec: Specification[T]):
        self.spec = spec

    def is_satisfied_by(self, candidate: T) -> bool:
        return not self.spec.is_satisfied_by(candidate)


# src/core/domain/specifications/request_specifications.py
from ..entities.request import Request
from ..value_objects.priority import Priority

class HighPrioritySpecification(Specification[Request]):
    """Specification for high priority requests."""

    def is_satisfied_by(self, request: Request) -> bool:
        return request.priority.value >= Priority.HIGH.value


class RequiresApprovalSpecification(Specification[Request]):
    """Specification for requests requiring approval."""

    def __init__(self, approval_threshold: Decimal):
        self.approval_threshold = approval_threshold

    def is_satisfied_by(self, request: Request) -> bool:
        # Check if request has financial impact above threshold
        if not request.financial_impact:
            return False
        return request.financial_impact.amount > self.approval_threshold


class RequiresEscalationSpecification(Specification[Request]):
    """Complex business rule for escalation."""

    def is_satisfied_by(self, request: Request) -> bool:
        # Escalate if:
        # 1. High priority AND requires approval
        # 2. OR has low confidence score
        # 3. OR customer is VIP

        high_priority_with_approval = (
            HighPrioritySpecification()
            .and_(RequiresApprovalSpecification(Decimal("1000.00")))
        )

        low_confidence = request.confidence_score < 0.6
        vip_customer = request.customer.tier == "enterprise"

        return (high_priority_with_approval.is_satisfied_by(request) or
                low_confidence or
                vip_customer)


# Usage
spec = RequiresEscalationSpecification()
if spec.is_satisfied_by(request):
    # Escalate
    pass
```

---

### 2.4 Type Safety with Protocols and NewType

**Implementation**:

```python
# src/shared/types/identifiers.py
from typing import NewType
from uuid import UUID, uuid4

# Strong typing for IDs
RequestId = NewType('RequestId', str)
CustomerId = NewType('CustomerId', str)
ThreadId = NewType('ThreadId', str)

def generate_request_id() -> RequestId:
    """Generate unique request ID."""
    return RequestId(f"req-{uuid4()}")

def generate_customer_id() -> CustomerId:
    """Generate unique customer ID."""
    return CustomerId(f"cust-{uuid4()}")

def generate_thread_id() -> ThreadId:
    """Generate unique thread ID."""
    return ThreadId(f"thread-{uuid4()}")


# src/core/application/ports/repositories/request_repository.py
from typing import Protocol, Optional, List
from ....domain.entities.request import Request
from .....shared.types.identifiers import RequestId, CustomerId

class RequestRepository(Protocol):
    """Repository protocol for requests.

    Using Protocol for structural subtyping (duck typing with type checking).
    """

    async def save(self, request: Request) -> None:
        """Save request."""
        ...

    async def find_by_id(self, request_id: RequestId) -> Optional[Request]:
        """Find request by ID."""
        ...

    async def find_by_customer(self, customer_id: CustomerId) -> List[Request]:
        """Find all requests for customer."""
        ...

    async def delete(self, request_id: RequestId) -> bool:
        """Delete request."""
        ...


# Usage - any class implementing these methods satisfies the protocol
class PostgresRequestRepository:
    """Postgres implementation of RequestRepository."""

    async def save(self, request: Request) -> None:
        # Implementation
        pass

    async def find_by_id(self, request_id: RequestId) -> Optional[Request]:
        # Implementation
        pass

    # This class automatically satisfies RequestRepository protocol
```

---

## 3. Security Hardening

### 3.1 Input Validation with Pydantic V2

**Implementation**:

```python
# src/adapters/inbound/api/rest/v1/schemas/request_schemas.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
from datetime import datetime
import re

class CreateRequestSchema(BaseModel):
    """Request creation schema with comprehensive validation."""

    customer_id: str = Field(
        ...,
        min_length=8,
        max_length=100,
        pattern=r'^cust-[a-zA-Z0-9-]{1,90}$',
        description="Customer identifier",
        examples=["cust-123e4567-e89b-12d3-a456-426614174000"]
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Customer message",
        examples=["How do I reset my password?"]
    )

    priority: Optional[int] = Field(
        default=2,
        ge=1,
        le=4,
        description="Priority level (1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL)"
    )

    category: Optional[str] = Field(
        default=None,
        pattern=r'^(account|billing|technical|product|refund|general)$',
        description="Request category"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        """Validate message content."""
        # Remove dangerous characters
        dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=']
        message_lower = v.lower()

        for pattern in dangerous_patterns:
            if pattern in message_lower:
                raise ValueError(f"Message contains forbidden pattern: {pattern}")

        # Check for excessive special characters
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in v) / len(v)
        if special_char_ratio > 0.3:
            raise ValueError("Message contains too many special characters")

        return v.strip()

    @field_validator('metadata')
    @classmethod
    def validate_metadata_size(cls, v: Optional[Dict]) -> Optional[Dict]:
        """Validate metadata size."""
        if v is None:
            return v

        # Limit metadata size
        import json
        metadata_json = json.dumps(v)
        if len(metadata_json) > 10000:
            raise ValueError("Metadata too large (max 10KB)")

        return v

    @model_validator(mode='after')
    def validate_business_rules(self):
        """Cross-field validation."""
        # If category is refund, priority must be at least MEDIUM
        if self.category == 'refund' and self.priority < 2:
            raise ValueError("Refund requests must have priority >= MEDIUM")

        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "cust-123e4567-e89b-12d3-a456-426614174000",
                    "message": "How do I reset my password?",
                    "priority": 2,
                    "category": "account"
                }
            ]
        }
    }
```

---

### 3.2 Authentication & Authorization

**Implementation**:

```python
# src/adapters/inbound/api/rest/v1/middleware/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import os

security = HTTPBearer()

class JWTAuth:
    """JWT authentication handler."""

    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


jwt_auth = JWTAuth()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = jwt_auth.verify_token(token)
    return payload


# src/adapters/inbound/api/rest/v1/middleware/rbac.py
from enum import Enum
from typing import List
from fastapi import HTTPException, status

class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    MANAGER = "manager"
    AGENT = "agent"
    CUSTOMER = "customer"


class Permission(str, Enum):
    """Permissions."""
    CREATE_REQUEST = "request:create"
    READ_REQUEST = "request:read"
    UPDATE_REQUEST = "request:update"
    DELETE_REQUEST = "request:delete"
    APPROVE_REQUEST = "request:approve"
    VIEW_ANALYTICS = "analytics:view"


ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],
    Role.MANAGER: [
        Permission.READ_REQUEST,
        Permission.UPDATE_REQUEST,
        Permission.APPROVE_REQUEST,
        Permission.VIEW_ANALYTICS
    ],
    Role.AGENT: [
        Permission.CREATE_REQUEST,
        Permission.READ_REQUEST,
        Permission.UPDATE_REQUEST
    ],
    Role.CUSTOMER: [
        Permission.CREATE_REQUEST,
        Permission.READ_REQUEST
    ]
}


def require_permission(required_permission: Permission):
    """Decorator to require specific permission."""

    async def permission_checker(
        current_user: dict = Depends(get_current_user)
    ):
        user_role = Role(current_user.get("role", "customer"))
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])

        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires {required_permission.value}"
            )

        return current_user

    return permission_checker


# Usage in endpoint
@app.post("/requests/{request_id}/approve")
async def approve_request(
    request_id: str,
    approval: ApprovalSchema,
    current_user: dict = Depends(require_permission(Permission.APPROVE_REQUEST))
):
    # Only users with APPROVE_REQUEST permission can access
    pass
```

---

### 3.3 Rate Limiting

**Implementation**:

```python
# src/adapters/inbound/api/rest/v1/middleware/rate_limit.py
from fastapi import Request, HTTPException, status
from typing import Optional
import time
from collections import defaultdict
import asyncio
from dataclasses import dataclass, field

@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    refill_rate: float  # tokens per second

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens."""
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Distributed rate limiter."""

    def __init__(self):
        self.buckets: dict[str, RateLimitBucket] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            if key not in self.buckets:
                self.buckets[key] = RateLimitBucket(
                    capacity=max_requests,
                    refill_rate=max_requests / window_seconds
                )

            bucket = self.buckets[key]
            return bucket.consume()


rate_limiter = RateLimiter()


async def rate_limit_middleware(
    request: Request,
    call_next
):
    """Rate limiting middleware."""
    # Get identifier (IP or user ID)
    identifier = request.client.host
    if hasattr(request.state, "user"):
        identifier = request.state.user.get("user_id", identifier)

    # Check rate limit (100 requests per minute)
    allowed = await rate_limiter.check_rate_limit(
        key=f"rate_limit:{identifier}",
        max_requests=100,
        window_seconds=60
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )

    response = await call_next(request)
    return response
```

---

## 4. Performance Optimization

### 4.1 Connection Pooling

**Implementation**:

```python
# src/adapters/outbound/persistence/postgres/connection_pool.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncpg
from asyncpg.pool import Pool
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """PostgreSQL connection pool manager."""

    def __init__(self):
        self._pool: Optional[Pool] = None

    async def initialize(self):
        """Initialize connection pool."""
        self._pool = await asyncpg.create_pool(
            dsn=os.getenv("POSTGRES_URL"),
            min_size=5,
            max_size=20,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            timeout=30,
            command_timeout=30,
            server_settings={
                'application_name': 'enterprise-agent-system',
                'jit': 'off'  # Disable JIT for faster simple queries
            }
        )
        logger.info(f"Database pool initialized (min=5, max=20)")

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator:
        """Acquire connection from pool."""
        if not self._pool:
            raise RuntimeError("Pool not initialized")

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args, timeout: float = 30):
        """Execute query using pool."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(self, query: str, *args, timeout: float = 30):
        """Fetch results using pool."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args, timeout: float = 30):
        """Fetch single row using pool."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)


# Global pool instance
db_pool = DatabaseConnectionPool()


# src/adapters/outbound/persistence/redis/connection_pool.py
import redis.asyncio as redis
from typing import Optional

class RedisConnectionPool:
    """Redis connection pool manager."""

    def __init__(self):
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize connection pool."""
        self._pool = redis.ConnectionPool.from_url(
            os.getenv("REDIS_URL"),
            max_connections=50,
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            health_check_interval=30
        )
        self._client = redis.Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()
        logger.info("Redis pool initialized (max_connections=50)")

    async def close(self):
        """Close connection pool."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis pool closed")

    @property
    def client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._client:
            raise RuntimeError("Pool not initialized")
        return self._client


redis_pool = RedisConnectionPool()
```

---

### 4.2 Caching Strategy

**Implementation**:

```python
# src/shared/caching/cache_decorator.py
from functools import wraps
from typing import Optional, Callable, Any
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

def cache(
    ttl: int = 300,
    key_prefix: Optional[str] = None,
    key_builder: Optional[Callable] = None
):
    """Caching decorator.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_builder: Custom function to build cache key
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            from ...adapters.outbound.persistence.redis.connection_pool import redis_pool

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key building
                prefix = key_prefix or f"cache:{func.__module__}.{func.__name__}"

                # Create deterministic key from args/kwargs
                key_data = {
                    'args': str(args),
                    'kwargs': sorted(kwargs.items())
                }
                key_hash = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode()
                ).hexdigest()

                cache_key = f"{prefix}:{key_hash}"

            # Try to get from cache
            try:
                cached = await redis_pool.client.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache GET error: {e}")

            # Cache miss - execute function
            logger.debug(f"Cache MISS: {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await redis_pool.client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Cache SET error: {e}")

            return result

        return wrapper
    return decorator


# Usage
@cache(ttl=600, key_prefix="customer")
async def get_customer_by_id(customer_id: str) -> Optional[dict]:
    """Get customer - cached for 10 minutes."""
    # Database query
    pass
```

---

## 5. Testing Strategy

### 5.1 Property-Based Testing with Hypothesis

**Implementation**:

```python
# tests/unit/core/domain/value_objects/test_money_properties.py
from hypothesis import given, strategies as st, assume
from decimal import Decimal
import pytest

from src.core.domain.value_objects.money import Money

# Custom strategies
money_strategy = st.builds(
    Money,
    amount=st.decimals(
        min_value=0,
        max_value=1000000,
        places=2
    ),
    currency=st.sampled_from(["USD", "EUR", "GBP", "JPY"])
)

@given(money_strategy)
def test_money_is_always_non_negative(money: Money):
    """Property: Money amount is always non-negative."""
    assert money.amount >= 0

@given(money_strategy, money_strategy)
def test_money_addition_is_commutative_same_currency(m1: Money, m2: Money):
    """Property: a + b = b + a for same currency."""
    assume(m1.currency == m2.currency)

    result1 = m1 + m2
    result2 = m2 + m1

    assert result1 == result2

@given(money_strategy, money_strategy, money_strategy)
def test_money_addition_is_associative(m1: Money, m2: Money, m3: Money):
    """Property: (a + b) + c = a + (b + c)."""
    assume(m1.currency == m2.currency == m3.currency)

    result1 = (m1 + m2) + m3
    result2 = m1 + (m2 + m3)

    assert result1 == result2

@given(money_strategy)
def test_money_identity_element(money: Money):
    """Property: m + 0 = m."""
    zero = Money(0, money.currency)
    result = money + zero

    assert result == money

@given(
    st.decimals(min_value=0, max_value=1000, places=2),
    st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10)
)
def test_money_allocation_sum_equals_original(amount: Decimal, ratios: list[int]):
    """Property: Sum of allocated amounts equals original."""
    money = Money(amount, "USD")
    allocated = money.allocate(ratios)

    total = sum(m.amount for m in allocated)

    # Should be equal within rounding error (2 decimal places)
    assert abs(total - money.amount) < Decimal("0.01")
```

---

### 5.2 Contract Testing

**Implementation**:

```python
# tests/contract/test_repository_contract.py
import pytest
from abc import ABC, abstractmethod

class RepositoryContractTest(ABC):
    """Base contract test for all repositories.

    Any repository implementation must pass these tests.
    """

    @abstractmethod
    @pytest.fixture
    def repository(self):
        """Provide repository implementation."""
        pass

    @abstractmethod
    @pytest.fixture
    def sample_entity(self):
        """Provide sample entity."""
        pass

    @pytest.mark.asyncio
    async def test_save_and_find_by_id(self, repository, sample_entity):
        """Contract: Save entity and find it by ID."""
        # Save
        await repository.save(sample_entity)

        # Find
        found = await repository.find_by_id(sample_entity.id)

        assert found is not None
        assert found.id == sample_entity.id

    @pytest.mark.asyncio
    async def test_find_by_id_not_found_returns_none(self, repository):
        """Contract: Find non-existent entity returns None."""
        result = await repository.find_by_id("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, repository, sample_entity):
        """Contract: Delete existing entity returns True."""
        await repository.save(sample_entity)
        result = await repository.delete(sample_entity.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_non_existent_returns_false(self, repository):
        """Contract: Delete non-existent entity returns False."""
        result = await repository.delete("non-existent-id")
        assert result is False


# tests/integration/persistence/test_postgres_request_repository.py
class TestPostgresRequestRepository(RepositoryContractTest):
    """Test that Postgres implementation satisfies repository contract."""

    @pytest.fixture
    def repository(self, db_connection):
        from src.adapters.outbound.persistence.postgres.repositories import PostgresRequestRepository
        return PostgresRequestRepository(db_connection)

    @pytest.fixture
    def sample_entity(self):
        from src.core.domain.entities.request import Request
        return Request.create(
            request_id="req-test-123",
            customer_id="cust-123",
            message="Test message"
        )
```

---

### 5.3 Mutation Testing

**Implementation**:

```bash
# Run mutation testing with mutmut
mutmut run --paths-to-mutate=src/

# Check survivors (mutations that didn't break tests)
mutmut results

# Show specific mutation
mutmut show 42
```

**Configuration**:

```python
# setup.cfg
[mutmut]
paths_to_mutate=src/
backup=False
runner=python -m pytest -x
tests_dir=tests/
dict_synonyms=Struct, NamedStruct
```

---

## 6. Observability & Monitoring

### 6.1 Structured Logging

**Implementation**:

```python
# src/shared/logging/structured_logger.py
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import uuid

# Context variables for correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

class StructuredLogger(logging.Logger):
    """Structured logger with correlation IDs."""

    def _log_structured(
        self,
        level: int,
        msg: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Any = None
    ):
        """Log with structured format."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
            "message": msg,
            "correlation_id": correlation_id.get(),
            "request_id": request_id.get(),
            "user_id": user_id.get(),
            "service": "enterprise-agent-system"
        }

        if extra:
            log_entry["extra"] = extra

        if exc_info:
            import traceback
            log_entry["exception"] = {
                "type": exc_info[0].__name__ if exc_info[0] else None,
                "message": str(exc_info[1]) if exc_info[1] else None,
                "traceback": traceback.format_exception(*exc_info)
            }

        super()._log(level, json.dumps(log_entry), ())

    def info(self, msg: str, extra: Optional[Dict] = None):
        self._log_structured(logging.INFO, msg, extra)

    def warning(self, msg: str, extra: Optional[Dict] = None):
        self._log_structured(logging.WARNING, msg, extra)

    def error(self, msg: str, extra: Optional[Dict] = None, exc_info: Any = None):
        self._log_structured(logging.ERROR, msg, extra, exc_info)


# Register custom logger
logging.setLoggerClass(StructuredLogger)


# src/adapters/inbound/api/rest/v1/middleware/correlation.py
from fastapi import Request
import uuid

async def correlation_middleware(request: Request, call_next):
    """Add correlation ID to all requests."""
    # Get or generate correlation ID
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    correlation_id.set(corr_id)

    # Generate request ID
    req_id = f"req-{uuid.uuid4()}"
    request_id.set(req_id)

    # Add to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Request-ID"] = req_id

    return response
```

---

### 6.2 Distributed Tracing with OpenTelemetry

**Implementation**:

```python
# src/shared/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import os

def setup_tracing(app):
    """Setup distributed tracing."""
    # Create tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_PORT", "6831")),
    )

    # Add span processor
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument database
    AsyncPGInstrumentor().instrument()

    # Instrument Redis
    RedisInstrumentor().instrument()

    return tracer


# Usage in code
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def process_request(request_id: str):
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("request.id", request_id)

        # Nested spans
        with tracer.start_as_current_span("triage"):
            # Triage logic
            pass

        with tracer.start_as_current_span("research"):
            # Research logic
            pass
```

---

### 6.3 Metrics with Prometheus

**Implementation**:

```python
# src/shared/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration',
    ['agent_type']
)

agent_execution_errors_total = Counter(
    'agent_execution_errors_total',
    'Agent execution errors',
    ['agent_type', 'error_type']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

llm_tokens_used = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens used',
    ['model', 'operation']
)

cache_hits_total = Counter(
    'cache_hits_total',
    'Cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Cache misses',
    ['cache_type']
)

app_info = Info('app', 'Application information')
app_info.info({
    'version': '1.0.0',
    'environment': 'production'
})


def track_execution_time(metric: Histogram, labels: dict):
    """Decorator to track execution time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        return wrapper
    return decorator


# Usage
@track_execution_time(
    agent_execution_duration_seconds,
    {'agent_type': 'triage'}
)
async def execute_triage(state):
    # Triage logic
    pass
```

---

## 7. API Design

### 7.1 Versioning Strategy

**Implementation**:

```
/api/v1/requests
/api/v2/requests

Headers:
Accept: application/vnd.enterprise-agent-system.v1+json
```

---

### 7.2 HATEOAS (Hypermedia)

**Implementation**:

```python
# src/adapters/inbound/api/rest/v1/schemas/responses.py
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional

class Link(BaseModel):
    """HATEOAS link."""
    href: HttpUrl
    rel: str
    method: str = "GET"
    description: Optional[str] = None

class RequestResponse(BaseModel):
    """Request response with HATEOAS links."""
    request_id: str
    status: str
    message: str
    created_at: datetime

    # HATEOAS links
    _links: Dict[str, Link]

    @classmethod
    def from_entity(cls, request: Request, base_url: str) -> 'RequestResponse':
        """Create response from entity with links."""
        links = {
            "self": Link(
                href=f"{base_url}/requests/{request.id}",
                rel="self",
                method="GET"
            ),
            "update": Link(
                href=f"{base_url}/requests/{request.id}",
                rel="update",
                method="PATCH"
            )
        }

        # Add conditional links based on state
        if request.status == "pending":
            links["approve"] = Link(
                href=f"{base_url}/requests/{request.id}/approve",
                rel="approve",
                method="POST"
            )

        if request.status == "completed":
            links["history"] = Link(
                href=f"{base_url}/requests/{request.id}/history",
                rel="history",
                method="GET"
            )

        return cls(
            request_id=request.id,
            status=request.status,
            message=request.message,
            created_at=request.created_at,
            _links=links
        )
```

---

## 8. Error Handling

### 8.1 Custom Exception Hierarchy

**Implementation**:

```python
# src/core/domain/exceptions/base.py
from typing import Optional, Dict, Any

class DomainException(Exception):
    """Base exception for domain errors."""

    def __init__(
        self,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }


class ValidationError(DomainException):
    """Validation error."""

    def __init__(self, message: str, field: Optional[str] = None, **details):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"field": field, **details}
        )


class BusinessRuleViolation(DomainException):
    """Business rule violation."""

    def __init__(self, message: str, rule: str, **details):
        super().__init__(
            message=message,
            code="BUSINESS_RULE_VIOLATION",
            details={"rule": rule, **details}
        )


class ResourceNotFound(DomainException):
    """Resource not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ConcurrencyConflict(DomainException):
    """Concurrency conflict."""

    def __init__(self, message: str, **details):
        super().__init__(
            message=message,
            code="CONCURRENCY_CONFLICT",
            details=details
        )


# src/adapters/inbound/api/rest/v1/error_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

async def domain_exception_handler(request: Request, exc: DomainException):
    """Handle domain exceptions."""
    logger.warning(
        f"Domain exception: {exc.code}",
        extra={"exception": exc.to_dict()}
    )

    status_map = {
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "BUSINESS_RULE_VIOLATION": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "RESOURCE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "CONCURRENCY_CONFLICT": status.HTTP_409_CONFLICT
    }

    return JSONResponse(
        status_code=status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR),
        content={
            "error": exc.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "error_type": "ValidationError",
                "message": "Request validation failed",
                "code": "VALIDATION_ERROR",
                "details": {
                    "errors": exc.errors()
                }
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )
```

---

## 9. Documentation

### 9.1 ADR (Architecture Decision Records)

**Template**:

```markdown
# ADR-001: Use Hexagonal Architecture

## Status
Accepted

## Context
Need to ensure domain logic is independent of infrastructure and can be easily tested.

## Decision
Adopt Hexagonal Architecture (Ports & Adapters) pattern.

## Consequences
### Positive
- Domain logic completely testable without infrastructure
- Easy to swap implementations
- Clear dependency boundaries

### Negative
- More boilerplate code
- Steeper learning curve

## Alternatives Considered
1. Layered architecture - rejected due to tight coupling
2. Clean architecture - similar to hexagonal, chosen hexagonal for simplicity
```

---

### 9.2 API Documentation with Examples

**Implementation**:

```python
# src/adapters/inbound/api/rest/v1/handlers/request_handlers.py
from fastapi import APIRouter, Depends, status
from typing import List

router = APIRouter(prefix="/requests", tags=["requests"])

@router.post(
    "",
    response_model=RequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new request",
    description="""
    Create a new customer support request.

    ## Business Rules
    - Message must be 1-10000 characters
    - Refund requests require priority >= MEDIUM
    - System automatically assigns category if not provided

    ## Example Scenarios

    ### Password Reset (Common)
    ```json
    {
      "customer_id": "cust-123",
      "message": "How do I reset my password?",
      "priority": 2,
      "category": "account"
    }
    ```

    ### Refund Request (Requires Approval)
    ```json
    {
      "customer_id": "cust-456",
      "message": "I need a refund for order #789",
      "priority": 3,
      "category": "refund",
      "metadata": {
        "order_id": "789",
        "amount": "99.99"
      }
    }
    ```
    """,
    responses={
        201: {
            "description": "Request created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
                        "status": "pending",
                        "created_at": "2024-01-01T12:00:00Z",
                        "_links": {
                            "self": {
                                "href": "/api/v1/requests/req-123e4567-e89b-12d3-a456-426614174000",
                                "rel": "self",
                                "method": "GET"
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Message too long (max 10000 chars)"
                        }
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded"
        }
    }
)
async def create_request(
    request: CreateRequestSchema,
    current_user: dict = Depends(get_current_user)
):
    """Create request endpoint."""
    pass
```

---

## 10. Infrastructure as Code

### 10.1 Terraform for Infrastructure

**Implementation**:

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }

  backend "s3" {
    bucket = "enterprise-agent-system-terraform"
    key    = "state/terraform.tfstate"
    region = "us-west-2"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "enterprise-agent-system"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["t3.large"]
      capacity_type  = "SPOT"
    }
  }
}

# RDS PostgreSQL
module "rds" {
  source = "terraform-aws-modules/rds/aws"

  identifier = "enterprise-agent-system-db"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.t3.large"

  allocated_storage     = 100
  max_allocated_storage = 1000

  db_name  = "agent_system"
  username = var.db_username
  port     = 5432

  multi_az               = true
  vpc_security_group_ids = [module.security_group.security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name

  backup_retention_period = 30
  backup_window           = "03:00-06:00"
  maintenance_window      = "Mon:00:00-Mon:03:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  tags = local.tags
}

# ElastiCache Redis
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"

  cluster_id           = "enterprise-agent-system"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 2
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379

  subnet_ids = module.vpc.private_subnets
}
```

---

## 11. Implementation Roadmap

### Week 1-2: Foundation & Architecture
**Priority**: P0 (Critical Bugs + Architecture)

**Tasks**:
- [ ] Create requirements.txt with pinned versions
- [ ] Create all missing __init__.py files
- [ ] Refactor to Hexagonal Architecture
  - [ ] Define ports (interfaces)
  - [ ] Create adapters directory structure
  - [ ] Move domain logic to core/
- [ ] Implement CQRS pattern
  - [ ] Define commands and queries
  - [ ] Create command handlers
  - [ ] Create query handlers
- [ ] Fix Customer mutable hash bug
- [ ] Implement immutable Money class
- [ ] Add domain events

**Deliverables**:
- Working project structure
- Fixed critical bugs
- Clean architecture foundation

---

### Week 3-4: Code Quality & Security
**Priority**: P1 (High Severity)

**Tasks**:
- [ ] Implement Result type pattern
- [ ] Add comprehensive validation with Pydantic V2
- [ ] Implement Specification pattern
- [ ] Strong typing with NewType and Protocols
- [ ] Fix API dependencies (replace None returns)
- [ ] Implement JWT authentication
- [ ] Add RBAC authorization
- [ ] Implement rate limiting
- [ ] Fix CORS configuration
- [ ] Add input sanitization

**Deliverables**:
- Type-safe codebase
- Secure API with auth
- Production-ready validation

---

### Week 5-6: Performance & Observability
**Priority**: P1-P2

**Tasks**:
- [ ] Implement connection pooling
  - [ ] PostgreSQL pool
  - [ ] Redis pool
  - [ ] HTTP client pool
- [ ] Add caching layer with decorators
- [ ] Implement structured logging
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Implement Prometheus metrics
- [ ] Create Grafana dashboards
- [ ] Add health checks
- [ ] Implement graceful shutdown

**Deliverables**:
- Optimized performance
- Full observability
- Production monitoring

---

### Week 7: Testing Excellence
**Priority**: P2

**Tasks**:
- [ ] Property-based tests with Hypothesis
- [ ] Contract tests for repositories
- [ ] Integration tests with real services
- [ ] E2E tests with realistic scenarios
- [ ] Mutation testing setup
- [ ] Performance tests
- [ ] Chaos engineering tests

**Deliverables**:
- 98%+ test coverage
- High-quality test suite
- CI/CD pipeline

---

### Week 8: Documentation & Polish
**Priority**: P3

**Tasks**:
- [ ] Create ADRs for key decisions
- [ ] Comprehensive API documentation
- [ ] OpenAPI spec with examples
- [ ] Runbooks for operations
- [ ] Migration guides
- [ ] Performance tuning guide
- [ ] Security audit documentation

**Deliverables**:
- Complete documentation
- Production-ready system
- Operational excellence

---

## Success Criteria for A+ Grade

### Code Quality (30%)
- [ ] 100% type hints with mypy strict mode passing
- [ ] No code smells (SonarQube Quality Gate A)
- [ ] Cyclomatic complexity < 10 for all functions
- [ ] Zero security vulnerabilities (Bandit, Safety)
- [ ] Consistent code style (Black, isort, flake8)

### Architecture (25%)
- [ ] Clean separation of concerns (Hexagonal Architecture)
- [ ] SOLID principles applied throughout
- [ ] Domain-driven design with rich models
- [ ] Event-driven architecture
- [ ] Proper dependency injection

### Testing (20%)
- [ ] 98%+ code coverage
- [ ] Property-based tests for core logic
- [ ] Contract tests for all interfaces
- [ ] Integration tests with real infrastructure
- [ ] Mutation testing score > 90%

### Security (15%)
- [ ] Authentication and authorization
- [ ] Input validation and sanitization
- [ ] Rate limiting and DoS protection
- [ ] Secrets management
- [ ] Security audit passing

### Performance (10%)
- [ ] < 100ms p50 latency
- [ ] < 500ms p99 latency
- [ ] Handle 1000+ concurrent users
- [ ] Efficient resource usage
- [ ] Proper caching strategy

---

## Conclusion

This refactoring plan transforms the codebase from B+ to A+ by applying Google-level engineering standards across architecture, code quality, security, performance, testing, and operations.

**Estimated Timeline**: 8 weeks
**Engineering Effort**: ~320 hours
**Investment**: High
**Return**: Production-ready, maintainable, scalable system

The refactored system will exemplify:
- **Technical Excellence**: Clean architecture, type safety, comprehensive testing
- **Operational Excellence**: Observability, monitoring, reliability
- **Security Excellence**: Defense in depth, secure by default
- **Performance Excellence**: Optimized, scalable, efficient

**Next Steps**:
1. Get stakeholder approval
2. Allocate engineering resources
3. Set up project tracking
4. Begin Week 1 implementation
5. Weekly progress reviews
6. Final QA assessment

---

*Prepared by: Senior Staff Engineer*
*Target Audience: Engineering Leadership*
*Classification: Internal Use Only*
