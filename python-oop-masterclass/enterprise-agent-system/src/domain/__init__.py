"""
Domain Layer - Core Business Logic

Contains domain models, value objects, entities, and business rules.
"""

from .models import (
    # Enums
    Priority,
    RequestCategory,
    RequestStatus,
    AgentType,
    DecisionType,

    # Value Objects
    Money,
    ContactInfo,
    TimeRange,
    Confidence,

    # Entities
    Customer,
    AgentDecision,
    Message,

    # Aggregate Root
    CustomerRequest,

    # Domain Services
    PriorityCalculator,
    ConfidenceCalculator,

    # Repository Interfaces
    CustomerRequestRepository,
    CustomerRepository,

    # Factory
    CustomerRequestFactory,

    # Specifications
    Specification,
    HighPrioritySpecification,
    RequiresApprovalSpecification,
    PremiumCustomerSpecification,
)

__all__ = [
    # Enums
    "Priority",
    "RequestCategory",
    "RequestStatus",
    "AgentType",
    "DecisionType",

    # Value Objects
    "Money",
    "ContactInfo",
    "TimeRange",
    "Confidence",

    # Entities
    "Customer",
    "AgentDecision",
    "Message",

    # Aggregate Root
    "CustomerRequest",

    # Domain Services
    "PriorityCalculator",
    "ConfidenceCalculator",

    # Repository Interfaces
    "CustomerRequestRepository",
    "CustomerRepository",

    # Factory
    "CustomerRequestFactory",

    # Specifications
    "Specification",
    "HighPrioritySpecification",
    "RequiresApprovalSpecification",
    "PremiumCustomerSpecification",
]
