"""
Improved Entity Classes for Enterprise Agent System

Fixes BUG-007: Customer mutable hash problem
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID
from typing import Dict, Any

from .value_objects import Money, ContactInfo


# ============================================================================
# CUSTOMER ENTITY - Fixed Mutable Hash Problem
# ============================================================================

@dataclass
class Customer:
    """Customer entity with identity.

    **FIX for BUG-007**: Removed __hash__ method to comply with Python's
    hashable contract. Mutable objects should not implement __hash__.

    The class still implements __eq__ for identity comparison based on ID,
    but cannot be used in sets/dicts. If hashability is needed, make the
    class frozen or use the ID directly as the dict key.

    Demonstrates:
    - Entity pattern (has ID)
    - Encapsulation (private attributes)
    - Properties for controlled access
    - Proper mutability handling
    """
    id: UUID
    name: str
    contact: ContactInfo
    tier: str = "standard"
    created_at: datetime = field(default_factory=datetime.now)
    _lifetime_value: Decimal = field(default=Decimal("0.00"), init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate customer data after initialization.

        Raises:
            ValueError: If validation fails
        """
        # Validate tier
        valid_tiers = {"standard", "premium", "enterprise"}
        if self.tier not in valid_tiers:
            raise ValueError(
                f"Invalid tier: '{self.tier}'. Must be one of: {valid_tiers}"
            )

        # Validate name
        if not self.name or not self.name.strip():
            raise ValueError("Customer name cannot be empty")

        if len(self.name) > 200:
            raise ValueError(f"Customer name too long: {len(self.name)} chars (max 200)")

    def __eq__(self, other: object) -> bool:
        """Entities are equal if IDs match.

        Args:
            other: Object to compare with

        Returns:
            bool: True if IDs match
        """
        if not isinstance(other, Customer):
            return NotImplemented
        return self.id == other.id

    # __hash__ REMOVED - Mutable objects should not be hashable
    # If you need to use Customer as dict key, use customer.id instead:
    #   customer_dict[customer.id] = data

    @property
    def lifetime_value(self) -> Decimal:
        """Get lifetime value (read-only).

        Returns:
            Decimal: Total lifetime value in USD
        """
        return self._lifetime_value

    def add_transaction(self, amount: Money) -> None:
        """Add transaction to lifetime value.

        **FIX for BUG-008**: Now raises error for non-USD instead of
        silently ignoring the transaction.

        Args:
            amount: Transaction amount (must be USD)

        Raises:
            ValueError: If currency is not USD
            TypeError: If amount is not Money instance
        """
        if not isinstance(amount, Money):
            raise TypeError(
                f"Expected Money instance, got {type(amount).__name__}"
            )

        if amount.currency != "USD":
            raise ValueError(
                f"Transaction currency must be USD, got: {amount.currency}. "
                f"Convert to USD before adding to lifetime value."
            )

        self._lifetime_value += amount.amount

    def is_premium(self) -> bool:
        """Check if customer is premium tier or higher.

        Returns:
            bool: True if tier is premium or enterprise
        """
        return self.tier in ["premium", "enterprise"]

    def get_priority_boost(self) -> int:
        """Get priority boost based on customer tier.

        Returns:
            int: Priority boost level (0-2)
        """
        boosts = {
            "standard": 0,
            "premium": 1,
            "enterprise": 2
        }
        return boosts.get(self.tier, 0)

    def upgrade_tier(self, new_tier: str) -> None:
        """Upgrade customer to a new tier.

        Args:
            new_tier: New tier level

        Raises:
            ValueError: If new tier is invalid or is a downgrade
        """
        tier_levels = {"standard": 1, "premium": 2, "enterprise": 3}

        if new_tier not in tier_levels:
            raise ValueError(
                f"Invalid tier: '{new_tier}'. Must be one of: {set(tier_levels.keys())}"
            )

        current_level = tier_levels[self.tier]
        new_level = tier_levels[new_tier]

        if new_level < current_level:
            raise ValueError(
                f"Cannot downgrade from {self.tier} to {new_tier}. "
                f"Use downgrade_tier() method instead."
            )

        if new_level == current_level:
            raise ValueError(f"Customer is already {self.tier} tier")

        self.tier = new_tier

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Customer(id={self.id}, name='{self.name}', "
            f"tier='{self.tier}', lifetime_value={self._lifetime_value})"
        )


# ============================================================================
# AGENT DECISION ENTITY
# ============================================================================

@dataclass
class AgentDecision:
    """Represents a decision made by an agent.

    Demonstrates:
    - Entity with identity
    - Immutable after creation (no setters)
    - Rich domain model
    """
    id: UUID
    agent_type: str  # From AgentType enum
    decision_type: str  # From DecisionType enum
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate decision after initialization.

        Raises:
            ValueError: If validation fails
        """
        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("Decision must have non-empty reasoning")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got: {self.confidence}"
            )

        if len(self.reasoning) > 5000:
            raise ValueError(
                f"Reasoning too long: {len(self.reasoning)} chars (max 5000)"
            )

    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if decision is confident.

        Args:
            threshold: Minimum confidence level (default: 0.8)

        Returns:
            bool: True if confidence >= threshold
        """
        return self.confidence >= threshold

    def requires_human_review(self) -> bool:
        """Check if decision requires human review.

        Returns:
            bool: True if confidence is low or decision type requires review
        """
        return (
            self.confidence < 0.6 or
            self.decision_type in ["escalate", "reject"]
        )

    def __eq__(self, other: object) -> bool:
        """Decisions are equal if IDs match."""
        if not isinstance(other, AgentDecision):
            return NotImplemented
        return self.id == other.id


# ============================================================================
# MESSAGE ENTITY
# ============================================================================

@dataclass
class Message:
    """Message entity for conversation tracking."""
    id: UUID
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message.

        Raises:
            ValueError: If validation fails
        """
        valid_roles = {"user", "assistant", "system"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role: '{self.role}'. Must be one of: {valid_roles}"
            )

        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")

        if len(self.content) > 50000:
            raise ValueError(
                f"Message content too long: {len(self.content)} chars (max 50000)"
            )

    def is_user_message(self) -> bool:
        """Check if message is from user.

        Returns:
            bool: True if role is 'user'
        """
        return self.role == "user"

    def contains_keywords(self, keywords: list[str]) -> bool:
        """Check if message contains any of the given keywords.

        Args:
            keywords: List of keywords to search for (case-insensitive)

        Returns:
            bool: True if any keyword is found
        """
        content_lower = self.content.lower()
        return any(kw.lower() in content_lower for kw in keywords)

    def __eq__(self, other: object) -> bool:
        """Messages are equal if IDs match."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.id == other.id
