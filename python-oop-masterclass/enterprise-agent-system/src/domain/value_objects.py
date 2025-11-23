"""
Improved Value Objects for Enterprise Agent System

This module contains production-ready value objects with:
- True immutability
- Comprehensive validation
- Type safety
- Better error handling
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
import re


# ============================================================================
# MONEY VALUE OBJECT - Production Quality
# ============================================================================

class Money:
    """Immutable money value object with comprehensive validation.

    Following DDD principles and addressing BUG-004, BUG-005:
    - True immutability using __slots__
    - Accepts multiple numeric types
    - Memory efficient
    - Prevents common currency bugs

    Examples:
        >>> money = Money("100.50", "USD")
        >>> money.amount
        Decimal('100.50')
        >>> doubled = money.multiply(2)
        >>> str(doubled)
        'USD 201.00'
    """

    __slots__ = ('_amount', '_currency')

    def __init__(self, amount: Decimal | float | int | str, currency: str = "USD"):
        """Initialize money value.

        Args:
            amount: Monetary amount (converted to Decimal for precision)
            currency: 3-letter ISO 4217 currency code

        Raises:
            ValueError: If amount is negative or currency is invalid
            TypeError: If amount type cannot be converted to Decimal
        """
        # Convert to Decimal for precision
        decimal_amount = self._to_decimal(amount)

        # Validate amount
        if decimal_amount < 0:
            raise ValueError(f"Amount cannot be negative: {decimal_amount}")

        # Validate and normalize currency
        currency_upper = currency.upper().strip()
        if len(currency_upper) != 3:
            raise ValueError(
                f"Currency must be 3-letter ISO code, got: '{currency}' (length: {len(currency_upper)})"
            )

        # Validate currency format (only letters)
        if not currency_upper.isalpha():
            raise ValueError(f"Currency code must contain only letters: '{currency_upper}'")

        # Use object.__setattr__ to set on frozen object
        object.__setattr__(self, '_amount', decimal_amount)
        object.__setattr__(self, '_currency', currency_upper)

    @staticmethod
    def _to_decimal(value: Decimal | float | int | str) -> Decimal:
        """Convert value to Decimal with proper precision.

        Args:
            value: Value to convert

        Returns:
            Decimal representation

        Raises:
            TypeError: If value type is not supported
            ValueError: If value cannot be converted
        """
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, str):
            try:
                return Decimal(value)
            except Exception as e:
                raise ValueError(f"Cannot convert string '{value}' to Decimal: {e}") from e
        if isinstance(value, float):
            # Convert float to string first to avoid precision issues
            # e.g., Decimal(0.1) != Decimal('0.1')
            return Decimal(str(value))
        raise TypeError(
            f"Cannot convert {type(value).__name__} to Decimal. "
            f"Supported types: Decimal, int, float, str"
        )

    @property
    def amount(self) -> Decimal:
        """Get amount (read-only).

        Returns:
            Decimal: The monetary amount
        """
        return self._amount

    @property
    def currency(self) -> str:
        """Get currency code (read-only).

        Returns:
            str: The 3-letter ISO currency code
        """
        return self._currency

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification (true immutability).

        Raises:
            AttributeError: Always raised to prevent modification
        """
        raise AttributeError(
            f"Money objects are immutable. Cannot set attribute '{name}'. "
            f"Create a new Money object instead."
        )

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion (true immutability).

        Raises:
            AttributeError: Always raised to prevent deletion
        """
        raise AttributeError(
            f"Money objects are immutable. Cannot delete attribute '{name}'."
        )

    def add(self, other: Money) -> Money:
        """Add two money amounts.

        Args:
            other: Another Money instance

        Returns:
            Money: New Money instance with sum

        Raises:
            TypeError: If other is not a Money instance
            ValueError: If currencies don't match
        """
        if not isinstance(other, Money):
            raise TypeError(
                f"Cannot add Money with {type(other).__name__}. "
                f"Both operands must be Money instances."
            )
        if self._currency != other._currency:
            raise ValueError(
                f"Currency mismatch: cannot add {self._currency} and {other._currency}. "
                f"Convert to same currency first."
            )
        return Money(self._amount + other._amount, self._currency)

    def subtract(self, other: Money) -> Money:
        """Subtract two money amounts.

        Args:
            other: Another Money instance

        Returns:
            Money: New Money instance with difference

        Raises:
            TypeError: If other is not a Money instance
            ValueError: If currencies don't match or result would be negative
        """
        if not isinstance(other, Money):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from Money. "
                f"Both operands must be Money instances."
            )
        if self._currency != other._currency:
            raise ValueError(
                f"Currency mismatch: cannot subtract {other._currency} from {self._currency}. "
                f"Convert to same currency first."
            )
        result_amount = self._amount - other._amount
        if result_amount < 0:
            raise ValueError(
                f"Subtraction would result in negative amount: {result_amount}. "
                f"Money cannot be negative."
            )
        return Money(result_amount, self._currency)

    def multiply(self, factor: Decimal | float | int) -> Money:
        """Multiply money by a factor.

        Args:
            factor: Multiplication factor (must be non-negative)

        Returns:
            Money: New Money instance

        Raises:
            ValueError: If factor is negative
            TypeError: If factor cannot be converted to Decimal

        Examples:
            >>> money = Money("10.00", "USD")
            >>> money.multiply(2.5)
            Money(amount=Decimal('25.00'), currency='USD')
        """
        decimal_factor = self._to_decimal(factor)
        if decimal_factor < 0:
            raise ValueError(
                f"Factor cannot be negative: {decimal_factor}. "
                f"Use positive factors only."
            )
        return Money(self._amount * decimal_factor, self._currency)

    def divide(self, divisor: Decimal | float | int) -> Money:
        """Divide money by a divisor.

        Args:
            divisor: Division divisor (must be positive)

        Returns:
            Money: New Money instance

        Raises:
            ValueError: If divisor is zero or negative
            TypeError: If divisor cannot be converted to Decimal

        Examples:
            >>> money = Money("100.00", "USD")
            >>> money.divide(4)
            Money(amount=Decimal('25.00'), currency='USD')
        """
        decimal_divisor = self._to_decimal(divisor)
        if decimal_divisor <= 0:
            raise ValueError(
                f"Divisor must be positive: {decimal_divisor}. "
                f"Cannot divide by zero or negative numbers."
            )
        return Money(self._amount / decimal_divisor, self._currency)

    def __eq__(self, other: object) -> bool:
        """Check equality.

        Args:
            other: Object to compare with

        Returns:
            bool: True if both amount and currency match
        """
        if not isinstance(other, Money):
            return NotImplemented
        return self._amount == other._amount and self._currency == other._currency

    def __lt__(self, other: Money) -> bool:
        """Less than comparison.

        Args:
            other: Money instance to compare with

        Returns:
            bool: True if this amount is less than other

        Raises:
            ValueError: If currencies don't match
        """
        if not isinstance(other, Money):
            return NotImplemented
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare {self._currency} with {other._currency}. "
                f"Convert to same currency first."
            )
        return self._amount < other._amount

    def __le__(self, other: Money) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, Money):
            return NotImplemented
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare {self._currency} with {other._currency}"
            )
        return self._amount <= other._amount

    def __gt__(self, other: Money) -> bool:
        """Greater than comparison."""
        if not isinstance(other, Money):
            return NotImplemented
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare {self._currency} with {other._currency}"
            )
        return self._amount > other._amount

    def __ge__(self, other: Money) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, Money):
            return NotImplemented
        if self._currency != other._currency:
            raise ValueError(
                f"Cannot compare {self._currency} with {other._currency}"
            )
        return self._amount >= other._amount

    def __hash__(self) -> int:
        """Hash for use in sets/dicts.

        Immutable objects can be safely hashed.

        Returns:
            int: Hash value based on amount and currency
        """
        return hash((self._amount, self._currency))

    def __repr__(self) -> str:
        """Developer-friendly representation.

        Returns:
            str: Unambiguous representation for debugging
        """
        return f"Money(amount=Decimal('{self._amount}'), currency='{self._currency}')"

    def __str__(self) -> str:
        """User-friendly representation.

        Returns:
            str: Formatted money string (e.g., "USD 100.50")
        """
        return f"{self._currency} {self._amount:.2f}"


# ============================================================================
# CONTACT INFO VALUE OBJECT - Improved Email Validation
# ============================================================================

# Comprehensive email regex (RFC 5322 simplified)
EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
)

@dataclass(frozen=True)
class ContactInfo:
    """Value object for contact information with proper email validation.

    Fixes BUG-006: Weak email validation.
    """
    email: str
    phone: str | None = None
    preferred_channel: str = "email"

    def __post_init__(self) -> None:
        """Validate contact information.

        Raises:
            ValueError: If email is invalid or fields violate constraints
        """
        # Email validation
        if not self.email:
            raise ValueError("Email cannot be empty")

        email_normalized = self.email.strip().lower()

        # Length check
        if len(email_normalized) > 254:
            raise ValueError(
                f"Email too long: {len(email_normalized)} chars (max 254)"
            )

        # Format validation using regex
        if not EMAIL_REGEX.match(email_normalized):
            raise ValueError(
                f"Invalid email format: '{self.email}'. "
                f"Must be in format: user@domain.tld"
            )

        # Local part validation (before @)
        local_part, domain = email_normalized.rsplit('@', 1)
        if len(local_part) > 64:
            raise ValueError(
                f"Email local part too long: {len(local_part)} chars (max 64)"
            )

        # Domain validation (basic)
        if domain.startswith('-') or domain.endswith('-'):
            raise ValueError(f"Invalid domain: '{domain}' (cannot start/end with hyphen)")

        # Check for test/example domains in production
        forbidden_domains = {'example.com', 'test.com', 'localhost', 'example.org'}
        if domain in forbidden_domains:
            raise ValueError(
                f"Invalid domain: '{domain}' is a test/example domain"
            )

        # Phone validation (if provided)
        if self.phone:
            # Basic phone validation (can be enhanced)
            phone_clean = ''.join(c for c in self.phone if c.isdigit() or c in '+- ()')
            if len(phone_clean) < 10:
                raise ValueError(
                    f"Phone number too short: '{self.phone}' (min 10 digits)"
                )

        # Preferred channel validation
        valid_channels = {'email', 'phone', 'sms'}
        if self.preferred_channel not in valid_channels:
            raise ValueError(
                f"Invalid preferred_channel: '{self.preferred_channel}'. "
                f"Must be one of: {valid_channels}"
            )

    def is_valid_for_notification(self, channel: str) -> bool:
        """Check if contact info supports the given notification channel.

        Args:
            channel: Notification channel ('email', 'phone', 'sms')

        Returns:
            bool: True if channel is supported
        """
        if channel == "email":
            return bool(self.email)
        elif channel in ["phone", "sms"]:
            return bool(self.phone)
        return False


# ============================================================================
# OTHER VALUE OBJECTS
# ============================================================================

@dataclass(frozen=True)
class TimeRange:
    """Value object for time range."""
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate time range."""
        if self.end < self.start:
            raise ValueError(
                f"End time ({self.end}) must be after start time ({self.start})"
            )

    def duration_seconds(self) -> float:
        """Calculate duration in seconds.

        Returns:
            float: Duration in seconds
        """
        return (self.end - self.start).total_seconds()

    def overlaps(self, other: TimeRange) -> bool:
        """Check if time ranges overlap.

        Args:
            other: Another TimeRange

        Returns:
            bool: True if ranges overlap
        """
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Confidence:
    """Value object for confidence scores."""
    score: float
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate confidence score."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got: {self.score}"
            )

    def is_high(self, threshold: float = 0.8) -> bool:
        """Check if confidence is high.

        Args:
            threshold: Minimum score to be considered high (default: 0.8)

        Returns:
            bool: True if score >= threshold
        """
        return self.score >= threshold

    def is_low(self, threshold: float = 0.5) -> bool:
        """Check if confidence is low.

        Args:
            threshold: Maximum score to be considered low (default: 0.5)

        Returns:
            bool: True if score < threshold
        """
        return self.score < threshold
