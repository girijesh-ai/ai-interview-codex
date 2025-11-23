"""
Unit Tests for Domain Models

Tests value objects, entities, and aggregate roots.

Demonstrates:
- Unit testing best practices
- AAA pattern (Arrange, Act, Assert)
- Parametrized tests
- Exception testing
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.domain.models import (
    Money,
    ContactInfo,
    TimeRange,
    Confidence,
    Customer,
    AgentDecision,
    Message,
    CustomerRequest,
    Priority,
    RequestCategory,
    RequestStatus,
    AgentType,
    DecisionType
)


# ============================================================================
# VALUE OBJECT TESTS
# ============================================================================

class TestMoney:
    """Tests for Money value object."""

    def test_create_money(self):
        """Test creating money instance."""
        # Arrange & Act
        money = Money(amount=100.50, currency="USD")

        # Assert
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_money_equality(self):
        """Test money equality."""
        # Arrange
        money1 = Money(amount=100.50, currency="USD")
        money2 = Money(amount=100.50, currency="USD")
        money3 = Money(amount=200.00, currency="USD")

        # Act & Assert
        assert money1 == money2
        assert money1 != money3

    def test_money_immutability(self):
        """Test money is immutable."""
        # Arrange
        money = Money(amount=100.50, currency="USD")

        # Act & Assert
        with pytest.raises(AttributeError):
            money.amount = Decimal("200.00")

    def test_money_addition(self):
        """Test adding money."""
        # Arrange
        money1 = Money(amount=100.50, currency="USD")
        money2 = Money(amount=50.25, currency="USD")

        # Act
        result = money1 + money2

        # Assert
        assert result.amount == Decimal("150.75")
        assert result.currency == "USD"

    def test_money_addition_different_currency_fails(self):
        """Test adding money with different currencies fails."""
        # Arrange
        money1 = Money(amount=100.50, currency="USD")
        money2 = Money(amount=50.25, currency="EUR")

        # Act & Assert
        with pytest.raises(ValueError, match="Currency mismatch"):
            _ = money1 + money2

    def test_money_subtraction(self):
        """Test subtracting money."""
        # Arrange
        money1 = Money(amount=100.50, currency="USD")
        money2 = Money(amount=30.25, currency="USD")

        # Act
        result = money1 - money2

        # Assert
        assert result.amount == Decimal("70.25")

    def test_money_negative_amount_fails(self):
        """Test negative amount fails."""
        # Act & Assert
        with pytest.raises(ValueError, match="Amount must be non-negative"):
            Money(amount=-10.00, currency="USD")


class TestContactInfo:
    """Tests for ContactInfo value object."""

    def test_create_contact_info(self):
        """Test creating contact info."""
        # Arrange & Act
        contact = ContactInfo(
            email="test@example.com",
            phone="+1234567890"
        )

        # Assert
        assert contact.email == "test@example.com"
        assert contact.phone == "+1234567890"

    def test_invalid_email_fails(self):
        """Test invalid email fails."""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid email"):
            ContactInfo(email="invalid-email", phone="+1234567890")

    @pytest.mark.parametrize("email", [
        "test@example.com",
        "user.name@domain.co.uk",
        "test+tag@example.com"
    ])
    def test_valid_emails(self, email):
        """Test various valid emails."""
        # Act
        contact = ContactInfo(email=email, phone="+1234567890")

        # Assert
        assert contact.email == email


class TestConfidence:
    """Tests for Confidence value object."""

    def test_create_confidence(self):
        """Test creating confidence."""
        # Arrange & Act
        confidence = Confidence(score=0.85, explanation="High confidence")

        # Assert
        assert confidence.score == 0.85
        assert confidence.explanation == "High confidence"

    def test_confidence_out_of_range_fails(self):
        """Test confidence out of range fails."""
        # Act & Assert
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            Confidence(score=1.5, explanation="Invalid")

    def test_confidence_is_high(self):
        """Test is_high method."""
        # Arrange
        high_conf = Confidence(score=0.9, explanation="High")
        low_conf = Confidence(score=0.5, explanation="Low")

        # Act & Assert
        assert high_conf.is_high(threshold=0.8)
        assert not low_conf.is_high(threshold=0.8)


# ============================================================================
# ENTITY TESTS
# ============================================================================

class TestCustomer:
    """Tests for Customer entity."""

    def test_create_customer(self, sample_contact_info):
        """Test creating customer."""
        # Arrange & Act
        customer = Customer(
            customer_id="cust-123",
            name="John Doe",
            contact_info=sample_contact_info,
            tier="premium"
        )

        # Assert
        assert customer.customer_id == "cust-123"
        assert customer.name == "John Doe"
        assert customer.tier == "premium"

    def test_customer_identity(self, sample_contact_info):
        """Test customer identity."""
        # Arrange
        customer1 = Customer("cust-123", "John", sample_contact_info)
        customer2 = Customer("cust-123", "Jane", sample_contact_info)
        customer3 = Customer("cust-456", "John", sample_contact_info)

        # Act & Assert
        assert customer1 == customer2  # Same ID
        assert customer1 != customer3  # Different ID

    def test_customer_is_premium(self, sample_contact_info):
        """Test is_premium method."""
        # Arrange
        premium = Customer("cust-1", "John", sample_contact_info, tier="premium")
        standard = Customer("cust-2", "Jane", sample_contact_info, tier="standard")

        # Act & Assert
        assert premium.is_premium()
        assert not standard.is_premium()


class TestAgentDecision:
    """Tests for AgentDecision entity."""

    def test_create_decision(self):
        """Test creating agent decision."""
        # Arrange
        confidence = Confidence(0.85, "High confidence")

        # Act
        decision = AgentDecision(
            decision_id="dec-123",
            agent_type=AgentType.TRIAGE,
            decision_type=DecisionType.CLASSIFY,
            confidence=confidence,
            reasoning="Customer mentioned account issue"
        )

        # Assert
        assert decision.agent_type == AgentType.TRIAGE
        assert decision.confidence.score == 0.85
        assert "account" in decision.reasoning.lower()

    def test_decision_requires_human_review(self):
        """Test requires_human_review method."""
        # Arrange
        low_conf = Confidence(0.5, "Low confidence")
        high_conf = Confidence(0.95, "High confidence")

        low_decision = AgentDecision(
            "dec-1", AgentType.SOLUTION, DecisionType.GENERATE,
            low_conf, "Uncertain"
        )
        high_decision = AgentDecision(
            "dec-2", AgentType.SOLUTION, DecisionType.GENERATE,
            high_conf, "Certain"
        )

        # Act & Assert
        assert low_decision.requires_human_review()
        assert not high_decision.requires_human_review()


# ============================================================================
# AGGREGATE ROOT TESTS
# ============================================================================

class TestCustomerRequest:
    """Tests for CustomerRequest aggregate root."""

    def test_create_customer_request(self, sample_customer):
        """Test creating customer request."""
        # Arrange & Act
        request = CustomerRequest(
            request_id="req-123",
            customer=sample_customer,
            initial_message="Need help with account"
        )

        # Assert
        assert request.id == "req-123"
        assert request.customer == sample_customer
        assert request.status == RequestStatus.PENDING
        assert len(request.messages) == 1

    def test_add_message(self, sample_customer_request):
        """Test adding message to request."""
        # Arrange
        initial_count = len(sample_customer_request.messages)

        # Act
        sample_customer_request.add_message(
            content="Additional information",
            sender="customer"
        )

        # Assert
        assert len(sample_customer_request.messages) == initial_count + 1

    def test_add_decision(self, sample_customer_request):
        """Test adding decision to request."""
        # Arrange
        confidence = Confidence(0.85, "High confidence")

        # Act
        decision = sample_customer_request.add_decision(
            agent_type=AgentType.TRIAGE,
            decision_type=DecisionType.CLASSIFY,
            confidence=confidence,
            reasoning="Classified as account"
        )

        # Assert
        assert len(sample_customer_request.decisions) == 1
        assert decision.agent_type == AgentType.TRIAGE

    def test_mark_as_completed(self, sample_customer_request):
        """Test marking request as completed."""
        # Arrange
        solution = "Your issue has been resolved"

        # Act
        sample_customer_request.mark_as_completed(solution)

        # Assert
        assert sample_customer_request.status == RequestStatus.COMPLETED
        assert sample_customer_request.solution == solution
        assert sample_customer_request.completed_at is not None

    def test_escalate(self, sample_customer_request):
        """Test escalating request."""
        # Arrange
        reason = "Complex issue requiring manager review"

        # Act
        sample_customer_request.escalate(reason)

        # Assert
        assert sample_customer_request.status == RequestStatus.ESCALATED
        assert sample_customer_request.escalation_reason == reason
        assert sample_customer_request.requires_approval is True

    def test_get_resolution_time(self, sample_customer_request):
        """Test getting resolution time."""
        # Arrange
        sample_customer_request.mark_as_completed("Done")

        # Act
        resolution_time = sample_customer_request.get_resolution_time()

        # Assert
        assert resolution_time is not None
        assert resolution_time >= timedelta(0)

    def test_is_high_priority(self, sample_customer):
        """Test is_high_priority method."""
        # Arrange
        high_request = CustomerRequest(
            "req-1", sample_customer, "Help",
            priority=Priority.CRITICAL
        )
        low_request = CustomerRequest(
            "req-2", sample_customer, "Help",
            priority=Priority.LOW
        )

        # Act & Assert
        assert high_request.is_high_priority()
        assert not low_request.is_high_priority()


# ============================================================================
# INVARIANT TESTS
# ============================================================================

class TestInvariants:
    """Tests for business rule invariants."""

    def test_cannot_complete_without_solution(self, sample_customer_request):
        """Test cannot complete without solution."""
        # Act & Assert
        with pytest.raises(ValueError, match="Solution cannot be empty"):
            sample_customer_request.mark_as_completed("")

    def test_cannot_escalate_without_reason(self, sample_customer_request):
        """Test cannot escalate without reason."""
        # Act & Assert
        with pytest.raises(ValueError, match="Escalation reason required"):
            sample_customer_request.escalate("")

    def test_low_confidence_triggers_approval(self, sample_customer_request):
        """Test low confidence triggers approval requirement."""
        # Arrange
        low_confidence = Confidence(0.4, "Low confidence")

        # Act
        sample_customer_request.add_decision(
            AgentType.SOLUTION,
            DecisionType.GENERATE,
            low_confidence,
            "Uncertain solution"
        )

        # Assert
        assert sample_customer_request.requires_approval is True


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("amount,currency,expected", [
        (100.00, "USD", True),
        (0.01, "EUR", True),
        (1000000.00, "GBP", True)
    ])
    def test_money_creation_valid(self, amount, currency, expected):
        """Test various valid money amounts."""
        # Act
        money = Money(amount=amount, currency=currency)

        # Assert
        assert (money is not None) == expected

    @pytest.mark.parametrize("priority,expected_high", [
        (Priority.LOW, False),
        (Priority.MEDIUM, False),
        (Priority.HIGH, True),
        (Priority.CRITICAL, True)
    ])
    def test_priority_levels(self, priority, expected_high, sample_customer):
        """Test different priority levels."""
        # Arrange
        request = CustomerRequest(
            "req-1", sample_customer, "Help",
            priority=priority
        )

        # Act & Assert
        assert request.is_high_priority() == expected_high
