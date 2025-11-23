"""
Unit Tests for State Management

Tests state builders, reducers, and utilities.

Demonstrates:
- Testing state management
- Testing custom reducers
- Testing builder pattern
"""

import pytest
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.state import (
    AgentState,
    StateBuilder,
    StateUtils,
    StateSnapshot,
    add_decision,
    merge_metrics,
    union_sets,
    merge_context
)
from src.domain.models import (
    Priority,
    RequestCategory,
    RequestStatus,
    AgentType,
    DecisionType
)


# ============================================================================
# STATE BUILDER TESTS
# ============================================================================

class TestStateBuilder:
    """Tests for StateBuilder."""

    def test_build_basic_state(self):
        """Test building basic state."""
        # Arrange & Act
        state = (
            StateBuilder()
            .with_request_id("req-123")
            .with_customer_id("cust-456")
            .with_thread_id("thread-789")
            .build()
        )

        # Assert
        assert state["request_id"] == "req-123"
        assert state["customer_id"] == "cust-456"
        assert state["thread_id"] == "thread-789"
        assert state["status"] == RequestStatus.PENDING.value

    def test_build_with_message(self):
        """Test building state with message."""
        # Arrange & Act
        state = (
            StateBuilder()
            .with_request_id("req-123")
            .with_customer_id("cust-456")
            .with_thread_id("thread-789")
            .with_initial_message("Help me!")
            .build()
        )

        # Assert
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Help me!"

    def test_build_with_priority_and_category(self):
        """Test building state with priority and category."""
        # Arrange & Act
        state = (
            StateBuilder()
            .with_request_id("req-123")
            .with_customer_id("cust-456")
            .with_thread_id("thread-789")
            .with_priority(Priority.HIGH)
            .with_category(RequestCategory.ACCOUNT)
            .build()
        )

        # Assert
        assert state["priority"] == Priority.HIGH.value
        assert state["category"] == RequestCategory.ACCOUNT.value

    def test_builder_fluent_interface(self):
        """Test builder fluent interface."""
        # Arrange & Act
        state = (
            StateBuilder()
            .with_request_id("req-1")
            .with_customer_id("cust-1")
            .with_thread_id("thread-1")
            .with_initial_message("Test")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.TECHNICAL)
            .with_session_data({"key": "value"})
            .build()
        )

        # Assert
        assert state["request_id"] == "req-1"
        assert state["session_data"]["key"] == "value"


# ============================================================================
# CUSTOM REDUCER TESTS
# ============================================================================

class TestCustomReducers:
    """Tests for custom reducers."""

    def test_add_decision_reducer(self):
        """Test add_decision reducer."""
        # Arrange
        existing = [
            {
                "decision_id": "dec-1",
                "agent_type": "triage",
                "timestamp": datetime.now().isoformat()
            }
        ]
        new = [
            {
                "decision_id": "dec-2",
                "agent_type": "research",
                "timestamp": datetime.now().isoformat()
            }
        ]

        # Act
        result = add_decision(existing, new)

        # Assert
        assert len(result) == 2
        assert result[0]["decision_id"] == "dec-1"
        assert result[1]["decision_id"] == "dec-2"

    def test_add_decision_deduplication(self):
        """Test add_decision deduplicates."""
        # Arrange
        existing = [
            {"decision_id": "dec-1", "agent_type": "triage"}
        ]
        new = [
            {"decision_id": "dec-1", "agent_type": "triage"},  # Duplicate
            {"decision_id": "dec-2", "agent_type": "research"}
        ]

        # Act
        result = add_decision(existing, new)

        # Assert
        assert len(result) == 2  # Only unique decisions

    def test_merge_metrics_count(self):
        """Test merge_metrics for counts."""
        # Arrange
        existing = {"triage_count": 5}
        new = {"triage_count": 3}

        # Act
        result = merge_metrics(existing, new)

        # Assert
        assert result["triage_count"] == 8  # Sum of counts

    def test_merge_metrics_average(self):
        """Test merge_metrics for averages."""
        # Arrange
        existing = {
            "duration_avg": 100.0,
            "duration_count": 5
        }
        new = {
            "duration_avg": 200.0,
            "duration_count": 3
        }

        # Act
        result = merge_metrics(existing, new)

        # Assert
        # Weighted average: (100*5 + 200*3) / (5+3) = 137.5
        assert result["duration_avg"] == pytest.approx(137.5)
        assert result["duration_count"] == 8

    def test_merge_metrics_max(self):
        """Test merge_metrics for max values."""
        # Arrange
        existing = {"response_time_max": 150.0}
        new = {"response_time_max": 200.0}

        # Act
        result = merge_metrics(existing, new)

        # Assert
        assert result["response_time_max"] == 200.0

    def test_union_sets_reducer(self):
        """Test union_sets reducer."""
        # Arrange
        existing = {"id-1", "id-2"}
        new = {"id-2", "id-3", "id-4"}

        # Act
        result = union_sets(existing, new)

        # Assert
        assert result == {"id-1", "id-2", "id-3", "id-4"}

    def test_merge_context_reducer(self):
        """Test merge_context reducer."""
        # Arrange
        existing = {
            "user_data": {"name": "John"},
            "preferences": {"theme": "dark"}
        }
        new = {
            "user_data": {"email": "john@example.com"},
            "settings": {"notifications": True}
        }

        # Act
        result = merge_context(existing, new)

        # Assert
        assert result["user_data"]["name"] == "John"
        assert result["user_data"]["email"] == "john@example.com"
        assert result["preferences"]["theme"] == "dark"
        assert result["settings"]["notifications"] is True


# ============================================================================
# STATE UTILS TESTS
# ============================================================================

class TestStateUtils:
    """Tests for StateUtils."""

    def test_get_last_user_message(self, sample_messages):
        """Test getting last user message."""
        # Arrange
        state = AgentState(
            request_id="req-1",
            customer_id="cust-1",
            thread_id="thread-1",
            messages=sample_messages,
            decisions=[],
            relevant_context_ids=set(),
            metrics={},
            session_data={},
            status="pending",
            priority=2
        )

        # Act
        last_message = StateUtils.get_last_user_message(state)

        # Assert
        assert last_message["role"] == "HumanMessage"
        assert "password" in last_message["content"]

    def test_record_decision(self, empty_agent_state):
        """Test recording decision."""
        # Arrange
        state = empty_agent_state

        # Act
        StateUtils.record_decision(
            state,
            AgentType.TRIAGE,
            DecisionType.CLASSIFY,
            0.85,
            "Classified as account"
        )

        # Assert
        assert len(state["decisions"]) == 1
        assert state["decisions"][0]["agent_type"] == AgentType.TRIAGE.value

    def test_update_metrics(self, empty_agent_state):
        """Test updating metrics."""
        # Arrange
        state = empty_agent_state

        # Act
        StateUtils.update_metrics(state, {
            "triage_duration_avg": 450.5,
            "triage_count": 1
        })

        # Assert
        assert state["metrics"]["triage_duration_avg"] == 450.5
        assert state["metrics"]["triage_count"] == 1

    def test_get_message_history(self, sample_messages):
        """Test getting message history."""
        # Arrange
        state = AgentState(
            request_id="req-1",
            customer_id="cust-1",
            thread_id="thread-1",
            messages=sample_messages,
            decisions=[],
            relevant_context_ids=set(),
            metrics={},
            session_data={},
            status="pending",
            priority=2
        )

        # Act
        history = StateUtils.get_message_history(state, limit=2)

        # Assert
        assert len(history) <= 2

    def test_get_execution_time(self):
        """Test getting execution time."""
        # Arrange
        state = StateBuilder().build()
        state["created_at"] = datetime.now().timestamp()

        # Act
        exec_time = StateUtils.get_execution_time(state)

        # Assert
        assert exec_time >= 0


# ============================================================================
# STATE SNAPSHOT TESTS
# ============================================================================

class TestStateSnapshot:
    """Tests for StateSnapshot."""

    def test_create_snapshot(self, sample_agent_state):
        """Test creating snapshot."""
        # Arrange & Act
        snapshot = StateSnapshot.from_state(sample_agent_state)

        # Assert
        assert snapshot.request_id == sample_agent_state["request_id"]
        assert snapshot.customer_id == sample_agent_state["customer_id"]
        assert snapshot.status == sample_agent_state["status"]

    def test_snapshot_serialization(self, sample_agent_state):
        """Test snapshot serialization."""
        # Arrange
        snapshot = StateSnapshot.from_state(sample_agent_state)

        # Act
        serialized = snapshot.to_dict()

        # Assert
        assert isinstance(serialized, dict)
        assert serialized["request_id"] == sample_agent_state["request_id"]

    def test_snapshot_restoration(self, sample_agent_state):
        """Test restoring from snapshot."""
        # Arrange
        snapshot = StateSnapshot.from_state(sample_agent_state)
        serialized = snapshot.to_dict()

        # Act
        restored = StateSnapshot.from_dict(serialized)

        # Assert
        assert restored.request_id == snapshot.request_id
        assert restored.status == snapshot.status


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestStateIntegration:
    """Integration tests for state management."""

    def test_full_state_lifecycle(self):
        """Test complete state lifecycle."""
        # Arrange - Create initial state
        state = (
            StateBuilder()
            .with_request_id("req-lifecycle")
            .with_customer_id("cust-lifecycle")
            .with_thread_id("thread-lifecycle")
            .with_initial_message("Test message")
            .with_priority(Priority.MEDIUM)
            .build()
        )

        # Act - Add decisions
        StateUtils.record_decision(
            state, AgentType.TRIAGE,
            DecisionType.CLASSIFY, 0.85, "Triaged"
        )

        # Act - Update metrics
        StateUtils.update_metrics(state, {
            "triage_duration_avg": 450.0,
            "triage_count": 1
        })

        # Act - Create snapshot
        snapshot = StateSnapshot.from_state(state)

        # Assert
        assert len(state["decisions"]) == 1
        assert state["metrics"]["triage_count"] == 1
        assert snapshot.request_id == "req-lifecycle"
