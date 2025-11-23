"""
End-to-End System Tests

Tests complete system with all components.

Demonstrates:
- E2E testing strategy
- Real infrastructure integration
- Performance testing
- System validation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from src.agents.state import StateBuilder
from src.agents.integration import WorkflowIntegration
from src.domain.models import Priority, RequestCategory
from src.infrastructure.kafka.events import EventType


# ============================================================================
# E2E TEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="module")
def e2e_config():
    """E2E test configuration."""
    return {
        "use_real_services": False,  # Set to True for real infrastructure
        "timeout": 30,
        "max_retries": 3
    }


# ============================================================================
# COMPLETE WORKFLOW TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteWorkflow:
    """End-to-end tests for complete workflow."""

    async def test_simple_account_request_e2e(
        self,
        mock_memory_manager,
        mock_event_producer,
        e2e_config
    ):
        """Test simple account request end-to-end."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-e2e-account-1")
            .with_customer_id("cust-e2e-1")
            .with_thread_id("thread-e2e-1")
            .with_initial_message("How do I reset my password?")
            .with_priority(Priority.MEDIUM)
            .build()
        )

        # Act - Execute complete workflow
        factory = integration.agent_factory

        # Step 1: Triage
        triage = factory.create_triage_agent()
        state = await triage(state)

        # Step 2: Research
        research = factory.create_research_agent()
        state = await research(state)

        # Step 3: Generate solution
        solution = factory.create_solution_agent()
        state = await solution(state)

        # Step 4: Quality check
        quality = factory.create_quality_agent()
        state = await quality(state)

        # Step 5: Persist conversation
        success = await integration.persist_conversation(state)

        # Assert
        assert state["category"] == "account"
        assert state.get("proposed_solution") is not None
        assert state.get("quality_passed") is not None
        assert success is True

        # Verify all events published
        event_types = [e.event_type for e in mock_event_producer.published_events]
        assert EventType.AGENT_STARTED in event_types
        assert EventType.AGENT_COMPLETED in event_types
        assert EventType.CONVERSATION_STORED in event_types

    async def test_technical_request_with_research_e2e(
        self,
        mock_memory_manager,
        mock_event_producer,
        e2e_config
    ):
        """Test technical request requiring extensive research."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-e2e-technical-1")
            .with_customer_id("cust-e2e-2")
            .with_thread_id("thread-e2e-2")
            .with_initial_message("How does the API rate limiting work?")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.TECHNICAL)
            .build()
        )

        # Act
        factory = integration.agent_factory

        triage = factory.create_triage_agent()
        state = await triage(state)

        research = factory.create_research_agent()
        state = await research(state)

        solution = factory.create_solution_agent()
        state = await solution(state)

        # Assert
        assert state["category"] == "technical"
        assert len(state.get("retrieved_documents", [])) > 0
        assert state.get("proposed_solution") is not None

        # Verify research events
        doc_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.DOCUMENTS_RETRIEVED
        ]
        assert len(doc_events) > 0

    async def test_billing_request_requiring_approval_e2e(
        self,
        mock_memory_manager,
        mock_event_producer,
        e2e_config
    ):
        """Test billing request requiring approval."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-e2e-billing-1")
            .with_customer_id("cust-e2e-3")
            .with_thread_id("thread-e2e-3")
            .with_initial_message("I need a refund of $500")
            .with_priority(Priority.HIGH)
            .with_category(RequestCategory.BILLING)
            .build()
        )

        # Act
        factory = integration.agent_factory

        triage = factory.create_triage_agent()
        state = await triage(state)

        research = factory.create_research_agent()
        state = await research(state)

        solution = factory.create_solution_agent()
        state = await solution(state)

        quality = factory.create_quality_agent()
        state = await quality(state)

        # Check if escalation needed
        if state.get("requires_approval"):
            escalation = factory.create_escalation_agent()
            state = await escalation(state)

        # Assert
        assert state["category"] == "billing"
        assert state.get("requires_approval") is True
        assert state.get("escalation_reason") is not None

        # Verify escalation event
        escalation_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.REQUEST_ESCALATED
        ]
        assert len(escalation_events) > 0

    async def test_critical_request_immediate_escalation_e2e(
        self,
        mock_memory_manager,
        mock_event_producer,
        e2e_config
    ):
        """Test critical request with immediate escalation."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-e2e-critical-1")
            .with_customer_id("cust-e2e-4")
            .with_thread_id("thread-e2e-4")
            .with_initial_message("System is completely down!")
            .with_priority(Priority.CRITICAL)
            .build()
        )

        # Act
        factory = integration.agent_factory

        triage = factory.create_triage_agent()
        state = await triage(state)

        escalation = factory.create_escalation_agent()
        state = await escalation(state)

        # Assert
        assert state["priority"] == Priority.CRITICAL.value
        assert state.get("requires_approval") is True


# ============================================================================
# MULTI-TURN CONVERSATION TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
class TestMultiTurnConversations:
    """Tests for multi-turn conversations."""

    async def test_conversation_with_clarifications_e2e(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test conversation requiring clarifications."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        # Initial state
        state = (
            StateBuilder()
            .with_request_id("req-e2e-multiturn-1")
            .with_customer_id("cust-e2e-5")
            .with_thread_id("thread-e2e-5")
            .with_initial_message("I have a problem")
            .build()
        )

        factory = integration.agent_factory

        # Turn 1: Initial triage
        triage = factory.create_triage_agent()
        state = await triage(state)

        # Turn 2: Customer provides more details
        from langchain_core.messages import HumanMessage
        state["messages"].append(
            HumanMessage(content="It's about my billing")
        )

        triage = factory.create_triage_agent()
        state = await triage(state)

        # Turn 3: Research and solution
        research = factory.create_research_agent()
        state = await research(state)

        solution = factory.create_solution_agent()
        state = await solution(state)

        # Assert
        assert len(state["messages"]) >= 3
        assert state.get("proposed_solution") is not None


# ============================================================================
# CONCURRENT REQUEST TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
class TestConcurrentRequests:
    """Tests for concurrent request processing."""

    async def test_process_multiple_requests_concurrently(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test processing multiple requests concurrently."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        # Create 10 different requests
        requests = []
        for i in range(10):
            state = (
                StateBuilder()
                .with_request_id(f"req-concurrent-{i}")
                .with_customer_id(f"cust-{i}")
                .with_thread_id(f"thread-{i}")
                .with_initial_message(f"Request {i}")
                .build()
            )
            requests.append(state)

        # Act - Process all concurrently
        triage = factory.create_triage_agent()

        results = await asyncio.gather(
            *[triage(state) for state in requests]
        )

        # Assert
        assert len(results) == 10
        for result in results:
            assert result["category"] is not None
            assert len(result["decisions"]) > 0

    async def test_concurrent_requests_isolated_state(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test that concurrent requests maintain isolated state."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state1 = (
            StateBuilder()
            .with_request_id("req-isolated-1")
            .with_customer_id("cust-1")
            .with_thread_id("thread-1")
            .with_initial_message("Account help")
            .build()
        )

        state2 = (
            StateBuilder()
            .with_request_id("req-isolated-2")
            .with_customer_id("cust-2")
            .with_thread_id("thread-2")
            .with_initial_message("Technical help")
            .build()
        )

        # Act
        triage = factory.create_triage_agent()

        result1, result2 = await asyncio.gather(
            triage(state1),
            triage(state2)
        )

        # Assert - States should be independent
        assert result1["request_id"] != result2["request_id"]
        assert result1["customer_id"] != result2["customer_id"]


# ============================================================================
# MEMORY INTEGRATION TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
class TestMemoryIntegration:
    """Tests for memory system integration."""

    async def test_conversation_persistence_and_retrieval(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test persisting and retrieving conversations."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-memory-1")
            .with_customer_id("cust-memory-1")
            .with_thread_id("thread-memory-1")
            .with_initial_message("Test persistence")
            .build()
        )

        # Act - Persist conversation
        success = await integration.persist_conversation(state)
        assert success is True

        # Verify event published
        stored_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.CONVERSATION_STORED
        ]
        assert len(stored_events) > 0

    async def test_context_retrieval_for_similar_requests(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test retrieving context for similar requests."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state = (
            StateBuilder()
            .with_request_id("req-context-1")
            .with_customer_id("cust-context-1")
            .with_thread_id("thread-context-1")
            .with_initial_message("Password reset help")
            .build()
        )

        # Act
        research = factory.create_research_agent()
        state = await research(state)

        # Assert - Should retrieve relevant documents
        assert len(state.get("retrieved_documents", [])) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
class TestPerformance:
    """Performance tests."""

    async def test_workflow_execution_time(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test workflow execution completes within time limit."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state = (
            StateBuilder()
            .with_request_id("req-perf-1")
            .with_customer_id("cust-perf-1")
            .with_thread_id("thread-perf-1")
            .with_initial_message("Performance test")
            .build()
        )

        # Act
        start_time = datetime.now()

        triage = factory.create_triage_agent()
        state = await triage(state)

        research = factory.create_research_agent()
        state = await research(state)

        solution = factory.create_solution_agent()
        state = await solution(state)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Assert - Should complete in reasonable time with mocks
        assert duration < 2.0  # 2 seconds

    async def test_high_load_handling(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test system handles high load."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory
        triage = factory.create_triage_agent()

        # Create 50 requests
        states = [
            StateBuilder()
            .with_request_id(f"req-load-{i}")
            .with_customer_id(f"cust-{i}")
            .with_thread_id(f"thread-{i}")
            .with_initial_message(f"Request {i}")
            .build()
            for i in range(50)
        ]

        # Act
        start_time = datetime.now()

        results = await asyncio.gather(
            *[triage(state) for state in states]
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Assert
        assert len(results) == 50
        assert all(r["category"] is not None for r in results)
        # Should handle 50 requests in reasonable time
        assert duration < 10.0  # 10 seconds


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
class TestErrorRecovery:
    """Tests for error handling and recovery."""

    async def test_agent_failure_recovery(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test system recovers from agent failure."""
        # Arrange
        from src.agents.integration import IntegratedAgent
        from src.domain.models import AgentType

        class FailingAgent(IntegratedAgent):
            def __init__(self, memory, events):
                super().__init__(AgentType.TRIAGE, memory, events)
                self.call_count = 0

            async def execute(self, state):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Simulated failure")
                return state

        agent = FailingAgent(mock_memory_manager, mock_event_producer)
        state = StateBuilder().build()

        # Act - First call fails
        result1 = await agent(state)
        assert result1.get("last_error") is not None

        # Second call succeeds
        result2 = await agent(result1)
        assert result2.get("last_error") is None

    async def test_partial_workflow_completion(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test handling partial workflow completion."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state = (
            StateBuilder()
            .with_request_id("req-partial-1")
            .with_customer_id("cust-partial-1")
            .with_thread_id("thread-partial-1")
            .with_initial_message("Test partial completion")
            .build()
        )

        # Act - Complete only part of workflow
        triage = factory.create_triage_agent()
        state = await triage(state)

        research = factory.create_research_agent()
        state = await research(state)

        # Assert - Should have valid partial state
        assert state["category"] is not None
        assert len(state["decisions"]) >= 2
        assert state.get("proposed_solution") is None  # Not yet generated
