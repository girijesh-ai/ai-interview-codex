"""
Integration Tests for Complete Workflow

Tests end-to-end workflow with all components.

Demonstrates:
- Integration testing
- Async testing
- Component integration
- Mock services
"""

import pytest
import asyncio
from datetime import datetime

from src.agents.state import StateBuilder
from src.agents.integration import (
    IntegratedAgentFactory,
    WorkflowIntegration
)
from src.domain.models import Priority, RequestCategory
from src.infrastructure.kafka.events import (
    EventType,
    RequestReceivedEvent,
    AgentStartedEvent,
    AgentCompletedEvent
)


# ============================================================================
# WORKFLOW INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    async def test_simple_request_workflow(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test simple request workflow end-to-end."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        # Create initial state
        state = (
            StateBuilder()
            .with_request_id("req-integration-1")
            .with_customer_id("cust-integration-1")
            .with_thread_id("thread-integration-1")
            .with_initial_message("How do I reset my password?")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.ACCOUNT)
            .build()
        )

        # Act - Execute agents
        triage_agent = factory.create_triage_agent()
        state = await triage_agent(state)

        research_agent = factory.create_research_agent()
        state = await research_agent(state)

        solution_agent = factory.create_solution_agent()
        state = await solution_agent(state)

        quality_agent = factory.create_quality_agent()
        state = await quality_agent(state)

        # Assert state
        assert state["category"] == "account"
        assert state["priority"] == Priority.MEDIUM.value
        assert len(state["decisions"]) >= 4
        assert state.get("proposed_solution") is not None
        assert state.get("quality_passed") is not None

        # Assert events published
        assert len(mock_event_producer.published_events) > 0

        # Verify event types
        event_types = [e.event_type for e in mock_event_producer.published_events]
        assert EventType.AGENT_STARTED in event_types
        assert EventType.AGENT_COMPLETED in event_types

    async def test_workflow_with_escalation(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test workflow that requires escalation."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        # Create high-priority request
        state = (
            StateBuilder()
            .with_request_id("req-escalation-1")
            .with_customer_id("cust-escalation-1")
            .with_thread_id("thread-escalation-1")
            .with_initial_message("I need a $1000 refund immediately!")
            .with_priority(Priority.CRITICAL)
            .with_category(RequestCategory.BILLING)
            .build()
        )

        # Act - Execute escalation path
        triage_agent = factory.create_triage_agent()
        state = await triage_agent(state)

        escalation_agent = factory.create_escalation_agent()
        state = await escalation_agent(state)

        # Assert
        assert state["requires_approval"] is True
        assert state.get("escalation_reason") is not None
        assert state["priority"] == Priority.CRITICAL.value

        # Verify escalation event
        escalation_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.REQUEST_ESCALATED
        ]
        assert len(escalation_events) > 0

    async def test_workflow_memory_integration(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test workflow memory integration."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state = (
            StateBuilder()
            .with_request_id("req-memory-1")
            .with_customer_id("cust-memory-1")
            .with_thread_id("thread-memory-1")
            .with_initial_message("Technical question")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.TECHNICAL)
            .build()
        )

        # Act - Execute research agent (uses memory)
        research_agent = factory.create_research_agent()
        state = await research_agent(state)

        # Assert - Context retrieved from memory
        assert len(state.get("retrieved_documents", [])) > 0

        # Verify context retrieved event
        context_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.CONTEXT_RETRIEVED
        ]
        assert len(context_events) > 0

    async def test_workflow_conversation_persistence(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test conversation persistence to vector DB."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-persist-1")
            .with_customer_id("cust-persist-1")
            .with_thread_id("thread-persist-1")
            .with_initial_message("Test persistence")
            .build()
        )

        # Act - Persist conversation
        success = await integration.persist_conversation(state)

        # Assert
        assert success is True

        # Verify conversation stored event
        stored_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.CONVERSATION_STORED
        ]
        assert len(stored_events) > 0

    async def test_workflow_state_optimization(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test state memory optimization."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        # Create state with many messages
        state = StateBuilder().build()
        state["request_id"] = "req-optimize-1"

        # Add 30 messages (exceeds limit of 20)
        from langchain_core.messages import HumanMessage
        for i in range(30):
            state["messages"].append(HumanMessage(content=f"Message {i}"))

        # Act - Optimize memory
        optimized_state = await integration.optimize_state_memory(state)

        # Assert - Should keep only 20 messages
        assert len(optimized_state["messages"]) == 20


# ============================================================================
# AGENT INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for individual agents."""

    async def test_triage_agent_integration(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test triage agent with memory and events."""
        # Arrange
        from src.agents.integration import IntegratedTriageAgent

        agent = IntegratedTriageAgent(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-triage-1")
            .with_customer_id("cust-triage-1")
            .with_thread_id("thread-triage-1")
            .with_initial_message("Account issue")
            .build()
        )

        # Act
        result_state = await agent(state)

        # Assert
        assert result_state["category"] is not None
        assert result_state["priority"] is not None
        assert len(result_state["decisions"]) > 0

        # Verify events
        assert len(mock_event_producer.published_events) >= 2  # Started + Completed

    async def test_research_agent_vector_db_integration(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test research agent vector DB integration."""
        # Arrange
        from src.agents.integration import IntegratedResearchAgent

        agent = IntegratedResearchAgent(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-research-1")
            .with_customer_id("cust-research-1")
            .with_thread_id("thread-research-1")
            .with_initial_message("How does feature X work?")
            .build()
        )

        # Act
        result_state = await agent(state)

        # Assert - Documents retrieved from vector DB
        assert len(result_state.get("retrieved_documents", [])) > 0

        # Verify documents retrieved event
        doc_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.DOCUMENTS_RETRIEVED
        ]
        assert len(doc_events) > 0

    async def test_solution_agent_context_integration(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test solution agent with context."""
        # Arrange
        from src.agents.integration import IntegratedSolutionAgent

        agent = IntegratedSolutionAgent(
            mock_memory_manager,
            mock_event_producer
        )

        state = (
            StateBuilder()
            .with_request_id("req-solution-1")
            .with_customer_id("cust-solution-1")
            .with_thread_id("thread-solution-1")
            .with_initial_message("Need solution")
            .build()
        )

        # Act
        result_state = await agent(state)

        # Assert
        assert result_state.get("proposed_solution") is not None
        assert result_state.get("solution_confidence") is not None

        # Verify solution generated event
        solution_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.SOLUTION_GENERATED
        ]
        assert len(solution_events) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Integration tests for error handling."""

    async def test_agent_error_handling(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test agent error handling and event publishing."""
        # Arrange
        from src.agents.integration import IntegratedAgent
        from src.domain.models import AgentType

        class FailingAgent(IntegratedAgent):
            def __init__(self, memory, events):
                super().__init__(AgentType.TRIAGE, memory, events)

            async def execute(self, state):
                raise Exception("Test error")

        agent = FailingAgent(mock_memory_manager, mock_event_producer)

        state = StateBuilder().build()

        # Act
        result_state = await agent(state)

        # Assert - Error recorded
        assert result_state.get("last_error") is not None
        assert "Test error" in result_state["last_error"]

        # Verify error event published
        error_events = [
            e for e in mock_event_producer.published_events
            if e.event_type == EventType.SYSTEM_ERROR
        ]
        assert len(error_events) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
class TestPerformance:
    """Performance integration tests."""

    async def test_concurrent_requests(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test handling multiple concurrent requests."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        # Create multiple requests
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

        # Act - Process concurrently
        agent = factory.create_triage_agent()

        results = await asyncio.gather(
            *[agent(state) for state in requests]
        )

        # Assert
        assert len(results) == 10
        for result in results:
            assert result["category"] is not None

    async def test_workflow_execution_time(
        self,
        mock_memory_manager,
        mock_event_producer
    ):
        """Test workflow execution time tracking."""
        # Arrange
        integration = WorkflowIntegration(
            mock_memory_manager,
            mock_event_producer
        )

        factory = integration.agent_factory

        state = (
            StateBuilder()
            .with_request_id("req-timing-1")
            .with_customer_id("cust-timing-1")
            .with_thread_id("thread-timing-1")
            .with_initial_message("Test timing")
            .build()
        )

        # Act
        start_time = datetime.now()

        agent = factory.create_triage_agent()
        result_state = await agent(state)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Assert - Should complete quickly with mocks
        assert duration < 1.0  # Less than 1 second

        # Verify metrics recorded
        assert len(result_state.get("metrics", {})) > 0
