"""
Agent Integration Layer

Connects:
- Memory Manager (3-tier memory)
- Event System (Kafka)
- Agent Nodes

Demonstrates:
- Facade pattern for unified interface
- Decorator pattern for event publishing
- Observer pattern for memory updates
- Dependency injection
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime
import logging

from .state import AgentState, StateUtils
from .nodes import BaseAgent
from ..memory.manager import MemoryManager, RetrievalStrategy
from ..infrastructure.kafka.producer import EventProducer
from ..infrastructure.kafka.events import (
    EventFactory,
    EventType,
    AgentStartedEvent,
    AgentCompletedEvent,
    AgentDecisionEvent,
    RequestTriagedEvent,
    DocumentsRetrievedEvent,
    SolutionGeneratedEvent,
    QualityCheckPassedEvent,
    QualityCheckFailedEvent,
    RequestEscalatedEvent,
    ContextRetrievedEvent,
    ConversationStoredEvent,
    SystemErrorEvent
)
from ..domain.models import AgentType, DecisionType

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT DECORATOR FOR EVENT PUBLISHING
# ============================================================================

def publish_agent_events(event_producer: EventProducer):
    """Decorator to publish events for agent execution.

    Demonstrates:
    - Decorator pattern
    - Aspect-oriented programming
    - Separation of concerns

    Args:
        event_producer: Kafka event producer

    Returns:
        Decorated function
    """
    def decorator(agent_func: Callable):
        @wraps(agent_func)
        async def wrapper(agent: BaseAgent, state: AgentState) -> AgentState:
            request_id = state.get("request_id", "unknown")
            agent_type = agent.agent_type.value
            start_time = datetime.now()

            # Publish agent started event
            await event_producer.publish(
                AgentStartedEvent(
                    request_id=request_id,
                    agent_type=agent_type,
                    agent_id=f"{agent_type}-{request_id}",
                    input_data={
                        "priority": state.get("priority"),
                        "category": state.get("category"),
                        "status": state.get("status")
                    }
                )
            )

            try:
                # Execute agent
                result_state = await agent_func(agent, state)

                # Calculate duration
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Publish agent completed event
                await event_producer.publish(
                    AgentCompletedEvent(
                        request_id=request_id,
                        agent_type=agent_type,
                        agent_id=f"{agent_type}-{request_id}",
                        duration_ms=duration_ms,
                        output_data={
                            "status": result_state.get("status"),
                            "decisions_count": len(result_state.get("decisions", []))
                        }
                    )
                )

                return result_state

            except Exception as e:
                # Publish error event
                await event_producer.publish(
                    SystemErrorEvent(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component=f"agent.{agent_type}",
                        recovery_attempted=False
                    )
                )
                raise

        return wrapper
    return decorator


# ============================================================================
# INTEGRATED AGENT BASE CLASS
# ============================================================================

class IntegratedAgent(BaseAgent):
    """Enhanced agent with memory and event integration.

    Demonstrates:
    - Decorator pattern
    - Dependency injection
    - Template method pattern
    """

    def __init__(
        self,
        agent_type: AgentType,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        """Initialize integrated agent.

        Args:
            agent_type: Type of agent
            memory_manager: Memory manager instance
            event_producer: Event producer instance
        """
        super().__init__(agent_type)
        self.memory = memory_manager
        self.events = event_producer

    async def __call__(self, state: AgentState) -> AgentState:
        """Execute agent with memory and events.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        state["current_agent"] = self.agent_type.value
        start_time = datetime.now()
        request_id = state.get("request_id", "unknown")

        try:
            # Publish started event
            await self.events.publish(
                AgentStartedEvent(
                    request_id=request_id,
                    agent_type=self.agent_type.value,
                    agent_id=f"{self.agent_type.value}-{request_id}",
                    input_data={"status": state.get("status")}
                )
            )

            # Execute agent logic
            state = await self.execute(state)

            # Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            StateUtils.update_metrics(state, {
                f"{self.agent_type.value}_duration_avg": duration,
                f"{self.agent_type.value}_success_count": 1
            })

            # Publish completed event
            await self.events.publish(
                AgentCompletedEvent(
                    request_id=request_id,
                    agent_type=self.agent_type.value,
                    agent_id=f"{self.agent_type.value}-{request_id}",
                    duration_ms=duration * 1000,
                    output_data={"status": state.get("status")}
                )
            )

            return state

        except Exception as e:
            state["last_error"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1

            # Publish error event
            await self.events.publish(
                SystemErrorEvent(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    component=f"agent.{self.agent_type.value}"
                )
            )

            logger.error(f"Agent {self.agent_type.value} failed: {e}")
            return state


# ============================================================================
# INTEGRATED TRIAGE AGENT
# ============================================================================

class IntegratedTriageAgent(IntegratedAgent):
    """Triage agent with memory and event integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        super().__init__(AgentType.TRIAGE, memory_manager, event_producer)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute triage with context retrieval and event publishing.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        request_id = state.get("request_id")
        last_message = StateUtils.get_last_user_message(state)

        # Retrieve context from memory
        context = await self.memory.get_context(
            last_message.get("content", ""),
            state,
            strategy=RetrievalStrategy.RECENT_FIRST,
            max_items=5
        )

        # Publish context retrieved event
        await self.events.publish(
            ContextRetrievedEvent(
                request_id=request_id,
                retrieval_strategy=context.retrieval_strategy.value,
                sources=["state", "cache", "vector_db"],
                item_count=len(context.relevant_documents) + len(context.recent_interactions),
                duration_ms=0  # Would be tracked by timer
            )
        )

        # Perform triage (simplified - would use LLM)
        category = "account"  # Would be determined by LLM
        priority = 2

        state["category"] = category
        state["priority"] = priority

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ROUTE,
            0.85,
            f"Triaged as {category} with priority {priority}"
        )

        # Publish triaged event
        await self.events.publish(
            RequestTriagedEvent(
                request_id=request_id,
                category=category,
                priority=priority,
                urgency="medium",
                complexity="low"
            )
        )

        # Cache interaction
        await self.memory.cache_interaction(
            request_id,
            {
                "agent": self.agent_type.value,
                "action": "triage",
                "category": category,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
        )

        return state


# ============================================================================
# INTEGRATED RESEARCH AGENT
# ============================================================================

class IntegratedResearchAgent(IntegratedAgent):
    """Research agent with memory and event integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        super().__init__(AgentType.RESEARCH, memory_manager, event_producer)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute research with vector DB and event publishing.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        request_id = state.get("request_id")
        last_message = StateUtils.get_last_user_message(state)
        query = last_message.get("content", "")

        # Search knowledge base
        kb_results = await self.memory.search_knowledge_base(
            query,
            k=5
        )

        # Search similar conversations
        conv_results = await self.memory.search_conversations(
            query,
            k=3
        )

        # Update state
        state["retrieved_documents"] = [
            {
                "id": r.id,
                "content": r.content[:200],  # Truncate
                "score": r.score,
                "source": "knowledge_base"
            }
            for r in kb_results
        ]

        # Publish documents retrieved event
        await self.events.publish(
            DocumentsRetrievedEvent(
                request_id=request_id,
                query=query,
                document_count=len(kb_results) + len(conv_results),
                sources=["knowledge_base", "conversations"],
                relevance_scores=[r.score for r in kb_results]
            )
        )

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.RETRIEVE,
            0.9,
            f"Retrieved {len(kb_results)} KB articles and {len(conv_results)} similar conversations"
        )

        # Cache retrieved documents
        await self.memory.cache_interaction(
            request_id,
            {
                "agent": self.agent_type.value,
                "action": "research",
                "documents_found": len(kb_results),
                "timestamp": datetime.now().isoformat()
            }
        )

        return state


# ============================================================================
# INTEGRATED SOLUTION AGENT
# ============================================================================

class IntegratedSolutionAgent(IntegratedAgent):
    """Solution agent with memory and event integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        super().__init__(AgentType.SOLUTION, memory_manager, event_producer)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute solution generation with context.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        request_id = state.get("request_id")
        last_message = StateUtils.get_last_user_message(state)

        # Get full context
        context = await self.memory.get_context(
            last_message.get("content", ""),
            state,
            strategy=RetrievalStrategy.HYBRID,
            max_items=10
        )

        # Generate solution (simplified - would use LLM with context)
        solution = self._generate_solution(state, context)
        confidence = 0.88

        state["proposed_solution"] = solution
        state["solution_confidence"] = confidence

        # Publish solution generated event
        await self.events.publish(
            SolutionGeneratedEvent(
                request_id=request_id,
                solution_type="llm_generated",
                confidence=confidence,
                template_used=None,
                tokens_used=0  # Would track actual tokens
            )
        )

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.GENERATE,
            confidence,
            "Generated solution based on retrieved context"
        )

        # Cache solution
        await self.memory.cache_interaction(
            request_id,
            {
                "agent": self.agent_type.value,
                "action": "solution",
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        )

        return state

    def _generate_solution(self, state: AgentState, context) -> str:
        """Generate solution (placeholder for LLM call)."""
        category = state.get("category", "general")
        return f"Here's the solution for your {category} request..."


# ============================================================================
# INTEGRATED QUALITY AGENT
# ============================================================================

class IntegratedQualityAgent(IntegratedAgent):
    """Quality agent with memory and event integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        super().__init__(AgentType.QUALITY, memory_manager, event_producer)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute quality check with event publishing.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        request_id = state.get("request_id")
        solution = state.get("proposed_solution", "")
        confidence = state.get("solution_confidence", 0.0)

        # Perform quality checks
        checks_passed = []
        checks_failed = []

        # Check 1: Solution not empty
        if solution and len(solution) > 10:
            checks_passed.append("non_empty")
        else:
            checks_failed.append("non_empty")

        # Check 2: Confidence threshold
        if confidence >= 0.8:
            checks_passed.append("confidence_threshold")
        else:
            checks_failed.append("confidence_threshold")

        # Check 3: No placeholder text
        if "[INSERT" not in solution and "TODO" not in solution:
            checks_passed.append("no_placeholders")
        else:
            checks_failed.append("no_placeholders")

        # Determine pass/fail
        quality_passed = len(checks_failed) == 0
        quality_score = len(checks_passed) / (len(checks_passed) + len(checks_failed))

        state["quality_passed"] = quality_passed
        state["quality_score"] = quality_score

        # Publish quality check event
        if quality_passed:
            await self.events.publish(
                QualityCheckPassedEvent(
                    request_id=request_id,
                    checks_passed=checks_passed,
                    quality_score=quality_score,
                    reviewer=self.agent_type.value
                )
            )
        else:
            await self.events.publish(
                QualityCheckFailedEvent(
                    request_id=request_id,
                    checks_failed=checks_failed,
                    issues=[f"Failed: {c}" for c in checks_failed],
                    reviewer=self.agent_type.value
                )
            )

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.VALIDATE,
            quality_score,
            f"Quality check {'passed' if quality_passed else 'failed'}"
        )

        return state


# ============================================================================
# INTEGRATED ESCALATION AGENT
# ============================================================================

class IntegratedEscalationAgent(IntegratedAgent):
    """Escalation agent with memory and event integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        super().__init__(AgentType.ESCALATION, memory_manager, event_producer)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute escalation with event publishing.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        request_id = state.get("request_id")
        priority = state.get("priority", 2)
        last_error = state.get("last_error", "Unknown issue")

        # Determine escalation reason
        if priority >= 4:
            reason = "High priority request"
        elif state.get("retry_count", 0) > 2:
            reason = "Multiple retry failures"
        else:
            reason = last_error

        # Mark for approval
        state["requires_approval"] = True
        state["escalation_reason"] = reason

        # Publish escalation event
        await self.events.publish(
            RequestEscalatedEvent(
                request_id=request_id,
                escalation_reason=reason,
                escalation_level=1,
                escalated_to="human_review",
                escalated_by=self.agent_type.value
            )
        )

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ESCALATE,
            1.0,
            f"Escalated: {reason}"
        )

        # Store escalation in cache
        await self.memory.cache_interaction(
            request_id,
            {
                "agent": self.agent_type.value,
                "action": "escalation",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )

        return state


# ============================================================================
# AGENT FACTORY WITH INTEGRATION
# ============================================================================

class IntegratedAgentFactory:
    """Factory for creating integrated agents.

    Demonstrates:
    - Factory pattern
    - Dependency injection
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        """Initialize factory.

        Args:
            memory_manager: Memory manager instance
            event_producer: Event producer instance
        """
        self.memory = memory_manager
        self.events = event_producer

    def create_triage_agent(self) -> IntegratedTriageAgent:
        """Create triage agent."""
        return IntegratedTriageAgent(self.memory, self.events)

    def create_research_agent(self) -> IntegratedResearchAgent:
        """Create research agent."""
        return IntegratedResearchAgent(self.memory, self.events)

    def create_solution_agent(self) -> IntegratedSolutionAgent:
        """Create solution agent."""
        return IntegratedSolutionAgent(self.memory, self.events)

    def create_quality_agent(self) -> IntegratedQualityAgent:
        """Create quality agent."""
        return IntegratedQualityAgent(self.memory, self.events)

    def create_escalation_agent(self) -> IntegratedEscalationAgent:
        """Create escalation agent."""
        return IntegratedEscalationAgent(self.memory, self.events)


# ============================================================================
# WORKFLOW INTEGRATION FACADE
# ============================================================================

class WorkflowIntegration:
    """Facade for integrated workflow execution.

    Demonstrates:
    - Facade pattern
    - Orchestration
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_producer: EventProducer
    ):
        """Initialize workflow integration.

        Args:
            memory_manager: Memory manager
            event_producer: Event producer
        """
        self.memory = memory_manager
        self.events = event_producer
        self.agent_factory = IntegratedAgentFactory(memory_manager, event_producer)

    async def persist_conversation(self, state: AgentState) -> bool:
        """Persist conversation to long-term memory.

        Args:
            state: Final state

        Returns:
            Success status
        """
        request_id = state.get("request_id")
        if not request_id:
            return False

        # Store in vector DB
        success = await self.memory.persist_state_to_long_term(state)

        if success:
            # Publish event
            messages = StateUtils.get_message_history(state)
            await self.events.publish(
                ConversationStoredEvent(
                    request_id=request_id,
                    message_count=len(messages),
                    storage_tier="vector_db",
                    ttl_seconds=None
                )
            )

        return success

    async def optimize_state_memory(self, state: AgentState) -> AgentState:
        """Optimize state by moving old data to cache.

        Args:
            state: Current state

        Returns:
            Optimized state
        """
        return await self.memory.optimize_memory(state, max_messages=20)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from .state import StateBuilder
    from ..domain.models import Priority, RequestCategory
    from ..memory.vector_store import VectorStoreFactory
    from ..memory.redis_cache import RedisCacheFactory
    from ..infrastructure.kafka.producer import ProducerFactory

    async def main():
        print("=== Integrated Agent Demo ===\n")

        # Setup components
        vector_store = VectorStoreFactory.create("weaviate", "http://localhost:8080")
        cache = await RedisCacheFactory.create("localhost", 6379)
        producer = await ProducerFactory.create("localhost:9092")

        memory = MemoryManager(vector_store, cache)
        integration = WorkflowIntegration(memory, producer)

        # Create initial state
        state = (
            StateBuilder()
            .with_request_id("req-integration-test")
            .with_customer_id("cust-123")
            .with_initial_message("How do I reset my password?")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.ACCOUNT)
            .build()
        )

        # Create agents
        factory = integration.agent_factory
        triage = factory.create_triage_agent()
        research = factory.create_research_agent()
        solution = factory.create_solution_agent()
        quality = factory.create_quality_agent()

        try:
            # Execute workflow
            print("1. Triage...")
            state = await triage(state)
            print(f"   Category: {state['category']}, Priority: {state['priority']}")

            print("\n2. Research...")
            state = await research(state)
            print(f"   Retrieved {len(state.get('retrieved_documents', []))} documents")

            print("\n3. Solution...")
            state = await solution(state)
            print(f"   Confidence: {state['solution_confidence']}")

            print("\n4. Quality...")
            state = await quality(state)
            print(f"   Quality passed: {state['quality_passed']}")

            print("\n5. Persist conversation...")
            await integration.persist_conversation(state)
            print("   Stored in vector DB")

            print("\nWorkflow complete!")

        finally:
            await cache.client.close()
            await producer.stop()

    asyncio.run(main())
