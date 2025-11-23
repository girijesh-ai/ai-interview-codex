"""
LangGraph Multi-Agent Workflow with Human-in-the-Loop

Demonstrates:
- StateGraph with conditional routing
- Multi-agent orchestration
- Human-in-the-loop with interrupt
- Checkpointing for fault tolerance
- Error handling and retry logic
"""

from typing import Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt

from .state import AgentState, StateUtils
from .nodes import (
    SupervisorNode,
    TriageNode,
    ResearchNode,
    SolutionNode,
    EscalationNode,
    QualityNode
)
from ..domain.models import AgentType, RequestStatus


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def route_after_supervisor(
    state: AgentState
) -> Literal["triage", "end"]:
    """Route from supervisor node.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    status = state.get("status")

    if status == RequestStatus.PENDING.value:
        return "triage"
    elif status in [RequestStatus.COMPLETED.value, RequestStatus.FAILED.value]:
        return "end"
    else:
        # Continue workflow
        return "triage"


def route_after_triage(
    state: AgentState
) -> Literal["research", "solution", "escalation"]:
    """Route from triage node based on complexity.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    priority = state.get("priority", 2)
    category = state.get("category")

    # High priority or refund category -> escalate immediately
    if priority >= 4 or category == "refund":
        return "escalation"

    # Technical or product questions -> research first
    if category in ["technical", "product"]:
        return "research"

    # Simple questions -> direct solution
    return "solution"


def route_after_research(
    state: AgentState
) -> Literal["solution", "escalation"]:
    """Route from research node based on findings.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    # Check if sufficient context was found
    retrieved_docs = state.get("retrieved_documents", [])

    if len(retrieved_docs) >= 3:
        # Sufficient information found
        return "solution"
    else:
        # Insufficient information -> escalate
        return "escalation"


def route_after_solution(
    state: AgentState
) -> Literal["quality", "escalation"]:
    """Route from solution node based on confidence.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    confidence = state.get("solution_confidence", 0.0)
    threshold = state.get("confidence_threshold", 0.8)

    if confidence >= threshold:
        return "quality"
    else:
        # Low confidence -> escalate for review
        return "escalation"


def route_after_quality(
    state: AgentState
) -> Literal["human_review", "deliver", "solution"]:
    """Route from quality node based on quality check.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    requires_approval = state.get("requires_approval", False)
    quality_passed = state.get("quality_passed", False)

    if requires_approval:
        return "human_review"
    elif quality_passed:
        return "deliver"
    else:
        # Quality check failed -> revise solution
        return "solution"


def route_after_escalation(
    state: AgentState
) -> Literal["human_decision", "solution"]:
    """Route from escalation node.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    # Check if we should retry with more context
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        # Try again with human guidance
        return "human_decision"
    else:
        # Max retries reached -> need human decision
        return "human_decision"


def route_after_human_review(
    state: AgentState
) -> Literal["deliver", "solution", "escalation"]:
    """Route after human review/approval.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    approval_status = state.get("approval_status")

    if approval_status == "approved":
        return "deliver"
    elif approval_status == "revision_needed":
        return "solution"
    else:
        # Rejected -> escalate further
        return "escalation"


# ============================================================================
# HUMAN-IN-THE-LOOP NODES
# ============================================================================

def human_review_node(state: AgentState) -> AgentState:
    """Human review node with interrupt.

    This node pauses execution and waits for human input.
    The graph can be resumed hours or days later.

    Args:
        state: Current agent state

    Returns:
        Updated state with approval decision
    """
    # Prepare approval request
    approval_request = {
        "request_id": state["request_id"],
        "customer_id": state["customer_id"],
        "priority": state["priority"],
        "category": state["category"],
        "proposed_solution": state.get("proposed_solution"),
        "confidence": state.get("solution_confidence"),
        "reason": "Requires human approval due to high priority or policy",
        "conversation": [
            {"role": msg.__class__.__name__, "content": msg.content}
            for msg in state["messages"][-5:]  # Last 5 messages
        ]
    }

    # INTERRUPT: Pause graph execution
    # State is saved, resources are freed
    # Can resume when human responds (seconds to days later)
    approval_response = interrupt(approval_request)

    # When resumed, update state with approval
    state["approval_status"] = approval_response.get("status", "pending")
    state["approved_by"] = approval_response.get("approver")
    state["approval_notes"] = approval_response.get("notes")
    state["requires_approval"] = False

    # Record decision
    StateUtils.record_decision(
        state,
        AgentType.ESCALATION,
        "APPROVE" if approval_response.get("status") == "approved" else "REJECT",
        1.0,
        f"Human {approval_response.get('status')}: {approval_response.get('notes', '')}"
    )

    return state


def human_decision_node(state: AgentState) -> AgentState:
    """Human decision node for complex escalations.

    Args:
        state: Current agent state

    Returns:
        Updated state with human guidance
    """
    decision_request = {
        "request_id": state["request_id"],
        "customer_id": state["customer_id"],
        "escalation_reason": state.get("last_error", "Complex case requiring human judgment"),
        "context": {
            "priority": state["priority"],
            "category": state["category"],
            "retry_count": state["retry_count"],
            "decisions": state.get("decisions", [])[-3:]  # Last 3 decisions
        },
        "options": [
            {"id": "continue", "label": "Continue with AI assistance"},
            {"id": "manual", "label": "Handle manually"},
            {"id": "close", "label": "Close request"}
        ]
    }

    # INTERRUPT: Wait for human decision
    decision_response = interrupt(decision_request)

    # Update state based on decision
    decision = decision_response.get("decision", "continue")

    if decision == "continue":
        # Add human guidance to session data
        state["session_data"]["human_guidance"] = decision_response.get("guidance", "")
        state["status"] = RequestStatus.RESEARCHING.value
    elif decision == "manual":
        state["status"] = RequestStatus.ESCALATED.value
        state["requires_approval"] = False
    else:  # close
        state["status"] = RequestStatus.COMPLETED.value
        state["proposed_solution"] = decision_response.get("resolution", "Closed by human decision")

    return state


def deliver_node(state: AgentState) -> AgentState:
    """Deliver final response to customer.

    Args:
        state: Current agent state

    Returns:
        Updated state with completion
    """
    solution = state.get("proposed_solution", "")

    # Add solution as AI message
    from langchain_core.messages import AIMessage
    state["messages"].append(
        AIMessage(content=solution)
    )

    # Update status
    state["status"] = RequestStatus.COMPLETED.value
    state["workflow_stage"] = "completed"

    # Record final metrics
    execution_time = StateUtils.get_execution_time(state)
    StateUtils.update_metrics(state, {
        "total_duration_avg": execution_time,
        "total_duration_count": 1,
        "completion_count": 1
    })

    return state


# ============================================================================
# GRAPH BUILDER
# ============================================================================

class AgentGraphBuilder:
    """Builder for creating the agent workflow graph.

    Demonstrates:
    - Builder pattern
    - Graph construction
    - Conditional routing
    - Human-in-the-loop integration
    """

    def __init__(self, checkpointer: PostgresSaver):
        """Initialize graph builder.

        Args:
            checkpointer: PostgreSQL checkpointer for state persistence
        """
        self.checkpointer = checkpointer
        self.graph = StateGraph(AgentState)

    def add_nodes(self) -> "AgentGraphBuilder":
        """Add all agent nodes to graph.

        Returns:
            Self for chaining
        """
        # Agent nodes
        self.graph.add_node("supervisor", SupervisorNode())
        self.graph.add_node("triage", TriageNode())
        self.graph.add_node("research", ResearchNode())
        self.graph.add_node("solution", SolutionNode())
        self.graph.add_node("escalation", EscalationNode())
        self.graph.add_node("quality", QualityNode())

        # HITL nodes
        self.graph.add_node("human_review", human_review_node)
        self.graph.add_node("human_decision", human_decision_node)

        # Delivery node
        self.graph.add_node("deliver", deliver_node)

        return self

    def add_edges(self) -> "AgentGraphBuilder":
        """Add edges with conditional routing.

        Returns:
            Self for chaining
        """
        # Set entry point
        self.graph.set_entry_point("supervisor")

        # Conditional routing from supervisor
        self.graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                "triage": "triage",
                "end": END
            }
        )

        # Conditional routing from triage
        self.graph.add_conditional_edges(
            "triage",
            route_after_triage,
            {
                "research": "research",
                "solution": "solution",
                "escalation": "escalation"
            }
        )

        # Conditional routing from research
        self.graph.add_conditional_edges(
            "research",
            route_after_research,
            {
                "solution": "solution",
                "escalation": "escalation"
            }
        )

        # Conditional routing from solution
        self.graph.add_conditional_edges(
            "solution",
            route_after_solution,
            {
                "quality": "quality",
                "escalation": "escalation"
            }
        )

        # Conditional routing from quality
        self.graph.add_conditional_edges(
            "quality",
            route_after_quality,
            {
                "human_review": "human_review",
                "deliver": "deliver",
                "solution": "solution"
            }
        )

        # Conditional routing from escalation
        self.graph.add_conditional_edges(
            "escalation",
            route_after_escalation,
            {
                "human_decision": "human_decision",
                "solution": "solution"
            }
        )

        # Conditional routing from human review
        self.graph.add_conditional_edges(
            "human_review",
            route_after_human_review,
            {
                "deliver": "deliver",
                "solution": "solution",
                "escalation": "escalation"
            }
        )

        # Human decision routes
        self.graph.add_conditional_edges(
            "human_decision",
            lambda state: "solution" if state["status"] == RequestStatus.RESEARCHING.value else "end",
            {
                "solution": "solution",
                "end": END
            }
        )

        # Deliver ends the workflow
        self.graph.add_edge("deliver", END)

        return self

    def build(self):
        """Build and compile the graph.

        Returns:
            Compiled graph ready for execution
        """
        # Compile with checkpointer for state persistence
        app = self.graph.compile(checkpointer=self.checkpointer)
        return app


# ============================================================================
# GRAPH FACTORY
# ============================================================================

def create_agent_graph(
    postgres_connection_string: str
) -> "CompiledGraph":
    """Factory function to create configured agent graph.

    Args:
        postgres_connection_string: PostgreSQL connection string

    Returns:
        Compiled graph ready for execution

    Example:
        >>> graph = create_agent_graph("postgresql://user:pass@localhost/agents")
        >>> result = graph.invoke(
        ...     initial_state,
        ...     config={"configurable": {"thread_id": "thread-123"}}
        ... )
    """
    # Create checkpointer
    checkpointer = PostgresSaver.from_conn_string(postgres_connection_string)

    # Build graph
    builder = AgentGraphBuilder(checkpointer)
    graph = (builder
             .add_nodes()
             .add_edges()
             .build())

    return graph


# ============================================================================
# GRAPH EXECUTION UTILITIES
# ============================================================================

class GraphExecutor:
    """Utility class for executing graphs with error handling.

    Demonstrates:
    - Facade pattern
    - Error handling
    - Logging integration
    """

    def __init__(self, graph, logger=None):
        """Initialize executor.

        Args:
            graph: Compiled LangGraph
            logger: Optional logger instance
        """
        self.graph = graph
        self.logger = logger

    async def execute(
        self,
        initial_state: AgentState,
        thread_id: str,
        checkpoint_id: str = None
    ) -> AgentState:
        """Execute graph with error handling.

        Args:
            initial_state: Initial state
            thread_id: Thread ID for state isolation
            checkpoint_id: Optional checkpoint to resume from

        Returns:
            Final state after execution

        Raises:
            GraphExecutionError: If execution fails
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        try:
            if self.logger:
                self.logger.info(f"Starting graph execution for thread {thread_id}")

            # Execute graph
            result = await self.graph.ainvoke(initial_state, config=config)

            if self.logger:
                self.logger.info(f"Graph execution completed for thread {thread_id}")

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"Graph execution failed: {str(e)}")
            raise GraphExecutionError(f"Failed to execute graph: {str(e)}") from e

    async def execute_with_streaming(
        self,
        initial_state: AgentState,
        thread_id: str
    ):
        """Execute graph with streaming updates.

        Args:
            initial_state: Initial state
            thread_id: Thread ID

        Yields:
            State updates as graph executes
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        async for event in self.graph.astream(initial_state, config=config):
            yield event

    async def resume_from_interrupt(
        self,
        thread_id: str,
        checkpoint_id: str,
        resume_value: dict
    ) -> AgentState:
        """Resume graph execution from interrupt point.

        Args:
            thread_id: Thread ID
            checkpoint_id: Checkpoint to resume from
            resume_value: Value to resume with (e.g., approval decision)

        Returns:
            Final state after resumption
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }

        if self.logger:
            self.logger.info(f"Resuming graph from checkpoint {checkpoint_id}")

        # Resume with provided value
        result = await self.graph.ainvoke(
            None,  # No new input
            config=config,
            resume_value=resume_value
        )

        return result


class GraphExecutionError(Exception):
    """Custom exception for graph execution errors."""
    pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from .state import StateBuilder
    from ..domain.models import Priority, RequestCategory

    async def main():
        # Create graph
        graph = create_agent_graph("postgresql://user:pass@localhost/agents")

        # Create initial state
        initial_state = (
            StateBuilder()
            .with_request_id("req-12345")
            .with_customer_id("cust-789")
            .with_thread_id("thread-abc")
            .with_initial_message("I need a refund for my order")
            .with_priority(Priority.HIGH)
            .with_category(RequestCategory.REFUND)
            .build()
        )

        # Execute graph
        executor = GraphExecutor(graph)

        try:
            result = await executor.execute(
                initial_state,
                thread_id="thread-abc"
            )

            print(f"Final status: {result['status']}")
            print(f"Decisions made: {len(result['decisions'])}")

        except GraphExecutionError as e:
            print(f"Execution failed: {e}")

    # Run
    asyncio.run(main())
