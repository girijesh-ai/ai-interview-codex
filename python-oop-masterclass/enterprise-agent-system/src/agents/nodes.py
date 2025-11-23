"""
Agent Node Implementations

All 6 agents implementing the multi-agent workflow:
1. Supervisor - Orchestrates workflow
2. Triage - Classifies and prioritizes
3. Research - Retrieves information (RAG)
4. Solution - Generates responses
5. Escalation - Handles complex cases
6. Quality - Reviews responses

Demonstrates:
- Strategy pattern (different agent strategies)
- Template method pattern (common agent structure)
- Single Responsibility Principle
- Dependency Injection
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState, StateUtils
from ..domain.models import (
    AgentType,
    DecisionType,
    RequestStatus,
    RequestCategory,
    Priority,
    Confidence
)


# ============================================================================
# BASE AGENT CLASS - Template Method Pattern
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents.

    Demonstrates:
    - Template method pattern
    - Abstract base class
    - Common agent structure
    """

    def __init__(
        self,
        agent_type: AgentType,
        llm: Optional[ChatOpenAI] = None
    ):
        """Initialize agent.

        Args:
            agent_type: Type of agent
            llm: Optional language model (defaults to GPT-4)
        """
        self.agent_type = agent_type
        self.llm = llm or ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )

    def __call__(self, state: AgentState) -> AgentState:
        """Execute agent (makes agent callable).

        Template method that defines the execution flow.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        # Update current agent
        state["current_agent"] = self.agent_type.value
        state["agent_history"] = state.get("agent_history", []) + [self.agent_type.value]

        # Metrics tracking
        start_time = datetime.now()

        try:
            # Execute agent-specific logic
            state = self.execute(state)

            # Record success metrics
            duration = (datetime.now() - start_time).total_seconds()
            StateUtils.update_metrics(state, {
                f"{self.agent_type.value}_duration_avg": duration,
                f"{self.agent_type.value}_duration_count": 1,
                f"{self.agent_type.value}_success_count": 1
            })

        except Exception as e:
            # Handle errors
            state["last_error"] = str(e)
            state["error_count"] = state.get("error_count", 0) + 1

            # Record error metrics
            StateUtils.update_metrics(state, {
                f"{self.agent_type.value}_error_count": 1
            })

            # Log error
            print(f"Agent {self.agent_type.value} error: {str(e)}")

        finally:
            # Always update timestamp
            state["updated_at"] = datetime.now().isoformat()

        return state

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute agent-specific logic.

        Must be implemented by subclasses.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        pass

    def _get_system_prompt(self) -> str:
        """Get agent-specific system prompt.

        Returns:
            System prompt string
        """
        return f"You are a {self.agent_type.value} agent in a customer support system."


# ============================================================================
# 1. SUPERVISOR AGENT
# ============================================================================

class SupervisorNode(BaseAgent):
    """Supervisor agent - orchestrates the workflow.

    Responsibilities:
    - Initialize workflow
    - Monitor progress
    - Coordinate agents
    - Handle workflow completion
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize supervisor agent."""
        super().__init__(AgentType.SUPERVISOR, llm)

    def execute(self, state: AgentState) -> AgentState:
        """Execute supervisor logic.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        status = state.get("status")

        # Initialize workflow if pending
        if status == RequestStatus.PENDING.value:
            state["workflow_stage"] = "initialized"
            state["status"] = RequestStatus.TRIAGED.value

            # Add system message
            StateUtils.add_system_message(
                state,
                "Request received and workflow initiated."
            )

        # Check for completion
        elif status == RequestStatus.COMPLETED.value:
            state["workflow_stage"] = "completed"

            # Record decision
            StateUtils.record_decision(
                state,
                self.agent_type,
                DecisionType.COMPLETE,
                1.0,
                "Workflow completed successfully"
            )

        # Monitor progress
        else:
            state["workflow_stage"] = "in_progress"

        return state

    def _get_system_prompt(self) -> str:
        """Get supervisor system prompt."""
        return """You are the Supervisor agent coordinating a customer support workflow.

Your responsibilities:
- Initialize and monitor the workflow
- Ensure smooth coordination between agents
- Track progress and handle completion
- Escalate issues when needed

Be efficient and decisive."""


# ============================================================================
# 2. TRIAGE AGENT
# ============================================================================

class TriageNode(BaseAgent):
    """Triage agent - classifies and prioritizes requests.

    Responsibilities:
    - Categorize request
    - Set priority
    - Route to appropriate agent
    - Extract key information
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize triage agent."""
        super().__init__(AgentType.TRIAGE, llm)

    def execute(self, state: AgentState) -> AgentState:
        """Execute triage logic.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        # Get last user message
        last_message = StateUtils.get_last_user_message(state)

        if not last_message:
            return state

        # Create triage prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", """Analyze this customer request and provide:
1. Category (account, billing, technical, product, refund, general)
2. Priority (1-4, where 4 is critical)
3. Key entities (customer issues, product names, etc.)
4. Sentiment (positive, neutral, negative)
5. Urgency indicators

Customer message: {message}

Provide your analysis in JSON format.""")
        ])

        # Get LLM analysis
        chain = prompt | self.llm
        response = chain.invoke({"message": last_message})

        # Parse response (simplified - would use structured output in production)
        analysis = self._parse_triage_response(response.content)

        # Update state
        if "category" in analysis:
            state["category"] = analysis["category"]

        if "priority" in analysis:
            # Adjust priority based on customer tier
            base_priority = analysis["priority"]
            # Would check customer tier here
            state["priority"] = min(base_priority, 4)

        # Record triage decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ROUTE,
            analysis.get("confidence", 0.8),
            f"Triaged as {analysis.get('category', 'unknown')} with priority {analysis.get('priority', 2)}"
        )

        # Update status
        state["status"] = RequestStatus.TRIAGED.value
        state["workflow_stage"] = "triaged"

        return state

    def _parse_triage_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM triage response.

        Args:
            response: LLM response text

        Returns:
            Parsed analysis dictionary
        """
        # Simplified parsing - would use structured output in production
        import json
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass

        # Fallback to defaults
        return {
            "category": "general",
            "priority": 2,
            "confidence": 0.5
        }

    def _get_system_prompt(self) -> str:
        """Get triage system prompt."""
        return """You are a Triage agent specializing in customer request classification.

Your responsibilities:
- Accurately categorize requests (account, billing, technical, product, refund, general)
- Assess priority based on urgency, customer tier, and business impact
- Extract key entities and context
- Identify sentiment and urgency indicators

Be accurate and decisive in your classification."""


# ============================================================================
# 3. RESEARCH AGENT
# ============================================================================

class ResearchNode(BaseAgent):
    """Research agent - retrieves information using RAG.

    Responsibilities:
    - Search knowledge base
    - Retrieve relevant documents
    - Find similar past cases
    - Gather context for solution
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        vector_store=None  # Will be injected
    ):
        """Initialize research agent.

        Args:
            llm: Language model
            vector_store: Vector database for RAG
        """
        super().__init__(AgentType.RESEARCH, llm)
        self.vector_store = vector_store

    def execute(self, state: AgentState) -> AgentState:
        """Execute research logic.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved context
        """
        # Get query from last message
        query = StateUtils.get_last_user_message(state)
        category = state.get("category", "general")

        if not query:
            return state

        # Build search query
        search_query = self._build_search_query(query, category)

        # Search vector store (simulated - would use real vector DB)
        retrieved_docs = self._search_knowledge_base(search_query, k=5)

        # Update state with retrieved documents
        state["retrieved_documents"] = state.get("retrieved_documents", []) + retrieved_docs

        # Extract context IDs
        context_ids = {doc["id"] for doc in retrieved_docs}
        state["relevant_context_ids"] = state.get("relevant_context_ids", set()) | context_ids

        # Calculate confidence based on retrieval quality
        avg_score = sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        confidence = min(avg_score, 1.0)

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ROUTE,
            confidence,
            f"Retrieved {len(retrieved_docs)} relevant documents with avg score {avg_score:.2f}"
        )

        # Update status
        state["status"] = RequestStatus.RESEARCHING.value
        state["workflow_stage"] = "research_completed"

        return state

    def _build_search_query(self, query: str, category: str) -> str:
        """Build enhanced search query.

        Args:
            query: Original query
            category: Request category

        Returns:
            Enhanced search query
        """
        # Add category context to improve search
        return f"[{category}] {query}"

    def _search_knowledge_base(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base (simulated).

        In production, this would query a real vector database.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of retrieved documents
        """
        # Simulated retrieval - in production, would use:
        # results = self.vector_store.similarity_search_with_score(query, k=k)

        # Return simulated documents
        return [
            {
                "id": f"doc_{i}",
                "content": f"Knowledge base content related to: {query}",
                "score": 0.85 - (i * 0.1),
                "metadata": {"source": "kb", "category": "general"}
            }
            for i in range(min(k, 3))
        ]

    def _get_system_prompt(self) -> str:
        """Get research system prompt."""
        return """You are a Research agent specializing in information retrieval.

Your responsibilities:
- Search the knowledge base for relevant information
- Retrieve similar past cases and solutions
- Gather comprehensive context for the solution agent
- Assess the quality and relevance of retrieved information

Be thorough and accurate in your research."""


# ============================================================================
# 4. SOLUTION AGENT
# ============================================================================

class SolutionNode(BaseAgent):
    """Solution agent - generates responses.

    Responsibilities:
    - Draft solutions based on context
    - Generate customer-friendly responses
    - Ensure accuracy and completeness
    - Assess confidence in solution
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize solution agent."""
        super().__init__(AgentType.SOLUTION, llm)

    def execute(self, state: AgentState) -> AgentState:
        """Execute solution generation logic.

        Args:
            state: Current agent state

        Returns:
            Updated state with proposed solution
        """
        # Get context
        query = StateUtils.get_last_user_message(state)
        category = state.get("category", "general")
        retrieved_docs = state.get("retrieved_documents", [])

        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['content']}"
            for i, doc in enumerate(retrieved_docs[:3])
        ])

        # Create solution prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", """Based on the following context, provide a helpful solution to the customer's request.

Customer request: {query}
Category: {category}

Context from knowledge base:
{context}

Provide:
1. A clear, customer-friendly solution
2. Step-by-step instructions if applicable
3. Any important warnings or notes
4. Your confidence level (0-1) in this solution

Format as JSON with fields: solution, confidence, notes""")
        ])

        # Generate solution
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "category": category,
            "context": context or "No specific context available"
        })

        # Parse response
        solution_data = self._parse_solution_response(response.content)

        # Update state
        state["proposed_solution"] = solution_data.get("solution", "")
        state["solution_confidence"] = solution_data.get("confidence", 0.7)

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ROUTE,
            solution_data.get("confidence", 0.7),
            "Generated solution based on retrieved context"
        )

        # Update status
        state["status"] = RequestStatus.DRAFTING.value
        state["workflow_stage"] = "solution_drafted"

        return state

    def _parse_solution_response(self, response: str) -> Dict[str, Any]:
        """Parse solution response.

        Args:
            response: LLM response

        Returns:
            Parsed solution data
        """
        import json
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass

        # Fallback
        return {
            "solution": response,
            "confidence": 0.7,
            "notes": ""
        }

    def _get_system_prompt(self) -> str:
        """Get solution system prompt."""
        return """You are a Solution agent specializing in customer support responses.

Your responsibilities:
- Generate accurate, helpful solutions
- Write in a clear, customer-friendly tone
- Include step-by-step instructions when appropriate
- Be honest about confidence levels
- Flag cases that may need human review

Always prioritize customer satisfaction and accuracy."""


# ============================================================================
# 5. ESCALATION AGENT
# ============================================================================

class EscalationNode(BaseAgent):
    """Escalation agent - handles complex cases.

    Responsibilities:
    - Identify cases needing human review
    - Prepare escalation requests
    - Handle policy-sensitive issues
    - Coordinate with human operators
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize escalation agent."""
        super().__init__(AgentType.ESCALATION, llm)

    def execute(self, state: AgentState) -> AgentState:
        """Execute escalation logic.

        Args:
            state: Current agent state

        Returns:
            Updated state with escalation details
        """
        # Determine escalation reason
        priority = state.get("priority", 2)
        category = state.get("category")
        confidence = state.get("solution_confidence", 0.0)

        reasons = []

        if priority >= 4:
            reasons.append("Critical priority request")

        if category == "refund":
            reasons.append("Refund request requires approval")

        if confidence < 0.6:
            reasons.append(f"Low confidence in solution ({confidence:.2f})")

        if not state.get("retrieved_documents"):
            reasons.append("Insufficient knowledge base information")

        escalation_reason = "; ".join(reasons) if reasons else "Complex case requiring human judgment"

        # Set escalation flags
        state["requires_approval"] = True
        state["status"] = RequestStatus.AWAITING_APPROVAL.value
        state["workflow_stage"] = "escalated"

        # Add system message
        StateUtils.add_system_message(
            state,
            f"Case escalated for human review. Reason: {escalation_reason}"
        )

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.ESCALATE,
            1.0,
            escalation_reason
        )

        return state

    def _get_system_prompt(self) -> str:
        """Get escalation system prompt."""
        return """You are an Escalation agent responsible for identifying cases that need human intervention.

Your responsibilities:
- Identify complex cases beyond AI capabilities
- Prepare clear escalation requests
- Ensure policy compliance
- Coordinate smooth handoffs to human operators

Be decisive but not overly cautious - escalate when truly necessary."""


# ============================================================================
# 6. QUALITY AGENT
# ============================================================================

class QualityNode(BaseAgent):
    """Quality agent - reviews responses before delivery.

    Responsibilities:
    - Check solution quality
    - Verify accuracy
    - Ensure tone is appropriate
    - Flag potential issues
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize quality agent."""
        super().__init__(AgentType.QUALITY, llm)

    def execute(self, state: AgentState) -> AgentState:
        """Execute quality check logic.

        Args:
            state: Current agent state

        Returns:
            Updated state with quality assessment
        """
        solution = state.get("proposed_solution", "")
        confidence = state.get("solution_confidence", 0.0)
        category = state.get("category", "general")

        # Quality checks
        quality_score = self._assess_quality(solution, confidence, category)

        # Determine if quality passed
        quality_threshold = 0.75
        quality_passed = quality_score >= quality_threshold

        # Check if human review needed
        requires_review = (
            not quality_passed or
            confidence < state.get("confidence_threshold", 0.8) or
            state.get("priority", 2) >= 4
        )

        # Update state
        state["quality_passed"] = quality_passed
        if requires_review:
            state["requires_approval"] = True

        # Record decision
        StateUtils.record_decision(
            state,
            self.agent_type,
            DecisionType.APPROVE if quality_passed else DecisionType.REJECT,
            quality_score,
            f"Quality check {'passed' if quality_passed else 'failed'} with score {quality_score:.2f}"
        )

        # Update status
        state["status"] = RequestStatus.QUALITY_CHECK.value
        state["workflow_stage"] = "quality_checked"

        return state

    def _assess_quality(
        self,
        solution: str,
        confidence: float,
        category: str
    ) -> float:
        """Assess solution quality.

        Args:
            solution: Proposed solution
            confidence: Solution confidence
            category: Request category

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        # Length check
        if len(solution) > 50:
            score += 0.2

        # Confidence contribution
        score += confidence * 0.5

        # Completeness (simplified - would use LLM in production)
        if "step" in solution.lower() or "first" in solution.lower():
            score += 0.2

        # Politeness
        if any(word in solution.lower() for word in ["please", "thank", "happy to"]):
            score += 0.1

        return min(score, 1.0)

    def _get_system_prompt(self) -> str:
        """Get quality system prompt."""
        return """You are a Quality agent ensuring response excellence.

Your responsibilities:
- Verify solution accuracy and completeness
- Check tone and professionalism
- Ensure policy compliance
- Flag potential issues before delivery

Maintain high standards while being practical."""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from .state import StateBuilder

    # Create initial state
    state = (
        StateBuilder()
        .with_request_id("req-123")
        .with_customer_id("cust-456")
        .with_initial_message("How do I reset my password?")
        .build()
    )

    # Test each agent
    print("Testing Supervisor Agent:")
    supervisor = SupervisorNode()
    state = supervisor(state)
    print(f"Status: {state['status']}")

    print("\nTesting Triage Agent:")
    triage = TriageNode()
    state = triage(state)
    print(f"Category: {state.get('category')}")
    print(f"Priority: {state.get('priority')}")

    print("\nTesting Research Agent:")
    research = ResearchNode()
    state = research(state)
    print(f"Retrieved docs: {len(state.get('retrieved_documents', []))}")

    print("\nTesting Solution Agent:")
    solution = SolutionNode()
    state = solution(state)
    print(f"Solution: {state.get('proposed_solution', '')[:100]}...")

    print("\nTesting Quality Agent:")
    quality = QualityNode()
    state = quality(state)
    print(f"Quality passed: {state.get('quality_passed')}")
