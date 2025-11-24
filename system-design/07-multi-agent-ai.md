# Multi-Agent AI System Design - FAANG Interview Guide

## Interview Format: Conversational & Iterative

This guide simulates a real Gen AI system design interview focused on multi-agent AI systems with agent orchestration, tool use, planning, and observability - reflecting 2025 best practices.

---

## Interview Timeline (45 minutes)

| Phase | Time | Your Actions |
|-------|------|--------------|
| Requirements Gathering | 5-7 min | Ask clarifying questions, define scope |
| High-Level Design | 10-12 min | Draw architecture, explain agent patterns |
| Deep Dive | 20-25 min | Detail orchestration, tool calling, safety |
| Trade-offs & Scale | 5-8 min | Discuss frameworks, patterns, limitations |

---

## 🎯 Problem Statement

**Interviewer:** "Design a multi-agent AI system for customer support that can handle complex inquiries by coordinating multiple specialized agents (e.g., billing agent, technical support agent, order tracking agent)."

---

## Phase 1: Requirements Gathering (5-7 minutes)

### Your Response Strategy

**You:** "Multi-agent systems are powerful for complex tasks. Let me understand the requirements."

### Critical Questions to Ask

**You:** "Let me clarify the scope:

1. **Use Case & Complexity:**
   - What types of customer inquiries? (simple FAQs vs complex troubleshooting?)
   - How many specialized agents needed?
   - Single-turn or multi-turn conversations?
   - Human-in-the-loop required?

2. **Agent Capabilities:**
   - What tools do agents need? (database queries, APIs, calculations?)
   - Can agents call other agents?
   - Planning required? (multi-step workflows)
   - Memory needed? (conversation history, user context)

3. **Scale & Latency:**
   - Concurrent conversations?
   - Response time requirements?
   - Cost constraints per conversation?

4. **Safety & Control:**
   - What safety guardrails?
   - Approval needed for certain actions?
   - Audit logging requirements?
   - Fallback to human agents?

5. **Evaluation:**
   - Success metrics? (resolution rate, user satisfaction?)
   - How to measure agent quality?"

### Interviewer's Answers

**Interviewer:** "Here's the scope:

- **Use Case:** E-commerce customer support (Amazon-scale)
- **Query Types:** 40% simple (order status), 40% medium (returns/refunds), 20% complex (technical issues)
- **Agents Needed:** 5 specialized agents (Order, Billing, Technical, Returns, Product Info)
- **Conversations:** Multi-turn, avg 5 messages per conversation
- **Tools:** Database queries, payment API, shipping API, product catalog
- **Scale:** 10K concurrent conversations, 100K conversations/day
- **Latency:** <10 seconds per agent response
- **Safety:** Require approval for refunds >$100, escalate to human for complex cases
- **Memory:** Track conversation context, user purchase history"

### Requirements Summary & Calculations

**You:** "Perfect! Let me summarize with cost/complexity analysis:

#### Functional Requirements
- 5 specialized agents: Order, Billing, Technical, Returns, Product Info
- Agent orchestrator: Routes queries to appropriate agent
- Tool calling: Database queries, payment API, shipping API, product catalog
- Multi-turn conversations (avg 5 messages, maintain context)
- Agent collaboration: Handoffs between agents (e.g., Order → Billing)
- Human escalation: Complex cases, sentiment detection
- Approval workflow: Refunds >$100 require approval
- Memory: Conversation history + user purchase context

#### Non-Functional Requirements & Calculations

**Scale:**
- 100K conversations/day × 5 messages/conv = **500K messages/day**
- 10K concurrent conversations (peak)
- Agent breakdown: 40% simple (1 agent), 40% medium (2 agents), 20% complex (3+ agents)
- Average agents per conversation: 1.8 agents

**LLM Calls:**
- 500K messages × 1.8 agents = **900K LLM calls/day = 27M/month**
- Per call: 1K tokens input (context) + 200 tokens output
- Total: 27M × 1.2K tokens = **32.4B tokens/month**

**Cost:**
- GPT-4: 32.4B × $0.03/1M = **$972/month** (input) + $324 (output) = $1,296/month
- Claude 3.5: 32.4B × $0.003/1M = **$97/month** (input) + $49 (output) = $146/month
- **Claude is 9× cheaper!**

**Latency Budget (10s per response):**
- Query classification: **500ms** (which agent to use?)
- Context retrieval: **1s** (user history, past conversation)
- Tool calls (if needed): **2-3s** (database/API calls)
- LLM generation: **4-5s** (agent response)
- Response post-processing: **500ms** (format, safety check)
- **Total: 8.5-10s**

**Storage:**
- Conversation history: 100K conv/day × 5 messages × 500 bytes = **250MB/day**
- User context: 10M users × 2KB = **20GB**
- Agent models: 5 agents × 100MB configs = **500MB**

**Success Metrics:**
- Resolution rate: >80% (resolved without human)
- User satisfaction: >4.2/5 stars
- Average resolution time: <3 minutes
- Escalation rate: <15% to human agents
- Cost per conversation: <$0.05

#### Key Challenges
- **Orchestration:** Route to correct agent(s), handle handoffs
- **Tool Reliability:** APIs fail → graceful degradation, retries
- **Context Management:** Track state across 5 messages × multiple agents
- **Cost Optimization:** 27M LLM calls/month → caching, prompt compression
- **Safety:** Prevent harmful actions, require approvals
- **Scalability:** 10K concurrent conversations
- **Safety:** Prevent harmful actions
- **Observability:** Debug complex multi-agent flows

Sounds good?"

**Interviewer:** "Yes, proceed."

---

## Phase 2: High-Level Design (10-12 minutes)

### Architecture Overview

**You:** "I'll design using the Supervisor Pattern with LangGraph orchestration (2025 industry standard)."

```mermaid
graph TB
    subgraph "User Layer"
        U[User Query:<br/>I want to return my laptop]
        UI[Chat Interface]
    end

    subgraph "Orchestration Layer - Supervisor"
        SUP[Supervisor Agent<br/>Route & Coordinate]
        PLAN[Planning Module<br/>ReAct / Plan-Execute]
        MEM[Memory Manager<br/>Conversation State]
    end

    subgraph "Specialized Agents"
        A1[Order Agent<br/>Track orders, status]
        A2[Billing Agent<br/>Payments, refunds]
        A3[Returns Agent<br/>Return policy, RMA]
        A4[Technical Agent<br/>Troubleshooting]
        A5[Product Agent<br/>Product info, specs]
    end

    subgraph "Tool Layer"
        T1[Order DB Query]
        T2[Payment API]
        T3[Shipping API]
        T4[Product Catalog]
        T5[Calculator]
        T6[Email Sender]
    end

    subgraph "Safety & Control"
        GUARD[Guardrails<br/>Content safety, PII detection]
        APPROVAL[Approval Workflow<br/>Human review for $>100]
        FALLBACK[Human Escalation<br/>Complex cases]
    end

    subgraph "Observability"
        TRACE[LangSmith Tracing<br/>Agent execution logs]
        MON[Monitoring<br/>Performance metrics]
        EVAL[Eval Framework<br/>Quality assessment]
    end

    subgraph "State Management"
        CONV[(Conversation Store<br/>Redis)]
        USER[(User Context DB<br/>PostgreSQL)]
        CACHE[(LLM Cache<br/>Reduce costs)]
    end

    U --> UI
    UI --> SUP
    SUP --> PLAN
    SUP --> MEM

    SUP --> A1
    SUP --> A2
    SUP --> A3
    SUP --> A4
    SUP --> A5

    A1 --> T1
    A1 --> T3
    A2 --> T2
    A3 --> T1
    A3 --> T3
    A4 --> T4
    A5 --> T4

    A2 --> GUARD
    GUARD --> APPROVAL
    SUP --> FALLBACK

    SUP --> TRACE
    A1 --> TRACE
    A2 --> TRACE

    TRACE --> MON
    TRACE --> EVAL

    MEM --> CONV
    SUP --> USER
    SUP --> CACHE

    style SUP fill:#FFD700
    style PLAN fill:#90EE90
    style GUARD fill:#FFB6C1
    style TRACE fill:#87CEEB
```

### Multi-Agent Patterns

**You:** "Let me explain the key patterns for multi-agent systems:

```mermaid
graph TB
    subgraph "Pattern 1: Supervisor (Our Choice)"
        S1[Supervisor Agent]
        S1 --> W1[Worker Agent 1]
        S1 --> W2[Worker Agent 2]
        S1 --> W3[Worker Agent N]
        W1 --> S1
        W2 --> S1
        W3 --> S1
    end

    subgraph "Pattern 2: Sequential Chain"
        C1[Agent 1] --> C2[Agent 2]
        C2 --> C3[Agent 3]
        C3 --> C4[Result]
    end

    subgraph "Pattern 3: Hierarchical"
        H1[Top-level Agent]
        H1 --> H2[Mid-level Agent 1]
        H1 --> H3[Mid-level Agent 2]
        H2 --> H4[Worker 1]
        H2 --> H5[Worker 2]
        H3 --> H6[Worker 3]
    end

    style S1 fill:#90EE90
    style C1 fill:#FFB6C1
    style H1 fill:#FFD700
```

**Why Supervisor Pattern:**
- **Centralized Control:** One agent decides routing
- **Flexibility:** Easy to add new agents
- **Debugging:** Clear execution trace
- **Safety:** Single point for guardrails

**Data Flow - Example:**

```
User: "I want to return my laptop and get a refund"
    ↓
Supervisor: Analyze query → Need Returns Agent + Billing Agent
    ↓
1. Route to Returns Agent:
   - Check return eligibility
   - Tools: Order DB, Return Policy API
   - Output: "Return approved, RMA #12345"
    ↓
2. Supervisor: Returns approved → Now handle refund
    ↓
3. Route to Billing Agent:
   - Process refund
   - Tools: Payment API
   - Safety: Check amount → $1200 > $100 → Requires approval
    ↓
4. Approval Workflow:
   - Notify human supervisor
   - Wait for approval
    ↓
5. Billing Agent: Process refund (approved)
    ↓
Supervisor: Combine responses → Return to user
```"

**Interviewer:** "How do you implement the supervisor and agent coordination? What framework do you use?"

---

## Phase 3: Deep Dive - Implementation with LangGraph (20-25 minutes)

### LangGraph Implementation

**You:** "LangGraph is the 2025 standard for multi-agent orchestration. Let me show you:

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import TypedDict, Annotated, List
import operator

# State definition
class AgentState(TypedDict):
    \"\"\"
    Shared state across all agents

    This is how agents communicate in LangGraph
    \"\"\"
    messages: Annotated[List[str], operator.add]  # Conversation history
    current_agent: str  # Which agent is active
    user_query: str  # Original user question
    user_context: dict  # User info (order history, etc.)
    tool_outputs: dict  # Results from tool calls
    requires_approval: bool  # Flag for approval workflow
    approved: bool  # Approval status
    next_action: str  # What to do next


class SupervisorAgent:
    \"\"\"
    Supervisor: Routes queries to appropriate specialist agents

    Uses LLM to decide:
    1. Which agent should handle this?
    2. Is the task complete?
    3. Should we escalate to human?
    \"\"\"

    def __init__(self, llm, specialist_agents):
        self.llm = llm
        self.specialist_agents = specialist_agents  # ['order', 'billing', 'returns', 'technical', 'product']

    def route(self, state: AgentState) -> AgentState:
        \"\"\"
        Decide which agent to route to

        Returns:
            Updated state with next_action set
        \"\"\"

        # Build routing prompt
        prompt = f\"\"\"You are a supervisor coordinating a customer support team.

Available specialist agents:
- order: Handle order tracking, status, history
- billing: Handle payments, refunds, invoices
- returns: Handle returns, exchanges, RMAs
- technical: Handle product issues, troubleshooting
- product: Handle product information, specifications

Conversation history:
{self.format_messages(state['messages'])}

Current query: {state['user_query']}

Based on the query, which agent should handle this? Or is the task complete?

Respond with JSON:
{{
    "next_agent": "agent_name or FINISH or HUMAN_ESCALATION",
    "reasoning": "why this agent"
}}\"\"\"

        # LLM decides routing
        response = self.llm.invoke(prompt)
        decision = json.loads(response.content)

        # Update state
        state['next_action'] = decision['next_agent']
        state['messages'].append(f"Supervisor: Routing to {decision['next_agent']} - {decision['reasoning']}")

        return state

    def format_messages(self, messages):
        return "\n".join(messages[-10:])  # Last 10 messages for context


class SpecialistAgent:
    \"\"\"
    Base class for specialist agents

    Each specialist:
    1. Has access to specific tools
    2. Uses ReAct pattern (Reasoning + Acting)
    3. Can request information from other agents
    \"\"\"

    def __init__(self, name: str, llm, tools: List[Tool]):
        self.name = name
        self.llm = llm
        self.tools = tools

        # Create ReAct agent
        self.agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=self.get_prompt()
        )

        self.executor = AgentExecutor(agent=self.agent, tools=tools)

    def get_prompt(self) -> PromptTemplate:
        \"\"\"Agent-specific prompt\"\"\"

        return PromptTemplate(
            input_variables=["input", "user_context", "tool_names", "tools", "agent_scratchpad"],
            template=\"\"\"You are a {name} specialist for customer support.

User Context:
{user_context}

Your tools:
{tools}

User Query:
{input}

Think step by step:
1. What information do I need?
2. Which tool should I use?
3. What is the answer?

{agent_scratchpad}\"\"\"
        )

    def execute(self, state: AgentState) -> AgentState:
        \"\"\"Execute agent with tools\"\"\"

        # Run agent
        result = self.executor.invoke({
            "input": state['user_query'],
            "user_context": state['user_context']
        })

        # Update state
        state['messages'].append(f"{self.name}: {result['output']}")

        # Check if approval needed (for billing agent)
        if self.name == "billing" and self.check_needs_approval(result):
            state['requires_approval'] = True
            state['approved'] = False  # Will wait for approval

        return state

    def check_needs_approval(self, result) -> bool:
        \"\"\"Check if action requires approval\"\"\"
        # Example: Refund > $100 needs approval
        if 'refund' in result['output'].lower():
            # Extract amount (simplified)
            import re
            match = re.search(r'\\$([0-9,]+)', result['output'])
            if match:
                amount = float(match.group(1).replace(',', ''))
                return amount > 100
        return False


# Define tools for each agent
class Tools:
    \"\"\"Tools available to agents\"\"\"

    @staticmethod
    def create_order_tools():
        return [
            Tool(
                name="get_order_status",
                func=lambda order_id: f"Order {order_id}: Shipped, arrives tomorrow",
                description="Get order status by order ID"
            ),
            Tool(
                name="get_order_history",
                func=lambda user_id: "Last 5 orders: [...]",
                description="Get user's order history"
            )
        ]

    @staticmethod
    def create_billing_tools():
        return [
            Tool(
                name="process_refund",
                func=lambda order_id, amount: f"Refund ${amount} processed for order {order_id}",
                description="Process refund (requires approval if > $100)"
            ),
            Tool(
                name="get_invoice",
                func=lambda order_id: "Invoice PDF URL",
                description="Get invoice for order"
            )
        ]

    @staticmethod
    def create_returns_tools():
        return [
            Tool(
                name="check_return_eligibility",
                func=lambda order_id: "Eligible for return (within 30 days)",
                description="Check if order is eligible for return"
            ),
            Tool(
                name="create_rma",
                func=lambda order_id: "RMA #12345 created. Print label: [URL]",
                description="Create return authorization"
            )
        ]


# Build LangGraph workflow
class CustomerSupportSystem:
    \"\"\"
    Multi-agent customer support system using LangGraph
    \"\"\"

    def __init__(self, llm):
        self.llm = llm

        # Create specialist agents
        self.agents = {
            'order': SpecialistAgent('order', llm, Tools.create_order_tools()),
            'billing': SpecialistAgent('billing', llm, Tools.create_billing_tools()),
            'returns': SpecialistAgent('returns', llm, Tools.create_returns_tools())
        }

        # Create supervisor
        self.supervisor = SupervisorAgent(llm, list(self.agents.keys()))

        # Build graph
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        \"\"\"
        Build execution graph

        Flow:
        User Query → Supervisor → Agent → Supervisor → ... → FINISH
        \"\"\"

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor.route)

        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, agent.execute)

        # Add approval node
        workflow.add_node("approval", self.approval_workflow)

        # Add edges (routing logic)
        workflow.add_conditional_edges(
            "supervisor",
            self.should_continue,
            {
                "order": "order",
                "billing": "billing",
                "returns": "returns",
                "approval": "approval",
                "FINISH": END
            }
        )

        # All agents return to supervisor
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "supervisor")

        # Approval returns to supervisor
        workflow.add_edge("approval", "supervisor")

        # Set entry point
        workflow.set_entry_point("supervisor")

        return workflow.compile()

    def should_continue(self, state: AgentState) -> str:
        \"\"\"
        Decide next step based on state

        Returns:
            Next node name
        \"\"\"

        # Check if approval needed
        if state.get('requires_approval') and not state.get('approved'):
            return "approval"

        # Check next action from supervisor
        next_action = state.get('next_action', '')

        if next_action == "FINISH":
            return "FINISH"
        elif next_action in self.agents:
            return next_action
        else:
            return "FINISH"

    def approval_workflow(self, state: AgentState) -> AgentState:
        \"\"\"
        Human approval for high-value actions

        In production:
        - Send notification to supervisor
        - Wait for approval in database
        - Timeout after 5 minutes
        \"\"\"

        # Simulate approval (in production, this would be async)
        print(f"⚠️  Approval required: {state['messages'][-1]}")
        approval = input("Approve? (yes/no): ")

        state['approved'] = approval.lower() == 'yes'
        state['requires_approval'] = False

        state['messages'].append(f"Approval: {'Granted' if state['approved'] else 'Denied'}")

        return state

    def run(self, user_query: str, user_context: dict = None):
        \"\"\"
        Run multi-agent system

        Args:
            user_query: User's question
            user_context: User info (order history, etc.)

        Returns:
            Final response
        \"\"\"

        # Initialize state
        initial_state = {
            "messages": [],
            "current_agent": "supervisor",
            "user_query": user_query,
            "user_context": user_context or {},
            "tool_outputs": {},
            "requires_approval": False,
            "approved": False,
            "next_action": ""
        }

        # Run graph
        final_state = self.graph.invoke(initial_state)

        # Extract final response
        return final_state['messages']


# Example usage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

system = CustomerSupportSystem(llm)

# Run conversation
response = system.run(
    user_query="I want to return my laptop (order #12345) and get a refund",
    user_context={
        "user_id": "user_789",
        "email": "user@example.com",
        "order_history": ["#12345 - Laptop $1299", "#11111 - Mouse $29"]
    }
)

# Output trace:
# Supervisor: Routing to returns - User wants to return an item
# Returns: Order #12345 is eligible for return. RMA #12345 created.
# Supervisor: Return approved, now process refund
# Billing: Refund $1299 requires approval
# ⚠️  Approval required: Refund $1299 for order #12345
# Approval: Granted
# Billing: Refund processed successfully
# Supervisor: Task complete
```

### Memory Management

**You:** "Multi-turn conversations need memory. Here's how:

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage

class MultiAgentMemory:
    \"\"\"
    Memory management for multi-agent systems

    Types:
    1. Short-term: Current conversation (last 10 turns)
    2. Long-term: User context (purchase history, preferences)
    3. Semantic: Vector store of past conversations
    \"\"\"

    def __init__(self, llm):
        # Short-term memory (conversation buffer)
        self.short_term = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            max_token_limit=2000  # Keep last ~10 turns
        )

        # Long-term memory (summarization)
        self.long_term = ConversationSummaryMemory(
            llm=llm,
            return_messages=True
        )

        # Semantic memory (vector store)
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings

        self.semantic = Chroma(
            embedding_function=OpenAIEmbeddings(),
            collection_name="conversation_memory"
        )

    def add_turn(self, user_msg: str, agent_msg: str, user_id: str):
        \"\"\"Add conversation turn to memory\"\"\"

        # Short-term
        self.short_term.save_context(
            {"input": user_msg},
            {"output": agent_msg}
        )

        # Long-term (summarize periodically)
        if len(self.short_term.chat_memory.messages) >= 10:
            summary = self.long_term.predict_new_summary(
                self.short_term.chat_memory.messages,
                ""
            )
            # Store summary
            self.store_summary(user_id, summary)

        # Semantic (for retrieval)
        self.semantic.add_texts(
            [f"User: {user_msg}\nAgent: {agent_msg}"],
            metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}]
        )

    def get_context(self, user_query: str, user_id: str) -> str:
        \"\"\"
        Get relevant context for new query

        Returns:
            Formatted context string
        \"\"\"

        # Short-term (recent conversation)
        recent = self.short_term.load_memory_variables({})

        # Semantic search (similar past conversations)
        similar = self.semantic.similarity_search(user_query, k=3)

        context = f\"\"\"
Recent conversation:
{recent['chat_history']}

Similar past conversations:
{self.format_similar(similar)}
\"\"\"

        return context

    def format_similar(self, docs):
        return "\n".join([doc.page_content for doc in docs])
```

### Safety & Guardrails

**You:** "Safety is critical in multi-agent systems:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class SafetyGuardrails:
    \"\"\"
    Safety checks for multi-agent systems

    Checks:
    1. Content safety (no harmful content)
    2. PII detection (don't expose sensitive data)
    3. Action validation (prevent unauthorized actions)
    4. Tool sandboxing (limit tool capabilities)
    \"\"\"

    def __init__(self, llm):
        self.llm = llm

        # Content moderation
        self.content_moderator = self.build_content_moderator()

        # PII detector
        self.pii_detector = self.build_pii_detector()

    def build_content_moderator(self):
        \"\"\"Check for harmful content\"\"\"

        prompt = PromptTemplate(
            input_variables=["text"],
            template=\"\"\"Check if this text contains harmful content:

Text: {text}

Categories to check:
- Hate speech
- Violence
- Self-harm
- Sexual content
- Illegal activity

Respond with JSON:
{{
    "is_safe": true/false,
    "categories": ["category1", ...],
    "confidence": 0-1
}}\"\"\"
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def build_pii_detector(self):
        \"\"\"Detect PII in text\"\"\"

        # Use regex + NER model
        import re
        from presidio_analyzer import AnalyzerEngine

        return AnalyzerEngine()

    def check_safety(self, text: str) -> dict:
        \"\"\"
        Run all safety checks

        Returns:
            {
                'is_safe': bool,
                'violations': [],
                'redacted_text': str
            }
        \"\"\"

        violations = []

        # 1. Content safety
        content_result = self.content_moderator.run(text)
        content_check = json.loads(content_result)

        if not content_check['is_safe']:
            violations.extend(content_check['categories'])

        # 2. PII detection
        pii_results = self.pii_detector.analyze(
            text=text,
            language='en'
        )

        if pii_results:
            violations.append('pii_detected')

            # Redact PII
            from presidio_anonymizer import AnonymizerEngine
            anonymizer = AnonymizerEngine()

            redacted = anonymizer.anonymize(
                text=text,
                analyzer_results=pii_results
            )
            redacted_text = redacted.text
        else:
            redacted_text = text

        return {
            'is_safe': len(violations) == 0,
            'violations': violations,
            'redacted_text': redacted_text
        }

    def validate_action(self, agent_name: str, action: str, params: dict) -> bool:
        \"\"\"
        Validate if agent is allowed to perform action

        Examples:
        - Billing agent can process refunds
        - Technical agent CANNOT process refunds
        - All agents can query databases
        \"\"\"

        permissions = {
            'order': ['get_order_status', 'get_order_history'],
            'billing': ['process_refund', 'get_invoice', 'charge_payment'],
            'returns': ['check_return_eligibility', 'create_rma'],
            'technical': ['get_product_info', 'get_troubleshooting_steps'],
            'product': ['get_product_info', 'get_specifications']
        }

        allowed_actions = permissions.get(agent_name, [])

        return action in allowed_actions
```

### Observability with LangSmith

**You:** "Debugging multi-agent systems is hard. LangSmith (2025 standard) helps:

```python
from langsmith import Client
from langsmith.run_helpers import traceable

class ObservabilityLayer:
    \"\"\"
    Tracing and monitoring for multi-agent systems
    \"\"\"

    def __init__(self):
        self.client = Client()

    @traceable(name="multi_agent_execution")
    def trace_execution(self, conversation_id: str, state: AgentState):
        \"\"\"
        Trace multi-agent execution

        LangSmith captures:
        - Agent transitions
        - Tool calls
        - LLM prompts and responses
        - Latency per step
        - Errors
        \"\"\"

        # This is automatically traced by LangSmith
        # No manual instrumentation needed!

        pass

    def log_metrics(self, conversation_id: str, metrics: dict):
        \"\"\"
        Log custom metrics

        Metrics:
        - Total latency
        - Number of agent switches
        - Number of tool calls
        - LLM tokens used
        - Cost
        \"\"\"

        self.client.create_feedback(
            run_id=conversation_id,
            key="metrics",
            score=metrics.get('user_satisfaction', 0),
            value=metrics
        )

    def analyze_failures(self):
        \"\"\"
        Analyze failed conversations

        Find:
        - Which agents failed most?
        - Which tools errored?
        - Where did timeouts occur?
        \"\"\"

        # Query LangSmith for failed runs
        failed_runs = self.client.list_runs(
            project_name="customer_support",
            filter="error == true",
            limit=100
        )

        # Analyze patterns
        error_patterns = {}
        for run in failed_runs:
            error_type = run.error_type
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

        return error_patterns
```

---

## Phase 4: Trade-offs & Framework Comparison (5-8 minutes)

**Interviewer:** "What are the trade-offs of multi-agent systems vs single-agent? And which framework should we use?"

### Multi-Agent vs Single-Agent

**You:** "Let me compare approaches:

| Criterion | Single Agent | Multi-Agent |
|-----------|--------------|-------------|
| **Complexity** | Simple to build and debug | Complex orchestration |
| **Specialization** | Generalist (jack of all trades) | Specialists (expert per domain) |
| **Scalability** | Limited by single LLM context | Scales with specialized agents |
| **Latency** | Faster (one LLM call) | Slower (multiple agents) |
| **Cost** | Lower (fewer LLM calls) | Higher (more calls for coordination) |
| **Accuracy** | Good for simple tasks | Better for complex, multi-domain |
| **Maintainability** | Easy to update prompt | Need to update multiple agents |
| **When to Use** | Simple Q&A, single-domain | Complex workflows, multi-domain |

**Decision Rule:**

```python
def should_use_multi_agent(task_characteristics):
    score = 0

    # Indicators for multi-agent
    if task_characteristics['multiple_domains']:  # e.g., billing + technical
        score += 3
    if task_characteristics['complex_workflow']:  # Multi-step with dependencies
        score += 2
    if task_characteristics['requires_specialized_knowledge']:
        score += 2
    if task_characteristics['tool_use_heavy']:  # Many different APIs/tools
        score += 1

    # Indicators against multi-agent
    if task_characteristics['latency_critical']:  # <1 second required
        score -= 2
    if task_characteristics['simple_qa']:  # Just FAQs
        score -= 3

    return score >= 4
```

### Framework Comparison (2025)

| Framework | Best For | Pros | Cons |
|-----------|----------|------|------|
| **LangGraph** | Complex workflows, stateful | Graph abstraction, great debugging | Learning curve |
| **AutoGen** | Research, experiments | Easy multi-agent conversations | Less production-ready |
| **CrewAI** | Simple hierarchical teams | Intuitive API, good defaults | Limited customization |
| **LangChain** | Simple chains | Mature, many integrations | Not designed for multi-agent |
| **Custom** | Full control needed | Total flexibility | High development cost |

**Our Choice: LangGraph**
- Industry standard in 2025
- Excellent observability (LangSmith integration)
- Production-ready
- Flexible for complex workflows

---

## Phase 5: Production Metrics & Interview Guidance

### Real Production Metrics (Multi-Agent Systems 2025)

**Scale:**
- Complex tasks: 100-1000 queries/day per enterprise
- Agents per task: 3-10 specialized agents
- Task completion time: 30 seconds to 5 minutes
- Success rate: 85-95% for well-scoped tasks

**Cost Analysis (GPT-4 based system):**
- Average task: 5 agents × 3 interactions × 2K tokens = 30K tokens
- Cost per task: $0.90-1.20 (GPT-4) vs $0.10-0.15 (Claude)
- At 1000 tasks/month: $900-1200 (GPT-4) vs $100-150 (Claude)

**Coordination Overhead:**
- Sequential execution: 5 agents × 3s = 15s
- Parallel execution: max(3s, 3s, 3s) = 3s (5x speedup)
- Communication overhead: 10-20% additional tokens

**Quality Metrics:**
- Task completion rate: >90%
- Agent agreement (consensus): >85%
- Hallucination detection: 3 agents voting reduces errors by 60%

### Interview Success Tips

**Strong candidates discuss:**
- When to use multiple agents vs single agent
- Agent specialization and task decomposition
- Communication protocols between agents
- Orchestration patterns (centralized vs decentralized)
- Cost-benefit analysis of agent coordination overhead

**Mistake:** "We'll use 10 agents for everything"
**Better:** "Agent count depends on task complexity. Simple Q&A: 1 agent. Research task: 3-5 agents (search, analyze, synthesize). Code generation: 4 agents (planner, coder, reviewer, tester)"

**Q:** "When should you NOT use multi-agent systems?"
**A:** "When: 1) Task is simple (single agent faster/cheaper), 2) Real-time latency critical (<1s), 3) Agent coordination overhead > benefits, 4) Budget-constrained (agents multiply costs 3-10x)"

**Q:** "How do agents reach consensus?"
**A:** "Voting mechanisms: 1) Majority voting (3+ agents), 2) Confidence-weighted voting, 3) Debate (agents critique each other), 4) Judge agent makes final decision, 5) Human-in-the-loop for critical decisions"

### Production Considerations

**Failure Handling:**
- Agent timeout: Fall back to single-agent mode
- Agent disagreement: Escalate to human or judge agent
- Infinite loops: Max iteration limits (10-20)
- Cost explosions: Budget caps per task ($5 max)

**Monitoring:**
- Track: Agent utilization, task completion time, cost per task, failure patterns
- Alert if: Success rate <80%, cost >2× baseline, latency >5 minutes

---

## Summary & Key Takeaways

**You:** "To summarize the Multi-Agent AI System:

### Architecture Highlights

1. **Supervisor Pattern:** Central orchestrator routes to specialists
2. **Tool Calling:** Agents use external APIs, databases
3. **State Management:** LangGraph state for agent communication
4. **Safety Guardrails:** Content moderation, PII detection, action validation
5. **Observability:** LangSmith tracing for debugging

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Supervisor pattern | Centralized control, easier debugging |
| LangGraph | 2025 standard, production-ready |
| ReAct agents | Reasoning + acting, interpretable |
| Approval workflow | Safety for high-value actions |
| LangSmith observability | Essential for debugging complex flows |

### Production Metrics

- **Latency:** <10s per response (Routing: 1s, Agent: 5s, Tools: 3s, Generation: 1s)
- **Accuracy:** 85% resolution without human escalation
- **Cost:** ~$0.20 per conversation (avg 3 agent calls)
- **Scale:** 10K concurrent conversations

### Common Patterns

1. **Supervisor → Worker:** Our customer support system
2. **Sequential Chain:** Document processing (parse → analyze → summarize)
3. **Hierarchical:** Enterprise workflows (manager → team → specialists)
4. **Collaborative:** Multiple agents debate to reach consensus

This design demonstrates:
- Multi-agent orchestration patterns
- Tool use and safety
- Production considerations (latency, cost, observability)
- 2025 best practices (LangGraph, LangSmith)"

---

## Staff-Level Deep Dives

### Agent Coordination Protocols & Communication

**Interviewer:** "How do agents communicate and coordinate efficiently in complex workflows?"

**You:** "Agent coordination is critical for multi-agent systems. Let me detail the communication protocols:

#### Inter-Agent Communication Patterns

```python
class AgentCommunicationProtocol:
    """
    Define how agents communicate with each other

    Patterns:
    1. Message Passing (async, decoupled)
    2. Shared State (LangGraph state)
    3. Event Broadcasting (pub-sub)
    4. Direct Invocation (RPC-style)
    """

    def __init__(self):
        self.message_queue = MessageQueue()
        self.shared_state = SharedState()
        self.event_bus = EventBus()

    def send_message(self, from_agent: str, to_agent: str, message: dict):
        """
        Message passing: Agent A sends message to Agent B

        Use case: Order agent tells billing agent to process refund

        Advantages:
        - Asynchronous (non-blocking)
        - Decoupled (agents don't need to know about each other)
        - Scalable (messages can be queued)

        Disadvantages:
        - Eventual consistency (not immediate)
        - Requires message queue infrastructure
        """

        envelope = {
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now(),
            'message_id': str(uuid.uuid4())
        }

        # Add to queue
        self.message_queue.enqueue(to_agent, envelope)

        return envelope['message_id']

    def broadcast_event(self, event_type: str, payload: dict):
        """
        Event broadcasting: Agent publishes event, subscribers react

        Use case: Order agent broadcasts "order_cancelled" event
                 → Billing agent listens and triggers refund
                 → Inventory agent listens and restocks

        Advantages:
        - Loosely coupled (publishers don't know subscribers)
        - Scalable (add subscribers without changing publisher)
        - Event sourcing (full audit trail)

        Disadvantages:
        - Complex debugging (event flow hard to trace)
        - Potential event storms (cascading events)
        """

        event = {
            'type': event_type,
            'payload': payload,
            'timestamp': datetime.now(),
            'event_id': str(uuid.uuid4())
        }

        # Publish to all subscribers
        self.event_bus.publish(event_type, event)

        return event['event_id']

    def update_shared_state(self, agent_id: str, updates: dict):
        """
        Shared state: All agents read/write to common state

        Use case: LangGraph state with conversation history

        Advantages:
        - Simple (all agents see same state)
        - Immediate consistency
        - Easy to debug (single source of truth)

        Disadvantages:
        - Tight coupling (agents depend on state schema)
        - Concurrency issues (race conditions)
        - Not scalable (single state bottleneck)
        """

        with self.shared_state.lock:
            # Atomic update
            for key, value in updates.items():
                self.shared_state.set(agent_id, key, value)

        return self.shared_state.get_snapshot()

    def invoke_agent(self, from_agent: str, target_agent: str, request: dict):
        """
        Direct invocation: Agent A calls Agent B synchronously

        Use case: Supervisor invokes specialist agent

        Advantages:
        - Simple (just function call)
        - Immediate response
        - Easy error handling

        Disadvantages:
        - Tight coupling (caller waits for callee)
        - Not scalable (blocking)
        - Failure propagation (one agent fails → all fail)
        """

        # Find target agent
        agent = self.get_agent(target_agent)

        # Invoke directly
        try:
            response = agent.execute(request)
            return {'status': 'success', 'response': response}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class MessageQueue:
    """
    Message queue for async agent communication

    Implementation: Redis Streams or RabbitMQ
    """

    def __init__(self):
        # Redis Streams for message queue
        import redis
        self.redis = redis.Redis(host='localhost', port=6379)

    def enqueue(self, agent_id: str, message: dict):
        """Add message to agent's queue"""

        queue_name = f"agent:{agent_id}:queue"

        # Add to Redis stream
        self.redis.xadd(
            queue_name,
            {
                'from': message['from'],
                'message': json.dumps(message['message']),
                'timestamp': message['timestamp'].isoformat()
            }
        )

    def dequeue(self, agent_id: str, block_ms: int = 1000):
        """Get next message from agent's queue"""

        queue_name = f"agent:{agent_id}:queue"

        # Read from Redis stream
        messages = self.redis.xread(
            {queue_name: '0'},  # Read from beginning
            count=1,
            block=block_ms
        )

        if messages:
            stream_name, message_list = messages[0]
            message_id, fields = message_list[0]

            return {
                'from': fields[b'from'].decode('utf-8'),
                'message': json.loads(fields[b'message']),
                'timestamp': datetime.fromisoformat(fields[b'timestamp'].decode('utf-8'))
            }

        return None


class EventBus:
    """
    Pub-sub event bus for agent communication

    Implementation: Redis Pub/Sub or Kafka
    """

    def __init__(self):
        import redis
        self.redis = redis.Redis(host='localhost', port=6379)
        self.pubsub = self.redis.pubsub()
        self.subscribers = {}  # event_type → [agent_ids]

    def subscribe(self, agent_id: str, event_types: List[str]):
        """Agent subscribes to event types"""

        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []

            self.subscribers[event_type].append(agent_id)

            # Subscribe to Redis channel
            channel = f"events:{event_type}"
            self.pubsub.subscribe(channel)

    def publish(self, event_type: str, event: dict):
        """Publish event to all subscribers"""

        channel = f"events:{event_type}"

        # Publish to Redis
        self.redis.publish(
            channel,
            json.dumps(event)
        )

        # Notify subscribers
        if event_type in self.subscribers:
            for agent_id in self.subscribers[event_type]:
                print(f"Notifying {agent_id} of event {event_type}")

    def listen(self, agent_id: str):
        """Listen for events (blocking)"""

        for message in self.pubsub.listen():
            if message['type'] == 'message':
                event = json.loads(message['data'])
                yield event
```

### Consensus Mechanisms

**Interviewer:** "How do multiple agents reach consensus when they disagree?"

**You:** "Consensus is critical when multiple agents provide conflicting answers:

#### Multi-Agent Consensus Strategies

```python
class ConsensusManager:
    """
    Manage consensus among multiple agents

    Strategies:
    1. Majority voting
    2. Confidence-weighted voting
    3. Debate and judge
    4. Mixture of experts
    5. Human-in-the-loop (final arbiter)
    """

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.voting_history = []

    def majority_vote(self, agent_responses: Dict[str, str]) -> str:
        """
        Simple majority voting

        Example:
        - Agent A: "Answer is X"
        - Agent B: "Answer is X"
        - Agent C: "Answer is Y"
        - Result: "Answer is X" (2 vs 1)

        Use case: Multiple agents verify same fact
        """

        from collections import Counter

        # Count votes
        votes = Counter(agent_responses.values())

        # Get majority
        majority_answer, count = votes.most_common(1)[0]

        # Require >50% for consensus
        if count > len(agent_responses) / 2:
            return {
                'answer': majority_answer,
                'confidence': count / len(agent_responses),
                'method': 'majority_vote'
            }
        else:
            return {
                'answer': None,
                'confidence': 0,
                'method': 'no_consensus'
            }

    def confidence_weighted_vote(self, agent_responses: Dict[str, dict]) -> str:
        """
        Weighted voting based on agent confidence

        Example:
        - Agent A: "Answer is X" (confidence: 0.9)
        - Agent B: "Answer is X" (confidence: 0.7)
        - Agent C: "Answer is Y" (confidence: 0.5)
        - Result: "Answer is X" (weighted score: 1.6 vs 0.5)

        Use case: Agents have varying expertise
        """

        answer_scores = {}

        for agent_id, response in agent_responses.items():
            answer = response['answer']
            confidence = response.get('confidence', 1.0)

            if answer not in answer_scores:
                answer_scores[answer] = 0

            answer_scores[answer] += confidence

        # Get highest weighted answer
        best_answer = max(answer_scores, key=answer_scores.get)

        total_weight = sum(answer_scores.values())

        return {
            'answer': best_answer,
            'confidence': answer_scores[best_answer] / total_weight,
            'method': 'confidence_weighted_vote',
            'scores': answer_scores
        }

    def debate_and_judge(self,
                        initial_responses: Dict[str, str],
                        judge_agent,
                        max_rounds: int = 3):
        """
        Agents debate, judge makes final decision

        Process:
        1. Each agent provides initial answer
        2. Agents critique each other's answers
        3. Agents revise based on critiques
        4. Repeat for max_rounds
        5. Judge agent makes final decision

        Example:
        Round 1:
        - Agent A: "Capital of France is Paris"
        - Agent B: "Capital of France is Lyon"

        Round 2 (critique):
        - Agent A: "Lyon is second-largest city, not capital"
        - Agent B: "You're right, Paris is the capital"

        Judge: "Both now agree, answer is Paris"

        Use case: Complex questions with nuance
        """

        debate_history = []
        current_answers = initial_responses

        for round_num in range(max_rounds):
            # Agents critique each other
            critiques = {}

            for agent_id, answer in current_answers.items():
                # Get other agents' answers
                other_answers = {
                    k: v for k, v in current_answers.items()
                    if k != agent_id
                }

                # Agent critiques others
                critique = self.generate_critique(
                    agent_id,
                    answer,
                    other_answers
                )

                critiques[agent_id] = critique

            # Agents revise based on critiques
            revised_answers = {}

            for agent_id, answer in current_answers.items():
                # Get critiques of this answer
                relevant_critiques = [
                    c for aid, c in critiques.items()
                    if aid != agent_id
                ]

                # Revise answer
                revised = self.revise_answer(
                    agent_id,
                    answer,
                    relevant_critiques
                )

                revised_answers[agent_id] = revised

            debate_history.append({
                'round': round_num,
                'answers': current_answers,
                'critiques': critiques,
                'revised': revised_answers
            })

            current_answers = revised_answers

            # Check if consensus reached
            if len(set(current_answers.values())) == 1:
                # All agents agree
                return {
                    'answer': list(current_answers.values())[0],
                    'confidence': 1.0,
                    'method': 'debate_consensus',
                    'rounds': round_num + 1,
                    'debate_history': debate_history
                }

        # No consensus, ask judge
        judge_decision = judge_agent.decide(
            debate_history=debate_history,
            final_answers=current_answers
        )

        return {
            'answer': judge_decision['answer'],
            'confidence': judge_decision['confidence'],
            'method': 'judge_decision',
            'rounds': max_rounds,
            'debate_history': debate_history,
            'judge_reasoning': judge_decision['reasoning']
        }

    def mixture_of_experts(self, agent_responses: Dict[str, dict], query: str):
        """
        Combine answers from multiple expert agents

        Use case: Different agents are experts in different domains

        Example:
        Query: "What's the capital and GDP of France?"

        Agent A (Geography expert): "Capital is Paris"
        Agent B (Economics expert): "GDP is $2.7T"

        Combined: "Capital is Paris, GDP is $2.7T"

        Strategy: Use all answers, each agent contributes their expertise
        """

        # Classify query to determine which agents are relevant
        relevant_agents = self.identify_relevant_agents(query)

        combined_answer = {}

        for agent_id, response in agent_responses.items():
            if agent_id in relevant_agents:
                # Weight by agent's expertise for this query
                expertise_weight = relevant_agents[agent_id]

                combined_answer[agent_id] = {
                    'answer': response['answer'],
                    'weight': expertise_weight
                }

        # Merge answers
        final_answer = self.merge_expert_answers(combined_answer)

        return {
            'answer': final_answer,
            'method': 'mixture_of_experts',
            'expert_contributions': combined_answer
        }

    def human_in_the_loop(self, agent_responses: Dict[str, str]):
        """
        Escalate to human when agents can't reach consensus

        Use case: High-stakes decisions, agents disagree significantly

        Flow:
        1. Agents provide answers
        2. No consensus reached
        3. Present to human reviewer
        4. Human makes final decision
        5. Log decision for future training
        """

        # Check if consensus possible
        if len(set(agent_responses.values())) > 3:
            # Too many different answers, escalate

            human_review_request = {
                'agent_responses': agent_responses,
                'request_time': datetime.now(),
                'priority': 'high'
            }

            # Send to human review queue
            human_decision = self.request_human_review(human_review_request)

            # Log for future training
            self.log_human_decision(agent_responses, human_decision)

            return {
                'answer': human_decision['answer'],
                'confidence': 1.0,  # Human decisions are authoritative
                'method': 'human_in_the_loop',
                'human_reasoning': human_decision['reasoning']
            }
```

### Failure Recovery & Circuit Breakers

**Interviewer:** "What happens when an agent fails or times out?"

**You:** "Agent failures are common in production. Let me explain our recovery strategies:

#### Failure Recovery System

```python
class AgentFailureRecovery:
    """
    Handle agent failures gracefully

    Failure modes:
    1. Agent timeout (takes too long)
    2. Agent error (exception thrown)
    3. Tool failure (API down)
    4. Invalid response (hallucination)
    5. Resource exhaustion (rate limits)
    """

    def __init__(self):
        self.circuit_breakers = {}
        self.fallback_agents = {}
        self.retry_policies = {}

    def execute_with_recovery(self, agent_id: str, task: dict):
        """
        Execute agent with failure recovery

        Recovery strategies:
        1. Retry with exponential backoff
        2. Circuit breaker (stop calling failing agent)
        3. Fallback to alternative agent
        4. Degrade gracefully (partial result)
        5. Escalate to human
        """

        # Check circuit breaker
        if self.is_circuit_open(agent_id):
            # Circuit is open, use fallback
            return self.execute_fallback(agent_id, task)

        # Execute with timeout
        try:
            result = self.execute_with_timeout(
                agent_id,
                task,
                timeout_seconds=30
            )

            # Success, close circuit if it was half-open
            self.record_success(agent_id)

            return result

        except TimeoutException:
            # Agent timed out
            self.record_failure(agent_id, 'timeout')

            # Retry with exponential backoff
            for attempt in range(3):
                wait_time = 2 ** attempt  # 1s, 2s, 4s

                time.sleep(wait_time)

                try:
                    result = self.execute_with_timeout(
                        agent_id,
                        task,
                        timeout_seconds=30
                    )

                    self.record_success(agent_id)
                    return result

                except TimeoutException:
                    continue

            # All retries failed, open circuit
            self.open_circuit(agent_id)

            # Use fallback
            return self.execute_fallback(agent_id, task)

        except AgentException as e:
            # Agent threw error
            self.record_failure(agent_id, 'error')

            # Don't retry on errors, use fallback immediately
            return self.execute_fallback(agent_id, task)

    def execute_with_timeout(self, agent_id: str, task: dict, timeout_seconds: int):
        """
        Execute agent with timeout

        Uses threading to enforce timeout
        """

        import threading

        result = None
        exception = None

        def run_agent():
            nonlocal result, exception
            try:
                agent = self.get_agent(agent_id)
                result = agent.execute(task)
            except Exception as e:
                exception = e

        # Run in thread with timeout
        thread = threading.Thread(target=run_agent)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Timeout
            raise TimeoutException(f"Agent {agent_id} timed out after {timeout_seconds}s")

        if exception:
            raise AgentException(f"Agent {agent_id} failed: {exception}")

        return result

    def execute_fallback(self, failed_agent_id: str, task: dict):
        """
        Execute fallback strategy when agent fails

        Fallback options:
        1. Alternative agent (if available)
        2. Cached response (if task seen before)
        3. Default/safe response
        4. Human escalation
        """

        # Option 1: Alternative agent
        if failed_agent_id in self.fallback_agents:
            fallback_id = self.fallback_agents[failed_agent_id]

            try:
                result = self.execute_with_timeout(
                    fallback_id,
                    task,
                    timeout_seconds=30
                )

                return {
                    'result': result,
                    'fallback_used': True,
                    'fallback_agent': fallback_id
                }

            except Exception:
                # Fallback also failed, continue to next option
                pass

        # Option 2: Cached response
        cached = self.get_cached_response(task)
        if cached:
            return {
                'result': cached,
                'fallback_used': True,
                'fallback_method': 'cache'
            }

        # Option 3: Default response
        default = self.get_default_response(failed_agent_id, task)
        if default:
            return {
                'result': default,
                'fallback_used': True,
                'fallback_method': 'default'
            }

        # Option 4: Human escalation
        return {
            'result': None,
            'fallback_used': True,
            'fallback_method': 'human_escalation',
            'escalation_reason': f'Agent {failed_agent_id} failed and no fallback available'
        }

    def is_circuit_open(self, agent_id: str) -> bool:
        """
        Check if circuit breaker is open for agent

        Circuit states:
        - CLOSED: Normal operation
        - OPEN: Agent is failing, don't call it
        - HALF_OPEN: Testing if agent recovered
        """

        if agent_id not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[agent_id]

        if breaker['state'] == 'CLOSED':
            return False

        elif breaker['state'] == 'OPEN':
            # Check if enough time has passed to try again
            time_since_open = (datetime.now() - breaker['opened_at']).seconds

            if time_since_open > breaker['timeout_seconds']:
                # Transition to HALF_OPEN
                breaker['state'] = 'HALF_OPEN'
                return False  # Allow one test call

            return True  # Still open

        elif breaker['state'] == 'HALF_OPEN':
            return False  # Allow test calls

    def record_failure(self, agent_id: str, failure_type: str):
        """Record agent failure for circuit breaker"""

        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                'state': 'CLOSED',
                'failure_count': 0,
                'success_count': 0,
                'threshold': 5,  # Open after 5 failures
                'timeout_seconds': 60  # Try again after 60s
            }

        breaker = self.circuit_breakers[agent_id]
        breaker['failure_count'] += 1

        # Check if should open circuit
        if breaker['failure_count'] >= breaker['threshold']:
            self.open_circuit(agent_id)

    def open_circuit(self, agent_id: str):
        """Open circuit breaker for agent"""

        breaker = self.circuit_breakers[agent_id]
        breaker['state'] = 'OPEN'
        breaker['opened_at'] = datetime.now()

        # Alert
        print(f"ALERT: Circuit breaker OPEN for agent {agent_id}")

    def record_success(self, agent_id: str):
        """Record agent success"""

        if agent_id not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[agent_id]
        breaker['success_count'] += 1

        # If in HALF_OPEN, close circuit after success
        if breaker['state'] == 'HALF_OPEN':
            breaker['state'] = 'CLOSED'
            breaker['failure_count'] = 0
            print(f"Circuit breaker CLOSED for agent {agent_id}")
```

### Dynamic Agent Selection

**Interviewer:** "How do you decide which agents to use for a given task?"

**You:** "Dynamic agent selection optimizes cost and latency:

```python
class DynamicAgentSelector:
    """
    Intelligently select agents based on task requirements

    Factors:
    1. Task complexity
    2. Agent capability match
    3. Agent cost
    4. Agent latency
    5. Agent current load
    """

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.agent_stats = {}  # Track performance

    def select_agents(self, task: dict) -> List[str]:
        """
        Select optimal agents for task

        Strategy:
        1. Classify task complexity
        2. Match to agent capabilities
        3. Optimize for cost + latency + quality
        """

        # Classify task
        complexity = self.classify_task_complexity(task)

        # Get candidate agents
        candidates = self.get_capable_agents(task)

        # Score agents
        scored_agents = []

        for agent_id in candidates:
            agent = self.agents[agent_id]
            stats = self.agent_stats.get(agent_id, {})

            # Compute score
            score = self.compute_agent_score(
                agent,
                task,
                complexity,
                stats
            )

            scored_agents.append((agent_id, score))

        # Select top agents
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # Select number of agents based on complexity
        if complexity == 'simple':
            num_agents = 1
        elif complexity == 'medium':
            num_agents = 2
        else:  # complex
            num_agents = 3

        selected = [agent_id for agent_id, score in scored_agents[:num_agents]]

        return selected

    def compute_agent_score(self,
                           agent: Agent,
                           task: dict,
                           complexity: str,
                           stats: dict) -> float:
        """
        Compute agent fitness score for task

        Score = Quality × Speed × (1/Cost) × Availability

        Where:
        - Quality: Agent's success rate on similar tasks
        - Speed: 1 / avg_latency
        - Cost: 1 / cost_per_task
        - Availability: 1 - current_load
        """

        # Quality (0-1)
        quality = stats.get('success_rate', 0.5)

        # Speed (normalized)
        avg_latency = stats.get('avg_latency_seconds', 5.0)
        speed = 1.0 / (1.0 + avg_latency)  # 0-1, higher is better

        # Cost (normalized)
        cost_per_task = stats.get('cost_per_task', 0.10)
        cost_score = 1.0 / (1.0 + cost_per_task * 10)  # 0-1, higher is better

        # Availability (0-1)
        current_load = stats.get('current_load', 0.0)
        availability = 1.0 - current_load

        # Combine (weighted)
        score = (
            0.4 * quality +
            0.2 * speed +
            0.2 * cost_score +
            0.2 * availability
        )

        return score
```

### Long-Term Maintenance

**You:** "Multi-agent systems require continuous maintenance:

#### Agent Retraining Schedule

```
Specialist Agents:
- Retrain: Bi-weekly (user queries evolve)
- Training data: Last 30 days of conversations
- Validation: Offline accuracy + online A/B test
- Deployment: Canary rollout (5% → 25% → 100%)

Supervisor Agent:
- Retrain: Monthly (routing patterns stable)
- Training data: Agent routing decisions + outcomes
- Validation: Routing accuracy (90%+ correct agent)

Tool Calling Models:
- Retrain: Weekly (API schemas change)
- Training data: Tool call examples + results
- Validation: Tool call success rate

Safety Guardrails:
- Retrain: Daily (adversarial attacks evolve)
- Training data: Flagged content + human reviews
- Validation: Precision/recall on test set
```

#### When to Re-architect

```
Signals for major changes:

1. Coordination overhead >50% of total cost
   → Consider: Consolidate agents, use single larger model

2. Consensus failure rate >20%
   → Improve: Better agent specialization, judge agent

3. Human escalation rate >30%
   → Improve: Agent capabilities, better training data

4. Average conversation requires >5 agents
   → Simplify: Agent consolidation, better routing

5. Latency SLO violated (>10s for >7 days)
   → Optimize: Agent parallelization, model distillation
```

### Organizational Context

**You:** "Multi-agent AI systems require specialized team:

#### Team Structure

```
Core Multi-Agent Team (12 engineers):
- 4 AI Engineers: Agent development, prompt engineering
- 3 Backend Engineers: Orchestration, state management
- 2 ML Infra Engineers: Model serving, LangGraph infrastructure
- 2 Data Engineers: Training data pipelines, conversation logs
- 1 Staff Engineer: Architecture, agent patterns

Partner Teams:
- Safety Team: Guardrails, content moderation
- Product: Agent capabilities, user experience
- Data Science: Evaluation metrics, A/B testing

On-Call:
- 24/7 on-call (customer support critical)
- Rotation: 1 week shifts, 4 engineers
- Escalation: Staff engineer → EM → Director
```

#### Cross-Functional Trade-offs

**You:** "Common conflicts:

**Product wants:**
- 10 specialized agents (comprehensive coverage)
- Solution: Better coverage, higher success rate

**Engineering concerns:**
- Coordination complexity, higher latency, 10× cost
- Solution: Fewer agents means faster, cheaper

**Resolution:**
- Analysis: 5 agents vs 10 agents A/B test
- Results: 5 agents = 85% success rate, avg 3s latency, $0.20/conv
- Results: 10 agents = 87% success rate, avg 7s latency, $0.60/conv
- Decision: 5 agents (2% quality loss not worth 3× cost + 2× latency)
- Compromise: Add 6th agent for high-value use cases only

**Safety wants:**
- Human approval for all refunds
- Solution: Maximum safety, no fraud

**Business wants:**
- Instant refunds (better UX)
- Solution: Higher customer satisfaction

**Resolution:**
- Risk-based approval: <$50 instant, $50-$100 auto with review, >$100 human approval
- Impact: 80% refunds instant, fraud rate <0.5%
- Decision: Balanced approach wins

---

## Sources

- [Multi-agent - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [How and when to build multi-agent systems](https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/)
- [LangChain & Multi-Agent AI in 2025: Framework, Tools & Use Cases](https://blogs.infoservices.com/artificial-intelligence/langchain-multi-agent-ai-framework-2025/)
- [Agent Orchestration: When to Use LangChain, LangGraph, AutoGen](https://medium.com/@akankshasinha247/agent-orchestration-when-to-use-langchain-langgraph-autogen-or-build-an-agentic-rag-system-cc298f785ea4)
- [LangChain State of AI Agents Report](https://www.langchain.com/stateofaiagents)
- [LangGraph: Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
