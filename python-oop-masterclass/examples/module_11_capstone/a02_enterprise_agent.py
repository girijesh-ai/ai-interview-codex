"""
Enterprise Agentic AI Example (Capstone)
=========================================
Demonstrates 2025 enterprise patterns:
- Multi-agent orchestration (Manager pattern)
- Memory-augmented state
- Human-in-the-Loop (HITL) controls
- Tool risk classification

Run demo: python a02_enterprise_agent.py
Run tests: pytest test_enterprise_agent.py -v
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import uuid
import asyncio


# ==============================================================================
# DOMAIN MODELS
# ==============================================================================

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    EXECUTOR = "executor"


class ActionRisk(str, Enum):
    LOW = "low"       # Auto-execute
    MEDIUM = "medium" # Log and proceed
    HIGH = "high"     # Require approval


@dataclass
class AgentMessage:
    role: AgentRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True
    risk_level: ActionRisk = ActionRisk.LOW


@dataclass
class AgentState:
    """Persistent state across conversation turns."""
    session_id: str
    messages: List[AgentMessage] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    pending_approval: Optional[ToolCall] = None
    completed: bool = False


# ==============================================================================
# MEMORY (Strategy Pattern)
# ==============================================================================

class Memory(ABC):
    @abstractmethod
    async def save_state(self, state: AgentState) -> None:
        pass
    
    @abstractmethod
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        pass


class InMemoryStore(Memory):
    def __init__(self):
        self._states: Dict[str, AgentState] = {}
    
    async def save_state(self, state: AgentState) -> None:
        self._states[state.session_id] = state
    
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        return self._states.get(session_id)


# ==============================================================================
# TOOLS WITH RISK LEVELS
# ==============================================================================

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def risk_level(self) -> ActionRisk:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        pass


class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Search for information"
    
    @property
    def risk_level(self) -> ActionRisk:
        return ActionRisk.LOW
    
    async def execute(self, query: str) -> str:
        return f"Search results for: {query}"


class DatabaseTool(Tool):
    @property
    def name(self) -> str:
        return "query_database"
    
    @property
    def description(self) -> str:
        return "Query customer database"
    
    @property
    def risk_level(self) -> ActionRisk:
        return ActionRisk.MEDIUM
    
    async def execute(self, sql: str) -> str:
        return f"DB results: [records for {sql}]"


class EmailTool(Tool):
    @property
    def name(self) -> str:
        return "send_email"
    
    @property
    def description(self) -> str:
        return "Send email (requires approval)"
    
    @property
    def risk_level(self) -> ActionRisk:
        return ActionRisk.HIGH
    
    async def execute(self, to: str, subject: str, body: str) -> str:
        return f"Email sent to {to}: {subject}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        return self._tools[name]
    
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


# ==============================================================================
# SPECIALIST AGENTS
# ==============================================================================

class Agent(ABC):
    @property
    @abstractmethod
    def role(self) -> AgentRole:
        pass
    
    @abstractmethod
    async def process(self, task: str, state: AgentState, llm: Callable) -> str:
        pass


class ResearchAgent(Agent):
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.RESEARCHER
    
    async def process(self, task: str, state: AgentState, llm: Callable) -> str:
        search = self.tools.get("search")
        result = await search.execute(query=task)
        
        state.tool_calls.append(ToolCall(
            tool_name="search",
            arguments={"query": task},
            result=result,
            risk_level=search.risk_level,
        ))
        
        return result


class AnalystAgent(Agent):
    @property
    def role(self) -> AgentRole:
        return AgentRole.ANALYST
    
    async def process(self, task: str, state: AgentState, llm: Callable) -> str:
        context = state.context.get("researcher_result", "No data")
        return f"Analysis of: {context[:50]}..."


class ExecutorAgent(Agent):
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.EXECUTOR
    
    async def process(self, task: str, state: AgentState, llm: Callable) -> str:
        # Determine tool based on task
        if "email" in task.lower():
            tool = self.tools.get("send_email")
            args = {"to": "user@example.com", "subject": "Update", "body": task}
        else:
            tool = self.tools.get("query_database")
            args = {"sql": f"SELECT * WHERE task='{task[:20]}'"}
        
        tool_call = ToolCall(
            tool_name=tool.name,
            arguments=args,
            risk_level=tool.risk_level,
        )
        
        # HITL: High-risk requires approval
        if tool.risk_level == ActionRisk.HIGH:
            state.pending_approval = tool_call
            return f"⚠️ Pending approval: {tool.name}"
        
        result = await tool.execute(**args)
        tool_call.result = result
        state.tool_calls.append(tool_call)
        return result


# ==============================================================================
# ORCHESTRATOR (Manager Pattern)
# ==============================================================================

class OrchestratorAgent:
    """Central coordinator for multi-agent workflow."""
    
    def __init__(
        self,
        agents: Dict[AgentRole, Agent],
        memory: Memory,
        llm_call: Callable,
    ):
        self.agents = agents
        self.memory = memory
        self.llm_call = llm_call
    
    async def plan(self, request: str) -> List[AgentRole]:
        """Simple planning - in production, use LLM."""
        steps = [AgentRole.RESEARCHER]
        
        if "analyze" in request.lower() or "insight" in request.lower():
            steps.append(AgentRole.ANALYST)
        
        if "email" in request.lower() or "send" in request.lower():
            steps.append(AgentRole.EXECUTOR)
        
        return steps
    
    async def run(self, request: str, session_id: str) -> AgentState:
        """Execute multi-agent workflow."""
        state = await self.memory.load_state(session_id)
        if not state:
            state = AgentState(session_id=session_id)
        
        state.messages.append(AgentMessage(
            role=AgentRole.ORCHESTRATOR,
            content=f"Request: {request}",
        ))
        
        # Plan
        steps = await self.plan(request)
        
        # Execute
        for role in steps:
            agent = self.agents.get(role)
            if not agent:
                continue
            
            result = await agent.process(request, state, self.llm_call)
            state.context[f"{role.value}_result"] = result
            state.messages.append(AgentMessage(role=role, content=result))
            
            if state.pending_approval:
                await self.memory.save_state(state)
                return state
        
        state.completed = True
        await self.memory.save_state(state)
        return state
    
    async def approve(self, session_id: str) -> AgentState:
        """Approve pending high-risk action."""
        state = await self.memory.load_state(session_id)
        if not state or not state.pending_approval:
            raise ValueError("No pending approval")
        
        tool_call = state.pending_approval
        tool = self.agents[AgentRole.EXECUTOR].tools.get(tool_call.tool_name)
        
        result = await tool.execute(**tool_call.arguments)
        tool_call.result = result
        state.tool_calls.append(tool_call)
        state.pending_approval = None
        state.messages.append(AgentMessage(
            role=AgentRole.EXECUTOR,
            content=f"✅ Approved: {result}",
        ))
        
        await self.memory.save_state(state)
        return state


# ==============================================================================
# FACTORY
# ==============================================================================

def create_enterprise_agent():
    """Factory to wire up the enterprise agent."""
    memory = InMemoryStore()
    
    tools = ToolRegistry()
    tools.register(SearchTool())
    tools.register(DatabaseTool())
    tools.register(EmailTool())
    
    agents = {
        AgentRole.RESEARCHER: ResearchAgent(tools),
        AgentRole.ANALYST: AnalystAgent(),
        AgentRole.EXECUTOR: ExecutorAgent(tools),
    }
    
    async def mock_llm(prompt: str) -> str:
        return f"LLM response to: {prompt[:30]}..."
    
    orchestrator = OrchestratorAgent(agents, memory, mock_llm)
    return orchestrator, memory


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate enterprise agent."""
    
    print("=" * 60)
    print("Enterprise Agentic AI Demo")
    print("=" * 60)
    
    orchestrator, memory = create_enterprise_agent()
    
    # ========== SIMPLE REQUEST ==========
    print("\n--- Simple Research Request ---")
    
    state = await orchestrator.run(
        "Research Python trends for 2025",
        session_id="session-1",
    )
    
    print(f"Session: {state.session_id}")
    print(f"Messages: {len(state.messages)}")
    for msg in state.messages:
        print(f"  [{msg.role.value}]: {msg.content[:50]}...")
    print(f"Tool calls: {len(state.tool_calls)}")
    print(f"Completed: {state.completed}")
    
    # ========== REQUEST WITH ANALYSIS ==========
    print("\n--- Research + Analyze Request ---")
    
    state = await orchestrator.run(
        "Research AI trends and provide insights",
        session_id="session-2",
    )
    
    print(f"Messages: {len(state.messages)}")
    print(f"Tool calls: {len(state.tool_calls)}")
    print(f"Context keys: {list(state.context.keys())}")
    
    # ========== HIGH-RISK WITH HITL ==========
    print("\n--- High-Risk Request (HITL) ---")
    
    state = await orchestrator.run(
        "Send email update to the customer",
        session_id="session-3",
    )
    
    print(f"Pending approval: {state.pending_approval is not None}")
    if state.pending_approval:
        print(f"Action: {state.pending_approval.tool_name}")
        print(f"Risk: {state.pending_approval.risk_level.value}")
        
        # Simulate approval
        print("\n--- Approving action... ---")
        state = await orchestrator.approve("session-3")
        print(f"Approved! Result: {state.messages[-1].content}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Enterprise Patterns Demonstrated:
- Multi-Agent: Orchestrator → Researcher → Analyst → Executor
- Memory: Persistent AgentState across calls
- HITL: High-risk actions require explicit approval
- Tool Risk: LOW/MEDIUM/HIGH classification
- Manager Pattern: Central orchestration

Based on 2025 enterprise agentic AI research.
""")


if __name__ == "__main__":
    asyncio.run(demo())
