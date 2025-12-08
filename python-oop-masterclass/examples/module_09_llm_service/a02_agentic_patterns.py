"""
Agentic Patterns Example
========================
Demonstrates:
- Tool abstraction (ABC)
- Tool Registry pattern
- ReAct agent loop
- Simple tool chains

Run with: python a02_agentic_patterns.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
from enum import Enum
import json


# ==============================================================================
# TOOL ABSTRACTION
# ==============================================================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: str = ""


class Tool(ABC):
    """
    Base class for agent tools.
    
    Simple interface:
    - name: Identifier
    - description: For LLM to understand
    - parameters: JSON Schema
    - execute(): Run it
    """
    
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
    def parameters(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ==============================================================================
# PRACTICAL TOOLS
# ==============================================================================

class CalculatorTool(Tool):
    """Basic math calculator."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform math calculations. Use for any arithmetic."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 2 * 3'"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, expression: str) -> ToolResult:
        try:
            # Safe: only allow math characters
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return ToolResult(False, "", "Invalid characters")
            
            result = eval(expression)
            return ToolResult(True, str(result))
        except Exception as e:
            return ToolResult(False, "", str(e))


class SearchTool(Tool):
    """Web search (mock for demo)."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    
    def execute(self, query: str) -> ToolResult:
        # Mock results
        results = {
            "python": "Python 3.12 is the latest version (Dec 2025).",
            "openai": "OpenAI offers GPT-4o and GPT-4o-mini models.",
            "fastapi": "FastAPI is a modern Python web framework.",
        }
        
        for key, answer in results.items():
            if key in query.lower():
                return ToolResult(True, answer)
        
        return ToolResult(True, f"No specific results for: {query}")


class WeatherTool(Tool):
    """Get weather (mock for demo)."""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get current weather for a city."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    
    def execute(self, city: str) -> ToolResult:
        # Mock weather
        return ToolResult(True, f"Weather in {city}: 22°C, Sunny")


# ==============================================================================
# TOOL REGISTRY
# ==============================================================================

class ToolRegistry:
    """Central registry for tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Add a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get tool by name."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name]
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool."""
        return self.get(name).execute(**kwargs)
    
    def list_tools(self) -> List[str]:
        """List tool names."""
        return list(self._tools.keys())
    
    def to_openai_tools(self) -> List[Dict]:
        """Get all tools in OpenAI format."""
        return [t.to_openai_schema() for t in self._tools.values()]


# ==============================================================================
# REACT AGENT
# ==============================================================================

class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


@dataclass
class AgentStep:
    """One reasoning step."""
    type: StepType
    content: str
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)


class ReActAgent:
    """
    ReAct Agent: Reason + Act loop.
    
    1. Think about what to do
    2. Choose and execute a tool
    3. Observe the result
    4. Repeat until answer found
    """
    
    def __init__(self, tools: ToolRegistry, max_steps: int = 5):
        self.tools = tools
        self.max_steps = max_steps
    
    def _build_prompt(self, question: str, steps: List[AgentStep]) -> str:
        """Build prompt with tool descriptions and history."""
        tool_desc = "\n".join([
            f"- {name}: {self.tools.get(name).description}"
            for name in self.tools.list_tools()
        ])
        
        prompt = f"""Solve the problem step by step.

Available tools:
{tool_desc}

Format for using tools:
Thought: [reasoning]
Action: [tool_name]
Action Input: {{"param": "value"}}

Format for final answer:
Thought: [reasoning]
Answer: [your answer]

Question: {question}
"""
        
        for step in steps:
            if step.type == StepType.THOUGHT:
                prompt += f"\nThought: {step.content}"
            elif step.type == StepType.ACTION:
                prompt += f"\nAction: {step.tool_name}"
                prompt += f"\nAction Input: {json.dumps(step.tool_args)}"
            elif step.type == StepType.OBSERVATION:
                prompt += f"\nObservation: {step.content}"
        
        return prompt
    
    def _parse(self, response: str) -> AgentStep:
        """Parse LLM response."""
        response = response.strip()
        
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            return AgentStep(StepType.ANSWER, answer)
        
        thought = ""
        if "Thought:" in response:
            thought = response.split("Thought:")[-1].split("Action:")[0].strip()
        
        if "Action:" in response:
            action_part = response.split("Action:")[-1]
            tool_name = action_part.split("\n")[0].strip()
            
            args = {}
            if "Action Input:" in action_part:
                try:
                    args_str = action_part.split("Action Input:")[-1].strip()
                    args = json.loads(args_str)
                except:
                    pass
            
            return AgentStep(StepType.ACTION, thought, tool_name, args)
        
        return AgentStep(StepType.THOUGHT, thought or response)
    
    def run(self, question: str, llm_call: Callable[[str], str]) -> str:
        """
        Run ReAct loop.
        
        llm_call: Function that takes prompt, returns response
        """
        steps: List[AgentStep] = []
        
        for _ in range(self.max_steps):
            prompt = self._build_prompt(question, steps)
            response = llm_call(prompt)
            step = self._parse(response)
            
            if step.type == StepType.ANSWER:
                return step.content
            
            steps.append(step)
            
            if step.type == StepType.ACTION:
                try:
                    result = self.tools.execute(step.tool_name, **step.tool_args)
                    obs = result.output if result.success else f"Error: {result.error}"
                except Exception as e:
                    obs = f"Error: {e}"
                
                steps.append(AgentStep(StepType.OBSERVATION, obs))
        
        return "Could not find answer within step limit."


# ==============================================================================
# SIMPLE MOCK LLM FOR DEMO
# ==============================================================================

class MockLLM:
    """
    Mock LLM that demonstrates ReAct reasoning.
    
    In production, replace with actual LLM call.
    """
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        
        # Simulate ReAct reasoning
        if "Observation:" not in prompt:
            # First call: decide to use a tool
            if "weather" in prompt.lower():
                return '''Thought: I need to get the weather information.
Action: get_weather
Action Input: {"city": "London"}'''
            elif "calculate" in prompt.lower() or "math" in prompt.lower():
                return '''Thought: I need to calculate this.
Action: calculator
Action Input: {"expression": "15 * 4 + 10"}'''
            elif "python" in prompt.lower():
                return '''Thought: I should search for current Python info.
Action: web_search
Action Input: {"query": "python version"}'''
            else:
                return "Thought: I don't have a relevant tool.\nAnswer: I cannot help with that."
        else:
            # Second call: we have an observation, give answer
            observation = prompt.split("Observation:")[-1].strip().split("\n")[0]
            return f"Thought: I have the information I need.\nAnswer: {observation}"



# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    """Demonstrate agentic patterns."""
    
    print("=" * 60)
    print("Agentic Patterns Demo")
    print("=" * 60)
    
    # ========== SETUP TOOLS ==========
    print("\n--- Tool Registry ---")
    
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    registry.register(WeatherTool())
    
    print(f"Registered tools: {registry.list_tools()}")
    
    # ========== DIRECT TOOL USE ==========
    print("\n--- Direct Tool Execution ---")
    
    calc_result = registry.execute("calculator", expression="100 / 4")
    print(f"Calculator: 100 / 4 = {calc_result.output}")
    
    search_result = registry.execute("web_search", query="What is FastAPI?")
    print(f"Search: {search_result.output}")
    
    weather_result = registry.execute("get_weather", city="Paris")
    print(f"Weather: {weather_result.output}")
    
    # ========== OPENAI FORMAT ==========
    print("\n--- OpenAI Tool Format ---")
    
    tools_schema = registry.to_openai_tools()
    print(f"Tools for API: {[t['function']['name'] for t in tools_schema]}")
    
    # ========== REACT AGENT ==========
    print("\n--- ReAct Agent ---")
    
    agent = ReActAgent(tools=registry, max_steps=3)
    mock_llm = MockLLM()
    
    # Test 1: Weather query
    print("\nQuestion: What's the weather in London?")
    answer = agent.run("What's the weather in London?", mock_llm)
    print(f"Answer: {answer}")
    
    # Test 2: Math query
    mock_llm = MockLLM()  # Reset
    print("\nQuestion: Calculate 15 * 4 + 10")
    answer = agent.run("Calculate 15 * 4 + 10", mock_llm)
    print(f"Answer: {answer}")
    
    # Test 3: Search query
    mock_llm = MockLLM()  # Reset
    print("\nQuestion: What's the current Python version?")
    answer = agent.run("What's the current Python version?", mock_llm)
    print(f"Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- Tool ABC: Clean interface for any tool
- Registry: Centralized tool management
- ReAct: Reason → Act → Observe loop
- Mock LLM: For testing without API calls

No over-engineering - simple, practical patterns.
""")


if __name__ == "__main__":
    demo()
