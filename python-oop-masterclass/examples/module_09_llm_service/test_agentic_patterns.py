"""
Tests for Agentic Patterns
===========================
Run with: pytest test_agentic_patterns.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a02_agentic_patterns import (
    ToolResult, Tool,
    CalculatorTool, SearchTool, WeatherTool,
    ToolRegistry, ReActAgent, MockLLM,
)


# ==============================================================================
# TOOL TESTS
# ==============================================================================

class TestCalculatorTool:
    """Test calculator tool."""
    
    def test_addition(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        assert result.success
        assert result.output == "4"
    
    def test_multiplication(self):
        tool = CalculatorTool()
        result = tool.execute(expression="15 * 4")
        assert result.success
        assert result.output == "60"
    
    def test_complex_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="(10 + 5) * 2")
        assert result.success
        assert result.output == "30"
    
    def test_invalid_characters(self):
        tool = CalculatorTool()
        result = tool.execute(expression="import os")
        assert not result.success
        assert "Invalid" in result.error


class TestSearchTool:
    """Test search tool."""
    
    def test_python_search(self):
        tool = SearchTool()
        result = tool.execute(query="python version")
        assert result.success
        assert "Python" in result.output
    
    def test_unknown_query(self):
        tool = SearchTool()
        result = tool.execute(query="random unknown topic")
        assert result.success
        assert "No specific" in result.output


class TestWeatherTool:
    """Test weather tool."""
    
    def test_get_weather(self):
        tool = WeatherTool()
        result = tool.execute(city="Tokyo")
        assert result.success
        assert "Tokyo" in result.output


# ==============================================================================
# REGISTRY TESTS
# ==============================================================================

class TestToolRegistry:
    """Test tool registry."""
    
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())
        
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "web_search" in tools
    
    def test_execute_tool(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        result = registry.execute("calculator", expression="5 * 5")
        assert result.output == "25"
    
    def test_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError):
            registry.get("nonexistent")
    
    def test_openai_format(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        schemas = registry.to_openai_tools()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "calculator"


# ==============================================================================
# REACT AGENT TESTS
# ==============================================================================

class TestReActAgent:
    """Test ReAct agent."""
    
    def test_weather_query(self):
        registry = ToolRegistry()
        registry.register(WeatherTool())
        
        agent = ReActAgent(tools=registry, max_steps=3)
        mock_llm = MockLLM()
        
        answer = agent.run("What's the weather in London?", mock_llm)
        assert "London" in answer or "22°C" in answer
    
    def test_calculator_query(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        agent = ReActAgent(tools=registry, max_steps=3)
        mock_llm = MockLLM()
        
        answer = agent.run("Calculate 15 * 4 + 10", mock_llm)
        assert "70" in answer
    
    def test_search_query(self):
        registry = ToolRegistry()
        registry.register(SearchTool())
        
        agent = ReActAgent(tools=registry, max_steps=3)
        mock_llm = MockLLM()
        
        answer = agent.run("What's the current Python version?", mock_llm)
        assert "Python" in answer


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
