"""
Tests for Enterprise Agent
===========================
Run with: pytest test_enterprise_agent.py -v
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a02_enterprise_agent import (
    AgentRole, ActionRisk, AgentState, AgentMessage, ToolCall,
    InMemoryStore, ToolRegistry,
    SearchTool, DatabaseTool, EmailTool,
    ResearchAgent, AnalystAgent, ExecutorAgent,
    OrchestratorAgent,
    create_enterprise_agent,
)


# ==============================================================================
# DOMAIN MODEL TESTS
# ==============================================================================

class TestDomainModels:
    def test_agent_state_creation(self):
        state = AgentState(session_id="test-123")
        assert state.session_id == "test-123"
        assert state.completed == False
        assert state.pending_approval is None
    
    def test_tool_call_risk_levels(self):
        assert ActionRisk.LOW.value == "low"
        assert ActionRisk.HIGH.value == "high"


# ==============================================================================
# MEMORY TESTS
# ==============================================================================

class TestInMemoryStore:
    def test_save_and_load(self):
        async def run():
            memory = InMemoryStore()
            state = AgentState(session_id="s1")
            state.messages.append(AgentMessage(AgentRole.ORCHESTRATOR, "test"))
            
            await memory.save_state(state)
            loaded = await memory.load_state("s1")
            return loaded
        
        loaded = asyncio.run(run())
        assert loaded is not None
        assert len(loaded.messages) == 1
    
    def test_load_nonexistent(self):
        async def run():
            memory = InMemoryStore()
            return await memory.load_state("nonexistent")
        
        result = asyncio.run(run())
        assert result is None


# ==============================================================================
# TOOL TESTS
# ==============================================================================

class TestTools:
    def test_search_tool_low_risk(self):
        tool = SearchTool()
        assert tool.name == "search"
        assert tool.risk_level == ActionRisk.LOW
    
    def test_email_tool_high_risk(self):
        tool = EmailTool()
        assert tool.name == "send_email"
        assert tool.risk_level == ActionRisk.HIGH
    
    def test_tool_execution(self):
        async def run():
            tool = SearchTool()
            result = await tool.execute(query="test query")
            return result
        
        result = asyncio.run(run())
        assert "test query" in result


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(EmailTool())
        
        assert "search" in registry.list_tools()
        assert "send_email" in registry.list_tools()


# ==============================================================================
# AGENT TESTS
# ==============================================================================

class TestResearchAgent:
    def test_process_adds_tool_call(self):
        async def run():
            tools = ToolRegistry()
            tools.register(SearchTool())
            
            agent = ResearchAgent(tools)
            state = AgentState(session_id="test")
            
            async def mock_llm(p): return "mock"
            
            result = await agent.process("test task", state, mock_llm)
            return state, result
        
        state, result = asyncio.run(run())
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].tool_name == "search"


class TestExecutorAgent:
    def test_high_risk_triggers_approval(self):
        async def run():
            tools = ToolRegistry()
            tools.register(EmailTool())
            tools.register(DatabaseTool())
            
            agent = ExecutorAgent(tools)
            state = AgentState(session_id="test")
            
            async def mock_llm(p): return "mock"
            
            await agent.process("send email to customer", state, mock_llm)
            return state
        
        state = asyncio.run(run())
        assert state.pending_approval is not None
        assert state.pending_approval.risk_level == ActionRisk.HIGH


# ==============================================================================
# ORCHESTRATOR TESTS
# ==============================================================================

class TestOrchestratorAgent:
    def test_simple_request(self):
        async def run():
            orchestrator, memory = create_enterprise_agent()
            state = await orchestrator.run("Research Python", "test-session")
            return state
        
        state = asyncio.run(run())
        assert state.session_id == "test-session"
        assert len(state.messages) > 0
    
    def test_hitl_workflow(self):
        async def run():
            orchestrator, memory = create_enterprise_agent()
            
            # Request that triggers HITL
            state = await orchestrator.run("Send email update", "hitl-session")
            
            # Should have pending approval
            assert state.pending_approval is not None
            
            # Approve
            state = await orchestrator.approve("hitl-session")
            return state
        
        state = asyncio.run(run())
        assert state.pending_approval is None
        assert len(state.tool_calls) > 0


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
