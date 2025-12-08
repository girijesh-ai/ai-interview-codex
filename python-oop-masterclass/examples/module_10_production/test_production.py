"""
Tests for Production Patterns
==============================
Run with: pytest test_production.py -v
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a01_production_patterns import (
    MockLLMClient, MockLLMResponse,
    HealthChecker, RequestContext,
    JSONFormatter,
)


# ==============================================================================
# MOCK LLM TESTS
# ==============================================================================

class TestMockLLMClient:
    """Test mock LLM client."""
    
    def test_basic_response(self):
        async def run():
            mock = MockLLMClient()
            mock.add_response("Test response")
            result = await mock.complete([{"role": "user", "content": "Hi"}])
            return result
        
        result = asyncio.run(run())
        assert result["content"] == "Test response"
        assert result["model"] == "gpt-4o-mini"
    
    def test_multiple_responses(self):
        async def run():
            mock = MockLLMClient()
            mock.add_response("First")
            mock.add_response("Second")
            
            r1 = await mock.complete([])
            r2 = await mock.complete([])
            return r1, r2
        
        r1, r2 = asyncio.run(run())
        assert r1["content"] == "First"
        assert r2["content"] == "Second"
    
    def test_call_tracking(self):
        async def run():
            mock = MockLLMClient()
            await mock.complete([{"role": "user", "content": "Test"}])
            return mock
        
        mock = asyncio.run(run())
        assert mock.call_count == 1
        assert len(mock.calls) == 1
        assert mock.calls[0]["messages"][0]["content"] == "Test"
    
    def test_streaming(self):
        async def run():
            mock = MockLLMClient()
            mock.add_response("Hello world")
            
            tokens = []
            async for token in mock.stream([]):
                tokens.append(token)
            return tokens
        
        tokens = asyncio.run(run())
        assert len(tokens) == 2
        assert "Hello" in tokens[0]


class TestMockLLMResponse:
    """Test mock response object."""
    
    def test_default_usage(self):
        response = MockLLMResponse(content="Test")
        assert response.usage["total_tokens"] == 30
    
    def test_to_dict(self):
        response = MockLLMResponse(content="Test", model="test-model")
        d = response.to_dict()
        assert d["content"] == "Test"
        assert d["model"] == "test-model"


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================

class TestHealthChecker:
    """Test health checker."""
    
    def test_all_healthy(self):
        async def run():
            checker = HealthChecker()
            checker.add_check("db", lambda: True)
            checker.add_check("cache", lambda: True)
            return await checker.run_checks()
        
        result = asyncio.run(run())
        assert result["status"] == "healthy"
        assert result["checks"]["db"] == "ok"
    
    def test_degraded_on_failure(self):
        async def run():
            checker = HealthChecker()
            checker.add_check("db", lambda: True)
            checker.add_check("cache", lambda: False)
            return await checker.run_checks()
        
        result = asyncio.run(run())
        assert result["status"] == "degraded"
        assert result["checks"]["cache"] == "error"
    
    def test_exception_handling(self):
        async def run():
            checker = HealthChecker()
            checker.add_check("failing", lambda: 1/0)
            return await checker.run_checks()
        
        result = asyncio.run(run())
        assert result["status"] == "degraded"
        assert "error:" in result["checks"]["failing"]


# ==============================================================================
# REQUEST CONTEXT TESTS
# ==============================================================================

class TestRequestContext:
    """Test request ID tracking."""
    
    def test_generate_id(self):
        request_id = RequestContext.generate_id()
        assert len(request_id) == 8
    
    def test_get_set_id(self):
        RequestContext.set_id("test-123")
        assert RequestContext.get_id() == "test-123"


# ==============================================================================
# JSON FORMATTER TESTS
# ==============================================================================

class TestJSONFormatter:
    """Test JSON log formatter."""
    
    def test_format_basic(self):
        import logging
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        import json
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
