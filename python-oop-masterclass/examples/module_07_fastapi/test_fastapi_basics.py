"""
Tests for Module 07a: FastAPI Fundamentals
==========================================
Demonstrates testing patterns for FastAPI applications:
- TestClient for endpoint testing
- Dependency overrides for mocking
- Async testing patterns
- Response validation

Run with: pytest test_fastapi_basics.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from our example (use relative import)
from a01_fastapi_basics import (
    app,
    get_llm_client,
    get_settings,
    Settings,
    MockLLMClient,
    ChatRequest,
    ChatMessage,
    ChatResponse,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    return MockLLMClient(delay=0)  # No delay for fast tests


@pytest.fixture
def override_dependencies(mock_llm_client):
    """Override dependencies for testing."""
    def get_mock_llm():
        return mock_llm_client
    
    def get_test_settings():
        return Settings(
            app_name="Test AI API",
            environment="test",
            default_model="test-model",
            api_key="test-key"
        )
    
    app.dependency_overrides[get_llm_client] = get_mock_llm
    app.dependency_overrides[get_settings] = get_test_settings
    
    yield
    
    # Cleanup
    app.dependency_overrides.clear()


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================

class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_200(self, client, override_dependencies):
        """Health check should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_check_returns_status(self, client, override_dependencies):
        """Health check should return status field."""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_returns_app_name(self, client, override_dependencies):
        """Health check should return app name from settings."""
        response = client.get("/")
        data = response.json()
        assert data["app"] == "Test AI API"


# ==============================================================================
# MODELS ENDPOINT TESTS
# ==============================================================================

class TestModelsEndpoint:
    """Tests for /models endpoints."""
    
    def test_list_models_returns_list(self, client, override_dependencies):
        """List models should return a list."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_list_models_has_expected_providers(self, client, override_dependencies):
        """List models should include multiple providers."""
        response = client.get("/models")
        data = response.json()
        providers = {m["provider"] for m in data["models"]}
        assert "openai" in providers
    
    def test_get_model_existing(self, client, override_dependencies):
        """Get existing model should return details."""
        response = client.get("/models/gpt-4")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt-4"
        assert data["provider"] == "openai"
    
    def test_get_model_not_found(self, client, override_dependencies):
        """Get non-existent model should return 404."""
        response = client.get("/models/nonexistent-model")
        assert response.status_code == 404


# ==============================================================================
# CHAT COMPLETION TESTS
# ==============================================================================

class TestChatCompletion:
    """Tests for /v1/chat/completions endpoint."""
    
    def test_chat_completion_success(self, client, override_dependencies, mock_llm_client):
        """Valid chat request should return completion."""
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["model"] == "gpt-4"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
    
    def test_chat_completion_validates_messages(self, client, override_dependencies):
        """Chat request with empty messages should fail."""
        request_data = {
            "model": "gpt-4",
            "messages": []
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completion_validates_temperature(self, client, override_dependencies):
        """Chat request with invalid temperature should fail."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 5.0  # Invalid: max is 2.0
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
    
    def test_chat_completion_returns_usage(self, client, override_dependencies):
        """Chat response should include token usage."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        data = response.json()
        
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]


# ==============================================================================
# BATCH ENDPOINT TESTS
# ==============================================================================

class TestBatchEndpoint:
    """Tests for /v1/batch endpoint."""
    
    def test_batch_processes_multiple_prompts(self, client, override_dependencies):
        """Batch should process multiple prompts."""
        request_data = {
            "prompts": ["Hello", "World", "Test"],
            "model": "gpt-4"
        }
        
        response = client.post("/v1/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["responses"]) == 3
    
    def test_batch_returns_timing(self, client, override_dependencies):
        """Batch should return total time."""
        request_data = {
            "prompts": ["Hello"],
            "model": "gpt-4"
        }
        
        response = client.post("/v1/batch", json=request_data)
        data = response.json()
        
        assert "total_time_ms" in data
        assert data["total_time_ms"] >= 0
    
    def test_batch_validates_max_prompts(self, client, override_dependencies):
        """Batch should reject too many prompts."""
        request_data = {
            "prompts": [f"Prompt {i}" for i in range(15)],  # Max is 10
            "model": "gpt-4"
        }
        
        response = client.post("/v1/batch", json=request_data)
        assert response.status_code == 422


# ==============================================================================
# CONVERSATION MESSAGES TESTS
# ==============================================================================

class TestConversationMessages:
    """Tests for /conversations/{id}/messages endpoint."""
    
    def test_get_messages_with_filters(self, client, override_dependencies):
        """Get messages should accept query filters."""
        response = client.get(
            "/conversations/conv-123/messages",
            params={"role": "user", "limit": 10}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["conversation_id"] == "conv-123"
    
    def test_get_messages_validates_limit(self, client, override_dependencies):
        """Get messages should validate limit range."""
        response = client.get(
            "/conversations/conv-123/messages",
            params={"limit": 500}  # Max is 200
        )
        assert response.status_code == 422


# ==============================================================================
# DEPENDENCY INJECTION TESTS
# ==============================================================================

class TestDependencyInjection:
    """Tests demonstrating dependency injection patterns."""
    
    def test_can_override_llm_client(self, client):
        """LLM client can be replaced for testing."""
        # Create a custom mock
        custom_mock = MockLLMClient(delay=0)
        
        def get_custom_mock():
            return custom_mock
        
        app.dependency_overrides[get_llm_client] = get_custom_mock
        
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Test"}]
                }
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()
    
    def test_can_override_settings(self, client):
        """Settings can be overridden for testing."""
        def get_custom_settings():
            return Settings(
                app_name="Custom Test App",
                environment="custom",
                default_model="custom-model",
                api_key="custom-key"
            )
        
        app.dependency_overrides[get_settings] = get_custom_settings
        
        try:
            response = client.get("/")
            data = response.json()
            assert data["app"] == "Custom Test App"
        finally:
            app.dependency_overrides.clear()


# ==============================================================================
# PYDANTIC MODEL TESTS
# ==============================================================================

class TestPydanticModels:
    """Tests for Pydantic request/response models."""
    
    def test_chat_message_valid(self):
        """ChatMessage should accept valid data."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_chat_message_validates_role(self):
        """ChatMessage should reject invalid role."""
        with pytest.raises(ValueError):
            ChatMessage(role="invalid", content="Hello")
    
    def test_chat_message_validates_content_length(self):
        """ChatMessage should reject empty content."""
        with pytest.raises(ValueError):
            ChatMessage(role="user", content="")
    
    def test_chat_request_defaults(self):
        """ChatRequest should have sensible defaults."""
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")]
        )
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens is None


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
