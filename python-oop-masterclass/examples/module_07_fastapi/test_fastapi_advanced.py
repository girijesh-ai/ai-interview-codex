"""
Tests for Module 07b: FastAPI Advanced
======================================
Tests for middleware, streaming, WebSocket, auth, and rate limiting.

Run with: pytest test_fastapi_advanced.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a02_fastapi_advanced import (
    app,
    rate_limiter,
    API_KEYS,
    verify_api_key,
    get_current_user,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def client():
    """Create test client."""
    # Reset rate limiter between tests
    rate_limiter.requests.clear()
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# MIDDLEWARE TESTS
# ==============================================================================

class TestMiddleware:
    """Tests for custom middleware."""
    
    def test_request_id_header_added(self, client):
        """Response should include X-Request-ID header."""
        response = client.get("/")
        assert "X-Request-ID" in response.headers
    
    def test_process_time_header_added(self, client):
        """Response should include X-Process-Time header."""
        response = client.get("/")
        assert "X-Process-Time" in response.headers
        # Should be a valid float
        float(response.headers["X-Process-Time"])
    
    def test_request_id_preserved(self, client):
        """Custom request ID should be preserved."""
        custom_id = "test-123"
        response = client.get("/", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id


# ==============================================================================
# SSE STREAMING TESTS
# ==============================================================================

class TestSSEStreaming:
    """Tests for SSE streaming endpoint."""
    
    def test_stream_returns_event_stream(self, client):
        """Streaming endpoint should return text/event-stream."""
        response = client.post(
            "/v1/stream",
            json={"message": "Hello", "model": "gpt-4"}
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
    
    def test_stream_contains_tokens(self, client):
        """Stream should contain token events."""
        response = client.post(
            "/v1/stream",
            json={"message": "Test", "model": "gpt-4"}
        )
        content = response.text
        assert "data:" in content
        assert "token" in content
    
    def test_stream_ends_with_done(self, client):
        """Stream should end with [DONE] signal."""
        response = client.post(
            "/v1/stream",
            json={"message": "Test", "model": "gpt-4"}
        )
        assert "[DONE]" in response.text


# ==============================================================================
# BACKGROUND TASKS TESTS
# ==============================================================================

class TestBackgroundTasks:
    """Tests for background task endpoint."""
    
    def test_chat_returns_response(self, client):
        """Chat endpoint should return response immediately."""
        response = client.post(
            "/v1/chat-with-logging",
            json={"message": "Hello", "model": "gpt-4"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "tokens" in data
    
    def test_chat_calculates_tokens(self, client):
        """Response should include token count."""
        response = client.post(
            "/v1/chat-with-logging",
            json={"message": "Hello world", "model": "gpt-4"}
        )
        data = response.json()
        assert data["tokens"] > 0


# ==============================================================================
# API KEY AUTH TESTS
# ==============================================================================

class TestAPIKeyAuth:
    """Tests for API key authentication."""
    
    def test_missing_api_key_returns_401(self, client):
        """Request without API key should return 401."""
        response = client.get("/v1/protected/apikey")
        assert response.status_code == 401
    
    def test_invalid_api_key_returns_401(self, client):
        """Request with invalid API key should return 401."""
        response = client.get(
            "/v1/protected/apikey",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
    
    def test_valid_api_key_succeeds(self, client):
        """Request with valid API key should succeed."""
        response = client.get(
            "/v1/protected/apikey",
            headers={"X-API-Key": "sk-demo-key-123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"] == "demo_user"
        assert data["tier"] == "pro"


# ==============================================================================
# JWT AUTH TESTS
# ==============================================================================

class TestJWTAuth:
    """Tests for JWT authentication."""
    
    def test_login_with_valid_credentials(self, client):
        """Valid login should return token."""
        response = client.post(
            "/auth/login",
            json={"username": "demo", "password": "demo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_with_invalid_credentials(self, client):
        """Invalid login should return 401."""
        response = client.post(
            "/auth/login",
            json={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401
    
    def test_protected_without_token_returns_401(self, client):
        """Protected endpoint without token should return 401."""
        response = client.get("/v1/protected/jwt")
        assert response.status_code == 401
    
    def test_protected_with_valid_token(self, client):
        """Protected endpoint with valid token should succeed."""
        # First login
        login_response = client.post(
            "/auth/login",
            json={"username": "demo", "password": "demo"}
        )
        token = login_response.json()["access_token"]
        
        # Then access protected endpoint
        response = client.get(
            "/v1/protected/jwt",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, demo!"


# ==============================================================================
# RATE LIMITING TESTS
# ==============================================================================

class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_returns_remaining(self, client):
        """Response should include remaining requests."""
        response = client.get("/v1/rate-limited")
        assert response.status_code == 200
        data = response.json()
        assert "remaining" in data
    
    def test_rate_limit_decrements(self, client):
        """Remaining count should decrement."""
        r1 = client.get("/v1/rate-limited")
        r2 = client.get("/v1/rate-limited")
        
        remaining1 = r1.json()["remaining"]
        remaining2 = r2.json()["remaining"]
        
        assert remaining2 < remaining1
    
    def test_rate_limit_exceeded_returns_429(self, client):
        """Exceeding rate limit should return 429."""
        # Make 5 requests (the limit)
        for _ in range(5):
            client.get("/v1/rate-limited")
        
        # 6th request should fail
        response = client.get("/v1/rate-limited")
        assert response.status_code == 429
        assert "Retry-After" in response.headers


# ==============================================================================
# WEBSOCKET TESTS
# ==============================================================================

class TestWebSocket:
    """Tests for WebSocket endpoint."""
    
    def test_websocket_connect(self, client):
        """Should be able to connect to WebSocket."""
        with client.websocket_connect("/ws/chat/test-client") as ws:
            # Send a message
            ws.send_text(json.dumps({"message": "Hello"}))
            
            # Receive tokens
            tokens = []
            while True:
                data = ws.receive_json()
                if data.get("done"):
                    break
                tokens.append(data.get("token", ""))
            
            assert len(tokens) > 0


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================

class TestHealthCheck:
    """Tests for health check."""
    
    def test_health_returns_200(self, client):
        """Health check should return 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Health check should return status."""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "healthy"


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
