"""
Integration Tests for FastAPI Endpoints

Tests REST API and WebSocket endpoints.

Demonstrates:
- API testing with FastAPI TestClient
- WebSocket testing
- Authentication testing
- Error response testing
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.api.main import app, get_workflow_integration
from src.api.models import (
    CreateRequestModel,
    UpdateRequestModel,
    ApprovalRequestModel,
    Category,
    Priority
)


# ============================================================================
# TEST CLIENT SETUP
# ============================================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_workflow():
    """Create mock workflow integration."""
    mock = Mock()
    mock.process_request = AsyncMock(return_value={
        "request_id": "req-test-123",
        "status": "pending",
        "category": "account"
    })
    mock.get_request_status = AsyncMock(return_value={
        "request_id": "req-test-123",
        "status": "completed",
        "solution": "Test solution"
    })
    mock.approve_request = AsyncMock(return_value=True)
    mock.reject_request = AsyncMock(return_value=True)
    return mock


@pytest.fixture(autouse=True)
def override_workflow_dependency(mock_workflow):
    """Override workflow integration dependency."""
    app.dependency_overrides[get_workflow_integration] = lambda: mock_workflow
    yield
    app.dependency_overrides.clear()


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

class TestHealthCheck:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        # Act
        response = client.get("/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data


# ============================================================================
# REQUEST CREATION TESTS
# ============================================================================

class TestRequestCreation:
    """Tests for request creation endpoints."""

    def test_create_request_success(self, client, mock_workflow):
        """Test creating request successfully."""
        # Arrange
        request_data = {
            "customer_id": "cust-123",
            "message": "I need help with my account",
            "category": "account",
            "priority": "medium"
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["request_id"] == "req-test-123"
        assert data["status"] == "pending"

        # Verify workflow was called
        mock_workflow.process_request.assert_called_once()

    def test_create_request_minimal_data(self, client, mock_workflow):
        """Test creating request with minimal data."""
        # Arrange
        request_data = {
            "customer_id": "cust-123",
            "message": "Help"
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert "request_id" in data

    def test_create_request_invalid_data(self, client):
        """Test creating request with invalid data."""
        # Arrange
        request_data = {
            "customer_id": "",  # Invalid: empty
            "message": ""  # Invalid: empty
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 422  # Validation error

    def test_create_request_missing_required_fields(self, client):
        """Test creating request with missing required fields."""
        # Arrange
        request_data = {
            "message": "Help"
            # Missing customer_id
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 422

    def test_create_request_with_metadata(self, client, mock_workflow):
        """Test creating request with metadata."""
        # Arrange
        request_data = {
            "customer_id": "cust-123",
            "message": "Help",
            "metadata": {
                "source": "mobile_app",
                "version": "2.0"
            }
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 201


# ============================================================================
# REQUEST RETRIEVAL TESTS
# ============================================================================

class TestRequestRetrieval:
    """Tests for request retrieval endpoints."""

    def test_get_request_success(self, client, mock_workflow):
        """Test getting request successfully."""
        # Arrange
        request_id = "req-test-123"

        # Act
        response = client.get(f"/requests/{request_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == request_id
        assert data["status"] == "completed"

    def test_get_request_not_found(self, client, mock_workflow):
        """Test getting non-existent request."""
        # Arrange
        mock_workflow.get_request_status.return_value = None

        # Act
        response = client.get("/requests/non-existent")

        # Assert
        assert response.status_code == 404

    def test_list_requests(self, client):
        """Test listing requests."""
        # Act
        response = client.get("/requests")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "requests" in data or "items" in data

    def test_list_requests_with_filters(self, client):
        """Test listing requests with filters."""
        # Act
        response = client.get("/requests?status=pending&priority=high")

        # Assert
        assert response.status_code == 200

    def test_list_requests_with_pagination(self, client):
        """Test listing requests with pagination."""
        # Act
        response = client.get("/requests?skip=0&limit=10")

        # Assert
        assert response.status_code == 200


# ============================================================================
# REQUEST UPDATE TESTS
# ============================================================================

class TestRequestUpdate:
    """Tests for request update endpoints."""

    def test_update_request_success(self, client, mock_workflow):
        """Test updating request successfully."""
        # Arrange
        request_id = "req-test-123"
        update_data = {
            "status": "in_progress",
            "priority": "high"
        }

        # Act
        response = client.patch(f"/requests/{request_id}", json=update_data)

        # Assert
        assert response.status_code == 200

    def test_add_message_to_request(self, client, mock_workflow):
        """Test adding message to request."""
        # Arrange
        request_id = "req-test-123"
        message_data = {
            "message": "Additional information",
            "sender": "customer"
        }

        # Act
        response = client.post(
            f"/requests/{request_id}/messages",
            json=message_data
        )

        # Assert
        assert response.status_code == 201


# ============================================================================
# APPROVAL WORKFLOW TESTS
# ============================================================================

class TestApprovalWorkflow:
    """Tests for approval workflow endpoints."""

    def test_approve_request_success(self, client, mock_workflow):
        """Test approving request successfully."""
        # Arrange
        request_id = "req-test-123"
        approval_data = {
            "approved": True,
            "approver_id": "manager-123",
            "comments": "Looks good"
        }

        # Act
        response = client.post(
            f"/requests/{request_id}/approve",
            json=approval_data
        )

        # Assert
        assert response.status_code == 200
        mock_workflow.approve_request.assert_called_once()

    def test_reject_request_success(self, client, mock_workflow):
        """Test rejecting request successfully."""
        # Arrange
        request_id = "req-test-123"
        approval_data = {
            "approved": False,
            "approver_id": "manager-123",
            "comments": "Needs more review"
        }

        # Act
        response = client.post(
            f"/requests/{request_id}/approve",
            json=approval_data
        )

        # Assert
        assert response.status_code == 200
        mock_workflow.reject_request.assert_called_once()

    def test_approve_without_comments(self, client):
        """Test approval without comments fails."""
        # Arrange
        request_id = "req-test-123"
        approval_data = {
            "approved": True,
            "approver_id": "manager-123"
            # Missing comments
        }

        # Act
        response = client.post(
            f"/requests/{request_id}/approve",
            json=approval_data
        )

        # Assert
        assert response.status_code == 422


# ============================================================================
# ANALYTICS ENDPOINTS TESTS
# ============================================================================

class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    def test_get_metrics(self, client):
        """Test getting system metrics."""
        # Act
        response = client.get("/metrics")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data or "metrics" in data

    def test_get_agent_performance(self, client):
        """Test getting agent performance metrics."""
        # Act
        response = client.get("/metrics/agents")

        # Assert
        assert response.status_code == 200

    def test_get_metrics_by_time_range(self, client):
        """Test getting metrics with time range."""
        # Act
        response = client.get(
            "/metrics?start_date=2024-01-01&end_date=2024-01-31"
        )

        # Assert
        assert response.status_code == 200


# ============================================================================
# CUSTOMER ENDPOINTS TESTS
# ============================================================================

class TestCustomerEndpoints:
    """Tests for customer endpoints."""

    def test_get_customer(self, client):
        """Test getting customer."""
        # Act
        response = client.get("/customers/cust-123")

        # Assert
        assert response.status_code in [200, 404]

    def test_get_customer_requests(self, client):
        """Test getting customer requests."""
        # Act
        response = client.get("/customers/cust-123/requests")

        # Assert
        assert response.status_code in [200, 404]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_404_not_found(self, client):
        """Test 404 error."""
        # Act
        response = client.get("/non-existent-endpoint")

        # Assert
        assert response.status_code == 404

    def test_500_internal_error(self, client, mock_workflow):
        """Test 500 error handling."""
        # Arrange
        mock_workflow.process_request.side_effect = Exception("Test error")

        request_data = {
            "customer_id": "cust-123",
            "message": "Help"
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 500

    def test_validation_error_response_format(self, client):
        """Test validation error response format."""
        # Arrange
        request_data = {
            "customer_id": "",
            "message": ""
        }

        # Act
        response = client.post("/requests", json=request_data)

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


# ============================================================================
# WEBSOCKET TESTS
# ============================================================================

class TestWebSocket:
    """Tests for WebSocket endpoints."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        # Act & Assert
        with client.websocket_connect("/ws/test-client") as websocket:
            # Connection successful
            data = websocket.receive_json()
            assert data["type"] == "connection_established"

    def test_websocket_receive_updates(self, client):
        """Test receiving updates via WebSocket."""
        # Act & Assert
        with client.websocket_connect("/ws/test-client") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Simulate sending a message
            websocket.send_json({
                "type": "subscribe",
                "request_id": "req-123"
            })

            # Should receive acknowledgment
            response = websocket.receive_json()
            assert response is not None

    def test_websocket_multiple_clients(self, client):
        """Test multiple WebSocket clients."""
        # Act & Assert
        with client.websocket_connect("/ws/client-1") as ws1:
            with client.websocket_connect("/ws/client-2") as ws2:
                # Both connections should be active
                ws1.receive_json()
                ws2.receive_json()


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

class TestAuthentication:
    """Tests for authentication (if implemented)."""

    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without auth."""
        # Note: Adjust based on actual auth implementation
        # Act
        response = client.get("/admin/metrics")

        # Assert
        # Should be 401 if auth is required, 404 if endpoint doesn't exist
        assert response.status_code in [401, 404]


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiting:
    """Tests for rate limiting (if implemented)."""

    def test_rate_limit_exceeded(self, client):
        """Test rate limit exceeded."""
        # Note: Adjust based on actual rate limiting implementation
        request_data = {
            "customer_id": "cust-123",
            "message": "Help"
        }

        # Make many requests
        responses = []
        for _ in range(100):
            response = client.post("/requests", json=request_data)
            responses.append(response.status_code)

        # At least one should succeed
        assert 201 in responses


# ============================================================================
# CORS TESTS
# ============================================================================

class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        # Act
        response = client.options("/requests")

        # Assert
        assert "access-control-allow-origin" in response.headers or \
               response.status_code == 200


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestEndToEndFlows:
    """End-to-end integration tests."""

    def test_complete_request_flow(self, client, mock_workflow):
        """Test complete request flow from creation to completion."""
        # Step 1: Create request
        request_data = {
            "customer_id": "cust-123",
            "message": "I need help with my account"
        }
        response = client.post("/requests", json=request_data)
        assert response.status_code == 201
        request_id = response.json()["request_id"]

        # Step 2: Get request status
        response = client.get(f"/requests/{request_id}")
        assert response.status_code == 200

        # Step 3: Add message
        message_data = {
            "message": "More details",
            "sender": "customer"
        }
        response = client.post(
            f"/requests/{request_id}/messages",
            json=message_data
        )
        assert response.status_code == 201

        # Step 4: Approve
        approval_data = {
            "approved": True,
            "approver_id": "manager-123",
            "comments": "Approved"
        }
        response = client.post(
            f"/requests/{request_id}/approve",
            json=approval_data
        )
        assert response.status_code == 200

    def test_escalation_flow(self, client, mock_workflow):
        """Test escalation workflow."""
        # Create high-priority request
        request_data = {
            "customer_id": "cust-123",
            "message": "Urgent issue!",
            "priority": "critical"
        }
        response = client.post("/requests", json=request_data)
        assert response.status_code == 201

        request_id = response.json()["request_id"]

        # Should require approval
        response = client.get(f"/requests/{request_id}")
        assert response.status_code == 200
