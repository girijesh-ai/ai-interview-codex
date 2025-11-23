"""
API Request/Response Models

Demonstrates:
- Pydantic models for validation
- DTO pattern
- Type safety
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class RequestStatus(str, Enum):
    """Request status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class Priority(int, Enum):
    """Request priority."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Category(str, Enum):
    """Request category."""
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    PRODUCT = "product"
    GENERAL = "general"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class CreateRequestModel(BaseModel):
    """Create customer request."""
    customer_id: str = Field(..., description="Customer ID")
    message: str = Field(..., min_length=1, description="Customer message")
    category: Optional[Category] = Field(None, description="Request category")
    priority: Optional[Priority] = Field(Priority.MEDIUM, description="Request priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "cust-123",
                "message": "How do I reset my password?",
                "category": "account",
                "priority": 2
            }
        }


class UpdateRequestModel(BaseModel):
    """Update request."""
    message: Optional[str] = Field(None, description="Additional message")
    priority: Optional[Priority] = Field(None, description="Updated priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class RequestResponseModel(BaseModel):
    """Request response."""
    request_id: str
    customer_id: str
    status: RequestStatus
    category: Optional[str] = None
    priority: int
    message: str
    solution: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req-123",
                "customer_id": "cust-456",
                "status": "completed",
                "category": "account",
                "priority": 2,
                "message": "How do I reset my password?",
                "solution": "You can reset your password by...",
                "created_at": "2025-01-01T10:00:00Z",
                "updated_at": "2025-01-01T10:05:00Z"
            }
        }


class RequestListResponse(BaseModel):
    """List of requests."""
    requests: List[RequestResponseModel]
    total: int
    page: int
    page_size: int


# ============================================================================
# APPROVAL MODELS
# ============================================================================

class ApprovalRequestModel(BaseModel):
    """Approval request."""
    approved: bool = Field(..., description="Approval decision")
    notes: Optional[str] = Field(None, description="Approval notes")
    approver: str = Field(..., description="Approver identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "approved": True,
                "notes": "Approved for processing",
                "approver": "manager@example.com"
            }
        }


class ApprovalResponseModel(BaseModel):
    """Approval response."""
    request_id: str
    approved: bool
    approver: str
    notes: Optional[str] = None
    approved_at: datetime


# ============================================================================
# SESSION MODELS
# ============================================================================

class SessionModel(BaseModel):
    """Session information."""
    session_id: str
    customer_id: str
    created_at: datetime
    last_activity: datetime
    data: Dict[str, Any]


# ============================================================================
# METRICS MODELS
# ============================================================================

class MetricsModel(BaseModel):
    """System metrics."""
    total_requests: int
    completed_requests: int
    failed_requests: int
    avg_resolution_time_seconds: float
    cache_hit_rate: float
    active_sessions: int
    timestamp: datetime


class AgentMetricsModel(BaseModel):
    """Agent-specific metrics."""
    agent_type: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_duration_ms: float
    error_rate: float


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================

class ComponentHealthModel(BaseModel):
    """Component health status."""
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckModel(BaseModel):
    """Health check response."""
    overall_status: str
    components: Dict[str, ComponentHealthModel]
    timestamp: datetime


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorModel(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Request not found",
                "detail": "No request with ID req-123",
                "code": "NOT_FOUND",
                "timestamp": "2025-01-01T10:00:00Z"
            }
        }


# ============================================================================
# WEBSOCKET MODELS
# ============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket message."""
    type: str  # subscribe, unsubscribe, event, error
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class SubscriptionModel(BaseModel):
    """WebSocket subscription."""
    request_id: Optional[str] = None
    customer_id: Optional[str] = None
    event_types: List[str] = Field(default_factory=list)


# ============================================================================
# PAGINATION MODELS
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")


# ============================================================================
# FILTER MODELS
# ============================================================================

class RequestFilterModel(BaseModel):
    """Request filter parameters."""
    customer_id: Optional[str] = None
    status: Optional[RequestStatus] = None
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
