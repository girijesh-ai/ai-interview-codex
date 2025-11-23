"""
Enhanced API Models with Comprehensive Validation

Improvements over original models.py:
- Field-level validation with regex patterns
- Cross-field validation
- Security checks (XSS, injection prevention)
- Size limits on all fields
- Better error messages
- Pydantic V2 best practices
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import re


# ============================================================================
# ENUMS
# ============================================================================

class RequestStatus(str, Enum):
    """Request status with string values."""
    PENDING = "pending"
    PROCESSING = "processing"
    TRIAGED = "triaged"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    AWAITING_APPROVAL = "awaiting_approval"


class Priority(int, Enum):
    """Request priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Category(str, Enum):
    """Request categories."""
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    PRODUCT = "product"
    REFUND = "refund"
    GENERAL = "general"


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# ID format patterns
CUSTOMER_ID_PATTERN = r'^cust-[a-zA-Z0-9-]{8,100}$'
REQUEST_ID_PATTERN = r'^req-[a-zA-Z0-9-]{8,100}$'

# Dangerous content patterns (XSS prevention)
DANGEROUS_PATTERNS = [
    r'<script[^>]*>',
    r'javascript:',
    r'onerror\s*=',
    r'onclick\s*=',
    r'onload\s*=',
    r'<iframe[^>]*>',
    r'eval\s*\(',
    r'expression\s*\(',
]

# Compile regex patterns
CUSTOMER_ID_REGEX = re.compile(CUSTOMER_ID_PATTERN)
REQUEST_ID_REGEX = re.compile(REQUEST_ID_PATTERN)
DANGEROUS_CONTENT_REGEX = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


# ============================================================================
# BASE MODELS WITH COMMON VALIDATION
# ============================================================================

class BaseAPIModel(BaseModel):
    """Base model with common configuration."""

    model_config = {
        "str_strip_whitespace": True,  # Auto-strip whitespace
        "validate_assignment": True,    # Validate on assignment
        "use_enum_values": True,        # Use enum values not names
    }


# ============================================================================
# REQUEST MODELS
# ============================================================================

class CreateRequestModel(BaseAPIModel):
    """Create customer request with comprehensive validation."""

    customer_id: str = Field(
        ...,
        min_length=8,
        max_length=150,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in format 'cust-{uuid}'",
        examples=["cust-123e4567-e89b-12d3-a456-426614174000"]
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Customer message (1-10000 characters)",
        examples=["How do I reset my password?"]
    )

    category: Optional[Category] = Field(
        default=None,
        description="Request category (auto-detected if not provided)"
    )

    priority: Optional[Priority] = Field(
        default=Priority.MEDIUM,
        description="Priority level (1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (max 10KB JSON)"
    )

    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        """Validate message content for security and quality.

        Checks:
        - No dangerous HTML/JavaScript patterns
        - Reasonable character distribution
        - Not just whitespace or special characters

        Args:
            v: Message content

        Returns:
            str: Validated message

        Raises:
            ValueError: If message contains forbidden content
        """
        # Check for dangerous patterns (XSS prevention)
        message_lower = v.lower()
        for pattern in DANGEROUS_CONTENT_REGEX:
            if pattern.search(message_lower):
                raise ValueError(
                    f"Message contains forbidden pattern. "
                    f"Please remove HTML/JavaScript content."
                )

        # Check special character ratio
        if len(v) > 0:
            special_char_count = sum(
                1 for c in v if not c.isalnum() and not c.isspace()
            )
            special_char_ratio = special_char_count / len(v)

            if special_char_ratio > 0.5:
                raise ValueError(
                    "Message contains too many special characters "
                    f"({special_char_ratio:.0%}). Maximum allowed: 50%"
                )

        # Check for actual content (not just whitespace/punctuation)
        alphanumeric_count = sum(1 for c in v if c.isalnum())
        if alphanumeric_count < 3:
            raise ValueError(
                "Message must contain at least 3 alphanumeric characters"
            )

        return v.strip()

    @field_validator('metadata')
    @classmethod
    def validate_metadata_size(cls, v: Optional[Dict]) -> Optional[Dict]:
        """Validate metadata size and content.

        Args:
            v: Metadata dictionary

        Returns:
            Optional[Dict]: Validated metadata

        Raises:
            ValueError: If metadata is too large or invalid
        """
        if v is None:
            return v

        # Check size
        import json
        try:
            metadata_json = json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")

        if len(metadata_json) > 10240:  # 10KB
            raise ValueError(
                f"Metadata too large: {len(metadata_json)} bytes (max 10KB)"
            )

        # Check nesting depth
        def get_depth(obj, current_depth=0):
            if not isinstance(obj, dict):
                return current_depth
            if not obj:
                return current_depth + 1
            return max(get_depth(v, current_depth + 1) for v in obj.values())

        if get_depth(v) > 5:
            raise ValueError("Metadata nesting too deep (max 5 levels)")

        return v

    @model_validator(mode='after')
    def validate_business_rules(self) -> 'CreateRequestModel':
        """Cross-field validation for business rules.

        Returns:
            CreateRequestModel: Validated model

        Raises:
            ValueError: If business rules are violated
        """
        # Business rule: Refund requests must have priority >= MEDIUM
        if self.category == Category.REFUND and self.priority == Priority.LOW:
            raise ValueError(
                "Refund requests must have priority >= MEDIUM (got LOW)"
            )

        # Business rule: Critical priority requires explanation in message
        if self.priority == Priority.CRITICAL:
            urgent_keywords = ['urgent', 'emergency', 'critical', 'asap']
            if not any(kw in self.message.lower() for kw in urgent_keywords):
                raise ValueError(
                    "CRITICAL priority requires explanation. "
                    "Please include urgency keywords: urgent, emergency, critical, or asap"
                )

        return self

    model_config = {
        **BaseAPIModel.model_config,
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "cust-123e4567-e89b-12d3-a456-426614174000",
                    "message": "How do I reset my password?",
                    "category": "account",
                    "priority": 2
                },
                {
                    "customer_id": "cust-987fcdeb-51a2-43f7-b9c3-123456789abc",
                    "message": "URGENT: Need refund for order #12345",
                    "category": "refund",
                    "priority": 4,
                    "metadata": {"order_id": "12345", "amount": "99.99"}
                }
            ]
        }
    }


class UpdateRequestModel(BaseAPIModel):
    """Update existing request."""

    message: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=10000,
        description="Additional message or update"
    )

    priority: Optional[Priority] = Field(
        default=None,
        description="Updated priority level"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata (merged with existing)"
    )

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: Optional[str]) -> Optional[str]:
        """Validate message if provided."""
        if v is not None:
            # Reuse same validation as CreateRequestModel
            return CreateRequestModel.validate_message_content(v)
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class RequestResponseModel(BaseAPIModel):
    """Request response with full details."""

    request_id: str = Field(
        ...,
        pattern=REQUEST_ID_PATTERN,
        description="Request identifier"
    )

    customer_id: str = Field(
        ...,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier"
    )

    status: RequestStatus = Field(
        ...,
        description="Current request status"
    )

    category: Optional[str] = Field(
        default=None,
        description="Detected or assigned category"
    )

    priority: int = Field(
        ...,
        ge=1,
        le=4,
        description="Priority level (1-4)"
    )

    message: str = Field(
        ...,
        description="Original customer message"
    )

    solution: Optional[str] = Field(
        default=None,
        description="Generated solution (if completed)"
    )

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in solution (0.0-1.0)"
    )

    requires_approval: bool = Field(
        default=False,
        description="Whether request requires human approval"
    )

    created_at: datetime = Field(
        ...,
        description="Request creation timestamp"
    )

    updated_at: datetime = Field(
        ...,
        description="Last update timestamp"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional request metadata"
    )

    model_config = {
        **BaseAPIModel.model_config,
        "json_schema_extra": {
            "example": {
                "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
                "customer_id": "cust-987fcdeb-51a2-43f7-b9c3-123456789abc",
                "status": "completed",
                "category": "account",
                "priority": 2,
                "message": "How do I reset my password?",
                "solution": "You can reset your password by...",
                "confidence_score": 0.95,
                "requires_approval": False,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:31:23Z"
            }
        }
    }


class RequestListResponse(BaseAPIModel):
    """Paginated list of requests."""

    requests: List[RequestResponseModel] = Field(
        ...,
        description="List of requests"
    )

    total: int = Field(
        ...,
        ge=0,
        description="Total number of requests"
    )

    page: int = Field(
        ...,
        ge=1,
        description="Current page number"
    )

    page_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of items per page"
    )

    has_next: bool = Field(
        ...,
        description="Whether there are more pages"
    )

    @model_validator(mode='after')
    def compute_has_next(self) -> 'RequestListResponse':
        """Compute has_next based on total and pagination."""
        self.has_next = (self.page * self.page_size) < self.total
        return self


# ============================================================================
# FILTER & PAGINATION MODELS
# ============================================================================

class PaginationParams(BaseAPIModel):
    """Pagination parameters."""

    page: int = Field(
        default=1,
        ge=1,
        le=10000,
        description="Page number (1-indexed)"
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page (max 100)"
    )


class RequestFilterModel(BaseAPIModel):
    """Request filter parameters."""

    status: Optional[RequestStatus] = Field(
        default=None,
        description="Filter by status"
    )

    category: Optional[Category] = Field(
        default=None,
        description="Filter by category"
    )

    priority: Optional[Priority] = Field(
        default=None,
        description="Filter by priority"
    )

    customer_id: Optional[str] = Field(
        default=None,
        pattern=CUSTOMER_ID_PATTERN,
        description="Filter by customer ID"
    )

    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter by creation date (after)"
    )

    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter by creation date (before)"
    )

    @model_validator(mode='after')
    def validate_date_range(self) -> 'RequestFilterModel':
        """Validate date range if both provided."""
        if self.created_after and self.created_before:
            if self.created_after >= self.created_before:
                raise ValueError(
                    "created_after must be before created_before"
                )
        return self


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorModel(BaseAPIModel):
    """Standardized error response."""

    error: str = Field(
        ...,
        description="Error message"
    )

    code: str = Field(
        ...,
        description="Error code"
    )

    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )

    field_errors: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Field-specific validation errors"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing"
    )


# ============================================================================
# APPROVAL MODELS
# ============================================================================

class ApprovalRequestModel(BaseAPIModel):
    """Request approval/rejection."""

    approved: bool = Field(
        ...,
        description="Whether to approve (true) or reject (false)"
    )

    approver: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Approver identifier or name"
    )

    notes: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Approval/rejection notes"
    )

    @field_validator('notes')
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        """Validate approval notes."""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        return v


class ApprovalResponseModel(BaseAPIModel):
    """Approval response."""

    request_id: str
    approved: bool
    approver: str
    notes: Optional[str] = None
    approved_at: datetime


# ============================================================================
# SESSION MODELS
# ============================================================================

class SessionModel(BaseAPIModel):
    """Session information."""
    session_id: str
    customer_id: str
    created_at: datetime
    last_activity: datetime
    data: Dict[str, Any]


# ============================================================================
# METRICS MODELS
# ============================================================================

class MetricsModel(BaseAPIModel):
    """System metrics."""
    total_requests: int
    completed_requests: int
    failed_requests: int
    avg_resolution_time_seconds: float
    cache_hit_rate: float
    active_sessions: int
    timestamp: datetime


class AgentMetricsModel(BaseAPIModel):
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

class ComponentHealthModel(BaseAPIModel):
    """Component health status."""
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckModel(BaseAPIModel):
    """Health check response."""
    overall_status: str
    components: Dict[str, ComponentHealthModel]
    timestamp: datetime


# ============================================================================
# WEBSOCKET MODELS
# ============================================================================

class WebSocketMessage(BaseAPIModel):
    """WebSocket message."""
    type: str  # subscribe, unsubscribe, event, error
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionModel(BaseAPIModel):
    """WebSocket subscription."""
    request_id: Optional[str] = None
    customer_id: Optional[str] = None
    event_types: List[str] = Field(default_factory=list)
