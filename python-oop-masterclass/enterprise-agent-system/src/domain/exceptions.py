"""
Domain Exception Classes

Custom exceptions for better error handling and debugging.
Fixes BUG-009: Proper exception handling instead of silent failures.
"""

from typing import Any, Dict, Optional


# ============================================================================
# BASE DOMAIN EXCEPTION
# ============================================================================

class DomainException(Exception):
    """Base exception for all domain-level errors.

    Provides structured error information including error codes and context.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize domain exception.

        Args:
            message: Human-readable error message
            code: Machine-readable error code (default: class name)
            details: Additional error context
        """
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.upper()
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses.

        Returns:
            Dict containing error information
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }

    def __str__(self) -> str:
        """String representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"[{self.code}] {self.message} ({details_str})"
        return f"[{self.code}] {self.message}"


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(DomainException):
    """Raised when input validation fails.

    Used for user input, API request validation, etc.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        **details: Any
    ):
        """Initialize validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            **details: Additional validation context
        """
        all_details = {"field": field, **details} if field else details
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=all_details
        )


class BusinessRuleViolation(DomainException):
    """Raised when a business rule is violated.

    Used for domain logic constraints that user shouldn't violate.
    """

    def __init__(
        self,
        message: str,
        rule: str,
        **details: Any
    ):
        """Initialize business rule violation.

        Args:
            message: Error message
            rule: Name of the violated business rule
            **details: Additional context
        """
        super().__init__(
            message=message,
            code="BUSINESS_RULE_VIOLATION",
            details={"rule": rule, **details}
        )


class InvariantViolation(DomainException):
    """Raised when a domain invariant is violated.

    Invariants are conditions that must always be true.
    """

    def __init__(
        self,
        message: str,
        invariant: str,
        **details: Any
    ):
        """Initialize invariant violation.

        Args:
            message: Error message
            invariant: Name of the violated invariant
            **details: Additional context
        """
        super().__init__(
            message=message,
            code="INVARIANT_VIOLATION",
            details={"invariant": invariant, **details}
        )


# ============================================================================
# RESOURCE EXCEPTIONS
# ============================================================================

class ResourceNotFound(DomainException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        **details: Any
    ):
        """Initialize resource not found error.

        Args:
            resource_type: Type of resource (e.g., "Customer", "Request")
            resource_id: ID of the resource
            **details: Additional context
        """
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="RESOURCE_NOT_FOUND",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                **details
            }
        )


class ResourceAlreadyExists(DomainException):
    """Raised when attempting to create a resource that already exists."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        **details: Any
    ):
        """Initialize resource already exists error.

        Args:
            resource_type: Type of resource
            resource_id: ID of the resource
            **details: Additional context
        """
        super().__init__(
            message=f"{resource_type} already exists: {resource_id}",
            code="RESOURCE_ALREADY_EXISTS",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                **details
            }
        )


# ============================================================================
# CONCURRENCY EXCEPTIONS
# ============================================================================

class ConcurrencyConflict(DomainException):
    """Raised when a concurrent modification conflict occurs."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **details: Any
    ):
        """Initialize concurrency conflict error.

        Args:
            message: Error message
            resource_type: Type of resource with conflict
            resource_id: ID of the resource
            **details: Additional context
        """
        all_details = details.copy()
        if resource_type:
            all_details["resource_type"] = resource_type
        if resource_id:
            all_details["resource_id"] = resource_id

        super().__init__(
            message=message,
            code="CONCURRENCY_CONFLICT",
            details=all_details
        )


# ============================================================================
# AGENT EXCEPTIONS
# ============================================================================

class AgentExecutionError(DomainException):
    """Raised when an agent fails to execute.

    Fixes BUG-009: Silent agent failures.
    """

    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        state_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        **details: Any
    ):
        """Initialize agent execution error.

        Args:
            message: Error message
            agent_type: Type of agent that failed
            state_id: ID of the state being processed
            original_error: Original exception that caused the failure
            **details: Additional context
        """
        all_details = details.copy()
        if agent_type:
            all_details["agent_type"] = agent_type
        if state_id:
            all_details["state_id"] = state_id
        if original_error:
            all_details["original_error"] = str(original_error)
            all_details["original_error_type"] = type(original_error).__name__

        super().__init__(
            message=message,
            code="AGENT_EXECUTION_ERROR",
            details=all_details
        )

        # Store original exception for __cause__ chain
        if original_error:
            self.__cause__ = original_error


class WorkflowError(DomainException):
    """Raised when a workflow fails to complete."""

    def __init__(
        self,
        message: str,
        workflow_stage: Optional[str] = None,
        request_id: Optional[str] = None,
        **details: Any
    ):
        """Initialize workflow error.

        Args:
            message: Error message
            workflow_stage: Stage where workflow failed
            request_id: ID of the request being processed
            **details: Additional context
        """
        all_details = details.copy()
        if workflow_stage:
            all_details["workflow_stage"] = workflow_stage
        if request_id:
            all_details["request_id"] = request_id

        super().__init__(
            message=message,
            code="WORKFLOW_ERROR",
            details=all_details
        )


# ============================================================================
# INTEGRATION EXCEPTIONS
# ============================================================================

class ExternalServiceError(DomainException):
    """Raised when an external service call fails."""

    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: Optional[int] = None,
        **details: Any
    ):
        """Initialize external service error.

        Args:
            message: Error message
            service_name: Name of the external service
            status_code: HTTP status code (if applicable)
            **details: Additional context
        """
        all_details = {"service_name": service_name, **details}
        if status_code:
            all_details["status_code"] = status_code

        super().__init__(
            message=message,
            code="EXTERNAL_SERVICE_ERROR",
            details=all_details
        )


class RateLimitExceeded(DomainException):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: int,
        window_seconds: int,
        retry_after_seconds: Optional[int] = None,
        **details: Any
    ):
        """Initialize rate limit exceeded error.

        Args:
            message: Error message
            limit: Number of allowed requests
            window_seconds: Time window in seconds
            retry_after_seconds: Seconds until retry is allowed
            **details: Additional context
        """
        all_details = {
            "limit": limit,
            "window_seconds": window_seconds,
            **details
        }
        if retry_after_seconds:
            all_details["retry_after_seconds"] = retry_after_seconds

        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            details=all_details
        )


# ============================================================================
# AUTHENTICATION & AUTHORIZATION EXCEPTIONS
# ============================================================================

class AuthenticationError(DomainException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **details: Any):
        """Initialize authentication error.

        Args:
            message: Error message
            **details: Additional context
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(DomainException):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        **details: Any
    ):
        """Initialize authorization error.

        Args:
            message: Error message
            required_permission: Permission that was required
            **details: Additional context
        """
        all_details = details.copy()
        if required_permission:
            all_details["required_permission"] = required_permission

        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            details=all_details
        )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(DomainException):
    """Raised when system configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **details: Any
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid
            **details: Additional context
        """
        all_details = details.copy()
        if config_key:
            all_details["config_key"] = config_key

        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            details=all_details
        )
