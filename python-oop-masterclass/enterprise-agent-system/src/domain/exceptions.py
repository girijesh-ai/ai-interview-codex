"""
Domain Exception Classes

Essential custom exceptions for error handling and debugging.
Fixes BUG-009: Proper exception handling instead of silent failures.

Kept only exceptions that are actually used or essential for production.
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
# AGENT EXCEPTIONS
# ============================================================================

class AgentExecutionError(DomainException):
    """Raised when an agent fails to execute.

    Fixes BUG-009: Silent agent failures.
    Actually used in src/agents/nodes.py
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
