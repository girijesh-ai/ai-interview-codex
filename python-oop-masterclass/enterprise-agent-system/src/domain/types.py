"""
Strong Type Definitions for Enterprise Agent System

This module provides type-safe ID types using NewType to prevent
mixing different ID types and improve code clarity.

Benefits:
- Type checker catches ID mismatches
- Self-documenting code
- No runtime overhead (NewType is erased at runtime)
- IDE autocomplete support
"""

from typing import NewType
from uuid import UUID, uuid4

# ============================================================================
# STRONG ID TYPES - Type Safety
# ============================================================================

# Request IDs
RequestId = NewType('RequestId', str)
"""Type-safe request identifier (e.g., 'req-123e4567...')"""

# Customer IDs
CustomerId = NewType('CustomerId', str)
"""Type-safe customer identifier (e.g., 'cust-123e4567...')"""

# Thread/Session IDs
ThreadId = NewType('ThreadId', str)
"""Type-safe thread identifier for conversation tracking"""

SessionId = NewType('SessionId', str)
"""Type-safe session identifier for user sessions"""

# Agent IDs
AgentId = NewType('AgentId', str)
"""Type-safe agent identifier"""

# Decision IDs
DecisionId = NewType('DecisionId', str)
"""Type-safe decision identifier"""

# Message IDs
MessageId = NewType('MessageId', str)
"""Type-safe message identifier"""

# Document IDs (for RAG)
DocumentId = NewType('DocumentId', str)
"""Type-safe document identifier in vector store"""


# ============================================================================
# ID GENERATORS - Factory Functions
# ============================================================================

def generate_request_id() -> RequestId:
    """Generate a unique request ID.

    Returns:
        RequestId: New request identifier

    Example:
        >>> req_id = generate_request_id()
        >>> req_id
        'req-123e4567-e89b-12d3-a456-426614174000'
    """
    return RequestId(f"req-{uuid4()}")


def generate_customer_id() -> CustomerId:
    """Generate a unique customer ID.

    Returns:
        CustomerId: New customer identifier
    """
    return CustomerId(f"cust-{uuid4()}")


def generate_thread_id() -> ThreadId:
    """Generate a unique thread ID.

    Returns:
        ThreadId: New thread identifier
    """
    return ThreadId(f"thread-{uuid4()}")


def generate_session_id() -> SessionId:
    """Generate a unique session ID.

    Returns:
        SessionId: New session identifier
    """
    return SessionId(f"session-{uuid4()}")


def generate_agent_id() -> AgentId:
    """Generate a unique agent ID.

    Returns:
        AgentId: New agent identifier
    """
    return AgentId(f"agent-{uuid4()}")


def generate_decision_id() -> DecisionId:
    """Generate a unique decision ID.

    Returns:
        DecisionId: New decision identifier
    """
    return DecisionId(f"decision-{uuid4()}")


def generate_message_id() -> MessageId:
    """Generate a unique message ID.

    Returns:
        MessageId: New message identifier
    """
    return MessageId(f"msg-{uuid4()}")


def generate_document_id() -> DocumentId:
    """Generate a unique document ID.

    Returns:
        DocumentId: New document identifier
    """
    return DocumentId(f"doc-{uuid4()}")


# ============================================================================
# ID VALIDATION
# ============================================================================

def is_valid_request_id(value: str) -> bool:
    """Check if string is a valid request ID format.

    Args:
        value: String to validate

    Returns:
        bool: True if valid request ID format
    """
    return value.startswith("req-") and len(value) > 4


def is_valid_customer_id(value: str) -> bool:
    """Check if string is a valid customer ID format.

    Args:
        value: String to validate

    Returns:
        bool: True if valid customer ID format
    """
    return value.startswith("cust-") and len(value) > 5


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example Usage:

# Type-safe function signatures
def process_request(request_id: RequestId, customer_id: CustomerId) -> None:
    '''Process a customer request.

    Type checker ensures you can't accidentally pass wrong ID types.
    '''
    pass

# Generate IDs
req_id = generate_request_id()
cust_id = generate_customer_id()

# This works - types match
process_request(req_id, cust_id)

# This fails type checking - types don't match
process_request(cust_id, req_id)  # Type error!

# This also fails - can't pass plain string
process_request("req-123", "cust-456")  # Type error!

# Must explicitly cast if needed (e.g., from database)
db_req_id = "req-123"
process_request(RequestId(db_req_id), cust_id)  # OK with explicit cast

# Type-safe collections
requests: dict[RequestId, str] = {}
customers: dict[CustomerId, str] = {}

requests[req_id] = "data"  # OK
requests[cust_id] = "data"  # Type error!
"""
