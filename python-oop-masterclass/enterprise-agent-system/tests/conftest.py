"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.

Demonstrates:
- Test fixture pattern
- Dependency injection for tests
- Test data factories
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.domain.models import (
    Money,
    ContactInfo,
    Customer,
    CustomerRequest,
    Priority,
    RequestCategory,
    RequestStatus
)
from src.agents.state import AgentState, StateBuilder


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# DOMAIN MODEL FIXTURES
# ============================================================================

@pytest.fixture
def sample_money():
    """Create sample money value object."""
    return Money(amount=100.50, currency="USD")


@pytest.fixture
def sample_contact_info():
    """Create sample contact info."""
    return ContactInfo(
        email="test@example.com",
        phone="+1234567890"
    )


@pytest.fixture
def sample_customer(sample_contact_info):
    """Create sample customer."""
    return Customer(
        customer_id="cust-test-123",
        name="Test Customer",
        contact_info=sample_contact_info,
        tier="premium",
        metadata={"test": True}
    )


@pytest.fixture
def sample_customer_request(sample_customer):
    """Create sample customer request."""
    return CustomerRequest(
        request_id="req-test-123",
        customer=sample_customer,
        initial_message="Test message",
        priority=Priority.MEDIUM
    )


# ============================================================================
# STATE FIXTURES
# ============================================================================

@pytest.fixture
def sample_agent_state():
    """Create sample agent state."""
    return (
        StateBuilder()
        .with_request_id("req-test-123")
        .with_customer_id("cust-test-123")
        .with_thread_id("thread-test-123")
        .with_initial_message("How do I reset my password?")
        .with_priority(Priority.MEDIUM)
        .with_category(RequestCategory.ACCOUNT)
        .build()
    )


@pytest.fixture
def empty_agent_state():
    """Create empty agent state."""
    return AgentState(
        request_id="req-empty",
        customer_id="cust-empty",
        thread_id="thread-empty",
        messages=[],
        decisions=[],
        relevant_context_ids=set(),
        metrics={},
        session_data={},
        status=RequestStatus.PENDING.value,
        priority=2
    )


# ============================================================================
# MOCK SERVICES
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    class MockVectorStore:
        def __init__(self):
            self.documents = {}

        async def add_documents(self, docs, namespace="default"):
            for doc in docs:
                self.documents[doc.id] = doc
            return True

        async def similarity_search(self, query, k=5, namespace="default", filters=None):
            # Return mock results
            from src.memory.vector_store import SearchResult
            return [
                SearchResult(
                    id=f"doc-{i}",
                    content=f"Mock document {i} for query: {query}",
                    score=0.9 - (i * 0.1),
                    metadata={"mock": True}
                )
                for i in range(min(k, 3))
            ]

        async def get_document(self, doc_id, namespace="default"):
            return self.documents.get(doc_id)

        async def delete_documents(self, doc_ids, namespace="default"):
            for doc_id in doc_ids:
                self.documents.pop(doc_id, None)
            return True

        async def ping(self):
            return True

    return MockVectorStore()


@pytest.fixture
def mock_redis_cache():
    """Create mock Redis cache."""
    class MockRedisCache:
        def __init__(self):
            self.cache = {}

        async def get(self, key):
            return self.cache.get(key.to_string() if hasattr(key, 'to_string') else str(key))

        async def set(self, key, value, ttl=None):
            self.cache[key.to_string() if hasattr(key, 'to_string') else str(key)] = value
            return True

        async def delete(self, key):
            key_str = key.to_string() if hasattr(key, 'to_string') else str(key)
            if key_str in self.cache:
                del self.cache[key_str]
                return True
            return False

        async def exists(self, key):
            return (key.to_string() if hasattr(key, 'to_string') else str(key)) in self.cache

        async def ping(self):
            return True

        async def get_stats(self):
            from src.memory.redis_cache import CacheStats
            return CacheStats(
                total_keys=len(self.cache),
                hits=10,
                misses=2,
                hit_rate=0.83
            )

    return MockRedisCache()


@pytest.fixture
def mock_event_producer():
    """Create mock event producer."""
    class MockEventProducer:
        def __init__(self):
            self.published_events = []

        async def publish(self, event):
            self.published_events.append(event)
            from src.infrastructure.kafka.producer import PublishResult
            return PublishResult(
                success=True,
                event_id=event.event_id,
                topic="test-topic",
                partition=0,
                offset=len(self.published_events)
            )

        async def publish_batch(self, events):
            results = []
            for event in events:
                result = await self.publish(event)
                results.append(result)
            return results

        async def start(self):
            pass

        async def stop(self):
            pass

    return MockEventProducer()


# ============================================================================
# MEMORY MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def mock_memory_manager(mock_vector_store, mock_redis_cache):
    """Create mock memory manager."""
    from src.memory.manager import MemoryManager
    return MemoryManager(mock_vector_store, mock_redis_cache)


# ============================================================================
# TEST DATA FACTORIES
# ============================================================================

class CustomerFactory:
    """Factory for creating test customers."""

    @staticmethod
    def create(customer_id="test-cust", **kwargs):
        """Create test customer."""
        defaults = {
            "name": "Test Customer",
            "contact_info": ContactInfo(
                email="test@example.com",
                phone="+1234567890"
            ),
            "tier": "standard",
            "metadata": {}
        }
        defaults.update(kwargs)
        return Customer(customer_id=customer_id, **defaults)


class RequestFactory:
    """Factory for creating test requests."""

    @staticmethod
    def create(request_id="test-req", customer=None, **kwargs):
        """Create test request."""
        if customer is None:
            customer = CustomerFactory.create()

        defaults = {
            "initial_message": "Test message",
            "priority": Priority.MEDIUM
        }
        defaults.update(kwargs)

        return CustomerRequest(
            request_id=request_id,
            customer=customer,
            **defaults
        )


class StateFactory:
    """Factory for creating test states."""

    @staticmethod
    def create(**kwargs):
        """Create test state."""
        builder = StateBuilder()

        # Set defaults
        builder = (builder
                   .with_request_id(kwargs.get("request_id", "test-req"))
                   .with_customer_id(kwargs.get("customer_id", "test-cust"))
                   .with_thread_id(kwargs.get("thread_id", "test-thread")))

        if "message" in kwargs:
            builder = builder.with_initial_message(kwargs["message"])

        if "priority" in kwargs:
            builder = builder.with_priority(kwargs["priority"])

        if "category" in kwargs:
            builder = builder.with_category(kwargs["category"])

        return builder.build()


@pytest.fixture
def customer_factory():
    """Customer factory fixture."""
    return CustomerFactory


@pytest.fixture
def request_factory():
    """Request factory fixture."""
    return RequestFactory


@pytest.fixture
def state_factory():
    """State factory fixture."""
    return StateFactory


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def sample_messages():
    """Sample message list."""
    from langchain_core.messages import HumanMessage, AIMessage
    return [
        HumanMessage(content="Hello, I need help"),
        AIMessage(content="I'm here to help!"),
        HumanMessage(content="How do I reset my password?")
    ]


@pytest.fixture
def sample_decisions():
    """Sample decision list."""
    return [
        {
            "agent_type": "triage",
            "decision_type": "classify",
            "confidence": 0.85,
            "reasoning": "Classified as account category",
            "timestamp": datetime.now().isoformat()
        },
        {
            "agent_type": "research",
            "decision_type": "retrieve",
            "confidence": 0.92,
            "reasoning": "Retrieved 5 relevant documents",
            "timestamp": datetime.now().isoformat()
        }
    ]


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {
        "triage_duration_avg": 450.5,
        "triage_success_count": 1,
        "research_duration_avg": 1200.3,
        "research_success_count": 1
    }
