"""
Memory Manager - Coordinates 3-Tier Memory System

Architecture:
- Tier 1: LangGraph State (Short-term - Current session working memory)
- Tier 2: Redis Cache (Session - Minutes to hours)
- Tier 3: Vector DB (Long-term - Persistent semantic memory)

Demonstrates:
- Facade pattern for unified interface
- Mediator pattern for coordinating memory tiers
- Strategy pattern for retrieval strategies
- Command pattern for memory operations
- Observer pattern for memory events
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..agents.state import AgentState, StateUtils
from .vector_store import VectorStore, SearchResult, Document
from .redis_cache import CacheRepository, CacheKey, CacheNamespace, SessionCache

logger = logging.getLogger(__name__)


# ============================================================================
# VALUE OBJECTS & ENUMS
# ============================================================================

class MemoryTier(str, Enum):
    """Memory tier levels."""
    STATE = "state"  # LangGraph state (short-term)
    CACHE = "cache"  # Redis cache (session)
    VECTOR = "vector"  # Vector DB (long-term)


class RetrievalStrategy(str, Enum):
    """Context retrieval strategies."""
    RECENT_FIRST = "recent_first"  # Most recent interactions
    RELEVANT_FIRST = "relevant_first"  # Most semantically relevant
    HYBRID = "hybrid"  # Combine recency and relevance
    FULL_HISTORY = "full_history"  # Complete conversation history


@dataclass
class MemoryContext:
    """Consolidated memory context from all tiers.

    Demonstrates:
    - Value object pattern
    - Aggregation of multiple sources
    """
    # From State (Tier 1)
    current_messages: List[Dict[str, Any]] = field(default_factory=list)
    current_decisions: List[Dict[str, Any]] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)

    # From Cache (Tier 2)
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_metrics: Dict[str, float] = field(default_factory=dict)

    # From Vector DB (Tier 3)
    relevant_documents: List[SearchResult] = field(default_factory=list)
    similar_conversations: List[SearchResult] = field(default_factory=list)
    knowledge_base: List[SearchResult] = field(default_factory=list)

    # Metadata
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    retrieved_at: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get memory context summary."""
        return {
            "current_messages": len(self.current_messages),
            "decisions": len(self.current_decisions),
            "recent_interactions": len(self.recent_interactions),
            "relevant_docs": len(self.relevant_documents),
            "similar_conversations": len(self.similar_conversations),
            "kb_articles": len(self.knowledge_base),
            "retrieval_strategy": self.retrieval_strategy.value,
            "total_tokens": self.total_tokens
        }


@dataclass
class MemoryStats:
    """Memory system statistics."""
    state_channels: int = 0
    cache_keys: int = 0
    vector_documents: int = 0
    cache_hit_rate: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    total_retrievals: int = 0


# ============================================================================
# RETRIEVAL STRATEGY INTERFACE
# ============================================================================

class ContextRetrievalStrategy(ABC):
    """Abstract strategy for context retrieval.

    Demonstrates:
    - Strategy pattern
    - Open/closed principle (OCP)
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        state: AgentState,
        cache: CacheRepository,
        vector_store: VectorStore,
        max_items: int = 10
    ) -> MemoryContext:
        """Retrieve context using this strategy."""
        pass


class RecentFirstStrategy(ContextRetrievalStrategy):
    """Prioritize recent interactions."""

    async def retrieve(
        self,
        query: str,
        state: AgentState,
        cache: CacheRepository,
        vector_store: VectorStore,
        max_items: int = 10
    ) -> MemoryContext:
        """Retrieve recent-first context."""
        context = MemoryContext(retrieval_strategy=RetrievalStrategy.RECENT_FIRST)

        # Tier 1: Current state (most recent)
        context.current_messages = StateUtils.get_message_history(state, limit=5)
        context.current_decisions = state.get("decisions", [])[-5:]
        context.session_data = state.get("session_data", {})

        # Tier 2: Recent cache
        request_id = state.get("request_id")
        if request_id:
            cache_key = CacheKey(CacheNamespace.USER, request_id, "recent")
            cached_recent = await cache.get(cache_key)
            if cached_recent:
                context.recent_interactions = cached_recent[:max_items]

        # Tier 3: Recent from vector DB (if needed)
        if len(context.recent_interactions) < max_items:
            results = await vector_store.similarity_search(
                query,
                k=max_items - len(context.recent_interactions),
                namespace="conversations"
            )
            context.similar_conversations = results

        return context


class RelevantFirstStrategy(ContextRetrievalStrategy):
    """Prioritize semantic relevance."""

    async def retrieve(
        self,
        query: str,
        state: AgentState,
        cache: CacheRepository,
        vector_store: VectorStore,
        max_items: int = 10
    ) -> MemoryContext:
        """Retrieve relevance-first context."""
        context = MemoryContext(retrieval_strategy=RetrievalStrategy.RELEVANT_FIRST)

        # Tier 1: Current state
        context.current_messages = StateUtils.get_message_history(state, limit=3)
        context.current_decisions = state.get("decisions", [])[-3:]

        # Tier 3: Relevant documents (priority)
        kb_results = await vector_store.similarity_search(
            query,
            k=max_items,
            namespace="knowledge_base"
        )
        context.knowledge_base = kb_results

        # Similar conversations
        conv_results = await vector_store.similarity_search(
            query,
            k=5,
            namespace="conversations"
        )
        context.similar_conversations = conv_results

        # Tier 2: User preferences
        customer_id = state.get("customer_id")
        if customer_id:
            cache_key = CacheKey(CacheNamespace.USER, customer_id, "preferences")
            prefs = await cache.get(cache_key)
            if prefs:
                context.user_preferences = prefs

        return context


class HybridStrategy(ContextRetrievalStrategy):
    """Balanced approach combining recency and relevance."""

    async def retrieve(
        self,
        query: str,
        state: AgentState,
        cache: CacheRepository,
        vector_store: VectorStore,
        max_items: int = 10
    ) -> MemoryContext:
        """Retrieve hybrid context."""
        context = MemoryContext(retrieval_strategy=RetrievalStrategy.HYBRID)

        # Tier 1: Current state (always include)
        context.current_messages = StateUtils.get_message_history(state, limit=5)
        context.current_decisions = state.get("decisions", [])[-5:]
        context.session_data = state.get("session_data", {})

        # Tier 2: Recent interactions + preferences
        request_id = state.get("request_id")
        customer_id = state.get("customer_id")

        if request_id:
            cache_key = CacheKey(CacheNamespace.USER, request_id, "recent")
            cached_recent = await cache.get(cache_key)
            if cached_recent:
                context.recent_interactions = cached_recent[:3]

        if customer_id:
            cache_key = CacheKey(CacheNamespace.USER, customer_id, "preferences")
            prefs = await cache.get(cache_key)
            if prefs:
                context.user_preferences = prefs

        # Tier 3: Relevant documents (half recent, half relevant)
        kb_results = await vector_store.similarity_search(
            query,
            k=max_items // 2,
            namespace="knowledge_base"
        )
        context.knowledge_base = kb_results

        conv_results = await vector_store.similarity_search(
            query,
            k=max_items // 2,
            namespace="conversations"
        )
        context.similar_conversations = conv_results

        return context


class StrategyFactory:
    """Factory for creating retrieval strategies.

    Demonstrates:
    - Factory pattern
    """

    @staticmethod
    def create(strategy: RetrievalStrategy) -> ContextRetrievalStrategy:
        """Create retrieval strategy."""
        if strategy == RetrievalStrategy.RECENT_FIRST:
            return RecentFirstStrategy()
        elif strategy == RetrievalStrategy.RELEVANT_FIRST:
            return RelevantFirstStrategy()
        elif strategy == RetrievalStrategy.HYBRID:
            return HybridStrategy()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """Coordinates 3-tier memory system.

    Demonstrates:
    - Facade pattern (unified interface)
    - Mediator pattern (coordinates components)
    - Strategy pattern (retrieval strategies)
    - Single responsibility principle (SRP)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        cache: CacheRepository,
        default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ):
        """Initialize memory manager.

        Args:
            vector_store: Vector database
            cache: Redis cache
            default_strategy: Default retrieval strategy
        """
        self.vector_store = vector_store
        self.cache = cache
        self.session_cache = SessionCache(cache)
        self.default_strategy = default_strategy
        self._stats = MemoryStats()

    # ========================================================================
    # CONTEXT RETRIEVAL
    # ========================================================================

    async def get_context(
        self,
        query: str,
        state: AgentState,
        strategy: Optional[RetrievalStrategy] = None,
        max_items: int = 10
    ) -> MemoryContext:
        """Retrieve context from all memory tiers.

        Args:
            query: Query string
            state: Current agent state (Tier 1)
            strategy: Retrieval strategy
            max_items: Maximum items to retrieve

        Returns:
            Consolidated memory context
        """
        start_time = datetime.now()
        strategy = strategy or self.default_strategy

        logger.info(f"Retrieving context with strategy: {strategy.value}")

        # Get strategy implementation
        retrieval_strategy = StrategyFactory.create(strategy)

        # Execute retrieval
        context = await retrieval_strategy.retrieve(
            query,
            state,
            self.cache,
            self.vector_store,
            max_items
        )

        # Update stats
        duration = (datetime.now() - start_time).total_seconds() * 1000
        self._stats.total_retrievals += 1
        self._stats.avg_retrieval_time_ms = (
            (self._stats.avg_retrieval_time_ms * (self._stats.total_retrievals - 1) + duration)
            / self._stats.total_retrievals
        )

        logger.info(f"Context retrieved in {duration:.2f}ms: {context.get_summary()}")

        return context

    # ========================================================================
    # TIER 1: STATE OPERATIONS
    # ========================================================================

    def get_from_state(
        self,
        state: AgentState,
        key: str,
        default: Any = None
    ) -> Any:
        """Get value from state (Tier 1).

        Args:
            state: Agent state
            key: State key
            default: Default value

        Returns:
            Value from state
        """
        return state.get(key, default)

    def update_state(
        self,
        state: AgentState,
        updates: Dict[str, Any]
    ) -> AgentState:
        """Update state (Tier 1).

        Args:
            state: Agent state
            updates: Updates to apply

        Returns:
            Updated state
        """
        state.update(updates)
        return state

    # ========================================================================
    # TIER 2: CACHE OPERATIONS
    # ========================================================================

    async def cache_interaction(
        self,
        request_id: str,
        interaction: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Cache interaction (Tier 2).

        Args:
            request_id: Request identifier
            interaction: Interaction data
            ttl: Time to live

        Returns:
            Success status
        """
        # Get existing interactions
        cache_key = CacheKey(CacheNamespace.USER, request_id, "recent")
        existing = await self.cache.get(cache_key) or []

        # Add new interaction
        existing.append(interaction)

        # Keep only recent N interactions
        if len(existing) > 50:
            existing = existing[-50:]

        # Store back
        return await self.cache.set(cache_key, existing, ttl)

    async def get_cached_interactions(
        self,
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Get cached interactions (Tier 2).

        Args:
            request_id: Request identifier

        Returns:
            List of interactions
        """
        cache_key = CacheKey(CacheNamespace.USER, request_id, "recent")
        return await self.cache.get(cache_key) or []

    async def cache_user_preferences(
        self,
        customer_id: str,
        preferences: Dict[str, Any],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Cache user preferences (Tier 2).

        Args:
            customer_id: Customer identifier
            preferences: User preferences
            ttl: Time to live

        Returns:
            Success status
        """
        cache_key = CacheKey(CacheNamespace.USER, customer_id, "preferences")
        return await self.cache.set(cache_key, preferences, ttl)

    async def get_user_preferences(
        self,
        customer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get user preferences (Tier 2).

        Args:
            customer_id: Customer identifier

        Returns:
            User preferences or None
        """
        cache_key = CacheKey(CacheNamespace.USER, customer_id, "preferences")
        return await self.cache.get(cache_key)

    # ========================================================================
    # TIER 3: VECTOR DB OPERATIONS
    # ========================================================================

    async def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store conversation in vector DB (Tier 3).

        Args:
            conversation_id: Conversation identifier
            messages: Conversation messages
            metadata: Additional metadata

        Returns:
            Success status
        """
        # Combine messages into text
        content = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])

        # Create document
        doc = Document(
            id=conversation_id,
            content=content,
            metadata={
                **(metadata or {}),
                "message_count": len(messages),
                "stored_at": datetime.now().isoformat()
            }
        )

        # Store in vector DB
        return await self.vector_store.add_documents([doc], namespace="conversations")

    async def search_conversations(
        self,
        query: str,
        k: int = 5
    ) -> List[SearchResult]:
        """Search similar conversations (Tier 3).

        Args:
            query: Search query
            k: Number of results

        Returns:
            Search results
        """
        return await self.vector_store.similarity_search(
            query,
            k=k,
            namespace="conversations"
        )

    async def search_knowledge_base(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search knowledge base (Tier 3).

        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters

        Returns:
            Search results
        """
        return await self.vector_store.similarity_search(
            query,
            k=k,
            namespace="knowledge_base",
            filters=filters
        )

    async def add_knowledge_article(
        self,
        article_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add article to knowledge base (Tier 3).

        Args:
            article_id: Article identifier
            content: Article content
            metadata: Additional metadata

        Returns:
            Success status
        """
        doc = Document(
            id=article_id,
            content=content,
            metadata={
                **(metadata or {}),
                "added_at": datetime.now().isoformat()
            }
        )

        return await self.vector_store.add_documents([doc], namespace="knowledge_base")

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    async def create_session(
        self,
        session_id: str,
        initial_data: Dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """Create new session (Tier 2).

        Args:
            session_id: Session identifier
            initial_data: Initial session data
            ttl: Session TTL (30 minutes)

        Returns:
            Success status
        """
        return await self.session_cache.store_session(session_id, initial_data, ttl)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data (Tier 2).

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        return await self.session_cache.get_session(session_id)

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data (Tier 2).

        Args:
            session_id: Session identifier
            updates: Updates to apply

        Returns:
            Success status
        """
        return await self.session_cache.update_session(session_id, updates)

    async def end_session(self, session_id: str) -> bool:
        """End session (Tier 2).

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        return await self.session_cache.delete_session(session_id)

    # ========================================================================
    # MEMORY OPTIMIZATION
    # ========================================================================

    async def optimize_memory(
        self,
        state: AgentState,
        max_messages: int = 20
    ) -> AgentState:
        """Optimize state memory.

        Moves old data from state to cache/vector DB.

        Args:
            state: Agent state
            max_messages: Max messages to keep in state

        Returns:
            Optimized state
        """
        messages = state.get("messages", [])

        if len(messages) > max_messages:
            # Get messages to archive
            to_archive = messages[:-max_messages]

            # Store in cache
            request_id = state.get("request_id")
            if request_id:
                await self.cache_interaction(
                    request_id,
                    {
                        "messages": [
                            {"role": msg.__class__.__name__, "content": msg.content}
                            for msg in to_archive
                        ],
                        "archived_at": datetime.now().isoformat()
                    }
                )

            # Keep only recent messages in state
            state["messages"] = messages[-max_messages:]

            logger.info(f"Optimized state: archived {len(to_archive)} messages")

        return state

    async def persist_state_to_long_term(
        self,
        state: AgentState
    ) -> bool:
        """Persist state to long-term memory (Tier 3).

        Args:
            state: Agent state

        Returns:
            Success status
        """
        request_id = state.get("request_id")
        if not request_id:
            return False

        # Extract conversation
        messages = StateUtils.get_message_history(state)

        # Store in vector DB
        return await self.store_conversation(
            request_id,
            messages,
            metadata={
                "customer_id": state.get("customer_id"),
                "category": state.get("category"),
                "priority": state.get("priority"),
                "status": state.get("status"),
                "persisted_at": datetime.now().isoformat()
            }
        )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_stats(self) -> MemoryStats:
        """Get memory system statistics.

        Returns:
            Memory statistics
        """
        # Cache stats
        cache_stats = await self.cache.get_stats()
        self._stats.cache_keys = cache_stats.total_keys
        self._stats.cache_hit_rate = cache_stats.hit_rate

        # Vector DB stats (if available)
        # Note: Implementation depends on vector store

        return self._stats

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all memory tiers.

        Returns:
            Health status for each tier
        """
        health = {
            "state": True,  # Always available
            "cache": False,
            "vector": False
        }

        # Check cache
        try:
            health["cache"] = await self.cache.ping()
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")

        # Check vector DB
        try:
            health["vector"] = await self.vector_store.ping()
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")

        return health


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from .vector_store import VectorStoreFactory
    from .redis_cache import RedisCacheFactory
    from ..agents.state import StateBuilder
    from ..domain.models import Priority, RequestCategory

    async def main():
        print("=== Memory Manager Demo ===\n")

        # Create components
        vector_store = VectorStoreFactory.create(
            backend="weaviate",
            url="http://localhost:8080"
        )

        cache = await RedisCacheFactory.create(
            host="localhost",
            port=6379
        )

        # Create memory manager
        memory = MemoryManager(vector_store, cache)

        # Check health
        print("Checking memory system health...")
        health = await memory.health_check()
        print(f"Health: {health}\n")

        # Create initial state
        state = (
            StateBuilder()
            .with_request_id("req-123")
            .with_customer_id("cust-456")
            .with_initial_message("How do I reset my password?")
            .with_priority(Priority.MEDIUM)
            .with_category(RequestCategory.ACCOUNT)
            .build()
        )

        # Test context retrieval
        print("Retrieving context (Hybrid strategy)...")
        context = await memory.get_context(
            "password reset",
            state,
            strategy=RetrievalStrategy.HYBRID,
            max_items=5
        )
        print(f"Context summary: {context.get_summary()}\n")

        # Test caching interaction
        print("Caching interaction...")
        await memory.cache_interaction(
            "req-123",
            {
                "query": "How do I reset my password?",
                "response": "You can reset your password by...",
                "timestamp": datetime.now().isoformat()
            }
        )
        print("Interaction cached\n")

        # Test knowledge base
        print("Adding knowledge article...")
        await memory.add_knowledge_article(
            "kb-001",
            "Password Reset Guide: To reset your password, visit...",
            metadata={"category": "account", "tags": ["password", "security"]}
        )
        print("Article added\n")

        # Search knowledge base
        print("Searching knowledge base...")
        kb_results = await memory.search_knowledge_base("password reset", k=3)
        print(f"Found {len(kb_results)} articles\n")

        # Get stats
        print("Memory statistics:")
        stats = await memory.get_stats()
        print(f"  Cache keys: {stats.cache_keys}")
        print(f"  Cache hit rate: {stats.cache_hit_rate:.2%}")
        print(f"  Avg retrieval time: {stats.avg_retrieval_time_ms:.2f}ms")
        print(f"  Total retrievals: {stats.total_retrievals}\n")

        # Cleanup
        await cache.client.close()
        print("Demo complete!")

    # Run
    asyncio.run(main())
