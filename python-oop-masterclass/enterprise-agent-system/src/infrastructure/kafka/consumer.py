"""
Kafka Event Consumer

Demonstrates:
- Consumer pattern for event processing
- Observer pattern for event handlers
- Strategy pattern for processing strategies
- Error handling and retry logic
- Offset management
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
import logging
import asyncio
from datetime import datetime

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import ConsumerRecord

from .events import BaseEvent, EventType, EventFactory
from .topics import TopicName

logger = logging.getLogger(__name__)


# ============================================================================
# VALUE OBJECTS
# ============================================================================

@dataclass
class ConsumerConfig:
    """Consumer configuration.

    Demonstrates:
    - Value object pattern
    - Configuration as code
    """
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "agent-consumer-group"
    client_id: str = "agent-consumer"
    auto_offset_reset: str = "earliest"  # earliest, latest, none
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 10000
    heartbeat_interval_ms: int = 3000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    security_protocol: str = "PLAINTEXT"


@dataclass
class ConsumerMetrics:
    """Consumer metrics."""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    processing_errors: int = 0
    last_offset: Dict[str, int] = field(default_factory=dict)
    avg_processing_time_ms: float = 0.0


# ============================================================================
# EVENT HANDLER INTERFACE
# ============================================================================

class EventHandler(ABC):
    """Abstract event handler.

    Demonstrates:
    - Observer pattern
    - Strategy pattern for different handlers
    """

    @abstractmethod
    async def handle(self, event: BaseEvent) -> bool:
        """Handle event.

        Args:
            event: Event to handle

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle event type.

        Args:
            event_type: Event type

        Returns:
            True if can handle
        """
        pass


# ============================================================================
# SPECIFIC EVENT HANDLERS
# ============================================================================

class RequestReceivedHandler(EventHandler):
    """Handler for request received events."""

    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle."""
        return event_type == EventType.REQUEST_RECEIVED

    async def handle(self, event: BaseEvent) -> bool:
        """Handle request received event."""
        logger.info(f"Handling REQUEST_RECEIVED: {event.event_id}")
        # Implementation: Start workflow, initialize state, etc.
        return True


class ApprovalRequestedHandler(EventHandler):
    """Handler for approval requested events."""

    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle."""
        return event_type == EventType.APPROVAL_REQUESTED

    async def handle(self, event: BaseEvent) -> bool:
        """Handle approval requested event."""
        logger.info(f"Handling APPROVAL_REQUESTED: {event.event_id}")
        # Implementation: Send notification, update dashboard, etc.
        return True


class SystemErrorHandler(EventHandler):
    """Handler for system error events."""

    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle."""
        return event_type == EventType.SYSTEM_ERROR

    async def handle(self, event: BaseEvent) -> bool:
        """Handle system error event."""
        logger.error(f"Handling SYSTEM_ERROR: {event.event_id}")
        # Implementation: Alert, log, create ticket, etc.
        return True


class PerformanceMetricHandler(EventHandler):
    """Handler for performance metric events."""

    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle."""
        return event_type == EventType.PERFORMANCE_METRIC

    async def handle(self, event: BaseEvent) -> bool:
        """Handle performance metric event."""
        logger.debug(f"Handling PERFORMANCE_METRIC: {event.event_id}")
        # Implementation: Store metrics, update dashboard, etc.
        return True


# ============================================================================
# HANDLER REGISTRY
# ============================================================================

class EventHandlerRegistry:
    """Registry for event handlers.

    Demonstrates:
    - Registry pattern
    - Mediator pattern
    """

    def __init__(self):
        """Initialize registry."""
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.wildcard_handlers: List[EventHandler] = []

    def register(
        self,
        handler: EventHandler,
        event_types: Optional[List[EventType]] = None
    ) -> None:
        """Register event handler.

        Args:
            handler: Event handler
            event_types: Specific event types (None for all)
        """
        if event_types is None:
            # Register for all events
            self.wildcard_handlers.append(handler)
            logger.info(f"Registered wildcard handler: {handler.__class__.__name__}")
        else:
            # Register for specific event types
            for event_type in event_types:
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                self.handlers[event_type].append(handler)
                logger.info(
                    f"Registered handler {handler.__class__.__name__} "
                    f"for {event_type.value}"
                )

    def unregister(self, handler: EventHandler) -> None:
        """Unregister event handler.

        Args:
            handler: Event handler to remove
        """
        # Remove from wildcard handlers
        if handler in self.wildcard_handlers:
            self.wildcard_handlers.remove(handler)

        # Remove from specific handlers
        for event_type in self.handlers:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)

    def get_handlers(self, event_type: EventType) -> List[EventHandler]:
        """Get handlers for event type.

        Args:
            event_type: Event type

        Returns:
            List of handlers
        """
        handlers = []

        # Add specific handlers
        if event_type in self.handlers:
            handlers.extend(self.handlers[event_type])

        # Add wildcard handlers
        handlers.extend(self.wildcard_handlers)

        return handlers


# ============================================================================
# CONSUMER INTERFACE
# ============================================================================

class EventConsumer(ABC):
    """Abstract event consumer.

    Demonstrates:
    - Consumer pattern
    - Interface segregation principle (ISP)
    """

    @abstractmethod
    async def start(self) -> None:
        """Start consumer."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop consumer."""
        pass

    @abstractmethod
    async def consume(self) -> None:
        """Consume events."""
        pass

    @abstractmethod
    def register_handler(
        self,
        handler: EventHandler,
        event_types: Optional[List[EventType]] = None
    ) -> None:
        """Register event handler."""
        pass

    @abstractmethod
    def get_metrics(self) -> ConsumerMetrics:
        """Get consumer metrics."""
        pass


# ============================================================================
# KAFKA CONSUMER IMPLEMENTATION
# ============================================================================

class KafkaEventConsumer(EventConsumer):
    """Kafka event consumer implementation.

    Demonstrates:
    - Consumer pattern
    - Observer pattern (handlers)
    - Error handling
    - Metrics tracking
    """

    def __init__(
        self,
        config: ConsumerConfig,
        topics: List[TopicName]
    ):
        """Initialize Kafka consumer.

        Args:
            config: Consumer configuration
            topics: Topics to subscribe to
        """
        self.config = config
        self.topics = [t.value for t in topics]
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.registry = EventHandlerRegistry()
        self._metrics = ConsumerMetrics()
        self._running = False
        self._consume_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start Kafka consumer."""
        if self._running:
            logger.warning("Consumer already running")
            return

        logger.info("Starting Kafka consumer...")

        try:
            # Create consumer
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                client_id=self.config.client_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_records=self.config.max_poll_records,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                fetch_min_bytes=self.config.fetch_min_bytes,
                fetch_max_wait_ms=self.config.fetch_max_wait_ms,
                security_protocol=self.config.security_protocol,
                # Deserialization
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                value_deserializer=lambda v: v.decode('utf-8') if v else None
            )

            # Start consumer
            await self.consumer.start()
            self._running = True

            logger.info(
                f"Kafka consumer started: {self.config.bootstrap_servers} "
                f"topics={self.topics}"
            )

            # Start consuming in background
            self._consume_task = asyncio.create_task(self.consume())

        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self) -> None:
        """Stop Kafka consumer."""
        if not self._running:
            return

        logger.info("Stopping Kafka consumer...")

        try:
            self._running = False

            # Cancel consume task
            if self._consume_task:
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass

            # Stop consumer
            if self.consumer:
                await self.consumer.stop()

            logger.info("Kafka consumer stopped")

        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")

    async def consume(self) -> None:
        """Consume events from Kafka."""
        logger.info("Starting event consumption...")

        try:
            async for msg in self.consumer:
                if not self._running:
                    break

                await self._process_message(msg)

        except asyncio.CancelledError:
            logger.info("Consumption cancelled")
        except Exception as e:
            logger.error(f"Error in consume loop: {e}")

    async def _process_message(self, msg: ConsumerRecord) -> None:
        """Process single message.

        Args:
            msg: Kafka message
        """
        start_time = datetime.now()
        self._metrics.messages_consumed += 1

        try:
            # Parse event
            event_data = eval(msg.value)  # Safe because we control the format
            event = EventFactory.create_from_dict(event_data)

            logger.debug(
                f"Processing event: {event.event_type.value} "
                f"[{msg.topic}:{msg.partition}:{msg.offset}]"
            )

            # Get handlers
            handlers = self.registry.get_handlers(event.event_type)

            if not handlers:
                logger.warning(f"No handlers for event type: {event.event_type.value}")
                return

            # Execute handlers
            results = await asyncio.gather(
                *[handler.handle(event) for handler in handlers],
                return_exceptions=True
            )

            # Check results
            success = all(
                r is True or (not isinstance(r, Exception))
                for r in results
            )

            if success:
                self._metrics.messages_processed += 1
            else:
                self._metrics.messages_failed += 1
                logger.error(f"Handler failed for event: {event.event_id}")

            # Update offset tracking
            self._metrics.last_offset[msg.topic] = msg.offset

            # Update processing time
            duration = (datetime.now() - start_time).total_seconds() * 1000
            total = self._metrics.messages_processed
            self._metrics.avg_processing_time_ms = (
                (self._metrics.avg_processing_time_ms * (total - 1) + duration)
                / total
            )

        except Exception as e:
            self._metrics.processing_errors += 1
            logger.error(f"Error processing message: {e}")

    def register_handler(
        self,
        handler: EventHandler,
        event_types: Optional[List[EventType]] = None
    ) -> None:
        """Register event handler.

        Args:
            handler: Event handler
            event_types: Event types to handle (None for all)
        """
        self.registry.register(handler, event_types)

    def unregister_handler(self, handler: EventHandler) -> None:
        """Unregister event handler.

        Args:
            handler: Event handler to remove
        """
        self.registry.unregister(handler)

    def get_metrics(self) -> ConsumerMetrics:
        """Get consumer metrics.

        Returns:
            Consumer metrics
        """
        return self._metrics


# ============================================================================
# CONSUMER FACTORY
# ============================================================================

class ConsumerFactory:
    """Factory for creating consumers.

    Demonstrates:
    - Factory pattern
    - Configuration management
    """

    @staticmethod
    async def create(
        topics: List[TopicName],
        group_id: str = "agent-consumer-group",
        bootstrap_servers: str = "localhost:9092",
        **kwargs
    ) -> KafkaEventConsumer:
        """Create and start Kafka consumer.

        Args:
            topics: Topics to subscribe to
            group_id: Consumer group ID
            bootstrap_servers: Kafka servers
            **kwargs: Additional config

        Returns:
            Started Kafka consumer
        """
        config = ConsumerConfig(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            **kwargs
        )

        consumer = KafkaEventConsumer(config, topics)
        await consumer.start()

        return consumer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("=== Kafka Consumer Demo ===\n")

        # Create consumer
        consumer = await ConsumerFactory.create(
            topics=[
                TopicName.CUSTOMER_REQUESTS,
                TopicName.AGENT_ACTIONS,
                TopicName.HUMAN_APPROVALS
            ],
            group_id="demo-consumer-group",
            bootstrap_servers="localhost:9092"
        )

        try:
            # Register handlers
            consumer.register_handler(
                RequestReceivedHandler(),
                [EventType.REQUEST_RECEIVED]
            )
            consumer.register_handler(
                ApprovalRequestedHandler(),
                [EventType.APPROVAL_REQUESTED]
            )
            consumer.register_handler(
                SystemErrorHandler(),
                [EventType.SYSTEM_ERROR]
            )

            print("Consumer started, waiting for events...")
            print("Press Ctrl+C to stop\n")

            # Run for a while
            await asyncio.sleep(60)

        except KeyboardInterrupt:
            print("\nStopping consumer...")

        finally:
            # Get metrics
            print("\nConsumer metrics:")
            metrics = consumer.get_metrics()
            print(f"  Messages consumed: {metrics.messages_consumed}")
            print(f"  Messages processed: {metrics.messages_processed}")
            print(f"  Messages failed: {metrics.messages_failed}")
            print(f"  Avg processing time: {metrics.avg_processing_time_ms:.2f}ms")

            # Stop consumer
            await consumer.stop()
            print("Consumer stopped")

    # Run
    asyncio.run(main())
