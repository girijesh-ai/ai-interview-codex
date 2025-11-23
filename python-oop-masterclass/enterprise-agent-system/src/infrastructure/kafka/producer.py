"""
Kafka Event Producer

Demonstrates:
- Producer pattern for event publishing
- Async operations
- Error handling and retries
- Batch processing
- Monitoring and metrics
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from .events import BaseEvent, EventType
from .topics import TopicConfig, TopicName

logger = logging.getLogger(__name__)


# ============================================================================
# VALUE OBJECTS
# ============================================================================

@dataclass
class ProducerConfig:
    """Producer configuration.

    Demonstrates:
    - Value object pattern
    - Configuration as code
    """
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "agent-producer"
    compression_type: Optional[str] = "gzip"  # None, gzip, snappy, lz4, zstd
    acks: str = "all"  # 0, 1, all
    retries: int = 3
    retry_backoff_ms: int = 100
    request_timeout_ms: int = 30000
    max_batch_size: int = 16384
    linger_ms: int = 10  # Wait time to batch messages
    enable_idempotence: bool = True
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL


@dataclass
class PublishResult:
    """Result of publishing an event.

    Demonstrates:
    - Value object for results
    """
    success: bool
    event_id: str
    topic: str
    partition: Optional[int] = None
    offset: Optional[int] = None
    timestamp: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ProducerMetrics:
    """Producer metrics."""
    events_sent: int = 0
    events_failed: int = 0
    bytes_sent: int = 0
    batches_sent: int = 0
    avg_batch_size: float = 0.0
    send_errors: int = 0


# ============================================================================
# PRODUCER INTERFACE
# ============================================================================

class EventProducer(ABC):
    """Abstract event producer.

    Demonstrates:
    - Repository/Producer pattern
    - Interface segregation principle (ISP)
    """

    @abstractmethod
    async def publish(
        self,
        event: BaseEvent,
        topic: Optional[TopicName] = None
    ) -> PublishResult:
        """Publish single event."""
        pass

    @abstractmethod
    async def publish_batch(
        self,
        events: List[BaseEvent],
        topic: Optional[TopicName] = None
    ) -> List[PublishResult]:
        """Publish batch of events."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start producer."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop producer."""
        pass

    @abstractmethod
    def get_metrics(self) -> ProducerMetrics:
        """Get producer metrics."""
        pass


# ============================================================================
# KAFKA PRODUCER IMPLEMENTATION
# ============================================================================

class KafkaEventProducer(EventProducer):
    """Kafka event producer implementation.

    Demonstrates:
    - Producer pattern
    - Async operations
    - Error handling
    - Metrics tracking
    """

    def __init__(
        self,
        config: ProducerConfig,
        topic_config: TopicConfig
    ):
        """Initialize Kafka producer.

        Args:
            config: Producer configuration
            topic_config: Topic configuration
        """
        self.config = config
        self.topic_config = topic_config
        self.producer: Optional[AIOKafkaProducer] = None
        self._metrics = ProducerMetrics()
        self._running = False

    async def start(self) -> None:
        """Start Kafka producer."""
        if self._running:
            logger.warning("Producer already running")
            return

        logger.info("Starting Kafka producer...")

        try:
            # Create producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                compression_type=self.config.compression_type,
                acks=self.config.acks,
                retries=self.config.retries,
                retry_backoff_ms=self.config.retry_backoff_ms,
                request_timeout_ms=self.config.request_timeout_ms,
                max_batch_size=self.config.max_batch_size,
                linger_ms=self.config.linger_ms,
                enable_idempotence=self.config.enable_idempotence,
                security_protocol=self.config.security_protocol,
                # Serialization
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v
            )

            # Start producer
            await self.producer.start()
            self._running = True

            logger.info(f"Kafka producer started: {self.config.bootstrap_servers}")

        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        """Stop Kafka producer."""
        if not self._running:
            return

        logger.info("Stopping Kafka producer...")

        try:
            if self.producer:
                await self.producer.stop()
                self._running = False
                logger.info("Kafka producer stopped")
        except Exception as e:
            logger.error(f"Error stopping producer: {e}")

    async def publish(
        self,
        event: BaseEvent,
        topic: Optional[TopicName] = None
    ) -> PublishResult:
        """Publish single event.

        Args:
            event: Event to publish
            topic: Optional topic override

        Returns:
            Publish result
        """
        if not self._running:
            return PublishResult(
                success=False,
                event_id=event.event_id,
                topic="",
                error="Producer not running"
            )

        # Determine topic
        topic_name = topic or self._get_topic_for_event(event.event_type)

        try:
            # Serialize event
            key = event.request_id if hasattr(event, 'request_id') else event.event_id
            value = event.to_json()

            # Send to Kafka
            record_metadata = await self.producer.send_and_wait(
                topic_name.value,
                value=value,
                key=key
            )

            # Update metrics
            self._metrics.events_sent += 1
            self._metrics.bytes_sent += len(value)

            logger.debug(
                f"Event published: {event.event_type.value} -> "
                f"{topic_name.value}:{record_metadata.partition}:{record_metadata.offset}"
            )

            return PublishResult(
                success=True,
                event_id=event.event_id,
                topic=topic_name.value,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
                timestamp=record_metadata.timestamp
            )

        except KafkaError as e:
            self._metrics.events_failed += 1
            self._metrics.send_errors += 1
            logger.error(f"Kafka error publishing event: {e}")

            return PublishResult(
                success=False,
                event_id=event.event_id,
                topic=topic_name.value,
                error=str(e)
            )

        except Exception as e:
            self._metrics.events_failed += 1
            logger.error(f"Error publishing event: {e}")

            return PublishResult(
                success=False,
                event_id=event.event_id,
                topic=topic_name.value,
                error=str(e)
            )

    async def publish_batch(
        self,
        events: List[BaseEvent],
        topic: Optional[TopicName] = None
    ) -> List[PublishResult]:
        """Publish batch of events.

        Args:
            events: Events to publish
            topic: Optional topic override

        Returns:
            List of publish results
        """
        if not self._running:
            return [
                PublishResult(
                    success=False,
                    event_id=e.event_id,
                    topic="",
                    error="Producer not running"
                )
                for e in events
            ]

        results = []
        batch_start = datetime.now()

        try:
            # Send all events
            for event in events:
                result = await self.publish(event, topic)
                results.append(result)

            # Update batch metrics
            self._metrics.batches_sent += 1
            batch_size = len(events)

            # Calculate average batch size
            total_batches = self._metrics.batches_sent
            self._metrics.avg_batch_size = (
                (self._metrics.avg_batch_size * (total_batches - 1) + batch_size)
                / total_batches
            )

            duration = (datetime.now() - batch_start).total_seconds()
            logger.info(
                f"Batch published: {batch_size} events in {duration:.3f}s "
                f"({batch_size/duration:.1f} events/s)"
            )

        except Exception as e:
            logger.error(f"Error in batch publish: {e}")

        return results

    def _get_topic_for_event(self, event_type: EventType) -> TopicName:
        """Get topic for event type.

        Args:
            event_type: Event type

        Returns:
            Topic name
        """
        # Map event types to topics
        if event_type in [
            EventType.REQUEST_RECEIVED,
            EventType.REQUEST_STARTED,
            EventType.REQUEST_COMPLETED,
            EventType.REQUEST_FAILED
        ]:
            return TopicName.CUSTOMER_REQUESTS

        elif event_type in [
            EventType.AGENT_STARTED,
            EventType.AGENT_COMPLETED,
            EventType.AGENT_FAILED,
            EventType.AGENT_DECISION,
            EventType.REQUEST_TRIAGED,
            EventType.RESEARCH_STARTED,
            EventType.DOCUMENTS_RETRIEVED,
            EventType.SOLUTION_GENERATED,
            EventType.QUALITY_CHECK_PASSED,
            EventType.QUALITY_CHECK_FAILED,
            EventType.REQUEST_ESCALATED
        ]:
            return TopicName.AGENT_ACTIONS

        elif event_type in [
            EventType.APPROVAL_REQUESTED,
            EventType.APPROVAL_GRANTED,
            EventType.APPROVAL_REJECTED,
            EventType.HUMAN_DECISION_NEEDED,
            EventType.HUMAN_DECISION_RECEIVED
        ]:
            return TopicName.HUMAN_APPROVALS

        elif event_type in [
            EventType.PERFORMANCE_METRIC,
            EventType.HEALTH_CHECK
        ]:
            return TopicName.SYSTEM_ANALYTICS

        elif event_type == EventType.SYSTEM_ERROR:
            return TopicName.SYSTEM_ANALYTICS

        else:
            return TopicName.AGENT_ACTIONS

    def get_metrics(self) -> ProducerMetrics:
        """Get producer metrics.

        Returns:
            Producer metrics
        """
        return self._metrics


# ============================================================================
# PRODUCER FACTORY
# ============================================================================

class ProducerFactory:
    """Factory for creating producers.

    Demonstrates:
    - Factory pattern
    - Configuration management
    """

    @staticmethod
    async def create(
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "agent-producer",
        **kwargs
    ) -> KafkaEventProducer:
        """Create and start Kafka producer.

        Args:
            bootstrap_servers: Kafka servers
            client_id: Client identifier
            **kwargs: Additional config

        Returns:
            Started Kafka producer
        """
        config = ProducerConfig(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            **kwargs
        )

        topic_config = TopicConfig()

        producer = KafkaEventProducer(config, topic_config)
        await producer.start()

        return producer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from .events import RequestReceivedEvent, AgentDecisionEvent

    async def main():
        print("=== Kafka Producer Demo ===\n")

        # Create producer
        producer = await ProducerFactory.create(
            bootstrap_servers="localhost:9092",
            client_id="demo-producer"
        )

        try:
            # Publish single event
            print("Publishing single event...")
            event1 = RequestReceivedEvent(
                request_id="req-123",
                customer_id="cust-456",
                channel="chat",
                initial_message="I need help"
            )

            result1 = await producer.publish(event1)
            print(f"Result: {result1}")

            # Publish batch
            print("\nPublishing batch...")
            events = [
                AgentDecisionEvent(
                    request_id="req-123",
                    agent_type="triage",
                    decision_type="classify",
                    confidence=0.85,
                    reasoning="Category detected"
                ),
                AgentDecisionEvent(
                    request_id="req-123",
                    agent_type="solution",
                    decision_type="generate",
                    confidence=0.92,
                    reasoning="Template match found"
                )
            ]

            results = await producer.publish_batch(events)
            print(f"Batch results: {len([r for r in results if r.success])} succeeded")

            # Get metrics
            print("\nProducer metrics:")
            metrics = producer.get_metrics()
            print(f"  Events sent: {metrics.events_sent}")
            print(f"  Events failed: {metrics.events_failed}")
            print(f"  Bytes sent: {metrics.bytes_sent}")
            print(f"  Batches sent: {metrics.batches_sent}")
            print(f"  Avg batch size: {metrics.avg_batch_size:.1f}")

        finally:
            # Stop producer
            await producer.stop()
            print("\nProducer stopped")

    # Run
    asyncio.run(main())
