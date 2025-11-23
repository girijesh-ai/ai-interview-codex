"""
Celery Configuration

Demonstrates:
- Configuration as code
- Environment-based configuration
- Type safety with dataclasses
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 5
    HIGH = 7
    CRITICAL = 9


class SerializerType(str, Enum):
    """Serializer types."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CeleryConfig:
    """Celery configuration.

    Demonstrates:
    - Value object pattern
    - Configuration as data
    """

    # Broker settings
    broker_url: str = "redis://localhost:6379/0"
    broker_connection_retry_on_startup: bool = True
    broker_connection_retry: bool = True
    broker_connection_max_retries: int = 10

    # Result backend
    result_backend: str = "redis://localhost:6379/1"
    result_backend_transport_options: Dict[str, Any] = None
    result_expires: int = 3600  # 1 hour
    result_persistent: bool = True

    # Serialization
    task_serializer: SerializerType = SerializerType.JSON
    result_serializer: SerializerType = SerializerType.JSON
    accept_content: List[str] = None

    # Task execution
    task_always_eager: bool = False  # Execute tasks synchronously (for testing)
    task_eager_propagates: bool = True
    task_ignore_result: bool = False
    task_store_errors_even_if_ignored: bool = True

    # Task routing
    task_routes: Dict[str, Dict[str, str]] = None
    task_default_queue: str = "default"
    task_default_exchange: str = "tasks"
    task_default_routing_key: str = "task.default"

    # Task time limits
    task_soft_time_limit: int = 300  # 5 minutes
    task_time_limit: int = 600  # 10 minutes
    task_acks_late: bool = True  # Acknowledge after task completion
    task_reject_on_worker_lost: bool = True

    # Worker settings
    worker_prefetch_multiplier: int = 4
    worker_max_tasks_per_child: int = 1000  # Restart worker after N tasks
    worker_disable_rate_limits: bool = False
    worker_send_task_events: bool = True
    worker_pool: str = "prefork"  # prefork, solo, threads, gevent

    # Monitoring
    task_send_sent_event: bool = True
    task_track_started: bool = True
    worker_enable_remote_control: bool = True

    # Beat scheduler (periodic tasks)
    beat_schedule: Dict[str, Dict[str, Any]] = None
    beat_scheduler: str = "celery.beat:PersistentScheduler"
    beat_schedule_filename: str = "celerybeat-schedule"

    # Timezone
    timezone: str = "UTC"
    enable_utc: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.accept_content is None:
            self.accept_content = ["json", "msgpack", "yaml"]

        if self.result_backend_transport_options is None:
            self.result_backend_transport_options = {
                "master_name": "mymaster"
            }

        if self.task_routes is None:
            self.task_routes = {
                "tasks.embedding.*": {"queue": "embedding"},
                "tasks.analytics.*": {"queue": "analytics"},
                "tasks.notifications.*": {"queue": "notifications"},
                "tasks.sync.*": {"queue": "sync"},
            }

        if self.beat_schedule is None:
            self.beat_schedule = {
                # Example periodic task
                "cleanup-old-data": {
                    "task": "tasks.maintenance.cleanup_old_data",
                    "schedule": 3600.0,  # Every hour
                    "options": {"queue": "maintenance"}
                },
                "generate-analytics": {
                    "task": "tasks.analytics.generate_daily_analytics",
                    "schedule": 86400.0,  # Every day
                    "options": {"queue": "analytics"}
                }
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Celery app configuration.

        Returns:
            Configuration dictionary
        """
        return {
            # Broker
            "broker_url": self.broker_url,
            "broker_connection_retry_on_startup": self.broker_connection_retry_on_startup,
            "broker_connection_retry": self.broker_connection_retry,
            "broker_connection_max_retries": self.broker_connection_max_retries,

            # Result backend
            "result_backend": self.result_backend,
            "result_backend_transport_options": self.result_backend_transport_options,
            "result_expires": self.result_expires,
            "result_persistent": self.result_persistent,

            # Serialization
            "task_serializer": self.task_serializer.value,
            "result_serializer": self.result_serializer.value,
            "accept_content": self.accept_content,

            # Task execution
            "task_always_eager": self.task_always_eager,
            "task_eager_propagates": self.task_eager_propagates,
            "task_ignore_result": self.task_ignore_result,
            "task_store_errors_even_if_ignored": self.task_store_errors_even_if_ignored,

            # Task routing
            "task_routes": self.task_routes,
            "task_default_queue": self.task_default_queue,
            "task_default_exchange": self.task_default_exchange,
            "task_default_routing_key": self.task_default_routing_key,

            # Task time limits
            "task_soft_time_limit": self.task_soft_time_limit,
            "task_time_limit": self.task_time_limit,
            "task_acks_late": self.task_acks_late,
            "task_reject_on_worker_lost": self.task_reject_on_worker_lost,

            # Worker
            "worker_prefetch_multiplier": self.worker_prefetch_multiplier,
            "worker_max_tasks_per_child": self.worker_max_tasks_per_child,
            "worker_disable_rate_limits": self.worker_disable_rate_limits,
            "worker_send_task_events": self.worker_send_task_events,
            "worker_pool": self.worker_pool,

            # Monitoring
            "task_send_sent_event": self.task_send_sent_event,
            "task_track_started": self.task_track_started,
            "worker_enable_remote_control": self.worker_enable_remote_control,

            # Beat
            "beat_schedule": self.beat_schedule,
            "beat_scheduler": self.beat_scheduler,
            "beat_schedule_filename": self.beat_schedule_filename,

            # Timezone
            "timezone": self.timezone,
            "enable_utc": self.enable_utc,
        }


# ============================================================================
# QUEUE DEFINITIONS
# ============================================================================

@dataclass
class QueueConfig:
    """Queue configuration.

    Demonstrates:
    - Value object pattern
    """
    name: str
    routing_key: str
    priority: int = 5
    max_length: Optional[int] = None
    message_ttl: Optional[int] = None  # Seconds

    def to_kombu_dict(self) -> Dict[str, Any]:
        """Convert to Kombu queue configuration."""
        from kombu import Queue, Exchange

        queue_args = {}
        if self.max_length:
            queue_args["x-max-length"] = self.max_length
        if self.message_ttl:
            queue_args["x-message-ttl"] = self.message_ttl * 1000  # ms

        return {
            "name": self.name,
            "exchange": Exchange("tasks", type="topic"),
            "routing_key": self.routing_key,
            "queue_arguments": queue_args
        }


class QueueManager:
    """Manages queue configurations.

    Demonstrates:
    - Configuration management
    - Factory pattern
    """

    def __init__(self):
        """Initialize queue manager."""
        self.queues = {
            "default": QueueConfig(
                name="default",
                routing_key="task.default",
                priority=TaskPriority.NORMAL,
                max_length=10000
            ),
            "embedding": QueueConfig(
                name="embedding",
                routing_key="task.embedding",
                priority=TaskPriority.NORMAL,
                max_length=5000,
                message_ttl=3600  # 1 hour
            ),
            "analytics": QueueConfig(
                name="analytics",
                routing_key="task.analytics",
                priority=TaskPriority.LOW,
                max_length=1000,
                message_ttl=7200  # 2 hours
            ),
            "notifications": QueueConfig(
                name="notifications",
                routing_key="task.notifications",
                priority=TaskPriority.HIGH,
                max_length=20000,
                message_ttl=300  # 5 minutes
            ),
            "sync": QueueConfig(
                name="sync",
                routing_key="task.sync",
                priority=TaskPriority.NORMAL,
                max_length=5000
            ),
            "maintenance": QueueConfig(
                name="maintenance",
                routing_key="task.maintenance",
                priority=TaskPriority.LOW,
                max_length=100
            )
        }

    def get_queue(self, name: str) -> QueueConfig:
        """Get queue configuration.

        Args:
            name: Queue name

        Returns:
            Queue configuration
        """
        return self.queues.get(name, self.queues["default"])

    def get_all_queues(self) -> List[QueueConfig]:
        """Get all queue configurations.

        Returns:
            List of queue configurations
        """
        return list(self.queues.values())


# ============================================================================
# CONFIGURATION FACTORY
# ============================================================================

class ConfigFactory:
    """Factory for creating Celery configurations.

    Demonstrates:
    - Factory pattern
    - Environment-based configuration
    """

    @staticmethod
    def create_development_config() -> CeleryConfig:
        """Create development configuration.

        Returns:
            Development configuration
        """
        return CeleryConfig(
            broker_url="redis://localhost:6379/0",
            result_backend="redis://localhost:6379/1",
            task_always_eager=False,
            worker_pool="prefork",
            worker_prefetch_multiplier=1,
            task_soft_time_limit=60,
            task_time_limit=120
        )

    @staticmethod
    def create_production_config() -> CeleryConfig:
        """Create production configuration.

        Returns:
            Production configuration
        """
        return CeleryConfig(
            broker_url="redis://redis-cluster:6379/0",
            result_backend="redis://redis-cluster:6379/1",
            task_always_eager=False,
            worker_pool="prefork",
            worker_prefetch_multiplier=4,
            worker_max_tasks_per_child=1000,
            task_soft_time_limit=300,
            task_time_limit=600,
            task_acks_late=True,
            result_expires=7200  # 2 hours
        )

    @staticmethod
    def create_testing_config() -> CeleryConfig:
        """Create testing configuration.

        Returns:
            Testing configuration
        """
        return CeleryConfig(
            broker_url="memory://",
            result_backend="cache+memory://",
            task_always_eager=True,  # Execute synchronously
            task_eager_propagates=True,
            task_store_errors_even_if_ignored=True
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Development config
    dev_config = ConfigFactory.create_development_config()
    print("Development Config:")
    print(f"  Broker: {dev_config.broker_url}")
    print(f"  Pool: {dev_config.worker_pool}")
    print(f"  Time limit: {dev_config.task_time_limit}s")

    # Production config
    prod_config = ConfigFactory.create_production_config()
    print("\nProduction Config:")
    print(f"  Broker: {prod_config.broker_url}")
    print(f"  Pool: {prod_config.worker_pool}")
    print(f"  Time limit: {prod_config.task_time_limit}s")

    # Queue manager
    print("\nQueues:")
    queue_mgr = QueueManager()
    for queue in queue_mgr.get_all_queues():
        print(f"  - {queue.name}: {queue.routing_key} (priority={queue.priority})")
