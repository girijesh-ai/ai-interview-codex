"""
Celery Application Setup

Demonstrates:
- Celery application configuration
- Worker setup
- Task registration
- Monitoring integration
"""

from celery import Celery, Task
from celery.signals import (
    task_prerun,
    task_postrun,
    task_failure,
    task_retry,
    worker_ready,
    worker_shutdown
)
from typing import Any, Optional
import logging
from datetime import datetime

from .config import CeleryConfig, ConfigFactory, QueueManager

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM TASK BASE CLASS
# ============================================================================

class BaseTask(Task):
    """Base task with enhanced functionality.

    Demonstrates:
    - Template method pattern
    - Hook methods for lifecycle events
    - Error handling
    """

    # Retry configuration
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True  # Exponential backoff
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True  # Add randomness to backoff

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when task succeeds.

        Args:
            retval: Return value
            task_id: Task ID
            args: Task args
            kwargs: Task kwargs
        """
        logger.info(f"Task {self.name} [{task_id}] succeeded")

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any
    ) -> None:
        """Called when task fails.

        Args:
            exc: Exception raised
            task_id: Task ID
            args: Task args
            kwargs: Task kwargs
            einfo: Exception info
        """
        logger.error(
            f"Task {self.name} [{task_id}] failed: {exc}",
            exc_info=einfo
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any
    ) -> None:
        """Called when task is retried.

        Args:
            exc: Exception that caused retry
            task_id: Task ID
            args: Task args
            kwargs: Task kwargs
            einfo: Exception info
        """
        logger.warning(
            f"Task {self.name} [{task_id}] retrying due to: {exc}"
        )

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called before task execution.

        Args:
            task_id: Task ID
            args: Task args
            kwargs: Task kwargs
        """
        logger.debug(f"Task {self.name} [{task_id}] starting")

    def after_return(
        self,
        status: str,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any
    ) -> None:
        """Called after task returns.

        Args:
            status: Task status
            retval: Return value
            task_id: Task ID
            args: Task args
            kwargs: Task kwargs
            einfo: Exception info
        """
        logger.debug(f"Task {self.name} [{task_id}] returned with status: {status}")


# ============================================================================
# CELERY APPLICATION
# ============================================================================

def create_celery_app(config: Optional[CeleryConfig] = None) -> Celery:
    """Create Celery application.

    Demonstrates:
    - Factory pattern
    - Configuration injection

    Args:
        config: Celery configuration (defaults to development)

    Returns:
        Configured Celery app
    """
    if config is None:
        config = ConfigFactory.create_development_config()

    # Create Celery app
    app = Celery("enterprise_agent_system")

    # Update configuration
    app.config_from_object(config.to_dict())

    # Set base task class
    app.Task = BaseTask

    # Register task modules
    app.autodiscover_tasks([
        "src.infrastructure.celery.tasks.embedding",
        "src.infrastructure.celery.tasks.analytics",
        "src.infrastructure.celery.tasks.notifications",
        "src.infrastructure.celery.tasks.sync",
        "src.infrastructure.celery.tasks.maintenance"
    ])

    logger.info("Celery app created successfully")

    return app


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Handle task pre-run signal.

    Args:
        sender: Task sender
        task_id: Task ID
        task: Task instance
        args: Task args
        kwargs: Task kwargs
        **extra: Extra args
    """
    logger.info(f"Task starting: {task.name} [{task_id}]")


@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    **extra
):
    """Handle task post-run signal.

    Args:
        sender: Task sender
        task_id: Task ID
        task: Task instance
        args: Task args
        kwargs: Task kwargs
        retval: Return value
        **extra: Extra args
    """
    logger.info(f"Task completed: {task.name} [{task_id}]")


@task_failure.connect
def task_failure_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **extra
):
    """Handle task failure signal.

    Args:
        sender: Task sender
        task_id: Task ID
        exception: Exception raised
        args: Task args
        kwargs: Task kwargs
        traceback: Traceback
        einfo: Exception info
        **extra: Extra args
    """
    logger.error(
        f"Task failed: {sender.name} [{task_id}] - {exception}",
        exc_info=einfo
    )


@task_retry.connect
def task_retry_handler(
    sender=None,
    task_id=None,
    reason=None,
    einfo=None,
    **extra
):
    """Handle task retry signal.

    Args:
        sender: Task sender
        task_id: Task ID
        reason: Retry reason
        einfo: Exception info
        **extra: Extra args
    """
    logger.warning(f"Task retrying: {sender.name} [{task_id}] - {reason}")


@worker_ready.connect
def worker_ready_handler(sender=None, **extra):
    """Handle worker ready signal.

    Args:
        sender: Worker sender
        **extra: Extra args
    """
    logger.info(f"Worker ready: {sender.hostname}")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **extra):
    """Handle worker shutdown signal.

    Args:
        sender: Worker sender
        **extra: Extra args
    """
    logger.info(f"Worker shutting down: {sender.hostname}")


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

# Create default app instance
celery_app = create_celery_app()


# ============================================================================
# WORKER MANAGEMENT
# ============================================================================

class WorkerManager:
    """Manages Celery workers.

    Demonstrates:
    - Management pattern
    - Worker lifecycle
    """

    def __init__(self, app: Celery):
        """Initialize worker manager.

        Args:
            app: Celery application
        """
        self.app = app

    def inspect_active_tasks(self) -> dict:
        """Get active tasks from all workers.

        Returns:
            Dictionary of worker -> tasks
        """
        inspect = self.app.control.inspect()
        return inspect.active()

    def inspect_scheduled_tasks(self) -> dict:
        """Get scheduled tasks from all workers.

        Returns:
            Dictionary of worker -> tasks
        """
        inspect = self.app.control.inspect()
        return inspect.scheduled()

    def inspect_registered_tasks(self) -> dict:
        """Get registered tasks from all workers.

        Returns:
            Dictionary of worker -> tasks
        """
        inspect = self.app.control.inspect()
        return inspect.registered()

    def inspect_stats(self) -> dict:
        """Get stats from all workers.

        Returns:
            Dictionary of worker -> stats
        """
        inspect = self.app.control.inspect()
        return inspect.stats()

    def revoke_task(self, task_id: str, terminate: bool = False) -> None:
        """Revoke a task.

        Args:
            task_id: Task ID to revoke
            terminate: Whether to terminate if already executing
        """
        self.app.control.revoke(task_id, terminate=terminate)
        logger.info(f"Revoked task: {task_id}")

    def purge_queue(self, queue_name: str) -> int:
        """Purge all messages from a queue.

        Args:
            queue_name: Queue name

        Returns:
            Number of messages purged
        """
        with self.app.connection_or_acquire() as conn:
            return conn.default_channel.queue_purge(queue_name)

    def shutdown_worker(self, worker_name: str) -> None:
        """Shutdown a specific worker.

        Args:
            worker_name: Worker name
        """
        self.app.control.shutdown(destination=[worker_name])
        logger.info(f"Shutdown worker: {worker_name}")

    def add_consumer(
        self,
        queue: str,
        exchange: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> None:
        """Add a new queue consumer.

        Args:
            queue: Queue name
            exchange: Exchange name
            routing_key: Routing key
        """
        self.app.control.add_consumer(
            queue,
            exchange=exchange,
            routing_key=routing_key
        )
        logger.info(f"Added consumer for queue: {queue}")

    def cancel_consumer(self, queue: str) -> None:
        """Cancel a queue consumer.

        Args:
            queue: Queue name
        """
        self.app.control.cancel_consumer(queue)
        logger.info(f"Cancelled consumer for queue: {queue}")


# ============================================================================
# TASK MONITORING
# ============================================================================

class TaskMonitor:
    """Monitors task execution.

    Demonstrates:
    - Observer pattern
    - Metrics collection
    """

    def __init__(self, app: Celery):
        """Initialize task monitor.

        Args:
            app: Celery application
        """
        self.app = app
        self.metrics = {
            "total_tasks": 0,
            "succeeded_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0
        }

    def record_task_start(self, task_id: str, task_name: str) -> None:
        """Record task start.

        Args:
            task_id: Task ID
            task_name: Task name
        """
        self.metrics["total_tasks"] += 1
        logger.debug(f"Task started: {task_name} [{task_id}]")

    def record_task_success(self, task_id: str, task_name: str) -> None:
        """Record task success.

        Args:
            task_id: Task ID
            task_name: Task name
        """
        self.metrics["succeeded_tasks"] += 1
        logger.debug(f"Task succeeded: {task_name} [{task_id}]")

    def record_task_failure(self, task_id: str, task_name: str, error: str) -> None:
        """Record task failure.

        Args:
            task_id: Task ID
            task_name: Task name
            error: Error message
        """
        self.metrics["failed_tasks"] += 1
        logger.error(f"Task failed: {task_name} [{task_id}] - {error}")

    def record_task_retry(self, task_id: str, task_name: str) -> None:
        """Record task retry.

        Args:
            task_id: Task ID
            task_name: Task name
        """
        self.metrics["retried_tasks"] += 1
        logger.warning(f"Task retried: {task_name} [{task_id}]")

    def get_metrics(self) -> dict:
        """Get current metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.copy()

    def get_success_rate(self) -> float:
        """Get success rate.

        Returns:
            Success rate (0.0 to 1.0)
        """
        total = self.metrics["succeeded_tasks"] + self.metrics["failed_tasks"]
        if total == 0:
            return 0.0
        return self.metrics["succeeded_tasks"] / total


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create app
    app = create_celery_app()

    print("Celery App Configuration:")
    print(f"  Broker: {app.conf.broker_url}")
    print(f"  Backend: {app.conf.result_backend}")
    print(f"  Serializer: {app.conf.task_serializer}")

    # Worker manager
    manager = WorkerManager(app)

    print("\nWorker Manager Commands:")
    print("  - inspect_active_tasks()")
    print("  - inspect_scheduled_tasks()")
    print("  - revoke_task(task_id)")
    print("  - purge_queue(queue_name)")

    # Task monitor
    monitor = TaskMonitor(app)
    print("\nTask Monitor initialized")
    print(f"  Metrics: {monitor.get_metrics()}")
