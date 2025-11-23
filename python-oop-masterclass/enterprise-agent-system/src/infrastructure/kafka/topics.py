"""
Kafka Topic Definitions and Configuration

Demonstrates:
- Configuration as code
- Type safety with enums
- Topic management
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================================
# TOPIC NAMES
# ============================================================================

class TopicName(str, Enum):
    """Kafka topic names."""
    CUSTOMER_REQUESTS = "customer.requests"
    AGENT_ACTIONS = "agent.actions"
    HUMAN_APPROVALS = "human.approvals"
    SYSTEM_ANALYTICS = "system.analytics"
    NOTIFICATIONS = "notifications.outbound"


# ============================================================================
# TOPIC CONFIGURATION
# ============================================================================

@dataclass
class TopicSpec:
    """Topic specification.

    Demonstrates:
    - Value object pattern
    - Configuration as data
    """
    name: str
    partitions: int = 3
    replication_factor: int = 1
    retention_ms: int = 604800000  # 7 days
    cleanup_policy: str = "delete"  # delete or compact
    compression_type: str = "gzip"
    min_insync_replicas: int = 1


class TopicConfig:
    """Topic configuration manager.

    Demonstrates:
    - Configuration management
    - Singleton-like access
    """

    def __init__(self):
        """Initialize topic configurations."""
        self.topics = {
            TopicName.CUSTOMER_REQUESTS: TopicSpec(
                name=TopicName.CUSTOMER_REQUESTS.value,
                partitions=6,
                replication_factor=2,
                retention_ms=2592000000,  # 30 days
                cleanup_policy="delete",
                compression_type="gzip"
            ),
            TopicName.AGENT_ACTIONS: TopicSpec(
                name=TopicName.AGENT_ACTIONS.value,
                partitions=12,
                replication_factor=2,
                retention_ms=604800000,  # 7 days
                cleanup_policy="delete",
                compression_type="gzip"
            ),
            TopicName.HUMAN_APPROVALS: TopicSpec(
                name=TopicName.HUMAN_APPROVALS.value,
                partitions=3,
                replication_factor=2,
                retention_ms=2592000000,  # 30 days
                cleanup_policy="delete",
                compression_type="gzip"
            ),
            TopicName.SYSTEM_ANALYTICS: TopicSpec(
                name=TopicName.SYSTEM_ANALYTICS.value,
                partitions=6,
                replication_factor=1,
                retention_ms=604800000,  # 7 days
                cleanup_policy="delete",
                compression_type="gzip"
            ),
            TopicName.NOTIFICATIONS: TopicSpec(
                name=TopicName.NOTIFICATIONS.value,
                partitions=3,
                replication_factor=1,
                retention_ms=86400000,  # 1 day
                cleanup_policy="delete",
                compression_type="gzip"
            )
        }

    def get_topic(self, topic_name: TopicName) -> TopicSpec:
        """Get topic specification.

        Args:
            topic_name: Topic name

        Returns:
            Topic specification
        """
        return self.topics[topic_name]

    def get_all_topics(self) -> Dict[TopicName, TopicSpec]:
        """Get all topic specifications.

        Returns:
            Dictionary of topic specifications
        """
        return self.topics.copy()
