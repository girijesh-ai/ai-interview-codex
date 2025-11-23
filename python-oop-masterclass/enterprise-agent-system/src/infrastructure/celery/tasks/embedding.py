"""
Embedding Generation Tasks

Background tasks for generating embeddings and updating vector DB.

Demonstrates:
- Async task pattern
- Batch processing
- Error handling
- Progress tracking
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..app import celery_app
from ....memory.vector_store import VectorStoreFactory, Document

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDING TASKS
# ============================================================================

@celery_app.task(
    bind=True,
    name="tasks.embedding.generate_conversation_embedding",
    queue="embedding",
    max_retries=3
)
def generate_conversation_embedding(
    self,
    conversation_id: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate embedding for conversation and store in vector DB.

    Args:
        self: Task instance (bound)
        conversation_id: Conversation ID
        messages: List of messages
        metadata: Additional metadata

    Returns:
        Result dictionary
    """
    try:
        logger.info(f"Generating embedding for conversation: {conversation_id}")

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
                "processed_at": datetime.now().isoformat(),
                "task_id": self.request.id
            }
        )

        # Store in vector DB
        vector_store = VectorStoreFactory.create("weaviate", "http://localhost:8080")
        success = vector_store.add_documents([doc], namespace="conversations")

        if success:
            logger.info(f"Successfully stored embedding for: {conversation_id}")
            return {
                "conversation_id": conversation_id,
                "status": "success",
                "message_count": len(messages),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to store document in vector DB")

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(
    bind=True,
    name="tasks.embedding.batch_generate_embeddings",
    queue="embedding",
    max_retries=2
)
def batch_generate_embeddings(
    self,
    conversations: List[Dict[str, Any]],
    namespace: str = "conversations"
) -> Dict[str, Any]:
    """Generate embeddings for multiple conversations in batch.

    Args:
        self: Task instance (bound)
        conversations: List of conversation dicts
        namespace: Vector DB namespace

    Returns:
        Batch result
    """
    try:
        logger.info(f"Batch generating embeddings for {len(conversations)} conversations")

        # Update state to track progress
        self.update_state(
            state='PROCESSING',
            meta={'current': 0, 'total': len(conversations)}
        )

        # Create documents
        documents = []
        for i, conv in enumerate(conversations):
            content = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in conv.get("messages", [])
            ])

            doc = Document(
                id=conv.get("id"),
                content=content,
                metadata={
                    **conv.get("metadata", {}),
                    "processed_at": datetime.now().isoformat(),
                    "batch_task_id": self.request.id
                }
            )
            documents.append(doc)

            # Update progress
            self.update_state(
                state='PROCESSING',
                meta={'current': i + 1, 'total': len(conversations)}
            )

        # Store in vector DB
        vector_store = VectorStoreFactory.create("weaviate", "http://localhost:8080")
        success = vector_store.add_documents(documents, namespace=namespace)

        if success:
            logger.info(f"Successfully stored {len(documents)} embeddings")
            return {
                "status": "success",
                "count": len(documents),
                "namespace": namespace,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to store documents in vector DB")

    except Exception as e:
        logger.error(f"Error in batch embedding generation: {e}")
        raise self.retry(exc=e, countdown=120)


@celery_app.task(
    name="tasks.embedding.update_knowledge_base",
    queue="embedding",
    max_retries=3
)
def update_knowledge_base(
    article_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Update knowledge base with new article.

    Args:
        article_id: Article ID
        content: Article content
        metadata: Additional metadata

    Returns:
        Result dictionary
    """
    try:
        logger.info(f"Updating knowledge base with article: {article_id}")

        # Create document
        doc = Document(
            id=article_id,
            content=content,
            metadata={
                **(metadata or {}),
                "added_at": datetime.now().isoformat(),
                "type": "knowledge_article"
            }
        )

        # Store in vector DB
        vector_store = VectorStoreFactory.create("weaviate", "http://localhost:8080")
        success = vector_store.add_documents([doc], namespace="knowledge_base")

        if success:
            logger.info(f"Successfully added article: {article_id}")
            return {
                "article_id": article_id,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to add article to knowledge base")

    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        raise


@celery_app.task(
    name="tasks.embedding.reindex_conversations",
    queue="embedding",
    soft_time_limit=3600,  # 1 hour
    time_limit=7200  # 2 hours
)
def reindex_conversations(batch_size: int = 100) -> Dict[str, Any]:
    """Reindex all conversations (maintenance task).

    Args:
        batch_size: Batch size for processing

    Returns:
        Reindex result
    """
    try:
        logger.info("Starting conversation reindexing")

        # This would fetch conversations from database
        # For now, just a placeholder

        total_processed = 0
        total_batches = 0

        logger.info(f"Reindexing complete: {total_processed} conversations in {total_batches} batches")

        return {
            "status": "success",
            "total_processed": total_processed,
            "total_batches": total_batches,
            "batch_size": batch_size,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error reindexing conversations: {e}")
        raise


# ============================================================================
# EMBEDDING CLEANUP TASKS
# ============================================================================

@celery_app.task(
    name="tasks.embedding.cleanup_old_embeddings",
    queue="maintenance"
)
def cleanup_old_embeddings(days_old: int = 90) -> Dict[str, Any]:
    """Clean up old embeddings from vector DB.

    Args:
        days_old: Age threshold in days

    Returns:
        Cleanup result
    """
    try:
        logger.info(f"Cleaning up embeddings older than {days_old} days")

        # This would delete old embeddings from vector DB
        # Implementation depends on vector store capabilities

        deleted_count = 0

        logger.info(f"Cleaned up {deleted_count} old embeddings")

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "days_old": days_old,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error cleaning up embeddings: {e}")
        raise


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate single conversation embedding
    result = generate_conversation_embedding.delay(
        conversation_id="conv-123",
        messages=[
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "You can reset your password by..."}
        ],
        metadata={"category": "account", "priority": 2}
    )

    print(f"Task submitted: {result.id}")
    print(f"Task state: {result.state}")

    # Wait for result (blocking)
    # result_data = result.get(timeout=30)
    # print(f"Result: {result_data}")
