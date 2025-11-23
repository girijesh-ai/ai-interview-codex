"""
Vector Database Integration for Long-Term Semantic Memory

Demonstrates:
- Repository pattern for vector storage
- Strategy pattern for different vector DB backends
- Semantic search with embeddings
- RAG (Retrieval Augmented Generation)
- Factory pattern for vector store creation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ============================================================================
# VALUE OBJECTS
# ============================================================================

@dataclass(frozen=True)
class SearchResult:
    """Value object for search results.

    Demonstrates:
    - Value object pattern
    - Immutability
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

    def is_relevant(self, threshold: float = 0.7) -> bool:
        """Check if result is relevant.

        Args:
            threshold: Relevance threshold

        Returns:
            True if score >= threshold
        """
        return self.score >= threshold


@dataclass
class EmbeddingRequest:
    """Request for generating embeddings.

    Demonstrates:
    - Data transfer object
    - Request/Response pattern
    """
    text: str
    metadata: Dict[str, Any]
    request_id: str = None

    def __post_init__(self):
        """Initialize request ID."""
        if self.request_id is None:
            self.request_id = str(uuid4())


# ============================================================================
# ABSTRACT VECTOR STORE - Repository Pattern
# ============================================================================

class VectorStore(ABC):
    """Abstract vector store interface.

    Demonstrates:
    - Repository pattern
    - Abstract base class
    - Dependency Inversion Principle
    """

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        namespace: str = "default"
    ) -> List[str]:
        """Add documents to vector store.

        Args:
            documents: Documents to add
            namespace: Namespace for organization

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results
            namespace: Namespace to search
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default"
    ) -> List[Tuple[Document, float]]:
        """Search with explicit scores.

        Args:
            query: Search query
            k: Number of results
            namespace: Namespace to search

        Returns:
            List of (document, score) tuples
        """
        pass

    @abstractmethod
    async def delete_documents(
        self,
        ids: List[str],
        namespace: str = "default"
    ) -> bool:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete
            namespace: Namespace

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_document(
        self,
        doc_id: str,
        namespace: str = "default"
    ) -> Optional[Document]:
        """Get document by ID.

        Args:
            doc_id: Document ID
            namespace: Namespace

        Returns:
            Document or None
        """
        pass


# ============================================================================
# WEAVIATE IMPLEMENTATION - Strategy Pattern
# ============================================================================

class WeaviateVectorStore(VectorStore):
    """Weaviate vector database implementation.

    Demonstrates:
    - Strategy pattern
    - Concrete implementation
    - External system integration
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        embedding_function: Optional[OpenAIEmbeddings] = None
    ):
        """Initialize Weaviate client.

        Args:
            url: Weaviate instance URL
            api_key: Optional API key
            embedding_function: Function to generate embeddings
        """
        self.url = url
        self.api_key = api_key
        self.embedding_function = embedding_function or OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # Initialize Weaviate client (lazy)
        self._client = None

    @property
    def client(self):
        """Get Weaviate client (lazy initialization).

        Returns:
            Weaviate client
        """
        if self._client is None:
            import weaviate

            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)

            self._client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )

        return self._client

    async def add_documents(
        self,
        documents: List[Document],
        namespace: str = "default"
    ) -> List[str]:
        """Add documents to Weaviate.

        Args:
            documents: Documents to add
            namespace: Collection namespace

        Returns:
            List of document IDs
        """
        collection_name = self._get_collection_name(namespace)

        # Ensure collection exists
        await self._ensure_collection(collection_name)

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_function.aembed_documents(texts)

        # Add to Weaviate
        doc_ids = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid4())

            data_object = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": datetime.now().isoformat()
            }

            self.client.data_object.create(
                data_object=data_object,
                class_name=collection_name,
                uuid=doc_id,
                vector=embedding
            )

            doc_ids.append(doc_id)

        return doc_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Semantic similarity search in Weaviate.

        Args:
            query: Search query
            k: Number of results
            namespace: Collection namespace
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        collection_name = self._get_collection_name(namespace)

        # Generate query embedding
        query_embedding = await self.embedding_function.aembed_query(query)

        # Build Weaviate query
        query_builder = (
            self.client.query
            .get(collection_name, ["content", "metadata", "timestamp"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .with_additional(["certainty", "id"])
        )

        # Add filter if provided
        if filter:
            query_builder = query_builder.with_where(filter)

        # Execute query
        result = query_builder.do()

        # Parse results
        search_results = []

        if "data" in result and "Get" in result["data"]:
            items = result["data"]["Get"].get(collection_name, [])

            for item in items:
                search_result = SearchResult(
                    id=item["_additional"]["id"],
                    content=item["content"],
                    score=item["_additional"]["certainty"],
                    metadata=item.get("metadata", {})
                )
                search_results.append(search_result)

        return search_results

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default"
    ) -> List[Tuple[Document, float]]:
        """Search with document objects and scores.

        Args:
            query: Search query
            k: Number of results
            namespace: Collection namespace

        Returns:
            List of (Document, score) tuples
        """
        search_results = await self.similarity_search(query, k, namespace)

        return [
            (
                Document(
                    page_content=result.content,
                    metadata=result.metadata
                ),
                result.score
            )
            for result in search_results
        ]

    async def delete_documents(
        self,
        ids: List[str],
        namespace: str = "default"
    ) -> bool:
        """Delete documents from Weaviate.

        Args:
            ids: Document IDs to delete
            namespace: Collection namespace

        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(namespace)

        try:
            for doc_id in ids:
                self.client.data_object.delete(
                    uuid=doc_id,
                    class_name=collection_name
                )
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    async def get_document(
        self,
        doc_id: str,
        namespace: str = "default"
    ) -> Optional[Document]:
        """Get document by ID from Weaviate.

        Args:
            doc_id: Document ID
            namespace: Collection namespace

        Returns:
            Document or None
        """
        collection_name = self._get_collection_name(namespace)

        try:
            result = self.client.data_object.get_by_id(
                uuid=doc_id,
                class_name=collection_name
            )

            if result:
                return Document(
                    page_content=result["properties"]["content"],
                    metadata=result["properties"].get("metadata", {})
                )
        except Exception as e:
            print(f"Error getting document: {e}")

        return None

    def _get_collection_name(self, namespace: str) -> str:
        """Get Weaviate collection name.

        Args:
            namespace: Namespace

        Returns:
            Collection name (capitalized for Weaviate)
        """
        return f"AgentMemory_{namespace.capitalize()}"

    async def _ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists in Weaviate.

        Args:
            collection_name: Collection name
        """
        # Check if class exists
        schema = self.client.schema.get()
        class_names = [c["class"] for c in schema.get("classes", [])]

        if collection_name not in class_names:
            # Create class
            class_obj = {
                "class": collection_name,
                "description": f"Agent memory for {collection_name}",
                "vectorizer": "none",  # We provide vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Document metadata"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    }
                ]
            }

            self.client.schema.create_class(class_obj)


# ============================================================================
# QDRANT IMPLEMENTATION - Alternative Strategy
# ============================================================================

class QdrantVectorStore(VectorStore):
    """Qdrant vector database implementation.

    Demonstrates:
    - Strategy pattern
    - Alternative implementation
    - Flexible backend support
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        embedding_function: Optional[OpenAIEmbeddings] = None
    ):
        """Initialize Qdrant client.

        Args:
            url: Qdrant instance URL
            api_key: Optional API key
            embedding_function: Function to generate embeddings
        """
        self.url = url
        self.api_key = api_key
        self.embedding_function = embedding_function or OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self._client = None

    @property
    def client(self):
        """Get Qdrant client (lazy initialization).

        Returns:
            Qdrant client
        """
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )

        return self._client

    async def add_documents(
        self,
        documents: List[Document],
        namespace: str = "default"
    ) -> List[str]:
        """Add documents to Qdrant.

        Args:
            documents: Documents to add
            namespace: Collection namespace

        Returns:
            List of document IDs
        """
        from qdrant_client.models import PointStruct

        collection_name = self._get_collection_name(namespace)

        # Ensure collection exists
        await self._ensure_collection(collection_name)

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_function.aembed_documents(texts)

        # Create points
        points = []
        doc_ids = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid4())
            doc_ids.append(doc_id)

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)

        # Upload points
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

        return doc_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Semantic similarity search in Qdrant.

        Args:
            query: Search query
            k: Number of results
            namespace: Collection namespace
            filter: Optional metadata filter

        Returns:
            List of search results
        """
        collection_name = self._get_collection_name(namespace)

        # Generate query embedding
        query_embedding = await self.embedding_function.aembed_query(query)

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=filter
        )

        # Convert to SearchResult objects
        search_results = []

        for result in results:
            search_result = SearchResult(
                id=str(result.id),
                content=result.payload["content"],
                score=result.score,
                metadata=result.payload.get("metadata", {})
            )
            search_results.append(search_result)

        return search_results

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default"
    ) -> List[Tuple[Document, float]]:
        """Search with document objects and scores.

        Args:
            query: Search query
            k: Number of results
            namespace: Collection namespace

        Returns:
            List of (Document, score) tuples
        """
        search_results = await self.similarity_search(query, k, namespace)

        return [
            (
                Document(
                    page_content=result.content,
                    metadata=result.metadata
                ),
                result.score
            )
            for result in search_results
        ]

    async def delete_documents(
        self,
        ids: List[str],
        namespace: str = "default"
    ) -> bool:
        """Delete documents from Qdrant.

        Args:
            ids: Document IDs to delete
            namespace: Collection namespace

        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(namespace)

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=ids
            )
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    async def get_document(
        self,
        doc_id: str,
        namespace: str = "default"
    ) -> Optional[Document]:
        """Get document by ID from Qdrant.

        Args:
            doc_id: Document ID
            namespace: Collection namespace

        Returns:
            Document or None
        """
        collection_name = self._get_collection_name(namespace)

        try:
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=[doc_id]
            )

            if results:
                result = results[0]
                return Document(
                    page_content=result.payload["content"],
                    metadata=result.payload.get("metadata", {})
                )
        except Exception as e:
            print(f"Error getting document: {e}")

        return None

    def _get_collection_name(self, namespace: str) -> str:
        """Get Qdrant collection name.

        Args:
            namespace: Namespace

        Returns:
            Collection name
        """
        return f"agent_memory_{namespace}"

    async def _ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists in Qdrant.

        Args:
            collection_name: Collection name
        """
        from qdrant_client.models import Distance, VectorParams

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=Distance.COSINE
                )
            )


# ============================================================================
# FACTORY PATTERN - Vector Store Creation
# ============================================================================

class VectorStoreFactory:
    """Factory for creating vector stores.

    Demonstrates:
    - Factory pattern
    - Configuration-driven creation
    - Easy extension for new backends
    """

    @staticmethod
    def create(
        backend: str,
        url: str,
        api_key: Optional[str] = None,
        embedding_function: Optional[OpenAIEmbeddings] = None
    ) -> VectorStore:
        """Create vector store based on backend type.

        Args:
            backend: Backend type ("weaviate" or "qdrant")
            url: Vector store URL
            api_key: Optional API key
            embedding_function: Optional embedding function

        Returns:
            VectorStore instance

        Raises:
            ValueError: If backend is not supported
        """
        backend = backend.lower()

        if backend == "weaviate":
            return WeaviateVectorStore(url, api_key, embedding_function)
        elif backend == "qdrant":
            return QdrantVectorStore(url, api_key, embedding_function)
        else:
            raise ValueError(f"Unsupported vector store backend: {backend}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Create vector store using factory
        vector_store = VectorStoreFactory.create(
            backend="weaviate",
            url="http://localhost:8080",
            api_key=None
        )

        # Add documents
        documents = [
            Document(
                page_content="How to reset your password: Go to settings and click 'Reset Password'",
                metadata={"category": "account", "type": "faq"}
            ),
            Document(
                page_content="Refund policy: We offer full refunds within 30 days",
                metadata={"category": "billing", "type": "policy"}
            )
        ]

        doc_ids = await vector_store.add_documents(documents, namespace="knowledge_base")
        print(f"Added documents: {doc_ids}")

        # Search
        results = await vector_store.similarity_search(
            "How do I reset my password?",
            k=3,
            namespace="knowledge_base"
        )

        print(f"\nSearch results:")
        for result in results:
            print(f"- {result.content[:100]}... (score: {result.score:.2f})")

    # Run
    asyncio.run(main())
