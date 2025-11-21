# Weaviate Vector Database - Complete Guide

> A comprehensive, hands-on guide to Weaviate for ML/AI engineers preparing for interviews and building production systems.

## Table of Contents
1. [Introduction to Weaviate](#introduction-to-weaviate)
2. [Key Concepts](#key-concepts)
3. [Installation & Setup](#installation--setup)
4. [Core Operations](#core-operations)
5. [Vector Search](#vector-search)
6. [Filters & Conditional Queries](#filters--conditional-queries)
7. [Integrations](#integrations)
8. [Production Best Practices](#production-best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Interview Questions](#interview-questions)

---

## Introduction to Weaviate

### What is Weaviate?

**Weaviate** is an open-source vector database designed for storing and searching large-scale vector embeddings. It's optimized for semantic search, recommendation systems, and AI-powered applications.

**Key Features:**
- **Vector-native storage** - Built specifically for vector embeddings
- **Hybrid search** - Combines vector (semantic) and keyword (BM25) search
- **Modular architecture** - Pluggable vectorizers (OpenAI, Cohere, HuggingFace)
- **GraphQL API** - Intuitive query language
- **HNSW indexing** - Hierarchical Navigable Small World for fast ANN search
- **Horizontal scalability** - Distributed architecture for production
- **RESTful + gRPC** - Multiple API options

### Why Weaviate?

**Advantages:**
- **Performance** - Sub-millisecond vector search at scale
- **Flexibility** - Bring your own vectors or use built-in vectorizers
- **Rich filtering** - Combine vector similarity with attribute filters
- **Multi-tenancy** - Native support for isolated data per tenant
- **Generative search** - Built-in RAG with LLM integration

**Use Cases:**
- Semantic search over documents, products, images
- Recommendation engines
- RAG (Retrieval-Augmented Generation) systems
- Anomaly detection
- Question answering systems
- Content discovery and clustering

### Weaviate vs Other Vector Databases

| Feature | Weaviate | Pinecone | Qdrant | Chroma | Milvus |
|---------|----------|----------|--------|--------|--------|
| **Open Source** | Yes | No | Yes | Yes | Yes |
| **Cloud Option** | Yes | Yes | Yes | No | Yes |
| **Hybrid Search** | Yes | No | Yes | No | Yes |
| **GraphQL API** | Yes | No | No | No | No |
| **Multi-tenancy** | Native | Yes | Yes | Limited | Yes |
| **Built-in Vectorizers** | Yes | No | Limited | Yes | No |
| **HNSW Index** | Yes | Yes | Yes | Yes | Yes |
| **Production Ready** | Yes | Yes | Yes | Emerging | Yes |

**When to Choose Weaviate:**
- Need hybrid search (vector + keyword)
- GraphQL API preference
- Want built-in vectorizer modules
- Require multi-tenancy
- Open-source requirement with cloud option

---

## Key Concepts

### 1. Collections (formerly Classes)

**Collections** are schemas that define the structure of your data. Similar to tables in SQL or collections in MongoDB.

```python
collection = {
    "class": "Article",
    "description": "News articles with embeddings",
    "vectorizer": "text2vec-openai",
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Article title"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Article body"
        },
        {
            "name": "author",
            "dataType": ["text"],
            "description": "Article author"
        },
        {
            "name": "publishedDate",
            "dataType": ["date"],
            "description": "Publication date"
        }
    ]
}
```

**Key Properties:**
- `class`: Collection name (CamelCase convention)
- `vectorizer`: Module to generate embeddings (or "none" for custom vectors)
- `properties`: Fields in your data with types
- `moduleConfig`: Configuration for vectorizers, generative modules

### 2. Objects

**Objects** are individual data entries within a collection. Each object has:
- Properties (data fields)
- Vector embedding (stored automatically or provided)
- UUID (unique identifier)

### 3. Vectors

**Vectors** are numerical representations of your data (embeddings). Weaviate can:
- **Auto-generate** vectors using built-in modules (OpenAI, Cohere, etc.)
- **Accept custom vectors** you generate externally
- Store vectors using **HNSW** (Hierarchical Navigable Small World) index

**Vector Dimensions:**
- OpenAI Ada-002: 1536 dimensions
- Cohere embed-v3: 1024 dimensions
- HuggingFace (varies): 384-1024 dimensions

### 4. Schema

**Schema** defines the structure of all collections in your Weaviate instance. Best practices:
- Define schema explicitly before inserting data
- Use appropriate data types (text, int, boolean, date, etc.)
- Configure vectorizers per collection
- Set indexing options for performance

### 5. HNSW Index

**HNSW** (Hierarchical Navigable Small World) is Weaviate's default indexing algorithm for vector search.

**How it works:**
- Creates a multi-layer graph structure
- Navigates from top layer (sparse) to bottom (dense)
- Balances speed vs accuracy

**Configuration:**
```python
"vectorIndexConfig": {
    "ef": 100,              # Higher = more accurate but slower
    "efConstruction": 128,  # Higher = better quality index
    "maxConnections": 64    # More connections = faster search
}
```

**Trade-offs:**
- `ef`: Query-time accuracy (50-100 for balanced, 100+ for high accuracy)
- `efConstruction`: Build-time quality (128-512)
- `maxConnections`: Memory vs speed (16-128)

---

## Installation & Setup

### Option 1: Docker (Recommended for Local Development)

**Using Docker Compose:**

```yaml
# docker-compose.yml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.27.1
    ports:
    - 8080:8080
    - 50051:50051
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface,generative-openai,generative-cohere'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

**Start Weaviate:**
```bash
docker-compose up -d
```

**Verify it's running:**
```bash
curl http://localhost:8080/v1/meta
```

**Stop Weaviate:**
```bash
docker-compose down
```

**With data persistence:**
```bash
docker-compose down  # Stops but keeps data
docker-compose down -v  # Stops and removes data
```

### Option 2: Weaviate Cloud Service (WCS)

**Steps:**
1. Go to [https://console.weaviate.cloud](https://console.weaviate.cloud)
2. Create account and cluster
3. Get cluster URL and API key
4. Use in Python client

**Advantages:**
- Managed infrastructure
- Auto-scaling
- Built-in monitoring
- High availability

### Option 3: Kubernetes (Production)

```bash
helm repo add weaviate https://weaviate.github.io/weaviate-helm
helm install weaviate weaviate/weaviate
```

### Python Client Installation

```bash
pip install weaviate-client
```

**Versions:**
- **v3.x** - Legacy client (deprecated)
- **v4.x** - Current recommended version (async support, better typing)

---

## Core Operations

### 1. Connect to Weaviate

**Local instance:**
```python
import weaviate

# V4 client (recommended)
client = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

# Check connection
print(client.is_ready())
```

**Cloud instance:**
```python
import weaviate

client = weaviate.connect_to_wcs(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.AuthApiKey(api_key="YOUR-API-KEY")
)
```

### 2. Create Collection (Schema)

**Basic collection:**
```python
from weaviate.classes.config import Configure

# Create collection
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    properties=[
        {
            "name": "title",
            "dataType": ["text"]
        },
        {
            "name": "content",
            "dataType": ["text"]
        },
        {
            "name": "author",
            "dataType": ["text"]
        }
    ]
)

print("Collection created successfully!")
```

**With custom vectorizer config:**
```python
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric="cosine",
        ef=100,
        ef_construction=128,
        max_connections=64
    ),
    properties=[
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Article title",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Article content"
        }
    ]
)
```

### 3. Insert Data (Create Objects)

**Single object:**
```python
articles = client.collections.get("Article")

uuid = articles.data.insert(
    properties={
        "title": "Weaviate Vector Database Guide",
        "content": "Weaviate is an open-source vector database...",
        "author": "AI Engineer"
    }
)

print(f"Object created with UUID: {uuid}")
```

**Batch insert (recommended for multiple objects):**
```python
articles = client.collections.get("Article")

data_objects = [
    {
        "title": "Introduction to RAG",
        "content": "Retrieval-Augmented Generation combines...",
        "author": "John Doe"
    },
    {
        "title": "LLM Fine-tuning with LoRA",
        "content": "LoRA is a parameter-efficient method...",
        "author": "Jane Smith"
    },
    {
        "title": "Vector Databases Comparison",
        "content": "Vector databases like Weaviate, Pinecone...",
        "author": "AI Engineer"
    }
]

# Batch insert
with articles.batch.dynamic() as batch:
    for obj in data_objects:
        batch.add_object(properties=obj)

print(f"Inserted {len(data_objects)} objects")
```

**With custom vectors:**
```python
import numpy as np

# Generate or get your vectors (e.g., from OpenAI)
vector = np.random.rand(1536).tolist()  # 1536 for OpenAI Ada-002

articles.data.insert(
    properties={
        "title": "Custom Vector Example",
        "content": "Using pre-computed vectors...",
        "author": "Developer"
    },
    vector=vector
)
```

### 4. Read Data (Retrieve Objects)

**Get by UUID:**
```python
articles = client.collections.get("Article")

obj = articles.query.fetch_object_by_id(uuid="<uuid-here>")
print(obj.properties)
```

**Fetch multiple objects:**
```python
response = articles.query.fetch_objects(limit=10)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Author: {obj.properties['author']}\n")
```

**Fetch with specific properties:**
```python
response = articles.query.fetch_objects(
    limit=5,
    return_properties=["title", "author"]
)
```

### 5. Update Data

**Update object properties:**
```python
articles.data.update(
    uuid="<uuid-here>",
    properties={
        "title": "Updated Title",
        "author": "Updated Author"
    }
)
```

**Replace entire object:**
```python
articles.data.replace(
    uuid="<uuid-here>",
    properties={
        "title": "Completely New Title",
        "content": "Completely new content...",
        "author": "New Author"
    }
)
```

### 6. Delete Data

**Delete single object:**
```python
articles.data.delete_by_id(uuid="<uuid-here>")
```

**Delete with filter:**
```python
articles.data.delete_many(
    where={
        "path": ["author"],
        "operator": "Equal",
        "valueText": "John Doe"
    }
)
```

**Delete entire collection:**
```python
client.collections.delete("Article")
```

---

## Vector Search

### 1. Similarity Search (Near Vector)

**Basic semantic search:**
```python
articles = client.collections.get("Article")

response = articles.query.near_text(
    query="machine learning and AI",
    limit=5
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Distance: {obj.metadata.distance}\n")
```

**With custom vector:**
```python
import numpy as np

query_vector = np.random.rand(1536).tolist()

response = articles.query.near_vector(
    near_vector=query_vector,
    limit=5
)
```

**With certainty threshold:**
```python
response = articles.query.near_text(
    query="deep learning",
    limit=10,
    certainty=0.7  # Only return results with certainty >= 0.7
)
```

### 2. Hybrid Search (Vector + Keyword)

**Combines BM25 (keyword) and vector search:**
```python
response = articles.query.hybrid(
    query="machine learning transformers",
    limit=10,
    alpha=0.5  # 0 = pure keyword, 1 = pure vector, 0.5 = balanced
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Score: {obj.metadata.score}\n")
```

**Alpha parameter tuning:**
- `alpha=0.0` - Pure BM25 keyword search
- `alpha=0.5` - Balanced hybrid search (default)
- `alpha=1.0` - Pure vector search

**When to use hybrid search:**
- User queries with specific keywords (e.g., product names, IDs)
- Need to balance semantic similarity with exact matches
- Heterogeneous queries (some semantic, some exact)

### 3. Near Object Search

**Find similar objects to a specific object:**
```python
response = articles.query.near_object(
    near_object="<uuid-of-reference-object>",
    limit=5
)
```

---

## Filters & Conditional Queries

### 1. Basic Filters

**Equality filter:**
```python
response = articles.query.fetch_objects(
    filters={
        "path": ["author"],
        "operator": "Equal",
        "valueText": "John Doe"
    },
    limit=10
)
```

**Comparison filters:**
```python
# Greater than
response = articles.query.fetch_objects(
    filters={
        "path": ["publishedDate"],
        "operator": "GreaterThan",
        "valueDate": "2024-01-01T00:00:00Z"
    }
)

# Less than or equal
response = articles.query.fetch_objects(
    filters={
        "path": ["views"],
        "operator": "LessThanEqual",
        "valueInt": 1000
    }
)
```

### 2. Multiple Filters (AND/OR)

**AND filter:**
```python
response = articles.query.fetch_objects(
    filters={
        "operator": "And",
        "operands": [
            {
                "path": ["author"],
                "operator": "Equal",
                "valueText": "John Doe"
            },
            {
                "path": ["views"],
                "operator": "GreaterThan",
                "valueInt": 500
            }
        ]
    }
)
```

**OR filter:**
```python
response = articles.query.fetch_objects(
    filters={
        "operator": "Or",
        "operands": [
            {
                "path": ["author"],
                "operator": "Equal",
                "valueText": "John Doe"
            },
            {
                "path": ["author"],
                "operator": "Equal",
                "valueText": "Jane Smith"
            }
        ]
    }
)
```

### 3. Vector Search with Filters

**Combine similarity search with filters:**
```python
response = articles.query.near_text(
    query="machine learning",
    filters={
        "path": ["author"],
        "operator": "Equal",
        "valueText": "John Doe"
    },
    limit=5
)
```

**Hybrid search with filters:**
```python
response = articles.query.hybrid(
    query="transformers neural networks",
    filters={
        "operator": "And",
        "operands": [
            {
                "path": ["publishedDate"],
                "operator": "GreaterThan",
                "valueDate": "2023-01-01T00:00:00Z"
            },
            {
                "path": ["views"],
                "operator": "GreaterThan",
                "valueInt": 100
            }
        ]
    },
    alpha=0.7,
    limit=10
)
```

### 4. Text Matching Filters

**Like (contains):**
```python
response = articles.query.fetch_objects(
    filters={
        "path": ["title"],
        "operator": "Like",
        "valueText": "*machine*"  # Wildcards: * = any characters
    }
)
```

**ContainsAny/ContainsAll (for arrays):**
```python
response = articles.query.fetch_objects(
    filters={
        "path": ["tags"],
        "operator": "ContainsAny",
        "valueText": ["AI", "ML", "Deep Learning"]
    }
)
```

---

## Integrations

### 1. OpenAI Integration

**Setup:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create collection with OpenAI vectorizer
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"  # or text-embedding-3-large, ada-002
    )
)
```

**Models:**
- `text-embedding-3-small` - 1536 dims, faster, cheaper
- `text-embedding-3-large` - 3072 dims, more accurate
- `text-embedding-ada-002` - 1536 dims, legacy (still good)

**Generative search (RAG):**
```python
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    generative_config=Configure.Generative.openai(
        model="gpt-4o-mini"
    )
)

# Use generative search
response = articles.generate.near_text(
    query="explain machine learning",
    single_prompt="Summarize this article: {content}",
    limit=3
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Generated: {obj.generated}\n")
```

### 2. Cohere Integration

**Setup:**
```python
os.environ["COHERE_API_KEY"] = "your-api-key"

articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_cohere(
        model="embed-english-v3.0"  # or embed-multilingual-v3.0
    )
)
```

**Models:**
- `embed-english-v3.0` - English only, 1024 dims
- `embed-multilingual-v3.0` - 100+ languages, 1024 dims
- `embed-english-light-v3.0` - Lightweight, faster

### 3. HuggingFace Integration

**Setup:**
```python
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)
```

**Popular models:**
- `sentence-transformers/all-MiniLM-L6-v2` - 384 dims, fast
- `sentence-transformers/all-mpnet-base-v2` - 768 dims, accurate
- `BAAI/bge-base-en-v1.5` - 768 dims, state-of-the-art

**Inference endpoints:**
```python
vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
    model="sentence-transformers/all-MiniLM-L6-v2",
    endpoint_url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
)
```

### 4. Custom Vectors (Bring Your Own)

**Disable built-in vectorizer:**
```python
articles = client.collections.create(
    name="Article",
    vectorizer_config=Configure.Vectorizer.none()  # No auto-vectorization
)

# Insert with custom vectors
from openai import OpenAI
openai_client = OpenAI()

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Insert object with custom vector
articles.data.insert(
    properties={
        "title": "Custom Vector Example",
        "content": "Content here..."
    },
    vector=get_embedding("Custom Vector Example Content here...")
)
```

---

## Production Best Practices

### 1. Connection Management

**Use context managers:**
```python
import weaviate

with weaviate.connect_to_local() as client:
    articles = client.collections.get("Article")
    # Do operations
    response = articles.query.near_text(query="test", limit=5)

# Connection automatically closed
```

**Connection pooling for high-throughput:**
```python
client = weaviate.connect_to_local(
    additional_config=weaviate.Config(
        connection_config=weaviate.ConnectionConfig(
            session_pool_connections=20,
            session_pool_maxsize=100
        )
    )
)
```

### 2. Batch Operations

**Always use batching for bulk inserts:**
```python
articles = client.collections.get("Article")

# Good: Batch insert
with articles.batch.dynamic() as batch:
    for obj in large_dataset:
        batch.add_object(properties=obj)

# Bad: Individual inserts (slow!)
for obj in large_dataset:
    articles.data.insert(properties=obj)  # Don't do this
```

**Batch configuration:**
```python
with articles.batch.fixed_size(batch_size=100) as batch:
    for obj in dataset:
        batch.add_object(properties=obj)
```

**Error handling in batches:**
```python
with articles.batch.dynamic() as batch:
    for obj in dataset:
        batch.add_object(properties=obj)

    # Check for failures
    if failed_objects := batch.failed_objects:
        for obj in failed_objects:
            print(f"Failed: {obj.message}")
```

### 3. Indexing Strategy

**HNSW tuning for different scenarios:**

**High accuracy (slower indexing):**
```python
vector_index_config=Configure.VectorIndex.hnsw(
    ef=200,
    ef_construction=256,
    max_connections=128
)
```

**Balanced (default):**
```python
vector_index_config=Configure.VectorIndex.hnsw(
    ef=100,
    ef_construction=128,
    max_connections=64
)
```

**Fast indexing (lower accuracy):**
```python
vector_index_config=Configure.VectorIndex.hnsw(
    ef=50,
    ef_construction=64,
    max_connections=32
)
```

### 4. Multi-tenancy

**Enable multi-tenancy:**
```python
articles = client.collections.create(
    name="Article",
    multi_tenancy_config=Configure.multi_tenancy(enabled=True)
)

# Add tenants
articles.tenants.create(["tenant1", "tenant2", "tenant3"])

# Insert data for specific tenant
articles_tenant1 = articles.with_tenant("tenant1")
articles_tenant1.data.insert(properties={...})

# Query for specific tenant
response = articles_tenant1.query.near_text(query="test", limit=5)
```

**Use cases:**
- SaaS applications with isolated customer data
- Multi-organization platforms
- Compliance requirements (data residency)

### 5. Monitoring & Observability

**Health check:**
```python
if client.is_ready():
    print("Weaviate is ready!")
```

**Get cluster metadata:**
```python
meta = client.get_meta()
print(f"Version: {meta['version']}")
print(f"Modules: {meta['modules']}")
```

**Collection statistics:**
```python
articles = client.collections.get("Article")
stats = articles.aggregate.over_all(total_count=True)
print(f"Total objects: {stats.total_count}")
```

### 6. Backup & Restore

**Create backup:**
```python
client.backup.create(
    backup_id="my-backup-2024",
    backend="filesystem",
    include_collections=["Article", "User"]
)
```

**Restore backup:**
```python
client.backup.restore(
    backup_id="my-backup-2024",
    backend="filesystem"
)
```

### 7. Security

**API key authentication:**
```python
import weaviate
from weaviate.auth import AuthApiKey

client = weaviate.connect_to_wcs(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=AuthApiKey(api_key="your-api-key")
)
```

**OIDC authentication (for enterprise):**
```python
from weaviate.auth import AuthClientCredentials

client = weaviate.Client(
    url="https://your-instance.com",
    auth_client_secret=AuthClientCredentials(
        client_secret="your-secret",
        scope="openid"
    )
)
```

---

## Performance Optimization

### 1. Vector Compression

**Product Quantization (PQ):**
```python
vector_index_config=Configure.VectorIndex.hnsw(
    quantizer=Configure.VectorIndex.Quantizer.pq(
        segments=0,  # Auto-determine
        training_limit=100000
    )
)
```

**Binary Quantization (BQ):**
```python
vector_index_config=Configure.VectorIndex.hnsw(
    quantizer=Configure.VectorIndex.Quantizer.bq()
)
```

**Trade-offs:**
- **PQ**: 4-8x compression, <5% accuracy loss, best for large datasets
- **BQ**: 32x compression, 5-10% accuracy loss, fastest

### 2. Caching

**Enable query caching:**
```python
# Weaviate automatically caches frequently accessed objects
# Tune OS-level page cache for better performance
```

**Application-level caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str):
    response = articles.query.near_text(query=query, limit=5)
    return response.objects
```

### 3. Sharding

**Configure sharding for horizontal scaling:**
```python
articles = client.collections.create(
    name="Article",
    sharding_config=Configure.sharding(
        virtual_per_physical=128,
        desired_count=3  # Number of shards
    )
)
```

### 4. Query Optimization

**Return only needed properties:**
```python
# Good: Only return needed properties
response = articles.query.near_text(
    query="test",
    return_properties=["title", "author"],
    limit=5
)

# Bad: Return all properties (slower)
response = articles.query.near_text(query="test", limit=5)
```

**Use limit appropriately:**
```python
# Don't fetch more than you need
response = articles.query.near_text(query="test", limit=10)  # Not 1000
```

**Pagination for large results:**
```python
# Fetch in batches
offset = 0
batch_size = 100

while True:
    response = articles.query.fetch_objects(
        limit=batch_size,
        offset=offset
    )

    if not response.objects:
        break

    # Process batch
    for obj in response.objects:
        process(obj)

    offset += batch_size
```

---

## Interview Questions

### Fundamentals

**Q1: What is Weaviate and how does it differ from traditional databases?**

**Answer:**
Weaviate is a vector database designed for storing and searching vector embeddings. Key differences:
- **Storage**: Optimized for high-dimensional vectors (not just rows/columns)
- **Search**: Similarity search using vector distance metrics (not exact matches)
- **Indexing**: HNSW algorithm for approximate nearest neighbor (ANN) search
- **Use cases**: Semantic search, recommendations, RAG systems

Traditional databases (SQL/NoSQL) are optimized for exact matches and structured queries, while vector databases excel at semantic similarity.

---

**Q2: Explain HNSW indexing algorithm. Why does Weaviate use it?**

**Answer:**
HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.

**How it works:**
1. Creates a multi-layer graph (hierarchy of connections)
2. Top layers are sparse (long-distance jumps)
3. Bottom layers are dense (fine-grained search)
4. Search navigates from top to bottom, progressively refining

**Advantages:**
- Sub-linear search time: O(log N)
- High recall (accuracy): >95% with proper tuning
- Efficient memory usage
- Incremental updates (no need to rebuild index)

**Trade-offs:**
- Build time: Slower than simpler methods (IVF)
- Memory: More than flat indexes
- Tuning: Requires parameter optimization (ef, efConstruction, maxConnections)

Weaviate uses HNSW because it offers the best balance of speed, accuracy, and scalability for production systems.

---

**Q3: What is hybrid search? When would you use it over pure vector search?**

**Answer:**
Hybrid search combines:
- **BM25** (keyword-based, sparse vectors)
- **Vector search** (semantic, dense vectors)

**Formula:**
```
score = alpha * vector_score + (1 - alpha) * bm25_score
```

**When to use hybrid search:**
- Queries with specific keywords (product IDs, names)
- Mixed intent (semantic + exact match)
- Better ranking for ambiguous queries
- E-commerce: "Nike Air Max" (exact) vs "comfortable running shoes" (semantic)

**Alpha tuning:**
- `alpha=0`: Pure keyword (BM25)
- `alpha=0.5`: Balanced (default)
- `alpha=1`: Pure vector search

**Example:**
Query: "iPhone 15 Pro features"
- Pure vector: Might miss "iPhone 15 Pro" exact match
- Hybrid: Ranks "iPhone 15 Pro" higher while understanding "features" semantically

---

### Architecture & Design

**Q4: Design a RAG system using Weaviate. How would you optimize for 10K queries/day?**

**Answer:**

**Architecture:**
```
User Query → Query Preprocessing → Weaviate Hybrid Search →
Top-K Results → Context Assembly → LLM (GPT-4) → Response
```

**Implementation:**
1. **Collection schema:**
```python
docs = client.collections.create(
    name="Documents",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    generative_config=Configure.Generative.openai(model="gpt-4o-mini")
)
```

2. **Chunking strategy:**
- Chunk size: 512 tokens (balance context vs granularity)
- Overlap: 50 tokens (preserve context at boundaries)
- Metadata: source_doc, chunk_id, page_number

3. **Query flow:**
```python
# Hybrid search for better retrieval
response = docs.generate.hybrid(
    query=user_query,
    alpha=0.7,  # Favor vector search
    limit=5,    # Top 5 chunks
    grouped_task="Answer the question: {user_query} using: {content}"
)
```

**Optimization for 10K queries/day:**
- **Caching**: Cache common queries (Redis), 30-50% hit rate
- **Model routing**: Use GPT-4o-mini for simple queries, GPT-4 for complex
- **Batching**: Process similar queries together
- **HNSW tuning**: ef=100 (balanced), efConstruction=128
- **Horizontal scaling**: 2-3 Weaviate replicas with load balancer
- **Monitoring**: Track latency (p50, p95, p99), cache hit rate, cost per query

**Cost estimate:**
- Embeddings: 10K queries × $0.00002 = $0.20/day
- LLM: 10K queries × $0.01 = $100/day (with caching: ~$50/day)
- Infrastructure: $100/day (Weaviate + Redis)
- **Total**: ~$150/day with optimizations

---

**Q5: How would you implement multi-tenancy in Weaviate for a SaaS application?**

**Answer:**

**Approach: Native Multi-tenancy**

**Setup:**
```python
# Enable multi-tenancy
users = client.collections.create(
    name="UserData",
    multi_tenancy_config=Configure.multi_tenancy(enabled=True)
)

# Add tenants (customers)
users.tenants.create(["company_a", "company_b", "company_c"])
```

**Per-tenant operations:**
```python
# Insert data for Company A
company_a_collection = users.with_tenant("company_a")
company_a_collection.data.insert(properties={...})

# Query for Company A (isolated)
response = company_a_collection.query.near_text(query="test", limit=10)
```

**Advantages:**
- **Data isolation**: Physical separation at shard level
- **Performance**: No cross-tenant queries
- **Compliance**: Meets data residency requirements
- **Scalability**: Tenants distributed across shards

**Alternative: Application-level tenancy (not recommended)**
```python
# Don't do this - use native multi-tenancy instead
response = collection.query.near_text(
    query="test",
    filters={"path": ["tenant_id"], "operator": "Equal", "valueText": "company_a"}
)
```

**Disadvantages of app-level:**
- No physical isolation (security risk)
- All tenants in same shards (noisy neighbor)
- Requires filters on every query (error-prone)

**Production considerations:**
- **Tenant management**: API for creating/deleting tenants
- **Monitoring**: Per-tenant metrics (query count, latency, storage)
- **Billing**: Track usage per tenant
- **Limits**: Set per-tenant quotas (objects, queries/day)

---

### Production & Optimization

**Q6: Your Weaviate queries are slow (>2s per query). How would you debug and optimize?**

**Answer:**

**Step 1: Identify bottleneck**
```python
import time

start = time.time()
response = articles.query.near_text(query="test", limit=10)
print(f"Query time: {time.time() - start}s")

# Check if issue is:
# - Network latency
# - Weaviate processing
# - Large result size
```

**Step 2: Check HNSW configuration**
```python
# Get current config
collection = client.collections.get("Article")
config = collection.config.get()

# If ef is too high (>200), reduce it
client.collections.update(
    name="Article",
    vector_index_config=Configure.VectorIndex.hnsw(ef=100)
)
```

**Step 3: Optimize query**
```python
# Bad: Returning all properties and large limit
response = articles.query.near_text(query="test", limit=100)

# Good: Return only needed properties and reasonable limit
response = articles.query.near_text(
    query="test",
    limit=10,
    return_properties=["title", "author"]  # Only what you need
)
```

**Step 4: Enable compression**
```python
# Add PQ compression for faster search
vector_index_config=Configure.VectorIndex.hnsw(
    quantizer=Configure.VectorIndex.Quantizer.pq()
)
```

**Step 5: Check resource utilization**
```bash
# CPU, memory, disk I/O
docker stats weaviate

# If CPU is bottleneck: scale horizontally (more replicas)
# If memory is bottleneck: reduce maxConnections or use compression
# If disk is bottleneck: use SSD, increase IOPS
```

**Step 6: Application-level optimizations**
- **Caching**: Cache frequent queries (Redis)
- **Connection pooling**: Reuse connections
- **Async queries**: Use asyncio for concurrent queries
- **Batch queries**: Combine multiple queries where possible

**Common issues:**
- `ef` too high → Lower ef to 50-100
- Large vector dimensions → Use smaller embedding model or compression
- No compression → Enable PQ/BQ
- Network latency → Deploy Weaviate closer to application
- Cold start → Add warm-up queries after restart

---

**Q7: How do you handle updates to embeddings when underlying documents change?**

**Answer:**

**Option 1: Incremental updates (recommended)**
```python
# When document is updated
articles.data.update(
    uuid=doc_uuid,
    properties={
        "content": updated_content,
        # Vector is automatically regenerated by vectorizer
    }
)
```

**Option 2: Delete and re-insert**
```python
# Delete old
articles.data.delete_by_id(uuid=doc_uuid)

# Insert new
articles.data.insert(properties={
    "title": title,
    "content": updated_content
})
```

**Option 3: Batch updates**
```python
# For bulk updates
with articles.batch.dynamic() as batch:
    for doc_id, new_content in updated_docs:
        # Delete old
        articles.data.delete_by_id(uuid=doc_id)

        # Insert new
        batch.add_object(properties={
            "id": doc_id,
            "content": new_content
        })
```

**Production strategy:**
```python
# Track document versions
articles = client.collections.create(
    name="Article",
    properties=[
        {"name": "doc_id", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "version", "dataType": ["int"]},
        {"name": "last_updated", "dataType": ["date"]}
    ]
)

# Update flow
def update_document(doc_id: str, new_content: str):
    # Fetch current version
    current = articles.query.fetch_objects(
        filters={"path": ["doc_id"], "operator": "Equal", "valueText": doc_id},
        limit=1
    )

    if current.objects:
        # Delete old version
        articles.data.delete_by_id(uuid=current.objects[0].uuid)

    # Insert new version
    articles.data.insert(properties={
        "doc_id": doc_id,
        "content": new_content,
        "version": current.objects[0].properties['version'] + 1 if current.objects else 1,
        "last_updated": datetime.now().isoformat()
    })
```

**Considerations:**
- **Downtime**: Updates are atomic, no downtime
- **Re-indexing**: HNSW handles incremental updates efficiently
- **Cost**: Re-generating embeddings costs API calls (OpenAI, Cohere)
- **Monitoring**: Track update frequency, failed updates

---

**Q8: Compare Weaviate, Pinecone, and Qdrant. When would you choose each?**

**Answer:**

| Criteria | Weaviate | Pinecone | Qdrant |
|----------|----------|----------|--------|
| **Open Source** | Yes | No | Yes |
| **Deployment** | Self-hosted + Cloud | Cloud only | Self-hosted + Cloud |
| **Hybrid Search** | Yes (built-in) | No | Yes |
| **Built-in Vectorizers** | Yes (OpenAI, Cohere, HF) | No | Limited |
| **GraphQL API** | Yes | No | No |
| **Pricing** | Free (self-hosted) | Pay-per-index | Free (self-hosted) |
| **Multi-tenancy** | Native | Yes | Yes |
| **Performance** | Excellent | Excellent | Excellent |
| **Maturity** | Production-ready | Production-ready | Production-ready |

**Choose Weaviate when:**
- Need hybrid search (vector + keyword)
- Want built-in vectorizers (less code)
- Prefer GraphQL API
- Open-source requirement
- Complex filtering requirements

**Choose Pinecone when:**
- Want fully managed service (no ops)
- Willing to pay for convenience
- Simple vector search use case
- Need enterprise support

**Choose Qdrant when:**
- Want open-source with Rust performance
- Need advanced filtering (better than Weaviate)
- Prefer REST API
- Want on-premise deployment

**Personal recommendation:**
- **Startups/MVPs**: Weaviate (open-source, flexible, hybrid search)
- **Enterprises**: Pinecone (managed) or Weaviate Cloud
- **Advanced filtering**: Qdrant
- **Hybrid search**: Weaviate

---

## Summary

### Key Takeaways

1. **Weaviate is a vector-native database** optimized for semantic search and RAG systems
2. **HNSW indexing** provides sub-linear search time with high recall
3. **Hybrid search** combines vector and keyword search for better results
4. **Built-in vectorizers** (OpenAI, Cohere, HuggingFace) simplify implementation
5. **Multi-tenancy** enables SaaS applications with data isolation
6. **Production-ready** with compression, sharding, monitoring

### Next Steps

1. **Practice**: Complete the hands-on notebook (`weaviate-hands-on-guide.ipynb`)
2. **Build**: Implement a RAG system with Weaviate
3. **Optimize**: Experiment with HNSW tuning, compression, caching
4. **Scale**: Deploy multi-tenant Weaviate cluster

### Resources

- [Official Documentation](https://weaviate.io/developers/weaviate)
- [Python Client Docs](https://weaviate.io/developers/weaviate/client-libraries/python)
- [GitHub Repository](https://github.com/weaviate/weaviate)
- [Weaviate Cloud](https://console.weaviate.cloud)
- [Community Forum](https://forum.weaviate.io)

---

**Last Updated:** 2025-01
**Next Guide:** Pinecone Vector Database Guide
