# LLM System Design Examples for Cisco Interview

## Design Template - Use this structure for any system design question

### 1. Requirements Gathering (5 minutes)
- Functional requirements
- Non-functional requirements (scale, latency, cost)
- Constraints and assumptions

### 2. High-Level Architecture (10 minutes)
- Draw component diagram
- Explain data flow
- Identify key technologies

### 3. Deep Dive (15 minutes)
- Pick 2-3 components to detail
- Discuss trade-offs
- Address scalability and reliability

### 4. Evaluation & Monitoring (5 minutes)
- Metrics to track
- How to measure success
- Failure modes and mitigations

---

# Example 1: LLM-based Cybersecurity Threat Detection System

## Problem Statement
Design a system that uses LLMs to analyze security logs, network traffic, and threat intelligence to detect and explain potential security threats in real-time for enterprise networks.

## 1. Requirements Gathering

### Functional Requirements
- Ingest security logs from multiple sources (firewalls, IDS/IPS, endpoints)
- Analyze logs in near real-time (< 1 minute latency)
- Detect anomalies and potential threats
- Generate natural language explanations of threats
- Provide recommendations for mitigation
- Support queries like "What are the current threats?" or "Explain this alert"

### Non-Functional Requirements
- **Scale**: 10M logs/day, 1000 concurrent users
- **Latency**: < 1 second for queries, < 1 minute for log processing
- **Availability**: 99.9% uptime
- **Cost**: Optimize for cost-effective LLM usage
- **Security**: On-premise deployment option, data encryption
- **Compliance**: SOC 2, HIPAA compatible

### Assumptions
- Logs are already collected by existing SIEM
- Historical data available for training
- Network traffic is captured by sensors

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Firewall │  │   IDS    │  │ Endpoint │  │  Network │       │
│  │   Logs   │  │   Logs   │  │   Logs   │  │  Traffic │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └──────────────┴──────────────┴─────────────┘             │
│                            │                                     │
│                      Kafka Streams                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    Processing Pipeline                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Log        │    │  Feature     │    │   Anomaly    │      │
│  │ Normalization│───▶│ Extraction   │───▶│  Detection   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                   │              │
└───────────────────────────────────────────────────┼──────────────┘
                                                    │
┌───────────────────────────────────────────────────┼──────────────┐
│                     LLM Layer                     │              │
│  ┌────────────────────────────────────────────────▼────────┐    │
│  │                 Threat Analysis                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐         │    │
│  │  │   RAG    │  │   LLM    │  │  Post-       │         │    │
│  │  │  Retrieval│─▶│ (GPT-4/ │─▶│ Processing   │         │    │
│  │  │          │  │  Llama)  │  │              │         │    │
│  │  └────┬─────┘  └──────────┘  └──────────────┘         │    │
│  │       │                                                 │    │
│  │  ┌────▼─────────────────┐                             │    │
│  │  │  Vector DB (Pinecone) │                             │    │
│  │  │  - Historical threats │                             │    │
│  │  │  - Threat intel feeds │                             │    │
│  │  │  - Playbooks         │                             │    │
│  │  └──────────────────────┘                             │    │
│  └─────────────────────────┬─────────────────────────────┘    │
└────────────────────────────┼──────────────────────────────────┘
                             │
┌────────────────────────────┴──────────────────────────────────┐
│                     Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Dashboard   │  │   REST API   │  │   Alerts &   │        │
│  │  (React)     │  │              │  │ Notifications│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

## 3. Deep Dive into Key Components

### 3.1 Data Ingestion & Processing Pipeline

**Architecture:**
```python
# Kafka Consumer for real-time log ingestion
- Topics: firewall_logs, ids_logs, endpoint_logs
- Consumer groups for parallel processing
- Schema registry for data validation
```

**Log Normalization:**
```python
class LogNormalizer:
    """
    Normalize logs from different sources into common format
    """
    def normalize(self, raw_log):
        return {
            'timestamp': self.extract_timestamp(raw_log),
            'source_ip': self.extract_ip(raw_log, 'source'),
            'dest_ip': self.extract_ip(raw_log, 'dest'),
            'event_type': self.classify_event(raw_log),
            'severity': self.map_severity(raw_log),
            'raw_message': raw_log['message']
        }
```

**Feature Extraction:**
- Statistical features: connection frequency, data volume, unusual ports
- Temporal features: time of day, day of week patterns
- Behavioral features: deviation from baseline
- Network graph features: node centrality, community detection

**Anomaly Detection (Traditional ML):**
- **Why**: Filter obvious non-threats before expensive LLM calls
- **Models**: Isolation Forest, LSTM autoencoders
- **Output**: Anomaly score (0-1), if > 0.7 → send to LLM

**Trade-offs:**
- Use lightweight ML for initial filtering (90% of logs filtered)
- Only send suspicious events to LLM (cost optimization)
- Balance: false negatives vs LLM cost

### 3.2 RAG System for Threat Intelligence

**Vector Database Schema:**
```python
{
  'id': 'threat_12345',
  'embedding': [0.1, 0.2, ...],  # 1536-dim
  'metadata': {
    'threat_type': 'malware',
    'cvss_score': 9.8,
    'indicators': ['IP: 1.2.3.4', 'Hash: abc123'],
    'mitre_attack_id': 'T1566',
    'description': '...',
    'mitigation': '...',
    'last_updated': '2024-10-13'
  }
}
```

**Knowledge Sources:**
1. **Threat Intelligence Feeds**
   - MITRE ATT&CK framework
   - CVE database
   - Vendor threat feeds
   - Internal historical incidents

2. **Playbooks & Documentation**
   - Incident response procedures
   - Security policies
   - Past incident reports

3. **Chunking Strategy**
   - Semantic chunking by threat/CVE
   - Chunk size: 512 tokens
   - Metadata enrichment for filtering

**Retrieval Strategy:**
```python
def retrieve_context(anomaly_log):
    # Generate query embedding
    query = f"""
    Security event: {anomaly_log['event_type']}
    Source: {anomaly_log['source_ip']}
    Destination: {anomaly_log['dest_ip']}
    Behavior: {anomaly_log['description']}
    """

    query_embedding = embedding_model.encode(query)

    # Retrieve top-k similar threats
    results = vector_db.query(
        vector=query_embedding,
        filter={
            'metadata.severity': {'$gte': 'medium'},
            'metadata.last_updated': {'$gte': '2024-01-01'}
        },
        top_k=5
    )

    return results
```

### 3.3 LLM Inference & Optimization

**Model Selection Strategy:**
- **Tier 1 (Simple)**: Fine-tuned smaller model (Llama-7B) for standard threats
- **Tier 2 (Complex)**: Large model (GPT-4) for novel/complex threats
- **Router**: Classify complexity before routing

**Prompt Engineering:**
```python
SYSTEM_PROMPT = """
You are a cybersecurity expert analyzing security events.
Provide clear, actionable explanations suitable for SOC analysts.

Output Format:
1. Threat Assessment (Low/Medium/High/Critical)
2. Description (2-3 sentences)
3. Potential Impact
4. Recommended Actions
5. MITRE ATT&CK Mapping
"""

def build_prompt(anomaly_log, retrieved_context):
    context_str = "\n".join([
        f"Similar Threat {i+1}: {ctx['description']}\n"
        f"Mitigation: {ctx['mitigation']}"
        for i, ctx in enumerate(retrieved_context)
    ])

    return f"""
{SYSTEM_PROMPT}

## Historical Context:
{context_str}

## Current Event:
Timestamp: {anomaly_log['timestamp']}
Source IP: {anomaly_log['source_ip']}
Destination IP: {anomaly_log['dest_ip']}
Event Type: {anomaly_log['event_type']}
Anomaly Score: {anomaly_log['anomaly_score']}
Details: {anomaly_log['raw_message']}

Analyze this security event and provide assessment.
"""
```

**Optimization Techniques:**

1. **Caching**
```python
# Cache LLM responses for similar events
cache_key = hash(f"{event_type}_{source_ip}_{dest_ip}_{similarity_threshold}")
if cache_key in redis_cache and age < 1_hour:
    return cached_response
```

2. **Batching**
```python
# Batch low-priority events (every 5 minutes)
# Process high-priority events immediately
if severity >= 'high':
    process_immediately()
else:
    add_to_batch_queue()
```

3. **Model Quantization**
- Use INT8 quantization for on-premise deployment
- Reduces memory by 4x, minimal accuracy loss

### 3.4 Scalability & Performance

**Horizontal Scaling:**
- Kafka partitioning by source (firewall, IDS, endpoint)
- Multiple consumer groups for parallel processing
- LLM serving: vLLM with continuous batching
- Auto-scaling based on queue depth

**Performance Targets:**
```
Log Ingestion: 10K logs/second
Anomaly Detection: < 100ms per log
LLM Inference: < 2 seconds per query
End-to-end: < 1 minute (95th percentile)
```

**Cost Optimization:**
```
Filtering:       10M logs/day
After ML:         1M logs/day (90% filtered)
After rules:      100K logs/day (99% filtered)
LLM calls:        100K/day
Cost/call:        $0.01 (using Llama-70B)
Daily cost:       $1,000
Monthly cost:     $30,000
```

## 4. Monitoring & Evaluation

### Metrics to Track

**System Metrics:**
- Ingestion lag (Kafka consumer lag)
- Processing latency (p50, p95, p99)
- LLM inference time
- Cache hit rate
- Error rates
- Cost per threat detected

**ML Metrics:**
- Anomaly detection: Precision, Recall, F1
- False positive rate (critical metric!)
- Alert fatigue score
- Mean time to detect (MTTD)
- Mean time to respond (MTTR)

**Business Metrics:**
- Threats detected
- False alarms reduced
- Analyst productivity improvement
- Cost savings vs manual analysis

### Evaluation Framework

**Offline Evaluation:**
```python
# Use historical labeled data
test_set = load_labeled_threats()

for event in test_set:
    prediction = system.analyze(event)
    ground_truth = event['true_label']

    evaluate(prediction, ground_truth)

# Metrics
- Threat detection rate
- False positive rate
- Explanation quality (human eval)
```

**Online Evaluation:**
- A/B testing: New model vs baseline
- Analyst feedback: thumbs up/down
- Shadow deployment: Compare with production

**Human-in-the-Loop:**
- Analyst reviews LLM explanations
- Feedback loop for continuous improvement
- Weekly review of false positives/negatives

### Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation |
|-------------|--------|------------|
| LLM service down | No threat explanations | Fallback to rule-based alerts |
| Vector DB failure | No context retrieval | Use cached embeddings |
| Kafka lag spike | Delayed detection | Auto-scale consumers, alerts |
| High false positives | Alert fatigue | Adjust ML thresholds, human review |
| Model drift | Degraded performance | Monitor metrics, scheduled retraining |

## 5. Technology Stack

```python
# Data Ingestion
- Apache Kafka
- Schema Registry
- Kafka Connect

# Processing
- Apache Flink / Spark Streaming
- Python (Pandas, scikit-learn)

# ML/LLM
- PyTorch / TensorFlow
- Hugging Face Transformers
- vLLM or TGI for serving
- Llama-70B or GPT-4

# Vector Database
- Pinecone (managed)
- or Weaviate (self-hosted)

# Storage
- PostgreSQL (metadata)
- S3 / MinIO (logs, models)
- Redis (caching)

# Monitoring
- Prometheus + Grafana
- ELK Stack (logs)
- MLflow (experiment tracking)

# Infrastructure
- Kubernetes
- Docker
- Terraform
```

---

# Example 2: Enterprise Semantic Search for Network Documentation

## Problem Statement
Design an LLM-powered semantic search system for Cisco's network documentation, configuration guides, and troubleshooting knowledge base. Support natural language queries from network engineers.

## 1. Requirements

### Functional
- Semantic search across 100K+ documents
- Support queries like "How do I configure BGP on ASR router?"
- Provide code examples and configuration snippets
- Multi-modal search (text + diagrams)
- Conversational follow-ups

### Non-Functional
- **Latency**: < 500ms for search
- **Scale**: 10K engineers, 1M queries/month
- **Accuracy**: > 90% relevance
- **Freshness**: New docs indexed within 1 hour

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Document Sources                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  Cisco  │  │   Wiki  │  │ Support │  │  GitHub │       │
│  │  Docs   │  │         │  │  Cases  │  │  Repos  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
└───────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │
┌───────┴────────────┴────────────┴────────────┴─────────────┐
│                  Ingestion Pipeline                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Document │  │ Chunking │  │ Metadata │  │Embedding │   │
│  │ Parsing  │─▶│  & OCR   │─▶│Extraction│─▶│Generation│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────┬───┘   │
└──────────────────────────────────────────────────────┼──────┘
                                                       │
┌──────────────────────────────────────────────────────┼──────┐
│                    Storage Layer                     │      │
│  ┌─────────────────────────────────────────────────┐│      │
│  │         Vector Database (Weaviate)              ││      │
│  │  - Document embeddings (1536-dim)              ││      │
│  │  - Metadata index                               ││      │
│  │  - Hybrid search (dense + sparse)               ││      │
│  └─────────────────────────────────────────────────┘│      │
│  ┌─────────────────────────────────────────────────┐│      │
│  │         Document Store (PostgreSQL)             ││      │
│  │  - Full document content                        ││      │
│  │  - Version history                              ││      │
│  └─────────────────────────────────────────────────┘│      │
└──────────────────────────────────────────────────────┼──────┘
                                                       │
┌──────────────────────────────────────────────────────┼──────┐
│                    Query Processing                  │      │
│  User Query ───▶ Query Understanding ───▶ Retrieval ─┘      │
│                  (Query expansion,        (Hybrid search)    │
│                   Intent classification)                     │
│                                                              │
│  Retrieved Docs ───▶ Reranking ───▶ LLM Generation          │
│                     (Cross-encoder)  (GPT-4/Llama)           │
│                                            │                 │
└────────────────────────────────────────────┼─────────────────┘
                                             │
                                          Response
```

## 3. Deep Dive

### 3.1 Document Processing Pipeline

**Chunking Strategy:**
```python
class DocumentChunker:
    def chunk_document(self, doc):
        # Hierarchical chunking
        chunks = []

        # 1. Split by sections
        sections = self.split_by_headers(doc)

        for section in sections:
            # 2. Split long sections
            if len(section) > 512:
                sub_chunks = self.semantic_split(section, max_length=512)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)

        # 3. Create parent-child relationships
        for chunk in chunks:
            chunk['parent_section'] = section.title
            chunk['document_id'] = doc.id

        return chunks

    def semantic_split(self, text, max_length=512):
        # Split at sentence boundaries to preserve meaning
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_length = len(sent.split())
            if current_length + sent_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_length = sent_length
            else:
                current_chunk.append(sent)
                current_length += sent_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
```

**Metadata Extraction:**
```python
{
  'chunk_id': 'doc123_chunk_5',
  'document_id': 'doc123',
  'title': 'BGP Configuration Guide',
  'section': 'Basic BGP Setup',
  'product': 'ASR 9000',
  'version': 'IOS XR 7.5',
  'doc_type': 'configuration_guide',
  'keywords': ['bgp', 'routing', 'autonomous-system'],
  'code_snippets': ['router bgp 65000'],
  'last_updated': '2024-09-15',
  'chunk_text': '...',
  'embedding': [0.1, 0.2, ...]
}
```

### 3.2 Hybrid Search Strategy

**Why Hybrid?**
- Dense (semantic): Understands meaning, handles synonyms
- Sparse (BM25): Exact keyword match, better for technical terms

**Implementation:**
```python
def hybrid_search(query, alpha=0.7):
    """
    Combine dense and sparse search

    alpha: weight for dense search (0-1)
    1-alpha: weight for sparse search
    """
    # Dense search (semantic)
    query_embedding = embedding_model.encode(query)
    dense_results = vector_db.search(
        vector=query_embedding,
        limit=20
    )

    # Sparse search (BM25)
    sparse_results = bm25_index.search(
        query=query,
        limit=20
    )

    # Fusion: Reciprocal Rank Fusion
    combined_scores = {}
    for rank, doc in enumerate(dense_results):
        combined_scores[doc.id] = combined_scores.get(doc.id, 0) + \
                                  alpha / (rank + 1)

    for rank, doc in enumerate(sparse_results):
        combined_scores[doc.id] = combined_scores.get(doc.id, 0) + \
                                  (1 - alpha) / (rank + 1)

    # Sort by combined score
    ranked_docs = sorted(combined_scores.items(),
                        key=lambda x: x[1],
                        reverse=True)

    return ranked_docs[:10]
```

### 3.3 Reranking with Cross-Encoder

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank(query, initial_results, top_k=5):
    """
    Rerank top results using cross-encoder for better relevance
    """
    # Create query-document pairs
    pairs = [[query, doc['text']] for doc in initial_results]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)

    # Sort by score
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [initial_results[i] for i in ranked_indices]
```

### 3.4 LLM Generation with Citations

**Prompt:**
```python
def build_answer_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[{i+1}] {doc['title']} (Product: {doc['product']})\n{doc['text']}"
        for i, doc in enumerate(retrieved_docs)
    ])

    return f"""
You are a Cisco network engineering assistant. Answer the question using ONLY the provided documentation.

Question: {query}

Documentation:
{context}

Instructions:
1. Provide a clear, step-by-step answer
2. Include relevant configuration commands
3. Cite sources using [1], [2], etc.
4. If the documentation doesn't contain the answer, say so
5. Include warnings about common pitfalls

Answer:
"""
```

**Response with Citations:**
```
To configure BGP on an ASR 9000 router [1]:

1. Enter global configuration mode:
   router bgp 65000

2. Configure the router ID:
   bgp router-id 1.1.1.1

3. Add neighbor:
   neighbor 192.168.1.1 remote-as 65001

⚠️ Important: Ensure your autonomous system number matches your network design [2].

Sources:
[1] BGP Configuration Guide for ASR 9000 - Section 2.3
[2] ASR 9000 Best Practices - BGP Design Considerations
```

## 4. Evaluation & Monitoring

**Search Quality Metrics:**
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG@k**: Ranking quality of top-k results
- **Precision@k**: Relevant docs in top-k
- **User satisfaction**: Click-through rate, dwell time

**LLM Generation Metrics:**
- **Faithfulness**: Answer grounded in retrieved docs
- **Citation accuracy**: Correct source attribution
- **Completeness**: Covers all aspects of query
- **User feedback**: Thumbs up/down

---

# Example 3: AI-Powered Network Anomaly Detection & Root Cause Analysis

## Problem Statement
Design a system that uses LLMs to detect network anomalies from telemetry data and provide root cause analysis with natural language explanations.

## 1. Architecture Overview

```
Network Devices → Telemetry Collectors → Time-Series DB (InfluxDB)
                                              ↓
                                        Anomaly Detection
                                        (Prophet, LSTM)
                                              ↓
                                        Anomaly Events
                                              ↓
                                   ┌──────────┴──────────┐
                                   │                     │
                              RAG System              LLM Agent
                           (Historical data)      (Root cause analysis)
                                   │                     │
                                   └──────────┬──────────┘
                                              ↓
                                         Explanation
                                    (Natural language RCA)
```

## Key Components

### 1. Time-Series Anomaly Detection
```python
# Multi-variate time-series analysis
features = [
    'cpu_utilization',
    'memory_usage',
    'bandwidth_in',
    'bandwidth_out',
    'packet_loss',
    'latency',
    'error_rate'
]

# LSTM Autoencoder for anomaly detection
model = LSTMAutoencoder(
    input_dim=len(features),
    hidden_dim=64,
    num_layers=2
)

# Detect anomalies based on reconstruction error
reconstruction_error = model.predict(data)
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold
```

### 2. Root Cause Analysis Agent

```python
class RCAAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {
            'query_metrics': self.query_time_series,
            'get_topology': self.get_network_topology,
            'check_recent_changes': self.get_change_log,
            'search_similar_incidents': self.search_knowledge_base
        }

    def analyze(self, anomaly):
        # Agent workflow
        context = self.gather_context(anomaly)
        hypothesis = self.generate_hypothesis(context)
        root_cause = self.validate_hypothesis(hypothesis)
        explanation = self.generate_explanation(root_cause)

        return explanation

    def gather_context(self, anomaly):
        # Use LLM to decide which tools to call
        plan = self.llm.generate(f"""
        Anomaly detected: {anomaly}
        Available tools: {list(self.tools.keys())}
        Which information do you need to diagnose this?
        """)

        # Execute tools
        context = {}
        for tool in plan['tools_to_use']:
            context[tool] = self.tools[tool](anomaly)

        return context
```

### 3. Explanation Generation

```python
def generate_explanation(root_cause, context):
    prompt = f"""
    Generate a root cause analysis report for network operators.

    Anomaly: {context['anomaly']}
    Root Cause: {root_cause}
    Related Metrics: {context['metrics']}
    Network Topology: {context['topology']}
    Recent Changes: {context['changes']}

    Provide:
    1. Executive Summary (2-3 sentences)
    2. Detailed Analysis
    3. Impact Assessment
    4. Recommended Actions
    5. Prevention Measures
    """

    return llm.generate(prompt)
```

**Output Example:**
```
Executive Summary:
High packet loss detected on router ASR9K-CORE-1 (15% loss) caused by
interface congestion on GigabitEthernet0/0/1. Traffic spike from new
application deployment exceeded link capacity.

Detailed Analysis:
At 14:32 UTC, automated monitoring detected packet loss increasing from
baseline 0.1% to 15% on ASR9K-CORE-1. Root cause analysis revealed:
- Interface Gi0/0/1 bandwidth utilization: 98% (normal: 45%)
- Correlated with new microservice deployment at 14:30 UTC
- Traffic pattern shows sustained 800Mbps increase

Impact Assessment:
- 500 users experiencing degraded performance
- Mission-critical application latency: 250ms (SLA: 100ms)
- Estimated revenue impact: $X/hour

Recommended Actions:
1. IMMEDIATE: Enable QoS policies to prioritize critical traffic
2. SHORT-TERM: Upgrade link to 10G (ETA: 2 hours)
3. LONG-TERM: Implement traffic shaping for new application

Prevention Measures:
- Pre-deployment capacity planning
- Automated traffic throttling for new services
- Enhanced monitoring thresholds
```

---

## Questions to Ask Interviewers

When interviewer asks "Do you have any questions for me?", use these:

### For Technical Rounds (Ranjan, Maithri)

1. "What are the biggest technical challenges the AI Platform team is currently facing?"

2. "How do you balance research innovation with product delivery timelines?"

3. "What's your approach to evaluating and adopting new LLM technologies?"

4. "Can you describe a recent ML system that didn't work out as expected and what you learned?"

5. "How does the team handle the trade-off between model performance and production constraints?"

### For Hiring Manager (Disha)

1. "What does success look like for this role in the first 6 months and first year?"

2. "How do you see the AI Platform team evolving in the next 2-3 years?"

3. "What are the key skills or experiences you're looking for that would make someone exceptionally successful in this role?"

4. "How do you foster innovation while maintaining production reliability?"

### For Leadership Round (Sangeeta)

1. "How does Cisco support professional development and growth for engineering managers?"

2. "What's the culture like within the team? How do you encourage collaboration between research and engineering?"

3. "How does the organization balance AI innovation with Cisco's core networking/security business?"

