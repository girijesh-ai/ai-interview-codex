# Technical Cheat Sheet - Cisco ML/AI Interview

## 1. Transformer Architecture Quick Reference

### Core Components
```
Input -> Embedding -> Positional Encoding ->
Encoder/Decoder Blocks -> Output Layer
```

**Key Mechanisms:**
- **Self-Attention**: Q, K, V matrices; Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Multi-Head Attention**: Multiple attention heads in parallel
- **Feed-Forward Network**: Two linear transformations with ReLU
- **Layer Normalization**: Normalize across features
- **Residual Connections**: Skip connections around each sub-layer

### Popular Architectures

**BERT (Encoder-only)**
- Bidirectional context
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Best for: Classification, NER, Q&A

**GPT (Decoder-only)**
- Autoregressive generation
- Causal masking (can't see future)
- Best for: Text generation, completion

**T5 (Encoder-Decoder)**
- Text-to-text framework
- All tasks as text generation
- Best for: Translation, summarization, Q&A

**LLaMA/Mistral (Decoder-only)**
- RoPE (Rotary Position Embedding)
- Grouped-Query Attention
- SwiGLU activation
- Best for: General purpose, efficient inference

## 2. Fine-tuning Techniques

### Full Fine-tuning
- **Pros**: Best performance, full adaptation
- **Cons**: Expensive, requires large dataset, storage intensive
- **When to use**: Critical applications, have resources and data

### PEFT (Parameter-Efficient Fine-Tuning)

**LoRA (Low-Rank Adaptation)**
- Adds trainable low-rank matrices to attention layers
- Trains ~0.1-1% of parameters
- Formula: W' = W + BA (where B and A are low-rank)
- **Pros**: Fast, memory efficient, modular
- **Cons**: Slight performance drop vs full fine-tuning

**QLoRA (Quantized LoRA)**
- LoRA + 4-bit quantization
- Enables fine-tuning on consumer GPUs
- **Pros**: Extremely memory efficient
- **Cons**: Additional quantization overhead

**Adapter Layers**
- Small bottleneck layers inserted between transformer blocks
- Freeze original model, train adapters
- **Pros**: Multiple task-specific adapters
- **Cons**: Adds inference latency

**Prefix Tuning**
- Learns continuous task-specific vectors
- Prepended to input at each layer
- **Pros**: No architectural changes
- **Cons**: Limited capacity

### RLHF (Reinforcement Learning from Human Feedback)
```
1. Supervised Fine-tuning (SFT)
2. Reward Model Training
3. PPO Optimization
```
- **Use for**: Alignment, safety, instruction following
- **Challenges**: Expensive annotation, reward hacking

## 3. RAG System Design

### Architecture Components

```
User Query
    ↓
Query Processing & Embedding
    ↓
Vector Database Retrieval (top-k)
    ↓
Context Ranking & Filtering
    ↓
Prompt Construction (Query + Context)
    ↓
LLM Generation
    ↓
Response Post-processing
    ↓
User Response
```

### Key Design Decisions

**Chunking Strategy:**
- Fixed-size chunks (256-512 tokens)
- Semantic chunks (paragraph/section based)
- Sliding window with overlap
- **Trade-off**: Smaller chunks = better retrieval, less context

**Embedding Model:**
- Sentence-BERT, OpenAI embeddings
- Domain-specific fine-tuned embeddings
- Dimension: 384-1536
- **Trade-off**: Size vs quality vs speed

**Vector Database:**
- **FAISS**: Fast, local, no server needed
- **Pinecone**: Managed, scalable, real-time updates
- **Weaviate**: Open-source, rich filtering
- **Milvus**: High performance, Kubernetes native

**Retrieval Strategy:**
- Top-k cosine similarity (k=3-10)
- Hybrid search (dense + sparse/BM25)
- Reranking with cross-encoder
- Metadata filtering

**Prompt Engineering:**
```
System: You are a helpful assistant. Use the context below.

Context: {retrieved_chunks}

Question: {user_query}

Instructions: Answer based on context. If unsure, say so.
```

### RAG Optimization Techniques

**Retrieval Optimization:**
- Multi-query retrieval (generate multiple queries)
- Hypothetical document embeddings (HyDE)
- Parent-child chunking (retrieve small, pass large)
- Ensemble retrieval (multiple methods)

**Generation Optimization:**
- Prompt compression
- Selective context (only relevant parts)
- Chain-of-thought prompting
- Citation tracking

**Evaluation Metrics:**
- Retrieval: Precision@k, Recall@k, MRR, NDCG
- Generation: BLEU, ROUGE, Faithfulness, Relevance
- End-to-end: Answer correctness, latency, cost

## 4. Model Evaluation Metrics

### Pre-training Metrics
- **Perplexity**: Lower is better; exp(cross-entropy loss)
- **Loss curves**: Training vs validation

### Fine-tuning Metrics
- **Accuracy**: Classification tasks
- **F1 Score**: Imbalanced datasets
- **BLEU/ROUGE**: Generation quality
- **Exact Match**: Q&A tasks

### LLM-specific Metrics
- **MMLU**: Multi-task language understanding
- **HumanEval**: Code generation
- **TruthfulQA**: Truthfulness
- **HellaSwag**: Common sense reasoning

### Production Metrics
- **Latency**: p50, p95, p99
- **Throughput**: Requests/second
- **Cost per 1K tokens**
- **User satisfaction**: Thumbs up/down
- **Hallucination rate**: Manual/automated checks

## 5. Production ML Best Practices

### MLOps Pipeline

```
Data Collection → Data Processing → Feature Engineering →
Model Training → Model Evaluation → Model Registry →
Model Serving → Monitoring → Retraining
```

### Key Components

**Data Pipeline:**
- ETL/ELT processes
- Data validation (Great Expectations)
- Feature store (Feast, Tecton)
- Version control (DVC, LakeFS)

**Training Pipeline:**
- Experiment tracking (MLflow, W&B)
- Hyperparameter tuning (Optuna, Ray Tune)
- Distributed training (DeepSpeed, FSDP)
- Checkpointing and recovery

**Model Registry:**
- Version control
- Model metadata
- A/B testing support
- Rollback capability

**Serving Infrastructure:**
- **Batch inference**: Large-scale processing
- **Real-time inference**: Low latency (<100ms)
- **Streaming inference**: Continuous processing

**Monitoring:**
- Model drift detection
- Data drift detection
- Performance metrics
- Resource utilization
- Error tracking

### Deployment Strategies

**Blue-Green Deployment:**
- Two identical environments
- Instant rollback
- Zero downtime

**Canary Deployment:**
- Gradual rollout (1% → 10% → 50% → 100%)
- Monitor metrics
- Automated rollback on issues

**Shadow Deployment:**
- Run new model in parallel
- Don't serve predictions
- Compare with production

## 6. LLM Inference Optimization

### Quantization
- **FP16**: 2x memory reduction, minimal quality loss
- **INT8**: 4x reduction, slight quality loss
- **INT4**: 8x reduction, noticeable quality loss
- **GPTQ/AWQ**: Advanced quantization methods

### KV-Cache Optimization
- Cache key-value pairs during generation
- Reduces computation for long sequences
- **Trade-off**: Memory vs compute

### Batching Strategies
- **Static batching**: Fixed batch size
- **Dynamic batching**: Combine requests
- **Continuous batching**: Iteration-level batching (vLLM)

### Model Parallelism
- **Tensor parallelism**: Split layers across GPUs
- **Pipeline parallelism**: Different layers on different GPUs
- **Data parallelism**: Replicate model, split data

### Serving Frameworks
- **vLLM**: High-throughput, PagedAttention
- **TGI (Text Generation Inference)**: Hugging Face's solution
- **TensorRT-LLM**: NVIDIA's optimized inference
- **Ray Serve**: Scalable model serving

## 7. Agentic Frameworks (Bonus)

### Agent Architecture Patterns

**ReAct (Reasoning + Acting)**
```
Thought: I need to search for information
Action: search("query")
Observation: Results...
Thought: Based on results, I should...
Action: Final answer
```

**Plan-and-Execute**
```
1. Plan: Break down into steps
2. Execute: Run each step
3. Replan: Adjust based on results
```

### Key Components
- **Tools**: Functions agent can call (search, calculator, API)
- **Memory**: Short-term (conversation) + Long-term (vector DB)
- **Planning**: Chain-of-thought, tree-of-thought
- **Execution**: Tool calling, result parsing

### Popular Frameworks
- **LangChain**: Comprehensive, many integrations
- **LlamaIndex**: Data-focused, great for RAG
- **AutoGPT**: Autonomous agents
- **BabyAGI**: Task-driven autonomous agent

## 8. Cybersecurity ML Applications

### Threat Detection
- Anomaly detection in network traffic
- Malware classification
- Phishing email detection
- Log analysis and SIEM

### LLM for Security
- Security policy Q&A
- Threat intelligence summarization
- Vulnerability explanation
- Automated incident response
- Code vulnerability detection

### Key Challenges
- Adversarial attacks
- False positive rate
- Real-time processing
- Explainability requirements
- Privacy and compliance

## Quick Decision Framework

### RAG vs Fine-tuning?
- **Need external data** → RAG
- **Change behavior/style** → Fine-tuning
- **Both** → Combine them

### Which fine-tuning method?
- **Best quality, have resources** → Full fine-tuning
- **Limited resources** → LoRA
- **Very limited GPU** → QLoRA
- **Multiple tasks** → Adapter layers

### Batch vs Real-time inference?
- **Latency sensitive (<1s)** → Real-time
- **Process large data** → Batch
- **Cost sensitive** → Batch
- **User-facing** → Real-time

### How to reduce latency?
1. Model quantization
2. Smaller model (distillation)
3. Caching common queries
4. Batch processing
5. Better hardware (GPU/TPU)
6. Model parallelism

### How to reduce cost?
1. Use smaller models when possible
2. Implement model tiering
3. Cache responses
4. Optimize prompt length
5. Use spot instances
6. Quantization
