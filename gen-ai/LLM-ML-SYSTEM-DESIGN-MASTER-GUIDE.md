# LLM & ML System Design - Complete Master Guide

## Cisco Engineering Manager Interview Preparation

> **Comprehensive guide covering all LLM and ML system design topics for Staff+ level interviews**

---

## Table of Contents

1. [Overview of Materials](#overview-of-materials)
2. [Interview Format](#interview-format)
3. [System Design Framework](#system-design-framework)
4. [Core Topics Covered](#core-topics-covered)
5. [Resources Summary](#resources-summary)
6. [Study Timeline](#study-timeline)
7. [Common System Design Questions](#common-system-design-questions)
8. [Evaluation Criteria](#evaluation-criteria)

---

## Overview of Materials

You have **7 comprehensive documents** covering all aspects of LLM and ML system design:

### ✅ **1. System Design Examples (32KB)**
**File:** `system-design-examples.md`

**What's Covered:**
- 3 Complete system design examples with full architecture
- LLM-based Cybersecurity Threat Detection System
- Enterprise Semantic Search for Network Documentation
- AI-Powered Network Anomaly Detection & Root Cause Analysis
- End-to-end designs with code examples

**When to Use:**
- Practicing full system design interview questions
- Understanding how to structure system design responses
- Learning LLM + traditional ML hybrid architectures

---

### ✅ **2. LLM Production Complete Guide (40KB)**
**File:** `llm-production-complete-guide.md`

**What's Covered:**
- **LLM Inference Optimization**
  - vLLM vs TGI (23x throughput improvement)
  - Continuous batching
  - PagedAttention (80% KV cache reduction)
  - Quantization (AWQ, GPTQ, GGUF)
  - Flash Attention (2-4x speedup)
  - Speculative decoding

- **Prompt Engineering Mastery**
  - Chain of Thought (CoT)
  - Tree of Thoughts (ToT)
  - ReAct (Reasoning + Acting)
  - Self-Consistency
  - Prompt optimization frameworks

- **LLM Safety & Security**
  - Prompt injection detection
  - Jailbreaking defenses
  - Content moderation
  - Multi-layer safety systems

- **Distributed Training**
  - Data Parallelism (DP)
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
  - FSDP, DeepSpeed ZeRO

**When to Use:**
- LLM-specific interview questions
- Production optimization discussions
- Safety and security scenarios
- Training large models (70B+)

---

### ✅ **3. MLOps & Production ML Guide (70KB)**
**File:** `mlops-production-ml-guide.md`

**What's Covered:**
- **Model Serving Patterns**
  - TorchServe, TensorFlow Serving, vLLM
  - A/B testing, Canary deployment, Shadow deployment
  - Multi-model serving, Ensemble patterns
  - Latency < 100ms, throughput > 10K req/s

- **Model Monitoring & Drift Detection**
  - Data drift (PSI, KS test, KL divergence)
  - Concept drift
  - Evidently AI, Fiddler AI
  - Automated retraining triggers

- **Feature Stores**
  - Feast (open-source)
  - Tecton (managed)
  - Point-in-time correctness
  - Online vs offline stores

- **CI/CD for ML**
  - GitHub Actions workflows
  - Data validation (Great Expectations)
  - Model evaluation pipelines
  - Gradual rollout strategies

- **Experiment Tracking**
  - MLflow vs Weights & Biases
  - Hyperparameter tuning
  - Model registry

**When to Use:**
- MLOps architecture discussions
- Production ML system design
- Model deployment strategies
- Monitoring and maintenance questions

---

### ✅ **4. Production RAG Systems Guide (67KB)**
**File:** `production-rag-systems-guide.md`

**What's Covered:**
- **RAG Architecture Patterns**
  - Naive RAG
  - Advanced RAG (query rewriting, hybrid search)
  - Agentic RAG (adaptive retrieval)

- **Chunking Strategies**
  - Fixed-size, semantic, hierarchical
  - Optimal chunk sizes for different use cases
  - Parent-child document relationships

- **Retrieval Optimization**
  - Hybrid search (dense + sparse)
  - Re-ranking with cross-encoders
  - Query expansion and decomposition

- **Evaluation Frameworks**
  - RAGAS metrics (faithfulness, answer relevance)
  - Human evaluation protocols
  - A/B testing strategies

- **Production Best Practices**
  - Caching strategies
  - Fallback mechanisms
  - Cost optimization
  - Latency optimization

**When to Use:**
- RAG system design questions
- Semantic search architecture
- Document Q&A systems
- Knowledge base applications

---

### ✅ **5. Transformer Architecture Guide (41KB)**
**File:** `transformer-architecture-complete-guide.md`

**What's Covered:**
- **Complete Transformer Architecture**
  - Self-attention mechanism (detailed)
  - Multi-head attention
  - Positional encoding
  - Feed-forward networks
  - Layer normalization

- **Encoder-Decoder Architecture**
  - Original transformer (machine translation)
  - Encoder-only (BERT)
  - Decoder-only (GPT, LLaMA)

- **Training Techniques**
  - Pre-training objectives (MLM, CLM)
  - Fine-tuning strategies
  - Transfer learning

- **Implementation Details**
  - PyTorch code examples
  - Complexity analysis
  - Optimization techniques

**When to Use:**
- Deep technical discussions
- Architecture design questions
- When interviewer asks "Explain transformers"
- Model selection decisions

---

### ✅ **6. Attention Mechanisms Guide (43KB)**
**File:** `attention-mechanisms-comprehensive-guide.md`

**What's Covered:**
- **Attention Variants**
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Self-Attention vs Cross-Attention
  - Causal (Masked) Attention

- **Efficient Attention**
  - Flash Attention (exact, 2-4x faster)
  - Linear Attention (approximation)
  - Sparse Attention patterns
  - Sliding Window Attention

- **Long Context Handling**
  - RoPE (Rotary Position Embedding)
  - ALiBi (Attention with Linear Biases)
  - Context window extensions

- **Implementation from Scratch**
  - NumPy implementations
  - PyTorch implementations
  - Optimization techniques

**When to Use:**
- Detailed attention mechanism questions
- Performance optimization discussions
- Long context challenges
- Model architecture decisions

---

### ✅ **7. Embedding Models Guide (50KB)**
**File:** `embedding-models-comprehensive-guide.md`

**What's Covered:**
- **Embedding Model Architectures**
  - Word2Vec, GloVe (classic)
  - BERT, RoBERTa (contextual)
  - Sentence-BERT, MPNet (sentence encoders)
  - E5, BGE (state-of-art 2025)

- **Training Techniques**
  - Contrastive learning
  - In-batch negatives
  - Hard negative mining
  - Knowledge distillation

- **Evaluation Metrics**
  - MTEB benchmark
  - Retrieval metrics (MRR, NDCG, Recall@k)
  - Semantic similarity
  - Domain-specific evaluation

- **Production Deployment**
  - Model selection (OpenAI, Cohere, BGE)
  - Dimensionality reduction
  - Quantization for embeddings
  - Caching strategies

**When to Use:**
- Semantic search system design
- RAG retrieval component
- Recommendation systems
- Similarity search at scale

---

## Interview Format

### ML/LLM System Design Round Structure (60-75 minutes)

**Phase 1: Requirements Gathering (5-10 min)**
- Clarify functional requirements
- Understand non-functional requirements (scale, latency, cost)
- Define success metrics
- Identify constraints

**Phase 2: High-Level Design (10-15 min)**
- Draw component diagram
- Explain data flow
- Identify key technologies
- Discuss trade-offs

**Phase 3: Deep Dive (25-35 min)**
- Pick 2-3 components to detail
- Discuss algorithms and data structures
- Address scalability
- Handle failure modes
- Optimization strategies

**Phase 4: Evaluation & Monitoring (5-10 min)**
- Metrics to track
- How to measure success
- Failure scenarios and mitigations

**Phase 5: Q&A (5-10 min)**
- Answer clarifying questions
- Discuss alternative approaches
- Ask questions about the team/role

---

## System Design Framework

### Universal Template (Use for ANY System Design Question)

```
1. CLARIFY (5 minutes)
   - What? Functional requirements
   - How much? Scale (users, requests, data)
   - How fast? Latency requirements
   - How reliable? Availability, consistency
   - How much $? Cost constraints

2. HIGH-LEVEL (10 minutes)
   - Draw boxes: Components
   - Draw arrows: Data flow
   - Label: Technologies
   - Annotate: Why each choice

3. DEEP DIVE (30 minutes)
   Pick 2-3 components:
   - Data structures
   - Algorithms
   - Trade-offs
   - Alternatives
   - Code snippets (pseudo-code OK)

4. SCALE & OPTIMIZE (10 minutes)
   - Bottlenecks: Identify
   - Solutions: Cache, shard, replicate
   - Monitoring: What to track
   - Failure modes: How to handle

5. METRICS (5 minutes)
   - Business metrics
   - Technical metrics
   - How to evaluate
```

---

## Core Topics Covered

### 1. LLM Inference & Optimization

**Key Concepts:**
- vLLM continuous batching (23x throughput)
- PagedAttention (80% memory reduction)
- Quantization: GPTQ (4-bit), AWQ (4-bit), FP16
- Flash Attention (2-4x speedup)
- Speculative decoding (2-3x speedup)
- Model parallelism (TP, PP, DP)

**Interview Questions:**
- "Design a system to serve 100K LLM requests/day with <2s latency"
- "How would you optimize inference cost for a production LLM API?"
- "Explain continuous batching and when it's beneficial"

**Where to Find:**
- `llm-production-complete-guide.md` - Sections 1, 5
- `transformer-architecture-complete-guide.md` - Implementation details

---

### 2. Prompt Engineering & Agentic Systems

**Key Concepts:**
- Chain of Thought (40-60% accuracy improvement)
- Tree of Thoughts (explore multiple paths)
- ReAct (Reasoning + Acting with tools)
- Self-Consistency (majority voting)
- Anthropic's 6 agentic patterns

**Interview Questions:**
- "Design a research agent that answers complex questions"
- "How would you build a coding assistant that uses tools?"
- "Explain ReAct and when to use it vs Chain of Thought"

**Where to Find:**
- `llm-production-complete-guide.md` - Section 2
- `agentic-ai/` folder - All 7 notebooks
- `agentic-ai/01-anthropic-patterns-interactive.ipynb` - All 6 patterns

---

### 3. RAG System Design

**Key Concepts:**
- Chunking strategies (fixed, semantic, hierarchical)
- Hybrid search (dense + sparse)
- Re-ranking with cross-encoders
- Query expansion and decomposition
- Agentic RAG (adaptive retrieval)
- Evaluation (RAGAS, faithfulness, relevance)

**Interview Questions:**
- "Design a document Q&A system for 1M documents"
- "How would you handle multi-lingual semantic search?"
- "Design a RAG system for legal document analysis"

**Where to Find:**
- `production-rag-systems-guide.md` - Complete RAG guide
- `agentic-ai/06-agentic-rag-patterns.ipynb` - Agentic RAG
- `system-design-examples.md` - Example 2 (Semantic Search)

---

### 4. MLOps & Model Serving

**Key Concepts:**
- Serving frameworks (TorchServe, vLLM, Triton)
- Deployment patterns (A/B, Canary, Shadow)
- Monitoring (drift detection, performance)
- Feature stores (Feast, Tecton)
- CI/CD for ML (GitHub Actions)

**Interview Questions:**
- "Design a model serving system for 10K req/s with <50ms latency"
- "How would you detect and handle model drift?"
- "Explain your approach to gradual model rollout"

**Where to Find:**
- `mlops-production-ml-guide.md` - Complete MLOps guide
- `system-design-examples.md` - All examples include MLOps

---

### 5. LLM Safety & Security

**Key Concepts:**
- Prompt injection detection
- Jailbreaking prevention
- Content moderation (OpenAI API)
- PII detection and redaction
- Multi-layer defense (input/output validation)

**Interview Questions:**
- "How would you prevent prompt injection attacks?"
- "Design a content moderation pipeline for a chatbot"
- "Explain jailbreaking and how to defend against it"

**Where to Find:**
- `llm-production-complete-guide.md` - Section 3
- Code examples for injection detection

---

### 6. Distributed Training

**Key Concepts:**
- Data Parallelism (DDP)
- Tensor Parallelism (Megatron)
- Pipeline Parallelism (GPipe)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO (ZeRO-1/2/3)

**Interview Questions:**
- "How would you train a 70B parameter model?"
- "Explain FSDP and when to use it over DDP"
- "Design a training pipeline for continuous pre-training"

**Where to Find:**
- `llm-production-complete-guide.md` - Section 4
- `transformer-architecture-complete-guide.md` - Training section

---

## Resources Summary

### By File Type

| File | Size | Focus | Best For |
|------|------|-------|----------|
| **system-design-examples.md** | 32KB | End-to-end examples | Practice, Templates |
| **llm-production-complete-guide.md** | 40KB | LLM inference, safety | Production LLM systems |
| **mlops-production-ml-guide.md** | 70KB | Deployment, monitoring | MLOps architecture |
| **production-rag-systems-guide.md** | 67KB | RAG patterns | Semantic search, Q&A |
| **transformer-architecture-complete-guide.md** | 41KB | Deep technical | Architecture questions |
| **attention-mechanisms-comprehensive-guide.md** | 43KB | Attention details | Performance optimization |
| **embedding-models-comprehensive-guide.md** | 50KB | Embeddings | Retrieval systems |
| **agentic-ai/** | 7 notebooks | Agents, frameworks | Agentic systems |

### By Interview Topic

| Topic | Primary Resources | Secondary Resources |
|-------|------------------|---------------------|
| **LLM Inference** | llm-production-complete-guide.md | transformer-architecture-complete-guide.md |
| **RAG Systems** | production-rag-systems-guide.md, system-design-examples.md | embedding-models-comprehensive-guide.md |
| **MLOps** | mlops-production-ml-guide.md | system-design-examples.md |
| **Agentic AI** | agentic-ai/ folder (7 notebooks) | llm-production-complete-guide.md (prompt engineering) |
| **Model Training** | llm-production-complete-guide.md (distributed training) | transformer-architecture-complete-guide.md |
| **Safety** | llm-production-complete-guide.md (section 3) | - |

---

## Study Timeline

### Week 1-2: Foundation (Priority 1)

**Day 1-2: System Design Framework**
- [ ] Read `system-design-examples.md` - All 3 examples
- [ ] Practice drawing architectures
- [ ] Memorize the universal template

**Day 3-4: LLM Production Basics**
- [ ] Read `llm-production-complete-guide.md` - Sections 1-2
- [ ] Understand vLLM, continuous batching
- [ ] Learn prompt engineering patterns

**Day 5-6: RAG Fundamentals**
- [ ] Read `production-rag-systems-guide.md` - First 50%
- [ ] Understand chunking, hybrid search, re-ranking
- [ ] Review Example 2 in system-design-examples.md

**Day 7-8: MLOps Basics**
- [ ] Read `mlops-production-ml-guide.md` - Model Serving section
- [ ] Understand deployment patterns (A/B, Canary)
- [ ] Learn monitoring and drift detection

**Day 9-10: Review & Practice**
- [ ] Do 2-3 mock system design interviews
- [ ] Time yourself (60 minutes)
- [ ] Focus on using the template

---

### Week 3: Deep Dive (Priority 2)

**Day 1-2: Advanced LLM Topics**
- [ ] Read `llm-production-complete-guide.md` - Sections 3-4
- [ ] Learn safety patterns
- [ ] Understand distributed training

**Day 3-4: Agentic Systems**
- [ ] Read `agentic-ai/01-anthropic-patterns-interactive.ipynb`
- [ ] Review all 6 Anthropic patterns
- [ ] Understand when to use workflows vs agents

**Day 5-6: Advanced RAG**
- [ ] Read `production-rag-systems-guide.md` - Second 50%
- [ ] Learn agentic RAG patterns
- [ ] Review `agentic-ai/06-agentic-rag-patterns.ipynb`

**Day 7: Technical Deep Dives**
- [ ] Read `transformer-architecture-complete-guide.md`
- [ ] Read `attention-mechanisms-comprehensive-guide.md`
- [ ] Focus on architectures and optimizations

---

### Week 4: Integration & Practice (Priority 3)

**Day 1-3: Framework Comparisons**
- [ ] Read `agentic-ai/07-framework-comparison-guide.md`
- [ ] Understand LangGraph vs CrewAI vs AutoGen
- [ ] Learn when to choose each framework

**Day 4-5: Complete System Design Practice**
- [ ] Practice all 3 examples from system-design-examples.md
- [ ] Add your own variations
- [ ] Focus on trade-offs and alternatives

**Day 6-7: Mock Interviews**
- [ ] Do 5+ complete mock interviews
- [ ] Cover different types: RAG, LLM serving, MLOps
- [ ] Record yourself and review
- [ ] Practice explaining trade-offs

---

## Common System Design Questions

### Category 1: LLM Serving (30%)

**Q1: Design a system to serve a 70B LLM API**
- Requirements: 10K req/s, <2s latency, <$0.01/req
- Focus: vLLM, quantization, batching, caching
- **Resource:** llm-production-complete-guide.md

**Q2: Build a chatbot for customer support**
- Requirements: Contextual, safe, scalable
- Focus: RAG, conversation memory, safety filters
- **Resource:** production-rag-systems-guide.md + llm-production-complete-guide.md

**Q3: Design a code generation API**
- Requirements: Multi-language, fast, accurate
- Focus: Model selection, caching, evaluation
- **Resource:** system-design-examples.md patterns

---

### Category 2: RAG Systems (25%)

**Q4: Design a document Q&A system for 1M documents**
- Requirements: Semantic search, citations, multi-format
- Focus: Chunking, hybrid search, re-ranking
- **Resource:** production-rag-systems-guide.md + system-design-examples.md (Example 2)

**Q5: Build a legal document analysis system**
- Requirements: High accuracy, citations, compliance
- Focus: Hierarchical chunking, citation tracking, human-in-loop
- **Resource:** production-rag-systems-guide.md (advanced patterns)

**Q6: Enterprise knowledge base with multi-source search**
- Requirements: Combine docs, wikis, code, databases
- Focus: Multi-source routing, agentic RAG
- **Resource:** agentic-ai/06-agentic-rag-patterns.ipynb

---

### Category 3: MLOps & Deployment (20%)

**Q7: Design a model serving system for 10K req/s**
- Requirements: <50ms latency, A/B testing, monitoring
- Focus: TorchServe/vLLM, load balancing, metrics
- **Resource:** mlops-production-ml-guide.md (Model Serving)

**Q8: Build an ML monitoring and retraining pipeline**
- Requirements: Drift detection, auto-retraining, alerts
- Focus: Evidently AI, data drift, concept drift
- **Resource:** mlops-production-ml-guide.md (Monitoring)

**Q9: Design CI/CD for ML models**
- Requirements: Automated testing, gradual rollout, rollback
- Focus: GitHub Actions, canary deployment, shadow mode
- **Resource:** mlops-production-ml-guide.md (CI/CD)

---

### Category 4: Agentic Systems (15%)

**Q10: Design a research agent**
- Requirements: Multi-source search, synthesis, citations
- Focus: Query decomposition, multi-hop retrieval
- **Resource:** agentic-ai/05-research-agents-deep-reasoning.ipynb

**Q11: Build a coding assistant with tool use**
- Requirements: Code search, execution, debugging
- Focus: ReAct pattern, tool integration
- **Resource:** agentic-ai/01-anthropic-patterns-interactive.ipynb + llm-production-complete-guide.md

**Q12: Design a multi-agent content generation system**
- Requirements: Researcher, writer, editor collaboration
- Focus: CrewAI patterns, orchestration
- **Resource:** agentic-ai/04-multi-agent-systems.ipynb

---

### Category 5: Security & Safety (10%)

**Q13: Design a content moderation pipeline**
- Requirements: Real-time, multi-language, PII detection
- Focus: Moderation API, custom filters, multi-layer
- **Resource:** llm-production-complete-guide.md (Safety)

**Q14: Build prompt injection defense system**
- Requirements: Detect and block malicious prompts
- Focus: Pattern detection, input validation
- **Resource:** llm-production-complete-guide.md (Safety)

---

## Evaluation Criteria

### What Interviewers Look For

**1. Requirements Clarification (10%)**
- Do you ask clarifying questions?
- Do you understand functional vs non-functional requirements?
- Do you define success metrics?

**2. High-Level Design (25%)**
- Can you draw clear component diagrams?
- Do you explain data flow?
- Can you justify technology choices?
- Do you consider alternatives?

**3. Deep Dive (35%)**
- Can you go deep on 2-3 components?
- Do you understand algorithms and data structures?
- Can you write pseudo-code?
- Do you discuss trade-offs?

**4. Scalability & Optimization (20%)**
- Can you identify bottlenecks?
- Do you propose concrete optimizations?
- Can you calculate capacity (back-of-envelope)?
- Do you handle failure scenarios?

**5. Communication (10%)**
- Do you explain clearly?
- Do you think aloud?
- Do you adapt based on feedback?
- Do you manage time well?

---

### Scoring Rubric

| Level | Description | What It Looks Like |
|-------|-------------|-------------------|
| **Strong Hire** | Exceeds expectations in all areas | Excellent design, handles edge cases, proposes optimizations, demonstrates expertise |
| **Hire** | Meets expectations in most areas | Solid design, understands trade-offs, addresses scalability, communicates well |
| **Borderline** | Meets some expectations, gaps in others | Decent design but missing details, or good technical depth but poor communication |
| **No Hire** | Falls short in multiple areas | Incomplete design, missing fundamentals, poor communication, can't handle complexity |

---

## Interview Day Checklist

### 1 Week Before
- [ ] Reviewed all 7 key resources
- [ ] Practiced 10+ system design questions
- [ ] Can draw architectures quickly
- [ ] Know trade-offs for all major decisions

### 3 Days Before
- [ ] Review system-design-examples.md
- [ ] Practice explaining architectures aloud
- [ ] Review common bottlenecks and solutions
- [ ] Prepare questions for interviewers

### 1 Day Before
- [ ] Light review (don't cram)
- [ ] Review the universal template
- [ ] Get good sleep
- [ ] Prepare environment (quiet space, whiteboard/paper)

### Day Of
- [ ] Arrive 10 minutes early
- [ ] Have pen and paper ready
- [ ] Bring water
- [ ] Stay calm and think aloud
- [ ] Ask clarifying questions
- [ ] Manage time (use template)

---

## Quick Reference: Decision Matrices

### LLM Serving

| Req/s | Latency | Model Size | Solution |
|-------|---------|------------|----------|
| <100 | Any | Any | HuggingFace Transformers |
| 100-1K | <2s | <13B | TorchServe + quantization |
| 1K-10K | <2s | Any | vLLM + batching |
| 10K+ | <2s | <70B | vLLM + quantization + replicas |

### RAG Chunking

| Document Type | Chunk Size | Strategy |
|--------------|-----------|----------|
| Short docs (<10 pages) | 256-512 | Fixed-size |
| Long docs (10-100 pages) | 512-1024 | Semantic |
| Structured docs | Variable | Hierarchical |
| Code | By function | Semantic |

### Model Deployment

| Risk | Strategy |
|------|----------|
| Low | Blue-green (instant switch) |
| Medium | Canary (5% → 25% → 100%) |
| High | Shadow (0% traffic, compare offline) |

### Distributed Training

| Model Size | Strategy |
|-----------|----------|
| <7B | DDP (Data Parallel) |
| 7-30B | FSDP or ZeRO-2 |
| 30-70B | ZeRO-3 + TP |
| 70B+ | ZeRO-3 + TP + PP |

---

## Conclusion

You now have **comprehensive coverage** of all LLM and ML system design topics:

### **Total Coverage:**
- ✅ **7 detailed guides** (343KB of content)
- ✅ **7 agentic AI notebooks** (145 pages)
- ✅ **3 complete system design examples**
- ✅ **50+ mermaid diagrams**
- ✅ **100+ code examples**

### **You're Ready If You Can:**
1. ✅ Draw a complete RAG system architecture in 10 minutes
2. ✅ Explain vLLM continuous batching and PagedAttention
3. ✅ Design a model serving system for 10K req/s
4. ✅ Discuss all 6 Anthropic agentic patterns
5. ✅ Explain data drift detection and mitigation
6. ✅ Design LLM safety defense (prompt injection)
7. ✅ Choose right distributed training strategy for 70B model
8. ✅ Discuss trade-offs for every major design decision

### **Your Advantage:**
- Most candidates study scattered resources
- You have **structured, comprehensive, Cisco-relevant materials**
- All materials include **production considerations**
- **Deep + broad coverage** from fundamentals to advanced

---

**Good Luck with Your Cisco Engineering Manager Interview! 🚀**

You've prepared thoroughly, your knowledge is deep, and you understand the trade-offs. Go show them what you know!

---

*Last Updated: January 2025*
*Created for: Cisco ML/AI Engineering Manager Interview*
*Coverage: LLM Systems, MLOps, RAG, Agentic AI, Safety, Training*
