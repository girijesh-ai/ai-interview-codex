# Module 06: Production Deployment

**Level:** Advanced (Senior Engineer)  
**Duration:** 8-10 hours  
**Prerequisites:** Modules 01-05 completed  
**Updated:** December 2025 - Includes LangGraph Platform, BYOC, and enterprise patterns

## Table of Contents
1. [Introduction](#introduction)
2. [Deployment Options Overview](#deployment-options-overview)
3. [LangGraph Platform (Cloud SaaS)](#langgraph-platform-cloud-saas)
4. [Bring Your Own Cloud (BYOC)](#bring-your-own-cloud-byoc)
5. [Self-Hosted Enterprise](#self-hosted-enterprise)
6. [Monitoring with LangSmith](#monitoring-with-langsmith)
7. [Security & Authentication](#security--authentication)
8. [Scaling Strategies](#scaling-strategies)
9. [Enterprise Case Studies](#enterprise-case-studies)
10. [Best Practices & Production Checklist](#best-practices--production-checklist)

---

## Introduction

**Production deployment** of LangGraph applications requires understanding multiple deployment models, monitoring strategies, and enterprise patterns. This module covers the complete journey from development to production-grade deployment.

> **🆕 What's New in 2024-2025**
> - LangGraph Platform (rebranded to LangSmith Deployment in Oct 2025)
> - BYOC (Bring Your Own Cloud) for AWS
> - Self-hosted enterprise with Helm charts
> - Enhanced LangSmith integration for monitoring
> - Production multi-agent deployments at scale

### Learning Objectives

By the end of this module, you will:
- ✅ Understand all LangGraph deployment options
- ✅ Deploy to LangGraph Platform (Cloud SaaS)
- ✅ Implement BYOC deployment on AWS
- ✅ Set up self-hosted enterprise with Kubernetes/Helm
- ✅ Integrate LangSmith for monitoring and observability
- ✅ Implement enterprise security patterns
- ✅ Scale production deployments
- ✅ Learn from real enterprise case studies

---

## Deployment Options Overview

LangGraph offers **four primary deployment models** as of December 2025:

### Deployment Matrix

| Feature | Cloud SaaS | BYOC | Self-Hosted Lite | Self-Hosted Enterprise |
|---------|------------|------|------------------|------------------------|
| **Managed by** | LangChain | Hybrid | You | You |
| **Infrastructure** | LangChain | Your AWS VPC | Your infra | Your infra |
| **Control** | Low | Medium | High | Maximum |
| **Maintenance** | None | Low | Medium | High |
| **Scalability** | Auto | Auto | Limited | Unlimited |
| **Max Nodes** | Unlimited | Unlimited | 1M | Unlimited |
| **RBAC/SSO** | ✅ | ✅ | ❌ | ✅ |
| **Best For** | Quick start | Enterprise | Development | Large enterprise |
| **License** | Pay-as-you-go | Required | Free | Required |

```mermaid
graph TD
    A[Choose Deployment] --> B{Requirements?}
    B -->|Quick start| C[Cloud SaaS]
    B -->|Data sovereignty| D[BYOC]
    B -->|Full control| E[Self-Hosted]
    
    D --> F{Cloud?}
    F -->|AWS| G[BYOC AWS]
    F -->|Other| H[Self-Hosted]
    
    E --> I{Scale?}
    I -->|Small| J[Lite]
    I -->|Large| K[Enterprise]
    
    style C fill:#90EE90
    style G fill:#FFD700
    style K fill:#FFB6C1
```

---

## LangGraph Platform (Cloud SaaS)

### What is LangGraph Platform?

Fully managed deployment platform for LangGraph applications providing:
- **Zero infrastructure management**
- **Automatic scaling**
- **Built-in monitoring (LangSmith)**
- **Version control for agents**
- **Managed checkpointers and stores**

### Quick Start Deployment

```bash
# Install LangGraph CLI
pip install -U langgraph-cli

# Create new LangGraph application
langgraph new my-agent

# Navigate to project
cd my-agent

# Deploy to platform
langgraph deploy
```

### Platform Architecture

```mermaid
graph LR
    Client[Client Apps] --> API[LangGraph API]
    API --> Server[LangGraph Server<br/>Auto-scaled]
    Server --> DB[(PostgreSQL<br/>Managed)]
    Server --> Redis[(Redis<br/>Managed)]
    Server --> LangSmith[LangSmith<br/>Monitoring]
    
    style Server fill:#e1f5ff
    style DB fill:#90EE90
    style LangSmith fill:#FFD700
```

### Configuration

Create `langgraph.json`:

```json
{
  "dependencies": ["langgraph", "langchain-openai"],
  "graphs": {
    "my_agent": "./src/agent.py:graph"
  },
  "env": {
    "OPENAI_API_KEY": "$OPENAI_API_KEY"
  }
}
```

### Using the Deployed Agent

```python
from langgraph_sdk import get_client

# Connect to deployed agent
client = get_client(url="https://my-agent.langgraph.app")

# Run agent
thread = client.threads.create()
response = client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="my_agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]}
)

# Stream results
for chunk in client.runs.stream(thread_id=thread["thread_id"], run_id=response["run_id"]):
    print(chunk)
```

### Platform Features

**1. Version Control**
```bash
# Deploy new version
langgraph deploy --revision v2.0

# Rollback to previous version
langgraph rollback v1.5
```

**2. Team Collaboration**
- **Workspaces**: Organize agents by team/project
- **RBAC**: Role-based access control
- **Shared agents**: Reuse across organization

**3. Built-in Observability**
- Automatic LangSmith tracing
- Performance metrics
- Error tracking
- Request logs

---

## Bring Your Own Cloud (BYOC)

### BYOC Architecture

**Hybrid model**: LangChain manages control plane, you host data plane in your AWS VPC.

```mermaid
graph TB
    subgraph "LangChain Control Plane"
        CP[Management Console]
        Monitor[Monitoring]
    end
    
    subgraph "Your AWS VPC"
        subgraph "Data Plane"
            ECS[ECS/Fargate<br/>LangGraph Server]
            RDS[(RDS PostgreSQL)]
            Redis[(ElastiCache Redis)]
        end
    end
    
    CP -.Manages.-> ECS
    Monitor -.Observes.-> ECS
    ECS --> RDS
    ECS --> Redis
    
    Client[Your Apps] --> ECS
    
    style ECS fill:#e1f5ff
    style RDS fill:#90EE90
    style Redis fill:#FFB6C1
```

### BYOC Setup (AWS)

**Step 1: Prerequisites**
- AWS account with admin access
- LangGraph Platform license key
- Terraform installed
- AWS CLI configured

**Step 2: Use LangChain's Terraform Module**

```hcl
# main.tf
module "langgraph_byoc" {
  source = "langchain/langgraph-byoc/aws"
  version = "~> 1.0"
  
  # Your configuration
  project_name = "my-company-agents"
  environment = "production"
  
  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
  
  # Database configuration
  db_instance_class = "db.r6g.large"
  db_allocated_storage = 100
  
  # Compute configuration
  ecs_task_cpu = 2048
  ecs_task_memory = 4096
  ecs_desired_count = 3
  
  # License
  lang graph_license_key = var.langgraph_license_key
  
  # Tags
  tags = {
    Environment = "production"
    Team = "ai-platform"
    ManagedBy = "terraform"
  }
}

output "langgraph_endpoint" {
  value = module.langgraph_byoc.api_endpoint
}

output "rds_endpoint" {
  value = module.langgraph_byoc.rds_endpoint
  sensitive = true
}
```

**Step 3: Deploy**

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="langgraph_license_key=$LICENSE_KEY"

# Apply
terraform apply -auto-approve
```

### BYOC Benefits

1. **Data Sovereignty**: Data never leaves your AWS account
2. **Compliance**: Meet regulatory requirements (HIPAA, SOC2, etc.)
3. **Network Control**: Deploy in private VPCs
4. **Cost Control**: Direct AWS billing
5. **Managed Service**: LangChain handles updates and maintenance

---

## Self-Hosted Enterprise

### Kubernetes Deployment with Helm

For maximum control, deploy LangGraph on your own Kubernetes cluster.

**Architecture:**

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        Ingress[Ingress Controller] --> LG[LangGraph Pods<br/>3+ replicas]
        LG --> PG[(PostgreSQL<br/>StatefulSet)]
        LG --> RD[(Redis<br/>StatefulSet)]
        
        HPA[Horizontal Pod<br/>Autoscaler] -.Scales.-> LG
        PVC[(Persistent<br/>Volume)] --> PG
    end
    
    LoadBalancer[Load Balancer] --> Ingress
    
    style LG fill:#e1f5ff
    style PG fill:#90EE90
    style RD fill:#FFB6C1
```

### Step 1: Prerequisites

```bash
# Kubernetes cluster (EKS, GKE, AKS, or on-prem)
kubectl cluster-info

# Helm installed
helm version

# Add LangGraph Helm repo
helm repo add langchain https://langchain-ai.github.io/helm
helm repo update
```

### Step 2 Configure Values

Create `values.yaml`:

```yaml
# values.yaml
global:
  image:
    repository: langchain/langgraph-api
    tag: "latest"
    pullPolicy: IfNotPresent

replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

postgresql:
  enabled: true
  auth:
    database: langgraph
    username: langgraph
    password: changeme  # Use secret in production!
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        memory: "4Gi"
        cpu: "2000m"

redis:
  enabled: true
  auth:
    password: changeme  # Use secret in production!
  master:
    persistence:
      size: 10Gi

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: agents.mycompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: langgraph-tls
      hosts:
        - agents.mycompany.com

env:
  - name: LANGSMITH_API_KEY
    valueFrom:
      secretKeyRef:
        name: langgraph-secrets
        key: langsmith-api-key
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: langgraph-secrets
        key: openai-api-key

serviceMonitor:
  enabled: true  # For Prometheus monitoring
```

### Step 3: Deploy

```bash
# Create namespace
kubectl create namespace langgraph-prod

# Create secrets
kubectl create secret generic langgraph-secrets \
  --from-literal=langsmith-api-key=$LANGSMITH_KEY \
  --from-literal=openai-api-key=$OPENAI_KEY \
  -n langgraph-prod

# Install with Helm
helm install langgraph langchain/langgraph-cloud \
  --namespace langgraph-prod \
  --values values.yaml

# Verify deployment
kubectl get pods -n langgraph-prod
kubectl get svc -n langgraph-prod
```

### Step 4: Upgrade

```bash
# Update configuration
vim values.yaml

# Upgrade deployment
helm upgrade langgraph langchain/langgraph-cloud \
  --namespace langgraph-prod \
  --values values.yaml
```

---

## Monitoring with LangSmith

### LangSmith Integration

LangSmith provides observability for production LangGraph applications.

**Setup:**

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-agents"

# Your LangGraph app automatically traces to LangSmith
app = workflow.compile(checkpointer=checkpointer)
```

### Key Metrics to Monitor

1. **Request Metrics**
   - Throughput (requests/second)
   - Latency (p50, p95, p99)
   - Error rate
   
2. **Agent Metrics**
   - Steps per execution
   - Tool call frequency
   - LLM token usage
   - Cost per request

3. **System Metrics**
   - CPU utilization
   - Memory usage
   - Database connections
   - Queue depth

### LangSmith Dashboard

```python
# Custom metrics
from langsmith import traceable

@traceable(name="critical_operation")
def important_node(state):
    # Automatically traced
    result = process(state)
    return {"result": result}
```

### Alerts Setup

Configure alerts in LangSmith for:
- Error rate > 5%
- Latency p95 > 2 seconds
- Cost per day > threshold
- Specific error patterns

---

## Security & Authentication

### API Authentication

```python
# Production API with authentication
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not verify_jwt(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

app = FastAPI()

@app.post("/invoke")
async def invoke_agent(request: dict, token: str = Depends(verify_token)):
    result = agent_app.invoke(request)
    return result
```

### Secrets Management

**Use AWS Secrets Manager:**

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Load secrets
secrets = get_secret('langgraph/production/api-keys')
os.environ['OPENAI_API_KEY'] = secrets['openai_key']
```

### Network Security

```yaml
# Kubernetes Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: langgraph-network-policy
spec:
  podSelector:
    matchLabels:
      app: langgraph
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
```

---

## Scaling Strategies

### Horizontal Scaling

```python
# LangGraph Server handles horizontal scaling automatically
# Each instance processes requests from shared queue

# Configure auto-scaling
autoscaling_config = {
    "min_instances": 3,
    "max_instances": 20,
    "target_cpu_utilization": 70,
    "target_memory_utilization": 80,
    "scale_up_cooldown": 60,  # seconds
    "scale_down_cooldown": 300
}
```

### Database Scaling

**Read Replicas:**

```python
# PostgreSQL with read replicas
from sqlalchemy import create_engine

# Write to primary
write_engine = create_engine(
    "postgresql://user:pass@primary.rds.amazonaws.com/langgraph"
)

# Read from replica
read_engine = create_engine(
    "postgresql://user:pass@replica.rds.amazonaws.com/langgraph"
)
```

### Caching Strategy

```python
from functools import lru_cache
import redis

# Redis caching for expensive operations
redis_client = redis.Redis(host='redis.prod', port=6379, db=0)

def cached_llm_call(prompt: str):
    cache_key = f"llm:{hash(prompt)}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Call LLM
    result = llm.invoke(prompt)
    
    # Cache result (24 hour TTL)
    redis_client.setex(cache_key, 86400, json.dumps(result))
    
    return result
```

---

## Enterprise Case Studies

### Case Study 1: LinkedIn's SQL Bot

**Challenge**: Deploy multi-agent system for SQL query generation at scale

**Solution**:
- Self-hosted on Kubernetes
- 10+ specialized agents
- Command-based routing
- PostgreSQL for state persistence
- 100K+ queries/day

**Architecture**:
```
User Query → Classifier Agent → SQL Generator Agent → Validator Agent → Executor
                                          ↓
                                   Optimization Agent
```

**Key Learnings**:
- Command tool essential for dynamic routing
- Caching reduced LLM costs by 60%
- Monitoring critical for debugging

### Case Study 2: Uber's Testing Framework

**Challenge**: Automated code testing with AI agents

**Solution**:
- LangGraph Platform (Cloud SaaS)
- Multi-agent test generation
- Human-in-the-loop for approval
- Integrated with CI/CD

**Results**:
- 40% reduction in manual testing time
- 95% test coverage
- Deployed in 2 weeks

### Case Study 3: Healthcare HIPAA-Compliant Deployment

**Challenge**: Deploy clinical decision support agent with HIPAA compliance

**Solution**:
- BYOC on AWS in private VPC
- End-to-end encryption
- Audit logging with LangSmith
- Zero data leaves VPC

**Compliance Features**:
- BAA (Business Associate Agreement) with LangChain
- Data encryption at rest and in transit
- Access logging and monitoring
- Regular security audits

---

## Best Practices & Production Checklist

### Pre-Deployment Checklist

**Code Quality**
- [ ] All tests passing (unit, integration, end-to-end)
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Documentation updated

**Infrastructure**
- [ ] Database backups configured
- [ ] Monitoring and alerting setup
- [ ] Secrets management implemented
- [ ] Auto-scaling configured
- [ ] DR (Disaster Recovery) plan documented

**Security**
- [ ] Authentication implemented
- [ ] Authorization/RBAC configured
- [ ] Secrets rotated
- [ ] Network policies applied
- [ ] Encryption enabled

**Observability**
- [ ] LangSmith tracing enabled
- [ ] Custom metrics defined
- [ ] Alerts configured
- [ ] Dashboards created
- [ ] Log aggregation setup

### Production Best Practices

1. **Start Small, Scale Gradually**
   ```
   Development → Staging → Production (10%) → Production (100%)
   ```

2. **Implement Circuit Breakers**
   ```python
   from pybreaker import CircuitBreaker
   
   breaker = CircuitBreaker(fail_max=5, timeout_duration=60)
   
   @breaker
   def call_external_api():
       return requests.get("https://api.example.com")
   ```

3. **Use Feature Flags**
   ```python
   if feature_flags.is_enabled("new_agent_version"):
       agent = new_agent_v2
   else:
       agent = stable_agent_v1
   ```

4. **Monitor Everything**
   - Request rates
   - Error rates
   - Latency percentiles
   - Cost per request
   - Resource utilization

5. **Plan for Failures**
   - Circuit breakers for external deps
   - Graceful degradation
   - Retry with exponential backoff
   - Dead letter queues

---

## Summary & Next Steps

### What You Learned

✅ **Deployment Options**: Cloud SaaS, BYOC, Self-hosted (Lite & Enterprise)  
✅ **LangGraph Platform**: Fully managed deployment with auto-scaling  
✅ **BYOC on AWS**: Hybrid model with Terraform modules  
✅ **Kubernetes/Helm**: Self-hosted enterprise deployment  
✅ **Monitoring**: LangSmith integration and observability  
✅ **Security**: Authentication, secrets management, network policies  
✅ **Scaling**: Horizontal scaling, caching, database optimization  
✅ **Real Examples**: LinkedIn, Uber, Healthcare deployments  

### Key Takeaways

1. **Choose the Right Deployment Model** for your requirements
2. **Start with Cloud SaaS** for fastest time-to-production
3. **Use BYOC** when data sovereignty required
4. **Self-Host** only when necessary (full control needed)
5. **Monitor from Day 1** with LangSmith
6. **Security is Not Optional** in production
7. **Plan for Scale** even if starting small

### Practice Now

Continue to **[module-06-practice.ipynb](./module-06-practice.ipynb)** to practice:
- Deploying to LangGraph Platform
- Setting up monitoring
- Configuring BYOC with Terraform (simulation)
- Implementing production patterns

### Next Module

**Module 07: Advanced Patterns & Optimization** will cover:
- Circuit breakers and resilience
- Performance optimization
- Cost optimization strategies
- Advanced error handling

---

## Additional Resources

- [Lang Graph Platform Documentation](https://python.langchain.com/docs/langgraph/)
- [LangSmith Deployment Guide](https://python.langchain.com/langsmith/deployments)
- [BYOC Setup Guide](https://docs.langchain.com/langgraph/deployment/byoc)
- [Helm Charts Repository](https://github.com/langchain-ai/helm)
- [LinkedIn Case Study](https://langchain.com/case-studies)
- [Production Best Practices](https://python.langchain.com/docs/langgraph/deployment/)

---

**Ready to deploy to production?** Let's practice! 🚀
