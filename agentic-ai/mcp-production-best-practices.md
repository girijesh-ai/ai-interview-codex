# MCP Production Best Practices Guide

**Production-Grade Model Context Protocol Deployment**

**Version**: 1.0 (2025)
**Target Audience**: ML Engineers, DevOps Engineers, Platform Engineers
**Prerequisites**: Understanding of MCP fundamentals, container orchestration, cloud infrastructure

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Patterns](#architecture-patterns)
3. [Security Best Practices](#security-best-practices)
4. [Scalability and Performance](#scalability-and-performance)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Error Handling and Resilience](#error-handling-and-resilience)
7. [Testing Strategies](#testing-strategies)
8. [Deployment Strategies](#deployment-strategies)
9. [Cost Optimization](#cost-optimization)
10. [Compliance and Governance](#compliance-and-governance)
11. [Multi-Tenancy Patterns](#multi-tenancy-patterns)
12. [Performance Benchmarks](#performance-benchmarks)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [Migration Strategies](#migration-strategies)
15. [Production Checklist](#production-checklist)

---

## Executive Summary

### What is Production-Grade MCP?

Production-grade MCP deployment goes beyond basic implementation to include:

- **Security**: mTLS, authentication, authorization, secrets management
- **Scalability**: Load balancing, horizontal scaling, connection pooling
- **Reliability**: Circuit breakers, retries, failover, disaster recovery
- **Observability**: Metrics, logs, traces, alerting
- **Performance**: Caching, optimization, request batching
- **Compliance**: Audit logging, data governance, regulatory requirements

### Production Maturity Model

```mermaid
graph LR
    A[Level 1: Development] --> B[Level 2: Staging]
    B --> C[Level 3: Production]
    C --> D[Level 4: Enterprise]

    A1[Single server<br/>No auth<br/>Local only] --> A
    B1[Multiple servers<br/>Basic auth<br/>Internal network] --> B
    C1[HA setup<br/>mTLS<br/>Monitoring] --> C
    D1[Multi-region<br/>Zero-trust<br/>Full compliance] --> D
```

### Key Production Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Server Availability | 99.9% | < 99.5% |
| P95 Latency | < 100ms | > 500ms |
| Error Rate | < 0.1% | > 1% |
| Connection Pool Utilization | 60-80% | > 90% |
| Memory Usage | < 80% | > 90% |
| Tool Success Rate | > 99% | < 95% |

---

## Architecture Patterns

### Pattern 1: High-Availability MCP Server Cluster

**Use Case**: Mission-critical production workloads requiring 99.99% uptime

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[AWS ALB / NGINX]
    end

    subgraph "MCP Server Cluster"
        S1[MCP Server 1<br/>AZ-1a]
        S2[MCP Server 2<br/>AZ-1b]
        S3[MCP Server 3<br/>AZ-1c]
    end

    subgraph "Shared State"
        Redis[Redis Cluster<br/>Session Store]
        DB[(PostgreSQL<br/>RDS Multi-AZ)]
    end

    subgraph "Monitoring"
        Prom[Prometheus]
        Graf[Grafana]
    end

    LB --> S1
    LB --> S2
    LB --> S3

    S1 --> Redis
    S2 --> Redis
    S3 --> Redis

    S1 --> DB
    S2 --> DB
    S3 --> DB

    S1 --> Prom
    S2 --> Prom
    S3 --> Prom

    Prom --> Graf
```

**Implementation**:

```python
# production_mcp_server.py
import asyncio
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.types import TextContent
import structlog
import prometheus_client
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
import ssl

logger = structlog.get_logger()

# Metrics
REQUEST_COUNTER = prometheus_client.Counter(
    'mcp_requests_total',
    'Total MCP requests',
    ['tool_name', 'status']
)
REQUEST_DURATION = prometheus_client.Histogram(
    'mcp_request_duration_seconds',
    'Request duration',
    ['tool_name']
)
ACTIVE_CONNECTIONS = prometheus_client.Gauge(
    'mcp_active_connections',
    'Active MCP connections'
)

class ProductionMCPServer:
    """Production-grade MCP server with HA features"""

    def __init__(self, config: dict):
        self.config = config
        self.server = Server(config['server_name'])

        # Connection pools
        self.redis = None
        self.db_engine = None

        # Health check
        self.healthy = False

        # Circuit breaker state
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )

    async def initialize(self):
        """Initialize all connections"""
        try:
            # Redis for session store
            self.redis = await Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config['redis_password'],
                ssl=True,
                ssl_cert_reqs='required',
                max_connections=50,
                socket_keepalive=True,
                health_check_interval=30
            )

            # Database connection pool
            self.db_engine = create_async_engine(
                self.config['database_url'],
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            # Register tools
            self._register_tools()

            self.healthy = True
            logger.info("Server initialized successfully")

        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            raise

    def _register_tools(self):
        """Register all MCP tools with monitoring"""

        @self.server.tool()
        async def get_user_data(user_id: str) -> list[TextContent]:
            """Get user data with full production features"""

            # Increment request counter
            REQUEST_COUNTER.labels(tool_name='get_user_data', status='started').inc()

            # Start timer
            with REQUEST_DURATION.labels(tool_name='get_user_data').time():
                try:
                    # Circuit breaker check
                    if self.circuit_breaker.is_open():
                        REQUEST_COUNTER.labels(
                            tool_name='get_user_data',
                            status='circuit_open'
                        ).inc()
                        return [TextContent(
                            type="text",
                            text="Service temporarily unavailable"
                        )]

                    # Try cache first
                    cache_key = f"user:{user_id}"
                    cached = await self.redis.get(cache_key)

                    if cached:
                        logger.info("Cache hit", user_id=user_id)
                        REQUEST_COUNTER.labels(
                            tool_name='get_user_data',
                            status='cache_hit'
                        ).inc()
                        return [TextContent(type="text", text=cached.decode())]

                    # Database query with timeout
                    async with asyncio.timeout(5.0):
                        async with self.db_engine.begin() as conn:
                            result = await conn.execute(
                                "SELECT * FROM users WHERE id = %s",
                                (user_id,)
                            )
                            user = result.fetchone()

                    if not user:
                        REQUEST_COUNTER.labels(
                            tool_name='get_user_data',
                            status='not_found'
                        ).inc()
                        return [TextContent(type="text", text="User not found")]

                    # Cache result (15 min TTL)
                    user_data = str(dict(user))
                    await self.redis.setex(cache_key, 900, user_data)

                    # Record success
                    self.circuit_breaker.record_success()
                    REQUEST_COUNTER.labels(
                        tool_name='get_user_data',
                        status='success'
                    ).inc()

                    return [TextContent(type="text", text=user_data)]

                except asyncio.TimeoutError:
                    logger.error("Query timeout", user_id=user_id)
                    self.circuit_breaker.record_failure()
                    REQUEST_COUNTER.labels(
                        tool_name='get_user_data',
                        status='timeout'
                    ).inc()
                    return [TextContent(type="text", text="Request timeout")]

                except Exception as e:
                    logger.error("Tool error", error=str(e), user_id=user_id)
                    self.circuit_breaker.record_failure()
                    REQUEST_COUNTER.labels(
                        tool_name='get_user_data',
                        status='error'
                    ).inc()
                    raise

    async def health_check(self) -> dict:
        """Comprehensive health check"""
        checks = {
            "server": self.healthy,
            "redis": False,
            "database": False,
            "circuit_breaker": not self.circuit_breaker.is_open()
        }

        # Check Redis
        try:
            await self.redis.ping()
            checks["redis"] = True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))

        # Check Database
        try:
            async with self.db_engine.begin() as conn:
                await conn.execute("SELECT 1")
            checks["database"] = True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))

        return {
            "healthy": all(checks.values()),
            "checks": checks
        }

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down server")
        self.healthy = False

        if self.redis:
            await self.redis.close()

        if self.db_engine:
            await self.db_engine.dispose()

        logger.info("Shutdown complete")


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    def __init__(self, failure_threshold: int, recovery_timeout: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "open":
            # Check if recovery timeout has elapsed
            if (asyncio.get_event_loop().time() - self.last_failure_time
                > self.recovery_timeout):
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        """Record successful request"""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count
            )
```

### Pattern 2: Multi-Region Deployment

**Use Case**: Global applications requiring low latency worldwide

```mermaid
graph TB
    subgraph "Global Load Balancer"
        GLB[Route53 / CloudFlare]
    end

    subgraph "US-East Region"
        USE1[MCP Cluster US-East]
        USDB1[(RDS US-East)]
    end

    subgraph "EU-West Region"
        EUW1[MCP Cluster EU-West]
        EUDB1[(RDS EU-West)]
    end

    subgraph "AP-South Region"
        APS1[MCP Cluster AP-South]
        APSDB1[(RDS AP-South)]
    end

    GLB -->|Geo-routing| USE1
    GLB -->|Geo-routing| EUW1
    GLB -->|Geo-routing| APS1

    USE1 --> USDB1
    EUW1 --> EUDB1
    APS1 --> APSDB1

    USDB1 -.->|Replication| EUDB1
    EUDB1 -.->|Replication| APSDB1
```

**Infrastructure as Code (Terraform)**:

```hcl
# terraform/mcp_multi_region.tf

# Define regions
locals {
  regions = {
    us-east-1 = {
      name = "us-east"
      cidr = "10.0.0.0/16"
    }
    eu-west-1 = {
      name = "eu-west"
      cidr = "10.1.0.0/16"
    }
    ap-south-1 = {
      name = "ap-south"
      cidr = "10.2.0.0/16"
    }
  }
}

# ECS cluster per region
module "mcp_cluster" {
  for_each = local.regions
  source   = "./modules/mcp-cluster"

  region              = each.key
  name                = "mcp-${each.value.name}"
  vpc_cidr            = each.value.cidr
  instance_type       = "t3.medium"
  desired_capacity    = 3
  max_capacity        = 10

  # Auto-scaling
  scale_up_threshold   = 70
  scale_down_threshold = 30

  # Health check
  health_check_path = "/health"
  health_check_interval = 30

  # Secrets
  secrets_arn = aws_secretsmanager_secret.mcp_config[each.key].arn

  tags = {
    Environment = "production"
    Service     = "mcp"
    Region      = each.value.name
  }
}

# Global accelerator for low-latency routing
resource "aws_globalaccelerator_accelerator" "mcp" {
  name            = "mcp-global-accelerator"
  ip_address_type = "IPV4"
  enabled         = true
}

resource "aws_globalaccelerator_listener" "mcp" {
  accelerator_arn = aws_globalaccelerator_accelerator.mcp.id
  protocol        = "TCP"

  port_range {
    from_port = 443
    to_port   = 443
  }
}

# Endpoint groups per region
resource "aws_globalaccelerator_endpoint_group" "mcp" {
  for_each = local.regions

  listener_arn = aws_globalaccelerator_listener.mcp.id
  endpoint_group_region = each.key

  endpoint_configuration {
    endpoint_id = module.mcp_cluster[each.key].load_balancer_arn
    weight      = 100
  }

  health_check_interval_seconds = 30
  health_check_path             = "/health"
  health_check_protocol         = "HTTPS"
  threshold_count               = 3

  traffic_dial_percentage = 100
}

# RDS Aurora Global Database
resource "aws_rds_global_cluster" "mcp" {
  global_cluster_identifier = "mcp-global"
  engine                    = "aurora-postgresql"
  engine_version            = "14.6"
  database_name             = "mcp_production"
}

# Primary cluster (us-east-1)
resource "aws_rds_cluster" "primary" {
  provider                  = aws.us-east-1
  cluster_identifier        = "mcp-primary"
  engine                    = "aurora-postgresql"
  engine_version            = "14.6"
  global_cluster_identifier = aws_rds_global_cluster.mcp.id
  master_username           = "mcp_admin"
  master_password           = random_password.db_password.result
  database_name             = "mcp_production"

  backup_retention_period = 7
  preferred_backup_window = "03:00-04:00"

  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = {
    Role = "primary"
  }
}

# Secondary clusters
resource "aws_rds_cluster" "secondary" {
  for_each = {
    eu-west-1  = aws.eu-west-1
    ap-south-1 = aws.ap-south-1
  }

  provider                  = each.value
  cluster_identifier        = "mcp-secondary-${each.key}"
  engine                    = "aurora-postgresql"
  engine_version            = "14.6"
  global_cluster_identifier = aws_rds_global_cluster.mcp.id

  backup_retention_period = 7

  depends_on = [aws_rds_cluster.primary]

  tags = {
    Role = "secondary"
  }
}
```

### Pattern 3: Serverless MCP on AWS Lambda

**Use Case**: Variable workloads, cost optimization, event-driven architectures

```python
# serverless_mcp_handler.py
import json
import asyncio
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
import boto3
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.metrics import MetricUnit

logger = Logger()
tracer = Tracer()
metrics = Metrics()

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
secrets_client = boto3.client('secretsmanager')

# Global server instance (reused across invocations)
mcp_server = Server("serverless-mcp")

@mcp_server.tool()
async def query_data(table_name: str, key: str) -> list[TextContent]:
    """Query DynamoDB with metrics"""

    metrics.add_metric(name="QueryDataInvocations", unit=MetricUnit.Count, value=1)

    try:
        table = dynamodb.Table(table_name)
        response = table.get_item(Key={'id': key})

        if 'Item' in response:
            metrics.add_metric(name="QueryDataSuccess", unit=MetricUnit.Count, value=1)
            return [TextContent(
                type="text",
                text=json.dumps(response['Item'])
            )]
        else:
            metrics.add_metric(name="QueryDataNotFound", unit=MetricUnit.Count, value=1)
            return [TextContent(type="text", text="Item not found")]

    except Exception as e:
        logger.exception("Query failed")
        metrics.add_metric(name="QueryDataErrors", unit=MetricUnit.Count, value=1)
        raise

@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    """AWS Lambda handler for MCP server"""

    # Log invocation
    logger.info("MCP Lambda invoked", extra={
        "request_id": context.request_id,
        "event": event
    })

    # Run async MCP handling
    return asyncio.run(handle_mcp_request(event, context))

async def handle_mcp_request(event, context):
    """Handle MCP request asynchronously"""

    # Parse event (API Gateway format)
    body = json.loads(event.get('body', '{}'))

    # Initialize SSE transport
    async with SseServerTransport('/sse') as transport:
        # Run MCP server
        await mcp_server.run(
            transport,
            mcp.server.InitializationOptions(
                server_name="serverless-mcp",
                server_version="1.0.0"
            )
        )

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({'status': 'success'})
    }
```

**Serverless Framework Configuration**:

```yaml
# serverless.yml
service: mcp-serverless

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 1024
  timeout: 30

  environment:
    POWERTOOLS_SERVICE_NAME: mcp-serverless
    POWERTOOLS_METRICS_NAMESPACE: MCP
    LOG_LEVEL: INFO

  iam:
    role:
      statements:
        - Effect: Allow
          Action:
            - dynamodb:GetItem
            - dynamodb:Query
            - dynamodb:Scan
          Resource: !GetAtt MCPDataTable.Arn

        - Effect: Allow
          Action:
            - secretsmanager:GetSecretValue
          Resource: !Ref MCPSecrets

  layers:
    - !Ref PythonRequirementsLambdaLayer

functions:
  mcpHandler:
    handler: serverless_mcp_handler.lambda_handler
    events:
      - http:
          path: /mcp
          method: post
          cors: true

      - http:
          path: /sse
          method: get
          cors: true

    # Reserved concurrency
    reservedConcurrency: 100

    # Provisioned concurrency for low latency
    provisionedConcurrency: 5

    # VPC configuration (if needed)
    vpc:
      securityGroupIds:
        - !Ref MCPSecurityGroup
      subnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

resources:
  Resources:
    MCPDataTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: mcp-data-${self:provider.stage}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH

        # Point-in-time recovery
        PointInTimeRecoverySpecification:
          PointInTimeRecoveryEnabled: true

        # Encryption
        SSESpecification:
          SSEEnabled: true
          SSEType: KMS

    MCPSecrets:
      Type: AWS::SecretsManager::Secret
      Properties:
        Name: mcp-config-${self:provider.stage}
        Description: MCP configuration secrets
        SecretString: |
          {
            "api_key": "placeholder",
            "db_password": "placeholder"
          }

plugins:
  - serverless-python-requirements
  - serverless-plugin-tracing

custom:
  pythonRequirements:
    dockerizePip: true
    layer: true
    zip: true
    slim: true
```

---

## Security Best Practices

### 1. Authentication and Authorization

**Multi-Layer Authentication**:

```python
# auth_middleware.py
import jwt
import hashlib
from functools import wraps
from typing import Optional
import structlog

logger = structlog.get_logger()

class MCPAuthMiddleware:
    """Multi-layer authentication for MCP servers"""

    def __init__(self, config: dict):
        self.jwt_secret = config['jwt_secret']
        self.api_key_hash = config['api_key_hash']
        self.require_mfa = config.get('require_mfa', False)

    async def authenticate(self, request) -> Optional[dict]:
        """
        Authenticate request using multiple methods

        Returns user context if successful, None otherwise
        """

        # Layer 1: API Key validation
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning("Missing API key")
            return None

        if not self._verify_api_key(api_key):
            logger.warning("Invalid API key", api_key_prefix=api_key[:8])
            return None

        # Layer 2: JWT token validation
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return None

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS256'],
                options={
                    'verify_exp': True,
                    'verify_iat': True,
                    'verify_nbf': True
                }
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None

        # Layer 3: MFA verification (if required)
        if self.require_mfa:
            mfa_token = request.headers.get('X-MFA-Token')
            if not mfa_token or not self._verify_mfa(payload['user_id'], mfa_token):
                logger.warning("MFA verification failed", user_id=payload['user_id'])
                return None

        # Extract user context
        user_context = {
            'user_id': payload['user_id'],
            'email': payload.get('email'),
            'roles': payload.get('roles', []),
            'permissions': payload.get('permissions', []),
            'session_id': payload.get('session_id')
        }

        logger.info("Authentication successful", user_id=user_context['user_id'])
        return user_context

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify API key using constant-time comparison"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return hmac.compare_digest(key_hash, self.api_key_hash)

    def _verify_mfa(self, user_id: str, mfa_token: str) -> bool:
        """Verify MFA token (TOTP)"""
        # Implementation depends on MFA provider
        # Example using pyotp:
        import pyotp

        # Get user's MFA secret from secure storage
        user_mfa_secret = self._get_user_mfa_secret(user_id)
        if not user_mfa_secret:
            return False

        totp = pyotp.TOTP(user_mfa_secret)
        return totp.verify(mfa_token, valid_window=1)

    def authorize(self, user_context: dict, required_permission: str) -> bool:
        """Check if user has required permission"""

        if not user_context:
            return False

        # Check permissions
        if required_permission in user_context.get('permissions', []):
            return True

        # Check role-based permissions
        role_permissions = self._get_role_permissions(user_context.get('roles', []))
        if required_permission in role_permissions:
            return True

        logger.warning(
            "Authorization failed",
            user_id=user_context.get('user_id'),
            required_permission=required_permission
        )
        return False

    def _get_role_permissions(self, roles: list[str]) -> set[str]:
        """Get permissions for given roles"""
        # This should be loaded from database/config
        role_permission_map = {
            'admin': {'read', 'write', 'delete', 'manage'},
            'developer': {'read', 'write'},
            'analyst': {'read'},
        }

        permissions = set()
        for role in roles:
            permissions.update(role_permission_map.get(role, set()))

        return permissions


# Usage in MCP server
def require_permission(permission: str):
    """Decorator to require specific permission for tool"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user context from request context
            user_context = get_current_user_context()

            # Check authorization
            auth = get_auth_middleware()
            if not auth.authorize(user_context, permission):
                raise PermissionError(f"Permission denied: {permission}")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Example protected tool
@server.tool()
@require_permission('write')
async def update_user(user_id: str, data: dict) -> list[TextContent]:
    """Update user data (requires 'write' permission)"""
    # Implementation
    pass
```

### 2. mTLS Implementation

**Certificate Management**:

```python
# mtls_setup.py
import ssl
from pathlib import Path
import structlog

logger = structlog.get_logger()

class MTLSConfig:
    """Mutual TLS configuration for MCP servers"""

    def __init__(self, cert_dir: Path):
        self.cert_dir = cert_dir
        self.ca_cert = cert_dir / "ca.crt"
        self.server_cert = cert_dir / "server.crt"
        self.server_key = cert_dir / "server.key"

    def create_ssl_context(self, is_server: bool = True) -> ssl.SSLContext:
        """Create SSL context with mTLS"""

        if is_server:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(
                certfile=str(self.server_cert),
                keyfile=str(self.server_key)
            )
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # Load CA certificate for client verification
        context.load_verify_locations(cafile=str(self.ca_cert))

        # Require client certificates
        context.verify_mode = ssl.CERT_REQUIRED

        # Use modern TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_3

        # Strong cipher suites only
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')

        # Disable compression (CRIME attack prevention)
        context.options |= ssl.OP_NO_COMPRESSION

        # Enable hostname checking
        if not is_server:
            context.check_hostname = True

        logger.info("SSL context created", is_server=is_server)
        return context


# Usage in production server
async def run_mtls_server(config: dict):
    """Run MCP server with mTLS"""

    mtls_config = MTLSConfig(Path(config['cert_dir']))
    ssl_context = mtls_config.create_ssl_context(is_server=True)

    # Create server with SSL context
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_context=ssl_context,
        ssl_keyfile=None,  # Already loaded in context
        ssl_certfile=None,  # Already loaded in context
        ssl_ca_certs=str(mtls_config.ca_cert),
        ssl_cert_reqs=ssl.CERT_REQUIRED
    )
```

**Certificate Rotation**:

```bash
#!/bin/bash
# cert_rotation.sh

set -euo pipefail

CERT_DIR="/etc/mcp/certs"
BACKUP_DIR="/etc/mcp/certs/backups"
VAULT_ADDR="https://vault.internal.company.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

# Check if certificates are expiring soon (30 days)
check_expiration() {
    local cert_file=$1
    local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local now_epoch=$(date +%s)
    local days_until_expiry=$(( ($expiry_epoch - $now_epoch) / 86400 ))

    echo $days_until_expiry
}

# Backup current certificates
backup_certs() {
    log "Backing up current certificates"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/$timestamp"

    mkdir -p "$backup_path"
    cp "$CERT_DIR"/*.{crt,key} "$backup_path/" || true

    log "Certificates backed up to $backup_path"
}

# Fetch new certificates from Vault
fetch_new_certs() {
    log "Fetching new certificates from Vault"

    # Login to Vault
    vault login -method=aws

    # Generate new server certificate
    vault write -format=json \
        pki/issue/mcp-server \
        common_name="mcp.internal.company.com" \
        alt_names="mcp-prod-1.internal.company.com,mcp-prod-2.internal.company.com" \
        ttl="8760h" > /tmp/new_cert.json

    # Extract certificate and key
    jq -r '.data.certificate' /tmp/new_cert.json > "$CERT_DIR/server.crt.new"
    jq -r '.data.private_key' /tmp/new_cert.json > "$CERT_DIR/server.key.new"
    jq -r '.data.ca_chain[]' /tmp/new_cert.json > "$CERT_DIR/ca.crt.new"

    # Set secure permissions
    chmod 600 "$CERT_DIR/server.key.new"
    chmod 644 "$CERT_DIR/server.crt.new"
    chmod 644 "$CERT_DIR/ca.crt.new"

    log "New certificates fetched successfully"
}

# Validate new certificates
validate_certs() {
    log "Validating new certificates"

    # Verify certificate chain
    if ! openssl verify -CAfile "$CERT_DIR/ca.crt.new" "$CERT_DIR/server.crt.new"; then
        error "Certificate validation failed"
        return 1
    fi

    # Verify key matches certificate
    local cert_modulus=$(openssl x509 -noout -modulus -in "$CERT_DIR/server.crt.new" | openssl md5)
    local key_modulus=$(openssl rsa -noout -modulus -in "$CERT_DIR/server.key.new" | openssl md5)

    if [ "$cert_modulus" != "$key_modulus" ]; then
        error "Certificate and key do not match"
        return 1
    fi

    log "Certificate validation successful"
    return 0
}

# Atomic certificate replacement
replace_certs() {
    log "Replacing certificates"

    mv "$CERT_DIR/server.crt.new" "$CERT_DIR/server.crt"
    mv "$CERT_DIR/server.key.new" "$CERT_DIR/server.key"
    mv "$CERT_DIR/ca.crt.new" "$CERT_DIR/ca.crt"

    log "Certificates replaced"
}

# Reload MCP servers
reload_servers() {
    log "Reloading MCP servers"

    # Send SIGHUP to reload certificates without downtime
    for pod in $(kubectl get pods -n mcp -l app=mcp-server -o name); do
        kubectl exec -n mcp "$pod" -- kill -HUP 1
    done

    log "Servers reloaded"
}

# Main rotation logic
main() {
    log "Starting certificate rotation"

    # Check if rotation is needed
    local days_until_expiry=$(check_expiration "$CERT_DIR/server.crt")

    if [ $days_until_expiry -gt 30 ]; then
        log "Certificate valid for $days_until_expiry days, no rotation needed"
        exit 0
    fi

    log "Certificate expires in $days_until_expiry days, rotating"

    # Perform rotation
    backup_certs
    fetch_new_certs

    if validate_certs; then
        replace_certs
        reload_servers
        log "Certificate rotation completed successfully"
    else
        error "Certificate rotation failed validation"
        exit 1
    fi
}

main "$@"
```

### 3. Secrets Management

**AWS Secrets Manager Integration**:

```python
# secrets_manager.py
import boto3
import json
from functools import lru_cache
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class SecretsManager:
    """AWS Secrets Manager integration with caching"""

    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client('secretsmanager', region_name=region)
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)

    @lru_cache(maxsize=100)
    def get_secret(self, secret_name: str) -> dict:
        """
        Get secret from AWS Secrets Manager with caching

        Caching prevents excessive API calls and reduces latency
        """

        # Check cache first
        if secret_name in self.cache:
            cached_value, cached_time = self.cache[secret_name]
            if datetime.now() - cached_time < self.cache_ttl:
                logger.debug("Secret cache hit", secret_name=secret_name)
                return cached_value

        # Fetch from Secrets Manager
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_value = json.loads(response['SecretString'])

            # Update cache
            self.cache[secret_name] = (secret_value, datetime.now())

            logger.info("Secret fetched from AWS", secret_name=secret_name)
            return secret_value

        except Exception as e:
            logger.error("Failed to fetch secret", secret_name=secret_name, error=str(e))
            raise

    def rotate_secret(self, secret_name: str):
        """Trigger secret rotation"""

        try:
            self.client.rotate_secret(SecretId=secret_name)

            # Invalidate cache
            if secret_name in self.cache:
                del self.cache[secret_name]

            logger.info("Secret rotation triggered", secret_name=secret_name)

        except Exception as e:
            logger.error("Secret rotation failed", secret_name=secret_name, error=str(e))
            raise


# Usage in MCP server
secrets = SecretsManager()

async def get_database_credentials() -> dict:
    """Get database credentials from Secrets Manager"""
    return secrets.get_secret('mcp/production/database')

async def get_api_keys() -> dict:
    """Get API keys from Secrets Manager"""
    return secrets.get_secret('mcp/production/api-keys')
```

---

## Scalability and Performance

### 1. Connection Pooling

**Database Connection Pool**:

```python
# db_pool.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
import structlog

logger = structlog.get_logger()

class DatabasePool:
    """Production-grade database connection pool"""

    def __init__(self, database_url: str, config: dict):
        self.engine = create_async_engine(
            database_url,

            # Pool configuration
            poolclass=QueuePool,
            pool_size=20,          # Core pool size
            max_overflow=10,       # Additional connections during peak
            pool_timeout=30,       # Wait time for connection
            pool_recycle=3600,     # Recycle connections every hour
            pool_pre_ping=True,    # Verify connections before use

            # Connection arguments
            connect_args={
                "server_settings": {
                    "application_name": "mcp-server",
                    "jit": "off"  # Disable JIT for predictable performance
                },
                "command_timeout": 60,
                "timeout": 10
            },

            # Echo SQL for debugging (disable in production)
            echo=config.get('echo_sql', False),

            # Execution options
            execution_options={
                "isolation_level": "READ COMMITTED"
            }
        )

        # Session factory
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info("Database pool initialized", pool_size=20, max_overflow=10)

    async def get_session(self) -> AsyncSession:
        """Get database session from pool"""
        return self.session_factory()

    async def health_check(self) -> bool:
        """Check database pool health"""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    async def get_pool_stats(self) -> dict:
        """Get pool statistics"""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow()
        }

    async def dispose(self):
        """Dispose connection pool"""
        await self.engine.dispose()
        logger.info("Database pool disposed")
```

### 2. Request Batching

**Batch Processing for Efficiency**:

```python
# batch_processor.py
import asyncio
from collections import defaultdict
from typing import Any, Callable
import structlog

logger = structlog.get_logger()

class RequestBatcher:
    """
    Batch multiple requests to reduce database/API calls

    Example: 100 individual queries → 1 batched query
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_wait_ms: int = 10,
        processor: Callable = None
    ):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.processor = processor

        self.pending_requests = defaultdict(list)
        self.locks = defaultdict(asyncio.Lock)

    async def submit(self, batch_key: str, request_data: Any) -> Any:
        """
        Submit request for batching

        batch_key: Groups related requests (e.g., 'user_data', 'accounts')
        request_data: Individual request data
        """

        # Create future for this request
        future = asyncio.Future()

        async with self.locks[batch_key]:
            self.pending_requests[batch_key].append((request_data, future))

            # Check if batch is full
            if len(self.pending_requests[batch_key]) >= self.batch_size:
                await self._flush_batch(batch_key)
            else:
                # Schedule flush after max_wait_ms
                asyncio.create_task(self._schedule_flush(batch_key))

        # Wait for result
        return await future

    async def _schedule_flush(self, batch_key: str):
        """Flush batch after timeout"""
        await asyncio.sleep(self.max_wait_ms / 1000.0)

        async with self.locks[batch_key]:
            if self.pending_requests[batch_key]:
                await self._flush_batch(batch_key)

    async def _flush_batch(self, batch_key: str):
        """Process batched requests"""

        if not self.pending_requests[batch_key]:
            return

        batch = self.pending_requests[batch_key]
        self.pending_requests[batch_key] = []

        logger.info("Flushing batch", batch_key=batch_key, size=len(batch))

        try:
            # Extract request data
            requests = [req for req, _ in batch]

            # Process batch (single DB/API call)
            results = await self.processor(batch_key, requests)

            # Distribute results to futures
            for i, (_, future) in enumerate(batch):
                if not future.done():
                    future.set_result(results[i])

        except Exception as e:
            logger.error("Batch processing failed", error=str(e))

            # Propagate error to all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)


# Example: Batch user data queries
async def batch_get_users(batch_key: str, user_ids: list[str]) -> list[dict]:
    """Get multiple users in single query"""

    async with db_pool.get_session() as session:
        result = await session.execute(
            "SELECT * FROM users WHERE id = ANY(%s)",
            (user_ids,)
        )

        users = result.fetchall()

        # Create lookup map
        user_map = {user['id']: dict(user) for user in users}

        # Return in original order
        return [user_map.get(uid) for uid in user_ids]


# Usage in MCP tool
batcher = RequestBatcher(
    batch_size=100,
    max_wait_ms=10,
    processor=batch_get_users
)

@server.tool()
async def get_user(user_id: str) -> list[TextContent]:
    """Get user data (batched automatically)"""

    user_data = await batcher.submit('users', user_id)

    return [TextContent(
        type="text",
        text=json.dumps(user_data)
    )]
```

### 3. Caching Strategies

**Multi-Layer Caching**:

```python
# cache_strategy.py
import asyncio
from typing import Any, Optional
import hashlib
import json
from redis.asyncio import Redis
import structlog

logger = structlog.get_logger()

class MultiLayerCache:
    """
    Three-layer caching strategy:
    1. In-memory (fastest, limited size)
    2. Redis (fast, shared across instances)
    3. Database (slowest, source of truth)
    """

    def __init__(self, redis: Redis, max_memory_items: int = 1000):
        self.redis = redis
        self.memory_cache = {}
        self.max_memory_items = max_memory_items

        # LRU tracking
        self.access_order = []

    async def get(
        self,
        key: str,
        fetch_fn: Optional[callable] = None,
        ttl: int = 300
    ) -> Optional[Any]:
        """
        Get value from cache (L1 → L2 → L3)

        key: Cache key
        fetch_fn: Function to fetch from source if not cached
        ttl: Time-to-live in seconds
        """

        # L1: Check in-memory cache
        if key in self.memory_cache:
            logger.debug("L1 cache hit", key=key)
            self._update_lru(key)
            return self.memory_cache[key]

        # L2: Check Redis
        redis_value = await self.redis.get(key)
        if redis_value:
            logger.debug("L2 cache hit", key=key)
            value = json.loads(redis_value)

            # Populate L1 cache
            await self._set_memory(key, value)
            return value

        # L3: Fetch from source
        if fetch_fn:
            logger.debug("Cache miss, fetching from source", key=key)
            value = await fetch_fn()

            if value is not None:
                # Populate all layers
                await self.set(key, value, ttl)

            return value

        return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in all cache layers"""

        # L1: In-memory
        await self._set_memory(key, value)

        # L2: Redis
        await self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )

        logger.debug("Value cached", key=key, ttl=ttl)

    async def _set_memory(self, key: str, value: Any):
        """Set value in memory cache with LRU eviction"""

        # Evict if cache is full
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.memory_cache[lru_key]

        self.memory_cache[key] = value
        self._update_lru(key)

    def _update_lru(self, key: str):
        """Update LRU access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    async def invalidate(self, key: str):
        """Invalidate cache entry"""

        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

        # Remove from Redis
        await self.redis.delete(key)

        logger.info("Cache invalidated", key=key)

    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""

        # Clear matching keys from memory
        keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.memory_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

        # Clear from Redis
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )

            if keys:
                await self.redis.delete(*keys)

            if cursor == 0:
                break

        logger.info("Cache pattern invalidated", pattern=pattern)


# Cache key generation
def generate_cache_key(prefix: str, params: dict) -> str:
    """Generate deterministic cache key"""

    # Sort params for consistency
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()

    return f"{prefix}:{param_hash}"


# Usage in MCP tool
cache = MultiLayerCache(redis_client, max_memory_items=1000)

@server.tool()
async def get_user_profile(user_id: str) -> list[TextContent]:
    """Get user profile with multi-layer caching"""

    cache_key = generate_cache_key("user_profile", {"user_id": user_id})

    async def fetch_from_db():
        async with db_pool.get_session() as session:
            result = await session.execute(
                "SELECT * FROM user_profiles WHERE user_id = %s",
                (user_id,)
            )
            return dict(result.fetchone())

    # Get from cache (or fetch if not cached)
    profile = await cache.get(
        cache_key,
        fetch_fn=fetch_from_db,
        ttl=600  # 10 minutes
    )

    return [TextContent(
        type="text",
        text=json.dumps(profile)
    )]


@server.tool()
async def update_user_profile(user_id: str, data: dict) -> list[TextContent]:
    """Update user profile and invalidate cache"""

    async with db_pool.get_session() as session:
        await session.execute(
            "UPDATE user_profiles SET data = %s WHERE user_id = %s",
            (json.dumps(data), user_id)
        )
        await session.commit()

    # Invalidate cache
    cache_key = generate_cache_key("user_profile", {"user_id": user_id})
    await cache.invalidate(cache_key)

    return [TextContent(type="text", text="Profile updated")]
```

---

## Monitoring and Observability

### 1. Comprehensive Metrics

**Prometheus Metrics**:

```python
# metrics.py
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, push_to_gateway
)
import time
from functools import wraps
import structlog

logger = structlog.get_logger()

# Create custom registry
registry = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'mcp_requests_total',
    'Total MCP requests',
    ['tool_name', 'status', 'client_id'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'mcp_request_duration_seconds',
    'Request duration in seconds',
    ['tool_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

REQUEST_SIZE = Summary(
    'mcp_request_size_bytes',
    'Request size in bytes',
    ['tool_name'],
    registry=registry
)

RESPONSE_SIZE = Summary(
    'mcp_response_size_bytes',
    'Response size in bytes',
    ['tool_name'],
    registry=registry
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'mcp_active_connections',
    'Number of active MCP connections',
    registry=registry
)

POOL_CONNECTIONS = Gauge(
    'mcp_pool_connections',
    'Database pool connection stats',
    ['state'],
    registry=registry
)

CACHE_HITS = Counter(
    'mcp_cache_hits_total',
    'Cache hit count',
    ['layer', 'cache_name'],
    registry=registry
)

CACHE_MISSES = Counter(
    'mcp_cache_misses_total',
    'Cache miss count',
    ['cache_name'],
    registry=registry
)

# Error metrics
ERROR_COUNT = Counter(
    'mcp_errors_total',
    'Total errors',
    ['error_type', 'tool_name'],
    registry=registry
)

# Business metrics
TOOL_INVOCATIONS = Counter(
    'mcp_tool_invocations_total',
    'Tool invocation count',
    ['tool_name', 'user_id'],
    registry=registry
)


def track_metrics(tool_name: str):
    """Decorator to track tool metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            # Track request size
            request_size = len(str(args) + str(kwargs))
            REQUEST_SIZE.labels(tool_name=tool_name).observe(request_size)

            try:
                result = await func(*args, **kwargs)

                # Track response size
                response_size = len(str(result))
                RESPONSE_SIZE.labels(tool_name=tool_name).observe(response_size)

                return result

            except Exception as e:
                status = "error"
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    tool_name=tool_name
                ).inc()
                raise

            finally:
                # Track duration
                duration = time.time() - start_time
                REQUEST_DURATION.labels(tool_name=tool_name).observe(duration)

                # Track count
                REQUEST_COUNT.labels(
                    tool_name=tool_name,
                    status=status,
                    client_id=get_current_client_id()
                ).inc()

        return wrapper
    return decorator


# Usage
@server.tool()
@track_metrics('get_account_balance')
async def get_account_balance(account_id: str) -> list[TextContent]:
    """Get account balance with metrics tracking"""
    # Implementation
    pass
```

**Grafana Dashboard (JSON)**:

```json
{
  "dashboard": {
    "title": "MCP Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(mcp_requests_total[5m])",
            "legendFormat": "{{tool_name}} - {{status}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mcp_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{tool_name}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(mcp_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ],
        "type": "graph",
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "type": "gt",
                "params": [0.01]
              },
              "query": {
                "params": ["A", "5m", "now"]
              }
            }
          ],
          "executionErrorState": "alerting",
          "frequency": "1m",
          "handler": 1,
          "name": "High Error Rate",
          "noDataState": "no_data",
          "notifications": [
            {
              "uid": "pagerduty"
            }
          ]
        }
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(mcp_cache_hits_total[5m]) / (rate(mcp_cache_hits_total[5m]) + rate(mcp_cache_misses_total[5m]))",
            "legendFormat": "{{cache_name}}"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Active Connections",
        "targets": [
          {
            "expr": "mcp_active_connections"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Database Pool",
        "targets": [
          {
            "expr": "mcp_pool_connections",
            "legendFormat": "{{state}}"
          }
        ],
        "type": "graph"
      }
    ],
    "refresh": "10s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
```

### 2. Structured Logging

**Production Logging Configuration**:

```python
# logging_config.py
import structlog
import logging
import sys
from pythonjsonlogger import jsonlogger

def configure_logging(
    level: str = "INFO",
    environment: str = "production"
):
    """Configure structured logging for production"""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

    # Structlog processors
    processors = [
        # Add log level
        structlog.stdlib.add_log_level,

        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),

        # Add caller info
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),

        # Stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,

        # Format as JSON for production
        structlog.processors.JSONRenderer() if environment == "production"
        else structlog.dev.ConsoleRenderer()
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Usage
logger = structlog.get_logger()

@server.tool()
async def process_transaction(transaction_id: str, amount: float) -> list[TextContent]:
    """Process transaction with structured logging"""

    logger.info(
        "Processing transaction",
        transaction_id=transaction_id,
        amount=amount,
        user_id=get_current_user_id()
    )

    try:
        result = await execute_transaction(transaction_id, amount)

        logger.info(
            "Transaction completed",
            transaction_id=transaction_id,
            status=result['status'],
            duration_ms=result['duration']
        )

        return [TextContent(type="text", text="Transaction processed")]

    except InsufficientFundsError as e:
        logger.warning(
            "Insufficient funds",
            transaction_id=transaction_id,
            required=amount,
            available=e.available
        )
        raise

    except Exception as e:
        logger.error(
            "Transaction failed",
            transaction_id=transaction_id,
            error=str(e),
            exc_info=True
        )
        raise
```

### 3. Distributed Tracing

**OpenTelemetry Integration**:

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from functools import wraps
import structlog

logger = structlog.get_logger()

def configure_tracing(service_name: str, otlp_endpoint: str):
    """Configure distributed tracing"""

    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "production"
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=False
    )

    provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Auto-instrument libraries
    SQLAlchemyInstrumentor().instrument()
    RedisInstrumentor().instrument()
    RequestsInstrumentor().instrument()

    logger.info("Tracing configured", service_name=service_name)


# Get tracer
tracer = trace.get_tracer(__name__)


def traced_tool(tool_name: str):
    """Decorator to trace tool execution"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                f"mcp.tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "tool.args": str(args),
                    "user.id": get_current_user_id()
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("tool.status", "success")
                    return result

                except Exception as e:
                    span.set_attribute("tool.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


# Usage
@server.tool()
@traced_tool('get_user_orders')
async def get_user_orders(user_id: str) -> list[TextContent]:
    """Get user orders with distributed tracing"""

    # Create child span for database query
    with tracer.start_as_current_span("database.query.orders") as db_span:
        db_span.set_attribute("db.user_id", user_id)

        async with db_pool.get_session() as session:
            result = await session.execute(
                "SELECT * FROM orders WHERE user_id = %s",
                (user_id,)
            )
            orders = result.fetchall()

        db_span.set_attribute("db.row_count", len(orders))

    # Create child span for cache operation
    with tracer.start_as_current_span("cache.set") as cache_span:
        cache_key = f"orders:{user_id}"
        await cache.set(cache_key, orders, ttl=300)
        cache_span.set_attribute("cache.key", cache_key)

    return [TextContent(
        type="text",
        text=json.dumps([dict(order) for order in orders])
    )]
```

---

## Error Handling and Resilience

### 1. Retry Strategies

**Exponential Backoff with Jitter**:

```python
# retry.py
import asyncio
import random
from functools import wraps
from typing import Type, Tuple
import structlog

logger = structlog.get_logger()

class RetryConfig:
    """Retry configuration"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""

        # Exponential backoff
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


def retry_async(config: RetryConfig):
    """Decorator for async functions with retry logic"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            "Max retries exceeded",
                            function=func.__name__,
                            attempts=config.max_attempts,
                            error=str(e)
                        )
                        break

                    # Calculate delay
                    delay = config.calculate_delay(attempt)

                    logger.warning(
                        "Retrying after error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=config.max_attempts,
                        delay_seconds=delay,
                        error=str(e)
                    )

                    # Wait before retry
                    await asyncio.sleep(delay)

            # All retries failed
            raise last_exception

        return wrapper
    return decorator


# Usage
@server.tool()
@retry_async(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
))
async def call_external_api(endpoint: str) -> list[TextContent]:
    """Call external API with retry logic"""

    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint, timeout=10.0)
        response.raise_for_status()
        return [TextContent(type="text", text=response.text)]
```

### 2. Circuit Breaker (Full Implementation)

```python
# circuit_breaker.py
import asyncio
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5         # Failures before opening
    success_threshold: int = 2         # Successes to close from half-open
    timeout: timedelta = timedelta(seconds=60)  # Time before half-open
    window: timedelta = timedelta(minutes=1)    # Rolling window for failures

class CircuitBreaker:
    """Production-grade circuit breaker"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()

        # Rolling window for failures
        self.recent_failures = []

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        # Check state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

        # Clear failure count on success in closed state
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            self.recent_failures = []

    def _on_failure(self):
        """Handle failed call"""

        now = datetime.now()
        self.last_failure_time = now

        # Add to rolling window
        self.recent_failures.append(now)

        # Remove old failures outside window
        cutoff = now - self.config.window
        self.recent_failures = [
            f for f in self.recent_failures if f > cutoff
        ]

        # Count failures in window
        failure_count = len(self.recent_failures)

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens circuit
            self._transition_to_open()

        elif failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""

        if self.last_failure_time is None:
            return True

        time_since_last_failure = datetime.now() - self.last_failure_time
        return time_since_last_failure >= self.config.timeout

    def _transition_to_closed(self):
        """Transition to CLOSED state"""

        logger.info(
            "Circuit breaker closing",
            name=self.name,
            previous_state=self.state.value
        )

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.recent_failures = []
        self.last_state_change = datetime.now()

    def _transition_to_open(self):
        """Transition to OPEN state"""

        logger.warning(
            "Circuit breaker opening",
            name=self.name,
            failure_count=len(self.recent_failures)
        )

        self.state = CircuitState.OPEN
        self.success_count = 0
        self.last_state_change = datetime.now()

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""

        logger.info(
            "Circuit breaker half-open",
            name=self.name
        )

        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.last_state_change = datetime.now()

    def get_state(self) -> dict:
        """Get circuit breaker state"""

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": len(self.recent_failures),
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat()
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Usage
db_circuit_breaker = CircuitBreaker(
    name="database",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=timedelta(seconds=60),
        window=timedelta(minutes=1)
    )
)

@server.tool()
async def query_database(query: str) -> list[TextContent]:
    """Query database with circuit breaker protection"""

    async def execute_query():
        async with db_pool.get_session() as session:
            result = await session.execute(query)
            return result.fetchall()

    try:
        rows = await db_circuit_breaker.call(execute_query)
        return [TextContent(type="text", text=str(rows))]

    except CircuitBreakerOpenError as e:
        logger.warning("Circuit breaker open, using fallback")
        return [TextContent(
            type="text",
            text="Service temporarily unavailable, please try again later"
        )]
```

### 3. Graceful Degradation

```python
# degradation.py
from typing import Any, Optional, Callable
import structlog

logger = structlog.get_logger()

class GracefulDegradation:
    """Graceful degradation for MCP tools"""

    def __init__(self):
        self.fallback_handlers = {}

    def register_fallback(
        self,
        tool_name: str,
        fallback: Callable
    ):
        """Register fallback handler for tool"""
        self.fallback_handlers[tool_name] = fallback

    async def execute_with_fallback(
        self,
        tool_name: str,
        primary: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute tool with fallback on failure"""

        try:
            # Try primary implementation
            return await primary(*args, **kwargs)

        except Exception as e:
            logger.warning(
                "Primary handler failed, using fallback",
                tool_name=tool_name,
                error=str(e)
            )

            # Use fallback
            if tool_name in self.fallback_handlers:
                fallback = self.fallback_handlers[tool_name]
                return await fallback(*args, **kwargs)
            else:
                # No fallback available
                raise


degradation = GracefulDegradation()

# Register fallbacks
degradation.register_fallback(
    'get_recommendations',
    lambda user_id: [TextContent(
        type="text",
        text="Personalized recommendations temporarily unavailable. Showing popular items."
    )]
)

@server.tool()
async def get_recommendations(user_id: str) -> list[TextContent]:
    """Get recommendations with graceful degradation"""

    async def get_personalized_recommendations():
        # Call ML model
        response = await ml_api.predict(user_id)
        return [TextContent(type="text", text=str(response))]

    return await degradation.execute_with_fallback(
        'get_recommendations',
        get_personalized_recommendations,
        user_id
    )
```

---

## Testing Strategies

### 1. Unit Testing

```python
# test_mcp_tools.py
import pytest
from unittest.mock import AsyncMock, patch
from mcp.types import TextContent
import json

@pytest.fixture
async def mcp_server():
    """Fixture for MCP server"""
    from production_mcp_server import ProductionMCPServer

    config = {
        'server_name': 'test-server',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_password': 'test',
        'database_url': 'postgresql://test:test@localhost/test'
    }

    server = ProductionMCPServer(config)
    await server.initialize()

    yield server

    await server.shutdown()

@pytest.mark.asyncio
async def test_get_user_data_success(mcp_server):
    """Test successful user data retrieval"""

    # Mock database query
    with patch.object(mcp_server.db_engine, 'begin') as mock_begin:
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = {
            'id': '123',
            'name': 'Test User',
            'email': 'test@example.com'
        }
        mock_conn.execute.return_value = mock_result
        mock_begin.return_value.__aenter__.return_value = mock_conn

        # Call tool
        result = await mcp_server.server.get_user_data(user_id='123')

        # Assertions
        assert len(result) == 1
        assert result[0].type == "text"

        data = json.loads(result[0].text)
        assert data['id'] == '123'
        assert data['name'] == 'Test User'

@pytest.mark.asyncio
async def test_get_user_data_cache_hit(mcp_server):
    """Test cache hit scenario"""

    # Pre-populate cache
    await mcp_server.redis.setex(
        'user:123',
        900,
        json.dumps({'id': '123', 'name': 'Cached User'})
    )

    # Call tool (should hit cache)
    result = await mcp_server.server.get_user_data(user_id='123')

    # Assertions
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data['name'] == 'Cached User'

@pytest.mark.asyncio
async def test_get_user_data_timeout(mcp_server):
    """Test timeout handling"""

    # Mock slow database query
    with patch.object(mcp_server.db_engine, 'begin') as mock_begin:
        async def slow_query(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = slow_query
        mock_begin.return_value.__aenter__.return_value = mock_conn

        # Call tool
        result = await mcp_server.server.get_user_data(user_id='123')

        # Should return timeout message
        assert "timeout" in result[0].text.lower()

@pytest.mark.asyncio
async def test_circuit_breaker_opens(mcp_server):
    """Test circuit breaker opening after failures"""

    # Cause multiple failures
    with patch.object(mcp_server.db_engine, 'begin') as mock_begin:
        mock_begin.side_effect = Exception("Database error")

        for _ in range(5):
            try:
                await mcp_server.server.get_user_data(user_id='123')
            except:
                pass

        # Circuit should be open
        assert mcp_server.circuit_breaker.is_open()

        # Next call should fail fast
        result = await mcp_server.server.get_user_data(user_id='123')
        assert "unavailable" in result[0].text.lower()
```

### 2. Integration Testing

```python
# test_integration.py
import pytest
import asyncio
from mcp.client import Client
from mcp.client.session import ClientSession
import docker

@pytest.fixture(scope="module")
async def test_infrastructure():
    """Set up test infrastructure (Redis, PostgreSQL, MCP server)"""

    # Start Docker containers
    client = docker.from_env()

    # Start Redis
    redis_container = client.containers.run(
        "redis:7",
        detach=True,
        ports={'6379/tcp': 6379},
        name="test-redis"
    )

    # Start PostgreSQL
    postgres_container = client.containers.run(
        "postgres:15",
        detach=True,
        environment={
            'POSTGRES_PASSWORD': 'test',
            'POSTGRES_DB': 'test'
        },
        ports={'5432/tcp': 5432},
        name="test-postgres"
    )

    # Wait for containers to be ready
    await asyncio.sleep(5)

    # Start MCP server
    # ... (server startup code)

    yield {
        'redis': redis_container,
        'postgres': postgres_container
    }

    # Cleanup
    redis_container.stop()
    redis_container.remove()
    postgres_container.stop()
    postgres_container.remove()

@pytest.mark.asyncio
async def test_end_to_end_flow(test_infrastructure):
    """Test complete end-to-end flow"""

    # Create MCP client
    client = Client()

    async with client.connect("http://localhost:8000/mcp") as session:
        # Test tool invocation
        result = await session.call_tool(
            "get_user_data",
            arguments={"user_id": "test-123"}
        )

        assert result.isError == False
        assert len(result.content) > 0

        # Test caching
        start = asyncio.get_event_loop().time()
        result2 = await session.call_tool(
            "get_user_data",
            arguments={"user_id": "test-123"}
        )
        duration = asyncio.get_event_loop().time() - start

        # Cached request should be faster
        assert duration < 0.01  # < 10ms

@pytest.mark.asyncio
async def test_concurrent_requests(test_infrastructure):
    """Test handling of concurrent requests"""

    client = Client()

    async with client.connect("http://localhost:8000/mcp") as session:
        # Make 100 concurrent requests
        tasks = [
            session.call_tool(
                "get_user_data",
                arguments={"user_id": f"user-{i}"}
            )
            for i in range(100)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(not r.isError for r in results)
```

### 3. Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time
from dataclasses import dataclass
import statistics
import structlog

logger = structlog.get_logger()

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    url: str
    concurrent_users: int = 100
    requests_per_user: int = 100
    ramp_up_time: int = 10  # seconds

@dataclass
class LoadTestResults:
    """Load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    requests_per_second: float
    errors: dict

async def load_test_worker(
    worker_id: int,
    config: LoadTestConfig,
    results: list
):
    """Single load test worker"""

    async with aiohttp.ClientSession() as session:
        for i in range(config.requests_per_user):
            start = time.time()

            try:
                async with session.post(
                    config.url,
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "get_user_data",
                            "arguments": {
                                "user_id": f"user-{worker_id}-{i}"
                            }
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.json()
                    latency = time.time() - start

                    results.append({
                        'success': response.status == 200,
                        'latency': latency,
                        'status': response.status
                    })

            except Exception as e:
                latency = time.time() - start
                results.append({
                    'success': False,
                    'latency': latency,
                    'error': str(e)
                })

async def run_load_test(config: LoadTestConfig) -> LoadTestResults:
    """Run load test"""

    logger.info(
        "Starting load test",
        url=config.url,
        concurrent_users=config.concurrent_users,
        requests_per_user=config.requests_per_user
    )

    results = []

    # Create workers with ramp-up
    workers = []
    ramp_delay = config.ramp_up_time / config.concurrent_users

    test_start = time.time()

    for worker_id in range(config.concurrent_users):
        worker = asyncio.create_task(
            load_test_worker(worker_id, config, results)
        )
        workers.append(worker)

        # Ramp up delay
        if worker_id < config.concurrent_users - 1:
            await asyncio.sleep(ramp_delay)

    # Wait for all workers to complete
    await asyncio.gather(*workers)

    test_duration = time.time() - test_start

    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency'] for r in results]

    # Calculate error distribution
    errors = {}
    for r in failed:
        error = r.get('error', 'Unknown')
        errors[error] = errors.get(error, 0) + 1

    return LoadTestResults(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        p50_latency=statistics.median(latencies) if latencies else 0,
        p95_latency=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
        p99_latency=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
        requests_per_second=len(results) / test_duration,
        errors=errors
    )

# Run load test
if __name__ == "__main__":
    config = LoadTestConfig(
        url="http://localhost:8000/mcp",
        concurrent_users=100,
        requests_per_user=100
    )

    results = asyncio.run(run_load_test(config))

    print(f"\n=== Load Test Results ===")
    print(f"Total Requests: {results.total_requests}")
    print(f"Successful: {results.successful_requests}")
    print(f"Failed: {results.failed_requests}")
    print(f"Success Rate: {results.successful_requests / results.total_requests * 100:.2f}%")
    print(f"\n=== Latency ===")
    print(f"Average: {results.avg_latency*1000:.2f}ms")
    print(f"P50: {results.p50_latency*1000:.2f}ms")
    print(f"P95: {results.p95_latency*1000:.2f}ms")
    print(f"P99: {results.p99_latency*1000:.2f}ms")
    print(f"\n=== Throughput ===")
    print(f"Requests/sec: {results.requests_per_second:.2f}")

    if results.errors:
        print(f"\n=== Errors ===")
        for error, count in results.errors.items():
            print(f"{error}: {count}")
```

---

## Deployment Strategies

### 1. Blue-Green Deployment

**Kubernetes Blue-Green Deployment**:

```yaml
# k8s/blue-green-deployment.yaml

# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server-blue
  namespace: mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
      version: blue
  template:
    metadata:
      labels:
        app: mcp-server
        version: blue
    spec:
      containers:
      - name: mcp-server
        image: mcp-server:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: VERSION
          value: "blue"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server-green
  namespace: mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
      version: green
  template:
    metadata:
      labels:
        app: mcp-server
        version: green
    spec:
      containers:
      - name: mcp-server
        image: mcp-server:v1.1.0  # New version
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: VERSION
          value: "green"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# Service (points to blue or green)
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
  namespace: mcp
spec:
  selector:
    app: mcp-server
    version: blue  # Change to 'green' to switch
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**Blue-Green Deployment Script**:

```bash
#!/bin/bash
# deploy_blue_green.sh

set -euo pipefail

NAMESPACE="mcp"
NEW_VERSION=$1
CURRENT_COLOR=$(kubectl get service mcp-server -n $NAMESPACE -o jsonpath='{.spec.selector.version}')

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "Current version: $CURRENT_COLOR"
echo "Deploying new version to: $NEW_COLOR"

# Update deployment image
kubectl set image deployment/mcp-server-$NEW_COLOR \
    mcp-server=mcp-server:$NEW_VERSION \
    -n $NAMESPACE

# Wait for rollout to complete
echo "Waiting for rollout..."
kubectl rollout status deployment/mcp-server-$NEW_COLOR -n $NAMESPACE --timeout=5m

# Run smoke tests
echo "Running smoke tests..."
POD=$(kubectl get pods -n $NAMESPACE -l app=mcp-server,version=$NEW_COLOR -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -- curl -f http://localhost:8000/health

# Switch traffic
echo "Switching traffic to $NEW_COLOR..."
kubectl patch service mcp-server -n $NAMESPACE \
    -p "{\"spec\":{\"selector\":{\"version\":\"$NEW_COLOR\"}}}"

echo "Deployment complete! Traffic now on $NEW_COLOR"
echo "Old version ($CURRENT_COLOR) still running for rollback if needed"
echo ""
echo "To rollback, run:"
echo "kubectl patch service mcp-server -n $NAMESPACE -p '{\"spec\":{\"selector\":{\"version\":\"$CURRENT_COLOR\"}}}'"
```

### 2. Canary Deployment

**Istio Canary Deployment**:

```yaml
# k8s/canary-deployment.yaml

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: mcp-server
  namespace: mcp
spec:
  hosts:
  - mcp-server
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: mcp-server
        subset: canary
  - route:
    - destination:
        host: mcp-server
        subset: stable
      weight: 90
    - destination:
        host: mcp-server
        subset: canary
      weight: 10  # 10% canary traffic

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: mcp-server
  namespace: mcp
spec:
  host: mcp-server
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
```

**Gradual Canary Rollout Script**:

```bash
#!/bin/bash
# canary_rollout.sh

set -euo pipefail

NAMESPACE="mcp"
NEW_VERSION=$1

# Traffic percentages
STAGES=(10 25 50 75 100)

echo "Starting canary deployment for version $NEW_VERSION"

# Deploy canary
kubectl set image deployment/mcp-server-canary \
    mcp-server=mcp-server:$NEW_VERSION \
    -n $NAMESPACE

kubectl rollout status deployment/mcp-server-canary -n $NAMESPACE

for PERCENTAGE in "${STAGES[@]}"; do
    echo "Increasing canary traffic to $PERCENTAGE%"

    # Update traffic split
    kubectl patch virtualservice mcp-server -n $NAMESPACE --type merge -p "
spec:
  http:
  - route:
    - destination:
        host: mcp-server
        subset: stable
      weight: $((100 - PERCENTAGE))
    - destination:
        host: mcp-server
        subset: canary
      weight: $PERCENTAGE
"

    # Monitor metrics for 5 minutes
    echo "Monitoring metrics for 5 minutes..."
    sleep 300

    # Check error rate
    ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(mcp_errors_total{version=\"canary\"}[5m])" | jq -r '.data.result[0].value[1]')

    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo "Error rate too high ($ERROR_RATE), rolling back"

        # Rollback
        kubectl patch virtualservice mcp-server -n $NAMESPACE --type merge -p "
spec:
  http:
  - route:
    - destination:
        host: mcp-server
        subset: stable
      weight: 100
"
        exit 1
    fi

    echo "Metrics look good, continuing..."
done

echo "Canary deployment successful!"
echo "Promoting canary to stable..."

# Update stable deployment
kubectl set image deployment/mcp-server-stable \
    mcp-server=mcp-server:$NEW_VERSION \
    -n $NAMESPACE

kubectl rollout status deployment/mcp-server-stable -n $NAMESPACE

# Route all traffic to stable
kubectl patch virtualservice mcp-server -n $NAMESPACE --type merge -p "
spec:
  http:
  - route:
    - destination:
        host: mcp-server
        subset: stable
      weight: 100
"

echo "Deployment complete!"
```

---

## Cost Optimization

### 1. Auto-Scaling

**Kubernetes HPA (Horizontal Pod Autoscaler)**:

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
  namespace: mcp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: mcp_active_connections
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### 2. Cost Monitoring

```python
# cost_monitor.py
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = structlog.get_logger()

@dataclass
class CostMetrics:
    """Cost metrics for MCP deployment"""
    compute_cost: float
    database_cost: float
    cache_cost: float
    network_cost: float
    storage_cost: float
    llm_api_cost: float
    total_cost: float
    requests_served: int
    cost_per_request: float

class CostMonitor:
    """Monitor and optimize deployment costs"""

    def __init__(self):
        self.hourly_metrics = []

    async def calculate_costs(self, period: timedelta) -> CostMetrics:
        """Calculate costs for given period"""

        # Compute costs (ECS Fargate)
        # 2 vCPU, 4GB RAM, 3 tasks = $0.12/hour
        compute_hours = period.total_seconds() / 3600
        compute_cost = compute_hours * 0.12 * 3

        # Database (RDS PostgreSQL db.r5.large)
        # $0.24/hour
        database_cost = compute_hours * 0.24

        # Cache (ElastiCache Redis cache.r5.large)
        # $0.208/hour
        cache_cost = compute_hours * 0.208

        # Network transfer
        # Assume 100GB/day @ $0.09/GB
        network_cost = (period.days or 1) * 100 * 0.09

        # Storage (EBS)
        # 500GB @ $0.10/GB/month
        storage_cost = 500 * 0.10 / 30 * (period.days or 1)

        # LLM API costs (if using external LLM)
        llm_api_cost = await self._calculate_llm_costs(period)

        total_cost = (
            compute_cost +
            database_cost +
            cache_cost +
            network_cost +
            storage_cost +
            llm_api_cost
        )

        # Get request count
        requests_served = await self._get_request_count(period)

        cost_per_request = total_cost / requests_served if requests_served > 0 else 0

        return CostMetrics(
            compute_cost=compute_cost,
            database_cost=database_cost,
            cache_cost=cache_cost,
            network_cost=network_cost,
            storage_cost=storage_cost,
            llm_api_cost=llm_api_cost,
            total_cost=total_cost,
            requests_served=requests_served,
            cost_per_request=cost_per_request
        )

    async def get_optimization_recommendations(self) -> list[str]:
        """Get cost optimization recommendations"""

        recommendations = []

        # Analyze cache hit rate
        cache_hit_rate = await self._get_cache_hit_rate()
        if cache_hit_rate < 0.7:
            recommendations.append(
                f"Low cache hit rate ({cache_hit_rate:.0%}). "
                "Consider increasing cache size or TTL."
            )

        # Analyze CPU utilization
        avg_cpu = await self._get_avg_cpu_utilization()
        if avg_cpu < 0.3:
            recommendations.append(
                f"Low CPU utilization ({avg_cpu:.0%}). "
                "Consider reducing instance size or count."
            )

        # Analyze database connections
        db_connection_utilization = await self._get_db_connection_utilization()
        if db_connection_utilization < 0.3:
            recommendations.append(
                "Low database connection utilization. "
                "Consider reducing pool size."
            )

        # Analyze network transfer
        egress_gb = await self._get_network_egress()
        if egress_gb > 1000:  # > 1TB/day
            recommendations.append(
                f"High network egress ({egress_gb}GB/day). "
                "Consider using CloudFront CDN or compression."
            )

        return recommendations
```

---

## Production Checklist

### Pre-Deployment Checklist

```yaml
# production-checklist.yaml

security:
  - ☐ mTLS configured and tested
  - ☐ API keys rotated and stored in secrets manager
  - ☐ Authentication/authorization implemented
  - ☐ Rate limiting configured
  - ☐ Input validation on all tools
  - ☐ SQL injection prevention
  - ☐ XSS prevention
  - ☐ CORS configured correctly
  - ☐ Security headers set
  - ☐ Vulnerability scanning completed

performance:
  - ☐ Connection pooling configured
  - ☐ Caching strategy implemented
  - ☐ Database indexes optimized
  - ☐ Query performance tested
  - ☐ Load testing completed
  - ☐ Auto-scaling configured
  - ☐ CDN configured (if applicable)

reliability:
  - ☐ Health checks implemented
  - ☐ Readiness probes configured
  - ☐ Circuit breakers implemented
  - ☐ Retry logic with backoff
  - ☐ Graceful shutdown handling
  - ☐ Disaster recovery plan
  - ☐ Backup strategy implemented
  - ☐ Failover tested

observability:
  - ☐ Structured logging configured
  - ☐ Metrics exported to Prometheus
  - ☐ Dashboards created in Grafana
  - ☐ Alerts configured in PagerDuty
  - ☐ Distributed tracing enabled
  - ☐ Error tracking (Sentry) configured
  - ☐ Log aggregation (ELK/Splunk) configured

deployment:
  - ☐ Blue-green or canary strategy chosen
  - ☐ Rollback procedure documented
  - ☐ Smoke tests automated
  - ☐ Integration tests passing
  - ☐ Load tests passing
  - ☐ Deployment runbook created
  - ☐ Incident response plan documented

compliance:
  - ☐ Data retention policies configured
  - ☐ Audit logging enabled
  - ☐ PII handling reviewed
  - ☐ Compliance requirements met (GDPR, SOC2, etc.)
  - ☐ Security audit completed
  - ☐ Privacy policy updated

documentation:
  - ☐ API documentation complete
  - ☐ Architecture diagrams updated
  - ☐ Runbooks created
  - ☐ Troubleshooting guide written
  - ☐ On-call procedures documented
  - ☐ SLAs defined
```

---

## Conclusion

This production best practices guide covers all essential aspects of deploying MCP servers at scale:

1. **Architecture Patterns**: HA clusters, multi-region, serverless
2. **Security**: mTLS, authentication, secrets management
3. **Scalability**: Connection pooling, caching, load balancing
4. **Monitoring**: Metrics, logging, tracing
5. **Resilience**: Circuit breakers, retries, graceful degradation
6. **Testing**: Unit, integration, load testing
7. **Deployment**: Blue-green, canary strategies
8. **Cost Optimization**: Auto-scaling, monitoring

Following these patterns ensures your MCP deployment is:
- **Secure**: Multi-layer security with zero-trust principles
- **Scalable**: Handles millions of requests/day
- **Reliable**: 99.9%+ uptime with fault tolerance
- **Observable**: Complete visibility into system behavior
- **Cost-Effective**: Optimized resource utilization

**Next Steps**:
1. Choose architecture pattern based on requirements
2. Implement security controls
3. Set up monitoring and observability
4. Load test your deployment
5. Create runbooks and documentation
6. Deploy using blue-green or canary strategy

---

**Document Status**: Production-Ready
**Last Updated**: 2025
**Maintainer**: MCP Platform Team
