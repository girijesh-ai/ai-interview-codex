# 📖 API Usage Guide

Complete guide to using the Enterprise Agent System API with practical examples.

---

## Table of Contents

1. [Authentication](#authentication)
2. [Making Requests](#making-requests)
3. [API Endpoints](#api-endpoints)
4. [Request Lifecycle](#request-lifecycle)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)
7. [Code Examples](#code-examples)

---

## Authentication

Currently, the API is open for development. Production authentication coming soon.

```python
# For future use
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}
```

---

## Making Requests

### Base URL
```
Development: http://localhost:8000
Production: https://your-domain.com
```

### Common Headers
```http
Content-Type: application/json
Accept: application/json
```

---

## API Endpoints

### 1. Create a Request

**POST** `/requests`

Create a new customer support request.

**Request Body:**
```json
{
  "customer_id": "cust-123e4567-e89b-12d3-a456-426614174000",
  "message": "How do I reset my password?",
  "category": "account",
  "priority": 2,
  "metadata": {
    "source": "web",
    "user_agent": "Mozilla/5.0..."
  }
}
```

**Response:**
```json
{
  "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
  "customer_id": "cust-123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "category": "account",
  "priority": 2,
  "message": "How do I reset my password?",
  "solution": null,
  "confidence_score": null,
  "requires_approval": false,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "source": "web"
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/requests \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust-12345",
    "message": "How do I reset my password?",
    "category": "account",
    "priority": 2
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/requests",
    json={
        "customer_id": "cust-12345",
        "message": "How do I reset my password?",
        "category": "account",
        "priority": 2
    }
)

data = response.json()
request_id = data["request_id"]
print(f"Created request: {request_id}")
```

---

### 2. Get Request Status

**GET** `/requests/{request_id}`

Retrieve the status and details of a request.

**Response:**
```json
{
  "request_id": "req-123",
  "customer_id": "cust-456",
  "status": "completed",
  "category": "account",
  "priority": 2,
  "message": "How do I reset my password?",
  "solution": "To reset your password:\n1. Go to login page\n2. Click 'Forgot Password'\n3. Enter your email...",
  "confidence_score": 0.95,
  "requires_approval": false,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:31:23Z"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/requests/req-123
```

**Python Example:**
```python
import requests

request_id = "req-123"
response = requests.get(f"http://localhost:8000/requests/{request_id}")
data = response.json()

print(f"Status: {data['status']}")
print(f"Solution: {data['solution']}")
```

---

### 3. List Requests

**GET** `/requests`

List requests with filtering and pagination.

**Query Parameters:**
- `status`: Filter by status (pending, processing, completed, etc.)
- `category`: Filter by category (account, billing, technical, etc.)
- `priority`: Filter by priority (1-4)
- `customer_id`: Filter by customer
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20, max: 100)

**Example:**
```bash
curl "http://localhost:8000/requests?status=completed&page=1&page_size=10"
```

**Response:**
```json
{
  "requests": [
    {
      "request_id": "req-1",
      "status": "completed",
      ...
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 10,
  "has_next": true
}
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/requests",
    params={
        "status": "completed",
        "customer_id": "cust-123",
        "page": 1,
        "page_size": 20
    }
)

data = response.json()
print(f"Found {data['total']} requests")
for req in data['requests']:
    print(f"  - {req['request_id']}: {req['status']}")
```

---

### 4. Update Request

**PATCH** `/requests/{request_id}`

Update an existing request.

**Request Body:**
```json
{
  "priority": 4,
  "message": "URGENT: Still unable to reset password",
  "metadata": {
    "attempts": 3
  }
}
```

**cURL Example:**
```bash
curl -X PATCH http://localhost:8000/requests/req-123 \
  -H "Content-Type: application/json" \
  -d '{
    "priority": 4,
    "message": "URGENT: Still unable to reset password"
  }'
```

---

### 5. Approve/Reject Request

**POST** `/requests/{request_id}/approve`

Approve or reject a request that requires human review.

**Request Body:**
```json
{
  "approved": true,
  "approver": "manager@example.com",
  "notes": "Approved for immediate processing"
}
```

**Response:**
```json
{
  "request_id": "req-123",
  "approved": true,
  "approver": "manager@example.com",
  "notes": "Approved for immediate processing",
  "approved_at": "2024-01-15T10:35:00Z"
}
```

---

### 6. Health Check

**GET** `/health`

Check system health and component status.

**Response:**
```json
{
  "overall_status": "healthy",
  "components": {
    "api": {
      "status": "healthy",
      "latency_ms": 1.0
    },
    "langgraph": {
      "status": "healthy",
      "latency_ms": 15.0
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 2.0
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### 7. Get Metrics

**GET** `/metrics`

Retrieve system metrics and statistics.

**Response:**
```json
{
  "total_requests": 1000,
  "completed_requests": 950,
  "failed_requests": 50,
  "avg_resolution_time_seconds": 125.5,
  "cache_hit_rate": 0.87,
  "active_sessions": 42,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Request Lifecycle

### Status Flow

```
pending → triaged → researching → drafting → quality_check → completed
            ↓                                       ↓
        escalated ←───────────────────────→ awaiting_approval
```

### Typical Request Flow

1. **Create Request** → Status: `pending`
2. **Triage Agent** → Status: `triaged`, category assigned
3. **Research Agent** → Status: `researching`, gathering context
4. **Solution Agent** → Status: `drafting`, generating solution
5. **Quality Agent** → Status: `quality_check`, validation
6. **Completion** → Status: `completed`, solution delivered

### Escalation Flow

If confidence is low or priority is critical:
1. Status changes to `escalated`
2. `requires_approval` = `true`
3. Human reviews via `/approve` endpoint
4. After approval → `completed`

---

## Error Handling

### Error Response Format

All errors return JSON with details:

```json
{
  "error": "Validation failed",
  "code": "VALIDATION_ERROR",
  "detail": "Field 'customer_id' is required",
  "field_errors": {
    "customer_id": ["Field required"]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req-trace-123"
}
```

### Common HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation failed
- `500 Internal Server Error` - Server error

### Error Examples

**400 Bad Request:**
```json
{
  "error": "Invalid email format",
  "code": "VALIDATION_ERROR",
  "detail": "Email must be in format: user@domain.tld"
}
```

**404 Not Found:**
```json
{
  "error": "Request not found: req-99999",
  "code": "RESOURCE_NOT_FOUND",
  "detail": "No request with ID req-99999"
}
```

**422 Validation Error:**
```json
{
  "error": "Validation failed",
  "code": "VALIDATION_ERROR",
  "field_errors": {
    "message": ["Message too short (min 1 character)"],
    "priority": ["Priority must be between 1 and 4"]
  }
}
```

---

## Best Practices

### 1. Use Specific Customer IDs
```python
# Good
customer_id = "cust-123e4567-e89b-12d3-a456-426614174000"

# Bad
customer_id = "123"
```

### 2. Include Metadata for Context
```python
{
  "customer_id": "cust-123",
  "message": "Cannot access account",
  "metadata": {
    "browser": "Chrome 120",
    "ip": "192.168.1.1",
    "session_id": "sess-456"
  }
}
```

### 3. Handle Pagination for Large Results
```python
def fetch_all_requests(status="completed"):
    page = 1
    all_requests = []

    while True:
        response = requests.get(
            "http://localhost:8000/requests",
            params={"status": status, "page": page, "page_size": 100}
        )
        data = response.json()
        all_requests.extend(data['requests'])

        if not data['has_next']:
            break
        page += 1

    return all_requests
```

### 4. Implement Retry Logic
```python
import time
import requests
from requests.exceptions import RequestException

def create_request_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/requests",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Code Examples

### Complete Python Client

```python
import requests
from typing import Dict, List, Optional

class EnterpriseAgentClient:
    """Client for Enterprise Agent System API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def create_request(
        self,
        customer_id: str,
        message: str,
        category: Optional[str] = None,
        priority: int = 2,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a new request."""
        response = self.session.post(
            f"{self.base_url}/requests",
            json={
                "customer_id": customer_id,
                "message": message,
                "category": category,
                "priority": priority,
                "metadata": metadata
            }
        )
        response.raise_for_status()
        return response.json()

    def get_request(self, request_id: str) -> Dict:
        """Get request by ID."""
        response = self.session.get(f"{self.base_url}/requests/{request_id}")
        response.raise_for_status()
        return response.json()

    def list_requests(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """List requests with pagination."""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status

        response = self.session.get(
            f"{self.base_url}/requests",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def approve_request(
        self,
        request_id: str,
        approved: bool,
        approver: str,
        notes: Optional[str] = None
    ) -> Dict:
        """Approve or reject a request."""
        response = self.session.post(
            f"{self.base_url}/requests/{request_id}/approve",
            json={
                "approved": approved,
                "approver": approver,
                "notes": notes
            }
        )
        response.raise_for_status()
        return response.json()

    def get_health(self) -> Dict:
        """Get system health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = EnterpriseAgentClient()

# Create request
request = client.create_request(
    customer_id="cust-123",
    message="How do I reset my password?",
    category="account",
    priority=2
)
print(f"Created: {request['request_id']}")

# Check status
status = client.get_request(request['request_id'])
print(f"Status: {status['status']}")

# List all completed requests
completed = client.list_requests(status="completed", page_size=10)
print(f"Found {completed['total']} completed requests")
```

---

### JavaScript/TypeScript Example

```typescript
class EnterpriseAgentClient {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  async createRequest(data: {
    customer_id: string;
    message: string;
    category?: string;
    priority?: number;
    metadata?: Record<string, any>;
  }) {
    const response = await fetch(`${this.baseUrl}/requests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  async getRequest(requestId: string) {
    const response = await fetch(`${this.baseUrl}/requests/${requestId}`);
    if (!response.ok) throw new Error(`Request not found: ${requestId}`);
    return response.json();
  }
}

// Usage
const client = new EnterpriseAgentClient();

const request = await client.createRequest({
  customer_id: 'cust-123',
  message: 'How do I reset my password?',
  category: 'account',
  priority: 2,
});

console.log(`Created request: ${request.request_id}`);
```

---

## WebSocket Real-Time Updates

### Connect to WebSocket

```python
import asyncio
import websockets
import json

async def listen_for_updates(client_id: str):
    uri = f"ws://localhost:8000/ws/{client_id}"

    async with websockets.connect(uri) as websocket:
        # Subscribe to request updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "data": {
                "request_id": "req-123",
                "event_types": ["status_change", "solution_ready"]
            }
        }))

        # Listen for updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Update: {data}")

# Run
asyncio.run(listen_for_updates("client-123"))
```

---

## Rate Limiting

**Current Limits:**
- 100 requests per minute per client
- 1000 requests per hour per client

**Response Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642253400
```

---

## Need Help?

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **GitHub Issues**: [Report issues](https://github.com/your-org/enterprise-agent-system/issues)
- **Full README**: [README.md](./README.md)

---

**Happy coding! 🚀**
