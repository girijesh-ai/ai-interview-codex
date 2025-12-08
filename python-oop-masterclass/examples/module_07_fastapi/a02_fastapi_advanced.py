"""
FastAPI Advanced Patterns Example
=================================
Demonstrates:
- Custom middleware (timing, logging, request ID)
- SSE streaming for LLM responses
- WebSocket real-time chat
- Background tasks
- JWT authentication
- API key authentication
- Rate limiting

Run with: uvicorn 02_fastapi_advanced:app --reload
"""

from fastapi import (
    FastAPI, Depends, HTTPException, Request, Response,
    WebSocket, WebSocketDisconnect, BackgroundTasks, Security
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, AsyncIterator, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio
import json
import time
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SECRET_KEY = "demo-secret-key"  # Use environment variable in production
API_KEYS = {
    "sk-demo-key-123": {"user": "demo_user", "tier": "pro"},
}


# ==============================================================================
# MIDDLEWARE
# ==============================================================================

class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID for tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        
        logger.info(f"[{request_id}] {response.status_code} ({duration:.3f}s)")
        return response


# ==============================================================================
# LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Application starting...")
    yield
    logger.info("[SHUTDOWN] Application shutting down...")


# ==============================================================================
# APPLICATION
# ==============================================================================

app = FastAPI(
    title="FastAPI Advanced Patterns",
    description="Middleware, streaming, WebSockets, auth",
    version="1.0.0",
    lifespan=lifespan,
)

# Register middleware (order: first = outermost)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)


# ==============================================================================
# LLM STREAMING HELPERS
# ==============================================================================

async def stream_llm_tokens(message: str) -> AsyncIterator[str]:
    """Simulate streaming LLM response token by token."""
    response = f"I received your message: {message}. This is a streaming response."
    for word in response.split():
        await asyncio.sleep(0.05)
        yield word + " "


# ==============================================================================
# SSE STREAMING ENDPOINTS
# ==============================================================================

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4"


async def generate_sse_events(message: str) -> AsyncIterator[str]:
    """Generate SSE-formatted events."""
    async for token in stream_llm_tokens(message):
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/stream")
async def stream_chat(request: ChatRequest):
    """SSE streaming endpoint."""
    return StreamingResponse(
        generate_sse_events(request.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ==============================================================================
# WEBSOCKET CHAT
# ==============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
    
    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self.connections[client_id] = ws
    
    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)
    
    async def send(self, message: dict, client_id: str):
        if ws := self.connections.get(client_id):
            await ws.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/chat/{client_id}")
async def websocket_chat(websocket: WebSocket, client_id: str):
    """WebSocket chat with streaming responses."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # Stream response
            async for token in stream_llm_tokens(request.get("message", "")):
                await websocket.send_json({"token": token})
            
            await websocket.send_json({"done": True})
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)


# ==============================================================================
# BACKGROUND TASKS
# ==============================================================================

async def log_usage(user_id: str, tokens: int):
    """Log usage asynchronously (runs after response)."""
    await asyncio.sleep(0.1)
    logger.info(f"[USAGE] user={user_id} tokens={tokens}")


@app.post("/v1/chat-with-logging")
async def chat_with_logging(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Chat endpoint with background logging."""
    response = f"Echo: {request.message}"
    tokens = len(request.message.split()) + len(response.split())
    
    # Schedule background task
    background_tasks.add_task(log_usage, "demo_user", tokens)
    
    return {"response": response, "tokens": tokens}


# ==============================================================================
# API KEY AUTHENTICATION
# ==============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Verify API key."""
    if not api_key:
        raise HTTPException(401, {"code": "missing_key", "message": "API key required"})
    if api_key not in API_KEYS:
        raise HTTPException(401, {"code": "invalid_key", "message": "Invalid API key"})
    return API_KEYS[api_key]


@app.get("/v1/protected/apikey")
async def protected_with_apikey(user: dict = Depends(verify_api_key)):
    """Protected endpoint using API key."""
    return {"message": f"Hello, {user['user']}!", "tier": user["tier"]}


# ==============================================================================
# JWT AUTHENTICATION
# ==============================================================================

security = HTTPBearer(auto_error=False)

# Simple JWT implementation (use PyJWT in production)
def create_token(user_id: str) -> str:
    """Create a simple token (use real JWT in production)."""
    payload = {"sub": user_id, "exp": (datetime.utcnow() + timedelta(hours=1)).isoformat()}
    return f"token.{user_id}.{payload['exp']}"


def verify_jwt(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """Verify JWT token."""
    if not credentials:
        raise HTTPException(401, {"code": "missing_token", "message": "Token required"})
    
    token = credentials.credentials
    if not token.startswith("token."):
        raise HTTPException(401, {"code": "invalid_token", "message": "Invalid token"})
    
    parts = token.split(".")
    if len(parts) < 2:
        raise HTTPException(401, {"code": "invalid_token", "message": "Invalid token"})
    
    return parts[1]  # User ID


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> str:
    """Get current user from JWT."""
    return verify_jwt(credentials)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login and get token."""
    if request.username == "demo" and request.password == "demo":
        token = create_token(request.username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(401, {"code": "invalid_credentials", "message": "Invalid credentials"})


@app.get("/v1/protected/jwt")
async def protected_with_jwt(user_id: str = Depends(get_current_user)):
    """Protected endpoint using JWT."""
    return {"message": f"Hello, {user_id}!"}


# ==============================================================================
# RATE LIMITING
# ==============================================================================

class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.rpm = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
    
    def check(self, user_id: str) -> int:
        """Check rate limit. Returns remaining requests or raises 429."""
        now = datetime.now()
        window = now - timedelta(minutes=1)
        
        user_reqs = [t for t in self.requests.get(user_id, []) if t > window]
        self.requests[user_id] = user_reqs
        
        if len(user_reqs) >= self.rpm:
            raise HTTPException(
                429,
                {"code": "rate_limit", "message": "Too many requests"},
                headers={"Retry-After": "60"}
            )
        
        self.requests[user_id].append(now)
        return self.rpm - len(user_reqs) - 1


rate_limiter = RateLimiter(requests_per_minute=5)


@app.get("/v1/rate-limited")
async def rate_limited_endpoint():
    """Rate-limited endpoint (5 requests/minute)."""
    remaining = rate_limiter.check("demo_user")
    return {"message": "Success", "remaining": remaining}


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.get("/")
async def health():
    """Health check."""
    return {"status": "healthy", "version": "1.0.0"}


# ==============================================================================
# RUN INSTRUCTIONS
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ===== FastAPI Advanced Patterns =====
    
    Run: uvicorn 02_fastapi_advanced:app --reload
    Docs: http://localhost:8000/docs
    
    Test endpoints:
    - GET  /                          # Health check
    - POST /v1/stream                 # SSE streaming
    - WS   /ws/chat/{client_id}       # WebSocket chat
    - POST /v1/chat-with-logging      # Background tasks
    - GET  /v1/protected/apikey       # API key auth (X-API-Key: sk-demo-key-123)
    - POST /auth/login                # Get JWT (username: demo, password: demo)
    - GET  /v1/protected/jwt          # JWT auth
    - GET  /v1/rate-limited           # Rate limiting (5/min)
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
