"""
FastAPI Fundamentals Example
============================
A complete FastAPI application demonstrating:
- Routes, path/query params
- Pydantic request/response models
- Async endpoints
- Dependency injection
- Lifespan management
- Error handling

Run with: uvicorn 01_fastapi_basics:app --reload
Visit: http://localhost:8000/docs
"""

from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Protocol
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
import asyncio


# ==============================================================================
# CONFIGURATION & SETTINGS
# ==============================================================================

class Settings(BaseModel):
    """Application settings loaded from environment."""
    app_name: str = "AI Chat API"
    environment: str = "development"
    default_model: str = "gpt-4"
    api_key: str = "sk-demo-key"


@lru_cache()
def get_settings() -> Settings:
    """Load settings once and cache."""
    return Settings()


# ==============================================================================
# LLM CLIENT PROTOCOL & IMPLEMENTATIONS
# ==============================================================================

class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def complete(self, messages: List[Dict[str, str]], model: str) -> str: ...


class MockLLMClient:
    """Mock LLM client for development/testing."""
    
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_count = 0
    
    async def complete(self, messages: List[Dict[str, str]], model: str) -> str:
        """Simulate LLM response with delay."""
        await asyncio.sleep(self.delay)
        self.call_count += 1
        last_msg = messages[-1]["content"] if messages else "empty"
        return f"[{model}] Response #{self.call_count}: {last_msg[:50]}"


# Global client instance (initialized in lifespan)
llm_client: Optional[MockLLMClient] = None


# ==============================================================================
# LIFESPAN MANAGEMENT
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.
    
    - Code before 'yield' runs on startup
    - Code after 'yield' runs on shutdown
    """
    global llm_client
    
    # Startup
    print("[STARTUP] Starting AI Chat API...")
    llm_client = MockLLMClient()
    print("[STARTUP] LLM client initialized")
    
    yield  # Application runs here
    
    # Shutdown
    print("[SHUTDOWN] Shutting down...")
    llm_client = None
    print("[SHUTDOWN] Cleanup complete")


# ==============================================================================
# APPLICATION
# ==============================================================================

app = FastAPI(
    title="AI Chat API",
    description="FastAPI fundamentals with LLM examples",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class ChatMessage(BaseModel):
    """Single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=100000)
    
    class Config:
        json_schema_extra = {
            "example": {"role": "user", "content": "Hello!"}
        }


class ChatRequest(BaseModel):
    """Request to create a chat completion."""
    model: str = Field("gpt-4", description="Model to use")
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatResponse(BaseModel):
    """Response from chat completion."""
    id: str
    model: str
    created: datetime
    choices: List[ChatChoice]
    usage: TokenUsage


class ErrorDetail(BaseModel):
    """Structured error detail."""
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: ErrorDetail


# ==============================================================================
# DEPENDENCY INJECTION
# ==============================================================================

def get_llm_client() -> LLMClient:
    """Dependency to get the LLM client."""
    if llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return llm_client


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"id": "gpt-4", "provider": "openai"},
            {"id": "gpt-3.5-turbo", "provider": "openai"},
            {"id": "claude-3-opus", "provider": "anthropic"},
            {"id": "llama2", "provider": "ollama"},
        ]
    }


@app.get("/models/{model_id}")
async def get_model(
    model_id: str = Path(..., description="Model identifier")
):
    """Get model information."""
    models = {
        "gpt-4": {"provider": "openai", "context_length": 8192},
        "gpt-3.5-turbo": {"provider": "openai", "context_length": 4096},
        "claude-3-opus": {"provider": "anthropic", "context_length": 200000},
    }
    
    if model_id not in models:
        raise HTTPException(404, f"Model '{model_id}' not found")
    
    return {"id": model_id, **models[model_id]}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    """
    Create a chat completion (OpenAI-compatible endpoint).
    
    This demonstrates:
    - Pydantic request validation
    - Dependency injection for LLM client
    - Async endpoint for I/O-bound operations
    """
    # Convert Pydantic models to dicts for LLM
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Call LLM (async)
    response_content = await llm.complete(messages, request.model)
    
    # Build response
    return ChatResponse(
        id=f"chatcmpl-{datetime.now().timestamp()}",
        model=request.model,
        created=datetime.now(),
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(response_content.split()),
            total_tokens=0,  # Will be calculated
        ),
    )


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str = Path(..., min_length=1),
    role: Optional[str] = Query(None, description="Filter by role"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    """
    Get messages from a conversation.
    
    Demonstrates:
    - Path parameters (conversation_id)
    - Query parameters with validation (role, limit, skip)
    """
    # In production: fetch from database
    messages = [
        {"role": "user", "content": "Hello!", "index": 0},
        {"role": "assistant", "content": "Hi there!", "index": 1},
        {"role": "user", "content": "How are you?", "index": 2},
    ]
    
    # Apply filters
    if role:
        messages = [m for m in messages if m["role"] == role]
    
    messages = messages[skip:skip + limit]
    
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "total": len(messages),
    }


# ==============================================================================
# BATCH ENDPOINT (ASYNC CONCURRENCY)
# ==============================================================================

class BatchRequest(BaseModel):
    """Batch completion request."""
    prompts: List[str] = Field(..., min_length=1, max_length=10)
    model: str = "gpt-4"


class BatchResponse(BaseModel):
    """Batch completion response."""
    responses: List[str]
    total_time_ms: float


@app.post("/v1/batch", response_model=BatchResponse)
async def batch_complete(
    request: BatchRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    """
    Process multiple prompts concurrently.
    
    Demonstrates async concurrency with asyncio.gather.
    3 prompts * 100ms each = ~100ms total (not 300ms!)
    """
    import time
    start = time.time()
    
    # Create tasks for concurrent execution
    tasks = [
        llm.complete([{"role": "user", "content": p}], request.model)
        for p in request.prompts
    ]
    
    # Run all concurrently
    responses = await asyncio.gather(*tasks)
    
    elapsed_ms = (time.time() - start) * 1000
    
    return BatchResponse(
        responses=list(responses),
        total_time_ms=elapsed_ms,
    )


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

class ModelNotFoundError(Exception):
    """Custom exception for model not found."""
    def __init__(self, model: str):
        self.model = model
        self.message = f"Model '{model}' not found"


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc: ModelNotFoundError):
    """Handle model not found errors."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error=ErrorDetail(code="model_not_found", message=exc.message)
        ).model_dump(),
    )


# ==============================================================================
# RUN INSTRUCTIONS
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    FastAPI Fundamentals                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Run with: uvicorn 01_fastapi_basics:app --reload            ║
    ║  Docs at:  http://localhost:8000/docs                        ║
    ║                                                               ║
    ║  Example requests:                                            ║
    ║  - GET  http://localhost:8000/                               ║
    ║  - GET  http://localhost:8000/models                         ║
    ║  - POST http://localhost:8000/v1/chat/completions            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
