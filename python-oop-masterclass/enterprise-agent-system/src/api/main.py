"""
FastAPI Application

Main API application with REST endpoints and WebSocket support.

Demonstrates:
- FastAPI application structure
- REST API design
- WebSocket integration
- Dependency injection
- Error handling
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

from .models import (
    CreateRequestModel,
    UpdateRequestModel,
    RequestResponseModel,
    RequestListResponse,
    ApprovalRequestModel,
    ApprovalResponseModel,
    SessionModel,
    MetricsModel,
    HealthCheckModel,
    ComponentHealthModel,
    ErrorModel,
    WebSocketMessage,
    SubscriptionModel,
    PaginationParams,
    RequestFilterModel
)

logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Enterprise AI Agent System API",
    description="Production-ready AI agent system with LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

# These would be actual service instances in production
class Dependencies:
    """Dependency injection container."""

    @staticmethod
    async def get_memory_manager():
        """Get memory manager instance."""
        # Would return actual MemoryManager
        return None

    @staticmethod
    async def get_event_producer():
        """Get event producer instance."""
        # Would return actual EventProducer
        return None

    @staticmethod
    async def get_workflow_integration():
        """Get workflow integration instance."""
        # Would return actual WorkflowIntegration
        return None


# ============================================================================
# REQUEST ENDPOINTS
# ============================================================================

@app.post(
    "/requests",
    response_model=RequestResponseModel,
    status_code=status.HTTP_201_CREATED,
    tags=["Requests"]
)
async def create_request(request: CreateRequestModel) -> RequestResponseModel:
    """Create a new customer request.

    Args:
        request: Request data

    Returns:
        Created request
    """
    try:
        logger.info(f"Creating request for customer: {request.customer_id}")

        # This would integrate with the actual workflow
        # For now, return mock response
        response = RequestResponseModel(
            request_id=f"req-{datetime.now().timestamp()}",
            customer_id=request.customer_id,
            status="processing",
            category=request.category.value if request.category else "general",
            priority=request.priority.value if request.priority else 2,
            message=request.message,
            solution=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=request.metadata
        )

        return response

    except Exception as e:
        logger.error(f"Error creating request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/requests/{request_id}",
    response_model=RequestResponseModel,
    tags=["Requests"]
)
async def get_request(request_id: str) -> RequestResponseModel:
    """Get request by ID.

    Args:
        request_id: Request ID

    Returns:
        Request data
    """
    try:
        logger.info(f"Getting request: {request_id}")

        # This would query from database/state
        # For now, return mock response
        response = RequestResponseModel(
            request_id=request_id,
            customer_id="cust-123",
            status="completed",
            category="account",
            priority=2,
            message="How do I reset my password?",
            solution="You can reset your password by visiting...",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        return response

    except Exception as e:
        logger.error(f"Error getting request: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request not found: {request_id}"
        )


@app.get(
    "/requests",
    response_model=RequestListResponse,
    tags=["Requests"]
)
async def list_requests(
    filters: RequestFilterModel = Depends(),
    pagination: PaginationParams = Depends()
) -> RequestListResponse:
    """List requests with filtering and pagination.

    Args:
        filters: Filter parameters
        pagination: Pagination parameters

    Returns:
        Paginated request list
    """
    try:
        logger.info(f"Listing requests (page={pagination.page}, size={pagination.page_size})")

        # This would query from database with filters
        # For now, return mock response
        requests = [
            RequestResponseModel(
                request_id=f"req-{i}",
                customer_id="cust-123",
                status="completed",
                category="account",
                priority=2,
                message=f"Request {i}",
                solution="Solution provided",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(pagination.page_size)
        ]

        return RequestListResponse(
            requests=requests,
            total=100,
            page=pagination.page,
            page_size=pagination.page_size
        )

    except Exception as e:
        logger.error(f"Error listing requests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.patch(
    "/requests/{request_id}",
    response_model=RequestResponseModel,
    tags=["Requests"]
)
async def update_request(
    request_id: str,
    update: UpdateRequestModel
) -> RequestResponseModel:
    """Update request.

    Args:
        request_id: Request ID
        update: Update data

    Returns:
        Updated request
    """
    try:
        logger.info(f"Updating request: {request_id}")

        # This would update in database/state
        # For now, return mock response
        response = RequestResponseModel(
            request_id=request_id,
            customer_id="cust-123",
            status="processing",
            category="account",
            priority=update.priority.value if update.priority else 2,
            message=update.message or "Original message",
            solution=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=update.metadata
        )

        return response

    except Exception as e:
        logger.error(f"Error updating request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# APPROVAL ENDPOINTS
# ============================================================================

@app.post(
    "/requests/{request_id}/approve",
    response_model=ApprovalResponseModel,
    tags=["Approvals"]
)
async def approve_request(
    request_id: str,
    approval: ApprovalRequestModel
) -> ApprovalResponseModel:
    """Approve or reject a request.

    Args:
        request_id: Request ID
        approval: Approval decision

    Returns:
        Approval result
    """
    try:
        logger.info(f"Processing approval for request: {request_id}")

        # This would update the workflow state
        # For now, return mock response
        response = ApprovalResponseModel(
            request_id=request_id,
            approved=approval.approved,
            approver=approval.approver,
            notes=approval.notes,
            approved_at=datetime.now()
        )

        return response

    except Exception as e:
        logger.error(f"Error processing approval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@app.get(
    "/sessions/{session_id}",
    response_model=SessionModel,
    tags=["Sessions"]
)
async def get_session(session_id: str) -> SessionModel:
    """Get session data.

    Args:
        session_id: Session ID

    Returns:
        Session data
    """
    try:
        logger.info(f"Getting session: {session_id}")

        # This would query from Redis
        # For now, return mock response
        response = SessionModel(
            session_id=session_id,
            customer_id="cust-123",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            data={"key": "value"}
        )

        return response

    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.get(
    "/metrics",
    response_model=MetricsModel,
    tags=["Metrics"]
)
async def get_metrics() -> MetricsModel:
    """Get system metrics.

    Returns:
        System metrics
    """
    try:
        logger.info("Getting system metrics")

        # This would aggregate from various sources
        # For now, return mock response
        response = MetricsModel(
            total_requests=1000,
            completed_requests=950,
            failed_requests=50,
            avg_resolution_time_seconds=125.5,
            cache_hit_rate=0.87,
            active_sessions=42,
            timestamp=datetime.now()
        )

        return response

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get(
    "/health",
    response_model=HealthCheckModel,
    tags=["Health"]
)
async def health_check() -> HealthCheckModel:
    """Check system health.

    Returns:
        Health status
    """
    try:
        # Check all components
        components = {
            "api": ComponentHealthModel(status="healthy", latency_ms=1.0),
            "langgraph": ComponentHealthModel(status="healthy", latency_ms=15.0),
            "vector_db": ComponentHealthModel(status="healthy", latency_ms=8.0),
            "redis": ComponentHealthModel(status="healthy", latency_ms=2.0),
            "kafka": ComponentHealthModel(status="healthy", latency_ms=5.0),
            "celery": ComponentHealthModel(status="healthy", latency_ms=3.0)
        }

        overall_status = "healthy" if all(
            c.status == "healthy" for c in components.values()
        ) else "degraded"

        return HealthCheckModel(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheckModel(
            overall_status="unhealthy",
            components={},
            timestamp=datetime.now()
        )


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, SubscriptionModel] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        """Connect client.

        Args:
            client_id: Client ID
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")

    def disconnect(self, client_id: str):
        """Disconnect client.

        Args:
            client_id: Client ID
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client disconnected: {client_id}")

    async def send_message(self, client_id: str, message: WebSocketMessage):
        """Send message to client.

        Args:
            client_id: Client ID
            message: Message to send
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message.model_dump())

    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all clients.

        Args:
            message: Message to broadcast
        """
        for websocket in self.active_connections.values():
            await websocket.send_json(message.model_dump())


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates.

    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await manager.connect(client_id, websocket)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)

            # Handle message
            if message.type == "subscribe":
                # Subscribe to events
                subscription = SubscriptionModel(**message.data)
                manager.subscriptions[client_id] = subscription
                logger.info(f"Client {client_id} subscribed to: {subscription}")

                # Send confirmation
                await manager.send_message(
                    client_id,
                    WebSocketMessage(
                        type="subscribed",
                        data={"status": "success"}
                    )
                )

            elif message.type == "unsubscribe":
                # Unsubscribe
                if client_id in manager.subscriptions:
                    del manager.subscriptions[client_id]
                logger.info(f"Client {client_id} unsubscribed")

            elif message.type == "ping":
                # Respond to ping
                await manager.send_message(
                    client_id,
                    WebSocketMessage(type="pong", data={})
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorModel(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorModel(
            error="Internal server error",
            detail=str(exc),
            code="INTERNAL_ERROR"
        ).model_dump()
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Starting Enterprise AI Agent System API")
    # Initialize services here


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Shutting down Enterprise AI Agent System API")
    # Cleanup services here


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Enterprise AI Agent System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
