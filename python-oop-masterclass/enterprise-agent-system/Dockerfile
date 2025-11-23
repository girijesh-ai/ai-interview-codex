# ============================================================================
# Multi-stage Dockerfile for Enterprise Agent System
# ============================================================================

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# ============================================================================
# Development stage
# ============================================================================

FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Testing stage
# ============================================================================

FROM development as testing

# Switch to root for package installation
USER root

# Install additional testing tools
RUN pip install pytest-xdist pytest-timeout

# Copy test files
COPY --chown=appuser:appuser tests/ tests/

# Switch back to non-root user
USER appuser

# Run tests by default
CMD ["pytest", "-v", "--cov=src", "--cov-report=term-missing"]

# ============================================================================
# Builder stage for production dependencies
# ============================================================================

FROM base as builder

# Copy requirements
COPY requirements.txt ./

# Install dependencies to a virtual environment
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-warn-script-location -r requirements.txt

# ============================================================================
# Production stage
# ============================================================================

FROM base as production

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
