#!/bin/bash

# ============================================================================
# Development Environment Startup Script
# ============================================================================

set -e

echo "Starting Enterprise Agent System - Development Environment"
echo "=========================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please update .env with your configuration"
fi

# Check for required environment variables
if ! grep -q "OPENAI_API_KEY=your-openai-api-key-here" .env; then
    echo "Warning: Please set your OPENAI_API_KEY in .env file"
fi

# Start infrastructure services
echo ""
echo "Starting infrastructure services..."
docker-compose up -d redis postgres weaviate zookeeper kafka

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."

# Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "Redis: Ready"
else
    echo "Redis: Not ready"
fi

# PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "PostgreSQL: Ready"
else
    echo "PostgreSQL: Not ready"
fi

# Weaviate
if curl -f http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "Weaviate: Ready"
else
    echo "Weaviate: Not ready"
fi

# Run database migrations
echo ""
echo "Running database initialization..."
docker-compose exec -T postgres psql -U postgres -d agent_system -f /docker-entrypoint-initdb.d/init.sql > /dev/null 2>&1 || true

# Start application services
echo ""
echo "Starting application services..."
docker-compose up -d api celery-worker celery-beat

# Start monitoring services (optional)
echo ""
read -p "Start monitoring services (Prometheus, Grafana, Flower)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose up -d prometheus grafana flower
fi

# Start admin tools (optional)
echo ""
read -p "Start admin tools (Redis Commander, PgAdmin, Kafka UI)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose up -d redis-commander pgadmin kafka-ui
fi

echo ""
echo "=========================================================="
echo "Services Started Successfully!"
echo "=========================================================="
echo ""
echo "Application:"
echo "  - API:          http://localhost:8000"
echo "  - API Docs:     http://localhost:8000/docs"
echo "  - Health:       http://localhost:8000/health"
echo ""
echo "Infrastructure:"
echo "  - Redis:        localhost:6379"
echo "  - PostgreSQL:   localhost:5432"
echo "  - Weaviate:     http://localhost:8080"
echo "  - Kafka:        localhost:9092"
echo ""
echo "Monitoring (if started):"
echo "  - Prometheus:   http://localhost:9090"
echo "  - Grafana:      http://localhost:3000 (admin/admin)"
echo "  - Flower:       http://localhost:5555 (admin/admin)"
echo ""
echo "Admin Tools (if started):"
echo "  - Redis Commander: http://localhost:8081"
echo "  - PgAdmin:         http://localhost:5050 (admin@example.com/admin)"
echo "  - Kafka UI:        http://localhost:8082"
echo ""
echo "To view logs:   docker-compose logs -f [service-name]"
echo "To stop all:    docker-compose down"
echo "=========================================================="
