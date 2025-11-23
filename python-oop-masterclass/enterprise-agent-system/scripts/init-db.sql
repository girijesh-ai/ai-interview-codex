-- ============================================================================
-- Database Initialization Script
-- ============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- Checkpointing Tables (for LangGraph)
-- ============================================================================

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at DESC);

-- ============================================================================
-- Request Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS customer_requests (
    request_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT,
    priority INTEGER DEFAULT 2,
    initial_message TEXT NOT NULL,
    solution TEXT,
    escalation_reason TEXT,
    requires_approval BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_requests_customer ON customer_requests(customer_id);
CREATE INDEX IF NOT EXISTS idx_requests_status ON customer_requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_created ON customer_requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_priority ON customer_requests(priority DESC);

-- ============================================================================
-- Decision Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES customer_requests(request_id),
    agent_type TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decisions_request ON agent_decisions(request_id);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent_type);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON agent_decisions(created_at DESC);

-- ============================================================================
-- Message Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id TEXT NOT NULL REFERENCES customer_requests(request_id),
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_request ON messages(request_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);

-- ============================================================================
-- Analytics Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    dimensions JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);

CREATE TABLE IF NOT EXISTS agent_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type TEXT NOT NULL,
    request_id TEXT,
    duration_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_performance_agent ON agent_performance(agent_type);
CREATE INDEX IF NOT EXISTS idx_performance_created ON agent_performance(created_at DESC);

-- ============================================================================
-- Event Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    request_id TEXT,
    payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);

-- ============================================================================
-- Customer Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    tier TEXT DEFAULT 'standard',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers(tier);

-- ============================================================================
-- Triggers
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_customer_requests_updated_at
    BEFORE UPDATE ON customer_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views
-- ============================================================================

-- Request summary view
CREATE OR REPLACE VIEW request_summary AS
SELECT
    r.request_id,
    r.customer_id,
    r.status,
    r.category,
    r.priority,
    r.created_at,
    COUNT(DISTINCT d.decision_id) as decision_count,
    COUNT(DISTINCT m.message_id) as message_count,
    EXTRACT(EPOCH FROM (COALESCE(r.completed_at, CURRENT_TIMESTAMP) - r.created_at)) as duration_seconds
FROM customer_requests r
LEFT JOIN agent_decisions d ON r.request_id = d.request_id
LEFT JOIN messages m ON r.request_id = m.request_id
GROUP BY r.request_id;

-- Agent performance view
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT
    agent_type,
    COUNT(*) as total_executions,
    AVG(duration_ms) as avg_duration_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
    MAX(created_at) as last_execution
FROM agent_performance
GROUP BY agent_type;

-- ============================================================================
-- Sample Data (Optional - Comment out for production)
-- ============================================================================

-- Insert sample customer
-- INSERT INTO customers (customer_id, name, email, tier) VALUES
-- ('cust-sample-1', 'Sample Customer', 'sample@example.com', 'premium')
-- ON CONFLICT (customer_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
