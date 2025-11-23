"""
Production-Ready Configuration Management

Validates all required environment variables at startup to fail fast.
Uses Pydantic Settings for type validation and defaults.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with validation.

    All settings are loaded from environment variables.
    Required variables must be set or application will fail to start.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================

    app_name: str = Field(
        default="Enterprise Agent System",
        description="Application name"
    )

    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )

    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v

    # ========================================================================
    # API SETTINGS
    # ========================================================================

    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )

    api_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of API workers"
    )

    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ========================================================================
    # LLM SETTINGS
    # ========================================================================

    openai_api_key: str = Field(
        ...,  # Required
        description="OpenAI API key"
    )

    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Default LLM model"
    )

    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )

    llm_max_tokens: int = Field(
        default=2000,
        ge=1,
        le=128000,
        description="Max tokens per LLM response"
    )

    llm_timeout: int = Field(
        default=60,
        ge=1,
        le=300,
        description="LLM request timeout in seconds"
    )

    # ========================================================================
    # DATABASE SETTINGS
    # ========================================================================

    postgres_url: str = Field(
        ...,  # Required
        description="PostgreSQL connection URL"
    )

    postgres_pool_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="PostgreSQL connection pool size"
    )

    postgres_max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="PostgreSQL max overflow connections"
    )

    # ========================================================================
    # REDIS SETTINGS
    # ========================================================================

    redis_url: str = Field(
        ...,  # Required
        description="Redis connection URL"
    )

    redis_max_connections: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Redis max connections"
    )

    redis_socket_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Redis socket timeout in seconds"
    )

    # ========================================================================
    # VECTOR DATABASE SETTINGS
    # ========================================================================

    weaviate_url: str = Field(
        ...,  # Required
        description="Weaviate vector database URL"
    )

    weaviate_api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key (if required)"
    )

    # ========================================================================
    # KAFKA SETTINGS
    # ========================================================================

    kafka_bootstrap_servers: str = Field(
        ...,  # Required
        description="Kafka bootstrap servers"
    )

    kafka_group_id: str = Field(
        default="enterprise-agent-system",
        description="Kafka consumer group ID"
    )

    @field_validator("kafka_bootstrap_servers", mode="before")
    @classmethod
    def parse_kafka_servers(cls, v):
        """Parse Kafka servers from comma-separated string."""
        if isinstance(v, str):
            return v  # Keep as string, parsed later
        return v

    # ========================================================================
    # CELERY SETTINGS
    # ========================================================================

    celery_broker_url: str = Field(
        ...,  # Required
        description="Celery broker URL"
    )

    celery_result_backend: str = Field(
        ...,  # Required
        description="Celery result backend URL"
    )

    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================

    secret_key: str = Field(
        ...,  # Required
        min_length=32,
        description="Secret key for JWT and encryption"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    jwt_expiration_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="JWT token expiration in minutes"
    )

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    rate_limit_per_minute: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Max requests per minute per user"
    )

    # ========================================================================
    # MONITORING SETTINGS
    # ========================================================================

    jaeger_host: str = Field(
        default="localhost",
        description="Jaeger tracing host"
    )

    jaeger_port: int = Field(
        default=6831,
        ge=1,
        le=65535,
        description="Jaeger tracing port"
    )

    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    # ========================================================================
    # LOGGING SETTINGS
    # ========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v_upper

    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = {"json", "text"}
        if v not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v

    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================

    feature_auth_enabled: bool = Field(
        default=False,
        description="Enable authentication"
    )

    feature_rate_limiting_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    feature_tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton).

    Returns:
        Settings: Application settings instance

    Raises:
        ValidationError: If required environment variables are missing
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def validate_settings() -> None:
    """Validate all settings at application startup.

    This should be called early in application startup to fail fast
    if configuration is invalid.

    Raises:
        ValidationError: If any required settings are missing or invalid
    """
    settings = get_settings()

    # Additional custom validations
    if settings.environment == "production":
        # Production-specific checks
        if settings.debug:
            raise ValueError("Debug mode cannot be enabled in production")

        if settings.secret_key == "your-secret-key-here":
            raise ValueError("Must set a real SECRET_KEY in production")

        if "*" in settings.cors_origins or "localhost" in str(settings.cors_origins):
            raise ValueError("Cannot use wildcard or localhost CORS in production")

    # Log configuration (don't log secrets!)
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "Configuration loaded",
        extra={
            "environment": settings.environment,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "llm_model": settings.llm_model,
        }
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().environment == "development"


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().environment == "production"


def is_staging() -> bool:
    """Check if running in staging environment."""
    return get_settings().environment == "staging"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# In main application startup:
from src.config import validate_settings, get_settings

# Validate configuration early (fails fast if misconfigured)
try:
    validate_settings()
except ValidationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

# Use settings throughout application
settings = get_settings()
llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    temperature=settings.llm_temperature
)
"""
