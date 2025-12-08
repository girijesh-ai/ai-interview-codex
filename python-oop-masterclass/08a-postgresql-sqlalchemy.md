# Module 08a: PostgreSQL + SQLAlchemy for AI Applications

> **Level**: Intermediate → Advanced | **Time**: 12-15 hours | **Prerequisites**: Module 07

## Overview

This module covers database patterns for AI applications using PostgreSQL and SQLAlchemy 2.0 with async support. You'll learn to store conversations, user data, embeddings metadata, and implement production-grade data access patterns.

### Why This Stack for AI?

```mermaid
flowchart LR
    subgraph "AI Data Needs"
        C[Conversations] --> PG[(PostgreSQL)]
        U[Users/Auth] --> PG
        M[Model Configs] --> PG
        L[Usage Logs] --> PG
    end
    
    subgraph "Why PostgreSQL"
        R[Relational] --> J[JSONB for flexibility]
        T[Transactions] --> A[ACID compliance]
        S[Scalable] --> P[Proven at scale]
    end
```

| Component | Purpose |
|-----------|---------|
| **PostgreSQL** | Reliable, ACID-compliant relational database |
| **SQLAlchemy 2.0** | Modern async ORM with type hints |
| **asyncpg** | High-performance async PostgreSQL driver |
| **Alembic** | Database migrations |

> **Interview Insight**: "Why PostgreSQL for AI apps?" — JSONB for flexible LLM response storage, full-text search for conversation history, proven reliability at scale.

### Key Concepts to Master

| Concept | What It Is | Interview Importance |
|---------|-----------|----------------------|
| **Mapped[T]** | Type-annotated column definitions | ⭐⭐⭐ SQLAlchemy 2.0 syntax |
| **AsyncSession** | Non-blocking DB operations | ⭐⭐⭐ FastAPI integration |
| **Repository Pattern** | Abstract data access layer | ⭐⭐⭐ Testability |
| **Unit of Work** | Manage transactions | ⭐⭐ Multi-table operations |
| **Alembic** | Schema migrations | ⭐⭐ Production deployments |

> **Common Pitfall**: Forgetting `expire_on_commit=False` in async sessions causes "greenlet_spawn" errors when accessing model attributes after commit.

---


## Part 1: SQLAlchemy 2.0 Setup

### Understanding the New Style

SQLAlchemy 2.0 introduced significant changes:
- **Declarative mapping** with type annotations
- **Async-first** design with `AsyncSession`
- **Better type checking** support

```mermaid
flowchart TB
    subgraph "SQLAlchemy 2.0 Architecture"
        E[Engine] --> P[Connection Pool]
        E --> S[SessionMaker]
        S --> AS[AsyncSession]
        AS --> Q[Queries]
        Q --> M[Models]
    end
```

### Database Configuration

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from pydantic_settings import BaseSettings
from functools import lru_cache


# ==============================================================================
# SETTINGS
# ==============================================================================

class DatabaseSettings(BaseSettings):
    """Database configuration from environment."""
    
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/aidb"
    pool_size: int = 5
    max_overflow: int = 10
    echo_sql: bool = False
    
    class Config:
        env_prefix = "DB_"


@lru_cache()
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings()


# ==============================================================================
# ENGINE & SESSION
# ==============================================================================

def create_engine(settings: DatabaseSettings):
    """Create async database engine."""
    return create_async_engine(
        settings.database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        echo=settings.echo_sql,
        # Important for async
        pool_pre_ping=True,  # Verify connections before use
    )


# Create global engine and session factory
settings = get_db_settings()
engine = create_engine(settings)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects usable after commit
)


# ==============================================================================
# BASE MODEL
# ==============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass
```

---

## Part 2: Defining Models for AI

### Core Models

```python
from sqlalchemy import (
    String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, Index,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
from typing import Optional, List
import uuid


class Base(DeclarativeBase):
    """Base with common columns."""
    pass


# ==============================================================================
# USER MODEL
# ==============================================================================

class User(Base):
    """User account for AI platform."""
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(String(50), default="free")
    
    # Usage tracking
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens_limit: Mapped[int] = mapped_column(Integer, default=100000)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )


# ==============================================================================
# CONVERSATION MODEL
# ==============================================================================

class Conversation(Base):
    """Chat conversation with message history."""
    
    __tablename__ = "conversations"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE")
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model: Mapped[str] = mapped_column(String(100), default="gpt-4")
    
    # Store conversation settings as JSONB
    settings: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False
    )
    
    # Token usage for this conversation
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )


# ==============================================================================
# MESSAGE MODEL
# ==============================================================================

class Message(Base):
    """Individual message in a conversation."""
    
    __tablename__ = "messages"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    
    # Token counts
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Store tool calls, function results as JSONB
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ==============================================================================
# API KEY MODEL
# ==============================================================================

class APIKey(Base):
    """User API keys for authentication."""
    
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE")
    )
    
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="api_keys")
```

---

## Part 3: Repository Pattern

### Why Repository Pattern?

The Repository Pattern abstracts data access, providing:
1. **Testability**: Mock repositories for unit tests
2. **Flexibility**: Swap databases without changing business logic
3. **Single Responsibility**: Data access logic in one place

```mermaid
flowchart TB
    subgraph "Without Repository"
        E1[Endpoint] --> DB1[(Database)]
        E2[Endpoint] --> DB1
        Note1[SQL scattered everywhere]
    end
    
    subgraph "With Repository"
        E3[Endpoint] --> R[Repository]
        E4[Endpoint] --> R
        R --> DB2[(Database)]
        Note2[Centralized data access]
    end
```

### Generic Repository

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Type
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

T = TypeVar("T", bound=Base)


class Repository(ABC, Generic[T]):
    """Abstract base repository."""
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[T]:
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        pass


class SQLAlchemyRepository(Repository[T]):
    """SQLAlchemy implementation of Repository."""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: UUID) -> Optional[T]:
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())
    
    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: T) -> T:
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0


# ==============================================================================
# SPECIALIZED REPOSITORIES
# ==============================================================================

class UserRepository(SQLAlchemyRepository[User]):
    """Repository for User operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def update_tokens(self, user_id: UUID, tokens: int) -> None:
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(tokens_used=User.tokens_used + tokens)
        )


class ConversationRepository(SQLAlchemyRepository[Conversation]):
    """Repository for Conversation operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Conversation)
    
    async def get_user_conversations(
        self,
        user_id: UUID,
        limit: int = 20
    ) -> List[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_with_messages(
        self,
        conversation_id: UUID
    ) -> Optional[Conversation]:
        from sqlalchemy.orm import selectinload
        
        result = await self.session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()
```

---

## Part 4: Unit of Work Pattern

### Coordinating Transactions

The Unit of Work pattern manages transactions across multiple repositories:

```mermaid
sequenceDiagram
    participant E as Endpoint
    participant UoW as Unit of Work
    participant R1 as UserRepo
    participant R2 as ConvRepo
    participant DB as Database
    
    E->>UoW: async with uow:
    E->>R1: Create user
    E->>R2: Create conversation
    E->>UoW: await uow.commit()
    UoW->>DB: BEGIN TRANSACTION
    UoW->>DB: INSERT user
    UoW->>DB: INSERT conversation
    UoW->>DB: COMMIT
```

### Implementation

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator


class UnitOfWork:
    """Manages database transactions and repositories."""
    
    def __init__(self, session_factory: async_sessionmaker):
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
    
    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork not started")
        return self._session
    
    @property
    def users(self) -> UserRepository:
        return UserRepository(self.session)
    
    @property
    def conversations(self) -> ConversationRepository:
        return ConversationRepository(self.session)
    
    async def __aenter__(self) -> "UnitOfWork":
        self._session = self._session_factory()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.rollback()
        await self._session.close()
        self._session = None
    
    async def commit(self):
        await self.session.commit()
    
    async def rollback(self):
        await self.session.rollback()


# Factory function for dependency injection
def get_unit_of_work() -> UnitOfWork:
    return UnitOfWork(async_session_factory)
```

---

## Part 5: FastAPI Integration

### Dependency Injection

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List
from uuid import UUID

app = FastAPI()


# ==============================================================================
# SCHEMAS (Pydantic)
# ==============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    tier: str
    tokens_used: int
    
    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: str | None = None
    model: str = "gpt-4"


class ConversationResponse(BaseModel):
    id: UUID
    title: str | None
    model: str
    total_tokens: int
    
    class Config:
        from_attributes = True


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.post("/users", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    uow: UnitOfWork = Depends(get_unit_of_work)
):
    """Create a new user."""
    async with uow:
        # Check if email exists
        existing = await uow.users.get_by_email(data.email)
        if existing:
            raise HTTPException(400, "Email already registered")
        
        # Create user
        user = User(
            email=data.email,
            hashed_password=hash_password(data.password)  # Implement this
        )
        await uow.users.create(user)
        await uow.commit()
        
        return user


@app.get("/users/{user_id}/conversations", response_model=List[ConversationResponse])
async def get_user_conversations(
    user_id: UUID,
    uow: UnitOfWork = Depends(get_unit_of_work)
):
    """Get user's conversations."""
    async with uow:
        conversations = await uow.conversations.get_user_conversations(user_id)
        return conversations


@app.post("/users/{user_id}/conversations", response_model=ConversationResponse)
async def create_conversation(
    user_id: UUID,
    data: ConversationCreate,
    uow: UnitOfWork = Depends(get_unit_of_work)
):
    """Create a new conversation."""
    async with uow:
        # Verify user exists
        user = await uow.users.get_by_id(user_id)
        if not user:
            raise HTTPException(404, "User not found")
        
        conversation = Conversation(
            user_id=user_id,
            title=data.title,
            model=data.model
        )
        await uow.conversations.create(conversation)
        await uow.commit()
        
        return conversation
```

---

## Part 6: Alembic Migrations

### Setup

```bash
# Install alembic
pip install alembic

# Initialize (creates alembic/ folder)
alembic init alembic

# Edit alembic.ini: set sqlalchemy.url
# Edit alembic/env.py: import your models
```

### Async Configuration (alembic/env.py)

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import your models
from app.models import Base
from app.config import get_db_settings

config = context.config
settings = get_db_settings()

# Set URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Common Commands

```bash
# Create a migration
alembic revision --autogenerate -m "Add users table"

# Apply all migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history
```

---

## Summary

| Pattern | Purpose |
|---------|---------|
| **SQLAlchemy 2.0** | Modern async ORM with type hints |
| **Mapped columns** | `Mapped[T]` for type-safe columns |
| **Repository** | Abstract data access, enable testing |
| **Unit of Work** | Manage transactions across repositories |
| **Alembic** | Version-controlled schema migrations |

---

## Next Steps

Continue to **[Module 08b: MongoDB Deep Dive](08b-mongodb-deep-dive.md)** for:
- PyMongo Async API (Motor deprecated)
- Advanced aggregation pipelines
- Indexing strategies
- Transactions

