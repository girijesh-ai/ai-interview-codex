"""
SQLAlchemy 2.0 Async Example
============================
Demonstrates:
- Async engine and session setup
- Model definitions with Mapped columns
- Repository pattern
- Unit of Work pattern
- FastAPI integration

Note: This example uses SQLite for simplicity.
For production, use PostgreSQL with asyncpg.

Run with: python a01_sqlalchemy_models.py
"""

from sqlalchemy import (
    String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, Index,
    func, select, update, delete,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession,
)
from datetime import datetime
from typing import Optional, List, TypeVar, Generic, Type
from abc import ABC, abstractmethod
from uuid import uuid4
import asyncio


# ==============================================================================
# BASE MODEL
# ==============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ==============================================================================
# MODELS
# ==============================================================================

class User(Base):
    """User account."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    tier: Mapped[str] = mapped_column(String(50), default="free")
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"User(id={self.id}, email={self.email})"


class Conversation(Base):
    """Chat conversation."""
    
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model: Mapped[str] = mapped_column(String(100), default="gpt-4")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, title={self.title})"


class Message(Base):
    """Message in a conversation."""
    
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    
    def __repr__(self) -> str:
        return f"Message(role={self.role}, content={self.content[:30]}...)"


# ==============================================================================
# DATABASE SETUP
# ==============================================================================

# Use SQLite for demo (use PostgreSQL in production)
DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ==============================================================================
# REPOSITORY PATTERN
# ==============================================================================

T = TypeVar("T", bound=Base)


class Repository(ABC, Generic[T]):
    """Abstract repository."""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        pass


class SQLAlchemyRepository(Repository[T]):
    """SQLAlchemy implementation."""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: str) -> Optional[T]:
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 100) -> List[T]:
        result = await self.session.execute(
            select(self.model).limit(limit)
        )
        return list(result.scalars().all())
    
    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: str) -> bool:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0


class UserRepository(SQLAlchemyRepository[User]):
    """User-specific operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()


class ConversationRepository(SQLAlchemyRepository[Conversation]):
    """Conversation-specific operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Conversation)
    
    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        return list(result.scalars().all())


# ==============================================================================
# UNIT OF WORK
# ==============================================================================

class UnitOfWork:
    """Manages transactions and repositories."""
    
    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
    
    @property
    def users(self) -> UserRepository:
        return UserRepository(self._session)
    
    @property
    def conversations(self) -> ConversationRepository:
        return ConversationRepository(self._session)
    
    async def __aenter__(self) -> "UnitOfWork":
        self._session = self._session_factory()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        await self._session.close()
    
    async def commit(self):
        await self._session.commit()
    
    async def rollback(self):
        await self._session.rollback()


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate SQLAlchemy patterns."""
    
    print("=" * 60)
    print("SQLAlchemy 2.0 Async Demo")
    print("=" * 60)
    
    # Initialize database
    await init_db()
    print("\n1. Database initialized")
    
    # Create unit of work
    async with UnitOfWork(async_session) as uow:
        # Create a user
        user = User(
            email="alice@example.com",
            name="Alice",
            tier="pro"
        )
        await uow.users.create(user)
        await uow.commit()
        print(f"\n2. Created user: {user}")
        
        # Create a conversation
        conv = Conversation(
            user_id=user.id,
            title="SQLAlchemy Help",
            model="gpt-4"
        )
        await uow.conversations.create(conv)
        await uow.commit()
        print(f"\n3. Created conversation: {conv}")
    
    # Query in new unit of work
    async with UnitOfWork(async_session) as uow:
        # Find user by email
        found_user = await uow.users.get_by_email("alice@example.com")
        print(f"\n4. Found user by email: {found_user}")
        
        # Get user's conversations
        convs = await uow.conversations.get_user_conversations(found_user.id)
        print(f"\n5. User conversations: {convs}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- SQLAlchemy 2.0 Mapped columns
- Async engine and sessions
- Repository pattern (UserRepository, ConversationRepository)
- Unit of Work pattern (transaction management)
- Relationships (User -> Conversations)
""")


if __name__ == "__main__":
    asyncio.run(demo())
