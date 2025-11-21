# Backend Engineering, Testing & Production Patterns

## Learning Objectives
- Master backend engineering patterns (Repository, Service Layer, DI)
- Implement comprehensive testing strategies
- Apply performance optimization techniques
- Build production-ready Python applications
- Deploy and monitor systems at scale

## Table of Contents
1. [Backend Engineering Patterns](#backend-patterns)
2. [Dependency Injection](#dependency-injection)
3. [Testing Strategies](#testing-strategies)
4. [Performance Optimization](#performance-optimization)
5. [Production Patterns](#production-patterns)

---

## Backend Engineering Patterns

### Repository Pattern

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class User:
    """User entity."""
    id: Optional[int]
    username: str
    email: str


class Repository(ABC, Generic[T]):
    """Generic repository pattern."""

    @abstractmethod
    async def find_by_id(self, id: int) -> Optional[T]:
        """Find entity by ID."""
        pass

    @abstractmethod
    async def find_all(self) -> List[T]:
        """Find all entities."""
        pass

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity."""
        pass

    @abstractmethod
    async def delete(self, id: int) -> bool:
        """Delete entity."""
        pass


class UserRepository(Repository[User]):
    """User repository implementation."""

    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1

    async def find_by_id(self, id: int) -> Optional[User]:
        """Find user by ID."""
        return self._users.get(id)

    async def find_all(self) -> List[User]:
        """Find all users."""
        return list(self._users.values())

    async def save(self, user: User) -> User:
        """Save user."""
        if user.id is None:
            user.id = self._next_id
            self._next_id += 1
        self._users[user.id] = user
        return user

    async def delete(self, id: int) -> bool:
        """Delete user."""
        if id in self._users:
            del self._users[id]
            return True
        return False

    async def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None
```

### Service Layer Pattern

```python
class UserService:
    """User service layer."""

    def __init__(
        self,
        user_repository: UserRepository,
        email_service: "EmailService",
        logger: "Logger"
    ):
        self.user_repository = user_repository
        self.email_service = email_service
        self.logger = logger

    async def register_user(self, username: str, email: str) -> User:
        """Register new user with business logic.

        Args:
            username: Username
            email: Email address

        Returns:
            Created user

        Raises:
            ValueError: If validation fails
        """
        # Validation
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")

        if '@' not in email:
            raise ValueError("Invalid email format")

        # Check if user exists
        existing = await self.user_repository.find_by_email(email)
        if existing:
            raise ValueError(f"User with email {email} already exists")

        # Create user
        user = User(id=None, username=username, email=email)
        user = await self.user_repository.save(user)

        # Send welcome email (business logic)
        await self.email_service.send_welcome_email(user)

        # Log action
        self.logger.info(f"User registered: {user.username}")

        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return await self.user_repository.find_by_id(user_id)

    async def update_email(self, user_id: int, new_email: str) -> Optional[User]:
        """Update user email."""
        user = await self.user_repository.find_by_id(user_id)
        if not user:
            return None

        # Validation
        if '@' not in new_email:
            raise ValueError("Invalid email format")

        user.email = new_email
        user = await self.user_repository.save(user)

        self.logger.info(f"Email updated for user {user_id}")

        return user
```

### Unit of Work Pattern

```python
class UnitOfWork:
    """Unit of Work pattern for transaction management."""

    def __init__(self):
        self._new_objects = []
        self._dirty_objects = []
        self._removed_objects = []

    def register_new(self, obj):
        """Register new object."""
        self._new_objects.append(obj)

    def register_dirty(self, obj):
        """Register modified object."""
        if obj not in self._dirty_objects:
            self._dirty_objects.append(obj)

    def register_removed(self, obj):
        """Register removed object."""
        self._removed_objects.append(obj)

    async def commit(self):
        """Commit all changes."""
        # Insert new objects
        for obj in self._new_objects:
            await self._insert(obj)

        # Update dirty objects
        for obj in self._dirty_objects:
            await self._update(obj)

        # Delete removed objects
        for obj in self._removed_objects:
            await self._delete(obj)

        # Clear lists
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._removed_objects.clear()

    async def _insert(self, obj):
        """Insert object."""
        print(f"Inserting {obj}")

    async def _update(self, obj):
        """Update object."""
        print(f"Updating {obj}")

    async def _delete(self, obj):
        """Delete object."""
        print(f"Deleting {obj}")
```

---

## Dependency Injection

### Manual Dependency Injection

```python
from typing import Protocol


class Logger(Protocol):
    """Logger interface."""

    def log(self, message: str) -> None:
        """Log message."""
        ...


class ConsoleLogger:
    """Console logger implementation."""

    def log(self, message: str) -> None:
        """Log to console."""
        print(f"[LOG] {message}")


class FileLogger:
    """File logger implementation."""

    def __init__(self, filename: str):
        self.filename = filename

    def log(self, message: str) -> None:
        """Log to file."""
        print(f"[FILE:{self.filename}] {message}")


class EmailService:
    """Email service with injected logger."""

    def __init__(self, logger: Logger):
        self.logger = logger

    async def send_email(self, to: str, subject: str, body: str) -> None:
        """Send email."""
        self.logger.log(f"Sending email to {to}: {subject}")
        # Send email logic


# Manual dependency injection
console_logger = ConsoleLogger()
email_service = EmailService(console_logger)

# Easy to swap implementations
file_logger = FileLogger("app.log")
email_service2 = EmailService(file_logger)
```

### Dependency Injection Container

```python
from typing import Dict, Type, Callable, Any


class DIContainer:
    """Dependency injection container."""

    def __init__(self):
        self._services: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}

    def register(self, interface: Type, implementation: Callable, singleton: bool = False):
        """Register service."""
        self._services[interface] = implementation
        if singleton:
            self._singletons[interface] = None

    def resolve(self, interface: Type) -> Any:
        """Resolve service."""
        if interface in self._singletons:
            if self._singletons[interface] is None:
                self._singletons[interface] = self._services[interface](self)
            return self._singletons[interface]

        if interface in self._services:
            return self._services[interface](self)

        raise ValueError(f"Service {interface} not registered")


# Usage
container = DIContainer()

# Register services
container.register(Logger, lambda c: ConsoleLogger(), singleton=True)
container.register(UserRepository, lambda c: UserRepository())
container.register(
    UserService,
    lambda c: UserService(
        c.resolve(UserRepository),
        c.resolve(EmailService),
        c.resolve(Logger)
    )
)

# Resolve services
user_service = container.resolve(UserService)
```

---

## Testing Strategies

### Unit Testing with pytest

```python
import pytest
from unittest.mock import Mock, AsyncMock


class TestUserService:
    """Test user service."""

    @pytest.fixture
    def user_repository(self):
        """Mock user repository."""
        return Mock(spec=UserRepository)

    @pytest.fixture
    def email_service(self):
        """Mock email service."""
        return Mock()

    @pytest.fixture
    def logger(self):
        """Mock logger."""
        return Mock()

    @pytest.fixture
    def user_service(self, user_repository, email_service, logger):
        """Create user service with mocks."""
        return UserService(user_repository, email_service, logger)

    @pytest.mark.asyncio
    async def test_register_user_success(self, user_service, user_repository, email_service):
        """Test successful user registration."""
        # Arrange
        user_repository.find_by_email = AsyncMock(return_value=None)
        user_repository.save = AsyncMock(
            return_value=User(id=1, username="alice", email="alice@example.com")
        )

        # Act
        user = await user_service.register_user("alice", "alice@example.com")

        # Assert
        assert user.id == 1
        assert user.username == "alice"
        user_repository.save.assert_called_once()
        email_service.send_welcome_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, user_service, user_repository):
        """Test registration with duplicate email."""
        # Arrange
        existing_user = User(id=1, username="existing", email="alice@example.com")
        user_repository.find_by_email = AsyncMock(return_value=existing_user)

        # Act & Assert
        with pytest.raises(ValueError, match="already exists"):
            await user_service.register_user("alice", "alice@example.com")

    @pytest.mark.parametrize("username,email,error_msg", [
        ("ab", "alice@example.com", "at least 3 characters"),
        ("alice", "invalid-email", "Invalid email format"),
    ])
    @pytest.mark.asyncio
    async def test_register_user_validation(
        self, user_service, username, email, error_msg
    ):
        """Test validation errors."""
        with pytest.raises(ValueError, match=error_msg):
            await user_service.register_user(username, email)
```

### Integration Testing

```python
@pytest.mark.integration
class TestUserServiceIntegration:
    """Integration tests with real dependencies."""

    @pytest.fixture
    async def database(self):
        """Set up test database."""
        # Set up database
        db = TestDatabase()
        await db.setup()
        yield db
        await db.teardown()

    @pytest.fixture
    def user_repository(self, database):
        """Real repository with test database."""
        return UserRepository(database)

    @pytest.mark.asyncio
    async def test_full_user_flow(self, user_repository):
        """Test complete user flow."""
        # Create user
        user = User(id=None, username="alice", email="alice@example.com")
        saved_user = await user_repository.save(user)
        assert saved_user.id is not None

        # Retrieve user
        found_user = await user_repository.find_by_id(saved_user.id)
        assert found_user.username == "alice"

        # Update user
        found_user.email = "newemail@example.com"
        await user_repository.save(found_user)

        # Verify update
        updated_user = await user_repository.find_by_id(saved_user.id)
        assert updated_user.email == "newemail@example.com"

        # Delete user
        deleted = await user_repository.delete(saved_user.id)
        assert deleted is True

        # Verify deletion
        deleted_user = await user_repository.find_by_id(saved_user.id)
        assert deleted_user is None
```

### Test-Driven Development (TDD)

```python
# Step 1: Write failing test
def test_calculate_discount():
    """Test discount calculation."""
    calculator = DiscountCalculator()
    price = 100.0
    discount_percent = 10.0

    discounted_price = calculator.calculate(price, discount_percent)

    assert discounted_price == 90.0


# Step 2: Implement minimum code to pass
class DiscountCalculator:
    """Calculate discounts."""

    def calculate(self, price: float, discount_percent: float) -> float:
        """Calculate discounted price."""
        return price * (1 - discount_percent / 100)


# Step 3: Refactor
class DiscountCalculator:
    """Calculate discounts with validation."""

    def calculate(self, price: float, discount_percent: float) -> float:
        """Calculate discounted price.

        Args:
            price: Original price
            discount_percent: Discount percentage (0-100)

        Returns:
            Discounted price

        Raises:
            ValueError: If inputs are invalid
        """
        if price < 0:
            raise ValueError("Price cannot be negative")

        if not 0 <= discount_percent <= 100:
            raise ValueError("Discount must be 0-100")

        return price * (1 - discount_percent / 100)
```

---

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
import time


class CachedService:
    """Service with caching."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}

    def get_cached(self, key: str, ttl: int = 60) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            if time.time() - self._cache_ttl[key] < ttl:
                return self._cache[key]
            else:
                # Cache expired
                del self._cache[key]
                del self._cache_ttl[key]
        return None

    def set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = value
        self._cache_ttl[key] = time.time()

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user with caching."""
        cache_key = f"user:{user_id}"

        # Check cache
        cached = self.get_cached(cache_key)
        if cached:
            return cached

        # Fetch from database
        user = await self._fetch_from_db(user_id)

        # Cache result
        if user:
            self.set_cached(cache_key, user)

        return user

    async def _fetch_from_db(self, user_id: int) -> Optional[User]:
        """Fetch user from database."""
        # Database query
        await asyncio.sleep(0.1)  # Simulate query
        return User(id=user_id, username="user", email="user@example.com")


# Decorator-based caching
@lru_cache(maxsize=128)
def expensive_calculation(n: int) -> int:
    """Expensive calculation with caching."""
    time.sleep(1)  # Simulate expensive operation
    return n * n
```

### Database Optimization

```python
class OptimizedUserRepository:
    """Repository with database optimizations."""

    async def find_users_with_posts(self, limit: int = 10) -> List[Dict]:
        """Find users with posts (N+1 problem solution)."""
        # Bad: N+1 queries
        # users = await self.find_all()
        # for user in users:
        #     user.posts = await self.find_posts_by_user(user.id)

        # Good: Single query with join
        query = """
            SELECT u.*, p.*
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id
            LIMIT ?
        """
        # Execute query and group results
        return []  # Simulated

    async def batch_insert(self, users: List[User]) -> None:
        """Batch insert for performance."""
        # Bad: Individual inserts
        # for user in users:
        #     await self.save(user)

        # Good: Batch insert
        query = "INSERT INTO users (username, email) VALUES (?, ?)"
        values = [(u.username, u.email) for u in users]
        # Execute batch insert
        pass


class ConnectionPool:
    """Database connection pool."""

    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self._pool: List[Any] = []

    async def acquire(self):
        """Acquire connection from pool."""
        if self._pool:
            return self._pool.pop()

        # Create new connection if pool not full
        return await self._create_connection()

    async def release(self, conn):
        """Release connection back to pool."""
        if len(self._pool) < self.max_size:
            self._pool.append(conn)

    async def _create_connection(self):
        """Create new database connection."""
        # Create connection
        return Mock()
```

---

## Production Patterns

### Logging and Monitoring

```python
import logging
from datetime import datetime


class StructuredLogger:
    """Structured logging for production."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log(self, level: str, message: str, **kwargs):
        """Log structured message."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }

        if level == 'ERROR':
            self.logger.error(log_data)
        elif level == 'WARNING':
            self.logger.warning(log_data)
        else:
            self.logger.info(log_data)


class MetricsCollector:
    """Collect application metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record(self, metric_name: str, value: float):
        """Record metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> float:
        """Get average metric value."""
        values = self.metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
```

### Circuit Breaker Pattern

```python
from enum import Enum
import time


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.timeout:
                self._state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0

            return result

        except self.expected_exception:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

            raise
```

### Retry with Exponential Backoff

```python
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> T:
    """Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier

    Returns:
        Function result

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= backoff_factor
```

---

## Summary

### Backend Patterns Quick Reference

| Pattern | Purpose | Use When |
|---------|---------|----------|
| Repository | Data access abstraction | Separate data access from business logic |
| Service Layer | Business logic | Complex business rules |
| Unit of Work | Transaction management | Multiple operations need to be atomic |
| Dependency Injection | Loose coupling | Testing, flexibility |

### Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **TDD**: Write tests first, then implement
- **Mocking**: Isolate dependencies in tests

### Performance Tips

- Use caching for expensive operations
- Implement connection pooling
- Batch database operations
- Use async for I/O-bound tasks

### Production Checklist

- Structured logging
- Metrics collection
- Circuit breakers for external services
- Retry logic with backoff
- Health checks
- Graceful shutdown

### Next Steps
Continue to **08-real-world-applications.md** for complete project examples.

## Interview Questions

### Mid-Level
1. Explain Repository vs DAO pattern
2. What is dependency injection and why use it?
3. How do you write effective unit tests?
4. What caching strategies do you know?

### Senior Level
5. Design a resilient microservice architecture
6. Implement comprehensive testing strategy
7. Optimize database performance for scale
8. Build production monitoring system

### Staff Level
9. Architect distributed system with fault tolerance
10. Design testing strategy for microservices
11. Scale system to handle millions of requests
12. Lead production incident response
