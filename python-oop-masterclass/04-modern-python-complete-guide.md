# Modern Python: Type Hints, Dataclasses, Async, and Pydantic

## Learning Objectives
- Master type hints and mypy for static type checking
- Use dataclasses for clean, concise classes
- Implement async/await for concurrent programming
- Apply Pydantic for data validation and settings management
- Use pattern matching (Python 3.10+)
- Build production-ready modern Python applications

## Table of Contents
1. [Type Hints and Annotations](#type-hints)
2. [Dataclasses](#dataclasses)
3. [Async/Await Programming](#async-await)
4. [Pydantic for Validation](#pydantic)
5. [Pattern Matching](#pattern-matching)
6. [Production Patterns](#production-patterns)

---

## Type Hints

### Basic Type Hints

```python
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable

# Basic types
def greet(name: str) -> str:
    """Greet a person.

    Args:
        name: Person's name

    Returns:
        Greeting message
    """
    return f"Hello, {name}!"


# Collections
def process_numbers(numbers: List[int]) -> float:
    """Calculate average of numbers.

    Args:
        numbers: List of integers

    Returns:
        Average value
    """
    return sum(numbers) / len(numbers) if numbers else 0.0


# Dictionary types
def get_user_info() -> Dict[str, Union[str, int]]:
    """Get user information.

    Returns:
        User data dictionary
    """
    return {"name": "Alice", "age": 30, "email": "alice@example.com"}


# Optional types
def find_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Find user by ID.

    Args:
        user_id: User identifier

    Returns:
        User data or None if not found
    """
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)


# Callable types
def apply_operation(value: int, operation: Callable[[int], int]) -> int:
    """Apply operation to value.

    Args:
        value: Input value
        operation: Function to apply

    Returns:
        Result of operation
    """
    return operation(value)


# Usage
result = apply_operation(10, lambda x: x * 2)  # 20
```

### Advanced Type Hints

```python
from typing import TypeVar, Generic, Protocol, Literal, Final

# Type variables
T = TypeVar('T')


class Stack(Generic[T]):
    """Generic stack implementation.

    Type Parameters:
        T: Type of elements in the stack
    """

    def __init__(self) -> None:
        """Initialize empty stack."""
        self._items: List[T] = []

    def push(self, item: T) -> None:
        """Push item onto stack.

        Args:
            item: Item to push
        """
        self._items.append(item)

    def pop(self) -> T:
        """Pop item from stack.

        Returns:
            Popped item

        Raises:
            IndexError: If stack is empty
        """
        return self._items.pop()

    def peek(self) -> Optional[T]:
        """Peek at top item.

        Returns:
            Top item or None
        """
        return self._items[-1] if self._items else None


# Usage with type checking
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")


# Literal types
Status = Literal["pending", "processing", "completed", "failed"]


def update_status(status: Status) -> None:
    """Update status.

    Args:
        status: New status (must be one of the literals)
    """
    print(f"Status updated to: {status}")


update_status("completed")  # OK
# update_status("invalid")  # Type checker will catch this


# Final (constant)
MAX_RETRIES: Final[int] = 3


# Protocol for structural subtyping
class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: Any) -> bool:
        """Less than comparison."""
        ...


def get_min(items: List[Comparable]) -> Comparable:
    """Get minimum item.

    Args:
        items: List of comparable items

    Returns:
        Minimum item
    """
    return min(items)
```

---

## Dataclasses

### Basic Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class User:
    """User model using dataclass.

    Attributes:
        id: User ID
        username: Username
        email: Email address
        created_at: Creation timestamp
        is_active: Whether user is active
    """
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


# Usage
user = User(id=1, username="alice", email="alice@example.com")
print(user)
# User(id=1, username='alice', email='alice@example.com', created_at=..., is_active=True)


@dataclass(frozen=True)
class Point:
    """Immutable point."""
    x: float
    y: float

    def distance_from_origin(self) -> float:
        """Calculate distance from origin.

        Returns:
            Distance from (0, 0)
        """
        return (self.x ** 2 + self.y ** 2) ** 0.5


point = Point(3.0, 4.0)
print(point.distance_from_origin())  # 5.0
# point.x = 10  # Error: frozen


@dataclass
class Product:
    """Product with default factory."""
    name: str
    price: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        if self.price < 0:
            raise ValueError("Price cannot be negative")


product = Product(name="Laptop", price=999.99)
product.tags.append("electronics")
```

### Advanced Dataclasses

```python
from dataclasses import dataclass, field, asdict, astuple
from typing import ClassVar


@dataclass(order=True)
class Person:
    """Person with ordering support."""
    sort_index: int = field(init=False, repr=False)
    name: str
    age: int
    email: str

    # Class variable
    count: ClassVar[int] = 0

    def __post_init__(self) -> None:
        """Set sort index based on age."""
        self.sort_index = self.age
        Person.count += 1


people = [
    Person("Alice", 30, "alice@example.com"),
    Person("Bob", 25, "bob@example.com"),
    Person("Charlie", 35, "charlie@example.com"),
]

sorted_people = sorted(people)
for person in sorted_people:
    print(f"{person.name}: {person.age}")
# Bob: 25
# Alice: 30
# Charlie: 35


# Convert to dict and tuple
person = Person("Alice", 30, "alice@example.com")
person_dict = asdict(person)
person_tuple = astuple(person)
print(person_dict)
print(person_tuple)
```

---

## Async/Await

### Basic Async

```python
import asyncio
from typing import List


async def fetch_data(url: str, delay: float = 1.0) -> str:
    """Simulate fetching data from URL.

    Args:
        url: URL to fetch
        delay: Simulated delay

    Returns:
        Fetched data
    """
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)  # Simulate I/O
    print(f"Completed {url}")
    return f"Data from {url}"


async def main() -> None:
    """Main async function."""
    # Sequential execution
    result1 = await fetch_data("https://api.example.com/users")
    result2 = await fetch_data("https://api.example.com/posts")

    print(result1)
    print(result2)


# Run async code
# asyncio.run(main())
```

### Concurrent Async Operations

```python
async def fetch_all_concurrent(urls: List[str]) -> List[str]:
    """Fetch all URLs concurrently.

    Args:
        urls: List of URLs to fetch

    Returns:
        List of fetched data
    """
    # Create tasks for concurrent execution
    tasks = [fetch_data(url, delay=1.0) for url in urls]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    return results


async def main_concurrent() -> None:
    """Run concurrent fetches."""
    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
    ]

    results = await fetch_all_concurrent(urls)
    for result in results:
        print(result)


# asyncio.run(main_concurrent())
```

### Async Context Managers

```python
from typing import AsyncIterator


class AsyncDatabase:
    """Async database connection."""

    def __init__(self, connection_string: str) -> None:
        """Initialize database.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.connected = False

    async def __aenter__(self) -> "AsyncDatabase":
        """Enter async context.

        Returns:
            Database instance
        """
        print(f"Connecting to {self.connection_string}")
        await asyncio.sleep(0.5)  # Simulate connection
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        print("Disconnecting from database")
        await asyncio.sleep(0.5)  # Simulate disconnection
        self.connected = False

    async def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query.

        Args:
            sql: SQL query

        Returns:
            Query results
        """
        if not self.connected:
            raise RuntimeError("Not connected")

        print(f"Executing: {sql}")
        await asyncio.sleep(0.1)  # Simulate query
        return [{"id": 1, "name": "Alice"}]


async def use_database() -> None:
    """Use async database."""
    async with AsyncDatabase("postgresql://localhost/mydb") as db:
        results = await db.query("SELECT * FROM users")
        print(f"Results: {results}")


# asyncio.run(use_database())
```

### Async Iterators

```python
class AsyncRange:
    """Async iterator for range."""

    def __init__(self, start: int, end: int) -> None:
        """Initialize async range.

        Args:
            start: Start value
            end: End value
        """
        self.start = start
        self.end = end
        self.current = start

    def __aiter__(self) -> "AsyncRange":
        """Return async iterator.

        Returns:
            Self
        """
        return self

    async def __anext__(self) -> int:
        """Get next value.

        Returns:
            Next value

        Raises:
            StopAsyncIteration: When iteration is complete
        """
        if self.current >= self.end:
            raise StopAsyncIteration

        await asyncio.sleep(0.1)  # Simulate async operation
        value = self.current
        self.current += 1
        return value


async def iterate_async() -> None:
    """Iterate asynchronously."""
    async for i in AsyncRange(0, 5):
        print(f"Value: {i}")


# asyncio.run(iterate_async())
```

---

## Pydantic

### Basic Pydantic Models

```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional
from datetime import datetime


class User(BaseModel):
    """User model with validation.

    Attributes:
        id: User ID
        username: Username (3-50 chars)
        email: Email address
        age: Age (must be 18+)
        created_at: Creation timestamp
    """
    id: int
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=18, le=150)
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('username')
    def username_alphanumeric(cls, v: str) -> str:
        """Validate username is alphanumeric.

        Args:
            v: Username value

        Returns:
            Validated username

        Raises:
            ValueError: If username is not alphanumeric
        """
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Usage
user = User(
    id=1,
    username="alice123",
    email="alice@example.com",
    age=28
)

print(user.json())  # JSON serialization
print(user.dict())  # Dictionary

# Validation errors
try:
    invalid_user = User(
        id=2,
        username="ab",  # Too short
        email="invalid-email",
        age=15  # Too young
    )
except Exception as e:
    print(f"Validation error: {e}")
```

### Nested Models

```python
class Address(BaseModel):
    """Address model."""
    street: str
    city: str
    country: str
    postal_code: str


class Company(BaseModel):
    """Company model."""
    name: str
    address: Address
    employees: List[User]


# Usage
company = Company(
    name="Tech Corp",
    address=Address(
        street="123 Main St",
        city="San Francisco",
        country="USA",
        postal_code="94105"
    ),
    employees=[
        User(id=1, username="alice", email="alice@example.com", age=28),
        User(id=2, username="bob", email="bob@example.com", age=32),
    ]
)

print(company.json(indent=2))
```

### Settings Management

```python
from pydantic import BaseSettings, PostgresDsn


class Settings(BaseSettings):
    """Application settings.

    Automatically loads from environment variables.
    """
    app_name: str = "MyApp"
    debug: bool = False
    database_url: PostgresDsn
    secret_key: str
    max_connections: int = 10

    class Config:
        """Settings config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Usage (reads from .env file or environment)
# settings = Settings()
# print(settings.database_url)
```

---

## Pattern Matching

### Basic Pattern Matching (Python 3.10+)

```python
def process_command(command: Dict[str, Any]) -> None:
    """Process command using pattern matching.

    Args:
        command: Command dictionary
    """
    match command:
        case {"action": "create", "type": "user", "data": data}:
            print(f"Creating user: {data}")

        case {"action": "update", "type": "user", "id": user_id, "data": data}:
            print(f"Updating user {user_id}: {data}")

        case {"action": "delete", "type": "user", "id": user_id}:
            print(f"Deleting user {user_id}")

        case {"action": "list", "type": "user"}:
            print("Listing users")

        case _:
            print(f"Unknown command: {command}")


# Usage
process_command({"action": "create", "type": "user", "data": {"name": "Alice"}})
process_command({"action": "update", "type": "user", "id": 1, "data": {"age": 30}})
process_command({"action": "delete", "type": "user", "id": 1})
```

### Advanced Pattern Matching

```python
from dataclasses import dataclass


@dataclass
class Point:
    """2D Point."""
    x: float
    y: float


@dataclass
class Circle:
    """Circle shape."""
    center: Point
    radius: float


@dataclass
class Rectangle:
    """Rectangle shape."""
    top_left: Point
    width: float
    height: float


def describe_shape(shape: Union[Point, Circle, Rectangle]) -> str:
    """Describe shape using pattern matching.

    Args:
        shape: Shape object

    Returns:
        Shape description
    """
    match shape:
        case Point(x=0, y=0):
            return "Origin point"

        case Point(x=x, y=0):
            return f"Point on x-axis at {x}"

        case Point(x=0, y=y):
            return f"Point on y-axis at {y}"

        case Point(x=x, y=y):
            return f"Point at ({x}, {y})"

        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle at origin with radius {r}"

        case Circle(radius=r):
            return f"Circle with radius {r}"

        case Rectangle(width=w, height=h) if w == h:
            return f"Square with side {w}"

        case Rectangle(width=w, height=h):
            return f"Rectangle {w}x{h}"

        case _:
            return "Unknown shape"


# Usage
shapes = [
    Point(0, 0),
    Point(5, 0),
    Circle(Point(0, 0), 5),
    Rectangle(Point(0, 0), 10, 10),
    Rectangle(Point(0, 0), 10, 5),
]

for shape in shapes:
    print(describe_shape(shape))
```

---

## Production Patterns

### Type-Safe Configuration

```python
from pydantic import BaseSettings, validator
from typing import List


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    username: str
    password: str
    database: str

    @property
    def dsn(self) -> str:
        """Get database DSN.

        Returns:
            Connection string
        """
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class AppConfig(BaseSettings):
    """Application configuration."""
    environment: Literal["development", "staging", "production"]
    debug: bool = False
    allowed_hosts: List[str] = Field(default_factory=list)
    database: DatabaseConfig

    @validator('debug')
    def debug_in_production(cls, v: bool, values: Dict[str, Any]) -> bool:
        """Ensure debug is False in production.

        Args:
            v: Debug value
            values: Other values

        Returns:
            Validated debug value

        Raises:
            ValueError: If debug is True in production
        """
        if v and values.get('environment') == 'production':
            raise ValueError('Debug cannot be enabled in production')
        return v

    class Config:
        """Config."""
        env_nested_delimiter = '__'


# Usage
config = AppConfig(
    environment="development",
    debug=True,
    database={
        "host": "localhost",
        "port": 5432,
        "username": "user",
        "password": "pass",
        "database": "mydb"
    }
)
```

### Async API Client

```python
import aiohttp
from typing import Dict, Any


class AsyncAPIClient:
    """Async API client."""

    def __init__(self, base_url: str, api_key: str) -> None:
        """Initialize client.

        Args:
            base_url: API base URL
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncAPIClient":
        """Enter async context.

        Returns:
            Client instance
        """
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        if self._session:
            await self._session.close()

    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request.

        Args:
            endpoint: API endpoint

        Returns:
            Response data
        """
        if not self._session:
            raise RuntimeError("Client not initialized")

        url = f"{self.base_url}{endpoint}"
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request.

        Args:
            endpoint: API endpoint
            data: Request data

        Returns:
            Response data
        """
        if not self._session:
            raise RuntimeError("Client not initialized")

        url = f"{self.base_url}{endpoint}"
        async with self._session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()


async def use_api_client() -> None:
    """Use API client."""
    async with AsyncAPIClient("https://api.example.com", "secret-key") as client:
        users = await client.get("/users")
        print(f"Users: {users}")

        new_user = await client.post("/users", {"name": "Alice", "email": "alice@example.com"})
        print(f"Created: {new_user}")


# asyncio.run(use_api_client())
```

---

## Summary

Modern Python features covered:
- **Type Hints**: Static type checking with mypy
- **Dataclasses**: Clean, concise data containers
- **Async/Await**: Concurrent programming
- **Pydantic**: Data validation and settings
- **Pattern Matching**: Structural pattern matching

### Next Steps
Continue to **05-design-patterns-complete.md** for all 23 GoF design patterns.

## Interview Questions

### Mid-Level
1. What are the benefits of type hints?
2. When would you use a dataclass vs a regular class?
3. Explain async/await and when to use it
4. What is Pydantic and why is it useful?

### Senior Level
5. Design a type-safe configuration system
6. Implement an async connection pool
7. Build a validation framework with Pydantic
8. Optimize async performance for I/O-bound operations
