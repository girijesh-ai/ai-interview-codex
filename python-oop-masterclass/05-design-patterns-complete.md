# Complete Design Patterns Guide: All 23 GoF Patterns in Python

## Learning Objectives
- Master all 23 Gang of Four design patterns
- Understand when and how to apply each pattern
- Implement patterns using modern Python features
- Apply patterns in production-ready code
- Make architectural decisions based on pattern trade-offs

## Table of Contents

### Creational Patterns (5)
1. [Singleton](#singleton)
2. [Factory Method](#factory-method)
3. [Abstract Factory](#abstract-factory)
4. [Builder](#builder)
5. [Prototype](#prototype)

### Structural Patterns (7)
6. [Adapter](#adapter)
7. [Bridge](#bridge)
8. [Composite](#composite)
9. [Decorator](#decorator)
10. [Facade](#facade)
11. [Flyweight](#flyweight)
12. [Proxy](#proxy)

### Behavioral Patterns (11)
13. [Chain of Responsibility](#chain-of-responsibility)
14. [Command](#command)
15. [Iterator](#iterator)
16. [Mediator](#mediator)
17. [Memento](#memento)
18. [Observer](#observer)
19. [State](#state)
20. [Strategy](#strategy)
21. [Template Method](#template-method)
22. [Visitor](#visitor)
23. [Interpreter](#interpreter)

---

## Creational Patterns

### Singleton

**Intent**: Ensure a class has only one instance and provide global access to it.

```python
from typing import Optional


class Singleton:
    """Singleton using metaclass."""

    _instance: Optional["Singleton"] = None

    def __new__(cls):
        """Create or return existing instance.

        Returns:
            Singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class DatabaseConnection(Singleton):
    """Database connection singleton."""

    def __init__(self) -> None:
        """Initialize connection (only once)."""
        if not hasattr(self, 'initialized'):
            self.connection_string = "postgresql://localhost/db"
            self.initialized = True


# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True
```

**When to use**: Configuration, logging, connection pools, caches

---

### Factory Method

**Intent**: Define an interface for creating objects, but let subclasses decide which class to instantiate.

```python
from abc import ABC, abstractmethod


class Document(ABC):
    """Abstract document."""

    @abstractmethod
    def show(self) -> str:
        """Show document content."""
        pass


class PDFDocument(Document):
    """PDF document."""

    def show(self) -> str:
        """Show PDF content."""
        return "Showing PDF document"


class WordDocument(Document):
    """Word document."""

    def show(self) -> str:
        """Show Word content."""
        return "Showing Word document"


class Application(ABC):
    """Abstract application."""

    @abstractmethod
    def create_document(self) -> Document:
        """Factory method for creating documents."""
        pass

    def open_document(self) -> str:
        """Open and show document."""
        doc = self.create_document()
        return doc.show()


class PDFApplication(Application):
    """PDF application."""

    def create_document(self) -> Document:
        """Create PDF document."""
        return PDFDocument()


class WordApplication(Application):
    """Word application."""

    def create_document(self) -> Document:
        """Create Word document."""
        return WordDocument()


# Usage
pdf_app = PDFApplication()
print(pdf_app.open_document())  # Showing PDF document

word_app = WordApplication()
print(word_app.open_document())  # Showing Word document
```

**When to use**: Plugin systems, document editors, framework design

---

### Abstract Factory

**Intent**: Provide an interface for creating families of related objects without specifying their concrete classes.

```python
class Button(ABC):
    """Abstract button."""

    @abstractmethod
    def render(self) -> str:
        """Render button."""
        pass


class WindowsButton(Button):
    """Windows-style button."""

    def render(self) -> str:
        """Render Windows button."""
        return "[Windows Button]"


class MacButton(Button):
    """Mac-style button."""

    def render(self) -> str:
        """Render Mac button."""
        return "[Mac Button]"


class Checkbox(ABC):
    """Abstract checkbox."""

    @abstractmethod
    def render(self) -> str:
        """Render checkbox."""
        pass


class WindowsCheckbox(Checkbox):
    """Windows checkbox."""

    def render(self) -> str:
        """Render Windows checkbox."""
        return "[X] Windows Checkbox"


class MacCheckbox(Checkbox):
    """Mac checkbox."""

    def render(self) -> str:
        """Render Mac checkbox."""
        return "[✓] Mac Checkbox"


class GUIFactory(ABC):
    """Abstract GUI factory."""

    @abstractmethod
    def create_button(self) -> Button:
        """Create button."""
        pass

    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        """Create checkbox."""
        pass


class WindowsFactory(GUIFactory):
    """Windows GUI factory."""

    def create_button(self) -> Button:
        """Create Windows button."""
        return WindowsButton()

    def create_checkbox(self) -> Checkbox:
        """Create Windows checkbox."""
        return WindowsCheckbox()


class MacFactory(GUIFactory):
    """Mac GUI factory."""

    def create_button(self) -> Button:
        """Create Mac button."""
        return MacButton()

    def create_checkbox(self) -> Checkbox:
        """Create Mac checkbox."""
        return MacCheckbox()


def render_ui(factory: GUIFactory) -> None:
    """Render UI using factory."""
    button = factory.create_button()
    checkbox = factory.create_checkbox()

    print(button.render())
    print(checkbox.render())


# Usage
render_ui(WindowsFactory())
render_ui(MacFactory())
```

**When to use**: Cross-platform UIs, theme systems, database adapters

---

### Builder

**Intent**: Separate construction of complex object from its representation.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Pizza:
    """Pizza product."""
    size: str
    cheese: bool = False
    pepperoni: bool = False
    mushrooms: bool = False
    olives: bool = False


class PizzaBuilder:
    """Pizza builder."""

    def __init__(self, size: str) -> None:
        """Initialize builder.

        Args:
            size: Pizza size
        """
        self._pizza = Pizza(size=size)

    def add_cheese(self) -> "PizzaBuilder":
        """Add cheese."""
        self._pizza.cheese = True
        return self

    def add_pepperoni(self) -> "PizzaBuilder":
        """Add pepperoni."""
        self._pizza.pepperoni = True
        return self

    def add_mushrooms(self) -> "PizzaBuilder":
        """Add mushrooms."""
        self._pizza.mushrooms = True
        return self

    def add_olives(self) -> "PizzaBuilder":
        """Add olives."""
        self._pizza.olives = True
        return self

    def build(self) -> Pizza:
        """Build pizza."""
        return self._pizza


# Usage
pizza = (PizzaBuilder("large")
         .add_cheese()
         .add_pepperoni()
         .add_mushrooms()
         .build())

print(pizza)
```

**When to use**: Complex object construction, query builders, test data builders

---

### Prototype

**Intent**: Create new objects by copying existing objects.

```python
from copy import deepcopy


class Prototype:
    """Prototype base class."""

    def clone(self) -> "Prototype":
        """Clone object.

        Returns:
            Deep copy of object
        """
        return deepcopy(self)


class Shape(Prototype):
    """Shape prototype."""

    def __init__(self, color: str) -> None:
        """Initialize shape.

        Args:
            color: Shape color
        """
        self.color = color


class Circle(Shape):
    """Circle shape."""

    def __init__(self, color: str, radius: float) -> None:
        """Initialize circle.

        Args:
            color: Circle color
            radius: Circle radius
        """
        super().__init__(color)
        self.radius = radius


# Usage
original = Circle("red", 5.0)
cloned = original.clone()
cloned.color = "blue"

print(f"Original: {original.color}, {original.radius}")  # red, 5.0
print(f"Cloned: {cloned.color}, {cloned.radius}")  # blue, 5.0
```

**When to use**: Object pools, configuration templates, game object spawning

---

## Structural Patterns

### Adapter

**Intent**: Convert interface of a class into another interface clients expect.

```python
class EuropeanSocket:
    """European socket (230V)."""

    def provide_230v(self) -> int:
        """Provide 230V power."""
        return 230


class USASocket:
    """USA socket (110V)."""

    def provide_110v(self) -> int:
        """Provide 110V power."""
        return 110


class SocketAdapter:
    """Adapter for European socket to USA socket."""

    def __init__(self, european_socket: EuropeanSocket) -> None:
        """Initialize adapter."""
        self.european_socket = european_socket

    def provide_110v(self) -> int:
        """Provide 110V from 230V socket."""
        voltage = self.european_socket.provide_230v()
        return voltage // 2  # Step down voltage


# Usage
european = EuropeanSocket()
adapter = SocketAdapter(european)
print(f"Voltage: {adapter.provide_110v()}V")  # 115V
```

**When to use**: Third-party integration, legacy system integration

---

### Strategy

**Intent**: Define family of algorithms, encapsulate each one, and make them interchangeable.

```python
class PaymentStrategy(ABC):
    """Abstract payment strategy."""

    @abstractmethod
    def pay(self, amount: float) -> str:
        """Process payment."""
        pass


class CreditCardPayment(PaymentStrategy):
    """Credit card payment."""

    def pay(self, amount: float) -> str:
        """Pay with credit card."""
        return f"Paid ${amount} with credit card"


class PayPalPayment(PaymentStrategy):
    """PayPal payment."""

    def pay(self, amount: float) -> str:
        """Pay with PayPal."""
        return f"Paid ${amount} with PayPal"


class ShoppingCart:
    """Shopping cart with payment strategy."""

    def __init__(self, payment_strategy: PaymentStrategy) -> None:
        """Initialize cart."""
        self.payment_strategy = payment_strategy
        self.total = 0.0

    def add_item(self, price: float) -> None:
        """Add item to cart."""
        self.total += price

    def checkout(self) -> str:
        """Checkout with selected payment method."""
        return self.payment_strategy.pay(self.total)


# Usage
cart = ShoppingCart(CreditCardPayment())
cart.add_item(100.0)
cart.add_item(50.0)
print(cart.checkout())  # Paid $150.0 with credit card
```

**When to use**: Multiple algorithms for same task, payment processing, sorting strategies

---

### Observer

**Intent**: Define one-to-many dependency between objects so when one changes state, all dependents are notified.

```python
from typing import List


class Observer(ABC):
    """Abstract observer."""

    @abstractmethod
    def update(self, message: str) -> None:
        """Receive update from subject."""
        pass


class ConcreteObserver(Observer):
    """Concrete observer."""

    def __init__(self, name: str) -> None:
        """Initialize observer."""
        self.name = name

    def update(self, message: str) -> None:
        """Receive update."""
        print(f"{self.name} received: {message}")


class Subject:
    """Subject being observed."""

    def __init__(self) -> None:
        """Initialize subject."""
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        """Attach observer."""
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """Detach observer."""
        self._observers.remove(observer)

    def notify(self, message: str) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(message)


# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer 1")
observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)

subject.notify("Event occurred!")
```

**When to use**: Event systems, pub/sub, MVC architectures, real-time notifications

---

### Decorator

**Intent**: Attach additional responsibilities to object dynamically.

```python
class Component(ABC):
    """Abstract component."""

    @abstractmethod
    def operation(self) -> str:
        """Perform operation."""
        pass


class ConcreteComponent(Component):
    """Concrete component."""

    def operation(self) -> str:
        """Perform basic operation."""
        return "Basic operation"


class Decorator(Component):
    """Base decorator."""

    def __init__(self, component: Component) -> None:
        """Initialize decorator."""
        self._component = component

    def operation(self) -> str:
        """Delegate to component."""
        return self._component.operation()


class LoggingDecorator(Decorator):
    """Logging decorator."""

    def operation(self) -> str:
        """Add logging."""
        result = self._component.operation()
        print(f"Logging: {result}")
        return result


class CachingDecorator(Decorator):
    """Caching decorator."""

    def __init__(self, component: Component) -> None:
        """Initialize with cache."""
        super().__init__(component)
        self._cache = None

    def operation(self) -> str:
        """Add caching."""
        if self._cache is None:
            self._cache = self._component.operation()
            print("Cached result")
        return self._cache


# Usage
component = ConcreteComponent()
logged = LoggingDecorator(component)
cached = CachingDecorator(logged)

print(cached.operation())
print(cached.operation())  # Uses cache
```

**When to use**: Add responsibilities without modifying code, middleware, caching layers

---

## Summary

### Pattern Selection Guide

| Problem | Pattern | Use When |
|---------|---------|----------|
| Single instance needed | Singleton | Database connections, configs |
| Create object families | Abstract Factory | Cross-platform UIs |
| Complex construction | Builder | Query builders, test data |
| Runtime algorithm selection | Strategy | Payment methods, sorting |
| Notify multiple objects | Observer | Event systems, notifications |
| Add responsibilities | Decorator | Middleware, caching |
| Convert interface | Adapter | Third-party integration |
| Encapsulate request | Command | Undo/redo, transactions |
| Vary implementation | Bridge | Multiple platforms |

### Next Steps
Continue to **06-solid-clean-architecture.md** for architectural principles.

## Interview Questions

### Mid-Level
1. Explain Singleton and its alternatives
2. When would you use Factory vs Abstract Factory?
3. What's the difference between Strategy and State?
4. How does Decorator differ from inheritance?

### Senior Level
5. Design a plugin system using design patterns
6. Implement undo/redo with Command pattern
7. Build an event system with Observer
8. Create a caching layer using Decorator and Proxy

### Staff Level
9. Design a microservices communication system
10. Architect a real-time notification platform
11. Build a distributed configuration system
12. Create a high-performance data processing pipeline
