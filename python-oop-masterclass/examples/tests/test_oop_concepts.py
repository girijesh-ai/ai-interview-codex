"""
Tests for Python OOP Masterclass Examples

This test suite uses unittest (no external dependencies).
Verifies all OOP concepts from Modules 01-03.

Run tests:
    python3 tests/test_oop_concepts.py -v
    python3 -m unittest discover tests -v
"""

import unittest
import sys
import os
from typing import Any
from datetime import datetime
from abc import ABC, abstractmethod

# Add examples to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================================
# Module 01: OOP Fundamentals Tests
# =============================================================================

class TestModule01ChatMessage(unittest.TestCase):
    """Tests for 01_chat_message.py concepts."""
    
    def test_class_creation(self):
        """Test basic class instantiation."""
        class ChatMessage:
            def __init__(self, role: str, content: str):
                self.role = role
                self.content = content
        
        msg = ChatMessage("user", "Hello!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello!")
    
    def test_init_sets_attributes(self):
        """Test __init__ properly initializes attributes."""
        class TestClass:
            def __init__(self, value: str):
                self.value = value
        
        obj = TestClass("test")
        self.assertEqual(obj.value, "test")
        self.assertTrue(hasattr(obj, "value"))
    
    def test_self_refers_to_instance(self):
        """Test that self correctly references the instance."""
        class Counter:
            def __init__(self):
                self.count = 0
            
            def increment(self):
                self.count += 1
                return self
        
        c1 = Counter()
        c2 = Counter()
        c1.increment()
        
        self.assertEqual(c1.count, 1)
        self.assertEqual(c2.count, 0)  # Independent instances


class TestModule01Methods(unittest.TestCase):
    """Tests for method types."""
    
    def test_instance_method(self):
        """Test instance method can access self."""
        class Client:
            def __init__(self, name):
                self.name = name
            
            def get_name(self):
                return self.name
        
        client = Client("test")
        self.assertEqual(client.get_name(), "test")
    
    def test_class_method(self):
        """Test class method receives cls."""
        class Factory:
            instances = 0
            
            @classmethod
            def create(cls):
                cls.instances += 1
                return cls()
        
        Factory.create()
        Factory.create()
        self.assertEqual(Factory.instances, 2)
    
    def test_static_method(self):
        """Test static method doesn't need instance."""
        class Utils:
            @staticmethod
            def add(a, b):
                return a + b
        
        self.assertEqual(Utils.add(2, 3), 5)


class TestModule01Properties(unittest.TestCase):
    """Tests for properties."""
    
    def test_property_getter(self):
        """Test property getter."""
        class Temperature:
            def __init__(self, celsius):
                self._celsius = celsius
            
            @property
            def fahrenheit(self):
                return self._celsius * 9/5 + 32
        
        temp = Temperature(0)
        self.assertEqual(temp.fahrenheit, 32)
    
    def test_property_setter_validation(self):
        """Test property setter with validation."""
        class PositiveNumber:
            def __init__(self, value):
                self.value = value
            
            @property
            def value(self):
                return self._value
            
            @value.setter
            def value(self, v):
                if v < 0:
                    raise ValueError("Must be positive")
                self._value = v
        
        num = PositiveNumber(5)
        
        with self.assertRaises(ValueError):
            num.value = -1


class TestModule01DunderMethods(unittest.TestCase):
    """Tests for dunder methods."""
    
    def test_str_representation(self):
        """Test __str__ for user-friendly output."""
        class Message:
            def __init__(self, content):
                self.content = content
            
            def __str__(self):
                return f"Message: {self.content}"
        
        msg = Message("Hello")
        self.assertEqual(str(msg), "Message: Hello")
    
    def test_eq_for_equality(self):
        """Test __eq__ for equality comparison."""
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __eq__(self, other):
                return self.x == other.x and self.y == other.y
        
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        p3 = Point(3, 4)
        
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    def test_len_for_containers(self):
        """Test __len__ for custom containers."""
        class Conversation:
            def __init__(self):
                self.messages = []
            
            def add(self, msg):
                self.messages.append(msg)
            
            def __len__(self):
                return len(self.messages)
        
        conv = Conversation()
        self.assertEqual(len(conv), 0)
        
        conv.add("Hello")
        conv.add("Hi")
        self.assertEqual(len(conv), 2)


# =============================================================================
# Module 02: OOP Principles Tests
# =============================================================================

class TestModule02Encapsulation(unittest.TestCase):
    """Tests for encapsulation."""
    
    def test_protected_attribute_convention(self):
        """Test _underscore indicates protected."""
        class Secure:
            def __init__(self, secret):
                self._secret = secret
            
            def get_masked(self):
                return self._secret[:2] + "***"
        
        s = Secure("password")
        self.assertEqual(s.get_masked(), "pa***")
        self.assertEqual(s._secret, "password")  # Still accessible
    
    def test_private_name_mangling(self):
        """Test __double_underscore name mangling."""
        class Private:
            def __init__(self, value):
                self.__value = value
        
        p = Private("secret")
        
        with self.assertRaises(AttributeError):
            _ = p.__value


class TestModule02Inheritance(unittest.TestCase):
    """Tests for inheritance."""
    
    def test_single_inheritance(self):
        """Test basic inheritance."""
        class Animal:
            def speak(self):
                return "..."
        
        class Dog(Animal):
            def speak(self):
                return "Woof!"
        
        dog = Dog()
        self.assertEqual(dog.speak(), "Woof!")
        self.assertIsInstance(dog, Animal)
    
    def test_super_calls_parent(self):
        """Test super() calls parent method."""
        class Base:
            def __init__(self, value):
                self.value = value
        
        class Child(Base):
            def __init__(self, value, extra):
                super().__init__(value)
                self.extra = extra
        
        child = Child("base", "child")
        self.assertEqual(child.value, "base")
        self.assertEqual(child.extra, "child")


class TestModule02Polymorphism(unittest.TestCase):
    """Tests for polymorphism."""
    
    def test_method_overriding(self):
        """Test polymorphic method calls."""
        class LLM:
            def complete(self, prompt):
                raise NotImplementedError
        
        class OpenAI(LLM):
            def complete(self, prompt):
                return f"OpenAI: {prompt}"
        
        class Anthropic(LLM):
            def complete(self, prompt):
                return f"Claude: {prompt}"
        
        def process(llm, prompt):
            return llm.complete(prompt)
        
        self.assertEqual(process(OpenAI(), "Hi"), "OpenAI: Hi")
        self.assertEqual(process(Anthropic(), "Hi"), "Claude: Hi")
    
    def test_duck_typing(self):
        """Test duck typing - no inheritance needed."""
        class CustomLLM:
            def complete(self, prompt):
                return f"Custom: {prompt}"
        
        def process(llm, prompt):
            return llm.complete(prompt)
        
        self.assertEqual(process(CustomLLM(), "Hi"), "Custom: Hi")


class TestModule02Abstraction(unittest.TestCase):
    """Tests for abstract classes."""
    
    def test_abc_cannot_instantiate(self):
        """Test ABC cannot be instantiated."""
        class Shape(ABC):
            @abstractmethod
            def area(self):
                pass
        
        with self.assertRaises(TypeError):
            Shape()
    
    def test_concrete_class_must_implement(self):
        """Test concrete class must implement abstract methods."""
        class Shape(ABC):
            @abstractmethod
            def area(self):
                pass
        
        class Circle(Shape):
            def __init__(self, radius):
                self.radius = radius
            
            def area(self):
                return 3.14159 * self.radius ** 2
        
        circle = Circle(5)
        self.assertGreater(circle.area(), 78)


# =============================================================================
# Module 03: Advanced OOP Tests
# =============================================================================

class TestModule03Metaclasses(unittest.TestCase):
    """Tests for metaclasses."""
    
    def test_metaclass_auto_registration(self):
        """Test metaclass auto-registers classes."""
        registry = {}
        
        class RegistryMeta(type):
            def __new__(mcs, name, bases, namespace):
                cls = super().__new__(mcs, name, bases, namespace)
                if name != "Base":
                    registry[name] = cls
                return cls
        
        class Base(metaclass=RegistryMeta):
            pass
        
        class ToolA(Base):
            pass
        
        class ToolB(Base):
            pass
        
        self.assertIn("ToolA", registry)
        self.assertIn("ToolB", registry)
        self.assertNotIn("Base", registry)


class TestModule03NewVsInit(unittest.TestCase):
    """Tests for __new__ vs __init__."""
    
    def test_singleton_with_new(self):
        """Test singleton pattern using __new__."""
        class Singleton:
            _instance = None
            
            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance
        
        s1 = Singleton()
        s2 = Singleton()
        
        self.assertIs(s1, s2)


class TestModule03Descriptors(unittest.TestCase):
    """Tests for descriptors."""
    
    def test_descriptor_get_set(self):
        """Test descriptor __get__ and __set__."""
        class Positive:
            def __set_name__(self, owner, name):
                self.name = f"_{name}"
            
            def __get__(self, obj, type=None):
                return getattr(obj, self.name, 0)
            
            def __set__(self, obj, value):
                if value < 0:
                    raise ValueError("Must be positive")
                setattr(obj, self.name, value)
        
        class Account:
            balance = Positive()
        
        acc = Account()
        acc.balance = 100
        self.assertEqual(acc.balance, 100)
        
        with self.assertRaises(ValueError):
            acc.balance = -50


class TestModule03Slots(unittest.TestCase):
    """Tests for __slots__."""
    
    def test_slots_no_dict(self):
        """Test __slots__ prevents __dict__."""
        class Slotted:
            __slots__ = ("x", "y")
            
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        obj = Slotted(1, 2)
        
        with self.assertRaises(AttributeError):
            _ = obj.__dict__
    
    def test_slots_prevents_dynamic_attrs(self):
        """Test __slots__ prevents adding attributes."""
        class Slotted:
            __slots__ = ("x",)
            
            def __init__(self, x):
                self.x = x
        
        obj = Slotted(1)
        
        with self.assertRaises(AttributeError):
            obj.y = 2


class TestModule03ContextManagers(unittest.TestCase):
    """Tests for context managers."""
    
    def test_enter_exit_called(self):
        """Test __enter__ and __exit__ are called."""
        calls = []
        
        class Manager:
            def __enter__(self):
                calls.append("enter")
                return self
            
            def __exit__(self, *args):
                calls.append("exit")
                return False
        
        with Manager():
            calls.append("body")
        
        self.assertEqual(calls, ["enter", "body", "exit"])
    
    def test_exit_on_exception(self):
        """Test __exit__ called even on exception."""
        cleanup_called = False
        
        class Manager:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                nonlocal cleanup_called
                cleanup_called = True
                return False
        
        with self.assertRaises(ValueError):
            with Manager():
                raise ValueError("test")
        
        self.assertTrue(cleanup_called)


class TestModule03Generators(unittest.TestCase):
    """Tests for generators."""
    
    def test_generator_yields(self):
        """Test generator yields values."""
        def gen():
            yield 1
            yield 2
            yield 3
        
        result = list(gen())
        self.assertEqual(result, [1, 2, 3])
    
    def test_generator_is_lazy(self):
        """Test generator is lazy - produces on demand."""
        calls = []
        
        def gen():
            for i in range(3):
                calls.append(i)
                yield i
        
        g = gen()
        self.assertEqual(calls, [])  # Nothing produced yet
        
        next(g)
        self.assertEqual(calls, [0])


class TestModule03Decorators(unittest.TestCase):
    """Tests for decorators."""
    
    def test_basic_decorator(self):
        """Test basic function decorator."""
        def uppercase(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs).upper()
            return wrapper
        
        @uppercase
        def greet(name):
            return f"hello {name}"
        
        self.assertEqual(greet("world"), "HELLO WORLD")
    
    def test_decorator_with_args(self):
        """Test decorator with arguments."""
        def repeat(times):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    return [func(*args, **kwargs) for _ in range(times)]
                return wrapper
            return decorator
        
        @repeat(3)
        def say_hi():
            return "hi"
        
        self.assertEqual(say_hi(), ["hi", "hi", "hi"])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple concepts."""
    
    def test_llm_client_full_stack(self):
        """Test LLM client with all OOP concepts."""
        class BaseLLM(ABC):
            def __init__(self, model: str):
                self._model = model
                self._messages = []
            
            @property
            def model(self) -> str:
                return self._model
            
            @abstractmethod
            def complete(self, prompt: str) -> str:
                pass
            
            def __len__(self):
                return len(self._messages)
        
        class MockLLM(BaseLLM):
            def complete(self, prompt: str) -> str:
                response = f"Response to: {prompt}"
                self._messages.append({"prompt": prompt, "response": response})
                return response
        
        llm = MockLLM("mock-model")
        self.assertEqual(llm.model, "mock-model")
        
        response = llm.complete("Hello")
        self.assertIn("Hello", response)
        self.assertEqual(len(llm), 1)


# =============================================================================
# Module 04: Modern Python Tests
# =============================================================================

class TestModule04TypeVar(unittest.TestCase):
    """Tests for TypeVar and Generics."""
    
    def test_generic_class(self):
        """Test Generic[T] class works with different types."""
        from typing import TypeVar, Generic, Dict, Optional, List
        
        T = TypeVar('T')
        
        class Repository(Generic[T]):
            def __init__(self):
                self._items: Dict[str, T] = {}
            
            def add(self, id: str, item: T) -> None:
                self._items[id] = item
            
            def get(self, id: str) -> Optional[T]:
                return self._items.get(id)
        
        # Test with str
        str_repo: Repository[str] = Repository()
        str_repo.add("1", "hello")
        self.assertEqual(str_repo.get("1"), "hello")
        
        # Test with int
        int_repo: Repository[int] = Repository()
        int_repo.add("1", 42)
        self.assertEqual(int_repo.get("1"), 42)
    
    def test_bounded_typevar(self):
        """Test bounded TypeVar constrains types."""
        from typing import TypeVar, List
        
        class Animal:
            def speak(self) -> str:
                return "..."
        
        class Dog(Animal):
            def speak(self) -> str:
                return "Woof!"
        
        T = TypeVar('T', bound=Animal)
        
        def make_speak(animal: T) -> str:
            return animal.speak()
        
        dog = Dog()
        self.assertEqual(make_speak(dog), "Woof!")


class TestModule04Protocols(unittest.TestCase):
    """Tests for Protocols (structural typing)."""
    
    def test_protocol_structural_typing(self):
        """Test Protocol matches structurally."""
        from typing import Protocol, runtime_checkable
        
        @runtime_checkable
        class Speakable(Protocol):
            def speak(self) -> str: ...
        
        # Class doesn't inherit from Speakable
        class Robot:
            def speak(self) -> str:
                return "Beep boop"
        
        robot = Robot()
        
        # But it's still "Speakable" because it has speak()
        self.assertTrue(isinstance(robot, Speakable))
        self.assertEqual(robot.speak(), "Beep boop")
    
    def test_protocol_function_accepts_any_matching(self):
        """Test function accepts any Protocol-matching object."""
        from typing import Protocol
        
        class Completable(Protocol):
            def complete(self, prompt: str) -> str: ...
        
        class OpenAI:
            def complete(self, prompt: str) -> str:
                return f"OpenAI: {prompt}"
        
        class Claude:
            def complete(self, prompt: str) -> str:
                return f"Claude: {prompt}"
        
        def process(llm: Completable, prompt: str) -> str:
            return llm.complete(prompt)
        
        # Both work even without inheritance
        self.assertEqual(process(OpenAI(), "Hi"), "OpenAI: Hi")
        self.assertEqual(process(Claude(), "Hi"), "Claude: Hi")


class TestModule04Dataclasses(unittest.TestCase):
    """Tests for dataclasses."""
    
    def test_dataclass_auto_init(self):
        """Test dataclass generates __init__."""
        from dataclasses import dataclass
        
        @dataclass
        class Message:
            role: str
            content: str
        
        msg = Message("user", "Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")
    
    def test_dataclass_auto_eq(self):
        """Test dataclass generates __eq__."""
        from dataclasses import dataclass
        
        @dataclass
        class Point:
            x: int
            y: int
        
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        p3 = Point(3, 4)
        
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    def test_dataclass_frozen(self):
        """Test frozen dataclass is immutable."""
        from dataclasses import dataclass
        
        @dataclass(frozen=True)
        class ImmutableMessage:
            role: str
            content: str
        
        msg = ImmutableMessage("user", "Hello")
        
        with self.assertRaises(Exception):  # FrozenInstanceError
            msg.role = "assistant"
    
    def test_dataclass_order(self):
        """Test order=True enables sorting."""
        from dataclasses import dataclass
        
        @dataclass(order=True)
        class Priority:
            level: int
            name: str
        
        items = [Priority(3, "low"), Priority(1, "high"), Priority(2, "med")]
        sorted_items = sorted(items)
        
        self.assertEqual(sorted_items[0].level, 1)
        self.assertEqual(sorted_items[2].level, 3)
    
    def test_dataclass_field_default_factory(self):
        """Test field with default_factory for mutable defaults."""
        from dataclasses import dataclass, field
        from typing import List
        
        @dataclass
        class Container:
            items: List[str] = field(default_factory=list)
        
        c1 = Container()
        c2 = Container()
        
        c1.items.append("item1")
        
        # c2 is unaffected - separate list
        self.assertEqual(len(c1.items), 1)
        self.assertEqual(len(c2.items), 0)
    
    def test_dataclass_post_init(self):
        """Test __post_init__ for validation."""
        from dataclasses import dataclass
        
        @dataclass
        class PositiveNumber:
            value: int
            
            def __post_init__(self):
                if self.value < 0:
                    raise ValueError("Must be positive")
        
        num = PositiveNumber(5)
        self.assertEqual(num.value, 5)
        
        with self.assertRaises(ValueError):
            PositiveNumber(-1)


class TestModule04Async(unittest.TestCase):
    """Tests for async patterns."""
    
    def test_async_function(self):
        """Test basic async function."""
        import asyncio
        
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b
        
        result = asyncio.run(async_add(2, 3))
        self.assertEqual(result, 5)
    
    def test_async_gather(self):
        """Test asyncio.gather for concurrent execution."""
        import asyncio
        
        async def delayed_value(value: int) -> int:
            await asyncio.sleep(0.01)
            return value
        
        async def gather_all():
            tasks = [delayed_value(i) for i in range(5)]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(gather_all())
        self.assertEqual(results, [0, 1, 2, 3, 4])
    
    def test_async_context_manager(self):
        """Test async context manager."""
        import asyncio
        
        class AsyncResource:
            def __init__(self):
                self.opened = False
                self.closed = False
            
            async def __aenter__(self):
                self.opened = True
                return self
            
            async def __aexit__(self, *args):
                self.closed = True
                return False
        
        async def use_resource():
            resource = AsyncResource()
            async with resource:
                pass
            return resource
        
        resource = asyncio.run(use_resource())
        self.assertTrue(resource.opened)
        self.assertTrue(resource.closed)
    
    def test_async_iterator(self):
        """Test async iterator."""
        import asyncio
        
        class AsyncRange:
            def __init__(self, n: int):
                self.n = n
                self.i = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.i >= self.n:
                    raise StopAsyncIteration
                await asyncio.sleep(0.001)
                value = self.i
                self.i += 1
                return value
        
        async def collect():
            result = []
            async for i in AsyncRange(3):
                result.append(i)
            return result
        
        result = asyncio.run(collect())
        self.assertEqual(result, [0, 1, 2])


class TestModule04PatternMatching(unittest.TestCase):
    """Tests for pattern matching (Python 3.10+)."""
    
    def test_pattern_matching_available(self):
        """Test pattern matching concepts (version-safe)."""
        import sys
        
        # Pattern matching requires 3.10+
        # Test the equivalent behavior with if/elif
        from dataclasses import dataclass
        
        @dataclass
        class ToolCall:
            name: str
            args: dict
        
        def route_tool(tool: ToolCall) -> str:
            # This is what match/case does under the hood
            if tool.name == "search":
                return f"Searching: {tool.args.get('query', '')}"
            elif tool.name == "calculate":
                return f"Calculating: {tool.args.get('expr', '')}"
            else:
                return f"Unknown: {tool.name}"
        
        self.assertEqual(
            route_tool(ToolCall("search", {"query": "python"})),
            "Searching: python"
        )
        self.assertEqual(
            route_tool(ToolCall("calculate", {"expr": "2+2"})),
            "Calculating: 2+2"
        )
        self.assertEqual(
            route_tool(ToolCall("unknown", {})),
            "Unknown: unknown"
        )


# =============================================================================
# Module 05: Design Patterns Tests
# =============================================================================

class TestModule05Singleton(unittest.TestCase):
    """Tests for Singleton pattern."""
    
    def test_singleton_same_instance(self):
        """Test that singleton returns same instance."""
        class Singleton:
            _instance = None
            
            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance
        
        s1 = Singleton()
        s2 = Singleton()
        
        self.assertIs(s1, s2)


class TestModule05FactoryMethod(unittest.TestCase):
    """Tests for Factory Method pattern."""
    
    def test_factory_creates_correct_type(self):
        """Test factory method creates appropriate type."""
        class Animal(ABC):
            @abstractmethod
            def speak(self) -> str: pass
        
        class Dog(Animal):
            def speak(self) -> str:
                return "Woof!"
        
        class Cat(Animal):
            def speak(self) -> str:
                return "Meow!"
        
        class AnimalFactory(ABC):
            @abstractmethod
            def create(self) -> Animal: pass
        
        class DogFactory(AnimalFactory):
            def create(self) -> Animal:
                return Dog()
        
        class CatFactory(AnimalFactory):
            def create(self) -> Animal:
                return Cat()
        
        dog_factory = DogFactory()
        cat_factory = CatFactory()
        
        self.assertEqual(dog_factory.create().speak(), "Woof!")
        self.assertEqual(cat_factory.create().speak(), "Meow!")


class TestModule05Builder(unittest.TestCase):
    """Tests for Builder pattern."""
    
    def test_builder_fluent_interface(self):
        """Test builder provides fluent interface."""
        from dataclasses import dataclass, field
        from typing import List
        
        @dataclass
        class Config:
            name: str = ""
            settings: List[str] = field(default_factory=list)
        
        class ConfigBuilder:
            def __init__(self):
                self._config = Config()
            
            def with_name(self, name: str) -> "ConfigBuilder":
                self._config.name = name
                return self
            
            def with_setting(self, setting: str) -> "ConfigBuilder":
                self._config.settings.append(setting)
                return self
            
            def build(self) -> Config:
                return self._config
        
        config = (
            ConfigBuilder()
            .with_name("Test")
            .with_setting("A")
            .with_setting("B")
            .build()
        )
        
        self.assertEqual(config.name, "Test")
        self.assertEqual(config.settings, ["A", "B"])


class TestModule05Adapter(unittest.TestCase):
    """Tests for Adapter pattern."""
    
    def test_adapter_converts_interface(self):
        """Test adapter converts one interface to another."""
        # Target interface
        class USPlug(ABC):
            @abstractmethod
            def provide_110v(self) -> int: pass
        
        # Adaptee (different interface)
        class EUPlug:
            def provide_230v(self) -> int:
                return 230
        
        # Adapter
        class EUtoUSAdapter(USPlug):
            def __init__(self, eu_plug: EUPlug):
                self._eu_plug = eu_plug
            
            def provide_110v(self) -> int:
                return self._eu_plug.provide_230v() // 2
        
        eu_plug = EUPlug()
        adapter = EUtoUSAdapter(eu_plug)
        
        self.assertEqual(adapter.provide_110v(), 115)


class TestModule05Decorator(unittest.TestCase):
    """Tests for Decorator pattern."""
    
    def test_decorator_adds_behavior(self):
        """Test decorator adds behavior without modifying original."""
        class Component(ABC):
            @abstractmethod
            def operation(self) -> str: pass
        
        class ConcreteComponent(Component):
            def operation(self) -> str:
                return "Base"
        
        class Decorator(Component):
            def __init__(self, component: Component):
                self._component = component
            
            def operation(self) -> str:
                return self._component.operation()
        
        class UppercaseDecorator(Decorator):
            def operation(self) -> str:
                return self._component.operation().upper()
        
        class PrefixDecorator(Decorator):
            def operation(self) -> str:
                return f"PREFIX:{self._component.operation()}"
        
        # Stack decorators
        component = ConcreteComponent()
        decorated = PrefixDecorator(UppercaseDecorator(component))
        
        self.assertEqual(decorated.operation(), "PREFIX:BASE")


class TestModule05Strategy(unittest.TestCase):
    """Tests for Strategy pattern."""
    
    def test_strategy_swappable(self):
        """Test strategies can be swapped at runtime."""
        class Strategy(ABC):
            @abstractmethod
            def execute(self, data: int) -> int: pass
        
        class DoubleStrategy(Strategy):
            def execute(self, data: int) -> int:
                return data * 2
        
        class SquareStrategy(Strategy):
            def execute(self, data: int) -> int:
                return data ** 2
        
        class Context:
            def __init__(self, strategy: Strategy):
                self._strategy = strategy
            
            def set_strategy(self, strategy: Strategy):
                self._strategy = strategy
            
            def process(self, data: int) -> int:
                return self._strategy.execute(data)
        
        context = Context(DoubleStrategy())
        self.assertEqual(context.process(5), 10)
        
        context.set_strategy(SquareStrategy())
        self.assertEqual(context.process(5), 25)


class TestModule05Observer(unittest.TestCase):
    """Tests for Observer pattern."""
    
    def test_observer_notified(self):
        """Test observers are notified on changes."""
        class Observer(ABC):
            @abstractmethod
            def update(self, message: str): pass
        
        class ConcreteObserver(Observer):
            def __init__(self):
                self.messages = []
            
            def update(self, message: str):
                self.messages.append(message)
        
        class Subject:
            def __init__(self):
                self._observers = []
            
            def attach(self, observer: Observer):
                self._observers.append(observer)
            
            def notify(self, message: str):
                for obs in self._observers:
                    obs.update(message)
        
        subject = Subject()
        observer1 = ConcreteObserver()
        observer2 = ConcreteObserver()
        
        subject.attach(observer1)
        subject.attach(observer2)
        
        subject.notify("Hello")
        
        self.assertEqual(observer1.messages, ["Hello"])
        self.assertEqual(observer2.messages, ["Hello"])


class TestModule05State(unittest.TestCase):
    """Tests for State pattern."""
    
    def test_state_changes_behavior(self):
        """Test state changes object behavior."""
        class State(ABC):
            @abstractmethod
            def handle(self) -> str: pass
        
        class IdleState(State):
            def handle(self) -> str:
                return "Idle"
        
        class ActiveState(State):
            def handle(self) -> str:
                return "Active"
        
        class Machine:
            def __init__(self):
                self._state = IdleState()
            
            def set_state(self, state: State):
                self._state = state
            
            def handle(self) -> str:
                return self._state.handle()
        
        machine = Machine()
        self.assertEqual(machine.handle(), "Idle")
        
        machine.set_state(ActiveState())
        self.assertEqual(machine.handle(), "Active")


class TestModule05ChainOfResponsibility(unittest.TestCase):
    """Tests for Chain of Responsibility pattern."""
    
    def test_chain_passes_request(self):
        """Test chain passes request through handlers."""
        class Handler(ABC):
            def __init__(self):
                self._next = None
            
            def set_next(self, handler: "Handler") -> "Handler":
                self._next = handler
                return handler
            
            def handle(self, request: int) -> int:
                result = self._process(request)
                if self._next:
                    return self._next.handle(result)
                return result
            
            @abstractmethod
            def _process(self, request: int) -> int: pass
        
        class AddOneHandler(Handler):
            def _process(self, request: int) -> int:
                return request + 1
        
        class DoubleHandler(Handler):
            def _process(self, request: int) -> int:
                return request * 2
        
        h1 = AddOneHandler()
        h2 = DoubleHandler()
        h1.set_next(h2)
        
        # (5 + 1) * 2 = 12
        self.assertEqual(h1.handle(5), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)


