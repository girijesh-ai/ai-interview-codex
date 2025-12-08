"""
Module 03, Example 01: Metaclasses - Tool Auto-Registration

This example demonstrates:
- What metaclasses are
- How to create a metaclass
- Auto-registering tools with metaclasses
- Validation metaclasses

Run this file:
    python 01_metaclasses.py

Follow along with: 03-advanced-oop-agent-architecture.md
"""

from typing import Dict, Type, Any, Optional
from abc import abstractmethod


# =============================================================================
# PART 1: Understanding Metaclasses
# =============================================================================

print("=== Part 1: What is a Metaclass? ===")
print("""
A metaclass is "the class of a class."

Normal objects:
    my_instance = MyClass()
    - my_instance is an INSTANCE of MyClass
    - MyClass is the CLASS

Classes are also objects:
    class MyClass: pass
    - MyClass is an INSTANCE of type
    - type is the METACLASS

So: type is the default metaclass of all classes.
When you write 'class Foo:', Python calls type() to create it.
""")

# Proof that classes are instances of type
class RegularClass:
    pass

print(f"type(RegularClass) = {type(RegularClass)}")  # <class 'type'>
print(f"isinstance(RegularClass, type) = {isinstance(RegularClass, type)}")


# =============================================================================
# PART 2: Basic Metaclass for Tool Registration
# =============================================================================

class ToolMeta(type):
    """Metaclass that auto-registers tools.
    
    When any class using this metaclass is created,
    it's automatically added to the registry.
    
    This is called automatically when you define a class with:
        class MyTool(BaseTool, metaclass=ToolMeta):
    """
    
    # Class-level registry (shared across all uses of this metaclass)
    _registry: Dict[str, Type] = {}
    
    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict
    ) -> type:
        """Called when a new class is CREATED (not instantiated).
        
        Args:
            mcs: The metaclass itself (like 'cls' for regular classes)
            name: Name of the class being created
            bases: Tuple of base classes
            namespace: Dict of class attributes and methods
            
        Returns:
            The newly created class
        """
        # First, create the class using the default mechanism
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Don't register the base class itself
        if name != "BaseTool" and not namespace.get("__abstract__", False):
            # Get tool name from class attribute or use class name
            tool_name = namespace.get("name", name.lower())
            mcs._registry[tool_name] = cls
            print(f"🔧 Auto-registered tool: {tool_name}")
        
        return cls
    
    @classmethod
    def get_tool(mcs, name: str) -> Optional[Type]:
        """Get tool class by name."""
        return mcs._registry.get(name)
    
    @classmethod
    def list_tools(mcs) -> list:
        """List all registered tools."""
        return list(mcs._registry.keys())


class BaseTool(metaclass=ToolMeta):
    """Base class for all agent tools.
    
    By specifying metaclass=ToolMeta, any subclass of BaseTool
    will automatically be registered.
    """
    
    name: str = "base"
    description: str = "Base tool"
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        ...


# =============================================================================
# PART 3: Auto-Registration in Action
# =============================================================================

print("\n=== Part 3: Auto-Registration ===")

# These tools are registered AUTOMATICALLY when the class is defined!

class SearchTool(BaseTool):
    """Search the web."""
    
    name = "search"
    description = "Search the web for information"
    
    def execute(self, query: str) -> str:
        return f"Search results for: {query}"


class CalculatorTool(BaseTool):
    """Perform calculations."""
    
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def execute(self, expression: str) -> str:
        try:
            result = eval(expression)  # Don't do this in production!
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"


class CodeExecutorTool(BaseTool):
    """Execute Python code."""
    
    name = "code_executor"
    description = "Execute Python code snippets"
    
    def execute(self, code: str) -> str:
        return f"Executed: {code[:50]}..."


# Check the registry
print(f"\nRegistered tools: {ToolMeta.list_tools()}")

# Get and use a tool by name
SearchToolClass = ToolMeta.get_tool("search")
if SearchToolClass:
    search = SearchToolClass()
    print(f"Search result: {search.execute(query='Python tutorials')}")


# =============================================================================
# PART 4: Validating Metaclass
# =============================================================================

print("\n=== Part 4: Validation Metaclass ===")


class ValidatedToolMeta(type):
    """Metaclass that validates tool definitions at class creation time."""
    
    _registry: Dict[str, Type] = {}
    REQUIRED_ATTRS = ["name", "description"]
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip validation for the base class
        if name == "ValidatedTool":
            return cls
        
        # Validate required attributes
        for attr in mcs.REQUIRED_ATTRS:
            if attr not in namespace:
                raise TypeError(
                    f"Tool '{name}' must define '{attr}' attribute"
                )
        
        # Validate execute method exists
        if "execute" not in namespace:
            raise TypeError(
                f"Tool '{name}' must implement 'execute' method"
            )
        
        # If all validations pass, register the tool
        mcs._registry[namespace["name"]] = cls
        print(f"✅ Validated and registered: {namespace['name']}")
        
        return cls


class ValidatedTool(metaclass=ValidatedToolMeta):
    """Base tool with mandatory validation."""
    pass


# This will work (all required attributes present)
class GoodTool(ValidatedTool):
    name = "good"
    description = "A properly defined tool"
    
    def execute(self) -> str:
        return "Good!"


# This would raise TypeError at class definition time!
# Uncomment to see the error:

# class BadTool(ValidatedTool):
#     # Missing: name, description, execute
#     pass


# =============================================================================
# PART 5: When to Use Metaclasses
# =============================================================================

print("\n=== Part 5: When to Use Metaclasses ===")
print("""
USE METACLASSES WHEN:
✓ Auto-registering classes (plugins, tools, handlers)
✓ Validating class definitions at creation time
✓ Adding/modifying class attributes automatically
✓ Implementing ORMs, serialization frameworks

AVOID METACLASSES WHEN:
✗ A class decorator would work
✗ __init_subclass__ would work (simpler alternative)
✗ The logic could be in __init__ or a base class

Rule of thumb:
    __init_subclass__ > class decorator > metaclass
    (prefer simpler solutions first)
""")

# Show relationship
print("\nClass creation hierarchy:")
print(f"  SearchTool is instance of: {type(SearchTool).__name__}")
print(f"  ToolMeta is instance of: {type(ToolMeta).__name__}")
