"""
SOLID Principles for AI Systems
================================
Demonstrates all 5 SOLID principles applied to LLM/AI applications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol
from datetime import datetime
import json
import re


# ==============================================================================
# S - SINGLE RESPONSIBILITY PRINCIPLE
# ==============================================================================

print("=" * 60)
print("S - Single Responsibility Principle")
print("=" * 60)


# ❌ BAD: One class with multiple responsibilities
class BadLLMService:
    """Violates SRP - does too many things."""
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
    
    def generate(self, user_input: str) -> str:
        # Builds prompt (responsibility 1)
        prompt = f"System: You are helpful.\nUser: {user_input}"
        
        # Checks cache (responsibility 2)
        if prompt in self._cache:
            return self._cache[prompt]
        
        # "Calls" API (responsibility 3)
        response = f"Response to: {user_input}"
        
        # Logs (responsibility 4)
        print(f"[LOG] Generated: {response[:30]}...")
        
        # Caches (responsibility 5)
        self._cache[prompt] = response
        
        return response


# ✅ GOOD: Each class has ONE responsibility

@dataclass
class PromptBuilder:
    """Builds prompts only."""
    system_prompt: str = "You are a helpful assistant."
    
    def build(self, user_input: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]


class LLMClient:
    """Calls LLM API only."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def complete(self, messages: List[Dict[str, str]]) -> str:
        # Simulated API call
        user_msg = messages[-1]["content"] if messages else ""
        return f"[{self.model}] Response to: {user_msg}"


class ResponseParser:
    """Parses LLM responses only."""
    
    def extract_json(self, response: str) -> Optional[Dict]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    def extract_code(self, response: str) -> str:
        match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        return match.group(1) if match else ""


class ResponseCache:
    """Caches responses only."""
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
    
    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        self._cache[key] = value


# Demo: SRP allows easy composition
builder = PromptBuilder(system_prompt="You are a code assistant.")
client = LLMClient(model="gpt-4")
parser = ResponseParser()
cache = ResponseCache()

messages = builder.build("Write a hello world function")
response = client.complete(messages)
print(f"SRP Demo - Response: {response}")


# ==============================================================================
# O - OPEN/CLOSED PRINCIPLE
# ==============================================================================

print("\n" + "=" * 60)
print("O - Open/Closed Principle")
print("=" * 60)


# Base abstraction (closed for modification)
class LLMProvider(ABC):
    """Abstract LLM provider - add new providers, don't modify this."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass


# Extensions (open for extension)
class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        return f"[OpenAI {self.model}] {prompt}"
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3"):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        return f"[Anthropic {self.model}] {prompt}"
    
    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class OllamaProvider(LLMProvider):
    """New provider added WITHOUT modifying existing code."""
    
    def __init__(self, model: str = "llama2"):
        self.model = model
    
    def complete(self, prompt: str) -> str:
        return f"[Ollama {self.model}] {prompt}"
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


# Client code works with any provider - never needs to change
class Agent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def run(self, task: str) -> str:
        return self.llm.complete(task)


# Demo: OCP in action
providers = [
    OpenAIProvider(),
    AnthropicProvider(),
    OllamaProvider(),
]

for provider in providers:
    agent = Agent(llm=provider)
    result = agent.run("Explain OCP")
    print(f"OCP Demo - {provider.get_model_name()}: {result[:40]}...")


# ==============================================================================
# L - LISKOV SUBSTITUTION PRINCIPLE
# ==============================================================================

print("\n" + "=" * 60)
print("L - Liskov Substitution Principle")
print("=" * 60)


# ✅ GOOD: All providers can substitute for each other
class MockProvider(LLMProvider):
    """Mock for testing - fully substitutable."""
    
    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0
    
    def complete(self, prompt: str) -> str:
        self.call_count += 1
        return self.response
    
    def get_model_name(self) -> str:
        return "mock"


def test_agent_with_any_provider(provider: LLMProvider) -> bool:
    """Works with ANY LLMProvider - LSP guarantee."""
    agent = Agent(llm=provider)
    result = agent.run("Test task")
    return isinstance(result, str) and len(result) > 0


# Demo: LSP allows substitution
real_provider = OpenAIProvider()
mock_provider = MockProvider(response="Controlled test output")

print(f"LSP Demo - Real provider works: {test_agent_with_any_provider(real_provider)}")
print(f"LSP Demo - Mock provider works: {test_agent_with_any_provider(mock_provider)}")
print(f"LSP Demo - Mock call count: {mock_provider.call_count}")


# ==============================================================================
# I - INTERFACE SEGREGATION PRINCIPLE
# ==============================================================================

print("\n" + "=" * 60)
print("I - Interface Segregation Principle")
print("=" * 60)


# Segregated interfaces using Protocols
class Completable(Protocol):
    """Interface for text completion."""
    def complete(self, prompt: str) -> str: ...


class Embeddable(Protocol):
    """Interface for embeddings."""
    def embed(self, texts: List[str]) -> List[List[float]]: ...


# Provider that implements both
class FullProvider:
    """Implements both Completable and Embeddable."""
    
    def complete(self, prompt: str) -> str:
        return f"Completed: {prompt}"
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


# Provider that only implements Completable
class SimpleProvider:
    """Only implements Completable - and that's fine!"""
    
    def complete(self, prompt: str) -> str:
        return f"Simple: {prompt}"


# Functions only require what they need
def generate_response(llm: Completable, prompt: str) -> str:
    """Only needs Completable."""
    return llm.complete(prompt)


def build_embeddings(embedder: Embeddable, docs: List[str]) -> List[List[float]]:
    """Only needs Embeddable."""
    return embedder.embed(docs)


# Demo: ISP allows flexible implementations
full = FullProvider()
simple = SimpleProvider()

print(f"ISP Demo - Full provider can complete: {generate_response(full, 'test')}")
print(f"ISP Demo - Simple provider can complete: {generate_response(simple, 'test')}")
print(f"ISP Demo - Full provider can embed: {build_embeddings(full, ['doc1', 'doc2'])}")
# simple.embed() would fail - and that's expected! It doesn't implement Embeddable


# ==============================================================================
# D - DEPENDENCY INVERSION PRINCIPLE
# ==============================================================================

print("\n" + "=" * 60)
print("D - Dependency Inversion Principle")
print("=" * 60)


# Abstraction (interface)
class LLMGateway(ABC):
    """High-level modules depend on this abstraction."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


# High-level module depends on abstraction
class ResearchAgent:
    def __init__(self, llm: LLMGateway):  # Depends on abstraction
        self.llm = llm
    
    def research(self, topic: str) -> str:
        return self.llm.generate(f"Research: {topic}")


# Low-level implementations
class OpenAIGateway(LLMGateway):
    def generate(self, prompt: str) -> str:
        return f"[OpenAI] {prompt}"


class MockGateway(LLMGateway):
    def generate(self, prompt: str) -> str:
        return f"[Mock] {prompt}"


# Dependency injection at composition root
def create_agent(environment: str) -> ResearchAgent:
    """Factory that wires dependencies."""
    if environment == "production":
        return ResearchAgent(llm=OpenAIGateway())
    else:
        return ResearchAgent(llm=MockGateway())


# Demo: DIP enables easy testing and swapping
prod_agent = create_agent("production")
test_agent = create_agent("test")

print(f"DIP Demo - Production: {prod_agent.research('Python')}")
print(f"DIP Demo - Test: {test_agent.research('Python')}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("SOLID Summary")
print("=" * 60)

summary = """
✅ S - Single Responsibility: PromptBuilder, LLMClient, ResponseParser are separate
✅ O - Open/Closed: Add new providers (Ollama) without modifying Agent
✅ L - Liskov Substitution: MockProvider substitutes for OpenAIProvider
✅ I - Interface Segregation: Completable vs Embeddable protocols
✅ D - Dependency Inversion: ResearchAgent depends on LLMGateway, not OpenAI
"""
print(summary)


if __name__ == "__main__":
    print("\n✅ All SOLID principles demonstrated successfully!")
