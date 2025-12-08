"""
Multi-Provider LLM Service Example
===================================
Demonstrates:
- Strategy Pattern for provider abstraction
- Factory Pattern for dynamic creation
- Streaming responses
- Fallback/retry logic

Run with: python a01_llm_providers.py

Note: Uses mock implementations for demo.
For production, install: openai, anthropic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncIterator, Optional, Type
from enum import Enum
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# DATA MODELS (Simple, no over-engineering)
# ==============================================================================

class Role(str, Enum):
    """Message roles - same across all providers."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A chat message."""
    role: Role
    content: str


@dataclass
class LLMResponse:
    """Standardized response from any provider."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    provider: str = ""


# ==============================================================================
# STRATEGY PATTERN: Provider Interface
# ==============================================================================

class LLMProvider(ABC):
    """
    Abstract base for LLM providers.
    
    Strategy Pattern: Each provider implements this interface.
    Business logic depends on abstraction, not concrete providers.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate complete response."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        pass


# ==============================================================================
# MOCK PROVIDERS (For demo without API keys)
# ==============================================================================

class MockOpenAIProvider(LLMProvider):
    """
    Mock OpenAI provider for demo.
    
    In production, use:
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
    """
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "openai"
    
    async def complete(
        self,
        messages: List[Message],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Simulate OpenAI completion."""
        await asyncio.sleep(0.1)  # Simulate API latency
        
        # In production:
        # response = await self.client.chat.completions.create(
        #     model=model,
        #     messages=[{"role": m.role.value, "content": m.content} for m in messages],
        #     temperature=temperature,
        # )
        # return LLMResponse(
        #     content=response.choices[0].message.content,
        #     model=response.model,
        #     usage={"total_tokens": response.usage.total_tokens},
        #     provider=self.name,
        # )
        
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        return LLMResponse(
            content=f"[OpenAI {model}] Response to: {user_msg[:50]}",
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            provider=self.name,
        )
    
    async def stream(
        self,
        messages: List[Message],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Simulate streaming."""
        # In production:
        # stream = await self.client.chat.completions.create(
        #     model=model, messages=[...], stream=True
        # )
        # async for chunk in stream:
        #     if chunk.choices[0].delta.content:
        #         yield chunk.choices[0].delta.content
        
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        response = f"[OpenAI {model}] Streaming response to: {user_msg[:30]}"
        
        for word in response.split():
            await asyncio.sleep(0.05)
            yield word + " "


class MockAnthropicProvider(LLMProvider):
    """
    Mock Anthropic provider for demo.
    
    In production, use:
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
    """
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    def _extract_system(self, messages: List[Message]) -> tuple[str, List[Message]]:
        """
        Anthropic handles system prompt separately.
        
        Key API difference from OpenAI.
        """
        system = ""
        others = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system = m.content
            else:
                others.append(m)
        return system, others
    
    async def complete(
        self,
        messages: List[Message],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Simulate Anthropic completion."""
        await asyncio.sleep(0.1)
        
        # In production:
        # system, msgs = self._extract_system(messages)
        # response = await self.client.messages.create(
        #     model=model,
        #     system=system,
        #     messages=[{"role": m.role.value, "content": m.content} for m in msgs],
        #     max_tokens=1024,
        # )
        # return LLMResponse(
        #     content=response.content[0].text,
        #     model=response.model,
        #     usage={"total_tokens": response.usage.input_tokens + response.usage.output_tokens},
        #     provider=self.name,
        # )
        
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        return LLMResponse(
            content=f"[Anthropic {model}] Response to: {user_msg[:50]}",
            model=model,
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            provider=self.name,
        )
    
    async def stream(
        self,
        messages: List[Message],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Simulate streaming."""
        # In production:
        # async with self.client.messages.stream(...) as stream:
        #     async for text in stream.text_stream:
        #         yield text
        
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        response = f"[Anthropic {model}] Streaming response to: {user_msg[:30]}"
        
        for word in response.split():
            await asyncio.sleep(0.05)
            yield word + " "


class MockOllamaProvider(LLMProvider):
    """
    Mock Ollama provider for local models.
    
    Ollama runs locally, no API key needed.
    Great for development and privacy.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    @property
    def name(self) -> str:
        return "ollama"
    
    async def complete(
        self,
        messages: List[Message],
        model: str = "llama2",
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Simulate Ollama completion."""
        await asyncio.sleep(0.1)
        
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        return LLMResponse(
            content=f"[Ollama {model}] Response to: {user_msg[:50]}",
            model=model,
            usage={"total_tokens": 25},
            provider=self.name,
        )
    
    async def stream(
        self,
        messages: List[Message],
        model: str = "llama2",
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Simulate streaming."""
        user_msg = next((m.content for m in messages if m.role == Role.USER), "")
        response = f"[Ollama {model}] Streaming response to: {user_msg[:30]}"
        
        for word in response.split():
            await asyncio.sleep(0.05)
            yield word + " "


# ==============================================================================
# FACTORY PATTERN: Provider Creation
# ==============================================================================

class ProviderFactory:
    """
    Factory for creating LLM providers.
    
    Why Factory?
    - Centralized creation logic
    - Easy to add new providers
    - Config-driven instantiation
    """
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": MockOpenAIProvider,
        "anthropic": MockAnthropicProvider,
        "ollama": MockOllamaProvider,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]):
        """Register a new provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> LLMProvider:
        """Create a provider by name."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(cls._providers.keys())}")
        return cls._providers[name](**kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


# ==============================================================================
# ROUTER: Fallback and Retry Logic
# ==============================================================================

class LLMRouter:
    """
    Route LLM requests with fallback support.
    
    Production features:
    - Retry with exponential backoff
    - Automatic fallback to secondary provider
    - Logging for observability
    """
    
    def __init__(
        self,
        primary: LLMProvider,
        fallback: Optional[LLMProvider] = None,
        max_retries: int = 2,
    ):
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
    
    async def complete(
        self,
        messages: List[Message],
        model: str,
        **kwargs,
    ) -> LLMResponse:
        """Complete with retry and fallback."""
        
        # Try primary
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Trying {self.primary.name} (attempt {attempt + 1})")
                return await self.primary.complete(messages, model, **kwargs)
            except Exception as e:
                logger.warning(f"{self.primary.name} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Try fallback
        if self.fallback:
            logger.info(f"Falling back to {self.fallback.name}")
            return await self.fallback.complete(messages, model, **kwargs)
        
        raise RuntimeError("All providers failed")
    
    async def stream(
        self,
        messages: List[Message],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream with fallback on failure."""
        try:
            async for token in self.primary.stream(messages, model, **kwargs):
                yield token
        except Exception as e:
            logger.warning(f"{self.primary.name} stream failed: {e}")
            if self.fallback:
                async for token in self.fallback.stream(messages, model, **kwargs):
                    yield token
            else:
                raise


# ==============================================================================
# DEMO
# ==============================================================================

async def demo():
    """Demonstrate multi-provider LLM service."""
    
    print("=" * 60)
    print("Multi-Provider LLM Service Demo")
    print("=" * 60)
    
    # ========== FACTORY ==========
    print("\n--- Factory Pattern ---")
    print(f"Available providers: {ProviderFactory.list_providers()}")
    
    openai = ProviderFactory.create("openai", api_key="sk-demo")
    anthropic = ProviderFactory.create("anthropic", api_key="sk-ant-demo")
    ollama = ProviderFactory.create("ollama")
    
    # ========== STRATEGY ==========
    print("\n--- Strategy Pattern (Same Interface) ---")
    
    messages = [
        Message(Role.SYSTEM, "You are a helpful assistant."),
        Message(Role.USER, "What is Python?"),
    ]
    
    for provider in [openai, anthropic, ollama]:
        response = await provider.complete(messages, model="default")
        print(f"{provider.name}: {response.content}")
    
    # ========== STREAMING ==========
    print("\n--- Streaming Response ---")
    
    print("OpenAI streaming: ", end="", flush=True)
    async for token in openai.stream(messages, model="gpt-4o-mini"):
        print(token, end="", flush=True)
    print()
    
    # ========== ROUTER WITH FALLBACK ==========
    print("\n--- Router with Fallback ---")
    
    router = LLMRouter(
        primary=openai,
        fallback=anthropic,
        max_retries=1,
    )
    
    response = await router.complete(messages, model="gpt-4o-mini")
    print(f"Router response: {response.content}")
    print(f"Provider used: {response.provider}")
    
    # ========== STREAMING VIA ROUTER ==========
    print("\n--- Streaming via Router ---")
    
    print("Router streaming: ", end="", flush=True)
    async for token in router.stream(messages, model="gpt-4o-mini"):
        print(token, end="", flush=True)
    print()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    print("""
Key Patterns Demonstrated:
- Strategy: LLMProvider ABC with multiple implementations
- Factory: ProviderFactory.create() for dynamic instantiation
- Adapter: Message conversion in each provider
- Router: Retry + fallback for resilience

No over-engineering - just clean OOP for real-world use.
""")


if __name__ == "__main__":
    asyncio.run(demo())
