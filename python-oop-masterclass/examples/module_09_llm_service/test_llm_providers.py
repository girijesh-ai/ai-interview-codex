"""
Tests for Multi-Provider LLM Service
=====================================
Run with: pytest test_llm_providers.py -v
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a01_llm_providers import (
    Message, Role, LLMResponse,
    MockOpenAIProvider, MockAnthropicProvider, MockOllamaProvider,
    ProviderFactory, LLMRouter,
)


# ==============================================================================
# MESSAGE TESTS
# ==============================================================================

class TestMessage:
    """Test Message data class."""
    
    def test_create_user_message(self):
        msg = Message(Role.USER, "Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
    
    def test_create_system_message(self):
        msg = Message(Role.SYSTEM, "You are helpful")
        assert msg.role == Role.SYSTEM


# ==============================================================================
# PROVIDER TESTS
# ==============================================================================

class TestOpenAIProvider:
    """Test OpenAI provider."""
    
    def test_provider_name(self):
        provider = MockOpenAIProvider()
        assert provider.name == "openai"
    
    def test_complete(self):
        async def run():
            provider = MockOpenAIProvider()
            messages = [Message(Role.USER, "Test question")]
            response = await provider.complete(messages, "gpt-4o-mini")
            return response
        
        response = asyncio.run(run())
        assert response.content
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"
    
    def test_stream(self):
        async def run():
            provider = MockOpenAIProvider()
            messages = [Message(Role.USER, "Test")]
            tokens = []
            async for token in provider.stream(messages, "gpt-4o-mini"):
                tokens.append(token)
            return tokens
        
        tokens = asyncio.run(run())
        assert len(tokens) > 0


class TestAnthropicProvider:
    """Test Anthropic provider."""
    
    def test_provider_name(self):
        provider = MockAnthropicProvider()
        assert provider.name == "anthropic"
    
    def test_complete(self):
        async def run():
            provider = MockAnthropicProvider()
            messages = [
                Message(Role.SYSTEM, "You are helpful"),
                Message(Role.USER, "Test"),
            ]
            return await provider.complete(messages, "claude-3-5-sonnet-20241022")
        
        response = asyncio.run(run())
        assert response.provider == "anthropic"


class TestOllamaProvider:
    """Test Ollama provider."""
    
    def test_provider_name(self):
        provider = MockOllamaProvider()
        assert provider.name == "ollama"
    
    def test_complete(self):
        async def run():
            provider = MockOllamaProvider()
            messages = [Message(Role.USER, "Test")]
            return await provider.complete(messages, "llama2")
        
        response = asyncio.run(run())
        assert response.provider == "ollama"
        assert response.model == "llama2"


# ==============================================================================
# FACTORY TESTS
# ==============================================================================

class TestProviderFactory:
    """Test Factory pattern."""
    
    def test_list_providers(self):
        providers = ProviderFactory.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
    
    def test_create_openai(self):
        provider = ProviderFactory.create("openai", api_key="test")
        assert provider.name == "openai"
    
    def test_create_anthropic(self):
        provider = ProviderFactory.create("anthropic", api_key="test")
        assert provider.name == "anthropic"
    
    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            ProviderFactory.create("unknown_provider")


# ==============================================================================
# ROUTER TESTS
# ==============================================================================

class TestLLMRouter:
    """Test Router with fallback."""
    
    def test_primary_success(self):
        async def run():
            primary = MockOpenAIProvider()
            router = LLMRouter(primary=primary)
            messages = [Message(Role.USER, "Test")]
            return await router.complete(messages, "gpt-4o-mini")
        
        response = asyncio.run(run())
        assert response.provider == "openai"
    
    def test_router_stream(self):
        async def run():
            primary = MockOpenAIProvider()
            router = LLMRouter(primary=primary)
            messages = [Message(Role.USER, "Test")]
            tokens = []
            async for token in router.stream(messages, "gpt-4o-mini"):
                tokens.append(token)
            return tokens
        
        tokens = asyncio.run(run())
        assert len(tokens) > 0


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
