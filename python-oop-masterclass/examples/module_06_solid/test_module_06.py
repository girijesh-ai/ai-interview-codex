"""
Tests for Module 06: SOLID Principles & Clean Architecture
===========================================================
"""

import pytest
from datetime import datetime
from typing import List

# Import from our examples
from module_06_solid.01_solid_principles import (
    PromptBuilder,
    LLMClient,
    ResponseParser,
    ResponseCache,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockProvider,
    Agent,
    LLMGateway,
    ResearchAgent,
    OpenAIGateway,
    MockGateway,
    FullProvider,
    SimpleProvider,
    generate_response,
    build_embeddings,
)

from module_06_solid.02_clean_architecture import (
    Message,
    Conversation,
    ChatUseCase,
    SummarizeUseCase,
    MockLLMGateway,
    InMemoryConversationRepository,
    Container,
)


# ==============================================================================
# SOLID Principles Tests
# ==============================================================================

class TestSingleResponsibility:
    """Test SRP: Each class has one responsibility."""
    
    def test_prompt_builder_only_builds_prompts(self):
        builder = PromptBuilder(system_prompt="Test system")
        messages = builder.build("Hello")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_llm_client_only_completes(self):
        client = LLMClient(model="test-model")
        response = client.complete([{"role": "user", "content": "Hi"}])
        
        assert "test-model" in response
        assert "Hi" in response
    
    def test_response_parser_only_parses(self):
        parser = ResponseParser()
        
        # Test JSON extraction
        assert parser.extract_json('{"key": "value"}') == {"key": "value"}
        assert parser.extract_json("not json") is None
        
        # Test code extraction
        code = parser.extract_code("```python\nprint('hello')\n```")
        assert "print" in code
    
    def test_cache_only_caches(self):
        cache = ResponseCache()
        
        assert cache.get("missing") is None
        cache.set("key", "value")
        assert cache.get("key") == "value"


class TestOpenClosed:
    """Test OCP: Open for extension, closed for modification."""
    
    def test_agent_works_with_any_provider(self):
        """Agent doesn't need modification for new providers."""
        providers = [
            OpenAIProvider(),
            AnthropicProvider(),
            OllamaProvider(),
        ]
        
        for provider in providers:
            agent = Agent(llm=provider)
            result = agent.run("Test task")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_can_add_new_provider_without_changes(self):
        """New providers can be added without modifying Agent."""
        class NewProvider(LLMProvider):
            def complete(self, prompt: str) -> str:
                return f"[New] {prompt}"
            def get_model_name(self) -> str:
                return "new"
        
        agent = Agent(llm=NewProvider())
        result = agent.run("Test")
        assert "[New]" in result


class TestLiskovSubstitution:
    """Test LSP: Subclasses can substitute for base classes."""
    
    def test_mock_substitutes_for_real_provider(self):
        """MockProvider can replace OpenAIProvider."""
        mock = MockProvider(response="Controlled output")
        agent = Agent(llm=mock)
        
        result = agent.run("Any task")
        
        assert result == "Controlled output"
        assert mock.call_count == 1
    
    def test_all_providers_return_string(self):
        """All providers honor the contract."""
        providers: List[LLMProvider] = [
            OpenAIProvider(),
            AnthropicProvider(),
            OllamaProvider(),
            MockProvider(),
        ]
        
        for provider in providers:
            result = provider.complete("test")
            assert isinstance(result, str)


class TestInterfaceSegregation:
    """Test ISP: Clients only depend on interfaces they use."""
    
    def test_completable_only_interface(self):
        """SimpleProvider only needs Completable."""
        simple = SimpleProvider()
        result = generate_response(simple, "test")
        assert isinstance(result, str)
    
    def test_embeddable_interface_separate(self):
        """FullProvider implements Embeddable too."""
        full = FullProvider()
        embeddings = build_embeddings(full, ["doc1", "doc2"])
        assert len(embeddings) == 2
        assert all(isinstance(e, list) for e in embeddings)


class TestDependencyInversion:
    """Test DIP: Depend on abstractions, not concretions."""
    
    def test_agent_depends_on_abstraction(self):
        """ResearchAgent depends on LLMGateway, not concrete class."""
        # Works with OpenAI
        prod_agent = ResearchAgent(llm=OpenAIGateway())
        result1 = prod_agent.research("Python")
        assert "OpenAI" in result1
        
        # Works with Mock
        test_agent = ResearchAgent(llm=MockGateway())
        result2 = test_agent.research("Python")
        assert "Mock" in result2


# ==============================================================================
# Clean Architecture Tests
# ==============================================================================

class TestEntities:
    """Test Layer 1: Entities."""
    
    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
    
    def test_conversation_add_message(self):
        conv = Conversation(id="test-1")
        msg = conv.add_message("user", "Hi")
        
        assert len(conv.messages) == 1
        assert msg.role == "user"
    
    def test_conversation_get_context(self):
        conv = Conversation(id="test-1")
        for i in range(15):
            conv.add_message("user", f"Message {i}")
        
        context = conv.get_context(limit=5)
        assert len(context) == 5
        assert context[-1].content == "Message 14"


class TestUseCases:
    """Test Layer 2: Use Cases."""
    
    def test_chat_use_case_sends_message(self):
        mock_llm = MockLLMGateway(response="Bot response")
        repo = InMemoryConversationRepository()
        use_case = ChatUseCase(mock_llm, repo)
        
        response = use_case.send_message("conv-1", "Hello")
        
        assert response == "Bot response"
        assert len(mock_llm.calls) == 1
    
    def test_chat_use_case_persists_conversation(self):
        mock_llm = MockLLMGateway()
        repo = InMemoryConversationRepository()
        use_case = ChatUseCase(mock_llm, repo)
        
        use_case.send_message("conv-1", "First")
        use_case.send_message("conv-1", "Second")
        
        conv = repo.get("conv-1")
        assert len(conv.messages) == 4  # 2 user + 2 assistant
    
    def test_chat_use_case_validates_empty_message(self):
        mock_llm = MockLLMGateway()
        repo = InMemoryConversationRepository()
        use_case = ChatUseCase(mock_llm, repo)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            use_case.send_message("conv-1", "   ")
    
    def test_get_history(self):
        mock_llm = MockLLMGateway()
        repo = InMemoryConversationRepository()
        use_case = ChatUseCase(mock_llm, repo)
        
        use_case.send_message("conv-1", "Hello")
        history = use_case.get_history("conv-1")
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestAdapters:
    """Test Layer 3: Interface Adapters (Repositories)."""
    
    def test_in_memory_repo_crud(self):
        repo = InMemoryConversationRepository()
        conv = Conversation(id="test-1")
        conv.add_message("user", "Hello")
        
        # Save
        repo.save(conv)
        
        # Get
        retrieved = repo.get("test-1")
        assert retrieved is not None
        assert len(retrieved.messages) == 1
        
        # Delete
        assert repo.delete("test-1") is True
        assert repo.get("test-1") is None


class TestFrameworks:
    """Test Layer 4: Frameworks & Drivers."""
    
    def test_mock_llm_gateway_records_calls(self):
        gateway = MockLLMGateway(response="Test response")
        
        msg = Message(role="user", content="Hello")
        result = gateway.generate([msg])
        
        assert result == "Test response"
        assert len(gateway.calls) == 1
        assert gateway.calls[0][0].content == "Hello"


class TestContainer:
    """Test Dependency Injection Container."""
    
    def test_development_container(self):
        container = Container(environment="development")
        use_case = container.get_chat_use_case()
        
        response = use_case.send_message("test", "Hi")
        assert "gpt-3.5-turbo" in response
    
    def test_test_container(self):
        container = Container(environment="test")
        use_case = container.get_chat_use_case()
        
        response = use_case.send_message("test", "Hi")
        assert response == "Mock response"
    
    def test_production_container(self):
        container = Container(environment="production")
        use_case = container.get_chat_use_case()
        
        response = use_case.send_message("test", "Hi")
        assert "gpt-4" in response


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
