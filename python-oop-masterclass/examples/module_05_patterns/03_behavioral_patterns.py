"""
Module 05, Example 03: Behavioral Patterns for AI Systems

Covers:
- Strategy: Swappable LLM providers
- Observer: Event-driven agent communication
- Chain of Responsibility: Multi-step processing
- Command: Undo/redo for agent actions
- State: Agent state machine
- Template Method: LLM call workflow

Run this file:
    python 03_behavioral_patterns.py

Follow along with: 05-design-patterns-complete.md
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# PATTERN 1: STRATEGY - Swappable LLM Providers
# =============================================================================

print("=== Pattern 1: Strategy - Swappable LLM Providers ===")


class CompletionStrategy(ABC):
    """Strategy interface for LLM completion.
    
    Strategy Pattern:
    - Define family of algorithms
    - Make them interchangeable
    - Select algorithm at runtime
    """
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        pass


class GPT4Strategy(CompletionStrategy):
    """High quality, higher cost."""
    
    def complete(self, prompt: str) -> str:
        return f"[GPT-4 High Quality] {prompt[:30]}..."
    
    def get_cost_per_token(self) -> float:
        return 0.00003


class GPT35Strategy(CompletionStrategy):
    """Fast and cheap."""
    
    def complete(self, prompt: str) -> str:
        return f"[GPT-3.5 Fast] {prompt[:30]}..."
    
    def get_cost_per_token(self) -> float:
        return 0.000002


class LocalLlamaStrategy(CompletionStrategy):
    """Free but slower."""
    
    def complete(self, prompt: str) -> str:
        return f"[Llama Local] {prompt[:30]}..."
    
    def get_cost_per_token(self) -> float:
        return 0.0


class AIAssistant:
    """Assistant that uses strategy for LLM calls."""
    
    def __init__(self, strategy: CompletionStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: CompletionStrategy) -> None:
        """Change strategy at runtime."""
        self._strategy = strategy
    
    def answer(self, question: str) -> str:
        return self._strategy.complete(question)
    
    def estimate_cost(self, tokens: int) -> float:
        return tokens * self._strategy.get_cost_per_token()


# Usage - switch strategies based on context
assistant = AIAssistant(GPT4Strategy())
print(f"Complex task: {assistant.answer('Explain quantum computing')}")
print(f"Estimated cost (1000 tokens): ${assistant.estimate_cost(1000):.4f}")

assistant.set_strategy(GPT35Strategy())
print(f"Simple task: {assistant.answer('What is 2+2')}")
print(f"Estimated cost (1000 tokens): ${assistant.estimate_cost(1000):.6f}")


# =============================================================================
# PATTERN 2: OBSERVER - Agent Event System
# =============================================================================

print("\n=== Pattern 2: Observer - Agent Events ===")


class AgentEvent:
    """Event data for agent notifications."""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data


class AgentObserver(ABC):
    """Observer interface for agent events.
    
    Observer Pattern:
    - Define one-to-many dependency
    - When subject changes, all observers notified
    - Loose coupling between components
    """
    
    @abstractmethod
    def on_event(self, event: AgentEvent) -> None:
        pass


class LoggingObserver(AgentObserver):
    """Log all agent events."""
    
    def on_event(self, event: AgentEvent) -> None:
        print(f"  [LOG] {event.event_type}: {event.data}")


class MetricsObserver(AgentObserver):
    """Collect metrics from events."""
    
    def __init__(self):
        self.event_counts: Dict[str, int] = {}
    
    def on_event(self, event: AgentEvent) -> None:
        self.event_counts[event.event_type] = \
            self.event_counts.get(event.event_type, 0) + 1


class AlertObserver(AgentObserver):
    """Alert on specific events."""
    
    def on_event(self, event: AgentEvent) -> None:
        if event.event_type == "error":
            print(f"  [ALERT] Error occurred: {event.data}")


class ObservableAgent:
    """Agent that emits events to observers."""
    
    def __init__(self, name: str):
        self.name = name
        self._observers: List[AgentObserver] = []
    
    def add_observer(self, observer: AgentObserver) -> None:
        self._observers.append(observer)
    
    def remove_observer(self, observer: AgentObserver) -> None:
        self._observers.remove(observer)
    
    def _notify(self, event: AgentEvent) -> None:
        for observer in self._observers:
            observer.on_event(event)
    
    def execute_task(self, task: str) -> str:
        # Emit start event
        self._notify(AgentEvent("task_started", {"task": task}))
        
        # Do work
        result = f"Completed: {task}"
        
        # Emit completion
        self._notify(AgentEvent("task_completed", {
            "task": task,
            "result": result
        }))
        
        return result


# Setup observers
agent = ObservableAgent("Worker")
logger = LoggingObserver()
metrics = MetricsObserver()

agent.add_observer(logger)
agent.add_observer(metrics)

# Execute - observers automatically notified
agent.execute_task("Process data")
agent.execute_task("Generate report")

print(f"Metrics collected: {metrics.event_counts}")


# =============================================================================
# PATTERN 3: CHAIN OF RESPONSIBILITY - Processing Pipeline
# =============================================================================

print("\n=== Pattern 3: Chain of Responsibility - Pipeline ===")


@dataclass
class Message:
    """Message passing through the chain."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False


class Handler(ABC):
    """Handler in the chain.
    
    Chain of Responsibility:
    - Chain of handlers process request
    - Each handler decides to process or pass on
    - Decouple sender from receiver
    """
    
    def __init__(self):
        self._next: Optional[Handler] = None
    
    def set_next(self, handler: "Handler") -> "Handler":
        self._next = handler
        return handler
    
    def handle(self, message: Message) -> Message:
        # Process this handler
        message = self._process(message)
        
        # Pass to next if not stopped and next exists
        if self._next and not message.should_stop:
            return self._next.handle(message)
        
        return message
    
    @abstractmethod
    def _process(self, message: Message) -> Message:
        pass


class InputValidator(Handler):
    """Validate input."""
    
    def _process(self, message: Message) -> Message:
        if len(message.content) < 2:
            message.should_stop = True
            message.content = "Error: Input too short"
            return message
        
        message.metadata["validated"] = True
        print(f"  [Validator] ✓ Input valid")
        return message


class ContentFilter(Handler):
    """Filter inappropriate content."""
    
    BLOCKED_WORDS = {"spam", "malicious"}
    
    def _process(self, message: Message) -> Message:
        for word in self.BLOCKED_WORDS:
            if word in message.content.lower():
                message.should_stop = True
                message.content = "Error: Content blocked"
                return message
        
        message.metadata["filtered"] = True
        print(f"  [Filter] ✓ Content safe")
        return message


class Enricher(Handler):
    """Enrich message with context."""
    
    def _process(self, message: Message) -> Message:
        message.metadata["enriched"] = True
        message.metadata["timestamp"] = "2024-01-01"
        print(f"  [Enricher] ✓ Added metadata")
        return message


class LLMProcessor(Handler):
    """Final LLM processing."""
    
    def _process(self, message: Message) -> Message:
        message.content = f"LLM Response to: {message.content[:30]}"
        message.metadata["processed"] = True
        print(f"  [LLM] ✓ Generated response")
        return message


# Build chain
validator = InputValidator()
filter_handler = ContentFilter()
enricher = Enricher()
llm = LLMProcessor()

validator.set_next(filter_handler).set_next(enricher).set_next(llm)

# Process messages
print("Valid message:")
result = validator.handle(Message("What is Python?"))
print(f"Result: {result.content}")

print("\nBlocked message:")
result = validator.handle(Message("spam content"))
print(f"Result: {result.content}")


# =============================================================================
# PATTERN 4: COMMAND - Undo/Redo for Agent Actions
# =============================================================================

print("\n=== Pattern 4: Command - Undo/Redo ===")


class Command(ABC):
    """Command interface.
    
    Command Pattern:
    - Encapsulate request as object
    - Parameterize actions
    - Support undo/redo
    """
    
    @abstractmethod
    def execute(self) -> str:
        pass
    
    @abstractmethod
    def undo(self) -> str:
        pass


class Document:
    """Document being edited."""
    
    def __init__(self):
        self.content = ""
    
    def append(self, text: str) -> None:
        self.content += text
    
    def remove_last(self, length: int) -> None:
        self.content = self.content[:-length]


class AppendCommand(Command):
    """Command to append text."""
    
    def __init__(self, document: Document, text: str):
        self._document = document
        self._text = text
    
    def execute(self) -> str:
        self._document.append(self._text)
        return f"Appended: {self._text}"
    
    def undo(self) -> str:
        self._document.remove_last(len(self._text))
        return f"Removed: {self._text}"


class CommandHistory:
    """Maintains command history for undo/redo."""
    
    def __init__(self):
        self._history: List[Command] = []
        self._position = -1
    
    def execute(self, command: Command) -> str:
        # Remove any redo history
        self._history = self._history[:self._position + 1]
        
        result = command.execute()
        self._history.append(command)
        self._position += 1
        
        return result
    
    def undo(self) -> Optional[str]:
        if self._position < 0:
            return None
        
        result = self._history[self._position].undo()
        self._position -= 1
        return result
    
    def redo(self) -> Optional[str]:
        if self._position >= len(self._history) - 1:
            return None
        
        self._position += 1
        return self._history[self._position].execute()


# Usage
doc = Document()
history = CommandHistory()

print(history.execute(AppendCommand(doc, "Hello ")))
print(history.execute(AppendCommand(doc, "World")))
print(f"Document: '{doc.content}'")

print(history.undo())
print(f"After undo: '{doc.content}'")

print(history.redo())
print(f"After redo: '{doc.content}'")


# =============================================================================
# PATTERN 5: STATE - Agent State Machine
# =============================================================================

print("\n=== Pattern 5: State - Agent State Machine ===")


class AgentState(ABC):
    """Agent state interface.
    
    State Pattern:
    - Allow object to change behavior based on state
    - Encapsulate state-specific behavior
    - Cleaner than large if/elif blocks
    """
    
    @abstractmethod
    def handle(self, agent: "StatefulAgent", input: str) -> str:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class IdleState(AgentState):
    def handle(self, agent: "StatefulAgent", input: str) -> str:
        if input == "start":
            agent.set_state(ThinkingState())
            return "Starting to think..."
        return "Waiting for task..."
    
    def get_name(self) -> str:
        return "IDLE"


class ThinkingState(AgentState):
    def handle(self, agent: "StatefulAgent", input: str) -> str:
        if input == "plan_ready":
            agent.set_state(ExecutingState())
            return "Plan ready, executing..."
        elif input == "error":
            agent.set_state(IdleState())
            return "Error, returning to idle"
        return "Thinking..."
    
    def get_name(self) -> str:
        return "THINKING"


class ExecutingState(AgentState):
    def handle(self, agent: "StatefulAgent", input: str) -> str:
        if input == "done":
            agent.set_state(IdleState())
            return "Task complete!"
        elif input == "error":
            agent.set_state(ThinkingState())
            return "Error, re-thinking..."
        return "Executing action..."
    
    def get_name(self) -> str:
        return "EXECUTING"


class StatefulAgent:
    """Agent with state machine."""
    
    def __init__(self):
        self._state: AgentState = IdleState()
    
    def set_state(self, state: AgentState) -> None:
        print(f"  State: {self._state.get_name()} -> {state.get_name()}")
        self._state = state
    
    def process(self, input: str) -> str:
        return self._state.handle(self, input)
    
    @property
    def current_state(self) -> str:
        return self._state.get_name()


# State transitions
agent = StatefulAgent()
print(f"Current: {agent.current_state}")

print(agent.process("start"))
print(agent.process("plan_ready"))
print(agent.process("done"))


# =============================================================================
# PATTERN 6: TEMPLATE METHOD - LLM Call Workflow
# =============================================================================

print("\n=== Pattern 6: Template Method - LLM Workflow ===")


class LLMWorkflow(ABC):
    """Template for LLM call workflow.
    
    Template Method Pattern:
    - Define algorithm skeleton in base class
    - Let subclasses override specific steps
    - Ensures consistent workflow structure
    """
    
    def execute(self, prompt: str) -> str:
        """Template method - defines the algorithm."""
        # Step 1: Preprocess
        processed_prompt = self._preprocess(prompt)
        
        # Step 2: Validate
        if not self._validate(processed_prompt):
            return "Invalid prompt"
        
        # Step 3: Call LLM (abstract - must implement)
        response = self._call_llm(processed_prompt)
        
        # Step 4: Postprocess
        final_response = self._postprocess(response)
        
        return final_response
    
    def _preprocess(self, prompt: str) -> str:
        """Default preprocessing - can be overridden."""
        return prompt.strip()
    
    def _validate(self, prompt: str) -> bool:
        """Default validation - can be overridden."""
        return len(prompt) > 0
    
    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Must be implemented by subclasses."""
        pass
    
    def _postprocess(self, response: str) -> str:
        """Default postprocessing - can be overridden."""
        return response


class ChatWorkflow(LLMWorkflow):
    """Chat-specific workflow."""
    
    def _preprocess(self, prompt: str) -> str:
        # Add chat formatting
        return f"User: {prompt.strip()}\nAssistant:"
    
    def _call_llm(self, prompt: str) -> str:
        return f"[Chat Response to {prompt[:30]}...]"


class CodeWorkflow(LLMWorkflow):
    """Code generation workflow."""
    
    def _preprocess(self, prompt: str) -> str:
        return f"Write Python code: {prompt.strip()}"
    
    def _call_llm(self, prompt: str) -> str:
        return f"```python\n# Code for: {prompt[:20]}\nprint('hello')\n```"
    
    def _postprocess(self, response: str) -> str:
        # Extract just the code
        return response


# Same workflow structure, different behavior
chat_flow = ChatWorkflow()
code_flow = CodeWorkflow()

print(f"Chat: {chat_flow.execute('Hello!')}")
print(f"Code: {code_flow.execute('sort a list')}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("BEHAVIORAL PATTERNS FOR AI SYSTEMS")
print("=" * 60)
print("""
┌────────────────────────────────────────────────────────────┐
│ STRATEGY:                                                   │
│   Use for: Interchangeable algorithms                       │
│   AI Use: Switch LLM providers, embedding models            │
├────────────────────────────────────────────────────────────┤
│ OBSERVER:                                                   │
│   Use for: Event notification                               │
│   AI Use: Agent events, logging, metrics                    │
├────────────────────────────────────────────────────────────┤
│ CHAIN OF RESPONSIBILITY:                                    │
│   Use for: Processing pipeline                              │
│   AI Use: Input validation, filtering, enrichment           │
├────────────────────────────────────────────────────────────┤
│ COMMAND:                                                    │
│   Use for: Encapsulate actions, undo/redo                   │
│   AI Use: Agent action history, rollback                    │
├────────────────────────────────────────────────────────────┤
│ STATE:                                                      │
│   Use for: State-dependent behavior                         │
│   AI Use: Agent lifecycle (idle→thinking→executing)         │
├────────────────────────────────────────────────────────────┤
│ TEMPLATE METHOD:                                            │
│   Use for: Define algorithm skeleton                        │
│   AI Use: LLM call workflow, prompt processing              │
└────────────────────────────────────────────────────────────┘
""")
