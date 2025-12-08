# AI-Themed OOP Examples

This directory contains **runnable Python code** that accompanies the Python OOP Masterclass modules.

## Structure

```
examples/
├── module_01_fundamentals/     # Matches 01-oop-fundamentals-llm-clients.md
│   ├── 01_chat_message.py      # ChatMessage class basics
│   ├── 02_llm_client.py        # LLMClient with methods
│   ├── 03_properties.py        # Properties and encapsulation
│   └── 04_dunder_methods.py    # Special methods
│
├── module_02_principles/       # Matches 02-oop-principles-multi-provider.md
│   ├── 01_encapsulation.py     # Secure LLM client
│   ├── 02_inheritance.py       # Provider hierarchy
│   ├── 03_polymorphism.py      # Unified interface
│   ├── 04_abstraction.py       # ABCs and protocols
│   └── 05_mixins.py            # Streaming, function calling
│
├── module_03_advanced/         # Matches 03-advanced-oop-agent-architecture.md
│   ├── 01_metaclasses.py       # Tool auto-registration
│   ├── 02_new_vs_init.py       # Singleton pattern
│   ├── 03_descriptors.py       # Validated prompts
│   ├── 04_slots.py             # Memory optimization
│   ├── 05_context_managers.py  # Conversation sessions
│   ├── 06_generators.py        # Streaming responses
│   └── 07_decorators.py        # Retry, cache, tools
│
└── shared/                     # Reusable components
    ├── __init__.py
    ├── types.py                # Common types (ChatMessage, etc.)
    └── providers.py            # Base provider classes
```

## Usage

Each file is independently runnable:

```bash
cd examples/module_01_fundamentals
python 01_chat_message.py
```

## Following Along

1. Open the corresponding markdown module (e.g., `01-oop-fundamentals-llm-clients.md`)
2. Run the matching Python file to see the code in action
3. Modify and experiment with the code
