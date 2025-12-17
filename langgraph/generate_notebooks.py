#!/usr/bin/env python3
"""
Complete notebook generator for Lang Graph course (Modules 01-08).
Generates practice notebooks with exercises aligned to 2024-2025 features.
"""

import json
from pathlib import Path

BASE_DIR = Path("/Users/girijesh/Documents/ai-interview-codex/langgraph")

def create_notebook_structure(cells):
    """Create standard notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def create_cell(cell_type, source, metadata=None):
    """Helper to create notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

# Module 04: HITL with InjectedState
def module_04_cells():
    return [
        create_cell("markdown", [
            "# Module 04: Human-in-the-Loop Patterns\n\n",
            "**Updated:** December 2025 - Includes InjectedState and Command from tools\n\n",
            "## Exercises\n",
            "1. Tool with InjectedState\n",
            "2. Tool returning Command\n",
            "3. Multi-stage approval workflow\n",
            "4. Session-aware agent\n"
        ]),
        create_cell("code", [
            "# Setup\n",
            "!pip install -q langgraph langchain langchain-openai\n\n",
            "from langchain.tools import tool\n",
            "from langgraph.prebuilt import InjectedState\n",
            "from langgraph.types import Command\n",
            "from typing import Annotated\n\n",
            "print('✅ InjectedState available!')"
        ]),
        create_cell(" markdown", [
            "## Exercise 1: Tool with InjectedState 🎯\n\n",
            "Create a tool that accesses user context without LLM seeing it.\n"
        ]),
        create_cell("code", [
            "@tool\n",
            "def search_with_context(\n",
            "    query: str,\n",
            "    state: Annotated[dict, InjectedState]\n",
            ") -> str:\n",
            "    '''TODO: Use state[\"user_id\"] to personalize search'''\n", 
            "    pass\n\n",
            "# Test showing LLM doesn't see state parameter\n",
            "# print(search_with_context.args_schema.schema())"
        ]),
        create_cell("markdown", [
            "## Exercise 2: Tool Returning Command 🎯\n\n",
            "Tool that routes based on risk assessment.\n"
        ]),
        create_cell("code", [
            "@tool\n",
            "def risky_action(\n",
            "    action: str,\n",
            "    state: Annotated[dict, InjectedState]\n",
            ") -> Command:\n",
            "    '''TODO: Return Command routing to approval if high risk'''\n",
            "    pass"
        ]),
        create_cell("markdown", [
            "## Exercise 3: Multi-Stage Approval 🎯\n\n",
            "Implement 3-stage approval with checkpoints.\n"
        ]),
        create_cell("code", [
            "from langgraph.checkpoint.memory import MemorySaver\n\n",
            "# TODO: Build workflow with interrupt_after at each approval stage\n",
            "checkpointer = MemorySaver()\n",
            "# app = workflow.compile(checkpointer=checkpointer, interrupt_after=[...])"
        ])
    ]

# Module 05: Persistence
def module_05_cells():
    return [
        create_cell("markdown", [
            "# Module 05: Persistence & Memory\n\n",
            "## Exercises\n",
            "1. PostgreSQL checkpointer\n",
            "2. Thread management\n",
            "3. State history and time-travel\n",
            "4. Memory optimization\n"
        ]),
        create_cell("code", [
            "!pip install -q langgraph psycopg2-binary\n\n",
            "from langgraph.checkpoint.postgres import PostgresSaver\n",
            "from langgraph.checkpoint.memory import MemorySaver\n\n",
            "print('✅ Checkpointers ready!')"
        ]),
        create_cell("markdown", "## Exercise 1: PostgreSQL Checkpointer 🎯\n"),
        create_cell("code", [
            "# TODO: Setup PostgresCheckpointer\n",
            "# checkpointer = PostgresSaver.from_conn_string('postgresql://...')\n",
            "# app = workflow.compile(checkpointer=checkpointer)"
        ])
    ]

# Module 06: Production Deployment  
def module_06_cells():
    return [
        create_cell("markdown", [
            "# Module 06: Production Deployment\n\n",
            "## Exercises\n",
            "1. LangSmith monitoring\n",
            "2. Deployment configuration\n",
            "3. Security patterns\n",
            "4. Load testing\n"
        ]),
        create_cell("code", [
            "import os\n",
            "os.environ['LANGCHAIN_TRACING_V2'] = 'true'\n\n",
            "from langsmith import traceable\n\n",
            "print('✅ LangSmith tracing enabled!')"
        ]),
        create_cell("markdown", "## Exercise 1: Monitoring with LangSmith 🎯\n"),
        create_cell("code", [
            "@traceable(name='critical_node', tags=['production'])\n",
            "def monitored_node(state):\n",
            "    # TODO: Implement with automatic tracing\n",
            "    pass"
        ])
    ]

# Module 07: Advanced Patterns
def module_07_cells():
    return [
        create_cell("markdown", [
            "# Module 07: Advanced Patterns & Optimization\n\n",
            "## Exercises\n",
            "1. Circuit breakers\n",
            "2. Caching strategies\n",
            "3. Performance profiling\n",
            "4. Cost optimization\n"
        ]),
        create_cell("code", [
            "!pip install -q pybreaker redis\n\n",
            "from pybreaker import CircuitBreaker\n",
            "import redis\n\n",
            "print('✅ Optimization tools ready!')"
        ])
    ]

# Module 08: Multi-Agent
def module_08_cells():
    return [
        create_cell("markdown", [
            "# Module 08: Multi-Agent Systems\n\n",
            "**Updated:** December 2025 - Command-based multi-agent patterns\n\n",
            "## Exercises\n",
            "1. Supervisor-worker with Command\n",
            "2. Agent communication protocols\n",
            "3. Dynamic team assembly\n",
            "4. Load-balanced multi-agent\n"
        ]),
        create_cell("code", [
            "from langgraph.types import Command\n\n",
            "print('✅ Multi-agent tools ready!')"
        ]),
        create_cell("markdown", "## Exercise 1: Supervisor-Worker 🎯\n"),
        create_cell("code", [
            "def supervisor(state) -> Command:\n",
            "    '''TODO: Route to specialist workers using Command'''\n",
            "    pass\n\n",
            "def research_worker(state) -> Command:\n",
            "    '''TODO: Do research, route back to supervisor'''\n",
            "    pass"
        ])
    ]

# Modules 01-02: Keep simpler since docs weren't updated
def module_01_cells():
    return [
        create_cell("markdown", "# Module 01: LangGraph Fundamentals\n\n## Basic Exercises\n"),
        create_cell("code", "!pip install -q langgraph\nprint('✅ Ready!')"),
        create_cell("markdown", "## Exercise: Build First Graph 🎯\n"),
        create_cell("code", "from langgraph.graph import StateGraph, START, END\n# TODO: Build your first graph")
    ]

def module_02_cells():
    return [
        create_cell("markdown", "# Module 02: State Management\n\n## Exercises on State Schemas\n"),
        create_cell("code", "from typing import TypedDict, Annotated\nprint('✅ State tools ready!')"),
        create_cell("markdown", "## Exercise: Custom Reducers🎯\n"),
        create_cell("code", "# TODO: Implement custom state reducer")
    ]

# Generate all notebooks
def generate_all():
    modules = {
        "01": module_01_cells(),
        "02": module_02_cells(),
        "04": module_04_cells(),
        "05": module_05_cells(),
        "06": module_06_cells(),
        "07": module_07_cells(),
        "08": module_08_cells(),
    }
    
    for num, cells in modules.items():
        notebook = create_notebook_structure(cells)
        path = BASE_DIR / f"module-{num}-practice.ipynb"
        with open(path, 'w') as f:
            json.dump(notebook, f, indent=2)
        print(f"✅ Created module-{num}-practice.ipynb")
    
    print(f"\n🎉 Generated {len(modules)} practice notebooks!")
    print("Note: Module 03 was created earlier with full exercises")

if __name__ == "__main__":
    generate_all()
