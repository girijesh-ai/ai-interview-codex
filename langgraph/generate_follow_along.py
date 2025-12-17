#!/usr/bin/env python3
"""
Generate follow-along notebooks for modules 03-08.
Extracts key examples from theory documents.
"""

import json
from pathlib import Path

BASE_DIR = Path("/Users/girijesh/Documents/ai-interview-codex/langgraph")

def create_cell(cell_type, source):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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

# Module 03: Command Tool
def module_03_cells():
    return [
        create_cell("markdown", "# Module 03: Advanced Control Flow - Follow Along\n\n**Key Topics:** Command tool, Edgeless graphs, Dynamic routing\n"),
        create_cell("code", "%pip install -q -U langgraph\n\nfrom langgraph.types import Command\nfrom langgraph.graph import StateGraph, START, END\nfrom typing import TypedDict\n\nprint('✅ Command tool ready!')"),
        create_cell("markdown", "## Example 1: Command-Based Routing\n"),
        create_cell("code", "class TaskState(TypedDict):\n    task: str\n    priority: str\n    result: str\n\ndef router_node(state: TaskState) -> Command:\n    if 'urgent' in state['task'].lower():\n        return Command(update={'priority': 'high'}, goto='urgent_handler')\n    return Command(update={'priority': 'normal'}, goto='normal_handler')\n\ndef urgent_handler(state: TaskState) -> Command:\n    return Command(update={'result': f\"URGENT: {state['task']}\"}, goto=END)\n\ndef normal_handler(state: TaskState) -> Command:\n    return Command(update={'result': f\"Normal: {state['task']}\"}, goto=END)\n\nworkflow = StateGraph(TaskState)\nworkflow.add_node('router', router_node)\nworkflow.add_node('urgent_handler', urgent_handler)\nworkflow.add_node('normal_handler', normal_handler)\nworkflow.add_edge(START, 'router')\napp = workflow.compile()\n\nresult = app.invoke({'task': 'urgent: fix bug', 'priority': '', 'result': ''})\nprint(f\"Result: {result['result']}\")\nprint('✅ Edgeless graph - only START edge defined!')"),
        create_cell("markdown", "## Example 2: Conditional vs Command\n"),
        create_cell("code", "# Old way: Conditional edges\ndef old_router(state):\n    return 'path_a' if state['score'] > 0.8 else 'path_b'\n\n# workflow.add_conditional_edges('node', old_router, {'path_a': 'a', 'path_b': 'b'})\n\n# New way: Command (more flexible)\ndef new_router(state) -> Command:\n    if state['score'] > 0.8:\n        return Command(update={'confidence': 'high'}, goto='path_a')\n    return Command(update={'confidence': 'low'}, goto='path_b')\n\nprint('Command provides more control - can update state AND route!')"),
        create_cell("markdown", "## Summary\n\nYou've seen:\n- ✅ Command tool for edgeless graphs\n- ✅ Dynamic routing with Command\n- ✅ Comparison with conditional edges\n\n**Next:** `module-03-practice.ipynb`")
    ]

# Continue with modules 04-08...
modules_config = {
    '03': module_03_cells(),
    # Add other modules here
}

# Generate Module 03
notebook = create_notebook(module_03_cells())
with open(BASE_DIR / "module-03-follow-along.ipynb", 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Module 03 follow-along created!")
print("Creating modules 04-08...")
