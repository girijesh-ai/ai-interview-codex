import json

# Create comprehensive notebook with all 6 patterns
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Helper function to create cells
def md(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def code(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

# Add cells
cells = [
    md("""# Anthropic Agentic Patterns: Complete Guide with Theory + Code

**Based on:** Anthropic Research "Building Effective Agents" (2025)  
**Authors:** Erik Schluntz, Barry Zhang  
**Source:** https://www.anthropic.com/research/building-effective-agents

## Table of Contents

1. [Foundational Concepts](#foundation)
2. [Pattern 1: Prompt Chaining](#pattern1)
3. [Pattern 2: Routing](#pattern2)
4. [Pattern 3: Parallelization](#pattern3)
5. [Pattern 4: Orchestrator-Workers](#pattern4)
6. [Pattern 5: Evaluator-Optimizer](#pattern5)
7. [Pattern 6: Autonomous Agents](#pattern6)
8. [Production Best Practices](#production)
9. [Interview Q&A](#interview)"""),

    md("""## Setup"""),
    
    code("""# Install required packages
!pip install -q anthropic python-dotenv matplotlib"""),

    code("""import anthropic
import os
import json
import asyncio
import time
from collections import Counter
from typing import List, Dict, Callable, Literal
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
print("✓ Setup complete!")"""),

    md("""<a id="foundation"></a>
# Foundational Concepts

## Workflows vs Agents: The Critical Distinction

```mermaid
graph TB
    subgraph "WORKFLOWS"
        W1[Input] --> W2[Step 1<br/>Predefined]
        W2 --> W3[Step 2<br/>Predefined]
        W3 --> W4[Step 3<br/>Predefined]
        W4 --> W5[Output]
        style W2 fill:#c8e6c9
        style W3 fill:#c8e6c9
        style W4 fill:#c8e6c9
    end
    
    subgraph "AGENTS"
        A1[Input] --> A2{LLM<br/>Decides}
        A2 -->|Tool A| A3[Execute]
        A2 -->|Tool B| A4[Execute]
        A3 --> A2
        A4 --> A2
        A2 -->|Done| A5[Output]
        style A2 fill:#ffab91
    end
```

### Key Definitions

**Workflows (Patterns 1-5):**
- Developer defines explicit control flow
- Deterministic, predictable execution
- Lower cost and latency
- Easier to debug and test

**Agents (Pattern 6):**
- LLM dynamically decides actions
- Adaptive to unexpected situations  
- Higher cost and latency
- Requires extensive testing

### Anthropic's Golden Rule

> Start simple. Use workflows (patterns 1-5) first. Only move to agents when you have a clear need that workflows can't address.

## The 6 Patterns Overview

```mermaid
graph LR
    A[Simple] --> B[Prompt Chaining]
    B --> C[Routing]
    C --> D[Parallelization]
    D --> E[Orchestrator-Workers]
    E --> F[Evaluator-Optimizer]
    F --> G[Autonomous Agents]
    G --> H[Complex]
    
    style B fill:#c8e6c9
    style C fill:#c8e6c9
    style D fill:#fff9c4
    style E fill:#ffe0b2
    style F fill:#ffe0b2
    style G fill:#ffab91
```

| Pattern | Complexity | Cost | Use When |
|---------|-----------|------|----------|
| **1. Prompt Chaining** | Low | $ | Sequential multi-step tasks |
| **2. Routing** | Low | $ | Distinct query categories |
| **3. Parallelization** | Medium | $$$ | Independent subtasks or voting |
| **4. Orchestrator-Workers** | High | $$$$ | Dynamic task decomposition |
| **5. Evaluator-Optimizer** | High | $$$$ | Iterative quality improvement |
| **6. Autonomous Agents** | Very High | $$$$$ | Unpredictable, adaptive tasks |"""),

    md("""<a id="pattern1"></a>
---
# Pattern 1: Prompt Chaining

## Theory

**Definition:** Decompose tasks into sequential steps where each LLM call processes the previous output.

**When to use:**
- Clear sequence of transformations
- Document generation pipelines
- Tasks requiring intermediate validation

**Architecture:**

```mermaid
sequenceDiagram
    participant User
    participant Step1 as LLM Step 1
    participant Step2 as LLM Step 2
    participant Step3 as LLM Step 3

    User->>Step1: Input
    Step1-->>Step2: Output 1
    Step2-->>Step3: Output 2
    Step3-->>User: Final Output
    
    Note over Step1,Step3: Each step specialized<br/>for one task
```

**Tradeoffs:**
- ✅ Higher accuracy (specialized prompts)
- ✅ Easier debugging (inspect intermediate steps)
- ✅ Better control
- ⚠️ Higher latency (sequential calls)
- ⚠️ Multiple API costs"""),

    code("""# Pattern 1 Implementation
def prompt_chaining_marketing(product_description: str) -> dict:
    print("=" * 60)
    print("PROMPT CHAINING: 3-Step Marketing Pipeline")
    print("=" * 60)
    
    # Step 1: Extract features
    print("\\n[Step 1/3] Extracting features...")
    step1 = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Extract 3-5 key features from:\\n{product_description}"
        }]
    )
    features = step1.content[0].text
    print(f"Features:\\n{features}")
    
    # Step 2: Generate marketing copy
    print("\\n[Step 2/3] Generating marketing copy...")
    step2 = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Create compelling marketing copy from these features:\\n{features}"
        }]
    )
    copy = step2.content[0].text
    print(f"Marketing copy:\\n{copy[:200]}...")
    
    # Step 3: Translate
    print("\\n[Step 3/3] Translating to Spanish...")
    step3 = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Translate to Spanish:\\n{copy}"
        }]
    )
    spanish = step3.content[0].text
    
    print("\\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    return {"features": features, "english": copy, "spanish": spanish}

# Test
product = "SmartWatch Pro X: 7-day battery, heart rate monitoring, GPS, water-resistant 50m"
result = prompt_chaining_marketing(product)"""),

    md("""<a id="pattern2"></a>
---
# Pattern 2: Routing

## Theory

**Definition:** Classify input and route to specialized handlers.

**Architecture:**

```mermaid
graph TD
    A[User Query] --> B[Router LLM]
    B -->|technical| C[Tech Handler]
    B -->|billing| D[Billing Handler]
    B -->|general| E[General Handler]
    C --> F[Response]
    D --> F
    E --> F
    
    style B fill:#fff9c4
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#e1f5ff
```

**Benefits:**
- Specialized prompts per category
- Can route to different models (Haiku vs Sonnet)
- Cost optimization
- Better accuracy per domain"""),

    code("""# Pattern 2 Implementation
def route_and_handle(query: str) -> dict:
    # Step 1: Route
    route_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f'''Classify this query: "{query}"
Categories: technical, billing, general
Respond with ONLY the category name.'''
        }]
    )
    
    category = route_response.content[0].text.strip().lower()
    print(f"Routed to: {category.upper()}")
    
    # Step 2: Handle with specialized prompt
    handlers = {
        "technical": "You are a technical support specialist. Provide detailed troubleshooting.",
        "billing": "You are a billing specialist. Explain charges clearly.",
        "general": "You are a customer service rep. Be helpful and concise."
    }
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=handlers.get(category, handlers["general"]),
        messages=[{"role": "user", "content": query}]
    )
    
    return {"category": category, "response": response.content[0].text}

# Test
queries = [
    "My app keeps crashing when exporting data",
    "I was charged twice this month",
    "What are your business hours?"
]

for q in queries:
    print(f"\\nQuery: {q}")
    result = route_and_handle(q)
    print(f"Response: {result['response'][:150]}...\\n" + "-"*60)"""),

    md("""<a id="pattern3"></a>
---
# Pattern 3: Parallelization

## Theory

**Two Variants:**
1. **Voting:** Same task multiple times (diverse perspectives)
2. **Sectioning:** Split task into independent parts

**Voting Architecture:**

```mermaid
graph TD
    A[Content] --> B[Eval 1: Conservative]
    A --> C[Eval 2: Balanced]
    A --> D[Eval 3: Permissive]
    B --> E[Vote: UNSAFE]
    C --> F[Vote: SAFE]
    D --> G[Vote: SAFE]
    E --> H[Majority Vote]
    F --> H
    G --> H
    H --> I[Final: SAFE]
```

**Performance:**

```mermaid
gantt
    title Sequential vs Parallel (3 calls)
    dateFormat X
    axisFormat %s
    section Sequential
    Call 1: 0, 2
    Call 2: 2, 4
    Call 3: 4, 6
    section Parallel
    Call 1: 0, 2
    Call 2: 0, 2
    Call 3: 0, 2
```

**Speedup:** ~3x with 3 parallel calls"""),

    code("""# Pattern 3 Implementation (Voting)
async def call_llm_async(prompt: str, system: str = None) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system if system else anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

async def voting_moderation(content: str):
    perspectives = [
        "You are CONSERVATIVE. Prioritize safety.",
        "You are BALANCED. Consider context.",
        "You are PERMISSIVE. Allow unless clearly harmful."
    ]
    
    prompt = f'Evaluate if safe for public platform: "{content}"\\nRespond: SAFE or UNSAFE + reasoning'
    
    # Execute in parallel
    tasks = [call_llm_async(prompt, sys) for sys in perspectives]
    responses = await asyncio.gather(*tasks)
    
    # Count votes
    votes = [("safe" if "SAFE" in r and "UNSAFE" not in r else "unsafe") for r in responses]
    decision = Counter(votes).most_common(1)[0][0]
    
    print(f"Votes: {dict(Counter(votes))} → Final: {decision.upper()}")
    return decision

# Test
content = "Check out this amazing product at totally-legit-site.com!"
result = await voting_moderation(content)"""),

    md("""<a id="pattern4"></a>
---
# Pattern 4: Orchestrator-Workers

## Theory

**Definition:** Central orchestrator dynamically decomposes task, delegates to workers, synthesizes results.

**Architecture:**

```mermaid
graph TD
    A[Complex Task] --> B[Orchestrator:<br/>Plan]
    B --> C[Subtask 1]
    B --> D[Subtask 2]
    B --> E[Subtask 3]
    C --> F[Worker 1]
    D --> G[Worker 2]
    E --> H[Worker 3]
    F --> I[Orchestrator:<br/>Synthesize]
    G --> I
    H --> I
    I --> J[Final Output]
    
    style B fill:#fff9c4
    style I fill:#fff9c4
```

**Workflow:**
1. Orchestrator plans subtasks
2. Workers execute in parallel
3. Orchestrator synthesizes results

**Most expensive** but most flexible pattern."""),

    code("""# Pattern 4 Implementation
class OrchestratorWorker:
    def __init__(self, client):
        self.client = client
    
    def plan(self, task: str) -> list:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f'''Break this task into 3-4 subtasks as JSON array:
Task: {task}

Format: [{{"id": 1, "description": "...", "expertise": "..."}}]'''
            }]
        )
        
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        return json.loads(text.strip())
    
    def execute_worker(self, subtask: dict) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"You are a {subtask['expertise']} specialist.\\n\\nExecute: {subtask['description']}"
            }]
        )
        return response.content[0].text
    
    def synthesize(self, task: str, results: list) -> str:
        combined = "\\n\\n".join([f"Subtask {r['id']}: {r['result']}" for r in results])
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"Synthesize these results for task '{task}':\\n\\n{combined}"
            }]
        )
        return response.content[0].text
    
    def execute(self, task: str) -> str:
        print("Phase 1: Planning...")
        subtasks = self.plan(task)
        print(f"Created {len(subtasks)} subtasks")
        
        print("\\nPhase 2: Workers executing...")
        results = []
        for st in subtasks:
            result = self.execute_worker(st)
            results.append({"id": st["id"], "result": result})
            print(f"  Worker {st['id']} complete")
        
        print("\\nPhase 3: Synthesizing...")
        return self.synthesize(task, results)

# Test
ow = OrchestratorWorker(client)
task = "Create go-to-market strategy for AI productivity app for remote teams"
final = ow.execute(task)
print(f"\\nFinal output:\\n{final[:300]}...")"""),

    md("""<a id="pattern5"></a>
---
# Pattern 5: Evaluator-Optimizer

## Theory

**Definition:** Iterative refinement through generate → evaluate → refine loop.

**Workflow:**

```mermaid
graph LR
    A[Task] --> B[Generator]
    B --> C[Output v1]
    C --> D[Evaluator]
    D --> E{Score >= 8?}
    E -->|No| F[Feedback]
    F --> B
    E -->|Yes| G[Final Output]
    
    style B fill:#e1f5ff
    style D fill:#fff9c4
```

**Use Cases:**
- High-quality writing
- Code generation with tests
- Tasks with clear quality criteria"""),

    code("""# Pattern 5 Implementation
class EvaluatorOptimizer:
    def __init__(self, client, max_iterations=3):
        self.client = client
        self.max_iterations = max_iterations
    
    def generate(self, task: str, feedback: str = None) -> str:
        prompt = task if not feedback else f"{task}\\n\\nImprove based on feedback:\\n{feedback}"
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def evaluate(self, task: str, output: str) -> dict:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f'''Evaluate this output:
Task: {task}
Output: {output}

Provide JSON: {{"score": 1-10, "strengths": [...], "weaknesses": [...], "suggestions": [...]}}'''
            }]
        )
        
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        return json.loads(text.strip())
    
    def execute(self, task: str, threshold=8) -> dict:
        current_output = None
        feedback = None
        
        for i in range(self.max_iterations):
            print(f"\\nIteration {i+1}")
            
            # Generate
            current_output = self.generate(task, feedback)
            print(f"Generated {len(current_output)} chars")
            
            # Evaluate
            eval_result = self.evaluate(task, current_output)
            score = eval_result['score']
            print(f"Score: {score}/10")
            
            if score >= threshold:
                print("Threshold met!")
                break
            
            # Prepare feedback
            feedback = "\\n".join(eval_result.get('suggestions', []))
        
        return {"output": current_output, "score": score, "iterations": i+1}

# Test
eo = EvaluatorOptimizer(client)
task = "Write professional email explaining 2-week project delay due to API issues"
result = eo.execute(task, threshold=8)
print(f"\\nFinal (score {result['score']}/10):\\n{result['output'][:200]}...")"""),

    md("""<a id="pattern6"></a>
---
# Pattern 6: Autonomous Agents

## Theory

**Definition:** LLM dynamically selects tools and controls its own execution loop.

**Architecture:**

```mermaid
graph TD
    A[User Query] --> B{Agent LLM}
    B -->|Needs tool| C[Select Tool]
    C --> D[Execute Tool]
    D --> E[Observe Result]
    E --> B
    B -->|Complete| F[Final Answer]
    
    style B fill:#ffab91
```

**Anthropic's Warning:** 
> Agents have higher costs and error rates. Use extensively tested guardrails.

**Best for:**
- Customer support
- Research tasks
- Code generation"""),

    code("""# Pattern 6 Implementation
TOOLS = [
    {
        "name": "search_web",
        "description": "Search for information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Math calculations",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

def search_web(query: str) -> str:
    return f"Search results for '{query}': Latest AI research shows significant progress..."

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

TOOL_FUNCTIONS = {"search_web": search_web, "calculate": calculate}

class AutonomousAgent:
    def __init__(self, client, tools, tool_functions, max_iterations=5):
        self.client = client
        self.tools = tools
        self.tool_functions = tool_functions
        self.max_iterations = max_iterations
    
    def run(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        
        for i in range(self.max_iterations):
            print(f"Iteration {i+1}")
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                tools=self.tools,
                messages=messages
            )
            
            if response.stop_reason == "end_turn":
                final = next((b.text for b in response.content if hasattr(b, "text")), None)
                print("Agent complete!")
                return final
            
            elif response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  Using tool: {block.name}")
                        result = self.tool_functions[block.name](**block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                messages.append({"role": "user", "content": tool_results})
        
        return "Max iterations reached"

# Test
agent = AutonomousAgent(client, TOOLS, TOOL_FUNCTIONS)
result = agent.run("Calculate 15% of 240, then search for AI agent best practices")
print(f"\\nFinal: {result}")"""),

    md("""<a id="production"></a>
---
# Production Best Practices

## Pattern Selection Guide

```mermaid
flowchart TD
    Start[Task Requirements] --> Q1{Clear sequential<br/>steps?}
    Q1 -->|Yes| Chain[Prompt Chaining]
    Q1 -->|No| Q2{Distinct<br/>categories?}
    Q2 -->|Yes| Route[Routing]
    Q2 -->|No| Q3{Independent<br/>subtasks?}
    Q3 -->|Yes| Para[Parallelization]
    Q3 -->|No| Q4{Complex dynamic<br/>decomposition?}
    Q4 -->|Yes| Orch[Orchestrator-Workers]
    Q4 -->|No| Q5{Need iterative<br/>refinement?}
    Q5 -->|Yes| Eval[Evaluator-Optimizer]
    Q5 -->|No| Q6{Truly unpredictable<br/>needs?}
    Q6 -->|Yes| Agent[Autonomous Agent]
    Q6 -->|No| Chain2[Prompt Chaining<br/>Simplified]
```

## Cost vs Performance

| Pattern | Latency | Cost | Reliability | Use When |
|---------|---------|------|-------------|----------|
| Prompt Chaining | Medium | $ | High | Sequential pipelines |
| Routing | Low | $ | High | Categorization needed |
| Parallelization | Low | $$$ | Medium | Speed critical |
| Orchestrator-Workers | High | $$$$ | Medium | Complex coordination |
| Evaluator-Optimizer | High | $$$$ | High | Quality critical |
| Autonomous Agents | Variable | $$$$$ | Low | Dynamic adaptation |"""),

    md("""<a id="interview"></a>
---
# Interview Q&A

## Common Questions

**Q1: Explain workflows vs agents.**

**Answer:**
- **Workflows (Patterns 1-5):** Developer-defined control flow. Deterministic, cheaper, easier to debug.
- **Agents (Pattern 6):** LLM-controlled execution. Adaptive but expensive and requires extensive testing.
- **When to use:** Start with workflows. Use agents only when dynamic adaptation is essential.

**Q2: When would you use prompt chaining vs parallelization?**

**Answer:**
- **Prompt Chaining:** Steps depend on previous outputs (e.g., extract → generate → translate)
- **Parallelization:** Independent subtasks or need multiple perspectives (e.g., content moderation voting)

**Q3: How do you optimize agentic system costs?**

**Answer:**
1. Use workflows instead of agents when possible
2. Cache results for repeated queries
3. Use smaller models (Haiku) for simple steps
4. Limit iterations in evaluator-optimizer
5. Monitor and alert on cost spikes

**Q4: What are the risks of autonomous agents?**

**Answer:**
- Higher costs (unpredictable iterations)
- Lower reliability (can make wrong tool choices)
- Harder to debug
- **Mitigation:** Extensive testing, safety guardrails, rate limiting, human-in-loop for critical actions

**Q5: Which pattern for document generation with translation?**

**Answer:** Prompt Chaining
- Clear sequence: extract features → generate copy → translate
- Each step specialized
- Easy to debug intermediate outputs
- Trade-off: 3 API calls vs 1, but higher quality"""),

    md("""---
# Summary

## What We Covered

All 6 agentic patterns from Anthropic's research:

1. ✅ **Prompt Chaining** - Sequential multi-step workflows
2. ✅ **Routing** - Classification and specialized handling
3. ✅ **Parallelization** - Voting and sectioning for speed/quality
4. ✅ **Orchestrator-Workers** - Dynamic task decomposition
5. ✅ **Evaluator-Optimizer** - Iterative refinement loops
6. ✅ **Autonomous Agents** - Self-directed tool usage

## Key Takeaways

- Start simple (workflows) before moving to agents
- Match pattern complexity to task requirements
- Monitor costs and quality in production
- Use specialized prompts per step/category
- Test extensively, especially for agents

## Resources

- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Anthropic Cookbook: Agent Patterns](https://github.com/anthropics/anthropic-cookbook)
- Next: LangGraph workflows, Multi-agent systems""")
]

# Add all cells
notebook["cells"] = cells

# Save
with open('01-anthropic-patterns-interactive.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Comprehensive notebook created with theory + diagrams + code!")
