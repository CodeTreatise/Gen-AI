---
title: "Chain-of-thought in agents"
---

# Chain-of-thought in agents

## Introduction

Chain-of-thought (CoT) reasoning is the practice of having an LLM show its work — breaking a problem into explicit intermediate steps before arriving at an answer. While CoT was originally a prompting technique for improving accuracy on math and logic problems, it has become an essential capability in agent systems where every decision must be traceable, debuggable, and explainable.

In the context of agents, chain-of-thought goes beyond simple problem-solving. It becomes the agent's internal monologue — the running narrative that explains *why* the agent chose a particular tool, *how* it interpreted a tool's output, and *what* it plans to do next. Without this explicit reasoning, agents become black boxes that occasionally produce correct results but offer no insight into their decision-making process.

### What we'll cover

- How chain-of-thought differs from basic prompting in agent contexts
- Explicit reasoning steps and how to structure them
- Problem decomposition techniques for complex tasks
- Reasoning documentation for auditability
- Building debug-friendly agent output

### Prerequisites

- Understanding of the [ReAct pattern](./01-react-pattern.md) and how reasoning interleaves with action
- Familiarity with the [agent loop structure](../02-agent-architecture/02-agent-loop-structure.md)
- Basic Python knowledge

---

## Chain-of-thought beyond prompting

In a standalone LLM call, chain-of-thought means adding "Let's think step by step" to a prompt. In an agent system, chain-of-thought is structural — it's woven into how the agent processes every turn of its execution loop.

### Simple CoT vs agent CoT

| Aspect | Simple CoT (single LLM call) | Agent CoT (multi-turn loop) |
|--------|------|------|
| Scope | One question, one answer | Multi-step task over many iterations |
| Grounding | Reasoning from training data only | Reasoning grounded in real tool outputs |
| Self-correction | Cannot revise reasoning | Can update reasoning based on observations |
| Persistence | Reasoning lost after response | Reasoning persists in agent state |
| Purpose | Improve accuracy | Improve accuracy *and* provide auditability |

The distinction matters because agents face challenges that single LLM calls don't:

- **Compounding errors** — Each tool call can introduce errors that affect all subsequent decisions
- **Context drift** — Over many turns, the agent can lose track of the original goal
- **Tool ambiguity** — Multiple tools might seem applicable, requiring explicit justification
- **Partial information** — The agent must reason about what it knows *and* what it doesn't

> **🔑 Key concept:** In agent systems, chain-of-thought is not an optimization technique. It's a reliability mechanism that prevents the agent from making unjustified leaps in reasoning.

---

## Explicit reasoning steps

Explicit reasoning steps are structured checkpoints where the agent articulates its thinking. Rather than letting the LLM implicitly decide what to do, we prompt it to state each step of its logic.

### The anatomy of an explicit reasoning step

Every reasoning step in an agent should answer three questions:

1. **What do I know?** — Summarize the relevant information gathered so far
2. **What do I need?** — Identify gaps in knowledge or remaining subtasks
3. **What will I do next?** — State the specific action and why it's the right choice

Here's how this looks in practice:

```python
from agents import Agent

agent = Agent(
    name="analysis_agent",
    instructions="""You are an analytical research agent.

    Before EVERY action, state your reasoning explicitly:

    KNOWN: [What facts you have so far]
    NEEDED: [What information is still missing]
    NEXT: [What specific action you'll take and why]

    After receiving tool results, evaluate them:

    RECEIVED: [Summarize what the tool returned]
    ASSESSMENT: [Is this sufficient? Any contradictions?]
    DECISION: [Continue gathering data or formulate answer]

    Never skip these reasoning steps, even for simple queries.""",
    tools=[search_tool, analyze_tool],
)
```

### Why structure matters

Without structure, reasoning traces become inconsistent. One turn the agent might reason carefully; the next it might jump straight to a tool call. Structured steps create a consistent rhythm:

```
KNOWN: The user wants to compare Python and JavaScript performance for web
scraping. I have no data yet.
NEEDED: Benchmark data or performance characteristics for both languages in
web scraping scenarios.
NEXT: I'll search for Python web scraping performance benchmarks first, since
it's the more commonly discussed language for this task.

[Tool call: search("Python web scraping performance benchmarks 2024")]

RECEIVED: Python with aiohttp can handle ~500 concurrent requests; Scrapy
processes ~2000 pages/minute on average hardware.
ASSESSMENT: Good data for Python. I still need JavaScript/Node.js comparisons.
DECISION: Search for Node.js web scraping performance next.
```

This structure serves two audiences:

- **The agent itself** — Structured reasoning reduces the chance of the LLM "forgetting" what it was doing mid-task
- **The developer** — When something goes wrong, you can pinpoint exactly where the reasoning broke down

---

## Problem decomposition

Complex questions rarely have single-step answers. Problem decomposition is the practice of breaking a hard question into simpler sub-questions that can each be answered independently.

### Decomposition strategies

Agents use several decomposition strategies depending on the task type:

#### Sequential decomposition

Each sub-question depends on the answer to the previous one:

```
Main question: "What percentage of a company's revenue comes from its newest
product?"

Decomposition:
1. What is the company's total revenue? → $10M
2. What is the newest product? → Product X, launched Q3 2024
3. What is Product X's revenue? → $2.5M
4. Calculate: $2.5M / $10M = 25%
```

#### Parallel decomposition

Sub-questions can be answered independently and combined:

```
Main question: "Compare the education systems of Finland and South Korea."

Decomposition (independent):
1. What are the key features of Finland's education system?
2. What are the key features of South Korea's education system?
3. [After both answered] Compare and synthesize findings.
```

#### Hierarchical decomposition

Sub-questions have sub-sub-questions forming a tree:

```
Main question: "Should we migrate from PostgreSQL to MongoDB?"

Level 1:
├── What are our current PostgreSQL pain points?
│   ├── Query performance issues?
│   └── Schema migration difficulties?
├── What would MongoDB improve?
│   ├── Flexible schema benefits for our use case?
│   └── Horizontal scaling needs?
└── What are the migration risks?
    ├── Data migration complexity?
    └── Application code changes needed?
```

### Implementing decomposition in agents

We can implement explicit decomposition by instructing the agent to plan before acting:

```python
from agents import Agent, Runner, function_tool

@function_tool
def research(topic: str) -> str:
    """Research a specific topic and return findings.

    Args:
        topic: A specific, focused research question.
    """
    # Simulated research results
    return f"Research findings for '{topic}': [detailed information]"

agent = Agent(
    name="decomposition_agent",
    instructions="""You are a research agent that handles complex questions
    by breaking them into simpler parts.

    When you receive a complex question:

    STEP 1 - DECOMPOSE:
    List all sub-questions needed to answer the main question.
    Mark each as [SEQUENTIAL] or [PARALLEL] based on dependencies.

    STEP 2 - EXECUTE:
    Research each sub-question, starting with independent ones.
    For sequential questions, wait for dependencies before proceeding.

    STEP 3 - SYNTHESIZE:
    Combine all findings into a coherent answer to the original question.
    Cite which sub-question produced each piece of evidence.

    Always decompose before researching. Never try to answer a complex
    question in a single tool call.""",
    tools=[research],
)

async def main():
    result = await Runner.run(
        agent,
        "What are the pros and cons of using Rust vs Go for building "
        "a high-performance web API that needs to handle 100K requests/sec?"
    )
    print(result.final_output)

import asyncio
asyncio.run(main())
```

**Output:**
```
The agent decomposes this into sub-questions about:
1. Rust's HTTP server performance characteristics
2. Go's HTTP server performance characteristics
3. Developer experience comparison (learning curve, ecosystem)
4. Deployment and operational differences
5. Synthesis comparing both for the 100K req/sec requirement
```

> **🤖 AI Context:** Problem decomposition is how agents implement the "divide and conquer" strategy. Anthropic's orchestrator-workers pattern uses this explicitly — a central LLM breaks down tasks and delegates sub-tasks to specialized worker agents.

---

## Reasoning documentation

Reasoning documentation is the practice of capturing and storing the agent's chain-of-thought for later analysis. This goes beyond logging — it creates a structured record that can be used for debugging, compliance, evaluation, and improvement.

### What to document

| Element | Purpose | Example |
|---------|---------|---------|
| Initial goal interpretation | Verify the agent understood the task | "User wants weather forecast, not current conditions" |
| Decomposition plan | Audit the approach | "Breaking into: location lookup → API call → format response" |
| Tool selection rationale | Debug wrong tool choices | "Using weather_api, not search, because user wants real-time data" |
| Observation evaluation | Catch data quality issues | "API returned data for wrong city — retrying with explicit coordinates" |
| Confidence assessment | Identify uncertain answers | "High confidence on temperature, low confidence on pollen count" |

### Implementing reasoning logs

Here's a pattern for capturing structured reasoning in the OpenAI Agents SDK using lifecycle hooks:

```python
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from agents import Agent, Runner, AgentHooks, RunContextWrapper, function_tool

@dataclass
class ReasoningLog:
    """Captures the agent's reasoning for a single run."""
    entries: list[dict] = field(default_factory=list)

    def add(self, phase: str, content: str):
        self.entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "content": content,
        })

    def export(self) -> str:
        return json.dumps(self.entries, indent=2)

class ReasoningHooks(AgentHooks):
    """Hooks that capture reasoning at each lifecycle point."""

    async def on_start(self, context: RunContextWrapper, agent: Agent):
        log = ReasoningLog()
        context.context["reasoning_log"] = log
        log.add("start", f"Agent '{agent.name}' beginning execution")

    async def on_tool_start(self, context: RunContextWrapper, agent, tool):
        log = context.context.get("reasoning_log")
        if log:
            log.add("tool_start", f"Calling tool: {tool.name}")

    async def on_tool_end(self, context: RunContextWrapper, agent, tool, result):
        log = context.context.get("reasoning_log")
        if log:
            log.add("tool_end", f"Tool '{tool.name}' returned: {result[:200]}")

    async def on_end(self, context: RunContextWrapper, agent, output):
        log = context.context.get("reasoning_log")
        if log:
            log.add("end", f"Final output produced")
            print("=== Reasoning Log ===")
            print(log.export())
```

**Output:**
```json
[
  {"timestamp": "2025-01-16T10:30:00Z", "phase": "start", "content": "Agent 'research_agent' beginning execution"},
  {"timestamp": "2025-01-16T10:30:01Z", "phase": "tool_start", "content": "Calling tool: search_facts"},
  {"timestamp": "2025-01-16T10:30:02Z", "phase": "tool_end", "content": "Tool 'search_facts' returned: Tokyo has..."},
  {"timestamp": "2025-01-16T10:30:03Z", "phase": "end", "content": "Final output produced"}
]
```

---

## Debug-friendly output

Building agents that produce debug-friendly output is the difference between a prototype you demo once and a system you can maintain in production.

### The debug visibility principle

Every agent decision should be inspectable without modifying the agent's code. This means building debug affordances in from the start, not adding them after things break.

### Techniques for debug-friendly agents

#### 1. Verbose mode via instructions

```python
import os

debug_mode = os.getenv("AGENT_DEBUG", "false") == "true"

agent = Agent(
    name="debuggable_agent",
    instructions=f"""You are a helpful assistant.
    {"DEBUG MODE: Before every action, explain your full reasoning including what alternatives you considered and why you rejected them." if debug_mode else ""}
    Use tools to answer questions accurately.""",
    tools=[search_tool],
)
```

#### 2. Structured trace output

Design agents to produce output that's both human-readable and machine-parseable:

```python
@function_tool
def log_reasoning(
    step_number: int,
    phase: str,
    reasoning: str,
    confidence: float,
) -> str:
    """Log a reasoning step for debugging.

    Call this BEFORE each tool call to document your thinking.

    Args:
        step_number: Sequential step number (1, 2, 3...).
        phase: Current phase - "analyze", "search", "evaluate", or "synthesize".
        reasoning: Your reasoning for this step.
        confidence: How confident you are (0.0 to 1.0).
    """
    entry = {
        "step": step_number,
        "phase": phase,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    print(f"[TRACE] Step {step_number} ({phase}): {reasoning}")
    return f"Reasoning logged for step {step_number}"
```

#### 3. Checkpointing with LangGraph

LangGraph's persistence layer provides built-in debugging through state checkpoints:

```python
from langgraph.checkpoint.memory import MemorySaver

# Compile graph with checkpointing
checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)

# Run with a thread_id for persistence
config = {"configurable": {"thread_id": "debug-session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    config,
)

# Inspect the state at any checkpoint
state = agent.get_state(config)
print("Current state:", state.values)
print("Step history:", state.metadata)

# Time travel: view the state at a previous checkpoint
for history in agent.get_state_history(config):
    print(f"Step: {history.metadata.get('step')}")
    print(f"Messages: {len(history.values['messages'])}")
```

**Output:**
```
Current state: {'messages': [HumanMessage(...), AIMessage(...), ToolMessage(...)]}
Step history: {'step': 3, 'source': 'loop'}
Step: 3
Messages: 5
Step: 2
Messages: 3
Step: 1
Messages: 1
```

> **💡 Tip:** LangGraph's `get_state_history()` is invaluable during development. It lets you replay the exact sequence of reasoning decisions the agent made, inspect intermediate states, and even modify state to "rewind" and replay from a different point.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Always require reasoning before action | Prevents impulsive tool calls based on pattern matching |
| Use structured reasoning formats | Makes traces consistent and machine-parseable |
| Decompose before executing | Complex questions answered in parts are more accurate |
| Log all reasoning, not just actions | Tool call logs alone don't explain *why* the agent acted |
| Include confidence levels | Helps identify when the agent is guessing vs certain |
| Design for debug from day one | Retrofitting observability is 10x harder than building it in |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Reasoning traces consume too many tokens | Set a reasoning token budget; use built-in thinking where available |
| Agent skips reasoning on "easy" questions | Make reasoning mandatory in instructions, not optional |
| Decomposition creates too many sub-tasks | Limit decomposition depth (2-3 levels max) |
| Debug output mixed into user responses | Separate internal reasoning from user-facing output |
| No reasoning on error recovery | Require explicit reasoning when a tool call fails |

---

## Hands-on exercise

### Your task

Build an agent that uses explicit chain-of-thought reasoning to solve a multi-step analysis problem, with all reasoning steps captured in a structured log.

### Requirements

1. Create an agent with a `KNOWN/NEEDED/NEXT` reasoning structure in its instructions
2. Give it tools for `lookup_data(metric: str)` and `compare(values: list)` 
3. Ask it to compare three companies on revenue, employee count, and growth rate
4. Capture all reasoning steps in a list and print them at the end

### Expected result

A structured reasoning log showing the agent's thought process at each step, followed by a synthesized comparison of the three companies.

<details>
<summary>💡 Hints (click to expand)</summary>

- Return mock data from your tools — focus on the reasoning structure, not the data
- Use a Python list as a simple accumulator for reasoning steps
- The agent should produce at least 9 tool calls (3 companies × 3 metrics)
- Consider whether some lookups can be described as parallel

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import asyncio
from agents import Agent, Runner, function_tool

reasoning_log = []

@function_tool
def lookup_data(company: str, metric: str) -> str:
    """Look up a specific business metric for a company.

    Args:
        company: Company name (e.g., "Acme Corp").
        metric: One of "revenue", "employees", or "growth_rate".
    """
    data = {
        ("acme", "revenue"): "$5.2B",
        ("acme", "employees"): "12,000",
        ("acme", "growth_rate"): "15%",
        ("globex", "revenue"): "$3.8B",
        ("globex", "employees"): "8,500",
        ("globex", "growth_rate"): "22%",
        ("initech", "revenue"): "$7.1B",
        ("initech", "employees"): "25,000",
        ("initech", "growth_rate"): "8%",
    }
    key = (company.lower(), metric.lower())
    result = data.get(key, "Data not available")
    reasoning_log.append(f"LOOKUP: {company}/{metric} → {result}")
    return result

@function_tool
def compare(values: str) -> str:
    """Compare a set of values and identify the highest and lowest.

    Args:
        values: A JSON string of company-value pairs,
                e.g., '{"Acme": "5.2", "Globex": "3.8"}'
    """
    reasoning_log.append(f"COMPARE: {values}")
    return f"Comparison complete for: {values}"

agent = Agent(
    name="analyst",
    instructions="""You are a business analyst agent.

    Before EVERY action, state your reasoning:
    KNOWN: [Facts gathered so far]
    NEEDED: [What's still missing]
    NEXT: [Your next action and why]

    Compare Acme Corp, Globex, and Initech on:
    - Revenue
    - Employee count
    - Growth rate

    Look up each metric individually, then synthesize.""",
    tools=[lookup_data, compare],
)

async def main():
    result = await Runner.run(
        agent,
        "Compare Acme Corp, Globex, and Initech across all business metrics."
    )
    print("\n=== Reasoning Log ===")
    for entry in reasoning_log:
        print(f"  {entry}")
    print(f"\n=== Final Analysis ===\n{result.final_output}")

asyncio.run(main())
```

**Expected output:**
```
=== Reasoning Log ===
  LOOKUP: Acme Corp/revenue → $5.2B
  LOOKUP: Globex/revenue → $3.8B
  LOOKUP: Initech/revenue → $7.1B
  LOOKUP: Acme Corp/employees → 12,000
  LOOKUP: Globex/employees → 8,500
  LOOKUP: Initech/employees → 25,000
  LOOKUP: Acme Corp/growth_rate → 15%
  LOOKUP: Globex/growth_rate → 22%
  LOOKUP: Initech/growth_rate → 8%

=== Final Analysis ===
Based on my research, here's the comparison:
- Revenue: Initech leads ($7.1B), followed by Acme ($5.2B), then Globex ($3.8B)
- Employees: Initech largest (25K), Acme (12K), Globex (8.5K)
- Growth: Globex fastest (22%), Acme (15%), Initech (8%)
```

</details>

### Bonus challenges

- [ ] Add a `confidence` field to each reasoning step and flag any step where confidence is below 0.7
- [ ] Implement hierarchical decomposition where the agent first plans the comparison categories, then fills in data
- [ ] Export the reasoning log as a Markdown document with headers for each phase

---

## Summary

✅ Chain-of-thought in agents is a **structural reliability mechanism**, not just a prompting trick — it prevents compounding errors across multi-step tasks

✅ Explicit reasoning steps (`KNOWN/NEEDED/NEXT`) create consistent, auditable decision-making that both the agent and developers benefit from

✅ Problem decomposition breaks complex tasks into simpler sub-questions that can be answered sequentially, in parallel, or hierarchically

✅ Reasoning documentation through structured logs enables debugging, compliance, and systematic agent improvement

✅ Debug-friendly output should be designed in from the start — use lifecycle hooks, checkpointing, and structured trace formats

**Next:** [Task Decomposition](./03-task-decomposition.md)

---

## Further reading

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — The foundational CoT paper by Wei et al. (2022)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — Orchestrator-workers pattern and decomposition
- [LangGraph: Persistence & Checkpointing](https://docs.langchain.com/oss/python/langgraph/persistence) — State inspection and time travel debugging
- [OpenAI Agents SDK: Lifecycle Hooks](https://openai.github.io/openai-agents-python/ref/lifecycle/) — Hooking into agent events for logging

*[Back to Reasoning & Planning Overview](./00-reasoning-planning.md)*

<!--
Sources Consulted:
- Anthropic Building Effective Agents (orchestrator-workers, decomposition): https://www.anthropic.com/engineering/building-effective-agents
- Google ADK LLM Agents (PlanReActPlanner structured output): https://google.github.io/adk-docs/agents/llm-agents/
- OpenAI Agents SDK (Agent hooks, lifecycle): https://openai.github.io/openai-agents-python/agents/
- Chain-of-Thought Prompting paper: https://arxiv.org/abs/2201.11903
- LangGraph overview (state management, checkpointing): https://docs.langchain.com/oss/python/langgraph/overview
-->
