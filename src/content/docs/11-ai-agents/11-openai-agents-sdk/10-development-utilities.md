---
title: "Development Utilities"
---

# Development Utilities

## Introduction

The OpenAI Agents SDK includes developer tools that accelerate building and debugging agents. The **demo REPL** provides an interactive terminal for rapid testing, and the **visualization** module renders agent graphs. These utilities are for development only — not production.

### What we'll cover

- The interactive demo REPL (`run_demo_loop`)
- Agent visualization and graph rendering
- Debugging tips and common workflows

### Prerequisites

- [Agent Class Fundamentals](./01-agent-class-fundamentals.md)
- [Runner Execution Model](./02-runner-execution-model.md)

---

## Interactive demo REPL

`run_demo_loop` launches an interactive terminal session with your agent. It handles input, streams responses, and maintains conversation history — perfect for rapid prototyping:

```python
from agents import Agent, function_tool, Runner

@function_tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.
    
    Args:
        expression: The math expression to evaluate.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

agent = Agent(
    name="Calculator",
    instructions="Help users with math. Use the calculate tool for computations.",
    tools=[calculate],
)

if __name__ == "__main__":
    from agents import run_demo_loop
    import asyncio
    asyncio.run(run_demo_loop(agent))
```

### Running the REPL

```bash
python my_agent.py
```

**Output:**
```
Agent: Calculator
Enter your message (or 'quit'/'exit' to stop):

> What is 42 * 17?

Agent: 42 × 17 = 714

> And divide that by 3

Agent: 714 ÷ 3 = 238

> quit
```

### REPL features

| Feature | Behavior |
|---------|----------|
| **Streaming** | Responses appear token-by-token as they're generated |
| **Conversation history** | Maintains context across turns |
| **Tool execution** | Tools run automatically; results are displayed |
| **Handoffs** | If the agent hands off, the REPL follows the handoff |
| **Exit** | Type `quit`, `exit`, or press `Ctrl+D` |

### Using REPL with context

```python
from dataclasses import dataclass
from agents import Agent, run_demo_loop

@dataclass
class DevContext:
    debug_mode: bool = True
    request_count: int = 0

agent = Agent[DevContext](
    name="Debug Agent",
    instructions="You are in debug mode. Be extra verbose.",
)

import asyncio

ctx = DevContext(debug_mode=True)
asyncio.run(run_demo_loop(agent, context=ctx))
```

---

## Agent visualization

The SDK can render agent configurations as visual graphs, showing tools, handoffs, and relationships:

```python
from agents import Agent, function_tool

@function_tool
def search(query: str) -> str:
    """Search for information.
    
    Args:
        query: Search query.
    """
    return f"Results for: {query}"

@function_tool
def save_note(note: str) -> str:
    """Save a note.
    
    Args:
        note: The note content.
    """
    return f"Saved: {note}"

# Define agents with handoffs
researcher = Agent(
    name="Researcher",
    instructions="Research topics using the search tool.",
    tools=[search],
)

writer = Agent(
    name="Writer",
    instructions="Write content and save notes.",
    tools=[save_note],
)

coordinator = Agent(
    name="Coordinator",
    instructions="Coordinate research and writing tasks.",
    handoffs=[researcher, writer],
)
```

### Inspecting agent configuration

Even without the visualization module, we can inspect agent structure programmatically:

```python
def inspect_agent(agent: Agent, indent: int = 0) -> None:
    """Print an agent's configuration tree."""
    prefix = "  " * indent
    print(f"{prefix}📋 Agent: {agent.name}")
    print(f"{prefix}   Model: {agent.model}")
    
    if agent.tools:
        print(f"{prefix}   🔧 Tools:")
        for tool in agent.tools:
            print(f"{prefix}      - {tool.name}")
    
    if agent.handoffs:
        print(f"{prefix}   🤝 Handoffs:")
        for handoff in agent.handoffs:
            target = handoff.agent if hasattr(handoff, 'agent') else handoff
            print(f"{prefix}      → {target.name}")
            inspect_agent(target, indent + 2)

inspect_agent(coordinator)
```

**Output:**
```
📋 Agent: Coordinator
   Model: gpt-4o
   🤝 Handoffs:
      → Researcher
        📋 Agent: Researcher
           Model: gpt-4o
           🔧 Tools:
              - search
      → Writer
        📋 Agent: Writer
           Model: gpt-4o
           🔧 Tools:
              - save_note
```

---

## Debugging workflows

### Pattern 1: Verbose logging

```python
import logging

# Enable SDK debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai.agents").setLevel(logging.DEBUG)

# Now every SDK operation logs details
result = Runner.run_sync(agent, "Hello")
```

### Pattern 2: Inspect run results

```python
from agents import Agent, Runner

agent = Agent(name="Test", instructions="Be helpful.")
result = Runner.run_sync(agent, "Hello")

# Inspect the result object
print(f"Final output: {result.final_output}")
print(f"Last agent: {result.last_agent.name}")
print(f"Input list length: {len(result.to_input_list())}")

# Check raw items for debugging
for item in result.raw_responses:
    print(f"  Response ID: {item.id}")
    print(f"  Usage: {item.usage}")
```

### Pattern 3: Test agents in isolation

```python
from agents import Agent, Runner, RunConfig

# Test with limited turns to catch infinite loops
config = RunConfig(max_turns=3)

try:
    result = Runner.run_sync(agent, "Complex task", run_config=config)
except Exception as e:
    print(f"Agent failed within 3 turns: {e}")
```

### Pattern 4: Quick smoke test

```python
def smoke_test(agent: Agent, test_inputs: list[str]) -> None:
    """Quick test an agent with multiple inputs."""
    print(f"\n🧪 Testing: {agent.name}")
    print("=" * 50)
    
    for i, input_text in enumerate(test_inputs, 1):
        try:
            result = Runner.run_sync(agent, input_text)
            status = "✅"
            output = result.final_output[:100]
        except Exception as e:
            status = "❌"
            output = str(e)[:100]
        
        print(f"  {status} Test {i}: '{input_text[:50]}...'")
        print(f"     → {output}")

smoke_test(agent, [
    "Hello, how are you?",
    "What's the weather?",
    "Help me with code",
])
```

**Output:**
```
🧪 Testing: Assistant
==================================================
  ✅ Test 1: 'Hello, how are you?...'
     → I'm doing well! How can I help you today?
  ✅ Test 2: 'What's the weather?...'
     → I don't have access to weather data, but I can help with other things!
  ✅ Test 3: 'Help me with code...'
     → Sure! What programming language and what are you trying to build?
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use `run_demo_loop` for first tests | Faster than writing test scripts |
| Inspect agents before running | Catch misconfigured tools and handoffs early |
| Set `max_turns=5` during development | Prevents runaway agents from burning tokens |
| Use `smoke_test` for regression checks | Catches breaking changes after edits |
| Enable debug logging for hard-to-find bugs | See exactly what the SDK sends and receives |
| Never use `run_demo_loop` in production | It's a development tool, not a production interface |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using REPL in production code | Use `Runner.run()` with proper input handling |
| No max_turns limit during testing | Always set `RunConfig(max_turns=N)` during development |
| Debugging without tracing | Enable tracing to see the full execution timeline |
| Testing handoffs without inspecting agent tree | Use `inspect_agent()` to verify handoff configuration |
| Forgetting `asyncio.run()` for `run_demo_loop` | The REPL is async — wrap in `asyncio.run()` in scripts |

---

## Hands-on exercise

### Your task

Build a **multi-agent system** and test it thoroughly using the SDK's development utilities.

### Requirements

1. Create three agents: `Router`, `FAQ Bot`, `Technical Support` with handoffs
2. Use `inspect_agent()` to print the agent tree
3. Write a `smoke_test()` function that tests 5 different inputs
4. Set up `run_demo_loop` for interactive testing
5. Add `RunConfig(max_turns=5)` to prevent runaway execution

### Expected result

The agent tree prints correctly, smoke tests pass, and the REPL allows interactive testing.

<details>
<summary>💡 Hints (click to expand)</summary>

- Router agent: `handoffs=[faq_bot, tech_support]`
- FAQ Bot: answers common questions, no tools needed
- Tech Support: has a `create_ticket` tool
- Smoke test inputs: general greeting, FAQ question, technical issue, edge case, gibberish

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import asyncio
from agents import Agent, Runner, RunConfig, function_tool, run_demo_loop

@function_tool
def create_ticket(title: str, description: str) -> str:
    """Create a support ticket.
    
    Args:
        title: Ticket title.
        description: Ticket description.
    """
    return f"Ticket created: '{title}' (ID: TKT-001)"

faq_bot = Agent(
    name="FAQ Bot",
    instructions="Answer common questions about our product. Keep answers brief.",
)

tech_support = Agent(
    name="Technical Support",
    instructions="Help with technical issues. Create tickets for complex problems.",
    tools=[create_ticket],
)

router = Agent(
    name="Router",
    instructions="""Route user requests:
    - General questions → FAQ Bot
    - Technical issues → Technical Support
    Always hand off — don't answer directly.""",
    handoffs=[faq_bot, tech_support],
)

# 1. Inspect the agent tree
def inspect_agent(agent, indent=0):
    prefix = "  " * indent
    print(f"{prefix}📋 {agent.name}")
    if agent.tools:
        for tool in agent.tools:
            print(f"{prefix}  🔧 {tool.name}")
    if agent.handoffs:
        for h in agent.handoffs:
            target = h.agent if hasattr(h, 'agent') else h
            inspect_agent(target, indent + 1)

print("Agent Tree:")
inspect_agent(router)

# 2. Smoke test
def smoke_test(agent, inputs):
    config = RunConfig(max_turns=5)
    print(f"\n🧪 Testing: {agent.name}")
    for i, text in enumerate(inputs, 1):
        try:
            result = Runner.run_sync(agent, text, run_config=config)
            print(f"  ✅ Test {i}: {text[:40]} → {result.final_output[:60]}")
        except Exception as e:
            print(f"  ❌ Test {i}: {text[:40]} → {str(e)[:60]}")

smoke_test(router, [
    "Hello!",
    "What does your product do?",
    "My app keeps crashing on login",
    "asdfghjkl",
    "I need a refund",
])

# 3. Interactive REPL
if input("\nStart interactive REPL? (y/n): ").lower() == "y":
    asyncio.run(run_demo_loop(router))
```

</details>

### Bonus challenges

- [ ] Add a `StatsHook` that counts tool calls and handoffs during testing
- [ ] Create a `benchmark()` function that measures response times for each test
- [ ] Export smoke test results to a JSON file for tracking over time

---

## Summary

✅ `run_demo_loop(agent)` provides an interactive REPL for rapid testing with streaming and history

✅ Agent inspection reveals tools, handoffs, and configuration before running

✅ `RunConfig(max_turns=N)` prevents runaway agents during development

✅ Smoke tests with multiple inputs catch regressions quickly

✅ Debug logging (`logging.getLogger("openai.agents")`) shows SDK internals

**Next:** [LiteLLM Model Support](./11-litellm-model-support.md)

---

## Further reading

- [REPL reference](https://openai.github.io/openai-agents-python/ref/repl/) — run_demo_loop API
- [Visualization docs](https://openai.github.io/openai-agents-python/visualization/) — Agent graph rendering
- [RunConfig reference](https://openai.github.io/openai-agents-python/ref/run_config/) — Configuration options

---

*[Back to OpenAI Agents SDK Overview](./00-openai-agents-sdk.md)*

<!-- 
Sources Consulted:
- REPL reference: https://openai.github.io/openai-agents-python/ref/repl/
- Visualization: https://openai.github.io/openai-agents-python/visualization/
- Running agents: https://openai.github.io/openai-agents-python/running_agents/
-->
