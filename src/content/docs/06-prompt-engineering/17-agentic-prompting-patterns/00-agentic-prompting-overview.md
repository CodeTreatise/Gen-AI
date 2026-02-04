---
title: "Agentic Prompting Patterns"
---

# Agentic Prompting Patterns

## Introduction

Agentic prompting is fundamentally different from single-turn text generation. Instead of crafting one perfect prompt, you're designing a system where the model autonomously plans, executes tools, observes results, and adapts its approach—all while staying on track toward a goal.

This lesson introduces the core concepts and patterns for building effective AI agents through prompting.

> **🔑 Key Insight:** Agents are not just models with tools attached. They require prompts that define goals, establish boundaries, and guide iterative reasoning across multiple turns.

### What We'll Cover

- The agent loop architecture
- Tool definition best practices
- Multi-turn context management
- Goal-oriented prompting
- MCP and dynamic tool discovery
- Safety patterns for agentic systems
- Computer use prompting

### Prerequisites

- [Function Calling & Tool Use](../../10-function-calling-tool-use/)
- [Chain of Thought Prompting](../07-chain-of-thought-prompting/)

---

## What is Agentic Prompting?

Traditional prompting is a single request-response cycle. Agentic prompting creates **autonomous loops** where the model:

1. **Receives a goal** (not step-by-step instructions)
2. **Plans** how to achieve it
3. **Executes tools** to gather information or take actions
4. **Observes** the results
5. **Adapts** its approach based on what it learned
6. **Repeats** until the goal is achieved

```
┌─────────────────────────────────────────────────────────┐
│                     USER GOAL                            │
│         "Find and summarize competitor pricing"          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   Plan   │───▶│ Execute  │───▶│ Observe  │          │
│  │          │    │   Tool   │    │ Results  │          │
│  └──────────┘    └──────────┘    └────┬─────┘          │
│       ▲                               │                 │
│       │         ┌──────────┐          │                 │
│       └─────────│  Adapt   │◀─────────┘                 │
│                 │ Approach │                            │
│                 └────┬─────┘                            │
│                      │                                  │
│              Goal Achieved?                             │
│              ─────────────                              │
│              No ↓    Yes → Exit                         │
└─────────────────────────────────────────────────────────┘
```

---

## The Tool Calling Flow

The core mechanism enabling agents is **tool calling**—the model's ability to request execution of external functions.

### OpenAI Tool Calling Flow

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Define tools
tools = [
    {
        "type": "function",
        "name": "search_database",
        "description": "Search the product database for items matching a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms for finding products"
                },
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "home", "all"],
                    "description": "Product category to search within"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# 2. Send initial request
input_messages = [
    {"role": "user", "content": "Find me wireless headphones under $100"}
]

response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_messages
)

# 3. Check for tool calls
for item in response.output:
    if item.type == "function_call":
        # 4. Execute the function
        args = json.loads(item.arguments)
        result = search_database(**args)  # Your implementation
        
        # 5. Send result back
        input_messages += response.output
        input_messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result)
        })

# 6. Get final response
final_response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_messages
)

print(final_response.output_text)
```

### Anthropic Tool Calling Flow

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "search_database",
        "description": "Search the product database for items matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms for finding products"
                },
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "home", "all"],
                    "description": "Product category to search within"
                }
            },
            "required": ["query"]
        }
    }
]

messages = [
    {"role": "user", "content": "Find me wireless headphones under $100"}
]

# Initial request
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Process tool use
while response.stop_reason == "tool_use":
    # Find tool use blocks
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            # Execute tool
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            })
    
    # Continue conversation
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

print(response.content[0].text)
```

---

## Agent Loop Patterns

### Basic Agent Loop

```python
async def agent_loop(
    client,
    model: str,
    tools: list,
    goal: str,
    max_iterations: int = 10
) -> str:
    """
    Basic agent loop that runs until goal is achieved
    or max iterations reached.
    """
    
    messages = [{"role": "user", "content": goal}]
    
    for iteration in range(max_iterations):
        response = client.responses.create(
            model=model,
            tools=tools,
            input=messages
        )
        
        # Check for tool calls
        tool_calls = [
            item for item in response.output 
            if item.type == "function_call"
        ]
        
        if not tool_calls:
            # No more tool calls - agent is done
            return response.output_text
        
        # Execute tools and add results
        messages += response.output
        for call in tool_calls:
            result = await execute_tool(call.name, call.arguments)
            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })
    
    return "Max iterations reached without completion"
```

### ReAct Pattern (Reason + Act)

The ReAct pattern alternates between reasoning and action:

```python
system_prompt = """
You are an assistant that solves problems step by step.

For each step, you MUST follow this format:

**Thought:** [Your reasoning about what to do next]
**Action:** [Tool call or final answer]
**Observation:** [Result from tool - I will provide this]

Continue this cycle until you have the final answer.

When you have the final answer, respond with:
**Thought:** I now have all the information needed.
**Final Answer:** [Your complete response]
"""

# The model naturally follows this pattern when given ReAct instructions
```

### Plan-and-Execute Pattern

For complex tasks, have the agent plan first:

```python
planning_prompt = """
You are a planning agent. Given a goal, create a detailed plan.

Goal: {goal}

Create a numbered list of steps to achieve this goal.
For each step, indicate:
1. What action to take
2. What tool to use (if any)
3. What information is needed
4. How to verify success

Be thorough but efficient. Avoid unnecessary steps.
"""

execution_prompt = """
You are an execution agent. Follow this plan step by step:

{plan}

Current Step: {current_step}

Execute this step using the available tools.
Report the result and whether the step succeeded.
"""
```

---

## Tool Definition Best Practices

### Clear, Actionable Descriptions

```python
# ❌ Bad: Vague description
{
    "name": "search",
    "description": "Search for things"
}

# ✅ Good: Specific, actionable description
{
    "name": "search_products",
    "description": "Search the e-commerce product catalog. Returns matching products with name, price, and availability. Use for finding specific items or browsing categories. Returns up to 20 results by default."
}
```

### Parameter Documentation

```python
# ✅ Good: Well-documented parameters
{
    "name": "book_appointment",
    "description": "Book an appointment at available time slots.",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. Must be a weekday within the next 30 days."
            },
            "time": {
                "type": "string",
                "description": "Time in HH:MM format (24-hour). Available slots: 09:00-17:00."
            },
            "duration_minutes": {
                "type": "integer",
                "enum": [30, 60, 90],
                "description": "Appointment length. Default is 60 minutes."
            },
            "notes": {
                "type": "string",
                "description": "Optional notes for the appointment (max 500 chars)."
            }
        },
        "required": ["date", "time"],
        "additionalProperties": False
    }
}
```

### Error Handling Guidance

```python
{
    "name": "transfer_funds",
    "description": """
    Transfer money between accounts.
    
    Returns: {"success": true, "transaction_id": "..."} on success
    
    Possible errors:
    - "insufficient_funds": Source account lacks funds
    - "invalid_account": Account number not found
    - "daily_limit_exceeded": Transfer exceeds $10,000 daily limit
    - "account_frozen": Account is temporarily frozen
    
    On error, explain the issue to the user and suggest alternatives.
    """,
    "parameters": { ... }
}
```

---

## Goal-Oriented Prompting

### Define the End Goal, Not the Steps

```python
# ❌ Bad: Prescribing steps
prompt = """
1. First, search for competitor websites
2. Then, visit each website
3. Find the pricing page
4. Extract the prices
5. Create a comparison table
"""

# ✅ Good: Define goal, let agent plan
prompt = """
Goal: Create a pricing comparison table for our top 3 competitors.

Success Criteria:
- Include all pricing tiers for each competitor
- Note any hidden fees or requirements
- Format as a markdown table
- Include date of data collection

You have access to web search and page retrieval tools.
Plan your approach and execute it.
"""
```

### Trust the Model to Plan

```python
system_prompt = """
You are an autonomous research assistant.

When given a research goal:
1. Analyze what information is needed
2. Plan your research approach
3. Execute using available tools
4. Synthesize findings into a coherent response

You may need multiple tool calls to gather comprehensive information.
Adapt your approach based on what you learn.
"""
```

---

## Lesson Structure

This lesson is organized into the following sub-lessons:

| File | Topic |
|------|-------|
| [01-tool-definition-prompts.md](./01-tool-definition-prompts.md) | Writing effective tool descriptions and schemas |
| [02-multi-turn-agent-loops.md](./02-multi-turn-agent-loops.md) | Managing context across turns |
| [03-mcp-agentic-patterns.md](./03-mcp-agentic-patterns.md) | MCP tool discovery and integration |
| [04-computer-use-prompting.md](./04-computer-use-prompting.md) | Screen control and automation |
| [05-agentic-safety-patterns.md](./05-agentic-safety-patterns.md) | Safety, confirmations, and guardrails |

---

## Summary

✅ **Agents are loops, not single requests:** Plan → Execute → Observe → Adapt
✅ **Define goals, not steps:** Trust the model to plan its approach
✅ **Tool definitions matter:** Clear descriptions, good parameter docs, error guidance
✅ **Multiple patterns exist:** ReAct, Plan-and-Execute, recursive agents
✅ **Safety is essential:** Build in confirmations, limits, and human oversight

**Next:** [Tool Definition Prompts](./01-tool-definition-prompts.md)

---

## Further Reading

- [OpenAI Agents Guide](https://platform.openai.com/docs/guides/agents)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

<!-- 
Sources Consulted:
- OpenAI Agents Guide: Agent loop patterns, tool calling flow
- OpenAI Function Calling: Tool schema, strict mode, best practices
- Anthropic Tool Use: Tool definitions, client vs server tools
-->
