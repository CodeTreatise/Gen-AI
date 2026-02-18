---
title: "Agent Class Fundamentals"
---

# Agent Class Fundamentals

## Introduction

The `Agent` class is the central building block of the OpenAI Agents SDK. Every AI-powered workflow starts with defining an agent — an LLM equipped with instructions, tools, and behavior constraints. Whether you're building a simple chatbot or a complex multi-agent system, understanding the `Agent` class is essential.

In this lesson, we explore every aspect of creating and configuring agents, from basic properties to advanced features like dynamic instructions, output types, lifecycle hooks, and agent cloning.

### What we'll cover

- Creating agents with `name`, `instructions`, and `model`
- Configuring output types with Pydantic models
- Writing dynamic instructions that adapt at runtime
- Controlling tool usage behavior
- Using lifecycle hooks for event monitoring
- Cloning agents for variants

### Prerequisites

- Python fundamentals (classes, decorators, async/await)
- Pydantic basics (models, `Field`)
- Completed the [OpenAI Agents SDK overview](./00-openai-agents-sdk.md)

---

## Creating your first agent

An agent needs just one required parameter — `name`. Everything else has sensible defaults:

```python
from agents import Agent

# Minimal agent — uses default model and no special instructions
agent = Agent(name="Assistant")
```

In practice, you'll almost always set `instructions` to tell the agent how to behave:

```python
from agents import Agent

agent = Agent(
    name="Customer Support",
    instructions="You are a friendly customer support agent. Help users with their questions about our products. Be concise and professional.",
)
```

> **🔑 Key concept:** `instructions` is the system prompt. It shapes the agent's personality, knowledge boundaries, and response style. Write it like you're briefing a human employee.

---

## Core agent properties

The `Agent` class accepts several configuration parameters. Here are the most important ones:

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | **Required.** Identifies the agent in logs, traces, and handoffs |
| `instructions` | `str \| Callable` | System prompt — static string or dynamic function |
| `model` | `str \| Model` | LLM to use (default: SDK default, typically `gpt-4o`) |
| `tools` | `list[Tool]` | Functions and services the agent can call |
| `handoffs` | `list[Agent \| Handoff]` | Other agents this agent can delegate to |
| `output_type` | `type[BaseModel]` | Structured output format (Pydantic model) |
| `input_guardrails` | `list[InputGuardrail]` | Validators that check user input |
| `output_guardrails` | `list[OutputGuardrail]` | Validators that check agent output |
| `model_settings` | `ModelSettings` | Temperature, tool_choice, etc. |

### Setting the model

```python
from agents import Agent

# Use a specific model
agent = Agent(
    name="Writer",
    instructions="Write creative stories.",
    model="gpt-4o",
)

# Use a mini model for cost efficiency
fast_agent = Agent(
    name="Classifier",
    instructions="Classify the user's intent.",
    model="gpt-4o-mini",
)
```

### Configuring model settings

Fine-tune LLM behavior with `ModelSettings`:

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Creative Writer",
    instructions="Write creative fiction.",
    model_settings=ModelSettings(
        temperature=0.9,
        top_p=0.95,
        tool_choice="auto",
    ),
)
```

---

## Structured output types

By default, agents return free-form text. With `output_type`, you can force the agent to return structured data using a Pydantic model:

```python
from pydantic import BaseModel, Field
from agents import Agent, Runner

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1 to 10")
    summary: str = Field(description="Brief review summary")
    recommended: bool = Field(description="Whether you recommend this movie")

agent = Agent(
    name="Movie Critic",
    instructions="You are a movie critic. Analyze movies and provide structured reviews.",
    output_type=MovieReview,
)

result = Runner.run_sync(agent, "Review the movie Inception")
review = result.final_output  # This is a MovieReview instance
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Recommended: {review.recommended}")
```

**Output:**
```
Title: Inception
Rating: 9/10
Recommended: True
```

> **🤖 AI Context:** Under the hood, `output_type` uses OpenAI's Structured Outputs feature. The Pydantic model's JSON schema is passed to the API, ensuring the LLM always returns valid, parseable data — no regex extraction needed.

---

## Dynamic instructions

Static instructions work for many cases, but sometimes you need instructions that adapt based on context — the current user, time of day, or application state:

```python
import datetime
from agents import Agent, RunContextWrapper

def get_instructions(
    context: RunContextWrapper[dict], agent: Agent
) -> str:
    user_name = context.context.get("user_name", "there")
    current_hour = datetime.datetime.now().hour
    
    if current_hour < 12:
        greeting = "Good morning"
    elif current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    
    return f"""{greeting}, {user_name}!
You are a helpful assistant. Be concise and friendly.
The current time is {datetime.datetime.now().strftime('%H:%M')}."""

agent = Agent(
    name="Greeter",
    instructions=get_instructions,  # Pass the function, not its result
)
```

The function signature must accept `(context: RunContextWrapper, agent: Agent)` and return a `str`. It can also be `async`:

```python
async def get_instructions(context: RunContextWrapper, agent: Agent) -> str:
    # Could fetch data from a database here
    return "You are a helpful assistant."
```

> **💡 Tip:** Dynamic instructions are called on every turn, so keep them fast. Avoid expensive database queries — use the context object to cache data instead.

---

## Prompt templates

For teams managing prompts externally, the SDK supports static prompt templates with versioning:

```python
from agents import Agent

agent = Agent(
    name="Support Agent",
    prompt={
        "id": "pmpt_support_v2",
        "version": "1",
        "variables": {
            "company_name": "Acme Corp",
            "tone": "professional",
        },
    },
)
```

> **Note:** When you use `prompt`, the agent ignores `instructions` — they're mutually exclusive.

---

## Controlling tool use behavior

When an agent has tools, you can control what happens after a tool is called:

```python
from agents import Agent, ModelSettings

# Default: LLM runs again after tool call to generate final response
agent_default = Agent(
    name="Default",
    tools=[...],
    # tool_use_behavior="run_llm_again"  ← this is the default
)

# Stop after the first tool call and return its output directly
agent_stop = Agent(
    name="Quick",
    tools=[...],
    tool_use_behavior="stop_on_first_tool",
)
```

For more granular control, use `StopAtTools`:

```python
from agents import Agent, StopAtTools

agent = Agent(
    name="Router",
    tools=[search_tool, calculate_tool, classify_tool],
    tool_use_behavior=StopAtTools(
        stop_at_tool_names=["classify_tool"]
    ),
)
```

### The `reset_tool_choice` parameter

When `tool_choice` is set to `"required"`, the LLM must call a tool every turn — which can cause infinite loops. The `reset_tool_choice` parameter (default `True`) automatically resets `tool_choice` to `"auto"` after the first tool call:

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Tool User",
    model_settings=ModelSettings(tool_choice="required"),
    reset_tool_choice=True,  # Prevents infinite tool-calling loops
    tools=[...],
)
```

---

## Lifecycle hooks

Monitor agent events using `AgentHooks`. This is useful for logging, metrics, or debugging:

```python
from agents import Agent, AgentHooks, RunContextWrapper, Tool

class LoggingHooks(AgentHooks):
    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        print(f"🚀 Agent '{agent.name}' starting")
    
    async def on_end(self, context: RunContextWrapper, agent: Agent, output) -> None:
        print(f"✅ Agent '{agent.name}' finished")
    
    async def on_tool_start(
        self, context: RunContextWrapper, agent: Agent, tool: Tool
    ) -> None:
        print(f"🔧 Calling tool: {tool.name}")
    
    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        print(f"📦 Tool result: {result[:100]}...")

agent = Agent(
    name="Monitored Agent",
    instructions="You are helpful.",
    hooks=LoggingHooks(),
    tools=[...],
)
```

**Output (during execution):**
```
🚀 Agent 'Monitored Agent' starting
🔧 Calling tool: fetch_weather
📦 Tool result: The weather in San Francisco is sunny and 72°F...
✅ Agent 'Monitored Agent' finished
```

---

## Cloning agents

Need a variant of an existing agent? Use `clone()` instead of redefining everything:

```python
from agents import Agent

base_agent = Agent(
    name="Support Agent",
    instructions="You are a helpful support agent.",
    model="gpt-4o",
    tools=[search_tool, ticket_tool],
)

# Create a Spanish-speaking variant
spanish_agent = base_agent.clone(
    name="Support Agent (ES)",
    instructions="Eres un agente de soporte útil. Responde siempre en español.",
)

# Create a premium variant with a stronger model
premium_agent = base_agent.clone(
    name="Premium Support",
    model="gpt-4o",
    instructions="You are a premium support agent. Provide extra detailed help.",
)
```

> **💡 Tip:** `clone()` performs a shallow copy — the new agent shares the same tool instances. This is usually what you want, but be aware if your tools hold mutable state.

---

## Generic agents with typed context

Agents support Python generics to ensure type safety with your context object:

```python
from dataclasses import dataclass
from agents import Agent

@dataclass
class AppContext:
    user_id: str
    is_premium: bool
    db_connection: object

# Agent is typed with AppContext
agent = Agent[AppContext](
    name="Support",
    instructions="Help the user with their account.",
    tools=[...],
)
```

This typing ensures that all tool functions, hooks, and dynamic instructions receive properly typed context — catching errors at development time rather than runtime.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Write clear, specific `instructions` | Vague instructions lead to inconsistent behavior |
| Use `output_type` for data extraction | Eliminates brittle text parsing |
| Prefer dynamic instructions over stuffing context into input | Keeps system prompt clean and focused |
| Set `temperature=0` for deterministic tasks | Classification, extraction, routing benefit from consistency |
| Use `clone()` for agent variants | DRY principle — change once, update everywhere |
| Name agents descriptively | Helps with debugging traces and handoff readability |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Setting `instructions` and `prompt` together | Use one or the other — `prompt` overrides `instructions` |
| Forgetting `reset_tool_choice` with `tool_choice="required"` | Keep default `True` to prevent infinite loops |
| Making dynamic instructions slow (DB calls each turn) | Cache data in context object, not in instructions function |
| Using `output_type` without descriptions on fields | Add `Field(description=...)` so the LLM knows what each field means |
| Defining too many tools on one agent | Split into specialized agents with handoffs for clarity |

---

## Hands-on exercise

### Your task

Build a **Job Application Analyzer** agent that accepts a job description (free text) and returns a structured analysis.

### Requirements

1. Create a Pydantic model `JobAnalysis` with fields: `title` (str), `seniority_level` (str), `required_skills` (list of str), `salary_mentioned` (bool), `remote_friendly` (bool)
2. Create an agent with `output_type=JobAnalysis`
3. Use dynamic instructions that include the current date
4. Run the agent with a sample job posting and print the structured result

### Expected result

The agent should return a `JobAnalysis` object with extracted fields from the job posting.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `from pydantic import BaseModel, Field` for the output model
- Dynamic instructions take `(context, agent)` and return a string
- Use `Runner.run_sync()` for synchronous execution
- Access the structured output via `result.final_output`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import datetime
from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper

class JobAnalysis(BaseModel):
    title: str = Field(description="The job title")
    seniority_level: str = Field(description="Junior, Mid, Senior, Lead, etc.")
    required_skills: list[str] = Field(description="List of required technical skills")
    salary_mentioned: bool = Field(description="Whether salary info is included")
    remote_friendly: bool = Field(description="Whether remote work is mentioned")

def dynamic_instructions(context: RunContextWrapper, agent: Agent) -> str:
    today = datetime.date.today().isoformat()
    return f"""You are a job posting analyst. Today's date is {today}.
Analyze job postings and extract structured information.
Be precise with skill names and seniority assessment."""

agent = Agent(
    name="Job Analyzer",
    instructions=dynamic_instructions,
    output_type=JobAnalysis,
    model="gpt-4o-mini",
)

job_posting = """
Senior Full-Stack Engineer at TechCorp
We're looking for an experienced engineer to join our remote-first team.
Requirements: 5+ years with React, Node.js, TypeScript, PostgreSQL.
Nice to have: AWS, Docker, GraphQL.
Competitive salary: $150k-$200k base.
"""

result = Runner.run_sync(agent, job_posting)
analysis = result.final_output
print(f"Title: {analysis.title}")
print(f"Level: {analysis.seniority_level}")
print(f"Skills: {', '.join(analysis.required_skills)}")
print(f"Salary mentioned: {analysis.salary_mentioned}")
print(f"Remote: {analysis.remote_friendly}")
```

**Output:**
```
Title: Senior Full-Stack Engineer
Level: Senior
Skills: React, Node.js, TypeScript, PostgreSQL, AWS, Docker, GraphQL
Salary mentioned: True
Remote: True
```

</details>

### Bonus challenges

- [ ] Add a `clone()` variant that analyzes job postings in Spanish
- [ ] Add lifecycle hooks that log when the agent starts and finishes
- [ ] Experiment with `model_settings=ModelSettings(temperature=0)` for consistency

---

## Summary

✅ The `Agent` class requires only `name` but accepts rich configuration via `instructions`, `model`, `tools`, and more

✅ `output_type` enforces structured responses using Pydantic models — no manual parsing needed

✅ Dynamic instructions adapt at runtime using functions that receive context and agent references

✅ `tool_use_behavior` and `reset_tool_choice` give fine-grained control over the agent loop

✅ `clone()` creates agent variants efficiently, and generics (`Agent[T]`) provide type-safe context

**Next:** [Runner Execution Model](./02-runner-execution-model.md)

---

## Further reading

- [OpenAI Agents SDK — Agents docs](https://openai.github.io/openai-agents-python/agents/) — Full Agent class reference
- [Pydantic Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — How output_type works under the hood
- [Agent API Reference](https://openai.github.io/openai-agents-python/ref/agent/) — Complete parameter list

---

*[Back to OpenAI Agents SDK Overview](./00-openai-agents-sdk.md)*

<!-- 
Sources Consulted:
- OpenAI Agents SDK Agents page: https://openai.github.io/openai-agents-python/agents/
- OpenAI Agents SDK homepage: https://openai.github.io/openai-agents-python/
- Agent API reference: https://openai.github.io/openai-agents-python/ref/agent/
-->
