---
title: "Python-Native Frameworks"
---

# Python-Native Frameworks

## Introduction

Two frameworks stand out for taking a **Python-first** approach to agent development: **Pydantic AI** brings type safety, dependency injection, and structured validation to agents, while **Smolagents** by Hugging Face offers a minimal, code-first design where agents write Python instead of JSON tool calls.

Both frameworks prioritize developer experience and Python idioms over complex abstractions. If you value type hints, IDE autocomplete, and clean Pythonic APIs, these are your tools.

### What we'll cover

- Pydantic AI's type-safe agent architecture
- Dependency injection and structured output
- Smolagents' code agent pattern
- When to choose each framework

### Prerequisites

- Python type hints and Pydantic basics
- Agent tool-calling concepts (Lesson 05)
- Async/await proficiency

---

## Pydantic AI

Pydantic AI brings the "FastAPI feeling" to agent development. Built by the team behind Pydantic (the validation library used by virtually every Python AI framework), it provides **fully type-safe** agents with dependency injection, structured output, and observability through Pydantic Logfire.

### Creating an agent

```python
# pip install pydantic-ai
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Be concise. Reply with one sentence.",
)

result = agent.run_sync("What is Python?")
print(result.output)
```

**Output:**
```
Python is a high-level, interpreted programming language known for its readability and versatility.
```

### Structured output

Pydantic AI's defining feature is type-safe structured output. The agent's response is validated against a Pydantic model automatically.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class MovieReview(BaseModel):
    title: str = Field(description="Name of the movie")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    summary: str = Field(description="One-sentence summary")
    recommended: bool = Field(description="Whether to recommend this movie")

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=MovieReview,
    instructions="You are a film critic. Analyze the given movie.",
)

result = agent.run_sync("Inception (2010)")
review: MovieReview = result.output  # Fully typed!
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
print(f"Recommended: {review.recommended}")
```

**Output:**
```
Title: Inception
Rating: 9.2/10
Summary: A mind-bending thriller where dream infiltrators navigate layers of consciousness.
Recommended: True
```

> **🔑 Key concept:** The output is a validated Pydantic model, not a raw string. Your IDE provides autocomplete for `review.title`, `review.rating`, etc.

### Dependency injection

Pydantic AI uses a dependency injection pattern inspired by FastAPI. Define a `deps_type` and tools receive it via `RunContext`:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    user_id: int
    db_connection: str  # simplified — use a real DB client

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=AppDeps,
    instructions="Help the user with their account.",
)

@agent.tool
async def get_balance(ctx: RunContext[AppDeps]) -> str:
    """Get the current account balance."""
    # ctx.deps gives you typed access to dependencies
    return f"User {ctx.deps.user_id} has a balance of $1,234.56"

@agent.tool
async def get_transactions(ctx: RunContext[AppDeps], limit: int = 5) -> str:
    """Get recent transactions."""
    return f"Last {limit} transactions for user {ctx.deps.user_id}: ..."

# Run with injected dependencies
result = agent.run_sync(
    "What is my balance?",
    deps=AppDeps(user_id=42, db_connection="postgresql://..."),
)
print(result.output)
```

**Output:**
```
Your current account balance is $1,234.56.
```

### Dynamic instructions

Instructions can be dynamic functions that receive context:

```python
from pydantic_ai import Agent, RunContext

agent = Agent("openai:gpt-4o-mini", deps_type=AppDeps)

@agent.instructions
async def add_context(ctx: RunContext[AppDeps]) -> str:
    return f"You are helping user #{ctx.deps.user_id}. Be polite and professional."
```

### Observability with Logfire

```python
import logfire
from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent("openai:gpt-4o-mini", instructions="Be helpful.")
result = agent.run_sync("Hello!")
# All agent calls, tool invocations, and LLM interactions are traced in Logfire
```

---

## Smolagents (Hugging Face)

Smolagents takes a radically different approach: instead of generating JSON tool calls, agents write **Python code** to solve tasks. This enables natural composition — function nesting, loops, conditionals — things that JSON-based tool calling cannot express.

### Core concept: code agents

```python
# pip install 'smolagents[toolkit]'
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()  # Uses HuggingFace Inference API
agent = CodeAgent(tools=[], model=model)

result = agent.run("Calculate the sum of squares from 1 to 10")
print(result)
```

**Output:**
```
385
```

> **🤖 AI Context:** The agent wrote `sum(i**2 for i in range(1, 11))` internally — real Python code, not a tool call.

### Adding tools

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

result = agent.run("Search for the latest Python version and tell me what's new")
print(result)
```

### Tool-calling agent (alternative mode)

For scenarios where code execution isn't desired, Smolagents also supports traditional tool calling:

```python
from smolagents import ToolCallingAgent, InferenceClientModel

model = InferenceClientModel()
agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

result = agent.run("What is the population of Tokyo?")
print(result)
```

### Custom tools

```python
from smolagents import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to check weather for.
    """
    # In production, call a real weather API
    return f"The weather in {city} is 18°C with light clouds."

agent = CodeAgent(
    tools=[get_weather],
    model=InferenceClientModel(),
)

result = agent.run("What's the weather in Paris and London?")
print(result)
```

**Output:**
```
The weather in Paris is 18°C with light clouds.
The weather in London is 18°C with light clouds.
```

> **Note:** Because the agent writes code, it can call `get_weather("Paris")` and `get_weather("London")` in a single step — no multi-turn tool calling needed.

### Using different models

```python
from smolagents import CodeAgent, LiteLLMModel, InferenceClientModel

# HuggingFace Inference API
model = InferenceClientModel(model_id="meta-llama/Llama-3-70b-Instruct")

# OpenAI / Anthropic via LiteLLM
model = LiteLLMModel(model_id="gpt-4o")
model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514")

# Local models via Transformers
from smolagents import TransformersModel
model = TransformersModel(model_id="meta-llama/Llama-3-8b-Instruct")
```

### MCP integration

Smolagents can load tools from any MCP server:

```python
from smolagents import ToolCollection, CodeAgent, InferenceClientModel

# Load tools from an MCP server
tools = ToolCollection.from_mcp("npx @modelcontextprotocol/server-filesystem /tmp")

agent = CodeAgent(
    tools=tools,
    model=InferenceClientModel(),
)
```

### Secure code execution

Since code agents execute Python, sandboxing is critical for production:

```python
from smolagents import CodeAgent, InferenceClientModel

# Execute in a Docker sandbox
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    executor_type="e2b",       # Options: "e2b", "docker", "modal"
    executor_kwargs={"api_key": "your-e2b-key"},
)
```

---

## Pydantic AI vs Smolagents comparison

| Feature | Pydantic AI | Smolagents |
|---------|------------|------------|
| **Agent pattern** | Tool-calling (JSON) | Code generation (Python) |
| **Type safety** | Full — Pydantic models | Minimal |
| **Structured output** | Built-in validation | Manual parsing |
| **Dependencies** | Injection via `RunContext` | Not built-in |
| **Observability** | Pydantic Logfire (OTel) | Basic logging |
| **Model support** | All major providers | HF Inference, LiteLLM, local |
| **MCP support** | Yes (client + server) | Yes (via `ToolCollection`) |
| **Best for** | Production apps with validation | Research, code generation |
| **Install** | `pip install pydantic-ai` | `pip install smolagents` |
| **GitHub stars** | 15k+ | 15k+ |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use Pydantic AI for validated output | Structured output catches type errors at generation time |
| Use `deps_type` for database/API clients | Dependency injection makes agents testable |
| Use Smolagents `CodeAgent` for computation | Code composition beats multi-turn tool calling for math/logic |
| Always sandbox code agents | `CodeAgent` executes arbitrary Python — never trust LLM output |
| Use `@agent.tool` decorator for type inference | Pydantic AI reads type hints from the function signature |
| Use `@tool` decorator in Smolagents | Docstrings become tool descriptions for the LLM |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Returning untyped dicts from Pydantic AI agents | Define an `output_type` Pydantic model for structured responses |
| Running CodeAgent without sandboxing | Use `executor_type="e2b"` or `"docker"` in production |
| Ignoring `RunContext` for dependency access | Inject DB clients, API keys via `deps_type` — don't use globals |
| Using CodeAgent for simple Q&A | Use `ToolCallingAgent` when code generation isn't needed |
| Not testing with `TestModel` | Pydantic AI provides `TestModel` for deterministic testing |
| Mixing sync and async in Pydantic AI | Use `agent.run()` (async) or `agent.run_sync()` consistently |

---

## Hands-on exercise

### Your task

Build a type-safe customer support agent using Pydantic AI with structured output and dependency injection.

### Requirements

1. Define a `SupportResponse` Pydantic model with fields: `answer`, `category`, `priority` (1-5), and `needs_escalation` (bool)
2. Create an agent with `deps_type` that includes a `customer_id`
3. Add a tool that looks up customer information using the injected dependencies
4. Run the agent and verify the output is properly typed

### Expected result

The agent returns a validated `SupportResponse` object with correct types for all fields.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Field(description=...)` to guide the LLM's structured output
- `ge=1, le=5` on `priority` constrains the range
- `RunContext[YourDepsType]` gives typed access to dependencies in tools
- Test with `agent.run_sync("message", deps=YourDeps(...))`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

class SupportResponse(BaseModel):
    answer: str = Field(description="Response to the customer")
    category: str = Field(description="billing, technical, account, or general")
    priority: int = Field(description="Priority level", ge=1, le=5)
    needs_escalation: bool = Field(description="Whether to escalate to a human")

@dataclass
class SupportDeps:
    customer_id: int

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=SupportResponse,
    deps_type=SupportDeps,
    instructions="You are a support agent. Analyze queries and provide structured responses.",
)

@agent.tool
async def get_customer_info(ctx: RunContext[SupportDeps]) -> str:
    """Look up customer information."""
    return f"Customer #{ctx.deps.customer_id}: Premium plan, active since 2023"

result = agent.run_sync("I can't access my account", deps=SupportDeps(customer_id=42))
response: SupportResponse = result.output
print(f"Category: {response.category}, Priority: {response.priority}")
print(f"Escalate: {response.needs_escalation}")
```

</details>

### Bonus challenges

- [ ] Add a Smolagents `CodeAgent` that performs data analysis on CSV files
- [ ] Implement Pydantic AI's `@agent.instructions` for dynamic context
- [ ] Compare response quality between `CodeAgent` and `ToolCallingAgent`

---

## Summary

✅ **Pydantic AI** provides type-safe agents with validated structured output and dependency injection  
✅ **Smolagents** enables code agents that write Python instead of JSON tool calls  
✅ Pydantic AI excels at **production apps** needing validation and observability  
✅ Smolagents excels at **computation and composition** through natural code generation  
✅ Both support MCP integration and multiple model providers  

**Previous:** [Established Frameworks](./01-established-frameworks.md)  
**Next:** [Emerging Frameworks](./03-emerging-frameworks.md)  
**Back to:** [Additional Frameworks & Tools](./00-additional-frameworks-tools.md)

---

## Further Reading

- [Pydantic AI Documentation](https://ai.pydantic.dev/) — Full framework reference
- [Pydantic AI Examples](https://ai.pydantic.dev/examples/setup/) — Bank support, RAG, streaming
- [Smolagents Documentation](https://huggingface.co/docs/smolagents/) — Guided tour and tutorials
- [Smolagents GitHub](https://github.com/huggingface/smolagents) — Source code and examples
- [Pydantic Logfire](https://pydantic.dev/logfire) — Observability platform

<!--
Sources Consulted:
- Pydantic AI: https://ai.pydantic.dev/
- Pydantic AI examples: https://ai.pydantic.dev/examples/setup/
- Smolagents: https://huggingface.co/docs/smolagents/en/index
- Smolagents guided tour: https://huggingface.co/docs/smolagents/en/guided_tour
-->
