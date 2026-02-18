---
title: "Tool Implementation Patterns"
---

# Tool Implementation Patterns

## Introduction

Tools give agents the ability to **take actions** — fetch data from APIs, run calculations, search the web, execute code, and interact with external systems. Without tools, an agent is limited to what it already knows. With tools, it becomes a capable problem-solver that can gather information and act on the real world.

The OpenAI Agents SDK supports five categories of tools, each with different trade-offs. In this lesson, we cover all of them with practical patterns.

### What we'll cover

- Function tools with `@function_tool`
- Hosted OpenAI tools (web search, file search, code interpreter)
- Local runtime tools (computer, shell, apply patch)
- Agents as tools (orchestrator pattern)
- Custom `FunctionTool` objects
- Error handling in tools

### Prerequisites

- [Agent Class Fundamentals](./01-agent-class-fundamentals.md)
- [Runner Execution Model](./02-runner-execution-model.md)
- Python type hints and docstrings

---

## Function tools with `@function_tool`

The most common pattern — wrap any Python function as an agent tool:

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The city name to check weather for.
    """
    # In production, this would call a real weather API
    return f"The weather in {city} is sunny, 72°F."

agent = Agent(
    name="Weather Assistant",
    instructions="Help users check the weather.",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

**Output:**
```
The weather in Tokyo is sunny, 72°F.
```

> **🔑 Key concept:** The SDK automatically extracts the tool's name, description, and parameter schema from your function. The function name becomes the tool name, the docstring becomes the description, and type annotations define the schema.

### Automatic schema extraction

The SDK uses `inspect` to parse your function signature and `griffe` to parse docstrings. Supported docstring formats are Google, Sphinx, and NumPy:

```python
from typing_extensions import TypedDict

class Location(TypedDict):
    lat: float
    long: float

@function_tool
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    return "sunny"
```

### Overriding the tool name

```python
@function_tool(name_override="search_database")
def query_db(query: str) -> str:
    """Search our product database."""
    return f"Results for: {query}"
```

### Async function tools

Tools can be async — useful for I/O-bound operations:

```python
import httpx

@function_tool
async def fetch_stock_price(symbol: str) -> str:
    """Fetch the current stock price.
    
    Args:
        symbol: The stock ticker symbol (e.g., AAPL).
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/stocks/{symbol}")
        data = response.json()
        return f"{symbol}: ${data['price']}"
```

---

## Hosted OpenAI tools

OpenAI provides built-in tools that run on their servers — no implementation needed:

```python
from agents import Agent, WebSearchTool, FileSearchTool, CodeInterpreterTool

agent = Agent(
    name="Research Assistant",
    instructions="Help users research topics using web search and analysis.",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["vs_abc123"],
        ),
        CodeInterpreterTool(),
    ],
)
```

| Tool | Purpose | Runs on |
|------|---------|---------|
| `WebSearchTool()` | Search the web for current information | OpenAI servers |
| `FileSearchTool()` | Search your OpenAI Vector Stores | OpenAI servers |
| `CodeInterpreterTool()` | Execute Python code in a sandbox | OpenAI servers |
| `ImageGenerationTool()` | Generate images from text prompts | OpenAI servers |
| `HostedMCPTool()` | Connect to a remote MCP server | OpenAI servers |

> **Note:** Hosted tools only work with the `OpenAIResponsesModel` (the default). If you're using LiteLLM or chat completions, use function tools instead.

---

## Local runtime tools

These tools run in your environment and require you to implement their interfaces:

```python
from agents import Agent, ShellTool

async def run_shell(request):
    """Execute a shell command and return output."""
    import asyncio
    proc = await asyncio.create_subprocess_shell(
        request,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode() or stderr.decode()

agent = Agent(
    name="DevOps Agent",
    instructions="Help with system administration tasks.",
    tools=[
        ShellTool(executor=run_shell),
    ],
)
```

| Tool | Interface to implement | Use case |
|------|----------------------|----------|
| `ComputerTool` | `Computer` / `AsyncComputer` | GUI/browser automation |
| `ShellTool` | Shell executor function | Run shell commands |
| `LocalShellTool` | Built-in local shell | Quick local commands |
| `ApplyPatchTool` | `ApplyPatchEditor` | Apply file diffs |

> **Warning:** Local runtime tools execute code in your environment. Always implement proper sandboxing and validation in production.

---

## Agents as tools (orchestrator pattern)

Instead of handing off to another agent (which transfers control), you can use an agent **as a tool** — the orchestrator keeps control and uses the sub-agent's response:

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish Translator",
    instructions="Translate the user's message to Spanish.",
)

french_agent = Agent(
    name="French Translator",
    instructions="Translate the user's message to French.",
)

orchestrator = Agent(
    name="Translation Hub",
    instructions=(
        "You are a translation agent. Use the tools to translate "
        "the user's message into the requested languages."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate text to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate text to French",
        ),
    ],
)

result = Runner.run_sync(
    orchestrator,
    "Translate 'Hello, how are you?' to both Spanish and French."
)
print(result.final_output)
```

**Output:**
```
Here are the translations:
- Spanish: ¡Hola, ¿cómo estás?
- French: Bonjour, comment allez-vous ?
```

### Structured input for tool-agents

By default, `as_tool()` expects a single string input. For structured input, pass a Pydantic model:

```python
from pydantic import BaseModel, Field

class TranslationInput(BaseModel):
    text: str = Field(description="Text to translate.")
    source: str = Field(description="Source language.")
    target: str = Field(description="Target language.")

translator_tool = translator_agent.as_tool(
    tool_name="translate_text",
    tool_description="Translate text between languages.",
    parameters=TranslationInput,
    include_input_schema=True,
)
```

### Conditional tool enabling

Dynamically enable or disable tools at runtime:

```python
from agents import Agent, RunContextWrapper, AgentBase

def is_premium_user(ctx: RunContextWrapper, agent: AgentBase) -> bool:
    """Only enable advanced tools for premium users."""
    return ctx.context.get("is_premium", False)

agent = Agent(
    name="Assistant",
    tools=[
        basic_agent.as_tool(
            tool_name="basic_search",
            tool_description="Basic search",
            is_enabled=True,  # Always available
        ),
        advanced_agent.as_tool(
            tool_name="deep_analysis",
            tool_description="Deep analysis with premium models",
            is_enabled=is_premium_user,  # Runtime check
        ),
    ],
)
```

---

## Custom FunctionTool objects

When you need full control, create a `FunctionTool` directly:

```python
from pydantic import BaseModel
from agents import FunctionTool, RunContextWrapper
from typing import Any

class UserQuery(BaseModel):
    username: str
    include_history: bool = False

async def search_user(ctx: RunContextWrapper[Any], args: str) -> str:
    """Search for a user in the database."""
    parsed = UserQuery.model_validate_json(args)
    # ... database query logic ...
    return f"Found user: {parsed.username}"

user_search_tool = FunctionTool(
    name="search_user",
    description="Search for a user in the database by username",
    params_json_schema=UserQuery.model_json_schema(),
    on_invoke_tool=search_user,
)
```

---

## Returning images and files from tools

Tools can return more than text:

```python
from agents import function_tool, ToolOutputImage, ToolOutputFileContent
import base64

@function_tool
def generate_chart(data_type: str) -> list:
    """Generate a chart visualization.
    
    Args:
        data_type: The type of chart to generate.
    """
    # Generate chart image bytes
    chart_bytes = create_chart(data_type)  # Your chart library
    
    return [
        ToolOutputImage(
            image_data=base64.b64encode(chart_bytes).decode(),
            media_type="image/png",
        ),
        # Can also include text explanation
        "Chart generated successfully.",
    ]
```

---

## Error handling in tools

### Default error handling

By default, if a tool throws an exception, the SDK sends a generic error message to the LLM:

```python
@function_tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    return str(a / b)  # ZeroDivisionError if b=0 — SDK handles it
```

### Custom error function

Provide a friendlier error message:

```python
from agents import function_tool, RunContextWrapper
from typing import Any

def friendly_error(context: RunContextWrapper[Any], error: Exception) -> str:
    """Provide a user-friendly error message."""
    print(f"Tool error: {error}")
    return "This operation failed. Please try different parameters."

@function_tool(failure_error_function=friendly_error)
def risky_operation(query: str) -> str:
    """Perform a risky database operation."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    return f"Result: {query}"
```

### Re-raising errors

Pass `None` to let errors propagate to your code:

```python
@function_tool(failure_error_function=None)
def critical_operation(data: str) -> str:
    """An operation where failures should stop execution."""
    # If this throws, the exception propagates to your try/except
    return process(data)
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Write clear docstrings with `Args:` | The LLM uses these to understand when and how to call the tool |
| Use type hints on all parameters | Enables automatic JSON schema generation |
| Return strings from tools | The LLM processes text — return human-readable results |
| Keep tools focused (single responsibility) | The LLM chooses better with clear, distinct tools |
| Use `is_enabled` for feature gating | Prevents unauthorized access to premium tools |
| Prefer `@function_tool` over `FunctionTool` | Less boilerplate, automatic schema extraction |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No docstring on `@function_tool` | Always add a description — the LLM needs it to decide when to call the tool |
| Returning complex objects | Return strings or use `ToolOutputImage` / `ToolOutputFileContent` |
| Blocking I/O in sync tools | Use `async` tools for network calls and file I/O |
| Too many tools on one agent | Limit to 5-10 tools; split complex agents into multi-agent systems |
| Missing `Args:` in docstrings | The SDK extracts parameter descriptions from docstrings |
| Not handling tool errors | Use `failure_error_function` or wrap risky code in try/except |

---

## Hands-on exercise

### Your task

Build a **unit converter agent** with three function tools: temperature, distance, and weight conversion.

### Requirements

1. Create three `@function_tool` functions: `convert_temperature`, `convert_distance`, `convert_weight`
2. Each tool should accept `value` (float), `from_unit` (str), `to_unit` (str) and return the converted value
3. Add proper docstrings with `Args:` descriptions
4. Create an agent with all three tools
5. Test with: "Convert 100°F to Celsius, 5 miles to kilometers, and 10 pounds to kilograms"

### Expected result

The agent should call all three tools and provide a combined answer.

<details>
<summary>💡 Hints (click to expand)</summary>

- Temperature: `(°F - 32) × 5/9 = °C`
- Distance: `1 mile = 1.60934 km`
- Weight: `1 pound = 0.453592 kg`
- Return formatted strings like `"100°F = 37.78°C"`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from agents import Agent, Runner, function_tool

@function_tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between units.
    
    Args:
        value: The temperature value to convert.
        from_unit: Source unit (celsius, fahrenheit, kelvin).
        to_unit: Target unit (celsius, fahrenheit, kelvin).
    """
    # Normalize to Celsius first
    if from_unit.lower().startswith("f"):
        celsius = (value - 32) * 5 / 9
    elif from_unit.lower().startswith("k"):
        celsius = value - 273.15
    else:
        celsius = value
    
    # Convert from Celsius to target
    if to_unit.lower().startswith("f"):
        result = celsius * 9 / 5 + 32
    elif to_unit.lower().startswith("k"):
        result = celsius + 273.15
    else:
        result = celsius
    
    return f"{value} {from_unit} = {result:.2f} {to_unit}"

@function_tool
def convert_distance(value: float, from_unit: str, to_unit: str) -> str:
    """Convert distance between units.
    
    Args:
        value: The distance value to convert.
        from_unit: Source unit (miles, kilometers, meters, feet).
        to_unit: Target unit (miles, kilometers, meters, feet).
    """
    # Convert to meters first
    to_meters = {"miles": 1609.34, "kilometers": 1000, "meters": 1, "feet": 0.3048}
    meters = value * to_meters.get(from_unit.lower(), 1)
    result = meters / to_meters.get(to_unit.lower(), 1)
    return f"{value} {from_unit} = {result:.2f} {to_unit}"

@function_tool
def convert_weight(value: float, from_unit: str, to_unit: str) -> str:
    """Convert weight between units.
    
    Args:
        value: The weight value to convert.
        from_unit: Source unit (pounds, kilograms, ounces, grams).
        to_unit: Target unit (pounds, kilograms, ounces, grams).
    """
    to_grams = {"pounds": 453.592, "kilograms": 1000, "ounces": 28.3495, "grams": 1}
    grams = value * to_grams.get(from_unit.lower(), 1)
    result = grams / to_grams.get(to_unit.lower(), 1)
    return f"{value} {from_unit} = {result:.2f} {to_unit}"

agent = Agent(
    name="Unit Converter",
    instructions="You convert units. Use the appropriate tool for each conversion.",
    tools=[convert_temperature, convert_distance, convert_weight],
)

result = Runner.run_sync(
    agent,
    "Convert 100°F to Celsius, 5 miles to kilometers, and 10 pounds to kilograms"
)
print(result.final_output)
```

**Output:**
```
Here are the conversions:
- 100°F = 37.78°C
- 5 miles = 8.05 kilometers
- 10 pounds = 4.54 kilograms
```

</details>

### Bonus challenges

- [ ] Add a hosted `WebSearchTool()` for looking up real-time exchange rates
- [ ] Create a custom `FunctionTool` with a Pydantic input schema
- [ ] Implement `is_enabled` to conditionally show metric vs imperial tools

---

## Summary

✅ `@function_tool` automatically extracts name, description, and schema from Python functions

✅ Hosted tools (WebSearchTool, FileSearchTool, CodeInterpreterTool) run on OpenAI servers — zero implementation required

✅ Agents as tools (`agent.as_tool()`) enable the orchestrator pattern where a central agent coordinates specialists

✅ `is_enabled` provides runtime feature gating — conditionally show tools based on user permissions or context

✅ Error handling via `failure_error_function` lets you provide friendly error messages to the LLM

**Next:** [Handoffs and Multi-Agent Orchestration](./04-handoffs-multi-agent.md)

---

## Further reading

- [Tools docs](https://openai.github.io/openai-agents-python/tools/) — Complete tool reference
- [MCP integration](https://openai.github.io/openai-agents-python/mcp/) — Model Context Protocol servers
- [Tool API reference](https://openai.github.io/openai-agents-python/ref/tool/) — FunctionTool, WebSearchTool, etc.

---

*[Back to OpenAI Agents SDK Overview](./00-openai-agents-sdk.md)*

<!-- 
Sources Consulted:
- OpenAI Agents SDK Tools page: https://openai.github.io/openai-agents-python/tools/
- OpenAI Agents SDK MCP page: https://openai.github.io/openai-agents-python/mcp/
- Tool API reference: https://openai.github.io/openai-agents-python/ref/tool/
-->
