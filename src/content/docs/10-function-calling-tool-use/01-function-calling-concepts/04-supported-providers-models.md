---
title: "Supported Providers and Models (2025-2026)"
---

# Supported Providers and Models (2025-2026)

## Introduction

Function calling is now a standard capability across all major AI providers, but each implements it differently. The core concept is identical — define tools, receive structured calls, execute and return results — yet the API field names, response structures, and advanced features vary significantly. Understanding these differences is essential for building cross-provider applications or choosing the right provider for your use case.

This lesson provides a comprehensive comparison of function calling across OpenAI, Anthropic, Google Gemini, and open-source models as of 2025-2026.

### What we'll cover

- OpenAI function calling: Responses API and Chat Completions API
- Anthropic tool use: Messages API
- Google Gemini: function calling with the Gen AI SDK
- Open-source models with tool support
- API structure differences side-by-side
- Feature comparison across providers

### Prerequisites

- Understanding of function calling lifecycle ([Lesson 01](./01-what-is-function-calling.md))
- Experience making API calls (Unit 4)
- Familiarity with JSON Schema basics

---

## Provider overview

| Provider | Models with Function Calling | API | Strict Mode | Parallel Calls | Custom Tools |
|----------|------------------------------|-----|-------------|----------------|--------------|
| **OpenAI** | GPT-5, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, o4-mini, o3 | Responses API, Chat Completions | ✅ (default in Responses) | ✅ | ✅ (CFG support) |
| **Anthropic** | Claude Opus 4.6, Claude Sonnet 4, Claude Haiku 4.5 | Messages API | ✅ | ✅ | — |
| **Google** | Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, Gemini 2.5 Flash | Gen AI SDK / REST | ✅ (VALIDATED mode) | ✅ | — |
| **Open-source** | Llama 3.2, Mistral, Qwen 2.5, Command R+ | Various (via Ollama, vLLM, etc.) | Varies | Varies | — |

---

## OpenAI function calling

OpenAI offers two APIs for function calling. The **Responses API** is the current recommended API, while **Chat Completions** remains available for backward compatibility.

### Responses API (recommended)

The Responses API is OpenAI's latest interface, designed for tool use and agentic workflows:

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location", "units"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# Send request
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

# Process function calls
for item in response.output:
    if item.type == "function_call":
        print(f"Function: {item.name}")
        print(f"Arguments: {item.arguments}")
        print(f"Call ID: {item.call_id}")
```

**Output:**
```
Function: get_weather
Arguments: {"location": "Paris, France", "units": "celsius"}
Call ID: call_abc123xyz
```

### Returning results (Responses API)

```python
# Build input with function call results
input_messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]
input_messages += response.output  # Include model's function call

# Add function result
input_messages.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": json.dumps({"temperature": 18, "condition": "partly cloudy"})
})

# Get final response
final = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools
)
print(final.output_text)
```

**Output:**
```
It's currently 18°C and partly cloudy in Paris.
```

### Chat Completions API (legacy)

The older API uses a different structure but the same concept:

```python
# Chat Completions uses a different response structure
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {  # Note: nested under "function" key
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "units"]
            },
            "strict": True
        }
    }]
)

# Response structure differs
choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    for tool_call in choice.message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        print(f"ID: {tool_call.id}")
```

**Output:**
```
Function: get_weather
Arguments: {"location": "Paris, France", "units": "celsius"}
ID: call_abc123
```

### Key OpenAI differences: Responses vs. Chat Completions

| Aspect | Responses API | Chat Completions API |
|--------|--------------|---------------------|
| Tool definition | Top-level `name`, `parameters` | Nested under `function` key |
| Response format | `output` array with `type: "function_call"` | `choices[0].message.tool_calls` |
| Result format | `function_call_output` with `call_id` | `tool` role message with `tool_call_id` |
| Strict mode | Default (opt out with `strict: false`) | Opt-in with `strict: true` |
| Reasoning items | Included in output (for reasoning models) | Not available |
| Custom tools | ✅ Supported (with CFG) | ❌ Not supported |
| Built-in tools | `web_search`, `code_interpreter`, etc. | Limited |

> **💡 Tip:** For new projects, use the Responses API. It's designed for modern tool use patterns and supports features like reasoning items, custom tools, and built-in platform tools.

### OpenAI supported models

| Model | Function Calling | Strict Mode | Parallel Calls | Best For |
|-------|-----------------|-------------|----------------|----------|
| **GPT-5** | ✅ | ✅ | ✅ | Complex reasoning + tool use |
| **GPT-4.1** | ✅ | ✅ | ✅ | Best balance of capability and cost |
| **GPT-4.1-mini** | ✅ | ✅ | ✅ | Cost-effective general use |
| **GPT-4.1-nano** | ✅ | ✅ | ⚠️ (may duplicate) | Fastest, lowest cost |
| **o4-mini** | ✅ | ✅ | ✅ | Reasoning-optimized tool use |
| **o3** | ✅ | ✅ | ✅ | Heavy reasoning tasks |

> **Note:** GPT-4.1-nano may sometimes duplicate tool calls when parallel tool calls are enabled. OpenAI recommends disabling parallel calls for this model.

---

## Anthropic tool use

Anthropic implements function calling through their **Messages API** using the term "tool use."

### Defining and calling tools

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "input_schema": {  # Note: "input_schema" not "parameters"
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location", "units"]
        }
    }],
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
)

# Check stop reason
print(f"Stop reason: {response.stop_reason}")

# Extract tool use blocks
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")  # Note: "input" not "arguments"
        print(f"ID: {block.id}")
```

**Output:**
```
Stop reason: tool_use
Tool: get_weather
Input: {'location': 'Paris, France', 'units': 'celsius'}
ID: toolu_abc123
```

### Returning results (Anthropic)

Anthropic uses a specific message structure for tool results:

```python
# Continue the conversation with tool results
response_2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "units"]
        }
    }],
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response.content},  # Model's tool_use response
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_abc123",  # Must match the tool_use ID
                "content": json.dumps({
                    "temperature": 18,
                    "condition": "partly cloudy"
                })
            }]
        }
    ]
)

print(response_2.content[0].text)
```

**Output:**
```
It's currently 18°C and partly cloudy in Paris, France.
```

### Anthropic tool types

Anthropic distinguishes between **client tools** and **server tools**:

| Tool Type | Execution | Implementation |
|-----------|-----------|---------------|
| **Client tools** (custom) | Your code | You define schema and execute |
| **Server tools** (web search) | Anthropic's servers | Just enable — automatic execution |
| **Anthropic-defined** (computer use, text editor) | Your code | Anthropic defines schema, you implement |

```python
# Server tool example: web search (executes on Anthropic's servers)
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3
    }],
    messages=[{"role": "user", "content": "What happened in tech news today?"}]
)
# Web search executes automatically — no client-side implementation needed
```

### Anthropic supported models

| Model | Tool Use | Strict Mode | Parallel Calls | Token Overhead |
|-------|----------|-------------|----------------|---------------|
| **Claude Opus 4.6** | ✅ | ✅ | ✅ | 346 tokens (auto), 313 (any) |
| **Claude Sonnet 4** | ✅ | ✅ | ✅ | 346 tokens (auto), 313 (any) |
| **Claude Haiku 4.5** | ✅ | ✅ | ✅ | 346 tokens (auto), 313 (any) |

> **Note:** Tool use adds a system prompt automatically. The token overhead listed above is for the tool use system prompt alone, not including your tool definitions.

---

## Google Gemini function calling

Google implements function calling through the **Gen AI SDK** using `function_declarations`.

### Defining and calling functions

```python
from google import genai
from google.genai import types

client = genai.Client()

# Define function declaration
get_weather_decl = {
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country, e.g. 'Paris, France'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location", "units"]
    }
}

# Wrap in Tool object
tools = types.Tool(function_declarations=[get_weather_decl])
config = types.GenerateContentConfig(tools=[tools])

# Send request
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Paris?",
    config=config
)

# Extract function call
function_call = response.candidates[0].content.parts[0].function_call
print(f"Function: {function_call.name}")
print(f"Args: {function_call.args}")
```

**Output:**
```
Function: get_weather
Args: {'location': 'Paris, France', 'units': 'celsius'}
```

### Returning results (Gemini)

```python
# Create function response part
function_response_part = types.Part.from_function_response(
    name="get_weather",
    response={"result": {"temperature": 18, "condition": "partly cloudy"}}
)

# Build conversation with results
contents = [
    types.Content(role="user", parts=[
        types.Part(text="What's the weather in Paris?")
    ]),
    response.candidates[0].content,  # Model's function call
    types.Content(role="user", parts=[function_response_part])
]

# Get final response
final_response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=config
)
print(final_response.text)
```

**Output:**
```
It's currently 18°C and partly cloudy in Paris, France.
```

### Gemini automatic function calling (Python SDK)

Gemini's Python SDK offers a unique feature: **automatic function calling**, where you pass Python functions directly and the SDK handles the entire call loop:

```python
from google import genai
from google.genai import types

def get_weather(location: str, units: str = "celsius") -> dict:
    """Get current weather for a location.
    
    Args:
        location: City and country, e.g. 'Paris, France'
        units: Temperature units (celsius or fahrenheit)
    
    Returns:
        Dictionary with temperature and condition.
    """
    # Your actual implementation
    return {"temperature": 18, "condition": "partly cloudy"}

client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_weather]  # Pass Python function directly!
)

# The SDK automatically:
# 1. Converts function to declaration
# 2. Detects function call in response
# 3. Calls your Python function
# 4. Sends result back to model
# 5. Returns final text response
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Paris?",
    config=config
)
print(response.text)
```

**Output:**
```
It's currently 18°C and partly cloudy in Paris.
```

> **Note:** Automatic function calling is a Python SDK-only feature. When using JavaScript or REST, you must handle the function call loop manually.

### Gemini function calling modes

| Mode | Behavior | API Value |
|------|----------|-----------|
| **AUTO** (default) | Model decides freely | `mode="AUTO"` |
| **ANY** | Must call a function | `mode="ANY"` |
| **NONE** | Cannot call functions | `mode="NONE"` |
| **VALIDATED** (preview) | Call or text, with schema guarantee | `mode="VALIDATED"` |

```python
# Force function calling
config = types.GenerateContentConfig(
    tools=[tools],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY",
            allowed_function_names=["get_weather"]  # Optional: restrict to specific functions
        )
    )
)
```

### Gemini supported models

| Model | Function Calling | Parallel Calls | Compositional Calls |
|-------|-----------------|----------------|-------------------|
| **Gemini 3 Pro** | ✅ | ✅ | ✅ |
| **Gemini 3 Flash** | ✅ | ✅ | ✅ |
| **Gemini 2.5 Pro** | ✅ | ✅ | ✅ |
| **Gemini 2.5 Flash** | ✅ | ✅ | ✅ |
| **Gemini 2.5 Flash-Lite** | ✅ | ✅ | ✅ |
| **Gemini 2.0 Flash** | ✅ | ✅ | ✅ |
| **Gemini 2.0 Flash-Lite** | ❌ | ❌ | ❌ |

---

## Open-source models

Several open-source models support function calling, though capabilities vary significantly.

### Llama 3.2 (Meta)

Llama 3.2 supports tool use through a specific prompt format:

```python
# Llama uses a chat template with tool definitions
# Typically served via Ollama, vLLM, or Together AI
# Tool format follows a structured prompt template

# Example using Ollama with the OpenAI-compatible API
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
)
```

### Other open-source models with tool support

| Model | Tool Support | Notes |
|-------|-------------|-------|
| **Llama 3.2** (8B, 70B) | ✅ | Best open-source tool use |
| **Mistral Large** | ✅ | Strong function calling |
| **Qwen 2.5** (7B, 72B) | ✅ | Good multilingual tool use |
| **Command R+** (Cohere) | ✅ | Enterprise-focused |
| **DeepSeek V3** | ✅ | Cost-effective option |

> **Warning:** Open-source models generally do not support strict mode. Schema adherence is best-effort, and you should always validate function call arguments before execution.

---

## Side-by-side comparison

### Tool definition format

```python
# ===== OpenAI (Responses API) =====
openai_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"],
        "additionalProperties": False
    },
    "strict": True
}

# ===== Anthropic =====
anthropic_tool = {
    "name": "get_weather",
    "description": "Get weather for a location.",
    "input_schema": {  # Different key name
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}

# ===== Gemini =====
gemini_tool = {
    "name": "get_weather",
    "description": "Get weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"  # Descriptions on properties
            }
        },
        "required": ["location"]
    }
}
# Wrapped in: types.Tool(function_declarations=[gemini_tool])
```

### Function call response format

```python
# ===== OpenAI (Responses API) =====
# item.type == "function_call"
# item.name → function name
# item.arguments → JSON string
# item.call_id → unique ID

# ===== Anthropic =====
# block.type == "tool_use"
# block.name → function name
# block.input → dict (already parsed!)
# block.id → unique ID
# response.stop_reason == "tool_use"

# ===== Gemini =====
# part.function_call.name → function name
# part.function_call.args → dict (already parsed!)
# No explicit call ID (positional)
```

### Result return format

```python
# ===== OpenAI (Responses API) =====
{"type": "function_call_output", "call_id": "call_abc", "output": "json_string"}

# ===== Anthropic =====
# In "user" role message:
{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "json_string"}

# ===== Gemini =====
# Using SDK helper:
# types.Part.from_function_response(name="func_name", response={...})
```

---

## Cross-provider adapter pattern

When building applications that work across providers, a common pattern is creating an adapter:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolCall:
    """Provider-agnostic tool call representation."""
    name: str
    arguments: dict[str, Any]
    call_id: str

def extract_tool_calls(response: Any, provider: str) -> list[ToolCall]:
    """Extract tool calls from any provider's response."""
    calls = []
    
    if provider == "openai":
        for item in response.output:
            if item.type == "function_call":
                import json
                calls.append(ToolCall(
                    name=item.name,
                    arguments=json.loads(item.arguments),
                    call_id=item.call_id
                ))
    
    elif provider == "anthropic":
        for block in response.content:
            if block.type == "tool_use":
                calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input,  # Already a dict
                    call_id=block.id
                ))
    
    elif provider == "gemini":
        for i, part in enumerate(response.candidates[0].content.parts):
            if part.function_call:
                calls.append(ToolCall(
                    name=part.function_call.name,
                    arguments=dict(part.function_call.args),
                    call_id=f"gemini_call_{i}"  # Generate ID
                ))
    
    return calls
```

**Output:**
```
# Regardless of provider, you get:
# [ToolCall(name="get_weather", arguments={"location": "Paris"}, call_id="...")]
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Start with one provider, then abstract | Build working code first, add cross-provider support later |
| Use strict mode when available | Guarantees valid function calls across every request |
| Test tool definitions with each provider | Behavior differences emerge in edge cases |
| Keep an eye on token overhead | Each provider adds different system prompt tokens for tool use |
| Read provider-specific docs for updates | Function calling APIs evolve rapidly |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Assuming identical APIs across providers | Map field names explicitly: `parameters` vs `input_schema` |
| Using `arguments` as a dict (OpenAI) | OpenAI returns arguments as a JSON string — use `json.loads()` |
| Forgetting `call_id` / `tool_use_id` matching | Each result must reference the correct call ID |
| Expecting strict mode in open-source models | Add explicit validation for open-source model outputs |
| Not accounting for provider token overhead | Factor in the ~300-350 system prompt tokens for tool use |

---

## Hands-on exercise

### Your task

Implement the same function calling example across two different providers.

### Requirements

1. Define a `search_products` tool with these parameters:
   - `query` (string, required)
   - `category` (enum: "electronics", "clothing", "books", "home")
   - `max_price` (number, optional / nullable)
2. Write the tool definition in both OpenAI Responses API and Anthropic Messages API formats
3. Write the code to extract function calls from each provider's response
4. Write the code to return results to each provider

### Expected result

Two complete implementations showing the same functionality with provider-specific API differences.

<details>
<summary>💡 Hints (click to expand)</summary>

- OpenAI uses `parameters`, Anthropic uses `input_schema`
- OpenAI returns `arguments` as a JSON string, Anthropic returns `input` as a dict
- For strict mode in OpenAI, all fields must be in `required` and optional fields use `["number", "null"]`
- Anthropic's `tool_choice` uses `{"type": "auto"}` not just `"auto"`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

# ============================================================
# OpenAI Responses API Version
# ============================================================
from openai import OpenAI

openai_client = OpenAI()

openai_tool = {
    "type": "function",
    "name": "search_products",
    "description": "Search the product catalog by query, category, and price range.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms for product lookup"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books", "home"],
                "description": "Product category filter"
            },
            "max_price": {
                "type": ["number", "null"],
                "description": "Maximum price filter, null for no limit"
            }
        },
        "required": ["query", "category", "max_price"],
        "additionalProperties": False
    },
    "strict": True
}

# Make request
openai_response = openai_client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Find me wireless headphones under $100"}],
    tools=[openai_tool]
)

# Extract function call
input_messages = [{"role": "user", "content": "Find me wireless headphones under $100"}]
input_messages += openai_response.output

for item in openai_response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)  # JSON string → dict
        # Execute your function
        result = {"products": [{"name": "BT-500", "price": 79.99}]}
        
        # Return result
        input_messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result)
        })

# ============================================================
# Anthropic Messages API Version
# ============================================================
import anthropic

anthropic_client = anthropic.Anthropic()

anthropic_tool = {
    "name": "search_products",
    "description": "Search the product catalog by query, category, and price range.",
    "input_schema": {  # Different key
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms for product lookup"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books", "home"],
                "description": "Product category filter"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price filter"
            }
        },
        "required": ["query", "category"]  # max_price is truly optional here
    }
}

# Make request
anthropic_response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[anthropic_tool],
    messages=[
        {"role": "user", "content": "Find me wireless headphones under $100"}
    ]
)

# Extract tool use
for block in anthropic_response.content:
    if block.type == "tool_use":
        args = block.input  # Already a dict!
        # Execute your function
        result = {"products": [{"name": "BT-500", "price": 79.99}]}
        
        # Return result in follow-up message
        final = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[anthropic_tool],
            messages=[
                {"role": "user", "content": "Find me wireless headphones under $100"},
                {"role": "assistant", "content": anthropic_response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    }]
                }
            ]
        )
```

</details>

### Bonus challenges

- [ ] Add the Gemini implementation as a third version
- [ ] Build the cross-provider adapter class from the lesson for all three
- [ ] Compare token usage across providers for the same tool definition

---

## Summary

✅ **OpenAI** offers two APIs — Responses (recommended) and Chat Completions (legacy) — with strict mode enabled by default in Responses

✅ **Anthropic** uses "tool use" terminology with `input_schema` (not `parameters`) and returns `input` as a dict (not a JSON string)

✅ **Google Gemini** wraps declarations in `types.Tool` objects and uniquely supports **automatic function calling** in the Python SDK

✅ **Open-source models** (Llama 3.2, Mistral, Qwen) support tool use but lack strict mode — always validate their outputs

✅ The core lifecycle is identical across providers: **define → call → execute → return → respond** — only the field names differ

✅ Build a **cross-provider adapter** when your application needs to work with multiple AI providers

**Next:** [Reasoning Models and Function Calling](./05-reasoning-models.md)

---

[← Previous: Function Calling vs. Prompting](./03-function-calling-vs-prompting.md) | [Back to Function Calling Concepts](./00-function-calling-concepts.md) | [Next: Reasoning Models →](./05-reasoning-models.md)

<!--
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API Migration: https://platform.openai.com/docs/guides/migrate-to-responses
- Anthropic Tool Use Documentation: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Anthropic Pricing (Tool Use Tokens): https://platform.claude.com/docs/en/docs/build-with-claude/tool-use#pricing
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Google Gemini Supported Models: https://ai.google.dev/gemini-api/docs/function-calling#supported-models
-->
