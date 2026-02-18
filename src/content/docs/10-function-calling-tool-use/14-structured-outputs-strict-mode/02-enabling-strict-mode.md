---
title: "Enabling Strict Mode"
---

# Enabling Strict Mode

## Introduction

Knowing *what* strict mode does is one thing — knowing *how to turn it on* across different providers is another. Each major AI platform implements strict schema enforcement with slightly different syntax, configuration, and defaults. In this lesson, we walk through enabling strict mode for OpenAI, Anthropic, and Google Gemini, highlighting the specific parameters, API differences, and gotchas for each.

### What we'll cover

- Enabling strict mode in OpenAI's function calling and structured outputs
- Enabling strict tool use in Anthropic's Messages API
- Using Gemini's `VALIDATED` function calling mode
- Default behaviors and how they differ across providers
- Using SDK helpers (Pydantic, Zod) for automatic schema generation

### Prerequisites

- Understanding of what strict mode guarantees ([Sub-lesson 01](./01-what-is-strict-mode.md))
- Basic experience with at least one AI provider's API
- Python and/or JavaScript development environment set up

---

## OpenAI: `strict: true` in function definitions

OpenAI was the first major provider to introduce strict mode for function calling, launching it as part of the **Structured Outputs** feature. There are two ways to use it: in function (tool) definitions and in response format configuration.

### Strict mode in function calling

Add `"strict": true` at the top level of your function definition:

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Retrieves current weather for a given location.",
        "strict": True,                          # ← Enable strict mode
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
                    "description": "Temperature unit"
                }
            },
            "required": ["location", "units"],    # ← All fields required
            "additionalProperties": False          # ← Must be false
        }
    }
]

response = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# The function call arguments are guaranteed to match the schema
for item in response.output:
    if item.type == "function_call":
        print(item.arguments)
        # Always valid: {"location": "Tokyo, Japan", "units": "celsius"}
```

**Output:**
```json
{"location": "Tokyo, Japan", "units": "celsius"}
```

### Strict mode in structured response format

You can also use strict mode to control the model's text response format (not just function calls):

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract weather data from the text."},
        {"role": "user", "content": "It's 22°C and sunny in Berlin with 45% humidity."}
    ],
    text_format=WeatherReport
)

report = response.output_parsed
print(f"{report.location}: {report.temperature}°C, {report.conditions}")
```

**Output:**
```
Berlin: 22.0°C, sunny
```

> **Note:** When using Pydantic models with the OpenAI SDK's `parse()` method, strict mode is enabled automatically. The SDK converts your Pydantic model to a JSON Schema with `strict: true`, `additionalProperties: false`, and all fields in `required`.

### OpenAI default behavior

| API | Default strict behavior |
|-----|------------------------|
| **Responses API** | Normalizes schemas into strict mode automatically (sets `additionalProperties: false`, marks all fields as `required`) |
| **Chat Completions API** | Non-strict by default — best-effort schema following |

> **Warning:** The Responses API's automatic normalization can make previously optional fields mandatory. If you explicitly want non-strict behavior in the Responses API, set `strict: false`.

---

## Anthropic: `strict: true` in tool definitions

Anthropic added strict tool use as part of their **Structured Outputs** feature. The syntax is similar to OpenAI's approach.

### Enabling strict tool use

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
    tools=[
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "strict": True,                         # ← Enable strict mode
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"],
                "additionalProperties": False         # ← Recommended
            }
        }
    ]
)

# Extract the tool call
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
        # Guaranteed to match input_schema
```

**Output:**
```
Tool: get_weather
Input: {'location': 'San Francisco, CA'}
```

### Anthropic JSON outputs (response format)

Anthropic also supports structured JSON response format via `output_config`:

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Extract: John Smith, john@test.com, wants Enterprise plan"}
    ],
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "plan": {"type": "string"}
                },
                "required": ["name", "email", "plan"],
                "additionalProperties": False
            }
        }
    }
)

print(response.content[0].text)
```

**Output:**
```json
{"name": "John Smith", "email": "john@test.com", "plan": "Enterprise"}
```

> **🤖 AI Context:** Anthropic's strict tool use and JSON outputs are two separate features that can be used independently or together. Strict tool use validates function call inputs; JSON outputs validate the model's text response format.

### Using Pydantic with Anthropic's SDK

```python
from pydantic import BaseModel
from anthropic import Anthropic, transform_schema

class ContactInfo(BaseModel):
    name: str
    email: str
    plan_interest: str

client = Anthropic()

# Option 1: Using parse() — handles schema transformation automatically
response = client.messages.parse(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Extract: Alice, alice@co.com, Pro plan"}],
    output_format=ContactInfo
)

print(response.parsed_output)  # ContactInfo object
```

**Output:**
```
name='Alice' email='alice@co.com' plan_interest='Pro'
```

---

## Google Gemini: `VALIDATED` mode

Gemini approaches strict mode differently from OpenAI and Anthropic. Instead of a `strict` flag on individual tool definitions, Gemini uses **function calling modes** configured at the request level.

### Function calling modes

| Mode | Behavior |
|------|----------|
| `AUTO` (default) | Model decides whether to call a function or respond with text; no schema enforcement guarantee |
| `ANY` | Model must call a function; guarantees schema adherence |
| `NONE` | Model cannot call functions |
| `VALIDATED` (Preview) | Model may call functions or respond with text; guarantees schema adherence when it does call a function |

### Enabling VALIDATED mode

```python
from google import genai
from google.genai import types

client = genai.Client()

# Define function declaration
get_weather_fn = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'London'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location", "unit"]
    }
}

tools = types.Tool(function_declarations=[get_weather_fn])

# Configure VALIDATED mode for schema enforcement
config = types.GenerateContentConfig(
    tools=[tools],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="VALIDATED"                       # ← Schema enforcement
        )
    )
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the temperature in London?",
    config=config
)

# Extract function call
if response.candidates[0].content.parts[0].function_call:
    fn_call = response.candidates[0].content.parts[0].function_call
    print(f"Function: {fn_call.name}")
    print(f"Args: {fn_call.args}")
```

**Output:**
```
Function: get_weather
Args: {'location': 'London', 'unit': 'celsius'}
```

### ANY mode: forced function calling with schema adherence

If you want the model to *always* call a function (never respond with plain text), use `ANY` mode:

```python
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

> **Important:** `ANY` mode forces a function call on every request. If the user's prompt doesn't relate to any available function, the model will still attempt a function call, which may produce unexpected results. Use `VALIDATED` mode when you want schema enforcement but also want the model to respond with text when appropriate.

---

## Provider comparison

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| **Strict flag location** | Per-tool: `strict: true` | Per-tool: `strict: true` | Request-level: `mode: "VALIDATED"` |
| **Schema key** | `parameters` | `input_schema` | `parameters` (in function declaration) |
| **Enforcement mechanism** | Constrained decoding | Constrained sampling | Output validation |
| **SDK Pydantic support** | `parse()` with `text_format` | `parse()` with `output_format` | `from_callable()` for auto-declaration |
| **Supported models** | GPT-4o and later | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 | Gemini 2.0 Flash and later |
| **Default behavior** | Responses API auto-normalizes to strict | Non-strict by default | AUTO mode (non-strict) |

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| ✅ Enable strict mode explicitly | Don't rely on default behaviors — they vary across providers and API versions |
| ✅ Use SDK schema helpers | Pydantic (Python) and Zod (JavaScript) generate compliant schemas automatically |
| ✅ Test your schema before deploying | Invalid schemas will cause API errors at request time, not at definition time |
| ✅ Read provider-specific docs | Schema requirements differ subtly between OpenAI, Anthropic, and Gemini |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using OpenAI's `parameters` key with Anthropic's API | Anthropic uses `input_schema`, not `parameters` |
| Assuming Gemini's `ANY` mode is the same as `VALIDATED` | `ANY` forces a function call every time; `VALIDATED` allows text responses too |
| Forgetting that OpenAI Responses API auto-normalizes | Previously optional fields become required — set `strict: false` if you need non-strict behavior |
| Not including `additionalProperties: false` | Required for OpenAI strict mode; strongly recommended for all providers |

---

## Hands-on exercise

### Your task

Implement the same `search_products` tool with strict mode enabled across two different providers.

### Requirements

1. Define a `search_products` function with fields: `query` (string), `category` (enum: "electronics", "clothing", "books"), `max_results` (integer)
2. Write the tool definition for **OpenAI** with `strict: true`
3. Write the same tool definition for **Anthropic** with `strict: true`
4. Note the three syntax differences between the two definitions

### Expected result

Two working tool definitions, each in their provider's format, and a list of syntax differences.

<details>
<summary>💡 Hints (click to expand)</summary>

- OpenAI uses `parameters`; Anthropic uses `input_schema`
- The `strict` flag goes at the same level as `name` and `description` in both
- Both require `additionalProperties: false` and all fields in `required`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**OpenAI format:**
```json
{
    "type": "function",
    "name": "search_products",
    "description": "Search for products in the catalog",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category to search in"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return"
            }
        },
        "required": ["query", "category", "max_results"],
        "additionalProperties": false
    }
}
```

**Anthropic format:**
```json
{
    "name": "search_products",
    "description": "Search for products in the catalog",
    "strict": true,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category to search in"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return"
            }
        },
        "required": ["query", "category", "max_results"],
        "additionalProperties": false
    }
}
```

**Three syntax differences:**

1. OpenAI wraps the tool in `{"type": "function", ...}` — Anthropic omits the `type` wrapper
2. OpenAI uses `"parameters"` for the schema — Anthropic uses `"input_schema"`
3. OpenAI uses Python `False` / JSON `false` for `additionalProperties` — both work the same way, but the key names differ

</details>

### Bonus challenges

- [ ] Write the equivalent Gemini tool definition using `VALIDATED` mode
- [ ] Use Pydantic `BaseModel` to generate the schema automatically for both OpenAI and Anthropic

---

## Summary

✅ **OpenAI** uses `strict: true` per-tool definition with `parameters` schema and auto-normalizes in the Responses API

✅ **Anthropic** uses `strict: true` per-tool definition with `input_schema` and offers both strict tool use and JSON output features

✅ **Gemini** uses request-level `VALIDATED` mode (or `ANY` for forced calling) rather than per-tool strict flags

✅ All three providers require `additionalProperties: false` and all fields in `required` for full schema enforcement

✅ Use SDK helpers (Pydantic `parse()`, Zod) to generate compliant schemas automatically and avoid manual errors

---

**Previous:** [What Is Strict Mode ←](./01-what-is-strict-mode.md)

**Next:** [Schema Requirements for Strict Mode →](./03-schema-requirements.md)

---

## Further reading

- [OpenAI Function Calling — Strict Mode](https://platform.openai.com/docs/guides/function-calling#strict-mode) — Official OpenAI strict mode documentation
- [Anthropic Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) — Strict tool use and JSON outputs
- [Gemini Function Calling Modes](https://ai.google.dev/gemini-api/docs/function-calling#function-calling-modes) — VALIDATED and ANY mode configuration

---

*[Back to Structured Outputs & Strict Mode Overview](./00-structured-outputs-strict-mode.md)*

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- OpenAI Structured Outputs Guide: https://platform.openai.com/docs/guides/structured-outputs
- Anthropic Structured Outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Anthropic Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
