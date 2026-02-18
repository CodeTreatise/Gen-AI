---
title: "Provider-Specific Definition Formats"
---

# Provider-Specific Definition Formats

## Introduction

Each AI provider uses a slightly different format for function definitions. The core concepts are the same — name, description, parameters — but the structure, field names, and wrapping differ. If you are building a multi-provider application or switching between providers, you need to understand these differences and know how to convert between formats.

This lesson covers the exact definition format for OpenAI, Anthropic, and Gemini, highlights the key differences, and provides a conversion utility.

### What we'll cover

- OpenAI's `tools` array with `function` objects (Responses API and Chat Completions)
- Anthropic's `tools` array with `input_schema`
- Gemini's `function_declarations` in a `Tool` object
- Side-by-side format comparison
- A cross-provider conversion function

### Prerequisites

- Understanding of function definition anatomy ([Lesson 01](./01-function-definition-structure.md))
- Basic familiarity with at least one provider's API

---

## OpenAI format

OpenAI defines functions in a `tools` array. Each tool has a `type: "function"` wrapper, plus `name`, `description`, `parameters`, and `strict` fields.

### Responses API (recommended)

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a location. "
            "Use when the user asks about weather or temperature."
        ),
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
            "required": ["location", "units"],
            "additionalProperties": False
        },
        "strict": True
    }
]

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# Extract the function call
for item in response.output:
    if item.type == "function_call":
        print(f"Function: {item.name}")
        print(f"Arguments: {item.arguments}")
        print(f"Call ID: {item.call_id}")
```

**Output:**
```
Function: get_weather
Arguments: {"location":"Tokyo, Japan","units":"celsius"}
Call ID: call_abc123
```

### Chat Completions API (legacy)

The Chat Completions API wraps the function inside an extra `function` key:

```python
# Chat Completions format — note the extra "function" wrapper
tools_chat = [
    {
        "type": "function",
        "function": {                          # Extra wrapper
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country"
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
    }
]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools_chat
)
```

### Responses vs. Chat Completions structure

| Aspect | Responses API | Chat Completions |
|--------|--------------|------------------|
| Tool wrapper | `{"type": "function", "name": ...}` | `{"type": "function", "function": {"name": ...}}` |
| `strict` location | Top-level sibling of `name` | Inside `function` object |
| `parameters` location | Top-level sibling of `name` | Inside `function` object |
| Default strict mode | `true` (auto-normalized) | `false` (best-effort) |
| Recommended for | New projects | Existing integrations |

> **💡 Tip:** If you are starting a new project, use the Responses API. It has a simpler structure and strict mode is the default.

---

## Anthropic format

Anthropic uses `input_schema` instead of `parameters`, and there is no `type: "function"` wrapper. Tools are defined directly with `name`, `description`, and `input_schema`.

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a location. "
            "Use when the user asks about weather or temperature."
        ),
        "input_schema": {                      # NOT "parameters"
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
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Extract function call
for block in response.content:
    if block.type == "tool_use":
        print(f"Function: {block.name}")
        print(f"Arguments: {block.input}")     # Already a dict, NOT JSON string
        print(f"Call ID: {block.id}")
```

**Output:**
```
Function: get_weather
Arguments: {'location': 'Tokyo, Japan', 'units': 'celsius'}
Call ID: toolu_01abc123
```

### Anthropic with strict mode

Anthropic supports `strict: true` to guarantee schema conformance:

```python
tools_strict = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "strict": True,                        # Enable strict mode
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location", "units"]
        }
    }
]
```

### Key Anthropic differences

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Schema field | `parameters` | `input_schema` |
| Tool type wrapper | `"type": "function"` required | No type wrapper |
| Argument format in response | JSON string (must `json.loads()`) | Dict (already parsed) |
| Call ID field | `call_id` | `id` |
| Response block type | `function_call` | `tool_use` |
| `additionalProperties` | Required for strict | Not required |
| Token overhead | Included in standard tokens | ~346 extra tokens per request |

---

## Gemini format

Gemini uses `function_declarations` inside a `types.Tool` object. The schema follows a subset of OpenAPI, similar to JSON Schema.

```python
from google import genai
from google.genai import types

client = genai.Client()

# Define function declaration (dict format)
get_weather_declaration = {
    "name": "get_weather",
    "description": (
        "Get current weather conditions for a location. "
        "Use when the user asks about weather or temperature."
    ),
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
tools = types.Tool(function_declarations=[get_weather_declaration])
config = types.GenerateContentConfig(tools=[tools])

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Tokyo?",
    config=config
)

# Extract function call
for part in response.candidates[0].content.parts:
    if part.function_call:
        print(f"Function: {part.function_call.name}")
        print(f"Arguments: {dict(part.function_call.args)}")
```

**Output:**
```
Function: get_weather
Arguments: {'location': 'Tokyo, Japan', 'units': 'celsius'}
```

### Gemini automatic declaration from Python functions

Gemini's Python SDK can generate declarations directly from Python functions:

```python
from google.genai import types

def get_weather(location: str, units: str) -> dict:
    """Get current weather conditions for a location.

    Args:
        location: City and country, e.g. 'Paris, France'
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        Dictionary with temperature, humidity, and conditions.
    """
    # Implementation here
    return {"temp": 22, "units": units, "condition": "Sunny"}

# Auto-generate declaration from the function
fn_decl = types.FunctionDeclaration.from_callable(
    callable=get_weather, client=client
)
print(fn_decl.to_json_dict())
```

**Output:**
```json
{
  "name": "get_weather",
  "description": "Get current weather conditions for a location.\n\nArgs:\n    location: ...",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "units": {"type": "string"}
    },
    "required": ["location", "units"]
  }
}
```

> **Note:** The auto-generated declaration uses the entire docstring as the top-level description. It does not parse individual `Args:` descriptions into property-level descriptions. For maximum accuracy, define declarations manually with per-property descriptions.

### Key Gemini differences

| Aspect | OpenAI | Gemini |
|--------|--------|--------|
| Tool container | `tools` list of dicts | `types.Tool(function_declarations=[...])` |
| Config location | `tools` param in API call | `types.GenerateContentConfig(tools=[...])` |
| Schema field | `parameters` | `parameters` (same name) |
| Strict mode | `strict: true` field | Not available — use `VALIDATED` mode instead |
| Arguments in response | JSON string | Dict (via `part.function_call.args`) |
| Auto-declaration | Not available | `FunctionDeclaration.from_callable()` |
| Type wrapper | `"type": "function"` required | Not needed — flat declaration |

---

## Side-by-side comparison

Here is the same function defined for all three providers:

```python
# ═══════════════════════════════════════════════════════
# OpenAI (Responses API)
# ═══════════════════════════════════════════════════════
openai_tool = {
    "type": "function",
    "name": "search_products",
    "description": "Search for products by query and category.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category"
            }
        },
        "required": ["query", "category"],
        "additionalProperties": False
    },
    "strict": True
}

# ═══════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════
anthropic_tool = {
    "name": "search_products",
    "description": "Search for products by query and category.",
    "input_schema": {                          # Different key name
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category"
            }
        },
        "required": ["query", "category"]
        # No additionalProperties needed
    }
}

# ═══════════════════════════════════════════════════════
# Gemini
# ═══════════════════════════════════════════════════════
gemini_declaration = {
    "name": "search_products",
    "description": "Search for products by query and category.",
    "parameters": {                            # Same key as OpenAI
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books"],
                "description": "Product category"
            }
        },
        "required": ["query", "category"]
        # No additionalProperties needed
    }
}
# Then wrap: types.Tool(function_declarations=[gemini_declaration])
```

### Format differences summary

| Field | OpenAI Responses | OpenAI Chat | Anthropic | Gemini |
|-------|-----------------|-------------|-----------|--------|
| Type wrapper | `"type": "function"` | `"type": "function"` | ❌ None | ❌ None |
| Function wrapper | ❌ None | `"function": {...}` | ❌ None | ❌ None |
| Schema key | `parameters` | `parameters` | `input_schema` | `parameters` |
| Strict mode | `strict: true` | `strict: true` | `strict: true` | `VALIDATED` mode |
| `additionalProperties` | Required (strict) | Required (strict) | Optional | Not used |
| Tool container | `tools=[...]` | `tools=[...]` | `tools=[...]` | `types.Tool(function_declarations=[...])` |

---

## Cross-provider conversion

Here is a utility that converts between provider formats:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class UniversalTool:
    """Provider-agnostic tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True

    def to_openai_responses(self) -> dict:
        """Convert to OpenAI Responses API format."""
        params = dict(self.parameters)
        if self.strict:
            params.setdefault("additionalProperties", False)
            # Ensure all properties are in required
            props = params.get("properties", {})
            params["required"] = list(props.keys())
        
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "strict": self.strict
        }

    def to_openai_chat(self) -> dict:
        """Convert to OpenAI Chat Completions format."""
        responses_format = self.to_openai_responses()
        # Move everything except "type" into "function" wrapper
        return {
            "type": "function",
            "function": {
                "name": responses_format["name"],
                "description": responses_format["description"],
                "parameters": responses_format["parameters"],
                "strict": responses_format["strict"]
            }
        }

    def to_anthropic(self) -> dict:
        """Convert to Anthropic format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def to_gemini(self) -> dict:
        """Convert to Gemini function_declaration format."""
        # Remove additionalProperties (not supported by Gemini)
        params = dict(self.parameters)
        params.pop("additionalProperties", None)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params
        }

    @classmethod
    def from_openai(cls, tool: dict) -> "UniversalTool":
        """Parse from OpenAI Responses API format."""
        return cls(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}),
            strict=tool.get("strict", False)
        )

    @classmethod
    def from_anthropic(cls, tool: dict) -> "UniversalTool":
        """Parse from Anthropic format."""
        return cls(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool.get("input_schema", {}),
            strict=tool.get("strict", False)
        )

# Usage
tool = UniversalTool(
    name="get_weather",
    description="Get current weather for a location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location", "units"]
    }
)

import json
print("OpenAI Responses:")
print(json.dumps(tool.to_openai_responses(), indent=2))
print("\nAnthropic:")
print(json.dumps(tool.to_anthropic(), indent=2))
print("\nGemini:")
print(json.dumps(tool.to_gemini(), indent=2))
```

**Output:**
```
OpenAI Responses:
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location", "units"],
    "additionalProperties": false
  },
  "strict": true
}

Anthropic:
{
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location", "units"]
  }
}

Gemini:
{
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location", "units"]
  }
}
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use the Responses API for new OpenAI projects | Simpler structure, strict mode default |
| Remember `input_schema` for Anthropic | Most common conversion mistake |
| Don't use `additionalProperties` with Gemini | Not supported — will cause errors |
| Test definitions against each provider's validation | Catch format issues before production |
| Use a `UniversalTool` abstraction in multi-provider apps | Single source of truth for tool definitions |
| Remove OpenAI-specific fields when converting to other providers | `strict`, `additionalProperties`, `type: "function"` wrapper |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `parameters` in Anthropic definitions | Use `input_schema` — Anthropic rejects `parameters` |
| Forgetting the `type: "function"` wrapper for OpenAI | Required for Responses API — requests fail without it |
| Including `additionalProperties` in Gemini declarations | Remove it — Gemini doesn't support this field |
| Using `json.loads()` on Anthropic arguments | Anthropic returns arguments as a dict, not a JSON string |
| Mixing Responses API and Chat Completions formats | Choose one — the `function` wrapper differs |
| Assuming `strict: true` works on all providers | OpenAI and Anthropic support it; Gemini uses `VALIDATED` mode |

---

## Hands-on exercise

### Your task

Write a function definition for a `send_notification` tool that works across all three providers.

### Requirements

1. Define the tool once using the `UniversalTool` class (or similar abstraction)
2. The function should accept:
   - `recipient` (required): user ID or email
   - `title` (required): notification title
   - `message` (required): notification body
   - `priority` (required): one of "low", "normal", "high"
   - `channel` (optional): "email", "push", "sms", or null
3. Generate valid definitions for OpenAI, Anthropic, and Gemini
4. Print all three and verify the structural differences

### Expected result

Three structurally different but functionally identical tool definitions.

<details>
<summary>💡 Hints (click to expand)</summary>

- Start from the `UniversalTool` dataclass in this lesson
- For the optional `channel` parameter, use `type: ["string", "null"]` in the universal format
- Remember to remove `additionalProperties` when converting to Gemini
- Verify that Anthropic output uses `input_schema`, not `parameters`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

tool = UniversalTool(
    name="send_notification",
    description=(
        "Send a notification to a user via their preferred channel. "
        "Use when the user asks to send a reminder, alert, or message. "
        "If no channel is specified, defaults to push notification."
    ),
    parameters={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "User ID or email address of the recipient"
            },
            "title": {
                "type": "string",
                "description": "Short notification title (3-10 words)"
            },
            "message": {
                "type": "string",
                "description": "Notification body text"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Notification priority level"
            },
            "channel": {
                "type": ["string", "null"],
                "enum": ["email", "push", "sms"],
                "description": "Delivery channel, or null for default (push)"
            }
        },
        "required": ["recipient", "title", "message", "priority", "channel"]
    }
)

# Generate all three formats
formats = {
    "OpenAI Responses": tool.to_openai_responses(),
    "Anthropic": tool.to_anthropic(),
    "Gemini": tool.to_gemini()
}

for name, fmt in formats.items():
    print(f"\n{'='*50}")
    print(f"{name}:")
    print(f"{'='*50}")
    print(json.dumps(fmt, indent=2))
```

</details>

### Bonus challenges

- [ ] Add a `from_gemini` classmethod to the `UniversalTool` class
- [ ] Handle the `type: ["string", "null"]` conversion for Gemini (which may need `nullable: true` instead)
- [ ] Build a test that validates each format against its provider's API (mock or real)

---

## Summary

✅ **OpenAI Responses API** uses `type: "function"` at top level with `name`, `description`, `parameters`, and `strict` as siblings — simplest format

✅ **OpenAI Chat Completions** wraps everything inside an extra `"function": {...}` key — more verbose, legacy format

✅ **Anthropic** uses `input_schema` instead of `parameters`, has no type wrapper, and returns arguments as dicts (not JSON strings)

✅ **Gemini** uses `function_declarations` inside `types.Tool()`, supports auto-declaration from Python functions, and does not use `additionalProperties`

✅ A **`UniversalTool` abstraction** lets you define tools once and convert to any provider format — essential for multi-provider applications

✅ The most common conversion mistakes are `parameters` vs. `input_schema` naming and forgetting to remove provider-specific fields

**Next:** [Naming Conventions](./03-naming-conventions.md)

---

[← Previous: Function Definition Structure](./01-function-definition-structure.md) | [Back to Defining Functions](./00-defining-functions.md) | [Next: Naming Conventions →](./03-naming-conventions.md)

<!--
Sources Consulted:
- OpenAI Function Calling Guide (Defining Functions): https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API vs Chat Completions: https://platform.openai.com/docs/guides/migrate-to-responses
- Anthropic Tool Use (input_schema format): https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling (function_declarations): https://ai.google.dev/gemini-api/docs/function-calling
- Google GenAI SDK (from_callable): https://ai.google.dev/gemini-api/docs/function-calling#automatic-function-calling-python-only
-->
