---
title: "Result Format Structure"
---

# Result Format Structure

## Introduction

Each AI provider has a specific format for receiving function results. Get the format wrong and the API rejects your request. Get it right but structure it poorly, and the model misinterprets your data. This lesson covers the exact result format for OpenAI, Anthropic, and Gemini — the message types, required fields, and how to construct them in Python.

We'll build each format from scratch, show how they fit into the conversation history, and then create a unified layer that handles all three providers transparently.

### What we'll cover

- OpenAI's `function_call_output` format with `call_id`
- Anthropic's `tool_result` content block with `tool_use_id` and `is_error`
- Gemini's `Part.from_function_response()` with function name and response dict
- Provider-specific message roles and conversation structure
- Building a unified result formatting layer

### Prerequisites

- Handling function calls ([Lesson 04](../04-handling-function-calls/00-handling-function-calls.md))
- Executing functions ([Lesson 05](../05-function-execution/00-function-execution.md))
- Basic understanding of each provider's API structure

---

## OpenAI: `function_call_output`

OpenAI's Responses API uses `function_call_output` items to return results. Each result is appended directly to the `input` list alongside the model's original function call output.

### The format

```python
{
    "type": "function_call_output",
    "call_id": "call_abc123",     # Must match the function_call's call_id
    "output": "string result"      # Always a string (or array of content objects)
}
```

### Key fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | Always `"function_call_output"` |
| `call_id` | string | ✅ | Must match the `call_id` from the model's `function_call` |
| `output` | string or array | ✅ | The result — typically a JSON string |

### Complete example

```python
from openai import OpenAI
import json

client = OpenAI()

# Define a tool
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
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# Step 1: Send the initial request
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

# Step 2: Collect the model's output (includes function_call items)
input_messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
input_messages += response.output  # Append all output items (including reasoning)

# Step 3: Execute each function call and append results
for item in response.output:
    if item.type == "function_call":
        # Execute the function
        result = get_weather(**json.loads(item.arguments))
        
        # Append the result in the required format
        input_messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,       # Links result to the specific call
            "output": json.dumps(result)    # Must be a string
        })

# Step 4: Send results back for the final response
final_response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)

print(final_response.output_text)
```

**Output:**
```
The current weather in Tokyo is 18°C and partly cloudy.
```

### Important OpenAI details

> **Note:** For reasoning models like GPT-5 or o4-mini, any **reasoning items** returned in the model's response must also be passed back alongside tool call outputs. The `response.output` list may contain both `reasoning` and `function_call` items — append them all.

```python
# Correct: append ALL output items, then add function_call_output items
input_messages += response.output  # Includes reasoning + function_call items

# Then append results for each function_call
for item in response.output:
    if item.type == "function_call":
        input_messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(execute_function(item.name, item.arguments))
        })
```

### Output as content array (images and files)

OpenAI also accepts an array of content objects instead of a plain string, useful for returning images or files:

```python
# Return an image from a function
input_messages.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": [
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/chart.png"
            }
        }
    ]
})
```

---

## Anthropic: `tool_result` content block

Anthropic's Messages API uses `tool_result` content blocks within a `user` role message. The key difference from OpenAI: results go inside a message's `content` array, not as standalone items.

### The format

```python
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_abc123",  # Must match tool_use block's id
            "content": "string result"       # String or array of content blocks
        }
    ]
}
```

### Key fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | Always `"tool_result"` |
| `tool_use_id` | string | ✅ | Must match the `id` from the model's `tool_use` block |
| `content` | string or array | ✅ | The result data |
| `is_error` | boolean | ❌ | Set to `true` to indicate an error result |

### Complete example

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'"
                }
            },
            "required": ["location"]
        }
    }
]

# Step 1: Send the initial request
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=messages,
)

# Step 2: Check if the model wants to use a tool
if response.stop_reason == "tool_use":
    # Find the tool_use block in the response
    tool_use_block = next(
        block for block in response.content
        if block.type == "tool_use"
    )
    
    # Step 3: Execute the function
    result = get_weather(**tool_use_block.input)
    
    # Step 4: Send the result back
    # Must include the assistant's response AND the tool result
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,   # Links to the tool_use block
                "content": json.dumps(result)         # The result as a string
            }
        ]
    })
    
    # Step 5: Get the final response
    final_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    
    print(final_response.content[0].text)
```

**Output:**
```
The weather in Tokyo is currently 18°C and partly cloudy.
```

### The `is_error` flag

Anthropic uniquely provides an `is_error` boolean field. When set to `true`, the model knows the function failed and adjusts its response accordingly — for example, apologizing for the issue or suggesting alternatives:

```python
# Returning an error result
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "is_error": True,
            "content": "Error: Location 'Atlantis' not found. Please provide a valid city name."
        }
    ]
})
```

### Rich content in results

Anthropic supports an array of content blocks in the result, including text and images:

```python
# Return text + image in a tool result
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"temperature": 18, "condition": "partly cloudy"})
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_encoded_chart
                    }
                }
            ]
        }
    ]
})
```

### Conversation structure detail

Anthropic requires strict message alternation: `user` → `assistant` → `user` → `assistant`. The tool result message must be a `user` message, and you must include the assistant's tool_use response before it:

```python
# Correct conversation structure
messages = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": response.content},      # Contains tool_use block
    {"role": "user", "content": [{"type": "tool_result", ...}]}  # Tool result
]
```

---

## Gemini: `Part.from_function_response()`

Google Gemini uses `Part.from_function_response()` to create function response parts. Unlike OpenAI and Anthropic, Gemini identifies results by **function name** (not a call ID) and accepts a Python **dictionary** directly (not a JSON string).

### The format

```python
from google.genai import types

function_response_part = types.Part.from_function_response(
    name="get_weather",           # Must match the function_call name
    response={"result": result}   # Dict — not a JSON string
)
```

### Key fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Must match the function call's name |
| `response` | dict | ✅ | The result as a Python dictionary |

### Complete example

```python
from google import genai
from google.genai import types

client = genai.Client()

# Define tools
tools = types.Tool(function_declarations=[{
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country"
            }
        },
        "required": ["location"]
    }
}])

config = types.GenerateContentConfig(tools=[tools])

# Step 1: Build the conversation
contents = [
    types.Content(
        role="user",
        parts=[types.Part(text="What's the weather in Tokyo?")]
    )
]

# Step 2: Get the model's response (contains function_call)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=config,
)

# Step 3: Extract and execute the function call
tool_call = response.candidates[0].content.parts[0].function_call
result = get_weather(**tool_call.args)

# Step 4: Create the function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result}  # Wrap in a dict — Gemini expects this
)

# Step 5: Append the model's response AND the function result
contents.append(response.candidates[0].content)  # Model's function_call turn
contents.append(
    types.Content(role="user", parts=[function_response_part])
)

# Step 6: Get the final response
final_response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=config,
)

print(final_response.text)
```

**Output:**
```
The current weather in Tokyo is 18°C with partly cloudy skies.
```

### Important Gemini details

> **Warning:** Gemini identifies function results by **name**, not by a unique call ID. If the model calls the same function twice in one turn (e.g., `get_weather` for two cities), you must return results in the **same order** the calls were made. Include all function response parts in a single `Content` object.

```python
# Multiple function results — maintain order
function_responses = []
for fn_call in response.function_calls:
    result = execute_function(fn_call.name, fn_call.args)
    function_responses.append(
        types.Part.from_function_response(
            name=fn_call.name,
            response={"result": result}
        )
    )

# All responses in one Content message
contents.append(response.candidates[0].content)
contents.append(types.Content(role="user", parts=function_responses))
```

### Gemini 3 thought signatures

When using Gemini 3 models, always send back the complete `response.candidates[0].content` — it may contain `thought_signature` data that the model needs to maintain context. The SDK handles this automatically, but if you're manipulating conversation history manually, never strip or merge parts that contain thought signatures:

```python
# Correct: append the complete model response content
contents.append(response.candidates[0].content)

# Wrong: manually reconstructing parts (may lose thought signatures)
# contents.append(types.Content(role="model", parts=[some_part]))
```

---

## Side-by-side comparison

Here's the same function result formatted for all three providers:

```python
import json

# The raw function result
weather_result = {
    "location": "Tokyo",
    "temperature": 18,
    "units": "celsius",
    "condition": "partly cloudy"
}

# --- OpenAI ---
openai_result = {
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": json.dumps(weather_result)   # String
}

# --- Anthropic ---
anthropic_result = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_abc123",
            "content": json.dumps(weather_result)  # String
        }
    ]
}

# --- Gemini ---
from google.genai import types
gemini_result = types.Part.from_function_response(
    name="get_weather",
    response={"result": weather_result}     # Dict (not string!)
)
```

---

## Unified result formatting layer

In production, you'll likely support multiple providers. Here's a unified formatter:

```python
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class FunctionResult:
    """Provider-agnostic function result."""
    call_id: str          # call_id (OpenAI), tool_use_id (Anthropic), name (Gemini)
    function_name: str
    result: Any           # The raw result from function execution
    is_error: bool = False


def format_result(result: FunctionResult, provider: Provider) -> dict:
    """Format a function result for a specific provider."""
    
    # Serialize the result to a string (for OpenAI and Anthropic)
    output_str = json.dumps(result.result) if not isinstance(result.result, str) else result.result
    
    if provider == Provider.OPENAI:
        return {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": output_str
        }
    
    elif provider == Provider.ANTHROPIC:
        tool_result = {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": output_str
        }
        if result.is_error:
            tool_result["is_error"] = True
        return tool_result
    
    elif provider == Provider.GEMINI:
        from google.genai import types
        response_data = result.result if isinstance(result.result, dict) else {"result": result.result}
        return types.Part.from_function_response(
            name=result.function_name,
            response=response_data
        )
    
    raise ValueError(f"Unknown provider: {provider}")
```

**Usage:**
```python
# Create a provider-agnostic result
func_result = FunctionResult(
    call_id="call_abc123",
    function_name="get_weather",
    result={"temperature": 18, "condition": "partly cloudy"},
    is_error=False
)

# Format for any provider
openai_formatted = format_result(func_result, Provider.OPENAI)
anthropic_formatted = format_result(func_result, Provider.ANTHROPIC)
gemini_formatted = format_result(func_result, Provider.GEMINI)
```

**Output (OpenAI format):**
```python
{
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": '{"temperature": 18, "condition": "partly cloudy"}'
}
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Always match the call ID exactly | Mismatched IDs cause API errors — copy directly from the model's response |
| Stringify results for OpenAI/Anthropic | Both expect strings; passing dicts causes type errors |
| Keep Gemini results as dicts | Gemini's `from_function_response` expects a Python dictionary |
| Include all output items (OpenAI) | Reasoning items must be passed back with function results |
| Maintain message alternation (Anthropic) | `user` → `assistant` → `user` — breaking this causes API errors |
| Preserve thought signatures (Gemini 3) | Append complete `response.candidates[0].content` — don't reconstruct parts |
| Return results in call order (parallel) | Especially important for Gemini which uses names, not unique IDs |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Passing a dict to OpenAI's `output` field | Use `json.dumps()` to convert to string first |
| Forgetting the `role: "user"` wrapper for Anthropic | Tool results must be inside a `user` message |
| Using `call_id` for Anthropic (it's `tool_use_id`) | Each provider uses different ID field names |
| Stringifying Gemini results | Pass a Python dict, not a JSON string |
| Skipping the assistant message before Anthropic tool results | Must include `{"role": "assistant", "content": response.content}` first |
| Stripping reasoning items from OpenAI output | Append all `response.output` items, then add `function_call_output` items |
| Merging Gemini parts that contain thought signatures | Append `response.candidates[0].content` as-is |

---

## Hands-on exercise

### Your task

Build a function that takes a raw Python function result and formats it correctly for all three providers. Test it with a weather function that returns a dictionary.

### Requirements

1. Create a `format_for_provider(result, call_id, function_name, provider)` function
2. Handle both successful results and error results
3. For OpenAI: return a `function_call_output` dict with stringified output
4. For Anthropic: return a complete `user` message with `tool_result` block (include `is_error` for errors)
5. For Gemini: return a `Part.from_function_response()` (mock with a dict for testing)
6. Test with both a successful weather result and an error result

### Expected result

```python
# Successful result
format_for_provider(
    result={"temp": 22, "condition": "sunny"},
    call_id="call_123",
    function_name="get_weather",
    provider="openai"
)
# → {"type": "function_call_output", "call_id": "call_123", "output": '{"temp": 22, ...}'}

# Error result
format_for_provider(
    result="City not found",
    call_id="toolu_456",
    function_name="get_weather",
    provider="anthropic",
    is_error=True
)
# → {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_456", 
#     "is_error": true, "content": "City not found"}]}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `json.dumps()` for OpenAI and Anthropic when the result is not already a string
- Remember that Anthropic's result goes inside a `{"role": "user", "content": [...]}` wrapper
- Gemini's format uses `name` instead of `call_id`
- Handle the `is_error` flag — only Anthropic supports it natively, but you can encode errors in the output string for OpenAI and Gemini too

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from typing import Any


def format_for_provider(
    result: Any,
    call_id: str,
    function_name: str,
    provider: str,
    is_error: bool = False
) -> dict:
    """Format a function result for a specific AI provider.
    
    Args:
        result: The raw function result (dict, string, list, etc.)
        call_id: The call ID from the model's function call
        function_name: The name of the function that was called
        provider: One of 'openai', 'anthropic', 'gemini'
        is_error: Whether the result represents an error
    
    Returns:
        A properly formatted result dict for the specified provider
    """
    # Stringify the result for providers that need it
    output_str = json.dumps(result) if not isinstance(result, str) else result
    
    if provider == "openai":
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_str
        }
    
    elif provider == "anthropic":
        tool_result = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": output_str
        }
        if is_error:
            tool_result["is_error"] = True
        
        return {
            "role": "user",
            "content": [tool_result]
        }
    
    elif provider == "gemini":
        # Gemini accepts dicts directly
        if isinstance(result, dict):
            response_data = {"result": result}
        else:
            response_data = {"result": result}
        
        # In production: return types.Part.from_function_response(...)
        # For testing, we return the equivalent dict
        return {
            "provider": "gemini",
            "function_response": {
                "name": function_name,
                "response": response_data
            }
        }
    
    raise ValueError(f"Unknown provider: {provider}")


# Test with successful result
weather = {"temp": 22, "condition": "sunny", "location": "Paris"}

for provider in ["openai", "anthropic", "gemini"]:
    formatted = format_for_provider(
        result=weather,
        call_id="call_123",
        function_name="get_weather",
        provider=provider
    )
    print(f"\n{provider.upper()}:")
    print(json.dumps(formatted, indent=2))

# Test with error result
for provider in ["openai", "anthropic", "gemini"]:
    formatted = format_for_provider(
        result="City 'Atlantis' not found",
        call_id="call_456",
        function_name="get_weather",
        provider=provider,
        is_error=True
    )
    print(f"\n{provider.upper()} (error):")
    print(json.dumps(formatted, indent=2))
```

**Output:**
```
OPENAI:
{
  "type": "function_call_output",
  "call_id": "call_123",
  "output": "{\"temp\": 22, \"condition\": \"sunny\", \"location\": \"Paris\"}"
}

ANTHROPIC:
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_123",
      "content": "{\"temp\": 22, \"condition\": \"sunny\", \"location\": \"Paris\"}"
    }
  ]
}

GEMINI:
{
  "provider": "gemini",
  "function_response": {
    "name": "get_weather",
    "response": {"result": {"temp": 22, "condition": "sunny", "location": "Paris"}}
  }
}

OPENAI (error):
{
  "type": "function_call_output",
  "call_id": "call_456",
  "output": "City 'Atlantis' not found"
}

ANTHROPIC (error):
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_456",
      "is_error": true,
      "content": "City 'Atlantis' not found"
    }
  ]
}

GEMINI (error):
{
  "provider": "gemini",
  "function_response": {
    "name": "get_weather",
    "response": {"result": "City 'Atlantis' not found"}
  }
}
```

</details>

### Bonus challenges

- [ ] Add support for returning image results in OpenAI's array format
- [ ] Handle parallel function calls — format multiple results at once
- [ ] Add validation that `call_id` is not empty or None before formatting

---

## Summary

✅ OpenAI uses `function_call_output` items with `call_id` — results must be JSON strings

✅ Anthropic uses `tool_result` content blocks with `tool_use_id` inside `user` messages — supports `is_error` flag

✅ Gemini uses `Part.from_function_response()` with function `name` — accepts Python dicts directly

✅ Always match the call identifier exactly — mismatched IDs cause API rejections

✅ Include all model output items (reasoning, function calls) when sending results back

✅ Preserve Gemini 3 thought signatures by appending the complete response content

**Next:** [Stringifying Results →](./02-stringifying-results.md) — Converting complex Python objects into model-friendly strings

---

[← Previous: Lesson Overview](./00-returning-results.md) | [Back to Lesson Overview](./00-returning-results.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API Reference: https://platform.openai.com/docs/api-reference/responses/create
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Gemini Function Calling Tutorial: https://ai.google.dev/gemini-api/docs/function-calling
-->
