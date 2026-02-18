---
title: "Detecting Function Calls in Responses"
---

# Detecting Function Calls in Responses

## Introduction

When you send a request with tools defined, the model can respond in two ways: with a regular text message, or with one or more function call requests. Your code must reliably distinguish between these cases before doing anything else. A missed function call means your application ignores the model's request entirely. A false positive means you try to parse text as a function call and crash.

Each provider structures function call signals differently. OpenAI's Responses API uses an `output` array with typed items. Anthropic signals tool use through `stop_reason` and `tool_use` content blocks. Google Gemini embeds `function_call` objects in response parts. This lesson walks through detection logic for each provider with working code examples.

### What we'll cover

- Detecting function calls in OpenAI Responses API output
- Detecting function calls in OpenAI Chat Completions (legacy)
- Detecting tool use in Anthropic responses
- Detecting function calls in Google Gemini responses
- Handling multiple simultaneous function calls
- Building a unified detection layer

### Prerequisites

- Function calling concepts ([Lesson 01](../01-function-calling-concepts/00-function-calling-concepts.md))
- Defining functions ([Lesson 02](../02-defining-functions/00-defining-functions.md))
- JSON Schema for parameters ([Lesson 03](../03-json-schema-for-parameters/00-json-schema-for-parameters.md))

---

## OpenAI Responses API detection

The Responses API (OpenAI's current recommended interface) returns an `output` array containing different item types. Function calls appear as items with `type: "function_call"`.

### Basic detection

```python
from openai import OpenAI
import json

client = OpenAI()

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

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)

# Check each output item for function calls
function_calls = [
    item for item in response.output
    if item.type == "function_call"
]

if function_calls:
    print(f"Model wants to call {len(function_calls)} function(s)")
    for call in function_calls:
        print(f"  Function: {call.name}")
        print(f"  Call ID:  {call.call_id}")
        print(f"  Args:     {call.arguments}")
else:
    # Regular text response
    print(f"Text response: {response.output_text}")
```

**Output:**
```
Model wants to call 1 function(s)
  Function: get_weather
  Call ID:  call_12345xyz
  Args:     {"location":"Paris, France"}
```

### What the output array looks like

The `output` array can contain different item types mixed together:

```python
# response.output might contain:
[
    {
        "type": "reasoning",       # Reasoning items (GPT-5, o-series)
        "id": "rs_abc123",
        "summary": [...]
    },
    {
        "type": "function_call",   # A function call request
        "id": "fc_12345xyz",
        "call_id": "call_12345xyz",
        "name": "get_weather",
        "arguments": "{\"location\":\"Paris, France\"}"
    },
    {
        "type": "message",         # A text message
        "id": "msg_xyz789",
        "role": "assistant",
        "content": [...]
    }
]
```

> **Note:** The `output` array can contain reasoning items, function calls, and messages in any combination. Always filter by `type` rather than assuming a specific order or count.

### Detection helper function

```python
from dataclasses import dataclass


@dataclass
class DetectedCall:
    """Normalized representation of a detected function call."""
    name: str
    call_id: str
    raw_arguments: str  # JSON string for OpenAI


def detect_openai_function_calls(response) -> list[DetectedCall]:
    """Extract function calls from an OpenAI Responses API response."""
    calls = []
    for item in response.output:
        if item.type == "function_call":
            calls.append(DetectedCall(
                name=item.name,
                call_id=item.call_id,
                raw_arguments=item.arguments,
            ))
    return calls


# Usage
calls = detect_openai_function_calls(response)
print(f"Detected {len(calls)} function call(s)")
```

**Output:**
```
Detected 1 function call(s)
```

---

## OpenAI Chat Completions detection (legacy)

The older Chat Completions API uses a different structure. Function calls appear in the `tool_calls` array on the assistant message, and the `finish_reason` is `"tool_calls"`.

```python
# Chat Completions format (legacy — still supported)
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
)

message = response.choices[0].message

# Detection: check finish_reason or tool_calls
if response.choices[0].finish_reason == "tool_calls":
    print("Model wants to call tools")
    for tool_call in message.tool_calls:
        print(f"  Function: {tool_call.function.name}")
        print(f"  Call ID:  {tool_call.id}")
        print(f"  Args:     {tool_call.function.arguments}")
else:
    print(f"Text response: {message.content}")
```

**Output:**
```
Model wants to call tools
  Function: get_weather
  Call ID:  call_abc123
  Args:     {"location":"Paris, France"}
```

### Key differences from Responses API

| Aspect | Responses API | Chat Completions |
|--------|--------------|------------------|
| Where calls live | `response.output` items | `message.tool_calls` array |
| Type indicator | `item.type == "function_call"` | `finish_reason == "tool_calls"` |
| Call ID field | `item.call_id` | `tool_call.id` |
| Function name | `item.name` | `tool_call.function.name` |
| Arguments | `item.arguments` | `tool_call.function.arguments` |
| Multiple calls | Multiple items in `output` | Multiple entries in `tool_calls` |

> **Warning:** OpenAI recommends migrating to the Responses API for new applications. Chat Completions is still supported but the Responses API provides better intelligence and features like reasoning model support.

---

## Anthropic detection

Anthropic signals tool use through two mechanisms: the response `stop_reason` is `"tool_use"`, and the response `content` array contains `tool_use` blocks.

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
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
    }],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

# Detection method 1: check stop_reason
if response.stop_reason == "tool_use":
    print("Claude wants to use tools")

# Detection method 2: filter content blocks by type
tool_calls = [
    block for block in response.content
    if block.type == "tool_use"
]

if tool_calls:
    for call in tool_calls:
        print(f"  Tool:   {call.name}")
        print(f"  ID:     {call.id}")
        print(f"  Input:  {call.input}")  # Already a Python dict!
```

**Output:**
```
Claude wants to use tools
  Tool:   get_weather
  ID:     toolu_01A2B3C4D5
  Input:  {'location': 'Paris, France'}
```

### Anthropic content block structure

Claude's response `content` is an array that can mix text and tool use blocks:

```python
# response.content might contain:
[
    {
        "type": "text",
        "text": "I'll check the weather for you."
    },
    {
        "type": "tool_use",
        "id": "toolu_01A2B3C4D5",
        "name": "get_weather",
        "input": {"location": "Paris, France"}  # Already parsed!
    }
]
```

> **Important:** Claude often includes a text block *before* the tool use block explaining what it's about to do. Your detection logic should look for `tool_use` blocks specifically, not just check whether `content` is non-empty.

### Detection helper function

```python
@dataclass
class DetectedCall:
    """Normalized representation of a detected function call."""
    name: str
    call_id: str
    raw_arguments: dict  # Already a dict for Anthropic


def detect_anthropic_function_calls(response) -> list[DetectedCall]:
    """Extract function calls from an Anthropic Messages API response."""
    calls = []
    for block in response.content:
        if block.type == "tool_use":
            calls.append(DetectedCall(
                name=block.name,
                call_id=block.id,
                raw_arguments=block.input,
            ))
    return calls
```

---

## Google Gemini detection

Gemini embeds function calls as `function_call` objects within the response `parts`. The SDK provides a convenient `function_calls` property on the response.

```python
from google import genai
from google.genai import types

client = genai.Client()

tools = types.Tool(function_declarations=[{
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
    }
}])
config = types.GenerateContentConfig(tools=[tools])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Paris?",
    config=config,
)

# Detection method 1: use the SDK's function_calls property
if response.function_calls:
    for call in response.function_calls:
        print(f"  Function: {call.name}")
        print(f"  Args:     {call.args}")  # Already a dict

# Detection method 2: check parts manually
for part in response.candidates[0].content.parts:
    if part.function_call:
        fc = part.function_call
        print(f"  Function: {fc.name}")
        print(f"  Args:     {fc.args}")
```

**Output:**
```
  Function: get_weather
  Args:     {'location': 'Paris, France'}
```

### Gemini response structure

```python
# response.candidates[0].content.parts might contain:
[
    Part(
        function_call=FunctionCall(
            name="get_weather",
            args={"location": "Paris, France"}  # Already a dict
        )
    )
]
```

> **Note:** Gemini does not assign explicit call IDs like OpenAI and Anthropic. When returning function results, you reference the function by name rather than by ID. This matters when handling parallel calls to the same function — you match results to calls by position.

### Detection helper function

```python
def detect_gemini_function_calls(response) -> list[DetectedCall]:
    """Extract function calls from a Gemini response."""
    calls = []
    if response.function_calls:
        for idx, call in enumerate(response.function_calls):
            calls.append(DetectedCall(
                name=call.name,
                call_id=f"gemini_call_{idx}",  # Synthetic ID
                raw_arguments=dict(call.args) if call.args else {},
            ))
    return calls
```

---

## Detecting multiple simultaneous calls

All three providers support the model making multiple function calls in a single response. This is called **parallel function calling** and it's critical to handle correctly.

### OpenAI — multiple output items

```python
response = client.responses.create(
    model="gpt-4.1",
    input=[{
        "role": "user",
        "content": "What's the weather in Paris and Tokyo?"
    }],
    tools=tools,
)

# Filter all function_call items
function_calls = [
    item for item in response.output
    if item.type == "function_call"
]

print(f"Number of calls: {len(function_calls)}")
for call in function_calls:
    print(f"  {call.name}({call.arguments})")
```

**Output:**
```
Number of calls: 2
  get_weather({"location":"Paris, France"})
  get_weather({"location":"Tokyo, Japan"})
```

### Anthropic — multiple tool_use blocks

```python
# Claude can return multiple tool_use blocks
tool_calls = [
    block for block in response.content
    if block.type == "tool_use"
]

# Each has its own unique ID
for call in tool_calls:
    print(f"  {call.name} (id: {call.id})")
```

**Output:**
```
  get_weather (id: toolu_01A2B3C4D5)
  get_weather (id: toolu_06E7F8G9H0)
```

### Gemini — multiple function_call parts

```python
# Gemini returns multiple function_call parts
if response.function_calls:
    print(f"Number of calls: {len(response.function_calls)}")
    for call in response.function_calls:
        print(f"  {call.name}({call.args})")
```

**Output:**
```
Number of calls: 2
  get_weather({'location': 'Paris, France'})
  get_weather({'location': 'Tokyo, Japan'})
```

> **Warning:** Never assume the model will make exactly one function call. Always iterate over the full list. Even if your prompt seems to require only one call, the model might split the request or add additional calls.

---

## Building a unified detection layer

In production systems that support multiple providers, a unified detection interface simplifies downstream code. Here's a pattern that normalizes function call detection across all three providers:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class FunctionCallRequest:
    """Provider-agnostic representation of a function call."""
    provider: Provider
    name: str
    call_id: str
    arguments: Any       # dict for Anthropic/Gemini, JSON string for OpenAI
    raw_item: Any        # Original provider-specific object


def detect_function_calls(
    provider: Provider,
    response: Any,
) -> list[FunctionCallRequest]:
    """Detect function calls in a response from any supported provider."""
    
    if provider == Provider.OPENAI:
        return [
            FunctionCallRequest(
                provider=Provider.OPENAI,
                name=item.name,
                call_id=item.call_id,
                arguments=item.arguments,
                raw_item=item,
            )
            for item in response.output
            if item.type == "function_call"
        ]
    
    elif provider == Provider.ANTHROPIC:
        return [
            FunctionCallRequest(
                provider=Provider.ANTHROPIC,
                name=block.name,
                call_id=block.id,
                arguments=block.input,
                raw_item=block,
            )
            for block in response.content
            if block.type == "tool_use"
        ]
    
    elif provider == Provider.GEMINI:
        calls = []
        if response.function_calls:
            for idx, call in enumerate(response.function_calls):
                calls.append(FunctionCallRequest(
                    provider=Provider.GEMINI,
                    name=call.name,
                    call_id=f"gemini_{idx}",
                    arguments=dict(call.args) if call.args else {},
                    raw_item=call,
                ))
        return calls
    
    raise ValueError(f"Unsupported provider: {provider}")


# Usage
calls = detect_function_calls(Provider.OPENAI, response)
if calls:
    print(f"Detected {len(calls)} function call(s)")
    for call in calls:
        print(f"  [{call.provider.value}] {call.name} → {call.call_id}")
```

**Output:**
```
Detected 2 function call(s)
  [openai] get_weather → call_12345xyz
  [openai] get_weather → call_67890abc
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Always iterate over all calls | The model can return zero, one, or many function calls |
| Filter by type, not position | Function calls can appear at any position in the output array |
| Keep the original response object | You'll need it for reasoning items and context management |
| Check for mixed content | Anthropic often includes text *and* tool_use blocks together |
| Handle the "no calls" case | Sometimes the model answers directly without calling functions |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Checking only `response.output[0]` | Iterate over all items: `[i for i in response.output if i.type == "function_call"]` |
| Assuming `stop_reason == "tool_use"` means exactly one call | Anthropic can return multiple `tool_use` blocks with one stop reason |
| Treating Gemini's missing call ID as an error | Gemini uses function names and position instead of explicit IDs |
| Ignoring text blocks in Anthropic responses | Claude often explains what it's doing before calling a tool |
| Hard-coding for one provider | Use a detection layer that normalizes across providers |

---

## Hands-on exercise

### Your task

Build a function `analyze_response` that takes a provider name and a mock response object, detects any function calls, and returns a summary report.

### Requirements

1. Accept `"openai"`, `"anthropic"`, or `"gemini"` as the provider
2. Return a dict with `has_calls` (bool), `call_count` (int), and `calls` (list of dicts with `name` and `call_id`)
3. Handle the case where the response contains no function calls
4. Handle mixed content (text + function calls)

### Expected result

```python
result = analyze_response("openai", mock_openai_response)
# {'has_calls': True, 'call_count': 2, 'calls': [
#     {'name': 'get_weather', 'call_id': 'call_123'},
#     {'name': 'get_weather', 'call_id': 'call_456'}
# ]}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `detect_function_calls` pattern from the unified detection section
- Create mock response objects using `@dataclass` or `SimpleNamespace`
- Remember Gemini doesn't have call IDs — generate synthetic ones

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from types import SimpleNamespace


# Mock response objects for testing
def create_mock_openai_response():
    return SimpleNamespace(output=[
        SimpleNamespace(type="message", content="Checking weather..."),
        SimpleNamespace(
            type="function_call",
            name="get_weather",
            call_id="call_123",
            arguments='{"location": "Paris"}',
        ),
        SimpleNamespace(
            type="function_call",
            name="get_weather",
            call_id="call_456",
            arguments='{"location": "Tokyo"}',
        ),
    ])


def create_mock_anthropic_response():
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_abc",
                name="get_weather",
                input={"location": "Paris"},
            ),
        ],
    )


def create_mock_gemini_response():
    return SimpleNamespace(
        function_calls=[
            SimpleNamespace(name="get_weather", args={"location": "Paris"}),
        ]
    )


def analyze_response(provider: str, response) -> dict:
    """Analyze a response and return a function call summary."""
    calls = []

    if provider == "openai":
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                calls.append({
                    "name": item.name,
                    "call_id": item.call_id,
                })

    elif provider == "anthropic":
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                calls.append({
                    "name": block.name,
                    "call_id": block.id,
                })

    elif provider == "gemini":
        if response.function_calls:
            for idx, call in enumerate(response.function_calls):
                calls.append({
                    "name": call.name,
                    "call_id": f"gemini_{idx}",
                })

    return {
        "has_calls": len(calls) > 0,
        "call_count": len(calls),
        "calls": calls,
    }


# Test all providers
for provider, mock_fn in [
    ("openai", create_mock_openai_response),
    ("anthropic", create_mock_anthropic_response),
    ("gemini", create_mock_gemini_response),
]:
    result = analyze_response(provider, mock_fn())
    print(f"{provider}: {result}")
```

**Output:**
```
openai: {'has_calls': True, 'call_count': 2, 'calls': [{'name': 'get_weather', 'call_id': 'call_123'}, {'name': 'get_weather', 'call_id': 'call_456'}]}
anthropic: {'has_calls': True, 'call_count': 1, 'calls': [{'name': 'get_weather', 'call_id': 'toolu_abc'}]}
gemini: {'has_calls': True, 'call_count': 1, 'calls': [{'name': 'get_weather', 'call_id': 'gemini_0'}]}
```

</details>

### Bonus challenges

- [ ] Add support for OpenAI Chat Completions (legacy) format
- [ ] Handle edge cases: empty `output`, `None` content, missing attributes
- [ ] Add a `text_content` field to the result that captures any text the model included alongside function calls

---

## Summary

✅ OpenAI Responses API signals function calls via `type: "function_call"` items in the `output` array — filter by type, never assume position

✅ Anthropic signals tool use through `stop_reason: "tool_use"` and `tool_use` content blocks — always check for mixed text + tool blocks

✅ Gemini embeds `function_call` objects in response parts — use `response.function_calls` for convenience, and note that Gemini doesn't use explicit call IDs

✅ Always handle zero, one, or multiple function calls — never assume the count

✅ A unified detection layer normalizes provider differences and simplifies downstream processing

**Next:** [Provider Response Structures](./02-provider-response-structures.md) — Deep dive into the field-by-field differences between providers

---

[← Previous: Handling Function Calls Overview](./00-handling-function-calls.md) | [Next: Provider Response Structures →](./02-provider-response-structures.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API Reference: https://platform.openai.com/docs/api-reference/responses
- Anthropic Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
