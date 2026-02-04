---
title: "The Response Format Parameter"
---

# The Response Format Parameter

## Introduction

The `response_format` parameter is your control center for JSON output. It determines whether you get raw JSON, schema-enforced JSON, or plain text. Understanding this parameter is key to choosing the right output strategy.

### What We'll Cover

- The `response_format` parameter options
- `json_object` vs `json_schema` configuration
- Provider-specific implementations
- Migration from JSON mode to Structured Outputs
- Performance and latency considerations

### Prerequisites

- [JSON Mode in API Calls](./01-json-mode-api.md)
- [Structured Outputs with Schemas](./02-structured-outputs-schemas.md)

---

## Response Format Options

### OpenAI Options

| Type | Value | Schema Required | Guarantee |
|------|-------|-----------------|-----------|
| Text (default) | `{"type": "text"}` | No | None |
| JSON Mode | `{"type": "json_object"}` | No | Valid JSON |
| Structured | `{"type": "json_schema", ...}` | Yes | Schema compliance |

### Basic Configuration

```python
from openai import OpenAI
client = OpenAI()

# Default: Plain text
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
    # No response_format = text output
)

# JSON Mode
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Respond in JSON format."},
        {"role": "user", "content": "List three colors"}
    ],
    response_format={"type": "json_object"}
)

# Structured Outputs
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "List three colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "color_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"],
                "additionalProperties": False
            }
        }
    }
)
```

---

## JSON Object Configuration

### Minimal Setup

```python
response_format = {"type": "json_object"}
```

### Requirements

| Requirement | Description |
|-------------|-------------|
| Mention "JSON" | Must appear in system or user message |
| Model support | GPT-4 Turbo and later |
| Complete response | Set adequate `max_tokens` |

### Complete Example

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Output your response as JSON."
        },
        {
            "role": "user",
            "content": "What are the primary colors?"
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=500  # Ensure complete output
)

# Parse result
import json
data = json.loads(response.choices[0].message.content)
```

**Possible output (structure not guaranteed):**
```json
{
    "colors": ["red", "yellow", "blue"],
    "note": "These are the primary colors in traditional color theory"
}
```

---

## JSON Schema Configuration

### Full Structure

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "schema_name",         # Required: identifier
        "strict": True,                 # Recommended: enables constrained decoding
        "schema": {                     # Required: JSON Schema definition
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"}
            },
            "required": ["field1", "field2"],
            "additionalProperties": False
        }
    }
}
```

### Schema Name Requirements

| Rule | Example |
|------|---------|
| Must be provided | `"name": "my_schema"` |
| Alphanumeric + underscore | `my_schema_v1` ✅ |
| No spaces or special chars | `my-schema` ❌ |
| Max 64 characters | Keep concise |

### The `strict` Parameter

```python
# Strict mode (recommended)
"strict": True
# - Guarantees exact schema match
# - Model cannot produce non-compliant output
# - Requires all fields in "required"
# - Requires "additionalProperties": false

# Non-strict mode
"strict": False  # or omitted
# - Best-effort schema compliance
# - May produce extra fields
# - May omit optional fields
# - More flexible, less reliable
```

---

## Provider Comparison

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "strict": True,
            "schema": {...}
        }
    }
)
```

### Anthropic

Anthropic uses tool_use for structured output rather than `response_format`:

```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[
        {
            "name": "structured_output",
            "description": "Output data in structured format",
            "input_schema": {
                "type": "object",
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "number"}
                },
                "required": ["field1", "field2"]
            }
        }
    ],
    tool_choice={"type": "tool", "name": "structured_output"},
    messages=[...]
)

# Extract from tool use block
data = response.content[0].input
```

### Google Gemini

```python
import google.generativeai as genai
from pydantic import BaseModel

class MyOutput(BaseModel):
    field1: str
    field2: int

model = genai.GenerativeModel(
    "gemini-1.5-pro",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": MyOutput
    }
)

response = model.generate_content("...")
```

---

## SDK Convenience Methods

### OpenAI Responses API

```python
from pydantic import BaseModel

class ExtractedData(BaseModel):
    title: str
    items: list[str]

# Direct parse method
response = client.responses.parse(
    model="gpt-4o",
    input=[{"role": "user", "content": "..."}],
    text_format=ExtractedData  # Automatic schema conversion
)

data = response.output_parsed  # Already a Pydantic instance
```

### OpenAI Chat Completions Parse

```python
response = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=ExtractedData  # Pass class directly
)

data = response.choices[0].message.parsed
```

---

## Migrating from JSON Mode to Structured Outputs

### Before (JSON Mode)

```python
# JSON mode with prompt-based structure
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """Respond in JSON format:
{
    "name": "string",
    "score": number,
    "tags": ["string"]
}"""
        },
        {"role": "user", "content": "Analyze this product..."}
    ],
    response_format={"type": "json_object"}
)

# Manual parsing with error handling
import json
try:
    data = json.loads(response.choices[0].message.content)
    name = data.get("name", "Unknown")  # May be missing
    score = data.get("score", 0)        # May be wrong type
except json.JSONDecodeError:
    # Handle parse error
    pass
```

### After (Structured Outputs)

```python
from pydantic import BaseModel

class ProductAnalysis(BaseModel):
    name: str
    score: float
    tags: list[str]

response = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "Analyze this product..."}
    ],
    response_format=ProductAnalysis
)

# Direct access - guaranteed structure
data = response.choices[0].message.parsed
print(data.name)   # Always present
print(data.score)  # Always float
print(data.tags)   # Always list of strings
```

---

## Performance Considerations

### First Request Latency

When using a new schema for the first time:

```
First request:  ~500-1000ms extra (schema processing)
Subsequent:     Normal latency
```

> **Tip:** Warm up schemas in development to avoid first-request latency in production.

### Schema Complexity Impact

| Schema Feature | Performance Impact |
|----------------|-------------------|
| Simple flat object | Minimal |
| Nested objects | Slight increase |
| Large enums | Moderate increase |
| Recursive schemas | Higher |
| 100+ properties | Significant |

### Caching Behavior

OpenAI caches processed schemas. Same schema = same performance:

```python
# These use the same cached schema
schema1 = {"type": "object", "properties": {"a": {"type": "string"}}, ...}
schema2 = {"type": "object", "properties": {"a": {"type": "string"}}, ...}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Pydantic for schemas | Automatic conversion, type safety |
| Start with strict mode | Guarantee compliance from day one |
| Keep schemas focused | Better performance, clearer output |
| Add field descriptions | Help model understand intent |
| Set appropriate max_tokens | Prevent truncation |
| Cache schemas | Avoid processing overhead |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting "JSON" in messages | Required for json_object mode |
| Missing schema name | Always provide `"name"` field |
| Omitting `required` array | List all fields in strict mode |
| Low max_tokens | Set high enough for complete output |
| Using unsupported models | Check model compatibility |

---

## Hands-on Exercise

### Your Task

Configure `response_format` for an API that extracts event information.

### Requirements

1. Create schema for: event_name, date, location, attendee_count
2. Use `json_schema` type with strict mode
3. Handle both JSON mode and Structured Outputs
4. Compare the outputs

<details>
<summary>💡 Hints (click to expand)</summary>

- Schema name must be alphanumeric with underscores
- All fields go in `required` for strict mode
- Don't forget `additionalProperties: false`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
import json

client = OpenAI()

event_text = "Tech Conference 2025, March 15, San Francisco, expecting 500 attendees"

# Approach 1: JSON Mode (flexible but unreliable structure)
json_mode_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """Extract event info and respond in JSON format:
{
    "event_name": "string",
    "date": "string",
    "location": "string",
    "attendee_count": number
}"""
        },
        {"role": "user", "content": event_text}
    ],
    response_format={"type": "json_object"}
)

json_mode_data = json.loads(json_mode_response.choices[0].message.content)
print("JSON Mode:", json_mode_data)

# Approach 2: Structured Outputs (guaranteed structure)
structured_response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": f"Extract event info: {event_text}"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "event_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "event_name": {
                        "type": "string",
                        "description": "The name or title of the event"
                    },
                    "date": {
                        "type": "string",
                        "description": "Event date in any format"
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location or venue"
                    },
                    "attendee_count": {
                        "type": "integer",
                        "description": "Expected or actual number of attendees"
                    }
                },
                "required": ["event_name", "date", "location", "attendee_count"],
                "additionalProperties": False
            }
        }
    }
)

structured_data = json.loads(structured_response.choices[0].message.content)
print("Structured:", structured_data)

# Compare reliability
print("\n--- Comparison ---")
print(f"JSON Mode has 'event_name': {'event_name' in json_mode_data}")
print(f"Structured has 'event_name': {'event_name' in structured_data}")  # Always True
```

**Expected output:**
```
JSON Mode: {"event_name": "Tech Conference 2025", "date": "March 15", "location": "San Francisco", "attendee_count": 500}
Structured: {"event_name": "Tech Conference 2025", "date": "March 15", "location": "San Francisco", "attendee_count": 500}

--- Comparison ---
JSON Mode has 'event_name': True
Structured has 'event_name': True
```

</details>

### Bonus Challenge

- [ ] Add optional fields (website, description) using null union

---

## Summary

✅ `response_format` controls JSON output behavior

✅ `json_object` = valid JSON, no schema guarantee

✅ `json_schema` with `strict: true` = guaranteed compliance

✅ Use SDK parse methods for cleaner code

✅ Consider first-request latency for new schemas

**Next:** [Pydantic and Zod Schema Definitions](./04-pydantic-zod-schemas.md)

---

## Further Reading

- [OpenAI Response Format Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- OpenAI API Reference: https://platform.openai.com/docs/api-reference/chat/create
-->
