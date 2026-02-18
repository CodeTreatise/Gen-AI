---
title: "Required vs. Optional Parameters"
---

# Required vs. Optional Parameters

## Introduction

Not every function parameter is mandatory. A search function needs a query but doesn't always need a page number. A weather function needs a city but might not need a temperature unit. How you express "required" and "optional" in your schema directly affects whether the model sends the right arguments — and how your code handles what it receives.

This lesson covers the `required` keyword, the difference between required and optional parameters across providers, the null union pattern for strict mode, and why `default` values don't work the way you might expect.

### What we'll cover

- How the `required` array works in JSON Schema
- Required vs. optional behavior across providers
- The null union pattern for strict mode optional fields
- Default values: what works and what doesn't
- Handling optional parameters in function implementations

### Prerequisites

- JSON Schema basics ([Lesson 01](./01-json-schema-basics.md))
- Strict mode requirements ([Lesson 02](./02-strict-mode-requirements.md))
- Property types ([Lesson 03](./03-property-types.md))

---

## The `required` array

The `required` keyword is an array of strings that lists which properties the model must always include. Properties not in `required` are optional — the model may or may not include them.

### How it works

```python
import json

# Properties not in "required" are optional
schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query text"
        },
        "category": {
            "type": "string",
            "enum": ["all", "books", "electronics", "clothing"],
            "description": "Category to search in"
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum number of results to return"
        },
        "sort_by": {
            "type": "string",
            "enum": ["relevance", "price", "rating", "newest"],
            "description": "How to sort results"
        }
    },
    "required": ["query"]  # Only query is required
}

# Valid outputs from the model:
valid_minimal = {"query": "wireless headphones"}
valid_partial = {"query": "wireless headphones", "category": "electronics"}
valid_full = {
    "query": "wireless headphones",
    "category": "electronics",
    "max_results": 20,
    "sort_by": "rating"
}

print("All valid outputs:")
print(f"  Minimal: {json.dumps(valid_minimal)}")
print(f"  Partial: {json.dumps(valid_partial)}")
print(f"  Full:    {json.dumps(valid_full)}")
```

**Output:**
```
All valid outputs:
  Minimal: {"query": "wireless headphones"}
  Partial: {"query": "wireless headphones", "category": "electronics"}
  Full:    {"query": "wireless headphones", "category": "electronics", "max_results": 20, "sort_by": "rating"}
```

### Rules for the `required` array

| Rule | Detail |
|------|--------|
| Must be an array of strings | `"required": ["name", "age"]` |
| Each string must match a property name | Must exist in `properties` |
| No duplicates | Each property name appears once |
| Can be empty | `"required": []` (all optional) |
| Order doesn't matter | `["age", "name"]` = `["name", "age"]` |

---

## Required behavior across providers

The three major providers handle required/optional parameters differently, especially in strict mode.

### Standard mode (all providers)

In standard mode, all three providers follow the same pattern: properties in `required` must be present; others are optional.

```python
import json

# Standard mode — same across providers
standard_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
        }
    },
    "required": ["location"]  # units is optional
}

# Model might generate either:
with_optional = {"location": "London", "units": "celsius"}
without_optional = {"location": "London"}

print("Standard mode behavior (all providers):")
print(f"  With optional:    {json.dumps(with_optional)}")
print(f"  Without optional: {json.dumps(without_optional)}")
print(f"  Both are valid ✅")
```

**Output:**
```
Standard mode behavior (all providers):
  With optional:    {"location": "London", "units": "celsius"}
  Without optional: {"location": "London"}
  Both are valid ✅
```

### Strict mode (OpenAI)

In OpenAI strict mode, every property must be in `required`. Optional parameters use the null union pattern.

```python
import json

# ❌ This FAILS in OpenAI strict mode
fails_strict = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                  "description": "Temperature unit"}
    },
    "required": ["location"],  # ← "units" not listed = ERROR
    "additionalProperties": False
}

# ✅ This WORKS in OpenAI strict mode
works_strict = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name"},
        "units": {
            "type": ["string", "null"],
            "enum": ["celsius", "fahrenheit", None],
            "description": "Temperature unit, or null for default (celsius)"
        }
    },
    "required": ["location", "units"],  # ← Both listed
    "additionalProperties": False
}

# In strict mode, model always generates both properties:
strict_with_value = {"location": "London", "units": "celsius"}
strict_with_null = {"location": "London", "units": None}

print("OpenAI strict mode:")
print(f"  With value: {json.dumps(strict_with_value)}")
print(f"  With null:  {json.dumps(strict_with_null)}")
```

**Output:**
```
OpenAI strict mode:
  With value: {"location": "London", "units": "celsius"}
  With null:  {"location": "London", "units": null}
```

### Anthropic and Gemini

Anthropic and Gemini don't require all properties in `required`, even with their strict/validated modes. They handle optional properties natively.

```python
import json

# Anthropic — standard required behavior, even with strict: true
anthropic_tool = {
    "name": "get_weather",
    "description": "Get weather for a location",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                      "description": "Temperature unit"}
        },
        "required": ["location"]  # ← Only location required, works fine
    }
}

# Gemini — same standard behavior
gemini_function = {
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                      "description": "Temperature unit"}
        },
        "required": ["location"]  # ← Standard required behavior
    }
}

print("Anthropic & Gemini: standard required/optional behavior")
print("No need for null unions — just omit from required array")
```

**Output:**
```
Anthropic & Gemini: standard required/optional behavior
No need for null unions — just omit from required array
```

---

## The null union pattern in depth

The null union pattern is the standard way to express "optional" in OpenAI strict mode. Let's look at all the variations.

### Basic null union

```python
import json

# String that can be null
nullable_string = {
    "type": ["string", "null"],
    "description": "Optional value, or null if not provided"
}

# Integer that can be null
nullable_integer = {
    "type": ["integer", "null"],
    "description": "Optional count, or null if not applicable"
}

# Boolean that can be null
nullable_boolean = {
    "type": ["boolean", "null"],
    "description": "Optional flag, or null if no preference"
}

# Number that can be null
nullable_number = {
    "type": ["number", "null"],
    "description": "Optional amount, or null if not specified"
}

types_map = {
    "nullable_string": nullable_string,
    "nullable_integer": nullable_integer,
    "nullable_boolean": nullable_boolean,
    "nullable_number": nullable_number
}

for name, schema in types_map.items():
    print(f"{name}: type = {schema['type']}")
```

**Output:**
```
nullable_string: type = ['string', 'null']
nullable_integer: type = ['integer', 'null']
nullable_boolean: type = ['boolean', 'null']
nullable_number: type = ['number', 'null']
```

### Null union with enum

When a property has an `enum` and you want it to be nullable, add `None` (Python's `null`) to the enum array:

```python
import json

# ❌ WRONG: null not in enum list
wrong_nullable_enum = {
    "type": ["string", "null"],
    "enum": ["low", "medium", "high"],
    "description": "Priority level or null"
}
# The model can't generate null because null isn't in the enum!

# ✅ CORRECT: null included in enum list
correct_nullable_enum = {
    "type": ["string", "null"],
    "enum": ["low", "medium", "high", None],
    "description": "Priority level, or null for default"
}

print("Wrong (null not in enum):", json.dumps(wrong_nullable_enum["enum"]))
print("Correct (null in enum):  ", json.dumps(correct_nullable_enum["enum"]))
```

**Output:**
```
Wrong (null not in enum): ["low", "medium", "high"]
Correct (null in enum):   ["low", "medium", "high", null]
```

### Null union with objects and arrays

Nullable objects and arrays follow the same pattern:

```python
import json

# Nullable object
nullable_object = {
    "type": ["object", "null"],
    "properties": {
        "street": {"type": "string", "description": "Street address"},
        "city": {"type": "string", "description": "City name"}
    },
    "required": ["street", "city"],
    "additionalProperties": False,
    "description": "Shipping address, or null if same as billing"
}

# Nullable array
nullable_array = {
    "type": ["array", "null"],
    "items": {"type": "string"},
    "description": "List of tags, or null if no tags"
}

print("Nullable object — model sends either an object or null")
print("Nullable array  — model sends either an array or null")
```

**Output:**
```
Nullable object — model sends either an object or null
Nullable array  — model sends either an array or null
```

---

## Default values

JSON Schema supports a `default` keyword, but it has limited use in function calling.

### Why `default` doesn't work in strict mode

```python
import json

# ❌ Default values are NOT enforced in strict mode
schema_with_default = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "limit": {
            "type": "integer",
            "default": 10,  # ← This is ignored in strict mode
            "description": "Number of results (default: 10)"
        }
    },
    "required": ["query"]
}

print("The 'default' keyword in JSON Schema:")
print("  • Is an ANNOTATION — not enforced by the schema")
print("  • NOT supported in OpenAI strict mode")
print("  • The model never sees or uses the default value")
print("  • Your function code must handle defaults")
```

**Output:**
```
The 'default' keyword in JSON Schema:
  • Is an ANNOTATION — not enforced by the schema
  • NOT supported in OpenAI strict mode
  • The model never sees or uses the default value
  • Your function code must handle defaults
```

### The right way to handle defaults

Instead of using the `default` keyword, document defaults in the description and handle them in your function:

```python
import json
from typing import Optional

# ✅ Document defaults in descriptions, handle in code
strict_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query text"
        },
        "limit": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 100,
            "description": "Max results to return (1-100). Null for default of 10."
        },
        "offset": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Number of results to skip. Null for default of 0."
        },
        "sort_by": {
            "type": ["string", "null"],
            "enum": ["relevance", "date", "popularity", None],
            "description": "Sort order. Null for default of 'relevance'."
        }
    },
    "required": ["query", "limit", "offset", "sort_by"],
    "additionalProperties": False
}

# Handle defaults in your function implementation
def search_products(
    query: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    sort_by: Optional[str] = None
) -> dict:
    """Search products with sensible defaults for null values."""
    actual_limit = limit if limit is not None else 10
    actual_offset = offset if offset is not None else 0
    actual_sort = sort_by if sort_by is not None else "relevance"

    return {
        "query": query,
        "limit": actual_limit,
        "offset": actual_offset,
        "sort_by": actual_sort
    }

# When the model sends nulls, defaults kick in
result = search_products(query="headphones", limit=None, offset=None, sort_by=None)
print("Model sent nulls, function applied defaults:")
print(json.dumps(result, indent=2))
```

**Output:**
```json
Model sent nulls, function applied defaults:
{
  "query": "headphones",
  "limit": 10,
  "offset": 0,
  "sort_by": "relevance"
}
```

---

## Cross-provider strategy

Here's a practical approach for writing schemas that work across providers:

```python
import json

def make_parameter(
    param_type: str,
    description: str,
    required: bool = True,
    strict_mode: bool = True,
    **kwargs
) -> dict:
    """Create a parameter schema that works across providers.
    
    For strict mode, optional parameters use null union.
    For standard mode, optional parameters are simply omitted from required.
    """
    schema = {"description": description}

    if strict_mode and not required:
        # Null union for strict mode optional fields
        schema["type"] = [param_type, "null"]
        # If there's an enum, add None to it
        if "enum" in kwargs:
            kwargs["enum"] = kwargs["enum"] + [None]
    else:
        schema["type"] = param_type

    schema.update(kwargs)
    return schema

# Build a schema using the helper
def build_schema(strict_mode: bool = True) -> dict:
    properties = {
        "city": make_parameter("string", "City name", required=True, strict_mode=strict_mode),
        "units": make_parameter("string", "Temperature unit or null for celsius",
                                required=False, strict_mode=strict_mode,
                                enum=["celsius", "fahrenheit"]),
        "days": make_parameter("integer", "Forecast days (1-14) or null for 3",
                               required=False, strict_mode=strict_mode,
                               minimum=1, maximum=14)
    }

    required_fields = ["city"]
    if strict_mode:
        required_fields = list(properties.keys())

    schema = {
        "type": "object",
        "properties": properties,
        "required": required_fields
    }

    if strict_mode:
        schema["additionalProperties"] = False

    return schema

# Compare strict vs standard
print("STRICT MODE schema:")
print(json.dumps(build_schema(strict_mode=True), indent=2))
print("\nSTANDARD MODE schema:")
print(json.dumps(build_schema(strict_mode=False), indent=2))
```

**Output:**
```json
STRICT MODE schema:
{
  "type": "object",
  "properties": {
    "city": { "type": "string", "description": "City name" },
    "units": {
      "type": ["string", "null"],
      "description": "Temperature unit or null for celsius",
      "enum": ["celsius", "fahrenheit", null]
    },
    "days": {
      "type": ["integer", "null"],
      "description": "Forecast days (1-14) or null for 3",
      "minimum": 1,
      "maximum": 14
    }
  },
  "required": ["city", "units", "days"],
  "additionalProperties": false
}

STANDARD MODE schema:
{
  "type": "object",
  "properties": {
    "city": { "type": "string", "description": "City name" },
    "units": {
      "type": "string",
      "description": "Temperature unit or null for celsius",
      "enum": ["celsius", "fahrenheit"]
    },
    "days": {
      "type": "integer",
      "description": "Forecast days (1-14) or null for 3",
      "minimum": 1,
      "maximum": 14
    }
  },
  "required": ["city"]
}
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Make essential parameters required | The model won't omit them |
| Use null unions for optional fields in strict mode | The only way to express "optional" |
| Document defaults in descriptions | The model (and developers) know what happens with null |
| Handle null values in function code | Always check `is not None` before using values |
| Keep required parameters minimal in standard mode | Let the model decide what's relevant |
| Use the same schema across providers with adaptation | Write once, adapt per provider |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Relying on `default` keyword | Document default in `description`, handle in code |
| Making everything required in standard mode | Only require what's truly necessary |
| Forgetting to add `null` to enum arrays | Always add `None` to enum when using null union |
| Not handling null in function implementation | Check `if value is not None` for every nullable field |
| Using different schemas per provider without testing | Test each provider-specific schema separately |
| Making all fields optional | Always have at least one required field |

---

## Hands-on exercise

### Your task

Write a function schema for `send_notification` that works in both strict and standard mode. The function sends a notification to a user.

Parameters:
1. `user_id` — required string
2. `message` — required string (1-500 characters)
3. `channel` — required, one of: "email", "sms", "push", "in_app"
4. `priority` — optional, one of: "low", "normal", "high"
5. `schedule_at` — optional date-time string
6. `data` — optional object with arbitrary key-value pairs

### Requirements

1. Write the strict mode version (OpenAI)
2. Write the standard mode version (Anthropic/Gemini)
3. Write the Python function that handles both versions (null vs. missing)

<details>
<summary>💡 Hints (click to expand)</summary>

- Strict mode: all 6 properties in `required`, null unions for optional ones
- Standard mode: only 3 properties in `required`
- In your Python function, use `kwargs.get("priority")` for standard mode
- For the `data` object in strict mode, you can't use `additionalProperties: true` — consider a different approach like a JSON string

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from typing import Optional

# STRICT MODE (OpenAI)
strict_schema = {
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "description": "User ID to send the notification to"
        },
        "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "Notification message, 1-500 characters"
        },
        "channel": {
            "type": "string",
            "enum": ["email", "sms", "push", "in_app"],
            "description": "Delivery channel for the notification"
        },
        "priority": {
            "type": ["string", "null"],
            "enum": ["low", "normal", "high", None],
            "description": "Priority level, or null for 'normal'"
        },
        "schedule_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "When to send (ISO 8601), or null for immediate"
        },
        "data_json": {
            "type": ["string", "null"],
            "description": "Additional data as a JSON string, or null if none"
        }
    },
    "required": ["user_id", "message", "channel", "priority", "schedule_at", "data_json"],
    "additionalProperties": False
}

# STANDARD MODE (Anthropic/Gemini)
standard_schema = {
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "description": "User ID to send the notification to"
        },
        "message": {
            "type": "string",
            "description": "Notification message, 1-500 characters"
        },
        "channel": {
            "type": "string",
            "enum": ["email", "sms", "push", "in_app"],
            "description": "Delivery channel for the notification"
        },
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high"],
            "description": "Priority level (default: 'normal')"
        },
        "schedule_at": {
            "type": "string",
            "format": "date-time",
            "description": "When to send (ISO 8601), omit for immediate"
        },
        "data": {
            "type": "object",
            "description": "Additional key-value data for the notification"
        }
    },
    "required": ["user_id", "message", "channel"]
}

# Python function that handles both
def send_notification(
    user_id: str,
    message: str,
    channel: str,
    priority: Optional[str] = None,
    schedule_at: Optional[str] = None,
    data: Optional[dict] = None,
    data_json: Optional[str] = None
) -> dict:
    """Send a notification. Handles both strict and standard mode inputs."""
    actual_priority = priority if priority is not None else "normal"
    actual_data = data
    if data_json is not None:
        actual_data = json.loads(data_json)

    return {
        "user_id": user_id,
        "message": message,
        "channel": channel,
        "priority": actual_priority,
        "schedule_at": schedule_at,
        "data": actual_data
    }

# Test with strict mode input (nulls)
strict_result = send_notification(
    user_id="usr_123",
    message="Your order shipped!",
    channel="push",
    priority=None,
    schedule_at=None,
    data_json=None
)
print("Strict mode result:", json.dumps(strict_result, indent=2))
```

</details>

---

## Summary

✅ The `required` array lists properties the model must always include — unlisted properties are optional

✅ In OpenAI strict mode, ALL properties must be in `required` — use null unions for optional fields

✅ Anthropic and Gemini support standard required/optional behavior even with schema validation

✅ `default` values are annotations, not enforced — handle defaults in your function code

✅ Document default values in `description` so the model knows what null means

**Next:** [Enums for Constrained Values](./05-enums-constrained-values.md) — Using enums to restrict parameter values and improve accuracy

---

## Further reading

- [JSON Schema Required Properties](https://json-schema.org/understanding-json-schema/reference/object#required-properties) — Official required keyword documentation
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Strict mode required rules
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — Anthropic's tool parameter handling

---

[← Previous: Property Types](./03-property-types.md) | [Next: Enums for Constrained Values →](./05-enums-constrained-values.md)

<!-- 
Sources Consulted:
- JSON Schema Reference (object): https://json-schema.org/understanding-json-schema/reference/object
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use/overview
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
