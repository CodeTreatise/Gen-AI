---
title: "Optional Parameters in Strict Mode"
---

# Optional Parameters in Strict Mode

## Introduction

Strict mode requires every property to appear in the `required` array — so how do you handle parameters that are genuinely optional? A weather function doesn't always need a `unit` preference. A search tool shouldn't demand a `category` filter every time. The answer is the **nullable union type pattern**: the field is always *present*, but its value can be `null`.

This lesson covers how to implement optional parameters within strict mode's constraints, with patterns and examples across all major providers.

### What we'll cover

- The nullable union type pattern (`["type", "null"]`)
- How "required but nullable" differs from "optional"
- Practical patterns for common optional parameter scenarios
- Provider-specific syntax differences
- Using Pydantic `Optional` and `None` defaults

### Prerequisites

- Understanding of strict mode schema requirements ([Sub-lesson 03](./03-schema-requirements.md))
- Familiarity with JSON Schema types

---

## The problem: required ≠ has-a-value

In standard JSON Schema, optional fields are simple: omit them from the `required` array.

```json
// Standard JSON Schema: "unit" is optional — it may or may not appear
{
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
}
```

But strict mode requires all properties in `required`. So how do we express "the model might not have a value for this field"?

The solution is to separate two concepts:

| Concept | Standard JSON Schema | Strict mode |
|---------|---------------------|-------------|
| **Field presence** | Optional (omit from `required`) | Always present (in `required`) |
| **Value presence** | Always has a value if present | Can be `null` (nullable type) |

---

## The nullable union type pattern

Instead of making a field optional (which strict mode forbids), we make the field *nullable*. The field is always present in the output, but its value can be `null` when the model has no information for it.

### Basic syntax

Use a type array that includes `"null"`:

```json
{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The location to get weather for"
        },
        "unit": {
            "type": ["string", "null"],
            "description": "Temperature unit preference. Null if not specified.",
            "enum": ["celsius", "fahrenheit"]
        }
    },
    "required": ["location", "unit"],
    "additionalProperties": false
}
```

Now the model can produce either of these valid outputs:

```json
// When the user specifies a unit preference
{"location": "Tokyo", "unit": "celsius"}

// When the user doesn't mention a unit
{"location": "Tokyo", "unit": null}
```

Both outputs are valid against the schema. The field is always present; only the value varies.

---

## Practical patterns

### Pattern 1: nullable string

Use when a text value is sometimes not applicable:

```json
{
    "name": {
        "type": "string",
        "description": "User's full name"
    },
    "nickname": {
        "type": ["string", "null"],
        "description": "User's preferred nickname. Null if not provided."
    }
}
```

**Possible outputs:**
```json
{"name": "Alice Smith", "nickname": "Ali"}
{"name": "Bob Johnson", "nickname": null}
```

### Pattern 2: nullable integer

Use when a numeric value is sometimes unknown:

```json
{
    "product_name": {
        "type": "string",
        "description": "Name of the product"
    },
    "max_price": {
        "type": ["integer", "null"],
        "description": "Maximum price filter in cents. Null for no price limit."
    }
}
```

**Possible outputs:**
```json
{"product_name": "laptop", "max_price": 150000}
{"product_name": "laptop", "max_price": null}
```

### Pattern 3: nullable enum

Use when a categorical selection is optional:

```json
{
    "query": {
        "type": "string",
        "description": "Search query"
    },
    "sort_order": {
        "type": ["string", "null"],
        "enum": ["relevance", "date", "price_asc", "price_desc"],
        "description": "Sort order for results. Null for default ordering."
    }
}
```

**Possible outputs:**
```json
{"query": "wireless headphones", "sort_order": "price_asc"}
{"query": "wireless headphones", "sort_order": null}
```

### Pattern 4: nullable object

Use when an entire group of related fields is optional:

```json
{
    "title": {
        "type": "string",
        "description": "Event title"
    },
    "location": {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "city": {"type": "string"}
                },
                "required": ["venue", "city"],
                "additionalProperties": false
            },
            {"type": "null"}
        ],
        "description": "Event location. Null for virtual events."
    }
}
```

**Possible outputs:**
```json
{"title": "Tech Conference", "location": {"venue": "Convention Center", "city": "Austin"}}
{"title": "Webinar", "location": null}
```

> **Note:** For nullable objects, use `anyOf` with the object schema and `{"type": "null"}` rather than the array syntax. This is because you can't set `additionalProperties` on a type array.

### Pattern 5: nullable array

Use when a list of values is optional:

```json
{
    "message": {
        "type": "string",
        "description": "The message content"
    },
    "tags": {
        "anyOf": [
            {
                "type": "array",
                "items": {"type": "string"}
            },
            {"type": "null"}
        ],
        "description": "Tags for the message. Null if no tags."
    }
}
```

**Possible outputs:**
```json
{"message": "Hello world", "tags": ["greeting", "test"]}
{"message": "Quick note", "tags": null}
```

---

## Handling null in your application code

When using nullable parameters, your application code needs to handle `null` values gracefully:

```python
import json

def handle_weather_call(arguments: str):
    args = json.loads(arguments)
    
    location = args["location"]      # Always present, never null
    unit = args.get("unit")          # Always present, but might be None (null)
    
    if unit is None:
        # Apply default behavior
        unit = "celsius"
    
    return get_weather(location, unit)
```

A cleaner pattern using Python's `or` operator:

```python
def handle_weather_call(arguments: str):
    args = json.loads(arguments)
    
    location = args["location"]
    unit = args["unit"] or "celsius"  # Use default when null
    
    return get_weather(location, unit)
```

**Output:**
```python
# Input: {"location": "Paris", "unit": null}
# Result: get_weather("Paris", "celsius")

# Input: {"location": "Paris", "unit": "fahrenheit"}
# Result: get_weather("Paris", "fahrenheit")
```

---

## Using Pydantic for nullable fields

Pydantic makes nullable fields straightforward with `Optional` and `None` defaults:

```python
from pydantic import BaseModel
from typing import Optional

class WeatherParams(BaseModel):
    location: str
    unit: Optional[str] = None  # Generates nullable type in JSON Schema

class SearchParams(BaseModel):
    query: str
    category: Optional[str] = None
    max_results: Optional[int] = None
    sort_by: Optional[str] = None
```

The generated JSON Schema will correctly use the nullable pattern:

```python
print(SearchParams.model_json_schema())
```

**Output:**
```json
{
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "category": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
        "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
        "sort_by": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
    },
    "required": ["query"]
}
```

> **🤖 AI Context:** When using the OpenAI or Anthropic SDKs with Pydantic models, the SDK automatically transforms the schema to meet strict mode requirements (adding `additionalProperties: false`, putting all fields in `required`, etc.). You write natural Pydantic models; the SDK handles the rest.

---

## Complete example: multi-parameter tool with nullable fields

Here's a realistic tool definition with a mix of required and nullable parameters:

```json
{
    "type": "function",
    "name": "book_flight",
    "description": "Searches for and books flights",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Departure airport code (e.g., 'JFK')"
            },
            "destination": {
                "type": "string",
                "description": "Arrival airport code (e.g., 'LAX')"
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "Departure date in YYYY-MM-DD format"
            },
            "return_date": {
                "type": ["string", "null"],
                "description": "Return date for round trips. Null for one-way flights."
            },
            "passengers": {
                "type": "integer",
                "description": "Number of passengers"
            },
            "cabin_class": {
                "type": ["string", "null"],
                "enum": ["economy", "premium_economy", "business", "first"],
                "description": "Preferred cabin class. Null for any class."
            },
            "max_price": {
                "type": ["integer", "null"],
                "description": "Maximum price in USD. Null for no limit."
            },
            "preferred_airline": {
                "type": ["string", "null"],
                "description": "Preferred airline name. Null for no preference."
            }
        },
        "required": [
            "origin", "destination", "departure_date", "return_date",
            "passengers", "cabin_class", "max_price", "preferred_airline"
        ],
        "additionalProperties": false
    }
}
```

**Example output for "Book me a one-way economy flight from JFK to LAX tomorrow":**

```json
{
    "origin": "JFK",
    "destination": "LAX",
    "departure_date": "2025-07-15",
    "return_date": null,
    "passengers": 1,
    "cabin_class": "economy",
    "max_price": null,
    "preferred_airline": null
}
```

Every field is present. Required values have real data. Optional values are `null`.

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| ✅ Add clear descriptions explaining what null means | "Null for no preference" is better than just making it nullable |
| ✅ Use `null` for "not specified", not for "empty" | An empty string `""` is different from `null` — use them intentionally |
| ✅ Apply defaults in your application code | The schema handles structure; your code handles default values |
| ✅ Use Pydantic `Optional[Type] = None` | Generates correct nullable schemas automatically |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Trying to make fields optional by removing from `required` | Use `type: ["string", "null"]` pattern instead |
| Using `"default": null` in strict mode schema | Strict mode may not support `default` values — use nullable types |
| Forgetting to handle `null` in application code | Always check for `None` before using nullable field values |
| Using `anyOf` for simple nullable types when array syntax works | Use `type: ["string", "null"]` for primitives; save `anyOf` for objects and arrays |

---

## Hands-on exercise

### Your task

Design a strict-mode-compliant schema for a `create_event` tool where some fields are truly optional.

### Requirements

1. Required fields (always have a value): `title` (string), `start_time` (string with datetime format)
2. Optional fields (nullable): `description` (string), `end_time` (string), `location` (object with `venue` and `city`), `max_attendees` (integer)
3. All fields must be in `required` — use nullable types for optional ones
4. Write the handling code that applies sensible defaults for null fields

### Expected result

A valid strict-mode schema and Python code that processes the tool call.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `"type": ["string", "null"]` for nullable strings and integers
- Use `anyOf` with `{"type": "null"}` for nullable objects
- In the handler, use `args["field"] or default_value` for simple defaults

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Schema:**
```json
{
    "type": "function",
    "name": "create_event",
    "description": "Creates a calendar event",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title"
            },
            "start_time": {
                "type": "string",
                "format": "date-time",
                "description": "Event start time in ISO 8601 format"
            },
            "description": {
                "type": ["string", "null"],
                "description": "Event description. Null if not provided."
            },
            "end_time": {
                "type": ["string", "null"],
                "description": "Event end time. Null defaults to 1 hour after start."
            },
            "location": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "venue": {"type": "string"},
                            "city": {"type": "string"}
                        },
                        "required": ["venue", "city"],
                        "additionalProperties": false
                    },
                    {"type": "null"}
                ],
                "description": "Event location. Null for virtual events."
            },
            "max_attendees": {
                "type": ["integer", "null"],
                "description": "Max attendees. Null for no limit."
            }
        },
        "required": ["title", "start_time", "description", "end_time", "location", "max_attendees"],
        "additionalProperties": false
    }
}
```

**Handler code:**
```python
import json
from datetime import datetime, timedelta

def handle_create_event(arguments: str):
    args = json.loads(arguments)
    
    title = args["title"]
    start_time = datetime.fromisoformat(args["start_time"])
    
    # Apply defaults for nullable fields
    description = args["description"] or "No description"
    
    if args["end_time"]:
        end_time = datetime.fromisoformat(args["end_time"])
    else:
        end_time = start_time + timedelta(hours=1)
    
    location = args["location"] or {"venue": "Virtual", "city": "Online"}
    max_attendees = args["max_attendees"]  # None means no limit
    
    return create_event(
        title=title,
        start=start_time,
        end=end_time,
        description=description,
        location=location,
        max_attendees=max_attendees
    )
```
</details>

### Bonus challenges

- [ ] Convert the schema to a Pydantic model using `Optional[T]` annotations
- [ ] Add validation logic that rejects events where `end_time` is before `start_time`

---

## Summary

✅ Strict mode requires all fields in `required` — use the **nullable union type pattern** (`["type", "null"]`) to emulate optional fields

✅ The field is always *present* in the output, but its *value* can be `null` when the model has no information for it

✅ Use `type: ["string", "null"]` for primitives and `anyOf: [{schema}, {"type": "null"}]` for objects and arrays

✅ Always handle `null` values in your application code with sensible defaults

✅ Pydantic's `Optional[Type] = None` generates correct nullable schemas automatically through SDK helpers

---

**Previous:** [Schema Requirements for Strict Mode ←](./03-schema-requirements.md)

**Next:** [Unsupported Schema Features →](./05-unsupported-schema-features.md)

---

## Further reading

- [OpenAI: Optional Parameters with Unions](https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required) — Official guidance on nullable fields
- [Pydantic Optional Fields](https://docs.pydantic.dev/latest/concepts/fields/#optional-fields) — Python schema generation for nullable types
- [JSON Schema: Union Types with anyOf](https://json-schema.org/understanding-json-schema/reference/composition) — Schema composition for union types

---

*[Back to Structured Outputs & Strict Mode Overview](./00-structured-outputs-strict-mode.md)*

<!-- 
Sources Consulted:
- OpenAI Structured Outputs Guide — All fields must be required: https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
- OpenAI Function Calling — Strict Mode: https://platform.openai.com/docs/guides/function-calling#strict-mode
- Anthropic Structured Outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- JSON Schema Composition: https://json-schema.org/understanding-json-schema/reference/composition
-->
