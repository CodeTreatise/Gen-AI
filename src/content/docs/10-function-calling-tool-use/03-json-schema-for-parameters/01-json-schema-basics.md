---
title: "JSON Schema Basics"
---

# JSON Schema Basics

## Introduction

Every function you define for an AI model needs a parameters schema — a structured description of what inputs the function accepts. That schema is written in JSON Schema, a standard specification for describing the shape, types, and constraints of JSON data. Without a schema, the model would be guessing at what arguments to send. With a schema, it knows exactly what to construct.

In this lesson, we build a solid foundation in JSON Schema as it applies to function calling. We cover the core syntax, explain which subset of JSON Schema the major AI providers support, and walk through type definitions and property declarations from scratch.

### What we'll cover

- JSON Schema syntax and structure
- The OpenAPI schema subset used by AI providers
- Type definitions for every supported type
- Property definitions with descriptions and constraints
- Provider differences in schema support

### Prerequisites

- Familiarity with JSON syntax (objects, arrays, strings, numbers)
- Understanding of function calling concepts ([Lesson 01](../01-function-calling-concepts/00-function-calling-concepts.md))
- Basic understanding of function definitions ([Lesson 02](../02-defining-functions/00-defining-functions.md))

---

## What is JSON Schema?

JSON Schema is a vocabulary that lets you describe the structure and validation rules of JSON data. Think of it as a blueprint for JSON objects — it defines what properties an object should have, what types those properties should be, and what constraints apply.

In everyday terms, JSON Schema is to JSON what a form template is to a filled-out form. The template says "field 1 must be text, field 2 must be a number between 1 and 100, field 3 must be one of these three options." The filled-out form is the actual data.

### A minimal JSON Schema

Here's the simplest possible schema for a function parameter:

```json
{
  "type": "object",
  "properties": {
    "city": {
      "type": "string",
      "description": "The city name, e.g., 'San Francisco'"
    }
  },
  "required": ["city"]
}
```

This schema says: "I expect a JSON object with one property called `city`, which must be a string." When the model sees this schema, it knows to generate something like `{"city": "San Francisco"}`.

### Schema anatomy

Every function parameter schema has the same top-level structure:

```json
{
  "type": "object",
  "properties": {
    "param_1": { "type": "...", "description": "..." },
    "param_2": { "type": "...", "description": "..." }
  },
  "required": ["param_1"]
}
```

| Component | Purpose | Required? |
|-----------|---------|-----------|
| `type` | Always `"object"` for the root schema | ✅ Yes |
| `properties` | Defines each parameter as a named property | ✅ Yes |
| `required` | Lists which properties the model must always include | ✅ Recommended |
| `additionalProperties` | Controls whether extra properties are allowed | Provider-dependent |

> **🔑 Key concept:** The root of every function parameter schema is always `"type": "object"`. Individual parameters are defined inside `properties`. You never define a function's parameters as a bare string or array at the root level.

---

## The OpenAPI schema subset

AI providers don't support the full JSON Schema specification. Instead, they support a subset based on the [OpenAPI 3.0 Schema Object](https://spec.openapis.org/oas/v3.0.3#schema-object), which itself is a subset of JSON Schema Draft 2020-12.

### What's supported

The following JSON Schema features work across all three major providers (OpenAI, Anthropic, Google Gemini):

| Feature | Keyword | Example |
|---------|---------|---------|
| Type declaration | `type` | `"type": "string"` |
| Property definitions | `properties` | Object with named properties |
| Required fields | `required` | `["name", "age"]` |
| Descriptions | `description` | Human-readable text |
| Enumerations | `enum` | `["red", "green", "blue"]` |
| Nested objects | `properties` within `properties` | Objects inside objects |
| Arrays | `type: "array"` + `items` | Typed array items |
| Union types | `anyOf` | Multiple valid schemas |
| Reusable definitions | `$defs` + `$ref` | Schema reuse |

### What's NOT supported

These JSON Schema features are either unsupported or unsupported across providers:

| Feature | Keyword | Why Not? |
|---------|---------|----------|
| Conditional schemas | `if` / `then` / `else` | Too complex for model output |
| All-of composition | `allOf` | Not supported in strict mode |
| Negation | `not` | Ambiguous for model generation |
| Dependencies | `dependentRequired` | Not supported in strict mode |
| One-of | `oneOf` | Use `anyOf` instead |
| Pattern properties | `patternProperties` | Not supported in strict mode |

> **Warning:** Just because a keyword works in standard JSON Schema validation libraries doesn't mean it works with AI function calling. Always check provider-specific documentation.

### Provider-specific differences

While the core subset is similar, each provider has nuances:

```python
# The same function defined for all three providers

# OpenAI — uses 'parameters' with optional 'strict' mode
openai_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'London'"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city", "units"],
            "additionalProperties": False
        },
        "strict": True
    }
}

# Anthropic — uses 'input_schema'
anthropic_tool = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g., 'London'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["city", "units"]
    }
}

# Google Gemini — uses 'parameters' (OpenAPI subset)
gemini_tool = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g., 'London'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["city", "units"]
    }
}
```

**Output (schema comparison):**
```
All three providers:
- Root type is always "object"
- Properties define each parameter
- Required lists mandatory parameters
- Descriptions guide the model

Differences:
- OpenAI: "parameters" key + "strict" flag + "additionalProperties"
- Anthropic: "input_schema" key (no strict flag, use strict: true at tool level)
- Gemini: "parameters" key (OpenAPI subset, no additionalProperties support)
```

> **🤖 AI Context:** The schema is part of the system prompt that the model sees. Better schemas mean better function calls — the model uses the schema structure, types, and descriptions to decide what arguments to generate.

---

## Type definitions

JSON Schema supports six primitive types and two structural types. In function calling, you'll use all of them.

### Primitive types

| Type | JSON Schema | Python Equivalent | Example Value |
|------|------------|-------------------|---------------|
| String | `"type": "string"` | `str` | `"hello"` |
| Number | `"type": "number"` | `float` | `3.14` |
| Integer | `"type": "integer"` | `int` | `42` |
| Boolean | `"type": "boolean"` | `bool` | `true` |
| Null | `"type": "null"` | `None` | `null` |

### Structural types

| Type | JSON Schema | Python Equivalent | Example Value |
|------|------------|-------------------|---------------|
| Object | `"type": "object"` | `dict` | `{"key": "value"}` |
| Array | `"type": "array"` | `list` | `[1, 2, 3]` |

### Type declaration examples

```python
import json

# A schema demonstrating all common types
schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "User's full name"
        },
        "age": {
            "type": "integer",
            "description": "User's age in years"
        },
        "height": {
            "type": "number",
            "description": "User's height in meters"
        },
        "is_active": {
            "type": "boolean",
            "description": "Whether the user account is active"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tags associated with the user"
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip": {"type": "string"}
            },
            "required": ["street", "city", "zip"],
            "description": "User's mailing address"
        }
    },
    "required": ["name", "age", "is_active"]
}

print(json.dumps(schema, indent=2))
```

**Output:**
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "User's full name"
    },
    "age": {
      "type": "integer",
      "description": "User's age in years"
    },
    "height": {
      "type": "number",
      "description": "User's height in meters"
    },
    "is_active": {
      "type": "boolean",
      "description": "Whether the user account is active"
    },
    "tags": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of tags associated with the user"
    },
    "address": {
      "type": "object",
      "properties": {
        "street": { "type": "string" },
        "city": { "type": "string" },
        "zip": { "type": "string" }
      },
      "required": ["street", "city", "zip"],
      "description": "User's mailing address"
    }
  },
  "required": ["name", "age", "is_active"]
}
```

---

## Property definitions

Each property inside `properties` is a mini-schema that defines a single parameter. A property definition can include any combination of type, description, constraints, and nested structure.

### Anatomy of a property

```json
{
  "property_name": {
    "type": "string",
    "description": "What this parameter is for",
    "enum": ["option_a", "option_b"],
    "minLength": 1,
    "maxLength": 100
  }
}
```

| Field | Purpose | When to Use |
|-------|---------|-------------|
| `type` | The data type | Always — every property needs a type |
| `description` | Human-readable purpose | Always — guides the model's output |
| `enum` | Fixed set of allowed values | When only specific values are valid |
| `format` | Semantic type hint | When the string has a specific format (date, email, etc.) |
| `minimum` / `maximum` | Numeric range | When values must be within bounds |
| `minLength` / `maxLength` | String length | When string length matters |
| `items` | Schema for array elements | When the property is an array |
| `properties` | Nested object structure | When the property is an object |
| `default` | Default value hint | Not in strict mode; use for documentation |

### A complete property example

```python
import json

# A function for creating a support ticket
create_ticket_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Brief summary of the issue, 5-100 characters",
            "minLength": 5,
            "maxLength": 100
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Ticket priority level"
        },
        "category": {
            "type": "string",
            "enum": ["bug", "feature", "question", "documentation"],
            "description": "Type of support request"
        },
        "description": {
            "type": "string",
            "description": "Detailed description of the issue"
        },
        "affected_users": {
            "type": "integer",
            "description": "Estimated number of users affected by this issue",
            "minimum": 0
        },
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Labels to categorize the ticket, e.g., ['frontend', 'login']"
        }
    },
    "required": ["title", "priority", "category", "description"]
}

print(json.dumps(create_ticket_schema, indent=2))
```

**Output:**
```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "Brief summary of the issue, 5-100 characters",
      "minLength": 5,
      "maxLength": 100
    },
    "priority": {
      "type": "string",
      "enum": ["low", "medium", "high", "critical"],
      "description": "Ticket priority level"
    },
    "category": {
      "type": "string",
      "enum": ["bug", "feature", "question", "documentation"],
      "description": "Type of support request"
    },
    "description": {
      "type": "string",
      "description": "Detailed description of the issue"
    },
    "affected_users": {
      "type": "integer",
      "description": "Estimated number of users affected by this issue",
      "minimum": 0
    },
    "labels": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Labels to categorize the ticket, e.g., ['frontend', 'login']"
    }
  },
  "required": ["title", "priority", "category", "description"]
}
```

---

## Putting it all together

Let's build a complete function definition that demonstrates all the basics:

```python
from openai import OpenAI

client = OpenAI()

# Complete tool definition with well-structured schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the catalog by various criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text, e.g., 'wireless headphones'"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "home", "sports"],
                        "description": "Product category to filter by"
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price in USD",
                        "minimum": 0
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD",
                        "minimum": 0
                    },
                    "in_stock": {
                        "type": "boolean",
                        "description": "If true, only return products currently in stock"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "price_asc", "price_desc", "rating", "newest"],
                        "description": "How to sort the results"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50)",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# When strict mode is enabled, ALL properties must be required
# and optional ones use null union — see Lesson 02 for details
```

> **Note:** This example uses OpenAI's format. The same schema structure works with Anthropic (change `parameters` to `input_schema`) and Gemini (remove `strict` and `additionalProperties`).

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Always set `"type": "object"` at the root | Required by all providers — the root must be an object |
| Include `description` on every property | The model relies on descriptions to understand what to generate |
| Use the most specific type | `"integer"` instead of `"number"` when you need whole numbers |
| List `required` properties explicitly | Prevents the model from omitting critical parameters |
| Use `enum` for fixed value sets | More reliable than describing options in the description |
| Keep schemas focused | One function, one job — don't cram everything into one schema |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `"type": "string"` for the root schema | Always use `"type": "object"` as the root |
| Omitting `description` from properties | Add descriptions to every property — the model needs them |
| Using `number` when you mean `integer` | Use `integer` for whole numbers (counts, IDs, ages) |
| Defining parameters as a flat list | Wrap parameters in an `object` with `properties` |
| Using unsupported keywords like `oneOf` | Check provider docs — use `anyOf` instead of `oneOf` |
| Nesting too deeply (>10 levels) | Flatten your schema — providers limit nesting depth |

---

## Hands-on exercise

### Your task

Create a JSON Schema for a `book_appointment` function that schedules appointments. The function should accept:

1. A patient name (required string)
2. A doctor name (required string)
3. A date (required string in YYYY-MM-DD format)
4. A time slot (required, one of: "09:00", "10:00", "11:00", "14:00", "15:00", "16:00")
5. An appointment type (required, one of: "checkup", "follow_up", "consultation", "emergency")
6. Notes (optional string)
7. Whether it's a first visit (required boolean)

Write the complete parameter schema and wrap it in an OpenAI tool definition.

### Requirements

1. Use correct JSON Schema types for each parameter
2. Use `enum` for time slots and appointment types
3. Include descriptive `description` fields for every property
4. List all required fields in the `required` array
5. Use `format` for the date string if appropriate

### Expected result

A complete tool definition that, when given to an AI model, would generate valid arguments like:

```json
{
  "patient_name": "Jane Smith",
  "doctor_name": "Dr. Chen",
  "date": "2025-03-15",
  "time_slot": "10:00",
  "appointment_type": "checkup",
  "notes": "Annual physical exam",
  "is_first_visit": false
}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- The root `type` is always `"object"`
- Use `"format": "date"` for date strings
- Time slots are a fixed set — use `enum`
- Notes are optional, so don't include them in `required`
- Each property needs a `type` and `description`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

tool = {
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Schedule a medical appointment with a specific doctor",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {
                    "type": "string",
                    "description": "Full name of the patient"
                },
                "doctor_name": {
                    "type": "string",
                    "description": "Name of the doctor, e.g., 'Dr. Chen'"
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Appointment date in YYYY-MM-DD format, e.g., '2025-03-15'"
                },
                "time_slot": {
                    "type": "string",
                    "enum": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
                    "description": "Available time slot for the appointment"
                },
                "appointment_type": {
                    "type": "string",
                    "enum": ["checkup", "follow_up", "consultation", "emergency"],
                    "description": "Type of appointment being scheduled"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes or reason for the visit"
                },
                "is_first_visit": {
                    "type": "boolean",
                    "description": "Whether this is the patient's first visit to this doctor"
                }
            },
            "required": [
                "patient_name",
                "doctor_name",
                "date",
                "time_slot",
                "appointment_type",
                "is_first_visit"
            ]
        }
    }
}

print(json.dumps(tool, indent=2))
```

</details>

### Bonus challenges

- [ ] Convert the schema to Anthropic's `input_schema` format
- [ ] Convert the schema to Google Gemini's format
- [ ] Add strict mode support (what changes are needed?)

---

## Summary

✅ JSON Schema describes the structure, types, and constraints of function parameters

✅ The root of every parameter schema is `"type": "object"` with `properties` defining each parameter

✅ AI providers support an OpenAPI subset of JSON Schema — not the full specification

✅ Every property should have a `type` and `description` at minimum

✅ The three providers use the same schema structure with different wrapper keys (`parameters`, `input_schema`)

**Next:** [Strict Mode Requirements](./02-strict-mode-requirements.md) — How strict mode changes schema rules for guaranteed conformance

---

## Further reading

- [JSON Schema Official Reference](https://json-schema.org/understanding-json-schema/reference) — Complete type and keyword reference
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — OpenAI's schema support and restrictions
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — Anthropic's tool definition format
- [Google Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini's function declaration format

---

[← Previous: Overview](./00-json-schema-for-parameters.md) | [Next: Strict Mode Requirements →](./02-strict-mode-requirements.md)

<!-- 
Sources Consulted:
- JSON Schema Reference (string): https://json-schema.org/understanding-json-schema/reference/string
- JSON Schema Reference (object): https://json-schema.org/understanding-json-schema/reference/object
- JSON Schema Reference (array): https://json-schema.org/understanding-json-schema/reference/array
- JSON Schema Reference (numeric): https://json-schema.org/understanding-json-schema/reference/numeric
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
