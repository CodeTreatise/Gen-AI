---
title: "Structured Outputs with Schemas"
---

# Structured Outputs with Schemas

## Introduction

Structured Outputs is the evolution of JSON mode. While JSON mode guarantees valid JSON, Structured Outputs guarantees that the JSON adheres to your specified schema. This means correct field names, types, enum values, and required fields—every time.

> **🤖 AI Context:** Structured Outputs use constrained decoding to enforce schema compliance at the token generation level. The model literally cannot produce output that violates your schema.

### What We'll Cover

- OpenAI Structured Outputs feature
- Schema-based generation
- Strict mode vs flexible schemas
- Supported schema features and limitations
- Provider-specific implementations

### Prerequisites

- [JSON Mode in API Calls](./01-json-mode-api.md)

---

## JSON Mode vs Structured Outputs

| Feature | JSON Mode | Structured Outputs |
|---------|-----------|-------------------|
| Valid JSON | ✅ Guaranteed | ✅ Guaranteed |
| Schema adherence | ❌ No | ✅ Yes |
| Field names | ❌ May vary | ✅ Exact match |
| Field types | ❌ May vary | ✅ Enforced |
| Enum values | ❌ May hallucinate | ✅ Constrained |
| Required fields | ❌ May omit | ✅ Always present |
| Model support | Wide | Newer models only |

---

## OpenAI Structured Outputs

### Basic Usage

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# Define your schema
class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool

# Use the parse method with text_format
response = client.responses.parse(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Extract product information."},
        {"role": "user", "content": "Apple AirPods Pro - $249, electronics, available now"}
    ],
    text_format=ProductInfo
)

# Access parsed output directly
product = response.output_parsed
print(product.name)    # "Apple AirPods Pro"
print(product.price)   # 249.0
print(product.category) # "electronics"
```

### Using Chat Completions API

```python
response = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract product information."},
        {"role": "user", "content": "Apple AirPods Pro - $249, electronics"}
    ],
    response_format=ProductInfo
)

product = response.choices[0].message.parsed
```

### Using Raw JSON Schema

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract product information."},
        {"role": "user", "content": "Apple AirPods Pro - $249"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "category": {"type": "string"},
                    "in_stock": {"type": "boolean"}
                },
                "required": ["name", "price", "category", "in_stock"],
                "additionalProperties": False
            }
        }
    }
)

import json
product = json.loads(response.choices[0].message.content)
```

---

## Strict Mode

Strict mode (`"strict": true`) enables constrained decoding:

### What Strict Mode Requires

| Requirement | Description |
|-------------|-------------|
| All fields required | Every field must be in `required` array |
| No additional properties | Must set `additionalProperties: false` |
| Supported types only | String, number, boolean, object, array, enum, anyOf |

### Strict Mode Example

```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "classification",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["billing", "technical", "account"]
                },
                "confidence": {
                    "type": "number"
                },
                "reasoning": {
                    "type": "string"
                }
            },
            "required": ["category", "confidence", "reasoning"],
            "additionalProperties": false
        }
    }
}
```

### Emulating Optional Fields

Since all fields must be required, use union with null:

```json
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {
            "type": ["string", "null"]
        }
    },
    "required": ["name", "email", "phone"],
    "additionalProperties": false
}
```

The model will output `"phone": null` when phone isn't available.

---

## Supported Schema Features

### Types

| Type | Example | Notes |
|------|---------|-------|
| string | `{"type": "string"}` | Text values |
| number | `{"type": "number"}` | Decimals allowed |
| integer | `{"type": "integer"}` | Whole numbers only |
| boolean | `{"type": "boolean"}` | true/false |
| object | `{"type": "object", ...}` | Nested structures |
| array | `{"type": "array", ...}` | Lists |
| enum | `{"enum": ["a", "b"]}` | Fixed values |
| anyOf | `{"anyOf": [...]}` | Union types |

### String Constraints

```json
{
    "type": "string",
    "pattern": "^@[a-zA-Z0-9_]+$",
    "format": "email"
}
```

Supported formats:
- `date-time`, `date`, `time`
- `email`, `hostname`
- `ipv4`, `ipv6`
- `uuid`

### Number Constraints

```json
{
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 0.5
}
```

### Array Constraints

```json
{
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10
}
```

---

## Nested Objects and Definitions

### Nested Objects

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    email: str
    address: Address
    tags: List[str]

response = client.responses.parse(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "John Doe, john@example.com, 123 Main St, NYC, USA. Tags: customer, vip"}
    ],
    text_format=Person
)
```

### Using $defs for Reusable Schemas

```json
{
    "type": "object",
    "properties": {
        "billing_address": {"$ref": "#/$defs/address"},
        "shipping_address": {"$ref": "#/$defs/address"}
    },
    "$defs": {
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip": {"type": "string"}
            },
            "required": ["street", "city", "zip"],
            "additionalProperties": false
        }
    },
    "required": ["billing_address", "shipping_address"],
    "additionalProperties": false
}
```

---

## Recursive Schemas

Structured Outputs support self-referential schemas:

```json
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "children": {
            "type": "array",
            "items": {"$ref": "#"}
        }
    },
    "required": ["name", "children"],
    "additionalProperties": false
}
```

This enables structures like:
- Organizational hierarchies
- File system trees
- Comment threads
- UI component trees

---

## Schema Limitations

### Not Supported in Strict Mode

| Feature | Status |
|---------|--------|
| `allOf` | ❌ Not supported |
| `not` | ❌ Not supported |
| `if/then/else` | ❌ Not supported |
| Top-level `anyOf` | ❌ Root must be object |
| Optional fields | ⚠️ Use union with null |

### Size Limits

| Limit | Value |
|-------|-------|
| Total properties | 5,000 |
| Nesting depth | 10 levels |
| String content (names, enums) | 120,000 chars |
| Enum values | 1,000 total |

---

## Handling Refusals

When the model refuses for safety reasons, the response differs:

```python
response = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=MySchema
)

message = response.choices[0].message

# Check for refusal
if message.refusal:
    print(f"Model refused: {message.refusal}")
else:
    data = message.parsed
    # Use the data
```

### Refusal Response Structure

```json
{
    "content": [
        {
            "type": "refusal",
            "refusal": "I'm sorry, I cannot assist with that request."
        }
    ]
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Pydantic/Zod | Type safety and validation |
| Set `strict: true` | Guaranteed schema compliance |
| Handle refusals | Safety refusals bypass schema |
| Keep schemas focused | Simpler = more reliable |
| Use descriptions | Help model understand intent |
| Test edge cases | Empty inputs, unusual content |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Optional fields without null union | Use `["string", "null"]` |
| Missing `additionalProperties: false` | Always include in strict mode |
| Not all fields in `required` | List every field |
| Too deep nesting | Keep under 10 levels |
| Assuming refusal won't happen | Always check for refusals |

---

## Hands-on Exercise

### Your Task

Create a Structured Output schema for extracting meeting information.

### Requirements

1. Extract: title, datetime, attendees (array), agenda items (nested)
2. Use Pydantic for schema definition
3. Handle optional fields (location, duration)
4. Agenda items should have: topic, duration_minutes, presenter

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Optional[]` with default `None` for optional fields
- Pydantic converts to proper JSON Schema automatically
- How do you handle unknown presenters?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List

# Schema definitions
class AgendaItem(BaseModel):
    topic: str = Field(description="The agenda item topic")
    duration_minutes: int = Field(description="Expected duration in minutes")
    presenter: Optional[str] = Field(
        default=None, 
        description="Presenter name, null if not specified"
    )

class Attendee(BaseModel):
    name: str
    role: Optional[str] = None

class MeetingInfo(BaseModel):
    title: str = Field(description="Meeting title or subject")
    datetime: str = Field(description="Date and time in ISO 8601 format or as stated")
    location: Optional[str] = Field(
        default=None, 
        description="Meeting location, null if virtual or not specified"
    )
    duration_minutes: Optional[int] = Field(
        default=None,
        description="Total meeting duration in minutes"
    )
    attendees: List[Attendee] = Field(description="List of meeting attendees")
    agenda: List[AgendaItem] = Field(description="Meeting agenda items")

# Usage
client = OpenAI()

def extract_meeting(text: str) -> MeetingInfo:
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "Extract meeting information from the provided text."
            },
            {"role": "user", "content": text}
        ],
        text_format=MeetingInfo
    )
    
    # Check for refusal
    if hasattr(response, 'refusal') and response.refusal:
        raise ValueError(f"Model refused: {response.refusal}")
    
    return response.output_parsed

# Test
meeting_text = """
Team standup tomorrow at 10am in Conference Room B.
Attendees: Sarah (PM), Mike (Dev), Lisa (QA)
Agenda:
- Sprint review (15 min, Sarah)
- Bug triage (20 min, Lisa)
- Planning discussion (25 min)
Duration: 1 hour
"""

meeting = extract_meeting(meeting_text)
print(f"Title: {meeting.title}")
print(f"When: {meeting.datetime}")
print(f"Location: {meeting.location}")
print(f"Attendees: {[a.name for a in meeting.attendees]}")
for item in meeting.agenda:
    print(f"  - {item.topic} ({item.duration_minutes}min) - {item.presenter or 'TBD'}")
```

**Expected output:**
```
Title: Team standup
When: tomorrow at 10am
Location: Conference Room B
Attendees: ['Sarah', 'Mike', 'Lisa']
  - Sprint review (15min) - Sarah
  - Bug triage (20min) - Lisa
  - Planning discussion (25min) - TBD
```

</details>

### Bonus Challenge

- [ ] Add validation for duration (must be positive)
- [ ] Include a notes field for additional context

---

## Summary

✅ **Structured Outputs guarantee** schema compliance, not just valid JSON

✅ **Strict mode** enables constrained decoding

✅ **All fields required** with null union for optional

✅ **Nested objects and recursion** supported

✅ **Always handle refusals** as special case

**Next:** [Response Format Parameter](./03-response-format-parameter.md)

---

## Further Reading

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [JSON Schema Specification](https://json-schema.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
