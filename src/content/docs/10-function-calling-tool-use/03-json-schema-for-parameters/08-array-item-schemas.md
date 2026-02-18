---
title: "Array Item Schemas"
---

# Array Item Schemas

## Introduction

Functions frequently accept lists of things: a list of user IDs to look up, a list of tags to apply, a list of order items with quantities and prices. Arrays in JSON Schema define what those lists contain and how many items they allow. Getting array schemas right means the model generates properly typed, properly sized lists that your application can process without validation errors.

This lesson covers the `items` keyword for typing array elements, length constraints with `minItems` and `maxItems`, the `uniqueItems` constraint, and tuple validation with `prefixItems`.

### What we'll cover

- Defining array element types with `items`
- Array length constraints: `minItems` and `maxItems`
- Unique items constraint
- Tuple validation with `prefixItems`
- Arrays of objects (the most common pattern)
- Provider support for array features

### Prerequisites

- Property types ([Lesson 03](./03-property-types.md))
- Nested object schemas ([Lesson 06](./06-nested-object-schemas.md))

---

## Array basics with `items`

The `items` keyword defines the schema that every element in the array must match. Without `items`, an array accepts anything — with it, every element is validated.

### Simple typed arrays

```python
import json

# Array of strings
tags_property = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Tags to apply to the item"
}

# Array of integers
ids_property = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "User IDs to look up"
}

# Array of numbers
scores_property = {
    "type": "array",
    "items": {"type": "number"},
    "description": "Test scores as decimal values"
}

print("String array:", json.dumps(tags_property))
print("Integer array:", json.dumps(ids_property))
print("Number array:", json.dumps(scores_property))
```

**Output:**
```
String array: {"type": "array", "items": {"type": "string"}, "description": "Tags to apply to the item"}
Integer array: {"type": "array", "items": {"type": "integer"}, "description": "User IDs to look up"}
Number array: {"type": "array", "items": {"type": "number"}, "description": "Test scores as decimal values"}
```

### Array of enums

```python
import json

# Array where each element must be from a fixed set
permissions_property = {
    "type": "array",
    "items": {
        "type": "string",
        "enum": ["read", "write", "delete", "admin"]
    },
    "description": "Permissions to grant to the user"
}

# The model generates: ["read", "write"] ✅
# The model cannot generate: ["read", "execute"] ❌ ("execute" not in enum)

print(json.dumps(permissions_property, indent=2))
```

**Output:**
```json
{
  "type": "array",
  "items": {
    "type": "string",
    "enum": ["read", "write", "delete", "admin"]
  },
  "description": "Permissions to grant to the user"
}
```

> **🔑 Key concept:** The `items` schema applies to every element in the array. If `items` defines a string enum, every element must be one of those enum values.

---

## Array length constraints

Control how many items an array can contain using `minItems` and `maxItems`.

### Setting length bounds

```python
import json

# Require at least 1 tag, at most 10
tags_with_limits = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "description": "Tags to apply (1-10 tags required)"
}

# Exact count: fixed-size array
coordinates = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
    "description": "Latitude and longitude as [lat, lng]"
}

# Just a minimum
recipients = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "description": "Email addresses to send to (at least 1)"
}

print("Tags (1-10):", json.dumps(tags_with_limits))
print("Coordinates (exactly 2):", json.dumps(coordinates))
print("Recipients (at least 1):", json.dumps(recipients))
```

**Output:**
```
Tags (1-10): {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10, "description": "Tags to apply (1-10 tags required)"}
Coordinates (exactly 2): {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Latitude and longitude as [lat, lng]"}
Recipients (at least 1): {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Email addresses to send to (at least 1)"}
```

### Common length patterns

| Pattern | minItems | maxItems | Use case |
|---------|----------|----------|----------|
| At least one | 1 | — | Recipient lists, tags |
| Fixed size | N | N | Coordinates, RGB colors |
| Bounded | 1 | 50 | Search filters, selections |
| Optional (can be empty) | — | — | Optional tag lists |
| At most N | — | 5 | Top-N selections |

> **⚠️ Warning:** In OpenAI strict mode, `minItems` and `maxItems` are **not** supported. You'll need to document the expected length in the description instead.

---

## Unique items

The `uniqueItems` keyword ensures no duplicate values in an array.

```python
import json

# No duplicate tags allowed
unique_tags = {
    "type": "array",
    "items": {"type": "string"},
    "uniqueItems": True,
    "description": "Unique tags — no duplicates allowed"
}

# Without uniqueItems: ["python", "python", "ai"] ← valid
# With uniqueItems: ["python", "python", "ai"] ← invalid
# With uniqueItems: ["python", "ai", "ml"] ← valid

print(json.dumps(unique_tags, indent=2))
```

**Output:**
```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "uniqueItems": true,
  "description": "Unique tags \u2014 no duplicates allowed"
}
```

> **⚠️ Warning:** OpenAI strict mode does **not** support `uniqueItems`. Mention uniqueness in the description instead: `"description": "Tags to apply. Each tag must be unique — do not repeat tags."`

---

## Arrays of objects

The most common array pattern in function calling is an array of objects — a list of structured items.

### Basic array of objects

```python
import json

# Function to create multiple tasks at once
create_tasks_tool = {
    "type": "function",
    "function": {
        "name": "create_tasks",
        "description": "Create multiple tasks in a project",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project to add tasks to"
                },
                "tasks": {
                    "type": "array",
                    "description": "List of tasks to create",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Task title"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Task priority level"
                            },
                            "assignee": {
                                "type": ["string", "null"],
                                "description": "Person assigned, or null for unassigned"
                            }
                        },
                        "required": ["title", "priority", "assignee"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["project_id", "tasks"],
            "additionalProperties": False
        }
    }
}

# Example model output
example = {
    "project_id": "proj-123",
    "tasks": [
        {"title": "Design homepage", "priority": "high", "assignee": "alice"},
        {"title": "Write API docs", "priority": "medium", "assignee": null},
        {"title": "Fix login bug", "priority": "high", "assignee": "bob"}
    ]
}

print(f"Tasks created: {len(example['tasks'])}")
for task in example["tasks"]:
    assigned = task["assignee"] or "unassigned"
    print(f"  [{task['priority']}] {task['title']} → {assigned}")
```

**Output:**
```
Tasks created: 3
  [high] Design homepage → alice
  [medium] Write API docs → unassigned
  [high] Fix login bug → bob
```

> **🔑 Key concept:** Objects inside arrays follow the same strict mode rules: `additionalProperties: false`, all properties in `required`, and the null union pattern for optional fields.

---

## Tuple validation with `prefixItems`

Standard arrays have the same schema for every element. Tuple validation lets you define different schemas for specific positions.

```python
import json

# Tuple: [latitude, longitude, altitude]
# Each position has a different meaning
location_tuple = {
    "type": "array",
    "prefixItems": [
        {
            "type": "number",
            "description": "Latitude (-90 to 90)"
        },
        {
            "type": "number",
            "description": "Longitude (-180 to 180)"
        },
        {
            "type": "number",
            "description": "Altitude in meters"
        }
    ],
    "description": "Location as [latitude, longitude, altitude]"
}

# Another tuple: [name, age, active]
user_tuple = {
    "type": "array",
    "prefixItems": [
        {"type": "string", "description": "User name"},
        {"type": "integer", "description": "User age"},
        {"type": "boolean", "description": "Is active"}
    ],
    "description": "User data as [name, age, isActive]"
}

print("Location tuple positions:")
for i, item in enumerate(location_tuple["prefixItems"]):
    print(f"  [{i}]: {item['type']} — {item['description']}")

print("\nUser tuple positions:")
for i, item in enumerate(user_tuple["prefixItems"]):
    print(f"  [{i}]: {item['type']} — {item['description']}")
```

**Output:**
```
Location tuple positions:
  [0]: number — Latitude (-90 to 90)
  [1]: number — Longitude (-180 to 180)
  [2]: number — Altitude in meters

User tuple positions:
  [0]: string — User name
  [1]: integer — User age
  [2]: boolean — Is active
```

> **⚠️ Warning:** `prefixItems` is **not** supported in OpenAI strict mode. Use an object with named properties instead of a tuple. For example, instead of `[40.7, -74.0, 10]`, use `{"latitude": 40.7, "longitude": -74.0, "altitude": 10}`.

---

## Provider support for array features

| Feature | OpenAI (strict) | OpenAI (standard) | Anthropic | Gemini |
|---------|-----------------|-------------------|-----------|--------|
| `items` | ✅ | ✅ | ✅ | ✅ |
| `minItems` | ❌ | ✅ | ✅ | ✅ |
| `maxItems` | ❌ | ✅ | ✅ | ✅ |
| `uniqueItems` | ❌ | ✅ | ✅ | ❌ |
| `prefixItems` | ❌ | ✅ | ✅ | ❌ |
| Array of objects | ✅ | ✅ | ✅ | ✅ |

### Working with strict mode limitations

```python
import json

# ❌ Not allowed in OpenAI strict mode
non_strict_array = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 5,
    "uniqueItems": True,
    "description": "Tags"
}

# ✅ Strict mode compatible — constraints moved to description
strict_array = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Tags to apply (provide 1-5 tags, each tag must be unique, "
        "do not repeat any tag)"
    )
}

print("Non-strict:", json.dumps(non_strict_array))
print()
print("Strict-compatible:", json.dumps(strict_array))
print()
print("In strict mode, move constraints to the description")
```

**Output:**
```
Non-strict: {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "uniqueItems": true, "description": "Tags"}

Strict-compatible: {"type": "array", "items": {"type": "string"}, "description": "Tags to apply (provide 1-5 tags, each tag must be unique, do not repeat any tag)"}

In strict mode, move constraints to the description
```

---

## Nullable arrays

Making an entire array nullable (the array itself can be null, not the items):

```python
import json

# The array property can be null (no tags at all) or a list of strings
nullable_array = {
    "type": ["array", "null"],
    "items": {"type": "string"},
    "description": "Tags to apply, or null for no tags"
}

# Nullable items within an array (some items can be null)
array_with_nullable_items = {
    "type": "array",
    "items": {
        "type": ["string", "null"]
    },
    "description": "List of names, null entries represent unknown names"
}

print("Nullable array:", json.dumps(nullable_array))
print("  Valid: [\"a\", \"b\"] or null")
print()
print("Array with nullable items:", json.dumps(array_with_nullable_items))
print("  Valid: [\"alice\", null, \"bob\"]")
```

**Output:**
```
Nullable array: {"type": ["array", "null"], "items": {"type": "string"}, "description": "Tags to apply, or null for no tags"}
  Valid: ["a", "b"] or null

Array with nullable items: {"type": ["array", "null"], "items": {"type": ["string", "null"]}, "description": "List of names, null entries represent unknown names"}
  Valid: ["alice", null, "bob"]
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Always define `items` for arrays | Without it, the model can generate any type of element |
| Document length expectations in descriptions | Essential for strict mode where `minItems`/`maxItems` aren't supported |
| Use objects instead of tuples for strict mode | `prefixItems` isn't supported — named properties are clearer |
| Keep array item objects small | Large objects in arrays multiply schema complexity |
| Use enum items for fixed-choice lists | `["read", "write", "delete"]` is clearer than unconstrained strings |
| Add `additionalProperties: false` on array item objects | Required at every object level in strict mode |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Array without `items` schema | Always specify what types the array contains |
| Using `minItems`/`maxItems` in strict mode | Move constraints to the description field |
| Using `prefixItems` in strict mode | Use an object with named properties instead |
| Forgetting `additionalProperties` on array item objects | Every object needs it — inside arrays too |
| Deeply nested array-of-array-of-objects | Flatten or simplify — deep nesting reduces accuracy |
| Not handling empty arrays in code | Always check `len(items) > 0` before processing |

---

## Hands-on exercise

### Your task

Create a function schema for `create_invoice` with these parameters:

1. `customer_name` — required string
2. `customer_email` — required string
3. `line_items` — required array of objects, each with:
   - `description` (string) — what the item is
   - `quantity` (integer) — how many
   - `unit_price` (number) — price per unit
   - `tax_rate` (number) — tax rate as decimal (0.0 to 1.0)
   - `category` (enum: "product", "service", "subscription")
4. `notes` — optional (nullable) string
5. `payment_terms` — required enum: "net_15", "net_30", "net_60", "due_on_receipt"

### Requirements

1. Full strict mode compliance
2. Array item objects have `additionalProperties: false`
3. All properties at every level in `required`
4. Mention "at least 1 line item" in the `line_items` description

<details>
<summary>💡 Hints (click to expand)</summary>

- The `line_items` array items are objects — they need full strict compliance
- `notes` uses the null union pattern: `"type": ["string", "null"]`
- Describe the expected range for `tax_rate` in its description since `minimum`/`maximum` aren't supported in strict mode
- `category` is an enum inside the array item object

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

create_invoice_tool = {
    "type": "function",
    "function": {
        "name": "create_invoice",
        "description": "Create a new invoice with line items and payment terms",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Full name of the customer"
                },
                "customer_email": {
                    "type": "string",
                    "description": "Customer email address for invoice delivery"
                },
                "line_items": {
                    "type": "array",
                    "description": "Items on the invoice (at least 1 line item required)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "What the item is, e.g., 'Web Development Services'"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Number of units (must be at least 1)"
                            },
                            "unit_price": {
                                "type": "number",
                                "description": "Price per unit in dollars"
                            },
                            "tax_rate": {
                                "type": "number",
                                "description": "Tax rate as a decimal between 0.0 and 1.0, e.g., 0.08 for 8%"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["product", "service", "subscription"],
                                "description": "Type of line item"
                            }
                        },
                        "required": ["description", "quantity", "unit_price",
                                     "tax_rate", "category"],
                        "additionalProperties": False
                    }
                },
                "notes": {
                    "type": ["string", "null"],
                    "description": "Additional notes for the invoice, or null for none"
                },
                "payment_terms": {
                    "type": "string",
                    "enum": ["net_15", "net_30", "net_60", "due_on_receipt"],
                    "description": "Payment terms: net_15 (15 days), net_30, net_60, or due_on_receipt"
                }
            },
            "required": ["customer_name", "customer_email", "line_items",
                         "notes", "payment_terms"],
            "additionalProperties": False
        }
    }
}

print(json.dumps(create_invoice_tool, indent=2))
```

</details>

### Bonus challenges

- [ ] Add a `discounts` nullable array where each discount has a `code` (string), `percentage` (number), and `applies_to` (enum: "all", "products_only", "services_only")
- [ ] Write a Python function that validates a model's array output matches the schema's item type

---

## Summary

✅ The `items` keyword defines the schema for every element in an array — always specify it

✅ `minItems` and `maxItems` control array length but are not supported in OpenAI strict mode — use descriptions instead

✅ Arrays of objects are the most common pattern in function calling, and each object needs full strict mode compliance

✅ `prefixItems` for tuple validation isn't supported in strict mode — use named object properties instead

✅ When making arrays nullable, distinguish between a null array (`"type": ["array", "null"]`) and nullable items within an array

**Next:** [anyOf for Union Types](./09-anyof-union-types.md) — Defining parameters that accept multiple schemas

---

## Further reading

- [JSON Schema Array Reference](https://json-schema.org/understanding-json-schema/reference/array) — Full array keyword documentation
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Supported array features in strict mode
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — Array schemas in Anthropic tools

---

[← Previous: Recursive and Reusable Schemas](./07-recursive-reusable-schemas.md) | [Next: anyOf for Union Types →](./09-anyof-union-types.md)

<!-- 
Sources Consulted:
- JSON Schema Array Reference: https://json-schema.org/understanding-json-schema/reference/array
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Anthropic Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use/overview
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
