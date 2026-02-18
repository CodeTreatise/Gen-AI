---
title: "Schema Requirements for Strict Mode"
---

# Schema Requirements for Strict Mode

## Introduction

Enabling `strict: true` is only half the equation. Your JSON Schema itself must follow a specific set of rules for strict mode to accept it. If your schema violates any of these requirements, the API will reject your request with an error — not silently fall back to best-effort mode. Understanding these requirements up front saves you from frustrating debugging sessions.

In this lesson, we examine every rule your schema must follow, why each exists, and how to structure your schemas correctly from the start.

### What we'll cover

- The `additionalProperties: false` requirement and why it exists
- Why all fields must be in the `required` array
- Supported JSON Schema types for strict mode
- Object nesting depth and property count limits
- Enum and string length constraints
- Key ordering behavior

### Prerequisites

- Understanding of how to enable strict mode ([Sub-lesson 02](./02-enabling-strict-mode.md))
- Familiarity with JSON Schema syntax ([Lesson 03](../../03-json-schema-for-functions/00-json-schema-for-functions.md))

---

## Rule 1: `additionalProperties: false` on every object

The most important requirement for strict mode is that every object in your schema must set `additionalProperties: false`. This tells the model it can *only* produce the fields you've explicitly defined — no extras.

### Why this is required

Without `additionalProperties: false`, the model could theoretically add any extra key-value pair to an object. Constrained decoding needs a finite, known set of valid keys to enforce — `additionalProperties: false` makes that set explicit.

```json
// ✅ Correct: additionalProperties explicitly set to false
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "additionalProperties": false,
    "required": ["name", "age"]
}
```

```json
// ❌ Rejected: missing additionalProperties
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}
```

### Nested objects need it too

This rule applies recursively. If your schema has nested objects, *each one* must have `additionalProperties: false`:

```json
{
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "additionalProperties": false,
                    "required": ["city", "country"]
                }
            },
            "additionalProperties": false,
            "required": ["name", "address"]
        }
    },
    "additionalProperties": false,
    "required": ["user"]
}
```

> **Warning:** Forgetting `additionalProperties: false` on a deeply nested object is the most common schema rejection error. Check *every* object level.

---

## Rule 2: all fields must be in the `required` array

In strict mode, every property defined in `properties` must also appear in the `required` array. You cannot have optional fields in the traditional JSON Schema sense.

```json
// ✅ Correct: all properties listed in required
{
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location", "unit"],
    "additionalProperties": false
}
```

```json
// ❌ Rejected: "unit" is in properties but not in required
{
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"],
    "additionalProperties": false
}
```

### Why all fields must be required

Constrained decoding needs to know exactly what fields will appear in the output. If a field is optional, the model would need to decide at generation time whether to include it or not, which creates ambiguity in the grammar. By requiring all fields, the grammar is deterministic: every field appears, every time.

> **Tip:** If you need a field to be "optional" in practice, use the nullable union type pattern covered in [Sub-lesson 04: Optional Parameters](./04-optional-parameters.md).

---

## Rule 3: supported types

Strict mode supports a specific subset of JSON Schema types. The supported set is broad enough for most use cases, but not every JSON Schema type is available.

### Supported types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"hello"` |
| `number` | Floating-point numbers | `3.14` |
| `integer` | Whole numbers | `42` |
| `boolean` | True/false values | `true` |
| `object` | Key-value pairs | `{"key": "value"}` |
| `array` | Ordered lists | `[1, 2, 3]` |
| `enum` | Fixed set of values | One of `["red", "green", "blue"]` |
| `anyOf` | Union types | String or null |
| `null` | Null value (used in unions) | `null` |

### Supported property constraints

Different types support additional constraints that you can use within strict mode:

**String constraints:**

```json
{
    "type": "string",
    "pattern": "^@[a-zA-Z0-9_]+$",
    "format": "email"
}
```

Supported `format` values: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`.

**Number constraints:**

```json
{
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 0.5
}
```

Supported: `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`.

**Array constraints:**

```json
{
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10
}
```

Supported: `minItems`, `maxItems`.

> **Note:** These property constraints are supported for base models in OpenAI but *not* yet for fine-tuned models. Check your provider's documentation for current support status.

---

## Rule 4: root object must be an object type

The top-level schema must be a plain `object` type. You cannot use `anyOf` at the root level or have a non-object root.

```json
// ✅ Correct: root is a plain object
{
    "type": "object",
    "properties": {
        "result": {"type": "string"}
    },
    "required": ["result"],
    "additionalProperties": false
}
```

```json
// ❌ Rejected: root uses anyOf (discriminated union at top level)
{
    "anyOf": [
        {
            "type": "object",
            "properties": {"success": {"type": "boolean"}},
            "required": ["success"],
            "additionalProperties": false
        },
        {
            "type": "object", 
            "properties": {"error": {"type": "string"}},
            "required": ["error"],
            "additionalProperties": false
        }
    ]
}
```

### Workaround for union types at root level

Wrap the union inside a top-level object:

```json
// ✅ Workaround: wrap anyOf in an object property
{
    "type": "object",
    "properties": {
        "response": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                    "required": ["success"],
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                    "required": ["error"],
                    "additionalProperties": false
                }
            ]
        }
    },
    "required": ["response"],
    "additionalProperties": false
}
```

---

## Rule 5: nesting depth and size limits

Schemas have practical limits on complexity:

| Constraint | Limit |
|------------|-------|
| **Maximum nesting depth** | 10 levels |
| **Maximum total properties** | 5,000 across all objects |
| **Maximum enum values** | 1,000 across all enum properties |
| **Maximum string length** (names, enum values, const values) | 120,000 characters total |
| **Maximum enum string length** (single property with 250+ values) | 15,000 characters |

### Practical example of nesting depth

```json
// Level 1
{
    "type": "object",
    "properties": {
        "level2": {
            "type": "object",       // Level 2
            "properties": {
                "level3": {
                    "type": "object", // Level 3
                    "properties": {
                        // ... up to level 10
                    }
                }
            }
        }
    }
}
```

> **Tip:** If you're hitting nesting limits, flatten your schema. Instead of deeply nested objects, use a flatter structure with descriptive property names like `address_city` and `address_country` instead of `address.city` and `address.country`.

---

## Rule 6: key ordering is preserved

When strict mode generates output, properties appear in the same order they are defined in your schema. This is guaranteed behavior, not just a convention:

```json
// Schema defines: name → age → role
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"type": "string"}
    },
    "required": ["name", "age", "role"],
    "additionalProperties": false
}
```

**Output will always be:**
```json
{"name": "Alice", "age": 30, "role": "admin"}
```

**Never:**
```json
{"role": "admin", "name": "Alice", "age": 30}
```

This is useful when you need deterministic output ordering for display purposes or streaming parsers.

---

## Definitions and recursive schemas

Strict mode supports `$defs` (definitions) for reusable sub-schemas and recursive schemas using `$ref`:

### Definitions example

```json
{
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"$ref": "#/$defs/step"}
        },
        "final_answer": {"type": "string"}
    },
    "$defs": {
        "step": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "output": {"type": "string"}
            },
            "required": ["explanation", "output"],
            "additionalProperties": false
        }
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": false
}
```

### Recursive schema example

Recursive schemas use `$ref` to reference themselves, enabling tree-like structures:

```json
{
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "children": {
            "type": "array",
            "items": {"$ref": "#"}
        }
    },
    "required": ["value", "children"],
    "additionalProperties": false
}
```

This schema can produce nested trees of arbitrary depth (within the nesting limit).

---

## Complete compliant schema example

Here's a complete, strict-mode-compliant schema that uses multiple features:

```json
{
    "type": "function",
    "name": "create_invoice",
    "description": "Creates an invoice with line items",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "customer_name": {
                "type": "string",
                "description": "Full name of the customer"
            },
            "customer_email": {
                "type": "string",
                "format": "email",
                "description": "Customer's email address"
            },
            "items": {
                "type": "array",
                "description": "Line items on the invoice",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string"
                        },
                        "quantity": {
                            "type": "integer"
                        },
                        "unit_price": {
                            "type": "number"
                        }
                    },
                    "required": ["description", "quantity", "unit_price"],
                    "additionalProperties": false
                },
                "minItems": 1
            },
            "currency": {
                "type": "string",
                "enum": ["USD", "EUR", "GBP"]
            },
            "notes": {
                "type": ["string", "null"],
                "description": "Optional notes for the invoice"
            }
        },
        "required": ["customer_name", "customer_email", "items", "currency", "notes"],
        "additionalProperties": false
    }
}
```

This schema demonstrates: nested objects with their own `additionalProperties: false`, array items with object schemas, enum values, format constraints, nullable fields (covered in [Sub-lesson 04](./04-optional-parameters.md)), and all fields in `required`.

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| ✅ Start with strict mode in mind | Retrofitting existing schemas is harder than designing correctly from the start |
| ✅ Use SDK schema helpers | Pydantic and Zod generate compliant schemas automatically |
| ✅ Check nested objects carefully | Every object level needs `additionalProperties: false` |
| ✅ Keep schemas under 5 levels deep | Easier to maintain and well within the 10-level limit |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Missing `additionalProperties: false` on nested objects | Add it to *every* `"type": "object"` in your schema |
| Omitting fields from `required` for "optional" parameters | Include all fields in `required`; use nullable types instead |
| Using `anyOf` at the root level | Wrap in a top-level object with a property that uses `anyOf` |
| Exceeding 5,000 total properties | Simplify schema or split across multiple tool definitions |
| Assuming all JSON Schema features work | Only the supported subset is available — see [Sub-lesson 05](./05-unsupported-schema-features.md) |

---

## Hands-on exercise

### Your task

Take a non-compliant schema and fix all the issues to make it strict-mode compatible.

### Requirements

Fix this schema so it passes strict mode validation:

```json
{
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rating": {"type": "number"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "published": {"type": "boolean"}
            }
        }
    },
    "required": ["title", "rating"]
}
```

1. Identify all violations
2. Fix each one
3. Explain why each fix was needed

### Expected result

A valid strict-mode schema with all issues corrected.

<details>
<summary>💡 Hints (click to expand)</summary>

- Count how many `"type": "object"` entries are missing `additionalProperties: false`
- Compare the `properties` list with the `required` array
- Check if nested objects have their own `required` arrays

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Issues found:**

1. Root object missing `additionalProperties: false`
2. `tags` and `metadata` not in root `required` array
3. `metadata` object missing `additionalProperties: false`
4. `metadata` object missing `required` array for its properties

**Fixed schema:**
```json
{
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rating": {"type": "number"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "published": {"type": "boolean"}
            },
            "required": ["author", "published"],
            "additionalProperties": false
        }
    },
    "required": ["title", "rating", "tags", "metadata"],
    "additionalProperties": false
}
```

**Why each fix was needed:**

1. **`additionalProperties: false` on root** — prevents the model from adding unexpected top-level fields
2. **All properties in `required`** — strict mode requires every defined field to be present in the output
3. **`additionalProperties: false` on `metadata`** — every object needs this, including nested ones
4. **`required` on `metadata`** — nested object properties must also all be required

</details>

### Bonus challenges

- [ ] Add a nullable `description` field to the fixed schema using the `["string", "null"]` pattern
- [ ] Convert the fixed schema to a Pydantic model and verify it generates the same structure

---

## Summary

✅ Every object must have `additionalProperties: false` — at every nesting level

✅ All defined properties must appear in the `required` array — use nullable types for optional values

✅ Supported types include string, number, integer, boolean, object, array, enum, anyOf, and null

✅ Root schema must be a plain object — not `anyOf` or another type at the top level

✅ Schemas are limited to 10 nesting levels, 5,000 total properties, and 1,000 enum values

✅ Key ordering in output matches the order defined in your schema

---

**Previous:** [Enabling Strict Mode ←](./02-enabling-strict-mode.md)

**Next:** [Optional Parameters in Strict Mode →](./04-optional-parameters.md)

---

## Further reading

- [OpenAI Supported Schemas](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) — Full list of requirements and supported features
- [JSON Schema Reference](https://json-schema.org/understanding-json-schema/reference) — Understanding the JSON Schema specification
- [Anthropic JSON Schema Limitations](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations) — Anthropic-specific schema restrictions

---

*[Back to Structured Outputs & Strict Mode Overview](./00-structured-outputs-strict-mode.md)*

<!-- 
Sources Consulted:
- OpenAI Structured Outputs Supported Schemas: https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
- OpenAI Function Calling — Strict Mode: https://platform.openai.com/docs/guides/function-calling#strict-mode
- Anthropic Structured Outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- JSON Schema Reference: https://json-schema.org/understanding-json-schema/reference
-->
