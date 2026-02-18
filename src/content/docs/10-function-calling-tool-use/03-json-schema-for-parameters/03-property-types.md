---
title: "Property Types"
---

# Property Types

## Introduction

Every parameter in a function schema has a type. Choosing the right type is the difference between a model that sends `42` (an integer you can use immediately) and one that sends `"forty-two"` (a string you have to parse, if you can parse it at all). JSON Schema supports five primitive types relevant to function calling: string, number, integer, boolean, and null — plus constraints that narrow valid values within each type.

In this lesson, we cover each type in depth, explore the constraint keywords that make types precise, and show how to choose the right type for common parameter patterns.

### What we'll cover

- String type with `format`, `pattern`, `minLength`, and `maxLength`
- Number and integer types with range and multiple constraints
- The integer vs. number distinction
- Boolean type and when to use it
- Type selection guidelines for common parameters

### Prerequisites

- JSON Schema basics ([Lesson 01](./01-json-schema-basics.md))
- Strict mode requirements ([Lesson 02](./02-strict-mode-requirements.md))

---

## String type

Strings are the most common parameter type in function calling. They represent text, identifiers, dates, emails, and any free-form input. JSON Schema provides several constraint keywords to narrow what counts as a valid string.

### Basic string

```python
import json

# Simple string property
name_property = {
    "type": "string",
    "description": "The user's full name"
}

print(json.dumps(name_property, indent=2))
```

**Output:**
```json
{
  "type": "string",
  "description": "The user's full name"
}
```

### String length constraints

Use `minLength` and `maxLength` to control string length. Both values must be non-negative integers.

```python
import json

# String with length constraints
title_property = {
    "type": "string",
    "description": "Article title, 5-200 characters",
    "minLength": 5,
    "maxLength": 200
}

# Common patterns
examples = {
    "search_query": {
        "type": "string",
        "description": "Search query text",
        "minLength": 1,
        "maxLength": 500
    },
    "username": {
        "type": "string",
        "description": "Username, 3-30 characters, letters and numbers only",
        "minLength": 3,
        "maxLength": 30
    },
    "comment": {
        "type": "string",
        "description": "User comment, up to 2000 characters",
        "maxLength": 2000
    }
}

for name, schema in examples.items():
    constraints = []
    if "minLength" in schema:
        constraints.append(f"min: {schema['minLength']}")
    if "maxLength" in schema:
        constraints.append(f"max: {schema['maxLength']}")
    print(f"{name}: {', '.join(constraints)}")
```

**Output:**
```
search_query: min: 1, max: 500
username: min: 3, max: 30
comment: max: 2000
```

> **Warning:** `minLength` and `maxLength` are supported in OpenAI strict mode with base models, but NOT with fine-tuned models. When using fine-tuned models, document length constraints in the `description` instead.

### The `format` keyword

The `format` keyword adds semantic meaning to a string. It tells the model (and validators) that the string should follow a specific format.

```python
import json

# Supported format values for function calling
format_examples = {
    "date-time": {
        "type": "string",
        "format": "date-time",
        "description": "ISO 8601 date-time, e.g., '2025-03-15T10:30:00Z'"
    },
    "date": {
        "type": "string",
        "format": "date",
        "description": "ISO 8601 date, e.g., '2025-03-15'"
    },
    "time": {
        "type": "string",
        "format": "time",
        "description": "ISO 8601 time, e.g., '10:30:00'"
    },
    "email": {
        "type": "string",
        "format": "email",
        "description": "Email address, e.g., 'user@example.com'"
    },
    "uuid": {
        "type": "string",
        "format": "uuid",
        "description": "UUID v4, e.g., '550e8400-e29b-41d4-a716-446655440000'"
    },
    "ipv4": {
        "type": "string",
        "format": "ipv4",
        "description": "IPv4 address, e.g., '192.168.1.1'"
    },
    "uri": {
        "type": "string",
        "format": "uri",
        "description": "Full URI, e.g., 'https://example.com/path'"
    }
}

for fmt, schema in format_examples.items():
    print(f"format: {fmt:12s} → {schema['description']}")
```

**Output:**
```
format: date-time    → ISO 8601 date-time, e.g., '2025-03-15T10:30:00Z'
format: date         → ISO 8601 date, e.g., '2025-03-15'
format: time         → ISO 8601 time, e.g., '10:30:00'
format: email        → Email address, e.g., 'user@example.com'
format: uuid         → UUID v4, e.g., '550e8400-e29b-41d4-a716-446655440000'
format: ipv4         → IPv4 address, e.g., '192.168.1.1'
format: uri          → Full URI, e.g., 'https://example.com/path'
```

> **🔑 Key concept:** The `format` keyword is an annotation in JSON Schema — by default it doesn't enforce validation, it just describes the expected format. In OpenAI strict mode, however, `format` IS enforced through constrained decoding, making it a powerful validation tool.

### The `pattern` keyword

Use `pattern` to validate strings against a regular expression. The regex syntax follows ECMA 262 (JavaScript regex).

```python
import json

# Pattern examples for function calling
pattern_examples = {
    "phone_number": {
        "type": "string",
        "pattern": "^\\+?[1-9]\\d{1,14}$",
        "description": "Phone number in E.164 format, e.g., '+14155552671'"
    },
    "hex_color": {
        "type": "string",
        "pattern": "^#[0-9a-fA-F]{6}$",
        "description": "Hex color code, e.g., '#FF5733'"
    },
    "zip_code": {
        "type": "string",
        "pattern": "^\\d{5}(-\\d{4})?$",
        "description": "US ZIP code, e.g., '94102' or '94102-3456'"
    },
    "semantic_version": {
        "type": "string",
        "pattern": "^\\d+\\.\\d+\\.\\d+$",
        "description": "Semantic version, e.g., '2.1.0'"
    }
}

for name, schema in pattern_examples.items():
    print(f"{name}: pattern={schema['pattern']}")
```

**Output:**
```
phone_number: pattern=^\+?[1-9]\d{1,14}$
hex_color: pattern=^#[0-9a-fA-F]{6}$
zip_code: pattern=^\d{5}(-\d{4})?$
semantic_version: pattern=^\d+\.\d+\.\d+$
```

> **💡 Tip:** Always anchor your patterns with `^` and `$` to match the entire string. Without anchors, a pattern like `"[0-9]{3}"` would match `"abc123def"` because it finds "123" within the string.

---

## Number type

The `number` type accepts any numeric value — integers, decimals, and numbers in scientific notation.

### Basic number

```python
import json

# Simple number property
price_property = {
    "type": "number",
    "description": "Product price in USD"
}

# Valid number values
valid_numbers = [42, -1, 3.14, 0.99, 2.998e8, 0]
print("Valid number values:", valid_numbers)
```

**Output:**
```
Valid number values: [42, -1, 3.14, 0.99, 299800000.0, 0]
```

### Range constraints

Use `minimum`, `maximum`, `exclusiveMinimum`, and `exclusiveMaximum` to constrain the valid range.

```python
import json

# Range constraint examples
range_examples = {
    "temperature_celsius": {
        "type": "number",
        "minimum": -273.15,
        "maximum": 1000,
        "description": "Temperature in Celsius (-273.15 to 1000)"
    },
    "percentage": {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "description": "Percentage value from 0 to 100"
    },
    "positive_amount": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": "A positive amount (must be greater than 0)"
    },
    "probability": {
        "type": "number",
        "minimum": 0,
        "exclusiveMaximum": 1,
        "description": "Probability value, 0 ≤ p < 1"
    }
}

for name, schema in range_examples.items():
    constraints = []
    if "minimum" in schema:
        constraints.append(f"≥ {schema['minimum']}")
    if "exclusiveMinimum" in schema:
        constraints.append(f"> {schema['exclusiveMinimum']}")
    if "maximum" in schema:
        constraints.append(f"≤ {schema['maximum']}")
    if "exclusiveMaximum" in schema:
        constraints.append(f"< {schema['exclusiveMaximum']}")
    print(f"{name}: {', '.join(constraints)}")
```

**Output:**
```
temperature_celsius: ≥ -273.15, ≤ 1000
percentage: ≥ 0, ≤ 100
positive_amount: > 0
probability: ≥ 0, < 1
```

### The `multipleOf` constraint

Use `multipleOf` to restrict numbers to multiples of a given value. This is useful for prices (multiples of 0.01), quantities, or step values.

```python
import json

# multipleOf examples
multiple_examples = {
    "price_usd": {
        "type": "number",
        "multipleOf": 0.01,
        "minimum": 0,
        "description": "Price in USD, rounded to cents"
    },
    "angle_degrees": {
        "type": "number",
        "multipleOf": 15,
        "minimum": 0,
        "maximum": 360,
        "description": "Angle in 15-degree increments"
    },
    "quantity": {
        "type": "number",
        "multipleOf": 0.5,
        "minimum": 0,
        "description": "Quantity in half-unit increments"
    }
}

for name, schema in multiple_examples.items():
    print(f"{name}: multipleOf={schema['multipleOf']}")
```

**Output:**
```
price_usd: multipleOf=0.01
angle_degrees: multipleOf=15
quantity: multipleOf=0.5
```

---

## Integer vs. number

JSON doesn't distinguish between integers and floating-point numbers at the format level — `1` and `1.0` are the same value. But JSON Schema does make this distinction with separate types, and it matters for function calling.

### When to use each

```python
import json

# Integer: whole numbers only
integer_examples = {
    "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "count": {"type": "integer", "minimum": 0},
    "page_number": {"type": "integer", "minimum": 1},
    "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
    "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
    "http_status": {"type": "integer", "minimum": 100, "maximum": 599}
}

# Number: decimals allowed
number_examples = {
    "price": {"type": "number", "minimum": 0},
    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
    "longitude": {"type": "number", "minimum": -180, "maximum": 180},
    "weight_kg": {"type": "number", "minimum": 0},
    "rating": {"type": "number", "minimum": 0, "maximum": 5},
    "temperature": {"type": "number"}
}

print("Use INTEGER for:")
for name in integer_examples:
    print(f"  • {name}")

print("\nUse NUMBER for:")
for name in number_examples:
    print(f"  • {name}")
```

**Output:**
```
Use INTEGER for:
  • age
  • count
  • page_number
  • year
  • quantity
  • http_status

Use NUMBER for:
  • price
  • latitude
  • longitude
  • weight_kg
  • rating
  • temperature
```

### Key distinction

| Aspect | `integer` | `number` |
|--------|-----------|----------|
| Valid values | Whole numbers only | Any numeric value |
| `42` | ✅ Valid | ✅ Valid |
| `3.14` | ❌ Invalid | ✅ Valid |
| `1.0` | ✅ Valid (zero fractional part) | ✅ Valid |
| Python equivalent | `int` | `float` |
| Use when | Counting, IDs, indices | Measurements, prices, coordinates |

> **🔑 Key concept:** Use `integer` as the default for whole numbers. It's more restrictive, which means the model is less likely to send unexpected values like `3.5` when you expect `3`.

---

## Boolean type

Booleans are straightforward — `true` or `false`. They're the simplest type but easy to misuse.

### Basic boolean

```python
import json

# Boolean properties
boolean_properties = {
    "is_urgent": {
        "type": "boolean",
        "description": "Whether the task requires immediate attention"
    },
    "include_metadata": {
        "type": "boolean",
        "description": "If true, include file metadata in the response"
    },
    "dry_run": {
        "type": "boolean",
        "description": "If true, simulate the action without making changes"
    }
}

for name, schema in boolean_properties.items():
    print(f"{name}: {schema['description']}")
```

**Output:**
```
is_urgent: Whether the task requires immediate attention
include_metadata: If true, include file metadata in the response
dry_run: If true, simulate the action without making changes
```

### When to use boolean vs. enum

Sometimes a parameter looks like it should be boolean but actually has more than two states:

```python
import json

# ❌ Avoid: boolean when there are more than two states
bad_visibility = {
    "is_public": {
        "type": "boolean",
        "description": "Whether the post is public"
        # What about "unlisted" or "followers only"?
    }
}

# ✅ Better: enum when there are multiple states
good_visibility = {
    "visibility": {
        "type": "string",
        "enum": ["public", "private", "unlisted", "followers_only"],
        "description": "Post visibility level"
    }
}

# ❌ Avoid: boolean for tri-state values
bad_sort = {
    "sort_ascending": {
        "type": "boolean",
        "description": "Sort in ascending order"
        # What about "no sorting at all"?
    }
}

# ✅ Better: enum for tri-state
good_sort = {
    "sort_order": {
        "type": "string",
        "enum": ["ascending", "descending", "none"],
        "description": "Sort order for results"
    }
}

print("Boolean is right when there are exactly two states:")
print("  • is_active: true/false")
print("  • include_headers: true/false")
print("  • dry_run: true/false")
print()
print("Enum is right when there are more than two states:")
print("  • visibility: public/private/unlisted/followers_only")
print("  • sort_order: ascending/descending/none")
print("  • priority: low/medium/high/critical")
```

**Output:**
```
Boolean is right when there are exactly two states:
  • is_active: true/false
  • include_headers: true/false
  • dry_run: true/false

Enum is right when there are more than two states:
  • visibility: public/private/unlisted/followers_only
  • sort_order: ascending/descending/none
  • priority: low/medium/high/critical
```

---

## Type selection guide

Here's a quick reference for choosing the right type for common parameters:

| Parameter | Type | Constraints | Example Value |
|-----------|------|-------------|---------------|
| Name, title, message | `string` | `maxLength` | `"Buy groceries"` |
| Date | `string` | `format: "date"` | `"2025-03-15"` |
| Email | `string` | `format: "email"` | `"user@example.com"` |
| URL | `string` | `format: "uri"` | `"https://example.com"` |
| Fixed options | `string` | `enum` | `"high"` |
| Phone number | `string` | `pattern` | `"+14155552671"` |
| Price / weight / temp | `number` | `minimum`, `maximum` | `29.99` |
| Latitude / longitude | `number` | `minimum`, `maximum` | `37.7749` |
| Count / quantity / age | `integer` | `minimum`, `maximum` | `42` |
| Page number / offset | `integer` | `minimum: 1` | `3` |
| Year | `integer` | `minimum`, `maximum` | `2025` |
| Feature flag | `boolean` | — | `true` |
| List of strings | `array` | `items: {type: "string"}` | `["tag1", "tag2"]` |
| Nested data | `object` | `properties` | `{"street": "..."}` |

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use `integer` for whole numbers, `number` for decimals | Prevents the model from sending `3.5` when you need `3` |
| Always add `format` for dates, emails, and URLs | Makes the expected format explicit to the model |
| Use `enum` instead of boolean when >2 states exist | Avoids ambiguity and future expansion problems |
| Include example values in descriptions | The model learns from examples in the description |
| Set range constraints on numeric types | Prevents extreme or negative values where they don't make sense |
| Use `pattern` sparingly | Complex regex increases schema processing time |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `string` for dates without `format` | Add `"format": "date"` and an example in the description |
| Using `number` when you need `integer` | Use `integer` for counts, IDs, ages, and quantities |
| Using `boolean` for multi-state values | Use `enum` with descriptive values |
| Not setting `minimum: 0` on counts/prices | The model might generate negative values |
| Using `pattern` without anchoring (`^...$`) | The pattern matches substrings, not the whole string |
| Relying on `format` without a description example | Always include an example value in the description |

---

## Hands-on exercise

### Your task

Create a function schema for a `create_invoice` function with these parameters:

1. `customer_email` — email address (required)
2. `invoice_date` — date in ISO format (required)
3. `due_date` — date in ISO format (required)
4. `currency` — one of: USD, EUR, GBP, JPY (required)
5. `subtotal` — positive decimal number (required)
6. `tax_rate` — percentage from 0 to 100, multiples of 0.5 (required)
7. `line_item_count` — integer, at least 1 (required)
8. `is_recurring` — boolean (required)
9. `notes` — optional string, max 500 characters
10. `po_number` — optional string matching pattern `PO-XXXXX` (5 digits)

### Requirements

1. Use the correct type for each parameter
2. Apply appropriate constraints (format, pattern, minimum, maximum, etc.)
3. Include descriptive descriptions with examples
4. Make it strict-mode compatible (all required, null unions for optional)

### Expected result

A schema that produces valid output like:

```json
{
  "customer_email": "client@company.com",
  "invoice_date": "2025-04-01",
  "due_date": "2025-05-01",
  "currency": "USD",
  "subtotal": 1500.00,
  "tax_rate": 8.5,
  "line_item_count": 3,
  "is_recurring": false,
  "notes": "Q2 consulting services",
  "po_number": "PO-12345"
}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `format: "email"` for the email field
- Use `format: "date"` for the date fields
- Use `enum` for currency
- Use `multipleOf: 0.5` for the tax rate
- Use `integer` with `minimum: 1` for line item count
- Use `pattern: "^PO-\\d{5}$"` for the PO number
- For optional fields in strict mode, use `["string", "null"]` and add `null` to any enums

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

create_invoice_tool = {
    "type": "function",
    "function": {
        "name": "create_invoice",
        "description": "Create a new invoice for a customer",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "customer_email": {
                    "type": "string",
                    "format": "email",
                    "description": "Customer's email address, e.g., 'client@company.com'"
                },
                "invoice_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Invoice date in ISO format, e.g., '2025-04-01'"
                },
                "due_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Payment due date in ISO format, e.g., '2025-05-01'"
                },
                "currency": {
                    "type": "string",
                    "enum": ["USD", "EUR", "GBP", "JPY"],
                    "description": "Invoice currency code"
                },
                "subtotal": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Invoice subtotal before tax, must be positive"
                },
                "tax_rate": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "multipleOf": 0.5,
                    "description": "Tax rate as a percentage (0-100), in 0.5% increments, e.g., 8.5"
                },
                "line_item_count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of line items on the invoice"
                },
                "is_recurring": {
                    "type": "boolean",
                    "description": "Whether this is a recurring invoice"
                },
                "notes": {
                    "type": ["string", "null"],
                    "maxLength": 500,
                    "description": "Optional invoice notes, max 500 characters, or null"
                },
                "po_number": {
                    "type": ["string", "null"],
                    "pattern": "^PO-\\d{5}$",
                    "description": "Purchase order number in PO-XXXXX format, e.g., 'PO-12345', or null"
                }
            },
            "required": [
                "customer_email", "invoice_date", "due_date",
                "currency", "subtotal", "tax_rate",
                "line_item_count", "is_recurring", "notes", "po_number"
            ],
            "additionalProperties": False
        }
    }
}

print(json.dumps(create_invoice_tool, indent=2))
```

</details>

### Bonus challenges

- [ ] Add a `discount_percentage` parameter (number, 0-100, optional)
- [ ] Add a `payment_method` enum with appropriate values
- [ ] Convert the schema for Anthropic and Gemini formats

---

## Summary

✅ Strings support `format`, `pattern`, `minLength`, and `maxLength` constraints for precise validation

✅ Use `integer` for whole numbers and `number` for decimals — choosing correctly prevents type errors

✅ Range constraints (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`) prevent invalid values

✅ Use `boolean` only for true binary states — use `enum` for multi-state values

✅ Always include example values in `description` to guide the model toward correct formatting

**Next:** [Required vs. Optional Parameters](./04-required-optional-parameters.md) — Handling required fields and optional parameters across providers

---

## Further reading

- [JSON Schema String Reference](https://json-schema.org/understanding-json-schema/reference/string) — String constraints and formats
- [JSON Schema Numeric Reference](https://json-schema.org/understanding-json-schema/reference/numeric) — Number and integer details
- [OpenAI Structured Outputs — Supported Schemas](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) — Which constraints work in strict mode

---

[← Previous: Strict Mode Requirements](./02-strict-mode-requirements.md) | [Next: Required vs. Optional Parameters →](./04-required-optional-parameters.md)

<!-- 
Sources Consulted:
- JSON Schema Reference (string): https://json-schema.org/understanding-json-schema/reference/string
- JSON Schema Reference (numeric): https://json-schema.org/understanding-json-schema/reference/numeric
- JSON Schema Reference (type): https://json-schema.org/understanding-json-schema/reference/type
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
