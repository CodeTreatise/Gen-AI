---
title: "JSON Output Specification"
---

# JSON Output Specification

## Introduction

JSON is the most common format for structured AI outputs. When you need to parse model responses programmatically, JSON provides a reliable, language-agnostic format. This lesson covers how to specify JSON schemas in prompts, handle complex nested structures, and ensure valid JSON output.

> **🤖 AI Context:** JSON outputs enable direct integration with applications. Unlike free-form text, JSON can be parsed immediately into data structures without regex or manual extraction.

### What We'll Cover

- JSON schema descriptions in prompts
- Required vs optional fields
- Field types and validation
- Nested object structures
- Examples that guide output

### Prerequisites

- [Requesting Structured Outputs](./02-requesting-structured-outputs.md)

---

## Why JSON for AI Outputs

| Benefit | Description |
|---------|-------------|
| **Universal parsing** | Every language has JSON support |
| **Type preservation** | Numbers, booleans, arrays stay typed |
| **Nested structures** | Complex data hierarchies possible |
| **Validation** | JSON Schema can validate outputs |
| **API compatible** | Direct use in REST responses |

---

## Basic JSON Specification

### Simple Schema in Prompt

```markdown
# Instructions

Analyze the sentiment of the review.

# Output Format

Return a JSON object with these fields:
- sentiment: string ("positive", "negative", or "neutral")
- confidence: number (0.0 to 1.0)
- reasoning: string (one sentence explanation)

Example:
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "reasoning": "The review uses enthusiastic language and recommends the product."
}
```
```

### Complete Field Specification

For reliable outputs, specify every aspect of each field:

```markdown
# JSON Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| id | string | yes | UUID v4 format | Unique identifier |
| title | string | yes | 1-100 chars | Document title |
| score | number | yes | 0.0-1.0 | Relevance score |
| tags | array | yes | 1-10 items | Relevant tags |
| author | string | no | null if unknown | Author name |
```

---

## Field Types

Be explicit about expected types:

### Primitive Types

```markdown
# Field Types

- name: string (text in quotes)
- age: integer (whole number)
- price: number (decimal allowed)
- active: boolean (true or false, not quoted)
- id: null (when unknown)
```

### Array Types

```markdown
# Array Fields

- tags: array of strings
  Example: ["javascript", "web", "tutorial"]

- scores: array of numbers
  Example: [0.8, 0.65, 0.92]

- items: array of objects (see Item schema below)
```

### Enum Types

```markdown
# Enum Fields

The status field must be exactly one of:
- "pending"
- "approved"
- "rejected"

Do NOT use: "Pending", "APPROVED", or any variations.
```

---

## Required vs Optional Fields

### Handling Missing Data

```markdown
# Field Requirements

Required fields (must always be present):
- name: string
- email: string
- created_at: ISO 8601 date string

Optional fields (use null if not available):
- phone: string | null
- company: string | null
- notes: string | null

Example with missing optional data:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2025-01-15T10:30:00Z",
  "phone": null,
  "company": null,
  "notes": null
}
```
```

### Default Values

```markdown
# Defaults

If information is not provided:
- priority: default to "medium"
- active: default to true
- tags: default to empty array []
- count: default to 0
```

---

## Nested Object Structures

### Defining Nested Objects

```markdown
# Output Schema

```json
{
  "user": {
    "id": "string (UUID)",
    "profile": {
      "firstName": "string",
      "lastName": "string",
      "email": "string (valid email format)"
    }
  },
  "preferences": {
    "theme": "light | dark",
    "notifications": {
      "email": "boolean",
      "push": "boolean"
    }
  }
}
```
```

### Object Arrays

```markdown
# Items Array Schema

Each item in the "products" array should have:

```json
{
  "products": [
    {
      "sku": "string (product code)",
      "name": "string (product name)",
      "price": "number (in USD)",
      "quantity": "integer (1 or more)"
    }
  ]
}
```

Example:
```json
{
  "products": [
    {"sku": "ABC123", "name": "Widget", "price": 29.99, "quantity": 2},
    {"sku": "XYZ789", "name": "Gadget", "price": 49.99, "quantity": 1}
  ]
}
```
```

---

## JSON Examples in Prompts

Examples are the most effective way to communicate expected format:

### Single Example

```markdown
# Output Format

Return a JSON object matching this example:

```json
{
  "summary": "A 2-3 sentence summary of the document",
  "keyPoints": ["First key point", "Second key point", "Third key point"],
  "sentiment": "positive",
  "wordCount": 350
}
```
```

### Multiple Examples for Edge Cases

```markdown
# Examples

Standard case:
```json
{
  "found": true,
  "result": {
    "name": "John Smith",
    "email": "john@example.com"
  }
}
```

Not found case:
```json
{
  "found": false,
  "result": null
}
```

Partial data case:
```json
{
  "found": true,
  "result": {
    "name": "Jane Doe",
    "email": null
  }
}
```
```

---

## Formal JSON Schema

For complex applications, use JSON Schema notation:

### Inline JSON Schema

```markdown
# Output Specification (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["name", "email", "status"],
  "properties": {
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100
    },
    "email": {
      "type": "string",
      "format": "email"
    },
    "status": {
      "type": "string",
      "enum": ["active", "inactive", "pending"]
    },
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 10
    }
  },
  "additionalProperties": false
}
```
```

### Simplified Schema Reference

```markdown
# Schema (simplified notation)

{
  name: string (required, 1-100 chars)
  email: string (required, email format)
  status: "active" | "inactive" | "pending" (required)
  tags?: string[] (optional, max 10 items)
}
```

---

## Ensuring Valid JSON

### Common JSON Errors

| Error | Cause | Prevention |
|-------|-------|------------|
| Trailing comma | Extra comma after last item | Show example without trailing comma |
| Unquoted keys | `{name: "value"}` | Always show `{"name": "value"}` |
| Single quotes | `{'key': 'value'}` | Emphasize double quotes only |
| Unescaped quotes | `"He said "hello""` | Show escaped: `"He said \"hello\""` |
| Comments | `// comment` in JSON | State: "No comments allowed" |

### Explicit Instructions

```markdown
# JSON Requirements

- Use double quotes for all strings and keys
- No trailing commas after the last item
- No comments (// or /* */)
- Escape special characters in strings:
  - Quote: \"
  - Backslash: \\
  - Newline: \n
- Use null (not "null") for missing values
```

---

## Output Isolation

Prevent extra text from mixing with JSON:

### JSON-Only Output

```markdown
# Output

Return ONLY the JSON object. Do not include:
- Explanations before or after
- Markdown code block markers
- The word "json" or any labels

Start your response with { and end with }
```

### Wrapped Output

```markdown
# Output Format

You may provide brief reasoning, then output the JSON in a code block:

```json
{your JSON here}
```

I will extract only the content within the code block.
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Show complete examples | Models copy structure precisely |
| Specify all field types | Prevents type confusion |
| Handle null/missing cases | Ensures valid JSON always |
| Use consistent naming | camelCase or snake_case, not mixed |
| Avoid deep nesting | 3-4 levels max for clarity |
| Include edge case examples | Covers unusual situations |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| "Return JSON" (no schema) | Provide full schema with example |
| Allowing extra fields | Specify `additionalProperties: false` |
| No type for numbers | State integer vs decimal explicitly |
| Missing array item schema | Define what each array item contains |
| Assuming ISO dates | Specify: `"date": "YYYY-MM-DD"` |

---

## Hands-on Exercise

### Your Task

Create a JSON schema for an API that extracts meeting information from natural language.

### Input Example

```
"Let's meet tomorrow at 3pm in the downtown office to discuss Q4 planning. 
Invite Sarah, Mike, and the marketing team. It should take about 2 hours."
```

### Requirements

1. Extract: title, date/time, location, attendees, duration
2. Handle missing information with null
3. Attendees should support both names and groups
4. Duration in minutes

<details>
<summary>💡 Hints (click to expand)</summary>

- How do you represent "tomorrow"? Relative vs absolute?
- Should attendees be a simple array or have structure?
- What's the difference between a person and a group?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```markdown
# Instructions

Extract meeting information from the text and return structured JSON.

# Output Schema

```json
{
  "title": "string - Meeting subject/purpose",
  "datetime": {
    "relative": "string - Original time reference (e.g., 'tomorrow at 3pm')",
    "parsed": "string | null - ISO 8601 if parseable, otherwise null"
  },
  "location": "string | null - Meeting location",
  "duration_minutes": "number | null - Duration in minutes",
  "attendees": [
    {
      "name": "string - Person or group name",
      "type": "person | group"
    }
  ]
}
```

# Example

Input: "Quick sync with Dev team on Friday at 10am, 30 minutes, virtual"

Output:
```json
{
  "title": "Quick sync",
  "datetime": {
    "relative": "Friday at 10am",
    "parsed": null
  },
  "location": "virtual",
  "duration_minutes": 30,
  "attendees": [
    {"name": "Dev team", "type": "group"}
  ]
}
```

# Text to Parse

"Let's meet tomorrow at 3pm in the downtown office to discuss Q4 planning. 
Invite Sarah, Mike, and the marketing team. It should take about 2 hours."
```

**Expected output:**
```json
{
  "title": "Q4 planning discussion",
  "datetime": {
    "relative": "tomorrow at 3pm",
    "parsed": null
  },
  "location": "downtown office",
  "duration_minutes": 120,
  "attendees": [
    {"name": "Sarah", "type": "person"},
    {"name": "Mike", "type": "person"},
    {"name": "marketing team", "type": "group"}
  ]
}
```

</details>

### Bonus Challenge

- [ ] Add a `priority` field (inferred from urgency language)
- [ ] Include a `requires_preparation` boolean based on meeting type

---

## Summary

✅ **Explicit schemas** prevent format variations

✅ **Type specifications** ensure parseable data

✅ **Null handling** for missing information

✅ **Examples** are the most effective format guide

✅ **Output isolation** separates JSON from reasoning

**Next:** [Markdown Formatting Requests](./04-markdown-formatting-requests.md)

---

## Further Reading

- [JSON Schema](https://json-schema.org/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic for Python Validation](https://docs.pydantic.dev/)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- JSON Schema Specification: https://json-schema.org/
-->
