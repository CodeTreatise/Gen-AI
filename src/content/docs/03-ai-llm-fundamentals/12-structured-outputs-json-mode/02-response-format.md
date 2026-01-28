---
title: "Response Format Parameter"
---

# Response Format Parameter

## Introduction

The `response_format` parameter controls how LLMs structure their outputs. This parameter has evolved to support increasingly sophisticated output constraints.

### What We'll Cover

- Basic JSON mode configuration
- JSON Schema enforcement
- Strict mode for guaranteed compliance
- Error handling

---

## Configuration Options

### Overview

```python
response_format_options = {
    "text": "Default plain text output",
    "json_object": "Valid JSON, no schema",
    "json_schema": "Valid JSON matching provided schema"
}
```

---

## Basic JSON Mode

### Simple Configuration

```python
from openai import OpenAI

client = OpenAI()

# Enable JSON mode
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You extract data as JSON. Include keys: name, price, category."
        },
        {"role": "user", "content": "Parse: Red sneakers, $89.99, footwear"}
    ],
    response_format={"type": "json_object"}
)

# Parse the response
import json
data = json.loads(response.choices[0].message.content)
```

### Important Requirements

```python
# JSON mode requires mentioning "JSON" in the prompt
# This will fail:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 fruits"}],
    response_format={"type": "json_object"}  # Error!
)

# This works:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 fruits as JSON"}],
    response_format={"type": "json_object"}  # OK
)
```

---

## JSON Schema Mode

### Defining a Schema

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Extract info from: Red sneakers, $89.99, footwear"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "category": {"type": "string"}
                },
                "required": ["name", "price", "category"],
                "additionalProperties": False
            }
        }
    }
)
```

### Schema Structure

```python
json_schema_structure = {
    "type": "json_schema",
    "json_schema": {
        "name": "schema_name",           # Required: identifier
        "description": "...",            # Optional: helps model understand
        "strict": True,                  # Recommended: enables strict mode
        "schema": {
            # Standard JSON Schema definition
            "type": "object",
            "properties": {...},
            "required": [...],
            "additionalProperties": False  # Required for strict mode
        }
    }
}
```

---

## Strict Mode

### What Strict Mode Does

```
┌─────────────────────────────────────────────────────────────┐
│                       STRICT MODE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  WITHOUT strict: True                                        │
│  ├── Model attempts to follow schema                        │
│  ├── May include extra fields                               │
│  ├── May omit optional fields differently                   │
│  └── No guarantee of compliance                             │
│                                                              │
│  WITH strict: True                                           │
│  ├── Schema compiled to constrained grammar                 │
│  ├── 100% schema compliance guaranteed                      │
│  ├── No extra fields possible                               │
│  └── All fields exactly as specified                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Strict Mode Requirements

```python
strict_mode_requirements = {
    "additionalProperties": "Must be False for all objects",
    "required": "All properties must be listed",
    "supported_types": [
        "string", "number", "integer", "boolean",
        "object", "array", "enum", "anyOf", "null"
    ],
    "max_properties": 5000,
    "max_nesting": 10
}
```

### Complete Example

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Analyze sentiment: 'This product is amazing!'"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The analyzed text"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0-1"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["text", "sentiment", "confidence", "keywords"],
                "additionalProperties": False
            }
        }
    }
)
```

**Guaranteed output:**
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.95,
  "keywords": ["amazing", "product"]
}
```

---

## Working with Enums

### String Enums

```python
schema = {
    "type": "object",
    "properties": {
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"]
        },
        "status": {
            "type": "string", 
            "enum": ["open", "in_progress", "resolved", "closed"]
        }
    },
    "required": ["priority", "status"],
    "additionalProperties": False
}
```

### Numeric Enums

```python
schema = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5]
        }
    },
    "required": ["rating"],
    "additionalProperties": False
}
```

---

## Handling Optionals with anyOf

### Nullable Fields

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        }
    },
    "required": ["name", "email"],
    "additionalProperties": False
}
```

### Union Types

```python
schema = {
    "type": "object",
    "properties": {
        "result": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"type": "string"}
                    },
                    "required": ["success", "data"],
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "error": {"type": "string"}
                    },
                    "required": ["success", "error"],
                    "additionalProperties": False
                }
            ]
        }
    },
    "required": ["result"],
    "additionalProperties": False
}
```

---

## Error Handling

### Common Errors

```python
def handle_response_format_errors():
    """Handle common response format errors"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "..."}],
            response_format={"type": "json_schema", "json_schema": {...}}
        )
        
    except openai.BadRequestError as e:
        if "schema" in str(e):
            # Invalid schema definition
            print("Check schema structure")
        elif "json" in str(e).lower():
            # JSON not mentioned in prompt
            print("Add JSON instruction to prompt")
            
    except openai.APIError as e:
        # General API error
        print(f"API error: {e}")
```

### Validation Errors

```python
# Schema too complex
schema_limits = {
    "max_properties": 5000,
    "max_nesting_depth": 10,
    "max_enum_values": "reasonable limit"
}

# Check for strict mode violations
def validate_strict_schema(schema: dict) -> list[str]:
    """Validate schema for strict mode compliance"""
    errors = []
    
    if schema.get("type") == "object":
        if "additionalProperties" not in schema:
            errors.append("Missing additionalProperties: false")
        if schema.get("additionalProperties") != False:
            errors.append("additionalProperties must be false")
        
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for prop in properties:
            if prop not in required:
                errors.append(f"Property '{prop}' not in required list")
    
    return errors
```

---

## Summary

✅ **json_object**: Simple valid JSON output

✅ **json_schema**: Schema-enforced output

✅ **strict: true**: Guarantees 100% compliance

✅ **Enums**: Constrain to specific values

✅ **anyOf**: Handle nullable and union types

**Next:** [SDK Integrations](./03-sdk-integrations.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [JSON Mode vs Structured](./01-json-mode-vs-structured.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [SDK Integrations](./03-sdk-integrations.md) |
