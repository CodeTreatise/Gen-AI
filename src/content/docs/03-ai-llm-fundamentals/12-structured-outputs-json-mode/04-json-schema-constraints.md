---
title: "JSON Schema Constraints"
---

# JSON Schema Constraints

## Introduction

Structured outputs support a subset of JSON Schema. Understanding the supported features and limitations helps you design effective schemas.

### What We'll Cover

- Supported types and constraints
- Property limits
- Nesting depth limits
- Recursive schemas
- additionalProperties handling

---

## Supported Types

### Basic Types

```python
supported_types = {
    "string": {
        "description": "Text values",
        "constraints": ["enum"],
        "example": {"type": "string"}
    },
    "number": {
        "description": "Floating-point numbers",
        "constraints": ["enum"],
        "example": {"type": "number"}
    },
    "integer": {
        "description": "Whole numbers",
        "constraints": ["enum"],
        "example": {"type": "integer"}
    },
    "boolean": {
        "description": "true or false",
        "constraints": [],
        "example": {"type": "boolean"}
    },
    "null": {
        "description": "Null value",
        "constraints": [],
        "example": {"type": "null"}
    },
    "array": {
        "description": "Ordered list",
        "constraints": ["items"],
        "example": {"type": "array", "items": {"type": "string"}}
    },
    "object": {
        "description": "Key-value pairs",
        "constraints": ["properties", "required", "additionalProperties"],
        "example": {"type": "object", "properties": {...}}
    }
}
```

### Type Examples

```python
# String
{"type": "string"}

# String with enum
{"type": "string", "enum": ["small", "medium", "large"]}

# Number
{"type": "number"}

# Integer
{"type": "integer"}

# Boolean
{"type": "boolean"}

# Array of strings
{"type": "array", "items": {"type": "string"}}

# Array of objects
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        },
        "required": ["name", "value"],
        "additionalProperties": False
    }
}
```

---

## anyOf for Union Types

### Nullable Fields

```python
# Field that can be string or null
nullable_string = {
    "anyOf": [
        {"type": "string"},
        {"type": "null"}
    ]
}

# Complete example
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "nickname": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        }
    },
    "required": ["name", "nickname"],
    "additionalProperties": False
}
```

### Union of Different Types

```python
# Response can be success or error
result_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success"]},
                "data": {"type": "string"}
            },
            "required": ["status", "data"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["error"]},
                "message": {"type": "string"},
                "code": {"type": "integer"}
            },
            "required": ["status", "message", "code"],
            "additionalProperties": False
        }
    ]
}
```

---

## Property Limits

### Maximum Properties

```
┌─────────────────────────────────────────────────────────────┐
│                   PROPERTY LIMITS                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Maximum total properties: 5,000                             │
│                                                              │
│  Counts include:                                             │
│  ├── Top-level properties                                   │
│  ├── Nested object properties                               │
│  └── Array item object properties                           │
│                                                              │
│  Example counting:                                           │
│  {                                                           │
│    "user": {           ← 1 property                         │
│      "name": ...,      ← 1 property                         │
│      "address": {      ← 1 property                         │
│        "street": ...,  ← 1 property                         │
│        "city": ...     ← 1 property                         │
│      }                                                       │
│    }                                                         │
│  }                                                           │
│  Total: 5 properties                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Practical Implications

```python
# This is fine (few properties)
simple_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "value": {"type": "number"}
    }
}

# This could hit limits (many repeated structures)
large_schema = {
    "type": "object",
    "properties": {
        f"field_{i}": {"type": "string"}
        for i in range(1000)  # Still under 5000
    }
}

# Consider: Use arrays instead of numbered properties
better_schema = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["name", "value"],
                "additionalProperties": False
            }
        }
    },
    "required": ["fields"],
    "additionalProperties": False
}
```

---

## Nesting Depth

### Maximum Depth: 10 Levels

```python
# Level 1
{
    "type": "object",
    "properties": {
        "level1": {  # Level 2
            "type": "object",
            "properties": {
                "level2": {  # Level 3
                    "type": "object",
                    "properties": {
                        "level3": {  # Level 4
                            # ... up to level 10
                        }
                    }
                }
            }
        }
    }
}
```

### Counting Nesting

```python
nesting_rules = {
    "object_in_object": "Each nested object adds 1 level",
    "array_of_objects": "Array + items object counts as 2 levels",
    "anyOf": "Each branch starts from current level"
}

# Example: This is 4 levels deep
schema = {
    "type": "object",  # Level 1
    "properties": {
        "items": {
            "type": "array",  # Level 2
            "items": {
                "type": "object",  # Level 3
                "properties": {
                    "meta": {
                        "type": "object",  # Level 4
                        "properties": {
                            "tag": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}
```

### Flattening Deep Structures

```python
# Instead of deeply nested
deep_schema = {
    "company": {
        "department": {
            "team": {
                "member": {
                    "contact": {
                        "email": "..."
                    }
                }
            }
        }
    }
}

# Use flattened with references
flat_schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "department_name": {"type": "string"},
        "team_name": {"type": "string"},
        "member_name": {"type": "string"},
        "member_email": {"type": "string"}
    }
}
```

---

## Recursive Schemas

### Using $ref for Recursion

```python
# Tree structure with recursive nodes
recursive_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "children": {
            "type": "array",
            "items": {"$ref": "#"}  # Reference to root
        }
    },
    "required": ["name", "children"],
    "additionalProperties": False
}

# Example output:
# {
#   "name": "root",
#   "children": [
#     {
#       "name": "child1",
#       "children": []
#     },
#     {
#       "name": "child2",
#       "children": [
#         {"name": "grandchild", "children": []}
#       ]
#     }
#   ]
# }
```

### With $defs

```python
schema_with_defs = {
    "$defs": {
        "TreeNode": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/TreeNode"}
                }
            },
            "required": ["name", "value", "children"],
            "additionalProperties": False
        }
    },
    "$ref": "#/$defs/TreeNode"
}
```

---

## additionalProperties

### Why It Must Be False

```python
# Strict mode requires additionalProperties: false
# This prevents the model from adding unexpected fields

# Required for strict mode
valid_strict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}
    },
    "required": ["name"],
    "additionalProperties": False  # Required!
}

# Will fail in strict mode
invalid_strict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}
    },
    "required": ["name"]
    # Missing additionalProperties: false
}
```

### All Nested Objects Need It

```python
complete_schema = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"],
                    "additionalProperties": False  # Needed here too
                }
            },
            "required": ["name", "address"],
            "additionalProperties": False  # Needed here
        }
    },
    "required": ["user"],
    "additionalProperties": False  # Needed at root
}
```

---

## Unsupported Features

### Not Supported in Strict Mode

```python
unsupported_constraints = {
    "minLength": "String minimum length",
    "maxLength": "String maximum length",
    "pattern": "Regex patterns",
    "minimum": "Number minimum value",
    "maximum": "Number maximum value",
    "minItems": "Array minimum items",
    "maxItems": "Array maximum items",
    "uniqueItems": "Array uniqueness",
    "format": "String formats (email, uri, etc.)"
}

# These are ignored in structured outputs
# Use application-level validation instead
```

### Workarounds

```python
from pydantic import BaseModel, field_validator

class ValidatedModel(BaseModel):
    score: float
    tags: list[str]
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        # Post-parse validation
        if not 0 <= v <= 100:
            raise ValueError("Score must be 0-100")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags")
        return v
```

---

## Summary

✅ **Supported types**: string, number, integer, boolean, null, array, object

✅ **anyOf**: For nullable and union types

✅ **Limits**: 5,000 properties, 10 nesting levels

✅ **Recursion**: Use $ref for recursive structures

✅ **additionalProperties**: Must be false for all objects

**Next:** [Advanced Patterns](./05-advanced-patterns.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [SDK Integrations](./03-sdk-integrations.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [Advanced Patterns](./05-advanced-patterns.md) |
