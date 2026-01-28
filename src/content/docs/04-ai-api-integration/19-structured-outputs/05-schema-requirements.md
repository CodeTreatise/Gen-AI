---
title: "Schema Requirements"
---

# Schema Requirements

## Introduction

Structured Outputs requires schemas to follow specific rules. Understanding these requirements ensures your schemas work correctly and your outputs are guaranteed to match.

### What We'll Cover

- Required schema properties
- The `additionalProperties: false` rule
- All fields must be in `required`
- Supported types and formats
- Handling optional fields with null unions

### Prerequisites

- JSON Schema basics
- Pydantic or Zod experience
- Understanding of type systems

---

## Required Schema Properties

### The Three Rules

```python
from dataclasses import dataclass
from typing import Dict, Any, List
import json


@dataclass
class SchemaRule:
    """Schema rule for Structured Outputs."""
    
    rule: str
    requirement: str
    example_good: str
    example_bad: str
    error_if_violated: str


SCHEMA_RULES = [
    SchemaRule(
        rule="additionalProperties",
        requirement="Must be false for all objects",
        example_good='"additionalProperties": false',
        example_bad='missing or "additionalProperties": true',
        error_if_violated="API error: additionalProperties must be false"
    ),
    SchemaRule(
        rule="required",
        requirement="Must list ALL property names",
        example_good='"required": ["name", "age", "email"]',
        example_bad='"required": ["name"]  // missing age, email',
        error_if_violated="API error: all properties must be required"
    ),
    SchemaRule(
        rule="strict",
        requirement="Must be true in json_schema config",
        example_good='"strict": true',
        example_bad='missing strict or "strict": false',
        error_if_violated="Falls back to JSON mode, no schema guarantee"
    )
]


print("The Three Schema Rules")
print("=" * 60)

for i, rule in enumerate(SCHEMA_RULES, 1):
    print(f"\n{i}. {rule.rule.upper()}")
    print(f"   Requirement: {rule.requirement}")
    print(f"   ✅ Good: {rule.example_good}")
    print(f"   ❌ Bad: {rule.example_bad}")
    print(f"   Error: {rule.error_if_violated}")
```

### Valid vs Invalid Schema Examples

```python
# VALID schema - all rules followed
VALID_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Person's name"
        },
        "age": {
            "type": "integer",
            "description": "Age in years"
        },
        "email": {
            "type": "string",
            "description": "Email address"
        }
    },
    "required": ["name", "age", "email"],  # All properties listed
    "additionalProperties": False  # Must be false
}


# INVALID schema - missing additionalProperties
INVALID_SCHEMA_1 = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
    # Missing additionalProperties: false ❌
}


# INVALID schema - incomplete required
INVALID_SCHEMA_2 = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}  # This property exists...
    },
    "required": ["name", "age"],  # ...but not listed here ❌
    "additionalProperties": False
}


def validate_schema(schema: dict, path: str = "root") -> List[str]:
    """Validate schema against Structured Outputs rules."""
    
    errors = []
    
    if schema.get("type") == "object":
        # Check additionalProperties
        if schema.get("additionalProperties") != False:
            errors.append(f"{path}: additionalProperties must be false")
        
        # Check required lists all properties
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        for prop_name in properties:
            if prop_name not in required:
                errors.append(f"{path}: property '{prop_name}' not in required")
        
        # Recursively check nested objects
        for prop_name, prop_def in properties.items():
            if prop_def.get("type") == "object":
                errors.extend(validate_schema(prop_def, f"{path}.{prop_name}"))
            elif prop_def.get("type") == "array":
                items = prop_def.get("items", {})
                if items.get("type") == "object":
                    errors.extend(validate_schema(items, f"{path}.{prop_name}[]"))
    
    return errors


print("\n\nSchema Validation")
print("=" * 60)

print("\n✅ Valid Schema:")
errors = validate_schema(VALID_SCHEMA)
print(f"   Errors: {errors if errors else 'None'}")

print("\n❌ Invalid Schema 1 (missing additionalProperties):")
errors = validate_schema(INVALID_SCHEMA_1)
for e in errors:
    print(f"   - {e}")

print("\n❌ Invalid Schema 2 (incomplete required):")
errors = validate_schema(INVALID_SCHEMA_2)
for e in errors:
    print(f"   - {e}")
```

---

## The additionalProperties Rule

### Why It's Required

```python
class AdditionalPropertiesExplainer:
    """Explains additionalProperties requirement."""
    
    @staticmethod
    def explain():
        return """
additionalProperties: false is REQUIRED for Structured Outputs.

WHY?
----
1. Structured Outputs guarantees ONLY the properties you define
2. With additionalProperties: true, model could add any extra fields
3. This breaks the deterministic guarantee

WHAT IT MEANS:
--------------
- Model can ONLY output properties in your schema
- No extra fields will ever appear
- Output structure is 100% predictable

NESTED OBJECTS:
---------------
Every nested object ALSO needs additionalProperties: false!
"""


# Show nested object requirement
NESTED_SCHEMA = {
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
                        "zip": {"type": "string"}
                    },
                    "required": ["city", "zip"],
                    "additionalProperties": False  # Needed here too!
                }
            },
            "required": ["name", "address"],
            "additionalProperties": False  # And here!
        }
    },
    "required": ["user"],
    "additionalProperties": False  # And at root!
}


print(AdditionalPropertiesExplainer.explain())

print("\nNested Object Schema (all levels need additionalProperties: false)")
print(json.dumps(NESTED_SCHEMA, indent=2))
```

---

## Supported Types

### Basic Types

```python
from typing import Dict, List, Any


SUPPORTED_TYPES = {
    "string": {
        "json_schema": {"type": "string"},
        "pydantic": "str",
        "description": "Text values",
        "example": '"hello"'
    },
    "integer": {
        "json_schema": {"type": "integer"},
        "pydantic": "int",
        "description": "Whole numbers",
        "example": "42"
    },
    "number": {
        "json_schema": {"type": "number"},
        "pydantic": "float",
        "description": "Decimal numbers",
        "example": "3.14"
    },
    "boolean": {
        "json_schema": {"type": "boolean"},
        "pydantic": "bool",
        "description": "True/false values",
        "example": "true"
    },
    "array": {
        "json_schema": {"type": "array", "items": {"type": "string"}},
        "pydantic": "List[str]",
        "description": "Lists of values",
        "example": '["a", "b", "c"]'
    },
    "object": {
        "json_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "pydantic": "class MyModel(BaseModel): ...",
        "description": "Nested objects",
        "example": '{"key": "value"}'
    },
    "enum": {
        "json_schema": {"type": "string", "enum": ["a", "b", "c"]},
        "pydantic": "Literal['a', 'b', 'c'] or Enum",
        "description": "Fixed set of values",
        "example": '"a"'
    },
    "anyOf": {
        "json_schema": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "pydantic": "Optional[str]",
        "description": "One of multiple types",
        "example": '"value" or null'
    }
}


print("Supported Types")
print("=" * 60)

for type_name, info in SUPPORTED_TYPES.items():
    print(f"\n🔹 {type_name.upper()}")
    print(f"   JSON Schema: {json.dumps(info['json_schema'])}")
    print(f"   Pydantic: {info['pydantic']}")
    print(f"   Example: {info['example']}")
```

### Type Constraints

```python
# String constraints
STRING_CONSTRAINTS = {
    "type": "object",
    "properties": {
        # With pattern (regex)
        "username": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_]+$"
        },
        # With format
        "email": {
            "type": "string",
            "format": "email"
        },
        "date": {
            "type": "string",
            "format": "date"
        },
        "datetime": {
            "type": "string",
            "format": "date-time"
        },
        "uuid": {
            "type": "string",
            "format": "uuid"
        }
    },
    "required": ["username", "email", "date", "datetime", "uuid"],
    "additionalProperties": False
}


# Number constraints
NUMBER_CONSTRAINTS = {
    "type": "object",
    "properties": {
        # With min/max
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        # With exclusive bounds
        "rating": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 5
        },
        # Must be multiple of
        "quantity": {
            "type": "integer",
            "multipleOf": 1
        }
    },
    "required": ["age", "rating", "quantity"],
    "additionalProperties": False
}


# Array constraints
ARRAY_CONSTRAINTS = {
    "type": "object",
    "properties": {
        # With min/max items
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        }
    },
    "required": ["tags"],
    "additionalProperties": False
}


# Supported string formats
SUPPORTED_FORMATS = [
    "date-time",  # ISO 8601 datetime
    "time",       # Time only
    "date",       # Date only
    "duration",   # ISO 8601 duration
    "email",      # Email address
    "hostname",   # Internet hostname
    "ipv4",       # IPv4 address
    "ipv6",       # IPv6 address
    "uuid"        # UUID
]


print("\n\nType Constraints")
print("=" * 60)

print("\n📋 String with Constraints:")
print(json.dumps(STRING_CONSTRAINTS["properties"], indent=2))

print("\n📋 Number with Constraints:")
print(json.dumps(NUMBER_CONSTRAINTS["properties"], indent=2))

print("\n📋 Supported String Formats:")
for fmt in SUPPORTED_FORMATS:
    print(f"  - {fmt}")
```

---

## Optional Fields with Null Unions

### The Pattern for Optional Fields

```python
from pydantic import BaseModel, Field
from typing import Optional


# In Structured Outputs, ALL fields must be required
# But you can make them "optional" by allowing null


class UserWithOptionals(BaseModel):
    """User with optional fields using null union."""
    
    # Required field
    name: str = Field(description="User's name (required)")
    
    # Optional fields - use Optional[T] which becomes anyOf[T, null]
    email: Optional[str] = Field(
        default=None,
        description="Email address (optional)"
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number (optional)"
    )
    age: Optional[int] = Field(
        default=None,
        description="Age in years (optional)"
    )


# What the schema looks like
print("Optional Fields Pattern")
print("=" * 60)

print("\nPydantic Model:")
print("""
class User(BaseModel):
    name: str                    # Required
    email: Optional[str] = None  # Optional via null union
""")

print("\nGenerated JSON Schema:")
schema = UserWithOptionals.model_json_schema()
print(json.dumps(schema, indent=2))


# Manual schema for optional field
OPTIONAL_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "email": {
            # This is how you make a field "optional" - allow null
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        }
    },
    # BOTH fields still required!
    "required": ["name", "email"],
    "additionalProperties": False
}


print("\n\nManual Schema for Optional:")
print(json.dumps(OPTIONAL_FIELD_SCHEMA, indent=2))

print("""

KEY INSIGHT:
------------
In Structured Outputs, "optional" doesn't mean "maybe missing".
It means "will be present, but value might be null".

Field is ALWAYS in output, but can have value: null
""")
```

### Complete Optional Field Examples

```python
from typing import Optional, List


class OrderWithOptionals(BaseModel):
    """Order with various optional fields."""
    
    # Required
    order_id: str
    customer_name: str
    total: float
    items: List[str]
    
    # Optional fields
    notes: Optional[str] = None
    discount_code: Optional[str] = None
    gift_message: Optional[str] = None
    shipping_date: Optional[str] = None
    
    # Optional nested object would use Optional[NestedModel]


# Show possible outputs
POSSIBLE_OUTPUTS = [
    # All optionals are null
    {
        "order_id": "ORD-001",
        "customer_name": "Alice",
        "total": 99.99,
        "items": ["Widget", "Gadget"],
        "notes": None,
        "discount_code": None,
        "gift_message": None,
        "shipping_date": None
    },
    # Some optionals have values
    {
        "order_id": "ORD-002",
        "customer_name": "Bob",
        "total": 149.99,
        "items": ["Deluxe Widget"],
        "notes": "Leave at door",
        "discount_code": "SAVE10",
        "gift_message": None,
        "shipping_date": "2025-01-20"
    },
    # All optionals have values
    {
        "order_id": "ORD-003",
        "customer_name": "Carol",
        "total": 299.99,
        "items": ["Premium Package"],
        "notes": "Fragile",
        "discount_code": "VIP20",
        "gift_message": "Happy Birthday!",
        "shipping_date": "2025-01-15"
    }
]


print("\n\nOptional Field Outputs")
print("=" * 60)

for i, output in enumerate(POSSIBLE_OUTPUTS, 1):
    print(f"\n📦 Order {i}:")
    for key, value in output.items():
        if value is None:
            print(f"   {key}: null")
        elif isinstance(value, list):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
```

---

## Schema Definitions ($defs)

### Reusable Type Definitions

```python
# Using $defs for reusable types
SCHEMA_WITH_DEFS = {
    "type": "object",
    "properties": {
        "shipping_address": {
            "$ref": "#/$defs/Address"
        },
        "billing_address": {
            "$ref": "#/$defs/Address"
        },
        "items": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/OrderItem"
            }
        }
    },
    "required": ["shipping_address", "billing_address", "items"],
    "additionalProperties": False,
    
    # Definitions
    "$defs": {
        "Address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "zip": {"type": "string"},
                "country": {"type": "string"}
            },
            "required": ["street", "city", "state", "zip", "country"],
            "additionalProperties": False
        },
        "OrderItem": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "quantity": {"type": "integer"},
                "price": {"type": "number"}
            },
            "required": ["name", "quantity", "price"],
            "additionalProperties": False
        }
    }
}


print("Schema with Definitions ($defs)")
print("=" * 60)
print(json.dumps(SCHEMA_WITH_DEFS, indent=2))
```

---

## Hands-on Exercise

### Your Task

Build a schema validator that checks all Structured Outputs requirements and provides helpful error messages.

### Requirements

1. Validate additionalProperties on all objects
2. Check all properties are in required
3. Validate nested objects recursively
4. Provide clear, actionable error messages

<details>
<summary>💡 Hints</summary>

- Track the path for nested objects
- Check arrays with object items
- Suggest fixes in error messages
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json


@dataclass
class ValidationError:
    """Schema validation error."""
    
    path: str
    error_type: str
    message: str
    fix: str


class StructuredOutputsSchemaValidator:
    """Validates schemas for Structured Outputs compliance."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def validate(self, schema: dict) -> tuple[bool, List[ValidationError]]:
        """Validate schema and return errors."""
        
        self.errors = []
        self._validate_object(schema, "root")
        return len(self.errors) == 0, self.errors
    
    def _validate_object(self, schema: dict, path: str):
        """Validate an object schema."""
        
        if schema.get("type") != "object":
            return  # Not an object, skip
        
        # Rule 1: Check additionalProperties
        if "additionalProperties" not in schema:
            self.errors.append(ValidationError(
                path=path,
                error_type="missing_additionalProperties",
                message="additionalProperties is missing",
                fix=f'Add "additionalProperties": false to {path}'
            ))
        elif schema["additionalProperties"] != False:
            self.errors.append(ValidationError(
                path=path,
                error_type="invalid_additionalProperties",
                message=f"additionalProperties is {schema['additionalProperties']}, must be false",
                fix=f'Change "additionalProperties" to false at {path}'
            ))
        
        # Rule 2: Check required includes all properties
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        for prop_name in properties:
            if prop_name not in required:
                self.errors.append(ValidationError(
                    path=path,
                    error_type="missing_required",
                    message=f"Property '{prop_name}' not in required array",
                    fix=f'Add "{prop_name}" to the required array at {path}'
                ))
        
        # Check for properties in required but not in properties
        for req_name in required:
            if req_name not in properties:
                self.errors.append(ValidationError(
                    path=path,
                    error_type="extra_required",
                    message=f"'{req_name}' in required but not in properties",
                    fix=f'Remove "{req_name}" from required or add it to properties'
                ))
        
        # Recursively validate nested objects
        for prop_name, prop_def in properties.items():
            prop_path = f"{path}.{prop_name}"
            
            # Direct object
            if prop_def.get("type") == "object":
                self._validate_object(prop_def, prop_path)
            
            # Array of objects
            elif prop_def.get("type") == "array":
                items = prop_def.get("items", {})
                if items.get("type") == "object":
                    self._validate_object(items, f"{prop_path}[]")
            
            # anyOf with objects
            elif "anyOf" in prop_def:
                for i, option in enumerate(prop_def["anyOf"]):
                    if option.get("type") == "object":
                        self._validate_object(option, f"{prop_path}.anyOf[{i}]")
        
        # Check $defs
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                if def_schema.get("type") == "object":
                    self._validate_object(def_schema, f"$defs.{def_name}")
    
    def format_report(self) -> str:
        """Format validation errors as a report."""
        
        if not self.errors:
            return "✅ Schema is valid for Structured Outputs"
        
        lines = [
            f"❌ Found {len(self.errors)} validation error(s):",
            ""
        ]
        
        for i, error in enumerate(self.errors, 1):
            lines.append(f"{i}. {error.error_type.upper()}")
            lines.append(f"   Path: {error.path}")
            lines.append(f"   Error: {error.message}")
            lines.append(f"   Fix: {error.fix}")
            lines.append("")
        
        return "\n".join(lines)
    
    def auto_fix(self, schema: dict) -> dict:
        """Attempt to auto-fix schema issues."""
        
        return self._fix_object(schema.copy())
    
    def _fix_object(self, schema: dict) -> dict:
        """Fix an object schema."""
        
        if schema.get("type") != "object":
            return schema
        
        schema = schema.copy()
        
        # Fix additionalProperties
        schema["additionalProperties"] = False
        
        # Fix required
        properties = schema.get("properties", {})
        schema["required"] = list(properties.keys())
        
        # Fix nested objects
        if "properties" in schema:
            schema["properties"] = {
                name: self._fix_property(prop)
                for name, prop in schema["properties"].items()
            }
        
        # Fix $defs
        if "$defs" in schema:
            schema["$defs"] = {
                name: self._fix_object(def_schema)
                if def_schema.get("type") == "object"
                else def_schema
                for name, def_schema in schema["$defs"].items()
            }
        
        return schema
    
    def _fix_property(self, prop: dict) -> dict:
        """Fix a property definition."""
        
        prop = prop.copy()
        
        if prop.get("type") == "object":
            return self._fix_object(prop)
        
        if prop.get("type") == "array":
            items = prop.get("items", {})
            if items.get("type") == "object":
                prop["items"] = self._fix_object(items)
        
        if "anyOf" in prop:
            prop["anyOf"] = [
                self._fix_object(opt) if opt.get("type") == "object" else opt
                for opt in prop["anyOf"]
            ]
        
        return prop


# Test the validator
validator = StructuredOutputsSchemaValidator()

# Schema with multiple issues
BAD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "address": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "string"}
            },
            "required": ["city"]
            # Missing: additionalProperties, zip not in required
        }
    },
    "required": ["name"]
    # Missing: age, address, additionalProperties
}


print("Schema Validator")
print("=" * 60)

print("\n📋 Input Schema (with issues):")
print(json.dumps(BAD_SCHEMA, indent=2))

valid, errors = validator.validate(BAD_SCHEMA)
print(f"\n{validator.format_report()}")

# Auto-fix
fixed_schema = validator.auto_fix(BAD_SCHEMA)

print("\n📋 Auto-Fixed Schema:")
print(json.dumps(fixed_schema, indent=2))

# Validate fixed schema
valid, errors = validator.validate(fixed_schema)
print(f"\nAfter fix: {validator.format_report()}")
```

</details>

---

## Summary

✅ `additionalProperties: false` is required for all objects  
✅ All properties must be listed in `required` array  
✅ Nested objects also need these rules applied  
✅ Optional fields use null union: `["string", "null"]`  
✅ Supported types: string, integer, number, boolean, array, object, enum, anyOf

**Next:** [Schema Limitations](./06-schema-limitations.md)

---

## Further Reading

- [OpenAI Supported Schemas](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) — Official requirements
- [JSON Schema Spec](https://json-schema.org/understanding-json-schema/) — Full specification
- [Pydantic Field Types](https://docs.pydantic.dev/latest/concepts/fields/) — Python type definitions
