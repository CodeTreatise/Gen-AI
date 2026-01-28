---
title: "API Configuration"
---

# API Configuration

## Introduction

Structured Outputs can be configured through both the Responses API and Chat Completions API. Each has slightly different syntax, but both achieve the same guaranteed schema compliance.

### What We'll Cover

- Responses API configuration
- Chat Completions API configuration
- Schema format requirements
- Strict mode settings

### Prerequisites

- OpenAI API access
- JSON Schema basics
- Python or JavaScript SDK

---

## Responses API Configuration

The Responses API is OpenAI's newer, recommended API with cleaner syntax for Structured Outputs.

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json


@dataclass
class ResponsesAPIConfig:
    """Configuration for Responses API with Structured Outputs."""
    
    model: str
    input: List[Dict[str, str]]
    text_format: Optional[Dict[str, Any]] = None
    
    def to_request_body(self) -> dict:
        """Convert to API request body."""
        
        body = {
            "model": self.model,
            "input": self.input
        }
        
        if self.text_format:
            body["text"] = {"format": self.text_format}
        
        return body


# Basic configuration with inline schema
basic_config = ResponsesAPIConfig(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Extract person information."},
        {"role": "user", "content": "Alice is a 28-year-old designer."}
    ],
    text_format={
        "type": "json_schema",
        "name": "person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name"
                },
                "age": {
                    "type": "integer",
                    "description": "The person's age"
                },
                "occupation": {
                    "type": "string",
                    "description": "The person's job or profession"
                }
            },
            "required": ["name", "age", "occupation"],
            "additionalProperties": False
        }
    }
)


print("Responses API Configuration")
print("=" * 60)
print(json.dumps(basic_config.to_request_body(), indent=2))
```

### Using SDK with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List, Optional


# Define schema as Pydantic model
class Person(BaseModel):
    """Person extracted from text."""
    
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's job")


class TeamMember(BaseModel):
    """Team member with role."""
    
    person: Person
    role: str = Field(description="Role in the team")
    skills: List[str] = Field(description="List of skills")
    manager: Optional[str] = Field(
        default=None,
        description="Name of direct manager"
    )


# Usage with SDK (pseudocode - actual SDK call)
def extract_person_sdk_example():
    """Example SDK usage."""
    
    # The SDK call would look like:
    # response = client.responses.parse(
    #     model="gpt-4o",
    #     input=[
    #         {"role": "system", "content": "Extract person info."},
    #         {"role": "user", "content": "Alice is a 28-year-old designer."}
    #     ],
    #     text_format=Person  # Pydantic model directly!
    # )
    # 
    # person = response.output_parsed
    # print(f"{person.name}, {person.age}")
    
    pass


# Show the generated schema
print("\n\nPydantic Model → JSON Schema")
print("=" * 60)
print(json.dumps(Person.model_json_schema(), indent=2))

print("\n\nNested Model → JSON Schema")
print("=" * 60)
print(json.dumps(TeamMember.model_json_schema(), indent=2))
```

---

## Chat Completions API Configuration

The Chat Completions API uses `response_format` for Structured Outputs.

```python
@dataclass
class ChatCompletionsConfig:
    """Configuration for Chat Completions API."""
    
    model: str
    messages: List[Dict[str, str]]
    response_format: Optional[Dict[str, Any]] = None
    
    def to_request_body(self) -> dict:
        """Convert to API request body."""
        
        body = {
            "model": self.model,
            "messages": self.messages
        }
        
        if self.response_format:
            body["response_format"] = self.response_format
        
        return body


# Chat Completions configuration
chat_config = ChatCompletionsConfig(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract person information."},
        {"role": "user", "content": "Bob is a 35-year-old engineer."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"}
                },
                "required": ["name", "age", "occupation"],
                "additionalProperties": False
            }
        }
    }
)


print("\n\nChat Completions API Configuration")
print("=" * 60)
print(json.dumps(chat_config.to_request_body(), indent=2))
```

### Chat Completions with SDK

```python
# Chat Completions SDK usage
def extract_with_chat_completions():
    """Chat Completions with Structured Outputs."""
    
    # Using SDK's parse method:
    # response = client.chat.completions.parse(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "Extract person info."},
    #         {"role": "user", "content": "Carol is a 42-year-old doctor."}
    #     ],
    #     response_format=Person  # Pydantic model
    # )
    # 
    # person = response.choices[0].message.parsed
    # print(f"{person.name}, {person.age}")
    
    pass


# Comparison of API response structures
print("\n\nAPI Response Structure Comparison")
print("=" * 60)

print("""
📋 Responses API:
   response.output_parsed → Parsed Pydantic object
   response.output[0].content[0].text → Raw JSON string
   
📋 Chat Completions API:
   response.choices[0].message.parsed → Parsed object
   response.choices[0].message.content → Raw JSON string
""")
```

---

## Schema Format Requirements

### Required Properties

```python
@dataclass
class SchemaRequirement:
    """Schema requirement for Structured Outputs."""
    
    property_name: str
    required_value: Any
    description: str
    consequence_if_missing: str


SCHEMA_REQUIREMENTS = [
    SchemaRequirement(
        property_name="type",
        required_value="object",
        description="Root must be an object type",
        consequence_if_missing="API error: root must be object"
    ),
    SchemaRequirement(
        property_name="additionalProperties",
        required_value=False,
        description="Must be false for all objects",
        consequence_if_missing="API error: additionalProperties must be false"
    ),
    SchemaRequirement(
        property_name="required",
        required_value="[all property names]",
        description="All properties must be listed in required",
        consequence_if_missing="API error: all properties must be required"
    ),
    SchemaRequirement(
        property_name="strict",
        required_value=True,
        description="Must be true for Structured Outputs",
        consequence_if_missing="Falls back to JSON mode (no schema guarantee)"
    )
]


print("Schema Requirements")
print("=" * 60)

for req in SCHEMA_REQUIREMENTS:
    print(f"\n🔑 {req.property_name}: {req.required_value}")
    print(f"   {req.description}")
    print(f"   If missing: {req.consequence_if_missing}")
```

### Valid Schema Template

```python
def create_valid_schema(
    name: str,
    properties: Dict[str, Dict[str, Any]],
    descriptions: Optional[Dict[str, str]] = None
) -> dict:
    """Create a valid Structured Outputs schema."""
    
    descriptions = descriptions or {}
    
    # Build properties with descriptions
    schema_properties = {}
    for prop_name, prop_type in properties.items():
        prop_def = prop_type.copy()
        if prop_name in descriptions:
            prop_def["description"] = descriptions[prop_name]
        schema_properties[prop_name] = prop_def
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,  # Required!
            "schema": {
                "type": "object",
                "properties": schema_properties,
                "required": list(properties.keys()),  # All required!
                "additionalProperties": False  # Required!
            }
        }
    }


# Example usage
order_schema = create_valid_schema(
    name="order",
    properties={
        "order_id": {"type": "string"},
        "customer_name": {"type": "string"},
        "total": {"type": "number"},
        "items": {
            "type": "array",
            "items": {"type": "string"}
        },
        "shipped": {"type": "boolean"}
    },
    descriptions={
        "order_id": "Unique identifier for the order",
        "customer_name": "Name of the customer",
        "total": "Total price in dollars",
        "items": "List of item names",
        "shipped": "Whether the order has shipped"
    }
)

print("\n\nGenerated Valid Schema")
print("=" * 60)
print(json.dumps(order_schema, indent=2))
```

---

## Strict Mode

### strict: true Requirement

```python
class StrictModeExplainer:
    """Explains the strict mode requirement."""
    
    @staticmethod
    def explain():
        return """
The `strict: true` parameter is REQUIRED for Structured Outputs.

WITH strict: true (Structured Outputs):
  ✅ Schema is enforced at generation time
  ✅ Output always matches schema exactly
  ✅ No validation needed
  ✅ Constrained decoding active

WITHOUT strict: true (or with strict: false):
  ⚠️ Falls back to JSON mode
  ⚠️ Schema is treated as a hint, not a requirement
  ⚠️ Output may not match schema
  ⚠️ Validation and retries still needed

COMMON MISTAKE:
  Forgetting to set strict: true
  → API accepts the request
  → But schema is NOT enforced!
"""


# Show the difference
print(StrictModeExplainer.explain())


# Visual comparison
def compare_strict_modes():
    """Compare strict modes."""
    
    # With strict: true (Structured Outputs)
    with_strict = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,  # ← This enables Structured Outputs!
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
    
    # Without strict (falls back to JSON mode)
    without_strict = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            # Missing strict: true!
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
    
    return with_strict, without_strict


strict_on, strict_off = compare_strict_modes()

print("\n✅ With strict: true")
print(json.dumps(strict_on, indent=2))

print("\n\n❌ Missing strict: true (will NOT enforce schema!)")
print(json.dumps(strict_off, indent=2))
```

---

## Configuration Validation

```python
from typing import Tuple


class SchemaValidator:
    """Validate schemas for Structured Outputs compliance."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: dict) -> Tuple[bool, List[str], List[str]]:
        """Validate configuration."""
        
        self.errors = []
        self.warnings = []
        
        # Check for response_format or text.format
        if "response_format" in config:
            self._validate_chat_completions(config["response_format"])
        elif "text" in config and "format" in config["text"]:
            self._validate_responses_api(config["text"]["format"])
        else:
            self.errors.append("No schema format found in config")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_responses_api(self, format_config: dict):
        """Validate Responses API format."""
        
        if format_config.get("type") != "json_schema":
            self.errors.append("type must be 'json_schema'")
            return
        
        if not format_config.get("strict"):
            self.errors.append("strict must be true for Structured Outputs")
        
        if "schema" not in format_config:
            self.errors.append("schema is required")
            return
        
        self._validate_schema(format_config["schema"])
    
    def _validate_chat_completions(self, response_format: dict):
        """Validate Chat Completions response_format."""
        
        if response_format.get("type") != "json_schema":
            self.errors.append("type must be 'json_schema'")
            return
        
        json_schema = response_format.get("json_schema", {})
        
        if not json_schema.get("strict"):
            self.errors.append("strict must be true")
        
        if "schema" not in json_schema:
            self.errors.append("schema is required in json_schema")
            return
        
        self._validate_schema(json_schema["schema"])
    
    def _validate_schema(self, schema: dict, path: str = "root"):
        """Validate JSON schema structure."""
        
        if schema.get("type") == "object":
            # Check additionalProperties
            if schema.get("additionalProperties") != False:
                self.errors.append(
                    f"{path}: additionalProperties must be false"
                )
            
            # Check required contains all properties
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for prop in properties:
                if prop not in required:
                    self.errors.append(
                        f"{path}: property '{prop}' must be in required"
                    )
            
            # Recursively validate nested objects
            for prop_name, prop_def in properties.items():
                if prop_def.get("type") == "object":
                    self._validate_schema(
                        prop_def,
                        f"{path}.{prop_name}"
                    )
                elif prop_def.get("type") == "array":
                    items = prop_def.get("items", {})
                    if items.get("type") == "object":
                        self._validate_schema(
                            items,
                            f"{path}.{prop_name}[]"
                        )


# Test validation
validator = SchemaValidator()

# Valid config
valid_config = {
    "text": {
        "format": {
            "type": "json_schema",
            "name": "test",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    }
}

# Invalid config (missing strict, missing additionalProperties)
invalid_config = {
    "text": {
        "format": {
            "type": "json_schema",
            "name": "test",
            # Missing strict: true
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": []  # Missing "name"
                # Missing additionalProperties: false
            }
        }
    }
}


print("\n\nConfiguration Validation")
print("=" * 60)

print("\n✅ Valid Configuration:")
valid, errors, warnings = validator.validate(valid_config)
print(f"   Valid: {valid}")
print(f"   Errors: {errors}")

print("\n❌ Invalid Configuration:")
valid, errors, warnings = validator.validate(invalid_config)
print(f"   Valid: {valid}")
print(f"   Errors: {errors}")
```

---

## Hands-on Exercise

### Your Task

Build a configuration generator that creates valid Structured Outputs configs from simple field definitions.

### Requirements

1. Accept simple field type definitions
2. Generate complete, valid schemas
3. Support both API formats (Responses and Chat Completions)
4. Validate the generated configuration

<details>
<summary>💡 Hints</summary>

- Map simple types to JSON Schema types
- Auto-generate required array
- Always set additionalProperties: false
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json


class FieldType(Enum):
    """Simple field types for definition."""
    
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING_ARRAY = "string_array"
    INTEGER_ARRAY = "integer_array"
    OBJECT = "object"


@dataclass
class FieldDefinition:
    """Simple field definition."""
    
    name: str
    field_type: FieldType
    description: str = ""
    nullable: bool = False
    enum_values: Optional[List[str]] = None
    nested_fields: Optional[Dict[str, 'FieldDefinition']] = None


class StructuredOutputsConfigGenerator:
    """Generate valid Structured Outputs configurations."""
    
    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.fields: List[FieldDefinition] = []
    
    def add_field(
        self,
        name: str,
        field_type: Union[FieldType, str],
        description: str = "",
        nullable: bool = False,
        enum_values: List[str] = None,
        nested_fields: Dict = None
    ) -> 'StructuredOutputsConfigGenerator':
        """Add a field to the schema (fluent API)."""
        
        if isinstance(field_type, str):
            field_type = FieldType(field_type)
        
        self.fields.append(FieldDefinition(
            name=name,
            field_type=field_type,
            description=description,
            nullable=nullable,
            enum_values=enum_values,
            nested_fields=nested_fields
        ))
        
        return self
    
    def _build_field_schema(self, field: FieldDefinition) -> dict:
        """Build JSON Schema for a field."""
        
        type_mapping = {
            FieldType.STRING: {"type": "string"},
            FieldType.INTEGER: {"type": "integer"},
            FieldType.NUMBER: {"type": "number"},
            FieldType.BOOLEAN: {"type": "boolean"},
            FieldType.STRING_ARRAY: {
                "type": "array",
                "items": {"type": "string"}
            },
            FieldType.INTEGER_ARRAY: {
                "type": "array",
                "items": {"type": "integer"}
            }
        }
        
        if field.field_type == FieldType.OBJECT and field.nested_fields:
            schema = self._build_object_schema(field.nested_fields)
        else:
            schema = type_mapping.get(
                field.field_type,
                {"type": "string"}
            ).copy()
        
        # Add description
        if field.description:
            schema["description"] = field.description
        
        # Add enum
        if field.enum_values:
            schema["enum"] = field.enum_values
        
        # Handle nullable
        if field.nullable:
            if "type" in schema:
                current_type = schema["type"]
                schema["type"] = [current_type, "null"]
        
        return schema
    
    def _build_object_schema(
        self,
        fields: Dict[str, FieldDefinition]
    ) -> dict:
        """Build schema for nested object."""
        
        properties = {}
        required = []
        
        for name, field in fields.items():
            properties[name] = self._build_field_schema(field)
            required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def build_schema(self) -> dict:
        """Build the complete JSON Schema."""
        
        properties = {}
        required = []
        
        for field in self.fields:
            properties[field.name] = self._build_field_schema(field)
            required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def for_responses_api(self) -> dict:
        """Generate Responses API configuration."""
        
        return {
            "type": "json_schema",
            "name": self.schema_name,
            "strict": True,
            "schema": self.build_schema()
        }
    
    def for_chat_completions(self) -> dict:
        """Generate Chat Completions configuration."""
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_name,
                "strict": True,
                "schema": self.build_schema()
            }
        }
    
    def full_responses_request(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str
    ) -> dict:
        """Generate full Responses API request body."""
        
        return {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "text": {
                "format": self.for_responses_api()
            }
        }
    
    def full_chat_request(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str
    ) -> dict:
        """Generate full Chat Completions request body."""
        
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": self.for_chat_completions()
        }


# Example: Create an order extraction schema
generator = (
    StructuredOutputsConfigGenerator("order")
    .add_field("order_id", FieldType.STRING, "Unique order identifier")
    .add_field("customer_name", FieldType.STRING, "Customer's full name")
    .add_field("email", FieldType.STRING, "Customer email address")
    .add_field("total_amount", FieldType.NUMBER, "Total in dollars")
    .add_field("items", FieldType.STRING_ARRAY, "List of ordered items")
    .add_field("is_priority", FieldType.BOOLEAN, "Whether priority shipping")
    .add_field(
        "status", 
        FieldType.STRING, 
        "Order status",
        enum_values=["pending", "processing", "shipped", "delivered"]
    )
    .add_field(
        "notes",
        FieldType.STRING,
        "Optional order notes",
        nullable=True
    )
)


print("Configuration Generator")
print("=" * 60)

print("\n📋 Generated Schema:")
print(json.dumps(generator.build_schema(), indent=2))

print("\n\n📋 Responses API Format:")
print(json.dumps(generator.for_responses_api(), indent=2))

print("\n\n📋 Chat Completions Format:")
print(json.dumps(generator.for_chat_completions(), indent=2))

print("\n\n📋 Full Responses API Request:")
request = generator.full_responses_request(
    model="gpt-4o",
    system_prompt="Extract order information from the email.",
    user_prompt="Order #12345 for John Smith, total $99.99..."
)
print(json.dumps(request, indent=2))
```

</details>

---

## Summary

✅ Responses API uses `text.format` for schema configuration  
✅ Chat Completions uses `response_format` parameter  
✅ `strict: true` is required for Structured Outputs  
✅ All properties must be in `required` array  
✅ `additionalProperties: false` is mandatory

**Next:** [SDK Integration](./04-sdk-integration.md)

---

## Further Reading

- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) — API reference
- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) — API reference
- [JSON Schema Guide](https://json-schema.org/learn/getting-started-step-by-step) — Schema basics
