---
title: "Schema Conversion to Function Definitions"
---

# Schema Conversion to Function Definitions

## Introduction

The previous lesson showed how to extract operations from an OpenAPI spec and generate basic tool definitions. But real-world APIs have complex schemas — nested objects, `$ref` chains three levels deep, `allOf` compositions for inheritance, `oneOf` discriminators for polymorphism, and arrays of structured objects. Converting these accurately into tool parameter schemas is what separates a prototype from a production pipeline.

In this lesson, we tackle the hard parts of schema conversion: recursive `$ref` resolution, composition keywords (`allOf`, `oneOf`, `anyOf`), type mapping edge cases, and the provider-specific constraints that affect what schemas each AI model can actually handle.

### What we'll cover

- OpenAPI-to-JSON Schema type mapping
- Recursive `$ref` resolution with cycle detection
- Handling `allOf`, `oneOf`, and `anyOf` composition
- Enum, array, and nested object conversion
- Provider-specific schema constraints and workarounds

### Prerequisites

- Completed the previous lesson on auto-generating tools from specs
- Understanding of JSON Schema basics (types, properties, required)
- Familiarity with `$ref` pointers in OpenAPI

---

## OpenAPI-to-JSON Schema type mapping

OpenAPI 3.1 adopted full JSON Schema compatibility, but earlier versions (3.0.x) had a subset. When converting to tool definitions, we must handle both. The core type mappings are:

| OpenAPI Type | OpenAPI Format | JSON Schema for Tools | Notes |
|-------------|---------------|----------------------|-------|
| `string` | — | `"type": "string"` | Direct mapping |
| `string` | `date` | `"type": "string"` | Add description: "ISO 8601 date (YYYY-MM-DD)" |
| `string` | `date-time` | `"type": "string"` | Add description: "ISO 8601 datetime" |
| `string` | `email` | `"type": "string"` | Add description: "Email address" |
| `string` | `uri` | `"type": "string"` | Add description: "URL" |
| `string` | `uuid` | `"type": "string"` | Add description: "UUID v4" |
| `string` | `binary` | ⚠️ Skip | File uploads — not suitable for function calling |
| `integer` | `int32` | `"type": "integer"` | Direct mapping |
| `integer` | `int64` | `"type": "integer"` | Direct mapping |
| `number` | `float` | `"type": "number"` | Direct mapping |
| `number` | `double` | `"type": "number"` | Direct mapping |
| `boolean` | — | `"type": "boolean"` | Direct mapping |
| `array` | — | `"type": "array"` | Must include `items` schema |
| `object` | — | `"type": "object"` | Must include `properties` |
| `null` | — | `"type": "null"` | OpenAPI 3.1 only |

> **Note:** The `format` keyword is informational — AI providers don't validate against it. But including the format in the description helps the model generate correctly formatted values.

### The type converter

```python
def convert_type(schema: dict) -> dict:
    """Convert an OpenAPI schema to a tool-compatible JSON Schema property.
    
    Handles type, format, description, enum, default, and constraints.
    
    Args:
        schema: An OpenAPI schema object
        
    Returns:
        A JSON Schema property definition for tool parameters
    """
    result = {}
    
    schema_type = schema.get("type")
    schema_format = schema.get("format")
    
    # Skip binary/file types
    if schema_format == "binary":
        return {"type": "string", "description": "[Binary content - not supported]"}
    
    # Map type
    if schema_type:
        result["type"] = schema_type
    
    # Preserve description, enriching with format info
    description_parts = []
    if "description" in schema:
        description_parts.append(schema["description"])
    
    # Add format context to description
    format_hints = {
        "date": "Format: YYYY-MM-DD",
        "date-time": "Format: ISO 8601 datetime (e.g., 2025-01-15T10:30:00Z)",
        "email": "Must be a valid email address",
        "uri": "Must be a valid URL",
        "uuid": "Must be a UUID (e.g., 550e8400-e29b-41d4-a716-446655440000)",
    }
    if schema_format in format_hints:
        description_parts.append(format_hints[schema_format])
    
    if description_parts:
        result["description"] = ". ".join(description_parts)
    
    # Preserve enum values
    if "enum" in schema:
        result["enum"] = schema["enum"]
    
    # Preserve default
    if "default" in schema:
        result["default"] = schema["default"]
    
    # Preserve constraints (for documentation, not enforced by LLMs)
    for constraint in ["minimum", "maximum", "minLength", "maxLength", "pattern"]:
        if constraint in schema:
            result[constraint] = schema[constraint]
    
    return result
```

**Output:**
```python
# String with date format
print(convert_type({"type": "string", "format": "date", "description": "Order date"}))
# {"type": "string", "description": "Order date. Format: YYYY-MM-DD"}

# Enum
print(convert_type({"type": "string", "enum": ["available", "pending", "sold"]}))
# {"type": "string", "enum": ["available", "pending", "sold"]}
```

---

## Recursive `$ref` resolution with cycle detection

Real APIs reference shared schemas heavily. The Petstore spec alone has chains like `$ref: "#/components/schemas/Pet"` where `Pet` itself references `Category` and `Tag`. We need recursive resolution — but we must also detect cycles to avoid infinite loops.

```python
def resolve_schema(
    schema: dict,
    spec: dict,
    seen_refs: set | None = None,
    max_depth: int = 10,
) -> dict:
    """Recursively resolve a schema, handling $ref, nested objects, and arrays.
    
    Args:
        schema: Schema to resolve
        spec: Full OpenAPI spec
        seen_refs: Set of already-visited $ref paths (cycle detection)
        max_depth: Maximum recursion depth
        
    Returns:
        Fully resolved schema with no remaining $ref pointers
    """
    if seen_refs is None:
        seen_refs = set()
    
    if max_depth <= 0:
        return {"type": "object", "description": "[Schema too deeply nested]"}
    
    # Handle $ref
    if "$ref" in schema:
        ref_path = schema["$ref"]
        
        if ref_path in seen_refs:
            # Cycle detected — return a placeholder
            return {
                "type": "object",
                "description": f"[Circular reference to {ref_path.split('/')[-1]}]",
            }
        
        seen_refs = seen_refs | {ref_path}  # New set to avoid cross-branch pollution
        resolved = _follow_ref(ref_path, spec)
        return resolve_schema(resolved, spec, seen_refs, max_depth - 1)
    
    result = {}
    
    # Copy simple fields
    for key in ["type", "description", "enum", "default", "format",
                "minimum", "maximum", "minLength", "maxLength", "pattern"]:
        if key in schema:
            result[key] = schema[key]
    
    # Handle object properties
    if schema.get("type") == "object" or "properties" in schema:
        result["type"] = "object"
        if "properties" in schema:
            result["properties"] = {}
            for prop_name, prop_schema in schema["properties"].items():
                result["properties"][prop_name] = resolve_schema(
                    prop_schema, spec, seen_refs, max_depth - 1
                )
        if "required" in schema:
            result["required"] = schema["required"]
    
    # Handle arrays
    if schema.get("type") == "array" and "items" in schema:
        result["type"] = "array"
        result["items"] = resolve_schema(
            schema["items"], spec, seen_refs, max_depth - 1
        )
    
    return result


def _follow_ref(ref_path: str, spec: dict) -> dict:
    """Follow a $ref pointer to its target in the spec."""
    if not ref_path.startswith("#/"):
        raise ValueError(f"External references not supported: {ref_path}")
    
    parts = ref_path.lstrip("#/").split("/")
    target = spec
    for part in parts:
        # Handle JSON Pointer escaping
        part = part.replace("~1", "/").replace("~0", "~")
        if part not in target:
            raise KeyError(f"$ref path not found: {ref_path} (missing '{part}')")
        target = target[part]
    
    return target
```

**Output with a self-referencing schema:**
```python
# A schema where Category references itself (artificial cycle example)
spec_with_cycle = {
    "components": {
        "schemas": {
            "Category": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parent": {"$ref": "#/components/schemas/Category"},
                },
            }
        }
    }
}

resolved = resolve_schema(
    {"$ref": "#/components/schemas/Category"},
    spec_with_cycle,
)
print(json.dumps(resolved, indent=2))
```

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "parent": {
      "type": "object",
      "description": "[Circular reference to Category]"
    }
  }
}
```

> **Warning:** Cycle detection is essential. Without it, a self-referencing schema like a tree node (`TreeNode` has `children: TreeNode[]`) will cause infinite recursion and crash your application.

---

## Handling `allOf`, `oneOf`, and `anyOf`

OpenAPI uses composition keywords to build complex schemas from simpler ones. These are the most common patterns and how to convert them for tool definitions:

### `allOf` — Merge all schemas

`allOf` is used for inheritance and extension. We merge all sub-schemas into a single object:

```python
def resolve_allof(schemas: list[dict], spec: dict, seen_refs: set) -> dict:
    """Merge all schemas from an allOf into a single schema.
    
    Args:
        schemas: List of schemas to merge
        spec: Full OpenAPI spec
        seen_refs: Set of visited $ref paths
        
    Returns:
        Merged schema with combined properties and requirements
    """
    merged = {"type": "object", "properties": {}, "required": []}
    
    for sub_schema in schemas:
        resolved = resolve_schema(sub_schema, spec, seen_refs)
        
        # Merge properties
        if "properties" in resolved:
            merged["properties"].update(resolved["properties"])
        
        # Merge required fields
        if "required" in resolved:
            merged["required"].extend(resolved["required"])
        
        # Inherit description from first schema that has one
        if "description" in resolved and "description" not in merged:
            merged["description"] = resolved["description"]
    
    # Deduplicate required
    merged["required"] = list(set(merged["required"]))
    
    if not merged["required"]:
        del merged["required"]
    
    return merged
```

**Example — `allOf` in the wild:**
```yaml
# OpenAPI spec using allOf for inheritance
components:
  schemas:
    Pet:
      allOf:
        - $ref: "#/components/schemas/NewPet"
        - type: object
          required:
            - id
          properties:
            id:
              type: integer
              format: int64
    NewPet:
      type: object
      required:
        - name
      properties:
        name:
          type: string
        tag:
          type: string
```

**Resolved output:**
```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "tag": { "type": "string" },
    "id": { "type": "integer" }
  },
  "required": ["name", "id"]
}
```

### `oneOf` and `anyOf` — Pick one or combine

These are harder because function calling schemas don't natively support unions. The practical approach is to merge all possible properties and mark none as strictly required:

```python
def resolve_oneof_anyof(
    schemas: list[dict],
    spec: dict,
    seen_refs: set,
    keyword: str = "oneOf",
) -> dict:
    """Convert oneOf/anyOf to a merged schema with optional properties.
    
    Since tool schemas don't support unions, we merge all possible
    properties and add a description explaining the alternatives.
    
    Args:
        schemas: List of alternative schemas
        spec: Full OpenAPI spec
        seen_refs: Set of visited $ref paths
        keyword: "oneOf" or "anyOf" for description text
        
    Returns:
        Merged schema with description of alternatives
    """
    all_properties = {}
    variant_descriptions = []
    
    for i, sub_schema in enumerate(schemas, 1):
        resolved = resolve_schema(sub_schema, spec, seen_refs)
        
        if "properties" in resolved:
            all_properties.update(resolved["properties"])
        
        desc = resolved.get("description", resolved.get("type", f"Variant {i}"))
        variant_descriptions.append(desc)
    
    description = f"One of: {', '.join(variant_descriptions)}"
    
    result = {
        "type": "object",
        "description": description,
        "properties": all_properties,
    }
    
    return result
```

> **🤖 AI Context:** When the model sees a description like "One of: CreditCard, BankTransfer, PayPal," it typically asks the user which payment method they want or infers from context — even though the schema doesn't formally express the union.

### Unified resolver with composition support

Here is the updated `resolve_schema` function that handles all composition keywords:

```python
def resolve_schema_full(
    schema: dict,
    spec: dict,
    seen_refs: set | None = None,
    max_depth: int = 10,
) -> dict:
    """Fully resolve a schema including $ref and composition keywords.
    
    Handles: $ref, allOf, oneOf, anyOf, nested objects, arrays.
    """
    if seen_refs is None:
        seen_refs = set()
    
    if max_depth <= 0:
        return {"type": "object", "description": "[Schema too deeply nested]"}
    
    # Handle $ref
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path in seen_refs:
            return {"type": "object", "description": f"[Circular: {ref_path.split('/')[-1]}]"}
        seen_refs = seen_refs | {ref_path}
        resolved = _follow_ref(ref_path, spec)
        return resolve_schema_full(resolved, spec, seen_refs, max_depth - 1)
    
    # Handle allOf
    if "allOf" in schema:
        return resolve_allof(schema["allOf"], spec, seen_refs)
    
    # Handle oneOf / anyOf
    if "oneOf" in schema:
        return resolve_oneof_anyof(schema["oneOf"], spec, seen_refs, "oneOf")
    if "anyOf" in schema:
        return resolve_oneof_anyof(schema["anyOf"], spec, seen_refs, "anyOf")
    
    # Build result
    result = convert_type(schema)
    
    # Handle object properties
    if schema.get("type") == "object" or "properties" in schema:
        result["type"] = "object"
        if "properties" in schema:
            result["properties"] = {
                name: resolve_schema_full(prop, spec, seen_refs, max_depth - 1)
                for name, prop in schema["properties"].items()
            }
        if "required" in schema:
            result["required"] = schema["required"]
    
    # Handle arrays
    if schema.get("type") == "array" and "items" in schema:
        result["items"] = resolve_schema_full(
            schema["items"], spec, seen_refs, max_depth - 1
        )
    
    return result
```

---

## Provider-specific schema constraints

Each AI provider has its own limitations on what schemas it will accept. Knowing these prevents runtime errors:

### OpenAI strict mode constraints

When `strict: true`, OpenAI enforces:

```python
def apply_openai_strict_constraints(schema: dict) -> dict:
    """Transform a schema to comply with OpenAI strict mode.
    
    Rules:
    - additionalProperties must be false on all objects
    - All properties must be listed in required
    - No unsupported keywords (patternProperties, etc.)
    """
    if schema.get("type") != "object":
        return schema
    
    result = {**schema}
    
    # All objects must have additionalProperties: false
    result["additionalProperties"] = False
    
    # All properties must be required in strict mode
    if "properties" in result:
        result["required"] = list(result["properties"].keys())
        
        # Recursively apply to nested objects
        for prop_name, prop_schema in result["properties"].items():
            if prop_schema.get("type") == "object":
                result["properties"][prop_name] = apply_openai_strict_constraints(
                    prop_schema
                )
            elif prop_schema.get("type") == "array":
                items = prop_schema.get("items", {})
                if items.get("type") == "object":
                    prop_schema["items"] = apply_openai_strict_constraints(items)
    
    return result
```

> **Important:** In OpenAI strict mode, optional parameters become required in the schema. The model handles optionality through its own reasoning — it can pass `null` or empty values. This is a deliberate trade-off for guaranteed schema compliance.

### Anthropic schema handling

Anthropic is more permissive and closely follows standard JSON Schema:

```python
def apply_anthropic_constraints(schema: dict) -> dict:
    """Transform a schema for Anthropic tool use.
    
    Anthropic follows standard JSON Schema more closely.
    Key differences from OpenAI:
    - additionalProperties not required
    - Only truly required fields in 'required'
    - Supports nullable via type arrays
    """
    result = {**schema}
    
    # Remove OpenAI-specific strict mode artifacts
    result.pop("additionalProperties", None)
    
    # Keep only genuinely required fields
    # (don't force all properties into required)
    
    return result
```

### Gemini schema subset

Google Gemini uses a restricted subset of JSON Schema:

```python
def apply_gemini_constraints(schema: dict) -> dict:
    """Transform a schema for Google Gemini function declarations.
    
    Gemini supports a subset of JSON Schema:
    - Supported types: string, number, integer, boolean, array, object
    - No support for: null type, patternProperties, if/then/else
    - Uses 'nullable' keyword instead of type arrays
    - enum values must be strings
    """
    result = {}
    
    schema_type = schema.get("type")
    
    # Gemini doesn't support null type
    if schema_type == "null":
        return {"type": "string", "description": "null value", "nullable": True}
    
    if schema_type:
        result["type"] = schema_type
    
    if "description" in schema:
        result["description"] = schema["description"]
    
    # Gemini requires string enums
    if "enum" in schema:
        result["enum"] = [str(v) for v in schema["enum"]]
    
    # Handle nullable
    if schema.get("nullable"):
        result["nullable"] = True
    
    # Handle properties
    if "properties" in schema:
        result["properties"] = {
            name: apply_gemini_constraints(prop)
            for name, prop in schema["properties"].items()
        }
    
    if "required" in schema:
        result["required"] = schema["required"]
    
    if "items" in schema:
        result["items"] = apply_gemini_constraints(schema["items"])
    
    return result
```

### Provider constraint comparison

| Feature | OpenAI (Strict) | Anthropic | Gemini |
|---------|----------------|-----------|--------|
| `additionalProperties: false` | ✅ Required | ⬜ Optional | ⬜ Optional |
| All properties in `required` | ✅ Required | ❌ Only real required | ❌ Only real required |
| `null` type | ✅ Supported | ✅ Supported | ❌ Use `nullable` |
| `enum` types | Any JSON | Any JSON | Strings only |
| Max nesting depth | 5 levels | No documented limit | 5 levels |
| `allOf`/`oneOf` | ⚠️ Must flatten | ✅ Supported | ❌ Must flatten |
| `pattern` | ⬜ Ignored | ⬜ Ignored | ⬜ Ignored |

---

## Putting it all together

Here is a complete schema converter that produces valid output for any provider:

```python
import json
from copy import deepcopy


class SchemaConverter:
    """Convert OpenAPI schemas to provider-specific tool parameter schemas."""
    
    def __init__(self, spec: dict):
        self.spec = spec
    
    def convert(self, schema: dict, provider: str = "openai") -> dict:
        """Convert an OpenAPI schema to a tool parameter schema.
        
        Args:
            schema: OpenAPI schema (may contain $ref, allOf, etc.)
            provider: Target provider ("openai", "anthropic", "gemini")
            
        Returns:
            Provider-compatible JSON Schema for tool parameters
        """
        # Step 1: Resolve all references and compositions
        resolved = resolve_schema_full(schema, self.spec)
        
        # Step 2: Apply provider-specific constraints
        constraints = {
            "openai": apply_openai_strict_constraints,
            "anthropic": apply_anthropic_constraints,
            "gemini": apply_gemini_constraints,
        }
        
        apply_fn = constraints.get(provider)
        if not apply_fn:
            raise ValueError(f"Unknown provider: {provider}")
        
        return apply_fn(deepcopy(resolved))
    
    def convert_operation(self, operation: dict, provider: str = "openai") -> dict:
        """Convert a full operation into a tool definition.
        
        Args:
            operation: Extracted operation descriptor
            provider: Target provider
            
        Returns:
            Complete tool definition for the provider
        """
        # Build combined parameter schema
        param_schema = self._build_params(operation)
        
        # Apply provider constraints
        converted = self.convert(param_schema, provider)
        
        description = (
            operation.get("summary")
            or operation.get("description")
            or f"{operation['method']} {operation['path']}"
        )
        
        if provider == "openai":
            return {
                "type": "function",
                "name": operation["operation_id"],
                "description": description,
                "parameters": converted,
                "strict": True,
            }
        elif provider == "anthropic":
            return {
                "name": operation["operation_id"],
                "description": description,
                "input_schema": converted,
            }
        else:  # gemini
            return {
                "name": operation["operation_id"],
                "description": description,
                "parameters": converted,
            }
    
    def _build_params(self, operation: dict) -> dict:
        """Build a unified parameter schema from an operation."""
        properties = {}
        required = []
        
        for param in operation.get("parameters", []):
            param_schema = param.get("schema", {"type": "string"})
            resolved = resolve_schema_full(param_schema, self.spec)
            prop = convert_type(resolved)
            
            if "description" in param and "description" not in prop:
                prop["description"] = param["description"]
            
            properties[param["name"]] = prop
            
            if param.get("required"):
                required.append(param["name"])
        
        # Merge request body
        rb = operation.get("request_body")
        if rb:
            content = rb.get("content", {})
            json_schema = content.get("application/json", {}).get("schema", {})
            if json_schema:
                body = resolve_schema_full(json_schema, self.spec)
                if body.get("type") == "object" and "properties" in body:
                    properties.update(body["properties"])
                    required.extend(body.get("required", []))
        
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = list(set(required))
        
        return schema
```

**Usage:**
```python
converter = SchemaConverter(spec)

# Convert a single operation for all providers
for provider in ["openai", "anthropic", "gemini"]:
    tool = converter.convert_operation(operations[0], provider)
    print(f"\n--- {provider.upper()} ---")
    print(json.dumps(tool, indent=2))
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Always detect `$ref` cycles before resolving | Prevents infinite recursion on self-referencing schemas |
| Flatten `allOf` into merged objects | All providers work best with flat object schemas |
| Convert `oneOf`/`anyOf` to optional properties with descriptions | Models understand union types through descriptions better than schema keywords |
| Include `format` info in `description` | Models use descriptions for formatting guidance since `format` isn't enforced |
| Apply provider constraints as the last step | Keep the core conversion provider-agnostic for reuse |
| Limit schema depth to 3–5 levels | Deeply nested schemas confuse models and hit provider limits |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not handling circular `$ref` references | Track visited refs in a set, return placeholder on cycle |
| Passing `allOf` directly to tool schemas | Flatten into a single merged object before sending |
| Forgetting `additionalProperties: false` for OpenAI strict mode | Apply it recursively to all nested objects, not just the root |
| Using numeric enums for Gemini | Convert all enum values to strings for Gemini compatibility |
| Losing `required` fields during `allOf` merge | Collect `required` from all sub-schemas and deduplicate |
| Ignoring `nullable` in OpenAPI 3.0 | Convert `nullable: true` to appropriate provider format |

---

## Hands-on exercise

### Your task

Build a schema converter that handles a complex API spec with inheritance (`allOf`), discriminated unions (`oneOf`), and deep nesting. Verify the output works with all three providers.

### Requirements

1. Create a sample OpenAPI spec with:
   - A `Pet` schema using `allOf` to extend `BaseAnimal`
   - A `Payment` schema using `oneOf` for `CreditCard` and `BankTransfer`
   - A `Category` schema with a self-referencing `parent` field
2. Resolve all schemas using `resolve_schema_full`
3. Apply provider constraints for OpenAI, Anthropic, and Gemini
4. Verify no `$ref` remains in any output
5. Verify no infinite recursion on the self-referencing `Category`

### Expected result

Three sets of clean, flat schemas with no `$ref` pointers, no `allOf`/`oneOf` keywords, and provider-appropriate constraint handling.

<details>
<summary>💡 Hints (click to expand)</summary>

- Start by defining the schemas in a Python dict matching OpenAPI structure
- Use `json.dumps(result, indent=2)` to inspect resolved output at each step
- Check that `Category.parent` shows as `[Circular: Category]` and doesn't recurse
- For OpenAI, verify every object has `additionalProperties: false`
- For Gemini, verify enum values are all strings

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from copy import deepcopy

# Sample spec with allOf, oneOf, and circular reference
spec = {
    "openapi": "3.1.1",
    "info": {"title": "Complex API", "version": "1.0.0"},
    "paths": {},
    "components": {
        "schemas": {
            "BaseAnimal": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "Animal name"},
                    "species": {"type": "string", "description": "Species type"},
                },
            },
            "Pet": {
                "allOf": [
                    {"$ref": "#/components/schemas/BaseAnimal"},
                    {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {"type": "integer"},
                            "vaccinated": {"type": "boolean"},
                        },
                    },
                ],
            },
            "CreditCard": {
                "type": "object",
                "required": ["card_number"],
                "properties": {
                    "card_number": {"type": "string"},
                    "expiry": {"type": "string", "format": "date"},
                },
            },
            "BankTransfer": {
                "type": "object",
                "required": ["account_number"],
                "properties": {
                    "account_number": {"type": "string"},
                    "routing_number": {"type": "string"},
                },
            },
            "Payment": {
                "oneOf": [
                    {"$ref": "#/components/schemas/CreditCard"},
                    {"$ref": "#/components/schemas/BankTransfer"},
                ],
            },
            "Category": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parent": {"$ref": "#/components/schemas/Category"},
                },
            },
        },
    },
}

# Resolve each schema
for schema_name in ["Pet", "Payment", "Category"]:
    ref = {"$ref": f"#/components/schemas/{schema_name}"}
    resolved = resolve_schema_full(ref, spec)
    
    print(f"\n=== {schema_name} (resolved) ===")
    print(json.dumps(resolved, indent=2))
    
    # Apply provider constraints
    for provider_name, apply_fn in [
        ("OpenAI", apply_openai_strict_constraints),
        ("Anthropic", apply_anthropic_constraints),
        ("Gemini", apply_gemini_constraints),
    ]:
        constrained = apply_fn(deepcopy(resolved))
        has_ref = "$ref" in json.dumps(constrained)
        print(f"  {provider_name}: valid={not has_ref}, "
              f"props={list(constrained.get('properties', {}).keys())}")
```

**Expected Output:**
```
=== Pet (resolved) ===
{
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "Animal name"},
    "species": {"type": "string", "description": "Species type"},
    "id": {"type": "integer"},
    "vaccinated": {"type": "boolean"}
  },
  "required": ["name", "id"]
}
  OpenAI: valid=True, props=['name', 'species', 'id', 'vaccinated']
  Anthropic: valid=True, props=['name', 'species', 'id', 'vaccinated']
  Gemini: valid=True, props=['name', 'species', 'id', 'vaccinated']

=== Payment (resolved) ===
{
  "type": "object",
  "description": "One of: CreditCard, BankTransfer",
  "properties": {
    "card_number": {"type": "string"},
    "expiry": {"type": "string"},
    "account_number": {"type": "string"},
    "routing_number": {"type": "string"}
  }
}
  OpenAI: valid=True, props=['card_number', 'expiry', 'account_number', 'routing_number']
  Anthropic: valid=True, props=['card_number', 'expiry', 'account_number', 'routing_number']
  Gemini: valid=True, props=['card_number', 'expiry', 'account_number', 'routing_number']

=== Category (resolved) ===
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "parent": {"type": "object", "description": "[Circular: Category]"}
  }
}
  OpenAI: valid=True, props=['name', 'parent']
  Anthropic: valid=True, props=['name', 'parent']
  Gemini: valid=True, props=['name', 'parent']
```

</details>

### Bonus challenges

- [ ] Add support for `discriminator` in `oneOf` schemas to produce better descriptions (e.g., "Use 'type' field to select: credit_card, bank_transfer")
- [ ] Handle `additionalProperties` as a typed schema (e.g., `additionalProperties: {type: string}` for maps)
- [ ] Build a validation function that reports which provider constraints a schema violates before conversion

---

## Summary

✅ OpenAPI types map directly to JSON Schema types, but `format` information should be embedded in `description` fields since providers don't enforce formats

✅ Recursive `$ref` resolution with cycle detection prevents infinite loops on self-referencing schemas — always track visited refs

✅ `allOf` merges into a single flat object; `oneOf`/`anyOf` merge into optional properties with descriptive text, since tool schemas don't support unions

✅ Each provider has specific constraints: OpenAI strict mode requires `additionalProperties: false` and all properties in `required`; Gemini requires string enums and uses `nullable` instead of `null` type

✅ Apply provider constraints as the final step, keeping core resolution provider-agnostic

---

**Previous:** [Auto-Generating Tools from Specs](./01-auto-generating-tools-from-specs.md)

**Next:** [API Discovery for AI →](./03-api-discovery-for-ai.md)

<!--
Sources Consulted:
- OpenAPI Specification v3.1.1 — Schema Object: https://spec.openapis.org/oas/v3.1.1.html#schema-object
- JSON Schema Specification: https://json-schema.org/understanding-json-schema/
- OpenAI Function Calling — Structured Outputs: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use — Input Schema: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling — Supported Schema Fields: https://ai.google.dev/gemini-api/docs/function-calling
-->
