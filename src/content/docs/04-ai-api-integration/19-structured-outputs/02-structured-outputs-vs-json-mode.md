---
title: "Structured Outputs vs JSON Mode"
---

# Structured Outputs vs JSON Mode

## Introduction

Both Structured Outputs and JSON mode produce valid JSON, but they differ fundamentally in schema enforcement. Understanding when to use each helps you choose the right approach for your application.

### What We'll Cover

- Key differences between the two modes
- Schema enforcement comparison
- Model compatibility
- Migration from JSON mode to Structured Outputs

### Prerequisites

- Understanding of JSON format
- OpenAI API experience
- Basic schema concepts

---

## Side-by-Side Comparison

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class OutputFeature(Enum):
    """Features to compare."""
    
    VALID_JSON = "valid_json"
    SCHEMA_ADHERENCE = "schema_adherence"
    REQUIRED_FIELDS = "required_fields"
    TYPE_ENFORCEMENT = "type_enforcement"
    ENUM_CONSTRAINT = "enum_constraint"
    REFUSAL_HANDLING = "refusal_handling"
    PROMPT_REQUIREMENT = "prompt_requirement"
    MODEL_SUPPORT = "model_support"


@dataclass
class FeatureComparison:
    """Comparison of a feature across modes."""
    
    feature: str
    json_mode: str
    structured_outputs: str
    winner: str


COMPARISONS = [
    FeatureComparison(
        feature="Valid JSON output",
        json_mode="✅ Always valid JSON",
        structured_outputs="✅ Always valid JSON",
        winner="Tie"
    ),
    FeatureComparison(
        feature="Schema adherence",
        json_mode="❌ No guarantee",
        structured_outputs="✅ Guaranteed match",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Required fields",
        json_mode="❌ May be missing",
        structured_outputs="✅ Always present",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Type enforcement",
        json_mode="❌ Types may vary",
        structured_outputs="✅ Types enforced",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Enum values",
        json_mode="❌ Any value possible",
        structured_outputs="✅ Only defined values",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Refusal handling",
        json_mode="⚠️ Breaks JSON",
        structured_outputs="✅ Dedicated field",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Prompt requirement",
        json_mode='⚠️ Must say "JSON"',
        structured_outputs="❌ Not required",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Model support",
        json_mode="✅ GPT-3.5+",
        structured_outputs="⚠️ GPT-4o+ only",
        winner="JSON Mode"
    ),
    FeatureComparison(
        feature="Validation needed",
        json_mode="✅ Yes, always",
        structured_outputs="❌ No",
        winner="Structured Outputs"
    ),
    FeatureComparison(
        feature="Retry logic needed",
        json_mode="✅ Yes, for errors",
        structured_outputs="❌ No",
        winner="Structured Outputs"
    )
]


print("JSON Mode vs Structured Outputs")
print("=" * 70)
print(f"{'Feature':<25} {'JSON Mode':<20} {'Structured Outputs':<20}")
print("-" * 70)

for c in COMPARISONS:
    print(f"{c.feature:<25} {c.json_mode:<20} {c.structured_outputs:<20}")

# Count winners
structured_wins = sum(1 for c in COMPARISONS if c.winner == "Structured Outputs")
json_mode_wins = sum(1 for c in COMPARISONS if c.winner == "JSON Mode")
ties = sum(1 for c in COMPARISONS if c.winner == "Tie")

print("-" * 70)
print(f"Score: Structured Outputs {structured_wins} | JSON Mode {json_mode_wins} | Tie {ties}")
```

---

## Schema Enforcement Differences

### JSON Mode Behavior

```python
import json
from typing import Any


class JSONModeSimulator:
    """Simulates JSON mode behavior."""
    
    def __init__(self):
        # JSON mode only ensures valid JSON, not schema compliance
        pass
    
    def generate(self, prompt: str, target_schema: dict) -> dict:
        """
        Simulate JSON mode generation.
        
        Schema is IGNORED - model produces any valid JSON.
        """
        
        # These are all valid JSON mode outputs for:
        # schema = {"name": str, "age": int, "active": bool}
        
        possible_outputs = [
            # Correct output
            {"name": "Alice", "age": 30, "active": True},
            
            # Missing field - VALID in JSON mode
            {"name": "Bob", "age": 25},
            
            # Wrong type - VALID in JSON mode
            {"name": "Charlie", "age": "twenty-five", "active": True},
            
            # Extra field - VALID in JSON mode
            {"name": "Diana", "age": 28, "active": False, "role": "admin"},
            
            # Different structure entirely - VALID in JSON mode
            {"user": {"name": "Eve", "years_old": 35}},
            
            # Valid JSON but completely wrong
            ["Alice", 30, True]
        ]
        
        return possible_outputs  # Any of these could be returned


# Show the problem
simulator = JSONModeSimulator()

print("JSON Mode: Schema Not Enforced")
print("=" * 60)
print("\nRequested schema:")
print('  {"name": string, "age": integer, "active": boolean}')
print("\nPossible JSON Mode outputs (all valid JSON):")

for i, output in enumerate(simulator.generate("", {})):
    print(f"\n  Output {i + 1}: {json.dumps(output)}")
    
    # Check if it matches expected schema
    issues = []
    
    if not isinstance(output, dict):
        issues.append("Not an object")
    else:
        if "name" not in output:
            issues.append("Missing 'name'")
        if "age" not in output:
            issues.append("Missing 'age'")
        if "active" not in output:
            issues.append("Missing 'active'")
        if "age" in output and not isinstance(output.get("age"), int):
            issues.append("'age' not integer")
    
    if issues:
        print(f"    ❌ Schema issues: {', '.join(issues)}")
    else:
        print(f"    ✅ Matches schema")
```

### Structured Outputs Behavior

```python
class StructuredOutputsSimulator:
    """Simulates Structured Outputs behavior."""
    
    def __init__(self, schema: dict):
        self.schema = schema
        self._validate_schema(schema)
    
    def _validate_schema(self, schema: dict) -> None:
        """Validate schema meets requirements."""
        
        # Check additionalProperties
        if schema.get("additionalProperties") != False:
            raise ValueError("additionalProperties must be false")
        
        # Check all fields required
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for prop in properties:
            if prop not in required:
                raise ValueError(f"Property '{prop}' must be in required")
    
    def generate(self, prompt: str) -> dict:
        """
        Generate output constrained to schema.
        
        Schema is ENFORCED - only matching output possible.
        """
        
        # With Structured Outputs, ONLY this kind of output is possible:
        return {
            "name": "Alice",  # Always present, always string
            "age": 30,        # Always present, always integer
            "active": True    # Always present, always boolean
        }
        
        # These are IMPOSSIBLE with Structured Outputs:
        # - Missing fields
        # - Wrong types
        # - Extra fields
        # - Different structure


# Define proper schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"}
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False
}

print("\n\nStructured Outputs: Schema Enforced")
print("=" * 60)
print("\nSchema:")
print(json.dumps(schema, indent=2))
print("\nGuaranteed output structure:")

simulator = StructuredOutputsSimulator(schema)
output = simulator.generate("")
print(json.dumps(output, indent=2))

print("\n✅ Every output will match this structure exactly")
print("✅ No validation or retry logic needed")
```

---

## When to Use Each

### Decision Tree

```python
def choose_output_mode(
    model: str,
    needs_schema_guarantee: bool,
    schema_complexity: str,
    can_handle_retries: bool
) -> str:
    """Decide which output mode to use."""
    
    # Check model support
    structured_outputs_models = [
        "gpt-4o", "gpt-4o-mini", 
        "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"
    ]
    
    supports_structured = any(
        model.startswith(m.split("-2024")[0]) 
        for m in structured_outputs_models
    ) or model in structured_outputs_models
    
    # Decision logic
    if not supports_structured:
        return "JSON_MODE (model doesn't support Structured Outputs)"
    
    if needs_schema_guarantee:
        return "STRUCTURED_OUTPUTS (schema guarantee needed)"
    
    if schema_complexity == "simple" and can_handle_retries:
        return "JSON_MODE (simple schema, retries acceptable)"
    
    if schema_complexity in ["medium", "complex"]:
        return "STRUCTURED_OUTPUTS (complex schema benefits from guarantee)"
    
    return "STRUCTURED_OUTPUTS (recommended default for GPT-4o+)"


# Test scenarios
scenarios = [
    {
        "name": "Data extraction pipeline",
        "model": "gpt-4o",
        "needs_guarantee": True,
        "complexity": "complex",
        "can_retry": True
    },
    {
        "name": "Legacy system integration",
        "model": "gpt-3.5-turbo",
        "needs_guarantee": True,
        "complexity": "medium",
        "can_retry": True
    },
    {
        "name": "Simple yes/no response",
        "model": "gpt-4o",
        "needs_guarantee": False,
        "complexity": "simple",
        "can_retry": True
    },
    {
        "name": "Real-time classification",
        "model": "gpt-4o-mini",
        "needs_guarantee": True,
        "complexity": "medium",
        "can_retry": False
    }
]


print("\nOutput Mode Decision Examples")
print("=" * 60)

for s in scenarios:
    decision = choose_output_mode(
        s["model"],
        s["needs_guarantee"],
        s["complexity"],
        s["can_retry"]
    )
    
    print(f"\n📋 {s['name']}")
    print(f"   Model: {s['model']}")
    print(f"   Need guarantee: {s['needs_guarantee']}")
    print(f"   Complexity: {s['complexity']}")
    print(f"   → {decision}")
```

---

## Migration from JSON Mode

### Before: JSON Mode

```python
# JSON Mode approach - requires validation and retry

def extract_with_json_mode(client, text: str) -> dict:
    """Extract data using JSON mode with validation."""
    
    max_retries = 3
    
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Extract person info as JSON:
                    {
                        "name": "string",
                        "age": number,
                        "occupation": "string"
                    }
                    Return only valid JSON."""
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}  # JSON mode
        )
        
        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            
            # Manual validation
            if "name" not in data:
                raise ValueError("Missing 'name'")
            if "age" not in data:
                raise ValueError("Missing 'age'")
            if "occupation" not in data:
                raise ValueError("Missing 'occupation'")
            
            if not isinstance(data["age"], (int, float)):
                raise ValueError("'age' must be number")
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                raise
            # Retry with error feedback
            continue
    
    raise RuntimeError("Extraction failed")


# This requires:
# - Retry loop
# - JSON parsing with error handling  
# - Manual field validation
# - Type checking
# - Error recovery
```

### After: Structured Outputs

```python
from pydantic import BaseModel


class Person(BaseModel):
    """Person extracted from text."""
    name: str
    age: int
    occupation: str


def extract_with_structured_outputs(client, text: str) -> Person:
    """Extract data using Structured Outputs."""
    
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "Extract person information from the text."
            },
            {"role": "user", "content": text}
        ],
        text_format=Person
    )
    
    # Check for refusal
    if response.output_parsed is None:
        refusal = response.output[0].content[0]
        if hasattr(refusal, 'refusal'):
            raise ValueError(f"Refused: {refusal.refusal}")
    
    return response.output_parsed


# This eliminates:
# - Retry loop (not needed)
# - JSON parsing (SDK handles it)
# - Manual validation (schema enforced)
# - Type checking (types guaranteed)


print("Migration Comparison")
print("=" * 60)
print("""
JSON Mode Code:
  ├── Response format config
  ├── Retry loop (3 attempts)
  ├── JSON parsing with try/except
  ├── Field presence validation
  ├── Type validation
  └── Error recovery logic
  
  Total: ~40 lines
  
Structured Outputs Code:
  ├── Pydantic model definition
  ├── Single parse() call
  └── Optional refusal check
  
  Total: ~15 lines
  
Reduction: ~60% less code
Reliability: 100% vs ~85-95%
""")
```

---

## API Configuration Comparison

```python
# JSON Mode configuration (Chat Completions)
JSON_MODE_CONFIG = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "Return JSON..."},  # Must mention JSON!
        {"role": "user", "content": "..."}
    ],
    "response_format": {"type": "json_object"}
}


# Structured Outputs configuration (Chat Completions)
STRUCTURED_OUTPUTS_CONFIG = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "Extract info..."},  # No JSON mention needed
        {"role": "user", "content": "..."}
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,  # Required!
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
}


# Structured Outputs with Responses API (newer, cleaner)
RESPONSES_API_CONFIG = {
    "model": "gpt-4o",
    "input": [
        {"role": "system", "content": "Extract info..."},
        {"role": "user", "content": "..."}
    ],
    "text": {
        "format": {
            "type": "json_schema",
            "strict": True,
            "name": "person",
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
}


print("API Configuration Comparison")
print("=" * 60)

print("\n📋 JSON Mode:")
print('  response_format: {"type": "json_object"}')
print('  ⚠️ Must include "JSON" in prompt!')

print("\n📋 Structured Outputs (Chat Completions):")
print('  response_format: {"type": "json_schema", "json_schema": {...}}')
print('  strict: true required')
print('  Full schema definition needed')

print("\n📋 Structured Outputs (Responses API):")
print('  text.format: {"type": "json_schema", ...}')
print('  Or use SDK with Pydantic: text_format=MyModel')
```

---

## Hands-on Exercise

### Your Task

Build a migration helper that converts JSON mode code to Structured Outputs.

### Requirements

1. Parse a JSON mode configuration
2. Generate equivalent Structured Outputs config
3. Infer schema from prompt if possible
4. Show before/after comparison

<details>
<summary>💡 Hints</summary>

- Extract schema from prompt hints
- Add required Structured Outputs properties
- Validate the generated schema
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import re


@dataclass
class MigrationResult:
    """Result of migration from JSON mode to Structured Outputs."""
    
    original_config: dict
    new_config: dict
    inferred_schema: dict
    warnings: List[str]
    success: bool


class JSONModeToStructuredMigrator:
    """Migrate JSON mode to Structured Outputs."""
    
    def __init__(self):
        self.warnings: List[str] = []
    
    def migrate(self, json_mode_config: dict) -> MigrationResult:
        """Migrate configuration."""
        
        self.warnings = []
        
        # Extract components
        messages = json_mode_config.get("messages", [])
        model = json_mode_config.get("model", "gpt-4o")
        
        # Find system prompt for schema hints
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
        
        # Infer schema from prompt
        inferred_schema = self._infer_schema(system_prompt)
        
        if not inferred_schema:
            self.warnings.append(
                "Could not infer schema from prompt. Using generic object."
            )
            inferred_schema = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        
        # Build new configuration
        new_config = self._build_structured_config(
            json_mode_config,
            inferred_schema
        )
        
        return MigrationResult(
            original_config=json_mode_config,
            new_config=new_config,
            inferred_schema=inferred_schema,
            warnings=self.warnings,
            success=len(self.warnings) == 0
        )
    
    def _infer_schema(self, prompt: str) -> Optional[dict]:
        """Infer schema from system prompt."""
        
        # Look for JSON-like patterns in prompt
        # Pattern 1: {"field": "type", ...}
        json_pattern = r'\{[^{}]*"(\w+)":\s*"?(\w+)"?[^{}]*\}'
        
        matches = re.findall(json_pattern, prompt)
        
        if not matches:
            # Try to find field descriptions
            field_pattern = r'"(\w+)":\s*(\w+)'
            matches = re.findall(field_pattern, prompt)
        
        if not matches:
            return None
        
        # Build schema from matches
        properties = {}
        required = []
        
        type_mapping = {
            "string": "string",
            "str": "string",
            "number": "number",
            "int": "integer",
            "integer": "integer",
            "bool": "boolean",
            "boolean": "boolean",
            "array": "array",
            "list": "array",
            "object": "object"
        }
        
        for field_name, type_hint in matches:
            json_type = type_mapping.get(type_hint.lower(), "string")
            properties[field_name] = {"type": json_type}
            required.append(field_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _build_structured_config(
        self,
        original: dict,
        schema: dict
    ) -> dict:
        """Build Structured Outputs configuration."""
        
        # Convert messages format
        messages = original.get("messages", [])
        
        # Remove JSON instructions from system prompt
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Remove common JSON mode phrases
                content = re.sub(
                    r'(Return|Respond with|Output)( only)?( valid)? JSON[.\s]*',
                    '',
                    content,
                    flags=re.IGNORECASE
                )
                new_msg["content"] = content.strip()
            new_messages.append(new_msg)
        
        # Build new config for Responses API
        return {
            "model": original.get("model", "gpt-4o"),
            "input": new_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "extracted_data",
                    "strict": True,
                    "schema": schema
                }
            }
        }


# Test migration
migrator = JSONModeToStructuredMigrator()

# Example JSON mode config
json_mode_config = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": '''Extract person info as JSON:
            {
                "name": "string",
                "age": number,
                "occupation": "string"
            }
            Return only valid JSON.'''
        },
        {
            "role": "user",
            "content": "John is a 35-year-old software engineer."
        }
    ],
    "response_format": {"type": "json_object"}
}


result = migrator.migrate(json_mode_config)

print("Migration: JSON Mode → Structured Outputs")
print("=" * 60)

print("\n📋 ORIGINAL (JSON Mode):")
print(json.dumps(result.original_config, indent=2))

print("\n\n📋 MIGRATED (Structured Outputs):")
print(json.dumps(result.new_config, indent=2))

print("\n\n📋 INFERRED SCHEMA:")
print(json.dumps(result.inferred_schema, indent=2))

if result.warnings:
    print("\n⚠️ WARNINGS:")
    for w in result.warnings:
        print(f"  - {w}")
else:
    print("\n✅ Migration successful!")


# Generate Pydantic model
print("\n\n📋 EQUIVALENT PYDANTIC MODEL:")
print("""
from pydantic import BaseModel

class ExtractedData(BaseModel):
    name: str
    age: int
    occupation: str

# Usage:
response = client.responses.parse(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Extract person information."},
        {"role": "user", "content": "John is a 35-year-old software engineer."}
    ],
    text_format=ExtractedData
)

person = response.output_parsed
print(f"{person.name}, {person.age}, {person.occupation}")
""")
```

</details>

---

## Summary

✅ JSON mode ensures valid JSON but not schema compliance  
✅ Structured Outputs guarantees schema adherence  
✅ Structured Outputs requires GPT-4o or newer  
✅ Migration reduces code complexity by ~60%  
✅ Prefer Structured Outputs when model supports it

**Next:** [API Configuration](./03-api-configuration.md)

---

## Further Reading

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Official comparison
- [JSON Mode Documentation](https://platform.openai.com/docs/guides/text-generation/json-mode) — JSON mode details
- [Pydantic Documentation](https://docs.pydantic.dev/) — Python schema definition
