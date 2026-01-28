---
title: "Schema Limitations"
---

# Schema Limitations

## Introduction

While Structured Outputs is powerful, it has specific limitations on schema complexity, size, and supported JSON Schema features. Understanding these limits helps you design schemas that work within the constraints.

### What We'll Cover

- Maximum property and nesting limits
- Enum size restrictions
- Unsupported JSON Schema features
- Workarounds for limitations

### Prerequisites

- Schema requirements knowledge
- JSON Schema familiarity
- Understanding of schema design

---

## Property and Nesting Limits

### Size Limits

```python
from dataclasses import dataclass
from typing import Dict, Any, List
import json


@dataclass
class SchemaLimit:
    """Schema limitation."""
    
    limit_name: str
    maximum_value: int
    unit: str
    consequence: str


SCHEMA_LIMITS = [
    SchemaLimit(
        limit_name="Total object properties",
        maximum_value=5000,
        unit="properties across entire schema",
        consequence="API error if exceeded"
    ),
    SchemaLimit(
        limit_name="Nesting depth",
        maximum_value=10,
        unit="levels deep",
        consequence="API error if exceeded"
    ),
    SchemaLimit(
        limit_name="Total string length",
        maximum_value=120000,
        unit="characters (names, enums, const values)",
        consequence="API error if exceeded"
    ),
    SchemaLimit(
        limit_name="Enum values total",
        maximum_value=1000,
        unit="enum values across all enums",
        consequence="API error if exceeded"
    ),
    SchemaLimit(
        limit_name="Enum string length",
        maximum_value=15000,
        unit="characters (when >250 values)",
        consequence="API error if exceeded"
    )
]


print("Schema Limits")
print("=" * 60)

for limit in SCHEMA_LIMITS:
    print(f"\n🔢 {limit.limit_name}")
    print(f"   Maximum: {limit.maximum_value:,} {limit.unit}")
    print(f"   If exceeded: {limit.consequence}")
```

### Checking Schema Size

```python
class SchemaAnalyzer:
    """Analyze schema for limit compliance."""
    
    def __init__(self, schema: dict):
        self.schema = schema
        self.property_count = 0
        self.max_depth = 0
        self.enum_count = 0
        self.string_length = 0
    
    def analyze(self) -> dict:
        """Analyze schema and return metrics."""
        
        self._analyze_object(self.schema, depth=0)
        
        return {
            "property_count": self.property_count,
            "max_depth": self.max_depth,
            "enum_count": self.enum_count,
            "string_length": self.string_length,
            "within_limits": self._check_limits()
        }
    
    def _analyze_object(self, schema: dict, depth: int):
        """Recursively analyze schema."""
        
        self.max_depth = max(self.max_depth, depth)
        
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            self.property_count += len(properties)
            
            for prop_name, prop_def in properties.items():
                self.string_length += len(prop_name)
                
                if prop_def.get("type") == "object":
                    self._analyze_object(prop_def, depth + 1)
                elif prop_def.get("type") == "array":
                    items = prop_def.get("items", {})
                    if items.get("type") == "object":
                        self._analyze_object(items, depth + 1)
                
                if "enum" in prop_def:
                    self.enum_count += len(prop_def["enum"])
                    self.string_length += sum(
                        len(str(v)) for v in prop_def["enum"]
                    )
        
        # Check $defs
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                self.string_length += len(def_name)
                self._analyze_object(def_schema, depth)
    
    def _check_limits(self) -> dict:
        """Check against limits."""
        
        return {
            "properties": self.property_count <= 5000,
            "depth": self.max_depth <= 10,
            "enums": self.enum_count <= 1000,
            "strings": self.string_length <= 120000
        }
    
    def get_warnings(self) -> List[str]:
        """Get warning messages for near-limit items."""
        
        warnings = []
        
        if self.property_count > 4000:
            warnings.append(
                f"Properties ({self.property_count}) approaching limit of 5,000"
            )
        
        if self.max_depth > 8:
            warnings.append(
                f"Nesting depth ({self.max_depth}) approaching limit of 10"
            )
        
        if self.enum_count > 800:
            warnings.append(
                f"Enum values ({self.enum_count}) approaching limit of 1,000"
            )
        
        return warnings


# Example large schema
def create_large_schema(num_properties: int, depth: int) -> dict:
    """Create a schema with specified size."""
    
    def create_level(current_depth: int) -> dict:
        props = {}
        for i in range(min(10, num_properties)):
            if current_depth < depth:
                props[f"nested_{i}"] = create_level(current_depth + 1)
            else:
                props[f"field_{i}"] = {"type": "string"}
        
        return {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False
        }
    
    return create_level(0)


# Analyze schemas
print("\n\nSchema Size Analysis")
print("=" * 60)

# Small schema
small_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"],
    "additionalProperties": False
}

analyzer = SchemaAnalyzer(small_schema)
result = analyzer.analyze()

print("\n📋 Small Schema:")
print(f"   Properties: {result['property_count']}")
print(f"   Depth: {result['max_depth']}")
print(f"   Within limits: ✅")

# Deeper schema
deep_schema = create_large_schema(5, 8)
analyzer = SchemaAnalyzer(deep_schema)
result = analyzer.analyze()

print("\n📋 Deep Schema (8 levels):")
print(f"   Properties: {result['property_count']}")
print(f"   Depth: {result['max_depth']}")
print(f"   Within limits: {result['within_limits']}")
```

---

## Enum Limitations

### Enum Size Limits

```python
@dataclass
class EnumLimitation:
    """Enum-specific limitation."""
    
    rule: str
    limit: str
    impact: str


ENUM_LIMITATIONS = [
    EnumLimitation(
        rule="Total enum values",
        limit="1,000 values across all enums",
        impact="Must combine or reduce enums if exceeded"
    ),
    EnumLimitation(
        rule="Enum string length (>250 values)",
        limit="15,000 characters total",
        impact="Long enum values count against limit"
    ),
    EnumLimitation(
        rule="Single enum max",
        limit="No explicit limit, but affected by totals",
        impact="Very large single enums may hit string limit"
    )
]


print("Enum Limitations")
print("=" * 60)

for lim in ENUM_LIMITATIONS:
    print(f"\n🔸 {lim.rule}")
    print(f"   Limit: {lim.limit}")
    print(f"   Impact: {lim.impact}")


# Example: Problematic enum schema
LARGE_ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {
            "type": "string",
            "enum": [
                "United States of America",
                "United Kingdom of Great Britain and Northern Ireland",
                "Russian Federation",
                # ... imagine 250+ country names
            ]
        },
        "state": {
            "type": "string",
            "enum": [
                "Alabama", "Alaska", "Arizona",
                # ... all 50 states
            ]
        }
    },
    "required": ["country", "state"],
    "additionalProperties": False
}


# Workaround: Use string with pattern instead
ENUM_WORKAROUND_SCHEMA = {
    "type": "object",
    "properties": {
        # Instead of enum, use pattern for known formats
        "country_code": {
            "type": "string",
            "description": "ISO 3166-1 alpha-2 country code",
            "pattern": "^[A-Z]{2}$"  # e.g., "US", "GB", "RU"
        },
        "state_code": {
            "type": "string",
            "description": "US state abbreviation",
            "pattern": "^[A-Z]{2}$"  # e.g., "CA", "NY"
        }
    },
    "required": ["country_code", "state_code"],
    "additionalProperties": False
}


print("\n\nEnum Workaround")
print("=" * 60)
print("""
❌ Problem: Too many enum values

Original:
  country: { enum: ["United States of America", ...] }
  
✅ Solution: Use codes with pattern

Workaround:
  country_code: { 
    type: "string",
    pattern: "^[A-Z]{2}$",
    description: "ISO country code like US, GB"
  }
""")
```

---

## Unsupported JSON Schema Features

### Features Not Supported

```python
UNSUPPORTED_FEATURES = {
    "composition": {
        "features": ["allOf", "not", "dependentRequired", "dependentSchemas"],
        "reason": "Complex conditional logic",
        "workaround": "Flatten into single schema or use anyOf"
    },
    "conditionals": {
        "features": ["if", "then", "else"],
        "reason": "Conditional schema selection",
        "workaround": "Use anyOf with explicit options"
    },
    "string_constraints_finetuned": {
        "features": ["minLength", "maxLength", "pattern", "format"],
        "reason": "Not supported for fine-tuned models",
        "workaround": "Validate after extraction for fine-tuned models"
    },
    "number_constraints_finetuned": {
        "features": ["minimum", "maximum", "multipleOf"],
        "reason": "Not supported for fine-tuned models",
        "workaround": "Validate after extraction for fine-tuned models"
    },
    "object_constraints_finetuned": {
        "features": ["patternProperties"],
        "reason": "Dynamic property matching",
        "workaround": "Use explicit properties instead"
    },
    "array_constraints_finetuned": {
        "features": ["minItems", "maxItems"],
        "reason": "Not supported for fine-tuned models",
        "workaround": "Validate after extraction for fine-tuned models"
    }
}


print("Unsupported JSON Schema Features")
print("=" * 60)

for category, info in UNSUPPORTED_FEATURES.items():
    print(f"\n🚫 {category.replace('_', ' ').title()}")
    print(f"   Features: {', '.join(info['features'])}")
    print(f"   Reason: {info['reason']}")
    print(f"   Workaround: {info['workaround']}")
```

### Common Unsupported Patterns

```python
# allOf - NOT SUPPORTED
UNSUPPORTED_ALLOF = {
    "allOf": [
        {"type": "object", "properties": {"name": {"type": "string"}}},
        {"type": "object", "properties": {"age": {"type": "integer"}}}
    ]
}

# Workaround: Merge into single schema
ALLOF_WORKAROUND = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"],
    "additionalProperties": False
}


# if/then/else - NOT SUPPORTED
UNSUPPORTED_CONDITIONAL = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["personal", "business"]},
        "company_name": {"type": "string"}
    },
    "if": {
        "properties": {"type": {"const": "business"}}
    },
    "then": {
        "required": ["company_name"]
    }
}

# Workaround: Use anyOf with complete variants
CONDITIONAL_WORKAROUND = {
    "type": "object",
    "properties": {
        "account": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "const": "personal"},
                        "owner_name": {"type": "string"}
                    },
                    "required": ["type", "owner_name"],
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "const": "business"},
                        "company_name": {"type": "string"},
                        "contact_name": {"type": "string"}
                    },
                    "required": ["type", "company_name", "contact_name"],
                    "additionalProperties": False
                }
            ]
        }
    },
    "required": ["account"],
    "additionalProperties": False
}


print("\n\nWorkarounds for Unsupported Features")
print("=" * 60)

print("\n📋 allOf → Merge schemas")
print("   Before: allOf: [schema1, schema2]")
print("   After: Combine properties into single object")

print("\n📋 if/then/else → Use anyOf")
print("   Before: if: {cond} then: {schema1}")
print("   After: anyOf: [complete_variant1, complete_variant2]")

print("\n📋 not → Positive constraints")
print("   Before: not: {type: null}")
print("   After: Just specify the allowed type")
```

---

## Root Schema Restrictions

### Root Must Be Object

```python
# INVALID: Root is array
INVALID_ROOT_ARRAY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False
    }
}

# VALID: Wrap in object
VALID_ROOT_OBJECT = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False
            }
        }
    },
    "required": ["items"],
    "additionalProperties": False
}


# INVALID: Root is anyOf (discriminated union)
INVALID_ROOT_ANYOF = {
    "anyOf": [
        {
            "type": "object",
            "properties": {"success": {"type": "boolean"}, "data": {"type": "string"}},
            "required": ["success", "data"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {"success": {"type": "boolean"}, "error": {"type": "string"}},
            "required": ["success", "error"],
            "additionalProperties": False
        }
    ]
}

# VALID: Wrap anyOf in property
VALID_ANYOF_WRAPPED = {
    "type": "object",
    "properties": {
        "result": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "const": True},
                        "data": {"type": "string"}
                    },
                    "required": ["success", "data"],
                    "additionalProperties": False
                },
                {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "const": False},
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


print("Root Schema Restrictions")
print("=" * 60)

print("\n❌ Root cannot be array")
print("   Fix: Wrap in object with 'items' property")

print("\n❌ Root cannot be anyOf")
print("   Fix: Wrap anyOf in a property of the root object")

print("\n✅ Valid Root:")
print(json.dumps(VALID_ROOT_OBJECT, indent=2))
```

---

## Practical Workarounds

### Schema Simplification Strategies

```python
class SchemaSimplifier:
    """Strategies for simplifying schemas."""
    
    @staticmethod
    def flatten_deep_nesting(schema: dict, max_depth: int = 5) -> dict:
        """Flatten deeply nested schemas."""
        
        # Strategy: Move deep objects to $defs and reference them
        # This doesn't reduce actual depth but improves readability
        
        # Alternative: Restructure data model
        # Instead of: order.customer.address.street.number
        # Use: order.customer_address_street_number (flat)
        
        return schema  # Placeholder
    
    @staticmethod
    def reduce_enum_values(
        enum_values: List[str],
        max_values: int = 50
    ) -> dict:
        """Convert large enum to pattern-based validation."""
        
        if len(enum_values) <= max_values:
            return {"type": "string", "enum": enum_values}
        
        # Too many values, suggest alternatives
        return {
            "type": "string",
            "description": f"One of: {', '.join(enum_values[:5])}... and {len(enum_values) - 5} more. Model should select appropriate value."
        }
    
    @staticmethod
    def split_large_schema(
        schema: dict,
        max_properties: int = 100
    ) -> List[dict]:
        """Split large schema into multiple smaller ones."""
        
        properties = schema.get("properties", {})
        
        if len(properties) <= max_properties:
            return [schema]
        
        # Split into chunks
        prop_list = list(properties.items())
        chunks = []
        
        for i in range(0, len(prop_list), max_properties):
            chunk_props = dict(prop_list[i:i + max_properties])
            chunks.append({
                "type": "object",
                "properties": chunk_props,
                "required": list(chunk_props.keys()),
                "additionalProperties": False
            })
        
        return chunks


# Demonstration
print("\n\nSchema Simplification Strategies")
print("=" * 60)

print("""
🔧 Strategy 1: Flatten Deep Nesting
   Before: order.items[].product.category.subcategory.name
   After:  order.items[].product_category_subcategory_name
   
🔧 Strategy 2: Replace Large Enums
   Before: enum: ["value1", "value2", ... 500 values]
   After:  description: "Select from: value1, value2, etc."
           (Use post-processing validation)

🔧 Strategy 3: Split Large Schemas
   Before: 500 properties in one schema
   After:  5 extraction calls with 100 properties each
           (Merge results after extraction)

🔧 Strategy 4: Use $defs for Repetition
   Before: Same Address schema defined 3 times inline
   After:  One Address in $defs, referenced 3 times
""")
```

---

## Hands-on Exercise

### Your Task

Build a schema analyzer and optimizer that identifies limit violations and suggests fixes.

### Requirements

1. Count properties, depth, and enum values
2. Identify which limits are violated
3. Suggest specific optimizations
4. Optionally auto-fix simple issues

<details>
<summary>💡 Hints</summary>

- Track metrics during recursive traversal
- Compare against known limits
- Provide actionable suggestions
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json


@dataclass
class SchemaMetrics:
    """Metrics collected from schema analysis."""
    
    total_properties: int = 0
    max_depth: int = 0
    total_enums: int = 0
    total_enum_values: int = 0
    total_string_length: int = 0
    largest_enum_size: int = 0
    has_unsupported_features: List[str] = field(default_factory=list)


@dataclass
class LimitViolation:
    """A specific limit violation."""
    
    limit_name: str
    current_value: int
    max_value: int
    severity: str  # "error" or "warning"
    suggestion: str


class SchemaOptimizer:
    """Analyze and optimize schemas for Structured Outputs."""
    
    LIMITS = {
        "properties": 5000,
        "depth": 10,
        "enum_values": 1000,
        "string_length": 120000,
        "enum_string_length": 15000
    }
    
    UNSUPPORTED = {"allOf", "not", "if", "then", "else", 
                   "dependentRequired", "dependentSchemas"}
    
    def __init__(self, schema: dict):
        self.schema = schema
        self.metrics = SchemaMetrics()
        self.violations: List[LimitViolation] = []
    
    def analyze(self) -> SchemaMetrics:
        """Analyze schema and collect metrics."""
        
        self.metrics = SchemaMetrics()
        self._analyze_node(self.schema, depth=0)
        self._check_violations()
        return self.metrics
    
    def _analyze_node(self, node: dict, depth: int):
        """Recursively analyze a schema node."""
        
        self.metrics.max_depth = max(self.metrics.max_depth, depth)
        
        # Check for unsupported features
        for feature in self.UNSUPPORTED:
            if feature in node:
                if feature not in self.metrics.has_unsupported_features:
                    self.metrics.has_unsupported_features.append(feature)
        
        if node.get("type") == "object":
            properties = node.get("properties", {})
            self.metrics.total_properties += len(properties)
            
            for prop_name, prop_def in properties.items():
                self.metrics.total_string_length += len(prop_name)
                
                # Check for enums
                if "enum" in prop_def:
                    enum_values = prop_def["enum"]
                    self.metrics.total_enums += 1
                    self.metrics.total_enum_values += len(enum_values)
                    self.metrics.largest_enum_size = max(
                        self.metrics.largest_enum_size,
                        len(enum_values)
                    )
                    for val in enum_values:
                        self.metrics.total_string_length += len(str(val))
                
                # Recurse
                if prop_def.get("type") == "object":
                    self._analyze_node(prop_def, depth + 1)
                elif prop_def.get("type") == "array":
                    items = prop_def.get("items", {})
                    if items.get("type") == "object":
                        self._analyze_node(items, depth + 1)
                elif "anyOf" in prop_def:
                    for option in prop_def["anyOf"]:
                        if option.get("type") == "object":
                            self._analyze_node(option, depth + 1)
        
        # Check $defs
        if "$defs" in node:
            for def_name, def_schema in node["$defs"].items():
                self.metrics.total_string_length += len(def_name)
                self._analyze_node(def_schema, depth)
    
    def _check_violations(self):
        """Check metrics against limits."""
        
        self.violations = []
        
        # Properties
        if self.metrics.total_properties > self.LIMITS["properties"]:
            self.violations.append(LimitViolation(
                limit_name="Total Properties",
                current_value=self.metrics.total_properties,
                max_value=self.LIMITS["properties"],
                severity="error",
                suggestion="Split schema into multiple extraction calls"
            ))
        elif self.metrics.total_properties > self.LIMITS["properties"] * 0.8:
            self.violations.append(LimitViolation(
                limit_name="Total Properties",
                current_value=self.metrics.total_properties,
                max_value=self.LIMITS["properties"],
                severity="warning",
                suggestion="Consider reducing properties or planning for split"
            ))
        
        # Depth
        if self.metrics.max_depth > self.LIMITS["depth"]:
            self.violations.append(LimitViolation(
                limit_name="Nesting Depth",
                current_value=self.metrics.max_depth,
                max_value=self.LIMITS["depth"],
                severity="error",
                suggestion="Flatten nested objects or use $defs"
            ))
        
        # Enum values
        if self.metrics.total_enum_values > self.LIMITS["enum_values"]:
            self.violations.append(LimitViolation(
                limit_name="Total Enum Values",
                current_value=self.metrics.total_enum_values,
                max_value=self.LIMITS["enum_values"],
                severity="error",
                suggestion="Use string with description instead of large enums"
            ))
        
        # String length
        if self.metrics.total_string_length > self.LIMITS["string_length"]:
            self.violations.append(LimitViolation(
                limit_name="String Length",
                current_value=self.metrics.total_string_length,
                max_value=self.LIMITS["string_length"],
                severity="error",
                suggestion="Shorten property names or reduce enum values"
            ))
        
        # Unsupported features
        for feature in self.metrics.has_unsupported_features:
            self.violations.append(LimitViolation(
                limit_name=f"Unsupported: {feature}",
                current_value=1,
                max_value=0,
                severity="error",
                suggestion=f"Replace {feature} with supported alternative"
            ))
    
    def get_report(self) -> str:
        """Generate analysis report."""
        
        lines = [
            "Schema Analysis Report",
            "=" * 50,
            "",
            "📊 Metrics:",
            f"   Properties: {self.metrics.total_properties:,} / {self.LIMITS['properties']:,}",
            f"   Max Depth: {self.metrics.max_depth} / {self.LIMITS['depth']}",
            f"   Enum Values: {self.metrics.total_enum_values:,} / {self.LIMITS['enum_values']:,}",
            f"   String Length: {self.metrics.total_string_length:,} / {self.LIMITS['string_length']:,}",
            ""
        ]
        
        if self.metrics.has_unsupported_features:
            lines.append(f"⚠️  Unsupported Features: {', '.join(self.metrics.has_unsupported_features)}")
            lines.append("")
        
        if self.violations:
            lines.append("🚨 Violations:")
            for v in self.violations:
                icon = "❌" if v.severity == "error" else "⚠️"
                lines.append(f"   {icon} {v.limit_name}")
                lines.append(f"      Current: {v.current_value}, Max: {v.max_value}")
                lines.append(f"      Fix: {v.suggestion}")
        else:
            lines.append("✅ Schema is within all limits")
        
        return "\n".join(lines)
    
    def optimize(self) -> dict:
        """Return optimized schema (basic optimizations)."""
        
        optimized = json.loads(json.dumps(self.schema))  # Deep copy
        
        # Replace large enums with descriptions
        self._optimize_enums(optimized)
        
        return optimized
    
    def _optimize_enums(self, node: dict, path: str = "root"):
        """Replace large enums with string + description."""
        
        if node.get("type") == "object":
            properties = node.get("properties", {})
            
            for prop_name, prop_def in properties.items():
                if "enum" in prop_def:
                    enum_vals = prop_def["enum"]
                    if len(enum_vals) > 50:
                        # Replace with description
                        sample = enum_vals[:5]
                        properties[prop_name] = {
                            "type": "string",
                            "description": (
                                f"One of: {', '.join(sample)}, "
                                f"and {len(enum_vals) - 5} more options"
                            )
                        }
                elif prop_def.get("type") == "object":
                    self._optimize_enums(prop_def, f"{path}.{prop_name}")
                elif prop_def.get("type") == "array":
                    items = prop_def.get("items", {})
                    if items.get("type") == "object":
                        self._optimize_enums(items, f"{path}.{prop_name}[]")


# Test the optimizer
test_schema = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "country": {
                    "type": "string",
                    "enum": [f"Country_{i}" for i in range(100)]  # Large enum
                }
            },
            "required": ["name", "country"],
            "additionalProperties": False
        }
    },
    "required": ["user"],
    "additionalProperties": False,
    "allOf": [{"type": "object"}]  # Unsupported feature
}


optimizer = SchemaOptimizer(test_schema)
metrics = optimizer.analyze()

print(optimizer.get_report())

print("\n\n📋 Optimized Schema:")
optimized = optimizer.optimize()
print(json.dumps(optimized, indent=2))
```

</details>

---

## Summary

✅ Maximum 5,000 properties across entire schema  
✅ Maximum 10 levels of nesting depth  
✅ Maximum 1,000 enum values total  
✅ Features like `allOf`, `if/then/else` are not supported  
✅ Root must be an object, not array or anyOf

**Next:** [Streaming Structured Outputs](./07-streaming-structured-outputs.md)

---

## Further Reading

- [OpenAI Schema Limits](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) — Official limits
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html) — Full spec
- [Schema Design Patterns](https://json-schema.org/understanding-json-schema/structuring.html) — Best practices
