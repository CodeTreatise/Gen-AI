---
title: "Structured Outputs Overview"
---

# Structured Outputs Overview

## Introduction

Structured Outputs is a feature that guarantees AI model responses will always conform exactly to a JSON schema you define. Instead of hoping the model follows your formatting instructions, you get deterministic schema compliance every time.

### What We'll Cover

- What Structured Outputs provides
- Key benefits over traditional approaches
- When to use Structured Outputs
- How it works under the hood

### Prerequisites

- Basic JSON understanding
- OpenAI API familiarity
- Experience with data extraction tasks

---

## What Is Structured Outputs?

Structured Outputs ensures the model generates responses that adhere to your supplied JSON Schema. The model's output is constrained at the generation level, making schema violations impossible.

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class OutputMode(Enum):
    """Output modes for AI responses."""
    
    UNSTRUCTURED = "unstructured"  # Free-form text
    JSON_MODE = "json_mode"        # Valid JSON, no schema guarantee
    STRUCTURED = "structured"      # Schema-guaranteed JSON


@dataclass
class StructuredOutputsFeature:
    """Features of Structured Outputs."""
    
    name: str
    description: str
    benefit: str


# Core features
STRUCTURED_OUTPUTS_FEATURES = [
    StructuredOutputsFeature(
        name="Schema Adherence",
        description="Output always matches your JSON Schema exactly",
        benefit="No validation needed, no retry loops"
    ),
    StructuredOutputsFeature(
        name="Type Safety",
        description="All specified types are enforced (string, number, etc.)",
        benefit="Clean parsing into typed objects"
    ),
    StructuredOutputsFeature(
        name="Required Fields",
        description="All required fields are always present",
        benefit="No missing data handling needed"
    ),
    StructuredOutputsFeature(
        name="Enum Compliance",
        description="Enum values are constrained to your list",
        benefit="No invalid category values"
    ),
    StructuredOutputsFeature(
        name="Explicit Refusals",
        description="Safety refusals have dedicated field",
        benefit="Programmatic refusal detection"
    )
]


print("Structured Outputs Features")
print("=" * 50)
for feature in STRUCTURED_OUTPUTS_FEATURES:
    print(f"\n✅ {feature.name}")
    print(f"   {feature.description}")
    print(f"   Benefit: {feature.benefit}")
```

**Output:**
```
Structured Outputs Features
==================================================

✅ Schema Adherence
   Output always matches your JSON Schema exactly
   Benefit: No validation needed, no retry loops

✅ Type Safety
   All specified types are enforced (string, number, etc.)
   Benefit: Clean parsing into typed objects

✅ Required Fields
   All required fields are always present
   Benefit: No missing data handling needed

✅ Enum Compliance
   Enum values are constrained to your list
   Benefit: No invalid category values

✅ Explicit Refusals
   Safety refusals have dedicated field
   Benefit: Programmatic refusal detection
```

---

## The Problem It Solves

### Traditional Approach Challenges

```python
@dataclass
class TraditionalApproachProblem:
    """Problems with traditional text extraction."""
    
    problem: str
    example: str
    frequency: str  # How often it occurs


TRADITIONAL_PROBLEMS = [
    TraditionalApproachProblem(
        problem="Malformed JSON",
        example='{"name": "John", "age": 30,}  # trailing comma',
        frequency="5-15% of responses"
    ),
    TraditionalApproachProblem(
        problem="Missing required fields",
        example='{"name": "John"}  # missing age field',
        frequency="10-20% of responses"
    ),
    TraditionalApproachProblem(
        problem="Wrong types",
        example='{"name": "John", "age": "thirty"}  # string instead of number',
        frequency="5-10% of responses"
    ),
    TraditionalApproachProblem(
        problem="Invalid enum values",
        example='{"status": "kinda_done"}  # not in allowed values',
        frequency="5-10% of responses"
    ),
    TraditionalApproachProblem(
        problem="Extra fields",
        example='{"name": "John", "age": 30, "extra": "data"}',
        frequency="15-25% of responses"
    ),
    TraditionalApproachProblem(
        problem="Wrapped in markdown",
        example='```json\n{"name": "John"}\n```',
        frequency="10-30% of responses"
    )
]


def calculate_failure_rate() -> float:
    """Estimate compound failure rate."""
    
    # If each problem has ~10% chance, combined is higher
    success_rate = 1.0
    
    # Each problem reduces success rate
    problem_rates = [0.10, 0.15, 0.08, 0.08, 0.20, 0.15]
    
    for rate in problem_rates:
        success_rate *= (1 - rate)
    
    return 1 - success_rate


print("Traditional Extraction Problems")
print("=" * 50)
for p in TRADITIONAL_PROBLEMS:
    print(f"\n❌ {p.problem}")
    print(f"   Example: {p.example}")
    print(f"   Frequency: {p.frequency}")

print(f"\n📊 Estimated compound failure rate: {calculate_failure_rate():.1%}")
```

### The Traditional Retry Loop

```python
import json
from typing import TypeVar, Type

T = TypeVar('T')


class TraditionalExtractionError(Exception):
    """Error during traditional extraction."""
    pass


class TraditionalExtractor:
    """Traditional extraction with retry logic."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.attempt_count = 0
    
    def extract(
        self,
        response_text: str,
        expected_type: Type[T]
    ) -> T:
        """Extract data with retries."""
        
        self.attempt_count = 0
        last_error = None
        
        for attempt in range(self.max_retries):
            self.attempt_count = attempt + 1
            
            try:
                # Step 1: Clean markdown wrapping
                cleaned = self._clean_markdown(response_text)
                
                # Step 2: Parse JSON
                data = json.loads(cleaned)
                
                # Step 3: Validate structure
                self._validate_structure(data, expected_type)
                
                # Step 4: Convert to type
                return self._convert_to_type(data, expected_type)
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                response_text = self._request_retry(response_text, last_error)
                
            except ValueError as e:
                last_error = f"Validation error: {e}"
                response_text = self._request_retry(response_text, last_error)
        
        raise TraditionalExtractionError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown code blocks."""
        
        text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()
    
    def _validate_structure(self, data: dict, expected_type: Type) -> None:
        """Validate data structure."""
        
        # Check required fields
        if hasattr(expected_type, '__dataclass_fields__'):
            for field_name in expected_type.__dataclass_fields__:
                if field_name not in data:
                    raise ValueError(f"Missing required field: {field_name}")
    
    def _convert_to_type(self, data: dict, expected_type: Type[T]) -> T:
        """Convert dict to type."""
        
        if hasattr(expected_type, '__dataclass_fields__'):
            return expected_type(**data)
        return data
    
    def _request_retry(self, original: str, error: str) -> str:
        """Request another attempt (placeholder)."""
        # In reality, would call the API again
        return original


# Demonstrate the complexity
extractor = TraditionalExtractor(max_retries=3)

print("\nTraditional Extraction Flow")
print("=" * 50)
print("""
1. Receive response
   ↓
2. Check for markdown wrapping → Clean if present
   ↓
3. Parse JSON → Retry on failure
   ↓
4. Validate required fields → Retry on missing
   ↓
5. Check types → Retry on mismatch
   ↓
6. Check enum values → Retry on invalid
   ↓
7. Return data (or fail after max retries)

❌ Complex, error-prone, expensive (API calls for retries)
""")
```

---

## How Structured Outputs Works

### Constrained Decoding

```python
class ConstrainedDecoding:
    """Explains how Structured Outputs works internally."""
    
    @staticmethod
    def explain() -> str:
        return """
Structured Outputs uses constrained decoding:

1. SCHEMA ANALYSIS
   Your JSON Schema is analyzed before generation begins.
   The model understands which tokens are valid at each position.

2. TOKEN MASKING
   At each generation step, invalid tokens are masked out.
   - After opening brace, only valid property names are allowed
   - After property name, only colon is allowed
   - After colon, only tokens valid for that type are allowed
   - And so on...

3. GUARANTEED COMPLIANCE
   Since invalid tokens are never selected, the output
   always matches the schema exactly.

Example for {"name": string, "age": number}:
   
   Position 1: Must be {
   Position 2: Must be "name" or "age" (property names)
   Position 3: Must be :
   Position 4: Must be " for string start
   ...
   
The model literally cannot generate invalid output.
"""


@dataclass
class SchemaConstraint:
    """Constraint applied during generation."""
    
    position: str
    allowed_tokens: List[str]
    reason: str


# Example constraints for a simple schema
EXAMPLE_CONSTRAINTS = [
    SchemaConstraint(
        position="Start",
        allowed_tokens=["{"],
        reason="Object must start with opening brace"
    ),
    SchemaConstraint(
        position="After {",
        allowed_tokens=['"name"', '"age"'],
        reason="Only defined property names allowed"
    ),
    SchemaConstraint(
        position="After property name",
        allowed_tokens=[":"],
        reason="Colon must follow property name"
    ),
    SchemaConstraint(
        position="After 'name':",
        allowed_tokens=['"'],
        reason="String value must start with quote"
    ),
    SchemaConstraint(
        position="After 'age':",
        allowed_tokens=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        reason="Number value must start with digit"
    ),
    SchemaConstraint(
        position="End",
        allowed_tokens=["}"],
        reason="Object must end with closing brace"
    )
]


print(ConstrainedDecoding.explain())
print("\nExample Token Constraints")
print("=" * 50)
for c in EXAMPLE_CONSTRAINTS:
    print(f"\n📍 {c.position}")
    print(f"   Allowed: {c.allowed_tokens}")
    print(f"   Why: {c.reason}")
```

---

## When to Use Structured Outputs

### Decision Matrix

```python
@dataclass
class UseCase:
    """Use case for Structured Outputs."""
    
    scenario: str
    recommendation: str
    reason: str


USE_CASE_RECOMMENDATIONS = [
    # Use Structured Outputs
    UseCase(
        scenario="Data extraction from text",
        recommendation="✅ Use Structured Outputs",
        reason="Guarantees all fields present and typed correctly"
    ),
    UseCase(
        scenario="Classification with fixed categories",
        recommendation="✅ Use Structured Outputs",
        reason="Enum constraints ensure valid categories only"
    ),
    UseCase(
        scenario="API response formatting",
        recommendation="✅ Use Structured Outputs",
        reason="Consistent structure for downstream processing"
    ),
    UseCase(
        scenario="Chain-of-thought with structure",
        recommendation="✅ Use Structured Outputs",
        reason="Captures reasoning steps in parseable format"
    ),
    UseCase(
        scenario="UI component generation",
        recommendation="✅ Use Structured Outputs",
        reason="Generates valid component trees"
    ),
    
    # Don't use Structured Outputs
    UseCase(
        scenario="Free-form creative writing",
        recommendation="❌ Use plain text",
        reason="No structure needed, constrains creativity"
    ),
    UseCase(
        scenario="Simple yes/no questions",
        recommendation="⚠️ Consider plain text",
        reason="Overhead may not be worth it for simple cases"
    ),
    UseCase(
        scenario="Older models (pre-GPT-4o)",
        recommendation="❌ Use JSON mode",
        reason="Structured Outputs not supported"
    )
]


print("When to Use Structured Outputs")
print("=" * 50)

print("\n✅ RECOMMENDED USES:")
for uc in USE_CASE_RECOMMENDATIONS:
    if uc.recommendation.startswith("✅"):
        print(f"\n  {uc.scenario}")
        print(f"    → {uc.reason}")

print("\n\n❌ NOT RECOMMENDED:")
for uc in USE_CASE_RECOMMENDATIONS:
    if not uc.recommendation.startswith("✅"):
        print(f"\n  {uc.scenario}")
        print(f"    {uc.recommendation}")
        print(f"    → {uc.reason}")
```

---

## Supported Models

```python
@dataclass
class ModelSupport:
    """Model support for Structured Outputs."""
    
    model: str
    structured_outputs: bool
    json_mode: bool
    notes: str


MODEL_COMPATIBILITY = [
    ModelSupport(
        model="gpt-4o",
        structured_outputs=True,
        json_mode=True,
        notes="Full support, recommended"
    ),
    ModelSupport(
        model="gpt-4o-mini",
        structured_outputs=True,
        json_mode=True,
        notes="Full support, cost-effective"
    ),
    ModelSupport(
        model="gpt-4o-2024-08-06+",
        structured_outputs=True,
        json_mode=True,
        notes="First version with Structured Outputs"
    ),
    ModelSupport(
        model="gpt-4-turbo",
        structured_outputs=False,
        json_mode=True,
        notes="JSON mode only, no schema guarantee"
    ),
    ModelSupport(
        model="gpt-4",
        structured_outputs=False,
        json_mode=True,
        notes="JSON mode only"
    ),
    ModelSupport(
        model="gpt-3.5-turbo",
        structured_outputs=False,
        json_mode=True,
        notes="JSON mode only"
    )
]


print("Model Compatibility")
print("=" * 60)
print(f"{'Model':<25} {'Structured':<12} {'JSON Mode':<12}")
print("-" * 60)

for m in MODEL_COMPATIBILITY:
    structured = "✅" if m.structured_outputs else "❌"
    json_mode = "✅" if m.json_mode else "❌"
    print(f"{m.model:<25} {structured:<12} {json_mode:<12}")
    if m.notes:
        print(f"  └─ {m.notes}")
```

---

## Quick Comparison

| Feature | Plain Text | JSON Mode | Structured Outputs |
|---------|------------|-----------|-------------------|
| Valid JSON | ❌ No | ✅ Yes | ✅ Yes |
| Schema compliance | ❌ No | ❌ No | ✅ Yes |
| Required fields guaranteed | ❌ No | ❌ No | ✅ Yes |
| Type safety | ❌ No | ❌ No | ✅ Yes |
| Enum enforcement | ❌ No | ❌ No | ✅ Yes |
| Explicit refusals | ❌ No | ❌ No | ✅ Yes |
| Retry logic needed | ✅ Yes | ✅ Yes | ❌ No |
| Model requirements | Any | GPT-3.5+ | GPT-4o+ |

---

## Hands-on Exercise

### Your Task

Create a comparison tool that demonstrates the difference between traditional extraction and Structured Outputs.

### Requirements

1. Define a target schema for extraction
2. Simulate traditional extraction with error handling
3. Show how Structured Outputs simplifies the code
4. Compare reliability metrics

<details>
<summary>💡 Hints</summary>

- Use dataclasses to define your schema
- Track success/failure rates
- Show code complexity difference
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import json
import random


# Target schema
class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Task extracted from text."""
    
    title: str
    description: str
    priority: Priority
    due_date: Optional[str]
    assignee: str
    tags: List[str]


# Traditional extraction approach
class TraditionalTaskExtractor:
    """Traditional extraction with all the error handling."""
    
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        
    def extract(self, response: str) -> Optional[Task]:
        """Extract task with manual validation."""
        
        self.attempts += 1
        
        try:
            # Step 1: Clean response
            cleaned = self._clean_response(response)
            
            # Step 2: Parse JSON
            data = json.loads(cleaned)
            
            # Step 3: Validate required fields
            required = ["title", "description", "priority", "assignee", "tags"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing field: {field}")
            
            # Step 4: Validate types
            if not isinstance(data["title"], str):
                raise ValueError("title must be string")
            if not isinstance(data["tags"], list):
                raise ValueError("tags must be list")
            
            # Step 5: Validate enum
            valid_priorities = ["low", "medium", "high", "critical"]
            if data["priority"] not in valid_priorities:
                raise ValueError(f"Invalid priority: {data['priority']}")
            
            # Step 6: Convert to object
            task = Task(
                title=data["title"],
                description=data["description"],
                priority=Priority(data["priority"]),
                due_date=data.get("due_date"),
                assignee=data["assignee"],
                tags=data["tags"]
            )
            
            self.successes += 1
            return task
            
        except Exception as e:
            self.failures += 1
            print(f"  ❌ Extraction failed: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Clean markdown and whitespace."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        return response
    
    def get_stats(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / self.attempts if self.attempts > 0 else 0
        }


# Structured Outputs approach (simulated)
class StructuredTaskExtractor:
    """Structured Outputs extraction - always succeeds."""
    
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        
    def extract(self, response: str) -> Task:
        """Extract with Structured Outputs guarantee."""
        
        self.attempts += 1
        
        # With Structured Outputs, parsing always succeeds
        # The response is guaranteed to match the schema
        
        # In real code:
        # response = client.responses.parse(
        #     model="gpt-4o",
        #     input=prompt,
        #     text_format=Task
        # )
        # return response.output_parsed
        
        # Simulated successful extraction
        data = json.loads(response)
        
        task = Task(
            title=data["title"],
            description=data["description"],
            priority=Priority(data["priority"]),
            due_date=data.get("due_date"),
            assignee=data["assignee"],
            tags=data["tags"]
        )
        
        self.successes += 1
        return task
    
    def get_stats(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": 0,  # Always 0 with Structured Outputs
            "success_rate": 1.0  # Always 100%
        }


# Test data - mix of good and problematic responses
TEST_RESPONSES = [
    # Valid JSON
    '{"title": "Review PR", "description": "Review the authentication PR", "priority": "high", "due_date": "2025-01-20", "assignee": "alice", "tags": ["code-review", "urgent"]}',
    
    # Missing field
    '{"title": "Deploy app", "priority": "critical", "assignee": "bob", "tags": ["deploy"]}',
    
    # Invalid priority
    '{"title": "Write docs", "description": "API documentation", "priority": "super-high", "assignee": "carol", "tags": ["docs"]}',
    
    # Wrapped in markdown
    '```json\n{"title": "Fix bug", "description": "Fix login bug", "priority": "medium", "assignee": "dave", "tags": ["bug"]}\n```',
    
    # Wrong type (tags as string)
    '{"title": "Meeting", "description": "Team sync", "priority": "low", "assignee": "eve", "tags": "meeting,sync"}',
    
    # Valid JSON
    '{"title": "Test feature", "description": "Write tests", "priority": "medium", "due_date": "2025-01-25", "assignee": "frank", "tags": ["testing"]}',
]

# For Structured Outputs, all would be valid (model guarantees it)
STRUCTURED_RESPONSES = [
    '{"title": "Review PR", "description": "Review the authentication PR", "priority": "high", "due_date": "2025-01-20", "assignee": "alice", "tags": ["code-review", "urgent"]}',
    '{"title": "Deploy app", "description": "Deploy to production", "priority": "critical", "due_date": null, "assignee": "bob", "tags": ["deploy"]}',
    '{"title": "Write docs", "description": "API documentation", "priority": "high", "due_date": null, "assignee": "carol", "tags": ["docs"]}',
    '{"title": "Fix bug", "description": "Fix login bug", "priority": "medium", "due_date": null, "assignee": "dave", "tags": ["bug"]}',
    '{"title": "Meeting", "description": "Team sync", "priority": "low", "due_date": null, "assignee": "eve", "tags": ["meeting", "sync"]}',
    '{"title": "Test feature", "description": "Write tests", "priority": "medium", "due_date": "2025-01-25", "assignee": "frank", "tags": ["testing"]}',
]


# Compare approaches
print("=" * 60)
print("COMPARISON: Traditional vs Structured Outputs")
print("=" * 60)

print("\n📋 Traditional Extraction (with problematic data):")
print("-" * 50)
traditional = TraditionalTaskExtractor()
for i, response in enumerate(TEST_RESPONSES):
    print(f"\nAttempt {i + 1}:")
    result = traditional.extract(response)
    if result:
        print(f"  ✅ Extracted: {result.title} [{result.priority.value}]")

stats = traditional.get_stats()
print(f"\n📊 Traditional Stats:")
print(f"   Success rate: {stats['success_rate']:.1%}")
print(f"   Failures: {stats['failures']}/{stats['attempts']}")


print("\n\n📋 Structured Outputs (guaranteed valid):")
print("-" * 50)
structured = StructuredTaskExtractor()
for i, response in enumerate(STRUCTURED_RESPONSES):
    print(f"\nAttempt {i + 1}:")
    result = structured.extract(response)
    print(f"  ✅ Extracted: {result.title} [{result.priority.value}]")

stats = structured.get_stats()
print(f"\n📊 Structured Outputs Stats:")
print(f"   Success rate: {stats['success_rate']:.1%}")
print(f"   Failures: {stats['failures']}/{stats['attempts']}")


print("\n\n📐 Code Complexity Comparison:")
print("-" * 50)
print("""
Traditional Extraction:
  - JSON parsing with error handling
  - Markdown cleaning logic
  - Field presence validation
  - Type checking for each field
  - Enum value validation
  - Retry loop logic
  - Error recovery strategies
  
  Lines of code: ~80-100
  
Structured Outputs:
  - Define Pydantic/dataclass model
  - Call parse() method
  - Use result directly
  
  Lines of code: ~10-15
""")
```

</details>

---

## Summary

✅ Structured Outputs guarantees JSON schema compliance  
✅ Eliminates parsing errors and retry loops  
✅ Works through constrained decoding at generation time  
✅ Requires GPT-4o or newer models  
✅ Best for data extraction, classification, and structured responses

**Next:** [Structured Outputs vs JSON Mode](./02-structured-outputs-vs-json-mode.md)

---

## Further Reading

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) — Official documentation
- [JSON Schema Specification](https://json-schema.org/docs) — Schema definition reference
- [Structured Outputs Cookbook](https://cookbook.openai.com/examples/structured_outputs_intro) — OpenAI examples
