---
title: "Structured Outputs"
---

# Structured Outputs

## Introduction

Structured Outputs ensures AI responses conform exactly to a JSON schema you define. Instead of hoping the model follows your formatting instructions, you get guaranteed schema compliance every time.

### What We'll Cover

- Structured Outputs fundamentals
- Schema definition and requirements
- SDK integration with Pydantic and Zod
- Streaming structured data
- Handling refusals and edge cases

### Prerequisites

- OpenAI API experience
- Understanding of JSON Schema
- Python or JavaScript familiarity

---

## Lesson Topics

### [Structured Outputs Overview](./01-structured-outputs-overview.md)
Understand the fundamentals of Structured Outputs, how it guarantees JSON schema adherence, and why it simplifies AI application development.

### [Structured Outputs vs JSON Mode](./02-structured-outputs-vs-json-mode.md)
Compare Structured Outputs to JSON mode, understand schema enforcement differences, and learn when to use each approach.

### [API Configuration](./03-api-configuration.md)
Configure Structured Outputs in both the Responses API and Chat Completions API with proper schema setup.

### [SDK Integration](./04-sdk-integration.md)
Use Pydantic models (Python) and Zod schemas (JavaScript) for type-safe structured outputs with automatic parsing.

### [Schema Requirements](./05-schema-requirements.md)
Understand the schema rules, required properties, supported types, and how to handle optional fields.

### [Schema Limitations](./06-schema-limitations.md)
Learn about property limits, nesting depth, enum constraints, and unsupported JSON Schema features.

### [Streaming Structured Outputs](./07-streaming-structured-outputs.md)
Stream structured data progressively, handle partial JSON, and update UIs as data arrives.

### [Handling Refusals](./08-handling-refusals.md)
Detect and handle safety-based refusals gracefully with proper user messaging.

### [Use Cases and Patterns](./09-use-cases-patterns.md)
Apply Structured Outputs to data extraction, classification, chain-of-thought, and UI generation.

---

## Why Structured Outputs?

| Challenge with Regular Output | Structured Outputs Solution |
|-------------------------------|----------------------------|
| JSON sometimes malformed | Always valid JSON |
| Schema violations | Guaranteed compliance |
| Retry loops for validation | No retries needed |
| Complex prompt engineering | Simple schema definition |
| Parsing exceptions | Clean, typed responses |

---

## Quick Start

```python
from openai import OpenAI
from pydantic import BaseModel


class Person(BaseModel):
    """Extracted person information."""
    name: str
    age: int
    occupation: str


client = OpenAI()

response = client.responses.parse(
    model="gpt-4o",
    input="Extract: John is a 35-year-old software engineer.",
    text_format=Person
)

person = response.output_parsed
print(f"{person.name}, {person.age}, {person.occupation}")
# John, 35, software engineer
```

---

## Summary

Structured Outputs transforms AI responses from unpredictable text into reliable, typed data structures that your application can trust.

**Next:** [Structured Outputs Overview](./01-structured-outputs-overview.md)
