---
title: "JSON Mode & Structured Outputs"
---

# JSON Mode & Structured Outputs

## Overview

This lesson covers API-level features for controlling output format. Unlike prompt-based formatting, JSON mode and Structured Outputs provide guarantees that responses will be valid JSON adhering to your schema. These features are essential for building reliable production applications.

> **🤖 AI Context:** Structured Outputs eliminate the need for complex prompt engineering and retry logic. When a schema is enforced at the API level, you get guaranteed compliance—not just "best effort."

---

## Lesson Navigation

| # | Lesson | Focus |
|---|--------|-------|
| 1 | [JSON Mode in API Calls](./01-json-mode-api.md) | Enabling JSON mode, provider parameters |
| 2 | [Structured Outputs with Schemas](./02-structured-outputs-schemas.md) | Schema-based generation, strict mode |
| 3 | [Response Format Parameter](./03-response-format-parameter.md) | json_object vs json_schema configuration |
| 4 | [Pydantic & Zod Schemas](./04-pydantic-zod-schemas.md) | SDK integration, field constraints |
| 5 | [Ensuring Valid JSON Output](./05-ensuring-valid-json.md) | API guarantees, validation layers |
| 6 | [Error Handling for Malformed Outputs](./06-error-handling-malformed.md) | Parse errors, retry, recovery |
| 7 | [JSON Mode vs Prompting](./07-json-mode-vs-prompting.md) | When to use each approach |

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Enable JSON mode across different API providers
- Use Structured Outputs with JSON Schema enforcement
- Define schemas using Pydantic (Python) and Zod (TypeScript)
- Handle edge cases like refusals and incomplete responses
- Choose between JSON mode, Structured Outputs, and prompt-based formatting

---

## Quick Reference

### Feature Comparison

| Feature | JSON Mode | Structured Outputs |
|---------|-----------|-------------------|
| **Valid JSON** | ✅ Guaranteed | ✅ Guaranteed |
| **Schema adherence** | ❌ No | ✅ Yes |
| **Field types** | ❌ Not enforced | ✅ Enforced |
| **Enum values** | ❌ Not enforced | ✅ Enforced |
| **Model support** | Wide (GPT-3.5+) | Newer (GPT-4o+) |

### When to Use What

```
Need structured output?
├── Simple JSON, no schema → JSON Mode
├── Strict schema required → Structured Outputs
├── Older model / provider → JSON Mode + Validation
└── Maximum flexibility → Prompt-based + Validation
```

### API Quick Reference

**OpenAI JSON Mode:**
```python
response_format={"type": "json_object"}
```

**OpenAI Structured Outputs:**
```python
text_format=MyPydanticModel  # Using SDK helper
# OR
response_format={"type": "json_schema", "json_schema": {...}}
```

---

## Prerequisites

Before starting this lesson, you should have completed:

- [Output Formatting & Structured Prompting](../05-output-formatting-structured-prompting/)

---

## Key Concepts Preview

| Concept | Description |
|---------|-------------|
| **JSON Mode** | Guarantees valid JSON, but not schema compliance |
| **Structured Outputs** | Guarantees JSON matches your schema exactly |
| **response_format** | API parameter to enable JSON/schema modes |
| **Pydantic/Zod** | Type-safe schema definitions in Python/TypeScript |
| **Refusals** | When model declines due to safety, output differs |

---

**Next:** [JSON Mode in API Calls](./01-json-mode-api.md)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
