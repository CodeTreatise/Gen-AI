---
title: "Structured Outputs & JSON Mode"
---

# Structured Outputs & JSON Mode

## Overview

Structured outputs allow you to constrain LLM responses to specific formats, ensuring reliable machine-readable data. This capability is essential for building robust applications that parse and act on AI responses.

### What We'll Cover in This Section

1. [JSON Mode vs Structured Outputs](./01-json-mode-vs-structured.md) - Understanding the difference
2. [Response Format Parameter](./02-response-format.md) - Configuration options
3. [SDK Integrations](./03-sdk-integrations.md) - Pydantic and Zod patterns
4. [JSON Schema Constraints](./04-json-schema-constraints.md) - Rules and limitations
5. [Advanced Patterns](./05-advanced-patterns.md) - Streaming, refusals, CoT
6. [Function Schemas](./06-function-schemas.md) - Tool parameter validation
7. [Use Cases](./07-use-cases.md) - Practical applications

---

## Why Structured Outputs Matter

```
┌─────────────────────────────────────────────────────────────┐
│              UNSTRUCTURED vs STRUCTURED                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  UNSTRUCTURED (Traditional):                                 │
│  ┌─────────────────────────────────────┐                    │
│  │ "The product has 4.5 stars and      │                    │
│  │ costs $29.99. It's available in     │ → Parse manually   │
│  │ blue, red, and green..."            │   (error-prone)    │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  STRUCTURED (JSON Mode):                                     │
│  ┌─────────────────────────────────────┐                    │
│  │ {                                   │                    │
│  │   "rating": 4.5,                    │ → Direct parsing   │
│  │   "price": 29.99,                   │   (reliable)       │
│  │   "colors": ["blue","red","green"]  │                    │
│  │ }                                   │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Benefit | Description |
|---------|-------------|
| Reliability | Guaranteed valid JSON/schema compliance |
| Parsing | Direct conversion to native objects |
| Validation | Automatic type checking and constraints |
| Integration | Seamless with APIs and databases |

---

## Prerequisites

- Understanding of JSON format
- Familiarity with API integration
- Basic knowledge of schema concepts

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [AI Safety](./11-ai-safety-security-fundamentals/00-ai-safety-security-fundamentals.md) | [Unit 3 Overview](./00-overview.md) | [JSON Mode vs Structured](./01-json-mode-vs-structured.md) |
