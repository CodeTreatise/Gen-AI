---
title: "Structured Outputs & Strict Mode"
---

# Structured Outputs & Strict Mode

## Overview

When AI models call functions, they generate JSON arguments that your code must parse and execute. Without strict guarantees, models can hallucinate fields, produce wrong types, or omit required parameters — breaking your application at runtime. **Structured outputs with strict mode** solve this by constraining the model's output to always conform to your defined JSON Schema, eliminating an entire category of production failures.

This lesson explores how strict mode works across major AI providers, the schema requirements it enforces, how to handle optional parameters within those constraints, and the performance trade-offs involved.

---

## What we'll cover

- **What strict mode is** — guaranteed schema adherence through constrained decoding
- **Enabling strict mode** — provider-specific configuration (OpenAI, Anthropic, Gemini)
- **Schema requirements** — rules your schema must follow (`additionalProperties: false`, all fields `required`, etc.)
- **Optional parameters** — the nullable union type pattern for emulating optional fields
- **Unsupported schema features** — composition keywords and format constraints that aren't available
- **Schema caching and latency** — first-request compilation cost and cache behavior
- **Structured outputs vs JSON mode** — when to use which approach

---

## Prerequisites

- Understanding of function calling basics ([Lesson 01](../01-function-calling-fundamentals/00-function-calling-fundamentals.md))
- Familiarity with JSON Schema structure ([Lesson 03](../03-json-schema-for-functions/00-json-schema-for-functions.md))
- Experience defining tool schemas ([Lesson 04](../04-defining-tool-schemas/00-defining-tool-schemas.md))

---

## Sub-lessons

| # | Topic | Description |
|---|-------|-------------|
| 01 | [What Is Strict Mode](./01-what-is-strict-mode.md) | Guaranteed schema adherence, constrained decoding, and why it matters |
| 02 | [Enabling Strict Mode](./02-enabling-strict-mode.md) | Provider-specific configuration for OpenAI, Anthropic, and Gemini |
| 03 | [Schema Requirements for Strict Mode](./03-schema-requirements.md) | Mandatory rules your schema must follow |
| 04 | [Optional Parameters in Strict Mode](./04-optional-parameters.md) | The nullable union type pattern for flexibility |
| 05 | [Unsupported Schema Features](./05-unsupported-schema-features.md) | Composition keywords and constraints that aren't available |
| 06 | [Schema Caching and Latency](./06-schema-caching-and-latency.md) | Performance characteristics and cache behavior |
| 07 | [Structured Outputs vs JSON Mode](./07-structured-outputs-vs-json-mode.md) | Feature comparison and migration guidance |

---

**Previous:** [Lesson 13: Tool Versioning & Lifecycle](../13-tool-versioning-lifecycle/00-tool-versioning-lifecycle.md)

**Next:** [Lesson 15: Built-in Platform Tools](../15-built-in-platform-tools.md)
