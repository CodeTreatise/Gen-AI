---
title: "Output Formatting & Structured Prompting"
---

# Output Formatting & Structured Prompting

## Overview

This lesson covers techniques for controlling the format and structure of AI-generated responses. You'll learn how to use XML tags and markdown to structure prompts, request specific output formats, and ensure the model produces reliable, parseable outputs.

> **🤖 AI Context:** Output formatting is critical for production applications where you need to parse, display, or process model outputs programmatically. Inconsistent formatting breaks applications.

---

## Lesson Navigation

| # | Lesson | Focus |
|---|--------|-------|
| 1 | [XML & Markdown Structure](./01-xml-markdown-structure.md) | Structuring prompts with XML tags and markdown |
| 2 | [Requesting Structured Outputs](./02-requesting-structured-outputs.md) | Format specification and templates |
| 3 | [JSON Output Specification](./03-json-output-specification.md) | JSON schemas, field types, examples |
| 4 | [Markdown Formatting Requests](./04-markdown-formatting-requests.md) | Headings, lists, code blocks, tables |
| 5 | [XML & Structured Text Outputs](./05-xml-structured-outputs.md) | When to use XML, tag definitions |
| 6 | [Schema Definition in Prompts](./06-schema-definition.md) | TypeScript-style schemas, field descriptions |
| 7 | [Handling Format Compliance](./07-handling-format-compliance.md) | Validation, retry, fallback strategies |

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Structure prompts using XML tags and markdown for clarity
- Request and receive consistent structured outputs
- Specify JSON schemas for reliable data extraction
- Handle format compliance with validation and retry logic
- Choose the right output format for your use case

---

## Quick Reference

### Output Format Decision Tree

```
What do you need to do with the output?
├── Parse programmatically → JSON (with schema)
├── Display to users → Markdown
├── Process in pipeline → XML or JSON
└── Mixed (display + process) → Markdown with code blocks
```

### Format Comparison

| Format | Best For | Parsing | Human Readable |
|--------|----------|---------|----------------|
| **JSON** | APIs, data extraction | ✅ Easy | ❌ Poor |
| **Markdown** | Documentation, chat | ⚠️ Manual | ✅ Excellent |
| **XML** | Structured data, mixed content | ✅ Easy | ⚠️ Moderate |
| **Plain text** | Simple responses | ❌ Hard | ✅ Excellent |

### Prompt Structure Template

```markdown
# Identity
[Who the assistant is]

# Instructions
[What to do, output format requirements]

# Examples
<example id="1">
<input>Sample input</input>
<output>Sample output</output>
</example>

# Context
[Reference data if needed]
```

---

## Prerequisites

Before starting this lesson, you should have completed:

- [User Message Construction](../03-user-message-construction/)
- [Few-Shot Prompting](../04-few-shot-prompting/)

---

## Key Concepts Preview

| Concept | Description |
|---------|-------------|
| **XML tags** | Delineate sections of prompts and outputs |
| **JSON schema** | Define expected structure and field types |
| **Format specification** | Explicit instructions for output format |
| **Validation** | Check output against expected structure |
| **Fallback strategies** | Handle malformed outputs gracefully |

---

**Next:** [XML & Markdown Structure](./01-xml-markdown-structure.md)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
