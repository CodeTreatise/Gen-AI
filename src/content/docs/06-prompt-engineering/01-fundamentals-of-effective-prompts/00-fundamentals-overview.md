---
title: "Fundamentals of Effective Prompts"
---

# Fundamentals of Effective Prompts

## Introduction

Prompts are how you communicate with AI models. The quality of your outputs directly depends on how well you craft your inputs. This lesson covers the foundational principles that apply across all models and use cases — clarity, context, instruction structure, and framing.

Think of prompting like writing instructions for a brilliant but brand-new employee who has no context about your organization, preferences, or goals. The more precisely you explain what you want, the better the response.

### What We'll Cover

- Clarity and specificity in prompts
- Role and context setting
- Instruction ordering and priority
- Explicit vs. implicit instructions
- Positive vs. negative framing
- Balancing conciseness with detail

### Prerequisites

- Understanding of LLM fundamentals (Unit 3)
- Basic API integration experience (Unit 4)
- Access to an AI API (OpenAI, Anthropic, or Google)

---

## The Golden Rule

> **🔑 Key Insight:** Show your prompt to a colleague who has minimal context on the task. Ask them to follow the instructions. If they're confused, the AI will likely be too.

This simple test reveals ambiguity, missing context, and unclear expectations that might otherwise go unnoticed.

---

## Core Principles

### 1. Be Clear and Direct

Models perform best when you state exactly what you want:

```
❌ Vague: "Help me with my code"
✅ Clear: "Review this Python function for bugs and suggest improvements for readability"
```

### 2. Provide Context

Give the model the information it needs to understand your situation:

- What the task results will be used for
- Who the audience is
- What a successful completion looks like
- Any constraints or requirements

### 3. Use Structure

Organize complex prompts with clear sections:

```markdown
# Identity
You are a senior code reviewer.

# Task
Review the following function for:
1. Bugs and edge cases
2. Performance issues
3. Readability improvements

# Code
[code here]

# Output Format
Provide feedback as a numbered list.
```

### 4. Show, Don't Just Tell

Examples are often more effective than lengthy explanations. A single good example can convey formatting, tone, and style better than paragraphs of instructions.

---

## Lesson Structure

This lesson is divided into focused sub-lessons:

| Lesson | Topic | Key Concept |
|--------|-------|-------------|
| [01](./01-clarity-and-specificity.md) | Clarity & Specificity | Precise language, scope definition |
| [02](./02-role-and-context-setting.md) | Role & Context | "You are a..." patterns |
| [03](./03-instruction-ordering.md) | Instruction Ordering | Priority, logical flow |
| [04](./04-explicit-vs-implicit-instructions.md) | Explicit vs Implicit | When to spell things out |
| [05](./05-positive-vs-negative-framing.md) | Positive vs Negative | "Do X" vs "Don't do Y" |
| [06](./06-conciseness-vs-detail-balance.md) | Conciseness vs Detail | Finding the right balance |

---

## Quick Reference

### Prompt Structure Template

```markdown
# Identity (optional)
[Who the model should be]

# Context
[Background information needed]

# Task
[What you want done]

# Constraints
[Rules and limitations]

# Output Format
[How to structure the response]

# Examples (optional)
[Input/output pairs]
```

### Common Formatting

| Format | Use Case |
|--------|----------|
| **Markdown** | Headers, lists, emphasis |
| **XML tags** | Delineating sections, data boundaries |
| **JSON** | Structured data, schemas |
| **Numbered lists** | Sequential steps, priorities |

---

## Summary

✅ Treat prompts like instructions for a new employee with no context

✅ Be clear, specific, and direct about what you want

✅ Provide context about purpose, audience, and success criteria

✅ Use structure (headers, lists, tags) to organize complex prompts

✅ Examples often work better than lengthy explanations

**Next:** [Clarity and Specificity](./01-clarity-and-specificity.md)

---

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [Google Gemini Prompting Strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic Be Clear and Direct: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct
- Google Gemini Prompting Strategies: https://ai.google.dev/gemini-api/docs/prompting-strategies
-->
