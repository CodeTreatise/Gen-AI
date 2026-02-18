---
title: "Built-in Platform Tools"
---

# Built-in Platform Tools

## Introduction

Every major AI provider offers built-in tools — managed capabilities that the model can invoke without you writing any execution code. Unlike custom function calling, where you define the function, execute it yourself, and return results, built-in tools are executed entirely on the provider's infrastructure. The model calls them, the provider runs them, and you receive the final grounded response.

This lesson explores the built-in tools offered by OpenAI and Google Gemini, shows how to combine them with your own custom functions, and provides a decision framework for choosing between built-in and custom approaches.

### What we'll cover

- OpenAI built-in tools: web search, code interpreter, file search, and computer use
- Gemini built-in tools: Google Search grounding, Google Maps, code execution, URL context, computer use, and file search
- Combining built-in tools with custom function calling in a single request
- Decision framework for when to use built-in vs custom tools

### Prerequisites

- Understanding of function calling concepts ([Lesson 01](../01-function-calling-concepts/00-function-calling-concepts.md))
- Experience defining and handling function calls ([Lessons 02–06](../02-defining-functions/00-defining-functions.md))
- Familiarity with tool choice modes ([Lesson 09](../09-advanced-patterns/00-advanced-patterns.md))

---

## Sub-lessons

| # | Topic | Description |
|---|-------|-------------|
| 01 | [OpenAI Built-in Tools](./01-openai-built-in-tools.md) | Web search, code interpreter, file search, and computer use |
| 02 | [Gemini Built-in Tools](./02-gemini-built-in-tools.md) | Google Search grounding, Google Maps, code execution, URL context, computer use, and file search |
| 03 | [Combining Built-in with Custom Tools](./03-combining-built-in-with-custom-tools.md) | Multi-tool configurations, tool selection coordination, and use case combinations |
| 04 | [When to Use Built-in vs Custom](./04-when-to-use-built-in-vs-custom.md) | Decision framework, cost considerations, and hybrid approaches |

---

**Previous:** [Lesson 14: Structured Outputs & Strict Mode ←](../14-structured-outputs-strict-mode/00-structured-outputs-strict-mode.md)

**Next:** [Lesson 16: Thinking Models & Tool Use →](../16-thinking-models-tool-use/00-thinking-models-tool-use.md)

---

*[Back to Unit 10 Overview](../00-overview.md)*
