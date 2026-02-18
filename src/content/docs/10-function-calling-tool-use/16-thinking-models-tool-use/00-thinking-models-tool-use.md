---
title: "Thinking Models & Tool Use"
---

# Thinking Models & Tool Use

## Introduction

Thinking models — also called reasoning models — represent a fundamental shift in how AI approaches tool use. Instead of immediately selecting a function to call, these models engage in internal reasoning: analyzing the request, evaluating which tools are most appropriate, considering parameter values, and planning multi-step tool sequences. This results in significantly better tool selection accuracy, more precise parameter extraction, and more effective multi-tool orchestration.

All three major providers now offer thinking capabilities: OpenAI's GPT-5 family and o-series models, Google's Gemini 3 and 2.5 series, and Anthropic's Claude 4 family with extended thinking. Each implements thinking differently, and each has unique requirements for maintaining reasoning context during tool use loops.

This lesson covers how thinking models differ from standard models, the specific mechanics each provider uses to preserve reasoning across tool calls, and the best practices for getting the most out of thinking models in your tool-enabled applications.

### What we'll cover

| # | Sub-lesson | Description |
|---|-----------|-------------|
| 01 | [How Thinking Models Differ](./01-how-thinking-models-differ.md) | What makes reasoning models different, model landscape overview |
| 02 | [OpenAI Reasoning Models](./02-openai-reasoning-models.md) | GPT-5 family, o3/o4-mini, reasoning effort, summaries |
| 03 | [Reasoning Items (OpenAI)](./03-reasoning-items-openai.md) | Passing reasoning items with tool results, encrypted content |
| 04 | [Thought Signatures (Gemini)](./04-thought-signatures-gemini.md) | Gemini 3 mandatory signatures, SDK auto-handling |
| 05 | [Extended Thinking (Anthropic)](./05-extended-thinking-anthropic.md) | Thinking blocks, interleaved thinking, tool_choice limits |
| 06 | [Best Practices for Thinking Models](./06-best-practices-thinking-models.md) | Cross-provider guidance, prompting, cost optimization |

### Prerequisites

- Understanding of function calling concepts ([Lesson 01](../01-function-calling-concepts/00-function-calling-concepts.md))
- Experience with multi-turn tool calling ([Lesson 07](../07-multi-turn-function-calling/00-multi-turn-function-calling.md))
- Familiarity with parallel and sequential patterns ([Lesson 09](../09-advanced-patterns/00-advanced-patterns.md))

---

**Previous:** [Lesson 15: Built-in Platform Tools](../15-built-in-platform-tools/00-built-in-platform-tools.md) | **Next:** [Lesson 17: Multimodal Tool Use →](../17-multimodal-tool-use.md)
