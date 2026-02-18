---
title: "Multimodal Tool Use"
---

# Multimodal Tool Use

## Introduction

Function calling becomes dramatically more powerful when combined with multimodal capabilities — images, documents, and even full desktop environments. Instead of limiting AI interactions to text-based inputs and outputs, multimodal tool use lets models analyze screenshots before calling functions, return images and documents from tool responses, and even control browser interfaces autonomously.

This lesson explores the cutting edge of AI integration: the intersection of vision, document understanding, and function calling across OpenAI, Anthropic, and Google Gemini.

---

## What we'll cover

This lesson is divided into four sub-lessons:

| # | Sub-Lesson | Description |
|---|-----------|-------------|
| 01 | [Vision with Function Calling](./01-vision-with-function-calling.md) | Sending images as input alongside tool definitions — analyze-then-call patterns, screenshot analysis, and document understanding |
| 02 | [Multimodal Function Responses](./02-multimodal-function-responses.md) | Functions that return images and documents (Gemini 3 feature), `inlineData` with MIME types, and `displayName` references |
| 03 | [Supported MIME Types and Limits](./03-supported-mime-types.md) | Provider-specific format support matrix, size constraints, and token cost calculations for image inputs |
| 04 | [Computer Use Capabilities](./04-computer-use-capabilities.md) | Browser automation agents with Anthropic, OpenAI, and Gemini — agent loops, UI actions, safety checks, and sandboxing |

---

## Prerequisites

Before starting this lesson, you should be familiar with:

- Function calling fundamentals ([Lesson 03](../03-core-function-calling-flow/00-core-function-calling-flow.md))
- Multi-turn tool use patterns ([Lesson 05](../05-multi-turn-tool-use/00-multi-turn-tool-use.md))
- Provider-specific function calling ([Lessons 07–09](../07-openai-function-calling/00-openai-function-calling.md))
- Basic understanding of image/vision APIs from [Unit 13: Image & Multimodal AI](../../13-image-multimodal-ai/)

---

## Learning path

We recommend working through the sub-lessons in order:

1. **Start with vision + function calling** — learn how images flow into the tool-use pipeline as additional context for the model's decisions
2. **Explore multimodal responses** — understand how functions can return rich media (images, PDFs) back to the model, not just JSON text
3. **Review MIME types and limits** — know the practical constraints before building production features
4. **Discover computer use** — see how vision + tool use combine into fully autonomous browser agents

> **🤖 AI Context:** Multimodal tool use represents the convergence of two major AI capabilities — vision understanding and agentic tool use. This combination enables use cases like automated UI testing, visual data extraction, and autonomous web navigation that were impossible with text-only function calling.

---

**Next:** [Vision with Function Calling →](./01-vision-with-function-calling.md)

---

*[← Back to Unit 10 Overview](../00-overview.md)*
