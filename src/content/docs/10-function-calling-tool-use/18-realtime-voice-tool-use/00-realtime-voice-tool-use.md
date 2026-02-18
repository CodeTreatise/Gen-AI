---
title: "Real-time & voice tool use"
---

# Real-time & voice tool use

## Introduction

Voice agents that can take action in the real world — checking databases, calling APIs, or controlling devices — represent the next frontier of conversational AI. Instead of just generating spoken responses, these agents use **function calling within real-time audio sessions** to execute tasks while maintaining a natural conversation flow.

This matters because traditional REST-based function calling introduces a request-response cycle that breaks the illusion of real-time conversation. The Gemini Live API and OpenAI Realtime API solve this by embedding tool use directly into persistent WebSocket sessions, allowing the model to call functions mid-conversation without dropping the audio stream.

### What we'll cover

- Using function calling with Google's Gemini Live API over WebSocket connections
- Implementing tool use in OpenAI's Realtime API for voice agent sessions
- Designing voice-specific tool patterns for responsiveness, background processing, and error recovery

### Prerequisites

- Understanding of [function calling fundamentals](../01-function-calling-fundamentals/00-function-calling-fundamentals.md)
- Familiarity with [streaming and real-time function calling](../16-streaming-realtime-function-calling/00-streaming-realtime-function-calling.md)
- Basic knowledge of WebSocket connections and async programming
- API keys for Google Gemini and/or OpenAI

---

## Lesson contents

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Live API Function Calling (Gemini)](./01-live-api-function-calling.md) | WebSocket-based real-time tool use, async `NON_BLOCKING` mode, scheduling, and multi-tool sessions |
| 02 | [Realtime API Function Calling (OpenAI)](./02-realtime-api-function-calling.md) | Event-driven function calling, session configuration, response streaming, and interruption handling |
| 03 | [Voice Agent Tool Patterns](./03-voice-agent-tool-patterns.md) | Quick-response tools, background processing, user feedback, and error recovery for voice |

---

**Previous:** [Lesson 17: Multimodal Tool Use](../17-multimodal-tool-use/04-computer-use-capabilities.md)

---

*[Back to Unit 10 Overview](../00-overview.md)*
