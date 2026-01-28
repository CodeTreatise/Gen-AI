---
title: "Streaming & Response Modes"
---

# Streaming & Response Modes

## Overview

Streaming responses deliver tokens as they're generated, providing real-time feedback to users. Understanding streaming vs. non-streaming modes is essential for building responsive AI applications.

This lesson covers how streaming works, when to use it, and how to implement it effectively across different platforms and frameworks.

## What You'll Learn

This lesson covers seven essential topics:

1. **[Why Streaming Matters](./01-why-streaming-matters.md)** — User experience and perceived performance
2. **[Token-by-Token Generation](./02-token-by-token-generation.md)** — How streaming works internally
3. **[Streaming Transport Mechanisms](./03-transport-mechanisms.md)** — SSE, WebSockets, and protocols
4. **[Streaming vs Non-Streaming Trade-offs](./04-streaming-tradeoffs.md)** — When to use each approach
5. **[First Token Latency](./05-first-token-latency.md)** — Time to first token (TTFT)
6. **[Full Response Latency](./06-full-response-latency.md)** — Total generation time
7. **[Use Cases by Mode](./07-use-cases.md)** — Matching mode to application type

## Prerequisites

Before starting this lesson, you should have:

- Completed [Model Parameters & Settings](../05-model-parameters-settings/00-model-parameters-settings.md)
- Basic understanding of HTTP and async programming
- Familiarity with API integration concepts

## Streaming at a Glance

### Non-Streaming (Batch)

```
User sends request
        │
        ▼
   [────────────────────────────]  Model generates entire response
        │
        ▼
User receives complete response (after full wait)
```

### Streaming

```
User sends request
        │
        ▼
    Token 1 → User sees "The"
    Token 2 → User sees "The answer"
    Token 3 → User sees "The answer is"
    Token 4 → User sees "The answer is 42"
    ...
    Done → User has full response
```

## Quick Comparison

| Aspect | Non-Streaming | Streaming |
|--------|---------------|-----------|
| **User waits** | Full generation time | Until first token |
| **Perceived speed** | Slow | Fast |
| **Implementation** | Simple | More complex |
| **Error handling** | Straightforward | Requires chunk handling |
| **Best for** | Background jobs | Interactive UIs |

## Code Preview

### Non-Streaming

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=False  # Default
)

# User waits until this line executes
print(response.choices[0].message.content)
```

### Streaming

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True  # Enable streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
# User sees text appear character by character
```

## Learning Path

Complete topics in order:

```
01-why-streaming-matters.md
    ↓
02-token-by-token-generation.md
    ↓
03-transport-mechanisms.md
    ↓
04-streaming-tradeoffs.md
    ↓
05-first-token-latency.md
    ↓
06-full-response-latency.md
    ↓
07-use-cases.md
```

---

## Summary

This lesson explains streaming and non-streaming response modes for LLM APIs. Understanding these modes helps you build responsive, user-friendly AI applications.

**Next:** [Why Streaming Matters](./01-why-streaming-matters.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Model Parameters](../05-model-parameters-settings/00-model-parameters-settings.md) | [AI/LLM Fundamentals](../00-overview.md) | [Why Streaming Matters](./01-why-streaming-matters.md) |

