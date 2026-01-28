---
title: "Async Programming"
---

# Async Programming

## Overview

Async programming is essential for AI applications—making concurrent API calls to LLMs, handling streaming responses, and building responsive AI-powered services.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-understanding-async.md) | Understanding Async I/O | Concurrency vs parallelism, event loop |
| [02](./02-asyncio-fundamentals.md) | asyncio Fundamentals | async/await, coroutines |
| [03](./03-concurrent-tasks.md) | Concurrent Tasks | gather, create_task, timeouts |
| [04](./04-async-patterns-ai.md) | Async Patterns for AI | Streaming, rate limiting |
| [05](./05-error-handling.md) | Error Handling | TaskGroup, exceptions |
| [06](./06-async-libraries.md) | Async Libraries | httpx, aiohttp, FastAPI |

---

## Why Async for AI?

| Use Case | Benefit |
|----------|---------|
| **Multiple LLM calls** | Process queries in parallel |
| **Streaming responses** | Display text as it generates |
| **API rate limits** | Control concurrent requests |
| **Web services** | Handle many users concurrently |

---

## Quick Start

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(main())
```

**Output:**
```
Hello
(1 second delay)
World
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- Functions and generators
- HTTP client basics helpful

---

## Start Learning

Begin with [Understanding Async I/O](./01-understanding-async.md) to learn when and why to use async.

---

## Further Reading

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async IO](https://realpython.com/async-io-python/)
