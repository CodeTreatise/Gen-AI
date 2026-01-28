---
title: "HTTP & API Clients"
---

# HTTP & API Clients

## Overview

Python excels at making HTTP requests and consuming APIs—essential skills for AI development where you'll interact with LLM APIs, data services, and external systems.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-requests-library.md) | Requests Library | GET, POST, headers, sessions |
| [02](./02-request-configuration.md) | Request Configuration | Timeouts, retries, auth |
| [03](./03-json-apis.md) | Working with JSON APIs | Parsing, error handling, pagination |
| [04](./04-httpx.md) | HTTPX Modern Client | Sync/async, HTTP/2 |
| [05](./05-async-aiohttp.md) | Async with aiohttp | Concurrent requests |
| [06](./06-api-client-patterns.md) | API Client Patterns | Classes, retries, caching |

---

## Why HTTP Clients Matter

| Use Case | Example |
|----------|---------|
| **LLM APIs** | OpenAI, Anthropic, Google AI |
| **Data Sources** | REST APIs, webhooks |
| **Embeddings** | Vector database APIs |
| **Integrations** | Third-party services |

---

## Quick Start

```python
import requests

# Simple GET request
response = requests.get('https://api.github.com')
print(response.status_code)  # 200
print(response.json())       # Parse JSON response
```

---

## Installation

```bash
pip install requests httpx aiohttp
```

```python
import requests
import httpx
import aiohttp

print(requests.__version__)  # 2.x
print(httpx.__version__)     # 0.x
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- Understanding of HTTP basics
- JSON knowledge helpful

---

## Start Learning

Begin with [Requests Library](./01-requests-library.md) to learn the most popular HTTP client.

---

## Further Reading

- [Requests Documentation](https://requests.readthedocs.io/)
- [HTTPX Documentation](https://www.python-httpx.org/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
