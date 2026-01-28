---
title: "Async Libraries for AI"
---

# Async Libraries for AI

## Introduction

Python's async ecosystem includes powerful libraries for HTTP, files, databases, and AI frameworks. This lesson covers the most important async libraries for AI development.

### What We'll Cover

- httpx async client
- aiohttp for HTTP
- aiofiles for file I/O
- Database async drivers
- FastAPI async endpoints
- LangChain async operations

### Prerequisites

- asyncio fundamentals
- HTTP client basics

---

## httpx Async Client

### Basic Usage

```python
import asyncio
import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://api.github.com')
        return response.json()

result = asyncio.run(fetch_data())
print(result['current_user_url'])
```

### Concurrent Requests

```python
import asyncio
import httpx

async def fetch_all(urls: list[str]):
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

urls = [
    'https://api.github.com/users/python',
    'https://api.github.com/users/django',
    'https://api.github.com/users/fastapi'
]
results = asyncio.run(fetch_all(urls))
```

### Streaming Responses

```python
import asyncio
import httpx

async def stream_response():
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', 'https://httpbin.org/stream/5') as response:
            async for line in response.aiter_lines():
                print(line)

asyncio.run(stream_response())
```

### LLM API Client

```python
import asyncio
import httpx

class AsyncOpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
    
    async def chat_completion(self, messages: list[dict], **kwargs):
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": kwargs.get("model", "gpt-4"),
                    "messages": messages,
                    **kwargs
                }
            )
            return response.json()
    
    async def batch_completions(
        self, 
        message_batches: list[list[dict]], 
        max_concurrent: int = 5
    ):
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_completion(messages):
            async with semaphore:
                return await self.chat_completion(messages)
        
        return await asyncio.gather(*[
            limited_completion(msgs) for msgs in message_batches
        ])

# Usage
# client = AsyncOpenAIClient("your-api-key")
# result = asyncio.run(client.chat_completion([
#     {"role": "user", "content": "Hello!"}
# ]))
```

---

## aiohttp for HTTP

### Basic Usage

```python
import asyncio
import aiohttp

async def fetch(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

result = asyncio.run(fetch('https://api.github.com'))
```

### High-Performance Concurrent Requests

```python
import asyncio
import aiohttp

async def fetch_all(urls: list[str], max_concurrent: int = 10):
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        async def fetch_one(url):
            async with session.get(url) as response:
                return await response.json()
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

urls = [f'https://httpbin.org/get?id={i}' for i in range(50)]
results = asyncio.run(fetch_all(urls))
print(f"Fetched {len(results)} responses")
```

---

## aiofiles for File I/O

### Installation

```bash
pip install aiofiles
```

### Async File Operations

```python
import asyncio
import aiofiles

async def read_file(path: str) -> str:
    async with aiofiles.open(path, 'r') as f:
        return await f.read()

async def write_file(path: str, content: str):
    async with aiofiles.open(path, 'w') as f:
        await f.write(content)

async def main():
    # Write file
    await write_file('example.txt', 'Hello, async world!')
    
    # Read file
    content = await read_file('example.txt')
    print(content)

asyncio.run(main())
```

### Process Multiple Files

```python
import asyncio
import aiofiles
from pathlib import Path

async def process_file(path: Path) -> dict:
    """Read and process a file asynchronously."""
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
    
    return {
        'path': str(path),
        'lines': len(content.splitlines()),
        'chars': len(content)
    }

async def process_directory(directory: str) -> list[dict]:
    """Process all .txt files in a directory."""
    path = Path(directory)
    files = list(path.glob('*.txt'))
    
    tasks = [process_file(f) for f in files]
    return await asyncio.gather(*tasks)

# results = asyncio.run(process_directory('./data'))
```

---

## Async Database Access

### asyncpg (PostgreSQL)

```bash
pip install asyncpg
```

```python
import asyncio
import asyncpg

async def main():
    # Connect to database
    conn = await asyncpg.connect(
        'postgresql://user:password@localhost/database'
    )
    
    # Query
    rows = await conn.fetch('SELECT * FROM users LIMIT 10')
    for row in rows:
        print(dict(row))
    
    # Close connection
    await conn.close()

asyncio.run(main())
```

### Connection Pool

```python
import asyncio
import asyncpg

async def main():
    # Create connection pool
    pool = await asyncpg.create_pool(
        'postgresql://user:password@localhost/database',
        min_size=5,
        max_size=20
    )
    
    async def query_user(user_id: int):
        async with pool.acquire() as conn:
            return await conn.fetchrow(
                'SELECT * FROM users WHERE id = $1', 
                user_id
            )
    
    # Concurrent queries
    user_ids = [1, 2, 3, 4, 5]
    users = await asyncio.gather(*[
        query_user(uid) for uid in user_ids
    ])
    
    await pool.close()
    return users

# asyncio.run(main())
```

### SQLAlchemy Async

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/db"
)
async_session = sessionmaker(engine, class_=AsyncSession)

async def get_users():
    async with async_session() as session:
        result = await session.execute(select(User))
        return result.scalars().all()
```

---

## FastAPI Async Endpoints

### Basic Async Endpoint

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/user/{username}")
async def get_user(username: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f'https://api.github.com/users/{username}'
        )
        return response.json()
```

### Concurrent External Calls

```python
from fastapi import FastAPI
import asyncio
import httpx

app = FastAPI()

@app.get("/aggregate")
async def aggregate_data():
    """Fetch from multiple sources concurrently."""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            client.get('https://api.service1.com/data'),
            client.get('https://api.service2.com/data'),
            client.get('https://api.service3.com/data')
        )
    
    return {
        "service1": results[0].json(),
        "service2": results[1].json(),
        "service3": results[2].json()
    }
```

### Streaming Response

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def generate_stream():
    """Generate streaming content."""
    for i in range(10):
        yield f"data: chunk {i}\n\n"
        await asyncio.sleep(0.5)

@app.get("/stream")
async def stream():
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

---

## LangChain Async Operations

### Async LLM Calls

```python
from langchain_openai import ChatOpenAI
import asyncio

async def async_llm_calls():
    llm = ChatOpenAI(model="gpt-4")
    
    # Single async call
    response = await llm.ainvoke("What is AI?")
    print(response.content)
    
    # Batch async calls
    messages = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ]
    
    responses = await llm.abatch(messages)
    for msg, resp in zip(messages, responses):
        print(f"Q: {msg}")
        print(f"A: {resp.content[:100]}...")

# asyncio.run(async_llm_calls())
```

### Async Chains

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import asyncio

async def async_chain():
    llm = ChatOpenAI(model="gpt-4")
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in simple terms."
    )
    chain = prompt | llm | StrOutputParser()
    
    # Concurrent chain invocations
    topics = ["quantum computing", "blockchain", "AI"]
    
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]
    results = await asyncio.gather(*tasks)
    
    for topic, result in zip(topics, results):
        print(f"{topic}: {result[:100]}...")

# asyncio.run(async_chain())
```

### Async Streaming

```python
from langchain_openai import ChatOpenAI
import asyncio

async def stream_response():
    llm = ChatOpenAI(model="gpt-4", streaming=True)
    
    async for chunk in llm.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)
    print()

# asyncio.run(stream_response())
```

---

## Pattern: Async AI Service

```python
import asyncio
import httpx
from typing import AsyncIterator

class AsyncAIService:
    """Complete async AI service with rate limiting and streaming."""
    
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._client = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0
        )
        return self
    
    async def __aexit__(self, *args):
        await self._client.aclose()
    
    async def complete(self, prompt: str) -> str:
        """Single completion with rate limiting."""
        async with self.semaphore:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            return response.json()["choices"][0]["message"]["content"]
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming completion."""
        async with self.semaphore:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        import json
                        data = json.loads(line[6:])
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
    
    async def batch(self, prompts: list[str]) -> list[str]:
        """Batch completions with rate limiting."""
        tasks = [self.complete(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    async with AsyncAIService("api-key", max_concurrent=3) as service:
        # Single
        result = await service.complete("Hello!")
        
        # Batch
        results = await service.batch([
            "Question 1",
            "Question 2",
            "Question 3"
        ])
        
        # Stream
        async for chunk in service.stream("Tell me a story"):
            print(chunk, end="")
```

---

## Summary

| Library | Use Case | Key Methods |
|---------|----------|-------------|
| **httpx** | HTTP client | `AsyncClient`, `stream()` |
| **aiohttp** | High-perf HTTP | `ClientSession` |
| **aiofiles** | File I/O | `open()`, `read()`, `write()` |
| **asyncpg** | PostgreSQL | `connect()`, `fetch()` |
| **FastAPI** | Web framework | `async def endpoint()` |
| **LangChain** | AI chains | `ainvoke()`, `astream()` |

**Back to:** [Async Programming Overview](./00-async-programming.md)

---

## Further Reading

- [httpx Async](https://www.python-httpx.org/async/)
- [aiohttp](https://docs.aiohttp.org/)
- [FastAPI Async](https://fastapi.tiangolo.com/async/)
- [LangChain Async](https://python.langchain.com/docs/concepts/async)

<!-- 
Sources Consulted:
- httpx Docs: https://www.python-httpx.org/
- FastAPI Docs: https://fastapi.tiangolo.com/
-->
