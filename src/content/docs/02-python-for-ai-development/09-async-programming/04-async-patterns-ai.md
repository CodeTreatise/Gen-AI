---
title: "Async Patterns for AI"
---

# Async Patterns for AI

## Introduction

AI applications have unique async needs: streaming LLM responses, rate-limited API calls, and handling multiple model queries concurrently. This lesson covers patterns specific to AI development.

### What We'll Cover

- Concurrent LLM API calls
- Streaming responses
- Rate limiting with semaphores
- Producer-consumer pattern
- Async generators

### Prerequisites

- asyncio fundamentals
- Understanding of LLM APIs

---

## Concurrent LLM API Calls

### Multiple Queries in Parallel

```python
import asyncio
import httpx

async def query_llm(client: httpx.AsyncClient, prompt: str) -> str:
    """Send a prompt to an LLM API."""
    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]

async def process_multiple_prompts(prompts: list[str]) -> list[str]:
    """Process multiple prompts concurrently."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [query_llm(client, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

# Usage
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "Describe neural networks"
]
# results = asyncio.run(process_multiple_prompts(prompts))
```

### With Error Handling

```python
import asyncio
import httpx

async def safe_query_llm(client: httpx.AsyncClient, prompt: str, index: int):
    """Query LLM with error handling."""
    try:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        return {
            "index": index,
            "prompt": prompt,
            "response": data["choices"][0]["message"]["content"],
            "error": None
        }
    except httpx.TimeoutException:
        return {"index": index, "prompt": prompt, "response": None, "error": "timeout"}
    except httpx.HTTPStatusError as e:
        return {"index": index, "prompt": prompt, "response": None, "error": str(e)}
    except Exception as e:
        return {"index": index, "prompt": prompt, "response": None, "error": str(e)}

async def batch_query(prompts: list[str]):
    async with httpx.AsyncClient() as client:
        tasks = [
            safe_query_llm(client, prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)
```

---

## Streaming Responses

### Basic Streaming

```python
import asyncio
import httpx

async def stream_llm_response(prompt: str):
    """Stream response from LLM API."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
    print()  # Newline at end

# asyncio.run(stream_llm_response("Tell me a story"))
```

### Async Generator for Streaming

```python
import asyncio
import httpx
import json
from typing import AsyncGenerator

async def stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Yield response chunks as they arrive."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

async def main():
    full_response = ""
    async for chunk in stream_response("Explain async programming"):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print(f"\n\nTotal length: {len(full_response)}")

# asyncio.run(main())
```

### Multiple Streams

```python
import asyncio
from typing import AsyncGenerator

async def consume_stream(
    stream: AsyncGenerator[str, None], 
    name: str
) -> str:
    """Consume a stream and collect results."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        print(f"[{name}] Received chunk")
    return "".join(chunks)

async def main():
    prompts = ["Explain AI", "What is ML?", "Describe DL"]
    
    streams = [stream_response(p) for p in prompts]
    
    # Consume all streams concurrently
    results = await asyncio.gather(*[
        consume_stream(s, f"Stream-{i}") 
        for i, s in enumerate(streams)
    ])
    
    return results
```

---

## Rate Limiting with Semaphores

### Basic Semaphore

```python
import asyncio

async def query_with_limit(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    prompt: str
):
    """Query with concurrency limit."""
    async with semaphore:
        # Only N concurrent requests allowed
        print(f"Querying: {prompt[:30]}...")
        response = await client.post(
            "/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]}
        )
        return response.json()

async def rate_limited_batch(prompts: list[str], max_concurrent: int = 5):
    """Process prompts with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with httpx.AsyncClient(base_url="https://api.openai.com/v1") as client:
        tasks = [
            query_with_limit(semaphore, client, prompt)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
```

### Rate Limiter with Delay

```python
import asyncio
import time

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, max_tokens: int):
        self.rate = rate  # Tokens per second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

async def query_with_rate_limit(
    rate_limiter: RateLimiter,
    prompt: str
):
    await rate_limiter.acquire()
    print(f"[{time.strftime('%H:%M:%S')}] Querying: {prompt[:20]}...")
    await asyncio.sleep(0.5)  # Simulate API call
    return f"Response to: {prompt}"

async def main():
    # 2 requests per second, burst of 3
    limiter = RateLimiter(rate=2, max_tokens=3)
    
    prompts = [f"Prompt {i}" for i in range(10)]
    tasks = [query_with_rate_limit(limiter, p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    print(f"Completed {len(results)} requests")

asyncio.run(main())
```

---

## Producer-Consumer Pattern

### With asyncio.Queue

```python
import asyncio
from asyncio import Queue

async def producer(queue: Queue, prompts: list[str]):
    """Add prompts to the queue."""
    for prompt in prompts:
        await queue.put(prompt)
        print(f"Produced: {prompt[:30]}...")
    
    # Signal completion
    await queue.put(None)

async def consumer(queue: Queue, name: str, results: list):
    """Process prompts from the queue."""
    while True:
        prompt = await queue.get()
        
        if prompt is None:
            await queue.put(None)  # Pass signal to other consumers
            break
        
        print(f"[{name}] Processing: {prompt[:30]}...")
        await asyncio.sleep(1)  # Simulate LLM call
        results.append(f"Response to: {prompt}")
        queue.task_done()

async def main():
    prompts = [f"Question {i}" for i in range(10)]
    queue: Queue = Queue(maxsize=3)
    results = []
    
    # Start producer and consumers
    await asyncio.gather(
        producer(queue, prompts),
        consumer(queue, "Consumer-1", results),
        consumer(queue, "Consumer-2", results),
        consumer(queue, "Consumer-3", results)
    )
    
    print(f"\nProcessed {len(results)} items")

asyncio.run(main())
```

### With Batching

```python
import asyncio
from asyncio import Queue

async def batch_processor(queue: Queue, batch_size: int = 5):
    """Process items in batches."""
    batch = []
    
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            
            if item is None:
                break
            
            batch.append(item)
            
            if len(batch) >= batch_size:
                await process_batch(batch)
                batch = []
                
        except asyncio.TimeoutError:
            if batch:
                await process_batch(batch)
                batch = []
    
    if batch:
        await process_batch(batch)

async def process_batch(batch: list):
    """Process a batch of items (e.g., batch API call)."""
    print(f"Processing batch of {len(batch)} items")
    await asyncio.sleep(0.5)  # Simulate batch API call
    print(f"Batch complete")
```

---

## Async Context Manager Pattern

### LLM Session Manager

```python
import asyncio
import httpx
from typing import Optional

class AsyncLLMClient:
    """Async context manager for LLM API calls."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": kwargs.get("model", "gpt-4"),
                "messages": [{"role": "user", "content": prompt}],
                **kwargs
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    async def batch_complete(
        self, 
        prompts: list[str], 
        max_concurrent: int = 5
    ) -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_complete(prompt):
            async with semaphore:
                return await self.complete(prompt)
        
        return await asyncio.gather(*[
            limited_complete(p) for p in prompts
        ])

# Usage
async def main():
    async with AsyncLLMClient("api-key", "https://api.openai.com/v1") as client:
        # Single query
        response = await client.complete("Hello!")
        
        # Batch query
        responses = await client.batch_complete([
            "Question 1",
            "Question 2",
            "Question 3"
        ])
```

---

## Hands-on Exercise

### Your Task

```python
# Build an async AI query processor that:
# 1. Takes a list of prompts
# 2. Processes them with max 3 concurrent requests
# 3. Simulates streaming responses
# 4. Collects all results
# 5. Handles timeouts (2 seconds per request)
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
import random
from typing import AsyncGenerator

async def simulate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Simulate streaming LLM response."""
    response = f"Response to: {prompt}"
    words = response.split()
    
    for word in words:
        await asyncio.sleep(random.uniform(0.1, 0.3))
        yield word + " "

async def process_prompt(
    prompt: str, 
    semaphore: asyncio.Semaphore,
    index: int
) -> dict:
    """Process a single prompt with rate limiting."""
    async with semaphore:
        try:
            chunks = []
            print(f"[{index}] Starting: {prompt[:30]}...")
            
            async with asyncio.timeout(2.0):
                async for chunk in simulate_stream(prompt):
                    chunks.append(chunk)
            
            response = "".join(chunks)
            print(f"[{index}] Complete!")
            return {"index": index, "prompt": prompt, "response": response, "error": None}
            
        except asyncio.TimeoutError:
            print(f"[{index}] Timeout!")
            return {"index": index, "prompt": prompt, "response": None, "error": "timeout"}
        except Exception as e:
            return {"index": index, "prompt": prompt, "response": None, "error": str(e)}

async def process_all(prompts: list[str], max_concurrent: int = 3):
    """Process all prompts with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        process_prompt(prompt, semaphore, i)
        for i, prompt in enumerate(prompts)
    ]
    
    results = await asyncio.gather(*tasks)
    return results

async def main():
    prompts = [
        "Explain machine learning",
        "What is deep learning?",
        "Describe neural networks",
        "How does backpropagation work?",
        "What is gradient descent?",
        "Explain attention mechanism"
    ]
    
    print(f"Processing {len(prompts)} prompts with max 3 concurrent...\n")
    
    results = await process_all(prompts, max_concurrent=3)
    
    print("\n=== Results ===")
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    for r in successful:
        print(f"  [{r['index']}] {r['response'][:50]}...")

asyncio.run(main())
```
</details>

---

## Summary

✅ Use **`asyncio.gather()`** for concurrent LLM queries
✅ **Async generators** handle streaming responses elegantly
✅ **Semaphores** control concurrent API requests
✅ **Producer-consumer** pattern for queue-based processing
✅ **Async context managers** manage API client lifecycle
✅ Combine patterns for production AI systems

**Next:** [Error Handling](./05-error-handling.md)

---

## Further Reading

- [OpenAI API Streaming](https://platform.openai.com/docs/api-reference/streaming)
- [LangChain Async](https://python.langchain.com/docs/concepts/async)

<!-- 
Sources Consulted:
- Python asyncio Docs: https://docs.python.org/3/library/asyncio.html
-->
