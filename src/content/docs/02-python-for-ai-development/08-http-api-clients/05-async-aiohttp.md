---
title: "Async HTTP with aiohttp"
---

# Async HTTP with aiohttp

## Introduction

aiohttp is Python's leading async HTTP client/server library. It's built for high-performance scenarios where you need to make many concurrent requests.

### What We'll Cover

- Async context managers
- Concurrent requests
- Session management
- Connection pooling
- Performance patterns

### Prerequisites

- Python async/await
- Basic HTTP concepts

---

## Installation

```bash
pip install aiohttp
```

```python
import aiohttp
print(aiohttp.__version__)
```

---

## Basic Usage

### Simple GET Request

```python
import aiohttp
import asyncio

async def fetch(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(response.status)
            data = await response.json()
            return data

# Run
result = asyncio.run(fetch('https://api.github.com'))
print(result['current_user_url'])
```

### Key Concept: Two Context Managers

```python
import aiohttp
import asyncio

async def example():
    # 1. Session: manages connection pool
    async with aiohttp.ClientSession() as session:
        
        # 2. Response: the actual request
        async with session.get('https://httpbin.org/get') as response:
            data = await response.json()
            return data

asyncio.run(example())
```

---

## Session Management

### Reusing Sessions (Important!)

```python
import aiohttp
import asyncio

async def make_requests():
    # Create ONE session for multiple requests
    async with aiohttp.ClientSession() as session:
        
        # Request 1
        async with session.get('https://httpbin.org/get') as resp:
            data1 = await resp.json()
        
        # Request 2 (reuses connection)
        async with session.get('https://httpbin.org/headers') as resp:
            data2 = await resp.json()
        
        return data1, data2

# ❌ Bad: Creating new session per request
async def bad_example():
    for url in urls:
        async with aiohttp.ClientSession() as session:  # Wasteful!
            async with session.get(url) as resp:
                pass
```

### Session with Default Headers

```python
import aiohttp
import asyncio

async def github_requests():
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'MyApp/1.0'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get('https://api.github.com/users/python') as resp:
            return await resp.json()

result = asyncio.run(github_requests())
print(result['login'])
```

---

## HTTP Methods

### All HTTP Methods

```python
import aiohttp
import asyncio

async def all_methods():
    async with aiohttp.ClientSession() as session:
        # GET
        async with session.get('https://httpbin.org/get') as resp:
            get_data = await resp.json()
        
        # POST with JSON
        async with session.post(
            'https://httpbin.org/post',
            json={'key': 'value'}
        ) as resp:
            post_data = await resp.json()
        
        # PUT
        async with session.put(
            'https://httpbin.org/put',
            json={'update': 'data'}
        ) as resp:
            put_data = await resp.json()
        
        # DELETE
        async with session.delete('https://httpbin.org/delete') as resp:
            delete_data = await resp.json()
        
        return get_data, post_data

asyncio.run(all_methods())
```

### POST with Different Data Types

```python
import aiohttp
import asyncio

async def post_examples():
    async with aiohttp.ClientSession() as session:
        
        # JSON data
        async with session.post(
            'https://httpbin.org/post',
            json={'key': 'value'}
        ) as resp:
            print(await resp.json())
        
        # Form data
        async with session.post(
            'https://httpbin.org/post',
            data={'field': 'value'}
        ) as resp:
            print(await resp.json())
        
        # Raw bytes
        async with session.post(
            'https://httpbin.org/post',
            data=b'raw bytes'
        ) as resp:
            print(await resp.json())

asyncio.run(post_examples())
```

---

## Concurrent Requests

### Using asyncio.gather

```python
import aiohttp
import asyncio

async def fetch_all(urls: list[str]):
    async with aiohttp.ClientSession() as session:
        
        async def fetch_one(url: str):
            async with session.get(url) as response:
                return await response.json()
        
        # Execute all requests concurrently
        tasks = [fetch_one(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

urls = [
    'https://httpbin.org/get?id=1',
    'https://httpbin.org/get?id=2',
    'https://httpbin.org/get?id=3'
]

results = asyncio.run(fetch_all(urls))
print(f"Fetched {len(results)} responses")
```

### With Error Handling

```python
import aiohttp
import asyncio

async def fetch_with_errors(urls: list[str]):
    async with aiohttp.ClientSession() as session:
        
        async def fetch_one(url: str):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return {'url': url, 'data': await resp.json(), 'error': None}
            except asyncio.TimeoutError:
                return {'url': url, 'data': None, 'error': 'timeout'}
            except aiohttp.ClientError as e:
                return {'url': url, 'data': None, 'error': str(e)}
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks)

urls = [
    'https://httpbin.org/get',
    'https://httpbin.org/status/404',
    'https://invalid-domain-xyz.com'
]

results = asyncio.run(fetch_with_errors(urls))
for r in results:
    if r['error']:
        print(f"❌ {r['url']}: {r['error']}")
    else:
        print(f"✅ {r['url']}: success")
```

### Limiting Concurrency

```python
import aiohttp
import asyncio

async def fetch_limited(urls: list[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        
        async def fetch_one(url: str):
            async with semaphore:  # Limits concurrent requests
                async with session.get(url) as response:
                    return await response.json()
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks)

# Only 10 concurrent requests at a time
urls = [f'https://httpbin.org/get?id={i}' for i in range(100)]
results = asyncio.run(fetch_limited(urls, max_concurrent=10))
```

---

## Timeouts

### Configuring Timeouts

```python
import aiohttp
import asyncio

async def with_timeout():
    # Per-request timeout
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get('https://httpbin.org/delay/2') as resp:
            return await resp.json()

# Detailed timeout
timeout = aiohttp.ClientTimeout(
    total=30,      # Total request time
    connect=5,     # Connection establishment
    sock_read=10,  # Time between data chunks
    sock_connect=5 # Socket connection
)
```

### Handling Timeout Errors

```python
import aiohttp
import asyncio

async def safe_fetch(url: str):
    timeout = aiohttp.ClientTimeout(total=5)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                return await response.json()
    
    except asyncio.TimeoutError:
        print("Request timed out")
        return None
    except aiohttp.ClientError as e:
        print(f"Client error: {e}")
        return None

result = asyncio.run(safe_fetch('https://httpbin.org/delay/10'))
```

---

## Connection Pooling

### TCPConnector Configuration

```python
import aiohttp
import asyncio

async def with_connector():
    # Configure connection pool
    connector = aiohttp.TCPConnector(
        limit=100,           # Max connections total
        limit_per_host=10,   # Max per host
        ttl_dns_cache=300,   # DNS cache TTL
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get('https://httpbin.org/get') as resp:
            return await resp.json()

asyncio.run(with_connector())
```

---

## Streaming

### Stream Response

```python
import aiohttp
import asyncio

async def stream_download(url: str, filename: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            with open(filename, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    print(f"Downloaded {f.tell()} bytes")

# asyncio.run(stream_download('https://example.com/file.zip', 'output.zip'))
```

---

## Real-World Example

### API Client Class

```python
import aiohttp
import asyncio
from typing import Optional

class AsyncAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self._session
    
    async def get(self, endpoint: str, **kwargs):
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with session.get(url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, endpoint: str, data: dict, **kwargs):
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with session.post(url, json=data, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

# Usage
async def main():
    client = AsyncAPIClient('https://api.github.com', 'token')
    try:
        user = await client.get('/users/python')
        print(user['login'])
    finally:
        await client.close()

asyncio.run(main())
```

---

## Hands-on Exercise

### Your Task

```python
# Build an async web scraper that:
# 1. Fetches 10 URLs concurrently
# 2. Limits to 3 concurrent requests
# 3. Has 5-second timeout per request
# 4. Returns list of status codes and response sizes
```

<details>
<summary>✅ Solution</summary>

```python
import aiohttp
import asyncio

async def scrape_urls(urls: list[str], max_concurrent: int = 3):
    """Scrape multiple URLs with rate limiting."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = aiohttp.ClientTimeout(total=5)
    results = []
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        async def fetch_one(url: str):
            async with semaphore:
                try:
                    async with session.get(url) as response:
                        content = await response.read()
                        return {
                            'url': url,
                            'status': response.status,
                            'size': len(content),
                            'error': None
                        }
                except asyncio.TimeoutError:
                    return {'url': url, 'status': 0, 'size': 0, 'error': 'timeout'}
                except aiohttp.ClientError as e:
                    return {'url': url, 'status': 0, 'size': 0, 'error': str(e)}
        
        tasks = [fetch_one(url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    return results

# Test
urls = [
    'https://httpbin.org/get',
    'https://httpbin.org/status/200',
    'https://httpbin.org/status/404',
    'https://httpbin.org/delay/1',
    'https://httpbin.org/bytes/1024',
    'https://httpbin.org/bytes/2048',
    'https://httpbin.org/ip',
    'https://httpbin.org/headers',
    'https://httpbin.org/user-agent',
    'https://httpbin.org/uuid'
]

results = asyncio.run(scrape_urls(urls, max_concurrent=3))

print("\nResults:")
for r in results:
    if r['error']:
        print(f"❌ {r['url']}: {r['error']}")
    else:
        print(f"✅ {r['url']}: {r['status']} ({r['size']} bytes)")

# Summary
successful = sum(1 for r in results if r['error'] is None)
print(f"\nSuccess rate: {successful}/{len(results)}")
```
</details>

---

## Summary

✅ Use **`aiohttp.ClientSession()`** as context manager
✅ **Reuse sessions** for multiple requests
✅ Use **`asyncio.gather()`** for concurrent requests
✅ Limit concurrency with **`asyncio.Semaphore`**
✅ Configure **`aiohttp.ClientTimeout`** for reliability
✅ Use **`TCPConnector`** for connection pool tuning

**Next:** [API Client Patterns](./06-api-client-patterns.md)

---

## Further Reading

- [aiohttp Documentation](https://docs.aiohttp.org/)
- [aiohttp Client Quickstart](https://docs.aiohttp.org/en/stable/client_quickstart.html)

<!-- 
Sources Consulted:
- aiohttp Docs: https://docs.aiohttp.org/
-->
