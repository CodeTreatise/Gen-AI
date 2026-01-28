---
title: "HTTPX for Modern Python"
---

# HTTPX for Modern Python

## Introduction

HTTPX is a next-generation HTTP client that supports both sync and async operations, HTTP/2, and provides a modern API similar to requests.

### What We'll Cover

- Sync and async support
- HTTP/2
- Timeouts and connection pooling
- Streaming responses
- Comparison with requests

### Prerequisites

- Requests basics
- Python async/await helpful

---

## Installation

```bash
pip install httpx
# For HTTP/2 support:
pip install httpx[http2]
```

```python
import httpx
print(httpx.__version__)
```

---

## Synchronous Usage

### Basic Requests

```python
import httpx

# GET request
response = httpx.get('https://httpbin.org/get')
print(response.status_code)
print(response.json())

# POST with JSON
response = httpx.post(
    'https://httpbin.org/post',
    json={'key': 'value'}
)
print(response.json())
```

### Very Similar to Requests

```python
import httpx

# HTTPX API is nearly identical to requests
response = httpx.get(
    'https://api.github.com/users/python',
    headers={'Accept': 'application/json'},
    params={'per_page': 10},
    timeout=10.0
)

print(response.status_code)
print(response.json()['login'])
```

---

## Client Context

### Using Client (Recommended)

```python
import httpx

# Like requests.Session
with httpx.Client() as client:
    # Connection pooling, cookie persistence
    response = client.get('https://httpbin.org/cookies/set/name/value')
    
    # Cookies are persisted
    response = client.get('https://httpbin.org/cookies')
    print(response.json())
```

### Client Configuration

```python
import httpx

client = httpx.Client(
    base_url='https://api.github.com',
    headers={'Accept': 'application/vnd.github.v3+json'},
    timeout=30.0
)

# Now use relative URLs
response = client.get('/users/python')
print(response.json()['name'])

response = client.get('/users/python/repos')
print(len(response.json()))

client.close()
```

---

## Async Support

### Async Client

```python
import httpx
import asyncio

async def fetch_user(username: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f'https://api.github.com/users/{username}')
        return response.json()

# Run
user = asyncio.run(fetch_user('python'))
print(user['login'])
```

### Concurrent Requests

```python
import httpx
import asyncio

async def fetch_multiple_users(usernames: list[str]):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f'https://api.github.com/users/{name}')
            for name in usernames
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Fetch multiple users concurrently
usernames = ['python', 'django', 'flask']
users = asyncio.run(fetch_multiple_users(usernames))

for user in users:
    print(f"{user['login']}: {user['public_repos']} repos")
```

### Async with Rate Limiting

```python
import httpx
import asyncio

async def fetch_with_limit(client, url, semaphore):
    """Fetch with concurrency limit."""
    async with semaphore:
        response = await client.get(url)
        await asyncio.sleep(0.1)  # Rate limit
        return response.json()

async def fetch_all(urls: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [fetch_with_limit(client, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Example
urls = [f'https://httpbin.org/get?id={i}' for i in range(10)]
results = asyncio.run(fetch_all(urls))
print(f"Fetched {len(results)} responses")
```

---

## HTTP/2 Support

### Enable HTTP/2

```python
import httpx

# HTTP/2 requires the http2 extra
# pip install httpx[http2]

client = httpx.Client(http2=True)

response = client.get('https://www.google.com')
print(f"HTTP Version: {response.http_version}")  # HTTP/2

client.close()
```

### Async with HTTP/2

```python
import httpx
import asyncio

async def http2_request():
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get('https://www.cloudflare.com')
        print(f"HTTP Version: {response.http_version}")
        return response

asyncio.run(http2_request())
```

---

## Timeouts

### Configuring Timeouts

```python
import httpx

# Single timeout value
response = httpx.get('https://httpbin.org/get', timeout=5.0)

# Detailed timeout configuration
timeout = httpx.Timeout(
    connect=5.0,      # Time to establish connection
    read=10.0,        # Time to receive response
    write=5.0,        # Time to send request
    pool=5.0          # Time to acquire connection from pool
)

client = httpx.Client(timeout=timeout)
response = client.get('https://httpbin.org/delay/2')
client.close()
```

### Handling Timeouts

```python
import httpx

try:
    response = httpx.get('https://httpbin.org/delay/10', timeout=2.0)
except httpx.TimeoutException:
    print("Request timed out")
except httpx.ConnectError:
    print("Connection failed")
```

---

## Streaming Responses

### Stream Large Responses

```python
import httpx

# Stream download
with httpx.stream('GET', 'https://httpbin.org/bytes/10000') as response:
    for chunk in response.iter_bytes():
        print(f"Received {len(chunk)} bytes")
```

### Stream with Progress

```python
import httpx

def download_with_progress(url: str, filename: str):
    with httpx.stream('GET', url) as response:
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
                downloaded += len(chunk)
                
                if total:
                    progress = (downloaded / total) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')
        
        print("\nDownload complete!")

# download_with_progress('https://example.com/file.zip', 'file.zip')
```

### Async Streaming

```python
import httpx
import asyncio

async def stream_response(url: str):
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', url) as response:
            async for chunk in response.aiter_bytes():
                print(f"Received {len(chunk)} bytes")

# asyncio.run(stream_response('https://httpbin.org/bytes/5000'))
```

---

## HTTPX vs Requests

| Feature | Requests | HTTPX |
|---------|----------|-------|
| Sync Support | ✅ | ✅ |
| Async Support | ❌ | ✅ |
| HTTP/2 | ❌ | ✅ |
| Connection Pooling | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Similar API | — | ✅ |

### Migration from Requests

```python
# Requests
import requests
with requests.Session() as session:
    response = session.get('https://api.example.com')

# HTTPX (nearly identical)
import httpx
with httpx.Client() as client:
    response = client.get('https://api.example.com')
```

---

## Hands-on Exercise

### Your Task

```python
# Create an async function that:
# 1. Fetches 5 GitHub users concurrently
# 2. Uses HTTP/2
# 3. Has proper timeout handling
# 4. Returns list of (username, repo_count) tuples
```

<details>
<summary>✅ Solution</summary>

```python
import httpx
import asyncio

async def fetch_github_users(usernames: list[str]):
    """Fetch multiple GitHub users concurrently with HTTP/2."""
    
    results = []
    
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    
    async with httpx.AsyncClient(
        http2=True,
        timeout=timeout,
        headers={'Accept': 'application/vnd.github.v3+json'}
    ) as client:
        
        async def fetch_user(username: str):
            try:
                response = await client.get(
                    f'https://api.github.com/users/{username}'
                )
                response.raise_for_status()
                data = response.json()
                return (data['login'], data['public_repos'])
            except httpx.TimeoutException:
                print(f"Timeout fetching {username}")
                return (username, -1)
            except httpx.HTTPStatusError as e:
                print(f"Error fetching {username}: {e.response.status_code}")
                return (username, -1)
        
        # Fetch all concurrently
        tasks = [fetch_user(name) for name in usernames]
        results = await asyncio.gather(*tasks)
    
    return results

# Run
usernames = ['python', 'django', 'flask', 'fastapi', 'numpy']
results = asyncio.run(fetch_github_users(usernames))

print("\nGitHub Users:")
for username, repos in results:
    if repos >= 0:
        print(f"  {username}: {repos} repos")
    else:
        print(f"  {username}: (failed)")
```
</details>

---

## Summary

✅ **HTTPX** provides both sync and async APIs
✅ **`httpx.Client()`** for sync, **`httpx.AsyncClient()`** for async
✅ Enable **HTTP/2** with `http2=True`
✅ Use **`asyncio.gather()`** for concurrent requests
✅ Stream with **`client.stream()`** or **`response.aiter_bytes()`**
✅ API is very similar to requests—easy migration

**Next:** [Async with aiohttp](./05-async-aiohttp.md)

---

## Further Reading

- [HTTPX Documentation](https://www.python-httpx.org/)
- [Async Support](https://www.python-httpx.org/async/)

<!-- 
Sources Consulted:
- HTTPX Docs: https://www.python-httpx.org/
-->
