---
title: "Streaming Transport Mechanisms"
---

# Streaming Transport Mechanisms

## Introduction

LLM APIs use specific transport mechanisms to deliver streaming responses. Server-Sent Events (SSE) is the dominant choice, but understanding alternatives helps you build robust applications.

### What We'll Cover

- Server-Sent Events (SSE)
- WebSocket comparison
- HTTP/2 considerations
- Connection management

---

## Server-Sent Events (SSE)

SSE is the primary streaming mechanism for LLM APIs.

### What Is SSE?

```
SSE = Server-Sent Events
- One-way communication: Server → Client
- Uses standard HTTP connection
- Text-based protocol
- Built-in reconnection
- Native browser support
```

### SSE Protocol Format

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"1","content":"Hello"}

data: {"id":"2","content":" world"}

data: {"id":"3","content":"!"}

data: [DONE]
```

### SSE Message Structure

```python
# Each SSE message has:
# - event: (optional) event type
# - data: the payload
# - id: (optional) message ID for reconnection
# - retry: (optional) reconnection interval in ms

sse_message = """
event: message
id: 12345
data: {"delta": {"content": "Hello"}}
retry: 3000

"""
```

### Python SSE Client

```python
import httpx

def raw_sse_stream(url: str, headers: dict, data: dict):
    """Low-level SSE streaming"""
    
    with httpx.Client() as client:
        with client.stream(
            "POST",
            url,
            headers={
                **headers,
                "Accept": "text/event-stream",
            },
            json=data
        ) as response:
            
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                
                # Parse SSE messages
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    
                    for line in message.split("\n"):
                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                yield data
```

### JavaScript SSE Client

```javascript
// Browser-native EventSource (GET only)
const eventSource = new EventSource('/api/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
};

// For POST requests, use fetch with streaming
async function streamWithFetch(url, body) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        // Parse SSE format
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data !== '[DONE]') {
                    console.log(JSON.parse(data));
                }
            }
        }
    }
}
```

---

## WebSocket vs SSE Comparison

### Feature Comparison

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client | Bidirectional |
| Protocol | HTTP | WS/WSS |
| Reconnection | Built-in | Manual |
| Binary data | No (text only) | Yes |
| Firewall friendly | Very | Sometimes blocked |
| Browser support | Good | Excellent |
| Complexity | Simple | More complex |

### When to Use Each

```python
# SSE: Perfect for LLM streaming
# - One-way server pushes
# - Text-based (tokens are text)
# - Simple to implement
# - Works through proxies

# WebSocket: Better for real-time chat apps
# - User can send while receiving
# - Binary attachments
# - Maintained connection for multiple exchanges
# - Lower latency for back-and-forth

# Most LLM APIs use SSE because:
# 1. Generation is inherently one-way
# 2. HTTP infrastructure works reliably
# 3. Simpler to implement and debug
# 4. No connection upgrade complexity
```

### WebSocket for Chat Applications

```python
import asyncio
import websockets
import json

async def websocket_chat_client():
    """WebSocket client for a chat application"""
    
    async with websockets.connect("wss://chat.example.com/ws") as ws:
        # Send message
        await ws.send(json.dumps({
            "type": "message",
            "content": "Hello!"
        }))
        
        # Receive streaming response
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                
                if data["type"] == "token":
                    print(data["content"], end="", flush=True)
                elif data["type"] == "done":
                    break
            except websockets.ConnectionClosed:
                break

# Note: The LLM API call would still use SSE internally
# WebSocket is for your application's client-server communication
```

---

## HTTP/2 Streaming

HTTP/2 offers multiplexing advantages for streaming.

### HTTP/2 Benefits

```python
# HTTP/2 Features:
# - Multiplexing: Multiple streams over one connection
# - Header compression: Less overhead
# - Server push: Proactive resource sending
# - Stream prioritization: Important streams first

# For LLM streaming:
# - Single connection for multiple requests
# - Reduced connection overhead
# - Better for high-throughput applications
```

### Using HTTP/2 with Python

```python
import httpx

async def http2_streaming():
    """Stream using HTTP/2"""
    
    async with httpx.AsyncClient(http2=True) as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-...",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="")

# HTTP/2 is transparent - same code, better performance
```

### Connection Reuse

```python
# HTTP/2 connection multiplexing example

async def parallel_streams():
    """Multiple streams over single connection"""
    
    async with httpx.AsyncClient(http2=True) as client:
        # These share one TCP connection with HTTP/2
        tasks = [
            stream_request(client, "Query 1"),
            stream_request(client, "Query 2"),
            stream_request(client, "Query 3"),
        ]
        
        await asyncio.gather(*tasks)

async def stream_request(client, query):
    async with client.stream("POST", url, json={"query": query}) as resp:
        async for chunk in resp.aiter_text():
            print(f"[{query}] {chunk}")
```

---

## Connection Management

### Timeout Configuration

```python
import httpx
from openai import OpenAI

# OpenAI client timeout settings
client = OpenAI(
    timeout=httpx.Timeout(
        connect=5.0,      # Connection timeout
        read=60.0,        # Read timeout (per chunk)
        write=10.0,       # Write timeout
        pool=10.0         # Connection pool timeout
    )
)

# For streaming, read timeout applies to each chunk
# Long generations need appropriate timeouts
```

### Handling Disconnections

```python
import time
from typing import Iterator

def resilient_stream(
    client,
    messages: list,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Iterator[str]:
    """Stream with automatic retry on failure"""
    
    content_so_far = ""
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_so_far += content
                    yield content
            
            return  # Success, exit
            
        except Exception as e:
            retries += 1
            print(f"\nConnection error: {e}")
            
            if retries < max_retries:
                print(f"Retrying ({retries}/{max_retries})...")
                time.sleep(retry_delay * retries)
                
                # Continue from where we left off
                if content_so_far:
                    messages = messages + [
                        {"role": "assistant", "content": content_so_far},
                        {"role": "user", "content": "Please continue from where you left off."}
                    ]
            else:
                raise Exception(f"Failed after {max_retries} retries: {e}")
```

### Connection Pooling

```python
from openai import OpenAI
import httpx

# Production setup with connection pooling
client = OpenAI(
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_connections=100,        # Total connections
            max_keepalive_connections=20,  # Idle connections to keep
            keepalive_expiry=30.0       # Idle timeout
        )
    )
)

# Async version
async_client = OpenAI(
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20
        )
    )
)
```

---

## Implementing SSE Server

For building your own streaming API:

### FastAPI SSE Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

@app.post("/api/chat")
async def chat_stream(request: dict):
    """SSE streaming endpoint"""
    
    async def generate():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=request["messages"],
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                data = json.dumps({
                    "content": chunk.choices[0].delta.content
                })
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### Express.js SSE Endpoint

```javascript
const express = require('express');
const OpenAI = require('openai');

const app = express();
const openai = new OpenAI();

app.post('/api/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    const stream = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: req.body.messages,
        stream: true,
    });
    
    for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
            res.write(`data: ${JSON.stringify({ content })}\n\n`);
        }
    }
    
    res.write('data: [DONE]\n\n');
    res.end();
});
```

---

## Hands-on Exercise

### Your Task

Build and test an SSE parser:

```python
def parse_sse_stream(raw_stream: str) -> list:
    """
    Parse raw SSE stream into messages.
    
    Input format:
    data: {"content": "Hello"}
    
    data: {"content": " world"}
    
    data: [DONE]
    
    Output: [{"content": "Hello"}, {"content": " world"}]
    """
    
    messages = []
    
    # Your implementation here
    # 1. Split by double newline (message separator)
    # 2. For each message, find lines starting with "data: "
    # 3. Parse JSON (skip [DONE])
    # 4. Return list of parsed messages
    
    return messages

# Test data
test_stream = """data: {"content": "Hello"}

data: {"content": " world"}

data: {"content": "!"}

data: [DONE]

"""

# Test your parser
messages = parse_sse_stream(test_stream)
print("Parsed messages:", messages)

# Reconstruct content
full_content = "".join(m["content"] for m in messages)
print("Full content:", full_content)  # "Hello world!"
```

<details>
<summary>💡 Solution</summary>

```python
import json

def parse_sse_stream(raw_stream: str) -> list:
    messages = []
    
    # Split by double newline
    chunks = raw_stream.split("\n\n")
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                
                if data == "[DONE]":
                    continue
                
                try:
                    messages.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
    
    return messages
```

</details>

---

## Summary

✅ **SSE is the standard** for LLM streaming (one-way, text-based)

✅ **WebSocket** is better for bidirectional chat applications

✅ **HTTP/2** enables efficient multiplexing of multiple streams

✅ **Connection management** requires timeouts and retry logic

✅ **SSE format**: `data: {json}\n\n` with `[DONE]` terminator

✅ **Build servers** with FastAPI, Express, or other frameworks

**Next:** [Streaming vs Non-Streaming Trade-offs](./04-streaming-tradeoffs.md)

---

## Further Reading

- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — SSE specification
- [HTTP/2 Explained](https://http2.github.io/) — HTTP/2 protocol details

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Token-by-Token Generation](./02-token-by-token-generation.md) | [Streaming Response Modes](./00-streaming-response-modes.md) | [Streaming Trade-offs](./04-streaming-tradeoffs.md) |

