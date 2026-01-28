---
title: "ReadableStream API"
---

# ReadableStream API

## Introduction

The ReadableStream API provides low-level access to streaming data in both browsers and Node.js. When you need more control than EventSource offers—like POST requests with headers—ReadableStream is your tool.

### What We'll Cover

- ReadableStream fundamentals
- The `getReader()` method
- Reading chunks with async iteration
- TextDecoder for text processing
- Browser vs Node.js patterns

### Prerequisites

- Basic JavaScript async/await
- Understanding of SSE protocol

---

## ReadableStream Overview

```mermaid
flowchart LR
    subgraph Source
        A[HTTP Response]
    end
    
    subgraph ReadableStream
        B[.body]
    end
    
    subgraph Reader
        C[.getReader()]
    end
    
    subgraph Consumer
        D[.read() loop]
    end
    
    A --> B --> C --> D
```

### Getting a ReadableStream

```javascript
// From fetch response
const response = await fetch("/api/stream", {
    method: "POST",
    body: JSON.stringify({ prompt: "Hello" }),
    headers: { "Content-Type": "application/json" }
});

// The body is a ReadableStream
const stream = response.body;
console.log(stream instanceof ReadableStream); // true
```

---

## The Reader Pattern

### Getting a Reader

```javascript
const reader = stream.getReader();
```

> **Warning:** Once you get a reader, the stream is locked. Only one reader can exist at a time.

### Reading Chunks

```javascript
async function readStream(stream) {
    const reader = stream.getReader();
    
    try {
        while (true) {
            // read() returns { done: boolean, value: Uint8Array }
            const { done, value } = await reader.read();
            
            if (done) {
                console.log("Stream finished");
                break;
            }
            
            // value is a Uint8Array (raw bytes)
            console.log("Received chunk:", value.length, "bytes");
        }
    } finally {
        // Always release the lock
        reader.releaseLock();
    }
}
```

### Decoding Text

Raw chunks are bytes, not text:

```javascript
async function readTextStream(stream) {
    const reader = stream.getReader();
    const decoder = new TextDecoder("utf-8");
    let result = "";
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            // Decode bytes to text
            // stream: true handles split multi-byte characters
            const text = decoder.decode(value, { stream: true });
            result += text;
            console.log("Chunk:", text);
        }
        
        // Flush any remaining bytes
        result += decoder.decode();
        
    } finally {
        reader.releaseLock();
    }
    
    return result;
}
```

---

## Async Iteration Pattern

Modern environments support `for await...of`:

```javascript
async function streamWithAsyncIterator(stream) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    
    try {
        // Manual async iterator
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            yield decoder.decode(value, { stream: true });
        }
    } finally {
        reader.releaseLock();
    }
}

// Usage
for await (const chunk of streamWithAsyncIterator(response.body)) {
    process.stdout.write(chunk);
}
```

### Using `pipeThrough`

Transform streams with `pipeThrough`:

```javascript
// TextDecoderStream handles decoding automatically
async function* readTextChunks(response) {
    const reader = response.body
        .pipeThrough(new TextDecoderStream())
        .getReader();
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            yield value; // Already a string
        }
    } finally {
        reader.releaseLock();
    }
}
```

---

## Complete Streaming Example

```javascript
class StreamingClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    async *stream(prompt, options = {}) {
        const response = await fetch(`${this.baseUrl}/v1/responses`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${options.apiKey}`
            },
            body: JSON.stringify({
                model: options.model || "gpt-4.1",
                input: prompt,
                stream: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete SSE events
                const events = buffer.split("\n\n");
                buffer = events.pop(); // Keep incomplete event
                
                for (const event of events) {
                    const parsed = this.parseSSE(event);
                    if (parsed) yield parsed;
                }
            }
            
            // Process any remaining buffer
            if (buffer.trim()) {
                const parsed = this.parseSSE(buffer);
                if (parsed) yield parsed;
            }
            
        } finally {
            reader.releaseLock();
        }
    }
    
    parseSSE(eventString) {
        const lines = eventString.split("\n");
        let data = null;
        let eventType = "message";
        
        for (const line of lines) {
            if (line.startsWith("data: ")) {
                const content = line.slice(6);
                if (content === "[DONE]") {
                    return { type: "done", data: null };
                }
                try {
                    data = JSON.parse(content);
                } catch {
                    data = content;
                }
            } else if (line.startsWith("event: ")) {
                eventType = line.slice(7);
            }
        }
        
        return data ? { type: eventType, data } : null;
    }
}

// Usage
const client = new StreamingClient("https://api.openai.com");

for await (const event of client.stream("Write a haiku", { apiKey: API_KEY })) {
    if (event.type === "done") break;
    
    if (event.type === "response.output_text.delta") {
        process.stdout.write(event.data.delta);
    }
}
```

---

## Node.js Streams

### Using `node-fetch` or Native Fetch

```javascript
import { Readable } from "stream";

async function nodeStreamExample() {
    const response = await fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-4.1",
            input: "Hello",
            stream: true
        })
    });
    
    // Convert to Node.js readable stream
    const nodeStream = Readable.fromWeb(response.body);
    
    for await (const chunk of nodeStream) {
        const text = chunk.toString();
        process.stdout.write(text);
    }
}
```

### Using `axios` with Streams

```javascript
import axios from "axios";

async function axiosStreamExample() {
    const response = await axios({
        method: "post",
        url: "https://api.openai.com/v1/responses",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        data: {
            model: "gpt-4.1",
            input: "Hello",
            stream: true
        },
        responseType: "stream"
    });
    
    // axios returns a Node.js stream directly
    for await (const chunk of response.data) {
        const text = chunk.toString();
        process.stdout.write(text);
    }
}
```

### Using `httpx` in Python

```python
import httpx

async def stream_with_httpx():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/responses",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "gpt-4.1",
                "input": "Hello",
                "stream": True
            }
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
```

---

## Handling Split Characters

UTF-8 characters can span multiple chunks:

```javascript
class SafeTextDecoder {
    constructor() {
        this.decoder = new TextDecoder("utf-8");
    }
    
    decode(chunk, options = {}) {
        // stream: true tells decoder to hold incomplete sequences
        return this.decoder.decode(chunk, { stream: true, ...options });
    }
    
    flush() {
        // Decode any remaining bytes
        return this.decoder.decode();
    }
}

// Usage
const decoder = new SafeTextDecoder();

for await (const chunk of stream) {
    const text = decoder.decode(chunk);
    console.log(text);
}

// Don't forget to flush!
console.log(decoder.flush());
```

### Example: Split Emoji

```javascript
// Emoji "👋" is 4 bytes: F0 9F 91 8B
// If split across chunks:

const decoder = new TextDecoder();

// Chunk 1: first 2 bytes
const chunk1 = new Uint8Array([0xF0, 0x9F]);
console.log(decoder.decode(chunk1, { stream: true })); // "" (waiting)

// Chunk 2: last 2 bytes
const chunk2 = new Uint8Array([0x91, 0x8B]);
console.log(decoder.decode(chunk2, { stream: true })); // "👋" (complete)
```

---

## Browser Considerations

### Memory Management

```javascript
async function streamWithMemoryLimit(stream, maxSize = 10 * 1024 * 1024) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let totalSize = 0;
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            totalSize += value.length;
            
            if (totalSize > maxSize) {
                throw new Error(`Response exceeded ${maxSize} bytes`);
            }
            
            const text = decoder.decode(value, { stream: true });
            yield text;
        }
    } finally {
        reader.releaseLock();
    }
}
```

### Cancellation

```javascript
async function streamWithCancel(stream, signal) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    
    // Handle abort
    signal.addEventListener("abort", () => {
        reader.cancel();
    });
    
    try {
        while (true) {
            if (signal.aborted) break;
            
            const { done, value } = await reader.read();
            if (done) break;
            
            yield decoder.decode(value, { stream: true });
        }
    } finally {
        reader.releaseLock();
    }
}

// Usage
const controller = new AbortController();

// Cancel after 5 seconds
setTimeout(() => controller.abort(), 5000);

for await (const chunk of streamWithCancel(response.body, controller.signal)) {
    console.log(chunk);
}
```

---

## Comparison: Browser vs Node.js

| Feature | Browser | Node.js |
|---------|---------|---------|
| `response.body` | ReadableStream | ReadableStream (Node 18+) |
| `TextDecoderStream` | ✅ Native | ✅ Native (Node 18+) |
| `for await...of` | On reader | Direct on stream |
| `Readable.fromWeb()` | ❌ N/A | ✅ Convert web streams |
| `pipeThrough()` | ✅ Native | ✅ Native (Node 18+) |

---

## Hands-on Exercise

### Your Task

Build a `ChunkedStreamReader` class that provides a clean API for reading streaming responses.

### Requirements

1. Accept a fetch Response object
2. Yield decoded text chunks
3. Handle UTF-8 properly
4. Support cancellation via AbortSignal

### Expected Result

```javascript
const reader = new ChunkedStreamReader(response);

for await (const text of reader.read(signal)) {
    console.log(text);
}
```

<details>
<summary>💡 Hints</summary>

- Use TextDecoder with `stream: true`
- Check signal.aborted in the loop
- Remember to release the lock
</details>

<details>
<summary>✅ Solution</summary>

```javascript
class ChunkedStreamReader {
    constructor(response) {
        if (!response.body) {
            throw new Error("Response has no body");
        }
        this.stream = response.body;
    }
    
    async *read(signal) {
        const reader = this.stream.getReader();
        const decoder = new TextDecoder("utf-8");
        
        // Setup abort handling
        const abortHandler = () => {
            reader.cancel();
        };
        
        if (signal) {
            signal.addEventListener("abort", abortHandler);
        }
        
        try {
            while (true) {
                // Check for abort before reading
                if (signal?.aborted) {
                    throw new DOMException("Aborted", "AbortError");
                }
                
                const { done, value } = await reader.read();
                
                if (done) {
                    // Flush any remaining bytes
                    const remaining = decoder.decode();
                    if (remaining) yield remaining;
                    break;
                }
                
                // Decode with stream: true for multi-byte safety
                const text = decoder.decode(value, { stream: true });
                
                if (text) {
                    yield text;
                }
            }
        } catch (error) {
            if (error.name === "AbortError") {
                console.log("Stream reading aborted");
            }
            throw error;
        } finally {
            // Always cleanup
            if (signal) {
                signal.removeEventListener("abort", abortHandler);
            }
            reader.releaseLock();
        }
    }
    
    // Convenience method for SSE parsing
    async *readSSE(signal) {
        let buffer = "";
        
        for await (const chunk of this.read(signal)) {
            buffer += chunk;
            
            // Split on double newline
            while (buffer.includes("\n\n")) {
                const eventEnd = buffer.indexOf("\n\n");
                const eventStr = buffer.slice(0, eventEnd);
                buffer = buffer.slice(eventEnd + 2);
                
                const event = this.parseEvent(eventStr);
                if (event) yield event;
            }
        }
    }
    
    parseEvent(str) {
        const lines = str.split("\n");
        let data = "";
        let type = "message";
        
        for (const line of lines) {
            if (line.startsWith("data: ")) {
                data = line.slice(6);
            } else if (line.startsWith("event: ")) {
                type = line.slice(7);
            }
        }
        
        if (!data) return null;
        if (data === "[DONE]") return { type: "done", data: null };
        
        try {
            return { type, data: JSON.parse(data) };
        } catch {
            return { type, data };
        }
    }
}

// Test
async function test() {
    const controller = new AbortController();
    
    // Auto-cancel after 10 seconds
    setTimeout(() => controller.abort(), 10000);
    
    const response = await fetch("/api/stream");
    const reader = new ChunkedStreamReader(response);
    
    try {
        for await (const event of reader.readSSE(controller.signal)) {
            if (event.type === "done") break;
            console.log(event);
        }
    } catch (error) {
        if (error.name === "AbortError") {
            console.log("Cancelled by user");
        } else {
            throw error;
        }
    }
}
```

</details>

---

## Summary

✅ ReadableStream provides low-level streaming access  
✅ Use `getReader()` to get a reader, then `read()` in a loop  
✅ TextDecoder with `stream: true` handles split UTF-8 characters  
✅ Always call `reader.releaseLock()` in a finally block  
✅ Node.js 18+ supports web streams natively  
✅ Use AbortController for cancellation

**Next:** [Parsing Streamed Chunks](./04-parsing-chunks.md)

---

## Further Reading

- [MDN ReadableStream](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream) — Complete API reference
- [MDN Streams Guide](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Using_readable_streams) — Using readable streams
- [Node.js Streams](https://nodejs.org/api/stream.html) — Node.js stream documentation

<!-- 
Sources Consulted:
- MDN ReadableStream: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
- MDN Streams API: https://developer.mozilla.org/en-US/docs/Web/API/Streams_API
- Node.js Streams: https://nodejs.org/api/stream.html
-->
