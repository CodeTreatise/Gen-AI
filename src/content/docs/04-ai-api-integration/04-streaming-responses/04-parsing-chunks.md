---
title: "Parsing Streamed Chunks"
---

# Parsing Streamed Chunks

## Introduction

Raw streaming data arrives as continuous bytes that don't respect message boundaries. This lesson covers the techniques for correctly parsing SSE events, handling line breaks, buffering incomplete data, and dealing with edge cases.

### What We'll Cover

- Buffering incomplete lines
- Parsing SSE data lines
- Handling `data: [DONE]` termination
- JSON parsing each event
- Split UTF-8 character handling

### Prerequisites

- SSE protocol basics
- ReadableStream fundamentals

---

## The Parsing Challenge

Chunks arrive at arbitrary boundaries:

```mermaid
flowchart LR
    subgraph "Chunk 1"
        A["data: {\"del"]
    end
    
    subgraph "Chunk 2"
        B["ta\": \"Hel"]
    end
    
    subgraph "Chunk 3"
        C["lo\"}\n\ndata: {\"delta\": \" wor"]
    end
    
    subgraph "Chunk 4"
        D["ld\"}\n\n"]
    end
    
    A --> B --> C --> D
```

We need to reassemble complete events from these fragments.

---

## Line Buffer Implementation

### Basic Buffer Pattern

```javascript
class LineBuffer {
    constructor() {
        this.buffer = "";
    }
    
    add(chunk) {
        this.buffer += chunk;
        return this.extractLines();
    }
    
    extractLines() {
        const lines = [];
        let newlineIndex;
        
        while ((newlineIndex = this.buffer.indexOf("\n")) !== -1) {
            const line = this.buffer.slice(0, newlineIndex);
            this.buffer = this.buffer.slice(newlineIndex + 1);
            lines.push(line);
        }
        
        return lines;
    }
    
    flush() {
        const remaining = this.buffer;
        this.buffer = "";
        return remaining;
    }
}

// Usage
const buffer = new LineBuffer();

// Simulated chunks
const chunks = [
    "data: Hello\n",
    "data: Wor",
    "ld\n\ndata: Done\n\n"
];

for (const chunk of chunks) {
    const lines = buffer.add(chunk);
    console.log("Lines:", lines);
}
console.log("Remaining:", buffer.flush());
```

**Output:**
```
Lines: ["data: Hello"]
Lines: []
Lines: ["data: World", "", "data: Done", ""]
Remaining: ""
```

---

## SSE Event Parser

### Complete Event Extraction

```javascript
class SSEParser {
    constructor() {
        this.buffer = "";
    }
    
    parse(chunk) {
        this.buffer += chunk;
        const events = [];
        
        // SSE events are separated by double newlines
        while (this.buffer.includes("\n\n")) {
            const eventEnd = this.buffer.indexOf("\n\n");
            const eventStr = this.buffer.slice(0, eventEnd);
            this.buffer = this.buffer.slice(eventEnd + 2);
            
            const event = this.parseEvent(eventStr);
            if (event) events.push(event);
        }
        
        return events;
    }
    
    parseEvent(str) {
        if (!str.trim()) return null;
        
        const result = {
            event: "message",
            data: "",
            id: null
        };
        
        const dataLines = [];
        
        for (const line of str.split("\n")) {
            if (line.startsWith("data: ")) {
                dataLines.push(line.slice(6));
            } else if (line.startsWith("event: ")) {
                result.event = line.slice(7);
            } else if (line.startsWith("id: ")) {
                result.id = line.slice(4);
            }
            // Skip comments (lines starting with :)
        }
        
        result.data = dataLines.join("\n");
        return result.data ? result : null;
    }
    
    flush() {
        if (this.buffer.trim()) {
            const event = this.parseEvent(this.buffer);
            this.buffer = "";
            return event;
        }
        return null;
    }
}
```

### Python Implementation

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SSEEvent:
    event: str
    data: str
    id: Optional[str] = None

class SSEParser:
    def __init__(self):
        self.buffer = ""
    
    def parse(self, chunk: str) -> List[SSEEvent]:
        """Parse chunk and return complete events."""
        self.buffer += chunk
        events = []
        
        while "\n\n" in self.buffer:
            event_str, self.buffer = self.buffer.split("\n\n", 1)
            event = self._parse_event(event_str)
            if event:
                events.append(event)
        
        return events
    
    def _parse_event(self, event_str: str) -> Optional[SSEEvent]:
        if not event_str.strip():
            return None
        
        event_type = "message"
        event_id = None
        data_lines = []
        
        for line in event_str.split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("id: "):
                event_id = line[4:]
        
        data = "\n".join(data_lines)
        return SSEEvent(event_type, data, event_id) if data else None
    
    def flush(self) -> Optional[SSEEvent]:
        if self.buffer.strip():
            event = self._parse_event(self.buffer)
            self.buffer = ""
            return event
        return None
```

---

## Handling `data: [DONE]`

### Detection and Termination

```javascript
class OpenAISSEParser {
    constructor() {
        this.buffer = "";
        this.done = false;
    }
    
    *parse(chunk) {
        if (this.done) return;
        
        this.buffer += chunk;
        
        while (this.buffer.includes("\n\n")) {
            const eventEnd = this.buffer.indexOf("\n\n");
            const eventStr = this.buffer.slice(0, eventEnd);
            this.buffer = this.buffer.slice(eventEnd + 2);
            
            // Check for termination
            if (eventStr.includes("data: [DONE]")) {
                this.done = true;
                yield { type: "done", data: null };
                return;
            }
            
            const event = this.parseEvent(eventStr);
            if (event) yield event;
        }
    }
    
    parseEvent(str) {
        const dataMatch = str.match(/^data: (.+)$/m);
        if (!dataMatch) return null;
        
        const dataStr = dataMatch[1];
        
        // Skip [DONE] marker
        if (dataStr === "[DONE]") {
            return { type: "done", data: null };
        }
        
        try {
            const data = JSON.parse(dataStr);
            const eventMatch = str.match(/^event: (.+)$/m);
            return {
                type: eventMatch ? eventMatch[1] : "message",
                data
            };
        } catch {
            return { type: "message", data: dataStr };
        }
    }
}

// Usage
const parser = new OpenAISSEParser();

const stream = `data: {"delta":"Hello"}\n\ndata: {"delta":" world"}\n\ndata: [DONE]\n\n`;

for (const event of parser.parse(stream)) {
    console.log(event);
}
```

**Output:**
```
{ type: "message", data: { delta: "Hello" } }
{ type: "message", data: { delta: " world" } }
{ type: "done", data: null }
```

---

## JSON Parsing Robustness

### Handling Parse Errors

```javascript
function parseJSONSafely(str) {
    try {
        return { success: true, data: JSON.parse(str) };
    } catch (error) {
        return { success: false, error: error.message, raw: str };
    }
}

class RobustSSEParser {
    constructor(options = {}) {
        this.buffer = "";
        this.onParseError = options.onParseError || console.error;
    }
    
    *parse(chunk) {
        this.buffer += chunk;
        
        while (this.buffer.includes("\n\n")) {
            const eventEnd = this.buffer.indexOf("\n\n");
            const eventStr = this.buffer.slice(0, eventEnd);
            this.buffer = this.buffer.slice(eventEnd + 2);
            
            for (const line of eventStr.split("\n")) {
                if (!line.startsWith("data: ")) continue;
                
                const content = line.slice(6);
                
                if (content === "[DONE]") {
                    yield { type: "done" };
                    continue;
                }
                
                const result = parseJSONSafely(content);
                
                if (result.success) {
                    yield { type: "data", data: result.data };
                } else {
                    this.onParseError({
                        message: result.error,
                        raw: result.raw
                    });
                    // Yield raw string as fallback
                    yield { type: "raw", data: result.raw };
                }
            }
        }
    }
}
```

### Handling Malformed Events

```python
import json
from typing import Union

def safe_json_parse(data: str) -> Union[dict, str]:
    """Parse JSON with fallback to raw string."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data

class RobustParser:
    def __init__(self):
        self.buffer = ""
        self.errors = []
    
    def parse(self, chunk: str):
        self.buffer += chunk
        events = []
        
        while "\n\n" in self.buffer:
            event_str, self.buffer = self.buffer.split("\n\n", 1)
            
            for line in event_str.split("\n"):
                if not line.startswith("data: "):
                    continue
                
                content = line[6:]
                
                if content == "[DONE]":
                    events.append({"type": "done"})
                    continue
                
                parsed = safe_json_parse(content)
                
                if isinstance(parsed, dict):
                    events.append({"type": "data", "data": parsed})
                else:
                    self.errors.append(f"Failed to parse: {content[:50]}")
                    events.append({"type": "raw", "data": parsed})
        
        return events
```

---

## Handling Split UTF-8

### The Problem

UTF-8 characters can be 1-4 bytes:

| Character | Bytes | Example |
|-----------|-------|---------|
| ASCII | 1 | `A` = `0x41` |
| Latin | 2 | `é` = `0xC3 0xA9` |
| CJK | 3 | `中` = `0xE4 0xB8 0xAD` |
| Emoji | 4 | `👋` = `0xF0 0x9F 0x91 0x8B` |

### Solution: TextDecoder with stream mode

```javascript
class UTF8SafeParser {
    constructor() {
        this.decoder = new TextDecoder("utf-8");
        this.buffer = "";
    }
    
    addChunk(bytes) {
        // stream: true holds incomplete sequences
        const text = this.decoder.decode(bytes, { stream: true });
        this.buffer += text;
        return this.extractEvents();
    }
    
    finish() {
        // Flush any remaining bytes
        const remaining = this.decoder.decode();
        this.buffer += remaining;
        return this.extractEvents();
    }
    
    extractEvents() {
        const events = [];
        
        while (this.buffer.includes("\n\n")) {
            const idx = this.buffer.indexOf("\n\n");
            events.push(this.buffer.slice(0, idx));
            this.buffer = this.buffer.slice(idx + 2);
        }
        
        return events;
    }
}

// Usage
const parser = new UTF8SafeParser();

// Simulate split emoji (👋 split across chunks)
const chunk1 = new Uint8Array([0x64, 0x61, 0x74, 0x61, 0x3A, 0x20, 0xF0, 0x9F]);  // "data: " + first 2 bytes of 👋
const chunk2 = new Uint8Array([0x91, 0x8B, 0x0A, 0x0A]);  // last 2 bytes of 👋 + "\n\n"

console.log(parser.addChunk(chunk1));  // [] - waiting for complete char
console.log(parser.addChunk(chunk2));  // ["data: 👋"]
```

---

## Complete Production Parser

```javascript
class ProductionSSEParser {
    constructor(options = {}) {
        this.decoder = new TextDecoder("utf-8");
        this.buffer = "";
        this.done = false;
        this.eventCount = 0;
        this.byteCount = 0;
        
        // Options
        this.maxBufferSize = options.maxBufferSize || 1024 * 1024; // 1MB
        this.onError = options.onError || console.error;
    }
    
    *parseBytes(bytes) {
        if (this.done) return;
        
        this.byteCount += bytes.length;
        
        // Decode bytes to text
        const text = this.decoder.decode(bytes, { stream: true });
        this.buffer += text;
        
        // Check buffer size
        if (this.buffer.length > this.maxBufferSize) {
            this.onError(new Error("Buffer size exceeded"));
            this.buffer = "";
            return;
        }
        
        // Extract and yield events
        yield* this.extractEvents();
    }
    
    *parseText(text) {
        if (this.done) return;
        
        this.buffer += text;
        yield* this.extractEvents();
    }
    
    *extractEvents() {
        while (this.buffer.includes("\n\n")) {
            const idx = this.buffer.indexOf("\n\n");
            const eventStr = this.buffer.slice(0, idx);
            this.buffer = this.buffer.slice(idx + 2);
            
            const event = this.parseEvent(eventStr);
            if (event) {
                this.eventCount++;
                
                if (event.type === "done") {
                    this.done = true;
                }
                
                yield event;
                
                if (this.done) return;
            }
        }
    }
    
    parseEvent(str) {
        if (!str.trim()) return null;
        
        let eventType = "message";
        let eventId = null;
        const dataLines = [];
        
        for (const line of str.split("\n")) {
            // Skip comments
            if (line.startsWith(":")) continue;
            
            if (line.startsWith("data: ")) {
                const content = line.slice(6);
                
                // Check for termination
                if (content === "[DONE]") {
                    return { type: "done" };
                }
                
                dataLines.push(content);
            } else if (line.startsWith("event: ")) {
                eventType = line.slice(7);
            } else if (line.startsWith("id: ")) {
                eventId = line.slice(4);
            }
        }
        
        if (dataLines.length === 0) return null;
        
        const rawData = dataLines.join("\n");
        let parsedData;
        
        try {
            parsedData = JSON.parse(rawData);
        } catch {
            parsedData = rawData;
        }
        
        return {
            type: eventType,
            data: parsedData,
            id: eventId,
            raw: rawData
        };
    }
    
    finish() {
        // Flush decoder
        const remaining = this.decoder.decode();
        if (remaining) {
            this.buffer += remaining;
        }
        
        // Process any remaining buffer
        const events = [...this.extractEvents()];
        
        // Handle incomplete final event
        if (this.buffer.trim()) {
            events.push({
                type: "incomplete",
                raw: this.buffer
            });
        }
        
        return events;
    }
    
    getStats() {
        return {
            eventCount: this.eventCount,
            byteCount: this.byteCount,
            bufferSize: this.buffer.length,
            done: this.done
        };
    }
}
```

---

## Integration Example

```javascript
async function streamWithParser(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
    
    const reader = response.body.getReader();
    const parser = new ProductionSSEParser({
        onError: (err) => console.error("Parse error:", err)
    });
    
    const results = [];
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                // Finalize and get any remaining events
                results.push(...parser.finish());
                break;
            }
            
            for (const event of parser.parseBytes(value)) {
                if (event.type === "done") {
                    break;
                }
                results.push(event);
            }
        }
    } finally {
        reader.releaseLock();
        console.log("Stats:", parser.getStats());
    }
    
    return results;
}
```

---

## Hands-on Exercise

### Your Task

Build a streaming text accumulator that extracts and concatenates delta content.

### Requirements

1. Parse SSE events from chunks
2. Extract `delta` or `content` fields
3. Accumulate into final text
4. Handle `[DONE]` termination

### Expected Result

```javascript
const accumulator = new DeltaAccumulator();

for (const chunk of chunks) {
    accumulator.add(chunk);
}

console.log(accumulator.getText());
// "Hello world! How are you today?"
```

<details>
<summary>💡 Hints</summary>

- Look for `delta` in parsed JSON
- Also check nested paths like `choices[0].delta.content`
- Join all deltas at the end
</details>

<details>
<summary>✅ Solution</summary>

```javascript
class DeltaAccumulator {
    constructor() {
        this.buffer = "";
        this.deltas = [];
        this.done = false;
    }
    
    add(chunk) {
        this.buffer += chunk;
        
        while (this.buffer.includes("\n\n")) {
            const idx = this.buffer.indexOf("\n\n");
            const eventStr = this.buffer.slice(0, idx);
            this.buffer = this.buffer.slice(idx + 2);
            
            this.processEvent(eventStr);
        }
    }
    
    processEvent(eventStr) {
        for (const line of eventStr.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            
            const content = line.slice(6);
            
            if (content === "[DONE]") {
                this.done = true;
                return;
            }
            
            try {
                const data = JSON.parse(content);
                const delta = this.extractDelta(data);
                if (delta) {
                    this.deltas.push(delta);
                }
            } catch {
                // Skip non-JSON data
            }
        }
    }
    
    extractDelta(data) {
        // OpenAI Responses API format
        if (data.delta) {
            return data.delta;
        }
        
        // Chat Completions format
        if (data.choices?.[0]?.delta?.content) {
            return data.choices[0].delta.content;
        }
        
        // Anthropic format
        if (data.delta?.text) {
            return data.delta.text;
        }
        
        // Direct content
        if (data.content) {
            return data.content;
        }
        
        return null;
    }
    
    getText() {
        return this.deltas.join("");
    }
    
    isDone() {
        return this.done;
    }
    
    getDeltas() {
        return [...this.deltas];
    }
}

// Test
const accumulator = new DeltaAccumulator();

const testChunks = [
    'data: {"delta": "Hello"}\n\n',
    'data: {"delta": " wor"}\n\n',
    'data: {"delta": "ld!"}\n\n',
    'data: {"delta": " How are "}\n\ndata: {"delta": "you today?"}\n\n',
    'data: [DONE]\n\n'
];

for (const chunk of testChunks) {
    accumulator.add(chunk);
}

console.log(accumulator.getText());
// Output: "Hello world! How are you today?"

console.log("Done:", accumulator.isDone());
// Output: Done: true

console.log("Delta count:", accumulator.getDeltas().length);
// Output: Delta count: 5
```

</details>

---

## Summary

✅ Buffer incomplete data until double-newline event boundaries  
✅ Parse `data:`, `event:`, and `id:` fields from each event  
✅ Handle `data: [DONE]` as stream termination marker  
✅ Use TextDecoder with `stream: true` for UTF-8 safety  
✅ Always handle JSON parse errors gracefully  
✅ Track buffer size to prevent memory issues

**Next:** [Delta Content Handling](./05-delta-handling.md)

---

## Further Reading

- [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html) — Official parsing rules
- [TextDecoder API](https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder) — MDN reference
- [UTF-8 Encoding](https://en.wikipedia.org/wiki/UTF-8) — Understanding multi-byte sequences

<!-- 
Sources Consulted:
- WHATWG SSE Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
- MDN TextDecoder: https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder
-->
