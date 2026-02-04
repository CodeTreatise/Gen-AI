---
title: "Displaying Text as It Arrives"
---

# Displaying Text as It Arrives

## Introduction

When streaming AI responses, the challenge isn't receiving the data—it's displaying it efficiently. Naive approaches can cause janky animations, frozen UIs, and poor performance. Smart DOM updates make the difference between a smooth experience and a frustrating one.

In this lesson, we'll master the techniques for rendering streamed text without sacrificing performance.

### What We'll Cover

- Appending text to the DOM efficiently
- Batch updates for performance
- Avoiding layout thrashing
- Using requestAnimationFrame
- React and vanilla JavaScript implementations

### Prerequisites

- [Streaming Text Display Overview](./00-streaming-text-display.md)
- DOM manipulation basics
- Understanding of browser rendering pipeline

---

## The Naive Approach (Don't Do This)

```javascript
// ❌ BAD: Updates DOM for every chunk
async function streamResponse(url, container) {
  const response = await fetch(url);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    container.textContent += text;  // DOM update on EVERY chunk!
  }
}
```

### Why This Is Problematic

| Issue | Consequence |
|-------|-------------|
| **Excessive DOM updates** | Each chunk triggers layout recalculation |
| **Layout thrashing** | Reading and writing DOM alternately |
| **No batching** | Small chunks cause many tiny updates |
| **Blocking main thread** | UI freezes during rapid updates |

---

## Efficient DOM Updates

### Strategy 1: Text Node Manipulation

Use a single text node and update its content:

```javascript
function createStreamingContainer(container) {
  const textNode = document.createTextNode('');
  container.appendChild(textNode);
  
  let buffer = '';
  
  return {
    append(chunk) {
      buffer += chunk;
      textNode.textContent = buffer;
    },
    clear() {
      buffer = '';
      textNode.textContent = '';
    }
  };
}

// Usage
const streamer = createStreamingContainer(document.querySelector('.message-body'));
streamer.append('Hello ');
streamer.append('world!');
```

### Strategy 2: innerHTML with DocumentFragment

For content with HTML (like markdown), use fragments:

```javascript
function appendHTML(container, htmlChunk) {
  const template = document.createElement('template');
  template.innerHTML = htmlChunk;
  
  const fragment = document.createDocumentFragment();
  fragment.appendChild(template.content.cloneNode(true));
  
  container.appendChild(fragment);
}
```

### Strategy 3: Range-Based Insertion

For precise insertion points:

```javascript
function insertAtCursor(container, text) {
  const range = document.createRange();
  range.selectNodeContents(container);
  range.collapse(false); // Collapse to end
  
  const textNode = document.createTextNode(text);
  range.insertNode(textNode);
}
```

---

## Batch Updates for Performance

### Accumulate Then Render

Don't update the DOM for every chunk—batch them:

```javascript
class BatchedRenderer {
  constructor(container, flushInterval = 50) {
    this.container = container;
    this.buffer = '';
    this.flushInterval = flushInterval;
    this.flushTimeout = null;
    this.textNode = document.createTextNode('');
    this.fullText = '';
    
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.scheduleFlush();
  }
  
  scheduleFlush() {
    if (this.flushTimeout) return;
    
    this.flushTimeout = setTimeout(() => {
      this.flush();
    }, this.flushInterval);
  }
  
  flush() {
    if (this.buffer) {
      this.fullText += this.buffer;
      this.textNode.textContent = this.fullText;
      this.buffer = '';
    }
    this.flushTimeout = null;
  }
  
  complete() {
    clearTimeout(this.flushTimeout);
    this.flush();
  }
}

// Usage
const renderer = new BatchedRenderer(container, 50);

// In stream handler
for await (const chunk of stream) {
  renderer.append(chunk);
}
renderer.complete();
```

### Chunk Aggregation

Combine multiple small chunks before rendering:

```javascript
function createChunkAggregator(minChunkSize = 10) {
  let buffer = '';
  
  return {
    add(chunk) {
      buffer += chunk;
      
      if (buffer.length >= minChunkSize) {
        const toFlush = buffer;
        buffer = '';
        return toFlush;
      }
      return null;
    },
    
    flush() {
      const remaining = buffer;
      buffer = '';
      return remaining;
    }
  };
}

// Usage
const aggregator = createChunkAggregator(20);

for await (const chunk of stream) {
  const aggregated = aggregator.add(chunk);
  if (aggregated) {
    renderer.append(aggregated);
  }
}
// Don't forget remaining content
renderer.append(aggregator.flush());
```

---

## Avoiding Layout Thrashing

### What Is Layout Thrashing?

Layout thrashing occurs when you read and write DOM properties alternately:

```javascript
// ❌ BAD: Causes layout thrashing
for (const chunk of chunks) {
  container.textContent += chunk;     // Write
  const height = container.offsetHeight; // Read - forces layout!
  scrollContainer.scrollTop = height;    // Write
}
```

### Solution: Separate Read and Write Phases

```javascript
// ✅ GOOD: Batched reads and writes
function updateWithoutThrashing(container, chunks) {
  // Write phase
  let content = container.textContent;
  for (const chunk of chunks) {
    content += chunk;
  }
  container.textContent = content;
  
  // Read phase (single forced layout)
  const height = container.offsetHeight;
  
  // Write phase
  scrollContainer.scrollTop = height;
}
```

### Use CSS for Measurements

```css
/* Let CSS handle scroll behavior */
.message-list {
  overflow-y: auto;
  scroll-behavior: smooth;
}

.message-list[data-auto-scroll="true"] {
  /* Scroll to bottom using CSS anchor */
  overflow-anchor: auto;
}
```

---

## Using requestAnimationFrame

### Why requestAnimationFrame?

| Benefit | Description |
|---------|-------------|
| **Synced with display** | Updates happen before repaint |
| **Batched automatically** | Multiple calls merged into one frame |
| **Throttled naturally** | ~60fps max (16.67ms per frame) |
| **Paused when hidden** | Saves resources in background tabs |

### Basic Implementation

```javascript
class RAFRenderer {
  constructor(container) {
    this.container = container;
    this.buffer = '';
    this.fullText = '';
    this.textNode = document.createTextNode('');
    this.frameRequested = false;
    
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.scheduleRender();
  }
  
  scheduleRender() {
    if (this.frameRequested) return;
    
    this.frameRequested = true;
    requestAnimationFrame(() => {
      this.render();
      this.frameRequested = false;
    });
  }
  
  render() {
    if (this.buffer) {
      this.fullText += this.buffer;
      this.textNode.textContent = this.fullText;
      this.buffer = '';
    }
  }
  
  complete() {
    // Force final render
    this.render();
  }
}
```

### Combining with Batch Threshold

```javascript
class OptimizedRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.minBatchSize = options.minBatchSize || 10;
    this.maxWaitTime = options.maxWaitTime || 100;
    
    this.buffer = '';
    this.fullText = '';
    this.textNode = document.createTextNode('');
    this.lastRenderTime = 0;
    this.rafId = null;
    
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    
    const now = performance.now();
    const timeSinceLastRender = now - this.lastRenderTime;
    
    // Render if buffer is large enough OR max wait exceeded
    if (this.buffer.length >= this.minBatchSize || 
        timeSinceLastRender >= this.maxWaitTime) {
      this.scheduleRender();
    }
  }
  
  scheduleRender() {
    if (this.rafId) return;
    
    this.rafId = requestAnimationFrame(() => {
      this.render();
      this.rafId = null;
    });
  }
  
  render() {
    if (this.buffer) {
      this.fullText += this.buffer;
      this.textNode.textContent = this.fullText;
      this.buffer = '';
      this.lastRenderTime = performance.now();
    }
  }
  
  complete() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    this.render();
  }
}
```

---

## React Implementation

### Basic Streaming State

```jsx
function StreamingMessage({ stream }) {
  const [content, setContent] = useState('');
  const bufferRef = useRef('');
  const rafRef = useRef(null);
  
  useEffect(() => {
    if (!stream) return;
    
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    
    async function read() {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        bufferRef.current += chunk;
        
        // Schedule render
        if (!rafRef.current) {
          rafRef.current = requestAnimationFrame(() => {
            setContent(prev => prev + bufferRef.current);
            bufferRef.current = '';
            rafRef.current = null;
          });
        }
      }
    }
    
    read();
    
    return () => {
      reader.cancel();
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [stream]);
  
  return <div className="message-body">{content}</div>;
}
```

### With AI SDK useChat

```jsx
import { useChat } from 'ai/react';

function Chat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    // Built-in throttling
    experimental_throttle: 50,
  });
  
  return (
    <div className="chat">
      <div className="message-list">
        {messages.map(message => (
          <Message 
            key={message.id} 
            message={message}
            isStreaming={isLoading && message === messages.at(-1)}
          />
        ))}
      </div>
      
      <form onSubmit={handleSubmit}>
        <input 
          value={input} 
          onChange={handleInputChange}
          disabled={isLoading}
        />
      </form>
    </div>
  );
}
```

### Optimized React Component

```jsx
// Separate streaming content to minimize re-renders
const StreamingContent = memo(function StreamingContent({ content }) {
  return <div className="message-body">{content}</div>;
});

function Message({ message, isStreaming }) {
  return (
    <div className={`message ${message.role} ${isStreaming ? 'streaming' : ''}`}>
      <div className="message-header">
        {message.role === 'assistant' ? 'AI' : 'You'}
      </div>
      <StreamingContent content={message.content} />
      {isStreaming && <StreamingCursor />}
    </div>
  );
}
```

---

## Complete Streaming Handler

```javascript
class StreamingTextRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      batchInterval: 50,
      minBatchSize: 5,
      onChunk: null,
      onComplete: null,
      ...options
    };
    
    this.buffer = '';
    this.fullText = '';
    this.isStreaming = false;
    this.rafId = null;
    
    this.init();
  }
  
  init() {
    this.textNode = document.createTextNode('');
    this.container.innerHTML = '';
    this.container.appendChild(this.textNode);
  }
  
  async stream(response) {
    this.isStreaming = true;
    this.container.classList.add('streaming');
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        this.append(chunk);
        
        if (this.options.onChunk) {
          this.options.onChunk(chunk, this.fullText + this.buffer);
        }
      }
    } finally {
      this.complete();
    }
    
    return this.fullText;
  }
  
  append(chunk) {
    this.buffer += chunk;
    
    if (this.buffer.length >= this.options.minBatchSize) {
      this.scheduleRender();
    }
  }
  
  scheduleRender() {
    if (this.rafId) return;
    
    this.rafId = requestAnimationFrame(() => {
      this.flush();
      this.rafId = null;
    });
  }
  
  flush() {
    if (this.buffer) {
      this.fullText += this.buffer;
      this.textNode.textContent = this.fullText;
      this.buffer = '';
    }
  }
  
  complete() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    
    this.flush();
    this.isStreaming = false;
    this.container.classList.remove('streaming');
    
    if (this.options.onComplete) {
      this.options.onComplete(this.fullText);
    }
  }
  
  cancel() {
    this.complete();
  }
  
  reset() {
    this.buffer = '';
    this.fullText = '';
    this.textNode.textContent = '';
  }
}

// Usage
const renderer = new StreamingTextRenderer(document.querySelector('.message-body'), {
  batchInterval: 50,
  onChunk: (chunk, fullText) => {
    console.log('Received:', chunk.length, 'chars');
  },
  onComplete: (fullText) => {
    console.log('Complete:', fullText.length, 'chars');
  }
});

const response = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({ message }) });
await renderer.stream(response);
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Batch DOM updates | Update DOM for every chunk |
| Use requestAnimationFrame | Use setInterval for rendering |
| Separate read/write phases | Mix DOM reads and writes |
| Use text nodes for plain text | Use innerHTML for plain text |
| Cancel animations on cleanup | Leave orphaned animation frames |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Updating DOM in tight loop | Batch with RAF or setTimeout |
| Forgetting to flush buffer | Always call complete() at stream end |
| Not handling stream errors | Wrap in try/catch/finally |
| Memory leaks from listeners | Clean up in unmount/cleanup |
| Blocking UI with large chunks | Process in animation frames |

---

## Hands-on Exercise

### Your Task

Build a streaming text renderer that:
1. Batches updates using requestAnimationFrame
2. Has a minimum batch size of 10 characters
3. Forces render after 100ms even with small buffer
4. Handles stream cancellation gracefully

### Requirements

1. Create a class with `append()`, `complete()`, and `cancel()` methods
2. Use a text node for efficient updates
3. Track streaming state with a class on the container
4. Provide callbacks for chunk and completion events

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `performance.now()` to track time since last render
- Store RAF ID to cancel pending frames
- Flush buffer in both `complete()` and `cancel()`
- Add/remove `streaming` class for CSS styling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `StreamingTextRenderer` class in the "Complete Streaming Handler" section above.

</details>

---

## Summary

✅ **Batch DOM updates** using requestAnimationFrame  
✅ **Avoid layout thrashing** by separating read/write phases  
✅ **Use text nodes** for efficient plain text updates  
✅ **Set minimum batch sizes** to prevent excessive renders  
✅ **Clean up properly** with complete() and cancel() methods

---

## Further Reading

- [requestAnimationFrame - MDN](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame)
- [Avoid Layout Thrashing - web.dev](https://web.dev/avoid-large-complex-layouts-and-layout-thrashing/)
- [ReadableStream - MDN](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream)

---

**Previous:** [Streaming Text Display Overview](./00-streaming-text-display.md)  
**Next:** [Character vs Chunk Display](./02-character-vs-chunk-display.md)

<!-- 
Sources Consulted:
- MDN requestAnimationFrame: https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame
- web.dev Layout Thrashing: https://web.dev/avoid-large-complex-layouts-and-layout-thrashing/
- MDN ReadableStream: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
-->
