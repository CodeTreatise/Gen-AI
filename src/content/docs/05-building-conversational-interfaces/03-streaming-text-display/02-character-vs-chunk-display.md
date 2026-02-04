---
title: "Character vs Chunk Display"
---

# Character vs Chunk Display

## Introduction

Should streamed text appear character-by-character like a typewriter, or in chunks as received from the API? Each approach has trade-offs affecting perceived speed, naturalness, and performance.

In this lesson, we'll compare display strategies and implement optimized solutions for both.

### What We'll Cover

- Character-by-character typing effects
- Chunk-based display optimization
- Natural typing feel techniques
- Chunk size optimization
- Performance trade-offs
- Throttling strategies

### Prerequisites

- [Displaying Text as It Arrives](./01-displaying-text-as-it-arrives.md)
- CSS animations
- JavaScript timing functions

---

## Understanding the Trade-offs

### Comparison Table

| Aspect | Character-by-Character | Chunk Display |
|--------|------------------------|---------------|
| **Feel** | Natural, human-like | Fast, efficient |
| **Speed** | Slower perceived | Faster perceived |
| **CPU Usage** | Higher (many updates) | Lower (fewer updates) |
| **Use Case** | Dramatic effect, typing sim | Production chat apps |
| **Complexity** | More complex timing | Simpler implementation |

### When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| Production AI chat | Chunk display (faster UX) |
| Typewriter effect | Character-by-character |
| Code generation | Chunk (preserve syntax) |
| Storytelling apps | Character (dramatic feel) |
| Real-time transcription | Chunk (match audio pacing) |

---

## Character-by-Character Display

### Basic Typewriter Effect

```javascript
class TypewriterRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.speed = options.speed || 30;  // ms per character
    this.variation = options.variation || 10;  // ±ms randomness
    this.queue = '';
    this.isTyping = false;
    this.textNode = document.createTextNode('');
    this.displayedText = '';
    
    container.appendChild(this.textNode);
  }
  
  append(text) {
    this.queue += text;
    if (!this.isTyping) {
      this.type();
    }
  }
  
  async type() {
    this.isTyping = true;
    
    while (this.queue.length > 0) {
      const char = this.queue[0];
      this.queue = this.queue.slice(1);
      
      this.displayedText += char;
      this.textNode.textContent = this.displayedText;
      
      // Variable delay for natural feel
      const delay = this.speed + (Math.random() * 2 - 1) * this.variation;
      await this.wait(delay);
    }
    
    this.isTyping = false;
  }
  
  wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  complete() {
    // Immediately display remaining queue
    this.displayedText += this.queue;
    this.textNode.textContent = this.displayedText;
    this.queue = '';
    this.isTyping = false;
  }
}

// Usage
const typewriter = new TypewriterRenderer(container, {
  speed: 25,
  variation: 15
});

typewriter.append("Hello, I'm thinking about your question...");
```

### Natural Typing Patterns

Real typing has patterns—pauses at punctuation, faster for common words:

```javascript
class NaturalTypewriter {
  constructor(container, options = {}) {
    this.container = container;
    this.baseSpeed = options.speed || 30;
    this.queue = '';
    this.isTyping = false;
    this.textNode = document.createTextNode('');
    this.displayedText = '';
    
    container.appendChild(this.textNode);
  }
  
  getCharDelay(char, nextChar) {
    // Longer pause after punctuation
    if ('.!?'.includes(char)) {
      return this.baseSpeed * 8;
    }
    
    // Medium pause after comma/semicolon
    if (',;:'.includes(char)) {
      return this.baseSpeed * 3;
    }
    
    // Pause at end of word
    if (char === ' ' && nextChar && /[A-Z]/.test(nextChar)) {
      return this.baseSpeed * 2;
    }
    
    // Faster for common letter combinations
    const lastTwo = this.displayedText.slice(-1) + char;
    if (['th', 'he', 'in', 'er', 'an'].includes(lastTwo.toLowerCase())) {
      return this.baseSpeed * 0.7;
    }
    
    // Random variation
    return this.baseSpeed + (Math.random() - 0.5) * this.baseSpeed * 0.5;
  }
  
  async type() {
    this.isTyping = true;
    
    while (this.queue.length > 0) {
      const char = this.queue[0];
      const nextChar = this.queue[1];
      this.queue = this.queue.slice(1);
      
      this.displayedText += char;
      this.textNode.textContent = this.displayedText;
      
      const delay = this.getCharDelay(char, nextChar);
      await new Promise(r => setTimeout(r, delay));
    }
    
    this.isTyping = false;
  }
  
  append(text) {
    this.queue += text;
    if (!this.isTyping) {
      this.type();
    }
  }
}
```

### Performance-Optimized Typewriter

Batch visual updates while maintaining typing rhythm:

```javascript
class OptimizedTypewriter {
  constructor(container, options = {}) {
    this.container = container;
    this.speed = options.speed || 20;
    this.batchSize = options.batchSize || 3;  // Chars per render
    
    this.queue = '';
    this.displayedText = '';
    this.isTyping = false;
    this.rafId = null;
    
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
  }
  
  append(text) {
    this.queue += text;
    if (!this.isTyping) {
      this.startTyping();
    }
  }
  
  startTyping() {
    this.isTyping = true;
    this.lastTypeTime = performance.now();
    this.scheduleType();
  }
  
  scheduleType() {
    this.rafId = requestAnimationFrame((timestamp) => {
      const elapsed = timestamp - this.lastTypeTime;
      
      if (elapsed >= this.speed) {
        const charsToAdd = Math.min(
          this.batchSize,
          Math.floor(elapsed / this.speed),
          this.queue.length
        );
        
        if (charsToAdd > 0) {
          this.displayedText += this.queue.slice(0, charsToAdd);
          this.queue = this.queue.slice(charsToAdd);
          this.textNode.textContent = this.displayedText;
          this.lastTypeTime = timestamp;
        }
      }
      
      if (this.queue.length > 0) {
        this.scheduleType();
      } else {
        this.isTyping = false;
      }
    });
  }
  
  complete() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    this.displayedText += this.queue;
    this.textNode.textContent = this.displayedText;
    this.queue = '';
    this.isTyping = false;
  }
}
```

---

## Chunk Display Optimization

### Direct Chunk Rendering

For production apps, render chunks as they arrive:

```javascript
class ChunkRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.throttleMs = options.throttle || 50;
    
    this.buffer = '';
    this.displayedText = '';
    this.lastRender = 0;
    this.rafId = null;
    
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    
    const now = performance.now();
    if (now - this.lastRender >= this.throttleMs) {
      this.render();
    } else {
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
      this.displayedText += this.buffer;
      this.textNode.textContent = this.displayedText;
      this.buffer = '';
      this.lastRender = performance.now();
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

### Word-Boundary Chunking

Render at word boundaries for cleaner display:

```javascript
class WordBoundaryRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.minChunkSize = options.minChunkSize || 5;
    
    this.buffer = '';
    this.displayedText = '';
    this.textNode = document.createTextNode('');
    
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    
    // Find last word boundary
    const lastSpace = this.buffer.lastIndexOf(' ');
    const lastNewline = this.buffer.lastIndexOf('\n');
    const boundary = Math.max(lastSpace, lastNewline);
    
    if (boundary > this.minChunkSize) {
      const toRender = this.buffer.slice(0, boundary + 1);
      this.buffer = this.buffer.slice(boundary + 1);
      
      this.displayedText += toRender;
      this.textNode.textContent = this.displayedText;
    }
  }
  
  complete() {
    // Flush remaining buffer
    this.displayedText += this.buffer;
    this.textNode.textContent = this.displayedText;
    this.buffer = '';
  }
}
```

### Sentence-Aware Rendering

For more natural reading, render complete sentences:

```javascript
class SentenceRenderer {
  constructor(container) {
    this.container = container;
    this.buffer = '';
    this.displayedText = '';
    this.textNode = document.createTextNode('');
    
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.buffer += chunk;
    
    // Check for sentence endings
    const sentenceEnd = /[.!?]\s/g;
    let match;
    let lastEnd = 0;
    
    while ((match = sentenceEnd.exec(this.buffer)) !== null) {
      lastEnd = match.index + match[0].length;
    }
    
    if (lastEnd > 0) {
      const toRender = this.buffer.slice(0, lastEnd);
      this.buffer = this.buffer.slice(lastEnd);
      
      this.displayedText += toRender;
      this.textNode.textContent = this.displayedText;
    }
  }
  
  complete() {
    this.displayedText += this.buffer;
    this.textNode.textContent = this.displayedText;
    this.buffer = '';
  }
}
```

---

## Hybrid Approach

Combine chunk reception with smooth visual display:

```javascript
class HybridRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.displaySpeed = options.displaySpeed || 15;  // ms per char for display
    this.catchUpThreshold = options.catchUpThreshold || 100;  // chars behind
    
    this.received = '';
    this.displayed = '';
    this.isRendering = false;
    
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
  }
  
  append(chunk) {
    this.received += chunk;
    
    if (!this.isRendering) {
      this.startRendering();
    }
  }
  
  startRendering() {
    this.isRendering = true;
    this.renderLoop();
  }
  
  renderLoop() {
    if (this.displayed.length >= this.received.length) {
      this.isRendering = false;
      return;
    }
    
    const behind = this.received.length - this.displayed.length;
    
    // If too far behind, catch up faster
    const charsToAdd = behind > this.catchUpThreshold 
      ? Math.ceil(behind / 10)  // Catch up mode
      : 1;                       // Normal mode
    
    this.displayed = this.received.slice(0, this.displayed.length + charsToAdd);
    this.textNode.textContent = this.displayed;
    
    // Calculate delay based on how far behind we are
    const delay = behind > this.catchUpThreshold
      ? 1  // Fast catch-up
      : this.displaySpeed;
    
    setTimeout(() => this.renderLoop(), delay);
  }
  
  complete() {
    // Immediately show everything
    this.displayed = this.received;
    this.textNode.textContent = this.displayed;
    this.isRendering = false;
  }
}
```

---

## React Implementation

### Chunk Display with AI SDK

```jsx
import { useChat } from 'ai/react';

function Chat() {
  const { messages, input, handleInputChange, handleSubmit, status } = useChat({
    api: '/api/chat',
    experimental_throttle: 50,  // Built-in throttling
  });
  
  return (
    <div className="chat">
      <MessageList messages={messages} isStreaming={status === 'streaming'} />
      <InputForm 
        value={input}
        onChange={handleInputChange}
        onSubmit={handleSubmit}
        disabled={status === 'streaming'}
      />
    </div>
  );
}
```

### Custom Typewriter Effect

```jsx
function TypewriterMessage({ content, speed = 30 }) {
  const [displayed, setDisplayed] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (displayed.length >= content.length) {
      setIsComplete(true);
      return;
    }
    
    const timer = setTimeout(() => {
      setDisplayed(content.slice(0, displayed.length + 1));
    }, speed + (Math.random() - 0.5) * speed * 0.5);
    
    return () => clearTimeout(timer);
  }, [displayed, content, speed]);
  
  // Reset when content changes significantly
  useEffect(() => {
    if (content.length < displayed.length) {
      setDisplayed('');
      setIsComplete(false);
    }
  }, [content]);
  
  return (
    <div className="message-body">
      {displayed}
      {!isComplete && <span className="cursor">▋</span>}
    </div>
  );
}
```

### Adaptive Display Speed

```jsx
function AdaptiveStreamingMessage({ content, isStreaming }) {
  const [displayed, setDisplayed] = useState('');
  const prevContentRef = useRef('');
  
  useEffect(() => {
    if (!isStreaming) {
      // Not streaming, show everything
      setDisplayed(content);
      return;
    }
    
    const newContent = content.slice(prevContentRef.current.length);
    prevContentRef.current = content;
    
    if (newContent.length === 0) return;
    
    // Calculate display speed based on how much is buffered
    const buffered = content.length - displayed.length;
    const speed = buffered > 50 ? 5 : buffered > 20 ? 15 : 30;
    
    const timer = setTimeout(() => {
      const charsToAdd = buffered > 50 ? 5 : 1;
      setDisplayed(prev => content.slice(0, prev.length + charsToAdd));
    }, speed);
    
    return () => clearTimeout(timer);
  }, [content, displayed, isStreaming]);
  
  return <div className="message-body">{displayed}</div>;
}
```

---

## Throttling Strategies

### Time-Based Throttling

```javascript
function createThrottledRenderer(container, minInterval = 50) {
  let lastRender = 0;
  let pendingText = '';
  let displayedText = '';
  const textNode = document.createTextNode('');
  container.appendChild(textNode);
  
  function tryRender() {
    const now = performance.now();
    if (now - lastRender >= minInterval && pendingText) {
      displayedText += pendingText;
      textNode.textContent = displayedText;
      pendingText = '';
      lastRender = now;
    }
  }
  
  return {
    append(text) {
      pendingText += text;
      tryRender();
    },
    
    flush() {
      displayedText += pendingText;
      textNode.textContent = displayedText;
      pendingText = '';
    }
  };
}
```

### Frame-Based Throttling

```javascript
function createFrameThrottledRenderer(container) {
  let pendingText = '';
  let displayedText = '';
  let rafId = null;
  const textNode = document.createTextNode('');
  container.appendChild(textNode);
  
  function scheduleRender() {
    if (rafId) return;
    
    rafId = requestAnimationFrame(() => {
      if (pendingText) {
        displayedText += pendingText;
        textNode.textContent = displayedText;
        pendingText = '';
      }
      rafId = null;
    });
  }
  
  return {
    append(text) {
      pendingText += text;
      scheduleRender();
    },
    
    flush() {
      if (rafId) cancelAnimationFrame(rafId);
      displayedText += pendingText;
      textNode.textContent = displayedText;
      pendingText = '';
    }
  };
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use chunk display for production | Use character-by-character in production |
| Throttle renders to 50ms minimum | Render every incoming chunk |
| Provide "skip" for typewriter | Force users to wait |
| Adapt speed to buffer size | Use fixed timing always |
| Complete immediately on stop | Leave animation running |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Typewriter blocks stream reading | Use async queue with separate render loop |
| Display falls behind content | Implement catch-up mode |
| Animation continues after unmount | Clean up timers in useEffect |
| Jarring speed changes | Smooth transition between speeds |
| Memory leak in render loop | Cancel animation on complete |

---

## Hands-on Exercise

### Your Task

Build a hybrid renderer that:
1. Receives chunks and buffers them
2. Displays at a steady character rate
3. Catches up if buffer exceeds threshold
4. Completes immediately when streaming ends

### Requirements

1. Normal display speed: 20ms per character
2. Catch-up threshold: 50 characters behind
3. Catch-up speed: 5 characters at once
4. Smooth transition between modes

<details>
<summary>💡 Hints (click to expand)</summary>

- Track both `received` and `displayed` text
- Use setTimeout for the render loop
- Calculate `behind = received.length - displayed.length`
- In complete(), set displayed = received immediately

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `HybridRenderer` class in the "Hybrid Approach" section above.

</details>

---

## Summary

✅ **Character-by-character** creates natural typing feel but uses more CPU  
✅ **Chunk display** is faster and more efficient for production  
✅ **Hybrid approaches** balance speed with visual appeal  
✅ **Word/sentence boundaries** make chunk display feel more natural  
✅ **Throttling** prevents excessive DOM updates  
✅ **Catch-up mode** prevents display falling too far behind

---

## Further Reading

- [CSS Animations Performance](https://web.dev/animations-guide/)
- [setTimeout vs requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame)
- [AI SDK experimental_throttle](https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat)

---

**Previous:** [Displaying Text as It Arrives](./01-displaying-text-as-it-arrives.md)  
**Next:** [Cursor Indicators](./03-cursor-indicators.md)

<!-- 
Sources Consulted:
- web.dev CSS Animations: https://web.dev/animations-guide/
- Vercel AI SDK: https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat
-->
