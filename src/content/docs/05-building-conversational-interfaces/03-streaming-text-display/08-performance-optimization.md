---
title: "Performance Optimization"
---

# Performance Optimization

## Introduction

Streaming text can generate hundreds of DOM updates per second. Without optimization, this causes jank, dropped frames, and poor user experience. Smart batching, throttling, and render optimization keep the interface smooth.

In this lesson, we'll apply performance techniques specifically for streaming text display.

### What We'll Cover

- Render batching strategies
- Throttling and debouncing
- Virtual DOM efficiency
- Layout thrashing prevention
- Memory management
- Profiling and measurement

### Prerequisites

- [Displaying Text as It Arrives](./01-displaying-text-as-it-arrives.md)
- [AI SDK Streaming States](./07-ai-sdk-streaming-states.md)
- Browser DevTools familiarity

---

## Render Batching

### Time-Based Batching

Accumulate updates and render on interval:

```javascript
class BatchedRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.batchInterval = options.interval || 50;  // ms
    
    this.buffer = '';
    this.displayedText = '';
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
    
    this.intervalId = null;
    this.startBatching();
  }
  
  startBatching() {
    this.intervalId = setInterval(() => {
      this.flush();
    }, this.batchInterval);
  }
  
  append(text) {
    this.buffer += text;
  }
  
  flush() {
    if (this.buffer) {
      this.displayedText += this.buffer;
      this.textNode.textContent = this.displayedText;
      this.buffer = '';
    }
  }
  
  complete() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    this.flush();
  }
  
  destroy() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
}
```

### RAF-Based Batching

Use requestAnimationFrame for smoother updates:

```javascript
class RAFBatchedRenderer {
  constructor(container) {
    this.container = container;
    this.buffer = '';
    this.displayedText = '';
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
    
    this.rafId = null;
    this.isComplete = false;
  }
  
  append(text) {
    this.buffer += text;
    this.scheduleRender();
  }
  
  scheduleRender() {
    if (this.rafId || this.isComplete) return;
    
    this.rafId = requestAnimationFrame(() => {
      this.flush();
      this.rafId = null;
      
      // Continue if there's more in buffer
      if (this.buffer) {
        this.scheduleRender();
      }
    });
  }
  
  flush() {
    if (this.buffer) {
      this.displayedText += this.buffer;
      this.textNode.textContent = this.displayedText;
      this.buffer = '';
    }
  }
  
  complete() {
    this.isComplete = true;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    this.flush();
  }
}
```

### Double Buffering

Minimize visible layout changes:

```javascript
class DoubleBufferedRenderer {
  constructor(container) {
    this.container = container;
    
    // Create two content elements
    this.frontBuffer = document.createElement('div');
    this.backBuffer = document.createElement('div');
    
    this.frontBuffer.className = 'buffer active';
    this.backBuffer.className = 'buffer';
    this.backBuffer.style.display = 'none';
    
    container.appendChild(this.frontBuffer);
    container.appendChild(this.backBuffer);
    
    this.content = '';
    this.rafId = null;
  }
  
  append(text) {
    this.content += text;
    this.scheduleSwap();
  }
  
  scheduleSwap() {
    if (this.rafId) return;
    
    this.rafId = requestAnimationFrame(() => {
      // Write to back buffer
      this.backBuffer.textContent = this.content;
      
      // Swap buffers
      this.frontBuffer.style.display = 'none';
      this.backBuffer.style.display = '';
      
      // Swap references
      const temp = this.frontBuffer;
      this.frontBuffer = this.backBuffer;
      this.backBuffer = temp;
      
      this.rafId = null;
    });
  }
}
```

---

## Throttling Strategies

### Time-Based Throttle

```javascript
function createThrottledUpdater(callback, limit = 50) {
  let lastCall = 0;
  let pendingValue = null;
  let timeoutId = null;
  
  return function(value) {
    const now = Date.now();
    pendingValue = value;
    
    if (now - lastCall >= limit) {
      callback(value);
      lastCall = now;
      pendingValue = null;
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        callback(pendingValue);
        lastCall = Date.now();
        timeoutId = null;
        pendingValue = null;
      }, limit - (now - lastCall));
    }
  };
}

// Usage
const throttledUpdate = createThrottledUpdater((content) => {
  messageElement.textContent = content;
}, 50);

// Call on every chunk
stream.on('data', chunk => {
  buffer += chunk;
  throttledUpdate(buffer);
});
```

### Frame-Rate Throttle

Match display refresh rate:

```javascript
class FrameThrottledRenderer {
  constructor(container, fps = 30) {
    this.container = container;
    this.frameInterval = 1000 / fps;
    
    this.buffer = '';
    this.displayedText = '';
    this.lastFrame = 0;
    this.rafId = null;
    
    this.textNode = document.createTextNode('');
    container.appendChild(this.textNode);
  }
  
  append(text) {
    this.buffer += text;
    this.scheduleFrame();
  }
  
  scheduleFrame() {
    if (this.rafId) return;
    
    this.rafId = requestAnimationFrame((timestamp) => {
      const elapsed = timestamp - this.lastFrame;
      
      if (elapsed >= this.frameInterval) {
        this.flush();
        this.lastFrame = timestamp;
      }
      
      this.rafId = null;
      
      if (this.buffer) {
        this.scheduleFrame();
      }
    });
  }
  
  flush() {
    if (this.buffer) {
      this.displayedText += this.buffer;
      this.textNode.textContent = this.displayedText;
      this.buffer = '';
    }
  }
  
  complete() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    this.flush();
  }
}
```

---

## React Optimization

### Memoized Message Component

```jsx
const Message = memo(function Message({ content, role, isStreaming }) {
  return (
    <div className={`message ${role}`}>
      <span className="message-content">{content}</span>
      {isStreaming && <StreamingCursor />}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if content or streaming changed
  return (
    prevProps.content === nextProps.content &&
    prevProps.isStreaming === nextProps.isStreaming
  );
});
```

### Throttled State Updates

```jsx
function useThrottledState(initialValue, delay = 50) {
  const [value, setValue] = useState(initialValue);
  const [throttledValue, setThrottledValue] = useState(initialValue);
  const timeoutRef = useRef(null);
  const lastUpdateRef = useRef(Date.now());
  
  useEffect(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateRef.current;
    
    if (timeSinceLastUpdate >= delay) {
      setThrottledValue(value);
      lastUpdateRef.current = now;
    } else {
      timeoutRef.current = setTimeout(() => {
        setThrottledValue(value);
        lastUpdateRef.current = Date.now();
      }, delay - timeSinceLastUpdate);
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, delay]);
  
  return [throttledValue, setValue];
}
```

### Optimized Message List

```jsx
function OptimizedMessageList({ messages, status }) {
  // Only the last message changes during streaming
  const stableMessages = useMemo(() => {
    return messages.slice(0, -1);
  }, [messages.length > 1 ? messages[messages.length - 2]?.id : null]);
  
  const lastMessage = messages[messages.length - 1];
  const isStreaming = status === 'streaming';
  
  return (
    <div className="messages">
      {/* Stable messages - won't re-render during streaming */}
      {stableMessages.map(msg => (
        <Message 
          key={msg.id} 
          content={msg.content} 
          role={msg.role}
          isStreaming={false}
        />
      ))}
      
      {/* Only last message updates during streaming */}
      {lastMessage && (
        <StreamingMessage
          key={lastMessage.id}
          content={lastMessage.content}
          role={lastMessage.role}
          isStreaming={isStreaming && lastMessage.role === 'assistant'}
        />
      )}
    </div>
  );
}
```

### Streaming Message with Internal Throttle

```jsx
function StreamingMessage({ content, role, isStreaming }) {
  const [displayedContent, setDisplayedContent] = useState(content);
  const contentRef = useRef(content);
  const rafRef = useRef(null);
  
  useEffect(() => {
    contentRef.current = content;
    
    if (isStreaming && !rafRef.current) {
      const updateLoop = () => {
        setDisplayedContent(contentRef.current);
        if (isStreaming) {
          rafRef.current = requestAnimationFrame(updateLoop);
        }
      };
      rafRef.current = requestAnimationFrame(updateLoop);
    }
    
    if (!isStreaming) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setDisplayedContent(content);
    }
    
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [content, isStreaming]);
  
  return (
    <div className={`message ${role}`}>
      <span className="message-content">{displayedContent}</span>
      {isStreaming && <StreamingCursor />}
    </div>
  );
}
```

---

## Layout Thrashing Prevention

### Batch DOM Reads and Writes

```javascript
// ❌ Bad: Interleaved reads and writes
function badUpdate(elements) {
  elements.forEach(el => {
    const height = el.offsetHeight;  // Read
    el.style.height = height + 10 + 'px';  // Write
    // Forces layout recalculation on next read
  });
}

// ✅ Good: Batch reads, then batch writes
function goodUpdate(elements) {
  // Read phase
  const heights = elements.map(el => el.offsetHeight);
  
  // Write phase (in RAF for better timing)
  requestAnimationFrame(() => {
    elements.forEach((el, i) => {
      el.style.height = heights[i] + 10 + 'px';
    });
  });
}
```

### FastDOM Pattern

```javascript
class FastDOMRenderer {
  constructor() {
    this.readQueue = [];
    this.writeQueue = [];
    this.scheduled = false;
  }
  
  read(fn) {
    this.readQueue.push(fn);
    this.schedule();
  }
  
  write(fn) {
    this.writeQueue.push(fn);
    this.schedule();
  }
  
  schedule() {
    if (this.scheduled) return;
    this.scheduled = true;
    
    requestAnimationFrame(() => {
      // Execute all reads first
      this.readQueue.forEach(fn => fn());
      this.readQueue = [];
      
      // Then all writes
      this.writeQueue.forEach(fn => fn());
      this.writeQueue = [];
      
      this.scheduled = false;
    });
  }
}

// Usage
const fastDOM = new FastDOMRenderer();

function updateStreamingMessage(container, content) {
  let scrollHeight;
  
  fastDOM.read(() => {
    scrollHeight = container.scrollHeight;
  });
  
  fastDOM.write(() => {
    container.textContent = content;
    container.scrollTop = scrollHeight;
  });
}
```

---

## Memory Management

### Cleanup Streaming Resources

```javascript
class ManagedStreamRenderer {
  constructor(container) {
    this.container = container;
    this.textNode = null;
    this.rafId = null;
    this.intervalId = null;
    this.observers = [];
  }
  
  init() {
    this.textNode = document.createTextNode('');
    this.container.appendChild(this.textNode);
  }
  
  addObserver(observer) {
    this.observers.push(observer);
  }
  
  complete() {
    // Cancel pending animations
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    
    // Clear intervals
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    
    // Disconnect observers
    this.observers.forEach(obs => obs.disconnect());
    this.observers = [];
  }
  
  destroy() {
    this.complete();
    
    if (this.textNode && this.textNode.parentNode) {
      this.textNode.parentNode.removeChild(this.textNode);
    }
    this.textNode = null;
  }
}
```

### React Cleanup

```jsx
function StreamingMessage({ content, isStreaming }) {
  const rafRef = useRef(null);
  const observerRef = useRef(null);
  const contentRef = useRef(null);
  
  useEffect(() => {
    // Setup
    if (isStreaming && contentRef.current) {
      observerRef.current = new IntersectionObserver(
        // ... observer logic
      );
      observerRef.current.observe(contentRef.current);
    }
    
    // Cleanup on unmount or when streaming stops
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
    };
  }, [isStreaming]);
  
  return (
    <div ref={contentRef} className="message-content">
      {content}
    </div>
  );
}
```

### Long Chat Memory

For very long conversations:

```jsx
function VirtualizedMessages({ messages, status }) {
  const WINDOW_SIZE = 50;  // Render 50 messages at a time
  const [startIndex, setStartIndex] = useState(
    Math.max(0, messages.length - WINDOW_SIZE)
  );
  
  const visibleMessages = useMemo(() => {
    return messages.slice(startIndex, startIndex + WINDOW_SIZE);
  }, [messages, startIndex]);
  
  const handleScroll = useCallback((e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    
    // Load older messages when scrolling up
    if (scrollTop < 100 && startIndex > 0) {
      setStartIndex(Math.max(0, startIndex - 10));
    }
  }, [startIndex]);
  
  return (
    <div className="messages" onScroll={handleScroll}>
      {startIndex > 0 && (
        <button onClick={() => setStartIndex(0)}>
          Load older messages ({startIndex} hidden)
        </button>
      )}
      
      {visibleMessages.map(msg => (
        <Message key={msg.id} {...msg} />
      ))}
    </div>
  );
}
```

---

## Profiling and Measurement

### Performance Marks

```javascript
class ProfiledRenderer {
  constructor(container) {
    this.container = container;
    this.chunkCount = 0;
    this.renderCount = 0;
  }
  
  append(text) {
    performance.mark('chunk-received');
    this.chunkCount++;
    
    this.buffer += text;
    this.scheduleRender();
  }
  
  render() {
    performance.mark('render-start');
    
    this.textNode.textContent = this.displayedText + this.buffer;
    this.displayedText += this.buffer;
    this.buffer = '';
    
    performance.mark('render-end');
    performance.measure('render-time', 'render-start', 'render-end');
    
    this.renderCount++;
  }
  
  getStats() {
    const measures = performance.getEntriesByName('render-time');
    const avgRenderTime = measures.reduce((a, b) => a + b.duration, 0) / measures.length;
    
    return {
      chunks: this.chunkCount,
      renders: this.renderCount,
      avgRenderTime: avgRenderTime.toFixed(2) + 'ms',
      renderRatio: (this.renderCount / this.chunkCount * 100).toFixed(1) + '%'
    };
  }
  
  clearStats() {
    performance.clearMarks();
    performance.clearMeasures();
    this.chunkCount = 0;
    this.renderCount = 0;
  }
}
```

### React Performance Hook

```jsx
function useRenderProfiler(componentName) {
  const renderCountRef = useRef(0);
  const lastRenderRef = useRef(performance.now());
  
  useEffect(() => {
    renderCountRef.current++;
    const now = performance.now();
    const timeSinceLastRender = now - lastRenderRef.current;
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`${componentName} render #${renderCountRef.current}`, {
        timeSinceLastRender: timeSinceLastRender.toFixed(2) + 'ms'
      });
    }
    
    lastRenderRef.current = now;
  });
}

// Usage
function StreamingMessage({ content }) {
  useRenderProfiler('StreamingMessage');
  
  return <div>{content}</div>;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Batch DOM updates | Update on every chunk |
| Use `requestAnimationFrame` | Use `setTimeout(fn, 0)` |
| Memoize stable components | Re-render entire message list |
| Clean up resources | Leave RAF/intervals running |
| Profile in production mode | Only test in development |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| 100+ renders per second | Use `experimental_throttle` or custom batching |
| Memory leak from RAF | Always cancel in cleanup |
| Entire list re-renders | Separate stable and streaming messages |
| Layout thrashing | Batch reads before writes |
| Slow with long chats | Virtualize message list |

---

## Performance Checklist

Before deploying streaming chat:

- [ ] **Render batching**: Updates throttled to 20-60 FPS
- [ ] **Memoization**: Stable messages don't re-render
- [ ] **RAF cleanup**: Canceled on complete/unmount
- [ ] **Memory**: Old messages unloaded or virtualized
- [ ] **Layout**: No forced synchronous layouts
- [ ] **Profiling**: Tested with 1000+ messages

---

## Hands-on Exercise

### Your Task

Optimize a streaming message component to:
1. Batch renders to max 30 FPS
2. Prevent re-renders of non-streaming messages
3. Clean up all resources on complete
4. Profile render performance

### Requirements

1. Use RAF for throttling (not setInterval)
2. Memoize message components
3. Cancel all animations on cleanup
4. Log render stats to console

<details>
<summary>💡 Hints (click to expand)</summary>

- 30 FPS = ~33ms per frame
- Use `useRef` to track latest content
- Use `useEffect` cleanup function
- Use `React.memo` with custom comparison

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `StreamingMessage with Internal Throttle` and `OptimizedMessageList` components in the React Optimization section above.

</details>

---

## Summary

✅ **Batch updates** to reduce DOM operations  
✅ **RAF-based throttling** matches display refresh  
✅ **Memoization** prevents unnecessary re-renders  
✅ **Cleanup resources** to prevent memory leaks  
✅ **Batch DOM reads/writes** to prevent layout thrashing  
✅ **Profile performance** with browser DevTools

---

## Further Reading

- [requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame)
- [React Performance](https://react.dev/learn/render-and-commit)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)
- [FastDOM](https://github.com/wilsonpage/fastdom)

---

**Previous:** [AI SDK Streaming States](./07-ai-sdk-streaming-states.md)  
**Back to:** [Streaming Text Display Overview](./00-streaming-text-display.md)

<!-- 
Sources Consulted:
- MDN requestAnimationFrame: https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame
- React render and commit: https://react.dev/learn/render-and-commit
- Chrome Performance DevTools: https://developer.chrome.com/docs/devtools/performance/
-->
