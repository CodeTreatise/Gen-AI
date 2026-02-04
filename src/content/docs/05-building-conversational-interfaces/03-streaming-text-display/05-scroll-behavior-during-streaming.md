---
title: "Scroll Behavior During Streaming"
---

# Scroll Behavior During Streaming

## Introduction

When AI responses stream in, the chat must scroll to keep new content visible. Poor scroll handling frustrates users—they lose their place or can't see new text. Good scroll behavior feels invisible because it just works.

In this lesson, we'll implement smooth auto-scrolling that adapts to streaming content.

### What We'll Cover

- Auto-scroll to new content
- Smooth vs instant scrolling
- Scroll anchoring techniques
- Keeping the cursor visible
- Performance optimization

### Prerequisites

- [Displaying Text as It Arrives](./01-displaying-text-as-it-arrives.md)
- CSS `scroll-behavior` property
- JavaScript scroll APIs

---

## Basic Auto-Scroll

### Scroll to Bottom

The simplest approach—scroll to bottom on each update:

```javascript
function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}

// Usage during streaming
function handleStreamChunk(chunk) {
  appendContent(chunk);
  scrollToBottom(messagesContainer);
}
```

### Smooth Scroll to Bottom

Add smooth scrolling for better UX:

```javascript
function smoothScrollToBottom(container) {
  container.scrollTo({
    top: container.scrollHeight,
    behavior: 'smooth'
  });
}
```

### CSS-Based Smooth Scroll

```css
.messages-container {
  overflow-y: auto;
  scroll-behavior: smooth;
}
```

```javascript
// With CSS scroll-behavior, just set scrollTop
function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}
```

---

## Throttled Scrolling

Scrolling on every chunk is wasteful. Throttle scroll updates:

```javascript
class ThrottledScroller {
  constructor(container, options = {}) {
    this.container = container;
    this.throttleMs = options.throttle || 100;
    this.lastScroll = 0;
    this.pendingScroll = false;
  }
  
  scrollToBottom() {
    const now = Date.now();
    
    if (now - this.lastScroll >= this.throttleMs) {
      this.performScroll();
      this.lastScroll = now;
    } else if (!this.pendingScroll) {
      this.pendingScroll = true;
      setTimeout(() => {
        this.performScroll();
        this.pendingScroll = false;
        this.lastScroll = Date.now();
      }, this.throttleMs - (now - this.lastScroll));
    }
  }
  
  performScroll() {
    this.container.scrollTop = this.container.scrollHeight;
  }
}

// Usage
const scroller = new ThrottledScroller(messagesContainer, { throttle: 100 });

function handleStreamChunk(chunk) {
  appendContent(chunk);
  scroller.scrollToBottom();
}
```

### RAF-Based Throttling

Use `requestAnimationFrame` for smooth, efficient scrolling:

```javascript
class RAFScroller {
  constructor(container) {
    this.container = container;
    this.rafId = null;
  }
  
  scrollToBottom() {
    if (this.rafId) return;  // Already scheduled
    
    this.rafId = requestAnimationFrame(() => {
      this.container.scrollTop = this.container.scrollHeight;
      this.rafId = null;
    });
  }
  
  cancel() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }
}
```

---

## Scroll Anchoring

Modern browsers support scroll anchoring—keeping the user's scroll position stable when content above changes:

```css
.messages-container {
  overflow-anchor: auto;  /* Enable scroll anchoring */
}

/* Anchor to the bottom element */
.scroll-anchor {
  overflow-anchor: auto;
  height: 1px;
}
```

```html
<div class="messages-container">
  <div class="message">...</div>
  <div class="message">...</div>
  <!-- Anchor element at bottom -->
  <div class="scroll-anchor"></div>
</div>
```

### Anchor Element Approach

```javascript
class AnchoredScroller {
  constructor(container) {
    this.container = container;
    this.anchor = document.createElement('div');
    this.anchor.className = 'scroll-anchor';
    container.appendChild(this.anchor);
  }
  
  scrollToAnchor() {
    this.anchor.scrollIntoView({ 
      behavior: 'smooth',
      block: 'end'
    });
  }
  
  scrollToAnchorInstant() {
    this.anchor.scrollIntoView({ block: 'end' });
  }
}
```

---

## Scroll Modes

### Instant vs Smooth Comparison

| Mode | Use Case | Behavior |
|------|----------|----------|
| **Instant** | Rapid updates, initial load | Jump immediately |
| **Smooth** | User-facing updates | Animate scroll |
| **Auto** | Mixed content | Smooth when small distance |

### Adaptive Scroll Mode

```javascript
class AdaptiveScroller {
  constructor(container, options = {}) {
    this.container = container;
    this.smoothThreshold = options.smoothThreshold || 200;  // px
  }
  
  scrollToBottom() {
    const distance = this.container.scrollHeight - 
                    this.container.scrollTop - 
                    this.container.clientHeight;
    
    // Use instant scroll for large distances
    const behavior = distance > this.smoothThreshold ? 'instant' : 'smooth';
    
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior
    });
  }
}
```

### Scroll Only When Near Bottom

Only auto-scroll if user is already near the bottom:

```javascript
class SmartScroller {
  constructor(container, options = {}) {
    this.container = container;
    this.threshold = options.threshold || 100;  // px from bottom
  }
  
  isNearBottom() {
    const { scrollTop, scrollHeight, clientHeight } = this.container;
    return scrollHeight - scrollTop - clientHeight < this.threshold;
  }
  
  scrollToBottomIfNeeded() {
    if (this.isNearBottom()) {
      this.container.scrollTop = this.container.scrollHeight;
    }
  }
}
```

---

## Keeping Cursor Visible

Ensure the streaming cursor stays in view:

```javascript
class CursorVisibilityScroller {
  constructor(container, cursorElement) {
    this.container = container;
    this.cursor = cursorElement;
  }
  
  ensureCursorVisible() {
    const containerRect = this.container.getBoundingClientRect();
    const cursorRect = this.cursor.getBoundingClientRect();
    
    // Check if cursor is below visible area
    if (cursorRect.bottom > containerRect.bottom) {
      const scrollAmount = cursorRect.bottom - containerRect.bottom + 20;  // 20px padding
      this.container.scrollTop += scrollAmount;
    }
    
    // Check if cursor is above visible area
    if (cursorRect.top < containerRect.top) {
      const scrollAmount = containerRect.top - cursorRect.top + 20;
      this.container.scrollTop -= scrollAmount;
    }
  }
}
```

### Scroll Margin for Cursor

Use CSS `scroll-margin` to add padding:

```css
.streaming-cursor {
  scroll-margin-bottom: 50px;
}
```

```javascript
function scrollCursorIntoView(cursor) {
  cursor.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest'
  });
}
```

---

## React Implementation

### Basic Auto-Scroll Hook

```jsx
function useAutoScroll(dependency, options = {}) {
  const containerRef = useRef(null);
  const { enabled = true, smooth = true } = options;
  
  useEffect(() => {
    if (!enabled || !containerRef.current) return;
    
    const container = containerRef.current;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: smooth ? 'smooth' : 'instant'
    });
  }, [dependency, enabled, smooth]);
  
  return containerRef;
}

function Chat() {
  const { messages, isStreaming } = useChat();
  const containerRef = useAutoScroll(messages);
  
  return (
    <div ref={containerRef} className="messages-container">
      {messages.map(msg => (
        <Message key={msg.id} {...msg} />
      ))}
    </div>
  );
}
```

### Throttled Scroll Hook

```jsx
function useThrottledAutoScroll(dependency, options = {}) {
  const containerRef = useRef(null);
  const lastScrollRef = useRef(0);
  const rafIdRef = useRef(null);
  
  const { throttle = 100, enabled = true } = options;
  
  useEffect(() => {
    if (!enabled || !containerRef.current) return;
    
    const now = Date.now();
    
    if (now - lastScrollRef.current >= throttle) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
      lastScrollRef.current = now;
    } else if (!rafIdRef.current) {
      rafIdRef.current = requestAnimationFrame(() => {
        if (containerRef.current) {
          containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
        rafIdRef.current = null;
        lastScrollRef.current = Date.now();
      });
    }
    
    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [dependency, throttle, enabled]);
  
  return containerRef;
}
```

### Smart Scroll Component

```jsx
function MessageContainer({ children, isStreaming }) {
  const containerRef = useRef(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  
  // Check if near bottom before updates
  const checkScrollPosition = useCallback(() => {
    if (!containerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShouldAutoScroll(isNearBottom);
  }, []);
  
  // Auto-scroll when content changes (if near bottom)
  useEffect(() => {
    if (shouldAutoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [children, shouldAutoScroll]);
  
  // Track scroll position
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    container.addEventListener('scroll', checkScrollPosition);
    return () => container.removeEventListener('scroll', checkScrollPosition);
  }, [checkScrollPosition]);
  
  return (
    <div ref={containerRef} className="messages-container">
      {children}
    </div>
  );
}
```

### Scroll Anchor Component

```jsx
function MessagesWithAnchor({ messages, isStreaming }) {
  const anchorRef = useRef(null);
  
  useEffect(() => {
    if (isStreaming && anchorRef.current) {
      anchorRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, isStreaming]);
  
  return (
    <div className="messages-container">
      {messages.map(msg => (
        <Message key={msg.id} {...msg} />
      ))}
      <div ref={anchorRef} className="scroll-anchor" aria-hidden="true" />
    </div>
  );
}
```

---

## Performance Optimization

### Debounced Scroll

For less frequent, batched scroll updates:

```javascript
function debounce(fn, delay) {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

const debouncedScroll = debounce((container) => {
  container.scrollTop = container.scrollHeight;
}, 50);
```

### Passive Scroll Listener

For monitoring scroll position:

```javascript
container.addEventListener('scroll', handleScroll, { passive: true });
```

### Avoid Layout Thrashing

```javascript
// ❌ Bad: Causes layout thrashing
function badScrollCheck(container) {
  const isNearBottom = container.scrollHeight - container.scrollTop < 100;
  container.style.background = isNearBottom ? 'blue' : 'red';  // Write
  return container.scrollTop;  // Read - forces layout
}

// ✅ Good: Batch reads and writes
function goodScrollCheck(container) {
  // Read
  const { scrollHeight, scrollTop, clientHeight } = container;
  const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
  
  // Write (after all reads)
  requestAnimationFrame(() => {
    container.classList.toggle('near-bottom', isNearBottom);
  });
  
  return isNearBottom;
}
```

---

## Scroll with Intersection Observer

Use Intersection Observer for efficient scroll detection:

```javascript
class IntersectionScroller {
  constructor(container, options = {}) {
    this.container = container;
    this.callback = options.onVisibilityChange;
    
    // Create sentinel element at bottom
    this.sentinel = document.createElement('div');
    this.sentinel.className = 'scroll-sentinel';
    this.sentinel.style.height = '1px';
    container.appendChild(this.sentinel);
    
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          const isVisible = entry.isIntersecting;
          if (this.callback) {
            this.callback(isVisible);
          }
        });
      },
      { root: container, threshold: 0 }
    );
    
    this.observer.observe(this.sentinel);
  }
  
  destroy() {
    this.observer.disconnect();
    this.sentinel.remove();
  }
}

// Usage
const scroller = new IntersectionScroller(container, {
  onVisibilityChange: (bottomVisible) => {
    if (bottomVisible) {
      // User is at bottom, safe to auto-scroll
    } else {
      // User scrolled up, don't auto-scroll
    }
  }
});
```

### React Hook with Intersection Observer

```jsx
function useScrollVisibility(containerRef) {
  const [isAtBottom, setIsAtBottom] = useState(true);
  
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    // Create sentinel
    const sentinel = document.createElement('div');
    sentinel.style.height = '1px';
    container.appendChild(sentinel);
    
    const observer = new IntersectionObserver(
      ([entry]) => setIsAtBottom(entry.isIntersecting),
      { root: container, threshold: 0 }
    );
    
    observer.observe(sentinel);
    
    return () => {
      observer.disconnect();
      sentinel.remove();
    };
  }, [containerRef]);
  
  return isAtBottom;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Throttle scroll updates | Scroll on every chunk |
| Use `requestAnimationFrame` | Use synchronous scrolling in loop |
| Check if user is at bottom first | Force scroll regardless of position |
| Use passive event listeners | Block scroll events |
| Prefer CSS `scroll-behavior` | Always use JavaScript `scrollTo` |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Jerky scrolling during streaming | Throttle scroll calls |
| User loses scroll position | Only scroll if near bottom |
| Performance issues on long chats | Use Intersection Observer |
| Scroll conflicts with user input | Pause auto-scroll on user interaction |
| Layout thrashing | Batch DOM reads and writes |

---

## Hands-on Exercise

### Your Task

Build a scroll manager that:
1. Auto-scrolls during streaming
2. Only scrolls if user is near bottom
3. Uses throttling to prevent performance issues
4. Provides smooth scrolling experience

### Requirements

1. Threshold: 100px from bottom to trigger auto-scroll
2. Throttle: 100ms between scroll operations
3. Smooth scroll behavior
4. Intersection Observer for efficiency

<details>
<summary>💡 Hints (click to expand)</summary>

- Create a sentinel element at the bottom
- Use IntersectionObserver to detect when sentinel is visible
- Only trigger scroll when sentinel is visible
- Use requestAnimationFrame for smooth updates

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class OptimizedAutoScroller {
  constructor(container) {
    this.container = container;
    this.isAtBottom = true;
    this.rafId = null;
    
    // Create sentinel
    this.sentinel = document.createElement('div');
    this.sentinel.style.height = '1px';
    container.appendChild(this.sentinel);
    
    // Observe sentinel
    this.observer = new IntersectionObserver(
      ([entry]) => {
        this.isAtBottom = entry.isIntersecting;
      },
      { root: container, threshold: 0, rootMargin: '100px' }
    );
    
    this.observer.observe(this.sentinel);
  }
  
  scrollToBottom() {
    if (!this.isAtBottom) return;
    if (this.rafId) return;
    
    this.rafId = requestAnimationFrame(() => {
      this.container.scrollTo({
        top: this.container.scrollHeight,
        behavior: 'smooth'
      });
      this.rafId = null;
    });
  }
  
  destroy() {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.observer.disconnect();
    this.sentinel.remove();
  }
}
```

</details>

---

## Summary

✅ **Throttle scroll updates** to prevent performance issues  
✅ **Check user position** before auto-scrolling  
✅ **Use Intersection Observer** for efficient detection  
✅ **Smooth scrolling** improves perceived quality  
✅ **Scroll anchoring** keeps position stable  
✅ **RAF-based scrolling** ensures smooth updates

---

## Further Reading

- [scroll-behavior CSS](https://developer.mozilla.org/en-US/docs/Web/CSS/scroll-behavior)
- [Intersection Observer API](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API)
- [Scroll Anchoring](https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-anchor)

---

**Previous:** [Text Appearance Animations](./04-text-appearance-animations.md)  
**Next:** [Auto-Scroll with User Override](./06-auto-scroll-with-user-override.md)

<!-- 
Sources Consulted:
- MDN scroll-behavior: https://developer.mozilla.org/en-US/docs/Web/CSS/scroll-behavior
- MDN Intersection Observer: https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API
- MDN overflow-anchor: https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-anchor
-->
