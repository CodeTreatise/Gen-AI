---
title: "Cursor Indicators"
---

# Cursor Indicators

## Introduction

A blinking cursor signals that the AI is actively generating content—it's the digital equivalent of watching someone type. This visual indicator sets user expectations and creates engagement during streaming.

In this lesson, we'll implement various cursor styles, manage their positioning, and ensure smooth animations that don't impact performance.

### What We'll Cover

- Blinking cursor styles and animations
- Cursor position management
- Inline vs trailing cursor placement
- Accessibility considerations
- Performance optimization

### Prerequisites

- [Displaying Text as It Arrives](./01-displaying-text-as-it-arrives.md)
- CSS animations and keyframes
- DOM manipulation basics

---

## Cursor Styles

### Classic Block Cursor

```css
.cursor-block {
  display: inline-block;
  width: 0.6em;
  height: 1.1em;
  background-color: currentColor;
  vertical-align: text-bottom;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
```

### Line Cursor

```css
.cursor-line {
  display: inline-block;
  width: 2px;
  height: 1.2em;
  background-color: currentColor;
  vertical-align: text-bottom;
  animation: blink 1s step-end infinite;
}
```

### Underscore Cursor

```css
.cursor-underscore {
  display: inline-block;
  width: 0.6em;
  height: 2px;
  background-color: currentColor;
  vertical-align: baseline;
  animation: blink 1s step-end infinite;
}
```

### Unicode Cursors

For simplicity, use Unicode characters:

```javascript
const CURSORS = {
  block: '▋',      // LOWER HALF BLOCK
  fullBlock: '█',  // FULL BLOCK
  line: '│',       // BOX DRAWINGS LIGHT VERTICAL
  underscore: '_',
  dots: '…',       // HORIZONTAL ELLIPSIS
};
```

### Comparison

| Style | Character | Best For |
|-------|-----------|----------|
| Block | `▋` | Terminal feel, code |
| Full Block | `█` | High visibility |
| Line | `│` | Modern, minimal |
| Underscore | `_` | Classic terminal |
| Ellipsis | `…` | Thinking indicator |

---

## Smooth Blink Animation

### CSS-Based Blink

```css
/* Smoother blink with ease timing */
.cursor-smooth {
  animation: smooth-blink 1s ease-in-out infinite;
}

@keyframes smooth-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }  /* Not fully hidden */
}
```

### Pulse Effect

```css
.cursor-pulse {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { 
    opacity: 1;
    transform: scale(1);
  }
  50% { 
    opacity: 0.6;
    transform: scale(0.95);
  }
}
```

### Pause on Activity

Freeze the cursor during rapid updates:

```javascript
class CursorManager {
  constructor(cursorElement) {
    this.cursor = cursorElement;
    this.activityTimeout = null;
  }
  
  onTextUpdate() {
    // Pause blinking during activity
    this.cursor.style.animationPlayState = 'paused';
    this.cursor.style.opacity = '1';
    
    // Resume after brief pause
    clearTimeout(this.activityTimeout);
    this.activityTimeout = setTimeout(() => {
      this.cursor.style.animationPlayState = 'running';
    }, 150);
  }
  
  hide() {
    this.cursor.style.display = 'none';
    clearTimeout(this.activityTimeout);
  }
  
  show() {
    this.cursor.style.display = 'inline-block';
  }
}
```

---

## Cursor Positioning

### Inline Cursor (Follows Text)

```javascript
class InlineCursor {
  constructor(container) {
    this.container = container;
    this.textNode = document.createTextNode('');
    this.cursor = document.createElement('span');
    this.cursor.className = 'cursor-block';
    this.cursor.textContent = '▋';
    
    container.appendChild(this.textNode);
    container.appendChild(this.cursor);
  }
  
  updateText(text) {
    this.textNode.textContent = text;
    // Cursor automatically follows via DOM order
  }
  
  complete() {
    this.cursor.remove();
  }
}
```

### Fixed Position Cursor

```css
.message-body {
  position: relative;
}

.cursor-fixed {
  position: absolute;
  animation: blink 1s step-end infinite;
}
```

```javascript
class FixedCursor {
  constructor(container) {
    this.container = container;
    this.textNode = document.createTextNode('');
    this.cursor = document.createElement('span');
    this.cursor.className = 'cursor-fixed';
    this.cursor.textContent = '▋';
    
    container.appendChild(this.textNode);
    container.appendChild(this.cursor);
  }
  
  updateText(text) {
    this.textNode.textContent = text;
    this.updateCursorPosition();
  }
  
  updateCursorPosition() {
    // Position cursor at end of text
    const range = document.createRange();
    range.selectNodeContents(this.container);
    range.collapse(false);  // Collapse to end
    
    const rect = range.getBoundingClientRect();
    const containerRect = this.container.getBoundingClientRect();
    
    this.cursor.style.left = `${rect.left - containerRect.left}px`;
    this.cursor.style.top = `${rect.top - containerRect.top}px`;
  }
  
  complete() {
    this.cursor.remove();
  }
}
```

### Cursor After Line Breaks

Handle multi-line text correctly:

```javascript
class MultilineCursor {
  constructor(container) {
    this.container = container;
    this.content = document.createElement('span');
    this.cursor = document.createElement('span');
    this.cursor.className = 'cursor';
    this.cursor.textContent = '▋';
    
    container.appendChild(this.content);
    container.appendChild(this.cursor);
  }
  
  updateText(text) {
    // Convert newlines to <br> for proper rendering
    const html = text
      .split('\n')
      .map(line => this.escapeHtml(line))
      .join('<br>');
    
    this.content.innerHTML = html;
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  complete() {
    this.cursor.remove();
  }
}
```

---

## React Implementation

### Basic Cursor Component

```jsx
function StreamingCursor({ visible = true, style = 'block' }) {
  if (!visible) return null;
  
  const cursorChar = {
    block: '▋',
    line: '│',
    underscore: '_',
    dots: '…'
  }[style] || '▋';
  
  return (
    <span 
      className="streaming-cursor" 
      aria-hidden="true"
    >
      {cursorChar}
    </span>
  );
}
```

```css
.streaming-cursor {
  display: inline-block;
  animation: blink 1s step-end infinite;
  margin-left: 1px;
  font-weight: normal;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
```

### Message with Cursor

```jsx
function StreamingMessage({ content, isStreaming }) {
  return (
    <div className="message-body">
      <span className="message-text">{content}</span>
      <StreamingCursor visible={isStreaming} />
    </div>
  );
}
```

### Cursor with Activity Pause

```jsx
function ActivityAwareCursor({ isStreaming, lastUpdate }) {
  const [isPaused, setIsPaused] = useState(false);
  
  useEffect(() => {
    if (!isStreaming) return;
    
    // Pause on update
    setIsPaused(true);
    
    const timer = setTimeout(() => {
      setIsPaused(false);
    }, 150);
    
    return () => clearTimeout(timer);
  }, [lastUpdate, isStreaming]);
  
  if (!isStreaming) return null;
  
  return (
    <span 
      className={`streaming-cursor ${isPaused ? 'paused' : ''}`}
      aria-hidden="true"
    >
      ▋
    </span>
  );
}
```

```css
.streaming-cursor.paused {
  animation-play-state: paused;
  opacity: 1;
}
```

### Complete Streaming Message

```jsx
function CompleteStreamingMessage({ content, status }) {
  const isStreaming = status === 'streaming';
  const [displayedContent, setDisplayedContent] = useState('');
  const prevLengthRef = useRef(0);
  
  useEffect(() => {
    setDisplayedContent(content);
    prevLengthRef.current = content.length;
  }, [content]);
  
  const contentChanged = content.length !== prevLengthRef.current;
  
  return (
    <div className="message ai-message">
      <div className="message-avatar">
        <span aria-label="AI">🤖</span>
      </div>
      <div className="message-body">
        <span className="message-text">{displayedContent}</span>
        <ActivityAwareCursor 
          isStreaming={isStreaming}
          lastUpdate={content.length}
        />
      </div>
    </div>
  );
}
```

---

## Accessibility

### Screen Reader Considerations

```jsx
function AccessibleStreamingMessage({ content, isStreaming }) {
  return (
    <div 
      className="message-body"
      aria-live="polite"
      aria-busy={isStreaming}
    >
      <span className="message-text">{content}</span>
      
      {/* Hide cursor from screen readers */}
      <span className="streaming-cursor" aria-hidden="true">
        {isStreaming && '▋'}
      </span>
      
      {/* Announce streaming state */}
      <span className="sr-only">
        {isStreaming ? 'AI is typing...' : ''}
      </span>
    </div>
  );
}
```

### Reduced Motion Support

```css
.streaming-cursor {
  animation: blink 1s step-end infinite;
}

@media (prefers-reduced-motion: reduce) {
  .streaming-cursor {
    animation: none;
    opacity: 1;  /* Static cursor */
  }
}
```

### Focus Management

```jsx
function AccessibleChat({ messages, isStreaming }) {
  const streamingRef = useRef(null);
  
  useEffect(() => {
    if (isStreaming && streamingRef.current) {
      // Announce to screen readers without stealing focus
      streamingRef.current.setAttribute('aria-live', 'polite');
    }
  }, [isStreaming]);
  
  return (
    <div className="chat-messages" role="log" aria-label="Chat messages">
      {messages.map((msg, i) => (
        <div
          key={msg.id}
          ref={i === messages.length - 1 ? streamingRef : null}
          className={`message ${msg.role}`}
        >
          {msg.content}
          {i === messages.length - 1 && msg.role === 'assistant' && (
            <StreamingCursor visible={isStreaming} />
          )}
        </div>
      ))}
    </div>
  );
}
```

---

## Performance Optimization

### GPU-Accelerated Animation

```css
.streaming-cursor {
  /* Use opacity for GPU acceleration */
  animation: blink-opacity 1s step-end infinite;
  will-change: opacity;
}

@keyframes blink-opacity {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
```

### Avoid Layout Thrashing

```css
/* Bad: causes layout recalculation */
.cursor-bad {
  animation: blink-visibility 1s step-end infinite;
}

@keyframes blink-visibility {
  0%, 100% { visibility: visible; }
  50% { visibility: hidden; }
}

/* Good: opacity doesn't trigger layout */
.cursor-good {
  animation: blink-opacity 1s step-end infinite;
}
```

### Cursor Pooling

For many concurrent streams:

```javascript
const cursorPool = {
  available: [],
  
  acquire() {
    if (this.available.length > 0) {
      return this.available.pop();
    }
    
    const cursor = document.createElement('span');
    cursor.className = 'streaming-cursor';
    cursor.textContent = '▋';
    cursor.setAttribute('aria-hidden', 'true');
    return cursor;
  },
  
  release(cursor) {
    cursor.remove();
    this.available.push(cursor);
  }
};
```

---

## Custom Cursor Animations

### Typing Dots

```css
.typing-dots {
  display: inline-flex;
  gap: 2px;
  margin-left: 4px;
}

.typing-dots span {
  width: 4px;
  height: 4px;
  background: currentColor;
  border-radius: 50%;
  animation: dot-bounce 1.4s ease-in-out infinite;
}

.typing-dots span:nth-child(1) { animation-delay: 0s; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes dot-bounce {
  0%, 80%, 100% { 
    transform: translateY(0);
    opacity: 0.5;
  }
  40% { 
    transform: translateY(-4px);
    opacity: 1;
  }
}
```

```jsx
function TypingDots() {
  return (
    <span className="typing-dots" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </span>
  );
}
```

### Wave Cursor

```css
.cursor-wave {
  display: inline-block;
  animation: wave 1s ease-in-out infinite;
}

@keyframes wave {
  0%, 100% { transform: translateY(0); }
  25% { transform: translateY(-2px); }
  75% { transform: translateY(2px); }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `opacity` for blink animation | Use `visibility` or `display` |
| Hide cursor with `aria-hidden` | Leave cursor visible to screen readers |
| Pause animation during rapid updates | Let cursor blink during typing |
| Support `prefers-reduced-motion` | Force animations on all users |
| Use CSS for animations | Use JavaScript timers for blink |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Cursor blinks during text updates | Pause animation, show solid cursor |
| Cursor causes text reflow | Use inline element, not block |
| Animation continues after complete | Remove cursor element or hide |
| High CPU on many cursors | Use CSS animations, not JS timers |
| Cursor wrong position after wrap | Use inline positioning, not absolute |

---

## Hands-on Exercise

### Your Task

Create a cursor component that:
1. Blinks smoothly when idle
2. Stays solid during text updates
3. Respects reduced motion preferences
4. Is accessible (hidden from screen readers)

### Requirements

1. CSS-based blink animation
2. JavaScript to pause/resume on updates
3. `aria-hidden="true"` attribute
4. `prefers-reduced-motion` media query

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `animation-play-state` to pause
- Track last update time with useState
- Use useEffect to set pause timer
- Add media query to CSS, not JavaScript

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `ActivityAwareCursor` component and associated CSS in the React Implementation section above.

</details>

---

## Summary

✅ **Block cursor** (`▋`) is most common for AI streaming  
✅ **CSS animations** are more efficient than JavaScript timers  
✅ **Pause blinking** during active text updates  
✅ **Hide from screen readers** with `aria-hidden`  
✅ **Support reduced motion** for accessibility  
✅ **Use opacity** for GPU-accelerated animation

---

## Further Reading

- [CSS Animations Performance](https://developer.mozilla.org/en-US/docs/Web/Performance/CSS_JavaScript_animation_performance)
- [prefers-reduced-motion](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
- [ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)

---

**Previous:** [Character vs Chunk Display](./02-character-vs-chunk-display.md)  
**Next:** [Text Appearance Animations](./04-text-appearance-animations.md)

<!-- 
Sources Consulted:
- MDN CSS Animations: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations
- MDN prefers-reduced-motion: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion
- web.dev animation performance: https://web.dev/animations-guide/
-->
