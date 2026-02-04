---
title: "Typing Indicators"
---

# Typing Indicators

## Introduction

The typing indicator—those familiar bouncing dots—signals that the AI is processing. It bridges the gap between sending a message and receiving a response, reducing uncertainty and anxiety.

In this lesson, we'll build various typing indicator patterns, from classic dots to modern skeleton previews.

### What We'll Cover

- Animated dots patterns
- Pulsing indicators
- "AI is thinking" text messages
- Skeleton text previews
- Accessibility considerations

### Prerequisites

- CSS animations and keyframes
- React basics
- [AI SDK Streaming States](../03-streaming-text-display/07-ai-sdk-streaming-states.md)

---

## Animated Dots Pattern

### Classic Three-Dot Bounce

```jsx
function TypingDots() {
  return (
    <div className="typing-indicator" aria-label="AI is typing">
      <span className="dot"></span>
      <span className="dot"></span>
      <span className="dot"></span>
    </div>
  );
}
```

```css
.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 12px 16px;
  background: var(--message-bg, #f0f0f0);
  border-radius: 18px;
  width: fit-content;
}

.dot {
  width: 8px;
  height: 8px;
  background: var(--dot-color, #666);
  border-radius: 50%;
  animation: dot-bounce 1.4s ease-in-out infinite;
}

.dot:nth-child(1) { animation-delay: 0s; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes dot-bounce {
  0%, 80%, 100% {
    transform: translateY(0);
  }
  40% {
    transform: translateY(-6px);
  }
}
```

### Wave Pattern

```css
.typing-wave .dot {
  animation: dot-wave 1.2s ease-in-out infinite;
}

.typing-wave .dot:nth-child(1) { animation-delay: 0s; }
.typing-wave .dot:nth-child(2) { animation-delay: 0.15s; }
.typing-wave .dot:nth-child(3) { animation-delay: 0.3s; }

@keyframes dot-wave {
  0%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  50% {
    transform: translateY(-8px);
    opacity: 1;
  }
}
```

### Scaling Dots

```css
.typing-scale .dot {
  animation: dot-scale 1s ease-in-out infinite;
}

.typing-scale .dot:nth-child(1) { animation-delay: 0s; }
.typing-scale .dot:nth-child(2) { animation-delay: 0.2s; }
.typing-scale .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes dot-scale {
  0%, 100% {
    transform: scale(0.6);
    opacity: 0.4;
  }
  50% {
    transform: scale(1);
    opacity: 1;
  }
}
```

---

## Pulsing Indicators

### Single Pulse Circle

```jsx
function PulsingIndicator() {
  return (
    <div className="pulse-indicator" aria-label="Processing">
      <span className="pulse-ring"></span>
      <span className="pulse-core"></span>
    </div>
  );
}
```

```css
.pulse-indicator {
  position: relative;
  width: 16px;
  height: 16px;
}

.pulse-core {
  position: absolute;
  width: 8px;
  height: 8px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: var(--accent-color, #007bff);
  border-radius: 50%;
}

.pulse-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 2px solid var(--accent-color, #007bff);
  border-radius: 50%;
  animation: pulse-ring 1.5s ease-out infinite;
}

@keyframes pulse-ring {
  0% {
    transform: scale(0.5);
    opacity: 1;
  }
  100% {
    transform: scale(1.5);
    opacity: 0;
  }
}
```

### Breathing Indicator

```css
.breathing-indicator {
  width: 40px;
  height: 40px;
  background: var(--accent-color, #007bff);
  border-radius: 50%;
  animation: breathe 2s ease-in-out infinite;
}

@keyframes breathe {
  0%, 100% {
    transform: scale(1);
    opacity: 0.7;
  }
  50% {
    transform: scale(1.1);
    opacity: 1;
  }
}
```

### Ripple Effect

```jsx
function RippleIndicator() {
  return (
    <div className="ripple-container">
      <span className="ripple"></span>
      <span className="ripple"></span>
      <span className="ripple"></span>
    </div>
  );
}
```

```css
.ripple-container {
  position: relative;
  width: 40px;
  height: 40px;
}

.ripple {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 2px solid var(--accent-color, #007bff);
  border-radius: 50%;
  animation: ripple 2s ease-out infinite;
}

.ripple:nth-child(1) { animation-delay: 0s; }
.ripple:nth-child(2) { animation-delay: 0.5s; }
.ripple:nth-child(3) { animation-delay: 1s; }

@keyframes ripple {
  0% {
    transform: scale(0.5);
    opacity: 1;
  }
  100% {
    transform: scale(2);
    opacity: 0;
  }
}
```

---

## "AI is Thinking" Messages

### Text with Ellipsis Animation

```jsx
function ThinkingMessage({ text = "AI is thinking" }) {
  return (
    <div className="thinking-message" aria-live="polite">
      <span className="thinking-icon">🤔</span>
      <span className="thinking-text">{text}</span>
      <span className="animated-ellipsis">
        <span>.</span>
        <span>.</span>
        <span>.</span>
      </span>
    </div>
  );
}
```

```css
.thinking-message {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--message-bg, #f5f5f5);
  border-radius: 12px;
  font-size: 14px;
  color: var(--text-secondary, #666);
}

.thinking-icon {
  font-size: 18px;
}

.animated-ellipsis span {
  animation: ellipsis-fade 1.4s infinite;
  opacity: 0;
}

.animated-ellipsis span:nth-child(1) { animation-delay: 0s; }
.animated-ellipsis span:nth-child(2) { animation-delay: 0.2s; }
.animated-ellipsis span:nth-child(3) { animation-delay: 0.4s; }

@keyframes ellipsis-fade {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}
```

### Rotating Status Messages

```jsx
function RotatingStatus({ messages = [
  "Thinking...",
  "Analyzing your question...",
  "Generating response..."
], interval = 3000 }) {
  const [index, setIndex] = useState(0);
  
  useEffect(() => {
    const timer = setInterval(() => {
      setIndex(prev => (prev + 1) % messages.length);
    }, interval);
    
    return () => clearInterval(timer);
  }, [messages.length, interval]);
  
  return (
    <div className="rotating-status" aria-live="polite">
      <span className="status-text">{messages[index]}</span>
    </div>
  );
}
```

```css
.rotating-status {
  padding: 12px 16px;
  background: var(--message-bg, #f5f5f5);
  border-radius: 12px;
}

.status-text {
  animation: fade-swap 0.3s ease-out;
}

@keyframes fade-swap {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### Contextual Status

```jsx
function ContextualStatus({ query }) {
  const getMessage = () => {
    if (query.toLowerCase().includes('code')) {
      return "Writing code...";
    }
    if (query.toLowerCase().includes('explain')) {
      return "Preparing explanation...";
    }
    if (query.length > 200) {
      return "Processing your detailed request...";
    }
    return "Generating response...";
  };
  
  return (
    <div className="contextual-status">
      <span className="spinner"></span>
      <span>{getMessage()}</span>
    </div>
  );
}
```

---

## Skeleton Text Preview

### Basic Skeleton Lines

```jsx
function SkeletonMessage({ lines = 3 }) {
  return (
    <div className="skeleton-message" aria-label="Loading response">
      {Array.from({ length: lines }).map((_, i) => (
        <div 
          key={i}
          className="skeleton-line"
          style={{ width: `${70 + Math.random() * 30}%` }}
        />
      ))}
    </div>
  );
}
```

```css
.skeleton-message {
  padding: 16px;
  background: var(--message-bg, #f5f5f5);
  border-radius: 12px;
}

.skeleton-line {
  height: 14px;
  background: linear-gradient(
    90deg,
    #e0e0e0 25%,
    #f0f0f0 50%,
    #e0e0e0 75%
  );
  background-size: 200% 100%;
  border-radius: 4px;
  margin-bottom: 8px;
  animation: shimmer 1.5s infinite;
}

.skeleton-line:last-child {
  margin-bottom: 0;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Skeleton with Avatar

```jsx
function SkeletonMessageWithAvatar() {
  return (
    <div className="message-skeleton" aria-hidden="true">
      <div className="skeleton-avatar"></div>
      <div className="skeleton-content">
        <div className="skeleton-line" style={{ width: '40%' }}></div>
        <div className="skeleton-line" style={{ width: '80%' }}></div>
        <div className="skeleton-line" style={{ width: '60%' }}></div>
      </div>
    </div>
  );
}
```

```css
.message-skeleton {
  display: flex;
  gap: 12px;
  padding: 16px;
}

.skeleton-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(
    90deg,
    #e0e0e0 25%,
    #f0f0f0 50%,
    #e0e0e0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  flex-shrink: 0;
}

.skeleton-content {
  flex: 1;
}
```

### Transition to Real Content

```jsx
function AnimatedTransition({ isLoading, children }) {
  return (
    <div className="transition-container">
      {isLoading ? (
        <div className="skeleton-wrapper fade-out">
          <SkeletonMessage lines={3} />
        </div>
      ) : (
        <div className="content-wrapper fade-in">
          {children}
        </div>
      )}
    </div>
  );
}
```

```css
.fade-in {
  animation: fade-in 0.3s ease-out;
}

.fade-out {
  animation: fade-out 0.2s ease-out forwards;
}

@keyframes fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fade-out {
  from { opacity: 1; }
  to { opacity: 0; }
}
```

---

## React Implementation

### Complete Typing Indicator Component

```jsx
function TypingIndicator({ 
  variant = 'dots',  // 'dots' | 'pulse' | 'text' | 'skeleton'
  message = 'AI is thinking',
  skeletonLines = 3
}) {
  switch (variant) {
    case 'dots':
      return (
        <div className="typing-indicator dots" role="status" aria-label={message}>
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="sr-only">{message}</span>
        </div>
      );
      
    case 'pulse':
      return (
        <div className="typing-indicator pulse" role="status" aria-label={message}>
          <PulsingIndicator />
          <span className="sr-only">{message}</span>
        </div>
      );
      
    case 'text':
      return (
        <ThinkingMessage text={message} />
      );
      
    case 'skeleton':
      return (
        <SkeletonMessage lines={skeletonLines} />
      );
      
    default:
      return null;
  }
}
```

### With Chat Integration

```jsx
function ChatMessages({ messages, status }) {
  return (
    <div className="messages">
      {messages.map(msg => (
        <Message key={msg.id} {...msg} />
      ))}
      
      {status === 'submitted' && (
        <div className="message assistant">
          <div className="message-avatar">🤖</div>
          <TypingIndicator variant="dots" />
        </div>
      )}
    </div>
  );
}
```

### Customizable Hook

```jsx
function useTypingIndicator(status, options = {}) {
  const { delay = 0, variant = 'dots' } = options;
  const [showIndicator, setShowIndicator] = useState(false);
  
  useEffect(() => {
    let timeout;
    
    if (status === 'submitted') {
      // Optional delay before showing indicator
      timeout = setTimeout(() => {
        setShowIndicator(true);
      }, delay);
    } else {
      setShowIndicator(false);
    }
    
    return () => clearTimeout(timeout);
  }, [status, delay]);
  
  return {
    showIndicator,
    variant,
    IndicatorComponent: showIndicator ? (
      <TypingIndicator variant={variant} />
    ) : null
  };
}
```

---

## Accessibility

### Screen Reader Support

```jsx
function AccessibleTypingIndicator() {
  return (
    <div 
      className="typing-indicator"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {/* Visual indicator hidden from screen readers */}
      <span aria-hidden="true" className="dots">
        <span className="dot"></span>
        <span className="dot"></span>
        <span className="dot"></span>
      </span>
      
      {/* Screen reader text */}
      <span className="sr-only">AI is typing a response</span>
    </div>
  );
}
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  .dot,
  .pulse-ring,
  .skeleton-line {
    animation: none;
  }
  
  .dot {
    opacity: 0.6;
  }
  
  .dot:nth-child(2) {
    opacity: 0.8;
  }
  
  .dot:nth-child(3) {
    opacity: 1;
  }
}
```

### Focus Management

```jsx
function ChatWithIndicator({ messages, status }) {
  const indicatorRef = useRef(null);
  
  useEffect(() => {
    if (status === 'submitted' && indicatorRef.current) {
      // Announce to screen readers without moving focus
      indicatorRef.current.setAttribute('aria-live', 'polite');
    }
  }, [status]);
  
  return (
    <div className="messages">
      {messages.map(msg => <Message key={msg.id} {...msg} />)}
      
      {status === 'submitted' && (
        <div ref={indicatorRef} className="typing-wrapper">
          <TypingIndicator />
        </div>
      )}
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show indicator immediately on submit | Delay indicator start |
| Hide smoothly when content arrives | Abruptly remove indicator |
| Provide screen reader text | Leave indicator unlabeled |
| Support reduced motion | Force animations on all users |
| Match indicator to message style | Use inconsistent styling |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Indicator persists during streaming | Hide when status changes to 'streaming' |
| Animation causes performance issues | Use CSS animations, not JavaScript |
| No feedback for slow responses | Add rotating status after 3 seconds |
| Indicator layout shifts content | Reserve space with consistent sizing |
| Jarring transition to content | Animate fade between states |

---

## Hands-on Exercise

### Your Task

Create a multi-variant typing indicator that:
1. Shows bouncing dots for normal responses
2. Shows skeleton for long responses (> 3 seconds)
3. Supports reduced motion preference
4. Includes proper accessibility labels

### Requirements

1. Create three dot bounce animation
2. Add shimmer skeleton fallback
3. Use `prefers-reduced-motion` media query
4. Include `role="status"` and `aria-label`

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `useState` to track time since submission
- Switch to skeleton after timeout
- Use CSS for animation, not JavaScript
- Hide decorative elements with `aria-hidden`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
function AdaptiveTypingIndicator({ status }) {
  const [showSkeleton, setShowSkeleton] = useState(false);
  
  useEffect(() => {
    if (status === 'submitted') {
      const timer = setTimeout(() => {
        setShowSkeleton(true);
      }, 3000);
      
      return () => clearTimeout(timer);
    } else {
      setShowSkeleton(false);
    }
  }, [status]);
  
  if (status !== 'submitted') return null;
  
  return (
    <div role="status" aria-label="AI is generating a response">
      {showSkeleton ? (
        <SkeletonMessage lines={3} />
      ) : (
        <div className="typing-dots" aria-hidden="true">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </div>
      )}
      <span className="sr-only">AI is generating a response</span>
    </div>
  );
}
```

</details>

---

## Summary

✅ **Bouncing dots** are the classic typing indicator  
✅ **Pulsing effects** add modern feel  
✅ **Text messages** provide context ("Thinking...")  
✅ **Skeleton previews** work for longer waits  
✅ **Accessibility** requires labels and reduced motion  
✅ **Smooth transitions** improve perceived performance

---

## Further Reading

- [CSS Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/animation)
- [ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)
- [prefers-reduced-motion](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)

---

**Previous:** [Loading & Status Indicators Overview](./00-loading-status-indicators.md)  
**Next:** [Skeleton Loading States](./02-skeleton-loading-states.md)

<!-- 
Sources Consulted:
- MDN CSS Animations: https://developer.mozilla.org/en-US/docs/Web/CSS/animation
- MDN ARIA Live Regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- web.dev loading patterns: https://web.dev/patterns/
-->
