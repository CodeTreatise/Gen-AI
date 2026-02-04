---
title: "Skeleton Loading States"
---

# Skeleton Loading States

## Introduction

Skeleton screens show placeholder shapes that match expected content, reducing perceived load time. Unlike spinners that say "wait," skeletons say "content is coming in this shape"—setting accurate expectations.

In this lesson, we'll build skeleton loaders specifically designed for chat interfaces.

### What We'll Cover

- Message placeholder shapes
- Animated shimmer effects
- Realistic content shapes
- Smooth transitions to real content
- Performance considerations

### Prerequisites

- [Typing Indicators](./01-typing-indicators.md)
- CSS gradients and animations
- React component patterns

---

## Message Placeholder Shapes

### Basic Message Skeleton

```jsx
function MessageSkeleton({ lines = 3, hasAvatar = true }) {
  return (
    <div className="message-skeleton" aria-hidden="true">
      {hasAvatar && <div className="skeleton-avatar" />}
      <div className="skeleton-body">
        {Array.from({ length: lines }).map((_, i) => (
          <div 
            key={i}
            className="skeleton-line"
            style={{ width: getLineWidth(i, lines) }}
          />
        ))}
      </div>
    </div>
  );
}

function getLineWidth(index, total) {
  // First line shorter (like a greeting)
  if (index === 0) return '40%';
  // Last line shorter (trailing off)
  if (index === total - 1) return '60%';
  // Middle lines varied
  return `${70 + Math.random() * 25}%`;
}
```

```css
.message-skeleton {
  display: flex;
  gap: 12px;
  padding: 16px;
  max-width: 80%;
}

.skeleton-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--skeleton-bg, #e0e0e0);
  flex-shrink: 0;
}

.skeleton-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-line {
  height: 14px;
  background: var(--skeleton-bg, #e0e0e0);
  border-radius: 4px;
}
```

### Code Block Skeleton

```jsx
function CodeBlockSkeleton({ lines = 5 }) {
  return (
    <div className="code-skeleton">
      <div className="code-skeleton-header">
        <span className="skeleton-dot"></span>
        <span className="skeleton-dot"></span>
        <span className="skeleton-dot"></span>
      </div>
      <div className="code-skeleton-body">
        {Array.from({ length: lines }).map((_, i) => (
          <div 
            key={i}
            className="skeleton-code-line"
            style={{ 
              width: `${30 + Math.random() * 60}%`,
              marginLeft: getIndent(i)
            }}
          />
        ))}
      </div>
    </div>
  );
}

function getIndent(index) {
  // Simulate code indentation
  const indentLevel = index === 0 ? 0 : Math.floor(Math.random() * 3);
  return `${indentLevel * 16}px`;
}
```

```css
.code-skeleton {
  background: #1e1e1e;
  border-radius: 8px;
  overflow: hidden;
}

.code-skeleton-header {
  display: flex;
  gap: 6px;
  padding: 12px;
  background: #2d2d2d;
}

.skeleton-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #555;
}

.code-skeleton-body {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-code-line {
  height: 12px;
  background: #333;
  border-radius: 2px;
}
```

### List Item Skeleton

```jsx
function ListSkeleton({ items = 3 }) {
  return (
    <div className="list-skeleton">
      {Array.from({ length: items }).map((_, i) => (
        <div key={i} className="list-item-skeleton">
          <div className="skeleton-bullet"></div>
          <div className="skeleton-line" style={{ width: `${60 + Math.random() * 30}%` }}></div>
        </div>
      ))}
    </div>
  );
}
```

```css
.list-skeleton {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 8px 0;
}

.list-item-skeleton {
  display: flex;
  align-items: center;
  gap: 8px;
}

.skeleton-bullet {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--skeleton-bg, #e0e0e0);
  flex-shrink: 0;
}
```

---

## Animated Shimmer Effects

### CSS Gradient Shimmer

```css
.skeleton-shimmer {
  background: linear-gradient(
    90deg,
    var(--skeleton-bg, #e0e0e0) 0%,
    var(--skeleton-highlight, #f5f5f5) 50%,
    var(--skeleton-bg, #e0e0e0) 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
```

### Wave Shimmer

```css
.skeleton-wave {
  position: relative;
  overflow: hidden;
  background: var(--skeleton-bg, #e0e0e0);
}

.skeleton-wave::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.4) 50%,
    transparent 100%
  );
  animation: wave-shimmer 1.5s infinite;
}

@keyframes wave-shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}
```

### Pulse Shimmer

```css
.skeleton-pulse {
  background: var(--skeleton-bg, #e0e0e0);
  animation: pulse-shimmer 1.5s ease-in-out infinite;
}

@keyframes pulse-shimmer {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
```

### Shimmer Component

```jsx
function ShimmerEffect({ children, type = 'wave' }) {
  return (
    <div className={`skeleton-container skeleton-${type}`}>
      {children}
    </div>
  );
}

// Usage
<ShimmerEffect type="wave">
  <MessageSkeleton lines={3} />
</ShimmerEffect>
```

---

## Realistic Content Shapes

### Dynamic Line Widths

```jsx
function RealisticMessageSkeleton({ estimatedLength = 'medium' }) {
  const patterns = {
    short: [
      { width: '30%' },
      { width: '45%' }
    ],
    medium: [
      { width: '40%' },
      { width: '85%' },
      { width: '70%' },
      { width: '55%' }
    ],
    long: [
      { width: '35%' },
      { width: '90%' },
      { width: '80%' },
      { width: '95%' },
      { width: '75%' },
      { width: '60%' }
    ]
  };
  
  const lines = patterns[estimatedLength] || patterns.medium;
  
  return (
    <div className="realistic-skeleton">
      {lines.map((line, i) => (
        <div 
          key={i}
          className="skeleton-line skeleton-shimmer"
          style={{ width: line.width }}
        />
      ))}
    </div>
  );
}
```

### Mixed Content Skeleton

```jsx
function MixedContentSkeleton() {
  return (
    <div className="mixed-skeleton">
      {/* Text paragraph */}
      <div className="skeleton-paragraph">
        <div className="skeleton-line skeleton-shimmer" style={{ width: '90%' }} />
        <div className="skeleton-line skeleton-shimmer" style={{ width: '75%' }} />
        <div className="skeleton-line skeleton-shimmer" style={{ width: '80%' }} />
      </div>
      
      {/* Code block */}
      <div className="skeleton-code-block">
        <CodeBlockSkeleton lines={4} />
      </div>
      
      {/* Another paragraph */}
      <div className="skeleton-paragraph">
        <div className="skeleton-line skeleton-shimmer" style={{ width: '60%' }} />
        <div className="skeleton-line skeleton-shimmer" style={{ width: '70%' }} />
      </div>
    </div>
  );
}
```

```css
.mixed-skeleton {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.skeleton-paragraph {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-code-block {
  margin: 8px 0;
}
```

### Chat Bubble Shape

```jsx
function ChatBubbleSkeleton({ isUser = false }) {
  return (
    <div className={`bubble-skeleton ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-skeleton-content">
        <div className="skeleton-line skeleton-shimmer" style={{ width: '80%' }} />
        <div className="skeleton-line skeleton-shimmer" style={{ width: '60%' }} />
      </div>
      <div className="bubble-skeleton-tail" />
    </div>
  );
}
```

```css
.bubble-skeleton {
  position: relative;
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 18px;
  background: var(--skeleton-bg, #e0e0e0);
}

.bubble-skeleton.user {
  margin-left: auto;
  border-bottom-right-radius: 4px;
}

.bubble-skeleton.assistant {
  border-bottom-left-radius: 4px;
}

.bubble-skeleton-content {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bubble-skeleton .skeleton-line {
  height: 12px;
  background: rgba(255, 255, 255, 0.3);
}
```

---

## Transition to Content

### Fade Transition

```jsx
function FadeTransition({ isLoading, skeleton, children }) {
  return (
    <div className="fade-transition">
      <div className={`skeleton-layer ${isLoading ? 'visible' : 'hidden'}`}>
        {skeleton}
      </div>
      <div className={`content-layer ${isLoading ? 'hidden' : 'visible'}`}>
        {children}
      </div>
    </div>
  );
}
```

```css
.fade-transition {
  position: relative;
}

.skeleton-layer,
.content-layer {
  transition: opacity 0.3s ease-out;
}

.skeleton-layer.hidden,
.content-layer.hidden {
  opacity: 0;
  position: absolute;
  pointer-events: none;
}

.skeleton-layer.visible,
.content-layer.visible {
  opacity: 1;
}
```

### Crossfade Transition

```jsx
function CrossfadeTransition({ isLoading, skeleton, children }) {
  const [showSkeleton, setShowSkeleton] = useState(isLoading);
  
  useEffect(() => {
    if (!isLoading) {
      // Small delay to allow content to render before hiding skeleton
      const timer = setTimeout(() => {
        setShowSkeleton(false);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setShowSkeleton(true);
    }
  }, [isLoading]);
  
  return (
    <div className="crossfade-container">
      {showSkeleton && (
        <div className={`skeleton-wrapper ${isLoading ? '' : 'fading-out'}`}>
          {skeleton}
        </div>
      )}
      <div className={`content-wrapper ${isLoading ? 'hidden' : 'fading-in'}`}>
        {children}
      </div>
    </div>
  );
}
```

```css
.crossfade-container {
  position: relative;
}

.skeleton-wrapper {
  transition: opacity 0.2s ease-out;
}

.skeleton-wrapper.fading-out {
  opacity: 0;
}

.content-wrapper {
  transition: opacity 0.3s ease-out;
}

.content-wrapper.hidden {
  opacity: 0;
}

.content-wrapper.fading-in {
  opacity: 1;
}
```

### Progressive Reveal

```jsx
function ProgressiveReveal({ content, isComplete }) {
  const [revealedLength, setRevealedLength] = useState(0);
  
  useEffect(() => {
    if (isComplete) {
      setRevealedLength(content.length);
      return;
    }
    
    // Gradually reveal more content
    const interval = setInterval(() => {
      setRevealedLength(prev => {
        if (prev >= content.length) {
          clearInterval(interval);
          return prev;
        }
        return Math.min(prev + 10, content.length);
      });
    }, 50);
    
    return () => clearInterval(interval);
  }, [content, isComplete]);
  
  const revealed = content.slice(0, revealedLength);
  const remaining = content.length - revealedLength;
  
  return (
    <div className="progressive-reveal">
      <span className="revealed-content">{revealed}</span>
      {remaining > 0 && (
        <span className="skeleton-inline">
          {Array.from({ length: Math.ceil(remaining / 50) }).map((_, i) => (
            <span key={i} className="skeleton-word skeleton-shimmer" />
          ))}
        </span>
      )}
    </div>
  );
}
```

---

## React Implementation

### Configurable Skeleton System

```jsx
const SkeletonContext = createContext({
  animation: 'shimmer',
  speed: 1.5
});

function SkeletonProvider({ children, animation = 'shimmer', speed = 1.5 }) {
  return (
    <SkeletonContext.Provider value={{ animation, speed }}>
      {children}
    </SkeletonContext.Provider>
  );
}

function Skeleton({ width, height, borderRadius, className }) {
  const { animation, speed } = useContext(SkeletonContext);
  
  return (
    <div
      className={`skeleton skeleton-${animation} ${className || ''}`}
      style={{
        width,
        height,
        borderRadius,
        '--animation-speed': `${speed}s`
      }}
    />
  );
}
```

### Chat Skeleton Hook

```jsx
function useChatSkeleton(status, options = {}) {
  const { delay = 0, estimatedLength = 'medium' } = options;
  const [showSkeleton, setShowSkeleton] = useState(false);
  
  useEffect(() => {
    let timeout;
    
    if (status === 'submitted') {
      timeout = setTimeout(() => {
        setShowSkeleton(true);
      }, delay);
    } else {
      setShowSkeleton(false);
    }
    
    return () => clearTimeout(timeout);
  }, [status, delay]);
  
  return {
    showSkeleton,
    SkeletonComponent: showSkeleton ? (
      <RealisticMessageSkeleton estimatedLength={estimatedLength} />
    ) : null
  };
}
```

### Complete Message with Skeleton

```jsx
function MessageWithSkeleton({ message, status }) {
  const isCurrentlyStreaming = status === 'streaming' && message.role === 'assistant';
  const isLoading = status === 'submitted' && !message.content;
  
  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === 'user' ? '👤' : '🤖'}
      </div>
      
      <div className="message-content">
        <CrossfadeTransition
          isLoading={isLoading}
          skeleton={<MessageSkeleton lines={3} hasAvatar={false} />}
        >
          <div className="message-text">
            {message.content}
            {isCurrentlyStreaming && <span className="cursor">▋</span>}
          </div>
        </CrossfadeTransition>
      </div>
    </div>
  );
}
```

---

## Performance Optimization

### CSS-Only Skeletons

```css
/* Use CSS custom properties for efficiency */
.skeleton {
  --skeleton-bg: #e0e0e0;
  --skeleton-highlight: #f5f5f5;
  
  background: linear-gradient(
    90deg,
    var(--skeleton-bg) 0%,
    var(--skeleton-highlight) 50%,
    var(--skeleton-bg) 100%
  );
  background-size: 200% 100%;
  animation: shimmer var(--animation-speed, 1.5s) infinite;
}

/* GPU acceleration */
.skeleton {
  will-change: background-position;
  transform: translateZ(0);
}
```

### Lazy Skeleton Rendering

```jsx
function LazyMessageSkeleton({ count = 3 }) {
  const [visible, setVisible] = useState(1);
  
  useEffect(() => {
    // Gradually reveal more skeletons
    if (visible < count) {
      const timer = setTimeout(() => {
        setVisible(prev => Math.min(prev + 1, count));
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [visible, count]);
  
  return (
    <div className="skeleton-list">
      {Array.from({ length: visible }).map((_, i) => (
        <MessageSkeleton key={i} lines={3} />
      ))}
    </div>
  );
}
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  .skeleton,
  .skeleton-shimmer,
  .skeleton-wave,
  .skeleton-pulse {
    animation: none;
    background: var(--skeleton-bg, #e0e0e0);
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Match skeleton to expected content shape | Use generic rectangles |
| Use subtle shimmer animation | Use distracting animations |
| Transition smoothly to content | Abruptly swap skeleton for content |
| Support reduced motion | Force animations |
| Use CSS for animations | Use JavaScript for shimmer |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Skeleton looks nothing like content | Study actual message shapes |
| Flash of skeleton on fast loads | Add small delay before showing |
| Layout shift when content loads | Match skeleton dimensions exactly |
| Shimmer causes performance issues | Use `will-change` and GPU layers |
| Skeleton stays after content loads | Properly track loading state |

---

## Hands-on Exercise

### Your Task

Create a message skeleton system that:
1. Shows realistic chat bubble shapes
2. Animates with smooth shimmer
3. Transitions cleanly to real content
4. Supports different message lengths

### Requirements

1. Avatar + message body skeleton
2. Wave shimmer animation
3. Fade transition to content
4. Three size variants (short, medium, long)

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `::after` pseudo-element for wave shimmer
- Track `isLoading` state for transitions
- Use CSS Grid for layout stability
- Pre-define line patterns for each size

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
function ChatMessageSkeleton({ size = 'medium' }) {
  const linePatterns = {
    short: ['40%', '55%'],
    medium: ['35%', '80%', '65%'],
    long: ['30%', '90%', '85%', '70%', '50%']
  };
  
  return (
    <div className="chat-skeleton">
      <div className="skeleton-avatar skeleton-wave" />
      <div className="skeleton-content">
        {linePatterns[size].map((width, i) => (
          <div
            key={i}
            className="skeleton-line skeleton-wave"
            style={{ width }}
          />
        ))}
      </div>
    </div>
  );
}
```

</details>

---

## Summary

✅ **Shape skeletons** to match expected content  
✅ **Shimmer animations** add life without distraction  
✅ **Realistic patterns** set accurate expectations  
✅ **Smooth transitions** prevent jarring changes  
✅ **CSS animations** perform better than JavaScript  
✅ **Reduced motion** respects user preferences

---

## Further Reading

- [CSS Gradients](https://developer.mozilla.org/en-US/docs/Web/CSS/gradient)
- [will-change Property](https://developer.mozilla.org/en-US/docs/Web/CSS/will-change)
- [Skeleton Loading Pattern](https://uxdesign.cc/what-you-should-know-about-skeleton-screens-a820c45a571a)

---

**Previous:** [Typing Indicators](./01-typing-indicators.md)  
**Next:** [Progress Indicators](./03-progress-indicators.md)

<!-- 
Sources Consulted:
- MDN CSS Gradients: https://developer.mozilla.org/en-US/docs/Web/CSS/gradient
- MDN will-change: https://developer.mozilla.org/en-US/docs/Web/CSS/will-change
- Skeleton loading patterns: https://web.dev/patterns/
-->
