---
title: "Text Appearance Animations"
---

# Text Appearance Animations

## Introduction

How text appears on screen affects perceived quality. Abrupt text insertion feels jarring; smooth animations feel polished. We'll implement fade-in, typewriter, and other effects that enhance streaming without sacrificing performance.

In this lesson, we'll create various text appearance animations and learn when to use each.

### What We'll Cover

- Fade-in effects for streamed text
- Word and character animations
- Typewriter with cursor effects
- CSS transitions for text reveal
- Performance-optimized animations

### Prerequisites

- [Character vs Chunk Display](./02-character-vs-chunk-display.md)
- CSS animations and keyframes
- CSS transforms and transitions

---

## Fade-In Effects

### Word-by-Word Fade

```css
.fade-word {
  display: inline;
  animation: word-fade-in 0.3s ease-out forwards;
  opacity: 0;
}

@keyframes word-fade-in {
  from { 
    opacity: 0;
    transform: translateY(4px);
  }
  to { 
    opacity: 1;
    transform: translateY(0);
  }
}
```

```javascript
class FadeInRenderer {
  constructor(container) {
    this.container = container;
    this.buffer = '';
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.renderWords();
  }
  
  renderWords() {
    // Find complete words
    const words = this.buffer.split(/(\s+)/);
    const lastWord = words[words.length - 1];
    
    // Keep incomplete word in buffer
    if (lastWord && !lastWord.match(/\s$/)) {
      this.buffer = words.pop();
    } else {
      this.buffer = '';
    }
    
    // Render complete words with animation
    words.forEach((word, i) => {
      if (word) {
        const span = document.createElement('span');
        span.className = 'fade-word';
        span.textContent = word;
        span.style.animationDelay = `${i * 50}ms`;
        this.container.appendChild(span);
      }
    });
  }
  
  complete() {
    // Render remaining buffer
    if (this.buffer) {
      const span = document.createElement('span');
      span.className = 'fade-word';
      span.textContent = this.buffer;
      this.container.appendChild(span);
      this.buffer = '';
    }
  }
}
```

### Chunk Fade-In

For chunk-based rendering with fade effect:

```css
.fade-chunk {
  display: inline;
  animation: chunk-appear 0.2s ease-out forwards;
  opacity: 0;
}

@keyframes chunk-appear {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

```javascript
class ChunkFadeRenderer {
  constructor(container) {
    this.container = container;
    this.lastSpan = null;
  }
  
  append(chunk) {
    // If we have a previous span, remove its animation class
    if (this.lastSpan) {
      this.lastSpan.style.opacity = '1';
      this.lastSpan.classList.remove('fade-chunk');
    }
    
    // Create new span for this chunk
    const span = document.createElement('span');
    span.className = 'fade-chunk';
    span.textContent = chunk;
    this.container.appendChild(span);
    
    this.lastSpan = span;
  }
  
  complete() {
    if (this.lastSpan) {
      this.lastSpan.style.opacity = '1';
      this.lastSpan.classList.remove('fade-chunk');
    }
  }
}
```

---

## Slide-Up Animation

Text slides up as it appears:

```css
.slide-up-text {
  display: inline;
  animation: slide-up 0.25s ease-out forwards;
}

@keyframes slide-up {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### Staggered Slide Animation

```javascript
class StaggeredSlideRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.staggerDelay = options.staggerDelay || 30;
    this.wordIndex = 0;
    this.buffer = '';
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.renderWords();
  }
  
  renderWords() {
    const parts = this.buffer.split(/(\s+)/);
    
    // Keep incomplete word in buffer
    if (!this.buffer.endsWith(' ') && parts.length > 0) {
      this.buffer = parts.pop();
    } else {
      this.buffer = '';
    }
    
    parts.forEach(part => {
      if (!part) return;
      
      const span = document.createElement('span');
      span.className = 'slide-up-text';
      span.textContent = part;
      span.style.animationDelay = `${this.wordIndex * this.staggerDelay}ms`;
      this.container.appendChild(span);
      
      // Only increment for actual words, not whitespace
      if (part.trim()) {
        this.wordIndex++;
      }
    });
  }
  
  complete() {
    if (this.buffer) {
      const span = document.createElement('span');
      span.className = 'slide-up-text';
      span.textContent = this.buffer;
      this.container.appendChild(span);
      this.buffer = '';
    }
  }
}
```

---

## Typewriter with Effects

### Classic Typewriter

```css
.typewriter-container {
  font-family: 'Courier New', monospace;
}

.typewriter-char {
  display: inline;
  animation: type-appear 0.1s ease-out forwards;
  opacity: 0;
}

@keyframes type-appear {
  from { opacity: 0; }
  to { opacity: 1; }
}

.typewriter-cursor {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  background: currentColor;
  margin-left: 1px;
  animation: cursor-blink 0.8s step-end infinite;
  vertical-align: text-bottom;
}

@keyframes cursor-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
```

### Typewriter with Sound (Optional)

```javascript
class TypewriterWithSound {
  constructor(container, options = {}) {
    this.container = container;
    this.speed = options.speed || 50;
    this.soundEnabled = options.sound || false;
    
    this.queue = '';
    this.isTyping = false;
    
    // Preload sound
    if (this.soundEnabled) {
      this.keySound = new Audio('/sounds/key-press.mp3');
      this.keySound.volume = 0.1;
    }
    
    this.content = document.createElement('span');
    this.cursor = document.createElement('span');
    this.cursor.className = 'typewriter-cursor';
    
    container.appendChild(this.content);
    container.appendChild(this.cursor);
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
      
      // Create animated character
      const charSpan = document.createElement('span');
      charSpan.className = 'typewriter-char';
      charSpan.textContent = char;
      this.content.appendChild(charSpan);
      
      // Play sound (if enabled and not whitespace)
      if (this.soundEnabled && char.trim()) {
        this.playKeySound();
      }
      
      // Variable delay for natural feel
      const delay = this.getDelay(char);
      await this.wait(delay);
    }
    
    this.isTyping = false;
  }
  
  getDelay(char) {
    if ('.!?'.includes(char)) return this.speed * 4;
    if (',;:'.includes(char)) return this.speed * 2;
    return this.speed + (Math.random() - 0.5) * this.speed * 0.5;
  }
  
  playKeySound() {
    const sound = this.keySound.cloneNode();
    sound.volume = 0.05 + Math.random() * 0.1;
    sound.playbackRate = 0.9 + Math.random() * 0.2;
    sound.play().catch(() => {});  // Ignore autoplay errors
  }
  
  wait(ms) {
    return new Promise(r => setTimeout(r, ms));
  }
  
  complete() {
    this.content.textContent += this.queue;
    this.queue = '';
    this.cursor.remove();
    this.isTyping = false;
  }
}
```

---

## Gradient Reveal Effect

Text appears with a color gradient sweep:

```css
.gradient-reveal {
  background: linear-gradient(
    90deg,
    currentColor 0%,
    currentColor var(--reveal-progress),
    transparent var(--reveal-progress),
    transparent 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: reveal-sweep 0.5s ease-out forwards;
}

@keyframes reveal-sweep {
  from { --reveal-progress: 0%; }
  to { --reveal-progress: 100%; }
}
```

> **Note:** CSS custom properties in `@keyframes` require the `@property` rule for animation.

### With @property (Modern Browsers)

```css
@property --reveal-progress {
  syntax: '<percentage>';
  initial-value: 0%;
  inherits: false;
}

.gradient-reveal {
  --reveal-progress: 0%;
  background: linear-gradient(
    90deg,
    currentColor 0%,
    currentColor var(--reveal-progress),
    rgba(0, 0, 0, 0.1) var(--reveal-progress),
    rgba(0, 0, 0, 0.1) 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: reveal-sweep 0.8s ease-out forwards;
}

@keyframes reveal-sweep {
  to { --reveal-progress: 100%; }
}
```

---

## React Implementations

### Animated Word Component

```jsx
function AnimatedWord({ word, delay }) {
  return (
    <span 
      className="animated-word"
      style={{ animationDelay: `${delay}ms` }}
    >
      {word}
    </span>
  );
}

function FadeInMessage({ content, isStreaming }) {
  const [words, setWords] = useState([]);
  const prevContentRef = useRef('');
  
  useEffect(() => {
    const newContent = content.slice(prevContentRef.current.length);
    prevContentRef.current = content;
    
    if (newContent) {
      const newWords = newContent.split(/(\s+)/).filter(w => w);
      setWords(prev => [
        ...prev,
        ...newWords.map((word, i) => ({
          text: word,
          key: `${Date.now()}-${i}`,
          delay: i * 30
        }))
      ]);
    }
  }, [content]);
  
  return (
    <div className="message-body">
      {words.map(({ text, key, delay }) => (
        <AnimatedWord key={key} word={text} delay={delay} />
      ))}
    </div>
  );
}
```

```css
.animated-word {
  display: inline;
  animation: word-appear 0.3s ease-out forwards;
  opacity: 0;
}

@keyframes word-appear {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### Typewriter Hook

```jsx
function useTypewriter(content, speed = 30) {
  const [displayed, setDisplayed] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  useEffect(() => {
    if (displayed.length >= content.length) {
      setIsTyping(false);
      return;
    }
    
    setIsTyping(true);
    
    const timeout = setTimeout(() => {
      setDisplayed(content.slice(0, displayed.length + 1));
    }, speed + (Math.random() - 0.5) * speed * 0.5);
    
    return () => clearTimeout(timeout);
  }, [displayed, content, speed]);
  
  return { displayed, isTyping };
}

function TypewriterMessage({ content }) {
  const { displayed, isTyping } = useTypewriter(content, 25);
  
  return (
    <div className="message-body typewriter">
      {displayed}
      {isTyping && <span className="cursor">▋</span>}
    </div>
  );
}
```

### Configurable Animation Component

```jsx
function StreamingText({ 
  content, 
  isStreaming,
  animation = 'fade',  // 'fade' | 'slide' | 'typewriter' | 'none'
  speed = 'normal'     // 'slow' | 'normal' | 'fast'
}) {
  const speedMap = {
    slow: { delay: 60, stagger: 50 },
    normal: { delay: 30, stagger: 30 },
    fast: { delay: 15, stagger: 15 }
  };
  
  const settings = speedMap[speed];
  
  if (animation === 'none' || !isStreaming) {
    return <span>{content}</span>;
  }
  
  if (animation === 'typewriter') {
    return <TypewriterMessage content={content} speed={settings.delay} />;
  }
  
  // Word-based animations
  const words = content.split(/(\s+)/);
  
  return (
    <span className={`streaming-${animation}`}>
      {words.map((word, i) => (
        <span
          key={i}
          className={`word-${animation}`}
          style={{ animationDelay: `${i * settings.stagger}ms` }}
        >
          {word}
        </span>
      ))}
    </span>
  );
}
```

---

## Performance Considerations

### GPU-Accelerated Properties

```css
/* ✅ Good: GPU-accelerated */
.animate-good {
  transform: translateY(4px);
  opacity: 0;
  animation: appear-good 0.3s ease-out forwards;
}

@keyframes appear-good {
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

/* ❌ Bad: Triggers layout */
.animate-bad {
  margin-top: 4px;
  animation: appear-bad 0.3s ease-out forwards;
}

@keyframes appear-bad {
  to {
    margin-top: 0;
  }
}
```

### Reducing Animation Count

```javascript
class BatchAnimatedRenderer {
  constructor(container, options = {}) {
    this.container = container;
    this.batchSize = options.batchSize || 10;  // Words per animation group
    this.buffer = '';
    this.wordCount = 0;
    this.currentBatch = null;
  }
  
  append(chunk) {
    this.buffer += chunk;
    this.renderWords();
  }
  
  renderWords() {
    const words = this.buffer.split(/(\s+)/);
    
    // Keep incomplete word
    if (!this.buffer.endsWith(' ')) {
      this.buffer = words.pop() || '';
    } else {
      this.buffer = '';
    }
    
    words.forEach(word => {
      if (!word) return;
      
      // Create new batch if needed
      if (!this.currentBatch || this.wordCount % this.batchSize === 0) {
        this.currentBatch = document.createElement('span');
        this.currentBatch.className = 'word-batch';
        this.container.appendChild(this.currentBatch);
      }
      
      // Add word to current batch (no individual animation)
      const span = document.createElement('span');
      span.textContent = word;
      this.currentBatch.appendChild(span);
      
      if (word.trim()) {
        this.wordCount++;
      }
    });
  }
}
```

```css
.word-batch {
  display: inline;
  animation: batch-appear 0.2s ease-out forwards;
  opacity: 0;
}

@keyframes batch-appear {
  to { opacity: 1; }
}
```

### will-change Optimization

```css
.animated-word {
  will-change: transform, opacity;
  animation: word-appear 0.3s ease-out forwards;
}

/* Remove will-change after animation */
.animated-word.animation-complete {
  will-change: auto;
}
```

```javascript
// Remove will-change after animation
element.addEventListener('animationend', () => {
  element.classList.add('animation-complete');
});
```

---

## Reduced Motion Support

```css
/* Default animations */
.fade-word {
  animation: word-fade-in 0.3s ease-out forwards;
  opacity: 0;
}

/* Reduced motion: instant appearance */
@media (prefers-reduced-motion: reduce) {
  .fade-word,
  .slide-up-text,
  .typewriter-char {
    animation: none !important;
    opacity: 1 !important;
    transform: none !important;
  }
  
  .typewriter-cursor {
    animation: none !important;
    opacity: 1 !important;
  }
}
```

### React Hook for Motion Preference

```jsx
function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);
    
    const handler = (e) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);
  
  return prefersReducedMotion;
}

function AccessibleStreamingText({ content, isStreaming }) {
  const prefersReducedMotion = usePrefersReducedMotion();
  
  if (prefersReducedMotion) {
    // No animation for users who prefer reduced motion
    return <span>{content}</span>;
  }
  
  return <AnimatedStreamingText content={content} isStreaming={isStreaming} />;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `transform` and `opacity` | Animate `margin`, `width`, `height` |
| Batch words to reduce animations | Animate each character separately |
| Support reduced motion | Force animations on all users |
| Use CSS for simple animations | Use JavaScript for every effect |
| Clean up `will-change` after animation | Leave `will-change` permanently |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Animation causes jank | Use GPU-accelerated properties only |
| Too many animated elements | Batch words into groups |
| Animation continues after stream ends | Complete animation immediately |
| Words animate out of order | Use consistent delay calculation |
| Memory leak from animation listeners | Remove listeners on cleanup |

---

## Hands-on Exercise

### Your Task

Create a streaming text component with:
1. Word-by-word fade-in animation
2. Staggered delays for natural appearance
3. Support for reduced motion preference
4. Cleanup when streaming completes

### Requirements

1. Each word fades in with a slight upward motion
2. Words stagger with 30ms delay between each
3. Reduced motion: show text instantly
4. Cursor appears at end during streaming

<details>
<summary>💡 Hints (click to expand)</summary>

- Split content by whitespace using `/(\s+)/`
- Use `animationDelay` style for stagger
- Check `prefers-reduced-motion` media query
- Add cursor only when `isStreaming` is true

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
function StaggeredFadeMessage({ content, isStreaming }) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const words = content.split(/(\s+)/);
  
  if (prefersReducedMotion) {
    return (
      <div className="message-body">
        {content}
        {isStreaming && <span className="cursor">▋</span>}
      </div>
    );
  }
  
  return (
    <div className="message-body">
      {words.map((word, i) => (
        <span
          key={i}
          className="fade-word"
          style={{ animationDelay: `${i * 30}ms` }}
        >
          {word}
        </span>
      ))}
      {isStreaming && <span className="cursor">▋</span>}
    </div>
  );
}
```

</details>

---

## Summary

✅ **Fade-in** creates smooth text appearance  
✅ **Slide-up** adds subtle motion for polish  
✅ **Typewriter** provides dramatic character-by-character reveal  
✅ **GPU-accelerated** properties prevent jank  
✅ **Batch animations** reduce performance overhead  
✅ **Reduced motion** support is essential for accessibility

---

## Further Reading

- [CSS will-change](https://developer.mozilla.org/en-US/docs/Web/CSS/will-change)
- [High Performance Animations](https://web.dev/animations-guide/)
- [CSS Triggers](https://csstriggers.com/)

---

**Previous:** [Cursor Indicators](./03-cursor-indicators.md)  
**Next:** [Scroll Behavior During Streaming](./05-scroll-behavior-during-streaming.md)

<!-- 
Sources Consulted:
- web.dev animations: https://web.dev/animations-guide/
- MDN CSS will-change: https://developer.mozilla.org/en-US/docs/Web/CSS/will-change
- CSS Triggers: https://csstriggers.com/
-->
