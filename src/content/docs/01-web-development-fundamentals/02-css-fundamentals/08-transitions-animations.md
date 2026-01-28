---
title: "Transitions & Animations"
---

# Transitions & Animations

## Introduction

Motion brings interfaces to life. Thoughtful animations guide attention, provide feedback, and create smoother user experiences. CSS provides two mechanisms: transitions for simple A-to-B changes, and animations for complex, multi-step sequences.

For AI interfaces, animations communicate state changes—loading indicators while waiting for responses, smooth message appearances, and typing effects that humanize AI output.

### What We'll Cover

- CSS transitions for hover effects and state changes
- The `@keyframes` rule for defining animations
- Animation properties and timing functions
- Performance optimization techniques
- Accessibility with `prefers-reduced-motion`

### Prerequisites

- CSS fundamentals (selectors, properties)
- Understanding of CSS transforms (optional but helpful)

---

## CSS Transitions

Transitions animate property changes between two states:

```css
.button {
  background: #6366f1;
  transition: background 0.2s ease;
}

.button:hover {
  background: #4f46e5;
}
```

### Transition Properties

```css
.element {
  /* Individual properties */
  transition-property: background, transform;
  transition-duration: 0.3s;
  transition-timing-function: ease;
  transition-delay: 0s;
  
  /* Shorthand: property duration timing delay */
  transition: background 0.3s ease 0s;
  
  /* Multiple transitions */
  transition: 
    background 0.3s ease,
    transform 0.2s ease-out,
    box-shadow 0.3s ease;
  
  /* All animatable properties */
  transition: all 0.3s ease;
}
```

### What Can Be Transitioned?

Most numeric properties can transition:

| ✅ Animatable | ❌ Not Animatable |
|--------------|------------------|
| `opacity` | `display` |
| `transform` | `visibility` (sort of) |
| `color`, `background-color` | `font-family` |
| `width`, `height` | `background-image` |
| `margin`, `padding` | `position` |
| `border-width`, `border-radius` | |
| `box-shadow` | |
| `filter` | |

### Transition Timing Functions

```css
.element {
  /* Built-in curves */
  transition-timing-function: linear;
  transition-timing-function: ease;        /* Default */
  transition-timing-function: ease-in;     /* Slow start */
  transition-timing-function: ease-out;    /* Slow end */
  transition-timing-function: ease-in-out; /* Slow start and end */
  
  /* Custom cubic-bezier */
  transition-timing-function: cubic-bezier(0.68, -0.55, 0.27, 1.55);
  
  /* Step functions */
  transition-timing-function: steps(4);
  transition-timing-function: step-start;
  transition-timing-function: step-end;
}
```

> **Tip:** Use [cubic-bezier.com](https://cubic-bezier.com) to visualize and create custom easing curves.

### Common Transition Patterns

#### Hover Effects

```css
.card {
  transform: translateY(0);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}
```

#### Button Feedback

```css
.button {
  transform: scale(1);
  transition: transform 0.1s ease;
}

.button:active {
  transform: scale(0.97);
}
```

#### Input Focus

```css
.input {
  border: 2px solid #d1d5db;
  outline: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.input:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}
```

---

## CSS Animations

For complex, multi-step animations, use `@keyframes`:

```css
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.message {
  animation: fadeIn 0.3s ease;
}
```

### Keyframe Syntax

```css
/* From/To syntax */
@keyframes slidein {
  from {
    transform: translateX(-100%);
  }
  to {
    transform: translateX(0);
  }
}

/* Percentage syntax */
@keyframes bounce {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-20px);
  }
  100% {
    transform: translateY(0);
  }
}

/* Multiple keyframes */
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
```

### Animation Properties

```css
.element {
  /* Individual properties */
  animation-name: fadeIn;
  animation-duration: 0.3s;
  animation-timing-function: ease;
  animation-delay: 0s;
  animation-iteration-count: 1;        /* or 'infinite' */
  animation-direction: normal;         /* or 'reverse', 'alternate' */
  animation-fill-mode: none;           /* or 'forwards', 'backwards', 'both' */
  animation-play-state: running;       /* or 'paused' */
  
  /* Shorthand */
  animation: fadeIn 0.3s ease 0s 1 normal forwards running;
  
  /* Common shorthand */
  animation: fadeIn 0.3s ease forwards;
}
```

### Animation Direction

```css
@keyframes swing {
  from { transform: rotate(-10deg); }
  to { transform: rotate(10deg); }
}

.pendulum {
  /* normal: 0% → 100% */
  animation: swing 1s ease-in-out normal;
  
  /* reverse: 100% → 0% */
  animation: swing 1s ease-in-out reverse;
  
  /* alternate: 0% → 100% → 0% → 100%... */
  animation: swing 1s ease-in-out infinite alternate;
  
  /* alternate-reverse: 100% → 0% → 100%... */
  animation: swing 1s ease-in-out infinite alternate-reverse;
}
```

### Animation Fill Mode

Controls style before/after animation:

```css
.element {
  /* none: Element reverts to original state after animation */
  animation-fill-mode: none;
  
  /* forwards: Element keeps final keyframe styles */
  animation-fill-mode: forwards;
  
  /* backwards: Element applies first keyframe during delay */
  animation-fill-mode: backwards;
  
  /* both: Combines forwards and backwards */
  animation-fill-mode: both;
}
```

---

## Common Animation Patterns

### Fade In

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.fade-in {
  animation: fadeIn 0.3s ease forwards;
}
```

### Slide In

```css
@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.slide-in {
  animation: slideInUp 0.4s ease forwards;
}
```

### Loading Spinner

```css
@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #e5e7eb;
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
```

### Typing Indicator

```css
@keyframes bounce {
  0%, 80%, 100% {
    transform: translateY(0);
  }
  40% {
    transform: translateY(-6px);
  }
}

.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #9ca3af;
  border-radius: 50%;
  animation: bounce 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}
```

### Pulse Effect

```css
@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(99, 102, 241, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0);
  }
}

.pulse {
  animation: pulse 2s infinite;
}
```

### Skeleton Loading

```css
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.skeleton {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

.skeleton-text {
  height: 1em;
  margin-bottom: 0.5em;
}

.skeleton-text:last-child {
  width: 60%;
}
```

---

## Transform for Animation

`transform` is the most performant property to animate:

```css
/* Translation (movement) */
transform: translateX(20px);
transform: translateY(-10px);
transform: translate(20px, -10px);
transform: translate3d(20px, -10px, 0);

/* Rotation */
transform: rotate(45deg);
transform: rotateX(45deg);
transform: rotateY(45deg);
transform: rotate3d(1, 1, 0, 45deg);

/* Scale */
transform: scale(1.2);
transform: scaleX(0.5);
transform: scale(1.2, 0.8);

/* Skew */
transform: skewX(10deg);
transform: skew(10deg, 5deg);

/* Combining */
transform: translateY(-10px) rotate(5deg) scale(1.05);
```

### Transform Origin

```css
.element {
  transform-origin: center;      /* Default */
  transform-origin: top left;
  transform-origin: 50% 100%;
  transform-origin: 0 0;
}

/* Rotate from corner */
.corner-rotate {
  transform-origin: bottom left;
  transition: transform 0.3s;
}

.corner-rotate:hover {
  transform: rotate(-10deg);
}
```

---

## Performance Optimization

### The Compositing Properties

These properties are GPU-accelerated and don't trigger layout/paint:

- `transform`
- `opacity`

```css
/* ✅ Performant - uses transform */
.performant {
  transform: translateX(0);
  transition: transform 0.3s;
}
.performant:hover {
  transform: translateX(20px);
}

/* ❌ Not performant - triggers layout */
.slow {
  margin-left: 0;
  transition: margin-left 0.3s;
}
.slow:hover {
  margin-left: 20px;
}
```

### Will-Change

Hint to browser about upcoming animations:

```css
.element {
  will-change: transform, opacity;
}
```

**Use sparingly:**
- Apply before animation starts
- Remove after animation ends
- Don't apply to too many elements

```javascript
element.addEventListener('mouseenter', () => {
  element.style.willChange = 'transform';
});

element.addEventListener('animationend', () => {
  element.style.willChange = 'auto';
});
```

### Avoiding Layout Thrashing

```css
/* ❌ Animating width causes layout recalculation */
.drawer {
  width: 0;
  transition: width 0.3s;
}
.drawer.open {
  width: 300px;
}

/* ✅ Use transform instead */
.drawer {
  transform: translateX(-100%);
  transition: transform 0.3s;
}
.drawer.open {
  transform: translateX(0);
}
```

---

## Accessibility

### Respecting Motion Preferences

Always respect `prefers-reduced-motion`:

```css
/* Default animations */
.animated {
  animation: fadeIn 0.3s ease;
}

.slide-in {
  animation: slideIn 0.5s ease;
}

/* Reduce or remove for users who prefer less motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

### Alternative: Subtle Motion

```css
@media (prefers-reduced-motion: reduce) {
  .animated {
    /* Fade only, no movement */
    animation: fadeOnly 0.1s ease;
  }
}

@keyframes fadeOnly {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

### JavaScript Detection

```javascript
const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
).matches;

if (!prefersReducedMotion) {
  element.classList.add('animate');
}
```

---

## Controlling Animations with JavaScript

### Triggering via Classes

```css
.message {
  opacity: 0;
  transform: translateY(20px);
}

.message.visible {
  animation: slideIn 0.4s ease forwards;
}
```

```javascript
const message = document.querySelector('.message');
message.classList.add('visible');
```

### Animation Events

```javascript
element.addEventListener('animationstart', (e) => {
  console.log('Started:', e.animationName);
});

element.addEventListener('animationend', (e) => {
  console.log('Ended:', e.animationName);
  element.classList.remove('animate');
});

element.addEventListener('animationiteration', (e) => {
  console.log('Iteration complete');
});
```

### Transition Events

```javascript
element.addEventListener('transitionstart', (e) => {
  console.log('Transitioning:', e.propertyName);
});

element.addEventListener('transitionend', (e) => {
  console.log('Transition complete:', e.propertyName);
});
```

### Pausing and Resuming

```css
.paused {
  animation-play-state: paused;
}
```

```javascript
function toggleAnimation(element) {
  element.classList.toggle('paused');
}
```

---

## AI Interface Examples

### Streaming Text Effect

```css
@keyframes cursor-blink {
  50% { opacity: 0; }
}

.ai-response::after {
  content: '▋';
  animation: cursor-blink 1s step-end infinite;
}

.ai-response.complete::after {
  display: none;
}
```

### Message Appearance

```css
@keyframes messageIn {
  from {
    opacity: 0;
    transform: translateY(10px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.message {
  animation: messageIn 0.3s ease forwards;
}

/* Stagger multiple messages */
.message:nth-child(1) { animation-delay: 0s; }
.message:nth-child(2) { animation-delay: 0.1s; }
.message:nth-child(3) { animation-delay: 0.2s; }
```

### Processing State

```css
@keyframes thinking {
  0%, 100% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
}

.ai-thinking {
  background: linear-gradient(
    90deg,
    #f0f4ff 0%,
    #c7d2fe 50%,
    #f0f4ff 100%
  );
  background-size: 200% 100%;
  animation: thinking 2s ease infinite;
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Animate `transform` and `opacity` | GPU-accelerated, 60fps |
| Keep durations under 0.4s | Feels responsive |
| Use `ease-out` for entrances | Natural deceleration |
| Use `ease-in` for exits | Natural acceleration |
| Respect `prefers-reduced-motion` | Accessibility |
| Don't animate on page load | Distracting |
| Test on slower devices | Animations can lag |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Animating `width`/`height` | Use `transform: scale()` |
| Animating `margin`/`left` | Use `transform: translate()` |
| Using `will-change` everywhere | Apply only to animating elements |
| Infinite animations without purpose | Use sparingly, allow pause |
| No `animation-fill-mode` | Set `forwards` to keep final state |
| Forgetting motion preferences | Always add `prefers-reduced-motion` |

---

## Hands-on Exercise

### Your Task

Create an animated notification system:

1. Notifications slide in from the right
2. Auto-dismiss after 5 seconds with progress bar
3. Manual dismiss with fade out
4. Stacked notifications animate when one is removed
5. Respects reduced motion preferences

<details>
<summary>✅ Solution</summary>

```css
/* Notifications container */
.notifications {
  position: fixed;
  top: 1rem;
  right: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  z-index: 1000;
}

/* Base notification */
.notification {
  --duration: 5s;
  
  position: relative;
  padding: 1rem 2.5rem 1rem 1rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  min-width: 300px;
  max-width: 400px;
  
  animation: slideIn 0.3s ease forwards;
}

/* Slide in animation */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* Slide out animation */
@keyframes slideOut {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100%);
  }
}

.notification.dismissing {
  animation: slideOut 0.2s ease forwards;
}

/* Progress bar */
.notification::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3px;
  background: #6366f1;
  border-radius: 0 0 8px 8px;
  animation: progress var(--duration) linear forwards;
}

@keyframes progress {
  from { width: 100%; }
  to { width: 0%; }
}

/* Dismiss button */
.notification-dismiss {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: none;
  border: none;
  font-size: 1.25rem;
  cursor: pointer;
  opacity: 0.5;
  transition: opacity 0.2s;
}

.notification-dismiss:hover {
  opacity: 1;
}

/* Variants */
.notification--success {
  border-left: 4px solid #10b981;
}

.notification--error {
  border-left: 4px solid #ef4444;
}

.notification--error::after {
  background: #ef4444;
}

/* Stacking animation */
.notification {
  transition: transform 0.3s ease;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .notification {
    animation: fadeIn 0.1s ease forwards;
  }
  
  .notification.dismissing {
    animation: fadeOut 0.1s ease forwards;
  }
  
  .notification::after {
    animation: none;
    width: 100%;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
  }
}
```

```javascript
class NotificationManager {
  constructor() {
    this.container = document.createElement('div');
    this.container.className = 'notifications';
    document.body.appendChild(this.container);
  }
  
  show(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;
    notification.style.setProperty('--duration', `${duration}ms`);
    
    notification.innerHTML = `
      <span class="notification-message">${message}</span>
      <button class="notification-dismiss">×</button>
    `;
    
    // Manual dismiss
    const dismissBtn = notification.querySelector('.notification-dismiss');
    dismissBtn.addEventListener('click', () => this.dismiss(notification));
    
    // Auto dismiss
    const timer = setTimeout(() => this.dismiss(notification), duration);
    
    // Pause on hover
    notification.addEventListener('mouseenter', () => {
      clearTimeout(timer);
      notification.style.animationPlayState = 'paused';
      notification.querySelector('::after')?.style?.animationPlayState = 'paused';
    });
    
    this.container.appendChild(notification);
    
    return notification;
  }
  
  dismiss(notification) {
    notification.classList.add('dismissing');
    
    notification.addEventListener('animationend', () => {
      notification.remove();
    });
  }
}

// Usage
const notifications = new NotificationManager();
notifications.show('File uploaded successfully', 'success');
notifications.show('Connection lost', 'error', 8000);
```
</details>

---

## Summary

✅ **Transitions** animate between two states with `transition: property duration timing`

✅ **Animations** define complex sequences with `@keyframes` and `animation` property

✅ Use **`transform`** and **`opacity`** for smooth 60fps animations

✅ **Timing functions** control acceleration: `ease-out` for entrances, `ease-in` for exits

✅ **Always respect `prefers-reduced-motion`** for accessible animations

✅ Use **animation events** (`animationend`, `transitionend`) for JavaScript coordination

---

**Previous:** [CSS Variables](./07-css-variables.md)

**Next:** [Modern CSS Features](./09-modern-css-features.md)

<!-- 
Sources Consulted:
- MDN CSS Transitions: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_transitions
- MDN CSS Animations: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_animations
- MDN @keyframes: https://developer.mozilla.org/en-US/docs/Web/CSS/@keyframes
- MDN prefers-reduced-motion: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion
-->
