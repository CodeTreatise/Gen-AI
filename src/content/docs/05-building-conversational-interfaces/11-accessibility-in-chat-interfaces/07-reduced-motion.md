---
title: "Reduced Motion Preferences"
---

# Reduced Motion Preferences

## Introduction

Animations make chat interfaces feel alive—typing indicators pulse, messages slide in, buttons bounce on click. But for users with vestibular disorders, motion sensitivity, or certain cognitive differences, these same animations can cause dizziness, nausea, or difficulty focusing.

The `prefers-reduced-motion` media query lets users signal their preference through their operating system, and your interface can respond by reducing or eliminating motion. This isn't about removing delight—it's about providing equivalent experiences that don't cause physical discomfort.

### What We'll Cover

- Understanding motion sensitivity and vestibular disorders
- Detecting reduced motion preferences with CSS and JavaScript
- Strategies for reducing motion without losing meaning
- Animating safely when motion is acceptable
- Testing reduced motion implementations

### Prerequisites

- CSS animations and transitions
- JavaScript event handling
- Understanding of chat interface patterns

---

## Understanding Motion Sensitivity

Motion sensitivity affects more people than many developers realize.

### Who Is Affected?

| Condition | Symptoms from Motion | Estimated Prevalence |
|-----------|---------------------|---------------------|
| Vestibular disorders | Dizziness, vertigo, nausea | 35% of adults 40+ |
| Migraine with aura | Triggered headaches | 15% of population |
| Motion sickness prone | Nausea from screen motion | 25-40% of people |
| ADHD | Distraction from animations | 4-5% of adults |
| Epilepsy (photosensitive) | Seizure risk from flashing | 1 in 4,000 |

### What Triggers Reactions?

| Motion Type | Risk Level | Examples |
|-------------|------------|----------|
| Parallax scrolling | **High** | Background moves at different speed |
| Full-screen transitions | **High** | Page zoom, slide between views |
| Rapid/pulsing animation | **High** | Loading spinners, bouncing elements |
| Auto-playing video | **Medium** | Background videos, GIFs |
| Hover effects | **Low** | Button scaling, color transitions |
| Subtle fades | **Low** | Opacity changes, slow transitions |

> **Important:** Users enable reduced motion for medical reasons. Ignoring this preference can cause real physical harm.

---

## Detecting Reduced Motion Preferences

### Operating System Settings

Users set their preference at the OS level:

| OS | Setting Location |
|----|-----------------|
| macOS | System Preferences → Accessibility → Display → Reduce motion |
| Windows | Settings → Ease of Access → Display → Show animations |
| iOS | Settings → Accessibility → Motion → Reduce Motion |
| Android | Settings → Accessibility → Remove animations |

### CSS Media Query

```css
/* Default: Full motion experience */
.message {
  animation: slideIn 0.3s ease-out;
}

/* Reduced motion: Minimize or remove animation */
@media (prefers-reduced-motion: reduce) {
  .message {
    animation: none;
  }
}
```

### CSS Values

| Value | Meaning |
|-------|---------|
| `no-preference` | User hasn't set preference (or allows motion) |
| `reduce` | User prefers less motion |

### JavaScript Detection

```javascript
// Check preference
function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// Usage
if (prefersReducedMotion()) {
  // Use instant or minimal animations
  element.style.transition = 'none';
} else {
  // Use full animations
  element.style.transition = 'transform 0.3s ease-out';
}
```

### Reactive Detection

```javascript
// Listen for preference changes
const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

motionQuery.addEventListener('change', (event) => {
  if (event.matches) {
    disableAnimations();
  } else {
    enableAnimations();
  }
});

function disableAnimations() {
  document.documentElement.classList.add('reduce-motion');
}

function enableAnimations() {
  document.documentElement.classList.remove('reduce-motion');
}
```

---

## CSS Strategies for Reduced Motion

### Strategy 1: Remove Animation Entirely

```css
/* Full motion */
.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-dot {
  width: 8px;
  height: 8px;
  background: currentColor;
  border-radius: 50%;
  animation: bounce 1.4s ease-in-out infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}

/* Reduced motion: Replace with static indicator */
@media (prefers-reduced-motion: reduce) {
  .typing-dot {
    animation: none;
  }
  
  /* Alternative: Use opacity pulse instead of movement */
  .typing-indicator::after {
    content: "AI is typing...";
  }
  
  .typing-dot {
    display: none;
  }
}
```

### Strategy 2: Replace Motion with Opacity

```css
/* Full motion: Slide in */
.message {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Reduced motion: Fade in only (no movement) */
@media (prefers-reduced-motion: reduce) {
  .message {
    animation: fadeIn 0.2s ease-out;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
}
```

### Strategy 3: Instant Transitions

```css
/* Full motion */
.send-button {
  transition: background-color 0.2s, transform 0.2s;
}

.send-button:hover {
  background-color: var(--primary-hover);
  transform: scale(1.05);
}

.send-button:active {
  transform: scale(0.95);
}

/* Reduced motion: Instant state changes */
@media (prefers-reduced-motion: reduce) {
  .send-button {
    transition: none;
  }
  
  .send-button:hover,
  .send-button:active {
    transform: none;
    /* Color change is instant but still provides feedback */
  }
}
```

### Strategy 4: Global Reset

```css
/* Nuclear option: Remove ALL motion */
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

> **Warning:** Use the global reset cautiously—it may break functionality that depends on animation timing.

---

## JavaScript Animation Considerations

### Respecting Preference in JavaScript

```javascript
class AnimationController {
  constructor() {
    this.reducedMotion = this.checkReducedMotion();
    this.listenForChanges();
  }
  
  checkReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  
  listenForChanges() {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    mediaQuery.addEventListener('change', () => {
      this.reducedMotion = mediaQuery.matches;
    });
  }
  
  // Animation methods respect the preference
  animate(element, keyframes, options) {
    if (this.reducedMotion) {
      // Skip to final state
      const finalFrame = keyframes[keyframes.length - 1];
      Object.assign(element.style, finalFrame);
      return Promise.resolve();
    }
    
    return element.animate(keyframes, options).finished;
  }
  
  // Scroll methods respect the preference
  scrollTo(element, options = {}) {
    const behavior = this.reducedMotion ? 'auto' : 'smooth';
    element.scrollIntoView({ behavior, ...options });
  }
}

// Usage
const animator = new AnimationController();

// This respects user preference automatically
async function showNewMessage(messageElement) {
  await animator.animate(messageElement, [
    { opacity: 0, transform: 'translateY(20px)' },
    { opacity: 1, transform: 'translateY(0)' }
  ], { duration: 300 });
}
```

### Scroll Behavior

```javascript
// Smooth scroll to new message
function scrollToLatestMessage() {
  const latestMessage = document.querySelector('.message:last-child');
  
  // Check preference for scroll behavior
  const behavior = prefersReducedMotion() ? 'auto' : 'smooth';
  
  latestMessage.scrollIntoView({ 
    behavior, 
    block: 'end' 
  });
}
```

### Timer-Based Animations

```javascript
// Loading spinner with reduced motion alternative
class LoadingIndicator {
  constructor(container) {
    this.container = container;
    this.reducedMotion = prefersReducedMotion();
  }
  
  show() {
    if (this.reducedMotion) {
      // Static text indicator
      this.container.innerHTML = `
        <div class="loading-text" role="status">
          Loading<span class="loading-ellipsis">...</span>
        </div>
      `;
    } else {
      // Animated spinner
      this.container.innerHTML = `
        <div class="loading-spinner" role="status" aria-label="Loading">
          <div class="spinner"></div>
        </div>
      `;
    }
  }
  
  hide() {
    this.container.innerHTML = '';
  }
}
```

---

## Chat-Specific Motion Patterns

### New Message Appearance

```css
/* Full motion: Slide + fade */
.message--new {
  animation: messageEnter 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes messageEnter {
  from {
    opacity: 0;
    transform: translateY(30px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

/* Reduced motion: Quick fade only */
@media (prefers-reduced-motion: reduce) {
  .message--new {
    animation: quickFade 0.15s ease-out;
  }
  
  @keyframes quickFade {
    from { opacity: 0; }
    to { opacity: 1; }
  }
}
```

### Message Deletion

```css
/* Full motion: Shrink and fade */
.message--deleting {
  animation: messageExit 0.3s ease-in forwards;
}

@keyframes messageExit {
  to {
    opacity: 0;
    transform: scale(0.9);
    height: 0;
    margin: 0;
    padding: 0;
  }
}

/* Reduced motion: Instant removal */
@media (prefers-reduced-motion: reduce) {
  .message--deleting {
    animation: none;
    opacity: 0;
    display: none;
  }
}
```

### Streaming Response Effect

```css
/* Full motion: Cursor blink while streaming */
.streaming-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background: currentColor;
  margin-left: 2px;
  animation: blink 1s steps(2) infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* Reduced motion: Solid cursor, no blink */
@media (prefers-reduced-motion: reduce) {
  .streaming-cursor {
    animation: none;
    opacity: 0.7;
  }
}
```

### Modal/Dialog Animations

```css
/* Full motion: Scale and fade dialog */
.modal-overlay {
  animation: overlayFade 0.2s ease-out;
}

.modal-content {
  animation: modalEnter 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes overlayFade {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes modalEnter {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(-10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

/* Reduced motion: Instant appearance */
@media (prefers-reduced-motion: reduce) {
  .modal-overlay,
  .modal-content {
    animation: none;
  }
}
```

---

## Safe Animation Techniques

Some animations are generally safe even for motion-sensitive users.

### Safe Patterns

| Animation Type | Why It's Safe |
|----------------|---------------|
| Opacity/fade | No movement, no disorientation |
| Color changes | Static position, low intensity |
| Border/outline changes | Small, contained, non-moving |
| Very short transitions (<100ms) | Too fast to cause vestibular response |
| User-initiated | User expects and controls the motion |

### Example: Safe Button Feedback

```css
/* These are generally safe for all users */
.button {
  background-color: var(--primary);
  border: 2px solid transparent;
  /* Short transition is safe */
  transition: background-color 0.1s, border-color 0.1s;
}

.button:hover {
  background-color: var(--primary-hover);
}

.button:focus-visible {
  border-color: var(--focus-ring);
  /* Instant focus ring is safe */
}

.button:active {
  background-color: var(--primary-active);
  /* Color change, no transform */
}
```

### User-Initiated vs Auto-Playing

```javascript
// User-initiated animation is acceptable
sendButton.addEventListener('click', async () => {
  // User clicked, they expect something to happen
  if (!prefersReducedMotion()) {
    await animateSendButton();
  }
  await sendMessage();
});

// Auto-playing animation should respect preference
function showTypingIndicator() {
  if (prefersReducedMotion()) {
    // Static text instead of bouncing dots
    typingIndicator.innerHTML = 'AI is typing...';
  } else {
    // Animated bouncing dots
    typingIndicator.innerHTML = `
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    `;
  }
}
```

---

## Testing Reduced Motion

### Browser DevTools

```
Chrome/Edge:
1. Open DevTools (F12)
2. Command Palette (Ctrl+Shift+P)
3. Type "rendering"
4. Select "Show Rendering"
5. Find "Emulate CSS media feature prefers-reduced-motion"
6. Select "prefers-reduced-motion: reduce"

Firefox:
1. about:config
2. Search "ui.prefersReducedMotion"
3. Set to 1 (reduce) or 0 (no-preference)

Safari:
1. Develop menu → Experimental Features
2. Enable reduced motion simulation
```

### Automated Testing

```javascript
// Jest/Testing Library example
describe('Reduced motion', () => {
  beforeEach(() => {
    // Mock reduced motion preference
    Object.defineProperty(window, 'matchMedia', {
      value: jest.fn().mockImplementation(query => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
      })),
    });
  });
  
  test('typing indicator shows static text when reduced motion', () => {
    const { container } = render(<TypingIndicator />);
    
    expect(container.textContent).toBe('AI is typing...');
    expect(container.querySelector('.animated-dot')).toBeNull();
  });
  
  test('messages appear without animation', () => {
    const { container } = render(<Message content="Hello" />);
    const message = container.querySelector('.message');
    
    const styles = window.getComputedStyle(message);
    expect(styles.animation).toBe('none');
  });
});
```

### Manual Testing Checklist

| Test | Expected Behavior with Reduced Motion |
|------|--------------------------------------|
| Page load animations | Instant appearance |
| New message entrance | Fade only or instant |
| Typing indicator | Static text or solid dots |
| Modal/dialog open | Instant appearance |
| Smooth scroll | Jump scroll instead |
| Hover effects | No transform/movement |
| Auto-playing content | Paused or static |
| Loading spinners | Static text indicator |

---

## Complete Reduced Motion System

```css
/* ============================================
   Reduced Motion System for Chat Interface
   ============================================ */

/* CSS Custom Properties for animation control */
:root {
  --animation-duration: 0.3s;
  --transition-duration: 0.2s;
  --scroll-behavior: smooth;
}

@media (prefers-reduced-motion: reduce) {
  :root {
    --animation-duration: 0.01ms;
    --transition-duration: 0.01ms;
    --scroll-behavior: auto;
  }
}

/* Base styles using custom properties */
.message {
  animation: messageEnter var(--animation-duration) ease-out;
}

.button {
  transition: background-color var(--transition-duration);
}

html {
  scroll-behavior: var(--scroll-behavior);
}

/* Specific reduced motion overrides */
@media (prefers-reduced-motion: reduce) {
  /* Remove all transforms */
  .message,
  .modal,
  .tooltip {
    animation: fadeIn 0.01ms ease-out;
  }
  
  @keyframes fadeIn {
    to { opacity: 1; }
  }
  
  /* Static typing indicator */
  .typing-indicator .dot {
    animation: none;
    opacity: 0.6;
  }
  
  .typing-indicator::after {
    content: " (typing)";
    animation: none;
  }
  
  /* No hover transforms */
  .button:hover,
  .button:active {
    transform: none;
  }
  
  /* Static loading indicator */
  .loading-spinner {
    animation: none;
  }
  
  .loading-spinner::after {
    content: "Loading...";
    animation: none;
  }
  
  /* Disable parallax */
  .parallax {
    transform: none !important;
  }
  
  /* Instant focus */
  *:focus {
    transition: none;
  }
}
```

```javascript
// JavaScript Reduced Motion Utility
class MotionPreferences {
  constructor() {
    this.query = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.callbacks = new Set();
    
    this.query.addEventListener('change', () => {
      this.callbacks.forEach(cb => cb(this.prefersReduced));
    });
  }
  
  get prefersReduced() {
    return this.query.matches;
  }
  
  onChange(callback) {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }
  
  // Utility methods
  getDuration(fullDuration) {
    return this.prefersReduced ? 0 : fullDuration;
  }
  
  getScrollBehavior() {
    return this.prefersReduced ? 'auto' : 'smooth';
  }
  
  animate(element, keyframes, options) {
    if (this.prefersReduced) {
      const final = keyframes[keyframes.length - 1];
      Object.entries(final).forEach(([prop, value]) => {
        element.style[prop] = value;
      });
      return Promise.resolve();
    }
    return element.animate(keyframes, options).finished;
  }
}

// Global instance
const motion = new MotionPreferences();

// Usage examples
motion.onChange(reduced => {
  console.log(`Motion preference: ${reduced ? 'reduced' : 'full'}`);
});

// Animate with preference
await motion.animate(messageEl, [
  { opacity: 0, transform: 'translateY(20px)' },
  { opacity: 1, transform: 'translateY(0)' }
], { duration: motion.getDuration(300) });

// Scroll with preference
element.scrollIntoView({ behavior: motion.getScrollBehavior() });
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using global `animation: none !important` | Breaks animation-dependent functionality |
| Forgetting scroll-behavior | Include in reduced motion styles |
| Assuming reduced = no feedback | Replace with safe alternatives |
| Only testing with animations on | Always test both states |
| Ignoring preference changes | Listen for `change` event |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Default to reduced motion first | Progressive enhancement approach |
| Use CSS custom properties | Easy to toggle all durations |
| Provide meaningful alternatives | Static text instead of animated indicators |
| Test with OS setting enabled | Most accurate simulation |
| Listen for preference changes | User may toggle mid-session |
| Preserve essential motion | User-initiated actions can animate |

---

## Hands-on Exercise

### Your Task

Implement a chat message system that respects reduced motion preferences.

### Requirements

1. Detect `prefers-reduced-motion` in CSS and JavaScript
2. New messages: slide+fade with motion, instant with reduced
3. Typing indicator: bouncing dots with motion, static text with reduced
4. Smooth scroll with motion, instant scroll with reduced
5. Listen for preference changes at runtime

### Expected Result

- Full motion: Messages slide in, dots bounce, smooth scrolling
- Reduced motion: Messages appear instantly, static "AI is typing...", instant scrolling

<details>
<summary>💡 Hints (click to expand)</summary>

- Use CSS custom properties: `--animation-duration: 0.3s`
- Set `--animation-duration: 0ms` in reduced motion media query
- `matchMedia().addEventListener('change', callback)` for runtime changes
- `scroll-behavior: auto` is the reduced motion scroll

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```css
:root {
  --message-animation: messageSlide 0.3s ease-out;
  --scroll-behavior: smooth;
}

@media (prefers-reduced-motion: reduce) {
  :root {
    --message-animation: none;
    --scroll-behavior: auto;
  }
  
  .typing-dots { display: none; }
  .typing-text { display: block; }
}

.message { animation: var(--message-animation); }
html { scroll-behavior: var(--scroll-behavior); }

.typing-text { display: none; }

@keyframes messageSlide {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
```

```javascript
const motion = {
  query: window.matchMedia('(prefers-reduced-motion: reduce)'),
  get reduced() { return this.query.matches; },
  scroll(el) {
    el.scrollIntoView({ behavior: this.reduced ? 'auto' : 'smooth' });
  }
};

motion.query.addEventListener('change', (e) => {
  document.body.classList.toggle('reduced-motion', e.matches);
});

// Initial state
document.body.classList.toggle('reduced-motion', motion.reduced);
```

</details>

### Bonus Challenges

- [ ] Add a manual toggle that overrides OS preference
- [ ] Create safe loading animation that works for both states
- [ ] Implement progressive reduction (less motion vs no motion)

---

## Summary

✅ Use `@media (prefers-reduced-motion: reduce)` to detect user preference

✅ Replace motion with opacity fades or instant state changes

✅ Use CSS custom properties to easily toggle animation durations

✅ Include `scroll-behavior: auto` in reduced motion styles

✅ Listen for preference changes with `matchMedia().addEventListener('change')`

**Next:** [Lesson Summary](./00-accessibility-overview.md) | [Back to Unit Overview](../00-overview.md)

---

## Further Reading

- [MDN: prefers-reduced-motion](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion) - Complete reference
- [web.dev: Reduced Motion](https://web.dev/prefers-reduced-motion/) - Best practices and examples
- [A11y Project: Reduced Motion](https://www.a11yproject.com/posts/understanding-vestibular-disorders/) - Understanding vestibular disorders
- [Smashing Magazine: Reduced Motion](https://www.smashingmagazine.com/2021/10/respecting-users-motion-preferences/) - Comprehensive guide

<!--
Sources Consulted:
- MDN prefers-reduced-motion: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion
- WCAG 2.3.3 Animation from Interactions: https://www.w3.org/WAI/WCAG21/Understanding/animation-from-interactions
- web.dev Reduced Motion: https://web.dev/prefers-reduced-motion/
-->
