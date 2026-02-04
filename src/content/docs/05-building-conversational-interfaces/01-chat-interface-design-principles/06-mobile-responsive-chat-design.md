---
title: "Mobile-Responsive Chat Design"
---

# Mobile-Responsive Chat Design

## Introduction

Over 60% of chat interactions happen on mobile devices. A chat interface that works beautifully on desktop but fails on mobile will frustrate the majority of your users. Mobile chat design requires careful attention to touch targets, viewport handling, and the unique constraints of small screens.

In this lesson, we'll explore patterns for building chat interfaces that work flawlessly across all device sizes.

### What We'll Cover

- Flexible container layouts for different screen sizes
- Touch target sizing and spacing
- Virtual keyboard and viewport handling
- Input method adaptation
- Mobile-specific interaction patterns

### Prerequisites

- CSS responsive design fundamentals ([Unit 1](../../../01-web-development-fundamentals/02-css-fundamentals/00-css-fundamentals.md))
- Understanding of viewport units
- JavaScript event handling basics

---

## Flexible Container Layouts

### Mobile-First Approach

Start with mobile styles, then enhance for larger screens:

```css
/* Base: Mobile (< 640px) */
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  height: 100dvh; /* Dynamic viewport height */
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
}

.message {
  max-width: 90%; /* Allow wider messages on mobile */
  padding: 0.625rem 0.875rem;
  font-size: 0.9375rem; /* 15px - readable on mobile */
}

/* Tablet (640px+) */
@media (min-width: 640px) {
  .message-list {
    padding: 1rem;
  }
  
  .message {
    max-width: 80%;
    padding: 0.75rem 1rem;
    font-size: 1rem;
  }
}

/* Desktop (1024px+) */
@media (min-width: 1024px) {
  .chat-app {
    max-width: 900px;
    margin: 0 auto;
    border-left: 1px solid #e5e7eb;
    border-right: 1px solid #e5e7eb;
  }
  
  .message {
    max-width: 70%;
  }
}
```

### Dynamic Viewport Height

The `100vh` unit doesn't account for mobile browser UI (address bar, navigation). Use `dvh`:

```css
.chat-app {
  /* Fallback for older browsers */
  height: 100vh;
  
  /* Modern browsers: accounts for browser UI */
  height: 100dvh;
}

/* Alternative: CSS custom property */
:root {
  --app-height: 100vh;
}

@supports (height: 100dvh) {
  :root {
    --app-height: 100dvh;
  }
}

.chat-app {
  height: var(--app-height);
}
```

### JavaScript Fallback for Viewport Height

For browsers without `dvh` support:

```javascript
function setAppHeight() {
  const vh = window.innerHeight * 0.01;
  document.documentElement.style.setProperty('--vh', `${vh}px`);
}

setAppHeight();
window.addEventListener('resize', setAppHeight);
```

```css
.chat-app {
  height: 100vh; /* Fallback */
  height: calc(var(--vh, 1vh) * 100);
}
```

---

## Touch Target Sizing

### Minimum Touch Target Size

Apple and Google recommend minimum touch targets:

| Standard | Minimum Size |
|----------|--------------|
| Apple HIG | 44 × 44 points |
| Material Design | 48 × 48 dp |
| WCAG 2.2 | 24 × 24 CSS pixels (minimum) |

```css
/* Buttons and interactive elements */
.action-button {
  min-width: 44px;
  min-height: 44px;
  padding: 0.75rem;
}

/* Icon buttons need explicit sizing */
.icon-button {
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-button svg {
  width: 24px;
  height: 24px;
}
```

### Touch Target Spacing

Prevent accidental taps with adequate spacing:

```css
.message-actions {
  display: flex;
  gap: 0.5rem; /* Minimum spacing between buttons */
}

.action-button {
  /* Ensure no overlapping touch areas */
  margin: 0.25rem;
}
```

### Expandable Touch Areas

Make small visual elements easier to tap:

```css
.copy-button {
  /* Visual size */
  width: 32px;
  height: 32px;
  
  /* Larger touch area */
  position: relative;
}

.copy-button::before {
  content: '';
  position: absolute;
  inset: -8px; /* Extends touch area by 8px on each side */
}
```

---

## Virtual Keyboard Handling

### The Viewport Problem

When the virtual keyboard opens, it can:
- Push content up (iOS default)
- Resize the viewport (Android default)
- Cover the input field

### Using the Visual Viewport API

```javascript
if ('visualViewport' in window) {
  const viewport = window.visualViewport;
  
  function handleViewportChange() {
    const keyboardHeight = window.innerHeight - viewport.height;
    
    document.documentElement.style.setProperty(
      '--keyboard-height',
      `${keyboardHeight}px`
    );
    
    // Scroll input into view if hidden
    if (keyboardHeight > 0) {
      const input = document.querySelector('#message-input');
      if (input === document.activeElement) {
        input.scrollIntoView({ block: 'nearest' });
      }
    }
  }
  
  viewport.addEventListener('resize', handleViewportChange);
  viewport.addEventListener('scroll', handleViewportChange);
}
```

```css
.input-area {
  position: sticky;
  bottom: 0;
  /* Adjust for keyboard */
  padding-bottom: calc(0.75rem + var(--keyboard-height, 0px));
  transition: padding-bottom 0.15s ease;
}
```

### Viewport Meta Tag Options

```html
<!-- Default: Browser decides resize behavior -->
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<!-- iOS 15+: Content resizes with keyboard -->
<meta name="viewport" content="width=device-width, initial-scale=1.0, interactive-widget=resizes-content">

<!-- Prevent content resizing (keyboard overlays) -->
<meta name="viewport" content="width=device-width, initial-scale=1.0, interactive-widget=overlays-content">
```

### Scroll Into View on Focus

```javascript
const input = document.querySelector('#message-input');

input.addEventListener('focus', () => {
  // Small delay to let keyboard appear
  setTimeout(() => {
    input.scrollIntoView({ 
      behavior: 'smooth', 
      block: 'center' 
    });
  }, 300);
});
```

---

## Safe Area Handling

Modern phones have notches, rounded corners, and home indicators:

### Safe Area Insets

```css
.chat-app {
  /* Respect safe areas */
  padding-top: env(safe-area-inset-top);
  padding-left: env(safe-area-inset-left);
  padding-right: env(safe-area-inset-right);
}

.input-area {
  /* Ensure input isn't behind home indicator */
  padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
}

.chat-header {
  /* Ensure header isn't behind notch */
  padding-top: max(0.75rem, env(safe-area-inset-top));
}
```

### Full-Screen PWA Safe Areas

```css
/* When installed as PWA */
@media (display-mode: standalone) {
  .chat-app {
    padding-top: env(safe-area-inset-top);
  }
  
  .chat-header {
    /* Account for status bar in PWA mode */
    padding-top: calc(0.75rem + env(safe-area-inset-top));
  }
}
```

---

## Input Method Adaptation

### Flexible Input Container

```css
.input-area {
  display: flex;
  gap: 0.5rem;
  padding: 0.75rem;
  background: #fff;
  border-top: 1px solid #e5e7eb;
}

.input-container {
  flex: 1;
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  background: #f3f4f6;
  border-radius: 1.5rem;
  padding: 0.5rem 1rem;
}

#message-input {
  flex: 1;
  border: none;
  background: transparent;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  line-height: 1.5;
  max-height: 120px;
  outline: none;
}

.send-button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #3b82f6;
  color: white;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
```

### Auto-Growing Textarea

```javascript
const textarea = document.querySelector('#message-input');

textarea.addEventListener('input', () => {
  // Reset height to get accurate scrollHeight
  textarea.style.height = 'auto';
  
  // Set height to content (with max)
  const maxHeight = 120;
  const newHeight = Math.min(textarea.scrollHeight, maxHeight);
  textarea.style.height = `${newHeight}px`;
});

// Handle paste
textarea.addEventListener('paste', () => {
  requestAnimationFrame(() => {
    textarea.dispatchEvent(new Event('input'));
  });
});
```

### Attachment Button

```css
.attachment-button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
}

.attachment-button:active {
  background: #e5e7eb;
}
```

```html
<div class="input-area">
  <button class="attachment-button" aria-label="Attach file">
    <svg width="24" height="24" viewBox="0 0 24 24">
      <!-- Plus or paperclip icon -->
    </svg>
  </button>
  <div class="input-container">
    <textarea id="message-input" rows="1" placeholder="Message..."></textarea>
  </div>
  <button class="send-button" aria-label="Send message">
    <svg width="20" height="20" viewBox="0 0 24 24">
      <!-- Send arrow icon -->
    </svg>
  </button>
</div>
```

---

## Mobile-Specific Interactions

### Pull-to-Refresh Prevention

Disable pull-to-refresh on chat container:

```css
.message-list {
  overscroll-behavior: contain;
}

/* Prevent bouncing on iOS */
html, body {
  overscroll-behavior: none;
}
```

### Swipe Actions on Messages

```css
.message-wrapper {
  position: relative;
  overflow: hidden;
}

.message-content {
  transition: transform 0.2s ease;
}

.swipe-actions {
  position: absolute;
  top: 0;
  bottom: 0;
  right: 0;
  display: flex;
  align-items: center;
  padding: 0 1rem;
  background: #ef4444;
  color: white;
  transform: translateX(100%);
}

.message-wrapper.swiped .message-content {
  transform: translateX(-80px);
}

.message-wrapper.swiped .swipe-actions {
  transform: translateX(0);
}
```

```javascript
let touchStartX = 0;
let touchDelta = 0;

function handleTouchStart(e) {
  touchStartX = e.touches[0].clientX;
}

function handleTouchMove(e) {
  touchDelta = e.touches[0].clientX - touchStartX;
  
  if (touchDelta < -20) {
    const content = e.currentTarget.querySelector('.message-content');
    content.style.transform = `translateX(${Math.max(touchDelta, -80)}px)`;
  }
}

function handleTouchEnd(e) {
  const wrapper = e.currentTarget;
  const content = wrapper.querySelector('.message-content');
  
  if (touchDelta < -50) {
    wrapper.classList.add('swiped');
  } else {
    content.style.transform = '';
    wrapper.classList.remove('swiped');
  }
  
  touchDelta = 0;
}
```

### Long-Press Context Menu

```javascript
let longPressTimer = null;

function handleTouchStart(e, message) {
  longPressTimer = setTimeout(() => {
    showContextMenu(e.touches[0], message);
    // Haptic feedback if available
    if ('vibrate' in navigator) {
      navigator.vibrate(50);
    }
  }, 500);
}

function handleTouchEnd() {
  clearTimeout(longPressTimer);
}

function showContextMenu(touch, message) {
  const menu = document.createElement('div');
  menu.className = 'context-menu';
  menu.innerHTML = `
    <button>Copy</button>
    <button>Reply</button>
    <button>Delete</button>
  `;
  menu.style.top = `${touch.clientY}px`;
  menu.style.left = `${touch.clientX}px`;
  document.body.appendChild(menu);
}
```

```css
.context-menu {
  position: fixed;
  background: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  z-index: 1000;
}

.context-menu button {
  display: block;
  width: 100%;
  padding: 0.75rem 1.5rem;
  border: none;
  background: none;
  text-align: left;
  font-size: 1rem;
}

.context-menu button:active {
  background: #f3f4f6;
}
```

---

## Performance on Mobile

### Efficient Scrolling

```css
.message-list {
  /* Enable GPU acceleration */
  transform: translateZ(0);
  will-change: scroll-position;
  
  /* Use momentum scrolling on iOS */
  -webkit-overflow-scrolling: touch;
}
```

### Lazy Load Images

```javascript
function lazyLoadImages(container) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        observer.unobserve(img);
      }
    });
  }, { rootMargin: '100px' });
  
  container.querySelectorAll('img[data-src]').forEach(img => {
    observer.observe(img);
  });
}
```

### Debounce Expensive Operations

```javascript
function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

// Debounce scroll handlers
const handleScroll = debounce(() => {
  // Check if at top for "load more"
  // Update scroll position indicator
}, 100);

messageList.addEventListener('scroll', handleScroll);
```

---

## Complete Mobile Chat Component

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, interactive-widget=resizes-content">
  <meta name="theme-color" content="#ffffff">
  <title>Mobile Chat</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    :root {
      --keyboard-height: 0px;
    }
    
    html, body {
      height: 100%;
      overscroll-behavior: none;
    }
    
    .chat-app {
      display: flex;
      flex-direction: column;
      height: 100vh;
      height: 100dvh;
    }
    
    .chat-header {
      padding: 0.75rem 1rem;
      padding-top: max(0.75rem, env(safe-area-inset-top));
      background: #fff;
      border-bottom: 1px solid #e5e7eb;
      font-weight: 600;
    }
    
    .message-list {
      flex: 1;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 1rem;
      padding-left: max(1rem, env(safe-area-inset-left));
      padding-right: max(1rem, env(safe-area-inset-right));
    }
    
    .message {
      max-width: 85%;
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
      border-radius: 1.25rem;
      line-height: 1.5;
    }
    
    .message.user {
      background: #3b82f6;
      color: white;
      margin-left: auto;
      border-bottom-right-radius: 0.25rem;
    }
    
    .message.ai {
      background: #f3f4f6;
      border-bottom-left-radius: 0.25rem;
    }
    
    .input-area {
      display: flex;
      align-items: flex-end;
      gap: 0.5rem;
      padding: 0.75rem 1rem;
      padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
      padding-left: max(0.75rem, env(safe-area-inset-left));
      padding-right: max(0.75rem, env(safe-area-inset-right));
      background: #fff;
      border-top: 1px solid #e5e7eb;
    }
    
    .input-container {
      flex: 1;
      display: flex;
      align-items: flex-end;
      background: #f3f4f6;
      border-radius: 1.5rem;
      padding: 0.5rem 1rem;
    }
    
    #message-input {
      flex: 1;
      border: none;
      background: transparent;
      font-family: inherit;
      font-size: 1rem;
      line-height: 1.5;
      resize: none;
      outline: none;
      max-height: 120px;
    }
    
    .send-button {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: #3b82f6;
      color: white;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    
    .send-button:active {
      background: #2563eb;
    }
    
    @media (min-width: 640px) {
      .message {
        max-width: 75%;
      }
    }
    
    @media (min-width: 1024px) {
      .chat-app {
        max-width: 768px;
        margin: 0 auto;
        border-left: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
      }
      
      .message {
        max-width: 65%;
      }
    }
  </style>
</head>
<body>
  <div class="chat-app">
    <header class="chat-header">
      AI Assistant
    </header>
    
    <div class="message-list" role="log" aria-live="polite">
      <div class="message user">How do I make this mobile friendly?</div>
      <div class="message ai">Use responsive design with flexible layouts, proper touch targets, and handle the virtual keyboard...</div>
    </div>
    
    <div class="input-area">
      <div class="input-container">
        <textarea 
          id="message-input" 
          rows="1" 
          placeholder="Message..."
          aria-label="Type a message"
        ></textarea>
      </div>
      <button class="send-button" aria-label="Send">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z"/>
        </svg>
      </button>
    </div>
  </div>
  
  <script>
    const textarea = document.querySelector('#message-input');
    const sendButton = document.querySelector('.send-button');
    const messageList = document.querySelector('.message-list');
    
    // Auto-grow textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    });
    
    // Handle keyboard
    if ('visualViewport' in window) {
      window.visualViewport.addEventListener('resize', () => {
        const keyboardHeight = window.innerHeight - window.visualViewport.height;
        document.documentElement.style.setProperty(
          '--keyboard-height', 
          keyboardHeight + 'px'
        );
      });
    }
    
    // Send message
    function sendMessage() {
      const text = textarea.value.trim();
      if (!text) return;
      
      const message = document.createElement('div');
      message.className = 'message user';
      message.textContent = text;
      messageList.appendChild(message);
      
      textarea.value = '';
      textarea.style.height = 'auto';
      message.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
    
    sendButton.addEventListener('click', sendMessage);
    
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `100dvh` for viewport height | Rely on `100vh` alone |
| Make touch targets at least 44×44px | Use tiny buttons |
| Handle safe area insets | Ignore notches and home indicators |
| Use `overscroll-behavior: contain` | Allow pull-to-refresh in chat |
| Test on real devices | Only test in browser DevTools |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Layout breaks when keyboard opens | Use Visual Viewport API |
| Content hidden behind notch | Add `env(safe-area-inset-*)` |
| Buttons too small to tap | Minimum 44×44px touch targets |
| Rubber-banding scrolling | Use `overscroll-behavior: contain` |
| Text too small on mobile | Use 16px minimum font size |

---

## Summary

✅ **Mobile-first design** starts with small screens and enhances for larger  
✅ **Touch targets** of 44×44px minimum prevent frustrating mis-taps  
✅ **Virtual keyboard handling** keeps input visible and accessible  
✅ **Safe area insets** account for notches and home indicators  
✅ **Performance optimizations** ensure smooth scrolling on mobile

---

## Further Reading

- [Visual Viewport API](https://developer.mozilla.org/en-US/docs/Web/API/Visual_Viewport_API)
- [Safe Area Insets](https://developer.mozilla.org/en-US/docs/Web/CSS/env)
- [Apple HIG: Touch Targets](https://developer.apple.com/design/human-interface-guidelines/accessibility)
- [Dynamic Viewport Units](https://web.dev/blog/viewport-units)

---

**Previous:** [Accessibility Considerations](./05-accessibility-considerations.md)  
**Next:** [Empty States & Onboarding](./07-empty-states-onboarding.md)

<!-- 
Sources Consulted:
- MDN Visual Viewport API: https://developer.mozilla.org/en-US/docs/Web/API/Visual_Viewport_API
- MDN env(): https://developer.mozilla.org/en-US/docs/Web/CSS/env
- web.dev Viewport Units: https://web.dev/blog/viewport-units
- Apple HIG: https://developer.apple.com/design/human-interface-guidelines/
-->
