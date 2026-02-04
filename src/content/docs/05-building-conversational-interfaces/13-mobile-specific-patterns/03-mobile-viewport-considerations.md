---
title: "Mobile Viewport Considerations"
---

# Mobile Viewport Considerations

## Introduction

Modern smartphones have evolved far beyond simple rectangular screens. Today's devices feature notches, Dynamic Islands, rounded corners, home indicators, and edge-to-edge displays that intrude into the traditional viewport area. Understanding how to work with these display features is essential for creating chat interfaces that feel truly native.

This lesson explores safe area insets, dynamic viewport units, and orientation handling—the tools that let your chat interface adapt to any mobile device's unique screen geometry.

### What We'll Cover

- Safe area insets and the `env()` CSS function
- Handling notches, Dynamic Island, and rounded corners
- Dynamic viewport height units (`dvh`, `svh`, `lvh`)
- Managing orientation changes smoothly
- Browser chrome considerations

### Prerequisites

- Solid understanding of CSS viewport units (`vh`, `vw`)
- Familiarity with CSS custom properties
- Basic responsive design knowledge

---

## Understanding Safe Areas

Safe areas define regions of the screen where content can be displayed without being obscured by hardware features or system UI.

### What Can Obstruct Content?

| Feature | Description | Affected Area |
|---------|-------------|---------------|
| **Notch** | Camera/sensor cutout | Top center |
| **Dynamic Island** | iPhone 14+ animated pill | Top center |
| **Rounded corners** | Display edge curves | All corners |
| **Home indicator** | iOS home gesture bar | Bottom |
| **Status bar** | Time, battery, signal | Top |
| **Navigation bar** | Android back/home/recent | Bottom |

### Safe Area Visualization

```
┌────────────────────────────────┐
│▒▒▒▒▒▒▒▒▒▒[NOTCH]▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← safe-area-inset-top
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░                             ░│
│░      SAFE AREA              ░│
│░      Content goes here      ░│
│░                             ░│
│░                             ░│
│░                             ░│
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← safe-area-inset-bottom
│▒▒▒▒▒▒▒▒[HOME INDICATOR]▒▒▒▒▒▒▒│   (home indicator area)
└────────────────────────────────┘
 ↑                              ↑
 safe-area-inset-left           safe-area-inset-right
 (minimal on portrait)          (minimal on portrait)
```

---

## The `env()` CSS Function

The `env()` function provides access to environment variables defined by the user agent. For mobile safe areas, this includes inset values:

### Safe Area Environment Variables

| Variable | Description | Typical Use |
|----------|-------------|-------------|
| `safe-area-inset-top` | Distance from top edge to safe area | Header padding |
| `safe-area-inset-right` | Distance from right edge | Landscape layout |
| `safe-area-inset-bottom` | Distance from bottom edge | Input area, navigation |
| `safe-area-inset-left` | Distance from left edge | Landscape layout |

### Basic Usage

```css
.chat-container {
  /* Reserve space for safe areas */
  padding-top: env(safe-area-inset-top, 0px);
  padding-right: env(safe-area-inset-right, 0px);
  padding-bottom: env(safe-area-inset-bottom, 0px);
  padding-left: env(safe-area-inset-left, 0px);
}
```

> **Note:** The second parameter (`0px`) is a fallback value used when the environment variable isn't available.

### Enabling Safe Area Support

For `env()` to work, you must set the viewport meta tag properly:

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
```

| Viewport Fit Value | Behavior |
|--------------------|----------|
| `auto` (default) | Content constrained to safe area |
| `contain` | Same as `auto` |
| `cover` | Content extends to edges, you handle safe areas |

> **Warning:** Using `viewport-fit=cover` without handling safe areas will cause content to be hidden behind notches and home indicators.

---

## Implementing Safe Area Handling

### Header with Safe Area

```css
.chat-header {
  position: sticky;
  top: 0;
  z-index: 100;
  
  /* Add safe area padding to existing padding */
  padding-top: calc(12px + env(safe-area-inset-top, 0px));
  padding-left: calc(16px + env(safe-area-inset-left, 0px));
  padding-right: calc(16px + env(safe-area-inset-right, 0px));
  
  background: linear-gradient(to bottom, 
    #007AFF 0%, 
    #007AFF calc(100% - 1px), 
    rgba(0, 122, 255, 0.8) 100%
  );
  
  /* Extend background into safe area */
  margin-top: calc(-1 * env(safe-area-inset-top, 0px));
  padding-top: calc(12px + env(safe-area-inset-top, 0px));
}
```

### Input Area with Home Indicator

The input area is most critical—it must remain above the home indicator:

```css
.input-area {
  position: sticky;
  bottom: 0;
  
  display: flex;
  gap: 8px;
  
  /* Content padding */
  padding: 12px 16px;
  
  /* Add safe area insets */
  padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
  padding-left: calc(16px + env(safe-area-inset-left, 0px));
  padding-right: calc(16px + env(safe-area-inset-right, 0px));
  
  background: white;
  border-top: 1px solid #e0e0e0;
}
```

### Full-Screen Chat Layout

```css
.chat-app {
  display: flex;
  flex-direction: column;
  
  /* Use dynamic viewport height */
  min-height: 100dvh;
  
  /* Fallback for older browsers */
  min-height: 100vh;
  min-height: -webkit-fill-available;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  
  /* Horizontal safe area for landscape */
  padding-left: calc(16px + env(safe-area-inset-left, 0px));
  padding-right: calc(16px + env(safe-area-inset-right, 0px));
}
```

---

## Dynamic Viewport Units

Traditional viewport units (`vh`, `vw`) don't account for mobile browser UI changes. New viewport units solve this:

### The Three Viewport Sizes

| Viewport Type | Description | Use Case |
|---------------|-------------|----------|
| **Large Viewport** (`lvh`, `lvw`) | When browser UI is minimized | Fullscreen content after scroll |
| **Small Viewport** (`svh`, `svw`) | When browser UI is fully visible | Guaranteed visible area |
| **Dynamic Viewport** (`dvh`, `dvw`) | Current actual viewport | Live-updating layout |

### Visual Comparison

```
Browser UI Visible (Initial Load):
┌─────────────────────────────┐
│       Address Bar           │
├─────────────────────────────┤
│                             │
│    100svh = 100dvh          │ ← Small and dynamic equal
│    (smaller)                │
│                             │
├─────────────────────────────┤
│                             │
│    Extra space for          │
│    100lvh (larger)          │
│                             │
└─────────────────────────────┘

After Scrolling (UI Hidden):
┌─────────────────────────────┐
│                             │
│                             │
│    100lvh = 100dvh          │ ← Large and dynamic equal
│    (full screen)            │
│                             │
│                             │
│                             │
│                             │
│                             │
└─────────────────────────────┘
```

### Practical Usage

```css
/* Use dynamic viewport height for chat container */
.chat-container {
  height: 100dvh;
  
  /* Fallbacks for older browsers */
  height: 100vh;
  height: -webkit-fill-available;
}

/* Use small viewport for modals (always fits) */
.modal {
  max-height: 90svh;
}

/* Use large viewport for content behind headers */
.fullscreen-media {
  height: 100lvh;
}
```

### Chat App with Dynamic Height

```css
.chat-app {
  display: flex;
  flex-direction: column;
  
  /* Dynamic height adapts as browser chrome hides/shows */
  height: 100dvh;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  
  /* Enable smooth scrolling */
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
}

.input-area {
  flex-shrink: 0;
  padding: 12px;
  padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
}
```

### Browser Support

| Unit | Chrome | Safari | Firefox |
|------|--------|--------|---------|
| `dvh`, `dvw` | 108+ | 15.4+ | 101+ |
| `svh`, `svw` | 108+ | 15.4+ | 101+ |
| `lvh`, `lvw` | 108+ | 15.4+ | 101+ |
| `dvmax`, `dvmin` | 108+ | 15.4+ | 101+ |

### Fallback Strategy

```css
.chat-container {
  /* Level 1: Old browsers */
  height: 100vh;
  
  /* Level 2: iOS Safari quirk workaround */
  height: -webkit-fill-available;
  
  /* Level 3: Modern browsers with dynamic units */
  height: 100dvh;
}

/* JavaScript fallback for very old browsers */
@supports not (height: 100dvh) {
  .chat-container {
    height: calc(var(--viewport-height, 100vh));
  }
}
```

```javascript
// JavaScript fallback for browsers without dvh support
function setViewportHeight() {
  const vh = window.innerHeight * 0.01;
  document.documentElement.style.setProperty('--viewport-height', `${window.innerHeight}px`);
  document.documentElement.style.setProperty('--vh', `${vh}px`);
}

setViewportHeight();
window.addEventListener('resize', setViewportHeight);
```

---

## Handling Notches and Rounded Corners

### Notch-Aware Header

```css
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  
  /* Minimum height for touch targets */
  min-height: 44px;
  
  /* Add notch padding */
  padding-top: env(safe-area-inset-top, 0px);
  
  /* Horizontal padding for landscape notch */
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  
  background: white;
}
```

### Corner-Safe Fixed Elements

Rounded display corners can clip fixed position elements:

```css
.floating-action-button {
  position: fixed;
  
  /* Stay clear of rounded corners */
  bottom: calc(24px + env(safe-area-inset-bottom, 0px));
  right: calc(24px + env(safe-area-inset-right, 0px));
  
  width: 56px;
  height: 56px;
  border-radius: 50%;
}

.toast-notification {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  
  /* Stay below any notch/Dynamic Island */
  top: calc(60px + env(safe-area-inset-top, 0px));
  
  max-width: calc(100% - 32px - env(safe-area-inset-left, 0px) - env(safe-area-inset-right, 0px));
}
```

### Dynamic Island Consideration

iPhone 14 Pro and later have the Dynamic Island, which is larger than previous notches:

```css
.header-content {
  /* Provide extra room for Dynamic Island */
  padding-top: max(env(safe-area-inset-top, 0px), 12px);
  
  /* Center content below the island */
  text-align: center;
}

/* On phones with Dynamic Island, add more top padding */
@supports (padding: max(0px)) {
  .header-content {
    padding-top: max(env(safe-area-inset-top, 0px), 16px);
  }
}
```

---

## Orientation Changes

When the device rotates, safe area insets change dramatically:

### Portrait vs. Landscape Safe Areas

```
PORTRAIT:                    LANDSCAPE:
┌──────────────┐            ┌─────────────────────────────┐
│ ▓▓[NOTCH]▓▓  │            │░░│                      │▓▓│
│ ░            │            │░░│                      │▓▓│
│ ░  Content   │            │░░│      Content         │▓▓│
│ ░            │            │░░│                      │▓▓│
│ ░            │            │░░│                      │▓▓│
│ ▒▒▒▒▒▒▒▒▒▒▒▒ │            └─────────────────────────────┘
└──────────────┘             ↑                         ↑
                            Left inset             Right inset
                            (large)                (large)
```

### Orientation-Responsive Layout

```css
.chat-app {
  display: flex;
  flex-direction: column;
  min-height: 100dvh;
  
  /* Safe area padding */
  padding-left: env(safe-area-inset-left, 0px);
  padding-right: env(safe-area-inset-right, 0px);
}

/* Landscape optimizations */
@media (orientation: landscape) {
  .chat-app {
    flex-direction: row;
  }
  
  .sidebar {
    width: 320px;
    flex-shrink: 0;
  }
  
  .chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
  }
  
  .input-area {
    /* In landscape, bottom inset is usually minimal */
    padding-bottom: max(12px, env(safe-area-inset-bottom, 0px));
  }
}

/* Handle left notch in landscape (left-handed mode) */
@media (orientation: landscape) {
  .message-list {
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
}
```

### Detecting Orientation Changes

```javascript
class OrientationHandler {
  constructor() {
    this.orientation = this.getOrientation();
    
    // Modern API
    if (screen.orientation) {
      screen.orientation.addEventListener('change', () => {
        this.handleOrientationChange();
      });
    }
    
    // Fallback
    window.addEventListener('orientationchange', () => {
      // Wait for rotation animation
      setTimeout(() => this.handleOrientationChange(), 100);
    });
    
    // Also handle resize (catches some edge cases)
    window.addEventListener('resize', () => {
      const newOrientation = this.getOrientation();
      if (newOrientation !== this.orientation) {
        this.orientation = newOrientation;
        this.handleOrientationChange();
      }
    });
  }
  
  getOrientation() {
    if (screen.orientation) {
      return screen.orientation.type.includes('portrait') ? 'portrait' : 'landscape';
    }
    return window.innerHeight > window.innerWidth ? 'portrait' : 'landscape';
  }
  
  handleOrientationChange() {
    this.orientation = this.getOrientation();
    
    document.body.classList.remove('portrait', 'landscape');
    document.body.classList.add(this.orientation);
    
    // Recalculate any JavaScript-managed layouts
    this.recalculateLayout();
  }
  
  recalculateLayout() {
    // Force scroll position recalculation
    const messageList = document.querySelector('.message-list');
    if (messageList) {
      // Scroll to maintain position after rotation
      requestAnimationFrame(() => {
        messageList.scrollTop = messageList.scrollTop;
      });
    }
  }
}
```

---

## Browser Chrome Considerations

Mobile browsers have their own UI (chrome) that affects available space:

### Address Bar Behavior

```javascript
// Detect if browser chrome is visible
function isBrowserChromeVisible() {
  // Compare visual viewport to window inner height
  if (window.visualViewport) {
    return window.visualViewport.height < window.innerHeight * 0.95;
  }
  return false;
}
```

### Preventing Address Bar Show/Hide

For some chat apps, you might want consistent viewport:

```css
/* Prevent body scroll (can prevent address bar hiding) */
html, body {
  overflow: hidden;
  position: fixed;
  width: 100%;
  height: 100%;
}

.chat-app {
  height: 100%;
  overflow: hidden;
}

.message-list {
  overflow-y: auto;
}
```

### Working With Address Bar Changes

```javascript
// Track address bar show/hide
let lastViewportHeight = window.visualViewport?.height || window.innerHeight;

function handleViewportChange() {
  const currentHeight = window.visualViewport?.height || window.innerHeight;
  const heightDifference = lastViewportHeight - currentHeight;
  
  // Significant height change not caused by keyboard
  if (Math.abs(heightDifference) > 50 && Math.abs(heightDifference) < 150) {
    console.log('Browser chrome changed');
    // Address bar appeared or disappeared
  }
  
  lastViewportHeight = currentHeight;
}

if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', handleViewportChange);
}
```

---

## Complete Safe Area Implementation

Here's a comprehensive implementation for a chat interface:

```css
/* Reset and base setup */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  /* Prevent iOS text size adjustment */
  -webkit-text-size-adjust: 100%;
}

/* Chat app container */
.chat-app {
  display: flex;
  flex-direction: column;
  
  /* Dynamic viewport height with fallbacks */
  height: 100vh;
  height: -webkit-fill-available;
  height: 100dvh;
  
  /* Extend to full screen edges */
  overflow: hidden;
  
  /* Base safe area handling */
  padding-left: env(safe-area-inset-left, 0px);
  padding-right: env(safe-area-inset-right, 0px);
}

/* Header */
.chat-header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  
  min-height: 56px;
  padding: 12px 16px;
  
  /* Notch handling */
  padding-top: calc(12px + env(safe-area-inset-top, 0px));
  
  background: #007AFF;
  color: white;
}

/* Message list */
.message-list {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  
  /* Smooth scrolling */
  -webkit-overflow-scrolling: touch;
  scroll-behavior: smooth;
  
  /* Content padding */
  padding: 16px;
}

/* Input area */
.input-area {
  flex-shrink: 0;
  display: flex;
  align-items: flex-end;
  gap: 8px;
  
  padding: 12px 16px;
  
  /* Home indicator handling */
  padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
  
  background: white;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

/* Input field */
.chat-input {
  flex: 1;
  min-height: 44px;
  max-height: 120px;
  
  padding: 10px 16px;
  
  border: 1px solid #e0e0e0;
  border-radius: 22px;
  
  font-size: 16px; /* Prevents iOS zoom */
  line-height: 1.4;
  
  resize: none;
  outline: none;
}

.chat-input:focus {
  border-color: #007AFF;
}

/* Send button */
.send-button {
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  
  border: none;
  border-radius: 50%;
  
  background: #007AFF;
  color: white;
  
  font-size: 20px;
  cursor: pointer;
}

/* Landscape adjustments */
@media (orientation: landscape) {
  .chat-header {
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
  
  .message-list {
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
  
  .input-area {
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `viewport-fit=cover` and handle safe areas | Ignore safe area insets |
| Use `dvh` for main container height | Use fixed `100vh` on mobile |
| Add `env()` with fallback values | Assume `env()` always works |
| Test in both portrait and landscape | Only test portrait orientation |
| Use `max()` to ensure minimum padding | Let content touch edges |
| Provide CSS fallbacks for older browsers | Require newest browser features |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Content hidden behind notch | Add `env(safe-area-inset-top)` padding |
| Input covered by home indicator | Add `env(safe-area-inset-bottom)` |
| Fixed elements clipped by corners | Position away from screen edges |
| Layout breaks on rotation | Handle safe area changes in both orientations |
| `100vh` leaves gap on iOS Safari | Use `100dvh` or `-webkit-fill-available` |
| `env()` not working | Check `viewport-fit=cover` in meta tag |

---

## Hands-on Exercise

### Your Task

Create a chat interface that properly handles:

1. **Safe area insets** — Content never hidden by notches or home indicators
2. **Dynamic viewport height** — Uses `dvh` with fallbacks
3. **Orientation changes** — Works correctly in portrait and landscape
4. **Edge-to-edge design** — Background extends to edges, content stays safe

### Requirements

1. Use `viewport-fit=cover` in meta tag
2. Apply safe area insets to header, messages, and input
3. Test with different device simulators (iPhone, Android)
4. Handle landscape orientation with left/right insets

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `max()` to ensure minimum padding even when safe area is 0
- The message list needs horizontal safe area padding in landscape
- Test in Safari on iOS—it has the strictest behavior
- Chrome DevTools has a "Device frame" option to see notch simulation

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Safe Area Chat</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    html, body {
      height: 100%;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .chat-app {
      display: flex;
      flex-direction: column;
      
      /* Viewport height with fallbacks */
      height: 100vh;
      height: -webkit-fill-available;
      height: 100dvh;
      
      /* Edge-to-edge background */
      background: #f5f5f5;
    }
    
    .header {
      display: flex;
      align-items: center;
      gap: 12px;
      
      min-height: 56px;
      padding: 12px 16px;
      
      /* Safe area: notch + status bar */
      padding-top: calc(12px + env(safe-area-inset-top, 20px));
      padding-left: max(16px, env(safe-area-inset-left, 0px));
      padding-right: max(16px, env(safe-area-inset-right, 0px));
      
      background: #007AFF;
      color: white;
    }
    
    .header h1 {
      font-size: 17px;
      font-weight: 600;
    }
    
    .messages {
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      
      padding: 16px;
      padding-left: max(16px, env(safe-area-inset-left, 0px));
      padding-right: max(16px, env(safe-area-inset-right, 0px));
    }
    
    .message {
      max-width: 80%;
      margin-bottom: 12px;
      padding: 10px 14px;
      border-radius: 18px;
      background: white;
    }
    
    .message.sent {
      margin-left: auto;
      background: #007AFF;
      color: white;
    }
    
    .input-area {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      
      padding: 12px 16px;
      
      /* Safe area: home indicator */
      padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
      padding-left: max(16px, env(safe-area-inset-left, 0px));
      padding-right: max(16px, env(safe-area-inset-right, 0px));
      
      background: white;
      border-top: 1px solid #e0e0e0;
    }
    
    .input {
      flex: 1;
      min-height: 44px;
      padding: 10px 16px;
      
      border: 1px solid #e0e0e0;
      border-radius: 22px;
      
      font-size: 16px;
      outline: none;
    }
    
    .input:focus {
      border-color: #007AFF;
    }
    
    .send-btn {
      width: 44px;
      height: 44px;
      
      border: none;
      border-radius: 50%;
      
      background: #007AFF;
      color: white;
      font-size: 20px;
    }
    
    /* Landscape optimization */
    @media (orientation: landscape) and (max-height: 500px) {
      .header {
        min-height: 48px;
        padding-top: calc(8px + env(safe-area-inset-top, 0px));
        padding-bottom: 8px;
      }
      
      .messages {
        padding: 12px;
        padding-left: max(12px, env(safe-area-inset-left, 0px));
        padding-right: max(12px, env(safe-area-inset-right, 0px));
      }
      
      .input-area {
        padding: 8px 12px;
        padding-bottom: calc(8px + env(safe-area-inset-bottom, 0px));
      }
      
      .input {
        min-height: 36px;
        padding: 6px 14px;
      }
      
      .send-btn {
        width: 36px;
        height: 36px;
        font-size: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="chat-app">
    <header class="header">
      <button style="background:none;border:none;color:white;font-size:24px;">←</button>
      <h1>AI Assistant</h1>
    </header>
    
    <div class="messages">
      <div class="message">Hello! I'm an AI assistant. How can I help you today?</div>
      <div class="message sent">Hi! I need help with my project.</div>
      <div class="message">Of course! Tell me more about your project.</div>
    </div>
    
    <div class="input-area">
      <input type="text" class="input" placeholder="Message...">
      <button class="send-btn">➤</button>
    </div>
  </div>
</body>
</html>
```

</details>

---

## Summary

✅ Safe area insets protect content from notches, home indicators, and rounded corners  
✅ Use `viewport-fit=cover` in meta tag to enable full edge-to-edge design  
✅ `env()` provides access to safe area values with fallback support  
✅ Dynamic viewport units (`dvh`, `svh`, `lvh`) solve the mobile viewport height problem  
✅ Always test in both portrait and landscape orientations  
✅ Provide CSS fallbacks for browsers without modern viewport unit support  

**Next:** [Haptic Feedback](./04-haptic-feedback.md)

---

<!-- 
Sources Consulted:
- MDN env() CSS function: https://developer.mozilla.org/en-US/docs/Web/CSS/env
- MDN CSS length units: https://developer.mozilla.org/en-US/docs/Web/CSS/length
- Apple Human Interface Guidelines: Layout
- web.dev: New viewport units
-->
