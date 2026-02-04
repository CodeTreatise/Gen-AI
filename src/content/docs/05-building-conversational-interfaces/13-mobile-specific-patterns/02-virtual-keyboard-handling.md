---
title: "Virtual Keyboard Handling"
---

# Virtual Keyboard Handling

## Introduction

The virtual keyboard is both a blessing and a curse for mobile chat interfaces. It enables text input on touchscreen devices, but it can consume up to 50% of the viewport height, dramatically reshaping your layout and potentially hiding critical UI elements like the message input area itself.

In this lesson, we'll explore how to detect, measure, and respond to virtual keyboard appearance so your chat interface remains functional and user-friendly regardless of keyboard state.

### What We'll Cover

- The Visual Viewport API for tracking actual visible area
- The VirtualKeyboard API for fine-grained control
- CSS environment variables for keyboard-aware layouts
- Input focus management and auto-scrolling
- Keyboard dismiss patterns

### Prerequisites

- Understanding of CSS viewport units and positioning
- Familiarity with JavaScript event handling
- Basic knowledge of CSS custom properties

---

## The Two Viewports Problem

Mobile browsers have two distinct viewport concepts:

| Viewport | Description | Changes When Keyboard Opens |
|----------|-------------|----------------------------|
| **Layout Viewport** | The full page size the browser uses for layout | Often stays the same |
| **Visual Viewport** | What's actually visible on screen | Shrinks to accommodate keyboard |

This distinction causes many layout issues. Elements positioned relative to the layout viewport may be hidden behind the keyboard, while elements following the visual viewport stay visible.

```
┌─────────────────────────┐
│    Layout Viewport      │
│  (doesn't change)       │
│                         │
│  ┌─────────────────┐    │
│  │ Visual Viewport │    │
│  │  (shrinks)      │    │
│  └─────────────────┘    │
│ ┌─────────────────────┐ │
│ │   Virtual Keyboard  │ │
│ │   (covers layout    │ │
│ │    viewport)        │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

---

## Visual Viewport API

The Visual Viewport API lets you track the actual visible portion of the page, including changes caused by keyboard appearance, pinch-zoom, and browser UI.

### Basic Usage

```javascript
// Access the visual viewport
const viewport = window.visualViewport;

// Current dimensions
console.log('Width:', viewport.width);
console.log('Height:', viewport.height);

// Offset from layout viewport (changes during zoom/pan)
console.log('Offset Left:', viewport.offsetLeft);
console.log('Offset Top:', viewport.offsetTop);

// Pinch-zoom scale
console.log('Scale:', viewport.scale);
```

### Listening for Changes

```javascript
function handleViewportChange() {
  const viewport = window.visualViewport;
  
  console.log(`Visual viewport: ${viewport.width}×${viewport.height}`);
  console.log(`Offset: (${viewport.offsetLeft}, ${viewport.offsetTop})`);
  
  // Detect if keyboard is likely open
  // (visual viewport height significantly less than window height)
  const keyboardOpen = window.innerHeight - viewport.height > 150;
  
  document.body.classList.toggle('keyboard-open', keyboardOpen);
}

// Listen for resize (keyboard open/close)
window.visualViewport.addEventListener('resize', handleViewportChange);

// Listen for scroll (panning when zoomed)
window.visualViewport.addEventListener('scroll', handleViewportChange);
```

### Keeping Input Visible

A common issue: when the input focuses and keyboard opens, the input field scrolls out of view. Fix this by tracking the visual viewport:

```javascript
class ChatInputManager {
  constructor(inputElement) {
    this.input = inputElement;
    this.container = document.querySelector('.chat-container');
    
    this.input.addEventListener('focus', () => this.handleFocus());
    
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => {
        if (document.activeElement === this.input) {
          this.adjustForKeyboard();
        }
      });
    }
  }
  
  handleFocus() {
    // Give keyboard time to appear
    setTimeout(() => this.adjustForKeyboard(), 300);
  }
  
  adjustForKeyboard() {
    const viewport = window.visualViewport;
    if (!viewport) return;
    
    // Calculate available space
    const availableHeight = viewport.height;
    
    // Set container height to match visible area
    this.container.style.height = `${availableHeight}px`;
    
    // Ensure input is visible
    this.input.scrollIntoView({ 
      behavior: 'smooth', 
      block: 'end' 
    });
  }
}
```

### Browser Compatibility

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| visualViewport | 61+ | 79+ | 91+ | 13+ |
| resize event | 62+ | 79+ | 91+ | 13+ |
| scroll event | 62+ | 79+ | 91+ | 13+ |

---

## VirtualKeyboard API

The VirtualKeyboard API (supported in Chromium browsers) provides direct control over keyboard behavior and geometry information.

> **Warning:** This API has limited browser support. Safari and Firefox do not support it. Always implement fallbacks.

### Opting Into Manual Control

By default, browsers resize the viewport when the keyboard appears. You can opt out and handle it yourself:

```javascript
if ('virtualKeyboard' in navigator) {
  // Take control of keyboard handling
  navigator.virtualKeyboard.overlaysContent = true;
  
  // Now the viewport won't resize - keyboard overlays content
  // You're responsible for adjusting layout
}
```

### Getting Keyboard Geometry

```javascript
if ('virtualKeyboard' in navigator) {
  navigator.virtualKeyboard.overlaysContent = true;
  
  navigator.virtualKeyboard.addEventListener('geometrychange', (event) => {
    const { x, y, width, height } = event.target.boundingRect;
    
    console.log(`Keyboard: ${width}×${height} at (${x}, ${y})`);
    
    if (height > 0) {
      // Keyboard is visible - adjust layout
      adjustLayoutForKeyboard(height);
    } else {
      // Keyboard is hidden
      resetLayout();
    }
  });
}

function adjustLayoutForKeyboard(keyboardHeight) {
  const chatInput = document.querySelector('.chat-input-area');
  chatInput.style.transform = `translateY(-${keyboardHeight}px)`;
}

function resetLayout() {
  const chatInput = document.querySelector('.chat-input-area');
  chatInput.style.transform = '';
}
```

### Programmatic Control

```javascript
// Show the keyboard programmatically
navigator.virtualKeyboard.show();

// Hide the keyboard
navigator.virtualKeyboard.hide();
```

This is useful for custom `contenteditable` elements or when you want to control keyboard timing.

---

## CSS Keyboard Environment Variables

The VirtualKeyboard API exposes CSS environment variables for keyboard geometry:

```css
.chat-container {
  /* Reserve space for keyboard */
  padding-bottom: env(keyboard-inset-height, 0px);
}

.chat-input-area {
  /* Position above keyboard */
  bottom: env(keyboard-inset-height, 0px);
}
```

### Available Variables

| Variable | Description |
|----------|-------------|
| `keyboard-inset-top` | Distance from viewport top to keyboard |
| `keyboard-inset-right` | Distance from viewport right edge |
| `keyboard-inset-bottom` | Distance from viewport bottom (usually 0) |
| `keyboard-inset-left` | Distance from viewport left edge |
| `keyboard-inset-width` | Width of keyboard |
| `keyboard-inset-height` | Height of keyboard |

### Complete Keyboard-Aware Layout

```css
/* Enable keyboard overlay mode */
.chat-app {
  /* Fill available space */
  display: flex;
  flex-direction: column;
  height: 100vh;
  height: 100dvh; /* Dynamic viewport height fallback */
}

.message-list {
  flex: 1;
  overflow-y: auto;
  /* Adjust scroll area when keyboard appears */
  padding-bottom: env(keyboard-inset-height, 0px);
}

.input-area {
  position: sticky;
  bottom: 0;
  background: white;
  padding: 12px;
  /* Move above keyboard */
  transform: translateY(calc(-1 * env(keyboard-inset-height, 0px)));
}
```

### Fallback for Unsupported Browsers

```javascript
// Feature detection
const supportsKeyboardAPI = 'virtualKeyboard' in navigator;
const supportsKeyboardEnv = CSS.supports('height', 'env(keyboard-inset-height, 0px)');

if (supportsKeyboardAPI && supportsKeyboardEnv) {
  // Use native CSS variables
  navigator.virtualKeyboard.overlaysContent = true;
} else {
  // Fallback: use Visual Viewport API
  setupVisualViewportFallback();
}

function setupVisualViewportFallback() {
  if (!window.visualViewport) return;
  
  window.visualViewport.addEventListener('resize', () => {
    const keyboardHeight = window.innerHeight - window.visualViewport.height;
    document.documentElement.style.setProperty(
      '--keyboard-height',
      `${Math.max(0, keyboardHeight)}px`
    );
  });
}
```

```css
/* Use custom property as fallback */
.input-area {
  padding-bottom: var(--keyboard-height, 0px);
}
```

---

## Input Focus Management

Managing focus properly is critical for chat input. Users expect the input to remain visible and usable when the keyboard opens.

### Auto-Focus on Load

```javascript
// Focus input when chat opens
function initializeChat() {
  const input = document.querySelector('.chat-input');
  
  // Delay slightly to ensure layout is stable
  requestAnimationFrame(() => {
    input.focus();
  });
}
```

### Scroll Input Into View

```javascript
class ChatInput {
  constructor(inputElement) {
    this.input = inputElement;
    
    this.input.addEventListener('focus', () => {
      this.scrollIntoViewSafely();
    });
  }
  
  scrollIntoViewSafely() {
    // Wait for keyboard animation
    setTimeout(() => {
      // Use scrollIntoViewIfNeeded for better behavior if available
      if (this.input.scrollIntoViewIfNeeded) {
        this.input.scrollIntoViewIfNeeded(true);
      } else {
        this.input.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }
    }, 300);
  }
}
```

### Preventing Layout Jumps

When keyboard opens, content often jumps. Smooth this with CSS:

```css
.chat-container {
  /* Animate height changes */
  transition: height 0.25s ease-out;
}

.message-list {
  /* Prevent scroll position from jumping */
  overflow-anchor: auto;
}

/* Anchor scrolling to newest message */
.message-list > .message:last-child {
  overflow-anchor: auto;
}

.message-list > .message:not(:last-child) {
  overflow-anchor: none;
}
```

### Maintaining Scroll Position

```javascript
class MessageListScroller {
  constructor(listElement) {
    this.list = listElement;
    this.wasAtBottom = true;
    
    this.list.addEventListener('scroll', () => {
      this.checkIfAtBottom();
    });
    
    // Restore scroll position after keyboard events
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => {
        if (this.wasAtBottom) {
          this.scrollToBottom();
        }
      });
    }
  }
  
  checkIfAtBottom() {
    const threshold = 50;
    this.wasAtBottom = 
      this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < threshold;
  }
  
  scrollToBottom() {
    this.list.scrollTo({
      top: this.list.scrollHeight,
      behavior: 'smooth'
    });
  }
}
```

---

## Keyboard Dismiss Patterns

Users need intuitive ways to dismiss the keyboard without using the device's back button.

### Dismiss on Outside Tap

```javascript
document.addEventListener('pointerdown', (event) => {
  const input = document.querySelector('.chat-input');
  const inputArea = document.querySelector('.input-area');
  
  // If tap is outside input area, blur the input
  if (!inputArea.contains(event.target)) {
    input.blur();
  }
});
```

### Dismiss on Scroll

```javascript
const messageList = document.querySelector('.message-list');
const chatInput = document.querySelector('.chat-input');

let scrollTimeout;

messageList.addEventListener('scroll', () => {
  // Dismiss keyboard if user starts scrolling
  if (document.activeElement === chatInput) {
    // Debounce to avoid dismissing on small scroll adjustments
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      chatInput.blur();
    }, 100);
  }
}, { passive: true });
```

### Send Button Behavior

After sending a message, decide whether to keep the keyboard open:

```javascript
class ChatSender {
  constructor(input, sendButton) {
    this.input = input;
    this.keepKeyboardOpen = true; // Chat apps typically keep keyboard open
    
    sendButton.addEventListener('click', () => this.send());
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });
  }
  
  send() {
    const message = this.input.value.trim();
    if (!message) return;
    
    sendMessage(message);
    this.input.value = '';
    
    if (this.keepKeyboardOpen) {
      // Keep focus - keyboard stays open
      this.input.focus();
    } else {
      // Blur input - keyboard dismisses
      this.input.blur();
    }
  }
}
```

### Done Button Handler

iOS shows a "Done" button in certain input modes. Handle it:

```javascript
const input = document.querySelector('.chat-input');

input.addEventListener('keydown', (event) => {
  // iOS Done button triggers blur, which we can handle
  if (event.key === 'Enter' && input.inputMode === 'text') {
    // Prevent default form submission
    event.preventDefault();
    
    // Send message or just dismiss
    if (input.value.trim()) {
      sendMessage(input.value);
      input.value = '';
    }
  }
});
```

---

## Handling Different Input Modes

The `inputmode` attribute changes which keyboard layout appears:

```html
<!-- Standard text keyboard -->
<input type="text" inputmode="text">

<!-- Numeric keypad -->
<input type="text" inputmode="numeric">

<!-- Email keyboard (with @ and .com) -->
<input type="email" inputmode="email">

<!-- URL keyboard (with / and .com) -->
<input type="url" inputmode="url">

<!-- Telephone keypad -->
<input type="tel" inputmode="tel">

<!-- Search keyboard (with search key) -->
<input type="search" inputmode="search">
```

For chat interfaces, use the default `text` inputmode but consider:

```html
<!-- Enable emoji suggestions on modern keyboards -->
<input type="text" 
       inputmode="text" 
       autocomplete="off" 
       autocorrect="on" 
       autocapitalize="sentences"
       spellcheck="true">
```

---

## Complete Implementation Example

Here's a complete keyboard-aware chat input implementation:

```javascript
class KeyboardAwareChatInput {
  constructor(options) {
    this.container = document.querySelector(options.container);
    this.messageList = document.querySelector(options.messageList);
    this.inputArea = document.querySelector(options.inputArea);
    this.input = document.querySelector(options.input);
    
    this.keyboardHeight = 0;
    this.wasAtBottom = true;
    
    this.init();
  }
  
  init() {
    // Try VirtualKeyboard API first
    if ('virtualKeyboard' in navigator) {
      this.useVirtualKeyboardAPI();
    } else if (window.visualViewport) {
      this.useVisualViewportAPI();
    }
    
    this.setupScrollBehavior();
    this.setupDismissPatterns();
  }
  
  useVirtualKeyboardAPI() {
    navigator.virtualKeyboard.overlaysContent = true;
    
    navigator.virtualKeyboard.addEventListener('geometrychange', (event) => {
      this.keyboardHeight = event.target.boundingRect.height;
      this.updateLayout();
    });
  }
  
  useVisualViewportAPI() {
    const viewport = window.visualViewport;
    
    viewport.addEventListener('resize', () => {
      // Estimate keyboard height from viewport change
      const viewportHeight = viewport.height;
      const windowHeight = window.innerHeight;
      
      this.keyboardHeight = Math.max(0, windowHeight - viewportHeight - viewport.offsetTop);
      this.updateLayout();
    });
  }
  
  updateLayout() {
    // Update CSS custom property for keyboard height
    document.documentElement.style.setProperty(
      '--keyboard-height',
      `${this.keyboardHeight}px`
    );
    
    // If user was at bottom, keep them there
    if (this.wasAtBottom && this.keyboardHeight > 0) {
      requestAnimationFrame(() => {
        this.scrollToBottom();
      });
    }
  }
  
  setupScrollBehavior() {
    this.messageList.addEventListener('scroll', () => {
      const threshold = 50;
      this.wasAtBottom = 
        this.messageList.scrollHeight - 
        this.messageList.scrollTop - 
        this.messageList.clientHeight < threshold;
    }, { passive: true });
  }
  
  scrollToBottom() {
    this.messageList.scrollTo({
      top: this.messageList.scrollHeight,
      behavior: 'smooth'
    });
  }
  
  setupDismissPatterns() {
    // Dismiss on outside tap
    this.messageList.addEventListener('pointerdown', () => {
      if (document.activeElement === this.input && this.keyboardHeight > 0) {
        this.input.blur();
      }
    });
  }
}
```

```css
:root {
  --keyboard-height: 0px;
}

.chat-container {
  display: flex;
  flex-direction: column;
  height: 100dvh;
  height: 100vh; /* Fallback */
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  /* Adjust for keyboard */
  padding-bottom: calc(16px + var(--keyboard-height, 0px));
}

.input-area {
  position: sticky;
  bottom: 0;
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  background: white;
  border-top: 1px solid #e0e0e0;
  /* Move above keyboard */
  transform: translateY(calc(-1 * var(--keyboard-height, 0px)));
}

.chat-input {
  flex: 1;
  min-height: 44px;
  padding: 10px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 22px;
  font-size: 16px; /* Prevents iOS zoom */
  resize: none;
}

.send-button {
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: #007AFF;
  color: white;
  font-size: 20px;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use Visual Viewport API for cross-platform support | Rely only on VirtualKeyboard API |
| Keep the input visible when keyboard opens | Let keyboard hide the input |
| Maintain scroll position when keyboard appears | Let content jump unexpectedly |
| Provide intuitive keyboard dismiss patterns | Require back button for dismissal |
| Use `font-size: 16px` or larger to prevent iOS zoom | Use small font sizes in inputs |
| Test on real devices | Trust browser DevTools completely |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `100vh` for container height | Use `100dvh` (dynamic viewport height) |
| Not detecting keyboard state | Use Visual Viewport API `resize` event |
| Layout jumps when keyboard opens | Animate transitions, use CSS custom properties |
| Safari not working | Safari doesn't support VirtualKeyboard API; use Visual Viewport |
| Input zooms on iOS | Ensure `font-size` is at least 16px |

---

## Hands-on Exercise

### Your Task

Create a chat input that remains fully functional and visible when the virtual keyboard appears:

1. **Keyboard detection** — Detect when keyboard opens/closes
2. **Input stays visible** — Input area remains above keyboard
3. **Scroll preservation** — Message list maintains scroll position
4. **Dismiss patterns** — Keyboard dismisses on outside tap or scroll

### Requirements

1. Use Visual Viewport API (works cross-browser)
2. Implement VirtualKeyboard API as progressive enhancement
3. Handle both portrait and landscape orientations
4. Ensure input has 16px+ font size (prevents iOS zoom)

<details>
<summary>💡 Hints (click to expand)</summary>

- Store keyboard height in a CSS custom property for easy access
- Use `requestAnimationFrame` when updating layout to avoid jank
- The keyboard takes ~300ms to animate open/closed
- Test with hardware keyboard connected (keyboard height will be 0)

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    :root {
      --keyboard-height: 0px;
    }
    
    body {
      font-family: -apple-system, sans-serif;
    }
    
    .chat-app {
      display: flex;
      flex-direction: column;
      height: 100dvh;
      height: 100vh;
    }
    
    .header {
      padding: 16px;
      background: #007AFF;
      color: white;
      text-align: center;
    }
    
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      background: #f5f5f5;
    }
    
    .message {
      background: white;
      padding: 12px 16px;
      border-radius: 16px;
      margin-bottom: 8px;
      max-width: 80%;
    }
    
    .input-area {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      padding-bottom: calc(12px + var(--keyboard-height, 0px) + env(safe-area-inset-bottom, 0px));
      background: white;
      border-top: 1px solid #e0e0e0;
      transition: padding-bottom 0.25s ease-out;
    }
    
    .chat-input {
      flex: 1;
      min-height: 44px;
      padding: 10px 16px;
      border: 1px solid #e0e0e0;
      border-radius: 22px;
      font-size: 16px;
      outline: none;
    }
    
    .chat-input:focus {
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
      cursor: pointer;
    }
    
    .send-btn:active {
      opacity: 0.8;
    }
  </style>
</head>
<body>
  <div class="chat-app">
    <div class="header">Chat App</div>
    <div class="messages" id="messages">
      <div class="message">Hello! Try focusing the input below.</div>
      <div class="message">The input should stay visible above the keyboard.</div>
    </div>
    <div class="input-area">
      <input type="text" class="chat-input" id="chatInput" 
             placeholder="Type a message..." 
             autocomplete="off">
      <button class="send-btn" id="sendBtn">➤</button>
    </div>
  </div>
  
  <script>
    const messages = document.getElementById('messages');
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    
    let wasAtBottom = true;
    
    // Track if user is at bottom
    messages.addEventListener('scroll', () => {
      const threshold = 50;
      wasAtBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < threshold;
    }, { passive: true });
    
    function scrollToBottom() {
      messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });
    }
    
    function updateKeyboardHeight(height) {
      document.documentElement.style.setProperty('--keyboard-height', `${height}px`);
      
      if (wasAtBottom && height > 0) {
        requestAnimationFrame(scrollToBottom);
      }
    }
    
    // Try VirtualKeyboard API
    if ('virtualKeyboard' in navigator) {
      navigator.virtualKeyboard.overlaysContent = true;
      navigator.virtualKeyboard.addEventListener('geometrychange', (e) => {
        updateKeyboardHeight(e.target.boundingRect.height);
      });
    } 
    // Fallback to Visual Viewport
    else if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => {
        const keyboardHeight = window.innerHeight - window.visualViewport.height;
        updateKeyboardHeight(Math.max(0, keyboardHeight));
      });
    }
    
    // Dismiss on message list tap
    messages.addEventListener('pointerdown', () => {
      if (document.activeElement === input) {
        input.blur();
      }
    });
    
    // Send message
    function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      const msg = document.createElement('div');
      msg.className = 'message';
      msg.textContent = text;
      messages.appendChild(msg);
      
      input.value = '';
      scrollToBottom();
      input.focus();
    }
    
    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ Mobile browsers have layout and visual viewports—understand the difference  
✅ Visual Viewport API tracks actual visible area and works cross-browser  
✅ VirtualKeyboard API provides fine-grained control but only in Chromium  
✅ CSS `env(keyboard-inset-height)` enables declarative keyboard-aware layouts  
✅ Always keep the input field visible when keyboard opens  
✅ Implement intuitive keyboard dismiss patterns (outside tap, scroll)  

**Next:** [Mobile Viewport Considerations](./03-mobile-viewport-considerations.md)

---

<!-- 
Sources Consulted:
- MDN Visual Viewport API: https://developer.mozilla.org/en-US/docs/Web/API/VisualViewport
- MDN VirtualKeyboard API: https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard_API
- Chrome Developers VirtualKeyboard: https://developer.chrome.com/docs/web-platform/virtual-keyboard/
- MDN env() CSS function: https://developer.mozilla.org/en-US/docs/Web/CSS/env
-->
