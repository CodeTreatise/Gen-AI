---
title: "Touch-Optimized Interfaces"
---

# Touch-Optimized Interfaces

## Introduction

Touch interaction is fundamentally different from mouse-based interfaces. While a mouse cursor has pixel-perfect precision, a finger touch covers approximately 7mm×7mm of screen area—and users can't see exactly where they're touching because their finger obscures the target.

Building touch-optimized chat interfaces requires rethinking interaction patterns from the ground up: larger touch targets, gesture-based navigation, context menus triggered by long-press, and visual feedback that confirms every interaction.

### What We'll Cover

- Touch target sizing guidelines from Apple and Material Design
- Using Pointer Events for unified input handling
- Implementing swipe gestures for message actions
- Long-press context menus
- Visual and CSS touch feedback

### Prerequisites

- Understanding of JavaScript event handling
- Familiarity with CSS transitions and transforms
- Basic DOM manipulation knowledge

---

## Touch Target Sizes

The most critical accessibility and usability factor in mobile interfaces is touch target size. Both Apple's Human Interface Guidelines and Google's Material Design specify minimum sizes:

| Platform | Minimum Size | Recommended |
|----------|--------------|-------------|
| Apple HIG | 44×44 points | 44×44 points |
| Material Design | 48×48 dp | 48×48 dp |
| WCAG 2.2 | 24×24 CSS pixels | 44×44 CSS pixels |

> **🔑 Key Concept:** A "point" on iOS and "dp" on Android both equal 1 CSS pixel at 1x scale. On a 2x or 3x display, they're rendered at higher physical pixel counts but remain the same CSS size.

### Implementing Proper Touch Targets

Even if your visual element is smaller, the touchable area should meet minimum requirements:

```css
/* Chat message action button */
.message-action-btn {
  /* Visual size */
  width: 24px;
  height: 24px;
  
  /* Expand touch target without changing visual size */
  position: relative;
  padding: 0;
  margin: 0;
}

.message-action-btn::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 44px;
  height: 44px;
  /* Invisible but captures touches */
}
```

**Alternative approach using padding:**

```css
.send-button {
  /* Visual icon area */
  width: 24px;
  height: 24px;
  
  /* Expand touch area to 48×48 */
  padding: 12px;
  margin: -12px; /* Compensate for layout if needed */
  
  /* Visual styling */
  background: #007AFF;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

### Touch Target Spacing

Targets placed too close together cause mis-taps. Leave adequate spacing:

```css
.quick-reply-container {
  display: flex;
  gap: 8px; /* Minimum 8px between touch targets */
  flex-wrap: wrap;
  padding: 12px;
}

.quick-reply-chip {
  min-height: 44px;
  padding: 10px 16px;
  border-radius: 22px;
  background: #f0f0f0;
  border: none;
  font-size: 15px;
}
```

---

## Pointer Events API

Modern web development should use Pointer Events instead of separate touch and mouse event handling. Pointer Events unify mouse, touch, pen/stylus, and other pointing devices under a single API.

### Why Pointer Events?

| Aspect | Touch Events | Mouse Events | Pointer Events |
|--------|--------------|--------------|----------------|
| Touch support | ✅ | ❌ | ✅ |
| Mouse support | ❌ | ✅ | ✅ |
| Pen/stylus | ❌ | Partial | ✅ |
| Multi-touch | ✅ | ❌ | ✅ |
| Pressure sensitivity | ❌ | ❌ | ✅ |

### Core Pointer Events

```javascript
const chatMessage = document.querySelector('.chat-message');

// Pointer down - equivalent to mousedown/touchstart
chatMessage.addEventListener('pointerdown', (event) => {
  console.log('Pointer type:', event.pointerType); // "mouse", "touch", "pen"
  console.log('Pointer ID:', event.pointerId);
  console.log('Is primary:', event.isPrimary);
  console.log('Pressure:', event.pressure); // 0 to 1
  console.log('Width/Height:', event.width, event.height); // Touch area
});

// Pointer up - equivalent to mouseup/touchend
chatMessage.addEventListener('pointerup', (event) => {
  console.log('Pointer released');
});

// Pointer move - equivalent to mousemove/touchmove
chatMessage.addEventListener('pointermove', (event) => {
  console.log('Position:', event.clientX, event.clientY);
});

// Pointer cancel - touch interrupted (e.g., notification appeared)
chatMessage.addEventListener('pointercancel', (event) => {
  console.log('Pointer cancelled - clean up any state');
});
```

### Event Type Mapping

| Pointer Event | Mouse Equivalent | Touch Equivalent |
|---------------|------------------|------------------|
| `pointerdown` | `mousedown` | `touchstart` |
| `pointerup` | `mouseup` | `touchend` |
| `pointermove` | `mousemove` | `touchmove` |
| `pointerenter` | `mouseenter` | — |
| `pointerleave` | `mouseleave` | — |
| `pointerover` | `mouseover` | — |
| `pointerout` | `mouseout` | — |
| `pointercancel` | — | `touchcancel` |

### Detecting Input Type

You can adapt your interface based on how the user is interacting:

```javascript
function handleInteraction(event) {
  switch (event.pointerType) {
    case 'touch':
      // Show larger targets, enable gestures
      enableSwipeGestures();
      break;
    case 'mouse':
      // Enable hover states, smaller targets OK
      enableHoverEffects();
      break;
    case 'pen':
      // Could enable precision features like drawing
      enablePenFeatures();
      break;
  }
}

document.addEventListener('pointerdown', handleInteraction, { once: true });
```

---

## Controlling Touch Behavior with CSS

The `touch-action` CSS property controls how touch gestures are handled by the browser:

```css
/* Default - browser handles pan and zoom */
.normal-scroll-area {
  touch-action: auto;
}

/* Disable all touch gestures - you handle everything */
.custom-gesture-area {
  touch-action: none;
}

/* Allow horizontal panning only (for swipe actions) */
.horizontal-swipe-area {
  touch-action: pan-x;
}

/* Allow vertical scrolling only */
.message-list {
  touch-action: pan-y;
}

/* Allow panning but disable pinch-zoom */
.chat-container {
  touch-action: manipulation;
}
```

> **Warning:** Using `touch-action: none` disables scrolling. Only use it on elements where you're implementing custom gesture handling.

---

## Implementing Swipe Gestures

Swipe gestures are essential for mobile chat interfaces—swiping on a message to reply, delete, or reveal actions.

### Basic Swipe Detection

```javascript
class SwipeHandler {
  constructor(element, options = {}) {
    this.element = element;
    this.threshold = options.threshold || 50; // Minimum distance to trigger swipe
    this.restraint = options.restraint || 100; // Maximum perpendicular distance
    
    this.startX = 0;
    this.startY = 0;
    this.distX = 0;
    this.distY = 0;
    
    this.onSwipeLeft = options.onSwipeLeft || (() => {});
    this.onSwipeRight = options.onSwipeRight || (() => {});
    
    this.element.addEventListener('pointerdown', this.handleStart.bind(this));
    this.element.addEventListener('pointermove', this.handleMove.bind(this));
    this.element.addEventListener('pointerup', this.handleEnd.bind(this));
    this.element.addEventListener('pointercancel', this.handleCancel.bind(this));
  }
  
  handleStart(event) {
    // Only track primary pointer (first finger)
    if (!event.isPrimary) return;
    
    this.startX = event.clientX;
    this.startY = event.clientY;
    this.distX = 0;
    this.distY = 0;
    
    // Capture pointer to receive events even if it leaves the element
    this.element.setPointerCapture(event.pointerId);
  }
  
  handleMove(event) {
    if (!event.isPrimary) return;
    
    this.distX = event.clientX - this.startX;
    this.distY = event.clientY - this.startY;
    
    // Provide visual feedback during swipe
    if (Math.abs(this.distX) > Math.abs(this.distY)) {
      this.element.style.transform = `translateX(${this.distX}px)`;
    }
  }
  
  handleEnd(event) {
    if (!event.isPrimary) return;
    
    this.element.releasePointerCapture(event.pointerId);
    
    // Check if it qualifies as a swipe
    if (Math.abs(this.distX) >= this.threshold && 
        Math.abs(this.distY) <= this.restraint) {
      
      if (this.distX > 0) {
        this.onSwipeRight();
      } else {
        this.onSwipeLeft();
      }
    }
    
    // Animate back to original position
    this.element.style.transition = 'transform 0.2s ease-out';
    this.element.style.transform = 'translateX(0)';
    
    setTimeout(() => {
      this.element.style.transition = '';
    }, 200);
  }
  
  handleCancel(event) {
    this.element.releasePointerCapture(event.pointerId);
    this.element.style.transform = 'translateX(0)';
  }
}
```

### Using the Swipe Handler

```javascript
// Apply to all messages
document.querySelectorAll('.chat-message').forEach(message => {
  new SwipeHandler(message, {
    threshold: 60,
    onSwipeLeft: () => {
      showMessageActions(message, 'left');
    },
    onSwipeRight: () => {
      triggerReply(message);
    }
  });
});
```

### Swipe-to-Reveal Actions

A common pattern is revealing action buttons behind a message:

```html
<div class="message-container">
  <div class="message-actions left-actions">
    <button class="action-delete">🗑️</button>
  </div>
  <div class="message-content" data-swipeable>
    <p>Hello, this is a message!</p>
  </div>
  <div class="message-actions right-actions">
    <button class="action-reply">↩️</button>
  </div>
</div>
```

```css
.message-container {
  position: relative;
  overflow: hidden;
}

.message-actions {
  position: absolute;
  top: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  padding: 0 16px;
}

.left-actions {
  left: 0;
  background: #ff3b30;
}

.right-actions {
  right: 0;
  background: #007AFF;
}

.message-content {
  position: relative;
  background: white;
  z-index: 1;
  touch-action: pan-y; /* Allow vertical scroll, we handle horizontal */
}
```

---

## Long-Press Actions

Long-press (also called "press and hold") is the touch equivalent of right-click—revealing contextual actions.

### Implementing Long-Press Detection

```javascript
class LongPressHandler {
  constructor(element, options = {}) {
    this.element = element;
    this.duration = options.duration || 500; // Milliseconds to trigger
    this.onLongPress = options.onLongPress || (() => {});
    
    this.timer = null;
    this.isPressed = false;
    this.startPosition = { x: 0, y: 0 };
    this.moveThreshold = 10; // Cancel if finger moves more than this
    
    this.element.addEventListener('pointerdown', this.handleDown.bind(this));
    this.element.addEventListener('pointermove', this.handleMove.bind(this));
    this.element.addEventListener('pointerup', this.handleUp.bind(this));
    this.element.addEventListener('pointercancel', this.handleCancel.bind(this));
    this.element.addEventListener('contextmenu', this.handleContextMenu.bind(this));
  }
  
  handleDown(event) {
    if (!event.isPrimary) return;
    
    this.isPressed = true;
    this.startPosition = { x: event.clientX, y: event.clientY };
    
    this.timer = setTimeout(() => {
      if (this.isPressed) {
        this.onLongPress(event);
        
        // Provide haptic feedback if available
        if ('vibrate' in navigator) {
          navigator.vibrate(50);
        }
      }
    }, this.duration);
  }
  
  handleMove(event) {
    if (!this.isPressed) return;
    
    // Cancel if finger moved too far
    const distance = Math.hypot(
      event.clientX - this.startPosition.x,
      event.clientY - this.startPosition.y
    );
    
    if (distance > this.moveThreshold) {
      this.cancel();
    }
  }
  
  handleUp() {
    this.cancel();
  }
  
  handleCancel() {
    this.cancel();
  }
  
  handleContextMenu(event) {
    // Prevent browser's default context menu on long-press
    event.preventDefault();
  }
  
  cancel() {
    this.isPressed = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
```

### Creating a Context Menu

```javascript
class MessageContextMenu {
  constructor() {
    this.menu = this.createMenu();
    document.body.appendChild(this.menu);
    
    // Close on outside tap
    document.addEventListener('pointerdown', (e) => {
      if (!this.menu.contains(e.target)) {
        this.hide();
      }
    });
  }
  
  createMenu() {
    const menu = document.createElement('div');
    menu.className = 'context-menu';
    menu.innerHTML = `
      <button data-action="reply">↩️ Reply</button>
      <button data-action="copy">📋 Copy</button>
      <button data-action="forward">➡️ Forward</button>
      <button data-action="delete">🗑️ Delete</button>
    `;
    menu.hidden = true;
    return menu;
  }
  
  show(x, y, message) {
    this.currentMessage = message;
    
    // Position near the touch point
    this.menu.style.left = `${x}px`;
    this.menu.style.top = `${y}px`;
    
    // Keep menu on screen
    const rect = this.menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
      this.menu.style.left = `${window.innerWidth - rect.width - 16}px`;
    }
    if (rect.bottom > window.innerHeight) {
      this.menu.style.top = `${y - rect.height}px`;
    }
    
    this.menu.hidden = false;
    this.menu.classList.add('visible');
  }
  
  hide() {
    this.menu.classList.remove('visible');
    setTimeout(() => {
      this.menu.hidden = true;
    }, 200);
  }
}
```

```css
.context-menu {
  position: fixed;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  padding: 8px 0;
  min-width: 160px;
  z-index: 1000;
  opacity: 0;
  transform: scale(0.9);
  transition: opacity 0.2s, transform 0.2s;
}

.context-menu.visible {
  opacity: 1;
  transform: scale(1);
}

.context-menu button {
  display: block;
  width: 100%;
  padding: 12px 16px;
  border: none;
  background: transparent;
  text-align: left;
  font-size: 16px;
  min-height: 44px; /* Touch target */
}

.context-menu button:active {
  background: #f0f0f0;
}
```

### Connecting Long-Press to Context Menu

```javascript
const contextMenu = new MessageContextMenu();

document.querySelectorAll('.chat-message').forEach(message => {
  new LongPressHandler(message, {
    duration: 500,
    onLongPress: (event) => {
      contextMenu.show(event.clientX, event.clientY, message);
    }
  });
});
```

---

## Touch Feedback

Visual feedback is crucial for touch interfaces—users need confirmation that their touch registered.

### CSS Active States

```css
.chat-button {
  background: #007AFF;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  transition: transform 0.1s, opacity 0.1s;
  
  /* Prevent text selection on long-press */
  user-select: none;
  -webkit-user-select: none;
  
  /* Prevent tap highlight on some browsers */
  -webkit-tap-highlight-color: transparent;
}

.chat-button:active {
  transform: scale(0.97);
  opacity: 0.9;
}
```

### Ripple Effect (Material Design Style)

```javascript
function createRipple(event) {
  const button = event.currentTarget;
  
  const circle = document.createElement('span');
  const diameter = Math.max(button.clientWidth, button.clientHeight);
  const radius = diameter / 2;
  
  const rect = button.getBoundingClientRect();
  
  circle.style.width = circle.style.height = `${diameter}px`;
  circle.style.left = `${event.clientX - rect.left - radius}px`;
  circle.style.top = `${event.clientY - rect.top - radius}px`;
  circle.classList.add('ripple');
  
  // Remove existing ripple
  const existingRipple = button.querySelector('.ripple');
  if (existingRipple) {
    existingRipple.remove();
  }
  
  button.appendChild(circle);
  
  // Clean up after animation
  circle.addEventListener('animationend', () => {
    circle.remove();
  });
}

document.querySelectorAll('.ripple-button').forEach(button => {
  button.addEventListener('pointerdown', createRipple);
});
```

```css
.ripple-button {
  position: relative;
  overflow: hidden;
}

.ripple {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.4);
  transform: scale(0);
  animation: ripple-animation 0.6s linear;
  pointer-events: none;
}

@keyframes ripple-animation {
  to {
    transform: scale(4);
    opacity: 0;
  }
}
```

### Message Bubble Feedback

```css
.chat-message-bubble {
  transition: background-color 0.15s ease-out;
}

.chat-message-bubble:active {
  background-color: rgba(0, 0, 0, 0.05);
}

/* Or use a highlight overlay */
.chat-message-bubble {
  position: relative;
}

.chat-message-bubble::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0);
  transition: background 0.15s;
  pointer-events: none;
  border-radius: inherit;
}

.chat-message-bubble:active::after {
  background: rgba(0, 0, 0, 0.08);
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use 44×44px minimum touch targets | Create tiny tap areas |
| Use Pointer Events for unified handling | Mix touch and mouse event handlers |
| Provide immediate visual feedback | Leave taps with no confirmation |
| Add spacing between touch targets | Place buttons too close together |
| Use `touch-action` to prevent conflicts | Fight with browser gestures |
| Cancel gestures on `pointercancel` | Leave gestures in incomplete states |
| Test on real devices | Rely only on browser DevTools |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| 300ms tap delay on older browsers | Use `touch-action: manipulation` |
| Text selection during gestures | Add `user-select: none` to interactive elements |
| Scroll jank during swipe | Use `will-change: transform` sparingly |
| Gestures not working | Check `touch-action` CSS property |
| Can't scroll after adding gesture | Only use `touch-action: none` on the gesture element |

---

## Hands-on Exercise

### Your Task

Create a touch-optimized message list with the following features:

1. **Proper touch targets** — All buttons at least 44×44px
2. **Swipe-to-reply** — Swipe right to trigger reply
3. **Swipe-to-delete** — Swipe left to reveal delete button
4. **Long-press menu** — Show context menu with Copy/Reply/Delete
5. **Touch feedback** — Visual confirmation on every interaction

### Requirements

1. Use Pointer Events (not touch events)
2. Include pointer capture for reliable swipe tracking
3. Implement cancel handling for interrupted gestures
4. Add haptic feedback where supported

### Expected Result

A message list that feels native and responsive, with smooth gestures and clear feedback.

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with the `SwipeHandler` class and customize thresholds
- Use `setPointerCapture()` to track pointers that leave the element
- Add `touch-action: pan-y` to allow vertical scrolling while handling horizontal swipes
- Test on a real device—simulators don't capture the feel accurately

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f5f5f5;
    }
    
    .message-list {
      max-width: 500px;
      margin: 0 auto;
      padding: 16px;
    }
    
    .message-wrapper {
      position: relative;
      margin-bottom: 8px;
      overflow: hidden;
      border-radius: 16px;
    }
    
    .message-actions {
      position: absolute;
      inset: 0;
      display: flex;
    }
    
    .action-left, .action-right {
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 80px;
    }
    
    .action-left {
      background: #ff3b30;
      margin-right: auto;
    }
    
    .action-right {
      background: #007AFF;
      margin-left: auto;
    }
    
    .action-btn {
      width: 44px;
      height: 44px;
      border: none;
      background: transparent;
      color: white;
      font-size: 24px;
      cursor: pointer;
    }
    
    .message {
      position: relative;
      background: white;
      padding: 12px 16px;
      border-radius: 16px;
      touch-action: pan-y;
      user-select: none;
      -webkit-user-select: none;
      z-index: 1;
    }
    
    .message:active {
      background: #f8f8f8;
    }
    
    .message.swiping {
      transition: none;
    }
    
    .message:not(.swiping) {
      transition: transform 0.2s ease-out;
    }
    
    .context-menu {
      position: fixed;
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      padding: 8px 0;
      z-index: 1000;
      opacity: 0;
      transform: scale(0.9);
      transition: opacity 0.2s, transform 0.2s;
      pointer-events: none;
    }
    
    .context-menu.visible {
      opacity: 1;
      transform: scale(1);
      pointer-events: auto;
    }
    
    .context-menu button {
      display: block;
      width: 100%;
      min-height: 44px;
      padding: 12px 16px;
      border: none;
      background: transparent;
      text-align: left;
      font-size: 16px;
    }
    
    .context-menu button:active {
      background: #f0f0f0;
    }
  </style>
</head>
<body>
  <div class="message-list">
    <div class="message-wrapper">
      <div class="message-actions">
        <div class="action-left">
          <button class="action-btn">🗑️</button>
        </div>
        <div class="action-right">
          <button class="action-btn">↩️</button>
        </div>
      </div>
      <div class="message">Hello! This is a swipeable message.</div>
    </div>
    <div class="message-wrapper">
      <div class="message-actions">
        <div class="action-left">
          <button class="action-btn">🗑️</button>
        </div>
        <div class="action-right">
          <button class="action-btn">↩️</button>
        </div>
      </div>
      <div class="message">Try swiping left or right, or long-press!</div>
    </div>
  </div>
  
  <div class="context-menu" id="contextMenu">
    <button data-action="reply">↩️ Reply</button>
    <button data-action="copy">📋 Copy</button>
    <button data-action="delete">🗑️ Delete</button>
  </div>
  
  <script>
    const contextMenu = document.getElementById('contextMenu');
    
    document.querySelectorAll('.message').forEach(message => {
      let startX, startY, distX, isPressed, longPressTimer;
      
      message.addEventListener('pointerdown', (e) => {
        if (!e.isPrimary) return;
        
        startX = e.clientX;
        startY = e.clientY;
        distX = 0;
        isPressed = true;
        message.classList.add('swiping');
        message.setPointerCapture(e.pointerId);
        
        // Long press timer
        longPressTimer = setTimeout(() => {
          if (isPressed && Math.abs(distX) < 10) {
            if ('vibrate' in navigator) navigator.vibrate(50);
            showContextMenu(e.clientX, e.clientY);
          }
        }, 500);
      });
      
      message.addEventListener('pointermove', (e) => {
        if (!e.isPrimary || !isPressed) return;
        
        distX = e.clientX - startX;
        const distY = e.clientY - startY;
        
        // Cancel long press if moved
        if (Math.abs(distX) > 10 || Math.abs(distY) > 10) {
          clearTimeout(longPressTimer);
        }
        
        // Only apply horizontal movement
        if (Math.abs(distX) > Math.abs(distY)) {
          message.style.transform = `translateX(${distX}px)`;
        }
      });
      
      message.addEventListener('pointerup', (e) => {
        if (!e.isPrimary) return;
        cleanup(e, message);
        
        // Check for swipe
        if (Math.abs(distX) > 60) {
          if (distX > 0) {
            console.log('Reply triggered');
          } else {
            console.log('Delete revealed');
          }
        }
        
        message.style.transform = '';
      });
      
      message.addEventListener('pointercancel', (e) => cleanup(e, message));
      
      function cleanup(e, el) {
        clearTimeout(longPressTimer);
        isPressed = false;
        el.releasePointerCapture(e.pointerId);
        el.classList.remove('swiping');
      }
    });
    
    function showContextMenu(x, y) {
      contextMenu.style.left = `${Math.min(x, window.innerWidth - 180)}px`;
      contextMenu.style.top = `${Math.min(y, window.innerHeight - 150)}px`;
      contextMenu.classList.add('visible');
    }
    
    document.addEventListener('pointerdown', (e) => {
      if (!contextMenu.contains(e.target)) {
        contextMenu.classList.remove('visible');
      }
    });
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ Touch targets must be at least 44×44px (Apple) or 48×48dp (Material Design)  
✅ Pointer Events unify mouse, touch, and pen input under one API  
✅ Use `touch-action` CSS to control browser gesture handling  
✅ Swipe gestures require tracking start position, calculating distance, and threshold detection  
✅ Long-press menus provide contextual actions equivalent to right-click  
✅ Visual feedback on every interaction confirms user intent  

**Next:** [Virtual Keyboard Handling](./02-virtual-keyboard-handling.md)

---

<!-- 
Sources Consulted:
- MDN Pointer Events: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
- MDN touch-action: https://developer.mozilla.org/en-US/docs/Web/CSS/touch-action
- Apple HIG Touch Targets: https://developer.apple.com/design/human-interface-guidelines/inputs
- Material Design Touch Targets: https://m3.material.io/foundations/interaction/touch-targets
-->
