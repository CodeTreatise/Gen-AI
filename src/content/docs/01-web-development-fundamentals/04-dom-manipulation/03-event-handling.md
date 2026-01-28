---
title: "Event Handling"
---

# Event Handling

## Introduction

Static web pages are boring. **Events** bring pages to life—clicks, key presses, form submissions, scroll movements—all trigger JavaScript code that creates interactive experiences.

Whether you're building a chatbot interface that responds to button clicks or a form that validates in real-time, mastering event handling is essential for creating responsive, user-friendly applications.

### What We'll Cover
- Adding event listeners with `addEventListener()`
- Understanding the event object (`target`, `currentTarget`, `type`)
- Event propagation: bubbling and capturing phases
- Controlling propagation with `stopPropagation()` and `preventDefault()`
- Event delegation for efficient handling
- Common event types (click, input, submit, keydown, scroll)
- Creating custom events with `CustomEvent`
- Performance with passive event listeners

### Prerequisites
- [Selecting Elements](./01-selecting-elements.md) (querySelector, closest)
- [Creating & Modifying Elements](./02-creating-modifying-elements.md) (createElement, classList)
- [JavaScript Core Concepts](../../03-javascript-core-concepts/00-javascript-core-concepts.md) (functions, arrow functions)

---

## Adding Event Listeners

### addEventListener() - The Modern Approach

Attaches an event handler to an element **without overwriting** existing handlers.

**Syntax:**
```javascript
element.addEventListener(eventType, handlerFunction, options);
```

**Basic example:**
```javascript
const button = document.getElementById('submit-btn');

button.addEventListener('click', () => {
  console.log('Button clicked!');
});
```

**Output** (when button is clicked):
```
Button clicked!
```

### Multiple Listeners on the Same Element

```javascript
const button = document.getElementById('submit-btn');

// First listener
button.addEventListener('click', () => {
  console.log('First handler');
});

// Second listener (both execute)
button.addEventListener('click', () => {
  console.log('Second handler');
});
```

**Output** (when button is clicked):
```
First handler
Second handler
```

> **Note:** Both handlers execute in the order they were added. `addEventListener()` doesn't replace existing listeners.

### Removing Event Listeners

Use `removeEventListener()` with the **same function reference**.

```javascript
const button = document.getElementById('submit-btn');

function handleClick() {
  console.log('Button clicked!');
}

// Add listener
button.addEventListener('click', handleClick);

// Remove listener
button.removeEventListener('click', handleClick);
```

**⚠️ Anonymous functions can't be removed:**
```javascript
// ❌ Can't remove (no function reference)
button.addEventListener('click', () => {
  console.log('Clicked!');
});

// ✅ Can be removed (named function)
function handleClick() {
  console.log('Clicked!');
}
button.addEventListener('click', handleClick);
button.removeEventListener('click', handleClick);
```

---

## The Event Object

When an event fires, JavaScript passes an **event object** to your handler. It contains useful information about the event.

### Common Event Properties

```javascript
const button = document.getElementById('submit-btn');

button.addEventListener('click', (event) => {
  console.log('Event type:', event.type);              // "click"
  console.log('Target element:', event.target);        // <button id="submit-btn">
  console.log('CurrentTarget:', event.currentTarget);  // <button id="submit-btn">
  console.log('Timestamp:', event.timeStamp);          // 12345.67
});
```

**Output:**
```
Event type: click
Target element: <button id="submit-btn">Submit</button>
CurrentTarget: <button id="submit-btn">Submit</button>
Timestamp: 12345.67
```

### target vs currentTarget

- **`event.target`**: The element that **triggered** the event (the actual clicked element)
- **`event.currentTarget`**: The element the **listener is attached to**

```javascript
// HTML:
// <div id="container">
//   <button id="inner-btn">Click me</button>
// </div>

const container = document.getElementById('container');

container.addEventListener('click', (event) => {
  console.log('target:', event.target.id);            // "inner-btn" (what was clicked)
  console.log('currentTarget:', event.currentTarget.id); // "container" (where listener is)
});
```

**Output** (when clicking the button):
```
target: inner-btn
currentTarget: container
```

**Use case:** Event delegation (covered below).

---

## Event Propagation: Bubbling and Capturing

When an event occurs on an element, it doesn't just fire on that element—it **propagates** through the DOM tree in two phases.

### The Three Phases

1. **Capturing phase** (top-down): From `document` → down to target
2. **Target phase**: Event fires on the target element
3. **Bubbling phase** (bottom-up): From target → up to `document`

**Diagram:**
```
Document (capturing starts here)
  ↓
  html
    ↓
    body
      ↓
      div (container)
        ↓
        button (target) ← EVENT HAPPENS HERE
        ↑
      div (container)
    ↑
  body
↑
html
↑
Document (bubbling ends here)
```

### Bubbling (Default)

By default, events **bubble up** from the target to ancestors.

```javascript
// HTML:
// <div id="parent">
//   <button id="child">Click me</button>
// </div>

document.getElementById('parent').addEventListener('click', () => {
  console.log('Parent clicked');
});

document.getElementById('child').addEventListener('click', () => {
  console.log('Child clicked');
});
```

**Output** (when clicking the button):
```
Child clicked
Parent clicked
```

The event fires on `child` first, then **bubbles up** to `parent`.

### Capturing (Opt-in)

Set `{ capture: true }` to listen during the **capturing phase** (before the target).

```javascript
document.getElementById('parent').addEventListener('click', () => {
  console.log('Parent (capturing)');
}, { capture: true });

document.getElementById('child').addEventListener('click', () => {
  console.log('Child (target)');
});
```

**Output:**
```
Parent (capturing)
Child (target)
```

---

## Controlling Event Flow

### stopPropagation() - Stop Bubbling/Capturing

Prevents the event from reaching other listeners in the propagation chain.

```javascript
document.getElementById('parent').addEventListener('click', () => {
  console.log('Parent clicked');
});

document.getElementById('child').addEventListener('click', (event) => {
  console.log('Child clicked');
  event.stopPropagation(); // Stop bubbling to parent
});
```

**Output** (when clicking child):
```
Child clicked
```

The parent listener **never fires** because propagation stopped.

### preventDefault() - Cancel Default Action

Prevents the browser's default behavior for an event.

**Common use cases:**
- Stop form submission
- Prevent link navigation
- Block right-click context menu

```javascript
const form = document.getElementById('my-form');

form.addEventListener('submit', (event) => {
  event.preventDefault(); // Don't submit/reload page
  
  console.log('Form validation in progress...');
  
  // Handle form with JavaScript (e.g., AJAX submission)
});
```

**Output** (when submitting form):
```
Form validation in progress...
```

**Form doesn't reload the page** because `preventDefault()` stopped the default submission.

**Another example: Prevent link navigation**
```javascript
const link = document.querySelector('a');

link.addEventListener('click', (event) => {
  event.preventDefault(); // Don't navigate
  console.log('Link clicked, but navigation prevented');
});
```

---

## Event Delegation

Instead of attaching listeners to **many elements**, attach **one listener** to a parent and check which child was clicked.

### The Problem: Many Elements

```javascript
// ❌ Inefficient: Attach listener to every button
const buttons = document.querySelectorAll('.message-delete-btn');

buttons.forEach(button => {
  button.addEventListener('click', () => {
    console.log('Delete button clicked');
  });
});

// Problem: New buttons added later won't have listeners!
```

### The Solution: Event Delegation

```javascript
// ✅ Efficient: One listener on the parent
const messageContainer = document.getElementById('messages');

messageContainer.addEventListener('click', (event) => {
  // Check if clicked element is a delete button
  if (event.target.classList.contains('message-delete-btn')) {
    const messageId = event.target.dataset.messageId;
    console.log(`Delete button for message ${messageId} clicked`);
    
    // Remove the message
    const messageDiv = event.target.closest('.message');
    messageDiv.remove();
  }
});
```

**Benefits:**
- **One listener** instead of hundreds
- Works for **dynamically added** elements
- Better **performance** and memory usage

**When to use:**
- Lists (messages, todos, table rows)
- Dynamically generated content
- Large numbers of similar elements

---

## Common Event Types

### Mouse Events

```javascript
element.addEventListener('click', (e) => {
  console.log('Single click');
});

element.addEventListener('dblclick', (e) => {
  console.log('Double click');
});

element.addEventListener('mouseenter', (e) => {
  console.log('Mouse entered');
});

element.addEventListener('mouseleave', (e) => {
  console.log('Mouse left');
});

element.addEventListener('mousemove', (e) => {
  console.log('Mouse position:', e.clientX, e.clientY);
});
```

### Keyboard Events

```javascript
const input = document.getElementById('search-input');

input.addEventListener('keydown', (e) => {
  console.log('Key pressed:', e.key);
  
  if (e.key === 'Enter') {
    console.log('Enter key pressed!');
  }
  
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault(); // Prevent browser save dialog
    console.log('Ctrl+S pressed');
  }
});

input.addEventListener('keyup', (e) => {
  console.log('Key released:', e.key);
});
```

**Output** (when typing "Hi" and pressing Enter):
```
Key pressed: H
Key released: H
Key pressed: i
Key released: i
Key pressed: Enter
Enter key pressed!
Key released: Enter
```

### Form Events

```javascript
const input = document.getElementById('email-input');
const form = document.getElementById('signup-form');

// Fires on every keystroke/change
input.addEventListener('input', (e) => {
  console.log('Current value:', e.target.value);
});

// Fires when element loses focus
input.addEventListener('blur', (e) => {
  console.log('Input lost focus');
});

// Fires when element gains focus
input.addEventListener('focus', (e) => {
  console.log('Input focused');
});

// Fires on form submission
form.addEventListener('submit', (e) => {
  e.preventDefault();
  console.log('Form submitted');
});
```

### Scroll Events

```javascript
window.addEventListener('scroll', () => {
  console.log('Scroll position:', window.scrollY);
});

// For specific elements
const chatBox = document.getElementById('chat-box');

chatBox.addEventListener('scroll', () => {
  const atBottom = chatBox.scrollHeight - chatBox.scrollTop === chatBox.clientHeight;
  
  if (atBottom) {
    console.log('Scrolled to bottom');
  }
});
```

---

## Custom Events

Create your own events to communicate between components.

### Creating and Dispatching Custom Events

```javascript
// Create a custom event
const messageReceived = new CustomEvent('messageReceived', {
  detail: {
    text: 'Hello from AI!',
    timestamp: Date.now()
  }
});

// Listen for the custom event
document.addEventListener('messageReceived', (event) => {
  console.log('Message:', event.detail.text);
  console.log('Timestamp:', event.detail.timestamp);
});

// Dispatch (trigger) the event
document.dispatchEvent(messageReceived);
```

**Output:**
```
Message: Hello from AI!
Timestamp: 1704067200000
```

**Use case:** Decoupled components—one part of your app triggers an event, another part responds without direct connection.

---

## Passive Event Listeners (Performance)

**Problem:** Scroll and touch event listeners can **block scrolling** if they call `preventDefault()`.

**Solution:** Mark listeners as `{ passive: true }` to tell the browser they won't cancel scrolling.

```javascript
// ❌ Without passive (may cause scroll lag)
element.addEventListener('touchstart', (e) => {
  // Browser waits to see if preventDefault() is called
  console.log('Touch started');
});

// ✅ With passive (better scroll performance)
element.addEventListener('touchstart', (e) => {
  // Browser knows preventDefault() won't be called
  console.log('Touch started');
}, { passive: true });
```

**When to use:**
- Scroll listeners that **don't** call `preventDefault()`
- Touch listeners that **don't** call `preventDefault()`
- Any listener that's purely for tracking/analytics

> **Note:** If you set `passive: true`, calling `preventDefault()` will be **ignored** (with a console warning).

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| **Use event delegation for lists** | One listener vs hundreds—better performance and memory |
| **Call `preventDefault()` early** | Prevents default action before your code runs |
| **Use named functions for removable listeners** | Anonymous functions can't be removed |
| **Mark scroll/touch listeners as passive** | Improves scrolling performance |
| **Check `event.target` in delegation** | Ensure the correct element triggered the event |
| **Use `once: true` for one-time events** | Listener auto-removes after first trigger |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Attaching listeners to many elements | Use event delegation (one listener on parent) |
| Forgetting `preventDefault()` in form handlers | Always call `event.preventDefault()` for custom form handling |
| Using `stopPropagation()` unnecessarily | Only stop propagation when absolutely needed (can break other listeners) |
| Anonymous functions you can't remove | Use named functions or store function references |
| Not checking `event.target` in delegation | Validate that the clicked element is what you expect |
| Calling `preventDefault()` in passive listeners | Remove `passive: true` if you need `preventDefault()` |

---

## Hands-on Exercise

### Your Task

Build an **AI chat interface** with keyboard shortcuts and event delegation:
- Press **Enter** to send a message
- Press **Ctrl+K** to clear the chat
- Click a message to mark it as "selected" (highlight it)
- Delete a message by clicking a "Delete" button on each message

### Requirements

1. Create HTML with:
   - Text input (`<input id="message-input">`)
   - Empty `<div id="chat-container"></div>`

2. **Enter key**: Send the message (add to chat), clear input
   - Create a message `div` with class `message`
   - Add the message text
   - Add a "Delete" button with class `delete-btn`
   - Append to `chat-container`

3. **Ctrl+K**: Clear all messages

4. **Click message**: Toggle `selected` class (yellow background)

5. **Click delete button**: Remove the parent message
   - Use event delegation (one listener on `chat-container`)
   - Use `closest('.message')` to find parent

6. **Prevent default**: Pressing Enter shouldn't add newline, Ctrl+K shouldn't open browser search

### Expected Result

- Typing "Hello" and pressing Enter adds a message
- Clicking a message highlights it (yellow)
- Clicking "Delete" removes the message
- Pressing Ctrl+K clears all messages

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `event.key === 'Enter'` to detect Enter key
- Use `event.ctrlKey && event.key === 'k'` for Ctrl+K
- Call `event.preventDefault()` to stop default browser behavior
- Use event delegation on `chat-container` for clicks
- Check `event.target.classList.contains('delete-btn')` for delete clicks
- Check `event.target.classList.contains('message')` for message clicks
- Use `element.classList.toggle('selected')` to highlight
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**HTML:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Chat with Keyboard Shortcuts</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 600px;
      margin: 50px auto;
    }
    #message-input {
      width: 100%;
      padding: 15px;
      font-size: 16px;
      border: 2px solid #ddd;
      border-radius: 8px;
      box-sizing: border-box;
    }
    #chat-container {
      margin-top: 20px;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      min-height: 300px;
    }
    .message {
      padding: 15px;
      margin: 10px 0;
      background-color: #f0f0f0;
      border-radius: 8px;
      cursor: pointer;
      position: relative;
      transition: background-color 0.2s;
    }
    .message.selected {
      background-color: #fff3cd;
      border: 2px solid #ffc107;
    }
    .delete-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      padding: 5px 10px;
      font-size: 12px;
      background-color: #dc3545;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    .delete-btn:hover {
      background-color: #c82333;
    }
    .keyboard-hint {
      margin-top: 10px;
      font-size: 14px;
      color: #666;
    }
  </style>
</head>
<body>
  <h1>AI Chat Interface</h1>
  
  <input type="text" id="message-input" placeholder="Type a message and press Enter..." />
  
  <div class="keyboard-hint">
    <strong>Shortcuts:</strong> Enter = Send | Ctrl+K = Clear Chat | Click message to select
  </div>
  
  <div id="chat-container"></div>

  <script src="script.js"></script>
</body>
</html>
```

**JavaScript (script.js):**
```javascript
const messageInput = document.getElementById('message-input');
const chatContainer = document.getElementById('chat-container');

// Function to create a message element
function createMessage(text) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message');
  messageDiv.textContent = text;
  
  // Add delete button
  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = 'Delete';
  deleteBtn.classList.add('delete-btn');
  messageDiv.appendChild(deleteBtn);
  
  return messageDiv;
}

// Keyboard shortcuts
messageInput.addEventListener('keydown', (event) => {
  // Enter key: Send message
  if (event.key === 'Enter') {
    event.preventDefault();
    
    const text = messageInput.value.trim();
    if (!text) return;
    
    const message = createMessage(text);
    chatContainer.appendChild(message);
    
    messageInput.value = '';
  }
  
  // Ctrl+K: Clear chat
  if (event.ctrlKey && event.key === 'k') {
    event.preventDefault();
    chatContainer.replaceChildren();
  }
});

// Event delegation for clicks on messages and delete buttons
chatContainer.addEventListener('click', (event) => {
  // Click delete button: Remove message
  if (event.target.classList.contains('delete-btn')) {
    const messageDiv = event.target.closest('.message');
    messageDiv.remove();
    return; // Don't toggle selection
  }
  
  // Click message: Toggle selection
  if (event.target.classList.contains('message')) {
    event.target.classList.toggle('selected');
  }
});
```

</details>

### Bonus Challenges

- [ ] Add **Escape key** to deselect all messages
- [ ] Add **Ctrl+A** to select all messages
- [ ] Show a count of selected messages
- [ ] Add a "Delete Selected" button that removes all selected messages
- [ ] Prevent clicking the delete button from toggling selection (hint: `stopPropagation()`)

---

## Summary

✅ **addEventListener()** attaches handlers without overwriting existing ones  
✅ **Event object** (`target`, `currentTarget`, `type`) provides event details  
✅ **Event propagation** moves through capturing → target → bubbling phases  
✅ **preventDefault()** stops default actions (form submission, link navigation)  
✅ **Event delegation** attaches one listener to a parent for efficiency  
✅ **Custom events** enable decoupled component communication  
✅ **Passive listeners** improve scroll/touch performance

**Next:** [Form Handling](./04-form-handling.md)

---

## Further Reading

- [MDN: EventTarget.addEventListener()](https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener) - Complete reference
- [MDN: Event](https://developer.mozilla.org/en-US/docs/Web/API/Event) - Event object properties
- [MDN: Event reference](https://developer.mozilla.org/en-US/docs/Web/Events) - All event types
- [Form Handling](./04-form-handling.md) - Next lesson

<!-- 
Sources Consulted:
- MDN EventTarget.addEventListener: https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener
- MDN Event interface: https://developer.mozilla.org/en-US/docs/Web/API/Event
- MDN Event delegation: https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Building_blocks/Events#event_delegation
- MDN CustomEvent: https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent
-->
