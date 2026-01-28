---
title: "Observer APIs"
---

# Observer APIs

## Introduction

Sometimes you need to know when things change—when an element enters the viewport, when the DOM structure is modified, or when an element's size changes. Polling for these changes is inefficient and drains performance.

**Observer APIs** provide efficient, event-driven ways to monitor changes without constantly checking. They're essential for building performant, responsive applications—lazy loading images, infinite scroll, detecting when AI responses appear, tracking layout changes, and more.

### What We'll Cover
- `IntersectionObserver` for visibility detection (lazy loading, infinite scroll)
- `MutationObserver` for DOM change detection (content updates, dynamic elements)
- `ResizeObserver` for element size changes (responsive layouts)
- `PerformanceObserver` for performance metrics (optional advanced topic)
- Best practices and performance considerations

### Prerequisites
- [Event Handling](./03-event-handling.md) (addEventListener, callbacks)
- [Selecting Elements](./01-selecting-elements.md) (querySelector, querySelectorAll)
- [JavaScript Core Concepts](../../03-javascript-core-concepts/00-javascript-core-concepts.md) (callbacks, arrays)

---

## IntersectionObserver: Visibility Detection

The `IntersectionObserver` detects when an element **enters or exits the viewport** (or a parent container). Perfect for lazy loading, infinite scroll, and tracking visibility.

### Basic Usage

```javascript
// Create an observer
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      console.log('Element is visible:', entry.target);
    } else {
      console.log('Element is hidden:', entry.target);
    }
  });
});

// Start observing an element
const target = document.getElementById('watched-element');
observer.observe(target);
```

**Output** (when element enters viewport):
```
Element is visible: <div id="watched-element">...</div>
```

### Lazy Loading Images

Only load images when they're about to be visible.

```javascript
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      
      // Replace placeholder with real image
      img.src = img.dataset.src;
      img.classList.add('loaded');
      
      // Stop observing this image
      imageObserver.unobserve(img);
      
      console.log('Image loaded:', img.src);
    }
  });
});

// Observe all images with data-src attribute
const lazyImages = document.querySelectorAll('img[data-src]');
lazyImages.forEach(img => imageObserver.observe(img));
```

**HTML:**
```html
<img data-src="image1.jpg" alt="Lazy loaded image">
<img data-src="image2.jpg" alt="Lazy loaded image">
<img data-src="image3.jpg" alt="Lazy loaded image">
```

**Output:**
```
Image loaded: image1.jpg
(when user scrolls down)
Image loaded: image2.jpg
Image loaded: image3.jpg
```

### Infinite Scroll

Load more content when user reaches the bottom.

```javascript
const sentinel = document.getElementById('sentinel'); // Element at the bottom

const infiniteObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      console.log('Loading more content...');
      loadMoreMessages(); // Your function to fetch/add content
    }
  });
});

infiniteObserver.observe(sentinel);

function loadMoreMessages() {
  // Simulate loading
  const container = document.getElementById('message-container');
  
  for (let i = 0; i < 10; i++) {
    const message = document.createElement('div');
    message.textContent = `Message ${Date.now()}`;
    message.className = 'message';
    container.appendChild(message);
  }
}
```

**HTML:**
```html
<div id="message-container">
  <!-- Messages here -->
</div>
<div id="sentinel"></div> <!-- Invisible trigger element -->
```

**Output:**
```
Loading more content...
(10 new messages added)
Loading more content...
(10 more messages added)
```

### IntersectionObserver Options

```javascript
const options = {
  root: null,          // Viewport (null = browser viewport)
  rootMargin: '0px',   // Margin around root (like CSS margin)
  threshold: 0.5       // 0-1 or array: 0.5 = trigger at 50% visibility
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    console.log('Intersection ratio:', entry.intersectionRatio); // 0 to 1
    
    if (entry.intersectionRatio > 0.5) {
      console.log('More than 50% visible');
    }
  });
}, options);

observer.observe(targetElement);
```

**threshold examples:**
- `0` - Triggers as soon as 1 pixel is visible
- `0.5` - Triggers when 50% visible
- `1.0` - Triggers when 100% visible
- `[0, 0.5, 1.0]` - Triggers at 0%, 50%, and 100%

### IntersectionObserver Methods

```javascript
const observer = new IntersectionObserver(callback, options);

// Start observing
observer.observe(element);

// Stop observing a specific element
observer.unobserve(element);

// Stop observing all elements
observer.disconnect();

// Get current entries (synchronous)
const records = observer.takeRecords();
```

---

## MutationObserver: DOM Change Detection

The `MutationObserver` watches for changes to the DOM—added/removed elements, attribute changes, text content changes.

### Basic Usage

```javascript
const targetNode = document.getElementById('chat-container');

// Create observer
const observer = new MutationObserver((mutationsList) => {
  for (const mutation of mutationsList) {
    if (mutation.type === 'childList') {
      console.log('Children changed:', mutation);
      console.log('Added nodes:', mutation.addedNodes);
      console.log('Removed nodes:', mutation.removedNodes);
    }
  }
});

// Configure what to observe
const config = {
  childList: true,     // Watch for added/removed children
  attributes: true,    // Watch for attribute changes
  subtree: true,       // Watch all descendants, not just direct children
  characterData: true  // Watch for text content changes
};

// Start observing
observer.observe(targetNode, config);
```

**Output** (when a child is added):
```
Children changed: MutationRecord { type: "childList", ... }
Added nodes: NodeList [ <div class="message">...</div> ]
Removed nodes: NodeList []
```

### Detecting New Messages (Chat Interface)

Auto-scroll to bottom when new messages arrive.

```javascript
const chatContainer = document.getElementById('chat-container');

const chatObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
      console.log('New message added, scrolling to bottom');
      
      // Scroll to bottom
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  });
});

chatObserver.observe(chatContainer, {
  childList: true,  // Only watch for added/removed children
  subtree: false    // Don't watch descendants
});
```

**Use case:** Automatically scroll chat to bottom when AI response arrives.

### Watching Attribute Changes

```javascript
const button = document.getElementById('status-button');

const attributeObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    if (mutation.type === 'attributes') {
      console.log(`Attribute ${mutation.attributeName} changed`);
      console.log('Old value:', mutation.oldValue);
      console.log('New value:', mutation.target.getAttribute(mutation.attributeName));
    }
  });
});

attributeObserver.observe(button, {
  attributes: true,      // Watch attributes
  attributeOldValue: true // Record old values
});

// Test: Change an attribute
button.setAttribute('data-status', 'loading');
```

**Output:**
```
Attribute data-status changed
Old value: null
New value: loading
```

### MutationObserver Configuration Options

```javascript
const config = {
  childList: true,          // Watch for added/removed children
  attributes: true,          // Watch for attribute changes
  characterData: true,       // Watch for text content changes (text nodes)
  subtree: true,             // Watch all descendants
  attributeOldValue: true,   // Record old attribute values
  characterDataOldValue: true, // Record old text values
  attributeFilter: ['class', 'data-status'] // Only watch specific attributes
};
```

---

## ResizeObserver: Element Size Changes

The `ResizeObserver` detects when an element's size changes—from window resizes, content changes, or CSS animations.

### Basic Usage

```javascript
const targetElement = document.getElementById('chat-box');

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { width, height } = entry.contentRect;
    
    console.log(`Element resized to ${width}x${height}`);
    
    // Adjust layout based on size
    if (width < 400) {
      entry.target.classList.add('compact');
    } else {
      entry.target.classList.remove('compact');
    }
  }
});

resizeObserver.observe(targetElement);
```

**Output** (when element is resized):
```
Element resized to 350x600
Element resized to 600x800
```

### Responsive Text Size

Adjust font size based on container width.

```javascript
const container = document.getElementById('message-container');

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const width = entry.contentRect.width;
    
    // Calculate font size: 16px base + 0.5px per 50px width
    const fontSize = 16 + Math.floor(width / 50) * 0.5;
    
    entry.target.style.fontSize = `${fontSize}px`;
    console.log(`Container width: ${width}px, Font size: ${fontSize}px`);
  }
});

resizeObserver.observe(container);
```

**Output:**
```
Container width: 400px, Font size: 20px
Container width: 800px, Font size: 24px
```

### Detecting Content Overflow

```javascript
const textBox = document.getElementById('text-box');

const overflowObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const element = entry.target;
    
    if (element.scrollHeight > element.clientHeight) {
      console.log('Content is overflowing!');
      element.classList.add('overflowing');
    } else {
      console.log('Content fits');
      element.classList.remove('overflowing');
    }
  }
});

overflowObserver.observe(textBox);
```

---

## PerformanceObserver: Metrics Tracking (Advanced)

The `PerformanceObserver` monitors performance metrics—resource loading times, long tasks, navigation timing.

### Monitoring Long Tasks

```javascript
const perfObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.duration > 50) {
      console.warn(`Long task detected: ${entry.duration}ms`);
    }
  }
});

// Observe long tasks (tasks >50ms)
perfObserver.observe({ type: 'longtask', buffered: true });
```

**Output:**
```
Long task detected: 127.3ms
```

### Monitoring Resource Loading

```javascript
const resourceObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name} loaded in ${entry.duration}ms`);
  }
});

resourceObserver.observe({ type: 'resource', buffered: true });
```

**Output:**
```
https://example.com/style.css loaded in 45.2ms
https://example.com/script.js loaded in 123.7ms
```

> **Note:** `PerformanceObserver` is an advanced topic. Focus on `IntersectionObserver`, `MutationObserver`, and `ResizeObserver` first.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| **Unobserve when done** | Prevents memory leaks (especially for lazy loading) |
| **Use `disconnect()` when destroying components** | Clean up observers to avoid tracking deleted elements |
| **Keep callbacks lightweight** | Observers fire frequently—heavy logic can hurt performance |
| **Use `threshold` carefully** | Multiple thresholds trigger multiple callbacks |
| **Prefer observers over scroll listeners** | More efficient—browser optimizes observer scheduling |
| **Use `subtree: false` when possible** | Watching all descendants is more expensive |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not calling `unobserve()` after lazy loading | Element keeps being watched unnecessarily |
| Forgetting `disconnect()` on cleanup | Memory leak—observer holds references |
| Heavy computation in callbacks | Keep callbacks fast—observers fire often |
| Watching too many elements with MutationObserver | Be specific with selectors and `subtree` option |
| Not checking `isIntersecting` in IntersectionObserver | Callback fires for both entering and exiting |
| Using scroll events instead of IntersectionObserver | Scroll events fire constantly, observers are optimized |

---

## Hands-on Exercise

### Your Task

Build a **lazy-loading infinite scroll feed** for AI-generated content:
- Display 10 initial messages
- When user scrolls near the bottom, load 10 more
- Lazy-load images only when they become visible
- Auto-scroll chat to bottom when new messages are added

### Requirements

1. Create HTML with:
   - `<div id="feed-container"></div>`
   - `<div id="sentinel"></div>` (invisible trigger for infinite scroll)

2. **Initial content:** Generate 10 messages with placeholder images
   - Each message has an `<img data-src="https://via.placeholder.com/150">` (lazy load)

3. **IntersectionObserver for infinite scroll:**
   - Observe the `sentinel` element
   - When `isIntersecting`, load 10 more messages
   - Append new messages to `feed-container`

4. **IntersectionObserver for lazy loading:**
   - Observe all images with `data-src`
   - When visible, set `img.src = img.dataset.src`
   - Unobserve the image after loading

5. **MutationObserver for auto-scroll:**
   - Observe `feed-container` for new children
   - When children added, scroll to bottom

### Expected Result

- Page loads with 10 messages (images don't load yet)
- Scrolling down reveals images as they enter viewport
- Reaching bottom loads 10 more messages
- New messages auto-scroll the feed to bottom

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `IntersectionObserver` with `threshold: 1.0` for sentinel (trigger when fully visible)
- Use `IntersectionObserver` with `threshold: 0` for images (trigger as soon as visible)
- Create messages with `document.createElement()` and `appendChild()`
- Set image `src` from `dataset.src`: `img.src = img.dataset.src`
- Auto-scroll with `container.scrollTop = container.scrollHeight`
- Use `MutationObserver` with `childList: true` to watch for new messages
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**HTML:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lazy Loading Infinite Scroll</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 600px;
      margin: 50px auto;
    }
    #feed-container {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      max-height: 600px;
      overflow-y: scroll;
    }
    .message {
      padding: 15px;
      margin: 10px 0;
      background-color: #f9f9f9;
      border-radius: 8px;
      display: flex;
      align-items: center;
      gap: 15px;
    }
    .message img {
      width: 150px;
      height: 150px;
      border-radius: 8px;
      background-color: #ddd;
    }
    #sentinel {
      height: 20px;
      background-color: transparent;
    }
  </style>
</head>
<body>
  <h1>Infinite Scroll Feed</h1>
  <div id="feed-container">
    <!-- Messages will be added here -->
  </div>
  <div id="sentinel"></div>

  <script src="script.js"></script>
</body>
</html>
```

**JavaScript (script.js):**
```javascript
const feedContainer = document.getElementById('feed-container');
const sentinel = document.getElementById('sentinel');

let messageCount = 0;

// Function to create a message element
function createMessage() {
  messageCount++;
  
  const message = document.createElement('div');
  message.className = 'message';
  
  const img = document.createElement('img');
  img.dataset.src = `https://via.placeholder.com/150?text=Image+${messageCount}`;
  img.alt = `Message ${messageCount}`;
  
  const text = document.createElement('span');
  text.textContent = `Message ${messageCount}: This is AI-generated content.`;
  
  message.appendChild(img);
  message.appendChild(text);
  
  return message;
}

// Load initial messages
function loadMessages(count) {
  for (let i = 0; i < count; i++) {
    const message = createMessage();
    feedContainer.appendChild(message);
  }
  
  console.log(`Loaded ${count} messages`);
}

// 1. IntersectionObserver for infinite scroll
const infiniteObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      console.log('Reached bottom, loading more messages...');
      loadMessages(10);
    }
  });
}, {
  threshold: 1.0 // Trigger when sentinel fully visible
});

infiniteObserver.observe(sentinel);

// 2. IntersectionObserver for lazy loading images
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src; // Load image
      console.log('Image loaded:', img.src);
      imageObserver.unobserve(img); // Stop watching this image
    }
  });
}, {
  threshold: 0 // Trigger as soon as any part is visible
});

// 3. MutationObserver for auto-scroll
const scrollObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
      // Scroll to bottom when new messages added
      feedContainer.scrollTop = feedContainer.scrollHeight;
      
      // Observe new images for lazy loading
      mutation.addedNodes.forEach(node => {
        if (node.querySelector) {
          const img = node.querySelector('img[data-src]');
          if (img) imageObserver.observe(img);
        }
      });
    }
  });
});

scrollObserver.observe(feedContainer, {
  childList: true,
  subtree: false
});

// Load initial 10 messages
loadMessages(10);

// Observe initial images
const initialImages = feedContainer.querySelectorAll('img[data-src]');
initialImages.forEach(img => imageObserver.observe(img));
```

**Output** (in console):
```
Loaded 10 messages
Image loaded: https://via.placeholder.com/150?text=Image+1
Image loaded: https://via.placeholder.com/150?text=Image+2
...
Reached bottom, loading more messages...
Loaded 10 messages
```

</details>

### Bonus Challenges

- [ ] Add a "Back to Top" button that appears when user scrolls down
- [ ] Stop infinite scroll after 50 messages (show "End of feed")
- [ ] Add a loading spinner while messages are being added
- [ ] Implement a `ResizeObserver` to adjust message layout on window resize

---

## Summary

✅ **IntersectionObserver** detects visibility (lazy loading, infinite scroll)  
✅ **MutationObserver** watches DOM changes (new elements, attributes, text)  
✅ **ResizeObserver** tracks size changes (responsive layouts)  
✅ **PerformanceObserver** monitors metrics (advanced performance tracking)  
✅ **Always unobserve or disconnect** to prevent memory leaks  
✅ **Observers are more efficient** than polling or scroll events

**Next:** [Template-based Creation](./06-template-based-creation.md)

---

## Further Reading

- [MDN: IntersectionObserver](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver) - Visibility detection
- [MDN: MutationObserver](https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver) - DOM change tracking
- [MDN: ResizeObserver](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver) - Size change detection
- [Template-based Creation](./06-template-based-creation.md) - Next lesson

<!-- 
Sources Consulted:
- MDN IntersectionObserver: https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver
- MDN MutationObserver: https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
- MDN ResizeObserver: https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
- MDN PerformanceObserver: https://developer.mozilla.org/en-US/docs/Web/API/PerformanceObserver
-->
