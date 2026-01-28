---
title: "View Transitions"
---

# View Transitions

## Introduction

The View Transitions API enables smooth animated transitions between different views or states—page navigations, DOM updates, or multi-page applications. It lets you create native app-like experiences with minimal code.

For AI applications, view transitions can smooth the experience when switching between conversations, expanding chat windows, or navigating between different AI tools.

### What We'll Cover

- Basic view transitions for DOM updates
- Same-document transitions (SPA)
- Cross-document transitions (MPA)
- Customizing transition animations
- Named view transitions for complex layouts
- Accessibility and performance considerations

### Prerequisites

- CSS animations and transitions
- JavaScript DOM manipulation basics
- Understanding of async/await

---

## Basic View Transitions

### The `startViewTransition()` API

Wrap DOM changes in a view transition:

```javascript
document.startViewTransition(() => {
  // Make your DOM changes here
  container.innerHTML = newContent;
});
```

The browser automatically:
1. Captures the old state
2. Runs your update callback
3. Captures the new state
4. Animates between them

### Default Animation

Without any CSS, you get a cross-fade:

```javascript
button.addEventListener('click', async () => {
  if (!document.startViewTransition) {
    // Fallback for browsers without support
    updateDOM();
    return;
  }
  
  await document.startViewTransition(() => {
    updateDOM();
  }).ready;
});
```

---

## Same-Document Transitions (SPA)

For single-page applications with client-side routing:

```javascript
// Simple page transition
async function navigateTo(url) {
  const response = await fetch(url);
  const html = await response.text();
  
  if (!document.startViewTransition) {
    document.body.innerHTML = html;
    return;
  }
  
  const transition = document.startViewTransition(() => {
    document.body.innerHTML = html;
  });
  
  await transition.finished;
}
```

### With Navigation API

```javascript
navigation.addEventListener('navigate', (event) => {
  if (!event.canIntercept) return;
  
  event.intercept({
    async handler() {
      const response = await fetch(event.destination.url);
      const html = await response.text();
      
      await document.startViewTransition(() => {
        document.body.innerHTML = html;
      }).finished;
    }
  });
});
```

---

## Cross-Document Transitions (MPA)

For traditional multi-page applications, enable with CSS:

```css
/* Enable cross-document transitions */
@view-transition {
  navigation: auto;
}
```

The browser handles the rest automatically for same-origin navigations.

### Opt Out Specific Pages

```css
/* Disable for certain pages */
.no-transition {
  view-transition-name: none;
}
```

### Meta Tag Control

```html
<!-- Opt into view transitions -->
<meta name="view-transition" content="same-origin">
```

---

## Customizing Animations

### The Default Pseudo-Elements

View transitions create a pseudo-element tree:

```
::view-transition
└── ::view-transition-group(root)
    └── ::view-transition-image-pair(root)
        ├── ::view-transition-old(root)
        └── ::view-transition-new(root)
```

### Custom Fade Animation

```css
/* Longer fade */
::view-transition-old(root),
::view-transition-new(root) {
  animation-duration: 0.5s;
}

/* Asymmetric timing */
::view-transition-old(root) {
  animation: fade-out 0.25s ease-in forwards;
}

::view-transition-new(root) {
  animation: fade-in 0.5s ease-out forwards;
}

@keyframes fade-out {
  to { opacity: 0; }
}

@keyframes fade-in {
  from { opacity: 0; }
}
```

### Slide Transitions

```css
/* Slide from right */
@keyframes slide-from-right {
  from {
    transform: translateX(100%);
  }
}

@keyframes slide-to-left {
  to {
    transform: translateX(-100%);
  }
}

::view-transition-old(root) {
  animation: slide-to-left 0.3s ease-in-out;
}

::view-transition-new(root) {
  animation: slide-from-right 0.3s ease-in-out;
}
```

### Direction-Based Transitions

```css
/* Going forward */
.transition-forward::view-transition-old(root) {
  animation: slide-to-left 0.3s;
}

.transition-forward::view-transition-new(root) {
  animation: slide-from-right 0.3s;
}

/* Going back */
.transition-back::view-transition-old(root) {
  animation: slide-to-right 0.3s;
}

.transition-back::view-transition-new(root) {
  animation: slide-from-left 0.3s;
}
```

```javascript
function navigate(url, isBack = false) {
  document.documentElement.classList.toggle('transition-forward', !isBack);
  document.documentElement.classList.toggle('transition-back', isBack);
  
  document.startViewTransition(() => {
    // Update content
  });
}
```

---

## Named View Transitions

Give elements persistent identity across views:

```css
/* Page 1 */
.card-header {
  view-transition-name: hero-header;
}

/* Page 2 - same name = shared element transition */
.detail-header {
  view-transition-name: hero-header;
}
```

### How It Works

1. Elements with the same `view-transition-name` animate between positions
2. Like "hero" transitions in native apps
3. Must be unique per view (only one element per name)

### Morphing Elements

```css
/* Thumbnail */
.product-thumbnail {
  view-transition-name: product-image;
  width: 100px;
  height: 100px;
  border-radius: 8px;
}

/* Detail image */
.product-detail-image {
  view-transition-name: product-image;
  width: 100%;
  max-width: 500px;
  border-radius: 16px;
}
```

The element automatically:
- Animates size change
- Animates position change
- Cross-fades any style differences

### Multiple Named Transitions

```css
/* Header persists */
.site-header {
  view-transition-name: header;
}

/* Navigation persists */
.site-nav {
  view-transition-name: navigation;
}

/* Content changes with custom animation */
.main-content {
  view-transition-name: main;
}

/* Sidebar slides */
.sidebar {
  view-transition-name: sidebar;
}

/* Each can have unique animation */
::view-transition-old(main) {
  animation: fade-out 0.2s ease-out;
}

::view-transition-new(main) {
  animation: slide-up 0.3s ease-out;
}

::view-transition-old(sidebar),
::view-transition-new(sidebar) {
  animation-duration: 0s; /* No animation for sidebar */
}
```

---

## Dynamic View Transition Names

Set names dynamically with JavaScript:

```javascript
function selectCard(card) {
  // Give the clicked card a transition name
  card.style.viewTransitionName = 'selected-card';
  
  document.startViewTransition(() => {
    // Show detail view
    showDetailView(card.dataset.id);
  }).finished.then(() => {
    // Clean up
    card.style.viewTransitionName = '';
  });
}
```

### With CSS Custom Properties

```css
.card {
  view-transition-name: var(--card-transition-name, none);
}
```

```javascript
card.style.setProperty('--card-transition-name', `card-${card.id}`);
```

---

## Transition Lifecycle

### Promise-Based API

```javascript
const transition = document.startViewTransition(() => {
  updateDOM();
});

// When transition starts
await transition.ready;

// When DOM is updated
await transition.updateCallbackDone;

// When animation completes
await transition.finished;
```

### Running Custom JavaScript

```javascript
const transition = document.startViewTransition(() => {
  updateDOM();
});

// Add class during transition
await transition.ready;
document.documentElement.classList.add('transitioning');

await transition.finished;
document.documentElement.classList.remove('transitioning');
```

### Handling Errors

```javascript
try {
  const transition = document.startViewTransition(() => {
    updateDOM();
  });
  
  await transition.finished;
} catch (error) {
  console.error('Transition failed:', error);
  // DOM still updated, just no animation
}
```

---

## AI Interface Examples

### Chat Message Transitions

```css
.message {
  view-transition-name: none;
}

.message.new {
  view-transition-name: new-message;
}

::view-transition-new(new-message) {
  animation: message-in 0.3s ease-out;
}

@keyframes message-in {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
}
```

```javascript
function addMessage(content, isUser) {
  const message = createMessageElement(content, isUser);
  message.classList.add('new');
  
  document.startViewTransition(() => {
    chatContainer.appendChild(message);
  }).finished.then(() => {
    message.classList.remove('new');
  });
}
```

### Expanding Chat Window

```css
.chat-widget {
  view-transition-name: chat;
}

.chat-widget.minimized {
  width: 60px;
  height: 60px;
  border-radius: 50%;
}

.chat-widget.expanded {
  width: 400px;
  height: 600px;
  border-radius: 16px;
}

/* Smooth expansion animation */
::view-transition-group(chat) {
  animation-duration: 0.4s;
  animation-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Conversation Switching

```css
.conversation-list {
  view-transition-name: conversations;
}

.conversation-detail {
  view-transition-name: detail;
}

/* Slide in conversation list */
::view-transition-old(conversations) {
  animation: slide-out-left 0.3s ease-in;
}

::view-transition-new(detail) {
  animation: slide-in-right 0.3s ease-out;
}

/* Reverse when going back */
.going-back::view-transition-old(detail) {
  animation: slide-out-right 0.3s ease-in;
}

.going-back::view-transition-new(conversations) {
  animation: slide-in-left 0.3s ease-out;
}
```

---

## Performance Optimization

### Hardware Acceleration

```css
::view-transition-old(root),
::view-transition-new(root) {
  /* Force GPU layer */
  will-change: transform, opacity;
}
```

### Reduce Work During Transition

```javascript
const transition = document.startViewTransition(async () => {
  // Minimal DOM changes in callback
  content.innerHTML = newHTML;
});

// Heavy work after animation starts
transition.ready.then(() => {
  loadImages();
  initializeComponents();
});
```

### Skip Animations When Appropriate

```javascript
function navigate(url) {
  // Skip for same-page anchor links
  if (isSamePageAnchor(url)) {
    updateContent();
    return;
  }
  
  // Skip if user prefers reduced motion
  if (prefersReducedMotion()) {
    updateContent();
    return;
  }
  
  document.startViewTransition(() => updateContent());
}
```

---

## Accessibility

### Respecting Motion Preferences

```css
@media (prefers-reduced-motion: reduce) {
  ::view-transition-old(root),
  ::view-transition-new(root) {
    animation: none;
    mix-blend-mode: normal;
  }
}
```

### JavaScript Check

```javascript
const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
).matches;

function updateView() {
  if (prefersReducedMotion || !document.startViewTransition) {
    updateDOM();
    return;
  }
  
  document.startViewTransition(() => updateDOM());
}
```

### Focus Management

```javascript
const transition = document.startViewTransition(() => {
  updateDOM();
});

await transition.finished;

// Restore focus to appropriate element
const focusTarget = document.querySelector('[data-focus-target]');
focusTarget?.focus();
```

---

## Browser Support and Fallbacks

### Feature Detection

```javascript
if (!document.startViewTransition) {
  // Fallback for unsupported browsers
  updateDOM();
} else {
  document.startViewTransition(() => updateDOM());
}
```

### Polyfill Pattern

```javascript
async function viewTransition(callback) {
  if (document.startViewTransition) {
    const transition = document.startViewTransition(callback);
    return transition.finished;
  }
  
  // No transition, just run the callback
  await callback();
  return Promise.resolve();
}

// Usage
await viewTransition(() => {
  container.innerHTML = newContent;
});
```

### CSS Fallback

```css
/* Base styles without transitions */
.content {
  opacity: 1;
}

/* Enhanced with view transitions */
@supports (view-transition-name: test) {
  .content {
    view-transition-name: content;
  }
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always feature detect | Not all browsers support view transitions |
| Keep transitions short (<400ms) | Feels responsive, not slow |
| Use `prefers-reduced-motion` | Accessibility requirement |
| Name persistent elements | Better hero transitions |
| Clean up dynamic names | Prevent conflicts |
| Test with slow network | Ensure content loads before transition |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Duplicate `view-transition-name` | Each name must be unique per view |
| Missing feature detection | Always check `document.startViewTransition` |
| Heavy DOM changes in callback | Keep callback minimal, defer heavy work |
| Ignoring reduced motion | Always provide alternative |
| Too long animations | Keep under 400ms for responsiveness |
| Forgetting cross-origin limits | View transitions only work same-origin |

---

## Hands-on Exercise

### Your Task

Create an AI chat interface with view transitions:

1. Message list with smooth message additions
2. Expandable/collapsible sidebar with conversation list
3. Click conversation to smoothly transition to its messages
4. "New chat" button with entrance animation
5. Respect reduced motion preference

<details>
<summary>✅ Solution</summary>

```css
/* Base layout */
.chat-app {
  display: flex;
  height: 100vh;
}

/* Sidebar transitions */
.sidebar {
  view-transition-name: sidebar;
  width: 300px;
  border-right: 1px solid #e5e7eb;
}

.sidebar.collapsed {
  width: 0;
  overflow: hidden;
}

::view-transition-group(sidebar) {
  animation-duration: 0.3s;
  animation-timing-function: ease-in-out;
}

/* Main chat area */
.chat-main {
  view-transition-name: chat-main;
  flex: 1;
  display: flex;
  flex-direction: column;
}

/* Conversation list item */
.conversation-item {
  padding: 1rem;
  cursor: pointer;
  border-bottom: 1px solid #e5e7eb;
}

.conversation-item.active {
  background: #f0f4ff;
}

/* Messages container */
.messages {
  view-transition-name: messages;
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

/* Individual message */
.message {
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 1rem;
  max-width: 80%;
}

.message.new {
  view-transition-name: new-message;
}

::view-transition-new(new-message) {
  animation: message-appear 0.3s ease-out;
}

@keyframes message-appear {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
}

/* Conversation switch animation */
::view-transition-old(messages) {
  animation: fade-slide-out 0.2s ease-in;
}

::view-transition-new(messages) {
  animation: fade-slide-in 0.3s ease-out;
}

@keyframes fade-slide-out {
  to {
    opacity: 0;
    transform: translateX(-20px);
  }
}

@keyframes fade-slide-in {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
}

/* New chat button */
.new-chat-btn {
  view-transition-name: new-chat;
}

::view-transition-new(new-chat) {
  animation: pop-in 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes pop-in {
  from {
    transform: scale(0.8);
    opacity: 0;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.01ms;
  }
}
```

```javascript
class ChatApp {
  constructor() {
    this.prefersReducedMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches;
  }
  
  viewTransition(callback) {
    if (this.prefersReducedMotion || !document.startViewTransition) {
      callback();
      return Promise.resolve();
    }
    
    return document.startViewTransition(callback).finished;
  }
  
  async addMessage(content, isUser = false) {
    const message = document.createElement('div');
    message.className = `message ${isUser ? 'user' : 'assistant'} new`;
    message.textContent = content;
    
    await this.viewTransition(() => {
      this.messagesContainer.appendChild(message);
    });
    
    message.classList.remove('new');
  }
  
  async switchConversation(conversationId) {
    const messages = await this.fetchMessages(conversationId);
    
    await this.viewTransition(() => {
      this.renderMessages(messages);
      this.updateActiveConversation(conversationId);
    });
  }
  
  async toggleSidebar() {
    await this.viewTransition(() => {
      this.sidebar.classList.toggle('collapsed');
    });
  }
  
  async createNewChat() {
    const newConversation = await this.api.createConversation();
    
    await this.viewTransition(() => {
      this.addConversationToList(newConversation);
      this.clearMessages();
      this.updateActiveConversation(newConversation.id);
    });
  }
}

// Initialize
const chatApp = new ChatApp();

// Event listeners
document.querySelector('.new-chat-btn').addEventListener('click', () => {
  chatApp.createNewChat();
});

document.querySelector('.toggle-sidebar').addEventListener('click', () => {
  chatApp.toggleSidebar();
});

document.querySelectorAll('.conversation-item').forEach(item => {
  item.addEventListener('click', () => {
    chatApp.switchConversation(item.dataset.id);
  });
});
```

```html
<div class="chat-app">
  <aside class="sidebar">
    <button class="new-chat-btn">+ New Chat</button>
    <div class="conversation-list">
      <div class="conversation-item" data-id="1">
        Previous conversation...
      </div>
    </div>
  </aside>
  
  <main class="chat-main">
    <header class="chat-header">
      <button class="toggle-sidebar">☰</button>
      <h1>AI Assistant</h1>
    </header>
    
    <div class="messages">
      <!-- Messages rendered here -->
    </div>
    
    <div class="chat-input">
      <textarea placeholder="Type a message..."></textarea>
      <button>Send</button>
    </div>
  </main>
</div>
```
</details>

---

## Summary

✅ **View Transitions API** animates between DOM states with `startViewTransition()`

✅ **Same-document transitions** work for SPAs with JavaScript routing

✅ **Cross-document transitions** enable MPA transitions with `@view-transition`

✅ **Named transitions** with `view-transition-name` create hero animations

✅ **Customize** with `::view-transition-*` pseudo-elements

✅ **Always feature detect** and respect `prefers-reduced-motion`

✅ Use **short durations** (<400ms) for responsive feel

---

**Previous:** [Modern CSS Features](./09-modern-css-features.md)

**Back to:** [CSS Fundamentals Overview](./00-css-fundamentals.md)

---

## Further Reading

- [MDN View Transitions API](https://developer.mozilla.org/en-US/docs/Web/API/View_Transitions_API)
- [Chrome Developers: View Transitions](https://developer.chrome.com/docs/web-platform/view-transitions)
- [web.dev: View Transitions](https://web.dev/articles/view-transitions)

<!-- 
Sources Consulted:
- MDN View Transitions API: https://developer.mozilla.org/en-US/docs/Web/API/View_Transitions_API
- Chrome Developers View Transitions: https://developer.chrome.com/docs/web-platform/view-transitions
- web.dev View Transitions: https://web.dev/articles/view-transitions
-->
