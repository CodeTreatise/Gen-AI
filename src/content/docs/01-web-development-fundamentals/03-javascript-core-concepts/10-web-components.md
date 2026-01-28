---
title: "Web Components"
---

# Web Components

## Introduction

Web Components let you create reusable custom HTML elements with encapsulated functionality. Built on web standards (Custom Elements, Shadow DOM, HTML Templates), they work across frameworks and vanilla JavaScript. For AI applications, Web Components enable you to build chat interfaces, message bubbles, loading indicators, and interactive widgets that are truly modular and framework-agnostic.

Unlike framework-specific components, Web Components use native browser APIs—no build step required. They're especially useful for design systems, widget distribution, and micro-frontends.

### What We'll Cover
- Custom Elements: defining new HTML tags
- Shadow DOM: style and markup encapsulation
- HTML Templates and Slots: reusable markup
- Lifecycle callbacks
- Practical patterns for AI interfaces

### Prerequisites
- JavaScript classes
- DOM manipulation basics
- Understanding of HTML and CSS

---

## Custom Elements

Define your own HTML elements:

```javascript
class ChatMessage extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }
  
  connectedCallback() {
    const role = this.getAttribute('role') || 'user';
    const content = this.getAttribute('content') || '';
    
    this.shadowRoot.innerHTML = `
      <style>
        .message {
          padding: 10px;
          margin: 5px 0;
          border-radius: 8px;
          max-width: 80%;
        }
        .user {
          background: #007bff;
          color: white;
          margin-left: auto;
        }
        .assistant {
          background: #f1f1f1;
          color: black;
        }
      </style>
      <div class="message ${role}">
        <strong>${role}:</strong> ${content}
      </div>
    `;
  }
}

// Register the element
customElements.define('chat-message', ChatMessage);
```

**HTML Usage:**
```html
<chat-message role="user" content="Hello!"></chat-message>
<chat-message role="assistant" content="Hi there!"></chat-message>
```

**Result:** Styled message bubbles with encapsulated CSS.

---

## Shadow DOM

Encapsulates styles and markup:

```javascript
class StyledButton extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    
    this.shadowRoot.innerHTML = `
      <style>
        button {
          background: #4CAF50;
          color: white;
          padding: 10px 20px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        button:hover {
          background: #45a049;
        }
      </style>
      <button>
        <slot></slot>
      </button>
    `;
  }
}

customElements.define('styled-button', StyledButton);
```

**HTML Usage:**
```html
<styled-button>Click Me</styled-button>

<style>
  /* This won't affect the button inside Shadow DOM */
  button {
    background: red;
  }
</style>
```

The button stays green—Shadow DOM styles are encapsulated!

### Shadow DOM modes

- `mode: 'open'`: Accessible via `element.shadowRoot`
- `mode: 'closed'`: Not accessible (more encapsulation)

---

## Lifecycle Callbacks

Custom elements have lifecycle hooks:

```javascript
class ObservedElement extends HTMLElement {
  // 1. Called when element is created
  constructor() {
    super();
    console.log('Constructor called');
  }
  
  // 2. Called when element is added to DOM
  connectedCallback() {
    console.log('Connected to DOM');
    this.render();
  }
  
  // 3. Called when element is removed from DOM
  disconnectedCallback() {
    console.log('Disconnected from DOM');
    this.cleanup();
  }
  
  // 4. Called when observed attribute changes
  attributeChangedCallback(name, oldValue, newValue) {
    console.log(`Attribute ${name} changed from ${oldValue} to ${newValue}`);
    this.render();
  }
  
  // 5. Specify which attributes to observe
  static get observedAttributes() {
    return ['title', 'count'];
  }
  
  render() {
    const title = this.getAttribute('title') || 'Default';
    const count = this.getAttribute('count') || '0';
    this.innerHTML = `<h3>${title}: ${count}</h3>`;
  }
  
  cleanup() {
    // Clean up event listeners, timers, etc.
  }
}

customElements.define('observed-element', ObservedElement);
```

**HTML Usage:**
```html
<observed-element title="Messages" count="5"></observed-element>

<script>
  const el = document.querySelector('observed-element');
  el.setAttribute('count', '10');  // Triggers attributeChangedCallback
</script>
```

---

## HTML Templates and Slots

### Templates

Inert markup that can be cloned:

```html
<template id="message-template">
  <style>
    .message {
      padding: 10px;
      border-left: 3px solid #007bff;
      margin: 5px 0;
    }
  </style>
  <div class="message">
    <slot name="author">Anonymous</slot>:
    <slot name="content"></slot>
  </div>
</template>

<script>
  class TemplateMessage extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      
      const template = document.getElementById('message-template');
      this.shadowRoot.appendChild(template.content.cloneNode(true));
    }
  }
  
  customElements.define('template-message', TemplateMessage);
</script>
```

### Slots

Named placeholders for content projection:

```html
<template-message>
  <span slot="author">Alice</span>
  <span slot="content">Hello, how are you?</span>
</template-message>
```

**Result:** "Alice: Hello, how are you?" with template styling.

---

## Practical Example: AI Chat Widget

Complete chat interface component:

```javascript
class AiChatWidget extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.messages = [];
  }
  
  connectedCallback() {
    this.render();
    this.setupEventListeners();
  }
  
  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          width: 400px;
          height: 600px;
          border: 1px solid #ccc;
          border-radius: 8px;
          font-family: Arial, sans-serif;
        }
        
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
        }
        
        .messages {
          flex: 1;
          overflow-y: auto;
          padding: 10px;
          background: #f9f9f9;
        }
        
        .message {
          padding: 8px 12px;
          margin: 5px 0;
          border-radius: 12px;
          max-width: 80%;
        }
        
        .user {
          background: #007bff;
          color: white;
          margin-left: auto;
        }
        
        .assistant {
          background: white;
          color: black;
          border: 1px solid #ddd;
        }
        
        .input-area {
          display: flex;
          padding: 10px;
          border-top: 1px solid #ccc;
        }
        
        input {
          flex: 1;
          padding: 8px;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
        
        button {
          margin-left: 5px;
          padding: 8px 16px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        button:hover {
          background: #0056b3;
        }
      </style>
      
      <div class="chat-container">
        <div class="messages" id="messages"></div>
        <div class="input-area">
          <input type="text" id="input" placeholder="Type a message...">
          <button id="send">Send</button>
        </div>
      </div>
    `;
  }
  
  setupEventListeners() {
    const input = this.shadowRoot.getElementById('input');
    const sendButton = this.shadowRoot.getElementById('send');
    
    const sendMessage = () => {
      const content = input.value.trim();
      if (!content) return;
      
      this.addMessage('user', content);
      input.value = '';
      
      // Simulate AI response
      setTimeout(() => {
        this.addMessage('assistant', `Echo: ${content}`);
      }, 500);
    };
    
    sendButton.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
  }
  
  addMessage(role, content) {
    this.messages.push({ role, content, timestamp: Date.now() });
    
    const messagesContainer = this.shadowRoot.getElementById('messages');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    messageEl.textContent = content;
    messagesContainer.appendChild(messageEl);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Dispatch custom event
    this.dispatchEvent(new CustomEvent('message-sent', {
      detail: { role, content },
      bubbles: true,
      composed: true
    }));
  }
}

customElements.define('ai-chat-widget', AiChatWidget);
```

**HTML Usage:**
```html
<ai-chat-widget></ai-chat-widget>

<script>
  document.querySelector('ai-chat-widget')
    .addEventListener('message-sent', (e) => {
      console.log('Message:', e.detail);
    });
</script>
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Shadow DOM for style encapsulation | Prevents global CSS conflicts |
| Dispatch custom events for communication | Enables parent-child interaction |
| Use `:host` for component-level styling | Styles the custom element itself |
| Provide slots for flexible content | Allows consumers to customize markup |
| Use lifecycle callbacks appropriately | `connectedCallback` for setup, `disconnectedCallback` for cleanup |
| Keep components focused | Single responsibility—one component, one purpose |
| Use attributes for configuration | Enables declarative HTML usage |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting `super()` in constructor | Always call `super()` first |
| Using `innerHTML` before `connectedCallback` | Wait until element is in DOM |
| Not cleaning up in `disconnectedCallback` | Remove event listeners to prevent memory leaks |
| Forgetting to define `observedAttributes` | Attribute changes won't trigger callbacks |
| Using global styles with Shadow DOM | Styles need to be inside shadowRoot |
| Not handling missing attributes | Provide defaults in `getAttribute()` |

---

## Hands-on Exercise

### Your Task
Create a `<loading-spinner>` Web Component with customizable size, color, and text. Use Shadow DOM, attributes, and lifecycle callbacks.

### Requirements
1. Custom element: `<loading-spinner>`
2. Attributes: `size` (small/medium/large), `color`, `text`
3. Shadow DOM with encapsulated styles
4. Observe attribute changes and re-render
5. Animated spinner with CSS

### Expected Result
```html
<loading-spinner size="large" color="#007bff" text="Loading..."></loading-spinner>
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `attachShadow({ mode: 'open' })`
- Define `observedAttributes` to watch `size`, `color`, `text`
- Use CSS animations for spinner rotation
- Use `attributeChangedCallback` to trigger re-render
- Use CSS variables for dynamic colors
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class LoadingSpinner extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }
  
  static get observedAttributes() {
    return ['size', 'color', 'text'];
  }
  
  connectedCallback() {
    this.render();
  }
  
  attributeChangedCallback() {
    if (this.shadowRoot.innerHTML) {
      this.render();
    }
  }
  
  render() {
    const size = this.getAttribute('size') || 'medium';
    const color = this.getAttribute('color') || '#007bff';
    const text = this.getAttribute('text') || 'Loading...';
    
    const sizeMap = {
      small: '20px',
      medium: '40px',
      large: '60px'
    };
    
    const spinnerSize = sizeMap[size] || sizeMap.medium;
    
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: inline-block;
          text-align: center;
        }
        
        .spinner-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
        }
        
        .spinner {
          width: ${spinnerSize};
          height: ${spinnerSize};
          border: 4px solid #f3f3f3;
          border-top: 4px solid ${color};
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .text {
          color: ${color};
          font-size: 14px;
        }
      </style>
      
      <div class="spinner-container">
        <div class="spinner"></div>
        <div class="text">${text}</div>
      </div>
    `;
  }
}

customElements.define('loading-spinner', LoadingSpinner);

// Test
document.body.innerHTML += `
  <loading-spinner size="small" color="green" text="Processing..."></loading-spinner>
  <loading-spinner size="medium" color="#ff6b6b" text="Generating..."></loading-spinner>
  <loading-spinner size="large" color="#4ecdc4" text="AI Thinking..."></loading-spinner>
`;

// Dynamic update
setTimeout(() => {
  const spinner = document.querySelector('loading-spinner');
  spinner.setAttribute('text', 'Almost done!');
  spinner.setAttribute('color', '#ff9800');
}, 2000);
```
</details>

### Bonus Challenges
- [ ] Add `speed` attribute to control animation speed
- [ ] Support different spinner styles (dots, bars, pulse)
- [ ] Add `show` attribute to toggle visibility
- [ ] Emit `complete` event after a timeout

---

## Summary

✅ Custom Elements define new HTML tags with JavaScript classes
✅ Shadow DOM encapsulates styles and markup from global scope
✅ Use lifecycle callbacks (`connectedCallback`, etc.) for setup/teardown
✅ Slots enable flexible content projection into components
✅ Web Components work across frameworks—they're web standards

[Previous: Modules](./09-modules.md) | [Next: Signals](./11-signals.md)

---

<!-- 
Sources Consulted:
- MDN Web Components: https://developer.mozilla.org/en-US/docs/Web/API/Web_components
- MDN Custom Elements: https://developer.mozilla.org/en-US/docs/Web/API/Window/customElements
- MDN Shadow DOM: https://developer.mozilla.org/en-US/docs/Web/API/Web_components/Using_shadow_DOM
- MDN HTML Templates: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/template
-->
