---
title: "Message Container Structure"
---

# Message Container Structure

## Introduction

A well-structured message container is the foundation of a maintainable chat interface. The right HTML hierarchy makes styling predictable, accessibility manageable, and future enhancements easier to implement.

In this lesson, we'll design a message container architecture that handles all the complexity of modern chat UIs.

### What We'll Cover

- Semantic wrapper element hierarchy
- Content area structure and layout
- Action button placement strategies
- Metadata positioning patterns
- React and vanilla JS implementations

### Prerequisites

- HTML semantic elements ([Unit 1](../../../01-web-development-fundamentals/01-html-essentials/00-html-essentials.md))
- CSS Flexbox layout
- Understanding of component composition

---

## Wrapper Element Hierarchy

### The Three-Layer Structure

A robust message component uses three nested layers:

```html
<!-- Layer 1: Message Wrapper (positioning, alignment) -->
<div class="message-wrapper user">
  
  <!-- Layer 2: Message Container (visual styling) -->
  <article class="message-container">
    
    <!-- Layer 3: Content Sections -->
    <header class="message-header">...</header>
    <div class="message-body">...</div>
    <footer class="message-footer">...</footer>
    
  </article>
  
  <!-- Actions (outside container for positioning) -->
  <div class="message-actions">...</div>
</div>
```

### Why Three Layers?

| Layer | Responsibility | CSS Focus |
|-------|----------------|-----------|
| **Wrapper** | Positioning, alignment, spacing | `flex`, `margin`, `gap` |
| **Container** | Visual styling (bg, border, shadow) | `background`, `border-radius`, `padding` |
| **Sections** | Content structure, internal layout | `display`, internal spacing |

### Complete HTML Structure

```html
<div 
  class="message-wrapper user"
  data-message-id="msg-123"
  data-sender="user"
>
  <!-- Optional: Avatar -->
  <div class="message-avatar" aria-hidden="true">
    <img src="/avatar.jpg" alt="">
  </div>
  
  <!-- Main Container -->
  <article 
    class="message-container"
    tabindex="0"
    aria-label="Your message"
  >
    <!-- Header: Sender, timestamp, badges -->
    <header class="message-header">
      <span class="sender-name">You</span>
      <time datetime="2026-01-29T14:30:00" class="message-time">
        2:30 PM
      </time>
    </header>
    
    <!-- Body: Main content -->
    <div class="message-body">
      <p>How do I implement a chat interface?</p>
    </div>
    
    <!-- Footer: Metadata, status -->
    <footer class="message-footer">
      <span class="message-status">Sent</span>
    </footer>
  </article>
  
  <!-- Actions: Copy, edit, delete -->
  <div class="message-actions" role="group" aria-label="Message actions">
    <button aria-label="Edit message">Edit</button>
    <button aria-label="Delete message">Delete</button>
  </div>
</div>
```

---

## Content Area Structure

### Message Header

Contains sender info, timestamp, and contextual badges:

```css
.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
  font-size: 0.875rem;
}

.sender-name {
  font-weight: 600;
  color: #374151;
}

.message-time {
  color: #9ca3af;
  font-size: 0.75rem;
}

.model-badge {
  padding: 0.125rem 0.5rem;
  background: #e5e7eb;
  border-radius: 1rem;
  font-size: 0.75rem;
  color: #4b5563;
}
```

```html
<header class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge">GPT-4o</span>
  <time class="message-time">2:31 PM</time>
</header>
```

### Message Body

The main content area—handles text, code, images, and more:

```css
.message-body {
  line-height: 1.6;
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.message-body p {
  margin: 0 0 0.75rem;
}

.message-body p:last-child {
  margin-bottom: 0;
}

/* Code blocks */
.message-body pre {
  margin: 0.75rem 0;
  padding: 1rem;
  background: #1f2937;
  border-radius: 0.5rem;
  overflow-x: auto;
}

.message-body code {
  font-family: 'Fira Code', monospace;
  font-size: 0.875em;
}

/* Inline code */
.message-body :not(pre) > code {
  padding: 0.125rem 0.375rem;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 0.25rem;
}
```

### Message Footer

Metadata like token counts, edit status, or reaction buttons:

```css
.message-footer {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(0, 0, 0, 0.05);
  font-size: 0.75rem;
  color: #6b7280;
}

.token-count {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.message-status {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.message-status.sent::before {
  content: '✓';
  color: #10b981;
}

.message-status.pending::before {
  content: '◌';
  animation: pulse 1s infinite;
}
```

---

## Action Button Placement

### Hover-Revealed Actions

Show actions only on hover/focus for a cleaner interface:

```css
.message-wrapper {
  position: relative;
}

.message-actions {
  position: absolute;
  top: 0;
  right: 0;
  transform: translateY(-50%);
  display: flex;
  gap: 0.25rem;
  padding: 0.25rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.15s ease, visibility 0.15s ease;
}

.message-wrapper:hover .message-actions,
.message-wrapper:focus-within .message-actions {
  opacity: 1;
  visibility: visible;
}

.message-actions button {
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  color: #6b7280;
}

.message-actions button:hover {
  background: #f3f4f6;
  color: #374151;
}
```

```html
<div class="message-actions" role="group" aria-label="Message actions">
  <button aria-label="Copy message" title="Copy">
    <svg width="16" height="16" viewBox="0 0 24 24"><!-- copy icon --></svg>
  </button>
  <button aria-label="Edit message" title="Edit">
    <svg width="16" height="16" viewBox="0 0 24 24"><!-- edit icon --></svg>
  </button>
  <button aria-label="Regenerate response" title="Regenerate">
    <svg width="16" height="16" viewBox="0 0 24 24"><!-- refresh icon --></svg>
  </button>
</div>
```

### Inline Actions (Always Visible)

For key actions, keep them visible in the footer:

```html
<footer class="message-footer">
  <span class="token-count">342 tokens</span>
  <div class="inline-actions">
    <button class="action-btn" aria-label="Copy response">
      <svg><!-- copy --></svg>
      Copy
    </button>
    <button class="action-btn" aria-label="Regenerate">
      <svg><!-- refresh --></svg>
      Regenerate
    </button>
  </div>
</footer>
```

```css
.inline-actions {
  display: flex;
  gap: 0.5rem;
  margin-left: auto;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.375rem 0.625rem;
  background: none;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  color: #6b7280;
  cursor: pointer;
}

.action-btn:hover {
  background: #f9fafb;
  border-color: #d1d5db;
  color: #374151;
}

.action-btn svg {
  width: 14px;
  height: 14px;
}
```

### Mobile-Friendly Actions

On mobile, use a long-press menu instead of hover:

```javascript
let longPressTimer;

function setupLongPressActions() {
  const messages = document.querySelectorAll('.message-wrapper');
  
  messages.forEach(message => {
    message.addEventListener('touchstart', (e) => {
      longPressTimer = setTimeout(() => {
        showActionMenu(e.target.closest('.message-wrapper'));
      }, 500);
    });
    
    message.addEventListener('touchend', () => {
      clearTimeout(longPressTimer);
    });
    
    message.addEventListener('touchmove', () => {
      clearTimeout(longPressTimer);
    });
  });
}

function showActionMenu(messageWrapper) {
  const menu = document.createElement('div');
  menu.className = 'action-menu-mobile';
  menu.innerHTML = `
    <button data-action="copy">Copy</button>
    <button data-action="edit">Edit</button>
    <button data-action="delete">Delete</button>
  `;
  
  // Position and show menu
  document.body.appendChild(menu);
  
  // Add backdrop to close
  const backdrop = document.createElement('div');
  backdrop.className = 'action-menu-backdrop';
  backdrop.onclick = () => {
    menu.remove();
    backdrop.remove();
  };
  document.body.appendChild(backdrop);
}
```

---

## Metadata Positioning

### Top vs Bottom Placement

| Position | Best For | Example |
|----------|----------|---------|
| **Header** | Sender, model, timestamp | "Assistant · GPT-4o · 2:31 PM" |
| **Footer** | Token count, reactions, actions | "342 tokens · 👍 3" |
| **Inline** | Citations, links | "According to [source]..." |
| **Hover** | Full timestamp, message ID | Tooltip with details |

### Flexible Metadata Layout

```css
.message-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.meta-separator {
  width: 3px;
  height: 3px;
  background: currentColor;
  border-radius: 50%;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}
```

```html
<div class="message-meta">
  <span class="meta-item">
    <svg class="meta-icon"><!-- clock --></svg>
    2:31 PM
  </span>
  <span class="meta-separator" aria-hidden="true"></span>
  <span class="meta-item">
    <svg class="meta-icon"><!-- token --></svg>
    342 tokens
  </span>
  <span class="meta-separator" aria-hidden="true"></span>
  <span class="meta-item">
    GPT-4o
  </span>
</div>
```

---

## React Component Implementation

```jsx
// MessageWrapper.jsx
function MessageWrapper({ 
  message, 
  showAvatar = true,
  onCopy,
  onEdit,
  onRegenerate,
  onDelete 
}) {
  const isUser = message.role === 'user';
  const isAI = message.role === 'assistant';
  
  return (
    <div 
      className={`message-wrapper ${message.role}`}
      data-message-id={message.id}
    >
      {showAvatar && (
        <div className="message-avatar" aria-hidden="true">
          {isUser ? '👤' : '✨'}
        </div>
      )}
      
      <article 
        className="message-container"
        tabIndex={0}
        aria-label={`${isUser ? 'Your' : 'AI'} message`}
      >
        <MessageHeader 
          sender={message.role}
          model={message.model}
          timestamp={message.createdAt}
        />
        
        <MessageBody content={message.content} />
        
        {isAI && (
          <MessageFooter 
            tokenCount={message.usage?.totalTokens}
            status={message.status}
          />
        )}
      </article>
      
      <MessageActions
        message={message}
        onCopy={onCopy}
        onEdit={onEdit}
        onRegenerate={onRegenerate}
        onDelete={onDelete}
      />
    </div>
  );
}

function MessageHeader({ sender, model, timestamp }) {
  return (
    <header className="message-header">
      <span className="sender-name">
        {sender === 'user' ? 'You' : 'Assistant'}
      </span>
      {model && <span className="model-badge">{model}</span>}
      <time 
        className="message-time"
        dateTime={timestamp}
        title={new Date(timestamp).toLocaleString()}
      >
        {formatRelativeTime(timestamp)}
      </time>
    </header>
  );
}

function MessageBody({ content }) {
  // Handle different content types
  if (typeof content === 'string') {
    return (
      <div className="message-body">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    );
  }
  
  // Handle parts array (OpenAI format)
  return (
    <div className="message-body">
      {content.map((part, index) => (
        <MessagePart key={index} part={part} />
      ))}
    </div>
  );
}

function MessageFooter({ tokenCount, status }) {
  return (
    <footer className="message-footer">
      {tokenCount && (
        <span className="token-count">
          {tokenCount.toLocaleString()} tokens
        </span>
      )}
      {status && (
        <span className={`message-status ${status}`}>
          {status}
        </span>
      )}
    </footer>
  );
}

function MessageActions({ message, onCopy, onEdit, onRegenerate, onDelete }) {
  const isUser = message.role === 'user';
  
  return (
    <div className="message-actions" role="group" aria-label="Message actions">
      <button onClick={() => onCopy(message)} aria-label="Copy message">
        <CopyIcon />
      </button>
      
      {isUser && (
        <button onClick={() => onEdit(message)} aria-label="Edit message">
          <EditIcon />
        </button>
      )}
      
      {!isUser && (
        <button onClick={() => onRegenerate(message)} aria-label="Regenerate">
          <RefreshIcon />
        </button>
      )}
      
      <button onClick={() => onDelete(message)} aria-label="Delete message">
        <TrashIcon />
      </button>
    </div>
  );
}
```

---

## Vanilla JavaScript Implementation

```javascript
class MessageComponent {
  constructor(message, options = {}) {
    this.message = message;
    this.options = options;
    this.element = this.render();
  }
  
  render() {
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${this.message.role}`;
    wrapper.dataset.messageId = this.message.id;
    
    wrapper.innerHTML = `
      ${this.renderAvatar()}
      <article class="message-container" tabindex="0">
        ${this.renderHeader()}
        ${this.renderBody()}
        ${this.renderFooter()}
      </article>
      ${this.renderActions()}
    `;
    
    this.attachEventListeners(wrapper);
    return wrapper;
  }
  
  renderAvatar() {
    if (!this.options.showAvatar) return '';
    const icon = this.message.role === 'user' ? '👤' : '✨';
    return `<div class="message-avatar" aria-hidden="true">${icon}</div>`;
  }
  
  renderHeader() {
    const sender = this.message.role === 'user' ? 'You' : 'Assistant';
    const model = this.message.model 
      ? `<span class="model-badge">${this.message.model}</span>` 
      : '';
    const time = this.formatTime(this.message.createdAt);
    
    return `
      <header class="message-header">
        <span class="sender-name">${sender}</span>
        ${model}
        <time class="message-time" datetime="${this.message.createdAt}">
          ${time}
        </time>
      </header>
    `;
  }
  
  renderBody() {
    return `
      <div class="message-body">
        ${this.parseContent(this.message.content)}
      </div>
    `;
  }
  
  renderFooter() {
    if (this.message.role === 'user') return '';
    
    const tokens = this.message.usage?.totalTokens;
    return `
      <footer class="message-footer">
        ${tokens ? `<span class="token-count">${tokens} tokens</span>` : ''}
      </footer>
    `;
  }
  
  renderActions() {
    const isUser = this.message.role === 'user';
    return `
      <div class="message-actions" role="group">
        <button data-action="copy" aria-label="Copy">📋</button>
        ${isUser ? '<button data-action="edit" aria-label="Edit">✏️</button>' : ''}
        ${!isUser ? '<button data-action="regenerate" aria-label="Regenerate">🔄</button>' : ''}
      </div>
    `;
  }
  
  parseContent(content) {
    // Basic markdown parsing (use a library in production)
    return content
      .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }
  
  formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: 'numeric',
      minute: '2-digit'
    });
  }
  
  attachEventListeners(wrapper) {
    wrapper.querySelector('.message-actions')?.addEventListener('click', (e) => {
      const action = e.target.closest('button')?.dataset.action;
      if (action && this.options[`on${action.charAt(0).toUpperCase() + action.slice(1)}`]) {
        this.options[`on${action.charAt(0).toUpperCase() + action.slice(1)}`](this.message);
      }
    });
  }
}

// Usage
const message = new MessageComponent({
  id: 'msg-123',
  role: 'assistant',
  content: 'Hello! How can I help you today?',
  model: 'GPT-4o',
  createdAt: new Date().toISOString(),
  usage: { totalTokens: 42 }
}, {
  showAvatar: true,
  onCopy: (msg) => navigator.clipboard.writeText(msg.content),
  onRegenerate: (msg) => console.log('Regenerate:', msg.id)
});

document.querySelector('.message-list').appendChild(message.element);
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use semantic HTML (`article`, `header`, `footer`) | Use `div` for everything |
| Keep actions accessible via keyboard | Hide actions from keyboard users |
| Use `aria-label` for icon-only buttons | Leave buttons unlabeled |
| Separate positioning from styling | Mix layout and visual CSS |
| Make containers focusable for navigation | Ignore keyboard accessibility |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Actions visible only on hover | Add keyboard focus support |
| Deeply nested DOM structure | Flatten to 3 layers max |
| Hard-coded styles inline | Use CSS classes and variables |
| No mobile action alternative | Add long-press menu |
| Missing `aria-label` on actions | Label all interactive elements |

---

## Hands-on Exercise

### Your Task

Build a message component that:
1. Uses the three-layer wrapper structure
2. Includes header, body, and footer sections
3. Has hover-revealed action buttons
4. Works with keyboard navigation

### Requirements

1. Create HTML structure with semantic elements
2. Style with CSS for user and AI variants
3. Add hover/focus state for actions
4. Include proper ARIA labels

### Expected Result

A reusable message component that displays sender info, content, metadata, and action buttons.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `article` for the message container
- Apply `tabindex="0"` to make messages focusable
- Use `:focus-within` alongside `:hover` for actions
- Data attributes help with JavaScript event handling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Message Component</title>
  <style>
    .message-list {
      max-width: 48rem;
      margin: 2rem auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: 1rem;
    }
    
    .message-wrapper {
      display: flex;
      gap: 0.75rem;
      position: relative;
    }
    
    .message-wrapper.user {
      flex-direction: row-reverse;
    }
    
    .message-avatar {
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #e5e7eb;
      font-size: 1.25rem;
      flex-shrink: 0;
    }
    
    .message-wrapper.assistant .message-avatar {
      background: linear-gradient(135deg, #8b5cf6, #3b82f6);
    }
    
    .message-container {
      max-width: 75%;
      padding: 0.75rem 1rem;
      border-radius: 1rem;
      outline: none;
    }
    
    .message-container:focus {
      outline: 2px solid #3b82f6;
      outline-offset: 2px;
    }
    
    .message-wrapper.user .message-container {
      background: #3b82f6;
      color: white;
      border-bottom-right-radius: 0.25rem;
    }
    
    .message-wrapper.assistant .message-container {
      background: #f3f4f6;
      border-bottom-left-radius: 0.25rem;
    }
    
    .message-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
      font-size: 0.875rem;
    }
    
    .sender-name { font-weight: 600; }
    
    .model-badge {
      padding: 0.125rem 0.5rem;
      background: rgba(0,0,0,0.1);
      border-radius: 1rem;
      font-size: 0.75rem;
    }
    
    .message-time {
      margin-left: auto;
      font-size: 0.75rem;
      opacity: 0.7;
    }
    
    .message-body { line-height: 1.6; }
    
    .message-footer {
      margin-top: 0.5rem;
      font-size: 0.75rem;
      opacity: 0.7;
    }
    
    .message-actions {
      position: absolute;
      top: -0.5rem;
      right: 0;
      display: flex;
      gap: 0.25rem;
      padding: 0.25rem;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.15s, visibility 0.15s;
    }
    
    .message-wrapper.user .message-actions {
      right: auto;
      left: 0;
    }
    
    .message-wrapper:hover .message-actions,
    .message-wrapper:focus-within .message-actions {
      opacity: 1;
      visibility: visible;
    }
    
    .message-actions button {
      width: 2rem;
      height: 2rem;
      border: none;
      background: none;
      border-radius: 0.25rem;
      cursor: pointer;
    }
    
    .message-actions button:hover { background: #f3f4f6; }
  </style>
</head>
<body>
  <div class="message-list">
    <div class="message-wrapper user">
      <div class="message-avatar">👤</div>
      <article class="message-container" tabindex="0" aria-label="Your message">
        <header class="message-header">
          <span class="sender-name">You</span>
          <time class="message-time">2:30 PM</time>
        </header>
        <div class="message-body">
          How do I structure a message component?
        </div>
      </article>
      <div class="message-actions" role="group" aria-label="Message actions">
        <button aria-label="Copy">📋</button>
        <button aria-label="Edit">✏️</button>
      </div>
    </div>
    
    <div class="message-wrapper assistant">
      <div class="message-avatar">✨</div>
      <article class="message-container" tabindex="0" aria-label="AI response">
        <header class="message-header">
          <span class="sender-name">Assistant</span>
          <span class="model-badge">GPT-4o</span>
          <time class="message-time">2:31 PM</time>
        </header>
        <div class="message-body">
          Use a three-layer structure: wrapper for positioning, container for styling, and sections for content organization.
        </div>
        <footer class="message-footer">
          342 tokens
        </footer>
      </article>
      <div class="message-actions" role="group" aria-label="Message actions">
        <button aria-label="Copy">📋</button>
        <button aria-label="Regenerate">🔄</button>
      </div>
    </div>
  </div>
</body>
</html>
```

</details>

---

## Summary

✅ **Three-layer structure** separates positioning, styling, and content  
✅ **Semantic HTML** with `article`, `header`, `footer` improves accessibility  
✅ **Hover-revealed actions** keep the interface clean  
✅ **Keyboard focus** ensures actions are accessible without a mouse  
✅ **Component architecture** enables reuse across different message types

---

## Further Reading

- [HTML Sectioning Elements](https://developer.mozilla.org/en-US/docs/Web/HTML/Element#content_sectioning)
- [ARIA Button Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/button/)
- [CSS :focus-within](https://developer.mozilla.org/en-US/docs/Web/CSS/:focus-within)

---

**Previous:** [Message Components Overview](./00-message-components.md)  
**Next:** [User Message Styling](./02-user-message-styling.md)

<!-- 
Sources Consulted:
- MDN HTML Sectioning: https://developer.mozilla.org/en-US/docs/Web/HTML/Element
- WAI-ARIA APG: https://www.w3.org/WAI/ARIA/apg/patterns/
-->
