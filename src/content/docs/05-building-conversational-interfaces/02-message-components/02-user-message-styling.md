---
title: "User Message Styling"
---

# User Message Styling

## Introduction

User messages deserve distinct visual treatment that immediately identifies who sent them. Effective user message styling creates visual hierarchy, reinforces the conversation flow, and helps users track their own contributions.

In this lesson, we'll build comprehensive user message styles that are recognizable, accessible, and visually polished.

### What We'll Cover

- Alignment and positioning strategies
- Bubble design and background choices
- Typography and contrast for readability
- Sent/delivered/read status indicators
- Edit history and regeneration states
- Dark mode considerations

### Prerequisites

- [Message Container Structure](./01-message-container-structure.md)
- CSS custom properties (variables)
- Basic understanding of color contrast

---

## Alignment and Positioning

### Right-Aligned Convention

User messages traditionally appear on the right, mimicking SMS and messaging apps:

```css
.message-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem;
}

.message-wrapper {
  display: flex;
  gap: 0.75rem;
  max-width: 100%;
}

.message-wrapper.user {
  flex-direction: row-reverse;  /* Push to right */
  align-self: flex-end;        /* Right align in flex parent */
}

.message-wrapper.user .message-container {
  max-width: min(75%, 32rem);  /* Limit width */
}
```

### When to Break Convention

Some interfaces left-align all messages for better readability:

| Pattern | Best For | Why |
|---------|----------|-----|
| **Right-aligned user** | Conversational chat | Familiar pattern |
| **Left-aligned all** | Technical/code-heavy | Consistent reading flow |
| **Centered narrow** | Simple Q&A | Focus on content |

```css
/* Left-aligned user variant */
.message-list.left-aligned .message-wrapper.user {
  flex-direction: row;
  align-self: flex-start;
}
```

---

## Bubble Design

### Background Color Strategy

User bubbles typically use brand colors or accent tones:

```css
:root {
  /* Primary palette */
  --user-bubble-bg: #3b82f6;           /* Blue */
  --user-bubble-text: #ffffff;
  --user-bubble-border: transparent;
  
  /* Subtle palette alternative */
  --user-bubble-bg-subtle: #dbeafe;    /* Light blue */
  --user-bubble-text-subtle: #1e40af;
  
  /* Focus/active states */
  --user-bubble-focus: #1d4ed8;
}

.message-wrapper.user .message-container {
  background: var(--user-bubble-bg);
  color: var(--user-bubble-text);
  border: 1px solid var(--user-bubble-border);
}
```

### Border Radius Patterns

Asymmetric corners indicate message direction:

```css
.message-wrapper.user .message-container {
  border-radius: 1.25rem 1.25rem 0.375rem 1.25rem;
  /* 
    Top-left: rounded
    Top-right: rounded
    Bottom-right: pointed (indicates user)
    Bottom-left: rounded
  */
}

/* More subtle approach */
.message-wrapper.user.subtle .message-container {
  border-radius: 1rem;  /* Uniform corners */
}
```

### Visual Variations

```css
/* Gradient background */
.message-wrapper.user.gradient .message-container {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
}

/* Outlined style */
.message-wrapper.user.outlined .message-container {
  background: white;
  border: 2px solid var(--user-bubble-bg);
  color: var(--user-bubble-bg);
}

/* Glass effect */
.message-wrapper.user.glass .message-container {
  background: rgba(59, 130, 246, 0.9);
  backdrop-filter: blur(10px);
}
```

---

## Typography and Contrast

### Text Styling

```css
.message-wrapper.user .message-container {
  font-size: 0.9375rem;  /* 15px - slightly smaller than base */
  line-height: 1.5;
  letter-spacing: 0.01em;
}

.message-wrapper.user .message-body {
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
}

/* Links within user messages */
.message-wrapper.user .message-body a {
  color: inherit;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.message-wrapper.user .message-body a:hover {
  text-decoration-thickness: 2px;
}
```

### Contrast Requirements

WCAG requires 4.5:1 contrast for normal text:

| Background | Text Color | Contrast Ratio | Status |
|------------|------------|----------------|--------|
| `#3b82f6` (blue) | `#ffffff` | 4.5:1 | ✅ Pass |
| `#3b82f6` | `#f0f9ff` | 4.2:1 | ⚠️ Large text only |
| `#60a5fa` (light blue) | `#ffffff` | 2.8:1 | ❌ Fail |
| `#dbeafe` (very light) | `#1e40af` | 8.1:1 | ✅ Pass |

```css
/* High contrast mode */
@media (prefers-contrast: more) {
  .message-wrapper.user .message-container {
    background: #1e40af;  /* Darker blue */
    border: 2px solid white;
  }
}
```

### Code in User Messages

When users include code, adjust styling:

```css
.message-wrapper.user .message-body code {
  background: rgba(255, 255, 255, 0.2);
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-family: 'Fira Code', monospace;
  font-size: 0.875em;
}

.message-wrapper.user .message-body pre {
  background: rgba(0, 0, 0, 0.3);
  padding: 0.75rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0;
  overflow-x: auto;
}

.message-wrapper.user .message-body pre code {
  background: none;
  padding: 0;
  color: #e5e7eb;
}
```

---

## Status Indicators

### Sent, Delivered, Read States

```css
.message-status {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.8);
}

.message-status-icon {
  width: 1rem;
  height: 1rem;
}

/* Status variants */
.message-status.pending .message-status-icon {
  animation: pulse 1.5s infinite;
}

.message-status.sent .message-status-icon {
  color: rgba(255, 255, 255, 0.7);
}

.message-status.delivered .message-status-icon {
  color: rgba(255, 255, 255, 0.9);
}

.message-status.read .message-status-icon {
  color: #34d399;  /* Green checkmarks */
}

.message-status.failed {
  color: #fca5a5;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

```html
<footer class="message-footer">
  <div class="message-status sent">
    <svg class="message-status-icon" viewBox="0 0 24 24">
      <!-- Single checkmark for sent -->
      <path d="M5 12l5 5L20 7" stroke="currentColor" fill="none"/>
    </svg>
    <span class="sr-only">Sent</span>
  </div>
</footer>

<footer class="message-footer">
  <div class="message-status read">
    <svg class="message-status-icon" viewBox="0 0 24 24">
      <!-- Double checkmark for read -->
      <path d="M2 12l5 5L18 6M8 12l5 5L24 6" stroke="currentColor" fill="none"/>
    </svg>
    <span class="sr-only">Read</span>
  </div>
</footer>
```

### Error States

```css
.message-wrapper.user.error .message-container {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
}

.message-wrapper.user.error .message-footer {
  color: #dc2626;
}

.message-retry-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: none;
  border: 1px solid currentColor;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  color: inherit;
  cursor: pointer;
}

.message-retry-btn:hover {
  background: rgba(220, 38, 38, 0.1);
}
```

```html
<div class="message-wrapper user error">
  <article class="message-container">
    <div class="message-body">
      Message that failed to send...
    </div>
    <footer class="message-footer">
      <span class="error-text">Failed to send</span>
      <button class="message-retry-btn">
        <svg width="12" height="12"><!-- retry icon --></svg>
        Retry
      </button>
    </footer>
  </article>
</div>
```

---

## Edit History

### Showing Edited Status

```css
.message-edited-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  opacity: 0.7;
}

.message-edited-badge::before {
  content: '(edited)';
}

/* Hover to show edit time */
.message-edited-badge[title] {
  cursor: help;
  text-decoration: underline dotted;
}
```

### Edit Mode UI

```css
.message-wrapper.user.editing .message-container {
  background: var(--user-bubble-bg-subtle);
  border: 2px solid var(--user-bubble-bg);
}

.message-edit-form {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.message-edit-textarea {
  width: 100%;
  min-height: 4rem;
  padding: 0.75rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  font-family: inherit;
  font-size: inherit;
  line-height: 1.5;
  resize: vertical;
}

.message-edit-textarea:focus {
  outline: none;
  border-color: var(--user-bubble-bg);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.message-edit-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
}

.edit-btn {
  padding: 0.375rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  cursor: pointer;
}

.edit-btn.cancel {
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  color: #374151;
}

.edit-btn.save {
  background: var(--user-bubble-bg);
  border: none;
  color: white;
}
```

```html
<div class="message-wrapper user editing">
  <article class="message-container">
    <form class="message-edit-form">
      <textarea class="message-edit-textarea">Original message text...</textarea>
      <div class="message-edit-actions">
        <button type="button" class="edit-btn cancel">Cancel</button>
        <button type="submit" class="edit-btn save">Save</button>
      </div>
    </form>
  </article>
</div>
```

### Version History

For advanced interfaces, show previous versions:

```html
<div class="message-wrapper user">
  <article class="message-container">
    <div class="message-body">Updated message content</div>
    <footer class="message-footer">
      <button class="message-history-toggle" aria-expanded="false">
        <span class="message-edited-badge"></span>
        View history
      </button>
    </footer>
    
    <div class="message-history" hidden>
      <div class="history-version">
        <time>2:30 PM</time>
        <p>Original message content</p>
      </div>
    </div>
  </article>
</div>
```

---

## Dark Mode

### Color Adjustments

```css
@media (prefers-color-scheme: dark) {
  :root {
    --user-bubble-bg: #2563eb;           /* Slightly adjusted blue */
    --user-bubble-text: #ffffff;
    --user-bubble-bg-subtle: #1e3a5f;
    --user-bubble-text-subtle: #93c5fd;
    --user-bubble-focus: #60a5fa;
  }
}

/* Manual dark mode class */
.dark {
  --user-bubble-bg: #2563eb;
  --user-bubble-text: #ffffff;
}
```

### Adjusting Shadows and Borders

```css
@media (prefers-color-scheme: dark) {
  .message-wrapper.user .message-container {
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  }
  
  .message-wrapper.user .message-body code {
    background: rgba(0, 0, 0, 0.3);
  }
  
  .message-wrapper.user.error .message-container {
    background: #7f1d1d;
    border-color: #dc2626;
    color: #fecaca;
  }
}
```

### Complete Dark Mode Example

```css
:root {
  --user-bg: #3b82f6;
  --user-text: #ffffff;
  --user-meta: rgba(255, 255, 255, 0.7);
  --user-code-bg: rgba(255, 255, 255, 0.2);
}

@media (prefers-color-scheme: dark) {
  :root {
    --user-bg: #1d4ed8;
    --user-text: #f0f9ff;
    --user-meta: rgba(255, 255, 255, 0.6);
    --user-code-bg: rgba(0, 0, 0, 0.3);
  }
}

.message-wrapper.user .message-container {
  background: var(--user-bg);
  color: var(--user-text);
}

.message-wrapper.user .message-footer {
  color: var(--user-meta);
}

.message-wrapper.user .message-body code {
  background: var(--user-code-bg);
}
```

---

## Complete Component

### React Implementation

```jsx
// UserMessage.jsx
function UserMessage({ message, onEdit, onRetry }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(message.content);
  
  const handleSubmitEdit = (e) => {
    e.preventDefault();
    onEdit(message.id, editValue);
    setIsEditing(false);
  };
  
  return (
    <div 
      className={`message-wrapper user ${message.status} ${isEditing ? 'editing' : ''}`}
      data-message-id={message.id}
    >
      <article 
        className="message-container"
        tabIndex={0}
        aria-label="Your message"
      >
        <header className="message-header">
          <span className="sender-name">You</span>
          <time className="message-time">
            {formatTime(message.createdAt)}
          </time>
        </header>
        
        {isEditing ? (
          <form className="message-edit-form" onSubmit={handleSubmitEdit}>
            <textarea
              className="message-edit-textarea"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              autoFocus
            />
            <div className="message-edit-actions">
              <button 
                type="button" 
                className="edit-btn cancel"
                onClick={() => setIsEditing(false)}
              >
                Cancel
              </button>
              <button type="submit" className="edit-btn save">
                Save
              </button>
            </div>
          </form>
        ) : (
          <div className="message-body">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        
        <footer className="message-footer">
          {message.editedAt && (
            <span 
              className="message-edited-badge"
              title={`Edited at ${formatTime(message.editedAt)}`}
            />
          )}
          <MessageStatus status={message.status} onRetry={() => onRetry(message)} />
        </footer>
      </article>
      
      {!isEditing && (
        <MessageActions 
          onEdit={() => setIsEditing(true)}
          onCopy={() => navigator.clipboard.writeText(message.content)}
        />
      )}
    </div>
  );
}

function MessageStatus({ status, onRetry }) {
  if (status === 'failed') {
    return (
      <div className="message-status failed">
        <span>Failed to send</span>
        <button className="message-retry-btn" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }
  
  const icons = {
    pending: '◌',
    sent: '✓',
    delivered: '✓✓',
    read: '✓✓'
  };
  
  return (
    <div className={`message-status ${status}`}>
      <span className="message-status-icon">{icons[status]}</span>
      <span className="sr-only">{status}</span>
    </div>
  );
}
```

### Complete CSS

```css
/* User Message Styles */
.message-wrapper.user {
  flex-direction: row-reverse;
  align-self: flex-end;
}

.message-wrapper.user .message-container {
  background: var(--user-bubble-bg, #3b82f6);
  color: var(--user-bubble-text, #ffffff);
  border-radius: 1.25rem 1.25rem 0.375rem 1.25rem;
  max-width: min(75%, 32rem);
}

.message-wrapper.user .message-container:focus {
  outline: 2px solid var(--user-bubble-focus, #60a5fa);
  outline-offset: 2px;
}

.message-wrapper.user .message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
  font-size: 0.875rem;
}

.message-wrapper.user .sender-name {
  font-weight: 600;
}

.message-wrapper.user .message-time {
  font-size: 0.75rem;
  opacity: 0.8;
}

.message-wrapper.user .message-body {
  font-size: 0.9375rem;
  line-height: 1.5;
  word-wrap: break-word;
}

.message-wrapper.user .message-body a {
  color: inherit;
  text-decoration: underline;
}

.message-wrapper.user .message-body code {
  background: rgba(255, 255, 255, 0.2);
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
}

.message-wrapper.user .message-footer {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  opacity: 0.8;
}

/* Status states */
.message-wrapper.user.pending .message-container {
  opacity: 0.8;
}

.message-wrapper.user.failed .message-container {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
}

/* Edit mode */
.message-wrapper.user.editing .message-container {
  background: #dbeafe;
  color: #1e40af;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .message-wrapper.user .message-container {
    background: #1d4ed8;
  }
  
  .message-wrapper.user.failed .message-container {
    background: #7f1d1d;
    border-color: #dc2626;
    color: #fecaca;
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use brand colors that meet contrast | Pick colors without checking contrast |
| Show clear status feedback | Leave users guessing if message sent |
| Support editing inline | Force full-page edit flows |
| Test in both light and dark modes | Only test one theme |
| Use distinctive border-radius | Make user/AI bubbles identical |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Low contrast text on colored bg | Use contrast checker (4.5:1 minimum) |
| No visual feedback on send | Add pending/sent/delivered states |
| Edit discards on accidental click | Confirm before losing edits |
| Same styling as AI messages | Different alignment + color |
| Ignoring failed message state | Show error + retry option |

---

## Hands-on Exercise

### Your Task

Create user message styles with:
1. Right-aligned blue bubble
2. Asymmetric border radius
3. Sent/delivered/read indicators
4. Error state with retry button
5. Working edit mode

### Requirements

1. Use CSS custom properties for colors
2. Meet 4.5:1 contrast ratio
3. Add dark mode support
4. Include keyboard focus styles

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with `flex-direction: row-reverse`
- Use `rgba()` for semi-transparent overlays
- Test contrast at webaim.org/resources/contrastchecker
- Use `@media (prefers-color-scheme: dark)` for dark mode

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```css
:root {
  --user-bg: #3b82f6;
  --user-text: #ffffff;
  --user-bg-dark: #1d4ed8;
}

@media (prefers-color-scheme: dark) {
  :root {
    --user-bg: var(--user-bg-dark);
  }
}

.message-wrapper.user {
  flex-direction: row-reverse;
  align-self: flex-end;
}

.message-wrapper.user .message-container {
  background: var(--user-bg);
  color: var(--user-text);
  border-radius: 1.25rem 1.25rem 0.375rem 1.25rem;
  padding: 0.75rem 1rem;
  max-width: 75%;
}

.message-wrapper.user .message-container:focus {
  outline: 2px solid #60a5fa;
  outline-offset: 2px;
}

.message-status {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  opacity: 0.8;
}

.message-status.read {
  color: #34d399;
}

.message-wrapper.user.failed .message-container {
  background: #fef2f2;
  color: #991b1b;
  border: 1px solid #fca5a5;
}

.message-retry-btn {
  padding: 0.25rem 0.5rem;
  background: none;
  border: 1px solid currentColor;
  border-radius: 0.25rem;
  color: inherit;
  cursor: pointer;
}
```

</details>

---

## Summary

✅ **Right alignment** distinguishes user from AI messages  
✅ **Brand colors** create recognizable user bubbles  
✅ **Contrast ratios** ensure accessibility (4.5:1 minimum)  
✅ **Status indicators** provide send/delivery feedback  
✅ **Edit mode** allows inline message correction  
✅ **Dark mode** requires adjusted colors and shadows

---

## Further Reading

- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [CSS Custom Properties](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
- [prefers-color-scheme](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme)

---

**Previous:** [Message Container Structure](./01-message-container-structure.md)  
**Next:** [AI Response Styling](./03-ai-response-styling.md)

<!-- 
Sources Consulted:
- WebAIM Contrast Checker: https://webaim.org/resources/contrastchecker/
- MDN CSS Custom Properties: https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties
- MDN prefers-color-scheme: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme
-->
