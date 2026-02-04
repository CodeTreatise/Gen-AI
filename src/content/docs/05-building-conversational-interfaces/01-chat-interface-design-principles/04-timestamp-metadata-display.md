---
title: "Timestamp & Metadata Display"
---

# Timestamp & Metadata Display

## Introduction

Timestamps and metadata provide context that helps users navigate conversations. When did this message arrive? Which AI model responded? How many tokens were used? These details transform a simple chat into a professional, informative interface.

In this lesson, we'll explore patterns for displaying time, model information, token counts, and other metadata in chat interfaces.

### What We'll Cover

- Relative vs absolute timestamp formats
- Grouping messages by date/time
- Model version indicators
- Token count and cost display
- Metadata positioning strategies

### Prerequisites

- JavaScript Date handling
- CSS positioning fundamentals
- Understanding of Intl API for formatting

---

## Relative vs Absolute Timestamps

### Relative Timestamps ("2 minutes ago")

Relative times feel more natural for recent messages:

```javascript
function getRelativeTime(date) {
  const now = new Date();
  const diff = now - new Date(date);
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (seconds < 60) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  
  // Fall back to absolute for older messages
  return new Date(date).toLocaleDateString();
}
```

### Using the Intl.RelativeTimeFormat API

Modern browsers support native relative time formatting:

```javascript
const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

function getRelativeTime(date) {
  const now = new Date();
  const diff = (new Date(date) - now) / 1000; // seconds
  
  const units = [
    { unit: 'year', seconds: 31536000 },
    { unit: 'month', seconds: 2592000 },
    { unit: 'week', seconds: 604800 },
    { unit: 'day', seconds: 86400 },
    { unit: 'hour', seconds: 3600 },
    { unit: 'minute', seconds: 60 },
    { unit: 'second', seconds: 1 }
  ];
  
  for (const { unit, seconds } of units) {
    if (Math.abs(diff) >= seconds || unit === 'second') {
      const value = Math.round(diff / seconds);
      return rtf.format(value, unit);
    }
  }
}

// Usage
console.log(getRelativeTime(Date.now() - 5000));     // "5 seconds ago"
console.log(getRelativeTime(Date.now() - 3600000));  // "1 hour ago"
console.log(getRelativeTime(Date.now() - 86400000)); // "yesterday"
```

### Absolute Timestamps

For older messages or when precision matters, use absolute times:

```javascript
function formatAbsoluteTime(date) {
  const d = new Date(date);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  const timeFormat = new Intl.DateTimeFormat('en', {
    hour: 'numeric',
    minute: 'numeric'
  });
  
  const dateFormat = new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    year: d.getFullYear() !== today.getFullYear() ? 'numeric' : undefined
  });
  
  const time = timeFormat.format(d);
  
  if (d.toDateString() === today.toDateString()) {
    return `Today at ${time}`;
  }
  if (d.toDateString() === yesterday.toDateString()) {
    return `Yesterday at ${time}`;
  }
  return `${dateFormat.format(d)} at ${time}`;
}

// Usage
console.log(formatAbsoluteTime(new Date())); // "Today at 2:30 PM"
```

### Timestamp Display Patterns

```css
.timestamp {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

/* Show on hover for cleaner UI */
.message:hover .timestamp {
  opacity: 1;
}

.timestamp.hidden-default {
  opacity: 0;
  transition: opacity 0.2s ease;
}
```

```html
<div class="message ai">
  <p>Here's how you can implement that feature...</p>
  <span class="timestamp" title="January 29, 2026 at 2:30 PM">
    2 minutes ago
  </span>
</div>
```

> **Tip:** Always include the full timestamp in a `title` attribute so users can hover to see the exact time.

---

## Grouping Messages by Date

For long conversations, group messages by date:

```css
.date-divider {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1.5rem 0;
  color: #6b7280;
  font-size: 0.875rem;
}

.date-divider::before,
.date-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #e5e7eb;
}
```

```html
<div class="message-list">
  <div class="date-divider">
    <span>Yesterday</span>
  </div>
  
  <div class="message user">What's the best CSS framework?</div>
  <div class="message ai">That depends on your needs...</div>
  
  <div class="date-divider">
    <span>Today</span>
  </div>
  
  <div class="message user">Tell me more about Tailwind</div>
  <div class="message ai">Tailwind CSS is a utility-first framework...</div>
</div>
```

### Dynamic Date Grouping

```javascript
function groupMessagesByDate(messages) {
  const groups = [];
  let currentDate = null;
  
  for (const message of messages) {
    const messageDate = new Date(message.timestamp).toDateString();
    
    if (messageDate !== currentDate) {
      currentDate = messageDate;
      groups.push({
        type: 'date-divider',
        date: message.timestamp
      });
    }
    
    groups.push({
      type: 'message',
      ...message
    });
  }
  
  return groups;
}

function formatDateDivider(date) {
  const d = new Date(date);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  
  return new Intl.DateTimeFormat('en', {
    weekday: 'long',
    month: 'long',
    day: 'numeric'
  }).format(d);
}
```

---

## Model Version Indicators

Show which AI model generated each response:

### Badge Style

```css
.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.125rem 0.5rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
}

.model-badge.standard {
  background: #e5e7eb;
  color: #4b5563;
}

.model-badge.premium {
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #92400e;
}

.model-badge.reasoning {
  background: linear-gradient(135deg, #ddd6fe, #c4b5fd);
  color: #5b21b6;
}
```

```html
<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge standard">GPT-4o-mini</span>
</div>

<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge premium">GPT-4o</span>
</div>

<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge reasoning">o1-preview</span>
</div>
```

### Model Icons

```css
.model-icon {
  width: 1rem;
  height: 1rem;
  border-radius: 0.25rem;
}
```

```html
<div class="message-header">
  <img src="/icons/openai.svg" alt="OpenAI" class="model-icon">
  <span class="model-badge">GPT-4o</span>
</div>

<div class="message-header">
  <img src="/icons/anthropic.svg" alt="Anthropic" class="model-icon">
  <span class="model-badge">Claude 3.5 Sonnet</span>
</div>
```

---

## Token Count and Cost Display

For API-based applications, showing token usage helps users understand costs:

### Token Count Badge

```css
.token-count {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.token-count .icon {
  width: 0.875rem;
  height: 0.875rem;
}

.token-count.warning {
  color: #d97706;
}

.token-count.danger {
  color: #dc2626;
}
```

```html
<div class="message-footer">
  <span class="timestamp">2m ago</span>
  <span class="token-count">
    <svg class="icon" viewBox="0 0 16 16"><!-- token icon --></svg>
    342 tokens
  </span>
</div>

<!-- With cost -->
<div class="message-footer">
  <span class="timestamp">5m ago</span>
  <span class="token-count">
    1,247 tokens · $0.003
  </span>
</div>
```

### Conversation Token Summary

```css
.token-summary {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 1rem;
  background: #f9fafb;
  border-radius: 0.5rem;
  font-size: 0.875rem;
}

.token-bar {
  flex: 1;
  height: 0.5rem;
  background: #e5e7eb;
  border-radius: 0.25rem;
  overflow: hidden;
}

.token-bar-fill {
  height: 100%;
  background: #3b82f6;
  transition: width 0.3s ease;
}

.token-bar-fill.warning {
  background: #f59e0b;
}

.token-bar-fill.danger {
  background: #ef4444;
}
```

```html
<div class="token-summary">
  <span>Context: 12,450 / 128,000 tokens</span>
  <div class="token-bar">
    <div class="token-bar-fill" style="width: 9.7%"></div>
  </div>
  <span>9.7%</span>
</div>
```

### Real-Time Token Tracking

```javascript
function TokenDisplay({ usage }) {
  const percentage = (usage.total / usage.limit) * 100;
  const status = percentage > 90 ? 'danger' : percentage > 70 ? 'warning' : '';
  
  return (
    <div className="token-summary">
      <span>
        {usage.total.toLocaleString()} / {usage.limit.toLocaleString()} tokens
      </span>
      <div className="token-bar">
        <div 
          className={`token-bar-fill ${status}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className={`token-count ${status}`}>
        {percentage.toFixed(1)}%
      </span>
    </div>
  );
}
```

---

## Metadata Positioning Strategies

### Inline Metadata

Display metadata on the same line as the message:

```css
.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}

.message-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-left: auto;
  font-size: 0.75rem;
  color: #6b7280;
}
```

```html
<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge">GPT-4o</span>
  <div class="message-meta">
    <span class="token-count">256 tokens</span>
    <span class="timestamp">2m ago</span>
  </div>
</div>
```

### Footer Metadata

Place metadata below the message content:

```css
.message-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(0, 0, 0, 0.05);
  font-size: 0.75rem;
  color: #6b7280;
}
```

```html
<div class="message ai">
  <p>Here's the code you requested...</p>
  <div class="message-footer">
    <span class="timestamp">Today at 2:30 PM</span>
    <div class="meta-group">
      <span class="model-badge">Claude 3.5</span>
      <span class="token-count">1,024 tokens</span>
    </div>
  </div>
</div>
```

### Hover-Revealed Metadata

Show detailed metadata on hover:

```css
.message {
  position: relative;
}

.hover-meta {
  position: absolute;
  top: 0;
  right: 0;
  transform: translateY(-100%);
  padding: 0.25rem 0.5rem;
  background: #1f2937;
  color: white;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease;
}

.message:hover .hover-meta {
  opacity: 1;
}
```

```html
<div class="message ai">
  <p>Here's the response...</p>
  <div class="hover-meta">
    GPT-4o · 342 tokens · $0.001 · Jan 29, 2:30 PM
  </div>
</div>
```

---

## Complete Metadata Component

```css
.message-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.sender-name {
  font-weight: 600;
}

.model-badge {
  padding: 0.125rem 0.5rem;
  background: #e5e7eb;
  border-radius: 1rem;
  font-size: 0.75rem;
}

.message-body {
  padding: 0.75rem 1rem;
  background: #f3f4f6;
  border-radius: 1rem;
  line-height: 1.6;
}

.message-footer {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding-left: 0.5rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.message-footer .separator {
  width: 3px;
  height: 3px;
  background: currentColor;
  border-radius: 50%;
}
```

```html
<div class="message-wrapper ai">
  <div class="message-header">
    <span class="sender-name">Assistant</span>
    <span class="model-badge">Claude 3.5 Sonnet</span>
  </div>
  <div class="message-body">
    Here's how you can implement lazy loading for images...
  </div>
  <div class="message-footer">
    <span class="timestamp" title="January 29, 2026 at 2:30:45 PM">
      2 minutes ago
    </span>
    <span class="separator" aria-hidden="true"></span>
    <span class="token-count">486 tokens</span>
    <span class="separator" aria-hidden="true"></span>
    <span class="cost">$0.002</span>
  </div>
</div>
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use relative time for recent messages | Show only absolute timestamps |
| Provide full timestamp on hover | Hide exact time completely |
| Group messages by date for long chats | Show every message timestamp |
| Indicate model version for AI responses | Assume users know the model |
| Show token usage for API transparency | Hide usage from users |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Timestamps don't update | Use `setInterval` or re-render |
| Token counts confuse users | Add tooltip explaining tokens |
| Too much metadata clutter | Use hover-reveal patterns |
| Timezone mismatches | Use `Intl` API for local formatting |
| Outdated relative times | Update every 30-60 seconds |

---

## Hands-on Exercise

### Your Task

Build a message display with:
1. Relative timestamps that update dynamically
2. Date dividers for message grouping
3. Model badges with different tiers
4. Token count display with cost

### Requirements

1. Create a relative time formatter function
2. Add date dividers between different days
3. Style model badges for standard/premium tiers
4. Show token count and estimated cost

### Expected Result

A professional chat interface with rich metadata that provides context without overwhelming the conversation.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Intl.RelativeTimeFormat` for natural language times
- Calculate cost using typical API pricing (~$0.002/1K tokens)
- Update timestamps every 60 seconds with `setInterval`
- Use CSS `::before` and `::after` for date divider lines

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Message Metadata</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 48rem;
      margin: 0 auto;
      padding: 1rem;
      background: #f9fafb;
    }
    
    .message-list {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    
    .date-divider {
      display: flex;
      align-items: center;
      gap: 1rem;
      color: #6b7280;
      font-size: 0.875rem;
      margin: 0.5rem 0;
    }
    
    .date-divider::before,
    .date-divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: #e5e7eb;
    }
    
    .message-wrapper {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    
    .message-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .sender-name {
      font-weight: 600;
      font-size: 0.875rem;
    }
    
    .model-badge {
      padding: 0.125rem 0.5rem;
      border-radius: 1rem;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .model-badge.standard {
      background: #e5e7eb;
      color: #4b5563;
    }
    
    .model-badge.premium {
      background: linear-gradient(135deg, #fef3c7, #fde68a);
      color: #92400e;
    }
    
    .message-body {
      padding: 0.75rem 1rem;
      background: white;
      border-radius: 1rem;
      line-height: 1.6;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .message-footer {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding-left: 0.5rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .separator {
      width: 3px;
      height: 3px;
      background: currentColor;
      border-radius: 50%;
    }
  </style>
</head>
<body>
  <div class="message-list" id="messages"></div>
  
  <script>
    const messages = [
      { id: 1, sender: 'user', text: 'How do I optimize images?', timestamp: Date.now() - 86400000 - 3600000 },
      { id: 2, sender: 'ai', text: 'You can use lazy loading...', model: 'GPT-4o-mini', tokens: 156, timestamp: Date.now() - 86400000 - 3500000 },
      { id: 3, sender: 'user', text: 'What about WebP format?', timestamp: Date.now() - 300000 },
      { id: 4, sender: 'ai', text: 'WebP provides superior compression...', model: 'GPT-4o', tokens: 342, timestamp: Date.now() - 240000 }
    ];
    
    const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
    
    function getRelativeTime(timestamp) {
      const diff = (timestamp - Date.now()) / 1000;
      const units = [
        { unit: 'day', seconds: 86400 },
        { unit: 'hour', seconds: 3600 },
        { unit: 'minute', seconds: 60 },
        { unit: 'second', seconds: 1 }
      ];
      for (const { unit, seconds } of units) {
        if (Math.abs(diff) >= seconds || unit === 'second') {
          return rtf.format(Math.round(diff / seconds), unit);
        }
      }
    }
    
    function getDateLabel(timestamp) {
      const d = new Date(timestamp);
      const today = new Date();
      if (d.toDateString() === today.toDateString()) return 'Today';
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
      return d.toLocaleDateString('en', { weekday: 'long', month: 'long', day: 'numeric' });
    }
    
    function calculateCost(tokens) {
      return (tokens / 1000 * 0.002).toFixed(4);
    }
    
    function render() {
      const container = document.getElementById('messages');
      let html = '';
      let lastDate = null;
      
      for (const msg of messages) {
        const dateStr = new Date(msg.timestamp).toDateString();
        if (dateStr !== lastDate) {
          html += `<div class="date-divider"><span>${getDateLabel(msg.timestamp)}</span></div>`;
          lastDate = dateStr;
        }
        
        if (msg.sender === 'ai') {
          const tier = msg.model.includes('mini') ? 'standard' : 'premium';
          html += `
            <div class="message-wrapper">
              <div class="message-header">
                <span class="sender-name">Assistant</span>
                <span class="model-badge ${tier}">${msg.model}</span>
              </div>
              <div class="message-body">${msg.text}</div>
              <div class="message-footer">
                <span class="timestamp">${getRelativeTime(msg.timestamp)}</span>
                <span class="separator"></span>
                <span>${msg.tokens} tokens</span>
                <span class="separator"></span>
                <span>$${calculateCost(msg.tokens)}</span>
              </div>
            </div>
          `;
        } else {
          html += `
            <div class="message-wrapper" style="align-items: flex-end;">
              <div class="message-body" style="background: #3b82f6; color: white;">
                ${msg.text}
              </div>
              <div class="message-footer">${getRelativeTime(msg.timestamp)}</div>
            </div>
          `;
        }
      }
      
      container.innerHTML = html;
    }
    
    render();
    setInterval(render, 60000); // Update every minute
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ **Relative timestamps** ("2 minutes ago") feel natural for recent messages  
✅ **Date dividers** help users navigate long conversations  
✅ **Model badges** clarify which AI generated each response  
✅ **Token counts** provide transparency for API-based apps  
✅ **Strategic positioning** keeps metadata visible without cluttering

---

## Further Reading

- [Intl.RelativeTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat)
- [Intl.DateTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat)
- [OpenAI Token Pricing](https://openai.com/pricing)

---

**Previous:** [Sender Differentiation](./03-sender-differentiation.md)  
**Next:** [Accessibility Considerations](./05-accessibility-considerations.md)

<!-- 
Sources Consulted:
- MDN Intl.RelativeTimeFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat
- MDN Intl.DateTimeFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat
-->
