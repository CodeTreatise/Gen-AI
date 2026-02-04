---
title: "Sender Differentiation"
---

# Sender Differentiation

## Introduction

In a conversation between a user and an AI, clear visual distinction between messages is essential. Users need to instantly recognize who said what—without reading the content. Effective sender differentiation uses multiple visual cues: color, position, icons, and labels.

In this lesson, we'll explore strategies for distinguishing user and AI messages while maintaining accessibility and visual harmony.

### What We'll Cover

- Color coding strategies for user vs AI messages
- Avatar and icon placement patterns
- Left/right alignment conventions
- Name labels and role indicators
- Accessibility considerations for color differentiation

### Prerequisites

- CSS fundamentals ([Unit 1](../../../01-web-development-fundamentals/02-css-fundamentals/00-css-fundamentals.md))
- Understanding of color contrast requirements
- Basic knowledge of ARIA attributes

---

## Color Coding Strategies

Color is the most immediate visual differentiator. The key is choosing colors that contrast well with each other and their text content.

### Common Color Schemes

```css
:root {
  /* Scheme 1: Blue/Gray (ChatGPT, Claude style) */
  --user-bg: #3b82f6;    /* Blue */
  --user-text: #ffffff;
  --ai-bg: #f3f4f6;      /* Light gray */
  --ai-text: #1f2937;
  
  /* Scheme 2: Green/White (iMessage style) */
  --user-bg: #22c55e;    /* Green */
  --user-text: #ffffff;
  --ai-bg: #e5e7eb;
  --ai-text: #1f2937;
  
  /* Scheme 3: Purple/Cream (Distinctive) */
  --user-bg: #8b5cf6;    /* Purple */
  --user-text: #ffffff;
  --ai-bg: #fef3c7;      /* Cream */
  --ai-text: #1f2937;
}

.message.user {
  background: var(--user-bg);
  color: var(--user-text);
}

.message.ai {
  background: var(--ai-bg);
  color: var(--ai-text);
}
```

### Color Contrast Requirements

For accessibility, ensure sufficient contrast between text and background:

| WCAG Level | Contrast Ratio | Use Case |
|------------|----------------|----------|
| **AA** | 4.5:1 | Normal text (required) |
| **AA Large** | 3:1 | Large text (18px+ bold) |
| **AAA** | 7:1 | Enhanced accessibility |

```css
/* ✅ Good: White on blue = 8.59:1 contrast */
.message.user {
  background: #2563eb;
  color: #ffffff;
}

/* ❌ Bad: Light gray on white = 1.47:1 contrast */
.message.ai {
  background: #f9fafb;
  color: #d1d5db; /* Too low contrast! */
}
```

> **Warning:** Never rely on color alone to convey meaning. Always pair color with other visual cues (position, icons, labels) for users with color vision deficiencies.

### Dark Mode Colors

Adjust colors for dark backgrounds:

```css
@media (prefers-color-scheme: dark) {
  :root {
    --user-bg: #3b82f6;
    --user-text: #ffffff;
    --ai-bg: #374151;    /* Darker gray */
    --ai-text: #f9fafb;  /* Light text */
    --page-bg: #111827;
  }
}
```

---

## Avatar and Icon Placement

Avatars provide instant visual identification and add personality to the conversation.

### Avatar Position Patterns

```css
.message-wrapper {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
}

.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  flex-shrink: 0;
}

/* User messages: avatar on right */
.message-wrapper.user {
  flex-direction: row-reverse;
}

/* AI messages: avatar on left */
.message-wrapper.ai {
  flex-direction: row;
}
```

```html
<div class="message-wrapper user">
  <img src="/user-avatar.jpg" alt="You" class="avatar">
  <div class="message user">
    How do I implement dark mode?
  </div>
</div>

<div class="message-wrapper ai">
  <img src="/ai-avatar.svg" alt="AI Assistant" class="avatar">
  <div class="message ai">
    You can use the prefers-color-scheme media query...
  </div>
</div>
```

### Icon-Based Avatars

When user photos aren't available, use icons:

```css
.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
}

.avatar.user {
  background: var(--user-bg);
  color: var(--user-text);
}

.avatar.ai {
  background: linear-gradient(135deg, #8b5cf6, #3b82f6);
  color: white;
}
```

```html
<div class="avatar user" aria-hidden="true">👤</div>
<div class="avatar ai" aria-hidden="true">🤖</div>

<!-- Or use SVG icons -->
<div class="avatar ai" aria-hidden="true">
  <svg viewBox="0 0 24 24" width="20" height="20">
    <!-- AI icon SVG path -->
  </svg>
</div>
```

### Avatar Visibility Options

```css
/* Hide avatars for consecutive messages from same sender */
.message-wrapper.same-sender .avatar {
  visibility: hidden;
}

/* Smaller avatars for compact mode */
.compact .avatar {
  width: 1.5rem;
  height: 1.5rem;
}

/* Hide avatars entirely on mobile */
@media (max-width: 480px) {
  .avatar {
    display: none;
  }
}
```

---

## Left/Right Alignment

Position-based differentiation is intuitive and accessible:

### Standard Pattern (User Right, AI Left)

```css
.message-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.message {
  max-width: 80%;
}

.message.user {
  align-self: flex-end; /* Right side */
}

.message.ai {
  align-self: flex-start; /* Left side */
}
```

### Full-Width AI Messages

Some interfaces use full-width AI messages for better code/content display:

```css
.message.user {
  max-width: 80%;
  align-self: flex-end;
}

.message.ai {
  max-width: 100%;
  align-self: stretch;
  background: transparent;
  padding-left: 0;
  padding-right: 0;
}

/* Add subtle left border for AI messages */
.message.ai {
  border-left: 3px solid var(--ai-accent);
  padding-left: 1rem;
}
```

### Centered Layout (No Alignment)

For centered layouts, use other visual cues:

```css
.message-wrapper {
  display: flex;
  gap: 0.75rem;
  max-width: 48rem;
  margin: 0 auto;
}

/* Rely on avatars and colors, not position */
.message.user,
.message.ai {
  align-self: stretch;
}
```

---

## Name Labels and Role Indicators

Explicit labels provide clarity, especially for accessibility:

### Simple Labels

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
}

.sender-role {
  color: #6b7280;
  font-size: 0.75rem;
}
```

```html
<div class="message-wrapper ai">
  <div class="avatar ai">🤖</div>
  <div class="message-content">
    <div class="message-header">
      <span class="sender-name">Claude</span>
      <span class="sender-role">AI Assistant</span>
    </div>
    <div class="message ai">
      Here's how you can implement that feature...
    </div>
  </div>
</div>
```

### Model Indicators

Show which AI model generated the response:

```css
.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.125rem 0.5rem;
  border-radius: 1rem;
  background: #e5e7eb;
  font-size: 0.75rem;
  color: #4b5563;
}

.model-badge.premium {
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #92400e;
}
```

```html
<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge">GPT-4o</span>
</div>

<div class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge premium">o1-preview</span>
</div>
```

---

## Accessibility for Sender Differentiation

Ensure all users can identify message senders:

### Screen Reader Support

```html
<div 
  class="message-wrapper user" 
  role="article" 
  aria-label="Your message"
>
  <div class="avatar user" aria-hidden="true">👤</div>
  <div class="message user">
    How do I center a div?
  </div>
</div>

<div 
  class="message-wrapper ai" 
  role="article" 
  aria-label="AI response from Claude"
>
  <div class="avatar ai" aria-hidden="true">🤖</div>
  <div class="message ai">
    You can use Flexbox...
  </div>
</div>
```

### Visually Hidden Labels

Add labels that screen readers announce but aren't visible:

```css
.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

```html
<div class="message user">
  <span class="visually-hidden">You said:</span>
  How do I center a div?
</div>

<div class="message ai">
  <span class="visually-hidden">AI responded:</span>
  You can use Flexbox...
</div>
```

### Color Blindness Considerations

Don't rely on color alone—combine with other cues:

| Cue | Implementation |
|-----|----------------|
| **Position** | User right, AI left |
| **Shape** | Different border-radius per sender |
| **Icons** | Distinct avatars or emoji |
| **Labels** | Visible sender names |
| **Borders** | Colored left/right borders |

```css
/* Add distinct borders as secondary cue */
.message.user {
  border-right: 3px solid #3b82f6;
}

.message.ai {
  border-left: 3px solid #6b7280;
}
```

---

## Complete Sender Differentiation Example

```css
:root {
  --user-bg: #3b82f6;
  --user-text: #ffffff;
  --user-accent: #2563eb;
  --ai-bg: #f3f4f6;
  --ai-text: #1f2937;
  --ai-accent: #6b7280;
}

.message-wrapper {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  padding: 0.25rem 0;
}

.message-wrapper.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 0.875rem;
}

.avatar.user {
  background: var(--user-bg);
  color: var(--user-text);
}

.avatar.ai {
  background: linear-gradient(135deg, #8b5cf6, #3b82f6);
  color: white;
}

.message-content {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-width: 80%;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.sender-name {
  font-weight: 600;
  color: #374151;
}

.message {
  padding: 0.75rem 1rem;
  border-radius: 1.25rem;
  line-height: 1.5;
}

.message.user {
  background: var(--user-bg);
  color: var(--user-text);
  border-bottom-right-radius: 0.25rem;
}

.message.ai {
  background: var(--ai-bg);
  color: var(--ai-text);
  border-bottom-left-radius: 0.25rem;
}
```

```html
<div class="message-list" role="log" aria-live="polite" aria-label="Conversation">
  <div class="message-wrapper user" role="article" aria-label="Your message">
    <div class="avatar user" aria-hidden="true">👤</div>
    <div class="message-content">
      <div class="message-header">
        <span class="sender-name">You</span>
      </div>
      <div class="message user">
        What's the best way to learn CSS Grid?
      </div>
    </div>
  </div>
  
  <div class="message-wrapper ai" role="article" aria-label="AI response">
    <div class="avatar ai" aria-hidden="true">✨</div>
    <div class="message-content">
      <div class="message-header">
        <span class="sender-name">Assistant</span>
        <span class="model-badge">Claude 3.5</span>
      </div>
      <div class="message ai">
        I recommend starting with CSS-Tricks' Complete Guide to Grid...
      </div>
    </div>
  </div>
</div>
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use multiple cues (color + position + icons) | Rely on color alone |
| Maintain 4.5:1 text contrast ratio | Use low-contrast text colors |
| Add screen reader labels | Assume visual cues are enough |
| Keep avatars consistent across conversation | Change avatar styles mid-chat |
| Support dark mode with adjusted colors | Use colors that fail in dark mode |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Indistinguishable in grayscale | Add position/shape differentiation |
| Avatar images fail to load | Use CSS fallback or icon avatars |
| Labels hidden from screen readers | Add `aria-label` or visually-hidden text |
| Same alignment for all messages | Use flex `row-reverse` for user |
| No visual distinction without color | Add borders or background patterns |

---

## Hands-on Exercise

### Your Task

Create a message thread that:
1. Uses color, position, and avatars to differentiate senders
2. Includes accessible labels for screen readers
3. Shows model badges for AI responses
4. Works in both light and dark mode

### Requirements

1. Create HTML structure with message wrappers
2. Style user messages on right with blue background
3. Style AI messages on left with gray background
4. Add avatar icons and sender labels
5. Include proper ARIA attributes

### Expected Result

A visually distinct conversation where any user (including those with color blindness or using screen readers) can identify who said what.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `flex-direction: row-reverse` for user messages
- Add `role="article"` and `aria-label` to message wrappers
- Use `aria-hidden="true"` on decorative avatars
- Test with grayscale filter to verify non-color cues work

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sender Differentiation</title>
  <style>
    :root {
      --user-bg: #3b82f6;
      --user-text: #fff;
      --ai-bg: #f3f4f6;
      --ai-text: #1f2937;
      --page-bg: #fff;
    }
    
    @media (prefers-color-scheme: dark) {
      :root {
        --ai-bg: #374151;
        --ai-text: #f9fafb;
        --page-bg: #111827;
      }
      body { color: #f9fafb; }
    }
    
    body {
      font-family: system-ui, sans-serif;
      background: var(--page-bg);
      margin: 0;
      padding: 1rem;
    }
    
    .message-list {
      max-width: 48rem;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    
    .message-wrapper {
      display: flex;
      gap: 0.75rem;
      align-items: flex-start;
    }
    
    .message-wrapper.user {
      flex-direction: row-reverse;
    }
    
    .avatar {
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    
    .avatar.user { background: var(--user-bg); }
    .avatar.ai { background: linear-gradient(135deg, #8b5cf6, #3b82f6); }
    
    .message-content {
      max-width: 75%;
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
      background: #e5e7eb;
      border-radius: 1rem;
      font-size: 0.75rem;
    }
    
    .message {
      padding: 0.75rem 1rem;
      border-radius: 1.25rem;
      line-height: 1.5;
    }
    
    .message.user {
      background: var(--user-bg);
      color: var(--user-text);
      border-bottom-right-radius: 0.25rem;
    }
    
    .message.ai {
      background: var(--ai-bg);
      color: var(--ai-text);
      border-bottom-left-radius: 0.25rem;
    }
    
    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      margin: -1px;
      overflow: hidden;
      clip: rect(0,0,0,0);
    }
  </style>
</head>
<body>
  <div class="message-list" role="log" aria-live="polite">
    <div class="message-wrapper user" role="article" aria-label="Your message">
      <div class="avatar user" aria-hidden="true">👤</div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">You</span>
        </div>
        <div class="message user">
          <span class="visually-hidden">You said: </span>
          How do I implement dark mode in CSS?
        </div>
      </div>
    </div>
    
    <div class="message-wrapper ai" role="article" aria-label="AI response from Claude">
      <div class="avatar ai" aria-hidden="true">✨</div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Assistant</span>
          <span class="model-badge">Claude 3.5</span>
        </div>
        <div class="message ai">
          <span class="visually-hidden">AI responded: </span>
          You can use the <code>prefers-color-scheme</code> media query to detect the user's system preference...
        </div>
      </div>
    </div>
  </div>
</body>
</html>
```

</details>

---

## Summary

✅ **Color coding** provides instant visual differentiation (blue for user, gray for AI)  
✅ **Avatars** add personality and reinforce sender identity  
✅ **Position** (left/right alignment) works even without color vision  
✅ **Labels** provide explicit identification for accessibility  
✅ **Multiple cues** together ensure everyone can follow the conversation

---

## Further Reading

- [WebAIM Color Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Understanding Color Blindness](https://www.color-blindness.com/)
- [ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)

---

**Previous:** [Message Bubble Design](./02-message-bubble-design.md)  
**Next:** [Timestamp & Metadata Display](./04-timestamp-metadata-display.md)

<!-- 
Sources Consulted:
- WebAIM Contrast Checker: https://webaim.org/resources/contrastchecker/
- MDN ARIA Live Regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- Inclusive Components: https://inclusive-components.design/
-->
