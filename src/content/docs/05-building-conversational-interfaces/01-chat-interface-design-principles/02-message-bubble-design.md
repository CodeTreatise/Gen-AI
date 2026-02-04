---
title: "Message Bubble Design"
---

# Message Bubble Design

## Introduction

Message bubbles are the core visual element of any chat interface. Their design affects readability, visual hierarchy, and the overall feel of the conversation. A well-designed bubble makes messages easy to scan, distinguishes between participants, and provides subtle visual cues about message state.

In this lesson, we'll explore the principles and CSS techniques for creating professional message bubbles.

### What We'll Cover

- Bubble shape and border radius conventions
- Padding and internal spacing for readability
- Maximum width for optimal line length
- Shadow and depth effects for visual hierarchy
- CSS-only speech bubble tails

### Prerequisites

- CSS fundamentals ([Unit 1](../../../01-web-development-fundamentals/02-css-fundamentals/00-css-fundamentals.md))
- Understanding of the box model
- Basic CSS pseudo-elements

---

## Bubble Shape and Border Radius

The border radius of message bubbles creates distinct visual personalities:

### Rounded Rectangles (Modern Standard)

Most chat applications use generously rounded corners:

```css
.message {
  border-radius: 1.25rem; /* 20px - Fully rounded feel */
  padding: 0.75rem 1rem;
}

/* Reduce radius on the "tail" corner */
.message.user {
  border-bottom-right-radius: 0.25rem;
}

.message.ai {
  border-bottom-left-radius: 0.25rem;
}
```

### Border Radius Conventions

| Style | Radius | Use Case |
|-------|--------|----------|
| **Pill-shaped** | `1.5rem+` | Casual, friendly apps |
| **Rounded** | `0.75rem - 1.25rem` | Professional, modern |
| **Subtle** | `0.25rem - 0.5rem` | Formal, enterprise |
| **Sharp** | `0` | Technical, minimal |

```css
/* Pill-shaped (iMessage style) */
.bubble-pill {
  border-radius: 1.5rem;
}

/* Rounded (ChatGPT style) */
.bubble-rounded {
  border-radius: 1rem;
}

/* Subtle (Slack style) */
.bubble-subtle {
  border-radius: 0.375rem;
}

/* Sharp (Terminal style) */
.bubble-sharp {
  border-radius: 0;
}
```

> **🤖 AI Context:** AI chat interfaces often use slightly less rounded corners than messaging apps to convey a more "professional" or "intelligent" feel.

---

## Padding and Spacing

Internal padding affects readability and visual density:

### Internal Padding

```css
.message {
  /* Horizontal padding slightly larger than vertical */
  padding: 0.75rem 1rem;
}

/* For messages with code blocks or complex content */
.message.has-code {
  padding: 1rem;
}

/* Compact mode for dense conversations */
.message.compact {
  padding: 0.5rem 0.75rem;
}
```

### Message Spacing

The space between messages creates visual rhythm:

```css
.message-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem; /* Space between all messages */
}

/* Group consecutive messages from same sender */
.message + .message.same-sender {
  margin-top: -0.25rem; /* Tighter spacing */
}

/* Add extra space before new sender */
.message.new-sender {
  margin-top: 1rem;
}
```

### Spacing Guidelines

| Element | Spacing | Reason |
|---------|---------|--------|
| Between messages | `0.5rem` | Clear separation |
| Same sender group | `0.25rem` | Visual grouping |
| New sender | `1rem+` | Topic/speaker change |
| Internal padding | `0.75rem 1rem` | Readability |

---

## Maximum Width for Readability

Wide messages are hard to read. Constrain bubble width:

```css
.message {
  max-width: 80%;
  max-width: min(80%, 32rem); /* Cap at ~512px */
}

/* On mobile, allow slightly wider */
@media (max-width: 640px) {
  .message {
    max-width: 90%;
  }
}

/* Code blocks may need more width */
.message.has-code {
  max-width: min(90%, 48rem);
}
```

### Width Recommendations

| Device | Max Width | Line Length |
|--------|-----------|-------------|
| Mobile | `85-90%` | 40-50 chars |
| Tablet | `75-80%` | 50-65 chars |
| Desktop | `60-70%` | 60-75 chars |

> **Note:** The ideal line length for reading is 50-75 characters. Use `ch` units if you want precise character-based widths: `max-width: 65ch`.

---

## Shadow and Depth Effects

Subtle shadows add depth and help bubbles "float" above the background:

### Subtle Elevation

```css
.message {
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

/* Slightly more depth on hover/focus */
.message:focus-within {
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
```

### Layered Shadows (More Realistic)

```css
.message {
  box-shadow: 
    0 1px 2px rgba(0, 0, 0, 0.04),
    0 2px 4px rgba(0, 0, 0, 0.04),
    0 4px 8px rgba(0, 0, 0, 0.04);
}
```

### Shadow Presets

```css
:root {
  /* Design system shadow tokens */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 2px 4px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 4px 8px rgba(0, 0, 0, 0.15);
  
  /* Colored shadows for depth */
  --shadow-user: 0 2px 8px rgba(59, 130, 246, 0.2);
  --shadow-ai: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.message.user {
  box-shadow: var(--shadow-user);
}

.message.ai {
  box-shadow: var(--shadow-ai);
}
```

> **Warning:** Heavy shadows can feel dated. Modern interfaces prefer very subtle or no shadows, relying on background color contrast instead.

---

## CSS Speech Bubble Tails

For a classic chat look, add directional tails using CSS pseudo-elements:

### Triangle Tail

```css
.message {
  position: relative;
}

.message.user::after {
  content: '';
  position: absolute;
  bottom: 0;
  right: -8px;
  width: 0;
  height: 0;
  border: 8px solid transparent;
  border-left-color: #3b82f6; /* Match bubble color */
  border-bottom: 0;
  border-right: 0;
}

.message.ai::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: -8px;
  width: 0;
  height: 0;
  border: 8px solid transparent;
  border-right-color: #f3f4f6;
  border-bottom: 0;
  border-left: 0;
}
```

### Curved Tail (More Modern)

```css
.message.user {
  position: relative;
}

.message.user::before {
  content: '';
  position: absolute;
  bottom: 0;
  right: -7px;
  width: 20px;
  height: 20px;
  background: #3b82f6;
  border-bottom-left-radius: 15px;
}

.message.user::after {
  content: '';
  position: absolute;
  bottom: 0;
  right: -10px;
  width: 10px;
  height: 20px;
  background: #fff; /* Match page background */
  border-bottom-left-radius: 10px;
}
```

> **Tip:** Bubble tails are optional in modern chat UIs. Many apps (ChatGPT, Claude) skip them entirely, using alignment and color to distinguish senders.

---

## Complete Message Bubble Component

Here's a complete, production-ready message bubble:

```css
:root {
  --color-user-bg: #3b82f6;
  --color-user-text: #ffffff;
  --color-ai-bg: #f3f4f6;
  --color-ai-text: #1f2937;
  --color-border: #e5e7eb;
  --radius-bubble: 1.25rem;
  --radius-tail: 0.25rem;
}

.message {
  max-width: min(80%, 32rem);
  padding: 0.75rem 1rem;
  border-radius: var(--radius-bubble);
  line-height: 1.5;
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.message.user {
  background: var(--color-user-bg);
  color: var(--color-user-text);
  margin-left: auto;
  border-bottom-right-radius: var(--radius-tail);
}

.message.ai {
  background: var(--color-ai-bg);
  color: var(--color-ai-text);
  margin-right: auto;
  border-bottom-left-radius: var(--radius-tail);
}

/* Handle long words and URLs */
.message {
  hyphens: auto;
  -webkit-hyphens: auto;
}

/* Ensure code doesn't break layout */
.message code {
  word-break: break-all;
}

.message pre {
  overflow-x: auto;
  max-width: 100%;
}
```

```html
<div class="message-list" role="log" aria-live="polite">
  <div class="message user">
    How do I center a div in CSS?
  </div>
  <div class="message ai">
    You can center a div using Flexbox:
    
    <pre><code>.parent {
  display: flex;
  justify-content: center;
  align-items: center;
}</code></pre>
  </div>
</div>
```

---

## Dark Mode Support

Support both light and dark themes with CSS custom properties:

```css
:root {
  --color-user-bg: #3b82f6;
  --color-user-text: #ffffff;
  --color-ai-bg: #f3f4f6;
  --color-ai-text: #1f2937;
  --color-page-bg: #ffffff;
}

@media (prefers-color-scheme: dark) {
  :root {
    --color-user-bg: #2563eb;
    --color-user-text: #ffffff;
    --color-ai-bg: #374151;
    --color-ai-text: #f9fafb;
    --color-page-bg: #111827;
  }
}

/* Or use a class-based toggle */
.dark {
  --color-user-bg: #2563eb;
  --color-user-text: #ffffff;
  --color-ai-bg: #374151;
  --color-ai-text: #f9fafb;
  --color-page-bg: #111827;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use consistent border-radius across bubbles | Mix different radius values |
| Constrain max-width for readability | Let bubbles span full width |
| Use subtle shadows (if any) | Apply heavy drop shadows |
| Support dark mode with CSS variables | Hardcode colors |
| Handle long words with `overflow-wrap` | Let long URLs break layout |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Long URLs overflow the bubble | Use `overflow-wrap: break-word` |
| Code blocks break the layout | Add `overflow-x: auto` to `pre` |
| Bubbles too wide on desktop | Use `max-width: min(80%, 32rem)` |
| No visual difference for senders | Use distinct colors and alignment |
| Shadows too heavy for modern UI | Use `rgba` with low opacity |

---

## Hands-on Exercise

### Your Task

Create a message bubble component that:
1. Supports user and AI message styles
2. Has proper spacing and max-width
3. Includes dark mode support
4. Handles code blocks gracefully

### Requirements

1. Define CSS custom properties for theming
2. Create `.message`, `.message.user`, and `.message.ai` styles
3. Add dark mode using `prefers-color-scheme`
4. Include styles for code blocks inside messages

### Expected Result

A visually polished message bubble that works in both light and dark mode, with proper text wrapping and code block handling.

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with CSS custom properties in `:root`
- Use `@media (prefers-color-scheme: dark)` for auto dark mode
- Remember `overflow-wrap: break-word` for long content
- Add `overflow-x: auto` to `pre` elements

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```css
:root {
  --bubble-radius: 1.25rem;
  --bubble-tail-radius: 0.25rem;
  --bubble-padding: 0.75rem 1rem;
  --bubble-max-width: min(80%, 32rem);
  
  --color-user-bg: #3b82f6;
  --color-user-text: #ffffff;
  --color-ai-bg: #f3f4f6;
  --color-ai-text: #1f2937;
  --color-code-bg: #1f2937;
  --color-code-text: #e5e7eb;
}

@media (prefers-color-scheme: dark) {
  :root {
    --color-user-bg: #2563eb;
    --color-ai-bg: #374151;
    --color-ai-text: #f9fafb;
    --color-code-bg: #111827;
    --color-code-text: #e5e7eb;
  }
}

.message-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
}

.message {
  max-width: var(--bubble-max-width);
  padding: var(--bubble-padding);
  border-radius: var(--bubble-radius);
  line-height: 1.6;
  overflow-wrap: break-word;
  word-wrap: break-word;
  hyphens: auto;
}

.message.user {
  background: var(--color-user-bg);
  color: var(--color-user-text);
  margin-left: auto;
  border-bottom-right-radius: var(--bubble-tail-radius);
}

.message.ai {
  background: var(--color-ai-bg);
  color: var(--color-ai-text);
  margin-right: auto;
  border-bottom-left-radius: var(--bubble-tail-radius);
}

/* Code blocks */
.message code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.875em;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  background: rgba(0, 0, 0, 0.1);
}

.message.user code {
  background: rgba(255, 255, 255, 0.2);
}

.message pre {
  margin: 0.75rem 0;
  padding: 1rem;
  border-radius: 0.5rem;
  background: var(--color-code-bg);
  color: var(--color-code-text);
  overflow-x: auto;
}

.message pre code {
  padding: 0;
  background: none;
}
```

</details>

---

## Summary

✅ **Border radius** creates personality—rounded for friendly, sharp for technical  
✅ **Padding** of `0.75rem 1rem` provides comfortable readability  
✅ **Max-width** of 60-80% keeps line length optimal  
✅ **Shadows** should be subtle or absent in modern interfaces  
✅ **CSS variables** enable easy theming and dark mode support

---

## Further Reading

- [CSS Box Shadow Generator](https://cssgenerator.org/box-shadow-css-generator.html)
- [CSS Custom Properties Guide](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
- [prefers-color-scheme](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme)

---

**Previous:** [Conversation Layout Patterns](./01-conversation-layout-patterns.md)  
**Next:** [Sender Differentiation](./03-sender-differentiation.md)

<!-- 
Sources Consulted:
- MDN CSS border-radius: https://developer.mozilla.org/en-US/docs/Web/CSS/border-radius
- MDN prefers-color-scheme: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme
- CSS-Tricks Box Shadow: https://css-tricks.com/almanac/properties/b/box-shadow/
-->
