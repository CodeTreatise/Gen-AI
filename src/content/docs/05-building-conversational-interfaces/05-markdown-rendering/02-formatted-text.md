---
title: "Formatted Text Rendering"
---

# Formatted Text Rendering

## Introduction

Bold, italic, lists, and blockquotes—these core formatting elements make AI responses scannable and organized. Rendering them consistently across browsers while maintaining streaming compatibility requires attention to both styling and structure.

In this lesson, we'll implement beautiful, accessible formatted text rendering.

### What We'll Cover

- Inline formatting (bold, italic, strikethrough)
- Ordered and unordered lists
- Nested list handling
- Blockquotes styling
- Combined formatting patterns

### Prerequisites

- [Parsing Markdown](./01-parsing-markdown.md)
- CSS fundamentals
- React component basics

---

## Inline Formatting

### Bold and Italic

```jsx
import ReactMarkdown from 'react-markdown';

function FormattedText({ content }) {
  return (
    <ReactMarkdown
      components={{
        strong: ({ children }) => (
          <strong className="md-bold">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="md-italic">{children}</em>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

```css
.md-bold {
  font-weight: 600;
  color: var(--text-emphasis, #1a1a2e);
}

.md-italic {
  font-style: italic;
  color: var(--text-secondary, #4a4a6a);
}

/* Bold italic combination */
.md-bold .md-italic,
.md-italic .md-bold {
  font-weight: 600;
  font-style: italic;
}
```

### Strikethrough

```jsx
// Requires remark-gfm plugin for ~~strikethrough~~
import remarkGfm from 'remark-gfm';

function GfmMarkdown({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        del: ({ children }) => (
          <del className="md-strikethrough">{children}</del>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

```css
.md-strikethrough {
  text-decoration: line-through;
  color: var(--text-muted, #6c757d);
  opacity: 0.7;
}
```

### Inline Code

```jsx
components={{
  code: ({ inline, children }) => {
    if (inline) {
      return <code className="md-inline-code">{children}</code>;
    }
    // Block code handled separately
    return <code>{children}</code>;
  }
}}
```

```css
.md-inline-code {
  padding: 2px 6px;
  background: var(--code-bg, #f4f4f5);
  border-radius: 4px;
  font-family: 'Fira Code', 'Monaco', monospace;
  font-size: 0.9em;
  color: var(--code-text, #e11d48);
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .md-inline-code {
    background: var(--code-bg-dark, #27272a);
    color: var(--code-text-dark, #fb7185);
  }
}
```

---

## Unordered Lists

### Basic Styling

```jsx
components={{
  ul: ({ children }) => (
    <ul className="md-list md-list-unordered">{children}</ul>
  ),
  li: ({ children, ordered }) => (
    <li className={`md-list-item ${ordered ? 'ordered' : 'unordered'}`}>
      {children}
    </li>
  )
}}
```

```css
.md-list {
  margin: 16px 0;
  padding-left: 24px;
}

.md-list-unordered {
  list-style-type: disc;
}

.md-list-item {
  margin: 8px 0;
  line-height: 1.6;
}

/* Custom bullet points */
.md-list-unordered > .md-list-item::marker {
  color: var(--primary-color, #3b82f6);
}
```

### Custom Bullet Icons

```css
.md-list-unordered {
  list-style: none;
  padding-left: 20px;
}

.md-list-unordered > .md-list-item {
  position: relative;
  padding-left: 20px;
}

.md-list-unordered > .md-list-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 10px;
  width: 6px;
  height: 6px;
  background: var(--bullet-color, #3b82f6);
  border-radius: 50%;
}
```

---

## Ordered Lists

### Number Styling

```jsx
components={{
  ol: ({ children, start }) => (
    <ol 
      className="md-list md-list-ordered"
      start={start}
    >
      {children}
    </ol>
  )
}}
```

```css
.md-list-ordered {
  list-style-type: decimal;
  padding-left: 28px;
}

.md-list-ordered > .md-list-item::marker {
  color: var(--primary-color, #3b82f6);
  font-weight: 600;
}

/* Nested ordered lists */
.md-list-ordered .md-list-ordered {
  list-style-type: lower-alpha;
}

.md-list-ordered .md-list-ordered .md-list-ordered {
  list-style-type: lower-roman;
}
```

### Step-Style Lists

```css
.md-list-ordered.steps {
  list-style: none;
  counter-reset: step-counter;
  padding-left: 0;
}

.md-list-ordered.steps > .md-list-item {
  counter-increment: step-counter;
  position: relative;
  padding-left: 48px;
  margin-bottom: 16px;
}

.md-list-ordered.steps > .md-list-item::before {
  content: counter(step-counter);
  position: absolute;
  left: 0;
  top: 0;
  width: 32px;
  height: 32px;
  background: var(--primary-color, #3b82f6);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
}
```

---

## Nested Lists

### Handling Nesting

```jsx
function NestedListItem({ children, depth = 0 }) {
  return (
    <li 
      className="md-list-item"
      style={{ '--nesting-depth': depth }}
    >
      {children}
    </li>
  );
}
```

```css
.md-list .md-list {
  margin: 8px 0;
}

/* Different bullets for nesting levels */
.md-list-unordered > .md-list-item::before {
  background: var(--bullet-color, #3b82f6);
}

.md-list-unordered .md-list-unordered > .md-list-item::before {
  background: transparent;
  border: 2px solid var(--bullet-color, #3b82f6);
}

.md-list-unordered .md-list-unordered .md-list-unordered > .md-list-item::before {
  background: var(--bullet-color-muted, #93c5fd);
  width: 4px;
  height: 4px;
}
```

### Mixed List Types

```css
/* Unordered inside ordered */
.md-list-ordered .md-list-unordered {
  list-style-type: disc;
  padding-left: 20px;
}

/* Ordered inside unordered */
.md-list-unordered .md-list-ordered {
  list-style-type: decimal;
  padding-left: 24px;
}
```

---

## Blockquotes

### Basic Blockquote

```jsx
components={{
  blockquote: ({ children }) => (
    <blockquote className="md-blockquote">
      {children}
    </blockquote>
  )
}}
```

```css
.md-blockquote {
  margin: 16px 0;
  padding: 16px 20px;
  border-left: 4px solid var(--quote-border, #3b82f6);
  background: var(--quote-bg, #f8fafc);
  border-radius: 0 8px 8px 0;
}

.md-blockquote p {
  margin: 0;
  color: var(--text-secondary, #475569);
  font-style: italic;
}

.md-blockquote p + p {
  margin-top: 12px;
}
```

### Callout-Style Blockquotes

```jsx
function CalloutBlockquote({ children }) {
  // Detect callout type from content
  const text = React.Children.toArray(children)
    .map(child => child?.props?.children)
    .join('');
  
  let type = 'default';
  if (text.startsWith('Note:') || text.startsWith('💡')) type = 'note';
  if (text.startsWith('Warning:') || text.startsWith('⚠️')) type = 'warning';
  if (text.startsWith('Important:') || text.startsWith('🔑')) type = 'important';
  
  return (
    <blockquote className={`md-blockquote md-callout md-callout-${type}`}>
      {children}
    </blockquote>
  );
}
```

```css
.md-callout-note {
  border-color: var(--info-color, #0ea5e9);
  background: var(--info-bg, #f0f9ff);
}

.md-callout-warning {
  border-color: var(--warning-color, #f59e0b);
  background: var(--warning-bg, #fffbeb);
}

.md-callout-important {
  border-color: var(--error-color, #ef4444);
  background: var(--error-bg, #fef2f2);
}
```

### Nested Blockquotes

```css
.md-blockquote .md-blockquote {
  margin: 12px 0;
  border-left-width: 3px;
  background: rgba(0, 0, 0, 0.03);
}

.md-blockquote .md-blockquote .md-blockquote {
  border-left-width: 2px;
  background: rgba(0, 0, 0, 0.05);
}
```

---

## Combined Formatting

### Complete Markdown Component

```jsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function FullyFormattedMarkdown({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Inline formatting
        strong: ({ children }) => (
          <strong className="md-bold">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="md-italic">{children}</em>
        ),
        del: ({ children }) => (
          <del className="md-strikethrough">{children}</del>
        ),
        code: ({ inline, children }) => (
          inline 
            ? <code className="md-inline-code">{children}</code>
            : <code>{children}</code>
        ),
        
        // Lists
        ul: ({ children }) => (
          <ul className="md-list md-list-unordered">{children}</ul>
        ),
        ol: ({ children, start }) => (
          <ol className="md-list md-list-ordered" start={start}>
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="md-list-item">{children}</li>
        ),
        
        // Blockquotes
        blockquote: ({ children }) => (
          <CalloutBlockquote>{children}</CalloutBlockquote>
        ),
        
        // Paragraphs
        p: ({ children }) => (
          <p className="md-paragraph">{children}</p>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

### Complete Stylesheet

```css
/* Base typography */
.md-paragraph {
  margin: 16px 0;
  line-height: 1.7;
  color: var(--text-primary, #1f2937);
}

.md-paragraph:first-child {
  margin-top: 0;
}

.md-paragraph:last-child {
  margin-bottom: 0;
}

/* Inline formatting */
.md-bold {
  font-weight: 600;
}

.md-italic {
  font-style: italic;
}

.md-strikethrough {
  text-decoration: line-through;
  opacity: 0.7;
}

.md-inline-code {
  padding: 2px 6px;
  background: var(--code-bg, #f4f4f5);
  border-radius: 4px;
  font-family: 'Fira Code', monospace;
  font-size: 0.9em;
}

/* Lists */
.md-list {
  margin: 16px 0;
  padding-left: 24px;
}

.md-list-item {
  margin: 8px 0;
  line-height: 1.6;
}

/* Blockquotes */
.md-blockquote {
  margin: 16px 0;
  padding: 16px 20px;
  border-left: 4px solid var(--quote-border, #3b82f6);
  background: var(--quote-bg, #f8fafc);
  border-radius: 0 8px 8px 0;
}

.md-blockquote p {
  margin: 0;
  font-style: italic;
}

/* Horizontal rule */
.md-hr {
  border: none;
  height: 1px;
  background: var(--border-color, #e5e7eb);
  margin: 32px 0;
}
```

---

## Streaming Considerations

### Formatting During Stream

```jsx
function StreamingFormattedText({ content, isStreaming }) {
  // Add cursor for streaming text
  const displayContent = isStreaming 
    ? content + '▋' 
    : content;
  
  return (
    <div className={`formatted-text ${isStreaming ? 'streaming' : ''}`}>
      <FullyFormattedMarkdown content={displayContent} />
    </div>
  );
}
```

```css
.formatted-text.streaming .md-paragraph:last-child::after {
  content: '';
  display: inline-block;
  width: 2px;
  height: 1.2em;
  background: var(--cursor-color, #3b82f6);
  margin-left: 2px;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use semantic HTML elements | Use divs for everything |
| Provide adequate spacing | Cram content together |
| Style nested lists differently | Use same bullets at all levels |
| Add visual callout types | Use generic blockquotes only |
| Support dark mode | Hardcode light-only colors |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Lists inside paragraphs break | Ensure proper markdown spacing |
| Nested lists lose indentation | Use `padding-left` not `margin` |
| Inline code wraps awkwardly | Use `white-space: nowrap` for short code |
| Bold+italic don't combine | Test combined styles explicitly |
| Blockquote children not styled | Target `.md-blockquote p` specifically |

---

## Hands-on Exercise

### Your Task

Create a formatted markdown renderer that:
1. Styles bold, italic, and inline code
2. Renders nested unordered lists with different bullet styles
3. Displays callout-style blockquotes
4. Works with streaming content

### Requirements

1. Three levels of bullet styles
2. Callout types: note, warning, important
3. Streaming cursor animation
4. Dark mode support

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `::before` pseudo-elements for custom bullets
- Detect callout type from first line content
- CSS custom properties for dark mode
- Add cursor with `::after` on last paragraph

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `FullyFormattedMarkdown` and `CalloutBlockquote` components with the complete stylesheet above.

</details>

---

## Summary

✅ **Inline formatting** needs consistent, combinable styles  
✅ **Lists** require proper nesting and visual hierarchy  
✅ **Blockquotes** can be enhanced as callouts  
✅ **Semantic HTML** improves accessibility  
✅ **Streaming** needs special cursor handling  
✅ **Dark mode** should be supported via CSS variables

---

## Further Reading

- [MDN List Styling](https://developer.mozilla.org/en-US/docs/Web/CSS/list-style)
- [CSS Counters](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_counter_styles/Using_CSS_counters)
- [Blockquote Best Practices](https://css-tricks.com/quoting-in-html-quotations-citations-and-blockquotes/)

---

**Previous:** [Parsing Markdown](./01-parsing-markdown.md)  
**Next:** [Heading Rendering](./03-heading-rendering.md)

<!-- 
Sources Consulted:
- MDN list-style: https://developer.mozilla.org/en-US/docs/Web/CSS/list-style
- MDN blockquote: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/blockquote
- remark-gfm: https://github.com/remarkjs/remark-gfm
-->
