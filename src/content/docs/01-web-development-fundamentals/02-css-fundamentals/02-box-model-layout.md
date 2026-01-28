---
title: "Box Model & Layout"
---

# Box Model & Layout

## Introduction

Every element in CSS is a rectangular box. Understanding the box model—how content, padding, border, and margin work together—is fundamental to layout. Once you master this, positioning elements becomes predictable instead of frustrating.

This lesson covers the box model, display types, and positioning schemes that control how elements occupy space and relate to each other.

### What We'll Cover

- The four box model layers: content, padding, border, margin
- `box-sizing` and why `border-box` is essential
- Block vs inline vs inline-block display
- Position schemes: static, relative, absolute, fixed, sticky
- Stacking context and `z-index`
- Overflow handling

### Prerequisites

- CSS selectors and how to apply styles
- Basic HTML document structure

---

## The Box Model

Every element generates a box with four areas:

```
┌─────────────────────────────────────┐
│             MARGIN                  │
│   ┌───────────────────────────┐     │
│   │         BORDER            │     │
│   │   ┌───────────────────┐   │     │
│   │   │     PADDING       │   │     │
│   │   │   ┌───────────┐   │   │     │
│   │   │   │  CONTENT  │   │   │     │
│   │   │   └───────────┘   │   │     │
│   │   └───────────────────┘   │     │
│   └───────────────────────────┘     │
└─────────────────────────────────────┘
```

### Content

The actual content area where text and child elements appear:

```css
.box {
  width: 300px;
  height: 200px;
}
```

By default, `width` and `height` set the content area size only.

### Padding

Space between content and border—inside the element:

```css
.box {
  /* All sides */
  padding: 1rem;
  
  /* Vertical | Horizontal */
  padding: 1rem 2rem;
  
  /* Top | Horizontal | Bottom */
  padding: 1rem 2rem 1.5rem;
  
  /* Top | Right | Bottom | Left (clockwise) */
  padding: 1rem 2rem 1.5rem 2rem;
  
  /* Individual sides */
  padding-top: 1rem;
  padding-right: 2rem;
  padding-bottom: 1rem;
  padding-left: 2rem;
}
```

### Border

The edge of the element, between padding and margin:

```css
.box {
  /* Shorthand: width style color */
  border: 1px solid #d1d5db;
  
  /* Individual properties */
  border-width: 1px;
  border-style: solid;
  border-color: #d1d5db;
  
  /* Individual sides */
  border-top: 2px solid #6366f1;
  border-bottom: none;
  
  /* Border radius */
  border-radius: 8px;
  border-radius: 50%; /* Circle */
}
```

**Border styles:** `none`, `solid`, `dashed`, `dotted`, `double`, `groove`, `ridge`, `inset`, `outset`

### Margin

Space outside the border—between this element and others:

```css
.box {
  /* Same shorthand patterns as padding */
  margin: 1rem;
  margin: 1rem auto; /* Center horizontally */
  margin: 0 0 1rem 0;
  
  /* Auto for centering */
  margin-left: auto;
  margin-right: auto;
}
```

### Margin Collapsing

Vertical margins collapse—the larger wins:

```css
.first {
  margin-bottom: 20px;
}

.second {
  margin-top: 30px;
}

/* Space between them is 30px, not 50px */
```

**Margin collapse rules:**
- Only vertical margins collapse (not horizontal)
- Only adjacent siblings or parent/first-child
- Does NOT happen with flexbox or grid items
- Does NOT happen with floated or absolutely positioned elements

---

## `box-sizing`

### The Problem: Default `content-box`

By default, width/height apply to content only:

```css
.box {
  width: 300px;
  padding: 20px;
  border: 5px solid;
}

/* Actual rendered width: 300 + 20 + 20 + 5 + 5 = 350px! */
```

### The Solution: `border-box`

With `border-box`, width/height include padding and border:

```css
.box {
  box-sizing: border-box;
  width: 300px;
  padding: 20px;
  border: 5px solid;
}

/* Actual rendered width: 300px exactly */
/* Content area: 300 - 20 - 20 - 5 - 5 = 250px */
```

### Universal Reset

Apply `border-box` globally—this is standard practice:

```css
*, *::before, *::after {
  box-sizing: border-box;
}
```

Or inherit from html for flexibility:

```css
html {
  box-sizing: border-box;
}

*, *::before, *::after {
  box-sizing: inherit;
}
```

---

## Display Types

The `display` property determines how an element generates boxes.

### Block Elements

- Take full available width
- Start on a new line
- Respect width, height, margin, padding

```css
.block {
  display: block;
  width: 80%;
  margin: 1rem auto;
}
```

**Default block elements:** `<div>`, `<p>`, `<h1>`-`<h6>`, `<section>`, `<article>`, `<header>`, `<footer>`, `<ul>`, `<li>`, `<form>`

### Inline Elements

- Flow with text
- Only take needed width
- Ignore width/height
- Horizontal margin/padding works, vertical doesn't push content

```css
.highlight {
  display: inline;
  background: yellow;
  padding: 0 0.25em; /* Horizontal only */
}
```

**Default inline elements:** `<span>`, `<a>`, `<strong>`, `<em>`, `<code>`, `<img>` (inline-replaced)

### Inline-Block

- Flows inline like text
- Respects width, height, margin, padding like block

```css
.button {
  display: inline-block;
  padding: 0.5rem 1rem;
  width: 200px;
  text-align: center;
}
```

**Use case:** Buttons, navigation items, badges

### None

Removes element from layout entirely:

```css
.hidden {
  display: none;
}
```

> **Note:** `display: none` vs `visibility: hidden`—`none` removes from layout, `hidden` keeps the space.

---

## Positioning

The `position` property changes how an element is placed in the document.

### Static (Default)

Normal document flow, ignores `top`, `right`, `bottom`, `left`, `z-index`:

```css
.element {
  position: static;
}
```

### Relative

Positioned relative to its normal position:

```css
.element {
  position: relative;
  top: 10px;   /* Push down 10px from normal position */
  left: 20px;  /* Push right 20px from normal position */
}
```

- Element still occupies its original space
- Other elements don't move
- Creates a positioning context for absolute children

### Absolute

Removed from flow, positioned relative to nearest positioned ancestor:

```css
.parent {
  position: relative; /* Creates positioning context */
}

.child {
  position: absolute;
  top: 0;
  right: 0;
}
```

**Finding the positioning context:**
1. Look at parent: is it `position: relative/absolute/fixed/sticky`?
2. If not, check grandparent, and so on
3. If no positioned ancestor, uses the viewport

### Fixed

Positioned relative to the viewport, stays in place during scroll:

```css
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
}
```

**Use cases:** Navigation bars, chat widgets, cookie banners

### Sticky

Hybrid of relative and fixed—stays relative until a scroll threshold:

```css
.section-header {
  position: sticky;
  top: 0; /* Stick when reaching top of viewport */
  background: white;
  z-index: 10;
}
```

```css
/* Table headers that stick */
th {
  position: sticky;
  top: 0;
  background: #f3f4f6;
}
```

**Requirements for sticky:**
- Must specify at least one of `top`, `right`, `bottom`, `left`
- Parent cannot have `overflow: hidden` or `overflow: auto`
- Works within the parent's scrollable area

---

## Z-Index and Stacking

### Z-Index Basics

Controls stacking order for positioned elements:

```css
.bottom {
  position: relative;
  z-index: 1;
}

.top {
  position: relative;
  z-index: 2; /* Appears above .bottom */
}
```

### Stacking Context

A stacking context is an isolated group where z-index is calculated:

**Elements that create stacking context:**
- Root element (`<html>`)
- `position: absolute/relative/fixed/sticky` with `z-index` other than `auto`
- `opacity` less than 1
- `transform`, `filter`, `perspective`
- `isolation: isolate`
- `will-change` (for some values)

```css
/* Parent creates stacking context */
.parent {
  position: relative;
  z-index: 1;
}

/* This z-index: 9999 only works within parent's context */
.child {
  position: absolute;
  z-index: 9999;
}
```

### Isolation

Force a new stacking context without other effects:

```css
.modal-backdrop {
  isolation: isolate;
}
```

---

## Overflow

Controls what happens when content exceeds its container.

### Overflow Values

```css
.container {
  overflow: visible; /* Default, content spills out */
  overflow: hidden;  /* Content clipped, no scrollbar */
  overflow: scroll;  /* Always show scrollbars */
  overflow: auto;    /* Scrollbars only when needed */
  overflow: clip;    /* Like hidden, but no scroll programmatically */
}

/* Individual axes */
.container {
  overflow-x: auto;
  overflow-y: hidden;
}
```

### Common Patterns

#### Scrollable Container

```css
.chat-messages {
  height: 400px;
  overflow-y: auto;
  scroll-behavior: smooth;
}
```

#### Text Truncation

```css
.card-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

#### Multi-line Truncation

```css
.card-description {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

---

## Practical Layout Examples

### Centered Container

```css
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}
```

### Card Component

```css
.card {
  box-sizing: border-box;
  width: 100%;
  max-width: 400px;
  padding: 1.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: white;
}

.card-header {
  margin: -1.5rem -1.5rem 1rem -1.5rem; /* Bleed to edges */
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: #f9fafb;
  border-radius: 8px 8px 0 0;
}
```

### Modal Overlay

```css
.modal-overlay {
  position: fixed;
  inset: 0; /* Shorthand for top/right/bottom/left: 0 */
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  position: relative;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
  background: white;
  border-radius: 8px;
  padding: 2rem;
}
```

### Floating Action Button

```css
.fab {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: #6366f1;
  color: white;
  border: none;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 100;
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `box-sizing: border-box` globally | Predictable sizing |
| Prefer `padding` over `margin` for spacing inside components | Avoids margin collapse issues |
| Use `max-width` instead of `width` | Allows responsiveness |
| Create positioning context with `position: relative` | Required for absolute children |
| Use `inset: 0` shorthand | Cleaner than four properties |
| Prefer CSS for layout over absolute positioning | More maintainable |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting `position: relative` on parent | Add it for absolute children |
| Using `width: 100%` with padding | Use `box-sizing: border-box` |
| Absolute position without positioning context | Explicitly set `position: relative` on intended ancestor |
| `z-index` not working | Element needs position other than static |
| Sticky not sticking | Check parent overflow and scroll container |
| Margin collapse confusion | Use flexbox/grid or padding instead |

---

## Hands-on Exercise

### Your Task

Create a chat message component with:

1. A message bubble with proper padding and border-radius
2. A timestamp absolutely positioned in the corner
3. Different styles for "sent" (right-aligned) vs "received" (left-aligned)
4. A sticky date separator between message groups

### Requirements

1. Base `.message` class with box model properties
2. `.message--sent` and `.message--received` modifiers
3. Timestamp positioned bottom-right of message
4. Date separator that sticks to top of scroll container

<details>
<summary>✅ Solution</summary>

```css
/* Chat container */
.chat-container {
  height: 400px;
  overflow-y: auto;
  padding: 1rem;
  background: #f9fafb;
}

/* Date separator */
.date-separator {
  position: sticky;
  top: 0;
  text-align: center;
  padding: 0.5rem;
  background: #f9fafb;
  z-index: 10;
}

.date-separator span {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background: #e5e7eb;
  border-radius: 1rem;
  font-size: 0.75rem;
  color: #6b7280;
}

/* Base message */
.message {
  position: relative;
  max-width: 70%;
  padding: 0.75rem 1rem;
  padding-bottom: 1.25rem; /* Space for timestamp */
  margin-bottom: 0.5rem;
  border-radius: 1rem;
  box-sizing: border-box;
}

/* Timestamp */
.message-time {
  position: absolute;
  bottom: 0.25rem;
  right: 0.75rem;
  font-size: 0.625rem;
  color: #9ca3af;
}

/* Received message (left) */
.message--received {
  background: white;
  border: 1px solid #e5e7eb;
  margin-right: auto;
  border-bottom-left-radius: 4px;
}

/* Sent message (right) */
.message--sent {
  background: #6366f1;
  color: white;
  margin-left: auto;
  border-bottom-right-radius: 4px;
}

.message--sent .message-time {
  color: rgba(255, 255, 255, 0.7);
}
```

```html
<div class="chat-container">
  <div class="date-separator">
    <span>Today</span>
  </div>
  
  <div class="message message--received">
    Hello! How can I help you today?
    <span class="message-time">10:30 AM</span>
  </div>
  
  <div class="message message--sent">
    I have a question about CSS positioning.
    <span class="message-time">10:31 AM</span>
  </div>
</div>
```
</details>

---

## Summary

✅ The **box model** has four layers: content → padding → border → margin

✅ Always use **`box-sizing: border-box`** for predictable sizing

✅ **Block** elements take full width, **inline** flows with text, **inline-block** combines both behaviors

✅ **Relative** offsets from normal position, **absolute** positions within a positioned ancestor, **fixed** stays in viewport, **sticky** switches between relative and fixed

✅ **Z-index** only works on positioned elements and within stacking contexts

✅ **Overflow** controls clipping and scrolling when content exceeds its container

---

**Previous:** [Selectors & Specificity](./01-selectors-specificity.md)

**Next:** [Flexbox Layout](./03-flexbox-layout.md)

<!-- 
Sources Consulted:
- MDN Box Model: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_box_model
- MDN position: https://developer.mozilla.org/en-US/docs/Web/CSS/position
- MDN overflow: https://developer.mozilla.org/en-US/docs/Web/CSS/overflow
- MDN z-index: https://developer.mozilla.org/en-US/docs/Web/CSS/z-index
-->
