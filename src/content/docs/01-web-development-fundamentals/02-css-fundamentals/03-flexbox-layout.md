---
title: "Flexbox Layout"
---

# Flexbox Layout

## Introduction

Flexbox is a one-dimensional layout system designed for arranging items in rows or columns. Before flexbox, centering content and creating equal-height columns required hacks. Now, these are one-line solutions.

Flexbox excels at distributing space, aligning items, and handling dynamic content—exactly what you need for chat interfaces, toolbars, and responsive components.

### What We'll Cover

- Flex container properties: direction, wrap, justify, align
- Flex item properties: grow, shrink, basis
- Alignment techniques for common patterns
- Real-world layout examples

### Prerequisites

- CSS box model basics
- Understanding of display property

---

## Flex Container Basics

Enable flexbox by setting `display: flex` on a parent:

```css
.container {
  display: flex;
}
```

**Immediate effects:**
- Children become flex items
- Items line up in a row (default)
- Items shrink to fit their content
- Items stretch to equal height

```html
<div class="container">
  <div class="item">One</div>
  <div class="item">Two</div>
  <div class="item">Three</div>
</div>
```

### Inline Flex

Use `inline-flex` for containers that should flow inline:

```css
.button-group {
  display: inline-flex;
  gap: 0.5rem;
}
```

---

## Flex Direction

Controls the main axis—the direction items flow:

```css
.container {
  flex-direction: row;            /* Default: left to right */
  flex-direction: row-reverse;    /* Right to left */
  flex-direction: column;         /* Top to bottom */
  flex-direction: column-reverse; /* Bottom to top */
}
```

### Understanding Axes

| Direction | Main Axis | Cross Axis |
|-----------|-----------|------------|
| `row` | Horizontal (→) | Vertical (↓) |
| `row-reverse` | Horizontal (←) | Vertical (↓) |
| `column` | Vertical (↓) | Horizontal (→) |
| `column-reverse` | Vertical (↑) | Horizontal (→) |

---

## Flex Wrap

By default, items try to fit on one line. Enable wrapping:

```css
.container {
  flex-wrap: nowrap;       /* Default: single line, items shrink */
  flex-wrap: wrap;         /* Wrap to new lines as needed */
  flex-wrap: wrap-reverse; /* Wrap upward/leftward */
}
```

### Shorthand: `flex-flow`

Combine direction and wrap:

```css
.container {
  flex-flow: row wrap;
  /* Same as:
     flex-direction: row;
     flex-wrap: wrap;
  */
}
```

---

## Justify Content (Main Axis)

Distributes space along the main axis:

```css
.container {
  justify-content: flex-start;    /* Default: pack at start */
  justify-content: flex-end;      /* Pack at end */
  justify-content: center;        /* Center items */
  justify-content: space-between; /* Equal space between, none at edges */
  justify-content: space-around;  /* Equal space around each item */
  justify-content: space-evenly;  /* Equal space between and at edges */
}
```

### Visual Reference

```
flex-start:      [A][B][C]                    
flex-end:                          [A][B][C]
center:                   [A][B][C]           
space-between:   [A]          [B]          [C]
space-around:      [A]      [B]      [C]      
space-evenly:       [A]      [B]      [C]     
```

---

## Align Items (Cross Axis)

Aligns items along the cross axis (perpendicular to main):

```css
.container {
  align-items: stretch;    /* Default: fill container height */
  align-items: flex-start; /* Align at cross-start */
  align-items: flex-end;   /* Align at cross-end */
  align-items: center;     /* Center on cross axis */
  align-items: baseline;   /* Align text baselines */
}
```

### Centering: The One-Liner

```css
.container {
  display: flex;
  justify-content: center;
  align-items: center;
}
```

This centers content both horizontally and vertically.

---

## Align Content (Multi-Line)

Controls spacing between flex lines when wrapping:

```css
.container {
  display: flex;
  flex-wrap: wrap;
  
  align-content: flex-start;    /* Pack lines at start */
  align-content: flex-end;      /* Pack lines at end */
  align-content: center;        /* Center lines */
  align-content: space-between; /* Space between lines */
  align-content: space-around;  /* Space around lines */
  align-content: stretch;       /* Default: lines stretch to fill */
}
```

> **Note:** `align-content` only applies when there are multiple lines (wrap is enabled and items actually wrap).

---

## Gap

Space between flex items (no margin hacks needed):

```css
.container {
  display: flex;
  gap: 1rem;         /* Row and column gap */
  gap: 1rem 2rem;    /* Row gap, Column gap */
  row-gap: 1rem;     /* Only between rows */
  column-gap: 2rem;  /* Only between columns */
}
```

### Gap vs Margin

```css
/* Old way with margin */
.item {
  margin-right: 1rem;
}
.item:last-child {
  margin-right: 0;
}

/* New way with gap - cleaner! */
.container {
  gap: 1rem;
}
```

---

## Flex Item Properties

These apply to children of a flex container.

### Flex Grow

How much an item grows relative to siblings:

```css
.item {
  flex-grow: 0; /* Default: don't grow */
  flex-grow: 1; /* Grow to fill available space */
}

/* Proportional sizing */
.sidebar {
  flex-grow: 1; /* Takes 1 part */
}
.main {
  flex-grow: 3; /* Takes 3 parts (3x wider) */
}
```

### Flex Shrink

How much an item shrinks when space is tight:

```css
.item {
  flex-shrink: 1; /* Default: shrink equally */
  flex-shrink: 0; /* Never shrink below content size */
}

/* Logo shouldn't shrink */
.logo {
  flex-shrink: 0;
}
```

### Flex Basis

The initial size before growing/shrinking:

```css
.item {
  flex-basis: auto;   /* Default: use width/height or content */
  flex-basis: 200px;  /* Start at 200px */
  flex-basis: 30%;    /* Start at 30% of container */
  flex-basis: 0;      /* Start at 0, rely on flex-grow */
}
```

### Shorthand: `flex`

Combines grow, shrink, and basis:

```css
.item {
  flex: 0 1 auto;    /* Default: don't grow, can shrink, auto basis */
  flex: 1;           /* Same as: flex: 1 1 0 - grow equally */
  flex: auto;        /* Same as: flex: 1 1 auto */
  flex: none;        /* Same as: flex: 0 0 auto - rigid */
  flex: 1 0 200px;   /* Grow, don't shrink, start at 200px */
}
```

**Recommended patterns:**
- `flex: 1` — items share space equally
- `flex: 0 0 auto` — fixed size (or use `flex: none`)
- `flex: 1 1 0` — equal sizing ignoring content

---

## Align Self

Override `align-items` for individual items:

```css
.container {
  display: flex;
  align-items: flex-start;
}

.special-item {
  align-self: center;    /* This one centers differently */
  align-self: stretch;   /* This one stretches */
}
```

---

## Order

Change visual order without changing HTML:

```css
.item {
  order: 0; /* Default */
}

.first {
  order: -1; /* Appears first */
}

.last {
  order: 1; /* Appears last */
}
```

> **Accessibility warning:** Screen readers follow DOM order, not visual order. Use sparingly.

---

## Common Flexbox Patterns

### Navigation Bar

```css
.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}
```

```html
<nav class="nav">
  <a href="/" class="logo">Brand</a>
  <ul class="nav-links">
    <li><a href="#">Home</a></li>
    <li><a href="#">About</a></li>
    <li><a href="#">Contact</a></li>
  </ul>
</nav>
```

### Card Footer with Spacer

```css
.card {
  display: flex;
  flex-direction: column;
  height: 300px;
}

.card-content {
  flex: 1; /* Grows to push footer down */
}

.card-footer {
  margin-top: auto; /* Alternative: pushed to bottom */
}
```

### Input with Button

```css
.input-group {
  display: flex;
}

.input-group input {
  flex: 1;
  border-radius: 4px 0 0 4px;
}

.input-group button {
  flex-shrink: 0;
  border-radius: 0 4px 4px 0;
}
```

### Chat Interface Layout

```css
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.chat-header {
  flex-shrink: 0;
  padding: 1rem;
  border-bottom: 1px solid #e5e7eb;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.chat-input {
  flex-shrink: 0;
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  border-top: 1px solid #e5e7eb;
}

.chat-input textarea {
  flex: 1;
}

.chat-input button {
  flex-shrink: 0;
}
```

### Equal Height Cards

```css
.card-grid {
  display: flex;
  gap: 1rem;
}

.card {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.card-body {
  flex: 1; /* All cards same height */
}
```

### Centering Anything

```css
.center-me {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}
```

### Holy Grail Layout

```css
.holy-grail {
  display: flex;
  min-height: 100vh;
  flex-direction: column;
}

.header, .footer {
  flex-shrink: 0;
}

.main-content {
  flex: 1;
  display: flex;
}

.sidebar-left, .sidebar-right {
  flex: 0 0 200px;
}

.content {
  flex: 1;
}
```

---

## Responsive Flexbox

### Wrap on Small Screens

```css
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.toolbar-item {
  flex: 1 1 auto;
  min-width: 100px;
}

@media (max-width: 600px) {
  .toolbar-item {
    flex: 1 1 100%; /* Full width on mobile */
  }
}
```

### Row to Column

```css
.feature-section {
  display: flex;
  gap: 2rem;
}

@media (max-width: 768px) {
  .feature-section {
    flex-direction: column;
  }
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `gap` instead of margins | Cleaner, no last-child hacks |
| Set `flex-shrink: 0` on fixed elements | Prevents unwanted shrinking |
| Use `flex: 1` for equal sizing | Simpler than manual widths |
| Combine with `min-width: 0` | Prevents flex items from overflowing |
| Use `flex-direction: column` for vertical layouts | Natural for full-height designs |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Text overflowing flex items | Add `min-width: 0` or `overflow: hidden` |
| Items not wrapping | Add `flex-wrap: wrap` |
| Uneven item sizes with `flex: 1` | Use `flex: 1 1 0` instead of `flex: 1 1 auto` |
| Forgetting `height: 100%` on children | Flex doesn't automatically pass height down |
| Using `justify-content` for cross-axis | Remember: justify = main, align = cross |

### The Min-Width Gotcha

Flex items have `min-width: auto` by default, preventing shrinking below content:

```css
.item {
  /* Fix: allow shrinking below content size */
  min-width: 0;
  /* or */
  overflow: hidden;
}
```

---

## Hands-on Exercise

### Your Task

Build a responsive toolbar component:

1. Items arranged horizontally with gap
2. Logo on the left, navigation in the center, actions on the right
3. On mobile: stack vertically, full-width items
4. User avatar stays fixed size, search expands

### Requirements

1. Use flexbox only (no grid)
2. Items should not shrink smaller than their content
3. Search bar should expand to fill available space
4. Mobile breakpoint at 768px

<details>
<summary>✅ Solution</summary>

```css
.toolbar {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1rem;
  background: white;
  border-bottom: 1px solid #e5e7eb;
}

.toolbar-logo {
  flex-shrink: 0;
  font-weight: bold;
  font-size: 1.25rem;
}

.toolbar-nav {
  display: flex;
  gap: 1rem;
  flex-shrink: 0;
}

.toolbar-search {
  flex: 1; /* Expands to fill space */
  min-width: 100px;
}

.toolbar-search input {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 4px;
}

.toolbar-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* Mobile: Stack vertically */
@media (max-width: 768px) {
  .toolbar {
    flex-wrap: wrap;
  }
  
  .toolbar-logo {
    order: 1;
  }
  
  .toolbar-actions {
    order: 2;
    margin-left: auto;
  }
  
  .toolbar-search {
    order: 3;
    flex: 1 1 100%;
  }
  
  .toolbar-nav {
    order: 4;
    flex: 1 1 100%;
    justify-content: center;
  }
}
```

```html
<div class="toolbar">
  <div class="toolbar-logo">AppName</div>
  
  <nav class="toolbar-nav">
    <a href="#">Dashboard</a>
    <a href="#">Projects</a>
    <a href="#">Settings</a>
  </nav>
  
  <div class="toolbar-search">
    <input type="search" placeholder="Search...">
  </div>
  
  <div class="toolbar-actions">
    <button>Upgrade</button>
    <img src="avatar.jpg" alt="User" class="avatar">
  </div>
</div>
```
</details>

---

## Summary

✅ **`display: flex`** creates a flex container; children become flex items

✅ **`flex-direction`** sets the main axis (row or column)

✅ **`justify-content`** distributes space along the main axis

✅ **`align-items`** aligns items along the cross axis

✅ **`gap`** creates consistent spacing without margin hacks

✅ **`flex: 1`** makes items grow equally to fill space

✅ **`flex-shrink: 0`** prevents items from shrinking below their content

---

**Previous:** [Box Model & Layout](./02-box-model-layout.md)

**Next:** [Grid Layout](./04-grid-layout.md)

<!-- 
Sources Consulted:
- MDN Flexbox: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_flexible_box_layout
- MDN flex property: https://developer.mozilla.org/en-US/docs/Web/CSS/flex
- MDN justify-content: https://developer.mozilla.org/en-US/docs/Web/CSS/justify-content
- MDN align-items: https://developer.mozilla.org/en-US/docs/Web/CSS/align-items
-->
