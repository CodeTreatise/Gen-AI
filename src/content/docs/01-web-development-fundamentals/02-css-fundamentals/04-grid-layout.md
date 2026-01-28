---
title: "Grid Layout"
---

# Grid Layout

## Introduction

CSS Grid is a two-dimensional layout system designed for complex layouts. While Flexbox handles rows OR columns, Grid handles rows AND columns simultaneously—perfect for page layouts, dashboards, and image galleries.

Grid gives you precise control over both dimensions, making previously complex layouts simple to implement and maintain.

### What We'll Cover

- Grid container and item basics
- Defining rows and columns with `grid-template`
- Flexible sizing with `fr`, `minmax()`, and `auto-fit`
- Placing items with line numbers and named areas
- Alignment and spacing in grid layouts
- Responsive grid patterns

### Prerequisites

- CSS box model basics
- Understanding of flexbox concepts (helpful but not required)

---

## Creating a Grid

Enable grid by setting `display: grid` on a parent:

```css
.container {
  display: grid;
}
```

By default, items stack vertically like blocks. Define columns to see the grid effect:

```css
.container {
  display: grid;
  grid-template-columns: 200px 200px 200px;
}
```

This creates three 200px columns. Items flow left-to-right, wrapping to new rows.

---

## Defining Columns and Rows

### `grid-template-columns`

Sets the width of each column:

```css
.container {
  /* Fixed widths */
  grid-template-columns: 100px 200px 100px;
  
  /* Percentages */
  grid-template-columns: 25% 50% 25%;
  
  /* Mixed units */
  grid-template-columns: 200px auto 200px;
  
  /* Fractional units (divide available space) */
  grid-template-columns: 1fr 2fr 1fr;
}
```

### `grid-template-rows`

Sets the height of each row:

```css
.container {
  grid-template-rows: 100px 200px 100px;
  grid-template-rows: auto 1fr auto;
}
```

### The `fr` Unit

Fractional units divide available space proportionally:

```css
.container {
  grid-template-columns: 1fr 1fr 1fr; /* Equal thirds */
  grid-template-columns: 1fr 2fr;      /* 1:2 ratio */
  grid-template-columns: 200px 1fr;    /* Fixed + flexible */
}
```

`fr` respects the content minimum by default, similar to `flex-grow`.

---

## The `repeat()` Function

Avoid repetition:

```css
.container {
  /* Instead of: 1fr 1fr 1fr 1fr 1fr 1fr */
  grid-template-columns: repeat(6, 1fr);
  
  /* Patterns */
  grid-template-columns: repeat(3, 100px 200px);
  /* Results in: 100px 200px 100px 200px 100px 200px */
}
```

---

## Gap (Gutters)

Space between rows and columns:

```css
.container {
  gap: 1rem;            /* Row and column gap */
  gap: 1rem 2rem;       /* Row gap, Column gap */
  row-gap: 1rem;        /* Only between rows */
  column-gap: 2rem;     /* Only between columns */
}
```

---

## Responsive Grids with `auto-fit` and `auto-fill`

### `auto-fit`

Creates as many columns as fit, items stretch to fill:

```css
.container {
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}
```

**How it works:**
1. Create columns at least 250px wide
2. Fit as many as possible in the container
3. Stretch them to fill remaining space

### `auto-fill`

Same as `auto-fit`, but preserves empty tracks:

```css
.container {
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
}
```

**Difference:**
- `auto-fit`: Empty columns collapse, items stretch
- `auto-fill`: Empty columns remain as empty space

For most responsive grids, `auto-fit` is what you want.

---

## `minmax()`

Sets a size range:

```css
.container {
  grid-template-columns: minmax(200px, 400px) 1fr;
  /* First column: at least 200px, at most 400px */
  
  grid-template-columns: repeat(3, minmax(100px, 1fr));
  /* Each column: at least 100px, grows equally */
  
  grid-template-rows: minmax(100px, auto);
  /* At least 100px, expands for content */
}
```

---

## Placing Items

### Automatic Placement

Items automatically flow into the next available cell:

```css
.container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  /* Items flow left-to-right, then wrap */
}
```

### Line-Based Placement

Reference grid lines by number (lines are between cells):

```
       1     2     3     4
       |     |     |     |
   1 ──┼─────┼─────┼─────┤
       │  1  │  2  │  3  │
   2 ──┼─────┼─────┼─────┤
       │  4  │  5  │  6  │
   3 ──┼─────┼─────┼─────┤
```

```css
.item {
  grid-column-start: 1;
  grid-column-end: 3;   /* Spans columns 1-2 */
  
  grid-row-start: 1;
  grid-row-end: 3;      /* Spans rows 1-2 */
}

/* Shorthand */
.item {
  grid-column: 1 / 3;   /* Start at line 1, end at line 3 */
  grid-row: 1 / 3;
}

/* Using span */
.item {
  grid-column: 1 / span 2;  /* Start at 1, span 2 columns */
  grid-column: span 2;      /* Span 2 from current position */
}

/* Negative numbers (count from end) */
.item {
  grid-column: 1 / -1;  /* Full width */
}
```

### `grid-area` Shorthand

Combines row-start / column-start / row-end / column-end:

```css
.item {
  grid-area: 1 / 1 / 3 / 4;
  /* Row start: 1, Column start: 1, Row end: 3, Column end: 4 */
}
```

---

## Named Grid Areas

Define layout visually with names:

```css
.container {
  display: grid;
  grid-template-columns: 200px 1fr 200px;
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "header  header  header"
    "sidebar content aside"
    "footer  footer  footer";
  min-height: 100vh;
}

.header  { grid-area: header; }
.sidebar { grid-area: sidebar; }
.content { grid-area: content; }
.aside   { grid-area: aside; }
.footer  { grid-area: footer; }
```

**Rules:**
- Each row is a string of space-separated names
- Use `.` for empty cells
- Area names must be rectangular

### Empty Cells

```css
grid-template-areas:
  "header header header"
  "sidebar content ."
  "footer footer footer";
```

---

## Named Lines

Name grid lines for clearer placement:

```css
.container {
  grid-template-columns: 
    [full-start] 1fr 
    [content-start] 2fr 
    [content-end] 1fr 
    [full-end];
}

.hero {
  grid-column: full-start / full-end;
}

.article {
  grid-column: content-start / content-end;
}
```

---

## Implicit Grid

When items don't fit the explicit grid, new tracks are created automatically:

```css
.container {
  grid-template-columns: repeat(3, 1fr);
  /* Only defines columns */
  
  /* Size auto-created rows */
  grid-auto-rows: 100px;
  grid-auto-rows: minmax(100px, auto);
}
```

### Auto Flow

Controls how auto-placed items fill the grid:

```css
.container {
  grid-auto-flow: row;         /* Default: fill rows left-to-right */
  grid-auto-flow: column;      /* Fill columns top-to-bottom */
  grid-auto-flow: dense;       /* Fill holes aggressively */
  grid-auto-flow: row dense;   /* Row flow, fill gaps */
}
```

> **Note:** `dense` can cause items to appear out of DOM order—accessibility concern.

---

## Alignment

### Align Items (Vertical in Cell)

```css
.container {
  align-items: stretch;   /* Default: fill cell height */
  align-items: start;     /* Top of cell */
  align-items: end;       /* Bottom of cell */
  align-items: center;    /* Vertical center */
}
```

### Justify Items (Horizontal in Cell)

```css
.container {
  justify-items: stretch; /* Default: fill cell width */
  justify-items: start;   /* Left of cell */
  justify-items: end;     /* Right of cell */
  justify-items: center;  /* Horizontal center */
}
```

### Shorthand: `place-items`

```css
.container {
  place-items: center;          /* Both center */
  place-items: start end;       /* Align, Justify */
}
```

### Align Content (Grid in Container)

When grid is smaller than container, align the entire grid:

```css
.container {
  height: 500px;
  align-content: start;
  align-content: center;
  align-content: space-between;
}
```

### Justify Content (Grid in Container)

```css
.container {
  justify-content: center;
  justify-content: space-between;
}
```

### Single Item Alignment

Override container alignment for one item:

```css
.special {
  align-self: end;
  justify-self: center;
  place-self: end center;
}
```

---

## Grid vs Flexbox

| Use Case | Best Choice |
|----------|-------------|
| One-dimensional (row OR column) | Flexbox |
| Two-dimensional (rows AND columns) | Grid |
| Content should determine size | Flexbox |
| Layout should determine size | Grid |
| Vertical centering | Either (both easy) |
| Equal-height columns | Either |
| Complex page layouts | Grid |
| Component-level layouts | Often Flexbox |
| Image galleries | Grid |

**They work great together:**

```css
.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.card {
  display: flex;
  flex-direction: column;
}

.card-content {
  flex: 1;
}
```

---

## Common Grid Patterns

### Basic Card Grid

```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}
```

### Dashboard Layout

```css
.dashboard {
  display: grid;
  grid-template-columns: 250px 1fr;
  grid-template-rows: auto 1fr;
  grid-template-areas:
    "sidebar header"
    "sidebar main";
  min-height: 100vh;
}

.sidebar { grid-area: sidebar; }
.header  { grid-area: header; }
.main    { grid-area: main; }
```

### Magazine Layout

```css
.magazine {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-auto-rows: 200px;
  gap: 1rem;
}

.feature {
  grid-column: span 2;
  grid-row: span 2;
}

.standard {
  grid-column: span 1;
  grid-row: span 1;
}
```

### Chat Message Grid

```css
.chat-grid {
  display: grid;
  grid-template-columns: 48px 1fr;
  gap: 0.5rem 1rem;
  align-items: start;
}

.avatar {
  grid-row: span 2;
}

.message-header {
  align-self: end;
}

.message-body {
  align-self: start;
}
```

### Full-Bleed Layout

```css
.full-bleed {
  display: grid;
  grid-template-columns:
    1fr
    min(65ch, 100% - 2rem)
    1fr;
}

.full-bleed > * {
  grid-column: 2;
}

.full-bleed > .bleed {
  grid-column: 1 / -1;
  width: 100%;
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `auto-fit` with `minmax()` for responsive grids | No media queries needed |
| Name areas for complex layouts | Self-documenting layout |
| Use `fr` units over percentages | Accounts for gaps automatically |
| Set `grid-auto-rows: minmax(X, auto)` | Consistent minimum height |
| Combine Grid for layout, Flexbox for components | Each tool's strength |
| Use `gap` instead of margins | Consistent spacing |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Items overflowing grid | Add `min-width: 0` on items |
| `auto-fit` not wrapping | Container needs defined/constrained width |
| Grid items ignoring `height: 100%` | Grid auto-sizes rows; set explicit height or `grid-auto-rows` |
| Mixing `grid-template-areas` and line numbers | Pick one method per layout |
| Forgetting gaps reduce available `fr` space | That's correct behavior, adjust if needed |

### The Overflow Gotcha

Like flexbox, grid items default to `min-width: auto`:

```css
.grid-item {
  min-width: 0; /* Allow shrinking below content */
}
```

---

## Hands-on Exercise

### Your Task

Build a responsive image gallery with these requirements:

1. Grid of images that adapts to screen width
2. Minimum column width of 250px
3. Some images span 2 columns (featured)
4. Gap between images
5. Images maintain aspect ratio

### Requirements

1. Use `auto-fit` with `minmax()`
2. Featured images use `grid-column: span 2`
3. Handle the case where featured images shouldn't span on narrow screens

<details>
<summary>✅ Solution</summary>

```css
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.gallery-item {
  overflow: hidden;
  border-radius: 8px;
}

.gallery-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  aspect-ratio: 4 / 3;
}

/* Featured: spans 2 columns on wider screens */
.gallery-item--featured {
  grid-column: span 2;
  grid-row: span 2;
}

.gallery-item--featured img {
  aspect-ratio: 1 / 1;
}

/* On narrow screens, featured shouldn't span */
@media (max-width: 550px) {
  .gallery-item--featured {
    grid-column: span 1;
    grid-row: span 1;
  }
  
  .gallery-item--featured img {
    aspect-ratio: 4 / 3;
  }
}
```

```html
<div class="gallery">
  <div class="gallery-item gallery-item--featured">
    <img src="featured.jpg" alt="Featured photo">
  </div>
  <div class="gallery-item">
    <img src="photo1.jpg" alt="Photo 1">
  </div>
  <div class="gallery-item">
    <img src="photo2.jpg" alt="Photo 2">
  </div>
  <div class="gallery-item">
    <img src="photo3.jpg" alt="Photo 3">
  </div>
  <div class="gallery-item">
    <img src="photo4.jpg" alt="Photo 4">
  </div>
</div>
```
</details>

---

## Summary

✅ **CSS Grid** is for two-dimensional layouts (rows AND columns)

✅ **`grid-template-columns`** and **`grid-template-rows`** define the grid structure

✅ **`fr` units** distribute available space proportionally

✅ **`repeat(auto-fit, minmax(X, 1fr))`** creates responsive grids without media queries

✅ **Line numbers** or **named areas** place items precisely

✅ **`gap`** creates consistent gutters without margin math

✅ Combine **Grid** for page layout and **Flexbox** for component layout

---

**Previous:** [Flexbox Layout](./03-flexbox-layout.md)

**Next:** [Responsive Design](./05-responsive-design.md)

<!-- 
Sources Consulted:
- MDN CSS Grid Layout: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout
- MDN grid-template-columns: https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-columns
- MDN auto-fit/auto-fill: https://developer.mozilla.org/en-US/docs/Web/CSS/repeat
- MDN grid-template-areas: https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-areas
-->
