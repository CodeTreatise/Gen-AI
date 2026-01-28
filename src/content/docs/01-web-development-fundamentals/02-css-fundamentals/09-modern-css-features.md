---
title: "Modern CSS Features"
---

# Modern CSS Features

## Introduction

CSS continues to evolve rapidly. Features that once required JavaScript or complex workarounds are now built into the language. This lesson covers the most impactful modern CSS features you should know for 2024 and beyond.

These features let you write cleaner, more maintainable code while building sophisticated interfaces with less effort.

### What We'll Cover

- Container queries for component-based responsive design
- The `:has()` relational pseudo-class (parent selector)
- CSS nesting for cleaner stylesheets
- Cascade layers for managing specificity
- Logical properties for internationalization
- Modern color functions and color spaces
- Subgrid for aligned nested grids

### Prerequisites

- Solid understanding of CSS selectors
- Familiarity with Flexbox and Grid
- Understanding of media queries

---

## Container Queries

Container queries let components respond to their container's size rather than the viewport—perfect for reusable components.

### Basic Syntax

```css
/* Step 1: Define containment */
.card-container {
  container-type: inline-size;
  container-name: card;
}

/* Shorthand */
.card-container {
  container: card / inline-size;
}

/* Step 2: Query the container */
@container card (min-width: 400px) {
  .card {
    display: flex;
    gap: 1rem;
  }
  
  .card-image {
    width: 40%;
  }
}
```

### Container Types

```css
.container {
  /* Query inline size only (width in horizontal writing) */
  container-type: inline-size;
  
  /* Query both dimensions (rarely needed) */
  container-type: size;
  
  /* No size containment, for style queries */
  container-type: normal;
}
```

> **Note:** `size` containment requires explicit dimensions on the container.

### Container Query Units

Size relative to the container, not viewport:

| Unit | Description |
|------|-------------|
| `cqi` | 1% of container's inline size |
| `cqb` | 1% of container's block size |
| `cqw` | 1% of container's width |
| `cqh` | 1% of container's height |
| `cqmin` | Smaller of cqi/cqb |
| `cqmax` | Larger of cqi/cqb |

```css
.card-title {
  font-size: clamp(1rem, 5cqi, 1.5rem);
}
```

### Practical Example: Responsive Card

```css
.card-wrapper {
  container: card / inline-size;
}

.card {
  display: grid;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

/* Stack on small containers */
@container card (max-width: 399px) {
  .card {
    grid-template-columns: 1fr;
  }
  
  .card-image {
    aspect-ratio: 16 / 9;
  }
}

/* Side-by-side on larger containers */
@container card (min-width: 400px) {
  .card {
    grid-template-columns: 2fr 3fr;
    align-items: start;
  }
  
  .card-image {
    aspect-ratio: 1;
  }
}

/* Full featured on large containers */
@container card (min-width: 600px) {
  .card {
    grid-template-columns: 1fr 2fr 1fr;
  }
  
  .card-actions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
}
```

---

## The `:has()` Selector

The relational pseudo-class—finally, a parent selector!

### Basic Usage

```css
/* Style parent based on children */
.card:has(img) {
  padding-top: 0;
}

/* Form containing invalid inputs */
form:has(:invalid) {
  border-color: #ef4444;
}

/* Section with no content */
.section:has(:not(*)) {
  display: none;
}
```

### Common Patterns

#### Style Labels When Input is Focused

```css
label:has(+ input:focus) {
  color: #6366f1;
}

/* Or when input is inside label */
label:has(input:focus) {
  color: #6366f1;
}
```

#### Card Hover When Link Inside is Focused

```css
.card:has(a:focus) {
  outline: 2px solid #6366f1;
}
```

#### Show Elements Based on State

```css
/* Show error when input is invalid */
.form-group:has(input:invalid) .error-message {
  display: block;
}

/* Dim other items when one is hovered */
.list:has(.item:hover) .item:not(:hover) {
  opacity: 0.6;
}
```

#### Responsive Without Media Queries

```css
/* If sidebar exists, adjust main content */
.page:has(.sidebar) .main {
  max-width: 800px;
}

.page:not(:has(.sidebar)) .main {
  max-width: 1200px;
}
```

### Combining with Other Selectors

```css
/* Has specific child at position */
ul:has(li:nth-child(5)) {
  /* List has at least 5 items */
}

/* Multiple conditions */
.card:has(img):has(.featured-badge) {
  border: 2px solid gold;
}

/* Either condition */
.card:has(img, video) {
  grid-template-rows: auto 1fr;
}
```

> **Browser support:** Chrome 105+, Safari 15.4+, Firefox 121+

---

## CSS Nesting

Write nested selectors like Sass, natively:

### Basic Nesting

```css
.card {
  padding: 1rem;
  border: 1px solid #e5e7eb;
  
  .card-title {
    font-weight: bold;
    margin-bottom: 0.5rem;
  }
  
  .card-content {
    color: #64748b;
  }
}

/* Compiles to:
.card { padding: 1rem; border: 1px solid #e5e7eb; }
.card .card-title { font-weight: bold; margin-bottom: 0.5rem; }
.card .card-content { color: #64748b; }
*/
```

### The `&` Selector

Reference the parent selector:

```css
.button {
  background: #6366f1;
  
  /* &:hover → .button:hover */
  &:hover {
    background: #4f46e5;
  }
  
  /* &:focus → .button:focus */
  &:focus {
    outline: 2px solid #6366f1;
  }
  
  /* &.active → .button.active */
  &.active {
    background: #4338ca;
  }
  
  /* Suffix: &--primary → .button--primary */
  &--primary {
    background: #6366f1;
  }
}
```

### Nesting Media Queries

```css
.container {
  padding: 1rem;
  
  @media (min-width: 768px) {
    padding: 2rem;
  }
  
  @media (min-width: 1024px) {
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

### Nesting Container Queries

```css
.card-wrapper {
  container: card / inline-size;
  
  .card {
    padding: 1rem;
    
    @container card (min-width: 400px) {
      display: flex;
      gap: 1rem;
    }
  }
}
```

### Nesting Rules

1. Nested selectors must start with a symbol (`.`, `#`, `&`, `@`, etc.) or use `&`
2. Element selectors need `&` prefix: `& p { }` not `p { }`

```css
.parent {
  /* ✅ Correct */
  .child { }
  &:hover { }
  & p { }
  
  /* ❌ Won't work as expected */
  p { }  /* Needs: & p { } */
}
```

---

## Cascade Layers

Control the cascade order explicitly:

### Defining Layers

```css
/* Define layer order */
@layer reset, base, components, utilities;

/* Add rules to layers */
@layer reset {
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
  }
}

@layer base {
  body {
    font-family: system-ui;
    line-height: 1.5;
  }
}

@layer components {
  .button {
    padding: 0.5rem 1rem;
    background: #6366f1;
  }
}

@layer utilities {
  .hidden { display: none !important; }
  .flex { display: flex; }
}
```

### Layer Priority

Later layers beat earlier layers, regardless of specificity:

```css
@layer base, components;

@layer base {
  .button { background: gray; }  /* Loses */
}

@layer components {
  button { background: blue; }   /* Wins, even lower specificity */
}
```

### Unlayered Styles

Styles outside layers have highest priority:

```css
@layer base {
  .text { color: black; }
}

/* This wins over any layer */
.text { color: red; }
```

### Importing into Layers

```css
@import url('reset.css') layer(reset);
@import url('components.css') layer(components);
```

### Use Cases

```css
/* Third-party CSS in low-priority layer */
@layer vendor, app, overrides;

@import url('tailwind.css') layer(vendor);

@layer app {
  /* Your styles, beat vendor */
}

@layer overrides {
  /* Emergency overrides, beat everything */
}
```

---

## Logical Properties

Replace physical directions (left, right) with logical ones for internationalization:

### Writing Direction Agnostic

```css
.element {
  /* Physical (LTR specific) */
  margin-left: 1rem;
  padding-right: 2rem;
  border-top: 1px solid;
  
  /* Logical (works in RTL too) */
  margin-inline-start: 1rem;
  padding-inline-end: 2rem;
  border-block-start: 1px solid;
}
```

### Mapping Physical to Logical

| Physical | Logical (Horizontal) |
|----------|---------------------|
| `left` | `inline-start` |
| `right` | `inline-end` |
| `top` | `block-start` |
| `bottom` | `block-end` |
| `width` | `inline-size` |
| `height` | `block-size` |

### Shorthand Properties

```css
.element {
  /* margin-left + margin-right */
  margin-inline: 1rem;
  
  /* margin-top + margin-bottom */
  margin-block: 2rem;
  
  /* padding for inline axis */
  padding-inline: 1rem 2rem; /* start end */
  
  /* padding for block axis */
  padding-block: 1rem;
}
```

### Logical Sizing

```css
.container {
  /* Instead of width/height */
  inline-size: 100%;
  max-inline-size: 1200px;
  block-size: auto;
  min-block-size: 100vh;
}
```

### Inset Shorthand

```css
.overlay {
  position: fixed;
  
  /* Physical */
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  
  /* Logical */
  inset: 0;  /* All sides */
  inset-block: 0;  /* top + bottom */
  inset-inline: 1rem;  /* left + right */
}
```

---

## Modern Color

### New Color Functions

```css
.element {
  /* Relative color - adjust existing colors */
  --primary: #6366f1;
  
  /* Make it 20% lighter */
  background: hsl(from var(--primary) h s calc(l + 20%));
  
  /* Make it more saturated */
  background: hsl(from var(--primary) h calc(s + 10%) l);
  
  /* Complementary color */
  background: hsl(from var(--primary) calc(h + 180) s l);
}
```

### Color Mixing

```css
.element {
  /* Mix two colors */
  background: color-mix(in srgb, #6366f1, white 30%);
  
  /* Mix in different color spaces */
  background: color-mix(in oklch, #6366f1, #10b981);
  
  /* Create hover states */
  --color: #6366f1;
  background: var(--color);
}

.element:hover {
  background: color-mix(in srgb, var(--color), black 15%);
}
```

### OKLCH and OKLAB

Perceptually uniform color spaces:

```css
.element {
  /* OKLCH: lightness, chroma, hue */
  background: oklch(70% 0.15 250);
  
  /* Better for color scales */
  --gray-100: oklch(95% 0 0);
  --gray-500: oklch(55% 0 0);
  --gray-900: oklch(15% 0 0);
  
  /* Consistent perceived lightness */
  --red: oklch(65% 0.25 25);
  --green: oklch(65% 0.2 145);
  --blue: oklch(65% 0.2 260);
}
```

### Color Scheme

```css
:root {
  color-scheme: light dark;
}

/* Forces specific scheme */
.light-only {
  color-scheme: light;
}

.dark-only {
  color-scheme: dark;
}
```

---

## Subgrid

Nested grids that align with their parent:

```css
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.grid-item {
  display: grid;
  grid-template-columns: subgrid;
  grid-column: span 3;
}
```

### Card Grid with Aligned Content

```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
}

.card {
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 0.5rem;
}

/* Without subgrid, card content misaligns */
/* With subgrid, all cards' headers, content, footers align */

.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(3, auto);
  gap: 2rem 1rem;
}

.card {
  display: grid;
  grid-row: span 3;
  grid-template-rows: subgrid;
}
```

---

## Scroll-Driven Animations

Animate based on scroll position:

```css
@keyframes fade-in {
  from { opacity: 0; transform: translateY(50px); }
  to { opacity: 1; transform: translateY(0); }
}

.reveal {
  animation: fade-in linear both;
  animation-timeline: view();
  animation-range: entry 0% cover 40%;
}
```

### Scroll Progress Bar

```css
.progress-bar {
  position: fixed;
  top: 0;
  left: 0;
  height: 3px;
  background: #6366f1;
  transform-origin: left;
  animation: grow-progress linear;
  animation-timeline: scroll();
}

@keyframes grow-progress {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}
```

> **Browser support:** Chrome 115+, limited in others. Use with progressive enhancement.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use container queries for components | True component isolation |
| Prefer logical properties | Ready for internationalization |
| Define cascade layers upfront | Clear specificity management |
| Use `:has()` sparingly | Can be performance-intensive |
| Progressive enhancement | Not all browsers support all features |
| Check browser support | Use @supports for fallbacks |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `:has()` performance | Avoid complex `:has()` in frequently updating elements |
| Missing container type | `container-type: inline-size` required |
| Forgetting `&` in nesting | Element selectors need `& element { }` |
| Layer order mistakes | Define layer order explicitly at start |
| Over-nesting | Keep nesting shallow (2-3 levels max) |

---

## Hands-on Exercise

### Your Task

Create a modern card component that:

1. Uses container queries to adapt layout
2. Uses `:has()` to style parent based on content
3. Uses CSS nesting for clean organization
4. Uses logical properties for RTL support
5. Uses modern color mixing for hover states

<details>
<summary>✅ Solution</summary>

```css
/* Define layers */
@layer base, components;

@layer components {
  .card-container {
    container: card / inline-size;
  }

  .card {
    --card-bg: #ffffff;
    --card-border: #e5e7eb;
    --card-text: #1e293b;
    --card-accent: #6366f1;
    
    display: grid;
    gap: 1rem;
    padding: 1rem;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    color: var(--card-text);
    
    /* Nested styles */
    .card-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .card-title {
      font-weight: 600;
      margin: 0;
    }
    
    .card-content {
      color: color-mix(in srgb, var(--card-text), transparent 30%);
    }
    
    .card-footer {
      display: flex;
      gap: 0.5rem;
      padding-block-start: 0.5rem;
      border-block-start: 1px solid var(--card-border);
    }
    
    /* Hover using color-mix */
    &:hover {
      border-color: color-mix(in srgb, var(--card-accent), transparent 50%);
      box-shadow: 0 4px 12px color-mix(in srgb, var(--card-accent), transparent 85%);
    }
    
    /* Has image - adjust layout */
    &:has(.card-image) {
      padding: 0;
      
      .card-image {
        border-radius: 8px 8px 0 0;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        inline-size: 100%;
      }
      
      .card-body {
        padding: 1rem;
      }
    }
    
    /* Has badge - highlight */
    &:has(.badge-featured) {
      border-color: gold;
      border-width: 2px;
    }
    
    /* Container queries */
    @container card (min-width: 400px) {
      grid-template-columns: auto 1fr;
      
      &:has(.card-image) {
        .card-image {
          border-radius: 8px 0 0 8px;
          inline-size: 200px;
          block-size: 100%;
          aspect-ratio: auto;
        }
      }
    }
    
    @container card (min-width: 600px) {
      grid-template-columns: 200px 1fr auto;
      
      .card-actions {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-inline-start: 1rem;
        border-inline-start: 1px solid var(--card-border);
      }
    }
  }

  /* Button inside card */
  .card .button {
    padding-inline: 1rem;
    padding-block: 0.5rem;
    background: var(--card-accent);
    color: white;
    border: none;
    border-radius: 4px;
    
    &:hover {
      background: color-mix(in srgb, var(--card-accent), black 15%);
    }
  }
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
  @layer components {
    .card {
      --card-bg: #1e293b;
      --card-border: #334155;
      --card-text: #e2e8f0;
    }
  }
}
```

```html
<div class="card-container">
  <article class="card">
    <img src="image.jpg" alt="" class="card-image">
    <div class="card-body">
      <header class="card-header">
        <span class="badge-featured">Featured</span>
        <h2 class="card-title">Modern CSS Features</h2>
      </header>
      <p class="card-content">
        Learn about container queries, :has(), nesting, and more.
      </p>
      <footer class="card-footer">
        <button class="button">Learn More</button>
      </footer>
    </div>
    <div class="card-actions">
      <button class="button">Save</button>
      <button class="button">Share</button>
    </div>
  </article>
</div>
```
</details>

---

## Summary

✅ **Container queries** let components respond to their container, not viewport

✅ **`:has()`** selects parents based on children—the long-awaited parent selector

✅ **CSS nesting** brings Sass-like syntax natively to CSS

✅ **Cascade layers** provide explicit control over specificity

✅ **Logical properties** prepare layouts for RTL and vertical writing modes

✅ **Modern color** with `color-mix()` and OKLCH creates better color systems

✅ **Subgrid** aligns nested grids with their parents

---

**Previous:** [Transitions & Animations](./08-transitions-animations.md)

**Next:** [View Transitions](./10-view-transitions.md)

<!-- 
Sources Consulted:
- MDN Container Queries: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_containment/Container_queries
- MDN :has() selector: https://developer.mozilla.org/en-US/docs/Web/CSS/:has
- MDN CSS Nesting: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_nesting
- MDN Cascade Layers: https://developer.mozilla.org/en-US/docs/Learn/CSS/Building_blocks/Cascade_layers
-->
