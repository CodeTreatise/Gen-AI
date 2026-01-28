---
title: "Responsive Design"
---

# Responsive Design

## Introduction

Responsive design ensures your interface works across all screen sizes—from mobile phones to ultra-wide monitors. Rather than building separate sites for each device, we build one flexible layout that adapts.

For AI applications, this is critical: users access chat interfaces from phones while commuting, tablets while researching, and desktops while working.

### What We'll Cover

- The mobile-first approach and why it matters
- Fluid layouts with relative units
- Flexible typography and spacing
- Intrinsic design techniques
- Container queries for component-level responsiveness

### Prerequisites

- CSS Flexbox and Grid basics
- Understanding of viewport and units

---

## The Mobile-First Approach

Mobile-first means starting with styles for small screens, then adding complexity for larger ones:

```css
/* Base styles: mobile */
.container {
  padding: 1rem;
}

/* Tablet and up */
@media (min-width: 768px) {
  .container {
    padding: 2rem;
  }
}

/* Desktop and up */
@media (min-width: 1024px) {
  .container {
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

### Why Mobile-First?

| Benefit | Explanation |
|---------|-------------|
| Performance | Mobile loads only needed CSS |
| Progressive enhancement | Core experience works everywhere |
| Simpler base styles | Mobile layouts are typically simpler |
| Forces prioritization | Limited space means focus on essentials |

### Desktop-First (Avoid)

```css
/* Starting with desktop styles requires overriding them on mobile */
.container {
  display: grid;
  grid-template-columns: 250px 1fr 250px;
}

@media (max-width: 768px) {
  .container {
    display: block; /* Override everything */
  }
}
```

This leads to more overrides and larger CSS for mobile users.

---

## Viewport Meta Tag

Essential for mobile rendering:

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

**What each part does:**
- `width=device-width`: Layout width matches device screen
- `initial-scale=1`: No initial zoom

Without this, mobile browsers render at ~980px and scale down.

---

## Relative Units

Avoid fixed pixel values for flexible layouts:

### Percentage

Relative to parent element:

```css
.container {
  width: 100%;
  max-width: 1200px;
}

.sidebar {
  width: 25%;
}
```

### `em` and `rem`

- `em`: Relative to parent font-size
- `rem`: Relative to root (`<html>`) font-size

```css
html {
  font-size: 16px; /* Browser default */
}

.heading {
  font-size: 2rem;    /* 32px */
  margin-bottom: 1em; /* 32px (relative to this element) */
}

.button {
  padding: 0.5rem 1rem;
  font-size: 1rem;
}
```

**When to use which:**
- `rem` for consistent sizing regardless of context (spacing, font sizes)
- `em` when you want scaling relative to the element (padding relative to font size)

### Viewport Units

Relative to viewport dimensions:

| Unit | Description |
|------|-------------|
| `vw` | 1% of viewport width |
| `vh` | 1% of viewport height |
| `vmin` | 1% of smaller dimension |
| `vmax` | 1% of larger dimension |
| `dvh` | Dynamic viewport height (accounts for mobile browser UI) |
| `svh` | Small viewport height (mobile UI visible) |
| `lvh` | Large viewport height (mobile UI hidden) |

```css
.hero {
  height: 100vh;      /* Full viewport height */
  height: 100dvh;     /* Accounts for mobile browser chrome */
}

.full-bleed {
  width: 100vw;
  margin-left: calc(50% - 50vw);
}
```

> **Mobile tip:** Use `dvh` instead of `vh` to handle mobile browser address bar showing/hiding.

---

## Fluid Typography

### Using `clamp()`

Responsive font sizes without media queries:

```css
.heading {
  /* min, preferred, max */
  font-size: clamp(1.5rem, 4vw, 3rem);
}

.body-text {
  font-size: clamp(1rem, 2vw + 0.5rem, 1.25rem);
}
```

**How it works:**
1. Use the preferred value (middle)
2. But never smaller than the minimum
3. And never larger than the maximum

### Type Scale

Create a consistent type system:

```css
:root {
  --text-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
  --text-sm: clamp(0.875rem, 0.8rem + 0.375vw, 1rem);
  --text-base: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);
  --text-lg: clamp(1.125rem, 1rem + 0.625vw, 1.25rem);
  --text-xl: clamp(1.25rem, 1rem + 1.25vw, 1.75rem);
  --text-2xl: clamp(1.5rem, 1rem + 2.5vw, 2.5rem);
  --text-3xl: clamp(2rem, 1rem + 5vw, 4rem);
}

h1 { font-size: var(--text-3xl); }
h2 { font-size: var(--text-2xl); }
p  { font-size: var(--text-base); }
```

---

## Fluid Spacing

Apply the same fluid approach to spacing:

```css
:root {
  --space-xs: clamp(0.25rem, 0.5vw, 0.5rem);
  --space-sm: clamp(0.5rem, 1vw, 0.75rem);
  --space-md: clamp(1rem, 2vw, 1.5rem);
  --space-lg: clamp(1.5rem, 3vw, 2.5rem);
  --space-xl: clamp(2rem, 5vw, 4rem);
}

.section {
  padding: var(--space-xl) var(--space-md);
}

.card {
  padding: var(--space-md);
  margin-bottom: var(--space-lg);
}
```

---

## Intrinsic Design

Modern CSS properties that adapt without media queries:

### `min()`, `max()`, `clamp()`

```css
.container {
  /* Minimum of 90% width or 1200px */
  width: min(90%, 1200px);
  
  /* At least 300px, but prefer 50% */
  width: max(300px, 50%);
  
  /* Between 300px and 1200px, prefer 90% */
  width: clamp(300px, 90%, 1200px);
}
```

### Flexible Images

```css
img {
  max-width: 100%;
  height: auto;
}
```

### `aspect-ratio`

Maintain proportions:

```css
.video-container {
  aspect-ratio: 16 / 9;
  width: 100%;
}

.square-image {
  aspect-ratio: 1;
  object-fit: cover;
}
```

### Responsive Grid Without Media Queries

```css
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
  gap: 1rem;
}
```

The `min(300px, 100%)` ensures columns work even when the container is narrower than 300px.

---

## Container Queries

Style components based on their container size, not viewport:

```css
/* Define a containment context */
.card-container {
  container-type: inline-size;
  container-name: card;
}

/* Or shorthand */
.card-container {
  container: card / inline-size;
}

/* Style based on container width */
@container card (min-width: 400px) {
  .card {
    display: flex;
    gap: 1rem;
  }
  
  .card-image {
    width: 40%;
  }
}

@container card (max-width: 399px) {
  .card {
    display: block;
  }
  
  .card-image {
    width: 100%;
  }
}
```

### Container Query Units

```css
.card-title {
  font-size: clamp(1rem, 5cqi, 1.5rem);
}
```

| Unit | Description |
|------|-------------|
| `cqi` | 1% of container's inline size |
| `cqb` | 1% of container's block size |
| `cqw` | 1% of container's width |
| `cqh` | 1% of container's height |
| `cqmin` | Smaller of cqi/cqb |
| `cqmax` | Larger of cqi/cqb |

### Container Queries vs Media Queries

| Feature | Media Queries | Container Queries |
|---------|---------------|-------------------|
| Based on | Viewport | Container |
| Best for | Page layout | Components |
| Reusability | Limited | High |
| Browser support | Universal | Modern (2023+) |

---

## Responsive Images

### `srcset` for Resolution

```html
<img 
  src="photo-800.jpg"
  srcset="photo-400.jpg 400w,
          photo-800.jpg 800w,
          photo-1200.jpg 1200w"
  sizes="(max-width: 600px) 100vw,
         (max-width: 1200px) 50vw,
         800px"
  alt="Description">
```

### Art Direction with `<picture>`

```html
<picture>
  <source media="(min-width: 1024px)" srcset="hero-wide.jpg">
  <source media="(min-width: 768px)" srcset="hero-medium.jpg">
  <img src="hero-mobile.jpg" alt="Hero image">
</picture>
```

### Modern Formats

```html
<picture>
  <source type="image/avif" srcset="photo.avif">
  <source type="image/webp" srcset="photo.webp">
  <img src="photo.jpg" alt="Photo">
</picture>
```

---

## Responsive Patterns

### Stack to Horizontal

```css
.flex-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

@media (min-width: 768px) {
  .flex-container {
    flex-direction: row;
  }
}
```

### Sidebar Layout

```css
.layout {
  display: grid;
  grid-template-columns: 1fr;
}

@media (min-width: 768px) {
  .layout {
    grid-template-columns: 250px 1fr;
  }
}
```

### Intrinsic Sidebar (No Media Query)

```css
.layout {
  display: flex;
  flex-wrap: wrap;
}

.sidebar {
  flex: 1 1 200px; /* Grow, shrink, min 200px */
}

.main {
  flex: 999 1 400px; /* Grows much more, min 400px */
}
```

### Responsive Navigation

```css
.nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.nav-links {
  display: flex;
  gap: 1rem;
}

@media (max-width: 768px) {
  .nav-links {
    flex-basis: 100%;
    flex-direction: column;
  }
}
```

---

## Touch-Friendly Design

### Minimum Touch Target Size

```css
.button, .link {
  min-height: 44px;
  min-width: 44px;
  padding: 0.75rem 1rem;
}
```

WCAG recommends 44×44 pixels minimum for touch targets.

### Increased Spacing on Touch

```css
@media (hover: none) {
  .menu-item {
    padding: 1rem;
  }
}
```

### Hover Alternatives

```css
.tooltip-trigger {
  position: relative;
}

/* Hover for mouse users */
@media (hover: hover) {
  .tooltip-trigger:hover .tooltip {
    opacity: 1;
  }
}

/* Touch users need click/focus */
.tooltip-trigger:focus .tooltip {
  opacity: 1;
}
```

---

## Common Breakpoints

There's no universal standard, but these are common:

| Name | Width | Typical Devices |
|------|-------|-----------------|
| xs | 0-479px | Small phones |
| sm | 480-639px | Large phones |
| md | 640-767px | Tablets (portrait) |
| lg | 768-1023px | Tablets (landscape) |
| xl | 1024-1279px | Laptops |
| 2xl | 1280px+ | Desktops |

```css
/* Custom property approach */
:root {
  --breakpoint-sm: 480px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
}
```

> **Tip:** Choose breakpoints based on your content, not devices. When does your layout break? That's your breakpoint.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Mobile-first | Better performance, progressive enhancement |
| Use `rem` for fonts | Respects user preferences |
| Use `clamp()` for fluid sizing | Fewer media queries |
| Container queries for components | Reusable across layouts |
| Test on real devices | Simulators miss touch and performance issues |
| Avoid fixed heights | Content varies by language, font size |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `100vh` on mobile | Use `100dvh` or JavaScript solution |
| Fixed-width containers | Use `max-width` with percentage/vw |
| Hover-only interactions | Add focus/click alternatives |
| Tiny touch targets | Minimum 44×44px |
| Too many breakpoints | Use intrinsic design when possible |
| Ignoring landscape mobile | Test both orientations |

---

## Hands-on Exercise

### Your Task

Create a responsive hero section with:

1. Full viewport height on all devices (mobile-safe)
2. Centered content that scales with viewport
3. Fluid heading that grows from mobile to desktop
4. CTA button with proper touch target size
5. Background image that adapts to screen

### Requirements

1. No fixed pixel values except for minimums
2. Mobile-first approach
3. Works on both portrait and landscape orientations
4. Smooth scaling without jarring breakpoints

<details>
<summary>✅ Solution</summary>

```css
.hero {
  /* Full viewport, mobile-safe */
  min-height: 100dvh;
  
  /* Center content */
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  
  /* Fluid padding */
  padding: clamp(1rem, 5vw, 4rem);
  
  /* Background */
  background-image: url('hero-bg.jpg');
  background-size: cover;
  background-position: center;
  
  /* Text readable over image */
  color: white;
  position: relative;
}

/* Overlay for readability */
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  z-index: 0;
}

.hero > * {
  position: relative;
  z-index: 1;
}

.hero-title {
  /* Fluid typography */
  font-size: clamp(2rem, 8vw, 5rem);
  line-height: 1.1;
  margin-bottom: clamp(0.5rem, 2vw, 1rem);
  max-width: 20ch;
}

.hero-subtitle {
  font-size: clamp(1rem, 2.5vw, 1.5rem);
  margin-bottom: clamp(1.5rem, 4vw, 3rem);
  max-width: 50ch;
  opacity: 0.9;
}

.hero-cta {
  /* Minimum touch target */
  min-height: 44px;
  padding: clamp(0.75rem, 2vw, 1rem) clamp(1.5rem, 4vw, 2.5rem);
  
  font-size: clamp(1rem, 2vw, 1.25rem);
  font-weight: 600;
  
  background: white;
  color: #1e293b;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  
  transition: transform 0.2s, box-shadow 0.2s;
}

.hero-cta:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Landscape mobile: reduce vertical padding */
@media (orientation: landscape) and (max-height: 500px) {
  .hero {
    min-height: auto;
    padding-block: 2rem;
  }
}
```

```html
<section class="hero">
  <h1 class="hero-title">Build AI-Powered Experiences</h1>
  <p class="hero-subtitle">
    Learn to integrate large language models into modern web applications
  </p>
  <button class="hero-cta">Get Started Free</button>
</section>
```
</details>

---

## Summary

✅ **Mobile-first** approach means base styles for mobile, enhance for larger screens

✅ **Relative units** (`rem`, `%`, `vw`) create flexible layouts

✅ **`clamp()`** enables fluid typography and spacing without media queries

✅ **Container queries** style components based on container size, not viewport

✅ **Intrinsic design** uses CSS features like `auto-fit`, `minmax()`, and `flex-wrap` to adapt naturally

✅ **Touch-friendly** means 44×44px minimum targets and hover alternatives

---

**Previous:** [Grid Layout](./04-grid-layout.md)

**Next:** [Media Queries](./06-media-queries.md)

<!-- 
Sources Consulted:
- MDN Responsive Design: https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design
- MDN Container Queries: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_containment/Container_queries
- MDN clamp(): https://developer.mozilla.org/en-US/docs/Web/CSS/clamp
- MDN Viewport units: https://developer.mozilla.org/en-US/docs/Web/CSS/length#viewport-percentage_lengths
-->
