---
title: "Media Queries"
---

# Media Queries

## Introduction

Media queries are CSS conditions that apply styles based on device characteristics—screen width, orientation, color scheme preference, and more. They're the backbone of responsive design and progressive enhancement.

While intrinsic design reduces the need for media queries, you'll still use them for significant layout changes, user preference detection, and feature-based styling.

### What We'll Cover

- Media query syntax and operators
- Common breakpoints and patterns
- Feature queries with `@supports`
- User preference queries (dark mode, reduced motion)
- Print stylesheets

### Prerequisites

- CSS layout fundamentals (Flexbox, Grid)
- Understanding of responsive design principles

---

## Basic Syntax

### `@media` Rule

```css
@media (condition) {
  /* Styles applied when condition is true */
}
```

### Width-Based Queries

```css
/* Minimum width (mobile-first) */
@media (min-width: 768px) {
  .sidebar {
    display: block;
  }
}

/* Maximum width (desktop-first) */
@media (max-width: 767px) {
  .sidebar {
    display: none;
  }
}

/* Exact range */
@media (min-width: 768px) and (max-width: 1023px) {
  .container {
    padding: 2rem;
  }
}
```

### Modern Range Syntax

CSS Media Queries Level 4 introduced comparison operators:

```css
/* Instead of min-width: 768px */
@media (width >= 768px) {
  .sidebar { display: block; }
}

/* Range */
@media (768px <= width < 1024px) {
  .container { padding: 2rem; }
}

/* Less than */
@media (width < 768px) {
  .mobile-menu { display: block; }
}
```

> **Browser support:** Range syntax works in Chrome 104+, Firefox 103+, Safari 16.4+. Use traditional syntax for older browser support.

---

## Combining Conditions

### `and` - All Must Match

```css
@media (min-width: 768px) and (orientation: landscape) {
  .gallery {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

### `or` (comma) - Any Can Match

```css
@media (max-width: 767px), (orientation: portrait) {
  .sidebar {
    position: fixed;
  }
}
```

### `not` - Negation

```css
@media not print {
  .no-print {
    display: block;
  }
}

/* Negates entire query */
@media not (min-width: 768px) {
  /* Same as max-width: 767px */
}
```

### `only` - Hide from Old Browsers

```css
@media only screen and (min-width: 768px) {
  /* Old browsers that don't understand 'only' ignore the rule */
}
```

---

## Media Types

```css
@media screen { /* Computer screens, tablets, phones */ }
@media print { /* Print preview and printed pages */ }
@media all { /* All devices (default) */ }
```

```css
/* Screen-only styles */
@media screen and (min-width: 768px) {
  .navigation { display: flex; }
}

/* Print-only styles */
@media print {
  .navigation { display: none; }
}
```

---

## Common Media Features

### Dimensions

| Feature | Description |
|---------|-------------|
| `width` | Viewport width |
| `height` | Viewport height |
| `aspect-ratio` | Width-to-height ratio |
| `orientation` | `portrait` or `landscape` |

```css
@media (orientation: landscape) {
  .hero {
    height: 100vh;
  }
}

@media (aspect-ratio: 16/9) {
  .video-container {
    padding: 0;
  }
}
```

### Display Capabilities

| Feature | Description |
|---------|-------------|
| `hover` | Can the device hover? |
| `pointer` | Pointer precision |
| `any-hover` | Any input can hover? |
| `any-pointer` | Any input precision |

```css
/* Devices with hover capability */
@media (hover: hover) {
  .card:hover {
    transform: translateY(-4px);
  }
}

/* No hover (touch devices) */
@media (hover: none) {
  .tooltip-trigger:focus .tooltip {
    display: block;
  }
}

/* Fine pointer (mouse) */
@media (pointer: fine) {
  .slider-thumb {
    width: 16px;
  }
}

/* Coarse pointer (touch) */
@media (pointer: coarse) {
  .slider-thumb {
    width: 44px; /* Larger touch target */
  }
}
```

---

## User Preference Queries

### Color Scheme (Dark Mode)

```css
/* Light mode (default) */
:root {
  --bg-color: #ffffff;
  --text-color: #1e293b;
}

/* Dark mode preference */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #0f172a;
    --text-color: #e2e8f0;
  }
}

body {
  background: var(--bg-color);
  color: var(--text-color);
}
```

### Reduced Motion

Respect users who are sensitive to motion:

```css
/* Default: with animations */
.fade-in {
  animation: fadeIn 0.3s ease-out;
}

.slide-in {
  animation: slideIn 0.5s ease-out;
}

/* Remove animations for users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

> **Important:** Don't just `animation: none`—set a very short duration so JavaScript animation callbacks still fire.

### Contrast Preferences

```css
@media (prefers-contrast: more) {
  :root {
    --text-color: #000000;
    --border-color: #000000;
  }
  
  .button {
    border: 2px solid currentColor;
  }
}

@media (prefers-contrast: less) {
  :root {
    --text-color: #4a4a4a;
  }
}
```

### Reduced Transparency

```css
@media (prefers-reduced-transparency: reduce) {
  .modal-backdrop {
    background: #000000; /* Solid instead of semi-transparent */
  }
  
  .glass-panel {
    backdrop-filter: none;
    background: var(--bg-color);
  }
}
```

### Color Gamut

```css
@media (color-gamut: p3) {
  :root {
    --brand-color: color(display-p3 1 0.2 0.1);
  }
}

@media (color-gamut: srgb) {
  :root {
    --brand-color: #ff3b30;
  }
}
```

---

## Feature Queries (`@supports`)

Test for CSS feature support before using:

```css
/* Basic support check */
@supports (display: grid) {
  .container {
    display: grid;
  }
}

/* Fallback for no support */
.container {
  display: flex;
  flex-wrap: wrap;
}

@supports (display: grid) {
  .container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
  }
}
```

### Combining Conditions

```css
/* Both must be supported */
@supports (display: grid) and (gap: 1rem) {
  .grid { display: grid; gap: 1rem; }
}

/* Either supported */
@supports (backdrop-filter: blur(10px)) or (-webkit-backdrop-filter: blur(10px)) {
  .glass {
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
}

/* Not supported */
@supports not (aspect-ratio: 1) {
  .square::before {
    content: '';
    padding-top: 100%;
    float: left;
  }
}
```

### Common Feature Queries

```css
/* Container queries */
@supports (container-type: inline-size) {
  .card-wrapper {
    container-type: inline-size;
  }
}

/* :has() selector */
@supports selector(:has(*)) {
  .form:has(:invalid) {
    border-color: red;
  }
}

/* Subgrid */
@supports (grid-template-columns: subgrid) {
  .grid-item {
    display: grid;
    grid-template-columns: subgrid;
  }
}
```

---

## Print Stylesheets

Optimize for printing:

```css
@media print {
  /* Hide non-essential elements */
  nav, footer, .sidebar, .ads, .comments {
    display: none !important;
  }
  
  /* Ensure content is visible */
  body {
    color: black;
    background: white;
  }
  
  /* Show full URLs for links */
  a[href^="http"]::after {
    content: " (" attr(href) ")";
    font-size: 0.8em;
    color: #666;
  }
  
  /* Prevent page breaks inside elements */
  h1, h2, h3, img, figure {
    break-inside: avoid;
  }
  
  /* Always break before major sections */
  .chapter {
    break-before: page;
  }
  
  /* Avoid orphans and widows */
  p {
    orphans: 3;
    widows: 3;
  }
}
```

### Print-Specific Units

```css
@media print {
  @page {
    size: A4;
    margin: 2cm;
  }
  
  body {
    font-size: 12pt;
  }
}
```

---

## Organizing Media Queries

### Approach 1: Grouped at End

```css
/* Base styles */
.card { padding: 1rem; }

/* All media queries together */
@media (min-width: 768px) {
  .card { padding: 2rem; }
}

@media (min-width: 1024px) {
  .card { padding: 3rem; }
}
```

### Approach 2: Component-Based

```css
/* Card component with its queries */
.card {
  padding: 1rem;
}

@media (min-width: 768px) {
  .card { padding: 2rem; }
}

/* Navigation with its queries */
.nav {
  flex-direction: column;
}

@media (min-width: 768px) {
  .nav { flex-direction: row; }
}
```

### Using CSS Custom Properties

Reduce repetition:

```css
:root {
  --container-padding: 1rem;
  --grid-columns: 1;
}

@media (min-width: 768px) {
  :root {
    --container-padding: 2rem;
    --grid-columns: 2;
  }
}

@media (min-width: 1024px) {
  :root {
    --container-padding: 3rem;
    --grid-columns: 3;
  }
}

.container {
  padding: var(--container-padding);
}

.grid {
  grid-template-columns: repeat(var(--grid-columns), 1fr);
}
```

---

## Mobile-First Breakpoint System

```css
/* Base: mobile (0-479px) */
.container {
  padding: 1rem;
}

/* Small: 480px+ */
@media (min-width: 480px) {
  .container {
    padding: 1.5rem;
  }
}

/* Medium: 768px+ */
@media (min-width: 768px) {
  .container {
    max-width: 720px;
    margin: 0 auto;
  }
}

/* Large: 1024px+ */
@media (min-width: 1024px) {
  .container {
    max-width: 960px;
  }
}

/* Extra large: 1280px+ */
@media (min-width: 1280px) {
  .container {
    max-width: 1200px;
  }
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Mobile-first (`min-width`) | Smaller base CSS, progressive enhancement |
| Use meaningful breakpoints | Based on content, not devices |
| Respect user preferences | Dark mode, reduced motion, contrast |
| Test real devices | Emulators miss real-world issues |
| Don't over-query | Prefer intrinsic design when possible |
| Keep queries near components | Easier maintenance |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Overlapping breakpoints | Use `min-width: 768px` and `max-width: 767px` |
| Ignoring reduced motion | Always provide `prefers-reduced-motion` override |
| Device-specific breakpoints | Use content-based breakpoints |
| Forgetting print styles | Add basic print stylesheet |
| Not testing dark mode | Test both color schemes |
| Assuming mouse hover | Use `@media (hover: hover)` |

---

## Hands-on Exercise

### Your Task

Create a responsive, accessible theme system:

1. Light and dark mode based on system preference
2. Manual theme toggle override
3. Reduced motion support
4. High contrast support
5. Print-friendly version

### Requirements

1. CSS custom properties for theme values
2. `prefers-color-scheme` for default
3. `[data-theme]` attribute for manual override
4. Print styles that remove non-essential content

<details>
<summary>✅ Solution</summary>

```css
/* Base theme (light) */
:root {
  --color-bg: #ffffff;
  --color-surface: #f8fafc;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-primary: #6366f1;
  --color-border: #e2e8f0;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  --transition-speed: 0.2s;
}

/* Dark mode - system preference */
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #0f172a;
    --color-surface: #1e293b;
    --color-text: #e2e8f0;
    --color-text-muted: #94a3b8;
    --color-primary: #818cf8;
    --color-border: #334155;
    --shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }
}

/* Manual overrides via data attribute */
[data-theme="light"] {
  --color-bg: #ffffff;
  --color-surface: #f8fafc;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-primary: #6366f1;
  --color-border: #e2e8f0;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

[data-theme="dark"] {
  --color-bg: #0f172a;
  --color-surface: #1e293b;
  --color-text: #e2e8f0;
  --color-text-muted: #94a3b8;
  --color-primary: #818cf8;
  --color-border: #334155;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

/* High contrast support */
@media (prefers-contrast: more) {
  :root {
    --color-text: #000000;
    --color-border: #000000;
  }
  
  [data-theme="dark"],
  @media (prefers-color-scheme: dark) {
    :root {
      --color-text: #ffffff;
      --color-border: #ffffff;
    }
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  :root {
    --transition-speed: 0.01ms;
  }
  
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* Apply theme */
body {
  background: var(--color-bg);
  color: var(--color-text);
  transition: background var(--transition-speed), 
              color var(--transition-speed);
}

.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow);
}

/* Print styles */
@media print {
  :root {
    --color-bg: white;
    --color-text: black;
  }
  
  /* Hide interactive elements */
  .theme-toggle,
  nav,
  footer,
  .sidebar {
    display: none !important;
  }
  
  /* Ensure readability */
  body {
    font-size: 12pt;
    line-height: 1.5;
  }
  
  a {
    color: inherit;
    text-decoration: underline;
  }
  
  /* Show URLs */
  a[href^="http"]::after {
    content: " (" attr(href) ")";
    font-size: 0.8em;
  }
}
```

```html
<html data-theme=""><!-- Empty uses system preference -->
<head>
  <style>/* Above CSS */</style>
</head>
<body>
  <button class="theme-toggle" onclick="toggleTheme()">
    Toggle Theme
  </button>
  
  <script>
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.dataset.theme;
      
      if (current === 'dark') {
        html.dataset.theme = 'light';
      } else if (current === 'light') {
        html.dataset.theme = 'dark';
      } else {
        // First toggle: detect current and flip
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        html.dataset.theme = isDark ? 'light' : 'dark';
      }
      
      localStorage.setItem('theme', html.dataset.theme);
    }
    
    // Restore saved preference
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.dataset.theme = saved;
  </script>
</body>
</html>
```
</details>

---

## Summary

✅ **Media queries** apply styles conditionally based on device or user characteristics

✅ **Mobile-first** (`min-width`) is preferred over desktop-first (`max-width`)

✅ **User preference queries** respect dark mode, reduced motion, and contrast settings

✅ **`@supports`** enables progressive enhancement based on feature support

✅ **Print styles** ensure content remains readable when printed

✅ **Modern range syntax** (`width >= 768px`) is cleaner but requires recent browsers

---

**Previous:** [Responsive Design](./05-responsive-design.md)

**Next:** [CSS Variables](./07-css-variables.md)

<!-- 
Sources Consulted:
- MDN Media Queries: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_media_queries
- MDN prefers-color-scheme: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme
- MDN prefers-reduced-motion: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion
- MDN @supports: https://developer.mozilla.org/en-US/docs/Web/CSS/@supports
-->
