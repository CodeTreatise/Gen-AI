---
title: "Critical Rendering Path"
---

# Critical Rendering Path

## Introduction

The **critical rendering path** is the sequence of steps the browser takes to convert HTML, CSS, and JavaScript into pixels on screen. Understanding this process helps you optimize for faster first paint.

This lesson covers how browsers render pages and how to minimize blocking resources.

### What We'll Cover

- How browsers render pages
- Render-blocking resources
- Critical CSS
- async and defer for scripts
- Preload, prefetch, preconnect

### Prerequisites

- HTML/CSS/JavaScript fundamentals
- Understanding of HTTP requests
- Basic browser concepts

---

## How Browsers Render Pages

### The Rendering Pipeline

```
HTML Document
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. Parse HTML → DOM Tree                │
│    (Document Object Model)              │
└─────────────────────────────────────────┘
     │
     │ ← CSS files block this step
     ▼
┌─────────────────────────────────────────┐
│ 2. Parse CSS → CSSOM Tree               │
│    (CSS Object Model)                   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 3. Combine → Render Tree                │
│    (Only visible elements)              │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 4. Layout → Calculate positions         │
│    (Where elements go)                  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 5. Paint → Draw pixels                  │
│    (What elements look like)            │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 6. Composite → Layer composition        │
│    (GPU acceleration)                   │
└─────────────────────────────────────────┘
```

### What Blocks Rendering?

| Resource | Blocks? | Why |
|----------|---------|-----|
| **CSS** | Yes | Can't paint without styles |
| **Sync JS** | Yes | May modify DOM/CSSOM |
| **Fonts** | Partially | May hide text until loaded |
| **Images** | No | Paint placeholder, fill in later |

---

## Render-Blocking Resources

### The Problem

```html
<head>
  <!-- These block rendering -->
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="print.css">
  <link rel="stylesheet" href="theme.css">
  <script src="app.js"></script>
</head>
```

Browser must:
1. Download all CSS files
2. Parse all CSS
3. Download and execute JavaScript
4. ONLY THEN can it render anything

### Identifying Blocking Resources

Chrome DevTools:
1. Performance panel → Record page load
2. Look for long "Parse Stylesheet" blocks
3. Check Network panel for render-blocking indicators

Lighthouse flags:
- "Eliminate render-blocking resources"
- "Reduce unused CSS"

---

## Critical CSS

### What Is Critical CSS?

CSS needed to render above-the-fold content:

```html
<head>
  <!-- Critical CSS inlined -->
  <style>
    /* Only what's needed for first view */
    body { font-family: system-ui; margin: 0; }
    .header { background: #333; color: white; padding: 1rem; }
    .hero { padding: 2rem; }
    .hero h1 { font-size: 2.5rem; }
  </style>
  
  <!-- Full CSS loaded async -->
  <link rel="preload" href="styles.css" as="style" onload="this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="styles.css"></noscript>
</head>
```

### Extracting Critical CSS

**Tools:**
- **Critical** (npm package)
- **Critters** (Webpack plugin)
- **PurgeCSS** + inline

```javascript
// Using Critical
const critical = require('critical');

critical.generate({
  base: 'dist/',
  src: 'index.html',
  css: ['dist/styles.css'],
  dimensions: [
    { width: 320, height: 480 },
    { width: 1200, height: 800 }
  ],
  inline: true
});
```

### Async CSS Loading

```html
<!-- Method 1: Preload + onload -->
<link rel="preload" href="styles.css" as="style" onload="this.rel='stylesheet'">

<!-- Method 2: Media query swap -->
<link rel="stylesheet" href="styles.css" media="print" onload="this.media='all'">

<!-- Always include noscript fallback -->
<noscript><link rel="stylesheet" href="styles.css"></noscript>
```

---

## Script Loading

### Default Script Behavior

```html
<!-- Blocks HTML parsing, blocks rendering -->
<script src="app.js"></script>
```

```
HTML Parsing: ───────┐
                     │ (blocked)
Script Download:     ├──────────┐
                     │          │
Script Execute:      │          ├──────┐
                     │          │      │
HTML Parsing:        │          │      └────────→
```

### async Attribute

```html
<!-- Downloads in parallel, executes when ready -->
<script src="analytics.js" async></script>
```

```
HTML Parsing: ──────────────────────────────────→
              ↑
Script Download: ├──────────┐
                           │
Script Execute:            └──────┐
                                  │ (may interrupt parsing)
```

**Use for:** Independent scripts (analytics, ads)

### defer Attribute

```html
<!-- Downloads in parallel, executes after HTML parsed -->
<script src="app.js" defer></script>
```

```
HTML Parsing: ──────────────────────────────────→
              ↑                                  │
Script Download: ├──────────┐                    │
                           │                    │
Script Execute:            └─────────────────────┤
                                                 │ (after DOMContentLoaded)
```

**Use for:** Application code that needs the DOM

### Comparison

| Attribute | Download | Execute | Order | Use Case |
|-----------|----------|---------|-------|----------|
| (none) | Blocking | Immediate | Maintained | Legacy, inline |
| `async` | Parallel | When ready | Not guaranteed | Analytics, ads |
| `defer` | Parallel | After DOM | Maintained | App code |

### Module Scripts

```html
<!-- type="module" is deferred by default -->
<script type="module" src="app.js"></script>

<!-- Can also be async -->
<script type="module" src="analytics.js" async></script>
```

---

## Resource Hints

### Preload

**Load critical resources early:**

```html
<!-- Fonts (critical for LCP) -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>

<!-- Hero image -->
<link rel="preload" href="/hero.webp" as="image">

<!-- Critical script -->
<link rel="preload" href="/critical.js" as="script">
```

### Preconnect

**Establish early connections to origins:**

```html
<!-- Third-party APIs -->
<link rel="preconnect" href="https://api.example.com">

<!-- CDNs -->
<link rel="preconnect" href="https://cdn.example.com">

<!-- Font providers (need both) -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
```

### DNS-Prefetch

**Lightweight preconnect (just DNS):**

```html
<!-- For less critical origins -->
<link rel="dns-prefetch" href="https://analytics.example.com">
```

### Prefetch

**Load resources for next navigation:**

```html
<!-- Likely next page -->
<link rel="prefetch" href="/next-page.html">

<!-- Likely needed script -->
<link rel="prefetch" href="/feature-chunk.js">
```

### Resource Hint Summary

| Hint | Purpose | Priority | When |
|------|---------|----------|------|
| `preload` | Critical current page | High | Load immediately |
| `preconnect` | Reduce connection time | Medium | Before request |
| `dns-prefetch` | Reduce DNS lookup | Low | Before connection |
| `prefetch` | Future navigation | Low | When idle |

---

## Optimizing the Critical Path

### Step 1: Analyze

```bash
# Use Lighthouse
lighthouse https://example.com --view

# Check critical path length
# Goal: Minimize round trips before first paint
```

### Step 2: Minimize Blocking Resources

```html
<head>
  <!-- 1. Preconnect to critical origins -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  
  <!-- 2. Inline critical CSS -->
  <style>/* critical CSS */</style>
  
  <!-- 3. Preload critical resources -->
  <link rel="preload" href="/hero.webp" as="image">
  <link rel="preload" href="/fonts/main.woff2" as="font" crossorigin>
  
  <!-- 4. Async non-critical CSS -->
  <link rel="stylesheet" href="/styles.css" media="print" onload="this.media='all'">
</head>
<body>
  <!-- Content -->
  
  <!-- 5. Defer scripts -->
  <script src="/app.js" defer></script>
</body>
```

### Step 3: Reduce Critical Path Length

| Optimization | Impact |
|--------------|--------|
| Inline critical CSS | -1 round trip |
| Use HTTP/2 | Parallel requests |
| Use CDN | Lower latency |
| Enable compression | Faster downloads |
| Reduce CSS size | Faster parsing |

---

## Server-Side Rendering (SSR)

### CSR vs SSR

**Client-Side Rendering:**
```
1. Download HTML (empty shell)
2. Download JavaScript
3. Execute JavaScript
4. Render content
   ↳ First paint: 2-4 seconds
```

**Server-Side Rendering:**
```
1. Server renders HTML with content
2. Download HTML (full content)
3. First paint immediately
4. Download JavaScript for interactivity
   ↳ First paint: <1 second
```

### SSR Benefits

- Faster First Contentful Paint
- Better SEO
- Works without JavaScript
- Better for slow devices

### Streaming SSR

Modern frameworks support streaming:

```javascript
// React 18 streaming
import { renderToPipeableStream } from 'react-dom/server';

app.get('/', (req, res) => {
  const { pipe } = renderToPipeableStream(<App />, {
    onShellReady() {
      res.setHeader('content-type', 'text/html');
      pipe(res);
    }
  });
});
```

---

## Hands-on Exercise

### Your Task

Optimize this page's critical rendering path:

```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto">
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="components.css">
  <link rel="stylesheet" href="utilities.css">
  <script src="jquery.js"></script>
  <script src="lodash.js"></script>
  <script src="app.js"></script>
</head>
<body>
  <header>Site Header</header>
  <main>
    <img src="hero.jpg">
    <h1>Welcome</h1>
  </main>
</body>
</html>
```

<details>
<summary>✅ Solution</summary>

```html
<!DOCTYPE html>
<html>
<head>
  <!-- Preconnect to external origins -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  
  <!-- Preload critical resources -->
  <link rel="preload" href="hero.webp" as="image" fetchpriority="high">
  
  <!-- Inline critical CSS -->
  <style>
    body { font-family: 'Roboto', system-ui; margin: 0; }
    header { background: #333; color: white; padding: 1rem; }
    main { padding: 2rem; }
    h1 { font-size: 2.5rem; }
  </style>
  
  <!-- Async load full CSS -->
  <link rel="stylesheet" href="styles.css" media="print" onload="this.media='all'">
  
  <!-- Google Fonts with display=swap -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto&display=swap" media="print" onload="this.media='all'">
</head>
<body>
  <header>Site Header</header>
  <main>
    <picture>
      <source srcset="hero.avif" type="image/avif">
      <source srcset="hero.webp" type="image/webp">
      <img src="hero.jpg" alt="Hero" width="1200" height="600" fetchpriority="high">
    </picture>
    <h1>Welcome</h1>
  </main>
  
  <!-- Defer all scripts -->
  <script src="app.js" defer></script>
  <!-- Removed jQuery and Lodash - use native JS or tree-shake -->
</body>
</html>
```
</details>

---

## Summary

✅ **CSS blocks rendering**—inline critical CSS, async the rest
✅ **Scripts block by default**—use `defer` or `async`
✅ **Preload** critical resources (fonts, LCP image)
✅ **Preconnect** to third-party origins early
✅ **Minimize critical path length**—fewer round trips
✅ Consider **SSR** for faster first paint

**Next:** [Performance Measurement](./04-performance-measurement.md)

---

## Further Reading

- [web.dev Critical Rendering Path](https://web.dev/critical-rendering-path/)
- [Render Blocking Resources](https://web.dev/render-blocking-resources/)
- [Resource Hints](https://web.dev/preconnect-and-dns-prefetch/)

<!-- 
Sources Consulted:
- web.dev CRP: https://web.dev/critical-rendering-path/
- Chrome DevTools: https://developer.chrome.com/docs/devtools/
-->
