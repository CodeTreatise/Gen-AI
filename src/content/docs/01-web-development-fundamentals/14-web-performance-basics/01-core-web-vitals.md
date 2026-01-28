---
title: "Core Web Vitals"
---

# Core Web Vitals

## Introduction

**Core Web Vitals** are Google's standardized metrics for measuring user experience. They focus on loading, interactivity, and visual stability. Good Core Web Vitals improve search rankings and user satisfaction.

This lesson covers what Core Web Vitals measure, how to optimize for them, and how to set performance budgets.

### What We'll Cover

- Largest Contentful Paint (LCP)
- Interaction to Next Paint (INP)
- Cumulative Layout Shift (CLS)
- Measuring with web-vitals library
- Performance budgets

### Prerequisites

- Understanding of browser rendering
- Basic performance concepts
- DevTools familiarity

---

## The Three Core Web Vitals

### LCP: Largest Contentful Paint

**Measures:** How long until the largest visible element loads.

| Rating | Value |
|--------|-------|
| 🟢 Good | ≤ 2.5 seconds |
| 🟡 Needs Improvement | 2.5 - 4 seconds |
| 🔴 Poor | > 4 seconds |

**What counts as LCP:**
- `<img>` elements
- `<video>` poster images
- Elements with `background-image`
- Block-level text elements

**Common LCP issues:**
- Slow server response
- Render-blocking resources
- Slow resource load times
- Client-side rendering delays

### INP: Interaction to Next Paint

**Measures:** Responsiveness to user interactions (clicks, taps, key presses).

| Rating | Value |
|--------|-------|
| 🟢 Good | ≤ 200 ms |
| 🟡 Needs Improvement | 200 - 500 ms |
| 🔴 Poor | > 500 ms |

**INP replaced FID (First Input Delay)** in March 2024.

**What INP captures:**
- All interaction latency, not just first
- Time from input to visual update
- The 98th percentile of interactions

**Common INP issues:**
- Long JavaScript tasks
- Heavy event handlers
- Excessive DOM size
- Main thread blocking

### CLS: Cumulative Layout Shift

**Measures:** How much the page layout shifts unexpectedly.

| Rating | Value |
|--------|-------|
| 🟢 Good | ≤ 0.1 |
| 🟡 Needs Improvement | 0.1 - 0.25 |
| 🔴 Poor | > 0.25 |

**What causes CLS:**
- Images without dimensions
- Ads/embeds without reserved space
- Dynamically injected content
- Web fonts causing FOUT/FOIT

---

## Optimizing LCP

### 1. Optimize the LCP Element

```html
<!-- Preload the LCP image -->
<link rel="preload" href="hero.webp" as="image" fetchpriority="high">

<!-- Use responsive images -->
<img 
  src="hero.webp"
  srcset="hero-480.webp 480w, hero-800.webp 800w, hero-1200.webp 1200w"
  sizes="(max-width: 600px) 480px, 800px"
  alt="Hero image"
  fetchpriority="high"
  loading="eager">
```

### 2. Reduce Server Response Time

```
Target TTFB: < 800ms

Improvements:
- Use a CDN
- Cache at the edge
- Optimize database queries
- Use HTTP/2 or HTTP/3
```

### 3. Remove Render-Blocking Resources

```html
<!-- Inline critical CSS -->
<style>
  /* Critical above-the-fold CSS */
  .hero { ... }
</style>

<!-- Defer non-critical CSS -->
<link rel="preload" href="styles.css" as="style" onload="this.rel='stylesheet'">

<!-- Async/defer JavaScript -->
<script src="app.js" defer></script>
```

### 4. Optimize Images

```html
<!-- Use modern formats -->
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="...">
</picture>
```

---

## Optimizing INP

### 1. Break Up Long Tasks

```javascript
// ❌ Long blocking task
function processItems(items) {
  items.forEach(item => heavyOperation(item));
}

// ✅ Yield to main thread
async function processItems(items) {
  for (const item of items) {
    heavyOperation(item);
    
    // Yield every 50ms
    await scheduler.yield();  // Or use setTimeout
  }
}

// Fallback for older browsers
function yieldToMain() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
```

### 2. Optimize Event Handlers

```javascript
// ❌ Heavy work in handler
button.addEventListener('click', () => {
  processLargeDataset();  // Blocks rendering
  updateUI();
});

// ✅ Defer heavy work
button.addEventListener('click', () => {
  updateUI();  // Immediate feedback
  requestIdleCallback(() => processLargeDataset());
});
```

### 3. Reduce DOM Size

| DOM Size | Impact |
|----------|--------|
| < 800 nodes | Good |
| 800-1,400 nodes | Moderate |
| > 1,400 nodes | Poor |

```javascript
// Virtualize long lists
// Only render visible items + buffer
```

### 4. Use Web Workers

```javascript
// Move heavy computation off main thread
const worker = new Worker('processor.js');

worker.postMessage(largeData);
worker.onmessage = (e) => {
  updateUI(e.data);  // Only UI updates on main thread
};
```

---

## Optimizing CLS

### 1. Set Image Dimensions

```html
<!-- ✅ Always include width and height -->
<img src="photo.jpg" width="800" height="600" alt="...">

<!-- Or use aspect-ratio CSS -->
<style>
  .image-container {
    aspect-ratio: 4 / 3;
  }
</style>
```

### 2. Reserve Space for Dynamic Content

```html
<!-- Reserve space for ads -->
<div class="ad-container" style="min-height: 250px;">
  <!-- Ad loads here -->
</div>

<!-- Reserve space for embeds -->
<div class="video-container" style="aspect-ratio: 16/9;">
  <iframe src="..."></iframe>
</div>
```

### 3. Avoid Inserting Content Above Existing Content

```javascript
// ❌ Inserts at top, pushing content down
container.prepend(newElement);

// ✅ Insert at end or use transform animations
container.append(newElement);
```

### 4. Optimize Web Fonts

```css
/* Prevent invisible text (FOIT) */
@font-face {
  font-family: 'Custom Font';
  src: url('font.woff2') format('woff2');
  font-display: swap;  /* Show fallback immediately */
}
```

```html
<!-- Preload critical fonts -->
<link rel="preload" href="font.woff2" as="font" type="font/woff2" crossorigin>
```

---

## Measuring Core Web Vitals

### web-vitals Library

```bash
npm install web-vitals
```

```javascript
import { onLCP, onINP, onCLS } from 'web-vitals';

function sendToAnalytics(metric) {
  console.log(metric.name, metric.value);
  // Send to your analytics service
}

onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
```

### Chrome DevTools

1. Performance panel → Record page load
2. Look for Core Web Vitals markers
3. Timings section shows LCP, CLS

### PageSpeed Insights

Free tool from Google:
1. Visit [pagespeed.web.dev](https://pagespeed.web.dev/)
2. Enter URL
3. View field data (real users) and lab data (simulated)

### Chrome User Experience Report (CrUX)

Real-world data from Chrome users:
- Available in PageSpeed Insights
- BigQuery dataset
- CrUX API

---

## Performance Budgets

### What Is a Performance Budget?

Limits on metrics to maintain performance:

```json
{
  "budgets": [
    {
      "resourceType": "script",
      "budget": 200
    },
    {
      "resourceType": "total",
      "budget": 500
    },
    {
      "timings": [
        { "metric": "lcp", "budget": 2500 },
        { "metric": "cls", "budget": 0.1 },
        { "metric": "inp", "budget": 200 }
      ]
    }
  ]
}
```

### Setting Budgets

| Resource | Suggested Budget |
|----------|------------------|
| Total page weight | < 500 KB |
| JavaScript | < 200 KB |
| Images | < 200 KB |
| CSS | < 50 KB |
| Fonts | < 100 KB |

### Enforcing Budgets

**Lighthouse CI:**
```bash
npm install -g @lhci/cli

lhci autorun --config=lighthouserc.js
```

**Webpack Bundle Analyzer:**
```javascript
// webpack.config.js
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = {
  plugins: [new BundleAnalyzerPlugin()]
};
```

**bundlesize:**
```json
{
  "bundlesize": [
    { "path": "dist/*.js", "maxSize": "200 kB" }
  ]
}
```

---

## Hands-on Exercise

### Your Task

Measure Core Web Vitals on any website:

1. Open PageSpeed Insights
2. Enter a URL
3. Note the scores for LCP, INP, CLS
4. Identify the top 3 opportunities
5. For each issue, describe how you'd fix it

### Analysis Template

```markdown
## Core Web Vitals Analysis: [URL]

### Scores
- LCP: ___ (Good/Needs Improvement/Poor)
- INP: ___ (Good/Needs Improvement/Poor)
- CLS: ___ (Good/Needs Improvement/Poor)

### Top Opportunities
1. Issue: ___
   Fix: ___
   
2. Issue: ___
   Fix: ___
   
3. Issue: ___
   Fix: ___
```

---

## Summary

✅ **LCP** (Loading): Optimize largest element, use preload, reduce blocking
✅ **INP** (Interactivity): Break long tasks, optimize handlers, use workers
✅ **CLS** (Stability): Set dimensions, reserve space, optimize fonts
✅ Use **web-vitals library** for measurement
✅ Set **performance budgets** to prevent regression
✅ Test with **real user data** (CrUX) and **lab data** (Lighthouse)

**Next:** [Resource Optimization](./02-resource-optimization.md)

---

## Further Reading

- [web.dev Core Web Vitals](https://web.dev/vitals/)
- [Optimize LCP](https://web.dev/optimize-lcp/)
- [Optimize INP](https://web.dev/optimize-inp/)
- [Optimize CLS](https://web.dev/optimize-cls/)

<!-- 
Sources Consulted:
- web.dev Vitals: https://web.dev/vitals/
- Chrome DevTools: https://developer.chrome.com/docs/devtools/
-->
