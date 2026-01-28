---
title: "Web Performance Basics"
---

# Web Performance Basics

## Overview

Performance directly impacts user experience, conversions, and SEO. A slow website frustrates users and drives them away. Understanding Core Web Vitals, resource optimization, and the critical rendering path helps you build fast, responsive sites.

This lesson covers measuring and improving web performance from the ground up.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-core-web-vitals.md) | Core Web Vitals | LCP, INP, CLS, measuring, budgets |
| [02](./02-resource-optimization.md) | Resource Optimization | Images, fonts, code splitting, CDNs |
| [03](./03-critical-rendering-path.md) | Critical Rendering Path | Render-blocking, async/defer, preload |
| [04](./04-performance-measurement.md) | Performance Measurement | Performance API, Lighthouse, RUM |

---

## Why Performance Matters

### Impact on Business

| Metric | Impact |
|--------|--------|
| **1 second delay** | 7% drop in conversions |
| **3 second load** | 53% mobile users abandon |
| **Poor LCP** | Lower Google rankings |
| **Good performance** | Higher engagement, revenue |

### Core Web Vitals

Google's key metrics for user experience:

```
┌─────────────────────────────────────────────────────┐
│                 CORE WEB VITALS                     │
├─────────────────┬─────────────────┬─────────────────┤
│      LCP        │      INP        │      CLS        │
│   Loading       │  Interactivity  │ Visual Stability│
├─────────────────┼─────────────────┼─────────────────┤
│  Good: ≤2.5s    │  Good: ≤200ms   │  Good: ≤0.1     │
│  Poor: >4s      │  Poor: >500ms   │  Poor: >0.25    │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## Performance Checklist

### Quick Wins

| Optimization | Impact |
|--------------|--------|
| Compress images (WebP/AVIF) | High |
| Enable gzip/brotli | High |
| Use CDN | High |
| Lazy load images | Medium |
| Minify CSS/JS | Medium |
| Preload critical resources | Medium |

### Critical Path

```html
<!-- Preload hero image -->
<link rel="preload" href="hero.webp" as="image">

<!-- Async non-critical scripts -->
<script src="analytics.js" async></script>

<!-- Defer scripts until DOM ready -->
<script src="app.js" defer></script>

<!-- Preconnect to external origins -->
<link rel="preconnect" href="https://fonts.googleapis.com">
```

---

## Prerequisites

Before starting this lesson:
- HTML/CSS/JavaScript fundamentals
- Understanding of browser DevTools
- Basic server concepts (helpful)

---

## Start Learning

Begin with [Core Web Vitals](./01-core-web-vitals.md) to understand Google's key performance metrics.

---

## Further Reading

- [web.dev Performance](https://web.dev/performance/)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)
- [MDN Performance](https://developer.mozilla.org/en-US/docs/Web/Performance)
