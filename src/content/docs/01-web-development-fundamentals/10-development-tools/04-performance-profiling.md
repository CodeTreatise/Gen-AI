---
title: "Performance Profiling Basics"
---

# Performance Profiling Basics

## Introduction

A slow website frustrates users and hurts conversions. But "slow" can mean many things—slow to load, slow to respond, slow animations. The **Performance panel** helps you understand exactly where time is spent and what's blocking the main thread.

This lesson covers recording performance profiles, reading flame charts, and measuring Core Web Vitals—the metrics Google uses for ranking.

### What We'll Cover

- Performance panel overview
- Recording performance
- Flame charts
- Main thread analysis
- Memory profiling basics
- Lighthouse audits
- Core Web Vitals (LCP, FID, CLS, INP)

### Prerequisites

- Understanding of how browsers render pages
- JavaScript fundamentals
- Basic DevTools familiarity

---

## Performance Panel Overview

The Performance panel records everything that happens during page load or interaction:
- JavaScript execution
- Style calculations
- Layout
- Paint
- Composite

### Opening Performance Panel

1. Open DevTools (`F12`)
2. Click **Performance** tab
3. Click the **Record** button (circle)
4. Interact with the page
5. Click **Stop**

Or: Press `Ctrl+Shift+E` to start/stop recording.

---

## Recording Performance

### Page Load Recording

1. Click **reload icon** (circle with arrow) in Performance panel
2. Page reloads and records from the start
3. Recording stops when page is "idle"

### Interaction Recording

1. Click **Record** (circle)
2. Perform the interaction (click button, scroll, etc.)
3. Click **Stop**
4. Analyze the recording

### Recording Settings

| Setting | Purpose |
|---------|---------|
| **Disable JavaScript samples** | Faster recording, less detail |
| **Network throttling** | Simulate slow connections |
| **CPU throttling** | Simulate slower devices (4x, 6x slowdown) |
| **Screenshots** | Capture visual snapshots during recording |
| **Memory** | Include memory timeline |

> **Tip:** Enable CPU throttling (4x slowdown) to catch issues that only appear on slower devices.

---

## Reading Flame Charts

The flame chart shows what code executed and when.

### Anatomy of a Flame Chart

```
Timeline (ms):
0         100        200        300        400
├──────────┼──────────┼──────────┼──────────┤

Main Thread:
┌─────────────────────────────────────────────┐
│ Task                                        │
├─────────────┬───────────────────────────────┤
│ Parse HTML  │ Evaluate Script               │
│             ├───────────────────────────────┤
│             │ (anonymous)                   │
│             ├─────────────┬─────────────────┤
│             │ processData │ renderItems     │
│             ├──────┬──────┤                 │
│             │filter│map   │                 │
└─────────────┴──────┴──────┴─────────────────┘
```

### Reading the Chart

- **Width** = Time duration (wider = longer)
- **Depth** = Call stack (deeper = more nested calls)
- **Colors** = Activity type:
  - 🟡 Yellow = JavaScript
  - 🟣 Purple = Style/Layout
  - 🟢 Green = Paint
  - ⚪ Gray = Idle/System

### Finding Slow Functions

1. Look for wide bars (long execution time)
2. Click to see function name and file location
3. Look for repeated patterns (function called many times)

---

## Main Thread Analysis

The main thread handles:
- JavaScript execution
- DOM updates
- Event handling
- Style calculations
- Layout

### Long Tasks

Tasks over 50ms block user interaction:

```
Main Thread:
┌─────────────────────────────────────┐
│ Long Task (250ms) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Blocks UI
└─────────────────────────────────────┘
     ↑
     During this time:
     - Clicks don't register
     - Animations freeze
     - Page feels "janky"
```

### Identifying Blocking Code

1. Enable **Main** track in recording
2. Look for solid blocks of yellow (JS execution)
3. Hover to see task duration
4. Red corner = long task (>50ms)

### Common Causes

| Cause | Solution |
|-------|----------|
| Heavy computation | Web Workers |
| Large DOM updates | Virtual DOM, batching |
| Synchronous loops | Break into chunks with `setTimeout` |
| Forced reflows | Batch DOM reads/writes |

---

## Memory Profiling Basics

Memory issues cause slowdowns and crashes.

### Memory Timeline

Enable **Memory** checkbox before recording:

```
Memory:
       JS Heap
       ┌────────────────────────────────────┐
  40MB │    ╱╲      ╱╲      ╱╲             │
       │   ╱  ╲    ╱  ╲    ╱  ╲            │
  20MB │  ╱    ╲──╱    ╲──╱    ╲──         │
       │ ╱                                  │
   0MB └────────────────────────────────────┘
       Time →
       
       Healthy: Sawtooth pattern (GC working)
       Bad: Continuous rise (memory leak)
```

### Heap Snapshots

Take snapshots to find memory leaks:

1. Open **Memory** panel
2. Select **Heap snapshot**
3. Click **Take snapshot**
4. Perform actions
5. Take another snapshot
6. Compare to find retained objects

### Common Memory Leaks

| Leak Type | Cause | Fix |
|-----------|-------|-----|
| Event listeners | Not removed on cleanup | Remove in cleanup/unmount |
| Closures | References to large objects | Nullify references |
| Detached DOM | Elements removed but referenced | Clear references |
| Timers | setInterval not cleared | clearInterval on cleanup |

---

## Lighthouse Audits

Lighthouse provides automated performance scoring.

### Running an Audit

1. Open **Lighthouse** panel (or find via Command Menu)
2. Select categories:
   - Performance
   - Accessibility
   - Best Practices
   - SEO
   - PWA
3. Choose device (Mobile/Desktop)
4. Click **Analyze page load**

### Performance Score

Lighthouse scores 0-100 based on:

| Metric | Weight |
|--------|--------|
| First Contentful Paint (FCP) | 10% |
| Largest Contentful Paint (LCP) | 25% |
| Total Blocking Time (TBT) | 30% |
| Cumulative Layout Shift (CLS) | 25% |
| Speed Index | 10% |

### Interpreting Results

| Score | Rating |
|-------|--------|
| 90-100 | 🟢 Good |
| 50-89 | 🟡 Needs Improvement |
| 0-49 | 🔴 Poor |

### Actionable Recommendations

Lighthouse provides specific fixes:
- "Eliminate render-blocking resources"
- "Reduce unused JavaScript"
- "Properly size images"
- "Serve images in next-gen formats"

---

## Core Web Vitals

Google's metrics for real-world user experience.

### The Three Core Web Vitals

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

### Largest Contentful Paint (LCP)

**What:** Time until the largest visible element loads.

**Good:** ≤ 2.5 seconds

**What counts as LCP:**
- `<img>` elements
- `<video>` poster images
- Elements with `background-image`
- Block-level text elements

**Improve LCP:**
```html
<!-- Preload critical images -->
<link rel="preload" href="hero.jpg" as="image">

<!-- Optimize images -->
<img src="hero.webp" loading="eager" fetchpriority="high">
```

### Interaction to Next Paint (INP)

**What:** Responsiveness to user interactions (replaces FID).

**Good:** ≤ 200 milliseconds

**What triggers INP:**
- Clicks
- Taps
- Key presses

**Improve INP:**
```javascript
// Break up long tasks
function processItems(items) {
  const chunk = items.splice(0, 100);
  processChunk(chunk);
  
  if (items.length > 0) {
    // Yield to main thread
    setTimeout(() => processItems(items), 0);
  }
}
```

### Cumulative Layout Shift (CLS)

**What:** How much the page layout shifts unexpectedly.

**Good:** ≤ 0.1

**Common causes:**
- Images without dimensions
- Ads/embeds without reserved space
- Dynamically injected content
- FOUT (Flash of Unstyled Text)

**Improve CLS:**
```html
<!-- Always set dimensions -->
<img src="photo.jpg" width="800" height="600" alt="...">

<!-- Reserve space for dynamic content -->
<div style="min-height: 250px;">
  <!-- Ad loads here -->
</div>
```

### Measuring Core Web Vitals

```javascript
// Using web-vitals library
import { onLCP, onINP, onCLS } from 'web-vitals';

onLCP(console.log);   // {name: 'LCP', value: 2456, ...}
onINP(console.log);   // {name: 'INP', value: 89, ...}
onCLS(console.log);   // {name: 'CLS', value: 0.05, ...}
```

In DevTools:
1. Performance panel → enable **Web Vitals**
2. Record page load
3. See LCP, CLS markers on timeline

---

## Performance Tips Summary

### Quick Wins

| Issue | Quick Fix |
|-------|-----------|
| Large images | Compress, use WebP/AVIF |
| Render-blocking CSS | Inline critical CSS |
| Unused JavaScript | Code splitting |
| No caching | Set Cache-Control headers |
| No compression | Enable gzip/brotli |

### JavaScript Performance

```javascript
// ❌ Avoid forced reflow
const height = element.offsetHeight;  // Read
element.style.height = height + 'px'; // Write
const width = element.offsetWidth;    // Read - forces reflow!

// ✅ Batch reads and writes
const height = element.offsetHeight;  // Read
const width = element.offsetWidth;    // Read
element.style.height = height + 'px'; // Write
element.style.width = width + 'px';   // Write
```

### Image Optimization

```html
<!-- Responsive images -->
<img 
  srcset="small.jpg 480w, medium.jpg 800w, large.jpg 1200w"
  sizes="(max-width: 600px) 480px, 800px"
  src="medium.jpg"
  alt="Description"
>

<!-- Lazy loading -->
<img src="photo.jpg" loading="lazy" alt="...">
```

---

## Hands-on Exercise

### Your Task

Profile a real website:

1. Open any website (your own or a public site)
2. Open Performance panel
3. Enable **Screenshots** and **CPU throttling (4x)**
4. Click the reload button to record page load
5. Analyze:
   - How long until first paint?
   - What's the longest JavaScript task?
   - Are there layout shifts?

### Challenge

Run a Lighthouse audit and achieve a Performance score of 90+.

---

## Summary

✅ **Record** page loads and interactions
✅ **Flame charts** show function execution and call stacks
✅ Look for **long tasks** (>50ms) blocking the main thread
✅ **Memory timeline** reveals leaks (continuously rising)
✅ **Lighthouse** provides automated scoring and recommendations
✅ **Core Web Vitals**: LCP (loading), INP (interactivity), CLS (stability)

**Next:** [Advanced DevTools Features](./05-advanced-devtools.md)

---

## Further Reading

- [Chrome Performance Reference](https://developer.chrome.com/docs/devtools/performance/)
- [web.dev Core Web Vitals](https://web.dev/vitals/)
- [web.dev Performance](https://web.dev/learn/performance/)

<!-- 
Sources Consulted:
- Chrome DevTools Performance: https://developer.chrome.com/docs/devtools/performance/
- web.dev Core Web Vitals: https://web.dev/vitals/
-->
