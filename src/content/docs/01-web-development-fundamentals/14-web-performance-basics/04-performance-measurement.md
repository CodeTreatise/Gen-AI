---
title: "Performance Measurement"
---

# Performance Measurement

## Introduction

You can't improve what you don't measure. Performance measurement combines browser APIs, lab testing tools, and real-user monitoring to give a complete picture of site speed.

This lesson covers the Performance API, Lighthouse, and RUM concepts.

### What We'll Cover

- Performance API
- Navigation Timing API
- Resource Timing API
- Using Lighthouse
- Real User Monitoring (RUM) concepts

### Prerequisites

- JavaScript fundamentals
- Browser DevTools experience
- Understanding of Core Web Vitals

---

## Performance API

### Accessing Performance Data

```javascript
// Get performance timing
const timing = performance.timing;  // Deprecated, use newer APIs
const entries = performance.getEntries();

// Get specific entry types
const navigation = performance.getEntriesByType('navigation')[0];
const resources = performance.getEntriesByType('resource');
const paints = performance.getEntriesByType('paint');
```

### Performance Timeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Performance Timeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  navigationStart                                            │
│  │                                                          │
│  ├─ redirectStart ─ redirectEnd                            │
│  │                                                          │
│  ├─ fetchStart                                              │
│  │   │                                                      │
│  │   ├─ domainLookupStart ─ domainLookupEnd (DNS)          │
│  │   ├─ connectStart ─ connectEnd (TCP)                    │
│  │   │   └─ secureConnectionStart (TLS)                    │
│  │   │                                                      │
│  │   └─ requestStart ─ responseStart ─ responseEnd         │
│  │                                                          │
│  ├─ domInteractive                                          │
│  ├─ domContentLoadedEventStart ─ End                       │
│  ├─ domComplete                                             │
│  └─ loadEventStart ─ loadEventEnd                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Navigation Timing API

### Modern API (PerformanceNavigationTiming)

```javascript
const navigation = performance.getEntriesByType('navigation')[0];

// Key metrics
const dnsTime = navigation.domainLookupEnd - navigation.domainLookupStart;
const tcpTime = navigation.connectEnd - navigation.connectStart;
const ttfb = navigation.responseStart - navigation.requestStart;
const downloadTime = navigation.responseEnd - navigation.responseStart;
const domParsing = navigation.domInteractive - navigation.responseEnd;
const totalTime = navigation.loadEventEnd - navigation.startTime;

console.log(`
DNS lookup: ${dnsTime}ms
TCP connection: ${tcpTime}ms
TTFB: ${ttfb}ms
Download: ${downloadTime}ms
DOM parsing: ${domParsing}ms
Total: ${totalTime}ms
`);
```

### Key Metrics Explained

| Metric | Formula | Target |
|--------|---------|--------|
| **TTFB** | responseStart - requestStart | < 800ms |
| **First Byte** | responseStart - navigationStart | < 1s |
| **DOM Interactive** | domInteractive - navigationStart | < 3s |
| **Page Load** | loadEventEnd - navigationStart | < 4s |

---

## Resource Timing API

### Analyzing Resource Performance

```javascript
const resources = performance.getEntriesByType('resource');

resources.forEach(resource => {
  console.log({
    name: resource.name,
    type: resource.initiatorType,  // script, css, img, etc.
    duration: resource.duration,
    transferSize: resource.transferSize,
    decodedBodySize: resource.decodedBodySize
  });
});
```

### Find Slow Resources

```javascript
function getSlowResources(threshold = 1000) {
  return performance.getEntriesByType('resource')
    .filter(r => r.duration > threshold)
    .sort((a, b) => b.duration - a.duration)
    .map(r => ({
      url: r.name,
      duration: Math.round(r.duration),
      size: r.transferSize
    }));
}

console.table(getSlowResources());
```

### Resource Size Analysis

```javascript
function analyzeResources() {
  const resources = performance.getEntriesByType('resource');
  const byType = {};
  
  resources.forEach(r => {
    const type = r.initiatorType || 'other';
    byType[type] = byType[type] || { count: 0, size: 0 };
    byType[type].count++;
    byType[type].size += r.transferSize || 0;
  });
  
  return byType;
}

console.table(analyzeResources());
// Shows: script: {count: 5, size: 245000}, img: {count: 10, size: 320000}, etc.
```

---

## Paint Timing

### First Contentful Paint (FCP)

```javascript
const paintEntries = performance.getEntriesByType('paint');

const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint');
console.log('FCP:', fcp?.startTime, 'ms');

// Using PerformanceObserver for real-time
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name}: ${entry.startTime}ms`);
  }
});

observer.observe({ entryTypes: ['paint'] });
```

### Largest Contentful Paint (LCP)

```javascript
// LCP requires PerformanceObserver
new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime, 'ms');
  console.log('LCP Element:', lastEntry.element);
}).observe({ type: 'largest-contentful-paint', buffered: true });
```

---

## Long Tasks API

Find JavaScript that blocks the main thread:

```javascript
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Long Task detected:');
    console.log('Duration:', entry.duration, 'ms');
    console.log('Start Time:', entry.startTime, 'ms');
    console.log('Attribution:', entry.attribution);
  }
}).observe({ entryTypes: ['longtask'] });
```

A "long task" is any task taking > 50ms.

---

## Using Lighthouse

### Running Lighthouse

**In Chrome DevTools:**
1. Open DevTools (F12)
2. Lighthouse tab
3. Select categories
4. Click "Analyze page load"

**CLI:**
```bash
npm install -g lighthouse
lighthouse https://example.com --view
lighthouse https://example.com --output=json --output-path=./report.json
```

**Node:**
```javascript
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

async function runLighthouse(url) {
  const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
  const result = await lighthouse(url, {
    port: chrome.port,
    onlyCategories: ['performance']
  });
  await chrome.kill();
  return result.lhr;
}
```

### Understanding Lighthouse Scores

| Category | Weight | What It Measures |
|----------|--------|------------------|
| **Performance** | Various | Speed metrics |
| **Accessibility** | Pass/Fail | a11y issues |
| **Best Practices** | Pass/Fail | Security, modern code |
| **SEO** | Pass/Fail | Search optimization |
| **PWA** | Pass/Fail | Progressive Web App |

### Performance Metrics Weighting

| Metric | Weight (Lighthouse 10) |
|--------|------------------------|
| Total Blocking Time | 30% |
| Largest Contentful Paint | 25% |
| Cumulative Layout Shift | 25% |
| First Contentful Paint | 10% |
| Speed Index | 10% |

### Lighthouse CI

```bash
npm install -g @lhci/cli

# In CI pipeline
lhci autorun
```

```javascript
// lighthouserc.js
module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:3000/'],
      numberOfRuns: 3
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'first-contentful-paint': ['error', { maxNumericValue: 2000 }]
      }
    }
  }
};
```

---

## Real User Monitoring (RUM)

### What Is RUM?

**Lab data** (Lighthouse): Simulated conditions, consistent
**Field data** (RUM): Real users, actual conditions

```
Lab Data:                    Field Data (RUM):
┌────────────────┐           ┌────────────────┐
│ Fast network   │           │ Varied networks │
│ Fast device    │           │ Varied devices  │
│ Empty cache    │           │ Some have cache │
│ One location   │           │ Global users    │
└────────────────┘           └────────────────┘
```

### Collecting RUM Data

```javascript
// Using web-vitals library
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics({ name, value, id, rating }) {
  fetch('/analytics', {
    method: 'POST',
    body: JSON.stringify({
      metric: name,
      value: value,
      id: id,
      rating: rating,  // 'good', 'needs-improvement', 'poor'
      url: window.location.href,
      userAgent: navigator.userAgent
    }),
    keepalive: true  // Ensure it sends even on page unload
  });
}

onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);
```

### RUM Services

| Service | Type | Cost |
|---------|------|------|
| **Google Analytics** | Free | Web Vitals report |
| **CrUX** | Free | Chrome users |
| **Vercel Analytics** | Free tier | Vercel sites |
| **Sentry Performance** | Paid | Full featured |
| **New Relic** | Paid | Enterprise |

### Analyzing RUM Data

Focus on:
- **P75** (75th percentile): Most users experience this or better
- **Segmentation**: By device, connection, geography
- **Trends**: Is performance improving over time?

```javascript
// Calculate percentiles
function percentile(arr, p) {
  arr.sort((a, b) => a - b);
  const index = Math.ceil(arr.length * (p / 100)) - 1;
  return arr[index];
}

// P75 of LCP values
const lcpP75 = percentile(lcpValues, 75);
```

---

## Performance Budgets in Measurement

### Setting Up Monitoring

```javascript
// Check performance budget in real-time
function checkBudget() {
  const budget = {
    lcp: 2500,
    cls: 0.1,
    inp: 200
  };
  
  const violations = [];
  
  // Check after page load
  const navigation = performance.getEntriesByType('navigation')[0];
  if (navigation.duration > 4000) {
    violations.push('Page load exceeded 4s');
  }
  
  // Check resource budgets
  const resources = performance.getEntriesByType('resource');
  const jsSize = resources
    .filter(r => r.initiatorType === 'script')
    .reduce((sum, r) => sum + (r.transferSize || 0), 0);
  
  if (jsSize > 200000) {
    violations.push(`JS size: ${Math.round(jsSize/1000)}KB > 200KB budget`);
  }
  
  return violations;
}
```

---

## Hands-on Exercise

### Your Task

Create a performance dashboard:

```javascript
// Build a function that collects and displays key metrics
function getPerformanceReport() {
  // 1. Get navigation timing
  // 2. Get resource counts and sizes
  // 3. Get paint timings
  // 4. Return formatted report
}
```

<details>
<summary>✅ Solution</summary>

```javascript
function getPerformanceReport() {
  const nav = performance.getEntriesByType('navigation')[0];
  const resources = performance.getEntriesByType('resource');
  const paints = performance.getEntriesByType('paint');
  
  // Navigation metrics
  const timing = {
    dns: Math.round(nav.domainLookupEnd - nav.domainLookupStart),
    tcp: Math.round(nav.connectEnd - nav.connectStart),
    ttfb: Math.round(nav.responseStart - nav.requestStart),
    download: Math.round(nav.responseEnd - nav.responseStart),
    domInteractive: Math.round(nav.domInteractive),
    domComplete: Math.round(nav.domComplete),
    loadEvent: Math.round(nav.loadEventEnd)
  };
  
  // Resource summary
  const resourceSummary = {};
  resources.forEach(r => {
    const type = r.initiatorType || 'other';
    if (!resourceSummary[type]) {
      resourceSummary[type] = { count: 0, size: 0, time: 0 };
    }
    resourceSummary[type].count++;
    resourceSummary[type].size += r.transferSize || 0;
    resourceSummary[type].time += r.duration || 0;
  });
  
  // Paint timing
  const paintTiming = {};
  paints.forEach(p => {
    paintTiming[p.name] = Math.round(p.startTime);
  });
  
  console.log('=== Performance Report ===');
  console.log('\nTiming (ms):');
  console.table(timing);
  console.log('\nResources by Type:');
  console.table(resourceSummary);
  console.log('\nPaint Timing:');
  console.table(paintTiming);
  
  return { timing, resourceSummary, paintTiming };
}

// Run after page load
window.addEventListener('load', () => {
  setTimeout(getPerformanceReport, 0);
});
```
</details>

---

## Summary

✅ **Navigation Timing API** measures page load phases
✅ **Resource Timing API** analyzes individual resource performance
✅ **PerformanceObserver** tracks paint timing and long tasks
✅ **Lighthouse** provides comprehensive lab testing
✅ **RUM** captures real user experience (field data)
✅ Combine **lab and field data** for complete picture
✅ Set **performance budgets** and monitor continuously

**Back to:** [Web Performance Basics Overview](./00-web-performance-basics.md)

---

## Further Reading

- [web.dev Performance Measurement](https://web.dev/metrics/)
- [MDN Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API)
- [web-vitals Library](https://github.com/GoogleChrome/web-vitals)

<!-- 
Sources Consulted:
- MDN Performance API: https://developer.mozilla.org/en-US/docs/Web/API/Performance_API
- web.dev Metrics: https://web.dev/metrics/
-->
