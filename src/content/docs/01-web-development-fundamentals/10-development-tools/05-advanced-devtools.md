---
title: "Advanced DevTools Features"
---

# Advanced DevTools Features

## Introduction

Beyond the basics, DevTools offers powerful features for debugging complex issues: source maps for debugging minified code, breakpoints that pause on conditions, memory analysis for finding leaks, and more.

This lesson covers the advanced tools that separate debugging novices from experts.

### What We'll Cover

- Source maps for debugging
- Breakpoints and conditional breakpoints
- Memory heap snapshots
- Coverage panel for unused code
- Application panel (storage inspection)
- Rendering panel (paint flashing, FPS meter)

### Prerequisites

- Basic DevTools proficiency
- JavaScript debugging experience
- Understanding of how builds work (bundlers, minification)

---

## Source Maps

Source maps connect minified/bundled code back to your original source.

### The Problem

Minified code is unreadable:

```javascript
// Your code
function calculateDiscount(price, percentage) {
  const discount = price * (percentage / 100);
  return price - discount;
}

// After minification
function a(b,c){return b-b*(c/100)}
```

### How Source Maps Work

A `.map` file contains mappings between minified and original code:

```json
{
  "version": 3,
  "sources": ["src/utils.js"],
  "names": ["calculateDiscount", "price", "percentage"],
  "mappings": "AAAA,SAASA,gBAAgBC..."
}
```

### Enabling Source Maps

Most bundlers generate source maps automatically:

```javascript
// webpack.config.js
module.exports = {
  devtool: 'source-map',  // Full source maps
  // or 'cheap-module-source-map' for faster builds
};

// vite.config.js
export default {
  build: {
    sourcemap: true
  }
};
```

### Viewing Original Source

1. Open **Sources** panel
2. Look in the file tree (left sidebar)
3. Original files appear under `webpack://` or similar
4. Click to view and debug original code

### Source Map Settings

In DevTools Settings (gear icon):
- **Enable JavaScript source maps** - On by default
- **Enable CSS source maps** - For debugging Sass/Less

---

## Breakpoints

Pause execution to inspect state.

### Types of Breakpoints

| Type | How to Set | Use Case |
|------|------------|----------|
| Line | Click line number | Pause at specific line |
| Conditional | Right-click → Add conditional | Pause only when condition is true |
| Logpoint | Right-click → Add logpoint | Log without pausing |
| DOM | Elements panel → Break on... | Pause when DOM changes |
| XHR/Fetch | Sources → XHR/Fetch Breakpoints | Pause on network requests |
| Event Listener | Sources → Event Listener Breakpoints | Pause on specific events |

### Line Breakpoints

1. Open **Sources** panel
2. Navigate to your file
3. Click the line number
4. Blue marker appears
5. Run your code—execution pauses

### Conditional Breakpoints

Pause only when a condition is true:

```javascript
function processUser(user) {
  // Right-click line number → Add conditional breakpoint
  // Condition: user.role === 'admin'
  doSomething(user);
}
```

Now execution pauses only for admin users.

### Logpoints

Log without modifying code:

```javascript
// Right-click → Add logpoint
// Message: "Processing user:", user.name
function processUser(user) {
  doSomething(user);
}
```

Console shows: `Processing user: Alice` (without pausing)

### XHR/Fetch Breakpoints

1. Sources panel → **XHR/Fetch Breakpoints**
2. Click **+** to add
3. Enter URL pattern (e.g., `/api/users`)
4. Execution pauses on matching requests

### Event Listener Breakpoints

1. Sources panel → **Event Listener Breakpoints**
2. Expand categories (Mouse, Keyboard, etc.)
3. Check specific events (e.g., `click`)
4. All click handlers will pause

### Debugging Controls

When paused:

| Control | Shortcut | Action |
|---------|----------|--------|
| Resume | `F8` | Continue execution |
| Step Over | `F10` | Next line (skip functions) |
| Step Into | `F11` | Enter function |
| Step Out | `Shift+F11` | Exit current function |
| Deactivate | `Ctrl+F8` | Disable all breakpoints |

---

## Memory Heap Snapshots

Find memory leaks by comparing heap states.

### Taking Snapshots

1. Open **Memory** panel
2. Select **Heap snapshot**
3. Click **Take snapshot**

### Reading Snapshots

```
Heap Snapshot:
├── (array)          - 2,345 objects, 1.2 MB
├── (closure)        - 1,234 objects, 0.8 MB
├── (compiled code)  - 567 objects, 2.1 MB
├── HTMLDivElement   - 89 objects, 0.1 MB
└── MyClass          - 234 objects, 0.5 MB
```

### Finding Leaks

1. Take snapshot **before** action
2. Perform suspected leaky action
3. Take snapshot **after**
4. Select "Comparison" view
5. Look for objects that grew unexpectedly

### Retained Size vs Shallow Size

| Metric | Meaning |
|--------|---------|
| **Shallow size** | Memory of object itself |
| **Retained size** | Memory that would be freed if object is GC'd |

Large retained sizes indicate objects holding references to many others.

### Allocation Timeline

Track allocations over time:

1. Memory panel → **Allocation instrumentation on timeline**
2. Click **Start**
3. Perform actions
4. Click **Stop**
5. Blue bars = allocations, view what was allocated

---

## Coverage Panel

Find unused CSS and JavaScript.

### Opening Coverage

1. Command Menu (`Ctrl+Shift+P`)
2. Type "Show Coverage"
3. Click **Start instrumenting coverage and reload**

### Reading Results

```
URL                    │ Type │ Total Bytes │ Unused Bytes │ Usage
──────────────────────────────────────────────────────────────────
app.bundle.js          │ JS   │ 500 KB      │ 350 KB       │ 30%
styles.css             │ CSS  │ 100 KB      │ 65 KB        │ 35%
vendor.bundle.js       │ JS   │ 1.2 MB      │ 900 KB       │ 25%
```

### Viewing Unused Code

1. Click any file in Coverage results
2. Opens in Sources panel
3. **Red bars** = unused code
4. **Blue bars** = executed code

### Acting on Results

| Finding | Action |
|---------|--------|
| Unused vendor code | Tree-shake or lazy load |
| Unused CSS | PurgeCSS, remove manually |
| Unused app code | Code split, remove dead code |

---

## Application Panel

Inspect client-side storage and service workers.

### Storage Section

| Store | What it shows |
|-------|---------------|
| **Local Storage** | Key-value pairs |
| **Session Storage** | Session-scoped data |
| **IndexedDB** | Database structure and records |
| **Cookies** | All cookies for the origin |
| **Cache Storage** | Service worker cached responses |

### Viewing and Editing Storage

1. Expand **Local Storage**
2. Select your origin
3. See all key-value pairs
4. Double-click to edit
5. Right-click to delete

### Service Workers

See registered service workers:
- Status (activated, waiting, stopped)
- Update on reload option
- Unregister button
- Push and Sync simulation

### Clear Storage

1. Application panel → **Storage** section
2. Click **Clear site data**
3. Choose what to clear:
   - Cookies
   - Storage (localStorage, IndexedDB)
   - Cache
   - Service workers

### Manifest

For PWAs, view the web app manifest:
- App name and icons
- Start URL
- Display mode
- Theme colors

---

## Rendering Panel

Debug visual issues and measure performance.

### Opening Rendering

1. Command Menu (`Ctrl+Shift+P`)
2. Type "Show Rendering"

### Paint Flashing

Highlights areas being repainted:

- Green flashes = repaint happening
- Too much flashing = performance issue

**Common causes of excessive repaints:**
- Animations on wrong properties
- Scrolling without `will-change`
- Large images resizing

### Layout Shift Regions

Highlights CLS (Cumulative Layout Shift):
- Blue rectangles show shifting elements
- Helps identify what causes layout instability

### FPS Meter

Shows real-time frames per second:

```
┌─────────────────────────────┐
│ FPS: 60 ████████████████████│
│ GPU: 12 MB                  │
│ Frames: 16.7ms              │
└─────────────────────────────┘
```

- **60 FPS** = Smooth animations
- **Below 30 FPS** = User-visible jank

### Layer Borders

Shows compositor layer boundaries:
- Orange = layer edges
- Help understand what's GPU-accelerated

### Scrolling Performance Issues

Highlights scrolling problems:
- Event handlers on scroll
- Elements blocking fast scrolling

---

## Other Advanced Features

### Snippets

Save and run JavaScript snippets:

1. Sources panel → **Snippets** (left sidebar)
2. Click **New snippet**
3. Write reusable code
4. Run with `Ctrl+Enter`

```javascript
// Snippet: Log all event listeners
const listeners = getEventListeners(document);
console.table(Object.keys(listeners).map(type => ({
  type,
  count: listeners[type].length
})));
```

### Workspaces

Edit files directly in DevTools:

1. Sources panel → **Filesystem** tab
2. **Add folder to workspace**
3. Grant permission
4. Changes save to disk!

### Remote Debugging

Debug mobile devices:

1. Enable USB debugging on phone
2. Connect via USB
3. Visit `chrome://inspect` on desktop
4. Click **Inspect** next to your device

### Override Network Responses

Mock API responses without changing server:

1. Network panel → Right-click request
2. **Override content**
3. Edit response
4. Changes apply on next request

---

## Hands-on Exercise

### Your Task

Debug a memory leak:

```javascript
// This code has a memory leak - find it!
const cache = [];

function processData(data) {
  cache.push(data);  // Never cleared!
  
  // Process data...
  return transform(data);
}

// Called repeatedly
setInterval(() => {
  processData({ large: new Array(10000).fill('x') });
}, 100);
```

### Steps

1. Open Memory panel
2. Take heap snapshot
3. Wait 10 seconds
4. Take another snapshot
5. Compare—see the `cache` array growing

### Challenge

Use the Coverage panel to find unused code in a real website.

---

## Summary

✅ **Source maps** connect minified code to original
✅ **Conditional breakpoints** pause only when conditions match
✅ **Logpoints** log without pausing or modifying code
✅ **Heap snapshots** reveal memory leaks via comparison
✅ **Coverage panel** shows unused CSS/JS
✅ **Application panel** inspects storage and service workers
✅ **Rendering panel** shows paint flashing, FPS, and layers

**Next:** [Version Control with Git](./06-version-control-git.md)

---

## Further Reading

- [Chrome DevTools Sources Panel](https://developer.chrome.com/docs/devtools/sources/)
- [Chrome DevTools Memory](https://developer.chrome.com/docs/devtools/memory/)
- [Source Maps Introduction](https://web.dev/articles/source-maps)

<!-- 
Sources Consulted:
- Chrome DevTools: https://developer.chrome.com/docs/devtools/
- Source Maps: https://web.dev/articles/source-maps
-->
