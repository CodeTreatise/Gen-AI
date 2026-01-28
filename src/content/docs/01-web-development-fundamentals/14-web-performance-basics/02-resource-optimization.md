---
title: "Resource Optimization"
---

# Resource Optimization

## Introduction

Every byte sent to the browser affects load time. Optimizing resources—images, fonts, JavaScript, and CSS—is one of the most impactful performance improvements you can make.

This lesson covers practical techniques for reducing resource size and improving delivery.

### What We'll Cover

- Image optimization (WebP, AVIF, lazy loading)
- Font loading strategies (font-display)
- Code splitting concepts
- Minification and compression
- CDN basics

### Prerequisites

- HTML/CSS/JavaScript fundamentals
- Understanding of HTTP requests
- Basic build tool awareness

---

## Image Optimization

Images are typically the largest resources on a page.

### Modern Image Formats

| Format | Compression | Browser Support | Use Case |
|--------|-------------|-----------------|----------|
| **JPEG** | Lossy | Universal | Photos (legacy) |
| **PNG** | Lossless | Universal | Graphics, transparency |
| **WebP** | Both | 97%+ | Photos, graphics |
| **AVIF** | Both | 92%+ | Best compression |

### Using Modern Formats

```html
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Description" width="800" height="600">
</picture>
```

### Responsive Images

```html
<img 
  src="photo-800.jpg"
  srcset="photo-400.jpg 400w,
          photo-800.jpg 800w,
          photo-1200.jpg 1200w"
  sizes="(max-width: 600px) 400px,
         (max-width: 1000px) 800px,
         1200px"
  alt="Responsive photo">
```

### Lazy Loading

```html
<!-- Native lazy loading -->
<img src="photo.jpg" loading="lazy" alt="...">

<!-- Eager loading for LCP image -->
<img src="hero.jpg" loading="eager" fetchpriority="high" alt="...">
```

### Image Compression Tools

| Tool | Type | Use |
|------|------|-----|
| **Squoosh** | Online | Manual optimization |
| **ImageOptim** | App | Batch optimization |
| **Sharp** | Library | Build pipeline |
| **imgix/Cloudinary** | Service | Dynamic optimization |

```javascript
// Sharp in build pipeline
const sharp = require('sharp');

sharp('input.jpg')
  .resize(800, 600)
  .webp({ quality: 80 })
  .toFile('output.webp');
```

---

## Font Optimization

### font-display Strategies

```css
@font-face {
  font-family: 'Custom Font';
  src: url('font.woff2') format('woff2');
  font-display: swap;  /* Most common choice */
}
```

| Value | Behavior | Use Case |
|-------|----------|----------|
| `swap` | Show fallback immediately, swap when loaded | Body text |
| `block` | Hide text briefly, then show font | Logos, icons |
| `fallback` | Brief block, then fallback, may never swap | Balance |
| `optional` | May never show custom font | Non-critical |

### Preload Critical Fonts

```html
<link 
  rel="preload" 
  href="/fonts/main.woff2" 
  as="font" 
  type="font/woff2" 
  crossorigin>
```

### Subset Fonts

Only include characters you need:

```bash
# Using pyftsubset
pyftsubset font.ttf \
  --text="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" \
  --output-file=font-subset.woff2 \
  --flavor=woff2
```

Or use Google Fonts `text` parameter:
```html
<link href="https://fonts.googleapis.com/css2?family=Open+Sans&text=Hello" rel="stylesheet">
```

### Self-Host Fonts

```css
/* Faster than Google Fonts for most sites */
@font-face {
  font-family: 'Open Sans';
  src: url('/fonts/OpenSans-Regular.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
}
```

---

## Code Splitting

### Why Split Code?

```
Without splitting:
┌─────────────────────────────────────┐
│           bundle.js (500KB)          │
│  Login + Dashboard + Settings + ... │
└─────────────────────────────────────┘
  ↑ User downloads everything upfront

With splitting:
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ core.js (50KB)│ │login.js (30KB)│ │dash.js (80KB)│
└──────────────┘ └──────────────┘ └──────────────┘
  ↑ Load only what's needed
```

### Dynamic Imports

```javascript
// Static import - always loaded
import { Dashboard } from './Dashboard';

// Dynamic import - loaded on demand
const Dashboard = await import('./Dashboard');

// With React.lazy
const Dashboard = React.lazy(() => import('./Dashboard'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Dashboard />
    </Suspense>
  );
}
```

### Route-Based Splitting

```javascript
// React Router with lazy loading
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';

const Home = lazy(() => import('./pages/Home'));
const About = lazy(() => import('./pages/About'));
const Dashboard = lazy(() => import('./pages/Dashboard'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Suspense>
  );
}
```

### Vendor Splitting

```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          utils: ['lodash', 'date-fns']
        }
      }
    }
  }
};
```

---

## Minification and Compression

### Minification

Remove unnecessary characters from code:

```javascript
// Before minification
function calculateTotal(items) {
  let total = 0;
  for (const item of items) {
    total += item.price * item.quantity;
  }
  return total;
}

// After minification
function calculateTotal(t){let e=0;for(const n of t)e+=n.price*n.quantity;return e}
```

**Tools:**
- **Terser** - JavaScript
- **cssnano** - CSS
- **html-minifier** - HTML

Most build tools (Vite, webpack) minify automatically in production.

### Compression

Compress files for transfer:

| Algorithm | Compression | Speed | Support |
|-----------|-------------|-------|---------|
| **gzip** | Good | Fast | Universal |
| **Brotli** | Better | Slower | 97%+ |

### Enabling Compression

**nginx:**
```nginx
# Enable gzip
gzip on;
gzip_types text/plain text/css application/json application/javascript;

# Enable Brotli
brotli on;
brotli_types text/plain text/css application/json application/javascript;
```

**Express:**
```javascript
const compression = require('compression');
app.use(compression());
```

**Verify compression:**
```bash
curl -I -H "Accept-Encoding: gzip, br" https://example.com/app.js
# Look for: Content-Encoding: br (or gzip)
```

---

## CDN Basics

### What Is a CDN?

**Content Delivery Network** serves content from servers close to users:

```
Without CDN:
User in Tokyo → Server in New York (200ms latency)

With CDN:
User in Tokyo → Edge server in Tokyo (20ms latency)
```

### CDN Benefits

| Benefit | Impact |
|---------|--------|
| **Lower latency** | Faster load times |
| **Reduced server load** | Handles traffic spikes |
| **DDoS protection** | Absorbs attacks |
| **Automatic compression** | Brotli/gzip |
| **Image optimization** | On-the-fly conversion |

### Popular CDNs

| CDN | Specialty |
|-----|-----------|
| **Cloudflare** | Free tier, security |
| **AWS CloudFront** | AWS integration |
| **Fastly** | Edge computing |
| **Vercel Edge** | Frontend hosting |
| **Netlify** | JAMstack |

### CDN Configuration

```html
<!-- Use CDN for static assets -->
<link rel="stylesheet" href="https://cdn.example.com/styles.css">
<script src="https://cdn.example.com/app.js"></script>

<!-- Preconnect to CDN -->
<link rel="preconnect" href="https://cdn.example.com">
```

### Cache Headers

```http
# Cache static assets for 1 year
Cache-Control: public, max-age=31536000, immutable

# Cache HTML for short period
Cache-Control: public, max-age=3600

# Never cache (dynamic content)
Cache-Control: no-store
```

---

## Resource Loading Summary

### Loading Strategies

| Strategy | Use For | How |
|----------|---------|-----|
| **Preload** | Critical resources | `<link rel="preload">` |
| **Preconnect** | External origins | `<link rel="preconnect">` |
| **Prefetch** | Next page resources | `<link rel="prefetch">` |
| **Lazy load** | Below-fold images | `loading="lazy"` |
| **Async** | Non-blocking scripts | `<script async>` |
| **Defer** | After DOM scripts | `<script defer>` |

```html
<!-- Complete example -->
<head>
  <!-- Preconnect to external origins -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://cdn.example.com">
  
  <!-- Preload critical resources -->
  <link rel="preload" href="/fonts/main.woff2" as="font" crossorigin>
  <link rel="preload" href="/hero.webp" as="image">
  
  <!-- Critical CSS inline -->
  <style>/* critical CSS */</style>
  
  <!-- Defer non-critical CSS -->
  <link rel="stylesheet" href="/styles.css" media="print" onload="this.media='all'">
</head>
<body>
  <!-- Content -->
  
  <!-- Defer scripts -->
  <script src="/app.js" defer></script>
</body>
```

---

## Hands-on Exercise

### Your Task

Optimize this resource loading:

```html
<head>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto&display=swap">
  <link rel="stylesheet" href="/all-styles.css">
  <script src="/vendor.js"></script>
  <script src="/app.js"></script>
</head>
<body>
  <img src="/hero.png">
  <img src="/photo1.jpg">
  <img src="/photo2.jpg">
  <img src="/photo3.jpg">
</body>
```

### Optimized Version

<details>
<summary>✅ Solution</summary>

```html
<head>
  <!-- Preconnect to Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  
  <!-- Preload critical resources -->
  <link rel="preload" href="/hero.webp" as="image" fetchpriority="high">
  
  <!-- Self-host font or use display=swap -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto&display=swap">
  
  <!-- Critical CSS inline, defer rest -->
  <style>/* critical above-fold CSS */</style>
  <link rel="stylesheet" href="/styles.css" media="print" onload="this.media='all'">
</head>
<body>
  <!-- Hero with modern format, eager loading -->
  <picture>
    <source srcset="/hero.avif" type="image/avif">
    <source srcset="/hero.webp" type="image/webp">
    <img src="/hero.jpg" alt="..." width="1200" height="600" 
         loading="eager" fetchpriority="high">
  </picture>
  
  <!-- Other images lazy loaded -->
  <img src="/photo1.webp" alt="..." loading="lazy" width="400" height="300">
  <img src="/photo2.webp" alt="..." loading="lazy" width="400" height="300">
  <img src="/photo3.webp" alt="..." loading="lazy" width="400" height="300">
  
  <!-- Defer scripts -->
  <script src="/vendor.js" defer></script>
  <script src="/app.js" defer></script>
</body>
```
</details>

---

## Summary

✅ Use **WebP/AVIF** for images, provide fallbacks
✅ **Lazy load** below-fold images, eager load LCP
✅ Use **font-display: swap** and preload critical fonts
✅ **Code split** by route, load components on demand
✅ **Minify** code and enable **Brotli/gzip** compression
✅ Use a **CDN** for static assets with proper caching

**Next:** [Critical Rendering Path](./03-critical-rendering-path.md)

---

## Further Reading

- [web.dev Images](https://web.dev/fast/#optimize-your-images)
- [web.dev Fonts](https://web.dev/font-best-practices/)
- [Webpack Code Splitting](https://webpack.js.org/guides/code-splitting/)

<!-- 
Sources Consulted:
- web.dev Performance: https://web.dev/fast/
- MDN Performance: https://developer.mozilla.org/en-US/docs/Web/Performance
-->
