---
title: "Development Workflow"
---

# Development Workflow

## Introduction

Modern build tools provide features that dramatically improve the development experience: instant updates, debugging with original source code, and environment-specific configuration.

This lesson covers HMR, source maps, environment variables, and development vs production builds.

### What We'll Cover

- Hot Module Replacement (HMR)
- Source maps for debugging
- Environment variables
- Development vs production modes

### Prerequisites

- Build tools basics (Vite/Webpack)
- JavaScript debugging experience
- Basic understanding of bundling

---

## Hot Module Replacement (HMR)

### What Is HMR?

**Hot Module Replacement** updates code in the browser without a full page reload:

```
Traditional refresh:
Edit → Save → Full reload → Lose state → Navigate back

HMR:
Edit → Save → Module updates → State preserved → Instant
```

### HMR in Vite

Vite provides HMR out of the box:

```javascript
// main.js
import './style.css';  // CSS HMR works automatically
import { counter } from './counter.js';

// Accept module updates
if (import.meta.hot) {
  import.meta.hot.accept('./counter.js', (newModule) => {
    console.log('Counter module updated');
  });
}
```

### CSS HMR

CSS updates instantly without configuration:

```css
/* style.css - changes apply immediately */
.button {
  background: blue;  /* Change this, see it instantly */
}
```

### Framework HMR

Frameworks handle HMR automatically:

```jsx
// React with Vite - HMR works automatically
function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <button onClick={() => setCount(c => c + 1)}>
      Count: {count}  {/* Edit text, state preserved */}
    </button>
  );
}
```

### HMR API

```javascript
// Full HMR API in Vite
if (import.meta.hot) {
  // Accept updates to this module
  import.meta.hot.accept();
  
  // Accept updates to dependencies
  import.meta.hot.accept(['./dep1.js', './dep2.js'], ([mod1, mod2]) => {
    // Handle updates
  });
  
  // Cleanup before update
  import.meta.hot.dispose((data) => {
    // Save state, cleanup intervals, etc.
    data.savedState = currentState;
  });
  
  // Access data from previous version
  const previousData = import.meta.hot.data;
}
```

---

## Source Maps

### What Are Source Maps?

Source maps connect bundled/minified code back to original source:

```
Browser sees:          Source map points to:
┌──────────────────┐   ┌──────────────────┐
│ function a(b){   │   │ function         │
│ return b+1}      │ → │ addOne(num) {    │
│                  │   │   return num + 1;│
│                  │   │ }                │
└──────────────────┘   └──────────────────┘
  bundle.min.js          src/math.js:15
```

### Source Map Types

| Type | Size | Detail | Use Case |
|------|------|--------|----------|
| `source-map` | Large | Full | Production debugging |
| `cheap-source-map` | Medium | Lines only | Faster builds |
| `inline-source-map` | Larger | Embedded | Simple debugging |
| `hidden-source-map` | Large | No link | Secure production |
| `nosources-source-map` | Small | No source | Error tracking |

### Vite Source Maps

```javascript
// vite.config.js
export default defineConfig({
  build: {
    sourcemap: true,         // Generate source maps
    // sourcemap: 'hidden',  // Generate but don't link
  }
});
```

### Debugging with Source Maps

1. Open DevTools (F12)
2. Go to Sources panel
3. Find original files under `webpack://` or file paths
4. Set breakpoints in original TypeScript/JSX
5. Debug as if running original source

### Source Map Security

For production, consider:

```javascript
// Don't expose source maps publicly
export default defineConfig({
  build: {
    sourcemap: 'hidden'  // Generate but don't link in bundle
  }
});
```

Upload source maps to error tracking (Sentry, etc.) separately.

---

## Environment Variables

### Why Environment Variables?

Different values for different environments:

```
Development:          Production:
API_URL=localhost     API_URL=api.example.com
DEBUG=true            DEBUG=false
```

### Vite Environment Variables

```bash
# .env - all environments
VITE_APP_NAME=MyApp

# .env.local - local overrides (gitignored)
VITE_API_KEY=dev-key-123

# .env.development - dev only
VITE_API_URL=http://localhost:3000

# .env.production - production only
VITE_API_URL=https://api.example.com
```

### Using Environment Variables

```javascript
// In code (Vite requires VITE_ prefix)
const apiUrl = import.meta.env.VITE_API_URL;
const appName = import.meta.env.VITE_APP_NAME;
const mode = import.meta.env.MODE;  // 'development' or 'production'
const isDev = import.meta.env.DEV;  // true in development
const isProd = import.meta.env.PROD;  // true in production
```

### TypeScript Support

```typescript
// vite-env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_NAME: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### Loading Order

```
.env                # Always loaded
.env.local          # Always loaded, gitignored
.env.[mode]         # Only in specified mode
.env.[mode].local   # Only in specified mode, gitignored
```

Later files override earlier ones.

### Webpack Environment Variables

```javascript
// webpack.config.js
const webpack = require('webpack');

module.exports = {
  plugins: [
    new webpack.DefinePlugin({
      'process.env.API_URL': JSON.stringify(process.env.API_URL)
    })
  ]
};
```

---

## Development vs Production

### Mode Differences

| Aspect | Development | Production |
|--------|-------------|------------|
| **Speed** | Fast rebuilds | Optimized output |
| **Bundle** | Unbundled/minimal | Fully bundled |
| **Minification** | None | Full |
| **Source maps** | Inline | External/none |
| **HMR** | Enabled | Disabled |
| **Errors** | Detailed | Generic |

### Vite Modes

```bash
# Development (default for 'dev')
vite --mode development

# Production (default for 'build')
vite build --mode production

# Custom mode
vite --mode staging
```

### Conditional Code

```javascript
// This code is removed in production builds
if (import.meta.env.DEV) {
  console.log('Debug info:', data);
  window.__DEBUG_DATA__ = data;
}

// Production-only
if (import.meta.env.PROD) {
  initAnalytics();
}
```

### Build Optimization

```javascript
// vite.config.js
export default defineConfig({
  build: {
    // Minification
    minify: 'esbuild',  // or 'terser'
    
    // Target browsers
    target: 'es2020',
    
    // Code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          utils: ['lodash-es', 'date-fns']
        }
      }
    },
    
    // Chunk size warnings
    chunkSizeWarningLimit: 500  // KB
  }
});
```

### Analyzing Bundle

```bash
# Install analyzer
npm install -D rollup-plugin-visualizer

# vite.config.js
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    visualizer({
      open: true,
      gzipSize: true
    })
  ]
});
```

---

## Development Server Features

### Proxy Configuration

```javascript
// vite.config.js
export default defineConfig({
  server: {
    proxy: {
      // Simple proxy
      '/api': 'http://localhost:8080',
      
      // With options
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      
      // WebSocket proxy
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true
      }
    }
  }
});
```

### HTTPS in Development

```javascript
// vite.config.js
export default defineConfig({
  server: {
    https: true  // Uses self-signed cert
  }
});

// Or with custom certs
import fs from 'fs';

export default defineConfig({
  server: {
    https: {
      key: fs.readFileSync('localhost-key.pem'),
      cert: fs.readFileSync('localhost.pem')
    }
  }
});
```

### File System Access

```javascript
// vite.config.js
export default defineConfig({
  server: {
    fs: {
      // Allow serving files outside project root
      allow: ['..']
    }
  }
});
```

---

## Hands-on Exercise

### Your Task

Set up a project with environment-specific configuration:

1. Create .env files for development and production
2. Use environment variables in code
3. Add a proxy for API requests
4. Test both development and production builds

<details>
<summary>✅ Solution</summary>

```bash
# .env
VITE_APP_NAME=MyApp
VITE_APP_VERSION=1.0.0
```

```bash
# .env.development
VITE_API_URL=http://localhost:8080/api
VITE_DEBUG=true
```

```bash
# .env.production
VITE_API_URL=https://api.myapp.com
VITE_DEBUG=false
```

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  },
  build: {
    sourcemap: true
  }
});
```

```javascript
// main.js
const config = {
  appName: import.meta.env.VITE_APP_NAME,
  apiUrl: import.meta.env.VITE_API_URL,
  debug: import.meta.env.VITE_DEBUG === 'true'
};

console.log('App Config:', config);
console.log('Mode:', import.meta.env.MODE);

if (import.meta.env.DEV) {
  console.log('Running in development mode');
}

async function fetchData() {
  const response = await fetch(`${config.apiUrl}/users`);
  return response.json();
}
```

```bash
# Test
npm run dev    # Uses .env.development
npm run build  # Uses .env.production
npm run preview  # Preview production build
```
</details>

---

## Summary

✅ **HMR** provides instant updates while preserving state
✅ **Source maps** enable debugging with original source code
✅ **Environment variables** configure different environments (VITE_ prefix)
✅ **Development mode** prioritizes speed and debugging
✅ **Production mode** prioritizes optimization and bundle size
✅ **Proxy configuration** simplifies API development
✅ Use conditional code (`import.meta.env.DEV`) for debug-only features

**Next:** [Code Quality Tools](./04-code-quality-tools.md)

---

## Further Reading

- [Vite HMR API](https://vitejs.dev/guide/api-hmr.html)
- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)
- [Source Maps Explained](https://web.dev/source-maps/)

<!-- 
Sources Consulted:
- Vite docs: https://vitejs.dev/guide/
- web.dev source maps: https://web.dev/source-maps/
-->
