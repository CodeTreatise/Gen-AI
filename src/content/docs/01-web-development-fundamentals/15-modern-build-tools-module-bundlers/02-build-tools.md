---
title: "Modern Build Tools"
---

# Modern Build Tools

## Introduction

Build tools transform source code into production-ready bundles. Modern tools like Vite and esbuild are dramatically faster than older solutions, improving developer experience.

This lesson covers Vite, esbuild, Rollup, and Webpack fundamentals.

### What We'll Cover

- Vite for development and production
- esbuild for speed
- Rollup for libraries
- Webpack basics

### Prerequisites

- npm/package management
- JavaScript modules (import/export)
- Basic command line usage

---

## Why Build Tools?

### What Build Tools Do

| Task | Purpose |
|------|---------|
| **Bundling** | Combine many files into few |
| **Transpiling** | Convert TypeScript/JSX to JS |
| **Minification** | Remove whitespace, shorten names |
| **Code splitting** | Load code on demand |
| **Asset handling** | Process images, fonts, CSS |
| **Dev server** | Hot reload during development |

### Evolution of Build Tools

```
2012: Grunt (task runner)
  │
2014: Gulp (streaming tasks)
  │
2015: Webpack (bundler revolution)
  │
2020: esbuild (Go-based, 100x faster)
  │
2021: Vite (esbuild + Rollup)
  │
Now: Vite dominates new projects
```

---

## Vite

### Why Vite?

| Feature | Benefit |
|---------|---------|
| **Native ESM in dev** | No bundling needed |
| **esbuild pre-bundling** | Fast dependency handling |
| **Rollup for production** | Optimized builds |
| **HMR** | Instant updates |
| **TypeScript** | Built-in support |

### Getting Started

```bash
# Create new project
npm create vite@latest my-app

# Options: vanilla, vue, react, preact, svelte, lit, etc.

cd my-app
npm install
npm run dev
```

### Project Structure

```
my-app/
├── index.html          # Entry point
├── package.json
├── vite.config.js      # Configuration
├── src/
│   ├── main.js         # JS entry
│   └── style.css
└── public/             # Static assets
    └── favicon.ico
```

### Basic Configuration

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  root: './',                    // Project root
  base: '/',                     // Base public path
  publicDir: 'public',           // Static assets folder
  
  server: {
    port: 3000,                  // Dev server port
    open: true,                  // Open browser on start
    proxy: {
      '/api': 'http://localhost:8080'  // Proxy API requests
    }
  },
  
  build: {
    outDir: 'dist',              // Output directory
    sourcemap: true,             // Generate source maps
    minify: 'esbuild',           // Minifier (esbuild or terser)
    target: 'esnext'             // Browser target
  }
});
```

### Framework Templates

```bash
# React + TypeScript
npm create vite@latest my-app -- --template react-ts

# Vue
npm create vite@latest my-app -- --template vue

# Svelte
npm create vite@latest my-app -- --template svelte
```

### Build Commands

```bash
npm run dev      # Development server
npm run build    # Production build
npm run preview  # Preview production build locally
```

---

## esbuild

### Why esbuild?

**Written in Go, 10-100x faster than JavaScript bundlers.**

```
Bundling speed comparison (large project):
┌─────────────┬────────────┐
│ Bundler     │ Time       │
├─────────────┼────────────┤
│ esbuild     │ 0.3s       │
│ Rollup      │ 15s        │
│ Webpack     │ 45s        │
└─────────────┴────────────┘
```

### Using esbuild Directly

```bash
npm install -D esbuild
```

```bash
# Basic bundle
npx esbuild src/app.js --bundle --outfile=dist/bundle.js

# Production build
npx esbuild src/app.js --bundle --minify --outfile=dist/bundle.js

# With source maps
npx esbuild src/app.js --bundle --sourcemap --outfile=dist/bundle.js
```

### esbuild API

```javascript
// build.js
import * as esbuild from 'esbuild';

await esbuild.build({
  entryPoints: ['src/app.js'],
  bundle: true,
  minify: true,
  sourcemap: true,
  target: ['es2020'],
  outfile: 'dist/bundle.js',
  format: 'esm'
});
```

```bash
node build.js
```

### When to Use esbuild

- Build scripts and tooling
- Fast development builds
- Simple bundling needs
- Pre-bundling dependencies (what Vite does)

### esbuild Limitations

- No built-in HMR
- Limited plugin ecosystem
- Less code splitting control
- No HTML generation

---

## Rollup

### Why Rollup?

Best for **libraries** and **npm packages**:

| Feature | Benefit |
|---------|---------|
| **Tree-shaking** | Best dead code elimination |
| **Clean output** | Readable bundle code |
| **Multiple formats** | ESM, CJS, UMD, IIFE |
| **Plugin ecosystem** | Extensible |

### Basic Configuration

```javascript
// rollup.config.js
export default {
  input: 'src/index.js',
  output: [
    {
      file: 'dist/bundle.esm.js',
      format: 'esm'
    },
    {
      file: 'dist/bundle.cjs.js',
      format: 'cjs'
    },
    {
      file: 'dist/bundle.umd.js',
      format: 'umd',
      name: 'MyLibrary'
    }
  ]
};
```

### Common Plugins

```javascript
// rollup.config.js
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';

export default {
  input: 'src/index.ts',
  output: {
    file: 'dist/bundle.js',
    format: 'esm'
  },
  plugins: [
    resolve(),      // Resolve node_modules
    commonjs(),     // Convert CommonJS to ESM
    typescript(),   // Compile TypeScript
    terser()        // Minify
  ]
};
```

```bash
npm install -D @rollup/plugin-node-resolve @rollup/plugin-commonjs
npx rollup -c
```

---

## Webpack (Basics)

### When to Use Webpack

- Legacy projects already using it
- Complex loader requirements
- Need extensive plugin ecosystem
- Enterprise applications

### Basic Configuration

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
  entry: './src/index.js',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist')
  },
  mode: 'development', // or 'production'
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      },
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: 'babel-loader'
      }
    ]
  }
};
```

### Key Concepts

| Concept | Purpose |
|---------|---------|
| **Entry** | Where bundling starts |
| **Output** | Where bundle goes |
| **Loaders** | Transform files (CSS, images, etc.) |
| **Plugins** | Additional processing |
| **Mode** | Development vs production |

---

## Tool Comparison

### When to Use What

| Tool | Best For | Dev Experience |
|------|----------|----------------|
| **Vite** | Modern apps, SPAs | ⚡ Excellent |
| **esbuild** | Build scripts, tooling | Fast but basic |
| **Rollup** | Libraries, npm packages | Good for libs |
| **Webpack** | Complex legacy projects | Slower but flexible |

### Decision Flow

```
Starting new project?
    │
    ├── App/SPA → Vite
    │
    └── Library → Rollup

Have existing Webpack project?
    │
    ├── Working fine → Keep it
    │
    └── Want faster dev → Consider Vite migration
```

---

## TypeScript Support

### Vite + TypeScript

```bash
npm create vite@latest my-app -- --template vanilla-ts
```

```javascript
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  // TypeScript works out of the box
});
```

### esbuild + TypeScript

```bash
npx esbuild src/app.ts --bundle --outfile=dist/bundle.js
# esbuild handles TypeScript natively (type-stripping only)
```

### Type Checking

esbuild and Vite **strip types** but don't check them. Run tsc separately:

```json
{
  "scripts": {
    "build": "tsc --noEmit && vite build",
    "typecheck": "tsc --noEmit"
  }
}
```

---

## Hands-on Exercise

### Your Task

Create a Vite project with custom configuration:

1. Create a new Vite vanilla project
2. Add a proxy for API requests
3. Configure the build output
4. Add a simple build script

<details>
<summary>✅ Solution</summary>

```bash
npm create vite@latest build-demo -- --template vanilla
cd build-demo
npm install
```

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // Split vendor code
          vendor: []  // Add large deps here
        }
      }
    }
  }
});
```

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "build:analyze": "vite build --mode analyze"
  }
}
```

```bash
npm run dev    # Start dev server at http://localhost:3000
npm run build  # Build to dist/
```
</details>

---

## Summary

✅ **Vite** is the best choice for modern app development
✅ **esbuild** provides extreme speed for tooling and builds
✅ **Rollup** excels at library bundling with multiple formats
✅ **Webpack** remains viable for legacy and complex projects
✅ All tools support **TypeScript** (run tsc separately for type checking)
✅ Choose based on project type: app vs library vs legacy

**Next:** [Development Workflow](./03-development-workflow.md)

---

## Further Reading

- [Vite Documentation](https://vitejs.dev/)
- [esbuild Documentation](https://esbuild.github.io/)
- [Rollup Documentation](https://rollupjs.org/)
- [Webpack Documentation](https://webpack.js.org/)

<!-- 
Sources Consulted:
- Vite docs: https://vitejs.dev/
- esbuild docs: https://esbuild.github.io/
-->
