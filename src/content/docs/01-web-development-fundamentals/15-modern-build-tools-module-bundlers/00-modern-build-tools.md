---
title: "Modern Build Tools & Module Bundlers"
---

# Modern Build Tools & Module Bundlers

## Overview

Modern web development relies on build tools to transform, bundle, and optimize code. From package managers to bundlers to code quality tools, this toolchain improves developer experience and production performance.

This lesson covers the essential tools every web developer needs.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-package-managers.md) | Package Managers | npm, yarn, pnpm, package.json, semver |
| [02](./02-build-tools.md) | Modern Build Tools | Vite, esbuild, Rollup, Webpack |
| [03](./03-development-workflow.md) | Development Workflow | HMR, source maps, env variables |
| [04](./04-code-quality-tools.md) | Code Quality Tools | ESLint, Prettier, Husky, lint-staged |

---

## Why Build Tools?

### The Problem

Raw browser code has limitations:

```
// Can't import npm packages
import lodash from 'lodash';  ❌

// Can't use TypeScript
const x: number = 5;  ❌

// Can't use modern CSS features everywhere
@container (min-width: 400px) { }  ❌

// Can't optimize for production
// (minification, code splitting, etc.)
```

### The Solution

Build tools bridge the gap:

```
Source Code          Build Tool           Production Code
┌────────────┐      ┌──────────┐         ┌────────────┐
│ TypeScript │ ──→  │          │  ──→    │ Minified   │
│ JSX/TSX    │      │  Vite    │         │ Bundled    │
│ Modern CSS │      │  Webpack │         │ Optimized  │
│ npm deps   │      │  Rollup  │         │ Compatible │
└────────────┘      └──────────┘         └────────────┘
```

---

## Build Tool Ecosystem

### Quick Reference

| Tool | Purpose | Speed | When to Use |
|------|---------|-------|-------------|
| **npm** | Package management | - | Always (or yarn/pnpm) |
| **Vite** | Dev server + bundler | ⚡ Fast | New projects |
| **esbuild** | Bundler/transpiler | ⚡⚡ Fastest | Build scripts |
| **Rollup** | Library bundler | Fast | npm packages |
| **Webpack** | Full bundler | Slower | Complex needs |
| **ESLint** | Linting | - | Always |
| **Prettier** | Formatting | - | Always |

### Modern Stack Example

```json
{
  "name": "modern-app",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint . --fix",
    "format": "prettier --write ."
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0"
  }
}
```

---

## Prerequisites

Before starting this lesson:
- JavaScript fundamentals
- Basic command line usage
- Understanding of modules (import/export)

---

## Start Learning

Begin with [Package Managers](./01-package-managers.md) to understand npm, yarn, and dependency management.

---

## Further Reading

- [Vite Documentation](https://vitejs.dev/)
- [npm Documentation](https://docs.npmjs.com/)
- [ESLint Documentation](https://eslint.org/)
