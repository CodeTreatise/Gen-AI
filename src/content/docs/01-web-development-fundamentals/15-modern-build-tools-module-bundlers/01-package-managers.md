---
title: "Package Managers"
---

# Package Managers

## Introduction

**Package managers** handle downloading, installing, and managing third-party dependencies. They also provide scripts for building, testing, and running your project.

This lesson covers npm, yarn, pnpm, and essential package.json configuration.

### What We'll Cover

- npm fundamentals
- yarn and pnpm alternatives
- package.json structure
- Semantic versioning (semver)
- npm scripts

### Prerequisites

- Command line basics
- Node.js installed
- JavaScript fundamentals

---

## npm (Node Package Manager)

### Installing Packages

```bash
# Install a dependency (saved to dependencies)
npm install lodash

# Install dev dependency (saved to devDependencies)
npm install --save-dev typescript

# Shorthand
npm i lodash
npm i -D typescript

# Install globally (for CLI tools)
npm install -g serve

# Install all dependencies from package.json
npm install
```

### Removing Packages

```bash
npm uninstall lodash
npm un lodash  # Shorthand
```

### Updating Packages

```bash
# Check for outdated packages
npm outdated

# Update within semver range
npm update

# Update to latest (may break things)
npm install lodash@latest
```

---

## yarn

**yarn** was created by Facebook as a faster, more reliable alternative.

### yarn Commands

```bash
# Install (equivalent to npm install)
yarn
yarn install

# Add package
yarn add lodash
yarn add -D typescript  # Dev dependency

# Remove package
yarn remove lodash

# Run scripts
yarn dev
yarn build
```

### yarn vs npm

| Feature | npm | yarn |
|---------|-----|------|
| Speed | Good | Faster |
| Lock file | package-lock.json | yarn.lock |
| Workspaces | Supported | Better support |
| Plug'n'Play | No | Yes (optional) |

---

## pnpm

**pnpm** uses hard links to save disk space and is faster than npm/yarn.

### pnpm Commands

```bash
# Install
pnpm install
pnpm i

# Add package
pnpm add lodash
pnpm add -D typescript

# Remove package
pnpm remove lodash

# Run scripts
pnpm dev
pnpm run build
```

### Why pnpm?

```
npm/yarn:
project-a/node_modules/lodash (10MB)
project-b/node_modules/lodash (10MB)
project-c/node_modules/lodash (10MB)
Total: 30MB

pnpm:
~/.pnpm-store/lodash (10MB) ← Single copy
project-a/node_modules/lodash → link to store
project-b/node_modules/lodash → link to store
project-c/node_modules/lodash → link to store
Total: 10MB
```

---

## package.json Structure

```json
{
  "name": "my-app",
  "version": "1.0.0",
  "description": "My web application",
  "main": "dist/index.js",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint .",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "typescript": "^5.0.0",
    "eslint": "^8.0.0"
  },
  "engines": {
    "node": ">=18.0.0"
  },
  "license": "MIT"
}
```

### Key Fields

| Field | Purpose |
|-------|---------|
| `name` | Package name (required for publishing) |
| `version` | Current version |
| `type` | `"module"` for ES modules |
| `main` | Entry point for CommonJS |
| `module` | Entry point for ES modules |
| `scripts` | Runnable commands |
| `dependencies` | Production dependencies |
| `devDependencies` | Development-only dependencies |
| `engines` | Required Node.js version |

---

## Semantic Versioning (semver)

### Version Format

```
MAJOR.MINOR.PATCH
  │     │     │
  │     │     └── Bug fixes (backward compatible)
  │     └──────── New features (backward compatible)
  └────────────── Breaking changes
```

### Examples

| Change | From | To | Type |
|--------|------|-----|------|
| Bug fix | 1.2.3 | 1.2.4 | Patch |
| New feature | 1.2.3 | 1.3.0 | Minor |
| Breaking change | 1.2.3 | 2.0.0 | Major |

### Version Ranges

| Syntax | Meaning | Example |
|--------|---------|---------|
| `^1.2.3` | Minor + patch updates | 1.2.3 → 1.9.9 ✓, 2.0.0 ✗ |
| `~1.2.3` | Patch updates only | 1.2.3 → 1.2.9 ✓, 1.3.0 ✗ |
| `1.2.3` | Exact version | Only 1.2.3 |
| `>=1.2.3` | Any version ≥1.2.3 | 2.0.0 ✓ |
| `*` | Any version | All versions ✓ |

### Lock Files

Lock files ensure consistent installs across machines:

```
package.json:    "lodash": "^4.17.0"
                       ↓
Lock file:       lodash@4.17.21
                 (exact version locked)
```

| Package Manager | Lock File |
|-----------------|-----------|
| npm | package-lock.json |
| yarn | yarn.lock |
| pnpm | pnpm-lock.yaml |

> **Important:** Always commit lock files to version control.

---

## npm Scripts

### Common Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "start": "node dist/server.js",
    "test": "vitest",
    "test:watch": "vitest --watch",
    "lint": "eslint . --fix",
    "format": "prettier --write .",
    "typecheck": "tsc --noEmit"
  }
}
```

### Running Scripts

```bash
npm run dev
npm run build

# Special scripts (no 'run' needed)
npm start
npm test
```

### Pre/Post Hooks

```json
{
  "scripts": {
    "prebuild": "npm run lint",
    "build": "vite build",
    "postbuild": "npm run test"
  }
}
```

Running `npm run build` executes: prebuild → build → postbuild

### Chaining Commands

```json
{
  "scripts": {
    "validate": "npm run lint && npm run test && npm run build",
    "dev:full": "npm run build && npm run dev"
  }
}
```

---

## npx

**npx** runs packages without installing globally:

```bash
# Run without global install
npx create-vite my-app
npx serve dist
npx eslint --init

# Run specific version
npx typescript@5.0.0 --version

# Run from npm registry
npx cowsay "Hello"
```

---

## Workspaces (Monorepos)

### npm Workspaces

```json
{
  "name": "my-monorepo",
  "workspaces": [
    "packages/*"
  ]
}
```

```
my-monorepo/
├── package.json
├── packages/
│   ├── shared/
│   │   └── package.json
│   ├── web/
│   │   └── package.json
│   └── api/
│       └── package.json
```

```bash
# Run command in specific workspace
npm run build -w packages/web

# Run command in all workspaces
npm run build --workspaces
```

---

## Best Practices

### Dependency Management

| Practice | Reason |
|----------|--------|
| Lock exact versions for production | Reproducible builds |
| Use `^` for development | Get patches/features |
| Audit regularly | Security vulnerabilities |
| Keep dependencies updated | Bug fixes, performance |

```bash
# Security audit
npm audit
npm audit fix

# Interactive update tool
npx npm-check-updates -i
```

### Script Organization

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "build:analyze": "vite build --mode analyze",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "lint": "eslint .",
    "lint:fix": "eslint . --fix"
  }
}
```

---

## Hands-on Exercise

### Your Task

Set up a new project with proper package management:

1. Create a new project folder
2. Initialize package.json
3. Install vite as a dev dependency
4. Add useful npm scripts
5. Create a basic index.html

<details>
<summary>✅ Solution</summary>

```bash
mkdir my-project && cd my-project

npm init -y

npm install -D vite

# Edit package.json to add scripts
```

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  }
}
```

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
  <title>My Project</title>
</head>
<body>
  <h1>Hello Vite!</h1>
  <script type="module" src="/main.js"></script>
</body>
</html>
```

```javascript
// main.js
console.log('Hello from Vite!');
```

```bash
npm run dev
```
</details>

---

## Summary

✅ **npm** is the default package manager, yarn and pnpm are alternatives
✅ **package.json** defines dependencies and scripts
✅ **Semver** uses MAJOR.MINOR.PATCH versioning
✅ **Lock files** ensure reproducible installations—commit them
✅ **npm scripts** automate development tasks
✅ **npx** runs packages without global installation
✅ **Workspaces** enable monorepo setups

**Next:** [Modern Build Tools](./02-build-tools.md)

---

## Further Reading

- [npm Documentation](https://docs.npmjs.com/)
- [yarn Documentation](https://yarnpkg.com/)
- [pnpm Documentation](https://pnpm.io/)
- [Semver Specification](https://semver.org/)

<!-- 
Sources Consulted:
- npm docs: https://docs.npmjs.com/
- yarn docs: https://yarnpkg.com/
-->
