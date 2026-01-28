---
title: "TypeScript Configuration"
---

# TypeScript Configuration

## Introduction

The `tsconfig.json` file controls how TypeScript compiles your code. Understanding these options is essential for setting up projects correctly and enabling strict type checking.

This lesson covers essential tsconfig options, strict mode, and build workflows.

### What We'll Cover

- tsconfig.json structure
- Essential compiler options
- Strict mode and its benefits
- Compiling and building
- Type declaration files

### Prerequisites

- TypeScript basics
- npm/Node.js experience

---

## Getting Started

### Installing TypeScript

```bash
# Per project (recommended)
npm install --save-dev typescript

# Initialize tsconfig.json
npx tsc --init

# Check version
npx tsc --version
```

### Basic tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

---

## File Structure

### tsconfig.json Structure

```json
{
  "compilerOptions": {
    // How to compile
  },
  "include": [
    // Files to include
  ],
  "exclude": [
    // Files to exclude
  ],
  "extends": "./tsconfig.base.json",  // Inherit from another config
  "references": []  // Project references
}
```

### Include and Exclude

```json
{
  "include": [
    "src/**/*",           // All files in src
    "types/**/*.d.ts"     // Type definitions
  ],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts",       // Test files
    "**/*.spec.ts"
  ]
}
```

---

## Essential Compiler Options

### Target and Module

```json
{
  "compilerOptions": {
    // JavaScript version to output
    "target": "ES2020",
    
    // Module system
    "module": "ESNext",
    
    // Module resolution strategy
    "moduleResolution": "bundler"  // For Vite, Webpack
    // "moduleResolution": "node"  // For Node.js
  }
}
```

| target | Features Available |
|--------|-------------------|
| ES5 | Basic, needs polyfills |
| ES2015 | Classes, arrow functions, Promises |
| ES2020 | Optional chaining, nullish coalescing |
| ESNext | Latest features |

| module | Use Case |
|--------|----------|
| CommonJS | Node.js (require/exports) |
| ESNext | Modern bundlers (import/export) |
| NodeNext | Node.js with ES modules |

### Output Options

```json
{
  "compilerOptions": {
    // Output directory
    "outDir": "./dist",
    
    // Source directory (for cleaner output structure)
    "rootDir": "./src",
    
    // Generate source maps for debugging
    "sourceMap": true,
    
    // Generate .d.ts type declaration files
    "declaration": true,
    
    // Remove comments from output
    "removeComments": true
  }
}
```

### Module Interop

```json
{
  "compilerOptions": {
    // Allow default imports from CommonJS
    "esModuleInterop": true,
    
    // Ensure consistent imports
    "allowSyntheticDefaultImports": true,
    
    // Skip type checking of declaration files
    "skipLibCheck": true
  }
}
```

---

## Strict Mode

### Enable All Strict Checks

```json
{
  "compilerOptions": {
    "strict": true
  }
}
```

### What Strict Mode Enables

| Option | What It Does |
|--------|--------------|
| `noImplicitAny` | Error on implicit `any` type |
| `strictNullChecks` | `null` and `undefined` are distinct types |
| `strictFunctionTypes` | Stricter function type checking |
| `strictBindCallApply` | Check `bind`, `call`, `apply` |
| `strictPropertyInitialization` | Class properties must be initialized |
| `noImplicitThis` | Error on `this` with implicit `any` |
| `alwaysStrict` | Emit `"use strict"` |
| `useUnknownInCatchVariables` | `catch` variables are `unknown` |

### Examples of Strict Checks

```typescript
// noImplicitAny
function greet(name) {        // ❌ Parameter 'name' implicitly has 'any' type
  return `Hello, ${name}`;
}

function greet(name: string) { // ✅ Fixed
  return `Hello, ${name}`;
}

// strictNullChecks
let name: string = null;      // ❌ Type 'null' is not assignable to 'string'
let name: string | null = null; // ✅ Fixed

// strictPropertyInitialization
class User {
  name: string;  // ❌ Property 'name' has no initializer
  
  constructor() {}
}

class User {
  name: string;  // ✅ Fixed
  
  constructor(name: string) {
    this.name = name;
  }
}
```

> **Important:** Always use `"strict": true` for new projects!

---

## Additional Type Checking

```json
{
  "compilerOptions": {
    // Strict mode
    "strict": true,
    
    // Additional checks
    "noUnusedLocals": true,           // Error on unused variables
    "noUnusedParameters": true,       // Error on unused parameters
    "noImplicitReturns": true,        // All code paths must return
    "noFallthroughCasesInSwitch": true, // No fallthrough in switch
    "noUncheckedIndexedAccess": true, // Array access may be undefined
    "exactOptionalPropertyTypes": true // Strict optional property handling
  }
}
```

### noUncheckedIndexedAccess

```typescript
const arr = [1, 2, 3];

// Without noUncheckedIndexedAccess
const first = arr[0];  // number

// With noUncheckedIndexedAccess
const first = arr[0];  // number | undefined
// Forces you to handle possible undefined
```

---

## Path Mapping

### Absolute Imports

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"]
    }
  }
}
```

```typescript
// Instead of
import { Button } from '../../../components/Button';

// Use
import { Button } from '@components/Button';
```

> **Note:** Bundlers (Vite, Webpack) need matching alias config.

---

## Compiling TypeScript

### Basic Commands

```bash
# Compile using tsconfig.json
npx tsc

# Watch mode (recompile on changes)
npx tsc --watch

# Compile specific file
npx tsc src/app.ts

# Type check only (no output)
npx tsc --noEmit
```

### npm Scripts

```json
{
  "scripts": {
    "build": "tsc",
    "build:watch": "tsc --watch",
    "typecheck": "tsc --noEmit",
    "clean": "rm -rf dist"
  }
}
```

### With Build Tools

Modern bundlers handle TypeScript directly:

```json
// package.json with Vite
{
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "typecheck": "tsc --noEmit"
  }
}
```

---

## Type Declaration Files (.d.ts)

### What Are Declaration Files?

Declaration files provide type information for JavaScript libraries:

```typescript
// lodash.d.ts (simplified)
declare module 'lodash' {
  export function chunk<T>(array: T[], size: number): T[][];
  export function compact<T>(array: T[]): T[];
  // ...
}
```

### Installing Types

```bash
# Most libraries have types in @types
npm install --save-dev @types/node
npm install --save-dev @types/express
npm install --save-dev @types/lodash

# Some libraries include types (no @types needed)
npm install axios  # Types included
npm install zod    # Types included
```

### DefinitelyTyped

[DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped) hosts community type definitions:

```bash
# Search for types
npm search @types/library-name
```

### Writing Custom Declarations

```typescript
// types/custom.d.ts
declare module 'untyped-library' {
  export function doSomething(value: string): number;
  export const VERSION: string;
}

// Global declarations
declare global {
  interface Window {
    myApp: {
      version: string;
      init(): void;
    };
  }
}

export {};  // Make it a module
```

---

## Project References

For monorepos and large projects:

```json
// tsconfig.json (root)
{
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/client" },
    { "path": "./packages/server" }
  ]
}

// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist"
  }
}
```

```bash
# Build all referenced projects
npx tsc --build
```

---

## Common Configurations

### React Project

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "esModuleInterop": true
  },
  "include": ["src"]
}
```

### Node.js Project

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "skipLibCheck": true,
    "esModuleInterop": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

### Library

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "skipLibCheck": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## Hands-on Exercise

### Your Task

Create a TypeScript project configuration:

1. Initialize a new npm project
2. Install TypeScript
3. Create tsconfig.json with:
   - Strict mode enabled
   - ES2020 target
   - ESNext modules
   - Source maps
   - Output to `dist/`
   - Path alias `@/` for `src/`
4. Add build scripts

<details>
<summary>✅ Solution</summary>

```bash
mkdir ts-project && cd ts-project
npm init -y
npm install --save-dev typescript
```

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "sourceMap": true,
    "declaration": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },
    "noUnusedLocals": true,
    "noUnusedParameters": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

```json
// package.json (partial)
{
  "scripts": {
    "build": "tsc",
    "build:watch": "tsc --watch",
    "typecheck": "tsc --noEmit",
    "clean": "rm -rf dist"
  }
}
```

```bash
mkdir src
echo 'export const greet = (name: string): string => `Hello, ${name}!`;' > src/index.ts
npm run build
```
</details>

---

## Summary

✅ **tsconfig.json** controls TypeScript compilation
✅ Use **strict: true** for maximum type safety
✅ **target** sets output JavaScript version
✅ **module** sets module system (ESNext for bundlers)
✅ **noEmit** for type checking only (with bundlers)
✅ **@types** packages provide types for JS libraries
✅ **Path aliases** simplify imports
✅ Use **--watch** for development

**Next:** [Advanced Patterns](./05-advanced-patterns.md)

---

## Further Reading

- [TSConfig Reference](https://www.typescriptlang.org/tsconfig)
- [TypeScript Compiler Options](https://www.typescriptlang.org/docs/handbook/compiler-options.html)

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
- TSConfig Reference: https://www.typescriptlang.org/tsconfig
-->
