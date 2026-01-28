---
title: "TypeScript Essentials"
---

# TypeScript Essentials

## Overview

TypeScript adds **static typing** to JavaScript. It catches errors at compile time instead of runtime, provides better IDE support, and makes large codebases more maintainable. TypeScript is JavaScript with superpowers.

This lesson covers TypeScript fundamentals for JavaScript developers—enough to start using TypeScript in your projects immediately.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-type-system-basics.md) | Type System Basics | Primitives, arrays, unions, type inference |
| [02](./02-interfaces-type-aliases.md) | Interfaces & Type Aliases | Object shapes, extending, optional properties |
| [03](./03-functions-generics.md) | Functions & Generics | Typed functions, generic types, constraints |
| [04](./04-typescript-configuration.md) | TypeScript Configuration | tsconfig.json, strict mode, compilation |
| [05](./05-advanced-patterns.md) | Advanced Patterns | Type guards, utility types, discriminated unions |

---

## Why TypeScript?

### The Problem with JavaScript

```javascript
// JavaScript - no errors until runtime
function add(a, b) {
  return a + b;
}

add(5, "3");        // Returns "53" (string concatenation!)
add({ x: 1 });      // Returns "[object Object]undefined"
```

### TypeScript Solution

```typescript
// TypeScript - errors caught immediately
function add(a: number, b: number): number {
  return a + b;
}

add(5, "3");        // ❌ Error: Argument of type 'string' is not assignable
add({ x: 1 });      // ❌ Error: Expected 2 arguments, but got 1
add(5, 3);          // ✅ Returns 8
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Catch errors early** | Compile-time vs runtime errors |
| **Better IDE support** | Autocomplete, refactoring, navigation |
| **Self-documenting** | Types serve as documentation |
| **Safer refactoring** | Compiler catches breaking changes |
| **Gradual adoption** | Add to existing JS projects incrementally |

---

## Quick Start

```bash
# Install TypeScript
npm install -D typescript

# Initialize config
npx tsc --init

# Compile
npx tsc
```

---

## Prerequisites

Before starting this lesson:
- Solid JavaScript knowledge
- Understanding of ES6+ features (arrow functions, destructuring, modules)
- Node.js installed

---

## Start Learning

Begin with [Type System Basics](./01-type-system-basics.md) to learn TypeScript's core type system.

---

## Further Reading

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [TypeScript Playground](https://www.typescriptlang.org/play)
- [Total TypeScript](https://www.totaltypescript.com/)
