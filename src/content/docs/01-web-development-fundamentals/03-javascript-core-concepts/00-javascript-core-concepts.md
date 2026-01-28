---
title: "JavaScript Core Concepts - Overview"
---

# JavaScript Core Concepts - Overview

Modern JavaScript is the foundation for building AI-powered web applications. This lesson covers the core language features you'll use daily: variables, control flow, functions, scope, objects, data structures, arrays, destructuring, modules, and cutting-edge features like Web Components and Signals.

## Lesson Structure

This lesson is divided into 11 sub-lessons, each focusing on a specific aspect of JavaScript:

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01. Variables and Data Types](./01-variables-data-types.md) | var, let, const, primitives, type coercion | TDZ, hoisting, strict equality |
| [02. Control Structures](./02-control-structures.md) | if/else, switch, loops | Ternary operator, break/continue |
| [03. Functions](./03-functions.md) | Declarations, expressions, arrow functions | HOF, callbacks, parameters |
| [04. Scope and Closures](./04-scope-closures.md) | Lexical scope, closures, module pattern | Private variables, memory |
| [05. Objects and Prototypes](./05-objects-prototypes.md) | Object literals, classes, inheritance | Prototype chain, Symbol, BigInt |
| [06. Data Structures](./06-data-structures.md) | Map, Set, WeakMap, WeakSet | structuredClone, collections |
| [07. Array Methods](./07-arrays-methods.md) | map, filter, reduce, forEach | Method chaining, immutability |
| [08. Destructuring and Spread](./08-destructuring-spread.md) | Array/object destructuring, spread operator | Rest parameters, defaults |
| [09. Modules](./09-modules.md) | import/export, dynamic imports | ES Modules vs CommonJS |
| [10. Web Components](./10-web-components.md) | Custom Elements, Shadow DOM | Templates, slots, lifecycle |
| [11. Signals](./11-signals.md) | TC39 Signals proposal | Reactive state management |

## Learning Path

### Prerequisites
- Basic programming concepts (variables, functions, loops)
- HTML and CSS fundamentals
- Code editor (VS Code recommended)

### Recommended Order
1. Start with **Variables and Data Types** to understand JavaScript's type system
2. Master **Control Structures** for program flow
3. Deep dive into **Functions** and **Scope/Closures** together
4. Learn **Objects and Prototypes** for OOP patterns
5. Explore **Data Structures** and **Array Methods** for data manipulation
6. Practice **Destructuring and Spread** for modern syntax
7. Understand **Modules** for code organization
8. Explore **Web Components** for reusable UI elements
9. Preview **Signals** for future reactive patterns

### Time Estimate
- **Total**: ~8-10 hours for all 11 lessons
- **Per lesson**: 45-60 minutes (reading + exercises)

## Key Takeaways

By the end of this lesson, you will:
- ✅ Understand JavaScript's variable declarations and scoping rules
- ✅ Write functions using modern syntax (arrow functions, destructuring)
- ✅ Master closures and leverage them for data privacy
- ✅ Use ES6 classes and understand prototype-based inheritance
- ✅ Work with modern data structures (Map, Set, WeakMap)
- ✅ Transform arrays using functional methods (map, filter, reduce)
- ✅ Organize code with ES6 modules
- ✅ Create reusable Web Components
- ✅ Understand future JavaScript features (Signals)

## Next Steps

After completing this lesson:
- Move to [DOM Manipulation](../04-dom-manipulation.md) to interact with web pages
- Explore [Asynchronous JavaScript](../05-asynchronous-javascript.md) for API calls
- Apply these concepts in [HTTP & API Communication](../06-http-api-communication.md)

---

## Original Outline (For Reference)

<details>
<summary>Expand original outline</summary>

- Variables, data types, and operators
  - var, let, const differences
  - Hoisting behavior
  - Primitive types (string, number, boolean, null, undefined, symbol, bigint)
  - Reference types (objects, arrays, functions)
  - Type coercion and conversion
  - Comparison operators (== vs ===)
  - Logical operators (&&, ||, ??)
  - Optional chaining (?.)
  - Nullish coalescing (??)
- Control structures (conditionals, loops)
  - if/else statements
  - Switch statements
  - Ternary operator
  - for, while, do-while loops
  - for...of and for...in loops
  - break and continue statements
  - Labeled statements
- Functions (declarations, expressions, arrow functions)
  - Function declarations vs expressions
  - Arrow function syntax
  - Arrow functions and 'this' binding
  - Default parameters
  - Rest parameters (...args)
  - Function hoisting
  - Immediately Invoked Function Expressions (IIFE)
  - Higher-order functions
  - Callback functions
- Scope and closures
  - Global scope
  - Function scope
  - Block scope (let, const)
  - Lexical scoping
  - Closure definition and use cases
  - Memory considerations with closures
  - Module pattern using closures
- Objects and prototypes
  - Object literals
  - Property access (dot vs bracket notation)
  - Computed property names
  - Object methods and 'this'
  - Object.keys, Object.values, Object.entries
  - Object.assign and spread for objects
  - Prototype chain and prototypal inheritance
  - Object.create and Object.getPrototypeOf
  - Constructor functions
  - ES6 classes
  - Symbol type and well-known symbols
  - BigInt for large integers
- Advanced data structures
  - WeakMap and WeakSet
  - Map vs Object differences
  - Set operations and use cases
  - structuredClone() for deep cloning
  - Typed Arrays overview
- Arrays and array methods (map, filter, reduce, forEach)
  - Array creation and access
  - map: transforming arrays
  - filter: selecting elements
  - reduce: accumulating values
  - forEach: iteration without return
  - find and findIndex
  - some and every
  - includes and indexOf
  - slice and splice
  - sort and reverse
  - flat and flatMap
  - Chaining array methods
- Destructuring and spread operators
  - Array destructuring
  - Object destructuring
  - Default values in destructuring
  - Nested destructuring
  - Spread operator for arrays
  - Spread operator for objects
  - Rest pattern in destructuring
  - Practical use cases
- Modules (import/export)
  - Named exports
  - Default exports
  - Import syntax variations
  - Re-exporting
  - Dynamic imports (import())
  - Module scope
  - Circular dependencies
  - CommonJS vs ES Modules
- Web Components fundamentals
  - Custom Elements API (`customElements.define()`)
  - `HTMLElement` extension pattern
  - Lifecycle callbacks (connectedCallback, disconnectedCallback)
  - Shadow DOM for encapsulation
  - `attachShadow()` and shadow root
  - Slotted content with `<slot>`
  - CSS `::part()` for external styling
  - HTML `<template>` element for reusable markup
  - Autonomous vs customized built-in elements
- Signals (emerging standard)
  - TC39 Signals proposal overview
  - Reactive primitives concept
  - Signal, Computed, Effect patterns
  - Framework implementations (Solid, Preact, Angular)
  - Future of reactive JavaScript

</details>

---

**Navigation:**
- **Previous:** [CSS Fundamentals](../02-css-fundamentals/00-css-fundamentals.md)
- **Next:** [Start with Variables and Data Types](./01-variables-data-types.md)
- **Back to Unit:** [Web Development Fundamentals](../00-overview.md)