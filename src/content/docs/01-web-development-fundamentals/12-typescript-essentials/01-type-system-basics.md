---
title: "Type System Basics"
---

# Type System Basics

## Introduction

TypeScript's type system is the foundation of everything. Understanding basic types, type inference, and unions will make you productive immediately.

This lesson covers the core type annotations and how TypeScript infers types automatically.

### What We'll Cover

- Primitive types (string, number, boolean)
- Arrays and tuples
- Union and intersection types
- Type inference
- Special types (any, unknown, void, never)

### Prerequisites

- JavaScript fundamentals
- ES6+ syntax familiarity

---

## Primitive Types

### Basic Type Annotations

```typescript
// String
let name: string = "Alice";
let greeting: string = `Hello, ${name}`;

// Number (integers and floats)
let age: number = 30;
let price: number = 19.99;
let hex: number = 0xff;

// Boolean
let isActive: boolean = true;
let hasPermission: boolean = false;
```

### Type Annotation Syntax

```typescript
let variableName: Type = value;
    │             │      │
    │             │      └── Value (must match type)
    │             └──────── Type annotation
    └────────────────────── Variable name
```

---

## Arrays

### Array Type Syntax

```typescript
// Method 1: Type[]
let numbers: number[] = [1, 2, 3, 4, 5];
let names: string[] = ["Alice", "Bob", "Charlie"];

// Method 2: Array<Type> (generic syntax)
let scores: Array<number> = [95, 87, 92];
let tags: Array<string> = ["javascript", "typescript"];

// Mixed types with union
let mixed: (string | number)[] = [1, "two", 3, "four"];
```

### Accessing Array Elements

```typescript
const numbers: number[] = [1, 2, 3];

const first: number = numbers[0];        // 1
const last: number = numbers[numbers.length - 1];  // 3

// TypeScript knows array methods return correct types
const doubled: number[] = numbers.map(n => n * 2);
const sum: number = numbers.reduce((a, b) => a + b, 0);
```

---

## Tuples

Tuples are fixed-length arrays with specific types at each position:

```typescript
// Tuple: exactly [string, number]
let person: [string, number] = ["Alice", 30];

// Accessing tuple elements
const name: string = person[0];  // "Alice"
const age: number = person[1];   // 30

// Error: wrong type at position
person = [30, "Alice"];  // ❌ Error

// Error: wrong length
person = ["Alice"];      // ❌ Error

// Common use: return multiple values
function getNameAndAge(): [string, number] {
  return ["Alice", 30];
}

const [userName, userAge] = getNameAndAge();
```

### Named Tuples (TypeScript 4.0+)

```typescript
type Point = [x: number, y: number];
type RGB = [red: number, green: number, blue: number];

const point: Point = [10, 20];
const color: RGB = [255, 128, 0];
```

---

## Type Inference

TypeScript infers types when you don't annotate:

```typescript
// TypeScript infers these automatically
let message = "Hello";        // string
let count = 42;               // number
let active = true;            // boolean
let items = [1, 2, 3];        // number[]
let user = { name: "Alice" }; // { name: string }

// Inference from functions
const double = (n: number) => n * 2;  // Return type inferred as number

// Inference from context
const names = ["Alice", "Bob"];
names.forEach(name => {
  console.log(name.toUpperCase());  // name is inferred as string
});
```

### When to Annotate

```typescript
// ✅ Don't annotate obvious types
const PI = 3.14159;           // number (obvious)
const users = [];             // ❌ any[] - needs annotation!

// ✅ Annotate when inference isn't enough
const users: User[] = [];     // Now TypeScript knows the element type

// ✅ Annotate function parameters (required)
function greet(name: string) {
  return `Hello, ${name}`;
}

// ✅ Annotate when value starts as null
let data: string | null = null;
data = fetchData();
```

---

## Union Types

A value can be one of several types:

```typescript
// Union of primitives
let id: string | number;
id = "abc123";  // ✅ OK
id = 42;        // ✅ OK
id = true;      // ❌ Error

// Common pattern: nullable types
let username: string | null = null;
username = "Alice";  // ✅ OK

// Union in function parameters
function printId(id: string | number): void {
  console.log(`ID: ${id}`);
}

printId("abc");   // ✅ OK
printId(123);     // ✅ OK
```

### Type Narrowing

TypeScript narrows union types based on checks:

```typescript
function printId(id: string | number): void {
  // Type is string | number here
  
  if (typeof id === "string") {
    // Type narrowed to string
    console.log(id.toUpperCase());
  } else {
    // Type narrowed to number
    console.log(id.toFixed(2));
  }
}

// Array type narrowing
function printAll(strs: string | string[]): void {
  if (Array.isArray(strs)) {
    // Type: string[]
    strs.forEach(s => console.log(s));
  } else {
    // Type: string
    console.log(strs);
  }
}
```

### Common Narrowing Techniques

| Check | Narrows To |
|-------|------------|
| `typeof x === "string"` | string |
| `typeof x === "number"` | number |
| `Array.isArray(x)` | array type |
| `x === null` | null |
| `x !== undefined` | excludes undefined |
| `"property" in x` | type with that property |
| `x instanceof Class` | that class type |

---

## Intersection Types

Combine multiple types (must have ALL properties):

```typescript
type HasName = { name: string };
type HasAge = { age: number };

// Intersection: must have BOTH name AND age
type Person = HasName & HasAge;

const person: Person = {
  name: "Alice",
  age: 30
};

// Missing property = error
const incomplete: Person = {
  name: "Bob"
  // ❌ Error: Property 'age' is missing
};
```

### Union vs Intersection

```typescript
// Union: A OR B
type StringOrNumber = string | number;
let x: StringOrNumber = "hello";  // ✅
let y: StringOrNumber = 42;       // ✅

// Intersection: A AND B
type Named = { name: string };
type Aged = { age: number };
type Person = Named & Aged;

const person: Person = {
  name: "Alice",  // Required from Named
  age: 30         // Required from Aged
};
```

---

## Special Types

### any

Escape hatch—disables type checking:

```typescript
let flexible: any = "hello";
flexible = 42;           // ✅ No error
flexible = { x: 1 };     // ✅ No error
flexible.foo.bar.baz;    // ✅ No error (but crashes at runtime!)

// ⚠️ Avoid any when possible!
```

### unknown

Safer alternative to `any`:

```typescript
let userInput: unknown = getData();

// Can't use unknown directly
userInput.toUpperCase();  // ❌ Error

// Must narrow first
if (typeof userInput === "string") {
  userInput.toUpperCase();  // ✅ OK after type check
}
```

### void

Functions that don't return a value:

```typescript
function log(message: string): void {
  console.log(message);
  // No return statement
}

// void is different from undefined
function explicit(): void {
  return;  // OK
}

function alsoVoid(): void {
  return undefined;  // OK
}
```

### never

Values that never occur:

```typescript
// Function that never returns
function throwError(message: string): never {
  throw new Error(message);
}

// Infinite loop
function infiniteLoop(): never {
  while (true) {}
}

// Exhaustive checking
type Shape = "circle" | "square";

function getArea(shape: Shape): number {
  switch (shape) {
    case "circle":
      return Math.PI * 10 * 10;
    case "square":
      return 10 * 10;
    default:
      // If we add a new shape, TypeScript errors here
      const exhaustiveCheck: never = shape;
      throw new Error(`Unhandled shape: ${exhaustiveCheck}`);
  }
}
```

---

## Literal Types

Exact values as types:

```typescript
// String literal types
let direction: "north" | "south" | "east" | "west";
direction = "north";  // ✅ OK
direction = "up";     // ❌ Error

// Number literal types
let diceRoll: 1 | 2 | 3 | 4 | 5 | 6;
diceRoll = 3;   // ✅ OK
diceRoll = 7;   // ❌ Error

// Boolean literal
let alwaysTrue: true = true;
alwaysTrue = false;  // ❌ Error

// Common pattern: status values
type Status = "pending" | "approved" | "rejected";

function updateStatus(status: Status) {
  // Only these three values allowed
}
```

### const Assertions

```typescript
// Without const assertion
let config = { env: "production" };
// Type: { env: string }

// With const assertion
let config = { env: "production" } as const;
// Type: { readonly env: "production" }

// Arrays
const colors = ["red", "green", "blue"] as const;
// Type: readonly ["red", "green", "blue"]
```

---

## Hands-on Exercise

### Your Task

Fix the type errors in this code:

```typescript
// 1. Fix the array type
let scores = [];
scores.push(95);
scores.push("A");  // Should this be allowed?

// 2. Create a proper union type
let result;  // Can be number or error string
result = 42;
result = "Error: not found";

// 3. Create a tuple for coordinates
let point;
point = [10, 20];  // [x, y]

// 4. Narrow this union properly
function formatValue(value: string | number) {
  return value.toUpperCase();  // Error!
}
```

<details>
<summary>✅ Solution</summary>

```typescript
// 1. Decide: number[] only or mixed?
let scores: number[] = [];
scores.push(95);
// scores.push("A");  // Remove or change type to (string | number)[]

// 2. Union type for result
let result: number | string;
result = 42;
result = "Error: not found";

// 3. Tuple type for coordinates
let point: [number, number];
point = [10, 20];

// 4. Narrow with typeof
function formatValue(value: string | number): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  } else {
    return value.toString();
  }
}
```
</details>

---

## Summary

✅ **Primitive types**: `string`, `number`, `boolean`
✅ **Arrays**: `Type[]` or `Array<Type>`
✅ **Tuples**: Fixed-length arrays with specific types
✅ **Type inference**: TypeScript infers when possible
✅ **Union types**: `A | B` for "either/or"
✅ **Intersection types**: `A & B` for "both"
✅ **Narrowing**: Use `typeof`, `instanceof`, `in` to refine types
✅ Avoid `any`, prefer `unknown` when type is truly unknown

**Next:** [Interfaces & Type Aliases](./02-interfaces-type-aliases.md)

---

## Further Reading

- [TypeScript Basic Types](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html)
- [Narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html)

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
-->
