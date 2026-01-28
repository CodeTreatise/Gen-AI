---
title: "Functions & Generics"
---

# Functions & Generics

## Introduction

TypeScript's function types and generics enable you to write type-safe, reusable code. Generics are particularly powerful for creating flexible utilities and working with collections.

This lesson covers typed functions, generic functions, and generic constraints.

### What We'll Cover

- Function type annotations
- Optional and default parameters
- Function overloads
- Generic functions and types
- Generic constraints

### Prerequisites

- TypeScript basics
- JavaScript functions (arrow, regular, methods)

---

## Typed Functions

### Basic Function Types

```typescript
// Named function
function greet(name: string): string {
  return `Hello, ${name}!`;
}

// Arrow function
const add = (a: number, b: number): number => {
  return a + b;
};

// Implicit return
const multiply = (a: number, b: number): number => a * b;

// Void return
function log(message: string): void {
  console.log(message);
}
```

### Function Type Expressions

```typescript
// Define function type
type MathOperation = (a: number, b: number) => number;

// Use the type
const add: MathOperation = (a, b) => a + b;
const subtract: MathOperation = (a, b) => a - b;

// As parameter type
function calculate(
  a: number, 
  b: number, 
  operation: MathOperation
): number {
  return operation(a, b);
}

calculate(10, 5, add);      // 15
calculate(10, 5, subtract); // 5
```

### Call Signatures

```typescript
// Object with call signature
type Logger = {
  (message: string): void;
  level: string;
};

const logger: Logger = Object.assign(
  (message: string) => console.log(message),
  { level: "info" }
);

logger("Hello");       // Call it
console.log(logger.level);  // Access property
```

---

## Optional and Default Parameters

### Optional Parameters

```typescript
// Optional parameter (must come last)
function greet(name: string, greeting?: string): string {
  return `${greeting || "Hello"}, ${name}!`;
}

greet("Alice");           // "Hello, Alice!"
greet("Alice", "Hi");     // "Hi, Alice!"

// Multiple optional parameters
function createUser(
  name: string,
  email?: string,
  age?: number
): void {
  console.log(name, email, age);
}
```

### Default Parameters

```typescript
// Default value (no ? needed)
function greet(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`;
}

greet("Alice");           // "Hello, Alice!"
greet("Alice", "Hi");     // "Hi, Alice!"

// Default with destructuring
function createConfig({
  timeout = 5000,
  retries = 3,
  debug = false
} = {}): void {
  console.log({ timeout, retries, debug });
}

createConfig();                    // Uses all defaults
createConfig({ timeout: 10000 });  // Override one
```

---

## Rest Parameters

Collect remaining arguments into an array:

```typescript
// Rest parameter (must be last)
function sum(...numbers: number[]): number {
  return numbers.reduce((acc, n) => acc + n, 0);
}

sum(1, 2, 3);          // 6
sum(1, 2, 3, 4, 5);    // 15

// With other parameters
function greetAll(greeting: string, ...names: string[]): string[] {
  return names.map(name => `${greeting}, ${name}!`);
}

greetAll("Hello", "Alice", "Bob", "Charlie");
// ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
```

---

## Function Overloads

Multiple signatures for one function:

```typescript
// Overload signatures
function format(value: string): string;
function format(value: number): string;
function format(value: Date): string;

// Implementation signature
function format(value: string | number | Date): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  } else if (typeof value === "number") {
    return value.toFixed(2);
  } else {
    return value.toISOString();
  }
}

// TypeScript knows exact return types
const str = format("hello");    // string
const num = format(3.14159);    // string
const date = format(new Date()); // string
```

### When to Use Overloads

```typescript
// Better with overloads: different return types based on input
function getValue(key: "name"): string;
function getValue(key: "age"): number;
function getValue(key: "active"): boolean;
function getValue(key: string): string | number | boolean {
  const data = { name: "Alice", age: 30, active: true };
  return data[key as keyof typeof data];
}

const name = getValue("name");     // string
const age = getValue("age");       // number
const active = getValue("active"); // boolean
```

---

## Generic Functions

Generics create reusable components that work with multiple types:

### The Problem Without Generics

```typescript
// Loses type information
function firstElement(arr: any[]): any {
  return arr[0];
}

const first = firstElement([1, 2, 3]);
// first is 'any' - we lost the number type!
```

### The Solution With Generics

```typescript
// Generic function preserves type
function firstElement<T>(arr: T[]): T | undefined {
  return arr[0];
}

const first = firstElement([1, 2, 3]);     // number | undefined
const firstStr = firstElement(["a", "b"]); // string | undefined

// TypeScript infers T from the argument
```

### Generic Syntax

```typescript
function functionName<T>(param: T): T {
//                   │    │        │
//                   │    │        └── Return type uses T
//                   │    └─────────── Parameter uses T
//                   └──────────────── Type parameter
  return param;
}

// Multiple type parameters
function pair<T, U>(first: T, second: U): [T, U] {
  return [first, second];
}

const p = pair("hello", 42);  // [string, number]
```

### Common Generic Functions

```typescript
// Map function
function map<T, U>(arr: T[], fn: (item: T) => U): U[] {
  return arr.map(fn);
}

const doubled = map([1, 2, 3], n => n * 2);      // number[]
const lengths = map(["a", "bb"], s => s.length); // number[]

// Filter function
function filter<T>(arr: T[], predicate: (item: T) => boolean): T[] {
  return arr.filter(predicate);
}

const evens = filter([1, 2, 3, 4], n => n % 2 === 0);  // number[]
```

---

## Generic Constraints

Limit what types can be used with a generic:

### Using extends

```typescript
// T must have a length property
function longest<T extends { length: number }>(a: T, b: T): T {
  return a.length >= b.length ? a : b;
}

longest("hello", "hi");           // ✅ strings have length
longest([1, 2, 3], [1]);          // ✅ arrays have length
longest(10, 20);                  // ❌ Error: numbers don't have length
```

### Constraining to Specific Types

```typescript
// T must be string or number
function format<T extends string | number>(value: T): string {
  return String(value);
}

// T must be an object
function merge<T extends object, U extends object>(a: T, b: U): T & U {
  return { ...a, ...b };
}

const merged = merge({ name: "Alice" }, { age: 30 });
// { name: string; age: number }
```

### keyof Constraint

```typescript
// K must be a key of T
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const user = { name: "Alice", age: 30 };

getProperty(user, "name");  // ✅ Returns string
getProperty(user, "age");   // ✅ Returns number
getProperty(user, "email"); // ❌ Error: "email" not in user
```

---

## Generic Interfaces

```typescript
// Generic interface
interface Response<T> {
  data: T;
  status: number;
  message: string;
}

// Use with specific types
interface User {
  id: string;
  name: string;
}

const userResponse: Response<User> = {
  data: { id: "1", name: "Alice" },
  status: 200,
  message: "Success"
};

const numbersResponse: Response<number[]> = {
  data: [1, 2, 3],
  status: 200,
  message: "Success"
};
```

### Generic Type Aliases

```typescript
// Generic type alias
type Result<T, E = Error> = 
  | { success: true; value: T }
  | { success: false; error: E };

// Usage
function divide(a: number, b: number): Result<number, string> {
  if (b === 0) {
    return { success: false, error: "Division by zero" };
  }
  return { success: true, value: a / b };
}

const result = divide(10, 2);
if (result.success) {
  console.log(result.value);  // TypeScript knows it's number
} else {
  console.log(result.error);  // TypeScript knows it's string
}
```

---

## Generic Classes

```typescript
class Stack<T> {
  private items: T[] = [];
  
  push(item: T): void {
    this.items.push(item);
  }
  
  pop(): T | undefined {
    return this.items.pop();
  }
  
  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }
  
  isEmpty(): boolean {
    return this.items.length === 0;
  }
}

// Type-safe stack
const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
const n = numberStack.pop();  // number | undefined

const stringStack = new Stack<string>();
stringStack.push("hello");
const s = stringStack.pop();  // string | undefined
```

---

## Default Type Parameters

```typescript
// Default to string if not specified
interface Container<T = string> {
  value: T;
}

const strContainer: Container = { value: "hello" };      // T is string
const numContainer: Container<number> = { value: 42 };   // T is number

// Default with constraint
type EventHandler<T extends Event = Event> = (event: T) => void;

const handler: EventHandler = (e) => console.log(e);
const clickHandler: EventHandler<MouseEvent> = (e) => console.log(e.clientX);
```

---

## Hands-on Exercise

### Your Task

Create these generic utilities:

```typescript
// 1. A generic identity function
// identity(5) returns 5 (type: number)
// identity("hi") returns "hi" (type: string)

// 2. A generic function that returns the last element of an array
// last([1, 2, 3]) returns 3 (type: number | undefined)

// 3. A generic function that creates a pair/tuple
// makePair("name", 42) returns ["name", 42] (type: [string, number])

// 4. A generic function that safely accesses object properties
// getProperty({ a: 1 }, "a") returns 1
// getProperty({ a: 1 }, "b") should be a compile error
```

<details>
<summary>✅ Solution</summary>

```typescript
// 1. Identity function
function identity<T>(value: T): T {
  return value;
}

const num = identity(5);       // number
const str = identity("hello"); // string

// 2. Last element
function last<T>(arr: T[]): T | undefined {
  return arr[arr.length - 1];
}

const lastNum = last([1, 2, 3]);     // number | undefined
const lastStr = last(["a", "b"]);   // string | undefined

// 3. Make pair
function makePair<T, U>(first: T, second: U): [T, U] {
  return [first, second];
}

const pair = makePair("name", 42);  // [string, number]

// 4. Get property (with keyof constraint)
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const obj = { a: 1, b: "hello" };
const a = getProperty(obj, "a");  // number
const b = getProperty(obj, "b");  // string
// getProperty(obj, "c");         // ❌ Error!
```
</details>

---

## Summary

✅ **Type annotations** on parameters and return types
✅ **Optional parameters** with `?`, **default parameters** with `=`
✅ **Rest parameters** collect remaining arguments: `...args: T[]`
✅ **Function overloads** for different input/output combinations
✅ **Generics** preserve type information: `function fn<T>(x: T): T`
✅ **Constraints** limit generic types: `<T extends SomeType>`
✅ **keyof constraint** for safe property access
✅ Generic interfaces, types, and classes for reusable structures

**Next:** [TypeScript Configuration](./04-typescript-configuration.md)

---

## Further Reading

- [Functions](https://www.typescriptlang.org/docs/handbook/2/functions.html)
- [Generics](https://www.typescriptlang.org/docs/handbook/2/generics.html)

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
-->
