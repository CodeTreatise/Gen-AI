---
title: "Interfaces & Type Aliases"
---

# Interfaces & Type Aliases

## Introduction

Interfaces and type aliases define custom types for objects and complex data structures. They're essential for typing APIs, component props, and domain models.

This lesson covers when to use each and how to compose complex types.

### What We'll Cover

- Type aliases for custom types
- Interfaces for object shapes
- Optional and readonly properties
- Extending and composing types
- Type alias vs interface differences

### Prerequisites

- Basic TypeScript types
- Understanding of objects in JavaScript

---

## Type Aliases

Type aliases create named types:

```typescript
// Simple alias
type ID = string | number;

// Object type alias
type User = {
  id: ID;
  name: string;
  email: string;
};

// Function type alias
type Callback = (data: string) => void;

// Using the types
const userId: ID = "abc123";
const user: User = {
  id: "abc123",
  name: "Alice",
  email: "alice@example.com"
};
```

### Type Alias Syntax

```typescript
type TypeName = TypeDefinition;
     │           │
     │           └── What the type represents
     └────────────── Name for the type (PascalCase)
```

### Type Aliases for Unions

```typescript
// Status values
type Status = "pending" | "approved" | "rejected";

// API response
type ApiResponse<T> = {
  data: T;
  status: Status;
} | {
  error: string;
  status: "error";
};

// Event types
type MouseEvent = "click" | "dblclick" | "mouseenter" | "mouseleave";
type KeyboardEvent = "keydown" | "keyup" | "keypress";
type UIEvent = MouseEvent | KeyboardEvent;
```

---

## Interfaces

Interfaces define the shape of objects:

```typescript
interface User {
  id: string;
  name: string;
  email: string;
}

const user: User = {
  id: "123",
  name: "Alice",
  email: "alice@example.com"
};

// Missing property = error
const incomplete: User = {
  id: "123",
  name: "Alice"
  // ❌ Error: Property 'email' is missing
};

// Extra property = error (in some cases)
const extra: User = {
  id: "123",
  name: "Alice",
  email: "alice@example.com",
  age: 30  // ❌ Error in object literal
};
```

---

## Optional Properties

Properties that may not exist:

```typescript
interface Config {
  apiUrl: string;        // Required
  timeout?: number;      // Optional
  retries?: number;      // Optional
  debug?: boolean;       // Optional
}

// All valid:
const config1: Config = {
  apiUrl: "https://api.example.com"
};

const config2: Config = {
  apiUrl: "https://api.example.com",
  timeout: 5000
};

const config3: Config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
  retries: 3,
  debug: true
};
```

### Working with Optional Properties

```typescript
interface User {
  name: string;
  nickname?: string;
}

function greet(user: User): string {
  // user.nickname is string | undefined
  if (user.nickname) {
    return `Hi, ${user.nickname}!`;
  }
  return `Hello, ${user.name}`;
}

// Using nullish coalescing
function greet2(user: User): string {
  const displayName = user.nickname ?? user.name;
  return `Hello, ${displayName}!`;
}
```

---

## Readonly Properties

Properties that can't be changed after creation:

```typescript
interface User {
  readonly id: string;   // Can't change after creation
  name: string;          // Can change
  email: string;         // Can change
}

const user: User = {
  id: "123",
  name: "Alice",
  email: "alice@example.com"
};

user.name = "Bob";       // ✅ OK
user.id = "456";         // ❌ Error: Cannot assign to 'id'
```

### Readonly Arrays and Objects

```typescript
// Readonly array
const numbers: readonly number[] = [1, 2, 3];
numbers.push(4);     // ❌ Error
numbers[0] = 10;     // ❌ Error

// ReadonlyArray type
const names: ReadonlyArray<string> = ["Alice", "Bob"];

// Readonly object
type Point = Readonly<{
  x: number;
  y: number;
}>;

const origin: Point = { x: 0, y: 0 };
origin.x = 10;       // ❌ Error
```

---

## Extending Interfaces

Build on existing interfaces:

```typescript
interface Person {
  name: string;
  email: string;
}

// Extend with additional properties
interface Employee extends Person {
  employeeId: string;
  department: string;
}

// Employee has: name, email, employeeId, department
const employee: Employee = {
  name: "Alice",
  email: "alice@company.com",
  employeeId: "E123",
  department: "Engineering"
};

// Extend multiple interfaces
interface Timestamps {
  createdAt: Date;
  updatedAt: Date;
}

interface Document extends Person, Timestamps {
  content: string;
}
```

### Overriding Properties

```typescript
interface Animal {
  name: string;
  legs: number;
}

interface Dog extends Animal {
  legs: 4;  // Narrow to specific value
  breed: string;
}

const dog: Dog = {
  name: "Max",
  legs: 4,    // Must be exactly 4
  breed: "Labrador"
};
```

---

## Index Signatures

Dynamic property names:

```typescript
// String keys, string values
interface Dictionary {
  [key: string]: string;
}

const colors: Dictionary = {
  red: "#ff0000",
  green: "#00ff00",
  blue: "#0000ff"
};

// Numeric keys
interface StringArray {
  [index: number]: string;
}

const arr: StringArray = ["a", "b", "c"];

// Mixed: specific + dynamic properties
interface Config {
  name: string;              // Required
  version: string;           // Required
  [key: string]: string;     // Any additional string properties
}
```

---

## Type Alias vs Interface

### Similarities

Both can define object shapes:

```typescript
// Type alias
type UserType = {
  id: string;
  name: string;
};

// Interface
interface UserInterface {
  id: string;
  name: string;
}

// Both work the same for objects
const user1: UserType = { id: "1", name: "Alice" };
const user2: UserInterface = { id: "2", name: "Bob" };
```

### Key Differences

| Feature | Type Alias | Interface |
|---------|------------|-----------|
| Object shapes | ✅ | ✅ |
| Union types | ✅ `type A = B \| C` | ❌ |
| Intersection | ✅ `type A = B & C` | ✅ `extends` |
| Primitives | ✅ `type ID = string` | ❌ |
| Declaration merging | ❌ | ✅ |
| Extends/implements | ✅ (with &) | ✅ |

### Declaration Merging (Interfaces Only)

```typescript
// Interfaces with same name merge
interface User {
  name: string;
}

interface User {
  email: string;
}

// Result: User has both name and email
const user: User = {
  name: "Alice",
  email: "alice@example.com"
};

// Type aliases can't merge - error if same name
type Person = { name: string };
type Person = { email: string };  // ❌ Error: Duplicate identifier
```

### When to Use Which

```typescript
// Use INTERFACE for:
// - Object shapes (APIs, props, models)
// - When you might need to extend
// - Library type definitions (allows merging)

interface ApiResponse {
  data: unknown;
  status: number;
}

interface UserResponse extends ApiResponse {
  data: User;
}

// Use TYPE ALIAS for:
// - Unions
// - Primitives
// - Tuples
// - Complex compositions

type ID = string | number;
type Status = "pending" | "complete" | "failed";
type Coordinates = [number, number];
type Handler = (event: Event) => void;
```

---

## Composing Types

### Combining with Intersection

```typescript
type WithTimestamps = {
  createdAt: Date;
  updatedAt: Date;
};

type WithId = {
  id: string;
};

// Combine types
type Entity = WithId & WithTimestamps;

// Use in interface
interface User extends Entity {
  name: string;
  email: string;
}

// User now has: id, createdAt, updatedAt, name, email
```

### Mapped Types

```typescript
// Make all properties optional
type Partial<T> = {
  [P in keyof T]?: T[P];
};

// Make all properties required
type Required<T> = {
  [P in keyof T]-?: T[P];
};

// Make all properties readonly
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

// Usage
interface User {
  name: string;
  email: string;
}

type PartialUser = Partial<User>;
// { name?: string; email?: string; }

type ReadonlyUser = Readonly<User>;
// { readonly name: string; readonly email: string; }
```

---

## Function Properties

```typescript
interface Calculator {
  // Method syntax
  add(a: number, b: number): number;
  
  // Property syntax (same thing)
  subtract: (a: number, b: number) => number;
  
  // Optional method
  divide?(a: number, b: number): number;
}

const calc: Calculator = {
  add(a, b) {
    return a + b;
  },
  subtract: (a, b) => a - b
  // divide is optional
};
```

---

## Hands-on Exercise

### Your Task

Design types for a blog post system:

```typescript
// Create these types:
// 1. Author with id, name, email, and optional bio
// 2. Post with id, title, content, author, tags (string array), 
//    published (boolean), and timestamps
// 3. Draft that extends Post but published is always false
// 4. Comment with id, postId, author, content, and createdAt
```

<details>
<summary>✅ Solution</summary>

```typescript
// Base timestamp type
type Timestamps = {
  createdAt: Date;
  updatedAt: Date;
};

// Author interface
interface Author {
  id: string;
  name: string;
  email: string;
  bio?: string;
}

// Post interface
interface Post extends Timestamps {
  id: string;
  title: string;
  content: string;
  author: Author;
  tags: string[];
  published: boolean;
}

// Draft - published is always false
interface Draft extends Omit<Post, 'published'> {
  published: false;
}

// Or using type alias:
type DraftAlt = Post & { published: false };

// Comment interface
interface Comment {
  id: string;
  postId: string;
  author: Author;
  content: string;
  createdAt: Date;
}

// Usage
const author: Author = {
  id: "a1",
  name: "Alice",
  email: "alice@blog.com",
  bio: "Tech writer"
};

const draft: Draft = {
  id: "p1",
  title: "TypeScript Tips",
  content: "...",
  author,
  tags: ["typescript", "tutorial"],
  published: false,
  createdAt: new Date(),
  updatedAt: new Date()
};
```
</details>

---

## Summary

✅ **Type aliases** create named types for any type expression
✅ **Interfaces** define object shapes with excellent tooling support
✅ **Optional properties** use `?`: `name?: string`
✅ **Readonly properties** prevent modification: `readonly id: string`
✅ **Extend interfaces** to build on existing types
✅ **Use interfaces** for objects, **type aliases** for unions/primitives
✅ Interfaces support **declaration merging**, type aliases don't

**Next:** [Functions & Generics](./03-functions-generics.md)

---

## Further Reading

- [Object Types](https://www.typescriptlang.org/docs/handbook/2/objects.html)
- [Type vs Interface](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#differences-between-type-aliases-and-interfaces)

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
-->
