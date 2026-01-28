---
title: "Advanced Patterns"
---

# Advanced Patterns

## Introduction

TypeScript's advanced features enable powerful patterns for type safety and code reuse. Type guards, discriminated unions, and utility types help you build robust, maintainable applications.

This lesson covers patterns you'll encounter and use in real-world TypeScript projects.

### What We'll Cover

- Type guards and narrowing
- Discriminated unions
- Utility types (Partial, Pick, Omit, etc.)
- Template literal types
- Conditional types

### Prerequisites

- TypeScript basics
- Generics understanding

---

## Type Guards

Type guards narrow types at runtime:

### typeof Guards

```typescript
function format(value: string | number): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  } else {
    return value.toFixed(2);
  }
}
```

### instanceof Guards

```typescript
class Dog {
  bark() { console.log("Woof!"); }
}

class Cat {
  meow() { console.log("Meow!"); }
}

function speak(animal: Dog | Cat): void {
  if (animal instanceof Dog) {
    animal.bark();  // TypeScript knows it's Dog
  } else {
    animal.meow();  // TypeScript knows it's Cat
  }
}
```

### in Operator

```typescript
type Fish = { swim: () => void };
type Bird = { fly: () => void };

function move(animal: Fish | Bird): void {
  if ("swim" in animal) {
    animal.swim();  // Fish
  } else {
    animal.fly();   // Bird
  }
}
```

### Custom Type Guards

```typescript
// Type predicate: value is string
function isString(value: unknown): value is string {
  return typeof value === "string";
}

// Use the type guard
function process(value: unknown): void {
  if (isString(value)) {
    console.log(value.toUpperCase());  // TypeScript knows it's string
  }
}

// Complex type guard
interface User {
  name: string;
  email: string;
}

function isUser(obj: unknown): obj is User {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "name" in obj &&
    "email" in obj &&
    typeof (obj as User).name === "string" &&
    typeof (obj as User).email === "string"
  );
}

// Use it
function greetUser(data: unknown): void {
  if (isUser(data)) {
    console.log(`Hello, ${data.name}!`);  // Safe
  }
}
```

---

## Discriminated Unions

Unions with a shared "discriminant" property:

### Basic Pattern

```typescript
// Each type has a unique "type" property
type Success = {
  type: "success";
  data: string;
};

type Error = {
  type: "error";
  message: string;
};

type Loading = {
  type: "loading";
};

type State = Success | Error | Loading;

function handleState(state: State): string {
  switch (state.type) {
    case "success":
      return state.data;       // TypeScript knows it's Success
    case "error":
      return state.message;    // TypeScript knows it's Error
    case "loading":
      return "Loading...";     // TypeScript knows it's Loading
  }
}
```

### API Response Pattern

```typescript
type ApiResponse<T> =
  | { status: "success"; data: T }
  | { status: "error"; error: string }
  | { status: "loading" };

async function fetchUser(): Promise<ApiResponse<User>> {
  try {
    const response = await fetch("/api/user");
    const data = await response.json();
    return { status: "success", data };
  } catch (e) {
    return { status: "error", error: String(e) };
  }
}

// Using the response
const response = await fetchUser();

if (response.status === "success") {
  console.log(response.data.name);  // Safe access to data
} else if (response.status === "error") {
  console.error(response.error);    // Safe access to error
}
```

### Exhaustive Checking

```typescript
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; size: number }
  | { kind: "rectangle"; width: number; height: number };

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "square":
      return shape.size ** 2;
    case "rectangle":
      return shape.width * shape.height;
    default:
      // If you add a new shape, TypeScript errors here
      const _exhaustive: never = shape;
      throw new Error(`Unhandled shape: ${_exhaustive}`);
  }
}
```

---

## Utility Types

TypeScript provides built-in utility types:

### Partial<T>

Make all properties optional:

```typescript
interface User {
  name: string;
  email: string;
  age: number;
}

type PartialUser = Partial<User>;
// { name?: string; email?: string; age?: number }

// Useful for updates
function updateUser(id: string, updates: Partial<User>): void {
  // Can pass any subset of User properties
}

updateUser("123", { name: "Alice" });  // Only update name
```

### Required<T>

Make all properties required:

```typescript
interface Config {
  apiUrl?: string;
  timeout?: number;
}

type RequiredConfig = Required<Config>;
// { apiUrl: string; timeout: number }
```

### Pick<T, K>

Select specific properties:

```typescript
interface User {
  id: string;
  name: string;
  email: string;
  password: string;
}

type PublicUser = Pick<User, "id" | "name" | "email">;
// { id: string; name: string; email: string }
// (password excluded)
```

### Omit<T, K>

Remove specific properties:

```typescript
interface User {
  id: string;
  name: string;
  email: string;
  password: string;
}

type UserWithoutPassword = Omit<User, "password">;
// { id: string; name: string; email: string }
```

### Record<K, T>

Create object type with specific keys:

```typescript
type Status = "pending" | "active" | "completed";

type StatusCounts = Record<Status, number>;
// { pending: number; active: number; completed: number }

const counts: StatusCounts = {
  pending: 5,
  active: 10,
  completed: 20
};

// Dynamic keys
type StringMap = Record<string, string>;
const headers: StringMap = {
  "Content-Type": "application/json",
  "Authorization": "Bearer token"
};
```

### Readonly<T>

Make all properties readonly:

```typescript
interface User {
  name: string;
  email: string;
}

type ImmutableUser = Readonly<User>;

const user: ImmutableUser = { name: "Alice", email: "a@b.com" };
user.name = "Bob";  // ❌ Error: Cannot assign to 'name'
```

### Exclude and Extract

```typescript
type Status = "pending" | "active" | "completed" | "archived";

// Exclude: remove types from union
type ActiveStatus = Exclude<Status, "archived">;
// "pending" | "active" | "completed"

// Extract: keep only matching types
type FinishedStatus = Extract<Status, "completed" | "archived">;
// "completed" | "archived"
```

### NonNullable

Remove null and undefined:

```typescript
type MaybeString = string | null | undefined;
type DefinitelyString = NonNullable<MaybeString>;
// string
```

### ReturnType and Parameters

```typescript
function createUser(name: string, age: number) {
  return { id: Date.now(), name, age };
}

// Get return type of function
type UserType = ReturnType<typeof createUser>;
// { id: number; name: string; age: number }

// Get parameter types
type CreateUserParams = Parameters<typeof createUser>;
// [name: string, age: number]
```

---

## Template Literal Types

Create types from string patterns:

```typescript
// Basic template literal type
type Greeting = `Hello, ${string}!`;

const g1: Greeting = "Hello, Alice!";  // ✅
const g2: Greeting = "Hi, Bob!";       // ❌ Error

// Union expansion
type Color = "red" | "green" | "blue";
type Shade = "light" | "dark";

type ColorVariant = `${Shade}-${Color}`;
// "light-red" | "light-green" | "light-blue" | "dark-red" | ...

// Event handlers
type EventName = "click" | "focus" | "blur";
type Handler = `on${Capitalize<EventName>}`;
// "onClick" | "onFocus" | "onBlur"

// Practical example: CSS properties
type CSSProperty = "margin" | "padding";
type Direction = "top" | "right" | "bottom" | "left";
type BoxProperty = `${CSSProperty}-${Direction}`;
// "margin-top" | "margin-right" | ... | "padding-left"
```

### Intrinsic String Manipulation

```typescript
type Uppercased = Uppercase<"hello">;    // "HELLO"
type Lowercased = Lowercase<"HELLO">;    // "hello"
type Capitalized = Capitalize<"hello">;  // "Hello"
type Uncapitalized = Uncapitalize<"Hello">; // "hello"
```

---

## Conditional Types

Types that depend on conditions:

```typescript
// Basic conditional type
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;  // true
type B = IsString<number>;  // false

// Practical: unwrap arrays
type Unwrap<T> = T extends Array<infer U> ? U : T;

type C = Unwrap<string[]>;  // string
type D = Unwrap<number>;    // number

// Unwrap promises
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

type E = UnwrapPromise<Promise<string>>;  // string
type F = UnwrapPromise<number>;           // number

// Filter union types
type ExtractStrings<T> = T extends string ? T : never;

type Mixed = string | number | boolean;
type OnlyStrings = ExtractStrings<Mixed>;  // string
```

### infer Keyword

Extract types from complex structures:

```typescript
// Get array element type
type ArrayElement<T> = T extends (infer E)[] ? E : never;

type NumElement = ArrayElement<number[]>;  // number

// Get function return type (simplified ReturnType)
type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type Ret = MyReturnType<() => string>;  // string

// Get first argument type
type FirstArg<T> = T extends (first: infer F, ...rest: any[]) => any ? F : never;

type First = FirstArg<(name: string, age: number) => void>;  // string
```

---

## Mapped Types

Transform types systematically:

```typescript
// Make all properties optional (how Partial works)
type MyPartial<T> = {
  [K in keyof T]?: T[K];
};

// Make all properties readonly
type MyReadonly<T> = {
  readonly [K in keyof T]: T[K];
};

// Make all properties nullable
type Nullable<T> = {
  [K in keyof T]: T[K] | null;
};

// Remove readonly modifier
type Mutable<T> = {
  -readonly [K in keyof T]: T[K];
};

// Usage
interface User {
  readonly id: string;
  name: string;
}

type MutableUser = Mutable<User>;
// { id: string; name: string; }  // id is no longer readonly
```

### Key Remapping

```typescript
// Prefix all keys
type Prefixed<T> = {
  [K in keyof T as `prefix_${string & K}`]: T[K];
};

interface User {
  name: string;
  age: number;
}

type PrefixedUser = Prefixed<User>;
// { prefix_name: string; prefix_age: number }

// Filter keys
type OnlyStringKeys<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};
```

---

## Hands-on Exercise

### Your Task

Create these utility types and use them:

```typescript
// 1. Create a DeepPartial type that makes all nested properties optional

// 2. Create a type that extracts all keys with string values

// 3. Create a discriminated union for different notification types
//    (success, error, warning) with appropriate properties

// 4. Create a type-safe event emitter type
```

<details>
<summary>✅ Solution</summary>

```typescript
// 1. DeepPartial
type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

interface User {
  name: string;
  settings: {
    theme: string;
    notifications: {
      email: boolean;
      push: boolean;
    };
  };
}

type PartialUser = DeepPartial<User>;
// All nested properties are optional

// 2. Extract string keys
type StringKeys<T> = {
  [K in keyof T]: T[K] extends string ? K : never;
}[keyof T];

interface Mixed {
  name: string;
  age: number;
  email: string;
}

type StringProps = StringKeys<Mixed>;  // "name" | "email"

// 3. Discriminated union for notifications
type Notification =
  | { type: "success"; message: string; duration?: number }
  | { type: "error"; message: string; code: number }
  | { type: "warning"; message: string; dismissible: boolean };

function showNotification(notification: Notification): void {
  switch (notification.type) {
    case "success":
      console.log(`✅ ${notification.message}`);
      break;
    case "error":
      console.log(`❌ Error ${notification.code}: ${notification.message}`);
      break;
    case "warning":
      console.log(`⚠️ ${notification.message}`);
      break;
  }
}

// 4. Type-safe event emitter
type EventMap = {
  click: { x: number; y: number };
  focus: { target: HTMLElement };
  submit: { data: FormData };
};

type EventEmitter<T extends Record<string, unknown>> = {
  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): void;
  emit<K extends keyof T>(event: K, data: T[K]): void;
};

// Usage
declare const emitter: EventEmitter<EventMap>;

emitter.on("click", (data) => {
  console.log(data.x, data.y);  // TypeScript knows data is { x, y }
});

emitter.emit("submit", { data: new FormData() });
```
</details>

---

## Summary

✅ **Type guards** narrow types at runtime (`typeof`, `instanceof`, `in`)
✅ **Custom type guards** use `value is Type` predicates
✅ **Discriminated unions** use a shared property for type narrowing
✅ **Utility types**: `Partial`, `Required`, `Pick`, `Omit`, `Record`
✅ **Template literal types** create string pattern types
✅ **Conditional types** enable type-level logic with `extends ? :`
✅ **Mapped types** transform existing types systematically

**Back to:** [TypeScript Essentials Overview](./00-typescript-essentials.md)

---

## Further Reading

- [Narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html)
- [Utility Types](https://www.typescriptlang.org/docs/handbook/utility-types.html)
- [Template Literal Types](https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html)

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
-->
