---
title: "Variables and Data Types"
---

# Variables and Data Types

## Introduction

JavaScript's approach to variables and types sets the foundation for everything we build. Understanding how `let`, `const`, and `var` differ—especially around scoping and the temporal dead zone—prevents bugs that plague even experienced developers. When building AI interfaces, proper variable management becomes critical: we're handling API responses, managing conversation state, and processing user input in real time.

This lesson covers JavaScript's type system from primitives to type coercion behavior, giving you the knowledge to write predictable code that handles AI data reliably.

### What We'll Cover
- `var`, `let`, and `const` declarations with scope and hoisting behavior
- JavaScript's primitive data types and their characteristics
- Type coercion rules and how to handle implicit conversions
- Temporal Dead Zone (TDZ) and why it matters
- Operators and their type-specific behaviors

### Prerequisites
- Basic programming concepts (variables, types)
- HTML/JavaScript file setup
- Browser developer console access

---

## Variable Declarations: var, let, and const

JavaScript provides three ways to declare variables, each with different scoping rules and behaviors.

### The Legacy: var

Before ES2015, `var` was the only way to declare variables. It has **function scope** (or global scope if declared outside functions):

```javascript
function showVarScope() {
  var x = 1;
  if (true) {
    var x = 2;  // Same variable!
    console.log(x);  // 2
  }
  console.log(x);  // 2 - changed by the if block
}

showVarScope();
```

**Output:**
```
2
2
```

`var` declarations are **hoisted** to the top of their function scope and initialized with `undefined`:

```javascript
console.log(message);  // undefined (not ReferenceError!)
var message = "Hello";
console.log(message);  // "Hello"
```

**Output:**
```
undefined
Hello
```

> **Note:** Hoisting means declarations are processed before code execution. The variable exists throughout its scope, but holds `undefined` until the assignment line runs.

### Modern Declarations: let and const

ES2015 introduced `let` and `const`, which use **block scope** (limited to `{}` braces):

```javascript
function showBlockScope() {
  let x = 1;
  const y = 2;
  
  if (true) {
    let x = 3;      // Different variable - block-scoped
    const y = 4;    // Different variable
    console.log(x, y);  // 3 4
  }
  
  console.log(x, y);  // 1 2
}

showBlockScope();
```

**Output:**
```
3 4
1 2
```

### The Temporal Dead Zone (TDZ)

Unlike `var`, `let` and `const` variables cannot be accessed before their declaration—they're in a "temporal dead zone":

```javascript
console.log(temp);  // ReferenceError: Cannot access 'temp' before initialization
let temp = 25;

// This also applies to const
console.log(API_KEY);  // ReferenceError
const API_KEY = "sk-...";
```

The TDZ exists from the start of the block until the declaration line:

```javascript
{
  // TDZ for 'value' starts here
  console.log(value);  // ReferenceError
  let value = 42;      // TDZ ends here
  console.log(value);  // 42
}
```

> **Note:** The TDZ exists because `let`/`const` are hoisted but NOT initialized. This prevents bugs from using variables before they're properly set up.

### Redeclaration Rules

```javascript
var x = 1;
var x = 2;  // ✅ Allowed with var

let y = 1;
let y = 2;  // ❌ SyntaxError: Identifier 'y' has already been declared

const z = 1;
const z = 2;  // ❌ SyntaxError: Identifier 'z' has already been declared
```

### const: Immutable Binding, Mutable Values

`const` prevents **reassignment**, not mutation:

```javascript
const config = { apiUrl: "https://api.example.com" };
config = {};  // ❌ TypeError: Assignment to constant variable

config.apiUrl = "https://new-api.com";  // ✅ Allowed - mutating the object
console.log(config.apiUrl);  // "https://new-api.com"

const numbers = [1, 2, 3];
numbers.push(4);  // ✅ Allowed - mutating the array
console.log(numbers);  // [1, 2, 3, 4]
```

**Output:**
```
https://new-api.com
[1, 2, 3, 4]
```

---

## Primitive Data Types

JavaScript has 7 primitive types (immutable values stored directly in variables):

### 1. Number

All numeric values, including integers and floats. Uses IEEE-754 double-precision (64-bit):

```javascript
const integer = 42;
const float = 3.14159;
const negative = -273.15;
const exponential = 1.5e6;  // 1,500,000

console.log(Number.MAX_SAFE_INTEGER);  // 9007199254740991
console.log(0.1 + 0.2);  // 0.30000000000000004 (floating-point precision)
```

**Output:**
```
9007199254740991
0.30000000000000004
```

Special numeric values:

```javascript
const infinity = Infinity;
const negInfinity = -Infinity;
const notANumber = NaN;

console.log(1 / 0);  // Infinity
console.log(-1 / 0);  // -Infinity
console.log("abc" / 2);  // NaN
console.log(NaN === NaN);  // false (use Number.isNaN() instead)
```

**Output:**
```
Infinity
-Infinity
NaN
false
```

### 2. BigInt

For integers larger than `Number.MAX_SAFE_INTEGER`:

```javascript
const bigNumber = 9007199254740991n;  // Note the 'n' suffix
const calculated = BigInt(123) * BigInt(456);

console.log(bigNumber + 1n);  // 9007199254740992n
console.log(calculated);  // 56088n

// Cannot mix BigInt and Number
console.log(1n + 1);  // ❌ TypeError: Cannot mix BigInt and other types
```

**Output:**
```
9007199254740992n
56088n
TypeError: Cannot mix BigInt and other types
```

### 3. String

Text values enclosed in quotes (single, double, or backticks):

```javascript
const single = 'Hello';
const double = "World";
const template = `${single} ${double}!`;  // Template literal with interpolation

console.log(template);  // "Hello World!"
console.log(template.length);  // 12
console.log(template[0]);  // "H" (strings are array-like)
```

**Output:**
```
Hello World!
12
H
```

### 4. Boolean

Logical values `true` or `false`:

```javascript
const isActive = true;
const isComplete = false;

console.log(isActive && isComplete);  // false
console.log(isActive || isComplete);  // true
console.log(!isActive);  // false
```

**Output:**
```
false
true
false
```

### 5. undefined

Represents uninitialized variables or missing values:

```javascript
let uninitialized;
console.log(uninitialized);  // undefined

function noReturn() {}
console.log(noReturn());  // undefined

const obj = { name: "Alice" };
console.log(obj.age);  // undefined (property doesn't exist)
```

**Output:**
```
undefined
undefined
undefined
```

### 6. null

Represents intentional absence of value:

```javascript
let user = { name: "Bob", age: 30 };
user = null;  // Explicitly clear the object reference

console.log(user);  // null
console.log(typeof null);  // "object" (historical bug in JavaScript)
```

**Output:**
```
null
object
```

### 7. Symbol

Unique identifiers, often used for object properties:

```javascript
const id1 = Symbol('id');
const id2 = Symbol('id');

console.log(id1 === id2);  // false (each Symbol is unique)

const user = {
  name: "Charlie",
  [id1]: 12345  // Symbol as property key
};

console.log(user.name);  // "Charlie"
console.log(user[id1]);  // 12345
```

**Output:**
```
false
Charlie
12345
```

---

## Type Checking and typeof

The `typeof` operator returns a string indicating the type:

```javascript
console.log(typeof 42);  // "number"
console.log(typeof "text");  // "string"
console.log(typeof true);  // "boolean"
console.log(typeof undefined);  // "undefined"
console.log(typeof null);  // "object" (known bug)
console.log(typeof Symbol());  // "symbol"
console.log(typeof 123n);  // "bigint"
console.log(typeof {});  // "object"
console.log(typeof []);  // "object" (arrays are objects)
console.log(typeof function() {});  // "function"
```

**Output:**
```
number
string
boolean
undefined
object
symbol
bigint
object
object
function
```

---

## Type Coercion

JavaScript automatically converts types in certain operations (implicit coercion):

### String Coercion

```javascript
console.log("The answer is " + 42);  // "The answer is 42"
console.log("5" + 3);  // "53" (number to string)
console.log("5" - 3);  // 2 (string to number)
console.log("10" * "2");  // 20 (both to numbers)
```

**Output:**
```
The answer is 42
53
2
20
```

### Numeric Coercion

```javascript
console.log(+"42");  // 42 (unary + converts to number)
console.log(Number("123"));  // 123
console.log(parseInt("42px"));  // 42 (stops at non-digit)
console.log(parseFloat("3.14159"));  // 3.14159

console.log(true + 1);  // 2 (true becomes 1)
console.log(false + 1);  // 1 (false becomes 0)
console.log(null + 1);  // 1 (null becomes 0)
console.log(undefined + 1);  // NaN (undefined becomes NaN)
```

**Output:**
```
42
123
42
3.14159
2
1
1
NaN
```

### Boolean Coercion

Falsy values (convert to `false`): `false`, `0`, `-0`, `0n`, `""`, `null`, `undefined`, `NaN`

Everything else is truthy:

```javascript
console.log(Boolean(0));  // false
console.log(Boolean(""));  // false
console.log(Boolean(" "));  // true (non-empty string)
console.log(Boolean([]));  // true (empty array is truthy!)
console.log(Boolean({}));  // true (empty object is truthy!)

// In conditional contexts
if ("") {
  console.log("Won't run");
}

if ("text") {
  console.log("Will run");  // This runs
}
```

**Output:**
```
false
false
true
true
true
Will run
```

### Comparison Coercion

```javascript
console.log("5" == 5);  // true (loose equality coerces)
console.log("5" === 5);  // false (strict equality doesn't coerce)

console.log(null == undefined);  // true (special case)
console.log(null === undefined);  // false

console.log(0 == false);  // true
console.log(0 === false);  // false
```

**Output:**
```
true
false
true
false
true
false
```

> **Note:** Always prefer `===` (strict equality) to avoid unexpected coercion bugs.

---

## Operators

### Arithmetic Operators

```javascript
console.log(10 + 5);  // 15 (addition)
console.log(10 - 5);  // 5 (subtraction)
console.log(10 * 5);  // 50 (multiplication)
console.log(10 / 5);  // 2 (division)
console.log(10 % 3);  // 1 (remainder/modulo)
console.log(2 ** 3);  // 8 (exponentiation)
```

**Output:**
```
15
5
50
2
1
8
```

### Assignment Operators

```javascript
let x = 10;
x += 5;  // x = x + 5 → 15
x -= 3;  // x = x - 3 → 12
x *= 2;  // x = x * 2 → 24
x /= 4;  // x = x / 4 → 6
x %= 4;  // x = x % 4 → 2
x **= 3;  // x = x ** 3 → 8

console.log(x);  // 8
```

**Output:**
```
8
```

### Comparison Operators

```javascript
console.log(5 > 3);  // true
console.log(5 < 3);  // false
console.log(5 >= 5);  // true
console.log(5 <= 4);  // false
console.log(5 == "5");  // true (loose)
console.log(5 === "5");  // false (strict)
console.log(5 != "5");  // false (loose)
console.log(5 !== "5");  // true (strict)
```

**Output:**
```
true
false
true
false
true
false
false
true
```

### Logical Operators

```javascript
console.log(true && false);  // false (AND)
console.log(true || false);  // true (OR)
console.log(!true);  // false (NOT)

// Short-circuit evaluation
console.log(false && expensiveOperation());  // false (doesn't call function)
console.log(true || expensiveOperation());  // true (doesn't call function)

// Nullish coalescing (??) - returns right side if left is null/undefined
console.log(null ?? "default");  // "default"
console.log(0 ?? "default");  // 0 (not null/undefined)
console.log("" ?? "default");  // "" (not null/undefined)
```

**Output:**
```
false
true
false
false
true
default
0

```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `const` by default | Prevents accidental reassignment, signals immutable bindings |
| Use `let` when reassignment is needed | Clear intent that value will change |
| Avoid `var` entirely | Prevents scope confusion and hoisting bugs |
| Use `===` instead of `==` | Avoids unexpected type coercion bugs |
| Declare variables at the top of their scope | Makes TDZ behavior explicit and predictable |
| Use descriptive names | `userAge` instead of `x` improves readability |
| Initialize variables when declared | Avoids `undefined` state when possible |
| Use `Number.isNaN()` instead of `=== NaN` | `NaN === NaN` is always false |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `if (value == null)` checks both null and undefined | Use `if (value === null)` or `if (value == null)` intentionally |
| Accessing `let`/`const` before declaration | Declare at block start or where first used |
| Assuming `typeof null === "null"` | It's `"object"` due to historical bug; check explicitly |
| `const` makes values immutable | Only the binding is immutable; objects/arrays can still mutate |
| Floating-point precision: `0.1 + 0.2 !== 0.3` | Use libraries for precise decimal math or compare with tolerance |
| Forgetting `n` suffix for BigInt literals | Always use `123n` not `123` for BigInt values |

---

## Hands-on Exercise

### Your Task
Create a function that validates and processes user input from an AI chat interface. The function should handle various data types, perform type checking, and return a structured result.

### Requirements
1. Create a `processUserInput(input)` function
2. Check if input is a number, string, boolean, or other type
3. For strings: trim whitespace and check if empty
4. For numbers: check if valid (not NaN) and within range 1-100
5. For booleans: convert to "yes"/"no" string
6. Return an object with `{ type, value, isValid, message }`

### Expected Result
```javascript
processUserInput("  Hello AI  ");
// { type: "string", value: "Hello AI", isValid: true, message: "Valid string input" }

processUserInput(42);
// { type: "number", value: 42, isValid: true, message: "Valid number in range" }

processUserInput(true);
// { type: "boolean", value: "yes", isValid: true, message: "Boolean converted" }
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `typeof` to check input type
- Use `.trim()` to remove whitespace from strings
- Use `Number.isNaN()` to check for NaN
- Use conditional checks to validate ranges
- Build the return object step by step
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function processUserInput(input) {
  const result = {
    type: typeof input,
    value: input,
    isValid: false,
    message: ""
  };
  
  if (typeof input === "string") {
    const trimmed = input.trim();
    if (trimmed.length > 0) {
      result.value = trimmed;
      result.isValid = true;
      result.message = "Valid string input";
    } else {
      result.message = "Empty string after trimming";
    }
  } else if (typeof input === "number") {
    if (Number.isNaN(input)) {
      result.message = "Input is NaN";
    } else if (input >= 1 && input <= 100) {
      result.isValid = true;
      result.message = "Valid number in range";
    } else {
      result.message = "Number out of range (1-100)";
    }
  } else if (typeof input === "boolean") {
    result.value = input ? "yes" : "no";
    result.isValid = true;
    result.message = "Boolean converted";
  } else {
    result.message = `Unsupported type: ${typeof input}`;
  }
  
  return result;
}

// Test cases
console.log(processUserInput("  Hello AI  "));
console.log(processUserInput(42));
console.log(processUserInput(true));
console.log(processUserInput(NaN));
console.log(processUserInput(150));
console.log(processUserInput(""));
```
</details>

### Bonus Challenges
- [ ] Add support for arrays and objects in the validator
- [ ] Handle `null` and `undefined` inputs explicitly
- [ ] Add BigInt validation for large numbers
- [ ] Implement custom error messages for different validation failures

---

## Summary

✅ `let` and `const` provide block scoping and prevent hoisting bugs via the Temporal Dead Zone
✅ JavaScript has 7 primitive types: Number, BigInt, String, Boolean, undefined, null, Symbol
✅ Type coercion happens automatically but can cause bugs—use strict equality (`===`) to avoid it
✅ `const` prevents reassignment but doesn't make objects/arrays immutable
✅ Use `typeof` for type checking, but remember `typeof null === "object"`

[Next: Control Structures](./02-control-structures.md)

---

<!-- 
Sources Consulted:
- MDN let statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let
- MDN const statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/const
- MDN JavaScript Guide - Grammar and Types: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Grammar_and_Types
- MDN typeof operator: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/typeof
-->
