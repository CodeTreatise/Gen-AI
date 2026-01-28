---
title: "Functions"
---

# Functions

## Introduction

Functions are the building blocks of JavaScript applications—they encapsulate behavior, enable code reuse, and provide abstraction. When building AI-powered interfaces, you'll write functions to call APIs, process streaming responses, format conversation history, and handle user interactions. Understanding function declarations, expressions, arrow functions, and higher-order functions unlocks powerful programming patterns.

Modern JavaScript offers multiple ways to define functions, each with different characteristics around syntax, scope, and the `this` binding. Choosing the right pattern for each situation makes your code clearer and prevents subtle bugs.

### What We'll Cover
- Function declarations vs. expressions
- Arrow functions and their unique `this` behavior
- Parameters: default values, rest parameters, destructuring
- Return values and early returns
- Higher-order functions and callbacks
- Function scope and the call stack

### Prerequisites
- Variables and data types
- Control structures (if statements, loops)
- Understanding of objects (covered later, but helpful context)

---

## Function Declarations

The most common way to define functions:

```javascript
function greet(name) {
  return `Hello, ${name}!`;
}

console.log(greet("Alice"));  // "Hello, Alice!"
```

**Output:**
```
Hello, Alice!
```

Function declarations are **hoisted**—you can call them before their definition:

```javascript
console.log(add(5, 3));  // 8 - works due to hoisting

function add(a, b) {
  return a + b;
}
```

**Output:**
```
8
```

> **Note:** Hoisting moves function declarations to the top of their scope during compilation, so they're available throughout that scope.

Multiple parameters and return values:

```javascript
function calculateTotal(price, quantity, taxRate) {
  const subtotal = price * quantity;
  const tax = subtotal * taxRate;
  const total = subtotal + tax;
  return total;
}

console.log(calculateTotal(10, 3, 0.08));  // 32.4
```

**Output:**
```
32.4
```

---

## Function Expressions

Assigning a function to a variable:

```javascript
const multiply = function(a, b) {
  return a * b;
};

console.log(multiply(4, 5));  // 20
```

**Output:**
```
20
```

Function expressions are **not hoisted** like declarations:

```javascript
console.log(subtract(10, 3));  // ❌ ReferenceError: Cannot access 'subtract' before initialization

const subtract = function(a, b) {
  return a - b;
};
```

Named function expressions (helpful for debugging):

```javascript
const factorial = function fact(n) {
  if (n <= 1) return 1;
  return n * fact(n - 1);  // Recursive call using function name
};

console.log(factorial(5));  // 120
```

**Output:**
```
120
```

---

## Arrow Functions

Concise syntax introduced in ES2015:

```javascript
// Traditional function
const add = function(a, b) {
  return a + b;
};

// Arrow function
const addArrow = (a, b) => {
  return a + b;
};

// Concise arrow (implicit return)
const addConcise = (a, b) => a + b;

console.log(addConcise(3, 7));  // 10
```

**Output:**
```
10
```

Single parameter (parentheses optional):

```javascript
const square = x => x * x;
console.log(square(5));  // 25

const greet = name => `Hello, ${name}!`;
console.log(greet("Bob"));  // "Hello, Bob!"
```

**Output:**
```
25
Hello, Bob!
```

No parameters (parentheses required):

```javascript
const getRandom = () => Math.random();
console.log(getRandom());  // 0.something
```

Returning objects (wrap in parentheses):

```javascript
const makeUser = (name, age) => ({ name, age });
console.log(makeUser("Charlie", 30));  // { name: "Charlie", age: 30 }
```

**Output:**
```
{ name: "Charlie", age: 30 }
```

### Arrow Functions and `this`

Arrow functions **do not have their own `this`**—they inherit `this` from the enclosing scope (lexical `this`):

```javascript
const person = {
  name: "Alice",
  greet: function() {
    // Regular function - 'this' refers to person
    console.log("Regular:", this.name);
  },
  greetArrow: () => {
    // Arrow function - 'this' refers to global/undefined
    console.log("Arrow:", this.name);  // undefined
  }
};

person.greet();       // "Regular: Alice"
person.greetArrow();  // "Arrow: undefined"
```

**Output:**
```
Regular: Alice
Arrow: undefined
```

Arrow functions are ideal for callbacks that need to maintain `this`:

```javascript
class Timer {
  constructor() {
    this.seconds = 0;
  }
  
  start() {
    // Arrow function preserves 'this' from start() method
    setInterval(() => {
      this.seconds++;
      console.log(this.seconds);
    }, 1000);
  }
}

// With regular function, 'this' would be undefined in the callback
```

> **Note:** Arrow functions cannot be used as constructors (no `new`) and don't have `arguments` object. Use regular functions when you need these features.

---

## Parameters

### Default Parameters

Provide fallback values when arguments aren't passed:

```javascript
function greet(name = "Guest", greeting = "Hello") {
  return `${greeting}, ${name}!`;
}

console.log(greet());                    // "Hello, Guest!"
console.log(greet("Alice"));             // "Hello, Alice!"
console.log(greet("Bob", "Good morning"));  // "Good morning, Bob!"
```

**Output:**
```
Hello, Guest!
Hello, Alice!
Good morning, Bob!
```

Defaults can use previous parameters:

```javascript
function createApiUrl(endpoint, base = "https://api.example.com", version = "v1") {
  return `${base}/${version}/${endpoint}`;
}

console.log(createApiUrl("users"));  // "https://api.example.com/v1/users"
```

**Output:**
```
https://api.example.com/v1/users
```

### Rest Parameters

Collect remaining arguments into an array:

```javascript
function sum(...numbers) {
  return numbers.reduce((total, num) => total + num, 0);
}

console.log(sum(1, 2, 3));        // 6
console.log(sum(1, 2, 3, 4, 5));  // 15
```

**Output:**
```
6
15
```

Rest parameters must be last:

```javascript
function logInfo(level, ...messages) {
  console.log(`[${level}]`, ...messages);
}

logInfo("ERROR", "Connection failed", "Retrying...");
// [ERROR] Connection failed Retrying...
```

**Output:**
```
[ERROR] Connection failed Retrying...
```

### Parameter Destructuring

Extract values from objects/arrays passed as arguments:

```javascript
// Object destructuring
function displayUser({ name, age, city = "Unknown" }) {
  console.log(`${name}, ${age}, from ${city}`);
}

displayUser({ name: "Alice", age: 30, city: "NYC" });
// Alice, 30, from NYC

// Array destructuring
function getCoordinates([x, y]) {
  return { x, y };
}

console.log(getCoordinates([10, 20]));  // { x: 10, y: 20 }
```

**Output:**
```
Alice, 30, from NYC
{ x: 10, y: 20 }
```

---

## Return Values

Functions return `undefined` by default:

```javascript
function noReturn() {
  console.log("Doing something");
}

const result = noReturn();
console.log(result);  // undefined
```

**Output:**
```
Doing something
undefined
```

Early returns for guard clauses:

```javascript
function divide(a, b) {
  if (b === 0) {
    return "Cannot divide by zero";
  }
  return a / b;
}

console.log(divide(10, 2));  // 5
console.log(divide(10, 0));  // "Cannot divide by zero"
```

**Output:**
```
5
Cannot divide by zero
```

Returning multiple values (via object or array):

```javascript
function getStats(numbers) {
  const sum = numbers.reduce((a, b) => a + b, 0);
  const avg = sum / numbers.length;
  const max = Math.max(...numbers);
  const min = Math.min(...numbers);
  
  return { sum, avg, max, min };
}

const stats = getStats([1, 2, 3, 4, 5]);
console.log(stats);  // { sum: 15, avg: 3, max: 5, min: 1 }
```

**Output:**
```
{ sum: 15, avg: 3, max: 5, min: 1 }
```

---

## Higher-Order Functions

Functions that take other functions as arguments or return functions.

### Functions as Arguments (Callbacks)

```javascript
function processArray(arr, callback) {
  const result = [];
  for (const item of arr) {
    result.push(callback(item));
  }
  return result;
}

const numbers = [1, 2, 3, 4, 5];
const doubled = processArray(numbers, x => x * 2);
console.log(doubled);  // [2, 4, 6, 8, 10]
```

**Output:**
```
[2, 4, 6, 8, 10]
```

Real-world example: async operations

```javascript
function fetchData(url, onSuccess, onError) {
  // Simulate API call
  setTimeout(() => {
    if (url.includes("api")) {
      onSuccess({ data: "Response from " + url });
    } else {
      onError("Invalid URL");
    }
  }, 1000);
}

fetchData(
  "https://api.example.com/users",
  (response) => console.log("Success:", response.data),
  (error) => console.log("Error:", error)
);
```

### Functions Returning Functions

```javascript
function createMultiplier(factor) {
  return function(number) {
    return number * factor;
  };
}

const double = createMultiplier(2);
const triple = createMultiplier(3);

console.log(double(5));  // 10
console.log(triple(5));  // 15
```

**Output:**
```
10
15
```

Practical example: API configuration

```javascript
function createApiClient(baseUrl) {
  return function(endpoint) {
    return `${baseUrl}/${endpoint}`;
  };
}

const chatApi = createApiClient("https://api.chat.com/v1");
console.log(chatApi("messages"));  // "https://api.chat.com/v1/messages"
console.log(chatApi("users"));     // "https://api.chat.com/v1/users"
```

**Output:**
```
https://api.chat.com/v1/messages
https://api.chat.com/v1/users
```

---

## Function Scope and Call Stack

Functions create their own scope—variables declared inside are not accessible outside:

```javascript
function outer() {
  const outerVar = "I'm outer";
  
  function inner() {
    const innerVar = "I'm inner";
    console.log(outerVar);  // ✅ Can access outer scope
  }
  
  inner();
  console.log(innerVar);  // ❌ ReferenceError: innerVar is not defined
}

outer();
```

The call stack tracks function execution:

```javascript
function first() {
  console.log("First function");
  second();
  console.log("First function end");
}

function second() {
  console.log("Second function");
  third();
  console.log("Second function end");
}

function third() {
  console.log("Third function");
}

first();
```

**Output:**
```
First function
Second function
Third function
Second function end
First function end
```

> **Note:** Stack overflow occurs when the call stack exceeds its limit (common with unbounded recursion).

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use arrow functions for short callbacks | Concise and maintains lexical `this` |
| Use function declarations for named functions | Clear intent and hoisting enables top-down code organization |
| Prefer default parameters over conditional checks | Cleaner code: `function(x = 0)` vs `if (!x) x = 0` |
| Use descriptive function names | `calculateTotalPrice()` is clearer than `calc()` |
| Keep functions small and focused | Each function should do one thing well (Single Responsibility) |
| Use early returns for validation | Reduces nesting and improves readability |
| Use rest parameters instead of `arguments` | Rest parameters are real arrays with array methods |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using arrow functions as methods | Use regular functions when `this` binding is needed |
| Forgetting `return` statement | Function returns `undefined` without explicit `return` |
| Modifying parameters | Parameters are passed by value (primitives) or reference (objects)—avoid mutation |
| Calling hoisted functions before checking conditions | Even hoisted functions can have logic errors if called prematurely |
| Using `function` inside loops with `var` | Creates one shared variable; use `let` or arrow functions |
| Returning inside a callback thinking it returns from outer function | Callback's return doesn't affect outer function |

---

## Hands-on Exercise

### Your Task
Create a message processor for an AI chat application that uses various function patterns. The processor should handle message formatting, validation, and transformation using higher-order functions.

### Requirements
1. Create a `createMessageProcessor(config)` function that returns a processor object
2. The processor should have methods:
   - `format(message)`: Apply formatting based on config
   - `validate(message)`: Check message meets criteria
   - `process(messages)`: Transform an array of messages using both methods
3. Use arrow functions for callbacks, default parameters, and rest parameters
4. Config should support: `maxLength`, `prefix`, `transform` (callback)

### Expected Result
```javascript
const processor = createMessageProcessor({
  maxLength: 50,
  prefix: "[AI]",
  transform: (msg) => msg.toUpperCase()
});

processor.process(["hello", "how are you today"]);
// ["[AI] HELLO", "[AI] HOW ARE YOU TODAY"]
```

<details>
<summary>💡 Hints (click to expand)</summary>

- The outer function returns an object with methods
- Use arrow functions for method implementations to maintain scope
- Use default parameters in `createMessageProcessor`
- Use array methods with callbacks in `process`
- Validate length before formatting
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function createMessageProcessor(config = {}) {
  // Default configuration
  const {
    maxLength = 100,
    prefix = "",
    transform = (msg) => msg
  } = config;
  
  return {
    format: (message) => {
      let formatted = message;
      
      // Apply transformation
      formatted = transform(formatted);
      
      // Add prefix if configured
      if (prefix) {
        formatted = `${prefix} ${formatted}`;
      }
      
      return formatted;
    },
    
    validate: (message) => {
      return typeof message === "string" && message.length <= maxLength;
    },
    
    process: (messages) => {
      return messages
        .filter((msg) => this.validate(msg))
        .map((msg) => this.format(msg));
    }
  };
}

// Test cases
const processor = createMessageProcessor({
  maxLength: 50,
  prefix: "[AI]",
  transform: (msg) => msg.toUpperCase()
});

console.log(processor.format("hello"));
// [AI] HELLO

console.log(processor.validate("short"));
// true

console.log(processor.validate("a".repeat(100)));
// false

console.log(processor.process(["hello", "how are you today", "x".repeat(60)]));
// ["[AI] HELLO", "[AI] HOW ARE YOU TODAY"]
// (long message filtered out)
```
</details>

### Bonus Challenges
- [ ] Add a `compose` function that chains multiple transformations
- [ ] Implement a `retry` higher-order function for API calls
- [ ] Create a `debounce` function that delays execution
- [ ] Add method chaining support: `processor.format(...).validate(...)`

---

## Summary

✅ Function declarations are hoisted; expressions and arrow functions are not
✅ Arrow functions provide concise syntax and lexical `this` binding
✅ Default parameters, rest parameters, and destructuring make functions flexible
✅ Higher-order functions (functions that take or return functions) enable powerful abstractions
✅ Functions create scope—inner functions can access outer variables but not vice versa

[Previous: Control Structures](./02-control-structures.md) | [Next: Scope and Closures](./04-scope-closures.md)

---

<!-- 
Sources Consulted:
- MDN JavaScript Guide - Functions: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Functions
- MDN Arrow Functions: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Arrow_functions
- MDN Function Reference: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions
- MDN Default Parameters: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Default_parameters
- MDN Rest Parameters: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/rest_parameters
-->
