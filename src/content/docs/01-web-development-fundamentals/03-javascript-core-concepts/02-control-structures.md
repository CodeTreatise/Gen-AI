---
title: "Control Structures"
---

# Control Structures

## Introduction

Control structures determine the flow of your program—deciding which code runs when, and how many times. When building AI applications, you'll use conditionals to handle different API responses, loops to process conversation history, and switch statements to route user intents. Mastering these fundamentals ensures your code responds intelligently to varying conditions.

Every AI interface needs decision-making logic: "If the API returns an error, show a fallback message. If the user's message is too long, truncate it. While processing chunks, accumulate the response." Control structures make these behaviors possible.

### What We'll Cover
- Conditional statements (`if`, `else if`, `else`)
- Switch statements for multi-way branching
- Loops: `for`, `while`, `do-while`, `for...in`, `for...of`
- Ternary operator for concise conditionals
- Loop control with `break` and `continue`

### Prerequisites
- Variables and data types (previous lesson)
- Basic operators and comparisons
- Understanding of boolean values

---

## Conditional Statements

### if, else if, else

The `if` statement executes code when a condition is `true`:

```javascript
const temperature = 25;

if (temperature > 30) {
  console.log("It's hot!");
} else if (temperature > 20) {
  console.log("It's comfortable.");
} else {
  console.log("It's cold!");
}
```

**Output:**
```
It's comfortable.
```

Multiple conditions with logical operators:

```javascript
const userAge = 25;
const hasPermission = true;

if (userAge >= 18 && hasPermission) {
  console.log("Access granted");
} else {
  console.log("Access denied");
}
```

**Output:**
```
Access granted
```

> **Note:** The condition is evaluated as a boolean. Any value can be used—JavaScript will coerce it to `true` or `false` (see truthy/falsy values from previous lesson).

### Truthy and Falsy in Conditions

```javascript
const response = "";

if (response) {
  console.log("Got response:", response);
} else {
  console.log("No response received");
}
```

**Output:**
```
No response received
```

Checking for null or undefined:

```javascript
const apiKey = process.env.API_KEY;

if (!apiKey) {
  console.error("API key missing!");
  // Exit early or throw error
}

// Continue with valid apiKey
console.log("API key loaded");
```

### Block Scope with Conditionals

Remember: `let` and `const` are block-scoped (limited to `{}`):

```javascript
const score = 85;

if (score >= 80) {
  const grade = "A";
  console.log("Grade:", grade);  // ✅ Works
}

console.log("Grade:", grade);  // ❌ ReferenceError: grade is not defined
```

---

## Ternary Operator

A concise way to write simple `if-else` statements:

**Syntax:** `condition ? valueIfTrue : valueIfFalse`

```javascript
const age = 20;
const status = age >= 18 ? "adult" : "minor";
console.log(status);  // "adult"
```

**Output:**
```
adult
```

Ternary in function arguments:

```javascript
function greet(name, isFormal) {
  console.log(isFormal ? `Good day, ${name}.` : `Hey ${name}!`);
}

greet("Alice", true);   // "Good day, Alice."
greet("Bob", false);    // "Hey Bob!"
```

**Output:**
```
Good day, Alice.
Hey Bob!
```

Nested ternary (use sparingly—can become unreadable):

```javascript
const score = 75;
const grade = score >= 90 ? "A" : score >= 80 ? "B" : score >= 70 ? "C" : "F";
console.log(grade);  // "C"
```

**Output:**
```
C
```

> **Note:** Nested ternaries quickly become hard to read. For complex conditions, use regular `if-else` statements instead.

---

## Switch Statements

Use `switch` for multi-way branching based on a single value:

```javascript
const command = "start";

switch (command) {
  case "start":
    console.log("Starting application...");
    break;
  case "stop":
    console.log("Stopping application...");
    break;
  case "restart":
    console.log("Restarting application...");
    break;
  default:
    console.log("Unknown command");
}
```

**Output:**
```
Starting application...
```

### The Importance of break

Without `break`, execution "falls through" to the next case:

```javascript
const day = "Monday";

switch (day) {
  case "Monday":
    console.log("Week started");
    // No break - falls through!
  case "Tuesday":
    console.log("Still early in week");
    break;
  default:
    console.log("Later in week");
}
```

**Output:**
```
Week started
Still early in week
```

### Intentional Fall-through

Sometimes fall-through is useful (group multiple cases):

```javascript
const char = "a";

switch (char) {
  case "a":
  case "e":
  case "i":
  case "o":
  case "u":
    console.log("Vowel");
    break;
  default:
    console.log("Consonant");
}
```

**Output:**
```
Vowel
```

### Switch with Expressions

The case values can be expressions:

```javascript
const value = 10;

switch (true) {
  case value < 0:
    console.log("Negative");
    break;
  case value === 0:
    console.log("Zero");
    break;
  case value > 0:
    console.log("Positive");
    break;
}
```

**Output:**
```
Positive
```

> **Note:** This pattern uses `switch(true)` with boolean case expressions. It works, but `if-else` is often clearer for range checks.

---

## Loops

### for Loop

Classic loop with initialization, condition, and increment:

```javascript
for (let i = 0; i < 5; i++) {
  console.log("Iteration:", i);
}
```

**Output:**
```
Iteration: 0
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
```

Looping through arrays:

```javascript
const colors = ["red", "green", "blue"];

for (let i = 0; i < colors.length; i++) {
  console.log(`Color ${i}: ${colors[i]}`);
}
```

**Output:**
```
Color 0: red
Color 1: green
Color 2: blue
```

Counting backwards:

```javascript
for (let i = 5; i > 0; i--) {
  console.log("Countdown:", i);
}
console.log("Liftoff!");
```

**Output:**
```
Countdown: 5
Countdown: 4
Countdown: 3
Countdown: 2
Countdown: 1
Liftoff!
```

### while Loop

Repeats while condition is `true`:

```javascript
let count = 0;

while (count < 3) {
  console.log("Count:", count);
  count++;
}
```

**Output:**
```
Count: 0
Count: 1
Count: 2
```

Common pattern: process until condition met:

```javascript
let retries = 0;
const maxRetries = 3;
let success = false;

while (!success && retries < maxRetries) {
  console.log(`Attempt ${retries + 1}...`);
  // Simulate operation
  success = Math.random() > 0.7;  // 30% success rate
  retries++;
}

if (success) {
  console.log("Operation succeeded!");
} else {
  console.log("Max retries reached");
}
```

> **Note:** Be careful with `while` loops—if the condition never becomes `false`, you create an infinite loop that hangs your program.

### do-while Loop

Executes at least once, then checks condition:

```javascript
let userInput;

do {
  userInput = prompt("Enter 'quit' to exit:");
  console.log("You entered:", userInput);
} while (userInput !== "quit");
```

The difference between `while` and `do-while`:

```javascript
// while: may not execute at all
let x = 10;
while (x < 5) {
  console.log("This never runs");
}

// do-while: executes at least once
let y = 10;
do {
  console.log("This runs once:", y);
} while (y < 5);
```

**Output:**
```
This runs once: 10
```

### for...of Loop

Iterates over iterable objects (arrays, strings, Maps, Sets):

```javascript
const fruits = ["apple", "banana", "cherry"];

for (const fruit of fruits) {
  console.log(fruit);
}
```

**Output:**
```
apple
banana
cherry
```

Iterating strings:

```javascript
const word = "hello";

for (const char of word) {
  console.log(char);
}
```

**Output:**
```
h
e
l
l
o
```

With array destructuring:

```javascript
const users = [
  ["Alice", 25],
  ["Bob", 30],
  ["Charlie", 35]
];

for (const [name, age] of users) {
  console.log(`${name} is ${age} years old`);
}
```

**Output:**
```
Alice is 25 years old
Bob is 30 years old
Charlie is 35 years old
```

### for...in Loop

Iterates over object keys (enumerable properties):

```javascript
const person = {
  name: "Alice",
  age: 30,
  city: "New York"
};

for (const key in person) {
  console.log(`${key}: ${person[key]}`);
}
```

**Output:**
```
name: Alice
age: 30
city: New York
```

> **Note:** Avoid `for...in` with arrays—it iterates over indices as strings and includes inherited properties. Use `for...of` for arrays instead.

---

## Loop Control: break and continue

### break

Exits the loop immediately:

```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9];

for (const num of numbers) {
  if (num > 5) {
    break;  // Stop looping
  }
  console.log(num);
}
```

**Output:**
```
1
2
3
4
5
```

Finding the first match:

```javascript
const users = ["admin", "guest", "user", "moderator"];
let foundAdmin = false;

for (const user of users) {
  if (user === "admin") {
    foundAdmin = true;
    break;  // No need to check remaining users
  }
}

console.log("Has admin:", foundAdmin);
```

**Output:**
```
Has admin: true
```

### continue

Skips the current iteration and continues with the next:

```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9];

for (const num of numbers) {
  if (num % 2 === 0) {
    continue;  // Skip even numbers
  }
  console.log(num);
}
```

**Output:**
```
1
3
5
7
9
```

Filtering during processing:

```javascript
const messages = ["Hello", "", "World", null, "AI"];

for (const msg of messages) {
  if (!msg) {
    continue;  // Skip empty/null messages
  }
  console.log("Processing:", msg);
}
```

**Output:**
```
Processing: Hello
Processing: World
Processing: AI
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `===` in conditions, not `==` | Avoids unexpected type coercion bugs |
| Keep conditions simple and readable | Complex boolean logic is error-prone; extract to variables |
| Use `for...of` for arrays, not `for...in` | `for...in` iterates keys as strings, includes prototype chain |
| Add `break` to every `switch` case | Prevents accidental fall-through bugs |
| Avoid modifying loop counters inside the loop | Leads to infinite loops or skipped iterations |
| Use `const` in `for...of` loops | Each iteration gets a new binding, `const` prevents mutation |
| Guard against infinite loops | Always ensure loop condition will eventually be false |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `if (x = 5)` assigns instead of comparing | Use `if (x === 5)` for comparison |
| Forgetting `break` in `switch` causes fall-through | Always add `break` unless fall-through is intentional |
| `while (true)` without a `break` creates infinite loop | Ensure exit condition or use `break` when done |
| Using `for...in` on arrays | Use `for...of` or traditional `for` loop for arrays |
| Modifying array while looping with index | Copy array first, or loop backwards: `for (let i = arr.length - 1; i >= 0; i--)` |
| Shadowing outer variables in loop block | Use different variable names to avoid confusion |

---

## Hands-on Exercise

### Your Task
Create a function that processes an array of AI chat messages and filters them based on various conditions. The function should demonstrate multiple control structure patterns.

### Requirements
1. Create a `processMessages(messages)` function that takes an array of message objects
2. Each message has: `{ role: "user" | "assistant", content: string, timestamp: number }`
3. Skip messages with empty content
4. Stop processing if you encounter a message with role "system"
5. Count messages by role
6. Return an object with counts and processed messages

### Expected Result
```javascript
const messages = [
  { role: "user", content: "Hello", timestamp: 1000 },
  { role: "assistant", content: "Hi there!", timestamp: 2000 },
  { role: "user", content: "", timestamp: 3000 },
  { role: "assistant", content: "How can I help?", timestamp: 4000 },
  { role: "system", content: "Stop here", timestamp: 5000 },
  { role: "user", content: "Never processed", timestamp: 6000 }
];

processMessages(messages);
// {
//   processed: [
//     { role: "user", content: "Hello", timestamp: 1000 },
//     { role: "assistant", content: "Hi there!", timestamp: 2000 },
//     { role: "assistant", content: "How can I help?", timestamp: 4000 }
//   ],
//   counts: { user: 1, assistant: 2, system: 0 }
// }
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a `for...of` loop to iterate over messages
- Use `continue` to skip empty content
- Use `break` when encountering "system" role
- Track counts with an object: `{ user: 0, assistant: 0, system: 0 }`
- Build the processed array as you go
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function processMessages(messages) {
  const processed = [];
  const counts = { user: 0, assistant: 0, system: 0 };
  
  for (const message of messages) {
    // Stop if we hit a system message
    if (message.role === "system") {
      break;
    }
    
    // Skip messages with empty content
    if (!message.content || message.content.trim() === "") {
      continue;
    }
    
    // Count by role
    if (counts.hasOwnProperty(message.role)) {
      counts[message.role]++;
    }
    
    // Add to processed array
    processed.push(message);
  }
  
  return { processed, counts };
}

// Test
const messages = [
  { role: "user", content: "Hello", timestamp: 1000 },
  { role: "assistant", content: "Hi there!", timestamp: 2000 },
  { role: "user", content: "", timestamp: 3000 },
  { role: "assistant", content: "How can I help?", timestamp: 4000 },
  { role: "system", content: "Stop here", timestamp: 5000 },
  { role: "user", content: "Never processed", timestamp: 6000 }
];

console.log(processMessages(messages));
```
</details>

### Bonus Challenges
- [ ] Add a `maxMessages` parameter to limit processing
- [ ] Use a `switch` statement to handle different message roles
- [ ] Add timestamp filtering (only process messages after a certain time)
- [ ] Implement rate limiting: skip messages if more than 5 from same role in a row

---

## Summary

✅ Use `if-else` for conditional logic, ternary operator for simple cases
✅ `switch` statements provide clean multi-way branching—always add `break`
✅ `for...of` is the modern way to iterate arrays; avoid `for...in` on arrays
✅ `break` exits loops early; `continue` skips to the next iteration
✅ Guard against infinite loops by ensuring loop conditions eventually become false

[Previous: Variables and Data Types](./01-variables-data-types.md) | [Next: Functions](./03-functions.md)

---

<!-- 
Sources Consulted:
- MDN JavaScript Guide - Control Flow: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Control_flow_and_error_handling
- MDN JavaScript Guide - Loops and Iteration: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Loops_and_iteration
- MDN if...else: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/if...else
- MDN switch: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/switch
- MDN for...of: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/for...of
-->
