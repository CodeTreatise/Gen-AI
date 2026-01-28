---
title: "Array Methods"
---

# Array Methods

## Introduction

Arrays are the workhorse of data processing in JavaScript. Modern iterative methods like `map`, `filter`, `reduce`, and `forEach` transform how we handle collections—replacing manual loops with declarative, composable operations. When building AI applications, you'll use these methods to process API responses, transform conversation data, filter results, and aggregate information efficiently.

Understanding array methods is essential for functional programming patterns that make code more readable and less error-prone. These methods don't mutate the original array (mostly), enabling predictable data transformations.

### What We'll Cover
- Iterative methods: `forEach`, `map`, `filter`, `find`, `some`, `every`
- Reduction: `reduce` and `reduceRight`
- Method chaining for complex transformations
- Copying vs. mutating methods
- Performance considerations

### Prerequisites
- Arrays and basic iteration
- Arrow functions and callbacks
- Understanding of immutability concepts

---

## forEach: Iteration Without Return

Execute a function for each array element:

```javascript
const messages = ["Hello", "How are you?", "Goodbye"];

messages.forEach((msg, index) => {
  console.log(`${index}: ${msg}`);
});
```

**Output:**
```
0: Hello
1: How are you?
2: Goodbye
```

> **Note:** `forEach` returns `undefined`—use it for side effects, not transformations.

---

## map: Transform Each Element

Create a new array by transforming each element:

```javascript
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);

console.log(doubled);  // [2, 4, 6, 8, 10]
console.log(numbers);  // [1, 2, 3, 4, 5] - original unchanged
```

**Output:**
```
[ 2, 4, 6, 8, 10 ]
[ 1, 2, 3, 4, 5 ]
```

Transforming objects:

```javascript
const users = [
  { name: "Alice", age: 30 },
  { name: "Bob", age: 25 }
];

const names = users.map(user => user.name);
console.log(names);  // ["Alice", "Bob"]

const withIds = users.map((user, index) => ({
  ...user,
  id: index + 1
}));
console.log(withIds);
// [{ name: "Alice", age: 30, id: 1 }, { name: "Bob", age: 25, id: 2 }]
```

**Output:**
```
[ 'Alice', 'Bob' ]
[
  { name: 'Alice', age: 30, id: 1 },
  { name: 'Bob', age: 25, id: 2 }
]
```

---

## filter: Select Elements

Create a new array with elements that pass a test:

```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const evens = numbers.filter(n => n % 2 === 0);

console.log(evens);  // [2, 4, 6, 8, 10]
```

**Output:**
```
[ 2, 4, 6, 8, 10 ]
```

Filter objects:

```javascript
const messages = [
  { role: "user", content: "Hello", timestamp: 1000 },
  { role: "assistant", content: "Hi", timestamp: 2000 },
  { role: "user", content: "How are you?", timestamp: 3000 }
];

const userMessages = messages.filter(msg => msg.role === "user");
console.log(userMessages.length);  // 2
```

**Output:**
```
2
```

---

## find and findIndex

Find the first element that matches:

```javascript
const users = [
  { id: 1, name: "Alice" },
  { id: 2, name: "Bob" },
  { id: 3, name: "Charlie" }
];

const user = users.find(u => u.id === 2);
console.log(user);  // { id: 2, name: "Bob" }

const index = users.findIndex(u => u.name === "Charlie");
console.log(index);  // 2

const notFound = users.find(u => u.id === 99);
console.log(notFound);  // undefined
```

**Output:**
```
{ id: 2, name: 'Bob' }
2
undefined
```

---

## some and every

Test if at least one (`some`) or all (`every`) elements pass:

```javascript
const numbers = [1, 2, 3, 4, 5];

console.log(numbers.some(n => n > 4));   // true (5 is > 4)
console.log(numbers.every(n => n > 0));  // true (all positive)
console.log(numbers.every(n => n > 2));  // false (1, 2 are ≤ 2)
```

**Output:**
```
true
true
false
```

Practical example:

```javascript
const messages = [
  { role: "user", content: "Hello" },
  { role: "assistant", content: "" },
  { role: "user", content: "How are you?" }
];

const hasEmptyMessage = messages.some(msg => !msg.content);
console.log(hasEmptyMessage);  // true

const allHaveRole = messages.every(msg => msg.role);
console.log(allHaveRole);  // true
```

**Output:**
```
true
true
```

---

## reduce: Aggregate Values

Reduce an array to a single value:

```javascript
const numbers = [1, 2, 3, 4, 5];

// Sum
const sum = numbers.reduce((acc, n) => acc + n, 0);
console.log(sum);  // 15

// Product
const product = numbers.reduce((acc, n) => acc * n, 1);
console.log(product);  // 120
```

**Output:**
```
15
120
```

Building objects:

```javascript
const users = [
  { id: 1, name: "Alice" },
  { id: 2, name: "Bob" }
];

const usersById = users.reduce((acc, user) => {
  acc[user.id] = user;
  return acc;
}, {});

console.log(usersById);
// { '1': { id: 1, name: 'Alice' }, '2': { id: 2, name: 'Bob' } }
```

**Output:**
```
{
  '1': { id: 1, name: 'Alice' },
  '2': { id: 2, name: 'Bob' }
}
```

Counting occurrences:

```javascript
const words = ["apple", "banana", "apple", "cherry", "banana", "apple"];

const counts = words.reduce((acc, word) => {
  acc[word] = (acc[word] || 0) + 1;
  return acc;
}, {});

console.log(counts);  // { apple: 3, banana: 2, cherry: 1 }
```

**Output:**
```
{ apple: 3, banana: 2, cherry: 1 }
```

---

## Method Chaining

Combine methods for complex transformations:

```javascript
const users = [
  { name: "Alice", age: 30, active: true },
  { name: "Bob", age: 25, active: false },
  { name: "Charlie", age: 35, active: true },
  { name: "David", age: 28, active: true }
];

// Get names of active users over 27, uppercased
const result = users
  .filter(u => u.active)
  .filter(u => u.age > 27)
  .map(u => u.name.toUpperCase());

console.log(result);  // ["ALICE", "CHARLIE"]
```

**Output:**
```
[ 'ALICE', 'CHARLIE' ]
```

Processing conversation data:

```javascript
const messages = [
  { role: "user", content: "hello", timestamp: 1000 },
  { role: "assistant", content: "Hi there!", timestamp: 2000 },
  { role: "user", content: "", timestamp: 3000 },
  { role: "assistant", content: "How can I help?", timestamp: 4000 }
];

const userMessageCount = messages
  .filter(msg => msg.role === "user")
  .filter(msg => msg.content.trim() !== "")
  .length;

console.log(userMessageCount);  // 1
```

**Output:**
```
1
```

---

## Copying vs. Mutating Methods

### Non-Mutating (Return New Array)

| Method | Purpose |
|--------|---------|
| `map` | Transform elements |
| `filter` | Select elements |
| `slice` | Extract portion |
| `concat` | Combine arrays |
| `flat` | Flatten nested arrays |
| `flatMap` | Map then flatten |

### Mutating (Modify Original)

| Method | Purpose |
|--------|---------|
| `push` | Add to end |
| `pop` | Remove from end |
| `shift` | Remove from start |
| `unshift` | Add to start |
| `splice` | Add/remove at index |
| `sort` | Sort in place |
| `reverse` | Reverse in place |

Example:

```javascript
const original = [1, 2, 3];

// Non-mutating
const mapped = original.map(n => n * 2);
console.log(original);  // [1, 2, 3] - unchanged
console.log(mapped);    // [2, 4, 6]

// Mutating
original.push(4);
console.log(original);  // [1, 2, 3, 4] - changed
```

**Output:**
```
[ 1, 2, 3 ]
[ 2, 4, 6 ]
[ 1, 2, 3, 4 ]
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Prefer `map`/`filter`/`reduce` over `for` loops | More declarative, easier to reason about |
| Chain methods for readability | Each step is clear; avoid nested logic |
| Use `find` instead of `filter()[0]` | More efficient; stops at first match |
| Provide initial value to `reduce` | Prevents errors with empty arrays |
| Don't mutate during iteration | Use non-mutating methods for predictable code |
| Use `some`/`every` for boolean checks | Clearer intent than `filter` or manual loops |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mutating array during iteration | Use non-mutating methods or copy first |
| Forgetting `return` in `map` callback | Arrow functions with `{}` need explicit `return` |
| Using `forEach` when `map` is needed | `forEach` doesn't return; use `map` for transformations |
| Not providing initial value to `reduce` | Always provide: `.reduce((acc, n) => ..., initialValue)` |
| Assuming `filter` returns single value | It returns an array; use `find` for single element |
| Chaining too many methods | Balance readability with performance for large arrays |

---

## Hands-on Exercise

### Your Task
Process an array of AI chat messages using array methods. Calculate statistics, filter messages, and transform data using method chaining.

### Requirements
1. Given an array of message objects: `{ role, content, tokens, timestamp }`
2. Implement functions:
   - `getTotalTokens(messages)`: Sum all tokens
   - `getAverageMessageLength(messages)`: Average content length
   - `getMessagesByRole(messages, role)`: Filter by role
   - `formatForDisplay(messages)`: Transform to display format
   - `getConversationSummary(messages)`: Return object with statistics

### Expected Result
```javascript
const messages = [
  { role: "user", content: "Hello", tokens: 1, timestamp: 1000 },
  { role: "assistant", content: "Hi there!", tokens: 2, timestamp: 2000 },
  { role: "user", content: "How are you?", tokens: 3, timestamp: 3000 }
];

getTotalTokens(messages);  // 6
getAverageMessageLength(messages);  // 9.33...
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `reduce` to sum tokens
- Use `map` to get lengths, then `reduce` for average
- Use `filter` for role-based filtering
- Use `map` for transformation to display format
- Combine multiple operations in `getConversationSummary`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function getTotalTokens(messages) {
  return messages.reduce((sum, msg) => sum + msg.tokens, 0);
}

function getAverageMessageLength(messages) {
  if (messages.length === 0) return 0;
  
  const totalLength = messages.reduce((sum, msg) => sum + msg.content.length, 0);
  return totalLength / messages.length;
}

function getMessagesByRole(messages, role) {
  return messages.filter(msg => msg.role === role);
}

function formatForDisplay(messages) {
  return messages.map(msg => ({
    from: msg.role === "user" ? "You" : "AI",
    text: msg.content,
    time: new Date(msg.timestamp).toLocaleTimeString()
  }));
}

function getConversationSummary(messages) {
  const userMessages = messages.filter(msg => msg.role === "user");
  const assistantMessages = messages.filter(msg => msg.role === "assistant");
  
  return {
    totalMessages: messages.length,
    userMessages: userMessages.length,
    assistantMessages: assistantMessages.length,
    totalTokens: getTotalTokens(messages),
    averageLength: Math.round(getAverageMessageLength(messages)),
    firstMessage: messages[0]?.content || "No messages",
    lastMessage: messages[messages.length - 1]?.content || "No messages"
  };
}

// Test
const messages = [
  { role: "user", content: "Hello", tokens: 1, timestamp: 1000 },
  { role: "assistant", content: "Hi there!", tokens: 2, timestamp: 2000 },
  { role: "user", content: "How are you?", tokens: 3, timestamp: 3000 },
  { role: "assistant", content: "I'm doing well, thanks!", tokens: 5, timestamp: 4000 }
];

console.log("Total tokens:", getTotalTokens(messages));
console.log("Average length:", getAverageMessageLength(messages).toFixed(2));
console.log("User messages:", getMessagesByRole(messages, "user").length);
console.log("\nFormatted:");
console.log(formatForDisplay(messages));
console.log("\nSummary:");
console.log(getConversationSummary(messages));
```
</details>

### Bonus Challenges
- [ ] Implement `getTokensPerRole()` that returns object with tokens by role
- [ ] Add `findLongestMessage()` using `reduce`
- [ ] Create `groupByTimePeriod()` that groups messages by hour/day
- [ ] Implement `getMessageStats()` with min/max/median token counts

---

## Summary

✅ Use `map` to transform, `filter` to select, `reduce` to aggregate
✅ `find` returns first match; `some`/`every` test conditions across array
✅ Method chaining enables declarative data transformations
✅ Non-mutating methods (map, filter) preserve originals; mutating methods (push, splice) modify
✅ Always provide initial value to `reduce` for safety with empty arrays

[Previous: Data Structures](./06-data-structures.md) | [Next: Destructuring and Spread](./08-destructuring-spread.md)

---

<!-- 
Sources Consulted:
- MDN Array: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array
- MDN map: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map
- MDN filter: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter
- MDN reduce: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reduce
-->
