---
title: "Scope and Closures"
---

# Scope and Closures

## Introduction

Scope determines which variables are accessible in different parts of your code, while closures enable functions to "remember" variables from their creation context even after that context has finished executing. These concepts are foundational to JavaScript—understanding them unlocks patterns like private variables, module patterns, and callback behaviors that are essential in AI application development.

When building conversational interfaces, closures let you maintain state across multiple API calls, create configurable API clients, and implement event handlers that remember user context. Mastering scope prevents variable collision bugs and makes your code more predictable.

### What We'll Cover
- Global, function, and block scope
- Scope chain and variable resolution
- Lexical scoping
- Closures: what they are and how they work
- Practical closure patterns (private methods, factories, module pattern)
- Common closure gotchas with loops

### Prerequisites
- Variables and data types (especially `let`, `const`, `var`)
- Functions (declarations, expressions, arrow functions)
- Basic understanding of object properties

---

## Understanding Scope

Scope defines where variables are accessible. JavaScript has three main scope types:

### Global Scope

Variables declared outside any function or block:

```javascript
const apiUrl = "https://api.example.com";  // Global scope

function makeRequest() {
  console.log(apiUrl);  // ✅ Can access global variable
}

makeRequest();  // "https://api.example.com"
```

**Output:**
```
https://api.example.com
```

> **Note:** Global variables are accessible everywhere but can lead to naming conflicts. Minimize global scope usage.

### Function Scope

Variables declared with `var` are function-scoped (or global if outside functions):

```javascript
function processData() {
  var result = "Processing...";
  
  if (true) {
    var result = "Done!";  // Same variable (function-scoped)
  }
  
  console.log(result);  // "Done!" - modified by if block
}

processData();
```

**Output:**
```
Done!
```

### Block Scope

Variables declared with `let` and `const` are block-scoped (limited to `{}`):

```javascript
function processData() {
  let result = "Processing...";
  
  if (true) {
    let result = "Done!";  // Different variable (block-scoped)
    console.log("Inside:", result);
  }
  
  console.log("Outside:", result);
}

processData();
```

**Output:**
```
Inside: Done!
Outside: Processing...
```

Real-world example with loops:

```javascript
for (let i = 0; i < 3; i++) {
  setTimeout(() => {
    console.log("let:", i);  // Each iteration has own i
  }, 100);
}

for (var j = 0; j < 3; j++) {
  setTimeout(() => {
    console.log("var:", j);  // All iterations share same j
  }, 100);
}
```

**Output:**
```
let: 0
let: 1
let: 2
var: 3
var: 3
var: 3
```

> **Note:** With `var`, all timeout callbacks share the same `j`, which becomes `3` after the loop completes. With `let`, each iteration creates a new `i` binding.

---

## Scope Chain

When JavaScript looks up a variable, it starts at the current scope and moves outward until found:

```javascript
const global = "I'm global";

function outer() {
  const outerVar = "I'm outer";
  
  function inner() {
    const innerVar = "I'm inner";
    
    console.log(innerVar);   // ✅ Found in current scope
    console.log(outerVar);   // ✅ Found in outer scope
    console.log(global);     // ✅ Found in global scope
  }
  
  inner();
}

outer();
```

**Output:**
```
I'm inner
I'm outer
I'm global
```

Shadowing (inner variable hides outer):

```javascript
const value = "outer";

function test() {
  const value = "inner";
  console.log(value);  // "inner" (shadows outer value)
}

test();
console.log(value);  // "outer" (outer scope unchanged)
```

**Output:**
```
inner
outer
```

---

## Lexical Scoping

JavaScript uses **lexical scoping** (also called static scoping): a function's scope is determined by where it's *written* in the code, not where it's *called* from.

```javascript
const name = "Alice";

function greet() {
  console.log(`Hello, ${name}`);  // Uses 'name' from where greet is defined
}

function run() {
  const name = "Bob";
  greet();  // Still prints "Alice", not "Bob"
}

run();
```

**Output:**
```
Hello, Alice
```

Lexical scoping is what enables closures:

```javascript
function createCounter() {
  let count = 0;  // Local to createCounter
  
  return function() {
    count++;      // Accesses count from outer scope
    return count;
  };
}

const counter = createCounter();
console.log(counter());  // 1
console.log(counter());  // 2
console.log(counter());  // 3
```

**Output:**
```
1
2
3
```

> **Note:** The inner function "remembers" `count` even after `createCounter()` finishes executing. This is a closure.

---

## Closures

A **closure** is a function that retains access to its outer (enclosing) function's variables, even after the outer function has finished executing.

### Basic Closure Example

```javascript
function makeGreeter(greeting) {
  // 'greeting' is captured in the closure
  return function(name) {
    return `${greeting}, ${name}!`;
  };
}

const sayHello = makeGreeter("Hello");
const sayHi = makeGreeter("Hi");

console.log(sayHello("Alice"));  // "Hello, Alice!"
console.log(sayHi("Bob"));       // "Hi, Bob!"
```

**Output:**
```
Hello, Alice!
Hi, Bob!
```

Each returned function has its own closure with its own `greeting` value.

### How Closures Work

When a function is created, it maintains a reference to its lexical environment (the scope in which it was created):

```javascript
function outer() {
  const outerVar = "I'm from outer";
  
  function inner() {
    console.log(outerVar);  // Closure over outerVar
  }
  
  return inner;
}

const myFunction = outer();  // outer() finishes executing
myFunction();                // But outerVar is still accessible!
```

**Output:**
```
I'm from outer
```

Even though `outer()` has finished and its execution context is gone, `inner()` still has access to `outerVar` through the closure.

---

## Practical Closure Patterns

### 1. Private Variables (Data Privacy)

Closures enable private state—variables that can't be accessed directly:

```javascript
function createBankAccount(initialBalance) {
  let balance = initialBalance;  // Private variable
  
  return {
    deposit(amount) {
      balance += amount;
      return balance;
    },
    withdraw(amount) {
      if (amount > balance) {
        return "Insufficient funds";
      }
      balance -= amount;
      return balance;
    },
    getBalance() {
      return balance;
    }
  };
}

const account = createBankAccount(100);
console.log(account.getBalance());  // 100
console.log(account.deposit(50));   // 150
console.log(account.withdraw(30));  // 120
console.log(account.balance);       // undefined (private!)
```

**Output:**
```
100
150
120
undefined
```

### 2. Function Factories

Generate specialized functions with pre-configured behavior:

```javascript
function createApiClient(baseUrl, apiKey) {
  // Both parameters are captured in closures
  
  return {
    get(endpoint) {
      return `GET ${baseUrl}/${endpoint} [Auth: ${apiKey}]`;
    },
    post(endpoint, data) {
      return `POST ${baseUrl}/${endpoint} [Auth: ${apiKey}] Data: ${JSON.stringify(data)}`;
    }
  };
}

const chatApi = createApiClient("https://api.chat.com/v1", "sk-abc123");
console.log(chatApi.get("messages"));
console.log(chatApi.post("messages", { text: "Hello" }));
```

**Output:**
```
GET https://api.chat.com/v1/messages [Auth: sk-abc123]
POST https://api.chat.com/v1/messages [Auth: sk-abc123] Data: {"text":"Hello"}
```

### 3. Module Pattern

Create modules with private and public members:

```javascript
const chatModule = (function() {
  // Private variables and functions
  let messageHistory = [];
  
  function formatMessage(text) {
    return `[${new Date().toISOString()}] ${text}`;
  }
  
  // Public API
  return {
    addMessage(text) {
      const formatted = formatMessage(text);
      messageHistory.push(formatted);
    },
    getHistory() {
      return [...messageHistory];  // Return copy, not reference
    },
    clearHistory() {
      messageHistory = [];
    }
  };
})();  // Immediately invoked

chatModule.addMessage("Hello");
chatModule.addMessage("How are you?");
console.log(chatModule.getHistory());
console.log(chatModule.messageHistory);  // undefined (private!)
```

**Output:**
```
[ '[2025-01-07T12:00:00.000Z] Hello', '[2025-01-07T12:00:00.000Z] How are you?' ]
undefined
```

### 4. Event Handlers with Context

Maintain context across asynchronous operations:

```javascript
function createButtonHandler(buttonId) {
  let clickCount = 0;
  
  return function() {
    clickCount++;
    console.log(`Button ${buttonId} clicked ${clickCount} times`);
  };
}

const button1Handler = createButtonHandler("submit");
const button2Handler = createButtonHandler("cancel");

// Simulate clicks
button1Handler();  // "Button submit clicked 1 times"
button1Handler();  // "Button submit clicked 2 times"
button2Handler();  // "Button cancel clicked 1 times"
```

**Output:**
```
Button submit clicked 1 times
Button submit clicked 2 times
Button cancel clicked 1 times
```

---

## Closure Gotchas

### The Loop Problem

A common mistake when using closures in loops:

```javascript
// ❌ Problem: All callbacks share the same 'i'
function createHandlers() {
  const handlers = [];
  
  for (var i = 0; i < 3; i++) {
    handlers.push(function() {
      console.log("Handler", i);
    });
  }
  
  return handlers;
}

const handlers = createHandlers();
handlers[0]();  // 3 (expected 0!)
handlers[1]();  // 3 (expected 1!)
handlers[2]();  // 3 (expected 2!)
```

**Output:**
```
Handler 3
Handler 3
Handler 3
```

**Solution 1:** Use `let` (creates new binding per iteration):

```javascript
function createHandlers() {
  const handlers = [];
  
  for (let i = 0; i < 3; i++) {  // Changed to 'let'
    handlers.push(function() {
      console.log("Handler", i);
    });
  }
  
  return handlers;
}

const handlers = createHandlers();
handlers[0]();  // 0
handlers[1]();  // 1
handlers[2]();  // 2
```

**Output:**
```
Handler 0
Handler 1
Handler 2
```

**Solution 2:** Use an IIFE to capture the value:

```javascript
function createHandlers() {
  const handlers = [];
  
  for (var i = 0; i < 3; i++) {
    (function(index) {  // IIFE captures 'i' as 'index'
      handlers.push(function() {
        console.log("Handler", index);
      });
    })(i);
  }
  
  return handlers;
}
```

---

## Memory Considerations

Closures keep referenced variables in memory:

```javascript
function createHeavyHandler() {
  const largeData = new Array(1000000).fill("data");  // Large array
  
  return function() {
    // Even if largeData isn't used, it's kept in memory
    console.log("Handler executed");
  };
}

// largeData stays in memory as long as the handler exists
```

Break closure references when done:

```javascript
function createHandler() {
  let data = fetchLargeData();
  
  return function(done = false) {
    if (done) {
      data = null;  // Allow garbage collection
      return;
    }
    processData(data);
  };
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `let`/`const` instead of `var` | Prevents common closure bugs with loops and blocks |
| Be mindful of closure memory usage | Closures keep outer variables in memory; clear references when done |
| Use closures for data privacy | Create private variables that can't be accessed directly |
| Prefer named functions in closures | Easier to debug with meaningful function names in stack traces |
| Return copies, not references | `return [...array]` prevents external mutation of private data |
| Use arrow functions to maintain `this` | Arrow functions don't create their own `this`, inheriting from closure |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `var` in loops with closures | Use `let` or create new scope with IIFE |
| Assuming closures copy values | Closures reference variables, not values—changes affect all closures |
| Creating unnecessary closures | Only use closures when you need to preserve state/context |
| Forgetting closures hold references | Explicitly null large objects in closures when done |
| Modifying closed-over variables unexpectedly | Be aware all closures share the same variable reference |
| Using closures for everything | Sometimes a class or module pattern is clearer |

---

## Hands-on Exercise

### Your Task
Create a chat conversation manager using closures. The manager should maintain private message history, provide methods to add messages, get history, and implement a feature to create filtered views of the conversation.

### Requirements
1. Create a `createConversationManager()` function that returns an object with methods
2. Private variables: `messages` array, `conversationId`
3. Public methods:
   - `addMessage(role, content)`: Add a message with timestamp
   - `getHistory()`: Return copy of all messages
   - `getFilteredHistory(role)`: Return messages from specific role
   - `createView(filterFn)`: Return a function that filters messages using the callback
4. Each message should be: `{ role, content, timestamp, id }`

### Expected Result
```javascript
const chat = createConversationManager();
chat.addMessage("user", "Hello");
chat.addMessage("assistant", "Hi there!");
chat.addMessage("user", "How are you?");

console.log(chat.getHistory().length);  // 3
console.log(chat.getFilteredHistory("user").length);  // 2

const userView = chat.createView(msg => msg.role === "user");
console.log(userView().length);  // 2
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `let messages = []` as private variable inside the outer function
- Generate unique IDs with a counter or `Date.now()`
- `getHistory()` should return `[...messages]` (copy)
- `createView` returns a function that closes over the `filterFn` parameter
- Use array methods like `.filter()` for filtering logic
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function createConversationManager() {
  // Private variables
  let messages = [];
  let messageIdCounter = 0;
  const conversationId = `conv-${Date.now()}`;
  
  return {
    addMessage(role, content) {
      const message = {
        id: `msg-${messageIdCounter++}`,
        role,
        content,
        timestamp: Date.now()
      };
      messages.push(message);
      return message;
    },
    
    getHistory() {
      // Return copy to prevent external mutation
      return [...messages];
    },
    
    getFilteredHistory(role) {
      return messages.filter(msg => msg.role === role);
    },
    
    createView(filterFn) {
      // Return a function that closes over filterFn and messages
      return () => messages.filter(filterFn);
    },
    
    getConversationId() {
      return conversationId;
    },
    
    clearHistory() {
      messages = [];
      messageIdCounter = 0;
    }
  };
}

// Test
const chat = createConversationManager();
console.log("Conversation ID:", chat.getConversationId());

chat.addMessage("user", "Hello");
chat.addMessage("assistant", "Hi there!");
chat.addMessage("user", "How are you?");
chat.addMessage("assistant", "I'm doing well, thanks!");

console.log("Total messages:", chat.getHistory().length);
console.log("User messages:", chat.getFilteredHistory("user").length);
console.log("Assistant messages:", chat.getFilteredHistory("assistant").length);

// Create custom view
const userView = chat.createView(msg => msg.role === "user");
console.log("User view:", userView().length);

// Verify privacy
console.log("Direct access to messages:", chat.messages);  // undefined
```
</details>

### Bonus Challenges
- [ ] Add a `getMessageById(id)` method that uses closures
- [ ] Implement message editing with history tracking (using closures)
- [ ] Create a `createSummary` method that returns a closure with cached summary
- [ ] Add rate limiting: max messages per time period using closure state

---

## Summary

✅ Scope determines variable accessibility: global, function, and block scope
✅ Lexical scoping means functions access variables from where they're defined, not called
✅ Closures let functions "remember" variables from their creation context
✅ Use closures for data privacy, function factories, and module patterns
✅ Be careful with closures in loops when using `var`—prefer `let` or IIFE

[Previous: Functions](./03-functions.md) | [Next: Objects and Prototypes](./05-objects-prototypes.md)

---

<!-- 
Sources Consulted:
- MDN Closures: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Closures
- MDN Scope: https://developer.mozilla.org/en-US/docs/Glossary/Scope
- MDN let statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let
- MDN var statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/var
-->
