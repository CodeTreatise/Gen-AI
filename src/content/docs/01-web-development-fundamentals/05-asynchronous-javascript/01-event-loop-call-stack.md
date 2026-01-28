---
title: "Event Loop & Call Stack"
---

# Event Loop & Call Stack

## Introduction

JavaScript is fundamentally different from many programming languages: it's **single-threaded**, meaning it can only execute one piece of code at a time. Yet, it handles countless asynchronous operations simultaneously—fetching data, responding to user clicks, running timers—without blocking. This magic happens through the event loop and call stack, the invisible infrastructure that makes JavaScript's asynchronous nature possible.

Understanding these mechanisms is critical for building AI applications. When your app calls an LLM API, processes responses, and updates the UI, all while remaining responsive to user input, you're relying on JavaScript's asynchronous execution model. Without this knowledge, you'll struggle to debug timing issues, avoid blocking operations, and build performant AI interfaces.

### What We'll Cover
- JavaScript's single-threaded execution model
- The call stack and how function calls are managed
- Task queues (macrotasks) and microtask queues
- How the event loop coordinates everything
- `setTimeout`, `setInterval`, and `requestAnimationFrame` internals
- Blocking vs. non-blocking code patterns

### Prerequisites
- Understanding of JavaScript functions and scope
- Basic knowledge of callbacks
- Familiarity with browser console

---

## JavaScript's Single-Threaded Nature

JavaScript runs on a **single thread of execution**. This means the JavaScript engine can only process one statement at a time. There's no parallel execution of JavaScript code in the main thread.

### Why Single-Threaded?

JavaScript was designed to manipulate the DOM (Document Object Model) in web browsers. If multiple threads could modify the DOM simultaneously, race conditions would create chaos—elements appearing and disappearing unpredictably, conflicting style changes, corrupted state.

The single-threaded model eliminates these issues: only one operation modifies the DOM at any given moment.

### Example: Demonstrating Blocking

```javascript
console.log('Start');

// This blocks the thread for ~3 seconds
const start = Date.now();
while (Date.now() - start < 3000) {
  // Busy-wait loop - blocks everything
}

console.log('End');
```

**Output:**
```
Start
(3-second pause - UI freezes completely)
End
```

During that 3-second loop, **nothing else can happen**: no button clicks register, no animations run, no other code executes. This is blocking behavior, and it's exactly what we need to avoid.

> **Note:** Never use busy-wait loops in production code. This example demonstrates blocking behavior for educational purposes only.

---

## The Call Stack

The **call stack** is a data structure that tracks function execution. When a function is called, it's "pushed" onto the stack. When it returns, it's "popped" off. This follows the Last-In-First-Out (LIFO) principle.

### Call Stack Visualization

```javascript
function greet(name) {
  return `Hello, ${name}!`;
}

function welcome(name) {
  const message = greet(name);
  console.log(message);
}

welcome('Alice');
```

**Call Stack Evolution:**

1. `welcome('Alice')` is called → **pushed onto stack**
2. Inside `welcome`, `greet('Alice')` is called → **pushed onto stack**
3. `greet` returns → **popped from stack**
4. `console.log` is called → **pushed onto stack**
5. `console.log` completes → **popped from stack**
6. `welcome` completes → **popped from stack**
7. Stack is empty → program complete

### Stack Overflow

If functions call themselves recursively without a base case, the stack fills up until it runs out of memory:

```javascript
function infiniteRecursion() {
  infiniteRecursion(); // No base case!
}

infiniteRecursion();
```

**Output:**
```
Uncaught RangeError: Maximum call stack size exceeded
```

Each browser has a limit (typically around 10,000-20,000 calls). When exceeded, you get a stack overflow error.

> **Key Insight:** The call stack only tracks **synchronous** function calls. Asynchronous operations (like `setTimeout` or API calls) are handled differently—they don't sit on the call stack waiting to complete.

---

## Task Queues (Macrotasks)

When asynchronous operations complete, they don't immediately execute. Instead, their callback functions are placed in a **task queue** (also called the macrotask queue).

### Common Macrotasks

- `setTimeout` callbacks
- `setInterval` callbacks
- I/O operations (file reading, network requests)
- UI rendering tasks
- `postMessage` callbacks

### How Task Queues Work

```javascript
console.log('Script start');

setTimeout(() => {
  console.log('setTimeout callback');
}, 0);

console.log('Script end');
```

**Output:**
```
Script start
Script end
setTimeout callback
```

Even with `0` milliseconds, `setTimeout` doesn't execute immediately. Here's why:

1. `console.log('Script start')` executes synchronously
2. `setTimeout` schedules its callback in the task queue (not the call stack)
3. `console.log('Script end')` executes synchronously
4. Call stack is now empty
5. Event loop picks the `setTimeout` callback from the queue
6. `console.log('setTimeout callback')` executes

---

## Microtask Queue

The **microtask queue** has higher priority than the task queue. Microtasks are processed **after the current synchronous code completes but before any macrotasks**.

### Common Microtasks

- Promise `.then()`, `.catch()`, `.finally()` callbacks
- `queueMicrotask()` callbacks
- `MutationObserver` callbacks
- `async`/`await` continuations

### Microtasks vs. Macrotasks

```javascript
console.log('Start');

setTimeout(() => {
  console.log('Timeout'); // Macrotask
}, 0);

Promise.resolve().then(() => {
  console.log('Promise'); // Microtask
});

console.log('End');
```

**Output:**
```
Start
End
Promise
Timeout
```

**Execution Order:**
1. Synchronous code: `Start` → `End`
2. Microtask queue: `Promise`
3. Macrotask queue: `Timeout`

> **Critical Rule:** All microtasks are processed before the next macrotask. If a microtask creates more microtasks, they're processed immediately, potentially starving macrotasks.

---

## The Event Loop

The **event loop** is the conductor orchestrating this entire system. It continuously monitors the call stack and queues, coordinating when code executes.

### Event Loop Algorithm

```
1. Execute synchronous code until call stack is empty
2. Check microtask queue
   - While microtask queue has items:
     - Remove oldest microtask
     - Execute it (may add more microtasks)
3. If microtask queue is empty:
   - Render UI changes (if needed)
   - Check macrotask queue
   - If macrotask queue has items:
     - Remove oldest macrotask
     - Execute it
4. Repeat from step 2
```

### Visualizing the Event Loop

```javascript
console.log('1: Sync start');

setTimeout(() => console.log('2: Macro 1'), 0);

Promise.resolve()
  .then(() => console.log('3: Micro 1'))
  .then(() => console.log('4: Micro 2'));

setTimeout(() => console.log('5: Macro 2'), 0);

console.log('6: Sync end');
```

**Output:**
```
1: Sync start
6: Sync end
3: Micro 1
4: Micro 2
2: Macro 1
5: Macro 2
```

**Step-by-step:**
1. Synchronous code executes: logs `1` and `6`
2. Call stack empty → event loop checks microtask queue
3. `Micro 1` executes, creating `Micro 2`
4. `Micro 2` executes
5. Microtask queue empty → event loop checks macrotask queue
6. `Macro 1` executes
7. Call stack empty → check microtasks (none) → check macrotasks
8. `Macro 2` executes

---

## setTimeout and setInterval

### setTimeout Internals

`setTimeout` schedules a function to run **after a minimum delay**. It's not guaranteed to run exactly at that time.

```javascript
setTimeout(() => {
  console.log('Executed after ~100ms');
}, 100);
```

**What Happens:**
1. Timer starts in the browser's timer module (not JavaScript)
2. After 100ms, callback is placed in the macrotask queue
3. Event loop picks it up when the call stack is empty

### Minimum Delay Clamping

Browsers enforce a **minimum 4ms delay** after 5 nested `setTimeout` calls:

```javascript
let count = 0;
function recursiveTimeout() {
  count++;
  console.log(`Call ${count}, Time: ${Date.now()}`);
  if (count < 10) {
    setTimeout(recursiveTimeout, 0);
  }
}
recursiveTimeout();
```

**Output (timing may vary):**
```
Call 1, Time: 1640000000000
Call 2, Time: 1640000000001
Call 3, Time: 1640000000002
Call 4, Time: 1640000000003
Call 5, Time: 1640000000004
Call 6, Time: 1640000000008  ← 4ms minimum kicks in
Call 7, Time: 1640000000012
...
```

### setInterval Caveats

`setInterval` can cause issues if the callback takes longer than the interval:

```javascript
setInterval(() => {
  // If this takes 150ms to complete...
  console.log('Task start:', Date.now());
  const start = Date.now();
  while (Date.now() - start < 150) {}
  console.log('Task end:', Date.now());
}, 100);
```

The interval tries to fire every 100ms, but tasks overlap if execution exceeds the interval. Browsers handle this by **skipping intervals** if a callback is still running.

**Better pattern: Recursive `setTimeout`**

```javascript
function recursiveTask() {
  console.log('Task executed');
  setTimeout(recursiveTask, 100); // Schedules after task completes
}
recursiveTask();
```

This ensures tasks never overlap.

---

## requestAnimationFrame

For animations, `requestAnimationFrame` is far superior to `setTimeout`. It syncs with the browser's refresh rate (~60 FPS = 16.67ms per frame).

### Why requestAnimationFrame?

```javascript
// ❌ BAD: Inconsistent timing, may miss frames
setInterval(() => {
  element.style.left = (parseInt(element.style.left) + 1) + 'px';
}, 16);

// ✅ GOOD: Syncs with display refresh
function animate() {
  element.style.left = (parseInt(element.style.left) + 1) + 'px';
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
```

**Benefits:**
- Pauses when tab is inactive (saves CPU/battery)
- Syncs with monitor refresh rate
- Batch DOM updates efficiently
- Avoids layout thrashing

**Output:** Smooth, efficient 60 FPS animation

---

## Blocking vs. Non-Blocking Code

### Blocking Example

```javascript
function fetchDataSync() {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', 'https://api.example.com/data', false); // Synchronous!
  xhr.send();
  return xhr.responseText;
}

console.log('Start');
const data = fetchDataSync(); // Blocks until response received
console.log('End');
```

The UI freezes completely until the network request completes. **Never do this.**

### Non-Blocking Pattern

```javascript
function fetchDataAsync() {
  return fetch('https://api.example.com/data');
}

console.log('Start');
fetchDataAsync().then(response => response.json()).then(data => {
  console.log('Data received:', data);
});
console.log('End');
```

**Output:**
```
Start
End
(later, when fetch completes)
Data received: { ... }
```

The UI remains responsive while waiting for the network.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Never block the main thread | Keeps UI responsive; users can interact while processing |
| Use `requestAnimationFrame` for animations | Syncs with display refresh, pauses when tab inactive |
| Prefer Promises/async-await over callbacks | Cleaner code, better error handling, easier to reason about |
| Break long tasks into chunks | Prevents UI freezing; use `setTimeout(fn, 0)` to yield |
| Understand microtask priority | Promises execute before timers; critical for execution order |
| Avoid `setInterval` for long-running tasks | Use recursive `setTimeout` to prevent overlap |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Assuming `setTimeout(fn, 0)` runs immediately | It's queued as a macrotask; synchronous code runs first |
| Using busy-wait loops | Use asynchronous patterns (timers, promises, async/await) |
| Expecting precise timing from `setTimeout` | It's a **minimum** delay; actual time varies with load |
| Creating infinite microtask loops | Always have an exit condition; starves macrotasks |
| Synchronous XHR or file operations | Use async APIs (fetch, FileReader with callbacks) |
| Forgetting about task queue priority | Microtasks always execute before macrotasks |

---

## Hands-on Exercise

### Your Task

Build an event loop visualizer that demonstrates the execution order of synchronous code, microtasks, and macrotasks. The tool should log each step with timing information to show how the event loop processes different types of operations.

### Requirements

1. Create buttons to trigger:
   - Synchronous code
   - `setTimeout` (macrotask)
   - Promise (microtask)
   - Mixed sequence of all three
2. Log each operation with:
   - Type (Sync/Micro/Macro)
   - Timestamp
   - Execution order number
3. Display logs in a `<div>` to visualize the event loop's behavior

### Expected Result

Clicking "Run Mixed Sequence" should show:
```
[1] Sync: Start - 1640000000000
[2] Sync: Middle - 1640000000001
[3] Sync: End - 1640000000002
[4] Microtask: Promise 1 - 1640000000003
[5] Microtask: Promise 2 - 1640000000004
[6] Macrotask: Timeout 1 - 1640000000020
[7] Macrotask: Timeout 2 - 1640000000025
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a global counter to track execution order
- Store timestamps with `Date.now()` or `performance.now()`
- Create a helper function `log(type, message)` to format entries
- Use `Promise.resolve().then()` for microtasks
- Use `setTimeout(() => {}, 0)` for macrotasks
- Clear the log display at the start of each sequence

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html>
<head>
  <title>Event Loop Visualizer</title>
  <style>
    #log {
      font-family: monospace;
      background: #1e1e1e;
      color: #d4d4d4;
      padding: 20px;
      border-radius: 8px;
      min-height: 300px;
      margin-top: 20px;
    }
    .log-entry {
      padding: 4px 0;
    }
    .sync { color: #4ec9b0; }
    .micro { color: #dcdcaa; }
    .macro { color: #ce9178; }
    button {
      padding: 10px 20px;
      margin: 5px;
      font-size: 16px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h1>Event Loop Visualizer</h1>
  <button id="sync-btn">Run Synchronous</button>
  <button id="micro-btn">Run Microtask</button>
  <button id="macro-btn">Run Macrotask</button>
  <button id="mixed-btn">Run Mixed Sequence</button>
  <button id="clear-btn">Clear Log</button>
  <div id="log"></div>

  <script>
    let executionOrder = 0;
    const logDiv = document.getElementById('log');

    function log(type, message) {
      executionOrder++;
      const timestamp = Date.now();
      const className = type.toLowerCase();
      const entry = document.createElement('div');
      entry.className = `log-entry ${className}`;
      entry.textContent = `[${executionOrder}] ${type}: ${message} - ${timestamp}`;
      logDiv.appendChild(entry);
    }

    function clearLog() {
      executionOrder = 0;
      logDiv.innerHTML = '';
    }

    document.getElementById('sync-btn').addEventListener('click', () => {
      clearLog();
      log('Sync', 'First operation');
      log('Sync', 'Second operation');
      log('Sync', 'Third operation');
    });

    document.getElementById('micro-btn').addEventListener('click', () => {
      clearLog();
      Promise.resolve().then(() => log('Microtask', 'Promise 1'));
      Promise.resolve().then(() => log('Microtask', 'Promise 2'));
      log('Sync', 'After promise creation');
    });

    document.getElementById('macro-btn').addEventListener('click', () => {
      clearLog();
      setTimeout(() => log('Macrotask', 'Timeout 1'), 0);
      setTimeout(() => log('Macrotask', 'Timeout 2'), 0);
      log('Sync', 'After setTimeout calls');
    });

    document.getElementById('mixed-btn').addEventListener('click', () => {
      clearLog();
      
      log('Sync', 'Start');
      
      setTimeout(() => log('Macrotask', 'Timeout 1'), 0);
      
      Promise.resolve().then(() => {
        log('Microtask', 'Promise 1');
        Promise.resolve().then(() => log('Microtask', 'Promise 1.1 (nested)'));
      });
      
      log('Sync', 'Middle');
      
      setTimeout(() => log('Macrotask', 'Timeout 2'), 0);
      
      Promise.resolve().then(() => log('Microtask', 'Promise 2'));
      
      log('Sync', 'End');
    });

    document.getElementById('clear-btn').addEventListener('click', clearLog);
  </script>
</body>
</html>
```

**Key Features:**
- Color-coded by operation type (sync/micro/macro)
- Shows execution order and timestamp
- Demonstrates microtask priority (executes before macrotasks)
- Nested microtasks show immediate execution
- Mixed sequence proves event loop behavior

</details>

### Bonus Challenges

- [ ] Add a "Recursive Microtask" button that creates 5 chained promises to show microtask queue processing
- [ ] Implement a `requestAnimationFrame` example that shows frame timing
- [ ] Add a long-running task button that blocks for 2 seconds to demonstrate UI freezing
- [ ] Create a visualization showing the call stack depth during recursive function calls

---

## Summary

✅ JavaScript is **single-threaded**: only one operation executes at a time  
✅ The **call stack** tracks synchronous function execution (LIFO)  
✅ Asynchronous operations use **task queues**: microtasks (promises) and macrotasks (timers)  
✅ The **event loop** coordinates execution: sync code → microtasks → UI rendering → macrotasks  
✅ `setTimeout(fn, 0)` doesn't run immediately; it queues as a macrotask after current code  
✅ **Microtasks have priority**: all promises execute before any timeout  
✅ Never block the main thread—use asynchronous patterns for responsiveness

**Next:** [Callbacks & Patterns](./02-callbacks-patterns.md)

---

<!-- 
Sources Consulted:
- MDN JavaScript Execution Model: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Execution_model
- MDN setTimeout: https://developer.mozilla.org/en-US/docs/Web/API/setTimeout
- MDN Event Loop: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Event_loop
- MDN Promise (Microtask Queue): https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise
-->
