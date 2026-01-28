---
title: "Console Debugging Techniques"
---

# Console Debugging Techniques

## Introduction

The Console is your JavaScript command center. It's where errors appear, where you test code snippets, and where strategic `console.log` statements reveal what's happening inside your application.

But `console.log` is just the beginning. The Console API offers powerful methods for formatting output, measuring performance, tracing execution, and organizing complex debugging sessions.

### What We'll Cover

- console.log, warn, error, info
- console.table for arrays/objects
- console.time and console.timeEnd
- console.trace for stack traces
- console.group for organization
- Console filtering
- Live expressions

### Prerequisites

- JavaScript fundamentals
- Basic DevTools familiarity

---

## Opening the Console

| Method | How |
|--------|-----|
| DevTools Console tab | `F12` then click Console |
| Direct to Console | `Ctrl+Shift+J` (Win/Linux) or `Cmd+Option+J` (Mac) |
| Console drawer | Press `Esc` in any DevTools panel |

The Console has two areas:
1. **Log output** - Messages from your code and the browser
2. **Command line** - REPL for running JavaScript

---

## Basic Console Methods

### console.log()

The workhorse for general output:

```javascript
console.log('Hello, Console!');
console.log('User:', user);
console.log('Count:', count, 'Items:', items);
```

**Output:**
```
Hello, Console!
User: {name: 'Alice', id: 123}
Count: 5 Items: ['a', 'b', 'c']
```

### console.warn()

Yellow warning messages:

```javascript
console.warn('Deprecated function used');
console.warn('API rate limit approaching:', remaining);
```

**Output:**
```
⚠️ Deprecated function used
⚠️ API rate limit approaching: 5
```

### console.error()

Red error messages with stack trace:

```javascript
console.error('Failed to load user data');
console.error('Error:', error.message);
```

**Output:**
```
❌ Failed to load user data
❌ Error: Network request failed
    at fetchUser (app.js:45)
    at async main (app.js:12)
```

### console.info()

Informational messages (same as log in most browsers):

```javascript
console.info('Application started');
console.info('Version:', APP_VERSION);
```

### console.debug()

Debug-level messages (hidden by default in some browsers):

```javascript
console.debug('Entering function with params:', params);
```

> **Tip:** In Chrome, enable "Verbose" log level to see debug messages.

---

## String Formatting

### Template Literals (Recommended)

```javascript
const user = 'Alice';
const score = 95;

console.log(`User ${user} scored ${score} points`);
// User Alice scored 95 points
```

### Format Specifiers

```javascript
console.log('Hello, %s!', 'World');           // %s = string
console.log('Count: %d items', 42);           // %d = number
console.log('Value: %f', 3.14159);            // %f = float
console.log('Object: %o', { a: 1 });          // %o = object
console.log('DOM: %O', document.body);        // %O = DOM element
```

### Styling Console Output

```javascript
console.log(
  '%cStyled Text',
  'color: blue; font-size: 20px; font-weight: bold;'
);

console.log(
  '%cSuccess%c - Operation completed',
  'background: green; color: white; padding: 2px 6px; border-radius: 3px;',
  'color: inherit;'
);
```

**Output:**
```
Styled Text (large blue bold)
Success - Operation completed (green badge)
```

---

## console.table()

Display arrays and objects as formatted tables:

```javascript
const users = [
  { id: 1, name: 'Alice', role: 'Admin' },
  { id: 2, name: 'Bob', role: 'User' },
  { id: 3, name: 'Charlie', role: 'User' }
];

console.table(users);
```

**Output:**
```
┌─────────┬────┬───────────┬─────────┐
│ (index) │ id │   name    │  role   │
├─────────┼────┼───────────┼─────────┤
│    0    │ 1  │  'Alice'  │ 'Admin' │
│    1    │ 2  │   'Bob'   │ 'User'  │
│    2    │ 3  │ 'Charlie' │ 'User'  │
└─────────┴────┴───────────┴─────────┘
```

### Selecting Columns

```javascript
// Only show specific properties
console.table(users, ['name', 'role']);
```

### Objects as Tables

```javascript
const stats = {
  visits: 1234,
  pageViews: 5678,
  bounceRate: 0.42
};

console.table(stats);
```

---

## Timing with console.time()

Measure code execution time:

```javascript
console.time('fetch-users');

const response = await fetch('/api/users');
const users = await response.json();

console.timeEnd('fetch-users');
// fetch-users: 234.56ms
```

### Multiple Timers

```javascript
console.time('total');

console.time('fetch');
const data = await fetchData();
console.timeEnd('fetch');  // fetch: 150ms

console.time('process');
const processed = processData(data);
console.timeEnd('process');  // process: 45ms

console.timeEnd('total');  // total: 198ms
```

### console.timeLog()

Check elapsed time without stopping:

```javascript
console.time('operation');

await step1();
console.timeLog('operation', 'after step 1');  // operation: 100ms after step 1

await step2();
console.timeLog('operation', 'after step 2');  // operation: 250ms after step 2

await step3();
console.timeEnd('operation');  // operation: 400ms
```

---

## Stack Traces with console.trace()

See the call stack at any point:

```javascript
function calculateTotal(items) {
  console.trace('Calculating total');
  return items.reduce((sum, item) => sum + item.price, 0);
}

function processOrder(order) {
  const total = calculateTotal(order.items);
  return { ...order, total };
}

function handleSubmit() {
  const order = getOrderData();
  processOrder(order);
}
```

**Output:**
```
Calculating total
    at calculateTotal (app.js:2)
    at processOrder (app.js:7)
    at handleSubmit (app.js:12)
    at HTMLButtonElement.onclick (index.html:15)
```

---

## Grouping with console.group()

Organize related logs:

```javascript
console.group('User Authentication');
console.log('Checking credentials...');
console.log('User found:', user.email);
console.log('Verifying password...');
console.log('Authentication successful');
console.groupEnd();
```

**Output:**
```
▼ User Authentication
    Checking credentials...
    User found: alice@example.com
    Verifying password...
    Authentication successful
```

### Collapsed Groups

Start collapsed (user must click to expand):

```javascript
console.groupCollapsed('API Response (click to expand)');
console.log('Status:', response.status);
console.log('Headers:', response.headers);
console.log('Body:', data);
console.groupEnd();
```

### Nested Groups

```javascript
console.group('Request');
console.log('URL:', url);
console.log('Method:', method);

console.group('Headers');
console.log('Content-Type:', contentType);
console.log('Authorization:', '***');
console.groupEnd();

console.group('Body');
console.log(body);
console.groupEnd();

console.groupEnd();
```

---

## Assertions with console.assert()

Log only if a condition is false:

```javascript
const age = 15;
console.assert(age >= 18, 'User is underage:', age);
// Assertion failed: User is underage: 15

const validUser = { id: 1, name: 'Alice' };
console.assert(validUser.id, 'User must have an ID');
// (nothing logged - assertion passed)
```

Useful for development-time sanity checks.

---

## Counting with console.count()

Count how many times code executes:

```javascript
function processItem(item) {
  console.count('processItem called');
  // ... processing logic
}

processItem(a);  // processItem called: 1
processItem(b);  // processItem called: 2
processItem(c);  // processItem called: 3

console.countReset('processItem called');
processItem(d);  // processItem called: 1
```

### Labeled Counts

```javascript
function handleEvent(type) {
  console.count(type);
}

handleEvent('click');    // click: 1
handleEvent('scroll');   // scroll: 1
handleEvent('click');    // click: 2
handleEvent('scroll');   // scroll: 2
handleEvent('click');    // click: 3
```

---

## Console Filtering

The Console can get noisy. Use filters to find what you need.

### Log Level Filter

Click the filter dropdown:
- **Verbose** - All messages including debug
- **Info** - Info, warnings, and errors
- **Warnings** - Warnings and errors only
- **Errors** - Errors only

### Text Filter

Type in the filter box:
- `user` - Show messages containing "user"
- `-error` - Hide messages containing "error"
- `/regex/` - Use regular expressions
- `url:app.js` - Filter by source file

### Message Source

Filter by source:
- User messages (your console.log)
- Violations (performance issues)
- Errors
- Warnings
- Info
- Network messages

---

## Live Expressions

Watch values update in real-time without cluttering the console.

### Creating Live Expressions

1. Click the **eye icon** in Console toolbar
2. Enter a JavaScript expression
3. The value updates automatically

```javascript
// Useful live expressions:
document.activeElement          // Currently focused element
performance.memory.usedJSHeapSize  // Memory usage
Date.now()                      // Current timestamp
myApp.state.user               // Application state
```

### Multiple Expressions

Add multiple live expressions to monitor several values simultaneously.

---

## Console in JavaScript

### Conditional Logging

```javascript
const DEBUG = true;

function debug(...args) {
  if (DEBUG) console.log('[DEBUG]', ...args);
}

debug('User clicked button');
// [DEBUG] User clicked button
```

### Production-Safe Logging

```javascript
const logger = {
  log: (...args) => {
    if (process.env.NODE_ENV !== 'production') {
      console.log(...args);
    }
  },
  error: (...args) => {
    // Always log errors
    console.error(...args);
    // Also send to error tracking service
    errorService.capture(args);
  }
};
```

### Object Inspection

```javascript
// console.log shows a live reference (may change)
const obj = { count: 0 };
console.log(obj);
obj.count = 5;
// Expanding in console shows count: 5 (current value)

// To capture a snapshot:
console.log(JSON.parse(JSON.stringify(obj)));
// Or:
console.log({ ...obj });
```

---

## Hands-on Exercise

### Your Task

Debug a simulated API call using console techniques.

```javascript
async function fetchUserData(userId) {
  console.group(`Fetching user ${userId}`);
  console.time('api-call');
  
  try {
    const response = await fetch(`/api/users/${userId}`);
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      console.error('Request failed:', response.statusText);
      throw new Error(response.statusText);
    }
    
    const data = await response.json();
    console.timeEnd('api-call');
    console.table(data);
    
    return data;
  } catch (error) {
    console.timeEnd('api-call');
    console.error('Fetch error:', error);
    console.trace('Error occurred here');
    throw error;
  } finally {
    console.groupEnd();
  }
}
```

### Practice

1. Add `console.count()` to track how many times the function is called
2. Add `console.warn()` if the response is slow (> 1 second)
3. Use styled console output to highlight success/failure

---

## Summary

| Method | Purpose |
|--------|---------|
| `console.log()` | General output |
| `console.warn()` | Warning (yellow) |
| `console.error()` | Error (red) + stack |
| `console.table()` | Format arrays/objects as tables |
| `console.time/timeEnd()` | Measure duration |
| `console.trace()` | Show call stack |
| `console.group()` | Organize related logs |
| `console.assert()` | Log if condition false |
| `console.count()` | Count executions |

**Next:** [Network Tab Analysis](./03-network-tab-analysis.md)

---

## Further Reading

- [MDN Console API](https://developer.mozilla.org/en-US/docs/Web/API/console)
- [Chrome Console Reference](https://developer.chrome.com/docs/devtools/console/api/)

<!-- 
Sources Consulted:
- MDN Console: https://developer.mozilla.org/en-US/docs/Web/API/console
- Chrome DevTools Console: https://developer.chrome.com/docs/devtools/console/
-->
