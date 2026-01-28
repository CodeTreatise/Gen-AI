---
title: "Callbacks & Patterns"
---

# Callbacks & Patterns

## Introduction

Before promises and async/await revolutionized JavaScript, **callbacks** were the only way to handle asynchronous operations. A callback is simply a function passed as an argument to another function, to be executed later when an operation completes. While modern JavaScript provides better alternatives, callbacks remain fundamental—understanding them helps you grasp why promises exist and how to work with legacy code.

Many AI libraries and APIs still use callback-based patterns, especially in Node.js environments. You'll encounter callbacks in file operations, stream processing, event handlers, and older npm packages. Knowing how to work with callbacks—and when to avoid them—is essential for building robust AI applications.

### What We'll Cover
- Callback function fundamentals
- Callback hell and the pyramid of doom
- Error-first callback pattern
- Strategies for managing callback complexity
- When callbacks are still the right choice

### Prerequisites
- Understanding of JavaScript functions
- Familiarity with higher-order functions
- Knowledge of the event loop (previous lesson)

---

## Callback Function Basics

A **callback** is a function passed to another function as an argument, intended to be invoked at a later time.

### Simple Callback Example

```javascript
function greet(name, callback) {
  console.log('Hello, ' + name);
  callback();
}

function sayGoodbye() {
  console.log('Goodbye!');
}

greet('Alice', sayGoodbye);
```

**Output:**
```
Hello, Alice
Goodbye!
```

**What happened:**
1. `greet` is called with `'Alice'` and the `sayGoodbye` function
2. `greet` logs the greeting
3. `greet` invokes `callback()`, which executes `sayGoodbye`

### Asynchronous Callbacks

Callbacks become powerful with asynchronous operations:

```javascript
function fetchUserData(userId, callback) {
  console.log('Fetching user data...');
  
  setTimeout(() => {
    const userData = { id: userId, name: 'Alice', email: 'alice@example.com' };
    callback(userData);
  }, 1000);
}

fetchUserData(123, (user) => {
  console.log('User data received:', user);
});

console.log('Request sent, waiting for response...');
```

**Output:**
```
Fetching user data...
Request sent, waiting for response...
(after 1 second)
User data received: { id: 123, name: 'Alice', email: 'alice@example.com' }
```

The callback executes **after** the synchronous code completes and the timer fires.

### Anonymous Callbacks

Callbacks are often defined inline as anonymous functions:

```javascript
setTimeout(() => {
  console.log('This is an anonymous callback');
}, 1000);

// Equivalent to:
function myCallback() {
  console.log('This is an anonymous callback');
}
setTimeout(myCallback, 1000);
```

Arrow functions (`=>`) are preferred for concise anonymous callbacks.

---

## Callback Hell (Pyramid of Doom)

When multiple asynchronous operations depend on each other, nested callbacks create deeply indented, hard-to-read code—known as **callback hell** or the **pyramid of doom**.

### The Problem

```javascript
getUserById(userId, (user) => {
  getPostsByUser(user.id, (posts) => {
    getCommentsForPost(posts[0].id, (comments) => {
      getLikesForComment(comments[0].id, (likes) => {
        console.log('Likes:', likes);
      });
    });
  });
});
```

Each level of nesting represents another asynchronous operation that depends on the previous result. This pattern has severe drawbacks:

1. **Hard to read**: Indentation grows horizontally, making code difficult to follow
2. **Difficult to debug**: Stack traces become confusing with multiple nested callbacks
3. **Error handling nightmare**: Must handle errors at every level
4. **Brittle**: Changes require modifying deeply nested code

### Real-World Callback Hell Example

```javascript
function processAIRequest(userInput, callback) {
  // Step 1: Authenticate user
  authenticateUser((error, token) => {
    if (error) return callback(error);
    
    // Step 2: Load AI model
    loadModel(token, (error, model) => {
      if (error) return callback(error);
      
      // Step 3: Preprocess input
      preprocessInput(userInput, (error, processed) => {
        if (error) return callback(error);
        
        // Step 4: Generate response
        model.generate(processed, (error, response) => {
          if (error) return callback(error);
          
          // Step 5: Log for analytics
          logAnalytics(response, (error) => {
            if (error) return callback(error);
            
            // Finally done!
            callback(null, response);
          });
        });
      });
    });
  });
}
```

This is the classic pyramid of doom. Notice how error handling repeats at every level, and the actual logic is buried in nested callbacks.

> **Note:** This pattern is why promises and async/await were invented. They flatten this structure and make error handling cleaner.

---

## Error-First Callback Pattern

Node.js established a convention for callback signatures: **the first parameter is always an error object** (or `null` if no error), and subsequent parameters contain result data.

### Error-First Convention

```javascript
function readFile(filename, callback) {
  // Simulate file reading
  setTimeout(() => {
    if (filename === 'missing.txt') {
      callback(new Error('File not found'), null);
    } else {
      callback(null, 'File content here');
    }
  }, 100);
}

// Usage:
readFile('data.txt', (error, data) => {
  if (error) {
    console.error('Error reading file:', error.message);
    return; // Exit early on error
  }
  console.log('File data:', data);
});
```

**Output (success case):**
```
File data: File content here
```

**Output (error case with 'missing.txt'):**
```
Error reading file: File not found
```

### Why Error-First?

1. **Consistency**: All callbacks follow the same pattern
2. **Forces error handling**: Can't accidentally ignore errors
3. **Early returns**: Easy to exit on error with `if (error) return`
4. **Clear intent**: Separates error path from success path

### Error-First in Practice

```javascript
function fetchUserData(userId, callback) {
  if (typeof userId !== 'number') {
    return callback(new Error('userId must be a number'), null);
  }
  
  setTimeout(() => {
    if (userId < 0) {
      callback(new Error('userId must be positive'), null);
    } else {
      callback(null, { id: userId, name: 'User ' + userId });
    }
  }, 100);
}

fetchUserData('invalid', (error, user) => {
  if (error) {
    console.error('Error:', error.message);
    return;
  }
  console.log('User:', user);
});
```

**Output:**
```
Error: userId must be a number
```

---

## Managing Callback Complexity

While callbacks are harder to manage than promises, several patterns help tame the complexity.

### 1. Named Functions (Extract Callbacks)

Instead of nesting anonymous functions, extract them as named functions:

```javascript
// ❌ BAD: Nested anonymous callbacks
fetchUser(userId, (user) => {
  fetchPosts(user.id, (posts) => {
    fetchComments(posts[0].id, (comments) => {
      console.log(comments);
    });
  });
});

// ✅ GOOD: Named functions
function handleUser(user) {
  fetchPosts(user.id, handlePosts);
}

function handlePosts(posts) {
  fetchComments(posts[0].id, handleComments);
}

function handleComments(comments) {
  console.log(comments);
}

fetchUser(userId, handleUser);
```

**Benefits:**
- Flattens nesting
- Functions are reusable
- Easier to test individually
- Stack traces are clearer

### 2. Early Returns for Error Handling

Handle errors immediately and return to avoid nesting:

```javascript
function processData(callback) {
  fetchUser((error, user) => {
    if (error) return callback(error);
    
    fetchPosts(user.id, (error, posts) => {
      if (error) return callback(error);
      
      callback(null, { user, posts });
    });
  });
}
```

This keeps error handling at the same indentation level.

### 3. Callback Libraries (async.js)

Before promises became standard, libraries like `async.js` provided utilities for managing callbacks:

```javascript
// Using async.waterfall to avoid nesting
async.waterfall([
  (callback) => {
    fetchUser(userId, callback);
  },
  (user, callback) => {
    fetchPosts(user.id, callback);
  },
  (posts, callback) => {
    fetchComments(posts[0].id, callback);
  }
], (error, comments) => {
  if (error) return console.error(error);
  console.log(comments);
});
```

> **Modern Alternative:** Use promises with `.then()` chaining or `async`/`await` instead of callback libraries.

### 4. Promisify Pattern

Convert callback-based functions to promises:

```javascript
function promisify(callbackFn) {
  return function(...args) {
    return new Promise((resolve, reject) => {
      callbackFn(...args, (error, result) => {
        if (error) reject(error);
        else resolve(result);
      });
    });
  };
}

// Usage:
const fetchUserPromise = promisify(fetchUser);

fetchUserPromise(userId)
  .then(user => console.log(user))
  .catch(error => console.error(error));
```

Node.js provides `util.promisify()` for this exact purpose.

---

## When Callbacks Are Still Useful

Despite promises being superior for most async operations, callbacks remain appropriate in certain scenarios.

### Event Listeners

DOM event handlers naturally use callbacks:

```javascript
button.addEventListener('click', (event) => {
  console.log('Button clicked!', event.target);
});
```

This isn't a "callback hell" scenario because events fire independently, not in sequence.

### Repeated Invocations

When a function needs to be called multiple times (not just once when an operation completes):

```javascript
// Callback is invoked for each item
array.forEach((item) => {
  console.log(item);
});

// Callback is invoked on every interval
const intervalId = setInterval(() => {
  console.log('Tick');
}, 1000);
```

Promises fulfill/reject only once, so callbacks are better for repeated actions.

### Synchronous Higher-Order Functions

Array methods use callbacks synchronously:

```javascript
const doubled = [1, 2, 3].map(num => num * 2);
const evens = [1, 2, 3, 4].filter(num => num % 2 === 0);
const sum = [1, 2, 3].reduce((acc, num) => acc + num, 0);
```

**Output:**
```
[2, 4, 6]
[2, 4]
6
```

These execute immediately, not asynchronously.

### Streaming Data

When processing data in chunks (like reading large files):

```javascript
const fs = require('fs');
const stream = fs.createReadStream('large-file.txt');

stream.on('data', (chunk) => {
  console.log('Received chunk:', chunk.length, 'bytes');
});

stream.on('end', () => {
  console.log('Finished reading file');
});
```

Promises would only trigger once at the end, missing intermediate chunks.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always use error-first callbacks in Node.js | Consistency with ecosystem; clear error handling |
| Handle errors at every callback level | Prevents silent failures and unhandled errors |
| Extract named functions instead of nesting | Improves readability, reusability, and testability |
| Convert to promises when possible | Modern syntax, better composition, easier error handling |
| Use `util.promisify` for Node.js callbacks | Standard library solution, well-tested and reliable |
| Avoid deeply nested callbacks (>3 levels) | Indicates need to refactor with promises or named functions |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting to call the callback | Always invoke callback—even on error (`callback(error)`) |
| Calling callback multiple times | Guard with `if` or early `return` to ensure single invocation |
| Not handling errors | Always check error parameter first: `if (error) return ...` |
| Deeply nested anonymous functions | Extract named functions or convert to promises |
| Mixing sync and async patterns | Be consistent—if function is async, always use callback |
| Ignoring callback return value | Callbacks usually return `undefined`; don't chain like promises |

---

## Hands-on Exercise

### Your Task

Build a simple file processing system using callbacks that demonstrates error handling, sequencing, and avoiding callback hell. The system should read a file, transform its content, and save the result—all using callback-based functions.

### Requirements

1. Create three callback-based functions:
   - `readFile(filename, callback)`: Simulates reading file content
   - `transformContent(content, callback)`: Converts text to uppercase
   - `writeFile(filename, content, callback)`: Simulates writing file
2. Implement proper error-first callbacks
3. Chain the operations: read → transform → write
4. Use named functions to avoid nesting
5. Display status messages and handle errors gracefully

### Expected Result

```
Starting file processing...
Reading input.txt...
Read successful: Hello, World!
Transforming content...
Transform successful: HELLO, WORLD!
Writing output.txt...
Write successful
Processing complete!
```

If any step fails (e.g., file not found):
```
Starting file processing...
Reading missing.txt...
Error reading file: File not found
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `setTimeout` to simulate async operations (100ms delay)
- Store a mock file system in an object: `{ 'input.txt': 'Hello, World!' }`
- For `readFile`, check if filename exists in mock file system
- For `writeFile`, add new entry to mock file system
- Use `if (error) return callback(error)` pattern
- Create a `processFile` function that chains all operations
- Name your intermediate callbacks: `handleRead`, `handleTransform`, `handleWrite`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
// Mock file system
const fileSystem = {
  'input.txt': 'Hello, World!',
  'data.txt': 'This is sample data'
};

// Simulate reading a file
function readFile(filename, callback) {
  console.log(`Reading ${filename}...`);
  
  setTimeout(() => {
    if (fileSystem[filename]) {
      const content = fileSystem[filename];
      console.log(`Read successful: ${content}`);
      callback(null, content);
    } else {
      callback(new Error('File not found'), null);
    }
  }, 100);
}

// Simulate transforming content
function transformContent(content, callback) {
  console.log('Transforming content...');
  
  setTimeout(() => {
    try {
      const transformed = content.toUpperCase();
      console.log(`Transform successful: ${transformed}`);
      callback(null, transformed);
    } catch (error) {
      callback(error, null);
    }
  }, 100);
}

// Simulate writing a file
function writeFile(filename, content, callback) {
  console.log(`Writing ${filename}...`);
  
  setTimeout(() => {
    fileSystem[filename] = content;
    console.log('Write successful');
    callback(null);
  }, 100);
}

// Process file with named callbacks (avoid nesting)
function processFile(inputFile, outputFile) {
  console.log('Starting file processing...\n');
  
  // Step 1: Read file
  readFile(inputFile, handleRead);
  
  function handleRead(error, content) {
    if (error) {
      console.error('Error reading file:', error.message);
      return;
    }
    
    // Step 2: Transform content
    transformContent(content, handleTransform);
  }
  
  function handleTransform(error, transformed) {
    if (error) {
      console.error('Error transforming content:', error.message);
      return;
    }
    
    // Step 3: Write file
    writeFile(outputFile, transformed, handleWrite);
  }
  
  function handleWrite(error) {
    if (error) {
      console.error('Error writing file:', error.message);
      return;
    }
    
    console.log('\nProcessing complete!');
  }
}

// Run the processor
processFile('input.txt', 'output.txt');

// Test error handling
setTimeout(() => {
  console.log('\n--- Testing Error Handling ---\n');
  processFile('missing.txt', 'output.txt');
}, 1500);
```

**Key Features:**
- Error-first callbacks throughout
- Named functions avoid callback hell
- Early returns on errors
- Clear console output shows each step
- Mock file system for testing
- Demonstrates both success and error paths

</details>

### Bonus Challenges

- [ ] Add a `validateContent` function that checks if content is empty
- [ ] Implement a `retry` mechanism that attempts failed operations 3 times
- [ ] Convert all three functions to return promises using a custom `promisify` helper
- [ ] Add a `processMultipleFiles` function that processes an array of files

---

## Summary

✅ **Callbacks** are functions passed as arguments to be executed later  
✅ **Callback hell** occurs with nested async operations; use named functions or promises  
✅ **Error-first pattern** (`callback(error, result)`) is Node.js convention  
✅ Always handle errors: `if (error) return callback(error)`  
✅ Extract named functions instead of nesting anonymous callbacks  
✅ Callbacks remain useful for events, repeated invocations, and streaming  
✅ Modern code should prefer promises and async/await over callbacks

**Next:** [Promises](./03-promises.md)

---

<!-- 
Sources Consulted:
- MDN Promise Guide: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises
- MDN async function: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function
- Node.js Error Handling: Callback patterns and error-first convention
-->
