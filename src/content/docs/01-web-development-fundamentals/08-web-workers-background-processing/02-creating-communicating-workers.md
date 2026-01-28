---
title: "Creating and Communicating with Workers"
---

# Creating and Communicating with Workers

## Introduction

Now that we understand what Web Workers are, let's dive into the practical aspects: how to create workers, send data to them, receive results, handle errors, and clean up when done.

The communication between the main thread and workers happens through a **message-passing system**. This pattern ensures thread safety—there's no shared memory that could lead to race conditions.

### What We'll Cover

- Creating workers with `new Worker()`
- Sending messages with `postMessage()`
- Receiving messages with `onmessage`
- Terminating workers properly
- Error handling in workers
- Inline workers with Blob URLs

### Prerequisites

- Understanding of Web Worker fundamentals
- Familiarity with event handling in JavaScript

---

## Creating a Worker

### Basic Worker Creation

To create a dedicated worker, use the `Worker` constructor with a URL to the worker script:

```javascript
// main.js
const worker = new Worker('worker.js');
```

The browser will:
1. Fetch the script at `worker.js`
2. Create a new thread
3. Execute the script in that thread

### Worker Script Location

Worker scripts must be served from the **same origin** as your page:

```javascript
// ✅ Same origin - works
const worker = new Worker('/scripts/worker.js');
const worker2 = new Worker('./worker.js');
const worker3 = new Worker('worker.js');

// ❌ Cross-origin - blocked by browser security
const worker4 = new Worker('https://other-domain.com/worker.js');
```

### Worker Options

The `Worker` constructor accepts an options object:

```javascript
const worker = new Worker('worker.js', {
  type: 'module',           // 'classic' (default) or 'module'
  credentials: 'same-origin', // 'omit', 'same-origin', or 'include'
  name: 'DataProcessor'     // Name for debugging
});
```

**Module Workers** (ES modules in workers):

```javascript
// main.js
const worker = new Worker('worker.js', { type: 'module' });

// worker.js (can use import/export)
import { processData } from './utils.js';

self.onmessage = (event) => {
  const result = processData(event.data);
  self.postMessage(result);
};
```

> **Note:** Module workers are supported in Chrome 80+, Firefox 114+, and Safari 15+.

---

## Sending Data with postMessage

The `postMessage()` method sends data from one thread to another:

```javascript
// Main thread → Worker
worker.postMessage('Hello, Worker!');

// Worker → Main thread (inside worker.js)
self.postMessage('Hello, Main Thread!');
```

### What Can Be Sent?

The data is serialized using the **structured clone algorithm**, which supports:

```javascript
// ✅ Primitives
worker.postMessage('string');
worker.postMessage(42);
worker.postMessage(true);
worker.postMessage(null);
worker.postMessage(undefined);

// ✅ Objects and arrays
worker.postMessage({ name: 'John', age: 30 });
worker.postMessage([1, 2, 3, 4, 5]);

// ✅ Nested structures
worker.postMessage({
  users: [
    { id: 1, name: 'Alice' },
    { id: 2, name: 'Bob' }
  ],
  metadata: { total: 2 }
});

// ✅ Typed arrays
worker.postMessage(new Uint8Array([1, 2, 3]));
worker.postMessage(new Float32Array([1.1, 2.2, 3.3]));

// ✅ Date, RegExp, Map, Set
worker.postMessage(new Date());
worker.postMessage(new Map([['key', 'value']]));
worker.postMessage(new Set([1, 2, 3]));

// ✅ Blob, File, ArrayBuffer
worker.postMessage(new Blob(['data']));
worker.postMessage(new ArrayBuffer(1024));
```

### What CANNOT Be Sent?

```javascript
// ❌ Functions
worker.postMessage(() => console.log('hi'));
// Error: Functions cannot be cloned

// ❌ DOM nodes
worker.postMessage(document.body);
// Error: HTMLElement cannot be cloned

// ❌ Symbols
worker.postMessage(Symbol('test'));
// Error: Symbol cannot be cloned

// ❌ Objects with circular references (unless handled)
const obj = { name: 'test' };
obj.self = obj;
worker.postMessage(obj);
// Error: Converting circular structure
```

---

## Receiving Data with onmessage

Listen for messages using the `message` event:

```javascript
// main.js
const worker = new Worker('worker.js');

// Method 1: onmessage property
worker.onmessage = (event) => {
  console.log('Received from worker:', event.data);
};

// Method 2: addEventListener
worker.addEventListener('message', (event) => {
  console.log('Received from worker:', event.data);
});
```

```javascript
// worker.js
self.onmessage = (event) => {
  console.log('Received from main:', event.data);
  
  // Process and respond
  const result = event.data.toUpperCase();
  self.postMessage(result);
};
```

### The MessageEvent Object

```javascript
worker.onmessage = (event) => {
  console.log(event.data);       // The sent data
  console.log(event.origin);     // Origin of the sender
  console.log(event.source);     // Reference to sender (null for workers)
  console.log(event.ports);      // MessagePort array (for channels)
};
```

### Bi-directional Communication Pattern

```javascript
// main.js
const worker = new Worker('calculator.js');

function calculate(operation, a, b) {
  return new Promise((resolve) => {
    const id = Date.now(); // Unique request ID
    
    const handler = (event) => {
      if (event.data.id === id) {
        worker.removeEventListener('message', handler);
        resolve(event.data.result);
      }
    };
    
    worker.addEventListener('message', handler);
    worker.postMessage({ id, operation, a, b });
  });
}

// Usage
const sum = await calculate('add', 5, 3);
console.log(sum); // 8
```

```javascript
// calculator.js
self.onmessage = (event) => {
  const { id, operation, a, b } = event.data;
  
  let result;
  switch (operation) {
    case 'add': result = a + b; break;
    case 'subtract': result = a - b; break;
    case 'multiply': result = a * b; break;
    case 'divide': result = a / b; break;
  }
  
  self.postMessage({ id, result });
};
```

---

## Terminating Workers

Workers consume memory and CPU. Always terminate them when no longer needed.

### From the Main Thread

```javascript
const worker = new Worker('worker.js');

// Do some work...

// Terminate when done
worker.terminate();

// After termination:
worker.postMessage('test'); // Silently fails, message not sent
```

### From Inside the Worker

```javascript
// worker.js
self.onmessage = (event) => {
  if (event.data === 'shutdown') {
    // Clean up resources
    console.log('Worker shutting down');
    self.close(); // Terminate self
    return;
  }
  
  // Normal processing...
};
```

### Termination Differences

| Method | Behavior |
|--------|----------|
| `worker.terminate()` | Immediate termination, no cleanup |
| `self.close()` | Allows current task to finish, then terminates |

> **Warning:** `terminate()` is abrupt—the worker doesn't get a chance to clean up. If you need graceful shutdown, use `self.close()` with a message protocol.

---

## Error Handling in Workers

Workers can fail in multiple ways. Handle all of them:

### Syntax/Runtime Errors

```javascript
// main.js
const worker = new Worker('worker.js');

worker.onerror = (event) => {
  console.error('Worker error:', event.message);
  console.error('File:', event.filename);
  console.error('Line:', event.lineno);
  console.error('Column:', event.colno);
  
  event.preventDefault(); // Prevents error from propagating
};
```

### Handling Errors Inside the Worker

```javascript
// worker.js
self.onmessage = async (event) => {
  try {
    const result = await riskyOperation(event.data);
    self.postMessage({ success: true, result });
  } catch (error) {
    self.postMessage({ 
      success: false, 
      error: error.message,
      stack: error.stack
    });
  }
};
```

### Complete Error Handling Pattern

```javascript
// main.js
const worker = new Worker('worker.js');

// Handle worker creation/script errors
worker.onerror = (event) => {
  console.error('Worker crashed:', event.message);
  // Maybe create a new worker?
};

// Handle unhandled rejections in the worker
worker.onmessageerror = (event) => {
  console.error('Message deserialization failed');
};

// Handle application-level errors
worker.onmessage = (event) => {
  if (event.data.error) {
    console.error('Operation failed:', event.data.error);
    return;
  }
  // Process successful result
  console.log('Result:', event.data.result);
};
```

---

## Inline Workers with Blob

Sometimes you want to create a worker without a separate file. Use Blob URLs:

```javascript
// Define worker code as a string
const workerCode = `
  self.onmessage = (event) => {
    const result = event.data * 2;
    self.postMessage(result);
  };
`;

// Create a Blob from the code
const blob = new Blob([workerCode], { type: 'application/javascript' });

// Create a URL for the Blob
const workerUrl = URL.createObjectURL(blob);

// Create the worker
const worker = new Worker(workerUrl);

worker.postMessage(21);
worker.onmessage = (e) => console.log(e.data); // 42

// Clean up when done
worker.terminate();
URL.revokeObjectURL(workerUrl);
```

### Factory Function for Inline Workers

```javascript
function createInlineWorker(fn) {
  // Convert function to string and wrap in worker handler
  const code = `
    const fn = ${fn.toString()};
    self.onmessage = async (event) => {
      try {
        const result = await fn(event.data);
        self.postMessage({ success: true, result });
      } catch (error) {
        self.postMessage({ success: false, error: error.message });
      }
    };
  `;
  
  const blob = new Blob([code], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);
  
  return {
    execute(data) {
      return new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
          if (e.data.success) resolve(e.data.result);
          else reject(new Error(e.data.error));
        };
        worker.postMessage(data);
      });
    },
    terminate() {
      worker.terminate();
      URL.revokeObjectURL(url);
    }
  };
}

// Usage
const doubler = createInlineWorker((n) => n * 2);
const result = await doubler.execute(21); // 42
doubler.terminate();
```

> **🤖 AI Context:** Inline workers are useful for AI applications where you want to dynamically create workers based on the type of processing needed—e.g., different workers for text processing, embedding generation, or result formatting.

---

## Message Channels

For more complex communication patterns, use `MessageChannel`:

```javascript
// main.js
const worker = new Worker('worker.js');
const channel = new MessageChannel();

// Send one port to the worker
worker.postMessage({ type: 'init' }, [channel.port2]);

// Use the other port for communication
channel.port1.onmessage = (event) => {
  console.log('Response:', event.data);
};

channel.port1.postMessage('Hello via channel!');
```

```javascript
// worker.js
let port;

self.onmessage = (event) => {
  if (event.data.type === 'init') {
    port = event.ports[0];
    port.onmessage = (e) => {
      port.postMessage('Received: ' + e.data);
    };
  }
};
```

---

## Hands-on Exercise

### Your Task

Create a worker pool that manages multiple workers for parallel processing.

### Requirements

1. Create a `WorkerPool` class that manages N workers
2. Implement `execute(data)` that assigns work to an available worker
3. Handle the case when all workers are busy (queue the work)
4. Return results as Promises

### Expected Result

```javascript
const pool = new WorkerPool('task-worker.js', 4);

// These run in parallel across 4 workers
const results = await Promise.all([
  pool.execute(task1),
  pool.execute(task2),
  pool.execute(task3),
  pool.execute(task4),
  pool.execute(task5), // Queued, runs when worker available
]);
```

<details>
<summary>💡 Hints</summary>

- Track which workers are busy with a Set or Map
- Use a queue array for pending tasks
- Process the queue when a worker becomes available
- Use unique IDs to match requests with responses

</details>

<details>
<summary>✅ Solution</summary>

```javascript
class WorkerPool {
  constructor(workerScript, poolSize = navigator.hardwareConcurrency || 4) {
    this.workers = [];
    this.available = [];
    this.queue = [];
    this.taskId = 0;
    this.pendingTasks = new Map();
    
    // Create workers
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerScript);
      worker.onmessage = (event) => this.handleMessage(worker, event);
      worker.onerror = (event) => this.handleError(worker, event);
      this.workers.push(worker);
      this.available.push(worker);
    }
  }
  
  execute(data) {
    return new Promise((resolve, reject) => {
      const taskId = ++this.taskId;
      
      this.pendingTasks.set(taskId, { resolve, reject });
      
      if (this.available.length > 0) {
        this.dispatch(taskId, data);
      } else {
        this.queue.push({ taskId, data });
      }
    });
  }
  
  dispatch(taskId, data) {
    const worker = this.available.pop();
    worker.currentTaskId = taskId;
    worker.postMessage({ taskId, data });
  }
  
  handleMessage(worker, event) {
    const { taskId, result, error } = event.data;
    const task = this.pendingTasks.get(taskId);
    
    if (task) {
      if (error) {
        task.reject(new Error(error));
      } else {
        task.resolve(result);
      }
      this.pendingTasks.delete(taskId);
    }
    
    // Worker is now available
    this.available.push(worker);
    
    // Process queued tasks
    if (this.queue.length > 0) {
      const { taskId, data } = this.queue.shift();
      this.dispatch(taskId, data);
    }
  }
  
  handleError(worker, event) {
    const task = this.pendingTasks.get(worker.currentTaskId);
    if (task) {
      task.reject(new Error(event.message));
      this.pendingTasks.delete(worker.currentTaskId);
    }
    this.available.push(worker);
  }
  
  terminate() {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.available = [];
    this.pendingTasks.clear();
    this.queue = [];
  }
}
```

**task-worker.js:**
```javascript
self.onmessage = (event) => {
  const { taskId, data } = event.data;
  
  try {
    // Simulate work
    const result = data * 2;
    self.postMessage({ taskId, result });
  } catch (error) {
    self.postMessage({ taskId, error: error.message });
  }
};
```

</details>

---

## Summary

✅ Create workers with `new Worker(url)` or `new Worker(url, { type: 'module' })`
✅ Send data with `postMessage()` - most data types are supported
✅ Receive with `onmessage` event handler
✅ Terminate with `worker.terminate()` or `self.close()`
✅ Handle errors with `onerror` event
✅ Use Blob URLs for inline workers

**Next:** [Transferable Objects](./03-transferable-objects.md)

---

## Further Reading

- [MDN Worker()](https://developer.mozilla.org/en-US/docs/Web/API/Worker/Worker) - Constructor reference
- [MDN postMessage()](https://developer.mozilla.org/en-US/docs/Web/API/Worker/postMessage) - Message API
- [Structured Clone Algorithm](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm) - Data serialization

<!-- 
Sources Consulted:
- MDN Web Workers API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
- MDN Worker: https://developer.mozilla.org/en-US/docs/Web/API/Worker
- MDN postMessage: https://developer.mozilla.org/en-US/docs/Web/API/Worker/postMessage
-->
