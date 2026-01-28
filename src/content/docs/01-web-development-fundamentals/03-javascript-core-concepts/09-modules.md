---
title: "JavaScript Modules"
---

# JavaScript Modules

## Introduction

Modules organize code into reusable, encapsulated units with explicit dependencies. Instead of putting all code in global scope or using script tags in specific orders, ES6 modules provide `import` and `export` statements that make dependencies clear and enable better tooling, tree-shaking, and code splitting.

When building AI applications with multiple files—API clients, prompt templates, utility functions, and UI components—modules prevent naming conflicts, improve maintainability, and enable efficient bundling for production.

### What We'll Cover
- ES6 module syntax: `import` and `export`
- Named vs. default exports
- Dynamic imports for code splitting
- Module loading and execution order
- CommonJS vs ES Modules
- Browser support and tooling

### Prerequisites
- Functions and classes
- File system basics
- Understanding of script loading in browsers

---

## Named Exports

Export multiple values from a module:

**`math.js`**
```javascript
export function add(a, b) {
  return a + b;
}

export function subtract(a, b) {
  return a - b;
}

export const PI = 3.14159;
```

**`main.js`**
```javascript
import { add, subtract, PI } from './math.js';

console.log(add(5, 3));      // 8
console.log(subtract(10, 4));  // 6
console.log(PI);             // 3.14159
```

**Output:**
```
8
6
3.14159
```

### Exporting After Declaration

```javascript
function multiply(a, b) {
  return a * b;
}

function divide(a, b) {
  return a / b;
}

export { multiply, divide };
```

### Renaming Exports

```javascript
function internalAdd(a, b) {
  return a + b;
}

export { internalAdd as add };
```

---

## Default Exports

One main export per module:

**`logger.js`**
```javascript
export default class Logger {
  log(message) {
    console.log(`[LOG] ${message}`);
  }
  
  error(message) {
    console.error(`[ERROR] ${message}`);
  }
}
```

**`main.js`**
```javascript
import Logger from './logger.js';

const logger = new Logger();
logger.log("Application started");  // [LOG] Application started
```

**Output:**
```
[LOG] Application started
```

You can name the default import anything:

```javascript
import MyLogger from './logger.js';  // Same as above
import Log from './logger.js';       // Also valid
```

### Mixing Named and Default Exports

**`api.js`**
```javascript
export default class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  
  fetch(endpoint) {
    return `Fetching ${this.baseUrl}/${endpoint}`;
  }
}

export const API_VERSION = "v1";
export const DEFAULT_TIMEOUT = 5000;
```

**`main.js`**
```javascript
import ApiClient, { API_VERSION, DEFAULT_TIMEOUT } from './api.js';

const client = new ApiClient("https://api.example.com");
console.log(client.fetch("users"));     // Fetching https://api.example.com/users
console.log(API_VERSION);               // "v1"
```

**Output:**
```
Fetching https://api.example.com/users
v1
```

---

## Import Patterns

### Renaming Imports

```javascript
import { add as sum, subtract as diff } from './math.js';

console.log(sum(5, 3));   // 8
console.log(diff(10, 4));  // 6
```

**Output:**
```
8
6
```

### Namespace Imports

Import everything as an object:

```javascript
import * as MathUtils from './math.js';

console.log(MathUtils.add(5, 3));  // 8
console.log(MathUtils.PI);         // 3.14159
```

**Output:**
```
8
3.14159
```

### Side-Effect Imports

Run module code without importing values:

**`polyfills.js`**
```javascript
// Add Array.prototype.last if not exists
if (!Array.prototype.last) {
  Array.prototype.last = function() {
    return this[this.length - 1];
  };
}
```

**`main.js`**
```javascript
import './polyfills.js';  // Just run the code

console.log([1, 2, 3].last());  // 3
```

**Output:**
```
3
```

---

## Dynamic Imports

Load modules on-demand (asynchronously):

```javascript
button.addEventListener('click', async () => {
  const module = await import('./heavy-module.js');
  module.initialize();
});
```

With destructuring:

```javascript
async function loadChatModule() {
  const { ChatClient, formatMessage } = await import('./chat.js');
  
  const client = new ChatClient();
  console.log(formatMessage("Hello"));
}

loadChatModule();
```

Conditional loading:

```javascript
async function loadAnalytics() {
  if (user.hasConsentedToAnalytics) {
    const analytics = await import('./analytics.js');
    analytics.default.track('page_view');
  }
}
```

Error handling:

```javascript
try {
  const module = await import('./optional-feature.js');
  module.activate();
} catch (error) {
  console.log("Feature not available:", error.message);
}
```

---

## Module Loading and Execution

Modules execute once, when first imported:

**`counter.js`**
```javascript
let count = 0;

export function increment() {
  count++;
}

export function getCount() {
  return count;
}

console.log("Counter module loaded");
```

**`module-a.js`**
```javascript
import { increment } from './counter.js';  // "Counter module loaded" prints
increment();
```

**`module-b.js`**
```javascript
import { getCount } from './counter.js';  // Doesn't print again
console.log(getCount());  // 1 (sees increment from module-a)
```

Modules are **singletons**—all imports share the same instance.

---

## Browser Usage

### HTML with type="module"

```html
<!DOCTYPE html>
<html>
<head>
  <title>ES Modules Example</title>
</head>
<body>
  <script type="module">
    import { greet } from './utils.js';
    greet("World");
  </script>
  
  <!-- Or external module script -->
  <script type="module" src="./main.js"></script>
</body>
</html>
```

### Import Maps

Map bare specifiers to URLs:

```html
<script type="importmap">
{
  "imports": {
    "lodash": "https://cdn.skypack.dev/lodash-es",
    "@/utils": "/src/utils.js"
  }
}
</script>

<script type="module">
  import _ from 'lodash';  // Resolves to CDN
  import { format } from '@/utils';  // Resolves to /src/utils.js
</script>
```

---

## ES Modules vs. CommonJS

| Feature | ES Modules | CommonJS |
|---------|------------|----------|
| Syntax | `import`/`export` | `require()`/`module.exports` |
| Environment | Browser & Node.js | Node.js only |
| Loading | Static (compile-time) | Dynamic (runtime) |
| Tree-shaking | Yes | No |
| Top-level await | Yes | No |
| File extension | `.mjs` or `.js` (with `"type": "module"`) | `.cjs` or `.js` (default) |

### CommonJS Example

**`math.cjs`**
```javascript
function add(a, b) {
  return a + b;
}

module.exports = { add };
```

**`main.cjs`**
```javascript
const { add } = require('./math.cjs');
console.log(add(5, 3));  // 8
```

### Interop in Node.js

Import CommonJS from ES Modules:

```javascript
// ES Module can import CommonJS
import pkg from './legacy.cjs';  // Default import for module.exports
```

Cannot directly `require()` ES Modules—use dynamic import:

```javascript
// CommonJS importing ES Module
(async () => {
  const module = await import('./es-module.js');
})();
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use named exports for utilities | Enables tree-shaking and clearer imports |
| Use default export for main class/component | Conventional for single-purpose modules |
| Keep modules focused (single responsibility) | Easier to test, reuse, and understand |
| Use dynamic imports for code splitting | Reduces initial bundle size |
| Avoid circular dependencies | Can cause initialization order issues |
| Use `.js` extension in imports | Required in browsers; good for consistency |
| Prefer ES Modules over CommonJS | Better tooling, browser support, and features |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Omitting file extension in browser imports | Always use `.js`: `import './utils.js'` |
| Mixing default and named in confusing ways | Be consistent; document module API |
| Creating circular dependencies | Refactor to shared module or lazy loading |
| Using `require()` in ES Modules | Use `import` or dynamic `import()` |
| Forgetting `type="module"` in HTML | Script won't parse as module |
| Assuming synchronous dynamic imports | `import()` returns a Promise—use `await` |

---

## Hands-on Exercise

### Your Task
Create a modular AI chat application with separate modules for API client, message formatting, and conversation management. Use both named and default exports, and implement dynamic loading.

### Requirements
1. Create `api-client.js` with default export `ApiClient` class
2. Create `formatters.js` with named exports for message formatting functions
3. Create `conversation.js` that imports from both modules
4. Implement dynamic import for an optional "analytics" module
5. Use proper module patterns (exports, imports, namespaces)

### Expected Structure
```
project/
├── api-client.js (default export)
├── formatters.js (named exports)
├── conversation.js (imports others)
├── analytics.js (dynamically loaded)
└── main.js (entry point)
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `export default class` for ApiClient
- Export multiple functions from formatters
- Use namespace import (`import * as`) for formatters
- Use dynamic `await import()` for analytics
- Create a main module that ties everything together
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**`api-client.js`**
```javascript
export default class ApiClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }
  
  async sendMessage(message) {
    console.log(`[API] POST ${this.baseUrl}/chat`);
    console.log(`[API] Message: ${message}`);
    
    // Simulate API call
    return {
      role: "assistant",
      content: `Response to: ${message}`,
      timestamp: Date.now()
    };
  }
}

export const API_VERSION = "v1";
```

**`formatters.js`**
```javascript
export function formatTimestamp(timestamp) {
  return new Date(timestamp).toLocaleTimeString();
}

export function formatMessage(message) {
  const time = formatTimestamp(message.timestamp);
  return `[${time}] ${message.role}: ${message.content}`;
}

export function formatConversation(messages) {
  return messages.map(formatMessage).join('\n');
}
```

**`conversation.js`**
```javascript
import ApiClient, { API_VERSION } from './api-client.js';
import * as Formatters from './formatters.js';

export class Conversation {
  #messages = [];
  #apiClient;
  
  constructor(baseUrl, apiKey) {
    this.#apiClient = new ApiClient(baseUrl, apiKey);
    console.log(`Conversation using API ${API_VERSION}`);
  }
  
  async sendMessage(content) {
    // Add user message
    const userMessage = {
      role: "user",
      content,
      timestamp: Date.now()
    };
    this.#messages.push(userMessage);
    
    // Get AI response
    const response = await this.#apiClient.sendMessage(content);
    this.#messages.push(response);
    
    return response;
  }
  
  getFormattedHistory() {
    return Formatters.formatConversation(this.#messages);
  }
  
  getMessages() {
    return [...this.#messages];
  }
}
```

**`analytics.js`**
```javascript
export function track(event, data) {
  console.log(`[Analytics] ${event}:`, data);
}

export function trackConversation(messageCount) {
  track('conversation_length', { messages: messageCount });
}
```

**`main.js`**
```javascript
import { Conversation } from './conversation.js';

async function main() {
  const conv = new Conversation('https://api.example.com', 'sk-test');
  
  await conv.sendMessage("Hello");
  await conv.sendMessage("How are you?");
  
  console.log("\n=== Conversation History ===");
  console.log(conv.getFormattedHistory());
  
  // Dynamically load analytics if user consents
  const userConsent = true;
  if (userConsent) {
    const analytics = await import('./analytics.js');
    analytics.trackConversation(conv.getMessages().length);
  }
}

main().catch(console.error);
```

**Usage:**
```bash
# Node.js (with ES Modules support)
node main.js

# Or in browser with type="module"
<script type="module" src="main.js"></script>
```
</details>

### Bonus Challenges
- [ ] Add a `plugins` system with dynamic module loading
- [ ] Implement hot module replacement for development
- [ ] Create a module that re-exports from multiple sub-modules
- [ ] Add import maps configuration for aliasing

---

## Summary

✅ Use `import`/`export` for ES Modules—static, tree-shakeable, browser-compatible
✅ Named exports for utilities, default export for main class/component
✅ Dynamic `import()` enables code splitting and lazy loading
✅ Modules execute once and are singletons—all imports share same instance
✅ Prefer ES Modules over CommonJS for modern JavaScript development

[Previous: Destructuring and Spread](./08-destructuring-spread.md) | [Next: Web Components](./10-web-components.md)

---

<!-- 
Sources Consulted:
- MDN import statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import
- MDN export statement: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/export
- MDN JavaScript Modules Guide: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules
- MDN Dynamic imports: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/import
-->
