---
title: "Data Structures"
---

# Data Structures

## Introduction

Beyond basic arrays and objects, JavaScript provides specialized data structures for specific use cases. `Map` and `Set` offer better performance and cleaner APIs than plain objects for certain scenarios, while `WeakMap` and `WeakSet` enable memory-efficient caching patterns. When building AI applications, you'll use these structures to manage conversation state, cache API responses, track unique entities, and organize complex data efficiently.

Understanding when to use each data structure—and their performance characteristics—helps you write scalable applications that handle large datasets gracefully.

### What We'll Cover
- `Set`: collections of unique values
- `Map`: key-value pairs with any type of key
- `WeakMap` and `WeakSet`: memory-efficient weak references
- `structuredClone()`: deep copying complex objects
- Performance characteristics and use cases

### Prerequisites
- Objects and arrays
- Basic understanding of references vs. values
- For comprehension (iterating collections)

---

## Set

A collection of **unique values** of any type:

```javascript
const numbers = new Set([1, 2, 3, 2, 1]);
console.log(numbers);  // Set(3) { 1, 2, 3 } - duplicates removed

numbers.add(4);
numbers.add(2);  // No effect - already exists
console.log(numbers.size);  // 4
```

**Output:**
```
Set(3) { 1, 2, 3 }
4
```

### Set Operations

```javascript
const tags = new Set();

// Add elements
tags.add("javascript");
tags.add("python");
tags.add("typescript");

// Check existence
console.log(tags.has("javascript"));  // true
console.log(tags.has("ruby"));        // false

// Delete elements
tags.delete("python");
console.log(tags.size);  // 2

// Iterate
for (const tag of tags) {
  console.log(tag);
}

// Convert to array
const tagsArray = [...tags];
console.log(tagsArray);  // ["javascript", "typescript"]
```

**Output:**
```
true
false
2
javascript
typescript
[ 'javascript', 'typescript' ]
```

### Practical Use: Remove Array Duplicates

```javascript
const numbers = [1, 2, 2, 3, 4, 4, 5];
const unique = [...new Set(numbers)];
console.log(unique);  // [1, 2, 3, 4, 5]
```

**Output:**
```
[ 1, 2, 3, 4, 5 ]
```

### Practical Use: Track Unique Users

```javascript
class ConversationTracker {
  #activeUsers = new Set();
  
  userJoined(userId) {
    this.#activeUsers.add(userId);
    console.log(`User ${userId} joined. Active: ${this.#activeUsers.size}`);
  }
  
  userLeft(userId) {
    this.#activeUsers.delete(userId);
    console.log(`User ${userId} left. Active: ${this.#activeUsers.size}`);
  }
  
  isUserActive(userId) {
    return this.#activeUsers.has(userId);
  }
}

const tracker = new Conversation Tracker();
tracker.userJoined("user1");
tracker.userJoined("user2");
tracker.userJoined("user1");  // No duplicate
tracker.userLeft("user1");
```

**Output:**
```
User user1 joined. Active: 1
User user2 joined. Active: 2
User user1 joined. Active: 2
User user1 left. Active: 1
```

---

## Map

Key-value pairs where **keys can be any type** (not just strings):

```javascript
const userRoles = new Map();

userRoles.set("alice", "admin");
userRoles.set("bob", "user");
userRoles.set("charlie", "moderator");

console.log(userRoles.get("alice"));  // "admin"
console.log(userRoles.has("bob"));    // true
console.log(userRoles.size);          // 3

userRoles.delete("charlie");
console.log(userRoles.size);          // 2
```

**Output:**
```
admin
true
3
2
```

### Non-String Keys

Objects as keys:

```javascript
const metadata = new Map();

const user1 = { id: 1, name: "Alice" };
const user2 = { id: 2, name: "Bob" };

metadata.set(user1, { lastSeen: "2025-01-07", messageCount: 42 });
metadata.set(user2, { lastSeen: "2025-01-06", messageCount: 15 });

console.log(metadata.get(user1));  // { lastSeen: "2025-01-07", messageCount: 42 }
```

**Output:**
```
{ lastSeen: '2025-01-07', messageCount: 42 }
```

### Iteration

```javascript
const config = new Map([
  ["host", "localhost"],
  ["port", 3000],
  ["debug", true]
]);

// Iterate entries
for (const [key, value] of config) {
  console.log(`${key}: ${value}`);
}

// Keys only
for (const key of config.keys()) {
  console.log(key);
}

// Values only
for (const value of config.values()) {
  console.log(value);
}

// forEach
config.forEach((value, key) => {
  console.log(`${key} = ${value}`);
});
```

**Output:**
```
host: localhost
port: 3000
debug: true
host
port
debug
localhost
3000
true
host = localhost
port = 3000
debug = true
```

### Map vs Object

| Feature | Map | Object |
|---------|-----|--------|
| Key types | Any type | Strings/Symbols only |
| Size | `.size` property | Manual `Object.keys().length` |
| Iteration order | Insertion order guaranteed | Not guaranteed (pre-ES2015) |
| Performance | Better for frequent add/remove | Faster for small, static data |
| Prototype | No inherited keys | Has prototype chain |

### Practical Use: API Response Cache

```javascript
class ApiCache {
  #cache = new Map();
  #maxSize = 100;
  
  get(url) {
    return this.#cache.get(url);
  }
  
  set(url, response) {
    // Implement LRU: remove oldest if at capacity
    if (this.#cache.size >= this.#maxSize) {
      const firstKey = this.#cache.keys().next().value;
      this.#cache.delete(firstKey);
    }
    
    this.#cache.set(url, {
      data: response,
      timestamp: Date.now()
    });
  }
  
  has(url) {
    const entry = this.#cache.get(url);
    if (!entry) return false;
    
    // Check if expired (5 minutes)
    const isExpired = Date.now() - entry.timestamp > 5 * 60 * 1000;
    if (isExpired) {
      this.#cache.delete(url);
      return false;
    }
    
    return true;
  }
}

const cache = new ApiCache();
cache.set("/api/users", { users: ["Alice", "Bob"] });
console.log(cache.has("/api/users"));  // true
console.log(cache.get("/api/users"));
```

**Output:**
```
true
{ data: { users: [ 'Alice', 'Bob' ] }, timestamp: 1704632400000 }
```

---

## WeakMap

Like `Map`, but with **weak references** to keys—keys must be objects and can be garbage collected:

```javascript
let user = { name: "Alice" };
const metadata = new WeakMap();

metadata.set(user, { loginCount: 5, lastLogin: Date.now() });
console.log(metadata.has(user));  // true

user = null;  // Object can be garbage collected
// metadata entry for user is automatically removed
```

**Output:**
```
true
```

### Use Case: Private Data

```javascript
const privateData = new WeakMap();

class User {
  constructor(name, password) {
    this.name = name;
    // Store sensitive data in WeakMap
    privateData.set(this, { password });
  }
  
  checkPassword(input) {
    const data = privateData.get(this);
    return data.password === input;
  }
}

const user = new User("Alice", "secret123");
console.log(user.checkPassword("wrong"));    // false
console.log(user.checkPassword("secret123"));  // true
console.log(user.password);                  // undefined (not accessible)
```

**Output:**
```
false
true
undefined
```

---

## WeakSet

Like `Set`, but with weak references—values must be objects:

```javascript
const clickedElements = new WeakSet();

function handleClick(element) {
  if (clickedElements.has(element)) {
    console.log("Already clicked");
    return;
  }
  
  clickedElements.add(element);
  console.log("First click!");
}

const button1 = { id: "btn1" };
const button2 = { id: "btn2" };

handleClick(button1);  // "First click!"
handleClick(button1);  // "Already clicked"
handleClick(button2);  // "First click!"
```

**Output:**
```
First click!
Already clicked
First click!
```

> **Note:** `WeakMap` and `WeakSet` are not iterable—you can't get all keys/values. This is because entries can be garbage collected at any time.

---

## structuredClone()

Deep clone objects including nested structures, Dates, RegExp, etc.:

```javascript
const original = {
  name: "Alice",
  metadata: {
    tags: ["javascript", "ai"],
    created: new Date("2025-01-01")
  },
  settings: new Map([["theme", "dark"]])
};

const clone = structuredClone(original);

// Modify clone
clone.metadata.tags.push("typescript");
clone.metadata.created.setFullYear(2026);

console.log(original.metadata.tags);  // ["javascript", "ai"] - unchanged
console.log(original.metadata.created.getFullYear());  // 2025 - unchanged
console.log(clone.metadata.tags);     // ["javascript", "ai", "typescript"]
console.log(clone.metadata.created.getFullYear());  // 2026
```

**Output:**
```
[ 'javascript', 'ai' ]
2025
[ 'javascript', 'ai', 'typescript' ]
2026
```

### Limitations of structuredClone

```javascript
// ❌ Cannot clone functions
const withFunction = { fn: () => {} };
// structuredClone(withFunction);  // DataCloneError

// ❌ Cannot clone Symbols
const withSymbol = { [Symbol("key")]: "value" };
// structuredClone(withSymbol);  // Symbols are lost

// ✅ Can clone: Objects, Arrays, Dates, RegExp, Map, Set, TypedArrays, etc.
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `Set` for unique collections | More explicit than checking arrays with `.includes()` |
| Use `Map` when keys aren't strings | Objects coerce keys to strings; Map preserves types |
| Use `WeakMap`/`WeakSet` for object metadata | Prevents memory leaks—entries auto-removed when objects are GC'd |
| Use `structuredClone()` for deep copies | Native, handles complex types (Date, Map, Set, etc.) |
| Consider performance: `Map` vs `Object` | Map is faster for frequent add/delete; Objects for static config |
| Use `Map` for large datasets | Better performance characteristics than objects for 100s+ keys |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using objects as Map keys without understanding references | Two `{}` are different references; use same object reference |
| Trying to iterate `WeakMap`/`WeakSet` | Not possible—use regular `Map`/`Set` if iteration needed |
| Assuming `structuredClone()` clones functions | It doesn't—manually copy functions or use custom deep clone |
| Using `Set` when order matters and values repeat | Use Array instead |
| Forgetting `Map`/`Set` are mutable | Freeze with custom wrapper if immutability needed |
| Using `WeakMap` with primitive keys | Keys must be objects; primitives aren't garbage collected |

---

## Hands-on Exercise

### Your Task
Create a conversation history manager using `Map`, `Set`, and `structuredClone`. The manager should track conversations, participants, and provide efficient lookup/filtering.

### Requirements
1. Create a `ConversationManager` class
2. Use `Map` to store conversations by ID
3. Use `Set` to track unique participants across all conversations
4. Implement methods:
   - `createConversation(id, participants)`: Create new conversation
   - `addMessage(conversationId, message)`: Add message to conversation
   - `getConversation(id)`: Return deep clone of conversation
   - `getAllParticipants()`: Return Set of all unique participants
   - `getConversationsByParticipant(participant)`: Filter conversations

### Expected Result
```javascript
const manager = new ConversationManager();
manager.createConversation("conv1", ["Alice", "Bob"]);
manager.addMessage("conv1", { from: "Alice", text: "Hello" });

const conv = manager.getConversation("conv1");
console.log(conv.messages.length);  // 1

const participants = manager.getAllParticipants();
console.log(participants.size);  // 2
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Map` with conversation ID as key, conversation object as value
- Use `Set` to accumulate unique participants
- Use `structuredClone()` in `getConversation()` to return deep copy
- Iterate Map entries with `for (const [id, conv] of this.conversations)`
- Add participants to Set when creating conversations
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class ConversationManager {
  #conversations = new Map();
  #allParticipants = new Set();
  
  createConversation(id, participants) {
    if (this.#conversations.has(id)) {
      throw new Error(`Conversation ${id} already exists`);
    }
    
    const conversation = {
      id,
      participants: [...participants],
      messages: [],
      created: new Date()
    };
    
    this.#conversations.set(id, conversation);
    
    // Track all unique participants
    participants.forEach(p => this.#allParticipants.add(p));
    
    return conversation;
  }
  
  addMessage(conversationId, message) {
    const conv = this.#conversations.get(conversationId);
    if (!conv) {
      throw new Error(`Conversation ${conversationId} not found`);
    }
    
    conv.messages.push({
      ...message,
      timestamp: Date.now()
    });
  }
  
  getConversation(id) {
    const conv = this.#conversations.get(id);
    if (!conv) return null;
    
    // Return deep clone to prevent external mutation
    return structuredClone(conv);
  }
  
  getAllParticipants() {
    return new Set(this.#allParticipants);  // Return copy
  }
  
  getConversationsByParticipant(participant) {
    const result = [];
    
    for (const [id, conv] of this.#conversations) {
      if (conv.participants.includes(participant)) {
        result.push(structuredClone(conv));
      }
    }
    
    return result;
  }
  
  getConversationCount() {
    return this.#conversations.size;
  }
}

// Test
const manager = new ConversationManager();

manager.createConversation("conv1", ["Alice", "Bob"]);
manager.createConversation("conv2", ["Alice", "Charlie"]);

manager.addMessage("conv1", { from: "Alice", text: "Hello Bob" });
manager.addMessage("conv1", { from: "Bob", text: "Hi Alice" });
manager.addMessage("conv2", { from: "Charlie", text: "Hey Alice" });

console.log("Total conversations:", manager.getConversationCount());
console.log("All participants:", [...manager.getAllParticipants()]);

const aliceConvs = manager.getConversationsByParticipant("Alice");
console.log("Alice's conversations:", aliceConvs.length);

const conv1 = manager.getConversation("conv1");
console.log("Conv1 messages:", conv1.messages.length);
```
</details>

### Bonus Challenges
- [ ] Add `WeakMap` to cache processed conversation data
- [ ] Implement conversation search with cached results
- [ ] Add TTL (time-to-live) for conversations using timestamps
- [ ] Use `WeakSet` to track "read" conversations per user

---

## Summary

✅ `Set` stores unique values; `Map` stores key-value pairs with any key type
✅ `WeakMap`/`WeakSet` use weak references—entries can be garbage collected
✅ Use `Map` over objects when keys aren't strings or when frequent add/delete
✅ `structuredClone()` deep clones objects including Dates, Maps, Sets (but not functions)
✅ Choose the right structure: Set for uniqueness, Map for lookups, WeakMap for metadata

[Previous: Objects and Prototypes](./05-objects-prototypes.md) | [Next: Array Methods](./07-arrays-methods.md)

---

<!-- 
Sources Consulted:
- MDN Map: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map
- MDN Set: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set
- MDN WeakMap: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WeakMap
- MDN WeakSet: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WeakSet
- MDN structuredClone: https://developer.mozilla.org/en-US/docs/Web/API/structuredClone
-->
