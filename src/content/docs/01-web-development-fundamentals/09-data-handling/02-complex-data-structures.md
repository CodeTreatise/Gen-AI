---
title: "Working with Complex Data Structures"
---

# Working with Complex Data Structures

## Introduction

Real-world APIs return deeply nested data. User profiles contain addresses containing cities. Orders contain items containing products containing categories. Learning to navigate, transform, and manage complex data structures is essential for building maintainable applications.

This lesson covers techniques for working with nested data, deep cloning, immutable updates, and using JavaScript's Map and Set collections.

### What We'll Cover

- Nested objects and arrays
- Deep cloning techniques
- Immutable update patterns
- Normalization for flat structures
- Maps and Sets in JavaScript

### Prerequisites

- JavaScript objects and arrays
- Spread operator basics
- Understanding of references vs values

---

## Nested Objects and Arrays

### Accessing Nested Data

```javascript
const user = {
  name: 'Alice',
  profile: {
    bio: 'Developer',
    social: {
      twitter: '@alice',
      github: 'alice-dev'
    }
  },
  posts: [
    { id: 1, title: 'First Post' },
    { id: 2, title: 'Second Post' }
  ]
};

// Dot notation
console.log(user.profile.social.twitter);  // '@alice'

// Bracket notation (for dynamic keys)
const platform = 'github';
console.log(user.profile.social[platform]);  // 'alice-dev'

// Array access
console.log(user.posts[0].title);  // 'First Post'
```

### Safe Access with Optional Chaining

Avoid errors when accessing properties that might not exist:

```javascript
const user = { name: 'Bob' };

// ❌ Throws error
// user.profile.social.twitter  // TypeError: Cannot read property 'social' of undefined

// ✅ Returns undefined safely
user.profile?.social?.twitter  // undefined

// With nullish coalescing for defaults
user.profile?.bio ?? 'No bio available'  // 'No bio available'
```

### Iterating Nested Structures

```javascript
const data = {
  users: [
    { name: 'Alice', tags: ['dev', 'ai'] },
    { name: 'Bob', tags: ['design', 'ux'] }
  ]
};

// Get all unique tags
const allTags = data.users.flatMap(user => user.tags);
console.log(allTags);  // ['dev', 'ai', 'design', 'ux']

// Find user with specific tag
const devUsers = data.users.filter(user => user.tags.includes('dev'));
console.log(devUsers);  // [{ name: 'Alice', tags: ['dev', 'ai'] }]
```

---

## Deep Cloning Techniques

Objects and arrays are **reference types**. Copying a reference doesn't copy the data:

```javascript
const original = { name: 'Alice', hobbies: ['reading'] };
const shallowCopy = { ...original };

shallowCopy.hobbies.push('coding');
console.log(original.hobbies);  // ['reading', 'coding'] - MODIFIED!
```

### Shallow Copy Methods

```javascript
// Spread operator (objects)
const copy1 = { ...original };

// Object.assign
const copy2 = Object.assign({}, original);

// Spread operator (arrays)
const arrCopy = [...originalArray];

// Array.from
const arrCopy2 = Array.from(originalArray);
```

> **Warning:** Shallow copies only copy the first level. Nested objects/arrays are still references!

### Deep Clone with structuredClone()

The modern way to deep clone (available in all browsers since 2022):

```javascript
const original = {
  name: 'Alice',
  profile: {
    settings: {
      theme: 'dark'
    }
  },
  hobbies: ['reading', 'coding']
};

const deep = structuredClone(original);

// Modify the clone
deep.profile.settings.theme = 'light';
deep.hobbies.push('gaming');

// Original is unchanged
console.log(original.profile.settings.theme);  // 'dark'
console.log(original.hobbies);  // ['reading', 'coding']
```

### structuredClone Limitations

```javascript
// ❌ Cannot clone functions
structuredClone({ fn: () => {} });  // DataCloneError

// ❌ Cannot clone DOM elements
structuredClone({ el: document.body });  // DataCloneError

// ❌ Cannot clone class instances (loses prototype)
class User { greet() { return 'Hello'; } }
const user = new User();
const clone = structuredClone(user);
clone.greet();  // TypeError: clone.greet is not a function

// ✅ Can clone Dates, Maps, Sets, RegExp, ArrayBuffers
const obj = {
  date: new Date(),
  map: new Map([['a', 1]]),
  set: new Set([1, 2, 3])
};
const clone = structuredClone(obj);  // Works!
```

### JSON Round-Trip (Legacy)

Before `structuredClone`, developers used JSON:

```javascript
const deep = JSON.parse(JSON.stringify(original));
```

**Limitations:**
- Loses `undefined`, functions, symbols
- Converts Dates to strings
- Cannot handle circular references
- Slower than `structuredClone`

### Custom Deep Clone

For special cases:

```javascript
function deepClone(obj, seen = new WeakMap()) {
  // Handle primitives and null
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  // Handle circular references
  if (seen.has(obj)) {
    return seen.get(obj);
  }
  
  // Handle Date
  if (obj instanceof Date) {
    return new Date(obj.getTime());
  }
  
  // Handle Array
  if (Array.isArray(obj)) {
    const arrCopy = [];
    seen.set(obj, arrCopy);
    obj.forEach((item, index) => {
      arrCopy[index] = deepClone(item, seen);
    });
    return arrCopy;
  }
  
  // Handle Object
  const objCopy = {};
  seen.set(obj, objCopy);
  Object.keys(obj).forEach(key => {
    objCopy[key] = deepClone(obj[key], seen);
  });
  return objCopy;
}
```

---

## Immutable Update Patterns

Immutability means never modifying existing data—always create new objects. This is crucial for:
- React state management
- Redux/Zustand stores
- Undo/redo functionality
- Predictable data flow

### Updating Nested Objects

```javascript
const state = {
  user: {
    name: 'Alice',
    profile: {
      theme: 'dark',
      notifications: true
    }
  }
};

// ❌ Mutation
state.user.profile.theme = 'light';

// ✅ Immutable update (spread at each level)
const newState = {
  ...state,
  user: {
    ...state.user,
    profile: {
      ...state.user.profile,
      theme: 'light'
    }
  }
};
```

### Updating Arrays Immutably

```javascript
const items = [{ id: 1, name: 'A' }, { id: 2, name: 'B' }];

// Add item
const added = [...items, { id: 3, name: 'C' }];

// Remove item
const removed = items.filter(item => item.id !== 2);

// Update item
const updated = items.map(item =>
  item.id === 1 ? { ...item, name: 'Updated' } : item
);

// Insert at position
const inserted = [
  ...items.slice(0, 1),
  { id: 4, name: 'Inserted' },
  ...items.slice(1)
];
```

### Using Immer for Complex Updates

Immer lets you write "mutating" code that produces immutable updates:

```javascript
import { produce } from 'immer';

const state = {
  users: [
    { id: 1, name: 'Alice', posts: [] }
  ]
};

// Write mutations, get immutable result
const newState = produce(state, draft => {
  const user = draft.users.find(u => u.id === 1);
  user.name = 'Alicia';
  user.posts.push({ id: 1, title: 'New Post' });
});

// Original unchanged
console.log(state.users[0].name);  // 'Alice'
console.log(newState.users[0].name);  // 'Alicia'
```

---

## Normalization for Flat Structures

Deeply nested data is hard to update. **Normalization** flattens data into lookup tables.

### The Problem with Nested Data

```javascript
// Nested API response
const posts = [
  {
    id: 1,
    title: 'Hello',
    author: { id: 100, name: 'Alice' },
    comments: [
      { id: 1001, text: 'Great!', user: { id: 101, name: 'Bob' } }
    ]
  }
];

// To update Alice's name, you need to find every occurrence
// What if Alice is author of 100 posts? Update all of them?
```

### Normalized Structure

```javascript
const normalizedState = {
  posts: {
    byId: {
      1: { id: 1, title: 'Hello', authorId: 100, commentIds: [1001] }
    },
    allIds: [1]
  },
  users: {
    byId: {
      100: { id: 100, name: 'Alice' },
      101: { id: 101, name: 'Bob' }
    },
    allIds: [100, 101]
  },
  comments: {
    byId: {
      1001: { id: 1001, text: 'Great!', userId: 101 }
    },
    allIds: [1001]
  }
};

// Now updating Alice is simple
normalizedState.users.byId[100].name = 'Alicia';
```

### Normalizing Function

```javascript
function normalizeResponse(posts) {
  const normalized = {
    posts: { byId: {}, allIds: [] },
    users: { byId: {}, allIds: [] },
    comments: { byId: {}, allIds: [] }
  };
  
  posts.forEach(post => {
    // Normalize author
    if (!normalized.users.byId[post.author.id]) {
      normalized.users.byId[post.author.id] = post.author;
      normalized.users.allIds.push(post.author.id);
    }
    
    // Normalize comments
    const commentIds = post.comments.map(comment => {
      if (!normalized.users.byId[comment.user.id]) {
        normalized.users.byId[comment.user.id] = comment.user;
        normalized.users.allIds.push(comment.user.id);
      }
      
      normalized.comments.byId[comment.id] = {
        id: comment.id,
        text: comment.text,
        userId: comment.user.id
      };
      normalized.comments.allIds.push(comment.id);
      
      return comment.id;
    });
    
    // Normalize post
    normalized.posts.byId[post.id] = {
      id: post.id,
      title: post.title,
      authorId: post.author.id,
      commentIds
    };
    normalized.posts.allIds.push(post.id);
  });
  
  return normalized;
}
```

### Using normalizr Library

```javascript
import { normalize, schema } from 'normalizr';

const user = new schema.Entity('users');
const comment = new schema.Entity('comments', { user });
const post = new schema.Entity('posts', {
  author: user,
  comments: [comment]
});

const result = normalize(apiResponse, [post]);
// { entities: { users: {...}, posts: {...}, comments: {...} }, result: [1, 2, 3] }
```

---

## Maps and Sets

JavaScript's Map and Set provide better alternatives to objects and arrays for certain use cases.

### Map vs Object

| Feature | Object | Map |
|---------|--------|-----|
| Keys | Strings/Symbols only | Any value (objects, functions) |
| Order | Not guaranteed | Insertion order preserved |
| Size | Manual counting | `.size` property |
| Iteration | `Object.keys()` | Direct `.forEach()`, `.entries()` |
| Performance | Good for small data | Better for frequent add/delete |

### Map Usage

```javascript
const userCache = new Map();

// Set values (any type as key)
const user1 = { id: 1 };
userCache.set(user1, { name: 'Alice', loaded: true });
userCache.set('admin', { name: 'Admin User' });
userCache.set(42, { name: 'Answer' });

// Get values
console.log(userCache.get(user1));  // { name: 'Alice', loaded: true }
console.log(userCache.get('admin'));  // { name: 'Admin User' }

// Check existence
console.log(userCache.has(user1));  // true

// Size
console.log(userCache.size);  // 3

// Delete
userCache.delete(42);

// Iterate
userCache.forEach((value, key) => {
  console.log(key, value);
});

// Convert to array
const entries = [...userCache.entries()];
const keys = [...userCache.keys()];
const values = [...userCache.values()];

// Clear all
userCache.clear();
```

### Set Usage

Sets store unique values only:

```javascript
const uniqueTags = new Set();

uniqueTags.add('javascript');
uniqueTags.add('python');
uniqueTags.add('javascript');  // Duplicate ignored

console.log(uniqueTags.size);  // 2
console.log([...uniqueTags]);  // ['javascript', 'python']

// Common operations
uniqueTags.has('python');  // true
uniqueTags.delete('python');
uniqueTags.clear();

// Remove duplicates from array
const arr = [1, 2, 2, 3, 3, 3];
const unique = [...new Set(arr)];  // [1, 2, 3]
```

### WeakMap and WeakSet

Weak versions allow garbage collection of keys:

```javascript
const cache = new WeakMap();

function process(obj) {
  if (cache.has(obj)) {
    return cache.get(obj);
  }
  
  const result = expensiveOperation(obj);
  cache.set(obj, result);
  return result;
}

let data = { id: 1 };
process(data);

data = null;  // Object can be garbage collected
// WeakMap entry is also removed automatically
```

---

## Hands-on Exercise

### Your Task

Build a function that deeply merges two objects, combining nested properties.

### Requirements

1. Merge nested objects recursively
2. Arrays should be concatenated
3. Later values override earlier ones
4. Return a new object (no mutation)

```javascript
const obj1 = {
  name: 'Alice',
  settings: { theme: 'dark', fontSize: 14 },
  tags: ['dev']
};

const obj2 = {
  age: 30,
  settings: { fontSize: 16, lang: 'en' },
  tags: ['ai']
};

deepMerge(obj1, obj2);
// {
//   name: 'Alice',
//   age: 30,
//   settings: { theme: 'dark', fontSize: 16, lang: 'en' },
//   tags: ['dev', 'ai']
// }
```

<details>
<summary>✅ Solution</summary>

```javascript
function deepMerge(target, source) {
  const result = { ...target };
  
  for (const key of Object.keys(source)) {
    const targetVal = target[key];
    const sourceVal = source[key];
    
    if (Array.isArray(targetVal) && Array.isArray(sourceVal)) {
      // Concatenate arrays
      result[key] = [...targetVal, ...sourceVal];
    } else if (
      targetVal && sourceVal &&
      typeof targetVal === 'object' &&
      typeof sourceVal === 'object' &&
      !Array.isArray(targetVal) && !Array.isArray(sourceVal)
    ) {
      // Recursively merge objects
      result[key] = deepMerge(targetVal, sourceVal);
    } else {
      // Override with source value
      result[key] = sourceVal;
    }
  }
  
  return result;
}
```

</details>

---

## Summary

✅ Use **optional chaining** (`?.`) for safe nested access
✅ **structuredClone()** for deep cloning (modern browsers)
✅ Immutable updates: spread at every level or use **Immer**
✅ **Normalize** nested API data for easier updates
✅ Use **Map** for non-string keys and better iteration
✅ Use **Set** for unique values

**Next:** [Data Transformation and Normalization](./03-data-transformation.md)

---

## Further Reading

- [MDN structuredClone()](https://developer.mozilla.org/en-US/docs/Web/API/structuredClone)
- [MDN Map](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map)
- [MDN Set](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set)
- [Immer](https://immerjs.github.io/immer/) - Immutable updates made easy

<!-- 
Sources Consulted:
- MDN structuredClone: https://developer.mozilla.org/en-US/docs/Web/API/structuredClone
- MDN Map: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map
-->
