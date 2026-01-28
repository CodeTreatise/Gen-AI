---
title: "Data Transformation and Normalization"
---

# Data Transformation and Normalization

## Introduction

Raw API data rarely matches what your UI needs. You'll reshape nested responses, flatten hierarchies, validate incoming data, and transform formats. These transformations are the glue between your backend and frontend.

This lesson covers common data transformation patterns and validation strategies used in production applications.

### What We'll Cover

- Reshaping API responses
- Flattening nested data
- Denormalization for display
- Data validation patterns
- Schema validation concepts

### Prerequisites

- Array methods (map, filter, reduce)
- Object manipulation
- Understanding of API responses

---

## Reshaping API Responses

APIs return data in their own format. Your components need it differently.

### Basic Transformation

```javascript
// API returns this
const apiResponse = {
  user_id: 123,
  first_name: 'Alice',
  last_name: 'Smith',
  email_address: 'alice@example.com',
  created_at: '2025-01-24T10:00:00Z'
};

// Your component needs this
function transformUser(data) {
  return {
    id: data.user_id,
    name: `${data.first_name} ${data.last_name}`,
    email: data.email_address,
    createdAt: new Date(data.created_at),
    initials: `${data.first_name[0]}${data.last_name[0]}`
  };
}

const user = transformUser(apiResponse);
console.log(user);
// {
//   id: 123,
//   name: 'Alice Smith',
//   email: 'alice@example.com',
//   createdAt: Date object,
//   initials: 'AS'
// }
```

### Case Conversion

Convert between snake_case (common in Python APIs) and camelCase (JavaScript convention):

```javascript
// snake_case to camelCase
function snakeToCamel(str) {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

// Deep transform object keys
function transformKeys(obj, transformer) {
  if (Array.isArray(obj)) {
    return obj.map(item => transformKeys(item, transformer));
  }
  
  if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        transformer(key),
        transformKeys(value, transformer)
      ])
    );
  }
  
  return obj;
}

// Usage
const snakeData = {
  user_name: 'alice',
  user_profile: {
    avatar_url: 'https://...',
    created_at: '2025-01-24'
  }
};

const camelData = transformKeys(snakeData, snakeToCamel);
// {
//   userName: 'alice',
//   userProfile: {
//     avatarUrl: 'https://...',
//     createdAt: '2025-01-24'
//   }
// }
```

### Selective Mapping

Pick only the fields you need:

```javascript
function pick(obj, keys) {
  return keys.reduce((result, key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
    return result;
  }, {});
}

function omit(obj, keys) {
  return Object.fromEntries(
    Object.entries(obj).filter(([key]) => !keys.includes(key))
  );
}

// Usage
const user = { id: 1, name: 'Alice', password: 'secret', role: 'admin' };

pick(user, ['id', 'name']);  // { id: 1, name: 'Alice' }
omit(user, ['password']);     // { id: 1, name: 'Alice', role: 'admin' }
```

---

## Flattening Nested Data

Deep nesting makes data hard to access and update. Flatten for simpler handling.

### Flatten Object

```javascript
function flattenObject(obj, prefix = '') {
  return Object.entries(obj).reduce((acc, [key, value]) => {
    const newKey = prefix ? `${prefix}.${key}` : key;
    
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(acc, flattenObject(value, newKey));
    } else {
      acc[newKey] = value;
    }
    
    return acc;
  }, {});
}

const nested = {
  user: {
    name: 'Alice',
    address: {
      city: 'NYC',
      zip: '10001'
    }
  },
  active: true
};

const flat = flattenObject(nested);
// {
//   'user.name': 'Alice',
//   'user.address.city': 'NYC',
//   'user.address.zip': '10001',
//   'active': true
// }
```

### Unflatten Object

```javascript
function unflattenObject(obj) {
  const result = {};
  
  for (const [key, value] of Object.entries(obj)) {
    const keys = key.split('.');
    let current = result;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!(keys[i] in current)) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
  }
  
  return result;
}

const restored = unflattenObject(flat);
// Back to original nested structure
```

### Flatten for Forms

Common pattern: flatten nested data for form inputs, unflatten on submit:

```javascript
// API data → Form state
const formData = flattenObject(userData);

// Form inputs use dot notation keys
<input name="user.address.city" value={formData['user.address.city']} />

// On submit → API format
const submitData = unflattenObject(formData);
```

---

## Denormalization for Display

After normalizing data for storage, you often need to denormalize it for display.

### Assembling Related Data

```javascript
const state = {
  posts: {
    byId: {
      1: { id: 1, title: 'Hello', authorId: 100, tagIds: [1, 2] }
    }
  },
  users: {
    byId: {
      100: { id: 100, name: 'Alice', avatarUrl: '/alice.jpg' }
    }
  },
  tags: {
    byId: {
      1: { id: 1, name: 'javascript' },
      2: { id: 2, name: 'tutorial' }
    }
  }
};

// Denormalize for display
function getPostWithRelations(state, postId) {
  const post = state.posts.byId[postId];
  if (!post) return null;
  
  return {
    ...post,
    author: state.users.byId[post.authorId],
    tags: post.tagIds.map(id => state.tags.byId[id])
  };
}

const displayPost = getPostWithRelations(state, 1);
// {
//   id: 1,
//   title: 'Hello',
//   authorId: 100,
//   tagIds: [1, 2],
//   author: { id: 100, name: 'Alice', avatarUrl: '/alice.jpg' },
//   tags: [
//     { id: 1, name: 'javascript' },
//     { id: 2, name: 'tutorial' }
//   ]
// }
```

### Memoized Selectors

Avoid recomputing denormalized data on every render:

```javascript
// Simple memoization
function memoize(fn) {
  const cache = new Map();
  
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

const getPostWithRelationsMemoized = memoize(getPostWithRelations);
```

In React/Redux, use libraries like `reselect`:

```javascript
import { createSelector } from 'reselect';

const selectPosts = state => state.posts.byId;
const selectUsers = state => state.users.byId;
const selectPostId = (state, postId) => postId;

const selectPostWithAuthor = createSelector(
  [selectPosts, selectUsers, selectPostId],
  (posts, users, postId) => {
    const post = posts[postId];
    return post ? { ...post, author: users[post.authorId] } : null;
  }
);
```

---

## Data Validation Patterns

Never trust external data. Validate before using.

### Type Checking

```javascript
function validateUser(data) {
  const errors = [];
  
  if (typeof data !== 'object' || data === null) {
    return { valid: false, errors: ['Data must be an object'] };
  }
  
  if (typeof data.name !== 'string' || data.name.trim() === '') {
    errors.push('Name is required and must be a string');
  }
  
  if (typeof data.email !== 'string' || !data.email.includes('@')) {
    errors.push('Valid email is required');
  }
  
  if (data.age !== undefined && (typeof data.age !== 'number' || data.age < 0)) {
    errors.push('Age must be a positive number');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

// Usage
const result = validateUser({ name: '', email: 'invalid' });
// { valid: false, errors: ['Name is required...', 'Valid email...'] }
```

### Validation with Default Values

```javascript
function parseUserInput(data) {
  return {
    name: typeof data.name === 'string' ? data.name.trim() : '',
    email: typeof data.email === 'string' ? data.email.toLowerCase() : '',
    age: typeof data.age === 'number' && data.age > 0 ? data.age : null,
    role: ['admin', 'user', 'guest'].includes(data.role) ? data.role : 'guest',
    createdAt: data.createdAt instanceof Date ? data.createdAt : new Date()
  };
}
```

### Assertion Functions

```javascript
function assertString(value, fieldName) {
  if (typeof value !== 'string') {
    throw new TypeError(`${fieldName} must be a string, got ${typeof value}`);
  }
  return value;
}

function assertNumber(value, fieldName) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new TypeError(`${fieldName} must be a number`);
  }
  return value;
}

function assertArray(value, fieldName) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${fieldName} must be an array`);
  }
  return value;
}

// Usage
function processOrder(data) {
  const name = assertString(data.name, 'name');
  const quantity = assertNumber(data.quantity, 'quantity');
  const items = assertArray(data.items, 'items');
  
  // Safe to use now...
}
```

---

## Schema Validation

For complex validation, use schema validators.

### Zod (Popular TypeScript-First)

```javascript
import { z } from 'zod';

const UserSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  email: z.string().email('Invalid email'),
  age: z.number().positive().optional(),
  role: z.enum(['admin', 'user', 'guest']).default('user'),
  tags: z.array(z.string()).default([])
});

// Parse and validate
const result = UserSchema.safeParse({
  name: 'Alice',
  email: 'alice@example.com'
});

if (result.success) {
  console.log(result.data);
  // { name: 'Alice', email: 'alice@example.com', role: 'user', tags: [] }
} else {
  console.log(result.error.issues);
}

// Parse or throw
try {
  const user = UserSchema.parse(data);
} catch (error) {
  // ZodError with detailed issues
}
```

### Yup (Popular in React)

```javascript
import * as yup from 'yup';

const userSchema = yup.object({
  name: yup.string().required('Name is required'),
  email: yup.string().email('Invalid email').required(),
  age: yup.number().positive().integer(),
  website: yup.string().url()
});

// Validate
try {
  const validUser = await userSchema.validate(data);
} catch (error) {
  console.log(error.message);  // First validation error
  console.log(error.errors);   // All errors
}

// Check without throwing
const isValid = await userSchema.isValid(data);
```

### API Response Validation

```javascript
const ApiResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    users: z.array(UserSchema),
    pagination: z.object({
      page: z.number(),
      total: z.number(),
      hasMore: z.boolean()
    })
  }),
  error: z.string().optional()
});

async function fetchUsers() {
  const response = await fetch('/api/users');
  const json = await response.json();
  
  // Validate response shape
  const result = ApiResponseSchema.safeParse(json);
  
  if (!result.success) {
    console.error('Invalid API response:', result.error);
    throw new Error('API returned unexpected data format');
  }
  
  return result.data;
}
```

---

## Transform Pipelines

Chain transformations for readable data processing:

```javascript
// Utility for chaining
function pipe(...fns) {
  return (input) => fns.reduce((acc, fn) => fn(acc), input);
}

// Individual transformers
const parseResponse = (data) => data.results;
const filterActive = (users) => users.filter(u => u.active);
const sortByName = (users) => [...users].sort((a, b) => a.name.localeCompare(b.name));
const addDisplayName = (users) => users.map(u => ({
  ...u,
  displayName: `${u.firstName} ${u.lastName}`
}));
const toMap = (users) => new Map(users.map(u => [u.id, u]));

// Create pipeline
const processUsers = pipe(
  parseResponse,
  filterActive,
  sortByName,
  addDisplayName,
  toMap
);

// Use it
const userMap = processUsers(apiResponse);
```

### Async Pipeline

```javascript
async function asyncPipe(...fns) {
  return async (input) => {
    let result = input;
    for (const fn of fns) {
      result = await fn(result);
    }
    return result;
  };
}

const fetchAndProcess = asyncPipe(
  async (url) => fetch(url),
  async (response) => response.json(),
  (data) => data.items,
  (items) => items.filter(i => i.active)
);

const items = await fetchAndProcess('/api/items');
```

---

## Hands-on Exercise

### Your Task

Build a data transformer for an e-commerce API response.

### Input (API Response)

```javascript
const apiResponse = {
  order_id: 'ORD-123',
  customer: {
    customer_id: 'C-456',
    first_name: 'John',
    last_name: 'Doe',
    contact_email: 'john@example.com'
  },
  line_items: [
    { sku: 'PROD-1', product_name: 'Widget', qty: 2, unit_price: 9.99 },
    { sku: 'PROD-2', product_name: 'Gadget', qty: 1, unit_price: 19.99 }
  ],
  order_date: '2025-01-24T10:00:00Z'
};
```

### Output (Your Format)

```javascript
{
  id: 'ORD-123',
  customer: {
    id: 'C-456',
    name: 'John Doe',
    email: 'john@example.com'
  },
  items: [
    { sku: 'PROD-1', name: 'Widget', quantity: 2, price: 9.99, total: 19.98 },
    { sku: 'PROD-2', name: 'Gadget', quantity: 1, price: 19.99, total: 19.99 }
  ],
  orderDate: Date,
  subtotal: 39.97,
  itemCount: 3
}
```

<details>
<summary>✅ Solution</summary>

```javascript
function transformOrder(data) {
  const items = data.line_items.map(item => ({
    sku: item.sku,
    name: item.product_name,
    quantity: item.qty,
    price: item.unit_price,
    total: Math.round(item.qty * item.unit_price * 100) / 100
  }));
  
  return {
    id: data.order_id,
    customer: {
      id: data.customer.customer_id,
      name: `${data.customer.first_name} ${data.customer.last_name}`,
      email: data.customer.contact_email
    },
    items,
    orderDate: new Date(data.order_date),
    subtotal: items.reduce((sum, i) => sum + i.total, 0),
    itemCount: items.reduce((sum, i) => sum + i.quantity, 0)
  };
}
```

</details>

---

## Summary

✅ **Reshape** API responses to match your component needs
✅ Convert between **snake_case** and **camelCase**
✅ **Flatten** nested objects for forms and simpler access
✅ **Denormalize** stored data for display with selectors
✅ Always **validate** external data before using
✅ Use **schema libraries** (Zod, Yup) for complex validation
✅ Create **transform pipelines** for readable data processing

**Next:** [LocalStorage and SessionStorage](./04-localstorage-sessionstorage.md)

---

## Further Reading

- [Zod Documentation](https://zod.dev/) - TypeScript-first schema validation
- [Yup Documentation](https://github.com/jquense/yup) - Schema validation
- [Reselect](https://github.com/reduxjs/reselect) - Memoized selectors for Redux

<!-- 
Sources Consulted:
- Zod documentation: https://zod.dev/
- MDN Array methods: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array
-->
