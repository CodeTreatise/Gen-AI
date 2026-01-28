---
title: "Destructuring and Spread"
---

# Destructuring and Spread

## Introduction

Destructuring and spread operators are modern JavaScript features that make working with arrays and objects more concise and expressive. Instead of accessing properties with dot notation or indices, you can extract multiple values in a single statement. When building AI applications, these patterns simplify API response handling, function parameters, and data transformations.

These features reduce boilerplate code, make intent clearer, and enable elegant patterns like rest parameters, default values, and object composition.

### What We'll Cover
- Array destructuring with defaults and rest elements
- Object destructuring with renaming and nested patterns
- Spread operator for arrays and objects
- Practical patterns for API data and function parameters
- Combining destructuring with other features

### Prerequisites
- Arrays and objects
- Function parameters
- Understanding of references vs. copies

---

## Array Destructuring

Extract array elements into variables:

```javascript
const colors = ["red", "green", "blue"];

const [first, second, third] = colors;
console.log(first);   // "red"
console.log(second);  // "green"
console.log(third);   // "blue"
```

**Output:**
```
red
green
blue
```

Skip elements:

```javascript
const numbers = [1, 2, 3, 4, 5];

const [first, , third, , fifth] = numbers;
console.log(first, third, fifth);  // 1 3 5
```

**Output:**
```
1 3 5
```

### Default Values

```javascript
const [a = 1, b = 2, c = 3] = [10, 20];
console.log(a, b, c);  // 10 20 3 (c uses default)
```

**Output:**
```
10 20 3
```

### Rest Elements

Collect remaining elements:

```javascript
const numbers = [1, 2, 3, 4, 5];

const [first, second, ...rest] = numbers;
console.log(first);   // 1
console.log(second);  // 2
console.log(rest);    // [3, 4, 5]
```

**Output:**
```
1
2
[ 3, 4, 5 ]
```

### Swapping Variables

```javascript
let a = 1;
let b = 2;

[a, b] = [b, a];
console.log(a, b);  // 2 1
```

**Output:**
```
2 1
```

---

## Object Destructuring

Extract object properties:

```javascript
const user = {
  name: "Alice",
  age: 30,
  email: "alice@example.com"
};

const { name, age, email } = user;
console.log(name);   // "Alice"
console.log(age);    // 30
console.log(email);  // "alice@example.com"
```

**Output:**
```
Alice
30
alice@example.com
```

### Renaming Variables

```javascript
const user = { name: "Bob", age: 25 };

const { name: userName, age: userAge } = user;
console.log(userName);  // "Bob"
console.log(userAge);   // 25
// console.log(name);   // ReferenceError: name is not defined
```

**Output:**
```
Bob
25
```

### Default Values

```javascript
const user = { name: "Charlie" };

const { name, age = 18, role = "guest" } = user;
console.log(name);  // "Charlie"
console.log(age);   // 18 (default)
console.log(role);  // "guest" (default)
```

**Output:**
```
Charlie
18
guest
```

### Nested Destructuring

```javascript
const response = {
  data: {
    user: {
      name: "Alice",
      profile: {
        bio: "AI enthusiast"
      }
    }
  }
};

const { data: { user: { name, profile: { bio } } } } = response;
console.log(name);  // "Alice"
console.log(bio);   // "AI enthusiast"
```

**Output:**
```
Alice
AI enthusiast
```

### Rest in Objects

```javascript
const user = {
  name: "David",
  age: 28,
  email: "david@example.com",
  role: "admin"
};

const { name, ...otherInfo } = user;
console.log(name);       // "David"
console.log(otherInfo);  // { age: 28, email: "david@example.com", role: "admin" }
```

**Output:**
```
David
{ age: 28, email: 'david@example.com', role: 'admin' }
```

---

## Function Parameters

Destructure directly in function signatures:

### Array Parameters

```javascript
function getCoordinates([x, y]) {
  return { x, y };
}

console.log(getCoordinates([10, 20]));  // { x: 10, y: 20 }
```

**Output:**
```
{ x: 10, y: 20 }
```

### Object Parameters

```javascript
function greet({ name, age = 18 }) {
  return `Hello ${name}, you are ${age} years old`;
}

console.log(greet({ name: "Alice", age: 30 }));  // "Hello Alice, you are 30 years old"
console.log(greet({ name: "Bob" }));             // "Hello Bob, you are 18 years old"
```

**Output:**
```
Hello Alice, you are 30 years old
Hello Bob, you are 18 years old
```

API configuration example:

```javascript
function callApi({ 
  endpoint, 
  method = "GET", 
  headers = {}, 
  body = null 
}) {
  console.log(`${method} ${endpoint}`);
  console.log("Headers:", headers);
  console.log("Body:", body);
}

callApi({
  endpoint: "/api/users",
  method: "POST",
  body: { name: "Alice" }
});
```

**Output:**
```
POST /api/users
Headers: {}
Body: { name: 'Alice' }
```

---

## Spread Operator

### Array Spread

Copy arrays:

```javascript
const original = [1, 2, 3];
const copy = [...original];

copy.push(4);
console.log(original);  // [1, 2, 3] - unchanged
console.log(copy);      // [1, 2, 3, 4]
```

**Output:**
```
[ 1, 2, 3 ]
[ 1, 2, 3, 4 ]
```

Combine arrays:

```javascript
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];

console.log(combined);  // [1, 2, 3, 4, 5, 6]
```

**Output:**
```
[ 1, 2, 3, 4, 5, 6 ]
```

Insert elements:

```javascript
const numbers = [1, 2, 5];
const inserted = [1, 2, 3, 4, 5];

console.log([...numbers.slice(0, 2), 3, 4, ...numbers.slice(2)]);  // [1, 2, 3, 4, 5]
```

**Output:**
```
[ 1, 2, 3, 4, 5 ]
```

### Object Spread

Copy objects:

```javascript
const original = { name: "Alice", age: 30 };
const copy = { ...original };

copy.age = 31;
console.log(original.age);  // 30 - unchanged
console.log(copy.age);      // 31
```

**Output:**
```
30
31
```

Merge objects:

```javascript
const defaults = { theme: "light", notifications: true };
const userPrefs = { theme: "dark", language: "en" };

const config = { ...defaults, ...userPrefs };
console.log(config);
// { theme: "dark", notifications: true, language: "en" }
```

**Output:**
```
{ theme: 'dark', notifications: true, language: 'en' }
```

> **Note:** Properties from later objects overwrite earlier ones. Order matters!

### Shallow vs Deep Copy

Spread creates **shallow copies**:

```javascript
const original = {
  name: "Alice",
  settings: { theme: "light" }
};

const copy = { ...original };
copy.settings.theme = "dark";

console.log(original.settings.theme);  // "dark" - nested object shared!
```

**Output:**
```
dark
```

For deep copies, use `structuredClone()`:

```javascript
const original = {
  name: "Alice",
  settings: { theme: "light" }
};

const deepCopy = structuredClone(original);
deepCopy.settings.theme = "dark";

console.log(original.settings.theme);  // "light" - unchanged
```

**Output:**
```
light
```

---

## Practical Patterns

### API Response Handling

```javascript
const apiResponse = {
  status: 200,
  data: {
    user: {
      id: 1,
      name: "Alice",
      email: "alice@example.com"
    },
    metadata: {
      timestamp: 1704632400000
    }
  }
};

const {
  status,
  data: {
    user: { name, email },
    metadata: { timestamp }
  }
} = apiResponse;

console.log(`Status: ${status}`);
console.log(`User: ${name} (${email})`);
console.log(`Time: ${new Date(timestamp)}`);
```

**Output:**
```
Status: 200
User: Alice (alice@example.com)
Time: Tue Jan 07 2025 12:00:00 GMT+0000 (Coordinated Universal Time)
```

### Function Return Values

```javascript
function getStats(numbers) {
  return {
    min: Math.min(...numbers),
    max: Math.max(...numbers),
    sum: numbers.reduce((a, b) => a + b, 0),
    avg: numbers.reduce((a, b) => a + b, 0) / numbers.length
  };
}

const { min, max, avg } = getStats([1, 2, 3, 4, 5]);
console.log(`Range: ${min}-${max}, Average: ${avg}`);
```

**Output:**
```
Range: 1-5, Average: 3
```

### Updating Nested State

```javascript
const state = {
  user: {
    name: "Alice",
    settings: {
      theme: "light",
      notifications: true
    }
  }
};

// Update theme without mutating original
const newState = {
  ...state,
  user: {
    ...state.user,
    settings: {
      ...state.user.settings,
      theme: "dark"
    }
  }
};

console.log(state.user.settings.theme);     // "light" - unchanged
console.log(newState.user.settings.theme);  // "dark"
```

**Output:**
```
light
dark
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use destructuring in function parameters | Clearer intent, self-documenting code |
| Provide default values in destructuring | Handles missing properties gracefully |
| Use rest operator to collect remaining props | Enables flexible function signatures |
| Use spread for immutable updates | Avoids mutating original objects/arrays |
| Be mindful of shallow copies | Nested objects require deep cloning |
| Don't over-nest destructuring | Deep nesting reduces readability |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Destructuring `null` or `undefined` | Check existence or provide defaults: `const { a } = obj || {}` |
| Assuming spread creates deep copies | Use `structuredClone()` for nested structures |
| Forgetting parentheses with object destructuring in assignment | `({ a } = obj)` not `{ a } = obj` |
| Over-destructuring making code less readable | Balance convenience with clarity |
| Using spread on large arrays/objects | Can be slow; consider alternatives for performance-critical code |
| Confusing rest (`...rest`) with spread | Rest collects, spread expands |

---

## Hands-on Exercise

### Your Task
Create a message processing system using destructuring and spread operators. Handle API responses, transform data, and manage conversation state immutably.

### Requirements
1. Create `processApiResponse(response)` that destructures nested API data
2. Create `addMessage(conversation, message)` that returns updated conversation without mutation
3. Create `updateUserSettings(user, newSettings)` that merges settings immutably
4. Use destructuring in all function parameters
5. Use spread operators for all updates

### Expected Result
```javascript
const response = {
  data: {
    conversation: {
      id: "conv1",
      messages: [{ role: "user", content: "Hello" }]
    }
  }
};

const { data: { conversation: { id, messages } } } = response;
console.log(id);  // "conv1"
console.log(messages.length);  // 1
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use nested destructuring for deep API structures
- Use spread to copy arrays/objects before modifying
- Provide default values in destructuring
- Return new objects/arrays, never mutate originals
- Use rest operator to handle unknown properties
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
function processApiResponse(response) {
  const {
    status = 200,
    data: {
      conversation: {
        id,
        title = "Untitled",
        messages = [],
        participants = []
      } = {}
    } = {}
  } = response;
  
  return {
    id,
    title,
    messageCount: messages.length,
    participantCount: participants.length,
    firstMessage: messages[0]?.content || "No messages"
  };
}

function addMessage(conversation, { role, content }) {
  const newMessage = {
    role,
    content,
    timestamp: Date.now(),
    id: `msg-${Date.now()}`
  };
  
  return {
    ...conversation,
    messages: [...conversation.messages, newMessage]
  };
}

function updateUserSettings(user, newSettings) {
  return {
    ...user,
    settings: {
      ...user.settings,
      ...newSettings
    }
  };
}

function mergecreateConversationSummary({ id, messages, ...rest }) {
  const userMessages = messages.filter(m => m.role === "user");
  const assistantMessages = messages.filter(m => m.role === "assistant");
  
  return {
    id,
    stats: {
      total: messages.length,
      user: userMessages.length,
      assistant: assistantMessages.length
    },
    metadata: rest
  };
}

// Test
const apiResponse = {
  status: 200,
  data: {
    conversation: {
      id: "conv1",
      title: "AI Chat",
      messages: [
        { role: "user", content: "Hello" }
      ],
      participants: ["user1", "ai"]
    }
  }
};

console.log("Processed:", processApiResponse(apiResponse));

let conversation = { id: "conv1", messages: [] };
conversation = addMessage(conversation, { role: "user", content: "Hello" });
conversation = addMessage(conversation, { role: "assistant", content: "Hi!" });
console.log("Messages:", conversation.messages.length);

let user = { name: "Alice", settings: { theme: "light", lang: "en" } };
user = updateUserSettings(user, { theme: "dark" });
console.log("Theme:", user.settings.theme);
console.log("Lang:", user.settings.lang);  // Preserved

console.log("Summary:", createConversationSummary(conversation));
```
</details>

### Bonus Challenges
- [ ] Implement `removeMessage(conversation, messageId)` using filter and spread
- [ ] Create `combineConversations(...conversations)` that merges multiple convs
- [ ] Add `extractUserInfo({ name, age, ...details })` that separates concerns
- [ ] Implement nested update helper: `updateNested(obj, path, value)`

---

## Summary

✅ Destructuring extracts values from arrays/objects into variables
✅ Use defaults and rest elements for flexible destructuring
✅ Spread operator copies and merges arrays/objects (shallow copy)
✅ Destructure function parameters for self-documenting APIs
✅ Combine destructuring and spread for immutable state updates

[Previous: Array Methods](./07-arrays-methods.md) | [Next: Modules](./09-modules.md)

---

<!-- 
Sources Consulted:
- MDN Destructuring assignment: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment
- MDN Spread syntax: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax
- MDN Rest parameters: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/rest_parameters
-->
