---
title: "JSON Parsing and Stringification"
---

# JSON Parsing and Stringification

## Introduction

**JSON (JavaScript Object Notation)** is the universal data format for web APIs. Despite its name containing "JavaScript," JSON is language-agnostic and used everywhere—from REST APIs to configuration files to data storage.

This lesson covers the `JSON.parse()` and `JSON.stringify()` methods in depth, including advanced features like revivers, replacers, and handling edge cases.

### What We'll Cover

- JSON.parse() syntax and options
- JSON.stringify() syntax and options
- Handling parse errors
- Reviver function for custom parsing
- Replacer function for custom stringification
- Circular reference handling
- BigInt and Date serialization

### Prerequisites

- JavaScript objects and arrays
- Basic understanding of data types

---

## JSON.parse()

Converts a JSON string into a JavaScript value.

### Basic Usage

```javascript
const jsonString = '{"name": "Alice", "age": 30}';
const obj = JSON.parse(jsonString);

console.log(obj.name);  // "Alice"
console.log(obj.age);   // 30
```

**Output:**
```
Alice
30
```

### Parsing Different Types

JSON supports: objects, arrays, strings, numbers, booleans, and null.

```javascript
JSON.parse('{"key": "value"}');  // { key: "value" }
JSON.parse('[1, 2, 3]');          // [1, 2, 3]
JSON.parse('"hello"');            // "hello"
JSON.parse('42');                 // 42
JSON.parse('true');               // true
JSON.parse('null');               // null
```

### What JSON Does NOT Support

```javascript
// These will throw SyntaxError:
JSON.parse("undefined");     // undefined is not valid JSON
JSON.parse("NaN");           // NaN is not valid JSON
JSON.parse("Infinity");      // Infinity is not valid JSON

// Functions cannot be in JSON
JSON.parse('{"fn": function(){}}');  // SyntaxError

// Single quotes are invalid
JSON.parse("{'name': 'Alice'}");     // SyntaxError (must use double quotes)

// Trailing commas are invalid
JSON.parse('{"a": 1,}');             // SyntaxError
```

---

## Handling Parse Errors

Always wrap `JSON.parse()` in try/catch:

```javascript
function safeJsonParse(jsonString, fallback = null) {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('JSON parse error:', error.message);
    return fallback;
  }
}

// Usage
const data = safeJsonParse('{"valid": true}');      // { valid: true }
const invalid = safeJsonParse('not json', {});      // {} (fallback)
const malformed = safeJsonParse('{broken', []);     // [] (fallback)
```

### Detailed Error Information

```javascript
try {
  JSON.parse('{"name": "Alice", age: 30}');  // Missing quotes around age
} catch (error) {
  console.log(error.name);     // "SyntaxError"
  console.log(error.message);  // "Expected property name or '}' in JSON at position 18"
}
```

---

## The Reviver Function

The second argument to `JSON.parse()` is a **reviver** function that transforms values during parsing.

### Signature

```javascript
JSON.parse(text, reviver);

// reviver(key, value) => transformedValue
```

### Parsing Dates

JSON doesn't have a Date type—dates are strings. Use a reviver to convert them:

```javascript
const jsonWithDate = '{"name": "Meeting", "date": "2025-01-24T10:00:00.000Z"}';

const obj = JSON.parse(jsonWithDate, (key, value) => {
  // Check if value looks like an ISO date string
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
    return new Date(value);
  }
  return value;
});

console.log(obj.date instanceof Date);  // true
console.log(obj.date.getFullYear());    // 2025
```

**Output:**
```
true
2025
```

### Transforming Keys

```javascript
const json = '{"user_name": "alice", "user_age": 30}';

// Convert snake_case to camelCase
const obj = JSON.parse(json, function(key, value) {
  if (key === '') return value;  // Root object
  
  // Transform the key (this affects the parent object)
  const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
  
  // We can't change keys directly, but we can restructure
  if (camelKey !== key && this) {
    this[camelKey] = value;
    return undefined;  // Remove original key
  }
  return value;
});

console.log(obj);  // { userName: "alice", userAge: 30 }
```

### Reviver Traversal Order

The reviver is called bottom-up (children before parents):

```javascript
JSON.parse('{"a": {"b": 1}}', (key, value) => {
  console.log(key, value);
  return value;
});
```

**Output:**
```
b 1
a { b: 1 }
 { a: { b: 1 } }
```

Note: The empty string key represents the root.

---

## JSON.stringify()

Converts a JavaScript value to a JSON string.

### Basic Usage

```javascript
const obj = { name: 'Alice', age: 30 };
const jsonString = JSON.stringify(obj);

console.log(jsonString);  // '{"name":"Alice","age":30}'
```

### Pretty Printing

The third argument controls indentation:

```javascript
const obj = { name: 'Alice', hobbies: ['reading', 'coding'] };

// Compact (default)
JSON.stringify(obj);
// '{"name":"Alice","hobbies":["reading","coding"]}'

// 2-space indentation
JSON.stringify(obj, null, 2);
/*
{
  "name": "Alice",
  "hobbies": [
    "reading",
    "coding"
  ]
}
*/

// Tab indentation
JSON.stringify(obj, null, '\t');

// Custom string (max 10 characters)
JSON.stringify(obj, null, '  →  ');
```

### Values That Get Converted

```javascript
// undefined, functions, symbols are omitted from objects
JSON.stringify({ a: undefined, b: function(){}, c: Symbol() });
// '{}'

// In arrays, they become null
JSON.stringify([undefined, function(){}, Symbol()]);
// '[null,null,null]'

// NaN and Infinity become null
JSON.stringify({ a: NaN, b: Infinity, c: -Infinity });
// '{"a":null,"b":null,"c":null}'

// Dates become ISO strings
JSON.stringify({ date: new Date('2025-01-24') });
// '{"date":"2025-01-24T00:00:00.000Z"}'
```

---

## The Replacer Function

The second argument to `JSON.stringify()` filters or transforms values.

### Signature

```javascript
JSON.stringify(value, replacer, space);

// replacer can be:
// 1. A function: (key, value) => newValue
// 2. An array of allowed keys: ['name', 'age']
```

### Filtering Properties

```javascript
const user = {
  name: 'Alice',
  email: 'alice@example.com',
  password: 'secret123',  // Don't expose this!
  role: 'admin'
};

// Array replacer - whitelist of allowed keys
const safe = JSON.stringify(user, ['name', 'email', 'role']);
console.log(safe);
// '{"name":"Alice","email":"alice@example.com","role":"admin"}'
```

### Function Replacer

```javascript
const data = {
  name: 'Alice',
  password: 'secret',
  apiKey: 'sk-12345',
  profile: {
    bio: 'Developer',
    secretToken: 'abc123'
  }
};

// Redact sensitive fields
const redacted = JSON.stringify(data, (key, value) => {
  const sensitiveKeys = ['password', 'apiKey', 'secretToken'];
  if (sensitiveKeys.includes(key)) {
    return '[REDACTED]';
  }
  return value;
}, 2);

console.log(redacted);
```

**Output:**
```json
{
  "name": "Alice",
  "password": "[REDACTED]",
  "apiKey": "[REDACTED]",
  "profile": {
    "bio": "Developer",
    "secretToken": "[REDACTED]"
  }
}
```

### Custom toJSON Method

Objects can define their own serialization:

```javascript
class User {
  constructor(name, email, password) {
    this.name = name;
    this.email = email;
    this.password = password;
  }
  
  toJSON() {
    // Only expose safe properties
    return {
      name: this.name,
      email: this.email
    };
  }
}

const user = new User('Alice', 'alice@example.com', 'secret');
console.log(JSON.stringify(user));
// '{"name":"Alice","email":"alice@example.com"}'
```

---

## Circular Reference Handling

JSON.stringify throws on circular references:

```javascript
const obj = { name: 'Alice' };
obj.self = obj;  // Circular reference!

JSON.stringify(obj);
// TypeError: Converting circular structure to JSON
```

### Solution: Custom Replacer

```javascript
function stringifyWithCircular(obj) {
  const seen = new WeakSet();
  
  return JSON.stringify(obj, (key, value) => {
    if (typeof value === 'object' && value !== null) {
      if (seen.has(value)) {
        return '[Circular]';
      }
      seen.add(value);
    }
    return value;
  });
}

const obj = { name: 'Alice' };
obj.self = obj;

console.log(stringifyWithCircular(obj));
// '{"name":"Alice","self":"[Circular]"}'
```

### Using a Library

For complex cases, use a library like `flatted` or `circular-json`:

```javascript
import { stringify, parse } from 'flatted';

const obj = { name: 'Alice' };
obj.self = obj;

const json = stringify(obj);  // Works!
const restored = parse(json); // Restores circular refs
```

---

## BigInt Serialization

BigInt is not supported by JSON:

```javascript
JSON.stringify({ big: 123n });
// TypeError: Do not know how to serialize a BigInt
```

### Solution: Custom Serialization

```javascript
// Stringify with BigInt support
function stringifyWithBigInt(obj) {
  return JSON.stringify(obj, (key, value) => {
    if (typeof value === 'bigint') {
      return { __type: 'BigInt', value: value.toString() };
    }
    return value;
  });
}

// Parse with BigInt support
function parseWithBigInt(json) {
  return JSON.parse(json, (key, value) => {
    if (value && value.__type === 'BigInt') {
      return BigInt(value.value);
    }
    return value;
  });
}

// Usage
const data = { count: 9007199254740993n };  // Larger than Number.MAX_SAFE_INTEGER

const json = stringifyWithBigInt(data);
console.log(json);
// '{"count":{"__type":"BigInt","value":"9007199254740993"}}'

const restored = parseWithBigInt(json);
console.log(restored.count);  // 9007199254740993n
console.log(typeof restored.count);  // "bigint"
```

---

## Date Serialization

Dates serialize as ISO strings but parse as strings:

```javascript
const obj = { date: new Date('2025-01-24') };

const json = JSON.stringify(obj);
console.log(json);
// '{"date":"2025-01-24T00:00:00.000Z"}'

const parsed = JSON.parse(json);
console.log(parsed.date instanceof Date);  // false - it's a string!
console.log(typeof parsed.date);           // "string"
```

### Automatic Date Revival

```javascript
const dateReviver = (key, value) => {
  if (typeof value === 'string') {
    // ISO date pattern
    const datePattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/;
    if (datePattern.test(value)) {
      return new Date(value);
    }
  }
  return value;
};

const json = '{"created": "2025-01-24T10:30:00.000Z", "name": "Event"}';
const obj = JSON.parse(json, dateReviver);

console.log(obj.created instanceof Date);  // true
console.log(obj.created.toLocaleDateString());  // "1/24/2025"
```

---

## Performance Considerations

### Parsing Large JSON

```javascript
// For very large JSON (10MB+), consider streaming parsers
// or processing in Web Workers

// Move to worker
const worker = new Worker('json-worker.js');
worker.postMessage(largeJsonString);
worker.onmessage = (e) => {
  const parsed = e.data;
};
```

### Stringify Performance

```javascript
// Avoid stringifying repeatedly
// Cache if data doesn't change

// ❌ Bad - stringifies on every check
function hasChanged(obj) {
  const current = JSON.stringify(obj);
  if (current !== lastJson) {
    lastJson = current;
    return true;
  }
  return false;
}

// ✅ Better - use deep equality check or immutable data
```

---

## Hands-on Exercise

### Your Task

Create a `JSONStorage` class that wraps localStorage with automatic JSON serialization and date handling.

### Requirements

1. `set(key, value)` - Store any value as JSON
2. `get(key)` - Retrieve and parse, with Date revival
3. `remove(key)` - Delete a key
4. Handle errors gracefully

<details>
<summary>💡 Hints</summary>

- Use the date reviver pattern from earlier
- Wrap parse/stringify in try/catch
- Return null or undefined for missing keys

</details>

<details>
<summary>✅ Solution</summary>

```javascript
class JSONStorage {
  constructor(storage = localStorage) {
    this.storage = storage;
  }
  
  set(key, value) {
    try {
      const json = JSON.stringify(value);
      this.storage.setItem(key, json);
      return true;
    } catch (error) {
      console.error(`Failed to store ${key}:`, error);
      return false;
    }
  }
  
  get(key) {
    try {
      const json = this.storage.getItem(key);
      if (json === null) return null;
      
      return JSON.parse(json, this.dateReviver);
    } catch (error) {
      console.error(`Failed to retrieve ${key}:`, error);
      return null;
    }
  }
  
  remove(key) {
    this.storage.removeItem(key);
  }
  
  dateReviver(key, value) {
    if (typeof value === 'string') {
      const datePattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/;
      if (datePattern.test(value)) {
        return new Date(value);
      }
    }
    return value;
  }
}

// Usage
const storage = new JSONStorage();
storage.set('user', { name: 'Alice', joined: new Date() });

const user = storage.get('user');
console.log(user.joined instanceof Date);  // true
```

</details>

---

## Summary

✅ `JSON.parse()` converts JSON strings to JavaScript values
✅ `JSON.stringify()` converts JavaScript values to JSON
✅ Always wrap parsing in **try/catch**
✅ Use **reviver** functions to transform during parse (dates, case conversion)
✅ Use **replacer** functions to filter/transform during stringify
✅ Handle **circular references** with a WeakSet tracker
✅ **BigInt** and **Date** require custom serialization

**Next:** [Working with Complex Data Structures](./02-complex-data-structures.md)

---

## Further Reading

- [MDN JSON.parse()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse)
- [MDN JSON.stringify()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify)
- [JSON Specification](https://www.json.org/json-en.html)

<!-- 
Sources Consulted:
- MDN JSON.parse: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse
- MDN JSON.stringify: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify
-->
