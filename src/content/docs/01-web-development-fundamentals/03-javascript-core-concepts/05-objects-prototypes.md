---
title: "Objects and Prototypes"
---

# Objects and Prototypes

## Introduction

Objects are JavaScript's fundamental data structure—collections of key-value pairs that represent entities, configurations, and complex data. When building AI applications, you'll use objects to structure API requests, manage application state, represent user profiles, and configure models. Understanding object literals, the prototype chain, and modern class syntax enables you to organize code effectively and leverage JavaScript's object-oriented capabilities.

Unlike class-based languages, JavaScript uses **prototypal inheritance**, where objects can inherit properties from other objects. Modern JavaScript provides class syntax that makes this more familiar, but understanding prototypes reveals how the language truly works under the hood.

### What We'll Cover
- Object literals and property access
- Creating and manipulating objects
- Object methods and `this` binding
- Prototypes and the prototype chain
- Constructor functions and classes
- Modern features: `Symbol`, `BigInt`, property descriptors

### Prerequisites
- Variables and data types
- Functions and arrow functions
- Understanding of scope and `this` context

---

## Object Literals

The most common way to create objects:

```javascript
const user = {
  name: "Alice",
  age: 30,
  email: "alice@example.com"
};

console.log(user.name);     // "Alice" (dot notation)
console.log(user["age"]);   // 30 (bracket notation)
```

**Output:**
```
Alice
30
```

Bracket notation is useful for dynamic property access:

```javascript
const propertyName = "email";
console.log(user[propertyName]);  // "alice@example.com"

// Property names with spaces or special characters
const config = {
  "api-key": "abc123",
  "max retries": 3
};

console.log(config["api-key"]);       // "abc123"
console.log(config["max retries"]);   // 3
```

**Output:**
```
alice@example.com
abc123
3
```

---

## Object Methods

Functions as object properties:

```javascript
const calculator = {
  value: 0,
  add(n) {
    this.value += n;
    return this;  // Enable chaining
  },
  multiply(n) {
    this.value *= n;
    return this;
  },
  getResult() {
    return this.value;
  }
};

calculator.add(5).multiply(2);
console.log(calculator.getResult());  // 10
```

**Output:**
```
10
```

### The `this` Keyword

In methods, `this` refers to the object the method was called on:

```javascript
const person = {
  name: "Bob",
  greet() {
    console.log(`Hello, I'm ${this.name}`);
  }
};

person.greet();  // "Hello, I'm Bob"

// But if you extract the method:
const greetFunction = person.greet;
greetFunction();  // Error or undefined (this is not person)
```

**Output:**
```
Hello, I'm Bob
```

Arrow functions don't have their own `this`:

```javascript
const obj = {
  name: "Charlie",
  regularMethod() {
    console.log("Regular:", this.name);
  },
  arrowMethod: () => {
    console.log("Arrow:", this.name);  // 'this' from outer scope
  }
};

obj.regularMethod();  // "Regular: Charlie"
obj.arrowMethod();    // "Arrow: undefined"
```

**Output:**
```
Regular: Charlie
Arrow: undefined
```

---

## Creating and Manipulating Objects

### Adding and Modifying Properties

```javascript
const user = { name: "Alice" };

user.age = 30;           // Add property
user["email"] = "a@example.com";  // Add with bracket notation
user.name = "Alice Smith";        // Modify property

console.log(user);
// { name: "Alice Smith", age: 30, email: "a@example.com" }
```

**Output:**
```
{ name: 'Alice Smith', age: 30, email: 'a@example.com' }
```

### Deleting Properties

```javascript
const user = { name: "Bob", age: 25, temp: "delete me" };

delete user.temp;
console.log(user);  // { name: "Bob", age: 25 }
```

**Output:**
```
{ name: 'Bob', age: 25 }
```

### Checking Property Existence

```javascript
const user = { name: "Alice", age: 30 };

console.log("name" in user);       // true
console.log("email" in user);      // false
console.log(user.hasOwnProperty("age"));  // true
```

**Output:**
```
true
false
true
```

### Object.keys(), Object.values(), Object.entries()

```javascript
const config = {
  host: "localhost",
  port: 3000,
  debug: true
};

console.log(Object.keys(config));    // ["host", "port", "debug"]
console.log(Object.values(config));  // ["localhost", 3000, true]
console.log(Object.entries(config)); 
// [["host", "localhost"], ["port", 3000], ["debug", true]]
```

**Output:**
```
[ 'host', 'port', 'debug' ]
[ 'localhost', 3000, true ]
[ [ 'host', 'localhost' ], [ 'port', 3000 ], [ 'debug', true ] ]
```

### Object.assign() and Spread Operator

Copy and merge objects:

```javascript
const defaults = { theme: "light", notifications: true };
const userPrefs = { theme: "dark" };

// Object.assign (mutates first argument)
const config1 = Object.assign({}, defaults, userPrefs);
console.log(config1);  // { theme: "dark", notifications: true }

// Spread operator (more common)
const config2 = { ...defaults, ...userPrefs };
console.log(config2);  // { theme: "dark", notifications: true }
```

**Output:**
```
{ theme: 'dark', notifications: true }
{ theme: 'dark', notifications: true }
```

---

## Prototypes and Inheritance

Every object in JavaScript has a hidden `[[Prototype]]` property that links to another object.

### The Prototype Chain

```javascript
const animal = {
  eats: true,
  walk() {
    console.log("Animal walks");
  }
};

const rabbit = {
  jumps: true
};

// Set rabbit's prototype to animal
Object.setPrototypeOf(rabbit, animal);

console.log(rabbit.jumps);  // true (own property)
console.log(rabbit.eats);   // true (inherited from animal)
rabbit.walk();              // "Animal walks" (inherited method)
```

**Output:**
```
true
true
Animal walks
```

### Constructor Functions

Traditional way to create objects with shared methods:

```javascript
function User(name, email) {
  this.name = name;
  this.email = email;
}

// Methods on prototype (shared across instances)
User.prototype.greet = function() {
  return `Hello, I'm ${this.name}`;
};

const user1 = new User("Alice", "alice@example.com");
const user2 = new User("Bob", "bob@example.com");

console.log(user1.greet());  // "Hello, I'm Alice"
console.log(user2.greet());  // "Hello, I'm Bob"

// Both share the same greet method
console.log(user1.greet === user2.greet);  // true
```

**Output:**
```
Hello, I'm Alice
Hello, I'm Bob
true
```

---

## ES6 Classes

Modern syntax for creating objects with prototypes:

```javascript
class User {
  constructor(name, email) {
    this.name = name;
    this.email = email;
  }
  
  greet() {
    return `Hello, I'm ${this.name}`;
  }
  
  static createGuest() {
    return new User("Guest", "guest@example.com");
  }
}

const user = new User("Alice", "alice@example.com");
console.log(user.greet());  // "Hello, I'm Alice"

const guest = User.createGuest();
console.log(guest.name);  // "Guest"
```

**Output:**
```
Hello, I'm Alice
Guest
```

### Class Inheritance

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }
  
  speak() {
    return `${this.name} makes a sound`;
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name);  // Call parent constructor
    this.breed = breed;
  }
  
  speak() {
    return `${this.name} barks`;
  }
}

const dog = new Dog("Buddy", "Golden Retriever");
console.log(dog.speak());  // "Buddy barks"
console.log(dog.breed);    // "Golden Retriever"
```

**Output:**
```
Buddy barks
Golden Retriever
```

### Private Fields

Modern JavaScript supports private fields with `#`:

```javascript
class Counter {
  #count = 0;  // Private field
  
  increment() {
    this.#count++;
  }
  
  getCount() {
    return this.#count;
  }
}

const counter = new Counter();
counter.increment();
console.log(counter.getCount());  // 1
console.log(counter.#count);      // ❌ SyntaxError: Private field
```

**Output:**
```
1
SyntaxError: Private field '#count' must be declared in an enclosing class
```

---

## Symbol

Unique identifiers, often used for object properties that shouldn't conflict:

```javascript
const id1 = Symbol("id");
const id2 = Symbol("id");

console.log(id1 === id2);  // false (each Symbol is unique)

const user = {
  name: "Alice",
  [id1]: 12345  // Symbol as property key
};

console.log(user[id1]);  // 12345
console.log(user.id1);   // undefined (not a string property)

// Symbols don't appear in Object.keys()
console.log(Object.keys(user));  // ["name"]
```

**Output:**
```
false
12345
undefined
[ 'name' ]
```

### Well-Known Symbols

JavaScript has built-in symbols for customizing object behavior:

```javascript
const myObject = {
  [Symbol.toStringTag]: "MyCustomObject"
};

console.log(myObject.toString());  // "[object MyCustomObject]"
```

**Output:**
```
[object MyCustomObject]
```

---

## BigInt

For integers larger than `Number.MAX_SAFE_INTEGER`:

```javascript
const bigNumber = 1234567890123456789012345678901234567890n;
const calculated = bigNumber * 2n;

console.log(bigNumber);
console.log(calculated);

// Cannot mix BigInt and Number
// console.log(bigNumber + 1);  // ❌ TypeError

// Convert between types
const fromNumber = BigInt(123);
const toNumber = Number(123n);
```

**Output:**
```
1234567890123456789012345678901234567890n
2469135780246913578024691357802469135780n
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use object literals for simple data | Cleaner than `new Object()` |
| Use classes for objects with behavior | Provides clear structure and inheritance |
| Prefer composition over deep inheritance | Avoid complex inheritance chains; favor mixins/composition |
| Use `const` for objects | Prevents reassignment (properties can still be modified) |
| Use computed property names sparingly | `{ [key]: value }` is powerful but can reduce readability |
| Use `Object.freeze()` for immutable objects | Prevents any modifications to object |
| Use private fields for encapsulation | Prevents external access with `#` prefix |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting `new` with constructor functions | Use ES6 classes or check `new.target` in constructor |
| Using arrow functions as methods | Arrow functions don't have `this`—use regular methods |
| Modifying `Object.prototype` | Never modify built-in prototypes—causes global issues |
| Assuming `Object.keys()` returns Symbols | Symbols are excluded; use `Object.getOwnPropertySymbols()` |
| Confusing `in` with `hasOwnProperty` | `in` checks prototype chain; `hasOwnProperty` checks own properties only |
| Shallow copying objects with nested objects | Use deep copy libraries or `structuredClone()` for nested objects |

---

## Hands-on Exercise

### Your Task
Create a `ChatMessage` class and a `Conversation` class that manages a collection of messages. Implement inheritance, private fields, and object manipulation methods.

### Requirements
1. `ChatMessage` class with:
   - Properties: `role`, `content`, `timestamp`
   - Method: `toJSON()` returns plain object representation
2. `Conversation` class with:
   - Private field: `#messages` array
   - Method: `addMessage(role, content)` creates and stores a ChatMessage
   - Method: `getMessages()` returns copy of messages
   - Method: `getMessagesByRole(role)` filters by role
   - Static method: `createWithHistory(messages)` creates pre-populated conversation

### Expected Result
```javascript
const conv = new Conversation();
conv.addMessage("user", "Hello");
conv.addMessage("assistant", "Hi there!");

console.log(conv.getMessages().length);  // 2
console.log(conv.getMessagesByRole("user").length);  // 1
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `class` syntax for both ChatMessage and Conversation
- Private field syntax: `#messages = []`
- `addMessage` should create new ChatMessage instance
- Return copies of arrays with spread operator: `[...this.#messages]`
- Static methods use `static methodName()`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class ChatMessage {
  constructor(role, content) {
    this.role = role;
    this.content = content;
    this.timestamp = Date.now();
  }
  
  toJSON() {
    return {
      role: this.role,
      content: this.content,
      timestamp: this.timestamp
    };
  }
}

class Conversation {
  #messages = [];
  
  addMessage(role, content) {
    const message = new ChatMessage(role, content);
    this.#messages.push(message);
    return message;
  }
  
  getMessages() {
    return [...this.#messages];  // Return copy
  }
  
  getMessagesByRole(role) {
    return this.#messages.filter(msg => msg.role === role);
  }
  
  static createWithHistory(messages) {
    const conversation = new Conversation();
    messages.forEach(msg => {
      conversation.addMessage(msg.role, msg.content);
    });
    return conversation;
  }
  
  getMessageCount() {
    return this.#messages.length;
  }
}

// Test
const conv = new Conversation();
conv.addMessage("user", "Hello");
conv.addMessage("assistant", "Hi there!");
conv.addMessage("user", "How are you?");

console.log("Total messages:", conv.getMessages().length);
console.log("User messages:", conv.getMessagesByRole("user").length);
console.log("Assistant messages:", conv.getMessagesByRole("assistant").length);

// Test static method
const historyConv = Conversation.createWithHistory([
  { role: "user", content: "Previous message" },
  { role: "assistant", content: "Previous response" }
]);
console.log("History conv messages:", historyConv.getMessageCount());

// Verify private field
console.log("Direct access to #messages:", conv.#messages);  // SyntaxError
```
</details>

### Bonus Challenges
- [ ] Add a `Message` base class with `User Message` and `AssistantMessage` subclasses
- [ ] Implement message search with `findMessage(predicate)` method
- [ ] Add getters/setters for conversation metadata (title, created date)
- [ ] Implement `[Symbol.iterator]` to make Conversation iterable

---

## Summary

✅ Objects are key-value collections; use literals for simple data, classes for behavior
✅ Prototypes enable inheritance—every object has a hidden prototype link
✅ ES6 classes provide familiar syntax for constructor functions and prototypes
✅ Use `Symbol` for unique property keys; use `BigInt` for large integers
✅ Private fields with `#` enable true encapsulation in classes

[Previous: Scope and Closures](./04-scope-closures.md) | [Next: Data Structures](./06-data-structures.md)

---

<!-- 
Sources Consulted:
- MDN JavaScript Guide - Working with Objects: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Working_with_Objects
- MDN Classes: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Classes
- MDN Object.prototype: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object
- MDN Symbol: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Symbol
- MDN BigInt: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/BigInt
-->