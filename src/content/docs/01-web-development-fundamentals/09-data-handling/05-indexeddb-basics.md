---
title: "IndexedDB Basics"
---

# IndexedDB Basics

## Introduction

LocalStorage is great for small amounts of data, but what about storing thousands of records? Images? Large JSON datasets? That's where **IndexedDB** comes in—a full-fledged database in the browser.

IndexedDB is a NoSQL database that stores structured data (including files and blobs) with indexes for fast querying. It's asynchronous, transactional, and can store **far more data** than localStorage (often hundreds of MB or more).

### What We'll Cover

- What is IndexedDB
- Object stores (tables)
- Indexes for querying
- Transactions
- Cursor-based iteration
- IndexedDB vs localStorage
- Wrapper libraries (idb, Dexie)

### Prerequisites

- JavaScript Promises and async/await
- Understanding of databases (basic)
- LocalStorage fundamentals

---

## What is IndexedDB?

IndexedDB is a low-level API for storing large amounts of structured data in the browser.

### Key Features

| Feature | Description |
|---------|-------------|
| **Asynchronous** | Non-blocking, uses events or Promises |
| **Transactional** | All operations happen in transactions |
| **Object-oriented** | Stores JavaScript objects directly |
| **Indexed** | Create indexes for fast queries |
| **Large capacity** | Hundreds of MB (browser-dependent) |
| **Same-origin** | Data isolated per origin |

### When to Use IndexedDB

| Use Case | Best Storage |
|----------|--------------|
| User preferences (< 10 items) | localStorage |
| Auth tokens | sessionStorage |
| Offline-first app data | **IndexedDB** |
| Cached API responses | **IndexedDB** |
| Large file storage | **IndexedDB** |
| Search index | **IndexedDB** |

---

## Database and Object Stores

IndexedDB has a hierarchy: **Database → Object Store → Records**

Think of object stores as tables, but they store JavaScript objects.

### Opening a Database

```javascript
const request = indexedDB.open('MyDatabase', 1);

request.onerror = (event) => {
  console.error('Database error:', event.target.error);
};

request.onsuccess = (event) => {
  const db = event.target.result;
  console.log('Database opened:', db.name);
};

request.onupgradeneeded = (event) => {
  const db = event.target.result;
  
  // Create object store (only in upgrade)
  if (!db.objectStoreNames.contains('users')) {
    const store = db.createObjectStore('users', { keyPath: 'id' });
    console.log('Created users store');
  }
};
```

### The Upgrade Process

Schema changes only happen in `onupgradeneeded`:

```javascript
const request = indexedDB.open('MyDatabase', 2);  // Increment version

request.onupgradeneeded = (event) => {
  const db = event.target.result;
  const oldVersion = event.oldVersion;
  
  // Migration based on version
  if (oldVersion < 1) {
    db.createObjectStore('users', { keyPath: 'id' });
  }
  
  if (oldVersion < 2) {
    db.createObjectStore('posts', { keyPath: 'id', autoIncrement: true });
  }
};
```

### Key Types

```javascript
// Explicit key path (recommended)
db.createObjectStore('users', { keyPath: 'id' });
// { id: 123, name: 'Alice' }

// Auto-increment
db.createObjectStore('logs', { autoIncrement: true });
// Key is generated: 1, 2, 3...

// Out-of-line keys
db.createObjectStore('files', { });
// Provide key separately: store.add(file, 'my-key')
```

---

## Basic CRUD Operations

### Create (Add)

```javascript
function addUser(db, user) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readwrite');
    const store = tx.objectStore('users');
    const request = store.add(user);
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Usage
await addUser(db, { id: 1, name: 'Alice', email: 'alice@example.com' });
```

### Read (Get)

```javascript
function getUser(db, id) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const request = store.get(id);
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Usage
const user = await getUser(db, 1);
console.log(user);  // { id: 1, name: 'Alice', email: '...' }
```

### Update (Put)

```javascript
function updateUser(db, user) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readwrite');
    const store = tx.objectStore('users');
    const request = store.put(user);  // put = add or update
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Usage
await updateUser(db, { id: 1, name: 'Alicia', email: 'alicia@example.com' });
```

### Delete

```javascript
function deleteUser(db, id) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readwrite');
    const store = tx.objectStore('users');
    const request = store.delete(id);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Usage
await deleteUser(db, 1);
```

### Get All

```javascript
function getAllUsers(db) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const request = store.getAll();
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

const users = await getAllUsers(db);
```

---

## Indexes

Indexes allow fast lookups by properties other than the primary key.

### Creating Indexes

```javascript
request.onupgradeneeded = (event) => {
  const db = event.target.result;
  const store = db.createObjectStore('users', { keyPath: 'id' });
  
  // Create indexes
  store.createIndex('email', 'email', { unique: true });
  store.createIndex('role', 'role', { unique: false });
  store.createIndex('name', 'name', { unique: false });
};
```

### Querying by Index

```javascript
function getUserByEmail(db, email) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const index = store.index('email');
    const request = index.get(email);
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

const user = await getUserByEmail(db, 'alice@example.com');
```

### Getting All by Index Value

```javascript
function getUsersByRole(db, role) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const index = store.index('role');
    const request = index.getAll(role);
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

const admins = await getUsersByRole(db, 'admin');
```

---

## Transactions

All IndexedDB operations happen in transactions. Transactions ensure data integrity.

### Transaction Modes

| Mode | Description |
|------|-------------|
| `readonly` | Read data only (can run concurrently) |
| `readwrite` | Read and modify data (exclusive lock) |

### Transaction Scope

```javascript
// Transaction over multiple stores
const tx = db.transaction(['users', 'posts'], 'readwrite');
const userStore = tx.objectStore('users');
const postStore = tx.objectStore('posts');

// All operations share the transaction
userStore.put(user);
postStore.put(post);

// Wait for transaction to complete
tx.oncomplete = () => console.log('All operations committed');
tx.onerror = () => console.error('Transaction failed, rolled back');
```

### Transaction Lifecycle

```javascript
const tx = db.transaction('users', 'readwrite');

tx.oncomplete = () => {
  console.log('Transaction completed successfully');
};

tx.onerror = (event) => {
  console.error('Transaction error:', event.target.error);
};

tx.onabort = () => {
  console.log('Transaction aborted');
};

// Abort manually
tx.abort();
```

---

## Cursor-Based Iteration

For large datasets, cursors let you process records one at a time without loading everything into memory.

### Basic Cursor

```javascript
function iterateUsers(db, callback) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const request = store.openCursor();
    
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      
      if (cursor) {
        callback(cursor.value);
        cursor.continue();  // Move to next record
      } else {
        resolve();  // No more records
      }
    };
    
    request.onerror = () => reject(request.error);
  });
}

// Usage
await iterateUsers(db, (user) => {
  console.log(user.name);
});
```

### Cursor with Range

```javascript
function getUsersInRange(db, minId, maxId) {
  return new Promise((resolve, reject) => {
    const results = [];
    const range = IDBKeyRange.bound(minId, maxId);
    
    const tx = db.transaction('users', 'readonly');
    const store = tx.objectStore('users');
    const request = store.openCursor(range);
    
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      
      if (cursor) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    
    request.onerror = () => reject(request.error);
  });
}

const users = await getUsersInRange(db, 10, 50);
```

### Key Range Types

```javascript
IDBKeyRange.only(value)           // Exactly this value
IDBKeyRange.lowerBound(x)         // >= x
IDBKeyRange.lowerBound(x, true)   // > x (exclusive)
IDBKeyRange.upperBound(y)         // <= y
IDBKeyRange.upperBound(y, true)   // < y (exclusive)
IDBKeyRange.bound(x, y)           // x <= key <= y
IDBKeyRange.bound(x, y, true, true) // x < key < y
```

---

## IndexedDB vs localStorage

| Feature | localStorage | IndexedDB |
|---------|--------------|-----------|
| Data type | Strings only | Any (objects, blobs, files) |
| Capacity | ~5MB | Hundreds of MB |
| API | Synchronous | Asynchronous |
| Querying | Key lookup only | Indexes, ranges, cursors |
| Transactions | No | Yes |
| Structure | Flat key-value | Object stores with indexes |
| Use case | Simple settings | Complex data, offline apps |

---

## Wrapper Libraries

The raw IndexedDB API is verbose. Use wrapper libraries for cleaner code.

### idb (Recommended)

Lightweight Promise wrapper by Jake Archibald:

```javascript
import { openDB } from 'idb';

// Open database
const db = await openDB('MyApp', 1, {
  upgrade(db) {
    const store = db.createObjectStore('users', { keyPath: 'id' });
    store.createIndex('email', 'email');
  }
});

// CRUD operations
await db.add('users', { id: 1, name: 'Alice', email: 'alice@example.com' });

const user = await db.get('users', 1);
const allUsers = await db.getAll('users');
const byEmail = await db.getFromIndex('users', 'email', 'alice@example.com');

await db.put('users', { id: 1, name: 'Alicia', email: 'alicia@example.com' });
await db.delete('users', 1);

// Transactions
const tx = db.transaction('users', 'readwrite');
await tx.store.add({ id: 2, name: 'Bob', email: 'bob@example.com' });
await tx.store.add({ id: 3, name: 'Charlie', email: 'charlie@example.com' });
await tx.done;  // Wait for commit
```

### Dexie

Feature-rich with a query builder:

```javascript
import Dexie from 'dexie';

const db = new Dexie('MyApp');

// Define schema
db.version(1).stores({
  users: '++id, email, role',
  posts: '++id, userId, createdAt'
});

// CRUD
await db.users.add({ name: 'Alice', email: 'alice@example.com', role: 'admin' });

const user = await db.users.get(1);
const admins = await db.users.where('role').equals('admin').toArray();
const recent = await db.users.orderBy('id').reverse().limit(10).toArray();

// Complex queries
const result = await db.users
  .where('role')
  .anyOf(['admin', 'moderator'])
  .and(user => user.email.includes('@example.com'))
  .toArray();

// Relationships
const userPosts = await db.posts.where('userId').equals(userId).toArray();
```

---

## Complete Example: Offline Notes

```javascript
import { openDB } from 'idb';

class NotesDB {
  constructor() {
    this.dbPromise = openDB('NotesApp', 1, {
      upgrade(db) {
        const store = db.createObjectStore('notes', {
          keyPath: 'id',
          autoIncrement: true
        });
        store.createIndex('createdAt', 'createdAt');
        store.createIndex('synced', 'synced');
      }
    });
  }
  
  async add(content) {
    const db = await this.dbPromise;
    const note = {
      content,
      createdAt: new Date(),
      updatedAt: new Date(),
      synced: false
    };
    const id = await db.add('notes', note);
    return { ...note, id };
  }
  
  async update(id, content) {
    const db = await this.dbPromise;
    const note = await db.get('notes', id);
    if (!note) throw new Error('Note not found');
    
    note.content = content;
    note.updatedAt = new Date();
    note.synced = false;
    
    await db.put('notes', note);
    return note;
  }
  
  async delete(id) {
    const db = await this.dbPromise;
    await db.delete('notes', id);
  }
  
  async getAll() {
    const db = await this.dbPromise;
    return db.getAllFromIndex('notes', 'createdAt');
  }
  
  async getUnsynced() {
    const db = await this.dbPromise;
    return db.getAllFromIndex('notes', 'synced', false);
  }
  
  async markSynced(ids) {
    const db = await this.dbPromise;
    const tx = db.transaction('notes', 'readwrite');
    
    for (const id of ids) {
      const note = await tx.store.get(id);
      if (note) {
        note.synced = true;
        await tx.store.put(note);
      }
    }
    
    await tx.done;
  }
  
  async clear() {
    const db = await this.dbPromise;
    await db.clear('notes');
  }
}

// Usage
const notes = new NotesDB();

const note = await notes.add('My first note');
console.log(note);  // { id: 1, content: '...', createdAt: Date, synced: false }

await notes.update(note.id, 'Updated content');

const allNotes = await notes.getAll();
const unsynced = await notes.getUnsynced();

// Sync with server
const toSync = await notes.getUnsynced();
await syncWithServer(toSync);
await notes.markSynced(toSync.map(n => n.id));
```

---

## Hands-on Exercise

### Your Task

Build a simple **offline-capable todo list** using IndexedDB.

### Requirements

1. Add todos with title and completed status
2. Toggle completed status
3. Delete todos
4. Persist across page reloads
5. Use the `idb` library (or raw API)

<details>
<summary>💡 Hints</summary>

- Store: `{ id, title, completed, createdAt }`
- Use `autoIncrement` for IDs
- `put` to toggle completed status

</details>

<details>
<summary>✅ Solution</summary>

```javascript
import { openDB } from 'idb';

const dbPromise = openDB('TodoApp', 1, {
  upgrade(db) {
    db.createObjectStore('todos', { keyPath: 'id', autoIncrement: true });
  }
});

const TodoDB = {
  async add(title) {
    const db = await dbPromise;
    return db.add('todos', {
      title,
      completed: false,
      createdAt: new Date()
    });
  },
  
  async toggle(id) {
    const db = await dbPromise;
    const todo = await db.get('todos', id);
    todo.completed = !todo.completed;
    return db.put('todos', todo);
  },
  
  async delete(id) {
    const db = await dbPromise;
    return db.delete('todos', id);
  },
  
  async getAll() {
    const db = await dbPromise;
    return db.getAll('todos');
  }
};

// Usage
await TodoDB.add('Learn IndexedDB');
await TodoDB.add('Build offline app');

const todos = await TodoDB.getAll();
await TodoDB.toggle(todos[0].id);
```

</details>

---

## Summary

✅ **IndexedDB** stores large amounts of structured data (hundreds of MB)
✅ Uses **object stores** (like tables) with optional indexes
✅ All operations are **asynchronous** and **transactional**
✅ Use **indexes** for fast lookups by non-primary fields
✅ **Cursors** for iterating large datasets efficiently
✅ Use **idb** or **Dexie** wrappers for cleaner code
✅ Perfect for **offline-first** applications

**Back to:** [Data Handling Overview](./00-data-handling.md)

---

## Further Reading

- [MDN IndexedDB API](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)
- [idb Library](https://github.com/jakearchibald/idb) - Promise wrapper
- [Dexie.js](https://dexie.org/) - Feature-rich wrapper
- [web.dev IndexedDB](https://web.dev/articles/indexeddb) - Best practices

<!-- 
Sources Consulted:
- MDN IndexedDB API: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
- idb library: https://github.com/jakearchibald/idb
-->
