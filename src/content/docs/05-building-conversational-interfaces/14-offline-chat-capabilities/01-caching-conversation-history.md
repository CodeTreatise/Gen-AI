---
title: "Caching Conversation History"
---

# Caching Conversation History

## Introduction

The foundation of offline chat capabilities is storing conversation data locally. When users lose connectivity, they should still see their message history, browse past conversations, and even compose new messages. This requires a robust caching strategy using Service Workers, the Cache API, and IndexedDB.

This lesson covers how to cache both application assets and conversation data, choosing the right storage for each type of content.

### What We'll Cover

- Setting up a Service Worker for chat applications
- Caching static assets with the Cache API
- Storing messages in IndexedDB
- Implementing cache invalidation strategies
- Managing storage quotas

### Prerequisites

- JavaScript Promises and async/await
- Basic understanding of HTTP requests
- Familiarity with browser developer tools

---

## Service Worker Fundamentals

A Service Worker is a JavaScript file that runs in a separate thread from your main page, intercepting network requests and enabling offline functionality.

### Service Worker Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download   │───▶│   Install   │───▶│  Activate   │
└─────────────┘    └──────┬──────┘    └──────┬──────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Waiting   │    │   Running   │
                   │ (if old SW  │    │  (controls  │
                   │   exists)   │    │   pages)    │
                   └─────────────┘    └─────────────┘
```

| Phase | What Happens | When to Act |
|-------|-------------|-------------|
| **Download** | Browser fetches SW file | Automatic |
| **Install** | SW is installing | Cache static assets |
| **Activate** | Old SW replaced | Clean up old caches |
| **Running** | SW controls pages | Intercept fetch requests |

### Registering a Service Worker

```javascript
// main.js - in your app's entry point
async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    console.log('Service Workers not supported');
    return null;
  }
  
  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/'
    });
    
    console.log('SW registered:', registration.scope);
    
    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          // New version available
          showUpdateNotification();
        }
      });
    });
    
    return registration;
  } catch (error) {
    console.error('SW registration failed:', error);
    return null;
  }
}

registerServiceWorker();
```

### Basic Service Worker Structure

```javascript
// sw.js - Service Worker file
const CACHE_VERSION = 'v1';
const STATIC_CACHE = `chat-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `chat-dynamic-${CACHE_VERSION}`;

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/scripts/chat.js',
  '/images/logo.svg',
  '/offline.html'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting()) // Activate immediately
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => name.startsWith('chat-') && name !== STATIC_CACHE && name !== DYNAMIC_CACHE)
            .map(name => caches.delete(name))
        );
      })
      .then(() => self.clients.claim()) // Take control of all pages
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  event.respondWith(handleFetch(event.request));
});

async function handleFetch(request) {
  // Try cache first for same-origin requests
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Otherwise fetch from network
  try {
    const response = await fetch(request);
    
    // Cache successful GET requests
    if (request.method === 'GET' && response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Network failed, return offline page for navigation
    if (request.mode === 'navigate') {
      return caches.match('/offline.html');
    }
    throw error;
  }
}
```

---

## Caching Strategies

Different types of content benefit from different caching strategies:

### Strategy Comparison

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Cache First** | Check cache, fallback to network | Static assets, images |
| **Network First** | Try network, fallback to cache | API data, fresh content |
| **Stale While Revalidate** | Return cache, update in background | Frequently updated content |
| **Network Only** | Always use network | Authentication, real-time data |
| **Cache Only** | Always use cache | Offline-only assets |

### Cache First (for Static Assets)

```javascript
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  const networkResponse = await fetch(request);
  cache.put(request, networkResponse.clone());
  return networkResponse;
}
```

### Network First (for API Data)

```javascript
async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    throw error;
  }
}
```

### Stale While Revalidate

```javascript
async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(request);
  
  // Fetch update in background
  const fetchPromise = fetch(request)
    .then(networkResponse => {
      cache.put(request, networkResponse.clone());
      return networkResponse;
    })
    .catch(() => null);
  
  // Return cached version immediately, or wait for network
  return cachedResponse || fetchPromise;
}
```

### Strategy Router

```javascript
// sw.js - Route requests to appropriate strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Static assets - cache first
  if (request.destination === 'style' || 
      request.destination === 'script' ||
      request.destination === 'image') {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
    return;
  }
  
  // API requests - network first
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request, DYNAMIC_CACHE));
    return;
  }
  
  // Conversation list - stale while revalidate
  if (url.pathname.startsWith('/api/conversations')) {
    event.respondWith(staleWhileRevalidate(request, DYNAMIC_CACHE));
    return;
  }
  
  // Default - network first with offline fallback
  event.respondWith(
    networkFirst(request, DYNAMIC_CACHE)
      .catch(() => caches.match('/offline.html'))
  );
});
```

---

## IndexedDB for Messages

While the Cache API is perfect for request/response pairs, IndexedDB is better suited for storing structured message data that needs to be queried.

### Why IndexedDB for Messages?

| Feature | Cache API | IndexedDB |
|---------|-----------|-----------|
| Query by field | ❌ No | ✅ Yes (indexes) |
| Update individual items | ❌ Replace entire response | ✅ Yes |
| Complex data structures | ⚠️ Limited | ✅ Full support |
| Storage size | Moderate | Large |
| Search/filter | ❌ No | ✅ Yes |

### Message Database Schema

```javascript
// db.js - IndexedDB wrapper for chat messages
class ChatDatabase {
  constructor() {
    this.dbName = 'ChatApp';
    this.dbVersion = 1;
    this.db = null;
  }
  
  async open() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Messages store
        if (!db.objectStoreNames.contains('messages')) {
          const messageStore = db.createObjectStore('messages', { 
            keyPath: 'id' 
          });
          
          // Indexes for efficient queries
          messageStore.createIndex('conversationId', 'conversationId', { unique: false });
          messageStore.createIndex('timestamp', 'timestamp', { unique: false });
          messageStore.createIndex('senderId', 'senderId', { unique: false });
          messageStore.createIndex('status', 'status', { unique: false });
          
          // Compound index for conversation + timestamp
          messageStore.createIndex('conversation_timestamp', 
            ['conversationId', 'timestamp'], { unique: false });
        }
        
        // Conversations store
        if (!db.objectStoreNames.contains('conversations')) {
          const convStore = db.createObjectStore('conversations', { 
            keyPath: 'id' 
          });
          
          convStore.createIndex('updatedAt', 'updatedAt', { unique: false });
        }
        
        // Offline queue store
        if (!db.objectStoreNames.contains('outbox')) {
          const outboxStore = db.createObjectStore('outbox', { 
            keyPath: 'id',
            autoIncrement: true
          });
          
          outboxStore.createIndex('status', 'status', { unique: false });
          outboxStore.createIndex('createdAt', 'createdAt', { unique: false });
        }
      };
    });
  }
  
  async ensureOpen() {
    if (!this.db) {
      await this.open();
    }
    return this.db;
  }
}
```

### Message CRUD Operations

```javascript
// Extend ChatDatabase with message operations
class ChatDatabase {
  // ... previous code ...
  
  // Save a message
  async saveMessage(message) {
    const db = await this.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readwrite');
      const store = tx.objectStore('messages');
      
      const request = store.put({
        ...message,
        cachedAt: Date.now()
      });
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Save multiple messages (batch operation)
  async saveMessages(messages) {
    const db = await this.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readwrite');
      const store = tx.objectStore('messages');
      
      messages.forEach(message => {
        store.put({
          ...message,
          cachedAt: Date.now()
        });
      });
      
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
  
  // Get messages for a conversation
  async getMessagesByConversation(conversationId, limit = 50) {
    const db = await this.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readonly');
      const store = tx.objectStore('messages');
      const index = store.index('conversation_timestamp');
      
      const range = IDBKeyRange.bound(
        [conversationId, 0],
        [conversationId, Date.now()]
      );
      
      const messages = [];
      const request = index.openCursor(range, 'prev'); // Newest first
      
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        
        if (cursor && messages.length < limit) {
          messages.push(cursor.value);
          cursor.continue();
        } else {
          resolve(messages.reverse()); // Return in chronological order
        }
      };
      
      request.onerror = () => reject(request.error);
    });
  }
  
  // Get a single message by ID
  async getMessage(id) {
    const db = await this.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readonly');
      const store = tx.objectStore('messages');
      const request = store.get(id);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Update message status
  async updateMessageStatus(id, status) {
    const db = await this.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readwrite');
      const store = tx.objectStore('messages');
      
      const getRequest = store.get(id);
      
      getRequest.onsuccess = () => {
        const message = getRequest.result;
        if (message) {
          message.status = status;
          message.updatedAt = Date.now();
          store.put(message);
        }
      };
      
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
  
  // Delete old messages (cleanup)
  async deleteOldMessages(maxAge = 30 * 24 * 60 * 60 * 1000) {
    const db = await this.ensureOpen();
    const cutoff = Date.now() - maxAge;
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messages', 'readwrite');
      const store = tx.objectStore('messages');
      const index = store.index('timestamp');
      
      const range = IDBKeyRange.upperBound(cutoff);
      let deletedCount = 0;
      
      const request = index.openCursor(range);
      
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          deletedCount++;
          cursor.continue();
        }
      };
      
      tx.oncomplete = () => resolve(deletedCount);
      tx.onerror = () => reject(tx.error);
    });
  }
}

// Export singleton
const chatDB = new ChatDatabase();
export default chatDB;
```

---

## Integrating Caching with Chat UI

### Fetching Messages with Cache Fallback

```javascript
// chatService.js
import chatDB from './db.js';

class ChatService {
  constructor() {
    this.apiBase = '/api';
  }
  
  async getConversationMessages(conversationId) {
    try {
      // Try network first
      const response = await fetch(
        `${this.apiBase}/conversations/${conversationId}/messages`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      
      const messages = await response.json();
      
      // Cache the fresh data
      await chatDB.saveMessages(messages);
      
      return { messages, source: 'network' };
      
    } catch (error) {
      console.log('Network failed, trying cache:', error.message);
      
      // Fallback to cached data
      const cachedMessages = await chatDB.getMessagesByConversation(conversationId);
      
      if (cachedMessages.length > 0) {
        return { messages: cachedMessages, source: 'cache' };
      }
      
      throw new Error('No messages available offline');
    }
  }
  
  async sendMessage(conversationId, content) {
    const message = {
      id: `temp-${Date.now()}`,
      conversationId,
      content,
      senderId: 'current-user',
      timestamp: Date.now(),
      status: 'sending'
    };
    
    // Save locally immediately (optimistic update)
    await chatDB.saveMessage(message);
    
    try {
      const response = await fetch(
        `${this.apiBase}/conversations/${conversationId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      
      const serverMessage = await response.json();
      
      // Update with server response
      await chatDB.saveMessage({
        ...serverMessage,
        status: 'sent'
      });
      
      return { message: serverMessage, status: 'sent' };
      
    } catch (error) {
      // Mark as queued for later
      await chatDB.updateMessageStatus(message.id, 'queued');
      
      return { message, status: 'queued' };
    }
  }
}

export default new ChatService();
```

### UI Integration

```javascript
// chat.js - UI component
import chatService from './chatService.js';

class ChatUI {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentConversation = null;
  }
  
  async loadConversation(conversationId) {
    this.currentConversation = conversationId;
    this.showLoading();
    
    try {
      const { messages, source } = await chatService.getConversationMessages(conversationId);
      
      this.renderMessages(messages);
      
      // Show source indicator
      if (source === 'cache') {
        this.showOfflineNotice('Showing cached messages');
      }
      
    } catch (error) {
      this.showError(error.message);
    }
  }
  
  renderMessages(messages) {
    const list = this.container.querySelector('.message-list');
    list.innerHTML = '';
    
    messages.forEach(message => {
      const el = this.createMessageElement(message);
      list.appendChild(el);
    });
    
    this.scrollToBottom();
  }
  
  createMessageElement(message) {
    const div = document.createElement('div');
    div.className = `message ${message.status || 'sent'}`;
    div.dataset.id = message.id;
    
    div.innerHTML = `
      <div class="message-content">${message.content}</div>
      <div class="message-meta">
        <span class="time">${this.formatTime(message.timestamp)}</span>
        <span class="status">${this.getStatusIcon(message.status)}</span>
      </div>
    `;
    
    return div;
  }
  
  getStatusIcon(status) {
    const icons = {
      sending: '⏳',
      sent: '✓',
      delivered: '✓✓',
      read: '✓✓',
      queued: '📤',
      failed: '❌'
    };
    return icons[status] || '';
  }
  
  formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }
  
  showOfflineNotice(message) {
    const notice = document.createElement('div');
    notice.className = 'offline-notice';
    notice.textContent = message;
    this.container.prepend(notice);
  }
  
  showLoading() {
    this.container.classList.add('loading');
  }
  
  showError(message) {
    this.container.classList.remove('loading');
    // Show error UI
  }
  
  scrollToBottom() {
    const list = this.container.querySelector('.message-list');
    list.scrollTop = list.scrollHeight;
  }
}
```

---

## Cache Invalidation

Stale data is a major concern. Implement strategies to keep caches fresh:

### Time-Based Invalidation

```javascript
// Check if cached data is stale
function isCacheStale(cachedAt, maxAge = 5 * 60 * 1000) {
  return Date.now() - cachedAt > maxAge;
}

async function getMessagesWithFreshness(conversationId) {
  const cached = await chatDB.getMessagesByConversation(conversationId);
  
  if (cached.length > 0) {
    const newestMessage = cached[cached.length - 1];
    
    if (!isCacheStale(newestMessage.cachedAt)) {
      return { messages: cached, fresh: true };
    }
  }
  
  // Fetch fresh data
  try {
    const fresh = await fetchFromNetwork(conversationId);
    await chatDB.saveMessages(fresh);
    return { messages: fresh, fresh: true };
  } catch {
    return { messages: cached, fresh: false };
  }
}
```

### Event-Based Invalidation

```javascript
// Invalidate when receiving updates via WebSocket
websocket.on('message_update', async (data) => {
  await chatDB.saveMessage(data.message);
  ui.updateMessage(data.message);
});

websocket.on('conversation_update', async (data) => {
  // Clear cached messages for this conversation
  // and refresh
  await refreshConversation(data.conversationId);
});
```

### Version-Based Invalidation

```javascript
// sw.js - Cache versioning
const CACHE_VERSION = 'v2'; // Increment to invalidate all caches

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(names => {
      return Promise.all(
        names
          .filter(name => !name.includes(CACHE_VERSION))
          .map(name => caches.delete(name))
      );
    })
  );
});
```

---

## Storage Quota Management

Browsers limit how much data can be stored. Monitor and manage usage:

### Checking Storage Usage

```javascript
async function checkStorageQuota() {
  if ('storage' in navigator && 'estimate' in navigator.storage) {
    const estimate = await navigator.storage.estimate();
    
    const usage = estimate.usage || 0;
    const quota = estimate.quota || 0;
    const percent = quota > 0 ? (usage / quota * 100).toFixed(2) : 0;
    
    return {
      used: formatBytes(usage),
      total: formatBytes(quota),
      percent: `${percent}%`,
      available: formatBytes(quota - usage)
    };
  }
  
  return null;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
```

### Requesting Persistent Storage

```javascript
async function requestPersistentStorage() {
  if ('storage' in navigator && 'persist' in navigator.storage) {
    const isPersisted = await navigator.storage.persisted();
    
    if (!isPersisted) {
      const granted = await navigator.storage.persist();
      console.log('Persistent storage:', granted ? 'granted' : 'denied');
      return granted;
    }
    
    return true;
  }
  
  return false;
}
```

### Cleanup Strategies

```javascript
async function cleanupStorage() {
  const quota = await checkStorageQuota();
  
  // If using more than 80% of quota, clean up
  if (parseFloat(quota.percent) > 80) {
    console.log('Storage cleanup needed');
    
    // Delete messages older than 7 days
    const deletedCount = await chatDB.deleteOldMessages(7 * 24 * 60 * 60 * 1000);
    console.log(`Deleted ${deletedCount} old messages`);
    
    // Clear dynamic cache
    await caches.delete(DYNAMIC_CACHE);
  }
}

// Run cleanup periodically
setInterval(cleanupStorage, 60 * 60 * 1000); // Every hour
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Version your caches | Use a single cache indefinitely |
| Store messages in IndexedDB | Store messages in Cache API |
| Implement cache invalidation | Let caches grow stale forever |
| Monitor storage usage | Ignore quota limits |
| Handle IndexedDB errors | Assume operations always succeed |
| Test offline scenarios | Only test with good connectivity |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not handling SW registration errors | Wrap registration in try/catch |
| Caching POST requests | Only cache GET requests |
| Forgetting to clone responses | Use `response.clone()` before caching |
| Large cache size | Implement cleanup and limits |
| Not testing cache updates | Version caches and test upgrades |
| Ignoring IndexedDB version changes | Use `onupgradeneeded` properly |

---

## Hands-on Exercise

### Your Task

Implement a caching layer for a chat application:

1. **Service Worker** that caches static assets
2. **IndexedDB database** with messages and conversations stores
3. **Network-first strategy** for API requests with cache fallback
4. **Storage quota monitoring** with cleanup when needed

### Requirements

1. Register a Service Worker on page load
2. Cache at least 5 static assets during install
3. Store messages with conversation index
4. Fall back to cached messages when offline
5. Display whether data came from cache or network

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `event.waitUntil()` to extend the install event
- Clone responses before caching: `response.clone()`
- Create compound indexes for efficient queries
- Check `navigator.onLine` but don't rely on it exclusively

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**sw.js:**
```javascript
const CACHE_VERSION = 'v1';
const CACHE_NAME = `chat-app-${CACHE_VERSION}`;

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles.css',
  '/app.js',
  '/db.js',
  '/offline.html'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then(names => Promise.all(
        names
          .filter(name => name !== CACHE_NAME)
          .map(name => caches.delete(name))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  
  if (request.url.includes('/api/')) {
    // Network first for API
    event.respondWith(
      fetch(request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
  } else {
    // Cache first for static
    event.respondWith(
      caches.match(request)
        .then(cached => cached || fetch(request))
    );
  }
});
```

**db.js:**
```javascript
class ChatDB {
  constructor() {
    this.db = null;
  }
  
  async open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('ChatApp', 1);
      
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        
        const messages = db.createObjectStore('messages', { keyPath: 'id' });
        messages.createIndex('conversationId', 'conversationId');
        messages.createIndex('timestamp', 'timestamp');
        
        db.createObjectStore('conversations', { keyPath: 'id' });
      };
      
      req.onsuccess = () => {
        this.db = req.result;
        resolve(this.db);
      };
      req.onerror = () => reject(req.error);
    });
  }
  
  async saveMessage(msg) {
    if (!this.db) await this.open();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('messages', 'readwrite');
      tx.objectStore('messages').put({ ...msg, cachedAt: Date.now() });
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  }
  
  async getMessages(conversationId) {
    if (!this.db) await this.open();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('messages', 'readonly');
      const index = tx.objectStore('messages').index('conversationId');
      const req = index.getAll(conversationId);
      
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
}

export default new ChatDB();
```

**app.js:**
```javascript
import db from './db.js';

// Register Service Worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(reg => console.log('SW registered'))
    .catch(err => console.error('SW failed', err));
}

// Load messages with cache fallback
async function loadMessages(conversationId) {
  const indicator = document.getElementById('source');
  
  try {
    const res = await fetch(`/api/conversations/${conversationId}/messages`);
    const messages = await res.json();
    
    // Cache for offline
    for (const msg of messages) {
      await db.saveMessage(msg);
    }
    
    indicator.textContent = '🌐 Network';
    return messages;
    
  } catch (error) {
    // Offline fallback
    const cached = await db.getMessages(conversationId);
    indicator.textContent = '💾 Cache';
    return cached;
  }
}

// Check storage
async function showStorage() {
  if (navigator.storage?.estimate) {
    const { usage, quota } = await navigator.storage.estimate();
    console.log(`Using ${(usage/1024/1024).toFixed(2)}MB of ${(quota/1024/1024).toFixed(2)}MB`);
  }
}

showStorage();
```

</details>

---

## Summary

✅ Service Workers intercept network requests and enable offline functionality  
✅ Use Cache API for static assets and API response caching  
✅ Use IndexedDB for structured message data with efficient querying  
✅ Choose caching strategies based on content type (cache-first, network-first, stale-while-revalidate)  
✅ Implement cache invalidation to prevent stale data  
✅ Monitor storage quotas and implement cleanup strategies  

**Next:** [Offline Message Queueing](./02-offline-message-queueing.md)

---

<!-- 
Sources Consulted:
- MDN Service Worker API: https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API
- MDN Cache API: https://developer.mozilla.org/en-US/docs/Web/API/Cache
- MDN IndexedDB API: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
- MDN Storage API: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API
- web.dev Service Worker Lifecycle: https://web.dev/articles/service-worker-lifecycle
-->
