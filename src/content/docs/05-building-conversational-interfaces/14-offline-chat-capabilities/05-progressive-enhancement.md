---
title: "Progressive Enhancement"
---

# Progressive Enhancement

## Introduction

Progressive enhancement is a strategy where you build a solid baseline experience that works everywhere, then layer on advanced features for capable browsers. For offline chat, this means the app works without JavaScript or Service Workers, improves with basic JS, and delivers the full offline experience when all modern APIs are available.

This lesson covers designing chat applications that gracefully degrade when features aren't available while maximizing capabilities when they are.

### What We'll Cover

- Progressive enhancement principles for chat
- Feature detection strategies
- Layered functionality implementation
- Graceful degradation patterns
- Testing across capability levels

### Prerequisites

- Previous lessons in this unit
- Understanding of browser APIs
- JavaScript feature detection basics

---

## Progressive Enhancement Principles

### The Enhancement Pyramid

```
                    ┌─────────────────────┐
                    │   Full Offline      │  ← Service Workers,
                    │   Experience        │    Background Sync
                    ├─────────────────────┤
                    │   Enhanced JS       │  ← IndexedDB, Cache API
                    │   Features          │
                    ├─────────────────────┤
                    │   Basic JavaScript  │  ← Event handling,
                    │                     │    form validation
                    ├─────────────────────┤
                    │   HTML + CSS        │  ← Forms work,
                    │   (Core Experience) │    content readable
                    └─────────────────────┘
```

### Core Principles

| Principle | Description | Chat Example |
|-----------|-------------|--------------|
| **Core first** | Basic functionality without JS | Forms submit to server |
| **Feature detect** | Check before using APIs | Test for Service Worker support |
| **Enhance, don't require** | Add capabilities, don't break without | Offline is bonus, not required |
| **Fail gracefully** | Provide alternatives when features fail | Show cached messages if network fails |

---

## Feature Detection

### Checking for API Support

```javascript
// featureDetection.js
class FeatureDetection {
  constructor() {
    this.capabilities = this.detectCapabilities();
    console.log('Detected capabilities:', this.capabilities);
  }
  
  detectCapabilities() {
    return {
      // Core APIs
      serviceWorker: 'serviceWorker' in navigator,
      indexedDB: 'indexedDB' in window,
      cacheAPI: 'caches' in window,
      
      // Offline APIs
      backgroundSync: this.hasBackgroundSync(),
      periodicSync: this.hasPeriodicSync(),
      
      // Network APIs
      onLine: 'onLine' in navigator,
      connection: 'connection' in navigator,
      
      // Storage APIs
      storage: 'storage' in navigator,
      storageEstimate: this.hasStorageEstimate(),
      storagePersist: this.hasStoragePersist(),
      
      // Notification
      notification: 'Notification' in window,
      pushManager: this.hasPushManager(),
      
      // Other useful checks
      webSocket: 'WebSocket' in window,
      fetch: 'fetch' in window,
      promise: 'Promise' in window,
      asyncAwait: this.hasAsyncAwait()
    };
  }
  
  hasBackgroundSync() {
    if (!('serviceWorker' in navigator)) return false;
    return 'SyncManager' in window;
  }
  
  hasPeriodicSync() {
    if (!('serviceWorker' in navigator)) return false;
    return 'PeriodicSyncManager' in window;
  }
  
  hasStorageEstimate() {
    return 'storage' in navigator && 'estimate' in navigator.storage;
  }
  
  hasStoragePersist() {
    return 'storage' in navigator && 'persist' in navigator.storage;
  }
  
  hasPushManager() {
    return 'serviceWorker' in navigator && 'PushManager' in window;
  }
  
  hasAsyncAwait() {
    try {
      new Function('async () => {}');
      return true;
    } catch {
      return false;
    }
  }
  
  // Check capability
  has(feature) {
    return this.capabilities[feature] === true;
  }
  
  // Check multiple capabilities
  hasAll(...features) {
    return features.every(f => this.has(f));
  }
  
  // Check any capability
  hasAny(...features) {
    return features.some(f => this.has(f));
  }
  
  // Get capability tier
  getTier() {
    if (this.hasAll('serviceWorker', 'indexedDB', 'cacheAPI', 'backgroundSync')) {
      return 'full';
    }
    if (this.hasAll('serviceWorker', 'indexedDB', 'cacheAPI')) {
      return 'enhanced';
    }
    if (this.hasAll('indexedDB', 'fetch')) {
      return 'basic';
    }
    return 'minimal';
  }
}

export default new FeatureDetection();
```

### Using Feature Detection

```javascript
// Initialize app based on capabilities
import features from './featureDetection.js';

async function initializeApp() {
  const tier = features.getTier();
  console.log(`Initializing at tier: ${tier}`);
  
  switch (tier) {
    case 'full':
      await initFullOfflineSupport();
      break;
    case 'enhanced':
      await initEnhancedOfflineSupport();
      break;
    case 'basic':
      await initBasicCaching();
      break;
    case 'minimal':
      initMinimalFallback();
      break;
  }
  
  // Common initialization
  initUI();
}

async function initFullOfflineSupport() {
  // Register service worker
  const registration = await navigator.serviceWorker.register('/sw.js');
  
  // Setup background sync
  await setupBackgroundSync(registration);
  
  // Setup push notifications
  if (features.has('notification')) {
    await setupPushNotifications(registration);
  }
  
  // Request persistent storage
  if (features.has('storagePersist')) {
    await navigator.storage.persist();
  }
  
  console.log('Full offline support enabled');
}

async function initEnhancedOfflineSupport() {
  // Register service worker
  await navigator.serviceWorker.register('/sw.js');
  
  // No background sync - use fallback
  setupManualSync();
  
  console.log('Enhanced offline support enabled (no background sync)');
}

async function initBasicCaching() {
  // No service worker - use IndexedDB directly
  await initIndexedDBCache();
  
  // Setup manual online/offline handling
  setupConnectionHandling();
  
  console.log('Basic caching enabled');
}

function initMinimalFallback() {
  // No IndexedDB - use localStorage or just network
  console.log('Minimal mode - network only');
  showOfflineUnsupportedMessage();
}
```

---

## Layered Functionality

### Message Sending Layers

Each layer builds on the previous:

```javascript
// messageSender.js
import features from './featureDetection.js';

class MessageSender {
  constructor() {
    this.tier = features.getTier();
  }
  
  async send(conversationId, content) {
    switch (this.tier) {
      case 'full':
        return this.sendWithBackgroundSync(conversationId, content);
      case 'enhanced':
        return this.sendWithServiceWorker(conversationId, content);
      case 'basic':
        return this.sendWithIndexedDB(conversationId, content);
      default:
        return this.sendDirect(conversationId, content);
    }
  }
  
  // Tier: full - Background Sync queues and retries automatically
  async sendWithBackgroundSync(conversationId, content) {
    const message = await this.queueMessage(conversationId, content);
    
    // Show optimistic UI immediately
    this.showOptimisticMessage(message);
    
    // Register for background sync
    const registration = await navigator.serviceWorker.ready;
    await registration.sync.register('sync-messages');
    
    return message;
  }
  
  // Tier: enhanced - Service Worker caches but needs manual sync
  async sendWithServiceWorker(conversationId, content) {
    const message = await this.queueMessage(conversationId, content);
    
    this.showOptimisticMessage(message);
    
    // Try to send immediately if online
    if (navigator.onLine) {
      this.processQueue();
    } else {
      // Will sync when online event fires
      this.showQueuedNotice();
    }
    
    return message;
  }
  
  // Tier: basic - IndexedDB queue without Service Worker
  async sendWithIndexedDB(conversationId, content) {
    if (!navigator.onLine) {
      const message = await this.queueMessage(conversationId, content);
      this.showOptimisticMessage(message);
      this.showQueuedNotice();
      return message;
    }
    
    // Online - try to send directly with fallback
    try {
      return await this.sendDirect(conversationId, content);
    } catch (error) {
      // Network failed - queue it
      const message = await this.queueMessage(conversationId, content);
      this.showOptimisticMessage(message);
      return message;
    }
  }
  
  // Tier: minimal - Direct network only
  async sendDirect(conversationId, content) {
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return response.json();
  }
  
  async queueMessage(conversationId, content) {
    const message = {
      id: `local-${Date.now()}`,
      conversationId,
      content,
      status: 'pending',
      createdAt: Date.now()
    };
    
    await chatDB.saveToQueue(message);
    return message;
  }
  
  showOptimisticMessage(message) {
    // Add to UI immediately
    chatUI.addMessage(message);
  }
  
  showQueuedNotice() {
    showToast('Message queued - will send when online');
  }
  
  async processQueue() {
    // Process pending messages
    const pending = await chatDB.getPendingMessages();
    
    for (const message of pending) {
      try {
        await this.sendDirect(message.conversationId, message.content);
        await chatDB.removeFromQueue(message.id);
        chatUI.updateMessageStatus(message.id, 'sent');
      } catch {
        // Leave in queue for later
      }
    }
  }
}

export default new MessageSender();
```

### Message Loading Layers

```javascript
// messageLoader.js
import features from './featureDetection.js';

class MessageLoader {
  async loadConversation(conversationId) {
    const tier = features.getTier();
    
    if (tier === 'full' || tier === 'enhanced') {
      return this.loadWithCacheFirst(conversationId);
    } else if (tier === 'basic') {
      return this.loadWithIndexedDB(conversationId);
    } else {
      return this.loadNetworkOnly(conversationId);
    }
  }
  
  // Service Worker handles caching transparently
  async loadWithCacheFirst(conversationId) {
    try {
      const response = await fetch(`/api/conversations/${conversationId}/messages`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const messages = await response.json();
      return { messages, source: 'network' };
      
    } catch (error) {
      // Service Worker may serve cached response
      // If that also fails, check IndexedDB
      if (features.has('indexedDB')) {
        const cached = await chatDB.getMessages(conversationId);
        if (cached.length > 0) {
          return { messages: cached, source: 'cache' };
        }
      }
      
      throw error;
    }
  }
  
  // No Service Worker - manage cache manually
  async loadWithIndexedDB(conversationId) {
    if (navigator.onLine) {
      try {
        const response = await fetch(`/api/conversations/${conversationId}/messages`);
        const messages = await response.json();
        
        // Cache for offline
        await chatDB.saveMessages(messages);
        
        return { messages, source: 'network' };
      } catch {
        // Fall through to cache
      }
    }
    
    // Offline or network failed - use cache
    const cached = await chatDB.getMessages(conversationId);
    
    if (cached.length > 0) {
      return { messages: cached, source: 'cache' };
    }
    
    throw new Error('No cached messages available');
  }
  
  // No caching at all
  async loadNetworkOnly(conversationId) {
    if (!navigator.onLine) {
      throw new Error('You are offline');
    }
    
    const response = await fetch(`/api/conversations/${conversationId}/messages`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return { messages: await response.json(), source: 'network' };
  }
}

export default new MessageLoader();
```

---

## Graceful Degradation Patterns

### Feature Fallback Chain

```javascript
// syncStrategy.js
class SyncStrategy {
  async setupSync() {
    // Try Background Sync first
    if (await this.tryBackgroundSync()) {
      console.log('Using Background Sync');
      return 'background-sync';
    }
    
    // Fall back to visibility-based sync
    if (this.tryVisibilitySync()) {
      console.log('Using visibility-based sync');
      return 'visibility-sync';
    }
    
    // Fall back to polling
    if (this.tryPollingSync()) {
      console.log('Using polling sync');
      return 'polling-sync';
    }
    
    // Last resort: manual only
    console.log('Using manual sync only');
    return 'manual-sync';
  }
  
  async tryBackgroundSync() {
    if (!('serviceWorker' in navigator)) return false;
    
    try {
      const registration = await navigator.serviceWorker.ready;
      if (!('sync' in registration)) return false;
      
      // Register for sync
      await registration.sync.register('sync-messages');
      return true;
    } catch {
      return false;
    }
  }
  
  tryVisibilitySync() {
    // Sync when page becomes visible
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && navigator.onLine) {
        this.syncNow();
      }
    });
    
    // Sync when coming online
    window.addEventListener('online', () => {
      this.syncNow();
    });
    
    return true;
  }
  
  tryPollingSync() {
    // Poll every 30 seconds when online
    setInterval(() => {
      if (navigator.onLine && document.visibilityState === 'visible') {
        this.syncNow();
      }
    }, 30000);
    
    return true;
  }
  
  async syncNow() {
    // Implement actual sync logic
    console.log('Syncing...');
  }
}
```

### Storage Fallback Chain

```javascript
// storage.js
class StorageManager {
  constructor() {
    this.storage = this.initStorage();
  }
  
  initStorage() {
    // Try IndexedDB first
    if ('indexedDB' in window) {
      return new IndexedDBStorage();
    }
    
    // Fall back to localStorage
    if ('localStorage' in window) {
      console.log('IndexedDB unavailable, using localStorage');
      return new LocalStorageAdapter();
    }
    
    // Fall back to in-memory
    console.log('No persistent storage, using memory');
    return new MemoryStorage();
  }
  
  async save(key, value) {
    return this.storage.save(key, value);
  }
  
  async get(key) {
    return this.storage.get(key);
  }
  
  async delete(key) {
    return this.storage.delete(key);
  }
}

// IndexedDB implementation
class IndexedDBStorage {
  async save(key, value) {
    const db = await this.openDB();
    const tx = db.transaction('store', 'readwrite');
    tx.objectStore('store').put({ key, value });
    return new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  }
  
  async get(key) {
    const db = await this.openDB();
    const tx = db.transaction('store', 'readonly');
    const request = tx.objectStore('store').get(key);
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result?.value);
      request.onerror = () => reject(request.error);
    });
  }
  
  async delete(key) {
    const db = await this.openDB();
    const tx = db.transaction('store', 'readwrite');
    tx.objectStore('store').delete(key);
    return new Promise((resolve) => {
      tx.oncomplete = resolve;
    });
  }
  
  async openDB() {
    // Cached DB connection
    if (this.db) return this.db;
    
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('AppStorage', 1);
      request.onupgradeneeded = (e) => {
        e.target.result.createObjectStore('store', { keyPath: 'key' });
      };
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };
      request.onerror = () => reject(request.error);
    });
  }
}

// localStorage adapter (limited to strings, 5MB)
class LocalStorageAdapter {
  async save(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      if (e.name === 'QuotaExceededError') {
        // Clear old items and retry
        this.cleanup();
        localStorage.setItem(key, JSON.stringify(value));
      }
    }
  }
  
  async get(key) {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : undefined;
  }
  
  async delete(key) {
    localStorage.removeItem(key);
  }
  
  cleanup() {
    // Remove oldest items
    const keys = Object.keys(localStorage).sort();
    const toRemove = Math.floor(keys.length / 2);
    keys.slice(0, toRemove).forEach(k => localStorage.removeItem(k));
  }
}

// In-memory fallback
class MemoryStorage {
  constructor() {
    this.store = new Map();
  }
  
  async save(key, value) {
    this.store.set(key, value);
  }
  
  async get(key) {
    return this.store.get(key);
  }
  
  async delete(key) {
    this.store.delete(key);
  }
}

export default new StorageManager();
```

---

## HTML-First Approach

### Baseline HTML Form

The chat should work without JavaScript:

```html
<!-- Works without JS - form submits to server -->
<form action="/api/messages" method="POST" class="chat-form">
  <input type="hidden" name="conversationId" value="123">
  
  <label for="message-input" class="sr-only">Message</label>
  <input 
    type="text" 
    id="message-input"
    name="content" 
    placeholder="Type a message..."
    required
    autocomplete="off"
  >
  
  <button type="submit">Send</button>
</form>

<!-- Messages rendered server-side initially -->
<div class="messages" id="message-list">
  <article class="message received">
    <p>Hello!</p>
    <time datetime="2024-01-15T10:30:00Z">10:30 AM</time>
  </article>
  <article class="message sent">
    <p>Hi there!</p>
    <time datetime="2024-01-15T10:31:00Z">10:31 AM</time>
  </article>
</div>
```

### Progressive Enhancement with JavaScript

```javascript
// Enhance the form if JS is available
class ChatFormEnhancer {
  constructor(form) {
    this.form = form;
    this.enhance();
  }
  
  enhance() {
    // Prevent default form submission
    this.form.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleSubmit();
    });
    
    // Add JS-only features
    this.addTypingIndicator();
    this.addCharacterCount();
    this.enableEnterToSend();
  }
  
  async handleSubmit() {
    const input = this.form.querySelector('input[name="content"]');
    const content = input.value.trim();
    
    if (!content) return;
    
    // Use the appropriate send method based on capabilities
    await messageSender.send(
      this.form.querySelector('input[name="conversationId"]').value,
      content
    );
    
    input.value = '';
    input.focus();
  }
  
  addTypingIndicator() {
    const input = this.form.querySelector('input[name="content"]');
    let typingTimeout;
    
    input.addEventListener('input', () => {
      clearTimeout(typingTimeout);
      
      // Send typing indicator
      if (features.has('webSocket')) {
        websocket.send({ type: 'typing', conversationId: this.getConversationId() });
      }
      
      typingTimeout = setTimeout(() => {
        if (features.has('webSocket')) {
          websocket.send({ type: 'stopped-typing', conversationId: this.getConversationId() });
        }
      }, 1000);
    });
  }
  
  addCharacterCount() {
    const input = this.form.querySelector('input[name="content"]');
    const counter = document.createElement('span');
    counter.className = 'char-count';
    this.form.appendChild(counter);
    
    input.addEventListener('input', () => {
      counter.textContent = `${input.value.length}/1000`;
    });
  }
  
  enableEnterToSend() {
    const input = this.form.querySelector('input[name="content"]');
    
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.form.dispatchEvent(new Event('submit'));
      }
    });
  }
  
  getConversationId() {
    return this.form.querySelector('input[name="conversationId"]').value;
  }
}

// Only enhance if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

function init() {
  const form = document.querySelector('.chat-form');
  if (form) {
    new ChatFormEnhancer(form);
  }
}
```

---

## Testing Across Tiers

### Manual Testing Checklist

| Tier | How to Test | Expected Behavior |
|------|-------------|-------------------|
| **No JS** | Disable JavaScript in DevTools | Form submits, page reloads with new message |
| **Minimal** | Use very old browser or disable APIs | Basic send/receive works online only |
| **Basic** | Disable Service Worker | IndexedDB caching, manual sync |
| **Enhanced** | Use Firefox/Safari (no Background Sync) | SW caching, visibility-based sync |
| **Full** | Chrome with all APIs | Background Sync, push notifications |

### Automated Feature Testing

```javascript
// test/featureTests.js
describe('Progressive Enhancement', () => {
  describe('Feature Detection', () => {
    it('should correctly identify tier', () => {
      const features = new FeatureDetection();
      const tier = features.getTier();
      expect(['full', 'enhanced', 'basic', 'minimal']).toContain(tier);
    });
    
    it('should detect Service Worker support', () => {
      const features = new FeatureDetection();
      expect(typeof features.has('serviceWorker')).toBe('boolean');
    });
  });
  
  describe('Fallback Behavior', () => {
    it('should fall back to localStorage when IndexedDB fails', async () => {
      // Mock IndexedDB failure
      const originalIDB = window.indexedDB;
      delete window.indexedDB;
      
      const storage = new StorageManager();
      await storage.save('test', { value: 123 });
      const result = await storage.get('test');
      
      expect(result).toEqual({ value: 123 });
      
      window.indexedDB = originalIDB;
    });
    
    it('should work offline with cached data', async () => {
      // Seed cache
      await chatDB.saveMessages([
        { id: '1', content: 'Cached message', conversationId: 'conv1' }
      ]);
      
      // Simulate offline
      const originalOnline = navigator.onLine;
      Object.defineProperty(navigator, 'onLine', { value: false, writable: true });
      
      const loader = new MessageLoader();
      const { messages, source } = await loader.loadConversation('conv1');
      
      expect(source).toBe('cache');
      expect(messages.length).toBe(1);
      
      Object.defineProperty(navigator, 'onLine', { value: originalOnline });
    });
  });
});
```

### DevTools Testing

```javascript
// Helpers for testing in browser console
window.testOffline = {
  // Simulate offline mode
  goOffline() {
    // Chrome DevTools: Network > Offline checkbox is better
    // This is for programmatic testing
    Object.defineProperty(navigator, 'onLine', { 
      value: false, 
      writable: true,
      configurable: true
    });
    window.dispatchEvent(new Event('offline'));
  },
  
  // Simulate online mode
  goOnline() {
    Object.defineProperty(navigator, 'onLine', { 
      value: true, 
      writable: true,
      configurable: true 
    });
    window.dispatchEvent(new Event('online'));
  },
  
  // Check current tier
  getTier() {
    return features.getTier();
  },
  
  // List all capabilities
  getCapabilities() {
    return features.capabilities;
  }
};
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Start with working HTML forms | Require JavaScript for basic functionality |
| Detect features, not browsers | Use user agent sniffing |
| Provide fallbacks for every enhancement | Assume all users have modern browsers |
| Test at each capability tier | Only test in latest Chrome |
| Document capability requirements | Leave users guessing why features don't work |
| Gracefully handle API failures | Let uncaught errors break the app |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Testing only in Chrome | Test Firefox, Safari, and older browsers |
| Assuming Service Worker works | Always check registration success |
| Breaking the app when IndexedDB fails | Fall back to localStorage or memory |
| No visual feedback for degraded mode | Show what features are available |
| Forgetting server-side rendering | Provide HTML fallback for no-JS |
| Catching all errors silently | Log errors and inform users |

---

## Hands-on Exercise

### Your Task

Implement progressive enhancement for a chat feature:

1. **HTML baseline** that works without JavaScript
2. **Feature detection** for Service Worker, IndexedDB, and Background Sync
3. **Three tiers** of message sending (full, enhanced, basic)
4. **Fallback storage** chain (IndexedDB → localStorage → memory)

### Requirements

1. Form submits to server when JS is disabled
2. Feature detection identifies correct capability tier
3. Each tier uses appropriate APIs
4. Storage gracefully degrades
5. User can see which mode is active

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `<noscript>` to show messages when JS is off
- Feature detection should run synchronously on load
- Use `async`/`await` with try/catch for graceful failures
- Add a debug panel showing current tier

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chat</title>
  <style>
    .capability-badge {
      position: fixed;
      bottom: 10px;
      right: 10px;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 12px;
    }
    .capability-badge.full { background: #4caf50; color: white; }
    .capability-badge.enhanced { background: #2196f3; color: white; }
    .capability-badge.basic { background: #ff9800; color: white; }
    .capability-badge.minimal { background: #f44336; color: white; }
  </style>
</head>
<body>
  <!-- Works without JS -->
  <form action="/api/messages" method="POST" id="chat-form">
    <input type="hidden" name="conversationId" value="123">
    <input type="text" name="content" placeholder="Message..." required>
    <button type="submit">Send</button>
  </form>
  
  <div id="messages"></div>
  
  <noscript>
    <p>JavaScript is disabled. Chat will work but with limited features.</p>
  </noscript>
  
  <div id="capability-badge" class="capability-badge"></div>
  
  <script>
    // Feature Detection
    const features = {
      serviceWorker: 'serviceWorker' in navigator,
      indexedDB: 'indexedDB' in window,
      cacheAPI: 'caches' in window,
      backgroundSync: 'serviceWorker' in navigator && 'SyncManager' in window,
      localStorage: 'localStorage' in window
    };
    
    function getTier() {
      if (features.serviceWorker && features.indexedDB && features.backgroundSync) {
        return 'full';
      }
      if (features.serviceWorker && features.indexedDB) {
        return 'enhanced';
      }
      if (features.indexedDB || features.localStorage) {
        return 'basic';
      }
      return 'minimal';
    }
    
    // Storage fallback chain
    const storage = {
      async save(key, value) {
        if (features.indexedDB) {
          try {
            const db = await this.openDB();
            const tx = db.transaction('store', 'readwrite');
            tx.objectStore('store').put({ key, value });
            return;
          } catch (e) { console.log('IDB failed, trying localStorage'); }
        }
        
        if (features.localStorage) {
          try {
            localStorage.setItem(key, JSON.stringify(value));
            return;
          } catch (e) { console.log('localStorage failed'); }
        }
        
        // Memory fallback
        this.memory = this.memory || {};
        this.memory[key] = value;
      },
      
      async openDB() {
        return new Promise((resolve, reject) => {
          const req = indexedDB.open('ChatStore', 1);
          req.onupgradeneeded = e => {
            e.target.result.createObjectStore('store', { keyPath: 'key' });
          };
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => reject(req.error);
        });
      }
    };
    
    // Tiered message sending
    async function sendMessage(content) {
      const tier = getTier();
      
      switch (tier) {
        case 'full':
          await storage.save('pending-' + Date.now(), { content });
          const reg = await navigator.serviceWorker.ready;
          await reg.sync.register('sync-messages');
          break;
          
        case 'enhanced':
          await storage.save('pending-' + Date.now(), { content });
          if (navigator.onLine) processPending();
          break;
          
        case 'basic':
          if (navigator.onLine) {
            await fetch('/api/messages', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ content, conversationId: '123' })
            });
          } else {
            await storage.save('pending-' + Date.now(), { content });
          }
          break;
          
        default:
          await fetch('/api/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, conversationId: '123' })
          });
      }
    }
    
    // Show capability badge
    const tier = getTier();
    const badge = document.getElementById('capability-badge');
    badge.textContent = `Mode: ${tier}`;
    badge.classList.add(tier);
    
    // Enhance form
    document.getElementById('chat-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = e.target.querySelector('input[name="content"]');
      await sendMessage(input.value);
      input.value = '';
    });
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ Build a working baseline first, then enhance with JavaScript  
✅ Feature detect APIs before using them—never assume support  
✅ Implement fallback chains for storage (IndexedDB → localStorage → memory)  
✅ Implement fallback chains for sync (Background Sync → visibility → polling)  
✅ Test at every capability tier, not just the best case  
✅ Show users which mode is active and what features are available  

**Next:** [Return to Offline Capabilities Overview](./00-offline-capabilities-overview.md)

---

<!-- 
Sources Consulted:
- MDN Progressive Enhancement: https://developer.mozilla.org/en-US/docs/Glossary/Progressive_Enhancement
- web.dev Feature Detection: https://web.dev/articles/feature-detection
- MDN Using Feature Detection: https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Cross_browser_testing/Feature_detection
-->
