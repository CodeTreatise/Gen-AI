---
title: "Sync on Reconnection"
---

# Sync on Reconnection

## Introduction

When a user regains internet connectivity, queued messages need to sync with the server. This process involves detecting connection changes, sending pending data, receiving missed messages, and resolving any conflicts. Done well, synchronization feels automatic and seamless.

This lesson covers connection detection strategies, the Background Sync API, and conflict resolution approaches for chat applications.

### What We'll Cover

- Detecting online/offline status changes
- Using the Background Sync API (where supported)
- Implementing fallback sync for unsupported browsers
- Handling sync conflicts
- Providing sync progress feedback

### Prerequisites

- [Caching Conversation History](./01-caching-conversation-history.md)
- [Offline Message Queueing](./02-offline-message-queueing.md)
- Service Worker fundamentals

---

## Connection Detection

### The navigator.onLine API

The simplest way to detect connectivity:

```javascript
// Check current status
console.log('Online:', navigator.onLine); // true or false

// Listen for changes
window.addEventListener('online', () => {
  console.log('Connection restored');
});

window.addEventListener('offline', () => {
  console.log('Connection lost');
});
```

> **Warning:** `navigator.onLine` only tells you if the browser has a network connection—not if the internet is actually reachable. A computer connected to a router with no internet will still report `true`.

### Robust Connection Detection

For reliable detection, combine multiple approaches:

```javascript
// connectionManager.js
class ConnectionManager {
  constructor() {
    this.isOnline = navigator.onLine;
    this.listeners = new Set();
    this.checkEndpoint = '/api/health';
    this.checkInterval = null;
    
    this.setupEventListeners();
  }
  
  setupEventListeners() {
    window.addEventListener('online', () => this.handleOnline());
    window.addEventListener('offline', () => this.handleOffline());
    
    // Check on visibility change (user returns to tab)
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this.verifyConnection();
      }
    });
  }
  
  handleOnline() {
    console.log('Browser reports online');
    this.verifyConnection();
  }
  
  handleOffline() {
    console.log('Browser reports offline');
    this.setOnline(false);
  }
  
  // Actually test connectivity
  async verifyConnection() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(this.checkEndpoint, {
        method: 'HEAD',
        cache: 'no-store',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      this.setOnline(response.ok);
    } catch (error) {
      console.log('Connection check failed:', error.message);
      this.setOnline(false);
    }
  }
  
  setOnline(online) {
    if (this.isOnline === online) return;
    
    this.isOnline = online;
    console.log('Connection state changed:', online ? 'ONLINE' : 'OFFLINE');
    
    this.notify({
      type: online ? 'online' : 'offline',
      timestamp: Date.now()
    });
  }
  
  // Start periodic checks (for unstable connections)
  startPeriodicCheck(intervalMs = 30000) {
    this.stopPeriodicCheck();
    
    this.checkInterval = setInterval(() => {
      if (navigator.onLine) {
        this.verifyConnection();
      }
    }, intervalMs);
  }
  
  stopPeriodicCheck() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }
  
  // Subscribe to connection changes
  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notify(event) {
    this.listeners.forEach(callback => {
      try {
        callback(event);
      } catch (e) {
        console.error('Connection listener error:', e);
      }
    });
  }
}

export default new ConnectionManager();
```

### Network Information API (Advanced)

For more details about connection quality:

```javascript
// Check connection type (Chrome/Android only)
if ('connection' in navigator) {
  const conn = navigator.connection;
  
  console.log('Connection type:', conn.effectiveType); // 4g, 3g, 2g, slow-2g
  console.log('Downlink:', conn.downlink, 'Mbps');
  console.log('RTT:', conn.rtt, 'ms');
  console.log('Data saver:', conn.saveData);
  
  // Listen for changes
  conn.addEventListener('change', () => {
    console.log('Network conditions changed');
    
    // Adjust sync strategy based on connection
    if (conn.effectiveType === 'slow-2g' || conn.saveData) {
      // Use lightweight sync
      syncManager.setStrategy('minimal');
    } else {
      // Full sync
      syncManager.setStrategy('full');
    }
  });
}
```

---

## Background Sync API

The Background Sync API allows you to defer actions until the user has stable connectivity. It works even if the user leaves the page.

### Browser Support

| Browser | Support |
|---------|---------|
| Chrome | ✅ 49+ |
| Edge | ✅ 79+ |
| Firefox | ❌ Not supported |
| Safari | ❌ Not supported |

> **Important:** Due to limited browser support, always implement a fallback for browsers that don't support Background Sync.

### Registering a Sync Event

```javascript
// In your main app code
async function registerSync(tag) {
  if (!('serviceWorker' in navigator)) {
    console.log('Service Workers not supported');
    return false;
  }
  
  const registration = await navigator.serviceWorker.ready;
  
  if (!('sync' in registration)) {
    console.log('Background Sync not supported');
    return false;
  }
  
  try {
    await registration.sync.register(tag);
    console.log(`Sync registered: ${tag}`);
    return true;
  } catch (error) {
    console.error('Sync registration failed:', error);
    return false;
  }
}

// Usage
async function queueMessageForSync(message) {
  // Save message to IndexedDB queue
  await messageQueue.enqueue(message);
  
  // Request background sync
  const synced = await registerSync('sync-messages');
  
  if (!synced) {
    // Fallback: try to send immediately if online
    if (navigator.onLine) {
      queueProcessor.processQueue();
    }
  }
}
```

### Handling Sync Events in Service Worker

```javascript
// sw.js
self.addEventListener('sync', (event) => {
  console.log('Sync event:', event.tag);
  
  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  } else if (event.tag === 'sync-conversations') {
    event.waitUntil(syncConversations());
  }
});

async function syncMessages() {
  // Open IndexedDB
  const db = await openDatabase();
  
  // Get pending messages
  const tx = db.transaction('messageQueue', 'readonly');
  const store = tx.objectStore('messageQueue');
  const index = store.index('status');
  const pending = await promisify(index.getAll('pending'));
  
  console.log(`Syncing ${pending.length} messages`);
  
  // Send each message
  for (const message of pending) {
    try {
      await sendMessage(message);
      await updateMessageStatus(message.id, 'sent');
    } catch (error) {
      console.error('Failed to sync message:', error);
      // Will retry on next sync event
    }
  }
}

async function sendMessage(message) {
  const response = await fetch('/api/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversationId: message.conversationId,
      content: message.content,
      clientId: message.id
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  
  return response.json();
}

// Helper to promisify IndexedDB requests
function promisify(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
```

### Periodic Background Sync

For regularly syncing data (Chrome 80+):

```javascript
// Request periodic sync permission
async function setupPeriodicSync() {
  const registration = await navigator.serviceWorker.ready;
  
  if (!('periodicSync' in registration)) {
    console.log('Periodic Sync not supported');
    return;
  }
  
  // Check permission
  const status = await navigator.permissions.query({
    name: 'periodic-background-sync',
  });
  
  if (status.state !== 'granted') {
    console.log('Periodic Sync not permitted');
    return;
  }
  
  try {
    await registration.periodicSync.register('sync-conversations', {
      minInterval: 24 * 60 * 60 * 1000, // Once per day
    });
    console.log('Periodic sync registered');
  } catch (error) {
    console.error('Periodic sync registration failed:', error);
  }
}
```

```javascript
// sw.js - Handle periodic sync
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'sync-conversations') {
    event.waitUntil(syncConversationList());
  }
});

async function syncConversationList() {
  try {
    const response = await fetch('/api/conversations');
    const conversations = await response.json();
    
    // Update cached conversations
    const db = await openDatabase();
    const tx = db.transaction('conversations', 'readwrite');
    const store = tx.objectStore('conversations');
    
    for (const conv of conversations) {
      store.put(conv);
    }
  } catch (error) {
    console.error('Periodic sync failed:', error);
  }
}
```

---

## Fallback Sync Strategy

For browsers without Background Sync:

```javascript
// syncManager.js
import connectionManager from './connectionManager.js';
import messageQueue from './messageQueue.js';

class SyncManager {
  constructor() {
    this.isSyncing = false;
    this.syncQueue = [];
    this.retryDelay = 5000;
    this.backgroundSyncSupported = false;
    
    this.checkBackgroundSyncSupport();
    this.setupListeners();
  }
  
  async checkBackgroundSyncSupport() {
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.ready;
      this.backgroundSyncSupported = 'sync' in registration;
    }
    
    console.log('Background Sync supported:', this.backgroundSyncSupported);
  }
  
  setupListeners() {
    // Sync when coming back online
    connectionManager.subscribe((event) => {
      if (event.type === 'online') {
        this.sync();
      }
    });
    
    // Sync when page becomes visible
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && connectionManager.isOnline) {
        this.sync();
      }
    });
    
    // Sync on page load
    if (connectionManager.isOnline) {
      this.sync();
    }
  }
  
  async sync() {
    if (this.isSyncing) {
      console.log('Sync already in progress');
      return;
    }
    
    if (!connectionManager.isOnline) {
      console.log('Offline, skipping sync');
      return;
    }
    
    this.isSyncing = true;
    this.notify('sync-start');
    
    try {
      // 1. Send pending outbound messages
      await this.syncOutbound();
      
      // 2. Fetch inbound messages we might have missed
      await this.syncInbound();
      
      this.notify('sync-complete');
      
    } catch (error) {
      console.error('Sync failed:', error);
      this.notify('sync-error', error);
      
      // Schedule retry
      setTimeout(() => this.sync(), this.retryDelay);
      
    } finally {
      this.isSyncing = false;
    }
  }
  
  async syncOutbound() {
    const pending = await messageQueue.getPending();
    let sent = 0;
    let failed = 0;
    
    for (const message of pending) {
      try {
        await this.sendMessage(message);
        await messageQueue.dequeue(message.id);
        sent++;
      } catch (error) {
        await messageQueue.updateStatus(message.id, 'failed', error.message);
        failed++;
      }
    }
    
    console.log(`Outbound sync: ${sent} sent, ${failed} failed`);
    return { sent, failed };
  }
  
  async sendMessage(message) {
    const response = await fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversationId: message.conversationId,
        content: message.content,
        clientId: message.id
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return response.json();
  }
  
  async syncInbound() {
    // Get last sync timestamp
    const lastSync = await this.getLastSyncTime();
    
    // Fetch messages since last sync
    const response = await fetch(`/api/messages/since/${lastSync}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const messages = await response.json();
    
    // Save to local cache
    for (const message of messages) {
      await chatDB.saveMessage(message);
    }
    
    // Update sync timestamp
    await this.setLastSyncTime(Date.now());
    
    console.log(`Inbound sync: ${messages.length} messages received`);
    return messages;
  }
  
  async getLastSyncTime() {
    return parseInt(localStorage.getItem('lastSyncTime') || '0', 10);
  }
  
  async setLastSyncTime(timestamp) {
    localStorage.setItem('lastSyncTime', timestamp.toString());
  }
  
  // Event subscription
  listeners = new Set();
  
  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notify(event, data = null) {
    this.listeners.forEach(cb => cb({ type: event, data }));
  }
}

export default new SyncManager();
```

---

## Conflict Resolution

When the same conversation is modified both offline and on another device, conflicts arise.

### Common Conflict Scenarios

| Scenario | Challenge | Resolution Strategy |
|----------|-----------|---------------------|
| Same message edited | Which version wins? | Last-write-wins or merge |
| Message deleted on server | Client tries to update deleted message | Accept deletion |
| New messages on both sides | Messages may interleave oddly | Order by server timestamp |
| Conversation renamed | Both have different names | Server wins or prompt user |

### Conflict Resolution Strategies

```javascript
// conflictResolver.js
class ConflictResolver {
  constructor() {
    this.strategy = 'server-wins'; // Default strategy
  }
  
  setStrategy(strategy) {
    const valid = ['server-wins', 'client-wins', 'last-write-wins', 'manual'];
    if (!valid.includes(strategy)) {
      throw new Error(`Invalid strategy: ${strategy}`);
    }
    this.strategy = strategy;
  }
  
  async resolveMessageConflict(clientMessage, serverMessage) {
    console.log('Resolving message conflict:', {
      client: clientMessage.id,
      server: serverMessage.id
    });
    
    switch (this.strategy) {
      case 'server-wins':
        return this.acceptServer(clientMessage, serverMessage);
        
      case 'client-wins':
        return this.acceptClient(clientMessage, serverMessage);
        
      case 'last-write-wins':
        return this.lastWriteWins(clientMessage, serverMessage);
        
      case 'manual':
        return this.promptUser(clientMessage, serverMessage);
        
      default:
        return this.acceptServer(clientMessage, serverMessage);
    }
  }
  
  acceptServer(clientMessage, serverMessage) {
    // Update local cache with server version
    return {
      action: 'replace',
      message: serverMessage,
      discarded: clientMessage
    };
  }
  
  acceptClient(clientMessage, serverMessage) {
    // Re-send client version to server
    return {
      action: 'retry',
      message: clientMessage,
      overwrite: true
    };
  }
  
  lastWriteWins(clientMessage, serverMessage) {
    const clientTime = clientMessage.updatedAt || clientMessage.createdAt;
    const serverTime = serverMessage.updatedAt || serverMessage.createdAt;
    
    if (clientTime > serverTime) {
      return this.acceptClient(clientMessage, serverMessage);
    } else {
      return this.acceptServer(clientMessage, serverMessage);
    }
  }
  
  async promptUser(clientMessage, serverMessage) {
    return new Promise((resolve) => {
      showConflictDialog({
        clientMessage,
        serverMessage,
        onResolve: (choice) => {
          if (choice === 'client') {
            resolve(this.acceptClient(clientMessage, serverMessage));
          } else if (choice === 'server') {
            resolve(this.acceptServer(clientMessage, serverMessage));
          } else {
            // User chose to keep both
            resolve({
              action: 'duplicate',
              messages: [serverMessage, clientMessage]
            });
          }
        }
      });
    });
  }
}

export default new ConflictResolver();
```

### Detecting Conflicts During Sync

```javascript
// Enhanced sync with conflict detection
async function syncMessageWithConflictCheck(message) {
  try {
    const response = await fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversationId: message.conversationId,
        content: message.content,
        clientId: message.id,
        clientTimestamp: message.createdAt
      })
    });
    
    const result = await response.json();
    
    if (response.status === 409) {
      // Conflict detected
      const resolution = await conflictResolver.resolveMessageConflict(
        message,
        result.serverVersion
      );
      
      return await applyResolution(resolution);
    }
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return result;
    
  } catch (error) {
    throw error;
  }
}

async function applyResolution(resolution) {
  switch (resolution.action) {
    case 'replace':
      // Update local with server version
      await chatDB.saveMessage(resolution.message);
      await messageQueue.dequeue(resolution.discarded.id);
      return resolution.message;
      
    case 'retry':
      // Re-send with overwrite flag
      return await forceUpdateMessage(resolution.message);
      
    case 'duplicate':
      // Keep both messages
      for (const msg of resolution.messages) {
        await chatDB.saveMessage(msg);
      }
      return resolution.messages;
      
    default:
      throw new Error(`Unknown resolution action: ${resolution.action}`);
  }
}
```

---

## Sync Progress Feedback

Keep users informed during sync:

```javascript
// syncUI.js
import syncManager from './syncManager.js';

class SyncUI {
  constructor() {
    this.container = document.getElementById('sync-status');
    this.setupListeners();
  }
  
  setupListeners() {
    syncManager.subscribe((event) => {
      switch (event.type) {
        case 'sync-start':
          this.showSyncing();
          break;
        case 'sync-complete':
          this.showComplete();
          break;
        case 'sync-error':
          this.showError(event.data);
          break;
        case 'sync-progress':
          this.showProgress(event.data);
          break;
      }
    });
  }
  
  showSyncing() {
    this.container.innerHTML = `
      <div class="sync-indicator syncing">
        <span class="sync-spinner"></span>
        <span class="sync-text">Syncing...</span>
      </div>
    `;
    this.container.hidden = false;
  }
  
  showComplete() {
    this.container.innerHTML = `
      <div class="sync-indicator complete">
        <span class="sync-icon">✓</span>
        <span class="sync-text">Up to date</span>
      </div>
    `;
    
    // Hide after 2 seconds
    setTimeout(() => {
      this.container.hidden = true;
    }, 2000);
  }
  
  showError(error) {
    this.container.innerHTML = `
      <div class="sync-indicator error">
        <span class="sync-icon">⚠️</span>
        <span class="sync-text">Sync failed</span>
        <button class="retry-btn">Retry</button>
      </div>
    `;
    
    this.container.querySelector('.retry-btn').onclick = () => {
      syncManager.sync();
    };
  }
  
  showProgress(data) {
    const { current, total, type } = data;
    const percent = Math.round((current / total) * 100);
    
    this.container.innerHTML = `
      <div class="sync-indicator syncing">
        <span class="sync-text">
          ${type === 'outbound' ? 'Sending' : 'Receiving'} ${current}/${total}
        </span>
        <div class="sync-progress">
          <div class="sync-progress-bar" style="width: ${percent}%"></div>
        </div>
      </div>
    `;
  }
}

export default new SyncUI();
```

```css
/* Sync indicator styles */
.sync-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  font-size: 0.875rem;
}

.sync-indicator.syncing {
  background: var(--info-bg, #cce5ff);
  color: var(--info-text, #004085);
}

.sync-indicator.complete {
  background: var(--success-bg, #d4edda);
  color: var(--success-text, #155724);
}

.sync-indicator.error {
  background: var(--danger-bg, #f8d7da);
  color: var(--danger-text, #721c24);
}

.sync-spinner {
  width: 1rem;
  height: 1rem;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.sync-progress {
  width: 100px;
  height: 4px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 2px;
  overflow: hidden;
}

.sync-progress-bar {
  height: 100%;
  background: currentColor;
  transition: width 0.3s ease;
}

.retry-btn {
  margin-left: auto;
  padding: 0.25rem 0.5rem;
  background: transparent;
  border: 1px solid currentColor;
  border-radius: 0.25rem;
  cursor: pointer;
  color: inherit;
}
```

---

## Complete Sync Flow

### Putting It All Together

```javascript
// app.js - Initialize sync system
import connectionManager from './connectionManager.js';
import syncManager from './syncManager.js';
import syncUI from './syncUI.js';
import messageQueue from './messageQueue.js';

class ChatApp {
  async initialize() {
    // 1. Check for pending messages from last session
    const stats = await messageQueue.getQueueStats();
    console.log(`Found ${stats.pending} pending messages from last session`);
    
    // 2. Set up connection monitoring
    connectionManager.startPeriodicCheck(30000);
    
    // 3. Register for background sync if supported
    await this.setupBackgroundSync();
    
    // 4. Initial sync if online
    if (connectionManager.isOnline) {
      syncManager.sync();
    }
    
    console.log('Chat app initialized');
  }
  
  async setupBackgroundSync() {
    if (!('serviceWorker' in navigator)) return;
    
    const registration = await navigator.serviceWorker.ready;
    
    if ('sync' in registration) {
      // Register for sync when messages are queued
      messageQueue.subscribe(async (event, data) => {
        if (event === 'enqueue') {
          await registration.sync.register('sync-messages');
        }
      });
      
      console.log('Background Sync enabled');
    } else {
      console.log('Background Sync not available, using fallback');
    }
  }
}

const app = new ChatApp();
app.initialize();
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Verify connectivity before sync | Trust `navigator.onLine` alone |
| Implement fallback for Background Sync | Assume all browsers support it |
| Show sync progress to users | Sync silently with no feedback |
| Handle conflicts gracefully | Overwrite data without checking |
| Batch small requests | Sync each message with separate API calls |
| Implement exponential backoff | Retry immediately in a loop |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Syncing while still offline | Verify connection before starting |
| No conflict resolution strategy | Plan for conflicts upfront |
| Large sync payloads timeout | Paginate or chunk large syncs |
| Sync blocks UI | Run sync in background/worker |
| No retry on failure | Implement retry with backoff |
| Duplicate messages after sync | Use client IDs for deduplication |

---

## Hands-on Exercise

### Your Task

Implement a sync system that:

1. **Detects reconnection** using the online/offline events
2. **Syncs pending messages** when coming back online
3. **Shows sync progress** in the UI
4. **Handles basic conflicts** (server-wins strategy)

### Requirements

1. Connection manager that verifies actual connectivity
2. Sync triggered on reconnection and visibility change
3. Visual indicator showing sync status
4. At least one conflict resolution strategy

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `fetch` with a timeout to verify actual connectivity
- The `visibilitychange` event helps sync when user returns to tab
- Keep track of sync state to prevent concurrent syncs
- Use HTTP 409 status for conflict responses from server

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
// Minimal sync implementation
class SimpleSyncManager {
  constructor() {
    this.isSyncing = false;
    this.healthEndpoint = '/api/health';
    this.listeners = new Set();
    
    this.setupListeners();
  }
  
  setupListeners() {
    window.addEventListener('online', () => this.onConnectionChange(true));
    window.addEventListener('offline', () => this.onConnectionChange(false));
    
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this.checkAndSync();
      }
    });
  }
  
  onConnectionChange(online) {
    this.notify({ type: online ? 'online' : 'offline' });
    
    if (online) {
      this.checkAndSync();
    }
  }
  
  async checkAndSync() {
    // Verify connection
    try {
      const response = await fetch(this.healthEndpoint, {
        method: 'HEAD',
        cache: 'no-store'
      });
      
      if (response.ok) {
        await this.sync();
      }
    } catch {
      console.log('Not actually online');
    }
  }
  
  async sync() {
    if (this.isSyncing) return;
    
    this.isSyncing = true;
    this.notify({ type: 'sync-start' });
    
    try {
      // Get pending messages
      const pending = await messageQueue.getPending();
      
      for (let i = 0; i < pending.length; i++) {
        this.notify({
          type: 'sync-progress',
          data: { current: i + 1, total: pending.length }
        });
        
        try {
          const response = await fetch('/api/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pending[i])
          });
          
          if (response.status === 409) {
            // Conflict - server wins
            const server = await response.json();
            await chatDB.saveMessage(server.serverVersion);
          } else if (response.ok) {
            await messageQueue.dequeue(pending[i].id);
          }
        } catch (err) {
          console.error('Message sync failed:', err);
        }
      }
      
      this.notify({ type: 'sync-complete' });
      
    } catch (error) {
      this.notify({ type: 'sync-error', data: error });
    } finally {
      this.isSyncing = false;
    }
  }
  
  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notify(event) {
    this.listeners.forEach(cb => cb(event));
  }
}

// Usage
const syncManager = new SimpleSyncManager();

syncManager.subscribe((event) => {
  const indicator = document.getElementById('sync-indicator');
  
  switch (event.type) {
    case 'sync-start':
      indicator.textContent = '🔄 Syncing...';
      indicator.hidden = false;
      break;
    case 'sync-complete':
      indicator.textContent = '✅ Synced';
      setTimeout(() => indicator.hidden = true, 2000);
      break;
    case 'sync-error':
      indicator.textContent = '❌ Sync failed';
      break;
    case 'sync-progress':
      indicator.textContent = `🔄 ${event.data.current}/${event.data.total}`;
      break;
  }
});
```

</details>

---

## Summary

✅ `navigator.onLine` provides basic connectivity info but isn't always reliable  
✅ Verify actual connectivity with a server ping before syncing  
✅ Background Sync API works in Chromium browsers only—always have a fallback  
✅ Handle conflicts with a clear strategy (server-wins, client-wins, or manual)  
✅ Show sync progress to keep users informed  
✅ Trigger sync on reconnection, visibility change, and periodically  

**Next:** [Offline Indicators](./04-offline-indicators.md)

---

<!-- 
Sources Consulted:
- MDN Navigator.onLine: https://developer.mozilla.org/en-US/docs/Web/API/Navigator/onLine
- MDN Background Sync API: https://developer.mozilla.org/en-US/docs/Web/API/Background_Synchronization_API
- MDN online/offline events: https://developer.mozilla.org/en-US/docs/Web/API/Window/online_event
- web.dev Background Sync: https://web.dev/articles/background-sync
-->
