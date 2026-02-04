---
title: "Offline Message Queueing"
---

# Offline Message Queueing

## Introduction

When users send messages without internet connectivity, those messages need somewhere to go. Message queueing ensures that user actions aren't lost—messages are stored locally and transmitted when connectivity returns. This pattern is essential for a seamless offline experience.

This lesson covers building a robust message queue that persists across browser sessions, provides feedback to users, and integrates with the synchronization system.

### What We'll Cover

- Designing a message queue data structure
- Persisting the queue in IndexedDB
- Providing queue status feedback in the UI
- Managing queue operations (add, retry, remove)
- Handling queue overflow and limits

### Prerequisites

- [Caching Conversation History](./01-caching-conversation-history.md)
- Understanding of IndexedDB operations
- JavaScript async/await patterns

---

## Queue Architecture

A message queue for offline chat needs several key properties:

### Queue Requirements

| Requirement | Description |
|-------------|-------------|
| **Persistence** | Survives page reloads and browser restarts |
| **Order preservation** | Messages sent in sequence stay in sequence |
| **Status tracking** | Know which messages are pending, failed, or synced |
| **Retry support** | Failed messages can be retried automatically |
| **User visibility** | Users see which messages are queued |

### Queue State Machine

```
┌─────────────┐
│   Created   │
│  (pending)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Sending   │────▶│    Sent     │
│             │     │  (success)  │
└──────┬──────┘     └─────────────┘
       │
       │ (network error)
       ▼
┌─────────────┐     ┌─────────────┐
│   Failed    │────▶│   Sending   │ (retry)
│             │     │             │
└──────┬──────┘     └─────────────┘
       │
       │ (max retries)
       ▼
┌─────────────┐
│  Abandoned  │
│             │
└─────────────┘
```

---

## Queue Data Structure

### Message Schema

```javascript
const queuedMessage = {
  id: 'msg-uuid-123',           // Unique identifier
  conversationId: 'conv-456',   // Target conversation
  content: 'Hello!',            // Message content
  contentType: 'text',          // text, image, file, etc.
  
  // Queue metadata
  status: 'pending',            // pending, sending, failed, sent
  createdAt: 1704067200000,     // When user sent it
  attempts: 0,                  // Retry count
  lastAttempt: null,            // Last send attempt timestamp
  error: null,                  // Last error message
  
  // Optimistic UI data
  tempId: 'temp-1704067200000', // Temporary ID for UI
  optimisticData: {             // Data shown to user immediately
    senderName: 'You',
    avatar: '/avatars/me.png'
  }
};
```

### IndexedDB Schema for Queue

```javascript
// Extend the ChatDatabase from previous lesson
class ChatDatabase {
  async open() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('ChatApp', 2); // Increment version
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // ... existing stores ...
        
        // Message queue store
        if (!db.objectStoreNames.contains('messageQueue')) {
          const queueStore = db.createObjectStore('messageQueue', { 
            keyPath: 'id' 
          });
          
          queueStore.createIndex('status', 'status', { unique: false });
          queueStore.createIndex('conversationId', 'conversationId', { unique: false });
          queueStore.createIndex('createdAt', 'createdAt', { unique: false });
          
          // Compound index for processing order
          queueStore.createIndex('status_createdAt', 
            ['status', 'createdAt'], { unique: false });
        }
      };
      
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };
      
      request.onerror = () => reject(request.error);
    });
  }
}
```

---

## Queue Manager Implementation

### Core Queue Class

```javascript
// messageQueue.js
import chatDB from './db.js';

class MessageQueue {
  constructor() {
    this.listeners = new Set();
    this.processing = false;
    this.maxRetries = 3;
    this.retryDelays = [1000, 5000, 15000]; // Exponential backoff
  }
  
  // Generate unique message ID
  generateId() {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  // Add message to queue
  async enqueue(conversationId, content, contentType = 'text') {
    const message = {
      id: this.generateId(),
      conversationId,
      content,
      contentType,
      status: 'pending',
      createdAt: Date.now(),
      attempts: 0,
      lastAttempt: null,
      error: null,
      tempId: `temp-${Date.now()}`
    };
    
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readwrite');
      const store = tx.objectStore('messageQueue');
      
      const request = store.add(message);
      
      request.onsuccess = () => {
        this.notify('enqueue', message);
        resolve(message);
      };
      
      request.onerror = () => reject(request.error);
    });
  }
  
  // Get all pending messages
  async getPending() {
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readonly');
      const store = tx.objectStore('messageQueue');
      const index = store.index('status');
      
      const request = index.getAll('pending');
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Get queue count by status
  async getQueueStats() {
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readonly');
      const store = tx.objectStore('messageQueue');
      
      const stats = {
        pending: 0,
        sending: 0,
        failed: 0,
        total: 0
      };
      
      const request = store.openCursor();
      
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          stats[cursor.value.status] = (stats[cursor.value.status] || 0) + 1;
          stats.total++;
          cursor.continue();
        } else {
          resolve(stats);
        }
      };
      
      request.onerror = () => reject(request.error);
    });
  }
  
  // Update message status
  async updateStatus(id, status, error = null) {
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readwrite');
      const store = tx.objectStore('messageQueue');
      
      const getRequest = store.get(id);
      
      getRequest.onsuccess = () => {
        const message = getRequest.result;
        if (message) {
          message.status = status;
          message.lastAttempt = Date.now();
          if (error) message.error = error;
          if (status === 'sending') message.attempts++;
          
          store.put(message);
          this.notify('statusChange', message);
        }
      };
      
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
  
  // Remove message from queue (after successful send)
  async dequeue(id) {
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readwrite');
      const store = tx.objectStore('messageQueue');
      
      const request = store.delete(id);
      
      tx.oncomplete = () => {
        this.notify('dequeue', { id });
        resolve();
      };
      
      tx.onerror = () => reject(tx.error);
    });
  }
  
  // Event listener pattern
  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notify(event, data) {
    this.listeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (e) {
        console.error('Queue listener error:', e);
      }
    });
  }
}

export default new MessageQueue();
```

### Queue Processor

```javascript
// queueProcessor.js
import messageQueue from './messageQueue.js';

class QueueProcessor {
  constructor() {
    this.isProcessing = false;
    this.apiBase = '/api';
  }
  
  async processQueue() {
    if (this.isProcessing) {
      console.log('Queue already processing');
      return;
    }
    
    if (!navigator.onLine) {
      console.log('Offline, skipping queue processing');
      return;
    }
    
    this.isProcessing = true;
    
    try {
      const pending = await messageQueue.getPending();
      console.log(`Processing ${pending.length} queued messages`);
      
      // Process in order (FIFO)
      for (const message of pending.sort((a, b) => a.createdAt - b.createdAt)) {
        await this.processMessage(message);
      }
    } catch (error) {
      console.error('Queue processing error:', error);
    } finally {
      this.isProcessing = false;
    }
  }
  
  async processMessage(message) {
    // Check retry limit
    if (message.attempts >= messageQueue.maxRetries) {
      await messageQueue.updateStatus(message.id, 'abandoned', 'Max retries exceeded');
      return;
    }
    
    await messageQueue.updateStatus(message.id, 'sending');
    
    try {
      const response = await fetch(
        `${this.apiBase}/conversations/${message.conversationId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content: message.content,
            contentType: message.contentType,
            clientId: message.id // For deduplication on server
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const serverMessage = await response.json();
      
      // Save the real message to cache
      await chatDB.saveMessage({
        ...serverMessage,
        tempId: message.tempId // Link to optimistic message
      });
      
      // Remove from queue
      await messageQueue.dequeue(message.id);
      
      console.log(`Message ${message.id} sent successfully`);
      
    } catch (error) {
      console.error(`Failed to send message ${message.id}:`, error);
      
      await messageQueue.updateStatus(message.id, 'failed', error.message);
      
      // Schedule retry with exponential backoff
      const delay = messageQueue.retryDelays[message.attempts] || 30000;
      setTimeout(() => this.retryMessage(message.id), delay);
    }
  }
  
  async retryMessage(messageId) {
    const db = await chatDB.ensureOpen();
    
    const tx = db.transaction('messageQueue', 'readonly');
    const store = tx.objectStore('messageQueue');
    const request = store.get(messageId);
    
    request.onsuccess = async () => {
      const message = request.result;
      if (message && message.status === 'failed') {
        await messageQueue.updateStatus(messageId, 'pending');
        this.processQueue();
      }
    };
  }
  
  // Start processing when online
  startAutoProcessing() {
    // Process on reconnection
    window.addEventListener('online', () => {
      console.log('Back online, processing queue...');
      this.processQueue();
    });
    
    // Process on page load if online
    if (navigator.onLine) {
      this.processQueue();
    }
    
    // Periodic check (in case online event was missed)
    setInterval(() => {
      if (navigator.onLine && !this.isProcessing) {
        this.processQueue();
      }
    }, 30000);
  }
}

export default new QueueProcessor();
```

---

## Queue Status UI

### Visual Indicators

```html
<!-- Queue indicator in header -->
<header class="chat-header">
  <h1>Chat</h1>
  <div id="queue-indicator" class="queue-indicator hidden">
    <span class="queue-icon">📤</span>
    <span class="queue-count">0</span>
    <span class="queue-label">pending</span>
  </div>
</header>
```

```css
/* Queue indicator styles */
.queue-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  background: var(--warning-bg, #fff3cd);
  border-radius: 1rem;
  font-size: 0.875rem;
}

.queue-indicator.hidden {
  display: none;
}

.queue-indicator.processing {
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.queue-count {
  font-weight: 600;
  min-width: 1.5rem;
  text-align: center;
}

/* Message status indicators */
.message.pending .status-icon::after { content: '⏳'; }
.message.sending .status-icon::after { content: '📤'; }
.message.failed .status-icon::after { content: '❌'; }
.message.sent .status-icon::after { content: '✓'; }

.message.failed {
  opacity: 0.7;
  border-left: 3px solid var(--error-color, #dc3545);
}

.message.pending,
.message.sending {
  opacity: 0.85;
}
```

### Queue UI Controller

```javascript
// queueUI.js
import messageQueue from './messageQueue.js';

class QueueUI {
  constructor() {
    this.indicator = document.getElementById('queue-indicator');
    this.countEl = this.indicator?.querySelector('.queue-count');
    this.labelEl = this.indicator?.querySelector('.queue-label');
    
    this.setupListeners();
    this.updateIndicator();
  }
  
  setupListeners() {
    // Subscribe to queue changes
    messageQueue.subscribe((event, data) => {
      this.updateIndicator();
      this.updateMessageUI(event, data);
    });
  }
  
  async updateIndicator() {
    const stats = await messageQueue.getQueueStats();
    const pending = stats.pending + stats.sending + stats.failed;
    
    if (!this.indicator) return;
    
    if (pending === 0) {
      this.indicator.classList.add('hidden');
    } else {
      this.indicator.classList.remove('hidden');
      this.countEl.textContent = pending;
      this.labelEl.textContent = pending === 1 ? 'pending' : 'pending';
      
      if (stats.sending > 0) {
        this.indicator.classList.add('processing');
      } else {
        this.indicator.classList.remove('processing');
      }
    }
  }
  
  updateMessageUI(event, data) {
    if (event === 'enqueue') {
      this.showOptimisticMessage(data);
    } else if (event === 'statusChange') {
      this.updateMessageStatus(data);
    } else if (event === 'dequeue') {
      this.replaceWithServerMessage(data);
    }
  }
  
  showOptimisticMessage(message) {
    const container = document.querySelector(
      `.conversation[data-id="${message.conversationId}"] .message-list`
    );
    
    if (!container) return;
    
    const el = document.createElement('div');
    el.className = `message outgoing ${message.status}`;
    el.dataset.tempId = message.tempId;
    el.dataset.queueId = message.id;
    
    el.innerHTML = `
      <div class="message-content">${this.escapeHtml(message.content)}</div>
      <div class="message-meta">
        <span class="time">${this.formatTime(message.createdAt)}</span>
        <span class="status-icon"></span>
      </div>
    `;
    
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
  }
  
  updateMessageStatus(message) {
    const el = document.querySelector(`[data-queue-id="${message.id}"]`);
    if (el) {
      el.className = `message outgoing ${message.status}`;
      
      // Add retry button for failed messages
      if (message.status === 'failed') {
        this.addRetryButton(el, message);
      }
    }
  }
  
  addRetryButton(el, message) {
    // Remove existing retry button
    el.querySelector('.retry-btn')?.remove();
    
    const btn = document.createElement('button');
    btn.className = 'retry-btn';
    btn.textContent = 'Retry';
    btn.onclick = async () => {
      await messageQueue.updateStatus(message.id, 'pending');
      queueProcessor.processQueue();
    };
    
    el.appendChild(btn);
  }
  
  replaceWithServerMessage(data) {
    // Server message has been saved, UI will be updated
    // by the message subscription in the main chat UI
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}

export default new QueueUI();
```

---

## Queue Limits and Overflow

Prevent the queue from growing too large:

### Implementing Queue Limits

```javascript
class MessageQueue {
  constructor() {
    // ... existing code ...
    this.maxQueueSize = 100;       // Maximum messages in queue
    this.maxMessageSize = 10000;   // Maximum message content length
  }
  
  async enqueue(conversationId, content, contentType = 'text') {
    // Validate message size
    if (content.length > this.maxMessageSize) {
      throw new Error(`Message too large (max ${this.maxMessageSize} characters)`);
    }
    
    // Check queue size
    const stats = await this.getQueueStats();
    if (stats.total >= this.maxQueueSize) {
      throw new Error('Message queue is full. Please wait for messages to send.');
    }
    
    // ... rest of enqueue logic ...
  }
  
  async cleanupOldMessages(maxAge = 24 * 60 * 60 * 1000) {
    const db = await chatDB.ensureOpen();
    const cutoff = Date.now() - maxAge;
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readwrite');
      const store = tx.objectStore('messageQueue');
      const index = store.index('createdAt');
      
      let deletedCount = 0;
      const range = IDBKeyRange.upperBound(cutoff);
      
      const request = index.openCursor(range);
      
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          // Only delete failed or abandoned messages
          if (['failed', 'abandoned'].includes(cursor.value.status)) {
            cursor.delete();
            deletedCount++;
          }
          cursor.continue();
        }
      };
      
      tx.oncomplete = () => {
        if (deletedCount > 0) {
          this.notify('cleanup', { deletedCount });
        }
        resolve(deletedCount);
      };
      
      tx.onerror = () => reject(tx.error);
    });
  }
}
```

### User Feedback for Queue Issues

```javascript
// Handle queue errors in the UI
async function handleSendMessage(conversationId, content) {
  try {
    const message = await messageQueue.enqueue(conversationId, content);
    // Show optimistic message
    queueUI.showOptimisticMessage(message);
    
    // Try to send immediately if online
    if (navigator.onLine) {
      queueProcessor.processQueue();
    }
    
  } catch (error) {
    if (error.message.includes('queue is full')) {
      showNotification({
        type: 'warning',
        title: 'Queue Full',
        message: 'Too many pending messages. Please wait for some to send.',
        action: {
          label: 'View Queue',
          handler: () => showQueueDetails()
        }
      });
    } else if (error.message.includes('too large')) {
      showNotification({
        type: 'error',
        title: 'Message Too Large',
        message: 'Please shorten your message and try again.'
      });
    } else {
      showNotification({
        type: 'error',
        title: 'Send Failed',
        message: error.message
      });
    }
  }
}
```

---

## Queue Details View

Provide a detailed view of the queue for debugging and user control:

```javascript
// queueDetailsUI.js
class QueueDetailsUI {
  constructor() {
    this.modal = null;
  }
  
  async show() {
    const messages = await this.getQueuedMessages();
    this.modal = this.createModal(messages);
    document.body.appendChild(this.modal);
  }
  
  async getQueuedMessages() {
    const db = await chatDB.ensureOpen();
    
    return new Promise((resolve, reject) => {
      const tx = db.transaction('messageQueue', 'readonly');
      const store = tx.objectStore('messageQueue');
      const request = store.getAll();
      
      request.onsuccess = () => {
        const messages = request.result.sort((a, b) => b.createdAt - a.createdAt);
        resolve(messages);
      };
      
      request.onerror = () => reject(request.error);
    });
  }
  
  createModal(messages) {
    const modal = document.createElement('div');
    modal.className = 'queue-modal';
    
    modal.innerHTML = `
      <div class="queue-modal-content">
        <header class="queue-modal-header">
          <h2>Message Queue</h2>
          <button class="close-btn" aria-label="Close">×</button>
        </header>
        
        <div class="queue-modal-body">
          ${messages.length === 0 
            ? '<p class="empty-message">No messages in queue</p>'
            : messages.map(m => this.renderMessage(m)).join('')
          }
        </div>
        
        <footer class="queue-modal-footer">
          <button class="retry-all-btn">Retry All Failed</button>
          <button class="clear-failed-btn">Clear Failed</button>
        </footer>
      </div>
    `;
    
    // Event handlers
    modal.querySelector('.close-btn').onclick = () => this.hide();
    modal.querySelector('.retry-all-btn').onclick = () => this.retryAllFailed();
    modal.querySelector('.clear-failed-btn').onclick = () => this.clearFailed();
    
    return modal;
  }
  
  renderMessage(message) {
    const statusColors = {
      pending: '#ffc107',
      sending: '#17a2b8',
      failed: '#dc3545',
      abandoned: '#6c757d'
    };
    
    return `
      <div class="queue-item" data-id="${message.id}">
        <div class="queue-item-status" style="color: ${statusColors[message.status]}">
          ${message.status.toUpperCase()}
        </div>
        <div class="queue-item-content">
          <p class="queue-item-text">${this.truncate(message.content, 100)}</p>
          <div class="queue-item-meta">
            <span>Created: ${this.formatTime(message.createdAt)}</span>
            <span>Attempts: ${message.attempts}</span>
            ${message.error ? `<span class="error">Error: ${message.error}</span>` : ''}
          </div>
        </div>
        <div class="queue-item-actions">
          ${message.status === 'failed' ? `
            <button class="retry-btn" data-id="${message.id}">Retry</button>
            <button class="delete-btn" data-id="${message.id}">Delete</button>
          ` : ''}
        </div>
      </div>
    `;
  }
  
  truncate(text, length) {
    if (text.length <= length) return text;
    return text.substring(0, length) + '...';
  }
  
  formatTime(timestamp) {
    return new Date(timestamp).toLocaleString();
  }
  
  hide() {
    this.modal?.remove();
    this.modal = null;
  }
  
  async retryAllFailed() {
    const messages = await this.getQueuedMessages();
    const failed = messages.filter(m => m.status === 'failed');
    
    for (const message of failed) {
      await messageQueue.updateStatus(message.id, 'pending');
    }
    
    queueProcessor.processQueue();
    this.hide();
  }
  
  async clearFailed() {
    const messages = await this.getQueuedMessages();
    const failed = messages.filter(m => 
      m.status === 'failed' || m.status === 'abandoned'
    );
    
    for (const message of failed) {
      await messageQueue.dequeue(message.id);
    }
    
    this.hide();
  }
}

export default new QueueDetailsUI();
```

---

## Integrating with Chat Input

### Complete Send Flow

```javascript
// chatInput.js
import messageQueue from './messageQueue.js';
import queueProcessor from './queueProcessor.js';

class ChatInput {
  constructor(conversationId) {
    this.conversationId = conversationId;
    this.form = document.querySelector('.chat-input-form');
    this.input = this.form.querySelector('input[type="text"]');
    this.sendBtn = this.form.querySelector('button[type="submit"]');
    
    this.setupEventListeners();
  }
  
  setupEventListeners() {
    this.form.addEventListener('submit', (e) => {
      e.preventDefault();
      this.send();
    });
    
    // Character count
    this.input.addEventListener('input', () => {
      this.updateCharCount();
    });
  }
  
  async send() {
    const content = this.input.value.trim();
    
    if (!content) return;
    
    // Disable while processing
    this.setLoading(true);
    
    try {
      // Add to queue
      const message = await messageQueue.enqueue(
        this.conversationId,
        content,
        'text'
      );
      
      // Clear input
      this.input.value = '';
      this.updateCharCount();
      
      // Process immediately if online
      if (navigator.onLine) {
        queueProcessor.processQueue();
      } else {
        this.showOfflineNotice();
      }
      
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.setLoading(false);
      this.input.focus();
    }
  }
  
  setLoading(loading) {
    this.input.disabled = loading;
    this.sendBtn.disabled = loading;
    this.sendBtn.textContent = loading ? 'Sending...' : 'Send';
  }
  
  updateCharCount() {
    const count = this.input.value.length;
    const max = 10000; // Match queue limit
    const counter = this.form.querySelector('.char-count');
    
    if (counter) {
      counter.textContent = `${count}/${max}`;
      counter.classList.toggle('warning', count > max * 0.9);
      counter.classList.toggle('error', count >= max);
    }
  }
  
  showOfflineNotice() {
    const notice = document.createElement('div');
    notice.className = 'offline-notice';
    notice.textContent = 'You\'re offline. Message will be sent when connected.';
    
    this.form.appendChild(notice);
    
    setTimeout(() => notice.remove(), 5000);
  }
  
  showError(message) {
    const error = document.createElement('div');
    error.className = 'input-error';
    error.textContent = message;
    
    this.form.appendChild(error);
    
    setTimeout(() => error.remove(), 5000);
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show optimistic UI immediately | Wait for server confirmation to show message |
| Preserve queue across sessions | Store queue only in memory |
| Implement retry with backoff | Retry immediately in a tight loop |
| Set reasonable queue limits | Allow unlimited queue growth |
| Provide clear status feedback | Leave users guessing about message state |
| Allow manual retry for failures | Auto-delete failed messages |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Processing queue while offline | Check `navigator.onLine` before processing |
| Not handling duplicate sends | Use client ID for server-side deduplication |
| Ignoring queue on page load | Start queue processor on app initialization |
| Large messages blocking queue | Validate message size before queuing |
| No way to cancel queued messages | Provide delete option for pending messages |
| Queue processing race conditions | Use `isProcessing` flag to prevent concurrency |

---

## Hands-on Exercise

### Your Task

Build a message queue system with:

1. **Queue Manager** that persists messages to IndexedDB
2. **Queue Processor** that sends messages when online
3. **Status UI** showing pending message count
4. **Retry mechanism** for failed messages

### Requirements

1. Messages survive page refresh
2. Queue shows count in the UI header
3. Failed messages show retry button
4. Messages process in order (FIFO)
5. Maximum 3 retry attempts before abandoning

<details>
<summary>💡 Hints (click to expand)</summary>

- Use IndexedDB indexes for efficient status queries
- Process messages sequentially with `for...of`, not `forEach`
- Clone messages before modifying to avoid transaction issues
- Use event listeners to keep UI in sync with queue state

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the complete implementation in the sections above. Key files:
- `messageQueue.js` - Core queue with IndexedDB
- `queueProcessor.js` - Send logic with retry
- `queueUI.js` - Visual indicators
- `chatInput.js` - Integration with send form

**Quick test setup:**

```javascript
// Initialize on page load
import messageQueue from './messageQueue.js';
import queueProcessor from './queueProcessor.js';
import queueUI from './queueUI.js';

// Start auto-processing
queueProcessor.startAutoProcessing();

// Test enqueue
document.querySelector('#test-btn').onclick = async () => {
  await messageQueue.enqueue('conv-1', 'Test message ' + Date.now());
  console.log('Stats:', await messageQueue.getQueueStats());
};
```

</details>

---

## Summary

✅ Message queues store user actions until connectivity returns  
✅ IndexedDB provides persistent storage that survives browser sessions  
✅ Status tracking (pending → sending → sent/failed) enables clear user feedback  
✅ Retry with exponential backoff handles transient network failures  
✅ Queue limits prevent unbounded growth  
✅ Optimistic UI shows messages immediately for better UX  

**Next:** [Sync on Reconnection](./03-sync-on-reconnection.md)

---

<!-- 
Sources Consulted:
- MDN IndexedDB API: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
- MDN Navigator.onLine: https://developer.mozilla.org/en-US/docs/Web/API/Navigator/onLine
- web.dev Offline First: https://web.dev/articles/offline-cookbook
-->
