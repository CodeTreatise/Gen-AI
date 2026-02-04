---
title: "Conversation History Storage"
---

# Conversation History Storage

## Introduction

Chat history needs to persist across sessions, handle large conversations, and sync across devices. From simple localStorage to IndexedDB and server databases, the right storage strategy depends on your scale, privacy requirements, and offline needs.

In this lesson, we'll implement client-side and server-side storage for conversation history.

### What We'll Cover

- In-memory storage patterns
- localStorage for simple persistence
- IndexedDB for large histories
- Server-side storage strategies
- Hybrid approaches

### Prerequisites

- [Message Data Structures](./01-message-data-structures.md)
- Browser storage APIs
- Async/await patterns

---

## Storage Strategy Comparison

| Strategy | Capacity | Persistence | Speed | Offline | Use Case |
|----------|----------|-------------|-------|---------|----------|
| **Memory** | Unlimited | Session | Fastest | ❌ | Temporary/dev |
| **localStorage** | 5-10 MB | Permanent | Fast | ✅ | Small apps |
| **IndexedDB** | 50%+ disk | Permanent | Fast | ✅ | Large histories |
| **Server DB** | Unlimited | Permanent | Network | ❌ | Multi-device |
| **Hybrid** | Varies | Both | Optimized | ✅ | Production |

---

## In-Memory Storage

### Simple Store

```typescript
interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

class InMemoryStore {
  private conversations: Map<string, Conversation> = new Map();
  private activeConversationId: string | null = null;
  
  createConversation(title?: string): Conversation {
    const conversation: Conversation = {
      id: generateId(),
      title: title || 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    this.conversations.set(conversation.id, conversation);
    this.activeConversationId = conversation.id;
    return conversation;
  }
  
  getConversation(id: string): Conversation | undefined {
    return this.conversations.get(id);
  }
  
  getActiveConversation(): Conversation | undefined {
    if (!this.activeConversationId) return undefined;
    return this.conversations.get(this.activeConversationId);
  }
  
  addMessage(conversationId: string, message: Message): void {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) throw new Error('Conversation not found');
    
    conversation.messages.push(message);
    conversation.updatedAt = new Date();
  }
  
  updateMessage(conversationId: string, messageId: string, updates: Partial<Message>): void {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) return;
    
    const index = conversation.messages.findIndex(m => m.id === messageId);
    if (index !== -1) {
      conversation.messages[index] = {
        ...conversation.messages[index],
        ...updates,
        updatedAt: new Date()
      };
      conversation.updatedAt = new Date();
    }
  }
  
  listConversations(): Conversation[] {
    return Array.from(this.conversations.values())
      .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
  }
  
  deleteConversation(id: string): void {
    this.conversations.delete(id);
    if (this.activeConversationId === id) {
      this.activeConversationId = null;
    }
  }
}
```

### React Hook Wrapper

```typescript
import { useState, useCallback, useMemo } from 'react';

function useConversationStore() {
  const [store] = useState(() => new InMemoryStore());
  const [, forceUpdate] = useState({});
  
  const refresh = useCallback(() => forceUpdate({}), []);
  
  const createConversation = useCallback((title?: string) => {
    const conv = store.createConversation(title);
    refresh();
    return conv;
  }, [store, refresh]);
  
  const addMessage = useCallback((conversationId: string, message: Message) => {
    store.addMessage(conversationId, message);
    refresh();
  }, [store, refresh]);
  
  const conversations = useMemo(() => store.listConversations(), [store]);
  const activeConversation = useMemo(() => store.getActiveConversation(), [store]);
  
  return {
    conversations,
    activeConversation,
    createConversation,
    addMessage,
    // ... other methods
  };
}
```

---

## localStorage Storage

### Basic Implementation

```typescript
const STORAGE_KEY = 'chat_conversations';
const STORAGE_VERSION = 1;

interface StorageData {
  version: number;
  conversations: SerializedConversation[];
  activeConversationId: string | null;
}

interface SerializedConversation {
  id: string;
  title: string;
  messages: SerializedMessage[];
  createdAt: string;
  updatedAt: string;
}

class LocalStorageStore {
  private cache: Map<string, Conversation> = new Map();
  private activeConversationId: string | null = null;
  
  constructor() {
    this.load();
  }
  
  private load(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      
      const data: StorageData = JSON.parse(raw);
      
      // Handle version migrations
      if (data.version !== STORAGE_VERSION) {
        this.migrate(data);
        return;
      }
      
      for (const conv of data.conversations) {
        this.cache.set(conv.id, this.deserializeConversation(conv));
      }
      
      this.activeConversationId = data.activeConversationId;
    } catch (error) {
      console.error('Failed to load conversations:', error);
      // Clear corrupted data
      localStorage.removeItem(STORAGE_KEY);
    }
  }
  
  private save(): void {
    const data: StorageData = {
      version: STORAGE_VERSION,
      conversations: Array.from(this.cache.values()).map(
        c => this.serializeConversation(c)
      ),
      activeConversationId: this.activeConversationId
    };
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        this.handleQuotaExceeded();
      } else {
        throw error;
      }
    }
  }
  
  private handleQuotaExceeded(): void {
    // Remove oldest conversations
    const sorted = this.listConversations();
    const toRemove = sorted.slice(Math.floor(sorted.length / 2));
    
    for (const conv of toRemove) {
      this.cache.delete(conv.id);
    }
    
    console.warn(`Removed ${toRemove.length} old conversations due to storage limit`);
    this.save();
  }
  
  private serializeConversation(conv: Conversation): SerializedConversation {
    return {
      ...conv,
      createdAt: conv.createdAt.toISOString(),
      updatedAt: conv.updatedAt.toISOString(),
      messages: conv.messages.map(m => ({
        ...m,
        createdAt: m.createdAt.toISOString(),
        updatedAt: m.updatedAt?.toISOString()
      }))
    };
  }
  
  private deserializeConversation(data: SerializedConversation): Conversation {
    return {
      ...data,
      createdAt: new Date(data.createdAt),
      updatedAt: new Date(data.updatedAt),
      messages: data.messages.map(m => ({
        ...m,
        createdAt: new Date(m.createdAt),
        updatedAt: m.updatedAt ? new Date(m.updatedAt) : undefined
      }))
    };
  }
  
  // Same interface as InMemoryStore, but calls save() after mutations
  addMessage(conversationId: string, message: Message): void {
    const conversation = this.cache.get(conversationId);
    if (!conversation) throw new Error('Conversation not found');
    
    conversation.messages.push(message);
    conversation.updatedAt = new Date();
    this.save();
  }
  
  // ... other methods with save() calls
}
```

### Debounced Saving

```typescript
import { debounce } from 'lodash-es';

class DebouncedLocalStorageStore extends LocalStorageStore {
  private debouncedSave = debounce(() => {
    super.save();
  }, 500);
  
  protected save(): void {
    this.debouncedSave();
  }
  
  // Force immediate save (e.g., on page unload)
  flush(): void {
    this.debouncedSave.flush();
  }
}

// Save before page close
window.addEventListener('beforeunload', () => {
  store.flush();
});
```

---

## IndexedDB Storage

### Database Setup

```typescript
const DB_NAME = 'chat_db';
const DB_VERSION = 1;

interface ChatDB {
  conversations: Conversation;
  messages: Message & { conversationId: string };
}

class IndexedDBStore {
  private db: IDBDatabase | null = null;
  
  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      
      request.onerror = () => reject(request.error);
      
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Conversations store
        if (!db.objectStoreNames.contains('conversations')) {
          const convStore = db.createObjectStore('conversations', { keyPath: 'id' });
          convStore.createIndex('updatedAt', 'updatedAt');
        }
        
        // Messages store (separate for scalability)
        if (!db.objectStoreNames.contains('messages')) {
          const msgStore = db.createObjectStore('messages', { keyPath: 'id' });
          msgStore.createIndex('conversationId', 'conversationId');
          msgStore.createIndex('createdAt', 'createdAt');
        }
      };
    });
  }
  
  private getStore(name: 'conversations' | 'messages', mode: IDBTransactionMode = 'readonly'): IDBObjectStore {
    if (!this.db) throw new Error('Database not initialized');
    const tx = this.db.transaction(name, mode);
    return tx.objectStore(name);
  }
  
  async createConversation(title?: string): Promise<Conversation> {
    const conversation: Conversation = {
      id: generateId(),
      title: title || 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    return new Promise((resolve, reject) => {
      const store = this.getStore('conversations', 'readwrite');
      const request = store.add(conversation);
      request.onsuccess = () => resolve(conversation);
      request.onerror = () => reject(request.error);
    });
  }
  
  async getConversation(id: string): Promise<Conversation | undefined> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('conversations');
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  
  async addMessage(conversationId: string, message: Message): Promise<void> {
    const messageWithConv = { ...message, conversationId };
    
    return new Promise((resolve, reject) => {
      if (!this.db) return reject(new Error('DB not initialized'));
      
      const tx = this.db.transaction(['messages', 'conversations'], 'readwrite');
      
      // Add message
      tx.objectStore('messages').add(messageWithConv);
      
      // Update conversation timestamp
      const convStore = tx.objectStore('conversations');
      const getReq = convStore.get(conversationId);
      
      getReq.onsuccess = () => {
        const conv = getReq.result;
        if (conv) {
          conv.updatedAt = new Date();
          convStore.put(conv);
        }
      };
      
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
  
  async getMessages(conversationId: string): Promise<Message[]> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('messages');
      const index = store.index('conversationId');
      const request = index.getAll(conversationId);
      
      request.onsuccess = () => {
        const messages = request.result.sort(
          (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
        resolve(messages);
      };
      request.onerror = () => reject(request.error);
    });
  }
  
  async listConversations(limit = 50): Promise<Conversation[]> {
    return new Promise((resolve, reject) => {
      const store = this.getStore('conversations');
      const index = store.index('updatedAt');
      const conversations: Conversation[] = [];
      
      // Iterate in reverse order (newest first)
      const request = index.openCursor(null, 'prev');
      
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        
        if (cursor && conversations.length < limit) {
          conversations.push(cursor.value);
          cursor.continue();
        } else {
          resolve(conversations);
        }
      };
      
      request.onerror = () => reject(request.error);
    });
  }
  
  async deleteConversation(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) return reject(new Error('DB not initialized'));
      
      const tx = this.db.transaction(['conversations', 'messages'], 'readwrite');
      
      // Delete conversation
      tx.objectStore('conversations').delete(id);
      
      // Delete all messages in conversation
      const msgStore = tx.objectStore('messages');
      const index = msgStore.index('conversationId');
      const request = index.openCursor(IDBKeyRange.only(id));
      
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
      
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}
```

### React Hook for IndexedDB

```typescript
function useIndexedDBStore() {
  const [store] = useState(() => new IndexedDBStore());
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    store.init()
      .then(() => setIsReady(true))
      .catch(setError);
  }, [store]);
  
  return { store, isReady, error };
}
```

---

## Server-Side Storage

### API Design

```typescript
// API endpoints
interface ConversationAPI {
  // Conversations
  'GET /conversations': { response: Conversation[] };
  'POST /conversations': { body: { title?: string }; response: Conversation };
  'GET /conversations/:id': { response: Conversation };
  'DELETE /conversations/:id': { response: void };
  
  // Messages
  'GET /conversations/:id/messages': { response: Message[] };
  'POST /conversations/:id/messages': { body: Message; response: Message };
  'PATCH /conversations/:id/messages/:messageId': { 
    body: Partial<Message>; 
    response: Message 
  };
}
```

### Client Implementation

```typescript
class APIStore {
  private baseUrl: string;
  private token: string;
  
  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }
  
  private async fetch<T>(
    path: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.token}`,
        ...options.headers
      }
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  }
  
  async listConversations(): Promise<Conversation[]> {
    return this.fetch('/conversations');
  }
  
  async createConversation(title?: string): Promise<Conversation> {
    return this.fetch('/conversations', {
      method: 'POST',
      body: JSON.stringify({ title })
    });
  }
  
  async getMessages(conversationId: string): Promise<Message[]> {
    return this.fetch(`/conversations/${conversationId}/messages`);
  }
  
  async addMessage(conversationId: string, message: Message): Promise<Message> {
    return this.fetch(`/conversations/${conversationId}/messages`, {
      method: 'POST',
      body: JSON.stringify(message)
    });
  }
}
```

### Server Implementation (Node.js/Express)

```typescript
import express from 'express';
import { PrismaClient } from '@prisma/client';

const app = express();
const prisma = new PrismaClient();

app.get('/conversations', async (req, res) => {
  const userId = req.user.id;  // From auth middleware
  
  const conversations = await prisma.conversation.findMany({
    where: { userId },
    orderBy: { updatedAt: 'desc' },
    take: 50
  });
  
  res.json(conversations);
});

app.post('/conversations/:id/messages', async (req, res) => {
  const { id } = req.params;
  const { role, content, ...rest } = req.body;
  
  const message = await prisma.message.create({
    data: {
      conversationId: id,
      role,
      content,
      ...rest
    }
  });
  
  // Update conversation timestamp
  await prisma.conversation.update({
    where: { id },
    data: { updatedAt: new Date() }
  });
  
  res.json(message);
});
```

---

## Hybrid Storage Pattern

### Sync Strategy

```typescript
class HybridStore {
  private local: IndexedDBStore;
  private remote: APIStore;
  private syncQueue: SyncOperation[] = [];
  private isSyncing = false;
  
  constructor(local: IndexedDBStore, remote: APIStore) {
    this.local = local;
    this.remote = remote;
    
    // Sync when online
    window.addEventListener('online', () => this.sync());
  }
  
  async addMessage(conversationId: string, message: Message): Promise<void> {
    // 1. Save locally immediately
    await this.local.addMessage(conversationId, message);
    
    // 2. Queue for remote sync
    this.syncQueue.push({
      type: 'addMessage',
      conversationId,
      data: message,
      timestamp: Date.now()
    });
    
    // 3. Try to sync if online
    if (navigator.onLine) {
      this.sync();
    }
  }
  
  async sync(): Promise<void> {
    if (this.isSyncing || this.syncQueue.length === 0) return;
    
    this.isSyncing = true;
    
    try {
      while (this.syncQueue.length > 0) {
        const operation = this.syncQueue[0];
        
        switch (operation.type) {
          case 'addMessage':
            await this.remote.addMessage(
              operation.conversationId,
              operation.data as Message
            );
            break;
          // Handle other operation types
        }
        
        // Remove from queue after success
        this.syncQueue.shift();
      }
    } catch (error) {
      console.error('Sync failed:', error);
      // Operations remain in queue for retry
    } finally {
      this.isSyncing = false;
    }
  }
  
  async loadConversation(id: string): Promise<Conversation> {
    // Try local first
    const local = await this.local.getConversation(id);
    
    if (navigator.onLine) {
      try {
        // Fetch from remote and merge
        const remote = await this.remote.getConversation(id);
        const merged = this.mergeConversations(local, remote);
        await this.local.saveConversation(merged);
        return merged;
      } catch {
        // Fall back to local
      }
    }
    
    if (local) return local;
    throw new Error('Conversation not found');
  }
  
  private mergeConversations(
    local: Conversation | undefined, 
    remote: Conversation
  ): Conversation {
    if (!local) return remote;
    
    // Merge messages by ID, keeping newer versions
    const messageMap = new Map<string, Message>();
    
    for (const msg of [...local.messages, ...remote.messages]) {
      const existing = messageMap.get(msg.id);
      if (!existing || new Date(msg.updatedAt || msg.createdAt) > 
          new Date(existing.updatedAt || existing.createdAt)) {
        messageMap.set(msg.id, msg);
      }
    }
    
    return {
      ...remote,
      messages: Array.from(messageMap.values())
        .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
    };
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use IndexedDB for large data | Store MBs in localStorage |
| Debounce frequent saves | Save on every keystroke |
| Handle quota exceeded | Assume unlimited storage |
| Serialize dates properly | Store Date objects directly |
| Queue operations offline | Drop data when offline |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| localStorage quota hit | Switch to IndexedDB |
| Sync conflicts lose data | Use merge strategies |
| Blocking on DB init | Show loading state |
| No migration handling | Version your schema |
| Unhandled offline | Queue operations |

---

## Hands-on Exercise

### Your Task

Build a storage layer that:
1. Uses localStorage for small data
2. Falls back to IndexedDB for large histories
3. Syncs to server when online
4. Works offline

### Requirements

1. Implement size-based storage selection
2. Add offline queue for operations
3. Handle sync conflicts
4. Persist queue across page reloads

<details>
<summary>💡 Hints (click to expand)</summary>

- Check `new Blob([JSON.stringify(data)]).size` for size
- Use `navigator.onLine` for connectivity
- Store sync queue in localStorage (it's small)
- Use timestamps for conflict resolution

</details>

---

## Summary

✅ **In-memory** is fastest but temporary  
✅ **localStorage** works for small data  
✅ **IndexedDB** handles large histories  
✅ **Server storage** enables multi-device  
✅ **Hybrid** combines offline + sync  
✅ **Merge strategies** resolve conflicts

---

## Further Reading

- [IndexedDB API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)
- [localStorage - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)
- [Offline-First Web Apps](https://web.dev/offline-cookbook/)

---

**Previous:** [Message Parts Structure](./02-message-parts-structure.md)  
**Next:** [OpenAI Conversations API](./04-openai-conversations-api.md)

<!-- 
Sources Consulted:
- MDN IndexedDB: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
- MDN localStorage: https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
- web.dev Offline: https://web.dev/offline-cookbook/
-->
