---
title: "Offline Capabilities with Service Workers"
---

# Offline Capabilities with Service Workers

## Introduction

One of the most powerful features of Service Workers is enabling **offline functionality**. By intercepting fetch requests and serving cached responses, your web application can work without a network connection—just like a native app.

This lesson covers how to intercept requests, create offline fallback pages, implement offline-first strategies, and sync data when connectivity returns.

### What We'll Cover

- Intercepting fetch requests
- Offline fallback pages
- Offline-first strategies
- Sync when online

### Prerequisites

- Understanding of Service Worker lifecycle
- Familiarity with the Fetch API
- Basic knowledge of Promises

---

## Intercepting Fetch Requests

The `fetch` event fires for every network request made by controlled pages—HTML, CSS, JavaScript, images, API calls, everything.

### Basic Fetch Interception

```javascript
// service-worker.js
self.addEventListener('fetch', (event) => {
  console.log('Intercepted:', event.request.url);
  
  // Let the request go through normally
  event.respondWith(fetch(event.request));
});
```

### The FetchEvent Object

```javascript
self.addEventListener('fetch', (event) => {
  const request = event.request;
  
  console.log('URL:', request.url);
  console.log('Method:', request.method);       // GET, POST, etc.
  console.log('Mode:', request.mode);           // navigate, cors, no-cors, same-origin
  console.log('Destination:', request.destination); // document, script, style, image, etc.
  console.log('Headers:', [...request.headers]);
});
```

### Filtering Requests

Not all requests should be handled the same way:

```javascript
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Only handle same-origin requests
  if (url.origin !== location.origin) {
    return; // Let browser handle cross-origin requests
  }
  
  // Only handle GET requests
  if (event.request.method !== 'GET') {
    return;
  }
  
  // Skip API requests (handle differently)
  if (url.pathname.startsWith('/api/')) {
    return;
  }
  
  // Handle static assets
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
```

### Modifying Responses

You can modify responses before returning them:

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request).then(response => {
      // Clone the response (responses can only be used once)
      const modifiedResponse = response.clone();
      
      // Create new response with custom headers
      return new Response(modifiedResponse.body, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          ...Object.fromEntries(response.headers),
          'X-Custom-Header': 'Added by Service Worker'
        }
      });
    })
  );
});
```

---

## Offline Fallback Pages

The simplest offline strategy is showing a fallback page when the network is unavailable.

### Basic Offline Page

```javascript
const CACHE_NAME = 'offline-cache-v1';
const OFFLINE_URL = '/offline.html';

// Cache offline page during install
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.add(OFFLINE_URL))
  );
});

// Serve offline page when network fails
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => caches.match(OFFLINE_URL))
    );
  }
});
```

### Rich Offline Experience

```javascript
// service-worker.js
const CACHE_NAME = 'app-cache-v1';
const OFFLINE_ASSETS = [
  '/offline.html',
  '/offline.css',
  '/offline.js',
  '/images/offline-icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(OFFLINE_ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  
  // Navigation requests (HTML pages)
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .catch(() => caches.match('/offline.html'))
    );
    return;
  }
  
  // Image requests
  if (request.destination === 'image') {
    event.respondWith(
      fetch(request)
        .catch(() => caches.match('/images/offline-icon.svg'))
    );
    return;
  }
  
  // Other requests - try cache first
  event.respondWith(
    caches.match(request)
      .then(response => response || fetch(request))
  );
});
```

### Offline Page with Cached Content List

```html
<!-- offline.html -->
<!DOCTYPE html>
<html>
<head>
  <title>Offline</title>
  <link rel="stylesheet" href="/offline.css">
</head>
<body>
  <h1>You're Offline</h1>
  <p>But you can still access these cached pages:</p>
  <ul id="cached-pages"></ul>
  
  <script>
    // List cached pages
    caches.open('app-cache-v1').then(cache => {
      cache.keys().then(requests => {
        const list = document.getElementById('cached-pages');
        requests
          .filter(req => req.url.endsWith('.html'))
          .forEach(req => {
            const url = new URL(req.url);
            const li = document.createElement('li');
            li.innerHTML = `<a href="${url.pathname}">${url.pathname}</a>`;
            list.appendChild(li);
          });
      });
    });
  </script>
</body>
</html>
```

---

## Offline-First Strategies

**Offline-first** means serving cached content by default, then updating from the network. This provides instant loading and resilience.

### Cache-First with Network Update

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.open('dynamic-cache').then(cache => {
      return cache.match(event.request).then(cachedResponse => {
        // Start network request in background
        const networkFetch = fetch(event.request).then(networkResponse => {
          // Update cache with fresh response
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
        
        // Return cached response immediately, or wait for network
        return cachedResponse || networkFetch;
      });
    })
  );
});
```

### Stale-While-Revalidate

Serve cached content immediately while fetching updates in the background:

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.open('swr-cache').then(cache => {
      return cache.match(event.request).then(cachedResponse => {
        // Always fetch from network to update cache
        const fetchPromise = fetch(event.request).then(networkResponse => {
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
        
        // Return cached response if available, otherwise wait for network
        return cachedResponse || fetchPromise;
      });
    })
  );
});
```

### Network-First with Cache Fallback

Best for dynamic content that should be fresh when possible:

```javascript
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache successful responses
        const responseClone = response.clone();
        caches.open('network-first-cache').then(cache => {
          cache.put(event.request, responseClone);
        });
        return response;
      })
      .catch(() => {
        // Network failed, try cache
        return caches.match(event.request);
      })
  );
});
```

### Strategy Selection by Request Type

```javascript
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // API requests: Network first
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(event.request));
    return;
  }
  
  // Static assets: Cache first
  if (url.pathname.match(/\.(js|css|png|jpg|svg|woff2)$/)) {
    event.respondWith(cacheFirst(event.request));
    return;
  }
  
  // HTML pages: Stale-while-revalidate
  if (event.request.mode === 'navigate') {
    event.respondWith(staleWhileRevalidate(event.request));
    return;
  }
  
  // Default: Network with cache fallback
  event.respondWith(networkFirst(event.request));
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  return cached || fetch(request);
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    const cache = await caches.open('dynamic');
    cache.put(request, response.clone());
    return response;
  } catch {
    return caches.match(request);
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open('pages');
  const cached = await cache.match(request);
  
  const fetchPromise = fetch(request).then(response => {
    cache.put(request, response.clone());
    return response;
  });
  
  return cached || fetchPromise;
}
```

---

## Sync When Online

When offline, queue actions and sync when connectivity returns.

### Detecting Online/Offline State

```javascript
// In the page
window.addEventListener('online', () => {
  console.log('Back online!');
  syncPendingData();
});

window.addEventListener('offline', () => {
  console.log('Gone offline');
  showOfflineIndicator();
});

// Check current state
if (navigator.onLine) {
  console.log('Currently online');
} else {
  console.log('Currently offline');
}
```

### Queuing Offline Actions

```javascript
// page.js - Queue actions in IndexedDB when offline
async function submitForm(data) {
  if (navigator.onLine) {
    return fetch('/api/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
  
  // Offline: Queue for later
  await queueAction({
    url: '/api/submit',
    method: 'POST',
    body: data,
    timestamp: Date.now()
  });
  
  showNotification('Saved offline. Will sync when online.');
}

async function queueAction(action) {
  const db = await openDB();
  const tx = db.transaction('pending-actions', 'readwrite');
  await tx.store.add(action);
}

// Sync when online
window.addEventListener('online', async () => {
  const db = await openDB();
  const tx = db.transaction('pending-actions', 'readwrite');
  const actions = await tx.store.getAll();
  
  for (const action of actions) {
    try {
      await fetch(action.url, {
        method: action.method,
        body: JSON.stringify(action.body)
      });
      await tx.store.delete(action.id);
    } catch (error) {
      console.error('Sync failed:', error);
      break; // Stop on first failure
    }
  }
});
```

### Using Background Sync API (Preview)

The Background Sync API allows you to defer actions until the user has connectivity:

```javascript
// page.js - Register sync
async function submitForm(data) {
  // Store data in IndexedDB
  await saveToIndexedDB('outbox', data);
  
  // Register sync event
  const registration = await navigator.serviceWorker.ready;
  await registration.sync.register('sync-forms');
  
  showNotification('Will send when online');
}
```

```javascript
// service-worker.js - Handle sync
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-forms') {
    event.waitUntil(syncForms());
  }
});

async function syncForms() {
  const db = await openIndexedDB();
  const items = await db.getAll('outbox');
  
  for (const item of items) {
    await fetch('/api/submit', {
      method: 'POST',
      body: JSON.stringify(item.data)
    });
    await db.delete('outbox', item.id);
  }
}
```

---

## Complete Offline App Example

```javascript
// service-worker.js
const STATIC_CACHE = 'static-v1';
const DYNAMIC_CACHE = 'dynamic-v1';

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/app.js',
  '/styles.css',
  '/offline.html'
];

// Install: Cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Activate: Clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys
          .filter(key => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
          .map(key => caches.delete(key))
      );
    }).then(() => clients.claim())
  );
});

// Fetch: Serve with appropriate strategy
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Same-origin only
  if (url.origin !== location.origin) return;
  
  // Static assets: Cache first
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(request)
        .then(r => r || fetch(request))
    );
    return;
  }
  
  // API: Network first with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          const clone = response.clone();
          caches.open(DYNAMIC_CACHE).then(c => c.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }
  
  // Pages: Network first, offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .catch(() => caches.match('/offline.html'))
    );
    return;
  }
  
  // Everything else: Stale-while-revalidate
  event.respondWith(
    caches.match(request).then(cached => {
      const fetched = fetch(request).then(response => {
        caches.open(DYNAMIC_CACHE).then(c => c.put(request, response.clone()));
        return response;
      });
      return cached || fetched;
    })
  );
});
```

---

## Hands-on Exercise

### Your Task

Create an offline-capable news reader that caches articles for offline reading.

### Requirements

1. Cache the app shell (HTML, CSS, JS) on install
2. Cache articles when viewed
3. Show list of cached articles when offline
4. Display an offline indicator in the UI

<details>
<summary>💡 Hints</summary>

- Use different caches for shell and content
- Store article metadata in IndexedDB for the offline list
- Use `navigator.onLine` for the indicator

</details>

<details>
<summary>✅ Solution</summary>

See the complete example in the section above - it covers all these requirements!

</details>

---

## Summary

✅ Intercept all requests with the `fetch` event
✅ Serve offline fallback pages when network fails
✅ Use **Cache-first** for static assets
✅ Use **Network-first** for dynamic content
✅ Use **Stale-while-revalidate** for best of both
✅ Queue actions offline and sync when online

**Next:** [Caching Strategies](./07-caching-strategies.md)

---

## Further Reading

- [The Offline Cookbook](https://web.dev/articles/offline-cookbook) - Comprehensive caching patterns
- [MDN Service Worker Offline](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Tutorials/js13kGames/Offline_Service_workers) - Tutorial
- [Workbox](https://developers.google.com/web/tools/workbox) - Library for SW caching strategies

<!-- 
Sources Consulted:
- web.dev Offline Cookbook: https://web.dev/articles/offline-cookbook
- MDN Service Worker API: https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API
-->
