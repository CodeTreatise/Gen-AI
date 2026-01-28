---
title: "Request & Response Structure"
---

# Request & Response Structure

## Introduction

Every HTTP interaction follows a predictable structure: a client sends a **request**, and the server returns a **response**. Understanding the anatomy of these messages is essential for debugging API issues, building robust applications, and working effectively with web services.

Whether you're troubleshooting a failed API call, optimizing network performance, or building your own API, knowing what's inside HTTP messages helps you identify problems quickly and communicate effectively with backend developers.

### What We'll Cover

- The structure of HTTP request messages
- The anatomy of HTTP response messages
- How headers convey metadata
- Request and response body formats
- Content negotiation between client and server
- Inspecting HTTP traffic in browser DevTools

### Prerequisites

- Understanding of [HTTP Methods](./01-http-methods.md)
- Basic JavaScript and async/await syntax
- Access to browser developer tools

---

## HTTP Message Overview

HTTP is a text-based protocol (though HTTP/2+ uses binary framing). Both requests and responses follow a similar structure:

```mermaid
flowchart LR
    subgraph Request ["HTTP Request"]
        A[Request Line] --> B[Headers]
        B --> C[Empty Line]
        C --> D[Body - optional]
    end
    
    subgraph Response ["HTTP Response"]
        E[Status Line] --> F[Headers]
        F --> G[Empty Line]
        G --> H[Body - optional]
    end
    
    Request --> |Server processes| Response
```

### Message Components

| Component | Request | Response |
|-----------|---------|----------|
| **First Line** | Request line (method, path, version) | Status line (version, code, reason) |
| **Headers** | Metadata about request | Metadata about response |
| **Empty Line** | Separator (CRLF) | Separator (CRLF) |
| **Body** | Data being sent (optional) | Data being returned (optional) |

---

## HTTP Request Structure

An HTTP request consists of four parts: the request line, headers, an empty line, and an optional body.

### The Request Line

The first line of every HTTP request contains three elements:

```
METHOD /path/to/resource HTTP/1.1
```

| Element | Description | Example |
|---------|-------------|---------|
| **Method** | The HTTP verb | `GET`, `POST`, `PUT` |
| **Request-URI** | Path to the resource | `/api/users/123` |
| **HTTP Version** | Protocol version | `HTTP/1.1`, `HTTP/2` |

### Example Request

Here's a complete HTTP request to create a new user:

```http
POST /api/users HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Accept: application/json
Content-Length: 67

{"name": "Alice Johnson", "email": "alice@example.com", "role": "admin"}
```

Let's break this down:

```mermaid
flowchart TD
    subgraph RL ["Request Line"]
        A["POST /api/users HTTP/1.1"]
    end
    
    subgraph Headers ["Request Headers"]
        B["Host: api.example.com"]
        C["Content-Type: application/json"]
        D["Authorization: Bearer eyJ..."]
        E["Accept: application/json"]
        F["Content-Length: 67"]
    end
    
    subgraph Body ["Request Body"]
        G["{\"name\": \"Alice Johnson\", ...}"]
    end
    
    RL --> Headers
    Headers --> |Empty Line| Body
```

### Request Headers

Headers provide metadata about the request. They follow the format `Header-Name: value`:

```javascript
// These headers are sent with a fetch request
const response = await fetch('https://api.example.com/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',      // Format of request body
    'Accept': 'application/json',            // Expected response format
    'Authorization': 'Bearer token123',       // Authentication
    'X-Request-ID': 'req-abc-123'            // Custom header for tracing
  },
  body: JSON.stringify({ name: 'Alice' })
});
```

### Common Request Headers

| Header | Purpose | Example Value |
|--------|---------|---------------|
| `Host` | Target server (required in HTTP/1.1) | `api.example.com` |
| `Content-Type` | Format of request body | `application/json` |
| `Content-Length` | Size of body in bytes | `156` |
| `Accept` | Desired response format | `application/json` |
| `Authorization` | Authentication credentials | `Bearer eyJ...` |
| `User-Agent` | Client software info | `Mozilla/5.0...` |
| `Cookie` | Session/state data | `session=abc123` |
| `Origin` | Request origin (CORS) | `https://myapp.com` |

### Request Body

The body contains data sent to the server. Not all requests have bodies:

| Method | Typically Has Body? | Use Case |
|--------|---------------------|----------|
| GET | No | Query params in URL |
| POST | Yes | Creating resources |
| PUT | Yes | Replacing resources |
| PATCH | Yes | Updating resources |
| DELETE | Usually no | Sometimes for bulk delete |

```javascript
// POST request with JSON body
const createUser = async (userData) => {
  const response = await fetch('/api/users', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'  // Must match body format
    },
    body: JSON.stringify(userData)  // Body must be a string
  });
  return response.json();
};

// POST with form data
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('document', file);
  formData.append('description', 'My file');
  
  // Note: Don't set Content-Type - browser sets it with boundary
  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData
  });
  return response.json();
};
```

> **Warning:** Always set the `Content-Type` header to match your body format. Mismatches cause parsing errors on the server.

---

## HTTP Response Structure

The server's response follows a similar structure: status line, headers, empty line, and optional body.

### The Status Line

The first line contains the protocol version, status code, and reason phrase:

```
HTTP/1.1 200 OK
```

| Element | Description | Example |
|---------|-------------|---------|
| **HTTP Version** | Protocol version | `HTTP/1.1` |
| **Status Code** | 3-digit result code | `200`, `404`, `500` |
| **Reason Phrase** | Human-readable status | `OK`, `Not Found` |

### Example Response

Here's a complete HTTP response:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /api/users/456
X-Request-ID: req-abc-123
Date: Fri, 24 Jan 2025 10:30:00 GMT
Content-Length: 142

{
  "id": 456,
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "role": "admin",
  "createdAt": "2025-01-24T10:30:00Z"
}
```

### Response Headers

Response headers provide metadata about the response:

```javascript
const response = await fetch('/api/users');

// Access response headers
console.log(response.headers.get('Content-Type'));     // application/json
console.log(response.headers.get('Content-Length'));   // 1024
console.log(response.headers.get('Cache-Control'));    // max-age=3600
console.log(response.headers.get('X-RateLimit-Remaining')); // 99

// Iterate all headers
for (const [name, value] of response.headers) {
  console.log(`${name}: ${value}`);
}
```

### Common Response Headers

| Header | Purpose | Example Value |
|--------|---------|---------------|
| `Content-Type` | Format of response body | `application/json; charset=utf-8` |
| `Content-Length` | Size of body in bytes | `1024` |
| `Location` | URL of created resource | `/api/users/456` |
| `Cache-Control` | Caching directives | `max-age=3600, public` |
| `Set-Cookie` | Set browser cookie | `session=xyz; HttpOnly` |
| `Date` | Response timestamp | `Fri, 24 Jan 2025 10:30:00 GMT` |
| `ETag` | Resource version identifier | `"abc123"` |
| `X-RateLimit-*` | Rate limiting info | `X-RateLimit-Remaining: 99` |

### Response Body

The body contains the actual data returned by the server:

```javascript
const response = await fetch('/api/users/123');

// The response object provides methods to read the body
// Each method returns a Promise

// As JSON (most common for APIs)
const user = await response.json();

// As plain text
const text = await response.text();

// As binary blob (for files, images)
const blob = await response.blob();

// As ArrayBuffer (for raw binary)
const buffer = await response.arrayBuffer();

// As FormData (for multipart responses)
const formData = await response.formData();
```

> **Important:** You can only read the response body once. Calling `response.json()` after `response.text()` will throw an error. Clone the response if you need to read it multiple times.

```javascript
// Reading body multiple times
const response = await fetch('/api/data');
const clone = response.clone();

const json = await response.json();
const text = await clone.text();
```

---

## The Response Object in JavaScript

The Fetch API returns a `Response` object with properties and methods for inspecting the response:

### Response Properties

```javascript
const response = await fetch('/api/users');

// Status information
console.log(response.status);       // 200 (numeric status code)
console.log(response.statusText);   // "OK" (reason phrase)
console.log(response.ok);           // true (status 200-299)

// URL information
console.log(response.url);          // Full URL after redirects
console.log(response.redirected);   // true if redirected

// Response type (for CORS)
console.log(response.type);         // "basic", "cors", "opaque"

// Headers
console.log(response.headers);      // Headers object
```

### Checking Response Status

Always check the response status before parsing the body:

```javascript
async function fetchUser(id) {
  const response = await fetch(`/api/users/${id}`);
  
  // Check if request was successful
  if (!response.ok) {
    // Handle different error codes
    switch (response.status) {
      case 400:
        throw new Error('Invalid request');
      case 401:
        throw new Error('Authentication required');
      case 403:
        throw new Error('Access denied');
      case 404:
        throw new Error('User not found');
      case 429:
        throw new Error('Too many requests');
      default:
        throw new Error(`HTTP error: ${response.status}`);
    }
  }
  
  return response.json();
}
```

### Complete Response Handling Pattern

```javascript
async function apiRequest(url, options = {}) {
  try {
    const response = await fetch(url, options);
    
    // Log response metadata
    console.log(`${response.status} ${response.statusText}`);
    console.log(`Content-Type: ${response.headers.get('Content-Type')}`);
    
    // Handle non-success responses
    if (!response.ok) {
      // Try to parse error body
      let errorMessage = `HTTP ${response.status}`;
      try {
        const errorBody = await response.json();
        errorMessage = errorBody.message || errorBody.error || errorMessage;
      } catch {
        // Body isn't JSON, use status text
        errorMessage = response.statusText;
      }
      throw new Error(errorMessage);
    }
    
    // Handle empty responses (204 No Content)
    if (response.status === 204) {
      return null;
    }
    
    // Parse based on content type
    const contentType = response.headers.get('Content-Type') || '';
    
    if (contentType.includes('application/json')) {
      return response.json();
    } else if (contentType.includes('text/')) {
      return response.text();
    } else {
      return response.blob();
    }
  } catch (error) {
    if (error.name === 'TypeError') {
      // Network error (no response received)
      throw new Error('Network error: Unable to reach server');
    }
    throw error;
  }
}
```

---

## Content Negotiation

Content negotiation allows clients and servers to agree on the format of data being exchanged.

### Client-Driven Negotiation

The client specifies preferred formats using the `Accept` header:

```javascript
// Request JSON response
const response = await fetch('/api/users', {
  headers: {
    'Accept': 'application/json'
  }
});

// Request with fallback options
const response2 = await fetch('/api/data', {
  headers: {
    // Prefer JSON, accept XML as fallback, any as last resort
    'Accept': 'application/json, application/xml;q=0.9, */*;q=0.8'
  }
});
```

### Quality Values (q-values)

Quality values indicate preference (0 to 1, default 1):

```
Accept: application/json, application/xml;q=0.9, text/plain;q=0.5
```

| Format | q-value | Priority |
|--------|---------|----------|
| `application/json` | 1.0 (default) | Highest |
| `application/xml` | 0.9 | Second |
| `text/plain` | 0.5 | Third |

### Server Response

The server responds with the `Content-Type` header indicating the actual format:

```http
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Vary: Accept

{"users": [...]}
```

The `Vary: Accept` header tells caches that the response varies based on the `Accept` header.

### Language Negotiation

Similar negotiation works for languages:

```javascript
const response = await fetch('/api/content', {
  headers: {
    'Accept-Language': 'en-US, en;q=0.9, es;q=0.8'
  }
});
```

---

## Request and Response in Action

Let's trace a complete request-response cycle:

```mermaid
sequenceDiagram
    participant Client
    participant DNS
    participant Server
    participant Database
    
    Client->>DNS: Resolve api.example.com
    DNS-->>Client: 93.184.216.34
    
    Client->>Server: POST /api/users HTTP/1.1<br/>Host: api.example.com<br/>Content-Type: application/json<br/>{"name": "Alice"}
    
    Server->>Database: INSERT user
    Database-->>Server: User created (id: 456)
    
    Server-->>Client: HTTP/1.1 201 Created<br/>Location: /api/users/456<br/>{"id": 456, "name": "Alice"}
```

### Practical Example: User Registration

```javascript
async function registerUser(userData) {
  // Build the request
  const request = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client-Version': '1.0.0'
    },
    body: JSON.stringify(userData)
  };
  
  console.log('Request:', request);
  
  // Send the request
  const response = await fetch('/api/users', request);
  
  // Log response details
  console.log('Response Status:', response.status, response.statusText);
  console.log('Response Headers:');
  for (const [key, value] of response.headers) {
    console.log(`  ${key}: ${value}`);
  }
  
  // Handle the response
  if (response.status === 201) {
    const user = await response.json();
    const location = response.headers.get('Location');
    console.log('User created at:', location);
    return user;
  }
  
  if (response.status === 409) {
    throw new Error('User already exists');
  }
  
  if (response.status === 422) {
    const errors = await response.json();
    throw new Error(`Validation failed: ${JSON.stringify(errors)}`);
  }
  
  throw new Error(`Unexpected response: ${response.status}`);
}

// Usage
try {
  const user = await registerUser({
    name: 'Alice Johnson',
    email: 'alice@example.com',
    password: 'securePassword123'
  });
  console.log('Registered:', user);
} catch (error) {
  console.error('Registration failed:', error.message);
}
```

---

## Inspecting HTTP Traffic

### Browser DevTools

The Network tab in browser DevTools shows complete request/response details:

1. **Open DevTools** (F12 or Cmd+Option+I)
2. **Go to Network tab**
3. **Make a request**
4. **Click on the request** to see details

| Tab | Shows |
|-----|-------|
| Headers | Request/response headers |
| Payload | Request body (for POST/PUT) |
| Preview | Formatted response body |
| Response | Raw response body |
| Timing | Request timing breakdown |

### Programmatic Inspection

```javascript
// Create a wrapper to log all requests
async function debugFetch(url, options = {}) {
  console.group(`🌐 ${options.method || 'GET'} ${url}`);
  
  // Log request
  console.log('📤 Request Headers:', options.headers || {});
  if (options.body) {
    console.log('📤 Request Body:', options.body);
  }
  
  const startTime = performance.now();
  const response = await fetch(url, options);
  const duration = performance.now() - startTime;
  
  // Log response
  console.log(`📥 Response: ${response.status} ${response.statusText} (${duration.toFixed(0)}ms)`);
  console.log('📥 Response Headers:');
  for (const [key, value] of response.headers) {
    console.log(`   ${key}: ${value}`);
  }
  
  console.groupEnd();
  return response;
}

// Usage
const response = await debugFetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'Test' })
});
```

**Console Output:**
```
🌐 POST /api/users
  📤 Request Headers: {Content-Type: 'application/json'}
  📤 Request Body: {"name":"Test"}
  📥 Response: 201 Created (45ms)
  📥 Response Headers:
     content-type: application/json
     location: /api/users/789
```

---

## Best Practices

### 1. Always Check Response Status

```javascript
// ✅ DO: Check status before parsing
const response = await fetch('/api/data');
if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}
const data = await response.json();

// ❌ DON'T: Assume success
const data = await fetch('/api/data').then(r => r.json());
// This will try to parse error pages as JSON!
```

### 2. Set Appropriate Headers

```javascript
// ✅ DO: Set Content-Type for request body
await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

// ❌ DON'T: Omit Content-Type
await fetch('/api/users', {
  method: 'POST',
  body: JSON.stringify(data)  // Server may not parse correctly
});
```

### 3. Handle All Response Types

```javascript
// ✅ DO: Handle based on Content-Type
const contentType = response.headers.get('Content-Type');
const data = contentType?.includes('json') 
  ? await response.json()
  : await response.text();

// ❌ DON'T: Assume JSON
const data = await response.json();  // Throws if not JSON
```

### 4. Clone Before Multiple Reads

```javascript
// ✅ DO: Clone if reading multiple times
const response = await fetch('/api/data');
const clone = response.clone();
const json = await response.json();
const text = await clone.text();

// ❌ DON'T: Read body twice
const response = await fetch('/api/data');
const json = await response.json();
const text = await response.text();  // Error: body already consumed
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not checking `response.ok` | Always verify status before parsing body |
| Forgetting `Content-Type` header | Set header to match body format |
| Parsing body multiple times | Clone response before multiple reads |
| Ignoring empty responses (204) | Check status before calling `.json()` |
| Treating fetch errors as HTTP errors | Distinguish network errors from HTTP errors |
| Not handling different content types | Check `Content-Type` header before parsing |

---

## Hands-on Exercise

### Your Task

Build an HTTP request/response inspector that logs detailed information about API calls and creates a reusable wrapper for debugging network issues.

### Requirements

1. Create a `HttpInspector` class that wraps `fetch`
2. Log complete request details (method, URL, headers, body)
3. Log complete response details (status, headers, timing)
4. Handle and log errors appropriately
5. Support enabling/disabling logging

### Expected Result

A debugging tool that produces output like:

```
━━━ HTTP Request ━━━
POST https://api.example.com/users
Headers:
  Content-Type: application/json
  Authorization: Bearer ***
Body: {"name": "Alice"}

━━━ HTTP Response ━━━
Status: 201 Created (127ms)
Headers:
  content-type: application/json
  location: /api/users/456
Body: {"id": 456, "name": "Alice"}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `performance.now()` to measure request duration
- Store the original body before sending (it can only be read once)
- Mask sensitive headers like `Authorization`
- Clone the response to log the body while still returning it

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class HttpInspector {
  constructor(options = {}) {
    this.enabled = options.enabled ?? true;
    this.maskHeaders = options.maskHeaders ?? ['authorization', 'cookie', 'x-api-key'];
    this.logBody = options.logBody ?? true;
  }

  enable() { this.enabled = true; }
  disable() { this.enabled = false; }

  maskValue(value) {
    if (value.length <= 8) return '***';
    return value.substring(0, 4) + '***' + value.substring(value.length - 4);
  }

  formatHeaders(headers, mask = false) {
    const lines = [];
    const entries = headers instanceof Headers 
      ? [...headers.entries()] 
      : Object.entries(headers || {});
    
    for (const [key, value] of entries) {
      const shouldMask = mask && this.maskHeaders.includes(key.toLowerCase());
      lines.push(`  ${key}: ${shouldMask ? this.maskValue(value) : value}`);
    }
    return lines.join('\n');
  }

  log(...args) {
    if (this.enabled) console.log(...args);
  }

  async fetch(url, options = {}) {
    if (!this.enabled) {
      return fetch(url, options);
    }

    const method = options.method || 'GET';
    const startTime = performance.now();
    
    // Log request
    this.log('\n━━━ HTTP Request ━━━');
    this.log(`${method} ${url}`);
    
    if (options.headers && Object.keys(options.headers).length > 0) {
      this.log('Headers:');
      this.log(this.formatHeaders(options.headers, true));
    }
    
    if (options.body && this.logBody) {
      let bodyPreview = options.body;
      if (typeof bodyPreview === 'string' && bodyPreview.length > 500) {
        bodyPreview = bodyPreview.substring(0, 500) + '... (truncated)';
      }
      this.log(`Body: ${bodyPreview}`);
    }

    try {
      const response = await fetch(url, options);
      const duration = performance.now() - startTime;
      
      // Clone for logging (so caller can still read body)
      const clone = response.clone();
      
      // Log response
      this.log('\n━━━ HTTP Response ━━━');
      this.log(`Status: ${response.status} ${response.statusText} (${duration.toFixed(0)}ms)`);
      
      this.log('Headers:');
      this.log(this.formatHeaders(response.headers));
      
      if (this.logBody) {
        try {
          const contentType = response.headers.get('Content-Type') || '';
          let body;
          
          if (contentType.includes('json')) {
            body = JSON.stringify(await clone.json(), null, 2);
          } else if (contentType.includes('text')) {
            body = await clone.text();
          } else {
            body = `[Binary data: ${response.headers.get('Content-Length') || 'unknown'} bytes]`;
          }
          
          if (body.length > 1000) {
            body = body.substring(0, 1000) + '\n... (truncated)';
          }
          this.log(`Body: ${body}`);
        } catch (e) {
          this.log(`Body: [Unable to read: ${e.message}]`);
        }
      }
      
      this.log('━━━━━━━━━━━━━━━━━━\n');
      
      return response;
    } catch (error) {
      const duration = performance.now() - startTime;
      
      this.log('\n━━━ HTTP Error ━━━');
      this.log(`Error after ${duration.toFixed(0)}ms: ${error.message}`);
      this.log('━━━━━━━━━━━━━━━━━━\n');
      
      throw error;
    }
  }
}

// Usage
const inspector = new HttpInspector({ enabled: true });

// Use instead of fetch
const response = await inspector.fetch('https://api.example.com/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer my-secret-token-12345'
  },
  body: JSON.stringify({ name: 'Alice', email: 'alice@example.com' })
});

// Disable for production
inspector.disable();
```

</details>

### Bonus Challenges

- [ ] Add support for curl command generation
- [ ] Implement request/response history storage
- [ ] Add timing breakdown (DNS, connect, TLS, first byte, download)

---

## Summary

✅ HTTP requests have a **request line** (method, path, version), **headers**, and optional **body**

✅ HTTP responses have a **status line** (version, code, reason), **headers**, and optional **body**

✅ Headers provide **metadata** about the message (content type, auth, caching)

✅ The `Response` object provides `status`, `ok`, `headers`, and body reading methods

✅ **Content negotiation** uses `Accept` and `Content-Type` headers to agree on formats

✅ Always check `response.ok` before parsing the body

✅ Clone responses if you need to read the body multiple times

**Next:** [Headers & Content Types](./03-headers-content-types.md)

---

## Further Reading

- [MDN: HTTP Messages](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages) - Complete message structure reference
- [MDN: Response](https://developer.mozilla.org/en-US/docs/Web/API/Response) - Response object documentation
- [HTTP/2 Binary Framing](https://web.dev/performance-http2/) - How HTTP/2 changes the message format

---

<!-- 
Sources Consulted:
- MDN HTTP Messages: https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages
- MDN Using Fetch: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
- MDN Response: https://developer.mozilla.org/en-US/docs/Web/API/Response
-->
