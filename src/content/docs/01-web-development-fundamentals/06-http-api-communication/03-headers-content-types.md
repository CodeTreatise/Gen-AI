---
title: "Headers & Content Types"
---

# Headers & Content Types

## Introduction

HTTP headers are the **metadata layer** of web communication. They tell servers what kind of data you're sending, how you want the response formatted, who you are, and how responses should be cached. Mastering headers is essential for building applications that communicate effectively with APIs.

Every time you struggle with a CORS error, debug an authentication issue, or wonder why your file upload isn't working, you're dealing with headers. Understanding them transforms debugging from guesswork into systematic problem-solving.

### What We'll Cover

- Request headers: what clients send to servers
- Response headers: what servers send back
- Content-Type and MIME types for data formats
- Authentication headers and patterns
- Custom headers and naming conventions
- Common header-related issues and solutions

### Prerequisites

- Understanding of [HTTP Methods](./01-http-methods.md)
- Understanding of [Request & Response Structure](./02-request-response-structure.md)
- Basic JavaScript async/await syntax

---

## Understanding HTTP Headers

Headers are key-value pairs that provide metadata about HTTP messages. They follow a simple format:

```
Header-Name: header-value
```

### Header Categories

```mermaid
flowchart TD
    A[HTTP Headers] --> B[Request Headers]
    A --> C[Response Headers]
    A --> D[Representation Headers]
    A --> E[Payload Headers]
    
    B --> B1[Host, Accept, Authorization]
    C --> C1[Set-Cookie, Location, Server]
    D --> D1[Content-Type, Content-Encoding]
    E --> E1[Content-Length, Transfer-Encoding]
```

| Category | Purpose | Examples |
|----------|---------|----------|
| **Request** | Client preferences and context | `Accept`, `Authorization`, `User-Agent` |
| **Response** | Server information and instructions | `Set-Cookie`, `Location`, `Cache-Control` |
| **Representation** | Describe the body format | `Content-Type`, `Content-Encoding` |
| **Payload** | Describe the message body | `Content-Length`, `Transfer-Encoding` |

---

## Content-Type Header

The `Content-Type` header is arguably the most important header—it tells the recipient how to interpret the message body.

### Syntax

```
Content-Type: media-type; parameter=value
```

### Common MIME Types

| MIME Type | Use Case | Example |
|-----------|----------|---------|
| `application/json` | API data (most common) | `{"name": "Alice"}` |
| `application/x-www-form-urlencoded` | Form submissions | `name=Alice&email=a@b.com` |
| `multipart/form-data` | File uploads | Binary file data with boundaries |
| `text/plain` | Plain text | `Hello, World!` |
| `text/html` | Web pages | `<html>...</html>` |
| `application/octet-stream` | Binary data | Raw bytes |
| `image/png`, `image/jpeg` | Images | Image binary data |
| `application/pdf` | PDF documents | PDF binary data |

### JSON Content Type

```javascript
// Sending JSON data - most common in APIs
const response = await fetch('/api/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'Alice',
    email: 'alice@example.com'
  })
});
```

> **Important:** When sending JSON, always set `Content-Type: application/json`. Many servers reject requests without this header.

### Form URL Encoded

Traditional HTML form format:

```javascript
// Form-style data (like HTML forms submit)
const params = new URLSearchParams();
params.append('username', 'alice');
params.append('password', 'secret123');

const response = await fetch('/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: params.toString()
  // Body: "username=alice&password=secret123"
});
```

### Multipart Form Data

For file uploads and mixed content:

```javascript
// File upload with additional fields
const formData = new FormData();
formData.append('avatar', fileInput.files[0]);
formData.append('username', 'alice');

const response = await fetch('/upload', {
  method: 'POST',
  // DON'T set Content-Type manually - browser adds boundary automatically
  body: formData
});
```

> **Warning:** When using `FormData`, do NOT set the `Content-Type` header manually. The browser sets it automatically with the required boundary parameter.

### Character Encoding

Content-Type can include character encoding:

```
Content-Type: application/json; charset=utf-8
Content-Type: text/html; charset=iso-8859-1
```

```javascript
// Server response with encoding
const contentType = response.headers.get('Content-Type');
// "application/json; charset=utf-8"

// Parse the content type
const [mimeType, ...params] = contentType.split(';').map(s => s.trim());
console.log(mimeType);  // "application/json"
```

---

## Accept Header

The `Accept` header tells the server what content types the client can handle:

```javascript
const response = await fetch('/api/data', {
  headers: {
    'Accept': 'application/json'
  }
});
```

### Quality Values

Specify preferences with quality values (q):

```javascript
const response = await fetch('/api/data', {
  headers: {
    // Prefer JSON, accept XML as fallback
    'Accept': 'application/json, application/xml;q=0.9, */*;q=0.1'
  }
});
```

| Accept Value | Meaning |
|--------------|---------|
| `application/json` | Only accept JSON |
| `*/*` | Accept anything |
| `text/*` | Accept any text format |
| `application/json, text/plain;q=0.5` | Prefer JSON, accept plain text |

### Related Accept Headers

```javascript
const response = await fetch('/api/data', {
  headers: {
    'Accept': 'application/json',
    'Accept-Language': 'en-US, en;q=0.9, es;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Charset': 'utf-8'
  }
});
```

---

## Authorization Header

The `Authorization` header carries credentials for authenticating the client:

### Bearer Token (Most Common)

```javascript
// JWT or OAuth access token
const response = await fetch('/api/protected', {
  headers: {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
  }
});
```

### Basic Authentication

```javascript
// Username:password encoded in Base64
const credentials = btoa('username:password');
const response = await fetch('/api/protected', {
  headers: {
    'Authorization': `Basic ${credentials}`
  }
});
```

> **Warning:** Basic auth transmits credentials with every request. Only use over HTTPS and prefer token-based auth for APIs.

### API Key Authentication

Different APIs place keys in different locations:

```javascript
// In Authorization header
const response1 = await fetch('/api/data', {
  headers: { 'Authorization': 'ApiKey my-secret-key' }
});

// In custom header
const response2 = await fetch('/api/data', {
  headers: { 'X-API-Key': 'my-secret-key' }
});

// In query parameter (least secure)
const response3 = await fetch('/api/data?api_key=my-secret-key');
```

### Auth Header Patterns

```mermaid
flowchart TD
    A[Authentication Method] --> B{Token-based?}
    B -->|Yes| C{Type?}
    B -->|No| D[Basic Auth]
    
    C -->|JWT/OAuth| E["Authorization: Bearer token"]
    C -->|API Key| F["X-API-Key: key"]
    D --> G["Authorization: Basic base64"]
```

> **🤖 AI Context:** When calling LLM APIs like OpenAI or Anthropic, you'll typically use Bearer token authentication with your API key.

---

## Common Request Headers

### Host

Required in HTTP/1.1—identifies the target server:

```
Host: api.example.com
```

### User-Agent

Identifies the client software:

```javascript
// Browser sets this automatically, but you can override
const response = await fetch('/api/data', {
  headers: {
    'User-Agent': 'MyApp/1.0'
  }
});
```

### Origin and Referer

Used for CORS and tracking request sources:

```
Origin: https://myapp.example.com
Referer: https://myapp.example.com/page
```

> **Note:** Browsers set these automatically—you typically can't override them with JavaScript.

### Cache-Control (Request)

Control caching behavior for the request:

```javascript
const response = await fetch('/api/data', {
  headers: {
    'Cache-Control': 'no-cache'  // Validate with server before using cache
  }
});
```

### If-Modified-Since / If-None-Match

For conditional requests (caching):

```javascript
// Only get data if modified since this date
const response = await fetch('/api/data', {
  headers: {
    'If-Modified-Since': 'Wed, 22 Jan 2025 10:00:00 GMT'
  }
});

// Or use ETag for version checking
const response2 = await fetch('/api/data', {
  headers: {
    'If-None-Match': '"abc123"'
  }
});

// 304 Not Modified = use cached version
if (response.status === 304) {
  console.log('Using cached data');
}
```

---

## Common Response Headers

### Content-Type

Format of the response body:

```javascript
const response = await fetch('/api/data');
const contentType = response.headers.get('Content-Type');
// "application/json; charset=utf-8"
```

### Content-Length

Size of response body in bytes:

```javascript
const size = response.headers.get('Content-Length');
console.log(`Response size: ${size} bytes`);
```

### Location

URL for redirects or newly created resources:

```javascript
const response = await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'Alice' })
});

if (response.status === 201) {
  const location = response.headers.get('Location');
  console.log(`User created at: ${location}`);
  // "/api/users/456"
}
```

### Cache-Control (Response)

Instructions for caching the response:

```
Cache-Control: max-age=3600, public
Cache-Control: no-store
Cache-Control: private, max-age=600
```

| Directive | Meaning |
|-----------|---------|
| `max-age=3600` | Cache for 1 hour |
| `no-cache` | Validate before using cache |
| `no-store` | Don't cache at all |
| `public` | Any cache can store |
| `private` | Only browser can cache |

### Set-Cookie

Set cookies on the client:

```
Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Strict
```

```javascript
// Cookies are handled automatically by the browser
// To send cookies with fetch, use credentials option
const response = await fetch('/api/data', {
  credentials: 'include'  // Send cookies
});
```

### ETag

Version identifier for caching:

```javascript
const response = await fetch('/api/data');
const etag = response.headers.get('ETag');
// '"abc123"'

// Later, check if modified
const checkResponse = await fetch('/api/data', {
  headers: { 'If-None-Match': etag }
});

if (checkResponse.status === 304) {
  console.log('Data unchanged');
}
```

---

## CORS Headers

Cross-Origin Resource Sharing headers control cross-domain access:

### Response Headers

```
Access-Control-Allow-Origin: https://myapp.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400
```

### Request Headers (Preflight)

```
Access-Control-Request-Method: POST
Access-Control-Request-Headers: Content-Type, Authorization
```

> **Note:** We'll cover CORS in detail in [Lesson 8: CORS](./08-cors.md).

---

## Custom Headers

### Naming Conventions

```javascript
// Common custom header prefixes
const response = await fetch('/api/data', {
  headers: {
    'X-Request-ID': 'req-abc-123',      // Request tracing
    'X-Client-Version': '2.1.0',         // App version
    'X-Correlation-ID': 'corr-xyz-789'  // Distributed tracing
  }
});
```

> **Note:** The `X-` prefix is deprecated but still commonly used. Modern standards suggest just using descriptive names without prefix.

### Rate Limiting Headers

Many APIs return rate limit information:

```javascript
const response = await fetch('/api/data');

const limit = response.headers.get('X-RateLimit-Limit');
const remaining = response.headers.get('X-RateLimit-Remaining');
const reset = response.headers.get('X-RateLimit-Reset');

console.log(`Rate limit: ${remaining}/${limit}`);
console.log(`Resets at: ${new Date(reset * 1000)}`);

if (remaining === '0') {
  console.log('Rate limited! Wait before retrying.');
}
```

### Pagination Headers

```javascript
const response = await fetch('/api/users?page=2');

const total = response.headers.get('X-Total-Count');
const totalPages = response.headers.get('X-Total-Pages');
const link = response.headers.get('Link');

// Link header for navigation
// <https://api.example.com/users?page=3>; rel="next",
// <https://api.example.com/users?page=1>; rel="prev"
```

---

## Working with Headers in JavaScript

### Headers Object

```javascript
// Create Headers object
const headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Authorization', 'Bearer token');

// Or from object
const headers2 = new Headers({
  'Content-Type': 'application/json',
  'Accept': 'application/json'
});

// Methods
headers.has('Content-Type');           // true
headers.get('Content-Type');           // "application/json"
headers.set('Content-Type', 'text/plain');  // Replace
headers.delete('Authorization');        // Remove

// Iterate
for (const [name, value] of headers) {
  console.log(`${name}: ${value}`);
}
```

### Reading Response Headers

```javascript
const response = await fetch('/api/data');

// Get single header
const contentType = response.headers.get('Content-Type');

// Get all headers
const allHeaders = {};
for (const [name, value] of response.headers) {
  allHeaders[name] = value;
}
console.log(allHeaders);

// Note: Not all headers are accessible due to CORS
// Default accessible: Cache-Control, Content-Language, Content-Type,
// Expires, Last-Modified, Pragma
```

### CORS Header Restrictions

In cross-origin requests, only certain headers are accessible:

```javascript
// These are accessible by default (CORS-safelisted)
response.headers.get('Cache-Control');    // ✅ Accessible
response.headers.get('Content-Type');     // ✅ Accessible
response.headers.get('Content-Language'); // ✅ Accessible
response.headers.get('Expires');          // ✅ Accessible

// Custom headers need Access-Control-Expose-Headers
response.headers.get('X-Custom-Header');  // ❌ null (unless exposed)

// Server must send:
// Access-Control-Expose-Headers: X-Custom-Header
```

---

## Best Practices

### 1. Always Set Content-Type

```javascript
// ✅ DO: Set Content-Type for requests with body
await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

// ❌ DON'T: Omit Content-Type
await fetch('/api/users', {
  method: 'POST',
  body: JSON.stringify(data)  // Server may reject
});

// ❌ DON'T: Set Content-Type for FormData
await fetch('/api/upload', {
  method: 'POST',
  headers: { 'Content-Type': 'multipart/form-data' }, // Wrong!
  body: formData  // Browser sets correct Content-Type with boundary
});
```

### 2. Check Content-Type Before Parsing

```javascript
// ✅ DO: Verify content type before parsing
const response = await fetch('/api/data');
const contentType = response.headers.get('Content-Type') || '';

if (contentType.includes('application/json')) {
  const data = await response.json();
} else if (contentType.includes('text/')) {
  const text = await response.text();
} else {
  const blob = await response.blob();
}

// ❌ DON'T: Assume JSON
const data = await response.json();  // Throws if not JSON
```

### 3. Handle Missing Headers Gracefully

```javascript
// ✅ DO: Provide defaults
const contentLength = response.headers.get('Content-Length') || 'unknown';
const rateLimit = parseInt(response.headers.get('X-RateLimit-Remaining') || '100');

// ✅ DO: Check before using
const etag = response.headers.get('ETag');
if (etag) {
  cache.set(url, { etag, data });
}
```

### 4. Secure Sensitive Headers

```javascript
// ✅ DO: Store tokens securely
const token = await getTokenFromSecureStorage();
const response = await fetch('/api/data', {
  headers: { 'Authorization': `Bearer ${token}` }
});

// ❌ DON'T: Log sensitive headers
console.log('Headers:', request.headers);  // May expose tokens!

// ✅ DO: Mask sensitive values when logging
function maskHeader(name, value) {
  const sensitive = ['authorization', 'cookie', 'x-api-key'];
  if (sensitive.includes(name.toLowerCase())) {
    return value.substring(0, 10) + '***';
  }
  return value;
}
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Setting `Content-Type` for `FormData` | Let browser set it with boundary |
| Not setting `Content-Type` for JSON | Always set `application/json` |
| Assuming all headers are accessible | Check CORS exposed headers |
| Using Basic auth without HTTPS | Always use HTTPS for credentials |
| Hardcoding API keys in code | Use environment variables |
| Not handling missing headers | Provide sensible defaults |

---

## Quick Reference

### Request Headers Cheat Sheet

| Header | Purpose | Example Value |
|--------|---------|---------------|
| `Content-Type` | Body format | `application/json` |
| `Accept` | Desired response format | `application/json` |
| `Authorization` | Auth credentials | `Bearer eyJ...` |
| `User-Agent` | Client identifier | `MyApp/1.0` |
| `Accept-Language` | Language preference | `en-US, es;q=0.9` |
| `If-None-Match` | Conditional GET | `"abc123"` |
| `Cache-Control` | Cache instructions | `no-cache` |

### Response Headers Cheat Sheet

| Header | Purpose | Example Value |
|--------|---------|---------------|
| `Content-Type` | Body format | `application/json; charset=utf-8` |
| `Content-Length` | Body size | `1024` |
| `Location` | Redirect/created URL | `/api/users/456` |
| `Cache-Control` | Cache instructions | `max-age=3600` |
| `ETag` | Version identifier | `"abc123"` |
| `Set-Cookie` | Set browser cookie | `session=xyz; HttpOnly` |
| `X-RateLimit-*` | Rate limit info | `X-RateLimit-Remaining: 99` |

---

## Hands-on Exercise

### Your Task

Build a header utility library that simplifies common header operations for API clients, including content type detection, auth header generation, and response header parsing.

### Requirements

1. Create a `HeaderUtils` class with helper methods
2. Implement content type detection from various sources
3. Create auth header generators for different auth types
4. Parse common response headers into usable objects

### Expected Result

A utility that makes header handling cleaner and more consistent across your application.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a Map or object to store common content types
- The `btoa()` function creates Base64 encoding for Basic auth
- Consider parsing the Link header for pagination
- Rate limit headers often use Unix timestamps

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class HeaderUtils {
  // Common MIME types
  static MIME_TYPES = {
    json: 'application/json',
    form: 'application/x-www-form-urlencoded',
    text: 'text/plain',
    html: 'text/html',
    xml: 'application/xml',
    binary: 'application/octet-stream'
  };

  // Create headers for JSON requests
  static jsonHeaders(additionalHeaders = {}) {
    return new Headers({
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...additionalHeaders
    });
  }

  // Create Authorization header
  static authHeader(type, credentials) {
    switch (type.toLowerCase()) {
      case 'bearer':
        return { 'Authorization': `Bearer ${credentials}` };
      
      case 'basic':
        if (typeof credentials === 'object') {
          const encoded = btoa(`${credentials.username}:${credentials.password}`);
          return { 'Authorization': `Basic ${encoded}` };
        }
        return { 'Authorization': `Basic ${btoa(credentials)}` };
      
      case 'apikey':
        return { 'X-API-Key': credentials };
      
      default:
        return { 'Authorization': `${type} ${credentials}` };
    }
  }

  // Detect content type from response
  static getContentType(response) {
    const header = response.headers.get('Content-Type') || '';
    const [mimeType] = header.split(';');
    return mimeType.trim().toLowerCase();
  }

  // Check if response is JSON
  static isJson(response) {
    const type = this.getContentType(response);
    return type.includes('json');
  }

  // Parse response based on content type
  static async parseResponse(response) {
    const contentType = this.getContentType(response);
    
    if (contentType.includes('json')) {
      return response.json();
    }
    if (contentType.includes('text') || contentType.includes('html')) {
      return response.text();
    }
    if (contentType.includes('form-data')) {
      return response.formData();
    }
    return response.blob();
  }

  // Parse rate limit headers
  static parseRateLimit(response) {
    const limit = response.headers.get('X-RateLimit-Limit');
    const remaining = response.headers.get('X-RateLimit-Remaining');
    const reset = response.headers.get('X-RateLimit-Reset');

    if (!limit && !remaining) return null;

    return {
      limit: limit ? parseInt(limit) : null,
      remaining: remaining ? parseInt(remaining) : null,
      resetAt: reset ? new Date(parseInt(reset) * 1000) : null,
      isLimited: remaining === '0'
    };
  }

  // Parse Link header for pagination
  static parseLinkHeader(response) {
    const linkHeader = response.headers.get('Link');
    if (!linkHeader) return null;

    const links = {};
    const parts = linkHeader.split(',');

    for (const part of parts) {
      const match = part.match(/<([^>]+)>;\s*rel="([^"]+)"/);
      if (match) {
        links[match[2]] = match[1];
      }
    }

    return links;
  }

  // Parse caching headers
  static parseCacheHeaders(response) {
    return {
      etag: response.headers.get('ETag'),
      lastModified: response.headers.get('Last-Modified'),
      cacheControl: response.headers.get('Cache-Control'),
      expires: response.headers.get('Expires'),
      age: response.headers.get('Age')
    };
  }

  // Create conditional request headers from cache info
  static conditionalHeaders(cacheInfo) {
    const headers = {};
    if (cacheInfo.etag) {
      headers['If-None-Match'] = cacheInfo.etag;
    }
    if (cacheInfo.lastModified) {
      headers['If-Modified-Since'] = cacheInfo.lastModified;
    }
    return headers;
  }

  // Merge multiple header sources
  static merge(...headerSources) {
    const result = new Headers();
    
    for (const source of headerSources) {
      if (!source) continue;
      
      const entries = source instanceof Headers
        ? source.entries()
        : Object.entries(source);
      
      for (const [key, value] of entries) {
        result.set(key, value);
      }
    }
    
    return result;
  }

  // Mask sensitive headers for logging
  static maskForLogging(headers, sensitiveNames = ['authorization', 'cookie', 'x-api-key']) {
    const result = {};
    const entries = headers instanceof Headers ? headers.entries() : Object.entries(headers);
    
    for (const [name, value] of entries) {
      if (sensitiveNames.includes(name.toLowerCase())) {
        result[name] = value.length > 10 
          ? value.substring(0, 6) + '***' + value.slice(-4)
          : '***';
      } else {
        result[name] = value;
      }
    }
    
    return result;
  }
}

// Usage examples
async function demo() {
  // Create authenticated JSON request
  const headers = HeaderUtils.merge(
    HeaderUtils.jsonHeaders(),
    HeaderUtils.authHeader('bearer', 'my-token')
  );
  
  console.log(HeaderUtils.maskForLogging(headers));
  // { 'Content-Type': 'application/json', 'Authorization': 'Bearer***oken' }
  
  const response = await fetch('/api/users', { headers });
  
  // Parse response based on content type
  const data = await HeaderUtils.parseResponse(response);
  
  // Check rate limits
  const rateLimit = HeaderUtils.parseRateLimit(response);
  if (rateLimit?.isLimited) {
    console.log(`Rate limited until ${rateLimit.resetAt}`);
  }
  
  // Get pagination links
  const links = HeaderUtils.parseLinkHeader(response);
  if (links?.next) {
    console.log(`Next page: ${links.next}`);
  }
  
  // Store cache info
  const cacheInfo = HeaderUtils.parseCacheHeaders(response);
  
  // Later: conditional request
  const conditionalResponse = await fetch('/api/users', {
    headers: HeaderUtils.conditionalHeaders(cacheInfo)
  });
  
  if (conditionalResponse.status === 304) {
    console.log('Using cached data');
  }
}
```

</details>

### Bonus Challenges

- [ ] Add retry-after header parsing for 429 responses
- [ ] Implement header validation (check for invalid characters)
- [ ] Add support for parsing `WWW-Authenticate` header

---

## Summary

✅ **Content-Type** specifies the format of the message body—always set it for requests with a body

✅ **Accept** tells the server what response format you prefer

✅ **Authorization** carries authentication credentials (Bearer, Basic, API Key)

✅ Don't manually set **Content-Type** for `FormData`—let the browser handle it

✅ CORS restricts which **response headers** are accessible in cross-origin requests

✅ **Custom headers** with `X-` prefix carry application-specific metadata

✅ Always **check Content-Type** before parsing response bodies

**Next:** [Status Codes](./04-status-codes.md)

---

## Further Reading

- [MDN: HTTP Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers) - Complete header reference
- [MDN: MIME Types](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types) - Content type guide
- [RFC 7231: HTTP/1.1 Semantics](https://tools.ietf.org/html/rfc7231) - Official specification

---

<!-- 
Sources Consulted:
- MDN HTTP Headers: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
- MDN MIME Types: https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types
- MDN Using Fetch: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
-->
