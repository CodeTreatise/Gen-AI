---
title: "Network Tab Analysis"
---

# Network Tab Analysis

## Introduction

Every time your page loads an image, fetches data from an API, or downloads a JavaScript file, that request passes through the network. The **Network tab** lets you see every request, inspect its details, and diagnose performance issues.

This is essential for debugging API integrations, optimizing load times, and understanding how your application communicates with servers.

### What We'll Cover

- Request/response inspection
- Headers examination
- Preview and response tabs
- Timing breakdown
- Filtering requests
- Throttling (3G, offline simulation)
- WebSocket inspection
- Request blocking

### Prerequisites

- Understanding of HTTP basics
- Familiarity with APIs and fetch

---

## Opening the Network Tab

1. Open DevTools (`F12`)
2. Click the **Network** tab
3. **Reload the page** to capture requests from the start

> **Important:** The Network tab only records while it's open. Always reload after opening!

### Network Tab Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ [Filter bar] [Preserve log] [Disable cache] [Throttling]       │
├─────────────────────────────────────────────────────────────────┤
│ Name          │ Status │ Type    │ Size   │ Time  │ Waterfall  │
├───────────────┼────────┼─────────┼────────┼───────┼────────────│
│ index.html    │ 200    │ document│ 5.2 KB │ 45ms  │ ████       │
│ styles.css    │ 200    │ style   │ 12 KB  │ 23ms  │   ███      │
│ app.js        │ 200    │ script  │ 89 KB  │ 120ms │    ███████ │
│ /api/users    │ 200    │ fetch   │ 2.1 KB │ 340ms │        ████│
└─────────────────────────────────────────────────────────────────┘
```

---

## Request/Response Inspection

Click any request to see its details.

### Headers Tab

Shows HTTP headers for request and response:

```http
Request Headers:
  Request URL: https://api.example.com/users
  Request Method: GET
  Status Code: 200 OK
  Accept: application/json
  Authorization: Bearer sk-xxx...

Response Headers:
  Content-Type: application/json
  Content-Length: 2134
  Cache-Control: max-age=3600
  X-RateLimit-Remaining: 98
```

**Key headers to check:**
| Header | What it tells you |
|--------|-------------------|
| `Content-Type` | Data format (JSON, HTML, etc.) |
| `Status Code` | Success (2xx), redirect (3xx), error (4xx, 5xx) |
| `Cache-Control` | Caching behavior |
| `Authorization` | Auth tokens (for debugging) |
| `CORS headers` | Cross-origin permissions |

### Payload Tab

For POST/PUT requests, shows the request body:

```json
{
  "name": "Alice",
  "email": "alice@example.com",
  "role": "admin"
}
```

Toggle between:
- **Parsed** - Formatted view
- **Source** - Raw request body

### Preview Tab

Renders the response in a readable format:
- JSON is syntax-highlighted and expandable
- Images display visually
- HTML renders (for document requests)

### Response Tab

Shows the raw response body:

```json
{
  "id": 123,
  "name": "Alice",
  "email": "alice@example.com",
  "createdAt": "2025-01-24T10:00:00Z"
}
```

### Cookies Tab

Shows cookies sent with the request and set by the response.

---

## Timing Breakdown

The **Timing** tab shows exactly where time is spent:

```
Queued at:        0 ms
Started at:       2.5 ms

Resource Scheduling
├── Queuing:      2.5 ms

Connection Start
├── Stalled:      1.2 ms
├── DNS Lookup:   12 ms
├── Initial Connection: 45 ms
├── SSL:          28 ms

Request/Response
├── Request sent: 0.3 ms
├── Waiting (TTFB): 156 ms    ← Server processing time
├── Content Download: 23 ms

Total: 268 ms
```

### Key Timing Metrics

| Phase | What it measures |
|-------|------------------|
| **Queuing** | Waiting for network thread |
| **Stalled** | Blocked by browser limits |
| **DNS Lookup** | Domain name resolution |
| **Initial Connection** | TCP handshake |
| **SSL** | TLS/SSL negotiation |
| **TTFB** | Time to First Byte (server response time) |
| **Content Download** | Downloading the response |

### Diagnosing Slow Requests

| Slow Phase | Possible Cause |
|------------|----------------|
| DNS Lookup | DNS server issues, try DNS prefetch |
| Connection | Server far away, consider CDN |
| SSL | Certificate chain issues |
| TTFB | Slow server, database, or API |
| Download | Large response, enable compression |

---

## Filtering Requests

The filter bar helps find specific requests.

### Type Filters

Click to filter by type:
- **All** - Everything
- **Fetch/XHR** - API calls (most common for debugging)
- **JS** - JavaScript files
- **CSS** - Stylesheets
- **Img** - Images
- **Media** - Audio/video
- **Font** - Web fonts
- **Doc** - HTML documents
- **WS** - WebSockets
- **Manifest** - Web app manifests

### Text Filter

Type to search:
- `api` - URLs containing "api"
- `status-code:404` - Only 404 errors
- `method:POST` - Only POST requests
- `domain:api.example.com` - Specific domain
- `-analytics` - Exclude analytics requests
- `larger-than:1M` - Files over 1MB

### Invert Filter

Click the invert icon to show everything EXCEPT matches.

---

## Throttling Network Speed

Test your app on slow connections.

### Preset Profiles

| Profile | Download | Upload | Latency |
|---------|----------|--------|---------|
| Fast 3G | 1.5 Mbps | 750 Kbps | 562 ms |
| Slow 3G | 400 Kbps | 400 Kbps | 2000 ms |
| Offline | 0 | 0 | ∞ |

### Custom Throttling

1. Click throttling dropdown → **Add...**
2. Set download/upload speeds and latency
3. Save as a custom profile

### Testing Offline

1. Select **Offline** from throttling dropdown
2. Test that your app handles network errors gracefully
3. Verify service worker serves cached content

```javascript
// Your code should handle this:
try {
  const response = await fetch('/api/data');
} catch (error) {
  showOfflineMessage();
}
```

---

## WebSocket Inspection

For real-time connections, the Network tab shows WebSocket frames.

### Finding WebSocket Connections

1. Filter by **WS** type
2. Click on the WebSocket connection

### Messages Tab

Shows all frames sent and received:

```
Direction │ Data                                    │ Time
──────────┼─────────────────────────────────────────┼──────────
↑ Sent    │ {"type":"subscribe","channel":"chat"}   │ 0ms
↓ Received│ {"type":"ack","id":1}                   │ 15ms
↓ Received│ {"type":"message","text":"Hello"}       │ 2340ms
↑ Sent    │ {"type":"message","text":"Hi there"}    │ 5230ms
```

- **Green arrow** (↑) - Messages you sent
- **Red arrow** (↓) - Messages received

### Binary Frames

For binary data (images, audio), the tab shows the frame size and type.

---

## Request Blocking

Block specific requests to test fallbacks.

### Enable Request Blocking

1. Open **Command Menu** (`Ctrl+Shift+P`)
2. Type "Show Request Blocking"
3. The Request Blocking drawer appears

### Block Patterns

Add patterns to block:
- `*.js` - Block all JavaScript
- `api.example.com` - Block a domain
- `analytics` - Block anything with "analytics" in URL

### Use Cases

| Block | Test |
|-------|------|
| JavaScript | Graceful degradation |
| Third-party scripts | Performance without analytics/ads |
| API endpoint | Error handling |
| CSS | Content without styles |

---

## Preserve Log & Disable Cache

### Preserve Log

Keep network log across page navigations:
- Useful for tracking redirects
- Debug multi-page flows
- See what happens before a reload

### Disable Cache

Force fresh downloads (ignore cached files):
- Enable while DevTools is open
- Essential during development
- See actual load times without cache

---

## Copy and Export

### Copy as cURL

Right-click a request → **Copy** → **Copy as cURL**:

```bash
curl 'https://api.example.com/users' \
  -H 'Accept: application/json' \
  -H 'Authorization: Bearer sk-xxx' \
  --compressed
```

Paste into terminal to replay the request!

### Copy as Fetch

Get JavaScript code to reproduce the request:

```javascript
fetch("https://api.example.com/users", {
  "headers": {
    "accept": "application/json",
    "authorization": "Bearer sk-xxx"
  },
  "method": "GET"
});
```

### Export HAR

Save all requests as a HAR (HTTP Archive) file:
1. Right-click in request list
2. **Save all as HAR with content**
3. Share with teammates or import into other tools

---

## Practical Debugging Scenarios

### Debugging a Failed API Call

1. Filter by **Fetch/XHR**
2. Find the red (failed) request
3. Check **Status Code** (401? 404? 500?)
4. Check **Response** for error message
5. Check **Headers** for missing auth or CORS issues

### Diagnosing Slow Page Load

1. Reload with **Disable cache** enabled
2. Look at the **Waterfall** column
3. Find the longest bars
4. Check if they're sequential (blocking) or parallel
5. Click to see timing breakdown

### Checking CORS Issues

1. Look for blocked requests (red)
2. Check Console for CORS errors
3. In Network, check response headers:
   - `Access-Control-Allow-Origin`
   - `Access-Control-Allow-Methods`
   - `Access-Control-Allow-Headers`

---

## Hands-on Exercise

### Your Task

Analyze the network requests on a real website.

1. Open any web application (like GitHub, Twitter, or your own app)
2. Open Network tab and reload
3. Answer:
   - How many requests were made?
   - What was the largest file?
   - How long until the page was interactive?
   - Find an API call and inspect its response

### Challenge

1. Enable **Slow 3G** throttling
2. Reload and note the difference
3. Identify which resources cause the biggest delays

---

## Summary

✅ **Headers tab** shows HTTP request/response headers
✅ **Response tab** displays the response body
✅ **Timing tab** breaks down where time is spent
✅ Filter by **type** (Fetch/XHR, JS, CSS) or **text search**
✅ **Throttle** to simulate slow connections
✅ **WebSocket** messages visible in Messages tab
✅ **Block requests** to test error handling
✅ **Copy as cURL/Fetch** to reproduce requests

**Next:** [Performance Profiling Basics](./04-performance-profiling.md)

---

## Further Reading

- [Chrome Network Reference](https://developer.chrome.com/docs/devtools/network/reference/)
- [MDN HTTP Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)

<!-- 
Sources Consulted:
- Chrome DevTools Network: https://developer.chrome.com/docs/devtools/network/
- MDN HTTP: https://developer.mozilla.org/en-US/docs/Web/HTTP
-->
