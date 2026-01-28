---
title: "Content Security Policy (CSP)"
---

# Content Security Policy (CSP)

## Introduction

Content Security Policy is a powerful security header that prevents XSS attacks by controlling which resources can load and execute on your page. Think of it as a whitelist for your web page—anything not explicitly allowed is blocked.

CSP is one of the most effective defenses against XSS, but it requires careful configuration.

### What We'll Cover

- CSP headers and directives
- script-src, style-src, img-src
- Nonce-based CSP for inline scripts
- Report-uri for violation reporting
- CSP in meta tags

### Prerequisites

- Understanding of XSS attacks
- HTTP headers basics
- How browsers load resources

---

## CSP Basics

### How CSP Works

```
Browser receives page with CSP header:
Content-Security-Policy: script-src 'self'

Page tries to load:
✅ <script src="/app.js">         → Allowed (same origin)
❌ <script src="https://evil.com"> → Blocked (external)
❌ <script>alert('hi')</script>   → Blocked (inline)
```

### Setting CSP Headers

```javascript
// Express.js
app.use((req, res, next) => {
  res.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; script-src 'self'; style-src 'self'"
  );
  next();
});

// nginx
add_header Content-Security-Policy "default-src 'self'";

// Apache
Header set Content-Security-Policy "default-src 'self'"
```

### CSP Meta Tag

For static sites or when you can't set headers:

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self'">
```

> **Note:** Headers take precedence over meta tags. Some directives (like `frame-ancestors`) only work in headers.

---

## Key Directives

### Directive Overview

| Directive | Controls | Example |
|-----------|----------|---------|
| `default-src` | Fallback for all resources | `default-src 'self'` |
| `script-src` | JavaScript sources | `script-src 'self' cdn.example.com` |
| `style-src` | CSS sources | `style-src 'self' 'unsafe-inline'` |
| `img-src` | Image sources | `img-src 'self' data: https:` |
| `font-src` | Web font sources | `font-src 'self' fonts.googleapis.com` |
| `connect-src` | XHR, Fetch, WebSocket | `connect-src 'self' api.example.com` |
| `frame-src` | iframe sources | `frame-src youtube.com` |
| `object-src` | Plugins (Flash, Java) | `object-src 'none'` |
| `base-uri` | Base tag URLs | `base-uri 'self'` |
| `form-action` | Form submission targets | `form-action 'self'` |
| `frame-ancestors` | Who can frame this page | `frame-ancestors 'none'` |

### default-src

The fallback for any unspecified directive:

```http
# Everything from same origin
Content-Security-Policy: default-src 'self'

# This means:
# script-src 'self' (unless overridden)
# style-src 'self' (unless overridden)
# img-src 'self' (unless overridden)
# etc.
```

### script-src

Controls JavaScript execution:

```http
# Allow scripts from same origin + specific CDN
Content-Security-Policy: script-src 'self' https://cdn.example.com

# Allow inline scripts (not recommended)
Content-Security-Policy: script-src 'self' 'unsafe-inline'

# Allow eval() (not recommended)
Content-Security-Policy: script-src 'self' 'unsafe-eval'
```

### style-src

Controls CSS:

```http
# Allow same origin + inline styles
Content-Security-Policy: style-src 'self' 'unsafe-inline'

# Many sites need unsafe-inline for CSS frameworks
# Nonces are better but harder to implement for styles
```

### img-src

Controls images:

```http
# Allow same origin + data URLs + HTTPS images
Content-Security-Policy: img-src 'self' data: https:

# data: allows base64 embedded images
# https: allows any HTTPS image (careful!)
```

### connect-src

Controls network requests (fetch, XHR, WebSocket):

```http
# Allow API calls to specific domains
Content-Security-Policy: connect-src 'self' https://api.example.com wss://websocket.example.com
```

---

## Source Values

| Value | Meaning | Example |
|-------|---------|---------|
| `'self'` | Same origin | `script-src 'self'` |
| `'none'` | Block all | `object-src 'none'` |
| `'unsafe-inline'` | Allow inline code | `style-src 'unsafe-inline'` |
| `'unsafe-eval'` | Allow eval() | `script-src 'unsafe-eval'` |
| `https:` | Any HTTPS URL | `img-src https:` |
| `data:` | data: URLs | `img-src data:` |
| `blob:` | blob: URLs | `worker-src blob:` |
| `'nonce-xxx'` | Specific nonce | `script-src 'nonce-abc123'` |
| `'sha256-xxx'` | Specific hash | `script-src 'sha256-...'` |
| URL | Specific host | `script-src cdn.example.com` |

---

## Nonce-Based CSP

Nonces allow specific inline scripts while blocking others.

### How Nonces Work

```
1. Server generates random nonce: "abc123xyz"
2. Nonce added to CSP header: script-src 'nonce-abc123xyz'
3. Nonce added to allowed script tags: <script nonce="abc123xyz">
4. Browser executes ONLY scripts with matching nonce
```

### Implementation

```javascript
// Express.js with nonces
const crypto = require('crypto');

app.use((req, res, next) => {
  // Generate unique nonce per request
  res.locals.nonce = crypto.randomBytes(16).toString('base64');
  
  res.setHeader(
    'Content-Security-Policy',
    `script-src 'nonce-${res.locals.nonce}'`
  );
  next();
});

app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <body>
      <!-- This script executes -->
      <script nonce="${res.locals.nonce}">
        console.log('Allowed by nonce!');
      </script>
      
      <!-- This script is BLOCKED (no nonce) -->
      <script>
        console.log('Blocked!');
      </script>
    </body>
    </html>
  `);
});
```

### Nonce vs Hash

| Method | Use Case |
|--------|----------|
| **Nonce** | Dynamic pages where scripts can have nonce attribute |
| **Hash** | Static pages or when you can't add nonce attributes |

### Hash-Based CSP

```http
# Allow script with specific SHA-256 hash
Content-Security-Policy: script-src 'sha256-xyz123...'
```

Generate hash:

```bash
# Generate hash of script content
echo -n "console.log('hello')" | openssl sha256 -binary | base64
```

---

## Report-uri and report-to

Monitor CSP violations without breaking your site.

### Report-Only Mode

Test CSP without enforcing:

```http
# Report violations but don't block
Content-Security-Policy-Report-Only: default-src 'self'; report-uri /csp-violation
```

### Violation Reports

CSP sends JSON reports to your endpoint:

```javascript
// Express.js endpoint to receive reports
app.post('/csp-violation', express.json({ type: 'application/csp-report' }), (req, res) => {
  console.log('CSP Violation:', req.body['csp-report']);
  res.status(204).end();
});
```

Report format:

```json
{
  "csp-report": {
    "blocked-uri": "https://evil.com/script.js",
    "document-uri": "https://yoursite.com/page",
    "violated-directive": "script-src",
    "original-policy": "script-src 'self'"
  }
}
```

### Modern report-to

Newer reporting API (replacing report-uri):

```http
Content-Security-Policy: default-src 'self'; report-to csp-endpoint

# Also need Reporting-Endpoints header
Reporting-Endpoints: csp-endpoint="https://yoursite.com/csp-reports"
```

---

## Building a CSP

### Step-by-Step Approach

1. **Start with Report-Only**
   ```http
   Content-Security-Policy-Report-Only: default-src 'self'; report-uri /csp
   ```

2. **Monitor violations** in your logs

3. **Allowlist legitimate sources**
   ```http
   Content-Security-Policy-Report-Only: 
     default-src 'self';
     script-src 'self' https://cdn.example.com;
     img-src 'self' https: data:;
     report-uri /csp
   ```

4. **Switch to enforcing** when violations stop
   ```http
   Content-Security-Policy: default-src 'self'; ...
   ```

### Starter Template

```http
Content-Security-Policy: 
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self';
  connect-src 'self';
  frame-src 'none';
  object-src 'none';
  base-uri 'self';
  form-action 'self';
  frame-ancestors 'none';
  upgrade-insecure-requests;
```

### CSP for Common Scenarios

**Static Website:**
```http
Content-Security-Policy: 
  default-src 'self';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data:;
  frame-ancestors 'none';
  upgrade-insecure-requests
```

**React/Vue SPA:**
```http
Content-Security-Policy: 
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  connect-src 'self' https://api.yoursite.com;
  frame-ancestors 'none'
```

**With Third-Party Analytics:**
```http
Content-Security-Policy: 
  default-src 'self';
  script-src 'self' https://www.googletagmanager.com https://www.google-analytics.com;
  img-src 'self' https://www.google-analytics.com;
  connect-src 'self' https://www.google-analytics.com
```

---

## Common Pitfalls

### 1. Breaking Inline Styles

Many CSS frameworks use inline styles:

```http
# May break Bootstrap, Tailwind inline styles
style-src 'self'

# Often needed, but less secure
style-src 'self' 'unsafe-inline'
```

### 2. Forgetting connect-src

```javascript
// This will fail with strict CSP!
fetch('https://api.external.com/data')
```

```http
# Fix: add to connect-src
connect-src 'self' https://api.external.com
```

### 3. eval() in Libraries

Some libraries use `eval()`:

```http
# Required by some template engines
script-src 'self' 'unsafe-eval'

# Better: use libraries that don't need eval
```

### 4. data: URIs

```html
<!-- Needs img-src data: -->
<img src="data:image/png;base64,..." />
```

---

## Testing CSP

### Browser DevTools

1. Open DevTools → **Console**
2. CSP violations appear in red
3. Shows which directive blocked what

### Online Validators

- [CSP Evaluator](https://csp-evaluator.withgoogle.com/) by Google
- [Security Headers](https://securityheaders.com/)

### Automated Testing

```javascript
// Check CSP header in tests
const response = await fetch('/');
const csp = response.headers.get('content-security-policy');
expect(csp).toContain("default-src 'self'");
expect(csp).toContain("object-src 'none'");
```

---

## Hands-on Exercise

### Your Task

Create a CSP for a blog that:
1. Loads scripts from same origin only
2. Allows images from anywhere over HTTPS
3. Uses inline styles (for now)
4. Allows Google Fonts
5. Blocks all plugins
6. Prevents framing
7. Reports violations to `/csp-report`

<details>
<summary>✅ Solution</summary>

```http
Content-Security-Policy: 
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
  font-src 'self' https://fonts.gstatic.com;
  img-src 'self' https:;
  object-src 'none';
  frame-ancestors 'none';
  report-uri /csp-report
```
</details>

---

## Summary

✅ **CSP** is a whitelist for page resources
✅ **default-src** is the fallback for unspecified directives
✅ Use **nonces** for inline scripts instead of `'unsafe-inline'`
✅ **Report-Only** mode lets you test without breaking things
✅ **report-uri** sends violation reports for monitoring
✅ Start strict, then add sources as needed
✅ Avoid `'unsafe-eval'` and `'unsafe-inline'` when possible

**Next:** [Secure Communication](./03-secure-communication.md)

---

## Further Reading

- [MDN Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [CSP Quick Reference](https://content-security-policy.com/)
- [Google CSP Evaluator](https://csp-evaluator.withgoogle.com/)

<!-- 
Sources Consulted:
- MDN CSP: https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
- content-security-policy.com reference
-->
