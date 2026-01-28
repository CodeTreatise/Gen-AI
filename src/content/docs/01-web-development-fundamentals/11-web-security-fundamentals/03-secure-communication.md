---
title: "Secure Communication"
---

# Secure Communication

## Introduction

Data in transit is vulnerable. Without encryption, attackers can intercept credentials, session tokens, and sensitive information. **HTTPS** protects data between browser and server, while security headers like **HSTS** ensure browsers always use encryption.

This lesson covers how to ensure all communication is secure.

### What We'll Cover

- HTTPS and TLS basics
- Mixed content issues
- HSTS (HTTP Strict Transport Security)
- Certificate pinning concepts

### Prerequisites

- HTTP fundamentals
- Basic understanding of encryption concepts
- How browsers communicate with servers

---

## HTTPS and TLS

### The Problem with HTTP

HTTP sends everything in plain text:

```
Browser → Router → ISP → Server

         ↓ Anyone in the middle can see:
         
GET /login HTTP/1.1
Cookie: session=abc123
Content-Type: application/x-www-form-urlencoded

username=alice&password=secret123
```

### How HTTPS Protects

HTTPS = HTTP + TLS (Transport Layer Security)

```
Browser ═══ TLS Encrypted Tunnel ═══ Server

         ↓ Attackers see only:
         
[Encrypted gibberish]
hJ3kL9mN...xYz
```

### The TLS Handshake

```
1. Browser: "Hello, I support TLS 1.3, these ciphers..."
2. Server: "Let's use TLS 1.3, here's my certificate"
3. Browser: Validates certificate against trusted CAs
4. Both: Exchange keys using asymmetric encryption
5. Both: Switch to faster symmetric encryption
6. All data: Encrypted with shared secret
```

```
┌──────────┐                           ┌──────────┐
│  Browser │                           │  Server  │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │──── ClientHello (supported ciphers) ──→│
     │                                      │
     │←─── ServerHello + Certificate ───────│
     │                                      │
     │──── Key Exchange (encrypted) ────────→│
     │                                      │
     ├══════════════════════════════════════┤
     │     Encrypted Communication          │
     ├══════════════════════════════════════┤
```

### TLS Versions

| Version | Status | Notes |
|---------|--------|-------|
| SSL 2.0/3.0 | ❌ Deprecated | Vulnerable, don't use |
| TLS 1.0 | ❌ Deprecated | Disable in production |
| TLS 1.1 | ❌ Deprecated | Disable in production |
| TLS 1.2 | ✅ Minimum | Still secure, widely supported |
| TLS 1.3 | ✅ Preferred | Faster handshake, better security |

### Getting HTTPS

For development:
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

For production:
- **Let's Encrypt** - Free, automated certificates
- **Cloudflare** - Free SSL with CDN
- **AWS Certificate Manager** - Free for AWS resources

---

## Mixed Content

Mixed content occurs when HTTPS pages load HTTP resources.

### The Problem

```html
<!-- Page loaded over HTTPS -->
<!DOCTYPE html>
<html>
<head>
  <!-- ❌ BLOCKED: HTTP script on HTTPS page -->
  <script src="http://example.com/app.js"></script>
  
  <!-- ⚠️ WARNING: HTTP image (may be blocked) -->
  <img src="http://example.com/photo.jpg">
</head>
```

### Mixed Content Types

| Type | Examples | Browser Behavior |
|------|----------|------------------|
| **Active** | Scripts, stylesheets, iframes, XHR | **Blocked** by default |
| **Passive** | Images, audio, video | Warning, may load |

### Fixing Mixed Content

**1. Use Protocol-Relative URLs (legacy)**
```html
<!-- Inherits protocol from parent page -->
<script src="//cdn.example.com/app.js"></script>
```

**2. Use HTTPS (preferred)**
```html
<!-- Always use HTTPS -->
<script src="https://cdn.example.com/app.js"></script>
```

**3. upgrade-insecure-requests**
```http
Content-Security-Policy: upgrade-insecure-requests
```

Automatically upgrades HTTP to HTTPS:
```html
<!-- Browser rewrites to HTTPS automatically -->
<img src="http://example.com/photo.jpg">
<!-- Becomes -->
<img src="https://example.com/photo.jpg">
```

### Finding Mixed Content

1. Open DevTools → **Console**
2. Look for mixed content warnings/errors
3. Check **Network** tab for blocked requests

```javascript
// In Console - find all HTTP resources
Array.from(document.querySelectorAll('[src], [href]'))
  .filter(el => (el.src || el.href)?.startsWith('http://'))
  .forEach(el => console.log(el));
```

---

## HSTS (HTTP Strict Transport Security)

HSTS tells browsers to **always** use HTTPS, even if user types `http://`.

### The Problem Without HSTS

```
1. User types: example.com (no protocol)
2. Browser requests: http://example.com
3. Server redirects: → https://example.com
4. ⚠️ Gap: Initial HTTP request is vulnerable!
```

### With HSTS

```
1. User types: example.com
2. Browser remembers HSTS: "Always use HTTPS"
3. Browser requests: https://example.com directly
4. ✅ No vulnerable HTTP request!
```

### Setting HSTS

```http
# Basic HSTS (1 year)
Strict-Transport-Security: max-age=31536000

# Include subdomains
Strict-Transport-Security: max-age=31536000; includeSubDomains

# Preload list submission
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

```javascript
// Express.js
app.use((req, res, next) => {
  res.setHeader(
    'Strict-Transport-Security',
    'max-age=31536000; includeSubDomains'
  );
  next();
});
```

### HSTS Directives

| Directive | Meaning |
|-----------|---------|
| `max-age` | How long to remember (seconds) |
| `includeSubDomains` | Apply to all subdomains |
| `preload` | Eligible for browser preload list |

### HSTS Preload List

Get your site hardcoded into browsers:

1. Meet requirements:
   - Valid HTTPS certificate
   - Redirect HTTP to HTTPS
   - HSTS header with `preload`
   - All subdomains support HTTPS

2. Submit at [hstspreload.org](https://hstspreload.org/)

> **Warning:** Preloading is difficult to undo. Ensure HTTPS works for ALL subdomains before submitting.

---

## Certificate Concepts

### How Certificates Work

```
┌────────────────────────────────────────────────┐
│              Certificate Authority (CA)         │
│          (DigiCert, Let's Encrypt, etc.)       │
└───────────────────┬────────────────────────────┘
                    │ Issues certificate
                    ▼
┌────────────────────────────────────────────────┐
│              Server Certificate                 │
│  - Domain: example.com                         │
│  - Public Key                                  │
│  - Validity Period                             │
│  - CA Signature                                │
└───────────────────┬────────────────────────────┘
                    │ Browser validates against
                    ▼
┌────────────────────────────────────────────────┐
│         Browser's Trusted CA List              │
│     (Built into OS and browser)                │
└────────────────────────────────────────────────┘
```

### Certificate Validation

Browser checks:
1. ✅ Certificate is for correct domain
2. ✅ Certificate is not expired
3. ✅ Certificate is signed by trusted CA
4. ✅ Certificate is not revoked

### Certificate Types

| Type | Validation Level | Use Case |
|------|------------------|----------|
| **DV** (Domain Validation) | Domain ownership only | Personal sites, APIs |
| **OV** (Organization Validation) | Domain + organization | Business sites |
| **EV** (Extended Validation) | Domain + thorough org check | Banks, high-trust sites |

> **Note:** All three types provide the same level of encryption. The difference is verification of identity.

### Certificate Pinning

Pinning limits which certificates your app accepts, preventing CA compromise.

```javascript
// Node.js example - Pin specific certificate
const https = require('https');
const tls = require('tls');

const EXPECTED_FINGERPRINT = 'AA:BB:CC:DD...';

const options = {
  hostname: 'api.example.com',
  port: 443,
  checkServerIdentity: (host, cert) => {
    const fingerprint = cert.fingerprint256;
    if (fingerprint !== EXPECTED_FINGERPRINT) {
      throw new Error('Certificate fingerprint mismatch!');
    }
  }
};
```

> **Warning:** Certificate pinning requires careful maintenance. When certificates rotate, you must update pins or users get locked out.

---

## Additional Security Headers

### Full Security Headers Set

```http
# Force HTTPS
Strict-Transport-Security: max-age=31536000; includeSubDomains

# Prevent XSS
Content-Security-Policy: default-src 'self'

# Block framing
X-Frame-Options: DENY

# Prevent MIME sniffing
X-Content-Type-Options: nosniff

# Control referrer
Referrer-Policy: strict-origin-when-cross-origin

# Restrict browser features
Permissions-Policy: geolocation=(), camera=(), microphone=()
```

### Express.js Helmet

Use [Helmet](https://helmetjs.github.io/) for easy security headers:

```javascript
const helmet = require('helmet');

app.use(helmet());

// Or configure individually
app.use(helmet.hsts({ maxAge: 31536000 }));
app.use(helmet.noSniff());
app.use(helmet.frameguard({ action: 'deny' }));
```

---

## Testing Secure Communication

### Online Tools

| Tool | What It Tests |
|------|---------------|
| [SSL Labs](https://www.ssllabs.com/ssltest/) | TLS configuration, certificate |
| [Security Headers](https://securityheaders.com/) | HTTP security headers |
| [Mozilla Observatory](https://observatory.mozilla.org/) | Overall security posture |

### Browser DevTools

1. Click lock icon in address bar
2. View certificate details
3. Check **Security** tab in DevTools

### Command Line

```bash
# Check certificate
openssl s_client -connect example.com:443 -servername example.com

# Test TLS version support
openssl s_client -connect example.com:443 -tls1_3

# Check headers
curl -I https://example.com
```

---

## Implementation Checklist

### HTTPS Setup

- [ ] Obtain valid TLS certificate
- [ ] Configure TLS 1.2 minimum, prefer TLS 1.3
- [ ] Redirect all HTTP to HTTPS (301)
- [ ] Update all internal links to HTTPS

### Mixed Content

- [ ] Scan for HTTP resources on HTTPS pages
- [ ] Update all resource URLs to HTTPS
- [ ] Add `upgrade-insecure-requests` CSP directive

### HSTS

- [ ] Add HSTS header with sufficient max-age
- [ ] Test before adding `includeSubDomains`
- [ ] Consider preloading (only if all subdomains ready)

### Headers

- [ ] Set `X-Content-Type-Options: nosniff`
- [ ] Set `X-Frame-Options: DENY`
- [ ] Configure `Referrer-Policy`
- [ ] Review `Permissions-Policy` for unused features

---

## Hands-on Exercise

### Your Task

Check the security of any website:

1. Visit [SSL Labs](https://www.ssllabs.com/ssltest/)
2. Enter a domain (try your favorite sites)
3. Review the report:
   - What grade did it get?
   - What TLS versions are supported?
   - Are there any warnings?

4. Visit [Security Headers](https://securityheaders.com/)
5. Enter the same domain
6. Review which headers are missing

### Challenge

Configure a simple Node.js server with proper security headers:

```javascript
const express = require('express');
const app = express();

// Add your security headers here

app.get('/', (req, res) => {
  res.send('Secure server!');
});

app.listen(3000);
```

<details>
<summary>✅ Solution</summary>

```javascript
const express = require('express');
const app = express();

app.use((req, res, next) => {
  // HSTS
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  
  // CSP
  res.setHeader('Content-Security-Policy', "default-src 'self'");
  
  // Prevent framing
  res.setHeader('X-Frame-Options', 'DENY');
  
  // Prevent MIME sniffing
  res.setHeader('X-Content-Type-Options', 'nosniff');
  
  // Referrer policy
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // Permissions policy
  res.setHeader('Permissions-Policy', 'geolocation=(), camera=()');
  
  next();
});

app.get('/', (req, res) => {
  res.send('Secure server!');
});

app.listen(3000);
```
</details>

---

## Summary

✅ **HTTPS** encrypts all data in transit with TLS
✅ Use **TLS 1.2 minimum**, prefer TLS 1.3
✅ **Mixed content** breaks security—upgrade all resources to HTTPS
✅ **HSTS** forces browsers to always use HTTPS
✅ Consider **HSTS preloading** for maximum protection
✅ Use security header libraries like **Helmet** for easy implementation
✅ Test with **SSL Labs** and **Security Headers**

**Next:** [OWASP & Secure Coding](./04-owasp-secure-coding.md)

---

## Further Reading

- [MDN HTTPS](https://developer.mozilla.org/en-US/docs/Web/Security/Transport_Layer_Security)
- [HSTS Preload List](https://hstspreload.org/)
- [SSL Labs](https://www.ssllabs.com/ssltest/)
- [Let's Encrypt](https://letsencrypt.org/)

<!-- 
Sources Consulted:
- MDN TLS: https://developer.mozilla.org/en-US/docs/Web/Security/Transport_Layer_Security
- OWASP TLS Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Security_Cheat_Sheet.html
-->
