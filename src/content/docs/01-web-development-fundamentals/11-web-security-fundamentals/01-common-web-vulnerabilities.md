---
title: "Common Web Vulnerabilities"
---

# Common Web Vulnerabilities

## Introduction

Most web attacks exploit the same fundamental weaknesses: trusting user input, insufficient validation, and missing security controls. Understanding these vulnerabilities is the first step to preventing them.

This lesson covers the most common web vulnerabilities you'll encounter and how to defend against them.

### What We'll Cover

- Cross-Site Scripting (XSS) types and prevention
- Cross-Site Request Forgery (CSRF) protection
- SQL Injection (awareness for full-stack)
- Clickjacking and frame-busting
- Open redirects

### Prerequisites

- JavaScript fundamentals
- Understanding of HTTP and cookies
- Basic HTML forms

---

## Cross-Site Scripting (XSS)

XSS is the #1 web vulnerability. Attackers inject malicious scripts that execute in victims' browsers.

### The Danger

An XSS attack can:
- Steal session cookies
- Capture keystrokes
- Redirect to phishing sites
- Modify page content
- Access local storage data

### XSS Types

| Type | How It Works | Example |
|------|--------------|---------|
| **Reflected** | Script in URL, reflected in response | `site.com/search?q=<script>alert('xss')</script>` |
| **Stored** | Script saved to database, served to users | Malicious comment on a blog |
| **DOM-based** | Script manipulates client-side DOM | `location.hash` injected into page |

### Reflected XSS

```javascript
// ❌ VULNERABLE: Server echoes input
// URL: /search?q=<script>steal(document.cookie)</script>

app.get('/search', (req, res) => {
  res.send(`<h1>Results for: ${req.query.q}</h1>`);
  // Script executes when page loads!
});
```

### Stored XSS

```javascript
// ❌ VULNERABLE: User input stored and displayed
// Attacker saves: <script>sendToServer(document.cookie)</script>

app.get('/comments', (req, res) => {
  const comments = db.getComments();
  res.send(comments.map(c => `<p>${c.text}</p>`).join(''));
  // Malicious script runs for every visitor!
});
```

### DOM-based XSS

```javascript
// ❌ VULNERABLE: Client-side injection
// URL: page.html#<img src=x onerror=alert('XSS')>

const userInput = location.hash.substring(1);
document.getElementById('output').innerHTML = userInput;
```

### XSS Prevention

**1. Output Encoding (Escaping)**

```javascript
// ✅ SAFE: Escape HTML entities
function escapeHTML(str) {
  return str.replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

// Use when inserting user data
element.innerHTML = escapeHTML(userInput);

// Even better: use textContent (no parsing)
element.textContent = userInput;
```

**2. Content Security Policy (CSP)**

```http
Content-Security-Policy: script-src 'self'; object-src 'none';
```

Blocks inline scripts and untrusted sources.

**3. Use Safe APIs**

```javascript
// ✅ SAFE: textContent doesn't parse HTML
element.textContent = userInput;

// ✅ SAFE: setAttribute for attributes
element.setAttribute('data-name', userInput);

// ❌ DANGEROUS: innerHTML parses HTML
element.innerHTML = userInput;
```

**4. Template Libraries**

Modern frameworks auto-escape by default:

```jsx
// React automatically escapes
return <div>{userInput}</div>; // Safe!

// Vue also escapes
<div>{{ userInput }}</div> <!-- Safe! -->
```

---

## Cross-Site Request Forgery (CSRF)

CSRF tricks authenticated users into performing unwanted actions.

### How CSRF Works

```
1. Victim logs into bank.com (cookie set)
2. Victim visits evil.com
3. evil.com contains:
   <form action="bank.com/transfer" method="POST">
     <input name="to" value="attacker">
     <input name="amount" value="10000">
   </form>
   <script>document.forms[0].submit();</script>
4. Browser sends request WITH bank.com cookie
5. Transfer happens without victim's knowledge!
```

### CSRF Prevention

**1. CSRF Tokens**

```html
<!-- Server generates unique token per session -->
<form action="/transfer" method="POST">
  <input type="hidden" name="csrf_token" value="abc123xyz">
  <input name="amount" value="100">
  <button>Transfer</button>
</form>
```

```javascript
// Server validates token
app.post('/transfer', (req, res) => {
  if (req.body.csrf_token !== req.session.csrfToken) {
    return res.status(403).send('Invalid CSRF token');
  }
  // Process transfer...
});
```

**2. SameSite Cookies**

```javascript
// Set SameSite attribute on cookies
res.cookie('session', sessionId, {
  httpOnly: true,
  secure: true,
  sameSite: 'strict'  // Or 'lax'
});
```

| SameSite Value | Behavior |
|----------------|----------|
| `Strict` | Never sent cross-site |
| `Lax` | Sent on top-level navigation (links) |
| `None` | Always sent (requires Secure) |

**3. Check Origin/Referer Headers**

```javascript
app.post('/api/sensitive', (req, res) => {
  const origin = req.headers.origin || req.headers.referer;
  if (!origin || !origin.startsWith('https://mysite.com')) {
    return res.status(403).send('Invalid origin');
  }
  // Process request...
});
```

---

## SQL Injection

SQL injection manipulates database queries by inserting malicious SQL.

### The Attack

```javascript
// ❌ VULNERABLE: String concatenation
const query = `SELECT * FROM users WHERE username = '${username}'`;

// Attacker input: ' OR '1'='1
// Resulting query: SELECT * FROM users WHERE username = '' OR '1'='1'
// Returns ALL users!

// Attacker input: '; DROP TABLE users; --
// Deletes entire table!
```

### Prevention: Parameterized Queries

```javascript
// ✅ SAFE: Parameterized query (Node.js + PostgreSQL)
const result = await pool.query(
  'SELECT * FROM users WHERE username = $1',
  [username]
);

// ✅ SAFE: Using ORM (Prisma)
const user = await prisma.user.findUnique({
  where: { username: username }
});
```

### Frontend Awareness

Even as a frontend developer:
- **Validate input** before sending to server
- **Use APIs** that return JSON, not raw SQL
- **Report** if you see string-concatenated queries

---

## Clickjacking

Attacker overlays invisible elements to trick users into clicking.

### The Attack

```html
<!-- Attacker's site -->
<style>
  .hidden-frame {
    position: absolute;
    opacity: 0;
    z-index: 2;
  }
  .fake-button {
    position: absolute;
    z-index: 1;
  }
</style>

<!-- Invisible iframe over fake button -->
<button class="fake-button">Win a Prize!</button>
<iframe class="hidden-frame" src="https://bank.com/transfer?amount=1000"></iframe>

<!-- User thinks they're clicking "Win a Prize" -->
<!-- Actually clicking transfer button in hidden iframe -->
```

### Prevention: Frame-Busting

**1. X-Frame-Options Header**

```http
# Prevent ALL framing
X-Frame-Options: DENY

# Allow same-origin framing only
X-Frame-Options: SAMEORIGIN
```

**2. CSP frame-ancestors**

```http
# Modern replacement for X-Frame-Options
Content-Security-Policy: frame-ancestors 'none';

# Allow specific origins
Content-Security-Policy: frame-ancestors 'self' https://trusted.com;
```

**3. JavaScript Frame-Busting (Fallback)**

```javascript
// If framed, redirect to self
if (window.self !== window.top) {
  window.top.location = window.self.location;
}
```

---

## Open Redirects

Attackers abuse redirect functionality to send users to malicious sites.

### The Attack

```
Legitimate: https://example.com/login?redirect=/dashboard
Malicious:  https://example.com/login?redirect=https://evil.com
```

User trusts `example.com`, clicks link, ends up on `evil.com`.

### Prevention

```javascript
// ❌ VULNERABLE: Redirect to any URL
app.get('/redirect', (req, res) => {
  res.redirect(req.query.url);
});

// ✅ SAFE: Allowlist of valid redirects
const ALLOWED_REDIRECTS = ['/dashboard', '/profile', '/settings'];

app.get('/redirect', (req, res) => {
  const target = req.query.url;
  
  if (ALLOWED_REDIRECTS.includes(target)) {
    res.redirect(target);
  } else {
    res.redirect('/');
  }
});

// ✅ SAFE: Only allow relative URLs
app.get('/redirect', (req, res) => {
  const target = req.query.url;
  
  // Must start with / and not //
  if (target.startsWith('/') && !target.startsWith('//')) {
    res.redirect(target);
  } else {
    res.redirect('/');
  }
});
```

---

## Security Checklist

### XSS Prevention

- [ ] Use `textContent` instead of `innerHTML`
- [ ] Escape user input in HTML context
- [ ] Implement Content Security Policy
- [ ] Use templating libraries with auto-escaping

### CSRF Prevention

- [ ] Use CSRF tokens for state-changing requests
- [ ] Set `SameSite` attribute on cookies
- [ ] Validate Origin/Referer headers

### Injection Prevention

- [ ] Use parameterized queries (never concatenate)
- [ ] Validate and sanitize all input
- [ ] Use ORMs with built-in protection

### Framing Prevention

- [ ] Set `X-Frame-Options: DENY`
- [ ] Use `CSP: frame-ancestors 'none'`

### Redirect Safety

- [ ] Validate redirect URLs against allowlist
- [ ] Only allow relative URLs
- [ ] Never trust user-supplied redirect targets

---

## Hands-on Exercise

### Your Task

Identify vulnerabilities in this code:

```javascript
// Express.js route
app.get('/profile', (req, res) => {
  const name = req.query.name;
  res.send(`
    <h1>Welcome, ${name}!</h1>
    <a href="${req.query.next}">Continue</a>
  `);
});
```

### Questions

1. What type of XSS vulnerability exists?
2. What other vulnerability is present?
3. How would you fix both issues?

<details>
<summary>✅ Solution</summary>

**Vulnerabilities:**
1. **Reflected XSS** - `name` parameter is not escaped
2. **Open Redirect** - `next` parameter is not validated

**Fixed code:**
```javascript
const escapeHTML = require('escape-html');
const ALLOWED_PATHS = ['/dashboard', '/settings', '/home'];

app.get('/profile', (req, res) => {
  const name = escapeHTML(req.query.name || 'Guest');
  const next = ALLOWED_PATHS.includes(req.query.next) 
    ? req.query.next 
    : '/';
    
  res.send(`
    <h1>Welcome, ${name}!</h1>
    <a href="${next}">Continue</a>
  `);
});
```
</details>

---

## Summary

✅ **XSS** injects malicious scripts—prevent with encoding and CSP
✅ **CSRF** forges user actions—prevent with tokens and SameSite cookies
✅ **SQL Injection** manipulates queries—use parameterized queries
✅ **Clickjacking** uses hidden frames—block with X-Frame-Options
✅ **Open Redirects** abuse trust—validate against allowlist
✅ **Never trust user input**—validate and sanitize everything

**Next:** [Content Security Policy](./02-content-security-policy.md)

---

## Further Reading

- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
- [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
- [MDN Cross-Site Scripting](https://developer.mozilla.org/en-US/docs/Glossary/Cross-site_scripting)

<!-- 
Sources Consulted:
- OWASP Cheat Sheets: https://cheatsheetseries.owasp.org/
- MDN Web Security: https://developer.mozilla.org/en-US/docs/Web/Security
-->
