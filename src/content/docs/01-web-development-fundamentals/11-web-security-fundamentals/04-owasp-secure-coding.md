---
title: "OWASP & Secure Coding"
---

# OWASP & Secure Coding

## Introduction

The **Open Web Application Security Project (OWASP)** provides free, vendor-neutral security resources. Their **Top 10** is the definitive list of critical web security risks, used by organizations worldwide to prioritize security efforts.

This lesson covers the OWASP Top 10, security testing basics, and secure coding practices to build into your daily development workflow.

### What We'll Cover

- Understanding OWASP
- Top vulnerabilities overview
- Security testing basics
- Secure coding practices

### Prerequisites

- Understanding of common vulnerabilities (XSS, CSRF)
- Basic web development experience
- Familiarity with authentication concepts

---

## Understanding OWASP

### What is OWASP?

OWASP is:
- **Non-profit foundation** for application security
- **Community-driven** with open content
- **Industry standard** for security guidance
- **Vendor-neutral** and free to use

### Key OWASP Resources

| Resource | Description | URL |
|----------|-------------|-----|
| **Top 10** | Critical security risks | owasp.org/Top10 |
| **Cheat Sheets** | Practical security guidance | cheatsheetseries.owasp.org |
| **ASVS** | Verification standard | owasp.org/ASVS |
| **Testing Guide** | Security testing methodology | owasp.org/wstg |
| **ZAP** | Free security scanner | zaproxy.org |

---

## OWASP Top 10 (2021)

The most critical web application security risks:

### Overview

| # | Category | Description |
|---|----------|-------------|
| A01 | **Broken Access Control** | Users acting outside permissions |
| A02 | **Cryptographic Failures** | Sensitive data exposure |
| A03 | **Injection** | XSS, SQL injection, command injection |
| A04 | **Insecure Design** | Missing security controls |
| A05 | **Security Misconfiguration** | Improper setup, defaults |
| A06 | **Vulnerable Components** | Using outdated dependencies |
| A07 | **Authentication Failures** | Broken auth/session management |
| A08 | **Software & Data Integrity** | Untrusted updates, CI/CD attacks |
| A09 | **Security Logging Failures** | Insufficient monitoring |
| A10 | **SSRF** | Server-side request forgery |

---

### A01: Broken Access Control

Users accessing what they shouldn't.

```javascript
// ❌ VULNERABLE: No authorization check
app.get('/api/users/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json(user);  // Any user can access any profile!
});

// ✅ SECURE: Verify authorization
app.get('/api/users/:id', async (req, res) => {
  // Check if user can access this resource
  if (req.user.id !== req.params.id && !req.user.isAdmin) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  const user = await db.getUser(req.params.id);
  res.json(user);
});
```

**Prevention:**
- Deny by default
- Implement access control at controller/route level
- Disable directory listing
- Log access control failures

---

### A02: Cryptographic Failures

Sensitive data not properly protected.

```javascript
// ❌ BAD: Storing passwords in plain text
const user = { email, password };  // NO!

// ✅ GOOD: Hash passwords
const bcrypt = require('bcrypt');
const hashedPassword = await bcrypt.hash(password, 12);
const user = { email, password: hashedPassword };

// ✅ GOOD: Compare passwords securely
const isMatch = await bcrypt.compare(inputPassword, user.password);
```

**Prevention:**
- Classify data by sensitivity
- Encrypt sensitive data at rest
- Use strong algorithms (AES-256, bcrypt, argon2)
- Don't store sensitive data unnecessarily
- Use HTTPS everywhere

---

### A03: Injection

Untrusted data sent to an interpreter.

```javascript
// ❌ SQL INJECTION
const query = `SELECT * FROM users WHERE email = '${email}'`;

// ✅ PARAMETERIZED QUERY
const result = await db.query(
  'SELECT * FROM users WHERE email = $1',
  [email]
);

// ❌ XSS
element.innerHTML = userInput;

// ✅ SAFE DOM API
element.textContent = userInput;
```

**Prevention:**
- Use parameterized queries
- Validate and sanitize all input
- Escape output based on context
- Use CSP to mitigate XSS impact

---

### A04: Insecure Design

Security flaws in the design itself.

```javascript
// ❌ INSECURE DESIGN: No rate limiting
app.post('/api/password-reset', async (req, res) => {
  // Attacker can try millions of reset codes!
  if (req.body.code === storedCode) {
    // Reset password
  }
});

// ✅ SECURE DESIGN: Rate limit + account lockout
const rateLimit = require('express-rate-limit');

const resetLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 5,                     // 5 attempts
  message: 'Too many attempts, try again later'
});

app.post('/api/password-reset', resetLimiter, async (req, res) => {
  // Now protected against brute force
});
```

**Prevention:**
- Threat modeling during design
- Security requirements in user stories
- Rate limiting and lockouts
- Security review of architecture

---

### A05: Security Misconfiguration

Improper configuration of security controls.

```javascript
// ❌ EXPOSING STACK TRACES
app.use((err, req, res, next) => {
  res.status(500).json({ 
    error: err.message,
    stack: err.stack  // Leaks internal details!
  });
});

// ✅ HIDE DETAILS IN PRODUCTION
app.use((err, req, res, next) => {
  console.error(err);  // Log for debugging
  res.status(500).json({ 
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : err.message
  });
});
```

**Prevention:**
- Remove default accounts
- Disable unnecessary features
- Hide error details in production
- Review cloud/hosting configurations
- Automate configuration checks

---

### A06: Vulnerable Components

Using outdated or vulnerable dependencies.

```bash
# Check for vulnerabilities
npm audit

# See outdated packages
npm outdated

# Update packages
npm update

# Fix vulnerabilities automatically
npm audit fix
```

**Prevention:**
- Regularly update dependencies
- Remove unused dependencies
- Use tools: npm audit, Snyk, Dependabot
- Monitor security advisories
- Lock dependency versions

---

### A07: Authentication Failures

Weak authentication mechanisms.

```javascript
// ❌ WEAK: Short session, no rotation
req.session.cookie.maxAge = 24 * 60 * 60 * 1000;

// ✅ STRONG: Session configuration
app.use(session({
  secret: process.env.SESSION_SECRET,
  name: 'sessionId',              // Don't use default name
  cookie: {
    httpOnly: true,               // No JS access
    secure: true,                 // HTTPS only
    sameSite: 'strict',           // CSRF protection
    maxAge: 3600000               // 1 hour
  },
  resave: false,
  saveUninitialized: false
}));

// ✅ Rotate session on privilege change
app.post('/login', async (req, res) => {
  const user = await authenticate(req.body);
  if (user) {
    req.session.regenerate(() => {  // New session ID
      req.session.userId = user.id;
      res.redirect('/dashboard');
    });
  }
});
```

**Prevention:**
- Strong password requirements
- Multi-factor authentication
- Rate limit login attempts
- Secure session management
- Don't expose session IDs in URLs

---

### A08: Software & Data Integrity

Trusting unverified updates or data.

```html
<!-- ❌ RISKY: No integrity check -->
<script src="https://cdn.example.com/lib.js"></script>

<!-- ✅ SAFE: Subresource Integrity (SRI) -->
<script 
  src="https://cdn.example.com/lib.js"
  integrity="sha384-abc123..."
  crossorigin="anonymous">
</script>
```

```bash
# Generate SRI hash
openssl dgst -sha384 -binary lib.js | openssl base64 -A
```

**Prevention:**
- Use SRI for CDN resources
- Verify package signatures
- Secure CI/CD pipelines
- Code review for all changes

---

### A09: Security Logging Failures

Not detecting or responding to attacks.

```javascript
// ✅ Log security events
const logger = require('./logger');

app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  const user = await authenticate(email, password);
  
  if (user) {
    logger.info('Login successful', { 
      userId: user.id, 
      ip: req.ip 
    });
    // ... success handling
  } else {
    logger.warn('Login failed', { 
      email,  // Don't log password!
      ip: req.ip,
      userAgent: req.headers['user-agent']
    });
    // ... failure handling
  }
});

// ✅ Log access control failures
app.get('/admin', (req, res) => {
  if (!req.user.isAdmin) {
    logger.warn('Unauthorized admin access attempt', {
      userId: req.user.id,
      ip: req.ip
    });
    return res.status(403).send('Forbidden');
  }
  // ... admin page
});
```

**Prevention:**
- Log authentication events
- Log access control failures
- Log input validation failures
- Ensure logs are tamper-proof
- Set up alerting for suspicious patterns

---

### A10: Server-Side Request Forgery (SSRF)

Tricking server into making unintended requests.

```javascript
// ❌ VULNERABLE: User controls URL
app.get('/fetch', async (req, res) => {
  const response = await fetch(req.query.url);  // Dangerous!
  res.send(await response.text());
});

// Attacker: /fetch?url=http://169.254.169.254/meta-data/
// Accesses AWS metadata service!

// ✅ SAFE: Allowlist of valid URLs
const ALLOWED_HOSTS = ['api.example.com', 'cdn.example.com'];

app.get('/fetch', async (req, res) => {
  const url = new URL(req.query.url);
  
  if (!ALLOWED_HOSTS.includes(url.hostname)) {
    return res.status(400).send('Invalid URL');
  }
  
  const response = await fetch(url);
  res.send(await response.text());
});
```

**Prevention:**
- Validate and sanitize user-supplied URLs
- Use allowlists for permitted destinations
- Block internal IP ranges (10.x, 169.254.x, 127.x)
- Disable unnecessary URL schemes

---

## Security Testing Basics

### Testing Types

| Type | When | Tools |
|------|------|-------|
| **Static Analysis (SAST)** | During development | ESLint security rules, Semgrep |
| **Dynamic Analysis (DAST)** | Running application | OWASP ZAP, Burp Suite |
| **Dependency Scanning** | CI/CD | npm audit, Snyk, Dependabot |
| **Penetration Testing** | Before release | Manual testing, bug bounty |

### OWASP ZAP

Free, open-source security scanner:

1. Download from zaproxy.org
2. Configure browser proxy
3. Browse your application (ZAP records)
4. Run automated scans
5. Review findings

### ESLint Security Rules

```bash
npm install --save-dev eslint-plugin-security
```

```javascript
// .eslintrc.js
module.exports = {
  plugins: ['security'],
  extends: ['plugin:security/recommended']
};
```

Catches:
- `eval()` usage
- `child_process` with user input
- Regex DoS patterns

---

## Secure Coding Practices

### Input Validation

```javascript
// ✅ Validate all input
const { body, validationResult } = require('express-validator');

app.post('/register', [
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 8 }),
  body('age').isInt({ min: 13, max: 120 })
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  // Process valid input
});
```

### Secure Defaults

```javascript
// ✅ Security by default
const secureConfig = {
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict'
  },
  trustProxy: false,
  exposedHeaders: []  // Minimize info exposure
};
```

### Principle of Least Privilege

```javascript
// ✅ Minimal permissions
const dbUser = {
  // Read-only for public endpoints
  public: { 
    permissions: ['SELECT'] 
  },
  // Write for authenticated users
  authenticated: { 
    permissions: ['SELECT', 'INSERT', 'UPDATE'] 
  },
  // Admin only for administrative tasks
  admin: { 
    permissions: ['SELECT', 'INSERT', 'UPDATE', 'DELETE'] 
  }
};
```

### Defense in Depth

```javascript
// Multiple layers of protection
app.post('/transfer', [
  // Layer 1: Authentication
  requireAuth,
  
  // Layer 2: Rate limiting
  rateLimit({ windowMs: 60000, max: 10 }),
  
  // Layer 3: Input validation
  body('amount').isFloat({ min: 0.01, max: 10000 }),
  body('recipient').isEmail(),
  
  // Layer 4: CSRF protection
  csrfProtection,
  
  // Layer 5: Authorization
  requireRole('user'),
  
  // Layer 6: Business logic validation
  validateSufficientFunds,
  
], transferHandler);
```

---

## Security Checklist

### Development Phase

- [ ] Validate and sanitize all input
- [ ] Use parameterized queries
- [ ] Escape output based on context
- [ ] Implement proper authentication
- [ ] Apply least privilege principle
- [ ] Keep dependencies updated
- [ ] Use security linters

### Code Review

- [ ] Check for hardcoded secrets
- [ ] Verify access control on all endpoints
- [ ] Review error handling (no info leaks)
- [ ] Check for injection vulnerabilities
- [ ] Verify crypto usage (no custom crypto)

### Deployment

- [ ] HTTPS everywhere
- [ ] Security headers configured
- [ ] Debug mode disabled
- [ ] Default credentials changed
- [ ] Unnecessary features disabled
- [ ] Logging and monitoring in place

---

## Hands-on Exercise

### Your Task

Review this code for security issues:

```javascript
const express = require('express');
const app = express();

app.get('/user', (req, res) => {
  const id = req.query.id;
  const query = `SELECT * FROM users WHERE id = ${id}`;
  
  db.query(query, (err, result) => {
    if (err) {
      res.send(`Error: ${err.message}`);
    } else {
      res.send(result);
    }
  });
});

app.listen(3000);
```

**Find:**
1. What vulnerabilities exist?
2. Which OWASP Top 10 categories?
3. How would you fix them?

<details>
<summary>✅ Solution</summary>

**Vulnerabilities:**
1. **SQL Injection (A03)** - String concatenation in query
2. **Security Misconfiguration (A05)** - Error message exposes details
3. **Broken Access Control (A01)** - No authentication check

**Fixed code:**
```javascript
const express = require('express');
const app = express();

// Add authentication middleware
const requireAuth = require('./auth');

app.get('/user', requireAuth, async (req, res) => {
  try {
    const id = req.query.id;
    
    // Validate input
    if (!Number.isInteger(Number(id))) {
      return res.status(400).json({ error: 'Invalid ID' });
    }
    
    // Check authorization
    if (req.user.id !== Number(id) && !req.user.isAdmin) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Parameterized query
    const result = await db.query(
      'SELECT id, name, email FROM users WHERE id = $1',
      [id]
    );
    
    res.json(result.rows[0]);
  } catch (err) {
    console.error(err);  // Log for debugging
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(3000);
```
</details>

---

## Summary

✅ **OWASP Top 10** is the industry standard for web security risks
✅ **Broken Access Control** is #1—always verify authorization
✅ **Injection** remains critical—use parameterized queries
✅ **Keep dependencies updated**—use automated scanning
✅ **Defense in depth**—multiple layers of protection
✅ **Log security events**—detect and respond to attacks
✅ Use **OWASP resources**: Cheat Sheets, ZAP, ASVS

**Back to:** [Web Security Fundamentals Overview](./00-web-security-fundamentals.md)

---

## Further Reading

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP ZAP](https://www.zaproxy.org/)

<!-- 
Sources Consulted:
- OWASP Top 10 2021: https://owasp.org/www-project-top-ten/
- OWASP Cheat Sheets: https://cheatsheetseries.owasp.org/
-->
