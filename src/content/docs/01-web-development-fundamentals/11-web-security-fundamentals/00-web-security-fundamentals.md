---
title: "Web Security Fundamentals"
---

# Web Security Fundamentals

## Overview

Security isn't optional—it's essential. A single vulnerability can expose user data, damage your reputation, and result in legal consequences. Understanding web security helps you build applications that protect users from malicious actors.

This lesson covers the most common attacks (XSS, CSRF, injection) and the defenses that stop them (CSP, HTTPS, secure headers).

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-common-web-vulnerabilities.md) | Common Web Vulnerabilities | XSS, CSRF, SQL Injection, Clickjacking |
| [02](./02-content-security-policy.md) | Content Security Policy | CSP headers, directives, nonces |
| [03](./03-secure-communication.md) | Secure Communication | HTTPS, TLS, HSTS, mixed content |
| [04](./04-owasp-secure-coding.md) | OWASP & Secure Coding | Top 10, security testing, best practices |

---

## Why Security Matters

### The Stakes

| Impact | Example |
|--------|---------|
| **Data breach** | User credentials exposed |
| **Financial loss** | Fraud, regulatory fines |
| **Reputation damage** | Loss of user trust |
| **Legal liability** | GDPR, CCPA violations |
| **Service disruption** | DDoS, ransomware |

### The Attacker's Perspective

Attackers look for the easiest path:
- Unvalidated user input
- Missing security headers
- Outdated dependencies
- Exposed secrets in code
- Weak authentication

---

## Key Security Principles

### Defense in Depth

Multiple layers of protection:

```
┌─────────────────────────────────────────────┐
│              Network Firewall               │
├─────────────────────────────────────────────┤
│          Application Firewall (WAF)         │
├─────────────────────────────────────────────┤
│    Secure Headers (CSP, HSTS, X-Frame)     │
├─────────────────────────────────────────────┤
│         Input Validation & Sanitization     │
├─────────────────────────────────────────────┤
│     Authentication & Authorization          │
└─────────────────────────────────────────────┘
```

### Principle of Least Privilege

- Grant minimum necessary permissions
- Users: only access what they need
- Code: only request needed APIs
- Tokens: scope to required operations

### Trust No Input

Assume all input is malicious:
- User form data
- URL parameters
- Cookies
- HTTP headers
- File uploads
- API responses

---

## Security Headers Quick Reference

```http
# Prevent XSS
Content-Security-Policy: default-src 'self';

# Force HTTPS
Strict-Transport-Security: max-age=31536000; includeSubDomains

# Prevent framing (clickjacking)
X-Frame-Options: DENY

# Block MIME sniffing
X-Content-Type-Options: nosniff

# Control referrer info
Referrer-Policy: strict-origin-when-cross-origin

# Limit browser features
Permissions-Policy: geolocation=(), camera=()
```

---

## Common Attack Types

| Attack | Target | Prevention |
|--------|--------|------------|
| **XSS** | Execute malicious scripts | CSP, output encoding |
| **CSRF** | Forge user actions | CSRF tokens, SameSite cookies |
| **SQL Injection** | Manipulate database | Parameterized queries |
| **Clickjacking** | Hidden malicious UI | X-Frame-Options, CSP |
| **MitM** | Intercept communication | HTTPS, HSTS |

---

## Prerequisites

Before starting this lesson:
- HTML/CSS/JavaScript fundamentals
- Understanding of HTTP requests/responses
- Basic server-side concepts (helpful but not required)

---

## Start Learning

Begin with [Common Web Vulnerabilities](./01-common-web-vulnerabilities.md) to understand XSS, CSRF, and injection attacks.

---

## Further Reading

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [MDN Web Security](https://developer.mozilla.org/en-US/docs/Web/Security)
- [web.dev Safe and Secure](https://web.dev/secure/)
