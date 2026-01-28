---
title: "Accessibility Principles"
---

# Accessibility Principles

## Introduction

Web accessibility is guided by international standards, primarily the **Web Content Accessibility Guidelines (WCAG)**. Understanding these principles helps you build accessible sites from the start, rather than retrofitting later.

This lesson covers WCAG guidelines, POUR principles, legal requirements, and how browsers expose content to assistive technologies.

### What We'll Cover

- WCAG guidelines overview (A, AA, AAA)
- POUR principles (Perceivable, Operable, Understandable, Robust)
- Legal requirements (ADA, Section 508)
- Accessibility tree

### Prerequisites

- HTML fundamentals
- Understanding of how browsers render pages

---

## WCAG Guidelines

### What is WCAG?

**Web Content Accessibility Guidelines (WCAG)** are the international standard for web accessibility, created by the W3C.

Current version: **WCAG 2.2** (October 2023)

### Conformance Levels

| Level | Description | Target |
|-------|-------------|--------|
| **A** | Minimum accessibility | Must have |
| **AA** | Addresses major barriers | Industry standard |
| **AAA** | Highest accessibility | Specialized needs |

> **Note:** Most organizations target **Level AA**. Level AAA is often impractical for entire sites but useful for specific content.

### Key WCAG 2.2 Criteria

**Level A (Essential):**
| Criterion | Requirement |
|-----------|-------------|
| 1.1.1 Non-text Content | Provide text alternatives for images |
| 1.3.1 Info and Relationships | Use semantic HTML |
| 2.1.1 Keyboard | All functionality via keyboard |
| 2.4.1 Bypass Blocks | Skip navigation links |
| 4.1.1 Parsing | Valid HTML |

**Level AA (Standard):**
| Criterion | Requirement |
|-----------|-------------|
| 1.4.3 Contrast (Minimum) | 4.5:1 for normal text |
| 1.4.4 Resize Text | Page usable at 200% zoom |
| 2.4.6 Headings and Labels | Descriptive headings |
| 2.4.7 Focus Visible | Visible focus indicators |
| 3.1.2 Language of Parts | Mark language changes |

**Level AAA (Enhanced):**
| Criterion | Requirement |
|-----------|-------------|
| 1.4.6 Contrast (Enhanced) | 7:1 for normal text |
| 2.1.3 Keyboard (No Exception) | No keyboard traps |
| 2.4.9 Link Purpose | Links make sense alone |

---

## POUR Principles

WCAG is organized around four principles:

### 1. Perceivable

Users must be able to perceive the content.

```html
<!-- ✅ Perceivable: Image has text alternative -->
<img src="logo.png" alt="Acme Corp logo">

<!-- ✅ Perceivable: Video has captions -->
<video>
  <source src="intro.mp4" type="video/mp4">
  <track kind="captions" src="captions.vtt" srclang="en">
</video>

<!-- ✅ Perceivable: Sufficient contrast -->
<style>
  .text {
    color: #333;       /* Dark gray text */
    background: #fff;  /* White background */
    /* Contrast ratio: 12.6:1 ✓ */
  }
</style>
```

### 2. Operable

Users must be able to operate the interface.

```html
<!-- ✅ Operable: Button works with keyboard -->
<button onclick="submit()">Submit</button>

<!-- ❌ Not operable: Div with click handler -->
<div onclick="submit()">Submit</div>

<!-- ✅ Fixed: Add keyboard support -->
<div 
  role="button" 
  tabindex="0" 
  onclick="submit()"
  onkeydown="if(event.key==='Enter') submit()">
  Submit
</div>

<!-- ✅ Better: Just use a button -->
<button onclick="submit()">Submit</button>
```

### 3. Understandable

Users must be able to understand the content.

```html
<!-- ✅ Understandable: Clear error message -->
<label for="email">Email</label>
<input type="email" id="email" aria-describedby="email-error">
<span id="email-error" role="alert">
  Please enter a valid email address (e.g., name@example.com)
</span>

<!-- ✅ Understandable: Language specified -->
<html lang="en">
  <p>This is English content.</p>
  <p lang="fr">Ceci est du contenu français.</p>
</html>
```

### 4. Robust

Content must work with assistive technologies.

```html
<!-- ✅ Robust: Valid, semantic HTML -->
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/about">About</a></li>
  </ul>
</nav>

<!-- ✅ Robust: Custom widget with ARIA -->
<div 
  role="slider"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-valuenow="50"
  aria-label="Volume">
</div>
```

---

## Legal Requirements

### Why It Matters

Accessibility is the law in many countries:

| Region | Law | Applies To |
|--------|-----|------------|
| **USA** | ADA (Americans with Disabilities Act) | Public accommodations, state/local government |
| **USA** | Section 508 | Federal agencies and contractors |
| **EU** | European Accessibility Act | Private sector (2025) |
| **UK** | Equality Act 2010 | Service providers |
| **Canada** | AODA, ACA | Ontario, federal |

### ADA and Websites

Courts have ruled that websites are "places of public accommodation" under ADA:
- Major settlements against Target, Netflix, Domino's
- Class action lawsuits increasing
- WCAG 2.1 AA often cited as standard

### Section 508

Federal agencies must:
- Use WCAG 2.0 Level AA (or higher)
- Procure accessible technology
- Test for accessibility

---

## Accessibility Tree

### What Is It?

Browsers create an **accessibility tree** from the DOM—a simplified version for assistive technologies.

```
DOM Element                    Accessibility Object
────────────────               ────────────────────
<button>Click</button>    →    role: button
                               name: "Click"
                               state: focusable

<input type="text"        →    role: textbox
       id="name"               name: "Full Name"
       aria-label="...">       state: editable, focusable

<div class="box">         →    (often excluded or 
                                role: generic)
```

### Viewing the Accessibility Tree

**Chrome:**
1. DevTools → Elements
2. Accessibility pane (in sidebar)

**Firefox:**
1. DevTools → Accessibility tab
2. Shows full tree view

### What Screen Readers See

```html
<button class="btn btn-primary large" id="submit-form">
  <svg>...</svg> Submit Order
</button>
```

Screen reader announces: **"Submit Order, button"**

The screen reader ignores:
- CSS classes
- Visual styling
- Decorative SVG icon

It uses:
- Role (button)
- Accessible name ("Submit Order")
- State (focused, pressed, disabled)

---

## Accessibility Mindset

### Design Phase

- Include accessibility in requirements
- Review designs for contrast, focus states
- Consider keyboard-only navigation

### Development Phase

- Use semantic HTML first
- Add ARIA only when needed
- Test with keyboard constantly

### Testing Phase

- Automated scans (Lighthouse, axe)
- Manual keyboard testing
- Screen reader testing
- Real user testing

### The Curb-Cut Effect

Accessibility improvements help everyone:

| Feature | Helps | Also Benefits |
|---------|-------|---------------|
| Captions | Deaf users | Noisy environments, non-native speakers |
| Keyboard nav | Motor impairments | Power users, broken mouse |
| Clear headings | Screen readers | SEO, scanning content |
| High contrast | Low vision | Bright sunlight |
| Simple language | Cognitive | Non-experts, second language |

---

## Hands-on Exercise

### Your Task

Evaluate this code for accessibility issues:

```html
<div class="nav">
  <span onclick="goto('home')">Home</span>
  <span onclick="goto('about')">About</span>
</div>

<div class="main">
  <div class="title">Welcome to Our Site</div>
  <img src="hero.jpg">
  <div onclick="signup()">Sign Up Now</div>
</div>
```

### Questions

1. List all accessibility issues
2. Which WCAG criteria are violated?
3. Rewrite with proper accessibility

<details>
<summary>✅ Solution</summary>

**Issues:**
1. `<div class="nav">` - Not semantic (use `<nav>`)
2. `<span onclick>` - Not keyboard accessible (use `<a>` or `<button>`)
3. `<div class="title">` - Not a heading (use `<h1>`)
4. `<img>` - Missing alt text
5. `<div onclick>` - Not keyboard accessible (use `<button>`)

**Fixed code:**
```html
<nav aria-label="Main">
  <a href="/">Home</a>
  <a href="/about">About</a>
</nav>

<main>
  <h1>Welcome to Our Site</h1>
  <img src="hero.jpg" alt="Team members collaborating in modern office">
  <button onclick="signup()">Sign Up Now</button>
</main>
```
</details>

---

## Summary

✅ **WCAG** provides international accessibility standards
✅ Target **Level AA** for most projects
✅ **POUR**: Perceivable, Operable, Understandable, Robust
✅ Accessibility is **legally required** in many jurisdictions
✅ The **accessibility tree** is what assistive tech uses
✅ Accessibility benefits **everyone** (curb-cut effect)

**Next:** [Semantic HTML for Accessibility](./02-semantic-html-accessibility.md)

---

## Further Reading

- [WCAG 2.2 Quick Reference](https://www.w3.org/WAI/WCAG22/quickref/)
- [Understanding WCAG](https://www.w3.org/WAI/WCAG21/Understanding/)
- [WebAIM Introduction](https://webaim.org/intro/)

<!-- 
Sources Consulted:
- WCAG 2.2: https://www.w3.org/TR/WCAG22/
- WebAIM: https://webaim.org/
-->
