---
title: "Accessibility (a11y) Fundamentals"
---

# Accessibility (a11y) Fundamentals

## Overview

Accessibility ensures everyone can use your website, including people with disabilities. It's not just about compliance—it's about creating better experiences for all users. Accessible sites are often more usable, have better SEO, and reach wider audiences.

This lesson covers accessibility principles, semantic HTML, ARIA, keyboard navigation, and testing techniques.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-accessibility-principles.md) | Accessibility Principles | WCAG, POUR, legal requirements |
| [02](./02-semantic-html-accessibility.md) | Semantic HTML | Heading hierarchy, landmarks, labels |
| [03](./03-aria-essentials.md) | ARIA Essentials | Roles, states, properties, when to use |
| [04](./04-keyboard-accessibility.md) | Keyboard Accessibility | Focus management, tab order, skip links |
| [05](./05-testing-accessibility.md) | Testing Accessibility | Screen readers, automated tools, auditing |

---

## Why Accessibility Matters

### Who Benefits?

| Disability | Examples | Assistive Tech |
|------------|----------|----------------|
| **Visual** | Blind, low vision, color blind | Screen readers, magnifiers |
| **Auditory** | Deaf, hard of hearing | Captions, transcripts |
| **Motor** | Limited mobility, tremors | Keyboard, voice control, switch devices |
| **Cognitive** | Dyslexia, ADHD, memory | Clear layout, simple language |
| **Situational** | Bright sunlight, noisy environment | High contrast, captions |

> **Note:** 15% of the world's population has some form of disability. Accessibility benefits everyone.

---

## The POUR Principles

WCAG's foundation—every accessible experience must be:

```
┌──────────────────────────────────────────────────────┐
│                    PERCEIVABLE                       │
│        Can users perceive the content?               │
│   (text alternatives, captions, color contrast)      │
├──────────────────────────────────────────────────────┤
│                     OPERABLE                         │
│          Can users operate the UI?                   │
│    (keyboard access, enough time, no seizures)       │
├──────────────────────────────────────────────────────┤
│                   UNDERSTANDABLE                     │
│        Can users understand the content?             │
│     (readable, predictable, error prevention)        │
├──────────────────────────────────────────────────────┤
│                      ROBUST                          │
│       Does it work with assistive tech?              │
│      (valid HTML, compatible, future-proof)          │
└──────────────────────────────────────────────────────┘
```

---

## Quick Wins

### 1. Semantic HTML

```html
<!-- ❌ Non-semantic -->
<div class="nav">
  <div class="nav-item" onclick="navigate()">Home</div>
</div>

<!-- ✅ Semantic -->
<nav>
  <a href="/">Home</a>
</nav>
```

### 2. Alt Text for Images

```html
<!-- ❌ Missing alt -->
<img src="chart.png">

<!-- ✅ Descriptive alt -->
<img src="chart.png" alt="Bar chart showing 40% increase in sales Q4 2025">
```

### 3. Form Labels

```html
<!-- ❌ No label -->
<input type="email" placeholder="Email">

<!-- ✅ Proper label -->
<label for="email">Email</label>
<input type="email" id="email">
```

### 4. Color Contrast

Minimum contrast ratios:
- **Normal text**: 4.5:1
- **Large text** (18pt+): 3:1
- **UI components**: 3:1

### 5. Focus Indicators

```css
/* ❌ Never do this */
:focus { outline: none; }

/* ✅ Custom focus styles */
:focus {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}
```

---

## Accessibility Tree

Browsers create an accessibility tree from the DOM:

```
DOM:                          Accessibility Tree:
<nav>                         navigation
  <ul>                          list (3 items)
    <li><a>Home</a></li>          link "Home"
    <li><a>About</a></li>         link "About"
    <li><a>Contact</a></li>       link "Contact"
  </ul>
</nav>
```

Screen readers use this tree, not the visual rendering.

---

## Prerequisites

Before starting this lesson:
- HTML/CSS fundamentals
- Understanding of browser rendering
- Basic JavaScript (for interactive components)

---

## Start Learning

Begin with [Accessibility Principles](./01-accessibility-principles.md) to understand WCAG guidelines and legal requirements.

---

## Further Reading

- [WebAIM](https://webaim.org/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [A11y Project](https://www.a11yproject.com/)
- [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
