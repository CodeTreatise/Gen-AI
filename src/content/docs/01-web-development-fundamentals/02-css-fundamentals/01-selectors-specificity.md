---
title: "Selectors & Specificity"
---

# Selectors & Specificity

## Introduction

CSS selectors are patterns that match elements in your HTML. Mastering selectors means you can target exactly the elements you need—no more, no less. For AI interfaces, precise styling is essential: you might need to style AI responses differently from user messages, or highlight certain keywords in generated text.

This lesson covers all selector types and the specificity rules that determine which styles win when multiple rules target the same element.

### What We'll Cover

- Basic selectors: element, class, ID, universal
- Attribute selectors for matching element attributes
- Pseudo-classes for states and positions
- Pseudo-elements for generated content
- Combinators for relationship-based selection
- Specificity calculation and the cascade

### Prerequisites

- HTML document structure
- Basic understanding of CSS syntax (selectors, properties, values)

---

## Basic Selectors

### Element (Type) Selector

Targets all elements of a specific type:

```css
p {
  color: #333;
  line-height: 1.6;
}

h1 {
  font-size: 2rem;
}
```

### Class Selector

Targets elements with a specific class attribute (prefix with `.`):

```css
.message {
  padding: 1rem;
  border-radius: 8px;
}

.ai-response {
  background: #f0f4ff;
  border-left: 4px solid #6366f1;
}
```

```html
<div class="message ai-response">AI generated content here...</div>
```

### ID Selector

Targets a single element with a specific ID (prefix with `#`):

```css
#chat-container {
  max-width: 800px;
  margin: 0 auto;
}
```

> **Best practice:** Prefer classes over IDs for styling. IDs have high specificity and are meant for unique elements (like JavaScript hooks or anchor links).

### Universal Selector

Matches all elements:

```css
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
```

### Selector List (Grouping)

Apply the same styles to multiple selectors:

```css
h1, h2, h3 {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
}

.btn-primary, .btn-secondary {
  padding: 0.75rem 1.5rem;
  border-radius: 4px;
  cursor: pointer;
}
```

---

## Attribute Selectors

Target elements based on their attributes:

### Basic Attribute Presence

```css
/* Elements with a title attribute */
[title] {
  cursor: help;
}

/* Inputs with a required attribute */
input[required] {
  border-color: #ef4444;
}
```

### Attribute Value Matching

| Selector | Matches |
|----------|---------|
| `[attr="value"]` | Exact match |
| `[attr~="value"]` | Word in space-separated list |
| `[attr|="value"]` | Exact or starts with `value-` |
| `[attr^="value"]` | Starts with |
| `[attr$="value"]` | Ends with |
| `[attr*="value"]` | Contains |

```css
/* External links */
a[href^="https://"] {
  padding-right: 1rem;
  background: url('external-icon.svg') right center no-repeat;
}

/* PDF downloads */
a[href$=".pdf"] {
  color: #dc2626;
}

/* Links containing "api" */
a[href*="api"] {
  font-family: monospace;
}

/* Case-insensitive match */
a[href*="api" i] {
  color: #6366f1;
}
```

### Data Attributes

```css
/* Target custom data attributes */
[data-status="loading"] {
  opacity: 0.5;
  pointer-events: none;
}

[data-role="assistant"] {
  background: #f0fdf4;
}

[data-role="user"] {
  background: #eff6ff;
}
```

```html
<div class="message" data-role="assistant">Hello! How can I help?</div>
<div class="message" data-role="user">What is machine learning?</div>
```

---

## Pseudo-Classes

Pseudo-classes select elements based on state, position, or other dynamic conditions.

### User Action States

```css
/* Mouse hover */
.button:hover {
  background: #4f46e5;
}

/* Keyboard focus */
.button:focus {
  outline: 2px solid #6366f1;
  outline-offset: 2px;
}

/* Focus visible only from keyboard */
.button:focus-visible {
  outline: 2px solid #6366f1;
}

/* Active (being clicked) */
.button:active {
  transform: scale(0.98);
}
```

### Link States

```css
a:link {
  color: #2563eb;
}

a:visited {
  color: #7c3aed;
}

/* :any-link matches both :link and :visited */
a:any-link {
  text-decoration: none;
}
```

### Form States

```css
input:focus {
  border-color: #6366f1;
}

input:disabled {
  background: #f3f4f6;
  cursor: not-allowed;
}

input:required {
  border-left: 3px solid #ef4444;
}

input:valid {
  border-color: #10b981;
}

input:invalid {
  border-color: #ef4444;
}

/* Only shows invalid after user interaction */
input:user-invalid {
  border-color: #ef4444;
}

input:placeholder-shown {
  font-style: italic;
}
```

### Structural Pseudo-Classes

```css
/* First/last child */
li:first-child {
  font-weight: bold;
}

li:last-child {
  border-bottom: none;
}

/* Only child */
.container:only-child {
  margin: 0 auto;
}

/* nth-child patterns */
tr:nth-child(odd) {
  background: #f9fafb;
}

tr:nth-child(even) {
  background: white;
}

/* Every 3rd element */
.item:nth-child(3n) {
  margin-right: 0;
}

/* First 3 elements */
.item:nth-child(-n+3) {
  font-weight: bold;
}

/* Elements after the 5th */
.item:nth-child(n+6) {
  opacity: 0.7;
}
```

### Functional Pseudo-Classes

#### `:is()` - Matches Any

Reduces repetition in complex selectors:

```css
/* Without :is() */
header a:hover,
nav a:hover,
footer a:hover {
  color: #6366f1;
}

/* With :is() */
:is(header, nav, footer) a:hover {
  color: #6366f1;
}
```

> **Note:** `:is()` takes the specificity of its most specific argument.

#### `:where()` - Zero Specificity

Same as `:is()` but with zero specificity—great for defaults:

```css
/* Base styles that are easy to override */
:where(.button) {
  padding: 0.5rem 1rem;
  border-radius: 4px;
}
```

#### `:not()` - Negation

```css
/* All inputs except submit buttons */
input:not([type="submit"]) {
  border: 1px solid #d1d5db;
}

/* All links except those with .no-style */
a:not(.no-style) {
  color: #2563eb;
}

/* Multiple negations */
.item:not(:first-child):not(:last-child) {
  border-top: 1px solid #e5e7eb;
}
```

#### `:has()` - Parent Selector (Modern)

The relational pseudo-class—select elements based on their descendants:

```css
/* Card that contains an image */
.card:has(img) {
  padding-top: 0;
}

/* Form with invalid inputs */
form:has(input:invalid) {
  border-color: #ef4444;
}

/* Container with no children */
.container:has(:not(*)) {
  display: none;
}

/* Label when its input is focused */
label:has(+ input:focus) {
  color: #6366f1;
}
```

> **Browser support:** `:has()` is supported in all modern browsers (Chrome 105+, Firefox 121+, Safari 15.4+).

---

## Pseudo-Elements

Pseudo-elements create virtual elements for styling parts of content.

### `::before` and `::after`

Insert generated content:

```css
.required-field::after {
  content: " *";
  color: #ef4444;
}

.external-link::after {
  content: " ↗";
  font-size: 0.8em;
}

/* Decorative elements */
.section-title::before {
  content: "";
  display: inline-block;
  width: 4px;
  height: 1em;
  background: #6366f1;
  margin-right: 0.5rem;
}
```

### `::first-letter` and `::first-line`

```css
.article p:first-of-type::first-letter {
  font-size: 3rem;
  float: left;
  line-height: 1;
  margin-right: 0.5rem;
}

.intro::first-line {
  font-weight: bold;
}
```

### `::placeholder`

```css
input::placeholder {
  color: #9ca3af;
  font-style: italic;
}
```

### `::selection`

```css
::selection {
  background: #6366f1;
  color: white;
}
```

### `::marker`

Style list markers:

```css
li::marker {
  color: #6366f1;
  font-weight: bold;
}
```

---

## Combinators

Combinators select elements based on their relationship to other elements.

### Descendant Combinator (space)

Matches descendants at any depth:

```css
.chat-container .message {
  margin-bottom: 1rem;
}

article p {
  line-height: 1.7;
}
```

### Child Combinator (`>`)

Matches direct children only:

```css
.menu > li {
  display: inline-block;
}

/* Only direct children, not nested lists */
ul.nav > li > a {
  padding: 0.5rem 1rem;
}
```

### Next-Sibling Combinator (`+`)

Matches the immediately following sibling:

```css
/* Paragraph directly after a heading */
h2 + p {
  font-size: 1.1rem;
  margin-top: 0;
}

/* Input after label */
label + input {
  margin-left: 0.5rem;
}
```

### Subsequent-Sibling Combinator (`~`)

Matches all following siblings:

```css
/* All paragraphs after the first h2 */
h2 ~ p {
  color: #4b5563;
}

/* Show elements after a checked checkbox */
input:checked ~ .details {
  display: block;
}
```

---

## Specificity

When multiple rules target the same element, specificity determines which wins.

### Specificity Calculation

Specificity is calculated as three components: (A, B, C)

| Component | Counts |
|-----------|--------|
| A | ID selectors |
| B | Class, attribute, pseudo-class selectors |
| C | Element, pseudo-element selectors |

```css
/* (0, 0, 1) - one element */
p { }

/* (0, 1, 0) - one class */
.message { }

/* (1, 0, 0) - one ID */
#header { }

/* (0, 1, 1) - one class + one element */
p.intro { }

/* (0, 2, 1) - two classes + one element */
div.card.featured { }

/* (1, 1, 1) - one ID + one class + one element */
#main .content p { }

/* (0, 2, 0) - one attribute + one pseudo-class */
input[type="text"]:focus { }
```

### Comparing Specificity

Compare from left to right:

```
(1, 0, 0) beats (0, 10, 10) — IDs always beat classes
(0, 2, 0) beats (0, 1, 5) — more classes beats more elements
(0, 1, 3) beats (0, 1, 2) — tie broken by elements
```

### Specificity Examples

```css
/* Specificity: (0, 1, 0) */
.button {
  background: gray;
}

/* Specificity: (0, 2, 0) - wins! */
.primary.button {
  background: blue;
}

/* Specificity: (1, 0, 0) - wins over both! */
#submit-btn {
  background: green;
}
```

### `!important`

Overrides all specificity rules:

```css
.button {
  background: blue !important;
}
```

**Avoid `!important`** except for:
- Utility classes (`.hidden { display: none !important; }`)
- Overriding third-party CSS you can't modify

### Specificity Tips

| Tip | Why |
|-----|-----|
| Avoid IDs for styling | Too specific, hard to override |
| Keep specificity low | Easier to maintain |
| Use classes consistently | Predictable specificity |
| Use `:where()` for defaults | Zero specificity, easy to override |

---

## AI Chat Interface Example

```css
/* Message container - low specificity base */
.message {
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
}

/* Role-based styling using attributes */
.message[data-role="user"] {
  background: #eff6ff;
  margin-left: 2rem;
}

.message[data-role="assistant"] {
  background: #f0fdf4;
  margin-right: 2rem;
}

/* Streaming indicator */
.message[data-role="assistant"]:has(.streaming)::after {
  content: "▋";
  animation: blink 1s infinite;
}

/* Error state */
.message:has(.error) {
  background: #fef2f2;
  border-left: 4px solid #ef4444;
}

/* Code blocks in responses */
.message pre {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 1rem;
  border-radius: 4px;
  overflow-x: auto;
}

@keyframes blink {
  50% { opacity: 0; }
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Start with low specificity | Easier to override later |
| Use meaningful class names | Self-documenting code |
| Prefer classes over IDs | Consistent specificity |
| Use `:is()` to reduce repetition | Cleaner selectors |
| Use `:where()` for defaults | Zero specificity base |
| Limit nesting depth | Avoid (0, 0, 5) specificity |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Overusing IDs | Use classes instead |
| `!important` everywhere | Fix specificity at the source |
| Overly specific selectors | Simplify: `.card` not `div.card.featured` |
| Ignoring cascade order | Later rules win at same specificity |
| Not testing `:has()` support | Check browser compatibility |

---

## Hands-on Exercise

### Your Task

Create a notification component with these requirements:

1. Base `.notification` class with padding and border-radius
2. Variants: `.notification--success`, `.notification--error`, `.notification--warning`
3. Use `::before` to add an icon (✓, ✕, ⚠)
4. Style links inside notifications differently
5. Add a dismiss button that appears on hover

<details>
<summary>✅ Solution</summary>

```css
/* Base notification */
.notification {
  padding: 1rem 1rem 1rem 2.5rem;
  border-radius: 8px;
  border: 1px solid;
  position: relative;
  margin-bottom: 1rem;
}

/* Icon */
.notification::before {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
}

/* Variants */
.notification--success {
  background: #f0fdf4;
  border-color: #86efac;
  color: #166534;
}

.notification--success::before {
  content: "✓";
  color: #22c55e;
}

.notification--error {
  background: #fef2f2;
  border-color: #fecaca;
  color: #991b1b;
}

.notification--error::before {
  content: "✕";
  color: #ef4444;
}

.notification--warning {
  background: #fffbeb;
  border-color: #fde68a;
  color: #92400e;
}

.notification--warning::before {
  content: "⚠";
  color: #f59e0b;
}

/* Links in notifications */
.notification a {
  color: inherit;
  font-weight: 600;
  text-decoration: underline;
}

.notification a:hover {
  text-decoration: none;
}

/* Dismiss button */
.notification .dismiss {
  position: absolute;
  right: 0.5rem;
  top: 0.5rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.notification:hover .dismiss {
  opacity: 1;
}
```

```html
<div class="notification notification--success">
  Your file was uploaded successfully. <a href="#">View file</a>
  <button class="dismiss">×</button>
</div>

<div class="notification notification--error">
  Failed to connect to the API. <a href="#">Retry</a>
  <button class="dismiss">×</button>
</div>
```
</details>

---

## Summary

✅ Use **element**, **class**, **ID**, and **attribute** selectors to target elements

✅ **Pseudo-classes** like `:hover`, `:focus`, `:nth-child()`, and `:has()` select based on state and relationships

✅ **Pseudo-elements** like `::before` and `::after` create virtual elements for styling

✅ **Combinators** (` `, `>`, `+`, `~`) select elements based on DOM relationships

✅ **Specificity** is (IDs, Classes, Elements)—higher specificity wins

✅ Use `:is()` and `:where()` to write cleaner selectors with controlled specificity

---

**Previous:** [CSS Fundamentals Overview](./00-css-fundamentals.md)

**Next:** [Box Model & Layout](./02-box-model-layout.md)

<!-- 
Sources Consulted:
- MDN CSS Selectors: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_selectors
- MDN Specificity: https://developer.mozilla.org/en-US/docs/Web/CSS/Specificity
- MDN :has() selector: https://developer.mozilla.org/en-US/docs/Web/CSS/:has
- MDN :is() selector: https://developer.mozilla.org/en-US/docs/Web/CSS/:is
-->
