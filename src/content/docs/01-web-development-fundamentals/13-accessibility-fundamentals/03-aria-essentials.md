---
title: "ARIA Essentials"
---

# ARIA Essentials

## Introduction

**ARIA (Accessible Rich Internet Applications)** adds semantic information that HTML can't provide alone. It bridges the gap between complex UI widgets and assistive technologies. However, ARIA is a double-edged sword—misuse can make accessibility worse.

The first rule of ARIA: **Don't use ARIA if you can use native HTML.**

### What We'll Cover

- ARIA roles, states, and properties
- aria-label, aria-labelledby, aria-describedby
- aria-live for dynamic content
- aria-hidden and inert attribute
- When NOT to use ARIA

### Prerequisites

- HTML semantics
- Understanding of the accessibility tree
- Basic JavaScript for dynamic examples

---

## The First Rule of ARIA

> **No ARIA is better than bad ARIA.**

```html
<!-- ❌ Using ARIA when HTML works -->
<div role="button" tabindex="0" aria-pressed="false">Click me</div>

<!-- ✅ Just use a button -->
<button>Click me</button>
```

Native HTML gives you:
- Keyboard support for free
- Focus management
- Correct semantics
- Better browser/AT support

Use ARIA only when:
- No native HTML element exists
- You're building complex widgets
- You need to override default semantics

---

## ARIA Roles

Roles define what an element is.

### Common Roles

| Role | Purpose | Native Alternative |
|------|---------|-------------------|
| `button` | Clickable action | `<button>` |
| `link` | Navigation | `<a>` |
| `checkbox` | Toggle option | `<input type="checkbox">` |
| `slider` | Value in range | `<input type="range">` |
| `dialog` | Modal dialog | `<dialog>` |
| `alert` | Important message | — |
| `tab` | Tab interface | — |
| `menu` | Application menu | — |

### Role Categories

```
Document Structure Roles:
  article, heading, list, listitem, table, row, cell

Widget Roles:
  button, checkbox, slider, textbox, combobox, menu

Landmark Roles:
  banner, navigation, main, complementary, contentinfo

Live Region Roles:
  alert, status, log, timer

Window Roles:
  dialog, alertdialog
```

### Using Roles

```html
<!-- Custom slider -->
<div 
  role="slider"
  tabindex="0"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-valuenow="50"
  aria-label="Volume">
</div>

<!-- Alert that announces immediately -->
<div role="alert">
  Your session will expire in 5 minutes.
</div>

<!-- Tabs interface -->
<div role="tablist">
  <button role="tab" aria-selected="true" aria-controls="panel1">Tab 1</button>
  <button role="tab" aria-selected="false" aria-controls="panel2">Tab 2</button>
</div>
<div role="tabpanel" id="panel1">Content 1</div>
<div role="tabpanel" id="panel2" hidden>Content 2</div>
```

---

## ARIA States and Properties

### States (Change)

States are dynamic and change with user interaction:

| State | Values | Purpose |
|-------|--------|---------|
| `aria-checked` | true/false/mixed | Checkbox/radio state |
| `aria-selected` | true/false | Selected item |
| `aria-expanded` | true/false | Expandable section |
| `aria-pressed` | true/false/mixed | Toggle button |
| `aria-disabled` | true/false | Disabled control |
| `aria-hidden` | true/false | Hide from AT |
| `aria-busy` | true/false | Loading state |

### Properties (Static or Rare Changes)

| Property | Purpose |
|----------|---------|
| `aria-label` | Accessible name |
| `aria-labelledby` | Reference to labeling element |
| `aria-describedby` | Reference to description |
| `aria-controls` | Element this controls |
| `aria-owns` | DOM children not in subtree |
| `aria-live` | Live region politeness |
| `aria-required` | Required field |
| `aria-invalid` | Validation error |

### State Changes

```html
<!-- Toggle button -->
<button 
  aria-pressed="false"
  onclick="this.setAttribute('aria-pressed', 
    this.getAttribute('aria-pressed') === 'false')">
  Dark Mode
</button>

<!-- Expandable section -->
<button 
  aria-expanded="false" 
  aria-controls="details"
  onclick="toggleDetails()">
  Show Details
</button>
<div id="details" hidden>
  Detailed information here...
</div>
```

```javascript
function toggleDetails() {
  const button = document.querySelector('[aria-controls="details"]');
  const details = document.getElementById('details');
  const isExpanded = button.getAttribute('aria-expanded') === 'true';
  
  button.setAttribute('aria-expanded', !isExpanded);
  details.hidden = isExpanded;
}
```

---

## Naming Elements

### aria-label

Provides an accessible name directly:

```html
<!-- Icon-only button -->
<button aria-label="Close">
  <svg><!-- X icon --></svg>
</button>

<!-- Search landmark -->
<nav aria-label="Main navigation">...</nav>
<nav aria-label="Footer navigation">...</nav>

<!-- Input without visible label -->
<input type="search" aria-label="Search products">
```

### aria-labelledby

References another element's content:

```html
<!-- Dialog title as label -->
<div role="dialog" aria-labelledby="dialog-title">
  <h2 id="dialog-title">Confirm Action</h2>
  <p>Are you sure you want to proceed?</p>
</div>

<!-- Multiple sources -->
<div id="name">Jane</div>
<div id="job">Engineer</div>
<button aria-labelledby="name job">
  <!-- Announced as "Jane Engineer" -->
  View Profile
</button>
```

### aria-describedby

Provides additional description:

```html
<!-- Password help text -->
<label for="password">Password</label>
<input 
  type="password" 
  id="password"
  aria-describedby="password-requirements">
<div id="password-requirements">
  Must be 8+ characters with at least one number.
</div>

<!-- Error message -->
<input 
  type="email" 
  aria-invalid="true"
  aria-describedby="email-error">
<div id="email-error" role="alert">
  Please enter a valid email address.
</div>
```

### Priority

When multiple naming sources exist:

1. `aria-labelledby` (highest priority)
2. `aria-label`
3. Native labeling (`<label>`, alt text)
4. Content (text inside element)

---

## Live Regions

Live regions announce dynamic content changes.

### aria-live

```html
<!-- Polite: Waits for pause in speech -->
<div aria-live="polite">
  Results updated: 42 items found.
</div>

<!-- Assertive: Interrupts immediately -->
<div aria-live="assertive">
  Error: Connection lost!
</div>

<!-- Using role shorthand -->
<div role="status">Saved.</div>  <!-- aria-live="polite" -->
<div role="alert">Error!</div>    <!-- aria-live="assertive" -->
```

### Live Region Politeness

| Value | Behavior | Use Case |
|-------|----------|----------|
| `off` | Not announced | Default |
| `polite` | Waits for pause | Status updates, non-urgent |
| `assertive` | Interrupts | Errors, time-sensitive |

### Live Region Modifiers

```html
<div 
  aria-live="polite"
  aria-atomic="true"
  aria-relevant="additions text">
  <!-- aria-atomic: Announce entire region, not just changes -->
  <!-- aria-relevant: What changes to announce -->
</div>
```

### Common Patterns

```html
<!-- Loading indicator -->
<div aria-live="polite" aria-busy="true">
  Loading...
</div>

<!-- Search results -->
<div aria-live="polite">
  <span id="result-count">42 results found</span>
</div>

<!-- Form submission -->
<form>
  <!-- ... form fields ... -->
  <div role="status" aria-live="polite"></div>
</form>
```

```javascript
function showSuccess() {
  document.querySelector('[role="status"]').textContent = 
    'Form submitted successfully!';
}
```

---

## Hiding Content

### aria-hidden

Hides from accessibility tree but remains visible:

```html
<!-- Decorative icon next to text -->
<button>
  <span aria-hidden="true">🔍</span>
  Search
</button>

<!-- Decorative image -->
<img src="decoration.png" alt="" aria-hidden="true">

<!-- Content being animated out -->
<div class="modal closing" aria-hidden="true">...</div>
```

> **Warning:** Never use `aria-hidden="true"` on focusable elements!

### Hidden vs aria-hidden

| Technique | Visible | In AT | Focusable |
|-----------|---------|-------|-----------|
| CSS `display: none` | ❌ | ❌ | ❌ |
| CSS `visibility: hidden` | ❌ | ❌ | ❌ |
| HTML `hidden` | ❌ | ❌ | ❌ |
| `aria-hidden="true"` | ✅ | ❌ | ⚠️ Don't do |
| `.sr-only` | ❌ | ✅ | ✅ |

### inert Attribute

Makes content non-interactive AND hidden from AT:

```html
<!-- When modal is open, inert the rest -->
<main inert>
  <!-- Cannot focus or interact with anything here -->
</main>

<div class="modal" role="dialog">
  <!-- Only this is interactive -->
</div>
```

```javascript
function openModal() {
  document.querySelector('main').inert = true;
  document.querySelector('.modal').hidden = false;
}

function closeModal() {
  document.querySelector('main').inert = false;
  document.querySelector('.modal').hidden = true;
}
```

---

## When NOT to Use ARIA

### ❌ Replacing Semantic HTML

```html
<!-- Bad: ARIA for things HTML does -->
<div role="button" tabindex="0">Submit</div>
<div role="link" tabindex="0">Home</div>
<div role="checkbox" aria-checked="false">Option</div>

<!-- Good: Native HTML -->
<button>Submit</button>
<a href="/">Home</a>
<input type="checkbox"> Option
```

### ❌ Adding Roles Without Behavior

```html
<!-- Bad: Role without keyboard support -->
<div role="button">Click me</div>

<!-- Missing: keyboard handler, focus indication -->
```

If you add a role, you must implement the full behavior.

### ❌ Redundant ARIA

```html
<!-- Bad: Already implicit -->
<nav role="navigation">
<button role="button">
<a href="/" role="link">

<!-- Good: Just use the element -->
<nav>
<button>
<a href="/">
```

### ❌ aria-label on Non-Interactive Elements

```html
<!-- Bad: Won't be announced -->
<p aria-label="Important paragraph">Content</p>
<div aria-label="Section title">...</div>

<!-- Good: Use for interactive/landmark elements -->
<nav aria-label="Main">
<button aria-label="Close">
```

---

## ARIA Widget Patterns

### Disclosure (Show/Hide)

```html
<button 
  aria-expanded="false" 
  aria-controls="content-1">
  FAQ Question
</button>
<div id="content-1" hidden>
  FAQ Answer...
</div>
```

### Tabs

```html
<div role="tablist" aria-label="Options">
  <button 
    role="tab" 
    aria-selected="true"
    aria-controls="panel-1"
    id="tab-1">
    Tab 1
  </button>
  <button 
    role="tab" 
    aria-selected="false"
    aria-controls="panel-2"
    id="tab-2"
    tabindex="-1">
    Tab 2
  </button>
</div>

<div 
  role="tabpanel" 
  id="panel-1"
  aria-labelledby="tab-1">
  Content for tab 1
</div>

<div 
  role="tabpanel" 
  id="panel-2"
  aria-labelledby="tab-2"
  hidden>
  Content for tab 2
</div>
```

---

## Hands-on Exercise

### Your Task

Make this custom dropdown accessible:

```html
<div class="dropdown">
  <div class="trigger" onclick="toggle()">
    Select option
  </div>
  <ul class="menu" style="display: none;">
    <li onclick="select('a')">Option A</li>
    <li onclick="select('b')">Option B</li>
    <li onclick="select('c')">Option C</li>
  </ul>
</div>
```

<details>
<summary>✅ Solution</summary>

```html
<div class="dropdown">
  <button 
    aria-haspopup="listbox"
    aria-expanded="false"
    aria-controls="dropdown-menu"
    onclick="toggle()">
    Select option
  </button>
  <ul 
    role="listbox" 
    id="dropdown-menu"
    aria-label="Options"
    hidden>
    <li role="option" tabindex="-1" onclick="select('a')">Option A</li>
    <li role="option" tabindex="-1" onclick="select('b')">Option B</li>
    <li role="option" tabindex="-1" onclick="select('c')">Option C</li>
  </ul>
</div>
```

Also need keyboard support for arrow keys, Enter, Escape.
</details>

---

## Summary

✅ **First rule**: Prefer native HTML over ARIA
✅ **Roles** define what an element is
✅ **States** change with interaction (`aria-expanded`, `aria-selected`)
✅ Use **aria-label/labelledby** to name elements
✅ Use **aria-describedby** for additional help text
✅ **aria-live** announces dynamic content
✅ **aria-hidden** removes from accessibility tree (keep visible)
✅ **inert** makes content non-interactive

**Next:** [Keyboard Accessibility](./04-keyboard-accessibility.md)

---

## Further Reading

- [WAI-ARIA Practices](https://www.w3.org/WAI/ARIA/apg/)
- [MDN ARIA](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)
- [ARIA Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/patterns/)

<!-- 
Sources Consulted:
- WAI-ARIA 1.2: https://www.w3.org/TR/wai-aria-1.2/
- ARIA Practices Guide: https://www.w3.org/WAI/ARIA/apg/
-->
