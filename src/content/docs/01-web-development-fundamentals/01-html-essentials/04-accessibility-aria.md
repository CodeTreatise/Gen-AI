---
title: "Accessibility & ARIA"
---

# Accessibility & ARIA

## Introduction

Accessibility isn't optional—it's essential. Around 15% of the global population has some form of disability, and accessible HTML ensures your AI applications work for everyone. ARIA (Accessible Rich Internet Applications) extends HTML semantics for complex interactive components.

This lesson teaches you to write HTML that screen readers understand and users can navigate with keyboards alone—crucial skills for building inclusive AI interfaces.

### What We'll Cover

- Why accessibility matters for AI apps
- Built-in HTML accessibility features
- ARIA roles, states, and properties
- Keyboard navigation
- Focus management
- Accessible forms and error handling
- Testing accessibility

### Prerequisites

- HTML document structure
- Semantic elements
- Forms and input types

---

## Why Accessibility Matters

### Legal Requirements

Many countries have laws requiring accessible websites:
- **US:** ADA, Section 508
- **EU:** European Accessibility Act
- **UK:** Equality Act

### Business Impact

- 15%+ of users have disabilities
- SEO benefits (search engines are "blind")
- Better UX for everyone (curb-cut effect)
- Mobile users benefit from accessibility

### AI Applications

For AI-powered apps, accessibility is especially important:
- Chat interfaces need screen reader support
- AI-generated content must be accessible
- Voice input/output expands AI reach

---

## Built-in HTML Accessibility

Before reaching for ARIA, use semantic HTML:

### Semantic Elements Provide Meaning

| Element | Accessible Meaning |
|---------|-------------------|
| `<nav>` | Navigation region |
| `<main>` | Main content |
| `<button>` | Clickable action |
| `<a href>` | Link to destination |
| `<h1>`-`<h6>` | Heading hierarchy |
| `<ul>`, `<ol>` | List of items |
| `<table>` | Tabular data |

### The First Rule of ARIA

> **Don't use ARIA if native HTML works.**

```html
<!-- ❌ Bad: Using ARIA unnecessarily -->
<div role="button" tabindex="0" onclick="submit()">Submit</div>

<!-- ✅ Good: Native HTML element -->
<button type="submit">Submit</button>
```

Native elements have:
- Built-in keyboard support
- Correct ARIA semantics
- Expected browser behaviors

---

## ARIA Fundamentals

ARIA has three types of attributes:

### 1. Roles

Define what an element IS:

```html
<div role="alert">Your session will expire in 5 minutes.</div>
<div role="dialog" aria-modal="true">...</div>
<ul role="tablist">...</ul>
```

### 2. States

Dynamic conditions that change:

```html
<button aria-expanded="false">Show Menu</button>
<input aria-invalid="true">
<div aria-hidden="true">Decorative content</div>
```

### 3. Properties

Static characteristics:

```html
<input aria-label="Search">
<section aria-labelledby="section-title">
<div aria-describedby="help-text">
```

---

## Essential ARIA Attributes

### `aria-label`

Provides an accessible name when visible text isn't available:

```html
<!-- Icon-only button -->
<button aria-label="Close dialog">
  <svg><!-- X icon --></svg>
</button>

<!-- Search input -->
<input type="search" aria-label="Search products">
```

### `aria-labelledby`

References another element's text as the label:

```html
<h2 id="dialog-title">Delete Confirmation</h2>
<div role="dialog" aria-labelledby="dialog-title">
  <p>Are you sure you want to delete this item?</p>
</div>
```

### `aria-describedby`

Adds supplementary description:

```html
<label for="password">Password:</label>
<input type="password" id="password" 
       aria-describedby="password-help">
<p id="password-help">Must be at least 8 characters with one number.</p>
```

### `aria-hidden`

Hides decorative content from screen readers:

```html
<!-- Decorative icon -->
<span aria-hidden="true">🎨</span>

<!-- Skip decorative images -->
<img src="divider.png" alt="" aria-hidden="true">
```

> **Warning:** Never use `aria-hidden="true"` on focusable elements.

### `aria-live`

Announces dynamic content changes:

```html
<!-- AI response area -->
<div aria-live="polite" aria-atomic="true" id="ai-response">
  <!-- Content updates announced to screen readers -->
</div>
```

| Value | When to Use |
|-------|-------------|
| `polite` | Non-urgent updates (wait for silence) |
| `assertive` | Urgent alerts (interrupt immediately) |
| `off` | No announcements |

---

## Accessible Forms

### Labeling Inputs

Every input needs an accessible name:

```html
<!-- Method 1: Explicit label (best) -->
<label for="email">Email Address:</label>
<input type="email" id="email" name="email">

<!-- Method 2: Wrapping label -->
<label>
  Email Address:
  <input type="email" name="email">
</label>

<!-- Method 3: aria-label (when no visible label) -->
<input type="search" aria-label="Search the site">

<!-- Method 4: aria-labelledby (reference existing text) -->
<span id="email-label">Email Address:</span>
<input type="email" aria-labelledby="email-label">
```

### Required Fields

```html
<label for="name">Name: <span aria-hidden="true">*</span></label>
<input type="text" id="name" required aria-required="true">
```

Both `required` and `aria-required="true"` ensure compatibility.

### Error Messages

```html
<label for="email">Email:</label>
<input type="email" id="email" 
       aria-invalid="true" 
       aria-describedby="email-error">
<p id="email-error" role="alert">
  Please enter a valid email address.
</p>
```

### Complete Accessible Form

```html
<form aria-label="Contact form">
  <div>
    <label for="name">Name: <span aria-hidden="true">*</span></label>
    <input type="text" id="name" required aria-required="true">
  </div>
  
  <div>
    <label for="email">Email: <span aria-hidden="true">*</span></label>
    <input type="email" id="email" required 
           aria-describedby="email-help">
    <p id="email-help">We'll never share your email.</p>
  </div>
  
  <div>
    <label for="message">Message:</label>
    <textarea id="message" rows="4"></textarea>
  </div>
  
  <button type="submit">Send Message</button>
</form>
```

---

## Keyboard Navigation

### Focus Order

Elements receive focus in DOM order. Ensure it's logical:

```html
<!-- Tab order follows visual layout -->
<nav>...</nav>
<main>
  <form>
    <input>  <!-- Tab 1 -->
    <input>  <!-- Tab 2 -->
    <button> <!-- Tab 3 -->
  </form>
</main>
```

### `tabindex`

| Value | Behavior |
|-------|----------|
| `0` | Element is focusable in natural order |
| `-1` | Focusable only via JavaScript |
| `1+` | ❌ Avoid—disrupts natural order |

```html
<!-- Make a div focusable -->
<div tabindex="0" role="button" onclick="handleClick()">
  Custom Button
</div>

<!-- Remove from tab order but allow JS focus -->
<div tabindex="-1" id="modal">...</div>
```

### Skip Links

Allow keyboard users to skip navigation:

```html
<body>
  <a href="#main" class="skip-link">Skip to main content</a>
  <nav><!-- Long navigation --></nav>
  <main id="main" tabindex="-1">
    <!-- Main content -->
  </main>
</body>
```

```css
.skip-link {
  position: absolute;
  left: -9999px;
}
.skip-link:focus {
  left: 0;
  top: 0;
  background: #000;
  color: #fff;
  padding: 1rem;
}
```

---

## Interactive Component Patterns

### Expandable Section

```html
<button aria-expanded="false" aria-controls="details">
  Show Details
</button>
<div id="details" hidden>
  <p>Hidden content here...</p>
</div>

<script>
  const btn = document.querySelector('button');
  const details = document.getElementById('details');
  
  btn.addEventListener('click', () => {
    const expanded = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', !expanded);
    details.hidden = expanded;
  });
</script>
```

### Tab Panel

```html
<div role="tablist" aria-label="AI Models">
  <button role="tab" aria-selected="true" aria-controls="panel-gpt" id="tab-gpt">
    GPT-4
  </button>
  <button role="tab" aria-selected="false" aria-controls="panel-claude" id="tab-claude">
    Claude
  </button>
</div>

<div role="tabpanel" id="panel-gpt" aria-labelledby="tab-gpt">
  <p>GPT-4 is developed by OpenAI...</p>
</div>

<div role="tabpanel" id="panel-claude" aria-labelledby="tab-claude" hidden>
  <p>Claude is developed by Anthropic...</p>
</div>
```

### Modal Dialog

```html
<div role="dialog" aria-modal="true" aria-labelledby="modal-title">
  <h2 id="modal-title">Confirm Action</h2>
  <p>Are you sure you want to proceed?</p>
  <button>Cancel</button>
  <button>Confirm</button>
</div>
```

**Modal requirements:**
- Trap focus inside the modal
- Return focus to trigger element on close
- Close on Escape key

---

## AI Chat Interface Accessibility

### Accessible Chat Container

```html
<section aria-label="AI Chat" class="chat-container">
  <h2 id="chat-title" class="visually-hidden">Chat with AI Assistant</h2>
  
  <!-- Message history -->
  <div id="messages" role="log" aria-live="polite" aria-label="Chat messages">
    <div class="message user">
      <span class="visually-hidden">You said:</span>
      <p>What is machine learning?</p>
    </div>
    <div class="message assistant">
      <span class="visually-hidden">AI replied:</span>
      <p>Machine learning is a subset of AI...</p>
    </div>
  </div>
  
  <!-- Loading indicator -->
  <div id="loading" aria-live="polite" hidden>
    <span role="status">AI is thinking...</span>
  </div>
  
  <!-- Input form -->
  <form aria-label="Send message to AI">
    <label for="prompt" class="visually-hidden">Your message</label>
    <textarea id="prompt" placeholder="Type your message..." 
              aria-describedby="char-count"></textarea>
    <span id="char-count">0/4000 characters</span>
    <button type="submit" aria-label="Send message">
      <svg aria-hidden="true"><!-- send icon --></svg>
    </button>
  </form>
</section>
```

### Visually Hidden Utility

```css
.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

---

## Common ARIA Roles

### Landmark Roles

| Role | HTML Equivalent | Purpose |
|------|-----------------|---------|
| `banner` | `<header>` | Site header |
| `navigation` | `<nav>` | Navigation links |
| `main` | `<main>` | Main content |
| `contentinfo` | `<footer>` | Site footer |
| `complementary` | `<aside>` | Related content |
| `search` | `<search>` | Search functionality |

### Widget Roles

| Role | Purpose |
|------|---------|
| `button` | Clickable button |
| `dialog` | Modal window |
| `alert` | Important message |
| `alertdialog` | Modal requiring action |
| `tab`, `tabpanel` | Tabbed interface |
| `menu`, `menuitem` | Menu system |
| `progressbar` | Progress indicator |

---

## Testing Accessibility

### Keyboard Testing

1. Unplug your mouse
2. Tab through the entire page
3. Verify all interactive elements are reachable
4. Check focus is visible
5. Ensure logical tab order

### Screen Reader Testing

| Platform | Screen Reader | Shortcut |
|----------|--------------|----------|
| Windows | NVDA (free) | Download |
| Windows | Narrator | Win + Ctrl + Enter |
| macOS | VoiceOver | Cmd + F5 |
| Chrome | ChromeVox | Extension |

### Browser DevTools

1. **Chrome:** Lighthouse → Accessibility audit
2. **Firefox:** Accessibility Inspector
3. **Edge:** Issues panel → Accessibility

### Automated Tools

- **axe DevTools** (browser extension)
- **WAVE** (web accessibility evaluator)
- **Pa11y** (command line)

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use semantic HTML first | Built-in accessibility |
| Label all form inputs | Screen readers need them |
| Ensure keyboard navigation | Not everyone uses a mouse |
| Provide visible focus states | Shows current position |
| Test with screen readers | Catches hidden issues |
| Use color + icons/text | Color-blind users |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using divs for buttons | Use `<button>` element |
| Missing alt text | Add descriptive `alt` |
| Low color contrast | Use contrast checker |
| No focus indicator | Add `:focus` styles |
| Hiding with `display: none` | Use `aria-hidden` or `.visually-hidden` |
| ARIA without behavior | ARIA doesn't add functionality |

---

## Hands-on Exercise

### Your Task

Create an accessible accordion component with:
1. Three expandable sections
2. Proper ARIA attributes (`aria-expanded`, `aria-controls`)
3. Keyboard support (Enter/Space to toggle)
4. Visible focus states

<details>
<summary>💡 Hints</summary>

- Use `<button>` for triggers
- Connect buttons to panels with `aria-controls`
- Toggle `aria-expanded` on click
- Use `hidden` attribute on collapsed panels
</details>

<details>
<summary>✅ Solution</summary>

```html
<div class="accordion">
  <h3>
    <button aria-expanded="true" aria-controls="panel1" id="btn1">
      What is AI?
    </button>
  </h3>
  <div id="panel1" role="region" aria-labelledby="btn1">
    <p>Artificial Intelligence is the simulation of human intelligence...</p>
  </div>
  
  <h3>
    <button aria-expanded="false" aria-controls="panel2" id="btn2">
      What is Machine Learning?
    </button>
  </h3>
  <div id="panel2" role="region" aria-labelledby="btn2" hidden>
    <p>Machine Learning is a subset of AI that learns from data...</p>
  </div>
  
  <h3>
    <button aria-expanded="false" aria-controls="panel3" id="btn3">
      What is Deep Learning?
    </button>
  </h3>
  <div id="panel3" role="region" aria-labelledby="btn3" hidden>
    <p>Deep Learning uses neural networks with many layers...</p>
  </div>
</div>

<style>
  button:focus {
    outline: 2px solid #6366f1;
    outline-offset: 2px;
  }
</style>

<script>
  document.querySelectorAll('.accordion button').forEach(btn => {
    btn.addEventListener('click', () => {
      const expanded = btn.getAttribute('aria-expanded') === 'true';
      const panel = document.getElementById(btn.getAttribute('aria-controls'));
      
      btn.setAttribute('aria-expanded', !expanded);
      panel.hidden = expanded;
    });
  });
</script>
```
</details>

---

## Summary

✅ Use semantic HTML before ARIA—native elements have built-in accessibility

✅ ARIA has roles (what it is), states (dynamic), and properties (static)

✅ Essential attributes: `aria-label`, `aria-labelledby`, `aria-describedby`, `aria-live`

✅ Every form input needs an accessible name via `<label>` or ARIA

✅ Ensure keyboard navigation works—all interactive elements must be reachable

✅ Test with real screen readers, not just automated tools

---

**Previous:** [Forms & Input Types](./03-forms-input-types.md)

**Next:** [Meta Tags & SEO](./05-meta-tags-seo.md)
