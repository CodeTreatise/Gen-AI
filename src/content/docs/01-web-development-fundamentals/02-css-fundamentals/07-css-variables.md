---
title: "CSS Variables (Custom Properties)"
---

# CSS Variables (Custom Properties)

## Introduction

CSS Custom Properties—commonly called CSS variables—let you define reusable values that cascade like any other CSS property. Unlike preprocessor variables (Sass, Less), CSS variables are dynamic: they can change at runtime, respond to media queries, and be manipulated with JavaScript.

For AI interfaces, CSS variables enable dynamic theming, real-time customization, and consistent design tokens across your application.

### What We'll Cover

- Declaring and using custom properties
- Scope, inheritance, and the cascade
- Fallback values and error handling
- Dynamic theming with JavaScript
- Design token systems
- The `@property` rule for typed custom properties

### Prerequisites

- CSS basics (selectors, properties, values)
- Understanding of the cascade and inheritance

---

## Declaring Custom Properties

Define custom properties with `--` prefix:

```css
:root {
  --primary-color: #6366f1;
  --secondary-color: #10b981;
  --spacing-unit: 8px;
  --border-radius: 4px;
}
```

The `:root` selector targets the document root (`<html>`), making variables globally available.

### Naming Conventions

```css
:root {
  /* Semantic naming (recommended) */
  --color-primary: #6366f1;
  --color-background: #ffffff;
  --spacing-md: 1rem;
  
  /* Component namespacing */
  --button-bg: var(--color-primary);
  --button-radius: var(--border-radius);
  
  /* State variations */
  --color-primary-hover: #4f46e5;
  --color-primary-active: #4338ca;
}
```

---

## Using Custom Properties

Reference variables with `var()`:

```css
.button {
  background: var(--color-primary);
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
}

.card {
  border: 1px solid var(--color-border);
  background: var(--color-surface);
}
```

### Fallback Values

Provide a fallback if the variable is undefined:

```css
.button {
  /* Uses fallback if --color-primary is not defined */
  background: var(--color-primary, #6366f1);
  
  /* Nested fallbacks */
  color: var(--button-text, var(--color-text, #ffffff));
}
```

### Invalid Values

If a variable exists but has an invalid value for the property, the property resets to its inherited or initial value—not the fallback:

```css
:root {
  --size: red; /* Invalid for width */
}

.box {
  width: var(--size, 100px); /* Gets 'auto', not 100px */
}
```

---

## Scope and Inheritance

Custom properties follow the cascade and inherit like other properties.

### Global Scope

```css
:root {
  --global-color: #6366f1;
}

/* Available everywhere */
.any-element {
  color: var(--global-color);
}
```

### Local Scope

```css
.dark-section {
  --bg-color: #1e293b;
  --text-color: #e2e8f0;
  
  background: var(--bg-color);
  color: var(--text-color);
}

/* Children inherit */
.dark-section .card {
  background: var(--bg-color); /* Uses dark section's value */
}
```

### Overriding in Scope

```css
:root {
  --card-bg: #ffffff;
}

.dark-theme {
  --card-bg: #1e293b;
}

.card {
  background: var(--card-bg);
  /* White in normal context, dark in .dark-theme */
}
```

---

## Dynamic Theming

### Light/Dark Mode

```css
:root {
  /* Light theme (default) */
  --color-bg: #ffffff;
  --color-surface: #f8fafc;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-primary: #6366f1;
  --color-border: #e2e8f0;
}

/* Dark theme via class */
.dark {
  --color-bg: #0f172a;
  --color-surface: #1e293b;
  --color-text: #e2e8f0;
  --color-text-muted: #94a3b8;
  --color-primary: #818cf8;
  --color-border: #334155;
}

/* Or via media query */
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #0f172a;
    --color-surface: #1e293b;
    --color-text: #e2e8f0;
    --color-text-muted: #94a3b8;
    --color-primary: #818cf8;
    --color-border: #334155;
  }
}
```

### Component Variants

```css
.button {
  --button-bg: var(--color-primary);
  --button-text: #ffffff;
  --button-border: transparent;
  
  background: var(--button-bg);
  color: var(--button-text);
  border: 1px solid var(--button-border);
}

.button--secondary {
  --button-bg: transparent;
  --button-text: var(--color-primary);
  --button-border: var(--color-primary);
}

.button--danger {
  --button-bg: #ef4444;
  --button-text: #ffffff;
}
```

---

## JavaScript Integration

### Reading Variables

```javascript
// Get computed value
const styles = getComputedStyle(document.documentElement);
const primaryColor = styles.getPropertyValue('--color-primary').trim();
console.log(primaryColor); // "#6366f1"
```

### Setting Variables

```javascript
// Set on root (global)
document.documentElement.style.setProperty('--color-primary', '#10b981');

// Set on specific element
const card = document.querySelector('.card');
card.style.setProperty('--card-bg', '#fef2f2');
```

### Theme Toggle

```javascript
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.classList.contains('dark');
  
  html.classList.toggle('dark', !isDark);
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}

// Restore on load
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') {
  document.documentElement.classList.add('dark');
}
```

### Dynamic Values from JavaScript

```javascript
// Set accent color from AI response
function setAccentFromSentiment(sentiment) {
  const colors = {
    positive: '#10b981',
    neutral: '#6b7280',
    negative: '#ef4444'
  };
  
  document.documentElement.style.setProperty(
    '--accent-color', 
    colors[sentiment] || colors.neutral
  );
}

// Adjust based on scroll position
window.addEventListener('scroll', () => {
  const progress = window.scrollY / (document.body.scrollHeight - window.innerHeight);
  document.documentElement.style.setProperty('--scroll-progress', progress);
});
```

---

## Design Token System

Create a structured token system:

### Primitive Tokens

Raw values without semantic meaning:

```css
:root {
  /* Colors */
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-300: #d1d5db;
  --gray-400: #9ca3af;
  --gray-500: #6b7280;
  --gray-600: #4b5563;
  --gray-700: #374151;
  --gray-800: #1f2937;
  --gray-900: #111827;
  
  --indigo-500: #6366f1;
  --indigo-600: #4f46e5;
  --indigo-700: #4338ca;
  
  /* Spacing */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;
  
  /* Typography */
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;
  --font-size-2xl: 1.5rem;
}
```

### Semantic Tokens

Map primitives to meaning:

```css
:root {
  /* Colors */
  --color-background: var(--gray-50);
  --color-surface: #ffffff;
  --color-text: var(--gray-900);
  --color-text-secondary: var(--gray-600);
  --color-primary: var(--indigo-500);
  --color-primary-hover: var(--indigo-600);
  --color-border: var(--gray-200);
  
  /* Spacing */
  --spacing-xs: var(--space-1);
  --spacing-sm: var(--space-2);
  --spacing-md: var(--space-4);
  --spacing-lg: var(--space-6);
  --spacing-xl: var(--space-8);
  
  /* Component tokens */
  --button-padding-x: var(--spacing-md);
  --button-padding-y: var(--spacing-sm);
  --card-padding: var(--spacing-lg);
  --input-border: var(--color-border);
}
```

### Theme Switching

```css
[data-theme="dark"] {
  --color-background: var(--gray-900);
  --color-surface: var(--gray-800);
  --color-text: var(--gray-50);
  --color-text-secondary: var(--gray-400);
  --color-border: var(--gray-700);
  /* Primary can stay the same or shift */
  --color-primary: var(--indigo-400);
}
```

---

## `@property` - Typed Custom Properties

Register custom properties with type information for:
- Animated transitions
- Default values
- Inheritance control

```css
@property --gradient-angle {
  syntax: '<angle>';
  initial-value: 0deg;
  inherits: false;
}

@property --progress {
  syntax: '<percentage>';
  initial-value: 0%;
  inherits: false;
}

@property --color-start {
  syntax: '<color>';
  initial-value: #6366f1;
  inherits: true;
}
```

### Animating Custom Properties

Without `@property`, browsers can't interpolate custom property values:

```css
/* Won't animate - browser doesn't know it's an angle */
:root {
  --angle: 0deg;
}

.gradient:hover {
  --angle: 360deg;
  transition: --angle 1s; /* Doesn't work */
}
```

With `@property`:

```css
@property --angle {
  syntax: '<angle>';
  initial-value: 0deg;
  inherits: false;
}

.gradient {
  background: linear-gradient(var(--angle), #6366f1, #10b981);
  transition: --angle 1s ease;
}

.gradient:hover {
  --angle: 360deg; /* Animates smoothly! */
}
```

### Progress Indicators

```css
@property --progress {
  syntax: '<percentage>';
  initial-value: 0%;
  inherits: false;
}

.progress-ring {
  background: conic-gradient(
    var(--color-primary) var(--progress),
    var(--color-border) var(--progress)
  );
  transition: --progress 0.5s ease;
}

.progress-ring[data-value="75"] {
  --progress: 75%;
}
```

---

## Calculations with `calc()`

Combine variables with calculations:

```css
:root {
  --spacing-unit: 8px;
  --base-size: 1rem;
}

.element {
  /* Arithmetic */
  padding: calc(var(--spacing-unit) * 2);
  margin: calc(var(--spacing-unit) / 2);
  
  /* Combining units */
  width: calc(100% - var(--sidebar-width));
  
  /* Complex calculations */
  font-size: calc(var(--base-size) + 0.5vw);
}
```

### Responsive Spacing

```css
:root {
  --spacing-scale: 1;
}

@media (min-width: 768px) {
  :root {
    --spacing-scale: 1.25;
  }
}

@media (min-width: 1024px) {
  :root {
    --spacing-scale: 1.5;
  }
}

.section {
  padding: calc(2rem * var(--spacing-scale));
}
```

---

## Common Patterns

### Component Configuration

```css
.card {
  /* Private variables with defaults */
  --_padding: var(--card-padding, 1.5rem);
  --_radius: var(--card-radius, 8px);
  --_shadow: var(--card-shadow, 0 1px 3px rgba(0,0,0,0.1));
  
  padding: var(--_padding);
  border-radius: var(--_radius);
  box-shadow: var(--_shadow);
}

/* Override for specific use */
.compact-cards .card {
  --card-padding: 1rem;
  --card-radius: 4px;
}
```

### Conditional Values

```css
:root {
  --is-mobile: 0; /* false */
}

@media (max-width: 767px) {
  :root {
    --is-mobile: 1; /* true */
  }
}

.element {
  /* Show on mobile, hide on desktop */
  opacity: var(--is-mobile);
  
  /* Different values based on condition */
  font-size: calc(
    var(--is-mobile) * 14px + 
    (1 - var(--is-mobile)) * 16px
  );
}
```

### Color Manipulation

```css
:root {
  --primary-h: 239;
  --primary-s: 84%;
  --primary-l: 67%;
  
  --color-primary: hsl(var(--primary-h), var(--primary-s), var(--primary-l));
  --color-primary-light: hsl(var(--primary-h), var(--primary-s), 80%);
  --color-primary-dark: hsl(var(--primary-h), var(--primary-s), 45%);
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use semantic naming | `--color-text` over `--gray-800` |
| Define at appropriate scope | Global for tokens, local for components |
| Provide fallbacks | Graceful degradation |
| Use private variables (`--_name`) | Clear internal vs. API distinction |
| Keep primitives separate from semantics | Easier theming |
| Use `@property` for animations | Enables smooth transitions |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No fallback values | Always provide: `var(--x, fallback)` |
| Variables in selectors | Variables only work in property values |
| Missing `var()` | Must use `var(--name)`, not just `--name` |
| Expecting fallback on invalid values | Invalid values don't use fallback |
| Not inheriting when needed | Check if property should inherit |
| Overusing variables | Not everything needs to be a variable |

### Fallback Gotcha

```css
:root {
  --width: red; /* Wrong type, but exists */
}

.box {
  /* Fallback NOT used - variable exists but is invalid */
  width: var(--width, 100px); /* Gets 'auto', not 100px */
}
```

---

## Hands-on Exercise

### Your Task

Create a chat interface theme system with:

1. Light and dark themes using CSS variables
2. Customizable accent color
3. Message bubble that adapts to theme
4. Smooth theme transition animation
5. JavaScript toggle for theme and accent color

### Requirements

1. Define color, spacing, and typography tokens
2. Theme class toggles all colors
3. Accent color changeable via JavaScript
4. Use `@property` for smooth color transitions

<details>
<summary>✅ Solution</summary>

```css
/* Typed properties for animation */
@property --bg-color {
  syntax: '<color>';
  initial-value: #ffffff;
  inherits: true;
}

@property --surface-color {
  syntax: '<color>';
  initial-value: #f8fafc;
  inherits: true;
}

@property --text-color {
  syntax: '<color>';
  initial-value: #1e293b;
  inherits: true;
}

/* Token system */
:root {
  /* Primitives */
  --white: #ffffff;
  --gray-50: #f8fafc;
  --gray-100: #f1f5f9;
  --gray-800: #1e293b;
  --gray-900: #0f172a;
  
  /* Semantic tokens - Light theme */
  --bg-color: var(--white);
  --surface-color: var(--gray-50);
  --text-color: var(--gray-800);
  --text-muted: #64748b;
  --border-color: #e2e8f0;
  
  /* Customizable accent */
  --accent-h: 239;
  --accent-s: 84%;
  --accent-l: 67%;
  --accent-color: hsl(var(--accent-h), var(--accent-s), var(--accent-l));
  --accent-hover: hsl(var(--accent-h), var(--accent-s), calc(var(--accent-l) - 10%));
  
  /* Spacing */
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  
  /* Transitions */
  --transition-theme: 0.3s ease;
}

/* Dark theme */
.dark {
  --bg-color: var(--gray-900);
  --surface-color: var(--gray-800);
  --text-color: #e2e8f0;
  --text-muted: #94a3b8;
  --border-color: #334155;
}

/* Chat container */
.chat-container {
  background: var(--bg-color);
  color: var(--text-color);
  transition: 
    --bg-color var(--transition-theme),
    --text-color var(--transition-theme);
  min-height: 100vh;
  padding: var(--space-lg);
}

/* Messages */
.message {
  padding: var(--space-md);
  border-radius: 1rem;
  max-width: 70%;
  margin-bottom: var(--space-sm);
  transition: background var(--transition-theme);
}

.message--received {
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-bottom-left-radius: 0.25rem;
  margin-right: auto;
}

.message--sent {
  background: var(--accent-color);
  color: white;
  border-bottom-right-radius: 0.25rem;
  margin-left: auto;
}

/* Input area */
.chat-input {
  display: flex;
  gap: var(--space-sm);
  padding: var(--space-md);
  background: var(--surface-color);
  border-top: 1px solid var(--border-color);
}

.chat-input input {
  flex: 1;
  padding: var(--space-sm) var(--space-md);
  border: 1px solid var(--border-color);
  border-radius: 9999px;
  background: var(--bg-color);
  color: var(--text-color);
}

.chat-input button {
  background: var(--accent-color);
  color: white;
  border: none;
  padding: var(--space-sm) var(--space-md);
  border-radius: 9999px;
  cursor: pointer;
  transition: background 0.2s;
}

.chat-input button:hover {
  background: var(--accent-hover);
}

/* Theme toggle */
.theme-controls {
  display: flex;
  gap: var(--space-md);
  padding: var(--space-md);
}

.color-option {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 2px solid transparent;
  cursor: pointer;
}

.color-option.active {
  border-color: var(--text-color);
}
```

```javascript
// Theme toggle
function toggleTheme() {
  document.documentElement.classList.toggle('dark');
  const isDark = document.documentElement.classList.contains('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Accent color
function setAccentColor(hue) {
  document.documentElement.style.setProperty('--accent-h', hue);
  localStorage.setItem('accent-hue', hue);
}

// Presets
const colorPresets = {
  indigo: 239,
  emerald: 158,
  rose: 350,
  amber: 38
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Restore theme
  if (localStorage.getItem('theme') === 'dark') {
    document.documentElement.classList.add('dark');
  }
  
  // Restore accent
  const savedHue = localStorage.getItem('accent-hue');
  if (savedHue) setAccentColor(savedHue);
});
```

```html
<div class="theme-controls">
  <button onclick="toggleTheme()">Toggle Theme</button>
  <div class="color-options">
    <button 
      class="color-option" 
      style="background: hsl(239, 84%, 67%)"
      onclick="setAccentColor(239)">
    </button>
    <button 
      class="color-option" 
      style="background: hsl(158, 64%, 52%)"
      onclick="setAccentColor(158)">
    </button>
    <button 
      class="color-option" 
      style="background: hsl(350, 89%, 60%)"
      onclick="setAccentColor(350)">
    </button>
  </div>
</div>

<div class="chat-container">
  <div class="message message--received">
    Hello! How can I help you today?
  </div>
  <div class="message message--sent">
    I'd like to customize my theme.
  </div>
</div>

<div class="chat-input">
  <input type="text" placeholder="Type a message...">
  <button>Send</button>
</div>
```
</details>

---

## Summary

✅ **CSS variables** are declared with `--name` and used with `var(--name)`

✅ Variables **cascade and inherit** like regular properties

✅ Use **`:root`** for global scope, element selectors for local scope

✅ Always provide **fallback values**: `var(--color, #default)`

✅ **JavaScript** can read and modify variables at runtime

✅ **`@property`** enables type checking and animated transitions

✅ Build **design token systems** with primitives and semantic layers

---

**Previous:** [Media Queries](./06-media-queries.md)

**Next:** [Transitions & Animations](./08-transitions-animations.md)

<!-- 
Sources Consulted:
- MDN CSS Custom Properties: https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties
- MDN @property: https://developer.mozilla.org/en-US/docs/Web/CSS/@property
- MDN var(): https://developer.mozilla.org/en-US/docs/Web/CSS/var
-->
