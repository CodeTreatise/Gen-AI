---
title: "Keyboard Accessibility"
---

# Keyboard Accessibility

## Introduction

Many users navigate entirely with a keyboard—people with motor disabilities, power users, and those using assistive technologies. If your site isn't keyboard accessible, you're excluding a significant portion of users.

This lesson covers focus management, tab order, focus trapping for modals, and proper keyboard event handling.

### What We'll Cover

- Focus management
- Tab order and tabindex
- Focus trapping for modals
- Skip links
- Keyboard event handling

### Prerequisites

- HTML and JavaScript fundamentals
- Understanding of interactive elements
- Basic CSS for focus styling

---

## Focus Fundamentals

### What Is Focus?

Focus indicates which element receives keyboard input:

```
Tab key moves focus through interactive elements:

[Button] → [Link] → [Input] → [Checkbox] → [Button]
    ↑                                          │
    └──────────────────────────────────────────┘
```

### Focusable Elements

Naturally focusable (in the tab order):
- `<a href="...">` (with href)
- `<button>`
- `<input>`, `<select>`, `<textarea>`
- `<iframe>`
- Elements with `tabindex="0"`

NOT focusable by default:
- `<div>`, `<span>`, `<p>`
- `<a>` without href
- Disabled elements

### Focus Indicators

Never remove focus outlines without replacement:

```css
/* ❌ NEVER do this */
:focus {
  outline: none;
}

/* ✅ Custom focus styles */
:focus {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}

/* ✅ For modern browsers */
:focus-visible {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}

/* Remove outline only on mouse click, keep for keyboard */
:focus:not(:focus-visible) {
  outline: none;
}
```

### focus-visible

`:focus-visible` applies only when focus should be visible (keyboard navigation, not mouse):

```css
/* Focus style only for keyboard users */
button:focus-visible {
  outline: 2px solid blue;
}

/* Optional: Remove outline for mouse users */
button:focus:not(:focus-visible) {
  outline: none;
}
```

---

## Tabindex

Control tab order and focusability.

### Tabindex Values

| Value | Behavior |
|-------|----------|
| Not set | Default behavior (focusable if interactive) |
| `0` | Add to natural tab order |
| `-1` | Focusable via JS only, not in tab order |
| `1+` | ⚠️ AVOID - forces specific order |

### Using tabindex="0"

Make non-interactive elements focusable:

```html
<!-- Custom interactive element -->
<div 
  role="button" 
  tabindex="0"
  onclick="doSomething()"
  onkeydown="handleKeydown(event)">
  Click me
</div>
```

> **Note:** If you use `tabindex="0"`, you must also handle keyboard events!

### Using tabindex="-1"

Programmatically focusable but not in tab order:

```html
<!-- Skip link target -->
<main id="main-content" tabindex="-1">
  <h1>Page Title</h1>
</main>

<!-- Error message to focus -->
<div id="error" role="alert" tabindex="-1">
  Error: Please fix the form.
</div>
```

```javascript
// Focus programmatically
document.getElementById('error').focus();
```

### Avoid Positive Tabindex

```html
<!-- ❌ Don't do this -->
<input tabindex="3">
<input tabindex="1">
<input tabindex="2">

<!-- Creates confusing order: 2nd, 3rd, 1st -->
```

Positive values override DOM order, creating maintenance nightmares.

---

## Tab Order

### Natural Tab Order

Tab follows DOM order, not visual order:

```html
<button>First</button>
<button>Second</button>  <!-- DOM order -->
<button>Third</button>
```

```css
.buttons {
  display: flex;
  flex-direction: row-reverse;  /* Visual: Third, Second, First */
}
/* Tab order: First, Second, Third (DOM order) */
```

### CSS and Tab Order

CSS can create visual/DOM mismatches:

```css
/* ❌ Visual order != DOM order */
.grid {
  display: grid;
  grid-template-areas: "c a b";
}

/* ✅ Match DOM to visual order instead */
```

### Checking Tab Order

1. Click at start of page
2. Press Tab repeatedly
3. Verify:
   - All interactive elements receive focus
   - Order makes sense
   - Focus is always visible

---

## Focus Trapping

### Why Trap Focus?

When a modal opens, focus should:
1. Move into the modal
2. Stay within the modal
3. Return to trigger on close

```
Page (inert while modal open)
┌─────────────────────────────────────┐
│  [Button] [Link] [Input]            │
│                                     │
│   ┌─ Modal ─────────────────────┐   │
│   │  [Close]                    │   │
│   │                             │   │
│   │  [Input] [Input]            │   │
│   │                             │   │
│   │  [Cancel] [Submit]          │   │
│   └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘

Focus can only move between elements inside modal
```

### Focus Trap Implementation

```javascript
function trapFocus(modal) {
  const focusableSelector = 
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
  const focusables = modal.querySelectorAll(focusableSelector);
  const firstFocusable = focusables[0];
  const lastFocusable = focusables[focusables.length - 1];
  
  modal.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;
    
    if (e.shiftKey) {
      // Shift+Tab
      if (document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    }
  });
}
```

### Using inert

The modern approach—make outside content inert:

```html
<main id="main" inert>
  <!-- Cannot interact with anything here -->
</main>

<dialog id="modal">
  <h2>Modal Title</h2>
  <button>Close</button>
</dialog>
```

```javascript
function openModal() {
  document.getElementById('main').inert = true;
  const modal = document.getElementById('modal');
  modal.showModal();  // <dialog> handles focus!
}

function closeModal() {
  document.getElementById('main').inert = false;
  document.getElementById('modal').close();
  // Focus returns to trigger automatically
}
```

### Dialog Element

Native `<dialog>` handles much of this automatically:

```html
<button onclick="document.getElementById('dialog').showModal()">
  Open
</button>

<dialog id="dialog">
  <h2>Confirm Action</h2>
  <p>Are you sure?</p>
  <button onclick="this.closest('dialog').close()">Cancel</button>
  <button onclick="confirm()">Confirm</button>
</dialog>
```

---

## Skip Links

Allow keyboard users to skip repetitive navigation.

### Implementation

```html
<body>
  <!-- First focusable element -->
  <a href="#main-content" class="skip-link">
    Skip to main content
  </a>
  
  <nav>
    <!-- Long navigation -->
  </nav>
  
  <main id="main-content" tabindex="-1">
    <h1>Page Title</h1>
  </main>
</body>
```

```css
.skip-link {
  position: absolute;
  top: -100%;
  left: 0;
  background: #000;
  color: #fff;
  padding: 0.5rem 1rem;
  z-index: 1000;
}

.skip-link:focus {
  top: 0;  /* Visible when focused */
}
```

### How Skip Links Work

1. User presses Tab on page load
2. Skip link receives focus, becomes visible
3. User presses Enter to skip
4. Focus moves to main content
5. User continues tabbing through main content

---

## Keyboard Event Handling

### Essential Keys

| Key | Common Action |
|-----|---------------|
| Tab | Move to next focusable |
| Shift+Tab | Move to previous focusable |
| Enter | Activate button/link |
| Space | Activate button, toggle checkbox |
| Escape | Close dialog, cancel action |
| Arrow keys | Navigate within widgets |

### Handling Keyboard Events

```javascript
// Custom button
element.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    activate();
  }
});

// Modal close on Escape
modal.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeModal();
  }
});

// Arrow key navigation
list.addEventListener('keydown', (event) => {
  const items = list.querySelectorAll('[role="option"]');
  const current = list.querySelector('[aria-selected="true"]');
  let next;
  
  switch (event.key) {
    case 'ArrowDown':
      next = current.nextElementSibling || items[0];
      break;
    case 'ArrowUp':
      next = current.previousElementSibling || items[items.length - 1];
      break;
    case 'Home':
      next = items[0];
      break;
    case 'End':
      next = items[items.length - 1];
      break;
    default:
      return;
  }
  
  event.preventDefault();
  selectItem(next);
  next.focus();
});
```

### Roving Tabindex

For composite widgets (tabs, menus), only one item should be in the tab order:

```html
<div role="tablist">
  <button role="tab" tabindex="0" aria-selected="true">Tab 1</button>
  <button role="tab" tabindex="-1" aria-selected="false">Tab 2</button>
  <button role="tab" tabindex="-1" aria-selected="false">Tab 3</button>
</div>
```

```javascript
tabs.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') {
    // Move tabindex and focus to next tab
    currentTab.tabIndex = -1;
    nextTab.tabIndex = 0;
    nextTab.focus();
  }
});
```

---

## Common Keyboard Patterns

### Disclosure (Expand/Collapse)

```javascript
button.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    toggleExpanded();
  }
});
```

### Menu

```javascript
menu.addEventListener('keydown', (e) => {
  switch (e.key) {
    case 'ArrowDown':
      focusNextItem();
      break;
    case 'ArrowUp':
      focusPreviousItem();
      break;
    case 'Escape':
      closeMenu();
      triggerButton.focus();
      break;
  }
});
```

### Modal Dialog

```javascript
dialog.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeDialog();
    opener.focus();  // Return focus to trigger
  }
});
```

---

## Testing Keyboard Accessibility

### Manual Testing Checklist

1. ☐ Can navigate with Tab alone
2. ☐ Focus order is logical
3. ☐ Focus indicator always visible
4. ☐ All actions possible via keyboard
5. ☐ No keyboard traps
6. ☐ Skip link present and working
7. ☐ Modals trap focus
8. ☐ Escape closes dialogs

### Browser DevTools

1. Open DevTools
2. Elements panel → Accessibility tab
3. Check focusable elements and tab order

---

## Hands-on Exercise

### Your Task

Make this modal keyboard accessible:

```html
<button onclick="openModal()">Open Modal</button>

<div id="modal" class="modal" style="display:none;">
  <div class="modal-content">
    <span class="close" onclick="closeModal()">&times;</span>
    <h2>Modal Title</h2>
    <input type="text" placeholder="Name">
    <button onclick="submit()">Submit</button>
  </div>
</div>
```

### Requirements

1. Focus moves into modal on open
2. Focus is trapped within modal
3. Escape key closes modal
4. Focus returns to trigger button on close

<details>
<summary>✅ Solution</summary>

```html
<button id="open-btn" onclick="openModal()">Open Modal</button>

<div 
  id="modal" 
  class="modal" 
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  hidden>
  <div class="modal-content">
    <button class="close" onclick="closeModal()" aria-label="Close">×</button>
    <h2 id="modal-title">Modal Title</h2>
    <label for="name">Name</label>
    <input type="text" id="name">
    <button onclick="submit()">Submit</button>
  </div>
</div>

<script>
let lastFocused;

function openModal() {
  lastFocused = document.activeElement;
  const modal = document.getElementById('modal');
  modal.hidden = false;
  modal.querySelector('input').focus();
  document.body.classList.add('modal-open');
  
  modal.addEventListener('keydown', trapFocus);
}

function closeModal() {
  const modal = document.getElementById('modal');
  modal.hidden = true;
  modal.removeEventListener('keydown', trapFocus);
  lastFocused.focus();
}

function trapFocus(e) {
  if (e.key === 'Escape') {
    closeModal();
    return;
  }
  
  if (e.key !== 'Tab') return;
  
  const modal = document.getElementById('modal');
  const focusables = modal.querySelectorAll(
    'button, input, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}
</script>
```
</details>

---

## Summary

✅ **Focus indicators** must always be visible
✅ Use **tabindex="0"** sparingly, with keyboard handlers
✅ Use **tabindex="-1"** for programmatic focus
✅ **Avoid positive tabindex** values
✅ **Tab order** should follow visual/logical order
✅ **Trap focus** in modals, return on close
✅ **Skip links** help bypass repetitive content
✅ Handle **Enter, Space, Escape, Arrow keys** appropriately

**Next:** [Testing Accessibility](./05-testing-accessibility.md)

---

## Further Reading

- [WebAIM Keyboard Accessibility](https://webaim.org/techniques/keyboard/)
- [Focus Management (MDN)](https://developer.mozilla.org/en-US/docs/Web/Accessibility/Keyboard-navigable_JavaScript_widgets)
- [APG Keyboard Patterns](https://www.w3.org/WAI/ARIA/apg/practices/keyboard-interface/)

<!-- 
Sources Consulted:
- WebAIM Keyboard: https://webaim.org/techniques/keyboard/
- WAI-ARIA APG: https://www.w3.org/WAI/ARIA/apg/
-->
