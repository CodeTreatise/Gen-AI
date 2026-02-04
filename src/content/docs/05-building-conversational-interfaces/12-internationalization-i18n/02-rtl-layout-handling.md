---
title: "RTL Layout Handling"
---

# RTL Layout Handling

## Introduction

When building chat interfaces for languages like Arabic, Hebrew, Farsi, or Urdu, the entire layout must flip. Text flows from right to left, the send button moves to the left, message bubbles align on opposite sides, and navigation reverses. Get this wrong, and your interface becomes unusable for millions of users.

Modern CSS provides tools that make RTL support almost automatic—if you build with them from the start. We'll use CSS Logical Properties and the HTML `dir` attribute to create layouts that adapt seamlessly to any text direction.

### What We'll Cover

- HTML `dir` attribute and document direction
- CSS Logical Properties for direction-agnostic layouts
- Chat-specific RTL patterns (message bubbles, input areas)
- Handling bidirectional text (mixed LTR and RTL)
- Testing RTL layouts effectively

### Prerequisites

- CSS box model (margin, padding, positioning)
- Flexbox and CSS Grid basics
- Understanding of chat message layouts

---

## The HTML dir Attribute

The `dir` attribute controls text direction for an element and its descendants.

### Values

| Value | Behavior | Use Case |
|-------|----------|----------|
| `ltr` | Left-to-right | English, Spanish, most languages |
| `rtl` | Right-to-left | Arabic, Hebrew, Farsi, Urdu |
| `auto` | Algorithm detects direction | User-generated content |

### Setting Document Direction

```html
<!-- English interface -->
<html lang="en" dir="ltr">

<!-- Arabic interface -->
<html lang="ar" dir="rtl">
```

### Direction Inheritance

Child elements inherit direction from their parent:

```html
<html dir="rtl">
  <body>           <!-- inherits rtl -->
    <main>         <!-- inherits rtl -->
      <div>        <!-- inherits rtl -->
        Content flows right-to-left
      </div>
    </main>
  </body>
</html>
```

### Overriding Direction

Override for specific elements when needed:

```html
<html dir="rtl">
  <body>
    <!-- Most content is RTL -->
    <main>المحتوى العربي</main>
    
    <!-- Code blocks should always be LTR -->
    <pre dir="ltr">const x = 10;</pre>
    
    <!-- Email addresses are typically LTR -->
    <span dir="ltr">user@example.com</span>
  </body>
</html>
```

### Setting Direction with JavaScript

```javascript
function setDocumentDirection(isRTL) {
  document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
}

// Detect RTL from locale
function isRTLLocale(locale) {
  const rtlLocales = ['ar', 'he', 'fa', 'ur', 'yi', 'ps', 'sd'];
  const baseLocale = locale.split('-')[0].toLowerCase();
  return rtlLocales.includes(baseLocale);
}

// Apply on locale change
function applyLocale(locale) {
  document.documentElement.lang = locale;
  document.documentElement.dir = isRTLLocale(locale) ? 'rtl' : 'ltr';
}

// Usage
applyLocale('ar-SA');  // Sets dir="rtl"
applyLocale('en-US');  // Sets dir="ltr"
```

---

## CSS Logical Properties

Logical properties replace physical directions (left, right, top, bottom) with flow-relative concepts (inline-start, inline-end, block-start, block-end).

### The Concept

| Physical Concept | In LTR | In RTL | Logical Alternative |
|------------------|--------|--------|---------------------|
| Left | Start | End | `inline-start` |
| Right | End | Start | `inline-end` |
| Top | Start | Start | `block-start` |
| Bottom | End | End | `block-end` |

### Property Mapping

| Physical Property | Logical Property |
|-------------------|------------------|
| `margin-left` | `margin-inline-start` |
| `margin-right` | `margin-inline-end` |
| `padding-left` | `padding-inline-start` |
| `padding-right` | `padding-inline-end` |
| `left` | `inset-inline-start` |
| `right` | `inset-inline-end` |
| `text-align: left` | `text-align: start` |
| `text-align: right` | `text-align: end` |
| `border-left` | `border-inline-start` |
| `border-right` | `border-inline-end` |
| `width` | `inline-size` |
| `height` | `block-size` |

### Practical Examples

**Before (Physical - Breaks in RTL):**
```css
.message-bubble {
  margin-left: 10px;
  margin-right: 60px;
  padding-left: 16px;
  padding-right: 16px;
  border-left: 3px solid blue;
  text-align: left;
}
```

**After (Logical - Works in Both):**
```css
.message-bubble {
  margin-inline-start: 10px;
  margin-inline-end: 60px;
  padding-inline: 16px;  /* Shorthand for both */
  border-inline-start: 3px solid blue;
  text-align: start;
}
```

### Shorthand Properties

```css
.element {
  /* Block (vertical) */
  margin-block: 10px 20px;     /* top bottom */
  padding-block: 16px;         /* both */
  
  /* Inline (horizontal, direction-aware) */
  margin-inline: 10px 20px;    /* start end */
  padding-inline: 16px;        /* both */
  
  /* Inset (positioning) */
  inset-block: 0;              /* top and bottom: 0 */
  inset-inline: 0 auto;        /* start: 0, end: auto */
}
```

---

## Chat Interface RTL Patterns

### Message Container Layout

```css
.chat-container {
  display: flex;
  flex-direction: column;
  block-size: 100vh;
  inline-size: 100%;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding-block: 16px;
  padding-inline: 12px;
}

.chat-input-area {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-block-start: 1px solid var(--border-color);
}
```

### Message Bubble Alignment

Incoming messages on one side, outgoing on the other—automatically flips in RTL:

```css
.message {
  display: flex;
  margin-block-end: 8px;
}

/* User messages: end of inline axis */
.message--user {
  justify-content: flex-end;
}

/* AI/incoming messages: start of inline axis */
.message--ai {
  justify-content: flex-start;
}

.message-bubble {
  max-inline-size: 70%;
  padding-block: 10px;
  padding-inline: 14px;
  border-radius: 16px;
}

.message--user .message-bubble {
  background: var(--user-bubble-color);
  border-end-end-radius: 4px;  /* Bottom-right in LTR, bottom-left in RTL */
}

.message--ai .message-bubble {
  background: var(--ai-bubble-color);
  border-end-start-radius: 4px;  /* Bottom-left in LTR, bottom-right in RTL */
}
```

### Logical Border Radius

The `border-radius` logical properties use block/inline + start/end:

| Physical | Logical |
|----------|---------|
| `border-top-left-radius` | `border-start-start-radius` |
| `border-top-right-radius` | `border-start-end-radius` |
| `border-bottom-left-radius` | `border-end-start-radius` |
| `border-bottom-right-radius` | `border-end-end-radius` |

```css
/* Message bubble with tail on one corner */
.message-bubble--outgoing {
  border-radius: 16px;
  border-end-end-radius: 4px;  /* Tail corner */
}

.message-bubble--incoming {
  border-radius: 16px;
  border-end-start-radius: 4px;  /* Tail corner */
}
```

### Chat Input Area

```css
.chat-input {
  flex: 1;
  padding-block: 12px;
  padding-inline: 16px;
  border: 1px solid var(--border-color);
  border-radius: 24px;
  text-align: start;  /* Follows text direction */
}

.send-button {
  /* Position adapts automatically with flexbox */
  padding-block: 12px;
  padding-inline: 20px;
}

/* Attachment button before input */
.chat-input-area {
  display: flex;
  align-items: center;
  gap: 8px;
  /* In LTR: [attach] [input...] [send]
     In RTL: [send] [...input] [attach] */
}
```

### Navigation and Actions

```css
.chat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-block: 12px;
  padding-inline: 16px;
}

.back-button {
  /* Flexbox handles position */
  display: flex;
  align-items: center;
}

/* Flip arrow icon for RTL */
.back-button svg {
  transform: scaleX(1);  /* Default */
}

[dir="rtl"] .back-button svg {
  transform: scaleX(-1);  /* Flip horizontally */
}

.chat-actions {
  margin-inline-start: auto;  /* Push to end */
  display: flex;
  gap: 8px;
}
```

---

## Bidirectional Text Handling

Mixed-direction content requires special handling.

### The dir="auto" Value

Use `auto` when you don't know the text direction in advance:

```html
<!-- User-generated content -->
<div class="message-content" dir="auto">
  ${userMessage}
</div>
```

The algorithm looks at the first "strong" character (letter with inherent direction) to determine direction.

### Unicode Bidirectional Algorithm

```javascript
// Check if text starts with RTL character
function detectTextDirection(text) {
  const rtlChars = /[\u0591-\u07FF\u200F\u202B\u202E\uFB1D-\uFDFD\uFE70-\uFEFC]/;
  const ltrChars = /[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02B8]/;
  
  for (const char of text) {
    if (rtlChars.test(char)) return 'rtl';
    if (ltrChars.test(char)) return 'ltr';
  }
  
  return 'auto';
}

// Apply to message element
function renderMessage(text) {
  const direction = detectTextDirection(text);
  return `<div class="message-content" dir="${direction}">${escapeHtml(text)}</div>`;
}
```

### Embedding Direction

Use `<bdi>` (Bidirectional Isolate) for embedded content that might have different direction:

```html
<p>
  User <bdi>מרים</bdi> sent a message.
</p>

<p>
  User <bdi>Ahmed</bdi> sent a message.
</p>
```

The `<bdi>` element isolates the content from surrounding text, preventing display issues.

### Mixed Content Examples

```html
<!-- Chat showing user with Hebrew name -->
<div class="message" dir="ltr">
  <span class="username">
    <bdi>דוד</bdi>
  </span>
  <span class="message-content" dir="auto">
    Hello, how are you?
  </span>
</div>

<!-- Arabic message with English code -->
<div class="message" dir="rtl">
  <span class="message-content">
    استخدم الأمر <code dir="ltr">npm install</code> لتثبيت الحزم
  </span>
</div>
```

---

## Complete RTL Chat Styles

```css
/* ===== Base Reset ===== */
* {
  box-sizing: border-box;
}

/* ===== Chat Container ===== */
.chat-app {
  display: flex;
  flex-direction: column;
  block-size: 100vh;
  inline-size: 100%;
  max-inline-size: 800px;
  margin-inline: auto;
}

/* ===== Header ===== */
.chat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-block: 12px;
  padding-inline: 16px;
  border-block-end: 1px solid var(--border-color);
  background: var(--header-bg);
}

.back-button {
  display: flex;
  align-items: center;
  justify-content: center;
  inline-size: 40px;
  block-size: 40px;
  border-radius: 50%;
  border: none;
  background: transparent;
  cursor: pointer;
}

.back-button:hover {
  background: var(--hover-bg);
}

/* Flip directional icons */
[dir="rtl"] .icon-arrow-back,
[dir="rtl"] .icon-arrow-forward,
[dir="rtl"] .icon-chevron-right {
  transform: scaleX(-1);
}

.chat-title {
  flex: 1;
  font-size: 1.125rem;
  font-weight: 600;
  text-align: start;
}

.chat-actions {
  display: flex;
  gap: 4px;
}

/* ===== Message List ===== */
.message-list {
  flex: 1;
  overflow-y: auto;
  padding-block: 16px;
  padding-inline: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* ===== Message Bubbles ===== */
.message {
  display: flex;
  align-items: flex-end;
  gap: 8px;
}

.message--user {
  flex-direction: row-reverse;
}

.message--ai {
  flex-direction: row;
}

.message-avatar {
  inline-size: 32px;
  block-size: 32px;
  border-radius: 50%;
  flex-shrink: 0;
}

.message-bubble {
  max-inline-size: 70%;
  padding-block: 10px;
  padding-inline: 14px;
  line-height: 1.5;
}

.message--user .message-bubble {
  background: var(--user-bubble-bg);
  color: var(--user-bubble-text);
  border-radius: 16px;
  border-end-end-radius: 4px;
}

.message--ai .message-bubble {
  background: var(--ai-bubble-bg);
  color: var(--ai-bubble-text);
  border-radius: 16px;
  border-end-start-radius: 4px;
}

/* ===== Message Content ===== */
.message-content {
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.message-content code {
  direction: ltr;  /* Code is always LTR */
  unicode-bidi: embed;
}

.message-time {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-block-start: 4px;
  text-align: end;
}

/* ===== Typing Indicator ===== */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-inline-start: 48px;  /* Align with AI messages */
  color: var(--text-secondary);
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dot {
  inline-size: 8px;
  block-size: 8px;
  border-radius: 50%;
  background: var(--text-secondary);
  animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-8px); }
}

/* ===== Input Area ===== */
.chat-input-area {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding-block: 12px;
  padding-inline: 16px;
  border-block-start: 1px solid var(--border-color);
  background: var(--input-area-bg);
}

.attach-button {
  display: flex;
  align-items: center;
  justify-content: center;
  inline-size: 44px;
  block-size: 44px;
  border-radius: 50%;
  border: none;
  background: transparent;
  cursor: pointer;
  flex-shrink: 0;
}

.chat-input {
  flex: 1;
  min-block-size: 44px;
  max-block-size: 120px;
  padding-block: 10px;
  padding-inline: 16px;
  border: 1px solid var(--border-color);
  border-radius: 22px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  line-height: 1.5;
  text-align: start;
}

.chat-input:focus {
  outline: none;
  border-color: var(--focus-color);
}

.send-button {
  display: flex;
  align-items: center;
  justify-content: center;
  inline-size: 44px;
  block-size: 44px;
  border-radius: 50%;
  border: none;
  background: var(--primary-color);
  color: white;
  cursor: pointer;
  flex-shrink: 0;
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ===== RTL-Specific Adjustments ===== */
[dir="rtl"] .message--user {
  flex-direction: row;  /* Reverse of LTR reverse */
}

[dir="rtl"] .message--ai {
  flex-direction: row-reverse;  /* Reverse of LTR normal */
}

/* For elements that should maintain LTR regardless */
.ltr-only {
  direction: ltr;
  unicode-bidi: isolate;
}

/* Code blocks always LTR */
pre, code {
  direction: ltr;
  text-align: left;
  unicode-bidi: isolate;
}

/* Numbers often render better as LTR */
.numeric {
  direction: ltr;
  unicode-bidi: isolate;
}

/* ===== CSS Variables ===== */
:root {
  --border-color: #e0e0e0;
  --header-bg: #ffffff;
  --input-area-bg: #ffffff;
  --hover-bg: rgba(0, 0, 0, 0.05);
  --primary-color: #0066cc;
  --focus-color: #0066cc;
  --text-secondary: #666666;
  
  --user-bubble-bg: #0066cc;
  --user-bubble-text: #ffffff;
  --ai-bubble-bg: #f0f0f0;
  --ai-bubble-text: #1a1a1a;
}
```

---

## Testing RTL Layouts

### Browser DevTools Testing

```javascript
// Quick toggle for testing
function toggleRTL() {
  const html = document.documentElement;
  html.dir = html.dir === 'rtl' ? 'ltr' : 'rtl';
}

// Add to console or as a button
document.addEventListener('keydown', (e) => {
  // Ctrl+Shift+R to toggle RTL
  if (e.ctrlKey && e.shiftKey && e.key === 'R') {
    e.preventDefault();
    toggleRTL();
  }
});
```

### RTL Testing Checklist

| Check | What to Look For |
|-------|------------------|
| Layout flip | All elements mirror correctly |
| Text alignment | Text starts from correct side |
| Icon direction | Arrows, chevrons point correctly |
| Scrollbars | Appear on correct side |
| Input cursor | Appears at correct position |
| Message bubbles | User/AI sides swap correctly |
| Borders/shadows | Positioned on correct side |
| Animations | Slide directions reversed |

### Automated RTL Testing

```javascript
describe('RTL Layout', () => {
  beforeEach(() => {
    document.documentElement.dir = 'rtl';
    document.documentElement.lang = 'ar';
  });
  
  afterEach(() => {
    document.documentElement.dir = 'ltr';
    document.documentElement.lang = 'en';
  });
  
  test('user messages align to left in RTL', () => {
    const message = document.querySelector('.message--user');
    const rect = message.getBoundingClientRect();
    expect(rect.left).toBeLessThan(window.innerWidth / 2);
  });
  
  test('AI messages align to right in RTL', () => {
    const message = document.querySelector('.message--ai');
    const rect = message.getBoundingClientRect();
    expect(rect.right).toBeGreaterThan(window.innerWidth / 2);
  });
  
  test('input text aligns to right', () => {
    const input = document.querySelector('.chat-input');
    const styles = getComputedStyle(input);
    expect(styles.textAlign).toBe('start');  // Resolves to 'right' in RTL
  });
});
```

### Visual Regression Testing

```javascript
// With Playwright
test('chat interface RTL layout', async ({ page }) => {
  await page.goto('/chat');
  
  // Set RTL
  await page.evaluate(() => {
    document.documentElement.dir = 'rtl';
    document.documentElement.lang = 'ar';
  });
  
  // Take screenshot for comparison
  await expect(page).toHaveScreenshot('chat-rtl.png');
});
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `margin-left/right` | Use `margin-inline-start/end` |
| Using `text-align: left` | Use `text-align: start` |
| Hardcoding arrow directions | Flip with `transform: scaleX(-1)` in RTL |
| Forgetting code blocks | Always set `dir="ltr"` on code |
| Using `float: left/right` | Use Flexbox with logical alignment |
| Not testing early | Test RTL from the start of development |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use CSS Logical Properties everywhere | One codebase for all directions |
| Set `dir` on `<html>` element | Proper inheritance for all elements |
| Use `dir="auto"` for user content | Handles mixed-direction messages |
| Keep code/URLs as `dir="ltr"` | Technical content reads left-to-right |
| Use `<bdi>` for embedded names | Prevents direction conflicts |
| Flip directional icons via CSS | Icons mirror with layout |

---

## Hands-on Exercise

### Your Task

Create a chat message component that works in both LTR and RTL layouts.

### Requirements

1. Style user messages with a bubble tail on the end side
2. Style AI messages with a bubble tail on the start side
3. Position avatars correctly for both directions
4. Handle mixed-direction content in messages
5. Add a toggle button to switch directions

### Expected Result

- In LTR: User messages on right, AI on left
- In RTL: User messages on left, AI on right
- Code blocks always appear LTR regardless of interface direction

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `flex-direction: row-reverse` for user messages
- Apply `border-end-end-radius` for the bubble tail
- Use `dir="auto"` on message content
- Add `unicode-bidi: isolate` for embedded content

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <style>
    .message {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      margin-block-end: 12px;
    }
    
    .message--user {
      flex-direction: row-reverse;
    }
    
    .avatar {
      inline-size: 32px;
      block-size: 32px;
      border-radius: 50%;
      background: #ccc;
      flex-shrink: 0;
    }
    
    .bubble {
      max-inline-size: 60%;
      padding: 10px 14px;
      border-radius: 16px;
    }
    
    .message--user .bubble {
      background: #0066cc;
      color: white;
      border-end-end-radius: 4px;
    }
    
    .message--ai .bubble {
      background: #f0f0f0;
      border-end-start-radius: 4px;
    }
    
    .content {
      word-wrap: break-word;
    }
    
    .content code {
      direction: ltr;
      unicode-bidi: isolate;
    }
    
    .toggle-btn {
      position: fixed;
      inset-block-start: 10px;
      inset-inline-end: 10px;
      padding: 8px 16px;
    }
  </style>
</head>
<body>
  <button class="toggle-btn" onclick="toggleDir()">Toggle RTL</button>
  
  <div class="message message--ai">
    <div class="avatar"></div>
    <div class="bubble">
      <div class="content" dir="auto">Hello! How can I help?</div>
    </div>
  </div>
  
  <div class="message message--user">
    <div class="avatar"></div>
    <div class="bubble">
      <div class="content" dir="auto">مرحبا! كيف يمكنك المساعدة؟</div>
    </div>
  </div>
  
  <div class="message message--ai">
    <div class="bubble">
      <div class="content" dir="auto">
        استخدم <code>npm install</code> للتثبيت
      </div>
    </div>
  </div>
  
  <script>
    function toggleDir() {
      const html = document.documentElement;
      html.dir = html.dir === 'rtl' ? 'ltr' : 'rtl';
      html.lang = html.dir === 'rtl' ? 'ar' : 'en';
    }
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ Set `dir="rtl"` on the document for RTL languages, with JavaScript detection

✅ Replace physical CSS properties (`margin-left`) with logical ones (`margin-inline-start`)

✅ Use `flex-direction` and logical alignment for layout flipping

✅ Apply `dir="auto"` to user-generated content for mixed-direction support

✅ Always keep code, URLs, and numbers in `dir="ltr"`

**Next:** [Date and Time Localization](./03-date-time-localization.md)

---

## Further Reading

- [MDN: CSS Logical Properties](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_logical_properties_and_values) - Complete reference
- [MDN: dir attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/dir) - Direction attribute
- [web.dev: Building RTL-aware components](https://web.dev/learn/design/internationalization/) - Practical guide
- [RTL Styling 101](https://rtlstyling.com/) - Comprehensive RTL guide

<!--
Sources Consulted:
- MDN CSS Logical Properties: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_logical_properties_and_values
- MDN dir attribute: https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/dir
- W3C Writing Modes: https://www.w3.org/TR/css-writing-modes-4/
-->
