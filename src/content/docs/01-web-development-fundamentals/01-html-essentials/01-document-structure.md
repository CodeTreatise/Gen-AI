---
title: "Document Structure & Doctype"
---

# Document Structure & Doctype

## Introduction

Every HTML document follows a specific structure that tells browsers how to interpret and render content. Understanding this structure is essential—it's the foundation upon which every web page, including AI-powered interfaces, is built.

In this lesson, we'll explore the anatomy of an HTML document, from the doctype declaration to the closing `</html>` tag.

### What We'll Cover

- The `<!DOCTYPE html>` declaration and its purpose
- Essential structural elements: `<html>`, `<head>`, `<body>`
- Character encoding with `<meta charset>`
- Viewport configuration for responsive design
- The document outline and nesting rules

### Prerequisites

- A text editor (VS Code recommended)
- A web browser for testing
- Basic understanding of what HTML is

---

## The DOCTYPE Declaration

Every HTML5 document begins with a doctype declaration:

```html
<!DOCTYPE html>
```

### What It Does

The doctype tells the browser which version of HTML to use. `<!DOCTYPE html>` activates **standards mode**, ensuring consistent rendering across browsers.

### Why It Matters

| Mode | Behavior |
|------|----------|
| **Standards mode** | Browser follows HTML/CSS specifications exactly |
| **Quirks mode** | Browser emulates old, inconsistent behaviors |

Without a doctype, browsers fall back to quirks mode, causing layout inconsistencies.

> **Note:** Always place `<!DOCTYPE html>` on the very first line—no whitespace or comments before it.

---

## The HTML Element

The `<html>` element wraps all content on the page:

```html
<!DOCTYPE html>
<html lang="en">
  <!-- Everything goes here -->
</html>
```

### The `lang` Attribute

The `lang` attribute specifies the document's language:

```html
<html lang="en">      <!-- English -->
<html lang="es">      <!-- Spanish -->
<html lang="zh-CN">   <!-- Simplified Chinese -->
<html lang="ar">      <!-- Arabic -->
```

### Why `lang` Matters

| Benefit | Description |
|---------|-------------|
| **Screen readers** | Announce content with correct pronunciation |
| **Search engines** | Index content for the right language |
| **Translation tools** | Detect source language accurately |
| **CSS selectors** | Style based on language (`:lang(en)`) |

---

## The Head Section

The `<head>` contains metadata—information *about* the document, not visible content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Web Page</title>
</head>
<body>
  <!-- Visible content here -->
</body>
</html>
```

### Essential Head Elements

#### 1. Character Encoding

```html
<meta charset="UTF-8">
```

UTF-8 supports virtually all characters from all languages. Always declare it first in `<head>`.

#### 2. Viewport Meta Tag

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

This is **essential for responsive design**:

| Property | Purpose |
|----------|---------|
| `width=device-width` | Match viewport to device screen width |
| `initial-scale=1.0` | Set initial zoom level to 100% |

Without this, mobile browsers render pages at desktop width and zoom out.

#### 3. Document Title

```html
<title>Page Title | Site Name</title>
```

The title appears in:
- Browser tabs
- Bookmarks
- Search engine results
- Social media shares (as fallback)

---

## The Body Section

The `<body>` contains all visible content:

```html
<body>
  <header>
    <h1>Welcome to My Site</h1>
  </header>
  
  <main>
    <p>This is the main content.</p>
  </main>
  
  <footer>
    <p>&copy; 2025 My Site</p>
  </footer>
</body>
```

Everything users see and interact with goes inside `<body>`.

---

## Complete Document Template

Here's a complete, modern HTML5 template:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Character encoding - MUST be first -->
  <meta charset="UTF-8">
  
  <!-- Responsive viewport -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <!-- Page title -->
  <title>My Web Application</title>
  
  <!-- Description for SEO -->
  <meta name="description" content="A brief description of this page">
  
  <!-- Favicon -->
  <link rel="icon" href="/favicon.ico">
  
  <!-- Stylesheets -->
  <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
  <header>
    <nav>
      <a href="/">Home</a>
      <a href="/about">About</a>
    </nav>
  </header>
  
  <main>
    <h1>Page Heading</h1>
    <p>Main content goes here.</p>
  </main>
  
  <footer>
    <p>&copy; 2025 My Company</p>
  </footer>
  
  <!-- Scripts at the end for performance -->
  <script src="/js/app.js"></script>
</body>
</html>
```

---

## Nesting Rules

HTML elements must be properly nested:

```html
<!-- ✅ Correct nesting -->
<p>This is <strong>important</strong> text.</p>

<!-- ❌ Incorrect nesting -->
<p>This is <strong>important</p></strong>
```

### Key Rules

1. **Close tags in reverse order** of opening
2. **Block elements** (like `<div>`, `<p>`) can contain inline elements
3. **Inline elements** (like `<span>`, `<a>`) should not contain block elements
4. **Some elements are self-closing**: `<img>`, `<br>`, `<meta>`, `<link>`

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always include `<!DOCTYPE html>` | Prevents quirks mode |
| Set `lang` on `<html>` | Accessibility and SEO |
| Declare `charset` first in `<head>` | Prevents encoding issues |
| Include viewport meta | Essential for mobile |
| Use meaningful `<title>` | SEO and user experience |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting doctype | Always start with `<!DOCTYPE html>` |
| Missing viewport meta | Add `<meta name="viewport" ...>` |
| Wrong charset position | Place `<meta charset>` first in head |
| Empty or generic title | Write descriptive, unique titles |
| Content outside `<body>` | Ensure all visible content is in body |

---

## Hands-on Exercise

### Your Task

Create a properly structured HTML document for an AI chatbot interface.

### Requirements

1. Include all essential elements (doctype, html, head, body)
2. Set the language to English
3. Include proper charset and viewport
4. Give it a meaningful title: "AI Assistant | My App"
5. Add a header, main, and footer section in the body

### Expected Result

A valid HTML5 document that passes the [W3C Validator](https://validator.w3.org/).

<details>
<summary>💡 Hints</summary>

- Start with `<!DOCTYPE html>`
- Remember `lang="en"` on the html element
- Put meta tags before the title
</details>

<details>
<summary>✅ Solution</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Assistant | My App</title>
</head>
<body>
  <header>
    <h1>AI Assistant</h1>
  </header>
  
  <main>
    <div id="chat-container">
      <!-- Chat messages will appear here -->
    </div>
  </main>
  
  <footer>
    <p>Powered by AI</p>
  </footer>
</body>
</html>
```
</details>

---

## Summary

✅ Every HTML document starts with `<!DOCTYPE html>` to trigger standards mode

✅ The `<html>` element wraps everything and should include a `lang` attribute

✅ The `<head>` contains metadata: charset, viewport, title, and links

✅ The `<body>` contains all visible content

✅ Proper nesting and structure are essential for valid HTML

---

**Previous:** [HTML Essentials Overview](./00-html-essentials.md)

**Next:** [Semantic Elements](./02-semantic-elements.md)
