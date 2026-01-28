---
title: "Semantic Elements"
---

# Semantic Elements

## Introduction

Semantic HTML uses elements that convey meaning about the content they contain. Instead of generic `<div>` containers everywhere, semantic elements tell browsers, screen readers, and search engines what role each section plays.

This matters for AI integration: when building interfaces for chatbots, content generators, or accessible AI tools, semantic markup ensures your content is understandable by both humans and machines.

### What We'll Cover

- What semantic HTML means and why it matters
- Structural elements: `<header>`, `<nav>`, `<main>`, `<article>`, `<section>`, `<aside>`, `<footer>`
- Content elements: `<figure>`, `<figcaption>`, `<time>`, `<address>`
- Choosing the right element for each situation

### Prerequisites

- Understanding of HTML document structure
- Familiarity with basic HTML tags

---

## Why Semantic HTML Matters

### The Problem with `<div>` Soup

Consider this non-semantic markup:

```html
<div class="header">
  <div class="nav">
    <div class="nav-item">Home</div>
  </div>
</div>
<div class="content">
  <div class="article">
    <div class="title">Article Title</div>
  </div>
</div>
```

A browser sees only nested boxes. Screen readers can't identify navigation. Search engines struggle to find main content.

### The Semantic Solution

```html
<header>
  <nav>
    <a href="/">Home</a>
  </nav>
</header>
<main>
  <article>
    <h1>Article Title</h1>
  </article>
</main>
```

Now every element communicates its purpose.

### Benefits of Semantic HTML

| Benefit | Description |
|---------|-------------|
| **Accessibility** | Screen readers announce sections correctly |
| **SEO** | Search engines understand content hierarchy |
| **Maintainability** | Code is self-documenting |
| **AI parsing** | Content scrapers identify article text |
| **Styling** | Target elements by meaning, not class |

---

## Structural Semantic Elements

### `<header>`

Contains introductory content or navigation:

```html
<!-- Page header -->
<header>
  <h1>Site Name</h1>
  <nav>...</nav>
</header>

<!-- Article header -->
<article>
  <header>
    <h2>Article Title</h2>
    <p>Published on <time datetime="2025-01-22">January 22, 2025</time></p>
  </header>
</article>
```

> **Note:** A page can have multiple `<header>` elements—one for the page, one per article/section.

---

### `<nav>`

Contains major navigation links:

```html
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/products">Products</a></li>
    <li><a href="/about">About</a></li>
    <li><a href="/contact">Contact</a></li>
  </ul>
</nav>
```

**Use `<nav>` for:**
- Primary site navigation
- Table of contents
- Pagination

**Don't use `<nav>` for:**
- Every group of links (footer links are often fine without it)
- Single links

---

### `<main>`

Contains the primary content of the page:

```html
<body>
  <header>...</header>
  
  <main>
    <!-- The main content lives here -->
    <h1>Page Title</h1>
    <p>Primary content...</p>
  </main>
  
  <footer>...</footer>
</body>
```

**Rules:**
- Only ONE `<main>` per page
- Should not be inside `<article>`, `<aside>`, `<header>`, `<footer>`, or `<nav>`
- Represents content unique to this page (not repeated across pages)

---

### `<article>`

Contains self-contained, independently distributable content:

```html
<article>
  <header>
    <h2>Understanding AI Embeddings</h2>
    <p>By Jane Developer | <time datetime="2025-01-22">Jan 22, 2025</time></p>
  </header>
  
  <p>Embeddings are numerical representations of text...</p>
  
  <footer>
    <p>Tags: AI, Machine Learning, NLP</p>
  </footer>
</article>
```

**Use `<article>` for:**
- Blog posts
- News articles
- Forum posts
- Product cards
- Comments
- Interactive widgets

**Test:** Could this content make sense on its own in an RSS feed? Then use `<article>`.

---

### `<section>`

Groups related content with a heading:

```html
<section>
  <h2>Features</h2>
  <p>Our product offers...</p>
</section>

<section>
  <h2>Pricing</h2>
  <p>Choose a plan...</p>
</section>
```

**Rules:**
- Should typically have a heading
- Groups thematically related content
- Use when `<article>`, `<aside>`, or `<nav>` don't apply

---

### `<aside>`

Contains tangentially related content:

```html
<main>
  <article>
    <h1>Main Article</h1>
    <p>Primary content...</p>
    
    <aside>
      <h3>Related Terms</h3>
      <p>Definition of key terms...</p>
    </aside>
  </article>
</main>

<aside>
  <h2>Sidebar</h2>
  <nav>Related articles...</nav>
</aside>
```

**Use `<aside>` for:**
- Sidebars
- Pull quotes
- Advertising
- Related links
- Glossary terms

---

### `<footer>`

Contains closing content:

```html
<!-- Page footer -->
<footer>
  <p>&copy; 2025 My Company</p>
  <nav>
    <a href="/privacy">Privacy</a>
    <a href="/terms">Terms</a>
  </nav>
</footer>

<!-- Article footer -->
<article>
  <h2>Article Title</h2>
  <p>Content...</p>
  <footer>
    <p>Written by Jane Doe</p>
  </footer>
</article>
```

---

## Content Semantic Elements

### `<figure>` and `<figcaption>`

For self-contained media with captions:

```html
<figure>
  <img src="chart.png" alt="Sales growth chart showing 50% increase">
  <figcaption>Figure 1: Quarterly sales growth in 2024</figcaption>
</figure>
```

Works with images, diagrams, code snippets, videos, and more.

---

### `<time>`

Represents dates and times in machine-readable format:

```html
<p>Published on <time datetime="2025-01-22">January 22, 2025</time></p>

<p>Event starts at <time datetime="14:30">2:30 PM</time></p>

<p>Duration: <time datetime="PT2H30M">2 hours 30 minutes</time></p>
```

The `datetime` attribute provides the machine-readable version.

---

### `<address>`

Contact information for the author or owner:

```html
<address>
  <p>Contact the author:</p>
  <a href="mailto:jane@example.com">jane@example.com</a>
</address>
```

---

## Complete Page Structure Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Blog | Semantic HTML</title>
</head>
<body>
  <header>
    <h1>AI Developer Blog</h1>
    <nav aria-label="Main navigation">
      <a href="/">Home</a>
      <a href="/articles">Articles</a>
      <a href="/about">About</a>
    </nav>
  </header>
  
  <main>
    <article>
      <header>
        <h2>Getting Started with LLMs</h2>
        <p>By <address style="display:inline">Jane Doe</address> on 
           <time datetime="2025-01-22">January 22, 2025</time></p>
      </header>
      
      <section>
        <h3>What Are LLMs?</h3>
        <p>Large Language Models are...</p>
      </section>
      
      <section>
        <h3>How to Use Them</h3>
        <p>To integrate an LLM...</p>
        
        <figure>
          <img src="architecture.png" alt="LLM integration architecture diagram">
          <figcaption>Figure 1: Typical LLM integration architecture</figcaption>
        </figure>
      </section>
      
      <footer>
        <p>Tags: AI, LLM, Tutorial</p>
      </footer>
    </article>
    
    <aside>
      <h2>Related Articles</h2>
      <ul>
        <li><a href="/prompt-engineering">Prompt Engineering Guide</a></li>
        <li><a href="/embeddings">Understanding Embeddings</a></li>
      </ul>
    </aside>
  </main>
  
  <footer>
    <p>&copy; 2025 AI Developer Blog</p>
    <nav aria-label="Footer navigation">
      <a href="/privacy">Privacy</a>
      <a href="/terms">Terms</a>
    </nav>
  </footer>
</body>
</html>
```

---

## Choosing the Right Element

| Content Type | Element |
|--------------|---------|
| Site logo and primary nav | `<header>` |
| Main navigation links | `<nav>` |
| Primary page content | `<main>` |
| Blog post, comment, card | `<article>` |
| Thematic grouping with heading | `<section>` |
| Sidebar, related content | `<aside>` |
| Copyright, contact info | `<footer>` |
| Image with caption | `<figure>` + `<figcaption>` |
| Dates and times | `<time>` |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use one `<main>` per page | Landmarks must be unique |
| Add headings to sections | Assistive tech uses them for navigation |
| Use `aria-label` on multiple navs | Distinguishes them for screen readers |
| Don't overuse `<section>` | Only when content is thematically related |
| Prefer semantic over `<div>` | Self-documenting, accessible code |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `<section>` without heading | Add a heading or use `<div>` |
| Multiple `<main>` elements | Only one per page |
| `<article>` for non-standalone content | Use `<section>` instead |
| Forgetting `<nav>` aria-label | Add labels when multiple navs exist |
| Wrapping everything in `<section>` | Use `<div>` for pure styling containers |

---

## Hands-on Exercise

### Your Task

Convert this non-semantic HTML into proper semantic markup:

```html
<div class="header">
  <div class="logo">My Blog</div>
  <div class="nav">
    <div class="nav-item"><a href="/">Home</a></div>
    <div class="nav-item"><a href="/about">About</a></div>
  </div>
</div>
<div class="content">
  <div class="post">
    <div class="title">My First Post</div>
    <div class="date">January 22, 2025</div>
    <div class="body">This is my post content...</div>
  </div>
</div>
<div class="sidebar">
  <div class="widget">Related posts...</div>
</div>
<div class="footer">Copyright 2025</div>
```

### Expected Result

Clean semantic HTML using `<header>`, `<nav>`, `<main>`, `<article>`, `<aside>`, `<footer>`, and `<time>`.

<details>
<summary>✅ Solution</summary>

```html
<header>
  <h1>My Blog</h1>
  <nav>
    <a href="/">Home</a>
    <a href="/about">About</a>
  </nav>
</header>

<main>
  <article>
    <header>
      <h2>My First Post</h2>
      <time datetime="2025-01-22">January 22, 2025</time>
    </header>
    <p>This is my post content...</p>
  </article>
</main>

<aside>
  <h2>Related Posts</h2>
  <p>Related posts...</p>
</aside>

<footer>
  <p>&copy; 2025</p>
</footer>
```
</details>

---

## Summary

✅ Semantic elements communicate meaning, not just structure

✅ Use `<header>`, `<nav>`, `<main>`, `<article>`, `<section>`, `<aside>`, `<footer>` for page structure

✅ Only one `<main>` per page

✅ `<article>` is for self-contained, distributable content

✅ `<section>` groups related content and should have a heading

✅ Semantic HTML improves accessibility, SEO, and maintainability

---

**Previous:** [Document Structure & Doctype](./01-document-structure.md)

**Next:** [Forms & Input Types](./03-forms-input-types.md)
