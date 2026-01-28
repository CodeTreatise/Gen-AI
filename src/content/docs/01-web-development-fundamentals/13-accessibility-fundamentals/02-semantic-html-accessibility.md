---
title: "Semantic HTML for Accessibility"
---

# Semantic HTML for Accessibility

## Introduction

Semantic HTML is the foundation of accessible web development. Using the right HTML elements provides meaning to assistive technologies without any extra work. Semantic HTML is accessible by default—you just need to use it correctly.

This lesson covers semantic elements that communicate structure and purpose to all users.

### What We'll Cover

- Heading hierarchy
- Landmark roles
- Lists and tables
- Form labels and fieldsets
- Alternative text for images

### Prerequisites

- HTML fundamentals
- Understanding of document structure

---

## Heading Hierarchy

Headings create a navigable document outline. Screen reader users often navigate by headings.

### Proper Heading Structure

```html
<!-- ✅ Correct hierarchy -->
<h1>Web Accessibility Guide</h1>
  <h2>Introduction</h2>
  <h2>WCAG Guidelines</h2>
    <h3>Level A</h3>
    <h3>Level AA</h3>
    <h3>Level AAA</h3>
  <h2>Testing</h2>

<!-- ❌ Wrong: Skipping levels -->
<h1>Title</h1>
<h4>Subsection</h4>  <!-- Skipped h2 and h3! -->

<!-- ❌ Wrong: Using headings for styling -->
<h3>Regular paragraph styled as heading</h3>
```

### Heading Rules

| Rule | Reason |
|------|--------|
| Only one `<h1>` per page | Identifies main topic |
| Don't skip levels | Creates logical outline |
| Use for structure, not styling | CSS handles visual size |
| Keep headings concise | Easy to scan and navigate |

### Visual vs Semantic

```css
/* Style any heading level any way you want */
.section-title {
  font-size: 1.25rem;  /* Visual size */
  font-weight: 600;
}
```

```html
<!-- Semantic level is h2, but styled smaller -->
<h2 class="section-title">Section Title</h2>
```

---

## Landmark Roles

Landmarks help users jump to major sections of a page.

### HTML5 Landmark Elements

| Element | ARIA Role | Purpose |
|---------|-----------|---------|
| `<header>` | banner | Site header (when child of body) |
| `<nav>` | navigation | Navigation links |
| `<main>` | main | Main content (one per page) |
| `<aside>` | complementary | Sidebar, related content |
| `<footer>` | contentinfo | Site footer (when child of body) |
| `<section>` | region | Thematic grouping (with heading) |
| `<article>` | article | Self-contained content |
| `<form>` | form | When labeled |

### Page Structure Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Accessible Page</title>
</head>
<body>
  <header>
    <a href="/">Logo</a>
    <nav aria-label="Main">
      <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/about">About</a></li>
      </ul>
    </nav>
  </header>

  <main>
    <h1>Page Title</h1>
    
    <article>
      <h2>Article Title</h2>
      <p>Content...</p>
    </article>
  </main>

  <aside aria-label="Related">
    <h2>Related Articles</h2>
    <ul>...</ul>
  </aside>

  <footer>
    <nav aria-label="Footer">
      <ul>...</ul>
    </nav>
    <p>© 2025 Company</p>
  </footer>
</body>
</html>
```

### Labeling Landmarks

When you have multiple landmarks of the same type:

```html
<!-- Multiple navs need labels -->
<nav aria-label="Main">...</nav>
<nav aria-label="Footer">...</nav>

<!-- Or use aria-labelledby -->
<nav aria-labelledby="nav-heading">
  <h2 id="nav-heading">Resources</h2>
  <ul>...</ul>
</nav>
```

---

## Lists

Lists convey structure. Screen readers announce "list with 5 items."

### When to Use Lists

```html
<!-- ✅ Navigation is a list of links -->
<nav>
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/products">Products</a></li>
    <li><a href="/contact">Contact</a></li>
  </ul>
</nav>

<!-- ✅ Steps are an ordered list -->
<ol>
  <li>Enter your email</li>
  <li>Choose a password</li>
  <li>Verify your account</li>
</ol>

<!-- ✅ Features are an unordered list -->
<ul>
  <li>Free shipping</li>
  <li>30-day returns</li>
  <li>24/7 support</li>
</ul>

<!-- ✅ Term/definition pairs -->
<dl>
  <dt>HTML</dt>
  <dd>HyperText Markup Language</dd>
  <dt>CSS</dt>
  <dd>Cascading Style Sheets</dd>
</dl>
```

### Styling Lists

Remove bullets visually while keeping semantics:

```css
.clean-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

/* Restore semantics in Safari/VoiceOver */
.clean-list li::before {
  content: "\200B"; /* Zero-width space */
}
```

---

## Tables

Tables are for tabular data, not layout.

### Accessible Tables

```html
<table>
  <caption>Q4 2025 Sales by Region</caption>
  <thead>
    <tr>
      <th scope="col">Region</th>
      <th scope="col">Q1</th>
      <th scope="col">Q2</th>
      <th scope="col">Q3</th>
      <th scope="col">Q4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">North</th>
      <td>$10,000</td>
      <td>$12,000</td>
      <td>$15,000</td>
      <td>$18,000</td>
    </tr>
    <tr>
      <th scope="row">South</th>
      <td>$8,000</td>
      <td>$9,000</td>
      <td>$11,000</td>
      <td>$14,000</td>
    </tr>
  </tbody>
</table>
```

### Table Elements

| Element | Purpose |
|---------|---------|
| `<caption>` | Describes the table |
| `<thead>` | Header rows |
| `<tbody>` | Body rows |
| `<th scope="col">` | Column header |
| `<th scope="row">` | Row header |
| `<td>` | Data cell |

### Complex Tables

For tables with merged cells, use `id` and `headers`:

```html
<table>
  <tr>
    <th id="name">Name</th>
    <th id="q1" colspan="2">Q1</th>
  </tr>
  <tr>
    <th id="name2" headers="name"></th>
    <th id="jan" headers="q1">Jan</th>
    <th id="feb" headers="q1">Feb</th>
  </tr>
  <tr>
    <td headers="name">Alice</td>
    <td headers="q1 jan">$100</td>
    <td headers="q1 feb">$150</td>
  </tr>
</table>
```

---

## Form Labels

Every form input needs a label.

### Label Association

```html
<!-- ✅ Explicit label (recommended) -->
<label for="email">Email address</label>
<input type="email" id="email" name="email">

<!-- ✅ Implicit label (wrapping) -->
<label>
  Email address
  <input type="email" name="email">
</label>

<!-- ❌ No label - inaccessible! -->
<input type="email" placeholder="Email">
```

### Placeholder Is Not a Label

```html
<!-- ❌ Wrong: Placeholder disappears when typing -->
<input type="text" placeholder="Full name">

<!-- ✅ Correct: Label is always visible -->
<label for="name">Full name</label>
<input type="text" id="name" placeholder="e.g., John Smith">
```

### Required Fields

```html
<label for="email">
  Email address
  <span aria-hidden="true">*</span>
  <span class="sr-only">(required)</span>
</label>
<input type="email" id="email" required aria-required="true">

<!-- Screen reader hears: "Email address (required)" -->
```

### Help Text

```html
<label for="password">Password</label>
<input 
  type="password" 
  id="password"
  aria-describedby="password-help">
<p id="password-help">
  Must be at least 8 characters with one number.
</p>
```

### Fieldsets and Legends

Group related inputs:

```html
<fieldset>
  <legend>Shipping Address</legend>
  
  <label for="street">Street</label>
  <input type="text" id="street">
  
  <label for="city">City</label>
  <input type="text" id="city">
</fieldset>

<!-- Essential for radio groups -->
<fieldset>
  <legend>Payment Method</legend>
  
  <label>
    <input type="radio" name="payment" value="card">
    Credit Card
  </label>
  
  <label>
    <input type="radio" name="payment" value="paypal">
    PayPal
  </label>
</fieldset>
```

---

## Alternative Text

Alt text provides text equivalents for images.

### Writing Good Alt Text

```html
<!-- ✅ Descriptive: Conveys meaning -->
<img src="chart.png" alt="Bar chart: Sales grew 40% in Q4 2025">

<!-- ✅ Functional: Describes purpose -->
<img src="search.png" alt="Search">

<!-- ✅ Decorative: Empty alt -->
<img src="decorative-line.png" alt="">

<!-- ❌ Bad: Says "image" -->
<img src="photo.jpg" alt="Image of a sunset">

<!-- ❌ Bad: Filename -->
<img src="photo.jpg" alt="DSC_0042.jpg">

<!-- ❌ Bad: Too long -->
<img src="team.jpg" alt="This is a photograph of our team 
including John who is the CEO and Sarah who handles...">
```

### Alt Text Decision Tree

```
Is the image purely decorative?
├── Yes → alt=""
└── No → Does the image contain text?
    ├── Yes → Include the text in alt
    └── No → Is it functional (link/button)?
        ├── Yes → Describe the action
        └── No → Describe the content
```

### Complex Images

For charts, diagrams, infographics:

```html
<!-- Short alt + long description -->
<figure>
  <img 
    src="complex-chart.png" 
    alt="Q4 sales by region (details below)">
  <figcaption>
    <details>
      <summary>Chart data</summary>
      <p>North region: $18,000 (40% growth)</p>
      <p>South region: $14,000 (30% growth)</p>
      <!-- Full data... -->
    </details>
  </figcaption>
</figure>
```

### Background Images

CSS background images are invisible to screen readers:

```html
<!-- If meaningful, provide alternative -->
<div 
  class="hero-section" 
  role="img" 
  aria-label="Happy customers using our product">
</div>

<!-- Or use hidden text -->
<div class="hero-section">
  <span class="sr-only">Happy customers using our product</span>
</div>
```

---

## Screen Reader-Only Text

Hide visually but expose to assistive tech:

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

```html
<a href="/cart">
  <svg>...</svg>
  <span class="sr-only">Shopping cart - 3 items</span>
</a>
```

---

## Hands-on Exercise

### Your Task

Make this form accessible:

```html
<div class="form">
  <input type="text" placeholder="Name">
  <input type="email" placeholder="Email">
  <div class="radios">
    <input type="radio" name="contact" value="email"> Email
    <input type="radio" name="contact" value="phone"> Phone
  </div>
  <div class="button" onclick="submit()">Submit</div>
</div>
```

<details>
<summary>✅ Solution</summary>

```html
<form>
  <div>
    <label for="name">Name</label>
    <input type="text" id="name" required>
  </div>
  
  <div>
    <label for="email">Email</label>
    <input type="email" id="email" required>
  </div>
  
  <fieldset>
    <legend>Preferred contact method</legend>
    <label>
      <input type="radio" name="contact" value="email">
      Email
    </label>
    <label>
      <input type="radio" name="contact" value="phone">
      Phone
    </label>
  </fieldset>
  
  <button type="submit">Submit</button>
</form>
```
</details>

---

## Summary

✅ Use **one `<h1>`** per page with proper hierarchy
✅ **Landmark elements** (`<nav>`, `<main>`, `<aside>`) enable quick navigation
✅ **Lists** (`<ul>`, `<ol>`) convey structure to screen readers
✅ **Tables** need `<caption>`, `<th>`, and `scope` attributes
✅ **Labels** must be explicitly associated with form inputs
✅ **Alt text** should be descriptive, functional, or empty (decorative)

**Next:** [ARIA Essentials](./03-aria-essentials.md)

---

## Further Reading

- [MDN HTML Elements Reference](https://developer.mozilla.org/en-US/docs/Web/HTML/Element)
- [WebAIM Forms](https://webaim.org/techniques/forms/)
- [Alt Text Decision Tree](https://www.w3.org/WAI/tutorials/images/decision-tree/)

<!-- 
Sources Consulted:
- MDN Accessibility: https://developer.mozilla.org/en-US/docs/Web/Accessibility
- WebAIM: https://webaim.org/
-->
