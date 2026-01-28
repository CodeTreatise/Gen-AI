---
title: "Template-based DOM Creation"
---

# Template-based DOM Creation

## Introduction

Creating complex HTML structures with `createElement()` and `appendChild()` is verbose and hard to maintain. When you need to generate dozens of chat messages, product cards, or list items, there's a better way.

**Template-based approaches** let you define reusable HTML structures that can be cloned, populated with data, and efficiently inserted into the DOM. This lesson covers three powerful techniques: template literals, `DocumentFragment`, and the HTML `<template>` element.

### What We'll Cover
- Template literals for generating HTML strings
- `DocumentFragment` for batch DOM operations (performance)
- HTML `<template>` element for reusable, inert markup
- Cloning templates efficiently
- Sanitizing dynamic HTML to prevent XSS attacks

### Prerequisites
- [Creating & Modifying Elements](./02-creating-modifying-elements.md) (createElement, innerHTML)
- [JavaScript Core Concepts](../../03-javascript-core-concepts/00-javascript-core-concepts.md) (template literals, functions)
- [HTML Essentials](../../01-html-essentials/00-html-essentials.md) (element structure)

---

## Template Literals for HTML Strings

Template literals (backticks) make generating HTML much cleaner than string concatenation.

### Basic Template Literal

```javascript
const name = 'Alice';
const role = 'AI Assistant';

const html = `
  <div class="user-card">
    <h3>${name}</h3>
    <p>Role: ${role}</p>
  </div>
`;

console.log(html);
```

**Output:**
```html
<div class="user-card">
  <h3>Alice</h3>
  <p>Role: AI Assistant</p>
</div>
```

### Generating Multiple Elements

```javascript
const messages = [
  { user: 'Alice', text: 'Hello!' },
  { user: 'Bob', text: 'Hi there!' },
  { user: 'AI', text: 'How can I help?' }
];

const chatHTML = messages.map(msg => `
  <div class="message ${msg.user === 'AI' ? 'ai-message' : 'user-message'}">
    <strong>${msg.user}:</strong> ${msg.text}
  </div>
`).join('');

document.getElementById('chat').innerHTML = chatHTML;
```

**Output** (HTML in `#chat`):
```html
<div class="message user-message">
  <strong>Alice:</strong> Hello!
</div>
<div class="message user-message">
  <strong>Bob:</strong> Hi there!
</div>
<div class="message ai-message">
  <strong>AI:</strong> How can I help?
</div>
```

### insertAdjacentHTML for Adding Templates

Avoids replacing existing content (unlike `innerHTML`).

```javascript
function addMessage(user, text) {
  const messageHTML = `
    <div class="message">
      <strong>${user}:</strong> ${text}
    </div>
  `;
  
  const chatContainer = document.getElementById('chat');
  chatContainer.insertAdjacentHTML('beforeend', messageHTML);
}

addMessage('Alice', 'Hello!');
addMessage('AI', 'Hi Alice!');
```

**Positions:**
- `'beforebegin'` - Before the element
- `'afterbegin'` - Inside, before first child
- `'beforeend'` - Inside, after last child
- `'afterend'` - After the element

### ⚠️ Security Warning: XSS Risk

**Never** insert unsanitized user input into HTML.

```javascript
// ❌ DANGEROUS: XSS vulnerability
const userInput = '<img src=x onerror="alert(\'Hacked!\')">';
element.innerHTML = `<div>${userInput}</div>`; // Script executes!

// ✅ Safe: Use textContent for user data
const div = document.createElement('div');
div.textContent = userInput; // Shows literal text, no execution
container.appendChild(div);
```

**Rule:** Use `textContent` or sanitize with libraries like **DOMPurify** (covered below).

---

## DocumentFragment: Efficient Batch Operations

A `DocumentFragment` is a **lightweight container** for DOM nodes. It's not part of the active DOM tree, so modifications don't trigger reflows/repaints until you insert it.

### Why DocumentFragment?

**Problem:** Adding elements one by one causes **multiple reflows** (expensive).

```javascript
// ❌ Slow: 100 reflows (one per appendChild)
const container = document.getElementById('list');

for (let i = 0; i < 100; i++) {
  const item = document.createElement('li');
  item.textContent = `Item ${i}`;
  container.appendChild(item); // Reflow on each append
}
```

**Solution:** Use `DocumentFragment` to batch operations.

```javascript
// ✅ Fast: Only 1 reflow (when fragment is appended)
const container = document.getElementById('list');
const fragment = document.createDocumentFragment();

for (let i = 0; i < 100; i++) {
  const item = document.createElement('li');
  item.textContent = `Item ${i}`;
  fragment.appendChild(item); // No reflow (fragment is not in DOM)
}

container.appendChild(fragment); // Single reflow
```

**Result:** **Much faster** for large numbers of elements.

### Creating a DocumentFragment

```javascript
const fragment = document.createDocumentFragment();

// Add elements to fragment
const heading = document.createElement('h2');
heading.textContent = 'User Profile';

const paragraph = document.createElement('p');
paragraph.textContent = 'Welcome back!';

fragment.appendChild(heading);
fragment.appendChild(paragraph);

// Insert fragment into DOM (one operation)
document.getElementById('container').appendChild(fragment);
```

**Output** (HTML):
```html
<div id="container">
  <h2>User Profile</h2>
  <p>Welcome back!</p>
</div>
```

> **Note:** After appending, the fragment becomes empty—its children move to the container.

### Use Case: Rendering a List

```javascript
function renderMessages(messages) {
  const container = document.getElementById('message-list');
  const fragment = document.createDocumentFragment();
  
  messages.forEach(msg => {
    const div = document.createElement('div');
    div.className = 'message';
    div.textContent = msg.text;
    fragment.appendChild(div);
  });
  
  // Clear old messages and add new ones (single reflow)
  container.replaceChildren(fragment);
}

renderMessages([
  { text: 'Message 1' },
  { text: 'Message 2' },
  { text: 'Message 3' }
]);
```

---

## HTML `<template>` Element: Reusable Markup

The `<template>` element holds **inert HTML** that isn't rendered until you clone and activate it. Perfect for repeating structures like cards, table rows, or list items.

### Defining a Template

```html
<template id="message-template">
  <div class="message">
    <img class="avatar" src="" alt="Avatar">
    <div class="content">
      <strong class="username"></strong>
      <p class="text"></p>
    </div>
  </div>
</template>

<div id="chat-container"></div>
```

### Cloning and Using the Template

```javascript
function addMessageFromTemplate(username, text, avatarURL) {
  // Get the template
  const template = document.getElementById('message-template');
  
  // Clone the template content (deep copy)
  const clone = template.content.cloneNode(true);
  
  // Populate the clone with data
  clone.querySelector('.username').textContent = username;
  clone.querySelector('.text').textContent = text;
  clone.querySelector('.avatar').src = avatarURL;
  
  // Append to container
  document.getElementById('chat-container').appendChild(clone);
}

addMessageFromTemplate('Alice', 'Hello!', 'avatar1.png');
addMessageFromTemplate('AI', 'Hi Alice!', 'ai-avatar.png');
```

**Output** (HTML in `#chat-container`):
```html
<div class="message">
  <img class="avatar" src="avatar1.png" alt="Avatar">
  <div class="content">
    <strong class="username">Alice</strong>
    <p class="text">Hello!</p>
  </div>
</div>
<div class="message">
  <img class="avatar" src="ai-avatar.png" alt="Avatar">
  <div class="content">
    <strong class="username">AI</strong>
    <p class="text">Hi Alice!</p>
  </div>
</div>
```

### Why Use `<template>`?

| Benefit | Explanation |
|---------|-------------|
| **Inert** | Content inside `<template>` isn't rendered, scripts don't run, images don't load |
| **Reusable** | Define once, clone many times |
| **Clean HTML** | Markup stays in HTML, not buried in JavaScript strings |
| **Fast** | Browser parses template once, cloning is very fast |

### Template with DocumentFragment

The `template.content` is a **DocumentFragment**, so you get the performance benefits automatically.

```javascript
const template = document.getElementById('message-template');

// template.content is a DocumentFragment
console.log(template.content); // DocumentFragment

// Clone it
const clone = template.content.cloneNode(true);

// Modify and append (just like DocumentFragment)
clone.querySelector('.text').textContent = 'New message';
container.appendChild(clone);
```

---

## Declarative Shadow DOM with Templates

The `<template>` element supports **Declarative Shadow DOM** (shadowroot mode). This creates encapsulated components with scoped styles.

```html
<div id="user-card">
  <template shadowrootmode="open">
    <style>
      .card {
        border: 2px solid #333;
        padding: 20px;
        border-radius: 8px;
      }
      h3 {
        color: #007bff;
      }
    </style>
    <div class="card">
      <h3>Alice</h3>
      <p>AI Assistant</p>
    </div>
  </template>
</div>
```

**Result:** The styles are **scoped** to this component and won't affect other page elements.

> **Note:** Declarative Shadow DOM is a modern feature. Check [browser support](https://caniuse.com/declarative-shadow-dom).

---

## Sanitizing Dynamic HTML

When using `innerHTML` or `insertAdjacentHTML` with user-generated content, **sanitize** to prevent XSS attacks.

### DOMPurify (Recommended Library)

```javascript
// Install: npm install dompurify
// Or use CDN: <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>

import DOMPurify from 'dompurify';

const userInput = '<img src=x onerror="alert(\'XSS\')">';

// ✅ Sanitize before inserting
const clean = DOMPurify.sanitize(userInput);

element.innerHTML = clean; // Safe: "<img src=\"x\">"
```

**What DOMPurify does:**
- Removes dangerous attributes (`onerror`, `onclick`, etc.)
- Removes dangerous tags (`<script>`, `<object>`, `<embed>`)
- Allows safe HTML (`<p>`, `<strong>`, `<a>`, etc.)

### Trusted Types (Modern Browser API)

Trusted Types enforce that only sanitized HTML can be used with `innerHTML`.

```javascript
// Create a policy
const policy = trustedTypes.createPolicy('default', {
  createHTML: (input) => {
    // Sanitize input (use DOMPurify or custom logic)
    return DOMPurify.sanitize(input);
  }
});

// Use the policy
const userInput = '<img src=x onerror="alert(1)">';
element.innerHTML = policy.createHTML(userInput); // Sanitized
```

**Browser support:** Modern browsers (check [caniuse.com](https://caniuse.com/trusted-types)).

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| **Use `<template>` for repeated structures** | Cleaner, faster, more maintainable than string building |
| **Use `DocumentFragment` for batch operations** | Single reflow vs multiple (much faster) |
| **Sanitize user input with DOMPurify** | Prevents XSS attacks when using `innerHTML` |
| **Use `textContent` for plain text** | Safer than `innerHTML` (no HTML parsing) |
| **Clone templates with `cloneNode(true)`** | Deep copy includes all children |
| **Cache template references** | Don't query for template on every use |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `innerHTML` with unsanitized user input | Sanitize with DOMPurify or use `textContent` |
| Adding elements one by one in a loop | Use `DocumentFragment` to batch operations |
| Forgetting `cloneNode(true)` (shallow copy) | Use `cloneNode(true)` for deep copy (all children) |
| Not caching template reference | Query template once, reuse reference |
| Using template literals for user input | Escape user data or use `textContent` |
| Modifying template content directly | Clone first, modify clone, then append |

---

## Hands-on Exercise

### Your Task

Build a **dynamic product card generator** using templates:
- Define a `<template>` for product cards (image, name, price, "Add to Cart" button)
- Generate 6 product cards from data
- Use `DocumentFragment` for efficient insertion
- Handle "Add to Cart" clicks with event delegation
- Sanitize product descriptions with DOMPurify (simulated)

### Requirements

1. Create HTML `<template id="product-template">` with:
   - `<img class="product-image">`
   - `<h3 class="product-name"></h3>`
   - `<p class="product-description"></p>`
   - `<span class="product-price"></span>`
   - `<button class="add-to-cart">Add to Cart</button>`

2. Product data array:
```javascript
const products = [
  { id: 1, name: 'AI Assistant', price: 29.99, description: 'Your personal AI helper', image: 'ai.png' },
  { id: 2, name: 'Code Generator', price: 49.99, description: 'Generate code with AI', image: 'code.png' },
  // ... 4 more products
];
```

3. **Render products:**
   - Clone template for each product
   - Populate with product data
   - Add `data-product-id` attribute
   - Use `DocumentFragment` to batch append

4. **Event delegation:**
   - Attach one click listener to product container
   - Check if clicked element is "Add to Cart" button
   - Log: "Added product [ID] to cart"

5. **Sanitize descriptions** (simulate with a custom function):
```javascript
function sanitize(html) {
  return html.replace(/<script.*?>.*?<\/script>/gi, '');
}
```

### Expected Result

- 6 product cards displayed
- Clicking "Add to Cart" logs the product ID
- Descriptions are sanitized (no script tags)

<details>
<summary>💡 Hints (click to expand)</summary>

- Clone with `template.content.cloneNode(true)`
- Set data attribute with `element.dataset.productId = product.id`
- Use `fragment.appendChild(clone)` to build fragment
- Insert fragment with `container.appendChild(fragment)`
- Use event delegation: `container.addEventListener('click', ...)`
- Check button with `e.target.classList.contains('add-to-cart')`
- Get product ID from `e.target.closest('[data-product-id]').dataset.productId`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**HTML:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Product Card Generator</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 1200px;
      margin: 50px auto;
      padding: 20px;
    }
    #product-container {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 20px;
    }
    .product-card {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      transition: box-shadow 0.2s;
    }
    .product-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .product-image {
      width: 100%;
      height: 150px;
      object-fit: cover;
      border-radius: 8px;
      background-color: #f0f0f0;
    }
    .product-name {
      margin: 15px 0 10px;
      color: #333;
    }
    .product-description {
      font-size: 14px;
      color: #666;
      margin-bottom: 15px;
    }
    .product-price {
      font-size: 20px;
      font-weight: bold;
      color: #007bff;
      display: block;
      margin-bottom: 15px;
    }
    .add-to-cart {
      padding: 10px 20px;
      background-color: #28a745;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    .add-to-cart:hover {
      background-color: #218838;
    }
  </style>
</head>
<body>
  <h1>AI Product Catalog</h1>
  
  <!-- Template definition -->
  <template id="product-template">
    <div class="product-card">
      <img class="product-image" src="" alt="">
      <h3 class="product-name"></h3>
      <p class="product-description"></p>
      <span class="product-price"></span>
      <button class="add-to-cart">Add to Cart</button>
    </div>
  </template>
  
  <!-- Product container -->
  <div id="product-container"></div>

  <script src="script.js"></script>
</body>
</html>
```

**JavaScript (script.js):**
```javascript
// Product data
const products = [
  { id: 1, name: 'AI Assistant', price: 29.99, description: 'Your personal AI helper', image: 'https://via.placeholder.com/250x150?text=AI+Assistant' },
  { id: 2, name: 'Code Generator', price: 49.99, description: 'Generate code with AI', image: 'https://via.placeholder.com/250x150?text=Code+Gen' },
  { id: 3, name: 'Image Creator', price: 39.99, description: 'AI-powered image generation', image: 'https://via.placeholder.com/250x150?text=Image+AI' },
  { id: 4, name: 'Voice Assistant', price: 59.99, description: 'Natural voice interactions', image: 'https://via.placeholder.com/250x150?text=Voice+AI' },
  { id: 5, name: 'Data Analyzer', price: 79.99, description: 'Analyze data with AI', image: 'https://via.placeholder.com/250x150?text=Data+AI' },
  { id: 6, name: 'Translation Bot', price: 34.99, description: 'Multilingual AI translator', image: 'https://via.placeholder.com/250x150?text=Translator' }
];

// Sanitize function (simulated)
function sanitize(html) {
  // Remove script tags
  return html.replace(/<script.*?>.*?<\/script>/gi, '');
}

// Render products
function renderProducts(products) {
  const container = document.getElementById('product-container');
  const template = document.getElementById('product-template');
  const fragment = document.createDocumentFragment();
  
  products.forEach(product => {
    // Clone template
    const clone = template.content.cloneNode(true);
    
    // Get elements
    const card = clone.querySelector('.product-card');
    const image = clone.querySelector('.product-image');
    const name = clone.querySelector('.product-name');
    const description = clone.querySelector('.product-description');
    const price = clone.querySelector('.product-price');
    
    // Populate with data
    card.dataset.productId = product.id;
    image.src = product.image;
    image.alt = product.name;
    name.textContent = product.name;
    description.textContent = sanitize(product.description); // Sanitize
    price.textContent = `$${product.price.toFixed(2)}`;
    
    // Add to fragment
    fragment.appendChild(clone);
  });
  
  // Single DOM insertion
  container.appendChild(fragment);
  
  console.log(`Rendered ${products.length} products`);
}

// Event delegation for "Add to Cart" buttons
document.getElementById('product-container').addEventListener('click', (event) => {
  if (event.target.classList.contains('add-to-cart')) {
    const productCard = event.target.closest('.product-card');
    const productId = productCard.dataset.productId;
    
    console.log(`Added product ${productId} to cart`);
    
    // Visual feedback
    event.target.textContent = 'Added!';
    event.target.style.backgroundColor = '#6c757d';
    
    setTimeout(() => {
      event.target.textContent = 'Add to Cart';
      event.target.style.backgroundColor = '#28a745';
    }, 1000);
  }
});

// Render products on page load
renderProducts(products);
```

**Output** (in console):
```
Rendered 6 products
Added product 2 to cart
Added product 5 to cart
```

</details>

### Bonus Challenges

- [ ] Add a "Remove" button to products that hides them
- [ ] Add a shopping cart counter that updates when products are added
- [ ] Filter products by price range (add slider input)
- [ ] Sort products by name or price (add dropdown)
- [ ] Use real DOMPurify library for sanitization

---

## Summary

✅ **Template literals** make HTML generation cleaner and more readable  
✅ **DocumentFragment** batches DOM operations for better performance  
✅ **`<template>` element** provides reusable, inert markup that's fast to clone  
✅ **Always sanitize user input** with DOMPurify when using `innerHTML`  
✅ **Clone templates with `cloneNode(true)`** for deep copies  
✅ **Combine templates with fragments** for efficient rendering of many elements

**Next:** [Asynchronous JavaScript](../05-asynchronous-javascript.md)

---

## Further Reading

- [MDN: HTML `<template>` element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/template) - Template reference
- [MDN: DocumentFragment](https://developer.mozilla.org/en-US/docs/Web/API/DocumentFragment) - Fragment API
- [DOMPurify](https://github.com/cure53/DOMPurify) - HTML sanitization library
- [Asynchronous JavaScript](../05-asynchronous-javascript.md) - Next lesson

<!-- 
Sources Consulted:
- MDN HTML template element: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/template
- MDN DocumentFragment: https://developer.mozilla.org/en-US/docs/Web/API/DocumentFragment
- MDN insertAdjacentHTML: https://developer.mozilla.org/en-US/docs/Web/API/Element/insertAdjacentHTML
- DOMPurify GitHub: https://github.com/cure53/DOMPurify
-->
