---
title: "Forms & Input Types"
---

# Forms & Input Types

## Introduction

Forms are how users send data to your application. For AI-powered apps, forms are critical—they collect the prompts, uploads, and parameters that feed into your AI systems. Understanding HTML forms thoroughly lets you build better interfaces for chatbots, image analyzers, and other AI features.

### What We'll Cover

- Form structure: `<form>`, `<label>`, `<input>`
- All modern input types (text, email, number, date, file, and more)
- Form attributes: action, method, validation
- Grouping with `<fieldset>` and `<legend>`
- Select dropdowns, textareas, and buttons
- Client-side validation attributes

### Prerequisites

- HTML document structure
- Basic understanding of semantic elements

---

## Basic Form Structure

Every form starts with the `<form>` element:

```html
<form action="/submit" method="POST">
  <label for="name">Your Name:</label>
  <input type="text" id="name" name="name" required>
  
  <button type="submit">Send</button>
</form>
```

### Key Attributes

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `action` | Where to send form data | `/api/submit` |
| `method` | HTTP method | `GET` or `POST` |
| `enctype` | Encoding for file uploads | `multipart/form-data` |
| `novalidate` | Skip browser validation | `novalidate` |

### GET vs POST

| Method | Use Case | Data Location |
|--------|----------|---------------|
| `GET` | Search forms, filters | URL query string |
| `POST` | Login, file upload, mutations | Request body |

---

## Labels: The Accessibility Essential

Always pair inputs with labels:

```html
<!-- Method 1: for/id pairing (recommended) -->
<label for="email">Email:</label>
<input type="email" id="email" name="email">

<!-- Method 2: Wrapping -->
<label>
  Email:
  <input type="email" name="email">
</label>
```

### Why Labels Matter

- **Accessibility:** Screen readers announce the label
- **Usability:** Clicking the label focuses the input
- **Mobile:** Larger touch target

> **Never skip labels.** For hidden labels, use `aria-label` or visually-hidden CSS.

---

## Text Input Types

### Basic Text Input

```html
<input type="text" name="username" placeholder="Enter username">
```

### Email

```html
<input type="email" name="email" placeholder="you@example.com">
```

- Mobile keyboards show @ symbol
- Browser validates email format

### Password

```html
<input type="password" name="password" minlength="8">
```

- Characters are masked
- Use `minlength` for basic validation

### URL

```html
<input type="url" name="website" placeholder="https://example.com">
```

### Telephone

```html
<input type="tel" name="phone" pattern="[0-9]{3}-[0-9]{3}-[0-9]{4}">
```

- Mobile shows phone keypad
- Use `pattern` for format validation

### Search

```html
<input type="search" name="query" placeholder="Search...">
```

- Shows clear button in some browsers
- Semantic meaning for search functionality

---

## Numeric Input Types

### Number

```html
<input type="number" name="quantity" min="1" max="100" step="1">
```

| Attribute | Purpose |
|-----------|---------|
| `min` | Minimum value |
| `max` | Maximum value |
| `step` | Increment amount |

### Range (Slider)

```html
<label for="volume">Volume: <span id="volume-value">50</span></label>
<input type="range" id="volume" name="volume" min="0" max="100" value="50"
       oninput="document.getElementById('volume-value').textContent = this.value">
```

---

## Date and Time Inputs

### Date

```html
<input type="date" name="birthday" min="1900-01-01" max="2025-12-31">
```

### Time

```html
<input type="time" name="meeting-time" min="09:00" max="18:00">
```

### DateTime-Local

```html
<input type="datetime-local" name="appointment">
```

### Month and Week

```html
<input type="month" name="start-month">
<input type="week" name="start-week">
```

### Browser Support Note

Date inputs render as text in older browsers. Consider a JavaScript fallback for critical applications.

---

## Selection Inputs

### Checkbox

```html
<label>
  <input type="checkbox" name="agree" value="yes" required>
  I agree to the terms
</label>

<!-- Multiple checkboxes -->
<fieldset>
  <legend>Select features:</legend>
  <label><input type="checkbox" name="features" value="ai"> AI Integration</label>
  <label><input type="checkbox" name="features" value="api"> API Access</label>
  <label><input type="checkbox" name="features" value="sso"> SSO</label>
</fieldset>
```

### Radio Buttons

```html
<fieldset>
  <legend>Choose a plan:</legend>
  <label><input type="radio" name="plan" value="free"> Free</label>
  <label><input type="radio" name="plan" value="pro" checked> Pro</label>
  <label><input type="radio" name="plan" value="enterprise"> Enterprise</label>
</fieldset>
```

Radio buttons with the same `name` are mutually exclusive.

### Select Dropdown

```html
<label for="country">Country:</label>
<select id="country" name="country">
  <option value="">-- Select --</option>
  <option value="us">United States</option>
  <option value="uk">United Kingdom</option>
  <option value="ca">Canada</option>
</select>
```

### Multi-Select

```html
<select name="languages" multiple size="4">
  <option value="en">English</option>
  <option value="es">Spanish</option>
  <option value="fr">French</option>
  <option value="de">German</option>
</select>
```

### Option Groups

```html
<select name="car">
  <optgroup label="Swedish Cars">
    <option value="volvo">Volvo</option>
    <option value="saab">Saab</option>
  </optgroup>
  <optgroup label="German Cars">
    <option value="mercedes">Mercedes</option>
    <option value="audi">Audi</option>
  </optgroup>
</select>
```

---

## Textarea

For multi-line text input:

```html
<label for="message">Your Message:</label>
<textarea id="message" name="message" rows="5" cols="40" 
          placeholder="Type your message here..."></textarea>
```

### AI Prompt Input Example

```html
<label for="prompt">Enter your prompt:</label>
<textarea id="prompt" name="prompt" rows="4" 
          placeholder="Ask the AI anything..."
          maxlength="4000"
          required></textarea>
<small>Max 4,000 characters</small>
```

---

## File Uploads

### Single File

```html
<label for="avatar">Upload Avatar:</label>
<input type="file" id="avatar" name="avatar" accept="image/*">
```

### Multiple Files

```html
<input type="file" name="documents" multiple accept=".pdf,.doc,.docx">
```

### For AI Image Analysis

```html
<form action="/api/analyze" method="POST" enctype="multipart/form-data">
  <label for="image">Upload image for analysis:</label>
  <input type="file" id="image" name="image" 
         accept="image/png, image/jpeg, image/webp"
         required>
  <button type="submit">Analyze Image</button>
</form>
```

> **Important:** File uploads require `enctype="multipart/form-data"` on the form.

---

## Hidden and Special Inputs

### Hidden Fields

```html
<input type="hidden" name="user_id" value="12345">
<input type="hidden" name="csrf_token" value="abc123xyz">
```

### Color Picker

```html
<label for="theme-color">Choose theme color:</label>
<input type="color" id="theme-color" name="theme-color" value="#6366f1">
```

---

## Buttons

### Submit Button

```html
<button type="submit">Send Message</button>
```

### Reset Button

```html
<button type="reset">Clear Form</button>
```

### Regular Button (for JavaScript)

```html
<button type="button" onclick="showPreview()">Preview</button>
```

> **Always specify `type`** on buttons. Default is `submit`, which can cause unexpected form submissions.

---

## Form Validation Attributes

HTML5 provides built-in validation:

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `required` | Field must be filled | `required` |
| `minlength` | Minimum characters | `minlength="8"` |
| `maxlength` | Maximum characters | `maxlength="100"` |
| `min` | Minimum number/date | `min="0"` |
| `max` | Maximum number/date | `max="100"` |
| `pattern` | Regex pattern | `pattern="[A-Za-z]+"` |
| `step` | Number increment | `step="0.01"` |

### Example with Validation

```html
<form>
  <label for="username">Username (3-15 characters, letters only):</label>
  <input type="text" id="username" name="username"
         required
         minlength="3"
         maxlength="15"
         pattern="[A-Za-z]+"
         title="Only letters, 3-15 characters">
  
  <button type="submit">Register</button>
</form>
```

---

## Grouping with Fieldset

```html
<form>
  <fieldset>
    <legend>Personal Information</legend>
    <label for="fname">First Name:</label>
    <input type="text" id="fname" name="fname">
    
    <label for="lname">Last Name:</label>
    <input type="text" id="lname" name="lname">
  </fieldset>
  
  <fieldset>
    <legend>Contact Details</legend>
    <label for="email">Email:</label>
    <input type="email" id="email" name="email">
  </fieldset>
  
  <button type="submit">Submit</button>
</form>
```

---

## Complete AI Chat Form Example

```html
<form id="chat-form" action="/api/chat" method="POST">
  <fieldset>
    <legend>AI Chat Settings</legend>
    
    <label for="model">Model:</label>
    <select id="model" name="model">
      <option value="gpt-4o">GPT-4o</option>
      <option value="claude-3">Claude 3</option>
      <option value="gemini">Gemini Pro</option>
    </select>
    
    <label for="temperature">Creativity: <span id="temp-value">0.7</span></label>
    <input type="range" id="temperature" name="temperature" 
           min="0" max="1" step="0.1" value="0.7"
           oninput="document.getElementById('temp-value').textContent = this.value">
  </fieldset>
  
  <label for="prompt">Your Prompt:</label>
  <textarea id="prompt" name="prompt" rows="4" 
            placeholder="Ask me anything..." 
            maxlength="4000" required></textarea>
  
  <label for="image">Attach Image (optional):</label>
  <input type="file" id="image" name="image" accept="image/*">
  
  <button type="submit">Send to AI</button>
</form>
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always use `<label>` | Accessibility requirement |
| Set appropriate `type` | Mobile keyboards, validation |
| Use `required` for mandatory fields | Built-in validation |
| Add `autocomplete` attributes | Better UX |
| Use `placeholder` as hint, not label | Labels are essential |
| Set `enctype` for file uploads | Required for files |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Missing labels | Add `<label>` for every input |
| Placeholder as only label | Always include a real label |
| Wrong input type | Use `email` for emails, `tel` for phones |
| Forgetting `enctype` for files | Add `multipart/form-data` |
| Button without `type` | Specify `type="button"` or `type="submit"` |

---

## Hands-on Exercise

### Your Task

Create a form for an AI image generator with:
1. Text input for the prompt (required, max 500 chars)
2. Dropdown for style (realistic, artistic, cartoon)
3. Number input for image count (1-4)
4. Checkbox to agree to terms
5. Submit button

<details>
<summary>✅ Solution</summary>

```html
<form action="/api/generate" method="POST">
  <label for="prompt">Image Prompt:</label>
  <textarea id="prompt" name="prompt" required maxlength="500" rows="3"
            placeholder="Describe the image you want..."></textarea>
  
  <label for="style">Style:</label>
  <select id="style" name="style">
    <option value="realistic">Realistic</option>
    <option value="artistic">Artistic</option>
    <option value="cartoon">Cartoon</option>
  </select>
  
  <label for="count">Number of Images:</label>
  <input type="number" id="count" name="count" min="1" max="4" value="1">
  
  <label>
    <input type="checkbox" name="terms" required>
    I agree to the terms of service
  </label>
  
  <button type="submit">Generate Images</button>
</form>
```
</details>

---

## Summary

✅ Forms collect user input using `<form>`, `<input>`, `<label>`, and other elements

✅ Choose the right `type` for each input—it affects keyboard, validation, and UX

✅ Always pair inputs with labels for accessibility

✅ Use HTML5 validation attributes: `required`, `min`, `max`, `pattern`

✅ File uploads require `enctype="multipart/form-data"`

✅ Group related fields with `<fieldset>` and `<legend>`

---

**Previous:** [Semantic Elements](./02-semantic-elements.md)

**Next:** [Accessibility & ARIA](./04-accessibility-aria.md)
