---
title: "Form Handling"
---

# Form Handling

## Introduction

Forms are the primary way users provide input to web applications. Whether it's a simple search box, a chat prompt field, or a complex configuration panel for an AI model, proper form handling ensures data is captured correctly, validated before submission, and processed without page reloads.

Modern JavaScript provides powerful APIs for working with forms—from accessing field values to implementing custom validation logic. Let's master these tools.

### What We'll Cover
- Accessing form and input values
- The `FormData` API for collecting form data
- HTML5 validation attributes (`required`, `pattern`, `minlength`)
- Constraint Validation API (`checkValidity()`, `reportValidity()`, `setCustomValidity()`)
- Real-time validation patterns
- Handling form submission with JavaScript
- Preventing default submission behavior

### Prerequisites
- [Event Handling](./03-event-handling.md) (addEventListener, preventDefault)
- [Selecting Elements](./01-selecting-elements.md) (querySelector, getElementById)
- [JavaScript Core Concepts](../../03-javascript-core-concepts/00-javascript-core-concepts.md) (objects, functions)

---

## Accessing Form and Input Values

### Getting Input Values

```javascript
const emailInput = document.getElementById('email');
const messageInput = document.getElementById('message');

console.log('Email:', emailInput.value);
console.log('Message:', messageInput.value);
```

**Output** (if user entered values):
```
Email: user@example.com
Message: Hello AI!
```

### Getting Values from Different Input Types

```javascript
// Text input
const textValue = document.getElementById('username').value;

// Checkbox (checked or not)
const agreeCheckbox = document.getElementById('agree-terms');
const isChecked = agreeCheckbox.checked; // true or false

// Radio buttons (get selected value)
const selectedGender = document.querySelector('input[name="gender"]:checked');
console.log(selectedGender ? selectedGender.value : 'Not selected');

// Select dropdown
const countrySelect = document.getElementById('country');
const selectedCountry = countrySelect.value;

// Textarea
const bio = document.getElementById('bio').value;
```

**Output:**
```
john_doe
true
male
United States
Software developer with 10 years of experience.
```

### Accessing the Entire Form

```javascript
const form = document.getElementById('signup-form');

// Access inputs by name attribute
const email = form.elements.email.value;
const password = form.elements.password.value;

console.log('Email:', email);
console.log('Password:', password);
```

---

## The FormData API

The `FormData` API simplifies collecting **all form values** at once, especially for forms with many fields.

### Creating FormData from a Form

```javascript
const form = document.getElementById('signup-form');

form.addEventListener('submit', (event) => {
  event.preventDefault();
  
  // Create FormData object from form
  const formData = new FormData(form);
  
  // Get individual values
  console.log('Email:', formData.get('email'));
  console.log('Password:', formData.get('password'));
  
  // Iterate over all entries
  for (const [key, value] of formData.entries()) {
    console.log(`${key}: ${value}`);
  }
});
```

**Output** (when form is submitted):
```
Email: user@example.com
Password: ********
email: user@example.com
password: ********
username: john_doe
```

### Manually Building FormData

```javascript
const formData = new FormData();

formData.append('email', 'user@example.com');
formData.append('message', 'Hello AI!');
formData.append('timestamp', Date.now());

// Send to server
fetch('/api/contact', {
  method: 'POST',
  body: formData // Automatically sets correct Content-Type
});
```

### Working with FormData Methods

```javascript
const formData = new FormData();

// Add values
formData.append('username', 'john_doe');
formData.append('tags', 'javascript');
formData.append('tags', 'ai'); // Multiple values for same key

// Get first value
console.log(formData.get('username')); // "john_doe"

// Get all values for a key
console.log(formData.getAll('tags')); // ["javascript", "ai"]

// Check if key exists
console.log(formData.has('username')); // true

// Delete a key
formData.delete('tags');

// Update a value
formData.set('username', 'jane_doe'); // Replaces existing value
```

**Output:**
```
john_doe
["javascript", "ai"]
true
```

---

## HTML5 Validation Attributes

Modern browsers support **built-in validation** using HTML attributes. No JavaScript required for basic checks.

### Common Validation Attributes

```html
<form id="signup-form">
  <!-- Required field -->
  <input type="text" name="username" required>
  
  <!-- Email validation -->
  <input type="email" name="email" required>
  
  <!-- Minimum length -->
  <input type="password" name="password" minlength="8" required>
  
  <!-- Pattern (regex) -->
  <input type="text" name="phone" pattern="[0-9]{3}-[0-9]{3}-[0-9]{4}" 
         placeholder="123-456-7890" required>
  
  <!-- Min/Max for numbers -->
  <input type="number" name="age" min="18" max="120" required>
  
  <!-- Min/Max for dates -->
  <input type="date" name="start-date" min="2025-01-01" max="2025-12-31">
  
  <!-- Maxlength -->
  <textarea name="bio" maxlength="500"></textarea>
  
  <button type="submit">Submit</button>
</form>
```

**Browser behavior:**
- **Submission blocked** if validation fails
- **Error messages** shown automatically (browser's built-in UI)
- **CSS pseudo-classes** applied (`:valid`, `:invalid`)

### Styling with Validation Pseudo-classes

```css
/* Style valid inputs */
input:valid {
  border-color: green;
}

/* Style invalid inputs */
input:invalid {
  border-color: red;
}

/* Only show invalid state after user interaction */
input:user-invalid {
  border-color: red;
}
```

---

## Constraint Validation API

The **Constraint Validation API** lets you **check validity** and **customize error messages** programmatically.

### checkValidity() - Check if Form/Input is Valid

Returns `true` if valid, `false` if not (doesn't show messages to user).

```javascript
const form = document.getElementById('signup-form');
const emailInput = document.getElementById('email');

// Check single input
if (emailInput.checkValidity()) {
  console.log('Email is valid');
} else {
  console.log('Email is invalid');
}

// Check entire form
if (form.checkValidity()) {
  console.log('Form is valid');
} else {
  console.log('Form has errors');
}
```

**Output** (if email is empty and required):
```
Email is invalid
Form has errors
```

### reportValidity() - Check and Show Error Messages

Like `checkValidity()`, but **shows browser's validation UI** to the user.

```javascript
const form = document.getElementById('signup-form');

form.addEventListener('submit', (event) => {
  event.preventDefault();
  
  if (form.reportValidity()) {
    console.log('Form is valid, submitting...');
    // Submit form data
  } else {
    console.log('Form has errors (user sees messages)');
  }
});
```

**Browser behavior:** Error messages appear on invalid fields.

### setCustomValidity() - Set Custom Error Messages

Override browser's default error message with your own.

```javascript
const passwordInput = document.getElementById('password');
const confirmPasswordInput = document.getElementById('confirm-password');

confirmPasswordInput.addEventListener('input', () => {
  if (confirmPasswordInput.value !== passwordInput.value) {
    confirmPasswordInput.setCustomValidity('Passwords do not match');
  } else {
    confirmPasswordInput.setCustomValidity(''); // Clear error
  }
});
```

**Output:** If passwords don't match, browser shows "Passwords do not match" instead of generic message.

> **Important:** Always clear the custom message with `setCustomValidity('')` when the input becomes valid.

### ValidityState - Detailed Validation Info

Each input has a `validity` property with detailed error flags.

```javascript
const emailInput = document.getElementById('email');

console.log('Valid:', emailInput.validity.valid);
console.log('Missing value:', emailInput.validity.valueMissing);
console.log('Type mismatch:', emailInput.validity.typeMismatch);
console.log('Pattern mismatch:', emailInput.validity.patternMismatch);
console.log('Too long:', emailInput.validity.tooLong);
console.log('Too short:', emailInput.validity.tooShort);
console.log('Range underflow:', emailInput.validity.rangeUnderflow);
console.log('Range overflow:', emailInput.validity.rangeOverflow);

// Get the error message
console.log('Validation message:', emailInput.validationMessage);
```

**Output** (if email is empty):
```
Valid: false
Missing value: true
Type mismatch: false
Pattern mismatch: false
Too long: false
Too short: false
Range underflow: false
Range overflow: false
Validation message: Please fill out this field.
```

---

## Real-time Validation Patterns

Validate as the user types for better UX.

### Validate on Input (Every Keystroke)

```javascript
const emailInput = document.getElementById('email');
const emailError = document.getElementById('email-error');

emailInput.addEventListener('input', () => {
  if (emailInput.validity.valid) {
    emailError.textContent = ''; // Clear error
    emailInput.classList.remove('invalid');
  } else {
    emailError.textContent = emailInput.validationMessage;
    emailInput.classList.add('invalid');
  }
});
```

**HTML:**
```html
<input type="email" id="email" required>
<span id="email-error" style="color: red;"></span>
```

**Output** (when user types invalid email):
```
Please include an '@' in the email address.
```

### Validate on Blur (When User Leaves Field)

```javascript
const usernameInput = document.getElementById('username');
const usernameError = document.getElementById('username-error');

usernameInput.addEventListener('blur', () => {
  if (!usernameInput.checkValidity()) {
    usernameError.textContent = 'Username must be at least 3 characters';
  } else {
    usernameError.textContent = '';
  }
});
```

**Use case:** Don't annoy users with errors while they're still typing—wait until they move to the next field.

### Custom Validation Logic

```javascript
const passwordInput = document.getElementById('password');
const strengthIndicator = document.getElementById('password-strength');

passwordInput.addEventListener('input', () => {
  const password = passwordInput.value;
  let strength = 'Weak';
  
  if (password.length >= 12 && /[A-Z]/.test(password) && /[0-9]/.test(password)) {
    strength = 'Strong';
  } else if (password.length >= 8) {
    strength = 'Medium';
  }
  
  strengthIndicator.textContent = `Password strength: ${strength}`;
  
  // Set custom validity if too weak
  if (strength === 'Weak' && password.length > 0) {
    passwordInput.setCustomValidity('Password is too weak');
  } else {
    passwordInput.setCustomValidity('');
  }
});
```

**Output** (when user types "pass"):
```
Password strength: Weak
```

---

## Form Submission Handling

### Prevent Default Submission and Handle with JavaScript

```javascript
const form = document.getElementById('contact-form');

form.addEventListener('submit', async (event) => {
  event.preventDefault(); // Don't reload page
  
  // Check validity
  if (!form.checkValidity()) {
    form.reportValidity(); // Show errors
    return;
  }
  
  // Collect form data
  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());
  
  console.log('Submitting:', data);
  
  try {
    // Send to server
    const response = await fetch('/api/contact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    if (response.ok) {
      console.log('Form submitted successfully!');
      form.reset(); // Clear form
    } else {
      console.error('Submission failed');
    }
  } catch (error) {
    console.error('Network error:', error);
  }
});
```

**Output:**
```
Submitting: { email: "user@example.com", message: "Hello!" }
Form submitted successfully!
```

### Disabling Submit Button During Submission

```javascript
const form = document.getElementById('contact-form');
const submitButton = form.querySelector('button[type="submit"]');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  
  // Disable button to prevent double submission
  submitButton.disabled = true;
  submitButton.textContent = 'Submitting...';
  
  try {
    const formData = new FormData(form);
    const response = await fetch('/api/contact', {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      console.log('Success!');
    }
  } finally {
    // Re-enable button
    submitButton.disabled = false;
    submitButton.textContent = 'Submit';
  }
});
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| **Use HTML5 validation attributes** | Free validation with browser support (required, pattern, minlength) |
| **Validate on blur, not every keystroke** | Don't annoy users with errors while typing |
| **Always call `preventDefault()` in submit handlers** | Prevents page reload for JavaScript-handled forms |
| **Clear custom validity when input becomes valid** | Call `setCustomValidity('')` to remove error state |
| **Disable submit button during submission** | Prevents double submissions |
| **Use `FormData` for easy data collection** | Simpler than manually accessing each field |
| **Provide real-time feedback for passwords** | Show strength indicator as user types |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not calling `preventDefault()` in submit handler | Always use `event.preventDefault()` to avoid page reload |
| Forgetting to clear `setCustomValidity('')` | Input will remain invalid even if user fixes it |
| Validating on every keystroke (annoying) | Use `blur` event or wait until user pauses typing |
| Not checking `checkValidity()` before submission | Always validate before processing form data |
| Using `innerHTML` to show user input | Use `textContent` to prevent XSS attacks |
| Submitting forms without disabling button | User can click multiple times, causing duplicate submissions |

---

## Hands-on Exercise

### Your Task

Build an **AI chat prompt form** with real-time validation:
- **Email field** (required, valid email format)
- **Prompt field** (required, min 10 characters, max 500 characters)
- **Model selector** (dropdown with GPT-4, Claude, Gemini)
- Submit button shows error if validation fails
- Show character count for prompt field (updates in real-time)
- Disable submit button while "sending" (simulate 2-second delay)

### Requirements

1. Create HTML form with:
   - Email input (`type="email"`, `required`)
   - Textarea for prompt (`minlength="10"`, `maxlength="500"`, `required`)
   - Select dropdown for model (`required`)
   - Character counter (`<span id="char-count">0 / 500</span>`)
   - Submit button

2. **Real-time character count:**
   - Update counter on every keystroke in prompt field
   - Show remaining characters

3. **Validation:**
   - Check `form.checkValidity()` on submit
   - Show errors with `form.reportValidity()` if invalid

4. **Submission:**
   - `preventDefault()` to avoid page reload
   - Disable submit button, change text to "Sending..."
   - Simulate API call with `setTimeout(..., 2000)`
   - Log form data to console
   - Re-enable button after 2 seconds

### Expected Result

- Character counter updates as user types
- Submit button disabled if form invalid
- Clicking submit logs data and shows "Sending..." for 2 seconds
- Form resets after successful submission

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `input` event on textarea to update character count
- Get length with `textarea.value.length`
- Use `form.checkValidity()` before processing
- Disable button with `button.disabled = true`
- Use `setTimeout(() => { ... }, 2000)` to simulate delay
- Reset form with `form.reset()`
- Get FormData with `new FormData(form)`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**HTML:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Chat Prompt Form</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 600px;
      margin: 50px auto;
      padding: 20px;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 15px;
    }
    label {
      font-weight: bold;
    }
    input, textarea, select {
      padding: 10px;
      font-size: 14px;
      border: 2px solid #ddd;
      border-radius: 4px;
    }
    input:valid, textarea:valid, select:valid {
      border-color: green;
    }
    input:invalid, textarea:invalid, select:invalid {
      border-color: red;
    }
    #char-count {
      font-size: 12px;
      color: #666;
      text-align: right;
    }
    button {
      padding: 12px;
      font-size: 16px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    button:hover:not(:disabled) {
      background-color: #0056b3;
    }
    button:disabled {
      background-color: #ccc;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <h1>AI Chat Prompt</h1>
  
  <form id="prompt-form">
    <div>
      <label for="email">Email:</label>
      <input type="email" id="email" name="email" required>
    </div>
    
    <div>
      <label for="prompt">Your Prompt:</label>
      <textarea id="prompt" name="prompt" rows="6" 
                minlength="10" maxlength="500" required></textarea>
      <div id="char-count">0 / 500</div>
    </div>
    
    <div>
      <label for="model">AI Model:</label>
      <select id="model" name="model" required>
        <option value="">-- Select Model --</option>
        <option value="gpt-4">GPT-4</option>
        <option value="claude">Claude</option>
        <option value="gemini">Gemini</option>
      </select>
    </div>
    
    <button type="submit">Send Prompt</button>
  </form>

  <script src="script.js"></script>
</body>
</html>
```

**JavaScript (script.js):**
```javascript
const form = document.getElementById('prompt-form');
const promptTextarea = document.getElementById('prompt');
const charCountSpan = document.getElementById('char-count');
const submitButton = form.querySelector('button[type="submit"]');

// Update character count in real-time
promptTextarea.addEventListener('input', () => {
  const currentLength = promptTextarea.value.length;
  const maxLength = promptTextarea.maxLength;
  
  charCountSpan.textContent = `${currentLength} / ${maxLength}`;
  
  // Change color if approaching limit
  if (currentLength > maxLength - 50) {
    charCountSpan.style.color = 'red';
  } else {
    charCountSpan.style.color = '#666';
  }
});

// Handle form submission
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  
  // Check validity
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }
  
  // Collect form data
  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());
  
  console.log('Submitting prompt:', data);
  
  // Disable button during "submission"
  submitButton.disabled = true;
  submitButton.textContent = 'Sending...';
  
  // Simulate API call (2 second delay)
  setTimeout(() => {
    console.log('Prompt sent successfully!');
    
    // Reset form
    form.reset();
    charCountSpan.textContent = '0 / 500';
    charCountSpan.style.color = '#666';
    
    // Re-enable button
    submitButton.disabled = false;
    submitButton.textContent = 'Send Prompt';
    
    alert('Prompt sent successfully!');
  }, 2000);
});
```

**Output** (in console after submission):
```
Submitting prompt: { email: "user@example.com", prompt: "Explain quantum computing in simple terms", model: "gpt-4" }
Prompt sent successfully!
```

</details>

### Bonus Challenges

- [ ] Add a "Save Draft" button that stores form data in `localStorage`
- [ ] Restore saved draft on page load
- [ ] Add a custom validation message if prompt is too short
- [ ] Show a visual indicator (spinner) while submitting
- [ ] Add a "Clear" button that resets the form

---

## Summary

✅ **Access form values** with `input.value` or `form.elements.fieldName.value`  
✅ **Use FormData API** for easy data collection from entire forms  
✅ **HTML5 validation attributes** (`required`, `pattern`, `minlength`) provide built-in validation  
✅ **Constraint Validation API** (`checkValidity()`, `reportValidity()`, `setCustomValidity()`) for custom logic  
✅ **Real-time validation** on `input` or `blur` events improves UX  
✅ **Always call `preventDefault()`** in submit handlers for JavaScript-handled forms  
✅ **Disable submit button** during submission to prevent duplicate requests

**Next:** [Observer APIs](./05-observer-apis.md)

---

## Further Reading

- [MDN: FormData](https://developer.mozilla.org/en-US/docs/Web/API/FormData) - Complete FormData reference
- [MDN: Form Validation](https://developer.mozilla.org/en-US/docs/Learn/Forms/Form_validation) - Validation guide
- [MDN: Constraint Validation API](https://developer.mozilla.org/en-US/docs/Web/API/Constraint_validation) - API reference
- [Observer APIs](./05-observer-apis.md) - Next lesson

<!-- 
Sources Consulted:
- MDN FormData: https://developer.mozilla.org/en-US/docs/Web/API/FormData
- MDN Form validation: https://developer.mozilla.org/en-US/docs/Learn/Forms/Form_validation
- MDN HTMLFormElement: https://developer.mozilla.org/en-US/docs/Web/API/HTMLFormElement
- MDN ValidityState: https://developer.mozilla.org/en-US/docs/Web/API/ValidityState
-->
