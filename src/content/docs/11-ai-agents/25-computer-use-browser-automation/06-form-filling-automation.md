---
title: "Form Filling Automation"
---

# Form Filling Automation

## Introduction

Forms are the primary interface between users and web applications — login pages, registration flows, search queries, checkout processes, support tickets. Automating form filling is one of the most valuable capabilities for browser agents, but it requires understanding form structure, handling validation, dealing with multi-step flows, and managing dynamic fields.

In this lesson, we build reliable form automation using Playwright's input methods, combine them with AI-powered field detection for unfamiliar forms, and handle the edge cases that make forms tricky: dropdowns, checkboxes, file uploads, and validation errors.

### What We'll Cover
- Detecting and mapping form fields automatically
- Filling text inputs, dropdowns, checkboxes, and radio buttons
- Handling form validation and error recovery
- Multi-step form submission (wizards)
- AI-powered form understanding for unfamiliar forms
- File upload automation

### Prerequisites
- Playwright actions: click, type, scroll (Lesson 03)
- Visual element identification and locators (Lesson 04)
- HTML form elements (Unit 1)
- Basic understanding of form validation (Unit 1)

---

## Detecting Form Fields

Before filling a form, an agent needs to understand its structure — what fields exist, what type they are, and which are required.

### Automatic Form Mapping

```python
from playwright.sync_api import sync_playwright
import json

def map_form_fields(page, form_selector: str = "form") -> list:
    """Map all fields in a form with their types and attributes."""
    return page.evaluate(f"""
        (selector) => {{
            const form = document.querySelector(selector);
            if (!form) return [];
            
            const fields = [];
            const inputs = form.querySelectorAll(
                'input, select, textarea'
            );
            
            inputs.forEach(el => {{
                // Find associated label
                let label = '';
                if (el.id) {{
                    const labelEl = form.querySelector(
                        `label[for="${{el.id}}"]`
                    );
                    if (labelEl) label = labelEl.textContent.trim();
                }}
                if (!label && el.closest('label')) {{
                    label = el.closest('label').textContent.trim();
                }}
                
                const field = {{
                    tag: el.tagName.toLowerCase(),
                    type: el.type || 'text',
                    name: el.name || null,
                    id: el.id || null,
                    label: label || el.placeholder || el.name || 'unlabeled',
                    placeholder: el.placeholder || null,
                    required: el.required,
                    value: el.value,
                    options: null,
                    checked: null
                }};
                
                // Handle select elements
                if (el.tagName === 'SELECT') {{
                    field.options = Array.from(el.options).map(o => ({{
                        value: o.value,
                        text: o.textContent.trim(),
                        selected: o.selected
                    }}));
                }}
                
                // Handle checkboxes and radio buttons
                if (el.type === 'checkbox' || el.type === 'radio') {{
                    field.checked = el.checked;
                }}
                
                fields.push(field);
            }});
            
            return fields;
        }}
    """, form_selector)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form id="registration">
            <label for="name">Full Name</label>
            <input id="name" name="name" type="text" required placeholder="John Doe" />
            
            <label for="email">Email</label>
            <input id="email" name="email" type="email" required />
            
            <label for="country">Country</label>
            <select id="country" name="country">
                <option value="">Select...</option>
                <option value="us">United States</option>
                <option value="uk">United Kingdom</option>
                <option value="ca">Canada</option>
            </select>
            
            <label>
                <input type="checkbox" name="terms" required />
                I agree to terms
            </label>
            
            <label for="bio">Short Bio</label>
            <textarea id="bio" name="bio" placeholder="Tell us about yourself"></textarea>
            
            <button type="submit">Register</button>
        </form>
    """)
    
    fields = map_form_fields(page)
    print(f"Found {len(fields)} fields:")
    for f in fields:
        req = " (required)" if f["required"] else ""
        print(f"  {f['type']:10} | {f['label']}{req}")
        if f["options"]:
            for opt in f["options"]:
                print(f"             → {opt['text']} ({opt['value']})")
    
    browser.close()
```

**Output:**
```
Found 5 fields:
  text       | Full Name (required)
  email      | Email (required)
  select-one | Country
             → Select... ()
             → United States (us)
             → United Kingdom (uk)
             → Canada (ca)
  checkbox   | I agree to terms (required)
  textarea   | Short Bio
```

---

## Filling Different Field Types

### Text Inputs and Textareas

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="name">Name</label>
            <input id="name" type="text" />
            
            <label for="email">Email</label>
            <input id="email" type="email" />
            
            <label for="phone">Phone</label>
            <input id="phone" type="tel" />
            
            <label for="bio">Bio</label>
            <textarea id="bio"></textarea>
        </form>
    """)
    
    # Fill by label (recommended)
    page.get_by_label("Name").fill("Alice Johnson")
    page.get_by_label("Email").fill("alice@example.com")
    page.get_by_label("Phone").fill("+1-555-0123")
    page.get_by_label("Bio").fill("AI developer and researcher.\nLoves automation.")
    
    # Verify
    print(f"Name: {page.locator('#name').input_value()}")
    print(f"Email: {page.locator('#email').input_value()}")
    print(f"Phone: {page.locator('#phone').input_value()}")
    print(f"Bio: {page.locator('#bio').input_value()}")
    
    browser.close()
```

**Output:**
```
Name: Alice Johnson
Email: alice@example.com
Phone: +1-555-0123
Bio: AI developer and researcher.
Loves automation.
```

### Dropdowns (Select Elements)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="country">Country</label>
            <select id="country">
                <option value="">Select a country</option>
                <option value="us">United States</option>
                <option value="uk">United Kingdom</option>
                <option value="ca">Canada</option>
            </select>
            
            <label for="colors">Favorite Colors</label>
            <select id="colors" multiple>
                <option value="red">Red</option>
                <option value="blue">Blue</option>
                <option value="green">Green</option>
            </select>
        </form>
    """)
    
    # Select by value
    page.get_by_label("Country").select_option("uk")
    print(f"Country (by value): {page.locator('#country').input_value()}")
    
    # Select by visible text
    page.get_by_label("Country").select_option(label="Canada")
    print(f"Country (by label): {page.locator('#country').input_value()}")
    
    # Multi-select: select multiple options
    page.get_by_label("Favorite Colors").select_option(["red", "green"])
    selected = page.evaluate("""
        () => Array.from(document.getElementById('colors').selectedOptions)
            .map(o => o.textContent)
    """)
    print(f"Colors: {selected}")
    
    browser.close()
```

**Output:**
```
Country (by value): uk
Country (by label): ca
Colors: ['Red', 'Green']
```

### Checkboxes and Radio Buttons

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label>
                <input type="checkbox" name="newsletter" />
                Subscribe to newsletter
            </label>
            <label>
                <input type="checkbox" name="terms" />
                Accept terms and conditions
            </label>
            
            <fieldset>
                <legend>Preferred contact</legend>
                <label><input type="radio" name="contact" value="email" /> Email</label>
                <label><input type="radio" name="contact" value="phone" /> Phone</label>
                <label><input type="radio" name="contact" value="sms" /> SMS</label>
            </fieldset>
        </form>
    """)
    
    # Check checkboxes
    page.get_by_label("Subscribe to newsletter").check()
    page.get_by_label("Accept terms and conditions").check()
    
    print(f"Newsletter: {page.get_by_label('Subscribe to newsletter').is_checked()}")
    print(f"Terms: {page.get_by_label('Accept terms and conditions').is_checked()}")
    
    # Uncheck
    page.get_by_label("Subscribe to newsletter").uncheck()
    print(f"Newsletter after uncheck: {page.get_by_label('Subscribe to newsletter').is_checked()}")
    
    # Radio buttons
    page.get_by_label("Phone").check()
    print(f"Email selected: {page.get_by_label('Email').is_checked()}")
    print(f"Phone selected: {page.get_by_label('Phone').is_checked()}")
    
    browser.close()
```

**Output:**
```
Newsletter: True
Terms: True
Newsletter after uncheck: False
Email selected: False
Phone selected: True
```

### File Uploads

```python
from playwright.sync_api import sync_playwright
import tempfile
import os

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="avatar">Upload Avatar</label>
            <input id="avatar" type="file" accept="image/*" />
            
            <label for="docs">Upload Documents</label>
            <input id="docs" type="file" multiple />
            
            <div id="status"></div>
            <script>
                document.getElementById('avatar').addEventListener('change', e => {
                    document.getElementById('status').textContent = 
                        'Files: ' + Array.from(e.target.files).map(f => f.name).join(', ');
                });
            </script>
        </form>
    """)
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"fake image data")
        temp_path = f.name
    
    try:
        # Single file upload
        page.locator("#avatar").set_input_files(temp_path)
        status = page.locator("#status").inner_text()
        print(f"Upload status: {status}")
        
        # Clear file input
        page.locator("#avatar").set_input_files([])
        print("Files cleared")
    finally:
        os.unlink(temp_path)
    
    browser.close()
```

**Output:**
```
Upload status: Files: tmp12345678.png
Files cleared
```

---

## Handling Validation and Errors

Forms validate input and show error messages. An agent needs to detect validation failures and correct them.

### Detecting Validation Errors

```python
from playwright.sync_api import sync_playwright

def detect_form_errors(page) -> list:
    """Detect validation errors on the current page."""
    return page.evaluate("""
        () => {
            const errors = [];
            
            // Check HTML5 validation
            document.querySelectorAll('input, select, textarea').forEach(el => {
                if (!el.validity.valid) {
                    errors.push({
                        field: el.name || el.id || 'unknown',
                        message: el.validationMessage,
                        type: 'html5_validation'
                    });
                }
            });
            
            // Check for visible error messages
            const errorSelectors = [
                '.error', '.error-message', '[class*="error"]',
                '.invalid-feedback', '[role="alert"]'
            ];
            errorSelectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    if (el.textContent.trim()) {
                        errors.push({
                            field: 'page',
                            message: el.textContent.trim(),
                            type: 'visible_error'
                        });
                    }
                });
            });
            
            return errors;
        }
    """)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form id="myform">
            <label for="email">Email</label>
            <input id="email" name="email" type="email" required />
            <span class="error" id="email-error"></span>
            
            <label for="age">Age</label>
            <input id="age" name="age" type="number" min="18" max="120" required />
            <span class="error" id="age-error"></span>
            
            <button type="submit">Submit</button>
        </form>
        <script>
            document.getElementById('myform').addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('email');
                const age = document.getElementById('age');
                
                document.getElementById('email-error').textContent = 
                    email.validity.valid ? '' : 'Please enter a valid email';
                document.getElementById('age-error').textContent = 
                    age.validity.valid ? '' : 'Age must be between 18 and 120';
            });
        </script>
    """)
    
    # Try submitting empty form
    page.get_by_role("button", name="Submit").click()
    
    errors = detect_form_errors(page)
    print(f"Errors found: {len(errors)}")
    for err in errors:
        print(f"  {err['field']}: {err['message']} ({err['type']})")
    
    # Fix the errors
    page.get_by_label("Email").fill("alice@example.com")
    page.get_by_label("Age").fill("25")
    page.get_by_role("button", name="Submit").click()
    
    errors_after = detect_form_errors(page)
    print(f"\nErrors after fix: {len(errors_after)}")
    
    browser.close()
```

**Output:**
```
Errors found: 4
  email: Please fill out this field. (html5_validation)
  age: Please fill out this field. (html5_validation)
  page: Please enter a valid email (visible_error)
  page: Age must be between 18 and 120 (visible_error)

Errors after fix: 0
```

### Retry with Correction

```python
from playwright.sync_api import sync_playwright

class FormFiller:
    """Fill forms with automatic validation and retry."""
    
    def __init__(self, page):
        self.page = page
    
    def fill_form(self, data: dict, max_retries: int = 3) -> dict:
        """
        Fill a form with the given data and retry on validation errors.
        
        Args:
            data: Dict mapping field labels to values
            max_retries: Maximum correction attempts
        """
        result = {"success": False, "attempts": 0, "errors": []}
        
        for attempt in range(max_retries):
            result["attempts"] = attempt + 1
            
            # Fill all fields
            for label, value in data.items():
                field = self.page.get_by_label(label)
                if not field.count():
                    continue
                
                tag = field.evaluate("el => el.tagName.toLowerCase()")
                input_type = field.evaluate(
                    "el => el.type || 'text'"
                )
                
                if input_type == "checkbox":
                    if value:
                        field.check()
                    else:
                        field.uncheck()
                elif input_type == "radio":
                    field.check()
                elif tag == "select":
                    field.select_option(label=str(value))
                else:
                    field.fill(str(value))
            
            # Try to submit
            submit = self.page.get_by_role("button", name="Submit")
            if submit.count() == 0:
                submit = self.page.locator("[type='submit']")
            submit.click()
            
            # Check for errors
            self.page.wait_for_timeout(500)
            errors = self.page.locator(
                ".error, .error-message, [role='alert']"
            ).all_inner_texts()
            errors = [e for e in errors if e.strip()]
            
            if not errors:
                result["success"] = True
                return result
            
            result["errors"] = errors
        
        return result

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="email">Email</label>
            <input id="email" name="email" type="text" />
            <div class="error" id="email-err"></div>
            
            <button type="button" onclick="validate()">Submit</button>
            <div id="result"></div>
        </form>
        <script>
            function validate() {
                const email = document.getElementById('email').value;
                const err = document.getElementById('email-err');
                if (!email.includes('@')) {
                    err.textContent = 'Invalid email address';
                } else {
                    err.textContent = '';
                    document.getElementById('result').textContent = 'Success!';
                }
            }
        </script>
    """)
    
    filler = FormFiller(page)
    
    # First attempt with valid data
    result = filler.fill_form({"Email": "alice@example.com"})
    print(f"Result: {result}")
    print(f"Page says: {page.locator('#result').inner_text()}")
    
    browser.close()
```

**Output:**
```
Result: {'success': True, 'attempts': 1, 'errors': []}
Page says: Success!
```

---

## Multi-Step Forms (Wizards)

Many forms span multiple pages or steps. An agent needs to handle step navigation, state preservation, and conditional logic.

```python
from playwright.sync_api import sync_playwright
import json

class WizardHandler:
    """Handle multi-step form wizards."""
    
    def __init__(self, page):
        self.page = page
        self.steps_completed = []
    
    def detect_current_step(self) -> dict:
        """Detect which step of the wizard we're on."""
        return self.page.evaluate("""
            () => {
                // Look for step indicators
                const active = document.querySelector(
                    '.step.active, [aria-current="step"], .current-step'
                );
                const steps = document.querySelectorAll(
                    '.step, [role="tab"]'
                );
                const visibleFields = document.querySelectorAll(
                    'input:not([type="hidden"]):not([style*="display: none"]), select, textarea'
                );
                
                return {
                    currentStep: active ? active.textContent.trim() : 'unknown',
                    totalSteps: steps.length || 1,
                    visibleFields: Array.from(visibleFields).map(el => ({
                        type: el.type || el.tagName.toLowerCase(),
                        name: el.name,
                        label: el.labels?.[0]?.textContent?.trim() || 
                               el.placeholder || el.name
                    }))
                };
            }
        """)
    
    def fill_current_step(self, data: dict):
        """Fill all visible fields in the current step."""
        for label, value in data.items():
            field = self.page.get_by_label(label)
            if field.count() > 0 and field.is_visible():
                input_type = field.evaluate("el => el.type || 'text'")
                if input_type == "checkbox":
                    if value:
                        field.check()
                elif input_type in ("text", "email", "tel", "password", "number"):
                    field.fill(str(value))
                elif field.evaluate("el => el.tagName") == "SELECT":
                    field.select_option(label=str(value))
    
    def go_next(self) -> bool:
        """Click the Next button if it exists."""
        next_btn = self.page.get_by_role("button", name="Next")
        if next_btn.count() > 0:
            next_btn.click()
            self.page.wait_for_timeout(500)
            return True
        return False
    
    def submit(self) -> bool:
        """Click the final Submit button."""
        submit_btn = self.page.get_by_role("button", name="Submit")
        if submit_btn.count() > 0:
            submit_btn.click()
            self.page.wait_for_timeout(500)
            return True
        return False

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div id="wizard">
            <div class="step active" id="step1">
                <h3>Step 1: Personal Info</h3>
                <label for="name">Name</label>
                <input id="name" type="text" />
                <button onclick="nextStep(2)">Next</button>
            </div>
            <div class="step" id="step2" style="display:none">
                <h3>Step 2: Contact</h3>
                <label for="email">Email</label>
                <input id="email" type="email" />
                <button onclick="nextStep(3)">Next</button>
            </div>
            <div class="step" id="step3" style="display:none">
                <h3>Step 3: Confirm</h3>
                <div id="summary"></div>
                <button onclick="submitForm()">Submit</button>
            </div>
            <div id="result"></div>
        </div>
        <script>
            function nextStep(n) {
                document.querySelectorAll('.step').forEach(s => {
                    s.style.display = 'none';
                    s.classList.remove('active');
                });
                const step = document.getElementById('step' + n);
                step.style.display = 'block';
                step.classList.add('active');
                
                if (n === 3) {
                    const name = document.getElementById('name').value;
                    const email = document.getElementById('email').value;
                    document.getElementById('summary').textContent = 
                        `Name: ${name}, Email: ${email}`;
                }
            }
            function submitForm() {
                document.getElementById('result').textContent = 'Form submitted!';
            }
        </script>
    """)
    
    wizard = WizardHandler(page)
    
    # Step 1
    step_info = wizard.detect_current_step()
    print(f"Current step fields: {[f['label'] for f in step_info['visibleFields']]}")
    wizard.fill_current_step({"Name": "Alice Johnson"})
    wizard.go_next()
    
    # Step 2
    step_info = wizard.detect_current_step()
    print(f"Current step fields: {[f['label'] for f in step_info['visibleFields']]}")
    wizard.fill_current_step({"Email": "alice@example.com"})
    wizard.go_next()
    
    # Step 3: Confirm and submit
    summary = page.locator("#summary").inner_text()
    print(f"Summary: {summary}")
    wizard.submit()
    
    result = page.locator("#result").inner_text()
    print(f"Result: {result}")
    
    browser.close()
```

**Output:**
```
Current step fields: ['Name']
Current step fields: ['Email']
Summary: Name: Alice Johnson, Email: alice@example.com
Result: Form submitted!
```

---

## AI-Powered Form Understanding

For unfamiliar forms, an LLM can analyze the form structure and suggest how to fill it:

```python
import anthropic
import json

def ai_form_filler(form_fields: list, user_data: dict) -> dict:
    """
    Use an LLM to map user data to form fields.
    
    Args:
        form_fields: List of field descriptions from map_form_fields()
        user_data: Dict of user information (name, email, etc.)
    
    Returns:
        Dict mapping field labels/IDs to values
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Given these form fields and user data, determine the correct value for each field.

Form fields:
{json.dumps(form_fields, indent=2)}

User data:
{json.dumps(user_data, indent=2)}

Return a JSON object mapping each field's "label" to the value that should be entered.
For select fields, use the option's "value" attribute.
For checkboxes, use true/false.
Skip fields that don't match any user data.
Return ONLY JSON."""
        }]
    )
    
    return json.loads(response.content[0].text)

# Example
form_fields = [
    {"type": "text", "label": "Full Name", "required": True},
    {"type": "email", "label": "Email Address", "required": True},
    {"type": "select-one", "label": "Country", "options": [
        {"value": "us", "text": "United States"},
        {"value": "uk", "text": "United Kingdom"}
    ]},
    {"type": "checkbox", "label": "Subscribe to updates", "required": False}
]

user_data = {
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "country": "United States",
    "wants_newsletter": True
}

# mapping = ai_form_filler(form_fields, user_data)
# Expected:
expected_mapping = {
    "Full Name": "Alice Johnson",
    "Email Address": "alice@example.com",
    "Country": "us",
    "Subscribe to updates": True
}
print(json.dumps(expected_mapping, indent=2))
```

**Output:**
```json
{
  "Full Name": "Alice Johnson",
  "Email Address": "alice@example.com",
  "Country": "us",
  "Subscribe to updates": true
}
```

> **🤖 AI Context:** AI-powered form mapping is especially useful when an agent encounters forms it hasn't seen before. The LLM understands the semantic meaning of field labels ("Full Name" matches "name" in user data) and handles variations gracefully.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `get_by_label()` for field selection | Matches user experience and validates accessibility |
| Map form fields before filling | Understand the form structure to fill correctly |
| Check for validation errors after submission | Detect and fix errors automatically |
| Handle multi-step forms with state tracking | Know which step you're on and which fields are visible |
| Set realistic values for number/date fields | Min/max constraints cause silent failures |
| Wait after submission for confirmation | Pages may redirect or show async results |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `fill()` on checkboxes | Use `check()` and `uncheck()` instead |
| Not verifying submission success | Check for confirmation messages or URL changes |
| Filling hidden fields | Check `is_visible()` before filling |
| Ignoring `required` attributes | Ensure all required fields have values |
| Using `select_option(value)` with visible text | Use `select_option(label="text")` for display text |
| Not handling file upload separately | Use `set_input_files()` — `fill()` doesn't work for files |

---

## Hands-on Exercise

### Your Task

Build an `AutoFormFiller` class that automatically detects and fills a registration form.

### Requirements
1. Create an `AutoFormFiller` class that takes a Playwright page
2. Implement `scan_form()` — detect all form fields and their types
3. Implement `fill(data)` — fill the form with a dict of label-value pairs, handling text, email, select, checkbox, and textarea
4. Implement `submit()` — click the submit button and check for errors
5. Implement `get_result()` — return success/failure with any error messages
6. Test with a registration form that has name, email, country dropdown, terms checkbox, and submit button

### Expected Result
The form is auto-detected, filled with provided data, submitted, and the result reports success or failure.

<details>
<summary>💡 Hints (click to expand)</summary>

- `page.evaluate()` with `querySelectorAll` to find all form elements
- Check `el.tagName` and `el.type` to determine how to fill each field
- Use `get_by_label(label).fill()` for text inputs
- Use `get_by_label(label).select_option()` for dropdowns
- Use `get_by_label(label).check()` for checkboxes
- After submit, look for `.error` or `[role="alert"]` elements

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from playwright.sync_api import sync_playwright

class AutoFormFiller:
    """Automatically detect and fill web forms."""
    
    def __init__(self, page):
        self.page = page
        self._fields = []
        self._errors = []
    
    def scan_form(self) -> list:
        """Detect all form fields."""
        self._fields = self.page.evaluate("""
            () => {
                const fields = [];
                document.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.type === 'hidden' || el.type === 'submit') return;
                    
                    let label = '';
                    if (el.id) {
                        const lbl = document.querySelector(`label[for="${el.id}"]`);
                        if (lbl) label = lbl.textContent.trim();
                    }
                    if (!label && el.closest('label')) {
                        label = el.closest('label').textContent.trim();
                    }
                    
                    fields.push({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || 'text',
                        label: label || el.placeholder || el.name || 'unlabeled',
                        required: el.required,
                        visible: el.offsetParent !== null
                    });
                });
                return fields;
            }
        """)
        return self._fields
    
    def fill(self, data: dict):
        """Fill form fields by label."""
        for label, value in data.items():
            field = self.page.get_by_label(label)
            if field.count() == 0:
                continue
            
            tag = field.evaluate("el => el.tagName.toLowerCase()")
            input_type = field.evaluate("el => el.type || 'text'")
            
            if input_type == "checkbox":
                if value:
                    field.check()
                else:
                    field.uncheck()
            elif tag == "select":
                if isinstance(value, str) and len(value) <= 3:
                    field.select_option(value)  # Assume short strings are values
                else:
                    field.select_option(label=str(value))
            else:
                field.fill(str(value))
    
    def submit(self) -> bool:
        """Click submit and return success status."""
        btn = self.page.get_by_role("button", name="Register")
        if btn.count() == 0:
            btn = self.page.get_by_role("button", name="Submit")
        if btn.count() == 0:
            btn = self.page.locator("[type='submit']")
        
        btn.click()
        self.page.wait_for_timeout(500)
        return True
    
    def get_result(self) -> dict:
        """Check for errors or success messages."""
        errors = self.page.locator(
            ".error, .error-message, [role='alert']"
        ).all_inner_texts()
        errors = [e.strip() for e in errors if e.strip()]
        
        success = self.page.locator(
            ".success, .confirmation, #result"
        ).all_inner_texts()
        success = [s.strip() for s in success if s.strip()]
        
        return {
            "success": len(errors) == 0 and len(success) > 0,
            "errors": errors,
            "messages": success
        }

# Test
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="name">Full Name</label>
            <input id="name" type="text" required />
            
            <label for="email">Email</label>
            <input id="email" type="email" required />
            
            <label for="country">Country</label>
            <select id="country">
                <option value="">Select</option>
                <option value="us">United States</option>
                <option value="uk">United Kingdom</option>
            </select>
            
            <label><input type="checkbox" id="terms" /> Accept Terms</label>
            
            <button type="button" onclick="
                const name = document.getElementById('name').value;
                const email = document.getElementById('email').value;
                if (name && email) {
                    document.getElementById('result').textContent = 'Registration complete for ' + name;
                } else {
                    document.getElementById('result').textContent = '';
                }
            ">Register</button>
            <div id="result" class="success"></div>
        </form>
    """)
    
    filler = AutoFormFiller(page)
    
    # Scan
    fields = filler.scan_form()
    print(f"Fields found: {len(fields)}")
    for f in fields:
        print(f"  {f['type']}: {f['label']}")
    
    # Fill
    filler.fill({
        "Full Name": "Alice Johnson",
        "Email": "alice@example.com",
        "Country": "United States",
        "Accept Terms": True
    })
    
    # Submit
    filler.submit()
    
    # Check result
    result = filler.get_result()
    print(f"\nResult: {result}")
    
    browser.close()
```

**Output:**
```
Fields found: 4
  text: Full Name
  email: Email
  select-one: Country
  checkbox: Accept Terms

Result: {'success': True, 'errors': [], 'messages': ['Registration complete for Alice Johnson']}
```

</details>

### Bonus Challenges
- [ ] Add AI-powered field mapping: when `fill()` can't match a label, use an LLM to find the best match
- [ ] Implement date picker handling (detect date inputs and set values)
- [ ] Add support for drag-and-drop file upload zones (not just `<input type="file">`)

---

## Summary

✅ Map form fields before filling — detect types, labels, and required status to fill correctly

✅ Use `fill()` for text, `select_option()` for dropdowns, `check()`/`uncheck()` for checkboxes, and `set_input_files()` for uploads

✅ Always check for validation errors after submission and retry with corrections

✅ Multi-step forms require tracking which step is active and which fields are visible

✅ AI-powered form mapping helps agents fill unfamiliar forms by matching user data to field labels semantically

**Next:** [Testing with AI Agents](./07-testing-with-ai-agents.md)

**Previous:** [Web Scraping with AI](./05-web-scraping-with-ai.md)

---

## Further Reading

- [Playwright Input Actions](https://playwright.dev/python/docs/input) - Complete form interaction reference
- [Playwright Locators](https://playwright.dev/python/docs/locators) - Finding form elements
- [MDN HTML Forms](https://developer.mozilla.org/en-US/docs/Learn/Forms) - Form element reference

<!-- 
Sources Consulted:
- Playwright Input/Actions: https://playwright.dev/python/docs/input
- Playwright Locators: https://playwright.dev/python/docs/locators
- MDN Forms: https://developer.mozilla.org/en-US/docs/Learn/Forms
-->
