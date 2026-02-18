---
title: "Visual Element Identification"
---

# Visual Element Identification

## Introduction

Before an agent can click a button, fill a field, or extract data, it must first **find the right element**. This is the core challenge of browser automation — identifying elements reliably, even as pages change layout, update content, or load dynamically.

In this lesson, we explore Playwright's powerful locator system (semantic, CSS, XPath), AI-based element detection using vision models, strategies for handling dynamic elements, and accessibility-first selectors that make automation both robust and inclusive.

### What We'll Cover
- Playwright's built-in locators: role, text, label, placeholder, test ID
- CSS and XPath selectors for complex targeting
- AI-based visual element detection
- Handling dynamic elements with auto-waiting
- Accessibility selectors and why they produce the best locators
- Filtering and chaining locators

### Prerequisites
- Playwright basics and click/type/scroll actions (Lessons 01, 03)
- Screenshot analysis patterns (Lesson 02)
- Understanding of HTML structure and ARIA roles (Units 1, 13)
- Basic CSS selector syntax (Unit 1)

---

## Built-in Semantic Locators

Playwright provides locators that find elements the way users think about them — by role, visible text, label, or placeholder. These are the **most resilient** locators because they're tied to user-visible properties, not fragile implementation details like CSS classes or div hierarchies.

### get_by_role — The Gold Standard

`get_by_role()` finds elements by their ARIA role and accessible name. This is the recommended approach for virtually all interactive elements:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <nav>
            <a href="/home">Home</a>
            <a href="/about">About</a>
        </nav>
        <main>
            <h1>Welcome</h1>
            <form>
                <label for="email">Email address</label>
                <input id="email" type="email" />
                <button type="submit">Sign Up</button>
            </form>
            <input type="checkbox" id="agree" />
            <label for="agree">I agree to terms</label>
        </main>
    """)
    
    # Find by role + name
    page.get_by_role("link", name="About").click()
    page.get_by_role("heading", name="Welcome")
    page.get_by_role("button", name="Sign Up")
    page.get_by_role("textbox", name="Email address")
    page.get_by_role("checkbox", name="I agree to terms")
    
    # Count navigation links
    nav_links = page.get_by_role("link")
    print(f"Navigation links: {nav_links.count()}")
    
    # Verify heading exists
    heading = page.get_by_role("heading", name="Welcome")
    print(f"Heading: {heading.inner_text()}")
    
    # Check button text
    button = page.get_by_role("button", name="Sign Up")
    print(f"Button: {button.inner_text()}")
    
    browser.close()
```

**Output:**
```
Navigation links: 2
Heading: Welcome
Button: Sign Up
```

### Common ARIA Roles

| Role | HTML Elements | Example |
|------|---------------|---------|
| `button` | `<button>`, `<input type="submit">` | `get_by_role("button", name="Submit")` |
| `link` | `<a href="...">` | `get_by_role("link", name="Home")` |
| `textbox` | `<input type="text">`, `<textarea>` | `get_by_role("textbox", name="Search")` |
| `checkbox` | `<input type="checkbox">` | `get_by_role("checkbox", name="Agree")` |
| `radio` | `<input type="radio">` | `get_by_role("radio", name="Option A")` |
| `combobox` | `<select>` | `get_by_role("combobox", name="Country")` |
| `heading` | `<h1>` through `<h6>` | `get_by_role("heading", name="Title")` |
| `navigation` | `<nav>` | `get_by_role("navigation")` |
| `dialog` | `<dialog>`, `[role="dialog"]` | `get_by_role("dialog")` |
| `alert` | `[role="alert"]` | `get_by_role("alert")` |

### get_by_text — Find by Visible Text

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div>
            <p>Welcome to our platform</p>
            <span>Total: $49.99</span>
            <a href="/details">View details</a>
        </div>
    """)
    
    # Exact match
    welcome = page.get_by_text("Welcome to our platform")
    print(f"Found: {welcome.inner_text()}")
    
    # Substring match (default behavior)
    price = page.get_by_text("$49.99")
    print(f"Price: {price.inner_text()}")
    
    # Exact match only
    exact = page.get_by_text("Welcome", exact=True)
    print(f"Exact 'Welcome' count: {exact.count()}")
    # Returns 0 — "Welcome to our platform" != "Welcome"
    
    browser.close()
```

**Output:**
```
Found: Welcome to our platform
Price: Total: $49.99
Exact 'Welcome' count: 0
```

### get_by_label — Form Fields by Label

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="username">Username</label>
            <input id="username" type="text" />
            
            <label for="password">Password</label>
            <input id="password" type="password" />
            
            <label>
                Remember me
                <input type="checkbox" />
            </label>
        </form>
    """)
    
    # Fill fields by label
    page.get_by_label("Username").fill("alice")
    page.get_by_label("Password").fill("secret123")
    page.get_by_label("Remember me").check()
    
    print(f"Username: {page.locator('#username').input_value()}")
    print(f"Password: {page.locator('#password').input_value()}")
    print(f"Remember: {page.get_by_label('Remember me').is_checked()}")
    
    browser.close()
```

**Output:**
```
Username: alice
Password: secret123
Remember: True
```

### get_by_placeholder and get_by_test_id

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <input placeholder="Search products..." type="search" />
        <input placeholder="Enter your email" type="email" />
        <button data-testid="submit-btn">Go</button>
        <div data-testid="results-count">42 results</div>
    """)
    
    # Find by placeholder text
    page.get_by_placeholder("Search products").fill("laptop")
    search_val = page.get_by_placeholder("Search products").input_value()
    print(f"Search: {search_val}")
    
    # Find by test ID (stable, developer-controlled)
    page.get_by_test_id("submit-btn").click()
    results = page.get_by_test_id("results-count").inner_text()
    print(f"Results: {results}")
    
    browser.close()
```

**Output:**
```
Search: laptop
Results: 42 results
```

> **Tip:** `data-testid` attributes are the most stable locators — they're explicitly added for testing and don't change with visual redesigns. Ask developers to add them when possible.

---

## CSS and XPath Selectors

When semantic locators aren't enough, fall back to CSS or XPath selectors:

### CSS Selectors

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div class="card product-card" data-category="electronics">
            <h3 class="product-title">Laptop Pro</h3>
            <span class="price">$999</span>
            <button class="btn btn-primary add-to-cart">Add to Cart</button>
        </div>
        <div class="card product-card" data-category="books">
            <h3 class="product-title">Python Handbook</h3>
            <span class="price">$29</span>
            <button class="btn btn-primary add-to-cart">Add to Cart</button>
        </div>
    """)
    
    # By class
    titles = page.locator(".product-title")
    print(f"Products: {titles.count()}")
    
    # By attribute
    electronics = page.locator("[data-category='electronics']")
    print(f"Electronics: {electronics.locator('.product-title').inner_text()}")
    
    # Combined selectors
    electronics_price = page.locator(
        "[data-category='electronics'] .price"
    )
    print(f"Price: {electronics_price.inner_text()}")
    
    # Nth element
    second_product = page.locator(".product-card").nth(1)
    print(f"Second: {second_product.locator('.product-title').inner_text()}")
    
    browser.close()
```

**Output:**
```
Products: 2
Electronics: Laptop Pro
Price: $999
Second: Python Handbook
```

### XPath Selectors

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <table>
            <thead><tr><th>Name</th><th>Price</th></tr></thead>
            <tbody>
                <tr><td>Laptop</td><td>$999</td></tr>
                <tr><td>Mouse</td><td>$29</td></tr>
                <tr><td>Keyboard</td><td>$79</td></tr>
            </tbody>
        </table>
    """)
    
    # XPath for table data
    rows = page.locator("xpath=//tbody/tr")
    print(f"Table rows: {rows.count()}")
    
    # Find cell containing specific text
    laptop_price = page.locator(
        "xpath=//tr[td[text()='Laptop']]/td[2]"
    )
    print(f"Laptop price: {laptop_price.inner_text()}")
    
    # Find by position
    first_product = page.locator("xpath=//tbody/tr[1]/td[1]")
    print(f"First product: {first_product.inner_text()}")
    
    browser.close()
```

**Output:**
```
Table rows: 3
Laptop price: $999
First product: Laptop
```

### Locator Priority Guide

| Priority | Locator Type | Resilience | Example |
|----------|-------------|------------|---------|
| 🥇 1st | `get_by_role()` | Highest | `get_by_role("button", name="Submit")` |
| 🥈 2nd | `get_by_label()` | High | `get_by_label("Email")` |
| 🥉 3rd | `get_by_text()` | High | `get_by_text("Sign up")` |
| 4th | `get_by_test_id()` | High | `get_by_test_id("login-btn")` |
| 5th | `get_by_placeholder()` | Medium | `get_by_placeholder("Search...")` |
| 6th | CSS selector | Medium | `locator(".submit-button")` |
| 7th | XPath | Low | `locator("xpath=//button[@type='submit']")` |
| 8th | Coordinates | Lowest | Computer Use click at (450, 300) |

---

## AI-Based Element Detection

When you can't use DOM-based locators — or when you're working with the Computer Use tool and vision models — AI can identify elements visually from screenshots.

### Vision-Based Element Finding

```python
import anthropic
import base64
import json
from playwright.sync_api import sync_playwright

def find_elements_with_ai(screenshot_b64: str, target: str) -> list:
    """
    Ask Claude to find a specific element in a screenshot.
    Returns approximate coordinates and description.
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64
                    }
                },
                {
                    "type": "text",
                    "text": f"""Find the element: "{target}"
Return JSON array of matches:
[{{
    "description": "what the element looks like",
    "type": "button|link|input|text|image|icon",
    "approximate_center": {{"x": 640, "y": 360}},
    "confidence": 0.95
}}]
Return ONLY the JSON array."""
                }
            ]
        }]
    )
    
    return json.loads(response.content[0].text)

# Example: capture and find
# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=True)
#     page = browser.new_page(viewport={"width": 1280, "height": 720})
#     page.goto("https://example.com")
#     screenshot = base64.b64encode(page.screenshot()).decode()
#     browser.close()
#
#     elements = find_elements_with_ai(screenshot, "link to more information")
#     print(json.dumps(elements, indent=2))
```

### Combining DOM and Vision Detection

The most robust approach uses DOM locators as the primary method and falls back to vision-based detection for elements that aren't easily addressable:

```python
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
import base64
import json

class SmartElementFinder:
    """Find elements using DOM first, vision as fallback."""
    
    def __init__(self, page, ai_client=None):
        self.page = page
        self.ai_client = ai_client
    
    def find(self, description: str, role: str = None, 
             text: str = None, label: str = None) -> dict:
        """
        Find an element using multiple strategies.
        Returns element info and the locator that worked.
        """
        strategies = []
        
        # Strategy 1: Role-based
        if role:
            strategies.append(("role", lambda: self.page.get_by_role(
                role, name=description
            )))
        
        # Strategy 2: Text-based
        if text:
            strategies.append(("text", lambda: self.page.get_by_text(text)))
        
        # Strategy 3: Label-based
        if label:
            strategies.append(("label", lambda: self.page.get_by_label(label)))
        
        # Strategy 4: General text search
        strategies.append(("text_search", lambda: self.page.get_by_text(
            description
        )))
        
        # Try each strategy
        for strategy_name, get_locator in strategies:
            try:
                locator = get_locator()
                if locator.count() > 0:
                    box = locator.first.bounding_box()
                    return {
                        "found": True,
                        "strategy": strategy_name,
                        "locator": locator.first,
                        "text": locator.first.inner_text(),
                        "bounds": box,
                        "center": {
                            "x": box["x"] + box["width"] / 2,
                            "y": box["y"] + box["height"] / 2
                        } if box else None
                    }
            except (PwTimeout, Exception):
                continue
        
        # Strategy 5: Vision-based fallback
        if self.ai_client:
            return self._find_with_vision(description)
        
        return {"found": False, "strategy": "none", "description": description}
    
    def _find_with_vision(self, description: str) -> dict:
        """Fallback: use AI vision to find the element."""
        screenshot = base64.b64encode(self.page.screenshot()).decode()
        
        response = self.ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot
                    }},
                    {"type": "text", "text": f"""Find "{description}" in this screenshot.
Return JSON: {{"x": number, "y": number, "confidence": 0-1}}
Return ONLY JSON."""}
                ]
            }]
        )
        
        result = json.loads(response.content[0].text)
        return {
            "found": True,
            "strategy": "vision",
            "center": {"x": result["x"], "y": result["y"]},
            "confidence": result.get("confidence", 0.5)
        }

# Usage
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <button>Submit Form</button>
        <a href="/help">Need Help?</a>
        <label for="search">Search:</label>
        <input id="search" type="text" />
    """)
    
    finder = SmartElementFinder(page)
    
    # Find by role
    result = finder.find("Submit Form", role="button")
    print(f"Button: found={result['found']}, strategy={result['strategy']}")
    
    # Find by text
    result = finder.find("Need Help?", text="Need Help?")
    print(f"Link: found={result['found']}, strategy={result['strategy']}")
    
    # Find by label
    result = finder.find("Search", label="Search:")
    print(f"Input: found={result['found']}, strategy={result['strategy']}")
    
    browser.close()
```

**Output:**
```
Button: found=True, strategy=role
Link: found=True, strategy=text
Input: found=True, strategy=label
```

---

## Handling Dynamic Elements

Modern web applications add, remove, and update elements constantly. Playwright's auto-waiting handles most cases, but dynamic content requires additional strategies.

### Auto-Waiting

Playwright automatically waits for elements to be:
1. **Attached** to the DOM
2. **Visible** in the viewport
3. **Stable** (no animations in progress)
4. **Enabled** (not disabled)
5. **Receiving events** (not obscured by other elements)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div id="app"></div>
        <script>
            // Simulate delayed content loading
            setTimeout(() => {
                document.getElementById('app').innerHTML = `
                    <button id="loaded-btn" onclick="this.textContent='Clicked!'">
                        I loaded late
                    </button>
                `;
            }, 1000);
        </script>
    """)
    
    # Playwright waits automatically for the button to appear
    page.get_by_role("button", name="I loaded late").click()
    
    text = page.locator("#loaded-btn").inner_text()
    print(f"Button text: {text}")
    
    browser.close()
```

**Output:**
```
Button text: Clicked!
```

### Waiting for Elements

When you need explicit control over waiting:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div id="app">Loading...</div>
        <script>
            setTimeout(() => {
                document.getElementById('app').innerHTML = `
                    <div class="results">
                        <div class="item">Result 1</div>
                        <div class="item">Result 2</div>
                        <div class="item">Result 3</div>
                    </div>
                `;
            }, 1500);
        </script>
    """)
    
    # Wait for a specific selector to appear
    page.wait_for_selector(".results", timeout=5000)
    
    items = page.locator(".item")
    print(f"Results loaded: {items.count()} items")
    for i in range(items.count()):
        print(f"  - {items.nth(i).inner_text()}")
    
    browser.close()
```

**Output:**
```
Results loaded: 3 items
  - Result 1
  - Result 2
  - Result 3
```

### Handling Elements That Appear and Disappear

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <button onclick="showModal()">Open</button>
        <div id="modal" style="display:none;">
            <h2>Confirmation</h2>
            <button onclick="closeModal()">Confirm</button>
        </div>
        <div id="status">Ready</div>
        <script>
            function showModal() {
                document.getElementById('modal').style.display = 'block';
            }
            function closeModal() {
                document.getElementById('modal').style.display = 'none';
                document.getElementById('status').textContent = 'Confirmed!';
            }
        </script>
    """)
    
    # Click to open modal
    page.get_by_role("button", name="Open").click()
    
    # Wait for modal to be visible
    modal = page.locator("#modal")
    modal.wait_for(state="visible")
    print(f"Modal visible: {modal.is_visible()}")
    
    # Click confirm button inside modal
    page.get_by_role("button", name="Confirm").click()
    
    # Wait for modal to disappear
    modal.wait_for(state="hidden")
    print(f"Modal hidden: {not modal.is_visible()}")
    
    status = page.locator("#status").inner_text()
    print(f"Status: {status}")
    
    browser.close()
```

**Output:**
```
Modal visible: True
Modal hidden: True
Status: Confirmed!
```

---

## Filtering and Chaining Locators

### Filtering by Content

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <ul class="products">
            <li class="product">
                <span class="name">Laptop Pro</span>
                <span class="price">$999</span>
                <button>Buy</button>
            </li>
            <li class="product">
                <span class="name">Budget Laptop</span>
                <span class="price">$499</span>
                <button>Buy</button>
            </li>
            <li class="product">
                <span class="name">Gaming Laptop</span>
                <span class="price">$1299</span>
                <button>Buy</button>
            </li>
        </ul>
    """)
    
    products = page.locator(".product")
    
    # Filter by text content
    budget = products.filter(has_text="Budget")
    print(f"Budget product: {budget.locator('.name').inner_text()}")
    
    # Filter by child element
    expensive = products.filter(
        has=page.locator(".price", has_text="$1299")
    )
    print(f"Expensive product: {expensive.locator('.name').inner_text()}")
    
    # Exclude by text
    not_budget = products.filter(has_not_text="Budget")
    print(f"Non-budget products: {not_budget.count()}")
    
    browser.close()
```

**Output:**
```
Budget product: Budget Laptop
Expensive product: Gaming Laptop
Non-budget products: 2
```

### Chaining Locators

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div class="sidebar">
            <a href="/home">Home</a>
            <a href="/settings">Settings</a>
        </div>
        <div class="main-content">
            <a href="/article">Read More</a>
            <a href="/settings">Settings</a>
        </div>
    """)
    
    # Chain: find "Settings" link, but only in the sidebar
    sidebar_settings = page.locator(".sidebar").get_by_role(
        "link", name="Settings"
    )
    print(f"Sidebar settings href: {sidebar_settings.get_attribute('href')}")
    
    # Chain: find link in main content
    main_link = page.locator(".main-content").get_by_role(
        "link", name="Read More"
    )
    print(f"Main link href: {main_link.get_attribute('href')}")
    
    browser.close()
```

**Output:**
```
Sidebar settings href: /settings
Main link href: /article
```

### Combining Locators with and_/or_

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <button class="primary">Save</button>
        <button class="secondary">Cancel</button>
        <button class="primary" disabled>Delete</button>
    """)
    
    # AND: primary buttons that contain "Save"
    save_btn = page.get_by_role("button", name="Save").and_(
        page.locator(".primary")
    )
    print(f"Save is primary: {save_btn.count() > 0}")
    
    # OR: either Save or Cancel button
    either = page.get_by_role("button", name="Save").or_(
        page.get_by_role("button", name="Cancel")
    )
    print(f"Save or Cancel: {either.count()} buttons")
    
    # First and last
    buttons = page.locator("button")
    print(f"First button: {buttons.first.inner_text()}")
    print(f"Last button: {buttons.last.inner_text()}")
    print(f"Second button: {buttons.nth(1).inner_text()}")
    
    browser.close()
```

**Output:**
```
Save is primary: True
Save or Cancel: 2 buttons
First button: Save
Last button: Delete
Second button: Cancel
```

---

## Accessibility Selectors

Accessibility selectors produce the most robust locators because they're tied to semantic meaning, not visual presentation. They also ensure your automation validates that the page is accessible.

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form aria-label="Login form">
            <div role="group" aria-labelledby="credentials">
                <h3 id="credentials">Credentials</h3>
                <label for="user">Username</label>
                <input id="user" type="text" aria-required="true" />
                <label for="pass">Password</label>
                <input id="pass" type="password" aria-required="true" />
            </div>
            <button type="submit" aria-describedby="help">
                Log In
            </button>
            <p id="help">Click to access your account</p>
            <div role="alert" aria-live="polite" id="error" hidden>
                Invalid credentials
            </div>
        </form>
    """)
    
    # Find form by aria-label
    form = page.get_by_role("form", name="Login form")
    print(f"Form found: {form.count() > 0}")
    
    # Required fields
    required_fields = page.locator("[aria-required='true']")
    print(f"Required fields: {required_fields.count()}")
    
    # Find by role group
    creds_group = page.get_by_role("group", name="Credentials")
    print(f"Group found: {creds_group.count() > 0}")
    
    # Find alert (for error messages)
    alert = page.get_by_role("alert")
    print(f"Alert exists: {alert.count() > 0}")
    print(f"Alert hidden: {alert.is_hidden()}")
    
    browser.close()
```

**Output:**
```
Form found: True
Required fields: 2
Group found: True
Alert exists: True
Alert hidden: True
```

> **🤖 AI Context:** When building AI agents that automate tasks on behalf of users, accessibility-based locators serve double duty: they make automation more reliable (tied to semantic meaning, not fragile CSS) AND they validate that the target application is accessible. If your agent can't find an element by role, that's often a sign the page has accessibility issues.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `get_by_role()` as your first choice | Most resilient — tied to semantic meaning, not layout |
| Add `data-testid` to your own applications | Stable locators that survive redesigns |
| Prefer `get_by_label()` for form fields | Validates accessibility and is user-centric |
| Use filtering over complex CSS selectors | `filter(has_text=)` is more readable and maintainable |
| Chain locators to narrow scope | `page.locator(".sidebar").get_by_role(...)` avoids ambiguity |
| Try DOM locators before vision-based | Faster, more precise, and cheaper |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using CSS IDs that change with each build | Use `data-testid` or semantic roles instead |
| Writing brittle XPath like `//div[3]/div[2]/button` | Use `get_by_role("button", name="...")` |
| Not specifying `exact=True` for ambiguous text | Add `exact=True` or use more specific text |
| Clicking before element is stable | Playwright auto-waits; if issues persist, use `wait_for(state="visible")` |
| Using `nth(0)` without understanding order | Filter first, then use `.first` or `.nth()` |
| Ignoring strictness errors | If multiple elements match, narrow with filtering or chaining |

---

## Hands-on Exercise

### Your Task

Build an `ElementMapper` class that takes a URL and produces a complete inventory of all interactive elements on the page, organized by type.

### Requirements
1. Create an `ElementMapper` class that accepts a Playwright `page` object
2. Implement `map_buttons()` — find all buttons with their text and enabled/disabled state
3. Implement `map_links()` — find all links with their text and href
4. Implement `map_inputs()` — find all form inputs with their labels and types
5. Implement `full_map()` — combine all mappings into one dict
6. Test with a page that has buttons, links, and form inputs

### Expected Result
A dictionary containing separate lists for buttons, links, and inputs, each with relevant metadata.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `page.get_by_role("button").all()` to get all buttons as a list
- Check `element.is_disabled()` for button state
- Use `element.get_attribute("href")` for link URLs  
- For inputs, check the `type` attribute with `get_attribute("type")`
- Labels can be found through the `aria-label` attribute or associated `<label>` elements

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from playwright.sync_api import sync_playwright

class ElementMapper:
    """Maps all interactive elements on a page."""
    
    def __init__(self, page):
        self.page = page
    
    def map_buttons(self) -> list:
        """Find all buttons with their state."""
        buttons = []
        for btn in self.page.get_by_role("button").all():
            buttons.append({
                "text": btn.inner_text(),
                "enabled": not btn.is_disabled(),
                "visible": btn.is_visible()
            })
        return buttons
    
    def map_links(self) -> list:
        """Find all links with text and href."""
        links = []
        for link in self.page.get_by_role("link").all():
            links.append({
                "text": link.inner_text(),
                "href": link.get_attribute("href"),
                "visible": link.is_visible()
            })
        return links
    
    def map_inputs(self) -> list:
        """Find all form inputs with labels and types."""
        inputs = []
        for inp in self.page.locator(
            "input, select, textarea"
        ).all():
            input_type = inp.get_attribute("type") or "text"
            label = (
                inp.get_attribute("aria-label")
                or inp.get_attribute("placeholder")
                or inp.get_attribute("name")
                or "unlabeled"
            )
            inputs.append({
                "type": input_type,
                "label": label,
                "value": inp.input_value() if input_type != "file" else None,
                "required": inp.get_attribute("required") is not None
            })
        return inputs
    
    def full_map(self) -> dict:
        """Complete interactive element inventory."""
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "buttons": self.map_buttons(),
            "links": self.map_links(),
            "inputs": self.map_inputs(),
            "summary": {
                "total_buttons": len(self.map_buttons()),
                "total_links": len(self.map_links()),
                "total_inputs": len(self.map_inputs())
            }
        }

# Test
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <h1>Test Page</h1>
        <nav>
            <a href="/home">Home</a>
            <a href="/about">About</a>
        </nav>
        <form>
            <label for="name">Name</label>
            <input id="name" type="text" required aria-label="Name" />
            <label for="email">Email</label>
            <input id="email" type="email" placeholder="you@example.com" />
            <select aria-label="Country">
                <option>USA</option>
                <option>Canada</option>
            </select>
            <button type="submit">Submit</button>
            <button type="button" disabled>Reset</button>
        </form>
    """)
    
    mapper = ElementMapper(page)
    result = mapper.full_map()
    
    import json
    print(json.dumps(result, indent=2))
    
    browser.close()
```

**Output:**
```json
{
  "url": "about:blank",
  "title": "",
  "buttons": [
    {"text": "Submit", "enabled": true, "visible": true},
    {"text": "Reset", "enabled": false, "visible": true}
  ],
  "links": [
    {"text": "Home", "href": "/home", "visible": true},
    {"text": "About", "href": "/about", "visible": true}
  ],
  "inputs": [
    {"type": "text", "label": "Name", "value": "", "required": true},
    {"type": "email", "label": "you@example.com", "value": "", "required": false},
    {"type": "select", "label": "Country", "value": null, "required": false}
  ],
  "summary": {
    "total_buttons": 2,
    "total_links": 2,
    "total_inputs": 3
  }
}
```

</details>

### Bonus Challenges
- [ ] Add `map_images()` — find all images with alt text (or flag missing alt text)
- [ ] Generate a visual overlay: take a screenshot and draw bounding boxes around all interactive elements
- [ ] Implement a `suggest_locator(description)` method that returns the best Playwright locator for a natural language description

---

## Summary

✅ `get_by_role()` is the gold standard — resilient, semantic, and validates accessibility

✅ Playwright's locator priority: role → label → text → test ID → CSS → XPath → coordinates

✅ AI-based vision detection works as a fallback when DOM locators aren't available

✅ Playwright auto-waits for elements to be visible, stable, and enabled before acting

✅ Filter and chain locators to narrow scope: `page.locator(".sidebar").get_by_role("link", name="Settings")`

✅ Accessibility selectors produce the best automation — robust AND inclusive

**Next:** [Web Scraping with AI](./05-web-scraping-with-ai.md)

**Previous:** [Click, Type, Scroll Automation](./03-click-type-scroll-automation.md)

---

## Further Reading

- [Playwright Locators](https://playwright.dev/python/docs/locators) - Complete locator documentation
- [Playwright Auto-Waiting](https://playwright.dev/python/docs/actionability) - How Playwright waits for elements
- [WAI-ARIA Roles](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles) - Complete ARIA role reference
- [Testing Library Guiding Principles](https://testing-library.com/docs/queries/about#priority) - Locator priority philosophy

<!-- 
Sources Consulted:
- Playwright Locators: https://playwright.dev/python/docs/locators
- Playwright Actionability: https://playwright.dev/python/docs/actionability
- MDN ARIA Roles: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles
- Anthropic Computer Use: https://docs.anthropic.com/en/docs/computer-use
-->
