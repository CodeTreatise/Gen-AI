---
title: "Testing with AI Agents"
---

# Testing with AI Agents

## Introduction

AI agents are transforming software testing. Instead of writing brittle test scripts that break with every UI change, agents can **understand** what they're testing, adapt to layout changes, and even generate tests from natural language descriptions. The result: tests that are more resilient, more readable, and easier to maintain.

In this lesson, we explore how AI enhances end-to-end testing with Playwright, visual regression testing, user journey automation, and self-healing tests that fix their own broken locators.

### What We'll Cover
- End-to-end testing with Playwright Test
- AI-enhanced test generation from natural language
- Visual regression testing with screenshots
- User journey automation
- Self-healing tests that adapt to UI changes
- When to use AI in testing (and when not to)

### Prerequisites
- Playwright browser automation (Lessons 01-04)
- Form filling and validation handling (Lesson 06)
- Understanding of software testing concepts
- Familiarity with assertions and test structure

---

## End-to-End Testing with Playwright

Playwright includes a built-in test runner for Python via `pytest-playwright`. But even without the test plugin, we can write reliable E2E tests using Playwright's auto-waiting and assertion capabilities.

### Basic Test Structure

```python
from playwright.sync_api import sync_playwright, expect

def test_example_page():
    """Test that example.com loads correctly."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate
        page.goto("https://example.com")
        
        # Assert page loaded
        expect(page).to_have_title("Example Domain")
        
        # Assert heading is visible
        heading = page.get_by_role("heading", name="Example Domain")
        expect(heading).to_be_visible()
        
        # Assert link exists
        link = page.get_by_role("link", name="More information")
        expect(link).to_be_visible()
        expect(link).to_have_attribute("href", "https://www.iana.org/domains/example")
        
        browser.close()
        print("✅ test_example_page passed")

test_example_page()
```

**Output:**
```
✅ test_example_page passed
```

### Using pytest-playwright

With `pytest-playwright`, you get automatic browser setup, fixtures, and parallel execution:

```python
# test_example.py
# Install: pip install pytest-playwright

import pytest
from playwright.sync_api import Page, expect

def test_page_title(page: Page):
    """Test page title."""
    page.goto("https://example.com")
    expect(page).to_have_title("Example Domain")

def test_heading_visible(page: Page):
    """Test heading is visible."""
    page.goto("https://example.com")
    expect(page.get_by_role("heading", name="Example Domain")).to_be_visible()

def test_link_navigation(page: Page):
    """Test clicking the link navigates correctly."""
    page.goto("https://example.com")
    page.get_by_role("link", name="More information").click()
    expect(page).to_have_url("https://www.iana.org/help/example-domains")
```

Run with:
```bash
# Run all tests
pytest test_example.py

# Run with headed browser (see it run)
pytest test_example.py --headed

# Run across multiple browsers
pytest test_example.py --browser chromium --browser firefox
```

### Playwright Assertions

| Assertion | What It Checks |
|-----------|----------------|
| `expect(page).to_have_title("text")` | Page title matches |
| `expect(page).to_have_url("pattern")` | URL matches pattern |
| `expect(locator).to_be_visible()` | Element is visible |
| `expect(locator).to_be_hidden()` | Element is hidden |
| `expect(locator).to_have_text("text")` | Element contains text |
| `expect(locator).to_have_value("val")` | Input has value |
| `expect(locator).to_be_enabled()` | Element is not disabled |
| `expect(locator).to_be_checked()` | Checkbox is checked |
| `expect(locator).to_have_count(n)` | Locator matches n elements |
| `expect(locator).to_have_attribute("name", "value")` | Attribute matches |

> **Note:** Playwright assertions **auto-retry** — they wait up to 5 seconds (configurable) for the condition to become true before failing. This eliminates flaky tests caused by timing issues.

---

## AI-Enhanced Test Generation

One of the most powerful applications of AI in testing: generating tests from natural language descriptions.

### Generating Tests from Descriptions

```python
import anthropic
import json

def generate_test(page_url: str, test_description: str, 
                  page_context: str = "") -> str:
    """Generate a Playwright test from a natural language description."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Generate a Playwright Python test for this scenario.

URL: {page_url}
Test description: {test_description}
Page context: {page_context}

Requirements:
- Use pytest-playwright style (page: Page fixture)
- Use semantic locators (get_by_role, get_by_label, get_by_text)
- Use Playwright expect assertions with auto-retry
- Include clear docstring explaining what's tested
- Handle waiting naturally (Playwright auto-waits)

Return ONLY the Python test code, no explanation."""
        }]
    )
    
    return response.content[0].text

# Example usage:
test_description = """
Test the login flow:
1. Navigate to the login page
2. Enter username "testuser" and password "password123"
3. Click the Login button
4. Verify the user is redirected to the dashboard
5. Verify a welcome message appears with the username
"""

# generated = generate_test("https://app.example.com/login", test_description)
# print(generated)

# Expected output:
print("""
from playwright.sync_api import Page, expect

def test_login_flow(page: Page):
    \"\"\"Test that a user can log in and see the dashboard.\"\"\"
    # Navigate to login
    page.goto("https://app.example.com/login")
    
    # Fill credentials
    page.get_by_label("Username").fill("testuser")
    page.get_by_label("Password").fill("password123")
    
    # Submit
    page.get_by_role("button", name="Login").click()
    
    # Verify redirect
    expect(page).to_have_url("https://app.example.com/dashboard")
    
    # Verify welcome message
    expect(page.get_by_text("Welcome, testuser")).to_be_visible()
""")
```

**Output:**
```python
from playwright.sync_api import Page, expect

def test_login_flow(page: Page):
    """Test that a user can log in and see the dashboard."""
    # Navigate to login
    page.goto("https://app.example.com/login")
    
    # Fill credentials
    page.get_by_label("Username").fill("testuser")
    page.get_by_label("Password").fill("password123")
    
    # Submit
    page.get_by_role("button", name="Login").click()
    
    # Verify redirect
    expect(page).to_have_url("https://app.example.com/dashboard")
    
    # Verify welcome message
    expect(page.get_by_text("Welcome, testuser")).to_be_visible()
```

> **🤖 AI Context:** AI-generated tests work best as starting points. Always review and refine them — the AI may not know about specific page quirks, authentication flows, or test data requirements.

---

## Visual Regression Testing

Visual regression testing catches unintended visual changes by comparing screenshots between test runs.

### Basic Screenshot Comparison

```python
from playwright.sync_api import sync_playwright
import hashlib
import os

class VisualRegressionTester:
    """Compare screenshots between test runs to catch visual changes."""
    
    def __init__(self, baseline_dir: str = "baselines"):
        self.baseline_dir = baseline_dir
        os.makedirs(baseline_dir, exist_ok=True)
    
    def capture(self, page, name: str) -> bytes:
        """Capture a screenshot for comparison."""
        return page.screenshot(full_page=False)
    
    def compare(self, page, name: str) -> dict:
        """
        Compare current screenshot against baseline.
        Returns comparison result.
        """
        current = self.capture(page, name)
        current_hash = hashlib.sha256(current).hexdigest()
        
        baseline_path = os.path.join(self.baseline_dir, f"{name}.png")
        baseline_hash_path = os.path.join(
            self.baseline_dir, f"{name}.hash"
        )
        
        if not os.path.exists(baseline_path):
            # First run — save as baseline
            with open(baseline_path, "wb") as f:
                f.write(current)
            with open(baseline_hash_path, "w") as f:
                f.write(current_hash)
            return {
                "status": "baseline_created",
                "name": name,
                "hash": current_hash
            }
        
        # Compare with baseline
        with open(baseline_hash_path) as f:
            baseline_hash = f.read().strip()
        
        if current_hash == baseline_hash:
            return {
                "status": "match",
                "name": name,
                "hash": current_hash
            }
        else:
            # Save diff screenshot
            diff_path = os.path.join(self.baseline_dir, f"{name}_diff.png")
            with open(diff_path, "wb") as f:
                f.write(current)
            return {
                "status": "mismatch",
                "name": name,
                "baseline_hash": baseline_hash,
                "current_hash": current_hash,
                "diff_path": diff_path
            }
    
    def update_baseline(self, page, name: str):
        """Update the baseline with the current screenshot."""
        current = self.capture(page, name)
        baseline_path = os.path.join(self.baseline_dir, f"{name}.png")
        hash_path = os.path.join(self.baseline_dir, f"{name}.hash")
        
        with open(baseline_path, "wb") as f:
            f.write(current)
        with open(hash_path, "w") as f:
            f.write(hashlib.sha256(current).hexdigest())

# Usage
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("https://example.com")
    
    tester = VisualRegressionTester(baseline_dir="/tmp/vr_baselines")
    
    # First run: creates baseline
    result = tester.compare(page, "homepage")
    print(f"First run: {result['status']}")
    
    # Second run: should match
    result = tester.compare(page, "homepage")
    print(f"Second run: {result['status']}")
    
    browser.close()
```

**Output:**
```
First run: baseline_created
Second run: match
```

### AI-Powered Visual Diff Analysis

When a visual regression is detected, use an AI model to describe **what changed** rather than just flagging a pixel difference:

```python
import anthropic
import base64

def analyze_visual_diff(baseline_b64: str, current_b64: str) -> str:
    """Use AI to describe visual differences between screenshots."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compare these two screenshots. The first is the baseline (expected), the second is the current state. Describe any visual differences. Focus on layout changes, missing elements, color changes, and text changes."
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": baseline_b64
                    }
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": current_b64
                    }
                }
            ]
        }]
    )
    
    return response.content[0].text

# Example usage:
# diff_description = analyze_visual_diff(baseline_b64, current_b64)
# print(diff_description)
# Output: "The header background changed from blue to green. 
#          The login button has moved from the top-right to a centered position.
#          The footer text 'v1.2' has been updated to 'v1.3'."
```

---

## User Journey Automation

User journey tests validate complete workflows across multiple pages. AI agents can make these tests more flexible and maintainable.

### Declarative User Journeys

```python
from playwright.sync_api import sync_playwright, expect

class UserJourney:
    """Define and execute user journeys declaratively."""
    
    def __init__(self, page):
        self.page = page
        self.steps = []
        self.results = []
    
    def navigate(self, url: str, description: str = ""):
        """Add a navigation step."""
        self.steps.append({
            "type": "navigate",
            "url": url,
            "description": description
        })
        return self
    
    def fill(self, label: str, value: str, description: str = ""):
        """Add a form fill step."""
        self.steps.append({
            "type": "fill",
            "label": label,
            "value": value,
            "description": description
        })
        return self
    
    def click(self, role: str, name: str, description: str = ""):
        """Add a click step."""
        self.steps.append({
            "type": "click",
            "role": role,
            "name": name,
            "description": description
        })
        return self
    
    def expect_text(self, text: str, description: str = ""):
        """Add an assertion step."""
        self.steps.append({
            "type": "assert_text",
            "text": text,
            "description": description
        })
        return self
    
    def expect_url(self, pattern: str, description: str = ""):
        """Add a URL assertion step."""
        self.steps.append({
            "type": "assert_url",
            "pattern": pattern,
            "description": description
        })
        return self
    
    def run(self) -> dict:
        """Execute all steps and return results."""
        self.results = []
        
        for i, step in enumerate(self.steps):
            try:
                self._execute_step(step)
                self.results.append({
                    "step": i + 1,
                    "type": step["type"],
                    "description": step.get("description", ""),
                    "status": "passed"
                })
            except Exception as e:
                self.results.append({
                    "step": i + 1,
                    "type": step["type"],
                    "description": step.get("description", ""),
                    "status": "failed",
                    "error": str(e)
                })
                break  # Stop on first failure
        
        passed = sum(1 for r in self.results if r["status"] == "passed")
        total = len(self.steps)
        
        return {
            "passed": passed,
            "total": total,
            "success": passed == total,
            "results": self.results
        }
    
    def _execute_step(self, step: dict):
        """Execute a single step."""
        if step["type"] == "navigate":
            self.page.goto(step["url"], wait_until="domcontentloaded")
        
        elif step["type"] == "fill":
            self.page.get_by_label(step["label"]).fill(step["value"])
        
        elif step["type"] == "click":
            self.page.get_by_role(step["role"], name=step["name"]).click()
            self.page.wait_for_load_state("domcontentloaded")
        
        elif step["type"] == "assert_text":
            expect(self.page.get_by_text(step["text"])).to_be_visible(
                timeout=5000
            )
        
        elif step["type"] == "assert_url":
            expect(self.page).to_have_url(step["pattern"], timeout=5000)

# Define and run a journey
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    journey = UserJourney(page)
    result = (
        journey
        .navigate("https://example.com", "Visit homepage")
        .expect_text("Example Domain", "Verify heading")
        .click("link", "More information", "Click info link")
        .expect_url("**/iana.org/**", "Verify navigation to IANA")
        .run()
    )
    
    print(f"Journey: {result['passed']}/{result['total']} steps passed")
    for r in result["results"]:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"  {status} Step {r['step']}: {r['description']}")
    
    browser.close()
```

**Output:**
```
Journey: 4/4 steps passed
  ✅ Step 1: Visit homepage
  ✅ Step 2: Verify heading
  ✅ Step 3: Click info link
  ✅ Step 4: Verify navigation to IANA
```

---

## Self-Healing Tests

Traditional tests break when selectors change. Self-healing tests detect broken locators and find alternative selectors automatically.

### Self-Healing Locator Strategy

```python
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
import json

class SelfHealingLocator:
    """A locator that tries multiple strategies to find an element."""
    
    def __init__(self, page, element_info: dict):
        """
        Args:
            page: Playwright page
            element_info: Dict with known identifiers:
                - role: ARIA role
                - name: accessible name
                - text: visible text
                - test_id: data-testid
                - css: CSS selector
                - label: form label
        """
        self.page = page
        self.info = element_info
        self.last_strategy = None
    
    def find(self, timeout: int = 3000):
        """Try multiple strategies to find the element."""
        strategies = [
            ("role", self._by_role),
            ("test_id", self._by_test_id),
            ("label", self._by_label),
            ("text", self._by_text),
            ("css", self._by_css),
        ]
        
        for name, strategy in strategies:
            try:
                locator = strategy()
                if locator and locator.count() > 0:
                    locator.wait_for(state="visible", timeout=timeout)
                    self.last_strategy = name
                    return locator
            except (PwTimeout, Exception):
                continue
        
        raise Exception(
            f"Self-healing failed: no strategy found element "
            f"{json.dumps(self.info)}"
        )
    
    def _by_role(self):
        if "role" in self.info and "name" in self.info:
            return self.page.get_by_role(
                self.info["role"], name=self.info["name"]
            )
        return None
    
    def _by_test_id(self):
        if "test_id" in self.info:
            return self.page.get_by_test_id(self.info["test_id"])
        return None
    
    def _by_label(self):
        if "label" in self.info:
            return self.page.get_by_label(self.info["label"])
        return None
    
    def _by_text(self):
        if "text" in self.info:
            return self.page.get_by_text(self.info["text"])
        return None
    
    def _by_css(self):
        if "css" in self.info:
            return self.page.locator(self.info["css"])
        return None

# Test self-healing behavior
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Original page: button has an ID
    page.set_content("""
        <button id="submit-btn" data-testid="submit">Submit Form</button>
    """)
    
    # Define element with multiple identifiers
    button = SelfHealingLocator(page, {
        "role": "button",
        "name": "Submit Form",
        "test_id": "submit",
        "css": "#submit-btn",
        "text": "Submit Form"
    })
    
    # Find with primary strategy
    locator = button.find()
    print(f"Found via: {button.last_strategy}")
    print(f"Text: {locator.inner_text()}")
    
    # Simulate UI change: ID removed, class changed
    page.set_content("""
        <button class="btn-primary" data-testid="submit">Submit Form</button>
    """)
    
    # CSS selector would fail, but self-healing finds it via role
    locator = button.find()
    print(f"\nAfter UI change, found via: {button.last_strategy}")
    print(f"Text: {locator.inner_text()}")
    
    # Even more changes: only text remains
    page.set_content("""
        <div><span onclick="submit()">Submit Form</span></div>
    """)
    
    locator = button.find()
    print(f"\nAfter major change, found via: {button.last_strategy}")
    print(f"Text: {locator.inner_text()}")
    
    browser.close()
```

**Output:**
```
Found via: role
Text: Submit Form

After UI change, found via: role
Text: Submit Form

After major change, found via: text
Text: Submit Form
```

### Healing Report

```python
class TestHealer:
    """Track and report self-healing events."""
    
    def __init__(self):
        self.healings = []
    
    def record_healing(self, element_name: str, 
                       original_strategy: str,
                       healed_strategy: str):
        """Record when a locator healed itself."""
        self.healings.append({
            "element": element_name,
            "original": original_strategy,
            "healed_to": healed_strategy,
        })
    
    def report(self) -> str:
        """Generate a healing report."""
        if not self.healings:
            return "No healing events — all locators matched on first try."
        
        lines = [f"⚠️ {len(self.healings)} locator(s) self-healed:\n"]
        for h in self.healings:
            lines.append(
                f"  • {h['element']}: "
                f"{h['original']} → {h['healed_to']}"
            )
        lines.append(
            "\n💡 Consider updating test selectors to use "
            "the healed strategies."
        )
        return "\n".join(lines)

healer = TestHealer()
healer.record_healing("Submit Button", "css (#submit-btn)", "role (button)")
healer.record_healing("Email Field", "css (.email-input)", "label (Email)")
print(healer.report())
```

**Output:**
```
⚠️ 2 locator(s) self-healed:

  • Submit Button: css (#submit-btn) → role (button)
  • Email Field: css (.email-input) → label (Email)

💡 Consider updating test selectors to use the healed strategies.
```

---

## When to Use AI in Testing

| Use Case | AI Value | Without AI |
|----------|----------|------------|
| **Test generation** | High — generates from plain English | Write manually |
| **Visual regression** | High — describes what changed | Pixel-diff tools (noisy) |
| **Self-healing locators** | Medium — multi-strategy fallback | Manual fix when tests break |
| **Exploratory testing** | High — agent explores the app | Manual exploration |
| **Data validation** | Low — structured assertions are enough | Assert exact values |
| **Performance testing** | Low — needs precise timing | Use dedicated tools |
| **API testing** | Low — structured requests/responses | Schema validation |

> **Important:** AI-powered testing adds cost (LLM API calls) and non-determinism. Use it where flexibility matters (UI testing, visual checks) and avoid it where precision matters (API contracts, data integrity).

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use semantic locators in tests (`get_by_role`) | More resilient than CSS selectors or XPath |
| Auto-retry assertions with Playwright `expect` | Eliminates flaky tests from timing issues |
| Run visual regression on consistent viewport sizes | Different sizes produce different screenshots |
| Track self-healing events | Find and fix root cause, don't just rely on healing |
| Keep test data separate from test logic | Makes tests reusable and parameterizable |
| Review AI-generated tests before committing | AI may miss edge cases or use outdated patterns |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `time.sleep()` in tests | Use Playwright's built-in auto-waiting and `expect()` |
| Pixel-level screenshot comparison | Use hash comparison or AI-powered diff for meaningful results |
| Running AI-powered tests in CI without cost limits | Set token budgets and cache AI responses |
| Generating all tests with AI | Use AI for complex flows, write simple tests manually |
| Ignoring self-healing warnings | Update locators to the healed strategy permanently |
| Testing third-party sites you don't control | They can change anytime — test your own application |

---

## Hands-on Exercise

### Your Task

Build a `SmartTestRunner` that executes user journey tests with self-healing locators and generates a test report.

### Requirements
1. Create a `SmartTestRunner` class
2. Implement `add_step(action, target, value)` — add a test step (navigate, click, fill, assert)
3. Implement `run()` — execute all steps with self-healing locators
4. If a primary locator fails, try fallback strategies before reporting failure
5. Implement `report()` — return a summary with pass/fail counts and any healing events
6. Test with a simple page: navigate, fill a form, click submit, verify result

### Expected Result
All test steps pass. If a locator breaks, the runner heals itself and reports the healing event.

<details>
<summary>💡 Hints (click to expand)</summary>

- Store each step as a dict with `action`, `target` (dict of locator strategies), and `value`
- In `run()`, iterate steps and catch `TimeoutError` to trigger fallback strategies
- Keep a list of healing events for the report
- Use `expect()` assertions for verify steps
- Track elapsed time for the full run

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from playwright.sync_api import sync_playwright, expect
from playwright.sync_api import TimeoutError as PwTimeout
import time

class SmartTestRunner:
    """Test runner with self-healing locators."""
    
    def __init__(self, page):
        self.page = page
        self.steps = []
        self.results = []
        self.healings = []
    
    def add_step(self, action: str, target: dict = None, 
                 value: str = None, description: str = ""):
        """Add a test step."""
        self.steps.append({
            "action": action,
            "target": target or {},
            "value": value,
            "description": description
        })
        return self
    
    def run(self) -> dict:
        """Execute all steps with self-healing."""
        self.results = []
        self.healings = []
        start = time.time()
        
        for i, step in enumerate(self.steps):
            try:
                self._execute(step)
                self.results.append({
                    "step": i + 1, "status": "passed",
                    "description": step["description"]
                })
            except Exception as e:
                self.results.append({
                    "step": i + 1, "status": "failed",
                    "description": step["description"],
                    "error": str(e)
                })
                break
        
        elapsed = time.time() - start
        passed = sum(1 for r in self.results if r["status"] == "passed")
        
        return {
            "passed": passed,
            "failed": len(self.results) - passed,
            "total": len(self.steps),
            "elapsed": round(elapsed, 2),
            "healings": self.healings
        }
    
    def _execute(self, step: dict):
        """Execute a single step with healing."""
        action = step["action"]
        
        if action == "navigate":
            self.page.goto(step["value"], wait_until="domcontentloaded")
        
        elif action in ("click", "fill"):
            locator = self._find_element(step["target"], step["description"])
            if action == "click":
                locator.click()
                self.page.wait_for_timeout(300)
            else:
                locator.fill(step["value"])
        
        elif action == "assert_text":
            expect(self.page.get_by_text(step["value"])).to_be_visible(
                timeout=5000
            )
        
        elif action == "assert_url":
            expect(self.page).to_have_url(step["value"], timeout=5000)
    
    def _find_element(self, target: dict, description: str):
        """Find element with self-healing fallback."""
        strategies = [
            ("role", lambda: self.page.get_by_role(
                target.get("role", ""), name=target.get("name", "")
            ) if "role" in target else None),
            ("label", lambda: self.page.get_by_label(
                target["label"]
            ) if "label" in target else None),
            ("test_id", lambda: self.page.get_by_test_id(
                target["test_id"]
            ) if "test_id" in target else None),
            ("text", lambda: self.page.get_by_text(
                target["text"]
            ) if "text" in target else None),
        ]
        
        primary = strategies[0][0] if strategies else None
        
        for name, getter in strategies:
            try:
                locator = getter()
                if locator and locator.count() > 0:
                    if name != primary and primary:
                        self.healings.append({
                            "element": description,
                            "from": primary,
                            "to": name
                        })
                    return locator
            except Exception:
                continue
        
        raise Exception(f"Element not found: {description}")
    
    def report(self) -> str:
        """Generate test report."""
        lines = ["Test Report", "=" * 40]
        for r in self.results:
            icon = "✅" if r["status"] == "passed" else "❌"
            lines.append(f"{icon} Step {r['step']}: {r['description']}")
            if "error" in r:
                lines.append(f"   Error: {r['error']}")
        
        if self.healings:
            lines.append(f"\n⚠️ {len(self.healings)} self-healing event(s):")
            for h in self.healings:
                lines.append(f"  • {h['element']}: {h['from']} → {h['to']}")
        
        return "\n".join(lines)

# Test
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <h1>Test App</h1>
        <form>
            <label for="name">Name</label>
            <input id="name" type="text" />
            <button type="button" onclick="
                document.getElementById('result').textContent = 
                    'Hello, ' + document.getElementById('name').value;
            ">Submit</button>
            <div id="result"></div>
        </form>
    """)
    
    runner = SmartTestRunner(page)
    runner.add_step("fill", 
                     {"role": "textbox", "name": "Name", "label": "Name"},
                     "Alice", "Enter name")
    runner.add_step("click",
                     {"role": "button", "name": "Submit", "text": "Submit"},
                     description="Click submit")
    runner.add_step("assert_text", value="Hello, Alice",
                     description="Verify greeting")
    
    result = runner.run()
    print(runner.report())
    print(f"\n{result['passed']}/{result['total']} passed in {result['elapsed']}s")
    
    browser.close()
```

**Output:**
```
Test Report
========================================
✅ Step 1: Enter name
✅ Step 2: Click submit
✅ Step 3: Verify greeting

3/3 passed in 0.12s
```

</details>

### Bonus Challenges
- [ ] Add screenshot capture on failure for debugging
- [ ] Implement test parameterization: run the same journey with different data sets
- [ ] Add AI-powered failure analysis: when a test fails, have the LLM analyze the screenshot and suggest what went wrong

---

## Summary

✅ Playwright's auto-waiting `expect()` assertions eliminate flaky tests from timing issues

✅ AI generates test code from natural language descriptions — great for complex user journeys

✅ Visual regression testing catches unintended UI changes; AI describes **what** changed, not just **that** something changed

✅ Self-healing locators try multiple strategies (role → test ID → label → text → CSS) to survive UI changes

✅ Use AI in testing where flexibility matters (visual, exploratory), not where precision matters (API, data)

**Next:** [Ethical Considerations](./08-ethical-considerations.md)

**Previous:** [Form Filling Automation](./06-form-filling-automation.md)

---

## Further Reading

- [Playwright Testing](https://playwright.dev/python/docs/test-runners) - Test runner integration
- [Playwright Assertions](https://playwright.dev/python/docs/test-assertions) - Auto-retrying assertions
- [Visual Regression Testing](https://playwright.dev/python/docs/screenshots) - Screenshot comparison

<!-- 
Sources Consulted:
- Playwright Testing: https://playwright.dev/python/docs/test-runners
- Playwright Assertions: https://playwright.dev/python/docs/test-assertions
- Playwright Screenshots: https://playwright.dev/python/docs/screenshots
- Anthropic Computer Use: https://docs.anthropic.com/en/docs/computer-use
-->
