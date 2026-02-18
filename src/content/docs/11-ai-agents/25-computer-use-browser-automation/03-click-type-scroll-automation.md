---
title: "Click, Type, Scroll Automation"
---

# Click, Type, Scroll Automation

## Introduction

Every browser interaction comes down to three primitives: **clicking** elements, **typing** text, and **scrolling** to reveal content. Getting these right — with proper targeting, timing, and error handling — is the difference between a reliable automation agent and a fragile script that breaks on every page change.

In this lesson, we cover both DOM-based actions (Playwright — precise, fast, reliable) and vision-based actions (Anthropic Computer Use — flexible, works with any UI). We'll handle mouse clicks, text input, keyboard shortcuts, scrolling, and drag-and-drop.

### What We'll Cover
- Mouse click actions: single, double, right-click, hover
- Text input: filling fields, typing character-by-character, clearing
- Keyboard shortcuts and special key presses
- Scrolling: viewport, element-level, and infinite scroll handling
- Drag-and-drop interactions
- Comparing Playwright vs Computer Use for each action

### Prerequisites
- Playwright setup and navigation (Lesson 01)
- Screenshot capture and page analysis (Lesson 02)
- Understanding of HTML form elements (Unit 1)
- Basic async/await knowledge (Unit 2)

---

## Mouse Click Actions

### Basic Clicks with Playwright

Playwright's `click()` method auto-scrolls to the element, waits for it to be actionable, and then clicks:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com")
    
    # Click a link by role and name
    page.get_by_role("link", name="More information").click()
    print(f"Navigated to: {page.url}")
    
    browser.close()
```

**Output:**
```
Navigated to: https://www.iana.org/help/example-domains
```

> **Note:** Playwright automatically waits for the element to be visible, enabled, and stable before clicking. No manual waits needed.

### Click Variations

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Set up a test page with various clickable elements
    page.set_content("""
        <button id="btn" onclick="this.textContent='Clicked!'">Click me</button>
        <div id="dbl" ondblclick="this.textContent='Double clicked!'">Double click me</div>
        <div id="ctx" oncontextmenu="this.textContent='Right clicked!'; return false;">Right click me</div>
        <div id="hover" onmouseover="this.textContent='Hovered!'">Hover me</div>
    """)
    
    # Single click
    page.locator("#btn").click()
    print(f"Button: {page.locator('#btn').inner_text()}")
    
    # Double click
    page.locator("#dbl").dblclick()
    print(f"Div: {page.locator('#dbl').inner_text()}")
    
    # Right click
    page.locator("#ctx").click(button="right")
    print(f"Context: {page.locator('#ctx').inner_text()}")
    
    # Hover (no click)
    page.locator("#hover").hover()
    print(f"Hover: {page.locator('#hover').inner_text()}")
    
    browser.close()
```

**Output:**
```
Button: Clicked!
Div: Double clicked!
Context: Right clicked!
Hover: Hovered!
```

### Click with Modifiers and Position

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div id="target" style="width:200px; height:200px; background:#eee;"
             onclick="document.getElementById('output').textContent = 
                'x:' + event.offsetX + ' y:' + event.offsetY + 
                ' shift:' + event.shiftKey">
            Click area
        </div>
        <div id="output"></div>
    """)
    
    # Click at a specific position within the element
    page.locator("#target").click(position={"x": 50, "y": 30})
    print(f"Position click: {page.locator('#output').inner_text()}")
    
    # Click with Shift held
    page.locator("#target").click(modifiers=["Shift"])
    print(f"Shift click: {page.locator('#output').inner_text()}")
    
    # Force click (bypasses actionability checks)
    page.locator("#target").click(force=True)
    print(f"Force click: {page.locator('#output').inner_text()}")
    
    browser.close()
```

**Output:**
```
Position click: x:50 y:30 shift:false
Shift click: x:100 y:100 shift:true
Force click: x:100 y:100 shift:false
```

### Clicks with Anthropic Computer Use

With the Computer Use tool, clicks use screen coordinates:

```python
# Computer Use click actions (within the agent loop)
# These are sent as tool_use responses from Claude

# Single click at coordinates
click_action = {
    "action": "left_click",
    "coordinate": [450, 300]  # [x, y] in screen pixels
}

# Right click
right_click_action = {
    "action": "right_click",
    "coordinate": [450, 300]
}

# Double click
double_click_action = {
    "action": "double_click",
    "coordinate": [450, 300]
}

# Move mouse without clicking
move_action = {
    "action": "mouse_move",
    "coordinate": [450, 300]
}
```

> **🤖 AI Context:** The key difference: Playwright clicks on DOM elements by semantic identity ("the Submit button"), while Computer Use clicks on screen coordinates (pixel position 450, 300). DOM-based is more resilient to layout changes; coordinate-based works with any visual interface.

---

## Text Input

### Filling Form Fields

Playwright's `fill()` method clears the field first, then types the value. It's the recommended way to set input values:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="name">Name:</label>
            <input id="name" type="text" />
            
            <label for="email">Email:</label>
            <input id="email" type="email" />
            
            <label for="bio">Bio:</label>
            <textarea id="bio"></textarea>
        </form>
    """)
    
    # Fill by label (recommended — most accessible)
    page.get_by_label("Name:").fill("Alice Johnson")
    page.get_by_label("Email:").fill("alice@example.com")
    page.get_by_label("Bio:").fill("AI researcher and developer")
    
    # Verify values
    print(f"Name: {page.locator('#name').input_value()}")
    print(f"Email: {page.locator('#email').input_value()}")
    print(f"Bio: {page.locator('#bio').input_value()}")
    
    browser.close()
```

**Output:**
```
Name: Alice Johnson
Email: alice@example.com
Bio: AI researcher and developer
```

### Clearing and Replacing Text

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content('<input id="field" value="old value" />')
    
    field = page.locator("#field")
    
    # fill() automatically clears first
    field.fill("new value")
    print(f"After fill: {field.input_value()}")
    
    # Clear explicitly
    field.clear()
    print(f"After clear: '{field.input_value()}'")
    
    # Fill again
    field.fill("final value")
    print(f"After refill: {field.input_value()}")
    
    browser.close()
```

**Output:**
```
After fill: new value
After clear: ''
After refill: final value
```

### Character-by-Character Typing

Some applications rely on `keydown`/`keyup` events rather than `input` events. Use `press_sequentially()` for these:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <input id="search" type="text" />
        <div id="suggestions"></div>
        <script>
            document.getElementById('search').addEventListener('input', (e) => {
                document.getElementById('suggestions').textContent = 
                    'Searching: ' + e.target.value;
            });
        </script>
    """)
    
    # Type character by character (triggers input events per keystroke)
    page.locator("#search").press_sequentially("hello", delay=50)
    
    print(f"Input: {page.locator('#search').input_value()}")
    print(f"Suggestions: {page.locator('#suggestions').inner_text()}")
    
    browser.close()
```

**Output:**
```
Input: hello
Suggestions: Searching: hello
```

> **Warning:** Use `fill()` for most cases — it's faster and more reliable. Only use `press_sequentially()` when the application specifically needs individual keystrokes (autocomplete, live search, character counters).

### Text Input with Computer Use

```python
# Computer Use type action
type_action = {
    "action": "type",
    "text": "Hello, world!"
}

# The agent first clicks the input field, then types
# Step 1: Click the search box
click_input = {
    "action": "left_click",
    "coordinate": [640, 200]  # Search box location
}

# Step 2: Type the query
type_query = {
    "action": "type",
    "text": "Python tutorials"
}
```

---

## Keyboard Shortcuts and Special Keys

### Pressing Individual Keys

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <input id="field" type="text" />
        <div id="output"></div>
        <script>
            const field = document.getElementById('field');
            const output = document.getElementById('output');
            field.addEventListener('keydown', (e) => {
                output.textContent = 'Key: ' + e.key + ' Code: ' + e.code;
            });
        </script>
    """)
    
    field = page.locator("#field")
    field.fill("Hello World")
    
    # Press Enter
    field.press("Enter")
    print(f"Enter: {page.locator('#output').inner_text()}")
    
    # Press Tab
    field.press("Tab")
    print(f"Tab: {page.locator('#output').inner_text()}")
    
    # Press Escape
    field.press("Escape")
    print(f"Escape: {page.locator('#output').inner_text()}")
    
    browser.close()
```

**Output:**
```
Enter: Key: Enter Code: Enter
Tab: Key: Tab Code: Tab
Escape: Key: Escape Code: Escape
```

### Key Combinations

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <input id="field" type="text" value="Hello World" />
        <div id="log"></div>
        <script>
            const field = document.getElementById('field');
            field.addEventListener('keydown', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    document.getElementById('log').textContent = 
                        (e.ctrlKey ? 'Ctrl+' : 'Meta+') + e.key;
                }
            });
        </script>
    """)
    
    field = page.locator("#field")
    field.focus()
    
    # Select all (Ctrl+A)
    field.press("Control+a")
    print(f"Combo: {page.locator('#log').inner_text()}")
    
    # Copy (Ctrl+C)
    field.press("Control+c")
    print(f"Copy: {page.locator('#log').inner_text()}")
    
    browser.close()
```

**Output:**
```
Combo: Ctrl+a
Copy: Ctrl+c
```

### Common Key Names

| Key | Playwright Name | Computer Use Name |
|-----|----------------|-------------------|
| Enter/Return | `Enter` | `Return` |
| Tab | `Tab` | `Tab` |
| Escape | `Escape` | `Escape` |
| Backspace | `Backspace` | `BackSpace` |
| Delete | `Delete` | `Delete` |
| Arrow keys | `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight` | `Up`, `Down`, `Left`, `Right` |
| Select all | `Control+a` | `ctrl+a` |
| Copy | `Control+c` | `ctrl+c` |
| Paste | `Control+v` | `ctrl+v` |
| Undo | `Control+z` | `ctrl+z` |

### Key Actions with Computer Use

```python
# Press a single key
key_action = {
    "action": "key",
    "key": "Return"
}

# Key combination
combo_action = {
    "action": "key",
    "key": "ctrl+a"  # Select all
}

# Multiple keys in sequence
# First select all, then delete
select_all = {"action": "key", "key": "ctrl+a"}
delete = {"action": "key", "key": "BackSpace"}
```

---

## Scrolling

### Auto-Scrolling in Playwright

Playwright automatically scrolls elements into view before interacting with them. However, you may need manual scrolling for:
- Infinite scroll pages
- Revealing lazy-loaded content
- Testing scroll behavior

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Create a tall page
    page.set_content("""
        <div style="height: 3000px; padding: 20px;">
            <h1 id="top">Top of page</h1>
            <div style="margin-top: 2500px;">
                <button id="bottom" onclick="this.textContent='Found!'">
                    Bottom button
                </button>
            </div>
        </div>
    """)
    
    # Playwright auto-scrolls to the element before clicking
    page.locator("#bottom").click()
    print(f"Button: {page.locator('#bottom').inner_text()}")
    
    # Verify scroll position changed
    scroll_y = page.evaluate("window.scrollY")
    print(f"Scrolled to: {scroll_y}px")
    
    browser.close()
```

**Output:**
```
Button: Found!
Scrolled to: 2300px
```

### Manual Scrolling

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <div style="height: 5000px; padding: 20px;">
            <div id="pos"></div>
        </div>
        <script>
            window.addEventListener('scroll', () => {
                document.getElementById('pos').textContent = 
                    'Scroll: ' + Math.round(window.scrollY);
            });
        </script>
    """)
    
    # Scroll using mouse wheel
    page.mouse.wheel(0, 500)
    page.wait_for_timeout(100)
    pos1 = page.evaluate("window.scrollY")
    print(f"After wheel scroll: {pos1}px")
    
    # Scroll to specific position with JavaScript
    page.evaluate("window.scrollTo(0, 2000)")
    page.wait_for_timeout(100)
    pos2 = page.evaluate("window.scrollY")
    print(f"After scrollTo: {pos2}px")
    
    # Scroll to top
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(100)
    pos3 = page.evaluate("window.scrollY")
    print(f"After scroll to top: {pos3}px")
    
    # Scroll element into view
    page.locator("#pos").scroll_into_view_if_needed()
    
    browser.close()
```

**Output:**
```
After wheel scroll: 500px
After scrollTo: 2000px
After scroll to top: 0px
```

### Handling Infinite Scroll

Many modern sites load content as you scroll. Here's a pattern for handling this:

```python
from playwright.sync_api import sync_playwright

def scroll_to_load_all(page, max_scrolls: int = 20) -> int:
    """
    Scroll down repeatedly to load all content.
    Returns the total number of items found.
    """
    previous_height = 0
    scroll_count = 0
    
    while scroll_count < max_scrolls:
        # Get current scroll height
        current_height = page.evaluate(
            "document.documentElement.scrollHeight"
        )
        
        if current_height == previous_height:
            # No new content loaded — we're done
            break
        
        previous_height = current_height
        
        # Scroll to the bottom
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        
        # Wait for new content to load
        page.wait_for_timeout(1000)
        scroll_count += 1
    
    return scroll_count

# Usage example
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Create a simulated infinite scroll page
    page.set_content("""
        <div id="container"></div>
        <script>
            let count = 0;
            function addItems() {
                const container = document.getElementById('container');
                for (let i = 0; i < 10; i++) {
                    count++;
                    const div = document.createElement('div');
                    div.textContent = 'Item ' + count;
                    div.style.padding = '20px';
                    container.appendChild(div);
                }
            }
            addItems(); // Initial load
            
            // Load more on scroll
            let loading = false;
            window.addEventListener('scroll', () => {
                if (loading) return;
                if (window.innerHeight + window.scrollY >= 
                    document.documentElement.scrollHeight - 100) {
                    if (count < 50) {
                        loading = true;
                        setTimeout(() => {
                            addItems();
                            loading = false;
                        }, 200);
                    }
                }
            });
        </script>
    """)
    
    scrolls = scroll_to_load_all(page, max_scrolls=10)
    item_count = page.locator("#container > div").count()
    print(f"Scrolled {scrolls} times, loaded {item_count} items")
    
    browser.close()
```

**Output:**
```
Scrolled 5 times, loaded 50 items
```

### Scrolling with Computer Use

```python
# Computer Use scroll action
scroll_down = {
    "action": "scroll",
    "coordinate": [640, 400],  # Scroll at center of screen
    "delta_x": 0,
    "delta_y": 500  # Positive = scroll down
}

scroll_up = {
    "action": "scroll",
    "coordinate": [640, 400],
    "delta_x": 0,
    "delta_y": -500  # Negative = scroll up
}
```

---

## Drag and Drop

### Playwright Drag and Drop

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <style>
            .box { width: 100px; height: 100px; position: absolute; cursor: grab; }
            #draggable { background: #4CAF50; top: 50px; left: 50px; }
            #target { background: #2196F3; top: 50px; left: 300px; 
                      border: 2px dashed #ccc; }
            #result { position: absolute; top: 200px; left: 50px; }
        </style>
        <div id="draggable" class="box" draggable="true">Drag me</div>
        <div id="target" class="box">Drop here</div>
        <div id="result"></div>
        <script>
            const target = document.getElementById('target');
            target.addEventListener('dragover', e => e.preventDefault());
            target.addEventListener('drop', e => {
                e.preventDefault();
                document.getElementById('result').textContent = 'Dropped!';
            });
        </script>
    """)
    
    # Method 1: drag_to (simplest)
    source = page.locator("#draggable")
    target = page.locator("#target")
    source.drag_to(target)
    
    result = page.locator("#result").inner_text()
    print(f"Drag result: {result}")
    
    browser.close()
```

**Output:**
```
Drag result: Dropped!
```

### Manual Drag (for complex scenarios)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <style>
            #slider { width: 300px; height: 20px; background: #ddd; 
                      position: relative; margin: 50px; }
            #handle { width: 20px; height: 20px; background: #4CAF50; 
                      position: absolute; cursor: grab; top: 0; left: 0; }
            #value { margin: 50px; }
        </style>
        <div id="slider"><div id="handle"></div></div>
        <div id="value">0%</div>
        <script>
            const slider = document.getElementById('slider');
            const handle = document.getElementById('handle');
            let dragging = false;
            
            handle.addEventListener('mousedown', () => dragging = true);
            document.addEventListener('mouseup', () => dragging = false);
            document.addEventListener('mousemove', (e) => {
                if (!dragging) return;
                const rect = slider.getBoundingClientRect();
                let x = Math.max(0, Math.min(e.clientX - rect.left, 280));
                handle.style.left = x + 'px';
                let pct = Math.round((x / 280) * 100);
                document.getElementById('value').textContent = pct + '%';
            });
        </script>
    """)
    
    handle = page.locator("#handle")
    handle_box = handle.bounding_box()
    
    # Manual drag: hover → mouse down → move → mouse up
    page.mouse.move(
        handle_box["x"] + handle_box["width"] / 2,
        handle_box["y"] + handle_box["height"] / 2
    )
    page.mouse.down()
    
    # Move to 75% of the slider (roughly)
    slider_box = page.locator("#slider").bounding_box()
    target_x = slider_box["x"] + slider_box["width"] * 0.75
    page.mouse.move(target_x, handle_box["y"] + handle_box["height"] / 2)
    page.mouse.up()
    
    value = page.locator("#value").inner_text()
    print(f"Slider value: {value}")
    
    browser.close()
```

**Output:**
```
Slider value: 80%
```

### Drag with Computer Use

```python
# Computer Use drag action
drag_action = {
    "action": "drag",
    "start_coordinate": [100, 100],
    "end_coordinate": [350, 100]
}
```

---

## Action Comparison: Playwright vs Computer Use

| Action | Playwright | Computer Use |
|--------|-----------|--------------|
| **Click** | `locator.click()` — targets DOM element | `left_click(coordinate)` — targets screen pixel |
| **Type** | `locator.fill("text")` — sets value directly | `type(text)` — simulates keystrokes |
| **Key press** | `locator.press("Enter")` — uses key name | `key("Return")` — uses X11 key name |
| **Scroll** | `mouse.wheel(0, delta)` or auto-scroll | `scroll(coordinate, delta)` — at screen position |
| **Double click** | `locator.dblclick()` | `double_click(coordinate)` |
| **Right click** | `locator.click(button="right")` | `right_click(coordinate)` |
| **Drag** | `source.drag_to(target)` | `drag(start, end)` |
| **Hover** | `locator.hover()` | `mouse_move(coordinate)` |
| **Wait** | `page.wait_for_selector()` | `wait(duration)` |
| **Precision** | Element-level (exact) | Pixel-level (approximate) |
| **Speed** | Fast (direct DOM) | Slower (screenshot between actions) |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `fill()` instead of `press_sequentially()` | Faster and more reliable for setting field values |
| Let Playwright auto-scroll before clicks | Handles element visibility automatically |
| Use semantic locators for clicks (`get_by_role`) | More resilient to layout changes than coordinates |
| Add `wait_for_timeout()` after scrolling | Gives lazy-loaded content time to appear |
| Set `force=True` only as a last resort | Bypasses actionability checks — can cause silent failures |
| Use `bounding_box()` for manual drag coordinates | More reliable than hardcoded pixel values |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `press_sequentially()` for all text input | Use `fill()` — it's faster and handles clearing |
| Hardcoding scroll amounts | Calculate based on `scrollHeight` and `innerHeight` |
| Not waiting after scroll for lazy content | Add `wait_for_timeout()` or `wait_for_selector()` |
| Clicking coordinates that shift with viewport | Use DOM locators or recalculate with `bounding_box()` |
| Using `force=True` without understanding why | First investigate why the element isn't actionable |
| Forgetting Playwright auto-scrolls | Don't manually scroll before `click()` — it's redundant |

---

## Hands-on Exercise

### Your Task

Build an `InteractionAgent` class that wraps Playwright's action methods into an agent-friendly interface with clear method names and automatic error handling.

### Requirements
1. Create an `InteractionAgent` class that takes a Playwright `page` object
2. Implement `click_element(role, name)` — click by accessibility role and name
3. Implement `type_text(label, text)` — fill a form field by its label
4. Implement `press_key(key)` — press a keyboard shortcut
5. Implement `scroll_page(direction, amount)` — scroll up or down by a pixel amount
6. Implement `drag_element(source_text, target_text)` — drag one element to another
7. Each method should return a success/failure dict with the action taken

### Expected Result
The agent can perform all common browser interactions through simple method calls, with errors caught and reported rather than crashing.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `page.get_by_role(role, name=name)` for role-based element selection
- Use `page.get_by_label(label)` for form fields
- Wrap each action in try/except to catch `TimeoutError` and return failure info
- `page.mouse.wheel(0, amount)` scrolls down; use negative for up
- `source_locator.drag_to(target_locator)` handles the drag action
- Return dicts like `{"success": True, "action": "click", "target": "Submit button"}`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

class InteractionAgent:
    """Agent-friendly wrapper for browser interactions."""
    
    def __init__(self, page):
        self.page = page
    
    def click_element(self, role: str, name: str) -> dict:
        """Click an element by its accessibility role and name."""
        try:
            self.page.get_by_role(role, name=name).click(timeout=5000)
            return {
                "success": True,
                "action": "click",
                "target": f"{role}: {name}",
                "url_after": self.page.url
            }
        except PlaywrightTimeout:
            return {
                "success": False,
                "action": "click",
                "target": f"{role}: {name}",
                "error": "Element not found or not clickable"
            }
    
    def type_text(self, label: str, text: str) -> dict:
        """Fill a form field identified by its label."""
        try:
            self.page.get_by_label(label).fill(text)
            return {
                "success": True,
                "action": "type",
                "target": label,
                "value": text
            }
        except PlaywrightTimeout:
            return {
                "success": False,
                "action": "type",
                "target": label,
                "error": "Field not found"
            }
    
    def press_key(self, key: str) -> dict:
        """Press a key or key combination on the focused element."""
        try:
            self.page.keyboard.press(key)
            return {
                "success": True,
                "action": "key_press",
                "key": key
            }
        except Exception as e:
            return {
                "success": False,
                "action": "key_press",
                "key": key,
                "error": str(e)
            }
    
    def scroll_page(self, direction: str = "down", amount: int = 500) -> dict:
        """Scroll the page up or down."""
        delta = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(300)
        scroll_y = self.page.evaluate("window.scrollY")
        return {
            "success": True,
            "action": "scroll",
            "direction": direction,
            "amount": amount,
            "scroll_position": scroll_y
        }
    
    def drag_element(self, source_text: str, target_text: str) -> dict:
        """Drag one element to another by their visible text."""
        try:
            source = self.page.get_by_text(source_text)
            target = self.page.get_by_text(target_text)
            source.drag_to(target)
            return {
                "success": True,
                "action": "drag",
                "source": source_text,
                "target": target_text
            }
        except PlaywrightTimeout:
            return {
                "success": False,
                "action": "drag",
                "error": "Source or target element not found"
            }

# Test the agent
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("""
        <form>
            <label for="name">Full Name:</label>
            <input id="name" type="text" />
            <button type="button" 
                    onclick="this.textContent='Submitted!'">Submit</button>
        </form>
        <div style="height: 2000px;"></div>
    """)
    
    agent = InteractionAgent(page)
    
    # Type in a field
    result = agent.type_text("Full Name:", "Alice Johnson")
    print(f"Type: {result}")
    
    # Click a button
    result = agent.click_element("button", "Submit")
    print(f"Click: {result}")
    
    # Press a key
    result = agent.press_key("Tab")
    print(f"Key: {result}")
    
    # Scroll down
    result = agent.scroll_page("down", 500)
    print(f"Scroll: {result}")
    
    # Try clicking something that doesn't exist
    result = agent.click_element("button", "Nonexistent")
    print(f"Missing: {result}")
    
    browser.close()
```

**Output:**
```
Type: {'success': True, 'action': 'type', 'target': 'Full Name:', 'value': 'Alice Johnson'}
Click: {'success': True, 'action': 'click', 'target': 'button: Submit', 'url_after': 'about:blank'}
Key: {'success': True, 'action': 'key_press', 'key': 'Tab'}
Scroll: {'success': True, 'action': 'scroll', 'direction': 'down', 'amount': 500, 'scroll_position': 500}
Missing: {'success': False, 'action': 'click', 'target': 'button: Nonexistent', 'error': 'Element not found or not clickable'}
```

</details>

### Bonus Challenges
- [ ] Add a `history` list that records every action taken (for debugging or replay)
- [ ] Implement `click_at(x, y)` for coordinate-based clicking as a fallback
- [ ] Add retry logic: if `click_element` fails, try scrolling down and clicking again

---

## Summary

✅ Playwright's `click()` auto-waits and auto-scrolls — use `get_by_role()` for resilient targeting

✅ Use `fill()` for text input (fast, clears field) and `press_sequentially()` only when individual keystrokes matter

✅ Keyboard shortcuts use `press("Control+a")` in Playwright and `key("ctrl+a")` in Computer Use

✅ Playwright auto-scrolls elements into view, but manual scrolling is needed for infinite scroll and lazy-loaded content

✅ Computer Use actions operate on screen coordinates — less precise but works with any visual interface

**Next:** [Visual Element Identification](./04-visual-element-identification.md)

**Previous:** [Screen Understanding Capabilities](./02-screen-understanding-capabilities.md)

---

## Further Reading

- [Playwright Input Actions](https://playwright.dev/python/docs/input) - Complete input action reference
- [Playwright Auto-Waiting](https://playwright.dev/python/docs/actionability) - How Playwright decides when to act
- [Anthropic Computer Use Actions](https://docs.anthropic.com/en/docs/computer-use) - Complete action reference

<!-- 
Sources Consulted:
- Playwright Input/Actions: https://playwright.dev/python/docs/input
- Playwright Actionability: https://playwright.dev/python/docs/actionability
- Anthropic Computer Use: https://docs.anthropic.com/en/docs/computer-use
-->
