---
title: "Computer Use Prompting"
---

# Computer Use Prompting

## Introduction

Computer use allows AI agents to interact with graphical interfaces—clicking buttons, typing text, scrolling pages, and reading screen content. Instead of calling APIs, the agent sees screenshots and issues mouse/keyboard commands like a human would.

This capability enables automation of tasks that don't have APIs: legacy software, web applications, desktop programs, or any interface designed for humans.

> **⚠️ Warning:** Computer use is a beta feature with significant security implications. Always run in sandboxed environments with minimal privileges.

### What We'll Cover

- Computer use architecture
- Available actions (click, type, scroll, screenshot)
- Coordinate scaling for different resolutions
- Agent loop implementation
- Safety constraints and sandboxing
- Prompt patterns for reliable automation

### Prerequisites

- [Multi-Turn Agent Loops](./02-multi-turn-agent-loops.md)

---

## Computer Use Architecture

### How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   AI Model      │────►│   Your Code     │────►│   VM/Container  │
│   (Claude)      │◄────│   (Agent Loop)  │◄────│   (Desktop)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │  1. Analyze           │  2. Execute           │
        │     screenshot        │     action            │
        │                       │                       │
        │  3. Return            │  4. Capture           │
        │     action            │     screenshot        │
        │                       │                       │
        └───────────────────────┴───────────────────────┘
```

### The Agent Loop

1. **Capture screenshot** of current screen state
2. **Send to model** with computer use tools enabled
3. **Model returns action** (click, type, scroll, etc.)
4. **Execute action** on the virtual machine
5. **Capture new screenshot** showing result
6. **Repeat** until task complete

---

## Enabling Computer Use

### Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()

# Define the computer tool
computer_tool = {
    "type": "computer_20250124",  # Use computer_20251124 for Opus 4.5
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768
}

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    tools=[computer_tool],
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Open the browser and go to google.com"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_base64
                    }
                }
            ]
        }
    ],
    betas=["computer-use-2025-01-24"]
)
```

### Tool Versions

| Model | Tool Version | Additional Features |
|-------|--------------|---------------------|
| Claude Sonnet 4 | `computer_20250124` | Standard actions |
| Claude Opus 4.5 | `computer_20251124` | Adds `zoom` action |

---

## Available Actions

### Core Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `screenshot` | Capture current screen | None |
| `left_click` | Click at coordinates | `x`, `y` |
| `right_click` | Right-click at coordinates | `x`, `y` |
| `double_click` | Double-click at coordinates | `x`, `y` |
| `type` | Type text | `text` |
| `key` | Press key combination | `key` (e.g., "Return", "ctrl+c") |
| `mouse_move` | Move cursor | `x`, `y` |
| `scroll` | Scroll the page | `x`, `y`, `delta_x`, `delta_y` |
| `drag` | Drag from one point to another | `start_x`, `start_y`, `end_x`, `end_y` |
| `hold_key` | Hold modifier while clicking | `key`, `action` |
| `wait` | Pause execution | `duration` (seconds) |

### Opus 4.5 Exclusive: Zoom

```python
# Enable zoom capability
computer_tool = {
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1920,
    "display_height_px": 1080,
    "enable_zoom": True  # Enable zoom action
}
```

The `zoom` action captures a high-resolution crop of a specific region:

```python
{
    "action": "zoom",
    "center_x": 500,
    "center_y": 300,
    "factor": 2  # 2x magnification
}
```

Use zoom for:
- Reading small text
- Precisely clicking small buttons
- Inspecting dense UI areas

---

## Coordinate Scaling

### The Scaling Problem

High-resolution displays challenge the model's accuracy. Anthropic recommends scaling screenshots to a maximum of 1568 pixels on the longest side.

### Scaling Implementation

```python
def scale_screenshot(image_path: str, max_size: int = 1568) -> tuple:
    """Scale screenshot and return image with scale factor."""
    from PIL import Image
    import base64
    import io
    
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    # Calculate scale factor
    scale = min(max_size / original_width, max_size / original_height, 1.0)
    
    if scale < 1.0:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_image = base64.standard_b64encode(buffer.getvalue()).decode()
    
    return base64_image, scale

def scale_coordinates_back(x: int, y: int, scale: float) -> tuple:
    """Convert Claude's coordinates back to original resolution."""
    return int(x / scale), int(y / scale)
```

### Using Scaled Coordinates

```python
# Capture and scale screenshot
screenshot_b64, scale = scale_screenshot("screen.png")

# Send to Claude with scaled dimensions
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": int(1920 * scale),  # Scaled dimensions
        "display_height_px": int(1080 * scale)
    }],
    messages=[...],
    betas=["computer-use-2025-01-24"]
)

# When executing click, scale coordinates back
for block in response.content:
    if block.type == "tool_use" and block.name == "computer":
        action = block.input
        if action["action"] == "left_click":
            # Scale back to original resolution
            real_x, real_y = scale_coordinates_back(
                action["x"], action["y"], scale
            )
            execute_click(real_x, real_y)
```

---

## Complete Agent Loop

```python
import anthropic
import pyautogui
import base64
from PIL import Image
import io

client = anthropic.Anthropic()

class ComputerUseAgent:
    def __init__(self, display_width: int = 1920, display_height: int = 1080):
        self.display_width = display_width
        self.display_height = display_height
        self.max_screenshot_size = 1568
        self.scale = 1.0
    
    def capture_screenshot(self) -> str:
        """Capture and scale screenshot, return base64."""
        # Capture screen
        screenshot = pyautogui.screenshot()
        
        # Calculate scale
        self.scale = min(
            self.max_screenshot_size / screenshot.width,
            self.max_screenshot_size / screenshot.height,
            1.0
        )
        
        if self.scale < 1.0:
            new_size = (
                int(screenshot.width * self.scale),
                int(screenshot.height * self.scale)
            )
            screenshot = screenshot.resize(new_size, Image.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode()
    
    def execute_action(self, action: dict):
        """Execute a computer action."""
        action_type = action["action"]
        
        if action_type == "screenshot":
            pass  # Will be captured in next iteration
        
        elif action_type == "left_click":
            x = int(action["x"] / self.scale)
            y = int(action["y"] / self.scale)
            pyautogui.click(x, y)
        
        elif action_type == "right_click":
            x = int(action["x"] / self.scale)
            y = int(action["y"] / self.scale)
            pyautogui.rightClick(x, y)
        
        elif action_type == "double_click":
            x = int(action["x"] / self.scale)
            y = int(action["y"] / self.scale)
            pyautogui.doubleClick(x, y)
        
        elif action_type == "type":
            pyautogui.typewrite(action["text"], interval=0.05)
        
        elif action_type == "key":
            pyautogui.hotkey(*action["key"].split("+"))
        
        elif action_type == "scroll":
            x = int(action["x"] / self.scale)
            y = int(action["y"] / self.scale)
            pyautogui.scroll(action["delta_y"], x=x, y=y)
        
        elif action_type == "mouse_move":
            x = int(action["x"] / self.scale)
            y = int(action["y"] / self.scale)
            pyautogui.moveTo(x, y)
        
        elif action_type == "wait":
            import time
            time.sleep(action.get("duration", 1))
    
    def run(self, task: str, max_iterations: int = 50):
        """Run the computer use agent."""
        
        messages = []
        
        for iteration in range(max_iterations):
            # Capture current screen
            screenshot = self.capture_screenshot()
            
            # Build message with screenshot
            content = [{"type": "text", "text": task}] if not messages else []
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot
                }
            })
            
            if not messages:
                messages.append({"role": "user", "content": content})
            else:
                # Add screenshot as tool result
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": self.last_tool_id,
                        "content": content
                    }]
                })
            
            # Call Claude
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=[{
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": int(self.display_width * self.scale),
                    "display_height_px": int(self.display_height * self.scale)
                }],
                messages=messages,
                betas=["computer-use-2025-01-24"]
            )
            
            # Add assistant response
            messages.append({"role": "assistant", "content": response.content})
            
            # Check if done
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return "Task completed"
            
            # Process tool use
            for block in response.content:
                if block.type == "tool_use":
                    self.last_tool_id = block.id
                    print(f"Action: {block.input}")
                    self.execute_action(block.input)
                    
                    # Small delay to let UI update
                    import time
                    time.sleep(0.5)
        
        return "Max iterations reached"

# Usage
agent = ComputerUseAgent()
result = agent.run("Open Firefox and search for 'Python tutorials'")
```

---

## System Prompt for Computer Use

```python
COMPUTER_USE_SYSTEM = """
You are an AI assistant that can control a computer to complete tasks.

## Available Actions
- screenshot: Capture current screen state
- left_click: Click at (x, y) coordinates
- right_click: Right-click at (x, y)
- double_click: Double-click at (x, y)
- type: Type text using keyboard
- key: Press key combinations (e.g., "Return", "ctrl+c")
- scroll: Scroll at position with delta
- mouse_move: Move cursor to position
- wait: Pause for a duration

## Guidelines

1. ALWAYS take a screenshot first to see the current state
2. Click on the CENTER of buttons and links, not edges
3. Wait after clicks for UI to update before next action
4. If an action fails, try an alternative approach
5. Use keyboard shortcuts when available (faster, more reliable)
6. Scroll to find elements not visible on screen
7. Type slowly and verify text appears correctly

## Safety Rules

1. NEVER enter passwords or sensitive credentials
2. NEVER click on suspicious links or downloads
3. NEVER modify system settings without explicit permission
4. NEVER close important applications without saving
5. ASK for confirmation before irreversible actions

## Approach

1. Analyze the screenshot to understand current state
2. Plan the next action to progress toward the goal
3. Execute ONE action at a time
4. Wait for the result before planning the next step
5. If stuck, try a different approach rather than repeating

When the task is complete, explain what was accomplished.
"""
```

---

## Safety and Sandboxing

### Run in Isolated Environments

> **🔒 Security:** Never run computer use agents on your main system.

```python
# Use Docker for isolation
docker_config = """
FROM ubuntu:22.04

# Install desktop environment
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    firefox \
    python3

# Create non-root user with limited permissions
RUN useradd -m -s /bin/bash agent
USER agent

# No network access to sensitive internal services
# Mount only necessary directories as read-only
"""
```

### Limit Permissions

```python
# Configure minimal permissions
SAFETY_RULES = """
Allowed actions:
- Open and use the web browser
- Navigate to specified websites only
- Fill in forms with provided data
- Take screenshots for verification

Forbidden actions:
- Access file system outside /tmp
- Execute terminal commands
- Install software
- Access system settings
- Enter credentials for any service
- Download executable files
"""
```

### Human Confirmation

```python
SENSITIVE_PATTERNS = [
    r"password",
    r"login",
    r"credit.?card",
    r"payment",
    r"delete",
    r"remove",
    r"format",
    r"install"
]

def requires_confirmation(action: dict, screen_text: str) -> bool:
    """Check if action needs human approval."""
    import re
    
    # Check screen content for sensitive elements
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, screen_text, re.IGNORECASE):
            return True
    
    # Check typed text
    if action.get("action") == "type":
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, action.get("text", ""), re.IGNORECASE):
                return True
    
    return False

async def execute_with_confirmation(action: dict, screen_text: str):
    """Execute action, prompting for confirmation if needed."""
    
    if requires_confirmation(action, screen_text):
        print(f"⚠️ Sensitive action detected: {action}")
        print(f"Current screen contains: {screen_text[:200]}...")
        
        confirmed = await prompt_user("Allow this action? (yes/no)")
        if confirmed.lower() != "yes":
            return {"error": "User rejected action"}
    
    return execute_action(action)
```

---

## Common Patterns

### Wait for Page Load

```python
def wait_for_stable_screen(agent, timeout: int = 10) -> bool:
    """Wait for screen to stop changing (page loaded)."""
    import time
    
    previous_hash = None
    stable_count = 0
    
    for _ in range(timeout * 2):  # Check every 0.5 seconds
        screenshot = agent.capture_screenshot()
        current_hash = hash(screenshot)
        
        if current_hash == previous_hash:
            stable_count += 1
            if stable_count >= 3:  # Stable for 1.5 seconds
                return True
        else:
            stable_count = 0
        
        previous_hash = current_hash
        time.sleep(0.5)
    
    return False
```

### Find and Click Element

```python
# Prompt pattern for reliable element clicking
CLICK_ELEMENT_PROMPT = """
Task: Click the "{element_description}" button/link

Steps:
1. Take a screenshot to see current state
2. Find the element matching "{element_description}"
3. If not visible, scroll to find it
4. Click on the CENTER of the element
5. Wait for any resulting action to complete
6. Confirm the click was successful

If you cannot find the element after scrolling, report that it's not present.
"""
```

### Fill Form Fields

```python
FILL_FORM_PROMPT = """
Task: Fill in the form with the following data:
{form_data}

Steps:
1. Take a screenshot to see the form
2. For each field:
   a. Click on the input field
   b. Clear any existing content (ctrl+a, then type)
   c. Type the new value
   d. Tab to the next field or click the next field
3. After filling all fields, take a screenshot to verify
4. Do NOT submit the form yet

Report which fields were filled and any issues encountered.
"""
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Clicks missing targets | Reduce screenshot resolution, use zoom for small elements |
| Actions happening too fast | Add `wait` actions between steps |
| Text typing errors | Use `key` for special characters, type slower |
| Can't find elements | Scroll the page, check if element is in a modal/popup |
| Coordinates wrong | Verify scale factor calculation |

---

## Hands-on Exercise

### Your Task

Create a prompt and action sequence for a computer use agent that:
1. Opens a web browser
2. Navigates to a search engine
3. Searches for "Python programming tutorials"
4. Clicks on the first result

<details>
<summary>✅ Solution (click to expand)</summary>

```python
TASK_PROMPT = """
Complete the following task on this computer:

1. Open the Firefox web browser
2. Navigate to google.com
3. Search for "Python programming tutorials"
4. Click on the first search result

Guidelines:
- Take screenshots after each major action to verify progress
- Wait for pages to load before clicking
- If a step fails, try an alternative approach
- Report when each step is completed
"""

# Expected action sequence:
EXPECTED_ACTIONS = [
    # 1. Find and click browser icon (or use keyboard shortcut)
    {"action": "key", "key": "super"},  # Open app menu
    {"action": "wait", "duration": 1},
    {"action": "type", "text": "firefox"},
    {"action": "key", "key": "Return"},
    {"action": "wait", "duration": 3},  # Wait for browser
    
    # 2. Navigate to Google
    {"action": "key", "key": "ctrl+l"},  # Focus address bar
    {"action": "type", "text": "google.com"},
    {"action": "key", "key": "Return"},
    {"action": "wait", "duration": 2},
    
    # 3. Search
    {"action": "type", "text": "Python programming tutorials"},
    {"action": "key", "key": "Return"},
    {"action": "wait", "duration": 2},
    
    # 4. Click first result
    # (Claude will determine coordinates from screenshot)
    {"action": "left_click", "x": 500, "y": 300},  # Approximate
]

# The agent will adapt based on actual screenshots
# This is just an illustration of the expected flow
```

</details>

---

## Summary

✅ **Computer use enables GUI automation** without APIs
✅ **Scale screenshots** to 1568px max for reliable coordinate handling
✅ **Implement the agent loop:** screenshot → analyze → action → repeat
✅ **Always run in sandboxed environments** with minimal privileges
✅ **Add wait states** between actions for UI updates
✅ **Require human confirmation** for sensitive operations

**Next:** [Agentic Safety Patterns](./05-agentic-safety-patterns.md)

---

## Further Reading

- [Anthropic Computer Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
- [Computer Use Reference Implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
- [PyAutoGUI Documentation](https://pyautogui.readthedocs.io/)

---

<!-- 
Sources Consulted:
- Anthropic Computer Use: Tool types, actions, agent loop
- Anthropic Computer Use: Coordinate scaling, zoom feature
- Anthropic Computer Use: Security recommendations
-->
