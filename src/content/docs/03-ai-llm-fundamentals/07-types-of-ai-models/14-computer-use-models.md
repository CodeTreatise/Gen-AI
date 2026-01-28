---
title: "Computer Use Models"
---

# Computer Use Models

## Introduction

Computer use models can control computers through screenshots, mouse movements, and keyboard input. These models enable automation of GUI-based tasks.

### What We'll Cover

- Claude Computer Use
- Screen capture and control
- Browser automation
- Safety considerations

---

## Claude Computer Use

### Overview

```python
# Claude's computer use capability
# Model: claude-3-5-sonnet-20241022
# Beta feature requiring special tools

computer_use_tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1
    },
    {
        "type": "text_editor_20241022",
        "name": "str_replace_editor"
    },
    {
        "type": "bash_20241022",
        "name": "bash"
    }
]
```

### Basic Computer Use

```python
from anthropic import Anthropic

client = Anthropic()

def computer_use_session(task: str):
    """Run a computer use session"""
    
    messages = [{"role": "user", "content": task}]
    
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=computer_use_tools,
        messages=messages,
        betas=["computer-use-2024-10-22"]
    )
    
    return response

# Example: Take a screenshot
result = computer_use_session("Take a screenshot of the current screen")
```

### Handling Computer Actions

```python
import pyautogui
import base64
from PIL import Image
import io

def execute_computer_action(action: dict) -> dict:
    """Execute computer action from Claude"""
    
    action_type = action.get("action")
    
    if action_type == "screenshot":
        # Take screenshot
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        return {
            "type": "image",
            "data": base64.b64encode(buffer.getvalue()).decode()
        }
    
    elif action_type == "mouse_move":
        x, y = action["coordinate"]
        pyautogui.moveTo(x, y)
        return {"status": "moved"}
    
    elif action_type == "left_click":
        pyautogui.click()
        return {"status": "clicked"}
    
    elif action_type == "right_click":
        pyautogui.rightClick()
        return {"status": "right_clicked"}
    
    elif action_type == "double_click":
        pyautogui.doubleClick()
        return {"status": "double_clicked"}
    
    elif action_type == "type":
        pyautogui.write(action["text"])
        return {"status": "typed"}
    
    elif action_type == "key":
        pyautogui.press(action["key"])
        return {"status": f"pressed {action['key']}"}
    
    return {"error": f"Unknown action: {action_type}"}
```

---

## Complete Computer Use Loop

```python
class ComputerUseAgent:
    """Agent that can control a computer"""
    
    def __init__(self):
        self.client = Anthropic()
        self.tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1920,
                "display_height_px": 1080,
                "display_number": 1
            }
        ]
    
    def run(self, task: str, max_steps: int = 20):
        """Execute task using computer control"""
        
        messages = [{"role": "user", "content": task}]
        
        for step in range(max_steps):
            response = self.client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                tools=self.tools,
                messages=messages,
                betas=["computer-use-2024-10-22"]
            )
            
            # Check for completion
            if response.stop_reason == "end_turn":
                return self._extract_text(response.content)
            
            # Process tool uses
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute(block.input)
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        }]
                    })
        
        return "Max steps reached"
    
    def _execute(self, action: dict) -> list:
        """Execute action and return screenshot"""
        
        execute_computer_action(action)
        
        # Always return screenshot after action
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        
        return [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode()
            }
        }]
    
    def _extract_text(self, content: list) -> str:
        for block in content:
            if hasattr(block, 'text'):
                return block.text
        return ""
```

---

## Browser Automation

### Web Navigation

```python
class BrowserAgent(ComputerUseAgent):
    """Specialized for browser automation"""
    
    def open_browser(self, url: str):
        """Open browser to URL"""
        return self.run(f"Open a web browser and navigate to {url}")
    
    def fill_form(self, form_data: dict):
        """Fill out a form"""
        instructions = "Fill out the form with this data:\n"
        for field, value in form_data.items():
            instructions += f"- {field}: {value}\n"
        return self.run(instructions)
    
    def extract_data(self, description: str):
        """Extract data from current page"""
        return self.run(f"Extract the following from this page: {description}")

# Usage
agent = BrowserAgent()
agent.open_browser("https://example.com/form")
agent.fill_form({
    "Name": "John Doe",
    "Email": "john@example.com",
    "Message": "Hello, world!"
})
```

### Safe Browsing

```python
class SafeBrowserAgent(BrowserAgent):
    """Browser agent with safety controls"""
    
    BLOCKED_DOMAINS = ["bank.com", "payment.com", "admin."]
    BLOCKED_ACTIONS = ["delete", "remove", "purchase"]
    
    def run(self, task: str, **kwargs):
        # Check for blocked content
        task_lower = task.lower()
        
        for domain in self.BLOCKED_DOMAINS:
            if domain in task_lower:
                return "Action blocked: Cannot access sensitive domains"
        
        for action in self.BLOCKED_ACTIONS:
            if action in task_lower:
                return f"Action blocked: Cannot perform '{action}' actions"
        
        return super().run(task, **kwargs)
```

---

## OpenAI Operator

### Concept (Announced 2025)

```python
# OpenAI Operator - web task automation
# Note: API details may vary from this example

class OperatorAgent:
    """OpenAI's Operator for web tasks"""
    
    def __init__(self):
        self.client = OpenAI()
    
    async def run_web_task(self, task: str, website: str):
        """Execute web task"""
        
        response = await self.client.operator.create(
            task=task,
            starting_url=website,
            max_steps=30,
            sandbox=True  # Run in isolated environment
        )
        
        return {
            "status": response.status,
            "result": response.result,
            "steps_taken": response.steps
        }

# Example usage
agent = OperatorAgent()
result = await agent.run_web_task(
    task="Find the cheapest flight from NYC to LA on December 25",
    website="https://flights.example.com"
)
```

---

## Use Cases

### Form Filling

```python
def automate_form_filling(form_url: str, data: dict):
    """Automate filling out a web form"""
    
    agent = ComputerUseAgent()
    
    # Navigate to form
    agent.run(f"Open browser and go to {form_url}")
    
    # Fill each field
    for field_name, value in data.items():
        agent.run(f"Find the field labeled '{field_name}' and enter: {value}")
    
    # Submit
    agent.run("Click the submit button")
    
    return "Form submitted"
```

### Data Entry

```python
def bulk_data_entry(records: list, app_name: str):
    """Enter data into an application"""
    
    agent = ComputerUseAgent()
    
    for i, record in enumerate(records):
        agent.run(f"""
        In {app_name}:
        1. Click 'New Record'
        2. Enter: {record}
        3. Click 'Save'
        4. Wait for confirmation
        """)
        print(f"Processed record {i+1}/{len(records)}")
```

### Testing

```python
def automated_ui_test(test_case: dict):
    """Run UI test case"""
    
    agent = ComputerUseAgent()
    
    # Setup
    agent.run(test_case["setup"])
    
    # Execute steps
    for step in test_case["steps"]:
        agent.run(step["action"])
        
        # Verify
        result = agent.run(f"Check: {step['expected']}")
        if "not found" in result.lower():
            return {"status": "FAIL", "step": step}
    
    return {"status": "PASS"}
```

---

## Safety Considerations

### Critical Safeguards

```python
class SafeComputerAgent:
    """Computer agent with safety measures"""
    
    def __init__(self):
        self.confirm_destructive = True
        self.allowed_apps = ["browser", "notepad", "terminal"]
        self.blocked_paths = ["/etc", "/system", "C:\\Windows"]
    
    def run(self, task: str):
        # 1. Validate task
        if self._is_destructive(task) and self.confirm_destructive:
            if not self._get_user_confirmation(task):
                return "Action cancelled by user"
        
        # 2. Check allowed apps
        if not self._is_allowed_app(task):
            return "Application not in allowed list"
        
        # 3. Check paths
        if self._accesses_blocked_path(task):
            return "Cannot access system paths"
        
        # 4. Run in sandbox if possible
        return self._run_sandboxed(task)
    
    def _is_destructive(self, task: str) -> bool:
        destructive_keywords = ["delete", "remove", "format", "shutdown", "reboot"]
        return any(kw in task.lower() for kw in destructive_keywords)
    
    def _get_user_confirmation(self, task: str) -> bool:
        response = input(f"Confirm destructive action: {task}? (yes/no): ")
        return response.lower() == "yes"
```

### Sandboxing

```python
# Run computer use in isolated environment
sandbox_options = {
    "docker": "Run in Docker container",
    "vm": "Run in virtual machine",
    "cloud": "Run in cloud sandbox (e.g., E2B)",
}

# Example with E2B
from e2b import Sandbox

def run_in_sandbox(task: str):
    """Run computer use in E2B sandbox"""
    
    with Sandbox() as sandbox:
        # Execute task in isolated environment
        result = sandbox.run(task)
        return result
```

---

## Hands-on Exercise

### Your Task

Build a simple computer automation (simulated):

```python
class SimulatedComputerAgent:
    """Simulated computer agent for learning"""
    
    def __init__(self):
        self.screen_state = "desktop"
        self.open_windows = []
        self.cursor_pos = (500, 500)
    
    def screenshot(self) -> str:
        """Get current screen state"""
        return f"Screen: {self.screen_state}, Windows: {self.open_windows}, Cursor: {self.cursor_pos}"
    
    def click(self, x: int, y: int):
        """Simulate click"""
        self.cursor_pos = (x, y)
        print(f"Clicked at ({x}, {y})")
        
        # Simulate clicking on things
        if 0 <= x <= 100 and 0 <= y <= 50:
            self.open_windows.append("Browser")
            self.screen_state = "browser"
        return self.screenshot()
    
    def type_text(self, text: str):
        """Simulate typing"""
        print(f"Typed: {text}")
        return self.screenshot()
    
    def press_key(self, key: str):
        """Simulate key press"""
        print(f"Pressed: {key}")
        if key == "escape" and self.open_windows:
            self.open_windows.pop()
            self.screen_state = "desktop" if not self.open_windows else self.open_windows[-1]
        return self.screenshot()

# Test
agent = SimulatedComputerAgent()
print(agent.screenshot())
print(agent.click(50, 25))  # Click browser icon
print(agent.type_text("https://example.com"))
print(agent.press_key("enter"))
```

---

## Summary

✅ **Claude Computer Use**: Screenshot → Analyze → Act

✅ **Actions**: Mouse, keyboard, screenshot

✅ **Browser automation**: Form filling, data extraction

✅ **Safety critical**: Sandboxing, confirmations, restrictions

✅ **Use cases**: Testing, data entry, automation

**Next:** [3D & Spatial Models](./15-3d-spatial-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Agent & Tool-Use](./13-agent-tool-use-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [3D & Spatial Models](./15-3d-spatial-models.md) |

