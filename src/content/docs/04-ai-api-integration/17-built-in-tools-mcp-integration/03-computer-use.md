---
title: "Computer Use Tool"
---

# Computer Use Tool

## Introduction

Computer use tools allow AI models to interact with graphical interfaces through automated actions like clicking, typing, and taking screenshots. Both Anthropic and OpenAI offer computer use capabilities, enabling automation of desktop and browser tasks.

### What We'll Cover

- Anthropic's computer_use_20241022 tool
- Screen dimension configuration
- Action types and coordinates
- OpenAI's computer_use_preview
- Safety and sandboxing
- Human-in-the-loop patterns

### Prerequisites

- API access (Anthropic or OpenAI)
- Understanding of GUI automation
- Sandboxed environment for testing

---

## Anthropic Computer Use

### Basic Configuration

```python
from anthropic import Anthropic
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

client = Anthropic()

# Basic computer use request
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    tools=[{
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1
    }],
    messages=[{
        "role": "user",
        "content": "Take a screenshot of the current screen"
    }]
)

# Handle tool use
for block in response.content:
    if block.type == "tool_use" and block.name == "computer":
        print(f"Action: {block.input}")
```

### Computer Use Configuration

```python
@dataclass
class DisplayConfig:
    """Display configuration for computer use."""
    
    width_px: int = 1920
    height_px: int = 1080
    display_number: int = 1
    scale_factor: float = 1.0
    
    def to_tool(self) -> dict:
        """Convert to tool definition."""
        return {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": int(self.width_px / self.scale_factor),
            "display_height_px": int(self.height_px / self.scale_factor),
            "display_number": self.display_number
        }
    
    def scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates for actual display."""
        return (
            int(x * self.scale_factor),
            int(y * self.scale_factor)
        )


class ComputerUseClient:
    """Client for computer use interactions."""
    
    def __init__(self, display: DisplayConfig = None):
        self.client = Anthropic()
        self.display = display or DisplayConfig()
        self.action_history: List[dict] = []
    
    def execute(
        self,
        instruction: str,
        screenshot: Optional[str] = None,  # Base64 image
        max_actions: int = 10
    ) -> List[dict]:
        """Execute instruction with computer use."""
        
        messages = [{
            "role": "user",
            "content": self._build_content(instruction, screenshot)
        }]
        
        actions = []
        
        for _ in range(max_actions):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=[self.display.to_tool()],
                messages=messages
            )
            
            # Check for tool use
            tool_use = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "computer":
                    tool_use = block
                    break
            
            if not tool_use:
                # No more actions needed
                break
            
            action = tool_use.input
            actions.append(action)
            self.action_history.append(action)
            
            # Add tool result (screenshot) for next iteration
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "Action executed. Provide next screenshot here."
                }]
            })
        
        return actions
    
    def _build_content(
        self,
        instruction: str,
        screenshot: Optional[str]
    ) -> List[dict]:
        """Build message content."""
        
        content = []
        
        if screenshot:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot
                }
            })
        
        content.append({
            "type": "text",
            "text": instruction
        })
        
        return content


# Usage
computer = ComputerUseClient(DisplayConfig(
    width_px=1920,
    height_px=1080
))

# actions = computer.execute(
#     "Open the browser and navigate to example.com"
# )
```

---

## Action Types

### Action Type Definitions

```python
class ActionType(Enum):
    SCREENSHOT = "screenshot"
    CLICK = "mouse_click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MOVE = "mouse_move"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    DRAG = "drag"


@dataclass
class MouseAction:
    """Mouse action with coordinates."""
    
    action_type: ActionType
    x: int
    y: int
    button: str = "left"  # left, right, middle


@dataclass
class KeyboardAction:
    """Keyboard action."""
    
    action_type: ActionType
    text: Optional[str] = None  # For type
    key: Optional[str] = None   # For key press


@dataclass
class ScrollAction:
    """Scroll action."""
    
    x: int
    y: int
    scroll_x: int = 0
    scroll_y: int = 0  # Positive = down


@dataclass
class DragAction:
    """Drag action."""
    
    start_x: int
    start_y: int
    end_x: int
    end_y: int


def parse_action(action_input: dict) -> Any:
    """Parse action from API response."""
    
    action = action_input.get("action")
    
    if action == "screenshot":
        return ActionType.SCREENSHOT
    
    elif action == "mouse_click":
        return MouseAction(
            action_type=ActionType.CLICK,
            x=action_input.get("coordinate", [0, 0])[0],
            y=action_input.get("coordinate", [0, 0])[1]
        )
    
    elif action == "type":
        return KeyboardAction(
            action_type=ActionType.TYPE,
            text=action_input.get("text", "")
        )
    
    elif action == "key":
        return KeyboardAction(
            action_type=ActionType.KEY,
            key=action_input.get("key", "")
        )
    
    elif action == "scroll":
        coord = action_input.get("coordinate", [0, 0])
        return ScrollAction(
            x=coord[0],
            y=coord[1],
            scroll_x=action_input.get("scroll_x", 0),
            scroll_y=action_input.get("scroll_y", 0)
        )
    
    else:
        return action_input
```

### Action Executor

```python
from typing import Callable, Protocol

class ActionExecutor(Protocol):
    """Protocol for action executors."""
    
    def click(self, x: int, y: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def press_key(self, key: str) -> None: ...
    def scroll(self, x: int, y: int, dx: int, dy: int) -> None: ...
    def screenshot(self) -> bytes: ...


class LoggingExecutor:
    """Executor that logs actions (for testing)."""
    
    def __init__(self):
        self.actions: List[dict] = []
    
    def click(self, x: int, y: int) -> None:
        self.actions.append({"type": "click", "x": x, "y": y})
        print(f"Click at ({x}, {y})")
    
    def type_text(self, text: str) -> None:
        self.actions.append({"type": "type", "text": text})
        print(f"Type: {text}")
    
    def press_key(self, key: str) -> None:
        self.actions.append({"type": "key", "key": key})
        print(f"Press key: {key}")
    
    def scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self.actions.append({"type": "scroll", "x": x, "y": y, "dx": dx, "dy": dy})
        print(f"Scroll at ({x}, {y}) by ({dx}, {dy})")
    
    def screenshot(self) -> bytes:
        self.actions.append({"type": "screenshot"})
        print("Take screenshot")
        return b""  # Would return actual image bytes


class ComputerController:
    """Controller that executes computer actions."""
    
    def __init__(
        self,
        executor: ActionExecutor,
        display: DisplayConfig = None
    ):
        self.executor = executor
        self.display = display or DisplayConfig()
    
    def execute_action(self, action_input: dict) -> Optional[bytes]:
        """Execute a computer action."""
        
        action = action_input.get("action")
        
        if action == "screenshot":
            return self.executor.screenshot()
        
        elif action == "mouse_click":
            coord = action_input.get("coordinate", [0, 0])
            x, y = self.display.scale_coordinates(coord[0], coord[1])
            self.executor.click(x, y)
        
        elif action == "type":
            text = action_input.get("text", "")
            self.executor.type_text(text)
        
        elif action == "key":
            key = action_input.get("key", "")
            self.executor.press_key(key)
        
        elif action == "scroll":
            coord = action_input.get("coordinate", [0, 0])
            x, y = self.display.scale_coordinates(coord[0], coord[1])
            dx = action_input.get("scroll_x", 0)
            dy = action_input.get("scroll_y", 0)
            self.executor.scroll(x, y, dx, dy)
        
        return None


# Usage
executor = LoggingExecutor()
controller = ComputerController(executor)

# Simulate action execution
controller.execute_action({
    "action": "mouse_click",
    "coordinate": [500, 300]
})

controller.execute_action({
    "action": "type",
    "text": "Hello, World!"
})
```

---

## OpenAI Computer Use Preview

### Basic Configuration

```python
from openai import OpenAI

client = OpenAI()

# OpenAI computer use (preview)
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1920,
        "display_height": 1080
    }],
    input="Navigate to the search bar and search for AI news"
)

# Handle computer actions
for item in response.output:
    if hasattr(item, 'type') and item.type == 'computer_action':
        print(f"Action: {item.action}")
```

### OpenAI Computer Handler

```python
@dataclass
class OpenAIDisplayConfig:
    """Display config for OpenAI computer use."""
    
    width: int = 1920
    height: int = 1080
    environment: str = "browser"  # browser, desktop
    
    def to_tool(self) -> dict:
        return {
            "type": "computer_use_preview",
            "display_width": self.width,
            "display_height": self.height,
            "environment": self.environment
        }


class OpenAIComputerClient:
    """Client for OpenAI computer use."""
    
    def __init__(self, display: OpenAIDisplayConfig = None):
        self.client = OpenAI()
        self.display = display or OpenAIDisplayConfig()
        self.conversation: List[dict] = []
    
    def execute(
        self,
        instruction: str,
        screenshot_base64: Optional[str] = None
    ) -> List[dict]:
        """Execute instruction with computer use."""
        
        # Build input
        input_content = []
        
        if screenshot_base64:
            input_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}"
                }
            })
        
        input_content.append({
            "type": "text",
            "text": instruction
        })
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[self.display.to_tool()],
            input=input_content
        )
        
        actions = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'computer_action':
                actions.append(self._parse_action(item))
        
        return actions
    
    def _parse_action(self, item) -> dict:
        """Parse action from response."""
        
        action = item.action
        
        result = {
            "type": action.type
        }
        
        if hasattr(action, 'x'):
            result["x"] = action.x
        if hasattr(action, 'y'):
            result["y"] = action.y
        if hasattr(action, 'text'):
            result["text"] = action.text
        if hasattr(action, 'key'):
            result["key"] = action.key
        
        return result


# Usage
openai_computer = OpenAIComputerClient(OpenAIDisplayConfig(
    width=1920,
    height=1080,
    environment="browser"
))

# actions = openai_computer.execute("Click on the login button")
```

---

## Safety and Sandboxing

### Sandbox Configuration

```python
from enum import Enum
from typing import Set

class SandboxLevel(Enum):
    STRICT = "strict"      # Only safe actions
    MODERATE = "moderate"  # Block dangerous actions
    PERMISSIVE = "permissive"  # Allow most actions


@dataclass
class SandboxConfig:
    """Sandbox configuration for computer use."""
    
    level: SandboxLevel = SandboxLevel.MODERATE
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_keys: Set[str] = field(default_factory=lambda: {"Delete", "F1", "F2"})
    max_actions_per_session: int = 100
    require_confirmation_for: Set[str] = field(default_factory=lambda: {"type", "key"})
    
    def is_action_allowed(self, action: dict) -> Tuple[bool, str]:
        """Check if action is allowed."""
        
        action_type = action.get("action") or action.get("type")
        
        if self.level == SandboxLevel.STRICT:
            if action_type not in ["screenshot", "mouse_move"]:
                return False, f"Action {action_type} blocked in strict mode"
        
        if action_type == "key":
            key = action.get("key", "")
            if key in self.blocked_keys:
                return False, f"Key {key} is blocked"
        
        return True, "Allowed"


class SafeComputerController:
    """Controller with safety checks."""
    
    def __init__(
        self,
        executor: ActionExecutor,
        sandbox: SandboxConfig = None,
        confirmation_callback: Callable[[dict], bool] = None
    ):
        self.executor = executor
        self.sandbox = sandbox or SandboxConfig()
        self.confirmation_callback = confirmation_callback or self._default_confirmation
        self.action_count = 0
    
    def _default_confirmation(self, action: dict) -> bool:
        """Default confirmation (always approve)."""
        return True
    
    def execute_action(self, action: dict) -> Tuple[bool, Optional[bytes], str]:
        """Execute action with safety checks."""
        
        # Check action limit
        if self.action_count >= self.sandbox.max_actions_per_session:
            return False, None, "Action limit exceeded"
        
        # Check if action is allowed
        allowed, reason = self.sandbox.is_action_allowed(action)
        if not allowed:
            return False, None, reason
        
        # Check if confirmation required
        action_type = action.get("action") or action.get("type")
        if action_type in self.sandbox.require_confirmation_for:
            if not self.confirmation_callback(action):
                return False, None, "User declined action"
        
        # Execute
        self.action_count += 1
        
        try:
            if action_type == "screenshot":
                result = self.executor.screenshot()
                return True, result, "Screenshot taken"
            
            elif action_type in ["mouse_click", "click"]:
                x = action.get("x") or action.get("coordinate", [0, 0])[0]
                y = action.get("y") or action.get("coordinate", [0, 0])[1]
                self.executor.click(x, y)
                return True, None, f"Clicked at ({x}, {y})"
            
            elif action_type == "type":
                text = action.get("text", "")
                self.executor.type_text(text)
                return True, None, f"Typed: {text[:20]}..."
            
            elif action_type == "key":
                key = action.get("key", "")
                self.executor.press_key(key)
                return True, None, f"Pressed: {key}"
            
            else:
                return False, None, f"Unknown action: {action_type}"
        
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"


# Usage
executor = LoggingExecutor()
sandbox = SandboxConfig(
    level=SandboxLevel.MODERATE,
    blocked_keys={"Delete", "Escape", "F1"},
    max_actions_per_session=50
)

safe_controller = SafeComputerController(executor, sandbox)

# Test action
success, result, message = safe_controller.execute_action({
    "action": "mouse_click",
    "coordinate": [100, 200]
})
print(f"Success: {success}, Message: {message}")
```

### Human-in-the-Loop

```python
class HumanApprovalSystem:
    """Human approval for computer actions."""
    
    def __init__(
        self,
        auto_approve_types: Set[str] = None,
        approval_callback: Callable[[dict], bool] = None
    ):
        self.auto_approve = auto_approve_types or {"screenshot", "mouse_move"}
        self.approval_callback = approval_callback
        self.approval_history: List[dict] = []
    
    def request_approval(self, action: dict) -> bool:
        """Request human approval for action."""
        
        action_type = action.get("action") or action.get("type")
        
        # Auto-approve safe actions
        if action_type in self.auto_approve:
            self._record_approval(action, True, "auto")
            return True
        
        # Use callback if provided
        if self.approval_callback:
            approved = self.approval_callback(action)
            self._record_approval(action, approved, "callback")
            return approved
        
        # Default: interactive approval
        return self._interactive_approval(action)
    
    def _interactive_approval(self, action: dict) -> bool:
        """Interactive terminal approval."""
        
        print("\n" + "=" * 50)
        print("ACTION APPROVAL REQUIRED")
        print("=" * 50)
        print(f"Action: {action}")
        
        response = input("\nApprove? (y/n): ").strip().lower()
        approved = response in ("y", "yes")
        
        self._record_approval(action, approved, "interactive")
        return approved
    
    def _record_approval(self, action: dict, approved: bool, method: str):
        """Record approval decision."""
        self.approval_history.append({
            "action": action,
            "approved": approved,
            "method": method,
            "timestamp": time.time()
        })
    
    def get_approval_stats(self) -> dict:
        """Get approval statistics."""
        
        total = len(self.approval_history)
        if total == 0:
            return {"message": "No approvals recorded"}
        
        approved = sum(1 for a in self.approval_history if a["approved"])
        
        by_method = {}
        for record in self.approval_history:
            method = record["method"]
            by_method[method] = by_method.get(method, 0) + 1
        
        return {
            "total": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": approved / total * 100,
            "by_method": by_method
        }


# Usage
approval = HumanApprovalSystem(
    auto_approve_types={"screenshot"},
    approval_callback=lambda a: a.get("action") != "type"  # Block all typing
)

# Test
approved = approval.request_approval({"action": "screenshot"})
print(f"Screenshot approved: {approved}")  # True (auto)

approved = approval.request_approval({"action": "type", "text": "password"})
print(f"Type approved: {approved}")  # False (blocked)
```

---

## Complete Computer Use Pipeline

### End-to-End Implementation

```python
import base64
import time

class ComputerUsePipeline:
    """Complete computer use pipeline with safety."""
    
    def __init__(
        self,
        provider: str = "anthropic",  # anthropic or openai
        display: DisplayConfig = None,
        sandbox: SandboxConfig = None,
        executor: ActionExecutor = None
    ):
        self.provider = provider
        self.display = display or DisplayConfig()
        self.sandbox = sandbox or SandboxConfig(level=SandboxLevel.MODERATE)
        self.executor = executor or LoggingExecutor()
        self.approval = HumanApprovalSystem()
        
        if provider == "anthropic":
            self.client = Anthropic()
        else:
            self.client = OpenAI()
    
    def run_task(
        self,
        task: str,
        max_steps: int = 20,
        initial_screenshot: bytes = None
    ) -> dict:
        """Run a computer task."""
        
        steps = []
        current_screenshot = initial_screenshot
        
        for step_num in range(max_steps):
            # Get next action from model
            action = self._get_action(task, current_screenshot, steps)
            
            if action is None:
                break
            
            # Request approval
            if not self.approval.request_approval(action):
                steps.append({
                    "step": step_num + 1,
                    "action": action,
                    "status": "rejected",
                    "message": "User rejected action"
                })
                continue
            
            # Check sandbox
            allowed, reason = self.sandbox.is_action_allowed(action)
            if not allowed:
                steps.append({
                    "step": step_num + 1,
                    "action": action,
                    "status": "blocked",
                    "message": reason
                })
                continue
            
            # Execute action
            result = self._execute_action(action)
            
            steps.append({
                "step": step_num + 1,
                "action": action,
                "status": "executed",
                **result
            })
            
            # Update screenshot if action produced one
            if result.get("screenshot"):
                current_screenshot = result["screenshot"]
        
        return {
            "task": task,
            "total_steps": len(steps),
            "steps": steps,
            "approval_stats": self.approval.get_approval_stats()
        }
    
    def _get_action(
        self,
        task: str,
        screenshot: bytes,
        history: List[dict]
    ) -> Optional[dict]:
        """Get next action from model."""
        
        if self.provider == "anthropic":
            return self._get_anthropic_action(task, screenshot, history)
        else:
            return self._get_openai_action(task, screenshot, history)
    
    def _get_anthropic_action(
        self,
        task: str,
        screenshot: bytes,
        history: List[dict]
    ) -> Optional[dict]:
        """Get action from Anthropic."""
        
        content = []
        
        if screenshot:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(screenshot).decode()
                }
            })
        
        # Build prompt with history
        prompt = f"Task: {task}\n\n"
        if history:
            prompt += "Previous actions:\n"
            for step in history[-5:]:  # Last 5 steps
                prompt += f"- {step['action']}: {step['status']}\n"
        prompt += "\nWhat action should I take next?"
        
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[self.display.to_tool()],
            messages=[{"role": "user", "content": content}]
        )
        
        for block in response.content:
            if block.type == "tool_use" and block.name == "computer":
                return block.input
        
        return None
    
    def _get_openai_action(
        self,
        task: str,
        screenshot: bytes,
        history: List[dict]
    ) -> Optional[dict]:
        """Get action from OpenAI."""
        
        input_content = []
        
        if screenshot:
            input_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(screenshot).decode()}"
                }
            })
        
        prompt = f"Task: {task}"
        input_content.append({"type": "text", "text": prompt})
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "computer_use_preview",
                "display_width": self.display.width_px,
                "display_height": self.display.height_px
            }],
            input=input_content
        )
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'computer_action':
                return {"action": item.action.type, **vars(item.action)}
        
        return None
    
    def _execute_action(self, action: dict) -> dict:
        """Execute an action."""
        
        action_type = action.get("action")
        
        try:
            if action_type == "screenshot":
                screenshot = self.executor.screenshot()
                return {"screenshot": screenshot, "message": "Screenshot taken"}
            
            elif action_type in ["mouse_click", "click"]:
                coord = action.get("coordinate", [action.get("x", 0), action.get("y", 0)])
                x, y = self.display.scale_coordinates(coord[0], coord[1])
                self.executor.click(x, y)
                return {"message": f"Clicked at ({x}, {y})"}
            
            elif action_type == "type":
                text = action.get("text", "")
                self.executor.type_text(text)
                return {"message": f"Typed {len(text)} characters"}
            
            elif action_type == "key":
                key = action.get("key", "")
                self.executor.press_key(key)
                return {"message": f"Pressed {key}"}
            
            else:
                return {"message": f"Unhandled action: {action_type}"}
        
        except Exception as e:
            return {"error": str(e)}


# Usage
pipeline = ComputerUsePipeline(
    provider="anthropic",
    display=DisplayConfig(width_px=1920, height_px=1080),
    sandbox=SandboxConfig(level=SandboxLevel.MODERATE)
)

# result = pipeline.run_task(
#     "Open a web browser and search for 'AI news'",
#     max_steps=10
# )
```

---

## Hands-on Exercise

### Your Task

Build a safe computer use automation system.

### Requirements

1. Configure display settings
2. Implement action parsing
3. Add safety sandbox
4. Create approval workflow
5. Log all actions

<details>
<summary>💡 Hints</summary>

- Scale coordinates for display
- Block dangerous keys
- Require confirmation for typing
- Track action history
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from enum import Enum
from datetime import datetime
import json

class SafetyLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AutomationConfig:
    """Configuration for automation system."""
    
    display_width: int = 1920
    display_height: int = 1080
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    max_actions: int = 100
    log_file: Optional[str] = None
    
    # Safety settings
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "password", "secret", "token", "key"
    ])
    dangerous_keys: Set[str] = field(default_factory=lambda: {
        "Delete", "Backspace", "Escape"
    })


@dataclass
class ActionLog:
    """Log entry for an action."""
    
    timestamp: str
    action_type: str
    details: Dict[str, Any]
    approved: bool
    executed: bool
    result: str


class SafeAutomationSystem:
    """Complete safe automation system."""
    
    def __init__(self, config: AutomationConfig = None):
        self.config = config or AutomationConfig()
        self.action_count = 0
        self.logs: List[ActionLog] = []
        self.session_start = datetime.now()
    
    def parse_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and normalize action."""
        
        action_type = raw_action.get("action") or raw_action.get("type")
        
        parsed = {
            "type": action_type,
            "raw": raw_action
        }
        
        if action_type in ["mouse_click", "click"]:
            coord = raw_action.get("coordinate", [
                raw_action.get("x", 0),
                raw_action.get("y", 0)
            ])
            parsed["x"] = coord[0] if isinstance(coord, list) else coord
            parsed["y"] = coord[1] if isinstance(coord, list) else raw_action.get("y", 0)
        
        elif action_type == "type":
            parsed["text"] = raw_action.get("text", "")
        
        elif action_type == "key":
            parsed["key"] = raw_action.get("key", "")
        
        elif action_type == "scroll":
            parsed["scroll_x"] = raw_action.get("scroll_x", 0)
            parsed["scroll_y"] = raw_action.get("scroll_y", 0)
        
        return parsed
    
    def check_safety(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if action is safe."""
        
        action_type = action.get("type")
        
        # Check action count
        if self.action_count >= self.config.max_actions:
            return False, "Maximum actions reached"
        
        # High safety: only screenshots allowed
        if self.config.safety_level == SafetyLevel.HIGH:
            if action_type != "screenshot":
                return False, f"Only screenshots allowed in HIGH safety mode"
        
        # Check dangerous keys
        if action_type == "key":
            key = action.get("key", "")
            if key in self.config.dangerous_keys:
                return False, f"Dangerous key blocked: {key}"
        
        # Check blocked text patterns
        if action_type == "type":
            text = action.get("text", "").lower()
            for pattern in self.config.blocked_patterns:
                if pattern in text:
                    return False, f"Blocked pattern detected: {pattern}"
        
        # Check coordinates in bounds
        if action_type in ["click", "mouse_click"]:
            x, y = action.get("x", 0), action.get("y", 0)
            if x < 0 or x > self.config.display_width:
                return False, f"X coordinate out of bounds: {x}"
            if y < 0 or y > self.config.display_height:
                return False, f"Y coordinate out of bounds: {y}"
        
        return True, "Safe"
    
    def request_approval(
        self,
        action: Dict[str, Any],
        auto_approve: bool = False
    ) -> bool:
        """Request approval for action."""
        
        action_type = action.get("type")
        
        # Auto-approve safe actions
        safe_actions = {"screenshot", "mouse_move"}
        if action_type in safe_actions and auto_approve:
            return True
        
        # Medium safety: auto-approve clicks
        if self.config.safety_level == SafetyLevel.MEDIUM:
            if action_type in {"click", "mouse_click"} and auto_approve:
                return True
        
        # Low safety: auto-approve most actions
        if self.config.safety_level == SafetyLevel.LOW:
            if auto_approve:
                return True
        
        # Require manual approval
        print(f"\n[APPROVAL REQUIRED]")
        print(f"Action: {action_type}")
        print(f"Details: {json.dumps(action, indent=2)}")
        
        response = input("Approve? (y/n): ").strip().lower()
        return response in ("y", "yes")
    
    def execute_action(
        self,
        action: Dict[str, Any],
        executor: Callable[[Dict[str, Any]], Any] = None
    ) -> Dict[str, Any]:
        """Execute action with full pipeline."""
        
        # Parse action
        parsed = self.parse_action(action)
        
        # Check safety
        is_safe, safety_message = self.check_safety(parsed)
        
        if not is_safe:
            log = self._create_log(parsed, False, False, safety_message)
            self.logs.append(log)
            return {"success": False, "message": safety_message}
        
        # Request approval
        approved = self.request_approval(parsed, auto_approve=True)
        
        if not approved:
            log = self._create_log(parsed, False, False, "User rejected")
            self.logs.append(log)
            return {"success": False, "message": "User rejected action"}
        
        # Execute
        self.action_count += 1
        
        if executor:
            try:
                result = executor(parsed)
                message = "Executed successfully"
            except Exception as e:
                message = f"Error: {str(e)}"
                result = None
        else:
            # Log-only execution
            message = f"Logged: {parsed['type']}"
            result = None
        
        log = self._create_log(parsed, True, True, message)
        self.logs.append(log)
        
        return {"success": True, "message": message, "result": result}
    
    def _create_log(
        self,
        action: Dict[str, Any],
        approved: bool,
        executed: bool,
        result: str
    ) -> ActionLog:
        """Create log entry."""
        
        return ActionLog(
            timestamp=datetime.now().isoformat(),
            action_type=action.get("type", "unknown"),
            details=action,
            approved=approved,
            executed=executed,
            result=result
        )
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs as dicts."""
        return [
            {
                "timestamp": log.timestamp,
                "action_type": log.action_type,
                "approved": log.approved,
                "executed": log.executed,
                "result": log.result
            }
            for log in self.logs
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        
        total = len(self.logs)
        approved = sum(1 for log in self.logs if log.approved)
        executed = sum(1 for log in self.logs if log.executed)
        
        by_type = {}
        for log in self.logs:
            t = log.action_type
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "session_start": self.session_start.isoformat(),
            "total_actions": total,
            "approved": approved,
            "rejected": total - approved,
            "executed": executed,
            "failed": approved - executed,
            "by_type": by_type,
            "safety_level": self.config.safety_level.value
        }
    
    def save_logs(self, file_path: str = None):
        """Save logs to file."""
        
        path = file_path or self.config.log_file
        if not path:
            return
        
        with open(path, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "logs": self.get_logs()
            }, f, indent=2)


# Usage example
system = SafeAutomationSystem(AutomationConfig(
    display_width=1920,
    display_height=1080,
    safety_level=SafetyLevel.MEDIUM,
    max_actions=50,
    log_file="automation_log.json"
))

# Test actions
test_actions = [
    {"action": "screenshot"},
    {"action": "mouse_click", "coordinate": [500, 300]},
    {"action": "type", "text": "Hello, World!"},
    {"action": "type", "text": "my_password_123"},  # Should be blocked
    {"action": "key", "key": "Delete"},  # Should be blocked
]

for action in test_actions:
    result = system.execute_action(action)
    print(f"{action.get('action')}: {result['message']}")

# Summary
summary = system.get_summary()
print(f"\n=== Session Summary ===")
print(f"Total: {summary['total_actions']}")
print(f"Approved: {summary['approved']}")
print(f"Executed: {summary['executed']}")
print(f"By type: {summary['by_type']}")
```

</details>

---

## Summary

✅ Anthropic's computer_use_20241022 enables GUI automation  
✅ Screen dimensions configure the coordinate system  
✅ Actions include click, type, key, scroll, screenshot  
✅ OpenAI's computer_use_preview offers similar capabilities  
✅ Sandbox configurations prevent dangerous actions  
✅ Human-in-the-loop ensures safety for sensitive operations

**Next:** [MCP Fundamentals](./04-mcp-fundamentals.md)

---

## Further Reading

- [Anthropic Computer Use](https://docs.anthropic.com/en/docs/computer-use) — Official documentation
- [OpenAI Computer Use](https://platform.openai.com/docs/guides/computer-use) — Preview documentation
- [Safe Automation Practices](https://docs.anthropic.com/en/docs/computer-use#safety) — Safety guidelines
