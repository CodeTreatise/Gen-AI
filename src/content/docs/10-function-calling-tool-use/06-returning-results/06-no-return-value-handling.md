---
title: "No-Return-Value Handling"
---

# No-Return-Value Handling

## Introduction

Not every function returns data. Sending an email, toggling a setting, deleting a file, logging an event — these **side-effect functions** perform an action but have no meaningful data to send back. Yet every function call from an AI model **must receive a result**. If you skip the result, the API will reject the request, or the model will be stuck waiting.

This lesson covers how to handle functions that don't return data, what confirmation messages to send back, and how to communicate side effects clearly so the model knows the action completed.

### What we'll cover

- Why every function call needs a result (even void ones)
- Confirmation message patterns for side-effect functions
- Communicating what changed (side-effect descriptions)
- Conditional results: action succeeded vs. failed
- Building a void result formatter
- When to return data from "void" functions

### Prerequisites

- Result format structure ([Lesson 06-01](./01-result-format-structure.md))
- Error result formatting ([Lesson 06-05](./05-error-result-formatting.md))

---

## Every function call needs a result

All three providers require a result for every function call — even if the function doesn't return anything meaningful.

### What happens without a result

| Provider | Behavior when result is missing |
|----------|-------------------------------|
| **OpenAI** | API returns an error: each `function_call` must have a matching `function_call_output` |
| **Anthropic** | API returns an error: every `tool_use` block must have a corresponding `tool_result` |
| **Gemini** | Model gets stuck: it expects a `functionResponse` for every `functionCall` |

```python
# ❌ This will cause an API error (OpenAI)
# The model called send_email(), but you never sent a result back
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "user", "content": "Send an email to bob@example.com"},
        # Model generates function_call for send_email...
        # ...but no function_call_output is provided
        # API ERROR: missing function_call_output for call_id
    ],
)
```

> **Important:** Even Python functions that return `None` must produce a result message for the model. The model needs confirmation that the action was attempted.

---

## Confirmation message patterns

### Simple success confirmation

The most basic pattern: tell the model the action succeeded.

```python
import json


def format_void_result(action_description: str) -> str:
    """Format a result for a function that performs an action without returning data."""
    return json.dumps({
        "success": True,
        "message": action_description,
    })


# Examples
results = [
    format_void_result("Email sent to bob@example.com"),
    format_void_result("File 'report.pdf' deleted successfully"),
    format_void_result("User preferences updated"),
    format_void_result("Notification scheduled for 3:00 PM"),
]

for r in results:
    print(r)
```

**Output:**
```
{"success": true, "message": "Email sent to bob@example.com"}
{"success": true, "message": "File 'report.pdf' deleted successfully"}
{"success": true, "message": "User preferences updated"}
{"success": true, "message": "Notification scheduled for 3:00 PM"}
```

### Detailed confirmation with context

For important actions, include details about what was done:

```python
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email and return a detailed confirmation."""
    # ... actual email sending logic ...
    
    # Return confirmation details, not just "success"
    return {
        "success": True,
        "action": "email_sent",
        "details": {
            "recipient": to,
            "subject": subject,
            "body_length": len(body),
            "timestamp": "2025-02-06T14:30:00Z",
        },
        "message": f"Email sent to {to} with subject '{subject}'.",
    }


def delete_file(file_path: str) -> dict:
    """Delete a file and confirm what was removed."""
    # ... actual deletion logic ...
    
    return {
        "success": True,
        "action": "file_deleted",
        "details": {
            "path": file_path,
            "size_bytes": 1024,
            "was_directory": False,
        },
        "message": f"Deleted file '{file_path}' (1 KB).",
    }


def update_settings(settings: dict) -> dict:
    """Update user settings and confirm the changes."""
    # ... actual update logic ...
    
    return {
        "success": True,
        "action": "settings_updated",
        "changes": [
            {"field": k, "new_value": v} for k, v in settings.items()
        ],
        "message": f"Updated {len(settings)} setting(s).",
    }


# Usage
result = send_email("bob@example.com", "Meeting Tomorrow", "Hi Bob, ...")
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "success": true,
  "action": "email_sent",
  "details": {
    "recipient": "bob@example.com",
    "subject": "Meeting Tomorrow",
    "body_length": 11,
    "timestamp": "2025-02-06T14:30:00Z"
  },
  "message": "Email sent to bob@example.com with subject 'Meeting Tomorrow'."
}
```

---

## Communicating side effects

When a function changes state, the model should know **what changed**. This helps it answer follow-up questions without calling another function.

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SideEffect:
    """Describes a state change caused by a function."""
    entity: str          # What was affected
    action: str          # What happened to it
    details: dict = field(default_factory=dict)  # Additional context


class SideEffectTracker:
    """Track and report side effects of function execution."""
    
    def __init__(self):
        self.effects: list[SideEffect] = []
    
    def record(self, entity: str, action: str, **details):
        """Record a side effect."""
        self.effects.append(SideEffect(entity, action, details))
    
    def to_result(self) -> dict:
        """Convert tracked side effects to a result dictionary."""
        return {
            "success": True,
            "side_effects": [
                {
                    "entity": e.entity,
                    "action": e.action,
                    **e.details,
                }
                for e in self.effects
            ],
            "summary": self._summarize(),
        }
    
    def _summarize(self) -> str:
        """Create a human-readable summary."""
        if not self.effects:
            return "No changes were made."
        
        parts = []
        for e in self.effects:
            parts.append(f"{e.action} {e.entity}")
        return ". ".join(parts) + "."


# Example: a function that creates a project (multiple side effects)
def create_project(name: str, owner: str) -> dict:
    """Create a new project with default settings."""
    tracker = SideEffectTracker()
    
    # Each step is a side effect
    project_id = "proj_abc123"
    tracker.record("project", "created", id=project_id, name=name)
    tracker.record("directory", "created", path=f"/projects/{name}")
    tracker.record("user", "assigned as owner", user=owner, role="owner")
    tracker.record("settings", "initialized with defaults", 
                   visibility="private", auto_deploy=False)
    
    result = tracker.to_result()
    result["project_id"] = project_id  # Include the generated ID
    return result


result = create_project("my-app", "alice")
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "success": true,
  "side_effects": [
    {"entity": "project", "action": "created", "id": "proj_abc123", "name": "my-app"},
    {"entity": "directory", "action": "created", "path": "/projects/my-app"},
    {"entity": "user", "action": "assigned as owner", "user": "alice", "role": "owner"},
    {"entity": "settings", "action": "initialized with defaults", "visibility": "private", "auto_deploy": false}
  ],
  "summary": "created project. created directory. assigned as owner user. initialized with defaults settings.",
  "project_id": "proj_abc123"
}
```

> **Tip:** Including generated IDs (like `project_id`) in void results is extremely valuable. The model can use them in follow-up function calls without asking the user.

---

## Conditional results: success vs. no-op

Sometimes a function is called but the action isn't needed — for example, "add user to group" when the user is already in the group. Report this clearly:

```python
def add_user_to_group(user_id: str, group_id: str) -> dict:
    """Add a user to a group. Handle already-member case."""
    # Check if already a member (simulated)
    is_already_member = user_id == "user_001"  # Simulated check
    
    if is_already_member:
        return {
            "success": True,
            "action_taken": False,   # Key distinction
            "reason": "no_change_needed",
            "message": f"User '{user_id}' is already a member of group '{group_id}'. No changes made.",
        }
    
    # ... actual add logic ...
    return {
        "success": True,
        "action_taken": True,
        "message": f"User '{user_id}' added to group '{group_id}'.",
        "details": {
            "user_id": user_id,
            "group_id": group_id,
            "role": "member",
            "added_at": "2025-02-06T14:30:00Z",
        },
    }


# Already a member
result = add_user_to_group("user_001", "engineering")
print(json.dumps(result, indent=2))

# New member
print()
result = add_user_to_group("user_002", "engineering")
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "success": true,
  "action_taken": false,
  "reason": "no_change_needed",
  "message": "User 'user_001' is already a member of group 'engineering'. No changes made."
}

{
  "success": true,
  "action_taken": true,
  "message": "User 'user_002' added to group 'engineering'.",
  "details": {
    "user_id": "user_002",
    "group_id": "engineering",
    "role": "member",
    "added_at": "2025-02-06T14:30:00Z"
  }
}
```

> **Note:** The `action_taken` field helps the model distinguish between "I did the thing" and "the thing was already done." Both are successful — the difference is whether state changed.

---

## Void result formatter for all providers

A unified formatter for functions without return values:

```python
import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VoidResultConfig:
    """Configuration for void result formatting."""
    include_timestamp: bool = True
    include_side_effects: bool = True
    verbose: bool = False


class VoidResultFormatter:
    """Format results for functions that don't return data."""
    
    def __init__(self, provider: str = "openai", config: VoidResultConfig | None = None):
        self.provider = provider
        self.config = config or VoidResultConfig()
    
    def success(
        self,
        message: str,
        call_id: str = "",
        function_name: str = "",
        side_effects: list[dict] | None = None,
        generated_ids: dict | None = None,
    ) -> dict:
        """Format a successful void result."""
        result_data = {"success": True, "message": message}
        
        if self.config.include_timestamp:
            result_data["timestamp"] = datetime.now().isoformat()
        
        if side_effects and self.config.include_side_effects:
            result_data["side_effects"] = side_effects
        
        if generated_ids:
            result_data["generated_ids"] = generated_ids
        
        return self._format_for_provider(result_data, call_id, function_name)
    
    def no_op(
        self,
        reason: str,
        call_id: str = "",
        function_name: str = "",
    ) -> dict:
        """Format a no-op result (action wasn't needed)."""
        result_data = {
            "success": True,
            "action_taken": False,
            "message": reason,
        }
        return self._format_for_provider(result_data, call_id, function_name)
    
    def _format_for_provider(
        self,
        result_data: dict,
        call_id: str,
        function_name: str,
    ) -> dict:
        """Format the result for the specific provider."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result_data),
                }]
            }
        elif self.provider == "gemini":
            return {
                "function_name": function_name,
                "response": result_data,
            }
        else:  # openai
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result_data),
            }


# Usage
formatter = VoidResultFormatter(provider="openai")

# Simple success
result = formatter.success(
    message="Email sent to bob@example.com",
    call_id="call_abc123",
)
print("SUCCESS:")
print(json.dumps(result, indent=2))

# No-op
result = formatter.no_op(
    reason="User 'alice' is already in the 'admins' group. No changes needed.",
    call_id="call_def456",
)
print("\nNO-OP:")
print(json.dumps(result, indent=2))

# Success with generated IDs
result = formatter.success(
    message="Project 'my-app' created successfully.",
    call_id="call_ghi789",
    generated_ids={"project_id": "proj_abc123", "api_key": "key_xyz"},
    side_effects=[
        {"entity": "project", "action": "created"},
        {"entity": "api_key", "action": "generated"},
    ],
)
print("\nSUCCESS WITH IDS:")
print(json.dumps(result, indent=2))
```

**Output:**
```
SUCCESS:
{
  "type": "function_call_output",
  "call_id": "call_abc123",
  "output": "{\"success\": true, \"message\": \"Email sent to bob@example.com\", \"timestamp\": \"2025-02-06T14:30:00\"}"
}

NO-OP:
{
  "type": "function_call_output",
  "call_id": "call_def456",
  "output": "{\"success\": true, \"action_taken\": false, \"message\": \"User 'alice' is already in the 'admins' group. No changes needed.\"}"
}

SUCCESS WITH IDS:
{
  "type": "function_call_output",
  "call_id": "call_ghi789",
  "output": "{\"success\": true, \"message\": \"Project 'my-app' created successfully.\", \"timestamp\": \"2025-02-06T14:30:00\", \"side_effects\": [{\"entity\": \"project\", \"action\": \"created\"}, {\"entity\": \"api_key\", \"action\": \"generated\"}], \"generated_ids\": {\"project_id\": \"proj_abc123\", \"api_key\": \"key_xyz\"}}"
}
```

---

## When to return data from "void" functions

Some functions seem void but should actually return data. Use this decision guide:

| Scenario | Return what? | Why? |
|----------|-------------|------|
| Creating a resource | The generated ID | Model needs it for follow-up calls |
| Sending a message | Delivery status + timestamp | User may ask "when was it sent?" |
| Updating a record | Old vs. new values | Model can confirm what changed |
| Deleting a resource | What was deleted | Model can confirm the right thing was removed |
| Toggling a setting | The new state | Model knows the current state without re-querying |
| Scheduling an action | When it's scheduled | Model can tell the user the timing |

```python
# ❌ Truly void — gives model nothing to work with
def toggle_dark_mode_bad() -> str:
    # ... toggle logic ...
    return json.dumps({"success": True, "message": "Done"})

# ✅ Returns the new state — model knows what happened
def toggle_dark_mode_good(current_mode: str) -> str:
    new_mode = "light" if current_mode == "dark" else "dark"
    # ... toggle logic ...
    return json.dumps({
        "success": True,
        "message": f"Display mode changed to {new_mode}.",
        "previous_mode": current_mode,
        "current_mode": new_mode,
    })
```

> **🤖 AI Context:** When in doubt, return more information. The model can ignore extra data, but it can't invent information it wasn't given. A generated ID, a timestamp, or a new state value costs very few tokens but can prevent an entire extra function call.

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Always return a result, even for void functions | All providers require a result for every function call |
| Include `success: true/false` consistently | Gives the model a reliable field to check |
| Report generated IDs in the result | The model needs IDs for follow-up operations |
| Distinguish success from no-op with `action_taken` | Helps the model give accurate confirmations to users |
| Describe side effects explicitly | The model can answer follow-up questions without extra calls |
| Return the new state after mutations | Prevents unnecessary "get current state" calls |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Returning `None` or empty string | Return `{"success": true, "message": "..."}` |
| Not including generated IDs | Always return IDs, keys, or handles the model may need |
| Generic "Done" messages | Be specific: "Email sent to bob@example.com at 2:30 PM" |
| Not distinguishing success from no-op | Use `action_taken: false` when no change was needed |
| Omitting timestamps on time-sensitive actions | Include when the action occurred or is scheduled |
| Returning the same message for all void functions | Customize the message to describe what specifically happened |

---

## Hands-on exercise

### Your task

Build a `TaskExecutor` class that wraps multiple side-effect functions and returns properly formatted void results, including side-effect tracking and generated IDs.

### Requirements

1. Create a `TaskExecutor` that takes a `provider` parameter
2. Implement `execute(func_name, func, args, call_id)` that:
   - Executes the function
   - If it returns `None`, creates a void success result
   - If it returns a dict, includes the dict data in the result
   - If it raises an exception, returns an error result
3. Test with three functions: `send_notification()`, `create_project()`, `delete_unused_files()`

### Expected result

```python
executor = TaskExecutor(provider="anthropic")

# Void function (returns None)
result = executor.execute("send_notification", send_notification, 
                          {"user": "alice", "text": "Hello"}, "toolu_1")
# → success with message

# Function with generated ID
result = executor.execute("create_project", create_project,
                          {"name": "my-app"}, "toolu_2")
# → success with project_id

# Failed function
result = executor.execute("delete_files", delete_files,
                          {"path": "/nonexistent"}, "toolu_3")
# → error result
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Check if the function return value is `None` to decide between void and data results
- Use `isinstance(result, dict)` to detect structured return values
- Reuse the `VoidResultFormatter` for void results
- Wrap the execution in try/except for error handling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from datetime import datetime


class TaskExecutor:
    """Execute side-effect functions with proper result formatting."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.formatter = VoidResultFormatter(provider=provider)
    
    def execute(
        self,
        func_name: str,
        func: callable,
        args: dict,
        call_id: str,
    ) -> dict:
        """Execute a function and format the result appropriately."""
        try:
            result = func(**args)
            
            if result is None:
                # Void function — create confirmation message
                arg_summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
                return self.formatter.success(
                    message=f"{func_name}({arg_summary}) completed successfully.",
                    call_id=call_id,
                    function_name=func_name,
                )
            elif isinstance(result, dict):
                # Function returned data — include it
                generated_ids = {}
                for key in list(result.keys()):
                    if key.endswith("_id") or key == "id":
                        generated_ids[key] = result[key]
                
                return self.formatter.success(
                    message=result.get("message", f"{func_name} completed."),
                    call_id=call_id,
                    function_name=func_name,
                    generated_ids=generated_ids if generated_ids else None,
                    side_effects=result.get("side_effects"),
                )
            else:
                # Scalar return — wrap as message
                return self.formatter.success(
                    message=str(result),
                    call_id=call_id,
                    function_name=func_name,
                )
        
        except Exception as e:
            error_data = {
                "success": False,
                "error": True,
                "message": f"{func_name} failed: {str(e)}",
            }
            return self._format_error(error_data, call_id, func_name)
    
    def _format_error(self, error_data: dict, call_id: str, func_name: str) -> dict:
        """Format an error for the provider."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "is_error": True,
                    "content": error_data["message"],
                }]
            }
        elif self.provider == "gemini":
            return {"function_name": func_name, "response": error_data}
        else:
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(error_data),
            }


# Test functions
def send_notification(user: str, text: str):
    """Void function — returns None."""
    print(f"[SENT] Notification to {user}: {text}")
    return None

def create_project(name: str) -> dict:
    """Returns generated ID."""
    return {
        "project_id": "proj_new_123",
        "message": f"Project '{name}' created.",
        "side_effects": [{"entity": "project", "action": "created"}],
    }

def delete_files(path: str):
    """Raises an error."""
    raise FileNotFoundError(f"Path '{path}' does not exist")


# Execute all three
executor = TaskExecutor(provider="anthropic")

for name, func, args, cid in [
    ("send_notification", send_notification, {"user": "alice", "text": "Hello"}, "toolu_1"),
    ("create_project", create_project, {"name": "my-app"}, "toolu_2"),
    ("delete_files", delete_files, {"path": "/nonexistent"}, "toolu_3"),
]:
    result = executor.execute(name, func, args, cid)
    print(f"\n--- {name} ---")
    print(json.dumps(result, indent=2))
```

**Output:**
```
[SENT] Notification to alice: Hello

--- send_notification ---
{
  "role": "user",
  "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{\"success\": true, \"message\": \"send_notification(user='alice', text='Hello') completed successfully.\", \"timestamp\": \"2025-02-06T14:30:00\"}"}]
}

--- create_project ---
{
  "role": "user",
  "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "{\"success\": true, \"message\": \"Project 'my-app' created.\", \"timestamp\": \"2025-02-06T14:30:00\", \"side_effects\": [{\"entity\": \"project\", \"action\": \"created\"}], \"generated_ids\": {\"project_id\": \"proj_new_123\"}}"}]
}

--- delete_files ---
{
  "role": "user",
  "content": [{"type": "tool_result", "tool_use_id": "toolu_3", "is_error": true, "content": "delete_files failed: Path '/nonexistent' does not exist"}]
}
```

</details>

### Bonus challenges

- [ ] Add an `undo()` method that reverses the last action (using tracked side effects)
- [ ] Implement batch execution that runs multiple void functions and returns a combined result
- [ ] Add a dry-run mode that describes what would happen without executing

---

## Summary

✅ Every function call must return a result — even `None`-returning functions need a confirmation message

✅ Use `{"success": true, "message": "..."}` as the minimum void result

✅ Include generated IDs, timestamps, and new state values — the model can't invent data it wasn't given

✅ Distinguish success from no-op with `action_taken: true/false`

✅ Track and report side effects so the model knows what changed without extra function calls

✅ When in doubt, return more data — a few extra tokens prevent entire follow-up calls

**Next:** [Continuing the Conversation →](./07-continuing-conversation.md) — Sending results back and managing the conversation flow

---

[← Previous: Error Result Formatting](./05-error-result-formatting.md) | [Back to Lesson Overview](./00-returning-results.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Gemini Function Calling Tutorial: https://ai.google.dev/gemini-api/docs/function-calling
-->
