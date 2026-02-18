---
title: "Override mechanisms"
---

# Override mechanisms

## Introduction

Confirmation workflows ask "should we proceed?" Override mechanisms answer "stop *right now*." When an agent is heading in the wrong direction, taking too long, or doing something unexpected, users need immediate, reliable ways to take control. A kill switch that doesn't work isn't a safety feature — it's a liability.

In this lesson, we'll build override mechanisms that range from gentle course corrections to hard emergency stops, including ways to undo actions the agent has already completed.

### What we'll cover

- Manual override options for taking direct control of agent tasks
- Emergency stop mechanisms that halt execution immediately
- Direction changes that redirect agents mid-task
- Undo and rollback capabilities for reversing completed actions

### Prerequisites

- [Confirmation Workflows](./01-confirmation-workflows.md) — basic approval patterns
- [Feedback Incorporation](./03-feedback-incorporation.md) — how overrides feed into learning
- [Error Handling & Recovery](../08-error-handling-recovery/) — graceful failure patterns

---

## Manual override options

A manual override lets the user take direct control of a specific step while the agent handles everything else. This is different from stopping the agent — it's *replacing* the agent for one part of the work.

### Override levels

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime


class OverrideLevel(Enum):
    """How much control the user takes."""
    SUGGEST = "suggest"       # Agent proceeds, user's input is advisory
    REPLACE = "replace"       # User's output replaces agent's for this step
    TAKEOVER = "takeover"     # User handles this step and all remaining steps
    PILOT = "pilot"           # User drives, agent assists with suggestions


@dataclass
class OverrideRequest:
    """A user request to override agent behavior."""
    level: OverrideLevel
    target_step: str               # Which step to override
    user_input: Optional[Any] = None  # User-provided replacement
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class OverrideManager:
    """Manages manual overrides in an agent workflow."""
    
    def __init__(self):
        self.overrides: list[OverrideRequest] = []
        self.active_takeover: bool = False
    
    def request_override(
        self,
        level: OverrideLevel,
        target_step: str,
        user_input: Any = None,
        reason: str = ""
    ) -> OverrideRequest:
        """Register an override request."""
        override = OverrideRequest(
            level=level,
            target_step=target_step,
            user_input=user_input,
            reason=reason
        )
        self.overrides.append(override)
        
        if level == OverrideLevel.TAKEOVER:
            self.active_takeover = True
        
        return override
    
    def should_agent_execute(self, step_name: str) -> bool:
        """Check if the agent should execute this step or yield to the user."""
        if self.active_takeover:
            return False  # User has taken over
        
        # Check for step-specific overrides
        for override in reversed(self.overrides):
            if override.target_step == step_name:
                return override.level == OverrideLevel.SUGGEST
        
        return True  # No override, agent proceeds
    
    def get_override_for_step(self, step_name: str) -> Optional[OverrideRequest]:
        """Get the most recent override for a specific step."""
        for override in reversed(self.overrides):
            if override.target_step == step_name:
                return override
        return None


# Usage
manager = OverrideManager()

# User wants to write the email body themselves
manager.request_override(
    level=OverrideLevel.REPLACE,
    target_step="draft_email",
    user_input="Dear Team,\n\nPlease find attached the Q3 report.\n\nBest,\nAlice",
    reason="I want to phrase this carefully"
)

# Check each step
for step in ["gather_data", "draft_email", "review_tone", "send_email"]:
    should_run = manager.should_agent_execute(step)
    override = manager.get_override_for_step(step)
    status = "🤖 Agent" if should_run else "👤 User"
    print(f"{status} handles: {step}", end="")
    if override:
        print(f" (override: {override.level.value})")
    else:
        print()
```

**Output:**
```
🤖 Agent handles: gather_data
👤 User handles: draft_email (override: replace)
🤖 Agent handles: review_tone
🤖 Agent handles: send_email
```

### LangGraph override integration

```python
from langgraph.types import interrupt, Command


def overridable_step(state: dict) -> dict:
    """A step that checks for overrides before executing."""
    step_name = "draft_response"
    
    # Ask the user: proceed, override, or skip?
    decision = interrupt({
        "step": step_name,
        "agent_output": state.get("draft", ""),
        "question": "How would you like to handle this step?",
        "options": [
            "proceed",       # Let agent output stand
            "replace",       # Provide your own output
            "skip"           # Skip this step entirely
        ]
    })
    
    if isinstance(decision, dict) and decision.get("action") == "replace":
        return {"draft": decision["content"], "overridden": True}
    elif decision == "skip":
        return {"skipped_steps": [step_name]}
    
    return {}  # Agent output stands
```

> **🔑 Key concept:** Overrides should be non-destructive by default. The agent's original output is preserved in state even when the user overrides it — this allows comparison and learning from the difference.

---

## Emergency stops

Emergency stops are the safety net. When something goes wrong — the agent is calling external APIs with wrong data, spending money, or generating harmful content — users need a way to halt execution *immediately*.

### Stop signal design

```python
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class StopReason(Enum):
    USER_REQUESTED = "user_requested"
    SAFETY_VIOLATION = "safety_violation"
    COST_LIMIT = "cost_limit"
    TIMEOUT = "timeout"
    ERROR_CASCADE = "error_cascade"


@dataclass
class StopSignal:
    """An emergency stop signal."""
    reason: StopReason
    message: str = ""
    graceful: bool = True  # False = hard kill, True = finish current step
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class EmergencyStop:
    """Thread-safe emergency stop mechanism."""
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._signal: Optional[StopSignal] = None
        self._on_stop_callbacks: list = []
    
    def stop(
        self,
        reason: StopReason = StopReason.USER_REQUESTED,
        message: str = "",
        graceful: bool = True
    ) -> StopSignal:
        """Trigger an emergency stop."""
        self._signal = StopSignal(
            reason=reason,
            message=message,
            graceful=graceful
        )
        self._stop_event.set()
        
        # Notify all registered callbacks
        for callback in self._on_stop_callbacks:
            try:
                callback(self._signal)
            except Exception:
                pass  # Don't let callback errors block the stop
        
        return self._signal
    
    @property
    def is_stopped(self) -> bool:
        """Check if a stop has been triggered."""
        return self._stop_event.is_set()
    
    @property
    def signal(self) -> Optional[StopSignal]:
        return self._signal
    
    def on_stop(self, callback):
        """Register a callback to run when stop is triggered."""
        self._on_stop_callbacks.append(callback)
    
    def reset(self):
        """Reset the stop signal (allows resumption)."""
        self._stop_event.clear()
        self._signal = None
    
    def check(self, step_name: str = "") -> None:
        """Check if we should stop. Call this between steps.
        
        Raises:
            AgentStoppedError: If a stop signal is active.
        """
        if self.is_stopped:
            signal = self._signal
            raise AgentStoppedError(
                f"Agent stopped at '{step_name}': "
                f"{signal.reason.value} — {signal.message}"
            )


class AgentStoppedError(Exception):
    """Raised when an emergency stop is triggered."""
    pass


# Usage
emergency = EmergencyStop()

# Register a cleanup callback
emergency.on_stop(
    lambda sig: print(f"🛑 STOP: {sig.reason.value} — {sig.message}")
)

# Simulate a workflow with stop checks
steps = ["fetch_data", "analyze", "draft_response", "send_email"]

emergency.stop(
    reason=StopReason.COST_LIMIT,
    message="API costs exceeded $5.00 budget"
)

for step in steps:
    try:
        emergency.check(step)
        print(f"✅ Executing: {step}")
    except AgentStoppedError as e:
        print(f"⛔ {e}")
        break
```

**Output:**
```
🛑 STOP: cost_limit — API costs exceeded $5.00 budget
⛔ Agent stopped at 'fetch_data': cost_limit — API costs exceeded $5.00 budget
```

### Graceful vs. hard stops

```python
@dataclass
class StopPolicy:
    """Define what happens during a stop."""
    save_state: bool = True          # Save current state for resumption
    rollback_current: bool = False   # Undo the in-progress step
    notify_user: bool = True         # Show stop notification
    log_for_audit: bool = True       # Record in audit trail
    cleanup_resources: bool = True   # Release locks, close connections


STOP_POLICIES = {
    "graceful": StopPolicy(
        save_state=True,
        rollback_current=False,  # Finish current step, then stop
        notify_user=True,
        log_for_audit=True,
        cleanup_resources=True
    ),
    "hard": StopPolicy(
        save_state=True,
        rollback_current=True,   # Undo current step immediately
        notify_user=True,
        log_for_audit=True,
        cleanup_resources=True
    ),
    "panic": StopPolicy(
        save_state=False,        # Don't even save — just stop
        rollback_current=True,
        notify_user=True,
        log_for_audit=True,
        cleanup_resources=False  # May leak resources, but stops faster
    )
}

for name, policy in STOP_POLICIES.items():
    print(f"\n{name.upper()} stop:")
    print(f"  Save state: {policy.save_state}")
    print(f"  Rollback current: {policy.rollback_current}")
    print(f"  Cleanup resources: {policy.cleanup_resources}")
```

**Output:**
```
GRACEFUL stop:
  Save state: True
  Rollback current: False
  Cleanup resources: True

HARD stop:
  Save state: True
  Rollback current: True
  Cleanup resources: True

PANIC stop:
  Save state: False
  Rollback current: True
  Cleanup resources: False
```

> **Warning:** Always default to graceful stops. Hard stops can leave data in inconsistent states. Panic stops should only be used for genuine safety emergencies — not for "oops, wrong recipient."

---

## Direction changes

Sometimes you don't want to *stop* the agent — you want to *redirect* it. The agent is doing the right thing but heading toward the wrong goal. Direction changes modify the agent's objective mid-execution.

### Mid-task redirection

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DirectionChange:
    """A change in the agent's goal or approach mid-execution."""
    original_goal: str
    new_goal: str
    reason: str
    preserve_progress: bool = True  # Keep work done so far
    restart_from: Optional[str] = None  # Step to restart from
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class DirectionManager:
    """Manages mid-task direction changes."""
    
    def __init__(self):
        self.changes: list[DirectionChange] = []
        self.current_goal: str = ""
        self.completed_steps: list[str] = []
    
    def set_initial_goal(self, goal: str):
        """Set the starting goal."""
        self.current_goal = goal
    
    def change_direction(
        self,
        new_goal: str,
        reason: str,
        preserve_progress: bool = True,
        restart_from: Optional[str] = None
    ) -> DirectionChange:
        """Change the agent's direction mid-task."""
        change = DirectionChange(
            original_goal=self.current_goal,
            new_goal=new_goal,
            reason=reason,
            preserve_progress=preserve_progress,
            restart_from=restart_from
        )
        self.changes.append(change)
        self.current_goal = new_goal
        
        if not preserve_progress:
            self.completed_steps.clear()
        elif restart_from and restart_from in self.completed_steps:
            # Roll back to the restart point
            idx = self.completed_steps.index(restart_from)
            self.completed_steps = self.completed_steps[:idx]
        
        return change
    
    def complete_step(self, step: str):
        """Mark a step as completed."""
        self.completed_steps.append(step)
    
    def get_context_for_agent(self) -> dict:
        """Build context that includes direction change history."""
        return {
            "current_goal": self.current_goal,
            "completed_steps": self.completed_steps.copy(),
            "direction_changes": len(self.changes),
            "latest_change": (
                {
                    "from": self.changes[-1].original_goal,
                    "to": self.changes[-1].new_goal,
                    "reason": self.changes[-1].reason
                }
                if self.changes else None
            )
        }


# Usage
director = DirectionManager()
director.set_initial_goal("Write a formal apology email to the client")

# Agent completes some steps
director.complete_step("gather_context")
director.complete_step("draft_email")

# User changes direction
director.change_direction(
    new_goal="Write a brief status update instead of an apology",
    reason="Manager said the issue was already resolved",
    preserve_progress=True,
    restart_from="draft_email"
)

context = director.get_context_for_agent()
print(f"Current goal: {context['current_goal']}")
print(f"Completed steps: {context['completed_steps']}")
print(f"Direction changes: {context['direction_changes']}")
print(f"Latest change: {context['latest_change']}")
```

**Output:**
```
Current goal: Write a brief status update instead of an apology
Completed steps: ['gather_context']
Direction changes: 1
Latest change: {'from': 'Write a formal apology email to the client', 'to': 'Write a brief status update instead of an apology', 'reason': 'Manager said the issue was already resolved'}
```

### LangGraph direction change with `Command`

```python
from langgraph.types import interrupt, Command


def check_direction(state: dict) -> Command:
    """Checkpoint where the user can redirect the agent."""
    
    # Show progress and ask if the direction is right
    response = interrupt({
        "current_goal": state["goal"],
        "progress": state.get("completed_steps", []),
        "question": "Continue with this goal, or change direction?",
        "options": ["continue", "change_goal", "restart"]
    })
    
    if response == "continue":
        return Command(goto="next_step")
    
    if isinstance(response, dict) and response.get("action") == "change_goal":
        return Command(
            goto=response.get("restart_from", "next_step"),
            update={
                "goal": response["new_goal"],
                "direction_changes": state.get("direction_changes", 0) + 1
            }
        )
    
    if response == "restart":
        return Command(
            goto="start",
            update={"completed_steps": []}
        )
```

---

## Undo capabilities

The hardest override is undoing something already done. Once an email is sent or an API call is made, you can't un-ring that bell. But you can design systems that make undo *possible* for as many actions as you can.

### Action reversibility classification

```python
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any, Optional


class Reversibility(Enum):
    REVERSIBLE = "reversible"           # Can be fully undone
    COMPENSATABLE = "compensatable"     # Can't undo, but can take corrective action
    IRREVERSIBLE = "irreversible"       # Cannot be undone at all


@dataclass
class ReversibleAction:
    """An action with its undo counterpart."""
    name: str
    execute: Callable             # The forward action
    undo: Optional[Callable]      # The reverse action (None = irreversible)
    reversibility: Reversibility
    compensation: str = ""        # Description of compensating action
    
    def can_undo(self) -> bool:
        return self.reversibility == Reversibility.REVERSIBLE and self.undo is not None


EXAMPLE_ACTIONS = {
    "create_draft": ReversibleAction(
        name="create_draft",
        execute=lambda: "draft created",
        undo=lambda: "draft deleted",
        reversibility=Reversibility.REVERSIBLE
    ),
    "send_email": ReversibleAction(
        name="send_email",
        execute=lambda: "email sent",
        undo=None,
        reversibility=Reversibility.COMPENSATABLE,
        compensation="Send a follow-up correction email"
    ),
    "delete_record": ReversibleAction(
        name="delete_record",
        execute=lambda: "record deleted",
        undo=None,
        reversibility=Reversibility.IRREVERSIBLE
    )
}

for name, action in EXAMPLE_ACTIONS.items():
    print(f"{name}:")
    print(f"  Reversibility: {action.reversibility.value}")
    print(f"  Can undo: {action.can_undo()}")
    if action.compensation:
        print(f"  Compensation: {action.compensation}")
    print()
```

**Output:**
```
create_draft:
  Reversibility: reversible
  Can undo: True

send_email:
  Reversibility: compensatable
  Can undo: False
  Compensation: Send a follow-up correction email

delete_record:
  Reversibility: irreversible
  Can undo: False
```

### Undo stack

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


@dataclass
class ActionRecord:
    """A completed action that may be undoable."""
    action_name: str
    parameters: dict
    result: Any
    undo_fn: Optional[Callable] = None
    undone: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class UndoStack:
    """Maintains a stack of completed actions that can be undone."""
    
    def __init__(self, max_history: int = 50):
        self.history: list[ActionRecord] = []
        self.max_history = max_history
    
    def record(
        self,
        action_name: str,
        parameters: dict,
        result: Any,
        undo_fn: Optional[Callable] = None
    ) -> ActionRecord:
        """Record a completed action."""
        record = ActionRecord(
            action_name=action_name,
            parameters=parameters,
            result=result,
            undo_fn=undo_fn
        )
        self.history.append(record)
        
        # Trim old history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return record
    
    def undo_last(self) -> dict:
        """Undo the most recent undoable action."""
        for record in reversed(self.history):
            if not record.undone and record.undo_fn:
                try:
                    undo_result = record.undo_fn()
                    record.undone = True
                    return {
                        "success": True,
                        "action": record.action_name,
                        "undo_result": undo_result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "action": record.action_name,
                        "error": str(e)
                    }
        
        return {"success": False, "error": "No undoable actions in history"}
    
    def undo_to(self, action_name: str) -> list[dict]:
        """Undo all actions back to (and including) the named action."""
        results = []
        found = False
        
        for record in reversed(self.history):
            if record.undone:
                continue
            
            if record.undo_fn:
                try:
                    undo_result = record.undo_fn()
                    record.undone = True
                    results.append({
                        "action": record.action_name,
                        "success": True,
                        "result": undo_result
                    })
                except Exception as e:
                    results.append({
                        "action": record.action_name,
                        "success": False,
                        "error": str(e)
                    })
            
            if record.action_name == action_name:
                found = True
                break
        
        if not found:
            results.append({"error": f"Action '{action_name}' not found in history"})
        
        return results
    
    def get_undoable(self) -> list[str]:
        """List all actions that can still be undone."""
        return [
            r.action_name
            for r in reversed(self.history)
            if not r.undone and r.undo_fn
        ]


# Usage
stack = UndoStack()

# Simulate a workflow
stack.record(
    "create_folder", {"path": "/reports"},
    result="created",
    undo_fn=lambda: "folder deleted"
)
stack.record(
    "create_file", {"path": "/reports/q3.txt"},
    result="created",
    undo_fn=lambda: "file deleted"
)
stack.record(
    "send_notification", {"to": "team"},
    result="sent",
    undo_fn=None  # Can't unsend
)

print(f"Undoable actions: {stack.get_undoable()}")

result = stack.undo_last()
print(f"\nUndo last: {result}")

print(f"\nRemaining undoable: {stack.get_undoable()}")
```

**Output:**
```
Undoable actions: ['create_file', 'create_folder']

Undo last: {'success': True, 'action': 'create_file', 'undo_result': 'file deleted'}

Remaining undoable: ['create_folder']
```

> **🤖 AI Context:** Design agent actions for reversibility whenever possible. Instead of directly sending an email, create a draft and schedule it. Instead of deleting records, soft-delete them. This gives users a window to undo before the action becomes irreversible.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Default to graceful stops over hard kills | Hard stops can corrupt state and leave resources dangling |
| Classify every action's reversibility upfront | Users need to know *before* approving whether they can undo |
| Keep an undo stack with configurable depth | Even 10 levels of undo covers most "oops" moments |
| Make stop checks cheap and frequent | A stop check between every step costs microseconds but catches problems fast |
| Preserve the agent's output even when overridden | The difference between agent output and human override is valuable feedback data |
| Design for compensation, not just reversal | "Send a correction email" is better than "can't undo" |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Only supporting full stop — no partial override | Offer suggest, replace, takeover, and pilot levels |
| Emergency stop that doesn't actually stop background tasks | Use thread-safe events and check at every step boundary |
| Undo that doesn't handle dependencies | If action B depends on action A, undoing A must also undo B |
| No way to resume after a stop | Save state at every checkpoint so users can restart from where they stopped |
| Treating direction changes as errors | Direction changes are normal — log them as decisions, not failures |
| Override without recording *why* | Always capture the reason — it's essential for preference learning |

---

## Hands-on exercise

### Your task

Build a `SafeWorkflow` that combines emergency stops, manual overrides, direction changes, and undo capabilities into a unified control system.

### Requirements

1. An `EmergencyStop` mechanism with graceful and hard stop options
2. An `OverrideManager` that lets users replace specific steps
3. An `UndoStack` that tracks reversible actions
4. A workflow runner that checks for stops between steps, applies overrides, and records undo-able actions
5. A `status()` method showing current state, completed steps, and available overrides

### Expected result

```python
workflow = SafeWorkflow(steps=["fetch", "analyze", "draft", "review", "send"])

workflow.override("draft", user_input="My custom draft", level="replace")
workflow.run()  # Runs fetch, analyze with agent; uses user draft for draft step

workflow.undo_last()  # Undoes the most recent undoable step
workflow.emergency_stop(reason="User requested")
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Compose `EmergencyStop`, `OverrideManager`, and `UndoStack` into the `SafeWorkflow`
- The `run()` method loops through steps, checking `should_agent_execute()` and `emergency.check()` at each step
- Record each completed step on the undo stack with a mock undo function
- Wrap the main loop in a `try/except AgentStoppedError` block

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
class SafeWorkflow:
    """Unified workflow with stops, overrides, and undo."""
    
    def __init__(self, steps: list[str]):
        self.steps = steps
        self.emergency = EmergencyStop()
        self.overrides = OverrideManager()
        self.undo_stack = UndoStack()
        self.director = DirectionManager()
        self.completed: list[str] = []
        self.results: dict[str, Any] = {}
    
    def override(
        self,
        step: str,
        user_input: Any = None,
        level: str = "replace"
    ):
        """Set an override for a specific step."""
        level_map = {
            "suggest": OverrideLevel.SUGGEST,
            "replace": OverrideLevel.REPLACE,
            "takeover": OverrideLevel.TAKEOVER,
            "pilot": OverrideLevel.PILOT
        }
        self.overrides.request_override(
            level=level_map.get(level, OverrideLevel.REPLACE),
            target_step=step,
            user_input=user_input
        )
    
    def emergency_stop(self, reason: str = "", graceful: bool = True):
        """Trigger an emergency stop."""
        self.emergency.stop(
            reason=StopReason.USER_REQUESTED,
            message=reason,
            graceful=graceful
        )
    
    def undo_last(self) -> dict:
        """Undo the last completed action."""
        result = self.undo_stack.undo_last()
        if result.get("success"):
            action = result["action"]
            if action in self.completed:
                self.completed.remove(action)
                self.results.pop(action, None)
        return result
    
    def run(self) -> dict:
        """Execute the workflow with all safety mechanisms."""
        for step in self.steps:
            # Check for emergency stop
            try:
                self.emergency.check(step)
            except AgentStoppedError as e:
                return {
                    "status": "stopped",
                    "at_step": step,
                    "reason": str(e),
                    "completed": self.completed.copy()
                }
            
            # Check for overrides
            if self.overrides.should_agent_execute(step):
                result = f"Agent executed: {step}"
            else:
                override = self.overrides.get_override_for_step(step)
                result = override.user_input if override else f"Skipped: {step}"
            
            # Record for undo
            step_name = step  # Capture for closure
            self.undo_stack.record(
                action_name=step_name,
                parameters={},
                result=result,
                undo_fn=lambda s=step_name: f"Undone: {s}"
            )
            
            self.completed.append(step)
            self.results[step] = result
            print(f"  ✅ {step}: {result}")
        
        return {
            "status": "completed",
            "completed": self.completed.copy(),
            "results": self.results.copy()
        }
    
    def status(self) -> dict:
        """Get current workflow status."""
        return {
            "total_steps": len(self.steps),
            "completed": len(self.completed),
            "stopped": self.emergency.is_stopped,
            "overrides_set": len(self.overrides.overrides),
            "undoable": self.undo_stack.get_undoable()
        }
```
</details>

### Bonus challenges

- [ ] Add a "pause and resume" capability — like emergency stop but with built-in resumption from the paused step
- [ ] Implement cascading undo — undoing step 2 automatically undoes steps 3, 4, 5 that depended on it
- [ ] Add cost tracking — trigger automatic stops when cumulative API costs exceed a budget

---

## Summary

✅ **Override levels** (suggest, replace, takeover, pilot) give users granular control without forcing all-or-nothing choices between human and agent

✅ **Emergency stops** must be thread-safe, checked frequently, and default to graceful behavior that saves state for resumption

✅ **Direction changes** redirect the agent mid-task without losing work — they're normal workflow events, not errors

✅ **Undo capabilities** depend on action reversibility — design for reversibility upfront by using drafts, soft deletes, and scheduled sends instead of immediate irreversible actions

**Next:** [Collaborative Execution](./05-collaborative-execution.md)

---

## Further reading

- [LangGraph — Human-in-the-Loop Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) — pause, resume, and redirect with `Command`
- [Google PAIR — Feedback + Control](https://pair.withgoogle.com/chapter/feedback-controls/) — user control in AI systems
- [Error Handling & Recovery](../08-error-handling-recovery/) — graceful failure and recovery patterns

*[Back to Human-in-the-Loop overview](./00-human-in-the-loop.md)*

<!-- 
Sources Consulted:
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- Google PAIR Feedback + Control: https://pair.withgoogle.com/chapter/feedback-controls/
- Python threading documentation: https://docs.python.org/3/library/threading.html
-->
